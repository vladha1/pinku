"""
STT — local Whisper via faster-whisper.

Records from microphone using PyAudio, applies WebRTC VAD to detect
speech boundaries, then transcribes the utterance locally.

On M4 Mac Mini: faster-whisper uses MPS (Metal) acceleration automatically
when device="auto" and torch-mps is available; falls back to CPU otherwise.
The "base.en" model is fast (~100ms on M4) and very accurate for English.
For multilingual use, switch WHISPER_MODEL to "small" in config.py.
"""

from __future__ import annotations
import io
import os
import time
import wave
import threading
import collections
import numpy as np
from config import (
    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_LANGUAGE,
    MIC_SAMPLE_RATE, MIC_CHUNK_MS, VAD_SILENCE_SEC, VAD_MIN_SPEECH_MS,
)

# ── lazy imports (heavy libs loaded once on first use) ────────────────────────
_whisper_model  = None
_whisper_lock   = threading.Lock()

# ── Mic level meter (updated in _callback, read by dashboard) ─────────────────
_mic_rms: float = 0.0   # smoothed RMS updated every 30 ms in the PyAudio callback

def get_mic_level() -> float:
    """
    Return normalised mic level for the dashboard meter (0.0 – 1.0).
    Scale factor of 20× — at 3m distance rms is ~0.01-0.03, giving 0.2-0.6
    which maps to clearly visible bars.  8× was too subtle (2-3 px change).
    """
    return min(_mic_rms * 20.0, 1.0)

# Optional external log callback — set by pinku.py to route STT diagnostics
# into the dashboard log.  Signature: (level: str, msg: str) -> None
_log_cb = None

def set_log_callback(fn):
    global _log_cb
    _log_cb = fn

def _stt_log(msg: str):
    # STT diagnostics go to stdout/log file only — not the dashboard.
    # The dashboard Log tab gets the important events (wake, user transcript,
    # Pinku reply) via _log() in pinku.py; flooding it with per-frame noise
    # floor readings makes it unreadable.
    print(msg)

def _load_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is not None:
            return _whisper_model
        from faster_whisper import WhisperModel
        print(f"[STT] Loading Whisper model={WHISPER_MODEL!r} device={WHISPER_DEVICE!r} …")
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type="int8",   # int8 is fast + accurate on M4 CPU; use float16 for MPS
        )
        print("[STT] Whisper ready")
    return _whisper_model


# ── VAD helpers ───────────────────────────────────────────────────────────────

def _frame_bytes(chunk_ms: int = MIC_CHUNK_MS) -> int:
    """Bytes per VAD frame (16-bit mono)."""
    return int(MIC_SAMPLE_RATE * chunk_ms / 1000) * 2


def _pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(MIC_SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


# ── Main recording + transcription ────────────────────────────────────────────

class AudioRecorder:
    """
    Continuous mic listener using WebRTC VAD.

    Usage:
        rec = AudioRecorder()
        rec.start()
        utterance_pcm = rec.wait_for_utterance()  # blocks until speech detected + silence
        text = transcribe(utterance_pcm)
        rec.stop()
    """

    def __init__(self):
        import pyaudio
        self._pa  = pyaudio.PyAudio()
        self._stream   = None
        self._running  = False
        self._buf_lock = threading.Lock()
        self._frames: collections.deque[bytes] = collections.deque(maxlen=500)
        self._new_audio = threading.Event()

    def start(self):
        import pyaudio
        from config import MIC_DEVICE_INDEX
        kwargs = dict(
            format=pyaudio.paInt16,
            channels=1,
            rate=MIC_SAMPLE_RATE,
            input=True,
            frames_per_buffer=_frame_bytes() // 2,
            stream_callback=self._callback,
        )
        if MIC_DEVICE_INDEX >= 0:
            kwargs["input_device_index"] = MIC_DEVICE_INDEX
        self._stream = self._pa.open(**kwargs)
        self._running = True
        self._stream.start_stream()
        dev = f" (device {MIC_DEVICE_INDEX})" if MIC_DEVICE_INDEX >= 0 else " (system default)"
        print(f"[STT] Mic open{dev}")

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        self._pa.terminate()
        print("[STT] Mic closed")

    def _callback(self, in_data, frame_count, time_info, status):
        import pyaudio
        global _mic_rms
        with self._buf_lock:
            self._frames.append(in_data)
        self._new_audio.set()
        # Track smoothed RMS for the dashboard mic meter (fast EWA, no blocking)
        try:
            audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(audio ** 2)))
            _mic_rms = _mic_rms * 0.6 + rms * 0.4   # 40% weight → reacts quickly
        except Exception:
            pass
        return (None, pyaudio.paContinue)

    def wait_for_utterance(self,
                           stop_event: threading.Event | None = None,
                           timeout: float = 60.0) -> bytes | None:
        """
        Block until a complete utterance is detected (speech + trailing silence).
        Returns raw PCM bytes, or None on timeout/stop.

        Uses adaptive RMS energy detection rather than WebRTC VAD.
        WebRTC VAD fails for distant/quiet speech because:
          - VAD aggressiveness > 0 rejects quiet distant speech entirely
          - VAD aggressiveness = 0 marks everything as speech so silence is
            never detected and the utterance never ends

        Adaptive noise floor: noise_floor tracks the room's ambient level (TV,
        AC, etc.).  Speech is detected when a frame exceeds noise_floor * SPEECH_MULT.
        The floor only adapts during silence so loud TV doesn't raise it mid-speech.

        Ring-buffer pre-roll (600 ms) captures the start of speech that arrives
        before energy crosses the threshold — avoids clipping the first syllable.
        """
        frame_ms   = MIC_CHUNK_MS
        frame_b    = _frame_bytes(frame_ms)
        silence_frames_needed = int(VAD_SILENCE_SEC * 1000 / frame_ms)
        min_speech_frames     = int(VAD_MIN_SPEECH_MS / frame_ms)
        preroll_frames        = 20   # 600 ms ring buffer

        # ── Detection tuning ─────────────────────────────────────────────────
        # DETECT_GAIN: boost quiet distant speech for the threshold comparison.
        # Only used for detection — raw frames stored in speech_buf, amplified
        # later by amplify_pcm() before Whisper/Gemini.
        #
        # Adaptive noise floor:
        #   • Starts high (0.10) so the first few frames never false-trigger.
        #   • CALIBRATION_FRAMES (15 × 30 ms = 450 ms) run at fast alpha (0.35)
        #     so the floor drops to the actual room ambient before detection starts.
        #   • After calibration, adapts slowly (1 % per frame) during silence only.
        #   • If speech stays "on" for > MAX_SPEECH_SEC it's background noise;
        #     floor is nudged up and the capture is reset.
        DETECT_GAIN        = 8.0
        CALIBRATION_FRAMES = 15      # 450 ms fast-settle at start of each call
        CAL_ALPHA          = 0.35    # fast floor adaptation during calibration
        noise_floor        = 0.10    # starts HIGH — drops to ambient during calibration
        noise_alpha        = 0.01    # slow adaptation after calibration
        speech_mult        = 2.5     # speech must be 2.5× floor
        MAX_SPEECH_SEC     = 3.5     # if "speech" runs this long, assume background noise
        # Hard minimum: regardless of how quiet the room is, never trigger on
        # anything below this det value.  In a silent room the floor drops to
        # ~0.017 and thr falls to 0.043 — fan vibration / distant noise at
        # det=0.08-0.11 then false-triggers.  Real speech at 3m = det 0.14+.
        MIN_SPEECH_DET     = 0.12

        max_speech_frames = int(MAX_SPEECH_SEC * 1000 / frame_ms)

        ring:       collections.deque[bytes] = collections.deque(maxlen=preroll_frames)
        speech_buf: list[bytes] = []
        in_speech       = False
        silence_streak  = 0
        speech_frames   = 0
        total_frames    = 0
        peak_rms_det    = 0.0   # for diagnostics
        deadline = time.time() + timeout

        # Drain stale frames so we start clean
        with self._buf_lock:
            self._frames.clear()
        self._new_audio.clear()

        while time.time() < deadline:
            if stop_event and stop_event.is_set():
                return None

            self._new_audio.wait(timeout=0.1)
            self._new_audio.clear()

            with self._buf_lock:
                new_frames = list(self._frames)
                self._frames.clear()

            for raw in new_frames:
                chunks = [raw[i:i+frame_b] for i in range(0, len(raw), frame_b)
                          if len(raw[i:i+frame_b]) == frame_b]
                for chunk in chunks:
                    total_frames += 1
                    audio    = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    rms_raw  = float(np.sqrt(np.mean(audio ** 2)))
                    rms_det  = min(rms_raw * DETECT_GAIN, 1.0)

                    if rms_det > peak_rms_det:
                        peak_rms_det = rms_det

                    # ── Calibration phase ────────────────────────────────────
                    # Spend the first 450 ms just settling the noise floor to the
                    # real room ambient.  No speech detection during this window.
                    # Skip frames where raw RMS > 0.10 (TTS echo / chime) so a
                    # still-ringing speak output doesn't corrupt the floor and
                    # push the threshold above 1.0 (which makes speech undetectable).
                    if total_frames <= CALIBRATION_FRAMES:
                        ring.append(chunk)
                        # Only use quiet frames for calibration.
                        # Beep/chime echo at the mic = rms_raw ≈ 0.03-0.08.
                        # Ambient noise (fan, hum, quiet room) = rms_raw ≈ 0.003-0.020.
                        # Threshold 0.030 lets ambient through and blocks chime echo,
                        # preventing the floor from spiking to 0.39+ after unmute.
                        if rms_raw < 0.030:
                            noise_floor = noise_floor * (1 - CAL_ALPHA) + rms_det * CAL_ALPHA
                        continue

                    is_speech = (rms_det > noise_floor * speech_mult
                                 and rms_det > MIN_SPEECH_DET)

                    # Noise floor adapts during silence only
                    if not in_speech and not is_speech:
                        noise_floor = noise_floor * (1 - noise_alpha) + rms_det * noise_alpha

                    if not in_speech:
                        ring.append(chunk)
                        if is_speech:
                            in_speech = True
                            silence_streak = 0
                            speech_frames  = 1
                            speech_buf     = list(ring)
                            _stt_log(f"[STT] speech start — det={rms_det:.4f} "
                                     f"floor={noise_floor:.4f} thr={noise_floor*speech_mult:.4f}")
                            # Extend the deadline so the utterance has time to complete.
                            # Speech often starts near the end of a 5s window, leaving
                            # no time for the trailing silence — the call timed out and
                            # returned None even though the person was mid-sentence.
                            # Give at least MAX_SPEECH_SEC + 1s from now regardless of
                            # where we are in the original timeout window.
                            speech_deadline = time.time() + MAX_SPEECH_SEC + 1.0
                            if speech_deadline > deadline:
                                deadline = speech_deadline
                    else:
                        speech_buf.append(chunk)
                        if is_speech:
                            silence_streak = 0
                            speech_frames += 1
                            # ── Background-noise guard ────────────────────────
                            # If "speech" has been running for > MAX_SPEECH_SEC,
                            # it's almost certainly TV/music, not a voice command.
                            # Nudge the floor up and reset so we can re-calibrate.
                            if speech_frames >= max_speech_frames:
                                noise_floor = noise_floor * 0.85 + rms_det * 0.15
                                _stt_log(f"[STT] reset: >{MAX_SPEECH_SEC}s sustained audio "
                                         f"→ background noise, floor→{noise_floor:.4f}")
                                in_speech = False
                                speech_buf = []
                                ring.clear()
                                speech_frames = 0
                                silence_streak = 0
                        else:
                            silence_streak += 1
                            if silence_streak >= silence_frames_needed:
                                if speech_frames >= min_speech_frames:
                                    return b"".join(speech_buf)
                                # Too short — noise click; reset
                                in_speech = False
                                speech_buf = []
                                ring.clear()

        # Timed out — log diagnostics
        if total_frames > 0:
            _stt_log(f"[STT] no-speech: peak={peak_rms_det:.4f} floor={noise_floor:.4f} "
                     f"thr={noise_floor*speech_mult:.4f} frames={total_frames}")
        return None


# ── Pinku-echo filter ─────────────────────────────────────────────────────────
# When Pinku speaks, the mic can pick up her own TTS output and transcribe it
# as a new user utterance.  We register the last 3 replies she spoke; if the
# transcribed text shares a 4-gram with any of them it's an echo → discard.

_pinku_recent: list[str] = []   # last 3 spoken replies (registered by pinku.py)
_pinku_lock   = threading.Lock()

def register_pinku_speech(text: str):
    """Call immediately before TTS so echo-filter knows what to reject."""
    with _pinku_lock:
        _pinku_recent.insert(0, text.lower())
        del _pinku_recent[3:]          # keep only last 3

def _is_pinku_echo(text: str) -> bool:
    """True if text is likely Pinku's own TTS being picked up by the mic."""
    t = text.lower()
    words = t.split()
    if len(words) < 3:
        return False
    with _pinku_lock:
        for reply in _pinku_recent:
            # Check if any 4-gram from the transcript appears in Pinku's recent reply
            for i in range(len(words) - 3):
                ng = " ".join(words[i:i + 4])
                if ng in reply:
                    return True
    return False


# ── Hallucination filter ──────────────────────────────────────────────────────
# Whisper hallucinates these on silence / background noise — discard them.
_HALLUCINATION_EXACT = {
    "thank you", "thank you very much", "thanks very much",
    "thanks for watching", "thanks for listening",
    "you", "bye", "bye bye", "goodbye", "please subscribe",
    "subtitles by", "foreign", "thanks", "okay", "ok",
    "[music]", "[applause]", "[laughter]", "[noise]", "[silence]",
    "♪", "...", ". . .", "www.", ".com",
    "have a great day", "have a good day", "you're welcome",
    "it was a pleasure", "feel free to ask",
    # Whisper echoing its own prompt
    "pinku is the name of a home ai assistant",
    "the ai assistant is the name of a home ai assistant",
    "wake word pinku",
    # Common cricket/TV hallucinations
    "virat kohli", "india indians", "india indians.", "india india",
    "rohit sharma", "ms dhoni", "sachin tendulkar",
    "kishu lai", "k r k r", "r k r k",
    # Whisper echoing the old initial_prompt
    "ipl cricket match score. virat kohli. mumbai indians. csk",
    "ipl cricket match score. virat kohli. mumbai indians",
    "virat kohli, ipl cricket match score. virat kohli. mumbai indians. csk",
    "pinku, ipl cricket match score",
}

_HALLUCINATION_STARTS = (
    "subtitles", "transcript", "captions", "this video",
    "thanks for", "thank you for", "please like",
    # Whisper looping on TV speech patterns
    "r. k.", "k. r.", "r k r", "k r k",
)

def _is_hallucination(text: str) -> bool:
    t = text.strip().lower().rstrip(".,!?")
    if t in _HALLUCINATION_EXACT:
        return True
    if any(t.startswith(p) for p in _HALLUCINATION_STARTS):
        return True
    if len(t.split()) <= 1:     # single word → almost always noise
        # Exception: let wake-word variants through so _check_wake can act on them.
        # Silence-triggered echoes are already caught above by no_speech_prob.
        _WAKE_VARIANTS = {"pinku", "pinky", "pinko", "pink", "pingu", "pinkoo", "penku", "penko",
                          "पिंकू", "पिंकु", "पिंको", "पिंकी",
                          "पिकु", "पिकू"}   # Whisper drops anusvara ं on पिंकु
        if t not in _WAKE_VARIANTS:
            return True
    # Pinku's own TTS output being picked up by mic
    if any(phrase in t for phrase in ("going quiet", "going quite", "pinku ready", "okay bye")):
        return True
    # Repetition detector — "Virat Kohli. Virat Kohli. Virat Kohli." is noise
    words = t.split()
    if len(words) >= 4:
        # Check if the transcript is just a short phrase repeated
        for phrase_len in (1, 2, 3):
            phrase = " ".join(words[:phrase_len])
            repeats = sum(1 for i in range(0, len(words), phrase_len)
                          if " ".join(words[i:i+phrase_len]) == phrase)
            if repeats >= 3 and repeats * phrase_len >= len(words) * 0.75:
                return True
    return False


# Only accept Hindi and English — everything else is a hallucination on this device
_ALLOWED_LANGUAGES = {"en", "hi"}


_MIN_RMS    = float(os.getenv("WHISPER_MIN_RMS",    "0.004"))  # below this = silence/noise, skip Whisper
_TARGET_RMS = float(os.getenv("WHISPER_TARGET_RMS", "0.060"))  # amplify quiet speech up to this level
_MAX_GAIN   = float(os.getenv("WHISPER_MAX_GAIN",   "12.0"))   # cap amplification


def amplify_pcm(pcm: bytes) -> tuple[bytes, float]:
    """
    Amplify quiet PCM audio to _TARGET_RMS so distant speech is clear to Whisper/Gemini.
    Returns (amplified_pcm, original_rms).
    Audio below _MIN_RMS is returned unchanged — caller should discard it.
    """
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms >= _MIN_RMS and rms < _TARGET_RMS:
        gain = min(_TARGET_RMS / rms, _MAX_GAIN)
        audio = np.clip(audio * gain, -1.0, 1.0)
        pcm = (audio * 32767).astype(np.int16).tobytes()
    return pcm, rms


def transcribe(pcm: bytes) -> str:
    """
    Transcribe raw 16-bit mono PCM bytes using local Whisper.
    Returns the cleaned transcript string, or "" if likely hallucination/noise.
    Silently drops any language other than English or Hindi.
    """
    # Amplify quiet distant speech, then gate on minimum energy.
    pcm, rms = amplify_pcm(pcm)
    print(f"[STT] RMS={rms:.4f} dur={len(pcm)/(16000*2):.1f}s")
    if rms < _MIN_RMS:
        print(f"[STT] Dropped — RMS {rms:.4f} below threshold {_MIN_RMS}")
        return ""
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    model = _load_whisper()

    segments, info = model.transcribe(
        audio,
        language=WHISPER_LANGUAGE,   # None = auto-detect; set "en" or "hi" to force
        beam_size=3,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        no_speech_threshold=0.40,    # was 0.55 — more accepting of quiet/distant speech
        log_prob_threshold=-1.0,     # was -0.8 — accept lower-confidence transcriptions
        compression_ratio_threshold=2.4,
        # Minimal prompt — just the wake word so Whisper spells it right.
        # Longer prompts (cricket names etc.) get echoed back as hallucinations
        # when TV audio is in the room.
        initial_prompt="Pinku.",
    )

    segs = list(segments)
    if not segs:
        return ""

    # Drop non-English / non-Hindi (Portuguese, Spanish etc. = hallucination here)
    if info.language not in _ALLOWED_LANGUAGES:
        print(f"[STT] Dropped — language={info.language!r} (only en/hi accepted)")
        return ""

    # Drop if any segment has high no-speech probability.
    # Raised from 0.55 → 0.70: distant Hindi speech gets scored 0.55–0.65 by
    # Whisper and was being silently dropped despite real speech being present.
    if any(getattr(s, "no_speech_prob", 0) > 0.70 for s in segs):
        print(f"[STT] Dropped — no_speech_prob too high")
        return ""

    text = " ".join(s.text.strip() for s in segs).strip()

    if _is_hallucination(text):
        print(f"[STT] Dropped hallucination: {text!r}")
        return ""

    if _is_pinku_echo(text):
        print(f"[STT] Dropped echo of Pinku's own TTS: {text!r}")
        return ""

    print(f"[STT] lang={info.language} → {text!r}")
    return text


def preload():
    """Call at startup to load Whisper model before first request."""
    threading.Thread(target=_load_whisper, daemon=True).start()

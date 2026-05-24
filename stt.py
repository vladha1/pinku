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
        import webrtcvad
        import pyaudio
        self._vad = webrtcvad.Vad(2)   # aggressiveness 0-3; 2 = balanced
        self._pa  = pyaudio.PyAudio()
        self._stream   = None
        self._running  = False
        self._buf_lock = threading.Lock()
        self._frames: collections.deque[bytes] = collections.deque(maxlen=500)
        self._new_audio = threading.Event()

    def start(self):
        import pyaudio
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=MIC_SAMPLE_RATE,
            input=True,
            frames_per_buffer=_frame_bytes() // 2,
            stream_callback=self._callback,
        )
        self._running = True
        self._stream.start_stream()
        print("[STT] Mic open")

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        self._pa.terminate()
        print("[STT] Mic closed")

    def _callback(self, in_data, frame_count, time_info, status):
        import pyaudio
        with self._buf_lock:
            self._frames.append(in_data)
        self._new_audio.set()
        return (None, pyaudio.paContinue)

    def wait_for_utterance(self,
                           stop_event: threading.Event | None = None,
                           timeout: float = 60.0) -> bytes | None:
        """
        Block until a complete utterance is detected (speech + trailing silence).
        Returns raw PCM bytes, or None on timeout/stop.

        Ring-buffer pre-roll (300 ms) captures the start of speech that arrives
        before VAD fires — avoids clipping the first syllable.
        """
        frame_ms   = MIC_CHUNK_MS
        frame_b    = _frame_bytes(frame_ms)
        silence_frames_needed = int(VAD_SILENCE_SEC * 1000 / frame_ms)
        min_speech_frames     = int(VAD_MIN_SPEECH_MS / frame_ms)
        preroll_frames        = 10   # 300 ms ring buffer

        ring:       collections.deque[bytes] = collections.deque(maxlen=preroll_frames)
        speech_buf: list[bytes] = []
        in_speech       = False
        silence_streak  = 0
        speech_frames   = 0
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
                # Ensure exactly frame_b bytes for VAD
                chunks = [raw[i:i+frame_b] for i in range(0, len(raw), frame_b)
                          if len(raw[i:i+frame_b]) == frame_b]
                for chunk in chunks:
                    try:
                        is_speech = self._vad.is_speech(chunk, MIC_SAMPLE_RATE)
                    except Exception:
                        is_speech = False

                    if not in_speech:
                        ring.append(chunk)
                        if is_speech:
                            in_speech = True
                            silence_streak = 0
                            speech_frames  = 1
                            speech_buf     = list(ring)   # include preroll
                    else:
                        speech_buf.append(chunk)
                        if is_speech:
                            silence_streak = 0
                            speech_frames += 1
                        else:
                            silence_streak += 1
                            if silence_streak >= silence_frames_needed:
                                if speech_frames >= min_speech_frames:
                                    return b"".join(speech_buf)
                                # Too short — probably a noise click; reset
                                in_speech = False
                                speech_buf = []
                                ring.clear()

        return None   # timed out


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
    "wake word pinku", "pinku",
}

_HALLUCINATION_STARTS = (
    "subtitles", "transcript", "captions", "this video",
    "thanks for", "thank you for", "please like",
)

def _is_hallucination(text: str) -> bool:
    t = text.strip().lower().rstrip(".,!?")
    if t in _HALLUCINATION_EXACT:
        return True
    if any(t.startswith(p) for p in _HALLUCINATION_STARTS):
        return True
    if len(t.split()) <= 1:     # single word → almost always noise
        return True
    # Pinku's own TTS output being picked up by mic
    if any(phrase in t for phrase in ("going quiet", "going quite", "pinku ready", "okay bye")):
        return True
    return False


# Only accept Hindi and English — everything else is a hallucination on this device
_ALLOWED_LANGUAGES = {"en", "hi"}


def transcribe(pcm: bytes) -> str:
    """
    Transcribe raw 16-bit mono PCM bytes using local Whisper.
    Returns the cleaned transcript string, or "" if likely hallucination/noise.
    Silently drops any language other than English or Hindi.
    """
    model = _load_whisper()
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    segments, info = model.transcribe(
        audio,
        language=WHISPER_LANGUAGE,   # None = auto-detect; set "en" or "hi" to force
        beam_size=3,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        no_speech_threshold=0.55,
        log_prob_threshold=-0.8,
        compression_ratio_threshold=2.4,
        # Short prompt to bias wake-word recognition — do NOT include full sentences
        # (Whisper sometimes echoes long prompts as hallucinations)
        initial_prompt="Pinku.",
    )

    segs = list(segments)
    if not segs:
        return ""

    # Drop non-English / non-Hindi (Portuguese, Spanish etc. = hallucination here)
    if info.language not in _ALLOWED_LANGUAGES:
        print(f"[STT] Dropped — language={info.language!r} (only en/hi accepted)")
        return ""

    # Drop if any segment has high no-speech probability
    if any(getattr(s, "no_speech_prob", 0) > 0.55 for s in segs):
        print(f"[STT] Dropped — no_speech_prob too high")
        return ""

    text = " ".join(s.text.strip() for s in segs).strip()

    if _is_hallucination(text):
        print(f"[STT] Dropped hallucination: {text!r}")
        return ""

    print(f"[STT] lang={info.language} → {text!r}")
    return text


def preload():
    """Call at startup to load Whisper model before first request."""
    threading.Thread(target=_load_whisper, daemon=True).start()

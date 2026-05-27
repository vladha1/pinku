"""
TTS — edge-tts (neural, natural-sounding) with macOS `say` fallback.

Primary:  edge-tts  → Microsoft neural voices streamed to a temp file → afplay
Fallback: macOS `say` (built-in, always available)

Chimes match Pinky exactly:
  play_beep()   — soft two-note C5→G5        (wake / confirm / entry)
  play_think()  — short 1800 Hz tick         (sending to LLM, processing)
  play_error()  — descending G4→D4           (API / LLM failure)
  play_mute()   — descending 520→330 Hz      (going quiet / mute)
  play_unmute() — ascending 330→520→780 Hz   (resuming / unmute)

Install edge-tts: pip install edge-tts
List voices:      python3 -m edge_tts --list-voices
"""

from __future__ import annotations
import io
import math
import os
import re
import struct
import subprocess
import tempfile
import threading
import time
import wave
import asyncio
from config import SAY_VOICE_EN, SAY_VOICE_HI, SAY_RATE

# ── Edge-TTS voices ───────────────────────────────────────────────────────────
EDGE_VOICE_EN = os.getenv("EDGE_VOICE_EN", "hi-IN-SwaraNeural")
EDGE_VOICE_HI = os.getenv("EDGE_VOICE_HI", "hi-IN-SwaraNeural")

_speak_lock    = threading.Lock()
_current_proc: subprocess.Popen | None = None
_speaking      = False

try:
    import edge_tts as _edge_tts
    _EDGE_AVAILABLE = True
except ImportError:
    _EDGE_AVAILABLE = False
    print("[TTS] edge-tts not installed — using macOS say (pip install edge-tts)")


# ── WAV chime generator (matches Pinky exactly) ───────────────────────────────

_SR = 22050   # sample rate for generated chimes

def _write_wav(path: str, notes: list[tuple[float, float]],
               volume: float = 0.40, decay: float = 0.0):
    """
    Write a WAV file at `path` from a list of (frequency_hz, duration_sec) pairs.
    frequency=0 → silence gap.
    decay > 0   → exponential amplitude decay on the last note (bell-like ring).
    """
    frames: list[bytes] = []
    for i, (freq, dur) in enumerate(notes):
        n = int(_SR * dur)
        is_last = (i == len(notes) - 1)
        for k in range(n):
            if freq == 0:
                sample = 0.0
            else:
                t = k / _SR
                amp = volume
                if decay > 0 and is_last:
                    amp *= math.exp(-decay * t)
                sample = amp * math.sin(2 * math.pi * freq * t)
            frames.append(struct.pack("<h", int(sample * 32767)))
    pcm = b"".join(frames)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SR)
        wf.writeframes(pcm)
    with open(path, "wb") as f:
        f.write(buf.getvalue())


# Generate all chime WAVs once at startup into /tmp
_CHIME_PATH  = "/tmp/pinku_chime.wav"
_THINK_PATH  = "/tmp/pinku_think.wav"
_ERROR_PATH  = "/tmp/pinku_error.wav"
_MUTE_PATH   = "/tmp/pinku_mute.wav"
_UNMUTE_PATH = "/tmp/pinku_unmute.wav"

def _generate_chimes():
    # Wake / confirm — soft two-note C5→G5 (perfect fifth), gentle bell decay
    _write_wav(_CHIME_PATH,
               [(523, 0.10), (0, 0.018), (784, 0.36)],
               volume=0.20, decay=4.0)
    # Thinking — crisp short tick at 1800 Hz
    _write_wav(_THINK_PATH, [(1800, 0.04)], volume=0.35)
    # Error — descending G4→D4
    _write_wav(_ERROR_PATH, [(392, 0.12), (0, 0.03), (294, 0.18)], volume=0.30)
    # Mute — soft descending two-note (going quiet)
    _write_wav(_MUTE_PATH,  [(520, 0.10), (0, 0.03), (330, 0.20)], volume=0.28)
    # Unmute — bright ascending three-note (back online)
    _write_wav(_UNMUTE_PATH,[(330, 0.07), (0, 0.02), (520, 0.07), (0, 0.02), (780, 0.20)],
               volume=0.42)

try:
    _generate_chimes()
    print("[TTS] Chimes generated")
except Exception as e:
    print(f"[TTS] Chime generation failed: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_hindi(text: str) -> bool:
    return bool(re.search(r'[ऀ-ॿ]', text))


def _play(path: str):
    """Non-blocking afplay of a WAV file."""
    subprocess.Popen(["afplay", path],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── TTS ───────────────────────────────────────────────────────────────────────

def known_duration(text: str) -> float:
    """
    Return the expected TTS playback duration before speaking.
    - say:      word count ÷ SAY_RATE  (accurate)
    - edge-tts: word count ÷ 130 wpm  (neural voices are slower; conservative
                under-estimate is fine because _speak_reply re-checks after speak())
    """
    words = max(1, len(text.split()))
    if _EDGE_AVAILABLE:
        # edge-tts Hindi neural voice speaks ~100 wpm (numbers, Devanagari expand);
        # use 95 wpm to ensure the mute timer outlasts actual playback.
        return (words / 95.0) * 60.0
    return _estimate_say_duration(text)


def stop_speaking():
    """Interrupt current TTS immediately."""
    global _current_proc, _speaking
    if _current_proc and _current_proc.poll() is None:
        _current_proc.terminate()
    _speaking = False


def is_speaking() -> bool:
    return _speaking


def _speak_edge(text: str, voice: str) -> float:
    """
    Generate MP3 via edge-tts, read exact duration with afinfo,
    then play. Returns known duration so the caller can set the
    unmute timer before/during playback rather than after.
    """
    global _current_proc, _speaking
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    duration = 0.0
    try:
        async def _gen():
            comm = _edge_tts.Communicate(text, voice)
            await comm.save(tmp.name)
        asyncio.run(_gen())
        # Exact duration from file — known BEFORE playback starts
        duration = _get_file_duration(tmp.name)
        if duration == 0.0:
            # afinfo unavailable — fall back to word-count estimate
            duration = _estimate_say_duration(text)
        _current_proc = subprocess.Popen(
            ["afplay", tmp.name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _current_proc.wait()
    except Exception as e:
        print(f"[TTS] edge-tts error: {e} — falling back to say")
        return _speak_say(text, SAY_VOICE_HI if _is_hindi(text) else SAY_VOICE_EN)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
    return duration


def _estimate_say_duration(text: str) -> float:
    """Estimate `say` playback duration from word count + SAY_RATE."""
    words = max(1, len(text.split()))
    return (words / SAY_RATE) * 60.0


def _get_file_duration(path: str) -> float:
    """Read exact playback duration from an audio file via afinfo."""
    try:
        out = subprocess.check_output(
            ["afinfo", path], stderr=subprocess.DEVNULL, timeout=5,
        ).decode()
        for line in out.splitlines():
            if "estimated duration" in line:
                return float(line.split(":")[1].strip().split()[0])
    except Exception:
        pass
    return 0.0


def _speak_say(text: str, voice: str) -> float:
    """Returns known playback duration in seconds (computed before speaking)."""
    global _current_proc
    duration = _estimate_say_duration(text)
    try:
        _current_proc = subprocess.Popen(
            ["say", "-v", voice, "-r", str(SAY_RATE), text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _current_proc.wait()
    except Exception as e:
        print(f"[TTS] say error: {e}")
    return duration


def speak(text: str, prefer_hi: bool = False, block: bool = False) -> float:
    """
    Speak text via edge-tts (neural) or macOS say (fallback).
    prefer_hi=True → Hindi voice.  block=True → wait until done.
    Returns measured playback duration in seconds (only meaningful when block=True).
    """
    if not text or not text.strip():
        return 0.0

    duration: list[float] = [0.0]

    def _run():
        global _speaking
        stop_speaking()
        is_hi = prefer_hi or _is_hindi(text)
        with _speak_lock:
            _speaking = True
            try:
                if _EDGE_AVAILABLE:
                    voice = EDGE_VOICE_HI if is_hi else EDGE_VOICE_EN
                    duration[0] = _speak_edge(text, voice)
                else:
                    voice = SAY_VOICE_HI if is_hi else SAY_VOICE_EN
                    duration[0] = _speak_say(text, voice)
            finally:
                _speaking = False

    if block:
        _run()
        return duration[0]
    else:
        threading.Thread(target=_run, daemon=True).start()
        return 0.0


# ── Chime functions ───────────────────────────────────────────────────────────

# Entry chime cooldown — don't re-chime if person was just detected
_last_entry_chime_at: float = 0.0
_ENTRY_CHIME_COOLDOWN = 25.0   # seconds


def play_beep(entry: bool = False):
    """Rising 3-note C5→E5→G5 — wake confirmed / action acknowledged / person entry.
    Pass entry=True for camera person-detection chimes to apply cooldown."""
    global _last_entry_chime_at
    if entry:
        now = time.time()
        if now - _last_entry_chime_at < _ENTRY_CHIME_COOLDOWN:
            return
        _last_entry_chime_at = now
    _play(_CHIME_PATH)


def play_think():
    """Short 1800 Hz tick — sending to LLM, please wait."""
    _play(_THINK_PATH)


def play_error():
    """Descending G4→D4 — API or LLM call failed."""
    _play(_ERROR_PATH)


def play_mute():
    """Soft descending 520→330 Hz — mic going quiet / muted."""
    _play(_MUTE_PATH)


def play_unmute():
    """Bright ascending 330→520→780 Hz — mic resuming / back online."""
    _play(_UNMUTE_PATH)


# play_sleep is an alias for play_mute (same semantic: going quiet)
play_sleep = play_mute

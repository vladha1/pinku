"""
TTS — edge-tts (neural, natural-sounding) with macOS `say` fallback.

Primary:  edge-tts  → Microsoft neural voices streamed to a temp file → afplay
Fallback: macOS `say` (built-in, always available)

Chimes match Pinky exactly:
  play_beep()   — rising 3-note C5→E5→G5    (wake / confirm / entry)
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
    # Wake / confirm — soft rising C5→E5→G5, bell decay on final note
    _write_wav(_CHIME_PATH,
               [(523, 0.09), (0, 0.012), (659, 0.09), (0, 0.012), (784, 0.34)],
               volume=0.28, decay=5.0)
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

def stop_speaking():
    """Interrupt current TTS immediately."""
    global _current_proc, _speaking
    if _current_proc and _current_proc.poll() is None:
        _current_proc.terminate()
    _speaking = False


def is_speaking() -> bool:
    return _speaking


def _speak_edge(text: str, voice: str):
    global _current_proc, _speaking
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    try:
        async def _gen():
            comm = _edge_tts.Communicate(text, voice)
            await comm.save(tmp.name)
        asyncio.run(_gen())
        _current_proc = subprocess.Popen(
            ["afplay", tmp.name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _current_proc.wait()
    except Exception as e:
        print(f"[TTS] edge-tts error: {e} — falling back to say")
        _speak_say(text, SAY_VOICE_HI if _is_hindi(text) else SAY_VOICE_EN)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def _speak_say(text: str, voice: str):
    global _current_proc
    try:
        _current_proc = subprocess.Popen(
            ["say", "-v", voice, "-r", str(SAY_RATE), text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _current_proc.wait()
    except Exception as e:
        print(f"[TTS] say error: {e}")


def speak(text: str, prefer_hi: bool = False, block: bool = False):
    """
    Speak text via edge-tts (neural) or macOS say (fallback).
    prefer_hi=True → Hindi voice.  block=True → wait until done.
    """
    if not text or not text.strip():
        return

    def _run():
        global _speaking
        stop_speaking()
        is_hi = prefer_hi or _is_hindi(text)
        with _speak_lock:
            _speaking = True
            try:
                if _EDGE_AVAILABLE:
                    voice = EDGE_VOICE_HI if is_hi else EDGE_VOICE_EN
                    _speak_edge(text, voice)
                else:
                    voice = SAY_VOICE_HI if is_hi else SAY_VOICE_EN
                    _speak_say(text, voice)
            finally:
                _speaking = False

    if block:
        _run()
    else:
        threading.Thread(target=_run, daemon=True).start()


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

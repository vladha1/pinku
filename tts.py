"""
TTS — edge-tts (neural, natural-sounding) with macOS `say` fallback.

Primary:  edge-tts  → Microsoft neural voices streamed to a temp file → afplay
Fallback: macOS `say` (robotic but always available)

Install: pip install edge-tts
List voices: python3 -m edge_tts --list-voices
"""

import subprocess
import threading
import tempfile
import os
import re
import asyncio
from config import SAY_VOICE_EN, SAY_VOICE_HI, SAY_RATE

# Edge-TTS voice names  (override via env EDGE_VOICE_EN / EDGE_VOICE_HI)
EDGE_VOICE_EN = os.getenv("EDGE_VOICE_EN", "en-US-AriaNeural")   # warm, natural female
EDGE_VOICE_HI = os.getenv("EDGE_VOICE_HI", "hi-IN-SwaraNeural")  # Hindi neural female

_speak_lock    = threading.Lock()
_current_proc: subprocess.Popen | None = None
_speaking      = False

# Detect edge-tts availability once at import time
try:
    import edge_tts as _edge_tts
    _EDGE_AVAILABLE = True
except ImportError:
    _EDGE_AVAILABLE = False
    print("[TTS] edge-tts not installed — using macOS say (install: pip install edge-tts)")


def _is_hindi(text: str) -> bool:
    """Heuristic: contains Devanagari characters."""
    return bool(re.search(r'[ऀ-ॿ]', text))


def stop_speaking():
    """Interrupt current TTS immediately."""
    global _current_proc, _speaking
    if _current_proc and _current_proc.poll() is None:
        _current_proc.terminate()
    _speaking = False


def is_speaking() -> bool:
    return _speaking


def _speak_edge(text: str, voice: str):
    """Generate audio with edge-tts and play with afplay."""
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
    """macOS say fallback."""
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

    prefer_hi=True  → use Hindi voice
    block=True      → wait until done before returning
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


def play_beep():
    """Short confirmation tone."""
    subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def play_mute():
    subprocess.Popen(["afplay", "/System/Library/Sounds/Funk.aiff"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def play_unmute():
    subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

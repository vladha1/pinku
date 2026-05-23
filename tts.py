"""
TTS — macOS `say` command wrapper.

Primary: subprocess `say` (zero-dependency, built-in on macOS).
The `say` command uses the system's Neural TTS voices which are high-quality
on M4 Mac Mini (Nicky, Ava, Samantha, Lekha for Indian-English/Hindi).

List available voices: `say -v ?`
"""

import subprocess
import threading
import re
from config import SAY_VOICE_EN, SAY_VOICE_HI, SAY_RATE

_speak_lock    = threading.Lock()
_current_proc: subprocess.Popen | None = None
_speaking      = False


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


def speak(text: str, prefer_hi: bool = False, block: bool = False):
    """
    Speak text via macOS `say`.

    prefer_hi=True  → use Hindi/Indian-English voice (Lekha)
    block=True      → wait until done before returning
    """
    if not text or not text.strip():
        return

    def _run():
        global _current_proc, _speaking
        stop_speaking()           # interrupt anything in progress
        voice = SAY_VOICE_HI if (prefer_hi or _is_hindi(text)) else SAY_VOICE_EN
        with _speak_lock:
            _speaking = True
            try:
                _current_proc = subprocess.Popen(
                    ["say", "-v", voice, "-r", str(SAY_RATE), text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                _current_proc.wait()
            except FileNotFoundError:
                print("[TTS] ERROR: `say` not found — are you on macOS?")
            except Exception as e:
                print(f"[TTS] error: {e}")
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

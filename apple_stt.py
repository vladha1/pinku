"""
Apple on-device speech recognition via SFSpeechRecognizer.
~0.1-0.2s per utterance, fully on-device on M4, no network required.

Architecture: the CLI python3 binary that runs pinku.py lacks the
NSSpeechRecognitionUsageDescription entitlement needed to trigger the
macOS permission dialog.  The Python.app binary (same Homebrew install,
different executable) has it.  So we spawn stt_worker.py under the
Python.app binary as a persistent subprocess.

Protocol (stdin→worker, stdout←worker):
  in:  4-byte LE uint32 (PCM length) + raw 16-bit mono PCM bytes
  out: transcript line + "\\n"  (empty line = no speech)
  First line:  "READY\\n" if authorized, "ERROR: ...\\n" if not
"""
from __future__ import annotations
import os
import struct
import subprocess
import sys
import threading

_worker_proc: subprocess.Popen | None = None
_worker_lock  = threading.Lock()
_worker_ready = False


def _find_python_app() -> str | None:
    """Locate Python.app/Contents/MacOS/Python relative to sys.executable."""
    real = os.path.realpath(sys.executable)
    # On Homebrew, real resolves to something like:
    #   .../Python.framework/Versions/3.14/bin/python3.14
    # Going up 2 levels lands in the framework version dir (.../Versions/3.14)
    # and Python.app lives right there under Resources/.
    version_dir = os.path.dirname(os.path.dirname(real))
    candidate = os.path.join(version_dir, "Resources", "Python.app",
                             "Contents", "MacOS", "Python")
    if os.path.isfile(candidate):
        return candidate

    # Fallback: real may be in a plain Cellar bin/ — search under Frameworks/
    fw_root = os.path.join(version_dir, "Frameworks", "Python.framework", "Versions")
    if os.path.isdir(fw_root):
        for ver in sorted(os.listdir(fw_root), reverse=True):
            if ver == "Current":
                continue
            c = os.path.join(fw_root, ver, "Resources", "Python.app",
                             "Contents", "MacOS", "Python")
            if os.path.isfile(c):
                return c
    return None


def _venv_site() -> str:
    """Return the active venv's site-packages directory."""
    return os.path.join(
        sys.prefix, "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
    )


def _start_worker() -> bool:
    """Spawn stt_worker.py under Python.app binary. Returns True on success."""
    global _worker_proc, _worker_ready

    app_python = _find_python_app()
    if not app_python:
        print("[AppleSTT] Python.app binary not found — Apple STT unavailable")
        return False

    worker_script = os.path.join(os.path.dirname(__file__), "stt_worker.py")
    env = {**os.environ, "STT_VENV_SITE": _venv_site()}

    try:
        _worker_proc = subprocess.Popen(
            [app_python, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        # Block until worker signals ready (or error)
        line = _worker_proc.stdout.readline().decode().strip()
        if line == "READY":
            _worker_ready = True
            print(f"[AppleSTT] Worker ready — {app_python}")
            return True
        print(f"[AppleSTT] Worker error: {line}")
        _worker_proc = None
        _worker_ready = False
        return False
    except Exception as e:
        print(f"[AppleSTT] Could not start worker: {e}")
        _worker_proc = None
        _worker_ready = False
        return False


def request_authorization() -> bool:
    """
    Start the stt_worker subprocess.  The worker runs under Python.app which
    has the NSSpeechRecognitionUsageDescription entitlement, so macOS will
    show the permission dialog when the worker first starts.
    Returns True if the worker is ready (i.e., authorization succeeded).
    """
    with _worker_lock:
        if _worker_ready and _worker_proc and _worker_proc.poll() is None:
            return True
        return _start_worker()


def is_available() -> bool:
    """True if the worker subprocess is alive and ready."""
    return (
        _worker_ready
        and _worker_proc is not None
        and _worker_proc.poll() is None
    )


def transcribe(pcm: bytes, sample_rate: int = 16000) -> str:
    """
    Transcribe raw 16-bit mono PCM. Returns transcript string, "" on failure.
    Caller should fall back to Gemini audio if this returns "".
    sample_rate is ignored (worker always uses the rate embedded in the WAV).
    """
    global _worker_proc, _worker_ready
    with _worker_lock:
        if not _worker_ready or _worker_proc is None or _worker_proc.poll() is not None:
            return ""
        try:
            hdr = struct.pack("<I", len(pcm))
            _worker_proc.stdin.write(hdr + pcm)
            _worker_proc.stdin.flush()
            line = _worker_proc.stdout.readline()
            return line.decode().strip()
        except Exception as e:
            print(f"[AppleSTT] Worker comm error: {e} — restarting")
            _worker_proc  = None
            _worker_ready = False
            return ""

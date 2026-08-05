"""
Fast on-device Whisper STT for Apple Silicon via Apple MLX.
Runs whisper-large-v3 on the M4 Neural Engine — no permissions needed.

Install: pip install mlx-whisper
First run downloads the model (~1.5 GB) from HuggingFace automatically.

All MLX calls are serialised onto a single worker thread because Metal GPU
streams are per-thread; calling from multiple threads causes a runtime crash.
"""
from __future__ import annotations
import os
import queue
import threading
import time

import numpy as np

MODEL    = "mlx-community/whisper-large-v3-turbo"
LANGUAGE = os.environ.get("MLX_STT_LANGUAGE", "auto")  # auto-detect; retries as Hindi if Spanish

# Biases the decoder toward the wake word so "Pinky" isn't transcribed as
# Vicky / Wilkie / Nicky etc.  Passed as initial_prompt on wake-detection calls.
WAKE_PROMPT = os.environ.get("MLX_WAKE_PROMPT", "Talking to the assistant Pinky.")


def _is_hallucination(text: str) -> bool:
    """Detect Whisper looping hallucinations (e.g. 'of the episode' repeated 50×)."""
    words = text.split()
    if len(words) > 80:
        return True
    if len(words) >= 8:
        grams = [" ".join(words[i:i+4]) for i in range(len(words) - 3)]
        if max(grams.count(g) for g in set(grams)) > 2:
            return True
    return False

_AVAILABLE = False
_ready     = threading.Event()   # set once the worker has warmed up

# ── Single worker thread — all mlx_whisper calls must run here ───────────────
_q: "queue.Queue[tuple | None]" = queue.Queue()


def _try_import() -> bool:
    global _AVAILABLE
    try:
        import mlx_whisper   # noqa
        _AVAILABLE = True
    except ImportError:
        print("[STT] mlx-whisper not installed — "
              "run: pip install mlx-whisper")
    return _AVAILABLE


def _worker() -> None:
    """Persistent worker — owns the Metal GPU stream for all transcriptions."""
    global _AVAILABLE
    import mlx_whisper

    # Warm up: compile the model on this thread's Metal stream
    print(f"[STT] Loading {MODEL} …")
    t0 = time.time()
    silence = np.zeros(1600, dtype=np.float32)
    try:
        _transcribe_once(mlx_whisper, silence, "en")   # warm-up always in English
    except Exception as e:
        print(f"[STT] MLX warm-up failed: {e}")
        print("[STT] MLX disabled — Apple STT + Gemini audio remain active")
        _AVAILABLE = False
        _ready.set()   # unblock any is_ready() waiters
        return
    _ready.set()
    print(f"[STT] Ready ({time.time() - t0:.1f}s)")

    while True:
        task = _q.get()
        if task is None:
            break
        audio, lang, temperature, initial_prompt, result_box, done = task
        try:
            result_box.append(_transcribe_once(mlx_whisper, audio, lang,
                                               temperature, initial_prompt))
        except Exception as e:
            print(f"[STT] worker error: {e}")
            result_box.append({"text": "", "language": ""})
        finally:
            done.set()


def _transcribe_once(mlx_whisper, audio: np.ndarray, lang: str | None,
                     temperature: float = 0.0, initial_prompt: str = "") -> dict:
    """Single transcription call — must run on the worker thread.

    initial_prompt biases the decoder toward those words — used to prime the
    wake word ("Pinky") so it isn't mis-transcribed as Vicky/Wilkie/etc.
    """
    return mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MODEL,
        language=lang,
        temperature=temperature,
        initial_prompt=initial_prompt or None,
        verbose=False,
    )


_try_import()
_worker_thread: threading.Thread | None = None


def preload() -> None:
    """Start the persistent worker thread, which downloads+warms the model."""
    global _worker_thread
    if not _AVAILABLE:
        return
    _worker_thread = threading.Thread(
        target=_worker, daemon=True, name="mlx-whisper-worker")
    _worker_thread.start()


def is_ready() -> bool:
    """True once the model has been loaded and warmed up."""
    return _AVAILABLE and _ready.is_set()


def transcribe(pcm: bytes, sample_rate: int = 16000,
               initial_prompt: str = "") -> str:
    """
    Transcribe with auto language detection. Used in active sessions where
    the wake word is already confirmed — accuracy matters more than speed.
    Spanish/Urdu mis-detections are retried as Hindi.

    Pass initial_prompt=mlx_stt.WAKE_PROMPT on idle wake-word detection to
    bias the decoder toward "Pinky" (avoids Vicky/Wilkie mis-transcriptions).
    """
    if not _AVAILABLE or not _ready.is_set():
        return ""
    try:
        audio  = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        result = _send(audio, None, initial_prompt=initial_prompt)   # auto-detect
        detected = result.get("language", "")
        # Languages Whisper commonly confuses with Hindi (similar prosody / phonetics):
        # pl=Polish, ur=Urdu, es=Spanish, pt=Portuguese, tr=Turkish
        _HINDI_CONFUSED = {"es", "ur", "pl", "pt", "tr"}
        if detected in _HINDI_CONFUSED:
            result = _send(audio, "hi", temperature=0.3, initial_prompt=initial_prompt)
            print(f"[STT] re-ran as Hindi temp=0.3 (was {detected})")
            detected = result.get("language", "")
        # Hard drop: if still not English or Hindi, discard rather than hallucinate
        if detected and detected not in ("en", "hi", ""):
            print(f"[STT] dropped — language={detected!r} (only en/hi accepted)")
            return ""
        text = result.get("text", "").strip()
        if _is_hallucination(text):
            print(f"[STT] hallucination detected ({len(text.split())} words) — dropping")
            return ""
        return text
    except Exception as e:
        print(f"[STT] error: {e}")
        return ""


def transcribe_with_fallback(
    pcm: bytes,
    wake_check_fn,
    sample_rate: int = 16000,
) -> tuple[str, bool, str]:
    """
    Two-pass STT for the idle wake-word path:
      1. English pass (~0.85s) — fast, reliable Roman wake words
      2. Hindi pass (~0.85s more) — only if wake word not found in English

    Returns (transcript, triggered, command).
    wake_check_fn: callable(text) -> (triggered: bool, command: str)
    """
    if not _AVAILABLE or not _ready.is_set():
        return "", False, ""
    try:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

        # Pass 1: English
        en_result = _send(audio, "en")
        en_text   = en_result.get("text", "").strip()
        if _is_hallucination(en_text):
            print(f"[MLXWhisper/en] hallucination ({len(en_text.split())} words) — dropping")
            en_text = ""
        if en_text:
            triggered, command = wake_check_fn(en_text)
            if triggered:
                print(f"[MLXWhisper/en] {en_text!r}")
                return en_text, True, command

        # Pass 2: Hindi (only if English wake word missed)
        hi_result = _send(audio, "hi", temperature=0.3)
        hi_text   = hi_result.get("text", "").strip()
        if _is_hallucination(hi_text):
            print(f"[MLXWhisper/hi] hallucination ({len(hi_text.split())} words) — dropping")
            hi_text = ""
        if hi_text:
            triggered, command = wake_check_fn(hi_text)
            if en_text:
                print(f"[MLXWhisper/en] {en_text!r}  →  [MLXWhisper/hi] {hi_text!r}")
            else:
                print(f"[MLXWhisper/hi] {hi_text!r}")
            return hi_text, triggered, command

        return en_text or hi_text, False, ""
    except Exception as e:
        print(f"[STT] error: {e}")
        return "", False, ""


def _send(audio: np.ndarray, lang: str | None, timeout: float = 15.0,
          temperature: float = 0.0, initial_prompt: str = "") -> dict:
    """Queue a transcription task to the worker and wait for the result."""
    result_box: list[dict] = []
    done = threading.Event()
    _q.put((audio, lang, temperature, initial_prompt, result_box, done))
    done.wait(timeout=timeout)
    return result_box[0] if result_box else {"text": "", "language": ""}

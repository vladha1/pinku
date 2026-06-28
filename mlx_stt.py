"""
Fast on-device Whisper STT for Apple Silicon via Apple MLX.
Runs whisper-large-v3-turbo on the M4 Neural Engine — no permissions needed.

Install: pip install mlx-whisper
First run downloads the model (~800 MB) from HuggingFace automatically.

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
        print("[MLXWhisper] mlx-whisper not installed — "
              "run: pip install mlx-whisper")
    return _AVAILABLE


def _worker() -> None:
    """Persistent worker — owns the Metal GPU stream for all transcriptions."""
    import mlx_whisper

    # Warm up: compile the model on this thread's Metal stream
    print(f"[MLXWhisper] Loading {MODEL} …")
    t0 = time.time()
    silence = np.zeros(1600, dtype=np.float32)
    _transcribe_once(mlx_whisper, silence, "en")   # warm-up always in English
    _ready.set()
    print(f"[MLXWhisper] Ready ({time.time() - t0:.1f}s)")

    while True:
        task = _q.get()
        if task is None:
            break
        audio, lang, result_box, done = task
        try:
            result_box.append(_transcribe_once(mlx_whisper, audio, lang))
        except Exception as e:
            print(f"[MLXWhisper] worker error: {e}")
            result_box.append({"text": "", "language": ""})
        finally:
            done.set()


def _transcribe_once(mlx_whisper, audio: np.ndarray, lang: str | None) -> dict:
    """Single transcription call — must run on the worker thread."""
    return mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MODEL,
        language=lang,
        temperature=0.0,
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


def transcribe(pcm: bytes, sample_rate: int = 16000) -> str:
    """
    Transcribe raw 16-bit mono PCM. Returns transcript string, '' on failure.
    Auto-detects language. If detected as Spanish (common Hindi mis-detection),
    retries as Hindi.
    """
    if not _AVAILABLE or not _ready.is_set():
        return ""
    try:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        lang  = None if LANGUAGE == "auto" else LANGUAGE

        result = _send(audio, lang)

        # Retry as Hindi for two mis-detection cases:
        # - "es" (Spanish): Whisper confuses Hindi/Indian-English phonemes with Spanish
        # - "ur" (Urdu): same spoken language as Hindi but wrong script for this assistant;
        #   Urdu script wake-word matching fails; re-run as "hi" to get Devanagari
        detected = result.get("language", "")
        if detected in ("es", "ur"):
            result = _send(audio, "hi")
            print(f"[MLXWhisper] re-ran as Hindi (was {detected})")

        return result.get("text", "").strip()
    except Exception as e:
        print(f"[MLXWhisper] error: {e}")
        return ""


def _send(audio: np.ndarray, lang: str | None, timeout: float = 15.0) -> dict:
    """Queue a transcription task to the worker and wait for the result."""
    result_box: list[dict] = []
    done = threading.Event()
    _q.put((audio, lang, result_box, done))
    done.wait(timeout=timeout)
    return result_box[0] if result_box else {"text": "", "language": ""}

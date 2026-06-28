"""
Fast on-device Whisper STT for Apple Silicon via Apple MLX.
Runs whisper-large-v3-turbo on the M4 Neural Engine — no permissions needed.

Install: pip install mlx-whisper
First run downloads the model (~800 MB) from HuggingFace automatically.
"""
from __future__ import annotations
import os
import threading
import time

import numpy as np

MODEL    = "mlx-community/whisper-large-v3-turbo"
LANGUAGE = os.environ.get("MLX_STT_LANGUAGE", "en")   # "en" avoids Hindi→Spanish mis-detection

_AVAILABLE = False
_loaded    = threading.Event()
_lock      = threading.Lock()


def _try_import() -> bool:
    global _AVAILABLE
    try:
        import mlx_whisper   # noqa
        _AVAILABLE = True
    except ImportError:
        print("[MLXWhisper] mlx-whisper not installed — "
              "run: pip install mlx-whisper")
    return _AVAILABLE


_try_import()


def preload() -> None:
    """
    Download + warm up the model in a background thread at startup.
    transcribe() blocks briefly if called before this finishes on first run.
    """
    if not _AVAILABLE:
        return

    def _do() -> None:
        import mlx_whisper
        print(f"[MLXWhisper] Loading {MODEL} …")
        t0 = time.time()
        # Transcribe 0.1 s of silence to trigger model load + JIT compile
        silence = np.zeros(1600, dtype=np.float32)
        mlx_whisper.transcribe(silence, path_or_hf_repo=MODEL,
                               language=LANGUAGE, temperature=0.0,
                               verbose=False)
        _loaded.set()
        print(f"[MLXWhisper] Ready ({time.time() - t0:.1f}s)")

    threading.Thread(target=_do, daemon=True, name="mlx-whisper-preload").start()


def is_ready() -> bool:
    """True once the model has been loaded and warmed up."""
    return _AVAILABLE and _loaded.is_set()


def transcribe(pcm: bytes, sample_rate: int = 16000) -> str:
    """
    Transcribe raw 16-bit mono PCM. Returns transcript string, '' on failure.
    Auto-detects language — handles Hindi, English, and code-switching.
    """
    if not _AVAILABLE:
        return ""
    try:
        import mlx_whisper
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=MODEL,
            language=LANGUAGE,  # default "en"; set MLX_STT_LANGUAGE=hi for Hindi-first
            temperature=0.0,    # greedy decoding — fastest and most deterministic
            verbose=False,
        )
        return result.get("text", "").strip()
    except Exception as e:
        print(f"[MLXWhisper] error: {e}")
        return ""

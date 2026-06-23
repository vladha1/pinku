"""
Speaker identification gate using Resemblyzer.

Loads voice embeddings (profiles/*.npy) at startup.
identify(pcm) returns (speaker_name | None, confidence_score).

Two-tier threshold (configured in config.py):
  score >= SPEAKER_THRESHOLD_ACCEPT    → known speaker, name attached to context
  score in [UNCERTAIN, ACCEPT)         → uncertain, passes through as anonymous
  score < SPEAKER_THRESHOLD_UNCERTAIN  → definitely unknown, dropped by _voice_loop
"""
from __future__ import annotations
import os
import threading
import numpy as np
from config import (
    SPEAKER_PROFILES_DIR,
    SPEAKER_THRESHOLD_ACCEPT,
)

_encoder      = None
_encoder_lock = threading.Lock()
_profiles: dict[str, np.ndarray] = {}   # name → L2-normalised 256-d embedding


def _get_encoder():
    global _encoder
    if _encoder is not None:
        return _encoder
    with _encoder_lock:
        if _encoder is not None:
            return _encoder
        from resemblyzer import VoiceEncoder
        print("[SpeakerID] Loading VoiceEncoder …")
        _encoder = VoiceEncoder()
        print("[SpeakerID] VoiceEncoder ready")
    return _encoder


def load(profiles_dir: str = SPEAKER_PROFILES_DIR) -> int:
    """Load all .npy embeddings from profiles_dir. Returns count. Call once at startup."""
    _profiles.clear()
    if not os.path.isdir(profiles_dir):
        print(f"[SpeakerID] No profiles dir ({profiles_dir}) — speaker gate disabled")
        return 0

    count = 0
    for fname in sorted(os.listdir(profiles_dir)):
        if not fname.endswith(".npy"):
            continue
        name = fname[:-4]
        try:
            emb = np.load(os.path.join(profiles_dir, fname)).astype(np.float32)
            # Ensure unit length so dot product == cosine similarity
            norm = np.linalg.norm(emb)
            if norm > 1e-8:
                emb = emb / norm
            _profiles[name] = emb
            count += 1
        except Exception as e:
            print(f"[SpeakerID] Failed to load {fname}: {e}")

    if count:
        names = ", ".join(_profiles)
        print(f"[SpeakerID] {count} profile(s) loaded: {names}")
        _get_encoder()   # warm up encoder now so first identify() call isn't slow
    else:
        print("[SpeakerID] Profiles dir exists but no .npy files found — gate disabled")

    return count


def has_profiles() -> bool:
    return bool(_profiles)


def identify(pcm: bytes, sample_rate: int = 16000) -> tuple[str | None, float]:
    """
    Identify the speaker in raw 16-bit mono PCM audio.

    Returns (name, score):
      name  = best-matching profile name if score >= ACCEPT threshold, else None
      score = best cosine similarity (0.0 – 1.0); 1.0 returned when no profiles loaded

    Caller is responsible for acting on the score:
      >= SPEAKER_THRESHOLD_ACCEPT    → known speaker
      [UNCERTAIN, ACCEPT)            → pass through as anonymous
      < SPEAKER_THRESHOLD_UNCERTAIN  → drop silently
    """
    if not _profiles:
        return None, 1.0   # no profiles → open gate, pass everything

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    # Pad very short clips to 1 second so the encoder has enough context.
    # Shorter clips still work but give noisier scores.
    if len(audio) < sample_rate:
        audio = np.pad(audio, (0, sample_rate - len(audio)))

    enc = _get_encoder()
    try:
        from resemblyzer import preprocess_wav
        wav = preprocess_wav(audio)          # trims silence, ensures correct format
        embedding = enc.embed_utterance(wav) # returns unit-length 256-d vector
    except Exception as e:
        print(f"[SpeakerID] Embed error: {e} — passing through")
        return None, 1.0   # fail open

    best_name  = None
    best_score = 0.0

    for name, profile_emb in _profiles.items():
        # Both embeddings are unit-length → dot product == cosine similarity
        score = float(np.dot(embedding, profile_emb))
        if score > best_score:
            best_score = score
            best_name  = name

    if best_score >= SPEAKER_THRESHOLD_ACCEPT:
        return best_name, best_score
    return None, best_score

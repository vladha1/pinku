"""
Pinku configuration — all tunable constants in one place.
Edit here; no .env required for local-only setup.
"""

import os

# ── LLM (Ollama) ──────────────────────────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")   # fast on M4; swap to llama3.1:8b for depth

# ── STT (faster-whisper) ──────────────────────────────────────────────────────
WHISPER_MODEL     = os.getenv("WHISPER_MODEL", "small")  # "small" = better accuracy; "base" = faster but misses wake word
WHISPER_DEVICE    = "auto"      # "auto" picks mps on Apple Silicon, cpu otherwise
WHISPER_LANGUAGE  = None        # None = auto-detect; "en" / "hi" to force

# ── TTS (macOS say) ───────────────────────────────────────────────────────────
SAY_VOICE_EN = os.getenv("SAY_VOICE_EN", "Ava (Enhanced)")   # macOS English — download Enhanced/Premium in System Settings → Accessibility → Spoken Content
SAY_VOICE_HI = os.getenv("SAY_VOICE_HI", "Lekha")    # macOS Hindi voice (speaks Devanagari); fallback: Tara (Indian-EN)
SAY_RATE     = int(os.getenv("SAY_RATE", "185"))        # words per minute

# ── Voice activity detection ──────────────────────────────────────────────────
VAD_SILENCE_SEC   = float(os.getenv("VAD_SILENCE_SEC",   "0.7"))  # silence needed to end utterance — 0.7s is faster without cutting off speech
# Minimum echo-settle after TTS before mic reopens.
# Short replies (< 6s TTS) use this floor; longer replies scale with duration.
# 1.5 s is enough for a quiet room; raise to 2.5–3.0 if echo/false-triggers return.
TTS_SETTLE_MIN    = float(os.getenv("TTS_SETTLE_MIN", "0.4"))
VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS",   "150"))  # ignore clips shorter than this
# VAD aggressiveness 0-3: 0=breaks silence detection (everything=speech, never ends)
# 1 = permissive enough for distance, still detects silence gaps properly
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "1"))
MIC_SAMPLE_RATE   = 16000
MIC_CHUNK_MS      = 30     # VAD chunk size (ms) — must be 10, 20, or 30
# Explicit mic device index — avoids grabbing iPhone/Bluetooth mics that lock PortAudio.
# -1 = use system default.  Run: python3 -c "import pyaudio; p=pyaudio.PyAudio();
# [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"
MIC_DEVICE_INDEX  = int(os.getenv("MIC_DEVICE_INDEX", "-1"))

# ── Music generation ──────────────────────────────────────────────────────────
# Backend: "musicgen" (local Meta AudioCraft) or "abc" (Ollama → MIDI, lightweight)
MUSIC_BACKEND     = os.getenv("MUSIC_BACKEND",  "musicgen")
# MusicGen model size: "facebook/musicgen-small"  (~300 MB, ~15s/chunk on M4)
#                      "facebook/musicgen-medium" (~1.5 GB, ~45s/chunk, higher quality)
MUSICGEN_MODEL    = os.getenv("MUSICGEN_MODEL", "facebook/musicgen-small")
MUSIC_CHUNK_SEC   = int(os.getenv("MUSIC_CHUNK_SEC", "30"))   # seconds per generated clip
# Where to store the downloaded model weights.  Pinned to a fixed path so the
# model is never re-downloaded across restarts (HF default ~/.cache can vary
# depending on how the process is launched / which user runs it).
MUSIC_MODEL_CACHE = os.getenv(
    "MUSIC_MODEL_CACHE",
    os.path.join(os.path.expanduser("~"), "pinku", "models"),
)

# ── Web dashboard ─────────────────────────────────────────────────────────────
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "5100"))
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")

# ── Speaker identification ─────────────────────────────────────────────────────
# Resemblyzer-based local speaker gate. Audio from unrecognised voices is
# dropped before wake-word detection, eliminating TV / TTS-echo false triggers.
# Install: pip install resemblyzer
# Enroll:  python enroll_speakers.py --name <name> --record 10
#
# Set SPEAKER_ID_ENABLED=0 to bypass (e.g. when no profiles are enrolled yet).
SPEAKER_PROFILES_DIR        = os.getenv(
    "SPEAKER_PROFILES_DIR",
    os.path.join(os.path.expanduser("~"), "pinku", "profiles"),
)
SPEAKER_ID_ENABLED          = bool(int(os.getenv("SPEAKER_ID_ENABLED",          "1")))
# Cosine similarity thresholds (0.0–1.0):
#   >= ACCEPT    → confident match — attach speaker name to LLM context
#   [UNCERTAIN, ACCEPT) → marginal — pass through as anonymous, no name attached
#   < UNCERTAIN  → clearly unknown — drop silently (TV, echo, guests, etc.)
SPEAKER_THRESHOLD_ACCEPT    = float(os.getenv("SPEAKER_THRESHOLD_ACCEPT",    "0.80"))
SPEAKER_THRESHOLD_UNCERTAIN = float(os.getenv("SPEAKER_THRESHOLD_UNCERTAIN", "0.65"))
# Minimum margin by which the best-matching profile must beat the second-best.
# Prevents family-voice confusion where related speakers score similarly against
# the same profile (e.g. parent 0.74, child 0.68 → margin 0.06 < 0.08 → anonymous).
SPEAKER_ID_MARGIN           = float(os.getenv("SPEAKER_ID_MARGIN",           "0.08"))
# Seconds after Pinku finishes speaking during which the speaker gate uses
# SPEAKER_THRESHOLD_UNCERTAIN instead of SPEAKER_THRESHOLD_ACCEPT, making
# it easier for family/guests to follow up naturally without the wake word.
SPEAKER_POST_REPLY_WINDOW   = int(os.getenv("SPEAKER_POST_REPLY_WINDOW",   "12"))

# ── Wake / session ────────────────────────────────────────────────────────────
SESSION_TIMEOUT   = float(os.getenv("SESSION_TIMEOUT", "60.0"))  # seconds of silence before going idle (60 = enough time to think + respond)
LOG_DIR           = os.getenv("LOG_DIR", os.path.join(os.path.expanduser("~"), "pinku", "logs"))

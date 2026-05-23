"""
Pinku configuration — all tunable constants in one place.
Edit here; no .env required for local-only setup.
"""

import os

# ── LLM (Ollama) ──────────────────────────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")   # fast on M4; swap to llama3.1:8b for depth

# ── STT (faster-whisper) ──────────────────────────────────────────────────────
WHISPER_MODEL     = os.getenv("WHISPER_MODEL", "base.en")  # tiny/base/small/medium; base.en is good on M4
WHISPER_DEVICE    = "auto"      # "auto" picks mps on Apple Silicon, cpu otherwise
WHISPER_LANGUAGE  = None        # None = auto-detect; "en" / "hi" to force

# ── TTS (macOS say) ───────────────────────────────────────────────────────────
SAY_VOICE_EN = os.getenv("SAY_VOICE_EN", "Samantha")   # macOS English female voice
SAY_VOICE_HI = os.getenv("SAY_VOICE_HI", "Tara")      # Indian English female voice
SAY_RATE     = int(os.getenv("SAY_RATE", "185"))        # words per minute

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX  = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH  = int(os.getenv("CAMERA_WIDTH",  "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
CAMERA_FPS    = int(os.getenv("CAMERA_FPS",    "30"))
YOLO_MODEL    = os.getenv("YOLO_MODEL", "yolov8n.pt")  # nano; swap to yolov8s for better accuracy
YOLO_CONF     = float(os.getenv("YOLO_CONF", "0.45"))
DETECT_EVERY  = float(os.getenv("DETECT_EVERY", "2.0"))  # seconds between detection cycles

# ── Laser detection ───────────────────────────────────────────────────────────
LASER_STARS_MIN = int(os.getenv("LASER_STARS_MIN", "5"))    # ≥ this many dots = star field
LASER_COOLDOWN  = float(os.getenv("LASER_COOLDOWN", "6.0")) # seconds between laser actions

# ── Voice activity detection ──────────────────────────────────────────────────
VAD_SILENCE_SEC   = float(os.getenv("VAD_SILENCE_SEC",   "1.5"))  # silence to end utterance
VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS",   "500"))  # ignore clips shorter than this
MIC_SAMPLE_RATE   = 16000
MIC_CHUNK_MS      = 30     # VAD chunk size (ms) — must be 10, 20, or 30

# ── Web dashboard ─────────────────────────────────────────────────────────────
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "5100"))
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")

# ── Wake / session ────────────────────────────────────────────────────────────
SESSION_TIMEOUT   = float(os.getenv("SESSION_TIMEOUT", "45.0"))  # seconds of silence before going idle
LOG_DIR           = os.getenv("LOG_DIR", "logs")

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

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX  = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH  = int(os.getenv("CAMERA_WIDTH",  "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
CAMERA_FPS    = int(os.getenv("CAMERA_FPS",    "30"))
YOLO_MODEL    = os.getenv("YOLO_MODEL", "yolov8n.pt")   # person + object detection
GESTURE_MODEL = os.getenv("GESTURE_MODEL", "gesture_yolo.pt")  # downloaded on first run
YOLO_CONF     = float(os.getenv("YOLO_CONF", "0.45"))
YOLO_IGNORE   = set(os.getenv("YOLO_IGNORE", "kite,ceiling fan").split(","))
DETECT_EVERY  = float(os.getenv("DETECT_EVERY", "3.0"))  # seconds between detection cycles
# Minimum inter-shoulder width in pixels to count someone as "present" (proximity filter).
# Raise to require the person to be closer; lower to accept from further away.
# At 640px wide: ~100px ≈ 3-4m, ~140px ≈ 2m, ~60px ≈ 5-6m.
MIN_SHOULDER_PX = int(os.getenv("MIN_SHOULDER_PX", "100"))

# ── Laser dot detection ───────────────────────────────────────────────────────
# Detects a green laser pointer dot on the wall via HSV masking.
# Set LASER_DETECT=0 to disable.
LASER_DETECT      = bool(int(os.getenv("LASER_DETECT", "1")))
# HSV ranges (OpenCV scale: H 0-180, S 0-255, V 0-255).
# 532 nm green laser ≈ H 55-75. Loosen S/V if dot isn't detected.
LASER_HUE_LO      = int(os.getenv("LASER_HUE_LO",  "35"))
LASER_HUE_HI      = int(os.getenv("LASER_HUE_HI",  "95"))
LASER_SAT_MIN     = int(os.getenv("LASER_SAT_MIN",  "50"))   # lower = easier to detect
LASER_VAL_MIN     = int(os.getenv("LASER_VAL_MIN", "100"))   # lower = easier to detect
# Laser runs its own fast loop independent of YOLO (cheap HSV op, ~2ms)
LASER_POLL_SEC    = float(os.getenv("LASER_POLL_SEC", "0.12"))   # ~8 fps
# Movement threshold — only re-fire if dot moves more than this fraction of frame
LASER_MOVE_THRESH = float(os.getenv("LASER_MOVE_THRESH", "0.04"))
# Real laser dots are tiny — reject blobs larger than this many pixels.
# A 640x480 frame: a dot of radius ~11px = ~380px area. Projector sectors are 1000s of px.
LASER_MAX_AREA    = int(os.getenv("LASER_MAX_AREA", "400"))
# If more than this many blobs survive the filters in one frame, it's almost certainly
# projector noise or ambient green light — ignore the whole frame.
LASER_MAX_DOTS    = int(os.getenv("LASER_MAX_DOTS", "5"))

# ── Voice activity detection ──────────────────────────────────────────────────
VAD_SILENCE_SEC   = float(os.getenv("VAD_SILENCE_SEC",   "0.8"))  # silence needed to end utterance — 0.8s balances response speed vs cutting off speech
VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS",   "150"))  # ignore clips shorter than this
# VAD aggressiveness 0-3: 0=breaks silence detection (everything=speech, never ends)
# 1 = permissive enough for distance, still detects silence gaps properly
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "1"))
MIC_SAMPLE_RATE   = 16000
MIC_CHUNK_MS      = 30     # VAD chunk size (ms) — must be 10, 20, or 30

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

# ── Wake / session ────────────────────────────────────────────────────────────
SESSION_TIMEOUT   = float(os.getenv("SESSION_TIMEOUT", "60.0"))  # seconds of silence before going idle (60 = enough time to think + respond)
LOG_DIR           = os.getenv("LOG_DIR", os.path.join(os.path.expanduser("~"), "pinku", "logs"))

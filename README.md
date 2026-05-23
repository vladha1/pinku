# Pinku 🤖

Local AI home assistant for M4 Mac Mini.  
Everything runs on-device — no cloud APIs required.

| Component | Technology |
|-----------|-----------|
| **Speech → Text** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper `base.en`, MPS) |
| **Language / Routing** | [Ollama](https://ollama.com) · `llama3.2:3b` (or any pulled model) |
| **Text → Speech** | macOS `say` (built-in Neural TTS — Nicky, Lekha, etc.) |
| **Object detection** | YOLOv8n via [Ultralytics](https://github.com/ultralytics/ultralytics) (MPS) |
| **Pose / Gestures** | [MediaPipe](https://mediapipe.dev) |
| **Laser wake** | OpenCV HSV blob detection |
| **Dashboard** | Flask + SSE at `http://localhost:5100` |

---

## Quick start

### 1. Prerequisites

```bash
# Homebrew
brew install portaudio cmake

# PyTorch with MPS (Apple Silicon)
pip install torch torchvision

# Python deps
pip install -r requirements.txt
```

### 2. Ollama

```bash
# Install
brew install ollama

# Start the server
ollama serve &

# Pull a model (fast on M4 — ~2 GB)
ollama pull llama3.2:3b

# Optional: vision model for "describe what you see"
ollama pull llava:7b
```

### 3. Run

```bash
python pinku.py
```

Open the dashboard: **http://localhost:5100**

---

## Options

```
python pinku.py --help

  --no-camera      Skip camera (voice-only mode)
  --no-dashboard   Skip web UI
  --model NAME     Override Ollama model (e.g. llama3.1:8b, mistral)
  --port N         Dashboard port (default 5100)
```

---

## Configuration

All tunable constants live in `config.py` — no `.env` needed.

Key settings:

| Variable | Default | Notes |
|----------|---------|-------|
| `OLLAMA_MODEL` | `llama3.2:3b` | Swap to `llama3.1:8b` for better reasoning |
| `WHISPER_MODEL` | `base.en` | `small` for multilingual / Hindi |
| `WHISPER_LANGUAGE` | `None` (auto) | Set `"hi"` to force Hindi |
| `SAY_VOICE_EN` | `Nicky` | `say -v ?` to list voices |
| `SAY_VOICE_HI` | `Lekha` | Indian-English voice |
| `CAMERA_INDEX` | `0` | USB camera index |
| `DETECT_EVERY` | `2.0` | Seconds between detection cycles |
| `SESSION_TIMEOUT` | `30.0` | Seconds of silence before going idle |

---

## Laser control (ceiling camera)

Point a green laser at the ceiling:

| Gesture | Action |
|---------|--------|
| Single dot | Wake Pinku, open mic |
| Star projector (≥5 dots) | Toggle mute / unmute |

Adjust `LASER_STARS_MIN` in `config.py` if your projector produces more dots.

---

## Voices

List all available macOS voices:
```bash
say -v ?
```

Good options for `SAY_VOICE_EN`: `Nicky`, `Ava`, `Samantha`, `Zoe`  
Good options for `SAY_VOICE_HI`: `Lekha` (Indian English)

---

## Adding capabilities

- **Music**: implement `_handle_music_play()` in `pinku.py` using `mpv` or `afplay`
- **Smart home**: implement `_handle_lights_on/off()` using HomeKit / Home Assistant API  
- **Weather**: implement `_handle_weather()` calling a local weather service  
- **Face ID**: add `face_recognition` library + enroll faces in `camera.py`
- **Hindi STT**: change `WHISPER_MODEL = "small"` and `WHISPER_LANGUAGE = None` in `config.py`

---

## Architecture

```
Mic → AudioRecorder (PyAudio + WebRTC VAD)
    → stt.transcribe() (Whisper local)
    → llm.route() (Ollama → JSON action)
    → handler (chat / time / describe / scripture / …)
    → tts.speak() (macOS say)

Camera → CameraDetector thread
    → YOLO + MediaPipe (every 2s)
    → laser dot check (every frame)
    → callbacks → pinku.py handlers

Dashboard → Flask + SSE → browser
```

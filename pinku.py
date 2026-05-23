#!/usr/bin/env python3
"""
Pinku — local AI home assistant for M4 Mac Mini.

Stack:
  STT  : faster-whisper (local, MPS-accelerated on Apple Silicon)
  LLM  : Ollama (local, llama3.2:3b default)
  TTS  : macOS `say` (built-in neural voices)
  Vision: YOLOv8 + MediaPipe (local, MPS-accelerated)
  Laser: green dot detection for wake / mute toggle

Usage:
  python pinku.py
  python pinku.py --no-camera     # skip camera (voice only)
  python pinku.py --no-dashboard  # skip web UI
  python pinku.py --model llama3.1:8b
"""

from __future__ import annotations
import argparse
import threading
import time
import sys

import config
import stt
import tts
import llm
import logger as log_module
import dashboard


# ── Global state ──────────────────────────────────────────────────────────────

_muted         = threading.Event()   # set = muted (mic disabled)
_awake         = threading.Event()   # set = in active voice session
_stop_all      = threading.Event()
_session_hist: list[dict] = []       # [{role, content}] for LLM context
_det_logger    = log_module.DetectionLogger()


def is_muted() -> bool:
    return _muted.is_set()


# ── Action handlers ───────────────────────────────────────────────────────────

def _speak_reply(reply: str, is_hi: bool):
    dashboard.update_status(speaking=True)
    tts.speak(reply, prefer_hi=is_hi, block=True)
    dashboard.update_status(speaking=False, state="awake")


def _handle_chat(action: dict):
    tr   = action.get("transcript", "")
    lang = action.get("lang", "en")
    is_hi = lang == "hi"
    print(f"[Chat] {tr!r}")
    reply = llm.chat(tr, history=_session_hist)
    _session_hist.append({"role": "user",      "content": tr})
    _session_hist.append({"role": "assistant", "content": reply})
    if len(_session_hist) > 12:
        _session_hist[:] = _session_hist[-12:]
    print(f"[Pinku] {reply!r}")
    dashboard.update_status(last_transcript=tr, last_reply=reply)
    _speak_reply(reply, is_hi)


def _handle_time(action: dict):
    from datetime import datetime
    now  = datetime.now()
    lang = action.get("lang", "en")
    if lang == "hi":
        reply = f"अभी {now.strftime('%-I बजकर %M मिनट')} हैं।"
    else:
        reply = f"It's {now.strftime('%-I:%M %p')}."
    dashboard.update_status(last_transcript=action.get("transcript",""), last_reply=reply)
    _speak_reply(reply, lang == "hi")


def _handle_describe(action: dict):
    """Grab camera frame, describe via vision LLM."""
    from camera import get_frame, frame_to_b64
    frame = get_frame()
    if frame is None:
        tts.speak("Camera isn't available right now.")
        return
    tts.speak("Let me look…")
    b64  = frame_to_b64(frame)
    tr   = action.get("transcript", "What do you see?")
    lang = action.get("lang", "en")
    desc = llm.describe_image(b64, question=tr, is_hi=(lang == "hi"))
    print(f"[Vision] {desc!r}")
    dashboard.update_status(last_transcript=tr, last_reply=desc)
    _speak_reply(desc, lang == "hi")


def _handle_scripture(action: dict):
    topic = action.get("topic", "gita")
    tr    = action.get("transcript", "")
    lang  = action.get("lang", "en")
    is_hi = lang == "hi"

    _PROMPTS = {
        "gita": (
            "You are a Bhagavad Gita scholar. Give the most relevant shloka in Devanagari, "
            "its meaning in 1-2 sentences, and one crisp insight. Under 80 words after the shloka."
        ),
        "ramayana": (
            "You are a Ramayana scholar. Give the most relevant doha/chaupai in Devanagari, "
            "its meaning in 1-2 sentences, and one dharmic insight. Under 80 words after the verse."
        ),
        "patanjali": (
            "You are a Patanjali Yoga Sutras scholar. Give the relevant sutra in Devanagari, "
            "its meaning in 1-2 sentences, and one practical insight. Under 80 words after the sutra."
        ),
        "upanishads": (
            "You are an Upanishad scholar. Give the relevant mantra in Devanagari, "
            "its meaning in 1-2 sentences, and one Vedantic insight. Under 80 words after the verse."
        ),
        "vedas": (
            "You are a Vedic scholar. Give the relevant mantra in Devanagari, "
            "its meaning in 1-2 sentences, and one key insight. Under 80 words after the mantra."
        ),
        "madhushala": (
            "You are a Madhushala scholar. Give the relevant rubai in Devanagari Hindi (all 4 lines), "
            "its meaning in 1-2 sentences, and one line on the deeper metaphor. Under 90 words after the rubai."
        ),
    }
    system = _PROMPTS.get(topic, _PROMPTS["gita"])
    if is_hi:
        system += " Respond in Hindi (Devanagari)."
    reply = llm.chat(tr, system_extra=system)
    print(f"[Scripture:{topic}] {reply[:80]!r}")
    dashboard.update_status(last_transcript=tr, last_reply=reply)
    _speak_reply(reply, is_hi)


def _handle_mute():
    _muted.set()
    _awake.clear()
    tts.stop_speaking()
    tts.play_mute()
    dashboard.update_status(state="idle", muted=True)
    print("[Pinku] Muted")


def _handle_unmute():
    _muted.clear()
    tts.play_unmute()
    dashboard.update_status(state="idle", muted=False)
    print("[Pinku] Unmuted")
    _extend_session()


def _extend_session():
    """Open a voice session window."""
    _awake.set()
    dashboard.update_status(state="awake")
    # Auto-close after SESSION_TIMEOUT seconds of inactivity (handled in listen loop)


# ── Laser callbacks ───────────────────────────────────────────────────────────

def on_laser_wake():
    if is_muted():
        return
    tts.play_beep()
    _extend_session()
    print("[Laser] Wake → listening")


def on_laser_stars(currently_muted: bool):
    if currently_muted:
        _handle_unmute()
    else:
        _handle_mute()


# ── Detection callback ────────────────────────────────────────────────────────

def on_detection(event: dict):
    _det_logger.log(event)
    dashboard.push_detection(event)


# ── Main voice loop ───────────────────────────────────────────────────────────

def _voice_loop(recorder: stt.AudioRecorder):
    """
    Continuously:
    1. Wait for an utterance (VAD-gated)
    2. Transcribe with Whisper
    3. Route with Ollama → action
    4. Execute action
    """
    last_speech_at = time.time()

    while not _stop_all.is_set():
        if is_muted():
            time.sleep(0.2)
            continue

        # Session timeout
        if _awake.is_set() and time.time() - last_speech_at > config.SESSION_TIMEOUT:
            _awake.clear()
            dashboard.update_status(state="idle")
            print("[Pinku] Session timeout → idle")

        dashboard.update_status(state="awake" if _awake.is_set() else "idle")

        pcm = recorder.wait_for_utterance(stop_event=_stop_all, timeout=5.0)
        if pcm is None:
            continue

        text = stt.transcribe(pcm)
        if not text or len(text.split()) < 2:
            continue  # too short / noise

        last_speech_at = time.time()
        _awake.set()
        dashboard.update_status(state="processing", last_transcript=text)

        # Route
        action = llm.route(text)
        act    = action.get("action", "chat")
        print(f"[Route] action={act!r} text={text!r}")

        if act == "ignore":
            continue
        elif act == "mute":
            _handle_mute()
        elif act == "unmute":
            _handle_unmute()
        elif act == "time":
            _handle_time(action)
        elif act == "describe":
            _handle_describe(action)
        elif act == "scripture":
            _handle_scripture(action)
        elif act in ("music_play", "music_stop", "lights_on", "lights_off", "weather"):
            # Stubs — implement as needed
            tts.speak(f"Sorry, {act.replace('_',' ')} isn't set up yet.")
        else:  # "chat" and everything else
            _handle_chat(action)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Pinku — local home assistant")
    ap.add_argument("--no-camera",    action="store_true", help="Disable camera detection")
    ap.add_argument("--no-dashboard", action="store_true", help="Disable web dashboard")
    ap.add_argument("--model",        default=None,        help="Override Ollama model")
    ap.add_argument("--port",         type=int, default=config.DASHBOARD_PORT)
    args = ap.parse_args()

    if args.model:
        config.OLLAMA_MODEL = args.model

    print("=" * 50)
    print("  Pinku — local AI assistant")
    print(f"  LLM    : {config.OLLAMA_MODEL} via Ollama")
    print(f"  STT    : Whisper {config.WHISPER_MODEL}")
    print(f"  TTS    : macOS say ({config.SAY_VOICE_EN})")
    print("=" * 50)

    # ── Check Ollama ──────────────────────────────────────────────────────────
    models = llm.models_available()
    if not models:
        print("[WARN] Ollama not reachable — start with: ollama serve")
    else:
        print(f"[LLM] Available models: {', '.join(models)}")
        if config.OLLAMA_MODEL not in models:
            print(f"[WARN] {config.OLLAMA_MODEL!r} not pulled — run: ollama pull {config.OLLAMA_MODEL}")

    # ── Preload Whisper ───────────────────────────────────────────────────────
    stt.preload()

    # ── Dashboard ─────────────────────────────────────────────────────────────
    if not args.no_dashboard:
        dashboard.start(logger=_det_logger, port=args.port)

    # ── Camera ────────────────────────────────────────────────────────────────
    cam = None
    if not args.no_camera:
        from camera import CameraDetector
        cam = CameraDetector(
            on_detection=on_detection,
            on_laser_wake=on_laser_wake,
            on_laser_stars=on_laser_stars,
            is_muted_fn=is_muted,
        )
        cam.start()

    # ── Mic ───────────────────────────────────────────────────────────────────
    recorder = stt.AudioRecorder()
    recorder.start()

    tts.speak("Pinku ready.", prefer_hi=False)
    dashboard.update_status(state="idle", muted=False, model=config.OLLAMA_MODEL)

    try:
        _voice_loop(recorder)
    except KeyboardInterrupt:
        print("\n[Pinku] Shutting down …")
    finally:
        _stop_all.set()
        recorder.stop()
        if cam:
            cam.stop()


if __name__ == "__main__":
    main()

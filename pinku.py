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
import knowledge
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
    reply = llm.chat(tr, history=_session_hist, is_hi=is_hi)
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
    """Route all knowledge topics (scripture, yoga, history, music, etc.) via knowledge.py."""
    knowledge.handle(
        action,
        speak_fn   = _speak_reply,
        update_fn  = dashboard.update_status,
        session_hist = _session_hist,
    )


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


# ── Gesture action map ────────────────────────────────────────────────────────
#
# Gestures that trigger Pinku actions (require active session OR override).
# Cooldown prevents repeat-firing while hand is held.
_gesture_last_at: dict[str, float] = {}
_GESTURE_COOLDOWN = 4.0   # seconds between same-gesture actions

_GESTURE_ACTIONS = {
    # Gesture label   → (requires_session, action_fn_name)
    "Thumbs Up":      (False, "_gesture_thumbs_up"),
    "Thumbs Down":    (False, "_gesture_thumbs_down"),
    "Open Hand":      (False, "_gesture_open_hand"),
    "Fist":           (False, "_gesture_fist"),
    "Peace":          (True,  "_gesture_peace"),
    "Pointing":       (True,  "_gesture_pointing"),
    "Call Me":        (False, "_gesture_call_me"),
}

def _gesture_throttle(label: str) -> bool:
    """Return True if we should fire (not in cooldown)."""
    now = time.time()
    if now - _gesture_last_at.get(label, 0) < _GESTURE_COOLDOWN:
        return False
    _gesture_last_at[label] = now
    return True

def _gesture_thumbs_up():
    """👍 Thumbs Up — wake / positive ack."""
    if is_muted():
        return
    print("[Gesture] 👍 Thumbs Up → wake session")
    tts.play_beep()
    _extend_session()

def _gesture_thumbs_down():
    """👎 Thumbs Down — stop speaking / dismiss."""
    print("[Gesture] 👎 Thumbs Down → stop speaking")
    tts.stop_speaking()
    dashboard.update_status(state="awake" if _awake.is_set() else "idle")

def _gesture_open_hand():
    """🖐 Open Hand — pause / stop speaking."""
    print("[Gesture] 🖐 Open Hand → stop speaking")
    tts.stop_speaking()

def _gesture_fist():
    """✊ Fist — mute toggle."""
    print("[Gesture] ✊ Fist → mute toggle")
    if is_muted():
        _handle_unmute()
    else:
        _handle_mute()

def _gesture_peace():
    """✌️ Peace — ask what time it is."""
    print("[Gesture] ✌️ Peace → time")
    _handle_time({"transcript": "What time is it?", "lang": "en"})

def _gesture_pointing():
    """☝️ Pointing — describe what camera sees."""
    print("[Gesture] ☝️ Pointing → describe")
    _handle_describe({"transcript": "What do you see?", "lang": "en"})

def _gesture_call_me():
    """🤙 Call Me — unmute + open session."""
    print("[Gesture] 🤙 Call Me → unmute + wake")
    if is_muted():
        _handle_unmute()
    else:
        tts.play_beep()
        _extend_session()

_GESTURE_FN_MAP = {
    "_gesture_thumbs_up":   _gesture_thumbs_up,
    "_gesture_thumbs_down": _gesture_thumbs_down,
    "_gesture_open_hand":   _gesture_open_hand,
    "_gesture_fist":        _gesture_fist,
    "_gesture_peace":       _gesture_peace,
    "_gesture_pointing":    _gesture_pointing,
    "_gesture_call_me":     _gesture_call_me,
}

def _dispatch_gestures(event: dict):
    for g in event.get("gestures", []):
        label = g.get("gesture", "")
        if label not in _GESTURE_ACTIONS:
            continue
        requires_session, fn_name = _GESTURE_ACTIONS[label]
        if requires_session and not _awake.is_set():
            continue
        if not _gesture_throttle(label):
            continue
        fn = _GESTURE_FN_MAP.get(fn_name)
        if fn:
            threading.Thread(target=fn, daemon=True).start()


# ── Detection callback ────────────────────────────────────────────────────────

def on_detection(event: dict):
    _det_logger.log(event)
    dashboard.push_detection(event)
    _dispatch_gestures(event)


# ── Wake word / session end detection ────────────────────────────────────────

# Wake phrases — must appear at the START of the utterance.
# Sorted longest-first so "hey pinku" matches before bare "pinku".
_WAKE_PHRASES = [
    "hey pinku", "hi pinku", "ok pinku", "okay pinku",
    "hello pinku", "yo pinku", "pinku",
    # Common Whisper mishearings of "Pinku":
    "hey pinko", "hi pinko", "ok pinko", "hello pinko", "pinko",
    "hey pink", "hi pink", "ok pink", "pink",
    "hey pingu", "pingu",
]

# End-session phrases — only checked when already in session AND utterance
# is short (≤ 6 words) to avoid "I said goodbye to a friend" killing the chat.
_END_PHRASES = {
    "end chat", "end conversation", "stop listening", "go to sleep",
    "goodbye pinku", "bye pinku", "bye bye", "that's all",
    "thats all", "we're done", "were done", "stop",
    "ok thanks pinku", "thanks pinku", "thank you pinku",
}


import re as _re
# Catches Whisper mishearings of "Pinku" at the very start of an utterance.
# Handles punctuation in prefix ("Hey, Pinku"), leading dots/spaces Whisper adds,
# and Devanagari पिंकू (Hindi wake word).
_PINKU_RE = _re.compile(
    r'^[.\s]*'                                            # strip leading dots/spaces Whisper adds
    r'(?:(?:hey|hi|ok|okay|hello|yo|अरे|हे)[,.\s]+)?'   # optional prefix + any punctuation/space
    r'(pinku|pinko|pinco|pingo|pingu|pinkoo|penku|penko|pink|पिंकू|पिंकु|पिंको)\b[,\s।]*',
    _re.IGNORECASE,
)

def _check_wake(text: str) -> tuple[bool, str]:
    """
    Returns (triggered, command).
    Matches wake word at START of utterance only — mid-sentence ignored.
    Handles Whisper mishearings via regex (pinko, pink, pingo, etc.).
    """
    t = text.strip()
    # Strip leading filler
    for filler in ("um ", "uh ", "so ", "like ", "well ", "okay so "):
        if t.lower().startswith(filler):
            t = t[len(filler):]

    # Regex match — catches all Whisper mishearings in one shot
    m = _PINKU_RE.match(t)
    if m:
        rest = t[m.end():].strip(" ,.")
        return True, rest

    # Exact phrase fallback
    tl = t.lower()
    for phrase in _WAKE_PHRASES:
        if tl.startswith(phrase):
            rest = t[len(phrase):].strip(" ,.")
            return True, rest

    return False, text


def _check_end(text: str) -> bool:
    """
    True if this short utterance is a session-end command.
    Only fires when utterance ≤ 6 words, preventing mid-sentence false positives.
    """
    words = text.split()
    if len(words) > 6:
        return False
    t = text.lower().strip().rstrip(".,!?")
    return t in _END_PHRASES or any(t.startswith(p) for p in _END_PHRASES)


# ── Main voice loop ───────────────────────────────────────────────────────────

def _voice_loop(recorder: stt.AudioRecorder):
    """
    Idle mode  : only respond when wake phrase heard at start of utterance.
    Active mode: respond to everything; auto-expire after SESSION_TIMEOUT;
                 explicit end phrases close the session early.
    """
    last_speech_at = time.time()

    while not _stop_all.is_set():
        if is_muted():
            time.sleep(0.2)
            continue

        # ── Session timeout ───────────────────────────────────────────────────
        if _awake.is_set() and time.time() - last_speech_at > config.SESSION_TIMEOUT:
            _awake.clear()
            dashboard.update_status(state="idle")
            tts.speak("Going quiet.", block=False)
            print("[Pinku] Session timeout → idle")

        pcm = recorder.wait_for_utterance(stop_event=_stop_all, timeout=5.0)
        if pcm is None:
            continue

        try:
            text = stt.transcribe(pcm)
        except Exception as e:
            print(f"[STT] transcribe error: {e}")
            continue
        if not text:
            continue

        # ── Gate: idle mode ───────────────────────────────────────────────────
        if not _awake.is_set():
            triggered, command = _check_wake(text)
            if not triggered:
                print(f"[Pinku] Idle — ignored: {text!r}")
                continue
            # Wake confirmed
            _awake.set()
            last_speech_at = time.time()
            print(f"[Pinku] Wake word → session open (command={command!r})")
            if not command:
                # Just the wake word alone — beep and wait for next utterance
                tts.play_beep()
                dashboard.update_status(state="awake")
                continue
            text = command   # use the part after the wake word as the command

        # ── Active session ────────────────────────────────────────────────────
        last_speech_at = time.time()

        # Check for session-end phrase (short utterances only)
        if _check_end(text):
            _awake.clear()
            _session_hist.clear()
            tts.speak("Okay, bye for now.", block=False)
            dashboard.update_status(state="idle", last_transcript=text, last_reply="Okay, bye for now.")
            print("[Pinku] Session ended by user")
            continue

        dashboard.update_status(state="processing", last_transcript=text)

        # ── Route & execute ───────────────────────────────────────────────────
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
            tts.speak(f"Sorry, {act.replace('_', ' ')} isn't set up yet.")
        else:
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
        # Wire dashboard buttons → pinku actions
        dashboard.register_action("wake",         _extend_session)
        dashboard.register_action("mute",         _handle_mute)
        dashboard.register_action("unmute",       _handle_unmute)
        dashboard.register_action("stop",         tts.stop_speaking)
        dashboard.register_action("mute_toggle",  lambda: _handle_unmute() if is_muted() else _handle_mute())

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

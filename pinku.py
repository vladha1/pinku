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
_user_muted    = threading.Event()   # set = user manually muted (gesture/button)
_awake         = threading.Event()   # set = in active voice session
_stop_all      = threading.Event()
_session_hist: list[dict] = []       # [{role, content}] for LLM context
_det_logger    = log_module.DetectionLogger()

_last_speech_at: float = time.time()   # reset on every utterance / wake / gesture
_last_human_at:  float = 0.0           # last time camera saw a person in frame
_camera_enabled: bool  = False         # True once camera starts; enables auto-wake

# How long after last camera person detection before we consider room empty
_HUMAN_GONE_SEC = 12.0


def is_muted() -> bool:
    return _muted.is_set()


def _human_is_present() -> bool:
    """True if camera is running and saw someone within the last _HUMAN_GONE_SEC seconds."""
    return _camera_enabled and (time.time() - _last_human_at < _HUMAN_GONE_SEC)


# ── Logging helper ────────────────────────────────────────────────────────────

def _log(level: str, msg: str):
    """Print to stdout and stream to dashboard Log tab."""
    print(f"[{level.upper()}] {msg}")
    try:
        dashboard.log_message(level, msg)
    except Exception:
        pass


# ── Action handlers ───────────────────────────────────────────────────────────

def _speak_reply(reply: str, is_hi: bool):
    global _last_speech_at
    _log("pinku", reply)
    dashboard.update_status(speaking=True)
    _muted.set()                          # silence mic while speaking to prevent feedback
    tts.speak(reply, prefer_hi=is_hi, block=True)
    # Only clear mic mute if the user hasn't manually muted — don't override their intent
    if not _user_muted.is_set():
        _muted.clear()
    # Reset inactivity timer from end of reply so user has the full window to respond
    _last_speech_at = time.time()
    dashboard.update_status(speaking=False, state="awake" if _awake.is_set() else "idle")


def _handle_chat(action: dict):
    tr    = action.get("transcript", "")
    lang  = action.get("lang", "en")
    is_hi = lang == "hi"
    _log("user", tr)
    tts.play_think()                                          # tick: sending to LLM
    reply = llm.chat(tr, history=_session_hist, is_hi=is_hi)
    _session_hist.append({"role": "user",      "content": tr})
    _session_hist.append({"role": "assistant", "content": reply})
    if len(_session_hist) > 12:
        _session_hist[:] = _session_hist[-12:]
    dashboard.update_status(last_transcript=tr, last_reply=reply)
    _speak_reply(reply, is_hi)   # ← speak the answer


def _handle_time(action: dict):
    from datetime import datetime
    now  = datetime.now()
    lang = action.get("lang", "en")
    _log("user", action.get("transcript", ""))
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
    tr   = action.get("transcript", "What do you see?")
    lang = action.get("lang", "en")
    _log("user", tr)
    tts.speak("Let me look…")
    tts.play_think()
    b64  = frame_to_b64(frame)
    desc = llm.describe_image(b64, question=tr, is_hi=(lang == "hi"))
    print(f"[Vision] {desc!r}")
    dashboard.update_status(last_transcript=tr, last_reply=desc)
    _speak_reply(desc, lang == "hi")


def _handle_scripture(action: dict):
    """Route all knowledge topics (scripture, yoga, history, music, etc.) via knowledge.py."""
    tts.play_think()
    knowledge.handle(
        action,
        speak_fn     = _speak_reply,
        update_fn    = dashboard.update_status,
        session_hist = _session_hist,
        log_fn       = _log,
    )


def _handle_mute():
    _user_muted.set()    # track user intent separately from speaking-mute
    _muted.set()
    _awake.clear()
    tts.stop_speaking()
    tts.play_mute()
    dashboard.update_status(state="idle", muted=True)
    _log("info", "Muted 🔇")


def _handle_unmute():
    _user_muted.clear()
    _muted.clear()
    tts.play_unmute()
    dashboard.update_status(state="idle", muted=False)
    _log("info", "Unmuted 🎙️")
    _extend_session()


def _extend_session():
    """Open a voice session window — resets the inactivity timer and plays a wake chime."""
    global _last_speech_at
    _last_speech_at = time.time()   # reset so session doesn't time out immediately
    _awake.set()
    tts.play_beep()                 # chime: Pinku is awake and listening
    dashboard.update_status(state="awake")
    _log("wake", "Session open — listening")


# ── Gesture action map ────────────────────────────────────────────────────────
#
# Gestures detected by camera.py (OpenCV skin + convexity defects on wrist crop):
#   Thumbs Up, Thumbs Down, Open Hand, Fist, Peace
#
# _GESTURE_ACTIONS[label] = (requires_awake_session, handler_fn_name)
#   requires_awake_session=False → fires even in idle/muted state (good for wake gestures)
#   requires_awake_session=True  → only fires during an active voice session
#
# Cooldown prevents repeat-firing while hand is held steady.
_gesture_last_at: dict[str, float] = {}
_GESTURE_COOLDOWN = 8.0   # seconds between same-gesture re-fires

_GESTURE_ACTIONS: dict[str, tuple[bool, str]] = {
    "Open Hand": (False, "_gesture_open_hand"),  # 🖐 wave → wake / unmute (works anytime)
    # Fist removed — too many false positives from normal resting hand position
}


def _gesture_throttle(label: str) -> bool:
    """Return True if we should fire (not in cooldown)."""
    now = time.time()
    if now - _gesture_last_at.get(label, 0) < _GESTURE_COOLDOWN:
        return False
    _gesture_last_at[label] = now
    return True


def _gesture_open_hand():
    """🖐 Open hand / wave — wake Pinku and unmute."""
    print("[Gesture] 🖐 Open Hand → wake / unmute")
    _user_muted.clear()
    _muted.clear()
    _extend_session()


def _gesture_fist():
    """✊ Closed fist — sleep / mute."""
    print("[Gesture] ✊ Fist → sleep / mute")
    tts.stop_speaking()
    _handle_mute()


_GESTURE_FN_MAP: dict[str, object] = {
    "_gesture_open_hand": _gesture_open_hand,
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
    """Camera detection callback — auto-wakes on person presence, dispatches gestures."""
    global _last_human_at, _last_speech_at
    _det_logger.log(event)
    dashboard.push_detection(event)

    if event.get("persons", 0) > 0:
        _last_human_at = time.time()
        if not _user_muted.is_set():
            if not _awake.is_set():
                # Person walked in — open listening session, play entry chime (with cooldown)
                _last_speech_at = time.time()
                _awake.set()
                dashboard.update_status(state="awake")
                _log("info", "Person in frame — listening")
                tts.play_beep(entry=True)   # rising 3-note, suppressed if played recently
            else:
                # Already awake — keep the inactivity timer alive while person is visible
                _last_speech_at = time.time()

    if event.get("gestures"):
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
    global _last_speech_at

    while not _stop_all.is_set():
        if is_muted():
            time.sleep(0.2)
            continue

        # ── Session timeout ───────────────────────────────────────────────────
        # Only time out if the room also appears empty — person present keeps it alive
        if (_awake.is_set()
                and time.time() - _last_speech_at > config.SESSION_TIMEOUT
                and not _human_is_present()):
            _awake.clear()
            dashboard.update_status(state="idle")
            _log("info", f"Session timeout after {config.SESSION_TIMEOUT}s → idle")

        pcm = recorder.wait_for_utterance(stop_event=_stop_all, timeout=5.0)
        if pcm is None:
            continue

        # Discard audio captured while muted (mute may have been triggered mid-capture)
        if is_muted():
            continue

        try:
            text = stt.transcribe(pcm)
        except Exception as e:
            _log("error", f"STT error: {e}")
            continue
        if not text:
            continue

        _log("stt", text)

        # ── Gate: idle mode ───────────────────────────────────────────────────
        if not _awake.is_set():
            # Person in frame → no wake word needed (mirrors pinky behaviour)
            if _human_is_present() and not _user_muted.is_set():
                _awake.set()
                _last_speech_at = time.time()
                _log("info", "Person present — processing without wake word")
            else:
                triggered, command = _check_wake(text)
                if not triggered:
                    _log("info", f'Idle — no wake word in: "{text}"')
                    continue
                # Wake word confirmed
                _awake.set()
                _last_speech_at = time.time()
                _log("wake", f'Wake word detected! command="{command}"')
                if not command:
                    tts.play_beep()
                    dashboard.update_status(state="awake")
                    continue
                text = command

        # ── Active session ────────────────────────────────────────────────────
        _last_speech_at = time.time()

        # Check for session-end phrase (short utterances only)
        if _check_end(text):
            _awake.clear()
            _session_hist.clear()
            _log("info", "Session ended by user")
            tts.play_sleep()   # soft chime instead of spoken goodbye (avoids mic feedback)
            dashboard.update_status(state="idle", last_transcript=text, last_reply="")
            continue

        dashboard.update_status(state="processing", last_transcript=text)

        # ── Route & execute ───────────────────────────────────────────────────
        action = llm.route(text)
        act    = action.get("action", "chat")
        lang   = action.get("lang", "en")
        _log("route", f'action={act} lang={lang} — "{text}"')

        if act == "ignore":
            _log("info", "LLM classified as noise/ignore — skipping")
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
            _log("warn", f'Action "{act}" not yet implemented')
            tts.play_error()
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
        global _camera_enabled
        from camera import CameraDetector
        cam = CameraDetector(on_detection=on_detection)
        cam.start()
        _camera_enabled = True

    # ── Mic ───────────────────────────────────────────────────────────────────
    recorder = stt.AudioRecorder()
    recorder.start()

    dashboard.update_status(state="idle", muted=False, model=config.OLLAMA_MODEL)
    _log("info", f"Pinku ready — model={config.OLLAMA_MODEL} whisper={config.WHISPER_MODEL}")
    time.sleep(2.0)   # let mic settle before speaking so startup chime isn't heard as wake word
    tts.play_beep()   # simple chime instead of spoken "Pinku ready" (avoids mic feedback loop)

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

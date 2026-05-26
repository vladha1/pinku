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
import os
import threading
import time
import sys

# ── Single-instance lockfile ──────────────────────────────────────────────────
# Prevents two copies of pinku.py running at the same time (e.g. from repeated
# `bash start_pinku.command` calls before the old instance fully exits).
_LOCKFILE = "/tmp/pinku.lock"

def _acquire_lock():
    import fcntl
    _lock_fh = open(_LOCKFILE, "w")
    try:
        fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Read the stale PID so we can report it
        try:
            pid = open(_LOCKFILE).read().strip()
        except Exception:
            pid = "?"
        print(f"[Pinku] Another instance is already running (PID {pid}). Exiting.")
        sys.exit(1)
    _lock_fh.write(str(os.getpid()))
    _lock_fh.flush()
    return _lock_fh   # keep open — released automatically when process exits

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
_processing    = threading.Lock()    # held while handling an utterance end-to-end
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

def _brief_mute(seconds: float = 1.5):
    """Mute briefly after actions that don't speak — prevents re-capturing
    the same utterance or ambient echo immediately after processing."""
    if _user_muted.is_set():
        return
    _muted.set()
    def _clear():
        time.sleep(seconds)
        if not _user_muted.is_set():
            _muted.clear()
    threading.Thread(target=_clear, daemon=True, name="brief-mute").start()


def _speak_reply(reply: str, is_hi: bool):
    global _last_speech_at
    # Hard guard: if already speaking, discard this reply rather than overlap
    if tts.is_speaking():
        _log("info", "Already speaking — discarded duplicate reply")
        return
    _log("pinku", reply)
    dashboard.update_status(speaking=True)
    _muted.set()                          # silence mic while speaking to prevent feedback

    # Release the processing lock now — _muted + tts.is_speaking() guard from here.
    # This lets the voice loop accept the next question as soon as speech ends + settles,
    # without waiting for the full TTS playback to finish first.
    try:
        _processing.release()
    except RuntimeError:
        pass   # wasn't held (called outside the voice loop)

    # Ask TTS for the known duration before playback starts (say: word-count math;
    # edge-tts: afinfo on the generated MP3 file). Use it to schedule unmute precisely.
    known_duration = tts.known_duration(reply)
    settle          = max(1.5, min(known_duration * 0.25, 3.5))
    print(f"[TTS] known={known_duration:.1f}s  settle={settle:.1f}s  total_mute={known_duration+settle:.1f}s")

    # Start the unmute timer NOW, before playback begins.
    # It sleeps for known_duration + settle so it fires right as the echo fades.
    # The voice loop sees is_muted()=True and stays in its 0.2s sleep the whole
    # time — wait_for_utterance is never called while speech+echo are in the air.
    def _unmute_after():
        time.sleep(known_duration + settle)
        if not _user_muted.is_set():
            _muted.clear()
    threading.Thread(target=_unmute_after, daemon=True, name="tts-settle").start()

    tts.speak(reply, prefer_hi=is_hi, block=True)   # blocks for actual playback
    _last_speech_at = time.time()
    dashboard.update_status(speaking=False, state="awake" if _awake.is_set() else "idle")

    # Safety: if the timer fired early (estimate was short), _muted is already
    # cleared while echo is still in the air. Re-mute for a fresh settle period.
    if not _muted.is_set() and not _user_muted.is_set():
        print(f"[TTS] timer fired early — re-muting for {settle:.1f}s settle")
        _muted.set()
        def _re_settle():
            time.sleep(settle)
            if not _user_muted.is_set():
                _muted.clear()
        threading.Thread(target=_re_settle, daemon=True, name="tts-resiltle").start()


def _handle_chat(action: dict):
    tr    = action.get("transcript", "")
    lang  = action.get("lang", "en")
    is_hi = lang == "hi"
    _log("source", f"Gemini text ({llm.GEMINI_MODEL})")
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
    if lang == "hi":
        reply = f"अभी {now.strftime('%-I बजकर %M मिनट')} हैं।"
    else:
        reply = f"It's {now.strftime('%-I:%M %p')}."
    _log("source", "system clock")
    dashboard.update_status(last_transcript=action.get("transcript",""), last_reply=reply)
    _speak_reply(reply, lang == "hi")


def _handle_weather(action: dict):
    """Fetch weather from wttr.in — no API key needed."""
    import urllib.request as _ur
    lang = action.get("lang", "en")
    tr   = action.get("transcript", "")

    # Extract city from transcript — take the last capitalised word sequence
    # e.g. "how is the weather in Kolkata today" → "Kolkata"
    import re as _re2
    m = _re2.search(r'\bin\s+([A-Z][a-zA-Z\s]+?)(?:\s+today|\s+tomorrow|\?|$)', tr)
    city = m.group(1).strip() if m else "Mumbai"

    _log("source", f"wttr.in ({city})")
    try:
        url = f"https://wttr.in/{city.replace(' ', '+')}?format=3"
        with _ur.urlopen(url, timeout=8) as r:
            raw = r.read().decode().strip()
        # raw looks like: "Kolkata: ⛅️  +32°C"
        if lang == "hi":
            reply = f"{city} का मौसम: {raw.split(':', 1)[-1].strip()}"
        else:
            reply = raw
    except Exception as e:
        _log("warn", f"Weather fetch failed: {e}")
        reply = "Sorry, I couldn't get the weather right now."

    dashboard.update_status(last_transcript=tr, last_reply=reply)
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
    tts.speak("Let me look…")
    b64  = frame_to_b64(frame)
    desc = llm.describe_image(b64, question=tr, is_hi=(lang == "hi"))
    print(f"[Vision] {desc!r}")
    dashboard.update_status(last_transcript=tr, last_reply=desc)
    _speak_reply(desc, lang == "hi")


def _handle_scripture(action: dict):
    """Route all knowledge topics (scripture, yoga, history, music, etc.) via knowledge.py."""
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
    """🖐 Open hand / wave — stop speech if speaking, otherwise wake Pinku."""
    if tts.is_speaking():
        _log("wake", "🖐 Open Hand → stop speaking")
        tts.stop_speaking()
        _muted.clear()   # re-open mic so user can follow up immediately
        _user_muted.clear()
        _extend_session()
    else:
        _log("wake", "🖐 Open Hand gesture → wake")
        _user_muted.clear()
        _muted.clear()
        _extend_session()


def _gesture_fist():
    """✊ Closed fist — sleep / mute."""
    _log("wake", "✊ Fist gesture → sleep")
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
        now = time.time()
        was_absent = (now - _last_human_at) > 30.0   # truly left the room vs session timeout
        _last_human_at = now
        if not _user_muted.is_set():
            if not _awake.is_set():
                # Person detected — silently re-open session
                _last_speech_at = now
                _awake.set()
                dashboard.update_status(state="awake")
                _log("wake", "👤 Face detected → listening")
                # Only chime if person was genuinely absent (>30s) — not on every
                # session timeout re-trigger while they're still sitting in the room
                if was_absent:
                    tts.play_beep(entry=True)
            else:
                # Already awake — keep the inactivity timer alive while person is visible
                _last_speech_at = now

    if event.get("gestures"):
        _dispatch_gestures(event)


# ── Wake word / session end detection ────────────────────────────────────────

# Wake phrases — must appear at the START of the utterance.
# Sorted longest-first so "hey pinku" matches before bare "pinku".
_WAKE_PHRASES = [
    "hey pinku", "hi pinku", "ok pinku", "okay pinku",
    "hello pinku", "yo pinku", "pinku",
    "hey pinky", "hi pinky", "ok pinky", "okay pinky",
    "hello pinky", "yo pinky", "pinky",
    # Common Whisper mishearings of "Pinku"/"Pinky":
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
    r'(pinku|pinky|pinko|pinco|pingo|pingu|pinkoo|penku|penko|pink|पिंकू|पिंकु|पिंको|पिंकी)\b[,\s।]*',
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


# ── Action dispatcher (shared by Gemini and fallback paths) ──────────────────

def _dispatch_action(action: dict):
    """Execute a routed action dict — used by both Gemini and fallback paths."""
    act  = action.get("action", "chat")
    lang = action.get("lang", "en")
    if act == "ignore":
        _log("info", "Classified as noise/ignore — skipping")
        _brief_mute(1.0)   # brief pause so same audio isn't re-captured
        return
    elif act == "mute":
        _handle_mute()
    elif act == "unmute":
        _handle_unmute()
    elif act == "time":
        _handle_time(action)
    elif act == "describe":
        _handle_describe(action)
    elif act == "scripture":
        tts.play_think()
        _handle_scripture(action)
    elif act == "weather":
        _handle_weather(action)
    elif act in ("music_play", "music_stop", "lights_on", "lights_off"):
        _log("warn", f'Action "{act}" not yet implemented')
        tts.speak(f"Sorry, {act.replace('_', ' ')} isn't set up yet.")
    else:
        _handle_chat(action)


def _handle_gemini_result(result: dict):
    """
    Process the JSON dict returned by llm.transcribe_and_respond().
    Gemini has already transcribed + classified + (optionally) generated the reply.
    """
    global _last_speech_at

    transcript = result.get("transcript", "").strip()
    action     = result.get("action", "ignore")
    lang       = result.get("lang", "en")
    is_hi      = lang == "hi"

    if not transcript:
        return

    # ── Hallucination guard ───────────────────────────────────────────────────
    # If the transcript is only the wake word (nothing actionable after it),
    # treat as ignore — Gemini sometimes hallucinates "Pinku" from background noise.
    _WAKE_WORDS = {"pinku", "pinky", "pinko", "pink", "pingu", "pinkoo"}
    _transcript_words = [w.strip(".,!?।").lower() for w in transcript.split()]
    _content_words = [w for w in _transcript_words if w not in _WAKE_WORDS]
    if not _content_words:
        _log("info", f'Hallucinated wake word only: "{transcript}" — ignored')
        _brief_mute(1.0)
        return

    if action == "ignore":
        _log("info", f'Ignored: "{transcript}"')
        _brief_mute(1.0)
        return

    # Actions that Python handles — discard any Gemini-supplied reply (it ignores our
    # system prompt instruction to leave reply:"" and generates text anyway).
    _PYTHON_ACTIONS = {
        "time", "weather", "mute", "unmute", "describe",
        "music_play", "music_stop", "lights_on", "lights_off",
    }
    # weather is handled by Python (_handle_weather fetches from wttr.in)
    # so force reply="" to always go through dispatch, not Gemini's reply
    if action in _PYTHON_ACTIONS:
        reply = ""
    else:
        reply = result.get("reply", "").strip()

    # Bare wake word with no command — beep and wait
    # (happens when face is visible and person just says "Pinku" / "Pinky")
    if not reply and action not in _PYTHON_ACTIONS:
        _, command = _check_wake(transcript)
        if not command:
            _log("wake", f'🔤 Wake word heard — waiting for command')
            _awake.set()
            _last_speech_at = time.time()
            tts.play_beep()
            dashboard.update_status(state="awake")
            return

    _last_speech_at = time.time()

    # Session-end check (short utterances only)
    if _check_end(transcript) and _awake.is_set():
        _awake.clear()
        _session_hist.clear()
        _log("info", "Session ended by user")
        tts.play_sleep()
        dashboard.update_status(state="idle", last_transcript=transcript, last_reply="")
        return

    dashboard.update_status(state="processing", last_transcript=transcript)
    _log("user", transcript)

    if reply:
        # Gemini already generated the reply (chat / scripture)
        _log("source", f"Gemini audio ({llm.GEMINI_MODEL})")
        _session_hist.append({"role": "user",      "content": transcript})
        _session_hist.append({"role": "assistant", "content": reply})
        if len(_session_hist) > 12:
            _session_hist[:] = _session_hist[-12:]
        dashboard.update_status(last_transcript=transcript, last_reply=reply)
        _speak_reply(reply, is_hi)
    else:
        # Action needs Python handler (time, mute, describe, etc.)
        _dispatch_action({"action": action, "lang": lang,
                          "transcript": transcript, **result})


def _fallback_process(pcm: bytes):
    """
    Old pipeline: Whisper → Ollama route → handler.
    Used when Gemini audio transcription is unavailable or fails.
    """
    try:
        text = stt.transcribe(pcm)
    except Exception as e:
        _log("error", f"STT fallback error: {e}")
        return
    if not text:
        return
    _log("user",   text)
    _log("source", f"Whisper + Ollama fallback")
    action = llm.route(text)
    _log("info",   f"route={action.get('action')} lang={action.get('lang','en')}")
    _dispatch_action(action)


# ── Main voice loop ───────────────────────────────────────────────────────────

def _voice_loop(recorder: stt.AudioRecorder):
    """
    When face visible or session active → Gemini handles audio end-to-end
      (transcription + intent classification + reply in one API call).
    When idle with no face → Whisper locally for wake word detection only,
      then Gemini for the actual response.

    Fallback to Whisper + Ollama + Gemini if Gemini audio is unavailable.
    """
    global _last_speech_at

    while not _stop_all.is_set():
        if is_muted() or tts.is_speaking():
            time.sleep(0.2)
            continue

        # ── Session timeout ───────────────────────────────────────────────────
        if (_awake.is_set()
                and time.time() - _last_speech_at > config.SESSION_TIMEOUT
                and not _human_is_present()):
            _awake.clear()
            dashboard.update_status(state="idle")
            _log("info", f"Session timeout after {config.SESSION_TIMEOUT}s → idle")

        pcm = recorder.wait_for_utterance(stop_event=_stop_all, timeout=5.0)
        if pcm is None:
            continue

        if is_muted() or tts.is_speaking():
            continue

        # ── Processing lock: drop audio if already handling a previous utterance ─
        if not _processing.acquire(blocking=False):
            continue   # silently discard — previous response still in flight

        try:
            # ── Gate: Gemini path (face visible or session open) ──────────────
            if _human_is_present() or _awake.is_set():
                dashboard.update_status(state="processing")
                tts.play_think()
                result = llm.transcribe_and_respond(pcm, history=_session_hist)
                if result is not None:
                    _handle_gemini_result(result)
                else:
                    # Gemini unavailable → fall back to Whisper + Ollama
                    _log("warn", "Gemini audio unavailable — using Whisper fallback")
                    _fallback_process(pcm)
                continue

            # ── Gate: idle mode — Whisper locally for wake word only ─────────
            try:
                text = stt.transcribe(pcm)
            except Exception as e:
                _log("error", f"STT error: {e}")
                continue
            if not text:
                continue

            _log("stt", text)
            triggered, command = _check_wake(text)
            if not triggered:
                _log("info", f'Idle — no wake word: "{text}"')
                continue

            # Wake word confirmed
            _awake.set()
            _last_speech_at = time.time()
            _log("wake", f'🔤 Wake word → "{text.split()[0]}"')

            if not command:
                # Just the wake word — beep and wait for next utterance
                tts.play_beep()
                dashboard.update_status(state="awake")
                _brief_mute(1.2)   # prevent re-capturing the beep or same audio
                continue

            # Wake word + inline command — send audio to Gemini for quality response
            dashboard.update_status(state="processing")
            tts.play_think()
            result = llm.transcribe_and_respond(pcm, history=_session_hist)
            if result is not None:
                _handle_gemini_result(result)
            else:
                # Gemini unavailable → route the Whisper text we already have
                _fallback_process(pcm)

        finally:
            # _speak_reply releases early; for actions that don't speak
            # (ignore, mute, time dispatch, etc.) release it here.
            try:
                _processing.release()
            except RuntimeError:
                pass   # already released by _speak_reply


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    _acquire_lock()   # exit immediately if another instance is running
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

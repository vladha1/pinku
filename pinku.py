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
_settle_until:   float = 0.0           # voice loop blocked until this time (TTS + echo settle)

# Consecutive Gemini wake-word-only / hallucination counter.
# Gemini sometimes hallucinates "Pinku." from background noise.  Each such result
# increments this counter; a real command (with content words) resets it.
# When it reaches _HALLUC_MAX the session is forcibly closed so the voice loop
# drops back to cheap local Whisper wake-word detection instead of hammering
# the Gemini audio API every ~3 s.
_gemini_halluc_count: int = 0
_HALLUC_MAX:          int = 4   # close session after this many consecutive non-commands

# Cooldown for repeated identical (or near-identical) Pinku replies.
# Prevents the same personality trigger (e.g. "nahi nahayegi") from firing
# twice in quick succession when the TTS echo is picked up and re-processed.
_last_reply_text:  str   = ""
_last_reply_at:    float = 0.0
_REPLY_COOLDOWN:   float = 10.0   # seconds before the same reply can be spoken again

# Transcript-level deduplication — catches the double-processing bug where
# two audio captures of the same utterance are both queued (e.g., double
# face-detection created two slightly-offset audio chunks of the same speech).
# The output-side reply cooldown doesn't help when Gemini gives the same question
# different language responses ("The square root..." vs "नमस्ते! 36 का वर्गमूल…").
_last_user_transcript: str   = ""
_last_transcript_at:   float = 0.0
_TRANSCRIPT_DEDUP_SEC: float = 4.0   # seconds — same transcript in this window = duplicate

# When hallucinations force-close the session, face-detection cannot reopen it
# until this epoch — prevents the open→hallucinate×4→close→reopen cycle that
# burned through the Gemini daily quota.  Only a spoken wake word (Whisper) or
# explicit gesture can reopen the session during the cooldown.
_face_wake_blocked_until: float = 0.0
_FACE_WAKE_BLOCK_SEC:     float = 60.0   # seconds to block face-detection wake after hallucination flush

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
    global _last_speech_at, _settle_until, _last_reply_text, _last_reply_at
    # If user explicitly muted, discard reply entirely — don't speak even if
    # Gemini finished processing after the button was pressed.
    if _user_muted.is_set():
        _log("info", "User muted — discarding reply")
        return
    # Hard guard: if already speaking, discard this reply rather than overlap
    if tts.is_speaking():
        _log("info", "Already speaking — discarded duplicate reply")
        return
    # Cooldown: don't repeat the same short reply within _REPLY_COOLDOWN seconds.
    # Catches "nahi nahayegi" echoing back through the mic and re-triggering
    # the same personality response — especially when echo comes back in a
    # different script (Roman TTS → Devanagari transcription) so text matching fails.
    reply_norm = reply.strip().lower()
    if (reply_norm == _last_reply_text
            and time.time() - _last_reply_at < _REPLY_COOLDOWN):
        _log("info", f"Duplicate reply suppressed (cooldown): {reply!r}")
        return
    _last_reply_text = reply_norm
    _last_reply_at   = time.time()
    _log("pinku", reply)
    stt.register_pinku_speech(reply)   # echo filter: reject this text if mic picks it up
    dashboard.update_status(speaking=True)
    _muted.set()

    # Conservative upper-bound for _settle_until (in case TTS runs long).
    # After tts.speak() returns we re-anchor to actual end-of-speech so the
    # user only waits 'settle' seconds after the last word — not up to
    # (known_duration + settle) seconds from when TTS started.
    # Edge-tts can speak noticeably faster than the 95 WPM word-count estimate,
    # so without re-anchoring the mic would be dead for several extra seconds.
    known_duration = tts.known_duration(reply)
    # Minimum 4.0 s settle — gives room echo time to die down before the mic
    # reopens.  3 s was not enough for longer responses: a 10-second reply at
    # 0.15 coefficient only gave 3 s, but the reverb tail persisted longer.
    # 0.25 coefficient + 4-6 s range adds ~2 s of safety margin on long replies.
    settle          = max(4.0, min(known_duration * 0.25, 6.0))   # 4.0 – 6.0 s
    _settle_until   = time.time() + known_duration + settle        # upper bound

    print(f"[TTS] known={known_duration:.1f}s  settle={settle:.1f}s")

    # NOTE: _processing is intentionally NOT released here.
    # tts.speak(block=True) blocks the voice loop thread anyway, so releasing
    # early would only allow a second queued clip to slip in BEFORE the settle
    # window is enforced — causing double-processing.  The finally block in
    # _voice_loop releases the lock after _speak_reply returns.
    tts.speak(reply, prefer_hi=is_hi, block=True)
    _last_speech_at = time.time()
    dashboard.update_status(speaking=False, state="awake" if _awake.is_set() else "idle")

    # TTS is done. Re-anchor settle window to actual end of speech.
    # _muted stays set during settle; the voice loop won't call wait_for_utterance
    # until it clears.  wait_for_utterance drains the buffer on entry, so any
    # audio captured during TTS+settle is discarded before new listening begins.
    if not _user_muted.is_set():
        _settle_until = time.time() + settle
        _muted.set()   # keep muted during settle (may already be set)
        def _clear_after_settle():
            time.sleep(settle)
            if not _user_muted.is_set():
                _muted.clear()
        threading.Thread(target=_clear_after_settle, daemon=True, name="tts-settle").start()


def _handle_chat(action: dict):
    tr    = action.get("transcript", "")
    lang  = action.get("lang", "en")
    is_hi = lang == "hi"
    _log("source", f"Gemini + Google Search ({llm.GEMINI_MODEL})")
    reply = llm.chat(tr, history=_session_hist, is_hi=is_hi)
    _session_hist.append({"role": "user",      "content": tr})
    _session_hist.append({"role": "assistant", "content": reply})
    if len(_session_hist) > 12:
        _session_hist[:] = _session_hist[-12:]
    dashboard.update_status(last_transcript=tr, last_reply=reply)
    dashboard.record_conversation(tr, reply, lang=lang,
                                  source=f"Gemini + Google Search ({llm.GEMINI_MODEL})")
    _speak_reply(reply, is_hi)


def _handle_time(action: dict):
    from datetime import datetime
    import calendar as _cal
    import re as _re3
    now  = datetime.now()
    lang = action.get("lang", "en")
    tr   = action.get("transcript", "").lower()

    # Detect what was actually asked — time, date, day, month, year, or days-in-month
    _ask_days_count = bool(_re3.search(
        r'kitne\s*din|how\s*many\s*days?|days?\s*in|din\s*hain|din\s*hai|'
        r'महीने\s*में\s*कितने|कितने\s*दिन', tr))
    _ask_month  = bool(_re3.search(r'month|mahina|महीना|माह', tr))
    _ask_day    = bool(_re3.search(r'\bday\b|din\b|वार|दिन|weekday|week', tr))
    _ask_date   = bool(_re3.search(r'\bdate\b|tarikh|तारीख', tr))
    _ask_year   = bool(_re3.search(r'year|saal|साल|वर्ष', tr))
    _ask_time   = bool(_re3.search(r'\btime\b|baj|waqt|समय|बज|टाइम', tr))
    _ask_any_date = _ask_month or _ask_day or _ask_date or _ask_year or _ask_days_count
    _ask_only_time = _ask_time and not _ask_any_date

    # Python-computed calendar facts (always accurate)
    days_in_month = _cal.monthrange(now.year, now.month)[1]

    # Hindi month names
    _HI_MONTHS = ["जनवरी","फ़रवरी","मार्च","अप्रैल","मई","जून",
                  "जुलाई","अगस्त","सितंबर","अक्टूबर","नवंबर","दिसंबर"]
    _HI_DAYS   = ["सोमवार","मंगलवार","बुधवार","गुरुवार","शुक्रवार","शनिवार","रविवार"]

    if lang == "hi":
        time_str  = now.strftime('%-I बजकर %M मिनट')
        month_str = _HI_MONTHS[now.month - 1]
        day_str   = _HI_DAYS[now.weekday()]
        date_str  = f"{now.day} {month_str}"
        year_str  = str(now.year)

        if _ask_days_count:
            reply = f"{month_str} में {days_in_month} दिन हैं।"
        elif _ask_only_time:
            reply = f"अभी {time_str} हैं।"
        elif _ask_month and not _ask_time and not _ask_day and not _ask_date:
            reply = f"अभी {month_str} का महीना चल रहा है।"
        elif _ask_year and not _ask_time:
            reply = f"अभी {year_str} साल चल रहा है।"
        elif _ask_day and not _ask_time:
            reply = f"आज {day_str} है।"
        elif _ask_date and not _ask_time:
            reply = f"आज {date_str} है।"
        else:
            reply = f"आज {day_str}, {date_str} {year_str} है। अभी {time_str} हैं।"
    else:
        if _ask_days_count:
            reply = f"{now.strftime('%B')} has {days_in_month} days."
        elif _ask_only_time:
            reply = f"It's {now.strftime('%-I:%M %p')}."
        elif _ask_month and not _ask_time and not _ask_day and not _ask_date:
            reply = f"It's {now.strftime('%B')}."
        elif _ask_year and not _ask_time:
            reply = f"It's {now.year}."
        elif _ask_day and not _ask_time:
            reply = f"Today is {now.strftime('%A')}."
        elif _ask_date and not _ask_time:
            reply = f"Today is {now.strftime('%A, %d %B')}."
        else:
            reply = f"Today is {now.strftime('%A, %d %B %Y')}. It's {now.strftime('%-I:%M %p')}."

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
    city = m.group(1).strip() if m else "Gurugram"

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
    tr   = action.get("transcript", "What do you see?")
    lang = action.get("lang", "en")
    is_hi = lang == "hi"
    try:
        from camera import get_frame, frame_to_b64
        frame = get_frame()
    except Exception:
        frame = None
    if frame is None:
        reply = "कैमरा अभी उपलब्ध नहीं है।" if is_hi else "Camera isn't available right now."
        dashboard.update_status(last_transcript=tr, last_reply=reply)
        _speak_reply(reply, is_hi)
        return
    tts.speak("Let me look…" if not is_hi else "देखती हूँ…")
    b64  = frame_to_b64(frame)
    desc = llm.describe_image(b64, question=tr, is_hi=is_hi)
    print(f"[Vision] {desc!r}")
    # Gracefully handle vision model failures
    if desc == "__vision_error__" or (desc.startswith("[") and "error" in desc.lower()):
        desc = "माफ करना, अभी देख नहीं पा रही।" if is_hi else "Sorry, I couldn't see anything right now."
    dashboard.update_status(last_transcript=tr, last_reply=desc)
    _speak_reply(desc, is_hi)


def _handle_scripture(action: dict):
    """Route all knowledge topics (scripture, yoga, history, music, etc.) via knowledge.py."""
    knowledge.handle(
        action,
        speak_fn     = _speak_reply,
        update_fn    = dashboard.update_status,
        session_hist = _session_hist,
        log_fn       = _log,
    )



def _handle_sleep():
    """End the active session and return to standby / sleep.

    Pinku stays LISTENING in the background — she will auto-wake on face
    detection or wake word.  Use this to end a conversation without switching
    off the microphone entirely.  Different from mute: she can still wake up
    on her own.
    """
    global _last_speech_at
    tts.stop_speaking()
    _awake.clear()
    _session_hist.clear()
    _last_speech_at = 0.0
    tts.play_sleep()
    dashboard.update_status(state="idle")
    _log("info", "Sleeping 💤 — will wake on face or 'Pinku'")


def _handle_mute():
    """Hard mute — mic is fully disabled, no listening at all.

    Use when TV is playing loudly, people are sleeping, or you want Pinku
    completely silent.  She will NOT auto-wake on face detection or wake word
    while muted — only an explicit gesture/button unmutes her.
    """
    global _settle_until
    _user_muted.set()    # hard-mute flag checked in voice loop
    _muted.set()
    _awake.clear()
    _session_hist.clear()
    _settle_until = 0.0   # cancel any pending settle window
    tts.stop_speaking()
    tts.set_silent(True)  # suppress all tones while muted
    tts.play_mute()       # transition tone is still OK
    dashboard.update_status(state="idle", muted=True)
    _log("info", "Muted 🔇 — hard off, no listening")


def _handle_unmute():
    """Unmute from hard-mute state and open a fresh session."""
    global _settle_until
    tts.set_silent(False)  # re-enable tones before playing wake sound
    _user_muted.clear()
    _muted.clear()
    tts.play_unmute()
    dashboard.update_status(state="idle", muted=False)
    _log("info", "Unmuted 🎙️")
    _extend_session()
    # Settle: block the voice loop while the unmute chime + beep echo dies down.
    # 2.5s was not enough — the beep plays for ~0.5s, then room echo persists
    # for another 1-2s; calibration started while echo was still audible and
    # Whisper hallucinated "Pinku." from the tail.  4s gives full clearance.
    _settle_until = time.time() + 4.0
    _brief_mute(4.0)


def _handle_pause():
    """⏸ button handler.

    In active session  → sleep (end conversation, stay in standby).
    In standby/sleep   → no-op (already resting).
    Hard muted         → unmute back to standby.
    """
    if _user_muted.is_set():
        _handle_unmute()   # hard-mute → back to standby
    elif _awake.is_set() or tts.is_speaking():
        tts.stop_speaking()
        _handle_sleep()    # end active conversation
    # else: already sleeping — nothing to do


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
_gesture_throttle_lock = threading.Lock()   # makes read-check-write atomic
_GESTURE_COOLDOWN = 8.0   # seconds between same-gesture re-fires

# Lock that makes the face-detection wake block atomic.
# The camera thread fires on_detection at frame-rate (≥15 fps); without a lock
# two rapid frames both pass "if not _awake.is_set()" before either sets it,
# causing duplicate wake logs, duplicate beeps, and duplicate Gemini calls.
_face_wake_lock = threading.Lock()

_GESTURE_ACTIONS: dict[str, tuple[bool, str]] = {
    # requires_session=False → fires even from muted/idle state
    "Hands Up":  (False, "_gesture_hands_up"),   # 🙌 arm raised → unmute + wake
    "Open Hand": (False, "_gesture_open_hand"),  # 🖐 wave → mute toggle (any state)
    # Fist removed — too many false positives from normal resting hand
}


def _gesture_throttle(label: str) -> bool:
    """Return True if we should fire (not in cooldown).
    Lock makes the read-check-write atomic so simultaneous on_detection
    calls from the camera thread can't both slip past the cooldown."""
    with _gesture_throttle_lock:
        now = time.time()
        if now - _gesture_last_at.get(label, 0) < _GESTURE_COOLDOWN:
            return False
        _gesture_last_at[label] = now
        return True


def _gesture_hands_up():
    """🙌 Hands up (wrist above shoulder) — wake from muted state.
    This is the ONLY gesture that wakes Pinku when user-muted."""
    _log("wake", "🙌 Hands Up → unmute + wake")
    _handle_unmute()   # clears silent flag, plays unmute tone, opens session


def _gesture_open_hand():
    """🖐 Open hand / wave.
    Hard-muted  → unmute back to standby.
    In session  → sleep (end conversation, stay listening).
    Sleeping    → hard mute (next wave unmutes)."""
    if _user_muted.is_set():
        _log("wake", "🖐 Open Hand → unmute")
        _handle_unmute()
    elif _awake.is_set() or tts.is_speaking():
        tts.stop_speaking()
        _log("wake", "🖐 Open Hand → sleep")
        _handle_sleep()
    else:
        _log("wake", "🖐 Open Hand → mute")
        _handle_mute()


def _gesture_fist():
    """✊ Closed fist — sleep (end session, stay in standby)."""
    _log("wake", "✊ Fist → sleep")
    _handle_sleep()


_GESTURE_FN_MAP: dict[str, object] = {
    "_gesture_hands_up":  _gesture_hands_up,
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


# ── Laser dot handler ────────────────────────────────────────────────────────

import math as _math
import json as _json

_laser_was_present: bool = False   # True while a dot is on the wall
_dart_hits: list[dict] = []        # [{x, y, score, callout}] — last 20 throws

# ── Dart game state ───────────────────────────────────────────────────────────
_GAME_SHOTS = 5                    # shots per turn
_game_shots:   list[dict] = []     # current player's shots this turn
_game_rounds:  list[dict] = []     # completed turns [{player, shots, total}]
_game_player:  int  = 1            # current player number (1-based)
_game_awaiting: bool = False       # True after 5 shots — waiting for "Next Player"

# ── Laser calibration ─────────────────────────────────────────────────────────
# Bullseye centre in normalised frame coords (0-1).
# Loaded from ~/.pinku_laser_cal.json; defaults to frame-centre (0.5, 0.5).
_laser_bull_x: float = 0.5
_laser_bull_y: float = 0.5
_LASER_CAL_FILE = os.path.expanduser("~/.pinku_laser_cal.json")


def _load_laser_cal():
    """Load saved bullseye calibration from disk (called at startup)."""
    global _laser_bull_x, _laser_bull_y
    try:
        with open(_LASER_CAL_FILE) as f:
            cal = _json.load(f)
        _laser_bull_x = float(cal.get("bull_x", 0.5))
        _laser_bull_y = float(cal.get("bull_y", 0.5))
    except FileNotFoundError:
        pass   # first run — use defaults
    except Exception as e:
        _log("warn", f"Laser cal load failed: {e}")


def _save_laser_cal():
    """Persist current bullseye to disk."""
    try:
        with open(_LASER_CAL_FILE, "w") as f:
            _json.dump({"bull_x": _laser_bull_x, "bull_y": _laser_bull_y}, f)
    except Exception as e:
        _log("warn", f"Laser cal save failed: {e}")


def _dart_score(x: float, y: float) -> tuple[int, str]:
    """
    Map a normalised (x, y) position to a dartboard score using the calibrated
    bullseye centre.  100 = bullseye, decreasing rings outward, 0 = off-board.
    Returns (score, callout_text).
    """
    dx = (x - _laser_bull_x) * 2
    dy = (y - _laser_bull_y) * 2
    dist = _math.sqrt(dx*dx + dy*dy) / _math.sqrt(2)   # 0 = bull, 1 = corner

    if   dist < 0.035: return 100, "Bullseye! One hundred!"
    elif dist < 0.09:  return  75, "Seventy five!"
    elif dist < 0.16:  return  50, "Fifty!"
    elif dist < 0.24:  return  25, "Twenty five."
    elif dist < 0.35:  return  10, "Ten."
    else:              return   0, "Miss."


_LASER_MUTE_DOTS    = 10     # need this many simultaneous dots to trigger mute toggle
_LASER_MUTE_COOLDOWN = 5.0  # seconds — ignore repeated multi-dot events
_laser_mute_last_at: float = 0.0


def _handle_laser(dots: list[dict]):
    """
    0 dots   → mark absent (next appearance will rescore).
    1–9 dots → dart scoring only (single dot = throw; few dots = reflections, ignored).
    10+ dots → mute / unmute toggle, with 5-second cooldown to avoid accidental re-fires.
    """
    global _laser_was_present, _laser_mute_last_at

    # ── Dot gone — reset so next throw rescores ───────────────────────────────
    if not dots:
        if _laser_was_present:
            _log("laser", "🔴 Laser gone — ready for next throw")
            dashboard.update_status(laser=None)
        _laser_was_present = False
        return

    # ── Many dots: toggle mute (no score) ────────────────────────────────────
    if len(dots) >= _LASER_MUTE_DOTS:
        now = time.time()
        if now - _laser_mute_last_at < _LASER_MUTE_COOLDOWN:
            _laser_was_present = True
            return   # within cooldown — ignore
        _laser_mute_last_at = now
        _log("laser", f"🔴×{len(dots)} → mute toggle")
        if _user_muted.is_set():
            _handle_unmute()
        else:
            _handle_mute()
        _laser_was_present = True
        return

    # ── Single dot ────────────────────────────────────────────────────────────
    dot = dots[0]
    x, y = dot["x"], dot["y"]
    score, callout = _dart_score(x, y)
    dashboard.update_status(laser={"x": x, "y": y, "score": score})

    if _laser_was_present:
        return   # dot still on wall, just moved — don't rescore

    # Ignore new shots while waiting for "Next Player" tap
    if _game_awaiting:
        return

    # Fresh appearance — record hit, show on screen (no auto-callout)
    _laser_was_present = True
    _dart_hits.append({"x": x, "y": y, "score": score, "callout": callout})
    if len(_dart_hits) > 20:
        _dart_hits[:] = _dart_hits[-20:]
    _log("laser", f"🎯 ({x:.2f},{y:.2f}) → {score} pts")
    dashboard.push_dart_hit({"x": x, "y": y, "score": score, "callout": callout})

    # ── Game tracking ─────────────────────────────────────────────────────────
    _game_shots.append({"x": x, "y": y, "score": score})
    _push_game_state()
    if len(_game_shots) >= _GAME_SHOTS:
        _end_dart_turn()


def _push_game_state():
    dashboard.push_game_update({
        "player":     _game_player,
        "shots_done": len(_game_shots),
        "shots_total": _GAME_SHOTS,
        "turn_total": sum(s["score"] for s in _game_shots),
        "awaiting":   _game_awaiting,
        "rounds":     _game_rounds,
    })


def _end_dart_turn():
    global _game_awaiting
    total = sum(s["score"] for s in _game_shots)
    _game_rounds.append({
        "player": _game_player,
        "shots":  list(_game_shots),
        "total":  total,
    })
    _game_awaiting = True
    _log("laser", f"🎯 P{_game_player} turn done: {len(_game_shots)} shots → {total} pts")
    _push_game_state()


def _dart_next_player():
    global _game_player, _game_awaiting
    # If turn ended manually before 5 shots, record it first
    if not _game_awaiting and _game_shots:
        _end_dart_turn()
    _game_shots.clear()
    _dart_hits.clear()             # clear so reconnecting clients see only new turn
    _game_player += 1
    _game_awaiting = False
    dashboard.push_dart_hits_clear()   # wipe dots on overlay for new turn
    _push_game_state()
    _log("laser", f"🎯 P{_game_player}'s turn — ready")


def _dart_new_game():
    global _game_player, _game_awaiting
    _game_shots.clear()
    _game_rounds.clear()
    _game_player  = 1
    _game_awaiting = False
    _dart_reset()   # also clears _dart_hits + broadcasts dart_reset
    _push_game_state()
    _log("laser", "🎯 New dart game — P1's turn")


def _dart_speak_last():
    """Speak the most recent dart score aloud — triggered by dashboard button."""
    if not _dart_hits:
        threading.Thread(target=tts.speak,
            kwargs={"text": "No dart hits recorded yet.", "block": False},
            daemon=True, name="dart-speak").start()
        return
    hit = _dart_hits[-1]
    threading.Thread(target=tts.speak,
        kwargs={"text": hit["callout"], "block": False},
        daemon=True, name="dart-speak").start()


def _dart_reset():
    """Clear all recorded dart hits — triggered by dashboard button."""
    _dart_hits.clear()
    dashboard.dart_reset()
    _log("laser", "🎯 Dart hits cleared")


def _handle_laser_calibrate():
    """
    Set the scoring bullseye to wherever the laser dot currently is.
    Called from dashboard 'laser_calibrate' action or /api/laser/calibrate.
    Returns a dict so the REST endpoint can relay the result to the browser.
    """
    global _laser_bull_x, _laser_bull_y
    try:
        from camera import _laser_overlay_dots, _laser_overlay_lock
        with _laser_overlay_lock:
            dots = list(_laser_overlay_dots)
    except Exception as e:
        _log("warn", f"Laser calibrate: camera error — {e}")
        return {"ok": False, "error": str(e)}

    if not dots:
        _log("warn", "🎯 Calibrate failed — no dot detected (point laser at bullseye first)")
        return {"ok": False, "error": "No laser dot detected — point your laser at the bullseye and try again."}

    dot = dots[0]
    _laser_bull_x = dot["x"]
    _laser_bull_y = dot["y"]
    _save_laser_cal()

    # Update camera overlay so the bullseye ring redraws immediately
    try:
        from camera import set_laser_bull
        set_laser_bull(_laser_bull_x, _laser_bull_y)
    except Exception:
        pass

    _log("laser", f"🎯 Bullseye set → ({_laser_bull_x:.3f},{_laser_bull_y:.3f})")
    if not _user_muted.is_set():
        threading.Thread(
            target=tts.speak,
            kwargs={"text": "Bullseye calibrated.", "block": False},
            daemon=True, name="laser-cal",
        ).start()
    return {"ok": True, "bull_x": _laser_bull_x, "bull_y": _laser_bull_y}


# ── Detection callback ────────────────────────────────────────────────────────

def on_detection(event: dict):
    """Camera detection callback — auto-wakes on person presence, dispatches gestures."""
    global _last_human_at, _last_speech_at
    _det_logger.log(event)
    dashboard.push_detection(event)

    if event.get("persons", 0) > 0:
        now = time.time()
        _last_human_at = now
        if not _user_muted.is_set():
            # Lock makes the check-and-set atomic: camera fires on_detection at
            # frame-rate so two consecutive frames can both see _awake==False
            # and both trigger the wake logic before either sets _awake.
            with _face_wake_lock:
                if not _awake.is_set():
                    if time.time() < _face_wake_blocked_until:
                        # Session was recently closed due to hallucinations.
                        # Don't reopen from face-detection alone — require an
                        # explicit spoken wake word or gesture to break the cycle.
                        pass
                    else:
                        # Person detected — gentle ack chime + open session
                        # NOTE: chime is temporary (entry=True applies 25s cooldown)
                        _last_speech_at = now
                        _awake.set()
                        dashboard.update_status(state="awake")
                        _log("wake", "👤 Face detected → listening")
                        tts.play_beep(entry=True)
                else:
                    # Already awake — keep the inactivity timer alive while person is visible
                    _last_speech_at = now

    if "laser" in event:   # always call — empty list = dot gone
        _handle_laser(event["laser"])

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
    "hey pintu", "hi pintu", "pintu",
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
_PINKU_RE_START = _re.compile(
    r'^[.\s]*'                                                      # strip leading dots/spaces
    r'(?:(?:hey|hi|ok|okay|hello|yo|अरे|हे|आई|ए|अरी)[,.\s]+)?'   # optional Hindi/English prefix
    r'(pinku|pinky|pinko|pinco|pingo|pingu|pinkoo|penku|penko|pintu|pink'
    r'|पिंकू|पिंकु|पिंको|पिंकी'
    r'|पिकु|पिकू)\b[,\s।]*',                                       # पिकु = Whisper drops anusvara ं
    _re.IGNORECASE,
)
# Wake word anywhere in the utterance — "what's the time Pinky?" / "क्या हाल है पिकु?"
_PINKU_RE_ANY = _re.compile(
    r'\b(pinku|pinky|pinko|pinco|pingo|pingu|pinkoo|penku|penko|pintu|pink'
    r'|पिंकू|पिंकु|पिंको|पिंकी'
    r'|पिकु|पिकू)\b[\W]*$',
    _re.IGNORECASE,
)

def _check_wake(text: str) -> tuple[bool, str]:
    """
    Returns (triggered, command).
    Matches wake word at start OR end of utterance.
    e.g. "Pinky what's the time" and "what's the time Pinky" both work.
    """
    t = text.strip()
    # Strip leading filler
    for filler in ("um ", "uh ", "so ", "like ", "well ", "okay so "):
        if t.lower().startswith(filler):
            t = t[len(filler):]

    # Wake word at START — "Pinky, what's the time?"
    m = _PINKU_RE_START.match(t)
    if m:
        rest = t[m.end():].strip(" ,.")
        return True, rest

    # Exact phrase fallback (start)
    tl = t.lower()
    for phrase in _WAKE_PHRASES:
        if tl.startswith(phrase):
            rest = t[len(phrase):].strip(" ,.")
            return True, rest

    # Wake word at END — "what's the time Pinky?"
    m = _PINKU_RE_ANY.search(t)
    if m:
        command = t[:m.start()].strip(" ,.")
        return True, command

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
        _handle_scripture(action)
    elif act == "weather":
        _handle_weather(action)
    elif act in ("lights_on", "lights_off"):
        _log("warn", f'Action "{act}" not yet implemented')
        tts.speak(f"Sorry, lights control isn't set up yet.")
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
    # Correct Gemini language misclassification: if transcript has no Devanagari
    # characters, it can't be Hindi — force English.
    if lang == "hi" and not _re.search(r'[ऀ-ॿ]', transcript):
        lang = "en"
        # If Gemini also generated its reply in Hindi, discard it so the action
        # falls through to _dispatch_action → _handle_chat which will regenerate
        # in English.  Keeping a Hindi reply and just changing the voice would
        # speak Hindi words through the English TTS voice — sounds wrong.
        _reply_raw = result.get("reply", "")
        if _reply_raw and _re.search(r'[ऀ-ॿ]', _reply_raw):
            _log("info", "Gemini replied in Hindi to an English question — discarding reply, regenerating")
            result["reply"] = ""
    is_hi      = lang == "hi"

    if not transcript:
        return

    # ── Echo filter: Gemini path ──────────────────────────────────────────────
    # stt._is_pinku_echo() runs inside stt.transcribe() for the Whisper path,
    # but Gemini receives raw PCM and transcribes independently — the filter
    # was never applied to Gemini results.  Check here before doing anything.
    if stt._is_pinku_echo(transcript):
        _log("info", f'Gemini: echo of own TTS — ignored: "{transcript[:60]}"')
        _brief_mute(2.0)
        return

    # ── Transcript dedup — same utterance queued twice ────────────────────────
    # Root cause: double face-detection (two rapid frames both see _awake=False
    # before the lock fires) queues two slightly-offset audio captures of the
    # same speech.  The _face_wake_lock prevents future occurrences, but audio
    # already in the queue before the lock can still race through.  An identical
    # transcript within _TRANSCRIPT_DEDUP_SEC is almost certainly the same
    # utterance — skip the second to prevent duplicate responses.
    # NOTE: does NOT update _last_user_transcript for wake-word-only transcripts
    # (those have no content words); the hallucination guard below handles them.
    global _last_user_transcript, _last_transcript_at
    _t_key = transcript.strip().lower()
    _t_now = time.time()
    _t_words = [w.strip(".,!?।").lower() for w in transcript.split()]
    _WAKE_SET = {"pinku", "pinky", "pinko", "pink", "pingu", "pinkoo"}
    _has_content = any(w for w in _t_words if w not in _WAKE_SET)
    if _has_content:
        if _t_key == _last_user_transcript and _t_now - _last_transcript_at < _TRANSCRIPT_DEDUP_SEC:
            _log("info",
                 f"Duplicate transcript suppressed "
                 f"({_t_now - _last_transcript_at:.1f}s ago): \"{transcript[:60]}\"")
            _brief_mute(1.0)
            return
        # Record this transcript so a repeat within the window is caught.
        _last_user_transcript = _t_key
        _last_transcript_at   = _t_now

    # ── Hallucination guard ───────────────────────────────────────────────────
    # If the transcript is only the wake word (nothing actionable after it),
    # treat as ignore — Gemini sometimes hallucinates "Pinku" from background noise.
    global _gemini_halluc_count
    _WAKE_WORDS = {"pinku", "pinky", "pinko", "pink", "pingu", "pinkoo"}
    _transcript_words = [w.strip(".,!?।").lower() for w in transcript.split()]
    _content_words = [w for w in _transcript_words if w not in _WAKE_WORDS]
    if not _content_words:
        _gemini_halluc_count += 1
        if _gemini_halluc_count >= _HALLUC_MAX:
            # Too many consecutive hallucinations — Gemini is seeing background
            # noise and inventing wake words.  Close the session so the loop
            # falls back to cheap local Whisper wake-word detection instead of
            # hammering the Gemini audio API.
            global _face_wake_blocked_until
            _log("info",
                 f"Gemini hallucinated wake word {_gemini_halluc_count}× in a row "
                 f"— closing session, blocking face-wake for {_FACE_WAKE_BLOCK_SEC:.0f}s")
            _gemini_halluc_count = 0
            _awake.clear()
            _session_hist.clear()
            _face_wake_blocked_until = time.time() + _FACE_WAKE_BLOCK_SEC
            dashboard.update_status(state="idle")
            _brief_mute(6.0)   # long pause before resuming wake-word listening
            return
        if _human_is_present() or _awake.is_set():
            # Person is present — possible intentional wake-word call.
            # IMPORTANT: do NOT update _last_speech_at here.  If this is a
            # hallucination, updating the timer would keep the session open
            # forever, causing a perpetual Gemini-hammering feedback loop.
            _log("wake",
                 f'🔤 Wake word ({_gemini_halluc_count}/{_HALLUC_MAX}) — waiting for command')
            _awake.set()
            dashboard.update_status(state="awake")
            _brief_mute(3.0)   # was 0.8 s — longer throttle limits Gemini call rate
            return
        _log("info", f'Hallucinated wake word only: "{transcript}" — ignored')
        _brief_mute(3.0)   # was 1.0 s
        return

    if action == "ignore":
        _log("info", f'Ignored: "{transcript}"')
        _brief_mute(1.0)
        return

    # Actions that Python handles — discard any Gemini-supplied reply (it ignores our
    # system prompt instruction to leave reply:"" and generates text anyway).
    _PYTHON_ACTIONS = {
        "time", "weather", "mute", "unmute", "describe",
        "lights_on", "lights_off",
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
            # No beep — silently open session. Beep fires on face-detection entry
            # or intentional gesture only. Random wake-word hallucinations would
            # otherwise cause spurious chimes.
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

    # ── Confirmed real command — reset hallucination counter, play think tick ──
    # Fires only here, after all ignore/hallucination/wake-word-only paths have
    # returned early. This means beep = "I heard a real command, working on it."
    _gemini_halluc_count = 0   # real speech confirmed — clear consecutive-hallucination tally
    tts.play_think()

    dashboard.update_status(state="processing", last_transcript=transcript)
    _log("user", transcript)

    if reply:
        # Gemini already generated the reply (chat / scripture)
        src = f"Gemini audio ({llm.GEMINI_MODEL})"
        _log("source", src)
        _session_hist.append({"role": "user",      "content": transcript})
        _session_hist.append({"role": "assistant", "content": reply})
        if len(_session_hist) > 12:
            _session_hist[:] = _session_hist[-12:]
        dashboard.update_status(last_transcript=transcript, last_reply=reply)
        dashboard.record_conversation(transcript, reply, lang=lang, source=src)
        _speak_reply(reply, is_hi)
    else:
        # Action needs Python handler (time, mute, describe, etc.)
        _dispatch_action({"action": action, "lang": lang,
                          "transcript": transcript, **result})


def _has_repeated_phrase(text: str, n: int = 3) -> bool:
    """
    Return True if any n-gram (word sequence) appears more than once.
    Repeated phrases are a strong signal of song lyrics or TV audio —
    real speech almost never repeats 3-word chunks within one utterance.
    """
    words = text.lower().split()
    if len(words) < n * 2:
        return False
    seen: set[str] = set()
    for i in range(len(words) - n + 1):
        ng = " ".join(words[i:i + n])
        if ng in seen:
            return True
        seen.add(ng)
    return False


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

    # ── Repeated-phrase filter: song lyrics / TV audio ────────────────────────
    # Real speech never repeats a 3-word sequence in one utterance.
    # This catches "give her to me give her to me", "baby baby baby", etc.
    if _has_repeated_phrase(text):
        _log("info", f"Fallback: repeated phrase → ignoring (likely song/TV): {text!r}")
        return

    _log("user",   text)
    _log("source", f"Whisper + Ollama fallback")
    dashboard.update_status(state="processing", last_transcript=text)
    action = llm.route(text)
    act = action.get("action", "chat")

    # ── Sanity guard: Ollama 3b sometimes misroutes background speech as mute/unmute.
    # Require the actual command keyword to be present in the transcript.
    # Without this, a phrase like "they need permission here" can become "mute".
    _MUTE_KW   = {"mute", "stop", "quiet", "sleep", "shh", "chup", "band", "stop listening"}
    _UNMUTE_KW = {"unmute", "wake", "listen", "start", "pinku", "pinky"}
    if act == "mute" and not any(kw in text.lower() for kw in _MUTE_KW):
        _log("info", f"Fallback: rejected spurious mute (no keyword): {text!r}")
        action["action"] = "chat"
    elif act == "unmute" and not any(kw in text.lower() for kw in _UNMUTE_KW):
        _log("info", f"Fallback: rejected spurious unmute (no keyword): {text!r}")
        action["action"] = "chat"

    # ── Wake-word override: if Pinku is directly addressed, never ignore.
    # Ollama 3b classifies many clear commands as "ignore" when Gemini is down.
    # If the transcript contains Pinku's name, force chat so the personality
    # prompt (including "nahi nahayegi" for shower etc.) can fire.
    act = action.get("action", "chat")
    if act == "ignore" and _re.search(
            r'\b(pinku|pinky|pink|pinko|पिंकू|पिंकु|पिकु)\b', text, _re.IGNORECASE):
        _log("info", f"Fallback: overriding ignore→chat (wake word in transcript)")
        action["action"] = "chat"
        act = "chat"

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
        if tts.is_speaking():
            time.sleep(0.2)
            continue

        # ── Hard mute: mic completely off — no listening at all ──────────────
        # Use mute when TV is on loud or people are sleeping.
        # Sleep/idle (not muted) is the standby state — it still listens.
        if _user_muted.is_set():
            time.sleep(0.5)
            continue

        # ── Brief/automatic mute (TTS settle, post-action pause) — skip ──────
        if _muted.is_set():
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

        # ── Settle guard ─────────────────────────────────────────────────────────
        # _settle_until is set in _speak_reply to now + known_duration + settle.
        # This blocks new processing for the full TTS+echo window regardless of
        # whether the mute timer or the processing lock fires early.
        if time.time() < _settle_until:
            continue

        # ── Processing lock: drop audio if already handling a previous utterance ─
        if not _processing.acquire(blocking=False):
            continue   # silently discard — previous response still in flight

        try:
            # ── Gate: Gemini path (face visible or session open) ──────────────
            if _human_is_present() or _awake.is_set():
                # session_active=True → no wake word required; person is already
                # in conversation (face detected or awake session open).
                dashboard.update_status(state="processing")
                result = llm.transcribe_and_respond(
                    pcm, history=_session_hist, session_active=True)
                if result is not None:
                    _handle_gemini_result(result)
                else:
                    # Gemini unavailable or rate-guard skipped — fall back to Whisper + Ollama.
                    # Real errors are already logged in llm.py; don't double-log here.
                    _fallback_process(pcm)
                continue

            # ── Gate: idle mode — Whisper locally for wake word only ─────────
            dur = len(pcm) / (16000 * 2)
            if dur > 6.0:
                print(f"[STT] Idle — skipped long audio ({dur:.1f}s)")
                continue
            try:
                text = stt.transcribe(pcm)
            except Exception as e:
                _log("error", f"STT error: {e}")
                continue
            if not text:
                print(f"[STT] Idle — no speech ({dur:.1f}s)")
                continue

            _log("stt", text)
            triggered, command = _check_wake(text)
            if not triggered:
                print(f'[STT] Idle — no wake word: "{text}"')
                continue

            # Wake word confirmed — update question cloud immediately
            if command:
                dashboard.update_status(last_transcript=command)
            _awake.set()
            _last_speech_at = time.time()
            _log("wake", f'🔤 Wake word → "{text.split()[0]}"')

            if not command:
                # Wake word only — play beep so user knows Pinku heard them.
                tts.play_beep()
                dashboard.update_status(state="awake")
                _brief_mute(1.5)   # was 0.6s — longer pause prevents double-fire & beep echo re-capture
                continue

            # Wake word + inline command — send audio to Gemini for quality response
            dashboard.update_status(state="processing")
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
        dashboard.register_action("wake",             _extend_session)
        dashboard.register_action("sleep",            _handle_sleep)
        dashboard.register_action("mute",             _handle_mute)
        dashboard.register_action("unmute",           _handle_unmute)
        dashboard.register_action("pause",            _handle_pause)
        dashboard.register_action("stop",             _handle_pause)   # alias
        dashboard.register_action("mute_toggle",      lambda: _handle_unmute() if _user_muted.is_set() else _handle_mute())
        dashboard.register_action("laser_calibrate",  _handle_laser_calibrate)
        dashboard.register_action("dart_speak",       _dart_speak_last)
        dashboard.register_action("dart_reset",       _dart_reset)
        dashboard.register_action("dart_next_player", _dart_next_player)
        dashboard.register_action("dart_new_game",    _dart_new_game)

    # ── Laser calibration ─────────────────────────────────────────────────────
    _load_laser_cal()

    # ── Camera ────────────────────────────────────────────────────────────────
    cam = None
    if not args.no_camera:
        global _camera_enabled
        from camera import CameraDetector, set_laser_bull
        cam = CameraDetector(on_detection=on_detection)
        cam.start()
        _camera_enabled = True
        # Push calibrated bull to camera overlay renderer
        set_laser_bull(_laser_bull_x, _laser_bull_y)

    # ── Mic ───────────────────────────────────────────────────────────────────
    # Boost macOS input volume to maximum so distant speech reaches the mic pipeline
    try:
        import subprocess as _sp
        _sp.run(["osascript", "-e", "set volume input volume 100"],
                capture_output=True, timeout=3)
        print("[MIC] Input volume set to 100")
    except Exception as _e:
        print(f"[MIC] Could not set input volume: {_e}")
    stt.set_log_callback(_log)   # route STT diagnostics → dashboard log
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

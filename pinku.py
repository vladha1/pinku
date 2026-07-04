#!/usr/bin/env python3
"""
Pinku — local AI home assistant for M4 Mac Mini.

Stack:
  STT  : faster-whisper (local, MPS-accelerated on Apple Silicon)
  LLM  : Ollama (local, llama3.2:3b default)
  TTS  : macOS `say` (built-in neural voices)

Usage:
  python pinku.py
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
        try:
            pid = open(_LOCKFILE).read().strip()
        except Exception:
            pid = "?"
        print(f"[Pinku] Another instance is already running (PID {pid}). Exiting.")
        sys.exit(2)   # exit code 2 = lockfile conflict; start_pinku.command waits longer before retry
    _lock_fh.write(str(os.getpid()))
    _lock_fh.flush()
    return _lock_fh   # keep open — released automatically when process exits

import signal

import config
import stt
import tts
import llm
import knowledge
import dashboard
import speaker_id
import apple_stt
import mlx_stt
import math_handler


# ── SIGTERM handler — clean exit when pkill/kill sends SIGTERM ────────────────
# Without this, Python's default SIGTERM kills the process but bypasses the
# finally block in _voice_loop, leaving threads mid-flight.  Raising
# KeyboardInterrupt re-uses the existing clean-shutdown path in main().
def _handle_sigterm(signum, frame):
    print("[Pinku] Received SIGTERM — shutting down cleanly …")
    raise KeyboardInterrupt

signal.signal(signal.SIGTERM, _handle_sigterm)


# ── Global state ──────────────────────────────────────────────────────────────

_muted         = threading.Event()   # set = muted (mic disabled)
_user_muted    = threading.Event()   # set = user manually muted (gesture/button)
_awake         = threading.Event()   # set = in active voice session
_stop_all      = threading.Event()
_processing    = threading.Lock()    # held while handling an utterance end-to-end
_session_hist: list[dict] = []       # [{role, content}] for LLM context

_last_speech_at:     float = time.time()   # reset on every utterance / wake / gesture
_settle_until:       float = 0.0           # voice loop blocked until this time (TTS + echo settle)
_current_speaker:    str | None = None    # name of identified speaker for this utterance
_current_spk_score:  float = 0.0          # cosine similarity score for _current_speaker
_last_gate_at:       float = 0.0           # last time a PCM chunk passed the speaker gate
_last_pinku_spoke_at: float = 0.0         # when Pinku last finished speaking (post-reply window)

# Consecutive Gemini wake-word-only / hallucination counter.
# Gemini sometimes hallucinates "Pinku." from background noise.  Each such result
# increments this counter; a real command (with content words) resets it.
# When it reaches _HALLUC_MAX the session is forcibly closed.
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

def is_muted() -> bool:
    return _muted.is_set()


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


def _speak_reply(reply: str, is_hi: bool, _t_utterance: float = 0.0,
                 _t_spk: float = 0.0, _t_llm: float = 0.0):
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
    # Echo-settle: short replies use TTS_SETTLE_MIN (default 1.5 s); longer
    # replies scale at 0.25× duration so a 10 s reply gets ~2.5 s settle.
    # Cap at 6 s — more than enough for any room echo tail.
    settle          = max(config.TTS_SETTLE_MIN, min(known_duration * 0.25, 6.0))
    _settle_until   = time.time() + known_duration + settle        # upper bound

    print(f"[TTS] known={known_duration:.1f}s  settle={settle:.1f}s")
    if _t_utterance:
        _t_tts = time.time()
        print(
            f"[LATENCY] spk={_t_spk-_t_utterance:.2f}s"
            f"  gemini={_t_llm-_t_spk:.2f}s"
            f"  dispatch={_t_tts-_t_llm:.2f}s"
            f"  → total={_t_tts-_t_utterance:.2f}s  (utterance→first word)"
        )

    # NOTE: _processing is intentionally NOT released here.
    # tts.speak(block=True) blocks the voice loop thread anyway, so releasing
    # early would only allow a second queued clip to slip in BEFORE the settle
    # window is enforced — causing double-processing.  The finally block in
    # _voice_loop releases the lock after _speak_reply returns.
    actual_duration = tts.speak(reply, prefer_hi=is_hi, block=True)
    _last_speech_at = time.time()
    global _last_pinku_spoke_at
    _last_pinku_spoke_at = time.time()   # opens post-reply conversational window
    dashboard.update_status(speaking=False, state="awake" if _awake.is_set() else "idle")

    # Recalculate settle from actual measured duration (edge-tts returns exact
    # duration via afinfo; much more accurate than the 95 WPM pre-estimate).
    if actual_duration > 0:
        settle = max(config.TTS_SETTLE_MIN, min(actual_duration * 0.25, 6.0))
        print(f"[TTS] actual={actual_duration:.1f}s  settle={settle:.1f}s")

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
    spk_ctx = f"The person speaking is {_current_speaker}." if _current_speaker else ""
    reply = llm.chat(tr, history=_session_hist, is_hi=is_hi, system_extra=spk_ctx)
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
    tr = action.get("transcript", "")
    dashboard.update_status(last_transcript=tr, last_reply=reply)
    dashboard.record_conversation(tr, reply, lang=lang, source="system clock")
    _speak_reply(reply, lang == "hi")


def _handle_weather(action: dict):
    """Fetch weather from wttr.in — no API key needed."""
    import urllib.request as _ur, json as _json
    lang = action.get("lang", "en")
    tr   = action.get("transcript", "")

    import re as _re2
    # English: "weather in Kolkata today"
    m = _re2.search(r'\bin\s+([A-Z][a-zA-Z\s]+?)(?:\s+today|\s+tomorrow|\?|$)', tr)
    # Hindi transliteration: "Gurugram ka mausam" / "Delhi ki weather"
    if not m:
        m = _re2.search(r'([A-Z][a-zA-Z]+)\s+(?:ka|ki|ke)\s+(?:mausam|weather|mosam)', tr, _re2.IGNORECASE)
    city = m.group(1).strip() if m else "Gurugram"

    _log("source", f"wttr.in ({city})")
    try:
        url = f"https://wttr.in/{city.replace(' ', '+')}?format=j1"
        with _ur.urlopen(url, timeout=8) as r:
            data = _json.loads(r.read().decode())
        curr  = data["current_condition"][0]
        temp  = curr["temp_C"]
        desc  = curr["weatherDesc"][0]["value"]
        today = data["weather"][0]
        hi, lo = today["maxtempC"], today["mintempC"]
        if lang == "hi":
            reply = f"{city} का मौसम: {desc}, {temp}°C. आज High {hi}°C, Low {lo}°C."
        else:
            reply = f"{city}: {desc}, {temp}°C. Today High {hi}°C / Low {lo}°C."
    except Exception as e:
        _log("warn", f"Weather fetch failed: {e}")
        reply = "Sorry, I couldn't get the weather right now."

    dashboard.update_status(last_transcript=tr, last_reply=reply)
    dashboard.record_conversation(tr, reply, lang=lang, source=f"wttr.in ({city})")
    _speak_reply(reply, lang == "hi")


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


# ── Wake word / session end detection ────────────────────────────────────────

# Wake phrases — must appear at the START of the utterance.
# Sorted longest-first so "hey pinku" matches before bare "pinku".
_WAKE_PHRASES = [
    "hey pinku", "hi pinku", "ok pinku", "okay pinku",
    "hello pinku", "yo pinku", "pinku",
    "hey pinky", "hi pinky", "ok pinky", "okay pinky",
    "hello pinky", "yo pinky", "pinky",
    "hey pinkie", "hi pinkie", "pinkie",          # alternate spelling
    "hey pinki", "hi pinki", "pinki",             # Indian English spelling
    # Common Whisper mishearings of "Pinku"/"Pinky":
    "hey pinko", "hi pinko", "ok pinko", "hello pinko", "pinko",
    "hey pinkku", "hi pinkku", "pinkku",          # doubled k (Pinku)
    "hey pinkky", "hi pinkky", "pinkky",          # doubled k (Pinky)
    "hey pinkhu", "hi pinkhu", "pinkhu",          # inserted h (Pinku)
    "hey pinkhy", "hi pinkhy", "pinkhy",          # inserted h (Pinky)
    "hey pink", "hi pink", "ok pink", "pink",
    "hey piku", "hi piku", "ok piku", "piku",     # Pinku with dropped n
    "hey pico", "hi pico", "pico",               # Whisper English mishearing
    "hey pinsky", "hi pinsky", "pinsky",         # Pinky with s inserted
    "hey minkoo", "hi minkoo", "minkoo",         # m instead of p mishearing
    "hey minku", "hi minku", "minku",
    "hey minky", "hi minky", "minky",
    "hey piky", "hi piky", "piky",                # Pinky with dropped n
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
    r'(?:(?:hey|hi|ok|okay|hello|yo|अरे|हे|हाई|है|आई|ए|अरी|ओए|नमस्ते|हलो|हैलो)[,.\s]+)?'   # optional Hindi/English prefix
    r'(pinku|pinky|pinki|pinkie|pinko|pinco|pingo|pingu|pinkoo|pinkku|pinkky|pinkhu|pinkhy|penku|penko|pintu|pink|piku|piky'
    r'|पिंकू|पिंकु|पिंकि|पिंको|पिंकी|पिंक्व'
    r'|पिकु|पिकू|पिखु|पिखू|पिके|पिंका|पिका|पेंकू)(?!\w)[,\s।]*',   # Whisper mishearings
    _re.IGNORECASE,
)
# Wake word anywhere in the utterance — "what's the time Pinky?" / "क्या हाल है पिकु?"
_PINKU_RE_ANY = _re.compile(
    r'\b(pinku|pinky|pinki|pinkie|pinko|pinco|pingo|pingu|pinkoo|pinkku|pinkky|pinkhu|pinkhy|penku|penko|pintu|pink|piku|piky'
    r'|पिंकू|पिंकु|पिंकि|पिंको|पिंकी|पिंक्व'
    r'|पिकु|पिकू|पिखु|पिखू|पिके|पिंका|पिका|पेंकू)(?!\w)[\W]*$',
    _re.IGNORECASE,
)
# Pinky variant anywhere in text — used for Gemini fallback when local wake parse fails
# Note: use (?!\w) not \b at end — Devanagari vowel signs (ि ी ु ू ो) are \W so \b
# never fires after them (both sides of boundary are \W → no boundary).
_PINKU_RE_CONTAINS = _re.compile(
    r'\b(pinku|pinky|pinki|pinkie|pinko|pinco|pingo|pingu|pinkoo|pinkku|pinkky|pinkhu|pinkhy|penku|penko|pintu|pink|piku|piky'
    r'|पिंकू|पिंकु|पिंकि|पिंको|पिंकी|पिंक्व'
    r'|पिकु|पिकू|पिखु|पिखू|पिके|पिंका|पिका|पेंकू)(?!\w)',
    _re.IGNORECASE,
)

# "आई" = transliterated "Hi" that Whisper uses; "हलो"/"हैलो" = Whisper's Devanagari for "Hello"
_HINDI_PRE = {"हाई", "है", "हे", "अरे", "ओए", "नमस्ते", "आई", "हलो", "हैलो"}
# Devanagari fuzzy: initial consonant (प/त/म/ब) + vowel (ि/ी/े) +
# optional nasal/r + k-family consonant + optional ending vowel/cluster
# Include ि (U+093F), े (U+0947), ा (U+093E) — covers पिंकि/पिके/पिंका/पिका Whisper forms
_DEVANAGARI_WAKE_RE = _re.compile(
    r'^[पतमब][िीे]ं?र?[कखगर](?:[ूुीिेोा]|्[वय])?$'
)
def _devanagari_has_wake_shape(text: str) -> bool:
    """True if any word in a Devanagari transcript fuzzy-matches the Pinky phonetic shape.

    Used as a Gemini fallback trigger for utterances like "राई पिंका कैसे हो?" where
    the explicit wake-word lists don't cover the variant but the phonetic shape is correct.
    Only fires when the transcript already contains Devanagari script.
    """
    import unicodedata as _ucd
    for w in text.split():
        wn = _ucd.normalize("NFC", w.strip(",.!?।॥ '\""))
        if _DEVANAGARI_WAKE_RE.match(wn):
            return True
    return False

# Roman fuzzy targets — edit distance ≤ 2 catches pinkku, piku, minkoo, pinsky, etc.
_FUZZY_TARGETS = ["pinku", "pinky", "pinki", "pinkoo", "penku"]
_FUZZY_PREFIXES = set("pmbt")  # consonant substitutions seen in Whisper mishearings
_FUZZY_GREETINGS = {"hi", "hey", "ok", "okay", "hello", "yo"}


def _levenshtein(a: str, b: str) -> int:
    if len(a) > len(b):
        a, b = b, a
    row = list(range(len(a) + 1))
    for c2 in b:
        new = [row[0] + 1]
        for j, c1 in enumerate(a):
            new.append(min(new[-1] + 1, row[j + 1] + 1, row[j] + (c1 != c2)))
        row = new
    return row[-1]


def _is_fuzzy_wake(word: str) -> bool:
    """True if word is a plausible Whisper mishearing of Pinku/Pinky."""
    w = word.lower().strip(",.!?'\"")
    if not (4 <= len(w) <= 8):
        return False
    if w[0] not in _FUZZY_PREFIXES:
        return False
    return any(_levenshtein(w, t) <= 2 for t in _FUZZY_TARGETS)


def _check_wake(text: str) -> tuple[bool, str]:
    """
    Returns (triggered, command).
    Matches wake word at start OR end of utterance.
    e.g. "Pinky what's the time" and "what's the time Pinky" both work.
    """
    import unicodedata
    t = unicodedata.normalize("NFC", text.strip())
    for filler in ("um ", "uh ", "so ", "like ", "well ", "okay so "):
        if t.lower().startswith(filler):
            t = t[len(filler):]

    words = t.split()
    words_clean = [w.strip(",.!?।॥ '\"") for w in words]

    # ── Devanagari fuzzy check ─────────────────────────────────────────────
    for i, w in enumerate(words_clean):
        wn = unicodedata.normalize("NFC", w)
        if _DEVANAGARI_WAKE_RE.match(wn):
            if i == 0 or words_clean[i - 1] in _HINDI_PRE:
                return True, " ".join(words_clean[i + 1:]).strip()

    # ── Roman regex (explicit variants + exact phrase list) ────────────────
    m = _PINKU_RE_START.match(t)
    if m:
        return True, t[m.end():].strip(" ,.")

    tl = t.lower()
    for phrase in _WAKE_PHRASES:
        if tl.startswith(phrase):
            after_idx = len(phrase)
            if after_idx < len(tl) and tl[after_idx].isalnum():
                continue
            return True, t[after_idx:].strip(" ,.")

    # ── Roman fuzzy check (catches unseen Whisper variants) ────────────────
    wl = [w.lower().strip(",.!?'\"") for w in words]
    for i, w in enumerate(wl):
        if _is_fuzzy_wake(w):
            if i == 0 or wl[i - 1] in _FUZZY_GREETINGS:
                return True, " ".join(words[i + 1:]).strip(" ,.")

    # ── Wake word at END — "what's the time Pinky?" ────────────────────────
    m = _PINKU_RE_ANY.search(t)
    if m:
        return True, t[:m.start()].strip(" ,.")

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
    _t_utterance = result.get("_t_utterance", 0.0)
    _t_spk       = result.get("_t_spk",       0.0)
    _t_llm       = result.get("_t_llm",       0.0)

    transcript = result.get("transcript", "").strip()
    action     = result.get("action", "ignore")
    lang       = result.get("lang", "en")

    # ── Idle-mode wake-word guard ─────────────────────────────────────────────
    # When called with session_active=False (idle / known-speaker path), Gemini
    # must have heard the wake word.  If the transcript contains no wake-word
    # variant, Gemini hallucinated a command from background audio — force ignore.
    # This catches cases like Gemini returning "chat" for garbled ambient audio
    # even though the idle prompt explicitly requires the wake word.
    _session_was_active = result.get("_session_active", True)   # injected below
    if not _session_was_active and not _awake.is_set():
        _WAKE_RE = _re.compile(
            r'\b(pinku|pinky|pinki|pinko|pink|pingu|pinkoo|penku|penko'
            r'|पिंकू|पिंकु|पिंकि|पिंको|पिंकी|पिकु|पिकू|पिखु|पिखू|पिके|पिंका|पिका|पेंकू)(?!\w)',
            _re.IGNORECASE)
        if action != "ignore" and not _WAKE_RE.search(transcript):
            _log("info",
                 f'Idle hallucination suppressed — no wake word in: "{transcript[:60]}"')
            _brief_mute(1.5)
            return

    # Trust Gemini's language classification — transliterated Hindi
    # ("aath aur aath kitna hota hai") has no Devanagari but is still Hindi.
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
            _log("info",
                 f"Gemini hallucinated wake word {_gemini_halluc_count}× in a row "
                 f"— closing session")
            _gemini_halluc_count = 0
            _awake.clear()
            _session_hist.clear()
            dashboard.update_status(state="idle")
            _brief_mute(6.0)   # long pause before resuming wake-word listening
            return
        if _awake.is_set():
            # In session — possible intentional wake-word call.
            # IMPORTANT: do NOT update _last_speech_at here.  If this is a
            # hallucination, updating the timer would keep the session open
            # forever, causing a perpetual Gemini-hammering feedback loop.
            _log("wake",
                 f'🔤 Wake word ({_gemini_halluc_count}/{_HALLUC_MAX}) — waiting for command')
            _brief_mute(3.0)
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
        "time", "weather", "mute", "unmute",
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
    # Open an active session so follow-up queries use the fast Apple STT path
    # instead of going back through Gemini audio every time.
    _awake.set()

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

    dashboard.update_status(state="processing", last_transcript=transcript, speaker=_current_speaker)
    _log("user", transcript)

    if reply:
        # Gemini already generated the reply (chat / scripture)
        _stt = result.get("_stt_label")
        src = (f"{_stt} + Gemini text ({llm.GEMINI_MODEL})" if _stt
               else f"Gemini audio ({llm.GEMINI_MODEL})")
        _log("source", src)
        _session_hist.append({"role": "user",      "content": transcript})
        _session_hist.append({"role": "assistant", "content": reply})
        if len(_session_hist) > 12:
            _session_hist[:] = _session_hist[-12:]
        dashboard.update_status(last_transcript=transcript, last_reply=reply)
        dashboard.record_conversation(transcript, reply, lang=lang, source=src)
        _speak_reply(reply, is_hi,
                     _t_utterance=_t_utterance, _t_spk=_t_spk, _t_llm=_t_llm)
    else:
        # Action needs Python handler (time, mute, describe, etc.)
        # Put explicit overrides AFTER **result so they win over Gemini's original values.
        # ("lang": lang must come last — otherwise **result's lang:"hi" silently wins.)
        _dispatch_action({**result, "action": action, "lang": lang,
                          "transcript": transcript})


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
    dashboard.update_status(state="processing", last_transcript=text, speaker=_current_speaker)
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
            r'\b(pinku|pinky|pinki|pink|pinko|पिंकू|पिंकु|पिंकि|पिकु)(?!\w)', text, _re.IGNORECASE):
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
    global _last_speech_at, _current_speaker, _current_spk_score, _last_gate_at

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
                and time.time() - _last_speech_at > config.SESSION_TIMEOUT):
            _awake.clear()
            dashboard.update_status(state="idle")
            _log("info", f"Session timeout after {config.SESSION_TIMEOUT}s → idle")

        # ── Pause while dashboard Voices tab is recording enrollment audio ────
        if dashboard.is_enrolling():
            time.sleep(0.2)
            continue

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

        _t_utterance = time.time()   # ← latency clock starts when PCM is ready

        # Diagnostic: log every captured utterance so the dashboard log shows
        # whether the mic is picking up audio and the VAD is triggering.
        _dur_sec = len(pcm) / (16000 * 2)
        print(f"[STT] Captured {_dur_sec:.1f}s")

        # ── Speaker gate ──────────────────────────────────────────────────────────
        # Drop audio from unrecognised voices (TV, Pinky's TTS echo, guests) before
        # any expensive Whisper / Gemini call.  Falls back to open gate when no
        # profiles are enrolled so the system works out of the box.
        _spk_score = 1.0   # default when speaker ID is off or no profiles
        if config.SPEAKER_ID_ENABLED and speaker_id.has_profiles():
            _spk_name, _spk_score = speaker_id.identify(pcm)
            # Use a lower floor in idle mode — wake word is the real gate there,
            # so we only need to reject clearly-foreign audio (TV, echoes).
            # In an active session the tighter UNCERTAIN threshold applies.
            _spk_floor = (config.SPEAKER_THRESHOLD_UNCERTAIN if _awake.is_set()
                          else getattr(config, 'SPEAKER_THRESHOLD_IDLE', 0.50))
            if _spk_score < _spk_floor:
                _log("info", f"SpeakerID: unknown ({_spk_score:.2f}) — dropped")
                continue
            # Inter-utterance cooldown: drop tail-audio reblips arriving within 1.5s
            # of the previous utterance that passed the gate (prevents duplicate processing).
            # 1.5s is enough to catch TTS echoes (< 0.5s lag) while still letting
            # the user retry quickly if Pinku didn't hear them.
            _gate_now = time.time()
            if _gate_now - _last_gate_at < 1.5:
                continue
            _last_gate_at = _gate_now
            _current_speaker = _spk_name   # None if uncertain, name if >= ACCEPT
            _current_spk_score = _spk_score
            if _spk_name:
                _log("info", f"SpeakerID: {_spk_name} ({_spk_score:.2f})")
            else:
                _log("info", f"SpeakerID: uncertain ({_spk_score:.2f}) — passing anonymously")
        else:
            _current_speaker = None
            _current_spk_score = 0.0

        _t_spk = time.time()   # after speaker ID (or skipped)

        # ── Processing lock: drop audio if already handling a previous utterance ─
        if not _processing.acquire(blocking=False):
            continue   # silently discard — previous response still in flight

        try:
            # ── Gate: active session ─────────────────────────────────────────
            if _awake.is_set():
                # In an active session, accept anyone who passed the outer speaker
                # gate (>= SPEAKER_THRESHOLD_UNCERTAIN).  SPEAKER_THRESHOLD_ACCEPT
                # (0.80) is used only for *identifying* the speaker by name, not for gating
                # conversation.  Anyone who entered via wake word or while a known speaker
                # is active has earned their place in the session — dropping uncertain voices
                # here breaks multi-person conversations.
                if _current_speaker is None and _spk_score < config.SPEAKER_THRESHOLD_UNCERTAIN:
                    continue

                dashboard.update_status(state="processing", speaker=_current_speaker, spk_score=_current_spk_score)

                # ── Local STT: Apple STT → MLX Whisper large-v3 → Gemini text ──
                # Apple STT (~0.2s, English) first — fast and free.
                # MLX Whisper large-v3 (~1.5s, Neural Engine) for everything else.
                # Gemini text routing (~1s, cheap) uses the local transcript.
                # Gemini audio is kept as last-resort fallback only.
                _fast_transcript = apple_stt.transcribe(pcm)
                _stt_label = "AppleSTT"
                if not _fast_transcript:
                    _fast_transcript = mlx_stt.transcribe(pcm)
                    _stt_label = "MLXWhisper"

                if _fast_transcript:
                    _t_stt = time.time()
                    print(f"[{_stt_label}] {_fast_transcript!r}  ({_t_stt-_t_spk:.2f}s)")

                    # Math shortcut — no LLM call at all
                    _math_answer = math_handler.detect_and_compute(_fast_transcript)
                    if _math_answer:
                        _t_llm = time.time()
                        result = {
                            "transcript": _fast_transcript,
                            "lang": "en",
                            "action": "chat",
                            "reply": _math_answer,
                            "_session_active": True,
                            "_t_utterance": _t_utterance,
                            "_t_spk": _t_spk,
                            "_t_llm": _t_llm,
                        }
                        print(f"[MATH] {_math_answer}")
                        _handle_gemini_result(result)
                        continue

                    # Gemini text routing (transcript already known — no audio upload)
                    result = llm.text_route_and_respond(
                        _fast_transcript, history=_session_hist,
                        session_active=True, speaker=_current_speaker)
                    _t_llm = time.time()
                    if result is not None and result.get("action") != "ignore":
                        result["_session_active"] = True
                        result["_t_utterance"]    = _t_utterance
                        result["_t_spk"]          = _t_spk
                        result["_t_llm"]          = _t_llm
                        result["_stt_label"]      = _stt_label
                        _handle_gemini_result(result)
                        continue
                    if result is not None:
                        print(f"[STT] text-route ignored — falling back to Gemini audio")

                # Gemini audio fallback — when local STT failed or text-route ignored
                result = llm.transcribe_and_respond(
                    pcm, history=_session_hist, session_active=True,
                    speaker=_current_speaker)
                _t_llm = time.time()
                if result is not None:
                    result["_session_active"] = True
                    result["_t_utterance"]    = _t_utterance
                    result["_t_spk"]          = _t_spk
                    result["_t_llm"]          = _t_llm
                    _handle_gemini_result(result)
                continue

            # ── Gate: idle mode ───────────────────────────────────────────────
            dur = len(pcm) / (16000 * 2)
            if dur > 6.0:
                print(f"[STT] Idle — skipped long audio ({dur:.1f}s)")
                continue

            # Known speaker in idle mode.
            # Apple STT + math shortcut first (~0.2s) — saves a 4s Gemini round-trip
            # for simple arithmetic when the wake word is present.
            # Fall through to Gemini audio for everything else.
            if _current_speaker is not None:
                dashboard.update_status(state="processing", speaker=_current_speaker, spk_score=_current_spk_score)
                _idle_fast = apple_stt.transcribe(pcm)
                if _idle_fast:
                    _idle_triggered, _idle_cmd = _check_wake(_idle_fast)
                    if _idle_triggered and _idle_cmd:
                        _math_answer = math_handler.detect_and_compute(_idle_cmd)
                        if _math_answer:
                            _t_llm = time.time()
                            _math_result = {
                                "transcript": _idle_fast,
                                "lang": "en",
                                "action": "chat",
                                "reply": _math_answer,
                                "_session_active": False,
                                "_t_utterance": _t_utterance,
                                "_t_spk": _t_spk,
                                "_t_llm": _t_llm,
                            }
                            print(f"[MATH] {_math_answer}")
                            _handle_gemini_result(_math_result)
                            continue
                result = llm.transcribe_and_respond(
                    pcm, history=_session_hist,
                    session_active=False, speaker=_current_speaker)
                _t_llm = time.time()
                if result is not None:
                    result["_session_active"] = False
                    result["_t_utterance"]    = _t_utterance
                    result["_t_spk"]          = _t_spk
                    result["_t_llm"]          = _t_llm
                    _handle_gemini_result(result)
                else:
                    _fallback_process(pcm)
                continue

            # Uncertain/unknown speaker — Apple STT wake-word gate first.
            # Apple STT is local, ~0.2s, free.  Only send to Gemini if the
            # wake word is found, saving a call for every TV snippet or
            # background conversation that passes speaker ID.
            # If Apple STT is unavailable (returns None) fall through to Gemini.
            _unc_fast = apple_stt.transcribe(pcm)
            if _unc_fast is not None:
                if not _unc_fast:
                    continue   # silence / noise
                _unc_triggered, _unc_cmd = _check_wake(_unc_fast)
                if not _unc_triggered:
                    print(f'[STT] Idle — no wake word: "{_unc_fast}"')
                    continue

            dashboard.update_status(state="processing", speaker=_current_speaker, spk_score=_current_spk_score)
            result = llm.transcribe_and_respond(
                pcm, history=_session_hist,
                session_active=False, speaker=_current_speaker)
            _t_llm = time.time()
            if result is None:
                dashboard.update_status(state="idle")
                continue
            if result.get("action") == "ignore":
                dashboard.update_status(state="idle")
                continue
            result["_session_active"] = False
            result["_t_utterance"]    = _t_utterance
            result["_t_spk"]          = _t_spk
            result["_t_llm"]          = _t_llm
            _handle_gemini_result(result)

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
        dashboard.start(port=args.port)
        dashboard.start_stdout_tee()   # forward all print() to the log panel
        # Wire dashboard buttons → pinku actions
        dashboard.register_action("wake",             _extend_session)
        dashboard.register_action("sleep",            _handle_sleep)
        dashboard.register_action("mute",             _handle_mute)
        dashboard.register_action("unmute",           _handle_unmute)
        dashboard.register_action("pause",            _handle_pause)
        dashboard.register_action("stop",             _handle_pause)   # alias
        dashboard.register_action("mute_toggle",      lambda: _handle_unmute() if _user_muted.is_set() else _handle_mute())

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

    # ── Fast STT for active sessions ─────────────────────────────────────────
    apple_stt.request_authorization()   # no-op if permission not yet granted
    mlx_stt.preload()                   # background download + warm-up

    # ── Speaker identification ────────────────────────────────────────────────
    if config.SPEAKER_ID_ENABLED:
        n = speaker_id.load()
        if n:
            _log("info", f"SpeakerID: gate active — {n} profile(s) loaded: {', '.join(speaker_id._profiles)}")
        else:
            _log("info", "SpeakerID: no profiles enrolled — gate inactive (open to all voices)")

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


if __name__ == "__main__":
    main()

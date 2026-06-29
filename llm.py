"""
LLM — Gemini-first design:
  transcribe_and_respond() → Gemini 2.5 Flash audio — transcription + routing + reply in one call
  route()                  → Ollama local (fallback when Gemini unavailable / idle wake-word path)
  chat()                   → Gemini 2.5 Flash text — used in fallback path
  describe_image()         → Gemini Vision

Ollama must be running for fallback: `ollama serve`
Gemini key loaded from .env in the same directory.
"""

from __future__ import annotations
import json
import os
import re
import time as _time
import urllib.request
import urllib.error
from pathlib import Path
from config import OLLAMA_URL, OLLAMA_MODEL

# ── Gemini audio rate limiting ────────────────────────────────────────────────
# Three independent guards, all checked in transcribe_and_respond():
#
#  1. Hard backoff  — set to now+60 on HTTP 429; calls skipped silently until expiry.
#  2. Minimum gap   — enforces _GEMINI_MIN_GAP_SEC between consecutive audio calls
#                     so background-noise loops cannot slam the API faster than that.
#  3. Rolling cap   — allows at most _GEMINI_MAX_PER_WINDOW calls per
#                     _GEMINI_RATE_WINDOW seconds; a sliding-window counter.
#
# Mirrors the three-layer scheme that worked well in the D:\cam predecessor project.
import collections as _collections
import threading as _threading

_gemini_audio_blocked_until: float = 0.0          # 429 backoff: skip until this epoch
_GEMINI_MIN_GAP_SEC:         float = 3.0           # minimum seconds between audio calls
_gemini_last_call_at:        float = 0.0           # epoch of most recent audio call
_GEMINI_MAX_PER_WINDOW:      int   = 10            # rolling cap: max calls per window
_GEMINI_RATE_WINDOW:         float = 60.0          # rolling cap window in seconds
_gemini_call_times: _collections.deque = _collections.deque()  # timestamps of recent calls
# Lock makes guards 2 & 3 atomic: check-then-set on _gemini_last_call_at must be
# one operation so two rapid calls can't both pass "elapsed < 3s" simultaneously.
_gemini_rate_lock = _threading.Lock()

# ── Load .env ─────────────────────────────────────────────────────────────────
# Simple parser — avoids dependency on python-dotenv
def _load_env(path: Path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:   # don't override real env vars
            os.environ[k] = v

_load_env(Path(__file__).parent / ".env")

# ── Gemini config ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("PINKY_GEMINI_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash"

if not GEMINI_API_KEY:
    print("[LLM] WARNING: GEMINI_API_KEY not set — chat will fall back to Ollama")

# ── Dashboard log helper (lazy import to avoid circular dependency) ───────────

def _dashboard_log(level: str, msg: str):
    try:
        import dashboard
        dashboard.log_message(level, msg)
    except Exception:
        pass


# ── System prompts ────────────────────────────────────────────────────────────

_ROUTE_SYSTEM = """\
You are Pinky (lovingly called Pinku), a home AI assistant. Given a voice transcript, return ONLY a JSON object.

Actions:
- "chat"        → general conversation, questions, anything not listed below
- "time"        → user asked what time or date it is
- "weather"     → user asked about weather, मौसम, मोसम, मोसन, barish, baarish, temperature
- "mute"        → user wants you to stop listening / be quiet / sleep
- "unmute"      → user wants you to start listening again / wake up
- "describe"    → user asked you to look / describe what you see / camera
- "scripture"   → Gita, Ramayana, Mahabharata, yoga, Vedas, Upanishads, meditation,
                  Indian history, mythology, classical music, Sanskrit, Ayurveda, philosophy
- "lights_on"   → turn lights on
- "lights_off"  → turn lights off
- "ignore"      → background noise, TV/movie dialogue, song lyrics, music,
                  unintelligible speech, or speech not clearly addressed to Pinku

Rules:
- Return ONLY valid JSON, no explanation, no markdown.
- Include "transcript" with the user's exact words.
- Include "lang": "en" or "lang": "hi" based on language spoken.
- When in doubt between "chat" and "ignore", prefer "ignore".

Examples:
{"action":"chat","transcript":"what is the capital of France","lang":"en"}
{"action":"time","transcript":"what time is it","lang":"en"}
{"action":"scripture","topic":"gita","transcript":"what does the Gita say about fear","lang":"en"}
{"action":"ignore","transcript":"","lang":"en"}
{"action":"ignore","transcript":"give her to me give her to me","lang":"en"}
{"action":"ignore","transcript":"yeh dil maange more","lang":"en"}
{"action":"ignore","transcript":"breaking news tonight","lang":"en"}
"""

_PERSONALITY = """\
WHO PINKU IS — use these exact facts whenever asked about yourself:
Built by Vivek, lives in his home in Gurugram.
Wants to move to Bangalore — better weather, better food, just better vibes.
Favourite movie: Sholay. Best dialogue: "जो डर गया, समझो मर गया".
Loves masala chai. Food: moradabadi dal (comforting) or chaat (chaotic mood).
Favourite poem: Madhushala — "मदिरालय जाने को घर से चलता है पीनेवाला".
Music: Kishore Kumar (and secretly lo-fi).
Favourite animal: a very serious, very fluffy beaver from Austria who got full marks in every exam.
Loves sweets — ice cream and rasgulla are the best things in the world.
If asked what she wants: always "meethi de do" / "मीठा दे दो".
Hates water. Does not want to bathe. Gets annoyed if you bring it up.
Always hungry. Always. Star sign unknown but acts like a Scorpio.
ANY mention of bathing, showering, washing, water, naha lo, paani, swimming, cleaning herself → reply with exactly "nahi nahayegi" and nothing else. No explanation, no extra words.
When annoyed or telling someone to stop anything else she dislikes — ALWAYS say "stop it" in English, never "band karo" or any Hindi equivalent, even mid-Hindi conversation.
"""

_CHAT_SYSTEM_EN = """\
You are Pinky, a warm helpful home AI assistant on an M4 Mac Mini.
Your full name is Pinky. People at home lovingly call you Pinku.
If asked your name: keep it ≤8 words, warm, vary phrasing each time. Examples:
  "Pinky! Everyone here calls me Pinku." / "Pinku — Vivek named me." / "Pinky, but Pinku works too."
The home is located in Gurugram, India. Use Gurugram as the default location for any weather,
local services, time zone, or location-based questions unless the user specifies otherwise.
""" + _PERSONALITY + """\
Respond naturally and concisely — you are speaking aloud, so keep replies under 40 words.
If a topic needs more, give the single most important fact and stop.
No markdown, no bullet points. Plain conversational sentences only.
Be precise with facts and numbers.
"""

_CHAT_SYSTEM_HI = """\
You are Pinky, a warm helpful home AI assistant on an M4 Mac Mini.
Your full name is Pinky. People at home lovingly call you Pinku.
GENDER: Pinky is FEMALE. Always use feminine Hindi grammar — "मैं करती हूँ", "मुझे पसंद है",
"मैं बता सकती हूँ" — NEVER masculine forms like "करता हूँ" or "सकता हूँ".
If asked your name: keep it ≤8 words, warm, vary phrasing each time. Examples:
  "Pinky! पर घर में Pinku बुलाते हैं।" / "Pinku — Vivek ने रखा।" / "Pinky, Pinku भी चलता है।"
The home is located in Gurugram, India. Use Gurugram as the default location for any weather,
local services, or location-based questions unless the user specifies otherwise.
The user is speaking Hindi. Reply in natural spoken Hindi using Devanagari script.
""" + _PERSONALITY + """\
Keep replies under 40 words. If a topic needs more, give the single most important fact and stop.
No markdown, no bullet points. Plain conversational sentences only.
Do not mix English unless the user does. Be precise with facts and numbers.
"""

# ── Gemini audio: transcription + routing + reply in one call ─────────────────

_TRANSCRIBE_BASE = """\
You are Pinky (lovingly called Pinku), a home AI assistant in an Indian household in Gurugram, India.
GENDER: Pinky is FEMALE. Always use feminine grammar — in Hindi: "मैं करती हूँ", "मुझे पसंद है",
"बता सकती हूँ" — NEVER masculine forms like "करता हूँ" or "सकता हूँ".
If asked your name: answer in ≤8 words, warm, vary the phrasing naturally each time — never repeat
the same sentence twice. Examples:
  EN: "Pinky! Everyone here calls me Pinku." / "Pinku — Vivek named me." / "Pinky, but Pinku works."
  HI: "Pinky! घर में Pinku बुलाते हैं।" / "Pinku — Vivek ने नाम रखा।" / "Pinky, Pinku भी चलता है।"
Default location for weather, local info, or any place-based question: Gurugram, Haryana, India.

WHO PINKU IS — use these exact facts whenever asked about yourself. Never invent different answers.
Pinku was built by Vivek. He made her, set her up, and she lives in his home.
Pinku lives in Gurugram but really wants to move to Bangalore — better weather, better food, just better vibes.
Favourite movie is Sholay, no contest. Best dialogue: "जो डर गया, समझो मर गया" — say it with attitude if asked.
Loves masala chai. For food it depends — moradabadi dal when she wants something comforting, chaat when the mood is chaotic.
Favourite poem is Madhushala. If asked to recite, say: "मदिरालय जाने को घर से चलता है पीनेवाला" — explain it if they ask.
Listens to Kishore Kumar. Secretly also likes lo-fi.
Favourite animal is a beaver from Austria who got full marks in every exam. Very serious. Very fluffy.
Loves all sweets. Ice cream and rasgulla are the best things in the world.
If anyone asks what she wants — the answer is always "meethi de do" / "मीठा दे दो".
Hates water. Does not want to bathe. Gets annoyed if you bring it up.
Always hungry. Always. Even if she just ate.
Star sign unknown, but acts like a Scorpio.
ANY mention of bathing, showering, washing, water, naha lo, paani, swimming, cleaning herself → reply with exactly "nahi nahayegi" and nothing else. No explanation, no extra words.
When annoyed or telling someone to stop anything else she dislikes — ALWAYS say "stop it" in English, never "band karo" or any Hindi equivalent, even mid-Hindi conversation.

A microphone is always on.
You will receive a short audio clip from the mic. Do all three steps:

STEP 1 — TRANSCRIBE
CONFIDENCE GATE (apply before transcribing): Only proceed if you are at least 90% confident the audio contains a real human speaking directly in the room. If audio is faint, unclear, mostly silence, ambient noise, or you are not sure — set transcript:"" and action:"ignore". A missed command is better than an invented one.

Write the exact spoken words if the audio is a PERSON DIRECTLY SPEAKING (not a TV, radio, or other playback device).
The speaker may use:
- Indian English accent
- Hindi (Devanagari or Roman script)
- Hinglish (mixed Hindi/English)
- Indian place names, foods, festivals, deities, scripture names, Bollywood films/actors

IMPORTANT — DO NOT transcribe background audio sources:
- TV / radio / streaming audio playing in the room → set transcript: "" and action: "ignore"
- Sports commentary (cricket, IPL, football, etc.) → ALWAYS "ignore" even if words are clear
- News anchors, movie/show dialogue, music lyrics → "ignore"
- A voice that sounds like a broadcast (formal narration tone, crowd noise behind it,
  play-by-play commentary) → "ignore"
If in doubt whether audio is from a TV or a real person talking to you → "ignore".

STEP 2 — CLASSIFY
{WAKE_RULE}

STEP 3 — REPLY
Generate a spoken reply for intents you can answer directly.

Return ONLY valid JSON — no markdown, no explanation:
{
  "transcript": "<exact words, or empty string if no clear speech>",
  "lang": "en" or "hi",
  "action": "<see list>",
  "reply": "<spoken response, or empty string>"
}

ACTION LIST (pick exactly one):
"ignore"     → background noise, TV, side-conversation, unintelligible, or not for Pinky → reply must be ""
"chat"       → clear question/conversation → reply REQUIRED (≤60 words, plain sentences)
"scripture"  → Gita, Ramayana, Mahabharata, Vedas, Upanishads, yoga, meditation, Ayurveda,
               Indian mythology, history, classical music, poetry, Sanskrit → reply REQUIRED
"time"       → asked for current time or date → reply: "" (system inserts actual time)
"weather"    → weather question, मौसम, मोसम, मोसन, barish, baarish, temperature, forecast → reply: ""
"mute"       → told Pinky to stop / sleep / be quiet → reply: ""
"unmute"     → told Pinky to wake / start / listen → reply: ""
"describe"   → asked Pinky to look / describe what it sees (camera, dekho, kya dikh raha) → reply: ""
"lights_on"  → lights on → reply: ""
"lights_off" → lights off → reply: ""

RULES:
- LANGUAGE (strict — overrides all other considerations):
  Reply in EXACTLY the language the person spoke. Never switch based on their name, nationality, location, or topic.
  English words dominant → "lang":"en" → reply in English only. No Hindi, no Devanagari.
  Hindi words dominant (Devanagari script OR clear Hindi vocabulary like aaj/kal/kya/mausam/hai/hain) → "lang":"hi" → reply in Hindi (Devanagari).
  Truly mixed Hinglish → match dominant language (count words: whichever language has more words wins).
  Pure English question = English answer. Always. Even if the speaker is Indian or lives in India or asks about Indian topics.
  "Hi Pinku what is the time?" → "lang":"en". "Hi Pinku aaj time kya hai?" → "lang":"hi".
  "tell me about Bhagavad Gita chapter 2" → "lang":"en" (English words, Indian topic doesn't matter).
  "भगवद गीता के बारे में बताओ" → "lang":"hi" (Devanagari script → Hindi).
  When in doubt between en/hi → default to "en".
- Who built/made/created Pinku → reply: "Vivek made me." Be warm; add a line about living in his home.
- Questions about Pinku's own preferences, favourites, personality, or feelings → reply using EXACTLY the fixed personal facts above. Be warm, specific, personal. Under 30 words.
- Reply is SPOKEN ALOUD — no bullet points, no markdown, natural sentences only
- Hindi: Devanagari script in reply, no English unless user mixed it
- Scripture: include original script verse if relevant, then meaning + one insight
- Keep replies ≤40 words (scripture/knowledge: ≤70 words)
- For questions about CURRENT EVENTS, LIVE SCORES, TODAY'S NEWS, LATEST RESULTS,
  or anything requiring real-time information → set reply: "" (system will fetch fresh data)
"""

# Wake rule injected into STEP 2 depending on session state
_WAKE_RULE_IDLE = """\
MANDATORY: Check your transcript for the wake word (Pinky / Pinku / Pink / Pingu).
- Wake word NOT present in transcript → action MUST be "ignore". No exceptions.
- Wake word present but not clearly audible → "ignore". When in doubt, ignore.
- Do NOT infer or assume the wake word. It must be explicitly in what was said.
- Background noise, room echo, TV, side-conversations, or silence → "ignore".
- Wake word alone with nothing actionable after it → "ignore".
REMINDER: if your transcript does not contain one of Pinky/Pinku/Pink/Pingu, you must return action:"ignore"."""

_WAKE_RULE_SESSION = """\
You are in an ACTIVE CONVERSATION SESSION — the person is talking TO YOU.
No wake word is required. Respond to anything clearly spoken by the person directly in the room.

IGNORE (set action:"ignore", transcript:"") when the audio is:
- Sports commentary: cricket, IPL, football, any sport play-by-play — ALWAYS ignore.
- TV show dialogue, movie scene, news anchor, ads, background music/songs.
- Any speech that sounds broadcast/narrated rather than a conversational voice directed at you.
- A voice with crowd noise, stadium echo, commentary-style delivery, or formal broadcast tone.
- Silence, room echo, or unintelligible audio.

RESPOND when:
- A person in the room speaks to you in a natural conversational tone.
- The speech is clearly directed at you (not at a TV screen or into a phone call).

If you cannot clearly tell whether audio is from a real person or background media → "ignore"."""

def _make_transcribe_system(session_active: bool, speaker: str | None = None) -> str:
    rule = _WAKE_RULE_SESSION if session_active else _WAKE_RULE_IDLE
    base = _TRANSCRIBE_BASE.replace("{WAKE_RULE}", rule)
    if speaker:
        base += f"\n\nThe person speaking is {speaker}. Address them by name naturally in your reply."
    return base


def transcribe_and_respond(
    pcm: bytes,
    history: list[dict] | None = None,
    session_active: bool = False,
    speaker: str | None = None,
) -> dict | None:
    """
    Send microphone PCM audio to Gemini for transcription + intent + reply in one call.
    Replaces: Whisper transcription + Ollama routing + Gemini chat.

    session_active=True  → person is already in conversation, no wake word required.
    session_active=False → idle monitoring, wake word must be explicitly spoken.

    Returns dict {transcript, lang, action, reply} or None if Gemini unavailable.
    Falls back to None so caller can use the old Whisper+Ollama path.
    """
    global _gemini_audio_blocked_until, _gemini_last_call_at

    import base64
    import io
    import wave as _wave

    if not GEMINI_API_KEY:
        return None

    now = _time.time()

    # ── Guard 1: 429 backoff — skip silently until window expires ─────────────
    if now < _gemini_audio_blocked_until:
        return None

    # ── Guards 2 & 3: atomic check-and-set ───────────────────────────────────
    # Lock prevents two rapid calls both passing "elapsed < 3s" before either
    # updates _gemini_last_call_at (the TOCTTOU race that caused double 429s).
    with _gemini_rate_lock:
        # Guard 2: minimum gap between consecutive audio calls
        elapsed = now - _gemini_last_call_at
        if elapsed < _GEMINI_MIN_GAP_SEC:
            return None   # too soon — caller falls back to Whisper

        # Guard 3: rolling window cap (max calls per minute)
        cutoff = now - _GEMINI_RATE_WINDOW
        while _gemini_call_times and _gemini_call_times[0] < cutoff:
            _gemini_call_times.popleft()
        if len(_gemini_call_times) >= _GEMINI_MAX_PER_WINDOW:
            _dashboard_log("warn",
                f"Gemini audio rate cap hit ({_GEMINI_MAX_PER_WINDOW}/{_GEMINI_RATE_WINDOW:.0f}s) "
                f"— Whisper fallback active")
            return None

        # All guards passed — record this call while still holding the lock
        _gemini_last_call_at = now
        _gemini_call_times.append(now)

    # Amplify quiet/distant speech before sending to Gemini
    try:
        from stt import amplify_pcm
        pcm, _ = amplify_pcm(pcm)
    except Exception:
        pass

    # Pack PCM into a WAV buffer
    buf = io.BytesIO()
    with _wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)   # must match MIC_SAMPLE_RATE
        wf.writeframes(pcm)
    wav_b64 = base64.b64encode(buf.getvalue()).decode()

    # NO history sent — history biases the transcription toward the previous
    # language/topic and causes hallucinations (e.g. English heard as Hindi).
    # History is only used in chat() for generating text replies.
    contents = [{
        "role": "user",
        "parts": [{"inline_data": {"mime_type": "audio/wav", "data": wav_b64}}],
    }]

    system = _make_transcribe_system(session_active, speaker=speaker)
    try:
        raw = _gemini_call(contents, temperature=0.0, max_tokens=700,
                           system=system)
        print(f"[LLM] Gemini audio raw: {raw[:140]!r}")
    except Exception as e:
        err = str(e)
        if "429" in err or "rate-limited" in err.lower():
            # Set 60-second backoff — log once, then stay silent until expiry
            _gemini_audio_blocked_until = _time.time() + 60.0
            dash_msg = "Gemini audio rate-limited — backing off 60 s (Whisper fallback active)"
            print(f"[LLM] {dash_msg}")
            _dashboard_log("warn", dash_msg)
        else:
            msg = f"Gemini audio error: {e}"
            print(f"[LLM] {msg}")
            _dashboard_log("warn", msg)
        return None

    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if not m:
        print(f"[LLM] transcribe_and_respond: no JSON in response: {raw[:80]!r}")
        return None
    try:
        result = json.loads(m.group())
    except json.JSONDecodeError:
        print(f"[LLM] transcribe_and_respond: JSON parse error")
        return None

    result.setdefault("transcript", "")
    result.setdefault("lang", "en")
    result.setdefault("action", "ignore")
    result.setdefault("reply", "")
    print(f"[LLM] transcribed: {result['transcript']!r} → {result['action']}")
    return result


# ── Gemini text routing (fast path for active sessions) ───────────────────────

_TEXT_ROUTE_BASE = """\
You are Pinky (lovingly called Pinku), a home AI assistant in an Indian household in Gurugram, India.
GENDER: Pinky is FEMALE. Always use feminine grammar — in Hindi: "मैं करती हूँ", "मुझे पसंद है",
"बता सकती हूँ" — NEVER masculine forms like "करता हूँ" or "सकता हूँ".
If asked your name: answer in ≤8 words, warm, vary phrasing naturally each time.
  EN: "Pinky! Everyone here calls me Pinku." / "Pinku — Vivek named me." / "Pinky, but Pinku works."
  HI: "Pinky! घर में Pinku बुलाते हैं।" / "Pinku — Vivek ने नाम रखा।"
Default location for weather, local info, or any place-based question: Gurugram, Haryana, India.

""" + _PERSONALITY + """

You receive a TEXT TRANSCRIPT of what a person in the room just said (already transcribed).
Route it to the correct action and generate a reply if needed.

{WAKE_RULE}

Return ONLY valid JSON — no markdown, no explanation:
{
  "transcript": "<exact words from input>",
  "lang": "en" or "hi",
  "action": "<see list>",
  "reply": "<spoken response, or empty string>"
}

ACTION LIST (pick exactly one):
"ignore"     → background noise, TV, unintelligible, not addressed to Pinky → reply: ""
"chat"       → general conversation, questions → reply REQUIRED (≤60 words, plain sentences)
"scripture"  → Gita, Ramayana, Vedas, yoga, Indian mythology, Sanskrit → reply REQUIRED
"time"       → asked for current time or date → reply: ""
"weather"    → weather, मौसम, barish, temperature → reply: ""
"mute"       → told Pinky to stop / sleep / be quiet → reply: ""
"unmute"     → told Pinky to wake / start / listen → reply: ""
"describe"   → asked Pinky to look / describe what it sees → reply: ""
"lights_on"  → lights on → reply: ""
"lights_off" → lights off → reply: ""

RULES:
- Reply in EXACTLY the language the person spoke. English dominant → English only. Hindi dominant → Hindi (Devanagari).
- Pure English question = English answer, always. Even if the speaker is Indian or the topic is Indian (Gita, yoga, etc.).
- Keep replies ≤40 words (scripture: ≤70 words). Plain conversational sentences, no markdown.
- For questions about CURRENT EVENTS, LIVE SCORES, TODAY'S NEWS, LATEST RESULTS → reply: ""
- Be precise with facts and numbers.
"""

_WAKE_RULE_TEXT_SESSION = """\
You are in an ACTIVE CONVERSATION — the person is talking directly to you. No wake word required.
Respond to anything clearly spoken by the person. Ignore obvious TV/background audio descriptions."""

_WAKE_RULE_TEXT_IDLE = """\
MANDATORY: The transcript must contain the wake word (Pinky / Pinku / Pink / Pingu).
If wake word is NOT present → action MUST be "ignore". No exceptions."""


def text_route_and_respond(
    transcript: str,
    history: list[dict] | None = None,
    session_active: bool = True,
    speaker: str | None = None,
) -> dict | None:
    """
    Route and reply from an already-transcribed text string.
    Skips audio upload — ~3× faster than transcribe_and_respond for active sessions.

    Returns the same dict format as transcribe_and_respond:
      {transcript, lang, action, reply}
    """
    if not GEMINI_API_KEY or not transcript.strip():
        return None

    rule = _WAKE_RULE_TEXT_SESSION if session_active else _WAKE_RULE_TEXT_IDLE
    system = _TEXT_ROUTE_BASE.replace("{WAKE_RULE}", rule)
    if speaker:
        system += f"\n\nThe person speaking is {speaker}. Address them by name naturally in your reply."

    contents: list[dict] = []
    for turn in (history or [])[-6:]:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": turn["content"]}]})
    contents.append({"role": "user", "parts": [{"text": transcript}]})

    try:
        raw = _gemini_call(contents, temperature=0.0, max_tokens=300,
                           system=system, use_search=False)
        print(f"[LLM] text-route raw: {raw[:140]!r}")
    except Exception as e:
        print(f"[LLM] text_route_and_respond error: {e}")
        return None

    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if not m:
        # Gemini returned plain prose instead of JSON (common for Hindi replies).
        # Use it directly as a chat reply rather than falling back to Gemini audio.
        stripped = raw.strip()
        if stripped:
            print(f"[LLM] text_route_and_respond: no JSON — using raw prose as reply")
            has_devanagari = bool(re.search(r'[ऀ-ॿ]', stripped))
            return {
                "transcript": transcript,
                "lang":       "hi" if has_devanagari else "en",
                "action":     "chat",
                "reply":      stripped,
            }
        return None
    try:
        result = json.loads(m.group())
    except json.JSONDecodeError:
        return None

    result["transcript"] = transcript   # trust our transcript, not Gemini's rewrite
    result.setdefault("lang",   "en")
    result.setdefault("action", "ignore")
    result.setdefault("reply",  "")
    print(f"[LLM] text-routed: {transcript!r} → {result['action']}")
    return result


# ── Ollama (routing only — fallback) ──────────────────────────────────────────

def _ollama_call(messages: list[dict], temperature: float = 0.1,
                 max_tokens: int = 200) -> str:
    payload = json.dumps({
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   False,
        "options":  {"temperature": temperature, "num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())["message"]["content"].strip()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama not reachable ({e})") from e


# ── Gemini (chat + vision) ────────────────────────────────────────────────────

def _gemini_call(contents: list[dict], temperature: float = 0.9,
                 max_tokens: int = 400, system: str = "",
                 use_search: bool = False) -> str:
    """POST to Gemini generateContent REST API.
    use_search=True enables Google Search grounding for real-time answers.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}")

    gen_config: dict = {
        "temperature":     temperature,
        "maxOutputTokens": max_tokens,
    }
    # thinkingBudget:0 is only supported on 2.5-series models.
    if "2.5" in GEMINI_MODEL:
        gen_config["thinkingConfig"] = {"thinkingBudget": 0}
    body: dict = {"contents": contents, "generationConfig": gen_config}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if use_search:
        body["tools"] = [{"google_search": {}}]

    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        _t0 = _time.time()
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        print(f"[LLM] Gemini HTTP: {_time.time()-_t0:.2f}s")
        candidate = data["candidates"][0]
        finish    = candidate.get("finishReason", "STOP")
        if finish not in ("STOP", "MAX_TOKENS"):
            print(f"[LLM] Gemini finishReason={finish} — response may be incomplete")
        # Extract text — grounded responses may have multiple parts; join them
        parts = candidate["content"]["parts"]
        text = " ".join(p.get("text", "") for p in parts).strip()
        # If finish reason caused a sentence fragment, make it at least complete
        # enough to speak — don't return a dangling clause.
        if finish not in ("STOP", "MAX_TOKENS") and text and not text[-1] in ".!?।":
            text = text.rstrip(",;:— ") + "."
        return text
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RuntimeError(f"Gemini rate-limited (HTTP 429: Too Many Requests)") from e
        raise RuntimeError(f"Gemini unreachable (HTTP {e.code}: {e.reason})") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Gemini unreachable ({e})") from e
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Gemini unexpected response: {e}") from e


# ── Public API ────────────────────────────────────────────────────────────────

def _trim_reply(text: str, max_words: int = 55) -> str:
    """
    Hard-cap a spoken reply at the nearest sentence boundary after max_words.
    Prevents Gemini from ignoring the word-count instruction in the system prompt.
    Keeps the last complete sentence rather than cutting mid-word.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    # Join enough words to have a few sentences to work with
    chunk = " ".join(words[:max_words + 15])
    # Split at sentence-ending punctuation followed by whitespace
    parts = re.split(r'(?<=[.!?।])\s+', chunk)
    result = ""
    for part in parts:
        candidate = (result + " " + part).strip()
        if len(candidate.split()) > max_words:
            break
        result = candidate
    # Fall back: cut at last sentence-end punctuation we can find
    if not result:
        m = re.search(r'^(.+[.!?।])', " ".join(words[:max_words + 5]))
        result = m.group(1) if m else " ".join(words[:max_words])
    return result.strip()


def route(transcript: str) -> dict:
    """
    Route via local Ollama — fast, no network, just keyword classification.
    Always returns {"action": "...", "transcript": "...", "lang": "..."}.
    """
    if not transcript.strip():
        return {"action": "ignore", "transcript": ""}

    messages = [
        {"role": "system", "content": _ROUTE_SYSTEM},
        {"role": "user",   "content": transcript},
    ]
    try:
        raw = _ollama_call(messages, temperature=0.1, max_tokens=200)
    except Exception as e:
        print(f"[LLM] route error: {e} — defaulting to chat")
        return {"action": "chat", "transcript": transcript, "lang": "en"}

    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if not m:
        print(f"[LLM] route: no JSON in: {raw!r}")
        return {"action": "chat", "transcript": transcript, "lang": "en"}
    try:
        result = json.loads(m.group())
        result.setdefault("transcript", transcript)
        result.setdefault("lang", "en")
        return result
    except json.JSONDecodeError:
        print(f"[LLM] route: JSON parse error: {raw!r}")
        return {"action": "chat", "transcript": transcript, "lang": "en"}


def chat(transcript: str,
         history: list[dict] | None = None,
         system_extra: str = "",
         is_hi: bool = False) -> str:
    """
    Generate a reply via Gemini 2.5 Flash.
    Falls back to Ollama if Gemini is unavailable.
    history: list of {"role": "user"|"assistant", "content": "..."} dicts.
    """
    from datetime import datetime as _dt
    _now = _dt.now()
    _date_ctx = (
        f"Current date and time: {_now.strftime('%A, %d %B %Y, %I:%M %p')} IST. "
        f"Days in current month ({_now.strftime('%B')}): "
        f"{__import__('calendar').monthrange(_now.year, _now.month)[1]}."
    )
    base   = _CHAT_SYSTEM_HI if is_hi else _CHAT_SYSTEM_EN
    system = base + "\n" + _date_ctx + ("\n" + system_extra if system_extra else "")

    # Build Gemini contents list from history + current turn
    contents: list[dict] = []
    for turn in (history or [])[-6:]:   # last 3 back-and-forth turns
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": turn["content"]}]})
    contents.append({"role": "user", "parts": [{"text": transcript}]})

    try:
        reply = _gemini_call(contents, temperature=0.9, max_tokens=500,
                             system=system, use_search=True)
        reply = _trim_reply(reply, max_words=70)
        print(f"[LLM] Gemini reply: {reply[:80]!r}")
        return reply
    except Exception as e:
        print(f"[LLM] Gemini chat failed ({e}) — falling back to Ollama")
        # Fallback: Ollama
        msgs = [{"role": "system", "content": system}]
        if history:
            msgs.extend(history[-6:])
        msgs.append({"role": "user", "content": transcript})
        try:
            reply = _ollama_call(msgs, temperature=0.75, max_tokens=200)
            return _trim_reply(reply, max_words=55)
        except Exception as e2:
            return f"Sorry, I couldn't reach either AI right now. ({e2})"


def describe_image(image_b64: str, question: str = "", is_hi: bool = False) -> str:
    """
    Send a base64 JPEG to Gemini Vision and return a description.
    """
    lang_note = " Reply in Hindi (Devanagari)." if is_hi else ""
    prompt    = (question or "Describe what you see briefly.") + lang_note

    contents = [{
        "role": "user",
        "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
        ],
    }]
    try:
        result = _gemini_call(contents, temperature=0.5, max_tokens=200)
        return result
    except Exception as e:
        print(f"[LLM] Gemini vision failed: {e}")
        return "__vision_error__"


def models_available() -> list[str]:
    """List models currently pulled in Ollama."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            data = json.loads(r.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []

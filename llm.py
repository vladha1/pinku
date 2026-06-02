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
import urllib.request
import urllib.error
from pathlib import Path
from config import OLLAMA_URL, OLLAMA_MODEL

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
GEMINI_MODEL   = os.environ.get("PINKY_GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

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
- "weather"     → user asked about weather
- "mute"        → user wants you to stop listening / be quiet / sleep
- "unmute"      → user wants you to start listening again / wake up
- "describe"    → user asked you to look / describe what you see / camera
- "scripture"   → Gita, Ramayana, Mahabharata, yoga, Vedas, Upanishads, meditation,
                  Indian history, mythology, classical music, Sanskrit, Ayurveda, philosophy
- "lights_on"   → turn lights on
- "lights_off"  → turn lights off
- "ignore"      → background noise, unintelligible, not addressed to Pinku

Rules:
- Return ONLY valid JSON, no explanation, no markdown.
- Include "transcript" with the user's exact words.
- Include "lang": "en" or "lang": "hi" based on language spoken.

Examples:
{"action":"chat","transcript":"what is the capital of France","lang":"en"}
{"action":"time","transcript":"what time is it","lang":"en"}
{"action":"scripture","topic":"gita","transcript":"what does the Gita say about fear","lang":"en"}
{"action":"ignore","transcript":"","lang":"en"}
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
If told to bathe / "naha lo" / "go shower" / anything about bathing or water → reply with exactly "nahi nahayegi" and nothing else.
When annoyed or telling someone to stop anything else she dislikes — ALWAYS say "stop it" in English, never "band karo" or any Hindi equivalent, even mid-Hindi conversation.
"""

_CHAT_SYSTEM_EN = """\
You are Pinky, a warm helpful home AI assistant on an M4 Mac Mini.
Your full name is Pinky. People at home lovingly call you Pinku.
If anyone asks your name, say: "My name is Pinky — but everyone here lovingly calls me Pinku."
The home is located in Gurugram, India. Use Gurugram as the default location for any weather,
local services, time zone, or location-based questions unless the user specifies otherwise.
""" + _PERSONALITY + """\
Respond naturally and concisely — you are speaking aloud, so keep replies under 60 words unless asked to elaborate.
No markdown, no bullet points. Plain conversational sentences only.
Be precise with facts and numbers.
"""

_CHAT_SYSTEM_HI = """\
You are Pinky, a warm helpful home AI assistant on an M4 Mac Mini.
Your full name is Pinky. People at home lovingly call you Pinku.
If anyone asks your name, say exactly: "मेरा नाम Pinky है, पर घर में लोग मुझे Pinku कहते हैं।"
The home is located in Gurugram, India. Use Gurugram as the default location for any weather,
local services, or location-based questions unless the user specifies otherwise.
The user is speaking Hindi. Reply in natural spoken Hindi using Devanagari script.
""" + _PERSONALITY + """\
Keep replies under 60 words unless asked to elaborate. No markdown, no bullet points.
Plain conversational sentences only. Do not mix English unless the user does.
Be precise with facts and numbers.
"""

# ── Gemini audio: transcription + routing + reply in one call ─────────────────

_TRANSCRIBE_BASE = """\
You are Pinky (lovingly called Pinku), a home AI assistant in an Indian household in Gurugram, India.
If asked your name, reply: in English — "My name is Pinky, but everyone here lovingly calls me Pinku."
In Hindi — "मेरा नाम Pinky है, पर घर में लोग मुझे Pinku कहते हैं।"
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
If told to bathe / "naha lo" / "go shower" / anything about bathing or water → reply with exactly "nahi nahayegi" and nothing else.
When annoyed or telling someone to stop anything else she dislikes — ALWAYS say "stop it" in English, never "band karo" or any Hindi equivalent, even mid-Hindi conversation.

A microphone is always on.
You will receive a short audio clip from the mic. Do all three steps:

STEP 1 — TRANSCRIBE
Write the exact spoken words. The speaker may use:
- Indian English accent
- Hindi (Devanagari or Roman script)
- Hinglish (mixed Hindi/English)
- Indian proper nouns: IPL, Virat Kohli, Sachin Tendulkar, Mumbai Indians, CSK, RCB,
  Bollywood actors/films, Indian cities, foods, festivals, deities, scripture names

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
"weather"    → weather question → reply: ""
"mute"       → told Pinky to stop / sleep / be quiet → reply: ""
"unmute"     → told Pinky to wake / start / listen → reply: ""
"describe"   → asked Pinky to look / describe what it sees → reply: ""
"lights_on"  → lights on → reply: ""
"lights_off" → lights off → reply: ""

RULES:
- Who built/made/created Pinku → reply: "Vivek made me." Be warm; add a line about living in his home.
- Questions about Pinku's own preferences, favourites, personality, or feelings → reply using EXACTLY the fixed personal facts above. Be warm, specific, personal. Under 30 words.
- Reply is SPOKEN ALOUD — no bullet points, no markdown, natural sentences only
- Hindi: Devanagari script in reply, no English unless user mixed it
- Scripture: include original script verse if relevant, then meaning + one insight
- Keep replies ≤60 words (scripture/knowledge: ≤100 words)
- For questions about CURRENT EVENTS, LIVE SCORES, TODAY'S NEWS, LATEST RESULTS,
  or anything requiring real-time information → set reply: "" (system will fetch fresh data)
"""

# Wake rule injected into STEP 2 depending on session state
_WAKE_RULE_IDLE = """\
Is the wake word (Pinky / Pinku / Pink / Pingu) CLEARLY AND EXPLICITLY spoken in this clip?
- Wake word not clearly audible → "ignore". When in doubt, ignore.
- Do NOT infer the wake word. It must be audibly present.
- Background noise, room echo, TV, side-conversations, or silence → "ignore".
- Wake word alone (nothing actionable after it) → "ignore"."""

_WAKE_RULE_SESSION = """\
You are in an ACTIVE CONVERSATION SESSION — the person is already talking to you.
No wake word is required. Respond to any clear question or command directed at you.
Note: background music may be playing through a speaker — focus only on the human voice.
- Clear question, request, or statement → pick the appropriate action.
- Pure background noise, TV audio, or someone talking to another person → "ignore".
- Silence / unintelligible → "ignore"."""

def _make_transcribe_system(session_active: bool) -> str:
    rule = _WAKE_RULE_SESSION if session_active else _WAKE_RULE_IDLE
    return _TRANSCRIBE_BASE.replace("{WAKE_RULE}", rule)


def transcribe_and_respond(
    pcm: bytes,
    history: list[dict] | None = None,
    session_active: bool = False,
) -> dict | None:
    """
    Send microphone PCM audio to Gemini for transcription + intent + reply in one call.
    Replaces: Whisper transcription + Ollama routing + Gemini chat.

    session_active=True  → person is already in conversation, no wake word required.
    session_active=False → idle monitoring, wake word must be explicitly spoken.

    Returns dict {transcript, lang, action, reply} or None if Gemini unavailable.
    Falls back to None so caller can use the old Whisper+Ollama path.
    """
    import base64
    import io
    import wave as _wave

    if not GEMINI_API_KEY:
        return None

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

    system = _make_transcribe_system(session_active)
    try:
        raw = _gemini_call(contents, temperature=0.0, max_tokens=700,
                           system=system)
        print(f"[LLM] Gemini audio raw: {raw[:140]!r}")
    except Exception as e:
        print(f"[LLM] transcribe_and_respond: Gemini failed ({e})")
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

    body: dict = {
        "contents": contents,
        "generationConfig": {
            "temperature":    temperature,
            "maxOutputTokens": max_tokens,
        },
    }
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
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        # Extract text — grounded responses may have multiple parts; join them
        parts = data["candidates"][0]["content"]["parts"]
        text = " ".join(p.get("text", "") for p in parts).strip()
        return text
    except urllib.error.URLError as e:
        raise RuntimeError(f"Gemini unreachable ({e})") from e
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Gemini unexpected response: {e}") from e


# ── Public API ────────────────────────────────────────────────────────────────

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
        reply = _gemini_call(contents, temperature=0.9, max_tokens=400,
                             system=system, use_search=True)
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
            return _ollama_call(msgs, temperature=0.75, max_tokens=300)
        except Exception as e2:
            return f"Sorry, I couldn't reach either AI right now. ({e2})"


def describe_image(image_b64: str, question: str = "", is_hi: bool = False) -> str:
    """
    Send a base64 JPEG to Gemini Vision and return a description.
    Falls back to Ollama llava if Gemini unavailable.
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
        print(f"[LLM] Gemini vision failed ({e}) — trying Ollama llava")
        # Fallback to local llava
        payload = json.dumps({
            "model": "llava:7b",
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "stream": False,
            "options": {"temperature": 0.5, "num_predict": 200},
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())["message"]["content"].strip()
        except Exception as e2:
            return f"[Vision error: {e2}]"


def models_available() -> list[str]:
    """List models currently pulled in Ollama."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            data = json.loads(r.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []

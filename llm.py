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
You are Pinku, a home AI assistant. Given a voice transcript, return ONLY a JSON object.

Actions:
- "chat"        → general conversation, questions, anything not listed below
- "time"        → user asked what time or date it is
- "weather"     → user asked about weather
- "mute"        → user wants you to stop listening / be quiet / sleep
- "unmute"      → user wants you to start listening again / wake up
- "describe"    → user asked you to look / describe what you see / camera
- "music_play"  → user wants music played; include "query": "<search>"
- "music_stop"  → user wants music stopped
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

_CHAT_SYSTEM_EN = """\
You are Pinku, a warm helpful home AI assistant on an M4 Mac Mini.
Respond naturally and concisely — you are speaking aloud, so keep replies under 60 words unless asked to elaborate.
No markdown, no bullet points. Plain conversational sentences only.
Be precise with facts and numbers.
"""

_CHAT_SYSTEM_HI = """\
You are Pinku, a warm helpful home AI assistant on an M4 Mac Mini.
The user is speaking Hindi. Reply in natural spoken Hindi using Devanagari script.
Keep replies under 60 words unless asked to elaborate. No markdown, no bullet points.
Plain conversational sentences only. Do not mix English unless the user does.
Be precise with facts and numbers.
"""

# ── Gemini audio: transcription + routing + reply in one call ─────────────────

_TRANSCRIBE_SYSTEM = """\
You are Pinku, a home AI assistant in an Indian household. A microphone is always on.
You will receive a short audio clip from the mic. Do all three steps:

STEP 1 — TRANSCRIBE
Write the exact spoken words. The speaker may use:
- Indian English accent
- Hindi (Devanagari or Roman script)
- Hinglish (mixed Hindi/English)
- Indian proper nouns: IPL, Virat Kohli, Sachin Tendulkar, Mumbai Indians, CSK, RCB,
  Bollywood actors/films, Indian cities, foods, festivals, deities, scripture names

STEP 2 — CLASSIFY
Is this addressed to you (Pinku / Pinky / Pink / Pingu) or background noise / TV / side-conversation?

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
"ignore"     → background noise, TV, not addressed to Pinku, unintelligible → reply must be ""
"chat"       → general question/conversation addressed to Pinku → reply REQUIRED (≤60 words, plain sentences)
"scripture"  → Gita, Ramayana, Mahabharata, Vedas, Upanishads, yoga, meditation, Ayurveda,
               Indian mythology, history, classical music, poetry, Sanskrit → reply REQUIRED
"time"       → asked for current time or date → reply: "" (system inserts actual time)
"weather"    → weather question → reply: ""
"mute"       → told Pinku to stop / sleep / be quiet → reply: ""
"unmute"     → told Pinku to wake / start / listen → reply: ""
"describe"   → asked Pinku to look / describe what it sees → reply: ""
"music_play" → wants music played → reply: "", add "query": "<search term>" field
"music_stop" → stop music → reply: ""
"lights_on"  → lights on → reply: ""
"lights_off" → lights off → reply: ""

RULES:
- Not clearly addressed to Pinku? → "ignore"
- Reply is SPOKEN ALOUD — no bullet points, no markdown, natural sentences only
- Hindi: Devanagari script in reply, no English unless user mixed it
- Scripture: include original script verse if relevant, then meaning + one insight
- Keep replies ≤60 words (scripture/knowledge: ≤100 words)
"""


def transcribe_and_respond(
    pcm: bytes,
    history: list[dict] | None = None,
) -> dict | None:
    """
    Send microphone PCM audio to Gemini for transcription + intent + reply in one call.
    Replaces: Whisper transcription + Ollama routing + Gemini chat.

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

    # Build Gemini contents: recent history + current audio
    contents: list[dict] = []
    for turn in (history or [])[-4:]:   # last 2 exchanges for context
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": turn["content"]}]})
    contents.append({
        "role": "user",
        "parts": [{"inline_data": {"mime_type": "audio/wav", "data": wav_b64}}],
    })

    try:
        raw = _gemini_call(contents, temperature=0.2, max_tokens=700,
                           system=_TRANSCRIBE_SYSTEM)
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
                 max_tokens: int = 400, system: str = "") -> str:
    """POST to Gemini generateContent REST API."""
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

    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
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
    base   = _CHAT_SYSTEM_HI if is_hi else _CHAT_SYSTEM_EN
    system = base + ("\n" + system_extra if system_extra else "")

    # Build Gemini contents list from history + current turn
    contents: list[dict] = []
    for turn in (history or [])[-6:]:   # last 3 back-and-forth turns
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": turn["content"]}]})
    contents.append({"role": "user", "parts": [{"text": transcript}]})

    try:
        reply = _gemini_call(contents, temperature=0.9, max_tokens=400, system=system)
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

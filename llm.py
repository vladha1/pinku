"""
LLM — Ollama client for local inference.

Two modes:
  route(transcript) → structured JSON action dict
  chat(transcript, context) → plain text reply

Ollama must be running: `ollama serve`
Pull a model first: `ollama pull llama3.2:3b`

The routing prompt mirrors Pinky's Gemini router but targets a smaller
local model — keep the JSON schema simple and actions explicit.
"""

from __future__ import annotations
import json
import re
import urllib.request
import urllib.error
from config import OLLAMA_URL, OLLAMA_MODEL

# ── Routing schema ────────────────────────────────────────────────────────────

_ROUTE_SYSTEM = """\
You are Pinku, a home AI assistant running on an M4 Mac Mini.
Given a voice transcript, return ONLY a JSON object with the action to take.

Actions and when to use them:
- "chat"        → general conversation, questions, anything not listed below
- "time"        → user asked what time or date it is
- "weather"     → user asked about weather
- "mute"        → user wants you to stop listening / be quiet / sleep
- "unmute"      → user wants you to start listening again / wake up
- "describe"    → user asked you to look / describe what you see / camera
- "music_play"  → user wants music played; include "query": "<search>"
- "music_stop"  → user wants music stopped
- "scripture"   → user asked about Gita, Ramayana, Patanjali, Upanishads, Vedas, Madhushala
                  include "topic": "gita"|"ramayana"|"patanjali"|"upanishads"|"vedas"|"madhushala"
- "lights_on"   → turn lights on
- "lights_off"  → turn lights off
- "ignore"      → background noise, unintelligible, or clearly not addressed to Pinku

Rules:
- Return ONLY valid JSON, no explanation, no markdown.
- Include "transcript" field with the user's exact words.
- Include "lang": "en" or "lang": "hi" based on the language spoken.

Examples:
{"action":"chat","transcript":"what is the capital of France","lang":"en"}
{"action":"time","transcript":"what time is it","lang":"en"}
{"action":"scripture","topic":"gita","transcript":"what does the Gita say about fear","lang":"en"}
{"action":"music_play","query":"raag bhairav","transcript":"play raag bhairav","lang":"hi"}
{"action":"ignore","transcript":"","lang":"en"}
"""

_CHAT_SYSTEM_EN = """\
You are Pinku, a warm and helpful home AI assistant on an M4 Mac Mini.
Respond naturally and concisely — you are speaking aloud, so keep it under 60 words unless asked to elaborate.
No markdown, no bullet points. Plain conversational sentences only.
"""

_CHAT_SYSTEM_HI = """\
You are Pinku, a warm and helpful home AI assistant on an M4 Mac Mini.
The user is speaking Hindi. Reply in natural spoken Hindi using Devanagari script.
Keep it under 60 words unless asked to elaborate. No markdown, no bullet points.
Plain conversational sentences only. Do not mix English unless the user does.
"""

# ── Low-level Ollama call ─────────────────────────────────────────────────────

def _call(messages: list[dict], temperature: float = 0.7,
          max_tokens: int = 512, stream: bool = False) -> str:
    """POST to Ollama /api/chat and return the assistant content string."""
    payload = json.dumps({
        "model":   OLLAMA_MODEL,
        "messages": messages,
        "stream":  stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        return data["message"]["content"].strip()
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama not reachable at {OLLAMA_URL} — is `ollama serve` running? ({e})"
        ) from e


# ── Public API ────────────────────────────────────────────────────────────────

def route(transcript: str) -> dict:
    """
    Route a voice transcript to a structured action dict.
    Always returns a dict with at least {"action": "...", "transcript": "..."}.
    """
    if not transcript.strip():
        return {"action": "ignore", "transcript": ""}

    messages = [
        {"role": "system",    "content": _ROUTE_SYSTEM},
        {"role": "user",      "content": transcript},
    ]
    raw = _call(messages, temperature=0.1, max_tokens=200)

    # Extract JSON — model might wrap it in backticks
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if not m:
        print(f"[LLM] route: no JSON in response: {raw!r}")
        return {"action": "chat", "transcript": transcript, "lang": "en"}
    try:
        result = json.loads(m.group())
        result.setdefault("transcript", transcript)
        return result
    except json.JSONDecodeError:
        print(f"[LLM] route: JSON parse error: {raw!r}")
        return {"action": "chat", "transcript": transcript, "lang": "en"}


def chat(transcript: str,
         history: list[dict] | None = None,
         system_extra: str = "",
         is_hi: bool = False) -> str:
    """
    Generate a conversational reply.
    history: list of {"role": "user"|"assistant", "content": "..."} dicts.
    is_hi=True  → use Hindi system prompt so LLM replies in Devanagari.
    """
    base   = _CHAT_SYSTEM_HI if is_hi else _CHAT_SYSTEM_EN
    system = base + ("\n" + system_extra if system_extra else "")
    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history[-6:])   # last 3 turns for context
    messages.append({"role": "user", "content": transcript})
    return _call(messages, temperature=0.75, max_tokens=300)


def describe_image(image_b64: str, question: str = "", is_hi: bool = False) -> str:
    """
    Send a base64-encoded JPEG to a vision-capable Ollama model.
    Requires: `ollama pull llava` or `ollama pull moondream`

    Falls back gracefully if the model doesn't support vision.
    """
    vision_model = "llava:7b"   # or "moondream" for lighter
    lang_note = " Reply in Hindi." if is_hi else ""
    prompt = (question or "Describe what you see in this image briefly.") + lang_note

    payload = json.dumps({
        "model":  vision_model,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [image_b64],
        }],
        "stream": False,
        "options": {"temperature": 0.5, "num_predict": 200},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        return data["message"]["content"].strip()
    except Exception as e:
        return f"[Vision error: {e}]"


def models_available() -> list[str]:
    """List models currently pulled in Ollama."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            data = json.loads(r.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []

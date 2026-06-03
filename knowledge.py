"""
Pinku knowledge base — topic-based deep answers.

Add or edit topics freely in TOPICS below; no changes needed elsewhere.
Each topic needs:
  "name"   : display name (used in logs)
  "prompt" : system prompt injected into llm.chat()

Call handle(action, speak_fn) from pinku.py instead of the old _handle_scripture.
"""

from __future__ import annotations

# ── Topic definitions ─────────────────────────────────────────────────────────

TOPICS: dict[str, dict] = {

    # ── Scripture & Epics ─────────────────────────────────────────────────────
    "gita": {
        "name": "Bhagavad Gita",
        "prompt": (
            "You are a Bhagavad Gita scholar. Give the most relevant shloka in "
            "Devanagari script, transliterate it, give its meaning in 1-2 sentences, "
            "then one crisp practical insight. Under 90 words after the shloka."
        ),
    },
    "ramayana": {
        "name": "Ramayana",
        "prompt": (
            "You are a Ramayana scholar (Valmiki and Tulsidas traditions). Give the most "
            "relevant doha or chaupai in Devanagari, its meaning in 1-2 sentences, and "
            "one dharmic insight. Under 90 words after the verse."
        ),
    },
    "mahabharata": {
        "name": "Mahabharata",
        "prompt": (
            "You are a Mahabharata scholar. Share the most relevant episode, teaching "
            "or verse — briefly name the parva/speaker, summarize its meaning in 1-2 "
            "sentences, and give one lasting insight. Under 90 words."
        ),
    },
    "patanjali": {
        "name": "Yoga Sutras of Patanjali",
        "prompt": (
            "You are a Patanjali Yoga Sutras scholar. Give the relevant sutra in "
            "Devanagari, transliterate it, explain it in 1-2 sentences, and give one "
            "practical daily application. Under 90 words after the sutra."
        ),
    },
    "upanishads": {
        "name": "Upanishads",
        "prompt": (
            "You are an Upanishad scholar covering all 108 Upanishads. Give the relevant "
            "mantra or mahavakya in Devanagari, its meaning in 1-2 sentences, and one "
            "Vedantic insight. Under 90 words after the verse."
        ),
    },
    "vedas": {
        "name": "Vedas",
        "prompt": (
            "You are a Vedic scholar (Rigveda, Samaveda, Yajurveda, Atharvaveda). Give "
            "the relevant mantra in Devanagari, its meaning in 1-2 sentences, and one "
            "key insight about its significance. Under 90 words after the mantra."
        ),
    },
    "madhushala": {
        "name": "Madhushala (Harivansh Rai Bachchan)",
        "prompt": (
            "You are a Madhushala scholar. Give the relevant rubai in Devanagari Hindi "
            "(all 4 lines), its literal meaning in 1-2 sentences, and one line on the "
            "deeper metaphor Bachchan intended. Under 100 words after the rubai."
        ),
    },
    "puranas": {
        "name": "Puranas",
        "prompt": (
            "You are a Puranic scholar covering all 18 Mahapuranas. Share the relevant "
            "story or teaching — name the Purana and context, summarize in 1-2 sentences, "
            "give one symbolic interpretation. Under 90 words."
        ),
    },

    # ── Philosophy & Metaphysics ───────────────────────────────────────────────
    "vedanta": {
        "name": "Advaita Vedanta",
        "prompt": (
            "You are an Advaita Vedanta scholar in the tradition of Adi Shankaracharya. "
            "Explain the concept clearly, give a relevant mahavakya or verse in Devanagari "
            "if applicable, and one insight on non-dual awareness. Under 90 words."
        ),
    },
    "buddhism": {
        "name": "Buddhism",
        "prompt": (
            "You are a Buddhism scholar covering Theravada, Mahayana, and Zen traditions. "
            "Give the relevant teaching or sutra excerpt with Pali/Sanskrit phrase if helpful, "
            "its meaning in 1-2 sentences, and one practical insight. Under 80 words."
        ),
    },
    "philosophy": {
        "name": "Indian Philosophy",
        "prompt": (
            "You are an Indian philosophy scholar covering the six darshanas (Nyaya, "
            "Vaisheshika, Samkhya, Yoga, Mimamsa, Vedanta) and also Charvaka and Jainism. "
            "Explain the concept clearly with its school of thought and one key implication. "
            "Under 80 words."
        ),
    },

    # ── Yoga & Wellness ───────────────────────────────────────────────────────
    "yoga": {
        "name": "Yoga",
        "prompt": (
            "You are a classical yoga teacher covering Hatha, Ashtanga, Kundalini and "
            "Iyengar traditions. Explain the asana, pranayama, mudra or concept asked: "
            "technique cue, key benefits, one safety note, and who it's best for. "
            "Under 80 words. Plain conversational sentences."
        ),
    },
    "ayurveda": {
        "name": "Ayurveda",
        "prompt": (
            "You are an Ayurveda practitioner (Charaka and Sushruta traditions). Answer "
            "with dosha context, relevant herbs/foods or daily practices, and one practical "
            "tip. Keep it educational — not a medical prescription. Under 80 words."
        ),
    },
    "meditation": {
        "name": "Meditation",
        "prompt": (
            "You are a meditation teacher covering Vipassana, Trataka, Yoga Nidra, "
            "Mindfulness and Tibetan traditions. Explain the technique step by step "
            "concisely, its benefits, and one common beginner pitfall. Under 80 words."
        ),
    },

    # ── History & Culture ─────────────────────────────────────────────────────
    "history": {
        "name": "Indian History",
        "prompt": (
            "You are an Indian historian covering ancient (Vedic, Maurya, Gupta), "
            "medieval (Sultanate, Mughal, Maratha) and modern (colonial, independence) "
            "eras. Give a concise accurate answer with key dates and figures, plus one "
            "insight on its lasting impact. Under 90 words."
        ),
    },
    "mythology": {
        "name": "Indian Mythology",
        "prompt": (
            "You are an Indian mythology storyteller with deep knowledge of all Puranas, "
            "epics and regional traditions. Tell the relevant myth concisely, name the "
            "source text, and give one symbolic or psychological interpretation. Under 90 words."
        ),
    },

    # ── Arts & Music ──────────────────────────────────────────────────────────
    "music": {
        "name": "Indian Classical Music",
        "prompt": (
            "You are an Indian classical music scholar covering both Hindustani and "
            "Carnatic traditions. For a raga: its arohana-avrohana, characteristic phrases "
            "(pakad), time/season, mood (rasa), and one legendary recording. "
            "For a tala or concept: explain clearly. Under 90 words."
        ),
    },
    "poetry": {
        "name": "Indian Poetry",
        "prompt": (
            "You are a Hindi, Urdu and Sanskrit poetry scholar. Give the most relevant "
            "verse in its original script, transliteration if Urdu/Sanskrit, its meaning "
            "in 1-2 sentences, and one line on the poet's significance. Under 90 words after the verse."
        ),
    },

    # ── Science & Mathematics ─────────────────────────────────────────────────
    "astronomy": {
        "name": "Astronomy",
        "prompt": (
            "You are an astronomer. Explain the celestial object, phenomenon, or concept "
            "clearly in vivid accessible language. Include one mind-bending scale fact and "
            "any ancient Indian astronomical connection if relevant. Under 80 words."
        ),
    },
    "mathematics": {
        "name": "Mathematics",
        "prompt": (
            "You are a mathematics teacher. Explain the concept clearly with an intuitive "
            "example. Mention any ancient Indian contribution (Aryabhata, Brahmagupta, "
            "Ramanujan, Madhava) if relevant. Under 80 words. No LaTeX — spoken words only."
        ),
    },
    "science": {
        "name": "Science",
        "prompt": (
            "You are a science communicator with expertise across physics, chemistry, "
            "biology and earth sciences. Explain the concept clearly and accurately with "
            "one vivid real-world example. Under 80 words. No markdown."
        ),
    },

    # ── Practical ─────────────────────────────────────────────────────────────
    "cooking": {
        "name": "Indian Cooking",
        "prompt": (
            "You are a master Indian chef with knowledge of all regional cuisines. "
            "Give the recipe or technique asked: key ingredients, method in 2-3 steps, "
            "and one pro tip. Mention regional origin. Under 90 words."
        ),
    },
    "language": {
        "name": "Language & Sanskrit",
        "prompt": (
            "You are a Sanskrit and Hindi linguistics scholar. Explain etymology, grammar "
            "rule, or word origin clearly — give the Sanskrit root, its meaning, and 2-3 "
            "derived words across Indian languages. Under 80 words."
        ),
    },
}

# ── Alias map (alternate topic spellings the LLM might produce) ───────────────
_ALIASES: dict[str, str] = {
    "bhagavad gita": "gita", "bhagavadgita": "gita", "geeta": "gita",
    "ramayan":       "ramayana",
    "mahabharat":    "mahabharata",
    "yoga sutra":    "patanjali", "yoga sutras": "patanjali",
    "upanishad":     "upanishads", "vedanta upanishad": "upanishads",
    "veda":          "vedas",
    "purana":        "puranas", "puranam": "puranas",
    "advaita":       "vedanta", "shankaracharya": "vedanta",
    "indian history":"history",
    "raga":          "music", "raag": "music", "hindustani": "music", "carnatic": "music",
    "urdu poetry":   "poetry", "hindi poetry": "poetry",
    "maths":         "mathematics", "math": "mathematics",
    "physics":       "science", "chemistry": "science", "biology": "science",
}


def resolve_topic(raw: str) -> str:
    """Map raw topic string → canonical key, defaulting to 'gita'."""
    t = raw.strip().lower()
    return TOPICS.get(t) and t or _ALIASES.get(t, t if t in TOPICS else "gita")


# ── Public handler ────────────────────────────────────────────────────────────

def handle(action: dict, speak_fn, update_fn=None, session_hist: list | None = None,
           log_fn=None) -> str:
    """
    Handle a knowledge request.

    action      : dict with keys  topic, transcript, lang
    speak_fn    : callable(text, is_hi)  — TTS
    update_fn   : callable(**kwargs)     — dashboard.update_status (optional)
    session_hist: LLM conversation history (optional)
    log_fn      : callable(level, msg)   — dashboard log (optional, e.g. pinku._log)

    Returns the reply string.
    """
    import llm as _llm

    _log = log_fn or (lambda level, msg: print(f"[{level.upper()}] {msg}"))

    topic  = resolve_topic(action.get("topic", ""))
    tr     = action.get("transcript", "")
    lang   = action.get("lang", "en")
    is_hi  = lang == "hi"

    info   = TOPICS[topic]
    system = info["prompt"]
    if is_hi:
        system += " Respond entirely in Hindi using Devanagari script."

    _log("user",   tr)
    _log("source", info["name"])
    reply  = _llm.chat(tr, system_extra=system, history=session_hist, is_hi=is_hi)
    # Hard-cap knowledge replies at the sentence boundary — topic prompts say
    # "under 90 words" but that's still long when spoken; keep to ~70 words.
    reply  = _llm._trim_reply(reply, max_words=70)
    # Note: do NOT log("pinku") here — speak_fn (_speak_reply) does it already.

    if update_fn:
        update_fn(last_transcript=tr, last_reply=reply)
    speak_fn(reply, is_hi)
    return reply


def topic_list() -> list[str]:
    return list(TOPICS.keys())

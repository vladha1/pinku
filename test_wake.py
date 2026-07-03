#!/usr/bin/env python3
"""
Wake word detection tests for Pinku.
Run:  python test_wake.py [-v]

Tests _check_wake() and _devanagari_has_wake_shape() using real Whisper
transcript strings from production logs so regressions are caught before
deploying to the M4.

Coverage:
  - English Roman variants (pinky, pinku, pinki, pingu, pinkoo …)
  - Hindi Devanagari variants seen in real logs (पिंकी, पिंकि, पिके, पिकू …)
  - Greeting prefixes (hey, hi, hello, ok, हे, आई, हाई, हलो, हैलो …)
  - Wake word at START / anywhere / at END of utterance
  - Command extraction (the text returned after stripping the wake word)
  - _devanagari_has_wake_shape() fuzzy Gemini-fallback helper
  - True negatives — transcripts that must NOT trigger
"""

import sys
import types
import unittest

# ── Stub heavy hardware/ML dependencies before importing pinku ────────────────

_STUBS = [
    'config', 'stt', 'tts', 'llm', 'knowledge', 'dashboard',
    'speaker_id', 'apple_stt', 'mlx_stt', 'math_handler',
    'pyaudio', 'numpy', 'resemblyzer', 'flask',
]
for _m in _STUBS:
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

# config attributes accessed inside pinku functions (not module-level, but
# set here so any accidental top-level touch doesn't raise AttributeError)
_cfg = sys.modules['config']
_cfg.SPEAKER_PROFILES_DIR        = '/tmp'
_cfg.SPEAKER_THRESHOLD_UNCERTAIN = 0.65
_cfg.SPEAKER_THRESHOLD_ACCEPT    = 0.80
_cfg.SPEAKER_THRESHOLD_IDLE      = 0.50
_cfg.SPEAKER_ID_MARGIN           = 0.08
_cfg.SPEAKER_ID_ENABLED          = False
_cfg.MIC_DEVICE_INDEX            = -1
_cfg.DASHBOARD_PORT              = 5100
_cfg.DASHBOARD_HOST              = '0.0.0.0'
_cfg.LOG_DIR                     = '/tmp'
_cfg.OLLAMA_MODEL                = 'test-model'
_cfg.WHISPER_MODEL               = 'test-whisper'

_dash = sys.modules['dashboard']
_dash.update_status        = lambda **kw: None
_dash.record_conversation  = lambda *a, **kw: None
_dash.is_enrolling         = lambda: False
_dash.log_message          = lambda *a: None
_dash.start                = lambda *a, **kw: None

_tts = sys.modules['tts']
_tts.speak          = lambda *a, **kw: 0.0
_tts.play_beep      = lambda: None
_tts.play_sleep     = lambda: None
_tts.play_think     = lambda: None

_stt = sys.modules['stt']
_stt.get_mic_level            = lambda: 0.0
_stt.register_pinku_speech    = lambda *a: None

_llm = sys.modules['llm']
_llm.GEMINI_MODEL             = 'gemini-test'
_llm.transcribe_and_respond   = lambda *a, **kw: None
_llm.text_route_and_respond   = lambda *a, **kw: None

import pinku  # noqa: E402  (must come after stubs)

check_wake               = pinku._check_wake
devanagari_has_wake_shape = pinku._devanagari_has_wake_shape


# ─────────────────────────────────────────────────────────────────────────────

class TestCheckWakePositive(unittest.TestCase):
    """Transcripts that MUST trigger the wake word."""

    def _wake(self, text, expected_cmd=None):
        """Assert wake fires and optionally check the extracted command."""
        triggered, cmd = check_wake(text)
        self.assertTrue(triggered,
            f"Expected WAKE for: {text!r}  →  got (False, {cmd!r})")
        if expected_cmd is not None:
            self.assertEqual(cmd.lower().strip(".,?!"),
                             expected_cmd.lower().strip(".,?!"),
                f"Wrong command extracted from {text!r}: got {cmd!r}")

    # ── English, wake word at START ──────────────────────────────────────────

    def test_en_pinky_start(self):
        self._wake("Pinky what's the time", "what's the time")

    def test_en_pinku_start(self):
        self._wake("Pinku turn off the lights", "turn off the lights")

    def test_en_hey_pinky(self):
        self._wake("hey Pinky are you there", "are you there")

    def test_en_hi_pinku(self):
        self._wake("hi pinku how are you", "how are you")

    def test_en_hello_pinky(self):
        self._wake("hello Pinky what's the weather", "what's the weather")

    def test_en_ok_pinku(self):
        self._wake("Ok Pinku what's the weather")

    def test_en_okay_pinky(self):
        self._wake("okay Pinky tell me a joke", "tell me a joke")

    def test_en_filler_okay_so(self):
        self._wake("okay so Pinky tell me a joke", "tell me a joke")

    def test_en_pingu(self):
        self._wake("pingu set a timer", "set a timer")

    def test_en_penku(self):
        self._wake("penku what time is it", "what time is it")

    def test_en_pinkoo(self):
        self._wake("pinkoo play some music", "play some music")

    def test_en_piku(self):
        self._wake("piku what day is it", "what day is it")

    def test_en_pinki(self):
        self._wake("pinki are you awake", "are you awake")

    def test_en_pinko(self):
        self._wake("pinko translate this", "translate this")

    # ── English, wake word at END ────────────────────────────────────────────

    def test_en_pinky_end(self):
        self._wake("what time is it Pinky", "what time is it")

    def test_en_pinku_end(self):
        self._wake("what's the weather Pinku")

    def test_en_question_mark_end(self):
        self._wake("what's the time Pinky?")

    # ── Wake word only (no command) ──────────────────────────────────────────

    def test_en_wake_only(self):
        triggered, cmd = check_wake("Pinky")
        self.assertTrue(triggered)
        self.assertEqual(cmd, "")

    def test_en_hey_wake_only(self):
        triggered, cmd = check_wake("hey Pinky")
        self.assertTrue(triggered)
        self.assertEqual(cmd, "")

    # ── Hindi Devanagari, KNOWN GOOD from production logs ───────────────────

    def test_hi_hai_pinki(self):
        # From log: "है पिंकी, तुम कैसी हो?" — was working
        self._wake("है पिंकी, तुम कैसी हो?")

    def test_hi_aai_pinki(self):
        # From log: "आई पिंकि वोडिस दाई आप" — आई = Whisper Devanagari for "Hi"
        triggered, cmd = check_wake("आई पिंकि वोडिस दाई आप")
        self.assertTrue(triggered,
            "आई पिंकि should wake — आई is in _HINDI_PRE, पिंकि matches fuzzy RE")

    def test_hi_aai_pinki_2(self):
        # Variant spelling
        self._wake("आई पिंकी हाँ आई आई आई आई?")

    def test_hi_halo_pike(self):
        # From log: "हलो पिके गोड़ा बालिं" — हलो = Devanagari "hello", पिके = Pinky
        self._wake("हलो पिके गोड़ा बालिं")

    def test_hi_hailo_pike(self):
        self._wake("हैलो पिके क्या हाल है")

    def test_hi_are_pinku(self):
        self._wake("अरे पिंकू क्या हाल है")

    def test_hi_pinku_first_word(self):
        self._wake("पिंकू क्या बात है")

    def test_hi_pinki_standalone(self):
        self._wake("पिंकी")

    def test_hi_pike_variant(self):
        # पिके — Whisper drops nasal, produces "pike" sound
        self._wake("अरे पिके कैसे हो")

    def test_hi_pinka_variant(self):
        # पिंका — aa-vowel ending Whisper variant
        self._wake("हे पिंका सुनो")

    def test_hi_pika_variant(self):
        # पिका — no nasal, aa ending
        self._wake("हलो पिका")

    def test_hi_penku_variant(self):
        # पेंकू — e-vowel + nasal Whisper variant
        self._wake("हाई पेंकू क्या हाल")

    # ── Mixed script edge cases ──────────────────────────────────────────────

    def test_mixed_dot_prefix(self):
        # Whisper sometimes prepends "." to utterances
        self._wake(". Pinky what's happening")

    def test_mixed_comma_after_wake(self):
        self._wake("Pinky, tell me a joke", "tell me a joke")


class TestCheckWakeNegative(unittest.TestCase):
    """Transcripts that must NOT trigger the wake word."""

    def _no_wake(self, text):
        triggered, _ = check_wake(text)
        self.assertFalse(triggered,
            f"Expected NO WAKE for: {text!r}  →  incorrectly triggered")

    # ── Clearly unrelated English ────────────────────────────────────────────

    def test_no_wake_plain_english(self):
        self._no_wake("what's the time")

    def test_no_wake_weather_query(self):
        self._no_wake("I want to know the weather today")

    def test_no_wake_random_sentence(self):
        self._no_wake("let me think about this for a moment")

    # ── Devanagari without Pinky ─────────────────────────────────────────────

    def test_no_wake_devanagari_no_pinky(self):
        self._no_wake("क्या समय है?")

    def test_no_wake_devanagari_no_match(self):
        # From log: Whisper hallucination with no Pinky shape
        self._no_wake("आदे पीशी वो से ताएं")

    def test_no_wake_peti(self):
        # From log: "राई पेटी कैसे हो?" — पेटी (box) ≠ Pinky phonetically
        self._no_wake("राई पेटी कैसे हो?")

    def test_no_wake_devanagari_time_query(self):
        self._no_wake("कितना बजा है")

    # ── Partial/ambiguous ────────────────────────────────────────────────────

    def test_no_wake_just_hi(self):
        self._no_wake("hi")

    def test_no_wake_just_hey(self):
        self._no_wake("hey")

    def test_no_wake_just_hello(self):
        self._no_wake("hello")


class TestDevanagariHasWakeShape(unittest.TestCase):
    """Direct tests for the fuzzy Gemini-fallback helper."""

    def test_pike_matches(self):
        self.assertTrue(devanagari_has_wake_shape("राई पिके कैसे हो?"))

    def test_pinka_matches(self):
        self.assertTrue(devanagari_has_wake_shape("राई पिंका कैसे हो?"))

    def test_pika_matches(self):
        self.assertTrue(devanagari_has_wake_shape("राई पिका बोलो"))

    def test_binku_matches(self):
        # ब (B) is in the consonant class — catches B/P confusion
        self.assertTrue(devanagari_has_wake_shape("हलो बिंकू कैसे हो"))

    def test_peti_no_match(self):
        # पेटी — ट is NOT in k-family consonants; must not match
        self.assertFalse(devanagari_has_wake_shape("राई पेटी कैसे हो?"))

    def test_time_query_no_match(self):
        self.assertFalse(devanagari_has_wake_shape("क्या समय है?"))

    def test_empty_no_match(self):
        self.assertFalse(devanagari_has_wake_shape(""))

    def test_english_only_no_match(self):
        # No Devanagari at all — fast path should return False
        self.assertFalse(devanagari_has_wake_shape("what is the time"))


class TestCommandExtraction(unittest.TestCase):
    """Verify the command text extracted after the wake word is correct."""

    def _cmd(self, text):
        triggered, cmd = check_wake(text)
        self.assertTrue(triggered, f"Not triggered: {text!r}")
        return cmd

    def test_en_command_after_wake(self):
        self.assertIn("time", self._cmd("Pinky what's the time").lower())

    def test_en_command_before_wake(self):
        # Wake word at end — command is the text BEFORE it
        cmd = self._cmd("what's the time Pinky")
        self.assertIn("time", cmd.lower())

    def test_en_empty_command_wake_only(self):
        _, cmd = check_wake("Hey Pinku")
        self.assertEqual(cmd.strip("., "), "")

    def test_hi_command_after_wake(self):
        _, cmd = check_wake("है पिंकी, तुम कैसी हो?")
        self.assertIn("कैसी", cmd)


if __name__ == "__main__":
    unittest.main(verbosity=2)

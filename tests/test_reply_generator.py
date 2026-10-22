"""Unit tests for src/reply_generator.py."""

from __future__ import annotations

import unittest

from src.reply_generator import (
    FALLBACK_DRAFTS,
    MULTILINGUAL_INTENT_DRAFTS,
    detect_language,
    get_multilingual_draft_reply,
)
from src.triage_service import INTENT_DRAFT_REPLIES


class TestReplyGenerator(unittest.TestCase):
    """Test suite for language identification and multilingual draft replies."""

    def test_language_detection(self):
        """Verify accurate categorization of English, Hinglish, and Tanglish."""
        self.assertEqual(detect_language("Can you track my order 9821?"), "English")
        self.assertEqual(detect_language("Mera order kahan hai abhi tak nahi aaya"), "Hinglish")
        self.assertEqual(detect_language("En order delivery innum varala, enga irukku"), "Tanglish")
        self.assertEqual(detect_language("Bhai please yeh cancel kardo jaldi"), "Hinglish")
        self.assertEqual(detect_language("Account login panna mudiyala password maranthuten"), "Tanglish")
        # Blank / invalid fallback
        self.assertEqual(detect_language(""), "English")
        self.assertEqual(detect_language("   "), "English")

    def test_all_27_intents_covered_multilingually(self):
        """Verify all 27 intents have non-empty templates in English, Hinglish, and Tanglish."""
        self.assertEqual(len(INTENT_DRAFT_REPLIES), 27)

        for intent in INTENT_DRAFT_REPLIES:
            with self.subTest(intent=intent):
                self.assertIn(intent, MULTILINGUAL_INTENT_DRAFTS)
                drafts = MULTILINGUAL_INTENT_DRAFTS[intent]
                for lang in ["English", "Hinglish", "Tanglish"]:
                    draft = drafts.get(lang)
                    self.assertIsNotNone(draft, f"Missing {lang} draft for intent {intent}")
                    self.assertGreater(len(draft.strip()), 15, f"{lang} draft too short for {intent}")

    def test_get_multilingual_draft_reply_behavior(self):
        """Verify helper retrieves appropriate language draft and handles fallbacks."""
        # Known intent with specific language
        hinglish_draft = get_multilingual_draft_reply("track_order", language="Hinglish")
        self.assertIn("live status", hinglish_draft.lower())

        tanglish_draft = get_multilingual_draft_reply("track_order", language="Tanglish")
        self.assertIn("live", tanglish_draft.lower())

        # Unknown intent returns fallback
        fallback = get_multilingual_draft_reply("unknown_intent_xyz", language="Hinglish")
        self.assertEqual(fallback, FALLBACK_DRAFTS["Hinglish"])

        # None intent returns fallback
        fallback_none = get_multilingual_draft_reply(None, language="Tanglish")
        self.assertEqual(fallback_none, FALLBACK_DRAFTS["Tanglish"])


if __name__ == "__main__":
    unittest.main()

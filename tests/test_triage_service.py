"""Unit tests for src/triage_service.py without internet access."""

from __future__ import annotations

import unittest
import pandas as pd

from src.classifier import TriageClassifier
from src.triage_service import (
    DEFAULT_FALLBACK_DRAFT,
    get_draft_reply,
    run_benchmark_suite,
    triage_message,
)


class TestTriageService(unittest.TestCase):
    """Test suite for business logic and triage service."""

    @classmethod
    def setUpClass(cls):
        cls.train_data = pd.DataFrame(
            {
                "message": [
                    "Cancel this order right now",
                    "Please cancel my order",
                    "Track my delivery shipment",
                    "Where is my package status",
                ],
                "intent": [
                    "cancel_order",
                    "cancel_order",
                    "track_order",
                    "track_order",
                ],
                "category": [
                    "ORDER",
                    "ORDER",
                    "ORDER",
                    "ORDER",
                ],
            }
        )
        cls.classifier = TriageClassifier(confidence_threshold=0.50)
        cls.classifier.fit(
            cls.train_data["message"],
            cls.train_data["intent"],
            cls.train_data["category"],
        )

    def test_get_draft_reply(self):
        """Verify draft reply retrieval for known and unknown intents."""
        reply_cancel = get_draft_reply("cancel_order")
        self.assertIn("cancel", reply_cancel.lower())

        reply_payment = get_draft_reply("payment_issue")
        self.assertIn("payment", reply_payment.lower())

        # Unknown intent returns fallback
        self.assertEqual(get_draft_reply("nonexistent_intent"), DEFAULT_FALLBACK_DRAFT)
        self.assertEqual(get_draft_reply(None), DEFAULT_FALLBACK_DRAFT)

    def test_triage_blank_message(self):
        """Verify blank message returns safe error dictionary without throwing."""
        res = triage_message("   ", self.classifier)
        self.assertFalse(res["is_valid"])
        self.assertIn("empty", res["error"].lower())
        self.assertTrue(res["needs_escalation"])

    def test_triage_model_none_fallback(self):
        """Verify conservative fallback when model is unavailable."""
        res = triage_message("Need help with my delayed package", None)
        self.assertTrue(res["is_valid"])
        self.assertTrue(res["is_fallback"])
        self.assertTrue(res["needs_escalation"])
        self.assertIn("unavailable", res["escalation_reason"].lower())

    def test_triage_confident_prediction(self):
        """Verify standard confident prediction workflow."""
        res = triage_message("Please cancel my order", self.classifier, confidence_threshold=0.30)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["predicted_intent"], "cancel_order")
        self.assertEqual(res["predicted_category"], "ORDER")
        self.assertIn("draft_reply", res)

    def test_triage_low_confidence_escalation(self):
        """Verify low confidence triggers human escalation flag with explanation."""
        # By setting a high threshold of 0.99, prediction will be below threshold
        res = triage_message("Random unrelated sentence xyz", self.classifier, confidence_threshold=0.99)
        self.assertTrue(res["needs_escalation"])
        self.assertIn("below threshold", res["escalation_reason"].lower())

    def test_run_benchmark_suite(self):
        """Verify benchmark suite execution generates all 35 records."""
        results = run_benchmark_suite(self.classifier)
        self.assertEqual(len(results), 35)
        for item in results:
            self.assertIn("expected_intent", item)
            self.assertIn("predicted_intent", item)
            self.assertIn("intent_correct", item)
            self.assertIn("urgency_correct", item)
            self.assertIn("draft_reply", item)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for src/classifier.py without internet access."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
import pandas as pd

from src.classifier import (
    TriageClassifier,
    create_baseline_pipeline,
)


class TestClassifier(unittest.TestCase):
    """Test suite for baseline intent and urgency classifier."""

    @classmethod
    def setUpClass(cls):
        # Create a small synthetic dataset for testing
        cls.synthetic_train = pd.DataFrame(
            {
                "message": [
                    "I want to cancel my order please",
                    "Cancel this order right now",
                    "Can you track where my package is",
                    "Where is my parcel delivery status",
                    "Money deducted but order not placed",
                    "Payment failed debit card issue",
                    "Please send me the invoice copy",
                    "Need receipt and invoice for tax",
                ],
                "intent": [
                    "cancel_order",
                    "cancel_order",
                    "track_order",
                    "track_order",
                    "payment_issue",
                    "payment_issue",
                    "get_invoice",
                    "get_invoice",
                ],
                "category": [
                    "ORDER",
                    "ORDER",
                    "ORDER",
                    "ORDER",
                    "PAYMENT",
                    "PAYMENT",
                    "INVOICE",
                    "INVOICE",
                ],
            }
        )
        cls.classifier = TriageClassifier(confidence_threshold=0.40)
        cls.classifier.fit(
            cls.synthetic_train["message"],
            cls.synthetic_train["intent"],
            cls.synthetic_train["category"],
        )

    def test_predict_intent_and_confidence(self):
        """Verify intent prediction returns expected keys, mapped category, and bounded confidence."""
        res = self.classifier.predict_intent("Please cancel my order")
        self.assertIn("intent", res)
        self.assertIn("category", res)
        self.assertIn("confidence", res)
        self.assertIn("needs_escalation", res)

        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertEqual(res["intent"], "cancel_order")
        self.assertEqual(res["category"], "ORDER")

    def test_predict_missing_or_blank_input_raises_error(self):
        """Verify ValueError is raised on empty or whitespace strings."""
        with self.assertRaises(ValueError):
            self.classifier.predict_intent("")

        with self.assertRaises(ValueError):
            self.classifier.predict_intent("   ")

        with self.assertRaises(ValueError):
            self.classifier.predict_urgency("")

    def test_predict_urgency_heuristics(self):
        """Verify urgency determination logic with lexical cues and priority intents."""
        # High urgency via lexical marker (English)
        res_high_en = self.classifier.predict_urgency("This is extremely urgent, cancel it")
        self.assertEqual(res_high_en["urgency"], "HIGH")

        # High urgency via Hinglish marker
        res_high_hi = self.classifier.predict_urgency("Account se paisa kat gaya jaldi dekho")
        self.assertEqual(res_high_hi["urgency"], "HIGH")

        # High urgency via Tanglish marker
        res_high_ta = self.classifier.predict_urgency("Romba urgent, seekiram cancel pannunga")
        self.assertEqual(res_high_ta["urgency"], "HIGH")

        # Medium urgency via keyword
        res_med = self.classifier.predict_urgency("Having delay in delivery status")
        self.assertEqual(res_med["urgency"], "MEDIUM")

        # Low urgency standard query
        res_low = self.classifier.predict_urgency("Please share the invoice details for tax filing")
        self.assertEqual(res_low["urgency"], "LOW")

    def test_predict_full_triage(self):
        """Verify the full predict method combines intent, category, and urgency."""
        res = self.classifier.predict("Track my parcel delivery please")
        self.assertEqual(res["predicted_intent"], "track_order")
        self.assertEqual(res["predicted_category"], "ORDER")
        self.assertIn(res["urgency"], ["HIGH", "MEDIUM", "LOW"])
        self.assertIsInstance(res["confidence"], float)

    def test_model_save_and_load(self):
        """Verify that model artifacts can be serialized and reloaded cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"
            self.classifier.save(model_path)
            self.assertTrue(model_path.is_file())

            reloaded = TriageClassifier.load(model_path)
            self.assertEqual(reloaded.confidence_threshold, self.classifier.confidence_threshold)

            orig_res = self.classifier.predict_intent("Where is my parcel")
            reloaded_res = reloaded.predict_intent("Where is my parcel")

            self.assertEqual(orig_res["intent"], reloaded_res["intent"])
            self.assertEqual(orig_res["category"], reloaded_res["category"])
            self.assertAlmostEqual(orig_res["confidence"], reloaded_res["confidence"], places=4)


if __name__ == "__main__":
    unittest.main()

"""Regression test for order delivery status inquiry in Tanglish.

Issue report:
Input: "My order innum varala, please help pannunga!"
Expected taxonomy label: intent='track_order', category='ORDER'.
(Previous behavior: predicted 'change_order' at 18% confidence).
"""

from __future__ import annotations

import unittest
from src.classifier import TriageClassifier
from src.triage_service import (
    DEFAULT_MODEL_PATH,
    load_classifier_safe,
    triage_message,
)


class TestRegressionOrderStatus(unittest.TestCase):
    """Regression test ensuring order status in Tanglish classifies accurately."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = load_classifier_safe(DEFAULT_MODEL_PATH)

    def test_tanglish_order_arrival_prediction(self):
        """Verify 'My order innum varala, please help pannunga!' classifies as track_order."""
        if self.classifier is None:
            self.skipTest("Trained model artifact not found on disk; skipping live model check.")

        query = "My order innum varala, please help pannunga!"
        res = self.classifier.predict_intent(query)

        # Verified against Bitext taxonomy: inquiry about delayed/unreceived order transit status
        self.assertEqual(
            res["intent"],
            "track_order",
            f"Expected 'track_order' but received '{res['intent']}' with confidence {res['confidence']}",
        )
        self.assertEqual(res["category"], "ORDER")
        self.assertGreater(
            res["confidence"],
            0.50,
            f"Expected confidence > 50% but received {res['confidence']:.2%}",
        )

    def test_tanglish_order_arrival_triage_and_draft(self):
        """Verify end-to-end triage produces appropriate draft response and handles escalation threshold."""
        if self.classifier is None:
            self.skipTest("Trained model artifact not found on disk; skipping live model check.")

        query = "My order innum varala, please help pannunga!"

        # With default 40% threshold: confident prediction
        triage_default = triage_message(query, self.classifier, confidence_threshold=0.40)
        self.assertEqual(triage_default["predicted_intent"], "track_order")
        self.assertIn("track", triage_default["draft_reply"].lower())

        # With user-specified 90% threshold: correctly escalates due to high threshold
        triage_strict = triage_message(query, self.classifier, confidence_threshold=0.90)
        self.assertTrue(triage_strict["needs_escalation"])
        self.assertIn("below threshold", triage_strict["escalation_reason"].lower())


if __name__ == "__main__":
    unittest.main()

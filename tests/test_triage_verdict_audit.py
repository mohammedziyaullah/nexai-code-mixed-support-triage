"""Regression tests for Triage Verdict decision logic, escalation invariants, and product complaints."""

from __future__ import annotations

import unittest
from src.classifier import DEFAULT_CONFIDENCE_THRESHOLD, TriageClassifier
from src.triage_service import (
    DEFAULT_FALLBACK_DRAFT,
    DEFAULT_MODEL_PATH,
    load_classifier_safe,
    triage_message,
)


class TestTriageVerdictAudit(unittest.TestCase):
    """Rigorous boundary and invariant regression tests for the triage decision engine."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = load_classifier_safe(DEFAULT_MODEL_PATH)
        assert cls.classifier is not None, "Trained model artifact required for verdict tests."

    def test_confidence_below_threshold_never_automated_resolution(self):
        """Strict invariant: below-threshold predictions can NEVER produce Automated Resolution."""
        inquiry = "Please send invoice copy"
        res = triage_message(inquiry, self.classifier, confidence_threshold=0.99)
        self.assertTrue(res["needs_escalation"])
        self.assertEqual(res["triage_verdict"], "ESCALATE_TO_HUMAN")
        self.assertNotEqual(res["triage_verdict"], "AUTOMATED_RESOLUTION")
        self.assertIn("below threshold", res["escalation_reason"].lower())
        self.assertEqual(res["draft_reply"], DEFAULT_FALLBACK_DRAFT)

    def test_confidence_at_threshold_boundary(self):
        """Boundary test: exactly at threshold (conf == threshold), low-urgency queries do not escalate."""
        inquiry = "Please send invoice copy"
        raw_intent = self.classifier.predict_intent(inquiry)
        exact_conf = raw_intent["confidence"]

        res = triage_message(inquiry, self.classifier, confidence_threshold=exact_conf)
        self.assertFalse(res["needs_escalation"])
        self.assertEqual(res["triage_verdict"], "AUTOMATED_RESOLUTION")
        self.assertNotEqual(res["draft_reply"], DEFAULT_FALLBACK_DRAFT)

    def test_confidence_above_threshold_automated_resolution(self):
        """Standard path: above threshold with standard urgency resolves automatically."""
        inquiry = "Please send invoice copy"
        res = triage_message(inquiry, self.classifier, confidence_threshold=0.20)
        self.assertFalse(res["needs_escalation"])
        self.assertEqual(res["triage_verdict"], "AUTOMATED_RESOLUTION")
        self.assertIn("invoice", res["draft_reply"].lower())

    def test_high_urgency_escalates_even_with_high_confidence(self):
        """Invariant: high urgency always escalates to human oversight, but preserves intent draft."""
        # Unambiguous cancel order query with high confidence
        query = "Please cancel my order immediately"
        res = triage_message(query, self.classifier, confidence_threshold=0.30)
        self.assertGreater(res["confidence"], 0.30)
        self.assertEqual(res["urgency"], "HIGH")
        self.assertTrue(res["needs_escalation"])
        self.assertEqual(res["triage_verdict"], "ESCALATE_TO_HUMAN")
        self.assertIn("cancel", res["draft_reply"].lower())
        self.assertNotEqual(res["draft_reply"], DEFAULT_FALLBACK_DRAFT)

    def test_product_complaint_hinglish_bekaar_substandard(self):
        """Product dissatisfaction in Hinglish maps to complaint, FEEDBACK, and HIGH urgency."""
        res = triage_message("Bahut bekaar product bheja hai", self.classifier, confidence_threshold=0.40)
        self.assertEqual(res["predicted_intent"], "complaint")
        self.assertEqual(res["predicted_category"], "FEEDBACK")
        self.assertEqual(res["urgency"], "HIGH")
        self.assertTrue(res["needs_escalation"])
        self.assertEqual(res["triage_verdict"], "ESCALATE_TO_HUMAN")

    def test_product_complaint_hinglish_defective_not_working(self):
        """Malfunctioning / defective product in Hinglish maps to complaint, FEEDBACK, and HIGH urgency."""
        res = triage_message(
            "Yeh item kharaab nikla, bilkul kaam nahi kar raha",
            self.classifier,
            confidence_threshold=0.40,
        )
        self.assertEqual(res["predicted_intent"], "complaint")
        self.assertEqual(res["predicted_category"], "FEEDBACK")
        self.assertEqual(res["urgency"], "HIGH")
        self.assertTrue(res["needs_escalation"])
        self.assertEqual(res["triage_verdict"], "ESCALATE_TO_HUMAN")

    def test_product_complaint_tanglish_damaged(self):
        """Damaged delivery in Tanglish maps to complaint, FEEDBACK, and HIGH urgency."""
        res = triage_message(
            "Item damaged-a deliver pannitaanga, customer support complaint register pannanum",
            self.classifier,
            confidence_threshold=0.40,
        )
        self.assertEqual(res["predicted_intent"], "complaint")
        self.assertEqual(res["predicted_category"], "FEEDBACK")
        self.assertEqual(res["urgency"], "HIGH")
        self.assertTrue(res["needs_escalation"])
        self.assertEqual(res["triage_verdict"], "ESCALATE_TO_HUMAN")

    def test_product_complaint_tanglish_mosam_quality(self):
        """Substandard quality in Tanglish maps to complaint, FEEDBACK, and HIGH urgency."""
        res = triage_message(
            "Product work aagala, romba mosama irukku",
            self.classifier,
            confidence_threshold=0.40,
        )
        self.assertEqual(res["predicted_intent"], "complaint")
        self.assertEqual(res["predicted_category"], "FEEDBACK")
        self.assertEqual(res["urgency"], "HIGH")
        self.assertTrue(res["needs_escalation"])
        self.assertEqual(res["triage_verdict"], "ESCALATE_TO_HUMAN")

    def test_product_complaint_english_defective_broken(self):
        """Broken and defective product in English maps to complaint, FEEDBACK, and HIGH urgency."""
        res = triage_message(
            "I received a defective and completely broken item",
            self.classifier,
            confidence_threshold=0.40,
        )
        self.assertEqual(res["predicted_intent"], "complaint")
        self.assertEqual(res["predicted_category"], "FEEDBACK")
        self.assertEqual(res["urgency"], "HIGH")
        self.assertTrue(res["needs_escalation"])
        self.assertEqual(res["triage_verdict"], "ESCALATE_TO_HUMAN")


if __name__ == "__main__":
    unittest.main()

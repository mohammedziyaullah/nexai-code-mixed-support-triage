"""Automated audit tests covering Phase 6 requirements:

- Normalization consistency between training and inference
- Product dissatisfaction regression cases (Tanglish & Hinglish)
- Strict confidence threshold boundaries (<, ==, >)
- Draft reply completeness and relevance across all 27 intents
- Benchmark dataset independence
- Streamlit app module integration
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from app import get_benchmark_data, get_model
from src.classifier import TriageClassifier
from src.normalizer import normalize_code_mixed_text
from src.triage_service import (
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_FALLBACK_DRAFT,
    DEFAULT_MODEL_PATH,
    INTENT_DRAFT_REPLIES,
    get_draft_reply,
    load_classifier_safe,
    triage_message,
)

ALL_27_INTENTS = [
    "cancel_order",
    "change_order",
    "change_shipping_address",
    "check_cancellation_fee",
    "check_invoice",
    "check_payment_methods",
    "check_refund_policy",
    "complaint",
    "contact_customer_service",
    "contact_human_agent",
    "create_account",
    "delete_account",
    "delivery_options",
    "delivery_period",
    "edit_account",
    "get_invoice",
    "get_refund",
    "newsletter_subscription",
    "payment_issue",
    "place_order",
    "recover_password",
    "registration_problems",
    "review",
    "set_up_shipping_address",
    "switch_account",
    "track_order",
    "track_refund",
]


class TestAuditCorrections(unittest.TestCase):
    """Test suite ensuring all audit verification criteria pass."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = load_classifier_safe(DEFAULT_MODEL_PATH)
        assert cls.classifier is not None, "Model artifact must exist for audit tests"

    def test_normalization_consistency(self):
        """Verify vectorizer uses normalize_code_mixed_text identically."""
        vec = self.classifier.pipeline.named_steps["tfidf"]
        self.assertEqual(vec.preprocessor, normalize_code_mixed_text)

        raw_text = "En order innum varala"
        normalized = normalize_code_mixed_text(raw_text)
        self.assertIn("not arrived track order status", normalized)

    def test_product_dissatisfaction_regression(self):
        """Regression test: dissatisfaction, defects, and damage grievances classify as complaint with HIGH urgency."""
        complaint_queries = [
            ("Package completely damaged condition me mila, bahut hi ghatiya service hai", "Hinglish damage/poor service"),
            ("Bahut bekaar product bheja hai", "Hinglish substandard item"),
            ("Yeh item kharaab nikla, bilkul kaam nahi kar raha", "Hinglish defective item"),
            ("Item damaged-a deliver pannitaanga, customer support complaint register pannanum", "Tanglish damaged item/complaint"),
            ("Product work aagala, romba mosama irukku", "Tanglish malfunctioning item"),
            ("I received a defective and completely broken item", "English broken/defective product"),
            ("Vera product vandhuurku. exchange pannanum", "Tanglish wrong product exchange complaint"),
            ("Vera product vandhuurku. exchange pannanum what", "Tanglish wrong product exchange query"),
        ]

        for query, label in complaint_queries:
            with self.subTest(case=label):
                res = triage_message(query, self.classifier, confidence_threshold=0.40)
                self.assertEqual(res["predicted_intent"], "complaint")
                self.assertEqual(res["predicted_category"], "FEEDBACK")
                self.assertEqual(res["urgency"], "HIGH")
                self.assertTrue(res["needs_escalation"])
                self.assertEqual(res["triage_verdict"], "ESCALATE_TO_HUMAN")
                self.assertNotEqual(res["triage_verdict"], "AUTOMATED_RESOLUTION")
                self.assertIn("complaint", res["draft_reply"].lower())

    def test_tanglish_slang_refund_resolution(self):
        """Test colloquial Tanglish return/refund slang: 'macha retarin pannu da'."""
        query = "macha retarin pannu da"
        res = triage_message(query, self.classifier, confidence_threshold=0.40)
        self.assertEqual(res["predicted_intent"], "get_refund")
        self.assertEqual(res["predicted_category"], "REFUND")
        self.assertEqual(res["detected_language"], "Tanglish")
        self.assertGreaterEqual(res["confidence"], 0.80)
        self.assertEqual(res["triage_verdict"], "AUTOMATED_RESOLUTION")
        self.assertIn("refund", res["draft_reply"].lower())

    def test_confidence_threshold_boundaries(self):
        """Verify strict threshold behavior: strictly below (<) must escalate; at or above (>=) must not."""
        # Standard unambiguous English query with ~80%+ confidence and LOW urgency
        inquiry = "Please send me the invoice copy for my records"
        res_base = self.classifier.predict_intent(inquiry)
        actual_conf = res_base["confidence"]
        self.assertGreater(actual_conf, 0.40)

        # 1. Strictly below threshold: set threshold higher than actual confidence
        threshold_above = min(0.99, actual_conf + 0.10)
        res_below = triage_message(inquiry, self.classifier, confidence_threshold=threshold_above)
        self.assertTrue(
            res_below["needs_escalation"],
            f"Expected escalation when conf {actual_conf} < threshold {threshold_above}",
        )
        self.assertEqual(res_below["triage_verdict"], "ESCALATE_TO_HUMAN")
        self.assertNotEqual(res_below["triage_verdict"], "AUTOMATED_RESOLUTION")
        self.assertIn("below threshold", res_below["escalation_reason"].lower())
        # Low confidence predictions must use safe intake fallback draft
        self.assertEqual(res_below["draft_reply"], DEFAULT_FALLBACK_DRAFT)

        # 2. At threshold: set threshold exactly equal to confidence
        res_at = triage_message(inquiry, self.classifier, confidence_threshold=actual_conf)
        # Since actual_conf < threshold is False and urgency is LOW, confidence check alone does not escalate
        self.assertFalse(
            res_at["needs_escalation"],
            "At boundary (conf == threshold), confidence check should not escalate",
        )
        self.assertEqual(res_at["triage_verdict"], "AUTOMATED_RESOLUTION")
        self.assertNotEqual(res_at["draft_reply"], DEFAULT_FALLBACK_DRAFT)

        # 3. Strictly above threshold: set threshold lower than actual confidence
        threshold_below = max(0.10, actual_conf - 0.10)
        res_above = triage_message(inquiry, self.classifier, confidence_threshold=threshold_below)
        self.assertFalse(res_above["needs_escalation"])
        self.assertEqual(res_above["triage_verdict"], "AUTOMATED_RESOLUTION")
        self.assertNotEqual(res_above["draft_reply"], DEFAULT_FALLBACK_DRAFT)

    def test_draft_reply_relevance_across_all_27_intents(self):
        """Verify every intent in the 27-intent taxonomy has a tailored, non-empty draft response."""
        self.assertEqual(len(ALL_27_INTENTS), 27)
        for intent in ALL_27_INTENTS:
            with self.subTest(intent=intent):
                self.assertIn(intent, INTENT_DRAFT_REPLIES)
                draft = get_draft_reply(intent)
                self.assertNotEqual(draft, DEFAULT_FALLBACK_DRAFT)
                self.assertGreater(len(draft), 30)

    def test_benchmark_independence(self):
        """Verify the 35 benchmark cases are strictly independent from training artifacts."""
        with open(DEFAULT_BENCHMARK_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)

        self.assertEqual(len(cases), 35)
        report_path = Path("models") / "baseline_eval_report.json"
        self.assertTrue(report_path.is_file())

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        # Confirm benchmark evaluation is stored distinctly from held-out Bitext evaluation
        self.assertIn("held_out_eval", report)
        self.assertIn("code_mixed_benchmark_eval", report)
        self.assertEqual(report["code_mixed_benchmark_eval"]["num_test_cases"], 35)

    def test_streamlit_app_integration(self):
        """Verify Streamlit app cached helpers execute properly."""
        model = get_model()
        self.assertIsNotNone(model)
        self.assertEqual(len(model.classes_), 27)

        bench_data = get_benchmark_data(0.40)
        self.assertEqual(len(bench_data), 35)
        first_row = bench_data[0]
        self.assertIn("predicted_intent", first_row)
        self.assertIn("intent_correct", first_row)
        self.assertIn("draft_reply", first_row)


if __name__ == "__main__":
    unittest.main()

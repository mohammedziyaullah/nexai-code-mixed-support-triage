"""Unit tests verifying the 61-Intent Specialist Model and taxonomy."""

from __future__ import annotations

import json
from pathlib import Path
import unittest
import pandas as pd

from src.classifier import TriageClassifier
from src.reply_generator import get_multilingual_draft_reply
from src.triage_service import load_classifier_safe, triage_message

MODEL_61_PATH = Path("models") / "intent_classifier_61.joblib"
REPORT_61_PATH = Path("models") / "classifier_61_eval_report.json"
DATASET_61_PATH = Path("data") / "codemix_61_intents" / "CodeMix_Support_61_Intents_English_Tanglish_Hinglish.csv"
TAXONOMY_61_PATH = Path("data") / "codemix_61_intents" / "Intent_Taxonomy_61.csv"


class Test61Intents(unittest.TestCase):
    """Test suite for 61-intent classifier artifacts, taxonomy, and inference."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = load_classifier_safe(MODEL_61_PATH)
        assert cls.classifier is not None, "61-intent model artifact must exist"

    def test_model_artifact_and_classes(self):
        """Verify model artifact exists and has 61 classes."""
        self.assertTrue(MODEL_61_PATH.is_file())
        self.assertEqual(len(self.classifier.classes_), 61)

    def test_evaluation_report(self):
        """Verify evaluation report schema and metrics."""
        self.assertTrue(REPORT_61_PATH.is_file())
        with open(REPORT_61_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertEqual(report["total_intents"], 61)
        self.assertGreaterEqual(report["test_accuracy"], 0.60)
        self.assertIn("per_language_test", report)

    def test_taxonomy_dataset_integrity(self):
        """Verify 549 records and balanced 3-way language distribution."""
        self.assertTrue(DATASET_61_PATH.is_file())
        self.assertTrue(TAXONOMY_61_PATH.is_file())

        df = pd.read_csv(DATASET_61_PATH)
        self.assertEqual(len(df), 549)
        self.assertEqual(df["expected_intent"].nunique(), 61)
        self.assertEqual(df["language"].value_counts().to_dict(), {"English": 183, "Tanglish": 183, "Hinglish": 183})

    def test_fine_grained_inference(self):
        """Verify fine-grained inference on specific sub-intents."""
        # 1. Wrong product delivered
        res1 = self.classifier.predict_intent("Vera product vandhuurku. exchange pannanum what")
        self.assertEqual(res1["intent"], "wrong_product_received")

        # 2. Return request
        res2 = self.classifier.predict_intent("macha retarin pannu da")
        self.assertEqual(res2["intent"], "return_request")

    def test_multilingual_draft_mapping_for_61_intents(self):
        """Verify draft reply generation maps 61 intents to valid language drafts."""
        draft = get_multilingual_draft_reply("wrong_product_received", "Tanglish")
        self.assertIn("complaint", draft.lower())

        draft_refund = get_multilingual_draft_reply("return_request", "Tanglish")
        self.assertIn("refund", draft_refund.lower())


if __name__ == "__main__":
    unittest.main()

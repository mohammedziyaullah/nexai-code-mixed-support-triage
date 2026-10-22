"""Unit tests verifying the Multilingual Candidate Model and dataset governance."""

from __future__ import annotations

import json
from pathlib import Path
import unittest
import pandas as pd

from src.classifier import TriageClassifier
from src.triage_service import DEFAULT_BENCHMARK_PATH, load_classifier_safe

CANDIDATE_MODEL_PATH = Path("models") / "intent_classifier_multilingual_candidate.joblib"
CANDIDATE_REPORT_PATH = Path("models") / "candidate_multilingual_eval_report.json"
EXPANDED_DATASET_PATH = Path("data") / "CodeMixSupportAI_Master_Expanded_English_Tanglish_Hinglish.csv"
PROPOSAL_PATH = Path("data") / "intent_expansion_proposal.md"


class TestMultilingualCandidate(unittest.TestCase):
    """Test suite for candidate model artifacts, dataset validation, and leakage prevention."""

    def test_candidate_model_artifact_loads(self):
        """Verify candidate model artifact exists, loads safely, and has 27 classes."""
        self.assertTrue(CANDIDATE_MODEL_PATH.is_file(), "Candidate model artifact must exist")
        cand_model = load_classifier_safe(CANDIDATE_MODEL_PATH)
        self.assertIsNotNone(cand_model)
        self.assertEqual(len(cand_model.classes_), 27)

    def test_candidate_evaluation_report_schema(self):
        """Verify candidate evaluation report contains all required comparative metrics."""
        self.assertTrue(CANDIDATE_REPORT_PATH.is_file(), "Candidate evaluation report must exist")
        with open(CANDIDATE_REPORT_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)

        eval_res = report["evaluation_results"]
        self.assertIn("code_mixed_validation_split_130", eval_res)
        self.assertIn("curated_benchmark_held_out_35", eval_res)
        self.assertIn("english_bitext_held_out_5375", eval_res)

        # Confirm empirical gain on validation split
        val_metrics = eval_res["code_mixed_validation_split_130"]
        self.assertGreater(val_metrics["candidate_accuracy"], val_metrics["baseline_accuracy"])
        self.assertGreaterEqual(val_metrics["candidate_accuracy"], 0.70)

    def test_expanded_dataset_integrity(self):
        """Verify expanded dataset structure, row count, and language distributions."""
        self.assertTrue(EXPANDED_DATASET_PATH.is_file(), "Master expanded CSV must exist")
        df = pd.read_csv(EXPANDED_DATASET_PATH)

        self.assertEqual(len(df), 648)
        self.assertEqual(df["language"].value_counts().to_dict(), {"Tanglish": 324, "Hinglish": 324})
        self.assertEqual(df["expected_intent"].nunique(), 27)

    def test_zero_leakage_into_held_out_benchmark(self):
        """Critical test: verify candidate dataset has zero text overlap with benchmark."""
        df = pd.read_csv(EXPANDED_DATASET_PATH)
        with open(DEFAULT_BENCHMARK_PATH, "r", encoding="utf-8") as f:
            benchmark = json.load(f)

        bench_texts = {b["message"].strip().lower() for b in benchmark}
        cand_texts = {m.strip().lower() for m in df["message"]}

        overlap = bench_texts.intersection(cand_texts)
        self.assertEqual(len(overlap), 0, f"Data leakage detected! Overlapping items: {overlap}")

    def test_intent_expansion_proposal_integrity(self):
        """Verify intent expansion proposal exists and defines all 4 candidate intents."""
        self.assertTrue(PROPOSAL_PATH.is_file(), "Intent expansion proposal must exist")
        with open(PROPOSAL_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        for proposed_intent in ["warranty_claim", "track_repair_service", "store_locator", "feedback_suggestion"]:
            self.assertIn(proposed_intent, content, f"Proposal missing definition for {proposed_intent}")


if __name__ == "__main__":
    unittest.main()

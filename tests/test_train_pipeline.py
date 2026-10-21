"""Unit tests for the training and evaluation pipeline functions without network access."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest
import pandas as pd

from src.classifier import TriageClassifier
from src.train import (
    evaluate_code_mixed_benchmark,
    evaluate_model,
)


class TestTrainPipeline(unittest.TestCase):
    """Test suite for evaluation functions in train pipeline."""

    def setUp(self):
        self.synthetic_train = pd.DataFrame(
            {
                "message": [
                    "Cancel my order please",
                    "Please cancel order now",
                    "Track my order parcel",
                    "Where is my package delivery",
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
        self.classifier = TriageClassifier()
        self.classifier.fit(
            self.synthetic_train["message"],
            self.synthetic_train["intent"],
            self.synthetic_train["category"],
        )

    def test_evaluate_model_returns_metrics(self):
        """Verify evaluate_model returns accuracy, macro F1, and confidence."""
        test_df = pd.DataFrame(
            {
                "message": [
                    "Cancel order immediately",
                    "Track my package",
                ],
                "intent": [
                    "cancel_order",
                    "track_order",
                ],
            }
        )
        metrics = evaluate_model(self.classifier, test_df)

        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("weighted_f1", metrics)
        self.assertIn("average_confidence", metrics)
        self.assertEqual(metrics["num_samples"], 2)

    def test_evaluate_code_mixed_benchmark(self):
        """Verify benchmark evaluation with synthetic JSON file."""
        synthetic_benchmark = [
            {
                "id": "t1",
                "language": "Hinglish",
                "message": "Yeh order cancel kardo jaldi",
                "expected_intent": "cancel_order",
                "expected_category": "ORDER",
                "expected_urgency": "HIGH",
            },
            {
                "id": "t2",
                "language": "Tanglish",
                "message": "En order tracking enga irukku",
                "expected_intent": "track_order",
                "expected_category": "ORDER",
                "expected_urgency": "MEDIUM",
            },
        ]

        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump(synthetic_benchmark, tmp)
            tmp_path = Path(tmp.name)

        try:
            results = evaluate_code_mixed_benchmark(self.classifier, tmp_path)
            self.assertEqual(results["num_test_cases"], 2)
            self.assertIn("intent_accuracy", results)
            self.assertIn("urgency_accuracy", results)
            self.assertIn("language_breakdown", results)
            self.assertIn("Hinglish", results["language_breakdown"])
            self.assertIn("Tanglish", results["language_breakdown"])
        finally:
            if tmp_path.is_file():
                tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()

"""Dedicated training and evaluation pipeline for the 61-Intent Fine-Grained Classifier.

Trains a specialized classifier directly on the 549-sample CodeMix 61-Intent dataset
spanning English, Tanglish, and Hinglish across 8 broad customer lifecycle categories.

Artifacts generated:
- models/intent_classifier_61.joblib
- models/classifier_61_eval_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from src.classifier import TriageClassifier

DATASET_61_PATH = Path("data") / "codemix_61_intents" / "CodeMix_Support_61_Intents_English_Tanglish_Hinglish.csv"
TAXONOMY_61_PATH = Path("data") / "codemix_61_intents" / "Intent_Taxonomy_61.csv"
MODEL_61_PATH = Path("models") / "intent_classifier_61.joblib"
REPORT_61_PATH = Path("models") / "classifier_61_eval_report.json"


def train_61_intent_model(
    dataset_path: Path = DATASET_61_PATH,
    taxonomy_path: Path = TAXONOMY_61_PATH,
    model_save_path: Path = MODEL_61_PATH,
    report_save_path: Path = REPORT_61_PATH,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate the dedicated 61-intent fine-grained model."""
    print("=" * 70)
    print("Training 61-Intent Fine-Grained Specialist Classifier")
    print("=" * 70)

    # 1. Load taxonomy mapping
    if not taxonomy_path.is_file():
        raise FileNotFoundError(f"Missing taxonomy file at {taxonomy_path}")
    tax_df = pd.read_csv(taxonomy_path)
    intent_to_category = dict(zip(tax_df["expected_intent"], tax_df["intent_category"]))

    # 2. Load dataset
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Missing dataset at {dataset_path}")
    df = pd.read_csv(dataset_path)
    df["category"] = df["expected_intent"].map(intent_to_category)

    print(f"Total samples: {len(df)} across {df['expected_intent'].nunique()} intents")
    print(f"Languages: {df['language'].value_counts().to_dict()}")

    # 3. Stratified split (80/20)
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=random_state,
        stratify=df["expected_intent"],
    )
    print(f"Train split: {len(train_df)} | Held-out test split: {len(test_df)}")

    # 4. Train classifier
    clf = TriageClassifier(confidence_threshold=0.20)
    clf.fit(train_df["message"], train_df["expected_intent"], train_df["category"])

    # 5. Evaluate on test split
    y_true = list(test_df["expected_intent"])
    y_pred = [clf.predict_intent(m)["intent"] for m in test_df["message"]]

    test_acc = accuracy_score(y_true, y_pred)
    test_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\nTest Accuracy    : {test_acc:.2%}")
    print(f"Test Macro F1    : {test_macro_f1:.4f}")
    print(f"Test Weighted F1 : {test_weighted_f1:.4f}")

    # Per-language accuracy breakdown
    per_language: Dict[str, Dict[str, Any]] = {}
    for lang in ["English", "Tanglish", "Hinglish"]:
        sub = test_df[test_df["language"] == lang]
        if len(sub) > 0:
            sub_preds = [clf.predict_intent(m)["intent"] for m in sub["message"]]
            sub_acc = accuracy_score(sub["expected_intent"], sub_preds)
            per_language[lang] = {
                "samples": len(sub),
                "accuracy": round(sub_acc, 4),
            }
            print(f"  - {lang:8s} Accuracy: {sub_acc:.2%} ({len(sub)} samples)")

    # 6. Save model and report
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save(model_save_path)
    print(f"\nModel artifact saved to {model_save_path}")

    report = {
        "model_type": "TF-IDF + Multinomial Logistic Regression (61 Intents)",
        "total_intents": 61,
        "total_categories": len(tax_df["intent_category"].unique()),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_macro_f1, 4),
        "test_weighted_f1": round(test_weighted_f1, 4),
        "per_language_test": per_language,
        "intent_taxonomy": tax_df.to_dict(orient="records"),
    }

    report_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Evaluation report saved to {report_save_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 61-Intent Specialist Model")
    parser.add_argument("--save-path", type=Path, default=MODEL_61_PATH)
    parser.add_argument("--report-path", type=Path, default=REPORT_61_PATH)
    args = parser.parse_args()

    train_61_intent_model(
        model_save_path=args.save_path,
        report_save_path=args.report_path,
    )

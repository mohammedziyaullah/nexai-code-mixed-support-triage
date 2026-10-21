"""Training and evaluation pipeline for Code-Mixed Support Triage Assistant.

Executes a reproducible train/test split on the Bitext dataset, trains the
TF-IDF + Logistic Regression baseline on the training split only, evaluates
on the held-out split and the code-mixed benchmark, and serializes the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.classifier import TriageClassifier
from src.data_loader import load_bitext_dataset

DEFAULT_MODEL_PATH = Path("models") / "intent_classifier.joblib"
DEFAULT_REPORT_PATH = Path("models") / "baseline_eval_report.json"
BENCHMARK_PATH = Path("data") / "code_mixed_test_cases.json"


def evaluate_model(
    classifier: TriageClassifier,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate classifier on a held-out test split."""
    y_true = list(test_df["intent"])
    predictions = [classifier.predict_intent(msg) for msg in test_df["message"]]
    y_pred = [p["intent"] for p in predictions]
    confidences = [p["confidence"] for p in predictions]

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    avg_confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    unique_labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels).tolist()

    return {
        "num_samples": len(test_df),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "average_confidence": round(avg_confidence, 4),
        "labels_order": unique_labels,
        "confusion_matrix": cm,
        "detailed_report": report,
    }


def evaluate_code_mixed_benchmark(
    classifier: TriageClassifier,
    benchmark_path: Path = BENCHMARK_PATH,
) -> Dict[str, Any]:
    """Evaluate classifier on the curated code-mixed benchmark dataset."""
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Benchmark file not found at {benchmark_path}")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_items = json.load(f)

    y_intent_true = [item["expected_intent"] for item in benchmark_items]
    y_urgency_true = [item["expected_urgency"] for item in benchmark_items]

    predictions = [classifier.predict(item["message"]) for item in benchmark_items]
    y_intent_pred = [p["predicted_intent"] for p in predictions]
    y_urgency_pred = [p["urgency"] for p in predictions]
    confidences = [p["confidence"] for p in predictions]
    escalation_count = sum(1 for p in predictions if p["needs_escalation"])

    intent_acc = float(accuracy_score(y_intent_true, y_intent_pred))
    intent_macro_f1 = float(f1_score(y_intent_true, y_intent_pred, average="macro", zero_division=0))
    urgency_acc = float(accuracy_score(y_urgency_true, y_urgency_pred))
    urgency_macro_f1 = float(f1_score(y_urgency_true, y_urgency_pred, average="macro", zero_division=0))

    # Breakdown by language
    lang_breakdown: Dict[str, Any] = {}
    for lang in ["Hinglish", "Tanglish"]:
        subset_true = [
            y_intent_true[i] for i, item in enumerate(benchmark_items) if item.get("language") == lang
        ]
        subset_pred = [
            y_intent_pred[i] for i, item in enumerate(benchmark_items) if item.get("language") == lang
        ]
        if subset_true:
            lang_breakdown[lang] = {
                "count": len(subset_true),
                "intent_accuracy": round(accuracy_score(subset_true, subset_pred), 4),
            }

    return {
        "num_test_cases": len(benchmark_items),
        "intent_accuracy": round(intent_acc, 4),
        "intent_macro_f1": round(intent_macro_f1, 4),
        "urgency_accuracy": round(urgency_acc, 4),
        "urgency_macro_f1": round(urgency_macro_f1, 4),
        "average_confidence": round(sum(confidences) / len(confidences), 4),
        "escalated_count": escalation_count,
        "language_breakdown": lang_breakdown,
    }


def train_and_evaluate(
    model_save_path: Path = DEFAULT_MODEL_PATH,
    report_save_path: Path = DEFAULT_REPORT_PATH,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train the classifier on Bitext training split and evaluate on held-out and benchmark sets."""
    print("Loading official Bitext Customer Support dataset...")
    df = load_bitext_dataset()
    print(f"Dataset successfully loaded. Total samples: {len(df)}")
    print(f"Unique categories: {df['category'].nunique()} | Unique intents: {df['intent'].nunique()}")

    # Stratified train/test split on held-out data
    print(f"Splitting dataset: {(1 - test_size) * 100:.0f}% train / {test_size * 100:.0f}% held-out test...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["intent"],
    )
    print(f"Train set: {len(train_df)} samples | Test set: {len(test_df)} samples")

    # Fit classifier
    print("Training TF-IDF + Logistic Regression baseline...")
    classifier = TriageClassifier()
    classifier.fit(
        train_df["message"],
        train_df["intent"],
        train_df["category"],
    )
    print("Model training complete.")

    # Evaluate on held-out split
    print("Evaluating on held-out Bitext test split...")
    held_out_metrics = evaluate_model(classifier, test_df)
    print(f"Held-out Test Accuracy : {held_out_metrics['accuracy']:.4f}")
    print(f"Held-out Macro F1      : {held_out_metrics['macro_f1']:.4f}")
    print(f"Held-out Weighted F1   : {held_out_metrics['weighted_f1']:.4f}")

    # Evaluate on code-mixed benchmark
    print("\nEvaluating on team-curated Tanglish/Hinglish benchmark...")
    benchmark_metrics = evaluate_code_mixed_benchmark(classifier)
    print(f"Benchmark Cases        : {benchmark_metrics['num_test_cases']}")
    print(f"Benchmark Intent Acc   : {benchmark_metrics['intent_accuracy']:.4f}")
    print(f"Benchmark Intent F1    : {benchmark_metrics['intent_macro_f1']:.4f}")
    print(f"Benchmark Urgency Acc  : {benchmark_metrics['urgency_accuracy']:.4f}")
    print(f"Benchmark Escalations  : {benchmark_metrics['escalated_count']} / {benchmark_metrics['num_test_cases']}")

    # Save artifacts
    print(f"\nSaving model artifact to {model_save_path}...")
    classifier.save(model_save_path)

    full_report = {
        "dataset": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        "random_seed": random_state,
        "test_size": test_size,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "categories_count": int(df["category"].nunique()),
        "intents_count": int(df["intent"].nunique()),
        "train_label_distribution": train_df["intent"].value_counts().to_dict(),
        "test_label_distribution": test_df["intent"].value_counts().to_dict(),
        "model_artifact_path": str(model_save_path),
        "held_out_eval": held_out_metrics,
        "code_mixed_benchmark_eval": benchmark_metrics,
        "calibration_notice": (
            "Confidence scores are softmax estimates from Logistic Regression, not calibrated probabilities. "
            "Lower confidence on code-mixed benchmarks indicates out-of-distribution language shift, "
            "justifying threshold-based human escalation."
        ),
        "urgency_notice": (
            "Bitext lacks urgency labels. Urgency is evaluated against team-curated benchmark labels "
            "using an explainable heuristic triage engine."
        ),
    }

    report_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_save_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"Evaluation report saved to {report_save_path}")

    return full_report


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Support Triage Classifier.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    train_and_evaluate(args.model_path, args.report_path)


if __name__ == "__main__":
    main()

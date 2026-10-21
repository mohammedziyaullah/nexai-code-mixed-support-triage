# Models Directory

This directory stores serialized model artifacts, vectorizers, and configuration files for intent and urgency classification.

## Current Baseline Model

- **Artifact File:** `models/intent_classifier.joblib` (excluded from git commits via `.gitignore`)
- **Evaluation Report:** `models/baseline_eval_report.json`
- **Architecture:** TF-IDF n-gram vectorizer (`ngram_range=(1, 2)`, `sublinear_tf=True`) + balanced Logistic Regression (`C=1.0`, `max_iter=1000`).
- **Urgency Mechanism:** Explainable lexical heuristic across English, Hinglish, and Tanglish markers paired with high-impact intent priority classification.

## Baseline Benchmark Results

1. **Held-Out Test Split (5,375 Bitext samples, 20% stratified):**
   - **Accuracy:** 99.39%
   - **Macro F1:** 0.9939
   - **Weighted F1:** 0.9939
   - **Average Confidence:** 0.8829

2. **Curated Code-Mixed Benchmark (35 Tanglish/Hinglish test cases):**
   - **Intent Accuracy:** 68.57% (Hinglish: 66.67%, Tanglish: 70.59%)
   - **Intent Macro F1:** 0.5333
   - **Urgency Accuracy:** 77.14%
   - **Average Confidence:** 0.5749
   - **Escalations Triggered:** 13 / 35 (37.1%)

## Calibration & Urgency Notice

- Logistic regression confidence scores represent softmax output probabilities, not calibrated true posteriors.
- The drop in average confidence (0.88 -> 0.57) on out-of-domain code-mixed queries triggers threshold-based human escalation as designed.
- Bitext contains no ground-truth urgency annotations; urgency is rule-derived and auditable.

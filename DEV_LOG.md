# Development Log

## 21 October 2026

- Created the public GitHub repository for NexAI Hack 2026, PS 1.1: [nexai-code-mixed-support-triage](https://github.com/mohammedziyaullah/nexai-code-mixed-support-triage).
- Initialized the local Git repository and configured its GitHub remote.
- Prepared initial README.md, .gitignore, and .env.example files.
- Built application scaffold with initial Streamlit app and offline smoke tests.
- Implemented robust `src/data_loader.py` supporting dynamic schema inspection, column resolution, and template placeholder cleaning.
- Successfully verified live Hugging Face loading of `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (26,872 samples across 11 categories and 27 intents).
- Developed `src/classifier.py`: TF-IDF + Logistic Regression intent classifier, category mapping, and explainable lexical urgency triage heuristic for English, Hinglish, and Tanglish.
- Created `data/code_mixed_test_cases.json`: Curated 35 authentic team-written Tanglish (17) and Hinglish (18) customer support test cases with expected intent and urgency annotations.
- Implemented `src/train.py`: Reproducible stratified train/test split (80/20) and benchmark evaluation pipeline.
- Achieved baseline results:
  - Bitext Held-Out Test Split (5,375 samples): 99.39% accuracy, 0.9939 macro F1.
  - Code-Mixed Benchmark (35 samples): 68.57% intent accuracy, 77.14% urgency accuracy, with 13/35 cases routed to human escalation due to low confidence.
- Documented Responsible AI guidelines regarding classifier confidence calibration and absence of ground-truth urgency in Bitext.
- Updated unit test suite to 20 automated tests covering scaffold, data loading, classifier, pipeline, and benchmark dataset validation.
- Implemented modular `src/triage_service.py` providing suggested draft reply generation for all 27 intents, safe model loading, and conservative fallback handling.
- Built interactive Streamlit Support Triage Inbox in `app.py` with multi-language sample selection, real-time intent/category prediction, confidence metering, explainable urgency indicators, threshold-based human escalation, and interactive benchmark inspection.
- Investigated and resolved misclassification of Tanglish query *"My order innum varala, please help pannunga!"* (previously mispredicted as `change_order` at 18% confidence due to OOV of `innum varala` in English Bitext corpus).
- Created `src/normalizer.py`: General-purpose Romanized Indic preprocessor mapping colloquial action/status expressions into canonical customer support taxonomy without hardcoding test instances.
- Integrated normalization preprocessor into the TF-IDF + Logistic Regression pipeline in `src/classifier.py`.
- Re-evaluated model on held-out split (99.39% accuracy preserved) and code-mixed benchmark (intent accuracy jumped from 68.57% to 82.86%, macro F1 jumped from 0.5333 to 0.7246).
- Added regression test `tests/test_regression_order_status.py` and normalizer test `tests/test_normalizer.py`.
- Executed Full Project Audit and ML Corrections across Phases 1-8:
  - Harmonized human escalation logic across `src/classifier.py` and `src/triage_service.py` (`confidence < threshold` OR `urgency == "HIGH"`).
  - Enforced strict boundary conditions ensuring sub-threshold predictions never produce an automated resolution status.
  - Integrated full $27 \times 27$ confusion matrix and label distribution serialization into `src/train.py` and `models/baseline_eval_report.json`.
  - Added `tests/test_audit_corrections.py` covering product-dissatisfaction regression (Tanglish & Hinglish damage complaints $\rightarrow$ `complaint`, `HIGH` urgency), strict threshold boundaries ($<, ==, >$), draft reply completeness across all 27 intents, benchmark independence, and Streamlit app cache integration.
  - Executed automated test suite: 40 tests across 9 test modules, 100% passing offline in ~3.1 seconds.
  - Validated submission readiness: `.env.example` verified with placeholders only; binary models and local caches confirmed excluded via `.gitignore`; prepared 5-minute video outline, slide deck structure, and social media announcement templates in `README.md`.

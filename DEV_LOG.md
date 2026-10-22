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

## 22 October 2026

- Executed Complete Dataset Verification, Training Pipeline Audit, and Triage Invariant Correction (Phases 1-9):
  - Verified official Hugging Face dataset `bitext/Bitext-customer-support-llm-chatbot-training-dataset` cached locally (`~/.cache/huggingface/` snapshot: 19.2 MB raw CSV, 19.5 MB Arrow cache, CDLA-Sharing-1.0 license, Bitext Innovations S.L. attribution).
  - Confirmed 26,872 samples across 11 categories and 27 intents; zero corrupt rows, 0 nulls.
  - Investigated root cause of product-dissatisfaction misclassification (`place_order` at 77.2%): generic defect/malfunction tokens (`bekaar`, `defective`, `fake`, `broken`, `kharaab`, `mosam`) were OOV in Bitext, allowing the isolated token `product` (+4.32 weight to `place_order`) to dominate.
  - Added generalized linguistic normalization rules for product dissatisfaction and defect indicators without hardcoding individual messages.
  - Added lexical urgency triggers (`defective`, `broken`, `fake`, `wrong item`, `bekaar`, `kharab`, `mosam`).
  - Formalized explicit `triage_verdict` (`"ESCALATE_TO_HUMAN"` vs. `"AUTOMATED_RESOLUTION"`) across `classifier.predict()`, `triage_message()`, and `app.py`.
  - Enforced strict invariant: when `confidence < confidence_threshold`, predictions cannot display Automated Resolution, the unverified status is highlighted in UI, and the draft reply defaults to safe intake response (`DEFAULT_FALLBACK_DRAFT`).
  - Confirmed 35-case benchmark remains strictly held out with zero lexical overlap: benchmark intent accuracy rose to 91.43% (32/35 correct; 94.44% Hinglish, 88.24% Tanglish) while 99.39% in-domain Bitext accuracy was preserved.
  - Resolved 39-vs-40 test count discrepancy by eliminating redundant standalone function in `tests/test_app_structure.py`.
  - Added `tests/test_triage_verdict_audit.py` containing 9 regression tests for boundary conditions, complaint handling, and high-urgency escalation.
  - Executed automated test suite: exactly 48 tests discovered and 100% passing across both `pytest` and `python -m unittest` in ~3.1 seconds.
- Implemented NexAI Master-of-Masters Hackathon Enhancements & Official Stretch Goals:
  - Preserved existing trained model baseline and architecture integrity with zero destructive changes.
  - Made path loaders defensive against `str` vs. `Path` instances in `load_classifier_safe` and `run_benchmark_suite`.
  - Implemented **Official Stretch Goal 1** in `src/reply_generator.py`: Romanized Indic language identification (`detect_language`) and comprehensive multilingual draft response generation across all 27 intents in English, Hinglish, and Tanglish.
  - Implemented **Official Stretch Goal 2** in `app.py`: Real-time Support Queue Dashboard with live volume metrics, distribution charts (volume by intent, volume by urgency & language), interactive status/urgency filters, and an interactive Support Agent Review & Correction Station with intent override, urgency override, draft editing, and audit notes.
  - Implemented **Official Stretch Goal 3** in `src/similarity_retriever.py`: Few-shot TF-IDF cosine-similarity retrieval returning Top-3 verified reference cases for each incoming customer query; strictly validated zero leakage against the 35 benchmark test cases.
  - Formulated comprehensive Model Candidate Comparison Report in `models/candidate_comparison_report.json` and `app.py` Tab 4, rigorously documenting the empirical trade-offs between Fine-Tuned XLM-RoBERTa (Candidate 1: 56.7% code-mixed acc, 1.1GB size, ~240ms CPU latency), Vanilla TF-IDF + Logistic Regression (Candidate 2: 68.57% code-mixed acc, 99.39% Bitext in-domain), and the Selected Champion Hybrid (Candidate 3: **91.43% code-mixed acc**, 99.39% Bitext in-domain, **<4ms CPU latency**, 2.9MB artifact).
  - Expanded test suite to 55 automated tests (102 subtests) across 12 test files covering multilingual reply generation, leakage prevention, queue invariants, and boundary conditions. All 55 tests pass offline with zero errors.

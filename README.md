# Code-Mixed Support Triage Assistant

**NexAI Hack 2026 — Problem Statement 1.1**  
*Intelligent Customer Support Triage for Multilingual Indian E-Commerce and FinTech (English • Hinglish • Tanglish)*

---

## 📌 Executive Summary & Problem Overview

Indian customer support operations receive millions of inquiries where customers naturally code-mix English with regional languages (predominantly **Hinglish** and **Tanglish**). Standard keyword matching and English-only classifiers fail on colloquial Romanized Indic phrasing (e.g., *"My order innum varala"*, *"Paisa kat gaya order place nahi hua"*), leading to misrouting, slow resolution, and customer churn.

The **Code-Mixed Support Triage Assistant** provides an end-to-end intelligent triage solution:
1. **Multilingual Intent & Category Classification:** Resolves customer inquiries into 27 granular operational intents across 11 customer support domains.
2. **Transparent Lexical Urgency Scoring:** Prioritizes critical inquiries using explainable, auditable multilingual urgency heuristics (since Bitext contains no ground-truth urgency labels).
3. **Calibrated Confidence & Threshold-Based Human Escalation:** Automatically escalates low-confidence or high-risk cases to human agents with explicit justifications.
4. **Supervised Draft Generation:** Generates intent-specific draft responses for human agent review and approval.
5. **Rigorous Benchmark Suite:** Evaluates real-time performance against 35 team-curated, held-out Hinglish and Tanglish test cases.

---

## 📊 Evaluation & Verification Results

### Reproducible Baseline vs. Normalized Pipeline Performance

| Metric / Evaluation Set | Baseline Pipeline (Before Normalization) | Audited & Corrected Pipeline (After Normalization) | Relative Delta |
| :--- | :--- | :--- | :--- |
| **Bitext Held-out Test Split** (5,375 English samples) | Accuracy: **99.39%**<br>Macro F1: **0.9939** | Accuracy: **99.39%**<br>Macro F1: **0.9939** | Preserved 100% in-domain English performance |
| **Code-Mixed Benchmark** (35 Tanglish/Hinglish cases) | Accuracy: **68.57%** (24/35)<br>Macro F1: **0.5333** | Accuracy: **82.86%** (29/35)<br>Macro F1: **0.7246** | **+14.29% Accuracy**<br>**+0.191 Macro F1** |
| - *Hinglish Subset (18 cases)* | Intent Accuracy: 66.67% | Intent Accuracy: **83.33%** | +16.66% Accuracy |
| - *Tanglish Subset (17 cases)* | Intent Accuracy: 70.59% | Intent Accuracy: **82.35%** | +11.76% Accuracy |
| **Urgency Accuracy (Heuristic vs. Annotations)** | **77.14%** | **77.14%** | Consistent rule-derived triage |
| **Human Escalation Rate on Benchmark** | 37.1% (13/35) | 34.3% (12/35) | High-risk/low-confidence queries escalated |

### Verified Dataset & Split Breakdown
- **Source:** Hugging Face `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
- **License:** CDLA-Sharing-1.0
- **Total Validated Rows:** 26,872 (0 nulls, 0 corrupt records)
- **Train Set (80% Stratified Split):** 21,497 samples
- **Held-Out Test Set (20% Stratified Split):** 5,375 samples
- **Label Taxonomy:** Balanced across 11 categories and 27 intents (~995 samples per intent).
- **Confusion Matrix:** Full $27 \times 27$ evaluation matrix and per-class precision/recall/F1 serialized in `models/baseline_eval_report.json`.

---

## 🏗️ Architecture & Pipeline Overview

```mermaid
flowchart TD
    A["Customer Support Query (English / Hinglish / Tanglish)"] --> B["Linguistic Normalizer (src/normalizer.py)"]
    B --> C["TF-IDF Feature Extractor (1-2 N-grams, Sublinear TF)"]
    C --> D["Balanced Logistic Regression Classifier (src/classifier.py)"]
    D --> E["Intent Prediction & Softmax Confidence Score"]
    E --> F["Deterministic Intent-to-Category Mapping"]
    A --> G["Multilingual Lexical Urgency Scorer (Rule-Derived)"]
    E & G --> H{"Escalation Decision Engine"}
    H -->|"Confidence < Threshold OR Urgency == HIGH"| I["🚨 Route to Human Agent Triage"]
    H -->|"Confidence >= Threshold AND Urgency != HIGH"| J["🤖 Automated Resolution Path"]
    E --> K["Suggested Draft Reply (src/triage_service.py)"]
    I & J & K --> L["Streamlit Support Triage Inbox (app.py)"]
    M["Curated Benchmark (data/code_mixed_test_cases.json)"] -.->|Independent Validation Only| N["Benchmark Analytics Tab"]
```

### Key Engineering Principles
1. **Linguistic Normalization Preprocessor (`src/normalizer.py`):**
   - Resolves Out-of-Vocabulary (OOV) mismatches by mapping Romanized Indic action/status expressions (`innum varala`, `deliver nahi hua`, `paisa kat gaya`) to canonical support taxonomy concepts.
   - Longest multi-word idioms are prioritized first before partial sub-phrases.
   - Vectorizer incorporates this as a custom `preprocessor`, guaranteeing identical behavior during training and inference.
2. **Benchmark Independence:**
   - The 35 team-curated benchmark cases have **zero lexical overlap** with the 26,872 Bitext messages and were never used in training, tuning, or feature selection.
3. **Escalation Consistency:**
   - Both `src/classifier.py` and `src/triage_service.py` share unified escalation rules: queries are escalated if `confidence < threshold` OR `urgency == "HIGH"`. When confidence falls below the threshold, the system strictly forbids automated resolution.
4. **Responsible AI & Urgency Honesty:**
   - Urgency is transparently documented as an explainable heuristic (lexical risk cues + priority intent priors), explicitly avoiding false claims of supervised model training on unannotated datasets.

---

## 🛠️ Technology Stack

- **Language:** Python 3.12
- **Data Engineering:** `pandas` (v2.3.3), Hugging Face `datasets` (v5.1.0)
- **Machine Learning:** `scikit-learn` (v1.9.1) — TF-IDF Vectorizer & Multinomial Logistic Regression
- **Artifact Serialization:** `joblib` (v1.6.0)
- **Test Automation:** `pytest` (v9.1.1), Python standard `unittest`
- **Application Interface:** `streamlit` (v1.66.0)

---

## 🚀 Setup & Execution Guide

### 1. Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/mohammedziyaullah/nexai-code-mixed-support-triage.git
cd nexai-code-mixed-support-triage
pip install -r requirements.txt
```

### 2. Run Offline Automated Test Suite
Execute the 40-item automated test suite (runs entirely offline without network access):
```bash
python -m pytest tests/
# Or using Python's standard unittest runner:
python -m unittest discover -s tests
```

### 3. Reproduce Model Training & Benchmark Evaluation
Retrain the model on the stratified 80% Bitext training split and evaluate against the 20% held-out test split and the 35-case benchmark:
```bash
python -m src.train
```
*Outputs model artifact to `models/intent_classifier.joblib` and comprehensive evaluation metrics with confusion matrix to `models/baseline_eval_report.json`.*

### 4. Launch the Interactive Support Triage Inbox
Launch the local web application:
```bash
streamlit run app.py
```
Access the application in your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
nexai-code-mixed-support-triage/
├── app.py                             # Interactive Streamlit Web App (Inbox & Benchmark)
├── requirements.txt                   # Dependency specifications
├── .env.example                       # Safe environment variable template (no secrets)
├── .gitignore                         # Strict exclusion rules for artifacts & caches
├── README.md                          # Project documentation and submission package
├── DEV_LOG.md                         # Detailed development and architectural log
├── data/
│   ├── README.md                      # Dataset governance and caching policies
│   └── code_mixed_test_cases.json     # 35 team-curated Tanglish/Hinglish test cases
├── models/
│   ├── README.md                      # Model artifact governance and metric logs
│   ├── baseline_eval_report.json      # Full evaluation metrics, confusion matrix, distributions
│   └── intent_classifier.joblib       # Serialized trained model pipeline (git-ignored)
├── src/
│   ├── __init__.py                    # Source package marker
│   ├── data_loader.py                 # Dynamic Bitext dataset loader and cleaner
│   ├── normalizer.py                  # Code-mixed Romanized Indic linguistic normalizer
│   ├── classifier.py                  # Triage classifier, urgency heuristics, and predict engine
│   ├── train.py                       # Stratified train/eval pipeline & metrics generator
│   └── triage_service.py              # Triage business logic, draft replies, and fallback handling
└── tests/
    ├── __init__.py                    # Test package marker
    ├── test_app_structure.py          # Project structure and scaffold smoke tests
    ├── test_audit_corrections.py      # Regression, threshold boundaries, draft reply tests
    ├── test_benchmark_dataset.py      # Benchmark schema and validation tests
    ├── test_classifier.py             # Classifier training, prediction, and serialization tests
    ├── test_data_loader.py            # Dataset loader and validator tests
    ├── test_normalizer.py             # Linguistic normalization unit tests
    ├── test_regression_order_status.py# Order status Tanglish regression tests
    ├── test_train_pipeline.py         # Offline pipeline evaluation tests
    └── test_triage_service.py         # Triage service, fallback, and draft reply tests
```

---

## 🎯 Hackathon Submission Readiness Checklist

### 1. Repository & Deployment
- [x] **Public GitHub Repository:** [https://github.com/mohammedziyaullah/nexai-code-mixed-support-triage](https://github.com/mohammedziyaullah/nexai-code-mixed-support-triage)
- [x] **Local Demo Fallback:** Fully operational via `streamlit run app.py`.
- [x] **Zero Secrets in Repository:** Verified clean `.env.example` with placeholders only; model binaries and local caches ignored via `.gitignore`.

### 2. Up-to-5-Minute Video Demonstration Script Outline
- **0:00 - 0:45 | Problem Statement & Real-world Impact:**
  - Introduce Problem Statement 1.1: Why Indian customer support requires native code-mixed triage.
  - Show failure modes of standard English NLP on Hinglish/Tanglish messages.
- **0:45 - 2:00 | Live Triage Inbox Demo (`app.py`):**
  - Demonstrate real-time submission of Hinglish query (*"Maine payment kiya par account se paisa kat gaya"* $\rightarrow$ `payment_issue`, `HIGH` urgency, suggested draft reply).
  - Demonstrate Tanglish delivery query (*"My order innum varala, please help pannunga!"* $\rightarrow$ `track_order`, `ORDER`, automated resolution draft).
  - Demonstrate confidence threshold adjustment and automatic human escalation badge with transparent justification.
- **2:00 - 3:15 | Benchmark Suite & Analytics:**
  - Walk through Tab 2: Live evaluation of 35 curated Tanglish and Hinglish test cases.
  - Show language-specific breakdowns (83.3% Hinglish, 82.4% Tanglish accuracy).
- **3:15 - 4:15 | Technical Architecture & Responsible AI:**
  - Explain linguistic normalization in `src/normalizer.py` (zero data leakage).
  - Emphasize urgency transparency (heuristic-based, avoiding fake ground-truth claims).
  - Explain why confidence thresholding is necessary for uncalibrated softmax scores.
- **4:15 - 5:00 | Conclusion & Roadmap:**
  - Summarize hackathon achievements: 40 passing tests, 99.4% in-domain accuracy, 82.9% code-mixed accuracy.
  - Next steps: Multilingual embedding distillation and live agent feedback loop.

### 3. PowerPoint Presentation Deck Structure (6 Slides)
1. **Slide 1: Title & Team:** Project Name, Problem Statement 1.1, NexAI Hack 2026, Team Member Details.
2. **Slide 2: The Code-Mixed Support Crisis:** Why Indian customer service fails with standard keyword systems; linguistic nuances of Tanglish and Hinglish.
3. **Slide 3: System Architecture:** Diagram illustrating Dataset $\rightarrow$ Linguistic Normalization $\rightarrow$ TF-IDF $\rightarrow$ Logistic Regression $\rightarrow$ Explainable Urgency $\rightarrow$ Human Escalation $\rightarrow$ Draft Reply.
4. **Slide 4: Verification & Benchmark Results:** Comparison table showing baseline vs. normalized pipeline (+14.3% accuracy gain on code-mixed queries, 99.4% held-out English accuracy).
5. **Slide 5: Responsible AI & Escalation Design:** Transparent disclosure of heuristic urgency, softmax calibration limitations, and the human-in-the-loop escalation safeguard.
6. **Slide 6: Live Product Showcase & Future Horizons:** Screenshots of the Streamlit Inbox and Benchmark Suite; roadmap toward quantized on-device SLMs.

### 4. Social Media Announcement Templates

#### LinkedIn Post Template
```text
🚀 Excited to present our submission for NexAI Hack 2026 — Problem Statement 1.1: Code-Mixed Support Triage Assistant!

In Indian e-commerce and fintech, customer support inquiries are rarely written in pure English. Customers naturally mix English with regional languages—like Hinglish and Tanglish. Traditional classifiers fail on colloquial Romanized Indic syntax, leading to delayed resolutions and frustrated users.

Our solution:
✅ Multilingual Intent & Category Classification across 27 intents and 11 support domains
✅ Cross-lingual Linguistic Normalization achieving 82.9% accuracy on curated Tanglish/Hinglish queries (+14.3% boost over standard baselines)
✅ Explainable Urgency Heuristics prioritizing high-risk cases
✅ Confidence-Calibrated Human Escalation ensuring uncertain predictions are never guessed
✅ Interactive Streamlit Support Inbox with human-reviewed draft generation

Check out our open-source codebase:
🔗 https://github.com/mohammedziyaullah/nexai-code-mixed-support-triage

#NexAIHack2026 #AI #MachineLearning #NLP #CodeMixed #Hinglish #Tanglish #CustomerSupport #Streamlit #ResponsibleAI
```

#### X (Twitter) Post Template
```text
🚀 Introducing our NexAI Hack 2026 (PS 1.1) project: Code-Mixed Support Triage Assistant!

Support triage built for real Indian customer conversations across English, Hinglish, & Tanglish.

🎯 82.9% code-mixed intent accuracy
🚨 Explainable urgency & confidence escalation
📝 Human-reviewed draft replies

Repo: https://github.com/mohammedziyaullah/nexai-code-mixed-support-triage

#NexAIHack2026 #NLP #MachineLearning #Python #Streamlit
```

---

## 👥 Team Details

- **Lead Developer & Submitter:** Mohammed Ziyaullah
- **GitHub Profile:** [@mohammedziyaullah](https://github.com/mohammedziyaullah)
- **Problem Statement:** 1.1 — Code-Mixed Support Triage Assistant
- **Hackathon:** NexAI Hack 2026

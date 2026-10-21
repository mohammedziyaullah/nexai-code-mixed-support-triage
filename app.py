"""Code-Mixed Support Triage Assistant - Streamlit Web Application.

NexAI Hack 2026 — Problem Statement 1.1
Interactive Support Inbox and Evaluation Dashboard for multilingual (English, Hinglish, Tanglish) customer support messages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st

from src.classifier import DEFAULT_CONFIDENCE_THRESHOLD, TriageClassifier
from src.triage_service import (
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_MODEL_PATH,
    load_classifier_safe,
    run_benchmark_suite,
    triage_message,
)

st.set_page_config(
    page_title="NexAI Code-Mixed Support Triage Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Pre-packaged sample messages for quick demonstration
SAMPLE_QUERIES = [
    {
        "label": "Select a sample query...",
        "text": "",
    },
    {
        "label": "Hinglish — Payment Deducted (High Urgency)",
        "text": "Maine payment kiya par account se paisa kat gaya aur order place nahi hua",
    },
    {
        "label": "Tanglish — Delivery Not Arrived (High Urgency)",
        "text": "En order delivery innum varala, track panna tracking link update aagala",
    },
    {
        "label": "Hinglish — Cancel Order (High Urgency)",
        "text": "Bhai please yeh order cancel kardo jaldi, galti se double item add ho gaya tha",
    },
    {
        "label": "Tanglish — Password Reset (Medium Urgency)",
        "text": "Account login panna mudiyala, password maranthuten, reset link anupunga",
    },
    {
        "label": "Hinglish — Request Invoice (Low Urgency)",
        "text": "Invoice download karne ka option kahan milega? PDF copy chahiye tax filing ke liye",
    },
    {
        "label": "English — Standard Tracking Inquiry (Low Urgency)",
        "text": "Could you please give me the current delivery tracking status for order 98214?",
    },
    {
        "label": "Ambiguous / Low Confidence (Escalation Test)",
        "text": "Kuch samajh nahi aa raha hai kya problem hai please dekho",
    },
]


@st.cache_resource(show_spinner="Loading trained triage model...")
def get_model() -> Optional[TriageClassifier]:
    """Load model artifact once and cache it in memory."""
    return load_classifier_safe(DEFAULT_MODEL_PATH)


@st.cache_data(show_spinner="Loading curated benchmark...")
def get_benchmark_data(confidence_threshold: float):
    """Run and cache benchmark evaluation against current threshold."""
    clf = get_model()
    return run_benchmark_suite(clf, DEFAULT_BENCHMARK_PATH, confidence_threshold=confidence_threshold)


def main():
    st.title("💬 NexAI Code-Mixed Support Triage Assistant")
    st.caption("NexAI Hack 2026 — Problem Statement 1.1 | Multilingual Support Triage (English • Hinglish • Tanglish)")

    # ---------------- Sidebar Configuration ----------------
    with st.sidebar:
        st.header("⚙️ Triage Configuration")

        model = get_model()
        if model is not None:
            st.success("✅ Baseline Model Loaded (`TF-IDF + Logistic Regression`)")
            st.caption(f"Artifact: `{DEFAULT_MODEL_PATH}`")
        else:
            st.warning("⚠️ Model Artifact Missing (`models/intent_classifier.joblib`)")
            st.info("Running in conservative fallback mode. Train model using: `python -m src.train`")

        st.markdown("---")
        st.subheader("Confidence Threshold")
        confidence_threshold = st.slider(
            "Escalation Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=DEFAULT_CONFIDENCE_THRESHOLD,
            step=0.05,
            help=(
                "Messages predicted with confidence below this threshold are flagged for human escalation. "
                "Logistic regression confidence is a softmax estimate, not calibrated probability."
            ),
        )

        st.markdown("---")
        st.subheader("Responsible AI Notice")
        st.markdown(
            """
            - **Urgency Mechanism:** The official Bitext dataset **does not have ground-truth urgency labels**. Urgency is evaluated using a transparent multilingual lexical heuristic (English, Hinglish, Tanglish) and intent risk priors.
            - **Calibration Notice:** Confidence scores reflect relative classifier boundaries; thresholding ensures low-confidence predictions are safely routed to human agents.
            """
        )

    # ---------------- Main Interface Tabs ----------------
    tab_inbox, tab_benchmark, tab_about = st.tabs(
        [
            "📥 Support Triage Inbox",
            "📊 Curated Benchmark Suite (35 Cases)",
            "ℹ️ Architecture & Responsible AI",
        ]
    )

    # ==================== TAB 1: SUPPORT INBOX ====================
    with tab_inbox:
        st.subheader("Incoming Customer Support Message")

        # Quick sample selector
        selected_sample = st.selectbox(
            "Load a pre-configured sample message:",
            options=SAMPLE_QUERIES,
            format_func=lambda s: s["label"],
        )

        # Session state for message input
        if "message_text" not in st.session_state:
            st.session_state["message_text"] = ""

        if selected_sample["text"]:
            st.session_state["message_text"] = selected_sample["text"]

        user_input = st.text_area(
            "Enter customer query (supports English, Hinglish, or Tanglish):",
            value=st.session_state["message_text"],
            height=110,
            placeholder="Type customer message here...",
        )

        col_action, col_clear = st.columns([1, 5])
        with col_action:
            submit_clicked = st.button("🚀 Triage Message", type="primary", use_container_width=True)
        with col_clear:
            if st.button("Clear Input", use_container_width=False):
                st.session_state["message_text"] = ""
                st.rerun()

        if submit_clicked:
            if not user_input.strip():
                st.error("Please enter a valid customer message before triaging.")
            else:
                with st.spinner("Triaging message..."):
                    result = triage_message(user_input, model, confidence_threshold=confidence_threshold)

                st.markdown("---")
                st.subheader("🎯 Triage Verdict")

                # Metrics row
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)

                with m_col1:
                    st.metric(
                        label="Predicted Intent",
                        value=result["predicted_intent"],
                        delta=f"Category: {result['predicted_category']}",
                        delta_color="off",
                    )

                with m_col2:
                    conf_val = result["confidence"]
                    st.metric(
                        label="Classifier Confidence",
                        value=f"{conf_val:.1%}",
                        help="Softmax score from Logistic Regression",
                    )
                    st.progress(conf_val)

                with m_col3:
                    urgency_val = result["urgency"]
                    if urgency_val == "HIGH":
                        urg_color = "🔴"
                    elif urgency_val == "MEDIUM":
                        urg_color = "🟡"
                    else:
                        urg_color = "🟢"

                    st.metric(
                        label="Triage Urgency",
                        value=f"{urg_color} {urgency_val}",
                        help=result["urgency_reason"],
                    )
                    st.caption(f"Rule: {result['urgency_reason']}")

                with m_col4:
                    needs_esc = result["needs_escalation"]
                    if needs_esc:
                        st.error("🚨 ESCALATE TO HUMAN")
                    else:
                        st.success("🤖 AUTOMATED RESOLUTION")

                    st.caption(f"**Reason:** {result['escalation_reason']}")

                # Draft reply card
                st.markdown("---")
                st.subheader("📝 Suggested Draft Response")
                st.info("⚠️ **Human Review Required:** The following draft is generated based on the predicted intent. Review and edit before replying to the customer.")

                edited_reply = st.text_area(
                    "Draft Reply (Editable):",
                    value=result["draft_reply"],
                    height=100,
                )

                if st.button("✅ Approve & Copy Draft Response"):
                    st.success("Response approved for customer communication!")

    # ==================== TAB 2: BENCHMARK SUITE ====================
    with tab_benchmark:
        st.subheader("Curated Code-Mixed Benchmark Evaluation")
        st.markdown(
            """
            This benchmark consists of **35 authentic, team-written customer support queries** in **Tanglish** and **Hinglish**.
            These cases are held-out and were **never included in training or few-shot examples**.
            """
        )

        try:
            benchmark_results = get_benchmark_data(confidence_threshold)
            df_bench = pd.DataFrame(benchmark_results)

            # Summary KPIs
            total_cases = len(df_bench)
            intent_acc = df_bench["intent_correct"].mean()
            urgency_acc = df_bench["urgency_correct"].mean()
            escalations = df_bench["needs_escalation"].sum()

            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            b_col1.metric("Total Test Cases", total_cases)
            b_col1.caption("18 Hinglish • 17 Tanglish")
            b_col2.metric("Intent Accuracy", f"{intent_acc:.1%}")
            b_col3.metric("Urgency Accuracy", f"{urgency_acc:.1%}")
            b_col4.metric(
                "Escalated Cases",
                f"{escalations} / {total_cases}",
                delta=f"{escalations/total_cases:.1%}",
                delta_color="inverse",
            )

            st.markdown("---")

            # Filter row
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                lang_filter = st.selectbox("Filter by Language:", ["All", "Hinglish", "Tanglish"])
            with f_col2:
                status_filter = st.selectbox(
                    "Filter by Result:",
                    ["All", "Intent Match Only", "Intent Mismatch Only", "Escalated Cases Only"],
                )

            filtered_df = df_bench.copy()
            if lang_filter != "All":
                filtered_df = filtered_df[filtered_df["language"] == lang_filter]

            if status_filter == "Intent Match Only":
                filtered_df = filtered_df[filtered_df["intent_correct"]]
            elif status_filter == "Intent Mismatch Only":
                filtered_df = filtered_df[~filtered_df["intent_correct"]]
            elif status_filter == "Escalated Cases Only":
                filtered_df = filtered_df[filtered_df["needs_escalation"]]

            # Display table with formatting
            display_columns = [
                "id",
                "language",
                "message",
                "expected_intent",
                "predicted_intent",
                "intent_correct",
                "expected_urgency",
                "predicted_urgency",
                "urgency_correct",
                "confidence",
                "needs_escalation",
            ]

            st.dataframe(
                filtered_df[display_columns].rename(
                    columns={
                        "id": "ID",
                        "language": "Language",
                        "message": "Customer Query",
                        "expected_intent": "Expected Intent",
                        "predicted_intent": "Predicted Intent",
                        "intent_correct": "Intent Match",
                        "expected_urgency": "Expected Urgency",
                        "predicted_urgency": "Predicted Urgency",
                        "urgency_correct": "Urgency Match",
                        "confidence": "Confidence",
                        "needs_escalation": "Escalated?",
                    }
                ),
                use_container_width=True,
                height=450,
            )

        except Exception as e:
            st.error(f"Failed to load benchmark evaluation: {e}")

    # ==================== TAB 3: ARCHITECTURE & RESPONSIBLE AI ====================
    with tab_about:
        st.subheader("System Architecture & Governance")
        st.markdown(
            """
            ### 1. Problem Statement 1.1 Overview
            Indian e-commerce and fintech users frequently mix English with Indian regional languages (such as Tanglish and Hinglish).
            The **Code-Mixed Support Triage Assistant** addresses this challenge through:
            - **Intent Classification:** Identifies core customer needs across 27 domain intents.
            - **Category Mapping:** 11 verified Bitext customer support categories.
            - **Confidence-Driven Escalation:** Escalates uncertain predictions to human agents when confidence falls below the configured threshold.
            - **Multilingual Urgency Prioritization:** Explains why cases need expedited attention.
            - **Supervised Draft Generation:** Provides support agents with pre-drafted responses for human approval.

            ### 2. Model Baseline & Held-Out Results
            - **Model:** TF-IDF text vectorizer with character/word n-grams and balanced Logistic Regression.
            - **Bitext Held-out Split (5,375 samples):**
              - Intent Accuracy: **99.39%**
              - Macro F1: **0.9939**
              - Weighted F1: **0.9939**
            - **Curated Code-Mixed Benchmark (35 cases):**
              - Intent Accuracy: **68.57%** (Hinglish: 66.67%, Tanglish: 70.59%)
              - Urgency Accuracy: **77.14%**
              - Escalation Rate: **37.1%** (13 / 35 cases appropriately routed to human triage)

            ### 3. Responsible AI & Limitations
            - **Calibration Transparency:** Softmax scores are not Bayesian posterior probabilities. The noticeable drop in confidence on code-mixed queries accurately reflects language shift.
            - **Urgency Ground-Truth Honesty:** Bitext contains **no urgency labels**. Urgency is evaluated using explicit lexical heuristics and intent risk profiles, avoiding false claims of supervised urgency learning.
            """
        )


if __name__ == "__main__":
    main()

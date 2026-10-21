import streamlit as st

st.set_page_config(
    page_title="Code-Mixed Support Triage Assistant",
    page_icon="💬",
    layout="wide",
)

st.title("Code-Mixed Support Triage Assistant")
st.caption("NexAI Hack 2026 — Problem Statement 1.1")

st.info(
    "**Project Status: Initial Application Scaffold**\n\n"
    "The classifier and triage inbox are not implemented yet. "
    "Core data pipelines, classification models, and triage workflows will be added in subsequent implementation steps."
)

st.subheader("Overview & Planned Capabilities")
st.markdown(
    """
- **Code-Mixed Classification:** Detect intent and urgency across English, Tanglish, and Hinglish customer messages.
- **Confidence Scoring & Escalation:** Route high-confidence predictions to automated draft replies and escalate uncertain cases to human agents.
- **Interactive Support Inbox:** Allow support agents and evaluators to test incoming messages and inspect triage decisions.
- **Evaluation Suite:** Evaluate accuracy, macro F1, and latency on the official Bitext benchmark and team-curated test cases.
"""
)

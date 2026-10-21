"""Triage service module containing core business logic, draft replies, and fallback handling.

Decouples triage processing and draft generation from Streamlit UI presentation,
allowing thorough automated testing without browser dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.classifier import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    TriageClassifier,
)

DEFAULT_MODEL_PATH = Path("models") / "intent_classifier.joblib"
DEFAULT_BENCHMARK_PATH = Path("data") / "code_mixed_test_cases.json"

# Curated draft replies mapped to verified Bitext intents
INTENT_DRAFT_REPLIES: Dict[str, str] = {
    "cancel_order": (
        "I can help you cancel this order. If the package has not been dispatched yet, "
        "we can process the cancellation immediately. Could you please confirm your order number?"
    ),
    "change_order": (
        "I can help modify the items or quantity in your order before fulfillment. "
        "Please provide your order number and the exact changes you would like to make."
    ),
    "change_shipping_address": (
        "I can update your delivery destination before dispatch. "
        "Please reply with your order number and complete new shipping address including pincode."
    ),
    "check_cancellation_fee": (
        "Standard order cancellations made prior to dispatch do not incur cancellation fees. "
        "Please share your order ID so I can verify the terms for your specific items."
    ),
    "check_invoice": (
        "You can view and verify your itemized billing statement under 'Order History' in your account. "
        "Please let us know if you observe any discrepancy."
    ),
    "check_payment_methods": (
        "We support various payment modes including UPI (GPay, PhonePe, Paytm), Credit/Debit Cards, "
        "Net Banking, and Cash on Delivery where serviceable."
    ),
    "check_refund_policy": (
        "Our standard policy provides a 30-day return window for eligible items in original condition. "
        "Refunds are credited to the original payment source within 5 to 7 business days."
    ),
    "complaint": (
        "We sincerely apologize for the inconvenience and disappointing experience. "
        "Your complaint has been logged with top priority, and our support leadership is reviewing it."
    ),
    "contact_customer_service": (
        "Our customer support team is available 24/7. "
        "How can we best assist you with your inquiry today?"
    ),
    "contact_human_agent": (
        "I am connecting you with a human support agent right now. "
        "Please stay on this chat while I transfer your session with high priority."
    ),
    "create_account": (
        "To register a new account, click 'Sign Up', enter your mobile number or email, "
        "and verify using the OTP sent to your phone."
    ),
    "delete_account": (
        "We can assist with permanently closing your account and removing personal data. "
        "Please confirm your registered email address to receive the verification link."
    ),
    "delivery_options": (
        "We offer Standard Delivery (3-5 business days) and Express Delivery in select metro regions. "
        "Available options for your pincode appear during checkout."
    ),
    "delivery_period": (
        "Standard deliveries typically arrive within 3 to 5 business days depending on transit distance "
        "and destination pincode."
    ),
    "edit_account": (
        "You can update your personal contact details, mobile number, and addresses under 'Account Settings'."
    ),
    "get_invoice": (
        "I can help you retrieve your tax invoice. You can download the PDF directly under "
        "'My Orders' > 'Download Invoice', or I can email a copy to your registered address."
    ),
    "get_refund": (
        "To initiate a refund, please confirm your order number and the reason for return. "
        "Once the item inspection is complete, funds are processed back to your original payment mode."
    ),
    "newsletter_subscription": (
        "You can manage your promotional email preferences or unsubscribe at any time using the "
        "'Unsubscribe' link at the bottom of our emails or in Account Settings."
    ),
    "payment_issue": (
        "I understand your concern regarding the payment status. If money was debited from your bank "
        "without an order confirmation, banking networks typically auto-reverse the amount within 24-48 hours. "
        "Please provide your transaction reference so we can investigate."
    ),
    "place_order": (
        "I can guide you through placing your order. Please review your cart items and shipping details "
        "before confirming checkout."
    ),
    "recover_password": (
        "You can reset your account password by selecting 'Forgot Password' on the login screen. "
        "A secure password reset link will be sent to your registered email or phone."
    ),
    "registration_problems": (
        "If you are facing difficulties receiving OTP during sign-up, please ensure adequate network coverage "
        "or request a resend after 60 seconds."
    ),
    "review": (
        "Thank you for sharing your feedback! You can post your detailed rating and review directly "
        "on the product page under 'Customer Reviews'."
    ),
    "set_up_shipping_address": (
        "To save a new shipping address, navigate to 'Account' > 'Saved Addresses' and enter your "
        "recipient name, phone number, and address."
    ),
    "switch_account": (
        "To switch accounts, please log out from the top profile menu and sign in using your "
        "alternative credentials."
    ),
    "track_order": (
        "You can monitor live transit status for your shipment. "
        "Please provide your order number or tracking ID, or view live status under 'Track Order'."
    ),
    "track_refund": (
        "To check your refund progress, please provide your refund reference or order ID. "
        "Bank transfers typically take 3 to 5 business days to reflect in your account statement."
    ),
}

DEFAULT_FALLBACK_DRAFT = (
    "Thank you for reaching out to support. We have received your message and routed it to a "
    "customer service agent who will review the details and assist you directly."
)


def get_draft_reply(intent: Optional[str]) -> str:
    """Retrieve standard suggested draft response for an intent."""
    if not intent:
        return DEFAULT_FALLBACK_DRAFT
    return INTENT_DRAFT_REPLIES.get(intent.strip().lower(), DEFAULT_FALLBACK_DRAFT)


def load_classifier_safe(model_path: Path = DEFAULT_MODEL_PATH) -> Optional[TriageClassifier]:
    """Safely load trained classifier artifact without crashing if missing or corrupted."""
    try:
        if model_path.is_file():
            return TriageClassifier.load(model_path)
    except Exception:
        pass
    return None


def triage_message(
    message: str,
    classifier: Optional[TriageClassifier],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """Process a single customer message and generate complete triage verdict.

    Handles empty strings, model absence, confidence escalation, and urgency heuristics.

    Args:
        message: Raw customer message text.
        classifier: Trained TriageClassifier instance or None.
        confidence_threshold: Custom confidence threshold below which escalation occurs.

    Returns:
        Dict containing prediction details, confidence, urgency, escalation flag, and draft reply.
    """
    clean_message = (message or "").strip()
    if not clean_message:
        return {
            "is_valid": False,
            "error": "Message cannot be empty or pure whitespace.",
            "message": "",
            "predicted_intent": "UNKNOWN",
            "predicted_category": "UNKNOWN",
            "confidence": 0.0,
            "urgency": "LOW",
            "urgency_score": 0.20,
            "urgency_reason": "No message provided",
            "needs_escalation": True,
            "triage_verdict": "ESCALATE_TO_HUMAN",
            "escalation_reason": "Empty message submitted",
            "draft_reply": DEFAULT_FALLBACK_DRAFT,
            "is_fallback": True,
        }

    # If classifier is not loaded, use conservative fallback
    if classifier is None:
        # We can still run the lexical urgency heuristic directly from a dummy classifier
        fallback_clf = TriageClassifier()
        urgency_res = fallback_clf.predict_urgency(clean_message)
        return {
            "is_valid": True,
            "error": None,
            "message": clean_message,
            "predicted_intent": "MODEL_UNAVAILABLE",
            "predicted_category": "GENERAL",
            "confidence": 0.0,
            "urgency": urgency_res["urgency"],
            "urgency_score": urgency_res["urgency_score"],
            "urgency_reason": f"{urgency_res['urgency_reason']} (Fallback rule)",
            "needs_escalation": True,
            "triage_verdict": "ESCALATE_TO_HUMAN",
            "escalation_reason": "Model artifact unavailable; conservatively routed to human triage",
            "draft_reply": DEFAULT_FALLBACK_DRAFT,
            "is_fallback": True,
        }

    # Temporarily set custom confidence threshold
    original_threshold = classifier.confidence_threshold
    classifier.confidence_threshold = confidence_threshold

    try:
        triage_res = classifier.predict(clean_message)
        confidence = triage_res["confidence"]
        predicted_intent = triage_res["predicted_intent"]
        predicted_category = triage_res["predicted_category"]
        urgency = triage_res["urgency"]
        urgency_reason = triage_res["urgency_reason"]

        needs_escalation = False
        reasons: List[str] = []

        # Authoritative decision logic:
        # Strictly below threshold (<): must escalate to human and cannot display Automated Resolution.
        # At or above threshold (>=): confidence alone does not escalate.
        if confidence < confidence_threshold:
            needs_escalation = True
            reasons.append(
                f"Confidence ({confidence:.1%}) is below threshold ({confidence_threshold:.1%})"
            )
            # Low confidence predictions are unverified: provide safe intake draft
            draft_reply = DEFAULT_FALLBACK_DRAFT
        else:
            draft_reply = get_draft_reply(predicted_intent)

        if urgency == "HIGH":
            needs_escalation = True
            reasons.append("High operational urgency requires human oversight")

        triage_verdict = "ESCALATE_TO_HUMAN" if needs_escalation else "AUTOMATED_RESOLUTION"

        return {
            "is_valid": True,
            "error": None,
            "message": clean_message,
            "predicted_intent": predicted_intent,
            "predicted_category": predicted_category,
            "confidence": confidence,
            "urgency": urgency,
            "urgency_score": triage_res["urgency_score"],
            "urgency_reason": urgency_reason,
            "needs_escalation": needs_escalation,
            "triage_verdict": triage_verdict,
            "escalation_reason": " | ".join(reasons) if reasons else "Prediction confident and standard priority",
            "draft_reply": draft_reply,
            "is_fallback": False,
        }
    finally:
        classifier.confidence_threshold = original_threshold


def run_benchmark_suite(
    classifier: Optional[TriageClassifier],
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Execute triage on all benchmark test cases and return structured comparison records."""
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Benchmark file not found at {benchmark_path}")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results: List[Dict[str, Any]] = []
    for item in cases:
        msg = item["message"]
        res = triage_message(msg, classifier, confidence_threshold=confidence_threshold)

        intent_correct = (res["predicted_intent"].lower() == item["expected_intent"].lower())
        urgency_correct = (res["urgency"].upper() == item["expected_urgency"].upper())

        results.append(
            {
                "id": item.get("id"),
                "language": item.get("language"),
                "message": msg,
                "expected_intent": item.get("expected_intent"),
                "predicted_intent": res["predicted_intent"],
                "intent_correct": intent_correct,
                "expected_category": item.get("expected_category"),
                "predicted_category": res["predicted_category"],
                "expected_urgency": item.get("expected_urgency"),
                "predicted_urgency": res["urgency"],
                "urgency_correct": urgency_correct,
                "confidence": res["confidence"],
                "needs_escalation": res["needs_escalation"],
                "triage_verdict": res["triage_verdict"],
                "escalation_reason": res["escalation_reason"],
                "draft_reply": res["draft_reply"],
                "notes": item.get("notes", ""),
            }
        )

    return results

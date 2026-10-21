"""Baseline intent and urgency classifier for Code-Mixed Support Triage Assistant.

Model Architecture:
- Intent Classification: TF-IDF text vectorizer with Logistic Regression baseline.
- Category Derivation: Ground-truth mapping from predicted intent (verified 1-to-1 in Bitext).
- Urgency Determination: Explicit, rule-based urgency heuristic combining domain keywords
  (English, Hinglish, Tanglish) and high-impact intent priors.

Responsible AI & Calibration Notice:
- Confidence scores returned from `predict_proba` represent softmax scores of linear decision boundaries.
  They are NOT calibrated posterior probabilities and may exhibit overconfidence or sensitivity
  to out-of-domain / code-mixed linguistic shifts.
- Ground-truth urgency annotations DO NOT exist in the Bitext dataset. Urgency is derived via
  transparent heuristic rules rather than claiming supervised model learning on unverified labels.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.normalizer import normalize_code_mixed_text

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.40

# Lexical urgency markers across English, Hinglish, and Tanglish
HIGH_URGENCY_KEYWORDS = [
    # English
    r"\burgent\b",
    r"\basap\b",
    r"\bimmediately\b",
    r"\bemergency\b",
    r"\bfraud\b",
    r"\bunauthorized\b",
    r"\bstolen\b",
    r"\bhacked\b",
    r"\bdamaged\b",
    r"\bdouble charged\b",
    r"\bmoney deduct(?:ed)?\b",
    r"\bhuman agent\b",
    # Hinglish
    r"\bjaldi\b",
    r"\bturant\b",
    r"\bpaisa kat gaya\b",
    r"\bpaise kat gaye\b",
    r"\bpareshan\b",
    r"\bghatiya\b",
    r"\bkharaab\b",
    r"\bmadad karo\b",
    # Tanglish
    r"\bseekiram\b",
    r"\budane\b",
    r"\bpanam pochu\b",
    r"\bkasu pochu\b",
    r"\bromba urgent\b",
    r"\bthappu\b",
    r"\bseri illa\b",
]

MEDIUM_URGENCY_KEYWORDS = [
    r"\bpassword\b",
    r"\botp\b",
    r"\blogin\b",
    r"\bdelay\b",
    r"\bstuck\b",
    r"\bwaiting\b",
    r"\binnum varala\b",
    r"\bnahi aaya\b",
    r"\bupdate\b",
    r"\bdelete\b",
]

# High-impact intents that warrant elevated priority
HIGH_PRIORITY_INTENTS = {
    "payment_issue",
    "cancel_order",
    "contact_human_agent",
    "complaint",
    "change_shipping_address",
}

MEDIUM_PRIORITY_INTENTS = {
    "track_order",
    "track_refund",
    "recover_password",
    "registration_problems",
    "delete_account",
    "place_order",
    "check_refund_policy",
}


def create_baseline_pipeline() -> Pipeline:
    """Create a TF-IDF + Logistic Regression pipeline with code-mixed linguistic normalization."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_code_mixed_text,
                    ngram_range=(1, 2),
                    max_features=12000,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


class TriageClassifier:
    """Classifier and triage engine for customer support queries."""

    def __init__(
        self,
        pipeline: Optional[Pipeline] = None,
        intent_to_category: Optional[Dict[str, str]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self.pipeline = pipeline or create_baseline_pipeline()
        self.intent_to_category: Dict[str, str] = intent_to_category or {}
        self.confidence_threshold = confidence_threshold
        self.classes_: Optional[List[str]] = None

    def fit(
        self,
        messages: Iterable[str],
        intents: Iterable[str],
        categories: Optional[Iterable[str]] = None,
    ) -> TriageClassifier:
        """Fit the pipeline on messages and intent labels.

        Args:
            messages: Iterable of customer message strings.
            intents: Iterable of ground-truth intent labels.
            categories: Optional iterable of corresponding category labels.

        Returns:
            Self.
        """
        msg_list = list(messages)
        intent_list = list(intents)

        if not msg_list or not intent_list:
            raise ValueError("Training data cannot be empty.")
        if len(msg_list) != len(intent_list):
            raise ValueError("Length of messages and intents must match.")

        # Build intent -> category mapping if categories provided
        if categories is not None:
            cat_list = list(categories)
            if len(cat_list) != len(intent_list):
                raise ValueError("Length of categories must match intents.")
            self.intent_to_category = {}
            for intent, category in zip(intent_list, cat_list):
                if intent not in self.intent_to_category:
                    self.intent_to_category[str(intent)] = str(category)

        self.pipeline.fit(msg_list, intent_list)
        clf = self.pipeline.named_steps["clf"]
        self.classes_ = list(clf.classes_)
        return self

    def predict_intent(self, text: str) -> Dict[str, Any]:
        """Predict intent and calculate confidence score for a single message.

        Confidence Note:
            The confidence score is the maximum softmax probability from LogisticRegression.
            It reflects relative boundary distance, NOT calibrated true probability.

        Args:
            text: Customer message.

        Returns:
            Dict containing predicted intent, mapped category, confidence score,
            and confidence-based escalation flag.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input message must be a non-empty string.")

        cleaned_text = text.strip()
        probas = self.pipeline.predict_proba([cleaned_text])[0]
        max_idx = int(np.argmax(probas))
        predicted_intent = str(self.pipeline.classes_[max_idx])
        confidence = float(probas[max_idx])

        category = self.intent_to_category.get(predicted_intent, "UNKNOWN")
        needs_escalation = confidence < self.confidence_threshold

        return {
            "intent": predicted_intent,
            "category": category,
            "confidence": round(confidence, 4),
            "needs_escalation": needs_escalation,
            "escalation_reason": "Low prediction confidence" if needs_escalation else None,
        }

    def predict_urgency(
        self,
        text: str,
        predicted_intent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Determine triage urgency using auditable lexical cues and intent priors.

        Ground-Truth Notice:
            The Bitext dataset lacks urgency annotations. This heuristic provides an
            auditable, explainable triage mechanism without claiming learned urgency.

        Args:
            text: Customer message.
            predicted_intent: Optional predicted intent from step 1.

        Returns:
            Dict with urgency ('HIGH', 'MEDIUM', 'LOW'), score, reason, and detected cues.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input message must be a non-empty string.")

        msg_lower = text.strip().lower()
        matched_high_cues: List[str] = []
        for pattern in HIGH_URGENCY_KEYWORDS:
            found = re.findall(pattern, msg_lower)
            if found:
                matched_high_cues.extend(found)

        matched_medium_cues: List[str] = []
        for pattern in MEDIUM_URGENCY_KEYWORDS:
            found = re.findall(pattern, msg_lower)
            if found:
                matched_medium_cues.extend(found)

        # Urgency scoring rules
        if matched_high_cues:
            urgency = "HIGH"
            reason = f"High-urgency keywords detected: {', '.join(sorted(set(matched_high_cues)))}"
            score = 0.90
        elif predicted_intent in HIGH_PRIORITY_INTENTS:
            urgency = "HIGH"
            reason = f"High-impact priority intent: {predicted_intent}"
            score = 0.80
        elif matched_medium_cues:
            urgency = "MEDIUM"
            reason = f"Moderate urgency markers detected: {', '.join(sorted(set(matched_medium_cues)))}"
            score = 0.55
        elif predicted_intent in MEDIUM_PRIORITY_INTENTS:
            urgency = "MEDIUM"
            reason = f"Medium priority intent: {predicted_intent}"
            score = 0.50
        else:
            urgency = "LOW"
            reason = "Standard inquiry with no urgent cues"
            score = 0.20

        return {
            "urgency": urgency,
            "urgency_score": score,
            "urgency_reason": reason,
            "detected_high_cues": sorted(set(matched_high_cues)),
            "detected_medium_cues": sorted(set(matched_medium_cues)),
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """Perform full end-to-end support message triage.

        Args:
            text: Customer message.

        Returns:
            Full triage dictionary containing intent, category, confidence,
            urgency, escalation status, and reason.
        """
        intent_res = self.predict_intent(text)
        urgency_res = self.predict_urgency(text, predicted_intent=intent_res["intent"])

        # Determine combined escalation (consistent with triage_service)
        needs_escalation = intent_res["needs_escalation"] or (urgency_res["urgency"] == "HIGH")
        escalation_reasons: List[str] = []

        if intent_res["needs_escalation"]:
            escalation_reasons.append(
                f"Confidence ({intent_res['confidence']:.1%}) is below threshold ({self.confidence_threshold:.1%})"
            )
        if urgency_res["urgency"] == "HIGH":
            escalation_reasons.append("High operational urgency requires human oversight")

        return {
            "message": text,
            "predicted_intent": intent_res["intent"],
            "predicted_category": intent_res["category"],
            "confidence": intent_res["confidence"],
            "urgency": urgency_res["urgency"],
            "urgency_score": urgency_res["urgency_score"],
            "urgency_reason": urgency_res["urgency_reason"],
            "needs_escalation": needs_escalation,
            "escalation_reason": " | ".join(escalation_reasons) if escalation_reasons else "Prediction confident and standard priority",
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize model artifacts to disk."""
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "intent_to_category": self.intent_to_category,
            "confidence_threshold": self.confidence_threshold,
            "classes_": self.classes_,
        }
        joblib.dump(payload, target_path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> TriageClassifier:
        """Load serialized model artifact from disk."""
        source_path = Path(filepath)
        if not source_path.is_file():
            raise FileNotFoundError(f"Model artifact not found at {source_path}")
        payload = joblib.load(source_path)
        instance = cls(
            pipeline=payload["pipeline"],
            intent_to_category=payload.get("intent_to_category", {}),
            confidence_threshold=payload.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD),
        )
        instance.classes_ = payload.get("classes_")
        return instance

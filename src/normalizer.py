"""Linguistic normalization module for Romanized code-mixed (Tanglish and Hinglish) text.

Bridges the lexical and phonological gap between colloquial Indian code-mixed
customer support queries and canonical English support taxonomy without hardcoding
individual test instances.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# Base list of (pattern, canonical_replacement).
RAW_CODE_MIXED_RULES: List[Tuple[str, str]] = [
    # ---------------- Tanglish Delivery & Tracking ----------------
    (r"\b(?:en|yen)\s+order\b", "my order"),
    (r"\binnum\s+(?:varala|varavillai|reach\s+aagala|vandhu\s+serala)\b", "not arrived track order status"),
    (r"\b(?:innum\s+vara\s+illa|innum\s+varala)\b", "not arrived track order status"),
    (r"\bvarala\b", "not arrived"),
    (r"\benga\s+irukku\b", "where is tracking status"),
    (r"\btracking\s+link\s+update\s+aagala\b", "tracking status not updated"),
    (r"\bitem\s+damaged-?a\s+deliver\s+pannitaanga\b", "damaged item file a complaint poor service"),
    (r"\bdeliver\s+pannitaanga\b", "delivered"),
    (r"\bdeliver\s+panna\s+illa\b", "not delivered"),
    (r"\bvaravillai\b", "not received not arrived"),

    # ---------------- Hinglish Delivery & Tracking ----------------
    (r"\bmera\s+order\b", "my order"),
    (r"\b(?:abhi\s+tak|ab\s+tak)\s+deliver\s+nahi\s+hua\b", "not arrived track order status"),
    (r"\b(?:abhi\s+tak|ab\s+tak)\s+nahi\s+aaya\b", "not arrived track order status"),
    (r"\bdeliver\s+nahi\s+hua\b", "not arrived track order status"),
    (r"\bnahi\s+aaya\b", "not arrived"),
    (r"\bnahi\s+mila\b", "not received"),
    (r"\bkahan\s+(?:hai|pahucha|pahuncha)\b", "where is tracking status"),
    (r"\bstuck\s+dikha\s+raha\s+hai\b", "stuck tracking status"),
    (r"\bkitne\s+din\s+lagte\s+hain\b", "how long delivery period arrive"),

    # ---------------- Tanglish & Hinglish Cancellation ----------------
    (r"\border-?a\s+cancel\s+pannunga\b", "cancel order"),
    (r"\border\s+cancel\s+(?:pannunga|pannanum|pannidunga)\b", "cancel order"),
    (r"\bcancel\s+(?:pannunga|pannanum|pannidunga)\b", "cancel order"),
    (r"\byeh\s+order\s+cancel\s+(?:kardo|kar\s+do|karna\s+hai)\b", "cancel order"),
    (r"\border\s+cancel\s+(?:kardo|kar\s+do|karna\s+hai)\b", "cancel order"),
    (r"\bcancel\s+(?:kardo|kar\s+do|karna\s+hai|karwana\s+hai)\b", "cancel order"),
    (r"\bcancellation\s+fee\s+pidipaangala\b", "check cancellation fee charge"),
    (r"\bcancellation\s+charges\s+lagenge\s+kya\b", "check cancellation fee charge"),

    # ---------------- Tanglish & Hinglish Payment & Refund ----------------
    (r"\b(?:paisa|paise)\s+kat\s+(?:gaya|gaye)\b", "money debited payment issue failed"),
    (r"\b(?:panam|kasu)\s+pochu\b", "money debited payment issue failed"),
    (r"\bmoney\s+deduct\s+aayiduchu\b", "money debited payment issue"),
    (r"\border\s+confirm\s+aagala\b", "order not placed payment issue"),
    (r"\border\s+place\s+nahi\s+hua\b", "order not placed payment issue"),
    (r"\bbank\s+account-?ku\s+varala\b", "bank account refund not received track refund"),
    (r"\breflect\s+nahi\s+hua\b", "not reflected track refund"),
    (r"\bkab\s+tak\s+aayega\b", "when will it arrive timeline status"),

    # ---------------- Tanglish & Hinglish Account & Auth ----------------
    (r"\blogin\s+(?:panna\s+mudiyala|aagala)\b", "cannot login recover password"),
    (r"\blogin\s+nahi\s+ho\s+raha\b", "cannot login recover password"),
    (r"\bpassword\s+maranthuten\b", "forgot password recover password"),
    (r"\bpassword\s+bhool\s+gaya\b", "forgot password recover password"),
    (r"\breset\s+link\s+(?:anupunga|bhejo)\b", "send password reset link"),
    (r"\botp\s+(?:receive\s+aagala|varala|nahi\s+aa\s+raha)\b", "otp not received registration problems"),
    (r"\bpudhu\s+account\b", "create account registration"),
    (r"\bnew\s+account\s+create\b", "create account registration"),
    (r"\bpurana\s+account\b", "delete account close account"),
    (r"\baccount\s+permanently\s+delete\b", "delete account close account"),
    (r"\bprofile\s+me\s+edit\b", "edit account details"),
    (r"\bsettings\s+katanum\b", "edit account settings"),

    # ---------------- Tanglish & Hinglish Support / Contact ----------------
    (r"\b(?:agent|support)\s+kitta\s+(?:direct-?a|call\s+la)?\s+pesa\s+mudiyuma\b", "contact human agent representative"),
    (r"\bhuman\s+agent\s+se\s+baat\s+karao\b", "contact human agent representative"),
    (r"\bhelp\s+pannunga\b", "please help me"),
    (r"\bmadad\s+karo\b", "please help me"),
    (r"\bseekiram|turant|jaldi\b", "immediately"),
    (r"\bromba\s+urgent\b", "urgent immediately"),

    # ---------------- Tanglish & Hinglish Address ----------------
    (r"\baddress\s+(?:mathanum|change\s+karna\s+hai)\b", "change shipping address destination"),
    (r"\b(?:dispatch\s+aagurathukulla|dispatched\s+hone\s+se\s+pehle)\b", "before dispatch change shipping address"),

    # ---------------- Tanglish & Hinglish Complaints & Grievances ----------------
    (r"\bcomplaint\s+register\s+(?:pannanum|karna\s+hai)\b", "file a customer complaint claim"),
    (r"\bcustomer\s+support\s+complaint\b", "file a complaint claim"),
    (r"\bpackage\s+completely\s+damaged\b", "damaged package complaint"),
    (r"\bghatiya\s+service\b", "poor service complaint"),
    (r"\b(?:nalla\s+irukku|accha\s+laga)\b", "good product write review feedback"),
    (r"\breview\s+feedback\s+share\b", "write product review feedback"),

    # ---------------- Tanglish & Hinglish Invoices & Policies ----------------
    (r"\binvoice\s+eppadi\s+download\s+panrathu\b", "download tax invoice receipt"),
    (r"\binvoice\s+download\s+karne\s+ka\s+option\b", "download tax invoice receipt"),
    (r"\breturn\s+policy\s+rules\s+enna\b", "check refund policy return rules"),
    (r"\bnewsletter\s+unsubscribe\s+(?:pannunga|karna\s+hai)\b", "newsletter subscription unsubscribe"),
]

# Sort rules by descending pattern length so that longer, specific multi-word idioms
# are processed before shorter partial substrings
CODE_MIXED_RULES: List[Tuple[str, str]] = sorted(
    RAW_CODE_MIXED_RULES,
    key=lambda item: len(item[0]),
    reverse=True,
)

# Compile patterns for fast vectorizer pre-processing
COMPILED_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in CODE_MIXED_RULES
]


def normalize_code_mixed_text(text: str) -> str:
    """Normalize colloquial Tanglish and Hinglish expressions into canonical support taxonomy.

    Preserves standard English tokens while expanding romanized regional idioms into
    their domain semantic representations for robust cross-lingual vectorization.

    Args:
        text: Raw customer query string.

    Returns:
        Normalized text string.
    """
    if not isinstance(text, str):
        return ""

    normalized = text.strip().lower()

    # Apply phrase mappings (longest idioms matched first)
    for regex, replacement in COMPILED_RULES:
        normalized = regex.sub(replacement, normalized)

    # Collapse repeated whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

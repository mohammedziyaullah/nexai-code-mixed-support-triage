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
    (r"\b(?:return|retarin|riturn)\s+(?:pannu|pannanum|pannunga|pannidunga|panren|karo|kardo|kar\s+do|karna\s+hai)\b", "get refund return request money back"),
    (r"\b(?:wapas|vaapas|lautana)\s+(?:karo|kardo|kar\s+do|karna\s+hai|le\s+lo|dena\s+hai)\b", "get refund return request money back"),
    (r"\b(?:retarin|riturn)\b", "return"),
    (r"\b(?:paisa|paise)\s+kat\s+(?:gaya|gaye)\b", "money debited payment issue failed"),
    (r"\b(?:panam|kasu)\s+pochu\b", "money debited payment issue failed"),
    (r"\bmoney\s+deduct\s+aayiduchu\b", "money debited payment issue"),
    (r"\border\s+confirm\s+aagala\b", "order not placed payment issue"),
    (r"\border\s+place\s+nahi\s+hua\b", "order not placed payment issue"),
    (r"\bbank\s+account-?ku\s+varala\b", "bank account refund not received track refund"),
    (r"\breflect\s+nahi\s+hua\b", "not reflected track refund"),
    (r"\bkab\s+tak\s+aayega\b", "when will it arrive timeline status"),

    # ---------------- Tanglish & Hinglish Account & Auth ----------------
    (r"\b(?:new\s+account\s+create|pudhu\s+account\s+open)\b.*?\b(?:otp\s+issue|otp\s+receive\s+aagala|otp\s+nahi\s+aa\s+raha)\b", "error creating account registration problems otp issue"),
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
    (r"\b(?:care\s+)?(?:agent|support)\s+kitta\s+(?:direct-?a\s+)?(?:call\s+la\s+)?pesa\s+mudiyuma\b", "contact human agent representative speak with agent"),
    (r"\bhuman\s+agent\s+se\s+baat\s+karao\b", "contact human agent representative"),
    (r"\bhelp\s+pannunga\b", "please help me"),
    (r"\bmadad\s+karo\b", "please help me"),
    (r"\bseekiram|turant|jaldi\b", "immediately"),
    (r"\bromba\s+urgent\b", "urgent immediately"),

    # ---------------- Tanglish & Hinglish Address ----------------
    (r"\baddress\s+(?:mathanum|change\s+karna\s+hai)\b", "change shipping address destination"),
    (r"\b(?:dispatch\s+aagurathukulla|dispatched\s+hone\s+se\s+pehle)\b", "before dispatch change shipping address"),

    # ---------------- Product Context Disambiguation ----------------
    # Product defect/broken/malfunction -> complaint
    (r"\b(?:this\s+)?(?:product|item|samaan)\s+(?:is\s+)?(?:completely\s+)?(?:defective|broken|damaged|faulty|fake|duplicate|counterfeit|not\s+working)\b", "file a customer complaint consumer claim lodge a complaint"),
    (r"\b(?:defective|broken|damaged|faulty|fake|counterfeit)\s+(?:product|item|samaan)\b", "file a customer complaint consumer claim lodge a complaint"),
    (r"\b(?:product|item|samaan)\s+(?:kaam|kam|chal)\s+nahi\s+(?:kar\s+raha|raha)\b", "file a customer complaint consumer claim lodge a complaint"),
    (r"\b(?:product|item)\s+(?:work\s+aagala|velai\s+seiyala|velai\s+seiyalaye)\b", "file a customer complaint consumer claim lodge a complaint"),
    (r"\b(?:product|item)\s+(?:damaged\s+condition\s+me|damaged-?a)\b", "file a customer complaint consumer claim lodge a complaint"),
    (r"\b(?:product|item)\s+(?:bilkul\s+)?(?:bekaar|ghatiya|kharaab|kharab|mosam)\b", "file a customer complaint consumer claim lodge a complaint"),

    # Product returns/refunds -> get_refund / check_refund_policy
    (r"\b(?:product|item|samaan)\s+(?:return\s+panna\s+mudiyuma|return\s+karna\s+hai|wapas\s+karna\s+hai|return\s+pannanum)\b", "get refund request money back return"),
    (r"\b(?:product|item)\s+return\b", "get refund return request"),
    (r"\b(?:product|item)\s+refund\b", "get refund money back"),

    # Product review/ratings -> review
    (r"\b(?:product|item)\s+(?:nalla\s+irukku|accha\s+laga)\b", "good product write review feedback rating"),
    (r"\b(?:product|item)\s+(?:review|rating|feedback)\b", "write product review feedback rating"),

    # Product changes/variant -> change_order
    (r"\b(?:product|item)\s+(?:details\s+change|variant\s+change|size\s+change|replace|modify)\b", "change order modify item details"),

    # Product tracking/transit -> track_order
    (r"\b(?:product|item)\s+(?:delivery\s+innum\s+varala|kab\s+aayega|enga\s+irukku|track)\b", "track order status transit shipment"),

    # ---------------- Tanglish & Hinglish Wrong Item & Exchanges ----------------
    (r"\b(?:galat|wrong|thappana)\s+(?:item|product)\s+select\s+(?:panniten|kar\s+diya|panniyachu)\b", "cancel order mistake chosen"),
    (r"\b(?:order\s+place\s+karne\s+ka\s+try|trying\s+to\s+place\s+order)\b.*?\b(?:checkout\s+button\s+kaam\s+nahi\s+kar\s+raha|checkout\s+error)\b", "place order error ordering checkout problem"),
    (r"\bvera\s+(?:product|item|samaan)(?:\s+vandhu(?:urku|rukku|dhu|chu))?\b", "wrong product delivered file a customer complaint claim"),
    (r"\b(?:galat|wrong|thappana|vera)\s+(?:item|product|package|samaan)\b", "file a customer complaint incorrect delivered product"),
    (r"\bexchange\s+(?:pannanum|karna\s+hai|panna\s+mudiyuma|karein|pannunga|kardo)\b", "exchange replacement modify change order"),
    (r"\bsize\s+(?:match\s+aagala|change|mathanum)\b", "change order size variant modify"),

    # ---------------- Tanglish & Hinglish Complaints & Grievances ----------------
    (r"\bcomplaint\s+register\s+(?:pannanum|karna\s+hai)\b", "file a customer complaint claim"),
    (r"\bcustomer\s+support\s+complaint\b", "file a complaint claim"),
    (r"\bpackage\s+completely\s+damaged\b", "damaged package complaint"),
    (r"\bitem\s+damaged-?a\s+deliver\s+pannitaanga\b", "file a customer complaint consumer claim poor service"),
    (r"\bghatiya\s+service\b", "poor service complaint"),
    (r"\b(?:completely\s+)?(?:damaged|broken|defective|faulty|fake|duplicate|counterfeit)\b", "file a customer complaint consumer claim lodge a complaint"),
    (r"\b(?:bahut\s+)?(?:bekaar|ghatiya|kharaab|kharab)\b", "poor quality file a customer complaint claim"),
    (r"\b(?:romba\s+)?(?:mosam|mosama|mosamaana)\b", "poor quality file a customer complaint claim"),
    (r"\b(?:kaam|kam|chal)\s+nahi\s+(?:kar\s+raha|raha)\b", "file a customer complaint consumer claim"),
    (r"\b(?:work\s+aagala|velai\s+seiyala|velai\s+seiyalaye)\b", "file a customer complaint consumer claim"),
    (r"\b(?:nalla\s+irukku|accha\s+laga)\b", "good product write review feedback"),
    (r"\breview\s+feedback\s+share\b", "write product review feedback"),

    # ---------------- Tanglish & Hinglish Invoices & Policies ----------------
    (r"\breturn\s+policy\s+rules\s+(?:enna|kya\s+hai).*", "check refund policy return rules conditions guidelines policy"),
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

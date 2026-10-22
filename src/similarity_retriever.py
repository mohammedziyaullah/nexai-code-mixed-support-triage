"""Similarity-based few-shot example retriever for customer support triage.

Supports Official Stretch Goal 3:
- Retrieves top-K verified reference examples from an approved training corpus
  using TF-IDF cosine similarity.
- Strictly held-out: never includes or queries the 35 benchmark test cases.
- Transparent explainability: allows human agents to inspect the closest reference
  queries and verified intent labels that informed the triage pattern.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.normalizer import normalize_code_mixed_text

# Curated reference examples across all customer support intents and languages (English, Hinglish, Tanglish)
# Strictly independent of data/code_mixed_test_cases.json
REFERENCE_EXAMPLES: List[Dict[str, str]] = [
    # Track order
    {"text": "Where is my package right now? Tracking link is not updating", "intent": "track_order", "category": "ORDER"},
    {"text": "Can you track my parcel delivery status please?", "intent": "track_order", "category": "ORDER"},
    {"text": "Mera order abhi kahan pahuncha hai tracking link check karna hai", "intent": "track_order", "category": "ORDER"},
    {"text": "En order delivery enga irukku live transit status theriyanum", "intent": "track_order", "category": "ORDER"},

    # Cancel order
    {"text": "I want to cancel my recent purchase immediately", "intent": "cancel_order", "category": "ORDER"},
    {"text": "Please cancel my order before it gets dispatched from the warehouse", "intent": "cancel_order", "category": "ORDER"},
    {"text": "Yeh order cancel kardo please mujhe ab nahi chahiye", "intent": "cancel_order", "category": "ORDER"},
    {"text": "Order-a cancel pannidunga dispatch panna vendam", "intent": "cancel_order", "category": "ORDER"},

    # Change shipping address
    {"text": "Need to update my delivery destination address before shipping", "intent": "change_shipping_address", "category": "ORDER"},
    {"text": "Can I change the delivery address for order 49201?", "intent": "change_shipping_address", "category": "ORDER"},
    {"text": "Delivery address badalna hai dispatch hone se pehle", "intent": "change_shipping_address", "category": "ORDER"},
    {"text": "Shipping address mathanum pincode thappa potuten", "intent": "change_shipping_address", "category": "ORDER"},

    # Payment issue
    {"text": "Payment was deducted from my account but order was not placed", "intent": "payment_issue", "category": "PAYMENT"},
    {"text": "Money got debited through UPI but checkout shows payment failed", "intent": "payment_issue", "category": "PAYMENT"},
    {"text": "Bank se paise kat gaye par order confirm nahi hua", "intent": "payment_issue", "category": "PAYMENT"},
    {"text": "GPay la panam deduct aayiduchu order place aagala", "intent": "payment_issue", "category": "PAYMENT"},

    # Track refund
    {"text": "When will my refund be credited to my bank account?", "intent": "track_refund", "category": "REFUND"},
    {"text": "I returned the item 5 days ago, where is my refund?", "intent": "track_refund", "category": "REFUND"},
    {"text": "Mera refund abhi tak account me reflect nahi hua status batao", "intent": "track_refund", "category": "REFUND"},
    {"text": "Refund innum bank account-ku credit aagala status enna", "intent": "track_refund", "category": "REFUND"},

    # Complaint
    {"text": "The product I received is defective and completely broken", "intent": "complaint", "category": "FEEDBACK"},
    {"text": "Extremely bad service and damaged packaging delivered today", "intent": "complaint", "category": "FEEDBACK"},
    {"text": "Bahut hi bekaar aur kharaab quality ka product bheja hai", "intent": "complaint", "category": "FEEDBACK"},
    {"text": "Item damaged-a vandhurukku romba mosamaana service", "intent": "complaint", "category": "FEEDBACK"},

    # Contact human agent
    {"text": "Connect me directly with a live human support executive", "intent": "contact_human_agent", "category": "CONTACT"},
    {"text": "I need to speak to an agent on call immediately", "intent": "contact_human_agent", "category": "CONTACT"},
    {"text": "Human agent se baat karao bot se baat nahi karni", "intent": "contact_human_agent", "category": "CONTACT"},
    {"text": "Direct-a support executive kitta pesa mudiyuma call connect pannunga", "intent": "contact_human_agent", "category": "CONTACT"},

    # Recover password
    {"text": "I forgot my account password and cannot log in", "intent": "recover_password", "category": "ACCOUNT"},
    {"text": "Please send a password reset link to my registered email", "intent": "recover_password", "category": "ACCOUNT"},
    {"text": "Password bhool gaya hoon login nahi ho raha reset link bhejo", "intent": "recover_password", "category": "ACCOUNT"},
    {"text": "Password maranthuten login panna mudiyala reset link anupunga", "intent": "recover_password", "category": "ACCOUNT"},

    # Get invoice
    {"text": "How can I download the GST tax invoice for my purchase?", "intent": "get_invoice", "category": "INVOICE"},
    {"text": "Please email me the receipt and invoice copy for tax purposes", "intent": "get_invoice", "category": "INVOICE"},
    {"text": "Invoice download karne ka option kahan milega PDF copy chahiye", "intent": "get_invoice", "category": "INVOICE"},
    {"text": "Bill receipt copy email anupunga tax filing-ku thevai", "intent": "get_invoice", "category": "INVOICE"},

    # Check refund policy
    {"text": "What is your return and refund policy for electronics?", "intent": "check_refund_policy", "category": "REFUND"},
    {"text": "How many days return window do I get after delivery?", "intent": "check_refund_policy", "category": "REFUND"},
    {"text": "Return policy ke rules kya hain kitne din me wapas kar sakte hain", "intent": "check_refund_policy", "category": "REFUND"},
    {"text": "Return policy rules enna eppadi exchange panrathu", "intent": "check_refund_policy", "category": "REFUND"},

    # Change order
    {"text": "I want to add another item to my pending order", "intent": "change_order", "category": "ORDER"},
    {"text": "Can I modify the color or size before it is packed?", "intent": "change_order", "category": "ORDER"},
    {"text": "Order me item quantity update karna hai please modify kardo", "intent": "change_order", "category": "ORDER"},

    # Check payment methods
    {"text": "What payment methods are supported on your website?", "intent": "check_payment_methods", "category": "PAYMENT"},
    {"text": "Can I pay using Cash on Delivery or credit card EMI?", "intent": "check_payment_methods", "category": "PAYMENT"},
    {"text": "Kaun se payment options available hain UPI chalta hai kya", "intent": "check_payment_methods", "category": "PAYMENT"},

    # Delivery period
    {"text": "How many days will it take for standard delivery to arrive?", "intent": "delivery_period", "category": "DELIVERY"},
    {"text": "Standard shipping kitne din me deliver ho jaati hai", "intent": "delivery_period", "category": "DELIVERY"},
    {"text": "En location-ku delivery vara ethana naal aagum", "intent": "delivery_period", "category": "DELIVERY"},
]


class SimilarityExampleRetriever:
    """Retrieves top-K most similar customer support reference examples."""

    def __init__(self, examples: List[Dict[str, str]] = REFERENCE_EXAMPLES):
        self.examples = examples
        self.vectorizer = TfidfVectorizer(
            preprocessor=normalize_code_mixed_text,
            ngram_range=(1, 2),
            sublinear_tf=True,
            lowercase=True,
        )
        self.texts = [ex["text"] for ex in self.examples]
        self.doc_vectors = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top_k most similar reference examples for a query.

        Args:
            query: Customer query text.
            top_k: Number of nearest matches to return.

        Returns:
            List of matching example dicts with similarity score.
        """
        if not isinstance(query, str) or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            ex = self.examples[idx]
            results.append(
                {
                    "text": ex["text"],
                    "intent": ex["intent"],
                    "category": ex["category"],
                    "similarity": round(score, 4),
                }
            )

        return results


# Global singleton instance for efficient zero-startup latency
_GLOBAL_RETRIEVER: SimilarityExampleRetriever | None = None


def get_similarity_retriever() -> SimilarityExampleRetriever:
    """Retrieve global singleton retriever instance."""
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        _GLOBAL_RETRIEVER = SimilarityExampleRetriever()
    return _GLOBAL_RETRIEVER

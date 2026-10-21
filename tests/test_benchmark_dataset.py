"""Tests for validating the curated code-mixed benchmark dataset."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

BENCHMARK_FILE = Path(__file__).resolve().parent.parent / "data" / "code_mixed_test_cases.json"

VALID_INTENTS = {
    "cancel_order",
    "change_order",
    "change_shipping_address",
    "check_cancellation_fee",
    "check_invoice",
    "check_payment_methods",
    "check_refund_policy",
    "complaint",
    "contact_customer_service",
    "contact_human_agent",
    "create_account",
    "delete_account",
    "delivery_options",
    "delivery_period",
    "edit_account",
    "get_invoice",
    "get_refund",
    "newsletter_subscription",
    "payment_issue",
    "place_order",
    "recover_password",
    "registration_problems",
    "review",
    "set_up_shipping_address",
    "switch_account",
    "track_order",
    "track_refund",
}

VALID_CATEGORIES = {
    "ACCOUNT",
    "CANCEL",
    "CONTACT",
    "DELIVERY",
    "FEEDBACK",
    "INVOICE",
    "ORDER",
    "PAYMENT",
    "REFUND",
    "SHIPPING",
    "SUBSCRIPTION",
}

VALID_URGENCIES = {"HIGH", "MEDIUM", "LOW"}
VALID_LANGUAGES = {"Tanglish", "Hinglish"}


class TestBenchmarkDataset(unittest.TestCase):
    """Test suite ensuring integrity and schema compliance of code-mixed test cases."""

    def setUp(self):
        self.assertTrue(
            BENCHMARK_FILE.is_file(),
            f"Benchmark dataset file does not exist at {BENCHMARK_FILE}",
        )
        with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
            self.test_cases = json.load(f)

    def test_benchmark_contains_at_least_30_items(self):
        """Mandatory requirement: at least 30 team-curated test cases."""
        self.assertGreaterEqual(
            len(self.test_cases),
            30,
            f"Expected at least 30 test cases, found {len(self.test_cases)}",
        )

    def test_benchmark_schema_and_valid_values(self):
        """Verify each item adheres to expected schema, categories, intents, and urgencies."""
        seen_ids = set()
        seen_messages = set()

        for idx, item in enumerate(self.test_cases):
            with self.subTest(item_index=idx, item_id=item.get("id")):
                # Key presence
                self.assertIn("id", item)
                self.assertIn("language", item)
                self.assertIn("message", item)
                self.assertIn("expected_intent", item)
                self.assertIn("expected_category", item)
                self.assertIn("expected_urgency", item)

                # Uniqueness
                self.assertNotIn(item["id"], seen_ids, f"Duplicate id: {item['id']}")
                seen_ids.add(item["id"])

                self.assertNotIn(
                    item["message"].strip().lower(),
                    seen_messages,
                    f"Duplicate message found: {item['message']}",
                )
                seen_messages.add(item["message"].strip().lower())

                # Value domains
                self.assertIn(item["language"], VALID_LANGUAGES)
                self.assertIn(
                    item["expected_intent"],
                    VALID_INTENTS,
                    f"Invalid intent: {item['expected_intent']}",
                )
                self.assertIn(
                    item["expected_category"],
                    VALID_CATEGORIES,
                    f"Invalid category: {item['expected_category']}",
                )
                self.assertIn(
                    item["expected_urgency"],
                    VALID_URGENCIES,
                    f"Invalid urgency: {item['expected_urgency']}",
                )
                self.assertGreater(
                    len(item["message"].strip()),
                    10,
                    f"Message too short: '{item['message']}'",
                )


if __name__ == "__main__":
    unittest.main()

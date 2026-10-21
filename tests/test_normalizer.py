"""Unit tests for the code-mixed normalizer module."""

from __future__ import annotations

import unittest
from src.normalizer import normalize_code_mixed_text


class TestNormalizer(unittest.TestCase):
    """Test suite for Tanglish and Hinglish phrase normalizations."""

    def test_normalize_tanglish_order_arrival(self):
        query = "My order innum varala, please help pannunga!"
        normalized = normalize_code_mixed_text(query)
        self.assertIn("not arrived", normalized)
        self.assertIn("track order", normalized)
        self.assertIn("please help me", normalized)

    def test_normalize_hinglish_order_arrival(self):
        query = "Mera order abhi tak deliver nahi hua"
        normalized = normalize_code_mixed_text(query)
        self.assertIn("my order", normalized)
        self.assertIn("not arrived track order", normalized)

    def test_normalize_payment_issue_tanglish(self):
        query = "Bank account-la irunthu money deduct aayiduchu panam pochu"
        normalized = normalize_code_mixed_text(query)
        self.assertIn("money debited payment issue", normalized)

    def test_normalize_english_passthrough(self):
        query = "Where can I download the PDF invoice?"
        normalized = normalize_code_mixed_text(query)
        # English query should remain essentially intact (lowercased/stripped)
        self.assertIn("download", normalized)
        self.assertIn("invoice", normalized)

    def test_normalize_non_string_input(self):
        self.assertEqual(normalize_code_mixed_text(None), "")
        self.assertEqual(normalize_code_mixed_text(123), "")


if __name__ == "__main__":
    unittest.main()

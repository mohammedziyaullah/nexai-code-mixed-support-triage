"""Unit tests for src/data_loader.py using synthetic DataFrames without network access."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

from src.data_loader import (
    find_matching_column,
    inspect_dataset_structure,
    load_bitext_dataset,
    validate_and_clean_data,
    BITEXT_DATASET_ID,
)


class TestDataLoader(unittest.TestCase):
    """Test suite for data loading, inspection, validation, and cleaning."""

    def test_find_matching_column(self):
        columns = ["Flags", "Instruction", "Category", "Intent", "Response"]
        self.assertEqual(
            find_matching_column(columns, ["instruction", "message"]),
            "Instruction",
        )
        self.assertEqual(
            find_matching_column(columns, ["category", "topic"]),
            "Category",
        )
        self.assertIsNone(find_matching_column(columns, ["nonexistent_field"]))

    def test_validate_and_clean_data_bitext_schema(self):
        """Verify handling of actual Bitext columns (instruction, category, intent, response, flags)."""
        synthetic_data = pd.DataFrame(
            {
                "flags": ["B", "Q", "P"],
                "instruction": [
                    "How do I cancel my order? ",
                    "Where is my refund?  ",
                    "I want to change my delivery address",
                ],
                "category": ["order", "refund", "delivery"],
                "intent": ["cancel_order", "track_refund", "change_address"],
                "response": [
                    "You can cancel your order in your account.",
                    "Refunds typically take 3-5 days.",
                    "Address changes are supported before dispatch.",
                ],
            }
        )

        clean_df = validate_and_clean_data(synthetic_data)

        # Expected standardized columns
        self.assertIn("message", clean_df.columns)
        self.assertIn("category", clean_df.columns)
        self.assertIn("intent", clean_df.columns)
        self.assertIn("response", clean_df.columns)
        self.assertNotIn("flags", clean_df.columns)

        # Verify values and whitespace stripping
        self.assertEqual(len(clean_df), 3)
        self.assertEqual(clean_df.iloc[0]["message"], "How do I cancel my order?")
        self.assertEqual(clean_df.iloc[1]["message"], "Where is my refund?")
        self.assertEqual(clean_df.iloc[0]["category"], "order")
        self.assertEqual(clean_df.iloc[0]["intent"], "cancel_order")

    def test_validate_and_clean_data_alternative_schema(self):
        """Verify dynamic field resolution when columns have alternative valid names."""
        synthetic_data = pd.DataFrame(
            {
                "text": ["App crashed when making payment"],
                "intent_category": ["technical"],
                "label": ["app_crash"],
            }
        )

        clean_df = validate_and_clean_data(synthetic_data)

        self.assertEqual(list(clean_df.columns), ["message", "category", "intent"])
        self.assertEqual(clean_df.iloc[0]["message"], "App crashed when making payment")
        self.assertEqual(clean_df.iloc[0]["category"], "technical")
        self.assertEqual(clean_df.iloc[0]["intent"], "app_crash")

    def test_blank_and_whitespace_cleaning(self):
        """Verify blank messages, pure whitespace, and empty strings are cleanly dropped."""
        synthetic_data = pd.DataFrame(
            {
                "instruction": [
                    "Valid customer inquiry",
                    "   ",
                    "",
                    "Another valid inquiry",
                ],
                "category": ["support", "support", "billing", "billing"],
                "intent": ["help", "help", "invoice", "invoice"],
            }
        )

        clean_df = validate_and_clean_data(synthetic_data)

        self.assertEqual(len(clean_df), 2)
        self.assertEqual(
            list(clean_df["message"]),
            ["Valid customer inquiry", "Another valid inquiry"],
        )

    def test_template_placeholder_filtering(self):
        """Verify that obvious template placeholders are filtered out."""
        synthetic_data = pd.DataFrame(
            {
                "instruction": [
                    "Valid query about subscription",
                    "{{placeholder}}",
                    "[INSERT_TEXT]",
                    "<customer_query>",
                    "N/A",
                    "none",
                    "Real inquiry here",
                ],
                "category": [
                    "subscription",
                    "subscription",
                    "general",
                    "general",
                    "general",
                    "general",
                    "subscription",
                ],
                "intent": [
                    "cancel_sub",
                    "cancel_sub",
                    "general_query",
                    "general_query",
                    "general_query",
                    "general_query",
                    "renew_sub",
                ],
            }
        )

        clean_df = validate_and_clean_data(synthetic_data)

        self.assertEqual(len(clean_df), 2)
        self.assertEqual(
            list(clean_df["message"]),
            ["Valid query about subscription", "Real inquiry here"],
        )

    def test_missing_required_columns_raises_error(self):
        """Verify clear and descriptive ValueError when required fields are missing."""
        invalid_data = pd.DataFrame(
            {
                "response": ["Thank you for reaching out."],
                "flags": ["polite"],
            }
        )

        with self.assertRaises(ValueError) as ctx:
            validate_and_clean_data(invalid_data)

        error_message = str(ctx.exception)
        self.assertIn("Dataset validation failed: missing required semantic field(s)", error_message)
        self.assertIn("message/text", error_message)
        self.assertIn("category", error_message)
        self.assertIn("intent", error_message)
        self.assertIn("Available columns in dataset: ['response', 'flags']", error_message)

    def test_inspect_dataset_structure_df(self):
        """Verify dataset structure inspection for a DataFrame."""
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        structure = inspect_dataset_structure(df)

        self.assertEqual(structure["splits"], ["default"])
        self.assertEqual(structure["columns"]["default"], ["col1", "col2"])
        self.assertEqual(structure["row_counts"]["default"], 2)

    def test_inspect_dataset_structure_hf_mock(self):
        """Verify dataset structure inspection for mock Hugging Face DatasetDict."""
        mock_train = MagicMock()
        mock_train.column_names = ["instruction", "category", "intent"]
        mock_train.__len__.return_value = 100

        mock_test = MagicMock()
        mock_test.column_names = ["instruction", "category", "intent"]
        mock_test.__len__.return_value = 20

        mock_dataset_dict = {
            "train": mock_train,
            "test": mock_test,
        }

        structure = inspect_dataset_structure(mock_dataset_dict)
        self.assertEqual(set(structure["splits"]), {"train", "test"})
        self.assertEqual(structure["columns"]["train"], ["instruction", "category", "intent"])
        self.assertEqual(structure["row_counts"]["train"], 100)
        self.assertEqual(structure["row_counts"]["test"], 20)

    def test_load_bitext_dataset_with_mocked_hf(self):
        """Verify load_bitext_dataset functionality using a mock HF loader without internet access."""
        synthetic_raw = pd.DataFrame(
            {
                "flags": ["B"],
                "instruction": ["How to return an item?"],
                "category": ["refund"],
                "intent": ["return_item"],
                "response": ["Items can be returned within 30 days."],
            }
        )

        mock_split = MagicMock()
        mock_split.column_names = list(synthetic_raw.columns)
        mock_split.to_pandas.return_value = synthetic_raw
        mock_split.__len__.return_value = len(synthetic_raw)

        mock_dataset_dict = MagicMock()
        mock_dataset_dict.keys.return_value = ["train"]
        mock_dataset_dict.__getitem__.side_effect = lambda key: mock_split

        mock_datasets_module = MagicMock()
        mock_datasets_module.load_dataset.return_value = mock_dataset_dict

        with patch.dict(sys.modules, {"datasets": mock_datasets_module}):
            df = load_bitext_dataset(BITEXT_DATASET_ID)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["message"], "How to return an item?")
            self.assertEqual(df.iloc[0]["category"], "refund")
            self.assertEqual(df.iloc[0]["intent"], "return_item")
            mock_datasets_module.load_dataset.assert_called_once_with(BITEXT_DATASET_ID)

    def test_load_bitext_dataset_missing_dependency_raises_import_error(self):
        """Verify that an informative ImportError is raised if 'datasets' is not installed."""
        with patch.dict(sys.modules, {"datasets": None}):
            with self.assertRaises(ImportError) as ctx:
                load_bitext_dataset()
            self.assertIn("pip install datasets", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

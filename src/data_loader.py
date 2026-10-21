"""Dataset loader and validator for Code-Mixed Support Triage Assistant.

Loads, inspects, validates, and cleans customer support datasets,
specifically the Bitext Customer Support dataset from Hugging Face.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional
import pandas as pd

BITEXT_DATASET_ID: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

# Candidate column names across common customer support datasets
MESSAGE_CANDIDATES: list[str] = [
    "instruction",
    "message",
    "text",
    "query",
    "customer_message",
    "prompt",
]
CATEGORY_CANDIDATES: list[str] = [
    "category",
    "intent_category",
    "topic",
    "domain",
]
INTENT_CANDIDATES: list[str] = [
    "intent",
    "label",
    "sub_intent",
    "action",
]
RESPONSE_CANDIDATES: list[str] = [
    "response",
    "reply",
    "bot_response",
    "agent_response",
    "target",
]

# Regex to detect obvious template placeholders or filler messages
PLACEHOLDER_REGEX = re.compile(
    r"^(?:\[[\w\s_-]*\]|\{\{[\w\s_-]*\}\}|<[\w\s_-]*>|n/a|null|none|placeholder|\.\.\.|-)$",
    re.IGNORECASE,
)


def find_matching_column(
    columns: Iterable[str], candidates: list[str]
) -> Optional[str]:
    """Find the first matching column name from a list of candidates (case-insensitive)."""
    col_lookup = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        candidate_key = candidate.strip().lower()
        if candidate_key in col_lookup:
            return col_lookup[candidate_key]
    return None


def inspect_dataset_structure(dataset_obj: Any) -> Dict[str, Any]:
    """Inspect the splits, column names, and row counts of a dataset object.

    Args:
        dataset_obj: Hugging Face DatasetDict, Dataset, or pandas DataFrame.

    Returns:
        Dict containing detected splits, columns per split, and row counts.
    """
    summary: Dict[str, Any] = {
        "splits": [],
        "columns": {},
        "row_counts": {},
    }

    if isinstance(dataset_obj, pd.DataFrame):
        summary["splits"] = ["default"]
        summary["columns"]["default"] = list(dataset_obj.columns)
        summary["row_counts"]["default"] = len(dataset_obj)
        return summary

    # Hugging Face DatasetDict
    if hasattr(dataset_obj, "keys") and callable(getattr(dataset_obj, "keys")):
        summary["splits"] = list(dataset_obj.keys())
        for split in summary["splits"]:
            split_ds = dataset_obj[split]
            cols = getattr(split_ds, "column_names", [])
            summary["columns"][split] = list(cols)
            summary["row_counts"][split] = len(split_ds) if hasattr(split_ds, "__len__") else None
        return summary

    # Hugging Face single Dataset
    if hasattr(dataset_obj, "column_names"):
        summary["splits"] = ["default"]
        summary["columns"]["default"] = list(dataset_obj.column_names)
        summary["row_counts"]["default"] = len(dataset_obj) if hasattr(dataset_obj, "__len__") else None
        return summary

    raise TypeError(
        f"Unsupported dataset object type '{type(dataset_obj).__name__}'. "
        "Expected pandas DataFrame, Hugging Face DatasetDict, or Dataset."
    )


def validate_and_clean_data(
    df: pd.DataFrame,
    message_col: Optional[str] = None,
    category_col: Optional[str] = None,
    intent_col: Optional[str] = None,
    response_col: Optional[str] = None,
) -> pd.DataFrame:
    """Validate dataset fields, map to standard schema, and clean invalid rows.

    Inspects actual columns to resolve customer message, category, and intent fields.
    Does not invent labels or synthetic data.

    Args:
        df: Input pandas DataFrame.
        message_col: Specific column name for customer message (optional).
        category_col: Specific column name for category (optional).
        intent_col: Specific column name for intent (optional).
        response_col: Specific column name for response (optional).

    Returns:
        Cleaned pandas DataFrame with standardized column names:
        'message', 'category', 'intent' (and 'response' if present in input).

    Raises:
        ValueError: If required message, category, or intent columns cannot be found.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, received {type(df).__name__}.")

    available_cols = list(df.columns)

    # Resolve message column
    resolved_message_col = (
        message_col if (message_col and message_col in df.columns)
        else find_matching_column(available_cols, MESSAGE_CANDIDATES)
    )

    # Resolve category column
    resolved_category_col = (
        category_col if (category_col and category_col in df.columns)
        else find_matching_column(available_cols, CATEGORY_CANDIDATES)
    )

    # Resolve intent column
    resolved_intent_col = (
        intent_col if (intent_col and intent_col in df.columns)
        else find_matching_column(available_cols, INTENT_CANDIDATES)
    )

    # Resolve optional response column
    resolved_response_col = (
        response_col if (response_col and response_col in df.columns)
        else find_matching_column(available_cols, RESPONSE_CANDIDATES)
    )

    # Verify that all mandatory fields exist
    missing_fields: List[str] = []
    if not resolved_message_col:
        missing_fields.append(f"message/text (searched candidates: {MESSAGE_CANDIDATES})")
    if not resolved_category_col:
        missing_fields.append(f"category (searched candidates: {CATEGORY_CANDIDATES})")
    if not resolved_intent_col:
        missing_fields.append(f"intent (searched candidates: {INTENT_CANDIDATES})")

    if missing_fields:
        raise ValueError(
            f"Dataset validation failed: missing required semantic field(s):\n"
            f"  - " + "\n  - ".join(missing_fields) + "\n"
            f"Available columns in dataset: {available_cols}."
        )

    # Select only mapped columns that actually exist
    cols_to_extract = {
        "message": resolved_message_col,
        "category": resolved_category_col,
        "intent": resolved_intent_col,
    }
    if resolved_response_col:
        cols_to_extract["response"] = resolved_response_col

    clean_df = pd.DataFrame()
    for standard_col, src_col in cols_to_extract.items():
        # Strip leading and trailing whitespace from string data
        clean_df[standard_col] = df[src_col].astype(str).str.strip()

    # Drop null / NaN rows
    clean_df = clean_df.dropna(subset=["message", "category", "intent"])

    # Drop blank or empty strings
    clean_df = clean_df[
        (clean_df["message"] != "")
        & (clean_df["category"] != "")
        & (clean_df["intent"] != "")
    ]

    # Filter out obvious template placeholders (e.g. {{placeholder}}, [INSERT_TEXT], N/A)
    clean_df = clean_df[~clean_df["message"].str.match(PLACEHOLDER_REGEX)]
    clean_df = clean_df[~clean_df["category"].str.match(PLACEHOLDER_REGEX)]
    clean_df = clean_df[~clean_df["intent"].str.match(PLACEHOLDER_REGEX)]

    # Reset index cleanly
    clean_df = clean_df.reset_index(drop=True)

    return clean_df


def load_bitext_dataset(
    dataset_name: str = BITEXT_DATASET_ID,
    split: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load the official Bitext customer support dataset from Hugging Face and return a cleaned DataFrame.

    Inspects available splits and columns dynamically before extraction.
    Does not assume hardcoded field names.

    Args:
        dataset_name: Hugging Face dataset identifier.
        split: Dataset split to load (e.g., 'train'). If None, dynamically chooses 'train'
               or the first available split.
        **kwargs: Additional keyword arguments passed to `datasets.load_dataset`.

    Returns:
        Cleaned pandas DataFrame containing 'message', 'category', 'intent'
        (and 'response' if present in the source dataset).

    Raises:
        ImportError: If the Hugging Face `datasets` library is not installed.
        ValueError: If the dataset cannot be inspected or validated.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The Hugging Face 'datasets' package is required to load the Bitext dataset.\n"
            "Install dependencies with: pip install datasets"
        ) from exc

    # Load dataset
    if split:
        raw_data = load_dataset(dataset_name, split=split, **kwargs)
    else:
        raw_data = load_dataset(dataset_name, **kwargs)

    # Inspect dataset structure
    structure = inspect_dataset_structure(raw_data)

    # Determine split to convert to DataFrame
    if hasattr(raw_data, "keys") and callable(getattr(raw_data, "keys")):
        available_splits = structure["splits"]
        target_split = split if split in available_splits else ("train" if "train" in available_splits else available_splits[0])
        df = raw_data[target_split].to_pandas()
    elif hasattr(raw_data, "to_pandas"):
        df = raw_data.to_pandas()
    elif isinstance(raw_data, pd.DataFrame):
        df = raw_data
    else:
        df = pd.DataFrame(raw_data)

    return validate_and_clean_data(df)

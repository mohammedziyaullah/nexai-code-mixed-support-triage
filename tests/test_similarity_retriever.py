"""Unit tests for src/similarity_retriever.py."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from src.similarity_retriever import (
    REFERENCE_EXAMPLES,
    SimilarityExampleRetriever,
    get_similarity_retriever,
)
from src.triage_service import DEFAULT_BENCHMARK_PATH


class TestSimilarityRetriever(unittest.TestCase):
    """Test suite for few-shot similarity retrieval and dataset leakage prevention."""

    def test_benchmark_data_leakage_prevention(self):
        """Crucial test: ensure zero reference examples overlap with the 35 benchmark test cases."""
        with open(DEFAULT_BENCHMARK_PATH, "r", encoding="utf-8") as f:
            benchmarks = json.load(f)

        bench_texts = {b["message"].strip().lower() for b in benchmarks}
        ref_texts = {r["text"].strip().lower() for r in REFERENCE_EXAMPLES}

        overlap = bench_texts.intersection(ref_texts)
        self.assertEqual(
            len(overlap),
            0,
            f"Data leakage detected! Reference corpus overlaps with benchmark: {overlap}",
        )

    def test_retriever_query_top_k(self):
        """Verify retrieval returns requested top_k count ordered by similarity descending."""
        retriever = get_similarity_retriever()

        results = retriever.retrieve("En order innum varala da", top_k=3)
        self.assertEqual(len(results), 3)

        # First result should have highest similarity
        self.assertGreaterEqual(results[0]["similarity"], results[1]["similarity"])
        self.assertGreaterEqual(results[1]["similarity"], results[2]["similarity"])
        self.assertIn("track_order", results[0]["intent"])

    def test_retriever_blank_or_invalid_query(self):
        """Verify blank or invalid input returns empty list without error."""
        retriever = get_similarity_retriever()
        self.assertEqual(retriever.retrieve(""), [])
        self.assertEqual(retriever.retrieve("   "), [])
        self.assertEqual(retriever.retrieve(None), [])


if __name__ == "__main__":
    unittest.main()

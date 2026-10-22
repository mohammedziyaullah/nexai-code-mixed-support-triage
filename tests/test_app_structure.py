"""Smoke tests to verify project scaffolding and expected file structure."""

from pathlib import Path
import unittest

EXPECTED_PROJECT_FILES = [
    "README.md",
    ".gitignore",
    ".env.example",
    "DEV_LOG.md",
    "requirements.txt",
    "app.py",
    "src/__init__.py",
    "src/reply_generator.py",
    "src/similarity_retriever.py",
    "src/train_multilingual.py",
    "src/train_61_intents.py",
    "tests/__init__.py",
    "data/README.md",
    "data/intent_expansion_proposal.md",
    "models/README.md",
    "models/candidate_comparison_report.json",
    "models/candidate_multilingual_eval_report.json",
    "models/intent_classifier_61.joblib",
    "models/classifier_61_eval_report.json",
]


class TestProjectStructure(unittest.TestCase):
    """Verify that all core scaffold files exist in the project root."""

    def test_expected_files_exist(self):
        root_dir = Path(__file__).resolve().parent.parent
        for rel_path in EXPECTED_PROJECT_FILES:
            target_path = root_dir / rel_path
            self.assertTrue(
                target_path.is_file(),
                f"Expected project file is missing: {rel_path}",
            )


if __name__ == "__main__":
    unittest.main()

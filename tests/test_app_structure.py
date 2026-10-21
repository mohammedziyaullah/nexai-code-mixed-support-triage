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
    "tests/__init__.py",
    "data/README.md",
    "models/README.md",
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

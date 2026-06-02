import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_addon import build_addon  # noqa: E402


class BuildAddonTests(unittest.TestCase):
    def test_zip_contains_expected_package_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = build_addon(REPO_ROOT, dist_dir=tmp)
            self.assertTrue(zip_path.exists(), zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                names = set(zf.namelist())
        expected = [
            "open_jaywalker/__init__.py",
            "open_jaywalker/operators.py",
            "open_jaywalker/ui.py",
            "open_jaywalker/state.py",
            "open_jaywalker/pipeline/workflow.py",
            "open_jaywalker/pipeline/plan_summary.py",
            "open_jaywalker/phase3_classifier/classifier.py",
            "open_jaywalker/armature_inspector/inspector.py",
            "open_jaywalker/asam_human_builder/build_runner.py",
            "open_jaywalker/pipeline_paths.py",
        ]
        for name in expected:
            self.assertIn(name, names, name)
        self.assertFalse(
            any("__pycache__" in n or n.endswith(".pyc") for n in names),
            "bundle must not contain bytecode",
        )

    def test_returns_versioned_zip_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = build_addon(REPO_ROOT, dist_dir=tmp)
            self.assertTrue(zip_path.name.startswith("open_jaywalker-"))
            self.assertTrue(zip_path.name.endswith(".zip"))


if __name__ == "__main__":
    unittest.main()

import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline_paths import (  # noqa: E402
    ENV_VAR,
    MissingPrerequisiteError,
    require_asset_inputs,
    require_inspector_exports,
    resolve_asset_dir,
    resolve_output_root,
)


class PipelinePathsTests(unittest.TestCase):
    def setUp(self):
        self._original_env = os.environ.pop(ENV_VAR, None)

    def tearDown(self):
        if self._original_env is not None:
            os.environ[ENV_VAR] = self._original_env
        else:
            os.environ.pop(ENV_VAR, None)

    def test_resolve_output_root_defaults_to_repo_output_dir(self):
        self.assertEqual(resolve_output_root(), (REPO_ROOT / "output").resolve())

    def test_resolve_output_root_honors_env_override(self):
        override = tempfile.mkdtemp(prefix="pipeline_paths_test_")
        os.environ[ENV_VAR] = override
        self.assertEqual(resolve_output_root(), Path(override).resolve())

    def test_resolve_output_root_expands_user_home(self):
        os.environ[ENV_VAR] = "~"
        self.assertEqual(resolve_output_root(), Path("~").expanduser().resolve())

    def test_resolve_asset_dir_appends_asset_name(self):
        override = tempfile.mkdtemp(prefix="pipeline_paths_test_")
        os.environ[ENV_VAR] = override
        result = resolve_asset_dir("SomeAsset")
        self.assertEqual(result, (Path(override) / "SomeAsset").resolve())

    def test_resolve_asset_dir_creates_directory(self):
        override = tempfile.mkdtemp(prefix="pipeline_paths_test_")
        os.environ[ENV_VAR] = override
        result = resolve_asset_dir("NewAsset")
        self.assertTrue(result.is_dir())


class RequireAssetInputsTests(unittest.TestCase):
    """Shared prerequisite checks used by the classifier/builder CLI entrypoints (#27)."""

    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="prereq_test_"))

    def test_returns_resolved_dir_when_all_files_present(self):
        asset_dir = self._tmp_dir()
        (asset_dir / "classifier_report.json").write_text("{}", encoding="utf-8")
        (asset_dir / "build_plan.json").write_text("{}", encoding="utf-8")
        result = require_asset_inputs(
            asset_dir, ["classifier_report.json", "build_plan.json"], "phase 3 classifier"
        )
        self.assertEqual(result, asset_dir.resolve())

    def test_missing_build_plan_names_the_file_and_gives_hint(self):
        asset_dir = self._tmp_dir()
        (asset_dir / "classifier_report.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(MissingPrerequisiteError) as ctx:
            require_asset_inputs(
                asset_dir, ["classifier_report.json", "build_plan.json"], "phase 3 classifier"
            )
        message = str(ctx.exception)
        self.assertIn("build_plan.json", message)
        self.assertNotIn("classifier_report.json", message)  # present file not listed
        self.assertIn("--asset-dir", message)  # remediation hint
        self.assertIn("phase 3 classifier", message)

    def test_nonexistent_dir_reports_directory_not_found(self):
        missing = self._tmp_dir() / "does_not_exist"
        with self.assertRaises(MissingPrerequisiteError) as ctx:
            require_asset_inputs(missing, ["build_plan.json"], "phase 3 classifier")
        self.assertIn("Asset directory not found", str(ctx.exception))


class RequireInspectorExportsTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="prereq_inspector_test_"))

    def test_returns_resolved_dir_when_export_present(self):
        asset_dir = self._tmp_dir()
        (asset_dir / "Armature_all.json").write_text("{}", encoding="utf-8")
        self.assertEqual(require_inspector_exports(asset_dir), asset_dir.resolve())

    def test_accepts_filtered_export(self):
        asset_dir = self._tmp_dir()
        (asset_dir / "Armature_filtered.json").write_text("{}", encoding="utf-8")
        self.assertEqual(require_inspector_exports(asset_dir), asset_dir.resolve())

    def test_empty_dir_raises_with_inspector_hint(self):
        asset_dir = self._tmp_dir()
        with self.assertRaises(MissingPrerequisiteError) as ctx:
            require_inspector_exports(asset_dir)
        self.assertIn("inspector", str(ctx.exception))

    def test_nonexistent_dir_reports_directory_not_found(self):
        missing = self._tmp_dir() / "nope"
        with self.assertRaises(MissingPrerequisiteError) as ctx:
            require_inspector_exports(missing)
        self.assertIn("Asset directory not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.workflow import run_combined_workflow  # noqa: E402


class PipelineWorkflowTests(unittest.TestCase):
    def test_runs_classifier_after_exports_exist(self):
        temp_root = Path(tempfile.mkdtemp(prefix="pipeline_workflow_"))
        asset_dir = temp_root / "SavedAsset"
        asset_dir.mkdir(parents=True, exist_ok=True)
        classifier_calls = []
        summary_calls = []

        def inspect_scene_fn():
            return {
                "source_file": "C:/assets/SavedAsset.blend",
                "output_dir": str(asset_dir),
                "armature_count": 1,
                "armature_names": ["Rig"],
                "exported_files": [str(asset_dir / "Rig_all.json"), str(asset_dir / "Rig_filtered.json")],
                "diagnostics_ran": False,
            }

        def classify_fn(path):
            classifier_calls.append(path)
            return (
                {"recommended_primary_armature": "Rig"},
                {"root_resolutions": [{"mode": "reuse_existing_root"}]},
                path / "phase3_classification.json",
                path / "build_plan.json",
            )

        def print_summary_fn(report, build_plan, report_path, plan_path):
            summary_calls.append((report, build_plan, report_path, plan_path))

        result = run_combined_workflow(inspect_scene_fn, classify_fn, print_summary_fn)

        self.assertEqual(classifier_calls, [asset_dir.resolve()])
        self.assertEqual(len(summary_calls), 1)
        self.assertEqual(result["classifier_report"]["recommended_primary_armature"], "Rig")

    def test_skips_classifier_when_diagnostics_run(self):
        classifier_calls = []

        def inspect_scene_fn():
            return {
                "source_file": "(unsaved)",
                "output_dir": str(Path(tempfile.mkdtemp(prefix="pipeline_diag_")) / "unsaved"),
                "armature_count": 0,
                "armature_names": [],
                "exported_files": [],
                "diagnostics_ran": True,
            }

        def classify_fn(path):
            classifier_calls.append(path)
            raise AssertionError("classifier should not run")

        def print_summary_fn(report, report_path):
            raise AssertionError("summary should not print")

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            result = run_combined_workflow(inspect_scene_fn, classify_fn, print_summary_fn)

        self.assertEqual(classifier_calls, [])
        self.assertIsNone(result["classifier_report"])
        self.assertIn("Skipping Phase 3 classifier", buffer.getvalue())

    def test_warns_when_classifying_unsaved_output_folder(self):
        temp_root = Path(tempfile.mkdtemp(prefix="pipeline_unsaved_"))
        asset_dir = temp_root / "unsaved"
        asset_dir.mkdir(parents=True, exist_ok=True)

        def inspect_scene_fn():
            return {
                "source_file": "(unsaved)",
                "output_dir": str(asset_dir),
                "armature_count": 1,
                "armature_names": ["Rig"],
                "exported_files": [str(asset_dir / "Rig_all.json"), str(asset_dir / "Rig_filtered.json")],
                "diagnostics_ran": False,
            }

        def classify_fn(path):
            return (
                {"recommended_primary_armature": "Rig"},
                {"root_resolutions": [{"mode": "reuse_existing_root"}]},
                path / "phase3_classification.json",
                path / "build_plan.json",
            )

        def print_summary_fn(report, build_plan, report_path, plan_path):
            return None

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            run_combined_workflow(inspect_scene_fn, classify_fn, print_summary_fn)

        self.assertIn("output/unsaved", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()

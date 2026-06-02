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

from pipeline.workflow import run_combined_workflow, run_full_pipeline  # noqa: E402


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


class RunFullPipelineTests(unittest.TestCase):
    def _inspect_with_plan(self, asset_dir):
        def inspect_scene_fn():
            return {
                "source_file": "C:/assets/SavedAsset.blend",
                "output_dir": str(asset_dir),
                "armature_count": 1,
                "armature_names": ["Rig"],
                "exported_files": [
                    str(asset_dir / "Rig_all.json"),
                    str(asset_dir / "Rig_filtered.json"),
                ],
                "diagnostics_ran": False,
            }
        return inspect_scene_fn

    def _classify(self):
        def classify_fn(path):
            return (
                {"recommended_primary_armature": "Rig"},
                {"root_resolutions": [{"mode": "reuse_existing_root"}]},
                path / "phase3_classification.json",
                path / "build_plan.json",
            )
        return classify_fn

    def test_builds_when_confirmed(self):
        asset_dir = Path(tempfile.mkdtemp(prefix="full_pipeline_")) / "SavedAsset"
        asset_dir.mkdir(parents=True, exist_ok=True)
        build_calls = []

        result = run_full_pipeline(
            inspect_scene_fn=self._inspect_with_plan(asset_dir),
            classify_fn=self._classify(),
            print_summary_fn=lambda *a: None,
            confirm_fn=lambda asset, plan: True,
            build_fn=lambda asset: build_calls.append(asset) or {"built": True},
        )

        self.assertEqual(build_calls, [asset_dir.resolve()])
        self.assertEqual(result["build_result"], {"built": True})

    def test_does_not_build_when_declined(self):
        asset_dir = Path(tempfile.mkdtemp(prefix="full_pipeline_decline_")) / "SavedAsset"
        asset_dir.mkdir(parents=True, exist_ok=True)
        build_calls = []

        result = run_full_pipeline(
            inspect_scene_fn=self._inspect_with_plan(asset_dir),
            classify_fn=self._classify(),
            print_summary_fn=lambda *a: None,
            confirm_fn=lambda asset, plan: False,
            build_fn=lambda asset: build_calls.append(asset),
        )

        self.assertEqual(build_calls, [])
        self.assertIsNone(result["build_result"])

    def test_skips_build_and_confirm_when_no_plan(self):
        confirm_calls = []
        build_calls = []

        def inspect_scene_fn():
            return {
                "source_file": "(unsaved)",
                "output_dir": str(Path(tempfile.mkdtemp(prefix="full_no_plan_")) / "unsaved"),
                "armature_count": 0,
                "armature_names": [],
                "exported_files": [],
                "diagnostics_ran": True,
            }

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            result = run_full_pipeline(
                inspect_scene_fn=inspect_scene_fn,
                classify_fn=lambda path: (_ for _ in ()).throw(AssertionError("classify ran")),
                print_summary_fn=lambda *a: None,
                confirm_fn=lambda asset, plan: confirm_calls.append(True) or True,
                build_fn=lambda asset: build_calls.append(asset),
            )

        self.assertEqual(confirm_calls, [])
        self.assertEqual(build_calls, [])
        self.assertNotIn("build_result", result)


if __name__ == "__main__":
    unittest.main()

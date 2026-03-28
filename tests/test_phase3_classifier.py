import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase3_classifier.classifier import write_asset_report  # noqa: E402


FIXTURE_ROOT = REPO_ROOT / "src" / "armature_inspector" / "output"
CLI_PATH = REPO_ROOT / "src" / "phase3_classifier" / "main.py"


def _copy_asset_folder(asset_name: str) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix="phase3_classifier_"))
    destination = temp_root / asset_name
    shutil.copytree(FIXTURE_ROOT / asset_name, destination)
    return destination


class Phase3ClassifierTests(unittest.TestCase):
    def test_openmaterial_sample_maps_full_core(self):
        asset_dir = _copy_asset_folder("openmatexamplehuman")
        report, report_path = write_asset_report(asset_dir)

        self.assertTrue(report_path.exists())
        self.assertEqual(report["recommended_primary_armature"], "Armature")
        self.assertEqual(len(report["missing_core_targets"]), 0)
        self.assertGreaterEqual(
            sum(1 for payload in report["asam_targets"].values() if payload["action"] == "direct_map"),
            18,
        )
        self.assertEqual(report["asam_targets"]["Root"]["source_bone"], "Root")
        self.assertEqual(report["asam_targets"]["Head"]["source_bone"], "Head")

    def test_lowpoly_classifies_both_armatures_and_prefers_def_rig(self):
        asset_dir = _copy_asset_folder("LowPolyCharacter4")
        report, _ = write_asset_report(asset_dir)

        self.assertEqual(report["asset_summary"]["armature_count"], 2)
        self.assertEqual(sorted(report["asset_summary"]["discovered_armatures"]), ["metarig", "rig"])
        self.assertEqual(report["recommended_primary_armature"], "rig")

        armatures = {item["armature_name"]: item for item in report["armatures"]}
        self.assertEqual(armatures["rig"]["selected_inputs"]["primary"], "rig_filtered.json")
        self.assertEqual(armatures["rig"]["selected_inputs"]["support"], "rig_all.json")
        self.assertIn(report["asam_targets"]["Root"]["action"], {"direct_map", "alias_map"})
        self.assertIsNotNone(report["asam_targets"]["Root"]["source_bone"])

    def test_partial_rig_yields_review_and_create_actions(self):
        temp_root = Path(tempfile.mkdtemp(prefix="phase3_partial_"))
        asset_dir = temp_root / "partial_asset"
        asset_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "armature_name": "Simple",
            "source_file": "partial.blend",
            "filter": None,
            "bone_count": 3,
            "hierarchy": {
                "root": ["thigh.L"],
                "thigh.L": ["shin.L"],
                "shin.L": [],
            },
            "bones": [
                {"name": "root", "parent": None, "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 0.5], "length": 0.5},
                {"name": "thigh.L", "parent": "root", "head": [0.1, 0.0, 0.8], "tail": [0.1, 0.0, 0.4], "length": 0.4},
                {"name": "shin.L", "parent": "thigh.L", "head": [0.1, 0.0, 0.4], "tail": [0.1, 0.0, 0.1], "length": 0.3},
            ],
            "chains": {
                "spine": [],
                "leg": {"left": [["thigh.L", "shin.L"]], "right": [], "unsided": []},
                "arm": {"left": [], "right": [], "unsided": []},
            },
        }
        with (asset_dir / "Simple_all.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)

        report, _ = write_asset_report(asset_dir)

        self.assertIn("missing_spine_chain", report["review_flags"])
        self.assertGreater(len(report["missing_core_targets"]), 0)
        self.assertIn(report["asam_targets"]["Hip"]["action"], {"review", "create_in_builder"})
        self.assertEqual(report["asam_targets"]["Upper_Leg_Left"]["source_bone"], "thigh.L")

    def test_cli_entrypoint_writes_report(self):
        asset_dir = _copy_asset_folder("openmatexamplehuman")
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), "--asset-dir", str(asset_dir)],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        self.assertTrue((asset_dir / "phase3_classification.json").exists())
        self.assertIn("Recommended primary armature: Armature", result.stdout)


if __name__ == "__main__":
    unittest.main()

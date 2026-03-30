import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional


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


def _bone(name: str, parent: Optional[str], head, tail):
    length = math.sqrt(sum((tail[index] - head[index]) ** 2 for index in range(3)))
    return {
        "name": name,
        "parent": parent,
        "head": list(head),
        "tail": list(tail),
        "length": length,
    }


def _placement_metadata(bbox_min, bbox_max):
    return {
        "bounds_source": "meshes",
        "driven_meshes": ["BodyMesh"],
        "bbox_min": list(bbox_min),
        "bbox_max": list(bbox_max),
        "bbox_height": bbox_max[2] - bbox_min[2],
        "bbox_ground_center": [
            (bbox_min[0] + bbox_max[0]) / 2.0,
            (bbox_min[1] + bbox_max[1]) / 2.0,
            bbox_min[2],
        ],
        "forward_axis": {"index": 0, "name": "x", "sign": 1},
        "side_axis": {"index": 1, "name": "y", "sign": 1},
        "up_axis": {"index": 2, "name": "z", "sign": 1},
    }


def _write_single_armature_asset(asset_name: str, armature_name: str, bones, chains, placement_metadata=None) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix="phase3_synth_"))
    asset_dir = temp_root / asset_name
    asset_dir.mkdir(parents=True, exist_ok=True)

    hierarchy = {bone["name"]: [] for bone in bones}
    for bone in bones:
        parent = bone["parent"]
        if parent in hierarchy:
            hierarchy[parent].append(bone["name"])

    payload = {
        "armature_name": armature_name,
        "source_file": f"{asset_name}.blend",
        "filter": None,
        "bone_count": len(bones),
        "hierarchy": hierarchy,
        "bones": bones,
        "chains": chains,
    }
    if placement_metadata is not None:
        payload["placement_metadata"] = placement_metadata

    with (asset_dir / f"{armature_name}_all.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    return asset_dir


class Phase3ClassifierTests(unittest.TestCase):
    def test_openmaterial_sample_maps_full_core_and_reuses_root(self):
        asset_dir = _copy_asset_folder("openmatexamplehuman")
        report, build_plan, report_path, plan_path = write_asset_report(asset_dir)

        self.assertTrue(report_path.exists())
        self.assertTrue(plan_path.exists())
        self.assertEqual(report["recommended_primary_armature"], "Armature")
        self.assertEqual(len(report["missing_targets"]), 0)
        self.assertGreaterEqual(
            sum(1 for payload in report["semantic_mapping"].values() if payload["action"] == "direct_map"),
            18,
        )
        self.assertEqual(report["semantic_mapping"]["Root"]["source_bone"], "Root")
        self.assertEqual(report["semantic_mapping"]["Head"]["source_bone"], "Head")
        self.assertEqual(build_plan["root_resolution"]["mode"], "reuse_existing_root")
        self.assertNotIn("root_noncompliant", report["review_flags"])
        self.assertFalse(any(action["target"] == "Root" for action in build_plan["actions"]["rename"]))

    def test_lowpoly_classifies_both_armatures_and_requires_new_root(self):
        asset_dir = _copy_asset_folder("LowPolyCharacter4")
        report, build_plan, _, _ = write_asset_report(asset_dir)

        self.assertEqual(report["asset_summary"]["armature_count"], 2)
        self.assertEqual(sorted(report["asset_summary"]["discovered_armatures"]), ["metarig", "rig"])
        self.assertEqual(report["recommended_primary_armature"], "rig")

        armatures = {item["armature_name"]: item for item in report["armatures"]}
        self.assertEqual(armatures["rig"]["selected_inputs"]["primary"], "rig_filtered.json")
        self.assertEqual(armatures["rig"]["selected_inputs"]["support"], "rig_all.json")
        self.assertEqual(report["semantic_mapping"]["Root"]["action"], "repair_in_builder")
        self.assertEqual(build_plan["root_resolution"]["mode"], "create_new_root")
        self.assertEqual(build_plan["root_resolution"]["source_bone"], "root")
        self.assertIn("root_noncompliant", report["review_flags"])
        self.assertIn("multiple_source_roots", build_plan["root_resolution"]["failure_codes"])
        self.assertIn("root_not_vertical", build_plan["root_resolution"]["failure_codes"])
        self.assertFalse(any(action["target"] == "Root" for action in build_plan["actions"]["rename"]))

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

        report, build_plan, _, _ = write_asset_report(asset_dir)

        self.assertIn("missing_spine_chain", report["review_flags"])
        self.assertIn(report["semantic_mapping"]["Hip"]["action"], {"review", "create_in_builder"})
        self.assertEqual(report["semantic_mapping"]["Upper_Leg_Left"]["source_bone"], "thigh.L")
        self.assertEqual(build_plan["root_resolution"]["mode"], "review")

    def test_compliant_wrong_named_root_reuses_existing_root_with_alias(self):
        bones = [
            _bone("master", None, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            _bone("Hip", "master", (0.0, 0.0, 1.0), (0.0, 0.0, 1.15)),
            _bone("Lower_Spine", "Hip", (0.0, 0.0, 1.15), (0.0, 0.0, 1.35)),
            _bone("Upper_Spine", "Lower_Spine", (0.0, 0.0, 1.35), (0.0, 0.0, 1.55)),
            _bone("Upper_Leg_Left", "Hip", (0.0, 0.15, 1.0), (0.0, 0.15, 0.45)),
            _bone("Lower_Leg_Left", "Upper_Leg_Left", (0.0, 0.15, 0.45), (0.0, 0.15, 0.1)),
            _bone("Foot_Left", "Lower_Leg_Left", (0.0, 0.15, 0.1), (0.18, 0.15, 0.0)),
            _bone("Upper_Leg_Right", "Hip", (0.0, -0.15, 1.0), (0.0, -0.15, 0.45)),
            _bone("Lower_Leg_Right", "Upper_Leg_Right", (0.0, -0.15, 0.45), (0.0, -0.15, 0.1)),
            _bone("Foot_Right", "Lower_Leg_Right", (0.0, -0.15, 0.1), (0.18, -0.15, 0.0)),
        ]
        chains = {
            "spine": [["Lower_Spine", "Upper_Spine"]],
            "leg": {
                "left": [["Upper_Leg_Left", "Lower_Leg_Left", "Foot_Left"]],
                "right": [["Upper_Leg_Right", "Lower_Leg_Right", "Foot_Right"]],
                "unsided": [],
            },
            "arm": {"left": [], "right": [], "unsided": []},
        }
        asset_dir = _write_single_armature_asset(
            "compliant_alias_root",
            "Rig",
            bones,
            chains,
            _placement_metadata((-0.2, -0.25, 0.0), (0.2, 0.25, 1.8)),
        )

        report, build_plan, _, _ = write_asset_report(asset_dir)

        self.assertEqual(report["semantic_mapping"]["Root"]["action"], "alias_map")
        self.assertEqual(build_plan["root_resolution"]["mode"], "reuse_existing_root")
        self.assertTrue(build_plan["root_resolution"]["rename_source_to_target"])
        self.assertEqual(build_plan["root_resolution"]["source_bone"], "master")

    def test_horizontal_root_requires_new_root(self):
        bones = [
            _bone("Root", None, (0.0, 0.0, 0.0), (0.9, 0.0, 0.0)),
            _bone("Hip", "Root", (0.0, 0.0, 1.0), (0.0, 0.0, 1.15)),
            _bone("Lower_Spine", "Hip", (0.0, 0.0, 1.15), (0.0, 0.0, 1.35)),
            _bone("Upper_Leg_Left", "Hip", (0.0, 0.15, 1.0), (0.0, 0.15, 0.45)),
            _bone("Upper_Leg_Right", "Hip", (0.0, -0.15, 1.0), (0.0, -0.15, 0.45)),
        ]
        chains = {
            "spine": [["Lower_Spine"]],
            "leg": {
                "left": [["Upper_Leg_Left"]],
                "right": [["Upper_Leg_Right"]],
                "unsided": [],
            },
            "arm": {"left": [], "right": [], "unsided": []},
        }
        asset_dir = _write_single_armature_asset(
            "horizontal_root",
            "Rig",
            bones,
            chains,
            _placement_metadata((-0.2, -0.2, 0.0), (0.2, 0.2, 1.6)),
        )

        report, build_plan, _, _ = write_asset_report(asset_dir)

        self.assertEqual(report["semantic_mapping"]["Root"]["action"], "repair_in_builder")
        self.assertEqual(build_plan["root_resolution"]["mode"], "create_new_root")
        self.assertIn("root_not_vertical", build_plan["root_resolution"]["failure_codes"])

    def test_missing_root_with_valid_hip_repairs_in_builder(self):
        bones = [
            _bone("Hip", None, (0.0, 0.0, 1.0), (0.0, 0.0, 1.15)),
            _bone("Lower_Spine", "Hip", (0.0, 0.0, 1.15), (0.0, 0.0, 1.35)),
            _bone("Upper_Leg_Left", "Hip", (0.0, 0.15, 1.0), (0.0, 0.15, 0.45)),
            _bone("Upper_Leg_Right", "Hip", (0.0, -0.15, 1.0), (0.0, -0.15, 0.45)),
        ]
        chains = {
            "spine": [["Lower_Spine"]],
            "leg": {
                "left": [["Upper_Leg_Left"]],
                "right": [["Upper_Leg_Right"]],
                "unsided": [],
            },
            "arm": {"left": [], "right": [], "unsided": []},
        }
        asset_dir = _write_single_armature_asset(
            "missing_root",
            "Rig",
            bones,
            chains,
            _placement_metadata((-0.2, -0.2, 0.0), (0.2, 0.2, 1.6)),
        )

        report, build_plan, _, _ = write_asset_report(asset_dir)

        self.assertEqual(report["semantic_mapping"]["Root"]["action"], "repair_in_builder")
        self.assertEqual(build_plan["root_resolution"]["mode"], "create_new_root")
        self.assertIsNone(build_plan["root_resolution"]["source_bone"])
        self.assertIn("no_root_candidate", build_plan["root_resolution"]["failure_codes"])

    def test_multi_root_control_rig_repairs_root(self):
        bones = [
            _bone("root_ctrl", None, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            _bone("Hip", "root_ctrl", (0.0, 0.0, 1.0), (0.0, 0.0, 1.15)),
            _bone("Lower_Spine", "Hip", (0.0, 0.0, 1.15), (0.0, 0.0, 1.35)),
            _bone("Locator", None, (0.4, 0.0, 0.0), (0.4, 0.0, 0.4)),
        ]
        chains = {
            "spine": [["Lower_Spine"]],
            "leg": {"left": [], "right": [], "unsided": []},
            "arm": {"left": [], "right": [], "unsided": []},
        }
        asset_dir = _write_single_armature_asset(
            "multi_root_control",
            "Rig",
            bones,
            chains,
            _placement_metadata((-0.2, -0.2, 0.0), (0.2, 0.2, 1.6)),
        )

        report, build_plan, _, _ = write_asset_report(asset_dir)

        self.assertEqual(report["semantic_mapping"]["Root"]["action"], "repair_in_builder")
        self.assertEqual(build_plan["root_resolution"]["mode"], "create_new_root")
        self.assertIn("multiple_source_roots", build_plan["root_resolution"]["failure_codes"])
        self.assertIn("root_candidate_disallowed_role", build_plan["root_resolution"]["failure_codes"])

    def test_cli_entrypoint_writes_report(self):
        asset_dir = _copy_asset_folder("openmatexamplehuman")
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), "--asset-dir", str(asset_dir)],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        self.assertTrue((asset_dir / "classifier_report.json").exists())
        self.assertTrue((asset_dir / "build_plan.json").exists())
        self.assertIn("Recommended primary armature: Armature", result.stdout)
        self.assertIn("Root resolution: reuse_existing_root", result.stdout)


if __name__ == "__main__":
    unittest.main()

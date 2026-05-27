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

from phase3_classifier.classifier import (  # noqa: E402
    write_asset_report,
    _compute_mesh_bound_term,
    _compute_deform_evidence_term,
    _compute_extras_term,
)


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures"
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


def _mesh_binding(armature_name: str = "Rig"):
    return {
        "armature_object_name": armature_name,
        "meshes": [
            {
                "mesh_name": "BodyMesh",
                "armature_link": "modifier",
                "modifiers": [
                    {
                        "stack_index": 0,
                        "type": "ARMATURE",
                        "name": "Armature",
                        "object": armature_name,
                    }
                ],
                "vertex_groups": ["Hip"],
                "vertex_group_stats": {
                    "non_empty_group_count": 1,
                    "per_group": [{"name": "Hip", "weighted_vertex_count": 8}],
                },
                "material_slots": [{"slot_index": 0, "material_name": "BodyMat"}],
                "warnings": [],
            }
        ],
    }


def _write_single_armature_asset(
    asset_name: str,
    armature_name: str,
    bones,
    chains,
    placement_metadata=None,
    mesh_binding=None,
) -> Path:
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
    if mesh_binding is not None:
        payload["mesh_binding"] = mesh_binding

    with (asset_dir / f"{armature_name}_all.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    return asset_dir


class Phase3ClassifierTests(unittest.TestCase):
    def test_build_plan_propagates_recommended_mesh_binding(self):
        bones = [
            _bone("Root", None, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            _bone("Hip", "Root", (0.0, 0.0, 1.0), (0.0, 0.0, 1.1)),
            _bone("Lower_Spine", "Hip", (0.0, 0.0, 1.1), (0.0, 0.0, 1.25)),
            _bone("Upper_Spine", "Lower_Spine", (0.0, 0.0, 1.25), (0.0, 0.0, 1.45)),
            _bone("Upper_Leg_Left", "Hip", (0.0, 0.15, 1.0), (0.0, 0.15, 0.45)),
            _bone("Upper_Leg_Right", "Hip", (0.0, -0.15, 1.0), (0.0, -0.15, 0.45)),
        ]
        chains = {
            "spine": [["Lower_Spine", "Upper_Spine"]],
            "leg": {
                "left": [["Upper_Leg_Left"]],
                "right": [["Upper_Leg_Right"]],
                "unsided": [],
            },
            "arm": {"left": [], "right": [], "unsided": []},
        }
        binding = _mesh_binding("Rig")
        asset_dir = _write_single_armature_asset(
            "mesh_binding_propagation",
            "Rig",
            bones,
            chains,
            _placement_metadata((-0.2, -0.25, 0.0), (0.2, 0.25, 1.8)),
            binding,
        )

        report, build_plan, _, _ = write_asset_report(asset_dir)

        self.assertEqual(report["mesh_binding"], binding)
        self.assertEqual(build_plan["mesh_binding"], binding)
        self.assertEqual(build_plan["mesh_binding"]["armature_object_name"], build_plan["recommended_primary_armature"])

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
        self.assertEqual(report["semantic_mapping"]["Root"]["action"], "direct_map")
        self.assertEqual(build_plan["root_resolutions"][0]["mode"], "reuse_existing_root")
        self.assertNotIn("root_noncompliant", report["review_flags"])
        self.assertIn(
            "ASAM OpenMATERIAL 3D 7.3.3.1 General",
            build_plan["root_resolutions"][0]["spec_references"],
        )
        # Planar offset is now an advisory, not a violation note.
        self.assertNotIn(
            "root_origin_violation_against_asam_7_3_3_1",
            build_plan["root_resolutions"][0]["diagnostic_notes"],
        )
        self.assertIn(
            "mesh_bounds_offset_detected_root_at_local_origin",
            build_plan["root_resolutions"][0]["diagnostic_notes"],
        )
        planar_advisories = [
            a for a in build_plan["root_resolutions"][0].get("advisories", [])
            if a.startswith("root_head_off_ground_center_advisory:")
        ]
        self.assertEqual(len(planar_advisories), 1)
        self.assertFalse(any(action["target"] == "Root" for action in build_plan["actions"]["rename"]))
        # grp_root_local_origin equals bbox_ground_center (Grp_Root's world anchor).
        self.assertEqual(
            build_plan["root_resolutions"][0]["grp_root_local_origin"],
            build_plan["placement_metadata"]["bbox_ground_center"],
        )

    def test_compliant_root_has_near_zero_translation_offset(self):
        bones = [
            _bone("Root", None, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            _bone("Hip", "Root", (0.0, 0.0, 1.0), (0.0, 0.0, 1.15)),
            _bone("Lower_Spine", "Hip", (0.0, 0.0, 1.15), (0.0, 0.0, 1.35)),
            _bone("Upper_Spine", "Lower_Spine", (0.0, 0.0, 1.35), (0.0, 0.0, 1.55)),
            _bone("Upper_Leg_Left", "Hip", (0.0, 0.15, 1.0), (0.0, 0.15, 0.45)),
            _bone("Upper_Leg_Right", "Hip", (0.0, -0.15, 1.0), (0.0, -0.15, 0.45)),
        ]
        chains = {
            "spine": [["Lower_Spine", "Upper_Spine"]],
            "leg": {
                "left": [["Upper_Leg_Left"]],
                "right": [["Upper_Leg_Right"]],
                "unsided": [],
            },
            "arm": {"left": [], "right": [], "unsided": []},
        }
        asset_dir = _write_single_armature_asset(
            "compliant_root_offset",
            "Rig",
            bones,
            chains,
            _placement_metadata((-0.2, -0.25, 0.0), (0.2, 0.25, 1.8)),
        )

        _, build_plan, _, _ = write_asset_report(asset_dir)

        # For a compliant rig the bbox-ground-center coincides with the source root,
        # which sits at the source origin, so grp_root_local_origin is also near zero.
        origin = build_plan["root_resolutions"][0]["grp_root_local_origin"]
        for component in origin:
            self.assertLess(abs(component), 1e-6)

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
        self.assertEqual(report["semantic_mapping"]["Hip"]["action"], "repair_in_builder")
        self.assertIn("paired_sided_pelvis_requires_centering", report["semantic_mapping"]["Hip"]["notes"])
        self.assertEqual(build_plan["root_resolutions"][0]["mode"], "create_new_root")
        self.assertEqual(build_plan["root_resolutions"][0]["source_bone"], "root")
        self.assertIn("root_noncompliant", report["review_flags"])
        self.assertIn("multiple_source_roots", build_plan["root_resolutions"][0]["blocker_codes"])
        self.assertIn("root_not_vertical", build_plan["root_resolutions"][0]["blocker_codes"])
        self.assertFalse(any(action["target"] == "Root" for action in build_plan["actions"]["rename"]))
        self.assertFalse(any(action["target"] == "Hip" for action in build_plan["actions"]["rename"]))

    def test_paired_sided_pelvis_marks_hip_for_builder_repair(self):
        bones = [
            _bone("root", None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.9)),
            _bone("pelvis.L", "root", (0.0, 0.0, 0.9), (0.12, 0.0, 1.02)),
            _bone("pelvis.R", "root", (0.0, 0.0, 0.9), (-0.12, 0.0, 1.02)),
            _bone("spine", "root", (0.0, 0.0, 1.02), (0.0, 0.0, 1.25)),
            _bone("thigh.L", "pelvis.L", (0.12, 0.0, 0.9), (0.12, 0.0, 0.45)),
            _bone("thigh.R", "pelvis.R", (-0.12, 0.0, 0.9), (-0.12, 0.0, 0.45)),
        ]
        chains = {
            "spine": [["spine"]],
            "leg": {
                "left": [["thigh.L"]],
                "right": [["thigh.R"]],
                "unsided": [],
            },
            "arm": {"left": [], "right": [], "unsided": []},
        }
        asset_dir = _write_single_armature_asset(
            "paired_pelvis",
            "Rig",
            bones,
            chains,
            _placement_metadata((-0.3, -0.2, 0.0), (0.3, 0.2, 1.5)),
        )

        report, build_plan, _, _ = write_asset_report(asset_dir)

        self.assertEqual(report["semantic_mapping"]["Hip"]["action"], "repair_in_builder")
        self.assertIn("paired_sided_pelvis_requires_centering", report["semantic_mapping"]["Hip"]["notes"])
        self.assertIn(report["semantic_mapping"]["Hip"]["source_bone"], {"pelvis.L", "pelvis.R"})
        self.assertFalse(any(action["target"] == "Hip" for action in build_plan["actions"]["rename"]))

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
        self.assertEqual(build_plan["root_resolutions"][0]["mode"], "review")

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
        self.assertEqual(build_plan["root_resolutions"][0]["mode"], "reuse_existing_root")
        self.assertTrue(build_plan["root_resolutions"][0]["rename_source_to_target"])
        self.assertEqual(build_plan["root_resolutions"][0]["source_bone"], "master")
        self.assertIn(
            "ASAM OpenMATERIAL 3D 7.3.3.3.4 Root",
            build_plan["root_resolutions"][0]["spec_references"],
        )
        self.assertNotIn(
            "root_origin_violation_against_asam_7_3_3_1",
            build_plan["root_resolutions"][0]["diagnostic_notes"],
        )
        self.assertFalse(
            any(
                note.startswith("mesh_bounds_offset_detected")
                for note in build_plan["root_resolutions"][0]["diagnostic_notes"]
            )
        )

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
        self.assertEqual(build_plan["root_resolutions"][0]["mode"], "create_new_root")
        self.assertIn("root_not_vertical", build_plan["root_resolutions"][0]["blocker_codes"])

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
        self.assertEqual(build_plan["root_resolutions"][0]["mode"], "create_new_root")
        self.assertIsNone(build_plan["root_resolutions"][0]["source_bone"])
        self.assertIn("no_root_candidate", build_plan["root_resolutions"][0]["blocker_codes"])

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
        self.assertEqual(build_plan["root_resolutions"][0]["mode"], "create_new_root")
        self.assertIn("multiple_source_roots", build_plan["root_resolutions"][0]["blocker_codes"])
        self.assertIn("root_candidate_disallowed_role", build_plan["root_resolutions"][0]["blocker_codes"])

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

    def test_root_blocker_codes_helper_exists(self):
        from phase3_classifier.classifier import _root_blocker_codes, _root_advisory_codes  # noqa: F401
        self.assertTrue(callable(_root_blocker_codes))
        self.assertTrue(callable(_root_advisory_codes))

    def test_openmatexample_reuses_source_root_under_new_compliance_model(self):
        """openmatexamplehuman's source root has an ~8.6 cm planar offset from bbox center.
        Under the new model this is advisory, not a blocker; mode should be reuse."""
        asset_dir = _copy_asset_folder("openmatexamplehuman")
        report, build_plan, _, _ = write_asset_report(asset_dir)
        self.assertEqual(report["root_resolutions"][0]["mode"], "reuse_existing_root")
        self.assertEqual(report["root_resolutions"][0]["blocker_codes"], [])

    def test_classifier_emits_root_resolutions_list(self):
        asset_dir = _copy_asset_folder("openmatexamplehuman")
        report, build_plan, _, _ = write_asset_report(asset_dir)
        self.assertIn("root_resolutions", report)
        self.assertIsInstance(report["root_resolutions"], list)
        self.assertEqual(len(report["root_resolutions"]), 1)
        entry = report["root_resolutions"][0]
        self.assertEqual(entry["subtree_name"], "Grp_Root")
        self.assertEqual(entry["blocker_codes"], [])
        self.assertIn("advisories", entry)
        self.assertEqual(report["root_resolutions"], build_plan["root_resolutions"])
        self.assertNotIn("root_resolution", report)
        self.assertNotIn("root_resolution", build_plan)

    def test_grp_root_local_origin_equals_bbox_ground_center(self):
        asset_dir = _copy_asset_folder("openmatexamplehuman")
        _, build_plan, _, _ = write_asset_report(asset_dir)
        entry = build_plan["root_resolutions"][0]
        self.assertIn("grp_root_local_origin", entry)
        expected = build_plan["placement_metadata"]["bbox_ground_center"]
        self.assertEqual(entry["grp_root_local_origin"], expected)
        for obsolete in ("source_translation_offset", "target_head", "target_tail", "use_connect"):
            self.assertNotIn(obsolete, entry, "expected {0} to be removed".format(obsolete))

    def test_planar_offset_emitted_as_advisory_with_magnitude(self):
        asset_dir = _copy_asset_folder("openmatexamplehuman")
        report, _, _, _ = write_asset_report(asset_dir)
        advisories = report["root_resolutions"][0].get("advisories", [])
        planar = [a for a in advisories if a.startswith("root_head_off_ground_center_advisory:")]
        self.assertEqual(len(planar), 1, "expected one planar advisory, got {0}".format(advisories))
        magnitude = float(planar[0].split(":", 1)[1])
        # bbox_ground_center = (0.086318, 0.009727, ...); source root at (0, 0, 0).
        # planar = sqrt(0.086318**2 + 0.009727**2) ~= 0.086865
        self.assertAlmostEqual(magnitude, 0.086865, places=4)


class ArmatureScoringHelperTests(unittest.TestCase):
    def test_mesh_bound_term_zero_when_meshes_empty(self):
        binding = {"armature_object_name": "rig", "meshes": []}
        self.assertEqual(_compute_mesh_bound_term(binding, "rig"), 0.0)

    def test_mesh_bound_term_zero_when_binding_is_none(self):
        self.assertEqual(_compute_mesh_bound_term(None, "rig"), 0.0)

    def test_mesh_bound_term_partial_when_meshes_present_but_no_modifier_link(self):
        binding = {
            "armature_object_name": "rig",
            "meshes": [
                {
                    "mesh_name": "Cube",
                    "armature_link": "parent",
                    "modifiers": [],
                    "vertex_groups": [],
                }
            ],
        }
        self.assertEqual(_compute_mesh_bound_term(binding, "rig"), 6.0)

    def test_mesh_bound_term_full_credit_when_modifier_links_to_armature(self):
        binding = {
            "armature_object_name": "rig",
            "meshes": [
                {
                    "mesh_name": "Cube",
                    "armature_link": "parent_and_modifier",
                    "modifiers": [
                        {"stack_index": 0, "type": "ARMATURE", "name": "Armature", "object": "rig"}
                    ],
                    "vertex_groups": [],
                }
            ],
        }
        self.assertEqual(_compute_mesh_bound_term(binding, "rig"), 10.0)

    def test_mesh_bound_term_partial_when_modifier_links_to_other_armature(self):
        binding = {
            "armature_object_name": "rig",
            "meshes": [
                {
                    "mesh_name": "Cube",
                    "armature_link": "parent_and_modifier",
                    "modifiers": [
                        {"stack_index": 0, "type": "ARMATURE", "name": "Armature", "object": "other_rig"}
                    ],
                    "vertex_groups": [],
                }
            ],
        }
        self.assertEqual(_compute_mesh_bound_term(binding, "rig"), 6.0)

    def test_deform_evidence_term_zero_when_no_meshes(self):
        binding = {"armature_object_name": "rig", "meshes": []}
        bone_names = {"spine", "head", "hand.l"}
        self.assertEqual(_compute_deform_evidence_term(binding, bone_names), 0.0)

    def test_deform_evidence_term_zero_when_binding_is_none(self):
        bone_names = {"spine", "head"}
        self.assertEqual(_compute_deform_evidence_term(None, bone_names), 0.0)

    def test_deform_evidence_term_counts_direct_vertex_group_matches(self):
        binding = {
            "armature_object_name": "rig",
            "meshes": [
                {
                    "mesh_name": "Body",
                    "vertex_groups": ["spine", "head", "Hand.L", "unrelated_group"],
                    "modifiers": [],
                }
            ],
        }
        bone_names = {"spine", "head", "hand.l", "ignored_bone"}
        expected = round(min(math.log1p(3) * 2.0, 6.0), 3)
        self.assertAlmostEqual(_compute_deform_evidence_term(binding, bone_names), expected, places=3)

    def test_deform_evidence_term_strips_def_prefix_on_vertex_group_side(self):
        binding = {
            "armature_object_name": "rig",
            "meshes": [
                {
                    "mesh_name": "Body",
                    "vertex_groups": ["DEF-spine", "DEF-hand.L", "Hair_Front"],
                    "modifiers": [],
                }
            ],
        }
        bone_names = {"spine", "hand.l"}
        expected = round(min(math.log1p(2) * 2.0, 6.0), 3)
        self.assertAlmostEqual(_compute_deform_evidence_term(binding, bone_names), expected, places=3)

    def test_deform_evidence_term_counts_unique_bones_across_multiple_meshes(self):
        binding = {
            "armature_object_name": "rig",
            "meshes": [
                {"mesh_name": "A", "vertex_groups": ["spine"], "modifiers": []},
                {"mesh_name": "B", "vertex_groups": ["spine", "head"], "modifiers": []},
            ],
        }
        bone_names = {"spine", "head", "extra"}
        expected = round(min(math.log1p(2) * 2.0, 6.0), 3)
        self.assertAlmostEqual(_compute_deform_evidence_term(binding, bone_names), expected, places=3)

    def test_deform_evidence_term_caps_at_six(self):
        bone_names = {"bone_{}".format(i) for i in range(1000)}
        binding = {
            "armature_object_name": "rig",
            "meshes": [
                {
                    "mesh_name": "Body",
                    "vertex_groups": ["bone_{}".format(i) for i in range(1000)],
                    "modifiers": [],
                }
            ],
        }
        self.assertEqual(_compute_deform_evidence_term(binding, bone_names), 6.0)

    def test_extras_term_zero_when_empty(self):
        self.assertEqual(_compute_extras_term([]), 0.0)

    def test_extras_term_log_dampened_for_small_counts(self):
        extras = [{"bone_name": "spine.001"}, {"bone_name": "spine.002"}, {"bone_name": "spine.003"}]
        expected = round(min(math.log1p(3) * 0.5, 2.0), 3)
        self.assertEqual(_compute_extras_term(extras), expected)

    def test_extras_term_caps_at_two(self):
        extras = [{"bone_name": "bone_{}".format(i)} for i in range(200)]
        self.assertEqual(_compute_extras_term(extras), 2.0)


if __name__ == "__main__":
    unittest.main()

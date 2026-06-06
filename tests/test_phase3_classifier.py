import json
import math
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase3_classifier.classifier import (  # noqa: E402
    write_asset_report,
    _collect_mesh_binding_stats,
    _compute_mesh_bound_term,
    _compute_deform_evidence_term,
    _compute_extras_term,
    _compute_vertex_group_coverage,
    _compute_deform_purity,
    _compute_selection_tiebreaker,
    _demote_neck_coincident_upper_spine,
    _detect_convention,
    _normalize_bone_name,
    _detect_family_tags,
    _family_name_score,
    compute_name_quality,
    _reconciliation_tag,
    BoneInfo,
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


def _mesh_binding_for(armature_name, vertex_groups):
    """A mesh-binding where `armature_name` modifier-drives BodyMesh with the
    given vertex-group names (so meshes_count=1 and has_armature_modifier_link)."""
    return {
        "armature_object_name": armature_name,
        "meshes": [
            {
                "mesh_name": "BodyMesh",
                "armature_link": "modifier",
                "modifiers": [{"stack_index": 0, "type": "ARMATURE", "name": "Armature", "object": armature_name}],
                "vertex_groups": list(vertex_groups),
                "vertex_group_stats": {"non_empty_group_count": len(vertex_groups), "per_group": []},
                "material_slots": [],
                "warnings": [],
            }
        ],
    }


def _write_multi_armature_asset(asset_name, armatures):
    """Write one asset dir containing several `<armature>_all.json` files.

    `armatures` is a list of (armature_name, bones, chains, mesh_binding|None).
    """
    temp_root = Path(tempfile.mkdtemp(prefix="phase3_multi_"))
    asset_dir = temp_root / asset_name
    asset_dir.mkdir(parents=True, exist_ok=True)
    for armature_name, bones, chains, mesh_binding in armatures:
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
        if mesh_binding is not None:
            payload["mesh_binding"] = mesh_binding
        with (asset_dir / f"{armature_name}_all.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    return asset_dir


def _mixamo_character_without_toes():
    """Full mixamo skeleton minus the two ToeBase bones (so Full_Toes go unmapped)."""
    bones, chains = _mixamo_character()
    drop = {"mixamorig:LeftToeBase", "mixamorig:RightToeBase"}
    bones = [b for b in bones if b["name"] not in drop]
    chains = {
        "spine": chains["spine"],
        "leg": {side: [[n for n in chain if n not in drop] for chain in chains["leg"][side]] for side in chains["leg"]},
        "arm": chains["arm"],
    }
    return bones, chains


def _mixamo_with_extra_control_bones(extra_count=60):
    """Mixamo deform skeleton PLUS scaffolding bones absent from any vertex group.

    Covers the same vertex groups as the deform rig (so coverage ties) but has far
    lower deform purity (most bones don't drive skin)."""
    bones, chains = _mixamo_character()
    root = bones[0]["name"]
    for index in range(extra_count):
        bones.append(_bone(f"mixamorig:CTRL_Twist_{index}", root, (0.5, 0.0, 1.0), (0.5, 0.0, 1.05)))
    return bones, chains


def _biped_character():
    bones = [
        _bone("Bip01 COM", None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.95)),
        _bone("Bip01 Pelvis", "Bip01 COM", (0.0, 0.0, 0.95), (0.0, 0.0, 1.05)),
        _bone("Bip01 Spine0", "Bip01 Pelvis", (0.0, 0.0, 1.05), (0.0, 0.0, 1.25)),
        _bone("Bip01 Spine1", "Bip01 Spine0", (0.0, 0.0, 1.25), (0.0, 0.0, 1.45)),
        _bone("Bip01 Neck", "Bip01 Spine1", (0.0, 0.0, 1.45), (0.0, 0.0, 1.6)),
        _bone("Bip01 Head", "Bip01 Neck", (0.0, 0.0, 1.6), (0.0, 0.0, 1.75)),
        _bone("Bip01 LClavicle", "Bip01 Spine1", (0.0, 0.05, 1.45), (0.0, 0.18, 1.45)),
        _bone("Bip01 LUpperArm", "Bip01 LClavicle", (0.0, 0.18, 1.45), (0.0, 0.45, 1.45)),
        _bone("Bip01 LForeArm", "Bip01 LUpperArm", (0.0, 0.45, 1.45), (0.0, 0.7, 1.45)),
        _bone("Bip01 LHand", "Bip01 LForeArm", (0.0, 0.7, 1.45), (0.0, 0.8, 1.45)),
        _bone("Bip01 RClavicle", "Bip01 Spine1", (0.0, -0.05, 1.45), (0.0, -0.18, 1.45)),
        _bone("Bip01 RUpperArm", "Bip01 RClavicle", (0.0, -0.18, 1.45), (0.0, -0.45, 1.45)),
        _bone("Bip01 RForeArm", "Bip01 RUpperArm", (0.0, -0.45, 1.45), (0.0, -0.7, 1.45)),
        _bone("Bip01 RHand", "Bip01 RForeArm", (0.0, -0.7, 1.45), (0.0, -0.8, 1.45)),
        _bone("Bip01 LThigh", "Bip01 Pelvis", (0.0, 0.1, 0.95), (0.0, 0.1, 0.55)),
        _bone("Bip01 LCalf", "Bip01 LThigh", (0.0, 0.1, 0.55), (0.0, 0.1, 0.1)),
        _bone("Bip01 LFoot", "Bip01 LCalf", (0.0, 0.1, 0.1), (0.15, 0.1, 0.0)),
        _bone("Bip01 LToe0", "Bip01 LFoot", (0.15, 0.1, 0.0), (0.25, 0.1, 0.0)),
        _bone("Bip01 RThigh", "Bip01 Pelvis", (0.0, -0.1, 0.95), (0.0, -0.1, 0.55)),
        _bone("Bip01 RCalf", "Bip01 RThigh", (0.0, -0.1, 0.55), (0.0, -0.1, 0.1)),
        _bone("Bip01 RFoot", "Bip01 RCalf", (0.0, -0.1, 0.1), (0.15, -0.1, 0.0)),
        _bone("Bip01 RToe0", "Bip01 RFoot", (0.15, -0.1, 0.0), (0.25, -0.1, 0.0)),
    ]
    chains = {
        "spine": [["Bip01 Spine0", "Bip01 Spine1"]],
        "leg": {
            "left": [["Bip01 LThigh", "Bip01 LCalf", "Bip01 LFoot", "Bip01 LToe0"]],
            "right": [["Bip01 RThigh", "Bip01 RCalf", "Bip01 RFoot", "Bip01 RToe0"]],
            "unsided": [],
        },
        "arm": {
            "left": [["Bip01 LUpperArm", "Bip01 LForeArm", "Bip01 LHand"]],
            "right": [["Bip01 RUpperArm", "Bip01 RForeArm", "Bip01 RHand"]],
            "unsided": [],
        },
    }
    return bones, chains


def _mixamo_character():
    bones = [
        _bone("mixamorig:Hips", None, (0.0, 0.0, 0.95), (0.0, 0.0, 1.05)),
        _bone("mixamorig:Spine", "mixamorig:Hips", (0.0, 0.0, 1.05), (0.0, 0.0, 1.2)),
        _bone("mixamorig:Spine1", "mixamorig:Spine", (0.0, 0.0, 1.2), (0.0, 0.0, 1.35)),
        _bone("mixamorig:Spine2", "mixamorig:Spine1", (0.0, 0.0, 1.35), (0.0, 0.0, 1.45)),
        _bone("mixamorig:Neck", "mixamorig:Spine2", (0.0, 0.0, 1.45), (0.0, 0.0, 1.6)),
        _bone("mixamorig:Head", "mixamorig:Neck", (0.0, 0.0, 1.6), (0.0, 0.0, 1.75)),
        _bone("mixamorig:LeftShoulder", "mixamorig:Spine2", (0.0, 0.05, 1.45), (0.0, 0.18, 1.45)),
        _bone("mixamorig:LeftArm", "mixamorig:LeftShoulder", (0.0, 0.18, 1.45), (0.0, 0.45, 1.45)),
        _bone("mixamorig:LeftForeArm", "mixamorig:LeftArm", (0.0, 0.45, 1.45), (0.0, 0.7, 1.45)),
        _bone("mixamorig:LeftHand", "mixamorig:LeftForeArm", (0.0, 0.7, 1.45), (0.0, 0.8, 1.45)),
        _bone("mixamorig:RightShoulder", "mixamorig:Spine2", (0.0, -0.05, 1.45), (0.0, -0.18, 1.45)),
        _bone("mixamorig:RightArm", "mixamorig:RightShoulder", (0.0, -0.18, 1.45), (0.0, -0.45, 1.45)),
        _bone("mixamorig:RightForeArm", "mixamorig:RightArm", (0.0, -0.45, 1.45), (0.0, -0.7, 1.45)),
        _bone("mixamorig:RightHand", "mixamorig:RightForeArm", (0.0, -0.7, 1.45), (0.0, -0.8, 1.45)),
        _bone("mixamorig:LeftUpLeg", "mixamorig:Hips", (0.0, 0.1, 0.95), (0.0, 0.1, 0.55)),
        _bone("mixamorig:LeftLeg", "mixamorig:LeftUpLeg", (0.0, 0.1, 0.55), (0.0, 0.1, 0.1)),
        _bone("mixamorig:LeftFoot", "mixamorig:LeftLeg", (0.0, 0.1, 0.1), (0.15, 0.1, 0.0)),
        _bone("mixamorig:LeftToeBase", "mixamorig:LeftFoot", (0.15, 0.1, 0.0), (0.25, 0.1, 0.0)),
        _bone("mixamorig:RightUpLeg", "mixamorig:Hips", (0.0, -0.1, 0.95), (0.0, -0.1, 0.55)),
        _bone("mixamorig:RightLeg", "mixamorig:RightUpLeg", (0.0, -0.1, 0.55), (0.0, -0.1, 0.1)),
        _bone("mixamorig:RightFoot", "mixamorig:RightLeg", (0.0, -0.1, 0.1), (0.15, -0.1, 0.0)),
        _bone("mixamorig:RightToeBase", "mixamorig:RightFoot", (0.15, -0.1, 0.0), (0.25, -0.1, 0.0)),
    ]
    chains = {
        "spine": [["mixamorig:Spine", "mixamorig:Spine1", "mixamorig:Spine2"]],
        "leg": {
            "left": [["mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot", "mixamorig:LeftToeBase"]],
            "right": [["mixamorig:RightUpLeg", "mixamorig:RightLeg", "mixamorig:RightFoot", "mixamorig:RightToeBase"]],
            "unsided": [],
        },
        "arm": {
            "left": [["mixamorig:LeftArm", "mixamorig:LeftForeArm", "mixamorig:LeftHand"]],
            "right": [["mixamorig:RightArm", "mixamorig:RightForeArm", "mixamorig:RightHand"]],
            "unsided": [],
        },
    }
    return bones, chains


def _prefixed_biped(prefix: str, sub_root_parent: str):
    """A biped character renamed with a crowd prefix, parented under a shared root."""
    bones, chains = _biped_character()
    rename = {}
    out_bones = []
    for index, bone in enumerate(bones):
        old = bone["name"]
        new = prefix + old.replace("Bip01 ", "").replace(" ", "") + f"_{index}"
        rename[old] = new
    for bone in bones:
        new_bone = dict(bone)
        new_bone["name"] = rename[bone["name"]]
        parent = bone["parent"]
        new_bone["parent"] = rename[parent] if parent in rename else sub_root_parent
        out_bones.append(new_bone)

    def fix_chain_list(chain_list):
        return [[rename[name] for name in chain] for chain in chain_list]

    out_chains = {
        "spine": fix_chain_list(chains["spine"]),
        "leg": {side: fix_chain_list(chains["leg"][side]) for side in chains["leg"]},
        "arm": {side: fix_chain_list(chains["arm"][side]) for side in chains["arm"]},
    }
    return out_bones, out_chains


def _write_crowd_asset(asset_name: str):
    root = _bone("_rootJoint", None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.1))
    bones = [root]
    chains = {"spine": [], "leg": {"left": [], "right": [], "unsided": []}, "arm": {"left": [], "right": [], "unsided": []}}
    for prefix in ("Hero000", "Hero001"):
        char_bones, char_chains = _prefixed_biped(prefix, "_rootJoint")
        bones += char_bones
        chains["spine"] += char_chains["spine"]
        for side in ("left", "right", "unsided"):
            chains["leg"][side] += char_chains["leg"][side]
            chains["arm"][side] += char_chains["arm"][side]
    return _write_single_armature_asset(asset_name, "Object_4", bones, chains)


class CrowdDecompositionTests(unittest.TestCase):
    def test_multi_character_emits_characters_array(self):
        asset_dir = _write_crowd_asset("crowd_demo")
        report, build_plan, _, _ = write_asset_report(asset_dir)

        decomposition = report["asset_summary"]["character_decomposition"]
        self.assertTrue(decomposition["detected"])
        self.assertEqual(decomposition["character_ids"], ["Hero000", "Hero001"])
        self.assertEqual([c["character_id"] for c in report["characters"]], ["Hero000", "Hero001"])
        self.assertEqual([c["character_id"] for c in build_plan["characters"]], ["Hero000", "Hero001"])
        self.assertNotIn("semantic_mapping", report)

    def test_each_character_maps_its_own_hip(self):
        asset_dir = _write_crowd_asset("crowd_demo2")
        report, _, _, _ = write_asset_report(asset_dir)
        for character in report["characters"]:
            hip = character["semantic_mapping"].get("Hip", {})
            self.assertEqual(hip.get("source_bone"), "Pelvis")

    def test_single_character_has_no_characters_key(self):
        bones, chains = _biped_character()
        asset_dir = _write_single_armature_asset("single_biped", "Bip01", bones, chains)
        report, build_plan, _, _ = write_asset_report(asset_dir)
        self.assertNotIn("characters", report)
        self.assertNotIn("character_decomposition", report["asset_summary"])
        self.assertIn("semantic_mapping", report)


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
        asam_armatures = {item["armature_name"]: item for item in report["armatures"]}
        self.assertEqual(asam_armatures["Armature"]["detected_convention"], "none")
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
        self.assertEqual(armatures["rig"]["detected_convention"], "none")
        self.assertEqual(armatures["metarig"]["detected_convention"], "none")

        rig_summary = armatures["rig"]["summary"]
        metarig_summary = armatures["metarig"]["summary"]

        self.assertEqual(rig_summary["mesh_bound_term"], 10.0)
        self.assertEqual(metarig_summary["mesh_bound_term"], 0.0)
        self.assertGreater(rig_summary["deform_evidence_term"], 0.0)
        self.assertEqual(metarig_summary["deform_evidence_term"], 0.0)
        self.assertGreater(rig_summary["ranking_score"], metarig_summary["ranking_score"])
        self.assertIn("deform_evidence", rig_summary)
        self.assertTrue(rig_summary["deform_evidence"]["has_armature_modifier_link"])
        self.assertFalse(metarig_summary["deform_evidence"]["has_armature_modifier_link"])

        self.assertEqual(armatures["rig"]["selected_inputs"]["primary"], "rig_filtered.json")
        self.assertEqual(armatures["rig"]["selected_inputs"]["support"], "rig_all.json")
        self.assertEqual(report["semantic_mapping"]["Root"]["action"], "repair_in_builder")
        # Hip is reassigned from the lateral pelvis pair to the spine-root pivot
        # (DEF-spine); Lower_Spine shifts up to the next spine segment so the two no
        # longer collide. The displaced pelvis pair is recorded for preservation.
        self.assertEqual(report["semantic_mapping"]["Hip"]["action"], "alias_map")
        self.assertEqual(report["semantic_mapping"]["Hip"]["source_bone"], "DEF-spine")
        self.assertIn("hip_reassigned_to_spine_root_pivot", report["semantic_mapping"]["Hip"]["notes"])
        self.assertEqual(
            report["semantic_mapping"]["Hip"]["preserved_pelvis_pair_sources"],
            ["DEF-pelvis.L", "DEF-pelvis.R"],
        )
        self.assertEqual(report["semantic_mapping"]["Lower_Spine"]["source_bone"], "DEF-spine.001")
        self.assertEqual(build_plan["root_resolutions"][0]["mode"], "create_new_root")
        self.assertEqual(build_plan["root_resolutions"][0]["source_bone"], "root")
        self.assertIn("root_noncompliant", report["review_flags"])
        self.assertIn("multiple_source_roots", build_plan["root_resolutions"][0]["blocker_codes"])
        self.assertIn("root_not_vertical", build_plan["root_resolutions"][0]["blocker_codes"])
        self.assertFalse(any(action["target"] == "Root" for action in build_plan["actions"]["rename"]))
        # Hip now renames the spine-root source bone (alias_map), so it IS a rename.
        self.assertTrue(
            any(
                action["target"] == "Hip" and action["source"] == "DEF-spine"
                for action in build_plan["actions"]["rename"]
            )
        )

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

    def test_cli_entrypoint_missing_inputs_fails_cleanly(self):
        """Empty --asset-dir: exit 1, one-line error + hint, no traceback (#27)."""
        empty_dir = Path(tempfile.mkdtemp(prefix="phase3_classifier_empty_"))
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), "--asset-dir", str(empty_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1, msg=result.stdout + "\n" + result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn("inspector", result.stderr)
        self.assertIn("error:", result.stderr)

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

    def test_summary_emits_coverage_and_purity_fields(self):
        bones = [
            _bone("Hip", None, (0.0, 0.0, 1.0), (0.0, 0.0, 1.1)),
            _bone("Lower_Spine", "Hip", (0.0, 0.0, 1.1), (0.0, 0.0, 1.25)),
            _bone("Upper_Spine", "Lower_Spine", (0.0, 0.0, 1.25), (0.0, 0.0, 1.45)),
        ]
        chains = {
            "spine": [["Lower_Spine", "Upper_Spine"]],
            "leg": {"left": [], "right": [], "unsided": []},
            "arm": {"left": [], "right": [], "unsided": []},
        }
        binding = {
            "armature_object_name": "Rig",
            "meshes": [
                {
                    "mesh_name": "BodyMesh",
                    "modifiers": [{"type": "ARMATURE", "object": "Rig"}],
                    "vertex_groups": ["Hip", "Lower_Spine", "Upper_Spine"],
                }
            ],
        }
        asset_dir = _write_single_armature_asset(
            "coverage_fields", "Rig", bones, chains, None, binding
        )
        report, _, _, _ = write_asset_report(asset_dir)
        summary = report["armatures"][0]["summary"]
        self.assertIn("vertex_group_coverage", summary)
        self.assertIn("deform_purity", summary)
        self.assertEqual(summary["vertex_group_coverage"], 1.0)
        self.assertEqual(summary["deform_purity"], 1.0)
        self.assertEqual(summary["deform_evidence"]["vertex_group_count"], 3)


class ArmatureScoringHelperTests(unittest.TestCase):
    def test_mesh_bound_term_zero_when_meshes_count_zero(self):
        self.assertEqual(_compute_mesh_bound_term(0, False), 0.0)

    def test_mesh_bound_term_zero_even_when_modifier_link_present_but_no_meshes(self):
        self.assertEqual(_compute_mesh_bound_term(0, True), 0.0)

    def test_mesh_bound_term_partial_when_meshes_present_but_no_modifier_link(self):
        self.assertEqual(_compute_mesh_bound_term(1, False), 6.0)

    def test_mesh_bound_term_full_credit_when_modifier_link_present(self):
        self.assertEqual(_compute_mesh_bound_term(1, True), 10.0)

    def test_deform_evidence_term_zero_when_no_hits(self):
        self.assertEqual(_compute_deform_evidence_term(0), 0.0)

    def test_deform_evidence_term_log_dampened_for_small_counts(self):
        self.assertEqual(_compute_deform_evidence_term(3), round(math.log1p(3) * 2.0, 3))

    def test_deform_evidence_term_caps_at_six(self):
        self.assertEqual(_compute_deform_evidence_term(1000), 6.0)

    def test_collect_mesh_binding_stats_handles_none(self):
        result = _collect_mesh_binding_stats(None, "rig", {"spine"})
        self.assertEqual(
            result,
            {
                "meshes_count": 0,
                "has_armature_modifier_link": False,
                "vertex_group_bone_hits": 0,
                "vertex_group_count": 0,
            },
        )

    def test_collect_mesh_binding_stats_empty_meshes(self):
        binding = {"armature_object_name": "rig", "meshes": []}
        result = _collect_mesh_binding_stats(binding, "rig", {"spine"})
        self.assertEqual(result["meshes_count"], 0)
        self.assertFalse(result["has_armature_modifier_link"])
        self.assertEqual(result["vertex_group_bone_hits"], 0)

    def test_collect_mesh_binding_stats_full(self):
        binding = {
            "armature_object_name": "rig",
            "meshes": [
                {
                    "mesh_name": "Body",
                    "modifiers": [
                        {"stack_index": 0, "type": "ARMATURE", "name": "Armature", "object": "rig"}
                    ],
                    "vertex_groups": ["DEF-spine", "head", "Hair_Front"],
                }
            ],
        }
        result = _collect_mesh_binding_stats(binding, "rig", {"spine", "head"})
        self.assertEqual(result["meshes_count"], 1)
        self.assertTrue(result["has_armature_modifier_link"])
        self.assertEqual(result["vertex_group_bone_hits"], 2)

    def test_collect_mesh_binding_stats_reports_vertex_group_count(self):
        binding = {
            "meshes": [
                {
                    "modifiers": [{"type": "ARMATURE", "object": "rig"}],
                    "vertex_groups": ["spine", "head", "extra_mask"],
                }
            ]
        }
        result = _collect_mesh_binding_stats(binding, "rig", {"spine", "head"})
        self.assertEqual(result["vertex_group_count"], 3)
        self.assertEqual(result["vertex_group_bone_hits"], 2)

    def test_collect_mesh_binding_stats_modifier_to_other_armature(self):
        binding = {
            "armature_object_name": "rig",
            "meshes": [
                {
                    "mesh_name": "Body",
                    "modifiers": [
                        {"stack_index": 0, "type": "ARMATURE", "name": "Armature", "object": "other_rig"}
                    ],
                    "vertex_groups": ["spine"],
                }
            ],
        }
        result = _collect_mesh_binding_stats(binding, "rig", {"spine"})
        self.assertFalse(result["has_armature_modifier_link"])
        self.assertEqual(result["meshes_count"], 1)
        self.assertEqual(result["vertex_group_bone_hits"], 1)

    def test_extras_term_zero_when_empty(self):
        self.assertEqual(_compute_extras_term([]), 0.0)

    def test_extras_term_log_dampened_for_small_counts(self):
        extras = [{"bone_name": "spine.001"}, {"bone_name": "spine.002"}, {"bone_name": "spine.003"}]
        expected = round(min(math.log1p(3) * 0.5, 2.0), 3)
        self.assertEqual(_compute_extras_term(extras), expected)

    def test_extras_term_caps_at_two(self):
        extras = [{"bone_name": "bone_{}".format(i)} for i in range(200)]
        self.assertEqual(_compute_extras_term(extras), 2.0)


class SkinWeightSignalTests(unittest.TestCase):
    def test_coverage_is_hits_over_vertex_group_count(self):
        self.assertEqual(_compute_vertex_group_coverage(8, 10), 0.8)

    def test_coverage_full_when_all_groups_hit(self):
        self.assertEqual(_compute_vertex_group_coverage(20, 20), 1.0)

    def test_coverage_zero_guard_when_no_vertex_groups(self):
        self.assertEqual(_compute_vertex_group_coverage(0, 0), 0.0)

    def test_purity_is_hits_over_bone_count(self):
        self.assertEqual(_compute_deform_purity(20, 80), 0.25)

    def test_purity_full_for_pure_deform_rig(self):
        self.assertEqual(_compute_deform_purity(20, 20), 1.0)

    def test_purity_zero_guard_when_no_bones(self):
        self.assertEqual(_compute_deform_purity(0, 0), 0.0)


class ArmatureSelectionCascadeTests(unittest.TestCase):
    def test_lowpoly_selection_tiebreaker_audit_field(self):
        asset_dir = _copy_asset_folder("LowPolyCharacter4")
        report, _, _, _ = write_asset_report(asset_dir)

        ranking = report["asset_summary"]["ranking"]
        self.assertEqual(ranking[0]["armature_name"], "rig")
        # rig is mesh-bound and metarig is not, so the binding step of the cascade
        # decides the pick (more truthful than the old bundled-"score" audit).
        self.assertEqual(ranking[0]["selection_tiebreaker"], "mesh_bound_term")
        self.assertNotIn("selection_tiebreaker", ranking[1])

    def test_lowpoly_ranking_entries_include_new_terms(self):
        asset_dir = _copy_asset_folder("LowPolyCharacter4")
        report, _, _, _ = write_asset_report(asset_dir)

        for entry in report["asset_summary"]["ranking"]:
            self.assertIn("mesh_bound_term", entry)
            self.assertIn("deform_evidence_term", entry)
            self.assertIn("extras_term", entry)

    def test_sole_armature_reports_sole_candidate_tiebreaker(self):
        asset_dir = _copy_asset_folder("openmatexamplehuman")
        report, _, _, _ = write_asset_report(asset_dir)

        ranking = report["asset_summary"]["ranking"]
        self.assertEqual(len(ranking), 1)
        self.assertEqual(ranking[0]["selection_tiebreaker"], "sole_candidate")

    def _make_report(self, name, score, mesh_bound, coverage, purity, mapped, avg_conf):
        return {
            "armature_name": name,
            "summary": {
                "ranking_score": score,
                "mesh_bound_term": mesh_bound,
                "vertex_group_coverage": coverage,
                "deform_purity": purity,
                "mapped_core_targets": mapped,
                "average_confidence": avg_conf,
            },
        }

    def test_tiebreaker_sole_candidate(self):
        reports = [self._make_report("only", 100.0, 10.0, 1.0, 1.0, 22, 0.9)]
        self.assertEqual(_compute_selection_tiebreaker(reports), "sole_candidate")

    def test_tiebreaker_mesh_bound_decides_first(self):
        reports = [
            self._make_report("bound", 90.0, 10.0, 0.5, 0.5, 20, 0.9),
            self._make_report("unbound", 120.0, 0.0, 0.0, 0.0, 28, 0.95),
        ]
        # Even though "unbound" has a higher ranking_score, binding leads the cascade.
        self.assertEqual(_compute_selection_tiebreaker(reports), "mesh_bound_term")

    def test_tiebreaker_coverage_when_binding_ties(self):
        reports = [
            self._make_report("deform", 100.0, 10.0, 1.0, 1.0, 22, 0.9),
            self._make_report("control", 120.0, 10.0, 0.1, 0.1, 28, 0.95),
        ]
        self.assertEqual(_compute_selection_tiebreaker(reports), "vertex_group_coverage")

    def test_tiebreaker_purity_when_coverage_ties(self):
        reports = [
            self._make_report("deform", 100.0, 10.0, 1.0, 1.0, 22, 0.9),
            self._make_report("superset", 120.0, 10.0, 1.0, 0.2, 28, 0.95),
        ]
        self.assertEqual(_compute_selection_tiebreaker(reports), "deform_purity")

    def test_tiebreaker_score_when_binding_and_skin_signals_tie(self):
        reports = [
            self._make_report("a", 110.0, 0.0, 0.0, 0.0, 24, 0.9),
            self._make_report("b", 100.0, 0.0, 0.0, 0.0, 22, 0.9),
        ]
        # No armature bound: cascade falls through to ranking_score (today's behavior).
        self.assertEqual(_compute_selection_tiebreaker(reports), "score")

    def test_tiebreaker_mapped_targets_when_score_ties(self):
        reports = [
            self._make_report("a", 100.0, 10.0, 1.0, 1.0, 24, 0.9),
            self._make_report("b", 100.0, 10.0, 1.0, 1.0, 22, 0.9),
        ]
        self.assertEqual(_compute_selection_tiebreaker(reports), "mapped_core_targets")

    def test_tiebreaker_confidence_when_mapped_ties(self):
        reports = [
            self._make_report("a", 100.0, 10.0, 1.0, 1.0, 22, 0.95),
            self._make_report("b", 100.0, 10.0, 1.0, 1.0, 22, 0.9),
        ]
        self.assertEqual(_compute_selection_tiebreaker(reports), "average_confidence")

    def test_tiebreaker_armature_name_when_all_numeric_signals_tie(self):
        reports = [
            self._make_report("a", 100.0, 10.0, 1.0, 1.0, 22, 0.9),
            self._make_report("b", 100.0, 10.0, 1.0, 1.0, 22, 0.9),
        ]
        self.assertEqual(_compute_selection_tiebreaker(reports), "armature_name")


class ConventionDetectionTests(unittest.TestCase):
    @staticmethod
    def _data(names):
        return {"bones": [{"name": n, "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]} for n in names]}

    def test_detects_mixamo_from_namespace(self):
        data = self._data(["mixamorig:Hips", "mixamorig:LeftUpLeg"])
        self.assertEqual(_detect_convention(data, None), "mixamo")

    def test_detects_biped_from_bip_token(self):
        data = self._data(["Bip01 COM", "Bip01 Pelvis", "Bip01 LThigh"])
        self.assertEqual(_detect_convention(data, None), "biped")

    def test_detects_biped_from_com_and_pelvis(self):
        data = self._data(["COM", "Pelvis", "Spine0"])
        self.assertEqual(_detect_convention(data, None), "biped")

    def test_rigify_and_asam_detect_none(self):
        rigify = self._data(["DEF-spine", "DEF-pelvis.L", "DEF-pelvis.R", "DEF-upper_arm.L"])
        asam = self._data(["Root", "Hip", "Lower_Spine", "Head"])
        self.assertEqual(_detect_convention(rigify, None), "none")
        self.assertEqual(_detect_convention(asam, None), "none")

    def test_biped_prefix_dropped_keeps_compact_clean(self):
        # Without the drop, compact would be "bip01pelvis" and miss the alias.
        _, compact, tokens, _ = _normalize_bone_name("Bip01 Pelvis", "biped")
        self.assertEqual(compact, "pelvis")
        self.assertNotIn("bip01", tokens)

    def test_biped_prefix_kept_when_convention_none(self):
        _, compact, _, _ = _normalize_bone_name("Bip01 Pelvis", "none")
        self.assertEqual(compact, "bip01pelvis")

    def test_leading_side_still_detected_with_biped_prefix(self):
        _, compact, _, side = _normalize_bone_name("Bip01 LThigh", "biped")
        self.assertEqual(compact, "thigh")
        self.assertEqual(side, "left")

    def test_mixamo_upleg_gets_upper_leg_tag(self):
        # "upleg" compact matches no generic rule; alias must add the tag.
        tags = _detect_family_tags("upleg", "upleg", ["up", "leg"], "mixamo")
        self.assertIn("upper_leg", tags)

    def test_biped_com_gets_root_tag(self):
        tags = _detect_family_tags("com", "com", ["com"], "biped")
        self.assertIn("root", tags)

    def test_alias_tags_skip_when_convention_none(self):
        tags = _detect_family_tags("upleg", "upleg", ["up", "leg"], "none")
        self.assertNotIn("upper_leg", tags)

    @staticmethod
    def _bone_info(compact, tokens, side=None):
        return BoneInfo(
            name=compact, parent=None, head=(0, 0, 0), tail=(0, 0, 1), length=1.0,
            origin="primary", normalized_name="_".join(tokens), compact_name=compact,
            tokens=list(tokens), side=side, midpoint=(0, 0, 0.5), min_point=(0, 0, 0),
            max_point=(0, 0, 1), role="deform", role_modifier=1.0, family_tags=set(),
        )

    def test_mixamo_arm_is_upper_not_lower(self):
        bone = self._bone_info("arm", ["arm"])
        self.assertEqual(_family_name_score("upper_arm", bone, "mixamo"), 0.95)
        self.assertEqual(_family_name_score("lower_arm", bone, "mixamo"), 0.0)

    def test_mixamo_leg_is_lower_not_upper(self):
        bone = self._bone_info("leg", ["leg"])
        self.assertEqual(_family_name_score("lower_leg", bone, "mixamo"), 0.95)
        self.assertEqual(_family_name_score("upper_leg", bone, "mixamo"), 0.0)

    def test_biped_com_scores_root(self):
        bone = self._bone_info("com", ["com"])
        self.assertEqual(_family_name_score("root", bone, "biped"), 0.95)

    def test_non_alias_token_falls_through_to_generic(self):
        # "wrist" has no alias entry; generic hand rule (0.82) must still apply.
        bone = self._bone_info("wrist", ["wrist"])
        self.assertEqual(_family_name_score("hand", bone, "mixamo"), 0.82)


class NameQualityTests(unittest.TestCase):
    @staticmethod
    def _bone(compact, tokens, side=None, family_tags=None, origin="primary"):
        return BoneInfo(
            name=compact, parent=None, head=(0, 0, 0), tail=(0, 0, 1), length=1.0,
            origin=origin, normalized_name="_".join(tokens), compact_name=compact,
            tokens=list(tokens), side=side, midpoint=(0, 0, 0.5), min_point=(0, 0, 0),
            max_point=(0, 0, 1), role="deform", role_modifier=1.0,
            family_tags=set(family_tags or ()),
        )

    def test_junk_names_with_lr_substrings_score_low(self):
        # Regression: junk names that merely *contain* the letters l/r
        # (controlnode, qqlrx, lurkblob) must NOT earn side credit. A bare
        # substring scan wrongly inflated name_quality, flipping junk rigs to
        # the name-driven weighting branch. They carry no family tags and
        # BoneInfo.side is None, so quality must stay LOW (< 0.2).
        bones = {
            name: self._bone(name, [name])
            for name in ("controlnode", "qqlrx", "lurkblob", "rrll", "flurb")
        }
        quality = compute_name_quality({"bones": bones})
        self.assertLess(quality, 0.2)

    def test_real_side_tokens_earn_credit(self):
        # A genuine side token (side set, no family tag) earns partial credit.
        bones = {"thing_l": self._bone("thingl", ["thing"], side="left")}
        self.assertAlmostEqual(compute_name_quality({"bones": bones}), 0.3)

    def test_family_named_bones_score_high(self):
        bones = {"pelvis": self._bone("pelvis", ["pelvis"], family_tags={"hip"})}
        self.assertEqual(compute_name_quality({"bones": bones}), 1.0)


class ConventionReportTests(unittest.TestCase):
    def test_detected_convention_in_report(self):
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
            "leg": {"left": [["Upper_Leg_Left"]], "right": [["Upper_Leg_Right"]], "unsided": []},
            "arm": {"left": [], "right": [], "unsided": []},
        }
        asset_dir = _write_single_armature_asset(
            "convention_none_report", "Rig", bones, chains,
            _placement_metadata((-0.2, -0.25, 0.0), (0.2, 0.25, 1.8)),
        )
        report, _, _, _ = write_asset_report(asset_dir)
        armatures = {item["armature_name"]: item for item in report["armatures"]}
        self.assertEqual(armatures["Rig"]["detected_convention"], "none")

    def test_biped_character_maps_to_asam_targets(self):
        bones, chains = _biped_character()
        asset_dir = _write_single_armature_asset(
            "biped_single", "Bip01", bones, chains,
            _placement_metadata((-0.3, -0.4, 0.0), (0.4, 0.4, 1.8)),
        )
        report, _, _, _ = write_asset_report(asset_dir)
        armatures = {item["armature_name"]: item for item in report["armatures"]}
        self.assertEqual(armatures["Bip01"]["detected_convention"], "biped")

        mapping = report["semantic_mapping"]
        self.assertEqual(mapping["Hip"]["source_bone"], "Bip01 Pelvis")
        self.assertEqual(mapping["Lower_Spine"]["source_bone"], "Bip01 Spine0")
        self.assertEqual(mapping["Upper_Spine"]["source_bone"], "Bip01 Spine1")
        self.assertEqual(mapping["Shoulder_Left"]["source_bone"], "Bip01 LClavicle")
        self.assertEqual(mapping["Upper_Arm_Left"]["source_bone"], "Bip01 LUpperArm")
        self.assertEqual(mapping["Lower_Arm_Left"]["source_bone"], "Bip01 LForeArm")
        self.assertEqual(mapping["Upper_Leg_Right"]["source_bone"], "Bip01 RThigh")
        self.assertEqual(mapping["Lower_Leg_Left"]["source_bone"], "Bip01 LCalf")
        self.assertEqual(mapping["Root"]["source_bone"], "Bip01 COM")
        # Neck/Head are absent from the biped alias map and must still resolve via
        # the generic name scorer (the convention fall-through path).
        self.assertEqual(mapping["Neck"]["source_bone"], "Bip01 Neck")
        self.assertEqual(mapping["Head"]["source_bone"], "Bip01 Head")

    def test_mixamo_character_resolves_arm_leg_inversion(self):
        bones, chains = _mixamo_character()
        asset_dir = _write_single_armature_asset(
            "mixamo_single", "Armature", bones, chains,
            _placement_metadata((-0.3, -0.4, 0.0), (0.4, 0.4, 1.8)),
        )
        report, _, _, _ = write_asset_report(asset_dir)
        armatures = {item["armature_name"]: item for item in report["armatures"]}
        self.assertEqual(armatures["Armature"]["detected_convention"], "mixamo")

        mapping = report["semantic_mapping"]
        self.assertEqual(mapping["Hip"]["source_bone"], "mixamorig:Hips")
        # The inversion fix: Arm=upper, ForeArm=lower, UpLeg=upper, Leg=lower.
        self.assertEqual(mapping["Upper_Arm_Left"]["source_bone"], "mixamorig:LeftArm")
        self.assertEqual(mapping["Lower_Arm_Left"]["source_bone"], "mixamorig:LeftForeArm")
        self.assertEqual(mapping["Upper_Leg_Left"]["source_bone"], "mixamorig:LeftUpLeg")
        self.assertEqual(mapping["Lower_Leg_Left"]["source_bone"], "mixamorig:LeftLeg")
        self.assertEqual(mapping["Upper_Leg_Right"]["source_bone"], "mixamorig:RightUpLeg")
        self.assertEqual(mapping["Lower_Leg_Right"]["source_bone"], "mixamorig:RightLeg")
        self.assertEqual(mapping["Shoulder_Left"]["source_bone"], "mixamorig:LeftShoulder")
        self.assertEqual(mapping["Full_Toes_Left"]["source_bone"], "mixamorig:LeftToeBase")


class FixtureStabilityTests(unittest.TestCase):
    def test_known_fixtures_stay_single_character(self):
        for asset_name in ("LowPolyCharacter4", "openmatexamplehuman"):
            asset_dir = _copy_asset_folder(asset_name)
            report, build_plan, _, _ = write_asset_report(asset_dir)
            self.assertNotIn("characters", report, asset_name)
            self.assertNotIn("character_decomposition", report["asset_summary"], asset_name)
            self.assertNotIn("characters", build_plan, asset_name)
            self.assertIn("semantic_mapping", report, asset_name)


class MultiArmatureSelectionTests(unittest.TestCase):
    def test_deform_rig_wins_over_bushier_control_rig_on_binding_tie(self):
        # DeformRig: mixamo (no toes) -> its bone names ARE the mesh's vertex groups.
        deform_bones, deform_chains = _mixamo_character_without_toes()
        deform_vgs = [b["name"] for b in deform_bones]
        # ControlRig: full biped (incl. toes) -> maps MORE ASAM targets (higher
        # ranking_score) but its biped names do NOT match the mixamo vertex groups.
        control_bones, control_chains = _biped_character()

        asset_dir = _write_multi_armature_asset(
            "deform_vs_control",
            [
                ("DeformRig", deform_bones, deform_chains, _mesh_binding_for("DeformRig", deform_vgs)),
                ("ControlRig", control_bones, control_chains, _mesh_binding_for("ControlRig", deform_vgs)),
            ],
        )
        report, build_plan, _, _ = write_asset_report(asset_dir)

        self.assertEqual(report["recommended_primary_armature"], "DeformRig")
        self.assertEqual(build_plan["recommended_primary_armature"], "DeformRig")

        ranking = report["asset_summary"]["ranking"]
        top = ranking[0]
        self.assertEqual(top["armature_name"], "DeformRig")
        self.assertEqual(top["selection_tiebreaker"], "vertex_group_coverage")
        by_name = {entry["armature_name"]: entry for entry in ranking}
        self.assertEqual(by_name["DeformRig"]["mesh_bound_term"], 10.0)
        self.assertEqual(by_name["ControlRig"]["mesh_bound_term"], 10.0)
        self.assertGreater(
            by_name["DeformRig"]["vertex_group_coverage"],
            by_name["ControlRig"]["vertex_group_coverage"],
        )
        # The deform rig wins DESPITE a lower bundled ranking_score (pre-#32 the
        # bushier control rig would have been selected).
        self.assertGreater(
            by_name["ControlRig"]["ranking_score"],
            by_name["DeformRig"]["ranking_score"],
        )

    def test_superset_control_rig_loses_on_deform_purity(self):
        deform_bones, deform_chains = _mixamo_character()
        deform_vgs = [b["name"] for b in deform_bones]
        # ControlRig contains all deform names (coverage ~1.0 too) + 60 scaffolding bones.
        control_bones, control_chains = _mixamo_with_extra_control_bones(60)

        asset_dir = _write_multi_armature_asset(
            "superset_control",
            [
                ("DeformRig", deform_bones, deform_chains, _mesh_binding_for("DeformRig", deform_vgs)),
                ("ControlRig", control_bones, control_chains, _mesh_binding_for("ControlRig", deform_vgs)),
            ],
        )
        report, _, _, _ = write_asset_report(asset_dir)

        self.assertEqual(report["recommended_primary_armature"], "DeformRig")
        top = report["asset_summary"]["ranking"][0]
        self.assertEqual(top["selection_tiebreaker"], "deform_purity")
        by_name = {e["armature_name"]: e for e in report["asset_summary"]["ranking"]}
        self.assertEqual(
            by_name["DeformRig"]["vertex_group_coverage"],
            by_name["ControlRig"]["vertex_group_coverage"],
        )
        self.assertGreater(
            by_name["DeformRig"]["deform_purity"],
            by_name["ControlRig"]["deform_purity"],
        )

    def test_no_binding_falls_back_to_ranking_score(self):
        full_bones, full_chains = _biped_character()
        partial_bones, partial_chains = _mixamo_character_without_toes()
        # Neither armature has a mesh_binding -> both mesh_bound_term == 0.
        asset_dir = _write_multi_armature_asset(
            "no_binding",
            [
                ("FullRig", full_bones, full_chains, None),
                ("PartialRig", partial_bones, partial_chains, None),
            ],
        )
        report, _, _, _ = write_asset_report(asset_dir)

        ranking = report["asset_summary"]["ranking"]
        by_name = {e["armature_name"]: e for e in ranking}
        self.assertEqual(by_name["FullRig"]["mesh_bound_term"], 0.0)
        self.assertEqual(by_name["PartialRig"]["mesh_bound_term"], 0.0)
        # No binding: the better-mapped rig wins on the bundled scalar, as today.
        winner = report["recommended_primary_armature"]
        self.assertEqual(winner, max(by_name, key=lambda n: by_name[n]["ranking_score"]))
        self.assertEqual(ranking[0]["selection_tiebreaker"], "score")


class ReconciliationDecisionTests(unittest.TestCase):
    def test_structure_confirmed_by_name(self):
        self.assertEqual(_reconciliation_tag(0.8, 0.6), "structure_confirmed_by_name")
        self.assertEqual(_reconciliation_tag(0.20, 0.20), "structure_confirmed_by_name")

    def test_name_override(self):
        self.assertEqual(_reconciliation_tag(0.9, 0.0), "name_override")
        self.assertEqual(_reconciliation_tag(0.20, 0.19), "name_override")

    def test_structure_only(self):
        self.assertEqual(_reconciliation_tag(0.0, 0.9), "structure_only")
        self.assertEqual(_reconciliation_tag(0.19, 0.20), "structure_only")

    def test_conflict_review(self):
        self.assertEqual(_reconciliation_tag(0.0, 0.0), "conflict_review")
        self.assertEqual(_reconciliation_tag(0.19, 0.19), "conflict_review")

    def test_clean_fixture_hip_is_structure_confirmed_by_name(self):
        # End-to-end: a clean, well-named fixture target whose name AND structure
        # agree must surface decision == structure_confirmed_by_name.
        asset_dir = _copy_asset_folder("openmatexamplehuman")
        report, _plan, _rp, _pp = write_asset_report(asset_dir)
        targets = report.get("semantic_mapping")
        if targets is None:
            targets = report["characters"][0]["asam_targets"]
        self.assertEqual(targets["Hip"]["decision"], "structure_confirmed_by_name")

    def test_twist_bone_runner_up_guard_stem_equality(self):
        from phase3_classifier.classifier import CandidateScore, _decision_tag

        # Base candidate
        cand = CandidateScore(
            source_bone="DEF-arm",
            source_origin="primary",
            role="deform",
            selection_score=0.9,
            confidence=0.85,
            name_evidence=0.8,
            hierarchy_evidence=0.5,
            geometry_evidence=0.5,
            structural_evidence=0.5,
            notes=[],
        )

        # 1. Genuine twist-bone runner-up: DEF-arm vs DEF-arm.001
        ru_twin = CandidateScore(
            source_bone="DEF-arm.001",
            source_origin="primary",
            role="deform",
            selection_score=0.9,
            confidence=0.85,
            name_evidence=0.8,
            hierarchy_evidence=0.5,
            geometry_evidence=0.5,
            structural_evidence=0.5,
            notes=[],
        )
        candidates_by_target = {"Arm_Left": [cand, ru_twin]}
        decision = _decision_tag("Arm_Left", cand, candidates_by_target, {})
        # With stem equality, DEF-arm.001 is collapsed -> separation = 1.0 -> direct_map
        self.assertEqual(decision, "direct_map")

        # 2. Same prefix, distinct bone: DEF-arm vs DEF-arm_helper
        ru_distinct = CandidateScore(
            source_bone="DEF-arm_helper",
            source_origin="primary",
            role="deform",
            selection_score=0.9,
            confidence=0.85,
            name_evidence=0.8,
            hierarchy_evidence=0.5,
            geometry_evidence=0.5,
            structural_evidence=0.5,
            notes=[],
        )
        candidates_by_target = {"Arm_Left": [cand, ru_distinct]}
        decision = _decision_tag("Arm_Left", cand, candidates_by_target, {})
        # With stem equality, DEF-arm_helper is NOT collapsed -> separation = 0.0 -> review
        self.assertEqual(decision, "review")

        # 3. Suffixes on both: DEF-arm.001 vs DEF-arm.002
        cand_suffix = CandidateScore(
            source_bone="DEF-arm.001",
            source_origin="primary",
            role="deform",
            selection_score=0.9,
            confidence=0.85,
            name_evidence=0.8,
            hierarchy_evidence=0.5,
            geometry_evidence=0.5,
            structural_evidence=0.5,
            notes=[],
        )
        ru_suffix = CandidateScore(
            source_bone="DEF-arm.002",
            source_origin="primary",
            role="deform",
            selection_score=0.9,
            confidence=0.85,
            name_evidence=0.8,
            hierarchy_evidence=0.5,
            geometry_evidence=0.5,
            structural_evidence=0.5,
            notes=[],
        )
        candidates_by_target = {"Arm_Left": [cand_suffix, ru_suffix]}
        decision = _decision_tag("Arm_Left", cand_suffix, candidates_by_target, {})
        # Stems are equal -> separation = 1.0 -> direct_map
        self.assertEqual(decision, "direct_map")


class MappingPreservationTests(unittest.TestCase):
    maxDiff = None

    def _mapping(self, report):
        targets = report.get("semantic_mapping")
        if targets is None:
            targets = report["characters"][0]["asam_targets"]
        return {t: (p["source_bone"], p["action"]) for t, p in targets.items()}

    def test_committed_fixture_mapping_is_preserved(self):
        # D1 invariant: re-running the classifier on a fixture reproduces the
        # committed per-target (source_bone, action) mapping byte-for-byte.
        #
        # Both goldens now track the hierarchy-first structural behaviour shipped
        # on this branch: openmatexamplehuman tracks the current schema, and
        # LowPolyCharacter4 was re-baselined through the real pipeline once the
        # clean-rig regressions (control 'chest' out-naming the DEF upper-spine
        # segment; a non-finger arm bone leaking into Full_Thumb_*) were fixed.
        for asset in ("openmatexamplehuman", "LowPolyCharacter4"):
            golden_path = FIXTURE_ROOT / asset / "classifier_report.json"
            with golden_path.open(encoding="utf-8") as handle:
                golden = json.load(handle)
            asset_dir = _copy_asset_folder(asset)
            report, _plan, _rp, _pp = write_asset_report(asset_dir)
            self.assertEqual(self._mapping(report), self._mapping(golden), asset)

    def test_lowpoly_mapping_is_deterministic(self):
        # The classifier must also produce a stable mapping run-to-run.
        first, _p1, _r1, _pp1 = write_asset_report(_copy_asset_folder("LowPolyCharacter4"))
        second, _p2, _r2, _pp2 = write_asset_report(_copy_asset_folder("LowPolyCharacter4"))
        self.assertEqual(self._mapping(first), self._mapping(second))


class CleanRigRegressionTests(unittest.TestCase):
    """Pin the corrected mapping of the clean Rigify rig LowPolyCharacter4.

    Guards two regressions introduced by the hierarchy-first structural rewrite
    (a deform upper-spine bone dropping to a control 'chest'/review; a non-finger
    arm bone leaking into the Full_Thumb_* targets) AND the two accepted
    improvements from the same rewrite (Neck -> 'neck', Shoulder -> 'DEF-shoulder.*')
    against re-regression.
    """

    @classmethod
    def setUpClass(cls):
        report, _plan, _rp, _pp = write_asset_report(_copy_asset_folder("LowPolyCharacter4"))
        cls.targets = report["semantic_mapping"]

    def test_upper_spine_is_deform_spine(self):
        payload = self.targets["Upper_Spine"]
        self.assertIsNotNone(payload["source_bone"])
        self.assertTrue(
            payload["source_bone"].startswith("DEF-spine"),
            "Upper_Spine must map to a deform spine bone, got {0}".format(payload["source_bone"]),
        )
        self.assertIn(payload["action"], {"direct_map", "alias_map"})

    def test_full_thumb_has_no_candidate(self):
        # This rig carries no real thumb/finger bone; the thumb targets must stay
        # unmapped rather than grabbing a non-finger arm bone.
        self.assertIsNone(self.targets["Full_Thumb_Left"]["source_bone"])
        self.assertIsNone(self.targets["Full_Thumb_Right"]["source_bone"])

    def test_accepted_neck_and_shoulder_improvements(self):
        self.assertEqual(self.targets["Neck"]["source_bone"], "neck")
        self.assertEqual(self.targets["Shoulder_Left"]["source_bone"], "DEF-shoulder.L")
        self.assertEqual(self.targets["Shoulder_Right"]["source_bone"], "DEF-shoulder.R")


class JunkNameBenchmarkTests(unittest.TestCase):
    def _junk_asset(self):
        bones = [
            _bone("b0", None, (0, 0, 0.0), (0, 0, 0.1)),
            _bone("b1", "b0", (0, 0, 1.0), (0, 0, 1.1)),       # hip
            _bone("b2", "b1", (0, 0, 1.1), (0, 0, 1.4)),       # lower spine
            _bone("b3", "b2", (0, 0, 1.4), (0, 0, 1.7)),       # upper spine
            _bone("b4", "b3", (0, 0, 1.7), (0, 0, 1.85)),      # neck
            _bone("b5", "b4", (0, 0, 1.85), (0, 0, 2.0)),      # head
            _bone("aL0", "b3", (0, 0.2, 1.7), (0, 0.5, 1.6)), _bone("aL1", "aL0", (0, 0.5, 1.6), (0, 0.8, 1.5)), _bone("aL2", "aL1", (0, 0.8, 1.5), (0, 0.95, 1.45)),
            _bone("aR0", "b3", (0, -0.2, 1.7), (0, -0.5, 1.6)), _bone("aR1", "aR0", (0, -0.5, 1.6), (0, -0.8, 1.5)), _bone("aR2", "aR1", (0, -0.8, 1.5), (0, -0.95, 1.45)),
            _bone("lL0", "b1", (0, 0.1, 1.0), (0, 0.12, 0.55)), _bone("lL1", "lL0", (0, 0.12, 0.55), (0, 0.13, 0.1)), _bone("lL2", "lL1", (0, 0.13, 0.1), (0.15, 0.13, 0.0)),
            _bone("lR0", "b1", (0, -0.1, 1.0), (0, -0.12, 0.55)), _bone("lR1", "lR0", (0, -0.12, 0.55), (0, -0.13, 0.1)), _bone("lR2", "lR1", (0, -0.13, 0.1), (0.15, -0.13, 0.0)),
        ]
        return _write_single_armature_asset(
            "junkrig", "Rig", bones, {},
            placement_metadata=_placement_metadata((-0.5, -1.0, 0.0), (0.5, 1.0, 2.0)),
            mesh_binding=_mesh_binding_for("Rig", ["b1", "b5", "aL2", "lL2"]),
        )

    def test_structure_maps_core_targets_under_junk_names(self):
        report, _p, _r, _pp = write_asset_report(self._junk_asset())
        arm = report["armatures"][0]
        self.assertLessEqual(arm["name_quality"], 0.2)
        m = arm["asam_targets"]
        self.assertEqual(m["Hip"]["source_bone"], "b1")
        self.assertEqual(m["Head"]["source_bone"], "b5")
        self.assertEqual(m["Hand_Left"]["source_bone"], "aL2")
        self.assertEqual(m["Foot_Right"]["source_bone"], "lR2")


class MissingIntermediateBenchmarkTests(unittest.TestCase):
    def test_two_bone_leg_resolves_upper_and_foot(self):
        bones = [
            _bone("root", None, (0, 0, 0.0), (0, 0, 0.1)), _bone("hip", "root", (0, 0, 1.0), (0, 0, 1.1)),
            _bone("spine", "hip", (0, 0, 1.1), (0, 0, 1.6)), _bone("head", "spine", (0, 0, 1.6), (0, 0, 2.0)),
            _bone("thighL", "hip", (0, 0.1, 1.0), (0, 0.12, 0.2)), _bone("footL", "thighL", (0, 0.12, 0.2), (0.15, 0.13, 0.0)),
            _bone("thighR", "hip", (0, -0.1, 1.0), (0, -0.12, 0.2)), _bone("footR", "thighR", (0, -0.12, 0.2), (0.15, -0.13, 0.0)),
        ]
        asset_dir = _write_single_armature_asset(
            "missingmid", "Rig", bones, {},
            placement_metadata=_placement_metadata((-0.5, -1.0, 0.0), (0.5, 1.0, 2.0)),
            mesh_binding=_mesh_binding_for("Rig", ["hip", "thighL", "footL"]),
        )
        report, _p, _r, _pp = write_asset_report(asset_dir)
        m = report["armatures"][0]["asam_targets"]
        self.assertEqual(m["Upper_Leg_Left"]["source_bone"], "thighL")
        self.assertEqual(m["Foot_Left"]["source_bone"], "footL")
        self.assertEqual(m["Lower_Leg_Left"]["action"], "create_in_builder")


class BipedMultiSpinePromotionTests(unittest.TestCase):
    """A Biped with a 5-segment spine must map BOTH ASAM spine targets from real
    source bones, not collapse them to `review` (issue #55, the 1000idles2 case)."""

    def _five_spine_biped_asset(self, name):
        b = _bone
        bones = [
            b("Bip01 Pelvis", None, (0.0, 0.0, 0.95), (0.0, 0.0, 1.0)),
            b("Bip01 Spine0", "Bip01 Pelvis", (0.0, 0.0, 1.0), (0.0, 0.0, 1.1)),
            b("Bip01 Spine1", "Bip01 Spine0", (0.0, 0.0, 1.1), (0.0, 0.0, 1.2)),
            b("Bip01 Spine2", "Bip01 Spine1", (0.0, 0.0, 1.2), (0.0, 0.0, 1.3)),
            b("Bip01 Spine3", "Bip01 Spine2", (0.0, 0.0, 1.3), (0.0, 0.0, 1.4)),
            b("Bip01 Spine4", "Bip01 Spine3", (0.0, 0.0, 1.4), (0.0, 0.0, 1.5)),
            b("Bip01 Neck", "Bip01 Spine4", (0.0, 0.0, 1.5), (0.0, 0.0, 1.6)),
            b("Bip01 Head", "Bip01 Neck", (0.0, 0.0, 1.6), (0.0, 0.0, 1.75)),
        ]
        chains = {
            "spine": [[
                "Bip01 Spine0", "Bip01 Spine1", "Bip01 Spine2",
                "Bip01 Spine3", "Bip01 Spine4",
            ]],
            "leg": {"left": [], "right": [], "unsided": []},
            "arm": {"left": [], "right": [], "unsided": []},
        }
        return _write_single_armature_asset(name, "Bip01", bones, chains)

    def test_five_spine_biped_promotes_both_spine_targets(self):
        asset_dir = self._five_spine_biped_asset("biped5spine")
        report, _, _, _ = write_asset_report(asset_dir)
        sm = report["semantic_mapping"]
        spine_chain = {
            "Bip01 Spine0", "Bip01 Spine1", "Bip01 Spine2",
            "Bip01 Spine3", "Bip01 Spine4",
        }
        ambiguous = {a["target"] for a in report["ambiguous_targets"]}

        for target in ("Lower_Spine", "Upper_Spine"):
            self.assertEqual(
                sm[target]["action"], "direct_map",
                f"{target} should map directly, not be demoted to review",
            )
            self.assertIn(sm[target]["source_bone"], spine_chain)
            self.assertNotIn(target, ambiguous)

        # Lower spine must sit below upper spine on the up (z) axis.
        with open(asset_dir / "Bip01_all.json") as f:
            bones_by_name = {bn["name"]: bn for bn in json.load(f)["bones"]}
        lower = bones_by_name[sm["Lower_Spine"]["source_bone"]]
        upper = bones_by_name[sm["Upper_Spine"]["source_bone"]]
        self.assertLess(lower["head"][2], upper["head"][2])


class SingleSpineSplitTests(unittest.TestCase):
    """A rig with a single usable spine bone bracketed by a classified Hip and
    Neck must plan a geometric midpoint split into Lower_Spine + Upper_Spine
    (issue #56), instead of mapping one and synthesizing/collapsing the other."""

    def _single_spine_asset(self, name, *, include_neck=True):
        b = _bone
        bones = [
            b("Hips", None, (0.0, 0.0, 0.95), (0.0, 0.0, 1.0)),
            b("Spine", "Hips", (0.0, 0.0, 1.0), (0.0, 0.0, 1.4)),
            b("LeftUpLeg", "Hips", (0.1, 0.0, 0.95), (0.1, 0.0, 0.5)),
            b("LeftLeg", "LeftUpLeg", (0.1, 0.0, 0.5), (0.1, 0.0, 0.1)),
            b("LeftFoot", "LeftLeg", (0.1, 0.0, 0.1), (0.1, 0.1, 0.05)),
            b("RightUpLeg", "Hips", (-0.1, 0.0, 0.95), (-0.1, 0.0, 0.5)),
            b("RightLeg", "RightUpLeg", (-0.1, 0.0, 0.5), (-0.1, 0.0, 0.1)),
            b("RightFoot", "RightLeg", (-0.1, 0.0, 0.1), (-0.1, 0.1, 0.05)),
        ]
        if include_neck:
            bones += [
                b("Neck", "Spine", (0.0, 0.0, 1.4), (0.0, 0.0, 1.5)),
                b("Head", "Neck", (0.0, 0.0, 1.5), (0.0, 0.0, 1.65)),
            ]
        chains = {
            "spine": [["Spine"]],
            "leg": {"left": ["LeftUpLeg", "LeftLeg", "LeftFoot"],
                    "right": ["RightUpLeg", "RightLeg", "RightFoot"],
                    "unsided": []},
            "arm": {"left": [], "right": [], "unsided": []},
        }
        return _write_single_armature_asset(name, "Rig", bones, chains)

    def test_single_spine_plans_split_for_both_targets(self):
        asset_dir = self._single_spine_asset("single_spine")
        report, _, _, _ = write_asset_report(asset_dir)
        sm = report["semantic_mapping"]
        for target, half in (("Lower_Spine", "lower"), ("Upper_Spine", "upper")):
            self.assertEqual(sm[target]["action"], "split_in_builder", target)
            self.assertEqual(sm[target]["source_bone"], "Spine", target)
            self.assertEqual(sm[target]["split_half"], half, target)
        self.assertNotIn("Spine", [b["bone_name"] for b in report["unclassified_bones"]])
        self.assertNotIn("Lower_Spine", {a["target"] for a in report["ambiguous_targets"]})
        self.assertNotIn("Upper_Spine", report["missing_targets"])
        # The split resolves both spine targets, so the rig is not flagged as
        # missing its spine chain.
        self.assertNotIn("missing_spine_chain", report["review_flags"])

    def test_single_spine_without_neck_does_not_split(self):
        asset_dir = self._single_spine_asset("single_spine_no_neck", include_neck=False)
        report, _, _, _ = write_asset_report(asset_dir)
        sm = report["semantic_mapping"]
        self.assertNotEqual(sm["Lower_Spine"]["action"], "split_in_builder")
        self.assertNotEqual(sm["Upper_Spine"]["action"], "split_in_builder")

    def test_single_spine_emits_split_action_in_build_plan(self):
        asset_dir = self._single_spine_asset("single_spine_action")
        _report, build_plan, _r, _pp = write_asset_report(asset_dir)
        self.assertIn("split", build_plan["actions"])
        self.assertEqual(
            build_plan["actions"]["split"],
            [{"source": "Spine", "lower_target": "Lower_Spine",
              "upper_target": "Upper_Spine", "ratio": 0.5}],
        )
        rename_sources = {entry["source"] for entry in build_plan["actions"]["rename"]}
        self.assertNotIn("Spine", rename_sources)

    def test_multi_spine_does_not_split(self):
        b = _bone
        bones = [
            b("Bip01 Pelvis", None, (0.0, 0.0, 0.95), (0.0, 0.0, 1.0)),
            b("Bip01 Spine0", "Bip01 Pelvis", (0.0, 0.0, 1.0), (0.0, 0.0, 1.1)),
            b("Bip01 Spine1", "Bip01 Spine0", (0.0, 0.0, 1.1), (0.0, 0.0, 1.2)),
            b("Bip01 Spine2", "Bip01 Spine1", (0.0, 0.0, 1.2), (0.0, 0.0, 1.3)),
            b("Bip01 Neck", "Bip01 Spine2", (0.0, 0.0, 1.3), (0.0, 0.0, 1.4)),
            b("Bip01 Head", "Bip01 Neck", (0.0, 0.0, 1.4), (0.0, 0.0, 1.55)),
        ]
        chains = {
            "spine": [["Bip01 Spine0", "Bip01 Spine1", "Bip01 Spine2"]],
            "leg": {"left": [], "right": [], "unsided": []},
            "arm": {"left": [], "right": [], "unsided": []},
        }
        asset_dir = _write_single_armature_asset("multispine_nosplit", "Bip01", bones, chains)
        report, build_plan, _r, _pp = write_asset_report(asset_dir)
        sm = report["semantic_mapping"]
        self.assertNotEqual(sm["Lower_Spine"]["action"], "split_in_builder")
        self.assertNotEqual(sm["Upper_Spine"]["action"], "split_in_builder")
        self.assertEqual(build_plan["actions"]["split"], [])


def _guard_bone(head, parent=None):
    # Only .head (index-able by axis) and .parent (name or None) are read by the guard.
    return types.SimpleNamespace(head=list(head), parent=parent)


def _guard_ctx(bones):
    return {
        "bones": bones,
        "height_axis": 2,   # z is up in these fixtures
        "height_sign": 1,
        "height_span": 2.0,
    }


def _guard_payload(source_bone, action="alias_map", notes=None):
    return {
        "source_bone": source_bone,
        "action": action,
        "confidence": 0.7,
        "notes": list(notes or []),
    }


class SpineNeckGuardTests(unittest.TestCase):
    def test_demotes_upper_spine_when_head_coincides_with_neck(self):
        bones = {
            "s1": _guard_bone((0.0, 0.0, 1.0890), parent="hip"),
            "s2": _guard_bone((0.0, 0.0, 1.2145), parent="s1"),
            "s3": _guard_bone((0.0, 0.0, 1.3741), parent="s2"),
            "s4": _guard_bone((0.0, 0.0, 1.5338), parent="s3"),  # head == neck head
            "neck": _guard_bone((0.0, 0.0, 1.5338), parent="mch"),
        }
        resolved = {
            "Lower_Spine": _guard_payload("s1"),
            "Upper_Spine": _guard_payload("s4"),
            "Neck": _guard_payload("neck", action="direct_map"),
        }
        _demote_neck_coincident_upper_spine(resolved, _guard_ctx(bones))
        self.assertEqual(resolved["Upper_Spine"]["source_bone"], "s3")
        self.assertIn(
            "neck_coincident_upper_spine_demoted",
            resolved["Upper_Spine"]["notes"],
        )

    def test_noop_when_upper_spine_already_below_neck(self):
        bones = {
            "s1": _guard_bone((0.0, 0.0, 1.0890), parent="hip"),
            "s3": _guard_bone((0.0, 0.0, 1.3741), parent="s1"),
            "neck": _guard_bone((0.0, 0.0, 1.5338), parent="mch"),
        }
        resolved = {
            "Lower_Spine": _guard_payload("s1"),
            "Upper_Spine": _guard_payload("s3"),
            "Neck": _guard_payload("neck", action="direct_map"),
        }
        _demote_neck_coincident_upper_spine(resolved, _guard_ctx(bones))
        self.assertEqual(resolved["Upper_Spine"]["source_bone"], "s3")
        self.assertEqual(resolved["Upper_Spine"]["notes"], [])

    def test_degenerate_leaves_mapping_and_annotates(self):
        # Only spine bone below neck is Lower_Spine's own source -> nothing valid to pick.
        bones = {
            "s1": _guard_bone((0.0, 0.0, 1.0890), parent="hip"),
            "s4": _guard_bone((0.0, 0.0, 1.5338), parent="s1"),  # head == neck head
            "neck": _guard_bone((0.0, 0.0, 1.5338), parent="mch"),
        }
        resolved = {
            "Lower_Spine": _guard_payload("s1"),
            "Upper_Spine": _guard_payload("s4"),
            "Neck": _guard_payload("neck", action="direct_map"),
        }
        _demote_neck_coincident_upper_spine(resolved, _guard_ctx(bones))
        self.assertEqual(resolved["Upper_Spine"]["source_bone"], "s4")
        self.assertIn(
            "neck_coincident_upper_spine_unresolved",
            resolved["Upper_Spine"]["notes"],
        )

    def test_noop_when_upper_spine_is_split(self):
        # Single-spine split assigns both spine targets to one bone; guard must not touch it.
        bones = {
            "s1": _guard_bone((0.0, 0.0, 1.2000), parent="hip"),
            "neck": _guard_bone((0.0, 0.0, 1.5338), parent="mch"),
        }
        resolved = {
            "Lower_Spine": _guard_payload("s1", action="split_in_builder"),
            "Upper_Spine": _guard_payload("s1", action="split_in_builder"),
            "Neck": _guard_payload("neck", action="direct_map"),
        }
        _demote_neck_coincident_upper_spine(resolved, _guard_ctx(bones))
        self.assertEqual(resolved["Upper_Spine"]["source_bone"], "s1")
        self.assertEqual(resolved["Upper_Spine"]["notes"], [])

    def test_demotes_with_negative_height_sign(self):
        # Up is the -z direction (height_sign=-1): "higher" means more negative z.
        bones = {
            "s1": _guard_bone((0.0, 0.0, -1.0890), parent="hip"),
            "s3": _guard_bone((0.0, 0.0, -1.3741), parent="s1"),
            "s4": _guard_bone((0.0, 0.0, -1.5338), parent="s3"),  # head == neck head
            "neck": _guard_bone((0.0, 0.0, -1.5338), parent="mch"),
        }
        ctx = {"bones": bones, "height_axis": 2, "height_sign": -1, "height_span": 2.0}
        resolved = {
            "Lower_Spine": _guard_payload("s1"),
            "Upper_Spine": _guard_payload("s4"),
            "Neck": _guard_payload("neck", action="direct_map"),
        }
        _demote_neck_coincident_upper_spine(resolved, ctx)
        self.assertEqual(resolved["Upper_Spine"]["source_bone"], "s3")
        self.assertIn(
            "neck_coincident_upper_spine_demoted",
            resolved["Upper_Spine"]["notes"],
        )


if __name__ == "__main__":
    unittest.main()

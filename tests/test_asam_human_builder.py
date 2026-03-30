import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asam_human_builder.builder import (  # noqa: E402
    build_armature_spec,
    build_armature_spec_from_asset_dir,
    choose_generated_collection_action,
    resolve_default_asset_dir,
)
from phase3_classifier.classifier import CORE_TARGETS, TARGET_PARENTS, write_asset_report  # noqa: E402


FIXTURE_ROOT = REPO_ROOT / "src" / "armature_inspector" / "output"


def _copy_asset_folder(asset_name: str) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix="asam_builder_fixture_"))
    destination = temp_root / asset_name
    shutil.copytree(FIXTURE_ROOT / asset_name, destination)
    return destination


def _placement_metadata():
    return {
        "bounds_source": "meshes",
        "driven_meshes": ["BodyMesh"],
        "bbox_min": [-0.4, -0.3, 0.0],
        "bbox_max": [0.4, 0.3, 2.0],
        "bbox_height": 2.0,
        "bbox_ground_center": [0.0, 0.0, 0.0],
        "forward_axis": {"index": 0, "name": "x", "sign": 1},
        "side_axis": {"index": 1, "name": "y", "sign": 1},
        "up_axis": {"index": 2, "name": "z", "sign": 1},
    }


def _base_classifier_report(asset_name: str = "SyntheticAsset", armature_name: str = "Rig") -> dict:
    semantic_mapping = {
        target: {
            "source_bone": None,
            "confidence": 0.0,
            "action": "create_in_builder",
            "evidence": {
                "name": 0.0,
                "hierarchy": 0.0,
                "geometry": 0.0,
                "source_origin": None,
                "role": None,
            },
            "notes": [],
        }
        for target in CORE_TARGETS
    }
    return {
        "recommended_primary_armature": armature_name,
        "semantic_mapping": semantic_mapping,
    }


def _base_build_plan(asset_name: str = "SyntheticAsset", armature_name: str = "Rig") -> dict:
    return {
        "asset_name": asset_name,
        "recommended_primary_armature": armature_name,
        "root_resolution": {
            "mode": "create_new_root",
            "target_bone": "Root",
            "source_bone": None,
            "rename_source_to_target": False,
            "failure_codes": [],
            "target_head": [0.0, 0.0, 0.0],
            "target_tail": [0.0, 0.0, 1.0],
            "up_axis": {"index": 2, "name": "z", "sign": 1},
            "use_connect": False,
        },
        "placement_metadata": _placement_metadata(),
        "proposed_asam_hierarchy": {
            "object_nodes": [
                {"name": "Grp_Root", "parent": None},
                {"name": "Armature_{0}".format(asset_name), "parent": "Grp_Root"},
            ],
            "bone_parents": {
                target: (
                    TARGET_PARENTS[target].format(asset_name=asset_name)
                    if target == "Root"
                    else TARGET_PARENTS[target]
                )
                for target in CORE_TARGETS
            },
        },
        "extras_preserved": [],
    }


def _bone(name: str, parent: str, head, tail) -> dict:
    return {
        "name": name,
        "parent": parent,
        "head": [float(value) for value in head],
        "tail": [float(value) for value in tail],
        "length": sum((tail[index] - head[index]) ** 2 for index in range(3)) ** 0.5,
    }


def _spec_bone(build_spec: dict, name: str) -> dict:
    return next(bone for bone in build_spec["bones"] if bone["name"] == name)


class AsamHumanBuilderTests(unittest.TestCase):
    def test_openmaterial_fixture_reuses_existing_root_geometry(self):
        asset_dir = _copy_asset_folder("openmatexamplehuman")
        write_asset_report(asset_dir)

        _, build_plan, build_spec = build_armature_spec_from_asset_dir(asset_dir)

        self.assertEqual(build_plan["root_resolution"]["mode"], "reuse_existing_root")
        self.assertEqual(build_spec["source_armature_name"], "Armature")
        self.assertEqual(len(build_spec["bones"]), len(CORE_TARGETS))

        root_bone = _spec_bone(build_spec, "Root")
        self.assertEqual(root_bone["geometry_source"], "source_root")
        self.assertEqual(root_bone["source_bone"], "Root")
        self.assertFalse(root_bone["use_connect"])

    def test_lowpoly_fixture_creates_new_root_and_preserves_extras(self):
        asset_dir = _copy_asset_folder("LowPolyCharacter4")
        write_asset_report(asset_dir)

        _, build_plan, build_spec = build_armature_spec_from_asset_dir(asset_dir)

        self.assertEqual(build_plan["root_resolution"]["mode"], "create_new_root")
        self.assertEqual(build_spec["source_armature_name"], "rig")
        self.assertGreater(len(build_spec["extras_preserved"]), 0)

        root_bone = _spec_bone(build_spec, "Root")
        self.assertEqual(root_bone["geometry_source"], "root_resolution")
        self.assertEqual(root_bone["source_bone"], "root")
        self.assertEqual(root_bone["head"], build_plan["root_resolution"]["target_head"])
        self.assertEqual(root_bone["tail"], build_plan["root_resolution"]["target_tail"])

    def test_missing_right_limb_prefers_mirror_geometry(self):
        classifier_report = _base_classifier_report()
        build_plan = _base_build_plan()
        classifier_report["semantic_mapping"]["Lower_Arm_Left"]["action"] = "direct_map"
        classifier_report["semantic_mapping"]["Lower_Arm_Left"]["source_bone"] = "lower_arm.L"
        classifier_report["semantic_mapping"]["Lower_Arm_Right"]["action"] = "create_in_builder"

        source_bones = {
            "lower_arm.L": _bone("lower_arm.L", "Upper_Arm_Left", (0.2, 0.25, 1.4), (0.25, 0.45, 1.0)),
        }

        build_spec = build_armature_spec(classifier_report, build_plan, source_bones)
        right_bone = _spec_bone(build_spec, "Lower_Arm_Right")

        self.assertEqual(right_bone["geometry_source"], "mirrored_opposite")
        self.assertAlmostEqual(right_bone["head"][0], 0.2)
        self.assertAlmostEqual(right_bone["head"][1], -0.25)
        self.assertAlmostEqual(right_bone["tail"][1], -0.45)

    def test_missing_spine_segment_prefers_interpolation(self):
        classifier_report = _base_classifier_report()
        build_plan = _base_build_plan()
        classifier_report["semantic_mapping"]["Hip"]["action"] = "direct_map"
        classifier_report["semantic_mapping"]["Hip"]["source_bone"] = "Hip"
        classifier_report["semantic_mapping"]["Upper_Spine"]["action"] = "direct_map"
        classifier_report["semantic_mapping"]["Upper_Spine"]["source_bone"] = "Upper_Spine"

        source_bones = {
            "Hip": _bone("Hip", "Root", (0.0, 0.0, 1.0), (0.0, 0.0, 1.2)),
            "Upper_Spine": _bone("Upper_Spine", "Lower_Spine", (0.0, 0.0, 1.45), (0.0, 0.0, 1.7)),
        }

        build_spec = build_armature_spec(classifier_report, build_plan, source_bones)
        lower_spine = _spec_bone(build_spec, "Lower_Spine")

        self.assertEqual(lower_spine["geometry_source"], "interpolated_chain")
        self.assertEqual(lower_spine["head"], [0.0, 0.0, 1.2])
        self.assertEqual(lower_spine["tail"], [0.0, 0.0, 1.45])

    def test_missing_head_prefers_parent_extrapolation(self):
        classifier_report = _base_classifier_report()
        build_plan = _base_build_plan()
        classifier_report["semantic_mapping"]["Neck"]["action"] = "direct_map"
        classifier_report["semantic_mapping"]["Neck"]["source_bone"] = "Neck"

        source_bones = {
            "Neck": _bone("Neck", "Upper_Spine", (0.0, 0.0, 1.55), (0.0, 0.0, 1.72)),
        }

        build_spec = build_armature_spec(classifier_report, build_plan, source_bones)
        head = _spec_bone(build_spec, "Head")

        self.assertEqual(head["geometry_source"], "extrapolated_parent")
        self.assertEqual(head["head"], [0.0, 0.0, 1.72])
        self.assertGreater(head["tail"][2], head["head"][2])

    def test_generated_collection_action_rebuilds_only_safe_generated_output(self):
        action = choose_generated_collection_action(
            [
                {
                    "name": "ASAM_TestAsset",
                    "generated": True,
                    "asset_name": "TestAsset",
                }
            ],
            "ASAM_TestAsset",
            "TestAsset",
        )

        self.assertEqual(action, "rebuild")

    def test_generated_collection_action_rejects_unmarked_conflict(self):
        with self.assertRaises(ValueError):
            choose_generated_collection_action(
                [
                    {
                        "name": "ASAM_TestAsset",
                        "generated": False,
                        "asset_name": None,
                    }
                ],
                "ASAM_TestAsset",
                "TestAsset",
            )

    def test_resolve_default_asset_dir_uses_blend_name(self):
        resolved = resolve_default_asset_dir(
            "C:/assets/openmatexamplehuman.blend",
            REPO_ROOT / "src" / "asam_human_builder",
        )

        self.assertEqual(
            resolved,
            (REPO_ROOT / "src" / "armature_inspector" / "output" / "openmatexamplehuman").resolve(),
        )


if __name__ == "__main__":
    unittest.main()

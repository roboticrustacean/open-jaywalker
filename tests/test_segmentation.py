import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase3_classifier.segmentation import (  # noqa: E402
    derive_character_prefix,
    strip_character_prefix,
)

from phase3_classifier.segmentation import segment_recommended  # noqa: E402

from phase3_classifier.segmentation import (  # noqa: E402
    CharacterGroup,
    MIN_CHARACTER_BONES,
    detect_characters,
    build_character_view,
    assign_meshes_to_characters,  # noqa: E402
)

from phase3_classifier.classifier import ArmatureInput  # noqa: E402


class PrefixDerivationTests(unittest.TestCase):
    def test_derives_alpha_digit_prefix(self):
        self.assertEqual(derive_character_prefix("Female000Pelvis_093"), "Female000")
        self.assertEqual(derive_character_prefix("Male060Spine0_01385"), "Male060")

    def test_strips_colon_namespace_before_matching(self):
        self.assertEqual(derive_character_prefix("Scene:Female000Pelvis_093"), "Female000")

    def test_returns_none_without_alpha_digit_prefix(self):
        self.assertIsNone(derive_character_prefix("_rootJoint"))
        self.assertIsNone(derive_character_prefix("Pelvis"))

    def test_strip_prefix_exposes_role_token(self):
        self.assertEqual(strip_character_prefix("Female000Pelvis_093", "Female000"), "Pelvis")
        self.assertEqual(strip_character_prefix("Female000Spine0_01385", "Female000"), "Spine0")
        self.assertEqual(strip_character_prefix("Female000LCalf_010", "Female000"), "LCalf")
        self.assertEqual(strip_character_prefix("Female000COM_001", "Female000"), "COM")

    def test_strip_prefix_of_prefix_only_name_stays_nonempty(self):
        self.assertEqual(strip_character_prefix("Female000", "Female000"), "Female000")
        self.assertEqual(strip_character_prefix("Female000_5", "Female000"), "Female000_5")


def _crowd_bones():
    """Two 6-bone characters under a shared _rootJoint, plus one stray bone."""
    bones = [{"name": "_rootJoint", "parent": None}]
    for prefix in ("Hero000", "Hero001"):
        bones += [
            {"name": f"{prefix}COM_0", "parent": "_rootJoint"},
            {"name": f"{prefix}Pelvis_1", "parent": f"{prefix}COM_0"},
            {"name": f"{prefix}Spine0_2", "parent": f"{prefix}Pelvis_1"},
            {"name": f"{prefix}Spine1_3", "parent": f"{prefix}Spine0_2"},
            {"name": f"{prefix}Head_4", "parent": f"{prefix}Spine1_3"},
            {"name": f"{prefix}LThigh_5", "parent": f"{prefix}Pelvis_1"},
        ]
    return bones


def _crowd_primary_data():
    bones = []
    for prefix in ("Hero000", "Hero001"):
        bones += [
            {"name": f"{prefix}COM_0", "parent": "_rootJoint", "head": [0, 0, 0], "tail": [0, 0, 1], "length": 1.0},
            {"name": f"{prefix}Pelvis_1", "parent": f"{prefix}COM_0", "head": [0, 0, 1], "tail": [0, 0, 1.1], "length": 0.1},
            {"name": f"{prefix}Spine0_2", "parent": f"{prefix}Pelvis_1", "head": [0, 0, 1.1], "tail": [0, 0, 1.3], "length": 0.2},
            {"name": f"{prefix}Spine1_3", "parent": f"{prefix}Spine0_2", "head": [0, 0, 1.3], "tail": [0, 0, 1.5], "length": 0.2},
            {"name": f"{prefix}Head_4", "parent": f"{prefix}Spine1_3", "head": [0, 0, 1.5], "tail": [0, 0, 1.7], "length": 0.2},
            {"name": f"{prefix}LThigh_5", "parent": f"{prefix}Pelvis_1", "head": [0, 0.1, 1], "tail": [0, 0.1, 0.5], "length": 0.5},
        ]
    bones.insert(0, {"name": "_rootJoint", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 0.1], "length": 0.1})
    return {
        "armature_name": "Object_4",
        "source_file": "crowd.blend",
        "filter": None,
        "bone_count": len(bones),
        "bones": bones,
        "chains": {
            "spine": [[f"Hero000Spine0_2", f"Hero000Spine1_3"], [f"Hero001Spine0_2", f"Hero001Spine1_3"]],
            "leg": {"left": [], "right": [], "unsided": []},
            "arm": {"left": [], "right": [], "unsided": []},
        },
    }


class DetectCharactersTests(unittest.TestCase):
    def test_detects_two_connected_characters(self):
        groups, shared = detect_characters(_crowd_bones())
        ids = sorted(group.character_id for group in groups)
        self.assertEqual(ids, ["Hero000", "Hero001"])
        self.assertEqual(shared, ["_rootJoint"])

    def test_group_carries_original_bone_names(self):
        groups, _ = detect_characters(_crowd_bones())
        hero0 = next(group for group in groups if group.character_id == "Hero000")
        self.assertIn("Hero000Pelvis_1", hero0.bone_names)
        self.assertEqual(len(hero0.bone_names), 6)

    def test_below_floor_group_is_not_a_character(self):
        bones = [{"name": "_rootJoint", "parent": None}]
        bones += [{"name": f"Tiny000Bone_{i}", "parent": "_rootJoint"} for i in range(MIN_CHARACTER_BONES - 1)]
        groups, _ = detect_characters(bones)
        self.assertEqual(groups, [])

    def test_disconnected_group_flagged_but_kept(self):
        # Hero000 bones split into two roots under shared -> disconnected.
        bones = [{"name": "_rootJoint", "parent": None}]
        bones += [
            {"name": "Hero000A_0", "parent": "_rootJoint"},
            {"name": "Hero000B_1", "parent": "Hero000A_0"},
            {"name": "Hero000C_2", "parent": "Hero000B_1"},
            {"name": "Hero000D_3", "parent": "_rootJoint"},  # second root
            {"name": "Hero000E_4", "parent": "Hero000D_3"},
            {"name": "Hero000F_5", "parent": "Hero000E_4"},
        ]
        groups, _ = detect_characters(bones)
        self.assertEqual(len(groups), 1)
        self.assertFalse(groups[0].connected)


class BuildCharacterViewTests(unittest.TestCase):
    def setUp(self):
        self.primary = _crowd_primary_data()
        self.groups, _ = detect_characters(self.primary["bones"])
        self.hero0 = next(group for group in self.groups if group.character_id == "Hero000")

    def test_view_is_armature_input_named_by_character(self):
        view = build_character_view(self.primary, self.hero0, mesh_binding=None)
        self.assertIsInstance(view, ArmatureInput)
        self.assertEqual(view.armature_name, "Hero000")

    def test_bone_names_are_stripped_and_scoped(self):
        view = build_character_view(self.primary, self.hero0, mesh_binding=None)
        names = {bone["name"] for bone in view.primary_data["bones"]}
        self.assertEqual(names, {"COM", "Pelvis", "Spine0", "Spine1", "Head", "LThigh"})

    def test_subroot_parent_repointed_to_none(self):
        view = build_character_view(self.primary, self.hero0, mesh_binding=None)
        by_name = {bone["name"]: bone for bone in view.primary_data["bones"]}
        self.assertIsNone(by_name["COM"]["parent"])
        self.assertEqual(by_name["Pelvis"]["parent"], "COM")

    def test_chains_filtered_and_stripped(self):
        view = build_character_view(self.primary, self.hero0, mesh_binding=None)
        self.assertEqual(view.primary_data["chains"]["spine"], [["Spine0", "Spine1"]])

    def test_placement_metadata_omitted_for_per_character_derivation(self):
        view = build_character_view(self.primary, self.hero0, mesh_binding=None)
        self.assertNotIn("placement_metadata", view.primary_data)


def _crowd_mesh_binding():
    return {
        "armature_object_name": "Object_4",
        "meshes": [
            {
                "mesh_name": "Hero000_Body",
                "armature_link": "modifier",
                "modifiers": [],
                "vertex_groups": ["Hero000Pelvis_1", "Hero000LThigh_5"],
                "vertex_group_stats": {
                    "non_empty_group_count": 2,
                    "per_group": [
                        {"name": "Hero000Pelvis_1", "weighted_vertex_count": 5},
                        {"name": "Hero000LThigh_5", "weighted_vertex_count": 3},
                    ],
                },
                "material_slots": [],
                "warnings": [],
            },
            {
                "mesh_name": "Prop",
                "armature_link": "modifier",
                "modifiers": [],
                "vertex_groups": ["_rootJoint"],
                "vertex_group_stats": {
                    "non_empty_group_count": 1,
                    "per_group": [{"name": "_rootJoint", "weighted_vertex_count": 4}],
                },
                "material_slots": [],
                "warnings": [],
            },
        ],
    }


class AssignMeshesTests(unittest.TestCase):
    def setUp(self):
        primary = _crowd_primary_data()
        self.groups, _ = detect_characters(primary["bones"])

    def test_mesh_assigned_to_majority_prefix_character(self):
        per_char, unassigned = assign_meshes_to_characters(_crowd_mesh_binding(), self.groups, "Object_4")
        self.assertIn("Hero000", per_char)
        names = [mesh["mesh_name"] for mesh in per_char["Hero000"]["meshes"]]
        self.assertEqual(names, ["Hero000_Body"])

    def test_vertex_group_names_stripped(self):
        per_char, _ = assign_meshes_to_characters(_crowd_mesh_binding(), self.groups, "Object_4")
        mesh = per_char["Hero000"]["meshes"][0]
        self.assertEqual(mesh["vertex_groups"], ["Pelvis", "LThigh"])
        self.assertEqual(
            [entry["name"] for entry in mesh["vertex_group_stats"]["per_group"]],
            ["Pelvis", "LThigh"],
        )

    def test_unassignable_mesh_recorded(self):
        _, unassigned = assign_meshes_to_characters(_crowd_mesh_binding(), self.groups, "Object_4")
        self.assertEqual(unassigned, ["Prop"])

    def test_character_with_no_meshes_gets_empty_binding(self):
        per_char, _ = assign_meshes_to_characters(_crowd_mesh_binding(), self.groups, "Object_4")
        self.assertEqual(per_char["Hero001"]["meshes"], [])

    def test_assign_meshes_uses_source_armature_name(self):
        per_char, _unassigned = assign_meshes_to_characters(_crowd_mesh_binding(), self.groups, "Object_4")
        self.assertEqual(per_char["Hero000"]["armature_object_name"], "Object_4")
        self.assertEqual(per_char["Hero001"]["armature_object_name"], "Object_4")


class SegmentRecommendedTests(unittest.TestCase):
    def _recommended_input(self):
        primary = _crowd_primary_data()
        primary["mesh_binding"] = _crowd_mesh_binding()
        return ArmatureInput(
            armature_name="Object_4",
            all_path=Path("Object_4_all.json"),
            filtered_path=None,
            primary_path=Path("Object_4_all.json"),
            support_path=None,
            primary_data=primary,
            support_data=None,
        )

    def test_returns_none_for_single_character(self):
        primary = _crowd_primary_data()
        primary["bones"] = [b for b in primary["bones"] if not b["name"].startswith("Hero001")]
        single = ArmatureInput(
            armature_name="Object_4", all_path=None, filtered_path=None,
            primary_path=Path("Object_4_all.json"), support_path=None,
            primary_data=primary, support_data=None,
        )
        self.assertIsNone(segment_recommended("crowd", single))

    def test_emits_one_entry_per_character(self):
        result = segment_recommended("crowd", self._recommended_input())
        self.assertIsNotNone(result)
        self.assertEqual(result.decomposition["character_count"], 2)
        self.assertEqual(result.decomposition["character_ids"], ["Hero000", "Hero001"])
        report_ids = [c["character_id"] for c in result.report_characters]
        plan_ids = [c["character_id"] for c in result.plan_characters]
        self.assertEqual(report_ids, ["Hero000", "Hero001"])
        self.assertEqual(plan_ids, ["Hero000", "Hero001"])

    def test_per_character_mapping_uses_own_bones(self):
        result = segment_recommended("crowd", self._recommended_input())
        hero0 = next(c for c in result.report_characters if c["character_id"] == "Hero000")
        hip = hero0["semantic_mapping"].get("Hip", {})
        self.assertEqual(hip.get("source_bone"), "Pelvis")

    def test_decomposition_records_shared_and_unassigned(self):
        result = segment_recommended("crowd", self._recommended_input())
        self.assertEqual(result.decomposition["shared_bones"], ["_rootJoint"])
        self.assertEqual(result.decomposition["unassigned_meshes"], ["Prop"])


class BodyAnchorFromMeshesTests(unittest.TestCase):
    def test_character_body_anchor_from_owned_mesh_bbox(self):
        from phase3_classifier import segmentation
        meshes = [
            {"mesh_name": "Body", "bbox": {"min": [31.0, -11.0, 0.0], "max": [33.0, -9.0, 1.8]}},
        ]
        anchor = segmentation.body_anchor_from_meshes(meshes, up_axis_index=2, up_sign=1)
        self.assertAlmostEqual(anchor[0], 32.0)   # planar center
        self.assertAlmostEqual(anchor[1], -10.0)  # planar center
        self.assertAlmostEqual(anchor[2], 0.0)    # ground = min on up axis (up_sign>=0)

    def test_body_anchor_uses_max_on_up_axis_when_up_sign_negative(self):
        from phase3_classifier import segmentation
        meshes = [{"mesh_name": "B", "bbox": {"min": [0.0, 0.0, 0.0], "max": [2.0, 2.0, 2.0]}}]
        anchor = segmentation.body_anchor_from_meshes(meshes, up_axis_index=2, up_sign=-1)
        self.assertAlmostEqual(anchor[2], 2.0)    # ground = max when up_sign<0

    def test_body_anchor_unions_multiple_meshes(self):
        from phase3_classifier import segmentation
        meshes = [
            {"mesh_name": "A", "bbox": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}},
            {"mesh_name": "B", "bbox": {"min": [3.0, 3.0, 0.0], "max": [4.0, 4.0, 2.0]}},
        ]
        anchor = segmentation.body_anchor_from_meshes(meshes, up_axis_index=2, up_sign=1)
        self.assertAlmostEqual(anchor[0], 2.0)    # (min0=0, max4=4) -> center 2.0
        self.assertAlmostEqual(anchor[1], 2.0)

    def test_body_anchor_none_when_no_bboxes(self):
        from phase3_classifier import segmentation
        self.assertIsNone(segmentation.body_anchor_from_meshes([], up_axis_index=2, up_sign=1))
        self.assertIsNone(segmentation.body_anchor_from_meshes([{"mesh_name": "x"}], up_axis_index=2, up_sign=1))


class CharacterPlacementTests(unittest.TestCase):
    def _crowd_primary_data_distributed(self):
        """Two characters with bones at different XY positions (Hero001 offset by 3 m on Y)."""
        bones = []
        offsets = {"Hero000": [0.0, 0.0], "Hero001": [0.0, 3.0]}
        for prefix, (ox, oy) in offsets.items():
            bones += [
                {"name": f"{prefix}COM_0", "parent": "_rootJoint",
                 "head": [ox, oy, 0.0], "tail": [ox, oy, 0.1], "length": 0.1},
                {"name": f"{prefix}Pelvis_1", "parent": f"{prefix}COM_0",
                 "head": [ox, oy, 1.0], "tail": [ox, oy, 1.1], "length": 0.1},
                {"name": f"{prefix}Spine0_2", "parent": f"{prefix}Pelvis_1",
                 "head": [ox, oy, 1.1], "tail": [ox, oy, 1.3], "length": 0.2},
                {"name": f"{prefix}Spine1_3", "parent": f"{prefix}Spine0_2",
                 "head": [ox, oy, 1.3], "tail": [ox, oy, 1.5], "length": 0.2},
                {"name": f"{prefix}Head_4", "parent": f"{prefix}Spine1_3",
                 "head": [ox, oy, 1.5], "tail": [ox, oy, 1.7], "length": 0.2},
                {"name": f"{prefix}LThigh_5", "parent": f"{prefix}Pelvis_1",
                 "head": [ox + 0.1, oy, 1.0], "tail": [ox + 0.1, oy, 0.5], "length": 0.5},
            ]
        bones.insert(0, {"name": "_rootJoint", "parent": None,
                         "head": [0, 0, 0], "tail": [0, 0, 0.1], "length": 0.1})
        return {
            "armature_name": "Object_4",
            "source_file": "crowd.blend",
            "filter": None,
            "bone_count": len(bones),
            "bones": bones,
            "chains": {
                "spine": [
                    [f"Hero000Spine0_2", f"Hero000Spine1_3"],
                    [f"Hero001Spine0_2", f"Hero001Spine1_3"],
                ],
                "leg": {"left": [], "right": [], "unsided": []},
                "arm": {"left": [], "right": [], "unsided": []},
            },
        }

    def _crowd_mesh_binding_distributed(self):
        """Mesh binding where each character has a bbox at its respective position."""
        return {
            "armature_object_name": "Object_4",
            "meshes": [
                {
                    "mesh_name": "Hero000_Body",
                    "armature_link": "modifier",
                    "modifiers": [],
                    "vertex_groups": ["Hero000Pelvis_1", "Hero000LThigh_5"],
                    "vertex_group_stats": {
                        "non_empty_group_count": 2,
                        "per_group": [
                            {"name": "Hero000Pelvis_1", "weighted_vertex_count": 5},
                            {"name": "Hero000LThigh_5", "weighted_vertex_count": 3},
                        ],
                    },
                    "material_slots": [],
                    "warnings": [],
                    # Hero000 body is centred around (0, 0)
                    "bbox": {"min": [-0.5, -0.5, 0.0], "max": [0.5, 0.5, 1.8]},
                },
                {
                    "mesh_name": "Hero001_Body",
                    "armature_link": "modifier",
                    "modifiers": [],
                    "vertex_groups": ["Hero001Pelvis_1", "Hero001LThigh_5"],
                    "vertex_group_stats": {
                        "non_empty_group_count": 2,
                        "per_group": [
                            {"name": "Hero001Pelvis_1", "weighted_vertex_count": 5},
                            {"name": "Hero001LThigh_5", "weighted_vertex_count": 3},
                        ],
                    },
                    "material_slots": [],
                    "warnings": [],
                    # Hero001 body is centred around (0, 3)
                    "bbox": {"min": [-0.5, 2.5, 0.0], "max": [0.5, 3.5, 1.8]},
                },
            ],
        }

    def _segment_two_character_crowd_distributed(self):
        from phase3_classifier import segmentation
        primary = self._crowd_primary_data_distributed()
        primary["mesh_binding"] = self._crowd_mesh_binding_distributed()
        inp = ArmatureInput(
            armature_name="Object_4",
            all_path=Path("Object_4_all.json"),
            filtered_path=None,
            primary_path=Path("Object_4_all.json"),
            support_path=None,
            primary_data=primary,
            support_data=None,
        )
        return segmentation.segment_recommended("crowd", inp)

    def test_distributed_characters_preserve_source_positions(self):
        from phase3_classifier import segmentation
        result = self._segment_two_character_crowd_distributed()
        by_id = {c["character_id"]: c for c in result.plan_characters}
        rr0 = by_id["Hero000"]["root_resolutions"][0]
        rr1 = by_id["Hero001"]["root_resolutions"][0]
        self.assertEqual(rr0["placement_mode"], "source")
        self.assertEqual(rr1["placement_mode"], "source")
        self.assertNotEqual(rr0["grp_root_world_location"], rr1["grp_root_world_location"])


if __name__ == "__main__":
    unittest.main()

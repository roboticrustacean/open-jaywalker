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
        per_char, unassigned = assign_meshes_to_characters(_crowd_mesh_binding(), self.groups)
        self.assertIn("Hero000", per_char)
        names = [mesh["mesh_name"] for mesh in per_char["Hero000"]["meshes"]]
        self.assertEqual(names, ["Hero000_Body"])

    def test_vertex_group_names_stripped(self):
        per_char, _ = assign_meshes_to_characters(_crowd_mesh_binding(), self.groups)
        mesh = per_char["Hero000"]["meshes"][0]
        self.assertEqual(mesh["vertex_groups"], ["Pelvis", "LThigh"])
        self.assertEqual(
            [entry["name"] for entry in mesh["vertex_group_stats"]["per_group"]],
            ["Pelvis", "LThigh"],
        )

    def test_unassignable_mesh_recorded(self):
        _, unassigned = assign_meshes_to_characters(_crowd_mesh_binding(), self.groups)
        self.assertEqual(unassigned, ["Prop"])

    def test_character_with_no_meshes_gets_empty_binding(self):
        per_char, _ = assign_meshes_to_characters(_crowd_mesh_binding(), self.groups)
        self.assertEqual(per_char["Hero001"]["meshes"], [])


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


if __name__ == "__main__":
    unittest.main()

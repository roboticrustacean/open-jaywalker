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

from phase3_classifier.segmentation import (  # noqa: E402
    CharacterGroup,
    MIN_CHARACTER_BONES,
    detect_characters,
)


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


if __name__ == "__main__":
    unittest.main()

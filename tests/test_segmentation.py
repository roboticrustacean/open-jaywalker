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


if __name__ == "__main__":
    unittest.main()

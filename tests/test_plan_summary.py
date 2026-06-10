import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.plan_summary import summarize_plan  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "LowPolyCharacter4"


class SummarizePlanTests(unittest.TestCase):
    def _load(self):
        report = json.loads((FIXTURE / "classifier_report.json").read_text())
        plan = json.loads((FIXTURE / "build_plan.json").read_text())
        return report, plan

    def test_single_character_summary(self):
        report, plan = self._load()
        summary = summarize_plan(report, plan)
        self.assertEqual(summary["recommended_armature"], "rig")
        self.assertEqual(summary["total"], 28)
        self.assertEqual(summary["mapped"], 22)
        self.assertEqual(len(summary["missing_targets"]), 6)
        self.assertIn("Eye_Left", summary["missing_targets"])
        self.assertFalse(summary["is_crowd"])
        self.assertEqual(summary["character_count"], 0)
        self.assertEqual(summary["review_flags"], ["multiple_roots", "root_noncompliant"])
        self.assertEqual(summary["character_ids"], [])
        self.assertEqual(summary["missing_by_target"], [])

    def test_crowd_missing_by_target_aggregates(self):
        report = {
            "recommended_primary_armature": "_rootJoint",
            "missing_targets": [],
            "review_flags": [],
            "characters": [
                {"character_id": "c0", "missing_targets": ["Eye_Left", "Lower_Spine"]},
                {"character_id": "c1", "missing_targets": ["Eye_Left"]},
                {"character_id": "c2", "missing_targets": ["Eye_Left", "Lower_Spine"]},
            ],
        }
        plan = {"characters": [{"character_id": "c0"}, {"character_id": "c1"}, {"character_id": "c2"}]}
        summary = summarize_plan(report, plan)
        self.assertTrue(summary["is_crowd"])
        self.assertEqual(summary["character_count"], 3)
        self.assertEqual(
            summary["missing_by_target"],
            [{"target": "Eye_Left", "count": 3}, {"target": "Lower_Spine", "count": 2}],
        )

    def test_crowd_mapped_count_reflects_per_character_not_undecomposed_armature(self):
        # Regression (1000idles): the headline "Mapped: X/28" for a crowd must come
        # from the per-character decomposition, not from the top-level missing_targets
        # (which is the whole undecomposed crowd armature analyzed as one body and is
        # meaningless for a crowd). Each character here maps 22/28 (missing 6), so the
        # summary must report 22 -- not 28 - 12 = 16 from the top-level list.
        report = {
            "recommended_primary_armature": "Object_4",
            "missing_targets": [
                "Lower_Spine", "Upper_Spine", "Eye_Left", "Eye_Right",
                "Hand_Left", "Full_Thumb_Left", "Full_Fingers_Left", "Hand_Right",
                "Full_Thumb_Right", "Full_Fingers_Right", "Foot_Left", "Foot_Right",
            ],
            "review_flags": [],
            "characters": [
                {
                    "character_id": f"Female{i:03d}",
                    "missing_targets": [
                        "Eye_Left", "Eye_Right", "Full_Thumb_Left",
                        "Full_Fingers_Left", "Full_Thumb_Right", "Full_Fingers_Right",
                    ],
                }
                for i in range(3)
            ],
        }
        plan = {"characters": [{"character_id": f"Female{i:03d}"} for i in range(3)]}
        summary = summarize_plan(report, plan)
        self.assertTrue(summary["is_crowd"])
        self.assertEqual(summary["mapped"], 22)
        self.assertEqual(sorted(summary["missing_targets"]), [
            "Eye_Left", "Eye_Right", "Full_Fingers_Left",
            "Full_Fingers_Right", "Full_Thumb_Left", "Full_Thumb_Right",
        ])

    def test_crowd_summary_uses_character_count(self):
        report, _plan = self._load()
        crowd_plan = {"characters": [{"character_id": "c0"}, {"character_id": "c1"}]}
        summary = summarize_plan(report, crowd_plan)
        self.assertTrue(summary["is_crowd"])
        self.assertEqual(summary["character_count"], 2)
        self.assertEqual(summary["character_ids"], ["c0", "c1"])


if __name__ == "__main__":
    unittest.main()

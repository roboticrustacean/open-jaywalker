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

    def test_crowd_summary_uses_character_count(self):
        report, _plan = self._load()
        crowd_plan = {"characters": [{"character_id": "c0"}, {"character_id": "c1"}]}
        summary = summarize_plan(report, crowd_plan)
        self.assertTrue(summary["is_crowd"])
        self.assertEqual(summary["character_count"], 2)
        self.assertEqual(summary["character_ids"], ["c0", "c1"])


if __name__ == "__main__":
    unittest.main()

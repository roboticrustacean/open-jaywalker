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

    def test_crowd_summary_uses_character_count(self):
        report, _plan = self._load()
        crowd_plan = {"characters": [{"character_id": "c0"}, {"character_id": "c1"}]}
        summary = summarize_plan(report, crowd_plan)
        self.assertTrue(summary["is_crowd"])
        self.assertEqual(summary["character_count"], 2)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asam_human_builder import build_runner  # noqa: E402


class RunBuildTests(unittest.TestCase):
    def test_single_character_dispatch(self):
        resolved = {"crowd": False, "asset_name": "A", "specs": ["SPEC"]}
        with mock.patch.object(build_runner, "build_character_specs_from_asset_dir", return_value=resolved), \
             mock.patch.object(build_runner, "build_armature_in_blender", return_value={"exec": True}) as build_one, \
             mock.patch.object(build_runner, "write_builder_report", return_value=({"asset_name": "A"}, Path("/x/builder_report.json"))), \
             mock.patch.object(build_runner, "print_builder_summary"):
            report = build_runner.run_build(Path("/x"), bpy="BPY")

        build_one.assert_called_once_with("SPEC", "BPY")
        self.assertEqual(report, {"asset_name": "A"})

    def test_crowd_dispatch(self):
        resolved = {
            "crowd": True,
            "asset_name": "Crowd",
            "wrapper_collection_name": "Crowd_Humans",
            "decomposition": {"source_armature": "Root"},
            "character_specs": [("c0", "SPEC0")],
        }
        crowd_report = {"characters": [{"character_id": "c0"}], "failed_characters": []}
        with mock.patch.object(build_runner, "build_character_specs_from_asset_dir", return_value=resolved), \
             mock.patch.object(build_runner, "build_crowd_in_blender", return_value={"exec": True}) as build_crowd, \
             mock.patch.object(build_runner, "build_crowd_builder_report", return_value=crowd_report), \
             mock.patch.object(build_runner, "write_crowd_builder_report", return_value=(crowd_report, Path("/x/builder_report.json"))):
            report = build_runner.run_build(Path("/x"), bpy="BPY")

        build_crowd.assert_called_once_with("Crowd", "Crowd_Humans", [("c0", "SPEC0")], {"source_armature": "Root"}, "BPY")
        self.assertEqual(report, crowd_report)


if __name__ == "__main__":
    unittest.main()

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
        spec = {"generated_collection_name": "ASAM_A", "name": "SPEC"}
        resolved = {"crowd": False, "asset_name": "A", "specs": [spec]}
        with mock.patch.object(build_runner, "build_character_specs_from_asset_dir", return_value=resolved), \
             mock.patch.object(build_runner, "build_armature_in_blender", return_value={"exec": True}) as build_one, \
             mock.patch.object(build_runner, "write_builder_report", return_value=({"asset_name": "A"}, Path("/x/builder_report.json"))), \
             mock.patch.object(build_runner, "print_builder_summary"):
            report = build_runner.run_build(Path("/x"), bpy="BPY", packaging_mode="inplace_only")

        build_one.assert_called_once_with(spec, "BPY")
        # inplace_only performs no export but still records the packaging mode for a
        # consistent report shape across modes.
        self.assertEqual(
            report,
            {
                "asset_name": "A",
                "packaging_mode": "inplace_only",
                "exported_blend_path": None,
                "export_error": None,
            },
        )

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
            report = build_runner.run_build(Path("/x"), bpy="BPY", packaging_mode="inplace_only")

        build_crowd.assert_called_once_with("Crowd", "Crowd_Humans", [("c0", "SPEC0")], {"source_armature": "Root"}, "BPY")
        self.assertEqual(report, crowd_report)


class RunBuildPackagingTests(unittest.TestCase):
    def _patch_single(self):
        return [
            mock.patch.object(build_runner, "build_character_specs_from_asset_dir",
                              return_value={"crowd": False, "asset_name": "A",
                                            "specs": [{"generated_collection_name": "ASAM_A"}]}),
            mock.patch.object(build_runner, "build_armature_in_blender", return_value={"exec": True}),
            mock.patch.object(build_runner, "write_builder_report",
                              return_value=({"asset_name": "A"}, Path("/x/builder_report.json"))),
            mock.patch.object(build_runner, "print_builder_summary"),
        ]

    def test_inplace_export_calls_export(self):
        for p in self._patch_single():
            p.start(); self.addCleanup(p.stop)
        with mock.patch.object(build_runner, "export_generated_blend", return_value="/x/A_asam.blend") as export, \
             mock.patch.object(build_runner, "purge_previous_generated_artifacts") as purge:
            report = build_runner.run_build(Path("/x"), bpy="BPY", packaging_mode="inplace_export")
            export.assert_called_once()
            purge.assert_not_called()
            self.assertEqual(report["packaging_mode"], "inplace_export")
            self.assertEqual(report["exported_blend_path"], "/x/A_asam.blend")
            self.assertIsNone(report["export_error"])

    def test_inplace_only_skips_export(self):
        for p in self._patch_single():
            p.start(); self.addCleanup(p.stop)
        with mock.patch.object(build_runner, "export_generated_blend") as export:
            report = build_runner.run_build(Path("/x"), bpy="BPY", packaging_mode="inplace_only")
            export.assert_not_called()
            # packaging_mode is recorded even when no export happens (consistent report shape).
            self.assertEqual(report["packaging_mode"], "inplace_only")
            self.assertIsNone(report["exported_blend_path"])

    def test_separate_only_exports_then_purges(self):
        for p in self._patch_single():
            p.start(); self.addCleanup(p.stop)
        with mock.patch.object(build_runner, "export_generated_blend", return_value="/x/A_asam.blend"), \
             mock.patch.object(build_runner, "purge_previous_generated_artifacts") as purge:
            build_runner.run_build(Path("/x"), bpy="BPY", packaging_mode="separate_only")
            purge.assert_called_once()

    def test_export_failure_does_not_abort(self):
        for p in self._patch_single():
            p.start(); self.addCleanup(p.stop)
        with mock.patch.object(build_runner, "export_generated_blend", side_effect=RuntimeError("disk full")):
            report = build_runner.run_build(Path("/x"), bpy="BPY", packaging_mode="inplace_export")
            self.assertEqual(report.get("exported_blend_path"), None)
            self.assertIn("export_error", report)


class RunBuildWarningTests(unittest.TestCase):
    def test_single_character_warning_logged(self):
        import io
        spec = {"generated_collection_name": "ASAM_A", "name": "SPEC"}
        resolved = {"crowd": False, "asset_name": "A", "specs": [spec]}
        report_data = {
            "asset_name": "A",
            "targets_created_heuristically": ["Full_Fingers_Left", "Full_Fingers_Right"]
        }
        
        f = io.StringIO()
        with mock.patch("sys.stdout", new=f):
            with mock.patch.object(build_runner, "build_character_specs_from_asset_dir", return_value=resolved), \
                 mock.patch.object(build_runner, "build_armature_in_blender", return_value={"exec": True}), \
                 mock.patch.object(build_runner, "write_builder_report", return_value=(report_data, Path("/x/builder_report.json"))), \
                 mock.patch.object(build_runner, "print_builder_summary"):
                build_runner.run_build(Path("/x"), bpy="BPY", packaging_mode="inplace_only")
                
        output = f.getvalue()
        self.assertIn("WARNING: Synthesized inert bones added for compliance: Full_Fingers_Left, Full_Fingers_Right", output)

    def test_crowd_warning_logged(self):
        import io
        resolved = {
            "crowd": True,
            "asset_name": "Crowd",
            "wrapper_collection_name": "Crowd_Humans",
            "decomposition": {"source_armature": "Root"},
            "character_specs": [("c0", "SPEC0"), ("c1", "SPEC1")],
        }
        crowd_report = {
            "characters": [
                {
                    "character_id": "c0",
                    "targets_created_heuristically": ["Full_Fingers_Left"]
                },
                {
                    "character_id": "c1",
                    "targets_created_heuristically": ["Full_Toes_Right"]
                }
            ],
            "failed_characters": []
        }
        
        f = io.StringIO()
        with mock.patch("sys.stdout", new=f):
            with mock.patch.object(build_runner, "build_character_specs_from_asset_dir", return_value=resolved), \
                 mock.patch.object(build_runner, "build_crowd_in_blender", return_value={"exec": True}), \
                 mock.patch.object(build_runner, "build_crowd_builder_report", return_value=crowd_report), \
                 mock.patch.object(build_runner, "write_crowd_builder_report", return_value=(crowd_report, Path("/x/builder_report.json"))):
                build_runner.run_build(Path("/x"), bpy="BPY", packaging_mode="inplace_only")
                
        output = f.getvalue()
        self.assertIn("WARNING: Character 'c0' has synthesized inert bones added for compliance: Full_Fingers_Left", output)
        self.assertIn("WARNING: Character 'c1' has synthesized inert bones added for compliance: Full_Toes_Right", output)


if __name__ == "__main__":
    unittest.main()

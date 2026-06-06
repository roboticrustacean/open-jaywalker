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
            report = build_runner.run_build(Path("/x"), bpy="BPY", export_blend=False, export_gltf=False)

        build_one.assert_called_once_with(spec, "BPY")
        # No export requested: report records the flags with None paths.
        self.assertEqual(report["export_blend"], False)
        self.assertEqual(report["export_gltf"], False)
        self.assertIsNone(report["exported_blend_path"])
        self.assertIsNone(report["exported_gltf_path"])

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
            report = build_runner.run_build(Path("/x"), bpy="BPY", export_blend=False, export_gltf=False)

        build_crowd.assert_called_once_with("Crowd", "Crowd_Humans", [("c0", "SPEC0")], {"source_armature": "Root"}, "BPY")
        self.assertEqual(report, crowd_report)


class RunBuildExportTests(unittest.TestCase):
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

    def test_export_blend_calls_export(self):
        for p in self._patch_single():
            p.start(); self.addCleanup(p.stop)
        with mock.patch.object(build_runner, "export_generated_blend", return_value="/x/A_asam.blend") as export:
            report = build_runner.run_build(Path("/x"), bpy="BPY", export_blend=True, export_gltf=False)
            export.assert_called_once()
            self.assertTrue(report["export_blend"])
            self.assertEqual(report["exported_blend_path"], "/x/A_asam.blend")
            self.assertIsNone(report["export_blend_error"])

    def test_export_gltf_calls_export(self):
        for p in self._patch_single():
            p.start(); self.addCleanup(p.stop)
        with mock.patch.object(build_runner, "export_generated_gltf", return_value="/x/A_asam.glb") as export:
            report = build_runner.run_build(Path("/x"), bpy="BPY", export_blend=False, export_gltf=True)
            export.assert_called_once()
            self.assertTrue(report["export_gltf"])
            self.assertEqual(report["exported_gltf_path"], "/x/A_asam.glb")
            self.assertIsNone(report["export_gltf_error"])

    def test_no_export_skips_both(self):
        for p in self._patch_single():
            p.start(); self.addCleanup(p.stop)
        with mock.patch.object(build_runner, "export_generated_blend") as blend_export, \
             mock.patch.object(build_runner, "export_generated_gltf") as gltf_export:
            report = build_runner.run_build(Path("/x"), bpy="BPY", export_blend=False, export_gltf=False)
            blend_export.assert_not_called()
            gltf_export.assert_not_called()
            self.assertFalse(report["export_blend"])
            self.assertFalse(report["export_gltf"])
            self.assertIsNone(report["exported_blend_path"])
            self.assertIsNone(report["exported_gltf_path"])

    def test_export_blend_failure_does_not_abort(self):
        for p in self._patch_single():
            p.start(); self.addCleanup(p.stop)
        with mock.patch.object(build_runner, "export_generated_blend", side_effect=RuntimeError("disk full")):
            report = build_runner.run_build(Path("/x"), bpy="BPY", export_blend=True, export_gltf=False)
            self.assertIsNone(report["exported_blend_path"])
            self.assertEqual(report["export_blend_error"], "disk full")

    def test_export_gltf_failure_does_not_abort(self):
        for p in self._patch_single():
            p.start(); self.addCleanup(p.stop)
        with mock.patch.object(build_runner, "export_generated_gltf", side_effect=RuntimeError("glTF error")):
            report = build_runner.run_build(Path("/x"), bpy="BPY", export_blend=False, export_gltf=True)
            self.assertIsNone(report["exported_gltf_path"])
            self.assertEqual(report["export_gltf_error"], "glTF error")

    def test_default_exports_gltf_not_blend(self):
        """Verify the new defaults: export_gltf=True, export_blend=False."""
        for p in self._patch_single():
            p.start(); self.addCleanup(p.stop)
        with mock.patch.object(build_runner, "export_generated_gltf", return_value="/x/A_asam.glb") as gltf_export, \
             mock.patch.object(build_runner, "export_generated_blend") as blend_export:
            report = build_runner.run_build(Path("/x"), bpy="BPY")
            gltf_export.assert_called_once()
            blend_export.assert_not_called()
            self.assertTrue(report["export_gltf"])
            self.assertFalse(report["export_blend"])


class RunBuildPerCharacterExportTests(unittest.TestCase):
    def _patch_crowd(self):
        return [
            mock.patch.object(build_runner, "build_character_specs_from_asset_dir",
                              return_value={"crowd": True, "asset_name": "Crowd",
                                            "wrapper_collection_name": "ASAM_Crowd",
                                            "decomposition": {"source_armature": "Root"},
                                            "character_specs": [("c0", "S0"), ("c1", "S1")]}),
            mock.patch.object(build_runner, "build_crowd_in_blender", return_value={"exec": True}),
            mock.patch.object(build_runner, "build_crowd_builder_report",
                              return_value={"characters": [{"character_id": "c0"}, {"character_id": "c1"}],
                                            "failed_characters": []}),
            mock.patch.object(build_runner, "write_crowd_builder_report",
                              return_value=({"characters": [{"character_id": "c0"}, {"character_id": "c1"}],
                                             "failed_characters": []},
                                            Path("/x/builder_report.json"))),
        ]

    def test_per_character_blend_calls_crowd_function(self):
        for p in self._patch_crowd():
            p.start(); self.addCleanup(p.stop)
        with mock.patch.object(build_runner, "export_crowd_characters_blend",
                               return_value=["/x/blend/c0.blend", "/x/blend/c1.blend"]) as crowd_blend, \
             mock.patch.object(build_runner, "export_generated_blend") as single_blend:
            report = build_runner.run_build(
                Path("/x"), bpy="BPY", export_blend=True, export_gltf=False, per_character=True
            )
            crowd_blend.assert_called_once()
            single_blend.assert_not_called()
            self.assertEqual(report["exported_blend_paths"], ["/x/blend/c0.blend", "/x/blend/c1.blend"])
            self.assertIsNone(report["exported_blend_path"])

    def test_per_character_gltf_calls_crowd_function(self):
        for p in self._patch_crowd():
            p.start(); self.addCleanup(p.stop)
        with mock.patch.object(build_runner, "export_crowd_characters_gltf",
                               return_value=["/x/glb/c0.glb", "/x/glb/c1.glb"]) as crowd_gltf, \
             mock.patch.object(build_runner, "export_generated_gltf") as single_gltf:
            report = build_runner.run_build(
                Path("/x"), bpy="BPY", export_blend=False, export_gltf=True, per_character=True
            )
            crowd_gltf.assert_called_once()
            single_gltf.assert_not_called()
            self.assertEqual(report["exported_gltf_paths"], ["/x/glb/c0.glb", "/x/glb/c1.glb"])
            self.assertIsNone(report["exported_gltf_path"])

    def test_per_character_on_single_asset_uses_monolithic(self):
        """per_character=True on a non-crowd asset falls back to monolithic export."""
        patches = [
            mock.patch.object(build_runner, "build_character_specs_from_asset_dir",
                              return_value={"crowd": False, "asset_name": "A",
                                            "specs": [{"generated_collection_name": "ASAM_A"}]}),
            mock.patch.object(build_runner, "build_armature_in_blender", return_value={"exec": True}),
            mock.patch.object(build_runner, "write_builder_report",
                              return_value=({"asset_name": "A"}, Path("/x/builder_report.json"))),
            mock.patch.object(build_runner, "print_builder_summary"),
        ]
        for p in patches:
            p.start(); self.addCleanup(p.stop)
        with mock.patch.object(build_runner, "export_generated_gltf", return_value="/x/A_asam.glb") as single_gltf, \
             mock.patch.object(build_runner, "export_crowd_characters_gltf") as crowd_gltf:
            report = build_runner.run_build(
                Path("/x"), bpy="BPY", export_blend=False, export_gltf=True, per_character=True
            )
            single_gltf.assert_called_once()
            crowd_gltf.assert_not_called()


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
                build_runner.run_build(Path("/x"), bpy="BPY", export_blend=False, export_gltf=False)
                
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
                build_runner.run_build(Path("/x"), bpy="BPY", export_blend=False, export_gltf=False)
                
        output = f.getvalue()
        self.assertIn("WARNING: Character 'c0' has synthesized inert bones added for compliance: Full_Fingers_Left", output)
        self.assertIn("WARNING: Character 'c1' has synthesized inert bones added for compliance: Full_Toes_Right", output)


if __name__ == "__main__":
    unittest.main()

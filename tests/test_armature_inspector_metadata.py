import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
INSPECTOR_ROOT = REPO_ROOT / "src" / "armature_inspector"
if str(INSPECTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(INSPECTOR_ROOT))


class ArmatureInspectorMetadataTests(unittest.TestCase):
    def test_inspect_scene_returns_export_metadata(self):
        fake_bpy = types.SimpleNamespace(
            data=types.SimpleNamespace(filepath="C:/assets/Example.blend", objects=[], armatures=[]),
        )

        with mock.patch.dict(sys.modules, {"bpy": fake_bpy}):
            if "inspector" in sys.modules:
                del sys.modules["inspector"]
            inspector = importlib.import_module("inspector")

        fake_armature = types.SimpleNamespace(name="Rig")

        with mock.patch.object(inspector, "get_armature_objects", return_value=[fake_armature]), \
             mock.patch.object(inspector, "print_scene_summary"), \
             mock.patch.object(inspector, "print_armature_hierarchy"), \
             mock.patch.object(inspector, "print_detected_chains"), \
             mock.patch.object(inspector, "_has_prefix_bones", return_value=True), \
             mock.patch.object(inspector, "_get_output_dir", return_value="C:/tmp/output/Example"), \
             mock.patch.object(
                 inspector,
                 "export_armature_json",
                 return_value=("C:/tmp/output/Example/Rig_all.json", "C:/tmp/output/Example/Rig_filtered.json"),
             ):
            result = inspector.inspect_scene()

        self.assertEqual(result["source_file"], "C:/assets/Example.blend")
        self.assertEqual(result["output_dir"], "C:/tmp/output/Example")
        self.assertEqual(result["armature_count"], 1)
        self.assertEqual(result["armature_names"], ["Rig"])
        self.assertEqual(
            result["exported_files"],
            ["C:/tmp/output/Example/Rig_all.json", "C:/tmp/output/Example/Rig_filtered.json"],
        )
        self.assertFalse(result["diagnostics_ran"])


if __name__ == "__main__":
    unittest.main()

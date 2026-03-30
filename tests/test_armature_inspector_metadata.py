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

    def test_build_armature_placement_metadata_includes_bbox_ground_center_and_axes(self):
        fake_bpy = types.SimpleNamespace(
            data=types.SimpleNamespace(filepath="C:/assets/Example.blend", objects=[], armatures=[]),
        )

        with mock.patch.dict(sys.modules, {"bpy": fake_bpy}):
            if "inspector" in sys.modules:
                del sys.modules["inspector"]
            inspector = importlib.import_module("inspector")

        bones_data = [
            {"name": "Root", "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0]},
            {"name": "Hip", "head": [0.0, 0.0, 1.0], "tail": [0.0, 0.0, 1.2]},
            {"name": "Upper_Leg_Left", "head": [0.0, 0.2, 1.0], "tail": [0.0, 0.2, 0.3]},
            {"name": "Upper_Leg_Right", "head": [0.0, -0.2, 1.0], "tail": [0.0, -0.2, 0.3]},
            {"name": "Head", "head": [0.1, 0.0, 1.7], "tail": [0.2, 0.0, 1.9]},
        ]
        mesh_points = [
            [-0.3, -0.2, 0.0],
            [0.4, 0.2, 1.9],
        ]

        metadata = inspector.build_armature_placement_metadata(
            bones_data,
            mesh_points=mesh_points,
            driven_meshes=["BodyMesh"],
        )

        self.assertEqual(metadata["bbox_min"], [-0.3, -0.2, 0.0])
        self.assertEqual(metadata["bbox_max"], [0.4, 0.2, 1.9])
        self.assertAlmostEqual(metadata["bbox_ground_center"][0], 0.05)
        self.assertEqual(metadata["bbox_ground_center"][1:], [0.0, 0.0])
        self.assertAlmostEqual(metadata["bbox_height"], 1.9)
        self.assertEqual(metadata["up_axis"]["index"], 2)
        self.assertEqual(metadata["side_axis"]["index"], 1)
        self.assertEqual(metadata["forward_axis"]["index"], 0)
        self.assertEqual(metadata["driven_meshes"], ["BodyMesh"])


if __name__ == "__main__":
    unittest.main()

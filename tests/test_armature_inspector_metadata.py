import importlib
import json
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

    def test_tpose_up_axis_resolves_to_grounded_vertical_not_arm_span(self):
        # Regression for the remy_mixamo T-pose bug: arms extended along X make the
        # X bounding span slightly exceed standing height (Z), so the naive
        # "tallest-span axis = up" rule mis-picks X. The character rests on the
        # floor (feet at Z~=0), so the up axis must still resolve to Z.
        fake_bpy = types.SimpleNamespace(
            data=types.SimpleNamespace(filepath="C:/assets/Tpose.blend", objects=[], armatures=[]),
        )
        with mock.patch.dict(sys.modules, {"bpy": fake_bpy}):
            if "inspector" in sys.modules:
                del sys.modules["inspector"]
            inspector = importlib.import_module("inspector")

        bones_data = [
            {"name": "Hips", "head": [0.0, 0.0, 1.00], "tail": [0.0, 0.0, 1.10]},
            {"name": "Spine", "head": [0.0, 0.0, 1.10], "tail": [0.0, 0.0, 1.40]},
            {"name": "Neck", "head": [0.0, 0.0, 1.55], "tail": [0.0, 0.0, 1.70]},
            {"name": "Head", "head": [0.0, 0.0, 1.75], "tail": [0.0, 0.0, 1.90]},
            # Arms fully extended along X -> X midpoint span (~1.9) edges out the
            # Z height midpoint span (~1.77): the tie that defeats the naive heuristic.
            {"name": "LeftArm", "head": [0.10, 0.0, 1.50], "tail": [0.75, 0.0, 1.50]},
            {"name": "LeftHand", "head": [0.75, 0.0, 1.50], "tail": [1.15, 0.0, 1.50]},
            {"name": "RightArm", "head": [-0.10, 0.0, 1.50], "tail": [-0.75, 0.0, 1.50]},
            {"name": "RightHand", "head": [-0.75, 0.0, 1.50], "tail": [-1.15, 0.0, 1.50]},
            # Legs/feet land on the ground (Z ~= 0): the grounding signal for "up".
            {"name": "LeftUpLeg", "head": [0.10, 0.0, 1.00], "tail": [0.10, 0.0, 0.50]},
            {"name": "LeftFoot", "head": [0.10, 0.0, 0.10], "tail": [0.10, 0.15, 0.02]},
            {"name": "RightUpLeg", "head": [-0.10, 0.0, 1.00], "tail": [-0.10, 0.0, 0.50]},
            {"name": "RightFoot", "head": [-0.10, 0.0, 0.10], "tail": [-0.10, 0.15, 0.02]},
        ]

        metadata = inspector.build_armature_placement_metadata(bones_data)

        self.assertEqual(metadata["up_axis"]["index"], 2)
        self.assertEqual(metadata["up_axis"]["sign"], 1)

    def test_build_mesh_binding_no_meshes(self):
        fake_bpy = types.SimpleNamespace(
            data=types.SimpleNamespace(objects=[]),
        )
        with mock.patch.dict(sys.modules, {"bpy": fake_bpy}):
            if "inspector" in sys.modules:
                del sys.modules["inspector"]
            inspector = importlib.import_module("inspector")
        arm = types.SimpleNamespace(name="Rig")
        binding = inspector.build_mesh_binding(arm)
        self.assertEqual(binding["armature_object_name"], "Rig")
        self.assertEqual(binding["meshes"], [])

    def test_openmatexamplehuman_armature_all_has_mesh_binding(self):
        path = (
            REPO_ROOT
            / "tests"
            / "fixtures"
            / "openmatexamplehuman"
            / "Armature_all.json"
        )
        self.assertTrue(path.is_file(), msg=f"Missing {path}; regenerate exports")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("mesh_binding", data)
        mb = data["mesh_binding"]
        self.assertEqual(mb["armature_object_name"], data["armature_name"])
        self.assertIn("meshes", mb)
        names = [m["mesh_name"] for m in mb["meshes"]]
        self.assertEqual(names, sorted(names))
        for m in mb["meshes"]:
            self.assertIn(m["armature_link"], ("parent", "modifier", "parent_and_modifier"))
            self.assertIn("vertex_groups", m)
            self.assertIn("vertex_group_stats", m)
            self.assertIn("material_slots", m)
            self.assertIsInstance(m["warnings"], list)
        self.assertEqual(
            set(m["mesh_name"] for m in mb["meshes"]),
            set(data["placement_metadata"]["driven_meshes"]),
        )

    def test_lowpoly_rig_all_has_mesh_binding(self):
        path = (
            REPO_ROOT
            / "tests"
            / "fixtures"
            / "LowPolyCharacter4"
            / "rig_all.json"
        )
        self.assertTrue(path.is_file(), msg=f"Missing {path}; regenerate exports")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("mesh_binding", data)
        mb = data["mesh_binding"]
        self.assertEqual(mb["armature_object_name"], data["armature_name"])
        self.assertIn("meshes", mb)
        names = [m["mesh_name"] for m in mb["meshes"]]
        self.assertEqual(names, sorted(names))
        for m in mb["meshes"]:
            self.assertIn(m["armature_link"], ("parent", "modifier", "parent_and_modifier"))
            self.assertIn("vertex_groups", m)
            self.assertIn("vertex_group_stats", m)
            self.assertIn("material_slots", m)
            self.assertIsInstance(m["warnings"], list)
        self.assertEqual(
            set(m["mesh_name"] for m in mb["meshes"]),
            set(data["placement_metadata"]["driven_meshes"]),
        )


    def test_extract_armature_geometry_returns_world_coordinates(self):
        fake_bpy = types.SimpleNamespace(
            data=types.SimpleNamespace(filepath="", objects=[], armatures=[]),
        )
        with mock.patch.dict(sys.modules, {"bpy": fake_bpy}):
            if "inspector" in sys.modules:
                del sys.modules["inspector"]
            inspector = importlib.import_module("inspector")

        class FakeMatrix:
            # 0.0254 uniform scale, no rotation/translation — mirrors 1000idles Object_4.
            def __matmul__(self, vec):
                return type(vec)((0.0254 * vec[0], 0.0254 * vec[1], 0.0254 * vec[2]))

        class V(tuple):
            pass

        class FakeBone:
            def __init__(self):
                self.name = "Pelvis"
                self.parent = None
                self.head_local = V((0.0, 0.0, 35.0))
                self.tail_local = V((0.0, 0.0, 40.0))
                self.length = 5.0
                self.matrix_local = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        class FakeData:
            bones = [FakeBone()]

        class FakeArmature:
            data = FakeData()
            matrix_world = FakeMatrix()

        result = inspector.extract_armature_geometry(FakeArmature())
        self.assertAlmostEqual(result[0]["head"][2], 35.0 * 0.0254)
        self.assertAlmostEqual(result[0]["tail"][2], 40.0 * 0.0254)


    def test_collect_armature_mesh_bounds_returns_world_points(self):
        # FakeVector is a tuple subclass so indexing ([0],[1],[2]) and type(vec)((...))
        # both work — satisfying FakeMatrix.__matmul__.
        class FakeVector(tuple):
            pass

        class FakeMatrix:
            def __init__(self, s):
                self.s = s

            def __matmul__(self, vec):
                return type(vec)((self.s * vec[0], self.s * vec[1], self.s * vec[2]))

            def inverted(self):  # must NOT be called anymore
                raise AssertionError(
                    "armature inverse must not be used; bounds are world-space"
                )

        class V(tuple):
            pass

        class FakeMesh:
            type = "MESH"
            name = "Body"
            parent = None
            modifiers = []
            matrix_world = FakeMatrix(0.0254)
            bound_box = [V((0.0, 0.0, 0.0)), V((100.0, 0.0, 0.0))]

        fake_mathutils = types.SimpleNamespace(Vector=FakeVector)
        fake_bpy = types.SimpleNamespace(
            data=types.SimpleNamespace(filepath="", objects=[], armatures=[]),
        )
        # mathutils is imported inside the function body at call time, so it must be
        # present in sys.modules throughout module import AND the function call.
        with mock.patch.dict(sys.modules, {"bpy": fake_bpy, "mathutils": fake_mathutils}):
            if "inspector" in sys.modules:
                del sys.modules["inspector"]
            inspector = importlib.import_module("inspector")

            arm = type("Arm", (), {"matrix_world": FakeMatrix(0.0254)})()
            # Force the driven-mesh detection to return our fake mesh:
            original = inspector._meshes_driven_by_armature
            inspector._meshes_driven_by_armature = lambda a: [FakeMesh()]
            try:
                points, names = inspector._collect_armature_mesh_bounds(arm)
            finally:
                inspector._meshes_driven_by_armature = original

        self.assertEqual(names, ["Body"])
        # World-space: 100.0 * 0.0254 = 2.54 — NOT armature-local
        self.assertAlmostEqual(points[1][0], 100.0 * 0.0254)


    def test_build_mesh_binding_includes_world_bbox_per_mesh(self):
        class FakeVector(tuple):
            pass

        class FakeMatrix:
            def __matmul__(self, vec):
                # translate by (32, -10, 0) — multiply keeps coords then adds offset
                return type(vec)((vec[0] + 32.0, vec[1] - 10.0, vec[2]))

        class FakeVertexGroup:
            def __init__(self, name, index):
                self.name = name
                self.index = index

        class FakeMeshData:
            vertices = []

        class FakeMesh:
            type = "MESH"
            name = "Body"
            parent = None  # no parent link; modifier link will be set below
            modifiers = []
            vertex_groups = [FakeVertexGroup("Pelvis", 0)]
            data = FakeMeshData()
            material_slots = []
            matrix_world = FakeMatrix()
            # Unit cube corners (8 corners)
            bound_box = [
                FakeVector((0.0, 0.0, 0.0)),
                FakeVector((1.0, 0.0, 0.0)),
                FakeVector((1.0, 1.0, 0.0)),
                FakeVector((0.0, 1.0, 0.0)),
                FakeVector((0.0, 0.0, 1.0)),
                FakeVector((1.0, 0.0, 1.0)),
                FakeVector((1.0, 1.0, 1.0)),
                FakeVector((0.0, 1.0, 1.0)),
            ]

        fake_mathutils = types.SimpleNamespace(Vector=FakeVector)
        fake_bpy = types.SimpleNamespace(
            data=types.SimpleNamespace(filepath="", objects=[], armatures=[]),
        )

        with mock.patch.dict(sys.modules, {"bpy": fake_bpy, "mathutils": fake_mathutils}):
            if "inspector" in sys.modules:
                del sys.modules["inspector"]
            inspector = importlib.import_module("inspector")

            arm = types.SimpleNamespace(name="Rig")
            original = inspector._meshes_driven_by_armature
            inspector._meshes_driven_by_armature = lambda a: [FakeMesh()]
            try:
                binding = inspector.build_mesh_binding(arm)
            finally:
                inspector._meshes_driven_by_armature = original

        self.assertEqual(len(binding["meshes"]), 1)
        mesh = binding["meshes"][0]
        self.assertIn("bbox", mesh)
        self.assertEqual(sorted(mesh["bbox"].keys()), ["max", "min"])
        self.assertEqual(len(mesh["bbox"]["min"]), 3)
        self.assertEqual(len(mesh["bbox"]["max"]), 3)
        # The translation adds (32,-10,0) so min=[32,-10,0] max=[33,-9,1]
        self.assertAlmostEqual(mesh["bbox"]["min"][0], 32.0)
        self.assertAlmostEqual(mesh["bbox"]["min"][1], -10.0)
        self.assertAlmostEqual(mesh["bbox"]["min"][2], 0.0)
        self.assertAlmostEqual(mesh["bbox"]["max"][0], 33.0)
        self.assertAlmostEqual(mesh["bbox"]["max"][1], -9.0)
        self.assertAlmostEqual(mesh["bbox"]["max"][2], 1.0)


if __name__ == "__main__":
    unittest.main()

import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asam_human_builder.builder import (  # noqa: E402
    build_armature_spec,
    build_armature_spec_from_asset_dir,
    choose_generated_collection_action,
    resolve_default_asset_dir,
)
from asam_human_builder.blender_builder import build_armature_in_blender  # noqa: E402
from phase3_classifier.classifier import CORE_TARGETS, TARGET_PARENTS, write_asset_report  # noqa: E402


FIXTURE_ROOT = REPO_ROOT / "src" / "armature_inspector" / "output"


def _copy_asset_folder(asset_name: str) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix="asam_builder_fixture_"))
    destination = temp_root / asset_name
    shutil.copytree(FIXTURE_ROOT / asset_name, destination)
    return destination


def _placement_metadata():
    return {
        "bounds_source": "meshes",
        "driven_meshes": ["BodyMesh"],
        "bbox_min": [-0.4, -0.3, 0.0],
        "bbox_max": [0.4, 0.3, 2.0],
        "bbox_height": 2.0,
        "bbox_ground_center": [0.0, 0.0, 0.0],
        "forward_axis": {"index": 0, "name": "x", "sign": 1},
        "side_axis": {"index": 1, "name": "y", "sign": 1},
        "up_axis": {"index": 2, "name": "z", "sign": 1},
    }


def _base_classifier_report(asset_name: str = "SyntheticAsset", armature_name: str = "Rig") -> dict:
    semantic_mapping = {
        target: {
            "source_bone": None,
            "confidence": 0.0,
            "action": "create_in_builder",
            "evidence": {
                "name": 0.0,
                "hierarchy": 0.0,
                "geometry": 0.0,
                "source_origin": None,
                "role": None,
            },
            "notes": [],
        }
        for target in CORE_TARGETS
    }
    return {
        "recommended_primary_armature": armature_name,
        "semantic_mapping": semantic_mapping,
    }


def _base_build_plan(asset_name: str = "SyntheticAsset", armature_name: str = "Rig") -> dict:
    return {
        "asset_name": asset_name,
        "recommended_primary_armature": armature_name,
        "root_resolution": {
            "mode": "create_new_root",
            "target_bone": "Root",
            "source_bone": None,
            "rename_source_to_target": False,
            "failure_codes": [],
            "target_head": [0.0, 0.0, 0.0],
            "target_tail": [0.0, 0.0, 1.0],
            "up_axis": {"index": 2, "name": "z", "sign": 1},
            "use_connect": False,
        },
        "placement_metadata": _placement_metadata(),
        "proposed_asam_hierarchy": {
            "object_nodes": [
                {"name": "Grp_Root", "parent": None},
                {"name": "Armature_{0}".format(asset_name), "parent": "Grp_Root"},
            ],
            "bone_parents": {
                target: (
                    TARGET_PARENTS[target].format(asset_name=asset_name)
                    if target == "Root"
                    else TARGET_PARENTS[target]
                )
                for target in CORE_TARGETS
            },
        },
        "extras_preserved": [],
    }


def _bone(name: str, parent: str, head, tail) -> dict:
    return {
        "name": name,
        "parent": parent,
        "head": [float(value) for value in head],
        "tail": [float(value) for value in tail],
        "length": sum((tail[index] - head[index]) ** 2 for index in range(3)) ** 0.5,
    }


def _spec_bone(build_spec: dict, name: str) -> dict:
    return next(bone for bone in build_spec["bones"] if bone["name"] == name)


class _FakeProps:
    def __init__(self):
        self._props = {}

    def __getitem__(self, key):
        return self._props[key]

    def __setitem__(self, key, value):
        self._props[key] = value

    def get(self, key, default=None):
        return self._props.get(key, default)


class _FakeBone:
    def __init__(self, name: str):
        self.name = name
        self.head = [0.0, 0.0, 0.0]
        self.tail = [0.0, 0.0, 0.0]
        self.parent = None
        self.use_connect = False


class _FakeEditBones:
    def __init__(self):
        self._bones = {}

    def new(self, name: str):
        bone = _FakeBone(name)
        self._bones[name] = bone
        return bone

    def __getitem__(self, name: str):
        return self._bones[name]


class _FakeArmatureData(_FakeProps):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.users = 0
        self.edit_bones = _FakeEditBones()


class _FakeObject(_FakeProps):
    def __init__(self, name: str, data=None):
        super().__init__()
        self.name = name
        self.data = data
        self.parent = None
        self.mode = "OBJECT"
        self.selected = False
        self.empty_display_type = None
        self.type = "ARMATURE" if isinstance(data, _FakeArmatureData) else "EMPTY"

    def select_set(self, value: bool):
        self.selected = bool(value)


class _FakeObjectStore:
    def __init__(self):
        self._objects = {}

    def get(self, name: str):
        return self._objects.get(name)

    def new(self, name: str, data):
        obj = _FakeObject(name, data)
        self._objects[name] = obj
        if isinstance(data, _FakeArmatureData):
            data.users += 1
        return obj

    def remove(self, obj, do_unlink=True):
        self._objects.pop(obj.name, None)
        if isinstance(getattr(obj, "data", None), _FakeArmatureData):
            obj.data.users = max(0, obj.data.users - 1)


class _FakeArmatureStore:
    def __init__(self):
        self._armatures = {}

    def get(self, name: str):
        return self._armatures.get(name)

    def new(self, name: str):
        armature = _FakeArmatureData(name)
        self._armatures[name] = armature
        return armature

    def remove(self, armature):
        self._armatures.pop(armature.name, None)


class _FakeCollectionObjectLinks:
    def __init__(self, collection):
        self._collection = collection

    def link(self, obj):
        if obj not in self._collection._objects:
            self._collection._objects.append(obj)


class _FakeCollectionChildren:
    def __init__(self):
        self._children = {}

    def link(self, collection):
        self._children[collection.name] = collection

    def unlink(self, collection):
        self._children.pop(collection.name, None)

    def get(self, name: str):
        return self._children.get(name)


class _FakeCollection(_FakeProps):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self._objects = []
        self.objects = _FakeCollectionObjectLinks(self)
        self.children = _FakeCollectionChildren()

    @property
    def all_objects(self):
        return list(self._objects)


class _FakeCollectionStore:
    def __init__(self):
        self._collections = {}

    def __iter__(self):
        return iter(self._collections.values())

    def get(self, name: str):
        return self._collections.get(name)

    def new(self, name: str):
        collection = _FakeCollection(name)
        self._collections[name] = collection
        return collection

    def remove(self, collection):
        self._collections.pop(collection.name, None)


class _FakeViewLayerObjects:
    def __init__(self, context):
        self._context = context
        self._active = None

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, obj):
        self._active = obj
        self._context.object = obj


class _FakeContext:
    def __init__(self):
        self.object = None
        self.scene = type("Scene", (), {"collection": _FakeCollection("SceneRoot")})()
        self.view_layer = type("ViewLayer", (), {})()
        self.view_layer.objects = _FakeViewLayerObjects(self)


class _FakeObjectOps:
    def __init__(self, context):
        self._context = context

    def mode_set(self, mode: str):
        if self._context.object is not None:
            self._context.object.mode = mode

    def select_all(self, action: str):
        return None


class _FakeOps:
    def __init__(self, context):
        self.object = _FakeObjectOps(context)


class _FakeBpy:
    def __init__(self):
        self.context = _FakeContext()
        self.data = type("Data", (), {})()
        self.data.objects = _FakeObjectStore()
        self.data.armatures = _FakeArmatureStore()
        self.data.collections = _FakeCollectionStore()
        self.data.scenes = [self.context.scene]
        self.ops = _FakeOps(self.context)

    def add_source_armature(self, name: str):
        armature = self.data.armatures.new(name)
        source = self.data.objects.new(name, armature)
        return source


class AsamHumanBuilderTests(unittest.TestCase):
    def test_openmaterial_fixture_repairs_root_geometry(self):
        asset_dir = _copy_asset_folder("openmatexamplehuman")
        write_asset_report(asset_dir)

        _, build_plan, build_spec = build_armature_spec_from_asset_dir(asset_dir)

        self.assertEqual(build_plan["root_resolution"]["mode"], "create_new_root")
        self.assertEqual(build_spec["source_armature_name"], "Armature")
        self.assertEqual(len(build_spec["bones"]), len(CORE_TARGETS))

        root_bone = _spec_bone(build_spec, "Root")
        self.assertEqual(root_bone["geometry_source"], "root_resolution")
        self.assertEqual(root_bone["source_bone"], "Root")
        self.assertFalse(root_bone["use_connect"])

    def test_lowpoly_fixture_creates_new_root_and_preserves_extras(self):
        asset_dir = _copy_asset_folder("LowPolyCharacter4")
        write_asset_report(asset_dir)

        _, build_plan, build_spec = build_armature_spec_from_asset_dir(asset_dir)

        self.assertEqual(build_plan["root_resolution"]["mode"], "create_new_root")
        self.assertEqual(build_spec["source_armature_name"], "rig")
        self.assertGreater(len(build_spec["extras_preserved"]), 0)

        root_bone = _spec_bone(build_spec, "Root")
        self.assertEqual(root_bone["geometry_source"], "root_resolution")
        self.assertEqual(root_bone["source_bone"], "root")
        self.assertEqual(root_bone["head"], build_plan["root_resolution"]["target_head"])
        self.assertEqual(root_bone["tail"], build_plan["root_resolution"]["target_tail"])

        hip_bone = _spec_bone(build_spec, "Hip")
        side_axis = build_plan["placement_metadata"]["side_axis"]["index"]
        centerline = build_plan["placement_metadata"]["bbox_ground_center"][side_axis]
        lower_spine_bone = _spec_bone(build_spec, "Lower_Spine")

        self.assertEqual(hip_bone["geometry_source"], "centered_pelvis_pair")
        self.assertEqual(hip_bone["source_bone"], "DEF-pelvis.L")
        self.assertEqual(hip_bone["head"], root_bone["tail"])
        self.assertAlmostEqual(hip_bone["head"][side_axis], centerline)
        self.assertAlmostEqual(hip_bone["tail"][side_axis], centerline)
        for hip_value, spine_value in zip(hip_bone["tail"], lower_spine_bone["head"]):
            self.assertAlmostEqual(hip_value, spine_value, places=6)

    def test_repaired_paired_pelvis_creates_centered_hip_between_root_and_spine(self):
        classifier_report = _base_classifier_report()
        build_plan = _base_build_plan()
        classifier_report["semantic_mapping"]["Hip"]["action"] = "repair_in_builder"
        classifier_report["semantic_mapping"]["Hip"]["source_bone"] = "pelvis.L"
        classifier_report["semantic_mapping"]["Hip"]["notes"] = ["paired_sided_pelvis_requires_centering"]
        classifier_report["semantic_mapping"]["Lower_Spine"]["action"] = "direct_map"
        classifier_report["semantic_mapping"]["Lower_Spine"]["source_bone"] = "Lower_Spine"

        source_bones = {
            "pelvis.L": _bone("pelvis.L", "Root", (0.0, 0.0, 1.0), (0.15, 0.0, 1.1)),
            "pelvis.R": _bone("pelvis.R", "Root", (0.0, 0.0, 1.0), (-0.15, 0.0, 1.1)),
            "Lower_Spine": _bone("Lower_Spine", "Hip", (0.0, 0.0, 1.2), (0.0, 0.0, 1.45)),
        }

        build_spec = build_armature_spec(classifier_report, build_plan, source_bones)
        root_bone = _spec_bone(build_spec, "Root")
        hip_bone = _spec_bone(build_spec, "Hip")
        side_axis = build_plan["placement_metadata"]["side_axis"]["index"]
        centerline = build_plan["placement_metadata"]["bbox_ground_center"][side_axis]

        self.assertEqual(hip_bone["geometry_source"], "centered_pelvis_pair")
        self.assertEqual(hip_bone["source_bone"], "pelvis.L")
        self.assertEqual(hip_bone["head"], root_bone["tail"])
        self.assertEqual(hip_bone["tail"], [0.0, 0.0, 1.2])
        self.assertAlmostEqual(hip_bone["head"][side_axis], centerline)
        self.assertAlmostEqual(hip_bone["tail"][side_axis], centerline)

    def test_missing_right_limb_prefers_mirror_geometry(self):
        classifier_report = _base_classifier_report()
        build_plan = _base_build_plan()
        classifier_report["semantic_mapping"]["Lower_Arm_Left"]["action"] = "direct_map"
        classifier_report["semantic_mapping"]["Lower_Arm_Left"]["source_bone"] = "lower_arm.L"
        classifier_report["semantic_mapping"]["Lower_Arm_Right"]["action"] = "create_in_builder"

        source_bones = {
            "lower_arm.L": _bone("lower_arm.L", "Upper_Arm_Left", (0.2, 0.25, 1.4), (0.25, 0.45, 1.0)),
        }

        build_spec = build_armature_spec(classifier_report, build_plan, source_bones)
        right_bone = _spec_bone(build_spec, "Lower_Arm_Right")

        self.assertEqual(right_bone["geometry_source"], "mirrored_opposite")
        self.assertAlmostEqual(right_bone["head"][0], 0.2)
        self.assertAlmostEqual(right_bone["head"][1], -0.25)
        self.assertAlmostEqual(right_bone["tail"][1], -0.45)

    def test_missing_spine_segment_prefers_interpolation(self):
        classifier_report = _base_classifier_report()
        build_plan = _base_build_plan()
        classifier_report["semantic_mapping"]["Hip"]["action"] = "direct_map"
        classifier_report["semantic_mapping"]["Hip"]["source_bone"] = "Hip"
        classifier_report["semantic_mapping"]["Upper_Spine"]["action"] = "direct_map"
        classifier_report["semantic_mapping"]["Upper_Spine"]["source_bone"] = "Upper_Spine"

        source_bones = {
            "Hip": _bone("Hip", "Root", (0.0, 0.0, 1.0), (0.0, 0.0, 1.2)),
            "Upper_Spine": _bone("Upper_Spine", "Lower_Spine", (0.0, 0.0, 1.45), (0.0, 0.0, 1.7)),
        }

        build_spec = build_armature_spec(classifier_report, build_plan, source_bones)
        lower_spine = _spec_bone(build_spec, "Lower_Spine")

        self.assertEqual(lower_spine["geometry_source"], "interpolated_chain")
        self.assertEqual(lower_spine["head"], [0.0, 0.0, 1.2])
        self.assertEqual(lower_spine["tail"], [0.0, 0.0, 1.45])

    def test_missing_head_prefers_parent_extrapolation(self):
        classifier_report = _base_classifier_report()
        build_plan = _base_build_plan()
        classifier_report["semantic_mapping"]["Neck"]["action"] = "direct_map"
        classifier_report["semantic_mapping"]["Neck"]["source_bone"] = "Neck"

        source_bones = {
            "Neck": _bone("Neck", "Upper_Spine", (0.0, 0.0, 1.55), (0.0, 0.0, 1.72)),
        }

        build_spec = build_armature_spec(classifier_report, build_plan, source_bones)
        head = _spec_bone(build_spec, "Head")

        self.assertEqual(head["geometry_source"], "extrapolated_parent")
        self.assertEqual(head["head"], [0.0, 0.0, 1.72])
        self.assertGreater(head["tail"][2], head["head"][2])

    def test_generated_collection_action_rebuilds_only_safe_generated_output(self):
        action = choose_generated_collection_action(
            [
                {
                    "name": "ASAM_TestAsset",
                    "generated": True,
                    "asset_name": "TestAsset",
                }
            ],
            "ASAM_TestAsset",
            "TestAsset",
        )

        self.assertEqual(action, "rebuild")

    def test_generated_collection_action_rejects_unmarked_conflict(self):
        with self.assertRaises(ValueError):
            choose_generated_collection_action(
                [
                    {
                        "name": "ASAM_TestAsset",
                        "generated": False,
                        "asset_name": None,
                    }
                ],
                "ASAM_TestAsset",
                "TestAsset",
            )

    def test_resolve_default_asset_dir_uses_blend_name(self):
        resolved = resolve_default_asset_dir(
            "C:/assets/openmatexamplehuman.blend",
            REPO_ROOT / "src" / "asam_human_builder",
        )

        self.assertEqual(
            resolved,
            (REPO_ROOT / "src" / "armature_inspector" / "output" / "openmatexamplehuman").resolve(),
        )

    def test_blender_builder_uses_asset_scoped_group_root_name_when_scene_name_is_taken(self):
        bpy_module = _FakeBpy()
        bpy_module.add_source_armature("Armature")
        bpy_module.data.objects.new("Grp_Root", None)

        build_spec = {
            "asset_name": "openmatexamplehuman",
            "source_armature_name": "Armature",
            "generated_collection_name": "ASAM_openmatexamplehuman",
            "group_root_name": "Grp_Root",
            "generated_armature_name": "Armature_openmatexamplehuman",
            "bones": [],
        }

        execution_result = build_armature_in_blender(build_spec, bpy_module)

        self.assertEqual(execution_result["generated_collection_name"], "ASAM_openmatexamplehuman")
        self.assertEqual(execution_result["group_root_name"], "Grp_Root_openmatexamplehuman")
        self.assertEqual(execution_result["generated_armature_name"], "Armature_openmatexamplehuman")

    def test_blender_builder_suffixes_generated_armature_name_when_needed(self):
        bpy_module = _FakeBpy()
        bpy_module.add_source_armature("Armature")
        bpy_module.data.armatures.new("Armature_openmatexamplehuman")

        build_spec = {
            "asset_name": "openmatexamplehuman",
            "source_armature_name": "Armature",
            "generated_collection_name": "ASAM_openmatexamplehuman",
            "group_root_name": "Grp_Root",
            "generated_armature_name": "Armature_openmatexamplehuman",
            "bones": [],
        }

        execution_result = build_armature_in_blender(build_spec, bpy_module)

        self.assertEqual(execution_result["group_root_name"], "Grp_Root")
        self.assertEqual(execution_result["generated_armature_name"], "Armature_openmatexamplehuman_Generated")


if __name__ == "__main__":
    unittest.main()

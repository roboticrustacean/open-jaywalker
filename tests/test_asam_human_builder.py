import copy
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
    GENERATED_ASSET_KEY,
    GENERATED_MARKER_KEY,
    build_armature_spec,
    build_armature_spec_from_asset_dir,
    build_builder_report,
    choose_generated_collection_action,
    compute_vertex_group_remap_plan,
    resolve_default_asset_dir,
    validate_builder_inputs,
)
from asam_human_builder.blender_builder import (  # noqa: E402
    build_armature_in_blender,
    purge_previous_generated_artifacts,
)
from phase3_classifier.classifier import (  # noqa: E402
    CORE_TARGETS,
    DEFERRED_TARGETS,
    TARGET_PARENTS,
    write_asset_report,
)


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures"


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


def _mesh_binding(armature_name: str = "Rig") -> dict:
    return {
        "armature_object_name": armature_name,
        "meshes": [
            {
                "mesh_name": "BodyMesh",
                "armature_link": "modifier",
                "modifiers": [
                    {
                        "stack_index": 0,
                        "type": "ARMATURE",
                        "name": "Armature",
                        "object": armature_name,
                    }
                ],
                "vertex_groups": ["Hip"],
                "vertex_group_stats": {
                    "non_empty_group_count": 1,
                    "per_group": [{"name": "Hip", "weighted_vertex_count": 8}],
                },
                "material_slots": [{"slot_index": 0, "material_name": "BodyMat"}],
                "warnings": [],
            }
        ],
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
        "root_resolutions": [
            {
                "subtree_name": "Grp_Root",
                "mode": "create_new_root",
                "target_bone": "Root",
                "source_bone": None,
                "rename_source_to_target": False,
                "grp_root_local_origin": [0.0, 0.0, 0.0],
                "blocker_codes": [],
                "advisories": [],
                "up_axis": {"index": 2, "name": "z", "sign": 1},
            }
        ],
        "placement_metadata": _placement_metadata(),
        "mesh_binding": _mesh_binding(armature_name),
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


def _assert_vec_almost_equal(test_case: unittest.TestCase, actual, expected, places: int = 9) -> None:
    test_case.assertEqual(len(actual), len(expected))
    for actual_value, expected_value in zip(actual, expected):
        test_case.assertAlmostEqual(actual_value, expected_value, places=places)


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


class _FakeMeshData(_FakeProps):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.users = 0

    def copy(self):
        copied = _FakeMeshData("{0}_copy".format(self.name))
        copied._props = copy.deepcopy(self._props)
        return copied


class _FakeMaterial:
    def __init__(self, name: str):
        self.name = name

    def copy(self):
        return _FakeMaterial(self.name)


class _FakeMaterialSlot:
    def __init__(self, material_name: str):
        self.material = _FakeMaterial(material_name)

    def copy(self):
        copied = _FakeMaterialSlot(self.material.name)
        copied.material = self.material.copy()
        return copied


class _FakeModifier:
    def __init__(self, name: str, modifier_type: str, obj=None):
        self.name = name
        self.type = modifier_type
        self.object = obj

    def copy(self):
        return _FakeModifier(self.name, self.type, self.object)


class _FakeObject(_FakeProps):
    def __init__(self, name: str, data=None):
        super().__init__()
        self.name = name
        self.data = data
        self.parent = None
        self.mode = "OBJECT"
        self.selected = False
        self.empty_display_type = None
        self.modifiers = []
        self.vertex_groups = []
        self.material_slots = []
        self.matrix_world = ("world", name)
        self.location = (0.0, 0.0, 0.0)
        if isinstance(data, _FakeArmatureData):
            self.type = "ARMATURE"
        elif isinstance(data, _FakeMeshData):
            self.type = "MESH"
        else:
            self.type = "EMPTY"

    def copy(self):
        copied = _FakeObject("{0}_copy".format(self.name), self.data)
        copied.parent = self.parent
        copied.mode = self.mode
        copied.selected = self.selected
        copied.empty_display_type = self.empty_display_type
        copied.modifiers = [modifier.copy() for modifier in self.modifiers]
        copied.vertex_groups = list(self.vertex_groups)
        copied.material_slots = [slot.copy() for slot in self.material_slots]
        copied.matrix_world = self.matrix_world
        copied._props = copy.deepcopy(self._props)
        return copied

    def select_set(self, value: bool):
        self.selected = bool(value)


class _FakeObjectStore:
    def __init__(self):
        self._objects = {}

    def __iter__(self):
        return iter(list(self._objects.values()))

    def get(self, name: str):
        return self._objects.get(name)

    def contains(self, obj):
        return any(registered is obj for registered in self._objects.values())

    def new(self, name: str, data):
        obj = _FakeObject(name, data)
        self._objects[name] = obj
        if isinstance(data, (_FakeArmatureData, _FakeMeshData)):
            data.users += 1
        return obj

    def register(self, obj):
        if self.contains(obj):
            return obj
        existing = self._objects.get(obj.name)
        if existing is not None and existing is not obj:
            raise AssertionError("object name already registered: {0}".format(obj.name))
        self._objects[obj.name] = obj
        if isinstance(getattr(obj, "data", None), (_FakeArmatureData, _FakeMeshData)):
            obj.data.users += 1
        return obj

    def remove(self, obj, do_unlink=True):
        self._objects.pop(obj.name, None)
        if isinstance(getattr(obj, "data", None), (_FakeArmatureData, _FakeMeshData)):
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


class _FakeMeshStore:
    def __init__(self):
        self._meshes = {}

    def get(self, name: str):
        return self._meshes.get(name)

    def new(self, name: str):
        mesh = _FakeMeshData(name)
        self._meshes[name] = mesh
        return mesh

    def remove(self, mesh):
        self._meshes.pop(mesh.name, None)


class _FakeCollectionObjectLinks:
    def __init__(self, collection):
        self._collection = collection

    def link(self, obj):
        if obj not in self._collection._objects:
            self._collection._objects.append(obj)
        bpy_module = getattr(self._collection, "_bpy_module", None)
        if bpy_module is not None and not bpy_module.data.objects.contains(obj):
            bpy_module.data.objects.register(obj)


class _FakeCollectionChildren:
    def __init__(self):
        self._children = {}

    def link(self, collection):
        self._children[collection.name] = collection

    def unlink(self, collection):
        self._children.pop(collection.name, None)

    def get(self, name: str):
        return self._children.get(name)

    def __iter__(self):
        return iter(list(self._children.values()))


class _FakeCollection(_FakeProps):
    def __init__(self, name: str, bpy_module=None):
        super().__init__()
        self.name = name
        self._bpy_module = bpy_module
        self._objects = []
        self.objects = _FakeCollectionObjectLinks(self)
        self.children = _FakeCollectionChildren()

    @property
    def all_objects(self):
        return list(self._objects)


class _FakeCollectionStore:
    def __init__(self, bpy_module=None):
        self._bpy_module = bpy_module
        self._collections = {}

    def __iter__(self):
        return iter(self._collections.values())

    def get(self, name: str):
        return self._collections.get(name)

    def new(self, name: str):
        collection = _FakeCollection(name, self._bpy_module)
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
        self.data.meshes = _FakeMeshStore()
        self.data.collections = _FakeCollectionStore(self)
        self.data.scenes = [self.context.scene]
        # Expose isinstance-able type markers for purge_previous_generated_artifacts
        # to distinguish armature vs mesh data blocks without a parallel type-name
        # lookup. Real bpy exposes these via bpy.types.Armature / bpy.types.Mesh.
        self.types = type("Types", (), {"Armature": _FakeArmatureData, "Mesh": _FakeMeshData})()
        self.ops = _FakeOps(self.context)

    def add_source_armature(self, name: str):
        armature = self.data.armatures.new(name)
        source = self.data.objects.new(name, armature)
        return source

    def add_source_mesh(self, name: str, source_armature=None, armature_modifier=True):
        mesh_data = _FakeMeshData("{0}Data".format(name))
        source = self.data.objects.new(name, mesh_data)
        source.material_slots = [_FakeMaterialSlot("BodyMat")]
        if source_armature is not None:
            source.parent = source_armature
        if armature_modifier:
            source.modifiers.append(_FakeModifier("Armature", "ARMATURE", source_armature))
        return source


def _minimal_crowd_build_spec(asset_name="crowd", character_id="Hero000", source_armature="Object_4"):
    """A tiny generated-armature spec for fake-bpy crowd tests: one Hip from 'Pelvis'."""
    return {
        "asset_name": asset_name,
        "source_armature_name": source_armature,
        "generated_collection_name": "ASAM_{0}".format(asset_name),
        "group_root_name": "Grp_Root",
        "generated_armature_name": "Armature_{0}".format(asset_name),
        "grp_root_local_origin": [0.0, 0.0, 0.0],
        "bones": [
            {"name": "Hip", "parent_bone": None, "head": [0.0, 0.0, 1.0], "tail": [0.0, 0.0, 1.2],
             "use_connect": False, "geometry_source": "source_bone", "source_bone": "Pelvis",
             "semantic_action": "direct_map"},
        ],
        "mesh_binding": {"armature_object_name": source_armature, "meshes": []},
        "extras_preserved": [],
        "preserved_pelvis_pair": [],
        "warnings": [],
    }


def _fake_with_source_armature(source_armature="Object_4", meshes=None):
    """_FakeBpy with the source armature present, plus optional (mesh_name, [vgroups]) meshes."""
    bpy = _FakeBpy()
    armature = bpy.add_source_armature(source_armature)
    for mesh_name, vgroups in (meshes or []):
        mesh = bpy.add_source_mesh(mesh_name, armature)
        mesh.vertex_groups = list(vgroups)
    return bpy


class AsamHumanBuilderTests(unittest.TestCase):
    def test_builder_report_carries_root_anchor_metadata(self):
        """builder_report records grp_root_local_origin, root_mode, and preserved_source_root."""
        asset_dir = _copy_asset_folder("openmatexamplehuman")
        write_asset_report(asset_dir)
        _, build_plan, build_spec = build_armature_spec_from_asset_dir(asset_dir)
        bpy_module = _FakeBpy()
        bpy_module.add_source_armature("Armature")
        for mesh_record in build_plan["mesh_binding"]["meshes"]:
            bpy_module.add_source_mesh(
                mesh_record["mesh_name"],
                bpy_module.data.objects.get("Armature"),
            )
        execution_result = build_armature_in_blender(build_spec, bpy_module)
        report = build_builder_report(build_spec, execution_result)
        self.assertEqual(
            report["grp_root_local_origin"],
            build_plan["root_resolutions"][0]["grp_root_local_origin"],
        )
        self.assertEqual(report["root_mode"], "reuse_existing_root")
        self.assertIsNone(report["preserved_source_root"])

    def test_grp_root_location_set_to_bbox_ground_center(self):
        """Grp_Root empty's location must equal the source-frame bbox_ground_center."""
        asset_dir = _copy_asset_folder("openmatexamplehuman")
        write_asset_report(asset_dir)
        _, build_plan, build_spec = build_armature_spec_from_asset_dir(asset_dir)
        bpy_module = _FakeBpy()
        bpy_module.add_source_armature("Armature")
        for mesh_record in build_plan["mesh_binding"]["meshes"]:
            bpy_module.add_source_mesh(
                mesh_record["mesh_name"],
                bpy_module.data.objects.get("Armature"),
            )
        build_armature_in_blender(build_spec, bpy_module)
        grp_root = bpy_module.data.objects.get("Grp_Root")
        self.assertIsNotNone(grp_root)
        expected = build_plan["root_resolutions"][0]["grp_root_local_origin"]
        for actual, expected_value in zip(grp_root.location, expected):
            self.assertAlmostEqual(actual, expected_value, places=6)

    def test_mesh_world_matrix_not_translated_by_builder(self):
        """The duplicated mesh keeps the source mesh's matrix_world unchanged."""
        asset_dir = _copy_asset_folder("openmatexamplehuman")
        write_asset_report(asset_dir)
        _, build_plan, build_spec = build_armature_spec_from_asset_dir(asset_dir)
        bpy_module = _FakeBpy()
        bpy_module.add_source_armature("Armature")
        for mesh_record in build_plan["mesh_binding"]["meshes"]:
            mesh = bpy_module.add_source_mesh(
                mesh_record["mesh_name"],
                bpy_module.data.objects.get("Armature"),
            )
            mesh.matrix_world = ("source_world", mesh_record["mesh_name"])
        build_armature_in_blender(build_spec, bpy_module)
        for mesh_record in build_plan["mesh_binding"]["meshes"]:
            generated = bpy_module.data.objects.get("ASAM_{0}".format(mesh_record["mesh_name"]))
            self.assertIsNotNone(generated)
            self.assertEqual(generated.matrix_world, ("source_world", mesh_record["mesh_name"]))

    def test_synthesized_path_preserves_source_root_as_sibling_extra(self):
        """When mode == create_new_root and preserve_source_root_as_extra is true,
        the source root bone is appended as a non-ASAM sibling extra parented to
        Armature (parent_bone=None), keeping its source-bone identity for skin
        weight retention."""
        report = _base_classifier_report()
        plan = _base_build_plan()
        plan["root_resolutions"][0].update({
            "mode": "create_new_root",
            "source_bone": "Root",
            "preserve_source_root_as_extra": True,
            "grp_root_local_origin": [0.05, 0.0, 0.0],
        })
        source_bones = {
            "Root": _bone("Root", None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.6)),
        }
        spec = build_armature_spec(report, plan, source_bones)
        extra = next(
            (b for b in spec["bones"] if b.get("geometry_source") == "preserved_source_root"),
            None,
        )
        self.assertIsNotNone(extra)
        self.assertEqual(extra["source_bone"], "Root")
        # Source name collides with "Root" of the synthesized core target so the
        # generated extra is renamed.
        self.assertEqual(extra["name"], "Root_Source")
        # Source root.head was (0, 0, 0); local = source - grp_root_local_origin.
        self.assertAlmostEqual(extra["head"][0], -0.05, places=6)
        self.assertAlmostEqual(extra["head"][1], 0.0, places=6)
        self.assertAlmostEqual(extra["head"][2], 0.0, places=6)
        # Extra is parented to the Armature (not to Root)
        self.assertIsNone(extra["parent_bone"])

    def test_synthesized_path_skips_extra_when_flag_false(self):
        report = _base_classifier_report()
        plan = _base_build_plan()
        plan["root_resolutions"][0].update({
            "mode": "create_new_root",
            "source_bone": "Root",
            "preserve_source_root_as_extra": False,
            "grp_root_local_origin": [0.0, 0.0, 0.0],
        })
        source_bones = {
            "Root": _bone("Root", None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.6)),
        }
        spec = build_armature_spec(report, plan, source_bones)
        extras = [b for b in spec["bones"] if b.get("geometry_source") == "preserved_source_root"]
        self.assertEqual(extras, [])

    def test_synthesized_root_is_at_grp_root_local_origin(self):
        """When mode == create_new_root, the synthesized Root sits at (0,0,0)
        in spec coords (Grp_Root-local) with the tail offset along the up axis."""
        report = _base_classifier_report()
        plan = _base_build_plan()
        bbox_height = plan["placement_metadata"]["bbox_height"]
        plan["root_resolutions"][0].update({
            "mode": "create_new_root",
            "source_bone": "Root",
            "grp_root_local_origin": [0.1, 0.2, 0.3],
            "up_axis": {"index": 2, "name": "z", "sign": 1},
        })
        source_bones = {
            "Root": _bone("Root", None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.5)),
        }

        spec = build_armature_spec(report, plan, source_bones)
        root_bone = _spec_bone(spec, "Root")
        for value in root_bone["head"]:
            self.assertAlmostEqual(value, 0.0, places=6)
        # tail = (0, 0, +bbox_height * 0.18) in local coords
        self.assertAlmostEqual(root_bone["tail"][0], 0.0, places=6)
        self.assertAlmostEqual(root_bone["tail"][1], 0.0, places=6)
        self.assertAlmostEqual(root_bone["tail"][2], bbox_height * 0.18, places=6)

    def test_built_root_bone_is_in_grp_root_local_space(self):
        """build_armature_spec stores bone head/tail in Grp_Root-local frame
        (= source-world coords minus grp_root_local_origin)."""
        asset_dir = _copy_asset_folder("openmatexamplehuman")
        write_asset_report(asset_dir)
        _, build_plan, build_spec = build_armature_spec_from_asset_dir(asset_dir)
        grp_root_local_origin = build_plan["root_resolutions"][0]["grp_root_local_origin"]
        # openmatexamplehuman source root sits at source-world (0,0,0).
        source_root_head = (0.0, 0.0, 0.0)
        root_bone = _spec_bone(build_spec, "Root")
        expected_head = [source_root_head[i] - grp_root_local_origin[i] for i in range(3)]
        for actual, expected in zip(root_bone["head"], expected_head):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_openmaterial_fixture_reuses_source_root(self):
        asset_dir = _copy_asset_folder("openmatexamplehuman")
        write_asset_report(asset_dir)

        _, build_plan, build_spec = build_armature_spec_from_asset_dir(asset_dir)

        self.assertEqual(build_plan["root_resolutions"][0]["mode"], "reuse_existing_root")
        self.assertEqual(build_spec["source_armature_name"], "Armature")
        self.assertEqual(len(build_spec["bones"]), len(CORE_TARGETS))

        root_bone = _spec_bone(build_spec, "Root")
        self.assertEqual(root_bone["geometry_source"], "source_root")
        self.assertEqual(root_bone["source_bone"], "Root")
        self.assertFalse(root_bone["use_connect"])

    def test_lowpoly_fixture_creates_new_root_and_preserves_extras(self):
        asset_dir = _copy_asset_folder("LowPolyCharacter4")
        write_asset_report(asset_dir)

        _, build_plan, build_spec = build_armature_spec_from_asset_dir(asset_dir)

        self.assertEqual(build_plan["root_resolutions"][0]["mode"], "create_new_root")
        self.assertEqual(build_spec["source_armature_name"], "rig")
        self.assertGreater(len(build_spec["extras_preserved"]), 0)

        # Root preservation is disabled: the generated armature must contain exactly
        # one root bone (the synthesized vertical Root), not the horizontal source root.
        root_bones = [bone for bone in build_spec["bones"] if bone["name"].lower() == "root"]
        self.assertEqual([bone["name"] for bone in root_bones], ["Root"])
        self.assertEqual(
            [bone for bone in build_spec["bones"] if bone.get("geometry_source") == "preserved_source_root"],
            [],
        )

        root_bone = _spec_bone(build_spec, "Root")
        self.assertEqual(root_bone["geometry_source"], "root_resolution")
        self.assertEqual(root_bone["source_bone"], "root")
        # Synthesized Root sits at Grp_Root local origin after the _to_grp_root_local rebase.
        for value in root_bone["head"]:
            self.assertAlmostEqual(value, 0.0, places=6)

        hip_bone = _spec_bone(build_spec, "Hip")
        side_axis = build_plan["placement_metadata"]["side_axis"]["index"]
        up_axis = build_plan["placement_metadata"]["up_axis"]["index"]
        # In Grp_Root-local coords, the centerline is at 0 (bbox_ground_center
        # coincides with grp_root_local_origin on the side axis).
        local_centerline = 0.0

        # Hip is the spine-root pivot (DEF-spine), built from real source geometry.
        self.assertEqual(hip_bone["geometry_source"], "source_bone")
        self.assertEqual(hip_bone["source_bone"], "DEF-spine")
        # Hip head sits at the pelvis (level of the preserved pelvis-pair heads),
        # centered; the synthesized Root tail reaches the Hip head (Root -> Hip).
        pelvis_left = _spec_bone(build_spec, "DEF-pelvis_Left")
        self.assertAlmostEqual(hip_bone["head"][up_axis], pelvis_left["head"][up_axis], places=6)
        self.assertAlmostEqual(root_bone["tail"][up_axis], hip_bone["head"][up_axis], places=6)
        self.assertAlmostEqual(hip_bone["head"][side_axis], local_centerline, places=6)
        # Hip is a real bone pointing up to the lumbar (non-degenerate).
        self.assertGreater(hip_bone["tail"][up_axis], hip_bone["head"][up_axis])
        # ASAM hierarchy: Root -> Hip -> {Lower_Spine, Upper_Leg_*}; Lower_Spine sits
        # above Hip on the next spine segment (no overlap).
        self.assertEqual(hip_bone["parent_bone"], "Root")
        lower_spine_bone = _spec_bone(build_spec, "Lower_Spine")
        self.assertEqual(lower_spine_bone["parent_bone"], "Hip")
        self.assertEqual(lower_spine_bone["source_bone"], "DEF-spine.001")
        self.assertGreater(lower_spine_bone["head"][up_axis], hip_bone["head"][up_axis])
        self.assertEqual(_spec_bone(build_spec, "Upper_Leg_Left")["parent_bone"], "Hip")

        # Lock the realistic-data contract: both source pelvis bones must be
        # preserved as children of Hip under their ASAM spec-style names, and
        # the synthetic Hip itself must not appear in preserved_pelvis_pair.
        self.assertEqual(
            build_spec["preserved_pelvis_pair"],
            [
                {"source_bone_name": "DEF-pelvis.L", "generated_bone_name": "DEF-pelvis_Left", "parent": "Hip"},
                {"source_bone_name": "DEF-pelvis.R", "generated_bone_name": "DEF-pelvis_Right", "parent": "Hip"},
            ],
        )
        for generated_name, source_name in (("DEF-pelvis_Left", "DEF-pelvis.L"), ("DEF-pelvis_Right", "DEF-pelvis.R")):
            preserved_bone = _spec_bone(build_spec, generated_name)
            self.assertEqual(preserved_bone["parent_bone"], "Hip")
            self.assertEqual(preserved_bone["geometry_source"], "source_bone")
            self.assertEqual(preserved_bone["source_bone"], source_name)
            self.assertEqual(preserved_bone["semantic_action"], "preserve_paired_pelvis")

    def test_repaired_paired_pelvis_creates_centered_hip_at_pelvis_pointing_to_spine(self):
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
        up_axis = build_plan["placement_metadata"]["up_axis"]["index"]
        centerline = build_plan["placement_metadata"]["bbox_ground_center"][side_axis]

        self.assertEqual(hip_bone["geometry_source"], "centered_pelvis_pair")
        self.assertEqual(hip_bone["source_bone"], "pelvis.L")
        # Hip head is anchored at the centered pelvis-pair head (both pelvis heads
        # at z=1.0). The synthesized Root tail is extended up to meet the Hip head
        # (Root -> Hip), so the two connect at the pelvis.
        self.assertEqual(hip_bone["head"], [0.0, 0.0, 1.0])
        self.assertAlmostEqual(root_bone["tail"][up_axis], hip_bone["head"][up_axis])
        # Tail still points to the Lower_Spine head, so Hip connects pelvis -> spine.
        self.assertEqual(hip_bone["tail"], [0.0, 0.0, 1.2])
        self.assertAlmostEqual(hip_bone["head"][side_axis], centerline)
        self.assertAlmostEqual(hip_bone["tail"][side_axis], centerline)

    def test_paired_pelvis_preserved_as_hip_children(self):
        """Both source pelvis bones are added to the build spec as children of Hip
        with ASAM spec-style _Left / _Right suffixes."""
        classifier_report = _base_classifier_report()
        build_plan = _base_build_plan()
        classifier_report["semantic_mapping"]["Hip"]["action"] = "repair_in_builder"
        classifier_report["semantic_mapping"]["Hip"]["source_bone"] = "pelvis.L"
        classifier_report["semantic_mapping"]["Hip"]["notes"] = ["paired_sided_pelvis_requires_centering"]
        # Mesh binding must report non-zero weights for the pelvis groups so the
        # skip-if-unweighted gate (Task 4) lets them through.
        build_plan["mesh_binding"]["meshes"][0]["vertex_groups"] = ["pelvis.L", "pelvis.R"]
        build_plan["mesh_binding"]["meshes"][0]["vertex_group_stats"] = {
            "non_empty_group_count": 2,
            "per_group": [
                {"name": "pelvis.L", "weighted_vertex_count": 12},
                {"name": "pelvis.R", "weighted_vertex_count": 11},
            ],
        }

        source_bones = {
            "pelvis.L": _bone("pelvis.L", "Root", (0.0, 0.0, 1.0), (0.15, 0.0, 1.1)),
            "pelvis.R": _bone("pelvis.R", "Root", (0.0, 0.0, 1.0), (-0.15, 0.0, 1.1)),
        }

        build_spec = build_armature_spec(classifier_report, build_plan, source_bones)

        bone_names = [bone["name"] for bone in build_spec["bones"]]
        self.assertIn("pelvis_Left", bone_names)
        self.assertIn("pelvis_Right", bone_names)
        # The source-style names must NOT leak into the generated armature.
        self.assertNotIn("pelvis.L", bone_names)
        self.assertNotIn("pelvis.R", bone_names)

        for generated_name, source_name in (("pelvis_Left", "pelvis.L"), ("pelvis_Right", "pelvis.R")):
            preserved = _spec_bone(build_spec, generated_name)
            self.assertEqual(preserved["parent_bone"], "Hip")
            self.assertEqual(preserved["geometry_source"], "source_bone")
            self.assertEqual(preserved["source_bone"], source_name)
            self.assertEqual(preserved["semantic_action"], "preserve_paired_pelvis")

        self.assertEqual(
            build_spec["preserved_pelvis_pair"],
            [
                {"source_bone_name": "pelvis.L", "generated_bone_name": "pelvis_Left", "parent": "Hip"},
                {"source_bone_name": "pelvis.R", "generated_bone_name": "pelvis_Right", "parent": "Hip"},
            ],
        )

    def test_paired_pelvis_vertex_groups_rename_to_spec_style_names(self):
        """After preservation, pelvis vertex groups rename to the spec-style
        extension names - never to the synthetic Hip - and are not orphaned."""
        classifier_report = _base_classifier_report()
        build_plan = _base_build_plan()
        classifier_report["semantic_mapping"]["Hip"]["action"] = "repair_in_builder"
        classifier_report["semantic_mapping"]["Hip"]["source_bone"] = "pelvis.L"
        classifier_report["semantic_mapping"]["Hip"]["notes"] = ["paired_sided_pelvis_requires_centering"]
        build_plan["mesh_binding"]["meshes"][0]["vertex_groups"] = ["pelvis.L", "pelvis.R"]
        build_plan["mesh_binding"]["meshes"][0]["vertex_group_stats"] = {
            "non_empty_group_count": 2,
            "per_group": [
                {"name": "pelvis.L", "weighted_vertex_count": 12},
                {"name": "pelvis.R", "weighted_vertex_count": 11},
            ],
        }

        source_bones = {
            "pelvis.L": _bone("pelvis.L", "Root", (0.0, 0.0, 1.0), (0.15, 0.0, 1.1)),
            "pelvis.R": _bone("pelvis.R", "Root", (0.0, 0.0, 1.0), (-0.15, 0.0, 1.1)),
        }

        build_spec = build_armature_spec(classifier_report, build_plan, source_bones)
        plan = compute_vertex_group_remap_plan(
            build_spec["bones"],
            ["pelvis.L", "pelvis.R", "Hip"],
        )

        renames_by_source = {entry["source"]: entry["target"] for entry in plan["renames"]}
        self.assertEqual(renames_by_source.get("pelvis.L"), "pelvis_Left")
        self.assertEqual(renames_by_source.get("pelvis.R"), "pelvis_Right")
        # The synthetic Hip never claims pelvis groups.
        for rename in plan["renames"]:
            self.assertNotEqual(rename["target"], "Hip")
        # Neither pelvis group is reported as unmapped.
        self.assertNotIn("pelvis.L", plan["unmapped_groups"])
        self.assertNotIn("pelvis.R", plan["unmapped_groups"])

    def test_builder_report_includes_preserved_pelvis_pair_field(self):
        """build_builder_report carries the preservation entries through verbatim."""
        classifier_report = _base_classifier_report()
        build_plan = _base_build_plan()
        classifier_report["semantic_mapping"]["Hip"]["action"] = "repair_in_builder"
        classifier_report["semantic_mapping"]["Hip"]["source_bone"] = "pelvis.L"
        classifier_report["semantic_mapping"]["Hip"]["notes"] = ["paired_sided_pelvis_requires_centering"]
        build_plan["mesh_binding"]["meshes"][0]["vertex_groups"] = ["pelvis.L", "pelvis.R"]
        build_plan["mesh_binding"]["meshes"][0]["vertex_group_stats"] = {
            "non_empty_group_count": 2,
            "per_group": [
                {"name": "pelvis.L", "weighted_vertex_count": 12},
                {"name": "pelvis.R", "weighted_vertex_count": 11},
            ],
        }

        source_bones = {
            "pelvis.L": _bone("pelvis.L", "Root", (0.0, 0.0, 1.0), (0.15, 0.0, 1.1)),
            "pelvis.R": _bone("pelvis.R", "Root", (0.0, 0.0, 1.0), (-0.15, 0.0, 1.1)),
        }

        build_spec = build_armature_spec(classifier_report, build_plan, source_bones)
        execution_result = {
            "generated_collection_name": build_spec["generated_collection_name"],
            "group_root_name": build_spec["group_root_name"],
            "generated_armature_name": build_spec["generated_armature_name"],
            "collection_action": "create",
            "duplicated_meshes": [],
            "skipped_meshes": [],
            "mesh_warnings": [],
        }
        report = build_builder_report(build_spec, execution_result)

        self.assertEqual(report["preserved_pelvis_pair"], build_spec["preserved_pelvis_pair"])
        # built_core_targets covers only the 28 ASAM core bones - preserved extras are not counted.
        self.assertEqual(set(report["built_core_targets"]), set(CORE_TARGETS))
        # The spec-style names are excluded from the core count, not the source names.
        self.assertNotIn("pelvis_Left", report["built_core_targets"])
        self.assertNotIn("pelvis_Right", report["built_core_targets"])

    def test_unweighted_paired_pelvis_is_not_preserved(self):
        """When neither pelvis source bone carries skin weights, no extension
        bones are added — the synthetic Hip stands alone."""
        classifier_report = _base_classifier_report()
        build_plan = _base_build_plan()
        classifier_report["semantic_mapping"]["Hip"]["action"] = "repair_in_builder"
        classifier_report["semantic_mapping"]["Hip"]["source_bone"] = "pelvis.L"
        classifier_report["semantic_mapping"]["Hip"]["notes"] = ["paired_sided_pelvis_requires_centering"]
        build_plan["mesh_binding"]["meshes"][0]["vertex_group_stats"] = {
            "non_empty_group_count": 0,
            "per_group": [
                {"name": "pelvis.L", "weighted_vertex_count": 0},
                {"name": "pelvis.R", "weighted_vertex_count": 0},
            ],
        }

        source_bones = {
            "pelvis.L": _bone("pelvis.L", "Root", (0.0, 0.0, 1.0), (0.15, 0.0, 1.1)),
            "pelvis.R": _bone("pelvis.R", "Root", (0.0, 0.0, 1.0), (-0.15, 0.0, 1.1)),
        }

        build_spec = build_armature_spec(classifier_report, build_plan, source_bones)

        self.assertEqual(build_spec["preserved_pelvis_pair"], [])
        bone_names = [bone["name"] for bone in build_spec["bones"]]
        self.assertNotIn("pelvis_Left", bone_names)
        self.assertNotIn("pelvis_Right", bone_names)
        self.assertNotIn("pelvis.L", bone_names)
        self.assertNotIn("pelvis.R", bone_names)
        # Hip itself is still in the build spec — only the extras are gated.
        self.assertEqual(_spec_bone(build_spec, "Hip")["geometry_source"], "centered_pelvis_pair")

    def test_one_sided_weighted_paired_pelvis_preserves_only_weighted_side(self):
        """If only one pelvis side carries skin weight, only that side is preserved.
        The other side is dead weight in the source rig and would not deform anything."""
        classifier_report = _base_classifier_report()
        build_plan = _base_build_plan()
        classifier_report["semantic_mapping"]["Hip"]["action"] = "repair_in_builder"
        classifier_report["semantic_mapping"]["Hip"]["source_bone"] = "pelvis.L"
        classifier_report["semantic_mapping"]["Hip"]["notes"] = ["paired_sided_pelvis_requires_centering"]
        build_plan["mesh_binding"]["meshes"][0]["vertex_group_stats"] = {
            "non_empty_group_count": 1,
            "per_group": [
                {"name": "pelvis.L", "weighted_vertex_count": 12},
                {"name": "pelvis.R", "weighted_vertex_count": 0},
            ],
        }

        source_bones = {
            "pelvis.L": _bone("pelvis.L", "Root", (0.0, 0.0, 1.0), (0.15, 0.0, 1.1)),
            "pelvis.R": _bone("pelvis.R", "Root", (0.0, 0.0, 1.0), (-0.15, 0.0, 1.1)),
        }

        build_spec = build_armature_spec(classifier_report, build_plan, source_bones)

        self.assertEqual(
            build_spec["preserved_pelvis_pair"],
            [{"source_bone_name": "pelvis.L", "generated_bone_name": "pelvis_Left", "parent": "Hip"}],
        )
        bone_names = [bone["name"] for bone in build_spec["bones"]]
        self.assertIn("pelvis_Left", bone_names)
        self.assertNotIn("pelvis_Right", bone_names)

    def test_spec_style_side_suffix_rewrites_common_conventions(self):
        from asam_human_builder.geometry_resolution import _spec_style_side_suffix
        cases = [
            ("DEF-pelvis.L", "DEF-pelvis_Left"),
            ("DEF-pelvis.R", "DEF-pelvis_Right"),
            ("pelvis.L", "pelvis_Left"),
            ("pelvis.R", "pelvis_Right"),
            ("mixamorig_LeftUpLeg_L", "mixamorig_LeftUpLeg_Left"),
            ("hip_R", "hip_Right"),
            ("Pelvis-L", "Pelvis_Left"),
            ("Pelvis-R", "Pelvis_Right"),
            # already spec-style: leave unchanged
            ("DEF-pelvis_Left", "DEF-pelvis_Left"),
            ("DEF-pelvis_Right", "DEF-pelvis_Right"),
            # no recognizable side suffix: leave unchanged
            ("DEF-spine", "DEF-spine"),
            ("Hip", "Hip"),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(_spec_style_side_suffix(source), expected)

    def test_bone_has_skin_weight_reads_mesh_binding_stats(self):
        from asam_human_builder.builder import _bone_has_skin_weight
        mesh_binding = {
            "armature_object_name": "Rig",
            "meshes": [
                {
                    "mesh_name": "BodyMesh",
                    "vertex_group_stats": {
                        "non_empty_group_count": 1,
                        "per_group": [
                            {"name": "pelvis.L", "weighted_vertex_count": 12},
                            {"name": "pelvis.R", "weighted_vertex_count": 0},
                        ],
                    },
                },
                {
                    "mesh_name": "Hair",
                    "vertex_group_stats": {
                        "non_empty_group_count": 0,
                        "per_group": [
                            {"name": "pelvis.R", "weighted_vertex_count": 0},
                        ],
                    },
                },
            ],
        }
        # Weighted on at least one mesh: True.
        self.assertTrue(_bone_has_skin_weight("pelvis.L", mesh_binding))
        # Zero weight across all meshes that mention it: False.
        self.assertFalse(_bone_has_skin_weight("pelvis.R", mesh_binding))
        # Not present in any mesh's vertex_group_stats: False.
        self.assertFalse(_bone_has_skin_weight("nonexistent", mesh_binding))
        # Empty mesh_binding: False (and must not raise).
        self.assertFalse(_bone_has_skin_weight("pelvis.L", {}))
        self.assertFalse(_bone_has_skin_weight("pelvis.L", {"meshes": []}))

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

    def test_to_grp_root_local_subtracts_origin_componentwise(self):
        from asam_human_builder.builder import _to_grp_root_local
        local = _to_grp_root_local((0.5, -0.2, 1.0), (0.1, -0.2, 0.05))
        _assert_vec_almost_equal(self, local, [0.4, 0.0, 0.95])

    def test_validate_builder_inputs_rejects_malformed_grp_root_local_origin(self):
        report = _base_classifier_report()
        plan = _base_build_plan()
        plan["root_resolutions"][0]["grp_root_local_origin"] = [0.0, 0.0]

        with self.assertRaises(ValueError) as ctx:
            validate_builder_inputs(report, plan)
        self.assertIn("length-3", str(ctx.exception))

        plan["root_resolutions"][0]["grp_root_local_origin"] = [0.0, "oops", 0.0]
        with self.assertRaises(ValueError) as ctx:
            validate_builder_inputs(report, plan)
        self.assertIn("numeric", str(ctx.exception))

    def test_validate_builder_inputs_requires_mesh_binding(self):
        report = _base_classifier_report()
        plan = _base_build_plan()
        del plan["mesh_binding"]

        with self.assertRaisesRegex(ValueError, "build_plan is missing required fields: mesh_binding"):
            validate_builder_inputs(report, plan)

    def test_validate_builder_inputs_rejects_mesh_binding_armature_mismatch(self):
        report = _base_classifier_report(armature_name="Rig")
        plan = _base_build_plan(armature_name="Rig")
        plan["mesh_binding"]["armature_object_name"] = "OtherRig"

        with self.assertRaisesRegex(
            ValueError,
            "mesh_binding.armature_object_name must match recommended_primary_armature",
        ):
            validate_builder_inputs(report, plan)

    def test_build_armature_spec_copies_mesh_binding(self):
        report = _base_classifier_report()
        plan = _base_build_plan()
        source_bones = {"Root": _bone("Root", None, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))}

        spec = build_armature_spec(report, plan, source_bones)

        self.assertEqual(spec["mesh_binding"], plan["mesh_binding"])
        self.assertIsNot(spec["mesh_binding"], plan["mesh_binding"])

    def test_build_builder_report_includes_mesh_duplication_fields(self):
        build_spec = {
            "asset_name": "SyntheticAsset",
            "source_armature_name": "Rig",
            "bones": [],
            "extras_preserved": [],
            "warnings": [],
        }
        execution_result = {
            "generated_collection_name": "ASAM_SyntheticAsset",
            "group_root_name": "Grp_Root",
            "generated_armature_name": "Armature_SyntheticAsset",
            "collection_action": "create",
            "duplicated_meshes": [
                {
                    "source_mesh_name": "BodyMesh",
                    "generated_mesh_name": "ASAM_BodyMesh",
                    "armature_link": "modifier",
                    "retargeted_armature_modifiers": ["Armature"],
                }
            ],
            "skipped_meshes": [],
            "mesh_warnings": [],
        }

        report = build_builder_report(build_spec, execution_result)

        self.assertEqual(report["collection_action"], "create")
        self.assertEqual(report["duplicated_meshes"], execution_result["duplicated_meshes"])
        self.assertEqual(report["skipped_meshes"], [])
        self.assertEqual(report["mesh_warnings"], [])

    def test_build_armature_spec_rebases_bones_to_grp_root_local(self):
        """In create_new_root mode, bones are stored in Grp_Root-local coords
        (= source-world coords minus grp_root_local_origin)."""
        report = _base_classifier_report()
        plan = _base_build_plan()
        plan["root_resolutions"][0].update({
            "mode": "create_new_root",
            "source_bone": "Root",
            "grp_root_local_origin": [0.1, 0.0, 0.0],
        })
        report["semantic_mapping"]["Hip"].update({
            "source_bone": "Hip",
            "action": "direct_map",
            "confidence": 1.0,
        })
        source_bones = {
            "Root": _bone("Root", None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.5)),
            "Hip": _bone("Hip", "Root", (0.0, 0.0, 0.9), (0.0, 0.0, 1.0)),
        }

        spec = build_armature_spec(report, plan, source_bones)

        # Hip is direct-mapped from source; its source head was (0, 0, 0.9),
        # so the Grp_Root-local head is (-0.1, 0, 0.9).
        hip_bone = _spec_bone(spec, "Hip")
        self.assertAlmostEqual(hip_bone["head"][0], -0.1, places=9)
        self.assertAlmostEqual(hip_bone["head"][2], 0.9, places=9)
        self.assertEqual(spec["grp_root_local_origin"], [0.1, 0.0, 0.0])

    def test_build_armature_spec_identity_when_grp_root_local_origin_zero(self):
        report = _base_classifier_report()
        plan = _base_build_plan()
        plan["root_resolutions"][0].update({
            "mode": "create_new_root",
            "source_bone": "Root",
            "grp_root_local_origin": [0.0, 0.0, 0.0],
        })
        report["semantic_mapping"]["Hip"].update({
            "source_bone": "Hip",
            "action": "direct_map",
            "confidence": 1.0,
        })
        source_bones = {
            "Root": _bone("Root", None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.5)),
            "Hip": _bone("Hip", "Root", (0.0, 0.0, 0.9), (0.0, 0.0, 1.0)),
        }

        spec = build_armature_spec(report, plan, source_bones)

        hip_bone = _spec_bone(spec, "Hip")
        self.assertEqual(hip_bone["head"], [0.0, 0.0, 0.9])
        self.assertEqual(hip_bone["tail"], [0.0, 0.0, 1.0])

    def test_reuse_existing_root_keeps_source_position_in_grp_root_local(self):
        """Under the new model, a reused source root keeps its source position;
        the offset between source root.head and bbox_ground_center is preserved
        as a (negative) translation in Grp_Root-local space."""
        report = _base_classifier_report()
        plan = _base_build_plan()
        plan["root_resolutions"][0].update({
            "mode": "reuse_existing_root",
            "source_bone": "Root",
            "grp_root_local_origin": [0.01, 0.0, 0.0],
        })
        report["semantic_mapping"]["Root"].update({
            "source_bone": "Root",
            "action": "direct_map",
            "confidence": 1.0,
        })
        source_bones = {
            "Root": _bone("Root", None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.9)),
        }

        spec = build_armature_spec(report, plan, source_bones)

        root_bone = _spec_bone(spec, "Root")
        # source (0, 0, 0) - grp_root_local_origin (0.01, 0, 0) = (-0.01, 0, 0).
        self.assertAlmostEqual(root_bone["head"][0], -0.01, places=9)
        self.assertAlmostEqual(root_bone["head"][1], 0.0, places=9)
        self.assertAlmostEqual(root_bone["head"][2], 0.0, places=9)

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


class CrowdNamingTests(unittest.TestCase):
    def test_apply_character_naming_overrides_generated_names(self):
        from asam_human_builder.builder import apply_character_naming
        spec = {
            "generated_collection_name": "ASAM_crowd",
            "group_root_name": "Grp_Root",
            "generated_armature_name": "Armature_crowd",
        }
        apply_character_naming(spec, "crowd", "Hero000")
        self.assertEqual(spec["generated_collection_name"], "ASAM_crowd_Hero000")
        self.assertEqual(spec["group_root_name"], "Grp_Root_Hero000")
        self.assertEqual(spec["generated_armature_name"], "Armature_crowd_Hero000")

    def test_wrapper_collection_name(self):
        from asam_human_builder.builder import wrapper_collection_name
        self.assertEqual(wrapper_collection_name("crowd"), "ASAM_crowd")

    def test_resolve_default_asset_dir_uses_blend_name(self):
        resolved = resolve_default_asset_dir(
            "C:/assets/openmatexamplehuman.blend",
            REPO_ROOT / "src" / "asam_human_builder",
        )

        self.assertEqual(
            resolved,
            (REPO_ROOT / "output" / "openmatexamplehuman").resolve(),
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

    def test_fake_bpy_source_mesh_supports_copy_materials_modifiers_and_props(self):
        bpy_module = _FakeBpy()
        source_armature = bpy_module.add_source_armature("Rig")
        source_mesh = bpy_module.add_source_mesh("BodyMesh", source_armature)
        source_mesh.vertex_groups = ["Hip"]
        source_mesh.matrix_world = ("world", "BodyMesh")
        source_mesh["generated"] = False
        source_mesh.data["source"] = "inspector"

        copied_data = source_mesh.data.copy()
        copied_object = source_mesh.copy()

        self.assertEqual(source_mesh.type, "MESH")
        self.assertEqual(source_mesh.data.users, 1)
        self.assertEqual(source_mesh.material_slots[0].material.name, "BodyMat")
        self.assertEqual(source_mesh.modifiers[0].type, "ARMATURE")
        self.assertIs(source_mesh.modifiers[0].object, source_armature)
        self.assertEqual(copied_data.name, "BodyMeshData_copy")
        self.assertEqual(copied_data["source"], "inspector")
        self.assertEqual(copied_data.users, 0)
        self.assertEqual(copied_object.name, "BodyMesh_copy")
        self.assertIs(copied_object.data, source_mesh.data)
        self.assertIsNot(copied_object.modifiers[0], source_mesh.modifiers[0])
        self.assertIs(copied_object.modifiers[0].object, source_armature)
        self.assertEqual(copied_object.vertex_groups, ["Hip"])
        self.assertIsNot(copied_object.material_slots[0], source_mesh.material_slots[0])
        self.assertEqual(copied_object.material_slots[0].material.name, "BodyMat")
        self.assertEqual(copied_object.matrix_world, source_mesh.matrix_world)
        self.assertEqual(copied_object["generated"], False)

    def test_fake_collection_link_registers_copied_mesh_and_tracks_users(self):
        bpy_module = _FakeBpy()
        source_mesh = bpy_module.add_source_mesh("BodyMesh")
        copied_mesh = source_mesh.copy()
        copied_mesh.name = "ASAM_BodyMesh"
        copied_mesh.data = source_mesh.data.copy()
        collection = bpy_module.data.collections.new("ASAM_SyntheticAsset")

        collection.objects.link(copied_mesh)
        collection.objects.link(copied_mesh)

        self.assertIs(bpy_module.data.objects.get("ASAM_BodyMesh"), copied_mesh)
        self.assertEqual(collection.all_objects, [copied_mesh])
        self.assertEqual(source_mesh.data.users, 1)
        self.assertEqual(copied_mesh.data.users, 1)

        bpy_module.data.objects.remove(copied_mesh)

        self.assertIsNone(bpy_module.data.objects.get("ASAM_BodyMesh"))
        self.assertEqual(copied_mesh.data.users, 0)

    def test_fake_object_store_register_is_identity_aware_and_collision_safe(self):
        bpy_module = _FakeBpy()
        source_mesh = bpy_module.add_source_mesh("BodyMesh")
        copied_mesh = source_mesh.copy()
        copied_mesh.name = "ASAM_BodyMesh"
        copied_mesh.data = source_mesh.data.copy()
        collection = bpy_module.data.collections.new("ASAM_SyntheticAsset")

        collection.objects.link(copied_mesh)
        registered_users = copied_mesh.data.users
        bpy_module.data.objects.register(copied_mesh)

        self.assertTrue(bpy_module.data.objects.contains(copied_mesh))
        self.assertEqual(copied_mesh.data.users, registered_users)

        colliding_mesh = source_mesh.copy()
        colliding_mesh.name = copied_mesh.name
        with self.assertRaises((AssertionError, ValueError)):
            bpy_module.data.objects.register(colliding_mesh)

    def test_fake_mesh_and_object_copy_deepcopy_nested_custom_props(self):
        bpy_module = _FakeBpy()
        source_mesh = bpy_module.add_source_mesh("BodyMesh")
        source_mesh["metadata"] = {"tags": ["source"], "nested": {"side": "left"}}
        source_mesh.data["metadata"] = {"weights": [1.0], "nested": {"group": "Hip"}}

        copied_object = source_mesh.copy()
        copied_data = source_mesh.data.copy()
        copied_object["metadata"]["tags"].append("copy")
        copied_object["metadata"]["nested"]["side"] = "right"
        copied_data["metadata"]["weights"].append(0.5)
        copied_data["metadata"]["nested"]["group"] = "Spine"

        self.assertEqual(source_mesh["metadata"], {"tags": ["source"], "nested": {"side": "left"}})
        self.assertEqual(source_mesh.data["metadata"], {"weights": [1.0], "nested": {"group": "Hip"}})

    def test_blender_builder_duplicates_mesh_and_retargets_armature_modifier(self):
        bpy_module = _FakeBpy()
        source_armature = bpy_module.add_source_armature("Rig")
        source_mesh = bpy_module.add_source_mesh("BodyMesh", source_armature, armature_modifier=True)
        build_spec = {
            "asset_name": "SyntheticAsset",
            "source_armature_name": "Rig",
            "generated_collection_name": "ASAM_SyntheticAsset",
            "group_root_name": "Grp_Root",
            "generated_armature_name": "Armature_SyntheticAsset",
            "mesh_binding": _mesh_binding("Rig"),
            "bones": [],
        }

        execution_result = build_armature_in_blender(build_spec, bpy_module)

        generated_collection = bpy_module.data.collections.get("ASAM_SyntheticAsset")
        generated_mesh = bpy_module.data.objects.get("ASAM_BodyMesh")
        generated_armature = bpy_module.data.objects.get(execution_result["generated_armature_name"])
        self.assertIsNotNone(generated_mesh)
        self.assertIn(generated_mesh, generated_collection.all_objects)
        # Meshes must be children of the generated armature (ASAM hierarchy), not group_root.
        self.assertIs(generated_mesh.parent, generated_armature)
        self.assertTrue(generated_mesh.get(GENERATED_MARKER_KEY))
        self.assertEqual(generated_mesh.get(GENERATED_ASSET_KEY), "SyntheticAsset")
        self.assertTrue(generated_mesh.data.get(GENERATED_MARKER_KEY))
        self.assertEqual(generated_mesh.data.get(GENERATED_ASSET_KEY), "SyntheticAsset")
        self.assertIs(generated_mesh.modifiers[0].object, generated_armature)
        self.assertIs(source_mesh.modifiers[0].object, source_armature)
        self.assertEqual(generated_mesh.matrix_world, source_mesh.matrix_world)
        self.assertIsNot(generated_mesh.data, source_mesh.data)
        self.assertEqual(execution_result["duplicated_meshes"][0]["source_mesh_name"], "BodyMesh")
        self.assertEqual(execution_result["duplicated_meshes"][0]["generated_mesh_name"], "ASAM_BodyMesh")
        self.assertEqual(execution_result["duplicated_meshes"][0]["retargeted_armature_modifiers"], ["Armature"])

    def test_blender_builder_duplicates_parent_only_mesh_with_warning(self):
        bpy_module = _FakeBpy()
        source_armature = bpy_module.add_source_armature("Rig")
        bpy_module.add_source_mesh("BodyMesh", source_armature, armature_modifier=False)
        binding = _mesh_binding("Rig")
        binding["meshes"][0]["armature_link"] = "parent"
        binding["meshes"][0]["modifiers"] = []
        build_spec = {
            "asset_name": "SyntheticAsset",
            "source_armature_name": "Rig",
            "generated_collection_name": "ASAM_SyntheticAsset",
            "group_root_name": "Grp_Root",
            "generated_armature_name": "Armature_SyntheticAsset",
            "mesh_binding": binding,
            "bones": [],
        }

        execution_result = build_armature_in_blender(build_spec, bpy_module)

        generated_mesh = bpy_module.data.objects.get("ASAM_BodyMesh")
        self.assertIsNotNone(generated_mesh)
        self.assertEqual(generated_mesh.modifiers, [])
        self.assertIn("parent_only_no_armature_modifier:BodyMesh", execution_result["mesh_warnings"])

    def test_blender_builder_reports_missing_mesh_binding_entry(self):
        bpy_module = _FakeBpy()
        bpy_module.add_source_armature("Rig")
        build_spec = {
            "asset_name": "SyntheticAsset",
            "source_armature_name": "Rig",
            "generated_collection_name": "ASAM_SyntheticAsset",
            "group_root_name": "Grp_Root",
            "generated_armature_name": "Armature_SyntheticAsset",
            "mesh_binding": _mesh_binding("Rig"),
            "bones": [],
        }

        execution_result = build_armature_in_blender(build_spec, bpy_module)

        self.assertEqual(
            execution_result["skipped_meshes"],
            [{"mesh_name": "BodyMesh", "reason": "source_mesh_missing"}],
        )
        self.assertEqual(execution_result["duplicated_meshes"], [])

    def test_blender_builder_preserves_armature_modifiers_targeting_other_objects(self):
        bpy_module = _FakeBpy()
        source_armature = bpy_module.add_source_armature("Rig")
        other_armature = bpy_module.add_source_armature("OtherRig")
        source_mesh = bpy_module.add_source_mesh("BodyMesh", source_armature, armature_modifier=True)
        source_mesh.modifiers.append(_FakeModifier("OtherArmature", "ARMATURE", other_armature))
        build_spec = {
            "asset_name": "SyntheticAsset",
            "source_armature_name": "Rig",
            "generated_collection_name": "ASAM_SyntheticAsset",
            "group_root_name": "Grp_Root",
            "generated_armature_name": "Armature_SyntheticAsset",
            "mesh_binding": _mesh_binding("Rig"),
            "bones": [],
        }

        execution_result = build_armature_in_blender(build_spec, bpy_module)

        generated_mesh = bpy_module.data.objects.get("ASAM_BodyMesh")
        generated_armature = bpy_module.data.objects.get(execution_result["generated_armature_name"])
        self.assertIs(generated_mesh.modifiers[0].object, generated_armature)
        self.assertIs(generated_mesh.modifiers[1].object, other_armature)
        self.assertIs(source_mesh.modifiers[0].object, source_armature)
        self.assertIs(source_mesh.modifiers[1].object, other_armature)
        self.assertEqual(
            execution_result["duplicated_meshes"][0]["retargeted_armature_modifiers"],
            ["Armature"],
        )

    def test_blender_builder_parents_mesh_to_armature_not_group_root(self):
        """Generated meshes must be one level below the armature in the ASAM hierarchy."""
        bpy_module = _FakeBpy()
        source_armature = bpy_module.add_source_armature("Rig")
        bpy_module.add_source_mesh("BodyMesh", source_armature, armature_modifier=True)
        build_spec = {
            "asset_name": "SyntheticAsset",
            "source_armature_name": "Rig",
            "generated_collection_name": "ASAM_SyntheticAsset",
            "group_root_name": "Grp_Root",
            "generated_armature_name": "Armature_SyntheticAsset",
            "mesh_binding": _mesh_binding("Rig"),
            "bones": [],
        }

        execution_result = build_armature_in_blender(build_spec, bpy_module)

        generated_mesh = bpy_module.data.objects.get("ASAM_BodyMesh")
        generated_armature = bpy_module.data.objects.get(execution_result["generated_armature_name"])
        group_root = bpy_module.data.objects.get(execution_result["group_root_name"])
        self.assertIsNotNone(generated_mesh)
        self.assertIsNotNone(generated_armature)
        # Mesh parent must be the armature, not the group root.
        self.assertIs(generated_mesh.parent, generated_armature)
        self.assertIsNot(generated_mesh.parent, group_root)
        # The armature itself must still be parented to group_root.
        self.assertIs(generated_armature.parent, group_root)

    def test_blender_builder_preserves_duplicated_mesh_world_matrix(self):
        """The duplicated mesh keeps the source mesh's matrix_world unchanged.

        Grp_Root now carries the bbox-ground-center anchor as its own world
        location; Blender's parent inverse handles the rebase into Grp_Root-local
        space when the generated armature (a child of Grp_Root) becomes the
        duplicated mesh's parent. The builder no longer translates meshes.
        """
        bpy_module = _FakeBpy()
        source_armature = bpy_module.add_source_armature("Rig")
        source_mesh = bpy_module.add_source_mesh("BodyMesh", source_armature, armature_modifier=True)
        source_mesh.matrix_world = ("world", "BodyMesh_original")
        build_spec = {
            "asset_name": "SyntheticAsset",
            "source_armature_name": "Rig",
            "generated_collection_name": "ASAM_SyntheticAsset",
            "group_root_name": "Grp_Root",
            "generated_armature_name": "Armature_SyntheticAsset",
            "grp_root_local_origin": [0.1, 0.0, 0.05],
            "mesh_binding": _mesh_binding("Rig"),
            "bones": [],
        }

        build_armature_in_blender(build_spec, bpy_module)

        generated_mesh = bpy_module.data.objects.get("ASAM_BodyMesh")
        self.assertIsNotNone(generated_mesh)
        self.assertEqual(generated_mesh.matrix_world, ("world", "BodyMesh_original"))

    def test_blender_builder_renames_vertex_groups_to_asam_targets(self):
        """Duplicated mesh's Rigify-named vertex groups must be renamed to ASAM targets."""
        bpy_module = _FakeBpy()
        source_armature = bpy_module.add_source_armature("Rig")
        source_mesh = bpy_module.add_source_mesh("BodyMesh", source_armature, armature_modifier=True)
        source_mesh.vertex_groups = ["DEF-forearm.L", "DEF-hand.L", "DEF-upper_arm.L"]
        build_spec = {
            "asset_name": "SyntheticAsset",
            "source_armature_name": "Rig",
            "generated_collection_name": "ASAM_SyntheticAsset",
            "group_root_name": "Grp_Root",
            "generated_armature_name": "Armature_SyntheticAsset",
            "mesh_binding": _mesh_binding("Rig"),
            "bones": [
                _remap_bone("Upper_Arm_Left", "DEF-upper_arm.L"),
                _remap_bone("Lower_Arm_Left", "DEF-forearm.L"),
                _remap_bone("Hand_Left", "DEF-hand.L"),
            ],
        }

        execution_result = build_armature_in_blender(build_spec, bpy_module)

        generated_mesh = bpy_module.data.objects.get("ASAM_BodyMesh")
        self.assertIsNotNone(generated_mesh)
        self.assertEqual(
            sorted(generated_mesh.vertex_groups),
            ["Hand_Left", "Lower_Arm_Left", "Upper_Arm_Left"],
        )
        # Source mesh unchanged.
        self.assertEqual(
            sorted(source_mesh.vertex_groups),
            ["DEF-forearm.L", "DEF-hand.L", "DEF-upper_arm.L"],
        )

        remap = execution_result["duplicated_meshes"][0]["vertex_group_remap"]
        self.assertEqual(
            remap["renamed"],
            [
                {"source": "DEF-forearm.L", "target": "Lower_Arm_Left"},
                {"source": "DEF-hand.L", "target": "Hand_Left"},
                {"source": "DEF-upper_arm.L", "target": "Upper_Arm_Left"},
            ],
        )
        self.assertEqual(remap["unmapped"], [])
        self.assertEqual(remap["missing_source_groups"], [])
        self.assertEqual(remap["collisions"], [])
        self.assertNotIn("unmapped_vertex_groups:BodyMesh", execution_result["mesh_warnings"])

    def test_blender_builder_reports_unmapped_vertex_groups(self):
        """Vertex groups with no ASAM mapping must surface as a mesh_warning."""
        bpy_module = _FakeBpy()
        source_armature = bpy_module.add_source_armature("Rig")
        source_mesh = bpy_module.add_source_mesh("BodyMesh", source_armature, armature_modifier=True)
        source_mesh.vertex_groups = ["DEF-hand.L", "MCH-helper_bone"]
        build_spec = {
            "asset_name": "SyntheticAsset",
            "source_armature_name": "Rig",
            "generated_collection_name": "ASAM_SyntheticAsset",
            "group_root_name": "Grp_Root",
            "generated_armature_name": "Armature_SyntheticAsset",
            "mesh_binding": _mesh_binding("Rig"),
            "bones": [_remap_bone("Hand_Left", "DEF-hand.L")],
        }

        execution_result = build_armature_in_blender(build_spec, bpy_module)

        generated_mesh = bpy_module.data.objects.get("ASAM_BodyMesh")
        self.assertIn("Hand_Left", generated_mesh.vertex_groups)
        self.assertIn("MCH-helper_bone", generated_mesh.vertex_groups)  # left alone
        self.assertIn("unmapped_vertex_groups:BodyMesh", execution_result["mesh_warnings"])

        remap = execution_result["duplicated_meshes"][0]["vertex_group_remap"]
        self.assertEqual(remap["unmapped"], ["MCH-helper_bone"])

    def test_blender_builder_leaves_pre_compliant_groups_alone(self):
        """When source vertex groups already match ASAM names, no rename is needed."""
        bpy_module = _FakeBpy()
        source_armature = bpy_module.add_source_armature("Rig")
        source_mesh = bpy_module.add_source_mesh("BodyMesh", source_armature, armature_modifier=True)
        source_mesh.vertex_groups = ["Hip", "Upper_Arm_Left"]
        build_spec = {
            "asset_name": "SyntheticAsset",
            "source_armature_name": "Rig",
            "generated_collection_name": "ASAM_SyntheticAsset",
            "group_root_name": "Grp_Root",
            "generated_armature_name": "Armature_SyntheticAsset",
            "mesh_binding": _mesh_binding("Rig"),
            "bones": [
                _remap_bone("Hip", "Hip"),
                _remap_bone("Upper_Arm_Left", "Upper_Arm_Left"),
            ],
        }

        execution_result = build_armature_in_blender(build_spec, bpy_module)

        generated_mesh = bpy_module.data.objects.get("ASAM_BodyMesh")
        self.assertEqual(sorted(generated_mesh.vertex_groups), ["Hip", "Upper_Arm_Left"])
        remap = execution_result["duplicated_meshes"][0]["vertex_group_remap"]
        self.assertEqual(remap["renamed"], [])
        self.assertEqual(remap["unmapped"], [])
        self.assertNotIn("unmapped_vertex_groups:BodyMesh", execution_result["mesh_warnings"])

    def test_blender_builder_skips_offset_when_zero(self):
        """A zero source_translation_offset must not modify the mesh matrix_world."""
        bpy_module = _FakeBpy()
        source_armature = bpy_module.add_source_armature("Rig")
        source_mesh = bpy_module.add_source_mesh("BodyMesh", source_armature, armature_modifier=True)
        original_matrix = ("world", "BodyMesh_original")
        source_mesh.matrix_world = original_matrix
        build_spec = {
            "asset_name": "SyntheticAsset",
            "source_armature_name": "Rig",
            "generated_collection_name": "ASAM_SyntheticAsset",
            "group_root_name": "Grp_Root",
            "generated_armature_name": "Armature_SyntheticAsset",
            "source_translation_offset": [0.0, 0.0, 0.0],
            "mesh_binding": _mesh_binding("Rig"),
            "bones": [],
        }

        build_armature_in_blender(build_spec, bpy_module)

        generated_mesh = bpy_module.data.objects.get("ASAM_BodyMesh")
        self.assertIsNotNone(generated_mesh)
        # matrix_world must be the unmodified copy from the source.
        self.assertEqual(generated_mesh.matrix_world, original_matrix)

    def test_blender_builder_refuses_to_build_from_previously_generated_armature(self):
        """When the source armature carries GENERATED_MARKER_KEY (i.e. it is itself
        a previously-generated artifact, picked up by the classifier from a stale
        scene), the builder must hard-fail before mutating any collection."""
        bpy_module = _FakeBpy()
        source = bpy_module.add_source_armature("Armature_Test")
        source[GENERATED_MARKER_KEY] = True
        source[GENERATED_ASSET_KEY] = "Test"

        build_spec = {
            "asset_name": "Test",
            "source_armature_name": "Armature_Test",
            "generated_collection_name": "ASAM_Test",
            "group_root_name": "Grp_Root",
            "generated_armature_name": "Armature_Test_v2",
            "source_translation_offset": [0.0, 0.0, 0.0],
            "mesh_binding": _mesh_binding("Armature_Test"),
            "bones": [],
        }

        with self.assertRaises(ValueError) as ctx:
            build_armature_in_blender(build_spec, bpy_module)

        message = str(ctx.exception)
        self.assertIn("previously-generated armature", message)
        self.assertIn("Armature_Test", message)
        # The failure must precede any collection mutation: nothing new in
        # bpy.data.collections.
        self.assertIsNone(bpy_module.data.collections.get("ASAM_Test"))

    def test_purge_previous_generated_artifacts_removes_marked_objects_and_collections(self):
        """A FakeBpy with marked artifacts loses them on purge; unmarked source remains."""
        bpy_module = _FakeBpy()
        source = bpy_module.add_source_armature("rig")

        generated_collection = bpy_module.data.collections.new("ASAM_Test")
        generated_collection[GENERATED_MARKER_KEY] = True
        generated_collection[GENERATED_ASSET_KEY] = "Test"
        bpy_module.context.scene.collection.children.link(generated_collection)

        generated_armature_data = bpy_module.data.armatures.new("Armature_Test")
        generated_armature = bpy_module.data.objects.new("Armature_Test", generated_armature_data)
        generated_armature[GENERATED_MARKER_KEY] = True
        generated_armature[GENERATED_ASSET_KEY] = "Test"
        generated_collection.objects.link(generated_armature)

        generated_mesh_data = bpy_module.data.meshes.new("ASAM_BodyMeshData")
        generated_mesh = bpy_module.data.objects.new("ASAM_BodyMesh", generated_mesh_data)
        generated_mesh[GENERATED_MARKER_KEY] = True
        generated_mesh[GENERATED_ASSET_KEY] = "Test"
        generated_collection.objects.link(generated_mesh)

        removed = purge_previous_generated_artifacts(bpy_module)

        # 2 generated objects + 1 generated collection = 3 removed entries.
        self.assertEqual(removed, 3)
        # Unmarked source survives.
        self.assertIs(bpy_module.data.objects.get("rig"), source)
        # Generated objects are gone.
        self.assertIsNone(bpy_module.data.objects.get("Armature_Test"))
        self.assertIsNone(bpy_module.data.objects.get("ASAM_BodyMesh"))
        # Their data blocks are orphan-cleaned.
        self.assertIsNone(bpy_module.data.armatures.get("Armature_Test"))
        self.assertIsNone(bpy_module.data.meshes.get("ASAM_BodyMeshData"))
        # Generated collection is gone.
        self.assertIsNone(bpy_module.data.collections.get("ASAM_Test"))

    def test_purge_previous_generated_artifacts_is_idempotent_on_empty_scene(self):
        """An empty FakeBpy is a valid input; purge returns 0 and raises nothing."""
        bpy_module = _FakeBpy()
        removed = purge_previous_generated_artifacts(bpy_module)
        self.assertEqual(removed, 0)

    def test_purge_previous_generated_artifacts_leaves_unmarked_collections_alone(self):
        """Collections without GENERATED_MARKER_KEY survive the purge."""
        bpy_module = _FakeBpy()

        user_collection = bpy_module.data.collections.new("UserCollection")
        bpy_module.context.scene.collection.children.link(user_collection)

        generated_collection = bpy_module.data.collections.new("ASAM_Test")
        generated_collection[GENERATED_MARKER_KEY] = True
        generated_collection[GENERATED_ASSET_KEY] = "Test"
        bpy_module.context.scene.collection.children.link(generated_collection)

        removed = purge_previous_generated_artifacts(bpy_module)

        self.assertEqual(removed, 1)
        self.assertIsNotNone(bpy_module.data.collections.get("UserCollection"))
        self.assertIsNone(bpy_module.data.collections.get("ASAM_Test"))

    def test_deferred_targets_list_is_empty(self):
        """DEFERRED_TARGETS must be empty — all ASAM bones are now promoted to CORE_TARGETS."""
        self.assertEqual(
            DEFERRED_TARGETS,
            [],
            "DEFERRED_TARGETS should be empty once all ASAM §7.3.3 bones are in CORE_TARGETS.",
        )

    def test_full_asam_spec_bones_present_in_core_targets(self):
        """All 28 normative ASAM OpenMATERIAL 3D §7.3.3 bones must be in CORE_TARGETS."""
        required = {
            # Spine / head
            "Root", "Hip", "Lower_Spine", "Upper_Spine", "Neck", "Head",
            # Eyes (§7.3.3.3.10-11)
            "Eye_Left", "Eye_Right",
            # Left arm chain (§7.3.3.3.12-17)
            "Shoulder_Left", "Upper_Arm_Left", "Lower_Arm_Left", "Hand_Left",
            "Full_Thumb_Left", "Full_Fingers_Left",
            # Right arm chain (§7.3.3.3.18-23)
            "Shoulder_Right", "Upper_Arm_Right", "Lower_Arm_Right", "Hand_Right",
            "Full_Thumb_Right", "Full_Fingers_Right",
            # Left leg chain (§7.3.3.3.24-27)
            "Upper_Leg_Left", "Lower_Leg_Left", "Foot_Left", "Full_Toes_Left",
            # Right leg chain (§7.3.3.3.28-31)
            "Upper_Leg_Right", "Lower_Leg_Right", "Foot_Right", "Full_Toes_Right",
        }
        core_set = set(CORE_TARGETS)
        missing = required - core_set
        self.assertEqual(
            missing,
            set(),
            "The following ASAM-normative bones are missing from CORE_TARGETS: {0}".format(sorted(missing)),
        )
        self.assertEqual(len(CORE_TARGETS), 28, "CORE_TARGETS should have exactly 28 bones.")


class CrowdBlenderTests(unittest.TestCase):
    def test_build_armature_in_blender_nests_under_parent_collection(self):
        build_spec = _minimal_crowd_build_spec()
        fake = _fake_with_source_armature(build_spec["source_armature_name"])
        parent = fake.data.collections.new("ASAM_wrapper")
        result = build_armature_in_blender(build_spec, fake, parent_collection=parent)
        child_name = result["generated_collection_name"]
        # Child is linked under the wrapper, NOT the scene root.
        self.assertIsNotNone(parent.children.get(child_name))
        self.assertIsNone(fake.context.scene.collection.children.get(child_name))

    def test_build_armature_in_blender_strips_prefix_then_remaps(self):
        build_spec = _minimal_crowd_build_spec()   # Hip built from source_bone "Pelvis"
        build_spec["mesh_binding"] = {
            "armature_object_name": build_spec["source_armature_name"],
            "meshes": [{"mesh_name": "Body000", "armature_link": "modifier"}],
        }
        fake = _fake_with_source_armature(
            build_spec["source_armature_name"],
            meshes=[("Body000", ["Hero000Pelvis_093"])],
        )
        result = build_armature_in_blender(build_spec, fake, character_prefix="Hero000")
        generated = fake.data.objects.get(result["duplicated_meshes"][0]["generated_mesh_name"])
        # Prefixed group -> stripped ('Pelvis') -> ASAM target ('Hip').
        self.assertIn("Hip", generated.vertex_groups)
        self.assertNotIn("Hero000Pelvis_093", generated.vertex_groups)

    def test_build_crowd_in_blender_creates_wrapper_with_children(self):
        from asam_human_builder.builder import apply_character_naming
        from asam_human_builder.blender_builder import build_crowd_in_blender
        spec_a = apply_character_naming(_minimal_crowd_build_spec(), "crowd", "Hero000")
        spec_b = apply_character_naming(_minimal_crowd_build_spec(), "crowd", "Hero001")
        fake = _fake_with_source_armature("Object_4")   # one source armature, shared
        decomposition = {"source_armature": "Object_4", "character_count": 2,
                         "character_ids": ["Hero000", "Hero001"],
                         "shared_bones": [], "unassigned_meshes": []}
        result = build_crowd_in_blender(
            "crowd", "ASAM_crowd",
            [("Hero000", spec_a), ("Hero001", spec_b)], decomposition, fake,
        )
        wrapper = fake.data.collections.get("ASAM_crowd")
        self.assertIsNotNone(wrapper.children.get("ASAM_crowd_Hero000"))
        self.assertIsNotNone(wrapper.children.get("ASAM_crowd_Hero001"))
        self.assertEqual(len(result["characters"]), 2)
        self.assertEqual(result["failed_characters"], [])

    def test_build_crowd_in_blender_continues_past_failure(self):
        from asam_human_builder.builder import apply_character_naming
        from asam_human_builder.blender_builder import build_crowd_in_blender
        spec_ok = apply_character_naming(_minimal_crowd_build_spec(), "crowd", "Hero000")
        bad_spec = {"asset_name": "crowd"}   # missing keys -> raises inside builder
        fake = _fake_with_source_armature("Object_4")
        decomposition = {"source_armature": "Object_4", "character_count": 2,
                         "character_ids": ["Hero000", "BadOne"],
                         "shared_bones": [], "unassigned_meshes": []}
        result = build_crowd_in_blender(
            "crowd", "ASAM_crowd",
            [("Hero000", spec_ok), ("BadOne", bad_spec)], decomposition, fake,
        )
        self.assertEqual(len(result["characters"]), 1)
        self.assertEqual(result["characters"][0]["character_id"], "Hero000")
        self.assertEqual(len(result["failed_characters"]), 1)
        self.assertEqual(result["failed_characters"][0]["character_id"], "BadOne")


class CrowdDetectionTests(unittest.TestCase):
    def test_is_crowd_plan_true_when_characters_present(self):
        from asam_human_builder.builder import is_crowd_plan
        self.assertTrue(is_crowd_plan({"characters": [{"character_id": "A"}]}))

    def test_is_crowd_plan_false_when_absent_or_empty(self):
        from asam_human_builder.builder import is_crowd_plan
        self.assertFalse(is_crowd_plan({}))
        self.assertFalse(is_crowd_plan({"characters": []}))


def _remap_bone(name: str, source_bone, geometry_source: str = "source_bone") -> dict:
    """
    Build a spec-bone dict suitable for both compute_vertex_group_remap_plan and
    _populate_edit_bones. The geometry fields are placeholders - the remap helper
    only reads name/source_bone/geometry_source.
    """
    return {
        "name": name,
        "parent_bone": None,
        "head": [0.0, 0.0, 0.0],
        "tail": [0.0, 0.0, 1.0],
        "use_connect": False,
        "source_bone": source_bone,
        "geometry_source": geometry_source,
    }


class ComputeVertexGroupRemapPlanTests(unittest.TestCase):
    def test_one_to_one_rigify_names_become_asam(self):
        bones = [
            _remap_bone("Upper_Arm_Left", "DEF-upper_arm.L"),
            _remap_bone("Lower_Arm_Left", "DEF-forearm.L"),
            _remap_bone("Hand_Left", "DEF-hand.L"),
        ]
        groups = ["DEF-forearm.L", "DEF-hand.L", "DEF-upper_arm.L"]

        plan = compute_vertex_group_remap_plan(bones, groups)

        self.assertEqual(
            plan["renames"],
            [
                {"source": "DEF-forearm.L", "target": "Lower_Arm_Left"},
                {"source": "DEF-hand.L", "target": "Hand_Left"},
                {"source": "DEF-upper_arm.L", "target": "Upper_Arm_Left"},
            ],
        )
        self.assertEqual(plan["unmapped_groups"], [])
        self.assertEqual(plan["asam_targets_without_source_group"], [])
        self.assertEqual(plan["name_collisions"], [])

    def test_idempotent_for_already_asam_named_groups(self):
        # openmatexamplehuman case: source_bone equals the target name (pre-compliant).
        bones = [
            _remap_bone("Hip", "Hip"),
            _remap_bone("Upper_Arm_Left", "Upper_Arm_Left"),
        ]
        groups = ["Hip", "Upper_Arm_Left"]

        plan = compute_vertex_group_remap_plan(bones, groups)

        self.assertEqual(plan["renames"], [])
        self.assertEqual(plan["unmapped_groups"], [])
        self.assertEqual(plan["asam_targets_without_source_group"], [])
        self.assertEqual(plan["name_collisions"], [])

    def test_flags_unmapped_groups(self):
        bones = [_remap_bone("Hand_Left", "DEF-hand.L")]
        groups = ["DEF-hand.L", "DEF-stray_extra", "MCH-helper"]

        plan = compute_vertex_group_remap_plan(bones, groups)

        self.assertEqual(plan["renames"], [{"source": "DEF-hand.L", "target": "Hand_Left"}])
        self.assertEqual(plan["unmapped_groups"], ["DEF-stray_extra", "MCH-helper"])

    def test_flags_asam_targets_missing_a_source_group(self):
        bones = [
            _remap_bone("Hand_Left", "DEF-hand.L"),
            _remap_bone("Foot_Left", "DEF-foot.L"),
        ]
        groups = ["DEF-hand.L"]  # Foot_Left's source group is missing

        plan = compute_vertex_group_remap_plan(bones, groups)

        self.assertEqual(plan["renames"], [{"source": "DEF-hand.L", "target": "Hand_Left"}])
        self.assertEqual(plan["asam_targets_without_source_group"], ["Foot_Left"])

    def test_detects_name_collision_when_target_group_already_exists(self):
        bones = [_remap_bone("Hand_Left", "DEF-hand.L")]
        # The mesh already has a vertex group named Hand_Left; renaming DEF-hand.L would clobber it.
        groups = ["DEF-hand.L", "Hand_Left"]

        plan = compute_vertex_group_remap_plan(bones, groups)

        self.assertEqual(plan["renames"], [])
        self.assertEqual(
            plan["name_collisions"],
            [{"source": "DEF-hand.L", "target": "Hand_Left", "existing_group": "Hand_Left"}],
        )
        # The pre-existing ASAM group is not flagged as unmapped — it already matches a bone.
        self.assertNotIn("Hand_Left", plan["unmapped_groups"])

    def test_skips_synthesized_geometry_sources(self):
        # mirrored_opposite, interpolated_chain, extrapolated_parent: source_bone holds a target name,
        # not a real source bone. Must be ignored to avoid renaming groups to those.
        bones = [
            _remap_bone("Lower_Arm_Right", "Lower_Arm_Left", geometry_source="mirrored_opposite"),
            _remap_bone("Lower_Spine", "Hip", geometry_source="interpolated_chain"),
            _remap_bone("Head", "Neck", geometry_source="extrapolated_parent"),
        ]
        groups = ["Lower_Arm_Left", "Hip", "Neck"]

        plan = compute_vertex_group_remap_plan(bones, groups)

        self.assertEqual(plan["renames"], [])
        # Those groups have no bone in the generated armature - they are orphans.
        self.assertEqual(plan["unmapped_groups"], ["Hip", "Lower_Arm_Left", "Neck"])

    def test_centered_pelvis_pair_does_not_claim_pelvis_groups_for_hip(self):
        # When Hip is built via paired-pelvis centering, the synthetic Hip bone must
        # NOT claim either side's vertex group. The preserved extension bones (added
        # by _resolve_preserved_pelvis_pair upstream) take ownership via spec-style
        # renames: DEF-pelvis.L -> DEF-pelvis_Left, DEF-pelvis.R -> DEF-pelvis_Right.
        bones = [
            _remap_bone("Root", "root", geometry_source="root_resolution"),
            _remap_bone("Hip", "DEF-pelvis.L", geometry_source="centered_pelvis_pair"),
            _remap_bone("DEF-pelvis_Left", "DEF-pelvis.L", geometry_source="source_bone"),
            _remap_bone("DEF-pelvis_Right", "DEF-pelvis.R", geometry_source="source_bone"),
        ]
        groups = ["root", "DEF-pelvis.L", "DEF-pelvis.R"]

        plan = compute_vertex_group_remap_plan(bones, groups)

        renames_by_source = {entry["source"]: entry["target"] for entry in plan["renames"]}
        self.assertEqual(renames_by_source.get("DEF-pelvis.L"), "DEF-pelvis_Left")
        self.assertEqual(renames_by_source.get("DEF-pelvis.R"), "DEF-pelvis_Right")
        self.assertEqual(renames_by_source.get("root"), "Root")
        # No rename targets Hip - the synthetic Hip stays clear of pelvis groups.
        for rename in plan["renames"]:
            self.assertNotEqual(rename["target"], "Hip")
        self.assertEqual(plan["unmapped_groups"], [])

    def test_handles_empty_inputs(self):
        self.assertEqual(
            compute_vertex_group_remap_plan([], []),
            {
                "renames": [],
                "unmapped_groups": [],
                "asam_targets_without_source_group": [],
                "name_collisions": [],
            },
        )

    def test_skips_bones_with_no_source_bone(self):
        bones = [
            _remap_bone("Foot_Left", None, geometry_source="placement_fallback"),
            _remap_bone("Hand_Left", "DEF-hand.L"),
        ]
        groups = ["DEF-hand.L"]

        plan = compute_vertex_group_remap_plan(bones, groups)

        self.assertEqual(plan["renames"], [{"source": "DEF-hand.L", "target": "Hand_Left"}])
        self.assertEqual(plan["asam_targets_without_source_group"], [])


class CrowdFlatInputsTests(unittest.TestCase):
    def _inputs(self):
        classifier_report = {
            "asset_summary": {"character_decomposition": {"source_armature": "Object_4"}},
            "characters": [
                {"character_id": "Hero000", "semantic_mapping": {"Hip": {"action": "x"}}},
                {"character_id": "Hero001", "semantic_mapping": {"Hip": {"action": "y"}}},
            ],
        }
        build_plan = {
            "asset_name": "crowd",
            "characters": [
                {"character_id": "Hero000", "root_resolutions": [{"r": 0}],
                 "placement_metadata": {"p": 0}, "proposed_asam_hierarchy": {"h": 0},
                 "extras_preserved": [],
                 "mesh_binding": {"armature_object_name": "Object_4", "meshes": []}},
                {"character_id": "Hero001", "root_resolutions": [{"r": 1}],
                 "placement_metadata": {"p": 1}, "proposed_asam_hierarchy": {"h": 1},
                 "extras_preserved": [],
                 "mesh_binding": {"armature_object_name": "Object_4", "meshes": []}},
            ],
        }
        return classifier_report, build_plan

    def test_build_character_flat_inputs_pairs_by_id(self):
        from asam_human_builder.builder import build_character_flat_inputs
        classifier_report, build_plan = self._inputs()
        flat_report, flat_plan = build_character_flat_inputs(classifier_report, build_plan, "Hero001")
        self.assertEqual(flat_report["recommended_primary_armature"], "Object_4")
        self.assertEqual(flat_report["semantic_mapping"], {"Hip": {"action": "y"}})
        self.assertEqual(flat_plan["asset_name"], "crowd")
        self.assertEqual(flat_plan["recommended_primary_armature"], "Object_4")
        self.assertEqual(flat_plan["root_resolutions"], [{"r": 1}])
        self.assertEqual(flat_plan["mesh_binding"]["armature_object_name"], "Object_4")

    def test_build_character_flat_inputs_unknown_id_raises(self):
        from asam_human_builder.builder import build_character_flat_inputs
        classifier_report, build_plan = self._inputs()
        with self.assertRaises(KeyError):
            build_character_flat_inputs(classifier_report, build_plan, "Nope")


class CrowdSourceBoneSliceTests(unittest.TestCase):
    def test_slice_source_bones_filters_and_strips(self):
        from asam_human_builder.builder import slice_source_bones_for_character
        full_index = {
            "Hero000Pelvis_001": {"name": "Hero000Pelvis_001", "parent": "_rootJoint",
                                  "head": [0.0, 0.0, 1.0], "tail": [0.0, 0.0, 1.2], "length": 0.2},
            "Hero000Spine0_002": {"name": "Hero000Spine0_002", "parent": "Hero000Pelvis_001",
                                  "head": [0.0, 0.0, 1.2], "tail": [0.0, 0.0, 1.4], "length": 0.2},
            "Hero001Pelvis_003": {"name": "Hero001Pelvis_003", "parent": "_rootJoint",
                                  "head": [5.0, 0.0, 1.0], "tail": [5.0, 0.0, 1.2], "length": 0.2},
        }
        sliced = slice_source_bones_for_character(full_index, "Hero000")
        self.assertEqual(set(sliced), {"Pelvis", "Spine0"})
        self.assertEqual(sliced["Pelvis"]["parent"], None)          # parent outside group -> None
        self.assertEqual(sliced["Spine0"]["parent"], "Pelvis")      # parent inside group -> stripped
        self.assertEqual(sliced["Pelvis"]["head"], [0.0, 0.0, 1.0]) # world geometry preserved
        self.assertEqual(sliced["Pelvis"]["name"], "Pelvis")


class CrowdVertexGroupStripTests(unittest.TestCase):
    def test_compute_prefix_strip_renames(self):
        from asam_human_builder.builder import compute_prefix_strip_renames
        renames = compute_prefix_strip_renames(["Hero000Pelvis_093", "Hero000Spine0_094"], "Hero000")
        self.assertEqual(
            sorted(renames, key=lambda r: r["source"]),
            [{"source": "Hero000Pelvis_093", "target": "Pelvis"},
             {"source": "Hero000Spine0_094", "target": "Spine0"}],
        )

    def test_compute_prefix_strip_renames_skips_identity(self):
        from asam_human_builder.builder import compute_prefix_strip_renames
        # A group with no matching prefix strips to itself -> no rename emitted.
        self.assertEqual(compute_prefix_strip_renames(["Pelvis"], "Hero000"), [])


if __name__ == "__main__":
    unittest.main()

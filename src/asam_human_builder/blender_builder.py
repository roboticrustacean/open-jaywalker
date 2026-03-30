"""Blender execution helpers for generated ASAM human armatures."""

from __future__ import annotations

from typing import List, Optional

try:
    from .builder import GENERATED_ASSET_KEY, GENERATED_MARKER_KEY, choose_generated_collection_action
except ImportError:  # pragma: no cover - Blender script path fallback
    from builder import GENERATED_ASSET_KEY, GENERATED_MARKER_KEY, choose_generated_collection_action


def snapshot_existing_collections(bpy_module) -> List[dict]:
    """Capture collection metadata needed for safe rerun decisions."""
    collections = []
    for collection in bpy_module.data.collections:
        collections.append(
            {
                "name": collection.name,
                "generated": bool(collection.get(GENERATED_MARKER_KEY)),
                "asset_name": collection.get(GENERATED_ASSET_KEY),
            }
        )
    return collections


def build_armature_in_blender(build_spec: dict, bpy_module=None) -> dict:
    """Create or rebuild the generated ASAM armature in the open Blender file."""
    bpy_module = bpy_module or _require_bpy()
    asset_name = build_spec["asset_name"]
    collection_name = build_spec["generated_collection_name"]
    group_root_name = build_spec["group_root_name"]
    armature_name = build_spec["generated_armature_name"]

    source_armature = bpy_module.data.objects.get(build_spec["source_armature_name"])
    if source_armature is None or getattr(source_armature, "type", None) != "ARMATURE":
        raise ValueError(
            "Expected source armature '{0}' was not found in the open Blender file".format(
                build_spec["source_armature_name"]
            )
        )

    collection_action = choose_generated_collection_action(
        snapshot_existing_collections(bpy_module),
        collection_name,
        asset_name,
    )
    if collection_action == "rebuild":
        _remove_generated_collection(bpy_module, collection_name, asset_name)

    _ensure_generated_names_available(bpy_module, group_root_name, armature_name)
    collection = _create_collection(bpy_module, collection_name, asset_name)
    group_root = _create_group_root(bpy_module, group_root_name, asset_name, collection)
    armature_object = _create_armature_object(bpy_module, armature_name, asset_name, collection, group_root)
    _populate_edit_bones(bpy_module, armature_object, build_spec["bones"])

    return {
        "generated_collection_name": collection.name,
        "group_root_name": group_root.name,
        "generated_armature_name": armature_object.name,
        "collection_action": collection_action,
    }


def _require_bpy():
    try:
        import bpy  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - only hit outside Blender
        raise RuntimeError("The Blender builder requires bpy and must run inside Blender.") from exc
    return bpy


def _remove_generated_collection(bpy_module, collection_name: str, asset_name: str) -> None:
    collection = bpy_module.data.collections.get(collection_name)
    if collection is None:
        return
    if not bool(collection.get(GENERATED_MARKER_KEY)) or collection.get(GENERATED_ASSET_KEY) != asset_name:
        raise ValueError(
            "Refusing to rebuild over non-generated collection '{0}'".format(collection_name)
        )

    _ensure_object_mode(bpy_module)
    _validate_generated_collection_contents(collection, asset_name)

    for scene in bpy_module.data.scenes:
        if scene.collection.children.get(collection.name) is not None:
            scene.collection.children.unlink(collection)

    for parent in bpy_module.data.collections:
        if parent.children.get(collection.name) is not None:
            parent.children.unlink(collection)

    objects_to_remove = list(collection.all_objects)
    for obj in objects_to_remove:
        armature_data = obj.data if getattr(obj, "type", None) == "ARMATURE" else None
        bpy_module.data.objects.remove(obj, do_unlink=True)
        if armature_data is not None and armature_data.users == 0:
            bpy_module.data.armatures.remove(armature_data)

    bpy_module.data.collections.remove(collection)


def _validate_generated_collection_contents(collection, asset_name: str) -> None:
    for obj in collection.all_objects:
        if not bool(obj.get(GENERATED_MARKER_KEY)) or obj.get(GENERATED_ASSET_KEY) != asset_name:
            raise ValueError(
                "Refusing to remove collection '{0}' because it contains non-generated object '{1}'".format(
                    collection.name,
                    obj.name,
                )
            )


def _ensure_generated_names_available(bpy_module, group_root_name: str, armature_name: str) -> None:
    conflicts = []
    if bpy_module.data.objects.get(group_root_name) is not None:
        conflicts.append(group_root_name)
    if bpy_module.data.objects.get(armature_name) is not None:
        conflicts.append(armature_name)
    if bpy_module.data.armatures.get(armature_name) is not None:
        conflicts.append("{0} (armature data)".format(armature_name))

    if conflicts:
        raise ValueError(
            "Generated object names are already in use: {0}".format(", ".join(conflicts))
        )


def _create_collection(bpy_module, collection_name: str, asset_name: str):
    collection = bpy_module.data.collections.new(collection_name)
    collection[GENERATED_MARKER_KEY] = True
    collection[GENERATED_ASSET_KEY] = asset_name
    bpy_module.context.scene.collection.children.link(collection)
    return collection


def _create_group_root(bpy_module, group_root_name: str, asset_name: str, collection):
    group_root = bpy_module.data.objects.new(group_root_name, None)
    group_root.empty_display_type = "PLAIN_AXES"
    group_root[GENERATED_MARKER_KEY] = True
    group_root[GENERATED_ASSET_KEY] = asset_name
    collection.objects.link(group_root)
    return group_root


def _create_armature_object(bpy_module, armature_name: str, asset_name: str, collection, group_root):
    armature_data = bpy_module.data.armatures.new(armature_name)
    armature_data[GENERATED_MARKER_KEY] = True
    armature_data[GENERATED_ASSET_KEY] = asset_name

    armature_object = bpy_module.data.objects.new(armature_name, armature_data)
    armature_object.parent = group_root
    armature_object[GENERATED_MARKER_KEY] = True
    armature_object[GENERATED_ASSET_KEY] = asset_name
    collection.objects.link(armature_object)
    return armature_object


def _populate_edit_bones(bpy_module, armature_object, bones: List[dict]) -> None:
    _ensure_object_mode(bpy_module)
    _set_active_object(bpy_module, armature_object)
    bpy_module.ops.object.mode_set(mode="EDIT")

    edit_bones = armature_object.data.edit_bones
    for bone in bones:
        edit_bone = edit_bones.new(bone["name"])
        edit_bone.head = bone["head"]
        edit_bone.tail = bone["tail"]
        edit_bone.use_connect = bool(bone.get("use_connect", False))

    for bone in bones:
        parent_name = bone.get("parent_bone")
        if not parent_name:
            continue
        edit_bone = edit_bones[bone["name"]]
        edit_bone.parent = edit_bones[parent_name]
        edit_bone.use_connect = bool(bone.get("use_connect", False))

    bpy_module.ops.object.mode_set(mode="OBJECT")


def _set_active_object(bpy_module, obj) -> None:
    try:
        bpy_module.ops.object.select_all(action="DESELECT")
    except RuntimeError:
        pass
    obj.select_set(True)
    bpy_module.context.view_layer.objects.active = obj


def _ensure_object_mode(bpy_module) -> None:
    active_object: Optional[object] = getattr(bpy_module.context, "object", None)
    if active_object is None:
        return
    if getattr(active_object, "mode", "OBJECT") != "OBJECT":
        bpy_module.ops.object.mode_set(mode="OBJECT")

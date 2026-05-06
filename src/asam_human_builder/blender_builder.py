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

    group_root_name, armature_name = _resolve_generated_names(
        bpy_module,
        asset_name,
        group_root_name,
        armature_name,
    )
    collection = _create_collection(bpy_module, collection_name, asset_name)
    group_root = _create_group_root(bpy_module, group_root_name, asset_name, collection)
    armature_object = _create_armature_object(bpy_module, armature_name, asset_name, collection, group_root)
    _populate_edit_bones(bpy_module, armature_object, build_spec["bones"])
    source_translation_offset = build_spec.get("source_translation_offset", [0.0, 0.0, 0.0])
    mesh_result = _duplicate_bound_meshes(
        bpy_module,
        build_spec,
        collection,
        source_armature,
        armature_object,
        source_translation_offset,
    )

    return {
        "generated_collection_name": collection.name,
        "group_root_name": group_root.name,
        "generated_armature_name": armature_object.name,
        "collection_action": collection_action,
        "duplicated_meshes": mesh_result["duplicated_meshes"],
        "skipped_meshes": mesh_result["skipped_meshes"],
        "mesh_warnings": mesh_result["mesh_warnings"],
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


def _resolve_generated_names(
    bpy_module,
    asset_name: str,
    group_root_name: str,
    armature_name: str,
) -> tuple[str, str]:
    resolved_group_root = _resolve_unique_name(
        group_root_name,
        lambda name: bpy_module.data.objects.get(name) is not None,
        [
            "{0}_{1}".format(group_root_name, asset_name),
            "{0}_Generated".format(group_root_name),
        ],
    )
    resolved_armature = _resolve_unique_name(
        armature_name,
        lambda name: (
            bpy_module.data.objects.get(name) is not None
            or bpy_module.data.armatures.get(name) is not None
        ),
        ["{0}_Generated".format(armature_name)],
    )
    return resolved_group_root, resolved_armature


def _resolve_unique_name(preferred_name: str, is_in_use, fallback_bases: List[str]) -> str:
    candidates = [preferred_name]
    for base_name in fallback_bases:
        if base_name not in candidates:
            candidates.append(base_name)

    for candidate in candidates:
        if not is_in_use(candidate):
            return candidate

    suffix = 2
    while True:
        for base_name in candidates[1:] or [preferred_name]:
            candidate = "{0}_{1}".format(base_name, suffix)
            if not is_in_use(candidate):
                return candidate
        suffix += 1


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


def _duplicate_bound_meshes(
    bpy_module,
    build_spec: dict,
    collection,
    source_armature,
    generated_armature,
    source_translation_offset: List[float],
) -> dict:
    duplicated_meshes = []
    skipped_meshes = []
    mesh_warnings = []

    mesh_records = sorted(
        build_spec.get("mesh_binding", {}).get("meshes", []),
        key=lambda record: record.get("mesh_name") or "",
    )
    if not mesh_records:
        mesh_warnings.append("no_driven_meshes")

    for record in mesh_records:
        mesh_name = record.get("mesh_name")
        source_mesh = bpy_module.data.objects.get(mesh_name)
        if source_mesh is None:
            skipped_meshes.append({"mesh_name": mesh_name, "reason": "source_mesh_missing"})
            continue
        if getattr(source_mesh, "type", None) != "MESH":
            skipped_meshes.append({"mesh_name": mesh_name, "reason": "source_object_not_mesh"})
            continue

        generated_mesh = _copy_mesh_object(
            bpy_module,
            source_mesh,
            build_spec["asset_name"],
            collection,
            generated_armature,
            source_translation_offset,
        )
        retargeted = _retarget_armature_modifiers(generated_mesh, source_armature, generated_armature)
        if record.get("armature_link") == "parent" and not retargeted:
            mesh_warnings.append("parent_only_no_armature_modifier:{0}".format(mesh_name))

        duplicated_meshes.append(
            {
                "source_mesh_name": mesh_name,
                "generated_mesh_name": generated_mesh.name,
                "armature_link": record.get("armature_link"),
                "retargeted_armature_modifiers": retargeted,
            }
        )

    return {
        "duplicated_meshes": sorted(
            duplicated_meshes,
            key=lambda item: (item["source_mesh_name"] or "", item["generated_mesh_name"]),
        ),
        "skipped_meshes": sorted(skipped_meshes, key=lambda item: item["mesh_name"] or ""),
        "mesh_warnings": sorted(mesh_warnings),
    }


def _copy_mesh_object(
    bpy_module,
    source_mesh,
    asset_name: str,
    collection,
    parent_object,
    source_translation_offset: List[float],
):
    """Duplicate a mesh object, parent it to parent_object, and apply the armature offset."""
    generated_mesh = source_mesh.copy()
    generated_mesh.name = _resolve_unique_name(
        "ASAM_{0}".format(source_mesh.name),
        lambda name: bpy_module.data.objects.get(name) is not None,
        ["ASAM_{0}_Generated".format(source_mesh.name)],
    )
    if getattr(source_mesh, "data", None) is not None:
        generated_mesh.data = source_mesh.data.copy()
        generated_mesh.data.name = "{0}Data".format(generated_mesh.name)
        generated_mesh.data[GENERATED_MARKER_KEY] = True
        generated_mesh.data[GENERATED_ASSET_KEY] = asset_name

    world_matrix = getattr(source_mesh, "matrix_world", None)
    generated_mesh[GENERATED_MARKER_KEY] = True
    generated_mesh[GENERATED_ASSET_KEY] = asset_name
    collection.objects.link(generated_mesh)
    # Parent to the generated armature (one level below group_root) to match ASAM hierarchy.
    generated_mesh.parent = parent_object
    if world_matrix is not None:
        generated_mesh.matrix_world = world_matrix
    # Propagate the same translation offset that was applied to the armature bones so
    # the mesh remains aligned with the repositioned generated armature.
    if any(abs(float(value)) > 1e-9 for value in source_translation_offset):
        _apply_mesh_world_offset(generated_mesh, source_translation_offset)
    return generated_mesh


def _apply_mesh_world_offset(mesh_obj, offset: List[float]) -> None:
    """Shift mesh_obj's world-space position by offset using mathutils when available."""
    try:
        from mathutils import Matrix, Vector  # available inside Blender
        translation = Matrix.Translation(Vector(offset))
        mesh_obj.matrix_world = translation @ mesh_obj.matrix_world
    except ImportError:  # pragma: no cover - only hit outside Blender
        # Fallback for pure-Python test environments that stub matrix_world.
        current = getattr(mesh_obj, "matrix_world", None)
        mesh_obj.matrix_world = ("offset_applied", tuple(offset), current)


def _retarget_armature_modifiers(mesh_obj, source_armature, generated_armature) -> List[str]:
    retargeted = []
    for modifier in getattr(mesh_obj, "modifiers", []):
        if getattr(modifier, "type", None) != "ARMATURE":
            continue
        if getattr(modifier, "object", None) != source_armature:
            continue
        modifier.object = generated_armature
        retargeted.append(getattr(modifier, "name", ""))
    return sorted(retargeted)


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

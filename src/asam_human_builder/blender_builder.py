"""Blender execution helpers for generated ASAM human armatures."""

from __future__ import annotations

from typing import List, Optional

try:
    from .builder import (
        GENERATED_ASSET_KEY,
        GENERATED_MARKER_KEY,
        choose_generated_collection_action,
        compute_prefix_strip_renames,
        compute_vertex_group_remap_plan,
    )
except ImportError:  # pragma: no cover - Blender script path fallback
    from builder import (
        GENERATED_ASSET_KEY,
        GENERATED_MARKER_KEY,
        choose_generated_collection_action,
        compute_prefix_strip_renames,
        compute_vertex_group_remap_plan,
    )


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


def build_armature_in_blender(build_spec: dict, bpy_module=None, parent_collection=None, character_prefix=None) -> dict:
    """Create or rebuild the generated ASAM armature in the open Blender file.

    parent_collection: when supplied, the generated collection is linked under this
        collection instead of the scene root (crowd fan-out use case).
    character_prefix: when supplied, strips the prefix from duplicated meshes' vertex
        groups before the ASAM remap (used by a later task).
    """
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

    if source_armature.get(GENERATED_MARKER_KEY):
        # The classifier picked a previously-generated armature as the source.
        # If we proceeded, `_remove_generated_collection` below would delete the
        # collection that contains it and the build would crash on a freed
        # Blender object. Refuse cleanly and point the user at the upstream fix.
        raise ValueError(
            "Refusing to build from a previously-generated armature '{0}'. "
            "Close the .blend without saving and reopen it, or call "
            "purge_previous_generated_artifacts(bpy) before re-running the pipeline.".format(
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
    collection = _create_collection(bpy_module, collection_name, asset_name, parent_collection)
    grp_root_local_origin = build_spec.get("grp_root_local_origin", [0.0, 0.0, 0.0])
    group_root = _create_group_root(
        bpy_module, group_root_name, asset_name, collection, grp_root_local_origin
    )
    armature_object = _create_armature_object(bpy_module, armature_name, asset_name, collection, group_root)
    _populate_edit_bones(bpy_module, armature_object, build_spec["bones"])
    mesh_result = _duplicate_bound_meshes(
        bpy_module,
        build_spec,
        collection,
        source_armature,
        armature_object,
        character_prefix,
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


def purge_previous_generated_artifacts(bpy_module) -> int:
    """
    Remove every collection, object, and orphan data block left over from a
    previous builder run (anything carrying our GENERATED_MARKER_KEY).

    Without this step, a re-run inside the same Blender session sees the
    previous-run's generated `Armature_<asset>` in `bpy.data.objects`. The
    inspector exports it alongside the source rigs, and the classifier may
    pick it as the recommended primary armature (because it already carries
    clean ASAM-named bones). The build then crashes when it tries to look up
    the source again — its own collection-rebuild step has just removed it.

    Idempotent: returns 0 if nothing was found. Safe to call unconditionally
    at the start of every pipeline run.
    """
    try:
        bpy_module.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        pass

    removed = 0

    # Pass 1: remove generated objects and any orphan data blocks they leave behind.
    for obj in list(bpy_module.data.objects):
        if not obj.get(GENERATED_MARKER_KEY):
            continue
        data_block = obj.data if getattr(obj, "type", None) in ("ARMATURE", "MESH") else None
        bpy_module.data.objects.remove(obj, do_unlink=True)
        removed += 1
        if data_block is None:
            continue
        if getattr(data_block, "users", 0) != 0:
            continue
        # Use isinstance against bpy.types so we don't have to maintain a
        # parallel type-name lookup.
        if isinstance(data_block, bpy_module.types.Armature):
            bpy_module.data.armatures.remove(data_block)
        elif isinstance(data_block, bpy_module.types.Mesh):
            bpy_module.data.meshes.remove(data_block)

    # Pass 2: remove (now-empty) generated collections, unlinking from every scene first.
    for collection in list(bpy_module.data.collections):
        if not collection.get(GENERATED_MARKER_KEY):
            continue
        for scene in bpy_module.data.scenes:
            if scene.collection.children.get(collection.name) is not None:
                scene.collection.children.unlink(collection)
        for parent in bpy_module.data.collections:
            if parent.children.get(collection.name) is not None:
                parent.children.unlink(collection)
        bpy_module.data.collections.remove(collection)
        removed += 1

    return removed


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


def _create_collection(bpy_module, collection_name: str, asset_name: str, parent_collection=None):
    collection = bpy_module.data.collections.new(collection_name)
    collection[GENERATED_MARKER_KEY] = True
    collection[GENERATED_ASSET_KEY] = asset_name
    if parent_collection is not None:
        parent_collection.children.link(collection)
    else:
        bpy_module.context.scene.collection.children.link(collection)
    return collection


def _create_group_root(
    bpy_module,
    group_root_name: str,
    asset_name: str,
    collection,
    location,
):
    group_root = bpy_module.data.objects.new(group_root_name, None)
    group_root.empty_display_type = "PLAIN_AXES"
    group_root.location = tuple(float(value) for value in location)
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
    character_prefix=None,
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
        )
        retargeted = _retarget_armature_modifiers(generated_mesh, source_armature, generated_armature)
        if record.get("armature_link") == "parent" and not retargeted:
            mesh_warnings.append("parent_only_no_armature_modifier:{0}".format(mesh_name))

        if character_prefix:
            prefix_renames = compute_prefix_strip_renames(
                _vertex_group_names(generated_mesh), character_prefix
            )
            _apply_vertex_group_renames(generated_mesh, prefix_renames)

        group_names = _vertex_group_names(generated_mesh)
        remap_plan = compute_vertex_group_remap_plan(
            build_spec.get("bones", []),
            group_names,
        )
        _apply_vertex_group_renames(generated_mesh, remap_plan["renames"])
        if remap_plan["unmapped_groups"]:
            mesh_warnings.append("unmapped_vertex_groups:{0}".format(mesh_name))
        if remap_plan["asam_targets_without_source_group"]:
            mesh_warnings.append("missing_source_groups:{0}".format(mesh_name))
        if remap_plan["name_collisions"]:
            mesh_warnings.append("vertex_group_name_collision:{0}".format(mesh_name))

        duplicated_meshes.append(
            {
                "source_mesh_name": mesh_name,
                "generated_mesh_name": generated_mesh.name,
                "armature_link": record.get("armature_link"),
                "retargeted_armature_modifiers": retargeted,
                "vertex_group_remap": {
                    "renamed": remap_plan["renames"],
                    "unmapped": remap_plan["unmapped_groups"],
                    "missing_source_groups": remap_plan["asam_targets_without_source_group"],
                    "collisions": remap_plan["name_collisions"],
                },
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
):
    """Duplicate a mesh object and parent it to parent_object.

    The source mesh's ``matrix_world`` is preserved on the duplicate; Blender's
    parent-inverse handles the rebase into Grp_Root-local space when the
    generated armature (a descendant of Grp_Root) becomes the parent.
    """
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
    return generated_mesh


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


def _vertex_group_names(mesh_obj) -> List[str]:
    """Return current vertex group names on mesh_obj, working for real Blender or the test fake."""
    names: List[str] = []
    for group in getattr(mesh_obj, "vertex_groups", []):
        # Real Blender VertexGroup has a .name attribute; the test fake uses plain strings.
        name = getattr(group, "name", group)
        names.append(name)
    return names


def _apply_vertex_group_renames(mesh_obj, renames: List[dict]) -> None:
    """
    Apply a list of {"source": <old>, "target": <new>} renames to a mesh's vertex groups.

    Supports both real Blender VertexGroup objects (set .name) and the pure-Python test
    fake where vertex_groups is a list of strings (mutate the list element).
    """
    if not renames:
        return
    rename_lookup = {entry["source"]: entry["target"] for entry in renames}
    groups = getattr(mesh_obj, "vertex_groups", None)
    if groups is None:
        return
    for index, group in enumerate(groups):
        current_name = getattr(group, "name", group)
        new_name = rename_lookup.get(current_name)
        if new_name is None:
            continue
        if hasattr(group, "name"):
            group.name = new_name
        else:
            # The fake exposes vertex_groups as a list[str]; mutate in place.
            groups[index] = new_name


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


def build_crowd_in_blender(
    asset_name: str,
    wrapper_collection_name: str,
    character_specs,
    decomposition: dict,
    bpy_module=None,
) -> dict:
    """Build N per-character ASAM humans, each nested under one wrapper collection.

    Continue-on-error: a character that fails to build is recorded in
    `failed_characters` and does not abort the rest of the batch.
    """
    bpy_module = bpy_module or _require_bpy()

    _remove_generated_crowd_wrapper(bpy_module, wrapper_collection_name, asset_name)
    wrapper = _create_collection(bpy_module, wrapper_collection_name, asset_name)

    characters = []
    failed = []
    for character_id, spec in character_specs:
        try:
            result = build_armature_in_blender(
                spec, bpy_module, parent_collection=wrapper, character_prefix=character_id
            )
            result["character_id"] = character_id
            characters.append(result)
        except Exception as exc:  # continue-on-error by design
            failed.append({"character_id": character_id, "error": str(exc)})

    return {
        "wrapper_collection_name": wrapper.name,
        "characters": characters,
        "failed_characters": failed,
        "decomposition": decomposition,
    }


def _remove_generated_crowd_wrapper(bpy_module, wrapper_name: str, asset_name: str) -> None:
    """Remove a previously-generated crowd wrapper and its generated child collections.

    No-op if the wrapper is absent. Refuses to touch a wrapper that is not a safe
    generated output for this asset (mirrors _remove_generated_collection's guard).
    """
    wrapper = bpy_module.data.collections.get(wrapper_name)
    if wrapper is None:
        return
    if not bool(wrapper.get(GENERATED_MARKER_KEY)) or wrapper.get(GENERATED_ASSET_KEY) != asset_name:
        raise ValueError(
            "Refusing to rebuild over non-generated collection '{0}'".format(wrapper_name)
        )

    _ensure_object_mode(bpy_module)

    # Remove each generated child collection (objects + the collection itself).
    for child in list(wrapper.children):
        if not bool(child.get(GENERATED_MARKER_KEY)) or child.get(GENERATED_ASSET_KEY) != asset_name:
            raise ValueError(
                "Refusing to remove crowd child '{0}'; not a safe generated output".format(child.name)
            )
        for obj in list(child.all_objects):
            data_block = obj.data if getattr(obj, "type", None) == "ARMATURE" else None
            bpy_module.data.objects.remove(obj, do_unlink=True)
            if data_block is not None and getattr(data_block, "users", 0) == 0:
                bpy_module.data.armatures.remove(data_block)
        wrapper.children.unlink(child)
        bpy_module.data.collections.remove(child)

    for scene in bpy_module.data.scenes:
        if scene.collection.children.get(wrapper.name) is not None:
            scene.collection.children.unlink(wrapper)
    bpy_module.data.collections.remove(wrapper)

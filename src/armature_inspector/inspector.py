"""
Armature Inspector: Helper functions for analyzing Blender armatures.

This module provides utilities to detect armature objects in a Blender scene
and traverse/print their bone hierarchies. It also includes diagnostics for
scenes where armatures may be structured as Empty/node hierarchies and not Armature objects (e.g.
ASAM OpenMATERIAL 3D assets).
"""

import json
import os
import re
import sys
from pathlib import Path

import bpy


# region ARMATURE DETECTION

def get_armature_objects():
    """
    Find all armature objects across all Blender data (not just active scene).

    Returns:
        list: bpy.types.Object instances where type == 'ARMATURE'
    """
    return [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']


def _has_prefix_bones(armature_obj, prefix):
    """
    Check if an armature has any bones matching a prefix.

    Args:
        armature_obj: Blender armature object (bpy.types.Object)
        prefix: Prefix string to check (e.g., 'DEF-')

    Returns:
        bool: True if at least one bone name starts with the prefix
    """
    return any(b.name.startswith(prefix) for b in armature_obj.data.bones)

# endregion

# region HIERARCHY PRINTING

def _bone_matches_filter(bone, prefix_filter):
    """Check if a bone or any of its descendants match the filter."""
    if prefix_filter is None:
        return True
    if bone.name.startswith(prefix_filter):
        return True
    for child in bone.children:
        if _bone_matches_filter(child, prefix_filter):
            return True
    return False


def _count_filtered_bones(armature_obj, prefix_filter):
    """Count bones matching the filter."""
    if prefix_filter is None:
        return len(armature_obj.data.bones)
    return sum(1 for b in armature_obj.data.bones if b.name.startswith(prefix_filter))


def print_armature_hierarchy(armature_obj, prefix_filter=None):
    """
    Print the bone hierarchy of an armature object as a structured tree.

    Args:
        armature_obj: A Blender armature object (bpy.types.Object with type 'ARMATURE')
        prefix_filter: Optional prefix to filter bones (e.g., 'DEF-'). If set,
                      only bones with this prefix are shown. Parent bones that
                      don't match but have matching descendants are shown with
                      ellipsis.
    """
    arm_data = armature_obj.data
    root_bones = [b for b in arm_data.bones if b.parent is None]
    total_bones = len(arm_data.bones)
    filtered_count = _count_filtered_bones(armature_obj, prefix_filter)

    print(f"Armature: {armature_obj.name}")
    if prefix_filter:
        print(f"  Total bones: {total_bones} (showing {filtered_count} {prefix_filter}* bones)")
        print(f"  Hierarchy ({prefix_filter}* only):")
    else:
        print(f"  Total bones: {total_bones}")
        print("  Hierarchy:")

    for root in root_bones:
        _print_bone_recursive(root, indent_level=2, prefix_filter=prefix_filter)


def _print_bone_recursive(bone, indent_level, prefix_filter=None):
    """
    Recursively print a bone and its children with indentation.

    Args:
        bone: A Blender bone (bpy.types.Bone)
        indent_level: Current indentation level for tree visualization
        prefix_filter: Optional prefix to filter bones
    """
    indent = "  " * indent_level
    
    if prefix_filter is None:
        print(f"{indent}- {bone.name}")
        for child in bone.children:
            _print_bone_recursive(child, indent_level + 1, prefix_filter)
    else:
        if bone.name.startswith(prefix_filter):
            print(f"{indent}- {bone.name}")
            for child in bone.children:
                _print_bone_recursive(child, indent_level + 1, prefix_filter)
        elif _bone_matches_filter(bone, prefix_filter):
            print(f"{indent}- ({bone.name})")
            for child in bone.children:
                _print_bone_recursive(child, indent_level + 1, prefix_filter)


def print_object_hierarchy(obj, indent_level=0):
    """
    Recursively print a Blender object and its children (parent-child hierarchy).
    Works for any object type (Empty, Mesh, Armature, etc.).

    Args:
        obj: A Blender object (bpy.types.Object)
        indent_level: Current indentation level for tree visualization
    """
    indent = "  " * indent_level
    print(f"{indent}- {obj.name}  [type={obj.type}]")
    for child in obj.children:
        print_object_hierarchy(child, indent_level + 1)

# endregion

# region SCENE SUMMARY

def get_object_type_counts():
    """
    Count objects in bpy.data.objects grouped by type.

    Returns:
        dict: Mapping of object type string to count
    """
    type_counts = {}
    for obj in bpy.data.objects:
        t = obj.type
        type_counts[t] = type_counts.get(t, 0) + 1
    return type_counts


def is_default_scene():
    """
    Check if the current scene appears to be Blender's default startup scene.

    Returns:
        bool: True if this looks like the default scene
    """
    if bpy.data.filepath:
        return False

    obj_names = {obj.name for obj in bpy.data.objects}
    default_objects = {"Camera", "Cube", "Light"}
    return default_objects.issubset(obj_names) and len(obj_names) <= 4


def print_scene_summary():
    """
    Prints a brief summary of the current scene/file being inspected.
    """
    if is_default_scene():
        print("!" * 60)
        print("WARNING: Running on Blender's DEFAULT startup scene!")
        print("This is probably not the file you intended to inspect.")
        print("")
        print("To inspect a specific file, use one of these workflows:")
        print("  1. VSCode: Run 'Blender: Start', then open your file in")
        print("     that Blender window, then run 'Blender: Run Script'")
        print("  2. Blender: Open your file, go to Scripting workspace,")
        print("     open this script in Text Editor, press Alt+P to run")
        print("!" * 60)
        print("")

    filepath = bpy.data.filepath or "(unsaved / default scene)"
    print(f"File: {filepath}")

    type_counts = get_object_type_counts()
    total_objects = sum(type_counts.values())
    armature_count = type_counts.get('ARMATURE', 0)
    armature_datablocks = len(bpy.data.armatures)

    print(f"Total objects: {total_objects}")
    if type_counts:
        type_summary = ", ".join(f"{t}:{c}" for t, c in sorted(type_counts.items()))
        print(f"By type: {type_summary}")
    print(f"Armature datablocks: {armature_datablocks}")

# endregion

# region KINEMATIC CHAIN DETECTION


def _build_linear_chain_from(bone, name_keywords, prefix_filter=None):
    """
    Build a linear chain starting at a bone, following a single matching child.

    Args:
        bone: Starting Blender bone (bpy.types.Bone)
        name_keywords: Iterable of lowercase keywords to match in bone names
        prefix_filter: Optional prefix (e.g. 'DEF-'). When set, only children
                       that start with this prefix are considered, so the walk
                       stays within the filtered bone set.

    Returns:
        list[str]: Ordered list of bone names in the chain
    """
    chain = []
    current = bone
    keywords = [kw.lower() for kw in name_keywords]

    while current is not None:
        name_lower = current.name.lower()
        if not any(kw in name_lower for kw in keywords):
            break

        chain.append(current.name)

        candidate_children = (
            [c for c in current.children if c.name.startswith(prefix_filter)]
            if prefix_filter is not None
            else list(current.children)
        )
        matching_children = [
            child
            for child in candidate_children
            if any(kw in child.name.lower() for kw in keywords)
        ]
        if len(matching_children) != 1:
            break

        current = matching_children[0]

    return chain


def _find_chain_roots(bones, name_keywords, prefix_filter=None):
    """
    Find potential root bones for a chain based on name keywords.

    A root is defined as a bone whose name matches, but whose parent
    either does not exist or does not match the keywords.

    Args:
        bones: Iterable of Blender bones to search
        name_keywords: Keywords to match in bone names
        prefix_filter: Optional prefix (e.g. 'DEF-'). When set, only
                       prefix-matching bones are considered, and a parent that
                       does not start with the prefix is treated as absent,
                       so the DEF- segment starts are correctly found even
                       when parented to ORG-/MCH- structural bones.
    """
    roots = []
    keywords = [kw.lower() for kw in name_keywords]

    candidate_bones = (
        [b for b in bones if b.name.startswith(prefix_filter)]
        if prefix_filter is not None
        else list(bones)
    )

    for bone in candidate_bones:
        name_lower = bone.name.lower()
        if not any(kw in name_lower for kw in keywords):
            continue

        parent = bone.parent
        parent_is_in_filtered_set = (
            parent is not None
            and (prefix_filter is None or parent.name.startswith(prefix_filter))
            and any(kw in parent.name.lower() for kw in keywords)
        )
        if not parent_is_in_filtered_set:
            roots.append(bone)

    return roots


def detect_spine_chains(armature_obj, prefix_filter=None):
    """
    Detect candidate spine chains in an armature based on naming and hierarchy.

    Args:
        armature_obj: Blender armature object
        prefix_filter: Optional prefix (e.g. 'DEF-') to restrict detection to
                       a filtered bone set. Non-prefix parents are treated as
                       absent so DEF- segment roots are correctly identified.

    Returns:
        list[list[str]]: List of chains, each a list of bone names.
    """
    bones = armature_obj.data.bones
    roots = _find_chain_roots(bones, ["spine"], prefix_filter=prefix_filter)
    chains = []

    for root in roots:
        chain = _build_linear_chain_from(root, ["spine"], prefix_filter=prefix_filter)
        if len(chain) >= 2:
            chains.append(chain)

    return chains


def detect_leg_chains(armature_obj, prefix_filter=None):
    """
    Detect candidate leg chains (left/right) based on naming and hierarchy.

    Args:
        armature_obj: Blender armature object
        prefix_filter: Optional prefix (e.g. 'DEF-') to restrict detection to
                       a filtered bone set.

    Returns:
        dict[str, list[list[str]]]: Mapping side ("left", "right", "unsided")
        to lists of chains.
    """
    bones = armature_obj.data.bones
    result = {"left": [], "right": [], "unsided": []}

    for side_suffix, side_key in ((".L", "left"), (".R", "right")):
        roots = _find_chain_roots(
            [b for b in bones if b.name.endswith(side_suffix)],
            ["thigh", "leg"],
            prefix_filter=prefix_filter,
        )
        for root in roots:
            chain = _build_linear_chain_from(
                root,
                ["thigh", "shin", "leg", "calf", "foot", "toe"],
                prefix_filter=prefix_filter,
            )
            if len(chain) >= 2:
                result[side_key].append(chain)

    unsided_roots = _find_chain_roots(
        [b for b in bones if not (b.name.endswith(".L") or b.name.endswith(".R"))],
        ["thigh", "leg"],
        prefix_filter=prefix_filter,
    )
    for root in unsided_roots:
        chain = _build_linear_chain_from(
            root,
            ["thigh", "shin", "leg", "calf", "foot", "toe"],
            prefix_filter=prefix_filter,
        )
        if len(chain) >= 2:
            result["unsided"].append(chain)

    return result


def detect_arm_chains(armature_obj, prefix_filter=None):
    """
    Detect candidate arm chains (left/right) based on naming and hierarchy.

    Args:
        armature_obj: Blender armature object
        prefix_filter: Optional prefix (e.g. 'DEF-') to restrict detection to
                       a filtered bone set.

    Returns:
        dict[str, list[list[str]]]: Mapping side ("left", "right", "unsided")
        to lists of chains.
    """
    bones = armature_obj.data.bones
    result = {"left": [], "right": [], "unsided": []}

    for side_suffix, side_key in ((".L", "left"), (".R", "right")):
        roots = _find_chain_roots(
            [b for b in bones if b.name.endswith(side_suffix)],
            ["shoulder", "upper_arm", "arm"],
            prefix_filter=prefix_filter,
        )
        for root in roots:
            chain = _build_linear_chain_from(
                root,
                ["shoulder", "upper_arm", "arm", "forearm", "elbow", "hand", "wrist"],
                prefix_filter=prefix_filter,
            )
            if len(chain) >= 2:
                result[side_key].append(chain)

    unsided_roots = _find_chain_roots(
        [b for b in bones if not (b.name.endswith(".L") or b.name.endswith(".R"))],
        ["shoulder", "upper_arm", "arm"],
        prefix_filter=prefix_filter,
    )
    for root in unsided_roots:
        chain = _build_linear_chain_from(
            root,
            ["shoulder", "upper_arm", "arm", "forearm", "elbow", "hand", "wrist"],
            prefix_filter=prefix_filter,
        )
        if len(chain) >= 2:
            result["unsided"].append(chain)

    return result


def _format_chain_inline(chain, max_len=6):
    """
    Format a chain for log output, truncating if long.
    """
    if len(chain) <= max_len:
        return " -> ".join(chain)
    head = " -> ".join(chain[:3])
    tail = " -> ".join(chain[-2:])
    return f"{head} -> ... -> {tail} ({len(chain)} bones)"


def _filter_chains_data(chains_data, prefix_filter):
    """
    Filter chains data by trimming and splitting on non-prefix bones.

    Non-prefix bones at the start/end of a chain are trimmed. Non-prefix bones
    in the middle (e.g. ORG- connectors between DEF- segments) cause the chain
    to split, so that DEF- bones on either side are not falsely shown as directly
    connected. Sub-chains shorter than 2 bones are discarded.

    Args:
        chains_data: Dict with "spine", "leg", "arm" keys containing chain data
        prefix_filter: Prefix string to filter by (e.g., 'DEF-'), or None to skip filtering

    Returns:
        dict: Filtered chains data with same structure as input
    """
    if prefix_filter is None:
        return chains_data

    def split_and_trim_chain(chain):
        sub_chains = []
        current = []
        for bone_name in chain:
            if bone_name.startswith(prefix_filter):
                current.append(bone_name)
            else:
                if len(current) >= 2:
                    sub_chains.append(current)
                current = []
        if len(current) >= 2:
            sub_chains.append(current)
        return sub_chains

    def filter_chain_list(chain_list):
        result = []
        for chain in chain_list:
            result.extend(split_and_trim_chain(chain))
        return result

    def filter_sided_chains(sided_dict):
        return {
            side: filter_chain_list(chains)
            for side, chains in sided_dict.items()
        }

    return {
        "spine": filter_chain_list(chains_data["spine"]),
        "leg": filter_sided_chains(chains_data["leg"]),
        "arm": filter_sided_chains(chains_data["arm"]),
    }


def print_detected_chains(armature_obj):
    """
    Detect and print potential kinematic chains for an armature.
    """
    print("\n  Detected Kinematic Chains:")
    print("  " + "-" * 40)

    spine_chains = detect_spine_chains(armature_obj)
    leg_chains = detect_leg_chains(armature_obj)
    arm_chains = detect_arm_chains(armature_obj)

    total = 0

    if spine_chains:
        print(f"    Spine ({len(spine_chains)} chain(s)):")
        for idx, chain in enumerate(spine_chains, start=1):
            print(f"      [{idx}] {_format_chain_inline(chain)}")
            total += 1

    def _print_limb_group(label, chains_by_side):
        nonlocal total
        count = sum(len(v) for v in chains_by_side.values())
        if count == 0:
            return
        print(f"    {label} ({count} chain(s)):")
        for side_key, side_label in (("left", "Left"), ("right", "Right"), ("unsided", "Unsided")):
            for chain in chains_by_side[side_key]:
                print(f"      [{side_label}] {_format_chain_inline(chain)}")
                total += 1

    _print_limb_group("Leg", leg_chains)
    _print_limb_group("Arm", arm_chains)

    if total == 0:
        print("    (no obvious spine/leg/arm chains detected)")
        print("    Note: detection is heuristic and based on names + hierarchy.")

    print("  " + "-" * 40)
    print(f"  Total chains identified: {total}")

# endregion

# region BONE GEOMETRY EXTRACTION


def extract_bone_geometry(bone):
    """
    Extract geometry and basic transform data from a single bone.

    Args:
        bone: A Blender bone (bpy.types.Bone)

    Returns:
        dict: Geometry data for the bone.
    """
    return {
        "name": bone.name,
        "parent": bone.parent.name if bone.parent else None,
        "head": tuple(bone.head_local),
        "tail": tuple(bone.tail_local),
        "length": float(bone.length),
        "matrix_local": [list(row) for row in bone.matrix_local],
    }


def extract_armature_geometry(armature_obj, prefix_filter=None):
    """
    Extract geometry data for all bones in an armature.

    Args:
        armature_obj: Blender armature object (bpy.types.Object)
        prefix_filter: Optional prefix to restrict bones (e.g., 'DEF-')

    Returns:
        list[dict]: Geometry data for each bone.
    """
    bones = armature_obj.data.bones
    result = []
    for bone in bones:
        if prefix_filter is not None and not bone.name.startswith(prefix_filter):
            continue
        result.append(extract_bone_geometry(bone))
    return result


AXIS_NAMES = ("x", "y", "z")


def _detect_bone_side(name):
    """Infer left/right side markers from a bone name."""
    lowered = name.lower()
    raw_tokens = [token for token in re.split(r"[^a-z0-9]+", lowered) if token]
    for token in raw_tokens:
        if token in {"l", "left"}:
            return "left"
        if token in {"r", "right"}:
            return "right"
    return None


def _detect_bone_tags(name):
    """Extract coarse semantic tags from a bone name."""
    lowered = name.lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    tags = set()

    if "head" in compact:
        tags.add("head")
    if "neck" in compact:
        tags.add("neck")
    if "spine" in compact or "chest" in compact or "torso" in compact:
        tags.add("spine")
    if "foot" in compact or "toe" in compact or "ankle" in compact:
        tags.add("foot")
    if "leg" in compact or "thigh" in compact or "shin" in compact or "calf" in compact:
        tags.add("leg")

    return tags


def _mean(values):
    return sum(values) / float(len(values)) if values else 0.0


def _axis_cross_sign(axis_a, axis_b, axis_c):
    """Return +1 when e_a x e_b == e_c, else -1."""
    return 1 if ((axis_a + 1) % 3 == axis_b and (axis_b + 1) % 3 == axis_c) else -1


def _infer_axis_metadata_from_bones(bones_data):
    """
    Infer semantic forward/side/up axes from bone geometry.

    Returns:
        dict: Semantic axis metadata with index and sign information.
    """
    if not bones_data:
        return {
            "forward_axis": {"index": 0, "name": "x", "sign": 1},
            "side_axis": {"index": 1, "name": "y", "sign": 1},
            "up_axis": {"index": 2, "name": "z", "sign": 1},
        }

    midpoints = []
    side_groups = {"left": [], "right": []}
    upper_points = []
    lower_points = []

    for bone in bones_data:
        head = bone.get("head") or (0.0, 0.0, 0.0)
        tail = bone.get("tail") or (0.0, 0.0, 0.0)
        midpoint = [float((head[index] + tail[index]) / 2.0) for index in range(3)]
        midpoints.append(midpoint)

        side = _detect_bone_side(bone.get("name", ""))
        if side in side_groups:
            side_groups[side].append(midpoint)

        tags = _detect_bone_tags(bone.get("name", ""))
        if {"head", "neck", "spine"} & tags:
            upper_points.append(midpoint)
        if {"foot", "leg"} & tags:
            lower_points.append(midpoint)

    spans = []
    for axis in range(3):
        values = [point[axis] for point in midpoints]
        spans.append(max(values) - min(values))

    up_index = max(range(3), key=lambda axis: spans[axis])
    remaining_axes = [axis for axis in range(3) if axis != up_index]

    left_means = {
        axis: _mean([point[axis] for point in side_groups["left"]]) if side_groups["left"] else None
        for axis in remaining_axes
    }
    right_means = {
        axis: _mean([point[axis] for point in side_groups["right"]]) if side_groups["right"] else None
        for axis in remaining_axes
    }

    side_index = None
    best_side_delta = -1.0
    for axis in remaining_axes:
        left_mean = left_means[axis]
        right_mean = right_means[axis]
        if left_mean is None or right_mean is None:
            continue
        delta = abs(left_mean - right_mean)
        if delta > best_side_delta:
            best_side_delta = delta
            side_index = axis
    if side_index is None:
        side_index = max(remaining_axes, key=lambda axis: spans[axis])

    forward_index = next(axis for axis in range(3) if axis not in {up_index, side_index})

    up_sign = 1
    if upper_points and lower_points:
        upper_mean = _mean([point[up_index] for point in upper_points])
        lower_mean = _mean([point[up_index] for point in lower_points])
        if upper_mean < lower_mean:
            up_sign = -1

    side_sign = 1
    left_mean = left_means.get(side_index)
    right_mean = right_means.get(side_index)
    if left_mean is not None and right_mean is not None and left_mean < right_mean:
        side_sign = -1

    forward_sign = up_sign * _axis_cross_sign(forward_index, side_index, up_index) * side_sign
    return {
        "forward_axis": {"index": forward_index, "name": AXIS_NAMES[forward_index], "sign": int(forward_sign)},
        "side_axis": {"index": side_index, "name": AXIS_NAMES[side_index], "sign": int(side_sign)},
        "up_axis": {"index": up_index, "name": AXIS_NAMES[up_index], "sign": int(up_sign)},
    }


def _build_bbox_metadata(points, axis_metadata):
    """Build bounding box metadata from local-space points."""
    if not points:
        zero = [0.0, 0.0, 0.0]
        return {
            "bbox_min": list(zero),
            "bbox_max": list(zero),
            "bbox_height": 0.0,
            "bbox_ground_center": list(zero),
        }

    bbox_min = [min(point[index] for point in points) for index in range(3)]
    bbox_max = [max(point[index] for point in points) for index in range(3)]

    up_axis = axis_metadata["up_axis"]["index"]
    up_sign = axis_metadata["up_axis"]["sign"]
    bbox_height = abs(bbox_max[up_axis] - bbox_min[up_axis])
    bbox_ground_center = [(bbox_min[index] + bbox_max[index]) / 2.0 for index in range(3)]
    bbox_ground_center[up_axis] = bbox_min[up_axis] if up_sign >= 0 else bbox_max[up_axis]

    return {
        "bbox_min": [float(value) for value in bbox_min],
        "bbox_max": [float(value) for value in bbox_max],
        "bbox_height": float(bbox_height),
        "bbox_ground_center": [float(value) for value in bbox_ground_center],
    }


def build_armature_placement_metadata(bones_data, mesh_points=None, driven_meshes=None):
    """
    Build self-contained armature placement metadata from geometry inputs.

    Args:
        bones_data: List of exported bone payloads.
        mesh_points: Optional list of local-space mesh bound-box corner points.
        driven_meshes: Optional list of mesh object names contributing to the bounds.

    Returns:
        dict: Placement metadata for later classifier/builder steps.
    """
    axis_metadata = _infer_axis_metadata_from_bones(bones_data)

    if mesh_points:
        bounds_points = [[float(value) for value in point] for point in mesh_points]
        bounds_source = "meshes"
    else:
        bounds_points = []
        for bone in bones_data:
            if bone.get("head"):
                bounds_points.append([float(value) for value in bone["head"]])
            if bone.get("tail"):
                bounds_points.append([float(value) for value in bone["tail"]])
        bounds_source = "bones_fallback"

    metadata = {
        "bounds_source": bounds_source,
        "driven_meshes": sorted(set(driven_meshes or [])),
    }
    metadata.update(_build_bbox_metadata(bounds_points, axis_metadata))
    metadata.update(axis_metadata)
    return metadata

# endregion

# region MESH_BINDING


def _meshes_driven_by_armature(armature_obj):
    """
    Unique mesh objects parented to the armature or using it in an Armature modifier.
    """
    meshes_by_id = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        driven = obj.parent == armature_obj
        if not driven:
            for modifier in getattr(obj, "modifiers", []):
                if modifier.type == "ARMATURE" and getattr(modifier, "object", None) == armature_obj:
                    driven = True
                    break
        if driven:
            meshes_by_id[id(obj)] = obj
    return list(meshes_by_id.values())


def _collect_armature_mesh_bounds(armature_obj):
    """
    Collect mesh bound-box corner points in armature-local coordinates.
    """
    from mathutils import Vector

    armature_inverse = armature_obj.matrix_world.inverted()
    driven_meshes = []
    local_points = []

    for obj in _meshes_driven_by_armature(armature_obj):
        driven_meshes.append(obj.name)
        for corner in obj.bound_box:
            world_point = obj.matrix_world @ Vector(corner)
            local_point = armature_inverse @ world_point
            local_points.append([float(local_point[0]), float(local_point[1]), float(local_point[2])])

    return local_points, driven_meshes


def build_mesh_binding(armature_obj):
    """
    Build machine-readable mesh-to-armature binding metadata for JSON export.
    """
    mesh_bindings = []
    driven_meshes = sorted(_meshes_driven_by_armature(armature_obj), key=lambda obj: obj.name)

    for obj in driven_meshes:
        warnings = []
        parent_link = getattr(obj, "parent", None) == armature_obj
        modifier_link = False

        modifiers_payload = []
        for stack_index, modifier in enumerate(getattr(obj, "modifiers", [])):
            modifier_type = getattr(modifier, "type", None)
            modifier_object_name = None
            if modifier_type == "ARMATURE":
                modifier_target = getattr(modifier, "object", None)
                modifier_object_name = getattr(modifier_target, "name", None)
                if modifier_target == armature_obj:
                    modifier_link = True

            modifiers_payload.append(
                {
                    "stack_index": stack_index,
                    "type": modifier_type,
                    "name": getattr(modifier, "name", None),
                    "object": modifier_object_name,
                }
            )

        if parent_link and modifier_link:
            armature_link = "parent_and_modifier"
        elif parent_link:
            armature_link = "parent"
        else:
            armature_link = "modifier"

        vertex_group_records = sorted(
            list(getattr(obj, "vertex_groups", [])),
            key=lambda group: group.name,
        )
        vertex_groups = [group.name for group in vertex_group_records]
        group_index_to_name = {}
        weighted_vertex_counts = {}
        for fallback_index, group in enumerate(vertex_group_records):
            group_index = getattr(group, "index", fallback_index)
            group_index_to_name[group_index] = group.name
            weighted_vertex_counts[group_index] = 0

        mesh_data = getattr(obj, "data", None)
        if mesh_data is None:
            warnings.append("mesh_data_missing")
        else:
            for vertex in getattr(mesh_data, "vertices", []):
                seen_non_empty_groups = set()
                for group_element in getattr(vertex, "groups", []):
                    group_index = getattr(group_element, "group", None)
                    weight = float(getattr(group_element, "weight", 0.0))
                    if group_index not in weighted_vertex_counts:
                        continue
                    if weight > 1e-6:
                        seen_non_empty_groups.add(group_index)
                for group_index in seen_non_empty_groups:
                    weighted_vertex_counts[group_index] += 1

        per_group_stats = [
            {
                "name": group_name,
                "weighted_vertex_count": weighted_vertex_counts[group_index],
            }
            for group_index, group_name in sorted(
                group_index_to_name.items(),
                key=lambda entry: entry[1],
            )
        ]
        non_empty_group_count = sum(
            1 for entry in per_group_stats if entry["weighted_vertex_count"] > 0
        )

        material_slots = []
        for slot_index, slot in enumerate(getattr(obj, "material_slots", [])):
            material = getattr(slot, "material", None)
            material_slots.append(
                {
                    "slot_index": slot_index,
                    "material_name": getattr(material, "name", None) if material is not None else None,
                }
            )

        mesh_bindings.append(
            {
                "mesh_name": obj.name,
                "armature_link": armature_link,
                "modifiers": modifiers_payload,
                "vertex_groups": vertex_groups,
                "vertex_group_stats": {
                    "non_empty_group_count": non_empty_group_count,
                    "per_group": per_group_stats,
                },
                "material_slots": material_slots,
                "warnings": sorted(warnings),
            }
        )

    return {
        "armature_object_name": armature_obj.name,
        "meshes": mesh_bindings,
    }


# endregion


def _format_vec3(vec, precision=4):
    """Format a 3D vector for log output."""
    return f"({vec[0]:.{precision}f}, {vec[1]:.{precision}f}, {vec[2]:.{precision}f})"


def print_bone_geometry(armature_obj, prefix_filter="DEF-", max_bones=32):
    """
    Print bone geometry data for an armature.

    By default, only DEF-* deformation bones are shown. If no bones match
    that filter, it falls back to printing all bones.
    """
    bones_data = extract_armature_geometry(armature_obj, prefix_filter=prefix_filter)

    if not bones_data:
        bones_data = extract_armature_geometry(armature_obj, prefix_filter=None)
        filter_label = "all bones"
    else:
        filter_label = f"{prefix_filter}* bones"

    print(f"\n  Bone Geometry ({filter_label}):")
    print("  " + "-" * 50)

    if not bones_data:
        print("    (no bones found)")
        print("  " + "-" * 50)
        return

    shown = 0
    total = len(bones_data)

    for data in bones_data:
        if max_bones is not None and shown >= max_bones:
            remaining = total - shown
            if remaining > 0:
                print(f"    ... and {remaining} more bones")
            break

        head = _format_vec3(data["head"])
        tail = _format_vec3(data["tail"])
        parent = data["parent"] or "(root)"

        print(f"    {data['name']}")
        print(f"      head: {head}  tail: {tail}")
        print(f"      length: {data['length']:.4f}  parent: {parent}")

        shown += 1

    print("  " + "-" * 50)
    print(f"  Total bones shown: {shown} / {total}")


def _build_hierarchy_dict(armature_obj, prefix_filter=None):
    """
    Build a dictionary representing the bone hierarchy.
    
    Args:
        armature_obj: Blender armature object
        prefix_filter: Optional prefix to filter bones
        
    Returns:
        dict: Mapping of bone name to list of child bone names
    """
    hierarchy = {}
    bones = armature_obj.data.bones
    
    for bone in bones:
        if prefix_filter is not None and not bone.name.startswith(prefix_filter):
            continue
        
        if prefix_filter is not None:
            children = [c.name for c in bone.children if c.name.startswith(prefix_filter)]
        else:
            children = [c.name for c in bone.children]
        
        hierarchy[bone.name] = children
    
    return hierarchy


def _get_output_dir():
    """Get the output directory for JSON files.

    Resolves to the shared pipeline output root (configurable via
    OPEN_JAYWALKER_OUTPUT_ROOT). Falls back to "unsaved" when the .blend
    has not been saved to disk.
    """
    if bpy.data.filepath:
        blend_name = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
    else:
        blend_name = "unsaved"

    # Defensive sys.path patch + lazy pipeline_paths import: the inspector is
    # loaded from inside Blender via src/pipeline/main.py, which adds src/ to
    # sys.path before this module runs. Test environments that import the
    # inspector directly without configuring sys.path need this fallback.
    src_dir = str(Path(__file__).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    from pipeline_paths import resolve_asset_dir

    return str(resolve_asset_dir(blend_name))


def get_output_dir():
    """Expose the resolved output directory for the current Blender file."""
    return _get_output_dir()


def export_armature_json(armature_obj):
    """
    Export bone geometry and detected chains to JSON files.
    
    Exports TWO files per armature to the output/ folder:
    - {armature_name}_all.json - all bones (unfiltered)
    - {armature_name}_filtered.json - DEF-* bones only
    
    Args:
        armature_obj: Blender armature object (bpy.types.Object)
        
    Returns:
        tuple: Paths to the two JSON files (all_path, filtered_path)
    """
    output_dir = _get_output_dir()
    safe_name = armature_obj.name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    
    source_file = bpy.data.filepath or "(unsaved)"

    mesh_points, driven_meshes = _collect_armature_mesh_bounds(armature_obj)
    mesh_binding = build_mesh_binding(armature_obj)

    all_chains = {
        "spine": detect_spine_chains(armature_obj),
        "leg": detect_leg_chains(armature_obj),
        "arm": detect_arm_chains(armature_obj),
    }
    all_bones = extract_armature_geometry(armature_obj, prefix_filter=None)
    all_hierarchy = _build_hierarchy_dict(armature_obj, prefix_filter=None)
    all_data = {
        "armature_name": armature_obj.name,
        "source_file": source_file,
        "filter": None,
        "bone_count": len(all_bones),
        "hierarchy": all_hierarchy,
        "bones": all_bones,
        "chains": all_chains,
        "placement_metadata": build_armature_placement_metadata(
            all_bones,
            mesh_points=mesh_points,
            driven_meshes=driven_meshes,
        ),
        "mesh_binding": mesh_binding,
    }
    
    all_path = os.path.join(output_dir, f"{safe_name}_all.json")
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)
    
    default_prefix = "DEF-"
    prefix_filter = default_prefix if _has_prefix_bones(armature_obj, default_prefix) else None
    filtered_chains = {
        "spine": detect_spine_chains(armature_obj, prefix_filter=prefix_filter),
        "leg": detect_leg_chains(armature_obj, prefix_filter=prefix_filter),
        "arm": detect_arm_chains(armature_obj, prefix_filter=prefix_filter),
    }
    filtered_bones = extract_armature_geometry(armature_obj, prefix_filter=prefix_filter)
    filtered_hierarchy = _build_hierarchy_dict(armature_obj, prefix_filter=prefix_filter)
    filtered_data = {
        "armature_name": armature_obj.name,
        "source_file": source_file,
        "filter": prefix_filter,
        "bone_count": len(filtered_bones),
        "hierarchy": filtered_hierarchy,
        "bones": filtered_bones,
        "chains": filtered_chains,
        "placement_metadata": build_armature_placement_metadata(
            filtered_bones,
            mesh_points=mesh_points,
            driven_meshes=driven_meshes,
        ),
        "mesh_binding": mesh_binding,
    }
    
    filtered_path = os.path.join(output_dir, f"{safe_name}_filtered.json")
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"\n  Exported:")
    print(f"    All bones   -> {all_path}")
    filter_label = f"{prefix_filter}*" if prefix_filter else "all (no DEF- bones)"
    print(f"    {filter_label} bones -> {filtered_path}")
    
    return all_path, filtered_path

# region SCENE DIAGNOSTICS

def dump_all_objects():
    """
    Print every object in bpy.data.objects with its type, parent, and
    collection membership. Useful for diagnosing scenes where the expected
    armature is missing.
    """
    print("ALL OBJECTS IN bpy.data.objects:")
    print("-" * 60)

    if not bpy.data.objects:
        print("  (no objects)")
        return

    type_counts = {}
    for obj in bpy.data.objects:
        t = obj.type
        type_counts[t] = type_counts.get(t, 0) + 1

        parent_name = obj.parent.name if obj.parent else "(none)"
        collections = ", ".join(c.name for c in obj.users_collection) or "(none)"
        print(f"  {obj.name:40s}  type={t:12s}  parent={parent_name:30s}  collections={collections}")

    print("-" * 60)
    print("Object type summary:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")


def dump_collections():
    """
    Print all collections in the .blend file and which scene(s) they belong to.
    """
    print("\nCOLLECTIONS IN FILE:")
    print("-" * 60)

    for scene in bpy.data.scenes:
        print(f"  Scene: {scene.name}")
        _print_collection_tree(scene.collection, indent_level=2)


def _print_collection_tree(collection, indent_level):
    """Recursively print a collection tree."""
    indent = "  " * indent_level
    obj_count = len(collection.objects)
    print(f"{indent}[Collection] {collection.name}  ({obj_count} objects)")
    for child in collection.children:
        _print_collection_tree(child, indent_level + 1)


def dump_armature_datablocks():
    """
    Print all Armature datablocks in bpy.data.armatures, regardless of whether
    any object references them.
    """
    print("\nARMATURE DATABLOCKS IN bpy.data.armatures:")
    print("-" * 60)

    if not bpy.data.armatures:
        print("  (none)")
        return

    for arm in bpy.data.armatures:
        bone_count = len(arm.bones)
        users = arm.users
        print(f"  {arm.name:40s}  bones={bone_count}  users={users}")

# endregion

# region MAIN EXECUTION

def inspect_scene():
    """
    Inspect the current Blender scene for armatures and print their hierarchies.
    Also prints object-level parent-child trees for root objects if no armatures
    are found (to handle OpenMATERIAL-style Empty/node hierarchies).
    
    Console output shows filtered hierarchy (DEF-* bones only).
    JSON files are exported for both filtered and unfiltered data.
    """
    armatures = get_armature_objects()
    output_dir = _get_output_dir()
    source_file = bpy.data.filepath or "(unsaved)"
    result = {
        "source_file": source_file,
        "output_dir": output_dir,
        "armature_count": len(armatures),
        "armature_names": [],
        "exported_files": [],
        "diagnostics_ran": False,
    }

    print("=" * 60)
    print("ARMATURE INSPECTOR REPORT")
    print("=" * 60)
    print_scene_summary()
    print("-" * 60)

    if not armatures:
        print("No armature objects found.")
        print("Running full scene diagnostics...\n")
        run_diagnostics()
        result["diagnostics_ran"] = True
        return result

    print(f"Found {len(armatures)} armature(s).")
    print("-" * 60)

    for i, obj in enumerate(armatures, start=1):
        print(f"\n[{i}/{len(armatures)}]")
        default_prefix = "DEF-"
        prefix_filter = default_prefix if _has_prefix_bones(obj, default_prefix) else None
        print_armature_hierarchy(obj, prefix_filter=prefix_filter)
        print_detected_chains(obj)
        all_path, filtered_path = export_armature_json(obj)
        result["armature_names"].append(obj.name)
        result["exported_files"].extend([all_path, filtered_path])

    print("\n" + "=" * 60)
    print("END OF REPORT")
    print("=" * 60)
    return result


def run_diagnostics():
    """
    Run a full diagnostic dump of the scene to understand why no armatures
    were detected. Prints all objects, collections, armature datablocks,
    and the object parent-child hierarchy.
    """
    dump_all_objects()
    dump_armature_datablocks()
    dump_collections()

    print("\nOBJECT PARENT-CHILD HIERARCHY (root objects):")
    print("-" * 60)
    root_objects = [obj for obj in bpy.data.objects if obj.parent is None]
    if not root_objects:
        print("  (no root objects)")
    else:
        for obj in root_objects:
            print_object_hierarchy(obj, indent_level=1)

    print("\n" + "=" * 60)
    print("END OF DIAGNOSTICS")
    print("=" * 60)

# endregion MAIN EXECUTION

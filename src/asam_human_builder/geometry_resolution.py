"""
Geometry resolution for the ASAM human builder.

Given a target name from `CORE_TARGETS`, the semantic mapping from the classifier,
and the source rig's bone geometry, this module decides where the corresponding
generated bone's head/tail should sit. It also exposes the small vector-math
helpers and side-naming utilities that the resolver pipeline relies on.

The resolver is deterministic and pure: it never touches Blender state. The
builder.py orchestrator owns the spec construction, mesh-binding remap planning,
and report writing; this module owns the per-bone geometric decisions and the
fallback strategies (mirror across the centerline, interpolate between mapped
ancestors and descendants, extrapolate, place by bounding-box).
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple


RECOVERABLE_ACTIONS = {"direct_map", "alias_map", "repair_in_builder"}


# Synthesized Root bone tail length, as a fraction of the asset's bbox height,
# when the classifier forces `create_new_root` mode.
ROOT_TAIL_HEIGHT_RATIO = 0.18


# Per-target default bone length expressed as a fraction of the asset's bbox
# height. Used by `_default_length_for_target` when no source geometry is
# available and the resolver must synthesize a new bone.
DEFAULT_LENGTH_RATIOS = {
    "Hip": 0.05,
    "Lower_Spine": 0.10,
    "Upper_Spine": 0.12,
    "Neck": 0.05,
    "Head": 0.08,
    "Eye_Left": 0.02,
    "Eye_Right": 0.02,
    "Shoulder_Left": 0.06,
    "Shoulder_Right": 0.06,
    "Upper_Arm_Left": 0.16,
    "Upper_Arm_Right": 0.16,
    "Lower_Arm_Left": 0.15,
    "Lower_Arm_Right": 0.15,
    "Hand_Left": 0.07,
    "Hand_Right": 0.07,
    "Full_Thumb_Left": 0.05,
    "Full_Thumb_Right": 0.05,
    "Full_Fingers_Left": 0.06,
    "Full_Fingers_Right": 0.06,
    "Upper_Leg_Left": 0.22,
    "Upper_Leg_Right": 0.22,
    "Lower_Leg_Left": 0.22,
    "Lower_Leg_Right": 0.22,
    "Foot_Left": 0.10,
    "Foot_Right": 0.10,
    "Full_Toes_Left": 0.05,
    "Full_Toes_Right": 0.05,
}


def _resolve_target_geometry(
    target_name: str,
    semantic_mapping: dict,
    root_resolution: dict,
    source_bones: Dict[str, dict],
    placement_metadata: dict,
    bone_parents: dict,
    children_map: Dict[str, List[str]],
    resolved_geometry: Dict[str, dict],
    warnings: List[str],
) -> Tuple[dict, str, Optional[str]]:
    if target_name == "Root":
        return _resolve_root_geometry(root_resolution, source_bones, placement_metadata, warnings)

    payload = semantic_mapping[target_name]
    source_bone_name = payload.get("source_bone")
    if target_name == "Hip" and _hip_requires_centered_pelvis_pair(payload):
        geometry = _resolve_centered_hip_geometry(
            source_bone_name,
            semantic_mapping,
            root_resolution,
            source_bones,
            placement_metadata,
            bone_parents,
            children_map,
            resolved_geometry,
        )
        return geometry, "centered_pelvis_pair", source_bone_name

    if payload.get("action") in RECOVERABLE_ACTIONS and source_bone_name in source_bones:
        return copy.deepcopy(source_bones[source_bone_name]), "source_bone", source_bone_name

    if payload.get("action") in RECOVERABLE_ACTIONS and source_bone_name and source_bone_name not in source_bones:
        warnings.append("missing_source_geometry:{0}->{1}".format(target_name, source_bone_name))

    return _resolve_created_target_geometry(
        target_name,
        semantic_mapping,
        root_resolution,
        source_bones,
        placement_metadata,
        bone_parents,
        children_map,
        resolved_geometry,
    )


def _hip_requires_centered_pelvis_pair(payload: dict) -> bool:
    return payload.get("action") == "repair_in_builder" and "paired_sided_pelvis_requires_centering" in set(
        payload.get("notes", [])
    )


def _resolve_centered_hip_geometry(
    source_bone_name: Optional[str],
    semantic_mapping: dict,
    root_resolution: dict,
    source_bones: Dict[str, dict],
    placement_metadata: dict,
    bone_parents: dict,
    children_map: Dict[str, List[str]],
    resolved_geometry: Dict[str, dict],
) -> dict:
    root_geometry = resolved_geometry.get("Root")
    if root_geometry is None:
        root_geometry, _, _ = _resolve_root_geometry(root_resolution, source_bones, placement_metadata, [])

    lower_spine_geometry = _get_reference_geometry(
        "Lower_Spine",
        semantic_mapping,
        root_resolution,
        source_bones,
        resolved_geometry,
        placement_metadata,
    )

    centerline = _placement_centerline(placement_metadata)
    side_axis = int(placement_metadata["side_axis"]["index"])
    default_length = _default_length_for_target("Hip", placement_metadata)
    default_direction = _default_direction_for_target("Hip", placement_metadata, root_geometry)

    source_geometry = source_bones.get(source_bone_name) if source_bone_name in source_bones else None
    opposite_geometry = _find_opposite_pelvis_geometry(source_bone_name, source_bones)

    # ASAM Hip sits at the pelvis, where the source pelvis pair originates, not at
    # the synthesized Root's tail (an arbitrary bbox-ratio display stub). Anchor the
    # head to the centered pelvis-pair head so the bone is anatomically coherent and
    # stays the common parent of Lower_Spine and both legs.
    if source_geometry is not None and opposite_geometry is not None:
        head = [
            float((source_geometry["head"][index] + opposite_geometry["head"][index]) / 2.0)
            for index in range(3)
        ]
    elif source_geometry is not None:
        head = list(source_geometry["head"])
    else:
        head = list(root_geometry["tail"])
    head[side_axis] = centerline

    if lower_spine_geometry is not None:
        tail = list(lower_spine_geometry["head"])
        tail[side_axis] = centerline
        return _ensure_non_zero_geometry(head, tail, placement_metadata)

    if source_geometry is not None and opposite_geometry is not None:
        averaged_tail = [
            float((source_geometry["tail"][index] + opposite_geometry["tail"][index]) / 2.0)
            for index in range(3)
        ]
        averaged_tail[side_axis] = centerline
        if _distance(head, averaged_tail) > 1e-5:
            return _ensure_non_zero_geometry(head, averaged_tail, placement_metadata)

    tail = _offset_point(head, default_direction, default_length)
    tail[side_axis] = centerline
    return _ensure_non_zero_geometry(head, tail, placement_metadata)


def _placement_centerline(placement_metadata: dict) -> float:
    side_axis = int(placement_metadata["side_axis"]["index"])
    return float(placement_metadata["bbox_ground_center"][side_axis])


def _find_opposite_pelvis_geometry(source_bone_name: Optional[str], source_bones: Dict[str, dict]) -> Optional[dict]:
    if not source_bone_name:
        return None
    for candidate_name in _opposite_name_candidates(source_bone_name):
        if candidate_name in source_bones:
            return source_bones[candidate_name]
    return None


def _spec_style_side_suffix(name: str) -> str:
    """
    Rewrite a source-style L/R side suffix to ASAM-style _Left/_Right.

    ASAM OpenMATERIAL 3D §7.3.3.2: "The left and right side of the armature are
    indicated with the keywords 'Left' and 'Right' as a suffix." Extension bones
    that mirror a left/right pair should follow the same convention so consumer
    tooling treats them uniformly with the 28 core targets.

    Returns the input unchanged when no recognizable side suffix is present or
    when the name already uses the spec-style suffix.
    """
    if name.endswith("_Left") or name.endswith("_Right"):
        return name
    replacements = [
        (".L", "_Left"),
        (".R", "_Right"),
        ("_L", "_Left"),
        ("_R", "_Right"),
        ("-L", "_Left"),
        ("-R", "_Right"),
    ]
    for src_suffix, target_suffix in replacements:
        if name.endswith(src_suffix):
            return name[: -len(src_suffix)] + target_suffix
    return name


def _opposite_name_candidates(name: str) -> List[str]:
    replacements = [
        (".L", ".R"),
        (".R", ".L"),
        ("_L", "_R"),
        ("_R", "_L"),
        ("-L", "-R"),
        ("-R", "-L"),
        ("Left", "Right"),
        ("Right", "Left"),
        ("left", "right"),
        ("right", "left"),
    ]
    candidates: List[str] = []
    for old, new in replacements:
        if old in name:
            candidates.append(name.replace(old, new))
    return candidates


def _resolve_root_geometry(
    root_resolution: dict,
    source_bones: Dict[str, dict],
    placement_metadata: dict,
    warnings: List[str],
) -> Tuple[dict, str, Optional[str]]:
    source_bone_name = root_resolution.get("source_bone")
    mode = root_resolution.get("mode")

    if mode == "reuse_existing_root":
        if source_bone_name in source_bones:
            return (
                copy.deepcopy(source_bones[source_bone_name]),
                "source_root",
                source_bone_name,
            )
        if source_bone_name:
            warnings.append("missing_source_root_geometry:{0}".format(source_bone_name))

    # Synthesize a fresh Root at bbox_ground_center (in source-world coords); the
    # builder's _to_grp_root_local rebase later moves this to (0, 0, 0) in
    # Grp_Root-local space. Tail extends along the up axis by a bbox-derived ratio.
    grp_root_local_origin = root_resolution.get("grp_root_local_origin") or placement_metadata.get(
        "bbox_ground_center"
    ) or [0.0, 0.0, 0.0]
    up_axis = root_resolution.get("up_axis") or placement_metadata.get("up_axis") or {
        "index": 2,
        "sign": 1,
    }
    up_index = int(up_axis["index"])
    up_sign = int(up_axis["sign"])
    bbox_height = max(float(placement_metadata.get("bbox_height", 0.0)), 1e-4)
    tail_length = bbox_height * ROOT_TAIL_HEIGHT_RATIO

    head = [float(v) for v in grp_root_local_origin]
    tail = list(head)
    tail[up_index] = head[up_index] + up_sign * tail_length

    geometry = _ensure_non_zero_geometry(head, tail, placement_metadata)
    return geometry, "root_resolution", source_bone_name


def _resolve_created_target_geometry(
    target_name: str,
    semantic_mapping: dict,
    root_resolution: dict,
    source_bones: Dict[str, dict],
    placement_metadata: dict,
    bone_parents: dict,
    children_map: Dict[str, List[str]],
    resolved_geometry: Dict[str, dict],
) -> Tuple[dict, str, Optional[str]]:
    opposite_target = _opposite_target_name(target_name)
    opposite_geometry = None
    if opposite_target is not None:
        opposite_geometry = _get_reference_geometry(
            opposite_target,
            semantic_mapping,
            root_resolution,
            source_bones,
            resolved_geometry,
            placement_metadata,
        )
    if opposite_geometry is not None:
        mirrored = _mirror_geometry(opposite_geometry, placement_metadata)
        return mirrored, "mirrored_opposite", None

    ancestor_target, ancestor_geometry = _nearest_ancestor_reference(
        target_name,
        semantic_mapping,
        root_resolution,
        source_bones,
        resolved_geometry,
        placement_metadata,
        bone_parents,
    )
    descendant_target, descendant_geometry = _nearest_descendant_reference(
        target_name,
        semantic_mapping,
        root_resolution,
        source_bones,
        resolved_geometry,
        placement_metadata,
        children_map,
    )

    default_length = _default_length_for_target(target_name, placement_metadata)
    default_direction = _default_direction_for_target(target_name, placement_metadata, ancestor_geometry)

    if ancestor_geometry is not None and descendant_geometry is not None:
        geometry = _ensure_non_zero_geometry(ancestor_geometry["tail"], descendant_geometry["head"], placement_metadata)
        if _distance(geometry["head"], geometry["tail"]) < 1e-5:
            geometry = _ensure_non_zero_geometry(
                geometry["head"],
                _offset_point(geometry["head"], default_direction, default_length),
                placement_metadata,
            )
        return geometry, "interpolated_chain", ancestor_target or descendant_target

    if ancestor_geometry is not None:
        head = list(ancestor_geometry["tail"])
        tail = _offset_point(head, default_direction, default_length)
        return _ensure_non_zero_geometry(head, tail, placement_metadata), "extrapolated_parent", ancestor_target

    if descendant_geometry is not None:
        tail = list(descendant_geometry["head"])
        head = _offset_point(tail, default_direction, -default_length)
        return _ensure_non_zero_geometry(head, tail, placement_metadata), "extrapolated_child", descendant_target

    head = list(placement_metadata.get("bbox_ground_center", [0.0, 0.0, 0.0]))
    tail = _offset_point(head, default_direction, default_length)
    return _ensure_non_zero_geometry(head, tail, placement_metadata), "placement_fallback", None


def _get_reference_geometry(
    target_name: str,
    semantic_mapping: dict,
    root_resolution: dict,
    source_bones: Dict[str, dict],
    resolved_geometry: Dict[str, dict],
    placement_metadata: dict,
) -> Optional[dict]:
    if target_name in resolved_geometry:
        return resolved_geometry[target_name]

    if target_name == "Root":
        geometry, _, _ = _resolve_root_geometry(root_resolution, source_bones, placement_metadata, [])
        return geometry

    payload = semantic_mapping[target_name]
    source_bone_name = payload.get("source_bone")
    if payload.get("action") in RECOVERABLE_ACTIONS and source_bone_name in source_bones:
        return source_bones[source_bone_name]
    return None


def _nearest_ancestor_reference(
    target_name: str,
    semantic_mapping: dict,
    root_resolution: dict,
    source_bones: Dict[str, dict],
    resolved_geometry: Dict[str, dict],
    placement_metadata: dict,
    bone_parents: dict,
) -> Tuple[Optional[str], Optional[dict]]:
    current = bone_parents.get(target_name)
    while current in bone_parents:
        geometry = _get_reference_geometry(current, semantic_mapping, root_resolution, source_bones, resolved_geometry, placement_metadata)
        if geometry is not None:
            return current, geometry
        current = bone_parents.get(current)
    return None, None


def _nearest_descendant_reference(
    target_name: str,
    semantic_mapping: dict,
    root_resolution: dict,
    source_bones: Dict[str, dict],
    resolved_geometry: Dict[str, dict],
    placement_metadata: dict,
    children_map: Dict[str, List[str]],
) -> Tuple[Optional[str], Optional[dict]]:
    queue = list(children_map.get(target_name, []))
    while queue:
        current = queue.pop(0)
        geometry = _get_reference_geometry(current, semantic_mapping, root_resolution, source_bones, resolved_geometry, placement_metadata)
        if geometry is not None:
            return current, geometry
        queue.extend(children_map.get(current, []))
    return None, None


def _opposite_target_name(target_name: str) -> Optional[str]:
    if target_name.endswith("_Left"):
        return target_name[:-5] + "_Right"
    if target_name.endswith("_Right"):
        return target_name[:-6] + "_Left"
    return None


def _mirror_geometry(reference_geometry: dict, placement_metadata: dict) -> dict:
    side_axis = int(placement_metadata["side_axis"]["index"])
    centerline = float(placement_metadata["bbox_ground_center"][side_axis])

    head = list(reference_geometry["head"])
    tail = list(reference_geometry["tail"])
    head[side_axis] = (2.0 * centerline) - head[side_axis]
    tail[side_axis] = (2.0 * centerline) - tail[side_axis]
    return copy.deepcopy(
        {
            "name": reference_geometry.get("name"),
            "parent": reference_geometry.get("parent"),
            "head": head,
            "tail": tail,
            "length": _distance(head, tail),
        }
    )


def _default_length_for_target(target_name: str, placement_metadata: dict) -> float:
    bbox_height = max(float(placement_metadata.get("bbox_height", 0.0)), 1e-4)
    ratio = DEFAULT_LENGTH_RATIOS.get(target_name, 0.08)
    return bbox_height * ratio


def _default_direction_for_target(target_name: str, placement_metadata: dict, ancestor_geometry: Optional[dict]) -> List[float]:
    if ancestor_geometry is not None and _family_label(target_name) in {"Hip", "Lower_Spine", "Upper_Spine", "Neck", "Head"}:
        vector = _direction_vector(ancestor_geometry["head"], ancestor_geometry["tail"])
        if _distance([0.0, 0.0, 0.0], vector) > 1e-6:
            return vector

    up_axis = int(placement_metadata["up_axis"]["index"])
    up_sign = int(placement_metadata["up_axis"]["sign"])
    side_axis = int(placement_metadata["side_axis"]["index"])
    side_sign = int(placement_metadata["side_axis"]["sign"])
    forward_axis = int(placement_metadata["forward_axis"]["index"])
    forward_sign = int(placement_metadata["forward_axis"]["sign"])
    lateral_scale = side_sign if target_name.endswith("_Left") else -side_sign if target_name.endswith("_Right") else 0

    vector = [0.0, 0.0, 0.0]
    family = _family_label(target_name)

    if family in {"Root", "Hip", "Lower_Spine", "Upper_Spine", "Neck", "Head"}:
        vector[up_axis] = float(up_sign)
    elif family == "Shoulder":
        vector[side_axis] = float(lateral_scale or side_sign)
    elif family in {"Upper_Arm", "Lower_Arm", "Hand"}:
        vector[side_axis] = float(lateral_scale or side_sign)
        vector[up_axis] = float(-0.35 * up_sign)
    elif family in {"Full_Thumb", "Full_Fingers"}:
        # Fingers/thumb extend sideward and slightly downward like the hand.
        vector[side_axis] = float(lateral_scale or side_sign)
        vector[up_axis] = float(-0.35 * up_sign)
    elif family in {"Upper_Leg", "Lower_Leg"}:
        vector[up_axis] = float(-up_sign)
    elif family == "Foot":
        vector[forward_axis] = float(forward_sign)
    elif family == "Full_Toes":
        # Toes extend forward from the foot.
        vector[forward_axis] = float(forward_sign)
    elif family == "Eye":
        # Eyes point forward (gaze direction) per ASAM §7.3.3.3.10.
        vector[forward_axis] = float(forward_sign)
    else:
        vector[up_axis] = float(up_sign)

    return _normalize(vector)


def _family_label(target_name: str) -> str:
    if target_name.endswith("_Left"):
        return target_name[:-5]
    if target_name.endswith("_Right"):
        return target_name[:-6]
    return target_name


def _direction_vector(head: Sequence[float], tail: Sequence[float]) -> List[float]:
    return _normalize([float(tail[index] - head[index]) for index in range(3)])


def _normalize(vector: Sequence[float]) -> List[float]:
    length = math.sqrt(sum(component * component for component in vector))
    if length <= 1e-8:
        return [0.0, 0.0, 1.0]
    return [float(component / length) for component in vector]


def _offset_point(point: Sequence[float], direction: Sequence[float], distance: float) -> List[float]:
    return [float(point[index] + (direction[index] * distance)) for index in range(3)]


def _ensure_non_zero_geometry(head: Sequence[float], tail: Sequence[float], placement_metadata: dict) -> dict:
    fixed_head = [float(value) for value in head]
    fixed_tail = [float(value) for value in tail]
    if _distance(fixed_head, fixed_tail) <= 1e-6:
        direction = _default_direction_for_target("Root", placement_metadata, None)
        fixed_tail = _offset_point(fixed_head, direction, max(float(placement_metadata.get("bbox_height", 0.0)) * 0.01, 1e-3))

    return {
        "head": fixed_head,
        "tail": fixed_tail,
        "length": _distance(fixed_head, fixed_tail),
    }


def _distance(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    return math.sqrt(sum((point_b[index] - point_a[index]) ** 2 for index in range(3)))


def _resolve_preserved_source_root_extra(
    root_resolution: dict,
    source_bones: Dict[str, dict],
) -> Optional[dict]:
    """When mode == create_new_root and the source root is preservable, return
    its geometry as a sibling-extra bone payload (in source-world coords). The
    caller is responsible for the Grp_Root-local rebase. Returns None when the
    conditions are not met.
    """
    if root_resolution.get("mode") != "create_new_root":
        return None
    if not root_resolution.get("preserve_source_root_as_extra"):
        return None
    source_name = root_resolution.get("source_bone")
    if not source_name or source_name not in source_bones:
        return None
    source = source_bones[source_name]
    generated_name = source_name if source_name != "Root" else "Root_Source"
    return {
        "name": generated_name,
        "head": [float(v) for v in source["head"]],
        "tail": [float(v) for v in source["tail"]],
        "geometry_source": "preserved_source_root",
        "source_bone": source_name,
        "semantic_action": "preserve_extra",
    }


def _asam_roll_align_axis(placement_metadata: dict) -> List[float]:
    """World-space unit vector that a bone's local Z should align to under ASAM.

    ASAM OpenMATERIAL human geometry: the bone y-axis follows the bone direction and
    the z-axis points sidewards (x forward for upright bones, x up for foot/toe bones,
    which follows automatically). Aligning every generated bone's local Z to the world
    side axis reproduces that convention uniformly. Returns a unit vector along the
    placement side axis, signed by side_axis['sign'].
    """
    side = placement_metadata["side_axis"]
    index = int(side["index"])
    sign = float(side.get("sign", 1)) or 1.0
    axis = [0.0, 0.0, 0.0]
    axis[index] = 1.0 if sign >= 0 else -1.0
    return axis

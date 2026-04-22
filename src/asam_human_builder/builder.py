"""
Pure-Python planning layer for Blender-side ASAM human armature construction.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from phase3_classifier.classifier import CORE_TARGETS


RECOVERABLE_ACTIONS = {"direct_map", "alias_map", "repair_in_builder"}
GENERATED_MARKER_KEY = "open_jaywalker_generated"
GENERATED_ASSET_KEY = "open_jaywalker_asset"

REQUIRED_REPORT_FIELDS = {
    "recommended_primary_armature",
    "semantic_mapping",
}
REQUIRED_PLAN_FIELDS = {
    "asset_name",
    "recommended_primary_armature",
    "root_resolution",
    "placement_metadata",
    "proposed_asam_hierarchy",
}

DEFAULT_LENGTH_RATIOS = {
    "Hip": 0.05,
    "Lower_Spine": 0.10,
    "Upper_Spine": 0.12,
    "Neck": 0.05,
    "Head": 0.08,
    "Shoulder_Left": 0.06,
    "Shoulder_Right": 0.06,
    "Upper_Arm_Left": 0.16,
    "Upper_Arm_Right": 0.16,
    "Lower_Arm_Left": 0.15,
    "Lower_Arm_Right": 0.15,
    "Hand_Left": 0.07,
    "Hand_Right": 0.07,
    "Upper_Leg_Left": 0.22,
    "Upper_Leg_Right": 0.22,
    "Lower_Leg_Left": 0.22,
    "Lower_Leg_Right": 0.22,
    "Foot_Left": 0.10,
    "Foot_Right": 0.10,
}


def load_builder_inputs(asset_dir: Path) -> Tuple[dict, dict]:
    """Load classifier and build-plan JSON inputs for the builder."""
    asset_dir = Path(asset_dir).resolve()
    report_path = asset_dir / "classifier_report.json"
    plan_path = asset_dir / "build_plan.json"

    if not report_path.exists():
        raise FileNotFoundError("Missing classifier_report.json in {0}".format(asset_dir))
    if not plan_path.exists():
        raise FileNotFoundError("Missing build_plan.json in {0}".format(asset_dir))

    with report_path.open("r", encoding="utf-8") as handle:
        classifier_report = json.load(handle)
    with plan_path.open("r", encoding="utf-8") as handle:
        build_plan = json.load(handle)

    validate_builder_inputs(classifier_report, build_plan)
    return classifier_report, build_plan


def validate_builder_inputs(classifier_report: dict, build_plan: dict) -> None:
    """Fail fast when required builder inputs are absent."""
    missing_report = REQUIRED_REPORT_FIELDS.difference(classifier_report.keys())
    missing_plan = REQUIRED_PLAN_FIELDS.difference(build_plan.keys())

    if missing_report:
        raise ValueError("classifier_report is missing required fields: {0}".format(", ".join(sorted(missing_report))))
    if missing_plan:
        raise ValueError("build_plan is missing required fields: {0}".format(", ".join(sorted(missing_plan))))

    if not isinstance(classifier_report.get("semantic_mapping"), dict):
        raise ValueError("classifier_report.semantic_mapping must be a dictionary")
    if not isinstance(build_plan.get("root_resolution"), dict):
        raise ValueError("build_plan.root_resolution must be a dictionary")
    if not isinstance(build_plan.get("placement_metadata"), dict):
        raise ValueError("build_plan.placement_metadata must be a dictionary")
    if not isinstance(build_plan.get("proposed_asam_hierarchy"), dict):
        raise ValueError("build_plan.proposed_asam_hierarchy must be a dictionary")

    missing_targets = [target for target in CORE_TARGETS if target not in classifier_report["semantic_mapping"]]
    if missing_targets:
        raise ValueError("classifier_report.semantic_mapping is missing targets: {0}".format(", ".join(missing_targets)))

    bone_parents = build_plan["proposed_asam_hierarchy"].get("bone_parents")
    if not isinstance(bone_parents, dict):
        raise ValueError("build_plan.proposed_asam_hierarchy.bone_parents must be a dictionary")
    missing_parent_targets = [target for target in CORE_TARGETS if target not in bone_parents]
    if missing_parent_targets:
        raise ValueError(
            "build_plan.proposed_asam_hierarchy.bone_parents is missing targets: {0}".format(
                ", ".join(missing_parent_targets)
            )
        )

    if "source_translation_offset" in build_plan["root_resolution"]:
        offset = build_plan["root_resolution"]["source_translation_offset"]
        if not isinstance(offset, (list, tuple)) or len(offset) != 3:
            raise ValueError(
                "build_plan.root_resolution.source_translation_offset must be a length-3 sequence"
            )
        try:
            [float(value) for value in offset]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "build_plan.root_resolution.source_translation_offset entries must be numeric"
            ) from exc


def choose_generated_collection_action(existing_collections: Sequence[dict], expected_name: str, asset_name: str) -> str:
    """
    Decide whether the generated ASAM collection should be created or rebuilt.

    Raises when an unmarked collection already uses the reserved generated name.
    """
    for collection in existing_collections:
        if collection.get("name") != expected_name:
            continue
        if collection.get("generated") and collection.get("asset_name") == asset_name:
            return "rebuild"
        raise ValueError(
            "Collection name conflict for {0}; existing collection is not a safe generated output".format(
                expected_name
            )
        )
    return "create"


def build_source_bone_index_from_export(asset_dir: Path, armature_name: str) -> Dict[str, dict]:
    """Load source bone geometry from the exported inspector JSON for tests/offline planning."""
    asset_dir = Path(asset_dir).resolve()
    candidates = [
        asset_dir / "{0}_all.json".format(armature_name),
        asset_dir / "{0}_filtered.json".format(armature_name),
    ]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return {
                bone["name"]: {
                    "name": bone["name"],
                    "parent": bone.get("parent"),
                    "head": [float(value) for value in bone.get("head", (0.0, 0.0, 0.0))],
                    "tail": [float(value) for value in bone.get("tail", (0.0, 0.0, 0.0))],
                    "length": float(bone.get("length", 0.0)),
                }
                for bone in payload.get("bones", [])
            }

    raise FileNotFoundError("No exported armature JSON found for {0} in {1}".format(armature_name, asset_dir))


def resolve_default_asset_dir(blend_filepath: Optional[str], builder_dir: Optional[Path] = None) -> Path:
    """Resolve the default asset output folder from the open Blender file path."""
    builder_dir = Path(builder_dir or Path(__file__).resolve().parent)
    asset_name = Path(blend_filepath).stem if blend_filepath else "unsaved"
    return (builder_dir.parent / "armature_inspector" / "output" / asset_name).resolve()


def build_armature_spec_from_asset_dir(asset_dir: Path) -> Tuple[dict, dict, dict]:
    """Load builder inputs from disk and return the resolved armature spec."""
    classifier_report, build_plan = load_builder_inputs(asset_dir)
    source_bones = build_source_bone_index_from_export(asset_dir, build_plan["recommended_primary_armature"])
    return classifier_report, build_plan, build_armature_spec(classifier_report, build_plan, source_bones)


def build_armature_spec(classifier_report: dict, build_plan: dict, source_bones: Dict[str, dict]) -> dict:
    """
    Build a deterministic ASAM armature creation spec from classifier outputs.
    """
    validate_builder_inputs(classifier_report, build_plan)

    asset_name = build_plan["asset_name"]
    semantic_mapping = classifier_report["semantic_mapping"]
    root_resolution = build_plan["root_resolution"]
    placement_metadata = build_plan["placement_metadata"]
    bone_parents = build_plan["proposed_asam_hierarchy"]["bone_parents"]
    children_map = _build_children_map(bone_parents)

    if classifier_report["recommended_primary_armature"] != build_plan["recommended_primary_armature"]:
        raise ValueError("Classifier and build plan disagree on the recommended primary armature")

    source_translation_offset = [
        float(value)
        for value in root_resolution.get("source_translation_offset", [0.0, 0.0, 0.0])
    ]
    translated_source_bones = _translate_source_bones(source_bones, source_translation_offset)

    spec = {
        "asset_name": asset_name,
        "source_armature_name": build_plan["recommended_primary_armature"],
        "generated_collection_name": "ASAM_{0}".format(asset_name),
        "group_root_name": "Grp_Root",
        "generated_armature_name": "Armature_{0}".format(asset_name),
        "root_resolution": copy.deepcopy(root_resolution),
        "placement_metadata": copy.deepcopy(placement_metadata),
        "extras_preserved": copy.deepcopy(build_plan.get("extras_preserved", [])),
        "source_translation_offset": list(source_translation_offset),
        "bones": [],
        "warnings": [],
    }

    resolved_geometry: Dict[str, dict] = {}

    for target_name in CORE_TARGETS:
        geometry, geometry_source, source_bone_name = _resolve_target_geometry(
            target_name,
            semantic_mapping,
            root_resolution,
            translated_source_bones,
            placement_metadata,
            bone_parents,
            children_map,
            resolved_geometry,
            spec["warnings"],
        )
        resolved_geometry[target_name] = geometry
        parent_name = bone_parents.get(target_name)
        spec["bones"].append(
            {
                "name": target_name,
                "parent_bone": parent_name if parent_name in bone_parents else None,
                "head": geometry["head"],
                "tail": geometry["tail"],
                "use_connect": False,
                "geometry_source": geometry_source,
                "source_bone": source_bone_name,
                "semantic_action": semantic_mapping[target_name]["action"],
            }
        )

    return spec


def build_builder_report(build_spec: dict, execution_result: dict) -> dict:
    """Summarize a finished builder run for traceability."""
    source_targets = []
    created_targets = []
    for bone in build_spec["bones"]:
        if bone["geometry_source"] in {"source_bone", "source_root"}:
            source_targets.append(bone["name"])
        else:
            created_targets.append(bone["name"])

    return {
        "asset_name": build_spec["asset_name"],
        "source_armature_name": build_spec["source_armature_name"],
        "generated_collection_name": execution_result["generated_collection_name"],
        "group_root_name": execution_result["group_root_name"],
        "generated_armature_name": execution_result["generated_armature_name"],
        "built_core_targets": [bone["name"] for bone in build_spec["bones"]],
        "targets_from_source_geometry": source_targets,
        "targets_created_heuristically": created_targets,
        "preserved_extras_count": len(build_spec.get("extras_preserved", [])),
        "warnings": list(build_spec.get("warnings", [])),
    }


def write_builder_report(asset_dir: Path, build_spec: dict, execution_result: dict) -> Tuple[dict, Path]:
    """Write the builder report into the asset output folder."""
    asset_dir = Path(asset_dir).resolve()
    asset_dir.mkdir(parents=True, exist_ok=True)
    builder_report = build_builder_report(build_spec, execution_result)
    report_path = asset_dir / "builder_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(builder_report, handle, indent=2)
        handle.write("\n")
    return builder_report, report_path


def print_builder_summary(builder_report: dict, report_path: Path) -> None:
    """Print a compact builder summary for Blender/VSCode console usage."""
    print("ASAM Human Builder summary")
    print("Asset: {0}".format(builder_report["asset_name"]))
    print("Source armature: {0}".format(builder_report["source_armature_name"]))
    print("Generated collection: {0}".format(builder_report["generated_collection_name"]))
    print("Generated armature: {0}".format(builder_report["generated_armature_name"]))
    print("Built core targets: {0}".format(", ".join(builder_report["built_core_targets"])))
    print("From source geometry: {0}".format(", ".join(builder_report["targets_from_source_geometry"]) or "(none)"))
    print(
        "Created heuristically: {0}".format(
            ", ".join(builder_report["targets_created_heuristically"]) or "(none)"
        )
    )
    print("Preserved extras count: {0}".format(builder_report["preserved_extras_count"]))
    if builder_report["warnings"]:
        print("Warnings: {0}".format(", ".join(builder_report["warnings"])))
    print("Builder report written to: {0}".format(report_path))


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

    head = list(root_geometry["tail"])
    head[side_axis] = centerline

    if lower_spine_geometry is not None:
        tail = list(lower_spine_geometry["head"])
        tail[side_axis] = centerline
        return _ensure_non_zero_geometry(head, tail, placement_metadata)

    opposite_geometry = _find_opposite_pelvis_geometry(source_bone_name, source_bones)
    source_geometry = source_bones.get(source_bone_name) if source_bone_name in source_bones else None
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


def _translate_source_bones(source_bones: Dict[str, dict], offset: Sequence[float]) -> Dict[str, dict]:
    offset_values = [float(offset[index]) for index in range(3)]
    translated = copy.deepcopy(source_bones)
    for bone in translated.values():
        bone["head"] = [float(bone["head"][index]) + offset_values[index] for index in range(3)]
        bone["tail"] = [float(bone["tail"][index]) + offset_values[index] for index in range(3)]
    return translated


def _resolve_root_geometry(
    root_resolution: dict,
    source_bones: Dict[str, dict],
    placement_metadata: dict,
    warnings: List[str],
) -> Tuple[dict, str, Optional[str]]:
    source_bone_name = root_resolution.get("source_bone")

    if root_resolution.get("mode") == "reuse_existing_root" and source_bone_name in source_bones:
        return copy.deepcopy(source_bones[source_bone_name]), "source_root", source_bone_name

    if root_resolution.get("mode") == "reuse_existing_root" and source_bone_name:
        warnings.append("missing_source_root_geometry:{0}".format(source_bone_name))

    target_head = root_resolution.get("target_head") or placement_metadata.get("bbox_ground_center") or [0.0, 0.0, 0.0]
    target_tail = root_resolution.get("target_tail")
    if target_tail is None:
        bbox_height = max(float(placement_metadata.get("bbox_height", 0.0)), 1e-4)
        direction = _default_direction_for_target("Root", placement_metadata, None)
        target_tail = _offset_point(target_head, direction, bbox_height * 0.18)

    return _ensure_non_zero_geometry(target_head, target_tail, placement_metadata), "root_resolution", source_bone_name


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


def _build_children_map(bone_parents: dict) -> Dict[str, List[str]]:
    children_map = {target: [] for target in bone_parents}
    for target, parent in bone_parents.items():
        if parent in children_map:
            children_map[parent].append(target)
    return children_map


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
    elif family in {"Upper_Leg", "Lower_Leg"}:
        vector[up_axis] = float(-up_sign)
    elif family == "Foot":
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

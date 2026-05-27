"""
Pure-Python planning layer for Blender-side ASAM human armature construction.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from phase3_classifier.classifier import CORE_TARGETS

from asam_human_builder.geometry_resolution import (
    _hip_requires_centered_pelvis_pair,
    _opposite_name_candidates,
    _resolve_preserved_source_root_extra,
    _resolve_target_geometry,
    _spec_style_side_suffix,
)


GENERATED_MARKER_KEY = "open_jaywalker_generated"
GENERATED_ASSET_KEY = "open_jaywalker_asset"

# build_spec["bones"][i]["geometry_source"] values where "source_bone" carries a real
# source bone name suitable for vertex group remapping. Synthesized values (mirrored,
# interpolated, extrapolated, placement_fallback) instead hold an ASAM target name
# in source_bone and must be skipped by the remap planner.
#
# "centered_pelvis_pair" is intentionally absent: when Hip is built via paired-pelvis
# centering, both source pelvis bones are preserved as separate children of Hip
# (see _resolve_preserved_pelvis_pair), and the synthetic Hip must not steal either
# side's vertex group.
SOURCE_NAMED_GEOMETRY_SOURCES = frozenset(
    {"source_bone", "source_root", "root_resolution"}
)

REQUIRED_REPORT_FIELDS = {
    "recommended_primary_armature",
    "semantic_mapping",
}
REQUIRED_PLAN_FIELDS = {
    "asset_name",
    "recommended_primary_armature",
    "root_resolutions",
    "placement_metadata",
    "mesh_binding",
    "proposed_asam_hierarchy",
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
    root_resolutions = build_plan.get("root_resolutions")
    if not isinstance(root_resolutions, list) or not root_resolutions:
        raise ValueError("build_plan.root_resolutions must be a non-empty list")
    if not isinstance(root_resolutions[0], dict):
        raise ValueError("build_plan.root_resolutions[0] must be a dictionary")
    if len(root_resolutions) > 1:
        import warnings as _warnings
        _warnings.warn(
            "build_plan.root_resolutions length > 1 not yet supported; using index 0",
            stacklevel=2,
        )
    if not isinstance(build_plan.get("placement_metadata"), dict):
        raise ValueError("build_plan.placement_metadata must be a dictionary")
    if not isinstance(build_plan.get("mesh_binding"), dict):
        raise ValueError("build_plan.mesh_binding must be a dictionary")
    if build_plan["mesh_binding"].get("armature_object_name") != build_plan.get("recommended_primary_armature"):
        raise ValueError("build_plan.mesh_binding.armature_object_name must match recommended_primary_armature")
    if not isinstance(build_plan["mesh_binding"].get("meshes"), list):
        raise ValueError("build_plan.mesh_binding.meshes must be a list")
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

    grp_root_local_origin = root_resolutions[0].get("grp_root_local_origin")
    if (
        not isinstance(grp_root_local_origin, (list, tuple))
        or len(grp_root_local_origin) != 3
    ):
        raise ValueError(
            "build_plan.root_resolutions[0].grp_root_local_origin must be a length-3 numeric list"
        )
    try:
        [float(value) for value in grp_root_local_origin]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "build_plan.root_resolutions[0].grp_root_local_origin entries must be numeric"
        ) from exc


def compute_vertex_group_remap_plan(
    bones: Sequence[dict],
    vertex_group_names: Sequence[str],
) -> Dict[str, list]:
    """
    Plan how to rename a duplicated mesh's vertex groups so that they match the
    generated ASAM armature's bone names.

    For each spec bone whose geometry source carries a real source bone name
    (see SOURCE_NAMED_GEOMETRY_SOURCES), the source bone name on the source mesh's
    vertex group should be renamed to the ASAM target bone name. Identity renames
    (source name already equals target) are skipped. A rename is suppressed when
    the target name already exists as a separate vertex group on the mesh, to
    avoid clobbering weights; the conflict is reported instead.

    Args:
        bones: build_spec["bones"] entries (each has "name", "source_bone",
            "geometry_source"). May include synthesized bones; those are skipped.
        vertex_group_names: existing vertex group names on the duplicated mesh.

    Returns:
        {
            "renames": [{"source": ..., "target": ...}, ...] sorted by source name,
            "unmapped_groups": sorted list of group names with no ASAM target,
            "asam_targets_without_source_group": sorted list of ASAM targets whose
                expected source vertex group is not on the mesh,
            "name_collisions": sorted list of {"source", "target", "existing_group"}
                where rename would clobber an existing group of the target name.
        }
    """
    existing_groups = set(vertex_group_names)

    # Build source -> target map from bones whose source_bone is a real source name.
    source_to_target: Dict[str, str] = {}
    for bone in bones:
        source_name = bone.get("source_bone")
        target_name = bone.get("name")
        geometry_source = bone.get("geometry_source")
        if not source_name or not target_name:
            continue
        if geometry_source not in SOURCE_NAMED_GEOMETRY_SOURCES:
            continue
        if source_name == target_name:
            continue  # identity - no rename needed
        # Last-write-wins is deterministic because the caller's bones list is in
        # CORE_TARGETS order; collisions in source_to_target are not expected because
        # the classifier should not map one source bone to multiple ASAM targets.
        source_to_target.setdefault(source_name, target_name)

    renames: List[Dict[str, str]] = []
    collisions: List[Dict[str, str]] = []

    for group_name in sorted(existing_groups):
        if group_name not in source_to_target:
            continue
        target = source_to_target[group_name]
        if target in existing_groups and target != group_name:
            collisions.append(
                {"source": group_name, "target": target, "existing_group": target}
            )
            continue
        renames.append({"source": group_name, "target": target})

    # Unmapped: groups that are not in source_to_target AND are not already an ASAM
    # target name (a group already named after a generated bone is fine - Blender will
    # bind it directly to that bone).
    asam_target_names = {bone.get("name") for bone in bones if bone.get("name")}
    unmapped_groups = sorted(
        group_name
        for group_name in existing_groups
        if group_name not in source_to_target and group_name not in asam_target_names
    )

    # ASAM targets whose source vertex group is not present on this mesh.
    missing_targets = sorted(
        target
        for source_name, target in source_to_target.items()
        if source_name not in existing_groups
    )

    return {
        "renames": renames,
        "unmapped_groups": unmapped_groups,
        "asam_targets_without_source_group": missing_targets,
        "name_collisions": sorted(
            collisions,
            key=lambda entry: (entry["source"], entry["target"]),
        ),
    }


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
    """
    Resolve the default asset output folder from the open Blender file path.

    Delegates to pipeline_paths.resolve_asset_dir so the location honors the
    OPEN_JAYWALKER_OUTPUT_ROOT env override. The `builder_dir` argument is
    accepted for backwards compatibility with callers that pass it but is
    no longer used to compute the result.
    """
    del builder_dir  # signal that the argument is intentionally ignored
    asset_name = Path(blend_filepath).stem if blend_filepath else "unsaved"
    from pipeline_paths import resolve_asset_dir
    return resolve_asset_dir(asset_name)


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
    root_resolution = build_plan["root_resolutions"][0]
    placement_metadata = build_plan["placement_metadata"]
    bone_parents = build_plan["proposed_asam_hierarchy"]["bone_parents"]
    children_map = _build_children_map(bone_parents)

    if classifier_report["recommended_primary_armature"] != build_plan["recommended_primary_armature"]:
        raise ValueError("Classifier and build plan disagree on the recommended primary armature")

    grp_root_local_origin = [float(v) for v in root_resolution["grp_root_local_origin"]]

    spec = {
        "asset_name": asset_name,
        "source_armature_name": build_plan["recommended_primary_armature"],
        "generated_collection_name": "ASAM_{0}".format(asset_name),
        "group_root_name": "Grp_Root",
        "generated_armature_name": "Armature_{0}".format(asset_name),
        "root_resolution": copy.deepcopy(root_resolution),
        "placement_metadata": copy.deepcopy(placement_metadata),
        "mesh_binding": copy.deepcopy(build_plan["mesh_binding"]),
        "extras_preserved": copy.deepcopy(build_plan.get("extras_preserved", [])),
        "grp_root_local_origin": list(grp_root_local_origin),
        "bones": [],
        "preserved_pelvis_pair": [],
        "warnings": [],
    }

    resolved_geometry: Dict[str, dict] = {}

    for target_name in CORE_TARGETS:
        geometry, geometry_source, source_bone_name = _resolve_target_geometry(
            target_name,
            semantic_mapping,
            root_resolution,
            source_bones,
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
                "head": _to_grp_root_local(geometry["head"], grp_root_local_origin),
                "tail": _to_grp_root_local(geometry["tail"], grp_root_local_origin),
                "use_connect": False,
                "geometry_source": geometry_source,
                "source_bone": source_bone_name,
                "semantic_action": semantic_mapping[target_name]["action"],
            }
        )

    preserved_root = _resolve_preserved_source_root_extra(root_resolution, source_bones)
    if preserved_root is not None:
        spec["bones"].append(
            {
                "name": preserved_root["name"],
                "parent_bone": None,
                "head": _to_grp_root_local(preserved_root["head"], grp_root_local_origin),
                "tail": _to_grp_root_local(preserved_root["tail"], grp_root_local_origin),
                "use_connect": False,
                "geometry_source": preserved_root["geometry_source"],
                "source_bone": preserved_root["source_bone"],
                "semantic_action": preserved_root["semantic_action"],
            }
        )

    for entry in _resolve_preserved_pelvis_pair(
        semantic_mapping,
        source_bones,
        build_plan["mesh_binding"],
    ):
        spec["bones"].append(
            {
                "name": entry["generated_bone_name"],
                "parent_bone": "Hip",
                "head": _to_grp_root_local(entry["geometry"]["head"], grp_root_local_origin),
                "tail": _to_grp_root_local(entry["geometry"]["tail"], grp_root_local_origin),
                "use_connect": False,
                "geometry_source": "source_bone",
                "source_bone": entry["source_bone_name"],
                "semantic_action": "preserve_paired_pelvis",
            }
        )
        spec["preserved_pelvis_pair"].append(
            {
                "source_bone_name": entry["source_bone_name"],
                "generated_bone_name": entry["generated_bone_name"],
                "parent": "Hip",
            }
        )

    return spec


def build_builder_report(build_spec: dict, execution_result: dict) -> dict:
    """Summarize a finished builder run for traceability."""
    core_target_names = set(CORE_TARGETS)
    source_targets = []
    created_targets = []
    for bone in build_spec["bones"]:
        if bone["name"] not in core_target_names:
            continue
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
        "collection_action": execution_result.get("collection_action"),
        "duplicated_meshes": copy.deepcopy(execution_result.get("duplicated_meshes", [])),
        "skipped_meshes": copy.deepcopy(execution_result.get("skipped_meshes", [])),
        "mesh_warnings": list(execution_result.get("mesh_warnings", [])),
        "built_core_targets": [bone["name"] for bone in build_spec["bones"] if bone["name"] in core_target_names],
        "targets_from_source_geometry": source_targets,
        "targets_created_heuristically": created_targets,
        "preserved_extras_count": len(build_spec.get("extras_preserved", [])),
        "preserved_pelvis_pair": copy.deepcopy(build_spec.get("preserved_pelvis_pair", [])),
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
    print("Collection action: {0}".format(builder_report.get("collection_action") or "(unknown)"))
    duplicated_meshes = builder_report.get("duplicated_meshes", [])
    print("Duplicated meshes: {0}".format(len(duplicated_meshes)))
    if duplicated_meshes:
        print(
            "Generated mesh copies: {0}".format(
                ", ".join(mesh["generated_mesh_name"] for mesh in duplicated_meshes)
            )
        )
    print("Built core targets: {0}".format(", ".join(builder_report["built_core_targets"])))
    print("From source geometry: {0}".format(", ".join(builder_report["targets_from_source_geometry"]) or "(none)"))
    print(
        "Created heuristically: {0}".format(
            ", ".join(builder_report["targets_created_heuristically"]) or "(none)"
        )
    )
    print("Preserved extras count: {0}".format(builder_report["preserved_extras_count"]))
    preserved_pair = builder_report.get("preserved_pelvis_pair", [])
    if preserved_pair:
        print(
            "Preserved pelvis pair: {0}".format(
                ", ".join(entry["generated_bone_name"] for entry in preserved_pair)
            )
        )
    if builder_report["warnings"]:
        print("Warnings: {0}".format(", ".join(builder_report["warnings"])))
    print("Builder report written to: {0}".format(report_path))


def _bone_has_skin_weight(source_bone_name: str, mesh_binding: dict) -> bool:
    """
    Return True iff any mesh in mesh_binding has a vertex group named
    `source_bone_name` with weighted_vertex_count > 0.

    Tolerates missing keys: empty mesh_binding, missing vertex_group_stats,
    missing per_group, and missing weighted_vertex_count all yield False
    rather than raise. The inspector always populates the per_group list,
    so missing entries mean "not weighted on this mesh" and not "unknown".
    """
    if not source_bone_name or not isinstance(mesh_binding, dict):
        return False
    for mesh in mesh_binding.get("meshes", []) or []:
        stats = mesh.get("vertex_group_stats") or {}
        for group in stats.get("per_group", []) or []:
            if group.get("name") != source_bone_name:
                continue
            count = group.get("weighted_vertex_count")
            try:
                if int(count) > 0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _resolve_preserved_pelvis_pair(
    semantic_mapping: dict,
    source_bones: Dict[str, dict],
    mesh_binding: dict,
) -> List[dict]:
    """
    When Hip is built via paired-pelvis centering, preserve both source pelvis bones
    as children of the synthetic Hip so their vertex weights are not destroyed.

    A pelvis side is only preserved when at least one mesh in mesh_binding records
    weighted_vertex_count > 0 for that source bone — unweighted extras would clutter
    the generated armature without deforming anything.

    Returns a list of {"source_bone_name", "generated_bone_name", "geometry"} entries
    in deterministic (sorted) order. Empty list when the case does not apply or no
    side carries skin weight.
    """
    hip_payload = semantic_mapping.get("Hip", {})
    if not _hip_requires_centered_pelvis_pair(hip_payload):
        return []

    primary_name = hip_payload.get("source_bone")
    if not primary_name or primary_name not in source_bones:
        return []

    names: List[str] = [primary_name]
    for candidate in _opposite_name_candidates(primary_name):
        if candidate in source_bones and candidate not in names:
            names.append(candidate)
            break

    weighted_names = [name for name in names if _bone_has_skin_weight(name, mesh_binding)]
    entries = [
        {
            "source_bone_name": name,
            "generated_bone_name": _spec_style_side_suffix(name),
            "geometry": copy.deepcopy(source_bones[name]),
        }
        for name in weighted_names
    ]
    entries.sort(key=lambda entry: entry["source_bone_name"])
    return entries


def _to_grp_root_local(point: Sequence[float], origin: Sequence[float]) -> List[float]:
    """Translate a point from source-world coordinates into Grp_Root-local coordinates."""
    return [float(point[i]) - float(origin[i]) for i in range(3)]


def _build_children_map(bone_parents: dict) -> Dict[str, List[str]]:
    children_map = {target: [] for target in bone_parents}
    for target, parent in bone_parents.items():
        if parent in children_map:
            children_map[parent].append(target)
    return children_map



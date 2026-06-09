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
    _asam_roll_align_axis,
    _default_length_for_target,
    _direction_vector,
    _distance,
    _hip_requires_centered_pelvis_pair,
    _offset_point,
    _opposite_name_candidates,
    _resolve_eye_geometry,
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
    """Load classifier and build-plan JSON inputs, then validate the flat shape.

    Delegates the path resolution + missing-file + JSON parsing to
    `read_builder_inputs`; the only thing this adds is `validate_builder_inputs`,
    so the flat-shape callers and the no-validate (crowd) callers share one loader.
    """
    classifier_report, build_plan = read_builder_inputs(asset_dir)
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


def is_crowd_plan(build_plan: dict) -> bool:
    """True iff the build plan carries a non-empty per-character `characters` array."""
    return bool(build_plan.get("characters"))


def wrapper_collection_name(asset_name: str) -> str:
    """Name of the crowd wrapper collection that holds all per-character child collections."""
    return "ASAM_{0}".format(asset_name)


def apply_character_naming(spec: dict, asset_name: str, character_id: str) -> None:
    """Override the generated collection/group-root/armature names in place.

    Object names are global in bpy.data, so each character needs a unique suffix.
    Mutates `spec` and returns None (single-purpose API; callers keep their own
    reference to `spec`).
    """
    spec["generated_collection_name"] = "ASAM_{0}_{1}".format(asset_name, character_id)
    spec["group_root_name"] = "Grp_Root_{0}".format(character_id)
    spec["generated_armature_name"] = "Armature_{0}_{1}".format(asset_name, character_id)


def read_builder_inputs(asset_dir: Path) -> Tuple[dict, dict]:
    """Read classifier_report.json and build_plan.json WITHOUT flat-shape validation.

    Crowd reports pop the top-level `semantic_mapping`, which `validate_builder_inputs`
    requires; callers that may receive a crowd asset must use this and branch on
    `is_crowd_plan` before validating the flat shape.
    """
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
    return classifier_report, build_plan


def slice_source_bones_for_character(full_index: Dict[str, dict], character_id: str) -> Dict[str, dict]:
    """Filter a raw source-bone index to one character and strip its prefix from names.

    `full_index` is keyed by original bone names (from build_source_bone_index_from_export);
    the result is keyed by prefix-stripped names matching the per-character semantic_mapping.
    World geometry (head/tail/length) is preserved so each rig lands at its world location.
    """
    from phase3_classifier.segmentation import derive_character_prefix, strip_character_prefix

    sliced: Dict[str, dict] = {}
    for original_name, geometry in full_index.items():
        if derive_character_prefix(original_name) != character_id:
            continue
        stripped = strip_character_prefix(original_name, character_id)
        new_geometry = dict(geometry)
        new_geometry["name"] = stripped
        parent = geometry.get("parent")
        if parent and derive_character_prefix(parent) == character_id:
            new_geometry["parent"] = strip_character_prefix(parent, character_id)
        else:
            new_geometry["parent"] = None
        sliced[stripped] = new_geometry
    return sliced


def _find_character_entry(entries, character_id: str) -> dict:
    for entry in entries or []:
        if entry.get("character_id") == character_id:
            return entry
    raise KeyError("No character entry for {0}".format(character_id))


def build_character_flat_inputs(
    classifier_report: dict, build_plan: dict, character_id: str
) -> Tuple[dict, dict]:
    """Synthesize a flat-shaped (classifier_report, build_plan) pair for one character.

    The synthesized pair is fed to the unchanged single-character `build_armature_spec`.
    `recommended_primary_armature` is the real crowd source armature so mesh retargeting
    and flat validation line up; per-character mesh_binding.armature_object_name already
    equals it after the segmentation fix.
    """
    decomposition = classifier_report["asset_summary"]["character_decomposition"]
    source_armature = decomposition["source_armature"]
    report_entry = _find_character_entry(classifier_report.get("characters"), character_id)
    plan_entry = _find_character_entry(build_plan.get("characters"), character_id)

    flat_report = {
        "recommended_primary_armature": source_armature,
        "semantic_mapping": report_entry["semantic_mapping"],
    }
    flat_plan = {
        "asset_name": build_plan["asset_name"],
        "recommended_primary_armature": source_armature,
        "root_resolutions": plan_entry["root_resolutions"],
        "placement_metadata": plan_entry["placement_metadata"],
        "mesh_binding": plan_entry["mesh_binding"],
        "proposed_asam_hierarchy": plan_entry["proposed_asam_hierarchy"],
        "extras_preserved": plan_entry.get("extras_preserved", []),
    }
    return flat_report, flat_plan


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


# ASAM core-chain spanning (#57, #59): each bone's tail is set to its single core child's
# head so the spine AND limbs render as connected, upright/extended chains instead of copying
# short source "link" geometry. Head (no core child) extends along the up axis. Each listed
# bone has exactly one core child, so there is no multi-child ambiguity. Heads (joints) never
# move, so rest-pose deformation is unchanged.
_SPAN_CHILD = {
    "Hip": "Lower_Spine",
    "Lower_Spine": "Upper_Spine",
    "Upper_Spine": "Neck",
    "Neck": "Head",
    "Shoulder_Left": "Upper_Arm_Left",
    "Upper_Arm_Left": "Lower_Arm_Left",
    "Lower_Arm_Left": "Hand_Left",
    "Shoulder_Right": "Upper_Arm_Right",
    "Upper_Arm_Right": "Lower_Arm_Right",
    "Lower_Arm_Right": "Hand_Right",
    "Upper_Leg_Left": "Lower_Leg_Left",
    "Lower_Leg_Left": "Foot_Left",
    "Foot_Left": "Full_Toes_Left",
    "Upper_Leg_Right": "Lower_Leg_Right",
    "Lower_Leg_Right": "Foot_Right",
    "Foot_Right": "Full_Toes_Right",
}


def _compute_weight_merges(spec_bones, source_bones):
    """Map each orphaned source deform bone to the nearest ancestor that maps to an ASAM
    target (#58), so the builder can merge their vertex weights instead of orphaning them.

    Walks up the source-bone hierarchy (source_bones[...]["parent"]). A source bone that is
    itself a mapped ASAM source, or has no mapped ancestor, produces no entry. Returns a
    sorted list of {"source", "target"}.
    """
    source_to_target = {}
    for bone in spec_bones:
        source_name = bone.get("source_bone")
        target_name = bone.get("name")
        if (
            source_name
            and target_name
            and bone.get("geometry_source") in SOURCE_NAMED_GEOMETRY_SOURCES
            and source_name != target_name
        ):
            source_to_target.setdefault(source_name, target_name)

    merges = []
    for name, info in source_bones.items():
        if name in source_to_target:
            continue
        parent = info.get("parent")
        seen = set()
        while parent and parent not in seen:
            seen.add(parent)
            if parent in source_to_target:
                merges.append({"source": name, "target": source_to_target[parent]})
                break
            parent = (source_bones.get(parent) or {}).get("parent")
    return sorted(merges, key=lambda m: (m["source"], m["target"]))


def _retarget_tail(resolved_geometry, spec_bones, target, new_tail, grp_root_local_origin):
    """Rewrite one bone's tail (and the rebased spec tail), leaving its head/joint fixed.
    Skips a near-zero-length result. Shared by the spanning and terminal-orientation passes."""
    geom = resolved_geometry[target]
    if _distance(geom["head"], new_tail) <= 1e-6:
        return
    geom["tail"] = [float(v) for v in new_tail]
    geom["length"] = _distance(geom["head"], geom["tail"])
    local_tail = _to_grp_root_local(geom["tail"], grp_root_local_origin)
    for bone in spec_bones:
        if bone["name"] == target:
            bone["tail"] = local_tail
            break


def _span_core_chains(resolved_geometry, spec_bones, placement_metadata, grp_root_local_origin):
    """Set core-chain bone tails to their child's head so the spine and limbs run upright.

    Heads (joints) never move, which keeps rest-pose deformation unchanged; only tails are
    rewritten. Head, having no core child, extends along the up axis by its default length.
    A near-zero-length result (child head coincident with this bone's head) is skipped.
    """
    up = placement_metadata["up_axis"]
    up_index = int(up["index"])
    # Trust the upstream validator's sign (consistent with geometry_resolution); a
    # degenerate 0 yields a no-op Head extension caught by the zero-length guard.
    up_sign = int(up.get("sign", 1))

    for target, child in _SPAN_CHILD.items():
        if target in resolved_geometry and child in resolved_geometry:
            _retarget_tail(resolved_geometry, spec_bones, target, list(resolved_geometry[child]["head"]), grp_root_local_origin)

    if "Head" in resolved_geometry:
        head_geom = resolved_geometry["Head"]
        length = _default_length_for_target("Head", placement_metadata)
        new_tail = list(head_geom["head"])
        new_tail[up_index] = head_geom["head"][up_index] + (up_sign * length)
        _retarget_tail(resolved_geometry, spec_bones, "Head", new_tail, grp_root_local_origin)


# Terminal extremities have no spanned child; orient each one along its (post-spanning)
# parent's direction at a default length so hands/fingers/thumbs/toes flow out of the limb
# chain (#60). Ordered so hands are oriented before fingers/thumb (which then follow the
# oriented hand). Eye_* (forward gaze) and Head (up-extension) are intentionally excluded.
_TERMINAL_EXTREMITIES = [
    "Hand_Left", "Hand_Right",
    "Full_Fingers_Left", "Full_Fingers_Right",
    "Full_Thumb_Left", "Full_Thumb_Right",
    "Full_Toes_Left", "Full_Toes_Right",
]


def _orient_terminal_extremities(resolved_geometry, spec_bones, placement_metadata, grp_root_local_origin, bone_parents):
    """Point each terminal extremity along its parent's direction; head/joint unchanged."""
    for target in _TERMINAL_EXTREMITIES:
        if target not in resolved_geometry:
            continue
        parent_geom = resolved_geometry.get(bone_parents.get(target))
        if parent_geom is None:
            continue
        direction = _direction_vector(parent_geom["head"], parent_geom["tail"])
        if _distance([0.0, 0.0, 0.0], direction) <= 1e-6:
            continue
        length = _default_length_for_target(target, placement_metadata)
        new_tail = _offset_point(resolved_geometry[target]["head"], direction, length)
        _retarget_tail(resolved_geometry, spec_bones, target, new_tail, grp_root_local_origin)


def _re_resolve_synthesized_eyes(resolved_geometry, spec_bones, placement_metadata, grp_root_local_origin):
    """Re-anchor synthesized eyes using the spanning-corrected Head geometry.

    _span_core_chains rewrites Head's tail to point straight up. Eyes resolved
    before that pass may use a sideways or short source-bone tail as the crown,
    placing them at neck level. Running this after spanning gives eyes the
    corrected crown position.
    """
    head_geom = resolved_geometry.get("Head")
    if head_geom is None:
        return
    for eye in ("Eye_Left", "Eye_Right"):
        eye_bone = next((b for b in spec_bones if b["name"] == eye), None)
        if eye_bone is None:
            continue
        if eye_bone.get("geometry_source") != "eye_anchored":
            continue  # source-mapped eyes are correct as-is
        new_geom, _, _ = _resolve_eye_geometry(eye, head_geom, placement_metadata)
        resolved_geometry[eye] = new_geom
        eye_bone["head"] = _to_grp_root_local(new_geom["head"], grp_root_local_origin)
        eye_bone["tail"] = _to_grp_root_local(new_geom["tail"], grp_root_local_origin)


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
        "roll_align_axis": _asam_roll_align_axis(placement_metadata),
        "mesh_binding": copy.deepcopy(build_plan["mesh_binding"]),
        "extras_preserved": copy.deepcopy(build_plan.get("extras_preserved", [])),
        "grp_root_local_origin": list(grp_root_local_origin),
        "grp_root_world_location": [
            float(v)
            for v in root_resolution.get("grp_root_world_location", grp_root_local_origin)
        ],
        "placement_mode": root_resolution.get("placement_mode", "source"),
        "applied_grid_offset": root_resolution.get("applied_grid_offset"),
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

    # Extend the synthesized Root so its tail reaches the Hip head: ASAM wires
    # Root -> Hip, so a tall ground->pelvis Root should visibly meet the Hip.
    # Only applies to create_new_root; a reused source Root keeps its own geometry.
    if root_resolution.get("mode") == "create_new_root" and "Root" in resolved_geometry and "Hip" in resolved_geometry:
        up_index = int((root_resolution.get("up_axis") or placement_metadata["up_axis"])["index"])
        root_geometry = resolved_geometry["Root"]
        hip_head_up = resolved_geometry["Hip"]["head"][up_index]
        if hip_head_up > root_geometry["head"][up_index]:
            new_tail = list(root_geometry["tail"])
            new_tail[up_index] = hip_head_up
            root_geometry["tail"] = new_tail
            root_geometry["length"] = sum((new_tail[i] - root_geometry["head"][i]) ** 2 for i in range(3)) ** 0.5
            for bone in spec["bones"]:
                if bone["name"] == "Root":
                    bone["tail"] = _to_grp_root_local(new_tail, grp_root_local_origin)
                    break

    # Span the central chain so the spine renders upright (see _span_core_chains / #57).
    _span_core_chains(resolved_geometry, spec["bones"], placement_metadata, grp_root_local_origin)
    # Re-anchor synthesized eyes to the now-corrected Head geometry so they land in
    # the upper face rather than at neck level when the source Head had a sideways tail.
    _re_resolve_synthesized_eyes(resolved_geometry, spec["bones"], placement_metadata, grp_root_local_origin)

    _orient_terminal_extremities(
        resolved_geometry, spec["bones"], placement_metadata, grp_root_local_origin, bone_parents
    )

    spec["weight_merges"] = _compute_weight_merges(spec["bones"], source_bones)

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
    # Non-core source bones are NOT materialized as bones. The classifier flags every
    # non-core bone as an extra, which for a control rig (Rigify) is hundreds of
    # non-deforming MCH-/VIS-/IK/FK bones; materializing them all (#88) filled the clean
    # ASAM armature with stray, often long pole bones - the "spiky build". The coherent
    # rule is a single mechanism: the 28 core is the skeleton, and every non-core deform
    # bone's skin weights are consolidated into the nearest core bone by
    # _compute_weight_merges above (lossless - the mesh stays fully weighted). Control
    # bones skin nothing, so they simply never appear. Genuine separate-mesh decorators
    # (hair/clothing/accessories with their own mesh) are a future additive feature,
    # preserved via ASAM Hair_/Clothing_/Accessories_ containers rather than as body-mesh
    # skeleton subdivisions.
    mesh_binding = build_plan.get("mesh_binding", {})
    spec["merged_extra_deform_count"] = sum(
        1
        for merge in spec["weight_merges"]
        if _bone_has_skin_weight(merge["source"], mesh_binding)
    )

    spec["body_mesh_name"] = _select_body_mesh(
        mesh_binding, spec["bones"], spec["weight_merges"]
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

    preserved_source_root = next(
        (
            bone["name"]
            for bone in build_spec["bones"]
            if bone.get("geometry_source") == "preserved_source_root"
        ),
        None,
    )

    return {
        "asset_name": build_spec["asset_name"],
        "source_armature_name": build_spec["source_armature_name"],
        "generated_collection_name": execution_result["generated_collection_name"],
        "group_root_name": execution_result["group_root_name"],
        "generated_armature_name": execution_result["generated_armature_name"],
        "collection_action": execution_result.get("collection_action"),
        "grp_root_local_origin": list(build_spec.get("grp_root_local_origin", [])),
        "placement_mode": build_spec.get("placement_mode", "source"),
        "grp_root_location": list(build_spec.get("grp_root_world_location", build_spec.get("grp_root_local_origin", []))),
        "applied_grid_offset": build_spec.get("applied_grid_offset"),
        "root_mode": build_spec.get("root_resolution", {}).get("mode"),
        "preserved_source_root": preserved_source_root,
        "duplicated_meshes": copy.deepcopy(execution_result.get("duplicated_meshes", [])),
        "skipped_meshes": copy.deepcopy(execution_result.get("skipped_meshes", [])),
        "mesh_warnings": list(execution_result.get("mesh_warnings", [])),
        "built_core_targets": [bone["name"] for bone in build_spec["bones"] if bone["name"] in core_target_names],
        "targets_from_source_geometry": source_targets,
        "targets_created_heuristically": created_targets,
        "merged_extra_deform_count": build_spec.get("merged_extra_deform_count", 0),
        "preserved_pelvis_pair": copy.deepcopy(build_spec.get("preserved_pelvis_pair", [])),
        "hidden_source_objects": list(execution_result.get("hidden_source_objects", [])),
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


def success_message(builder_report: dict, report_path) -> str:
    """One-line completion banner for the builder (single or crowd)."""
    if "characters" in builder_report:
        built = len(builder_report.get("characters", []))
        failed = len(builder_report.get("failed_characters", []))
        return "Crowd built: {0} characters ({1} failed). Report: {2}".format(
            built, failed, report_path
        )
    core = len(builder_report.get("built_core_targets", []))
    merged = builder_report.get("merged_extra_deform_count", 0)
    asset = builder_report.get("asset_name", "")
    return "ASAM human built: {0} - {1} core targets, {2} extra deform bones merged into core. Report: {3}".format(
        asset, core, merged, report_path
    )


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
    print("Extra deform bones merged into core: {0}".format(builder_report.get("merged_extra_deform_count", 0)))
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


def _select_body_mesh(mesh_binding, spec_bones, weight_merges):
    """Pick the body mesh by weighted core coverage (name-free). Returns a mesh_name or None.

    Body score = number of a mesh's vertex groups that have weighted_vertex_count > 0 AND
    resolve to a core target (the group is a core target's source_bone, is itself a core
    target name, or maps to a core target via weight_merges). Tie-break by total weighted
    vertices, then by mesh_name (deterministic). Returns None when no mesh scores > 0.
    """
    core_sources = set()
    for bone in spec_bones or []:
        if bone.get("name") in CORE_TARGETS:
            core_sources.add(bone["name"])
            if bone.get("source_bone"):
                core_sources.add(bone["source_bone"])
    for merge in weight_merges or []:
        if merge.get("target") in CORE_TARGETS:
            core_sources.add(merge.get("source"))

    candidates = []
    for mesh in (mesh_binding or {}).get("meshes", []) or []:
        name = mesh.get("mesh_name")
        if not name:
            continue
        score = 0
        total = 0
        stats = mesh.get("vertex_group_stats") or {}
        for group in stats.get("per_group", []) or []:
            try:
                count = int(group.get("weighted_vertex_count") or 0)
            except (TypeError, ValueError):
                count = 0
            if count <= 0:
                continue
            total += count
            if group.get("name") in core_sources:
                score += 1
        if score > 0:
            candidates.append((score, total, name))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c[0], -c[1], c[2]))
    return candidates[0][2]


def _resolve_preserved_pelvis_pair(
    semantic_mapping: dict,
    source_bones: Dict[str, dict],
    mesh_binding: dict,
) -> List[dict]:
    """
    Preserve the lateral source pelvis bones as children of the generated Hip so
    their vertex weights are not destroyed.

    Two cases produce a pelvis pair:
    - Hip reassigned to the spine-root pivot: the displaced lateral pelvis bones are
      recorded on the Hip payload as ``preserved_pelvis_pair_sources``.
    - Legacy centered-pelvis Hip: derive the pair from the Hip source bone + its
      opposite-side counterpart.

    A pelvis side is only preserved when at least one mesh in mesh_binding records
    weighted_vertex_count > 0 for that source bone — unweighted extras would clutter
    the generated armature without deforming anything.

    Returns a list of {"source_bone_name", "generated_bone_name", "geometry"} entries
    in deterministic (sorted) order. Empty list when the case does not apply or no
    side carries skin weight.
    """
    hip_payload = semantic_mapping.get("Hip", {})

    recorded_sources = hip_payload.get("preserved_pelvis_pair_sources")
    if recorded_sources:
        names: List[str] = [name for name in recorded_sources if name in source_bones]
    elif _hip_requires_centered_pelvis_pair(hip_payload):
        primary_name = hip_payload.get("source_bone")
        if not primary_name or primary_name not in source_bones:
            return []
        names = [primary_name]
        for candidate in _opposite_name_candidates(primary_name):
            if candidate in source_bones and candidate not in names:
                names.append(candidate)
                break
    else:
        return []

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


def compute_prefix_strip_renames(vertex_group_names: Sequence[str], character_id: str) -> List[dict]:
    """Plan renames that strip a character prefix from a duplicated mesh's vertex groups.

    The source mesh keeps original group names ('Hero000Pelvis_093'); strip the prefix
    ('Pelvis') so the existing ASAM remap ('Pelvis' -> 'Hip') can match. Identity renames
    (already prefix-free) are skipped.
    """
    from phase3_classifier.segmentation import strip_character_prefix

    renames: List[dict] = []
    for name in vertex_group_names:
        target = strip_character_prefix(name, character_id)
        if target != name:
            renames.append({"source": name, "target": target})
    return renames


def build_character_specs_from_asset_dir(asset_dir: Path) -> dict:
    """Resolve build specs for an asset dir, fanning out crowd plans per character.

    Returns one of:
      single: {"crowd": False, "asset_name": str, "specs": [spec]}
      crowd:  {"crowd": True, "asset_name": str, "wrapper_collection_name": str,
               "decomposition": dict, "character_specs": [(character_id, spec), ...]}
    """
    asset_dir = Path(asset_dir).resolve()
    classifier_report, build_plan = read_builder_inputs(asset_dir)

    if not is_crowd_plan(build_plan):
        # Unchanged single-character path. build_armature_spec validates its inputs,
        # so no separate validate_builder_inputs call is needed here.
        source_bones = build_source_bone_index_from_export(
            asset_dir, build_plan["recommended_primary_armature"]
        )
        spec = build_armature_spec(classifier_report, build_plan, source_bones)
        return {"crowd": False, "asset_name": build_plan["asset_name"], "specs": [spec]}

    decomposition = classifier_report["asset_summary"]["character_decomposition"]
    source_armature = decomposition["source_armature"]
    asset_name = build_plan["asset_name"]
    full_index = build_source_bone_index_from_export(asset_dir, source_armature)

    character_specs = []
    for entry in build_plan["characters"]:
        character_id = entry["character_id"]
        flat_report, flat_plan = build_character_flat_inputs(
            classifier_report, build_plan, character_id
        )
        source_bones = slice_source_bones_for_character(full_index, character_id)
        spec = build_armature_spec(flat_report, flat_plan, source_bones)
        apply_character_naming(spec, asset_name, character_id)
        character_specs.append((character_id, spec))

    return {
        "crowd": True,
        "asset_name": asset_name,
        "wrapper_collection_name": wrapper_collection_name(asset_name),
        "decomposition": decomposition,
        "character_specs": character_specs,
    }


def build_crowd_builder_report(
    asset_name: str, decomposition: dict, character_specs, crowd_execution: dict
) -> dict:
    """Summarize a crowd build: one per-character builder report plus failures."""
    spec_by_id = {character_id: spec for character_id, spec in character_specs}
    char_reports = []
    for execution in crowd_execution["characters"]:
        character_id = execution["character_id"]
        spec = spec_by_id.get(character_id)
        if spec is None:
            continue
        report = build_builder_report(spec, execution)
        report["character_id"] = character_id
        char_reports.append(report)

    return {
        "asset_name": asset_name,
        "wrapper_collection_name": wrapper_collection_name(asset_name),
        "crowd": True,
        "character_decomposition_summary": {
            "source_armature": decomposition.get("source_armature"),
            "character_count": decomposition.get("character_count"),
            "character_ids": decomposition.get("character_ids", []),
            "shared_bones": decomposition.get("shared_bones", []),
            "unassigned_meshes": decomposition.get("unassigned_meshes", []),
        },
        "characters": char_reports,
        "failed_characters": crowd_execution.get("failed_characters", []),
    }


def write_crowd_builder_report(asset_dir: Path, crowd_report: dict) -> Tuple[dict, Path]:
    """Write a crowd builder report to builder_report.json."""
    asset_dir = Path(asset_dir).resolve()
    asset_dir.mkdir(parents=True, exist_ok=True)
    report_path = asset_dir / "builder_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(crowd_report, handle, indent=2)
        handle.write("\n")
    return crowd_report, report_path



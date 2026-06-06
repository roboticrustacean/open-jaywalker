"""
Offline Phase 3 skeleton classifier.

Consumes Phase 2 JSON exports and produces a deterministic OpenMATERIAL
classification report without modifying Blender data.
"""

from __future__ import annotations

# Late import of structural_skeleton is done at module level to avoid circular deps.
from phase3_classifier.structural_skeleton import infer_structural_skeleton, StructuralSkeleton

import copy
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Dict, FrozenSet, Iterable, List, Literal, Mapping, Optional, Sequence, Set, Tuple

# The set of naming conventions the classifier can detect and apply.
Convention = Literal["none", "biped", "mixamo"]


CORE_TARGETS: List[str] = [
    "Root",
    "Hip",
    "Lower_Spine",
    "Upper_Spine",
    "Neck",
    "Head",
    "Eye_Left",
    "Eye_Right",
    "Shoulder_Left",
    "Upper_Arm_Left",
    "Lower_Arm_Left",
    "Hand_Left",
    "Full_Thumb_Left",
    "Full_Fingers_Left",
    "Shoulder_Right",
    "Upper_Arm_Right",
    "Lower_Arm_Right",
    "Hand_Right",
    "Full_Thumb_Right",
    "Full_Fingers_Right",
    "Upper_Leg_Left",
    "Lower_Leg_Left",
    "Foot_Left",
    "Full_Toes_Left",
    "Upper_Leg_Right",
    "Lower_Leg_Right",
    "Foot_Right",
    "Full_Toes_Right",
]

# All former deferred targets are now promoted to CORE_TARGETS (full ASAM §7.3.3 compliance).
DEFERRED_TARGETS: List[str] = []

TARGET_PARENTS: Dict[str, str] = {
    "Root": "Armature_{asset_name}",
    "Hip": "Root",
    "Lower_Spine": "Hip",
    "Upper_Spine": "Lower_Spine",
    "Neck": "Upper_Spine",
    "Head": "Neck",
    "Eye_Left": "Head",
    "Eye_Right": "Head",
    "Shoulder_Left": "Upper_Spine",
    "Upper_Arm_Left": "Shoulder_Left",
    "Lower_Arm_Left": "Upper_Arm_Left",
    "Hand_Left": "Lower_Arm_Left",
    "Full_Thumb_Left": "Hand_Left",
    "Full_Fingers_Left": "Hand_Left",
    "Shoulder_Right": "Upper_Spine",
    "Upper_Arm_Right": "Shoulder_Right",
    "Lower_Arm_Right": "Upper_Arm_Right",
    "Hand_Right": "Lower_Arm_Right",
    "Full_Thumb_Right": "Hand_Right",
    "Full_Fingers_Right": "Hand_Right",
    "Upper_Leg_Left": "Hip",
    "Lower_Leg_Left": "Upper_Leg_Left",
    "Foot_Left": "Lower_Leg_Left",
    "Full_Toes_Left": "Foot_Left",
    "Upper_Leg_Right": "Hip",
    "Lower_Leg_Right": "Upper_Leg_Right",
    "Foot_Right": "Lower_Leg_Right",
    "Full_Toes_Right": "Foot_Right",
}

TARGET_PRIORITY: Dict[str, int] = {target: index for index, target in enumerate(CORE_TARGETS)}

ROLE_PREFIXES = {"def": "deform", "org": "auxiliary", "mch": "mechanism"}
CONTROL_TOKENS = {
    "ctrl",
    "control",
    "ik",
    "fk",
    "pole",
    "tweak",
    "twist",
    "heel",
    "widget",
    "wgt",
    "vis",
    "rot",
    "str",
    "mechanism",
}
EXTRA_TOKENS = {
    "hair",
    "cloth",
    "cape",
    "skirt",
    "weapon",
    "prop",
    "ear",
    "jaw",
    "face",
    "brow",
    "lip",
    "nose",
    "breast",
    "tail",
    "accessory",
}
SIDE_WORDS = {"l": "left", "left": "left", "r": "right", "right": "right"}

HEIGHT_BANDS: Dict[str, Tuple[float, float]] = {
    "Root": (0.0, 0.40),
    "Hip": (0.35, 0.70),
    "Lower_Spine": (0.40, 0.78),
    "Upper_Spine": (0.55, 0.93),
    "Neck": (0.72, 0.98),
    "Head": (0.82, 1.0),
    "Eye": (0.85, 1.0),
    "Shoulder": (0.58, 0.92),
    "Upper_Arm": (0.45, 0.86),
    "Lower_Arm": (0.28, 0.82),
    "Hand": (0.18, 0.72),
    "Full_Thumb": (0.18, 0.72),
    "Full_Fingers": (0.18, 0.72),
    "Upper_Leg": (0.28, 0.72),
    "Lower_Leg": (0.08, 0.55),
    "Foot": (0.0, 0.25),
    "Full_Toes": (0.0, 0.15),
}

AXIS_NAMES = ("x", "y", "z")
RECOVERABLE_ACTIONS = {"direct_map", "alias_map", "repair_in_builder"}
SPLIT_ACTION = "split_in_builder"
# Actions whose source bone is consumed by a generated target (so it must not be
# preserved as an extra) and which count toward an armature's mapped-target total.
MAPPED_ACTIONS = RECOVERABLE_ACTIONS | {SPLIT_ACTION}
ROOT_CENTER_TOLERANCE_RATIO = 0.02
ROOT_GROUND_TOLERANCE_RATIO = 0.01
ROOT_TAIL_TO_HIP_TOLERANCE_RATIO = 0.05
ROOT_UP_ALIGNMENT_COSINE = 0.95
ROOT_ORIGIN_SPEC_REFERENCES: Tuple[str, ...] = (
    "ASAM OpenMATERIAL 3D 7.3.3.1 General",
    "ASAM OpenMATERIAL 3D 7.3.3.3.2 Grp_Root",
    "ASAM OpenMATERIAL 3D 7.3.3.3.4 Root",
)

CORE_FAMILY_BY_TARGET: Dict[str, str] = {
    "Root": "root",
    "Hip": "hip",
    "Lower_Spine": "lower_spine",
    "Upper_Spine": "upper_spine",
    "Neck": "neck",
    "Head": "head",
    "Eye_Left": "eye",
    "Eye_Right": "eye",
    "Shoulder_Left": "shoulder",
    "Upper_Arm_Left": "upper_arm",
    "Lower_Arm_Left": "lower_arm",
    "Hand_Left": "hand",
    "Full_Thumb_Left": "full_thumb",
    "Full_Fingers_Left": "full_fingers",
    "Shoulder_Right": "shoulder",
    "Upper_Arm_Right": "upper_arm",
    "Lower_Arm_Right": "lower_arm",
    "Hand_Right": "hand",
    "Full_Thumb_Right": "full_thumb",
    "Full_Fingers_Right": "full_fingers",
    "Upper_Leg_Left": "upper_leg",
    "Lower_Leg_Left": "lower_leg",
    "Foot_Left": "foot",
    "Full_Toes_Left": "full_toes",
    "Upper_Leg_Right": "upper_leg",
    "Lower_Leg_Right": "lower_leg",
    "Foot_Right": "foot",
    "Full_Toes_Right": "full_toes",
}

# Translation from a scoring-family name (CORE_FAMILY_BY_TARGET values) to the
# family-tag vocabulary used by _detect_family_tags / _score_target_candidate.
SCORING_FAMILY_TO_TAGS: Dict[str, FrozenSet[str]] = {
    "root": frozenset({"root"}),
    "hip": frozenset({"hip"}),
    "lower_spine": frozenset({"spine", "lower_spine"}),
    "upper_spine": frozenset({"spine", "upper_spine"}),
    "neck": frozenset({"neck"}),
    "head": frozenset({"head"}),
    "eye": frozenset({"eye"}),
    "shoulder": frozenset({"shoulder"}),
    "upper_arm": frozenset({"upper_arm"}),
    "lower_arm": frozenset({"lower_arm"}),
    "hand": frozenset({"hand"}),
    "full_thumb": frozenset({"thumb"}),
    "full_fingers": frozenset({"finger"}),
    "upper_leg": frozenset({"upper_leg"}),
    "lower_leg": frozenset({"lower_leg"}),
    "foot": frozenset({"foot"}),
    "full_toes": frozenset({"toe"}),
}


@dataclass(frozen=True)
class BoneConvention:
    name: str
    # Keys are digit-stripped *compact* names; values are scoring-family names.
    # Wrapped in MappingProxyType so the shared module-level registry is read-only.
    alias_families: Mapping[str, FrozenSet[str]]


# Note: the `spine` stem maps to BOTH lower_spine and upper_spine because a
# bone named only "Spine<n>" cannot be split by name alone. Name scoring is
# therefore deliberately ambiguous for spine bones; the spine chain ordering
# (hierarchy/geometry evidence) disambiguates lower vs upper.
BONE_CONVENTIONS: Dict[str, BoneConvention] = {
    "biped": BoneConvention(
        name="biped",
        alias_families=MappingProxyType({
            "com": frozenset({"root"}),
            "pelvis": frozenset({"hip"}),
            "spine": frozenset({"lower_spine", "upper_spine"}),
            "clavicle": frozenset({"shoulder"}),
            "upperarm": frozenset({"upper_arm"}),
            "forearm": frozenset({"lower_arm"}),
            "hand": frozenset({"hand"}),
            "thigh": frozenset({"upper_leg"}),
            "calf": frozenset({"lower_leg"}),
            "foot": frozenset({"foot"}),
            "toe": frozenset({"full_toes"}),
        }),
    ),
    "mixamo": BoneConvention(
        name="mixamo",
        alias_families=MappingProxyType({
            "hips": frozenset({"hip"}),
            "spine": frozenset({"lower_spine", "upper_spine"}),
            "neck": frozenset({"neck"}),
            "head": frozenset({"head"}),
            "shoulder": frozenset({"shoulder"}),
            "arm": frozenset({"upper_arm"}),
            "forearm": frozenset({"lower_arm"}),
            "hand": frozenset({"hand"}),
            "upleg": frozenset({"upper_leg"}),
            "leg": frozenset({"lower_leg"}),
            "foot": frozenset({"foot"}),
            "toebase": frozenset({"full_toes"}),
        }),
    ),
}


def _compact_stem(compact_name: str) -> str:
    """Compact name with any trailing digits removed (spine0 -> spine, com1 -> com)."""
    return re.sub(r"\d+$", "", compact_name)


@dataclass
class ArmatureInput:
    armature_name: str
    all_path: Optional[Path]
    filtered_path: Optional[Path]
    primary_path: Path
    support_path: Optional[Path]
    primary_data: dict
    support_data: Optional[dict]


@dataclass
class BoneInfo:
    name: str
    parent: Optional[str]
    head: Tuple[float, float, float]
    tail: Tuple[float, float, float]
    length: float
    origin: str
    normalized_name: str
    compact_name: str
    tokens: List[str]
    side: Optional[str]
    midpoint: Tuple[float, float, float]
    min_point: Tuple[float, float, float]
    max_point: Tuple[float, float, float]
    role: str
    role_modifier: float
    family_tags: Set[str]
    child_names: List[str] = field(default_factory=list)


@dataclass
class CandidateScore:
    source_bone: str
    source_origin: str
    role: str
    selection_score: float
    confidence: float
    name_evidence: float
    hierarchy_evidence: float
    geometry_evidence: float
    structural_evidence: float
    notes: List[str]


def discover_armatures(asset_dir: Path) -> List[ArmatureInput]:
    groups: Dict[str, Dict[str, Path]] = {}
    loaded: Dict[Path, dict] = {}

    for json_path in sorted(asset_dir.glob("*.json")):
        name = json_path.name
        if name.endswith("_all.json"):
            groups.setdefault(name[: -len("_all.json")], {})["all"] = json_path
        elif name.endswith("_filtered.json"):
            groups.setdefault(name[: -len("_filtered.json")], {})["filtered"] = json_path

    armatures: List[ArmatureInput] = []
    for armature_name in sorted(groups):
        all_path = groups[armature_name].get("all")
        filtered_path = groups[armature_name].get("filtered")
        all_data = _load_json_cached(all_path, loaded) if all_path else None
        filtered_data = _load_json_cached(filtered_path, loaded) if filtered_path else None

        primary_path: Optional[Path] = None
        primary_data: Optional[dict] = None
        support_path: Optional[Path] = None
        support_data: Optional[dict] = None

        if filtered_data and filtered_data.get("filter") == "DEF-" and filtered_data.get("bone_count", 0) > 0:
            primary_path = filtered_path
            primary_data = filtered_data
            support_path = all_path
            support_data = all_data
        elif all_data:
            primary_path = all_path
            primary_data = all_data
        elif filtered_data:
            primary_path = filtered_path
            primary_data = filtered_data

        if primary_path and primary_data:
            armatures.append(
                ArmatureInput(
                    armature_name=armature_name,
                    all_path=all_path,
                    filtered_path=filtered_path,
                    primary_path=primary_path,
                    support_path=support_path,
                    primary_data=primary_data,
                    support_data=support_data,
                )
            )

    return armatures


# Ordered selection cascade. Each entry is (audit_label, summary_field); the sort
# key negates each field (higher is better) and _compute_selection_tiebreaker walks
# the same list so the two can never drift. Binding leads, then the skin-weight
# signals, then today's bundled scalar as the no-binding fallback.
_SELECTION_CASCADE: Tuple[Tuple[str, str], ...] = (
    ("mesh_bound_term", "mesh_bound_term"),
    ("vertex_group_coverage", "vertex_group_coverage"),
    ("deform_purity", "deform_purity"),
    ("score", "ranking_score"),
    ("mapped_core_targets", "mapped_core_targets"),
    ("average_confidence", "average_confidence"),
)


def _selection_key(report: dict) -> tuple:
    summary = report["summary"]
    return tuple(-summary[field] for _, field in _SELECTION_CASCADE) + (report["armature_name"],)


def classify_asset_folder(asset_dir: Path) -> Tuple[dict, dict]:
    asset_dir = asset_dir.resolve()
    if not asset_dir.is_dir():
        raise FileNotFoundError("Asset directory does not exist: {0}".format(asset_dir))

    armature_inputs = discover_armatures(asset_dir)
    if not armature_inputs:
        raise ValueError("No armature inspector JSON files were found in {0}".format(asset_dir))

    armature_reports = [classify_armature(asset_dir.name, armature_input) for armature_input in armature_inputs]

    ranked_reports = sorted(armature_reports, key=_selection_key)
    recommended_report = ranked_reports[0]
    selection_tiebreaker = _compute_selection_tiebreaker(ranked_reports)
    mesh_binding = copy.deepcopy(recommended_report.get("mesh_binding"))

    actions = _build_actions(recommended_report["asam_targets"])

    recommended_input = next(
        item for item in armature_inputs if item.armature_name == recommended_report["armature_name"]
    )
    from phase3_classifier.segmentation import segment_recommended  # local import avoids cycle
    segmentation = segment_recommended(asset_dir.name, recommended_input)

    asset_summary = {
        "asset_name": asset_dir.name,
        "asset_dir": str(asset_dir),
        "armature_count": len(armature_reports),
        "discovered_armatures": [report["armature_name"] for report in armature_reports],
        "ranking": [
            _build_ranking_entry(report, selection_tiebreaker if index == 0 else None)
            for index, report in enumerate(ranked_reports)
        ],
    }

    classifier_report = {
        "asset_summary": asset_summary,
        "armatures": armature_reports,
        "recommended_primary_armature": recommended_report["armature_name"],
        "semantic_mapping": copy.deepcopy(recommended_report["asam_targets"]),
        "root_resolutions": [copy.deepcopy(recommended_report["root_resolution"])],
        "placement_metadata": copy.deepcopy(recommended_report["placement_metadata"]),
        "mesh_binding": mesh_binding,
        "missing_targets": list(recommended_report["missing_core_targets"]),
        "ambiguous_targets": copy.deepcopy(recommended_report["ambiguous_targets"]),
        "unclassified_bones": copy.deepcopy(recommended_report["unclassified_bones"]),
        "review_flags": list(recommended_report["review_flags"]),
        "deferred_targets": list(recommended_report["deferred_targets"]),
    }

    build_plan = {
        "asset_name": asset_dir.name,
        "recommended_primary_armature": recommended_report["armature_name"],
        "actions": actions,
        "root_resolutions": [copy.deepcopy(recommended_report["root_resolution"])],
        "placement_metadata": copy.deepcopy(recommended_report["placement_metadata"]),
        "mesh_binding": copy.deepcopy(mesh_binding),
        "proposed_asam_hierarchy": copy.deepcopy(recommended_report["proposed_asam_hierarchy"]),
        "extras_preserved": copy.deepcopy(recommended_report["extras_preserved"]),
    }

    if segmentation is not None:
        asset_summary["character_decomposition"] = segmentation.decomposition
        classifier_report.pop("semantic_mapping")
        classifier_report["characters"] = segmentation.report_characters
        build_plan["characters"] = segmentation.plan_characters

    return classifier_report, build_plan


def _build_actions(asam_targets: Dict[str, dict]) -> Dict[str, list]:
    actions: Dict[str, list] = {"rename": [], "create": [], "split": []}
    split_pending: Dict[str, dict] = {}
    for target, payload in asam_targets.items():
        if target == "Root":
            continue
        if payload["action"] in {"direct_map", "alias_map"} and payload["source_bone"]:
            actions["rename"].append({"source": payload["source_bone"], "target": target})
        elif payload["action"] == SPLIT_ACTION:
            entry = split_pending.setdefault(
                payload["source_bone"],
                {"source": payload["source_bone"], "lower_target": None,
                 "upper_target": None, "ratio": 0.5},
            )
            if payload.get("split_half") == "lower":
                entry["lower_target"] = target
            else:
                entry["upper_target"] = target
        elif payload["action"] == "create_in_builder":
            actions["create"].append({"target": target, "parent": TARGET_PARENTS.get(target)})
    actions["split"] = [
        entry for entry in split_pending.values()
        if entry["lower_target"] and entry["upper_target"]
    ]
    return actions


def _build_ranking_entry(report: dict, selection_tiebreaker: Optional[str]) -> dict:
    summary = report["summary"]
    entry = {
        "armature_name": report["armature_name"],
        "ranking_score": summary["ranking_score"],
        "mapped_core_targets": summary["mapped_core_targets"],
        "average_confidence": summary["average_confidence"],
        "mesh_bound_term": summary["mesh_bound_term"],
        "deform_evidence_term": summary["deform_evidence_term"],
        "extras_term": summary["extras_term"],
        "vertex_group_coverage": summary["vertex_group_coverage"],
        "deform_purity": summary["deform_purity"],
        "primary_input": report["selected_inputs"]["primary"],
    }
    if selection_tiebreaker is not None:
        entry["selection_tiebreaker"] = selection_tiebreaker
    return entry


def _compute_selection_tiebreaker(ranked_reports: List[dict]) -> str:
    """Identify which step of the selection cascade decided the top pick.

    Returns 'sole_candidate' when there is only one armature, otherwise the audit
    label of the first cascade step on which the top candidate strictly beats the
    second-place candidate, or 'armature_name' when every numeric signal ties.
    """
    if len(ranked_reports) <= 1:
        return "sole_candidate"

    top_summary = ranked_reports[0]["summary"]
    runner_summary = ranked_reports[1]["summary"]

    for label, field in _SELECTION_CASCADE:
        if top_summary[field] != runner_summary[field]:
            return label
    return "armature_name"


def write_asset_report(asset_dir: Path) -> Tuple[dict, dict, Path, Path]:
    classifier_report, build_plan = classify_asset_folder(asset_dir)
    report_path = asset_dir / "classifier_report.json"
    plan_path = asset_dir / "build_plan.json"
    
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(classifier_report, handle, indent=2)
        handle.write("\n")
        
    with plan_path.open("w", encoding="utf-8") as handle:
        json.dump(build_plan, handle, indent=2)
        handle.write("\n")
        
    return classifier_report, build_plan, report_path, plan_path


def print_console_summary(report: dict, plan: dict, report_path: Path, plan_path: Path) -> None:
    asset_summary = report["asset_summary"]
    print("Phase 3 classifier summary")
    print("Asset folder: {0}".format(asset_summary["asset_dir"]))
    print("Discovered armatures: {0}".format(", ".join(asset_summary["discovered_armatures"])))
    print("")

    for armature_report in sorted(report["armatures"], key=lambda item: item["armature_name"]):
        summary = armature_report["summary"]
        selected_inputs = armature_report["selected_inputs"]
        print("- {0}".format(armature_report["armature_name"]))
        print("  primary={0}  support={1}".format(selected_inputs["primary"], selected_inputs["support"] or "(none)"))
        print(
            "  mapped={0}/{1}  avg_conf={2:.3f}  direct={3}  alias={4}  repair={5}".format(
                summary["mapped_core_targets"],
                len(CORE_TARGETS),
                summary["average_confidence"],
                summary["direct_map_targets"],
                summary["alias_map_targets"],
                summary.get("repair_targets", 0),
            )
        )
        print(
            "  missing={0}  ambiguous={1}  preserved_extras={2}".format(
                len(armature_report["missing_core_targets"]),
                len(armature_report["ambiguous_targets"]),
                len(armature_report["extras_preserved"]),
            )
        )
        if armature_report["review_flags"]:
            print("  flags={0}".format(", ".join(armature_report["review_flags"])))

    print("")
    print("Recommended primary armature: {0}".format(report["recommended_primary_armature"]))
    print("Root resolution: {0}".format(plan["root_resolutions"][0]["mode"]))
    print("Missing targets: {0}".format(", ".join(report["missing_targets"]) or "(none)"))
    crowd_characters = report.get("characters") or []
    if crowd_characters:
        missing_counts = {}
        for character in crowd_characters:
            for target in character.get("missing_targets", []):
                missing_counts[target] = missing_counts.get(target, 0) + 1
        print("Crowd missing-target counts (of {0} characters):".format(len(crowd_characters)))
        if missing_counts:
            for target, count in sorted(missing_counts.items(), key=lambda item: (-item[1], item[0])):
                print("  {0}: {1}".format(target, count))
        else:
            print("  (none)")
    print("Ambiguous targets: {0}".format(", ".join(item["target"] for item in report["ambiguous_targets"]) or "(none)"))
    print("Preserved extras count: {0}".format(len(plan["extras_preserved"])))
    print("Report written to: {0}".format(report_path))
    print("Build plan written to: {0}".format(plan_path))


def classify_armature(asset_name: str, armature_input: ArmatureInput) -> dict:
    convention = _detect_convention(armature_input.primary_data, armature_input.support_data)
    bones = _build_bone_index(armature_input.primary_data, armature_input.support_data, convention)
    context = _build_context(bones, armature_input.primary_data, armature_input.support_data, convention)
    name_quality = compute_name_quality(context)

    target_candidates: Dict[str, List[CandidateScore]] = {}
    for target_name in CORE_TARGETS:
        candidates: List[CandidateScore] = []
        for bone in bones.values():
            candidate = _score_target_candidate(target_name, bone, context, name_quality)
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(key=lambda item: (-item.selection_score, -item.confidence, item.source_bone))
        target_candidates[target_name] = candidates

    resolved_targets = _resolve_targets(target_candidates, context)
    _reconcile_hip_to_spine_pivot(resolved_targets, context)
    _plan_single_spine_split(resolved_targets, context)
    root_resolution = _resolve_root_compliance(resolved_targets, target_candidates.get("Root", []), context)
    mapped_bones = {
        payload["source_bone"]
        for payload in resolved_targets.values()
        if payload["source_bone"] is not None and payload["action"] in MAPPED_ACTIONS
    }

    extras_preserved, unclassified_bones = _classify_non_mapped_bones(bones, context, mapped_bones)
    review_flags = _build_review_flags(resolved_targets, root_resolution, context)
    ambiguous_targets = [
        {"target": target_name, "reason": payload["notes"], "confidence": payload["confidence"]}
        for target_name, payload in resolved_targets.items()
        if payload["action"] == "review"
    ]
    missing_targets = [
        target_name
        for target_name, payload in resolved_targets.items()
        if payload["action"] == "create_in_builder"
    ]

    return {
        "armature_name": armature_input.armature_name,
        "detected_convention": convention,
        "name_quality": round(name_quality, 3),
        "selected_inputs": {
            "primary": armature_input.primary_path.name,
            "support": armature_input.support_path.name if armature_input.support_path else None,
            "primary_filter": armature_input.primary_data.get("filter"),
            "source_file": armature_input.primary_data.get("source_file"),
        },
        "summary": _build_armature_summary(
            resolved_targets,
            review_flags,
            armature_input.primary_data.get("mesh_binding"),
            extras_preserved,
            bones,
            armature_input.armature_name,
            name_quality,
        ),
        "asam_targets": resolved_targets,
        "root_resolution": root_resolution,
        "placement_metadata": copy.deepcopy(context["placement_metadata"]),
        "mesh_binding": copy.deepcopy(armature_input.primary_data.get("mesh_binding")),
        "proposed_asam_hierarchy": _build_proposed_hierarchy(asset_name),
        "extras_preserved": extras_preserved,
        "unclassified_bones": unclassified_bones,
        "missing_core_targets": missing_targets,
        "ambiguous_targets": ambiguous_targets,
        "review_flags": review_flags,
        "deferred_targets": list(DEFERRED_TARGETS),
    }


def _build_context(bones: Dict[str, BoneInfo], primary_data: dict, support_data: Optional[dict], convention: Convention = "none") -> dict:
    children_map = {name: list(bone.child_names) for name, bone in bones.items()}
    derived_height_axis, derived_lateral_axis = _infer_axes(bones.values())
    derived_centerline = _median([bone.midpoint[derived_lateral_axis] for bone in bones.values()])
    derived_side_signs = _calibrate_side_signs(bones.values(), derived_lateral_axis)
    placement_metadata = _load_placement_metadata(
        primary_data,
        support_data,
        bones,
        derived_height_axis,
        derived_lateral_axis,
        derived_centerline,
        derived_side_signs,
    )

    height_axis = int(placement_metadata["up_axis"]["index"])
    lateral_axis = int(placement_metadata["side_axis"]["index"])
    forward_axis = int(placement_metadata["forward_axis"]["index"])
    height_sign = int(placement_metadata["up_axis"]["sign"])
    centerline = float(placement_metadata["bbox_ground_center"][lateral_axis])
    height_bounds_min = float(placement_metadata["bbox_min"][height_axis])
    height_bounds_max = float(placement_metadata["bbox_max"][height_axis])
    lateral_span = abs(float(placement_metadata["bbox_max"][lateral_axis]) - float(placement_metadata["bbox_min"][lateral_axis]))
    if lateral_span <= 1e-6:
        lateral_span = 1.0
    side_signs = {
        "left": int(placement_metadata["side_axis"]["sign"]),
        "right": -int(placement_metadata["side_axis"]["sign"]),
    }

    return {
        "bones": bones,
        "children_map": children_map,
        "placement_metadata": placement_metadata,
        "height_axis": height_axis,
        "forward_axis": forward_axis,
        "lateral_axis": lateral_axis,
        "height_sign": height_sign,
        "centerline": centerline,
        "height_bounds_min": height_bounds_min,
        "height_bounds_max": height_bounds_max,
        "height_span": max(abs(height_bounds_max - height_bounds_min), 1e-6),
        "lateral_span": lateral_span,
        "side_signs": side_signs,
        "spine_chain": _choose_spine_chain(primary_data, support_data),
        "arm_chains": _choose_limb_chains(bones, primary_data, support_data, "arm", lateral_axis, centerline, side_signs),
        "leg_chains": _choose_limb_chains(bones, primary_data, support_data, "leg", lateral_axis, centerline, side_signs),
        "convention": convention,
        "structural_skeleton": infer_structural_skeleton(
            bones,
            height_axis=height_axis,
            lateral_axis=lateral_axis,
            forward_axis=forward_axis,
            height_sign=height_sign,
            centerline=centerline,
            side_signs=side_signs,
            lateral_span=lateral_span,
            height_span=max(abs(height_bounds_max - height_bounds_min), 1e-6),
        ),
    }


def _load_placement_metadata(
    primary_data: dict,
    support_data: Optional[dict],
    bones: Dict[str, BoneInfo],
    derived_height_axis: int,
    derived_lateral_axis: int,
    derived_centerline: float,
    derived_side_signs: Dict[str, int],
) -> dict:
    for payload in (primary_data, support_data or {}):
        placement = payload.get("placement_metadata")
        if _placement_metadata_is_valid(placement):
            return copy.deepcopy(placement)

    return _derive_placement_metadata_from_bones(
        bones,
        derived_height_axis,
        derived_lateral_axis,
        derived_centerline,
        derived_side_signs,
    )


def _placement_metadata_is_valid(placement: Optional[dict]) -> bool:
    if not placement:
        return False
    required = {"bbox_min", "bbox_max", "bbox_ground_center", "forward_axis", "side_axis", "up_axis"}
    return required.issubset(set(placement.keys()))


def _derive_placement_metadata_from_bones(
    bones: Dict[str, BoneInfo],
    derived_height_axis: int,
    derived_lateral_axis: int,
    derived_centerline: float,
    derived_side_signs: Dict[str, int],
) -> dict:
    points: List[Tuple[float, float, float]] = []
    upper_candidates: List[float] = []
    lower_candidates: List[float] = []

    for bone in bones.values():
        points.extend([bone.head, bone.tail])
        if {"head", "neck", "upper_spine", "lower_spine", "spine"} & bone.family_tags:
            upper_candidates.append(bone.midpoint[derived_height_axis])
        if {"foot", "toe", "lower_leg", "upper_leg"} & bone.family_tags:
            lower_candidates.append(bone.midpoint[derived_height_axis])

    if points:
        bbox_min = [min(point[index] for point in points) for index in range(3)]
        bbox_max = [max(point[index] for point in points) for index in range(3)]
    else:
        bbox_min = [0.0, 0.0, 0.0]
        bbox_max = [0.0, 0.0, 0.0]

    height_sign = 1
    if upper_candidates and lower_candidates and _mean(upper_candidates) < _mean(lower_candidates):
        height_sign = -1

    side_sign = int(derived_side_signs.get("left", 1))
    forward_axis = next(axis for axis in range(3) if axis not in {derived_height_axis, derived_lateral_axis})
    forward_sign = height_sign * _axis_cross_sign(forward_axis, derived_lateral_axis, derived_height_axis) * side_sign

    forward_median = _median([bone.midpoint[forward_axis] for bone in bones.values()])
    bbox_ground_center = [(bbox_min[index] + bbox_max[index]) / 2.0 for index in range(3)]
    bbox_ground_center[forward_axis] = forward_median
    bbox_ground_center[derived_lateral_axis] = derived_centerline
    bbox_ground_center[derived_height_axis] = bbox_min[derived_height_axis] if height_sign >= 0 else bbox_max[derived_height_axis]

    return {
        "bounds_source": "bones_fallback",
        "driven_meshes": [],
        "bbox_min": [float(value) for value in bbox_min],
        "bbox_max": [float(value) for value in bbox_max],
        "bbox_height": float(abs(bbox_max[derived_height_axis] - bbox_min[derived_height_axis])),
        "bbox_ground_center": [float(value) for value in bbox_ground_center],
        "forward_axis": {"index": forward_axis, "name": AXIS_NAMES[forward_axis], "sign": int(forward_sign)},
        "side_axis": {"index": derived_lateral_axis, "name": AXIS_NAMES[derived_lateral_axis], "sign": int(side_sign)},
        "up_axis": {"index": derived_height_axis, "name": AXIS_NAMES[derived_height_axis], "sign": int(height_sign)},
    }


def _build_bone_index(primary_data: dict, support_data: Optional[dict], convention: Convention = "none") -> Dict[str, BoneInfo]:
    raw_bones: Dict[str, dict] = {}
    origins: Dict[str, str] = {}

    for bone_payload in (support_data or {}).get("bones", []):
        raw_bones[bone_payload["name"]] = bone_payload
        origins[bone_payload["name"]] = "support"

    for bone_payload in primary_data.get("bones", []):
        raw_bones[bone_payload["name"]] = bone_payload
        origins[bone_payload["name"]] = "primary"

    children_map: Dict[str, List[str]] = {name: [] for name in raw_bones}
    for payload in raw_bones.values():
        parent = payload.get("parent")
        if parent in children_map:
            children_map[parent].append(payload["name"])

    bone_index: Dict[str, BoneInfo] = {}
    for name in sorted(raw_bones):
        payload = raw_bones[name]
        normalized_name, compact_name, tokens, side = _normalize_bone_name(name, convention)
        family_tags = _detect_family_tags(normalized_name, compact_name, tokens, convention)
        role, role_modifier = _classify_bone_role(name, tokens)
        head = _to_float_triplet(payload.get("head"))
        tail = _to_float_triplet(payload.get("tail"))

        parent = payload.get("parent")
        if parent and parent not in raw_bones:
            parent = None

        midpoint = tuple((head[index] + tail[index]) / 2.0 for index in range(3))
        min_point = tuple(min(head[index], tail[index]) for index in range(3))
        max_point = tuple(max(head[index], tail[index]) for index in range(3))

        bone_index[name] = BoneInfo(
            name=name,
            parent=parent,
            head=head,
            tail=tail,
            length=float(payload.get("length", 0.0)),
            origin=origins[name],
            normalized_name=normalized_name,
            compact_name=compact_name,
            tokens=tokens,
            side=side,
            midpoint=midpoint,
            min_point=min_point,
            max_point=max_point,
            role=role,
            role_modifier=role_modifier,
            family_tags=family_tags,
            child_names=sorted(children_map.get(name, [])),
        )

    return bone_index


def _score_target_candidate(target_name: str, bone: BoneInfo, context: dict, name_quality: float) -> Optional[CandidateScore]:
    family = CORE_FAMILY_BY_TARGET[target_name]
    side = _target_side(target_name)
    name_score = _name_evidence(target_name, bone, context.get("convention", "none"))
    hierarchy_score = _hierarchy_evidence(target_name, bone, context)
    geometry_score = _geometry_evidence(target_name, bone, context)
    structural_score = _structural_evidence(target_name, bone, context)

    w_name, w_struct, w_geom = _confidence_weights(name_quality, bone.origin)
    confidence = round(_clamp((w_name * name_score) + (w_struct * structural_score) + (w_geom * geometry_score)), 3)

    if confidence < 0.10 and name_score < 0.20:
        return None

    source_modifier = 1.0 if bone.origin == "primary" else 0.90
    family_modifier = 1.0 if family in bone.family_tags else 0.95
    # Structural confirmation: a bone the name-free skeleton labeler actually
    # places at this target's anatomical position (matching family) is stronger
    # evidence than a name-only match. This lets a genuine deform segment with
    # topological confirmation win over a same-position control bone whose only
    # claim is a high-scoring name (e.g. a Rigify 'chest' control out-naming the
    # real DEF upper-spine segment). Scaled by the structural confidence, so a
    # weak/absent structural signal barely moves the score.
    structural_modifier = 1.0 + (0.06 * structural_score)
    selection_score = round(
        confidence * bone.role_modifier * source_modifier * family_modifier * structural_modifier,
        4,
    )

    notes: List[str] = []
    if bone.origin != "primary":
        notes.append("support_fallback")
    if bone.role in {"control", "mechanism"}:
        notes.append("control_or_mechanism_bone")
    if side and bone.side and bone.side != side:
        notes.append("opposite_side_name")
    if side and _side_geometry_conflict(side, bone, context):
        notes.append("geometry_side_conflict")

    return CandidateScore(
        source_bone=bone.name,
        source_origin=bone.origin,
        role=bone.role,
        selection_score=selection_score,
        confidence=confidence,
        name_evidence=round(name_score, 3),
        hierarchy_evidence=round(hierarchy_score, 3),
        geometry_evidence=round(geometry_score, 3),
        structural_evidence=round(structural_score, 3),
        notes=sorted(set(notes)),
    )


def _resolve_targets(target_candidates: Dict[str, List[CandidateScore]], context: dict) -> Dict[str, dict]:
    resolved: Dict[str, dict] = {}
    provisional_choices: Dict[str, Optional[CandidateScore]] = {}

    for target_name in CORE_TARGETS:
        candidates = target_candidates.get(target_name, [])
        provisional_choices[target_name] = candidates[0] if candidates else None

    source_to_targets: Dict[str, List[str]] = {}
    for target_name, candidate in provisional_choices.items():
        if candidate is not None:
            source_to_targets.setdefault(candidate.source_bone, []).append(target_name)

    contested_targets: Set[str] = set()
    for source_bone, targets in source_to_targets.items():
        if len(targets) <= 1:
            continue
        ordered_targets = sorted(
            targets,
            key=lambda target: (-provisional_choices[target].confidence, TARGET_PRIORITY[target]),  # type: ignore[union-attr]
        )
        contested_targets.update(ordered_targets[1:])

    for target_name in CORE_TARGETS:
        candidate = provisional_choices[target_name]
        if candidate is None:
            payload = _missing_target_payload("no_plausible_candidate")
            payload["decision"] = "conflict_review"
            resolved[target_name] = payload
            continue

        if target_name in contested_targets:
            notes = list(candidate.notes)
            notes.append("candidate_conflict")
            # When the only candidate this target could claim is a weak,
            # already-owned bone (no real name or confidence of its own), the
            # target genuinely has no source bone — e.g. a missing intermediate
            # joint like a thigh+foot leg with no shin. Hand it to the builder to
            # create rather than flagging an unactionable review.
            if candidate.confidence < 0.25 and candidate.name_evidence < 0.20:
                notes.append("contested_low_confidence_no_distinct_bone")
                action = "create_in_builder"
            else:
                action = "review"
            resolved[target_name] = {
                "source_bone": None,
                "confidence": round(candidate.confidence, 3),
                "action": action,
                "decision": "conflict_review",
                "evidence": {
                    "name": candidate.name_evidence,
                    "hierarchy": candidate.hierarchy_evidence,
                    "geometry": candidate.geometry_evidence,
                    "structural": candidate.structural_evidence,
                    "source_origin": candidate.source_origin,
                    "role": candidate.role,
                },
                "notes": sorted(set(notes)),
            }
            continue

        action = _decision_tag(target_name, candidate, target_candidates, context)
        notes = list(candidate.notes)
        if action in {"direct_map", "alias_map"} and _paired_sided_pelvis_requires_centering(
            target_name,
            candidate,
            target_candidates,
            context,
        ):
            action = "repair_in_builder"
            notes = sorted(set(notes + ["paired_sided_pelvis_requires_centering"]))
        resolved[target_name] = {
            "source_bone": candidate.source_bone if action != "create_in_builder" else None,
            "confidence": round(candidate.confidence, 3),
            "action": action,
            "decision": _reconciliation_tag(candidate.name_evidence, candidate.structural_evidence),
            "evidence": {
                "name": candidate.name_evidence,
                "hierarchy": candidate.hierarchy_evidence,
                "geometry": candidate.geometry_evidence,
                "structural": candidate.structural_evidence,
                "source_origin": candidate.source_origin,
                "role": candidate.role,
            },
            "notes": notes,
        }

    return resolved


def _mirror_sided_bone_name(name: str) -> Optional[str]:
    """Return the opposite-side counterpart of a sided bone name, or None."""
    for left, right in ((".L", ".R"), ("_L", "_R"), ("-L", "-R"), ("Left", "Right"), ("left", "right")):
        if name.endswith(left) or left in name:
            return name.replace(left, right)
        if name.endswith(right) or right in name:
            return name.replace(right, left)
    return None


def _reconcile_hip_to_spine_pivot(resolved_targets: Dict[str, dict], context: dict) -> None:
    """Reassign ASAM Hip from a lateral pelvis pair to the spine-root pivot.

    Rigify-style rigs expose no central Hip bone; the lowest spine bone is the
    true pelvis pivot (it parents the legs and the rest of the spine, head at the
    pelvis), while ``pelvis.L/.R`` are lateral deform helpers. When Hip resolved to
    such a lateral pelvis pair AND a spine-chain root sits at the pelvis, move Hip
    onto the spine root (real, non-degenerate geometry) and shift Lower_Spine up to
    the next spine segment so the two no longer collide. The displaced pelvis pair
    is recorded for preservation as Hip children (mesh drivers).

    No-ops for rigs with an explicit named Hip (those never carry the
    ``paired_sided_pelvis_requires_centering`` note), preserving their mapping.
    """
    hip_payload = resolved_targets.get("Hip", {})
    if "paired_sided_pelvis_requires_centering" not in hip_payload.get("notes", []):
        return

    spine_chain = context.get("spine_chain") or []
    if len(spine_chain) < 2:
        return

    bones = context["bones"]
    spine_root_name, next_spine_name = spine_chain[0], spine_chain[1]
    spine_root = bones.get(spine_root_name)
    next_spine = bones.get(next_spine_name)
    if spine_root is None or next_spine is None:
        return

    up = context["height_axis"]
    pelvis_name = hip_payload.get("source_bone")
    pelvis_bone = bones.get(pelvis_name) if pelvis_name else None
    if pelvis_bone is not None:
        # The spine root must sit at the pelvis (same level as the displaced pelvis
        # pair) to be the Hip pivot; otherwise leave the mapping alone.
        if abs(spine_root.head[up] - pelvis_bone.head[up]) > 0.05 * context["height_span"]:
            return

    # Do not steal a bone another target already claims.
    claimed = {
        payload.get("source_bone")
        for target, payload in resolved_targets.items()
        if target not in {"Hip", "Lower_Spine"} and payload.get("source_bone")
    }
    if spine_root_name in claimed or next_spine_name in claimed:
        return

    # The previous Lower_Spine payload describes the spine-root bone's geometry;
    # reuse its confidence/evidence for the reassigned Hip.
    prev_lower = resolved_targets.get("Lower_Spine", {})
    pelvis_pair = [pelvis_name] if pelvis_name else []
    mirror = _mirror_sided_bone_name(pelvis_name) if pelvis_name else None
    if mirror and mirror in bones and mirror not in pelvis_pair:
        pelvis_pair.append(mirror)

    resolved_targets["Hip"] = {
        "source_bone": spine_root_name,
        "confidence": prev_lower.get("confidence", 0.7) if prev_lower.get("source_bone") == spine_root_name else 0.7,
        "action": "alias_map",
        "decision": "structure_only",
        "evidence": copy.deepcopy(prev_lower.get("evidence", {
            "name": 0.0,
            "hierarchy": 1.0,
            "geometry": 0.0,
            "source_origin": spine_root.origin,
            "role": spine_root.role,
        })),
        "notes": ["hip_reassigned_to_spine_root_pivot"],
        "preserved_pelvis_pair_sources": sorted(pelvis_pair),
    }

    # Hip now owns the spine root, so Lower_Spine must sit on the next segment up.
    # This holds whether Lower_Spine had grabbed the spine root (older name-led
    # path) or already resolved to the next segment (structure-led path, where
    # the root is recognised as the hip and excluded from lower-spine scoring).
    if prev_lower.get("source_bone") in {spine_root_name, next_spine_name}:
        action = prev_lower.get("action")
        moved = prev_lower.get("source_bone") == spine_root_name
        notes = list(prev_lower.get("notes", []))
        if moved:
            notes.append("lower_spine_shifted_above_hip_pivot")
        prev_evidence = prev_lower.get("evidence", {})
        resolved_targets["Lower_Spine"] = {
            "source_bone": next_spine_name,
            "confidence": prev_lower.get("confidence", 0.7),
            "action": action if action in {"direct_map", "alias_map"} else "alias_map",
            "decision": _reconciliation_tag(
                prev_evidence.get("name", 0.0),
                prev_evidence.get("structural", 0.0),
            ),
            "evidence": {
                "name": prev_evidence.get("name", 0.0),
                "hierarchy": 1.0,
                "geometry": prev_evidence.get("geometry", 0.0),
                "source_origin": next_spine.origin,
                "role": next_spine.role,
            },
            "notes": sorted(set(notes)),
        }


def _resolve_root_compliance(resolved_targets: Dict[str, dict], root_candidates: List[CandidateScore], context: dict) -> dict:
    original_payload = copy.deepcopy(resolved_targets["Root"])
    candidate = root_candidates[0] if root_candidates else None
    candidate_bone = context["bones"].get(candidate.source_bone) if candidate is not None else None
    hip_payload = resolved_targets.get("Hip", {})
    hip_bone = context["bones"].get(hip_payload.get("source_bone")) if hip_payload.get("source_bone") else None
    source_translation_offset = _compute_source_translation_offset(candidate_bone, context)
    target_head, target_tail = _build_root_target_geometry(hip_bone, context, source_translation_offset)
    blocker_codes = _root_blocker_codes(original_payload, candidate, hip_bone, context)
    advisory_codes = _root_advisory_codes(candidate, context)
    diagnostics = _build_root_compliance_diagnostics(candidate_bone, context, blocker_codes)
    can_create_new_root = target_head is not None and target_tail is not None
    source_bone = candidate.source_bone if candidate_bone is not None and "root" in candidate_bone.family_tags else None

    placement_metadata = context["placement_metadata"]
    grp_root_local_origin = [float(value) for value in placement_metadata["bbox_ground_center"]]

    # Disabled for now: the source armature is left untouched and already retains its
    # own root, so duplicating it into the generated armature produces a confusing
    # second (often non-vertical) root. Re-enable by restoring the candidate-based
    # condition below.
    preserve_source_root_as_extra = False and (
        candidate is not None
        and candidate_bone is not None
        and candidate.role not in {"control", "mechanism"}
    )

    if candidate is not None and not blocker_codes:
        action = "direct_map" if candidate.source_bone == "Root" and candidate.name_evidence >= 0.95 else "alias_map"
        resolved_targets["Root"] = _target_payload_from_candidate(candidate, action, notes=[])
        return {
            "mode": "reuse_existing_root",
            "target_bone": "Root",
            "source_bone": candidate.source_bone,
            "rename_source_to_target": candidate.source_bone != "Root",
            "subtree_name": "Grp_Root",
            "grp_root_local_origin": grp_root_local_origin,
            "blocker_codes": [],
            "advisories": advisory_codes,
            "spec_references": list(ROOT_ORIGIN_SPEC_REFERENCES),
            "diagnostic_notes": diagnostics,
            "up_axis": copy.deepcopy(placement_metadata["up_axis"]),
        }

    if can_create_new_root:
        notes = sorted(set(list(original_payload.get("notes", [])) + blocker_codes))
        resolved_targets["Root"] = _target_payload_from_candidate(candidate, "repair_in_builder", notes=notes)
        return {
            "mode": "create_new_root",
            "target_bone": "Root",
            "source_bone": source_bone,
            "rename_source_to_target": False,
            "subtree_name": "Grp_Root",
            "grp_root_local_origin": grp_root_local_origin,
            "preserve_source_root_as_extra": (
                preserve_source_root_as_extra and source_bone is not None
            ),
            "blocker_codes": blocker_codes,
            "advisories": advisory_codes,
            "spec_references": list(ROOT_ORIGIN_SPEC_REFERENCES),
            "diagnostic_notes": diagnostics,
            "up_axis": copy.deepcopy(placement_metadata["up_axis"]),
        }

    notes = sorted(set(list(original_payload.get("notes", [])) + blocker_codes + ["insufficient_root_builder_inputs"]))
    resolved_targets["Root"] = _target_payload_from_candidate(candidate, "review", notes=notes)
    return {
        "mode": "review",
        "target_bone": "Root",
        "source_bone": source_bone,
        "rename_source_to_target": False,
        "subtree_name": "Grp_Root",
        "grp_root_local_origin": grp_root_local_origin,
        "blocker_codes": sorted(set(blocker_codes + ["insufficient_root_builder_inputs"])),
        "advisories": advisory_codes,
        "spec_references": list(ROOT_ORIGIN_SPEC_REFERENCES),
        "diagnostic_notes": diagnostics,
        "up_axis": copy.deepcopy(placement_metadata["up_axis"]),
    }


def _target_payload_from_candidate(candidate: Optional[CandidateScore], action: str, notes: List[str]) -> dict:
    if candidate is None:
        return {
            "source_bone": None,
            "confidence": 0.0,
            "action": action,
            "decision": "conflict_review",
            "evidence": {
                "name": 0.0,
                "hierarchy": 0.0,
                "geometry": 0.0,
                "source_origin": None,
                "role": None,
            },
            "notes": list(notes),
        }

    return {
        "source_bone": candidate.source_bone,
        "confidence": round(candidate.confidence, 3),
        "action": action,
        "decision": _reconciliation_tag(candidate.name_evidence, candidate.structural_evidence),
        "evidence": {
            "name": candidate.name_evidence,
            "hierarchy": candidate.hierarchy_evidence,
            "geometry": candidate.geometry_evidence,
            "source_origin": candidate.source_origin,
            "role": candidate.role,
        },
        "notes": list(notes),
    }


def _compute_source_translation_offset(
    candidate_bone: Optional[BoneInfo],
    context: dict,
) -> List[float]:
    placement = context.get("placement_metadata") or {}
    ground_center = placement.get("bbox_ground_center")
    if candidate_bone is None or ground_center is None:
        return [0.0, 0.0, 0.0]
    return [float(ground_center[index]) - float(candidate_bone.head[index]) for index in range(3)]


def _build_root_target_geometry(
    hip_bone: Optional[BoneInfo],
    context: dict,
    source_translation_offset: Sequence[float],
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    placement = context.get("placement_metadata") or {}
    ground_center = placement.get("bbox_ground_center")
    up_axis = (placement.get("up_axis") or {}).get("index")

    if ground_center is None or up_axis is None or hip_bone is None:
        return None, None

    up_axis_index = int(up_axis)
    target_head = [float(value) for value in ground_center]
    target_tail = list(target_head)
    offset_up = float(source_translation_offset[up_axis_index])
    target_tail[up_axis_index] = float(hip_bone.head[up_axis_index]) + offset_up

    return target_head, target_tail


def _root_advisory_codes(
    candidate: Optional[CandidateScore],
    context: dict,
) -> List[str]:
    """Advisory diagnostic codes for the source root.

    Each advisory is `<code>:<magnitude>` where magnitude is meters, rounded to 6
    decimals. These no longer block `reuse_existing_root`; they surface offsets that
    are intrinsic to the source asset (source-root vs mesh-bbox center mismatch).
    """
    if candidate is None:
        return []
    candidate_bone = context["bones"].get(candidate.source_bone)
    if candidate_bone is None:
        return []

    placement = context.get("placement_metadata") or {}
    ground_center = placement.get("bbox_ground_center")
    if ground_center is None:
        return []
    forward_axis = int(placement["forward_axis"]["index"])
    side_axis = int(placement["side_axis"]["index"])
    up_axis = int(placement["up_axis"]["index"])

    forward_delta = candidate_bone.head[forward_axis] - float(ground_center[forward_axis])
    side_delta = candidate_bone.head[side_axis] - float(ground_center[side_axis])
    up_delta = candidate_bone.head[up_axis] - float(ground_center[up_axis])

    planar_magnitude = math.sqrt((forward_delta * forward_delta) + (side_delta * side_delta))
    ground_magnitude = abs(up_delta)

    advisories: List[str] = []
    if planar_magnitude > 1e-6:
        advisories.append(
            "root_head_off_ground_center_advisory:{0:.6f}".format(planar_magnitude)
        )
    if ground_magnitude > 1e-6:
        advisories.append(
            "root_head_off_ground_advisory:{0:.6f}".format(ground_magnitude)
        )
    return advisories


def _root_blocker_codes(
    original_payload: dict,
    candidate: Optional[CandidateScore],
    hip_bone: Optional[BoneInfo],
    context: dict,
) -> List[str]:
    failure_codes: List[str] = []
    original_notes = set(original_payload.get("notes", []))
    if "candidate_conflict" in original_notes:
        failure_codes.append("candidate_conflict")

    if candidate is None:
        failure_codes.append("no_root_candidate")
        if hip_bone is None:
            failure_codes.append("hip_unresolved")
        return sorted(set(failure_codes))

    candidate_bone = context["bones"].get(candidate.source_bone)
    if candidate_bone is None:
        failure_codes.append("no_root_candidate")
        return sorted(set(failure_codes))

    if "root" not in candidate_bone.family_tags:
        failure_codes.append("no_root_candidate")
        failure_codes.append("root_candidate_missing_root_family")

    if candidate.role in {"control", "mechanism"}:
        failure_codes.append("root_candidate_disallowed_role")

    if hip_bone is None:
        failure_codes.append("hip_unresolved")

    actual_roots = [bone for bone in context["bones"].values() if bone.parent is None]
    if len(actual_roots) != 1:
        failure_codes.append("multiple_source_roots")
    elif actual_roots[0].name != candidate.source_bone:
        failure_codes.append("root_candidate_not_source_root")

    placement = context["placement_metadata"]
    bbox_height = max(float(placement.get("bbox_height", 0.0)), 1e-6)
    up_axis = int(placement["up_axis"]["index"])
    up_sign = int(placement["up_axis"]["sign"])

    direction_alignment = _aligned_axis_cosine(candidate_bone, up_axis, up_sign)
    if direction_alignment < ROOT_UP_ALIGNMENT_COSINE:
        failure_codes.append("root_not_vertical")

    if hip_bone is not None:
        tail_tolerance = bbox_height * ROOT_TAIL_TO_HIP_TOLERANCE_RATIO
        if abs(candidate_bone.tail[up_axis] - hip_bone.head[up_axis]) > tail_tolerance:
            failure_codes.append("root_tail_mismatch_hip")
        if candidate_bone.child_names != [hip_bone.name]:
            failure_codes.append("root_children_not_hip_only")

    return sorted(set(failure_codes))


def _build_root_compliance_diagnostics(
    candidate_bone: Optional[BoneInfo],
    context: dict,
    failure_codes: List[str],
) -> List[str]:
    if candidate_bone is None:
        return []

    placement = context["placement_metadata"]
    ground_center = placement["bbox_ground_center"]
    forward_axis = int(placement["forward_axis"]["index"])
    side_axis = int(placement["side_axis"]["index"])
    up_axis = int(placement["up_axis"]["index"])

    forward_delta = candidate_bone.head[forward_axis] - float(ground_center[forward_axis])
    side_delta = candidate_bone.head[side_axis] - float(ground_center[side_axis])
    up_delta = candidate_bone.head[up_axis] - float(ground_center[up_axis])
    planar_delta = math.sqrt((forward_delta * forward_delta) + (side_delta * side_delta))

    notes = [
        "root_head_delta_forward={0:.6f}".format(forward_delta),
        "root_head_delta_side={0:.6f}".format(side_delta),
        "root_head_delta_up={0:.6f}".format(up_delta),
    ]
    if "root_head_off_ground_center" in failure_codes:
        notes.append("root_origin_violation_against_asam_7_3_3_1")
    if "root_head_off_ground" in failure_codes:
        notes.append("root_ground_projection_violation_against_asam_7_3_3_1")

    # Distinguish root-bone non-compliance from likely mesh-to-skeleton offset.
    is_local_origin = all(abs(candidate_bone.head[index]) <= 1e-6 for index in range(3))
    if is_local_origin and planar_delta > 1e-6:
        notes.append("mesh_bounds_offset_detected_root_at_local_origin")

    return notes


def _aligned_axis_cosine(bone: BoneInfo, axis_index: int, axis_sign: int) -> float:
    direction = [bone.tail[index] - bone.head[index] for index in range(3)]
    length = math.sqrt(sum(component * component for component in direction))
    if length <= 1e-6:
        return 0.0
    axis_vector = [0.0, 0.0, 0.0]
    axis_vector[axis_index] = float(axis_sign)
    dot = sum((direction[index] / length) * axis_vector[index] for index in range(3))
    return abs(dot)


def _classify_non_mapped_bones(
    bones: Dict[str, BoneInfo],
    context: dict,
    mapped_bones: Set[str],
) -> Tuple[List[dict], List[dict]]:
    preserved: List[dict] = []
    unclassified: List[dict] = []

    spine_chain = set(context["spine_chain"])
    arm_chain_bones = set()
    for chain in context["arm_chains"].values():
        arm_chain_bones.update(chain)
    leg_chain_bones = set()
    for chain in context["leg_chains"].values():
        leg_chain_bones.update(chain)

    for bone_name in sorted(bones):
        if bone_name in mapped_bones:
            continue
        bone = bones[bone_name]
        reason: Optional[str] = None

        if bone.role in {"control", "mechanism"}:
            reason = "control_or_mechanism"
        elif any(tag in bone.family_tags for tag in {"toe", "eye", "finger", "thumb"}):
            reason = "deferred_semantic_detail"
        elif bone_name in spine_chain or bone_name in arm_chain_bones or bone_name in leg_chain_bones:
            reason = "supporting_chain_detail"
        elif any(token in EXTRA_TOKENS for token in bone.tokens):
            reason = "named_extra"

        payload = {
            "bone_name": bone.name,
            "reason": reason or "unclassified",
            "origin": bone.origin,
            "role": bone.role,
        }
        if reason:
            preserved.append(payload)
        else:
            unclassified.append(payload)

    return preserved, unclassified


def _plan_single_spine_split(resolved_targets: Dict[str, dict], context: dict) -> None:
    """Plan a geometric midpoint split of a lone spine bone (issue #56).

    Trigger: the chosen spine chain has exactly one usable bone AND Hip and Neck
    are already classified. By then the in-between bone is unambiguously spine
    (topology brackets it), so both ASAM spine targets are pointed at it and tagged
    `split_in_builder`; the builder bisects it at its midpoint. No-op for 0 or >=2
    spine bones, or when Hip/Neck are not classified (no bracket -> existing
    fallback). Mutates `resolved_targets` in place."""
    spine_chain = context.get("spine_chain") or []
    if len(spine_chain) != 1:
        return
    if resolved_targets["Hip"]["action"] not in RECOVERABLE_ACTIONS:
        return
    if resolved_targets["Neck"]["action"] not in {"direct_map", "alias_map"}:
        return

    spine_bone = spine_chain[0]
    for target, half in (("Lower_Spine", "lower"), ("Upper_Spine", "upper")):
        payload = resolved_targets[target]
        payload["action"] = SPLIT_ACTION
        payload["source_bone"] = spine_bone
        payload["split_half"] = half
        payload["confidence"] = 0.7
        if "single_spine_geometric_split" not in payload["notes"]:
            payload["notes"].append("single_spine_geometric_split")


def _demote_neck_coincident_upper_spine(resolved_targets: Dict[str, dict], context: dict) -> None:
    """Keep ASAM Upper_Spine strictly below Neck (issue #73).

    On some rigs the spine-segment chosen for Upper_Spine has its head at or above the
    joint the Neck bone starts from (e.g. Rigify `DEF-spine.004` head == `neck` head),
    so the two bones share a joint and render overlapping. The structural channel cannot
    see this — the colliding joint belongs to the parallel name-alias Neck bone, not the
    structural spine column — so we enforce the invariant here, after both targets have
    resolved to concrete source bones.

    When Upper_Spine's source head is not strictly below Neck's source head along the up
    axis, re-point Upper_Spine to the nearest ancestor spine segment whose head IS strictly
    below Neck (without stealing Lower_Spine's source). If no such segment exists, leave the
    mapping unchanged and annotate it for review. Only acts on directly-mapped spine
    segments; split/created/review mappings are left alone. Mutates `resolved_targets`."""
    upper = resolved_targets.get("Upper_Spine", {})
    neck = resolved_targets.get("Neck", {})
    if upper.get("action") not in {"direct_map", "alias_map"}:
        return
    if neck.get("action") not in {"direct_map", "alias_map"}:
        return

    bones = context["bones"]
    upper_bone = bones.get(upper.get("source_bone"))
    neck_bone = bones.get(neck.get("source_bone"))
    if upper_bone is None or neck_bone is None:
        return

    up = context["height_axis"]
    sign = context["height_sign"]
    eps = 1e-4
    neck_h = neck_bone.head[up] * sign

    # Already strictly below the neck joint -> nothing to do.
    if upper_bone.head[up] * sign < neck_h - eps:
        return

    lower_source = resolved_targets.get("Lower_Spine", {}).get("source_bone")

    # Walk up the source parent chain to the nearest ancestor strictly below the neck.
    chosen = None
    node = upper_bone.parent
    while node is not None:
        if node == lower_source:
            break  # don't collide with Lower_Spine; treat as degenerate
        candidate = bones.get(node)
        if candidate is None:
            break
        if candidate.head[up] * sign < neck_h - eps:
            chosen = node
            break
        node = candidate.parent

    if chosen is None:
        if "neck_coincident_upper_spine_unresolved" not in upper["notes"]:
            upper["notes"].append("neck_coincident_upper_spine_unresolved")
        return

    upper["source_bone"] = chosen
    if "neck_coincident_upper_spine_demoted" not in upper["notes"]:
        upper["notes"].append("neck_coincident_upper_spine_demoted")


def _build_review_flags(resolved_targets: Dict[str, dict], root_resolution: dict, context: dict) -> List[str]:
    flags: Set[str] = set()
    actual_roots = [bone for bone in context["bones"].values() if bone.parent is None]

    if len(actual_roots) > 1:
        flags.add("multiple_roots")
    if root_resolution.get("mode") != "reuse_existing_root":
        flags.add("root_noncompliant")
    if not context["spine_chain"] or any(
        resolved_targets[target]["action"] == "create_in_builder" for target in ("Lower_Spine", "Upper_Spine")
    ):
        flags.add("missing_spine_chain")
    if not context["leg_chains"].get("left") or not context["leg_chains"].get("right"):
        flags.add("no_leg_chain")
    if any(
        payload["source_bone"] is not None and payload["evidence"]["role"] in {"control", "mechanism"}
        for payload in resolved_targets.values()
    ):
        flags.add("control_bone_selected")
    if any("geometry_side_conflict" in payload["notes"] for payload in resolved_targets.values()):
        flags.add("side_conflict")
    if resolved_targets["Head"]["action"] != "direct_map" or resolved_targets["Neck"]["action"] not in {"direct_map", "alias_map"}:
        flags.add("insufficient_head_evidence")

    return sorted(flags)


def _collect_mesh_binding_stats(
    mesh_binding: Optional[dict],
    armature_name: str,
    armature_bone_names_lower: Set[str],
) -> dict:
    """Walk mesh_binding once and return the raw stats needed by armature scoring."""
    empty = {
        "meshes_count": 0,
        "has_armature_modifier_link": False,
        "vertex_group_bone_hits": 0,
        "vertex_group_count": 0,
    }
    if not mesh_binding:
        return dict(empty)
    meshes = mesh_binding.get("meshes") or []
    if not meshes:
        return dict(empty)

    has_armature_modifier_link = False
    vertex_group_names_lower: Set[str] = set()
    for mesh in meshes:
        for modifier in mesh.get("modifiers") or []:
            if modifier.get("type") == "ARMATURE" and modifier.get("object") == armature_name:
                has_armature_modifier_link = True
        for group_name in mesh.get("vertex_groups") or []:
            vertex_group_names_lower.add(_strip_vertex_group_prefix(group_name).lower())

    return {
        "meshes_count": len(meshes),
        "has_armature_modifier_link": has_armature_modifier_link,
        "vertex_group_bone_hits": len(armature_bone_names_lower & vertex_group_names_lower),
        "vertex_group_count": len(vertex_group_names_lower),
    }


def _compute_mesh_bound_term(meshes_count: int, has_armature_modifier_link: bool) -> float:
    """Score how strongly an armature is bound to a driven mesh.

    Returns 0.0 (no meshes), 6.0 (meshes present, no ARMATURE modifier link
    to this armature), or 10.0 (at least one ARMATURE modifier links to it).
    """
    if meshes_count <= 0:
        return 0.0
    return 10.0 if has_armature_modifier_link else 6.0


def _strip_vertex_group_prefix(name: str) -> str:
    """Strip a leading 'DEF-' prefix (case-insensitive) for matching against bone names."""
    if name[:4].lower() == "def-":
        return name[4:]
    return name


def _compute_deform_evidence_term(vertex_group_bone_hits: int) -> float:
    """Score how many of this armature's bones appear as vertex groups on driven meshes.

    Result is dampened by log1p and capped at 6.0.
    """
    return round(min(math.log1p(vertex_group_bone_hits) * 2.0, 6.0), 3)


def _compute_extras_term(extras_preserved: List[dict]) -> float:
    """Score the count of preserved anatomical extras. Dampened by log1p, capped at 2.0."""
    return round(min(math.log1p(len(extras_preserved)) * 0.5, 2.0), 3)


def _compute_vertex_group_coverage(vertex_group_bone_hits: int, vertex_group_count: int) -> float:
    """Fraction of the bound mesh's vertex groups this armature's bones cover (0..1).

    Convention-agnostic skin-weight signal: how much of the skin this armature
    accounts for. A true deform rig approaches 1.0; a pure control rig is low.
    """
    return round(vertex_group_bone_hits / max(vertex_group_count, 1), 3)


def _compute_deform_purity(vertex_group_bone_hits: int, armature_bone_count: int) -> float:
    """Fraction of this armature's bones that actually drive skin (0..1).

    Breaks a coverage tie between a clean deform rig (~1.0) and a bushy superset
    control rig that copies the deform bone names but adds scaffolding bones (low).
    """
    return round(vertex_group_bone_hits / max(armature_bone_count, 1), 3)


def _build_armature_summary(
    resolved_targets: Dict[str, dict],
    review_flags: List[str],
    mesh_binding: Optional[dict],
    extras_preserved: List[str],
    armature_bones: Dict[str, BoneInfo],
    armature_name: str,
    name_quality: float,
) -> dict:
    mapped = [payload for payload in resolved_targets.values() if payload["action"] in MAPPED_ACTIONS]
    average_confidence = round(sum(payload["confidence"] for payload in mapped) / max(len(mapped), 1), 3)
    direct_map_targets = sum(1 for payload in resolved_targets.values() if payload["action"] == "direct_map")
    alias_map_targets = sum(1 for payload in resolved_targets.values() if payload["action"] == "alias_map")
    repair_targets = sum(1 for payload in resolved_targets.values() if payload["action"] == "repair_in_builder")
    review_targets = sum(1 for payload in resolved_targets.values() if payload["action"] == "review")
    missing_targets = sum(1 for payload in resolved_targets.values() if payload["action"] == "create_in_builder")

    armature_bone_names_lower = {name.lower() for name in armature_bones.keys()}
    mesh_stats = _collect_mesh_binding_stats(mesh_binding, armature_name, armature_bone_names_lower)

    mesh_bound_term = _compute_mesh_bound_term(
        mesh_stats["meshes_count"],
        mesh_stats["has_armature_modifier_link"],
    )
    deform_evidence_term = _compute_deform_evidence_term(mesh_stats["vertex_group_bone_hits"])
    extras_term = _compute_extras_term(extras_preserved)
    vertex_group_coverage = _compute_vertex_group_coverage(
        mesh_stats["vertex_group_bone_hits"],
        mesh_stats["vertex_group_count"],
    )
    deform_purity = _compute_deform_purity(
        mesh_stats["vertex_group_bone_hits"],
        len(armature_bones),
    )

    ranking_score = round(
        (len(mapped) * 5.0)
        + (average_confidence * 10.0)
        + mesh_bound_term
        + deform_evidence_term
        + extras_term
        - (review_targets * 0.75)
        - (missing_targets * 1.2)
        - (len(review_flags) * 0.25),
        3,
    )

    return {
        "mapped_core_targets": len(mapped),
        "direct_map_targets": direct_map_targets,
        "alias_map_targets": alias_map_targets,
        "repair_targets": repair_targets,
        "review_targets": review_targets,
        "missing_core_targets": missing_targets,
        "average_confidence": average_confidence,
        "mesh_bound_term": mesh_bound_term,
        "deform_evidence_term": deform_evidence_term,
        "extras_term": extras_term,
        "vertex_group_coverage": vertex_group_coverage,
        "deform_purity": deform_purity,
        "deform_evidence": mesh_stats,
        "ranking_score": ranking_score,
        "name_quality": round(name_quality, 3),
    }


def _build_proposed_hierarchy(asset_name: str) -> dict:
    armature_name = "Armature_{0}".format(asset_name)
    bone_parents = {
        target: parent.format(asset_name=asset_name) if "{asset_name}" in parent else parent
        for target, parent in TARGET_PARENTS.items()
    }
    return {
        "object_nodes": [
            {"name": "Grp_Root", "parent": None},
            {"name": armature_name, "parent": "Grp_Root"},
        ],
        "bone_parents": bone_parents,
    }


def _missing_target_payload(note: str) -> dict:
    return {
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
        "notes": [note],
    }


def _paired_sided_pelvis_requires_centering(
    target_name: str,
    candidate: CandidateScore,
    target_candidates: Dict[str, List[CandidateScore]],
    context: dict,
) -> bool:
    if target_name != "Hip":
        return False

    bone = context["bones"].get(candidate.source_bone)
    if bone is None:
        return False
    if bone.side not in {"left", "right"}:
        return False
    if "hip" not in bone.family_tags:
        return False
    if not any(token in bone.compact_name for token in ("pelvis", "hip", "hips")):
        return False

    for alternate in target_candidates.get("Hip", []):
        if alternate.source_bone == candidate.source_bone:
            continue
        alternate_bone = context["bones"].get(alternate.source_bone)
        if alternate_bone is None:
            continue
        if alternate_bone.side not in {"left", "right"}:
            continue
        if alternate_bone.side == bone.side:
            continue
        if "hip" not in alternate_bone.family_tags:
            continue
        if not any(token in alternate_bone.compact_name for token in ("pelvis", "hip", "hips")):
            continue
        return True

    return False


def _target_side(target_name: str) -> Optional[str]:
    if target_name.endswith("_Left"):
        return "left"
    if target_name.endswith("_Right"):
        return "right"
    return None


def _name_evidence(target_name: str, bone: BoneInfo, convention: Convention = "none") -> float:
    family = CORE_FAMILY_BY_TARGET[target_name]
    side = _target_side(target_name)
    base_score = _family_name_score(family, bone, convention)

    if side and bone.side and bone.side != side:
        base_score *= 0.05
    if side and bone.side == side:
        base_score = min(1.0, base_score + 0.05)

    return _clamp(base_score)


def _family_name_score(family: str, bone: BoneInfo, convention: Convention = "none") -> float:
    if convention != "none":
        alias_map = BONE_CONVENTIONS[convention].alias_families
        mapped = alias_map.get(_compact_stem(bone.compact_name))
        if mapped is not None:
            return 0.95 if family in mapped else 0.0

    tokens = set(bone.tokens)
    compact = bone.compact_name

    if family == "root":
        if compact in {"root", "master", "armatureroot"}:
            return 1.0
        if "root" in compact:
            return 0.9
        return 0.0

    if family == "hip":
        if compact in {"hip", "hips", "pelvis"}:
            return 1.0
        if "pelvis" in compact or "hips" in compact or "hip" in tokens:
            return 0.95
        return 0.0

    if family == "lower_spine":
        if compact == "lowerspine":
            return 1.0
        if "torso" in tokens or "abdomen" in tokens:
            return 0.92
        if compact.startswith("spine") or "spine" in tokens:
            return 0.72
        return 0.0

    if family == "upper_spine":
        if compact == "upperspine":
            return 1.0
        if "chest" in tokens or "thorax" in tokens:
            return 0.98
        if compact.startswith("spine") or "spine" in tokens:
            return 0.72
        return 0.0

    if family == "neck":
        return 1.0 if ("neck" in tokens or compact == "neck") else 0.0

    if family == "head":
        return 1.0 if ("head" in tokens or compact == "head") else 0.0

    if family == "shoulder":
        return 1.0 if {"shoulder", "clavicle", "collar"} & tokens else 0.0

    if family == "upper_arm":
        if "upperarm" in compact or "bicep" in compact:
            return 1.0
        if "upper" in tokens and "arm" in tokens:
            return 0.96
        if compact == "arm":
            return 0.55
        return 0.0

    if family == "lower_arm":
        if "forearm" in compact:
            return 1.0
        if "lower" in tokens and "arm" in tokens:
            return 0.96
        if compact == "arm":
            return 0.42
        return 0.0

    if family == "hand":
        if compact == "hand":
            return 1.0
        if "wrist" in tokens:
            return 0.82
        return 0.0

    if family == "upper_leg":
        if "thigh" in compact:
            return 1.0
        if "upper" in tokens and "leg" in tokens:
            return 0.96
        if compact == "leg":
            return 0.55
        return 0.0

    if family == "lower_leg":
        if "shin" in compact or "calf" in compact:
            return 1.0
        if "lower" in tokens and "leg" in tokens:
            return 0.96
        if compact == "leg":
            return 0.42
        return 0.0

    if family == "foot":
        if compact == "foot":
            return 1.0
        if "ankle" in tokens:
            return 0.82
        return 0.0

    if family == "eye":
        if compact == "eye" or compact == "eyeball":
            return 1.0
        if "eye" in tokens:
            return 0.90
        return 0.0

    if family == "full_thumb":
        if "fullthumb" in compact or compact == "thumb":
            return 1.0
        if "thumb" in tokens:
            return 0.90
        return 0.0

    if family == "full_fingers":
        if "fullfingers" in compact:
            return 1.0
        if "finger" in tokens or compact in {"index", "middle", "ring", "pinky"}:
            return 0.80
        return 0.0

    if family == "full_toes":
        if "fulltoes" in compact:
            return 1.0
        if "toe" in tokens or "toes" in tokens:
            return 0.90
        return 0.0

    return 0.0


def _hierarchy_evidence(target_name: str, bone: BoneInfo, context: dict) -> float:
    family = CORE_FAMILY_BY_TARGET[target_name]
    side = _target_side(target_name)
    score = 0.0

    parent = context["bones"].get(bone.parent) if bone.parent else None
    children = [context["bones"][child_name] for child_name in bone.child_names if child_name in context["bones"]]

    spine_chain = context["spine_chain"]
    arm_chain = context["arm_chains"].get(side) if side else []
    leg_chain = context["leg_chains"].get(side) if side else []

    if family == "root":
        if bone.parent is None:
            score += 0.55
        if _has_descendant_family(bone.name, context, {"hip", "lower_spine", "upper_spine"}):
            score += 0.25
        if _has_descendant_family(bone.name, context, {"upper_leg", "lower_leg", "foot"}):
            score += 0.20

    elif family == "hip":
        if _has_descendant_family(bone.name, context, {"upper_leg", "lower_leg", "foot"}):
            score += 0.45
        if _has_descendant_family(bone.name, context, {"lower_spine", "upper_spine", "spine"}):
            score += 0.30
        if parent and "root" in parent.family_tags:
            score += 0.15
        if any("pelvis" in child.compact_name for child in children):
            score += 0.10

    elif family == "lower_spine":
        score = max(score, _score_spine_position(bone.name, spine_chain, prefer_end=False))
        if parent and ("hip" in parent.family_tags or "root" in parent.family_tags):
            score += 0.25
        if any("upper_spine" in child.family_tags or "neck" in child.family_tags or "spine" in child.family_tags for child in children):
            score += 0.20

    elif family == "upper_spine":
        score = max(score, _score_spine_position(bone.name, spine_chain, prefer_end=True))
        if parent and ("lower_spine" in parent.family_tags or "spine" in parent.family_tags):
            score += 0.20
        if any({"neck", "shoulder", "head"} & child.family_tags for child in children):
            score += 0.30

    elif family == "neck":
        if parent and ("upper_spine" in parent.family_tags or "spine" in parent.family_tags):
            score += 0.45
        if any("head" in child.family_tags for child in children):
            score += 0.40
        if _is_highest_named_head_neighbor(bone, context):
            score += 0.15

    elif family == "head":
        if parent and ("neck" in parent.family_tags or "upper_spine" in parent.family_tags):
            score += 0.35
        if not children or all({"eye", "face", "finger", "thumb"} & child.family_tags for child in children):
            score += 0.35
        if _is_highest_in_family_group(bone, context, {"head", "neck", "upper_spine"}):
            score += 0.30

    elif family == "shoulder":
        if parent and ("upper_spine" in parent.family_tags or "spine" in parent.family_tags):
            score += 0.30
        if any("upper_arm" in child.family_tags for child in children):
            score += 0.40
        if _chain_side_match(arm_chain, bone.name):
            score += 0.20

    elif family == "upper_arm":
        score = max(score, _score_limb_position(bone.name, arm_chain, "start"))
        if parent and "shoulder" in parent.family_tags:
            score += 0.20
        if any({"lower_arm", "hand"} & child.family_tags for child in children):
            score += 0.20

    elif family == "lower_arm":
        score = max(score, _score_limb_position(bone.name, arm_chain, "middle"))
        if parent and ("upper_arm" in parent.family_tags or "shoulder" in parent.family_tags):
            score += 0.20
        if any("hand" in child.family_tags for child in children):
            score += 0.20

    elif family == "hand":
        score = max(score, _score_limb_position(bone.name, arm_chain, "end"))
        if parent and ("lower_arm" in parent.family_tags or "upper_arm" in parent.family_tags):
            score += 0.20
        if _has_descendant_family(bone.name, context, {"finger", "thumb"}):
            score += 0.20

    elif family == "upper_leg":
        score = max(score, _score_limb_position(bone.name, leg_chain, "start"))
        if parent and "hip" in parent.family_tags:
            score += 0.20
        if any({"lower_leg", "foot"} & child.family_tags for child in children):
            score += 0.20

    elif family == "lower_leg":
        score = max(score, _score_limb_position(bone.name, leg_chain, "middle"))
        if parent and ("upper_leg" in parent.family_tags or "hip" in parent.family_tags):
            score += 0.20
        if any("foot" in child.family_tags for child in children):
            score += 0.20

    elif family == "foot":
        score = max(score, _score_limb_position(bone.name, leg_chain, "end"))
        if parent and ("lower_leg" in parent.family_tags or "upper_leg" in parent.family_tags):
            score += 0.20
        if _has_descendant_family(bone.name, context, {"toe"}):
            score += 0.20

    elif family == "eye":
        if parent and "head" in parent.family_tags:
            score += 0.70
        if not children:
            score += 0.20

    elif family == "full_thumb":
        if parent and "hand" in parent.family_tags:
            score += 0.70
        if "thumb" in bone.family_tags:
            score += 0.20

    elif family == "full_fingers":
        if parent and "hand" in parent.family_tags:
            score += 0.60
        if "finger" in bone.family_tags:
            score += 0.30

    elif family == "full_toes":
        if parent and "foot" in parent.family_tags:
            score += 0.70
        if "toe" in bone.family_tags:
            score += 0.20

    return _clamp(score)


def _geometry_evidence(target_name: str, bone: BoneInfo, context: dict) -> float:
    family_label = target_name.rsplit("_", 1)[0] if target_name.endswith(("_Left", "_Right")) else target_name
    side = _target_side(target_name)

    if family_label == "Root":
        center_score = _centerline_score(bone, context)
        low_score = _band_score(_endpoint_ratio(bone, context, "low"), 0.0, 0.12)
        return _clamp((0.55 * center_score) + (0.45 * low_score))

    if family_label in {"Hip", "Lower_Spine", "Upper_Spine", "Neck", "Head"}:
        height_score = _band_score(_height_ratio(bone, context), *HEIGHT_BANDS[family_label])
        center_score = _centerline_score(bone, context)
        return _clamp((0.6 * height_score) + (0.4 * center_score))

    height_score = _band_score(_height_ratio(bone, context), *HEIGHT_BANDS[family_label])
    side_score = _side_score(side, bone, context)
    return _clamp((0.6 * height_score) + (0.4 * side_score))


def _structural_evidence(target_name: str, bone: BoneInfo, context: dict) -> float:
    skel: StructuralSkeleton = context.get("structural_skeleton")
    if not skel:
        return 0.0

    family = CORE_FAMILY_BY_TARGET[target_name]
    side = _target_side(target_name)
    label = skel.labels.get(bone.name)

    if not label:
        return 0.0

    score = 0.0
    if label.family == family:
        score += label.confidence
    # Give partial credit if it's in the same broader category (e.g., upper vs lower leg)
    elif "leg" in label.family and "leg" in family:
        score += label.confidence * 0.5
    elif "arm" in label.family and "arm" in family:
        score += label.confidence * 0.5
    elif "spine" in label.family and "spine" in family:
        score += label.confidence * 0.5

    # The structural labeler never emits thumb/finger/toe/eye labels (it only
    # localises gross limb ends as hand/foot). When the structural family does
    # not match the target family at all, side agreement alone is NOT evidence
    # that the bone belongs to this target — e.g. a hand-labeled arm bone must
    # not earn positive structural credit for a Full_Thumb target just by being
    # on the same side. Only a matched (or broad-category) family carries a side
    # bonus; an opposite-side match on a genuine family still penalises.
    if score <= 0.0:
        return 0.0

    if side and label.side:
        if side == label.side:
            score += 0.2
        else:
            score -= 0.5

    return _clamp(score)


def compute_name_quality(context: dict) -> float:
    """Assess overall semantic clarity of the armature's bone names."""
    bones = context.get("bones", {})
    if not bones:
        return 0.0

    score = 0.0
    valid_bones = 0
    for bone in bones.values():
        if bone.origin != "primary":
            continue
        valid_bones += 1
        if bone.family_tags:
            score += 1.0
        elif bone.side is not None:
            # Credit a real side token (left/right), not a bare l/r substring
            # match — junk names like "control"/"lower" must stay LOW quality
            # so structure, not names, drives their mapping.
            score += 0.3

    if valid_bones == 0:
        return 0.0
    return score / valid_bones


def _confidence_weights(name_quality: float, origin: str) -> tuple[float, float, float]:
    """Return (w_name, w_struct, w_geom).
    If name_quality is high, names drive.
    If names are junk, structural topology drives.
    Support fallbacks (which only exist because they had valid names) always use high name weights.
    """
    if origin == "support":
        return (0.7, 0.1, 0.2)

    if name_quality >= 0.4:
        # High quality names: names drive
        return (0.8, 0.1, 0.1)
    elif name_quality >= 0.15:
        # Mediocre names: balanced weighting
        return (0.3, 0.4, 0.3)
    else:
        # Junk names: purely structural/geometry driven
        return (0.05, 0.6, 0.35)


def _numbered_stem(name: str) -> str:
    """Strip a trailing Blender duplicate suffix (``.001``, ``.012``) from a bone name."""
    return re.sub(r"\.\d+$", "", name)


def _runner_up_is_interchangeable(
    candidate: CandidateScore,
    runner_up: CandidateScore,
    target_name: str,
    context: dict,
) -> bool:
    """True when the runner-up is anatomically interchangeable with the candidate,
    so a near-zero separation must NOT demote a real mapping to review.

    Covers two cases:
    - Blender duplicate/twist twins and same-deform-chain segments that share a
      numbered stem ('DEF-spine.004' vs 'DEF-spine.005').
    - Adjacent segments of the same chosen spine chain regardless of digit-naming
      style ('Spine0' vs 'Spine1', 'mixamorig:Spine' vs 'mixamorig:Spine1'). The
      stem check above misses these because the digit is concatenated, not a '.NNN'
      suffix. Chain-position scoring already picks the correct end, so any other
      segment of the same chain is an interchangeable runner-up.
    """
    if _numbered_stem(runner_up.source_bone) == _numbered_stem(candidate.source_bone):
        return True
    # Only the two spine targets draw candidates from a shared multi-segment chain;
    # the other 26 core targets map to distinct bones, so the chain-membership relief
    # is scoped to them. Extend this set if a future target shares a long chain.
    if target_name in {"Lower_Spine", "Upper_Spine"}:
        spine_chain = context.get("spine_chain") or []
        if candidate.source_bone in spine_chain and runner_up.source_bone in spine_chain:
            return True
    return False


def _decision_tag(
    target_name: str,
    candidate: CandidateScore,
    candidates_by_target: Dict[str, List[CandidateScore]],
    context: dict,
) -> str:
    """Label the reconciliation decision category based on confident separation."""
    # Check if there is a close runner-up
    alternates = candidates_by_target.get(target_name, [])
    runner_up = next((c for c in alternates if c.source_bone != candidate.source_bone), None)

    separation = 1.0
    if runner_up:
        # Interchangeable runner-ups (numbered twins, or adjacent segments of the
        # same chosen chain) must not collapse separation and demote a real mapping
        # to review. See _runner_up_is_interchangeable for the two covered cases.
        if _runner_up_is_interchangeable(candidate, runner_up, target_name, context):
            separation = 1.0
        else:
            separation = candidate.selection_score - runner_up.selection_score

    # A sided pelvis pair deliberately returns two equal Hip candidates.
    if target_name == "Hip" and runner_up and _paired_sided_pelvis_requires_centering(target_name, candidate, candidates_by_target, context):
        separation = 1.0

    if candidate.confidence < 0.25 and candidate.name_evidence < 0.20:
        return "create_in_builder"
    if candidate.confidence >= 0.80 and separation >= 0.05:
        return "direct_map"
    if candidate.confidence >= 0.60 and separation >= 0.02:
        return "alias_map"
    return "review"


def _reconciliation_tag(name_evidence: float, structural_evidence: float) -> str:
    """Human-readable reconciliation decision (closes a #29 auditability gap).

    Reports HOW the name vs structural channels agreed for a resolved target —
    distinct from ``_decision_tag`` (which returns the builder ACTION). The 0.20
    floor is the same minimum-evidence threshold used elsewhere to treat a
    channel as "present".
    """
    name_present = name_evidence >= 0.20
    structural_present = structural_evidence >= 0.20
    if name_present and structural_present:
        return "structure_confirmed_by_name"
    if name_present and not structural_present:
        return "name_override"
    if structural_present and not name_present:
        return "structure_only"
    return "conflict_review"


def _score_spine_position(bone_name: str, spine_chain: Sequence[str], prefer_end: bool) -> float:
    if bone_name not in spine_chain:
        return 0.0
    if len(spine_chain) == 1:
        return 0.70

    index = spine_chain.index(bone_name)
    position = index / float(len(spine_chain) - 1)
    return _clamp(0.45 + (0.55 * (position if prefer_end else (1.0 - position))))


def _score_limb_position(bone_name: str, chain: Sequence[str], desired_position: str) -> float:
    if bone_name not in chain:
        return 0.0
    if len(chain) == 1:
        return 0.60

    index = chain.index(bone_name)
    ratio = index / float(len(chain) - 1)

    if desired_position == "start":
        return _clamp(0.40 + (0.60 * (1.0 - ratio)))
    if desired_position == "middle":
        return _clamp(1.0 - (abs(ratio - 0.55) / 0.55))
    return _clamp(0.40 + (0.60 * ratio))


def _chain_side_match(chain: Sequence[str], bone_name: str) -> bool:
    return bone_name in chain


def _is_highest_named_head_neighbor(bone: BoneInfo, context: dict) -> bool:
    head_related = [candidate for candidate in context["bones"].values() if {"head", "neck"} & candidate.family_tags]
    if not head_related:
        return False
    highest = max(head_related, key=lambda item: item.midpoint[context["height_axis"]])
    return highest.name != bone.name and bone.midpoint[context["height_axis"]] >= highest.midpoint[context["height_axis"]] - 0.15


def _is_highest_in_family_group(bone: BoneInfo, context: dict, families: Set[str]) -> bool:
    matches = [candidate for candidate in context["bones"].values() if candidate.family_tags & families]
    if not matches:
        return False
    highest = max(matches, key=lambda item: item.midpoint[context["height_axis"]])
    return highest.name == bone.name


def _has_descendant_family(bone_name: str, context: dict, families: Set[str]) -> bool:
    queue = list(context["children_map"].get(bone_name, []))
    visited: Set[str] = set()
    while queue:
        current = queue.pop(0)
        if current in visited or current not in context["bones"]:
            continue
        visited.add(current)
        if context["bones"][current].family_tags & families:
            return True
        queue.extend(context["children_map"].get(current, []))
    return False


def _choose_spine_chain(primary_data: dict, support_data: Optional[dict]) -> List[str]:
    chains = list(primary_data.get("chains", {}).get("spine", []))
    if not chains and support_data:
        chains = list(support_data.get("chains", {}).get("spine", []))
    if not chains:
        return []
    return list(max(chains, key=lambda chain: (len(chain), tuple(chain))))


def _choose_limb_chains(
    bones: Dict[str, BoneInfo],
    primary_data: dict,
    support_data: Optional[dict],
    limb_kind: str,
    lateral_axis: int,
    centerline: float,
    side_signs: Dict[str, int],
) -> Dict[str, List[str]]:
    chain_groups = _flatten_limb_chains(primary_data.get("chains", {}).get(limb_kind, {}))
    if not chain_groups and support_data:
        chain_groups = _flatten_limb_chains(support_data.get("chains", {}).get(limb_kind, {}))

    chosen: Dict[str, List[str]] = {"left": [], "right": []}
    best_scores: Dict[str, Tuple[int, float]] = {"left": (-1, -1.0), "right": (-1, -1.0)}

    for chain in chain_groups:
        side = _infer_chain_side(chain["bones"], chain["declared_side"], bones, lateral_axis, centerline, side_signs)
        if side not in {"left", "right"}:
            continue
        named_count = sum(1 for bone_name in chain["bones"] if bones.get(bone_name) and bones[bone_name].side == side)
        chain_score = (named_count, float(len(chain["bones"])))
        if chain_score > best_scores[side]:
            best_scores[side] = chain_score
            chosen[side] = list(chain["bones"])

    return chosen


def _flatten_limb_chains(limb_data: dict) -> List[dict]:
    chains: List[dict] = []
    if not isinstance(limb_data, dict):
        return chains

    for side_name in ("left", "right", "unsided"):
        for chain in limb_data.get(side_name, []):
            chains.append({"declared_side": None if side_name == "unsided" else side_name, "bones": list(chain)})
    return chains


def _infer_chain_side(
    chain: Sequence[str],
    declared_side: Optional[str],
    bones: Dict[str, BoneInfo],
    lateral_axis: int,
    centerline: float,
    side_signs: Dict[str, int],
) -> Optional[str]:
    if declared_side in {"left", "right"}:
        return declared_side

    side_votes = {"left": 0, "right": 0}
    for bone_name in chain:
        bone = bones.get(bone_name)
        if bone and bone.side in side_votes:
            side_votes[bone.side] += 1
    if side_votes["left"] != side_votes["right"]:
        return "left" if side_votes["left"] > side_votes["right"] else "right"

    positions = [bones[bone_name].midpoint[lateral_axis] for bone_name in chain if bone_name in bones]
    if not positions:
        return None

    offset = _mean(positions) - centerline
    if abs(offset) <= 1e-6:
        return None

    left_sign = side_signs.get("left")
    right_sign = side_signs.get("right")
    if left_sign is not None and right_sign is not None:
        if offset * left_sign > 0:
            return "left"
        if offset * right_sign > 0:
            return "right"

    return "left" if offset > 0 else "right"


def _detect_convention(primary_data: dict, support_data: Optional[dict]) -> Convention:
    """Identify a non-Rigify naming convention from the raw bone set.

    Conservative signatures avoid false positives on Rigify/ASAM rigs:
    - 'mixamo' when any raw name carries the 'mixamorig' namespace.
    - 'biped' when any token matches bip<digits>, or a 'com' compact stem
      co-occurs with a 'pelvis' compact stem.
    - 'none' otherwise.
    """
    raw_names: List[str] = []
    for payload in (primary_data, support_data or {}):
        for bone in payload.get("bones", []):
            raw_names.append(bone["name"])

    if any("mixamorig" in name.lower() for name in raw_names):
        return "mixamo"

    stems: Set[str] = set()
    has_bip_token = False
    for name in raw_names:
        _, compact, tokens, _ = _normalize_bone_name(name)
        if any(re.fullmatch(r"bip\d+", token) for token in tokens):
            has_bip_token = True
        stems.add(_compact_stem(compact))

    # The com+pelvis co-occurrence is a heuristic: a non-Biped rig that happens
    # to carry both a center-of-mass "COM" bone and a "Pelvis" bone would also
    # be classified as biped. No current fixture hits this, and the bip<n> token
    # is the primary signal; this is the conservative fallback for Biped rigs
    # exported without the Bip01 prefix.
    if has_bip_token or ("com" in stems and "pelvis" in stems):
        return "biped"
    return "none"


def _normalize_bone_name(name: str, convention: Convention = "none") -> Tuple[str, str, List[str], Optional[str]]:
    transformed = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    transformed = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", transformed)
    transformed = transformed.split(":")[-1].lower()
    raw_tokens = [token for token in re.split(r"[^a-z0-9]+", transformed) if token]

    cleaned_tokens: List[str] = []
    role_tokens = set(ROLE_PREFIXES) | CONTROL_TOKENS
    side: Optional[str] = None

    for token in raw_tokens:
        if convention == "biped" and re.fullmatch(r"bip\d+", token):
            continue
        if token in SIDE_WORDS:
            side = SIDE_WORDS[token]
            continue
        if token in role_tokens and not cleaned_tokens:
            continue
        cleaned_tokens.append(token)

    return "_".join(cleaned_tokens), "".join(cleaned_tokens), cleaned_tokens, side


def _detect_family_tags(
    normalized_name: str,
    compact_name: str,
    tokens: Sequence[str],
    convention: Convention = "none",
) -> Set[str]:
    family_tags: Set[str] = set()
    token_set = set(tokens)

    if "root" in compact_name or compact_name == "master":
        family_tags.add("root")
    if compact_name in {"hip", "hips", "pelvis"} or "pelvis" in compact_name or "hips" in compact_name:
        family_tags.add("hip")
    if compact_name.startswith("spine") or "spine" in token_set:
        family_tags.add("spine")
    if "torso" in token_set or "abdomen" in token_set:
        family_tags.update({"spine", "lower_spine"})
    if "chest" in token_set:
        family_tags.update({"spine", "upper_spine"})
    if "neck" in token_set:
        family_tags.add("neck")
    if "head" in token_set:
        family_tags.add("head")
    if {"shoulder", "clavicle", "collar"} & token_set:
        family_tags.add("shoulder")
    if "upperarm" in compact_name or ("upper" in token_set and "arm" in token_set) or "bicep" in compact_name:
        family_tags.add("upper_arm")
    if "forearm" in compact_name or ("lower" in token_set and "arm" in token_set):
        family_tags.add("lower_arm")
    if compact_name == "hand" or "wrist" in token_set:
        family_tags.add("hand")
    if "thigh" in compact_name or ("upper" in token_set and "leg" in token_set):
        family_tags.add("upper_leg")
    if "shin" in compact_name or "calf" in compact_name or ("lower" in token_set and "leg" in token_set):
        family_tags.add("lower_leg")
    if compact_name == "foot" or "ankle" in token_set:
        family_tags.add("foot")
    if "toe" in compact_name:
        family_tags.add("toe")
    if "eye" in compact_name:
        family_tags.add("eye")
    if "thumb" in compact_name:
        family_tags.add("thumb")
    if "finger" in compact_name or compact_name in {"index", "middle", "ring", "pinky"}:
        family_tags.add("finger")
    if "spine" in family_tags:
        family_tags.update({"lower_spine", "upper_spine"})

    # Alias tags are ADDITIVE here (union with whatever generic rules found), so a
    # bone may carry both generic and alias-derived tags. This is intentional and
    # differs from _family_name_score, where an alias hit is EXCLUSIVE (it returns
    # a single authoritative score and suppresses generic leanings).
    if convention != "none":
        alias_map = BONE_CONVENTIONS[convention].alias_families
        for scoring_family in alias_map.get(_compact_stem(compact_name), frozenset()):
            family_tags.update(SCORING_FAMILY_TO_TAGS[scoring_family])

    return family_tags


def _classify_bone_role(name: str, tokens: Sequence[str]) -> Tuple[str, float]:
    lowered = name.lower()
    prefix_match = re.match(r"^([a-z]+)[-_:.]", lowered)
    prefix = prefix_match.group(1) if prefix_match else None

    if prefix in ROLE_PREFIXES:
        role_name = ROLE_PREFIXES[prefix]
        if role_name == "deform":
            return "deform", 1.08
        if role_name == "auxiliary":
            return "auxiliary", 0.96
        return "mechanism", 0.45

    if any(token in CONTROL_TOKENS for token in tokens):
        return "control", 0.45

    return "unknown", 1.0


def _infer_axes(bones: Iterable[BoneInfo]) -> Tuple[int, int]:
    points = [bone.midpoint for bone in bones]
    if not points:
        return 2, 0

    spans = []
    for axis in range(3):
        axis_values = [point[axis] for point in points]
        spans.append(max(axis_values) - min(axis_values))

    height_axis = max(range(3), key=lambda axis: spans[axis])
    remaining_axes = [axis for axis in range(3) if axis != height_axis]
    lateral_axis = max(remaining_axes, key=lambda axis: spans[axis])
    return height_axis, lateral_axis


def _calibrate_side_signs(bones: Iterable[BoneInfo], lateral_axis: int) -> Dict[str, int]:
    left_positions = [bone.midpoint[lateral_axis] for bone in bones if bone.side == "left"]
    right_positions = [bone.midpoint[lateral_axis] for bone in bones if bone.side == "right"]
    if not left_positions or not right_positions:
        return {"left": 1, "right": -1}

    left_mean = _mean(left_positions)
    right_mean = _mean(right_positions)
    if math.isclose(left_mean, right_mean, abs_tol=1e-6):
        return {"left": 1, "right": -1}

    left_sign = 1 if left_mean > right_mean else -1
    return {"left": left_sign, "right": -left_sign}


def _axis_cross_sign(axis_a: int, axis_b: int, axis_c: int) -> int:
    return 1 if ((axis_a + 1) % 3 == axis_b and (axis_b + 1) % 3 == axis_c) else -1


def _height_ratio(bone: BoneInfo, context: dict) -> float:
    axis = context["height_axis"]
    sign = context["height_sign"]
    if sign >= 0:
        value = bone.midpoint[axis] - context["height_bounds_min"]
    else:
        value = context["height_bounds_max"] - bone.midpoint[axis]
    return _clamp(value / context["height_span"])


def _endpoint_ratio(bone: BoneInfo, context: dict, which: str) -> float:
    axis = context["height_axis"]
    sign = context["height_sign"]

    if sign >= 0:
        value = bone.min_point[axis] if which == "low" else bone.max_point[axis]
        normalized = value - context["height_bounds_min"]
    else:
        value = bone.max_point[axis] if which == "low" else bone.min_point[axis]
        normalized = context["height_bounds_max"] - value

    return _clamp(normalized / context["height_span"])


def _centerline_score(bone: BoneInfo, context: dict) -> float:
    offset = abs(bone.midpoint[context["lateral_axis"]] - context["centerline"])
    tolerance = max(context["lateral_span"] * 0.20, 1e-6)
    return _clamp(1.0 - (offset / tolerance))


def _side_score(expected_side: Optional[str], bone: BoneInfo, context: dict) -> float:
    if expected_side is None:
        return 0.5

    if bone.side == expected_side:
        name_component = 1.0
    elif bone.side is None:
        name_component = 0.45
    else:
        name_component = 0.05

    offset = bone.midpoint[context["lateral_axis"]] - context["centerline"]
    expected_sign = context["side_signs"].get(expected_side)
    if abs(offset) <= context["lateral_span"] * 0.05:
        geometry_component = 0.20
    elif expected_sign is None:
        geometry_component = 0.60
    else:
        geometry_component = 1.0 if offset * expected_sign > 0 else 0.05

    return _clamp((0.6 * name_component) + (0.4 * geometry_component))


def _side_geometry_conflict(expected_side: str, bone: BoneInfo, context: dict) -> bool:
    if bone.side not in {None, expected_side}:
        return True
    offset = bone.midpoint[context["lateral_axis"]] - context["centerline"]
    if abs(offset) <= context["lateral_span"] * 0.05:
        return False
    expected_sign = context["side_signs"].get(expected_side)
    return expected_sign is not None and offset * expected_sign < 0


def _band_score(value: float, lower: float, upper: float) -> float:
    if lower <= value <= upper:
        return 1.0
    span = max(upper - lower, 0.15)
    if value < lower:
        return _clamp(1.0 - ((lower - value) / span))
    return _clamp(1.0 - ((value - upper) / span))


def _to_float_triplet(values: Optional[Sequence[float]]) -> Tuple[float, float, float]:
    values = values or (0.0, 0.0, 0.0)
    return (float(values[0]), float(values[1]), float(values[2]))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _load_json_cached(path: Optional[Path], cache: Dict[Path, dict]) -> dict:
    if path is None:
        raise ValueError("Path is required")
    if path not in cache:
        with path.open("r", encoding="utf-8") as handle:
            cache[path] = json.load(handle)
    return cache[path]

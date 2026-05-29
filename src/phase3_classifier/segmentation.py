"""
Per-character ("crowd") armature decomposition for the Phase 3 classifier.

Pure Python: operates on already-loaded inspector JSON dicts. No bpy.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from phase3_classifier.classifier import ArmatureInput, classify_armature, _build_actions

# Leading run of letters followed by digits, e.g. "Female000", "Male060".
_PREFIX_RE = re.compile(r"^([A-Za-z]+\d+)")
# Trailing unique-id suffix, e.g. "_093" in "Pelvis_093".
_ID_SUFFIX_RE = re.compile(r"_\d+$")


def _strip_namespace(bone_name: str) -> str:
    return bone_name.split(":")[-1]


def derive_character_prefix(bone_name: str) -> Optional[str]:
    """Return the per-character prefix (e.g. 'Female000') or None if absent."""
    match = _PREFIX_RE.match(_strip_namespace(bone_name))
    return match.group(1) if match else None


def strip_character_prefix(bone_name: str, prefix: str) -> str:
    """Remove the namespace, the character prefix, and a trailing _<id> suffix.

    'Female000Pelvis_093' + 'Female000' -> 'Pelvis'.
    """
    stripped = _strip_namespace(bone_name)
    if stripped.startswith(prefix):
        stripped = stripped[len(prefix):]
    return _ID_SUFFIX_RE.sub("", stripped)


MIN_CHARACTER_BONES = 5


@dataclass(frozen=True)
class CharacterGroup:
    character_id: str
    prefix: str
    bone_names: List[str]   # original (un-stripped) names
    connected: bool


def group_bones_by_prefix(bones: List[dict]) -> Dict[Optional[str], List[str]]:
    """Map each derived prefix -> original bone names. Prefix-less bones key None."""
    groups: Dict[Optional[str], List[str]] = {}
    for bone in bones:
        name = bone["name"]
        prefix = derive_character_prefix(name)
        groups.setdefault(prefix, []).append(name)
    return groups


def _is_connected_subtree(bone_names: List[str], parent_by_name: Dict[str, Optional[str]]) -> bool:
    """True iff the group has exactly one in-group root and all members hang off it."""
    members = set(bone_names)
    roots = [name for name in members if parent_by_name.get(name) not in members]
    if len(roots) != 1:
        return False
    children: Dict[str, List[str]] = {name: [] for name in members}
    for name in members:
        parent = parent_by_name.get(name)
        if parent in members:
            children[parent].append(name)
    reached = set()
    stack = [roots[0]]
    while stack:
        current = stack.pop()
        if current in reached:
            continue
        reached.add(current)
        stack.extend(children[current])
    return reached == members


def detect_characters(bones: List[dict]) -> Tuple[List[CharacterGroup], List[str]]:
    """Return accepted character groups and the shared (prefix-less) bone names."""
    parent_by_name = {bone["name"]: bone.get("parent") for bone in bones}
    grouped = group_bones_by_prefix(bones)

    shared = sorted(grouped.get(None, []))
    characters: List[CharacterGroup] = []
    for prefix, names in grouped.items():
        if prefix is None or len(names) < MIN_CHARACTER_BONES:
            continue
        ordered = [bone["name"] for bone in bones if bone["name"] in set(names)]
        characters.append(
            CharacterGroup(
                character_id=prefix,
                prefix=prefix,
                bone_names=ordered,
                connected=_is_connected_subtree(ordered, parent_by_name),
            )
        )
    characters.sort(key=lambda group: group.character_id)
    return characters, shared


def _strip_chains(chains: dict, members: set, prefix: str) -> dict:
    """Filter chain bone lists to the group's members and strip the prefix."""

    def fix_list(chain_list):
        result = []
        for chain in chain_list:
            kept = [strip_character_prefix(name, prefix) for name in chain if name in members]
            if kept:
                result.append(kept)
        return result

    out = {}
    for key, value in (chains or {}).items():
        if isinstance(value, dict):
            out[key] = {side: fix_list(chain_list) for side, chain_list in value.items()}
        else:
            out[key] = fix_list(value)
    return out


def build_character_view(primary_data: dict, group: CharacterGroup, mesh_binding: Optional[dict]):
    """Synthesize an ArmatureInput scoped to one character, prefix stripped.

    placement_metadata is intentionally omitted so _build_context derives it
    per-character from the character's own bones.
    """
    members = set(group.bone_names)
    new_bones = []
    for bone in primary_data["bones"]:
        if bone["name"] not in members:
            continue
        new_bone = copy.deepcopy(bone)
        new_bone["name"] = strip_character_prefix(bone["name"], group.prefix)
        parent = bone.get("parent")
        new_bone["parent"] = strip_character_prefix(parent, group.prefix) if parent in members else None
        new_bones.append(new_bone)

    hierarchy = {bone["name"]: [] for bone in new_bones}
    for bone in new_bones:
        parent = bone["parent"]
        if parent in hierarchy:
            hierarchy[parent].append(bone["name"])

    view_primary = {
        "armature_name": group.character_id,
        "source_file": primary_data.get("source_file"),
        "filter": primary_data.get("filter"),
        "bone_count": len(new_bones),
        "hierarchy": hierarchy,
        "bones": new_bones,
        "chains": _strip_chains(primary_data.get("chains", {}), members, group.prefix),
    }
    if mesh_binding is not None:
        view_primary["mesh_binding"] = mesh_binding

    return ArmatureInput(
        armature_name=group.character_id,
        all_path=None,
        filtered_path=None,
        primary_path=Path(f"{group.character_id}_all.json"),
        support_path=None,
        primary_data=view_primary,
        support_data=None,
    )


def _mesh_owner_prefix(mesh: dict, known_prefixes: set) -> Optional[str]:
    """Pick the character prefix that owns the majority of weighted vertices."""
    weights: Dict[str, int] = {}
    per_group = mesh.get("vertex_group_stats", {}).get("per_group", [])
    for entry in per_group:
        prefix = derive_character_prefix(entry["name"])
        if prefix in known_prefixes:
            weights[prefix] = weights.get(prefix, 0) + int(entry.get("weighted_vertex_count", 0))
    if not weights:
        return None
    return max(sorted(weights), key=lambda prefix: weights[prefix])


def _strip_mesh_binding(mesh: dict, prefix: str) -> dict:
    """Strip the character prefix from a mesh's vertex-group names."""
    out = copy.deepcopy(mesh)
    out["vertex_groups"] = [strip_character_prefix(name, prefix) for name in mesh.get("vertex_groups", [])]
    stats = out.get("vertex_group_stats", {})
    for entry in stats.get("per_group", []):
        entry["name"] = strip_character_prefix(entry["name"], prefix)
    return out


def assign_meshes_to_characters(
    mesh_binding: Optional[dict],
    groups: List[CharacterGroup],
) -> Tuple[Dict[str, dict], List[str]]:
    """Return {character_id: mesh_binding} and a list of unassignable mesh names."""
    known = {group.prefix for group in groups}
    per_char: Dict[str, dict] = {
        group.character_id: {"armature_object_name": group.character_id, "meshes": []}
        for group in groups
    }
    unassigned: List[str] = []
    for mesh in (mesh_binding or {}).get("meshes", []):
        owner = _mesh_owner_prefix(mesh, known)
        if owner is None:
            unassigned.append(mesh["mesh_name"])
            continue
        per_char[owner]["meshes"].append(_strip_mesh_binding(mesh, owner))
    return per_char, sorted(unassigned)


@dataclass
class SegmentationResult:
    decomposition: dict
    report_characters: List[dict]
    plan_characters: List[dict]


def _report_character(character_id: str, report: dict) -> dict:
    return {
        "character_id": character_id,
        "detected_convention": report["detected_convention"],
        "semantic_mapping": report["asam_targets"],
        "missing_targets": report["missing_core_targets"],
        "ambiguous_targets": report["ambiguous_targets"],
        "unclassified_bones": report["unclassified_bones"],
        "review_flags": report["review_flags"],
        "root_resolutions": [report["root_resolution"]],
        "placement_metadata": report["placement_metadata"],
        "mesh_binding": report["mesh_binding"],
        "summary": report["summary"],
    }


def _plan_character(character_id: str, report: dict) -> dict:
    return {
        "character_id": character_id,
        "actions": _build_actions(report["asam_targets"]),
        "root_resolutions": [report["root_resolution"]],
        "placement_metadata": report["placement_metadata"],
        "mesh_binding": report["mesh_binding"],
        "proposed_asam_hierarchy": report["proposed_asam_hierarchy"],
        "extras_preserved": report["extras_preserved"],
    }


def segment_recommended(asset_name: str, recommended_input) -> Optional[SegmentationResult]:
    """Return per-character decomposition for a crowd armature, or None if single."""
    bones = recommended_input.primary_data.get("bones", [])
    groups, shared = detect_characters(bones)
    if len(groups) < 2:
        return None

    per_char_binding, unassigned = assign_meshes_to_characters(
        recommended_input.primary_data.get("mesh_binding"), groups
    )

    report_characters: List[dict] = []
    plan_characters: List[dict] = []
    for group in groups:
        view = build_character_view(
            recommended_input.primary_data, group, per_char_binding.get(group.character_id)
        )
        report = classify_armature(asset_name, view)
        if not group.connected:
            report["review_flags"] = list(report["review_flags"]) + ["character_disconnected"]
        report_characters.append(_report_character(group.character_id, report))
        plan_characters.append(_plan_character(group.character_id, report))

    decomposition = {
        "detected": True,
        "source_armature": recommended_input.armature_name,
        "character_count": len(groups),
        "character_ids": [group.character_id for group in groups],
        "shared_bones": shared,
        "unassigned_meshes": unassigned,
    }
    return SegmentationResult(decomposition, report_characters, plan_characters)

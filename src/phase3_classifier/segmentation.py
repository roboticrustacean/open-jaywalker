"""
Per-character ("crowd") armature decomposition for the Phase 3 classifier.

Pure Python: operates on already-loaded inspector JSON dicts. No bpy.
"""

from __future__ import annotations

import copy
import math
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
    result = _ID_SUFFIX_RE.sub("", stripped)
    # A bone named exactly the prefix (e.g. a sub-root 'Female000' / 'Female000_5')
    # would strip to empty; keep the namespace-stripped name so it stays a usable,
    # unique bone identifier rather than an empty string.
    return result or _strip_namespace(bone_name)


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
def _bone_centroid(bone_names: List[str], bones: List[dict]) -> Tuple[float, float, float]:
    """Compute the spatial centroid of a set of bones using their head and tail coordinates."""
    points = []
    for bone in bones:
        if bone["name"] in bone_names:
            if "head" in bone:
                points.append(bone["head"])
            if "tail" in bone:
                points.append(bone["tail"])
    if not points:
        return (0.0, 0.0, 0.0)
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
        sum(p[2] for p in points) / len(points)
    )

def _distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def detect_characters(bones: List[dict]) -> Tuple[List[CharacterGroup], List[str]]:
    """Return accepted character groups and the shared bone names, using graph connectivity."""
    parent_by_name = {bone["name"]: bone.get("parent") for bone in bones}
    grouped = group_bones_by_prefix(bones)

    char_data = []
    strays: List[List[str]] = []
    
    # 1. Process prefix-identified groups first
    for prefix, names in grouped.items():
        if prefix is None:
            continue
        if len(names) >= MIN_CHARACTER_BONES:
            name_set = set(names)
            ordered = [bone["name"] for bone in bones if bone["name"] in name_set]
            char_data.append({
                "id": prefix,
                "prefix": prefix,
                "bones": ordered,
                "connected": _is_connected_subtree(ordered, parent_by_name)
            })
        else:
            strays.append(names)

    # 2. Process prefix-less bones using graph connectivity
    prefixless_bones = grouped.get(None, [])
    members = set(prefixless_bones)
    neighbors = {name: [] for name in members}
    for name in members:
        parent = parent_by_name.get(name)
        if parent in members:
            neighbors[name].append(parent)
            neighbors[parent].append(name)
            
    visited = set()
    for name in sorted(members):
        if name not in visited:
            comp = []
            stack = [name]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    comp.append(curr)
                    stack.extend(neighbors[curr])
            
            if len(comp) >= MIN_CHARACTER_BONES:
                comp_set = set(comp)
                roots = [n for n in comp if parent_by_name.get(n) not in comp_set]
                roots.sort()
                root_bone = roots[0] if roots else sorted(comp)[0]
                ordered = [bone["name"] for bone in bones if bone["name"] in comp_set]
                char_data.append({
                    "id": root_bone,
                    "prefix": "",
                    "bones": ordered,
                    "connected": True
                })
            else:
                strays.append(comp)

    # 2b. Spatially cluster main components
    # Handles constraint-based deform rigs where limbs are disconnected in the hierarchy
    merged_char_data = []
    if char_data:
        spatial_adj = {i: [] for i in range(len(char_data))}
        for i in range(len(char_data)):
            c1 = _bone_centroid(char_data[i]["bones"], bones)
            p1 = char_data[i]["prefix"]
            for j in range(i + 1, len(char_data)):
                c2 = _bone_centroid(char_data[j]["bones"], bones)
                p2 = char_data[j]["prefix"]
                # Only merge if within 2.5m AND prefixes do not conflict
                if _distance(c1, c2) < 2.5 and (not p1 or not p2 or p1 == p2):
                    spatial_adj[i].append(j)
                    spatial_adj[j].append(i)
                    
        spatial_visited = set()
        for i in range(len(char_data)):
            if i not in spatial_visited:
                cluster_indices = []
                stack = [i]
                while stack:
                    curr = stack.pop()
                    if curr not in spatial_visited:
                        spatial_visited.add(curr)
                        cluster_indices.append(curr)
                        stack.extend(spatial_adj[curr])
                
                base_char = char_data[cluster_indices[0]]
                for j in cluster_indices[1:]:
                    base_char["bones"].extend(char_data[j]["bones"])
                    base_char["connected"] = base_char["connected"] and char_data[j]["connected"]
                    if char_data[j]["prefix"] and not base_char["prefix"]:
                        base_char["prefix"] = char_data[j]["prefix"]
                        base_char["id"] = char_data[j]["id"]
                merged_char_data.append(base_char)
        char_data = merged_char_data

    # 3. Spatially merge strays into the nearest character, IF they don't bridge multiple characters
    shared: List[str] = []
    
    bone_to_char_idx = {}
    for idx, c in enumerate(char_data):
        for b in c["bones"]:
            bone_to_char_idx[b] = idx

    for comp in strays:
        if not char_data:
            shared.extend(comp)
            continue
            
        connected_char_indices = set()
        for stray_bone in comp:
            parent = parent_by_name.get(stray_bone)
            if parent in bone_to_char_idx:
                connected_char_indices.add(bone_to_char_idx[parent])
            for b_name, b_parent in parent_by_name.items():
                if b_parent == stray_bone and b_name in bone_to_char_idx:
                    connected_char_indices.add(bone_to_char_idx[b_name])
                    
        if len(connected_char_indices) > 1:
            shared.extend(comp)
            continue

        centroid = _bone_centroid(comp, bones)
        best_idx = -1
        best_dist = float('inf')
        
        if len(connected_char_indices) == 1:
            best_idx = list(connected_char_indices)[0]
            best_dist = 0.0
        else:
            for idx, char in enumerate(char_data):
                char_centroid = _bone_centroid(char["bones"], bones)
                dist = _distance(centroid, char_centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
                
        if best_dist <= 2.0 and best_idx != -1:
            ordered_comp = [b["name"] for b in bones if b["name"] in comp]
            char_data[best_idx]["bones"].extend(ordered_comp)
        else:
            shared.extend(comp)

    characters = [
        CharacterGroup(
            character_id=c["id"],
            prefix=c["prefix"],
            bone_names=c["bones"],
            connected=c["connected"]
        ) for c in char_data
    ]
    characters.sort(key=lambda group: group.character_id)
    shared.sort()
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


def _mesh_owner_character(mesh: dict, groups: List[CharacterGroup]) -> Optional[str]:
    """Pick the character_id that owns the majority of weighted vertices."""
    weights: Dict[str, int] = {}
    
    bone_to_char = {}
    for group in groups:
        for bone_name in group.bone_names:
            bone_to_char[bone_name] = group.character_id
            
    per_group = mesh.get("vertex_group_stats", {}).get("per_group", [])
    for entry in per_group:
        bone_name = entry["name"]
        char_id = bone_to_char.get(bone_name)
        if char_id:
            weights[char_id] = weights.get(char_id, 0) + int(entry.get("weighted_vertex_count", 0))
            
    if not weights:
        return None
    return max(sorted(weights), key=lambda cid: weights[cid])


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
    source_armature_name: Optional[str],
) -> Tuple[Dict[str, dict], List[str]]:
    """Return {character_id: mesh_binding} and a list of unassignable mesh names."""
    id_to_prefix = {group.character_id: group.prefix for group in groups}
    
    per_char: Dict[str, dict] = {
        group.character_id: {"armature_object_name": source_armature_name, "meshes": []}
        for group in groups
    }
    unassigned: List[str] = []
    for mesh in (mesh_binding or {}).get("meshes", []):
        owner_id = _mesh_owner_character(mesh, groups)
        if owner_id is None:
            unassigned.append(mesh["mesh_name"])
            continue
        
        prefix = id_to_prefix[owner_id]
        per_char[owner_id]["meshes"].append(_strip_mesh_binding(mesh, prefix))
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


# Planar spread (metres) below which characters are treated as co-located and gridded.
COLOCATED_SPREAD_THRESHOLD = 0.25


def body_anchor_from_meshes(meshes, up_axis_index, up_sign):
    """Ground-center of the union of owned meshes' world bboxes, or None if absent.

    Planar axes use the bbox center; the up axis uses the ground (min when up_sign>=0,
    else max) so the anchor sits on the floor under the body.
    """
    boxes = [m["bbox"] for m in (meshes or []) if isinstance(m.get("bbox"), dict)]
    if not boxes:
        return None
    mins = [min(b["min"][i] for b in boxes) for i in range(3)]
    maxs = [max(b["max"][i] for b in boxes) for i in range(3)]
    center = [(mins[i] + maxs[i]) / 2.0 for i in range(3)]
    center[up_axis_index] = mins[up_axis_index] if up_sign >= 0 else maxs[up_axis_index]
    return [float(v) for v in center]


def _assign_character_placements(reports):
    """Set root_resolution['grp_root_world_location'] and ['placement_mode'] per report.

    'source' when bodies are distributed; 'grid' when co-located. Anchors come from
    each character's owned mesh bboxes (body), falling back to the bone-derived
    bbox_ground_center when a character has no mesh.

    Mutates the passed report dicts in place.
    """
    if not reports:
        return
    anchors = []
    for report in reports:
        placement = report["placement_metadata"]
        up = placement["up_axis"]
        anchor = body_anchor_from_meshes(
            report["mesh_binding"].get("meshes"),
            up_axis_index=int(up["index"]),
            up_sign=int(up["sign"]),
        )
        if anchor is None:
            anchor = [float(v) for v in placement["bbox_ground_center"]]
            report["review_flags"] = list(report["review_flags"]) + ["character_without_mesh_anchor"]
        anchors.append((report, anchor, placement))

    forward_idx, side_idx = _planar_axes(anchors[0][2])
    spread = _planar_spread(anchors, forward_idx, side_idx)

    if spread <= COLOCATED_SPREAD_THRESHOLD:
        _apply_grid(anchors, forward_idx, side_idx)
    else:
        for report, anchor, _placement in anchors:
            report["root_resolution"]["grp_root_world_location"] = anchor
            report["root_resolution"]["placement_mode"] = "source"


def _planar_axes(placement):
    up = int(placement["up_axis"]["index"])
    others = [i for i in range(3) if i != up]
    return others[0], others[1]


def _planar_spread(anchors, a, b):
    av = [anc[a] for _r, anc, _p in anchors]
    bv = [anc[b] for _r, anc, _p in anchors]
    return max(max(av) - min(av), max(bv) - min(bv))


def _character_footprint(placement, a, b):
    bbox_min = placement.get("bbox_min")
    bbox_max = placement.get("bbox_max")
    if not bbox_min or not bbox_max:
        return 1.0
    return max(abs(bbox_max[a] - bbox_min[a]), abs(bbox_max[b] - bbox_min[b]), 1e-3)


def _apply_grid(anchors, a, b):
    count = len(anchors)
    columns = max(1, int(math.ceil(math.sqrt(count))))
    spacing = max(_character_footprint(p, a, b) for _r, _anc, p in anchors) * 1.5
    for index, (report, anchor, _placement) in enumerate(anchors):
        row, col = divmod(index, columns)
        cell = list(anchor)
        cell[a] = col * spacing
        cell[b] = row * spacing
        report["root_resolution"]["grp_root_world_location"] = [float(v) for v in cell]
        report["root_resolution"]["placement_mode"] = "grid"
        report["root_resolution"]["applied_grid_offset"] = [
            float(cell[i] - anchor[i]) for i in range(3)
        ]


def segment_recommended(asset_name: str, recommended_input) -> Optional[SegmentationResult]:
    """Return per-character decomposition for a crowd armature, or None if single."""
    bones = recommended_input.primary_data.get("bones", [])
    groups, shared = detect_characters(bones)
    # Load-bearing single-character guarantee: a normal single rig (even one whose
    # bones all share a 'Bip01' prefix) yields exactly one group, so fan-out only
    # triggers on >=2 distinct accepted character prefixes. Everything else falls
    # through to the unchanged flat report path.
    if len(groups) < 2:
        return None

    per_char_binding, unassigned = assign_meshes_to_characters(
        recommended_input.primary_data.get("mesh_binding"),
        groups,
        recommended_input.armature_name,
    )

    raw_reports = []
    for group in groups:
        view = build_character_view(
            recommended_input.primary_data, group, per_char_binding.get(group.character_id)
        )
        report = classify_armature(asset_name, view)
        if not group.connected:
            report["review_flags"] = list(report["review_flags"]) + ["character_disconnected"]
        raw_reports.append((group, report))

    # Mutates each report's root_resolution in place (grp_root_world_location +
    # placement_mode); those dicts are the same objects _report_character/_plan_character
    # snapshot below, so the placement keys flow into both outputs.
    _assign_character_placements([report for _g, report in raw_reports])

    report_characters: List[dict] = []
    plan_characters: List[dict] = []
    for group, report in raw_reports:
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

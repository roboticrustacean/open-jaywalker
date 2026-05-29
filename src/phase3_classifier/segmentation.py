"""
Per-character ("crowd") armature decomposition for the Phase 3 classifier.

Pure Python: operates on already-loaded inspector JSON dicts. No bpy.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

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

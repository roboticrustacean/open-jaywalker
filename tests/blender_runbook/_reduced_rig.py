"""
Helper that programmatically reduces a source armature below the ASAM minimum
bone set so the pipeline's graceful-degradation behavior can be exercised
(Issue #22).

This module DESTRUCTIVELY edits the in-memory armature data and provides no
undo path - the caller is expected to run this once in a fresh Blender session
and CLOSE WITHOUT SAVING when done.

Deletion is by SUBSTRING PATTERN rather than exact name. Rigify rigs surface
the same conceptual bone (a foot, a hand) under multiple names per layer:
`DEF-foot.L`, `ORG-foot.L`, `MCH-foot_ik.L`, `foot_fk.L`, plain `foot.L`,
and so on. The classifier scores any of these by name, so removing only the
DEF-* layer leaves alternate sources behind and the missing-source path is
never exercised. The pattern list catches every variant; the caller is
expected to apply it to every armature in the scene.
"""

from __future__ import annotations

from typing import Iterable, List


# Substring patterns. A bone is deleted iff its name contains ANY of these.
#
# Coverage notes for the LowPolyCharacter4 fixture (Rigify):
#   - "spine.0" catches spine.001-006 across all prefixes (DEF-, MCH-, ORG-,
#     spine_fk, tweak_spine) and leaves the base spine bones (`spine`,
#     `DEF-spine`, `MCH-spine`, `ORG-spine`, `spine_fk`, `tweak_spine`) intact
#     so Lower_Spine still has a source.
#   - "foot" / "toe" / "hand" each catch their full constellation of variants
#     (foot.L, DEF-foot.L, ORG-foot.L, MCH-foot_ik.L, foot_fk.L,
#     foot_ik_target.L, foot_spin_ik.L, etc.).
LOWPOLY_REDUCED_DELETION_PATTERNS: List[str] = [
    "spine.0",
    "foot",
    "toe",
    "hand",
]


def reduce_armature_in_place(
    bpy_module,
    armature_obj,
    bone_name_patterns: Iterable[str],
) -> List[str]:
    """
    Enter EDIT mode on `armature_obj` and remove every bone whose name contains
    any of the listed substrings.

    Children of removed bones become detached (parent = None) before deletion
    so Blender's edit pass succeeds even when a removed bone parents bones we
    intend to keep. Returns the names that were actually deleted, for
    reporting. Empty list if the armature has no matching bones.
    """
    patterns = list(bone_name_patterns)
    deleted: List[str] = []

    try:
        bpy_module.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        pass

    bpy_module.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    bpy_module.ops.object.mode_set(mode="EDIT")

    edit_bones = armature_obj.data.edit_bones
    targets_set = {
        bone.name
        for bone in edit_bones
        if any(pattern in bone.name for pattern in patterns)
    }

    # Detach children of any to-be-removed bone first so reparenting doesn't
    # cascade unexpected deletions.
    for bone in list(edit_bones):
        if bone.parent and bone.parent.name in targets_set:
            bone.parent = None

    for name in sorted(targets_set):
        bone = edit_bones.get(name)
        if bone is None:
            continue
        edit_bones.remove(bone)
        deleted.append(name)

    bpy_module.ops.object.mode_set(mode="OBJECT")
    return deleted

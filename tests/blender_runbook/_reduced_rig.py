"""
Helper that programmatically reduces a source armature below the ASAM minimum
bone set so the pipeline's graceful-degradation behavior can be exercised
(Issue #22).

This module DESTRUCTIVELY edits the in-memory armature data and provides no
undo path - the caller is expected to run this once in a fresh Blender session
and CLOSE WITHOUT SAVING when done.
"""

from __future__ import annotations

from typing import Iterable, List


# Bones to delete on LowPolyCharacter4's `rig` armature to simulate a
# "fewer than ASAM minimum" source:
#   - collapse the seven-segment spine down to a single DEF-spine
#   - remove the hand bones (no Hand_Left/Right source)
#   - remove the foot+toe bones (no Foot_Left/Right, no Full_Toes_Left/Right)
LOWPOLY_REDUCED_DELETIONS: List[str] = [
    "DEF-spine.001",
    "DEF-spine.002",
    "DEF-spine.003",
    "DEF-spine.004",
    "DEF-spine.005",
    "DEF-spine.006",
    "DEF-hand.L",
    "DEF-hand.R",
    "DEF-foot.L",
    "DEF-foot.R",
    "DEF-toe.L",
    "DEF-toe.R",
]


def reduce_armature_in_place(bpy_module, armature_obj, bone_names_to_remove: Iterable[str]) -> List[str]:
    """
    Enter EDIT mode on `armature_obj` and remove the listed bones.

    Children of removed bones become detached (parent = None) so the edit pass
    succeeds even if the user lists a parent before its descendants. Returns
    the names that were actually deleted, for reporting.
    """
    targets = list(bone_names_to_remove)
    deleted: List[str] = []

    try:
        bpy_module.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        pass

    bpy_module.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    bpy_module.ops.object.mode_set(mode="EDIT")

    edit_bones = armature_obj.data.edit_bones
    # Detach children of any bone we are about to remove, in case the bone
    # being removed is a parent of bones we want to keep.
    targets_set = set(targets)
    for bone in list(edit_bones):
        if bone.parent and bone.parent.name in targets_set:
            bone.parent = None

    for name in targets:
        bone = edit_bones.get(name)
        if bone is None:
            continue
        edit_bones.remove(bone)
        deleted.append(name)

    bpy_module.ops.object.mode_set(mode="OBJECT")
    return deleted

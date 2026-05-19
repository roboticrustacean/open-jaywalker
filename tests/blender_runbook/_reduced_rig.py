"""
Helper that programmatically reduces a source armature below the ASAM minimum
bone set so the pipeline's graceful-degradation behavior can be exercised
(Issue #22).

This module DESTRUCTIVELY edits the in-memory armature data and provides no
undo path - the caller is expected to run this once in a fresh Blender session
and CLOSE WITHOUT SAVING when done.

Deletion is by SUBSTRING PATTERN rather than exact name. Rigify rigs surface
the same conceptual bone under five or more parallel layers (DEF-, ORG-, MCH-,
*_fk, *_ik, *_tweak, plain), so an exact-name list never catches enough to
genuinely starve the classifier. The pattern list catches every variant; a
small keep-name list rescues the single deformer we want surviving on each
ASAM axis (e.g. `DEF-spine` so Lower_Spine still has a source).

Constraints on surviving bones that reference a deleted bone are stripped in
the same pass to silence the Blender depsgraph's "Failed to add relation"
warnings - the constraints are inert anyway after their subtarget is gone.
The sweep checks every reference path a constraint can carry: `.subtarget`
(most constraints), `.pole_subtarget` (IK), and `.targets[*].subtarget` (the
Armature constraint that Rigify uses for SWITCH_PARENT).
"""

from __future__ import annotations

from typing import Iterable, List, Optional


# Substring patterns. A bone is deleted iff its name contains ANY of these
# AND its name is not in LOWPOLY_REDUCED_KEEP_NAMES.
#
# Coverage notes for the LowPolyCharacter4 fixture (Rigify):
#   - "spine" catches every spine variant across all prefixes and layers:
#     spine, DEF-spine, MCH-spine, ORG-spine, spine_fk, tweak_spine, all of
#     their numbered children (spine.001-006, DEF-spine.001-006, etc.), and
#     the metarig's `spine` bone too. The keep-list rescues `DEF-spine` so
#     Lower_Spine still has a valid deformer source.
#   - "foot" / "toe" / "hand" each catch their full constellation of
#     variants on every layer (foot.L, DEF-foot.L, foot_ik_target.L,
#     foot_spin_ik.L, foot_tweak.L, MCH-foot_roll.L, etc.).
LOWPOLY_REDUCED_DELETION_PATTERNS: List[str] = [
    "spine",
    "foot",
    "toe",
    "hand",
]


# Exact bone names that survive the deletion sweep even when they would
# otherwise match a pattern. Used to preserve the single base deformer bone
# per ASAM axis so the classifier maps Lower_Spine (and Hip, via the
# untouched pelvis pair) successfully.
LOWPOLY_REDUCED_KEEP_NAMES: List[str] = [
    "DEF-spine",
]


def reduce_armature_in_place(
    bpy_module,
    armature_obj,
    bone_name_patterns: Iterable[str],
    keep_names: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Enter EDIT mode on `armature_obj` and remove every bone whose name contains
    any of the listed substrings, except those listed in `keep_names`.

    Before deletion, pose-bone constraints whose subtarget refers to a
    to-be-deleted bone are removed so Blender's depsgraph stops emitting
    "Failed to add relation" warnings during evaluation. The constraints are
    inert anyway once the subtarget is gone.

    Children of removed bones are detached (parent = None) so the edit pass
    succeeds even when a removed bone parents bones we intend to keep.

    Returns the names that were actually deleted, for reporting. Empty list if
    the armature has no matching bones.
    """
    patterns = list(bone_name_patterns)
    keep_set = set(keep_names or ())
    deleted: List[str] = []

    try:
        bpy_module.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        pass

    bpy_module.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)

    # First pass (OBJECT/POSE mode): compute the target set from the data
    # bones, then strip pose-bone constraints whose subtarget is in the set.
    data_bones = armature_obj.data.bones
    targets_set = {
        bone.name
        for bone in data_bones
        if bone.name not in keep_set
        and any(pattern in bone.name for pattern in patterns)
    }

    pose_bones = getattr(armature_obj, "pose", None)
    if pose_bones is not None:
        for pose_bone in pose_bones.bones:
            stale = [
                constraint
                for constraint in pose_bone.constraints
                if _constraint_references_deleted_bone(constraint, targets_set)
            ]
            for constraint in stale:
                pose_bone.constraints.remove(constraint)

    # Second pass (EDIT mode): detach children of any to-be-removed bone, then
    # remove the bones themselves.
    bpy_module.ops.object.mode_set(mode="EDIT")
    edit_bones = armature_obj.data.edit_bones

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


def _constraint_references_deleted_bone(constraint, deleted_bone_names) -> bool:
    """
    Return True if any of `constraint`'s bone references points at a deleted bone.

    Covers the three reference paths a Blender bone constraint can carry:
      - `subtarget` — most constraints (Stretch To, Copy *, Damped Track, IK, ...)
      - `pole_subtarget` — IK constraint's pole target
      - `targets[*].subtarget` — Armature constraint (used by Rigify SWITCH_PARENT)

    Missing attributes (e.g. constraints without a pole or without a targets
    collection) are treated as "no reference of that kind" rather than as an
    error.
    """
    for attr in ("subtarget", "pole_subtarget"):
        if getattr(constraint, attr, "") in deleted_bone_names:
            return True
    targets = getattr(constraint, "targets", None)
    if targets is not None:
        for target in targets:
            if getattr(target, "subtarget", "") in deleted_bone_names:
                return True
    return False

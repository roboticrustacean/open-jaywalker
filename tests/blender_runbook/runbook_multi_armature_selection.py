"""
In-Blender multi-armature selection runbook (Issue #32).

Builds a synthetic scene with TWO armatures both modifier-bound to ONE mesh:
  - DeformRig: mixamo-named bones that match the mesh's vertex groups (owns skin).
  - ControlRig: a bushier biped-named rig (incl. toes -> more ASAM targets, higher
    ranking_score) whose names do NOT match the vertex groups.

Then runs inspector + classifier and asserts the classifier recommends DeformRig
(because it carries the skin weights), with selection_tiebreaker
== 'vertex_group_coverage'. Under the pre-#32 ranking ControlRig would have won.

This is NOT a pytest test: it requires an interactive Blender session and is not
discovered by `python -m unittest`. Run via VS Code `Blender: Run Script` or a
Blender MCP `execute code` call. It builds its own objects, so any open scene works.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

import bpy  # noqa: E402
import bmesh  # noqa: E402

from _harness import ensure_src_on_path, reload_pipeline_modules  # noqa: E402


# (name, head, tail, parent). Compact mixamo deform skeleton (no toes).
_DEFORM_BONES = [
    ("mixamorig:Hips", (0, 0, 0.95), (0, 0, 1.05), None),
    ("mixamorig:Spine", (0, 0, 1.05), (0, 0, 1.2), "mixamorig:Hips"),
    ("mixamorig:Spine1", (0, 0, 1.2), (0, 0, 1.35), "mixamorig:Spine"),
    ("mixamorig:Spine2", (0, 0, 1.35), (0, 0, 1.45), "mixamorig:Spine1"),
    ("mixamorig:Neck", (0, 0, 1.45), (0, 0, 1.6), "mixamorig:Spine2"),
    ("mixamorig:Head", (0, 0, 1.6), (0, 0, 1.75), "mixamorig:Neck"),
    ("mixamorig:LeftArm", (0, 0.18, 1.45), (0, 0.45, 1.45), "mixamorig:Spine2"),
    ("mixamorig:LeftForeArm", (0, 0.45, 1.45), (0, 0.7, 1.45), "mixamorig:LeftArm"),
    ("mixamorig:LeftHand", (0, 0.7, 1.45), (0, 0.8, 1.45), "mixamorig:LeftForeArm"),
    ("mixamorig:RightArm", (0, -0.18, 1.45), (0, -0.45, 1.45), "mixamorig:Spine2"),
    ("mixamorig:RightForeArm", (0, -0.45, 1.45), (0, -0.7, 1.45), "mixamorig:RightArm"),
    ("mixamorig:RightHand", (0, -0.7, 1.45), (0, -0.8, 1.45), "mixamorig:RightForeArm"),
    ("mixamorig:LeftUpLeg", (0, 0.1, 0.95), (0, 0.1, 0.55), "mixamorig:Hips"),
    ("mixamorig:LeftLeg", (0, 0.1, 0.55), (0, 0.1, 0.1), "mixamorig:LeftUpLeg"),
    ("mixamorig:LeftFoot", (0, 0.1, 0.1), (0.15, 0.1, 0.0), "mixamorig:LeftLeg"),
    ("mixamorig:RightUpLeg", (0, -0.1, 0.95), (0, -0.1, 0.55), "mixamorig:Hips"),
    ("mixamorig:RightLeg", (0, -0.1, 0.55), (0, -0.1, 0.1), "mixamorig:RightUpLeg"),
    ("mixamorig:RightFoot", (0, -0.1, 0.1), (0.15, -0.1, 0.0), "mixamorig:RightLeg"),
]

# Bushier biped control rig WITH toes; names do NOT match the mixamo vertex groups.
_CONTROL_BONES = [
    ("Bip01 COM", (0, 0, 0), (0, 0, 0.95), None),
    ("Bip01 Pelvis", (0, 0, 0.95), (0, 0, 1.05), "Bip01 COM"),
    ("Bip01 Spine0", (0, 0, 1.05), (0, 0, 1.25), "Bip01 Pelvis"),
    ("Bip01 Spine1", (0, 0, 1.25), (0, 0, 1.45), "Bip01 Spine0"),
    ("Bip01 Neck", (0, 0, 1.45), (0, 0, 1.6), "Bip01 Spine1"),
    ("Bip01 Head", (0, 0, 1.6), (0, 0, 1.75), "Bip01 Neck"),
    ("Bip01 LUpperArm", (0, 0.18, 1.45), (0, 0.45, 1.45), "Bip01 Spine1"),
    ("Bip01 LForeArm", (0, 0.45, 1.45), (0, 0.7, 1.45), "Bip01 LUpperArm"),
    ("Bip01 LHand", (0, 0.7, 1.45), (0, 0.8, 1.45), "Bip01 LForeArm"),
    ("Bip01 RUpperArm", (0, -0.18, 1.45), (0, -0.45, 1.45), "Bip01 Spine1"),
    ("Bip01 RForeArm", (0, -0.45, 1.45), (0, -0.7, 1.45), "Bip01 RUpperArm"),
    ("Bip01 RHand", (0, -0.7, 1.45), (0, -0.8, 1.45), "Bip01 RForeArm"),
    ("Bip01 LThigh", (0, 0.1, 0.95), (0, 0.1, 0.55), "Bip01 Pelvis"),
    ("Bip01 LCalf", (0, 0.1, 0.55), (0, 0.1, 0.1), "Bip01 LThigh"),
    ("Bip01 LFoot", (0, 0.1, 0.1), (0.15, 0.1, 0.0), "Bip01 LCalf"),
    ("Bip01 LToe0", (0.15, 0.1, 0.0), (0.25, 0.1, 0.0), "Bip01 LFoot"),
    ("Bip01 RThigh", (0, -0.1, 0.95), (0, -0.1, 0.55), "Bip01 Pelvis"),
    ("Bip01 RCalf", (0, -0.1, 0.55), (0, -0.1, 0.1), "Bip01 RThigh"),
    ("Bip01 RFoot", (0, -0.1, 0.1), (0.15, -0.1, 0.0), "Bip01 RCalf"),
    ("Bip01 RToe0", (0.15, -0.1, 0.0), (0.25, -0.1, 0.0), "Bip01 RFoot"),
]

_RUNBOOK_OBJECT_NAMES = ("DeformRig", "ControlRig", "BodyMesh")


def _purge_previous_runbook_objects():
    """Remove objects/armatures/meshes left by a prior run so names stay clean."""
    for name in _RUNBOOK_OBJECT_NAMES:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
    for arm in list(bpy.data.armatures):
        if arm.name.startswith(("DeformRig", "ControlRig")):
            bpy.data.armatures.remove(arm)


def _make_armature(name, bone_defs):
    arm_data = bpy.data.armatures.new(name + "_data")
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    created = {}
    for bname, head, tail, _parent in bone_defs:
        edit_bone = arm_data.edit_bones.new(bname)
        edit_bone.head = head
        edit_bone.tail = tail
        created[bname] = edit_bone
    for bname, _head, _tail, parent in bone_defs:
        if parent is not None:
            created[bname].parent = created[parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def _make_bound_mesh(name, vertex_group_names, armatures):
    mesh = bpy.data.meshes.new(name + "_data")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=0.4)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    for group_name in vertex_group_names:
        obj.vertex_groups.new(name=group_name)
    for arm in armatures:
        modifier = obj.modifiers.new(name="Armature_" + arm.name, type="ARMATURE")
        modifier.object = arm
    return obj


def _build_scene():
    _purge_previous_runbook_objects()
    deform = _make_armature("DeformRig", _DEFORM_BONES)
    control = _make_armature("ControlRig", _CONTROL_BONES)
    vertex_groups = [bone[0] for bone in _DEFORM_BONES]  # mesh skinned to the deform rig
    _make_bound_mesh("BodyMesh", vertex_groups, [deform, control])


def main():
    print("=" * 60)
    print("MULTI-ARMATURE SELECTION RUNBOOK (Issue #32)")
    print("=" * 60)

    ensure_src_on_path()
    reload_pipeline_modules()

    _build_scene()

    output_root = tempfile.mkdtemp(prefix="open_jaywalker_multiarm_runbook_")
    os.environ["OPEN_JAYWALKER_OUTPUT_ROOT"] = output_root
    print("Runbook output dir: {0}".format(output_root))

    from inspector import inspect_scene
    from classifier import write_asset_report

    inspect_result = inspect_scene()
    if not inspect_result:
        raise RuntimeError("Inspector exported nothing; expected two armatures + a mesh.")
    asset_dir = Path(inspect_result["output_dir"]).resolve()
    report = write_asset_report(asset_dir)[0]

    recommended = report["recommended_primary_armature"]
    ranking = report["asset_summary"]["ranking"]
    top = ranking[0]
    by_name = {entry["armature_name"]: entry for entry in ranking}

    print("Discovered armatures: {0}".format(report["asset_summary"]["discovered_armatures"]))
    for entry in ranking:
        print(
            "  {0}: mesh_bound={1} coverage={2} purity={3} score={4} tiebreak={5}".format(
                entry["armature_name"],
                entry["mesh_bound_term"],
                entry["vertex_group_coverage"],
                entry["deform_purity"],
                entry["ranking_score"],
                entry.get("selection_tiebreaker"),
            )
        )

    failures = []
    if recommended != "DeformRig":
        failures.append("expected recommended DeformRig, got {0}".format(recommended))
    if top.get("selection_tiebreaker") != "vertex_group_coverage":
        failures.append(
            "expected tiebreaker vertex_group_coverage, got {0}".format(top.get("selection_tiebreaker"))
        )
    if by_name.get("DeformRig", {}).get("mesh_bound_term") != 10.0:
        failures.append("DeformRig not fully mesh-bound (expected mesh_bound_term 10.0)")
    if by_name.get("ControlRig", {}).get("mesh_bound_term") != 10.0:
        failures.append("ControlRig not fully mesh-bound (expected mesh_bound_term 10.0)")

    print("-" * 60)
    print("Overall: {0}".format("PASS" if not failures else "FAIL"))
    for failure in failures:
        print("  - {0}".format(failure))


if __name__ == "__main__":
    main()
else:
    main()

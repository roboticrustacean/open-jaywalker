"""
In-Blender mesh-deformation runbook for the openmatexamplehuman fixture (Issue #21).

Also exercises the Issue #23 root-anchor contract: confirms that Grp_Root sits
at bbox_ground_center, the generated Root bone reuses the source root's world
position, and the duplicated mesh's matrix_world is preserved.

Usage:
  1. In VS Code: `Blender: Start`.
  2. In Blender: File -> Open the openmatexamplehuman `.blend` file
     (the project does not commit this file - it lives next to your other
     OpenMATERIAL sample assets).
  3. In VS Code: open this file, then `Blender: Run Script`.

Results are written to
`docs/testing/openmatexamplehuman_mesh_deformation_report.md`.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

import bpy  # noqa: E402

from _harness import run_full_pipeline, run_mesh_deformation_suite  # noqa: E402


POSE_CASES = [
    {"bone": "Upper_Arm_Left", "axis": "X", "degrees": 35.0},
    {"bone": "Upper_Leg_Right", "axis": "Y", "degrees": 30.0},
    {"bone": "Neck", "axis": "X", "degrees": 25.0},
]

ROOT_ANCHOR_TOLERANCE_M = 1e-5


def assert_root_anchor_contract(pipeline_result: dict) -> None:
    """Verify Issue #23 root-anchor invariants on the live Blender scene."""
    asset_dir = pipeline_result["asset_dir"]
    build_plan = json.loads((asset_dir / "build_plan.json").read_text(encoding="utf-8"))
    root_resolution = build_plan["root_resolutions"][0]
    expected_origin = tuple(root_resolution["grp_root_local_origin"])

    grp_root = bpy.data.objects.get("Grp_Root")
    assert grp_root is not None, "Grp_Root was not created"
    actual_origin = tuple(grp_root.location)
    assert all(
        abs(a - e) < ROOT_ANCHOR_TOLERANCE_M for a, e in zip(actual_origin, expected_origin)
    ), "Grp_Root.location {0} != expected {1}".format(actual_origin, expected_origin)

    generated_armature = bpy.data.objects.get(pipeline_result["generated_armature_name"])
    assert generated_armature is not None, "Generated armature missing after build"
    root_bone = generated_armature.data.bones.get("Root")
    assert root_bone is not None, "Generated Root bone missing"
    # In armature-local frame, openmatexamplehuman's source root sat at world
    # (0, 0, 0); the rebase to Grp_Root-local space yields (-grp_root_local_origin).
    expected_root_local = tuple(-v for v in expected_origin)
    actual_root_local = tuple(root_bone.head_local)
    assert all(
        abs(a - e) < ROOT_ANCHOR_TOLERANCE_M for a, e in zip(actual_root_local, expected_root_local)
    ), "Root.head_local {0} != expected {1}".format(actual_root_local, expected_root_local)

    for mesh_record in build_plan["mesh_binding"]["meshes"]:
        source = bpy.data.objects.get(mesh_record["mesh_name"])
        generated = bpy.data.objects.get("ASAM_{0}".format(mesh_record["mesh_name"]))
        if source is None or generated is None:
            continue
        assert source.matrix_world == generated.matrix_world, (
            "Generated mesh world matrix differs from source for {0}".format(
                mesh_record["mesh_name"]
            )
        )
    print("Root-anchor contract: OK")


def main() -> None:
    print("=" * 60)
    print("MESH DEFORMATION RUNBOOK - openmatexamplehuman (Issues #21, #23)")
    print("=" * 60)
    pipeline_result = run_full_pipeline(bpy)
    assert_root_anchor_contract(pipeline_result)
    run_mesh_deformation_suite(bpy, POSE_CASES, pipeline_result=pipeline_result)


if __name__ == "__main__":
    main()
else:
    main()

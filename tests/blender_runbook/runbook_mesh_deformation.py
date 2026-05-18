"""
In-Blender mesh-deformation runbook for the LowPolyCharacter4 fixture (Issue #21).

Usage:
  1. In VS Code: `Blender: Start` (launches Blender via the extension).
  2. In Blender: File -> Open -> sample_assets/LowPolyCharacter4.blend
  3. In VS Code: open this file, then `Blender: Run Script`.

The script runs the full pipeline (inspector -> classifier -> builder), then
pose-rotates a handful of bones on the GENERATED armature and asserts that the
duplicated mesh's evaluated bounding box moves. Results are written to
`docs/testing/LowPolyCharacter4_mesh_deformation_report.md`.
"""

from __future__ import annotations

import os
import sys

# Ensure tests/blender_runbook is importable when run via Blender: Run Script.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

import bpy  # noqa: E402

from _harness import run_mesh_deformation_suite  # noqa: E402


POSE_CASES = [
    {"bone": "Upper_Arm_Left", "axis": "X", "degrees": 35.0},
    {"bone": "Upper_Leg_Right", "axis": "Y", "degrees": 30.0},
    {"bone": "Lower_Spine", "axis": "X", "degrees": 20.0},
]


def main() -> None:
    print("=" * 60)
    print("MESH DEFORMATION RUNBOOK - LowPolyCharacter4 (Issue #21)")
    print("=" * 60)
    run_mesh_deformation_suite(bpy, POSE_CASES)


if __name__ == "__main__":
    main()
else:
    main()

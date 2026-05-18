"""
In-Blender mesh-deformation runbook for the openmatexamplehuman fixture (Issue #21).

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

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

import bpy  # noqa: E402

from _harness import run_mesh_deformation_suite  # noqa: E402


POSE_CASES = [
    {"bone": "Upper_Arm_Left", "axis": "X", "degrees": 35.0},
    {"bone": "Upper_Leg_Right", "axis": "Y", "degrees": 30.0},
    {"bone": "Neck", "axis": "X", "degrees": 25.0},
]


def main() -> None:
    print("=" * 60)
    print("MESH DEFORMATION RUNBOOK - openmatexamplehuman (Issue #21)")
    print("=" * 60)
    run_mesh_deformation_suite(bpy, POSE_CASES)


if __name__ == "__main__":
    main()
else:
    main()

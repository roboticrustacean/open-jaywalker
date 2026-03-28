"""
Combined Blender entrypoint for inspector + classifier.
"""

from __future__ import annotations

import importlib
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ARMATURE_DIR = os.path.join(SRC_DIR, "armature_inspector")
CLASSIFIER_DIR = os.path.join(SRC_DIR, "phase3_classifier")

for path in (SCRIPT_DIR, SRC_DIR, ARMATURE_DIR, CLASSIFIER_DIR):
    if path not in sys.path:
        sys.path.append(path)

import bpy  # noqa: F401

import inspector
import classifier
import workflow

importlib.reload(inspector)
importlib.reload(classifier)
importlib.reload(workflow)

from inspector import inspect_scene  # noqa: E402
from classifier import print_console_summary, write_asset_report  # noqa: E402
from workflow import run_combined_workflow  # noqa: E402


def main():
    """Run Phase 2 inspector and Phase 3 classifier in sequence."""
    print("\n" + "=" * 60)
    print("STARTING OPEN-JAYWALKER PIPELINE")
    print("=" * 60 + "\n")
    run_combined_workflow(
        inspect_scene_fn=inspect_scene,
        classify_fn=write_asset_report,
        print_summary_fn=print_console_summary,
    )


if __name__ == "__main__":
    main()
else:
    main()

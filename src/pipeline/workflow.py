"""
Combined Blender workflow orchestration for Phase 2 + Phase 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional


def run_combined_workflow(
    inspect_scene_fn: Callable[[], dict],
    classify_fn: Callable[[Path], tuple],
    print_summary_fn: Callable[[dict, Path], None],
) -> Optional[dict]:
    """
    Run the inspector first, then classify the freshly exported asset folder.

    The inspector remains the source of truth for Phase 2 JSON generation.
    Classification only runs when armatures were exported successfully.
    """
    inspect_result = inspect_scene_fn()
    if not inspect_result:
        return None

    output_dir = Path(inspect_result["output_dir"]).resolve()
    exported_files = list(inspect_result.get("exported_files", []))
    diagnostics_ran = bool(inspect_result.get("diagnostics_ran"))

    if diagnostics_ran or not exported_files:
        print("\nSkipping Phase 3 classifier because no armature exports were produced.")
        return {
            "inspect_result": inspect_result,
            "classifier_report": None,
            "classifier_report_path": None,
        }

    if Path(output_dir).name == "unsaved":
        print("\nWARNING: Classifying output/unsaved. This folder may contain files from prior runs.")

    print("\n" + "=" * 60)
    print("STARTING PHASE 3 CLASSIFIER")
    print("=" * 60 + "\n")

    classifier_report, classifier_report_path = classify_fn(output_dir)
    print_summary_fn(classifier_report, classifier_report_path)

    return {
        "inspect_result": inspect_result,
        "classifier_report": classifier_report,
        "classifier_report_path": str(classifier_report_path),
    }

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

    classifier_report, build_plan, classifier_report_path, build_plan_path = classify_fn(output_dir)
    print_summary_fn(classifier_report, build_plan, classifier_report_path, build_plan_path)

    return {
        "inspect_result": inspect_result,
        "classifier_report": classifier_report,
        "classifier_report_path": str(classifier_report_path),
        "build_plan": build_plan,
        "build_plan_path": str(build_plan_path),
    }


def run_full_pipeline(
    inspect_scene_fn: Callable[[], dict],
    classify_fn: Callable[[Path], tuple],
    print_summary_fn: Callable[..., None],
    confirm_fn: Callable[[Path, dict], bool],
    build_fn: Callable[[Path], object],
) -> Optional[dict]:
    """Run inspect -> classify, then gate on confirm_fn before build_fn.

    confirm_fn receives (asset_dir, build_plan) and returns whether to build.
    build_fn receives asset_dir and performs the build. Both are injected so the
    orchestration is unit-testable without Blender. Returns the combined-workflow
    result dict, augmented with "build_result" when a build was attempted.
    """
    result = run_combined_workflow(inspect_scene_fn, classify_fn, print_summary_fn)
    if not result or result.get("build_plan") is None:
        # No armatures / diagnostics-only: nothing to build.
        return result

    asset_dir = Path(result["inspect_result"]["output_dir"]).resolve()
    build_plan = result["build_plan"]

    if confirm_fn(asset_dir, build_plan):
        result["build_result"] = build_fn(asset_dir)
    else:
        result["build_result"] = None
    return result

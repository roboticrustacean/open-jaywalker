"""Shared build dispatch: build the ASAM human(s) for a resolved asset dir.

Used by both entry points (asam_human_builder/main.py and pipeline/main.py) so
the single/crowd dispatch lives in one place. bpy is passed in by the caller;
this module does not import bpy, so it is importable and testable headless.
"""

from __future__ import annotations

from pathlib import Path

from asam_human_builder.builder import (
    build_character_specs_from_asset_dir,
    build_crowd_builder_report,
    print_builder_summary,
    success_message,
    write_builder_report,
    write_crowd_builder_report,
)
from asam_human_builder.blender_builder import (
    build_armature_in_blender,
    build_crowd_in_blender,
)


def run_build(asset_dir, bpy) -> dict:
    """Resolve specs for asset_dir and build them (single or crowd).

    Returns the builder report (single) or crowd builder report (crowd).
    """
    asset_dir = Path(asset_dir).resolve()
    resolved = build_character_specs_from_asset_dir(asset_dir)

    if resolved["crowd"]:
        crowd_execution = build_crowd_in_blender(
            resolved["asset_name"],
            resolved["wrapper_collection_name"],
            resolved["character_specs"],
            resolved["decomposition"],
            bpy,
        )
        report = build_crowd_builder_report(
            resolved["asset_name"],
            resolved["decomposition"],
            resolved["character_specs"],
            crowd_execution,
        )
        _, report_path = write_crowd_builder_report(asset_dir, report)
    else:
        build_spec = resolved["specs"][0]
        execution_result = build_armature_in_blender(build_spec, bpy)
        report, report_path = write_builder_report(asset_dir, build_spec, execution_result)
        print_builder_summary(report, report_path)

    print(success_message(report, report_path))
    return report

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
    export_generated_blend,
    export_generated_gltf,
    purge_previous_generated_artifacts,
)

# Output packaging modes. Shared canonical definition so the addon (Task C3) and the
# builder reference the same literals instead of duplicating magic strings.
PACKAGING_INPLACE_EXPORT = "inplace_export"
PACKAGING_INPLACE_ONLY = "inplace_only"
PACKAGING_SEPARATE_ONLY = "separate_only"
PACKAGING_MODES = (PACKAGING_INPLACE_EXPORT, PACKAGING_INPLACE_ONLY, PACKAGING_SEPARATE_ONLY)


def run_build(asset_dir, bpy, packaging_mode=PACKAGING_INPLACE_EXPORT, export_gltf=False) -> dict:
    """Resolve specs for asset_dir and build them (single or crowd).

    packaging_mode controls post-build export:
      "inplace_export" (default): build in place, then export generated wrapper
          collection to <asset_dir>/<asset_name>_asam.blend.
      "inplace_only": build only, no export.
      "separate_only": export, then purge generated data from the open file.

    Export failures are recorded on the report and never abort the in-place build.

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
        wrapper_name = resolved["wrapper_collection_name"]
    else:
        build_spec = resolved["specs"][0]
        execution_result = build_armature_in_blender(build_spec, bpy)
        report, report_path = write_builder_report(asset_dir, build_spec, execution_result)
        print_builder_summary(report, report_path)
        wrapper_name = build_spec["generated_collection_name"]

    _export_if_requested(asset_dir, resolved["asset_name"], wrapper_name, bpy, packaging_mode, export_gltf, report)

    if resolved["crowd"]:
        for char_report in report.get("characters", []):
            heuristics = char_report.get("targets_created_heuristically", [])
            if heuristics:
                print("WARNING: Character '{0}' has synthesized inert bones added for compliance: {1}".format(
                    char_report.get("character_id"),
                    ", ".join(heuristics)
                ))
    else:
        heuristics = report.get("targets_created_heuristically", [])
        if heuristics:
            print("WARNING: Synthesized inert bones added for compliance: {0}".format(
                ", ".join(heuristics)
            ))

    print(success_message(report, report_path))
    return report


def _export_if_requested(asset_dir, asset_name, wrapper_name, bpy, packaging_mode, export_gltf, report):
    """Record packaging outcome on the report and export when requested.

    Modes:
      - PACKAGING_INPLACE_ONLY: no export, no purge.
      - PACKAGING_INPLACE_EXPORT: write the generated wrapper to a clean .blend.
      - PACKAGING_SEPARATE_ONLY: export, then purge generated data from the open file.

    `packaging_mode` is always recorded (even for inplace_only) so the report shape is
    consistent across modes. Export failures are caught and recorded; they never abort
    the already-completed in-place build.
    """
    report["packaging_mode"] = packaging_mode
    report["exported_blend_path"] = None
    report["export_error"] = None
    if packaging_mode == PACKAGING_INPLACE_ONLY:
        return
    out_path = str(Path(asset_dir) / "{0}_asam.blend".format(asset_name))
    try:
        report["exported_blend_path"] = export_generated_blend(bpy, wrapper_name, out_path)
    except Exception as exc:  # never abort the already-completed in-place build
        report["export_error"] = str(exc)
        
    report["export_gltf"] = export_gltf
    report["exported_gltf_path"] = None
    report["export_gltf_error"] = None
    if export_gltf:
        out_gltf = str(Path(asset_dir) / "{0}_asam.glb".format(asset_name))
        try:
            report["exported_gltf_path"] = export_generated_gltf(bpy, wrapper_name, out_gltf)
        except Exception as exc:
            report["export_gltf_error"] = str(exc)
            
    if packaging_mode == PACKAGING_SEPARATE_ONLY:
        purge_previous_generated_artifacts(bpy)

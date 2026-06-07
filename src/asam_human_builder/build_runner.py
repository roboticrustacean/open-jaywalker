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
    export_crowd_characters_blend,
    export_crowd_characters_gltf,
    export_generated_blend,
    export_generated_gltf,
    purge_previous_generated_artifacts,
)


def run_build(
    asset_dir, bpy, export_blend=False, export_gltf=True, per_character=False,
    export_dir=None,
) -> dict:
    """Resolve specs for asset_dir and build them (single or crowd).

    export_blend: export generated wrapper to .blend format.
    export_gltf: export generated wrapper to .glb format.
    per_character: when True and the build is a crowd, export each character
        as a separate file instead of a single monolithic file.
    export_dir: directory for .blend/.glb deliverables.  When *None*,
        defaults to ``<asset_dir>/exports/`` (or the OPEN_JAYWALKER_EXPORT_DIR
        override).  Pipeline JSON reports always stay in *asset_dir*.

    Export failures are recorded on the report and never abort the in-place build.

    Returns the builder report (single) or crowd builder report (crowd).
    """
    asset_dir = Path(asset_dir).resolve()
    resolved = build_character_specs_from_asset_dir(asset_dir)

    if export_dir is None:
        from pipeline_paths import resolve_export_dir
        export_dir = resolve_export_dir(resolved["asset_name"], asset_dir=asset_dir)
    else:
        export_dir = Path(export_dir).resolve()
        export_dir.mkdir(parents=True, exist_ok=True)

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

    is_crowd = resolved["crowd"]
    _export_if_requested(
        export_dir, resolved["asset_name"], wrapper_name, bpy,
        export_blend, export_gltf, per_character, is_crowd, report,
    )

    if is_crowd:
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


def _export_if_requested(
    export_dir, asset_name, wrapper_name, bpy,
    export_blend, export_gltf, per_character, is_crowd, report,
):
    """Record export outcome on the report and export when requested.

    export_blend / export_gltf control which formats are exported.
    per_character, when True for a crowd build, exports each character as a
    separate file instead of one monolithic archive.

    All export paths are relative to *export_dir* (not *asset_dir*).  Pipeline
    JSON reports remain in *asset_dir*; only binary deliverables land here.

    Export failures are caught and recorded; they never abort the
    already-completed in-place build.
    """
    report["export_dir"] = str(export_dir)
    report["export_blend"] = export_blend
    report["exported_blend_path"] = None
    report["exported_blend_paths"] = None
    report["export_blend_error"] = None
    if export_blend:
        try:
            if per_character and is_crowd:
                report["exported_blend_paths"] = export_crowd_characters_blend(
                    bpy, wrapper_name, str(export_dir)
                )
            else:
                out_path = str(Path(export_dir) / "{0}_asam.blend".format(asset_name))
                report["exported_blend_path"] = export_generated_blend(bpy, wrapper_name, out_path)
        except Exception as exc:
            report["export_blend_error"] = str(exc)

    report["export_gltf"] = export_gltf
    report["exported_gltf_path"] = None
    report["exported_gltf_paths"] = None
    report["export_gltf_error"] = None
    if export_gltf:
        try:
            if per_character and is_crowd:
                report["exported_gltf_paths"] = export_crowd_characters_gltf(
                    bpy, wrapper_name, str(export_dir)
                )
            else:
                out_gltf = str(Path(export_dir) / "{0}_asam.glb".format(asset_name))
                report["exported_gltf_path"] = export_generated_gltf(bpy, wrapper_name, out_gltf)
        except Exception as exc:
            report["export_gltf_error"] = str(exc)

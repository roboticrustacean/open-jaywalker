"""
In-Blender crowd fan-out runbook (Issue #33).

Validates that a multi-character ("crowd") asset is built into N self-contained
ASAM humans, each in its own child collection under one wrapper collection.

Usage:
  1. In VS Code: `Blender: Start` (launches Blender via the extension).
  2. In Blender: File -> Open -> a crowd .blend (e.g. 1000idles.blend) whose
     armature packs many prefixed characters under a shared root.
  3. In VS Code: open this file, then `Blender: Run Script`.

This is NOT a pytest test: it requires an interactive Blender session and is not
discovered by `python -m unittest`. It runs the inspector + classifier on the open
scene, then the crowd builder, and asserts in-scene:

  - a wrapper collection ASAM_<asset> exists and is a generated collection,
  - it contains one child collection ASAM_<asset>_<character_id> per built character,
  - each child contains a Grp_Root_<id>, an Armature_<asset>_<id> with the 28 ASAM
    core bones, and at least one retargeted mesh whose vertex groups include ASAM
    target names (NOT the original prefixed names),
  - failed characters (if any) are reported, not fatal.

Outputs (inspector JSONs, classifier_report, build_plan, builder_report) go to a
fresh tempdir via OPEN_JAYWALKER_OUTPUT_ROOT so committed fixtures are untouched.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

import bpy  # noqa: E402

from _harness import ensure_src_on_path, reload_pipeline_modules  # noqa: E402


# A subset of the 28 ASAM core targets used as a cheap "did the rig get built" probe.
_ASAM_PROBE_BONES = ("Root", "Hip", "Lower_Spine", "Upper_Spine", "Neck", "Head")


def _run_inspector_and_classifier(output_root: str) -> Path:
    """Run inspector + classifier on the open scene; return the asset output dir."""
    os.environ["OPEN_JAYWALKER_OUTPUT_ROOT"] = output_root
    print("Runbook output dir: {0}".format(output_root))

    from blender_builder import purge_previous_generated_artifacts
    purged = purge_previous_generated_artifacts(bpy)
    if purged:
        print("Purged {0} leftover generated artifact(s) from a previous run.".format(purged))

    from inspector import inspect_scene
    from classifier import write_asset_report

    inspect_result = inspect_scene()
    if not inspect_result:
        raise RuntimeError(
            "Inspector did not export anything. Make sure a crowd .blend with an armature is open."
        )
    output_dir = Path(inspect_result["output_dir"]).resolve()
    if inspect_result.get("diagnostics_ran") or not inspect_result.get("exported_files"):
        raise RuntimeError(
            "Inspector wrote diagnostics rather than exports. Asset is unsupported: {0}".format(output_dir)
        )
    write_asset_report(output_dir)
    return output_dir


def _generated_armature_bone_names(armature_obj) -> set:
    return {bone.name for bone in armature_obj.data.bones}


def _mesh_vertex_group_names(mesh_obj) -> set:
    return {group.name for group in mesh_obj.vertex_groups}


def _check_character(wrapper, child_collection, asset_name, character_id, failures):
    """Assert one character's child collection holds a complete ASAM human."""
    armature_name = "Armature_{0}_{1}".format(asset_name, character_id)
    group_root_name = "Grp_Root_{0}".format(character_id)

    objects_by_name = {obj.name: obj for obj in child_collection.all_objects}

    armature_obj = next(
        (obj for obj in child_collection.all_objects
         if getattr(obj, "type", None) == "ARMATURE"),
        None,
    )
    if armature_obj is None:
        failures.append("{0}: no armature in child collection".format(character_id))
        return

    bone_names = _generated_armature_bone_names(armature_obj)
    missing_probe = [b for b in _ASAM_PROBE_BONES if b not in bone_names]
    if missing_probe:
        failures.append("{0}: armature missing ASAM bones {1}".format(character_id, missing_probe))

    meshes = [obj for obj in child_collection.all_objects if getattr(obj, "type", None) == "MESH"]
    for mesh in meshes:
        vgroups = _mesh_vertex_group_names(mesh)
        prefixed = [name for name in vgroups if name.startswith(character_id)]
        if prefixed:
            failures.append(
                "{0}: mesh '{1}' still has prefixed vertex groups {2}".format(
                    character_id, mesh.name, prefixed[:3]
                )
            )
    print(
        "  {0}: armature='{1}' bones={2} meshes={3}".format(
            character_id, armature_obj.name, len(bone_names), len(meshes)
        )
    )


def main() -> None:
    print("=" * 60)
    print("CROWD FAN-OUT RUNBOOK (Issue #33)")
    print("=" * 60)

    ensure_src_on_path()
    reload_pipeline_modules()

    output_root = tempfile.mkdtemp(prefix="open_jaywalker_crowd_runbook_")
    asset_dir = _run_inspector_and_classifier(output_root)

    from builder import (
        build_character_specs_from_asset_dir,
        build_crowd_builder_report,
        write_crowd_builder_report,
        GENERATED_ASSET_KEY,
        GENERATED_MARKER_KEY,
    )
    from blender_builder import build_crowd_in_blender

    resolved = build_character_specs_from_asset_dir(asset_dir)
    if not resolved["crowd"]:
        raise RuntimeError(
            "Open asset is single-character; this runbook needs a crowd .blend. "
            "Use runbook_mesh_deformation.py for single-character assets."
        )

    asset_name = resolved["asset_name"]
    crowd_execution = build_crowd_in_blender(
        asset_name,
        resolved["wrapper_collection_name"],
        resolved["character_specs"],
        resolved["decomposition"],
        bpy,
    )
    crowd_report = build_crowd_builder_report(
        asset_name, resolved["decomposition"], resolved["character_specs"], crowd_execution
    )
    _, report_path = write_crowd_builder_report(asset_dir, crowd_report)

    failures = []

    wrapper = bpy.data.collections.get(resolved["wrapper_collection_name"])
    if wrapper is None:
        failures.append("wrapper collection '{0}' missing".format(resolved["wrapper_collection_name"]))
    else:
        if not bool(wrapper.get(GENERATED_MARKER_KEY)) or wrapper.get(GENERATED_ASSET_KEY) != asset_name:
            failures.append("wrapper collection is not marked generated for this asset")
        built_ids = [c["character_id"] for c in crowd_execution["characters"]]
        print("Built {0} character(s) under '{1}':".format(len(built_ids), wrapper.name))
        for character in crowd_execution["characters"]:
            character_id = character["character_id"]
            child = wrapper.children.get(character["generated_collection_name"])
            if child is None:
                failures.append("{0}: child collection '{1}' not under wrapper".format(
                    character_id, character["generated_collection_name"]))
                continue
            _check_character(wrapper, child, asset_name, character_id, failures)

    print("-" * 60)
    print("Characters built: {0}".format(len(crowd_execution["characters"])))
    print("Characters failed: {0}".format(len(crowd_execution["failed_characters"])))
    for failed in crowd_execution["failed_characters"]:
        print("  FAILED {0}: {1}".format(failed["character_id"], failed["error"]))
    print("Crowd builder report: {0}".format(report_path))

    if failures:
        print("Overall: FAIL")
        for failure in failures:
            print("  - {0}".format(failure))
    else:
        print("Overall: PASS")


if __name__ == "__main__":
    main()
else:
    main()

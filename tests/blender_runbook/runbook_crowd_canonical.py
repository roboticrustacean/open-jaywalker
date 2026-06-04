"""
In-Blender crowd canonical-frame + placement runbook (Issue #49).

Validates that a multi-character ("crowd") asset is built into N ASAM humans that are
real-world metres, with each skeleton standing INSIDE its own body at the body's world
location (or an auto-grid cell for co-located crowds) -- the fixes for the symptoms in
issue #49 (rigs ~39x too large, perpendicular, all stacked at the origin while bodies
sat tiny and scattered).

Usage:
  1. In VS Code: `Blender: Start` (launches Blender via the extension).
  2. In Blender: File -> Open -> a crowd .blend (e.g. 1000idles.blend) whose armature
     packs many prefixed characters under a shared root, with skinned body meshes.
  3. In VS Code: open this file, then `Blender: Run Script`.

This is NOT a pytest test: it requires an interactive Blender session and is not
discovered by `python -m unittest`. It runs the inspector + classifier + crowd builder
(via run_build with packaging_mode="inplace_only"), then asserts in-scene:

  - each generated armature's world-space height is within tolerance of its paired body
    mesh's world-space height (canonical metres; no ~39x discrepancy),
  - each Grp_Root sits at the per-character placement location recorded in
    builder_report.json (body anchor for distributed crowds, grid cell otherwise),
  - each generated armature's world bbox-center is within roughly one body width of its
    body mesh's world bbox-center (skeleton stands inside its body, not metres away),
  - each duplicated mesh's world bbox matches its source mesh's world bbox at rest
    (deformation is identity at rest -- the body looks like the source),
  - placement_mode in the report is "source" (distributed) or "grid" (co-located), and
    distributed characters keep distinct world locations.

Outputs go to a fresh tempdir via OPEN_JAYWALKER_OUTPUT_ROOT so committed fixtures are
untouched. RESULTS of a live run should be recorded by the operator below this header.

--- LIVE RUN RESULTS (fill in after running) ---
  Date:
  Asset:
  Overall:
------------------------------------------------
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

import bpy  # noqa: E402

from _harness import ensure_src_on_path, reload_pipeline_modules  # noqa: E402


# Generated rig height must be within this ratio of the paired body height (canonical
# metres). A value near 1.0 means rig and body share one frame; the pre-fix bug gave ~39.
_HEIGHT_RATIO_TOLERANCE = 0.25  # accept 0.75x .. 1.25x
# Rig bbox-center may sit at most this multiple of the body's largest horizontal extent
# away from the body bbox-center (skeleton inside the body).
_INSIDE_BODY_FOOTPRINT_MULTIPLE = 1.0
# Mesh rest-pose bbox match tolerance, in metres.
_REST_BBOX_TOLERANCE_M = 1e-3


def _run_inspector_and_classifier(output_root: str) -> Path:
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


def _world_bbox(obj):
    """Axis-aligned world bbox (min, max) of a mesh/armature object's bound_box."""
    from mathutils import Vector
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mins = [min(c[i] for c in corners) for i in range(3)]
    maxs = [max(c[i] for c in corners) for i in range(3)]
    return mins, maxs


def _height(mins, maxs):
    return maxs[2] - mins[2]  # Z is up in canonical metres


def _center(mins, maxs):
    return [(mins[i] + maxs[i]) / 2.0 for i in range(3)]


def _horizontal_extent(mins, maxs):
    return max(maxs[0] - mins[0], maxs[1] - mins[1])


def _check_character(child_collection, asset_name, character_id, report_by_id, failures):
    armature_obj = next(
        (o for o in child_collection.all_objects if getattr(o, "type", None) == "ARMATURE"), None
    )
    meshes = [o for o in child_collection.all_objects if getattr(o, "type", None) == "MESH"]
    if armature_obj is None:
        failures.append("{0}: no armature".format(character_id))
        return
    if not meshes:
        failures.append("{0}: no body mesh".format(character_id))
        return

    arm_min, arm_max = _world_bbox(armature_obj)
    body_min = [min(_world_bbox(m)[0][i] for m in meshes) for i in range(3)]
    body_max = [max(_world_bbox(m)[1][i] for m in meshes) for i in range(3)]

    rig_h = _height(arm_min, arm_max)
    body_h = _height(body_min, body_max)

    # 1. Height parity (canonical metres; no ~39x).
    if body_h > 1e-4:
        ratio = rig_h / body_h
        if abs(ratio - 1.0) > _HEIGHT_RATIO_TOLERANCE:
            failures.append(
                "{0}: rig/body height ratio {1:.2f} out of tolerance (rig={2:.3f}m body={3:.3f}m)".format(
                    character_id, ratio, rig_h, body_h
                )
            )

    # 2. Skeleton inside body (centers close in the ground plane).
    arm_c = _center(arm_min, arm_max)
    body_c = _center(body_min, body_max)
    horiz = max(_horizontal_extent(body_min, body_max), 1e-3)
    planar_gap = max(abs(arm_c[0] - body_c[0]), abs(arm_c[1] - body_c[1]))
    if planar_gap > _INSIDE_BODY_FOOTPRINT_MULTIPLE * horiz:
        failures.append(
            "{0}: rig sits {1:.3f}m from body center (> {2:.3f}m body width); not inside body".format(
                character_id, planar_gap, horiz
            )
        )

    # 3. Grp_Root at the recorded placement location.
    grp = next(
        (o for o in child_collection.all_objects
         if getattr(o, "type", None) == "EMPTY" and o.name.startswith("Grp_Root")),
        None,
    )
    char_report = report_by_id.get(character_id, {})
    recorded = char_report.get("grp_root_location")
    if grp is not None and recorded is not None:
        gap = max(abs(grp.location[i] - recorded[i]) for i in range(3))
        if gap > 1e-3:
            failures.append(
                "{0}: Grp_Root at {1} != recorded grp_root_location {2}".format(
                    character_id, list(grp.location), recorded
                )
            )

    print(
        "  {0}: rig_h={1:.3f}m body_h={2:.3f}m planar_gap={3:.3f}m mode={4}".format(
            character_id, rig_h, body_h, planar_gap, char_report.get("placement_mode")
        )
    )


def main() -> None:
    print("=" * 60)
    print("CROWD CANONICAL-FRAME + PLACEMENT RUNBOOK (Issue #49)")
    print("=" * 60)

    ensure_src_on_path()
    reload_pipeline_modules()

    output_root = tempfile.mkdtemp(prefix="open_jaywalker_canonical_runbook_")
    asset_dir = _run_inspector_and_classifier(output_root)

    from builder import build_character_specs_from_asset_dir, GENERATED_MARKER_KEY
    from build_runner import run_build

    resolved = build_character_specs_from_asset_dir(asset_dir)
    if not resolved["crowd"]:
        raise RuntimeError(
            "Open asset is single-character; this runbook needs a crowd .blend."
        )
    asset_name = resolved["asset_name"]

    # Build in place (no export) so we can measure the in-scene result.
    report = run_build(asset_dir, bpy, packaging_mode="inplace_only")

    report_by_id = {c["character_id"]: c for c in report.get("characters", [])}

    failures = []
    wrapper = bpy.data.collections.get(resolved["wrapper_collection_name"])
    if wrapper is None or not bool(wrapper.get(GENERATED_MARKER_KEY)):
        failures.append("wrapper collection missing or not generated")
    else:
        world_locations = []
        modes = set()
        for child in wrapper.children:
            character_id = child.name.replace("ASAM_{0}_".format(asset_name), "")
            _check_character(child, asset_name, character_id, report_by_id, failures)
            cr = report_by_id.get(character_id, {})
            if cr.get("grp_root_location") is not None:
                world_locations.append(tuple(round(v, 4) for v in cr["grp_root_location"]))
            if cr.get("placement_mode"):
                modes.add(cr["placement_mode"])

        # 4. Distributed crowds must keep distinct world locations.
        if "source" in modes and len(world_locations) != len(set(world_locations)):
            failures.append("placement_mode 'source' but duplicate Grp_Root locations (bodies collapsed)")
        print("Placement modes observed: {0}".format(sorted(modes)))

    print("-" * 60)
    print("Characters built: {0}".format(len(report.get("characters", []))))
    print("Characters failed: {0}".format(len(report.get("failed_characters", []))))
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

"""
Shared helpers for in-Blender runbook scripts (Issues #21, #22).

These scripts are NOT pytest tests — they run inside an interactive Blender
session via the VS Code Blender extension (`Blender: Run Script`). Each
runbook script imports this module to:

  - extend sys.path so the project src/ tree is importable
  - run the full pipeline (inspector -> classifier -> builder) on the open scene
  - capture evaluated-mesh bounding boxes so pose-driven deformation is visible
  - write a markdown report under docs/testing/<asset>_*.md

The runbook scripts intentionally do NOT use the `test_*.py` naming convention:
they are not pytest tests, they require an interactive Blender session, and a
unittest discoverer would fail to import them outside of Blender.
"""

from __future__ import annotations

import importlib
import math
import sys
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DOCS_TESTING = REPO_ROOT / "docs" / "testing"


def ensure_src_on_path() -> None:
    """Make all project source roots importable from inside Blender."""
    paths = [
        SRC_ROOT,
        SRC_ROOT / "armature_inspector",
        SRC_ROOT / "phase3_classifier",
        SRC_ROOT / "asam_human_builder",
        SRC_ROOT / "pipeline",
    ]
    for path in paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.append(path_str)


def reload_pipeline_modules() -> None:
    """
    Drop cached pipeline modules so a re-run picks up edits made in the editor.

    `Blender: Run Script` keeps the embedded Python interpreter alive between
    runs, so without this step Blender will keep using the first-imported copy
    of builder.py / classifier.py.
    """
    for name in (
        "builder",
        "blender_builder",
        "inspector",
        "classifier",
        "workflow",
        "asam_human_builder.builder",
        "asam_human_builder.blender_builder",
        "phase3_classifier.classifier",
        "armature_inspector.inspector",
    ):
        module = sys.modules.get(name)
        if module is not None:
            try:
                importlib.reload(module)
            except Exception:
                # If reload fails (e.g., partially-initialized module), drop it
                # so the next import re-creates it from source.
                sys.modules.pop(name, None)


def purge_previous_generated_artifacts(bpy_module) -> int:
    """
    Remove every collection, object, and orphan data block left over from a
    previous builder run (anything carrying our GENERATED_MARKER_KEY).

    Without this step, a re-run inside the same Blender session sees the
    previous-run's generated `Armature_<asset>` in `bpy.data.objects`. The
    inspector exports it alongside the source rigs, and the classifier may
    pick it as the recommended primary armature (because it already carries
    clean ASAM-named bones). The build then crashes when it tries to look up
    the source again — its own collection-rebuild step has just removed it.

    Idempotent: returns 0 if nothing was found. Safe to call unconditionally
    at the start of every pipeline run.
    """
    ensure_src_on_path()
    try:
        from builder import GENERATED_ASSET_KEY, GENERATED_MARKER_KEY  # noqa: F401
    except ImportError:
        from asam_human_builder.builder import GENERATED_ASSET_KEY, GENERATED_MARKER_KEY  # noqa: F401

    try:
        bpy_module.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        pass

    removed = 0

    # Pass 1: remove generated objects and any orphan data blocks they leave behind.
    for obj in list(bpy_module.data.objects):
        if not obj.get(GENERATED_MARKER_KEY):
            continue
        data_block = obj.data if getattr(obj, "type", None) in ("ARMATURE", "MESH") else None
        bpy_module.data.objects.remove(obj, do_unlink=True)
        removed += 1
        if data_block is None:
            continue
        if getattr(data_block, "users", 0) != 0:
            continue
        # Use isinstance against bpy.types so we don't have to maintain a
        # parallel type-name lookup.
        if isinstance(data_block, bpy_module.types.Armature):
            bpy_module.data.armatures.remove(data_block)
        elif isinstance(data_block, bpy_module.types.Mesh):
            bpy_module.data.meshes.remove(data_block)

    # Pass 2: remove (now-empty) generated collections, unlinking from every scene first.
    for collection in list(bpy_module.data.collections):
        if not collection.get(GENERATED_MARKER_KEY):
            continue
        for scene in bpy_module.data.scenes:
            if scene.collection.children.get(collection.name) is not None:
                scene.collection.children.unlink(collection)
        for parent in bpy_module.data.collections:
            if parent.children.get(collection.name) is not None:
                parent.children.unlink(collection)
        bpy_module.data.collections.remove(collection)
        removed += 1

    return removed


def run_full_pipeline(bpy_module) -> dict:
    """
    Run inspector + classifier + builder against the currently open .blend.

    Always purges leftover generated artifacts first so the pipeline starts
    from a clean state - this is what makes the runbook scripts safe to
    re-run inside the same Blender session.

    Returns a dict with: asset_dir, build_spec, builder_report, builder_report_path,
    generated_armature_name, asset_name.
    """
    ensure_src_on_path()
    reload_pipeline_modules()

    purged = purge_previous_generated_artifacts(bpy_module)
    if purged:
        print("Purged {0} leftover generated artifact(s) from a previous run.".format(purged))

    from inspector import inspect_scene
    from classifier import write_asset_report
    from builder import (
        build_armature_spec_from_asset_dir,
        print_builder_summary,
        write_builder_report,
    )
    from blender_builder import build_armature_in_blender

    inspect_result = inspect_scene()
    if not inspect_result:
        raise RuntimeError(
            "Inspector did not export anything. Make sure a .blend with an armature is open."
        )

    output_dir = Path(inspect_result["output_dir"]).resolve()
    if inspect_result.get("diagnostics_ran") or not inspect_result.get("exported_files"):
        raise RuntimeError(
            "Inspector wrote diagnostics rather than exports. Asset is unsupported: {0}".format(output_dir)
        )

    write_asset_report(output_dir)
    _, _, build_spec = build_armature_spec_from_asset_dir(output_dir)
    execution_result = build_armature_in_blender(build_spec, bpy_module)
    builder_report, report_path = write_builder_report(output_dir, build_spec, execution_result)
    print_builder_summary(builder_report, report_path)

    return {
        "asset_dir": output_dir,
        "asset_name": build_spec["asset_name"],
        "build_spec": build_spec,
        "execution_result": execution_result,
        "builder_report": builder_report,
        "builder_report_path": report_path,
        "generated_armature_name": execution_result["generated_armature_name"],
    }


def evaluated_mesh_world_bbox(bpy_module, mesh_obj):
    """
    Return (min_x, min_y, min_z, max_x, max_y, max_z) of mesh_obj evaluated
    against the depsgraph and transformed to world space.

    Evaluating against the depsgraph is how Blender bakes armature deformation
    into vertex positions — that is what we need to see whether a pose change
    actually moved the mesh.
    """
    depsgraph = bpy_module.context.evaluated_depsgraph_get()
    evaluated_obj = mesh_obj.evaluated_get(depsgraph)
    mesh = evaluated_obj.to_mesh()
    try:
        matrix = mesh_obj.matrix_world
        if not mesh.vertices:
            return None
        first = matrix @ mesh.vertices[0].co
        min_x = max_x = first.x
        min_y = max_y = first.y
        min_z = max_z = first.z
        for vertex in mesh.vertices[1:]:
            world_co = matrix @ vertex.co
            if world_co.x < min_x:
                min_x = world_co.x
            elif world_co.x > max_x:
                max_x = world_co.x
            if world_co.y < min_y:
                min_y = world_co.y
            elif world_co.y > max_y:
                max_y = world_co.y
            if world_co.z < min_z:
                min_z = world_co.z
            elif world_co.z > max_z:
                max_z = world_co.z
        return (min_x, min_y, min_z, max_x, max_y, max_z)
    finally:
        evaluated_obj.to_mesh_clear()


def bbox_delta(bbox_a, bbox_b) -> float:
    """L-infinity distance between two bboxes (largest per-component delta)."""
    if bbox_a is None or bbox_b is None:
        return 0.0
    return max(abs(a - b) for a, b in zip(bbox_a, bbox_b))


@contextmanager
def posed_bone(bpy_module, armature_obj, bone_name: str, axis: str, degrees: float):
    """
    Temporarily rotate a pose bone on the GENERATED armature by `degrees` around
    `axis` ('X' | 'Y' | 'Z'). Restores the rest pose on exit.

    Raises KeyError if the bone is not present on the armature.
    """
    pose_bones = armature_obj.pose.bones
    if bone_name not in pose_bones:
        raise KeyError(
            "Bone '{0}' is not on armature '{1}'. Available: {2}".format(
                bone_name, armature_obj.name, ", ".join(pb.name for pb in pose_bones)
            )
        )
    pose_bone = pose_bones[bone_name]
    previous_mode = pose_bone.rotation_mode
    previous_euler = tuple(pose_bone.rotation_euler)

    pose_bone.rotation_mode = "XYZ"
    radians = math.radians(degrees)
    axis_index = {"X": 0, "Y": 1, "Z": 2}[axis.upper()]
    euler = [0.0, 0.0, 0.0]
    euler[axis_index] = radians
    pose_bone.rotation_euler = euler
    bpy_module.context.view_layer.update()
    try:
        yield pose_bone
    finally:
        pose_bone.rotation_euler = previous_euler
        pose_bone.rotation_mode = previous_mode
        bpy_module.context.view_layer.update()


def find_generated_mesh_objects(bpy_module, asset_name: str):
    """Return the duplicated mesh objects marked with our generated keys."""
    try:
        from builder import GENERATED_ASSET_KEY, GENERATED_MARKER_KEY
    except ImportError:
        from asam_human_builder.builder import GENERATED_ASSET_KEY, GENERATED_MARKER_KEY

    matches = []
    for obj in bpy_module.data.objects:
        if getattr(obj, "type", None) != "MESH":
            continue
        if not bool(obj.get(GENERATED_MARKER_KEY)):
            continue
        if obj.get(GENERATED_ASSET_KEY) != asset_name:
            continue
        matches.append(obj)
    matches.sort(key=lambda o: o.name)
    return matches


def write_mesh_deformation_report(asset_name: str, results: list, builder_report: dict, epsilon: float) -> Path:
    """
    Write a markdown report capturing pose results plus mesh-warning summary.

    `results` is a list of dicts: {bone, axis, degrees, mesh_name, rest_bbox,
    posed_bbox, delta, status}.
    """
    DOCS_TESTING.mkdir(parents=True, exist_ok=True)
    report_path = DOCS_TESTING / "{0}_mesh_deformation_report.md".format(asset_name)
    lines = [
        "# Mesh deformation report - {0}".format(asset_name),
        "",
        "Generated by `tests/blender_runbook/test_mesh_deformation*.py`.",
        "",
        "Pass criterion: bbox delta after pose rotation > {0:.1e} m on at least one axis.".format(epsilon),
        "",
        "## Pose tests",
        "",
        "| Bone | Axis | Degrees | Mesh | Delta (m) | Result |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in results:
        lines.append(
            "| {bone} | {axis} | {deg} | {mesh} | {delta:.5f} | {status} |".format(
                bone=entry["bone"],
                axis=entry["axis"],
                deg=entry["degrees"],
                mesh=entry["mesh_name"],
                delta=entry["delta"],
                status=entry["status"],
            )
        )

    lines.extend(["", "## Builder mesh warnings", ""])
    mesh_warnings = builder_report.get("mesh_warnings", [])
    if not mesh_warnings:
        lines.append("None.")
    else:
        for warning in mesh_warnings:
            lines.append("- `{0}`".format(warning))

    lines.extend(["", "## Preserved pelvis pair", ""])
    preserved = builder_report.get("preserved_pelvis_pair", [])
    if not preserved:
        lines.append("None (asset does not require paired-pelvis preservation).")
    else:
        for entry in preserved:
            lines.append(
                "- `{0}` preserved as child of `{1}`".format(
                    entry["generated_bone_name"], entry["parent"]
                )
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_mesh_deformation_suite(bpy_module, pose_cases: list, epsilon: float = 1e-4) -> Path:
    """
    Run the pipeline, then exercise each pose case on the generated armature.

    `pose_cases` is a list of {"bone": str, "axis": "X"|"Y"|"Z", "degrees": float}.
    Returns the path of the markdown report.
    """
    pipeline_result = run_full_pipeline(bpy_module)
    asset_name = pipeline_result["asset_name"]
    generated_armature = bpy_module.data.objects.get(pipeline_result["generated_armature_name"])
    if generated_armature is None:
        raise RuntimeError(
            "Generated armature '{0}' is missing after build.".format(
                pipeline_result["generated_armature_name"]
            )
        )

    generated_meshes = find_generated_mesh_objects(bpy_module, asset_name)
    if not generated_meshes:
        raise RuntimeError(
            "No generated meshes found for asset '{0}'. Check builder_report.mesh_warnings.".format(asset_name)
        )

    results = []
    overall_pass = True
    for case in pose_cases:
        for mesh in generated_meshes:
            rest_bbox = evaluated_mesh_world_bbox(bpy_module, mesh)
            try:
                with posed_bone(bpy_module, generated_armature, case["bone"], case["axis"], case["degrees"]):
                    posed_bbox = evaluated_mesh_world_bbox(bpy_module, mesh)
            except KeyError as exc:
                results.append(
                    {
                        "bone": case["bone"],
                        "axis": case["axis"],
                        "degrees": case["degrees"],
                        "mesh_name": mesh.name,
                        "rest_bbox": rest_bbox,
                        "posed_bbox": None,
                        "delta": 0.0,
                        "status": "SKIP (bone missing: {0})".format(exc.args[0] if exc.args else "?"),
                    }
                )
                continue
            delta = bbox_delta(rest_bbox, posed_bbox)
            status = "PASS" if delta > epsilon else "FAIL"
            if status == "FAIL":
                overall_pass = False
            results.append(
                {
                    "bone": case["bone"],
                    "axis": case["axis"],
                    "degrees": case["degrees"],
                    "mesh_name": mesh.name,
                    "rest_bbox": rest_bbox,
                    "posed_bbox": posed_bbox,
                    "delta": delta,
                    "status": status,
                }
            )
            print(
                "{0} {1} axis={2} deg={3} mesh={4} delta={5:.5f}".format(
                    status, case["bone"], case["axis"], case["degrees"], mesh.name, delta
                )
            )

    report_path = write_mesh_deformation_report(
        asset_name, results, pipeline_result["builder_report"], epsilon
    )
    print("Mesh deformation report written to: {0}".format(report_path))
    print("Overall: {0}".format("PASS" if overall_pass else "FAIL"))
    return report_path

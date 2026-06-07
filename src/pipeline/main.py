"""
Combined Blender entrypoint (headless/CLI): inspector + classifier + (confirm) + builder.

Runs the full pipeline in one launch. After the classifier writes the plan, a
build gate decides whether to continue: a live [y/N] prompt when stdin is a TTY,
otherwise the OPEN_JAYWALKER_AUTO_BUILD env toggle / --build|--no-build arg
(default: stop and print how to build). Interactive GUI use is the Open Jaywalker
add-on; this entry point is for headless / automation / batch runs.
"""

from __future__ import annotations

import importlib
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ARMATURE_DIR = os.path.join(SRC_DIR, "armature_inspector")
CLASSIFIER_DIR = os.path.join(SRC_DIR, "phase3_classifier")
BUILDER_DIR = os.path.join(SRC_DIR, "asam_human_builder")

for path in (SCRIPT_DIR, SRC_DIR, ARMATURE_DIR, CLASSIFIER_DIR, BUILDER_DIR):
    if path not in sys.path:
        sys.path.append(path)

import bpy  # noqa: F401

import inspector
import classifier
import workflow
import build_runner
import blender_builder

importlib.reload(inspector)
importlib.reload(classifier)
importlib.reload(workflow)
importlib.reload(build_runner)
importlib.reload(blender_builder)

from inspector import inspect_scene  # noqa: E402
from classifier import print_console_summary, write_asset_report  # noqa: E402
from workflow import run_full_pipeline  # noqa: E402
from build_runner import run_build  # noqa: E402
from blender_builder import purge_previous_generated_artifacts  # noqa: E402
from pipeline.build_gate import resolve_build_decision  # noqa: E402


def _script_argv(argv):
    """Return the args after a `--` separator (Blender passes script args there)."""
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


def _stdin_isatty():
    """Best-effort TTY check. Blender's embedded stdin can be None or raise;
    treat any failure as "no interactive stdin" so we fall back to the toggle."""
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except (AttributeError, ValueError, OSError):
        return False


def _print_build_skipped(asset_dir, build_plan):
    from pipeline_paths import resolve_export_dir
    asset_name = asset_dir.name
    export_dir = resolve_export_dir(asset_name, asset_dir=asset_dir)
    print("\nPlan written. Build NOT run.")
    print("  Reports: {0}".format(asset_dir))
    print("  Exports: {0}".format(export_dir))
    print("  build_plan.json: {0}".format(os.path.join(str(asset_dir), "build_plan.json")))
    print("  To build: re-run with OPEN_JAYWALKER_AUTO_BUILD=1 (or pass `-- --build`),")
    print("  or run the builder entry: src/asam_human_builder/main.py -- --asset-dir {0}".format(asset_dir))


def _confirm_build(asset_dir, build_plan):
    decision = resolve_build_decision(
        stdin_isatty=_stdin_isatty(),
        env=os.environ,
        argv=_script_argv(sys.argv),
    )
    if decision is True:
        return True
    _print_build_skipped(asset_dir, build_plan)
    return False


def main():
    print("\n" + "=" * 60)
    print("STARTING OPEN-JAYWALKER PIPELINE")
    print("=" * 60 + "\n")
    purge_previous_generated_artifacts(bpy)
    run_full_pipeline(
        inspect_scene_fn=inspect_scene,
        classify_fn=write_asset_report,
        print_summary_fn=print_console_summary,
        confirm_fn=_confirm_build,
        build_fn=lambda asset_dir: run_build(asset_dir, bpy),
    )


if __name__ == "__main__":
    main()
else:
    main()

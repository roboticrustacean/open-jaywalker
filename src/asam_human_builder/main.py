"""Blender entrypoint for building a generated ASAM human armature."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
SRC_DIR = SCRIPT_DIR.parent

for path in (SCRIPT_DIR, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

try:
    import bpy  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - only hit outside Blender
    raise RuntimeError("src/asam_human_builder/main.py must run inside Blender.") from exc

import builder
import blender_builder
import build_runner

importlib.reload(builder)
importlib.reload(blender_builder)
importlib.reload(build_runner)

from builder import (  # noqa: E402
    resolve_default_asset_dir,
)
from build_runner import run_build  # noqa: E402
from pipeline_paths import MissingPrerequisiteError, require_asset_inputs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a generated ASAM OpenMATERIAL human armature from classifier outputs."
    )
    parser.add_argument(
        "--asset-dir",
        help="Optional path to one asset folder under output/<asset> (or wherever OPEN_JAYWALKER_OUTPUT_ROOT points).",
    )
    return parser


def _extract_script_argv(argv):
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else _extract_script_argv(sys.argv))

    asset_dir = Path(args.asset_dir).resolve() if args.asset_dir else resolve_default_asset_dir(bpy.data.filepath, SCRIPT_DIR)
    try:
        require_asset_inputs(
            asset_dir, ["classifier_report.json", "build_plan.json"], "phase 3 classifier"
        )
    except MissingPrerequisiteError as err:
        print("error: {0}".format(err), file=sys.stderr)
        return 1

    print("\n" + "=" * 60)
    print("STARTING ASAM HUMAN BUILDER")
    print("=" * 60 + "\n")

    run_build(asset_dir, bpy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
else:
    main()

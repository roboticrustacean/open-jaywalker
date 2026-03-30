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

importlib.reload(builder)
importlib.reload(blender_builder)

from builder import build_armature_spec_from_asset_dir, print_builder_summary, resolve_default_asset_dir, write_builder_report  # noqa: E402
from blender_builder import build_armature_in_blender  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a generated ASAM OpenMATERIAL human armature from classifier outputs."
    )
    parser.add_argument(
        "--asset-dir",
        help="Optional path to one asset folder under src/armature_inspector/output/<asset>.",
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
    if not asset_dir.exists():
        raise FileNotFoundError(
            "Builder input folder does not exist: {0}. Run src/pipeline/main.py or src/phase3_classifier/main.py first.".format(
                asset_dir
            )
        )

    print("\n" + "=" * 60)
    print("STARTING ASAM HUMAN BUILDER")
    print("=" * 60 + "\n")

    _, _, build_spec = build_armature_spec_from_asset_dir(asset_dir)
    execution_result = build_armature_in_blender(build_spec, bpy)
    builder_report, report_path = write_builder_report(asset_dir, build_spec, execution_result)
    print_builder_summary(builder_report, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
else:
    main()

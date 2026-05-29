"""
CLI entrypoint for the offline Phase 3 classifier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase3_classifier.classifier import print_console_summary, write_asset_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify Phase 2 armature inspector outputs into ASAM core skeleton targets."
    )
    parser.add_argument(
        "--asset-dir",
        required=True,
        help="Path to one asset folder under output/<asset> (or wherever OPEN_JAYWALKER_OUTPUT_ROOT points).",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    asset_dir = Path(args.asset_dir).resolve()
    classifier_report, build_plan, report_path, plan_path = write_asset_report(asset_dir)
    print_console_summary(classifier_report, build_plan, report_path, plan_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

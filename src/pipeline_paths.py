"""
Shared resolution for the pipeline's output directory.

Inspector, classifier, and builder all write per-asset JSON exports to a
single output root. By default that root is `<repo>/output/` so the path
is project-relative and obvious. The OPEN_JAYWALKER_OUTPUT_ROOT environment
variable overrides the default; runbook scripts and tests use the override
to redirect pipeline writes to a tempdir without touching the canonical
location.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence


ENV_VAR = "OPEN_JAYWALKER_OUTPUT_ROOT"

# Project root is two parents up from this file: src/pipeline_paths.py -> src/ -> <repo>/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "output"


def resolve_output_root() -> Path:
    """
    Return the directory under which the pipeline stages write per-asset
    outputs. Honors the OPEN_JAYWALKER_OUTPUT_ROOT environment variable when
    set (with `~` expansion); otherwise defaults to `<repo>/output/`.
    """
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_OUTPUT_ROOT.resolve()


def resolve_asset_dir(asset_name: str) -> Path:
    """
    Return the per-asset directory under the output root, creating it (and
    any missing parents) if it doesn't already exist.
    """
    asset_dir = resolve_output_root() / asset_name
    asset_dir.mkdir(parents=True, exist_ok=True)
    return asset_dir


class MissingPrerequisiteError(Exception):
    """A pipeline stage was run without the prerequisite outputs of the prior stage.

    Raised by the prerequisite checks below so CLI entrypoints can catch it and emit a
    single-line error + remediation hint (exit code 1) instead of a Python traceback.
    """


def _remediation(asset_dir: Path, produced_by: str) -> str:
    return "Run the {0} first (e.g. with --asset-dir {1}).".format(produced_by, asset_dir)


def require_asset_inputs(
    asset_dir, required_filenames: Sequence[str], produced_by: str
) -> Path:
    """Resolve `asset_dir` and assert each required file is present.

    Raises MissingPrerequisiteError with a one-line message + remediation hint when the
    directory or any required file is absent; otherwise returns the resolved directory.
    Used by the builder (classifier_report.json + build_plan.json).
    """
    asset_dir = Path(asset_dir).resolve()
    if not asset_dir.is_dir():
        raise MissingPrerequisiteError(
            "Asset directory not found: {0}. {1}".format(
                asset_dir, _remediation(asset_dir, produced_by)
            )
        )
    missing = [name for name in required_filenames if not (asset_dir / name).exists()]
    if missing:
        raise MissingPrerequisiteError(
            "Missing {0} in {1}. {2}".format(
                ", ".join(missing), asset_dir, _remediation(asset_dir, produced_by)
            )
        )
    return asset_dir


def require_inspector_exports(asset_dir) -> Path:
    """Classifier prerequisite: `asset_dir` exists and holds >=1 inspector export.

    Inspector exports are named per armature (`<armature>_all.json` /
    `<armature>_filtered.json`), so this matches by glob rather than a fixed filename.
    Raises MissingPrerequisiteError (one-line + remediation hint) otherwise.
    """
    asset_dir = Path(asset_dir).resolve()
    if not asset_dir.is_dir():
        raise MissingPrerequisiteError(
            "Asset directory not found: {0}. {1}".format(
                asset_dir, _remediation(asset_dir, "armature inspector")
            )
        )
    if not (list(asset_dir.glob("*_all.json")) or list(asset_dir.glob("*_filtered.json"))):
        raise MissingPrerequisiteError(
            "No armature inspector exports (*_all.json) in {0}. {1}".format(
                asset_dir, _remediation(asset_dir, "armature inspector")
            )
        )
    return asset_dir

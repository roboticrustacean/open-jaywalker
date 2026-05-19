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

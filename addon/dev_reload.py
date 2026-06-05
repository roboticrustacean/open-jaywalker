"""Dev convenience: drop cached pipeline modules so the next import reflects live src/ edits.

bpy-free so it is unit-testable. Used by the run/build operators when the "Reload pipeline
code on each run" add-on preference is enabled, removing the need to restart Blender after
editing src/ under the dev-link junction (see tools/setup_dev_link.ps1).
"""

from __future__ import annotations

import sys

# Pipeline packages (and loose modules) bundled from src/. The add-on's own package
# (open_jaywalker and its submodules), bpy, and third-party modules are deliberately NOT
# purged — only the pipeline code the developer edits.
_PIPELINE_PREFIXES = ("armature_inspector", "phase3_classifier", "asam_human_builder", "pipeline")
_PIPELINE_MODULES = ("pipeline_paths",)


def purge_pipeline_modules(modules=None):
    """Delete cached pipeline modules from ``modules`` (defaults to ``sys.modules``).

    Returns the sorted list of purged names. The next ``import`` of these packages reloads
    them from ``sys.path`` — i.e. the live ``src/`` exposed by the dev-link junction.
    """
    modules = sys.modules if modules is None else modules
    purged = [
        name
        for name in list(modules)
        if name.split(".")[0] in _PIPELINE_PREFIXES or name in _PIPELINE_MODULES
    ]
    for name in purged:
        del modules[name]
    return sorted(purged)

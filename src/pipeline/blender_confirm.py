"""Blender-only confirm popup for the pre-build checkpoint.

When the build decision is undecided (no interactive stdin and no toggle set) and
a GUI window exists, the pipeline entry point offers this confirm dialog instead
of stopping. Blender's dialogs are asynchronous — they return control to the
event loop and run their callback on click — so the build runs here in the
operator's execute(), not inline in the pipeline flow.

This module imports bpy and therefore only loads inside Blender.
"""

from __future__ import annotations

from pathlib import Path

import bpy

from build_runner import run_build


class OPENJAYWALKER_OT_confirm_build(bpy.types.Operator):
    bl_idname = "openjaywalker.confirm_build"
    bl_label = "Build ASAM human from this plan?"
    bl_options = {"REGISTER"}

    asset_dir: bpy.props.StringProperty(name="Asset directory", default="")

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        run_build(Path(self.asset_dir), bpy)
        return {"FINISHED"}


def _ensure_registered() -> None:
    """(Re-)register the operator. Safe to call on every pipeline run."""
    try:
        bpy.utils.unregister_class(OPENJAYWALKER_OT_confirm_build)
    except Exception:  # pragma: no cover - not registered yet
        pass
    bpy.utils.register_class(OPENJAYWALKER_OT_confirm_build)


def offer_build_confirmation(asset_dir) -> None:
    """Show the confirm dialog for asset_dir; build on OK (via the operator)."""
    _ensure_registered()
    bpy.ops.openjaywalker.confirm_build("INVOKE_DEFAULT", asset_dir=str(asset_dir))

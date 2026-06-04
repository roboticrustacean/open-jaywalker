bl_info = {
    "name": "Open Jaywalker",
    "author": "Open Jaywalker",
    "version": (0, 1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Open Jaywalker",
    "description": "Convert humanoid rigs into ASAM OpenMATERIAL-compliant humans.",
    "category": "Rigging",
}

import os
import sys

# When installed, the bundled pipeline packages sit beside this file. Put our own
# directory on sys.path so their absolute imports (phase3_classifier.*, etc.) resolve.
# Resolve symlinks so local dev setup finds the actual repo path
_ADDON_DIR = os.path.dirname(os.path.realpath(__file__))
if _ADDON_DIR not in sys.path:
    sys.path.append(_ADDON_DIR)

# For local development via symlink, src/ is outside addon/
_SRC_DIR = os.path.join(os.path.dirname(_ADDON_DIR), "src")
if os.path.exists(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

import bpy

from . import state
from . import prefs
from . import operators
from . import ui
from . import addon_updater_ops

_CLASSES = (
    prefs.OJAddonPreferences,
    state.OJSettings,
    operators.OJ_OT_run_pipeline,
    operators.OJ_OT_build,
    operators.OJ_OT_open_output,
    operators.OJ_OT_clean,
    ui.OJ_PT_panel,
)


def register():
    updater = addon_updater_ops.updater
    updater.user = "roboticrustacean"
    updater.repo = "open-jaywalker"
    addon_updater_ops.register(bl_info)

    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.open_jaywalker = bpy.props.PointerProperty(type=state.OJSettings)


def unregister():
    addon_updater_ops.unregister()
    del bpy.types.Scene.open_jaywalker
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

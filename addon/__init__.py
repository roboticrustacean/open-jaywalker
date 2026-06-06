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

from bpy.app.handlers import persistent
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
    operators.OJ_OT_export_blend,
    operators.OJ_OT_export_gltf,
    operators.OJ_OT_open_output,
    operators.OJ_OT_clean,
    operators.OJ_OT_reset_pipeline,
    ui.OJ_PT_panel,
)


@persistent
def load_post_handler(dummy1, dummy2=None):
    # Reset all settings properties to clean start
    import bpy
    scene = getattr(bpy.context, "scene", None)
    if scene is not None:
        s = getattr(scene, "open_jaywalker", None)
        if s is not None:
            s.has_plan = False
            s.built = False
            s.show_details = False
            s.asset_dir = ""
            s.recommended_armature = ""
            s.mapped = 0
            s.total = 28
            s.missing_csv = ""
            s.missing_by_target_csv = ""
            s.review_flags_csv = ""
            s.character_ids_csv = ""
            s.is_crowd = False
            s.character_count = 0
            s.build_succeeded = 0
            s.build_failed = 0
            s.failed_characters_csv = ""
            s.synthesized_bones_csv = ""
            s.synthesized_bones_by_character_csv = ""
            s.show_failed_details = False
            s.show_inert_details = False


def register():
    updater = addon_updater_ops.updater
    updater.user = "roboticrustacean"
    updater.repo = "open-jaywalker"
    addon_updater_ops.register(bl_info)

    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.open_jaywalker = bpy.props.PointerProperty(type=state.OJSettings)

    if load_post_handler not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(load_post_handler)


def unregister():
    addon_updater_ops.unregister()
    del bpy.types.Scene.open_jaywalker

    if load_post_handler in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(load_post_handler)

    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

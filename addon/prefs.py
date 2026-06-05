"""Add-on preferences: override the pipeline output root."""

import bpy
from . import addon_updater_ops

@addon_updater_ops.make_annotations
class OJAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    output_dir: bpy.props.StringProperty(
        name="Output directory",
        description="Override the pipeline output root (sets OPEN_JAYWALKER_OUTPUT_ROOT). Leave blank for the default.",
        subtype='DIR_PATH',
        default="",
    )

    dev_reload: bpy.props.BoolProperty(
        name="Reload pipeline code on each run (dev)",
        description=(
            "Developer option: before Run/Build, drop cached pipeline modules so live edits "
            "to src/ (via the dev-link junction) take effect without restarting Blender. "
            "Leave OFF for a normal zip install."
        ),
        default=False,
    )

    # addon updater preferences
    auto_check_update: bpy.props.BoolProperty(
        name="Auto-check for Update",
        description="If enabled, auto-check for updates using an interval",
        default=False)
        
    updater_interval_months: bpy.props.IntProperty(
        name='Months',
        description="Number of months between checking for updates",
        default=0,
        min=0)
        
    updater_interval_days: bpy.props.IntProperty(
        name='Days',
        description="Number of days between checking for updates",
        default=7,
        min=0,
        max=31)
        
    updater_interval_hours: bpy.props.IntProperty(
        name='Hours',
        description="Number of hours between checking for updates",
        default=0,
        min=0,
        max=23)
        
    updater_interval_minutes: bpy.props.IntProperty(
        name='Minutes',
        description="Number of minutes between checking for updates",
        default=0,
        min=0,
        max=59)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "output_dir")
        layout.label(
            text="Blank = default <repo>/output (or an existing OPEN_JAYWALKER_OUTPUT_ROOT).",
            icon='INFO',
        )
        layout.prop(self, "dev_reload")
        
        addon_updater_ops.update_settings_ui(self, context)

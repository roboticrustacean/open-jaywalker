"""Add-on preferences: override the pipeline output root."""

import bpy


class OJAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    output_dir: bpy.props.StringProperty(
        name="Output directory",
        description="Override the pipeline output root (sets OPEN_JAYWALKER_OUTPUT_ROOT). Leave blank for the default.",
        subtype='DIR_PATH',
        default="",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "output_dir")
        layout.label(
            text="Blank = default <repo>/output (or an existing OPEN_JAYWALKER_OUTPUT_ROOT).",
            icon='INFO',
        )

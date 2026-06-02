"""Open Jaywalker N-panel: Run pipeline -> plan summary -> Build."""

import bpy


class OJ_PT_panel(bpy.types.Panel):
    bl_label = "Open Jaywalker"
    bl_idname = "OJ_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Open Jaywalker"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.open_jaywalker

        layout.operator("open_jaywalker.run_pipeline", icon='PLAY')

        if not settings.has_plan:
            layout.label(text="Run the pipeline to generate a plan.")
            return

        box = layout.box()
        box.label(text="Plan ready", icon='CHECKMARK')
        box.label(text="Primary: {0}".format(settings.recommended_armature))
        box.label(text="Mapped: {0}/{1}".format(settings.mapped, settings.total))
        if settings.is_crowd:
            box.label(text="Crowd: {0} characters".format(settings.character_count))
        if settings.missing_csv:
            box.label(text="Missing targets:")
            for name in settings.missing_csv.split(", "):
                box.label(text="   - {0}".format(name))
        layout.operator("open_jaywalker.build", icon='MOD_ARMATURE')

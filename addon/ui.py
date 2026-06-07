"""Open Jaywalker N-panel: status, plan summary (collapsible), build, output."""

import bpy


def _draw_inert_bone_details(layout, s):
    """Draw the collapsible inert bone warning at the bottom of the panel.

    Placed after export/output sections so expanding it never displaces controls.
    """
    if s.is_crowd and s.synthesized_bones_by_character_csv:
        layout.separator()
        warn_box = layout.box()
        warn_box.prop(
            s, "show_inert_details",
            text="Inert Bone Details",
            icon=('TRIA_DOWN' if s.show_inert_details else 'TRIA_RIGHT'),
            emboss=False,
        )
        if s.show_inert_details:
            inner = warn_box.box()
            inner.alert = True
            inner.label(text="Synthesized Inert Bones (unbound):", icon='WARNING')
            for line in s.synthesized_bones_by_character_csv.split(" | "):
                if line:
                    inner.label(text="   {0}".format(line))
    elif not s.is_crowd and s.synthesized_bones_csv:
        layout.separator()
        warn_box = layout.box()
        warn_box.prop(
            s, "show_inert_details",
            text="Inert Bone Details",
            icon=('TRIA_DOWN' if s.show_inert_details else 'TRIA_RIGHT'),
            emboss=False,
        )
        if s.show_inert_details:
            inner = warn_box.box()
            inner.alert = True
            inner.label(text="Synthesized Inert Bones (unbound):", icon='WARNING')
            for name in s.synthesized_bones_csv.split(", "):
                if name:
                    inner.label(text="   - {0}".format(name))


class OJ_PT_panel(bpy.types.Panel):
    bl_label = "Open Jaywalker"
    bl_idname = "OJ_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Open Jaywalker"

    def draw(self, context):
        layout = self.layout
        s = context.scene.open_jaywalker

        if s.built:
            status, icon = "Built", 'CHECKMARK'
        elif s.has_plan:
            status, icon = "Plan ready", 'INFO'
        else:
            status, icon = "Idle", 'RADIOBUT_OFF'
        layout.label(text="Status: {0}".format(status), icon=icon)

        # Per-run output/export destination (overrides the add-on preferences).
        dest_box = layout.box()
        dest_box.label(text="Output folder", icon='FILE_FOLDER')
        dest_box.prop(s, "output_root", text="")
        dest_box.prop(s, "export_root", text="Exports")

        layout.operator("open_jaywalker.run_pipeline", icon='PLAY')

        if s.has_plan:
            layout.separator()
            box = layout.box()
            box.label(text="Plan", icon='PRESET')
            col = box.column(align=True)
            col.label(text="Primary rig: {0}".format(s.recommended_armature))
            mapped_row = col.row()
            mapped_row.alert = s.mapped < s.total
            mapped_row.label(
                text="Mapped: {0}/{1}".format(s.mapped, s.total),
                icon=('ERROR' if s.mapped < s.total else 'CHECKMARK'),
            )
            if s.is_crowd:
                col.label(text="Crowd: {0} characters".format(s.character_count))

            box.prop(
                s, "show_details",
                text="Details",
                icon=('TRIA_DOWN' if s.show_details else 'TRIA_RIGHT'),
                emboss=False,
            )
            if s.show_details:
                det = box.box()
                if s.is_crowd and s.missing_by_target_csv:
                    det.label(text="Missing across {0} characters:".format(s.character_count))
                    for entry in s.missing_by_target_csv.split(", "):
                        det.label(text="   - {0}".format(entry))
                elif not s.is_crowd and s.missing_csv:
                    det.label(text="Missing targets:")
                    for name in s.missing_csv.split(", "):
                        det.label(text="   - {0}".format(name))
                if s.review_flags_csv:
                    det.label(text="Review flags:")
                    for flag in s.review_flags_csv.split(", "):
                        det.label(text="   - {0}".format(flag))
                if s.is_crowd and s.character_ids_csv:
                    det.label(text="Characters:")
                    for cid in s.character_ids_csv.split(", "):
                        det.label(text="   - {0}".format(cid))
                if not s.missing_csv and not s.missing_by_target_csv and not s.review_flags_csv:
                    det.label(
                        text="All {0} core targets mapped; no review flags.".format(s.total),
                        icon='CHECKMARK',
                    )

            layout.prop(s, "export_blend")
            layout.prop(s, "export_gltf")
            if s.is_crowd:
                layout.prop(s, "per_character_export")
            build_row = layout.row()
            build_row.enabled = s.has_plan and not s.built
            build_row.operator("open_jaywalker.build", icon='MOD_ARMATURE')

        if s.built:
            layout.separator()
            res_box = layout.box()
            res_box.label(text="Build Results", icon='INFO')
            res_col = res_box.column(align=True)
            res_col.prop(s, "show_generated_armature")
            res_col.prop(s, "show_source")
            if s.is_crowd:
                res_col.label(text="Succeeded: {0}".format(s.build_succeeded), icon='CHECKMARK')
                if s.build_failed > 0:
                    res_col.label(text="Failed: {0}".format(s.build_failed), icon='ERROR')
                    res_box.prop(
                        s, "show_failed_details",
                        text="Failed Details",
                        icon=('TRIA_DOWN' if s.show_failed_details else 'TRIA_RIGHT'),
                        emboss=False,
                    )
                    if s.show_failed_details:
                        det = res_box.box()
                        det.label(text="Failed characters:")
                        for cid in s.failed_characters_csv.split(", "):
                            if cid:
                                det.label(text="   - {0}".format(cid))
            else:
                if s.build_succeeded > 0:
                    res_col.label(text="Successfully built character.", icon='CHECKMARK')
                else:
                    res_col.label(text="Build failed.", icon='ERROR')

            layout.separator()
            export_box = layout.box()
            export_box.label(text="Export", icon='EXPORT')
            if s.is_crowd:
                export_box.prop(s, "per_character_export")
            row = export_box.row(align=True)
            row.operator("open_jaywalker.export_blend", icon='FILE_BLEND', text=".blend")
            row.operator("open_jaywalker.export_gltf", icon='FILE_3D', text=".glb")

        if s.asset_dir:
            layout.separator()
            out_box = layout.box()
            out_box.label(text="Reports:", icon='FILE_FOLDER')
            out_box.label(text=s.asset_dir)
            out_box.operator("open_jaywalker.open_output", icon='FILEBROWSER', text="Open reports folder")
            if s.export_dir:
                out_box.label(text="Exports:", icon='EXPORT')
                out_box.label(text=s.export_dir)
                out_box.operator("open_jaywalker.open_exports", icon='FILEBROWSER', text="Open exports folder")
        elif not s.has_plan:
            layout.label(text="Run the pipeline to generate a plan.")

        if s.built:
            _draw_inert_bone_details(layout, s)

        layout.separator()
        layout.operator("open_jaywalker.reset_pipeline", icon='LOOP_BACK')
        layout.operator("open_jaywalker.clean", icon='TRASH')

"""Open Jaywalker operators: run the pipeline, build, open output folder."""

import os
from pathlib import Path

import bpy

# Pipeline entry points are imported locally inside each operator's execute() so the
# optional "dev_reload" purge can take effect per run; only the Clean operator needs a
# module-level import.
from asam_human_builder.blender_builder import purge_previous_generated_artifacts


def _addon_prefs(context):
    return context.preferences.addons[__package__].preferences


class OJ_OT_run_pipeline(bpy.types.Operator):
    bl_idname = "open_jaywalker.run_pipeline"
    bl_label = "Run pipeline"
    bl_description = "Inspect the scene, classify the rig, and prepare a build plan"

    def execute(self, context):
        settings = context.scene.open_jaywalker
        settings.has_plan = False
        settings.built = False

        prefs = _addon_prefs(context)
        if prefs.output_dir:
            os.environ["OPEN_JAYWALKER_OUTPUT_ROOT"] = bpy.path.abspath(prefs.output_dir)

        if getattr(prefs, "dev_reload", False):
            from . import dev_reload
            dev_reload.purge_pipeline_modules()
        # Import fresh so a dev_reload purge above takes effect this run (otherwise these
        # resolve to the already-cached modules, identical to the module-level imports).
        from armature_inspector.inspector import inspect_scene
        from phase3_classifier.classifier import write_asset_report
        from pipeline.plan_summary import summarize_plan
        from asam_human_builder.blender_builder import purge_previous_generated_artifacts

        purge_previous_generated_artifacts(bpy)

        result = inspect_scene()
        if not result or result.get("diagnostics_ran") or not result.get("exported_files"):
            self.report({'WARNING'}, "No armature exports produced; nothing to classify.")
            return {'FINISHED'}

        output_dir = Path(result["output_dir"]).resolve()
        classifier_report, build_plan, _report_path, _plan_path = write_asset_report(output_dir)
        summary = summarize_plan(classifier_report, build_plan)

        settings.asset_dir = str(output_dir)
        settings.recommended_armature = summary["recommended_armature"]
        settings.mapped = summary["mapped"]
        settings.total = summary["total"]
        settings.missing_csv = ", ".join(summary["missing_targets"])
        settings.missing_by_target_csv = ", ".join(
            "{0} x{1}".format(entry["target"], entry["count"]) for entry in summary["missing_by_target"]
        )
        settings.review_flags_csv = ", ".join(summary["review_flags"])
        settings.character_ids_csv = ", ".join(summary["character_ids"])
        settings.is_crowd = summary["is_crowd"]
        settings.character_count = summary["character_count"]
        settings.has_plan = True

        self.report(
            {'INFO'},
            "Plan ready: {0}, {1}/{2} mapped".format(
                summary["recommended_armature"], summary["mapped"], summary["total"]
            ),
        )
        return {'FINISHED'}


class OJ_OT_build(bpy.types.Operator):
    bl_idname = "open_jaywalker.build"
    bl_label = "Build"
    bl_description = "Build the ASAM human(s) from the prepared plan"

    @classmethod
    def poll(cls, context):
        s = context.scene.open_jaywalker
        return bool(s.has_plan) and not s.built

    def execute(self, context):
        settings = context.scene.open_jaywalker
        prefs = _addon_prefs(context)
        if getattr(prefs, "dev_reload", False):
            from . import dev_reload
            dev_reload.purge_pipeline_modules()
        from asam_human_builder.build_runner import run_build
        from asam_human_builder.builder import success_message
        asset_dir = Path(settings.asset_dir)
        context.window.cursor_set('WAIT')
        try:
            report = run_build(asset_dir, bpy, packaging_mode=settings.packaging_mode)
        except Exception as exc:  # surface as an operator error, never crash
            self.report({'ERROR'}, "Build failed: {0}".format(exc))
            return {'CANCELLED'}
        finally:
            context.window.cursor_set('DEFAULT')
        report_path = asset_dir / "builder_report.json"
        self.report({'INFO'}, success_message(report, report_path))
        settings.built = True

        if "characters" in report:
            settings.build_succeeded = len(report.get("characters", []))
            failed = report.get("failed_characters", [])
            settings.build_failed = len(failed)
            settings.failed_characters_csv = ", ".join(f.get("character_id", "") for f in failed)
        else:
            settings.build_succeeded = 1 if report.get("built_core_targets") else 0
            settings.build_failed = 0
            settings.failed_characters_csv = ""

        return {'FINISHED'}


class OJ_OT_open_output(bpy.types.Operator):
    bl_idname = "open_jaywalker.open_output"
    bl_label = "Open output folder"
    bl_description = "Open the asset output folder in the system file browser"

    @classmethod
    def poll(cls, context):
        return bool(context.scene.open_jaywalker.asset_dir)

    def execute(self, context):
        bpy.ops.wm.path_open(filepath=context.scene.open_jaywalker.asset_dir)
        return {'FINISHED'}


class OJ_OT_clean(bpy.types.Operator):
    bl_idname = "open_jaywalker.clean"
    bl_label = "Clean generated rigs"
    bl_description = "Remove all previously-generated ASAM rigs/collections from the scene (source is left untouched)"

    def execute(self, context):
        removed = purge_previous_generated_artifacts(bpy)
        context.scene.open_jaywalker.built = False
        self.report({'INFO'}, "Removed {0} generated object(s)/collection(s).".format(removed))
        return {'FINISHED'}

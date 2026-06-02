"""Open Jaywalker operators: run the pipeline, then build the ASAM human(s)."""

from pathlib import Path

import bpy

from armature_inspector.inspector import inspect_scene
from phase3_classifier.classifier import write_asset_report
from pipeline.plan_summary import summarize_plan
from asam_human_builder.blender_builder import purge_previous_generated_artifacts
from asam_human_builder.build_runner import run_build
from asam_human_builder.builder import success_message


class OJ_OT_run_pipeline(bpy.types.Operator):
    bl_idname = "open_jaywalker.run_pipeline"
    bl_label = "Run pipeline"
    bl_description = "Inspect the scene, classify the rig, and prepare a build plan"

    def execute(self, context):
        settings = context.scene.open_jaywalker
        settings.has_plan = False

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
        return bool(context.scene.open_jaywalker.has_plan)

    def execute(self, context):
        settings = context.scene.open_jaywalker
        asset_dir = Path(settings.asset_dir)
        try:
            report = run_build(asset_dir, bpy)
        except Exception as exc:  # surface as an operator error, never crash
            self.report({'ERROR'}, "Build failed: {0}".format(exc))
            return {'CANCELLED'}
        report_path = asset_dir / "builder_report.json"
        self.report({'INFO'}, success_message(report, report_path))
        settings.has_plan = False
        return {'FINISHED'}

"""Add-on state stored on the Scene (between Run pipeline and Build)."""

import bpy


def update_show_generated_armature(self, context):
    if context is None:
        return
    scene = getattr(context, "scene", None)
    if scene is None:
        return
    objects = getattr(scene, "objects", None)
    if objects is None:
        return

    show = getattr(self, "show_generated_armature", True)
    for obj in objects:
        if obj is None:
            continue
        if getattr(obj, "type", None) == "ARMATURE":
            is_generated = False
            if hasattr(obj, "get") and callable(obj.get):
                try:
                    is_generated = bool(obj.get("open_jaywalker_generated"))
                except Exception:
                    pass
            if not is_generated:
                is_generated = bool(getattr(obj, "open_jaywalker_generated", False))

            if is_generated:
                hide_set = getattr(obj, "hide_set", None)
                if callable(hide_set):
                    try:
                        hide_set(not show)
                    except Exception:
                        pass


class OJSettings(bpy.types.PropertyGroup):
    has_plan: bpy.props.BoolProperty(default=False)
    built: bpy.props.BoolProperty(default=False)
    show_details: bpy.props.BoolProperty(default=False)
    show_generated_armature: bpy.props.BoolProperty(
        name="Show Armature",
        description="Show or hide the generated ASAM armature(s) in the viewport",
        default=True,
        update=update_show_generated_armature,
    )
    asset_dir: bpy.props.StringProperty(default="")
    recommended_armature: bpy.props.StringProperty(default="")
    mapped: bpy.props.IntProperty(default=0)
    total: bpy.props.IntProperty(default=28)
    missing_csv: bpy.props.StringProperty(default="")
    missing_by_target_csv: bpy.props.StringProperty(default="")
    review_flags_csv: bpy.props.StringProperty(default="")
    character_ids_csv: bpy.props.StringProperty(default="")
    is_crowd: bpy.props.BoolProperty(default=False)
    character_count: bpy.props.IntProperty(default=0)
    build_succeeded: bpy.props.IntProperty(default=0)
    build_failed: bpy.props.IntProperty(default=0)
    failed_characters_csv: bpy.props.StringProperty(default="")
    synthesized_bones_csv: bpy.props.StringProperty(default="")
    synthesized_bones_by_character_csv: bpy.props.StringProperty(default="")
    export_blend: bpy.props.BoolProperty(
        default=False,
        name="Auto-export .blend",
        description="Automatically export the generated ASAM result to .blend after build"
    )
    export_gltf: bpy.props.BoolProperty(
        default=True,
        name="Auto-export .glb",
        description="Automatically export the generated ASAM result to .glb format after build"
    )
    per_character_export: bpy.props.BoolProperty(
        default=False,
        name="Per-character files",
        description="Export each crowd character as a separate file instead of one combined file"
    )

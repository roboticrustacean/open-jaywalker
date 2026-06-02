"""Add-on state stored on the Scene (between Run pipeline and Build)."""

import bpy


class OJSettings(bpy.types.PropertyGroup):
    has_plan: bpy.props.BoolProperty(default=False)
    built: bpy.props.BoolProperty(default=False)
    show_details: bpy.props.BoolProperty(default=False)
    asset_dir: bpy.props.StringProperty(default="")
    recommended_armature: bpy.props.StringProperty(default="")
    mapped: bpy.props.IntProperty(default=0)
    total: bpy.props.IntProperty(default=28)
    missing_csv: bpy.props.StringProperty(default="")
    review_flags_csv: bpy.props.StringProperty(default="")
    character_ids_csv: bpy.props.StringProperty(default="")
    is_crowd: bpy.props.BoolProperty(default=False)
    character_count: bpy.props.IntProperty(default=0)

"""Add-on state stored on the Scene (between Run pipeline and Build)."""

import bpy


def _iter_objects(context):
    """Yield non-None objects from the scene, guarding against missing attributes."""
    scene = getattr(context, "scene", None)
    if scene is None:
        return
    objects = getattr(scene, "objects", None)
    if objects is None:
        return
    for obj in objects:
        if obj is not None:
            yield obj


def _is_flagged(obj, prop_key):
    """Return True if *obj* carries the given custom property flag."""
    if hasattr(obj, "get") and callable(obj.get):
        try:
            if bool(obj.get(prop_key)):
                return True
        except Exception:
            pass
    return bool(getattr(obj, prop_key, False))


def _set_visibility(obj, show):
    hide_set = getattr(obj, "hide_set", None)
    if callable(hide_set):
        try:
            hide_set(not show)
        except Exception:
            pass


def update_show_generated_bones(self, context):
    """Show/hide generated ARMATURE and EMPTY objects."""
    if context is None:
        return
    show = getattr(self, "show_generated_bones", True)
    for obj in _iter_objects(context):
        if _is_flagged(obj, "open_jaywalker_generated") and obj.type in ("ARMATURE", "EMPTY"):
            _set_visibility(obj, show)


def update_show_generated_mesh(self, context):
    """Show/hide generated MESH objects."""
    if context is None:
        return
    show = getattr(self, "show_generated_mesh", True)
    for obj in _iter_objects(context):
        if _is_flagged(obj, "open_jaywalker_generated") and obj.type == "MESH":
            _set_visibility(obj, show)


def update_show_source_bones(self, context):
    """Show/hide source ARMATURE objects."""
    if context is None:
        return
    show = getattr(self, "show_source_bones", True)
    for obj in _iter_objects(context):
        if _is_flagged(obj, "open_jaywalker_source_hidden") and obj.type == "ARMATURE":
            _set_visibility(obj, show)


def update_show_source_mesh(self, context):
    """Show/hide source MESH objects."""
    if context is None:
        return
    show = getattr(self, "show_source_mesh", True)
    for obj in _iter_objects(context):
        if _is_flagged(obj, "open_jaywalker_source_hidden") and obj.type == "MESH":
            _set_visibility(obj, show)


class OJSettings(bpy.types.PropertyGroup):
    has_plan: bpy.props.BoolProperty(default=False)
    built: bpy.props.BoolProperty(default=False)
    show_details: bpy.props.BoolProperty(default=False)
    show_generated_bones: bpy.props.BoolProperty(
        name="Show generated bones",
        description="Show or hide the generated ASAM armature and empties in the viewport",
        default=True,
        update=update_show_generated_bones,
    )
    show_generated_mesh: bpy.props.BoolProperty(
        name="Show generated mesh",
        description="Show or hide the generated ASAM mesh in the viewport",
        default=True,
        update=update_show_generated_mesh,
    )
    show_source_bones: bpy.props.BoolProperty(
        name="Show source bones",
        description="Show or hide the original source armature in the viewport",
        default=True,
        update=update_show_source_bones,
    )
    show_source_mesh: bpy.props.BoolProperty(
        name="Show source mesh",
        description="Show or hide the original source mesh in the viewport",
        default=True,
        update=update_show_source_mesh,
    )
    asset_dir: bpy.props.StringProperty(default="")
    export_dir: bpy.props.StringProperty(
        name="Exports",
        description=(
            "Where .blend/.glb exports are written. Pre-filled with the resolved "
            "default after Run pipeline; edit it before building/exporting to "
            "redirect this run's exports."
        ),
        subtype='DIR_PATH',
        default="",
    )
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
    show_failed_details: bpy.props.BoolProperty(default=False)
    show_inert_details: bpy.props.BoolProperty(default=False)

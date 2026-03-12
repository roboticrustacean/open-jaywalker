"""
Armature Inspector: Helper functions for analyzing Blender armatures.

This module provides utilities to detect armature objects in a Blender scene
and traverse/print their bone hierarchies. It also includes diagnostics for
scenes where armatures may be structured as Empty/node hierarchies and not Armature objects (e.g.
ASAM OpenMATERIAL 3D assets).
"""

import bpy


# region ARMATURE DETECTION

def get_armature_objects():
    """
    Find all armature objects across all Blender data (not just active scene).

    Returns:
        list: bpy.types.Object instances where type == 'ARMATURE'
    """
    return [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']

# endregion

# region HIERARCHY PRINTING

def print_armature_hierarchy(armature_obj):
    """
    Print the bone hierarchy of an armature object as a structured tree.

    Args:
        armature_obj: A Blender armature object (bpy.types.Object with type 'ARMATURE')
    """
    arm_data = armature_obj.data
    root_bones = [b for b in arm_data.bones if b.parent is None]

    print(f"Armature: {armature_obj.name}")
    print(f"  Total bones: {len(arm_data.bones)}")
    print("  Hierarchy:")

    for root in root_bones:
        _print_bone_recursive(root, indent_level=2)


def _print_bone_recursive(bone, indent_level):
    """
    Recursively print a bone and its children with indentation.

    Args:
        bone: A Blender bone (bpy.types.Bone)
        indent_level: Current indentation level for tree visualization
    """
    indent = "  " * indent_level
    print(f"{indent}- {bone.name}")
    for child in bone.children:
        _print_bone_recursive(child, indent_level + 1)


def print_object_hierarchy(obj, indent_level=0):
    """
    Recursively print a Blender object and its children (parent-child hierarchy).
    Works for any object type (Empty, Mesh, Armature, etc.).

    Args:
        obj: A Blender object (bpy.types.Object)
        indent_level: Current indentation level for tree visualization
    """
    indent = "  " * indent_level
    print(f"{indent}- {obj.name}  [type={obj.type}]")
    for child in obj.children:
        print_object_hierarchy(child, indent_level + 1)

# endregion

# region SCENE SUMMARY

def get_object_type_counts():
    """
    Count objects in bpy.data.objects grouped by type.

    Returns:
        dict: Mapping of object type string to count
    """
    type_counts = {}
    for obj in bpy.data.objects:
        t = obj.type
        type_counts[t] = type_counts.get(t, 0) + 1
    return type_counts


def is_default_scene():
    """
    Check if the current scene appears to be Blender's default startup scene.

    Returns:
        bool: True if this looks like the default scene
    """
    if bpy.data.filepath:
        return False

    obj_names = {obj.name for obj in bpy.data.objects}
    default_objects = {"Camera", "Cube", "Light"}
    return default_objects.issubset(obj_names) and len(obj_names) <= 4


def print_scene_summary():
    """
    Prints a brief summary of the current scene/file being inspected.
    """
    if is_default_scene():
        print("!" * 60)
        print("WARNING: Running on Blender's DEFAULT startup scene!")
        print("This is probably not the file you intended to inspect.")
        print("")
        print("To inspect a specific file, use one of these workflows:")
        print("  1. VSCode: Run 'Blender: Start', then open your file in")
        print("     that Blender window, then run 'Blender: Run Script'")
        print("  2. Blender: Open your file, go to Scripting workspace,")
        print("     open this script in Text Editor, press Alt+P to run")
        print("!" * 60)
        print("")

    filepath = bpy.data.filepath or "(unsaved / default scene)"
    print(f"File: {filepath}")

    type_counts = get_object_type_counts()
    total_objects = sum(type_counts.values())
    armature_count = type_counts.get('ARMATURE', 0)
    armature_datablocks = len(bpy.data.armatures)

    print(f"Total objects: {total_objects}")
    if type_counts:
        type_summary = ", ".join(f"{t}:{c}" for t, c in sorted(type_counts.items()))
        print(f"By type: {type_summary}")
    print(f"Armature datablocks: {armature_datablocks}")

# endregion

# region SCENE DIAGNOSTICS

def dump_all_objects():
    """
    Print every object in bpy.data.objects with its type, parent, and
    collection membership. Useful for diagnosing scenes where the expected
    armature is missing.
    """
    print("ALL OBJECTS IN bpy.data.objects:")
    print("-" * 60)

    if not bpy.data.objects:
        print("  (no objects)")
        return

    type_counts = {}
    for obj in bpy.data.objects:
        t = obj.type
        type_counts[t] = type_counts.get(t, 0) + 1

        parent_name = obj.parent.name if obj.parent else "(none)"
        collections = ", ".join(c.name for c in obj.users_collection) or "(none)"
        print(f"  {obj.name:40s}  type={t:12s}  parent={parent_name:30s}  collections={collections}")

    print("-" * 60)
    print("Object type summary:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")


def dump_collections():
    """
    Print all collections in the .blend file and which scene(s) they belong to.
    """
    print("\nCOLLECTIONS IN FILE:")
    print("-" * 60)

    for scene in bpy.data.scenes:
        print(f"  Scene: {scene.name}")
        _print_collection_tree(scene.collection, indent_level=2)


def _print_collection_tree(collection, indent_level):
    """Recursively print a collection tree."""
    indent = "  " * indent_level
    obj_count = len(collection.objects)
    print(f"{indent}[Collection] {collection.name}  ({obj_count} objects)")
    for child in collection.children:
        _print_collection_tree(child, indent_level + 1)


def dump_armature_datablocks():
    """
    Print all Armature datablocks in bpy.data.armatures, regardless of whether
    any object references them.
    """
    print("\nARMATURE DATABLOCKS IN bpy.data.armatures:")
    print("-" * 60)

    if not bpy.data.armatures:
        print("  (none)")
        return

    for arm in bpy.data.armatures:
        bone_count = len(arm.bones)
        users = arm.users
        print(f"  {arm.name:40s}  bones={bone_count}  users={users}")

# endregion

# region MAIN EXECUTION

def inspect_scene():
    """
    Inspect the current Blender scene for armatures and print their hierarchies.
    Also prints object-level parent-child trees for root objects if no armatures
    are found (to handle OpenMATERIAL-style Empty/node hierarchies).
    """
    armatures = get_armature_objects()

    print("=" * 60)
    print("ARMATURE INSPECTOR REPORT")
    print("=" * 60)
    print_scene_summary()
    print("-" * 60)

    if not armatures:
        print("No armature objects found.")
        print("Running full scene diagnostics...\n")
        run_diagnostics()
        return

    print(f"Found {len(armatures)} armature(s).")
    print("-" * 60)

    for i, obj in enumerate(armatures, start=1):
        print(f"\n[{i}/{len(armatures)}]")
        print_armature_hierarchy(obj)

    print("\n" + "=" * 60)
    print("END OF REPORT")
    print("=" * 60)


def run_diagnostics():
    """
    Run a full diagnostic dump of the scene to understand why no armatures
    were detected. Prints all objects, collections, armature datablocks,
    and the object parent-child hierarchy.
    """
    dump_all_objects()
    dump_armature_datablocks()
    dump_collections()

    print("\nOBJECT PARENT-CHILD HIERARCHY (root objects):")
    print("-" * 60)
    root_objects = [obj for obj in bpy.data.objects if obj.parent is None]
    if not root_objects:
        print("  (no root objects)")
    else:
        for obj in root_objects:
            print_object_hierarchy(obj, indent_level=1)

    print("\n" + "=" * 60)
    print("END OF DIAGNOSTICS")
    print("=" * 60)

# endregion MAIN EXECUTION

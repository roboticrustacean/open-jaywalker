"""
Armature Inspector: Helper functions for analyzing Blender armatures.

This module provides utilities to detect armature objects in a Blender scene
and traverse/print their bone hierarchies. It also includes diagnostics for
scenes where armatures may be structured as Empty/node hierarchies and not Armature objects (e.g.
ASAM OpenMATERIAL 3D assets).
"""

import bpy
import json


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

# region KINEMATIC CHAIN DETECTION

CHAIN_PATTERNS = {
    "spine": {
        "keywords": ["spine", "pelvis", "chest", "torso", "neck", "head"],
        "prefixes": ["DEF-spine", "ORG-spine"],
    },
    "leg": {
        "keywords": ["thigh", "shin", "foot", "toe", "hip", "knee", "ankle", "leg"],
        "prefixes": ["DEF-thigh", "DEF-shin", "DEF-foot", "DEF-toe",
                     "ORG-thigh", "ORG-shin", "ORG-foot", "ORG-toe"],
        "sides": [".L", ".R"],
    },
    "arm": {
        "keywords": ["upper_arm", "forearm", "hand", "shoulder", "elbow", "wrist", "arm"],
        "prefixes": ["DEF-upper_arm", "DEF-forearm", "DEF-hand",
                     "ORG-upper_arm", "ORG-forearm", "ORG-hand",
                     "DEF-shoulder", "ORG-shoulder"],
        "sides": [".L", ".R"],
    },
}


def get_bone_role(bone_name):
    """
    Extract the role prefix from a bone name (DEF, ORG, MCH, VIS, etc.).
    
    Args:
        bone_name: Name of the bone
        
    Returns:
        str: The role prefix (e.g., "DEF", "ORG") or None if no known prefix
    """
    known_prefixes = ["DEF-", "ORG-", "MCH-", "VIS-"]
    for prefix in known_prefixes:
        if bone_name.startswith(prefix):
            return prefix[:-1]
    return None


def get_bone_side(bone_name):
    """
    Extract the side suffix from a bone name (.L, .R).
    
    Args:
        bone_name: Name of the bone
        
    Returns:
        str: ".L", ".R", or None for center/unsided bones
    """
    if bone_name.endswith(".L") or ".L." in bone_name:
        return ".L"
    if bone_name.endswith(".R") or ".R." in bone_name:
        return ".R"
    return None


def find_chain_root(bone, chain_type):
    """
    Given a bone, walk up the hierarchy to find the root of its kinematic chain.
    
    Args:
        bone: A Blender bone
        chain_type: One of "spine", "leg", "arm"
        
    Returns:
        The root bone of the chain, or the original bone if no parent matches
    """
    patterns = CHAIN_PATTERNS.get(chain_type, {})
    keywords = patterns.get("keywords", [])
    
    current = bone
    while current.parent:
        parent_name_lower = current.parent.name.lower()
        if any(kw in parent_name_lower for kw in keywords):
            current = current.parent
        else:
            break
    return current


def build_chain_from_bone(bone, chain_type, visited):
    """
    Build a linear chain starting from a bone, following single-child paths.
    
    Args:
        bone: Starting bone
        chain_type: One of "spine", "leg", "arm"
        visited: Set of already-visited bone names
        
    Returns:
        list: List of bone names forming the chain
    """
    patterns = CHAIN_PATTERNS.get(chain_type, {})
    keywords = patterns.get("keywords", [])
    
    chain = []
    current = bone
    
    while current and current.name not in visited:
        bone_name_lower = current.name.lower()
        if any(kw in bone_name_lower for kw in keywords):
            chain.append(current.name)
            visited.add(current.name)
            
            matching_children = []
            for child in current.children:
                child_name_lower = child.name.lower()
                if any(kw in child_name_lower for kw in keywords):
                    matching_children.append(child)
            
            if len(matching_children) == 1:
                current = matching_children[0]
            else:
                break
        else:
            break
    
    return chain


def detect_chains_by_prefix(bones, prefix, chain_type):
    """
    Detect chains that start with a specific prefix pattern.
    
    Args:
        bones: Collection of bones to search
        prefix: Prefix to match (e.g., "DEF-spine")
        chain_type: Type of chain for keyword matching
        
    Returns:
        list: List of chains, where each chain is a list of bone names
    """
    chains = []
    visited = set()
    
    matching_bones = [b for b in bones if b.name.startswith(prefix)]
    
    for bone in matching_bones:
        if bone.name in visited:
            continue
        
        root = find_chain_root(bone, chain_type)
        if root.name in visited:
            continue
            
        chain = build_chain_from_bone(root, chain_type, visited)
        if len(chain) >= 2:
            chains.append(chain)
    
    return chains


def detect_spine_chains(armature_data):
    """
    Detect spine chains in an armature.
    
    Args:
        armature_data: Blender armature data (bpy.types.Armature)
        
    Returns:
        list: List of detected spine chains
    """
    chains = []
    bones = armature_data.bones
    
    for prefix in CHAIN_PATTERNS["spine"]["prefixes"]:
        found = detect_chains_by_prefix(bones, prefix, "spine")
        chains.extend(found)
    
    return chains


def detect_limb_chains(armature_data, limb_type):
    """
    Detect limb chains (arm or leg) in an armature.
    
    Args:
        armature_data: Blender armature data (bpy.types.Armature)
        limb_type: "arm" or "leg"
        
    Returns:
        dict: Mapping of side (".L", ".R", or "center") to list of chains
    """
    chains_by_side = {".L": [], ".R": [], "center": []}
    bones = armature_data.bones
    patterns = CHAIN_PATTERNS[limb_type]
    
    for prefix in patterns["prefixes"]:
        for side in patterns.get("sides", []):
            side_prefix = prefix + side
            found = detect_chains_by_prefix(bones, side_prefix, limb_type)
            for chain in found:
                chains_by_side[side].append(chain)
    
    return chains_by_side


def detect_all_chains(armature_data):
    """
    Detect all kinematic chains in an armature.
    
    Args:
        armature_data: Blender armature data (bpy.types.Armature)
        
    Returns:
        dict: Dictionary with chain types as keys and detected chains as values
    """
    return {
        "spine": detect_spine_chains(armature_data),
        "leg": detect_limb_chains(armature_data, "leg"),
        "arm": detect_limb_chains(armature_data, "arm"),
    }


def format_chain(chain, max_display=5):
    """
    Format a chain for display, truncating if too long.
    
    Args:
        chain: List of bone names
        max_display: Maximum bones to show before truncating
        
    Returns:
        str: Formatted chain string
    """
    if len(chain) <= max_display:
        return " -> ".join(chain)
    else:
        first_part = chain[:2]
        last_part = chain[-2:]
        return f"{' -> '.join(first_part)} -> ... -> {' -> '.join(last_part)} ({len(chain)} bones)"


def print_detected_chains(armature_obj):
    """
    Detect and print kinematic chains found in an armature.
    
    Args:
        armature_obj: A Blender armature object
    """
    arm_data = armature_obj.data
    chains = detect_all_chains(arm_data)
    
    print(f"\n  Detected Kinematic Chains:")
    print("  " + "-" * 40)
    
    total_chains = 0
    
    spine_chains = chains["spine"]
    if spine_chains:
        print(f"    Spine ({len(spine_chains)} chain(s)):")
        for i, chain in enumerate(spine_chains, 1):
            print(f"      [{i}] {format_chain(chain)}")
            total_chains += 1
    
    leg_chains = chains["leg"]
    leg_count = sum(len(v) for v in leg_chains.values())
    if leg_count > 0:
        print(f"    Leg ({leg_count} chain(s)):")
        for side in [".L", ".R", "center"]:
            side_label = {".L": "Left", ".R": "Right", "center": "Center"}.get(side, side)
            for chain in leg_chains[side]:
                print(f"      [{side_label}] {format_chain(chain)}")
                total_chains += 1
    
    arm_chains = chains["arm"]
    arm_count = sum(len(v) for v in arm_chains.values())
    if arm_count > 0:
        print(f"    Arm ({arm_count} chain(s)):")
        for side in [".L", ".R", "center"]:
            side_label = {".L": "Left", ".R": "Right", "center": "Center"}.get(side, side)
            for chain in arm_chains[side]:
                print(f"      [{side_label}] {format_chain(chain)}")
                total_chains += 1
    
    if total_chains == 0:
        print("    (no standard chains detected)")
        print("    Note: Chain detection uses DEF-*/ORG-* naming conventions.")
        print("    Custom rigs may require different detection patterns.")
    
    print("  " + "-" * 40)
    print(f"  Total chains identified: {total_chains}")

# endregion

# region BONE GEOMETRY EXTRACTION

def extract_bone_geometry(bone):
    """
    Extract geometry and transform data from a single bone.
    
    Args:
        bone: A Blender bone (bpy.types.Bone)
        
    Returns:
        dict: Dictionary containing bone geometry data:
            - name: bone name
            - parent: parent bone name or None
            - head: head position in armature space (tuple)
            - tail: tail position in armature space (tuple)
            - length: bone length
            - matrix_local: 4x4 local transform matrix (list of lists)
    """
    return {
        "name": bone.name,
        "parent": bone.parent.name if bone.parent else None,
        "head": tuple(bone.head_local),
        "tail": tuple(bone.tail_local),
        "length": bone.length,
        "matrix_local": [list(row) for row in bone.matrix_local],
    }


def extract_armature_geometry(armature_obj, prefix_filter=None):
    """
    Extract geometry data for all bones in an armature.
    
    Args:
        armature_obj: A Blender armature object (bpy.types.Object with type 'ARMATURE')
        prefix_filter: Optional prefix to filter bones (e.g., "DEF-" for deformation bones only)
        
    Returns:
        list: List of bone geometry dictionaries
    """
    bones_data = []
    for bone in armature_obj.data.bones:
        if prefix_filter is None or bone.name.startswith(prefix_filter):
            bones_data.append(extract_bone_geometry(bone))
    return bones_data


def format_vector(vec, precision=4):
    """
    Format a 3D vector for display.
    
    Args:
        vec: A tuple or list of 3 floats
        precision: Number of decimal places
        
    Returns:
        str: Formatted vector string like "(0.0000, 1.2345, -0.5000)"
    """
    return f"({vec[0]:.{precision}f}, {vec[1]:.{precision}f}, {vec[2]:.{precision}f})"


def print_bone_geometry(armature_obj, prefix_filter="DEF-", max_bones=None):
    """
    Print bone geometry data for an armature.
    
    Args:
        armature_obj: A Blender armature object
        prefix_filter: Only show bones starting with this prefix. 
                      Use None to show all bones, "DEF-" for deformation bones only.
        max_bones: Maximum number of bones to print (None for all)
    """
    bones_data = extract_armature_geometry(armature_obj, prefix_filter)
    
    filter_label = f"{prefix_filter}* bones" if prefix_filter else "all bones"
    print(f"\n  Bone Geometry ({filter_label}):")
    print("  " + "-" * 50)
    
    if not bones_data:
        print(f"    (no bones matching filter '{prefix_filter}')")
        print("  " + "-" * 50)
        return
    
    displayed = 0
    for bone_data in bones_data:
        if max_bones is not None and displayed >= max_bones:
            remaining = len(bones_data) - displayed
            print(f"    ... and {remaining} more bones")
            break
        
        name = bone_data["name"]
        parent = bone_data["parent"] or "(root)"
        head = format_vector(bone_data["head"])
        tail = format_vector(bone_data["tail"])
        length = bone_data["length"]
        
        print(f"    {name}")
        print(f"      head: {head}  tail: {tail}")
        print(f"      length: {length:.4f}  parent: {parent}")
        
        displayed += 1
    
    print("  " + "-" * 50)
    print(f"  Total bones shown: {displayed} / {len(bones_data)}")


def _serialize_chains(chains_dict):
    """
    Convert chains dictionary to a JSON-serializable format.
    
    Args:
        chains_dict: Dictionary from detect_all_chains()
        
    Returns:
        dict: Serializable chains data
    """
    result = {}
    for chain_type, chains in chains_dict.items():
        if isinstance(chains, dict):
            result[chain_type] = {side: list(c) for side, c in chains.items()}
        else:
            result[chain_type] = [list(c) for c in chains]
    return result


def export_armature_data(armature_obj, filepath, prefix_filter=None):
    """
    Export armature data to a JSON file.
    
    This creates a structured data file containing bone geometry and 
    detected kinematic chains, suitable for use in later processing phases
    (e.g., OpenMATERIAL conversion).
    
    Args:
        armature_obj: A Blender armature object
        filepath: Path to the output JSON file
        prefix_filter: Optional prefix to filter bones (e.g., "DEF-")
    """
    arm_data = armature_obj.data
    
    data = {
        "armature_name": armature_obj.name,
        "bone_count": len(arm_data.bones),
        "filtered_prefix": prefix_filter,
        "bones": extract_armature_geometry(armature_obj, prefix_filter),
        "chains": _serialize_chains(detect_all_chains(arm_data)),
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported armature data to: {filepath}")

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
        print_detected_chains(obj)
        print_bone_geometry(obj, prefix_filter="DEF-", max_bones=20)

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

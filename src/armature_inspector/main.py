"""
Armature Inspector - Main entry point.

Detects armatures in the current Blender scene and prints structured bone
hierarchy trees.

=== WORKFLOW (with VSCode Blender Extension) ===

The extension launches a NEW Blender instance. To inspect a specific file:

    1. Run "Blender: Start" from VSCode command palette
       -> This launches Blender connected to VSCode
    2. In that Blender window, open your .blend file (File > Open)
    3. Back in VSCode, run "Blender: Run Script"
       -> The script runs in the connected Blender instance

The output includes a file summary so you can verify which file was inspected.
"""

import sys
import os
import importlib

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import bpy

import inspector
importlib.reload(inspector)
from inspector import inspect_scene


# region MAIN EXECUTION

def main():
    """Main entry point for the armature inspector."""
    print("\n" + "=" * 60)
    print("STARTING ARMATURE INSPECTOR")
    print("=" * 60 + "\n")

    inspect_scene()


if __name__ == "__main__":
    main()
else:
    main()

# endregion MAIN EXECUTION

## open-jaywalker
Rule-Based Pedestrian 3D Asset Pipeline for Traffic Simulations

### Blender Armature Inspector

The initial asset analysis tool lives under `src/armature_inspector` and inspects Blender scenes for armatures and bone hierarchies.

#### Requirements

- Blender (tested with 3.x)
- VSCode with a Blender integration extension (for example, `Jacques Lucke: Blender Development`) **or** Blender's built-in Text Editor

#### Usage with VSCode + Blender extension

1. In VSCode, run **Blender: Start** to launch a Blender instance connected to VSCode.
2. In the launched Blender window, open the `.blend` file you want to inspect (File → Open).
3. In VSCode, open `src/armature_inspector/main.py`.
4. Run **Blender: Run Script**.
5. Check Blender’s system console or the VSCode task output for the armature report and scene summary.

#### Usage directly in Blender

1. Open your target `.blend` file in Blender.
2. Switch to the **Scripting** workspace (or open a Text Editor).
3. Open `src/armature_inspector/main.py` in the Text Editor.
4. Press **Alt+P** (Run Script).
5. View the printed report in Blender’s system console.

The inspector will:

- Detect `ARMATURE` objects in the file
- Traverse each armature’s bone hierarchy
- Print a clear, structured tree of bones, including OpenMATERIAL 3D–style skeletons.

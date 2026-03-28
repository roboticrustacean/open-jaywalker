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
3. In VSCode, open `src/pipeline/main.py` for the full inspector + classifier workflow, or `src/armature_inspector/main.py` for inspector-only.
4. Run **Blender: Run Script**.
5. Check Blender’s system console or the VSCode task output for the armature report, exported JSON paths, and classifier summary.

#### Usage directly in Blender

1. Open your target `.blend` file in Blender.
2. Switch to the **Scripting** workspace (or open a Text Editor).
3. Open `src/pipeline/main.py` for the full inspector + classifier workflow, or `src/armature_inspector/main.py` for inspector-only.
4. Press **Alt+P** (Run Script).
5. View the printed report in Blender’s system console.

The combined Blender pipeline will:

- Detect `ARMATURE` objects in the file
- Traverse each armature’s bone hierarchy
- Export the same Phase 2 JSON files under `src/armature_inspector/output/<asset>`
- Run the Phase 3 classifier on that freshly exported folder
- Write `phase3_classification.json` into the same asset folder

The inspector-only entrypoint will:

- Detect `ARMATURE` objects in the file
- Traverse each armature’s bone hierarchy
- Print a clear, structured tree of bones, including OpenMATERIAL 3D–style skeletons.
- Export the same Phase 2 JSON files under `src/armature_inspector/output/<asset>`

### Phase 3 Offline Classifier

If you want to run the classifier separately in your `open-jaywalker` Conda environment after exporting JSON from Blender, run:

```powershell
conda activate open-jaywalker
python src/phase3_classifier/main.py --asset-dir src/armature_inspector/output/openmatexamplehuman
```

The classifier will:

- Load the exported Phase 2 JSON for each armature in the asset folder
- Prefer `DEF-` filtered outputs when available and use unfiltered data as fallback evidence
- Classify core ASAM human semantic targets into a structured report
- Recommend one primary armature for later ASAM conversion work
- Write `phase3_classification.json` back into the selected asset folder

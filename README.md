## open-jaywalker
Rule-Based Pedestrian 3D Asset Pipeline for Traffic Simulations

### Blender Armature Inspector

The initial asset analysis tool lives under `src/armature_inspector` and inspects Blender scenes for armatures and bone hierarchies.

#### Requirements

- Blender (tested with 5.1.0)
- VSCode with a Blender integration extension (for example, `Jacques Lucke: Blender Development`) **or** Blender's built-in Text Editor

#### Usage with VSCode + Blender extension

1. In VSCode, run **Blender: Start** to launch a Blender instance connected to VSCode.
2. In the launched Blender window, open the `.blend` file you want to inspect (File → Open).
3. In VSCode, open `src/pipeline/main.py` for the full inspector + classifier workflow, `src/armature_inspector/main.py` for inspector-only, or `src/asam_human_builder/main.py` to build a generated ASAM armature from an existing plan.
4. Run **Blender: Run Script**.
5. Check Blender’s system console or the VSCode task output for the armature report, exported JSON paths, and classifier summary.

#### Usage directly in Blender

1. Open your target `.blend` file in Blender.
2. Switch to the **Scripting** workspace (or open a Text Editor).
3. Open `src/pipeline/main.py` for the full inspector + classifier workflow, `src/armature_inspector/main.py` for inspector-only, or `src/asam_human_builder/main.py` to build a generated ASAM armature from an existing plan.
4. Press **Alt+P** (Run Script).
5. View the printed report in Blender’s system console.

The combined Blender pipeline will:

- Detect `ARMATURE` objects in the file
- Traverse each armature’s bone hierarchy
- Export the same Phase 2 JSON files under `src/armature_inspector/output/<asset>`
- Run the Phase 3 classifier on that freshly exported folder
- Write `classifier_report.json` and `build_plan.json` into the same asset folder

The inspector-only entrypoint will:

- Detect `ARMATURE` objects in the file
- Traverse each armature’s bone hierarchy
- Print a clear, structured tree of bones, including OpenMATERIAL 3D–style skeletons.
- Export the same Phase 2 JSON files under `src/armature_inspector/output/<asset>`

The builder entrypoint will:

- Load `classifier_report.json` and `build_plan.json` from `src/armature_inspector/output/<asset>`
- Validate that the recommended source armature exists in the currently open `.blend`
- Create or safely rebuild a generated collection named `ASAM_<AssetName>`
- Create `Grp_Root` and `Armature_<AssetName>` and build the supported ASAM core skeleton scope inside that new armature
- Leave the original source rig untouched so extra/control bones remain preserved on the source armature
- Write `builder_report.json` back into the same asset folder for traceability

#### Alternate Partial Executions

You can run the pipeline in four different ways depending on what you want to do:

1. Full Blender pipeline
   Open `src/pipeline/main.py` in Blender or via the VSCode Blender extension and run it.
   This performs analysis first, writes the Phase 2 JSON files, then immediately runs the Phase 3 classifier on that same asset folder.

2. Analysis only
   Open `src/armature_inspector/main.py` in Blender or via the VSCode Blender extension and run it.
   This only performs the armature inspection and writes the Phase 2 JSON files under `src/armature_inspector/output/<asset>`.

3. Classification only
   First make sure the asset already has Phase 2 JSON output under `src/armature_inspector/output/<asset>`.
   Then run the classifier separately from any Python environment that has this project's dependencies installed:

   ```powershell
   python src/phase3_classifier/main.py --asset-dir src/armature_inspector/output/<asset>
   ```

4. Builder only
   First make sure the asset already has fresh `classifier_report.json` and `build_plan.json` output under `src/armature_inspector/output/<asset>`.
   Then open the same `.blend` file in Blender and run `src/asam_human_builder/main.py`.
   This creates a separate generated ASAM armature instead of mutating the source rig in place.

This split workflow is useful when you want to:

- rerun only the classifier after changing classification rules
- inspect exported JSON files manually before classification
- keep Blender usage limited to the extraction/analysis step
- generate a new ASAM-compliant armature without modifying the original rig

### Phase 3 Offline Classifier

If you want to run the classifier separately after exporting JSON from Blender, run it from any Python environment that has this project's dependencies installed:

```powershell
python src/phase3_classifier/main.py --asset-dir src/armature_inspector/output/openmatexamplehuman
```

The classifier will:

- Load the exported Phase 2 JSON for each armature in the asset folder
- Prefer `DEF-` filtered outputs when available and use unfiltered data as fallback evidence
- Classify core ASAM human semantic targets into a structured report
- Recommend one primary armature for later ASAM conversion work
- Write `classifier_report.json` and `build_plan.json` back into the selected asset folder

### ASAM Human Builder

The Blender-side builder lives under `src/asam_human_builder` and consumes the saved classifier outputs to generate a separate ASAM-compliant core armature in the currently open `.blend`.

For the current v1 scope, it will:

- Read `recommended_primary_armature`, `semantic_mapping`, `root_resolution`, `placement_metadata`, and `proposed_asam_hierarchy`
- Create a new generated collection `ASAM_<AssetName>`
- Create `Grp_Root` and `Armature_<AssetName>`
- Build the supported ASAM core skeleton scope as new bones in that generated armature
- Reuse source-bone geometry where the classifier marked a recoverable mapping
- Create missing targets deterministically by mirroring, interpolating, or extrapolating from nearby mapped targets when needed
- Preserve non-ASAM extra bones by leaving them on the original source rig instead of copying them into the generated armature

Recommended workflow:

1. Run `src/pipeline/main.py` in Blender to refresh the Phase 2 and Phase 3 JSON outputs for the current `.blend`.
2. Run `src/asam_human_builder/main.py` in Blender to build or rebuild the generated ASAM armature from that saved plan.

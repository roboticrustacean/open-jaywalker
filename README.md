## open-jaywalker
Rule-Based Pedestrian 3D Asset Pipeline for Traffic Simulations

open-jaywalker takes arbitrary off-the-shelf humanoid rigs — any rigging
convention (Rigify, 3ds Max Biped, Mixamo, …), single-character or crowd — and
deterministically converts them into
[ASAM OpenMATERIAL](https://asam-ev.github.io/OpenMATERIAL-3D/asamopenmaterial/latest/specification/07_geometry/object-human/human-index.html)-compliant
humans: both the **rig** and the **skinned mesh**, correct and self-consistent.

The pipeline runs in three stages: a Blender **armature inspector** exports
per-asset JSON, a pure-Python **classifier** maps bones onto the 28 ASAM core
targets and writes a build plan, and a Blender **ASAM human builder** constructs
the compliant rig and rebinds the mesh from that plan.

### Documentation

Full design and reference docs live in the
[project wiki](https://github.com/roboticrustacean/open-jaywalker/wiki) —
see [Architecture](https://github.com/roboticrustacean/open-jaywalker/wiki/Architecture)
(data flow, the three reports, extraction schema, add-on/build workflow) and
[Skeleton Classification Rules](https://github.com/roboticrustacean/open-jaywalker/wiki/Skeleton-Classification-Rules)
(the 28 core targets, name-free structural inference, decision tags, crowd
decomposition).

To use the generated `.glb` assets in an OpenSCENARIO simulator such as
[esmini](https://github.com/eclipse/esmini) (which ingests OpenSceneGraph `.osgb`),
the repo ships an unsupported post-build conversion helper, `tools/glb_to_osgb.py`,
that wraps a local `osgconv`. Note that stock OpenSceneGraph builds cannot read `.glb`
directly (no glTF plugin), so the portable route is `.glb → .obj → .osgb`. The
[Using Outputs in esmini / OpenSCENARIO](https://github.com/roboticrustacean/open-jaywalker/wiki/Using-Outputs-in-esmini-and-OpenSCENARIO)
wiki page has the verified end-to-end recipe plus the scale and texture caveats.

### Blender Add-on (recommended)

The pipeline ships as an installable Blender add-on, **Open Jaywalker**, which is
the simplest way to run it interactively.

📺 **Video walkthrough:** https://youtu.be/kR923TMq4zc

#### Install

1. **Download the add-on zip.** Get `open_jaywalker-<version>.zip` from the
   [latest release](https://github.com/roboticrustacean/open-jaywalker/releases/latest).
   _(Alternatively, build it yourself from the repo root with
   `python tools/build_addon.py`, which writes `dist/open_jaywalker-<version>.zip`.)_
2. In Blender, open **Edit ▸ Preferences ▸ Add-ons**.
3. Click **Install from Disk…** (on older Blender, the **Install…** button), select
   the downloaded zip, and confirm.
4. Tick the checkbox next to **Open Jaywalker** to enable it.
5. Open your humanoid `.blend`, then in the 3D Viewport press `N` to open the
   sidebar and select the **Open Jaywalker** tab.

> **Install the built zip, not the repo `addon/` folder.** The `addon/` sources
> alone do not contain the bundled pipeline packages — those are assembled in only
> by `tools/build_addon.py` (and shipped in the released zip). Pointing Blender at
> the repo `addon/` directory yields a non-functional add-on.

#### Use it — the panel, button by button

The panel runs the pipeline top to bottom. A **Status** line at the top tracks
progress: *Idle → Plan ready → Built*.

1. **Run pipeline** — inspects the scene, classifies the skeleton against the 28
   ASAM core targets, and writes the plan. It deliberately stops here (a *build
   gate*) so you can review before anything is built.
2. **Plan** summary (appears after running):
   - **Primary rig** — the binding-decisive armature the classifier selected (the
     one that actually skins the mesh).
   - **Mapped: X/Y** — how many core targets were mapped directly (green check, or
     a red warning when some are missing and will be synthesized).
   - **Crowd: N characters** — for crowd assets, how many characters were
     decomposed.
   - **Details** (expandable) — missing targets, review flags (e.g.
     `multiple_roots`, `side_conflict`), and, for crowds, the per-character list.
3. **Export .blend / Export .glb** (and, for crowds, **Per-character export**) —
   toggles that export automatically right after a build.
4. **Build** — constructs the ASAM human(s): a `Grp_Root` container, the generated
   28-target armature, and the rebound mesh, leaving the source rig untouched. For
   crowds it builds one human per character (fan-out).
5. **Build Results** (after building):
   - **Generated: Bones / Mesh** and **Source: Bones / Mesh** — visibility toggles
     for before/after comparison and per-character isolation.
   - **Succeeded / Failed** counts for crowds, with an expandable **Failed
     Details** list.
6. **Export** — on-demand **.blend** / **.glb** buttons (plus **Per-character
   export** for crowds) to write out the generated result.
7. **Reports / Exports** — shows the output folder paths, with **Open reports
   folder** / **Open exports folder** buttons and an editable export directory.
8. **Reset** — clears the current plan/build state to start over.
9. **Clean** — removes the generated collection(s) from the scene.
10. **Inert Bone Details** (shown when applicable) — an expandable list of
    synthesized, unbound (inert) bones, surfaced transparently rather than hidden.

### Developer & Headless Workflows

#### Blender Armature Inspector

The initial asset analysis tool lives under `src/armature_inspector` and inspects Blender scenes for armatures and bone hierarchies.

##### Requirements

- Blender (tested with 5.1.0)
- VSCode with a Blender integration extension (for example, `Jacques Lucke: Blender Development`) **or** Blender's built-in Text Editor

##### Usage with VSCode + Blender extension

1. In VSCode, run **Blender: Start** to launch a Blender instance connected to VSCode.
2. In the launched Blender window, open the `.blend` file you want to inspect (File → Open).
3. In VSCode, open `src/pipeline/main.py` for the full inspector + classifier workflow, `src/armature_inspector/main.py` for inspector-only, or `src/asam_human_builder/main.py` to build a generated ASAM armature from an existing plan.
4. Run **Blender: Run Script**.
5. Check Blender’s system console or the VSCode task output for the armature report, exported JSON paths, and classifier summary.

##### Usage directly in Blender

1. Open your target `.blend` file in Blender.
2. Switch to the **Scripting** workspace (or open a Text Editor).
3. Open `src/pipeline/main.py` for the full inspector + classifier workflow, `src/armature_inspector/main.py` for inspector-only, or `src/asam_human_builder/main.py` to build a generated ASAM armature from an existing plan.
4. Press **Alt+P** (Run Script).
5. View the printed report in Blender’s system console.

The combined Blender pipeline will:

- Detect `ARMATURE` objects in the file
- Traverse each armature’s bone hierarchy
- Export the same Phase 2 JSON files under `output/<asset>`
- Run the Phase 3 classifier on that freshly exported folder
- Write `classifier_report.json` and `build_plan.json` into the same asset folder
- Then consult a build gate to decide whether to continue into the builder: a
  `[y/N]` prompt on an interactive terminal, otherwise the
  `OPEN_JAYWALKER_AUTO_BUILD` environment toggle or a `-- --build` / `-- --no-build`
  argument (default: stop after writing the plan and print how to build)

The inspector-only entrypoint will:

- Detect `ARMATURE` objects in the file
- Traverse each armature’s bone hierarchy
- Print a clear, structured tree of bones, including OpenMATERIAL 3D–style skeletons.
- Export the same Phase 2 JSON files under `output/<asset>`

The builder entrypoint will:

- Load `classifier_report.json` and `build_plan.json` from `output/<asset>`
- Validate that the recommended source armature exists in the currently open `.blend`
- Create or safely rebuild a generated collection named `ASAM_<AssetName>`
- Create `Grp_Root` and `Armature_<AssetName>` and build the 28 ASAM core skeleton targets inside that new armature
- Duplicate the meshes driven by the source armature, rebind them to the generated armature, and rename their vertex groups to the ASAM bone names so the mesh deforms correctly on the new rig
- For a crowd asset, build one ASAM human per character (fan-out)
- Leave the original source rig untouched so extra/control bones remain preserved on the source armature
- Write `builder_report.json` back into the same asset folder for traceability

##### Alternate Partial Executions

You can run the pipeline in four different ways depending on what you want to do:

1. Full Blender pipeline
   Open `src/pipeline/main.py` in Blender or via the VSCode Blender extension and run it.
   This performs analysis first, writes the Phase 2 JSON files, then immediately runs the Phase 3 classifier on that same asset folder.

2. Analysis only
   Open `src/armature_inspector/main.py` in Blender or via the VSCode Blender extension and run it.
   This only performs the armature inspection and writes the Phase 2 JSON files under `output/<asset>`.

3. Classification only
   First make sure the asset already has Phase 2 JSON output under `output/<asset>`.
   Then run the classifier separately from any Python environment that has this project's dependencies installed:

   ```powershell
   python src/phase3_classifier/main.py --asset-dir output/<asset>
   ```

4. Builder only
   First make sure the asset already has fresh `classifier_report.json` and `build_plan.json` output under `output/<asset>`.
   Then open the same `.blend` file in Blender and run `src/asam_human_builder/main.py`.
   This creates a separate generated ASAM armature instead of mutating the source rig in place.

This split workflow is useful when you want to:

- rerun only the classifier after changing classification rules
- inspect exported JSON files manually before classification
- keep Blender usage limited to the extraction/analysis step
- generate a new ASAM-compliant armature without modifying the original rig

Pipeline outputs land in `output/<asset>/` at the repo root. Set the
`OPEN_JAYWALKER_OUTPUT_ROOT` environment variable to redirect that root
elsewhere — handy when scripting or running automated tests that should
not touch the canonical location.

#### Phase 3 Offline Classifier

If you want to run the classifier separately after exporting JSON from Blender, run it from any Python environment that has this project's dependencies installed:

```powershell
python src/phase3_classifier/main.py --asset-dir output/openmatexamplehuman
```

The classifier will:

- Load the exported Phase 2 JSON for each armature in the asset folder
- Prefer `DEF-` filtered outputs when available and use unfiltered data as fallback evidence
- Classify core ASAM human semantic targets into a structured report
- Recommend one primary armature for later ASAM conversion work
- Write `classifier_report.json` and `build_plan.json` back into the selected asset folder

#### ASAM Human Builder

The Blender-side builder lives under `src/asam_human_builder` and consumes the saved classifier outputs to generate a separate ASAM-compliant core armature in the currently open `.blend`.

For the current v1 scope, it will:

- Read `recommended_primary_armature`, `semantic_mapping`, `root_resolutions`, `placement_metadata`, and `proposed_asam_hierarchy`
- Create a new generated collection `ASAM_<AssetName>`
- Anchor `Grp_Root` to the source frame's `bbox_ground_center` (per ASAM §7.3.3.3.2) and create `Armature_<AssetName>` underneath it with identity local transform
- Build the supported ASAM core skeleton scope as new bones in that generated armature, expressed in `Grp_Root`-local coordinates
- Reuse the source root bone (renamed to `Root`) when it passes structural compliance checks; planar and ground-Z offsets between the source root and `bbox_ground_center` are surfaced as advisories in `root_resolutions[0].advisories` and no longer block reuse
- When structural blockers fire, synthesize a fresh `Root` at `Grp_Root` local origin and preserve the source root as a sibling extra in the generated armature so its skin weights survive
- Reuse source-bone geometry where the classifier marked a recoverable mapping
- Create missing targets deterministically by mirroring, interpolating, or extrapolating from nearby mapped targets when needed
- Preserve non-ASAM extra bones by leaving them on the original source rig instead of copying them into the generated armature

Recommended workflow:

1. Run `src/pipeline/main.py` in Blender to refresh the Phase 2 and Phase 3 JSON outputs for the current `.blend`.
2. Run `src/asam_human_builder/main.py` in Blender to build or rebuild the generated ASAM armature from that saved plan.

# Manual runbook: Populate / 1000idles fixture (Issue #21)

This runbook validates the same success criterion as the scripted runbooks
under `tests/blender_runbook/runbook_mesh_deformation*.py` — that the
generated ASAM armature actually drives the duplicated mesh — but on a
heavier third-party fixture
(Blender Studio's "Populate" library / `1000idles.blend`). The fixture is too
large to commit, so this case is out-of-CI by design.

## What you need

- A `.blend` containing one character from the Blender Studio Populate library
  or the public `1000idles.blend` distribution. Any single Populate character
  with a single armature works.
- The character's primary armature must have at least: a hip/pelvis, a spine
  chain, both arms, both legs. (Populate characters all do.)
- Blender (the same Steam install you use for the rest of this project).
- VS Code with the Blender extension wired up.

## Procedure

1. **Open the fixture**: `File -> Open` the Populate character `.blend` in
   Blender.
2. **Note the armature name**: in the Outliner, identify the source armature
   object name (the rig). You'll cross-check it against
   `recommended_primary_armature` in `classifier_report.json` later.
3. **Run the pipeline**:
   - In VS Code, open `src/pipeline/main.py` and `Blender: Run Script`.
   - Then open `src/asam_human_builder/main.py` and `Blender: Run Script`.
4. **Locate generated objects**: in the Outliner, expand the new
   `ASAM_<AssetName>` collection. You should see:
   - `Grp_Root` (empty)
   - `Armature_<AssetName>` (the generated ASAM rig)
   - One or more `ASAM_<source-mesh-name>` mesh objects
5. **Capture rest-pose screenshot**: in the viewport, frame the generated mesh
   and take a screenshot. Save as
   `docs/testing/populate_<character>_rest.png`.
6. **Pose-test**: in `Pose Mode` on the GENERATED armature, rotate each of
   these bones by ~30° on a sensible axis (X for shoulder swing, Y for hip
   abduction, X for spine bend):
   - `Upper_Arm_Left`
   - `Upper_Leg_Right`
   - `Lower_Spine`
   Confirm that the duplicated mesh deforms — i.e. the limb/torso of the
   `ASAM_<mesh>` follows the bone, not the source rig.
7. **Capture posed screenshot**: save as
   `docs/testing/populate_<character>_posed.png`.
8. **Inspect reports**: open
   `output/<character>/builder_report.json` and verify:
   - `built_core_targets` contains all 28 ASAM bones.
   - `duplicated_meshes` is non-empty.
   - `mesh_warnings` is either empty or contains only the expected codes
     (`unmapped_vertex_groups:*` for non-deforming meshes).
9. **Record outcome**: append to the table below.

## Recorded runs

| Date | Character file | Result | Builder warnings | Notes |
| --- | --- | --- | --- | --- |
| _YYYY-MM-DD_ | _e.g. populate_male_01.blend_ | PASS / FAIL | _summary or count_ | _link to screenshots_ |

## Known caveats

- Populate characters use Rigify with a `DEF-` deform layer. The classifier
  prefers the `DEF-` filtered export when available — if the recommended
  armature is the IK/control armature rather than the deform armature, the
  generated rig will not drive the mesh correctly. Verify
  `recommended_primary_armature` in `classifier_report.json` matches the
  armature whose bones the mesh's vertex groups reference.
- If `1000idles.blend` contains multiple characters in the same file, run the
  pipeline once per character by hiding the other armatures before the
  inspector pass. The inspector iterates `bpy.data.objects`, not just the
  scene, so renaming may not be enough.

# Structural classification spot-check (junk-named rig)

A manual, in-Blender runbook to validate the **name-free structural classifier**
against a real Blender-built rig whose bones have deliberately meaningless names.
The synthetic `JunkNameBenchmarkTests` / `MissingIntermediateBenchmarkTests`
fixtures (in `tests/test_phase3_classifier.py`) assert the same contract on
hand-written JSON; this runbook confirms the contract still holds end-to-end when
the JSON is produced by the live armature inspector from an actual Blender scene.

This is **not** a pytest test — it requires an interactive Blender session and is
not discovered by `python -m unittest`.

## What it proves

When bone *names* carry no anatomical meaning, the classifier must fall back to
**topology + geometry** and still map the ASAM core targets to the right bones:

- `name_quality < 0.2` for the armature (names are junk, structure drives).
- `Hip`, `Head`, `Hand_Left`, `Hand_Right`, `Foot_Left`, `Foot_Right` resolve to
  the correct junk-named source bones.
- The reconciliation `decision` for structurally-recovered targets is
  `structure_only` (name channel contributed nothing), not `conflict_review`.

## Build the junk rig in Blender

Create a humanoid-topology armature with intentionally meaningless bone names.
The topology (parenting + head/tail placement), not the names, must be anatomically
plausible. A minimal Z-up humanoid (matching the synthetic benchmark):

| Junk name | Parent | Role (NOT encoded in the name) |
|-----------|--------|--------------------------------|
| `b0`      | —      | root / ground                  |
| `b1`      | `b0`   | hip                            |
| `b2`      | `b1`   | lower spine                    |
| `b3`      | `b2`   | upper spine                    |
| `b4`      | `b3`   | neck                           |
| `b5`      | `b4`   | head                           |
| `aL0..aL2`| `b3`   | left arm (upper, lower, hand)  |
| `aR0..aR2`| `b3`   | right arm                      |
| `lL0..lL2`| `b1`   | left leg (upper, lower, foot)  |
| `lR0..lR2`| `b1`   | right leg                      |

Steps:

1. In VS Code: `Blender: Start` (or launch Blender with the MCP add-on enabled).
2. Add an Armature, enter Edit Mode, and build the bones above with arms branching
   laterally off the upper spine and legs descending off the hip. Keep the spine
   on the centerline so the trunk extractor can find it.
3. Skin a simple body mesh to the rig and weight a few vertex groups to the junk
   names (e.g. `b1`, `b5`, `aL2`, `lL2`) so the inspector records a real
   mesh binding — the durable rule is the skin weights, never the names.
4. Save as `output/junkrig/junkrig.blend` (or any `output/<asset>/` folder).

## Export inspector JSON

With the junk `.blend` open in Blender, run the armature inspector so it writes
per-asset JSON next to the asset:

- In VS Code: open `src/armature_inspector/main.py`, then `Blender: Run Script`.

This produces `output/junkrig/<Armature>_all.json` (and `_filtered.json` if a
DEF- filter applies), including `placement_metadata` and `mesh_binding`.

## Run the classifier (any Python env)

```
python src/phase3_classifier/main.py --asset-dir output/junkrig
```

This writes `output/junkrig/classifier_report.json` and `build_plan.json`.

## Confirm the result

Open `classifier_report.json` and check the recommended armature entry:

- `armatures[0].name_quality` is `< 0.2`.
- Under `armatures[0].asam_targets` (or `semantic_mapping`):
  - `Hip.source_bone == "b1"`
  - `Head.source_bone == "b5"`
  - `Hand_Left.source_bone == "aL2"` and `Hand_Right.source_bone == "aR2"`
  - `Foot_Left.source_bone == "lL2"` and `Foot_Right.source_bone == "lR2"`
  - each of the above has `decision == "structure_only"` and an
    `evidence.structural` value well above `evidence.name`.

If any core target maps to the wrong junk bone, capture the offending
`asam_targets[...]` payload (with its `evidence` block) and the armature's
`name_quality`, then compare against the synthetic benchmark in
`tests/test_phase3_classifier.py::JunkNameBenchmarkTests` to localise whether the
gap is in the structural labeler (`src/phase3_classifier/structural_skeleton.py`)
or the confidence blend (`_confidence_weights` in `classifier.py`).

## Missing-intermediate variant (optional)

To reproduce `MissingIntermediateBenchmarkTests` in Blender, build a 2-bone leg
(thigh then foot, **no shin**) off the hip. After classifying, confirm:

- `Upper_Leg_Left.source_bone` is the thigh and `Foot_Left.source_bone` is the foot.
- `Lower_Leg_Left.action == "create_in_builder"` (the absent shin becomes a
  builder-created target rather than a stolen neighbour).

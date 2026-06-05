import sys
import unittest
from pathlib import Path

# dev_reload.py is bpy-free; import it as a top-level module (do NOT import the addon
# package, whose __init__ pulls in bpy).
ADDON_ROOT = Path(__file__).resolve().parents[1] / "addon"
if str(ADDON_ROOT) not in sys.path:
    sys.path.insert(0, str(ADDON_ROOT))

import dev_reload  # noqa: E402


class PurgePipelineModulesTests(unittest.TestCase):
    def test_purges_only_pipeline_modules(self):
        fake = {
            "asam_human_builder": object(),
            "asam_human_builder.builder": object(),
            "phase3_classifier.classifier": object(),
            "armature_inspector.inspector": object(),
            "pipeline.workflow": object(),
            "pipeline_paths": object(),
            "open_jaywalker": object(),           # add-on package - keep
            "open_jaywalker.operators": object(),  # add-on submodule - keep
            "bpy": object(),                       # keep
            "numpy": object(),                     # third-party - keep
        }
        purged = dev_reload.purge_pipeline_modules(fake)
        self.assertEqual(
            purged,
            sorted([
                "asam_human_builder",
                "asam_human_builder.builder",
                "phase3_classifier.classifier",
                "armature_inspector.inspector",
                "pipeline.workflow",
                "pipeline_paths",
            ]),
        )
        for gone in ("asam_human_builder", "asam_human_builder.builder", "pipeline_paths"):
            self.assertNotIn(gone, fake)
        for kept in ("open_jaywalker", "open_jaywalker.operators", "bpy", "numpy"):
            self.assertIn(kept, fake)

    def test_returns_empty_when_nothing_to_purge(self):
        self.assertEqual(dev_reload.purge_pipeline_modules({"bpy": object()}), [])

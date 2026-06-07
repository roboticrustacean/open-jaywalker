import importlib
import sys
import types
import unittest
from unittest import mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

class AddonInterfaceTests(unittest.TestCase):
    def setUp(self):
        # Save original environment variable to prevent polluting other tests
        import os
        self.orig_env_root = os.environ.get("OPEN_JAYWALKER_OUTPUT_ROOT")

        # Create a mock bpy module hierarchy
        self.fake_bpy = types.ModuleType("bpy")
        self.fake_bpy.app = types.ModuleType("bpy.app")
        self.fake_bpy.app.handlers = types.ModuleType("bpy.app.handlers")
        self.fake_bpy.app.handlers.persistent = lambda x: x  # decorator mock
        self.fake_bpy.app.handlers.load_post = []

        self.fake_bpy.types = types.SimpleNamespace(
            Operator=object,
            Panel=object,
            PropertyGroup=object,
            AddonPreferences=object,
            Scene=types.SimpleNamespace,
        )
        self.fake_bpy.props = types.SimpleNamespace(
            BoolProperty=mock.Mock(return_value=mock.Mock()),
            IntProperty=mock.Mock(return_value=mock.Mock()),
            StringProperty=mock.Mock(return_value=mock.Mock()),
            EnumProperty=mock.Mock(return_value=mock.Mock()),
            PointerProperty=mock.Mock(return_value=mock.Mock()),
        )
        self.fake_bpy.ops = types.SimpleNamespace(
            object=types.SimpleNamespace(
                mode_set=mock.Mock()
            )
        )
        self.fake_bpy.path = types.SimpleNamespace(
            abspath=lambda x: x
        )
        self.fake_bpy.data = types.SimpleNamespace(
            objects=[],
            collections=[],
            scenes=[]
        )

        # Patch sys.modules with our mocked bpy packages
        self.sys_modules_patch = {
            "bpy": self.fake_bpy,
            "bpy.app": self.fake_bpy.app,
            "bpy.app.handlers": self.fake_bpy.app.handlers
        }
        self.sys_modules_patcher = mock.patch.dict(sys.modules, self.sys_modules_patch)
        self.sys_modules_patcher.start()

        # Reload/import the addon modules
        if "addon.state" in sys.modules:
            del sys.modules["addon.state"]
        if "addon.operators" in sys.modules:
            del sys.modules["addon.operators"]
        if "addon.ui" in sys.modules:
            del sys.modules["addon.ui"]

        self.state = importlib.import_module("addon.state")
        self.operators = importlib.import_module("addon.operators")
        self.ui = importlib.import_module("addon.ui")

    def tearDown(self):
        import os
        self.sys_modules_patcher.stop()
        if self.orig_env_root is not None:
            os.environ["OPEN_JAYWALKER_OUTPUT_ROOT"] = self.orig_env_root
        elif "OPEN_JAYWALKER_OUTPUT_ROOT" in os.environ:
            del os.environ["OPEN_JAYWALKER_OUTPUT_ROOT"]

    def test_operators_clear_state_properties(self):
        # Test that run_pipeline and clean operators clear synthesized bones state
        mock_settings = types.SimpleNamespace(
            has_plan=True,
            built=True,
            synthesized_bones_csv="Full_Fingers_Left",
            synthesized_bones_by_character_csv="char0 (Full_Fingers_Left)",
            export_dir="some_dir",
            show_source_bones=False,
            show_source_mesh=False,
            show_generated_bones=True,
            show_generated_mesh=True,
        )
        mock_context = types.SimpleNamespace(
            scene=types.SimpleNamespace(open_jaywalker=mock_settings),
            preferences=types.SimpleNamespace(addons={
                "addon": types.SimpleNamespace(preferences=types.SimpleNamespace(output_dir="", export_dir="", dev_reload=False))
            })
        )

        with mock.patch("addon.operators._addon_prefs") as mock_prefs, \
             mock.patch("addon.operators.purge_previous_generated_artifacts") as mock_purge, \
             mock.patch("armature_inspector.inspector.inspect_scene", return_value=None):
            mock_prefs.return_value = types.SimpleNamespace(output_dir="C:/tmp", export_dir="", dev_reload=False)

            clean_op = self.operators.OJ_OT_clean()
            clean_op.report = mock.Mock()
            clean_op.execute(mock_context)

            self.assertEqual(mock_settings.synthesized_bones_csv, "")
            self.assertEqual(mock_settings.synthesized_bones_by_character_csv, "")
            self.assertFalse(mock_settings.built)

            # Re-populate
            mock_settings.synthesized_bones_csv = "Full_Fingers_Left"
            mock_settings.synthesized_bones_by_character_csv = "char0 (Full_Fingers_Left)"

            run_op = self.operators.OJ_OT_run_pipeline()
            run_op.report = mock.Mock()
            run_op.execute(mock_context)

            self.assertEqual(mock_settings.synthesized_bones_csv, "")
            self.assertEqual(mock_settings.synthesized_bones_by_character_csv, "")
            self.assertFalse(mock_settings.has_plan)
            self.assertFalse(mock_settings.built)

    def test_operator_build_populates_state_and_reports_warning_single(self):
        mock_settings = types.SimpleNamespace(
            has_plan=True,
            built=False,
            asset_dir="/x",
            export_dir="/x/exports",
            export_blend=False,
            export_gltf=True,
            per_character_export=False,
            build_succeeded=0,
            build_failed=0,
            failed_characters_csv="",
            synthesized_bones_csv="",
            synthesized_bones_by_character_csv="",
            show_source_bones=True,
            show_source_mesh=True,
        )
        mock_context = types.SimpleNamespace(
            scene=types.SimpleNamespace(open_jaywalker=mock_settings),
            window=types.SimpleNamespace(cursor_set=mock.Mock()),
            preferences=types.SimpleNamespace(addons={
                "addon": types.SimpleNamespace(preferences=types.SimpleNamespace(output_dir="", export_dir="", dev_reload=False))
            })
        )

        report_data = {
            "built_core_targets": ["Hip", "Root"],
            "targets_created_heuristically": ["Full_Fingers_Left", "Full_Fingers_Right"]
        }

        with mock.patch("addon.operators._addon_prefs") as mock_prefs, \
             mock.patch("asam_human_builder.build_runner.run_build", return_value=report_data) as mock_run_build, \
             mock.patch("asam_human_builder.builder.success_message", return_value="ASAM human built"):
            mock_prefs.return_value = types.SimpleNamespace(output_dir="C:/tmp", export_dir="", dev_reload=False)

            build_op = self.operators.OJ_OT_build()
            build_op.report = mock.Mock()
            build_op.execute(mock_context)

            self.assertTrue(mock_settings.built)
            self.assertEqual(mock_settings.synthesized_bones_csv, "Full_Fingers_Left, Full_Fingers_Right")
            self.assertEqual(mock_settings.synthesized_bones_by_character_csv, "")
            self.assertFalse(mock_settings.show_source_bones)
            self.assertFalse(mock_settings.show_source_mesh)

            build_op.report.assert_any_call(
                {'WARNING'},
                "Synthesized inert bones added for compliance: Full_Fingers_Left, Full_Fingers_Right"
            )

    def test_operator_build_populates_state_and_reports_warning_crowd(self):
        mock_settings = types.SimpleNamespace(
            has_plan=True,
            built=False,
            asset_dir="/x",
            export_dir="/x/exports",
            export_blend=False,
            export_gltf=True,
            per_character_export=False,
            build_succeeded=0,
            build_failed=0,
            failed_characters_csv="",
            synthesized_bones_csv="",
            synthesized_bones_by_character_csv="",
            show_source_bones=True,
            show_source_mesh=True,
        )
        mock_context = types.SimpleNamespace(
            scene=types.SimpleNamespace(open_jaywalker=mock_settings),
            window=types.SimpleNamespace(cursor_set=mock.Mock()),
            preferences=types.SimpleNamespace(addons={
                "addon": types.SimpleNamespace(preferences=types.SimpleNamespace(output_dir="", export_dir="", dev_reload=False))
            })
        )

        report_data = {
            "characters": [
                {
                    "character_id": "Hero000",
                    "targets_created_heuristically": ["Full_Fingers_Left"]
                },
                {
                    "character_id": "Hero001",
                    "targets_created_heuristically": ["Full_Toes_Right"]
                }
            ],
            "failed_characters": []
        }

        with mock.patch("addon.operators._addon_prefs") as mock_prefs, \
             mock.patch("asam_human_builder.build_runner.run_build", return_value=report_data) as mock_run_build, \
             mock.patch("asam_human_builder.builder.success_message", return_value="Crowd built"):
            mock_prefs.return_value = types.SimpleNamespace(output_dir="C:/tmp", export_dir="", dev_reload=False)

            build_op = self.operators.OJ_OT_build()
            build_op.report = mock.Mock()
            build_op.execute(mock_context)

            self.assertTrue(mock_settings.built)
            self.assertEqual(mock_settings.synthesized_bones_csv, "")
            self.assertEqual(mock_settings.synthesized_bones_by_character_csv, "Hero000 (Full_Fingers_Left) | Hero001 (Full_Toes_Right)")

            build_op.report.assert_any_call(
                {'WARNING'},
                "Synthesized inert bones added for compliance: Hero000 (Full_Fingers_Left), Hero001 (Full_Toes_Right)"
            )

    def test_operator_build_populates_state_for_mixed_crowd_success_failure(self):
        mock_settings = types.SimpleNamespace(
            has_plan=True,
            built=False,
            asset_dir="/x",
            export_dir="/x/exports",
            export_blend=False,
            export_gltf=True,
            per_character_export=False,
            build_succeeded=0,
            build_failed=0,
            failed_characters_csv="",
            synthesized_bones_csv="",
            synthesized_bones_by_character_csv="",
            show_source_bones=True,
            show_source_mesh=True,
        )
        mock_context = types.SimpleNamespace(
            scene=types.SimpleNamespace(open_jaywalker=mock_settings),
            window=types.SimpleNamespace(cursor_set=mock.Mock()),
            preferences=types.SimpleNamespace(addons={
                "addon": types.SimpleNamespace(preferences=types.SimpleNamespace(output_dir="", export_dir="", dev_reload=False))
            })
        )

        report_data = {
            "characters": [
                {
                    "character_id": "Hero000",
                    "targets_created_heuristically": []
                }
            ],
            "failed_characters": [
                {
                    "character_id": "Hero001",
                    "error": "Failed to resolve rig template"
                }
            ]
        }

        with mock.patch("addon.operators._addon_prefs") as mock_prefs, \
             mock.patch("asam_human_builder.build_runner.run_build", return_value=report_data) as mock_run_build, \
             mock.patch("asam_human_builder.builder.success_message", return_value="Crowd built"):
            mock_prefs.return_value = types.SimpleNamespace(output_dir="C:/tmp", export_dir="", dev_reload=False)

            build_op = self.operators.OJ_OT_build()
            build_op.report = mock.Mock()
            build_op.execute(mock_context)

            self.assertTrue(mock_settings.built)
            self.assertEqual(mock_settings.build_succeeded, 1)
            self.assertEqual(mock_settings.build_failed, 1)
            self.assertEqual(mock_settings.failed_characters_csv, "Hero001")

    def test_ui_draw_renders_warnings_without_crash(self):
        class MockLayout:
            def __init__(self):
                self.alert = False

            def separator(self, *args, **kwargs):
                pass

            def label(self, text="", icon='NONE', *args, **kwargs):
                pass

            def operator(self, name, icon='NONE', *args, **kwargs):
                pass

            def prop(self, data, prop_name, text="", icon='NONE', emboss=True, toggle=False, *args, **kwargs):
                pass

            def row(self, *args, **kwargs):
                return MockLayout()

            def box(self, *args, **kwargs):
                return MockLayout()

            def column(self, align=False, *args, **kwargs):
                return MockLayout()

        mock_layout = MockLayout()

        mock_settings = types.SimpleNamespace(
            built=True,
            has_plan=True,
            is_crowd=False,
            build_succeeded=1,
            build_failed=0,
            failed_characters_csv="",
            synthesized_bones_csv="Full_Fingers_Left, Full_Fingers_Right",
            synthesized_bones_by_character_csv="",
            asset_dir="/x",
            export_dir="/x/exports",
            recommended_armature="Rig",
            mapped=28,
            total=28,
            missing_csv="",
            missing_by_target_csv="",
            review_flags_csv="",
            character_ids_csv="",
            character_count=1,
            show_details=False,
            export_blend=False,
            export_gltf=True,
            per_character_export=False,
            show_failed_details=True,
            show_inert_details=True,
            show_generated_bones=True,
            show_generated_mesh=True,
            show_source_bones=False,
            show_source_mesh=False,
        )
        mock_context = types.SimpleNamespace(scene=types.SimpleNamespace(open_jaywalker=mock_settings))

        panel = self.ui.OJ_PT_panel()
        panel.layout = mock_layout

        panel.draw(mock_context)

        # Test crowd
        mock_settings.is_crowd = True
        mock_settings.synthesized_bones_csv = ""
        mock_settings.synthesized_bones_by_character_csv = "Hero000 (Full_Fingers_Left) | Hero001 (Full_Toes_Right)"

        panel.draw(mock_context)

    def test_ui_draw_renders_mixed_crowd_success_failure(self):
        class MockLayout:
            def __init__(self):
                self.labels = []
                self.alert = False

            def separator(self, *args, **kwargs):
                pass

            def label(self, text="", icon='NONE', *args, **kwargs):
                self.labels.append((text, icon))

            def operator(self, name, icon='NONE', *args, **kwargs):
                pass

            def prop(self, data, prop_name, text="", icon='NONE', emboss=True, toggle=False, *args, **kwargs):
                pass

            def row(self, *args, **kwargs):
                return self

            def box(self, *args, **kwargs):
                return self

            def column(self, align=False, *args, **kwargs):
                return self

        mock_layout = MockLayout()

        mock_settings = types.SimpleNamespace(
            built=True,
            has_plan=True,
            is_crowd=True,
            build_succeeded=5,
            build_failed=2,
            failed_characters_csv="char_01, char_03",
            synthesized_bones_csv="",
            synthesized_bones_by_character_csv="",
            asset_dir="/x",
            export_dir="/x/exports",
            recommended_armature="Rig",
            mapped=28,
            total=28,
            missing_csv="",
            missing_by_target_csv="",
            review_flags_csv="",
            character_ids_csv="",
            character_count=7,
            show_details=False,
            export_blend=False,
            export_gltf=True,
            per_character_export=False,
            show_failed_details=True,
            show_inert_details=True,
            show_generated_bones=True,
            show_generated_mesh=True,
            show_source_bones=False,
            show_source_mesh=False,
        )
        mock_context = types.SimpleNamespace(scene=types.SimpleNamespace(open_jaywalker=mock_settings))

        panel = self.ui.OJ_PT_panel()
        panel.layout = mock_layout

        panel.draw(mock_context)

        expected_labels = [
            ("Succeeded: 5", "CHECKMARK"),
            ("Failed: 2", "ERROR"),
            ("Failed characters:", "NONE"),
            ("   - char_01", "NONE"),
            ("   - char_03", "NONE")
        ]

        for text, icon in expected_labels:
            self.assertIn((text, icon), mock_layout.labels)

    def test_show_generated_bones_toggle_only_affects_armature_and_empty(self):
        """update_show_generated_bones only hides/shows ARMATURE and EMPTY generated objects."""
        class DictLikeMockObj:
            def __init__(self, type_str, gen_val):
                self.type = type_str
                self.props = {"open_jaywalker_generated": gen_val}
                self.hide_set = mock.Mock()
            def get(self, key, default=None):
                return self.props.get(key, default)

        gen_armature = DictLikeMockObj("ARMATURE", True)
        gen_empty = DictLikeMockObj("EMPTY", True)
        gen_mesh = DictLikeMockObj("MESH", True)
        non_gen_armature = DictLikeMockObj("ARMATURE", False)

        mock_settings = types.SimpleNamespace(show_generated_bones=False)
        mock_context = types.SimpleNamespace(
            scene=types.SimpleNamespace(
                objects=[gen_armature, gen_empty, gen_mesh, non_gen_armature, None]
            )
        )

        self.state.update_show_generated_bones(mock_settings, mock_context)

        gen_armature.hide_set.assert_called_once_with(True)
        gen_empty.hide_set.assert_called_once_with(True)
        gen_mesh.hide_set.assert_not_called()   # mesh must not be touched
        non_gen_armature.hide_set.assert_not_called()

        gen_armature.hide_set.reset_mock()
        gen_empty.hide_set.reset_mock()

        mock_settings.show_generated_bones = True
        self.state.update_show_generated_bones(mock_settings, mock_context)
        gen_armature.hide_set.assert_called_once_with(False)
        gen_empty.hide_set.assert_called_once_with(False)
        gen_mesh.hide_set.assert_not_called()

    def test_show_generated_mesh_toggle_only_affects_mesh(self):
        """update_show_generated_mesh only hides/shows MESH generated objects."""
        class DictLikeMockObj:
            def __init__(self, type_str, gen_val):
                self.type = type_str
                self.props = {"open_jaywalker_generated": gen_val}
                self.hide_set = mock.Mock()
            def get(self, key, default=None):
                return self.props.get(key, default)

        gen_armature = DictLikeMockObj("ARMATURE", True)
        gen_mesh = DictLikeMockObj("MESH", True)
        non_gen_mesh = DictLikeMockObj("MESH", False)

        mock_settings = types.SimpleNamespace(show_generated_mesh=False)
        mock_context = types.SimpleNamespace(
            scene=types.SimpleNamespace(
                objects=[gen_armature, gen_mesh, non_gen_mesh, None]
            )
        )

        self.state.update_show_generated_mesh(mock_settings, mock_context)

        gen_mesh.hide_set.assert_called_once_with(True)
        gen_armature.hide_set.assert_not_called()  # armature must not be touched
        non_gen_mesh.hide_set.assert_not_called()

        gen_mesh.hide_set.reset_mock()

        mock_settings.show_generated_mesh = True
        self.state.update_show_generated_mesh(mock_settings, mock_context)
        gen_mesh.hide_set.assert_called_once_with(False)
        gen_armature.hide_set.assert_not_called()

    def test_show_source_bones_toggle_only_affects_armature(self):
        """update_show_source_bones only hides/shows ARMATURE source objects."""
        class DictLikeMockObj:
            def __init__(self, type_str, source_val):
                self.type = type_str
                self.props = {"open_jaywalker_source_hidden": source_val}
                self.hide_set = mock.Mock()
            def get(self, key, default=None):
                return self.props.get(key, default)

        src_armature = DictLikeMockObj("ARMATURE", True)
        src_mesh = DictLikeMockObj("MESH", True)
        non_src_armature = DictLikeMockObj("ARMATURE", False)

        mock_settings = types.SimpleNamespace(show_source_bones=False)
        mock_context = types.SimpleNamespace(
            scene=types.SimpleNamespace(
                objects=[src_armature, src_mesh, non_src_armature, None]
            )
        )

        self.state.update_show_source_bones(mock_settings, mock_context)

        src_armature.hide_set.assert_called_once_with(True)
        src_mesh.hide_set.assert_not_called()   # mesh must not be touched
        non_src_armature.hide_set.assert_not_called()

        src_armature.hide_set.reset_mock()

        mock_settings.show_source_bones = True
        self.state.update_show_source_bones(mock_settings, mock_context)
        src_armature.hide_set.assert_called_once_with(False)
        src_mesh.hide_set.assert_not_called()

    def test_show_source_mesh_toggle_only_affects_mesh(self):
        """update_show_source_mesh only hides/shows MESH source objects."""
        class DictLikeMockObj:
            def __init__(self, type_str, source_val):
                self.type = type_str
                self.props = {"open_jaywalker_source_hidden": source_val}
                self.hide_set = mock.Mock()
            def get(self, key, default=None):
                return self.props.get(key, default)

        src_armature = DictLikeMockObj("ARMATURE", True)
        src_mesh = DictLikeMockObj("MESH", True)
        non_src_mesh = DictLikeMockObj("MESH", False)

        mock_settings = types.SimpleNamespace(show_source_mesh=False)
        mock_context = types.SimpleNamespace(
            scene=types.SimpleNamespace(
                objects=[src_armature, src_mesh, non_src_mesh, None]
            )
        )

        self.state.update_show_source_mesh(mock_settings, mock_context)

        src_mesh.hide_set.assert_called_once_with(True)
        src_armature.hide_set.assert_not_called()  # armature must not be touched
        non_src_mesh.hide_set.assert_not_called()

        src_mesh.hide_set.reset_mock()

        mock_settings.show_source_mesh = True
        self.state.update_show_source_mesh(mock_settings, mock_context)
        src_mesh.hide_set.assert_called_once_with(False)
        src_armature.hide_set.assert_not_called()

    def test_show_four_toggles_ui_layout(self):
        """Build Results section renders 4 granular toggles in 2 rows (Generated / Source)."""
        class MockLayout:
            def __init__(self):
                self.props_drawn = []
                self.labels_drawn = []

            def separator(self, *args, **kwargs):
                pass

            def label(self, text="", icon='NONE', *args, **kwargs):
                self.labels_drawn.append(text)

            def operator(self, name, icon='NONE', *args, **kwargs):
                pass

            def prop(self, data, prop_name, text="", icon='NONE', emboss=True, toggle=False, *args, **kwargs):
                self.props_drawn.append(prop_name)

            def row(self, *args, **kwargs):
                return self

            def box(self, *args, **kwargs):
                return self

            def column(self, align=False, *args, **kwargs):
                return self

        mock_layout = MockLayout()
        ui_settings = types.SimpleNamespace(
            built=True,
            has_plan=True,
            is_crowd=False,
            build_succeeded=1,
            build_failed=0,
            failed_characters_csv="",
            synthesized_bones_csv="",
            synthesized_bones_by_character_csv="",
            asset_dir="/x",
            export_dir="/x/exports",
            recommended_armature="Rig",
            mapped=28,
            total=28,
            missing_csv="",
            missing_by_target_csv="",
            review_flags_csv="",
            character_ids_csv="",
            character_count=1,
            show_details=False,
            export_blend=False,
            export_gltf=True,
            per_character_export=False,
            show_generated_bones=True,
            show_generated_mesh=True,
            show_source_bones=False,
            show_source_mesh=False,
            show_failed_details=False,
            show_inert_details=False,
        )
        ui_context = types.SimpleNamespace(scene=types.SimpleNamespace(open_jaywalker=ui_settings))

        panel = self.ui.OJ_PT_panel()
        panel.layout = mock_layout
        panel.draw(ui_context)

        # All 4 new props must appear
        self.assertIn("show_generated_bones", mock_layout.props_drawn)
        self.assertIn("show_generated_mesh", mock_layout.props_drawn)
        self.assertIn("show_source_bones", mock_layout.props_drawn)
        self.assertIn("show_source_mesh", mock_layout.props_drawn)

        # Old props must NOT appear
        self.assertNotIn("show_generated_armature", mock_layout.props_drawn)
        self.assertNotIn("show_source", mock_layout.props_drawn)

        # Row labels
        self.assertIn("Generated:", mock_layout.labels_drawn)
        self.assertIn("Source:", mock_layout.labels_drawn)

    def test_reset_pipeline_operator(self):
        import tempfile
        import shutil

        # Create a temp directory for mock assets
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, temp_dir)

        mock_settings = types.SimpleNamespace(
            has_plan=True,
            built=True,
            show_details=True,
            asset_dir=temp_dir,
            export_dir=temp_dir + "/exports",
            recommended_armature="Rig",
            mapped=15,
            total=28,
            missing_csv="some_missing",
            missing_by_target_csv="some_target",
            review_flags_csv="flags",
            character_ids_csv="ids",
            is_crowd=True,
            character_count=5,
            build_succeeded=4,
            build_failed=1,
            failed_characters_csv="char1",
            synthesized_bones_csv="bone1",
            synthesized_bones_by_character_csv="char1 (bone1)",
            show_failed_details=True,
            show_inert_details=True,
            show_source_bones=False,
            show_source_mesh=False,
            show_generated_bones=True,
            show_generated_mesh=True,
        )
        mock_context = types.SimpleNamespace(
            scene=types.SimpleNamespace(open_jaywalker=mock_settings)
        )

        # Create dummy report files in asset_dir
        reports = ["classifier_report.json", "build_plan.json", "builder_report.json"]
        for report in reports:
            with open(Path(temp_dir) / report, "w") as f:
                f.write("{}")

        # Verify files exist
        for report in reports:
            self.assertTrue((Path(temp_dir) / report).exists())

        with mock.patch("addon.operators.purge_previous_generated_artifacts") as mock_purge:
            reset_op = self.operators.OJ_OT_reset_pipeline()
            reset_op.report = mock.Mock()
            res = reset_op.execute(mock_context)

            self.assertEqual(res, {'FINISHED'})
            mock_purge.assert_called_once()

        # Verify files are deleted
        for report in reports:
            self.assertFalse((Path(temp_dir) / report).exists())

        # Verify state properties are reset to default values
        self.assertFalse(mock_settings.has_plan)
        self.assertFalse(mock_settings.built)
        self.assertFalse(mock_settings.show_details)
        self.assertEqual(mock_settings.asset_dir, "")
        self.assertEqual(mock_settings.export_dir, "")
        self.assertEqual(mock_settings.recommended_armature, "")
        self.assertEqual(mock_settings.mapped, 0)
        self.assertEqual(mock_settings.total, 28)
        self.assertEqual(mock_settings.missing_csv, "")
        self.assertEqual(mock_settings.missing_by_target_csv, "")
        self.assertEqual(mock_settings.review_flags_csv, "")
        self.assertEqual(mock_settings.character_ids_csv, "")
        self.assertFalse(mock_settings.is_crowd)
        self.assertEqual(mock_settings.character_count, 0)
        self.assertEqual(mock_settings.build_succeeded, 0)
        self.assertEqual(mock_settings.build_failed, 0)
        self.assertEqual(mock_settings.failed_characters_csv, "")
        self.assertEqual(mock_settings.synthesized_bones_csv, "")
        self.assertEqual(mock_settings.synthesized_bones_by_character_csv, "")
        self.assertFalse(mock_settings.show_failed_details)
        self.assertFalse(mock_settings.show_inert_details)
        self.assertTrue(mock_settings.show_source_bones)
        self.assertTrue(mock_settings.show_source_mesh)
        self.assertTrue(mock_settings.show_generated_bones)
        self.assertTrue(mock_settings.show_generated_mesh)

    def test_load_post_handler_resets_properties(self):
        # We need to test the load_post_handler defined in addon.__init__
        if "addon" in sys.modules:
            del sys.modules["addon"]
        addon = importlib.import_module("addon")

        mock_settings = types.SimpleNamespace(
            has_plan=True,
            built=True,
            show_details=True,
            asset_dir="/some/path",
            export_dir="/some/path/exports",
            recommended_armature="Rig",
            mapped=15,
            total=28,
            missing_csv="some_missing",
            missing_by_target_csv="some_target",
            review_flags_csv="flags",
            character_ids_csv="ids",
            is_crowd=True,
            character_count=5,
            build_succeeded=4,
            build_failed=1,
            failed_characters_csv="char1",
            synthesized_bones_csv="bone1",
            synthesized_bones_by_character_csv="char1 (bone1)",
            show_failed_details=True,
            show_inert_details=True,
            show_source_bones=False,
            show_source_mesh=False,
            show_generated_bones=True,
            show_generated_mesh=True,
        )

        # Set up fake_bpy.context
        self.fake_bpy.context = types.SimpleNamespace(
            scene=types.SimpleNamespace(open_jaywalker=mock_settings)
        )

        # Call the load_post_handler
        addon.load_post_handler(None, None)

        # Verify state properties are reset
        self.assertFalse(mock_settings.has_plan)
        self.assertFalse(mock_settings.built)
        self.assertFalse(mock_settings.show_details)
        self.assertEqual(mock_settings.asset_dir, "")
        self.assertEqual(mock_settings.export_dir, "")
        self.assertEqual(mock_settings.recommended_armature, "")
        self.assertEqual(mock_settings.mapped, 0)
        self.assertEqual(mock_settings.total, 28)
        self.assertEqual(mock_settings.missing_csv, "")
        self.assertEqual(mock_settings.missing_by_target_csv, "")
        self.assertEqual(mock_settings.review_flags_csv, "")
        self.assertEqual(mock_settings.character_ids_csv, "")
        self.assertFalse(mock_settings.is_crowd)
        self.assertEqual(mock_settings.character_count, 0)
        self.assertEqual(mock_settings.build_succeeded, 0)
        self.assertEqual(mock_settings.build_failed, 0)
        self.assertEqual(mock_settings.failed_characters_csv, "")
        self.assertEqual(mock_settings.synthesized_bones_csv, "")
        self.assertEqual(mock_settings.synthesized_bones_by_character_csv, "")
        self.assertFalse(mock_settings.show_failed_details)
        self.assertFalse(mock_settings.show_inert_details)
        self.assertTrue(mock_settings.show_source_bones)
        self.assertTrue(mock_settings.show_source_mesh)
        self.assertTrue(mock_settings.show_generated_bones)
        self.assertTrue(mock_settings.show_generated_mesh)

    def test_ui_draw_collapsible_sections_and_reset_button(self):
        class MockLayout:
            def __init__(self):
                self.labels = []
                self.operators = []
                self.props = []
                self.alert = False

            def separator(self, *args, **kwargs):
                pass

            def label(self, text="", icon='NONE', *args, **kwargs):
                self.labels.append((text, icon))

            def operator(self, name, icon='NONE', *args, **kwargs):
                self.operators.append((name, icon))

            def prop(self, data, prop_name, text="", icon='NONE', emboss=True, toggle=False, *args, **kwargs):
                self.props.append((prop_name, text, icon, emboss))

            def row(self, *args, **kwargs):
                return self

            def box(self, *args, **kwargs):
                return self

            def column(self, align=False, *args, **kwargs):
                return self

        # Test case 1: Crowd build failed and inert bones warning, collapsed state
        mock_settings = types.SimpleNamespace(
            built=True,
            has_plan=True,
            is_crowd=True,
            build_succeeded=1,
            build_failed=1,
            failed_characters_csv="char_01",
            synthesized_bones_csv="",
            synthesized_bones_by_character_csv="char_01 (Full_Fingers_Left)",
            asset_dir="/x",
            export_dir="/x/exports",
            recommended_armature="Rig",
            mapped=28,
            total=28,
            missing_csv="",
            missing_by_target_csv="",
            review_flags_csv="",
            character_ids_csv="",
            character_count=2,
            show_details=False,
            export_blend=False,
            export_gltf=True,
            per_character_export=False,
            show_failed_details=False,
            show_inert_details=False,
            show_generated_bones=True,
            show_generated_mesh=True,
            show_source_bones=False,
            show_source_mesh=False,
        )
        mock_context = types.SimpleNamespace(scene=types.SimpleNamespace(open_jaywalker=mock_settings))

        mock_layout = MockLayout()
        panel = self.ui.OJ_PT_panel()
        panel.layout = mock_layout
        panel.draw(mock_context)

        # Verify reset button is drawn
        self.assertIn(("open_jaywalker.reset_pipeline", "LOOP_BACK"), mock_layout.operators)

        # Verify collapsible properties are drawn with TRIA_RIGHT icon (collapsed)
        self.assertIn(("show_failed_details", "Failed Details", "TRIA_RIGHT", False), mock_layout.props)
        self.assertIn(("show_inert_details", "Inert Bone Details", "TRIA_RIGHT", False), mock_layout.props)

        # Since details are false, the actual details shouldn't be drawn (e.g. mock_layout.labels should NOT contain character names or bone details)
        self.assertNotIn(("   - char_01", "NONE"), mock_layout.labels)

        # Test case 2: expanded state
        mock_settings.show_failed_details = True
        mock_settings.show_inert_details = True

        mock_layout = MockLayout()
        panel.layout = mock_layout
        panel.draw(mock_context)

        # Verify collapsible properties are drawn with TRIA_DOWN icon (expanded)
        self.assertIn(("show_failed_details", "Failed Details", "TRIA_DOWN", False), mock_layout.props)
        self.assertIn(("show_inert_details", "Inert Bone Details", "TRIA_DOWN", False), mock_layout.props)

        # Verify the details are drawn
        self.assertIn(("Failed characters:", "NONE"), mock_layout.labels)
        self.assertIn(("   - char_01", "NONE"), mock_layout.labels)

    def test_inert_bone_details_drawn_after_reset_clean_buttons(self):
        """Inert bone details section must be drawn AFTER Reset/Clean operators."""
        class MockLayout:
            def __init__(self):
                self._order = []

            def separator(self, *args, **kwargs):
                pass

            def label(self, text="", icon='NONE', *args, **kwargs):
                pass

            def operator(self, name, icon='NONE', *args, **kwargs):
                self._order.append(("operator", name))

            def prop(self, data, prop_name, text="", icon='NONE', emboss=True, toggle=False, *args, **kwargs):
                # Track the inert details toggle so we can check ordering
                if prop_name == "show_inert_details":
                    self._order.append(("prop", prop_name))

            def row(self, *args, **kwargs):
                return self

            def box(self, *args, **kwargs):
                return self

            def column(self, align=False, *args, **kwargs):
                return self

        mock_settings = types.SimpleNamespace(
            built=True,
            has_plan=True,
            is_crowd=False,
            build_succeeded=1,
            build_failed=0,
            failed_characters_csv="",
            synthesized_bones_csv="Neck",
            synthesized_bones_by_character_csv="",
            asset_dir="/x",
            export_dir="/x/exports",
            recommended_armature="Rig",
            mapped=28,
            total=28,
            missing_csv="",
            missing_by_target_csv="",
            review_flags_csv="",
            character_ids_csv="",
            character_count=1,
            show_details=False,
            export_blend=False,
            export_gltf=True,
            per_character_export=False,
            show_failed_details=False,
            show_inert_details=False,
            show_generated_bones=True,
            show_generated_mesh=True,
            show_source_bones=False,
            show_source_mesh=False,
        )
        mock_context = types.SimpleNamespace(scene=types.SimpleNamespace(open_jaywalker=mock_settings))

        mock_layout = MockLayout()
        panel = self.ui.OJ_PT_panel()
        panel.layout = mock_layout
        panel.draw(mock_context)

        # Find positions in draw order
        reset_pos = next(
            (i for i, x in enumerate(mock_layout._order) if x == ("operator", "open_jaywalker.reset_pipeline")),
            None
        )
        inert_pos = next(
            (i for i, x in enumerate(mock_layout._order) if x == ("prop", "show_inert_details")),
            None
        )

        self.assertIsNotNone(reset_pos, "reset_pipeline operator not found in draw order")
        self.assertIsNotNone(inert_pos, "show_inert_details prop not found in draw order")
        self.assertGreater(inert_pos, reset_pos,
            "Inert bone details must be drawn after Reset button")

    def test_run_pipeline_sets_env_from_prefs(self):
        """Output/export env vars come from the add-on preferences."""
        import os
        self.addCleanup(lambda: os.environ.pop("OPEN_JAYWALKER_EXPORT_DIR", None))
        mock_settings = types.SimpleNamespace(
            has_plan=True, built=True,
            synthesized_bones_csv="", synthesized_bones_by_character_csv="",
            export_dir="", show_source_bones=False, show_source_mesh=False,
            show_generated_bones=True, show_generated_mesh=True,
        )
        mock_context = types.SimpleNamespace(
            scene=types.SimpleNamespace(open_jaywalker=mock_settings),
            preferences=types.SimpleNamespace(addons={
                "addon": types.SimpleNamespace(preferences=types.SimpleNamespace(
                    output_dir="C:/pref_out", export_dir="C:/pref_exp", dev_reload=False))
            })
        )
        with mock.patch("addon.operators._addon_prefs") as mock_prefs, \
             mock.patch("addon.operators.purge_previous_generated_artifacts"), \
             mock.patch("armature_inspector.inspector.inspect_scene", return_value=None):
            mock_prefs.return_value = types.SimpleNamespace(
                output_dir="C:/pref_out", export_dir="C:/pref_exp", dev_reload=False)
            run_op = self.operators.OJ_OT_run_pipeline()
            run_op.report = mock.Mock()
            run_op.execute(mock_context)
            self.assertEqual(os.environ.get("OPEN_JAYWALKER_OUTPUT_ROOT"), "C:/pref_out")
            self.assertEqual(os.environ.get("OPEN_JAYWALKER_EXPORT_DIR"), "C:/pref_exp")

    def test_ui_export_dir_editable_after_run_and_no_top_pickers(self):
        """After a run the Exports path is an editable export_dir field; no top pickers."""
        class MockLayout:
            def __init__(self):
                self.props = []
            def separator(self, *a, **k):
                pass
            def label(self, text="", icon='NONE', *a, **k):
                pass
            def operator(self, name, icon='NONE', *a, **k):
                pass
            def prop(self, data, prop_name, text="", icon='NONE', emboss=True, toggle=False, *a, **k):
                self.props.append(prop_name)
            def row(self, *a, **k):
                return self
            def box(self, *a, **k):
                return self
            def column(self, align=False, *a, **k):
                return self

        mock_layout = MockLayout()
        s = types.SimpleNamespace(
            built=False, has_plan=True, is_crowd=False,
            recommended_armature="Rig", mapped=28, total=28,
            missing_csv="", missing_by_target_csv="", review_flags_csv="",
            character_ids_csv="", character_count=0, show_details=False,
            export_blend=False, export_gltf=True, per_character_export=False,
            asset_dir="/x", export_dir="/x/exports",
        )
        ctx = types.SimpleNamespace(scene=types.SimpleNamespace(open_jaywalker=s))
        panel = self.ui.OJ_PT_panel()
        panel.layout = mock_layout
        panel.draw(ctx)
        # Exports is now an editable field bound to export_dir.
        self.assertIn("export_dir", mock_layout.props)
        # The removed top pickers must not reappear.
        self.assertNotIn("output_root", mock_layout.props)
        self.assertNotIn("export_root", mock_layout.props)

if __name__ == "__main__":
    unittest.main()

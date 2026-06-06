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
            synthesized_bones_by_character_csv="char0 (Full_Fingers_Left)"
        )
        mock_context = types.SimpleNamespace(
            scene=types.SimpleNamespace(open_jaywalker=mock_settings),
            preferences=types.SimpleNamespace(addons={
                "addon": types.SimpleNamespace(preferences=types.SimpleNamespace(output_dir="", dev_reload=False))
            })
        )
        
        with mock.patch("addon.operators._addon_prefs") as mock_prefs, \
             mock.patch("addon.operators.purge_previous_generated_artifacts") as mock_purge, \
             mock.patch("armature_inspector.inspector.inspect_scene", return_value=None):
            mock_prefs.return_value = types.SimpleNamespace(output_dir="C:/tmp", dev_reload=False)
            
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
            packaging_mode="inplace_only",
            export_gltf=False,
            build_succeeded=0,
            build_failed=0,
            failed_characters_csv="",
            synthesized_bones_csv="",
            synthesized_bones_by_character_csv=""
        )
        mock_context = types.SimpleNamespace(
            scene=types.SimpleNamespace(open_jaywalker=mock_settings),
            window=types.SimpleNamespace(cursor_set=mock.Mock()),
            preferences=types.SimpleNamespace(addons={
                "addon": types.SimpleNamespace(preferences=types.SimpleNamespace(output_dir="", dev_reload=False))
            })
        )
        
        report_data = {
            "built_core_targets": ["Hip", "Root"],
            "targets_created_heuristically": ["Full_Fingers_Left", "Full_Fingers_Right"]
        }
        
        with mock.patch("addon.operators._addon_prefs") as mock_prefs, \
             mock.patch("asam_human_builder.build_runner.run_build", return_value=report_data) as mock_run_build, \
             mock.patch("asam_human_builder.builder.success_message", return_value="ASAM human built"):
            mock_prefs.return_value = types.SimpleNamespace(output_dir="C:/tmp", dev_reload=False)
            
            build_op = self.operators.OJ_OT_build()
            build_op.report = mock.Mock()
            build_op.execute(mock_context)
            
            self.assertTrue(mock_settings.built)
            self.assertEqual(mock_settings.synthesized_bones_csv, "Full_Fingers_Left, Full_Fingers_Right")
            self.assertEqual(mock_settings.synthesized_bones_by_character_csv, "")
            
            build_op.report.assert_any_call(
                {'WARNING'},
                "Synthesized inert bones added for compliance: Full_Fingers_Left, Full_Fingers_Right"
            )

    def test_operator_build_populates_state_and_reports_warning_crowd(self):
        mock_settings = types.SimpleNamespace(
            has_plan=True,
            built=False,
            asset_dir="/x",
            packaging_mode="inplace_only",
            export_gltf=False,
            build_succeeded=0,
            build_failed=0,
            failed_characters_csv="",
            synthesized_bones_csv="",
            synthesized_bones_by_character_csv=""
        )
        mock_context = types.SimpleNamespace(
            scene=types.SimpleNamespace(open_jaywalker=mock_settings),
            window=types.SimpleNamespace(cursor_set=mock.Mock()),
            preferences=types.SimpleNamespace(addons={
                "addon": types.SimpleNamespace(preferences=types.SimpleNamespace(output_dir="", dev_reload=False))
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
            mock_prefs.return_value = types.SimpleNamespace(output_dir="C:/tmp", dev_reload=False)
            
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
            packaging_mode="inplace_only",
            export_gltf=False,
            build_succeeded=0,
            build_failed=0,
            failed_characters_csv="",
            synthesized_bones_csv="",
            synthesized_bones_by_character_csv=""
        )
        mock_context = types.SimpleNamespace(
            scene=types.SimpleNamespace(open_jaywalker=mock_settings),
            window=types.SimpleNamespace(cursor_set=mock.Mock()),
            preferences=types.SimpleNamespace(addons={
                "addon": types.SimpleNamespace(preferences=types.SimpleNamespace(output_dir="", dev_reload=False))
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
            mock_prefs.return_value = types.SimpleNamespace(output_dir="C:/tmp", dev_reload=False)
            
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
                
            def separator(self):
                pass
                
            def label(self, text="", icon='NONE'):
                pass
                
            def operator(self, name, icon='NONE'):
                pass
                
            def prop(self, data, prop_name, text="", icon='NONE', emboss=True):
                pass
                
            def row(self):
                return MockLayout()
                
            def box(self):
                return MockLayout()
                
            def column(self, align=False):
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
            recommended_armature="Rig",
            mapped=28,
            total=28,
            missing_csv="",
            missing_by_target_csv="",
            review_flags_csv="",
            character_ids_csv="",
            character_count=1,
            show_details=False,
            packaging_mode="inplace_only",
            export_gltf=False
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
                
            def separator(self):
                pass
                
            def label(self, text="", icon='NONE'):
                self.labels.append((text, icon))
                
            def operator(self, name, icon='NONE'):
                pass
                
            def prop(self, data, prop_name, text="", icon='NONE', emboss=True):
                pass
                
            def row(self):
                return self
                
            def box(self):
                return self
                
            def column(self, align=False):
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
            recommended_armature="Rig",
            mapped=28,
            total=28,
            missing_csv="",
            missing_by_target_csv="",
            review_flags_csv="",
            character_ids_csv="",
            character_count=7,
            show_details=False,
            packaging_mode="inplace_only",
            export_gltf=False
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

    def test_show_generated_armature_toggle_and_ui(self):
        # 1. Test update callback toggles hide_set on generated armatures
        mock_generated_armature = types.SimpleNamespace(
            type="ARMATURE",
            open_jaywalker_generated=True,
            hide_set=mock.Mock()
        )
        mock_other_object = types.SimpleNamespace(
            type="MESH",
            open_jaywalker_generated=True,
            hide_set=mock.Mock()
        )
        mock_non_gen_armature = types.SimpleNamespace(
            type="ARMATURE",
            open_jaywalker_generated=False,
            hide_set=mock.Mock()
        )
        
        class DictLikeMockObj:
            def __init__(self, type_str, gen_val):
                self.type = type_str
                self.props = {"open_jaywalker_generated": gen_val}
                self.hide_set = mock.Mock()
            def get(self, key, default=None):
                return self.props.get(key, default)

        mock_dict_gen_armature = DictLikeMockObj("ARMATURE", True)
        mock_dict_non_gen_armature = DictLikeMockObj("ARMATURE", False)

        mock_settings = types.SimpleNamespace(show_generated_armature=False)
        mock_context = types.SimpleNamespace(
            scene=types.SimpleNamespace(
                objects=[
                    mock_generated_armature,
                    mock_other_object,
                    mock_non_gen_armature,
                    mock_dict_gen_armature,
                    mock_dict_non_gen_armature,
                    None
                ]
            )
        )

        self.state.update_show_generated_armature(mock_settings, mock_context)
        
        mock_generated_armature.hide_set.assert_called_once_with(True)
        mock_other_object.hide_set.assert_not_called()
        mock_non_gen_armature.hide_set.assert_not_called()
        mock_dict_gen_armature.hide_set.assert_called_once_with(True)
        mock_dict_non_gen_armature.hide_set.assert_not_called()
        
        mock_generated_armature.hide_set.reset_mock()
        mock_dict_gen_armature.hide_set.reset_mock()

        mock_settings.show_generated_armature = True
        self.state.update_show_generated_armature(mock_settings, mock_context)
        mock_generated_armature.hide_set.assert_called_once_with(False)
        mock_dict_gen_armature.hide_set.assert_called_once_with(False)

        # 2. Test that UI renders the checkbox without crashing
        class MockLayout:
            def __init__(self):
                self.props_drawn = []
                
            def separator(self):
                pass
                
            def label(self, text="", icon='NONE'):
                pass
                
            def operator(self, name, icon='NONE'):
                pass
                
            def prop(self, data, prop_name, text="", icon='NONE', emboss=True):
                self.props_drawn.append((data, prop_name))
                
            def row(self):
                return self
                
            def box(self):
                return self
                
            def column(self, align=False):
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
            recommended_armature="Rig",
            mapped=28,
            total=28,
            missing_csv="",
            missing_by_target_csv="",
            review_flags_csv="",
            character_ids_csv="",
            character_count=1,
            show_details=False,
            packaging_mode="inplace_only",
            export_gltf=False,
            show_generated_armature=True
        )
        ui_context = types.SimpleNamespace(scene=types.SimpleNamespace(open_jaywalker=ui_settings))
        
        panel = self.ui.OJ_PT_panel()
        panel.layout = mock_layout
        panel.draw(ui_context)
        
        self.assertIn((ui_settings, "show_generated_armature"), mock_layout.props_drawn)

if __name__ == "__main__":
    unittest.main()

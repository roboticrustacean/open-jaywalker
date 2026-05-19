import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline_paths import (  # noqa: E402
    ENV_VAR,
    resolve_asset_dir,
    resolve_output_root,
)


class PipelinePathsTests(unittest.TestCase):
    def setUp(self):
        self._original_env = os.environ.pop(ENV_VAR, None)

    def tearDown(self):
        if self._original_env is not None:
            os.environ[ENV_VAR] = self._original_env
        else:
            os.environ.pop(ENV_VAR, None)

    def test_resolve_output_root_defaults_to_repo_output_dir(self):
        self.assertEqual(resolve_output_root(), (REPO_ROOT / "output").resolve())

    def test_resolve_output_root_honors_env_override(self):
        override = tempfile.mkdtemp(prefix="pipeline_paths_test_")
        os.environ[ENV_VAR] = override
        self.assertEqual(resolve_output_root(), Path(override).resolve())

    def test_resolve_output_root_expands_user_home(self):
        os.environ[ENV_VAR] = "~"
        self.assertEqual(resolve_output_root(), Path("~").expanduser().resolve())

    def test_resolve_asset_dir_appends_asset_name(self):
        override = tempfile.mkdtemp(prefix="pipeline_paths_test_")
        os.environ[ENV_VAR] = override
        result = resolve_asset_dir("SomeAsset")
        self.assertEqual(result, (Path(override) / "SomeAsset").resolve())

    def test_resolve_asset_dir_creates_directory(self):
        override = tempfile.mkdtemp(prefix="pipeline_paths_test_")
        os.environ[ENV_VAR] = override
        result = resolve_asset_dir("NewAsset")
        self.assertTrue(result.is_dir())


if __name__ == "__main__":
    unittest.main()

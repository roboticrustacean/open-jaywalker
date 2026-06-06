import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import glb_to_osgb  # noqa: E402


class _RecordingRunner:
    """Stand-in for subprocess.run that records calls instead of executing."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((list(cmd), kwargs))
        return 0


class ResolveOsgconvTests(unittest.TestCase):
    def test_explicit_takes_precedence(self):
        self.assertEqual(
            glb_to_osgb.resolve_osgconv(explicit="C:/tools/osgconv.exe"),
            "C:/tools/osgconv.exe",
        )

    def test_env_var_used_when_no_explicit(self):
        with unittest.mock.patch.dict("os.environ", {"OSGCONV": "/usr/bin/osgconv"}):
            with unittest.mock.patch.object(glb_to_osgb.shutil, "which", return_value=None):
                self.assertEqual(glb_to_osgb.resolve_osgconv(), "/usr/bin/osgconv")

    def test_which_used_as_last_resort(self):
        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            with unittest.mock.patch.object(
                glb_to_osgb.shutil, "which", return_value="/opt/osg/osgconv"
            ):
                self.assertEqual(glb_to_osgb.resolve_osgconv(), "/opt/osg/osgconv")

    def test_explicit_beats_env_and_which(self):
        with unittest.mock.patch.dict("os.environ", {"OSGCONV": "/env/osgconv"}):
            with unittest.mock.patch.object(
                glb_to_osgb.shutil, "which", return_value="/which/osgconv"
            ):
                self.assertEqual(
                    glb_to_osgb.resolve_osgconv(explicit="/explicit/osgconv"),
                    "/explicit/osgconv",
                )

    def test_unresolved_raises_with_guidance(self):
        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            with unittest.mock.patch.object(glb_to_osgb.shutil, "which", return_value=None):
                with self.assertRaises(FileNotFoundError) as ctx:
                    glb_to_osgb.resolve_osgconv()
        self.assertIn("osgconv", str(ctx.exception).lower())
        self.assertIn("openscenegraph", str(ctx.exception).lower())


class OutputPathTests(unittest.TestCase):
    def test_default_alongside_input(self):
        out = glb_to_osgb.output_path_for(Path("/a/b/char_asam.glb"))
        self.assertEqual(out, Path("/a/b/char_asam.osgb"))

    def test_outdir_override(self):
        out = glb_to_osgb.output_path_for(Path("/a/b/char_asam.glb"), outdir=Path("/out"))
        self.assertEqual(out, Path("/out/char_asam.osgb"))


class BuildCommandTests(unittest.TestCase):
    def test_argv_shape(self):
        cmd = glb_to_osgb.build_command("osgconv", Path("in.glb"), Path("out.osgb"))
        self.assertEqual(cmd, ["osgconv", "in.glb", "out.osgb"])


class ConvertTests(unittest.TestCase):
    def test_missing_input_raises(self):
        with self.assertRaises(FileNotFoundError):
            glb_to_osgb.convert(
                Path("does_not_exist.glb"),
                osgconv="osgconv",
                runner=_RecordingRunner(),
            )

    def test_convert_runs_command_and_returns_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            glb = Path(tmp) / "char_asam.glb"
            glb.write_bytes(b"glTF-stub")
            runner = _RecordingRunner()
            out = glb_to_osgb.convert(glb, osgconv="osgconv", runner=runner)
            self.assertEqual(out, glb.with_suffix(".osgb"))
            self.assertEqual(len(runner.calls), 1)
            cmd, kwargs = runner.calls[0]
            self.assertEqual(cmd, ["osgconv", str(glb), str(out)])
            self.assertTrue(kwargs.get("check"))

    def test_convert_outdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            glb = Path(tmp) / "char_asam.glb"
            glb.write_bytes(b"glTF-stub")
            outdir = Path(tmp) / "osgb"
            runner = _RecordingRunner()
            out = glb_to_osgb.convert(glb, outdir=outdir, osgconv="osgconv", runner=runner)
            self.assertEqual(out, outdir / "char_asam.osgb")
            self.assertTrue(outdir.exists())

    def test_convert_many(self):
        with tempfile.TemporaryDirectory() as tmp:
            globs = []
            for i in range(3):
                g = Path(tmp) / "char{0}_asam.glb".format(i)
                g.write_bytes(b"glTF-stub")
                globs.append(g)
            runner = _RecordingRunner()
            outs = glb_to_osgb.convert_many(globs, osgconv="osgconv", runner=runner)
            self.assertEqual(outs, [g.with_suffix(".osgb") for g in globs])
            self.assertEqual(len(runner.calls), 3)


if __name__ == "__main__":
    unittest.main()

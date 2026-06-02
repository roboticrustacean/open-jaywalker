import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.build_gate import resolve_build_decision  # noqa: E402


class BuildGateTests(unittest.TestCase):
    def test_tty_yes_answers_build(self):
        for answer in ("y", "Y", "yes", "YES", " yes "):
            decision = resolve_build_decision(
                stdin_isatty=True, env={}, argv=[], input_fn=lambda _p: answer
            )
            self.assertTrue(decision, answer)

    def test_tty_non_yes_does_not_build(self):
        for answer in ("", "n", "no", "nope", "build"):
            decision = resolve_build_decision(
                stdin_isatty=True, env={}, argv=[], input_fn=lambda _p: answer
            )
            self.assertFalse(decision, answer)

    def test_no_tty_defaults_to_no_build(self):
        self.assertFalse(
            resolve_build_decision(stdin_isatty=False, env={}, argv=[])
        )

    def test_no_tty_env_truthy_builds(self):
        for value in ("1", "true", "TRUE", "yes", " Yes "):
            self.assertTrue(
                resolve_build_decision(
                    stdin_isatty=False,
                    env={"OPEN_JAYWALKER_AUTO_BUILD": value},
                    argv=[],
                ),
                value,
            )

    def test_no_tty_env_falsy_does_not_build(self):
        for value in ("0", "false", "no", "", "maybe"):
            self.assertFalse(
                resolve_build_decision(
                    stdin_isatty=False,
                    env={"OPEN_JAYWALKER_AUTO_BUILD": value},
                    argv=[],
                ),
                value,
            )

    def test_build_arg_overrides_env(self):
        self.assertTrue(
            resolve_build_decision(
                stdin_isatty=False,
                env={"OPEN_JAYWALKER_AUTO_BUILD": "0"},
                argv=["--build"],
            )
        )

    def test_no_build_arg_overrides_env(self):
        self.assertFalse(
            resolve_build_decision(
                stdin_isatty=False,
                env={"OPEN_JAYWALKER_AUTO_BUILD": "1"},
                argv=["--no-build"],
            )
        )

    def test_arg_ignored_when_tty(self):
        # A live prompt takes precedence over the toggle.
        decision = resolve_build_decision(
            stdin_isatty=True, env={}, argv=["--build"], input_fn=lambda _p: "n"
        )
        self.assertFalse(decision)


if __name__ == "__main__":
    unittest.main()

"""
Pure decision logic for the pre-build checkpoint.

Decides whether the single-entry pipeline should proceed from a written plan to
building. Kept free of bpy and I/O so it is unit-testable headless. The entry
point (pipeline/main.py) supplies the real stdin/env/argv and handles printing.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

ENV_VAR = "OPEN_JAYWALKER_AUTO_BUILD"
_TRUTHY = {"1", "true", "yes"}


def _env_truthy(value) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def _parse_build_arg(argv: Sequence[str]):
    """Return True for --build, False for --no-build, None if neither is present."""
    if "--no-build" in argv:
        return False
    if "--build" in argv:
        return True
    return None


def resolve_build_decision(
    *,
    stdin_isatty: bool,
    env: Mapping[str, str],
    argv: Sequence[str],
    input_fn: Callable[[str], str] = input,
) -> Optional[bool]:
    """Decide whether to build after the plan is written.

    Returns True (build), False (explicit don't-build), or None (undecided —
    the caller decides: e.g. a GUI confirm popup if a window exists, else stop).

    - Interactive stdin (TTY): prompt [y/N]; only y/yes (case-insensitive) builds.
    - No TTY: --build/--no-build arg wins; else the OPEN_JAYWALKER_AUTO_BUILD env
      toggle if the var is set (truthy -> True, otherwise False); else None.
    """
    if stdin_isatty:
        answer = input_fn("Continue with build? [y/N] ")
        return answer.strip().lower() in {"y", "yes"}

    arg_decision = _parse_build_arg(argv)
    if arg_decision is not None:
        return arg_decision
    if ENV_VAR in env:
        return _env_truthy(env[ENV_VAR])
    return None

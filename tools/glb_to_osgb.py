"""Convert open-jaywalker's `.glb` exports to OpenSceneGraph `.osgb` for esmini.

open-jaywalker's canonical output is `.glb` (ASAM OpenMATERIAL is glTF-based). esmini
renders via OpenSceneGraph, which ingests `.osgb`. `.osgb` is not an ASAM/OpenX format
and OpenSCENARIO does not define a model container -- it is one simulator's ingestion
detail, on the consumer side of the asset boundary. So this is an *unsupported*,
post-build convenience that shells out to the user's local `osgconv` (shipped with
OpenSceneGraph). It is deliberately NOT part of the Blender add-on / core build path.

Resolution order for the `osgconv` executable: explicit argument -> ``OSGCONV`` env var
-> ``shutil.which("osgconv")``.

Run:
    python tools/glb_to_osgb.py <glb> [<glb> ...] [--osgconv PATH] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_INSTALL_HINT = (
    "Could not find the 'osgconv' executable. Install OpenSceneGraph "
    "(https://github.com/openscenegraph/OpenSceneGraph), then either add it to PATH, "
    "set the OSGCONV environment variable, or pass --osgconv <path>."
)


def resolve_osgconv(explicit=None) -> str:
    """Resolve the osgconv executable: explicit arg -> OSGCONV env -> PATH.

    Raises FileNotFoundError with install guidance when none resolve.
    """
    candidate = explicit or os.environ.get("OSGCONV") or shutil.which("osgconv")
    if not candidate:
        raise FileNotFoundError(_INSTALL_HINT)
    return candidate


def output_path_for(glb_path, outdir=None) -> Path:
    """Derive the `.osgb` output path for a `.glb` input."""
    glb_path = Path(glb_path)
    name = glb_path.with_suffix(".osgb").name
    return (Path(outdir) / name) if outdir else glb_path.with_suffix(".osgb")


def build_command(osgconv, glb_path, osgb_path) -> list:
    """Construct the osgconv argv. Factored out so it is assertable in tests."""
    return [osgconv, str(glb_path), str(osgb_path)]


def convert(glb_path, *, osgb_path=None, outdir=None, osgconv=None, runner=subprocess.run) -> Path:
    """Convert one `.glb` to `.osgb`; return the output path.

    Raises FileNotFoundError if the input is missing or osgconv cannot be resolved.
    `runner` is injectable for testing (defaults to subprocess.run).
    """
    glb_path = Path(glb_path)
    if not glb_path.is_file():
        raise FileNotFoundError("Input .glb not found: {0}".format(glb_path))
    exe = resolve_osgconv(osgconv)
    out = Path(osgb_path) if osgb_path else output_path_for(glb_path, outdir)
    out.parent.mkdir(parents=True, exist_ok=True)
    runner(build_command(exe, glb_path, out), check=True)
    return out


def convert_many(glb_paths, *, outdir=None, osgconv=None, runner=subprocess.run) -> list:
    """Convert several `.glb` files (e.g. a crowd's per-character exports)."""
    exe = resolve_osgconv(osgconv)
    return [
        convert(g, outdir=outdir, osgconv=exe, runner=runner)
        for g in glb_paths
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert open-jaywalker .glb exports to .osgb for esmini.",
    )
    parser.add_argument("glb", nargs="+", help="one or more .glb files to convert")
    parser.add_argument("--osgconv", help="path to the osgconv executable")
    parser.add_argument("--outdir", help="directory for the .osgb outputs")
    args = parser.parse_args(argv)
    try:
        outputs = convert_many(args.glb, outdir=args.outdir, osgconv=args.osgconv)
    except FileNotFoundError as exc:
        print("error: {0}".format(exc), file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print("error: osgconv failed (exit {0})".format(exc.returncode), file=sys.stderr)
        return 1
    for out in outputs:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Assemble the Open Jaywalker Blender add-on into an installable .zip.

Copies the pipeline packages from src/ plus the addon/ UI sources into a single
`open_jaywalker/` package and zips it to dist/. src/ stays the source of truth.

Run: python tools/build_addon.py
"""

from __future__ import annotations

import ast
import shutil
import zipfile
from pathlib import Path

PACKAGE_NAME = "open_jaywalker"
_SRC_PACKAGES = ("armature_inspector", "phase3_classifier", "asam_human_builder", "pipeline")
_SRC_MODULES = ("pipeline_paths.py",)


def _ignore_pycache(_dir, names):
    return [n for n in names if n == "__pycache__" or n.endswith(".pyc")]


def _read_version(init_path: Path) -> str:
    tree = ast.parse(Path(init_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "bl_info" for t in node.targets
        ):
            info = ast.literal_eval(node.value)
            return ".".join(str(p) for p in info.get("version", (0, 0, 0)))
    return "0.0.0"


def build_addon(repo_root, dist_dir=None) -> Path:
    repo_root = Path(repo_root).resolve()
    src = repo_root / "src"
    addon = repo_root / "addon"
    dist_dir = Path(dist_dir).resolve() if dist_dir else repo_root / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    staging = dist_dir / "_staging"
    pkg = staging / PACKAGE_NAME

    if staging.exists():
        shutil.rmtree(staging)
    pkg.mkdir(parents=True)

    # Add-on UI sources at the package root.
    for item in sorted(addon.iterdir()):
        if item.is_file() and item.suffix == ".py":
            shutil.copy2(item, pkg / item.name)

    # Bundled pipeline packages + loose modules from src/.
    for name in _SRC_PACKAGES:
        shutil.copytree(src / name, pkg / name, ignore=_ignore_pycache)
    for name in _SRC_MODULES:
        shutil.copy2(src / name, pkg / name)

    version = _read_version(addon / "__init__.py")
    zip_path = dist_dir / "{0}-{1}.zip".format(PACKAGE_NAME, version)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(pkg.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staging))

    shutil.rmtree(staging)
    return zip_path


if __name__ == "__main__":
    out = build_addon(Path(__file__).resolve().parents[1])
    print("Built add-on: {0}".format(out))

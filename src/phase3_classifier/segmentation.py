"""
Per-character ("crowd") armature decomposition for the Phase 3 classifier.

Pure Python: operates on already-loaded inspector JSON dicts. No bpy.
"""

from __future__ import annotations

import re
from typing import Optional

# Leading run of letters followed by digits, e.g. "Female000", "Male060".
_PREFIX_RE = re.compile(r"^([A-Za-z]+\d+)")
# Trailing unique-id suffix, e.g. "_093" in "Pelvis_093".
_ID_SUFFIX_RE = re.compile(r"_\d+$")


def _strip_namespace(bone_name: str) -> str:
    return bone_name.split(":")[-1]


def derive_character_prefix(bone_name: str) -> Optional[str]:
    """Return the per-character prefix (e.g. 'Female000') or None if absent."""
    match = _PREFIX_RE.match(_strip_namespace(bone_name))
    return match.group(1) if match else None


def strip_character_prefix(bone_name: str, prefix: str) -> str:
    """Remove the namespace, the character prefix, and a trailing _<id> suffix.

    'Female000Pelvis_093' + 'Female000' -> 'Pelvis'.
    """
    stripped = _strip_namespace(bone_name)
    if stripped.startswith(prefix):
        stripped = stripped[len(prefix):]
    return _ID_SUFFIX_RE.sub("", stripped)

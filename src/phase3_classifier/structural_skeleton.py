"""
Name-free structural skeleton inference for the Phase 3 classifier.

Reconstructs root / spine / limb chains and per-bone anatomical roles from bone
topology and geometry alone (no bone-name keywords). Consumed by the classifier
to provide a `structural_evidence` channel that survives junk-named rigs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class StructuralLabel:
    bone_name: str
    family: str
    side: Optional[str]
    confidence: float
    position: Optional[str] = None


@dataclass
class StructuralSkeleton:
    labels: Dict[str, StructuralLabel] = field(default_factory=dict)
    spine_chain: List[str] = field(default_factory=list)
    arm_chains: Dict[str, List[str]] = field(default_factory=lambda: {"left": [], "right": []})
    leg_chains: Dict[str, List[str]] = field(default_factory=lambda: {"left": [], "right": []})
    root_bone: Optional[str] = None


# ---------------------------------------------------------------------------
# A3: find_root
# ---------------------------------------------------------------------------

def _subtree_size(bones, name):
    seen = set()
    stack = [name]
    while stack:
        current = stack.pop()
        if current in seen or current not in bones:
            continue
        seen.add(current)
        stack.extend(bones[current].child_names)
    return len(seen)


def find_root(bones):
    """The body-spanning parentless bone (largest subtree; name tiebreak)."""
    roots = [name for name, bone in bones.items() if bone.parent is None]
    if not roots:
        return None
    return max(roots, key=lambda name: (_subtree_size(bones, name), name))


# ---------------------------------------------------------------------------
# A4: extract_trunk — central ascending chain
# ---------------------------------------------------------------------------

def _lateral_offset(bone, axes):
    return abs(bone.midpoint[axes["lateral_axis"]] - axes["centerline"])


def _height_of(bone, axes):
    h = bone.midpoint[axes["height_axis"]]
    return h if axes["height_sign"] >= 0 else -h


def extract_trunk(bones, root_name, axes):
    """Central ascending chain from the root: at each node follow the most
    central child that does not descend below the current node."""
    if root_name is None or root_name not in bones:
        return []
    trunk = [root_name]
    current = bones[root_name]
    visited = {root_name}
    while True:
        candidates = [
            bones[c] for c in current.child_names
            if c in bones and c not in visited
        ]
        if not candidates:
            break
        # central children that ascend (>= current height minus small tolerance)
        cur_h = _height_of(current, axes)
        tol = 0.02 * axes["height_span"]
        ascending = [b for b in candidates if _height_of(b, axes) >= cur_h - tol]
        pool = ascending or candidates
        nxt = min(pool, key=lambda b: (_lateral_offset(b, axes), b.name))
        # Stop if the chosen child is clearly lateral (a limb), not trunk.
        if _lateral_offset(nxt, axes) > 0.15 * axes["lateral_span"]:
            break
        # If multiple candidates exist and they branch symmetrically (like eyes),
        # the current node is the terminal trunk node; stop here.
        # Only stop if the chosen child is ITSELF lateral (not a central continuation
        # alongside symmetric branches like clavicles flanking a neck).
        if len(candidates) > 1 and _lateral_offset(nxt, axes) > 1e-6:
            sides_seen = set()
            for c in candidates:
                off = c.midpoint[axes["lateral_axis"]] - axes["centerline"]
                s = _side_of_offset(off, axes)
                if s is not None:
                    sides_seen.add(s)
            if "left" in sides_seen and "right" in sides_seen:
                break
        trunk.append(nxt.name)
        visited.add(nxt.name)
        current = nxt
    return trunk


# ---------------------------------------------------------------------------
# A5: symmetric_branch_chains — find mirrored limb pairs off a trunk node
# ---------------------------------------------------------------------------

def _linear_walk(bones, start_name, axes):
    """Follow the single longest-consistent child run from start_name."""
    chain = [start_name]
    current = bones[start_name]
    visited = {start_name}
    while True:
        kids = [bones[c] for c in current.child_names if c in bones and c not in visited]
        if len(kids) != 1:
            break
        nxt = kids[0]
        chain.append(nxt.name)
        visited.add(nxt.name)
        current = nxt
    return chain


def _side_of_offset(offset, axes):
    if abs(offset) <= 1e-9:
        return None
    if offset * axes["side_signs"]["left"] > 0:
        return "left"
    return "right"


def symmetric_branch_chains(bones, node_name, axes, exclude=frozenset()):
    """Mirrored pair of limb chains branching off node_name."""
    node = bones[node_name]
    out = {"left": [], "right": []}
    starts = [c for c in node.child_names if c in bones and c not in exclude]
    for start in starts:
        bone = bones[start]
        offset = bone.midpoint[axes["lateral_axis"]] - axes["centerline"]
        side = _side_of_offset(offset, axes)
        if side is None:
            continue
        chain = _linear_walk(bones, start, axes)
        if len(chain) > len(out[side]):
            out[side] = chain
    # Only accept as a symmetric pair when both sides found something.
    if out["left"] and out["right"]:
        return out
    return {"left": [], "right": []}


# ---------------------------------------------------------------------------
# A6: find_branch_points — hip pivot and shoulder girdle on the trunk
# ---------------------------------------------------------------------------

def find_branch_points(bones, trunk, axes):
    """(hip_index, shoulder_index) into trunk, or None for missing ones."""
    hip_index = None
    shoulder_index = None
    for i, name in enumerate(trunk):
        exclude = {trunk[i + 1]} if i + 1 < len(trunk) else set()
        pair = symmetric_branch_chains(bones, name, axes, exclude=exclude)
        if not (pair["left"] and pair["right"]):
            continue
        # Skip tiny branches (single-bone, like eyes) — they aren't limbs.
        if len(pair["left"]) < 2 and len(pair["right"]) < 2:
            continue
        # Classify branch direction: primarily vertical (legs) vs lateral (arms).
        # Use the left chain's tip as representative.
        node = bones[name]
        tip = bones[pair["left"][-1]]
        vertical_drop = abs(_height_of(node, axes) - _height_of(tip, axes))
        lateral_spread = abs(
            tip.midpoint[axes["lateral_axis"]] - node.midpoint[axes["lateral_axis"]]
        )
        primarily_vertical = vertical_drop > lateral_spread
        if primarily_vertical and hip_index is None:
            hip_index = i
        if not primarily_vertical:
            shoulder_index = i
    return hip_index, shoulder_index


# ---------------------------------------------------------------------------
# A7: segment_limb and segment_spine
# ---------------------------------------------------------------------------

def segment_limb(chain, kind):
    """Assign (family, position) to each bone of a limb chain.

    kind is "arm" or "leg". A 4-bone arm leads with a shoulder/clavicle.
    The last bone is the hand/foot; the first non-terminal is the upper
    segment; the remainder are the lower segment.
    """
    end_family = "hand" if kind == "arm" else "foot"
    upper_family = "upper_arm" if kind == "arm" else "upper_leg"
    lower_family = "lower_arm" if kind == "arm" else "lower_leg"

    result = {}
    n = len(chain)
    if n == 0:
        return result
    if n == 1:
        result[chain[0]] = (end_family, "end")
        return result

    body = list(chain)
    # Optional leading shoulder for arms when there are >= 4 segments.
    if kind == "arm" and n >= 4:
        result[body[0]] = ("shoulder", "start")
        body = body[1:]

    result[body[-1]] = (end_family, "end")
    result[body[0]] = (upper_family, "start")
    for mid in body[1:-1]:
        result[mid] = (lower_family, "middle")
    return result


def segment_spine(chain):
    """Lower half of the spine chain -> lower_spine, upper half -> upper_spine."""
    result = {}
    n = len(chain)
    for i, name in enumerate(chain):
        if n == 1:
            result[name] = "lower_spine"
        elif i < n / 2.0:
            result[name] = "lower_spine"
        else:
            result[name] = "upper_spine"
    return result


# ---------------------------------------------------------------------------
# A8: infer_structural_skeleton orchestrator
# ---------------------------------------------------------------------------

def _assign_side(bones, name, axes):
    offset = bones[name].midpoint[axes["lateral_axis"]] - axes["centerline"]
    return _side_of_offset(offset, axes)


def infer_structural_skeleton(
    bones,
    *,
    height_axis,
    lateral_axis,
    forward_axis,
    height_sign,
    centerline,
    side_signs,
    lateral_span,
    height_span,
):
    axes = dict(
        height_axis=height_axis, lateral_axis=lateral_axis,
        forward_axis=forward_axis, height_sign=height_sign,
        centerline=centerline, side_signs=side_signs,
        lateral_span=lateral_span, height_span=height_span,
    )
    skel = StructuralSkeleton()
    root = find_root(bones)
    skel.root_bone = root
    if root is None:
        return skel

    labels = {}
    labels[root] = StructuralLabel(root, "root", None, 0.6)

    trunk = extract_trunk(bones, root, axes)
    hip_i, sh_i = find_branch_points(bones, trunk, axes)

    # Hip + spine.
    if hip_i is not None:
        hip_name = trunk[hip_i]
        labels[hip_name] = StructuralLabel(hip_name, "hip", None, 0.7)
        spine_lo = hip_i + 1
        spine_hi = sh_i if sh_i is not None else len(trunk) - 1
        spine_chain = trunk[spine_lo:spine_hi + 1] if sh_i is not None else trunk[spine_lo:]
        # Exclude neck/head terminal from the spine (handled below).
        skel.spine_chain = list(spine_chain)
        for name, fam in segment_spine(skel.spine_chain).items():
            labels[name] = StructuralLabel(name, fam, None, 0.6,
                                           "start" if fam == "lower_spine" else "end")
        # Legs branch from the hip node.
        nxt = {trunk[hip_i + 1]} if hip_i + 1 < len(trunk) else set()
        legs = symmetric_branch_chains(bones, hip_name, axes, exclude=nxt)
        skel.leg_chains = legs
        for side in ("left", "right"):
            for name, (fam, pos) in segment_limb(legs[side], "leg").items():
                labels[name] = StructuralLabel(name, fam, side, 0.6, pos)

    # Arms + neck/head from shoulder girdle upward.
    if sh_i is not None:
        sh_name = trunk[sh_i]
        nxt = {trunk[sh_i + 1]} if sh_i + 1 < len(trunk) else set()
        arms = symmetric_branch_chains(bones, sh_name, axes, exclude=nxt)
        skel.arm_chains = arms
        for side in ("left", "right"):
            for name, (fam, pos) in segment_limb(arms[side], "arm").items():
                labels[name] = StructuralLabel(name, fam, side, 0.6, pos)
        # Neck/head = trunk above the shoulder node.
        above = trunk[sh_i + 1:]
        if above:
            labels[above[-1]] = StructuralLabel(above[-1], "head", None, 0.6)
            for name in above[:-1]:
                labels[name] = StructuralLabel(name, "neck", None, 0.55)
            # Eyes: small children of head, symmetric.
            head_name = above[-1]
            for child in bones[head_name].child_names:
                if child in bones:
                    labels[child] = StructuralLabel(
                        child, "eye", _assign_side(bones, child, axes), 0.5
                    )

    skel.labels = labels
    return skel

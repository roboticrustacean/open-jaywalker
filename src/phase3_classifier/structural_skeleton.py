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


def _forest_roots(bones):
    """Every bone that starts a tree: no parent, or a parent outside the set.

    Real generated rigs frequently have several parentless control bones (IK
    targets, ``MCH-*.parent`` helpers) alongside the true skeleton root, and may
    reference a parent that was filtered out of the bone set. Both are forest
    roots for our purposes.
    """
    return [
        name
        for name, bone in bones.items()
        if bone.parent is None or bone.parent not in bones
    ]


def find_root(bones):
    """The body-spanning forest root (largest subtree; name tiebreak)."""
    roots = _forest_roots(bones)
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
        if len(candidates) > 1 and _lateral_offset(nxt, axes) > 0.02 * axes["lateral_span"]:
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
# A8b: geometric reconstruction — for tangled rigs where one linear trunk walk
# cannot pass through both branch points (parallel DEF/ORG/MCH chains, limbs
# parented onto sibling control chains). Name-free: topology + geometry only.
# ---------------------------------------------------------------------------

def _is_central(bones, name, axes):
    return _lateral_offset(bones[name], axes) <= 0.03 * axes["lateral_span"]


def _longest_central_chain(bones, axes):
    """The longest topologically-connected run of near-centerline bones.

    Co-located duplicate spines (DEF / ORG / MCH overlays at identical
    positions) are disambiguated by preferring the chain that is actually a
    connected parent->child run — the deform spine is one such run while the
    control overlays are scattered. Returns bone names ordered low->high.
    """
    central = [n for n in bones if _is_central(bones, n, axes)]
    central_set = set(central)
    best = []
    for start in central:
        # Only begin a chain at a bone whose parent is not itself central, so we
        # measure each maximal run once from its base.
        parent = bones[start].parent
        if parent in central_set:
            continue
        chain = [start]
        current = start
        while True:
            kids = [
                c for c in bones[current].child_names
                if c in central_set
            ]
            if not kids:
                break
            # Continue along the most central, ascending child.
            kids.sort(key=lambda c: (_lateral_offset(bones[c], axes),
                                     -_height_of(bones[c], axes), c))
            nxt = kids[0]
            if _height_of(bones[nxt], axes) < _height_of(bones[current], axes):
                break
            chain.append(nxt)
            current = nxt
        if len(chain) > len(best):
            best = chain
    best.sort(key=lambda n: _height_of(bones[n], axes))
    return best


def _limb_tip_chains(bones, axes):
    """Every off-centerline limb chain, classified by gross shape.

    Walks up from each lateral childless tip toward the centerline; the run from
    the last off-centerline ancestor down to the tip is one limb chain. Each is
    tagged ``"leg"`` (vertical drop dominates) or ``"arm"`` (lateral spread
    dominates) with its lateral side. Returns ``[(kind, side, chain), ...]``.

    Rigs duplicate limbs many times over (DEF / FK / IK / tweak overlays); we
    keep *all* of them so the structural channel votes for every co-located
    bone, leaving the name channel to pick the canonical one downstream.
    """
    results = []
    for name, bone in bones.items():
        if [c for c in bone.child_names if c in bones]:
            continue  # not a tip
        if _lateral_offset(bone, axes) <= 0.06 * axes["lateral_span"]:
            continue  # too central to be a limb tip
        chain_up = [name]
        cur = bone.parent
        seen = {name}
        while cur in bones and cur not in seen and not _is_central(bones, cur, axes):
            chain_up.append(cur)
            seen.add(cur)
            cur = bones[cur].parent
        chain = list(reversed(chain_up))  # root (near body) -> tip
        if len(chain) < 2:
            continue
        side = _assign_side(bones, chain[0], axes)
        if side is None:
            continue
        root, tip = bones[chain[0]], bones[chain[-1]]
        drop = _height_of(root, axes) - _height_of(tip, axes)
        spread = _lateral_offset(tip, axes) - _lateral_offset(root, axes)
        # A real limb travels outward and/or downward; reject near-stationary
        # control clusters (heel rolls, IK pivots) that neither drop nor spread.
        if drop < 0.05 * axes["height_span"] and spread < 0.05 * axes["lateral_span"]:
            continue
        kind = "leg" if drop > spread else "arm"
        results.append((kind, side, chain))
    return results


def _best_pair(chains_by_side):
    """Pick the longest mirrored pair from {'left': [...chains], 'right': ...}."""
    out = {"left": [], "right": []}
    for side in ("left", "right"):
        for chain in chains_by_side[side]:
            if len(chain) > len(out[side]):
                out[side] = chain
    if out["left"] and out["right"]:
        return out
    return {"left": [], "right": []}


def _nearest_central_index(bones, central_chain, target_name, axes):
    """Index into central_chain of the bone closest in height to target."""
    if not central_chain:
        return None
    th = _height_of(bones[target_name], axes)
    return min(
        range(len(central_chain)),
        key=lambda i: abs(_height_of(bones[central_chain[i]], axes) - th),
    )


def _reconstruct_geometric(bones, axes):
    """Geometric body reconstruction for tangled rigs. Returns a populated
    StructuralSkeleton or None if no plausible body is found."""
    central = _longest_central_chain(bones, axes)
    if len(central) < 2:
        return None

    limbs = _limb_tip_chains(bones, axes)
    leg_by_side = {"left": [], "right": []}
    arm_by_side = {"left": [], "right": []}
    for kind, side, chain in limbs:
        (leg_by_side if kind == "leg" else arm_by_side)[side].append(chain)

    legs = _best_pair(leg_by_side)
    arms = _best_pair(arm_by_side)
    if not (legs["left"] and legs["right"]):
        return None

    skel = StructuralSkeleton()
    root = find_root(bones)
    skel.root_bone = root
    labels = {}
    if root is not None:
        labels[root] = StructuralLabel(root, "root", None, 0.6)

    # Hip = central bone nearest the (canonical) leg roots' height.
    hip_i = _nearest_central_index(bones, central, legs["left"][0], axes)
    hip_name = central[hip_i]
    labels[hip_name] = StructuralLabel(hip_name, "hip", None, 0.7)

    # Head = topmost central terminal; neck = central bones in the top height
    # band just below it. The spine runs the full central column between the hip
    # and the neck — it does NOT stop at the shoulder girdle (Rigify-style rigs
    # carry several spine segments above the arm branch before the neck begins).
    head_i = len(central) - 1
    top_h = _height_of(bones[central[head_i]], axes)
    neck_band = 0.08 * axes["height_span"]
    neck_lo = head_i
    while neck_lo - 1 > hip_i and \
            top_h - _height_of(bones[central[neck_lo - 1]], axes) <= neck_band:
        neck_lo -= 1
    neck_lo = max(neck_lo, hip_i + 1)

    spine_chain = central[hip_i + 1:neck_lo]
    skel.spine_chain = list(spine_chain)
    for name, fam in segment_spine(skel.spine_chain).items():
        labels[name] = StructuralLabel(name, fam, None, 0.6,
                                       "start" if fam == "lower_spine" else "end")

    # Label EVERY co-located limb overlay so the structural channel votes for
    # all candidates (DEF / FK / IK twins), not just one. The canonical pair is
    # surfaced on the skeleton for downstream chain consumers.
    skel.leg_chains = legs
    skel.arm_chains = arms
    for kind, by_side in (("leg", leg_by_side), ("arm", arm_by_side)):
        for side in ("left", "right"):
            for chain in by_side[side]:
                for name, (fam, pos) in segment_limb(chain, kind).items():
                    # Don't let a limb overlay clobber a spine/hip assignment.
                    if name in labels and labels[name].family in (
                        "hip", "lower_spine", "upper_spine", "root"
                    ):
                        continue
                    labels[name] = StructuralLabel(name, fam, side, 0.55, pos)

    # Neck / head = the top central band above the spine.
    labels[central[head_i]] = StructuralLabel(central[head_i], "head", None, 0.6)
    for name in central[neck_lo:head_i]:
        labels[name] = StructuralLabel(name, "neck", None, 0.55)

    skel.labels = labels
    return skel


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

    # Tangled / multi-root rigs: a single topological trunk walk cannot pass
    # through both branch points (limbs are parented onto sibling control
    # chains, parallel overlays duplicate the spine). Fall back to a geometric
    # reconstruction when the topological pass did not recover a real body.
    if not (skel.spine_chain and skel.leg_chains["left"] and skel.leg_chains["right"]):
        geometric = _reconstruct_geometric(bones, axes)
        if geometric is not None:
            return geometric

    return skel

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase3_classifier.structural_skeleton import (  # noqa: E402
    StructuralLabel,
    StructuralSkeleton,
)


class DataclassShapeTests(unittest.TestCase):
    def test_structural_label_fields(self):
        label = StructuralLabel(
            bone_name="b", family="hip", side=None, confidence=0.5, position="start"
        )
        self.assertEqual(label.bone_name, "b")
        self.assertEqual(label.family, "hip")
        self.assertIsNone(label.side)
        self.assertEqual(label.confidence, 0.5)
        self.assertEqual(label.position, "start")

    def test_structural_skeleton_defaults(self):
        skel = StructuralSkeleton()
        self.assertEqual(skel.labels, {})
        self.assertEqual(skel.spine_chain, [])
        self.assertEqual(skel.arm_chains, {"left": [], "right": []})
        self.assertEqual(skel.leg_chains, {"left": [], "right": []})
        self.assertIsNone(skel.root_bone)


class _FakeBone:
    """Duck-typed stand-in for BoneInfo: only the attributes the labeler reads."""
    def __init__(self, name, parent, head, tail, child_names):
        self.name = name
        self.parent = parent
        self.head = tuple(float(v) for v in head)
        self.tail = tuple(float(v) for v in tail)
        self.length = math.sqrt(sum((tail[i] - head[i]) ** 2 for i in range(3)))
        self.midpoint = tuple((head[i] + tail[i]) / 2.0 for i in range(3))
        self.child_names = list(child_names)


def _index(bone_defs):
    """bone_defs: list of (name, parent, head, tail). Returns {name: _FakeBone}."""
    children = {name: [] for (name, _p, _h, _t) in bone_defs}
    for name, parent, _h, _t in bone_defs:
        if parent in children:
            children[parent].append(name)
    return {
        name: _FakeBone(name, parent, head, tail, sorted(children[name]))
        for (name, parent, head, tail) in bone_defs
    }


def _upright_humanoid(prefix=""):
    """Clean Z-up humanoid. Z=height, Y=lateral (left=+Y), X=forward.
    Returns (bones_index, axes_kwargs)."""
    p = prefix
    defs = [
        (p + "root", None, (0, 0, 0.0), (0, 0, 0.1)),
        (p + "pelvis", p + "root", (0, 0, 1.0), (0, 0, 1.1)),
        (p + "spine1", p + "pelvis", (0, 0, 1.1), (0, 0, 1.4)),
        (p + "spine2", p + "spine1", (0, 0, 1.4), (0, 0, 1.7)),
        (p + "neck", p + "spine2", (0, 0, 1.7), (0, 0, 1.85)),
        (p + "head", p + "neck", (0, 0, 1.85), (0, 0, 2.0)),
        (p + "eyeL", p + "head", (0.1, 0.05, 1.95), (0.15, 0.05, 1.95)),
        (p + "eyeR", p + "head", (0.1, -0.05, 1.95), (0.15, -0.05, 1.95)),
        # left arm
        (p + "clavL", p + "spine2", (0, 0.05, 1.7), (0, 0.2, 1.7)),
        (p + "uarmL", p + "clavL", (0, 0.2, 1.7), (0, 0.5, 1.6)),
        (p + "larmL", p + "uarmL", (0, 0.5, 1.6), (0, 0.8, 1.5)),
        (p + "handL", p + "larmL", (0, 0.8, 1.5), (0, 0.95, 1.45)),
        # right arm
        (p + "clavR", p + "spine2", (0, -0.05, 1.7), (0, -0.2, 1.7)),
        (p + "uarmR", p + "clavR", (0, -0.2, 1.7), (0, -0.5, 1.6)),
        (p + "larmR", p + "uarmR", (0, -0.5, 1.6), (0, -0.8, 1.5)),
        (p + "handR", p + "larmR", (0, -0.8, 1.5), (0, -0.95, 1.45)),
        # left leg
        (p + "ulegL", p + "pelvis", (0, 0.1, 1.0), (0, 0.12, 0.55)),
        (p + "llegL", p + "ulegL", (0, 0.12, 0.55), (0, 0.13, 0.1)),
        (p + "footL", p + "llegL", (0, 0.13, 0.1), (0.15, 0.13, 0.0)),
        # right leg
        (p + "ulegR", p + "pelvis", (0, -0.1, 1.0), (0, -0.12, 0.55)),
        (p + "llegR", p + "ulegR", (0, -0.12, 0.55), (0, -0.13, 0.1)),
        (p + "footR", p + "llegR", (0, -0.13, 0.1), (0.15, -0.13, 0.0)),
    ]
    bones = _index(defs)
    axes = dict(
        height_axis=2, lateral_axis=1, forward_axis=0, height_sign=1,
        centerline=0.0, side_signs={"left": 1, "right": -1},
        lateral_span=2.0, height_span=2.0,
    )
    return bones, axes


class HelperSanityTests(unittest.TestCase):
    def test_upright_humanoid_shape(self):
        bones, _axes = _upright_humanoid()
        self.assertIsNone(bones["root"].parent)
        self.assertIn("pelvis", bones["root"].child_names)
        self.assertEqual(
            sorted(bones["pelvis"].child_names), ["spine1", "ulegL", "ulegR"]
        )


class FindRootTests(unittest.TestCase):
    def test_single_parentless_bone_is_root(self):
        from phase3_classifier.structural_skeleton import find_root
        bones, _axes = _upright_humanoid()
        self.assertEqual(find_root(bones), "root")

    def test_picks_body_spanning_subtree_when_multiple_parentless(self):
        from phase3_classifier.structural_skeleton import find_root
        bones, _axes = _upright_humanoid()
        # Add a stray parentless bone with no children (a floating prop).
        bones["prop"] = _FakeBone("prop", None, (5, 5, 5), (5, 5, 5.1), [])
        self.assertEqual(find_root(bones), "root")


class ExtractTrunkTests(unittest.TestCase):
    def test_trunk_is_central_vertical_chain(self):
        from phase3_classifier.structural_skeleton import extract_trunk
        bones, axes = _upright_humanoid()
        trunk = extract_trunk(bones, "root", axes)
        self.assertEqual(
            trunk, ["root", "pelvis", "spine1", "spine2", "neck", "head"]
        )


class SymmetricBranchTests(unittest.TestCase):
    def test_legs_branch_from_pelvis(self):
        from phase3_classifier.structural_skeleton import symmetric_branch_chains
        bones, axes = _upright_humanoid()
        pair = symmetric_branch_chains(bones, "pelvis", axes, exclude={"spine1"})
        self.assertEqual(pair["left"], ["ulegL", "llegL", "footL"])
        self.assertEqual(pair["right"], ["ulegR", "llegR", "footR"])

    def test_arms_branch_from_upper_spine(self):
        from phase3_classifier.structural_skeleton import symmetric_branch_chains
        bones, axes = _upright_humanoid()
        pair = symmetric_branch_chains(bones, "spine2", axes, exclude={"neck"})
        self.assertEqual(pair["left"], ["clavL", "uarmL", "larmL", "handL"])
        self.assertEqual(pair["right"], ["clavR", "uarmR", "larmR", "handR"])


class BranchPointTests(unittest.TestCase):
    def test_hip_and_shoulder_indices(self):
        from phase3_classifier.structural_skeleton import (
            extract_trunk, find_branch_points,
        )
        bones, axes = _upright_humanoid()
        trunk = extract_trunk(bones, "root", axes)
        hip_i, sh_i = find_branch_points(bones, trunk, axes)
        self.assertEqual(trunk[hip_i], "pelvis")
        self.assertEqual(trunk[sh_i], "spine2")


class SegmentTests(unittest.TestCase):
    def test_segment_leg_three_bones(self):
        from phase3_classifier.structural_skeleton import segment_limb
        fams = segment_limb(["ulegL", "llegL", "footL"], "leg")
        self.assertEqual(fams, {
            "ulegL": ("upper_leg", "start"),
            "llegL": ("lower_leg", "middle"),
            "footL": ("foot", "end"),
        })

    def test_segment_arm_with_clavicle(self):
        from phase3_classifier.structural_skeleton import segment_limb
        fams = segment_limb(["clavL", "uarmL", "larmL", "handL"], "arm")
        self.assertEqual(fams["clavL"], ("shoulder", "start"))
        self.assertEqual(fams["uarmL"], ("upper_arm", "start"))
        self.assertEqual(fams["larmL"], ("lower_arm", "middle"))
        self.assertEqual(fams["handL"], ("hand", "end"))

    def test_segment_spine_lower_upper(self):
        from phase3_classifier.structural_skeleton import segment_spine
        fams = segment_spine(["spine1", "spine2"])
        self.assertEqual(fams["spine1"], "lower_spine")
        self.assertEqual(fams["spine2"], "upper_spine")


class InferSkeletonTests(unittest.TestCase):
    def setUp(self):
        from phase3_classifier.structural_skeleton import infer_structural_skeleton
        self.bones, self.axes = _upright_humanoid()
        self.skel = infer_structural_skeleton(self.bones, **self.axes)

    def test_root_and_chains(self):
        self.assertEqual(self.skel.root_bone, "root")
        self.assertEqual(self.skel.spine_chain, ["spine1", "spine2"])
        self.assertEqual(self.skel.leg_chains["left"], ["ulegL", "llegL", "footL"])
        self.assertEqual(self.skel.arm_chains["right"],
                         ["clavR", "uarmR", "larmR", "handR"])

    def test_family_labels(self):
        labels = self.skel.labels
        self.assertEqual(labels["pelvis"].family, "hip")
        self.assertEqual(labels["spine1"].family, "lower_spine")
        self.assertEqual(labels["spine2"].family, "upper_spine")
        self.assertEqual(labels["neck"].family, "neck")
        self.assertEqual(labels["head"].family, "head")
        self.assertEqual(labels["handL"].family, "hand")
        self.assertEqual(labels["handL"].side, "left")
        self.assertEqual(labels["footR"].family, "foot")
        self.assertEqual(labels["footR"].side, "right")
        self.assertEqual(labels["root"].family, "root")

    def test_eyes_detected(self):
        self.assertEqual(self.skel.labels["eyeL"].family, "eye")
        self.assertEqual(self.skel.labels["eyeR"].family, "eye")
        self.assertEqual(self.skel.labels["eyeL"].side, "left")


class DegenerateRigTests(unittest.TestCase):
    def test_junk_names_same_topology_still_labels(self):
        from phase3_classifier.structural_skeleton import infer_structural_skeleton
        bones, axes = _upright_humanoid(prefix="zx9_")
        skel = infer_structural_skeleton(bones, **axes)
        self.assertEqual(skel.labels["zx9_pelvis"].family, "hip")
        self.assertEqual(skel.labels["zx9_handL"].family, "hand")
        self.assertEqual(skel.labels["zx9_footR"].family, "foot")

    def test_missing_intermediate_leg_two_bones(self):
        from phase3_classifier.structural_skeleton import segment_limb
        # 2-bone leg: upper then foot, no explicit lower.
        fams = segment_limb(["ulegL", "footL"], "leg")
        self.assertEqual(fams["ulegL"], ("upper_leg", "start"))
        self.assertEqual(fams["footL"], ("foot", "end"))


class RealRigStructuralTests(unittest.TestCase):
    """The labeler must find the real body skeleton on a bushy, multi-root,
    Rigify-generated rig (LowPolyCharacter4 `rig`, 217 bones, 10 parentless
    bones, DEF/ORG/MCH chains overlapping in space)."""

    @classmethod
    def setUpClass(cls):
        from phase3_classifier.classifier import (
            discover_armatures,
            _build_bone_index,
            _build_context,
            _detect_convention,
            _structural_evidence,
        )
        asset_dir = REPO_ROOT / "tests" / "fixtures" / "LowPolyCharacter4"
        infos = discover_armatures(asset_dir)
        info = next(x for x in infos if x.armature_name == "rig")
        convention = _detect_convention(info.primary_data, info.support_data)
        bones = _build_bone_index(info.primary_data, info.support_data, convention)
        context = _build_context(bones, info.primary_data, info.support_data, convention)
        cls.bones = bones
        cls.context = context
        cls.skel = context["structural_skeleton"]
        cls._structural_evidence = staticmethod(_structural_evidence)

    def test_non_empty_chains(self):
        self.assertTrue(self.skel.spine_chain, "spine_chain should not be empty")
        self.assertTrue(self.skel.leg_chains["left"], "left leg chain empty")
        self.assertTrue(self.skel.leg_chains["right"], "right leg chain empty")

    def test_known_core_bones_get_correct_families(self):
        labels = self.skel.labels
        self.assertEqual(labels["DEF-spine"].family, "hip")
        self.assertEqual(labels["DEF-spine.001"].family, "lower_spine")
        self.assertEqual(labels["DEF-thigh.L"].family, "upper_leg")
        self.assertEqual(labels["DEF-thigh.L"].side, "left")
        self.assertEqual(labels["DEF-hand.L"].family, "hand")
        self.assertEqual(labels["DEF-hand.L"].side, "left")

    def test_structural_evidence_positive_for_core_targets(self):
        ev = type(self)._structural_evidence
        self.assertGreater(ev("Hip", self.bones["DEF-spine"], self.context), 0.0)
        self.assertGreater(
            ev("Lower_Spine", self.bones["DEF-spine.001"], self.context), 0.0
        )
        self.assertGreater(
            ev("Upper_Leg_Left", self.bones["DEF-thigh.L"], self.context), 0.0
        )
        self.assertGreater(ev("Hand_Left", self.bones["DEF-hand.L"], self.context), 0.0)


if __name__ == "__main__":
    unittest.main()

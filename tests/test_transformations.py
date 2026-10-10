"""
Tests for the WCE Transformation Algebra.

This module tests the new abstractions:
- TransformationSpec
- InvariantSet
- CompositeTransformation
- MacroTransformation
- TransformGraph
- TransformNode
"""

import unittest
import sys
import os
from math import pi, isclose

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.invariants import (
    Invariant,
    InvariantSet,
    IDENTITY_INVARIANTS,
    TRANSLATION_INVARIANTS,
    ROTATION_INVARIANTS,
    UNIFORM_SCALE_INVARIANTS,
    NON_UNIFORM_SCALE_INVARIANTS,
    REFLECTION_INVARIANTS,
    SHEAR_INVARIANTS,
    STRETCH_INVARIANTS,
    TOPOLOGY_PRESERVING_INVARIANTS,
    ISOMETRY_INVARIANTS,
    CUT_INVARIANTS,
    MERGE_INVARIANTS,
)


class TestInvariantSet(unittest.TestCase):
    """Tests for InvariantSet."""
    
    def test_preserves_single(self):
        """Test preservation of a single invariant."""
        inv_set = InvariantSet({Invariant.DISTANCE, Invariant.ANGLE})
        self.assertTrue(inv_set.preserves(Invariant.DISTANCE))
        self.assertTrue(inv_set.preserves(Invariant.ANGLE))
        self.assertFalse(inv_set.preserves(Invariant.ORIENTATION))
    
    def test_preserves_all(self):
        """Test preservation of several invariants."""
        inv_set = InvariantSet({Invariant.DISTANCE, Invariant.ANGLE, Invariant.ORIENTATION})
        self.assertTrue(inv_set.preserves_all({Invariant.DISTANCE, Invariant.ANGLE}))
        self.assertFalse(inv_set.preserves_all({Invariant.DISTANCE, Invariant.MASS}))
    
    def test_preserves_any(self):
        """Test preservation of at least one invariant."""
        inv_set = InvariantSet({Invariant.DISTANCE, Invariant.ANGLE})
        self.assertTrue(inv_set.preserves_any({Invariant.DISTANCE, Invariant.MASS}))
        self.assertTrue(inv_set.preserves_any({Invariant.ANGLE, Invariant.MASS}))
        self.assertFalse(inv_set.preserves_any({Invariant.MASS, Invariant.TIME}))
    
    def test_union(self):
        """Test the union of invariant sets."""
        set1 = InvariantSet({Invariant.DISTANCE})
        set2 = InvariantSet({Invariant.ANGLE})
        union = set1.union(set2)
        self.assertTrue(union.preserves(Invariant.DISTANCE))
        self.assertTrue(union.preserves(Invariant.ANGLE))
    
    def test_intersection(self):
        """Test the intersection of invariant sets."""
        set1 = InvariantSet({Invariant.DISTANCE, Invariant.ANGLE})
        set2 = InvariantSet({Invariant.DISTANCE, Invariant.ORIENTATION})
        intersection = set1.intersection(set2)
        self.assertTrue(intersection.preserves(Invariant.DISTANCE))
        self.assertFalse(intersection.preserves(Invariant.ANGLE))
        self.assertFalse(intersection.preserves(Invariant.ORIENTATION))
    
    def test_lost(self):
        """Test the lost invariants."""
        original = InvariantSet({Invariant.DISTANCE, Invariant.ANGLE, Invariant.ORIENTATION})
        current = InvariantSet({Invariant.ANGLE, Invariant.ORIENTATION})
        lost = current.lost(original)
        self.assertTrue(lost.preserves(Invariant.DISTANCE))
        self.assertFalse(lost.preserves(Invariant.ANGLE))
    
    def test_preserves_geometry(self):
        """Test the preserves_geometry property."""
        # Translation preserves geometry
        self.assertTrue(TRANSLATION_INVARIANTS.preserves_geometry)
        # Rotation preserves geometry
        self.assertTrue(ROTATION_INVARIANTS.preserves_geometry)
        # Non-uniform scale does not preserve geometry
        self.assertFalse(NON_UNIFORM_SCALE_INVARIANTS.preserves_geometry)
    
    def test_preserves_topology(self):
        """Test the preserves_topology property."""
        # All these transformations preserve topology
        self.assertTrue(TRANSLATION_INVARIANTS.preserves_topology)
        self.assertTrue(ROTATION_INVARIANTS.preserves_topology)
        self.assertTrue(UNIFORM_SCALE_INVARIANTS.preserves_topology)
        self.assertTrue(NON_UNIFORM_SCALE_INVARIANTS.preserves_topology)


class TestTransformationSpec(unittest.TestCase):
    """Tests for TransformationSpec and its subclasses."""
    
    def test_identity_transformation(self):
        """Test the identity transformation."""
        from core.transformations import IdentityTransformation
        
        identity = IdentityTransformation()
        self.assertEqual(identity.name, "identity")
        self.assertTrue(identity.invertible)
        self.assertEqual(identity.apply((1, 2)), (1, 2))
        self.assertEqual(identity.apply(42), 42)
        
        # Inverse
        inv = identity.get_inverse()
        self.assertIsNotNone(inv)
        self.assertIsInstance(inv, IdentityTransformation)
    
    def test_identity_preserves_all(self):
        """Test that the identity preserves all invariants."""
        from core.transformations import IdentityTransformation
        identity = IdentityTransformation()
        # IDENTITY_INVARIANTS contains many invariants
        self.assertTrue(identity.preserves_geometry())
        self.assertTrue(identity.preserves_topology)


class TestTranslationTransformation(unittest.TestCase):
    """Tests for translation transformations."""
    
    def test_create_translation(self):
        """Test creation of a translation."""
        from core.transformations import create_translation_transformation
        
        t = create_translation_transformation(2.0, 3.0)
        self.assertEqual(t.name, "translate(2.0, 3.0)")
        self.assertTrue(t.invertible)
    
    def test_translation_apply(self):
        """Test application of a translation."""
        from core.transformations import create_translation_transformation
        
        t = create_translation_transformation(2.0, 3.0)
        self.assertEqual(t.apply((0, 0)), (2.0, 3.0))
        self.assertEqual(t.apply((1, 1)), (3.0, 4.0))
        self.assertEqual(t.apply((-1, -1)), (1.0, 2.0))
    
    def test_translation_inverse(self):
        """Test the inverse of a translation."""
        from core.transformations import create_translation_transformation
        
        t = create_translation_transformation(2.0, 3.0)
        inv = t.get_inverse()
        self.assertIsNotNone(inv)
        self.assertEqual(inv.name, "translate(-2.0, -3.0)")
        
        # Check that inv(t(p)) = p
        p = (5.0, 7.0)
        tp = t.apply(p)
        tpinv = inv.apply(tp)
        self.assertAlmostEqual(tpinv[0], p[0])
        self.assertAlmostEqual(tpinv[1], p[1])
    
    def test_translation_preserves(self):
        """Test the invariants preserved by a translation."""
        from core.transformations import create_translation_transformation
        
        t = create_translation_transformation(2.0, 0.0)
        self.assertTrue(t.preserves_geometry())
        self.assertTrue(t.preserves_topology)
        self.assertTrue(t.preserves_invariant(Invariant.DISTANCE))
        self.assertTrue(t.preserves_invariant(Invariant.ANGLE))


class TestRotationTransformation(unittest.TestCase):
    """Tests for rotation transformations."""
    
    def test_create_rotation(self):
        """Test creation of a rotation."""
        from core.transformations import create_rotation_transformation
        
        r = create_rotation_transformation(pi/2)
        self.assertIn("rotate", r.name)
        self.assertTrue(r.invertible)
    
    def test_rotation_90_degrees(self):
        """Test a 90-degree rotation."""
        from core.transformations import create_rotation_transformation
        
        r = create_rotation_transformation(pi/2)
        # (1, 0) rotated by 90° -> (0, 1)
        result = r.apply((1.0, 0.0))
        self.assertAlmostEqual(result[0], 0.0, places=5)
        self.assertAlmostEqual(result[1], 1.0, places=5)
    
    def test_rotation_inverse(self):
        """Test the inverse of a rotation."""
        from core.transformations import create_rotation_transformation
        
        r = create_rotation_transformation(pi/4)
        inv = r.get_inverse()
        self.assertIsNotNone(inv)
        
        # Check that inv(r(p)) = p
        p = (1.0, 0.0)
        rp = r.apply(p)
        rpinv = inv.apply(rp)
        self.assertAlmostEqual(rpinv[0], p[0], places=5)
        self.assertAlmostEqual(rpinv[1], p[1], places=5)
    
    def test_rotation_preserves(self):
        """Test the invariants preserved by a rotation."""
        from core.transformations import create_rotation_transformation
        
        r = create_rotation_transformation(pi/2)
        self.assertTrue(r.preserves_geometry())
        self.assertTrue(r.preserves_topology)
        self.assertTrue(r.preserves_invariant(Invariant.DISTANCE))
        self.assertTrue(r.preserves_invariant(Invariant.ANGLE))


class TestScaleTransformation(unittest.TestCase):
    """Tests for scale transformations."""
    
    def test_create_uniform_scale(self):
        """Test creation of a uniform scale."""
        from core.transformations import create_scale_transformation
        
        s = create_scale_transformation(2.0, 2.0)
        self.assertEqual(s.name, "scale(2.0, 2.0)")
        self.assertTrue(s.invertible)
    
    def test_create_non_uniform_scale(self):
        """Test creation of a non-uniform scale."""
        from core.transformations import create_scale_transformation
        
        s = create_scale_transformation(2.0, 3.0)
        self.assertEqual(s.name, "scale(2.0, 3.0)")
        self.assertTrue(s.invertible)
    
    def test_scale_apply(self):
        """Test application of a scale."""
        from core.transformations import create_scale_transformation
        
        s = create_scale_transformation(2.0, 3.0)
        self.assertEqual(s.apply((1.0, 1.0)), (2.0, 3.0))
        self.assertEqual(s.apply((-1.0, 2.0)), (-2.0, 6.0))
    
    def test_scale_inverse(self):
        """Test the inverse of a scale."""
        from core.transformations import create_scale_transformation
        
        s = create_scale_transformation(2.0, 3.0)
        inv = s.get_inverse()
        self.assertIsNotNone(inv)
        
        # Check that inv(s(p)) = p
        p = (4.0, 6.0)
        sp = s.apply(p)
        spinv = inv.apply(sp)
        self.assertAlmostEqual(spinv[0], p[0], places=5)
        self.assertAlmostEqual(spinv[1], p[1], places=5)
    
    def test_uniform_scale_preserves_angles(self):
        """Test that a uniform scale preserves angles."""
        from core.transformations import create_scale_transformation
        
        s = create_scale_transformation(2.0, 2.0)
        self.assertTrue(s.preserves_invariant(Invariant.ANGLE))
        self.assertTrue(s.preserves_invariant(Invariant.ORIENTATION))
    
    def test_non_uniform_scale_preserves_topology(self):
        """Test that a non-uniform scale preserves topology."""
        from core.transformations import create_scale_transformation
        
        s = create_scale_transformation(2.0, 3.0)
        self.assertTrue(s.preserves_topology)
        self.assertFalse(s.preserves_invariant(Invariant.ANGLE))


class TestCompositeTransformation(unittest.TestCase):
    """Tests for CompositeTransformation."""
    
    def test_compose_two(self):
        """Test composition of two transformations."""
        from core.transformations import (
            create_translation_transformation,
            create_rotation_transformation,
            CompositeTransformation,
        )
        
        t = create_translation_transformation(1.0, 0.0)
        r = create_rotation_transformation(0.0)
        composed = t.compose(r)
        
        self.assertIsInstance(composed, CompositeTransformation)
        self.assertEqual(composed.get_depth(), 2)
    
    def test_matmul_operator(self):
        """Test the @ operator for composition."""
        from core.transformations import (
            create_translation_transformation,
            create_rotation_transformation,
        )
        
        t = create_translation_transformation(1.0, 0.0)
        r = create_rotation_transformation(pi/2)
        composed = t @ r
        
        self.assertEqual(composed.get_depth(), 2)
    
    def test_compose_order(self):
        """Test that the composition order is correct."""
        from core.transformations import create_translation_transformation
        
        t1 = create_translation_transformation(1.0, 0.0)
        t2 = create_translation_transformation(0.0, 2.0)
        
        # t1 @ t2 means t2 ∘ t1, so t1 is applied first
        composed = t1 @ t2
        
        # Apply to (0, 0): t1 -> (1, 0), t2 -> (1, 2)
        result = composed.apply((0.0, 0.0))
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], 2.0)
    
    def test_invariants_intersection(self):
        """Test that the composed invariants are the intersection."""
        from core.transformations import (
            create_translation_transformation,
            create_scale_transformation,
        )
        
        t = create_translation_transformation(1.0, 0.0)
        s = create_scale_transformation(2.0, 3.0)
        
        composed = t @ s
        
        # Translation preserves geometry, non-uniform scale does not
        # So the composition does not preserve geometry
        self.assertFalse(composed.preserves_geometry())
        
        # But both preserve topology
        self.assertTrue(composed.preserves_topology)
    
    def test_composition_cost(self):
        """Test that the cost is cumulative."""
        from core.transformations import create_translation_transformation
        
        t1 = create_translation_transformation(1.0, 0.0)
        t2 = create_translation_transformation(0.0, 1.0)
        
        composed = t1 @ t2
        self.assertAlmostEqual(composed.cost, t1.cost + t2.cost)
    
    def test_invertible_composition(self):
        """Test that a composition of invertible transformations is invertible."""
        from core.transformations import create_translation_transformation
        
        t1 = create_translation_transformation(1.0, 0.0)
        t2 = create_translation_transformation(0.0, 1.0)
        
        composed = t1 @ t2
        self.assertTrue(composed.invertible)
        
        inv = composed.get_inverse()
        self.assertIsNotNone(inv)
    
    def test_composition_inverse_application(self):
        """Test that the inverse of a composition works correctly."""
        from core.transformations import create_translation_transformation
        
        t1 = create_translation_transformation(1.0, 0.0)
        t2 = create_translation_transformation(0.0, 1.0)
        
        composed = t1 @ t2
        inv = composed.get_inverse()
        
        p = (5.0, 5.0)
        transformed = composed.apply(p)
        inverted = inv.apply(transformed)
        
        self.assertAlmostEqual(inverted[0], p[0], places=5)
        self.assertAlmostEqual(inverted[1], p[1], places=5)


class TestMacroTransformation(unittest.TestCase):
    """Tests for MacroTransformation."""
    
    def test_macro_creation(self):
        """Test creation of a macro-transformation."""
        from core.transformations import (
            create_translation_transformation,
            MacroTransformation,
        )
        
        t = create_translation_transformation(1.0, 0.0)
        macro = MacroTransformation(
            name="move_right",
            base_transformation=t,
            description="Déplace vers la droite"
        )
        
        self.assertEqual(macro.name, "move_right")
        self.assertEqual(macro.description, "Déplace vers la droite")
    
    def test_macro_inherits_properties(self):
        """Test that a macro inherits the properties of the base transformation."""
        from core.transformations import (
            create_translation_transformation,
            MacroTransformation,
        )
        
        t = create_translation_transformation(1.0, 0.0)
        macro = MacroTransformation(name="move", base_transformation=t)
        
        self.assertEqual(macro.preserves, t.preserves)
        self.assertEqual(macro.invertible, t.invertible)
        self.assertEqual(macro.cost, t.cost)
    
    def test_macro_apply(self):
        """Test application of a macro."""
        from core.transformations import (
            create_translation_transformation,
            MacroTransformation,
        )
        
        t = create_translation_transformation(1.0, 0.0)
        macro = MacroTransformation(name="move", base_transformation=t)
        
        self.assertEqual(macro.apply((0, 0)), (1.0, 0.0))
    
    def test_macro_inverse(self):
        """Test the inverse of a macro."""
        from core.transformations import (
            create_translation_transformation,
            MacroTransformation,
        )
        
        t = create_translation_transformation(1.0, 0.0)
        macro = MacroTransformation(name="move", base_transformation=t)
        
        inv = macro.get_inverse()
        self.assertIsNotNone(inv)
        self.assertIsInstance(inv, MacroTransformation)
        self.assertIn("inverse", inv.name)
    
    def test_macro_unwrap(self):
        """Test unwrapping of a macro."""
        from core.transformations import (
            create_translation_transformation,
            MacroTransformation,
        )
        
        t = create_translation_transformation(1.0, 0.0)
        macro = MacroTransformation(name="move", base_transformation=t)
        
        unwrapped = macro.unwrap()
        self.assertEqual(unwrapped, t)
    
    def test_nested_macro_unwrap(self):
        """Test unwrapping of a nested macro."""
        from core.transformations import (
            create_translation_transformation,
            MacroTransformation,
        )
        
        t = create_translation_transformation(1.0, 0.0)
        macro1 = MacroTransformation(name="move", base_transformation=t)
        macro2 = MacroTransformation(name="move2", base_transformation=macro1)
        
        unwrapped = macro2.unwrap()
        self.assertEqual(unwrapped, t)


class TestCutAndMergeTransformations(unittest.TestCase):
    """Tests for CutTransformation and MergeTransformation."""
    
    def test_cut_transformation(self):
        """Test CutTransformation."""
        from core.transformations import CutTransformation
        from core.topology import DISK_TOPOLOGY
        
        cut = CutTransformation()
        self.assertEqual(cut.name, "cut")
        self.assertFalse(cut.invertible)
        
        result = cut.apply(DISK_TOPOLOGY)
        self.assertEqual(result.connected_components, 2)
    
    def test_merge_transformation(self):
        """Test MergeTransformation."""
        from core.transformations import MergeTransformation
        from core.topology import DISK_TOPOLOGY
        
        merge = MergeTransformation()
        self.assertEqual(merge.name, "merge")
        self.assertFalse(merge.invertible)
        
        # Apply to a state with 2 components
        from core.topology import TopologySignature
        two_disks = TopologySignature(connected_components=2, holes=0, boundary_components=2)
        result = merge.apply(two_disks)
        self.assertEqual(result.connected_components, 1)
    
    def test_cut_merge_roundtrip(self):
        """Test that cut followed by merge returns to the original (in number of components)."""
        from core.transformations import CutTransformation, MergeTransformation
        from core.topology import DISK_TOPOLOGY
        
        cut = CutTransformation()
        merge = MergeTransformation()
        
        after_cut = cut.apply(DISK_TOPOLOGY)
        after_merge = merge.apply(after_cut)
        
        self.assertEqual(after_merge.connected_components, DISK_TOPOLOGY.connected_components)


class TestTransformNode(unittest.TestCase):
    """Tests for TransformNode."""
    
    def test_root_node(self):
        """Test a root node."""
        from core.transform_graph import TransformNode
        from core.transformations import IdentityTransformation
        
        root = TransformNode(
            name="root",
            invariants=IdentityTransformation().preserves
        )
        
        self.assertTrue(root.is_root())
        self.assertTrue(root.is_leaf())
        self.assertEqual(root.depth, 0)
    
    def test_add_child(self):
        """Test adding a child."""
        from core.transform_graph import TransformNode
        from core.transformations import create_translation_transformation
        
        root = TransformNode(name="root")
        t = create_translation_transformation(1.0, 0.0)
        
        child = root.add_child(t)
        
        self.assertEqual(child.name, "root -> translate(1.0, 0.0)")
        self.assertEqual(child.depth, 1)
        self.assertEqual(len(root.children), 1)
        self.assertEqual(child.parent, root)
    
    def test_path(self):
        """Test the path of a node."""
        from core.transform_graph import TransformNode
        from core.transformations import create_translation_transformation
        
        root = TransformNode(name="root")
        t1 = create_translation_transformation(1.0, 0.0)
        t2 = create_translation_transformation(0.0, 1.0)
        
        child = root.add_child(t1)
        grandchild = child.add_child(t2)
        
        path = grandchild.get_path()
        self.assertEqual(len(path), 2)
        self.assertEqual(path[0].name, "translate(1.0, 0.0)")
        self.assertEqual(path[1].name, "translate(0.0, 1.0)")
    
    def test_invariants_propagation(self):
        """Test the propagation of invariants."""
        from core.transform_graph import TransformNode
        from core.transformations import create_translation_transformation
        from core.invariants import Invariant
        
        root = TransformNode(name="root")
        t = create_translation_transformation(1.0, 0.0)
        
        child = root.add_child(t)
        
        # Translation preserves many invariants
        self.assertTrue(child.invariants.preserves(Invariant.DISTANCE))
    
    def test_can_invert(self):
        """Test the invertibility check."""
        from core.transform_graph import TransformNode
        from core.transformations import (
            create_translation_transformation,
            CutTransformation,
        )
        
        root = TransformNode(name="root")
        t = create_translation_transformation(1.0, 0.0)
        
        child = root.add_child(t)
        self.assertTrue(child.can_invert())
        
        # Add a non-invertible transformation
        cut = CutTransformation()
        grandchild = child.add_child(cut)
        self.assertFalse(grandchild.can_invert())
    
    def test_composite_transformation(self):
        """Test the creation of a composite transformation from a node."""
        from core.transform_graph import TransformNode
        from core.transformations import create_translation_transformation
        
        root = TransformNode(name="root")
        t1 = create_translation_transformation(1.0, 0.0)
        t2 = create_translation_transformation(0.0, 1.0)
        
        child = root.add_child(t1)
        grandchild = child.add_child(t2)
        
        composite = grandchild.get_composite_transformation()
        self.assertEqual(composite.get_depth(), 2)


class TestTransformGraph(unittest.TestCase):
    """Tests for TransformGraph."""
    
    def test_graph_creation(self):
        """Test the creation of a graph."""
        from core.transform_graph import TransformNode, TransformGraph
        
        root = TransformNode(name="root")
        graph = TransformGraph(root=root)
        
        self.assertEqual(graph.current, root)
        self.assertEqual(len(graph.nodes), 1)
    
    def test_apply_transformation(self):
        """Test application of a transformation."""
        from core.transform_graph import TransformGraph, TransformNode
        from core.transformations import create_translation_transformation
        
        root = TransformNode(name="root")
        graph = TransformGraph(root=root)
        
        t = create_translation_transformation(1.0, 0.0)
        new_node = graph.apply_transformation(t)
        
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(graph.current, new_node)
    
    def test_undo(self):
        """Test undo."""
        from core.transform_graph import TransformGraph, TransformNode
        from core.transformations import create_translation_transformation
        
        root = TransformNode(name="root")
        graph = TransformGraph(root=root)
        
        t = create_translation_transformation(1.0, 0.0)
        graph.apply_transformation(t)
        
        self.assertTrue(graph.can_undo())
        old_current = graph.current
        new_current = graph.undo()
        
        self.assertEqual(new_current, root)
        self.assertNotEqual(graph.current, old_current)
    
    def test_undo_at_root(self):
        """Test that undo at the root fails."""
        from core.transform_graph import TransformGraph, TransformNode
        
        root = TransformNode(name="root")
        graph = TransformGraph(root=root)
        
        self.assertFalse(graph.can_undo())
        with self.assertRaises(ValueError):
            graph.undo()
    
    def test_reset_to_root(self):
        """Test resetting to the root."""
        from core.transform_graph import TransformGraph, TransformNode
        from core.transformations import create_translation_transformation
        
        root = TransformNode(name="root")
        graph = TransformGraph(root=root)
        
        t = create_translation_transformation(1.0, 0.0)
        graph.apply_transformation(t)
        
        self.assertNotEqual(graph.current, root)
        graph.reset_to_root()
        self.assertEqual(graph.current, root)
    
    def test_history_summary(self):
        """Test the history summary."""
        from core.transform_graph import TransformGraph, TransformNode
        from core.transformations import create_translation_transformation
        
        root = TransformNode(name="root")
        graph = TransformGraph(root=root)
        
        t = create_translation_transformation(1.0, 0.0)
        graph.apply_transformation(t)
        
        summary = graph.get_history_summary()
        self.assertIn('depth', summary)
        self.assertIn('total_nodes', summary)
        self.assertIn('preserved_invariants', summary)
        self.assertEqual(summary['depth'], 1)
        self.assertEqual(summary['total_nodes'], 2)


class TestCreateLinearGraph(unittest.TestCase):
    """Tests for the create_linear_graph function."""
    
    def test_linear_graph_creation(self):
        """Test the creation of a linear graph."""
        from core.transform_graph import create_linear_graph
        from core.transformations import create_translation_transformation
        
        transformations = [
            create_translation_transformation(1.0, 0.0),
            create_translation_transformation(0.0, 1.0),
        ]
        
        graph = create_linear_graph("initial", transformations)
        
        self.assertEqual(graph.current.depth, 2)
        self.assertEqual(len(graph.current.transformations), 2)
    
    def test_linear_graph_invariants(self):
        """Test the invariants in a linear graph."""
        from core.transform_graph import create_linear_graph
        from core.transformations import create_translation_transformation
        from core.invariants import Invariant
        
        transformations = [
            create_translation_transformation(1.0, 0.0),
            create_translation_transformation(0.0, 1.0),
        ]
        
        graph = create_linear_graph("initial", transformations)
        invariants = graph.get_current_invariants()
        
        # Translations preserve distance
        self.assertTrue(invariants.preserves(Invariant.DISTANCE))


class TestIntegrationWithExistingSystem(unittest.TestCase):
    """Integration tests with the existing system."""
    
    def test_transform2d_still_works(self):
        """Test that Transform2D still works."""
        from core.transform import Transform2D
        from math import pi
        
        T = Transform2D.translation(5.0, 0.0).compose(Transform2D.rotation(pi/2))
        
        p = (1.0, 0.0)
        pt = T.transform_point(*p)
        
        T_inv = T.inverse()
        ptt = T_inv.transform_point(*pt)
        
        self.assertAlmostEqual(ptt[0], p[0], places=5)
        self.assertAlmostEqual(ptt[1], p[1], places=5)
    
    def test_topology_still_works(self):
        """Test that topology still works."""
        from core.topology import TopologySignature, DISK_TOPOLOGY
        
        disk = TopologySignature(connected_components=1, holes=0, boundary_components=1)
        self.assertEqual(disk, DISK_TOPOLOGY)
        self.assertTrue(disk.is_simple())
    
    def test_deformation_still_works(self):
        """Test that deformations still work."""
        from core.deformation import StretchDeformation
        
        stretch = StretchDeformation(stretch_x=2.0, stretch_y=3.0)
        self.assertEqual(stretch.name, "stretch(2.0, 3.0)")
        self.assertTrue(stretch.topology_preservation.preserves_connected_components)
    
    def test_plane_collider_integration(self):
        """Test that PlaneCollider integrates with the new system."""
        from core.transform import Transform2D
        from physics.colliders import PlaneCollider
        
        # Create a PlaneCollider with Transform2D
        collider = PlaneCollider(
            normal_x=1.0,
            normal_y=0.0,
            transform=Transform2D.translation(5.0, 0.0)
        )
        
        self.assertTrue(collider.is_behind(6.0, 0.0))
        self.assertFalse(collider.is_behind(4.0, 0.0))


if __name__ == '__main__':
    unittest.main()

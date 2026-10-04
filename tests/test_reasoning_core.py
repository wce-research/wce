import unittest
from math import pi

from core.action import LiftAction, PushAction, ImpulseAction
from core.executor import execute_action
from core.invariants import Invariant, InvariantSet
from core.topology import DISK_TOPOLOGY, TopologyState
from core.transformation import Change, Transformation, TransformationSource, apply
from core.transformations import (
    CompositeTransformation, DeclarativeTransformationSpec, MacroTransformation,
    TransformationDomain, TransformationSpec, create_reflection_transformation,
    create_rotation_transformation, create_scale_transformation,
    create_shear_transformation, create_translation_transformation,
)
from core.world import Entity, World
from physics.engine import step
from physics.interactions import GroundCollision
from physics.laws import GravityLaw


class ReasoningCoreIntegrationTests(unittest.TestCase):
    def world(self, y=0.0):
        world = World()
        world.add_entity(Entity("balle", "objet", {"x": 0.0, "y": y, "vx": 0.0, "vy": 0.0, "mass": 1.0}, DISK_TOPOLOGY))
        return world

    def test_action_spec_executable_world_and_provenance(self):
        world = self.world()
        action = LiftAction("balle", 2.0)
        spec = action.to_spec(world)
        self.assertIsInstance(spec, TransformationSpec)
        executable = spec.compile(world)
        apply(world, executable.with_source(TransformationSource.ACTION))
        self.assertEqual(world.entities["balle"].properties["y"], 2.0)
        self.assertEqual(world.transform_graph.current.record.source_type, "action")

    def test_all_actions_carry_specs(self):
        world = self.world()
        for action in (LiftAction("balle", 1), PushAction("balle", 2), ImpulseAction("balle", 1)):
            self.assertIsNotNone(action.compile(world).spec)

    def test_law_spec_executable_world_and_provenance(self):
        world = self.world(2.0)
        law = GravityLaw()
        transformation = law.propose(world, .1)
        self.assertIsNotNone(transformation.spec)
        apply(world, transformation)
        self.assertEqual(world.transform_graph.current.record.source_type, "law")
        self.assertAlmostEqual(world.entities["balle"].properties["y"], 1.9019)

    def test_interaction_provenance_in_graph(self):
        world = self.world(.01)
        step(world, [GravityLaw()], .1, [GroundCollision()])
        self.assertEqual(world.transform_graph.current.record.source_type, "interaction")
        self.assertEqual(world.transform_graph.current.record.source_name, "GroundCollision")

    def test_every_successful_apply_reaches_history_and_graph(self):
        world = self.world()
        execute_action(world, LiftAction("balle", 1))
        execute_action(world, LiftAction("balle", 2))
        self.assertEqual(len(world.history), 2)
        self.assertEqual(world.transform_graph.current.depth, 2)

    def test_macro_is_spec_and_logical_operation(self):
        world = self.world()
        macro = MacroTransformation(name="place", steps=(create_translation_transformation(1, 0), create_scale_transformation(2, 2)))
        self.assertIsInstance(macro, TransformationSpec)
        apply(world, macro.compile(world))
        self.assertEqual(world.entities["balle"].properties["x"], 2.0)
        self.assertEqual(world.transform_graph.current.record.source_type, "macro")
        self.assertEqual(len(macro.base_transformation.transformations), 2)

    def test_composite_applies_to_world_with_common_invariants(self):
        world = self.world()
        composite = CompositeTransformation(transformations=(
            create_translation_transformation(1, 0), create_rotation_transformation(pi/2),
            create_scale_transformation(2, 3)))
        apply(world, composite.compile(world))
        self.assertAlmostEqual(world.entities["balle"].properties["y"], 3.0)
        self.assertTrue(composite.preserves.preserves(Invariant.CONNECTIVITY))
        self.assertEqual(composite.get_depth(), 3)

    def test_reflection_primitive_inverse_and_world(self):
        spec = create_reflection_transformation("y")
        self.assertEqual(spec.get_inverse().apply(spec.apply((2, 3))), (2, 3))
        world = self.world(); world.entities["balle"].properties["x"] = 2
        apply(world, spec.compile(world))
        self.assertEqual(world.entities["balle"].properties["x"], -2)

    def test_shear_primitive_inverse_and_composition(self):
        spec = create_shear_transformation(2, 0)
        point = spec.apply((1, 3))
        restored = spec.get_inverse().apply(point)
        self.assertAlmostEqual(restored[0], 1); self.assertAlmostEqual(restored[1], 3)
        self.assertIsInstance(spec.compose(create_translation_transformation(1, 0)), CompositeTransformation)

    def test_graph_branch_parent_children_and_path(self):
        world = self.world(); root = world.transform_graph.root
        execute_action(world, LiftAction("balle", 1)); first = world.transform_graph.current
        world.transform_graph.branch_from(root)
        world.entities["balle"].properties["y"] = 0
        execute_action(world, LiftAction("balle", 2)); second = world.transform_graph.current
        self.assertEqual(len(root.children), 2)
        self.assertIs(first.parent, root); self.assertIs(second.parent, root)
        self.assertEqual(second.get_path_names(), ["lift"])

    def test_explain_last_change(self):
        world = self.world(); execute_action(world, LiftAction("balle", 2))
        explanation = world.explain_last_change()
        self.assertIn("balle.y: 0.0 -> 2", explanation)
        self.assertIn("Cause: LiftAction", explanation)
        self.assertIn("Domain: physics", explanation)

    def test_topology_preserved_rotation_and_stretch(self):
        for spec in (create_rotation_transformation(pi/2), create_scale_transformation(2, 3)):
            world = self.world(); apply(world, spec.compile(world))
            entity = world.entities["balle"]
            self.assertEqual(entity.topology, DISK_TOPOLOGY)
            self.assertEqual(entity.topology_state, TopologyState.PRESERVED)

    def test_topology_unknown_when_not_guaranteed(self):
        world = self.world()
        spec = DeclarativeTransformationSpec(
            name="opaque", domain=TransformationDomain.WORLD, preserves=InvariantSet(set()),
            compiler=lambda w: Transformation("opaque", [Change("balle", "y", 0.0, 1.0)]))
        apply(world, spec.compile(world))
        self.assertEqual(world.entities["balle"].topology_state, TopologyState.UNKNOWN)


if __name__ == "__main__": unittest.main()

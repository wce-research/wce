"""Integration demonstration of the WCE reasoning core."""
from math import pi
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.action import LiftAction
from core.executor import execute_action
from core.topology import DISK_TOPOLOGY
from core.transformation import apply
from core.transformations import (
    MacroTransformation, create_rotation_transformation,
    create_scale_transformation, create_translation_transformation,
)
from core.world import Entity, World
from physics.engine import step
from physics.laws import GravityLaw


def invariant_names(spec):
    return ", ".join(sorted(i.name for i in spec.preserves.invariants))


def main():
    world = World()
    world.add_entity(Entity("balle", "geometry", {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "mass": 1.0}, DISK_TOPOLOGY))
    print("=== WCE REASONING CORE ===")
    print("\nInitial state:\nballe at (0.0, 0.0)")

    action = LiftAction("balle", 2.0)
    spec = action.to_spec(world)
    execute_action(world, action)
    print("\nAction: lift")
    print(f"TransformationSpec: domain={spec.domain.value} source=action")
    print("WorldTransformation: balle.y 0.0 -> 2.0")

    step(world, [GravityLaw()], .1)
    law_record = world.transform_graph.current.record
    print("\nLaw: gravity")
    for change in law_record.executable.changes:
        print(f"WorldTransformation: {change.entity}.{change.property} {change.before} -> {change.after}")

    print("\nTransformGraph:")
    print("C0 --lift--> C1 --gravity(dt=0.1)--> C2")
    print("\nExplain C2:\n" + world.explain_last_change())

    geometry = World()
    geometry.add_entity(Entity("forme", "geometry", {"x": 1.0, "y": 0.0}, DISK_TOPOLOGY))
    macro = MacroTransformation(name="place_shape", steps=(
        create_translation_transformation(1, 0),
        create_rotation_transformation(pi / 2),
        create_scale_transformation(2, 3),
    ))
    apply(geometry, macro.compile(geometry))
    entity = geometry.entities["forme"]
    print("\nMacroTransformation: place_shape")
    print("Steps: translate -> rotate -> stretch")
    print(f"Final state: ({entity.properties['x']:.4f}, {entity.properties['y']:.4f})")
    print(f"Common invariants: {invariant_names(macro)}")
    print(f"Topology: {entity.topology_state.value}")


if __name__ == "__main__": main()

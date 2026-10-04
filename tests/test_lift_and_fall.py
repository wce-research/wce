from core.world import World, Entity
from core.action import LiftAction
from core.executor import execute_action

from physics.constraints import non_negative_height
from physics.engine import simulate
from physics.interactions import GroundCollision
from physics.laws import GravityLaw


world = World()

world.add_entity(
    Entity(
        name="sol",
        kind="support",
        properties={
            "y": 0.0,
            "fixed": True,
        },
    )
)

world.add_entity(
    Entity(
        name="balle",
        kind="objet",
        properties={
            "y": 0.0,
            "vy": 0.0,
            "masse": 1.0,
        },
    )
)

world.add_relation(
    "balle",
    "supported_by",
    "sol",
)

world.add_constraint(
    non_negative_height()
)


print("=== ETAT INITIAL ===")
world.show()


action = LiftAction(
    entity_name="balle",
    target_y=2.0,
)

execute_action(
    world,
    action,
)


print("\n=== APRES ACTION ===")
world.show()


simulate(
    world,
    laws=[
        GravityLaw(),
    ],
    interactions=[
        GroundCollision(
            ground_y=0.0,
            restitution=0.0,
            support_name="sol",
        ),
    ],
    duration=2.0,
    dt=0.1,
)


print("\n=== APRES SIMULATION ===")
world.show()

print()
world.show_history()

from core.world import World, Entity

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
            "y": 2.0,
            "vy": 0.0,
            "masse": 1.0,
        },
    )
)

world.add_constraint(
    non_negative_height()
)

laws = [
    GravityLaw(),
]

interactions = [
    GroundCollision(
        ground_y=0.0,
        restitution=0.0,
        support_name="sol",
    )
]

print("AVANT")
world.show()

simulate(
    world,
    laws=laws,
    interactions=interactions,
    duration=2.0,
    dt=0.1,
)

print("\nAPRES")
world.show()

print()
world.show_history()

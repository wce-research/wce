from core.world import World, Entity
from physics.engine import simulate
from physics.laws import GravityLaw


world = World()

world.add_entity(
    Entity(
        name="balle",
        kind="objet",
        properties={
            "y": 10.0,
            "vy": 0.0,
            "masse": 1.0,
        },
    )
)

laws = [
    GravityLaw(),
]

print("AVANT")
world.show()

simulate(
    world,
    laws=laws,
    duration=1.0,
    dt=0.1,
)

print("\nAPRES 1 SECONDE")
world.show()

print()
world.show_history()

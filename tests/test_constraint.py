from core.world import World, Entity
from core.transformation import Change, Transformation, apply
from physics.constraints import non_negative_height


world = World()

world.add_entity(
    Entity(
        name="balle",
        kind="objet",
        properties={
            "y": 2,
        },
    )
)

world.add_constraint(
    non_negative_height()
)

print("AVANT")
world.show()

try:
    apply(
        world,
        Transformation(
            name="descente_impossible",
            changes=[
                Change(
                    "balle",
                    "y",
                    2,
                    -5,
                )
            ],
        ),
    )
except ValueError as error:
    print("\nREFUS")
    print(error)

print("\nAPRES")
world.show()

print()

world.show_history()

from core.world import World, Entity
from core.transformation import Change, Transformation, apply


world = World()

world.add_entity(
    Entity(
        name="balle",
        kind="objet",
        properties={
            "x": 0,
            "y": 10,
        },
    )
)


apply(
    world,
    Transformation(
        name="deplacement_horizontal",
        changes=[
            Change("balle", "x", 0, 5),
        ],
    ),
)


apply(
    world,
    Transformation(
        name="descente",
        changes=[
            Change("balle", "y", 10, 4),
        ],
    ),
)


apply(
    world,
    Transformation(
        name="chute_au_sol",
        changes=[
            Change("balle", "y", 4, 0),
        ],
    ),
)


world.show()

print()

world.show_history()

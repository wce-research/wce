from core.world import World, Entity


world = World()

world.add_entity(
    Entity(
        name="balle",
        kind="objet",
        properties={
            "forme": "sphere",
            "couleur": "rouge",
            "masse": 1.0,
        },
    )
)

world.add_entity(
    Entity(
        name="table",
        kind="objet",
        properties={
            "fixe": True,
        },
    )
)

world.add_relation(
    "balle",
    "sur",
    "table",
)

world.show()

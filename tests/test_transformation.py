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

move = Transformation(
    name="deplacement",
    changes=[
        Change(
            entity="balle",
            property="x",
            before=0,
            after=5,
        )
    ],
)

print("AVANT")
world.show()

apply(world, move)

print("\nTRANSFORMATION")
print(move)

print("\nAPRES")
world.show()

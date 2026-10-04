from core.world import World, Entity

from language.parser import (
    ParseError,
    extract_intent,
    parse_action,
)
from language.runtime import execute_text


def build_world():
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

    return world


print("=== INTENTS ===")

examples = [
    "Soulève la balle à 2 mètres",
    "Monte la balle à 300 cm",
    "Lève une balle à 1500 mm",
]

for text in examples:
    intent = extract_intent(text)
    print(repr(text), "->", intent)


print("\n=== ACTIONS ===")

for text in examples:
    action = parse_action(text)
    print(repr(text), "->", action)


print("\n=== EXECUTION ===")

world = build_world()

execute_text(
    world,
    "Soulève la balle à 250 cm",
)

world.show()

print()
world.show_history()


print("\n=== COMMANDE INCONNUE ===")

try:
    parse_action(
        "Fais danser la balle"
    )

except ParseError as error:
    print(error)

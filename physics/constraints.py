from core.constraint import Constraint


def non_negative_height():
    def check(world):
        for entity in world.entities.values():
            y = entity.properties.get("y")

            if y is not None and y < 0:
                return (
                    False,
                    f"{entity.name}.y = {y}, hauteur négative interdite"
                )

        return True, "ok"

    return Constraint(
        name="non_negative_height",
        check=check,
    )

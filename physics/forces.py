from dataclasses import dataclass


@dataclass(frozen=True)
class Force:
    entity: str
    fx: float = 0.0
    fy: float = 0.0
    source: str = ""


class ForceProvider:
    def forces(self, world) -> list[Force]:
        raise NotImplementedError


class GravityForce(ForceProvider):
    def __init__(self, g: float = -9.81):
        self.g = g

    def forces(self, world) -> list[Force]:
        result = []

        for entity in world.entities.values():
            if entity.properties.get("fixed"):
                continue

            if "y" not in entity.properties:
                continue

            mass = get_mass(entity)

            result.append(
                Force(
                    entity=entity.name,
                    fx=0.0,
                    fy=mass * self.g,
                    source="gravity",
                )
            )

        return result


class SupportForce(ForceProvider):
    """
    2D model:
    a static support exactly balances
    the vertical weight of the supported object.
    No horizontal friction in this version.
    """

    def __init__(self, g: float = -9.81):
        self.g = g

    def forces(self, world) -> list[Force]:
        result = []

        for relation in world.relations:
            if relation.predicate != "supported_by":
                continue

            entity = world.entities.get(
                relation.subject
            )

            if entity is None:
                continue

            if entity.properties.get("fixed"):
                continue

            mass = get_mass(entity)

            result.append(
                Force(
                    entity=entity.name,
                    fx=0.0,
                    fy=-(mass * self.g),
                    source=f"support:{relation.object}",
                )
            )

        return result


class ConstantForce(ForceProvider):
    """
    Generic force useful for experiments
    and for future push, pull, etc. engines.
    Now supports the x and y components.
    """

    def __init__(
        self,
        entity_name: str,
        fy: float = 0.0,
        fx: float = 0.0,
        source: str = "constant",
    ):
        self.entity_name = entity_name
        self.fx = fx
        self.fy = fy
        self.source = source

    def forces(self, world) -> list[Force]:
        if self.entity_name not in world.entities:
            return []

        return [
            Force(
                entity=self.entity_name,
                fx=self.fx,
                fy=self.fy,
                source=self.source,
            )
        ]


class ExternalForce(ForceProvider):

    def forces(self, world) -> list[Force]:
        result = []

        for entity in world.entities.values():
            fx = float(
                entity.properties.get(
                    "external_force_x",
                    0.0,
                )
            )
            fy = float(
                entity.properties.get(
                    "external_force_y",
                    0.0,
                )
            )

            if fx == 0.0 and fy == 0.0:
                continue

            result.append(
                Force(
                    entity=entity.name,
                    fx=fx,
                    fy=fy,
                    source="external",
                )
            )

        return result


def get_mass(entity) -> float:
    mass = entity.properties.get(
        "masse",
        entity.properties.get(
            "mass",
            1.0,
        ),
    )

    mass = float(mass)

    if mass <= 0:
        raise ValueError(
            f"Invalid mass for {entity.name!r}: {mass}"
        )

    return mass


def collect_forces(
    world,
    providers,
) -> list[Force]:
    result = []

    for provider in providers:
        result.extend(
            provider.forces(world)
        )

    return result


def net_force(
    world,
    providers,
    entity_name: str,
) -> tuple[float, float]:
    """
    Returns the net force (fx, fy) on an entity.
    For backward compatibility with existing code that expects
    a scalar, see net_force_y().
    """
    forces = collect_forces(world, providers)
    fx = sum(
        force.fx
        for force in forces
        if force.entity == entity_name
    )
    fy = sum(
        force.fy
        for force in forces
        if force.entity == entity_name
    )
    return (fx, fy)


def net_force_y(
    world,
    providers,
    entity_name: str,
) -> float:
    """
    Returns the vertical component of the net force.
    Maintains backward compatibility with the old interface.
    """
    _, fy = net_force(world, providers, entity_name)
    return fy

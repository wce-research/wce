from dataclasses import dataclass

from core.transformation import (
    Change,
    RelationChange,
    Transformation,
    TransformationSource,
)
from core.invariants import Invariant, InvariantSet
from core.transformations import DeclarativeTransformationSpec, TransformationDomain

from physics.forces import (
    ExternalForce,
    GravityForce,
    collect_forces,
    get_mass,
)


@dataclass
class Law:
    name: str

    def propose(
        self,
        world,
        dt: float,
    ) -> Transformation | None:
        executable = self.to_spec(dt).compile(world)
        return executable.with_source(TransformationSource.LAW) if executable is not None else None

    def to_spec(self, dt: float) -> DeclarativeTransformationSpec:
        return DeclarativeTransformationSpec(
            name=f"{self.name}(dt={dt})", domain=TransformationDomain.PHYSICS,
            parameters={"dt": dt, "source_name": type(self).__name__},
            preserves=InvariantSet({Invariant.MASS, Invariant.ENTITY_IDENTITY}),
            changes={"position", "velocity"},
            compiler=lambda world: self._propose_executable(world, dt),
        )

    def _propose_executable(self, world, dt):
        raise NotImplementedError


class ForceDynamicsLaw(Law):
    """
    2D Newtonian dynamics.

        Fx_net = sum(Fx)
        Fy_net = sum(Fy)

        ax = Fx_net / m
        ay = Fy_net / m

        vx(t+dt) = vx(t) + ax*dt
        vy(t+dt) = vy(t) + ay*dt

        x(t+dt) = x(t) + vx(t+dt)*dt
        y(t+dt) = y(t) + vy(t+dt)*dt

    The support is treated as a vertical unilateral constraint:

        - if the vertical forces push downwards,
          the support provides the necessary reaction;

        - if the resulting vertical forces push upwards,
          the contact is broken.

    Horizontally, no support constraint (no friction).
    """

    def __init__(
        self,
        providers,
        name: str = "dynamics",
    ):
        super().__init__(
            name=name
        )

        self.providers = list(
            providers
        )

    def _propose_executable(
        self,
        world,
        dt: float,
    ) -> Transformation | None:

        forces = collect_forces(
            world,
            self.providers,
        )

        # Group the forces by entity
        force_by_entity = {}

        for force in forces:
            if force.entity not in force_by_entity:
                force_by_entity[force.entity] = {"fx": 0.0, "fy": 0.0}

            force_by_entity[force.entity]["fx"] += force.fx
            force_by_entity[force.entity]["fy"] += force.fy

        changes = []
        relation_changes = []

        for entity in world.entities.values():
            if entity.properties.get("fixed"):
                continue

            # Check that we have at least one position property
            if "y" not in entity.properties and "x" not in entity.properties:
                continue

            mass = get_mass(entity)

            fx = force_by_entity.get(entity.name, {"fx": 0.0, "fy": 0.0})["fx"]
            fy = force_by_entity.get(entity.name, {"fx": 0.0, "fy": 0.0})["fy"]

            # --- Support handling (vertical only) ---
            support_relations = [
                relation
                for relation in world.relations
                if (
                    relation.subject == entity.name
                    and relation.predicate == "supported_by"
                )
            ]

            if support_relations:
                # The support can push upwards,
                # but can never pull the object downwards.
                if fy <= 0.0:
                    # Normal reaction:
                    # it exactly compensates the forces
                    # directed downwards.
                    fy = 0.0

                else:
                    # The vertical resultant is directed upwards:
                    # the object leaves its support.
                    for relation in support_relations:
                        relation_changes.append(
                            RelationChange(
                                action="remove",
                                subject=relation.subject,
                                predicate=relation.predicate,
                                object=relation.object,
                            )
                        )

            # --- Computation of the accelerations ---
            ax = fx / mass
            ay = fy / mass

            # --- Integration for x (if the property exists) ---
            if "x" in entity.properties:
                old_vx = float(
                    entity.properties.get(
                        "vx",
                        0.0,
                    )
                )

                old_x = float(
                    entity.properties["x"]
                )

                new_vx = old_vx + ax * dt

                # Semi-implicit Euler.
                new_x = old_x + new_vx * dt

                if new_vx != old_vx:
                    changes.append(
                        Change(
                            entity=entity.name,
                            property="vx",
                            before=old_vx,
                            after=new_vx,
                        )
                    )

                if new_x != old_x:
                    changes.append(
                        Change(
                            entity=entity.name,
                            property="x",
                            before=old_x,
                            after=new_x,
                        )
                    )

            # --- Integration for y (always present in the existing tests) ---
            if "y" in entity.properties:
                old_vy = float(
                    entity.properties.get(
                        "vy",
                        0.0,
                    )
                )

                old_y = float(
                    entity.properties["y"]
                )

                new_vy = old_vy + ay * dt

                # Semi-implicit Euler.
                new_y = old_y + new_vy * dt

                if new_vy != old_vy:
                    changes.append(
                        Change(
                            entity=entity.name,
                            property="vy",
                            before=old_vy,
                            after=new_vy,
                        )
                    )

                if new_y != old_y:
                    changes.append(
                        Change(
                            entity=entity.name,
                            property="y",
                            before=old_y,
                            after=new_y,
                        )
                    )

        if not changes and not relation_changes:
            return None

        return Transformation(
            name=f"{self.name}(dt={dt})",
            changes=changes,
            relation_changes=relation_changes,
            timestamp=world.time + dt,
        )


class GravityLaw(ForceDynamicsLaw):
    """
    Standard vertical dynamics of WCE.

    The explicit forces are:
      - weight
      - external force

    The support reaction is resolved
    dynamically by ForceDynamicsLaw.
    """

    def __init__(
        self,
        g: float = -9.81,
    ):
        super().__init__(
            providers=[
                GravityForce(g=g),
                ExternalForce(),
            ],
            name="gravity",
        )

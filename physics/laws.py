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
    Dynamique newtonienne 2D.

        Fx_net = somme(Fx)
        Fy_net = somme(Fy)

        ax = Fx_net / m
        ay = Fy_net / m

        vx(t+dt) = vx(t) + ax*dt
        vy(t+dt) = vy(t) + ay*dt

        x(t+dt) = x(t) + vx(t+dt)*dt
        y(t+dt) = y(t) + vy(t+dt)*dt

    Le support est traité comme une contrainte unilatérale verticale :

        - si les forces verticales poussent vers le bas,
          le support fournit la réaction nécessaire ;

        - si les forces verticales résultantes poussent vers le haut,
          le contact est rompu.

    En horizontal, pas de contrainte de support (pas de frottement).
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

        # Regrouper les forces par entité
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

            # Vérifier qu'on a au moins une propriété de position
            if "y" not in entity.properties and "x" not in entity.properties:
                continue

            mass = get_mass(entity)

            fx = force_by_entity.get(entity.name, {"fx": 0.0, "fy": 0.0})["fx"]
            fy = force_by_entity.get(entity.name, {"fx": 0.0, "fy": 0.0})["fy"]

            # --- Traitement du support (vertical uniquement) ---
            support_relations = [
                relation
                for relation in world.relations
                if (
                    relation.subject == entity.name
                    and relation.predicate == "supported_by"
                )
            ]

            if support_relations:
                # Le support peut pousser vers le haut,
                # mais ne peut jamais tirer l'objet vers le bas.
                if fy <= 0.0:
                    # Réaction normale :
                    # elle compense exactement les forces
                    # dirigées vers le bas.
                    fy = 0.0

                else:
                    # La résultante verticale est dirigée vers le haut :
                    # l'objet quitte son support.
                    for relation in support_relations:
                        relation_changes.append(
                            RelationChange(
                                action="remove",
                                subject=relation.subject,
                                predicate=relation.predicate,
                                object=relation.object,
                            )
                        )

            # --- Calcul des accélérations ---
            ax = fx / mass
            ay = fy / mass

            # --- Intégration pour x (si propriété existe) ---
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

                # Euler semi-implicite.
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

            # --- Intégration pour y (toujours présente dans les tests existants) ---
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

                # Euler semi-implicite.
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
    Dynamique verticale standard de WCE.

    Les forces explicites sont :
      - poids
      - force externe

    La réaction du support est résolue
    dynamiquement par ForceDynamicsLaw.
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

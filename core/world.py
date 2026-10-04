from dataclasses import dataclass, field
from typing import Any

from core.topology import TopologySignature, TopologyState


@dataclass
class Entity:
    name: str
    kind: str
    properties: dict[str, Any] = field(default_factory=dict)
    topology: TopologySignature | None = None
    topology_state: TopologyState = TopologyState.UNKNOWN

    def __post_init__(self):
        if self.topology is not None and self.topology_state is TopologyState.UNKNOWN:
            self.topology_state = TopologyState.KNOWN


@dataclass
class Relation:
    subject: str
    predicate: str
    object: str


class World:
    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self.history: list[Any] = []
        self.constraints: list[Any] = []
        self.time: float = 0.0
        # Import local pour garder World indépendant des détails de l'algèbre.
        from core.transform_graph import create_world_transform_graph
        self.transform_graph = create_world_transform_graph("C0")

    def add_entity(self, entity: Entity):
        self.entities[entity.name] = entity

    def add_relation(self, subject: str, predicate: str, object_: str):
        self.relations.append(
            Relation(subject, predicate, object_)
        )

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def record(self, transformation):
        self.history.append(transformation)

    def get_state_snapshot(self):
        """Référence légère et immuable des propriétés utiles au raisonnement."""
        return {
            name: dict(entity.properties)
            for name, entity in self.entities.items()
        }

    def explain_last_change(self) -> str:
        return self.transform_graph.explain_last()

    def validate(self):
        errors = []

        for constraint in self.constraints:
            valid, message = constraint.validate(self)

            if not valid:
                errors.append(
                    f"{constraint.name}: {message}"
                )

        return errors

    def show(self):
        print("=== WORLD ===")

        for entity in self.entities.values():
            print(
                f"ENTITY {entity.name!r} "
                f"kind={entity.kind!r} "
                f"properties={entity.properties}"
            )

        for relation in self.relations:
            print(
                f"RELATION "
                f"{relation.subject} "
                f"--{relation.predicate}--> "
                f"{relation.object}"
            )

    def show_history(self):
        print("=== HISTORY ===")

        if not self.history:
            print("(vide)")
            return

        for index, transformation in enumerate(
            self.history,
            start=1,
        ):
            print(f"{index}. {transformation.name}")

            for change in transformation.changes:
                print(
                    f"   {change.entity}.{change.property}: "
                    f"{change.before!r} -> {change.after!r}"
                )

            for change in transformation.relation_changes:
                symbol = "+" if change.action == "add" else "-"

                print(
                    f"   {symbol} RELATION "
                    f"{change.subject} "
                    f"--{change.predicate}--> "
                    f"{change.object}"
                )

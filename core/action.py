from dataclasses import dataclass

from core.transformation import (
    Change,
    RelationChange,
    Transformation,
    TransformationSource,
)
from core.invariants import Invariant, InvariantSet
from core.transformations import DeclarativeTransformationSpec, TransformationDomain


@dataclass
class Action:
    name: str

    def compile(self, world) -> Transformation:
        return self.to_spec(world).compile(world).with_source(TransformationSource.ACTION)

    def to_spec(self, world) -> DeclarativeTransformationSpec:
        return DeclarativeTransformationSpec(
            name=self.name, domain=TransformationDomain.PHYSICS,
            parameters={"source_name": type(self).__name__},
            preserves=InvariantSet({Invariant.ENTITY_IDENTITY, Invariant.MASS}),
            changes=set(), compiler=self._compile_executable,
        )

    def _compile_executable(self, world) -> Transformation:
        raise NotImplementedError


@dataclass
class LiftAction(Action):
    entity_name: str
    target_y: float
    support_name: str = "sol"

    def __init__(
        self,
        entity_name: str,
        target_y: float,
        support_name: str = "sol",
    ):
        super().__init__(name="lift")
        self.entity_name = entity_name
        self.target_y = target_y
        self.support_name = support_name

    def _compile_executable(self, world) -> Transformation:
        entity = world.entities[self.entity_name]

        changes = [
            Change(
                entity=self.entity_name,
                property="y",
                before=entity.properties["y"],
                after=self.target_y,
            )
        ]

        relation_changes = []

        for relation in world.relations:
            if (
                relation.subject == self.entity_name
                and relation.predicate == "supported_by"
            ):
                relation_changes.append(
                    RelationChange(
                        action="remove",
                        subject=relation.subject,
                        predicate=relation.predicate,
                        object=relation.object,
                    )
                )

        return Transformation(
            name=self.name,
            changes=changes,
            relation_changes=relation_changes,
        )


@dataclass
class PushAction(Action):
    entity_name: str
    force_y: float
    force_x: float = 0.0

    def __init__(
        self,
        entity_name: str,
        force_y: float,
        force_x: float = 0.0,
    ):
        super().__init__(name="push")
        self.entity_name = entity_name
        self.force_y = force_y
        self.force_x = force_x

    def _compile_executable(self, world) -> Transformation:
        entity = world.entities[self.entity_name]

        changes = []

        # Handle the horizontal component
        before_x = entity.properties.get("external_force_x")
        if before_x is None:
            before_x = 0.0

        if self.force_x != before_x:
            changes.append(
                Change(
                    entity=self.entity_name,
                    property="external_force_x",
                    before=entity.properties.get("external_force_x"),
                    after=self.force_x,
                )
            )

        # Handle the vertical component
        before_y = entity.properties.get("external_force_y")
        if before_y is None:
            before_y = 0.0

        if self.force_y != before_y:
            changes.append(
                Change(
                    entity=self.entity_name,
                    property="external_force_y",
                    before=entity.properties.get("external_force_y"),
                    after=self.force_y,
                )
            )

        return Transformation(
            name=self.name,
            changes=changes,
        )


@dataclass
class ImpulseAction(Action):
    entity_name: str
    impulse_y: float
    impulse_x: float = 0.0

    def __init__(
        self,
        entity_name: str,
        impulse_y: float,
        impulse_x: float = 0.0,
    ):
        super().__init__(name="impulse")
        self.entity_name = entity_name
        self.impulse_y = impulse_y
        self.impulse_x = impulse_x

    def _compile_executable(self, world) -> Transformation:
        entity = world.entities[self.entity_name]

        mass = float(
            entity.properties.get(
                "masse",
                entity.properties.get(
                    "mass",
                    1.0,
                ),
            )
        )

        if mass <= 0:
            raise ValueError(
                f"Invalid mass for {self.entity_name!r}: {mass}"
            )

        changes = []
        relation_changes = []

        # Vertical impulse
        old_vy = float(
            entity.properties.get(
                "vy",
                0.0,
            )
        )

        new_vy = old_vy + self.impulse_y / mass

        if new_vy != old_vy:
            changes.append(
                Change(
                    entity=self.entity_name,
                    property="vy",
                    before=old_vy,
                    after=new_vy,
                )
            )

        # Horizontal impulse
        old_vx = float(
            entity.properties.get(
                "vx",
                0.0,
            )
        )

        new_vx = old_vx + self.impulse_x / mass

        if new_vx != old_vx:
            changes.append(
                Change(
                    entity=self.entity_name,
                    property="vx",
                    before=old_vx,
                    after=new_vx,
                )
            )

        # If the vertical impulse is positive, leave the support
        if self.impulse_y > 0:
            for relation in world.relations:
                if (
                    relation.subject == self.entity_name
                    and relation.predicate == "supported_by"
                ):
                    relation_changes.append(
                        RelationChange(
                            action="remove",
                            subject=relation.subject,
                            predicate=relation.predicate,
                            object=relation.object,
                        )
                    )

        return Transformation(
            name=self.name,
            changes=changes,
            relation_changes=relation_changes,
        )

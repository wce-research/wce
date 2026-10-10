"""
Executable transformations for WCE.

This module provides the concrete representation of the transformations that modify
the state of the World, with a link to their conceptual specification (TransformationSpec).
"""

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING
from enum import Enum
import time

if TYPE_CHECKING:
    from core.transformations import TransformationSpec, InvariantSet


class TransformationSource(Enum):
    """Source/provenance of a transformation."""
    ACTION = "action"         # Produced by an Action (Lift, Push, etc.)
    LAW = "law"               # Produced by a Law (Gravity, ForceDynamics, etc.)
    INTERACTION = "interaction"  # Produced by an Interaction (collision, support, etc.)
    USER = "user"             # Direct user command
    SYSTEM = "system"         # Internal system transformation
    MACRO = "macro"           # MacroTransformation
    COMPOSITE = "composite"   # CompositeTransformation
    RULE = "rule"             # Symbolic / logical rule


@dataclass(frozen=True)
class Change:
    """Change of an entity property."""
    entity: str
    property: str
    before: Any
    after: Any


@dataclass(frozen=True)
class RelationChange:
    """Change of a relation between entities."""
    action: str  # "add" or "remove"
    subject: str
    predicate: str
    object: str


@dataclass
class Transformation:
    """
    Executable transformation on a World.
    
    This is the concrete representation that actually modifies the state of the world.
    It can be linked to a TransformationSpec for the semantics.
    """
    name: str
    changes: list[Change] = field(default_factory=list)
    relation_changes: list[RelationChange] = field(default_factory=list)
    
    # Link to the conceptual specification
    spec: Optional['TransformationSpec'] = None
    source: TransformationSource = TransformationSource.SYSTEM
    
    # Metadata for the graph
    timestamp: Optional[float] = None
    domain: Optional[str] = None
    invariants: Optional['InvariantSet'] = None
    
    def with_spec(self, spec: 'TransformationSpec', source: TransformationSource = None) -> 'Transformation':
        """Associates this transformation with a TransformationSpec."""
        return Transformation(
            name=self.name or spec.name,
            changes=self.changes,
            relation_changes=self.relation_changes,
            spec=spec,
            source=source or self.source,
            timestamp=self.timestamp,
            domain=self.domain,
            invariants=self.invariants,
        )
    
    def with_source(self, source: TransformationSource) -> 'Transformation':
        """Sets the source of this transformation."""
        return Transformation(
            name=self.name,
            changes=self.changes,
            relation_changes=self.relation_changes,
            spec=self.spec,
            source=source,
            timestamp=self.timestamp,
            domain=self.domain,
            invariants=self.invariants,
        )
    
    @property
    def is_empty(self) -> bool:
        """Checks whether the transformation does nothing."""
        return not self.changes and not self.relation_changes
    
    def __repr__(self) -> str:
        spec_name = f" spec={self.spec.name}" if self.spec else ""
        return f"Transformation(name='{self.name}', source={self.source.value}{spec_name})"


def apply(world, transformation: Transformation):
    """
    Applies a transformation to a World with validation and rollback.
    
    This function also records the transformation in the TransformGraph of the World
    if available.
    """
    # Set the timestamp if not specified
    before_snapshot = world.get_state_snapshot() if hasattr(world, 'get_state_snapshot') else None
    if transformation.timestamp is None:
        transformation = Transformation(
            name=transformation.name,
            changes=transformation.changes,
            relation_changes=transformation.relation_changes,
            spec=transformation.spec,
            source=transformation.source,
            timestamp=world.time if hasattr(world, "time") else time.time(),
            domain=transformation.domain,
            invariants=transformation.invariants,
        )
    
    # Determine the domain and the invariants if not specified
    if transformation.domain is None:
        if transformation.spec is not None:
            from core.transformations import TransformationDomain
            domain_value = transformation.spec.domain.value if hasattr(transformation.spec.domain, 'value') else str(transformation.spec.domain)
            transformation = Transformation(
                name=transformation.name,
                changes=transformation.changes,
                relation_changes=transformation.relation_changes,
                spec=transformation.spec,
                source=transformation.source,
                timestamp=transformation.timestamp,
                domain=domain_value,
                invariants=transformation.invariants,
            )
        else:
            transformation = Transformation(
                name=transformation.name,
                changes=transformation.changes,
                relation_changes=transformation.relation_changes,
                spec=transformation.spec,
                source=transformation.source,
                timestamp=transformation.timestamp,
                domain="world",
                invariants=transformation.invariants,
            )
    
    if transformation.invariants is None and transformation.spec is not None:
        transformation = Transformation(
            name=transformation.name,
            changes=transformation.changes,
            relation_changes=transformation.relation_changes,
            spec=transformation.spec,
            source=transformation.source,
            timestamp=transformation.timestamp,
            domain=transformation.domain,
            invariants=transformation.spec.preserves,
        )
    
    # 1. Check all properties before any mutation.
    for change in transformation.changes:
        if change.entity not in world.entities:
            raise ValueError(
                f"Unknown entity: {change.entity!r}"
            )

        entity = world.entities[change.entity]
        current = entity.properties.get(change.property)

        if current != change.before:
            raise ValueError(
                f"Transformation '{transformation.name}' impossible: "
                f"{change.entity}.{change.property} is {current!r}, "
                f"expected {change.before!r}"
            )

    # 2. Check the relational changes.
    for change in transformation.relation_changes:
        from core.world import Relation
        relation = Relation(
            change.subject,
            change.predicate,
            change.object,
        )

        exists = relation in world.relations

        if change.action == "add" and exists:
            raise ValueError(
                f"Relation already present: {relation}"
            )

        if change.action == "remove" and not exists:
            raise ValueError(
                f"Relation absent: {relation}"
            )

        if change.action not in {"add", "remove"}:
            raise ValueError(
                f"Unknown relational action: {change.action}"
            )

    # 3. Temporarily apply the property changes.
    for change in transformation.changes:
        world.entities[change.entity].properties[
            change.property
        ] = change.after

    # 4. Temporarily apply the relations.
    applied_relations = []

    for change in transformation.relation_changes:
        from core.world import Relation
        relation = Relation(
            change.subject,
            change.predicate,
            change.object,
        )

        if change.action == "add":
            world.relations.append(relation)

        elif change.action == "remove":
            world.relations.remove(relation)

        applied_relations.append((change, relation))

    # 5. Check the constraints.
    errors = world.validate()

    if errors:
        # rollback relations
        for change, relation in reversed(applied_relations):
            if change.action == "add":
                world.relations.remove(relation)

            elif change.action == "remove":
                world.relations.append(relation)

        # rollback properties
        for change in reversed(transformation.changes):
            world.entities[change.entity].properties[
                change.property
            ] = change.before

        raise ValueError(
            "Transformation refused: "
            + "; ".join(errors)
        )

    # 6. Record in the history of the World
    world.record(transformation)

    # A known topology remains explicit for the declared homeomorphisms;
    # otherwise it is invalidated without claiming to recompute it.
    from core.topology import TopologyState
    topology_preserved = bool(transformation.invariants and transformation.invariants.preserves_topology)
    touched = {change.entity for change in transformation.changes}
    if transformation.domain == "geometry":
        touched.update(name for name, entity in world.entities.items()
                       if "x" in entity.properties and "y" in entity.properties)
    for entity_name in touched:
        entity = world.entities[entity_name]
        if entity.topology is not None:
            entity.topology_state = TopologyState.PRESERVED if topology_preserved else TopologyState.UNKNOWN
    
    # 7. Record in the TransformGraph if available
    if hasattr(world, 'transform_graph') and world.transform_graph is not None:
        from core.transformations import DeclarativeTransformationSpec
        from core.invariants import InvariantSet
        from core.transform_graph import TransformRecord
        # Use the spec if available, otherwise create a minimal representation
        spec_to_record = transformation.spec if transformation.spec else DeclarativeTransformationSpec(
            name=transformation.name,
            domain=transformation.domain or "world",
            preserves=transformation.invariants or InvariantSet(set())
        )
        after_snapshot = world.get_state_snapshot() if hasattr(world, 'get_state_snapshot') else None
        source_name = spec_to_record.parameters.get("source_name", transformation.name)
        record = TransformRecord(
            spec=spec_to_record, executable=transformation,
            source_type=transformation.source.value, source_name=source_name,
            domain=transformation.domain or "world", time=transformation.timestamp,
            before=before_snapshot, after=after_snapshot,
            changed_properties=tuple(f"{c.entity}.{c.property}" for c in transformation.changes),
        )
        world.transform_graph.apply_transformation(
            spec_to_record,
            config=after_snapshot,
            topology=None,
            record=record,
        )

    return transformation

"""
Transformations exécutables pour WCE.

Ce module fournit la représentation concrète des transformations qui modifient
l'état du World, avec un lien vers leur spécification conceptuelle (TransformationSpec).
"""

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING
from enum import Enum
import time

if TYPE_CHECKING:
    from core.transformations import TransformationSpec, InvariantSet


class TransformationSource(Enum):
    """Source/provenance d'une transformation."""
    ACTION = "action"         # Produite par une Action (Lift, Push, etc.)
    LAW = "law"               # Produite par une Law (Gravity, ForceDynamics, etc.)
    INTERACTION = "interaction"  # Produite par une Interaction (collision, support, etc.)
    USER = "user"             # Commande directe utilisateur
    SYSTEM = "system"         # Transformation système interne
    MACRO = "macro"           # MacroTransformation
    COMPOSITE = "composite"   # CompositeTransformation
    RULE = "rule"             # Règle symbolique / logique


@dataclass(frozen=True)
class Change:
    """Changement d'une propriété d'entité."""
    entity: str
    property: str
    before: Any
    after: Any


@dataclass(frozen=True)
class RelationChange:
    """Changement d'une relation entre entités."""
    action: str  # "add" or "remove"
    subject: str
    predicate: str
    object: str


@dataclass
class Transformation:
    """
    Transformation exécutable sur un World.
    
    C'est la représentation concrète qui modifie réellement l'état du monde.
    Elle peut être liée à une TransformationSpec pour la sémantique.
    """
    name: str
    changes: list[Change] = field(default_factory=list)
    relation_changes: list[RelationChange] = field(default_factory=list)
    
    # Lien vers la spécification conceptuelle
    spec: Optional['TransformationSpec'] = None
    source: TransformationSource = TransformationSource.SYSTEM
    
    # Métadonnées pour le graphe
    timestamp: Optional[float] = None
    domain: Optional[str] = None
    invariants: Optional['InvariantSet'] = None
    
    def with_spec(self, spec: 'TransformationSpec', source: TransformationSource = None) -> 'Transformation':
        """Associe cette transformation à une TransformationSpec."""
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
        """Définit la source de cette transformation."""
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
        """Vérifie si la transformation ne fait rien."""
        return not self.changes and not self.relation_changes
    
    def __repr__(self) -> str:
        spec_name = f" spec={self.spec.name}" if self.spec else ""
        return f"Transformation(name='{self.name}', source={self.source.value}{spec_name})"


def apply(world, transformation: Transformation):
    """
    Applique une transformation à un World avec validation et rollback.
    
    Cette fonction enregistre aussi la transformation dans le TransformGraph du World
    si disponible.
    """
    # Définir le timestamp si non spécifié
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
    
    # Déterminer le domaine et les invariants si non spécifiés
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
    
    # 1. Vérifier toutes les propriétés avant toute mutation.
    for change in transformation.changes:
        if change.entity not in world.entities:
            raise ValueError(
                f"Entité inconnue : {change.entity!r}"
            )

        entity = world.entities[change.entity]
        current = entity.properties.get(change.property)

        if current != change.before:
            raise ValueError(
                f"Transformation '{transformation.name}' impossible : "
                f"{change.entity}.{change.property} vaut {current!r}, "
                f"attendu {change.before!r}"
            )

    # 2. Vérifier les changements relationnels.
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
                f"Relation déjà présente : {relation}"
            )

        if change.action == "remove" and not exists:
            raise ValueError(
                f"Relation absente : {relation}"
            )

        if change.action not in {"add", "remove"}:
            raise ValueError(
                f"Action relationnelle inconnue : {change.action}"
            )

    # 3. Appliquer temporairement les changements de propriétés.
    for change in transformation.changes:
        world.entities[change.entity].properties[
            change.property
        ] = change.after

    # 4. Appliquer temporairement les relations.
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

    # 5. Vérifier les contraintes.
    errors = world.validate()

    if errors:
        # rollback relations
        for change, relation in reversed(applied_relations):
            if change.action == "add":
                world.relations.remove(relation)

            elif change.action == "remove":
                world.relations.append(relation)

        # rollback propriétés
        for change in reversed(transformation.changes):
            world.entities[change.entity].properties[
                change.property
            ] = change.before

        raise ValueError(
            "Transformation refusée : "
            + "; ".join(errors)
        )

    # 6. Enregistrer dans l'historique du World
    world.record(transformation)

    # Une topologie connue reste explicite pour les homéomorphismes déclarés ;
    # sinon elle est invalidée sans prétendre la recalculer.
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
    
    # 7. Enregistrer dans le TransformGraph si disponible
    if hasattr(world, 'transform_graph') and world.transform_graph is not None:
        from core.transformations import DeclarativeTransformationSpec
        from core.invariants import InvariantSet
        from core.transform_graph import TransformRecord
        # Utiliser la spec si disponible, sinon créer une representation minimale
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

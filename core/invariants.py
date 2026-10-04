"""
Invariants pour WCE Transformation Algebra.

Ce module définit les invariants fondamentaux qui peuvent être préservés
ou modifiés par les transformations.

Un invariant est une propriété qui reste inchangée sous certaines transformations.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Set, Optional


class Invariant(Enum):
    """
    Invariants fondamentaux supportés par WCE.
    
    GEOMETRY:
    - DISTANCE: distances entre points sont préservées
    - ANGLE: angles entre vecteurs sont préservés  
    - ORIENTATION: orientation (sens horaire/anti-horaire) est préservée
    - AREA: aire/surface est préservée
    - VOLUME: volume (futur 3D) est préservé
    
    TOPOLOGY:
    - CONNECTIVITY: composantes connexes sont préservées
    - HOLES: nombre de trous est préservé
    - BOUNDARY_COMPONENTS: nombre de composantes de frontière est préservé
    - DIMENSION: dimension de l'objet est préservée
    
    PHYSICS:
    - ENERGY: énergie totale est préservée
    - MOMENTUM: quantité de mouvement est préservée
    - MASS: masse est préservée
    
    GENERAL:
    - TIME: temporal coherence
    - ENTITY_IDENTITY: identité des entités est préservée
    - RELATIONS: relations entre entités sont préservées
    """
    # Géométrie
    DISTANCE = auto()
    ANGLE = auto()
    ORIENTATION = auto()
    AREA = auto()
    VOLUME = auto()
    
    # Topologie
    CONNECTIVITY = auto()
    HOLES = auto()
    BOUNDARY_COMPONENTS = auto()
    DIMENSION = auto()
    
    # Physique
    ENERGY = auto()
    MOMENTUM = auto()
    MASS = auto()
    
    # Général
    TIME = auto()
    ENTITY_IDENTITY = auto()
    RELATIONS = auto()

    # Algèbre symbolique
    EQUALITY = auto()
    SOLUTION_SET = auto()
    DEGREE = auto()
    SIGN = auto()
    DOMAIN = auto()

    # Modèle neuronal (auto-transformation)
    SLOW_WEIGHTS = auto()
    MODEL_FUNCTION = auto()
    MODULE_SET = auto()
    MODEL_TOPOLOGY = auto()

    # Séquences symboliques
    SYMBOL_MULTISET = auto()
    SEQUENCE_LENGTH = auto()


# Ensembles d'invariants prédéfinis pour les catégories courantes

# Transformations géométriques qui préservent tout
RIGID_TRANSFORMATION_INVARIANTS: Set[Invariant] = {
    Invariant.DISTANCE,
    Invariant.ANGLE,
    Invariant.ORIENTATION,
    Invariant.AREA,
    Invariant.VOLUME,
    Invariant.CONNECTIVITY,
    Invariant.HOLES,
    Invariant.BOUNDARY_COMPONENTS,
    Invariant.DIMENSION,
}

# Transformations qui préservent la topologie (mais pas nécessairement la géométrie)
TOPOLOGY_PRESERVING_INVARIANTS: Set[Invariant] = {
    Invariant.CONNECTIVITY,
    Invariant.HOLES,
    Invariant.BOUNDARY_COMPONENTS,
    Invariant.DIMENSION,
}

# Transformations qui préservent la géométrie (isométries)
ISOMETRY_INVARIANTS: Set[Invariant] = {
    Invariant.DISTANCE,
    Invariant.ANGLE,
    Invariant.ORIENTATION,
}

# Transformations affines
AFFINE_INVARIANTS: Set[Invariant] = {
    Invariant.CONNECTIVITY,
    Invariant.HOLES,
    Invariant.BOUNDARY_COMPONENTS,
    Invariant.DIMENSION,
}


@dataclass(frozen=True)
class InvariantSet:
    """
    Ensemble d'invariants avec des opérations utilitaires.
    
    Permet de représenter quels invariants sont préservés par une transformation
    ou un ensemble de transformations.
    """
    invariants: Set[Invariant] = field(default_factory=set)
    
    def preserves(self, invariant: Invariant) -> bool:
        """Vérifie si un invariant spécifique est préservé."""
        return invariant in self.invariants
    
    def preserves_all(self, invariants: Set[Invariant]) -> bool:
        """Vérifie si tous les invariants spécifiés sont préservés."""
        return invariants.issubset(self.invariants)
    
    def preserves_any(self, invariants: Set[Invariant]) -> bool:
        """Vérifie si au moins un des invariants spécifiés est préservé."""
        return bool(self.invariants & invariants)
    
    def union(self, other: 'InvariantSet') -> 'InvariantSet':
        """Union avec un autre ensemble d'invariants."""
        return InvariantSet(self.invariants | other.invariants)
    
    def intersection(self, other: 'InvariantSet') -> 'InvariantSet':
        """Intersection avec un autre ensemble d'invariants."""
        return InvariantSet(self.invariants & other.invariants)
    
    def lost(self, original: 'InvariantSet') -> 'InvariantSet':
        """Retourne les invariants perdus par rapport à l'original."""
        return InvariantSet(original.invariants - self.invariants)
    
    @property
    def preserves_geometry(self) -> bool:
        """Préserve la géométrie (distances et angles)."""
        return self.preserves_all(ISOMETRY_INVARIANTS)
    
    @property
    def preserves_topology(self) -> bool:
        """Préserve la topologie."""
        return self.preserves_all(TOPOLOGY_PRESERVING_INVARIANTS)
    
    @property
    def preserves_orientation(self) -> bool:
        """Préserve l'orientation."""
        return self.preserves(Invariant.ORIENTATION)
    
    def __repr__(self) -> str:
        invariant_names = sorted([i.name for i in self.invariants])
        return f"InvariantSet({invariant_names})"


# Ensembles prédéfinis pour les types de transformations courants

IDENTITY_INVARIANTS = InvariantSet(RIGID_TRANSFORMATION_INVARIANTS | {Invariant.MASS, Invariant.ENERGY, Invariant.ENTITY_IDENTITY, Invariant.RELATIONS, Invariant.TIME})

EQUATION_EQUIVALENCE_INVARIANTS = InvariantSet({
    Invariant.EQUALITY, Invariant.SOLUTION_SET, Invariant.DOMAIN,
})

TRANSLATION_INVARIANTS = InvariantSet(RIGID_TRANSFORMATION_INVARIANTS | {Invariant.ORIENTATION})

ROTATION_INVARIANTS = InvariantSet(RIGID_TRANSFORMATION_INVARIANTS)

UNIFORM_SCALE_INVARIANTS = InvariantSet({
    Invariant.ANGLE,
    Invariant.ORIENTATION,
    Invariant.CONNECTIVITY,
    Invariant.HOLES,
    Invariant.BOUNDARY_COMPONENTS,
    Invariant.DIMENSION,
    Invariant.MASS,
    Invariant.ENTITY_IDENTITY,
    Invariant.RELATIONS,
})

NON_UNIFORM_SCALE_INVARIANTS = InvariantSet(TOPOLOGY_PRESERVING_INVARIANTS | {Invariant.MASS, Invariant.ENTITY_IDENTITY, Invariant.RELATIONS})

REFLECTION_INVARIANTS = InvariantSet({
    Invariant.DISTANCE,
    Invariant.ANGLE,
    Invariant.AREA,
    Invariant.VOLUME,
} | TOPOLOGY_PRESERVING_INVARIANTS | {Invariant.MASS, Invariant.ENTITY_IDENTITY, Invariant.RELATIONS})

SHEAR_INVARIANTS = InvariantSet(TOPOLOGY_PRESERVING_INVARIANTS | {Invariant.MASS, Invariant.ENTITY_IDENTITY, Invariant.RELATIONS})

STRETCH_INVARIANTS = NON_UNIFORM_SCALE_INVARIANTS

# Transformations qui peuvent modifier la topologie
CUT_INVARIANTS = InvariantSet({
    Invariant.DIMENSION,
    Invariant.MASS,
    Invariant.ENTITY_IDENTITY,
})

MERGE_INVARIANTS = InvariantSet({
    Invariant.DIMENSION,
    Invariant.ENTITY_IDENTITY,
})

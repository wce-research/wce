"""
Invariants for WCE Transformation Algebra.

This module defines the fundamental invariants that can be preserved
or modified by transformations.

An invariant is a property that remains unchanged under certain transformations.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Set, Optional


class Invariant(Enum):
    """
    Fundamental invariants supported by WCE.
    
    GEOMETRY:
    - DISTANCE: distances between points are preserved
    - ANGLE: angles between vectors are preserved  
    - ORIENTATION: orientation (clockwise/anticlockwise) is preserved
    - AREA: area/surface is preserved
    - VOLUME: volume (future 3D) is preserved
    
    TOPOLOGY:
    - CONNECTIVITY: connected components are preserved
    - HOLES: number of holes is preserved
    - BOUNDARY_COMPONENTS: number of boundary components is preserved
    - DIMENSION: dimension of the object is preserved
    
    PHYSICS:
    - ENERGY: total energy is preserved
    - MOMENTUM: momentum is preserved
    - MASS: mass is preserved
    
    GENERAL:
    - TIME: temporal coherence
    - ENTITY_IDENTITY: identity of the entities is preserved
    - RELATIONS: relations between entities are preserved
    """
    # Geometry
    DISTANCE = auto()
    ANGLE = auto()
    ORIENTATION = auto()
    AREA = auto()
    VOLUME = auto()
    
    # Topology
    CONNECTIVITY = auto()
    HOLES = auto()
    BOUNDARY_COMPONENTS = auto()
    DIMENSION = auto()
    
    # Physics
    ENERGY = auto()
    MOMENTUM = auto()
    MASS = auto()
    
    # General
    TIME = auto()
    ENTITY_IDENTITY = auto()
    RELATIONS = auto()

    # Symbolic algebra
    EQUALITY = auto()
    SOLUTION_SET = auto()
    DEGREE = auto()
    SIGN = auto()
    DOMAIN = auto()

    # Neural model (self-transformation)
    SLOW_WEIGHTS = auto()
    MODEL_FUNCTION = auto()
    MODULE_SET = auto()
    MODEL_TOPOLOGY = auto()

    # Symbolic sequences
    SYMBOL_MULTISET = auto()
    SEQUENCE_LENGTH = auto()


# Predefined invariant sets for the common categories

# Geometric transformations that preserve everything
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

# Transformations that preserve the topology (but not necessarily the geometry)
TOPOLOGY_PRESERVING_INVARIANTS: Set[Invariant] = {
    Invariant.CONNECTIVITY,
    Invariant.HOLES,
    Invariant.BOUNDARY_COMPONENTS,
    Invariant.DIMENSION,
}

# Transformations that preserve the geometry (isometries)
ISOMETRY_INVARIANTS: Set[Invariant] = {
    Invariant.DISTANCE,
    Invariant.ANGLE,
    Invariant.ORIENTATION,
}

# Affine transformations
AFFINE_INVARIANTS: Set[Invariant] = {
    Invariant.CONNECTIVITY,
    Invariant.HOLES,
    Invariant.BOUNDARY_COMPONENTS,
    Invariant.DIMENSION,
}


@dataclass(frozen=True)
class InvariantSet:
    """
    Set of invariants with utility operations.
    
    Makes it possible to represent which invariants are preserved by a transformation
    or a set of transformations.
    """
    invariants: Set[Invariant] = field(default_factory=set)
    
    def preserves(self, invariant: Invariant) -> bool:
        """Checks whether a specific invariant is preserved."""
        return invariant in self.invariants
    
    def preserves_all(self, invariants: Set[Invariant]) -> bool:
        """Checks whether all the specified invariants are preserved."""
        return invariants.issubset(self.invariants)
    
    def preserves_any(self, invariants: Set[Invariant]) -> bool:
        """Checks whether at least one of the specified invariants is preserved."""
        return bool(self.invariants & invariants)
    
    def union(self, other: 'InvariantSet') -> 'InvariantSet':
        """Union with another set of invariants."""
        return InvariantSet(self.invariants | other.invariants)
    
    def intersection(self, other: 'InvariantSet') -> 'InvariantSet':
        """Intersection with another set of invariants."""
        return InvariantSet(self.invariants & other.invariants)
    
    def lost(self, original: 'InvariantSet') -> 'InvariantSet':
        """Returns the invariants lost relative to the original."""
        return InvariantSet(original.invariants - self.invariants)
    
    @property
    def preserves_geometry(self) -> bool:
        """Preserves the geometry (distances and angles)."""
        return self.preserves_all(ISOMETRY_INVARIANTS)
    
    @property
    def preserves_topology(self) -> bool:
        """Preserves the topology."""
        return self.preserves_all(TOPOLOGY_PRESERVING_INVARIANTS)
    
    @property
    def preserves_orientation(self) -> bool:
        """Preserves the orientation."""
        return self.preserves(Invariant.ORIENTATION)
    
    def __repr__(self) -> str:
        invariant_names = sorted([i.name for i in self.invariants])
        return f"InvariantSet({invariant_names})"


# Predefined sets for the common transformation types

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

# Transformations that may modify the topology
CUT_INVARIANTS = InvariantSet({
    Invariant.DIMENSION,
    Invariant.MASS,
    Invariant.ENTITY_IDENTITY,
})

MERGE_INVARIANTS = InvariantSet({
    Invariant.DIMENSION,
    Invariant.ENTITY_IDENTITY,
})

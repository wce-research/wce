"""
Topology for WCE.

This module provides lightweight abstractions to represent the topological invariants
of geometric shapes.

Topology focuses on the properties that are preserved by continuous
deformations, such as:
- The number of connected components
- The number of holes
- The boundary components

This is distinct from geometry, which deals with exact positions and distances.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TopologyState(Enum):
    """Level of knowledge of the topology attached to an entity."""
    KNOWN = "known"
    PRESERVED = "preserved"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TopologySignature:
    """
    Topological signature of a shape or of a set of shapes.
    
    A topological signature captures the fundamental properties that are
    invariant under continuous deformation.
    
    Attributes:
        connected_components: Number of distinct connected components
        holes: Number of holes (enclosed void components)
        boundary_components: Number of separate boundary components
    
    Examples:
        - A segment: connected_components=1, holes=0, boundary_components=1
        - A disc: connected_components=1, holes=0, boundary_components=1
        - A ring: connected_components=1, holes=1, boundary_components=2
        - Two disjoint objects: connected_components=2, holes=0, boundary_components=2
    """
    connected_components: int = 1
    holes: int = 0
    boundary_components: int = 1
    
    def __post_init__(self):
        if self.connected_components < 0:
            raise ValueError(f"connected_components must be >= 0, not {self.connected_components}")
        if self.holes < 0:
            raise ValueError(f"holes must be >= 0, not {self.holes}")
        if self.boundary_components < 0:
            raise ValueError(f"boundary_components must be >= 0, not {self.boundary_components}")
        # An object with holes must have at least one connected component
        if self.holes > 0 and self.connected_components < 1:
            raise ValueError("An object with holes must have at least one connected component")
    
    def is_solid(self) -> bool:
        """Indicates whether the shape is solid (without holes)."""
        return self.holes == 0
    
    def is_simple(self) -> bool:
        """Indicates whether the shape is topologically simple (one component, without holes)."""
        return self.connected_components == 1 and self.holes == 0
    
    def is_connected(self) -> bool:
        """Indicates whether the shape is connected (a single component)."""
        return self.connected_components == 1


# Topological signatures for the basic shapes

# Segment: one dimension, connected, without holes, one boundary
SEGMENT_TOPOLOGY = TopologySignature(connected_components=1, holes=0, boundary_components=1)

# Disc (2D circle): connected, without holes, one boundary (the circle)
DISK_TOPOLOGY = TopologySignature(connected_components=1, holes=0, boundary_components=1)

# Circle (the boundary only): same as segment (1D)
CIRCLE_TOPOLOGY = SEGMENT_TOPOLOGY

# Ring (disc with a hole): connected, one hole, two boundaries
RING_TOPOLOGY = TopologySignature(connected_components=1, holes=1, boundary_components=2)

# Two disjoint objects
DISJOINT_TWO_TOPOLOGY = TopologySignature(connected_components=2, holes=0, boundary_components=2)

# Point: zero dimension, but we treat it as a connected component
POINT_TOPOLOGY = TopologySignature(connected_components=1, holes=0, boundary_components=0)


# Signatures for compositions

def combine_topologies(*signatures: TopologySignature) -> TopologySignature:
    """
    Combines several topological signatures (disjoint union).
    
    For a disjoint union of objects:
        - The connected components add up
        - The holes add up
        - The boundary components add up
    """
    total_cc = sum(s.connected_components for s in signatures)
    total_holes = sum(s.holes for s in signatures)
    total_bc = sum(s.boundary_components for s in signatures)
    
    return TopologySignature(
        connected_components=total_cc,
        holes=total_holes,
        boundary_components=total_bc,
    )


# Transformation of the topology

@dataclass(frozen=True)
class TopologyPreservation:
    """
    Indicates which topological properties are preserved by a transformation.
    
    Attributes:
        preserves_connected_components: The transformation does not change the number of connected components
        preserves_holes: The transformation does not change the number of holes
        preserves_boundary: The transformation does not change the structure of the boundary
        preserves_orientation: The transformation preserves the local orientation
    """
    preserves_connected_components: bool = True
    preserves_holes: bool = True
    preserves_boundary: bool = True
    preserves_orientation: bool = True
    
    @classmethod
    def full(cls) -> "TopologyPreservation":
        """All properties are preserved (e.g. translation, rotation)."""
        return cls(
            preserves_connected_components=True,
            preserves_holes=True,
            preserves_boundary=True,
            preserves_orientation=True,
        )
    
    @classmethod
    def continuous(cls) -> "TopologyPreservation":
        """Continuous deformation (e.g. stretching): preserves connectivity but may change the holes."""
        return cls(
            preserves_connected_components=True,
            preserves_holes=True,  # A continuous deformation does not create holes
            preserves_boundary=False,  # May change the structure of the boundary
            preserves_orientation=True,
        )
    
    @classmethod
    def cutting(cls) -> "TopologyPreservation":
        """Cutting: may change connectivity and create new boundaries."""
        return cls(
            preserves_connected_components=False,
            preserves_holes=False,
            preserves_boundary=False,
            preserves_orientation=True,
        )
    
    @classmethod
    def reflection(cls) -> "TopologyPreservation":
        """Reflection: preserves the topology but reverses the orientation."""
        return cls(
            preserves_connected_components=True,
            preserves_holes=True,
            preserves_boundary=True,
            preserves_orientation=False,
        )
    
    @classmethod
    def identity(cls) -> "TopologyPreservation":
        """Identity transformation: everything is preserved."""
        return cls.full()


# Functions to obtain the topological preservation of different transformations

def get_topology_preservation_translation() -> TopologyPreservation:
    """Translation preserves all topological properties."""
    return TopologyPreservation.full()


def get_topology_preservation_rotation() -> TopologyPreservation:
    """Rotation preserves all topological properties."""
    return TopologyPreservation.full()


def get_topology_preservation_uniform_scale(sx: float, sy: float) -> TopologyPreservation:
    """Uniform scale preserves the topology except possibly the orientation."""
    if sx < 0 and sy < 0:
        # Negative scale in both directions: preserves the orientation
        return TopologyPreservation.full()
    elif sx < 0 or sy < 0:
        # Negative scale in a single direction: reverses the orientation
        return TopologyPreservation(
            preserves_connected_components=True,
            preserves_holes=True,
            preserves_boundary=True,
            preserves_orientation=False,
        )
    else:
        return TopologyPreservation.full()


def get_topology_preservation_shear() -> TopologyPreservation:
    """Shear is a continuous deformation."""
    return TopologyPreservation.continuous()


def get_topology_preservation_general_scale(sx: float, sy: float) -> TopologyPreservation:
    """Non-uniform scale may change the orientation."""
    if sx * sy < 0:
        # Negative scale in opposite directions: may reverse the orientation
        return TopologyPreservation(
            preserves_connected_components=True,
            preserves_holes=True,
            preserves_boundary=True,
            preserves_orientation=False,
        )
    else:
        return TopologyPreservation.full()

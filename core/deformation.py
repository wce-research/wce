"""
Deformations for WCE.

This module provides abstractions to represent the geometric deformations
that transform shapes while preserving (or modifying in a controlled way)
their topological properties.

A deformation is a transformation that can be applied to a geometry
or to a set of points.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from core.transform import Transform2D
from core.topology import TopologyPreservation, TopologySignature


@dataclass(frozen=True)
class Deformation(ABC):
    """
    Base abstraction for a geometric deformation.
    
    A deformation is a parametric transformation that can be applied
    to a geometric shape.
    """
    
    @abstractmethod
    def get_transform(self) -> Transform2D:
        """
        Returns the transformation associated with this deformation.
        
        This transformation can be applied to the points to obtain
        the deformed shape.
        """
        pass
    
    @abstractmethod
    def apply_to_point(self, x: float, y: float) -> tuple[float, float]:
        """
        Applies the deformation to a point.
        
        This is equivalent to get_transform().transform_point(x, y), but can be
        overridden for more complex deformations.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the deformation."""
        pass
    
    @property
    @abstractmethod
    def topology_preservation(self) -> TopologyPreservation:
        """Which topological properties are preserved by this deformation."""
        pass
    
    def applies_to_geometry(self, topology: TopologySignature) -> bool:
        """
        Indicates whether this deformation can be applied to a geometry with
        a given topological signature.
        
        By default, deformations that preserve connectivity can be
        applied to any geometry.
        """
        if not self.topology_preservation.preserves_connected_components:
            # If the deformation does not preserve the connected components,
            # it can create discontinuities, so caution is required
            return True  # By default, we allow it
        return True


@dataclass(frozen=True)
class StretchDeformation(Deformation):
    """
    Stretch deformation (non-uniform scale).
    
    Stretches the shape differently in the x and y directions.
    
    Attributes:
        stretch_x: Stretch factor in the x direction (1.0 = no stretch)
        stretch_y: Stretch factor in the y direction (1.0 = no stretch)
    """
    stretch_x: float = 1.0
    stretch_y: float = 1.0
    
    @property
    def name(self) -> str:
        return f"stretch({self.stretch_x}, {self.stretch_y})"
    
    @property
    def topology_preservation(self) -> TopologyPreservation:
        """Computes the topological preservation as a function of the stretches."""
        # The stretch preserves connectivity and holes
        # but can reverse the orientation if the stretches are negative
        if self.stretch_x * self.stretch_y < 0:
            # Opposite stretches: reverses the orientation
            return TopologyPreservation(
                preserves_connected_components=True,
                preserves_holes=True,
                preserves_boundary=True,
                preserves_orientation=False,
            )
        else:
            # Stretches in the same direction: preserves the orientation
            return TopologyPreservation.full()
    
    def get_transform(self) -> Transform2D:
        """Returns the scale transformation."""
        return Transform2D.scale(self.stretch_x, self.stretch_y)
    
    def apply_to_point(self, x: float, y: float) -> tuple[float, float]:
        """Applies the stretch to a point."""
        return (x * self.stretch_x, y * self.stretch_y)





@dataclass(frozen=True)
class IdentityDeformation(Deformation):
    """
    Identity deformation (changes nothing).
    """
    
    @property
    def name(self) -> str:
        return "identity"
    
    @property
    def topology_preservation(self) -> TopologyPreservation:
        return TopologyPreservation.full()
    
    def get_transform(self) -> Transform2D:
        return Transform2D.identity()
    
    def apply_to_point(self, x: float, y: float) -> tuple[float, float]:
        return (x, y)


# Composite deformations

def compose_deformations(*deformations: Deformation) -> Deformation:
    """
    Combines several deformations into a single one.
    
    The order of application is: first deformation, then second, etc.
    (compose_deformations(d1, d2, d3) applies d1, then d2, then d3)
    """
    if not deformations:
        return IdentityDeformation()
    
    if len(deformations) == 1:
        return deformations[0]
    
    # For simple deformations, we can create a combined StretchDeformation
    # or ShearDeformation
    # For now, we create a CompositeDeformation
    return CompositeDeformation(deformations)


@dataclass(frozen=True)
class CompositeDeformation(Deformation):
    """
    Composite deformation (combination of several deformations).
    """
    components: tuple[Deformation, ...] = ()
    
    @property
    def name(self) -> str:
        return f"composite[{', '.join(d.name for d in self.components)}]"
    
    @property
    def topology_preservation(self) -> TopologyPreservation:
        """
        The topological preservation is the intersection of the preservations
        of all the components.
        """
        # Start with everything preserved
        result = TopologyPreservation.full()
        
        for d in self.components:
            # Combine: an attribute is preserved only if all the components preserve it
            result = TopologyPreservation(
                preserves_connected_components=(
                    result.preserves_connected_components and d.topology_preservation.preserves_connected_components
                ),
                preserves_holes=(
                    result.preserves_holes and d.topology_preservation.preserves_holes
                ),
                preserves_boundary=(
                    result.preserves_boundary and d.topology_preservation.preserves_boundary
                ),
                preserves_orientation=(
                    result.preserves_orientation and d.topology_preservation.preserves_orientation
                ),
            )
        
        return result
    
    def get_transform(self) -> Transform2D:
        """Returns the composed transformation."""
        result = Transform2D.identity()
        for d in reversed(self.components):
            # Compose in order: first deformation applied first
            # So we must compose in reverse order
            result = result.compose(d.get_transform())
        return result
    
    def apply_to_point(self, x: float, y: float) -> tuple[float, float]:
        """Applies all the deformations in order."""
        result_x, result_y = x, y
        for d in self.components:
            result_x, result_y = d.apply_to_point(result_x, result_y)
        return (result_x, result_y)


# Predefined basic deformations

STRETCH_X2 = StretchDeformation(stretch_x=2.0, stretch_y=1.0)
STRETCH_Y2 = StretchDeformation(stretch_x=1.0, stretch_y=2.0)
STRETCH_X05_Y2 = StretchDeformation(stretch_x=0.5, stretch_y=2.0)
IDENTITY_DEFORMATION = IdentityDeformation()


# Examples of deformations that change the geometry

def circle_to_ellipse(stretch_x: float, stretch_y: float) -> StretchDeformation:
    """
    Creates a deformation that transforms a circle into an ellipse.
    
    A circle has a simple topology (disc: connected_components=1, holes=0).
    An ellipse has the same topology.
    """
    return StretchDeformation(stretch_x=stretch_x, stretch_y=stretch_y)


def square_to_rectangle(width: float, height: float) -> StretchDeformation:
    """
    Creates a deformation that transforms a square into a rectangle.
    
    Both have the same topology.
    """
    # Assume the square has a size of 1x1
    return StretchDeformation(stretch_x=width, stretch_y=height)

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.transform import Transform2D


@dataclass(frozen=True)
class VerticalPlaneCollider:
    """
    Collider for a vertical plane in 2D.
    
    Represents a vertical boundary at a given x position,
    with a normal indicating the forbidden direction.
    
    Examples:
        - Right wall at x=5: normal = (-1, 0) → blocks positions > 5
        - Left wall at x=-5: normal = (+1, 0) → blocks positions < -5
    
    This collider does NOT contain the material properties (restitution, etc.)
    which are handled separately by the Material class.
    
    Attributes:
        x: position of the plane on the x axis
        normal_x: x component of the normal (must be +1 or -1)
    """
    x: float
    normal_x: float = -1.0

    def __post_init__(self):
        if self.normal_x not in (-1.0, 1.0):
            raise ValueError(
                f"normal_x must be +1.0 or -1.0, not {self.normal_x}"
            )

    @property
    def is_left_facing(self) -> bool:
        """The collider faces left (positive normal)."""
        return self.normal_x > 0

    @property
    def is_right_facing(self) -> bool:
        """The collider faces right (negative normal)."""
        return self.normal_x < 0

    def get_bound(self) -> tuple[float, float]:
        """
        Returns (min_x, max_x) for this collider.
        
        A right-facing collider (normal_x=-1) blocks at max_x = x
        A left-facing collider (normal_x=+1) blocks at min_x = x
        """
        if self.is_right_facing:
            return (-float("inf"), self.x)
        else:
            return (self.x, float("inf"))


@dataclass(frozen=True)
class PlaneCollider:
    """
    Generic collider for an arbitrarily oriented 2D plane.
    
    In its LOCAL frame, the plane passes through the origin with a unit normal.
    The transformation (Transform2D) places and orients the plane in the world.
    
    Local representation (simple):
        - normal: (normal_x, normal_y) - UNIT vector in the local frame
        - offset: distance from the local origin to the plane (along the normal)
                  offset >= 0 → forbidden side is in the direction of the normal
    
    Examples:
        - Vertical wall at x=5: local_normal=(1,0), offset=0, transform=translation(5,0)
          The plane is at x=5 in the world, with world normal=(1,0) -> blocks x > 5
          
        - Vertical wall at x=5 (other convention): local_normal=(-1,0), offset=0, 
          transform=translation(5,0)
          The plane is at x=5, world normal=(-1,0) -> blocks x < 5
          
        - Horizontal floor at y=0: local_normal=(0,1), offset=0, transform=translation(0,0)
          
        - Plane inclined at 45°: local_normal=(0,1), offset=0, 
          transform=rotation(pi/4) * translation(...)
    
    Attributes:
        normal_x: x component of the LOCAL normal (must form a unit vector)
        normal_y: y component of the LOCAL normal
        offset: distance from the local origin to the plane (may be negative)
        transform: transformation to the world frame (optional, identity by default)
    
    Note: The normal must be a UNIT vector. If not, it will be normalised.
    """
    normal_x: float
    normal_y: float
    offset: float = 0.0
    transform: Optional["Transform2D"] = None

    def __post_init__(self):
        # Normalise the normal
        length = (self.normal_x**2 + self.normal_y**2) ** 0.5
        if length > 0:
            object.__setattr__(self, 'normal_x', self.normal_x / length)
            object.__setattr__(self, 'normal_y', self.normal_y / length)
    
    @property
    def normal(self) -> tuple[float, float]:
        """Returns the local normal (already normalised)."""
        return (self.normal_x, self.normal_y)
    
    def world_normal(self, transform: Optional["Transform2D"] = None) -> tuple[float, float]:
        """
        Computes the normal in the world frame.
        
        Uses self.transform if transform is not provided.
        """
        t = transform or self.transform or Transform2D.identity()
        return t.transform_normal(self.normal_x, self.normal_y)
    
    def signed_distance(self, x: float, y: float, transform: Optional["Transform2D"] = None) -> float:
        """
        Computes the SIGNED distance from a world point to the plane.
        
        Meaning of the sign:
            - distance > 0: point on the side of the normal
            - distance < 0: point on the side opposite to the normal
            - distance = 0: point on the plane
        
        The distance is measured in the local frame after inverse transformation.
        """
        t = transform or self.transform or Transform2D.identity()
        
        # Transform the world point to the local frame
        inv_t = t.inverse()
        x_local, y_local = inv_t.transform_point(x, y)
        
        # Signed distance in the local frame: dot(normal, point) - offset
        return self.normal_x * x_local + self.normal_y * y_local - self.offset
    
    def is_behind(self, x: float, y: float, transform: Optional["Transform2D"] = None) -> bool:
        """
        Indicates whether a world point is "behind" the plane (forbidden side).
        
        By convention: the forbidden side is the one the normal points to.
        So a point is behind if signed_distance > 0.
        """
        return self.signed_distance(x, y, transform) > 0


@dataclass(frozen=True)
class Material:
    """
    Material associated with a collider.
    
    Separates the physical properties from the geometry.
    Can be extended with friction, density, etc.
    """
    restitution: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError(
                f"restitution must be in [0, 1], not {self.restitution}"
            )


# Default material for colliders without a specific material
DEFAULT_MATERIAL = Material(restitution=0.0)


def vertical_plane_to_plane_collider(
    vpc: VerticalPlaneCollider
) -> PlaneCollider:
    """
    Converts a VerticalPlaneCollider into an equivalent PlaneCollider.
    
    Convention:
        - VerticalPlaneCollider with normal_x=-1 (right wall) → 
          PlaneCollider with local_normal=(1,0), offset=0, transform=translation(vpc.x, 0)
        - VerticalPlaneCollider with normal_x=+1 (left wall) →
          PlaneCollider with local_normal=(-1,0), offset=0, transform=translation(vpc.x, 0)
    
    Both represent the same plane, but with opposite normals.
    """
    from core.transform import Transform2D
    
    if vpc.is_right_facing:
        # Right wall: blocks x > vpc.x
        # Local normal pointing right (1,0), offset=0, translation to (vpc.x, 0)
        return PlaneCollider(
            normal_x=1.0,
            normal_y=0.0,
            offset=0.0,
            transform=Transform2D.translation(vpc.x, 0.0),
        )
    else:
        # Left wall: blocks x < vpc.x
        # Local normal pointing left (-1,0), offset=0, translation to (vpc.x, 0)
        return PlaneCollider(
            normal_x=-1.0,
            normal_y=0.0,
            offset=0.0,
            transform=Transform2D.translation(vpc.x, 0.0),
        )

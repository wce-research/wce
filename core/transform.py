"""
2D transformations for WCE.

Conventions:
- Rotation: radians, counter-clockwise (mathematical standard)
- Point: (x, y)
- Vector: (vx, vy) - unchanged by translation
- Normal: unit vector, transformed differently with non-uniform scale
"""

from dataclasses import dataclass, field
from math import cos, sin
from typing import Tuple


@dataclass(frozen=True)
class Transform2D:
    """
    Affine 2D transformation: translation + rotation + scale.
    
    Represents:
        p_world = R(θ) * S(sx, sy) * p_local + translation
        
    For a local point (x, y):
        x_world = tx + sx * (cos(θ) * x - sin(θ) * y)
        y_world = ty + sy * (sin(θ) * x + cos(θ) * y)
    
    Attributes:
        tx: translation in x
        ty: translation in y
        rotation: angle in radians (0 = no rotation)
        scale_x: scale factor in x (1.0 = normal size)
        scale_y: scale factor in y (1.0 = normal size)
    """
    tx: float = 0.0
    ty: float = 0.0
    rotation: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    
    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transforms a POINT from the local frame to the world frame.
        A point is affected by the translation.
        """
        c = cos(self.rotation)
        s = sin(self.rotation)
        
        x_rot = c * x - s * y
        y_rot = s * x + c * y
        
        x_scaled = self.scale_x * x_rot
        y_scaled = self.scale_y * y_rot
        
        x_world = self.tx + x_scaled
        y_world = self.ty + y_scaled
        
        return (x_world, y_world)
    
    def transform_vector(self, vx: float, vy: float) -> Tuple[float, float]:
        """
        Transforms a VECTOR from the local frame to the world frame.
        A vector IS NOT affected by the translation.
        """
        c = cos(self.rotation)
        s = sin(self.rotation)
        
        vx_rot = c * vx - s * vy
        vy_rot = s * vx + c * vy
        
        vx_world = self.scale_x * vx_rot
        vy_world = self.scale_y * vy_rot
        
        return (vx_world, vy_world)
    
    def transform_normal(self, nx: float, ny: float) -> Tuple[float, float]:
        """
        Transforms a NORMAL from the local frame to the world frame.
        
        For a normal, the correct transformation uses the inverse transpose
        of the transformation matrix.
        """
        c = cos(self.rotation)
        s = sin(self.rotation)
        
        nx_rot = c * nx - s * ny
        ny_rot = s * nx + c * ny
        
        nx_world = nx_rot / self.scale_x
        ny_world = ny_rot / self.scale_y
        
        length = (nx_world**2 + ny_world**2) ** 0.5
        if length > 0:
            nx_world /= length
            ny_world /= length
        
        return (nx_world, ny_world)
    
    def inverse(self) -> "Transform2D":
        """
        Returns the inverse transformation.
        T^-1(p_world) = p_local
        """
        inv_rotation = -self.rotation
        
        inv_scale_x = 1.0 / self.scale_x if self.scale_x != 0 else 0.0
        inv_scale_y = 1.0 / self.scale_y if self.scale_y != 0 else 0.0
        
        c = cos(inv_rotation)
        s = sin(inv_rotation)
        
        tx_rot = c * self.tx - s * self.ty
        ty_rot = s * self.tx + c * self.ty
        
        tx_inv = -tx_rot / self.scale_x if self.scale_x != 0 else 0.0
        ty_inv = -ty_rot / self.scale_y if self.scale_y != 0 else 0.0
        
        return Transform2D(
            tx=tx_inv,
            ty=ty_inv,
            rotation=inv_rotation,
            scale_x=inv_scale_x,
            scale_y=inv_scale_y,
        )
    
    def compose(self, other: "Transform2D") -> "Transform2D":
        """
        Composes two transformations: T_final = self * other.
        T_final(p) = self(other(p))
        """
        final_rotation = self.rotation + other.rotation
        
        final_scale_x = self.scale_x * other.scale_x
        final_scale_y = self.scale_y * other.scale_y
        
        c_self = cos(self.rotation)
        s_self = sin(self.rotation)
        
        c_other = cos(other.rotation)
        s_other = sin(other.rotation)
        
        other_tx_in_self = c_self * other.tx * self.scale_x - s_self * other.ty * self.scale_y
        other_ty_in_self = s_self * other.tx * self.scale_x + c_self * other.ty * self.scale_y
        
        final_tx = self.tx + other_tx_in_self
        final_ty = self.ty + other_ty_in_self
        
        return Transform2D(
            tx=final_tx,
            ty=final_ty,
            rotation=final_rotation,
            scale_x=final_scale_x,
            scale_y=final_scale_y,
        )
    
    @classmethod
    def translation(cls, tx: float, ty: float) -> "Transform2D":
        """Creates a pure translation transformation."""
        return cls(tx=tx, ty=ty, rotation=0.0, scale_x=1.0, scale_y=1.0)
    
    @classmethod
    def rotation(cls, angle_radians: float) -> "Transform2D":
        """Creates a pure rotation transformation."""
        return cls(tx=0.0, ty=0.0, rotation=angle_radians, scale_x=1.0, scale_y=1.0)
    
    @classmethod
    def scale(cls, sx: float, sy: float = None) -> "Transform2D":
        """Creates a pure scale transformation."""
        if sy is None:
            sy = sx
        return cls(tx=0.0, ty=0.0, rotation=0.0, scale_x=sx, scale_y=sy)
    
    @classmethod
    def identity(cls) -> "Transform2D":
        """Identity transformation."""
        return cls(tx=0.0, ty=0.0, rotation=0.0, scale_x=1.0, scale_y=1.0)

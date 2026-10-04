"""
Transformations 2D pour WCE.

Conventions :
- Rotation : radians, sens anti-horaire (standard mathématique)
- Point : (x, y)
- Vecteur : (vx, vy) - inchangé par la translation
- Normale : vecteur unitaire, transformé différemment avec scale non-uniforme
"""

from dataclasses import dataclass, field
from math import cos, sin
from typing import Tuple


@dataclass(frozen=True)
class Transform2D:
    """
    Transformation 2D affine : translation + rotation + scale.
    
    Représente :
        p_world = R(θ) * S(sx, sy) * p_local + translation
        
    Pour un point local (x, y) :
        x_world = tx + sx * (cos(θ) * x - sin(θ) * y)
        y_world = ty + sy * (sin(θ) * x + cos(θ) * y)
    
    Attributs :
        tx : translation en x
        ty : translation en y
        rotation : angle en radians (0 = aucune rotation)
        scale_x : facteur d'échelle en x (1.0 = taille normale)
        scale_y : facteur d'échelle en y (1.0 = taille normale)
    """
    tx: float = 0.0
    ty: float = 0.0
    rotation: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    
    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transforme un POINT du repère local vers le repère monde.
        Un point est affecté par la translation.
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
        Transforme un VECTEUR du repère local vers le repère monde.
        Un vecteur N'EST PAS affecté par la translation.
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
        Transforme une NORMALE du repère local vers le repère monde.
        
        Pour une normale, la transformation correcte utilise l'inverse transposée
        de la matrice de transformation.
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
        Retourne la transformation inverse.
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
        Compose deux transformations : T_final = self * other.
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
        """Crée une transformation de translation pure."""
        return cls(tx=tx, ty=ty, rotation=0.0, scale_x=1.0, scale_y=1.0)
    
    @classmethod
    def rotation(cls, angle_radians: float) -> "Transform2D":
        """Crée une transformation de rotation pure."""
        return cls(tx=0.0, ty=0.0, rotation=angle_radians, scale_x=1.0, scale_y=1.0)
    
    @classmethod
    def scale(cls, sx: float, sy: float = None) -> "Transform2D":
        """Crée une transformation d'échelle pure."""
        if sy is None:
            sy = sx
        return cls(tx=0.0, ty=0.0, rotation=0.0, scale_x=sx, scale_y=sy)
    
    @classmethod
    def identity(cls) -> "Transform2D":
        """Transformation identité."""
        return cls(tx=0.0, ty=0.0, rotation=0.0, scale_x=1.0, scale_y=1.0)

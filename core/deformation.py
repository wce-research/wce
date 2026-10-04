"""
Déformations pour WCE.

Ce module fournit des abstractions pour représenter les déformations géométriques
qui transforment les formes tout en préservant (ou modifiant de manière contrôlée)
leurs propriétés topologiques.

Une déformation est une transformation qui peut être appliquée à une géométrie
ou à un ensemble de points.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from core.transform import Transform2D
from core.topology import TopologyPreservation, TopologySignature


@dataclass(frozen=True)
class Deformation(ABC):
    """
    Abstraction de base pour une déformation géométrique.
    
    Une déformation est une transformation paramétrique qui peut être appliquée
    à une forme géométrique.
    """
    
    @abstractmethod
    def get_transform(self) -> Transform2D:
        """
        Retourne la transformation associée à cette déformation.
        
        Cette transformation peut être appliquée aux points pour obtenir
        la forme déformée.
        """
        pass
    
    @abstractmethod
    def apply_to_point(self, x: float, y: float) -> tuple[float, float]:
        """
        Applique la déformation à un point.
        
        C'est équivalent à get_transform().transform_point(x, y), mais peut être
        surchargé pour des déformations plus complexes.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Nom de la déformation."""
        pass
    
    @property
    @abstractmethod
    def topology_preservation(self) -> TopologyPreservation:
        """Quelles propriétés topologiques sont préservées par cette déformation."""
        pass
    
    def applies_to_geometry(self, topology: TopologySignature) -> bool:
        """
        Indique si cette déformation peut être appliquée à une géométrie avec
        une signature topologique donnée.
        
        Par défaut, les déformations qui préservent la connectivité peuvent être
        appliquées à n'importe quelle géométrie.
        """
        if not self.topology_preservation.preserves_connected_components:
            # Si la déformation ne préserve pas les composantes connexes,
            # elle peut créer des discontinuités, donc il faut être prudent
            return True  # Par défaut, on autorise
        return True


@dataclass(frozen=True)
class StretchDeformation(Deformation):
    """
    Déformation d'étirement (scale non-uniforme).
    
    Étire la forme différemment dans les directions x et y.
    
    Attributs :
        stretch_x : Facteur d'étirement dans la direction x (1.0 = pas d'étirement)
        stretch_y : Facteur d'étirement dans la direction y (1.0 = pas d'étirement)
    """
    stretch_x: float = 1.0
    stretch_y: float = 1.0
    
    @property
    def name(self) -> str:
        return f"stretch({self.stretch_x}, {self.stretch_y})"
    
    @property
    def topology_preservation(self) -> TopologyPreservation:
        """Calcule la préservation topologique en fonction des étirements."""
        # Le stretch préserve la connectivité et les trous
        # mais peut inverser l'orientation si les étirements sont négatifs
        if self.stretch_x * self.stretch_y < 0:
            # Étirements opposés : inverse l'orientation
            return TopologyPreservation(
                preserves_connected_components=True,
                preserves_holes=True,
                preserves_boundary=True,
                preserves_orientation=False,
            )
        else:
            # Étirements dans la même direction : préserve l'orientation
            return TopologyPreservation.full()
    
    def get_transform(self) -> Transform2D:
        """Retourne la transformation de scale."""
        return Transform2D.scale(self.stretch_x, self.stretch_y)
    
    def apply_to_point(self, x: float, y: float) -> tuple[float, float]:
        """Applique l'étirement à un point."""
        return (x * self.stretch_x, y * self.stretch_y)





@dataclass(frozen=True)
class IdentityDeformation(Deformation):
    """
    Déformation identité (ne change rien).
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


# Déformations composites

def compose_deformations(*deformations: Deformation) -> Deformation:
    """
    Combine plusieurs déformations en une seule.
    
    L'ordre d'application est : première déformation, puis deuxième, etc.
    (compose_deformations(d1, d2, d3) applique d1, puis d2, puis d3)
    """
    if not deformations:
        return IdentityDeformation()
    
    if len(deformations) == 1:
        return deformations[0]
    
    # Pour les déformations simples, nous pouvons créer une StretchDeformation
    # ou ShearDeformation combinée
    # Pour l'instant, nous créons une CompositeDeformation
    return CompositeDeformation(deformations)


@dataclass(frozen=True)
class CompositeDeformation(Deformation):
    """
    Déformation composite (combinaison de plusieurs déformations).
    """
    components: tuple[Deformation, ...] = ()
    
    @property
    def name(self) -> str:
        return f"composite[{', '.join(d.name for d in self.components)}]"
    
    @property
    def topology_preservation(self) -> TopologyPreservation:
        """
        La préservation topologique est l'intersection des préservations
        de toutes les composantes.
        """
        # Commencer avec tout préservé
        result = TopologyPreservation.full()
        
        for d in self.components:
            # Combiner : un attribut n'est préservé que si toutes les composantes le préservent
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
        """Retourne la transformation composée."""
        result = Transform2D.identity()
        for d in reversed(self.components):
            # Composer dans l'ordre : première déformation appliquée en premier
            # Donc nous devons composer dans l'ordre inverse
            result = result.compose(d.get_transform())
        return result
    
    def apply_to_point(self, x: float, y: float) -> tuple[float, float]:
        """Applique toutes les déformations dans l'ordre."""
        result_x, result_y = x, y
        for d in self.components:
            result_x, result_y = d.apply_to_point(result_x, result_y)
        return (result_x, result_y)


# Déformations de base pré-définies

STRETCH_X2 = StretchDeformation(stretch_x=2.0, stretch_y=1.0)
STRETCH_Y2 = StretchDeformation(stretch_x=1.0, stretch_y=2.0)
STRETCH_X05_Y2 = StretchDeformation(stretch_x=0.5, stretch_y=2.0)
IDENTITY_DEFORMATION = IdentityDeformation()


# Exemples de déformations qui changent la géométrie

def circle_to_ellipse(stretch_x: float, stretch_y: float) -> StretchDeformation:
    """
    Crée une déformation qui transforme un cercle en une ellipse.
    
    Un cercle a une topologie simple (disque : connected_components=1, holes=0).
    Une ellipse a la même topologie.
    """
    return StretchDeformation(stretch_x=stretch_x, stretch_y=stretch_y)


def square_to_rectangle(width: float, height: float) -> StretchDeformation:
    """
    Crée une déformation qui transforme un carré en un rectangle.
    
    Les deux ont la même topologie.
    """
    # Supposons que le carré a une taille de 1x1
    return StretchDeformation(stretch_x=width, stretch_y=height)

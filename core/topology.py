"""
Topologie pour WCE.

Ce module fournit des abstractions légères pour représenter les invariants topologiques
des formes géométriques.

La topologie se concentre sur les propriétés qui sont préservées par les déformations
continues, comme :
- Le nombre de composantes connexes
- Le nombre de trous
- Les composantes de frontière

Cci est distinct de la géométrie, qui traite des positions et distances exactes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TopologyState(Enum):
    """Niveau de connaissance de la topologie attachée à une entité."""
    KNOWN = "known"
    PRESERVED = "preserved"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TopologySignature:
    """
    Signature topologique d'une forme ou d'un ensemble de formes.
    
    Une signature topologique capture les propriétés fondamentales qui sont
    invariantes par déformation continue.
    
    Attributs :
        connected_components : Nombre de composantes connexes distinctes
        holes : Nombre de trous (composantes de vide enfermées)
        boundary_components : Nombre de composantes de frontière séparées
    
    Exemples :
        - Un segment : connected_components=1, holes=0, boundary_components=1
        - Un disque : connected_components=1, holes=0, boundary_components=1
        - Un anneau : connected_components=1, holes=1, boundary_components=2
        - Deux objets disjoints : connected_components=2, holes=0, boundary_components=2
    """
    connected_components: int = 1
    holes: int = 0
    boundary_components: int = 1
    
    def __post_init__(self):
        if self.connected_components < 0:
            raise ValueError(f"connected_components doit être >= 0, pas {self.connected_components}")
        if self.holes < 0:
            raise ValueError(f"holes doit être >= 0, pas {self.holes}")
        if self.boundary_components < 0:
            raise ValueError(f"boundary_components doit être >= 0, pas {self.boundary_components}")
        # Un objet avec des trous doit avoir au moins une composante connexe
        if self.holes > 0 and self.connected_components < 1:
            raise ValueError("Un objet avec des trous doit avoir au moins une composante connexe")
    
    def is_solid(self) -> bool:
        """Indique si la forme est solide (sans trous)."""
        return self.holes == 0
    
    def is_simple(self) -> bool:
        """Indique si la forme est topologiquement simple (une composante, sans trous)."""
        return self.connected_components == 1 and self.holes == 0
    
    def is_connected(self) -> bool:
        """Indique si la forme est connexe (une seule composante)."""
        return self.connected_components == 1


# Signatures topologiques pour les formes de base

# Segment : une dimension, connecté, sans trous, une frontière
SEGMENT_TOPOLOGY = TopologySignature(connected_components=1, holes=0, boundary_components=1)

# Disque (cercle 2D) : connecté, sans trous, une frontière (le cercle)
DISK_TOPOLOGY = TopologySignature(connected_components=1, holes=0, boundary_components=1)

# Cercle (la frontière seulement) : même que segment (1D)
CIRCLE_TOPOLOGY = SEGMENT_TOPOLOGY

# Anneau (disque avec trou) : connecté, un trou, deux frontières
RING_TOPOLOGY = TopologySignature(connected_components=1, holes=1, boundary_components=2)

# Deux objets disjoints
DISJOINT_TWO_TOPOLOGY = TopologySignature(connected_components=2, holes=0, boundary_components=2)

# Point : zero dimension, mais nous le traitons comme une composante connexe
POINT_TOPOLOGY = TopologySignature(connected_components=1, holes=0, boundary_components=0)


# Signatures pour compositions

def combine_topologies(*signatures: TopologySignature) -> TopologySignature:
    """
    Combine plusieurs signatures topologiques (union disjointe).
    
    Pour une union disjointe d'objets :
        - Les composantes connexes s'additionnent
        - Les trous s'additionnent
        - Les composantes de frontière s'additionnent
    """
    total_cc = sum(s.connected_components for s in signatures)
    total_holes = sum(s.holes for s in signatures)
    total_bc = sum(s.boundary_components for s in signatures)
    
    return TopologySignature(
        connected_components=total_cc,
        holes=total_holes,
        boundary_components=total_bc,
    )


# Transformation de la topologie

@dataclass(frozen=True)
class TopologyPreservation:
    """
    Indique quelles propriétés topologiques sont préservées par une transformation.
    
    Attributs :
        preserves_connected_components : La transformation ne change pas le nombre de composantes connexes
        preserves_holes : La transformation ne change pas le nombre de trous
        preserves_boundary : La transformation ne change pas la structure de la frontière
        preserves_orientation : La transformation préserve l'orientation locale
    """
    preserves_connected_components: bool = True
    preserves_holes: bool = True
    preserves_boundary: bool = True
    preserves_orientation: bool = True
    
    @classmethod
    def full(cls) -> "TopologyPreservation":
        """Toutes les propriétés sont préservées (ex: translation, rotation)."""
        return cls(
            preserves_connected_components=True,
            preserves_holes=True,
            preserves_boundary=True,
            preserves_orientation=True,
        )
    
    @classmethod
    def continuous(cls) -> "TopologyPreservation":
        """Déformation continue (ex: étirement) : préserve la connectivité mais peut changer les trous."""
        return cls(
            preserves_connected_components=True,
            preserves_holes=True,  # Une déformation continue ne crée pas de trous
            preserves_boundary=False,  # Peut changer la structure de la frontière
            preserves_orientation=True,
        )
    
    @classmethod
    def cutting(cls) -> "TopologyPreservation":
        """Découpe : peut changer la connectivité et créer de nouvelles frontières."""
        return cls(
            preserves_connected_components=False,
            preserves_holes=False,
            preserves_boundary=False,
            preserves_orientation=True,
        )
    
    @classmethod
    def reflection(cls) -> "TopologyPreservation":
        """Réflexion : préserve la topologie mais inverse l'orientation."""
        return cls(
            preserves_connected_components=True,
            preserves_holes=True,
            preserves_boundary=True,
            preserves_orientation=False,
        )
    
    @classmethod
    def identity(cls) -> "TopologyPreservation":
        """Transformation identité : tout est préservé."""
        return cls.full()


# Fonctions pour obtenir la préservation topologique de différentes transformations

def get_topology_preservation_translation() -> TopologyPreservation:
    """La translation préserve toutes les propriétés topologiques."""
    return TopologyPreservation.full()


def get_topology_preservation_rotation() -> TopologyPreservation:
    """La rotation préserve toutes les propriétés topologiques."""
    return TopologyPreservation.full()


def get_topology_preservation_uniform_scale(sx: float, sy: float) -> TopologyPreservation:
    """Le scale uniforme préserve la topologie sauf éventuellement l'orientation."""
    if sx < 0 and sy < 0:
        # Scale négatif dans les deux directions : préserve l'orientation
        return TopologyPreservation.full()
    elif sx < 0 or sy < 0:
        # Scale négatif dans une seule direction : inverse l'orientation
        return TopologyPreservation(
            preserves_connected_components=True,
            preserves_holes=True,
            preserves_boundary=True,
            preserves_orientation=False,
        )
    else:
        return TopologyPreservation.full()


def get_topology_preservation_shear() -> TopologyPreservation:
    """Le cisaillement est une déformation continue."""
    return TopologyPreservation.continuous()


def get_topology_preservation_general_scale(sx: float, sy: float) -> TopologyPreservation:
    """Le scale non-uniforme peut changer l'orientation."""
    if sx * sy < 0:
        # Scale négatif dans des directions opposées : peut inverser l'orientation
        return TopologyPreservation(
            preserves_connected_components=True,
            preserves_holes=True,
            preserves_boundary=True,
            preserves_orientation=False,
        )
    else:
        return TopologyPreservation.full()

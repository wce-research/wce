from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.transform import Transform2D


@dataclass(frozen=True)
class VerticalPlaneCollider:
    """
    Collider pour un plan vertical en 2D.
    
    Représente une frontière verticale à une position x donnée,
    avec une normale indiquant la direction interdite.
    
    Exemples :
        - Mur droit à x=5 : normale = (-1, 0) → bloque les positions > 5
        - Mur gauche à x=-5 : normale = (+1, 0) → bloque les positions < -5
    
    Ce collider ne contient PAS les propriétés matérielles (restitution, etc.)
    qui sont gérées séparément par la classe Material.
    
    Attributs :
        x : position du plan sur l'axe x
        normal_x : composante x de la normale (doit être +1 ou -1)
    """
    x: float
    normal_x: float = -1.0

    def __post_init__(self):
        if self.normal_x not in (-1.0, 1.0):
            raise ValueError(
                f"normal_x doit être +1.0 ou -1.0, pas {self.normal_x}"
            )

    @property
    def is_left_facing(self) -> bool:
        """Le collider regarde vers la gauche (normale positive)."""
        return self.normal_x > 0

    @property
    def is_right_facing(self) -> bool:
        """Le collider regarde vers la droite (normale négative)."""
        return self.normal_x < 0

    def get_bound(self) -> tuple[float, float]:
        """
        Retourne (min_x, max_x) pour ce collider.
        
        Un collider right-facing (normal_x=-1) bloque à max_x = x
        Un collider left-facing (normal_x=+1) bloque à min_x = x
        """
        if self.is_right_facing:
            return (-float("inf"), self.x)
        else:
            return (self.x, float("inf"))


@dataclass(frozen=True)
class PlaneCollider:
    """
    Collider générique pour un plan 2D orienté arbitrairement.
    
    Dans son repère LOCAL, le plan passe par l'origine avec une normale unitaire.
    La transformation (Transform2D) place et oriente le plan dans le monde.
    
    Représentation locale (simple) :
        - normale : (normal_x, normal_y) - vecteur UNITAIRE dans le repère local
        - offset : distance de l'origine locale au plan (le long de la normale)
                  offset >= 0 → côté interdit est dans la direction de la normale
    
    Exemples :
        - Mur vertical à x=5 : normale_locale=(1,0), offset=0, transform=translation(5,0)
          Le plan est à x=5 dans le monde, avec normale monde=(1,0) -> bloque x > 5
          
        - Mur vertical à x=5 (autre convention) : normale_locale=(-1,0), offset=0, 
          transform=translation(5,0)
          Le plan est à x=5, normale monde=(-1,0) -> bloque x < 5
          
        - Sol horizontal à y=0 : normale_locale=(0,1), offset=0, transform=translation(0,0)
          
        - Plan incliné à 45° : normale_locale=(0,1), offset=0, 
          transform=rotation(pi/4) * translation(...)
    
    Attributs :
        normal_x : composante x de la normale LOCAL (doit former un vecteur unitaire)
        normal_y : composante y de la normale LOCAL
        offset : distance de l'origine locale au plan (peut être négatif)
        transform : transformation vers le repère monde (optionnel, identité par défaut)
    
    Note : La normale doit être UNITAIRE. Si non, elle sera normalisée.
    """
    normal_x: float
    normal_y: float
    offset: float = 0.0
    transform: Optional["Transform2D"] = None

    def __post_init__(self):
        # Normaliser la normale
        length = (self.normal_x**2 + self.normal_y**2) ** 0.5
        if length > 0:
            object.__setattr__(self, 'normal_x', self.normal_x / length)
            object.__setattr__(self, 'normal_y', self.normal_y / length)
    
    @property
    def normal(self) -> tuple[float, float]:
        """Retourne la normale locale (déjà normalisée)."""
        return (self.normal_x, self.normal_y)
    
    def world_normal(self, transform: Optional["Transform2D"] = None) -> tuple[float, float]:
        """
        Calcule la normale dans le repère monde.
        
        Utilise self.transform si transform n'est pas fourni.
        """
        t = transform or self.transform or Transform2D.identity()
        return t.transform_normal(self.normal_x, self.normal_y)
    
    def signed_distance(self, x: float, y: float, transform: Optional["Transform2D"] = None) -> float:
        """
        Calcule la distance SIGNÉE d'un point monde au plan.
        
        Signification du signe :
            - distance > 0 : point du côté de la normale
            - distance < 0 : point du côté opposé à la normale
            - distance = 0 : point sur le plan
        
        La distance est mesurée dans le repère local après transformation inverse.
        """
        t = transform or self.transform or Transform2D.identity()
        
        # Transformer le point monde vers le repère local
        inv_t = t.inverse()
        x_local, y_local = inv_t.transform_point(x, y)
        
        # Distance signée dans le repère local : dot(normal, point) - offset
        return self.normal_x * x_local + self.normal_y * y_local - self.offset
    
    def is_behind(self, x: float, y: float, transform: Optional["Transform2D"] = None) -> bool:
        """
        Indique si un point monde est "derrière" le plan (côté interdit).
        
        Par convention : le côté interdit est celui où la normale pointe.
        Donc un point est derrière si signed_distance > 0.
        """
        return self.signed_distance(x, y, transform) > 0


@dataclass(frozen=True)
class Material:
    """
    Matériau associé à un collider.
    
    Sépare les propriétés physiques de la géométrie.
    Peut être étendu avec friction, densité, etc.
    """
    restitution: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError(
                f"restitution doit être dans [0, 1], pas {self.restitution}"
            )


# Matériau par défaut pour les colliders sans matériau spécifique
DEFAULT_MATERIAL = Material(restitution=0.0)


def vertical_plane_to_plane_collider(
    vpc: VerticalPlaneCollider
) -> PlaneCollider:
    """
    Convertit un VerticalPlaneCollider en PlaneCollider équivalent.
    
    Convention :
        - VerticalPlaneCollider avec normal_x=-1 (mur droit) → 
          PlaneCollider avec normale_locale=(1,0), offset=0, transform=translation(vpc.x, 0)
        - VerticalPlaneCollider avec normal_x=+1 (mur gauche) →
          PlaneCollider avec normale_locale=(-1,0), offset=0, transform=translation(vpc.x, 0)
    
    Les deux représentent le même plan, mais avec des normales opposées.
    """
    from core.transform import Transform2D
    
    if vpc.is_right_facing:
        # Mur droit : bloque x > vpc.x
        # Normale locale vers la droite (1,0), offset=0, translation à (vpc.x, 0)
        return PlaneCollider(
            normal_x=1.0,
            normal_y=0.0,
            offset=0.0,
            transform=Transform2D.translation(vpc.x, 0.0),
        )
    else:
        # Mur gauche : bloque x < vpc.x
        # Normale locale vers la gauche (-1,0), offset=0, translation à (vpc.x, 0)
        return PlaneCollider(
            normal_x=-1.0,
            normal_y=0.0,
            offset=0.0,
            transform=Transform2D.translation(vpc.x, 0.0),
        )

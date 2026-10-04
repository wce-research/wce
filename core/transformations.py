"""
WCE Transformation Algebra.

Ce module fournit l'abstraction fondamentale des transformations comme concept
de premier ordre dans WCE.

Une transformation est une opération qui peut être appliquée à une configuration
pour produire une nouvelle configuration, avec des invariants et des changements
explicites.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Set, List, Tuple
from enum import Enum

from core.invariants import Invariant, InvariantSet, IDENTITY_INVARIANTS
from core.topology import TopologySignature, DISK_TOPOLOGY


class TransformationDomain(Enum):
    """Domaine d'application d'une transformation."""
    GEOMETRY = "geometry"           # Transformations géométriques (2D/3D)
    TOPOLOGY = "topology"           # Transformations topologiques
    PHYSICS = "physics"             # Transformations physiques
    WORLD = "world"                 # Transformations du monde (état)
    TIME = "time"                   # Transformations temporelles
    LANGUAGE = "language"            # Transformations linguistiques
    ALGEBRA = "algebra"              # Transformations symboliques
    SEQUENCE = "sequence"            # Transformations de séquences symboliques
    MODEL = "model"                  # Auto-transformations du modèle neuronal
    MEASUREMENT = "measurement"      # Mesure, changement de base / de référentiel, conditionnement (algèbre de mesure)
    GENERIC = "generic"             # Domaine générique/indéfini


@dataclass(frozen=True)
class TransformationSpec(ABC):
    """
    Spécification abstraite d'une transformation WCE.
    
    Une TransformationSpec décrit une transformation de manière conceptuelle,
    indépendamment de son implémentation concrète.
    
    Attributs :
        name : Nom descriptif de la transformation
        domain : Domaine d'application (geometry, physics, etc.)
        parameters : Dictionnaire des paramètres de la transformation
        invertible : Indique si la transformation a une inverse
        preserves : Ensemble d'invariants préservés par cette transformation
        changes : Ensemble de propriétés qui changent (optionnel)
        cost : Coût estimé de l'application (pour optimisation)
    """
    name: str
    domain: TransformationDomain = TransformationDomain.GENERIC
    parameters: Dict[str, Any] = field(default_factory=dict)
    invertible: bool = True
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet(set()))
    changes: Set[str] = field(default_factory=set)
    cost: float = 1.0
    preconditions: Tuple[Any, ...] = ()
    postconditions: Tuple[Any, ...] = ()
    
    @abstractmethod
    def apply(self, *args, **kwargs) -> Any:
        """
        Applique la transformation à un objet donné.
        
        L'implémentation dépend du domaine et du type de transformation.
        """
        pass
    
    def is_invertible(self) -> bool:
        """Indique si cette transformation a une inverse."""
        return self.invertible
    
    def preserves_invariant(self, invariant: Invariant) -> bool:
        """Vérifie si un invariant spécifique est préservé."""
        return self.preserves.preserves(invariant)
    
    def preserves_geometry(self) -> bool:
        """Indique si la géométrie est préservée (isométrie)."""
        return self.preserves.preserves_geometry
    
    def preserves_topology(self) -> bool:
        """Indique si la topologie est préservée."""
        return self.preserves.preserves_topology
    
    def get_inverse(self) -> Optional['TransformationSpec']:
        """
        Retourne la transformation inverse si elle existe.
        
        Doit être implémenté par les sous-classes pour fournir l'inverse.
        """
        if not self.invertible:
            return None
        # Par défaut, retourne None - les sous-classes doivent implémenter
        return None

    def compile(self, world) -> 'Transformation':
        """Compile la description en mutation atomique du World."""
        raise TypeError(f"{type(self).__name__} ne sait pas compiler un World")

    def applicability(self, context):
        """Retourne TRUE/FALSE/UNKNOWN pour les préconditions déclarées."""
        from core.reasoning import TruthValue
        values = [condition.evaluate(context) for condition in self.preconditions]
        if any(value is TruthValue.FALSE for value in values): return TruthValue.FALSE
        if any(value is TruthValue.UNKNOWN for value in values): return TruthValue.UNKNOWN
        return TruthValue.TRUE
    
    def compose(self, other: 'TransformationSpec') -> 'CompositeTransformation':
        """
        Crée une composition de cette transformation avec une autre.
        
        Le résultat est une CompositeTransformation qui applique les deux
        transformations dans l'ordre : self puis other (self est appliqué en premier).
        """
        return CompositeTransformation(transformations=(self, other))
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', domain={self.domain.value})"
    
    def __matmul__(self, other: 'TransformationSpec') -> 'CompositeTransformation':
        """Opérateur de composition : T1 @ T2 = T2 ∘ T1 (T1 appliqué en premier)."""
        return self.compose(other)


@dataclass(frozen=True)
class IdentityTransformation(TransformationSpec):
    """
    Transformation identité (ne change rien).
    
    Préserve tous les invariants.
    """
    name: str = "identity"
    domain: TransformationDomain = TransformationDomain.GENERIC
    preserves: InvariantSet = field(default_factory=lambda: IDENTITY_INVARIANTS)
    
    def apply(self, *args, **kwargs) -> Any:
        """Ne change pas l'input."""
        if len(args) == 1:
            return args[0]
        return args
    
    def get_inverse(self) -> 'IdentityTransformation':
        """L'inverse de l'identité est l'identité."""
        return IdentityTransformation()


@dataclass(frozen=True)
class CompositeTransformation(TransformationSpec):
    """
    Composition de plusieurs transformations.
    
    Applique les transformations dans l'ordre de la liste :
    [T1, T2, T3] signifie T3 ∘ T2 ∘ T1 (T1 est appliqué en premier).
    
    Les invariants préservés sont l'intersection des invariants de toutes
    les transformations composées.
    """
    name: str = "composite"
    transformations: Tuple[TransformationSpec, ...] = ()
    domain: TransformationDomain = TransformationDomain.GENERIC
    
    def __post_init__(self):
        # Calculer les paramètres agrégés
        params = {}
        for i, t in enumerate(self.transformations):
            params[f'transform_{i}'] = t.name
        object.__setattr__(self, 'parameters', params)
        
        # Calculer les invariants préservés (intersection)
        if self.transformations:
            preserved = self.transformations[0].preserves
            for t in self.transformations[1:]:
                preserved = preserved.intersection(t.preserves)
            object.__setattr__(self, 'preserves', preserved)
        else:
            object.__setattr__(self, 'preserves', InvariantSet(set()))
        
        # Invertible si toutes les composantes le sont
        invertible = all(t.invertible for t in self.transformations)
        object.__setattr__(self, 'invertible', invertible)
        
        # Coût cumulé
        cost = sum(t.cost for t in self.transformations)
        object.__setattr__(self, 'cost', cost)
        
        # Changements agrégés
        changes = set()
        for t in self.transformations:
            changes.update(t.changes)
        object.__setattr__(self, 'changes', changes)
    
    def apply(self, obj: Any, *args, **kwargs) -> Any:
        """Applique toutes les transformations dans l'ordre."""
        result = obj
        for t in self.transformations:
            result = t.apply(result, *args, **kwargs)
        return result
    
    def get_inverse(self) -> Optional['CompositeTransformation']:
        """
        Retourne la composition inverse (ordre inversé, chaque transformation inversée).
        """
        if not self.invertible:
            return None
        
        # Créer la liste des inverses dans l'ordre inverse
        inverse_transforms = []
        for t in reversed(self.transformations):
            inv = t.get_inverse()
            if inv is None:
                return None  # Ne devrait pas arriver si invertible=True
            inverse_transforms.append(inv)
        
        return CompositeTransformation(
            name=f"inverse({self.name})",
            transformations=tuple(inverse_transforms),
            domain=self.domain
        )

    def compile(self, world):
        if hasattr(world, "expression"):
            from core.transformation import Change, Transformation, TransformationSource
            before = world.expression
            after = self.apply(before)
            return Transformation(self.name, [Change("__math__", "expression", before, after)],
                                  spec=self, source=TransformationSource.COMPOSITE,
                                  domain=self.domain.value, invariants=self.preserves)
        return _compile_geometry_spec(self, world)
    
    def get_transformations(self) -> List[TransformationSpec]:
        """Retourne la liste des transformations composées."""
        return list(self.transformations)
    
    def get_depth(self) -> int:
        """Retourne le nombre de transformations dans la composition."""
        return len(self.transformations)
    
    def __repr__(self) -> str:
        transforms_str = " @ ".join([t.name for t in self.transformations])
        return f"CompositeTransformation({transforms_str})"


@dataclass(frozen=True)
class MacroTransformation(TransformationSpec):
    """
    Macro-transformation : encapsulation d'une composition fréquente comme nouvelle opération.
    
    Permet de créer des opérations réutilisables à partir de compositions de transformations
    primitives.
    
    Exemple : place_on_table = MacroTransformation("place_on_table", base_transformation=composed)
    
    Note : Cette classe est un wrapper simple autour d'une TransformationSpec,
    pas une sous-classe.
    """
    
    name: str = "macro"
    base_transformation: TransformationSpec = field(default_factory=IdentityTransformation)
    steps: Tuple[TransformationSpec, ...] = ()
    description: str = ""
    domain: TransformationDomain = TransformationDomain.GENERIC

    def __post_init__(self):
        base = (CompositeTransformation(name=f"{self.name}.steps", transformations=self.steps,
                                        domain=self.domain)
                if self.steps else self.base_transformation)
        object.__setattr__(self, 'base_transformation', base)
        object.__setattr__(self, 'parameters', dict(base.parameters))
        object.__setattr__(self, 'preserves', base.preserves)
        object.__setattr__(self, 'invertible', base.invertible)
        object.__setattr__(self, 'cost', base.cost)
        object.__setattr__(self, 'changes', set(base.changes))
    
    def apply(self, obj: Any, *args, **kwargs) -> Any:
        """Applique la transformation de base."""
        return self.base_transformation.apply(obj, *args, **kwargs)
    
    def get_inverse(self) -> Optional['MacroTransformation']:
        """Retourne l'inverse si la transformation de base est inversible."""
        inv = self.base_transformation.get_inverse()
        if inv is None:
            return None
        return MacroTransformation(
            name=f"inverse({self.name})",
            base_transformation=inv,
            description=f"Inverse de {self.description}"
        )
    
    def compile(self, world):
        compiled = self.base_transformation.compile(world)
        from core.transformation import TransformationSource
        return compiled.with_spec(self, TransformationSource.MACRO)
    
    def unwrap(self) -> TransformationSpec:
        """Retourne la transformation de base sous-jacente."""
        if isinstance(self.base_transformation, MacroTransformation):
            return self.base_transformation.unwrap()
        return self.base_transformation
    
    def __repr__(self) -> str:
        return f"MacroTransformation(name='{self.name}', base={self.base_transformation.name})"


@dataclass(frozen=True)
class DeclarativeTransformationSpec(TransformationSpec):
    """Spec multi-domaine dont le compilateur est fourni par son producteur."""
    compiler: Optional[Callable[[Any], Any]] = field(default=None, repr=False, compare=False)

    def apply(self, obj: Any, *args, **kwargs) -> Any:
        return obj

    def compile(self, world):
        if self.compiler is None:
            return super().compile(world)
        executable = self.compiler(world)
        if executable is None:
            return None
        return executable.with_spec(self)


@dataclass(frozen=True)
class CutTransformation(TransformationSpec):
    """
    Transformation de découpe - peut modifier la topologie.
    
    Cette transformation annonce qu'elle peut changer la topologie en créant
    de nouvelles composantes connexes ou de nouveaux trous.
    
    Note : L'implémentation concrète de la découpe n'est pas encore complète,
    mais cette classe permet de représenter conceptuellement l'opération.
    """
    name: str = "cut"
    domain: TransformationDomain = TransformationDomain.TOPOLOGY
    invertible: bool = False
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet({
        Invariant.DIMENSION,
        Invariant.MASS,
        Invariant.ENTITY_IDENTITY,
    }))
    changes: Set[str] = field(default_factory=lambda: {
        "topology",
        "connected_components",
        "boundary",
    })
    cost: float = 10.0  # Coût élevé car change la topologie
    
    def apply(self, topology: TopologySignature, **kwargs) -> TopologySignature:
        """
        Applique conceptuellement une découpe à une signature topologique.
        
        Augmente le nombre de composantes connexes et potentiellement de trous.
        """
        # Cette implémentation est conceptuelle
        # En réalité, la découpe dépend de la géométrie spécifique
        new_components = topology.connected_components + 1
        return TopologySignature(
            connected_components=new_components,
            holes=topology.holes,
            boundary_components=topology.boundary_components + 1
        )


@dataclass(frozen=True)
class MergeTransformation(TransformationSpec):
    """
    Transformation de fusion - peut modifier la topologie.
    
    Fusionne des composantes connexes.
    """
    name: str = "merge"
    domain: TransformationDomain = TransformationDomain.TOPOLOGY
    invertible: bool = False
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet({
        Invariant.DIMENSION,
        Invariant.ENTITY_IDENTITY,
    }))
    changes: Set[str] = field(default_factory=lambda: {
        "topology",
        "connected_components",
        "boundary",
    })
    cost: float = 10.0
    
    def apply(self, topology: TopologySignature, **kwargs) -> TopologySignature:
        """
        Applique conceptuellement une fusion à une signature topologique.
        
        Réduit le nombre de composantes connexes.
        """
        new_components = max(1, topology.connected_components - 1)
        return TopologySignature(
            connected_components=new_components,
            holes=topology.holes,
            boundary_components=max(1, topology.boundary_components - 1)
        )


# Fonctions utilitaires pour créer des transformations courantes

def create_translation_transformation(tx: float, ty: float) -> TransformationSpec:
    """Crée une TransformationSpec pour une translation 2D."""
    from core.transform import Transform2D
    from core.invariants import TRANSLATION_INVARIANTS
    
    # Créer un wrapper qui utilise Transform2D
    class TranslationTransformation(TransformationSpec):
        def __init__(self, tx: float, ty: float):
            object.__setattr__(self, 'name', f'translate({tx}, {ty})')
            object.__setattr__(self, 'domain', TransformationDomain.GEOMETRY)
            object.__setattr__(self, 'parameters', {'tx': tx, 'ty': ty})
            object.__setattr__(self, 'invertible', True)
            object.__setattr__(self, 'preserves', TRANSLATION_INVARIANTS)
            object.__setattr__(self, 'changes', {'position'})
            object.__setattr__(self, 'cost', 0.5)
        
        def apply(self, obj, *args, **kwargs):
            if isinstance(obj, tuple) and len(obj) == 2:
                return (obj[0] + self.parameters['tx'], obj[1] + self.parameters['ty'])
            elif hasattr(obj, 'x') and hasattr(obj, 'y'):
                return type(obj)(
                    x=obj.x + self.parameters['tx'],
                    y=obj.y + self.parameters['ty']
                )
            return obj
        
        def get_inverse(self):
            return TranslationTransformation(-self.parameters['tx'], -self.parameters['ty'])

        def compile(self, world):
            return _compile_geometry_spec(self, world)
    
    return TranslationTransformation(tx, ty)


def create_rotation_transformation(theta: float, name: str = None) -> TransformationSpec:
    """Crée une TransformationSpec pour une rotation 2D."""
    from core.invariants import ROTATION_INVARIANTS
    from math import cos, sin
    
    class RotationTransformation(TransformationSpec):
        def __init__(self, theta: float, name: str = None):
            object.__setattr__(self, 'name', name or f'rotate({theta})')
            object.__setattr__(self, 'domain', TransformationDomain.GEOMETRY)
            object.__setattr__(self, 'parameters', {'theta': theta})
            object.__setattr__(self, 'invertible', True)
            object.__setattr__(self, 'preserves', ROTATION_INVARIANTS)
            object.__setattr__(self, 'changes', {'orientation'})
            object.__setattr__(self, 'cost', 1.0)
        
        def apply(self, obj, *args, **kwargs):
            if isinstance(obj, tuple) and len(obj) == 2:
                x, y = obj
                c = cos(self.parameters['theta'])
                s = sin(self.parameters['theta'])
                return (c * x - s * y, s * x + c * y)
            return obj
        
        def get_inverse(self):
            return RotationTransformation(-self.parameters['theta'])

        def compile(self, world):
            return _compile_geometry_spec(self, world)
    
    return RotationTransformation(theta, name)


def create_scale_transformation(sx: float, sy: float) -> TransformationSpec:
    """Crée une TransformationSpec pour un scale 2D."""
    from core.invariants import UNIFORM_SCALE_INVARIANTS, NON_UNIFORM_SCALE_INVARIANTS
    
    class ScaleTransformation(TransformationSpec):
        def __init__(self, sx: float, sy: float):
            object.__setattr__(self, 'name', f'scale({sx}, {sy})')
            object.__setattr__(self, 'domain', TransformationDomain.GEOMETRY)
            object.__setattr__(self, 'parameters', {'sx': sx, 'sy': sy})
            object.__setattr__(self, 'invertible', sx != 0 and sy != 0)
            
            # Déterminer les invariants en fonction du type de scale
            if sx == sy:
                invariants = UNIFORM_SCALE_INVARIANTS
            else:
                invariants = NON_UNIFORM_SCALE_INVARIANTS
            object.__setattr__(self, 'preserves', invariants)
            
            object.__setattr__(self, 'changes', {'size', 'area'})
            object.__setattr__(self, 'cost', 1.0)
        
        def apply(self, obj, *args, **kwargs):
            if isinstance(obj, tuple) and len(obj) == 2:
                return (obj[0] * self.parameters['sx'], obj[1] * self.parameters['sy'])
            return obj
        
        def get_inverse(self):
            if not self.invertible:
                return None
            return ScaleTransformation(1.0 / self.parameters['sx'], 1.0 / self.parameters['sy'])

        def compile(self, world):
            return _compile_geometry_spec(self, world)
    
    return ScaleTransformation(sx, sy)


def _compile_geometry_spec(spec: TransformationSpec, world, entity_name: str = None):
    """Compile une primitive ponctuelle sur les entités possédant x et y."""
    from core.transformation import Change, Transformation, TransformationSource
    changes = []
    entities = ([world.entities[entity_name]] if entity_name else world.entities.values())
    for entity in entities:
        if "x" not in entity.properties or "y" not in entity.properties:
            continue
        before = (float(entity.properties["x"]), float(entity.properties["y"]))
        after = spec.apply(before)
        if after[0] != before[0]:
            changes.append(Change(entity.name, "x", entity.properties["x"], after[0]))
        if after[1] != before[1]:
            changes.append(Change(entity.name, "y", entity.properties["y"], after[1]))
    return Transformation(spec.name, changes=changes, spec=spec,
                          source=TransformationSource.USER,
                          domain=spec.domain.value, invariants=spec.preserves)


@dataclass(frozen=True)
class ReflectionTransformation(TransformationSpec):
    """Réflexion primitive autour de l'axe x, y ou d'une droite par l'origine."""
    name: str = "reflection"
    axis: str = "x"
    domain: TransformationDomain = TransformationDomain.GEOMETRY
    preserves: InvariantSet = field(default_factory=lambda: __import__(
        'core.invariants', fromlist=['REFLECTION_INVARIANTS']).REFLECTION_INVARIANTS)
    changes: Set[str] = field(default_factory=lambda: {"position", "orientation"})

    def apply(self, obj, *args, **kwargs):
        x, y = obj
        if self.axis == "x": return (x, -y)
        if self.axis == "y": return (-x, y)
        raise ValueError("axis doit être 'x' ou 'y'")

    def get_inverse(self): return self
    def compile(self, world): return _compile_geometry_spec(self, world)


@dataclass(frozen=True)
class ShearTransformation(TransformationSpec):
    """Cisaillement affine primitif x'=x+kx*y, y'=y+ky*x."""
    name: str = "shear"
    shear_x: float = 0.0
    shear_y: float = 0.0
    domain: TransformationDomain = TransformationDomain.GEOMETRY
    preserves: InvariantSet = field(default_factory=lambda: __import__(
        'core.invariants', fromlist=['SHEAR_INVARIANTS']).SHEAR_INVARIANTS)
    changes: Set[str] = field(default_factory=lambda: {"position", "shape"})

    def __post_init__(self):
        determinant = 1.0 - self.shear_x * self.shear_y
        object.__setattr__(self, 'invertible', determinant != 0.0)
        object.__setattr__(self, 'parameters', {"shear_x": self.shear_x, "shear_y": self.shear_y})

    def apply(self, obj, *args, **kwargs):
        x, y = obj
        return (x + self.shear_x * y, y + self.shear_y * x)

    def get_inverse(self):
        determinant = 1.0 - self.shear_x * self.shear_y
        if determinant == 0.0: return None
        # inverse matrix is not generally a unit-diagonal shear; use exact spec
        return MatrixTransformation(matrix=((1 / determinant, -self.shear_x / determinant),
                                             (-self.shear_y / determinant, 1 / determinant)),
                                    name=f"inverse({self.name})", preserves=self.preserves)

    def compile(self, world): return _compile_geometry_spec(self, world)


@dataclass(frozen=True)
class MatrixTransformation(TransformationSpec):
    matrix: Tuple[Tuple[float, float], Tuple[float, float]] = ((1., 0.), (0., 1.))
    name: str = "matrix"
    domain: TransformationDomain = TransformationDomain.GEOMETRY

    def apply(self, obj, *args, **kwargs):
        x, y = obj; m = self.matrix
        return (m[0][0]*x + m[0][1]*y, m[1][0]*x + m[1][1]*y)

    def compile(self, world): return _compile_geometry_spec(self, world)


def create_reflection_transformation(axis: str = "x") -> TransformationSpec:
    return ReflectionTransformation(name=f"reflect({axis})", axis=axis)


def create_shear_transformation(shear_x: float = 0.0, shear_y: float = 0.0) -> TransformationSpec:
    return ShearTransformation(name=f"shear({shear_x}, {shear_y})", shear_x=shear_x, shear_y=shear_y)

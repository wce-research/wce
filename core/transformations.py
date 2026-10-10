"""
WCE Transformation Algebra.

This module provides the fundamental abstraction of transformations as a first-class
concept in WCE.

A transformation is an operation that can be applied to a configuration
to produce a new configuration, with explicit invariants and
changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Set, List, Tuple
from enum import Enum

from core.invariants import Invariant, InvariantSet, IDENTITY_INVARIANTS
from core.topology import TopologySignature, DISK_TOPOLOGY


class TransformationDomain(Enum):
    """Domain of application of a transformation."""
    GEOMETRY = "geometry"           # Geometric transformations (2D/3D)
    TOPOLOGY = "topology"           # Topological transformations
    PHYSICS = "physics"             # Physical transformations
    WORLD = "world"                 # Transformations of the world (state)
    TIME = "time"                   # Temporal transformations
    LANGUAGE = "language"            # Linguistic transformations
    ALGEBRA = "algebra"              # Symbolic transformations
    SEQUENCE = "sequence"            # Transformations of symbolic sequences
    MODEL = "model"                  # Self-transformations of the neural model
    MEASUREMENT = "measurement"      # Measurement, change of basis / of reference frame, conditioning (measurement algebra)
    GENERIC = "generic"             # Generic/undefined domain


@dataclass(frozen=True)
class TransformationSpec(ABC):
    """
    Abstract specification of a WCE transformation.
    
    A TransformationSpec describes a transformation conceptually,
    independently of its concrete implementation.
    
    Attributes:
        name: Descriptive name of the transformation
        domain: Domain of application (geometry, physics, etc.)
        parameters: Dictionary of the parameters of the transformation
        invertible: Indicates whether the transformation has an inverse
        preserves: Set of invariants preserved by this transformation
        changes: Set of properties that change (optional)
        cost: Estimated cost of the application (for optimisation)
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
        Applies the transformation to a given object.
        
        The implementation depends on the domain and on the type of transformation.
        """
        pass
    
    def is_invertible(self) -> bool:
        """Indicates whether this transformation has an inverse."""
        return self.invertible
    
    def preserves_invariant(self, invariant: Invariant) -> bool:
        """Checks whether a specific invariant is preserved."""
        return self.preserves.preserves(invariant)
    
    def preserves_geometry(self) -> bool:
        """Indicates whether the geometry is preserved (isometry)."""
        return self.preserves.preserves_geometry
    
    def preserves_topology(self) -> bool:
        """Indicates whether the topology is preserved."""
        return self.preserves.preserves_topology
    
    def get_inverse(self) -> Optional['TransformationSpec']:
        """
        Returns the inverse transformation if it exists.
        
        Must be implemented by the subclasses to provide the inverse.
        """
        if not self.invertible:
            return None
        # By default, returns None - the subclasses must implement
        return None

    def compile(self, world) -> 'Transformation':
        """Compiles the description into an atomic mutation of the World."""
        raise TypeError(f"{type(self).__name__} cannot compile a World")

    def applicability(self, context):
        """Returns TRUE/FALSE/UNKNOWN for the declared preconditions."""
        from core.reasoning import TruthValue
        values = [condition.evaluate(context) for condition in self.preconditions]
        if any(value is TruthValue.FALSE for value in values): return TruthValue.FALSE
        if any(value is TruthValue.UNKNOWN for value in values): return TruthValue.UNKNOWN
        return TruthValue.TRUE
    
    def compose(self, other: 'TransformationSpec') -> 'CompositeTransformation':
        """
        Creates a composition of this transformation with another.
        
        The result is a CompositeTransformation that applies the two
        transformations in order: self then other (self is applied first).
        """
        return CompositeTransformation(transformations=(self, other))
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', domain={self.domain.value})"
    
    def __matmul__(self, other: 'TransformationSpec') -> 'CompositeTransformation':
        """Composition operator: T1 @ T2 = T2 ∘ T1 (T1 applied first)."""
        return self.compose(other)


@dataclass(frozen=True)
class IdentityTransformation(TransformationSpec):
    """
    Identity transformation (changes nothing).
    
    Preserves all invariants.
    """
    name: str = "identity"
    domain: TransformationDomain = TransformationDomain.GENERIC
    preserves: InvariantSet = field(default_factory=lambda: IDENTITY_INVARIANTS)
    
    def apply(self, *args, **kwargs) -> Any:
        """Does not change the input."""
        if len(args) == 1:
            return args[0]
        return args
    
    def get_inverse(self) -> 'IdentityTransformation':
        """The inverse of the identity is the identity."""
        return IdentityTransformation()


@dataclass(frozen=True)
class CompositeTransformation(TransformationSpec):
    """
    Composition of several transformations.
    
    Applies the transformations in the order of the list:
    [T1, T2, T3] means T3 ∘ T2 ∘ T1 (T1 is applied first).
    
    The preserved invariants are the intersection of the invariants of all
    the composed transformations.
    """
    name: str = "composite"
    transformations: Tuple[TransformationSpec, ...] = ()
    domain: TransformationDomain = TransformationDomain.GENERIC
    
    def __post_init__(self):
        # Compute the aggregated parameters
        params = {}
        for i, t in enumerate(self.transformations):
            params[f'transform_{i}'] = t.name
        object.__setattr__(self, 'parameters', params)
        
        # Compute the preserved invariants (intersection)
        if self.transformations:
            preserved = self.transformations[0].preserves
            for t in self.transformations[1:]:
                preserved = preserved.intersection(t.preserves)
            object.__setattr__(self, 'preserves', preserved)
        else:
            object.__setattr__(self, 'preserves', InvariantSet(set()))
        
        # Invertible if all the components are
        invertible = all(t.invertible for t in self.transformations)
        object.__setattr__(self, 'invertible', invertible)
        
        # Cumulative cost
        cost = sum(t.cost for t in self.transformations)
        object.__setattr__(self, 'cost', cost)
        
        # Aggregated changes
        changes = set()
        for t in self.transformations:
            changes.update(t.changes)
        object.__setattr__(self, 'changes', changes)
    
    def apply(self, obj: Any, *args, **kwargs) -> Any:
        """Applies all the transformations in order."""
        result = obj
        for t in self.transformations:
            result = t.apply(result, *args, **kwargs)
        return result
    
    def get_inverse(self) -> Optional['CompositeTransformation']:
        """
        Returns the inverse composition (reversed order, each transformation inverted).
        """
        if not self.invertible:
            return None
        
        # Create the list of inverses in reverse order
        inverse_transforms = []
        for t in reversed(self.transformations):
            inv = t.get_inverse()
            if inv is None:
                return None  # Should not happen if invertible=True
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
        """Returns the list of the composed transformations."""
        return list(self.transformations)
    
    def get_depth(self) -> int:
        """Returns the number of transformations in the composition."""
        return len(self.transformations)
    
    def __repr__(self) -> str:
        transforms_str = " @ ".join([t.name for t in self.transformations])
        return f"CompositeTransformation({transforms_str})"


@dataclass(frozen=True)
class MacroTransformation(TransformationSpec):
    """
    Macro-transformation: encapsulation of a frequent composition as a new operation.
    
    Makes it possible to create reusable operations from compositions of primitive
    transformations.
    
    Example: place_on_table = MacroTransformation("place_on_table", base_transformation=composed)
    
    Note: This class is a simple wrapper around a TransformationSpec,
    not a subclass.
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
        """Applies the base transformation."""
        return self.base_transformation.apply(obj, *args, **kwargs)
    
    def get_inverse(self) -> Optional['MacroTransformation']:
        """Returns the inverse if the base transformation is invertible."""
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
        """Returns the underlying base transformation."""
        if isinstance(self.base_transformation, MacroTransformation):
            return self.base_transformation.unwrap()
        return self.base_transformation
    
    def __repr__(self) -> str:
        return f"MacroTransformation(name='{self.name}', base={self.base_transformation.name})"


@dataclass(frozen=True)
class DeclarativeTransformationSpec(TransformationSpec):
    """Multi-domain spec whose compiler is provided by its producer."""
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
    Cut transformation - may modify the topology.
    
    This transformation announces that it may change the topology by creating
    new connected components or new holes.
    
    Note: The concrete implementation of the cut is not yet complete,
    but this class makes it possible to represent the operation conceptually.
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
    cost: float = 10.0  # High cost because it changes the topology
    
    def apply(self, topology: TopologySignature, **kwargs) -> TopologySignature:
        """
        Conceptually applies a cut to a topological signature.
        
        Increases the number of connected components and potentially of holes.
        """
        # This implementation is conceptual
        # In reality, the cut depends on the specific geometry
        new_components = topology.connected_components + 1
        return TopologySignature(
            connected_components=new_components,
            holes=topology.holes,
            boundary_components=topology.boundary_components + 1
        )


@dataclass(frozen=True)
class MergeTransformation(TransformationSpec):
    """
    Merge transformation - may modify the topology.
    
    Merges connected components.
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
        Conceptually applies a merge to a topological signature.
        
        Reduces the number of connected components.
        """
        new_components = max(1, topology.connected_components - 1)
        return TopologySignature(
            connected_components=new_components,
            holes=topology.holes,
            boundary_components=max(1, topology.boundary_components - 1)
        )


# Utility functions to create common transformations

def create_translation_transformation(tx: float, ty: float) -> TransformationSpec:
    """Creates a TransformationSpec for a 2D translation."""
    from core.transform import Transform2D
    from core.invariants import TRANSLATION_INVARIANTS
    
    # Create a wrapper that uses Transform2D
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
    """Creates a TransformationSpec for a 2D rotation."""
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
    """Creates a TransformationSpec for a 2D scale."""
    from core.invariants import UNIFORM_SCALE_INVARIANTS, NON_UNIFORM_SCALE_INVARIANTS
    
    class ScaleTransformation(TransformationSpec):
        def __init__(self, sx: float, sy: float):
            object.__setattr__(self, 'name', f'scale({sx}, {sy})')
            object.__setattr__(self, 'domain', TransformationDomain.GEOMETRY)
            object.__setattr__(self, 'parameters', {'sx': sx, 'sy': sy})
            object.__setattr__(self, 'invertible', sx != 0 and sy != 0)
            
            # Determine the invariants according to the type of scale
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
    """Compiles a pointwise primitive on the entities that have x and y."""
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
    """Primitive reflection about the x axis, the y axis or a line through the origin."""
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
        raise ValueError("axis must be 'x' or 'y'")

    def get_inverse(self): return self
    def compile(self, world): return _compile_geometry_spec(self, world)


@dataclass(frozen=True)
class ShearTransformation(TransformationSpec):
    """Primitive affine shear x'=x+kx*y, y'=y+ky*x."""
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

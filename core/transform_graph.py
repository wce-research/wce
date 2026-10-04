"""Graphe causal léger des transformations du World."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.invariants import IDENTITY_INVARIANTS, InvariantSet
from core.topology import TopologySignature
from core.transformations import CompositeTransformation, IdentityTransformation, TransformationSpec


@dataclass(frozen=True)
class TransformRecord:
    spec: TransformationSpec
    executable: Any = None
    source_type: str = "system"
    source_name: str = "system"
    domain: str = "world"
    time: float = 0.0
    before: Any = None
    after: Any = None
    changed_properties: tuple[str, ...] = ()


@dataclass
class TransformNode:
    name: str
    configuration: Any = None
    transformations: List[TransformationSpec] = field(default_factory=list)
    parent: Optional['TransformNode'] = None
    children: List['TransformNode'] = field(default_factory=list)
    invariants: InvariantSet = field(default_factory=lambda: IDENTITY_INVARIANTS)
    topology: Optional[TopologySignature] = None
    depth: int = 0
    record: Optional[TransformRecord] = None
    facts: Any = None
    assumptions: Any = None

    def add_child(self, transformation, child_config=None, topology=None, record=None, name=None):
        child = TransformNode(
            name=name or f"{self.name} -> {transformation.name}",
            configuration=child_config,
            transformations=self.transformations + [transformation],
            parent=self,
            invariants=self.invariants.intersection(transformation.preserves),
            topology=topology,
            depth=self.depth + 1,
            record=record,
            facts=self.facts.copy() if hasattr(self.facts, "copy") else self.facts,
            assumptions=self.assumptions.copy() if hasattr(self.assumptions, "copy") else self.assumptions,
        )
        self.children.append(child)
        return child

    def get_path(self): return list(self.transformations)
    def get_path_names(self): return [t.name for t in self.transformations]
    def get_preserved_invariants(self): return self.invariants
    def get_lost_invariants(self, original): return self.invariants.lost(original)
    def get_total_cost(self): return sum(t.cost for t in self.transformations)
    def is_root(self): return self.parent is None
    def is_leaf(self): return not self.children
    def can_invert(self): return all(t.invertible for t in self.transformations)
    def get_inverse_path(self):
        if not self.can_invert(): raise ValueError("Cannot invert non-invertible path")
        return [t.get_inverse() for t in reversed(self.transformations)]
    def get_composite_transformation(self):
        if not self.transformations: return IdentityTransformation()
        if len(self.transformations) == 1: return self.transformations[0]
        return CompositeTransformation(name=f"composite({self.name})", transformations=tuple(self.transformations))


@dataclass
class TransformGraph:
    root: TransformNode
    current: TransformNode = None
    nodes: Dict[str, TransformNode] = field(default_factory=dict)
    _sequence: int = 0

    def __post_init__(self):
        self.current = self.current or self.root
        self._index_nodes(self.root)

    def _index_nodes(self, node):
        self.nodes[node.name] = node
        for child in node.children: self._index_nodes(child)

    def apply_transformation(self, transformation, config=None, topology=None, *, record=None, parent=None):
        parent = parent or self.current
        self._sequence += 1
        node = parent.add_child(transformation, config, topology, record, name=f"C{self._sequence}")
        self.nodes[node.name] = node
        self.current = node
        return node

    def branch_from(self, node):
        if isinstance(node, str): node = self.nodes[node]
        self.current = node
        return node

    def get_current_path(self): return self.current.get_path()
    def get_current_invariants(self): return self.current.invariants
    def can_undo(self): return not self.current.is_root()
    def can_redo(self): return bool(self.current.children)
    def undo(self):
        if not self.can_undo(): raise ValueError("Cannot undo at root node")
        self.current = self.current.parent
        return self.current
    def reset_to_root(self): self.current = self.root
    def get_node_by_name(self, name): return self.nodes.get(name)
    def get_all_paths(self):
        paths = []
        def visit(node):
            if node.is_leaf(): paths.append(node.get_path())
            for child in node.children: visit(child)
        visit(self.root)
        return paths

    def explain_last(self):
        record = self.current.record
        if record is None: return "Aucune transformation appliquée."
        lines = []
        for change in getattr(record.executable, "changes", ()):
            lines.append(f"{change.entity}.{change.property}: {change.before} -> {change.after}")
        lines += [f"Cause: {record.source_name}", f"Transformation: {record.spec.name}",
                  f"Domain: {record.domain}", f"Time: {record.time}s"]
        return "\n".join(lines)

    def get_history_summary(self):
        return {'depth': self.current.depth, 'total_nodes': len(self.nodes),
                'total_transformations': len(self.current.transformations),
                'preserved_invariants': len(self.current.invariants.invariants),
                'preserved_invariant_names': sorted(i.name for i in self.current.invariants.invariants),
                'can_undo': self.can_undo()}


def create_linear_graph(name, transformations, initial_config=None, initial_topology=None):
    graph = TransformGraph(TransformNode(name, initial_config, topology=initial_topology))
    for transformation in transformations: graph.apply_transformation(transformation)
    return graph


def create_world_transform_graph(initial_world_state="initial"):
    return TransformGraph(TransformNode(initial_world_state))

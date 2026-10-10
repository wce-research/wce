"""Compact neural model as a WCE state: modules, slow/fast weights, connectivity.

Every structural or parametric modification goes through a ModelTransformationSpec,
applied via ModelState.apply() and traced in the TransformGraph.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from core.invariants import Invariant, InvariantSet
from core.reasoning import Condition, TruthValue
from core.transform_graph import TransformGraph, TransformNode, TransformRecord
from core.transformations import TransformationDomain, TransformationSpec

INPUT = "input"
PARAM_NAMES = ("w1", "b1", "w2", "b2")


# ---------------------------------------------------------------- modules

class NeuralModule:
    """Two-layer block: out = W2·relu(W1·x + b1) + b2 (+ relu if hidden)."""

    def __init__(self, name, input_dim, output_dim, hidden=16, *, specialization="generic",
                 activation=True, active=True, seed=0):
        gen = torch.Generator().manual_seed(seed)
        self.name, self.input_dim, self.output_dim = name, input_dim, output_dim
        self.specialization, self.activation, self.active = specialization, activation, active
        self.usage_count = 0
        scale1, scale2 = (2.0 / input_dim) ** 0.5, (1.0 / hidden) ** 0.5
        self.slow = {
            "w1": torch.randn(hidden, input_dim, generator=gen) * scale1,
            "b1": torch.zeros(hidden),
            "w2": torch.randn(output_dim, hidden, generator=gen) * scale2,
            "b2": torch.zeros(output_dim),
        }

    @property
    def hidden(self): return self.slow["w1"].shape[0]
    @property
    def parameter_count(self): return sum(t.numel() for t in self.slow.values())

    def forward(self, x, fast=None, override=None):
        params = override or {k: self.slow[k] + (fast[k] if fast and k in fast else 0) for k in PARAM_NAMES}
        h = torch.relu(x @ params["w1"].T + params["b1"])
        out = h @ params["w2"].T + params["b2"]
        return torch.relu(out) if self.activation else out

    def __repr__(self):
        return f"NeuralModule({self.name}, {self.input_dim}->{self.hidden}->{self.output_dim}, active={self.active})"


@dataclass
class Connection:
    source: str
    target: str
    strength: float = 1.0


@dataclass(frozen=True)
class ModelRelation:
    """Relational view of the model, read by RuleEngine.refresh_facts()."""
    subject: str
    predicate: str
    object: str


@dataclass
class ModelSnapshot:
    name: str
    modules: Dict[str, NeuralModule]
    connectivity: Dict[Tuple[str, str], Connection]
    fast_parameters: Dict[str, Dict[str, torch.Tensor]]
    node_name: str


@dataclass(frozen=True)
class PlasticityRegion:
    """Explicit subset of modules allowed to change."""
    name: str
    modules: Tuple[str, ...]

    def parameter_count(self, model):
        return sum(model.modules[m].parameter_count for m in self.modules if m in model.modules)


# ---------------------------------------------------------------- state

class ModelState:
    """The model itself as a WCE state."""

    def __init__(self, input_dim, output_module="head", *, max_connections=32,
                 max_capacity=512, max_region_fraction=0.6):
        self.input_dim = input_dim
        self.modules: Dict[str, NeuralModule] = {}
        self.fast_parameters: Dict[str, Dict[str, torch.Tensor]] = {}
        self.connectivity: Dict[Tuple[str, str], Connection] = {}
        self.snapshots: Dict[str, ModelSnapshot] = {}
        self.metadata = {"output": output_module, "max_connections": max_connections,
                         "max_capacity": max_capacity, "max_region_fraction": max_region_fraction}
        self.counters = {"modified_parameters": 0, "adaptation_steps": 0, "rollbacks": 0,
                         "growths": 0, "prunes": 0, "transformations": 0}
        self.transform_graph = TransformGraph(TransformNode("N0", configuration=self.summary()))

    # ---- views -------------------------------------------------------
    @property
    def parameters(self): return {n: m.slow for n, m in self.modules.items()}
    @property
    def active_modules(self): return {n for n, m in self.modules.items() if m.active}
    @property
    def capacity(self): return sum(m.hidden for m in self.modules.values())
    @property
    def active_capacity(self): return sum(self.modules[m].hidden for m in self.active_modules)
    @property
    def relations(self):
        rels = [ModelRelation(c.source, "connects", c.target) for c in self.connectivity.values()]
        rels += [ModelRelation(n, "is", "active" if m.active else "inactive") for n, m in self.modules.items()]
        return rels

    def dim_out(self, name): return self.input_dim if name == INPUT else self.modules[name].output_dim
    def total_parameters(self): return sum(m.parameter_count for m in self.modules.values())
    def active_parameters(self):
        return sum(self.modules[m].parameter_count for m in self.active_modules if m in self._reachable())
    def fast_parameter_count(self):
        return sum(t.numel() for d in self.fast_parameters.values() for t in d.values())

    def summary(self):
        return {"modules": sorted(self.modules), "active": sorted(self.active_modules),
                "connections": sorted(self.connectivity), "capacity": self.capacity}

    def metrics(self):
        return {"total_parameters": self.total_parameters(), "active_parameters": self.active_parameters(),
                "modified_parameters": self.counters["modified_parameters"],
                "fast_parameters": self.fast_parameter_count(),
                "modules_active": len(self.active_modules),
                "modules_inactive": len(self.modules) - len(self.active_modules),
                "connections": len(self.connectivity), "capacity": self.capacity,
                "inference_cost": self.inference_cost(),
                "adaptation_cost": self.counters["adaptation_steps"],
                "rollback_count": self.counters["rollbacks"], "growth_count": self.counters["growths"],
                "prune_count": self.counters["prunes"]}

    def inference_cost(self):
        """Approximate number of multiplications per example (reached modules only)."""
        return sum(self.modules[m].slow["w1"].numel() + self.modules[m].slow["w2"].numel()
                   for m in self._reachable())

    # ---- computation graph ------------------------------------------
    def _incoming(self, name):
        return [c for c in self.connectivity.values() if c.target == name
                and (c.source == INPUT or self.modules[c.source].active)]

    def _reachable(self):
        seen, frontier = set(), [INPUT]
        while frontier:
            node = frontier.pop()
            for c in self.connectivity.values():
                if c.source == node and c.target not in seen and self.modules[c.target].active:
                    seen.add(c.target); frontier.append(c.target)
        return seen

    def has_path(self, source, target):
        frontier, seen = [source], set()
        while frontier:
            node = frontier.pop()
            if node == target: return True
            seen.add(node)
            frontier += [c.target for c in self.connectivity.values() if c.source == node and c.target not in seen]
        return False

    def _ancestors(self, target):
        seen, frontier = set(), [target]
        while frontier:
            node = frontier.pop()
            for c in self._incoming(node) if node != INPUT else ():
                if c.source not in seen: seen.add(c.source); frontier.append(c.source)
        return seen

    def topological_order(self):
        reachable = self._reachable()
        out = self.metadata.get("output")
        if out in reachable: reachable &= self._ancestors(out) | {out}
        order, placed = [], {INPUT}
        while len(order) < len(reachable):
            ready = [m for m in sorted(reachable) if m not in placed
                     and all(c.source in placed for c in self._incoming(m) if c.source == INPUT or c.source in reachable)]
            if not ready: raise ValueError("cycle in the active connectivity")
            order += ready; placed.update(ready)
        return order

    def forward(self, x, overrides=None):
        outputs = {INPUT: x}
        for name in self.topological_order():
            incoming = [c for c in self._incoming(name) if c.source in outputs]
            if not incoming: continue
            module = self.modules[name]
            inp = sum(c.strength * outputs[c.source] for c in incoming)
            outputs[name] = module.forward(inp, self.fast_parameters.get(name),
                                           (overrides or {}).get(name))
            module.usage_count += 1
        out = self.metadata["output"]
        if out not in outputs: raise ValueError(f"output module '{out}' not reached")
        return outputs[out]

    def predict(self, x):
        with torch.no_grad(): return self.forward(x).argmax(dim=1)

    def accuracy(self, x, y): return float((self.predict(x) == y).float().mean())

    # ---- fast weights API ------------------------------------------
    def create_fast_delta(self, module):
        self.fast_parameters[module] = {k: torch.zeros_like(v) for k, v in self.modules[module].slow.items()}
        return self.fast_parameters[module]

    def apply_fast_delta(self, module, delta):
        current = self.fast_parameters.setdefault(module, {k: torch.zeros_like(v) for k, v in self.modules[module].slow.items()})
        for k, v in delta.items(): current[k] = current[k] + v

    def clear_fast_delta(self, module=None):
        for name in ([module] if module else list(self.fast_parameters)): self.fast_parameters.pop(name, None)

    def commit_fast_delta(self, module):
        delta = self.fast_parameters.pop(module, None)
        if delta is None: return 0
        changed = 0
        for k, v in delta.items():
            self.modules[module].slow[k] = self.modules[module].slow[k] + v
            changed += int((v != 0).sum())
        return changed

    def effective_weight(self, module, key):
        fast = self.fast_parameters.get(module, {})
        return self.modules[module].slow[key] + fast.get(key, 0)

    # ---- snapshots -------------------------------------------------
    def snapshot(self, name):
        snap = ModelSnapshot(name, copy.deepcopy(self.modules), copy.deepcopy(self.connectivity),
                             copy.deepcopy(self.fast_parameters), self.transform_graph.current.name)
        self.snapshots[name] = snap
        return snap

    def _restore_content(self, snap):
        self.modules = copy.deepcopy(snap.modules)
        self.connectivity = copy.deepcopy(snap.connectivity)
        self.fast_parameters = copy.deepcopy(snap.fast_parameters)

    def restore(self, name): return self.apply(RestoreSnapshot(snapshot=name))

    # ---- governed application -------------------------------------
    def apply(self, spec: 'ModelTransformationSpec'):
        """Single entry point of the modifications: preconditions, execution, trace."""
        from core.reasoning import AssumptionSet, FactBase, ReasoningContext
        context = ReasoningContext(self, FactBase(), AssumptionSet())
        status = spec.applicability(context)
        if status is not TruthValue.TRUE:
            failed = [str(c) for c in spec.preconditions if c.evaluate(context) is not TruthValue.TRUE]
            raise ModelTransformationError(f"{spec.name} refused: {', '.join(failed)}")
        before = self.summary()
        effects = spec.execute(self) or {}
        self.counters["transformations"] += 1
        parent = self.transform_graph.current
        record = TransformRecord(spec, effects, "plasticity", spec.provenance, "model",
                                 before=before, after=self.summary(),
                                 changed_properties=tuple(sorted(spec.changes)))
        node = parent.add_child(spec, self.summary(), record=record,
                                name=f"N{len(self.transform_graph.nodes)}")
        self.transform_graph.nodes[node.name] = node
        self.transform_graph.current = node
        return effects

    def history(self): return [n.name for n in _path_to(self.transform_graph.current)]


def _path_to(node):
    path = []
    while node is not None: path.append(node); node = node.parent
    return list(reversed(path))


class ModelTransformationError(ValueError):
    pass


# ---------------------------------------------------------------- conditions

@dataclass(frozen=True)
class ModuleExists(Condition):
    module: str
    def evaluate(self, context): return TruthValue.TRUE if self.module in context.state.modules else TruthValue.FALSE
    def __str__(self): return f"exists({self.module})"


@dataclass(frozen=True)
class ModuleAbsent(Condition):
    module: str
    def evaluate(self, context): return TruthValue.FALSE if self.module in context.state.modules else TruthValue.TRUE
    def __str__(self): return f"absent({self.module})"


@dataclass(frozen=True)
class ModuleIsActive(Condition):
    module: str
    expected: bool = True
    def evaluate(self, context):
        m = context.state.modules.get(self.module)
        if m is None: return TruthValue.FALSE
        return TruthValue.TRUE if m.active == self.expected else TruthValue.FALSE
    def __str__(self): return f"{'active' if self.expected else 'inactive'}({self.module})"


@dataclass(frozen=True)
class NodeExists(Condition):
    node: str
    def evaluate(self, context):
        return TruthValue.TRUE if self.node == INPUT or self.node in context.state.modules else TruthValue.FALSE
    def __str__(self): return f"node({self.node})"


@dataclass(frozen=True)
class ConnectionExists(Condition):
    source: str
    target: str
    expected: bool = True
    def evaluate(self, context):
        present = (self.source, self.target) in context.state.connectivity
        return TruthValue.TRUE if present == self.expected else TruthValue.FALSE
    def __str__(self): return f"{'' if self.expected else 'no_'}connection({self.source}->{self.target})"


@dataclass(frozen=True)
class DimsCompatible(Condition):
    source: str
    target: str
    def evaluate(self, context):
        s = context.state
        if (self.source != INPUT and self.source not in s.modules) or self.target not in s.modules: return TruthValue.FALSE
        return TruthValue.TRUE if s.dim_out(self.source) == s.modules[self.target].input_dim else TruthValue.FALSE
    def __str__(self): return f"dims({self.source}->{self.target})"


@dataclass(frozen=True)
class NoCycle(Condition):
    source: str
    target: str
    def evaluate(self, context):
        return TruthValue.FALSE if context.state.has_path(self.target, self.source) else TruthValue.TRUE
    def __str__(self): return f"acyclic({self.source}->{self.target})"


@dataclass(frozen=True)
class ConnectionBudget(Condition):
    def evaluate(self, context):
        s = context.state
        return TruthValue.TRUE if len(s.connectivity) < s.metadata["max_connections"] else TruthValue.FALSE
    def __str__(self): return "connections < max_connections"


@dataclass(frozen=True)
class CapacityBudget(Condition):
    extra: int
    def evaluate(self, context):
        s = context.state
        return TruthValue.TRUE if s.capacity + self.extra <= s.metadata["max_capacity"] else TruthValue.FALSE
    def __str__(self): return f"capacity + {self.extra} <= max_capacity"


@dataclass(frozen=True)
class RegionIsLocal(Condition):
    """An adaptation can touch only a bounded fraction of the network."""
    modules: Tuple[str, ...]
    def evaluate(self, context):
        s = context.state
        if any(m not in s.modules for m in self.modules): return TruthValue.FALSE
        size = sum(s.modules[m].parameter_count for m in self.modules)
        return TruthValue.TRUE if size <= s.metadata["max_region_fraction"] * s.total_parameters() else TruthValue.FALSE
    def __str__(self): return f"local_region({', '.join(self.modules)})"


@dataclass(frozen=True)
class SnapshotExists(Condition):
    name: str
    def evaluate(self, context): return TruthValue.TRUE if self.name in context.state.snapshots else TruthValue.FALSE
    def __str__(self): return f"snapshot({self.name})"


@dataclass(frozen=True)
class SnapshotHasModule(Condition):
    name: str
    module: str
    def evaluate(self, context):
        snap = context.state.snapshots.get(self.name)
        return TruthValue.TRUE if snap is not None and self.module in snap.modules else TruthValue.FALSE
    def __str__(self): return f"{self.module} in snapshot({self.name})"


@dataclass(frozen=True)
class MinUnits(Condition):
    module: str
    keep: int
    def evaluate(self, context):
        m = context.state.modules.get(self.module)
        return TruthValue.TRUE if m is not None and 1 <= self.keep <= m.hidden else TruthValue.FALSE
    def __str__(self): return f"1 <= keep({self.keep}) <= hidden({self.module})"


# ---------------------------------------------------------------- specs

STRUCTURE = InvariantSet({Invariant.MODULE_SET, Invariant.MODEL_TOPOLOGY})
WEIGHTS = InvariantSet({Invariant.SLOW_WEIGHTS})


@dataclass(frozen=True)
class ModelTransformationSpec(TransformationSpec):
    name: str = "model_transformation"
    domain: TransformationDomain = TransformationDomain.MODEL
    provenance: str = "plasticity_controller"

    def apply(self, model, *args, **kwargs): return model.apply(self)
    def execute(self, model): raise NotImplementedError


def _spec(cls):
    return dataclass(frozen=True)(cls)


@_spec
class AddModule(ModelTransformationSpec):
    name: str = "add_module"
    module: str = ""
    input_dim: int = 0
    output_dim: int = 0
    hidden: int = 16
    specialization: str = "generic"
    activation: bool = True
    active: bool = True
    seed: int = 0
    invertible: bool = True
    changes: frozenset = frozenset({"modules"})
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet({Invariant.SLOW_WEIGHTS, Invariant.MODEL_FUNCTION}))
    def __post_init__(self):
        object.__setattr__(self, "preconditions", (ModuleAbsent(self.module), CapacityBudget(self.hidden)))
        object.__setattr__(self, "cost", 1.0 + self.hidden / 16)
    def execute(self, model):
        model.modules[self.module] = NeuralModule(self.module, self.input_dim, self.output_dim, self.hidden,
                                                  specialization=self.specialization, activation=self.activation,
                                                  active=self.active, seed=self.seed)
        model.counters["growths"] += 1
        return {"added": self.module, "parameters": model.modules[self.module].parameter_count}
    def get_inverse(self): return PruneModule(module=self.module)


@_spec
class ActivateModule(ModelTransformationSpec):
    name: str = "activate_module"
    module: str = ""
    changes: frozenset = frozenset({"active_modules"})
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet({Invariant.SLOW_WEIGHTS, Invariant.MODULE_SET, Invariant.MODEL_TOPOLOGY}))
    cost: float = 0.1
    def __post_init__(self): object.__setattr__(self, "preconditions", (ModuleIsActive(self.module, False),))
    def execute(self, model): model.modules[self.module].active = True; return {"activated": self.module}
    def get_inverse(self): return DeactivateModule(module=self.module)


@_spec
class DeactivateModule(ModelTransformationSpec):
    name: str = "deactivate_module"
    module: str = ""
    changes: frozenset = frozenset({"active_modules"})
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet({Invariant.SLOW_WEIGHTS, Invariant.MODULE_SET, Invariant.MODEL_TOPOLOGY}))
    cost: float = 0.1
    def __post_init__(self): object.__setattr__(self, "preconditions", (ModuleIsActive(self.module, True),))
    def execute(self, model): model.modules[self.module].active = False; return {"deactivated": self.module}
    def get_inverse(self): return ActivateModule(module=self.module)


@_spec
class CloneModule(ModelTransformationSpec):
    name: str = "clone_module"
    module: str = ""
    new_name: str = ""
    changes: frozenset = frozenset({"modules"})
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet({Invariant.SLOW_WEIGHTS, Invariant.MODEL_FUNCTION}))
    def __post_init__(self):
        object.__setattr__(self, "preconditions", (ModuleExists(self.module), ModuleAbsent(self.new_name)))
    def execute(self, model):
        clone = copy.deepcopy(model.modules[self.module])
        clone.name, clone.active, clone.usage_count = self.new_name, False, 0
        model.modules[self.new_name] = clone
        model.counters["growths"] += 1
        return {"cloned": self.module, "into": self.new_name}
    def get_inverse(self): return PruneModule(module=self.new_name)


@_spec
class MergeModule(ModelTransformationSpec):
    """Merges `other` into `module` (mean of the weights), redirects its connections."""
    name: str = "merge_module"
    module: str = ""
    other: str = ""
    invertible: bool = False
    changes: frozenset = frozenset({"modules", "weights", "connectivity"})
    def __post_init__(self):
        object.__setattr__(self, "preconditions", (ModuleExists(self.module), ModuleExists(self.other)))
    def execute(self, model):
        a, b = model.modules[self.module], model.modules[self.other]
        if any(a.slow[k].shape != b.slow[k].shape for k in PARAM_NAMES):
            raise ModelTransformationError("incompatible shapes for a merge")
        for k in PARAM_NAMES: a.slow[k] = (a.slow[k] + b.slow[k]) / 2
        a.usage_count += b.usage_count
        for key, c in list(model.connectivity.items()):
            if self.other in key:
                del model.connectivity[key]
                src = self.module if c.source == self.other else c.source
                dst = self.module if c.target == self.other else c.target
                if src != dst: model.connectivity.setdefault((src, dst), Connection(src, dst, c.strength))
        del model.modules[self.other]; model.fast_parameters.pop(self.other, None)
        model.counters["prunes"] += 1
        return {"merged": self.other, "into": self.module}


@_spec
class PruneModule(ModelTransformationSpec):
    name: str = "prune_module"
    module: str = ""
    invertible: bool = False
    changes: frozenset = frozenset({"modules", "connectivity"})
    def __post_init__(self):
        object.__setattr__(self, "preconditions", (ModuleExists(self.module),))
        object.__setattr__(self, "cost", 0.5)
    def execute(self, model):
        removed = model.modules.pop(self.module)
        model.fast_parameters.pop(self.module, None)
        for key in [k for k in model.connectivity if self.module in k]: del model.connectivity[key]
        model.counters["prunes"] += 1
        return {"pruned": self.module, "parameters": removed.parameter_count}


@_spec
class GrowModule(ModelTransformationSpec):
    """Adds hidden units; the new columns of W2 are zero: function preserved."""
    name: str = "grow_module"
    module: str = ""
    units: int = 8
    seed: int = 0
    changes: frozenset = frozenset({"capacity"})
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet({Invariant.MODEL_FUNCTION, Invariant.MODULE_SET, Invariant.MODEL_TOPOLOGY}))
    def __post_init__(self):
        object.__setattr__(self, "preconditions", (ModuleExists(self.module), CapacityBudget(self.units)))
        object.__setattr__(self, "cost", self.units / 8)
    def execute(self, model):
        m = model.modules[self.module]
        gen = torch.Generator().manual_seed(self.seed)
        new_w1 = torch.randn(self.units, m.input_dim, generator=gen) * (2.0 / m.input_dim) ** 0.5
        m.slow["w1"] = torch.cat([m.slow["w1"], new_w1])
        m.slow["b1"] = torch.cat([m.slow["b1"], torch.zeros(self.units)])
        m.slow["w2"] = torch.cat([m.slow["w2"], torch.zeros(m.output_dim, self.units)], dim=1)
        model.fast_parameters.pop(self.module, None)
        model.counters["growths"] += 1
        return {"grown": self.module, "units": self.units, "hidden": m.hidden}


@_spec
class PruneUnits(ModelTransformationSpec):
    """Keeps the `keep` units of strongest contribution |W1_i|·|W2_:,i|."""
    name: str = "prune_units"
    module: str = ""
    keep: int = 8
    invertible: bool = False
    changes: frozenset = frozenset({"capacity", "weights"})
    def __post_init__(self): object.__setattr__(self, "preconditions", (MinUnits(self.module, self.keep),))
    def execute(self, model):
        m = model.modules[self.module]
        before = m.hidden
        score = m.slow["w1"].norm(dim=1) * m.slow["w2"].norm(dim=0)
        idx = torch.sort(torch.topk(score, self.keep).indices).values
        m.slow["w1"], m.slow["b1"], m.slow["w2"] = m.slow["w1"][idx], m.slow["b1"][idx], m.slow["w2"][:, idx]
        model.fast_parameters.pop(self.module, None)
        model.counters["prunes"] += 1
        return {"pruned_units": before - self.keep, "hidden": m.hidden}


def _connection_preconditions(source, target):
    return (NodeExists(source), NodeExists(target), ConnectionExists(source, target, False),
            DimsCompatible(source, target), NoCycle(source, target), ConnectionBudget())


@_spec
class CreateConnection(ModelTransformationSpec):
    name: str = "create_connection"
    source: str = ""
    target: str = ""
    strength: float = 1.0
    changes: frozenset = frozenset({"connectivity"})
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet({Invariant.SLOW_WEIGHTS, Invariant.MODULE_SET}))
    cost: float = 0.2
    def __post_init__(self): object.__setattr__(self, "preconditions", _connection_preconditions(self.source, self.target))
    def execute(self, model):
        model.connectivity[(self.source, self.target)] = Connection(self.source, self.target, self.strength)
        return {"connected": (self.source, self.target)}
    def get_inverse(self): return RemoveConnection(source=self.source, target=self.target)


@_spec
class RemoveConnection(ModelTransformationSpec):
    name: str = "remove_connection"
    source: str = ""
    target: str = ""
    changes: frozenset = frozenset({"connectivity"})
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet({Invariant.SLOW_WEIGHTS, Invariant.MODULE_SET}))
    cost: float = 0.2
    def __post_init__(self): object.__setattr__(self, "preconditions", (ConnectionExists(self.source, self.target),))
    def execute(self, model):
        c = model.connectivity.pop((self.source, self.target))
        return {"disconnected": (self.source, self.target), "strength": c.strength}
    def get_inverse(self): return CreateConnection(source=self.source, target=self.target)


MAX_STRENGTH = 2.0


@_spec
class StrengthenConnection(ModelTransformationSpec):
    name: str = "strengthen_connection"
    source: str = ""
    target: str = ""
    delta: float = 0.25
    changes: frozenset = frozenset({"connectivity"})
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet({Invariant.SLOW_WEIGHTS, Invariant.MODULE_SET, Invariant.MODEL_TOPOLOGY}))
    cost: float = 0.1
    def __post_init__(self): object.__setattr__(self, "preconditions", (ConnectionExists(self.source, self.target),))
    def execute(self, model):
        c = model.connectivity[(self.source, self.target)]
        c.strength = min(MAX_STRENGTH, c.strength + self.delta)
        return {"strength": c.strength}
    def get_inverse(self): return WeakenConnection(source=self.source, target=self.target, delta=self.delta)


@_spec
class WeakenConnection(ModelTransformationSpec):
    name: str = "weaken_connection"
    source: str = ""
    target: str = ""
    delta: float = 0.25
    changes: frozenset = frozenset({"connectivity"})
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet({Invariant.SLOW_WEIGHTS, Invariant.MODULE_SET, Invariant.MODEL_TOPOLOGY}))
    cost: float = 0.1
    def __post_init__(self): object.__setattr__(self, "preconditions", (ConnectionExists(self.source, self.target),))
    def execute(self, model):
        c = model.connectivity[(self.source, self.target)]
        c.strength = max(0.0, c.strength - self.delta)
        return {"strength": c.strength}
    def get_inverse(self): return StrengthenConnection(source=self.source, target=self.target, delta=self.delta)


@_spec
class RerouteConnection(ModelTransformationSpec):
    """source->old_target becomes source->new_target (same strength)."""
    name: str = "reroute_connection"
    source: str = ""
    old_target: str = ""
    new_target: str = ""
    changes: frozenset = frozenset({"connectivity"})
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet({Invariant.SLOW_WEIGHTS, Invariant.MODULE_SET}))
    cost: float = 0.3
    def __post_init__(self):
        object.__setattr__(self, "preconditions", (ConnectionExists(self.source, self.old_target),
                                                   NodeExists(self.new_target),
                                                   ConnectionExists(self.source, self.new_target, False),
                                                   DimsCompatible(self.source, self.new_target),
                                                   NoCycle(self.source, self.new_target)))
    def execute(self, model):
        c = model.connectivity.pop((self.source, self.old_target))
        model.connectivity[(self.source, self.new_target)] = Connection(self.source, self.new_target, c.strength)
        return {"rerouted": (self.source, self.old_target, self.new_target)}
    def get_inverse(self): return RerouteConnection(source=self.source, old_target=self.new_target, new_target=self.old_target)


@_spec
class AdaptWeights(ModelTransformationSpec):
    """Local learning: only the modules of the region receive gradients.

    fast=True: trains fast deltas (slow weights intact).
    fast=False: trains the slow weights of the region directly.
    """
    name: str = "adapt_weights"
    modules: Tuple[str, ...] = ()
    inputs: Any = field(default=None, compare=False, repr=False)
    targets: Any = field(default=None, compare=False, repr=False)
    steps: int = 300
    lr: float = 0.03
    fast: bool = True
    local: bool = True
    seed: int = 0
    keys: Tuple[str, ...] = PARAM_NAMES
    output: str = ""
    positions: int = 0          # >0: output = positions x vocabulary, loss per position
    invertible: bool = False
    changes: frozenset = frozenset({"weights"})
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet({Invariant.MODULE_SET, Invariant.MODEL_TOPOLOGY}))
    def __post_init__(self):
        object.__setattr__(self, "preconditions", tuple(ModuleExists(m) for m in self.modules)
                           + ((RegionIsLocal(tuple(self.modules)),) if self.local else ()))
        object.__setattr__(self, "cost", self.steps / 100)
        if self.fast:
            object.__setattr__(self, "preserves", InvariantSet({Invariant.MODULE_SET, Invariant.MODEL_TOPOLOGY, Invariant.SLOW_WEIGHTS}))

    @property
    def region(self): return PlasticityRegion(self.name, tuple(self.modules))

    def execute(self, model):
        torch.manual_seed(self.seed)
        previous_output = model.metadata.get("output")
        if self.output: model.metadata["output"] = self.output
        try:
            before = {m: {k: model.effective_weight(m, k).clone() for k in PARAM_NAMES} for m in self.modules}
            trainable = {}
            for m in self.modules:
                source = (model.fast_parameters.get(m) or model.create_fast_delta(m)) if self.fast else model.modules[m].slow
                trainable[m] = {k: source[k].clone().requires_grad_(True) for k in self.keys}
            params = [t for d in trainable.values() for t in d.values()]
            optimizer = torch.optim.Adam(params, lr=self.lr)
            for _ in range(self.steps):
                optimizer.zero_grad()
                overrides = {}
                for m, d in trainable.items():
                    slow, fast = model.modules[m].slow, model.fast_parameters.get(m, {})
                    if self.fast:
                        overrides[m] = {k: slow[k] + (d[k] if k in d else fast.get(k, 0)) for k in PARAM_NAMES}
                    else:
                        overrides[m] = {k: (d[k] if k in d else slow[k]) + fast.get(k, 0) for k in PARAM_NAMES}
                logits = model.forward(self.inputs, overrides)
                if self.positions:
                    logits = logits.reshape(-1, logits.shape[1] // self.positions)
                loss = torch.nn.functional.cross_entropy(logits, self.targets.reshape(-1))
                loss.backward(); optimizer.step()
            for m, d in trainable.items():
                target = model.fast_parameters[m] if self.fast else model.modules[m].slow
                for k in self.keys: target[k] = d[k].detach()
        finally:
            model.metadata["output"] = previous_output
        modified = sum(int((model.effective_weight(m, k) != before[m][k]).sum()) for m in self.modules for k in PARAM_NAMES)
        model.counters["modified_parameters"] += modified
        model.counters["adaptation_steps"] += self.steps
        return {"loss": float(loss.detach()), "modified_parameters": modified,
                "region_parameters": self.region.parameter_count(model)}


@_spec
class RestoreModule(ModelTransformationSpec):
    """Targeted regeneration: a single module returns to its snapshot state."""
    name: str = "restore_module"
    module: str = ""
    snapshot: str = ""
    invertible: bool = False
    changes: frozenset = frozenset({"weights"})
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet({Invariant.MODULE_SET, Invariant.MODEL_TOPOLOGY}))
    cost: float = 0.3
    def __post_init__(self):
        object.__setattr__(self, "preconditions", (SnapshotExists(self.snapshot), SnapshotHasModule(self.snapshot, self.module)))
    def execute(self, model):
        model.modules[self.module] = copy.deepcopy(model.snapshots[self.snapshot].modules[self.module])
        model.counters["rollbacks"] += 1
        return {"restored_module": self.module, "from": self.snapshot}


@_spec
class BindRoute(ModelTransformationSpec):
    """Routing table: a configuration points to a module."""
    name: str = "bind_route"
    configuration: str = ""
    module: str = ""
    changes: frozenset = frozenset({"routes"})
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet({Invariant.SLOW_WEIGHTS, Invariant.MODULE_SET, Invariant.MODEL_TOPOLOGY}))
    cost: float = 0.05
    def __post_init__(self): object.__setattr__(self, "preconditions", (ModuleExists(self.module),))
    def execute(self, model):
        routes = model.metadata.setdefault("routes", {})
        previous = routes.get(self.configuration)
        routes[self.configuration] = self.module
        return {"route": self.configuration, "module": self.module, "previous": previous}


@_spec
class CommitFastWeights(ModelTransformationSpec):
    """Consolidation: slow += fast for the given modules."""
    name: str = "commit_fast_weights"
    modules: Tuple[str, ...] = ()
    invertible: bool = False
    changes: frozenset = frozenset({"weights"})
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet({Invariant.MODEL_FUNCTION, Invariant.MODULE_SET, Invariant.MODEL_TOPOLOGY}))
    def __post_init__(self): object.__setattr__(self, "preconditions", tuple(ModuleExists(m) for m in self.modules))
    def execute(self, model): return {"committed": sum(model.commit_fast_delta(m) for m in self.modules)}


@_spec
class ClearFastWeights(ModelTransformationSpec):
    name: str = "clear_fast_weights"
    modules: Tuple[str, ...] = ()
    invertible: bool = False
    changes: frozenset = frozenset({"fast_weights"})
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet({Invariant.SLOW_WEIGHTS, Invariant.MODULE_SET, Invariant.MODEL_TOPOLOGY}))
    cost: float = 0.05
    def execute(self, model):
        for m in (self.modules or tuple(model.fast_parameters)): model.clear_fast_delta(m)
        return {"cleared": self.modules or "all"}


@_spec
class RestoreSnapshot(ModelTransformationSpec):
    """Regeneration: the state returns to the snapshot, the history is kept (new node)."""
    name: str = "restore_snapshot"
    snapshot: str = ""
    invertible: bool = False
    changes: frozenset = frozenset({"modules", "weights", "connectivity", "fast_weights"})
    cost: float = 0.5
    def __post_init__(self): object.__setattr__(self, "preconditions", (SnapshotExists(self.snapshot),))
    def execute(self, model):
        snap = model.snapshots[self.snapshot]
        model._restore_content(snap)
        model.counters["rollbacks"] += 1
        return {"restored": self.snapshot, "from_node": snap.node_name}


@_spec
class LoadFastDelta(ModelTransformationSpec):
    """Loads the fast parametric memory of a configuration into the working buffer."""
    name: str = "load_fast_delta"
    module: str = ""
    delta: Any = field(default=None, compare=False, repr=False)
    changes: frozenset = frozenset({"fast_weights"})
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet({Invariant.SLOW_WEIGHTS, Invariant.MODULE_SET, Invariant.MODEL_TOPOLOGY}))
    cost: float = 0.05
    def __post_init__(self): object.__setattr__(self, "preconditions", (ModuleExists(self.module),))
    def execute(self, model):
        buffer = {k: torch.zeros_like(v) for k, v in model.modules[self.module].slow.items()}
        buffer.update(self.delta or {})
        model.fast_parameters[self.module] = buffer
        return {"loaded": self.module}


@_spec
class RestoreIncremental(ModelTransformationSpec):
    """Exact restoration from an IncrementalSnapshot (shared references, no deep copy)."""
    name: str = "restore_incremental"
    snapshot: Any = field(default=None, compare=False, repr=False)
    invertible: bool = False
    changes: frozenset = frozenset({"modules", "weights", "connectivity", "fast_weights", "routes"})
    cost: float = 0.3
    def execute(self, model):
        snap = self.snapshot
        model.modules = {}
        for n, m in snap.modules.items():
            clone = copy.copy(m); clone.slow = dict(m.slow); model.modules[n] = clone
        model.fast_parameters = {m: dict(d) for m, d in snap.fast.items()}
        model.connectivity = {k: copy.copy(c) for k, c in snap.connectivity.items()}
        model.metadata["routes"] = dict(snap.routes)
        model.counters["rollbacks"] += 1
        return {"restored": snap.name, "from_node": snap.node_name}

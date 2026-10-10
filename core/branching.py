"""Shared-state branches: BaseModelState + BranchDelta, copy-on-write at the tensor level.

A branch shares the tensors of the base as long as they are not reassigned.
All the specs of the model replace the tensors (never an in-place mutation),
so the first write naturally produces a local copy.
"""
from __future__ import annotations

import copy

import torch

from core.model_state import ModelState
from core.transform_graph import TransformGraph, TransformNode


def _shallow_module(module):
    clone = copy.copy(module)
    clone.slow = dict(module.slow)          # new dict, same tensors
    return clone


def shared_branch(base: ModelState) -> ModelState:
    """Creates a branched view: only the containers are new, the tensors are shared."""
    branch = ModelState.__new__(ModelState)
    branch.input_dim = base.input_dim
    branch.modules = {name: _shallow_module(m) for name, m in base.modules.items()}
    branch.fast_parameters = {m: dict(d) for m, d in base.fast_parameters.items()}
    branch.connectivity = {k: copy.copy(c) for k, c in base.connectivity.items()}
    branch.snapshots = dict(base.snapshots)                    # immutable snapshots: shared
    branch.metadata = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.metadata.items()}
    branch.counters = dict(base.counters)
    branch.transform_graph = TransformGraph(TransformNode(base.transform_graph.current.name,
                                                          configuration=base.summary()))
    branch.base = base
    return branch


def _tensors(model):
    for m in model.modules.values():
        yield from m.slow.values()
    for d in model.fast_parameters.values():
        yield from d.values()


def _bytes(t): return t.numel() * t.element_size()


def branch_memory(branch: ModelState, base: ModelState):
    """Logical memory: bytes owned by the branch vs bytes of a full copy."""
    shared_ptrs = {t.data_ptr() for t in _tensors(base)}
    total = owned = 0
    seen = set()
    for t in _tensors(branch):
        if t.data_ptr() in seen: continue
        seen.add(t.data_ptr())
        total += _bytes(t)
        if t.data_ptr() not in shared_ptrs: owned += _bytes(t)
    return {"total_bytes": total, "owned_bytes": owned, "shared_bytes": total - owned,
            "owned_ratio": owned / total if total else 0.0}


def delta_summary(branch: ModelState, base: ModelState):
    """Explicit BranchDelta: what actually differs from the base."""
    changed = {}
    for name, module in branch.modules.items():
        if name not in base.modules: changed[name] = "new_module"; continue
        keys = [k for k, t in module.slow.items() if t is not base.modules[name].slow.get(k)]
        if keys: changed[name] = keys
    return {"modules": changed,
            "removed_modules": sorted(set(base.modules) - set(branch.modules)),
            "fast": sorted(m for m, d in branch.fast_parameters.items()
                           if m not in base.fast_parameters or any(d[k] is not base.fast_parameters[m].get(k) for k in d)),
            "routes": sorted(set(branch.connectivity) ^ set(base.connectivity)),
            "active": sorted(branch.active_modules ^ base.active_modules)}


def adopt(base: ModelState, branch: ModelState):
    """The selected branch becomes the active state; the history is replayed in the base graph."""
    path = []
    node = branch.transform_graph.current
    while node.parent is not None: path.append(node); node = node.parent
    base.modules, base.fast_parameters = branch.modules, branch.fast_parameters
    base.connectivity, base.metadata, base.counters = branch.connectivity, branch.metadata, branch.counters
    for node in reversed(path):
        parent = base.transform_graph.current
        child = parent.add_child(node.transformations[-1], node.configuration, record=node.record,
                                 name=f"N{len(base.transform_graph.nodes)}")
        base.transform_graph.nodes[child.name] = child
        base.transform_graph.current = child
    return base


def tensors_close(a: ModelState, b: ModelState, tolerance=1e-6):
    if sorted(a.modules) != sorted(b.modules): return False
    return all(a.modules[n].slow[k].shape == b.modules[n].slow[k].shape
               and torch.allclose(a.modules[n].slow[k], b.modules[n].slow[k], atol=tolerance)
               for n in a.modules for k in a.modules[n].slow)


# ================================================================ incremental snapshots

class IncrementalSnapshot:
    """Snapshot = references to the tensors (copy-on-write: never mutated in place).

    Only the tensors absent from the parent snapshot count as owned memory:
    base snapshot + delta snapshots, without a full copy per task.
    """

    def __init__(self, name, state: ModelState, parent: 'IncrementalSnapshot' = None):
        self.name, self.parent = name, parent
        self.modules = {n: _shallow_module(m) for n, m in state.modules.items()}
        self.fast = {m: dict(d) for m, d in state.fast_parameters.items()}
        self.connectivity = {k: copy.copy(c) for k, c in state.connectivity.items()}
        self.routes = dict(state.metadata.get("routes", {}))
        self.node_name = state.transform_graph.current.name

    def _tensors(self):
        for m in self.modules.values(): yield from m.slow.values()
        for d in self.fast.values(): yield from d.values()

    def delta(self):
        parent = self.parent
        base_modules = parent.modules if parent else {}
        return {
            "modified_tensors": {n: [k for k, t in m.slow.items()
                                     if n not in base_modules or base_modules[n].slow.get(k) is not t]
                                 for n, m in self.modules.items()},
            "added_modules": sorted(set(self.modules) - set(base_modules)),
            "removed_modules": sorted(set(base_modules) - set(self.modules)),
            "connectivity": sorted(set(self.connectivity) ^ set(parent.connectivity if parent else {})),
            "routes": {c: m for c, m in self.routes.items() if not parent or parent.routes.get(c) != m},
            "fast": sorted(self.fast)}

    def owned_bytes(self):
        parent_ptrs = {t.data_ptr() for t in self.parent._tensors()} if self.parent else set()
        seen, owned = set(), 0
        for t in self._tensors():
            if t.data_ptr() in seen or t.data_ptr() in parent_ptrs: continue
            seen.add(t.data_ptr()); owned += _bytes(t)
        return owned

    def full_bytes(self):
        seen, total = set(), 0
        for t in self._tensors():
            if t.data_ptr() not in seen: seen.add(t.data_ptr()); total += _bytes(t)
        return total

    def restore(self, state: ModelState):
        from core.model_state import RestoreIncremental
        return state.apply(RestoreIncremental(snapshot=self))

"""SAFE plasticity: candidate adaptation (fast weights = temporary perturbation zone) → invariant checks
→ stability radius → dangerous interactions → consolidation (CommitFastWeights) OR rollback
(ClearFastWeights). Continual-learning benchmark: 5 methods compared.

Tasks (12-bit input = 10 data bits + 2 task-code bits):
  A (code 10) : (b0 xor b1) or b2       — old
  C (code 01) : parity(b4, b6, b8)      — old
  B (code 11) : majority(b5..b9) xor b3 — new
Backpropagation is used only LOCALLY (fitting of the fast weights, parameter sensitivity).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from core.model_state import PARAM_NAMES, ClearFastWeights, CommitFastWeights, LoadFastDelta
from core.plasticity import all_inputs, build_base_model, task_a_labels, task_b_labels
from core.stability import accuracy_invariant, perturbed_model, stability_radius

DIM = 12
CODES = {"A": (1.0, 0.0), "C": (0.0, 1.0), "B": (1.0, 1.0)}


def task_c_labels(x): return ((x[:, 4] + x[:, 6] + x[:, 8]) % 2).long()


LABELERS = {"A": task_a_labels, "C": task_c_labels, "B": task_b_labels}


def task_data(task, seed=0, train_fraction=0.6):
    x10 = all_inputs(10); y = LABELERS[task](x10)
    x = torch.cat([x10, torch.tensor(CODES[task]).expand(len(x10), 2)], 1)
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    cut = int(train_fraction * len(x))
    return x[perm[:cut]], y[perm[:cut]], x[perm[cut:]], y[perm[cut:]]


def old_model(seed=0, steps=400):
    """Model that has mastered A and C (initial training, declared global)."""
    from core.model_state import AdaptWeights
    m = build_base_model(input_dim=DIM, seed=seed)
    xa, ya, _, _ = task_data("A", seed); xc, yc, _, _ = task_data("C", seed)
    m.apply(AdaptWeights(name="pretrain", modules=tuple(sorted(m.active_modules)), inputs=torch.cat([xa, xc]),
                         targets=torch.cat([ya, yc]), steps=steps, lr=0.02, fast=False, local=False, seed=seed))
    m.snapshot("old")
    return m


def _forward_with(model, deltas):
    return {m: {k: model.modules[m].slow[k] + deltas[m][k] for k in PARAM_NAMES} for m in deltas}


def train_fast(model, modules, x, y, steps=300, lr=0.03, seed=0, anchors=None, lam=0.0, mask=None, replay=None):
    """Trains FAST deltas (slow weights intact). anchors=(x_old, reference outputs): output-equivalence
    constraint (invariant); mask: allowed parameters (1); replay=(x, y): replay."""
    torch.manual_seed(seed)
    deltas = {m: {k: torch.zeros_like(model.modules[m].slow[k], requires_grad=True) for k in PARAM_NAMES} for m in modules}
    opt = torch.optim.Adam([t for d in deltas.values() for t in d.values()], lr=lr)
    f = torch.nn.functional
    for _ in range(steps):
        opt.zero_grad()
        eff = {m: {k: (deltas[m][k] * mask[m][k] if mask else deltas[m][k]) for k in PARAM_NAMES} for m in modules}
        loss = f.cross_entropy(model.forward(x, _forward_with(model, eff)), y)
        if replay is not None: loss = loss + f.cross_entropy(model.forward(replay[0], _forward_with(model, eff)), replay[1])
        if anchors is not None and lam > 0:
            loss = loss + lam * ((model.forward(anchors[0], _forward_with(model, eff)) - anchors[1]) ** 2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        return {m: {k: (deltas[m][k] * mask[m][k] if mask else deltas[m][k]).detach().clone() for k in PARAM_NAMES} for m in modules}


def load(model, deltas):
    for m, d in deltas.items(): model.apply(LoadFastDelta(module=m, delta=d))


def changed_parameters(deltas, tol=1e-6):
    return sum(int((v.abs() > tol).sum()) for d in deltas.values() for v in d.values())


# ================================================================ invariants of the old knowledge

@dataclass
class OldSkillInvariants:
    """Protected invariants: performance on the old tasks ≥ reference − tolerance; normalization (softmax:
    always true, checked); topology unchanged."""
    tests: Dict[str, tuple]
    reference: Dict[str, float]
    tolerance: float = 0.02
    topology: tuple = ()

    def check(self, model):
        out = {f"old_skill:{t}": model.accuracy(x, y) >= self.reference[t] - self.tolerance for t, (x, y) in self.tests.items()}
        out["topology"] = tuple(model.topological_order()) == self.topology
        with torch.no_grad():
            p = torch.softmax(model.forward(next(iter(self.tests.values()))[0][:16]), 1)
        out["probability_normalization"] = bool(torch.allclose(p.sum(1), torch.ones(len(p)), atol=1e-5))
        return out

    def retention(self, model): return {t: model.accuracy(x, y) for t, (x, y) in self.tests.items()}


def make_invariants(model, seed=0, tolerance=0.02):
    tests = {}
    for t in ("A", "C"):
        _, _, xv, yv = task_data(t, seed); tests[t] = (xv, yv)
    return OldSkillInvariants(tests, {t: model.accuracy(*tests[t]) for t in tests}, tolerance, tuple(model.topological_order()))


# ================================================================ safe consolidation

@dataclass
class ConsolidationReport:
    decision: str
    task_gain: float
    invariants: Dict[str, bool]
    radius_before: float
    radius_after: float
    dangerous_interaction: float
    reasons: List[str] = field(default_factory=list)


def safe_consolidate(model, deltas, x_new, y_new, inv: OldSkillInvariants, base_new_acc, min_gain=0.05,
                     radius_ratio=0.5, interaction_max=0.05, check_radius=True, trials=8):
    """1 task gain; 2 old invariants; 3 stability radius (must not shrink too much); 4 dangerous
    interaction (adaptation × small perturbation); 5 consolidate (Commit) or rollback (Clear)."""
    load(model, deltas)
    gain = model.accuracy(x_new, y_new) - base_new_acc
    checks = inv.check(model); reasons = []
    if gain < min_gain: reasons.append(f"gain insuffisant ({gain:.3f})")
    reasons += [f"invariant violé : {k}" for k, ok in checks.items() if not ok]
    rb = ra = 0.0; inter = 0.0
    if check_radius and not reasons:
        xa, ya = inv.tests["A"]; floor = inv.reference["A"] - inv.tolerance
        snapshot_fast = {m: {k: v.clone() for k, v in d.items()} for m, d in model.fast_parameters.items()}
        model.fast_parameters = {}
        rb = stability_radius(accuracy_invariant(model, xa, ya, ("enc", "expert_a", "head"), floor), 0.9, hi=0.4, trials=trials, iters=3).radius
        model.fast_parameters = snapshot_fast
        ra = stability_radius(accuracy_invariant(model, xa, ya, ("enc", "expert_a", "head"), floor), 0.9, hi=0.4, trials=trials, iters=3).radius
        if rb > 0 and ra < radius_ratio * rb: reasons.append(f"rayon de stabilité rétréci ({rb:.3f} -> {ra:.3f})")
        # interaction: accuracy loss under (adaptation + noise) beyond the sum of the separate losses
        base = model.accuracy(xa, ya); mag = max(rb, 1e-3) * 0.5
        noisy = sum(perturbed_model(model, ("enc", "expert_a", "head"), mag, s).accuracy(xa, ya) for s in range(4)) / 4
        model.fast_parameters, saved = {}, model.fast_parameters
        clean = model.accuracy(xa, ya)
        noisy0 = sum(perturbed_model(model, ("enc", "expert_a", "head"), mag, s).accuracy(xa, ya) for s in range(4)) / 4
        model.fast_parameters = saved
        inter = (clean - noisy) - (clean - base) - (clean - noisy0)
        if inter > interaction_max: reasons.append(f"interaction dangereuse ({inter:.3f})")
    if reasons:
        model.apply(ClearFastWeights()); model.counters["rollbacks"] += 1
        return ConsolidationReport("rollback", gain, checks, rb, ra, inter, reasons)
    model.apply(CommitFastWeights(modules=tuple(deltas)))
    return ConsolidationReport("consolidate", gain, checks, rb, ra, inter, ["acceptable"])


# ================================================================ continual-learning benchmark

METHODS = ("vanilla", "fast_weights", "replay", "invariant_constrained", "perturbation_guided")


def parameter_sensitivity(model, x_old, n_dirs=4, eps=0.05):
    """Sensitivity of the old output to each parameter: averaged |∂output/∂θ| (LOCAL differentiable tool)."""
    params = {m: {k: model.modules[m].slow[k].clone().requires_grad_(True) for k in PARAM_NAMES} for m in model.topological_order()}
    out = model.forward(x_old, params)
    g = torch.autograd.grad((out ** 2).sum(), [t for d in params.values() for t in d.values()])
    sens, i = {}, 0
    for m, d in params.items():
        sens[m] = {}
        for k in d: sens[m][k] = g[i].abs(); i += 1
    return sens


def run_method(method, seed=0, steps=600, base=None):
    model = copy.deepcopy(base) if base is not None else old_model(seed)
    inv = make_invariants(model, seed)
    xb, yb, xbv, ybv = task_data("B", seed + 100)
    base_new = model.accuracy(xbv, ybv)
    modules = tuple(model.topological_order())
    before = {m: {k: v.clone() for k, v in model.modules[m].slow.items()} for m in modules}
    rollbacks, cost, report = 0, steps, None
    old_x = torch.cat([task_data(t, seed)[0][:300] for t in ("A", "C")])      # stored old inputs (without labels)
    if method == "vanilla":
        load(model, train_fast(model, modules, xb, yb, steps, seed=seed)); model.apply(CommitFastWeights(modules=modules))
    elif method == "fast_weights":                            # CURRENT consolidation: commit without check
        load(model, train_fast(model, modules, xb, yb, steps, lr=0.01, seed=seed)); model.apply(CommitFastWeights(modules=modules))
    elif method == "replay":
        xa, ya, _, _ = task_data("A", seed); xc, yc, _, _ = task_data("C", seed)
        g = torch.Generator().manual_seed(seed); ia = torch.randperm(len(xa), generator=g)[:48]; ic = torch.randperm(len(xc), generator=g)[:48]
        load(model, train_fast(model, modules, xb, yb, steps, seed=seed, replay=(torch.cat([xa[ia], xc[ic]]), torch.cat([ya[ia], yc[ic]]))))
        model.apply(CommitFastWeights(modules=modules))
    elif method in ("invariant_constrained", "perturbation_guided"):
        with torch.no_grad(): anchors = (old_x, model.forward(old_x))     # UNLABELLED old inputs (reference outputs)
        mask = None
        if method == "perturbation_guided":                   # modify only the parameters with LOW sensitivity for the old knowledge
            sens = parameter_sensitivity(model, old_x)
            flat = torch.cat([v.flatten() for d in sens.values() for v in d.values()])
            thr = torch.quantile(flat, 0.5)
            mask = {m: {k: (v <= thr).float() for k, v in d.items()} for m, d in sens.items()}
        cost = 0
        for attempt, lam in enumerate((0.3, 1.0, 3.0)):     # candidate → checks → consolidate, or rollback then stronger constraint
            deltas = train_fast(model, modules, xb, yb, steps, seed=seed + attempt, anchors=anchors, lam=lam, mask=mask)
            report = safe_consolidate(model, deltas, xbv, ybv, inv, base_new)
            cost += steps
            if report.decision == "consolidate": break
            rollbacks += 1
            if any(r.startswith("gain") for r in report.reasons): break    # stronger constraint => even lower gain
    changed = sum(int((model.modules[m].slow[k] != before[m][k]).sum()) for m in modules for k in PARAM_NAMES)
    ret = inv.retention(model)
    return {"method": method, "new_task": round(model.accuracy(xbv, ybv), 4), "old_A": round(ret["A"], 4), "old_C": round(ret["C"], 4),
            "retention": round(sum(ret.values()) / len(ret), 4),
            "forgetting": round(sum(inv.reference[t] - ret[t] for t in ret) / len(ret), 4),
            "parameters_changed": changed, "rollbacks": rollbacks, "adaptation_cost": cost,
            "consolidation": report.decision if report else "commit", "reasons": report.reasons if report else []}


def continual_benchmark(seed=0, steps=600):
    base = old_model(seed)
    return {m: run_method(m, seed, steps, base) for m in METHODS}

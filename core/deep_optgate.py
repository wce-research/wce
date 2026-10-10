"""Objective Alignment & Conditional Optimization — when to optimise, when to preserve, and according to which objective.

    candidate → need_for_optimization? → NO : preserve
                                       → YES : optimize proposal → validate → accept / rollback

Everything here uses only OBSERVABLE information (observations, state of the candidate). No world
truth, no import of a world module.

Contents:
  * observable objectives testés SÉPARÉMENT [tested SEPARATELY] (no opaque composite score);
  * internal signals computed BEFORE any fitting;
  * `OptimizationDecision`: auditable decision (initial state, signals, estimated gain and risk, action, result,
    rollback);
  * `conditional_fit`: explicit and interpretable conditional gate (règle à seuils [threshold rule], no learned model).
The initial state is never lost: the decision keeps the initial state, the best state encountered and the final state.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from core.boundary_program import BNode
from core.deep_fit import ACCEPT_TOL, DT, FitCounter, FitData, compile_skeleton, decide, predict_batch, profiled_loglik

OBJECTIVES = ("surrogate", "profiled_soft", "discrete_error", "likelihood_prior", "local_boundary", "robust")
ACTIONS = ("PRESERVE", "OPTIMIZE")
GATE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "deep_configs", "optimization_gate.json")
DEFAULT_GATE = {"objective": "surrogate", "residual_z": 2.0, "gap_z": 1.0, "min_gain": 0.0}        # replaced by the tuning done in dev (results/deep_configs/optimization_gate.json)


# ================================================================ observable objectives (tested separately)

def _soft(fn, th, X, k): return fn(th, X, lambda z: torch.sigmoid(k * z)).clamp(1e-6, 1 - 1e-6)


def _nll(p, data: FitData, gamma, w=None):
    p1 = (1 - gamma) * (data.flip + (1 - 2 * data.flip) * p) + gamma * 0.5; w = data.weight if w is None else w
    return -(w * (data.y * torch.log(p1) + (1 - data.y) * torch.log(1 - p1))).sum(-1)


def make_objective(name: str, skel: BNode, data: FitData, theta0, gamma=0.02) -> Callable:
    """Returns loss(theta [K, P], k) → [K] for the objective `name`. All of them use only the observations.
      surrogate        : current softened robust likelihood (contamination γ = 0.02)
      profiled_soft    : PROFILED likelihood (noise estimated by the state itself) on the softened errors
      likelihood_prior : surrogate + simplicity prior on the DISPLACEMENT (‖θ − θ0‖² / 2σ², σ = 0.1):
                         "move away from the proposed state only if the data require it"
      local_boundary   : surrogate restricted to the neighbourhood of the boundary of the initial state (weight 1 near, 0.2 far)
      robust           : surrogate with strong contamination (γ = 0.2), less sensitive to outliers
    `discrete_error` is not differentiable: it is optimised by direct search (`pattern_search`)."""
    fn = compile_skeleton(skel); X = data.X[None]; th0 = torch.as_tensor(theta0, dtype=DT).reshape(1, -1)
    if name == "surrogate": return lambda th, k: _nll(_soft(fn, th, X, k), data, gamma)
    if name == "robust": return lambda th, k: _nll(_soft(fn, th, X, k), data, 0.2)
    if name == "likelihood_prior": return lambda th, k: _nll(_soft(fn, th, X, k), data, gamma) + ((th - th0) ** 2).sum(-1) / (2 * 0.1 ** 2)
    if name == "local_boundary":
        with torch.no_grad(): p0 = _soft(fn, th0, X, 8.0)[0]; wl = torch.where((p0 > 0.1) & (p0 < 0.9), torch.ones_like(p0), torch.full_like(p0, 0.2)) * data.weight
        return lambda th, k: _nll(_soft(fn, th, X, k), data, gamma, wl)
    if name == "profiled_soft":
        n = float(data.weight.sum())
        def f(th, k):
            p = _soft(fn, th, X, k); e = (data.weight * (data.y * (1 - p) + (1 - data.y) * p)).sum(-1); eps = ((e + 0.5) / (n + 1.0)).clamp(1e-3, 0.49)
            return -(e * eps.log() + (n - e) * (1 - eps).log())
        return f
    raise KeyError(name)


def hard_errors(skel: BNode, th, data: FitData):
    """Number of DISCRETE errors on the observations, per row."""
    return ((predict_batch(skel, th, data.X).to(DT) != data.y).to(DT) * data.weight).sum(-1)


def profiled(skel: BNode, row, data: FitData): return profiled_loglik(predict_batch(skel, row.reshape(1, -1), data.X)[0].to(DT), data)


@dataclass
class OptTrace:
    """States kept from an optimisation: initial, best encountered (valid internal criterion), final."""
    initial: torch.Tensor
    best: torch.Tensor
    best_step: int
    final: torch.Tensor
    steps: List[dict]
    objective: str


def optimize(skel: BNode, data: FitData, init, objective="surrogate", steps=100, lr=0.05, every=10, counter: Optional[FitCounter] = None) -> OptTrace:
    """Optimises ONE candidate according to `objective`. Mechanism unchanged for all differentiable objectives (Adam,
    stiffness 4 → 24, 100 steps): only the objective varies. Keeps the initial state, the best state (profiled likelihood
    on the observations) and the final state."""
    th0 = torch.as_tensor(init, dtype=DT).reshape(1, -1).clone()
    if objective == "discrete_error": return pattern_search(skel, data, th0, counter=counter)
    f = make_objective(objective, skel, data, th0); th = th0.clone().requires_grad_(True); opt = torch.optim.Adam([th], lr=lr); log = []; best = (profiled(skel, th0[0], data), th0[0].clone(), 0)
    def snap(t):
        cur = th.detach(); pl = profiled(skel, cur[0], data); nonlocal best
        if pl > best[0] + ACCEPT_TOL: best = (pl, cur[0].clone(), t)
        log.append({"step": t, "objective": float(f(cur, 24.0)[0]), "profiled": pl, "errors": float(hard_errors(skel, cur, data)[0]), "params": cur[0].tolist()})
    for t in range(steps):
        if t % every == 0: snap(t)
        opt.zero_grad(); f(th, 4.0 + 20.0 * t / steps).sum().backward(); opt.step()
    snap(steps)
    if counter is not None: counter.charge(1, steps)
    return OptTrace(th0[0], best[1], best[2], th.detach()[0].clone(), log, objective)


def pattern_search(skel: BNode, data: FitData, th0, rounds=4, step0=0.08, counter: Optional[FitCounter] = None) -> OptTrace:
    """Direct search on the observable DISCRETE ERROR: each parameter is tried at ±step; a move is
    kept only if it strictly reduces the number of errors; the step is halved at each round. Monotone by
    construction on its own criterion."""
    cur = th0.clone(); e = float(hard_errors(skel, cur, data)[0]); P = cur.shape[1]; log = [{"step": 0, "objective": e, "profiled": profiled(skel, cur[0], data), "errors": e, "params": cur[0].tolist()}]; step = step0; evals = 0
    for r in range(rounds):
        cand = cur.repeat(2 * P, 1); idx = torch.arange(P); cand[idx, idx] += step; cand[P + idx, idx] -= step; es = hard_errors(skel, cand, data); evals += 2 * P
        i = int(es.argmin())
        while P and float(es[i]) < e - 1e-9:
            cur = cand[i:i + 1].clone(); e = float(es[i]); cand = cur.repeat(2 * P, 1); cand[idx, idx] += step; cand[P + idx, idx] -= step; es = hard_errors(skel, cand, data); evals += 2 * P; i = int(es.argmin())
        step /= 2; log.append({"step": r + 1, "objective": e, "profiled": profiled(skel, cur[0], data), "errors": e, "params": cur[0].tolist()})
    if counter is not None: counter.charge(1, max(1, evals // 2))                      # cost: one evaluation ≈ half a gradient step
    return OptTrace(th0[0], cur[0].clone(), rounds, cur[0].clone(), log, "discrete_error")


# ================================================================ internal signals (before any fitting)

def signals(skel: BNode, data: FitData, init, gamma=0.02) -> dict:
    """OBSERVABLE signals of a candidate, computed before optimising:
      error_rate         initial quality (error rate on the observations)
      profiled_per_point profiled likelihood per observation
      residual_z         residual: spatial clustering of the errors (z > 0: the errors clump together → structured
                         misfit; z ≈ 0: errors scattered like noise)
      gap_z              confidence: gap to a non-parametric reference (vote of 7 neighbours, cross-validation);
                         z > 0: a local model explains the data better than the candidate
      grad_per_point     initial gradient of the surrogate objective (stiffness 24)
      probe_delta_errors disagreement between criteria: change in the DISCRETE errors after a small descent step on
                         the continuous objective (> 0: the continuous one asks for a move that the discrete one refuses)
      knn_error, n, n_params"""
    th0 = torch.as_tensor(init, dtype=DT).reshape(1, -1); n = int((data.weight > 0).sum()); pred = predict_batch(skel, th0, data.X)[0].to(DT); err = pred != data.y; k = int(err.sum()); e0 = k / max(1, n)
    D = torch.cdist(data.X, data.X); D.fill_diagonal_(float("inf")); nn = D.argmin(1); joint = float(err[nn][err].double().sum())
    z = lambda cnt, m, p: (cnt - m * p) / math.sqrt(max(1e-9, m * p * (1 - p))) if m > 0 else 0.0
    res_z = z(joint, k, min(0.49, max(0.005, e0))) if k >= 3 else 0.0
    idx = D.topk(min(7, n - 1), largest=False).indices; knn = float(((data.y[idx].mean(1) >= 0.5).to(DT) != data.y).double().mean()); gap = z(k, n, min(0.49, max(0.005, knn)))
    f = make_objective("surrogate", skel, data, th0, gamma); v = th0.clone().requires_grad_(True); g = torch.autograd.grad(f(v, 24.0).sum(), v)[0]
    probe = th0 - 0.01 * g / g.norm().clamp_min(1e-12); de = float(hard_errors(skel, probe, data)[0]) - k
    return {"error_rate": e0, "errors": float(k), "profiled_per_point": profiled_loglik(pred, data) / max(1, n), "residual_z": res_z, "gap_z": gap, "knn_error": knn, "grad_per_point": float(g.norm()) / max(1, n),
            "probe_delta_errors": de, "n": n, "n_params": int(th0.shape[1])}


# ================================================================ auditable decision

@dataclass
class OptimizationDecision:
    """AUDITABLE optimisation decision. Nothing is lost: initial state, best state encountered and final state of
    the optimiser are kept; `result` is the retained state."""
    initial: tuple
    signals: dict
    gain_estimate: float                    # errors that the optimisation could correct (excess over the local reference)
    risk_estimate: float                    # risk of degrading a state already consistent with the data (0 … 1)
    action: str                             # "PRESERVE" | "OPTIMIZE"
    reason: str
    objective: str = ""
    proposal: Optional[tuple] = None        # final state of the optimiser (None if PRESERVE)
    best_so_far: Optional[tuple] = None     # best state encountered according to the profiled likelihood
    value_before: Optional[float] = None
    value_after: Optional[float] = None
    accepted: Optional[bool] = None
    rolled_back: bool = False
    result: tuple = ()

    def to_dict(self): return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def load_gate() -> dict:
    try: return {**DEFAULT_GATE, **json.load(open(GATE_FILE))}
    except (OSError, ValueError): return dict(DEFAULT_GATE)


def need_for_optimization(sig: dict, gate: Optional[dict] = None):
    """EXPLICIT gate (threshold rule, readable): optimise only if the observations signal a STRUCTURED
    misfit — clumped errors (residual_z) or candidate worse than a local reference (gap_z). Otherwise the
    residual looks like noise: preserve. Returns (action, reason, estimated gain, estimated risk)."""
    gate = gate or load_gate(); gain = max(0.0, sig["error_rate"] - sig["knn_error"]) * sig["n"]
    risk = 1.0 / (1.0 + math.exp(max(sig["residual_z"] - gate["residual_z"], sig["gap_z"] - gate["gap_z"])))             # close to 1 when nothing signals a misfit
    if sig["residual_z"] >= gate["residual_z"]: return "OPTIMIZE", "résidu spatialement structuré", gain, risk
    if sig["gap_z"] >= gate["gap_z"]: return "OPTIMIZE", "candidat moins bon qu'une référence locale", gain, risk
    return "PRESERVE", "résidu compatible avec du bruit", gain, risk


def conditional_fit(skel: BNode, data: FitData, init, gate: Optional[dict] = None, counter: Optional[FitCounter] = None, steps=100) -> OptimizationDecision:
    """candidate → need_for_optimization? → NO : preserve → YES : optimize proposal → validate → accept / rollback.
    The validation is that of the guardrail (profiled likelihood), with a minimal gain `min_gain` (nats); the
    retained candidate is the BEST state encountered (never worse than the initial state on this criterion)."""
    gate = gate or load_gate(); th0 = torch.as_tensor(init, dtype=DT).reshape(-1).clone(); sig = signals(skel, data, th0); action, reason, gain, risk = need_for_optimization(sig, gate)
    d = OptimizationDecision(tuple(th0.tolist()), sig, gain, risk, action, reason, result=tuple(th0.tolist()))
    if action == "PRESERVE": return d
    tr = optimize(skel, data, th0, gate["objective"], steps=steps, counter=counter); d.objective = gate["objective"]; d.proposal = tuple(tr.final.tolist()); d.best_so_far = tuple(tr.best.tolist())
    finite = bool(torch.isfinite(tr.best).all()); vb = profiled(skel, th0, data); va = profiled(skel, tr.best, data) if finite else float("nan"); ok, why = decide(vb, va, finite, tol=max(ACCEPT_TOL, gate["min_gain"]))
    d.value_before, d.value_after, d.accepted, d.rolled_back = vb, va, ok, not ok; d.reason = f"{reason} ; {why}"; d.result = tuple(tr.best.tolist()) if ok else tuple(th0.tolist()); return d

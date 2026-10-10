"""BATCH fitting engine — ANOTHER IMPLEMENTATION of the fit, not a transparent optimisation.

Same likelihood (contamination), same stiffness schedule, same optimiser (Adam, same hyperparameters) as
`fit_robust` (previous milestone). What changes:
  * the tree traversal is COMPILED once per skeleton (parameter offsets precomputed);
  * K initialisations (or K candidates with the same skeleton) are fitted IN A SINGLE pass, tensor [K, P].

MEASURED speed / fidelity trade-off (docs/rapport_validated_optimizer.md):
  * speed: ≈ 7× per batch of 8 initialisations; ≈ 1.1× for an isolated fit;
  * fidelity: on short, well-conditioned cases the gap with `fit_robust` is ≈ 1e-7; on composite
    candidates and 100 steps, the parameter gap exceeds 1e-3 in a non-negligible share of the tasks (the old
    fitter rounds the initialisation to single precision and the optimisation amplifies this gap). The top-ranked one
    stays the same in ≈ 98 % of the tasks, the full order in ≈ 87 %.
The preregistered hypothesis H10(b) (gap ≤ 1e-6) is therefore REFUTED on the composites: this engine is NOT
numerically equivalent to the old one in general, although both have the same measured scientific behaviour
(same outcome at the 0.99 threshold on the same starting points).

This file also contains the VALIDATED fit (propose → evaluate → commit / rollback), the only safeguard of the pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from core.boundary_program import BNode, _inits

DT = torch.float64


@dataclass
class FitData:
    """Observations for the fit: points, labels, per-observation noise, weights (0 = don't-care point)."""
    X: torch.Tensor
    y: torch.Tensor
    flip: torch.Tensor
    weight: torch.Tensor

    @staticmethod
    def build(X, y, flip=0.05, weight=None):
        X = torch.nan_to_num(X).to(DT); n = len(X); f = flip if torch.is_tensor(flip) else torch.full((n,), float(flip))
        return FitData(X, y.to(DT), f.to(DT), (torch.ones(n) if weight is None else weight).to(DT))

    def with_target(self, y, weight=None): return FitData(self.X, y.to(DT), self.flip, self.weight if weight is None else weight.to(DT))
    def with_flip(self, flip): return FitData(self.X, self.y, torch.full((len(self.X),), float(flip), dtype=DT), self.weight)
    def add(self, x, y, flip): return FitData(torch.cat([self.X, x.to(DT)]), torch.cat([self.y, y.to(DT)]), torch.cat([self.flip, torch.full((len(x),), float(flip), dtype=DT)]), torch.cat([self.weight, torch.ones(len(x), dtype=DT)]))
    def __len__(self): return int((self.weight > 0).sum())


def _wrap(a): return torch.atan2(torch.sin(a), torch.cos(a))


_COMPILED: Dict[str, Callable] = {}


def compile_skeleton(skel: BNode) -> Callable:
    """Compiles a skeleton into a function f(theta [K, P], X [1 or K, N, 2], sig) → [K, N]. Same formulas as
    `core.boundary_program.evaluate`."""
    key = repr(skel.skeleton())
    if key in _COMPILED: return _COMPILED[key]
    off = [0]

    def build(n: BNode):
        m = n.own_count(); i = off[0]; off[0] += m; ch = [build(c) for c in n.children]
        if n.kind == "pred":
            if n.feat == "dot":
                def f0(th, X): return torch.cos(th[:, i:i + 1]) * X[..., 0] + torch.sin(th[:, i:i + 1]) * X[..., 1]
                if n.op == "le": return lambda th, X, sig: sig(th[:, i + 1:i + 2] - f0(th, X))
                if n.op == "ge": return lambda th, X, sig: sig(f0(th, X) - th[:, i + 1:i + 2])
                def dot_in(th, X, sig): f = f0(th, X); return sig(f - th[:, i + 1:i + 2]) * sig(th[:, i + 2:i + 3] - f)
                return dot_in
            if n.feat == "norm":
                if n.op == "le": return lambda th, X, sig: sig(th[:, i:i + 1] - X.norm(dim=-1))
                if n.op == "ge": return lambda th, X, sig: sig(X.norm(dim=-1) - th[:, i:i + 1])
                def norm_in(th, X, sig): f = X.norm(dim=-1); return sig(f - th[:, i:i + 1]) * sig(th[:, i + 1:i + 2] - f)
                return norm_in
            if n.feat == "angle":
                return lambda th, X, sig: sig(th[:, i + 1:i + 2].abs() - _wrap(torch.atan2(X[..., 1], X[..., 0]) - th[:, i:i + 1]).abs())
            raise KeyError(n.feat)
        c0 = ch[0]
        if n.kind == "translate": return lambda th, X, sig: c0(th, X - th[:, None, i:i + 2], sig)
        if n.kind == "rotate":
            def rot(th, X, sig):
                c, s = torch.cos(th[:, i:i + 1]), torch.sin(th[:, i:i + 1]); return c0(th, torch.stack([c * X[..., 0] + s * X[..., 1], -s * X[..., 0] + c * X[..., 1]], -1), sig)
            return rot
        if n.kind == "scale": return lambda th, X, sig: c0(th, X / th[:, None, i:i + 1].abs().clamp_min(1e-3), sig)
        if n.kind == "reflect":
            flipv = torch.tensor([1.0, -1.0], dtype=DT); return lambda th, X, sig: c0(th, X * flipv, sig)
        if n.kind == "not": return lambda th, X, sig: 1 - c0(th, X, sig)
        c1 = ch[1]
        if n.kind == "and": return lambda th, X, sig: c0(th, X, sig) * c1(th, X, sig)
        if n.kind == "or":
            def f_or(th, X, sig): a, b = c0(th, X, sig), c1(th, X, sig); return a + b - a * b
            return f_or
        if n.kind == "xor":
            def f_xor(th, X, sig): a, b = c0(th, X, sig), c1(th, X, sig); return a + b - 2 * a * b
            return f_xor
        raise KeyError(n.kind)

    fn = build(skel); _COMPILED[key] = fn; return fn


def forward(skel: BNode, theta, X, k: Optional[float]):
    """Soft probabilities [K, N] (finite k) or hard values (k None)."""
    sig = (lambda z: torch.sigmoid(k * z)) if k is not None else (lambda z: (z >= 0).to(DT))
    return compile_skeleton(skel)(theta, X[None] if X.dim() == 2 else X, sig)


def predict_batch(skel: BNode, theta, X): return (forward(skel, theta, torch.nan_to_num(X).to(DT), None) >= 0.5).long()


def loglik_batch(skel: BNode, theta, data: FitData, gamma=0.02, k=30.0):
    """log P(D | structure, θ) per row of the batch (contamination mixture, like `obs_loglik`)."""
    p = forward(skel, theta, data.X, k).clamp(1e-6, 1 - 1e-6); p1 = (1 - gamma) * (data.flip + (1 - 2 * data.flip) * p) + gamma * 0.5
    return (data.weight * (data.y * torch.log(p1) + (1 - data.y) * torch.log(1 - p1))).sum(1)


class FitCounter:
    """Counts the computation in "fit-equivalents" (one initialisation × 100 steps) and in candidates."""
    def __init__(self, budget: Optional[float] = None): self.fit_equivalents = 0.0; self.candidates = 0; self.calls = 0; self.budget = budget
    def charge(self, k, steps): self.fit_equivalents += k * steps / 100.0; self.calls += 1
    def exhausted(self): return self.budget is not None and self.fit_equivalents >= self.budget
    def remaining(self): return math.inf if self.budget is None else max(0.0, self.budget - self.fit_equivalents)


def profiled_loglik(pred, data: FitData):
    """logL of the HARD predictions with PROFILED noise: the flip rate ε is a nuisance parameter estimated
    per hypothesis (ε̂ = smoothed error rate). The noise of the world therefore does not have to be known: a deeper structure
    is preferred only if the drop in error pays for its complexity penalty."""
    w = data.weight; n = float(w.sum()); e = float(((pred != data.y).to(DT) * w).sum()); eps = min(0.49, max(1e-3, (e + 0.5) / (n + 1.0)))
    return e * math.log(eps) + (n - e) * math.log(1 - eps)


def fit_batch(skel: BNode, data: FitData, inits, steps=100, gamma=0.02, lr=0.05, counter: Optional[FitCounter] = None):
    """Fits K parameter sets of the same skeleton in one pass. Returns (theta [K, P], logL [K]).
    Schedule identical to `fit_robust`: Adam(lr), stiffness k = 4 + 20·t/steps, contamination likelihood."""
    th0 = inits if torch.is_tensor(inits) else torch.stack([torch.as_tensor(i, dtype=DT) for i in inits])
    th = th0.clone().to(DT).requires_grad_(True)
    if th.shape[1] == 0: return th.detach(), loglik_batch(skel, th.detach(), data, gamma)
    opt = torch.optim.Adam([th], lr=lr); fn = compile_skeleton(skel); X = data.X[None]; f, w, y = data.flip, data.weight, data.y
    for t in range(steps):
        opt.zero_grad(); kk = 4.0 + 20.0 * t / steps
        p = fn(th, X, lambda z: torch.sigmoid(kk * z)).clamp(1e-6, 1 - 1e-6); p1 = (1 - gamma) * (f + (1 - 2 * f) * p) + gamma * 0.5
        (-(w * (y * torch.log(p1) + (1 - y) * torch.log(1 - p1))).sum()).backward(); opt.step()
    th = th.detach()
    if counter is not None: counter.charge(len(th), steps)
    return th, loglik_batch(skel, th, data, gamma)


def heuristic_inits(skel: BNode, data: FitData, n_init: int, seed: int):
    """Heuristic initialisations of the previous milestone (`_inits`), on the points with non-zero weight."""
    m = data.weight > 0; g = torch.Generator().manual_seed(seed)
    return torch.stack([v.to(DT) for v in _inits(skel, data.X[m].float(), data.y[m].long(), n_init, g)])


def best_of(skel: BNode, theta, ll):
    """Best row of the batch → (fitted program, logL)."""
    i = int(ll.argmax()); return skel.skeleton().with_params(theta[i].tolist()), float(ll[i])


def approx_evidence(n_params: int, n_atoms: int, ll: float, n_obs: int, bits_per_node=3.0):
    """APPROXIMATE evidence (BIC-type + structural prior), same form as the previous milestone: it is not an
    exact marginal."""
    return ll - 0.5 * n_params * math.log(max(1, n_obs)) - (2 * n_atoms - 1) * bits_per_node * math.log(2)


# ================================================================ fit by SEPARATION (scientific change, not an optimisation)

def separation_fit(skel: BNode, P, Q, pair_weight, inits, steps=50, lr=0.05, counter: Optional[FitCounter] = None):
    """SCIENTIFIC CHANGE (candidate generation) — polarity-independent objective: fit an atom so that its
    boundary passes BETWEEN the two points of each boundary pair (neighbouring points with opposite labels).
    In an and / or / xor composition, an atom masked by an XOR is not correlated with the target, but its boundary remains
    locally visible: this objective finds it where the fit on the target fails.
    Separation probability of a pair: m(P)(1 − m(Q)) + m(Q)(1 − m(P)), m = soft membership.
    Returns (theta [K, np], hard weighted separation [K])."""
    th = (inits if torch.is_tensor(inits) else torch.stack([torch.as_tensor(i, dtype=DT) for i in inits])).clone().to(DT).requires_grad_(True)
    fn = compile_skeleton(skel); P, Q, w = P.to(DT)[None], Q.to(DT)[None], pair_weight.to(DT); opt = torch.optim.Adam([th], lr=lr)
    for t in range(steps):
        opt.zero_grad(); kk = 3.0 + 12.0 * t / steps; sig = lambda z: torch.sigmoid(kk * z)
        mp, mq = fn(th, P, sig), fn(th, Q, sig); (-(w * (mp * (1 - mq) + mq * (1 - mp))).sum()).backward(); opt.step()
    th = th.detach()
    if counter is not None: counter.charge(len(th), steps)
    hard = lambda z: (z >= 0).to(DT); return th, (w * (fn(th, P, hard) != fn(th, Q, hard)).to(DT)).sum(1)


# ================================================================ VALIDATED fit: propose → evaluate → commit / rollback

ACCEPT_TOL = 1e-9                 # numerical tolerance (nats): below it, the fitted state is not considered better
CRITERIA = ("profiled", "surrogate")


@dataclass(frozen=True)
class FitTransaction:
    """Trace of a validated fit. The initial state is never modified: it is a distinct tensor, kept until
    the decision."""
    before: tuple                  # parameters of the initial state (immutable)
    proposal: tuple                # parameters proposed by the optimiser
    criterion: str
    value_before: float
    value_after: float
    accepted: bool
    reason: str                    # "improved" | "not_improved" | "equal" | "non_finite_proposal" | "non_finite_before"

    @property
    def committed(self): return self.proposal if self.accepted else self.before


def fit_criterion(skel: BNode, theta_row, data: FitData, criterion="profiled", gamma=0.02) -> float:
    """OBSERVABLE criterion of a state (no truth): `profiled` = profiled likelihood of the hard predictions on the
    observations (the one the system then uses to record and rank); `surrogate` = soft robust
    likelihood (stiffness 30) that the optimiser maximises."""
    th = theta_row.reshape(1, -1).to(DT)
    if criterion == "profiled": return profiled_loglik(predict_batch(skel, th, data.X)[0].to(DT), data)
    if criterion == "surrogate": return float(loglik_batch(skel, th, data, gamma)[0])
    raise KeyError(criterion)


def decide(value_before: float, value_after: float, proposal_finite: bool, tol=ACCEPT_TOL):
    """Acceptance rule, deterministic:
      * non-finite proposal (NaN / inf in the parameters or in the criterion) → rollback;
      * initial state with a non-finite criterion and finite proposal → commit (one cannot do worse);
      * strict improvement beyond the tolerance (after − before > tol) → commit;
      * equality up to the tolerance (|after − before| ≤ tol) → rollback: without evidence of improvement, the initial state
        is kept;
      * otherwise → rollback."""
    if not proposal_finite or not math.isfinite(value_after): return False, "non_finite_proposal"
    if not math.isfinite(value_before): return True, "non_finite_before"
    d = value_after - value_before
    if d > tol: return True, "improved"
    return False, "equal" if abs(d) <= tol else "not_improved"


def validated_fit(skel: BNode, data: FitData, init_row, steps=100, gamma=0.02, lr=0.05, counter: Optional[FitCounter] = None, criterion="profiled", tol=ACCEPT_TOL):
    """Fit with transactional semantics: initial state → proposal of the optimiser → evaluation on the
    observable criterion → commit or rollback. The optimiser (`fit_batch`) is unchanged; only the replacement of the state
    is conditional. Returns (retained parameters [P], FitTransaction)."""
    before = torch.as_tensor(init_row, dtype=DT).reshape(-1).clone(); frozen = before.clone()                       # the initial state is not modified in place
    prop, _ = fit_batch(skel, data, before.reshape(1, -1), steps=steps, gamma=gamma, lr=lr, counter=counter); prop = prop[0]
    assert torch.equal(before, frozen), "the optimiser modified the initial state in place"
    finite = bool(torch.isfinite(prop).all()); vb = fit_criterion(skel, before, data, criterion, gamma); va = fit_criterion(skel, prop, data, criterion, gamma) if finite else float("nan")
    ok, reason = decide(vb, va, finite, tol); tx = FitTransaction(tuple(before.tolist()), tuple(prop.tolist()), criterion, vb, va, ok, reason)
    return (prop if ok else before), tx

"""CONTINUOUS, noisy, drifting, NON-enumerable world: x ∈ R^16 (+ continuous context cue ∈ R^8, no task_id).

Rule families (parameterised, compact): separation (half-space), ball (containment), angular sector
(orientation), relative distance between two 2D points, box, compared projection, change of reference frame
(rotation then threshold), compositions (and / or / xor), and a "new domain" family (sinusoidal boundary)
absent from the system's hypothesis library.
Noises: Gaussian, heavy (Student), missing values (NaN), outliers, partial corruption of dimensions.
Drift: one parameter of the rule (threshold, angle, radius) moves PROGRESSIVELY over several regimes
(same context); recurrences; novelties; new domains.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import torch

D, D_CUE = 16, 8
FAMILIES = ("halfspace", "ball", "sector", "reldist", "box", "projection", "frame")
NOVEL_FAMILIES = ("sine",)
OPS = ("and", "or", "xor")


# ================================================================ rules

@dataclass(frozen=True)
class Rule:
    family: str
    params: Tuple = ()

    def __call__(self, X): return rule_eval(self, X)

    def size(self): return 1 + len(_flat(self.params))


def _flat(p):
    out = []
    for v in p:
        if isinstance(v, (tuple, list)): out += _flat(v)
        elif isinstance(v, Rule): out += [0] * v.size()
        else: out.append(v)
    return out


def _clean(X): return torch.nan_to_num(X, nan=0.0)


def rule_eval(r: Rule, X):
    X = _clean(X); f, p = r.family, r.params
    if f == "halfspace":
        w, b = torch.tensor(p[0]), p[1]; return (X @ w > b).long()
    if f == "ball":
        dims, c, rad = list(p[0]), torch.tensor(p[1]), p[2]; return ((X[:, dims] - c).norm(dim=1) < rad).long()
    if f == "sector":
        i, j, th, width = p; a = torch.atan2(X[:, j], X[:, i]); d = torch.remainder(a - th + math.pi, 2 * math.pi) - math.pi
        return (d.abs() < width).long()
    if f == "reldist":
        i, j, k, l, tau = p; return ((X[:, [i, j]] - X[:, [k, l]]).norm(dim=1) < tau).long()
    if f == "box":
        i, j, a, b = p; return ((X[:, i].abs() < a) & (X[:, j].abs() < b)).long()
    if f == "projection":
        u, v, b = torch.tensor(p[0]), torch.tensor(p[1]), p[2]; return (X @ u > X @ v + b).long()
    if f == "frame":
        i, j, phi, b = p; return (math.cos(phi) * X[:, i] - math.sin(phi) * X[:, j] > b).long()
    if f == "sine":
        i, j, freq, b = p; return (X[:, j] > torch.sin(freq * X[:, i]) + b).long()
    if f == "not": return 1 - rule_eval(p[0], X)
    if f in OPS:
        a, c = rule_eval(p[0], X), rule_eval(p[1], X)
        return (a * c) if f == "and" else ((a + c) > 0).long() if f == "or" else (a != c).long()
    raise KeyError(f)


def _unit(rng, d=D):
    v = [rng.gauss(0, 1) for _ in range(d)]; n = math.sqrt(sum(x * x for x in v)); return tuple(x / n for x in v)


def random_rule(rng: random.Random, family=None, mu=None):
    """Rule of one family, threshold set for a ≈ 50 % balance under the input distribution of the regime."""
    f = family or rng.choice(FAMILIES); mu = mu if mu is not None else torch.zeros(D)
    g = torch.Generator().manual_seed(rng.randrange(10 ** 9)); Xs = mu + torch.randn(4000, D, generator=g)
    if f == "halfspace":
        w = _unit(rng); b = float((Xs @ torch.tensor(w)).median()); return Rule(f, (w, b))
    if f == "ball":
        dims = tuple(rng.sample(range(D), 3)); c = tuple(float(mu[d]) + rng.gauss(0, 0.5) for d in dims)
        rad = float((Xs[:, list(dims)] - torch.tensor(c)).norm(dim=1).median()); return Rule(f, (dims, c, rad))
    if f == "sector":
        i, j = rng.sample(range(D), 2); return Rule(f, (i, j, rng.uniform(-math.pi, math.pi), math.pi / 2))
    if f == "reldist":
        i, j, k, l = rng.sample(range(D), 4); tau = float((Xs[:, [i, j]] - Xs[:, [k, l]]).norm(dim=1).median()); return Rule(f, (i, j, k, l, tau))
    if f == "box":
        i, j = rng.sample(range(D), 2); a = float((Xs[:, i] - 0).abs().quantile(0.71)); b = float(Xs[:, j].abs().quantile(0.71)); return Rule(f, (i, j, a, b))
    if f == "projection":
        u, v = _unit(rng), _unit(rng); b = float((Xs @ (torch.tensor(u) - torch.tensor(v))).median()); return Rule(f, (u, v, b))
    if f == "frame":
        i, j = rng.sample(range(D), 2); phi = rng.uniform(-math.pi, math.pi)
        b = float((math.cos(phi) * Xs[:, i] - math.sin(phi) * Xs[:, j]).median()); return Rule(f, (i, j, phi, b))
    if f == "sine":
        i, j = rng.sample(range(D), 2); fr = rng.choice([2.0, 3.0]); b = float((Xs[:, j] - torch.sin(fr * Xs[:, i])).median()); return Rule(f, (i, j, fr, b))
    raise KeyError(f)


def drift_rule(r: Rule, step: float):
    """PROGRESSIVE drift: the threshold / angle / radius parameter moves by `step`."""
    f, p = r.family, list(r.params)
    if f in ("halfspace", "projection"): p[-1] = p[-1] + step
    elif f == "ball": p[2] = max(0.2, p[2] + step)
    elif f == "sector": p[2] = p[2] + step
    elif f in ("reldist",): p[4] = max(0.2, p[4] + step)
    elif f == "box": p[2] = max(0.2, p[2] + step)
    elif f in ("frame",): p[2] = p[2] + step
    elif f == "sine": p[3] = p[3] + step
    else: return r
    return Rule(f, tuple(p))


# ================================================================ noise

NOISES = ("clean", "gaussian", "heavy", "missing", "outliers", "corruption")


def apply_noise(X, kind, g: torch.Generator, level=1.0):
    X = X.clone()
    if kind == "gaussian": X = X + 0.15 * level * torch.randn(X.shape, generator=g)
    elif kind == "heavy":                                            # Student-t with 2 degrees of freedom (heavy tails)
        z = torch.randn(X.shape, generator=g); c = (torch.randn(X.shape, generator=g) ** 2 + torch.randn(X.shape, generator=g) ** 2) / 2
        X = X + 0.1 * level * z / c.sqrt().clamp_min(1e-3)
    elif kind == "missing": X[torch.rand(X.shape, generator=g) < 0.1 * level] = float("nan")
    elif kind == "outliers":
        m = torch.rand(len(X), generator=g) < 0.05 * level; X[m] = X[m] * 8.0
    elif kind == "corruption":
        d = int(torch.randint(0, D, (1,), generator=g)); X[:, d] = torch.randn(len(X), generator=g) * 3
    return X


# ================================================================ regimes and streams

@dataclass
class Regime:
    name: str
    rule: Rule
    mu: torch.Tensor                 # centre of the input distribution (region)
    scale: float
    cue: torch.Tensor                # cue prototype (hidden)
    kind: str                        # new | recurrence | composition | drift | new_domain | noise_shift
    noise: str = "clean"
    context_id: int = 0              # ground truth of the context (EVALUATION only)
    version: int = 0

    def sample(self, n, g: torch.Generator, noisy=True):
        X = self.mu + self.scale * torch.randn(n, D, generator=g)
        y = self.rule(X)
        if noisy: X = apply_noise(X, self.noise, g)
        C = self.cue + 0.15 * torch.randn(n, D_CUE, generator=g)
        return X, C, y

    def test_set(self, seed, n=512):
        g = torch.Generator().manual_seed(seed); X = self.mu + self.scale * torch.randn(n, D, generator=g)
        return X, self.cue + 0.15 * torch.randn(n, D_CUE, generator=g), self.rule(X)


def _cue(rng, used, min_dist=2.5):
    for _ in range(2000):
        c = torch.tensor([rng.gauss(0, 1.5) for _ in range(D_CUE)])
        if all(float((c - u).norm()) >= min_dist for u in used): return c
    raise RuntimeError("cue")


def stream(seed=0, n=100, p_recur=0.15, p_comp=0.15, p_drift=0.25, p_novel=0.05, p_noise=0.3, hold_out=("sine",), families=None):
    """Stream of n regimes: novelties, recurrences (same cue, same rule), compositions, progressive DRIFTS
    (same cue, shifted parameter), new domains (family outside the library), variable noise."""
    rng = random.Random(seed); regs: List[Regime] = []; cues = []; contexts = {}
    while len(regs) < n:
        u = rng.random(); live = [r for r in regs]
        noise = rng.choice(NOISES[1:]) if rng.random() < p_noise else "clean"
        if live and u < p_recur:
            r = rng.choice(live); regs.append(replace(r, name=f"R{len(regs)}", kind="recurrence", noise=noise)); continue
        if live and u < p_recur + p_drift:
            last = max((r for r in regs if r.kind != "composition"), key=lambda r: regs.index(r), default=None)
            base = regs[-1] if regs[-1].kind in ("new", "drift", "new_domain") and rng.random() < 0.7 else rng.choice(live)
            step = rng.choice([-1, 1]) * rng.uniform(0.25, 0.5)
            regs.append(replace(base, name=f"R{len(regs)}", rule=drift_rule(base.rule, step), kind="drift", noise=noise, version=base.version + 1))
            continue
        cue = _cue(rng, cues); cues.append(cue); cid = len(cues) - 1
        mu = torch.tensor([rng.gauss(0, 0.7) for _ in range(D)]); scale = rng.uniform(0.7, 1.3)
        if len(live) >= 2 and u < p_recur + p_drift + p_comp:
            a, b = rng.sample([r for r in live if r.rule.family not in OPS], 2) if len([r for r in live if r.rule.family not in OPS]) >= 2 else (live[0], live[-1])
            rule = Rule(rng.choice(OPS), (a.rule, b.rule)); kind = "composition"; mu = a.mu.clone(); scale = a.scale
            y = rule(mu + scale * torch.randn(2000, D, generator=torch.Generator().manual_seed(1)))
            if not 0.2 < float(y.float().mean()) < 0.8: rule = Rule("not", (a.rule,)) if rng.random() < 0.5 else a.rule; kind = "composition"
        elif rng.random() < p_novel / max(1e-9, 1 - (p_recur + p_drift + p_comp)):
            rule = random_rule(rng, rng.choice(hold_out), mu); kind = "new_domain"
        else:
            rule = random_rule(rng, rng.choice(families) if families else None, mu); kind = "new"
        regs.append(Regime(f"R{len(regs)}", rule, mu, scale, cue, kind, noise, cid))
    return regs


def superseded(regs: List[Regime]):
    """Indices of regimes whose rule was later replaced (same context, drift): not counted in retention."""
    last = {}
    for i, r in enumerate(regs): last[r.context_id] = i
    return {i for i, r in enumerate(regs) if last[r.context_id] != i and regs[last[r.context_id]].rule != r.rule}

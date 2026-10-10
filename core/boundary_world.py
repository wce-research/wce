"""Generator of the boundary induction BENCHMARK (EVALUATION side — the family names exist ONLY HERE).

The induction code (core/boundary_program.py and following) knows NONE of these families: it sees only
observations (x, y). "Held-out family" protocol:
  TRAINING families (development / anti-unification)           : halfspace, disc, strip, wedge
  VALIDATION families                                          : ring, two_lobe, oriented_rect
  HELD-OUT test families (never looked at during dev.)         : xor_halfspaces, crescent, capsule, wedge_ring
  SEALED families                                              : lens, l_shape, mirrored_crescent, strip_xor_disc
  partially symbolisable                                       : wiggle_halfspace (half-space + ripple)
  NOT representable (current depth)                            : sine5, checker, spiral, radial_wave
OBSERVED region: x0 ∈ [−2, 0.5], x1 ∈ [−2, 2]; EXTRAPOLATION region: x0 ∈ [0.5, 3], x1 ∈ [−3, 3].
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

TRAIN = ("halfspace", "disc", "strip", "wedge")
VALIDATION = ("ring", "two_lobe", "oriented_rect")
HELD_OUT = ("xor_halfspaces", "crescent", "capsule", "wedge_ring")
SEALED = ("lens", "l_shape", "mirrored_crescent", "strip_xor_disc")
PARTIAL = ("wiggle_halfspace",)
UNREPRESENTABLE = ("sine5", "checker", "spiral", "radial_wave")
REPRESENTABLE = TRAIN + VALIDATION + HELD_OUT + SEALED


def _u(rng, a, b): return rng.uniform(a, b)


def _dot(X, a): return math.cos(a) * X[:, 0] + math.sin(a) * X[:, 1]


def _disc(X, c, r): return (X - torch.tensor(c)).norm(dim=1) < r


@dataclass
class FamilyInstance:
    family: str
    params: Dict
    label: Callable
    representable: bool
    oracle: Optional[object] = None      # skeleton of the true program ("oracle" UPPER BOUND), never seen by the induction


def make_instance(family, rng: random.Random) -> FamilyInstance:
    """Instance of a family; the centres are near the observed / extrapolation boundary so that
    the extrapolation is NON-trivial."""
    from core.boundary_program import BNode
    P = lambda *a, **k: BNode(*a, **k)
    T = lambda child: P("translate", children=(child,))
    if family == "halfspace":
        a, b = _u(rng, -3.1, 3.1), _u(rng, -0.5, 0.5)
        return FamilyInstance(family, {"a": a, "b": b}, lambda X: (_dot(X, a) <= b).long(), True, P("pred", "dot", "le"))
    if family == "disc":
        c, r = (_u(rng, -0.5, 1.0), _u(rng, -1, 1)), _u(rng, 0.8, 1.5)
        return FamilyInstance(family, {"c": c, "r": r}, lambda X: _disc(X, c, r).long(), True, T(P("pred", "norm", "le")))
    if family == "strip":
        a, b, t = _u(rng, -3.1, 3.1), _u(rng, -0.4, 0.4), _u(rng, 0.3, 0.8)
        return FamilyInstance(family, {"a": a, "b": b, "t": t}, lambda X: ((_dot(X, a) - b).abs() <= t).long(), True, P("pred", "dot", "in"))
    if family == "wedge":
        c, th, w = (_u(rng, -1.5, 0.0), _u(rng, -0.5, 0.5)), _u(rng, -0.6, 0.6), _u(rng, 0.4, 0.9)
        def f(X, c=c, th=th, w=w):
            d = torch.atan2(X[:, 1] - c[1], X[:, 0] - c[0]) - th; return (torch.atan2(torch.sin(d), torch.cos(d)).abs() <= w).long()
        return FamilyInstance(family, {"c": c, "th": th, "w": w}, f, True, T(P("pred", "angle", "in")))
    if family == "ring":
        c, r1 = (_u(rng, -0.3, 0.8), _u(rng, -0.5, 0.5)), _u(rng, 0.5, 0.9); r2 = r1 + _u(rng, 0.4, 0.8)
        return FamilyInstance(family, {"c": c, "r1": r1, "r2": r2}, lambda X: (_disc(X, c, r2) & ~_disc(X, c, r1)).long(), True, T(P("pred", "norm", "in")))
    if family == "two_lobe":
        c1, c2 = (_u(rng, -1.6, -0.6), _u(rng, -1, 1)), (_u(rng, 0.6, 1.8), _u(rng, -1, 1)); r1, r2 = _u(rng, 0.5, 0.9), _u(rng, 0.5, 0.9)
        return FamilyInstance(family, {"c1": c1, "c2": c2}, lambda X: (_disc(X, c1, r1) | _disc(X, c2, r2)).long(), True,
                              P("or", children=(T(P("pred", "norm", "le")), T(P("pred", "norm", "le")))))
    if family == "oriented_rect":
        a, b1, b2, t1, t2 = _u(rng, -1.5, 1.5), _u(rng, 0.0, 0.8), _u(rng, -0.4, 0.4), _u(rng, 0.5, 1.0), _u(rng, 0.3, 0.7)
        return FamilyInstance(family, {"a": a}, lambda X: (((_dot(X, a) - b1).abs() <= t1) & ((_dot(X, a + math.pi / 2) - b2).abs() <= t2)).long(), True,
                              P("and", children=(P("pred", "dot", "in"), P("pred", "dot", "in"))))
    if family == "xor_halfspaces":
        a1, a2, b1, b2 = _u(rng, -3.1, 3.1), _u(rng, -3.1, 3.1), _u(rng, -0.4, 0.4), _u(rng, -0.4, 0.4)
        if abs(math.sin(a1 - a2)) < 0.4: a2 = a1 + 1.2
        return FamilyInstance(family, {}, lambda X: ((_dot(X, a1) <= b1) != (_dot(X, a2) <= b2)).long(), True,
                              P("xor", children=(P("pred", "dot", "le"), P("pred", "dot", "le"))))
    if family == "crescent":
        c, r = (_u(rng, -0.3, 0.6), _u(rng, -0.5, 0.5)), _u(rng, 1.0, 1.5); d = (c[0] + _u(rng, 0.4, 0.7), c[1] + _u(rng, -0.3, 0.3)); r2 = r * _u(rng, 0.7, 0.9)
        return FamilyInstance(family, {}, lambda X: (_disc(X, c, r) & ~_disc(X, d, r2)).long(), True,
                              P("and", children=(T(P("pred", "norm", "le")), T(P("pred", "norm", "ge")))))
    if family == "capsule":
        a, b, t = _u(rng, -0.8, 0.8), _u(rng, -0.3, 0.3), _u(rng, 0.3, 0.5); c1 = (-1.0, 0.0); c2 = (1.2, 0.0)
        def f(X, a=a, t=t):
            R = torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]); Z = X @ R
            seg = (Z[:, 1].abs() <= t) & (Z[:, 0] >= c1[0]) & (Z[:, 0] <= c2[0]); return (seg | _disc(Z, c1, t) | _disc(Z, c2, t)).long()
        return FamilyInstance(family, {}, f, True, None)                       # representable at depth 3 only (approx.)
    if family == "wedge_ring":
        c, r1 = (_u(rng, -0.6, 0.4), _u(rng, -0.4, 0.4)), _u(rng, 0.5, 0.8); r2 = r1 + _u(rng, 0.6, 1.0); th, w = _u(rng, -0.8, 0.8), _u(rng, 0.6, 1.2)
        def f(X, c=c, r1=r1, r2=r2, th=th, w=w):
            Z = X - torch.tensor(c); d = torch.atan2(Z[:, 1], Z[:, 0]) - th; n = Z.norm(dim=1)
            return ((n >= r1) & (n <= r2) & (torch.atan2(torch.sin(d), torch.cos(d)).abs() <= w)).long()
        return FamilyInstance(family, {}, f, True, T(P("and", children=(P("pred", "norm", "in"), P("pred", "angle", "in")))))
    if family == "lens":
        c1, c2 = (_u(rng, -0.6, 0.0), _u(rng, -0.4, 0.4)), (_u(rng, 0.2, 0.9), _u(rng, -0.4, 0.4)); r = _u(rng, 0.9, 1.3)
        return FamilyInstance(family, {}, lambda X: (_disc(X, c1, r) & _disc(X, c2, r)).long(), True,
                              P("and", children=(T(P("pred", "norm", "le")), T(P("pred", "norm", "le")))))
    if family == "l_shape":
        a = _u(rng, -0.5, 0.5); b1, b2 = _u(rng, -0.3, 0.3), _u(rng, -0.3, 0.3)
        return FamilyInstance(family, {}, lambda X: ((_dot(X, a) <= b1) & (_dot(X, a + math.pi / 2) <= b2)).long() | ((_dot(X, a) <= b1 - 1.0)).long(), True, None)
    if family == "mirrored_crescent":
        inst = make_instance("crescent", rng); return FamilyInstance(family, {}, lambda X: inst.label(X * torch.tensor([-1.0, 1.0])), True, inst.oracle)
    if family == "strip_xor_disc":
        s = make_instance("strip", rng); d = make_instance("disc", rng)
        return FamilyInstance(family, {}, lambda X: (s.label(X) != d.label(X)).long(), True, P("xor", children=(P("pred", "dot", "in"), T(P("pred", "norm", "le")))))
    if family == "wiggle_halfspace":
        a, b, amp = _u(rng, -3.1, 3.1), _u(rng, -0.3, 0.3), _u(rng, 0.12, 0.2)
        return FamilyInstance(family, {}, lambda X: (_dot(X, a) <= b + amp * torch.sin(4 * _dot(X, a + math.pi / 2))).long(), False, P("pred", "dot", "le"))
    if family == "sine5":
        ph = _u(rng, 0, 6.28); return FamilyInstance(family, {}, lambda X: (X[:, 1] > 0.8 * torch.sin(5 * X[:, 0] + ph)).long(), False)
    if family == "checker":
        s = _u(rng, 1.3, 1.8); return FamilyInstance(family, {}, lambda X: ((torch.floor(X[:, 0] * s) + torch.floor(X[:, 1] * s)) % 2).long(), False)
    if family == "spiral":
        k = _u(rng, 1.5, 2.5)
        def f(X, k=k):
            r = X.norm(dim=1); a = torch.atan2(X[:, 1], X[:, 0]); return (torch.sin(a + k * r) > 0).long()
        return FamilyInstance(family, {}, f, False)
    if family == "radial_wave":
        def f(X):
            r = X.norm(dim=1); a = torch.atan2(X[:, 1], X[:, 0]); return (r < 1.2 + 0.4 * torch.sin(6 * a)).long()
        return FamilyInstance(family, {}, f, False)
    raise KeyError(family)


def observed(n, g, label_noise=0.03, x_noise=0.02):
    X = torch.stack([torch.rand(n, generator=g) * 2.5 - 2.0, torch.rand(n, generator=g) * 4 - 2], 1)
    return X, label_noise, x_noise


def sample_task(inst: FamilyInstance, n, g, label_noise=0.03, x_noise=0.02):
    """Observations in the observed region: noisy x, noisy labels."""
    X = torch.stack([torch.rand(n, generator=g) * 2.5 - 2.0, torch.rand(n, generator=g) * 4 - 2], 1)
    y = inst.label(X); flip = torch.rand(n, generator=g) < label_noise
    return X + x_noise * torch.randn(X.shape, generator=g), torch.where(flip, 1 - y, y)


def in_region(n, g): return torch.stack([torch.rand(n, generator=g) * 2.5 - 2.0, torch.rand(n, generator=g) * 4 - 2], 1)
def out_region(n, g): return torch.stack([torch.rand(n, generator=g) * 2.5 + 0.5, torch.rand(n, generator=g) * 6 - 3], 1)
def full_region(n, g): return torch.rand(n, 2, generator=g) * 6 - 3


def balanced(inst, g, tries=20):
    for _ in range(tries):
        y = inst.label(in_region(800, g)).float().mean()
        if 0.08 < float(y) < 0.92: return True
    return False


def instances(families, seed, per_family):
    rng = random.Random(seed); g = torch.Generator().manual_seed(seed); out = []
    for f in families:
        k = 0
        while k < per_family:
            inst = make_instance(f, rng)
            if balanced(inst, g): out.append(inst); k += 1
    return out


def sealed_spec(): return {"families": list(SEALED) + ["wiggle_halfspace", "radial_wave", "spiral"], "seed": 990777, "per_family": 3}


def sealed_hash():
    s = sealed_spec(); g = torch.Generator().manual_seed(1)
    probe = full_region(64, g)
    sig = [[f.family, inst_sig(f, probe)] for f in instances(s["families"], s["seed"], s["per_family"])]
    return hashlib.sha256(json.dumps(sig).encode()).hexdigest()


def inst_sig(inst, probe): return "".join(str(int(v)) for v in inst.label(probe).tolist())


SEALED_SHA256 = "e163f5b01588c301486ed95545dc48bb95e298b214e1cbe56bb1a1cd777ef42d"   # frozen BEFORE any evaluation


def composed_instance(rng: random.Random) -> FamilyInstance:
    """NOVEL COMPOSITION of known families: rotation(A op [NOT] B), op ∈ {OR, AND, XOR} — never seen as such."""
    a = make_instance(rng.choice(TRAIN), rng); b = make_instance(rng.choice(TRAIN), rng); op = rng.choice(["or", "and", "xor"]); neg = rng.random() < 0.5
    phi = rng.uniform(-1.0, 1.0); c, s = math.cos(phi), math.sin(phi)
    def f(X):
        Z = torch.stack([c * X[:, 0] + s * X[:, 1], -s * X[:, 0] + c * X[:, 1]], 1); A = a.label(Z).bool(); B = b.label(Z).bool()
        if neg: B = ~B
        return (A | B if op == "or" else A & B if op == "and" else A ^ B).long()
    return FamilyInstance(f"composed:{a.family}-{op}{'-not' if neg else ''}-{b.family}", {}, f, True, None)


def double_held_out(family, rng: random.Random, shift=(1.6, 0.0), scale=1.6) -> FamilyInstance:
    """HELD-OUT structure + OUT-of-range parameters: instance dilated and shifted towards the extrapolation region."""
    inst = make_instance(family, rng); sx, sy = shift
    return FamilyInstance(family + ":double", {}, lambda X: inst.label((X - torch.tensor([sx, sy])) / scale), True, inst.oracle)


def balanced_inst(maker, g, rng, tries=30):
    for _ in range(tries):
        inst = maker(rng)
        if 0.08 < float(inst.label(in_region(800, g)).float().mean()) < 0.92: return inst
    return inst

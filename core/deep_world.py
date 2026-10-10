"""Evaluation world of the "Deep Compositional Structure Discovery & Model-Class Boundary" milestone.

EVALUATION side only: the inference code (core/deep_structure, deep_fit, deep_search, deep_boundary) never imports
this module. Independent of the world of the previous milestone (whose sealed set is consumed): new seeds,
new splits, new sealed set.

TRUE structures: composition trees of compositional depth d = number of atoms ∈ {1..6}, of controlled
size — each atom is EFFECTIVE (removing it changes the function on ≥ MIN_EFFECT of the observed region), otherwise the
announced depth would not be real.

Splits (disjoint seeds): dev (tuning), validation (selection, calibration), confirmatory test (16
seeds), sealed set (frozen before any tuning, evaluated only once). The "unseen" global compositions (reserved
signature) never appear in dev / validation.

OUT-OF-CLASS worlds: absent primitive, out-of-budget composition, absent relation, incompatible dynamics, distribution
shift, partially representable. HARD BUT VALID worlds: extreme noise, high depth, few
data.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from core.boundary_program import BNode, canonical, predict
from core.robust_world import SENSORS, Sensor

DEPTHS = (1, 2, 3, 4, 5, 6)
MIN_EFFECT = 0.04                     # minimum share of the observed region affected by each atom
OBS_XMAX = 1.2                        # passive observations cover x ≤ OBS_XMAX; beyond = extrapolation
REGIMES = {"rich": (600, 0.02), "standard": (300, 0.05), "hard": (160, 0.10)}      # (passive observations, world noise)
NOISE_KINDS = ("iid", "boundary", "grouped_outliers")
OOC_CATEGORIES = ("primitive_absent", "composition_absent", "relation_absent", "dynamic_incompatible", "distribution_shift", "partial")
HARD_VALID = ("extreme_noise", "deep", "few_data")
SPLIT_BASE = {"dev": 1_000_000, "validation": 2_000_000, "sealed": 3_000_000, "test": 4_000_000, "sealed_v2": 0}   # sealed_v2: seeds derived from a private salt (core.deep_sealed)
UNSEEN_MOD = 4                        # 1 global signature out of 4 (depth ≥ 3) is reserved for unseen compositions


# ================================================================ regions

def full_region(n, g): return torch.rand(n, 2, generator=g) * 6 - 3
def obs_region(n, g): return torch.stack([torch.rand(n, generator=g) * (OBS_XMAX + 3) - 3, torch.rand(n, generator=g) * 6 - 3], 1)
def out_region(n, g): return torch.stack([torch.rand(n, generator=g) * (3 - OBS_XMAX) + OBS_XMAX, torch.rand(n, generator=g) * 6 - 3], 1)


# ================================================================ true structures

ATOM_TYPES = ("half", "band", "disc", "antidisc", "ring", "sector")


def atom(rng: random.Random, kind: str):
    c = (rng.uniform(-2.0, 0.6), rng.uniform(-1.8, 1.8)); T = lambda ch: BNode("translate", params=c, children=(ch,))
    if kind == "half": a = rng.uniform(-math.pi, math.pi); return BNode("pred", "dot", "le", (a, math.cos(a) * c[0] + math.sin(a) * c[1]))
    if kind == "band":
        a = rng.uniform(-math.pi, math.pi); b = math.cos(a) * c[0] + math.sin(a) * c[1]; t = rng.uniform(0.4, 0.9); return BNode("pred", "dot", "in", (a, b - t, b + t))
    if kind == "disc": return T(BNode("pred", "norm", "le", (rng.uniform(0.8, 1.6),)))
    if kind == "antidisc": return T(BNode("pred", "norm", "ge", (rng.uniform(0.7, 1.2),)))
    if kind == "ring": r = rng.uniform(0.5, 0.9); return T(BNode("pred", "norm", "in", (r, r + rng.uniform(0.6, 1.0))))
    if kind == "sector": return T(BNode("pred", "angle", "in", (rng.uniform(-math.pi, math.pi), rng.uniform(0.5, 1.2))))
    raise KeyError(kind)


def random_tree(rng: random.Random, depth: int):
    """Composition tree with `depth` atoms: varied shapes (chain or balanced), combinators and / or / xor."""
    if depth == 1: return atom(rng, rng.choice(ATOM_TYPES))
    k = rng.randint(1, depth - 1) if rng.random() < 0.5 else 1
    return BNode(rng.choice(("and", "or", "xor")), children=(random_tree(rng, depth - k), random_tree(rng, k)))


def n_atoms(p: BNode): return sum(n_atoms(c) for c in p.children) if p.kind in ("and", "or", "xor") else 1


def _drop_variants(p: BNode):
    """Programs obtained by removing ONE atom (replacement of the parent combinator by the other child)."""
    out = []
    if p.kind in ("and", "or", "xor"):
        a, b = p.children; out += [a, b]
        out += [BNode(p.kind, children=(v, b)) for v in _drop_variants(a)] + [BNode(p.kind, children=(a, v)) for v in _drop_variants(b)]
    return out


def min_effect(p: BNode, probe):
    """Smallest share of the observed region modified by the removal of one atom (1.0 for a single atom)."""
    base = predict(p, probe); vs = _drop_variants(p)
    return min((float((predict(v, probe) != base).float().mean()) for v in vs), default=1.0)


def signature(p: BNode):
    """GLOBAL composition signature (canonical skeleton, without parameters)."""
    return hashlib.sha1(json.dumps(canonical(p).skeleton().to_dict(), sort_keys=True).encode()).hexdigest()


def is_reserved(p: BNode): return n_atoms(p) >= 3 and int(signature(p)[:8], 16) % UNSEEN_MOD == 0


def sub_signatures(p: BNode):
    """Signatures of the 2-atom subtrees (substructures which, for their part, are seen in development)."""
    out = set()
    def walk(n):
        if n.kind in ("and", "or", "xor"):
            if n_atoms(n) == 2: out.add(signature(n))
            for c in n.children: walk(c)
    walk(p); return out


def make_structure(seed: int, depth: int, unseen: Optional[bool] = None):
    """True structure of depth `depth`, balanced, with effective atoms. unseen=True: reserved signature;
    False: non-reserved signature; None: either."""
    rng = random.Random(seed); probe = obs_region(2000, torch.Generator().manual_seed(seed)); last = None
    for _ in range(4000):
        p = random_tree(rng, depth); m = float(predict(p, probe).float().mean())
        if not 0.15 < m < 0.85 or min_effect(p, probe) < MIN_EFFECT: continue
        last = p
        if unseen is None or depth < 3 or is_reserved(p) == unseen: return p
    if last is None: raise RuntimeError(f"structure not found (seed={seed}, depth={depth})")
    return last


# ================================================================ out-of-class worlds (non-representable functions)

def _ooc(rng: random.Random, category: str, seed: int):
    """Returns (label(X, t), underlying representable program | None)."""
    ph = rng.uniform(0, 6.28)
    if category == "primitive_absent":
        k = rng.randrange(3)
        if k == 0: f = rng.uniform(1.8, 2.6); return (lambda X, t=0.0: (X[:, 1] > 0.9 * torch.sin(f * X[:, 0] + ph)).long()), None
        if k == 1: a, c = rng.uniform(0.25, 0.5), rng.uniform(-1.5, 0.0); return (lambda X, t=0.0: (X[:, 1] > a * (X[:, 0] + 0.8) ** 2 + c).long()), None
        ax, ay = rng.uniform(2.0, 2.6), rng.uniform(0.6, 0.9); cx = rng.uniform(-1.4, -0.4)
        return (lambda X, t=0.0: ((((X[:, 0] - cx) / ax) ** 2 + (X[:, 1] / ay) ** 2) <= 1).long()), None
    if category == "composition_absent":                                   # union of 9 small discs: outside the depth budget
        cs = torch.tensor([[rng.uniform(-2.6, 1.0), rng.uniform(-2.6, 2.6)] for _ in range(9)]); r = rng.uniform(0.55, 0.7)
        return (lambda X, t=0.0: (torch.cdist(X, cs).min(1).values <= r).long()), None
    if category == "relation_absent":
        if rng.random() < 0.5: c = rng.uniform(0.3, 0.8); return (lambda X, t=0.0: ((X[:, 0] + 0.9) * X[:, 1] > c).long()), None
        s = rng.uniform(1.0, 1.4); return (lambda X, t=0.0: ((torch.floor((X[:, 0] + 3) / s) + torch.floor((X[:, 1] + 3) / s)) % 2).long()), None
    if category == "dynamic_incompatible":                                 # boundary that drifts with measurement time
        base = make_structure(seed + 17, 1); v = rng.uniform(0.9, 1.5) * rng.choice((-1, 1)); a = rng.uniform(-math.pi, math.pi)
        def lab(X, t=0.0):
            tt = torch.as_tensor(t, dtype=X.dtype).expand(len(X)); sh = torch.stack([math.cos(a) * v * tt, math.sin(a) * v * tt], 1)
            return predict(base, X - sh)
        return lab, base
    if category == "distribution_shift":                                   # representable in the observed zone, another law beyond
        p1 = make_structure(seed + 23, rng.choice((1, 2))); f = rng.uniform(1.8, 2.4)
        return (lambda X, t=0.0: torch.where(X[:, 0] <= OBS_XMAX, predict(p1, X), (X[:, 1] > torch.sin(f * X[:, 0] + ph)).long())), p1
    if category == "partial":                                              # representable structure XOR local non-representable pattern
        p1 = make_structure(seed + 29, rng.choice((1, 2))); c = torch.tensor([rng.uniform(-1.8, 0.2), rng.uniform(-1.5, 1.5)]); f = rng.uniform(3.0, 4.0)
        def lab(X, t=0.0):
            patch = ((X - c).norm(dim=1) < 1.3) & (torch.sin(f * X[:, 0] + ph) * torch.sin(f * X[:, 1]) > 0.15)
            return predict(p1, X) ^ patch.long()
        return lab, p1
    raise KeyError(category)


# ================================================================ instance, measurement process

@dataclass
class DeepWorld:
    uid: str                                   # OPAQUE identifier
    label: Callable                            # label(X, t) — truth (evaluation only)
    truth: Optional[BNode]                     # true program if the world is IN the class
    depth: int                                 # number of true atoms (0 out of class)
    in_class: bool
    category: str                              # "in_class" | out-of-class category | hard-valid category
    unseen: bool                               # reserved global composition (never seen in dev / validation)
    partial_truth: Optional[BNode] = None      # representable component of an out-of-class world (diagnostic)


@dataclass
class NoiseSpec:
    rate: float = 0.05
    kind: str = "iid"


def corrupt(world: DeepWorld, X, t, sensor: Sensor, spec: NoiseSpec, g):
    """Noisy measurement: i.i.d. flips (sensor + world), blur near the boundaries, grouped outliers."""
    y = world.label(X, t).clone(); n = len(X); base = torch.full((n,), sensor.flip + spec.rate)
    if sensor.blur > 0 or spec.kind == "boundary":
        d = torch.randn(X.shape, generator=g) * (sensor.blur + (0.12 if spec.kind == "boundary" else 0.0))
        near = (world.label(X + d, t) != y).float(); base = base * (0.5 if spec.kind == "boundary" else 1.0) + near * (0.3 if spec.kind == "boundary" else 0.2)
    flip = torch.rand(n, generator=g) < base.clamp(0, 0.49)
    if spec.kind == "grouped_outliers" and n >= 10:
        c = X[int(torch.randint(0, n, (1,), generator=g))]; flip = flip | (((X - c).norm(dim=1) < 0.3) & (torch.rand(n, generator=g) < 0.85))
    return torch.where(flip, 1 - y, y)


@dataclass
class DeepTask:
    world: DeepWorld
    X: torch.Tensor                            # passive observations (observed region)
    y: torch.Tensor
    t: torch.Tensor                            # observation time ∈ [0, 1]
    spec: NoiseSpec
    regime: str
    seed: int
    split: str
    clock: float = 1.0                         # current time: each active measurement advances it

    def oracle(self, x, sensor="standard", g=None):
        """ACTIVE measurement (never the truth): goes through the noise process, at the current instant."""
        self.clock += 0.02 * len(x)
        return corrupt(self.world, x, self.clock, SENSORS[sensor], self.spec, g)


def make_world(seed: int, depth: Optional[int] = None, category: str = "in_class", unseen: Optional[bool] = None) -> DeepWorld:
    rng = random.Random(seed); uid = hashlib.sha1(f"deep-{seed}-{depth}-{category}".encode()).hexdigest()[:12]
    if category in ("in_class",) + HARD_VALID:
        d = depth or {"extreme_noise": rng.choice((1, 2)), "deep": rng.choice((5, 6)), "few_data": rng.choice((2, 3))}.get(category, rng.choice(DEPTHS))
        p = make_structure(seed, d, unseen)
        return DeepWorld(uid, (lambda X, t=0.0, p=p: predict(p, X)), p, d, True, category, is_reserved(p))
    probe = obs_region(2000, torch.Generator().manual_seed(seed))
    for k in range(60):
        lab, part = _ooc(random.Random(seed * 131 + k), category, seed * 131 + k)
        if 0.12 < float(lab(probe, 0.5).float().mean()) < 0.88: break
    return DeepWorld(uid, lab, None, 0, False, category, False, part)


def make_task(seed: int, depth: Optional[int] = None, regime: str = "standard", category: str = "in_class", kind: str = "iid",
              unseen: Optional[bool] = None, split: str = "dev") -> DeepTask:
    """Task deterministic in its arguments. `seed` is an index LOCAL to the split (actual seed = base + seed)."""
    if split in ("dev", "validation") and unseen is None: unseen = False            # no reserved composition in dev / validation
    s = SPLIT_BASE[split] + seed; w = make_world(s, depth, category, unseen); n, rate = REGIMES[regime]
    if category == "extreme_noise": rate = 0.25
    if category == "few_data": n = 80
    g = torch.Generator().manual_seed(s + 1); X = obs_region(n, g); t = torch.rand(n, generator=g)
    spec = NoiseSpec(rate, kind); y = corrupt(w, X, t, SENSORS["standard"], spec, g)
    return DeepTask(w, X, y, t, spec, regime, s, split)


# ================================================================ sealed set (frozen before any tuning)
# RETIRED BEFORE EVALUATION — NEVER CONSUMED. This set is generated from seeds written in this file, so publishing
# the repository publishes its answers. It was never evaluated and must never be used for a scientific decision. The
# functions and the fingerprint below are kept for historical integrity only. The live sealed set is generated from
# a private salt: see core/deep_sealed.py.
SEALED_STATUS = "RETIRED BEFORE EVALUATION — NEVER CONSUMED"

def sealed_spec():
    """72 tasks: 36 in class (depths 1–6; unseen compositions for d ≥ 3 on half of them), 12 hard
    but valid, 24 out of class (6 categories × 4). Variable regimes and noises."""
    spec = []; i = 0
    for d in DEPTHS:
        for j in range(6):
            spec.append({"seed": i, "depth": d, "category": "in_class", "regime": ("standard", "rich", "hard")[j % 3], "kind": NOISE_KINDS[j % 3], "unseen": (j % 2 == 0) if d >= 3 else None}); i += 1
    for c in HARD_VALID:
        for j in range(4): spec.append({"seed": i, "depth": None, "category": c, "regime": ("standard", "rich")[j % 2], "kind": NOISE_KINDS[j % 3], "unseen": None}); i += 1
    for c in OOC_CATEGORIES:
        for j in range(4): spec.append({"seed": i, "depth": None, "category": c, "regime": ("standard", "rich")[j % 2], "kind": NOISE_KINDS[j % 3], "unseen": None}); i += 1
    return spec


def sealed_task(index: int) -> DeepTask:
    s = sealed_spec()[index]; return make_task(s["seed"], s["depth"], s["regime"], s["category"], s["kind"], s["unseen"], split="sealed")


def sealed_hash():
    """Fingerprint of the sealed set: identifiers, true labels on fixed probes, first observations. Computed
    without ever exposing the content to the inference code."""
    probe = full_region(96, torch.Generator().manual_seed(7)); sig = []
    for i, s in enumerate(sealed_spec()):
        t = sealed_task(i)
        sig.append([t.world.uid, "".join(str(int(v)) for v in t.world.label(probe, 0.5).tolist()), t.regime, t.spec.kind, round(t.spec.rate, 3), int(t.y.sum()), len(t.y)])
    return hashlib.sha256(json.dumps(sig).encode()).hexdigest()


SEALED_SHA256 = "d57e9d761d165ef9d86ec7ee5c9c5c7909c4b1622425c2d2379cbdddaf1a6ca2"   # frozen on 2026-10-04, BEFORE any tuning and any experiment

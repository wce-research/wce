"""EMPIRICAL stability of invariants under perturbation.

KAM-inspired stability: an invariant is "persistent" in a region if a large fraction of small
perturbations preserves it. It is a statistical and empirical notion: NO Hamiltonian system, no invariant
torus, no Diophantine condition is constructed or claimed. Likewise, a "resonance" here is a
NON-ADDITIVE INTERACTION between perturbations (StructuralResonanceCandidate), not a physical KAM resonance.
"""
from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from core.model_state import PARAM_NAMES, ModelState
from core.perturbation import Perturbation, unit_direction


# ================================================================ persistence of an invariant

@dataclass
class InvariantPersistence:
    """Survival probability of an invariant under random perturbations of a given amplitude (+ Wilson CI)."""
    invariant: str
    magnitude: float
    survived: int
    trials: int

    @property
    def probability(self): return self.survived / max(1, self.trials)

    @property
    def interval(self):
        n, p, z = self.trials, self.probability, 1.96
        if n == 0: return (0.0, 1.0)
        d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (max(0.0, c - h), min(1.0, c + h))


def persistence(holds: Callable[[float, int], bool], magnitude, trials=32, name="I"):
    """holds(magnitude, seed) -> does the invariant hold under the perturbation drawn with this seed?"""
    ok = sum(bool(holds(magnitude, s)) for s in range(trials))
    return InvariantPersistence(name, magnitude, ok, trials)


@dataclass
class StableRegion:
    """Region (amplitudes ≤ radius) where the survival of the invariant stays ≥ threshold (empirically)."""
    invariant: str
    radius: float
    threshold: float
    curve: List[InvariantPersistence] = field(default_factory=list)

    def contains(self, magnitude): return magnitude <= self.radius


def stability_radius(holds, threshold=0.9, lo=0.0, hi=1.0, trials=32, iters=8, grid=None, name="I"):
    """r_I = maximum amplitude such that P(invariant preserved) ≥ threshold. Geometric sweep then bisection
    (assumes a roughly decreasing survival; the full curve is kept for verification)."""
    grid = grid or [hi * 2 ** -k for k in range(10, -1, -1)]
    curve = [persistence(holds, m, trials, name) for m in grid]
    good = [c.magnitude for c in curve if c.probability >= threshold]
    if not good: return StableRegion(name, 0.0, threshold, curve)
    a = max(good); bigger = [c.magnitude for c in curve if c.magnitude > a]
    if not bigger: return StableRegion(name, a, threshold, curve)
    b = min(bigger)
    for _ in range(iters):
        mid = 0.5 * (a + b); c = persistence(holds, mid, trials, name); curve.append(c)
        if c.probability >= threshold: a = mid
        else: b = mid
    curve.sort(key=lambda c: c.magnitude)
    return StableRegion(name, a, threshold, curve)


# ================================================================ neural model: weight perturbations

def perturbed_model(model: ModelState, modules: Sequence[str], magnitude, seed, keys=PARAM_NAMES):
    """RELATIVE noise: W + magnitude·||W||·u (u random unit direction), on the designated modules."""
    m = copy.deepcopy(model)
    for j, mod in enumerate(modules):
        for i, k in enumerate(keys):
            W = m.modules[mod].slow[k]
            m.modules[mod].slow[k] = W + magnitude * float(W.norm().clamp_min(1e-6)) * unit_direction(W.shape, seed * 1000 + 31 * j + i)
    return m


def accuracy_invariant(model, x, y, modules, floor):
    """Invariant "accuracy of the old task ≥ floor" under perturbation of the given modules."""
    def holds(mag, seed): return perturbed_model(model, modules, mag, seed).accuracy(x, y) >= floor
    return holds


def stability_map(model, x, y, floor, modules=("enc", "expert_a", "head"), grid=(0.01, 0.02, 0.05, 0.1, 0.2, 0.4), trials=16):
    """Map: (parameter region = module) × amplitude → survival probability of the invariant."""
    return {mod: {m: persistence(accuracy_invariant(model, x, y, (mod,), floor), m, trials, mod).probability for m in grid}
            for mod in modules}


# ================================================================ non-additive interactions (structural "resonances")

@dataclass
class StructuralResonanceCandidate:
    """Pair / triple of perturbations whose joint effect deviates strongly from the sum of the individual effects.
    (Name: analogy; it is not a physical KAM resonance.)"""
    components: Tuple[str, ...]
    interaction: float
    joint: float
    singles: Tuple[float, ...]

    @property
    def relative(self): return abs(self.interaction) / max(1e-9, abs(self.joint) + sum(abs(s) for s in self.singles))


def interaction(response: Callable[[Tuple[str, ...]], float], combo: Tuple[str, ...]):
    """interaction = response(δ1+…+δk) − Σ response(δi) (effect of second order and above)."""
    singles = tuple(response((c,)) for c in combo); joint = response(combo)
    return StructuralResonanceCandidate(combo, joint - sum(singles), joint, singles)


class PerturbationInteractionGraph:
    """Nodes: components / transformations; edges: interaction strength. Dangerous pairs and triples."""

    def __init__(self, components: Sequence[str], response: Callable[[Tuple[str, ...]], float], triples=True):
        self.components = list(components); self.response = response
        self._cache: Dict[Tuple[str, ...], float] = {}
        def cached(combo):
            key = tuple(sorted(combo))
            if key not in self._cache: self._cache[key] = response(key)
            return self._cache[key]
        self.singles = {c: cached((c,)) for c in self.components}
        self.edges: Dict[Tuple[str, str], StructuralResonanceCandidate] = {}
        for a, b in itertools.combinations(self.components, 2):
            self.edges[(a, b)] = interaction(cached, (a, b))
        self.triples: Dict[Tuple[str, ...], StructuralResonanceCandidate] = {}
        if triples:
            for t in itertools.combinations(self.components, 3):
                j = cached(t); pair_sum = sum(self.edges[p].interaction for p in itertools.combinations(t, 2))
                s = tuple(self.singles[c] for c in t)
                self.triples[t] = StructuralResonanceCandidate(t, j - sum(s) - pair_sum, j, s)   # pure order-3 interaction

    def dangerous(self, threshold=0.1, relative=0.25):
        out = [e for e in list(self.edges.values()) + list(self.triples.values())
               if abs(e.interaction) >= threshold and e.relative >= relative]
        return sorted(out, key=lambda e: -abs(e.interaction))


# ================================================================ bifurcation-like transitions

@dataclass
class TransitionBoundary:
    """Threshold where a discrete quantity changes (routing, branch, invariant) or drops abruptly (performance)."""
    quantity: str
    threshold: float
    before: object
    after: object
    kind: str


def detect_transitions(param_values: Sequence[float], observe: Callable[[float], Dict[str, object]], jump=0.15, refine=6):
    """Sweeps a parameter; detects discrete changes and drops ≥ jump; refines each threshold by bisection."""
    obs = [observe(v) for v in param_values]; out = []
    for i in range(1, len(param_values)):
        for q in obs[i]:
            a, b = obs[i - 1][q], obs[i][q]
            numeric = isinstance(a, (int, float)) and not isinstance(a, bool)
            if (numeric and abs(a - b) >= jump) or (not numeric and a != b):
                lo, hi = param_values[i - 1], param_values[i]
                for _ in range(refine):
                    mid = 0.5 * (lo + hi); m = observe(mid)[q]
                    changed = (abs(m - a) >= jump / 2) if numeric else (m != a)
                    if changed: hi = mid
                    else: lo = mid
                out.append(TransitionBoundary(q, round(0.5 * (lo + hi), 6), a, b, "drop" if numeric else "switch"))
    return out


def routing_transition_observer(model, x, y, src="expert_b", tgt="head"):
    """Increasing strength of a parasitic routing: dominant decision, computation path, accuracy."""
    from core.perturbation import Perturbation
    def observe(s):
        m = Perturbation("routing", (src, tgt), s).apply(model)
        acc = m.accuracy(x, y)
        return {"accuracy": round(acc, 4), "branch": tuple(m.topological_order()), "invariant_acc>=0.95": acc >= 0.95}
    return observe


# ================================================================ Fourier: which invariants are stable in which basis

def fourier_sensitivity(n=128, trials=40, eps=0.05, seed=0):
    """Perturbation localised in TIME (impulse) vs localised in FREQUENCY (one line); invariants:
    dominant frequency, set of lines, local envelope (window), energy (Parseval). Survival rate."""
    from core.measurement import BasisTransform
    from core.measurement_experiments import make_signal, spectral_reasoning
    import random
    bt = BasisTransform(n); rng = random.Random(seed); res = {"time_perturbation": {}, "frequency_perturbation": {}}
    acc = {k: {"dominant": 0, "line_set": 0, "local_window": 0, "energy_rel_change": 0.0} for k in res}
    for t in range(trials):
        x, truth, dom = make_signal(rng, n); s0, d0 = spectral_reasoning(x, bt)
        amp = eps * float(x.norm())
        dt = torch.zeros(n, dtype=x.dtype); dt[rng.randrange(n)] = amp                       # time impulse
        f = rng.randrange(1, n // 2); tt = torch.arange(n, dtype=x.dtype)
        df = amp * math.sqrt(2.0 / n) * torch.cos(2 * math.pi * f * tt / n + rng.random() * 6.28)   # one line
        w = slice(0, 16)
        for key, d in (("time_perturbation", dt), ("frequency_perturbation", df)):
            y = x + d; s1, d1 = spectral_reasoning(y, bt)
            acc[key]["dominant"] += d1 == d0; acc[key]["line_set"] += s1 == s0
            acc[key]["local_window"] += bool(torch.allclose(y[w], x[w], atol=1e-9))
            acc[key]["energy_rel_change"] += abs(bt.energy(y) - bt.energy(x)) / bt.energy(x)
    for key in acc:
        res[key] = {k: round(v / trials, 3) for k, v in acc[key].items()}
    return res


# ================================================================ reference frames: true invariants vs coordinates

def frame_perturbation(trials=100, seed=0, eps=0.3):
    """Perturbs the reference frame (random rotation + translation): distance and angle must stay stable,
    the coordinates not (they are not invariants)."""
    from core.measurement import FramedValue, ReferenceFrame, distance
    g = torch.Generator().manual_seed(seed); out = {"distance": 0, "angle": 0, "coordinates": 0}
    for _ in range(trials):
        P = torch.randn(3, 2, generator=g, dtype=torch.float64)
        F = ReferenceFrame("F", tuple((eps * torch.randn(2, generator=g)).tolist()), float(eps * torch.randn(1, generator=g)))
        Q = [F.from_world(FramedValue(p, "world")) for p in P]
        W = [FramedValue(p, "world") for p in P]
        out["distance"] += abs(distance(Q[0], Q[1]) - distance(W[0], W[1])) < 1e-9
        ang = lambda A: math.atan2(*(lambda u, v: (float(u[0] * v[1] - u[1] * v[0]), float((u * v).sum())))(A[1].value - A[0].value, A[2].value - A[0].value))
        out["angle"] += abs(ang(Q) - ang(W)) < 1e-9
        out["coordinates"] += bool(torch.allclose(Q[0].value, W[0].value))
    return {k: v / trials for k, v in out.items()}


# ================================================================ algebra: perturbation of a chain of transformations

def algebra_chain_perturbation(trials=200, seed=0, chain_len=4):
    """Chain of admissible transformations; one link is replaced at random by a transformation from the catalogue.
    Prediction by the stabiliser (is the link in Stabilizer(solution_set)?) vs truth (recomputed solution
    set). Returns the accuracy of the prediction and the break rate."""
    import random
    from core.invariant_algebra import ALG_INVARIANTS, algebra_samples, algebra_transformations, build_index, solution_set
    rng = random.Random(seed); ix = build_index("algebra"); stab = ix.stabilizer("solution_set")
    G = algebra_transformations(values=(2, 3, -1, 5, -4)); byname = {t.name: t for t in G}
    admissible = [t for t in G if t.name in stab]
    correct = broke = 0; ood = 0
    for _ in range(trials):
        eq = rng.choice(algebra_samples()); chain = [rng.choice(admissible) for _ in range(chain_len)]
        i = rng.randrange(chain_len); repl = rng.choice(G); chain[i] = repl
        pred = all(t.name in stab for t in chain) if all(t.name in byname for t in chain) else None
        known = repl.name in {t.name for t in ix.transformations}
        if not known:                                   # transformation outside the index (parameter never seen)
            ood += 1; pred = _generalize(repl.name, stab)
        y = eq
        for t in chain: y = t(y)
        truth = solution_set(y) == solution_set(eq)
        correct += pred == truth; broke += not truth
    return {"prediction_accuracy": round(correct / trials, 3), "break_rate": round(broke / trials, 3), "ood_links": ood}


def _generalize(name, stab):
    """Generalisation by transformation FAMILY (name without parameter): admissible if the family is and the
    parameter is not 0 (only learned exception: MultiplyBothSides(0))."""
    fam = name.split("(")[0]; arg = name[len(fam):]
    fam_ok = any(t.split("(")[0] == fam for t in stab.transformations)
    return fam_ok and arg not in ("(0)", "(x)")

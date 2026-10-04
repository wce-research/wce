"""Stabilité EMPIRIQUE des invariants sous perturbation.

KAM-inspired stability : un invariant est « persistant » dans une région si une forte fraction de petites
perturbations le conserve. C'est une notion statistique et empirique : AUCUN système hamiltonien, aucun tore
invariant, aucune condition diophantienne n'est construit ni revendiqué. De même, une « résonance » ici est une
INTERACTION NON ADDITIVE entre perturbations (StructuralResonanceCandidate), pas une résonance KAM physique.
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


# ================================================================ persistance d'un invariant

@dataclass
class InvariantPersistence:
    """Probabilité de survie d'un invariant sous des perturbations aléatoires d'amplitude donnée (+ IC de Wilson)."""
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
    """holds(magnitude, seed) -> l'invariant tient-il sous la perturbation tirée avec cette graine ?"""
    ok = sum(bool(holds(magnitude, s)) for s in range(trials))
    return InvariantPersistence(name, magnitude, ok, trials)


@dataclass
class StableRegion:
    """Région (amplitudes ≤ radius) où la survie de l'invariant reste ≥ threshold (empiriquement)."""
    invariant: str
    radius: float
    threshold: float
    curve: List[InvariantPersistence] = field(default_factory=list)

    def contains(self, magnitude): return magnitude <= self.radius


def stability_radius(holds, threshold=0.9, lo=0.0, hi=1.0, trials=32, iters=8, grid=None, name="I"):
    """r_I = amplitude maximale telle que P(invariant conservé) ≥ seuil. Balayage géométrique puis bissection
    (suppose une survie à peu près décroissante ; la courbe complète est conservée pour vérification)."""
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


# ================================================================ modèle neuronal : perturbations de poids

def perturbed_model(model: ModelState, modules: Sequence[str], magnitude, seed, keys=PARAM_NAMES):
    """Bruit RELATIF : W + magnitude·||W||·u (u direction unitaire aléatoire), sur les modules désignés."""
    m = copy.deepcopy(model)
    for j, mod in enumerate(modules):
        for i, k in enumerate(keys):
            W = m.modules[mod].slow[k]
            m.modules[mod].slow[k] = W + magnitude * float(W.norm().clamp_min(1e-6)) * unit_direction(W.shape, seed * 1000 + 31 * j + i)
    return m


def accuracy_invariant(model, x, y, modules, floor):
    """Invariant « exactitude de l'ancienne tâche ≥ floor » sous perturbation des modules donnés."""
    def holds(mag, seed): return perturbed_model(model, modules, mag, seed).accuracy(x, y) >= floor
    return holds


def stability_map(model, x, y, floor, modules=("enc", "expert_a", "head"), grid=(0.01, 0.02, 0.05, 0.1, 0.2, 0.4), trials=16):
    """Carte : (région de paramètres = module) × amplitude → probabilité de survie de l'invariant."""
    return {mod: {m: persistence(accuracy_invariant(model, x, y, (mod,), floor), m, trials, mod).probability for m in grid}
            for mod in modules}


# ================================================================ interactions non additives (« résonances » structurelles)

@dataclass
class StructuralResonanceCandidate:
    """Couple / triple de perturbations dont l'effet conjoint s'écarte fortement de la somme des effets seuls.
    (Nom : analogie ; ce n'est pas une résonance KAM physique.)"""
    components: Tuple[str, ...]
    interaction: float
    joint: float
    singles: Tuple[float, ...]

    @property
    def relative(self): return abs(self.interaction) / max(1e-9, abs(self.joint) + sum(abs(s) for s in self.singles))


def interaction(response: Callable[[Tuple[str, ...]], float], combo: Tuple[str, ...]):
    """interaction = response(δ1+…+δk) − Σ response(δi) (effet de second ordre et plus)."""
    singles = tuple(response((c,)) for c in combo); joint = response(combo)
    return StructuralResonanceCandidate(combo, joint - sum(singles), joint, singles)


class PerturbationInteractionGraph:
    """Nœuds : composants / transformations ; arêtes : force d'interaction. Couples et triples dangereux."""

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
                self.triples[t] = StructuralResonanceCandidate(t, j - sum(s) - pair_sum, j, s)   # interaction d'ordre 3 pure

    def dangerous(self, threshold=0.1, relative=0.25):
        out = [e for e in list(self.edges.values()) + list(self.triples.values())
               if abs(e.interaction) >= threshold and e.relative >= relative]
        return sorted(out, key=lambda e: -abs(e.interaction))


# ================================================================ transitions de type bifurcation

@dataclass
class TransitionBoundary:
    """Seuil où une quantité discrète change (routage, branche, invariant) ou chute brusquement (performance)."""
    quantity: str
    threshold: float
    before: object
    after: object
    kind: str


def detect_transitions(param_values: Sequence[float], observe: Callable[[float], Dict[str, object]], jump=0.15, refine=6):
    """Balaye un paramètre ; détecte changements discrets et chutes ≥ jump ; affine chaque seuil par bissection."""
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
    """Force d'un routage parasite croissante : décision dominante, chemin de calcul, exactitude."""
    from core.perturbation import Perturbation
    def observe(s):
        m = Perturbation("routing", (src, tgt), s).apply(model)
        acc = m.accuracy(x, y)
        return {"accuracy": round(acc, 4), "branch": tuple(m.topological_order()), "invariant_acc>=0.95": acc >= 0.95}
    return observe


# ================================================================ Fourier : quels invariants sont stables dans quelle base

def fourier_sensitivity(n=128, trials=40, eps=0.05, seed=0):
    """Perturbation localisée en TEMPS (impulsion) vs localisée en FRÉQUENCE (une raie) ; invariants :
    fréquence dominante, ensemble des raies, enveloppe locale (fenêtre), énergie (Parseval). Taux de survie."""
    from core.measurement import BasisTransform
    from core.measurement_experiments import make_signal, spectral_reasoning
    import random
    bt = BasisTransform(n); rng = random.Random(seed); res = {"time_perturbation": {}, "frequency_perturbation": {}}
    acc = {k: {"dominant": 0, "line_set": 0, "local_window": 0, "energy_rel_change": 0.0} for k in res}
    for t in range(trials):
        x, truth, dom = make_signal(rng, n); s0, d0 = spectral_reasoning(x, bt)
        amp = eps * float(x.norm())
        dt = torch.zeros(n, dtype=x.dtype); dt[rng.randrange(n)] = amp                       # impulsion temporelle
        f = rng.randrange(1, n // 2); tt = torch.arange(n, dtype=x.dtype)
        df = amp * math.sqrt(2.0 / n) * torch.cos(2 * math.pi * f * tt / n + rng.random() * 6.28)   # une raie
        w = slice(0, 16)
        for key, d in (("time_perturbation", dt), ("frequency_perturbation", df)):
            y = x + d; s1, d1 = spectral_reasoning(y, bt)
            acc[key]["dominant"] += d1 == d0; acc[key]["line_set"] += s1 == s0
            acc[key]["local_window"] += bool(torch.allclose(y[w], x[w], atol=1e-9))
            acc[key]["energy_rel_change"] += abs(bt.energy(y) - bt.energy(x)) / bt.energy(x)
    for key in acc:
        res[key] = {k: round(v / trials, 3) for k, v in acc[key].items()}
    return res


# ================================================================ référentiels : invariants vrais vs coordonnées

def frame_perturbation(trials=100, seed=0, eps=0.3):
    """Perturbe le référentiel (rotation + translation aléatoires) : distance et angle doivent rester stables,
    les coordonnées non (ce ne sont pas des invariants)."""
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


# ================================================================ algèbre : perturbation d'une chaîne de transformations

def algebra_chain_perturbation(trials=200, seed=0, chain_len=4):
    """Chaîne de transformations admissibles ; on remplace un maillon au hasard par une transformation du catalogue.
    Prédiction par le stabilisateur (le maillon est-il dans Stabilizer(solution_set)?) vs vérité (ensemble des
    solutions recalculé). Retourne exactitude de la prédiction et taux de bris."""
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
        if not known:                                   # transformation hors index (paramètre jamais vu)
            ood += 1; pred = _generalize(repl.name, stab)
        y = eq
        for t in chain: y = t(y)
        truth = solution_set(y) == solution_set(eq)
        correct += pred == truth; broke += not truth
    return {"prediction_accuracy": round(correct / trials, 3), "break_rate": round(broke / trials, 3), "ood_links": ood}


def _generalize(name, stab):
    """Généralisation par FAMILLE de transformation (nom sans paramètre) : admissible si la famille l'est et que le
    paramètre n'est pas 0 (seule exception apprise : MultiplyBothSides(0))."""
    fam = name.split("(")[0]; arg = name[len(fam):]
    fam_ok = any(t.split("(")[0] == fam for t in stab.transformations)
    return fam_ok and arg not in ("(0)", "(x)")

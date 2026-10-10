"""Observational equivalence classes, structural identifiability and identifiability-driven active measurements.

S1 ~ S2 if the two fitted structures are indistinguishable under the CURRENT set of measurements. The system maintains
the class {S1, S2, …} instead of choosing arbitrarily, and the active measurements try to break it.

This module imports no world module and never receives the truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch

from core.boundary_program import predict
from core.deep_fit import DT, FitData, approx_evidence
from core.deep_search import SearchOutcome, profiled_loglik
from core.deep_structure import WorldHypothesis, algebraic_form, anti_unify, n_atoms

EQUIV_AGREEMENT = 0.99                    # minimal agreement on the measured points to be indistinguishable
EVIDENCE_WINDOW = math.log(20.0)          # plausibility window around the best evidence
POLICIES = ("class_split", "structural_ig", "disagreement", "random", "efa")


class StructuralIdentifiability:
    IDENTIFIABLE = "IDENTIFIABLE"                                       # a single plausible structure
    PARAMETERS_ONLY = "PARAMETERS_IDENTIFIABLE_STRUCTURE_NOT"           # same function, several structures
    PARTIAL = "PARTIALLY_IDENTIFIABLE"                                  # a common sub-structure, the rest undetermined
    OBSERVATIONALLY_EQUIVALENT = "OBSERVATIONALLY_EQUIVALENT"           # different functions, indistinguishable on the measurements
    NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"                               # indistinguishable even on the measurable points
    ALL = (IDENTIFIABLE, PARAMETERS_ONLY, PARTIAL, OBSERVATIONALLY_EQUIVALENT, NOT_IDENTIFIABLE)


FAILURE_REASONS = ("insufficient_data", "observational_equivalence", "missing_primitives", "bad_parametric_fit", "search_interrupted",
                   "combinatorial_explosion", "non_discriminating_observables")


@dataclass
class ObservationalEquivalenceClass:
    """Set of structures indistinguishable under the current measurements. `forms` = distinct algebraic forms."""
    members: List[WorldHypothesis]
    forms: List[str] = field(default_factory=list)

    @property
    def size(self): return len(self.forms)
    @property
    def representative(self): return max(self.members, key=lambda h: h.evidence)
    def contains_form(self, form: str): return form in self.forms
    def keys(self): return [h.key for h in self.members]


class HypothesisPool:
    """Plausible hypotheses + cached predictions on the measured points and on a pool of measurable points.
    Everything is recomputed WITHOUT refitting when a measurement arrives (profiled-noise rescoring)."""

    def __init__(self, hyps: Sequence[WorldHypothesis], data: FitData, pool: Optional[torch.Tensor] = None, window=EVIDENCE_WINDOW, max_hyps=40):
        hs = sorted(hyps, key=lambda h: -h.evidence)[:max_hyps]; self.window = window; self.pool = pool
        self.hyps = hs; self.data = data
        self.P_obs = torch.stack([predict(h.program, data.X.float()).to(DT) for h in hs]) if hs else torch.zeros(0, len(data.X), dtype=DT)
        self.P_pool = torch.stack([predict(h.program, pool).to(DT) for h in hs]) if hs and pool is not None else None
        self.forms = [algebraic_form(h.program) for h in hs]; self.rescore()

    def rescore(self):
        """Evidence (profiled noise) and masses of all the hypotheses on the current measurements."""
        n = len(self.data)
        self.ev = torch.tensor([approx_evidence(h.n_params, n_atoms(h.program), profiled_loglik(p, self.data), n) for h, p in zip(self.hyps, self.P_obs)], dtype=DT)
        self.mass = torch.softmax(self.ev, 0) if len(self.hyps) else self.ev
        for h, e, m in zip(self.hyps, self.ev.tolist(), self.mass.tolist()): h.evidence, h.mass = e, m

    def add(self, x, y, flip=0.1):
        self.data = self.data.add(x, y, flip); self.P_obs = torch.cat([self.P_obs, torch.stack([predict(h.program, x).to(DT) for h in self.hyps])], 1); self.rescore()

    def best_index(self): return int(self.ev.argmax())
    def plausible(self): b = float(self.ev.max()); return [i for i in range(len(self.hyps)) if float(self.ev[i]) >= b - self.window]
    def noise(self):
        i = self.best_index(); return min(0.45, max(0.01, float((self.P_obs[i] != self.data.y).double().mean())))

    def classes(self) -> List[ObservationalEquivalenceClass]:
        """Partition of the plausible hypotheses by single linkage: agreement ≥ EQUIV_AGREEMENT on the measured points."""
        idx = self.plausible(); parent = {i: i for i in idx}
        def find(i):
            while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
            return i
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if float((self.P_obs[i] == self.P_obs[j]).double().mean()) >= EQUIV_AGREEMENT: parent[find(i)] = find(j)
        groups: Dict[int, List[int]] = {}
        for i in idx: groups.setdefault(find(i), []).append(i)
        out = []
        for g in groups.values():
            forms = []
            for i in g:
                if self.forms[i] not in forms: forms.append(self.forms[i])
            out.append(ObservationalEquivalenceClass([self.hyps[i] for i in g], forms))
        out.sort(key=lambda c: -c.representative.evidence); return out

    def best_class(self) -> ObservationalEquivalenceClass: return self.classes()[0]

    def class_indices(self): keys = set(self.best_class().keys()); return [i for i, h in enumerate(self.hyps) if h.key in keys]

    def breakable(self):
        """Share of the pairs of the best class that differ on the measurable pool (breakable by a measurement)."""
        idx = self.class_indices()
        if self.P_pool is None or len(idx) < 2: return 0.0
        pairs = [(i, j) for a, i in enumerate(idx) for j in idx[a + 1:]]
        return sum(float((self.P_pool[i] != self.P_pool[j]).double().mean()) > 0.01 for i, j in pairs) / len(pairs)


def identifiability(pool: HypothesisPool, budget_exhausted=False, cap_reached=False) -> dict:
    """Structural identifiability diagnostic of the best class + possible reasons for non-recovery."""
    c = pool.best_class(); idx = pool.class_indices(); SI = StructuralIdentifiability; reasons = []
    if c.size <= 1: status = SI.IDENTIFIABLE
    else:
        brk = pool.breakable(); same_fn = pool.P_pool is not None and all(float((pool.P_pool[i] != pool.P_pool[idx[0]]).double().mean()) <= 0.01 for i in idx)
        fam = anti_unify([h.program for h in c.members])
        if same_fn: status = SI.PARAMETERS_ONLY; reasons.append("observational_equivalence")
        elif brk == 0.0 and pool.P_pool is not None: status = SI.NOT_IDENTIFIABLE; reasons.append("non_discriminating_observables")
        elif not fam.is_trivial and fam.variables: status = SI.PARTIAL; reasons.append("insufficient_data")
        else: status = SI.OBSERVATIONALLY_EQUIVALENT; reasons.append("insufficient_data")
    if budget_exhausted: reasons.append("search_interrupted")
    if cap_reached: reasons.append("combinatorial_explosion")
    return {"status": status, "class_size": c.size, "members": len(c.members), "classes": len(pool.classes()), "breakable": pool.breakable(), "reasons": reasons,
            "plausible": len(pool.plausible())}


# ================================================================ measurement policies (vectorised over the pool)

def _posteriors(mass, P, eps):
    """Posterior masses for each candidate point and each outcome y ∈ {0, 1}. mass [H], P [H, Q] → post [Q, 2, H], p(y) [Q, 2]."""
    l1 = torch.where(P.T == 1, 1 - eps, eps); lik = torch.stack([1 - l1, l1], 1)                 # [Q, 2, H]
    w = lik * mass[None, None, :]; py = w.sum(2); return w / py.clamp_min(1e-300)[..., None], py


def score_class_split(pool: HypothesisPool):
    """Expected reduction of the equivalence class: one vote per FORM (no mass), the point that splits the
    best class most evenly eliminates the most forms whatever the answer."""
    idx = pool.class_indices(); forms = {}
    for i in idx: forms.setdefault(pool.forms[i], i)
    rep = list(forms.values())
    if len(rep) < 2: return score_disagreement(pool)                    # class reduced to one form: separate the neighbouring classes
    p = pool.P_pool[rep].mean(0); return 2 * p * (1 - p)


def score_structural_ig(pool: HypothesisPool):
    idx = pool.plausible(); m = pool.mass[idx] / pool.mass[idx].sum(); post, py = _posteriors(m, pool.P_pool[idx], pool.noise())
    H = lambda q: -(q.clamp_min(1e-300) * q.clamp_min(1e-300).log()).sum(-1)
    return (H(m[None]) - (py * H(post)).sum(1)).clamp_min(0.0)                 # ≥ 0 in theory; rounding can give −1e-9


def score_disagreement(pool: HypothesisPool):
    idx = pool.plausible(); m = pool.mass[idx] / pool.mass[idx].sum(); p = (m[:, None] * pool.P_pool[idx]).sum(0); return 1 - (2 * p - 1).abs()


def score_efa(pool: HypothesisPool, n_probe=200):
    """Historical EFA: expected gain in functional accuracy of the weighted vote, on probes from the pool."""
    idx = pool.plausible(); m = pool.mass[idx] / pool.mass[idx].sum(); P = pool.P_pool[idx]; post, py = _posteriors(m, P, pool.noise()); probes = P[:, :n_probe]
    pred = post @ probes; acc = torch.maximum(pred, 1 - pred).mean(2); base = (m[None] @ probes); return (py * acc).sum(1) - torch.maximum(base, 1 - base).mean()


SCORES: Dict[str, Callable] = {"class_split": score_class_split, "structural_ig": score_structural_ig, "disagreement": score_disagreement, "efa": score_efa}


@dataclass
class ActiveResult:
    pool: HypothesisPool
    queries: int
    trace: List[dict]


def active_identification(outcome: SearchOutcome, data: FitData, oracle: Callable, pool_points: torch.Tensor, policy="class_split", budget=16, seed=0,
                          sensor="standard", chooser: Optional[Callable] = None) -> ActiveResult:
    """Active measurement loop: choose a point of the pool according to the policy, measure, rescore (without refitting).
    `oracle(x, sensor, g)` returns a NOISY label. `chooser(pool)` replaces the policy (evaluation oracle
    policy, supplied from outside)."""
    g = torch.Generator().manual_seed(seed); hp = HypothesisPool(list(outcome.all.values()), data, pool_points); trace = []; used = torch.zeros(len(pool_points), dtype=torch.bool)
    for q in range(budget):
        before = hp.best_class().size
        if chooser is not None: i = int(chooser(hp))
        elif policy == "random": i = int(torch.randint(0, len(pool_points), (1,), generator=g))
        else:
            sc = SCORES[policy](hp).clone(); sc[used] = -1.0
            if float(sc.max()) <= 1e-9 and policy != "disagreement": sc = score_disagreement(hp).clone(); sc[used] = -1.0
            i = int(sc.argmax())
        used[i] = True; x = pool_points[i:i + 1]; y = oracle(x, sensor, g); hp.add(x, y, hp.noise())
        trace.append({"query": q, "point": i, "class_before": before, "class_after": hp.best_class().size, "classes": len(hp.classes()), "best": hp.hyps[hp.best_index()].key[-40:]})
    return ActiveResult(hp, budget, trace)

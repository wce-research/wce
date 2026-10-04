"""Modèle de fiabilité des sources : bayésien, hiérarchique, prudent, évolutif, conditionné au domaine.

    global prior Beta(a0, b0)  →  prior du type (capteur / document / locuteur / base)  →  posterior de la source
    r_s | données ~ Beta(α_type + correct_s, β_type + incorrect_s)          (UNKNOWN n'est ni correct ni incorrect)

Les a priori de type sont estimés par Bayes empirique (méthode des moments, analytique, sans MCMC) sur des graines
d'entraînement / calibration, jamais sur le test. Fiabilité effective (poids d'une preuve) : moyenne, borne basse
crédible, ou moyenne ajustée au risque. Oubli exponentiel (γ) pour une fiabilité qui dérive ; r(source, type de clé)
rétréci vers r(source). Groupes de dépendance : des sources qui se recopient ne comptent qu'une fois.
Compatible avec l'interface SourceReliability (estimate / update / table / stats) utilisée par EvidenceGraph.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

_GRID = torch.linspace(1e-4, 1 - 1e-4, 4001)


def beta_quantiles(a, b, qs=(0.05, 0.95)):
    """Quantiles d'une Beta(a, b) par CDF numérique vectorisée (sans scipy)."""
    a = torch.as_tensor(float(a)); b = torch.as_tensor(float(b))
    logpdf = (a - 1) * torch.log(_GRID) + (b - 1) * torch.log1p(-_GRID)
    pdf = torch.exp(logpdf - logpdf.max())
    cdf = torch.cumsum(pdf, 0); cdf = cdf / cdf[-1]
    return tuple(float(_GRID[int(torch.searchsorted(cdf, torch.tensor(q)).clamp(max=len(_GRID) - 1))]) for q in qs)


def source_type(name: str) -> str:
    """Métadonnée PUBLIQUE : la nature de la source (pas sa fiabilité)."""
    base = name.split("|")[0]
    for prefix, t in (("sensor", "sensor"), ("document", "document"), ("speaker", "speaker"), ("inventory", "database"),
                      ("capteur", "sensor"), ("witness", "speaker"), ("copier", "document")):
        if base.startswith(prefix): return t
    return "speaker"


@dataclass
class SourceRecord:
    source_id: str
    source_type: str
    alpha0: float
    beta0: float
    correct: float = 0.0
    incorrect: float = 0.0
    unknown: float = 0.0
    seen: int = 0
    by_key: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(lambda: [0.0, 0.0]))
    history: List[Tuple[int, bool]] = field(default_factory=list)
    calibration_state: str = "prior"          # prior | estimated | calibrated | drifting

    @property
    def alpha(self): return self.alpha0 + self.correct

    @property
    def beta(self): return self.beta0 + self.incorrect

    @property
    def mean(self): return self.alpha / (self.alpha + self.beta)

    @property
    def sd(self):
        a, b = self.alpha, self.beta
        return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    def interval(self, level=0.9):
        lo = (1 - level) / 2
        return beta_quantiles(self.alpha, self.beta, (lo, 1 - lo))


class SourceReliabilityModel:
    MODES = ("mean", "lower", "risk")

    def __init__(self, global_prior=(4.0, 2.0), type_priors: Dict[str, Tuple[float, float]] = None, mode="lower",
                 level=0.8, risk_lambda=1.0, discount=1.0, domain_shrinkage=10.0, domain=True, cap=0.995):
        self.global_prior = global_prior
        self.type_priors = dict(type_priors or {})
        self.mode, self.level, self.risk_lambda = mode, level, risk_lambda
        self.discount, self.domain_shrinkage, self.domain, self.cap = discount, domain_shrinkage, domain, cap
        self.records: Dict[str, SourceRecord] = {}
        self.groups: Dict[str, str] = {}                # source -> groupe de dépendance
        self.selection: Dict[str, Tuple[float, float]] = {}   # source -> (P(analysée | vraie), P(analysée | fausse))
        self.time = 0
        self._cache = {}
        self.online = False
        self.conflict_detection = True
        self.conflict_threshold = 0.6          # seuil de CRÉDIBILITÉ pour déclarer un conflit (séparé de la pondération)

    # ---- structure --------------------------------------------------------------------------
    def record(self, s) -> SourceRecord:
        if s not in self.records:
            t = source_type(s)
            a0, b0 = self.type_priors.get(t, self.global_prior)
            self.records[s] = SourceRecord(s, t, a0, b0)
        return self.records[s]

    @property
    def stats(self):                                       # compatibilité : {source: [α, β]}
        return _StatsView(self)

    # ---- apprentissage ------------------------------------------------------------------------
    def observe(self, s, outcome: Optional[bool], key_type=None, weight=1.0, t=None):
        """outcome : True (correct), False (incorrect), None (inconnu : ni l'un ni l'autre)."""
        r = self.record(s); r.seen += 1; self._cache.clear()
        if t is not None and t > self.time and self.discount < 1.0:
            f = self.discount ** (t - self.time)
            for rr in self.records.values():
                rr.correct *= f; rr.incorrect *= f
                for kc in rr.by_key.values(): kc[0] *= f; kc[1] *= f
            self.time = t
        if outcome is None: r.unknown += weight; return
        if outcome: r.correct += weight
        else: r.incorrect += weight
        if key_type is not None: r.by_key[key_type][0 if outcome else 1] += weight
        r.history.append((t if t is not None else r.seen, bool(outcome)))
        r.calibration_state = "estimated"

    def update(self, source, agree):
        """Compatibilité SourceReliability : mise à jour EN LIGNE par consensus, désactivée par défaut (le consensus des
        autres sources est un signal biaisé ; l'historique étiqueté fait foi). self-correction : observe() explicite."""
        if self.online: self.observe(source, bool(agree))

    def fit_empirical_bayes(self, strength_cap=40.0):
        """A priori de type par méthode des moments sur les posteriors des sources du type (analytique, sans MCMC)."""
        by_type = defaultdict(list)
        for r in self.records.values():
            n = r.correct + r.incorrect
            if n >= 5: by_type[r.source_type].append((r.correct + 0.5) / (n + 1.0))
        out = {}
        for t, ms in by_type.items():
            m = sum(ms) / len(ms)
            v = sum((x - m) ** 2 for x in ms) / max(1, len(ms) - 1) if len(ms) > 1 else 0.0
            v = max(v, 1e-3)
            k = min(max(m * (1 - m) / v - 1, 2.0), strength_cap)             # force de l'a priori (bornée)
            out[t] = (m * k, (1 - m) * k)
        self.type_priors.update(out)
        for r in self.records.values():
            if r.source_type in out: r.alpha0, r.beta0 = out[r.source_type]
            r.calibration_state = "calibrated"
        self._cache.clear()
        return out

    # ---- fiabilité effective -----------------------------------------------------------------------
    def posterior(self, s, key_type=None):
        r = self.record(s)
        a, b = r.alpha, r.beta
        if self.domain and key_type is not None and key_type in r.by_key:
            c, i = r.by_key[key_type]
            k = self.domain_shrinkage                                         # rétrécissement vers r(source)
            m = a / (a + b)
            a, b = m * k + c, (1 - m) * k + i
        return a, b

    def effective(self, s, key_type=None, mode=None):
        mode = mode or self.mode
        key = (s, key_type, mode)
        if key in self._cache: return self._cache[key]
        a, b = self.posterior(s, key_type)
        m = a / (a + b)
        if mode == "mean": v = m
        elif mode == "lower": v = beta_quantiles(a, b, ((1 - self.level),))[0]
        else:
            sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))); v = m - self.risk_lambda * sd
        s_sel = self.selection.get(s.split("|")[0])
        if s_sel is not None:                                                 # correction de sélection (canal analysé)
            pt, pf = s_sel
            if pt > 0 and pf > 0: v = v * pt / (v * pt + (1 - v) * pf)
        v = min(max(v, 0.01), self.cap)
        self._cache[key] = v
        return v

    def estimate(self, s): return self.effective(s)             # interface EvidenceGraph

    def conflict_credibility(self, s):
        """Une source est-elle assez crédible pour constituer un conflit ? Moyenne a posteriori (pas la borne basse)."""
        return self.effective(s, mode="mean")

    def estimate_for(self, s, key_type): return self.effective(s, key_type)

    def interval(self, s, level=0.9): return self.record(s).interval(level)

    def table(self): return {s: round(self.effective(s), 3) for s in sorted(self.records)}

    # ---- dérive ------------------------------------------------------------------------------------
    def drift(self, s, window=20, z=2.0):
        """Taux d'erreur récent vs long terme (test z sur proportions) : dérive détectée ?"""
        h = self.record(s).history
        if len(h) < 2 * window: return False, 0.0
        recent = [ok for _, ok in h[-window:]]; past = [ok for _, ok in h[:-window]]
        p1, p0 = sum(recent) / len(recent), sum(past) / len(past)
        p = (sum(recent) + sum(past)) / len(h)
        se = math.sqrt(max(p * (1 - p) * (1 / len(recent) + 1 / len(past)), 1e-9))
        stat = (p0 - p1) / se
        if stat > z: self.record(s).calibration_state = "drifting"
        return stat > z, round(stat, 3)

    # ---- dépendances ----------------------------------------------------------------------------------
    def group_of(self, s): return self.groups.get(s.split("|")[0], s.split("|")[0])

    def detect_dependencies(self, vote_sets, min_shared=5, threshold=0.5):
        """Deux sources qui partagent leurs ERREURS bien plus que le hasard sont regroupées (recopie).
        vote_sets : liste de ({source: valeur}, valeur de référence)."""
        both_wrong, wrong_a, wrong_b, shared = Counter(), Counter(), Counter(), Counter()
        for votes, ref in vote_sets:
            srcs = sorted(votes)
            for i, a in enumerate(srcs):
                for b in srcs[i + 1:]:
                    shared[(a, b)] += 1                                   # taux d'erreur mesurés sur les clés PARTAGÉES
                    wrong_a[(a, b)] += votes[a] != ref; wrong_b[(a, b)] += votes[b] != ref
                    if votes[a] != ref and votes[b] != ref and votes[a] == votes[b]: both_wrong[(a, b)] += 1
        for (a, b), n in shared.items():
            if n < min_shared: continue
            wa, wb = wrong_a[(a, b)] / n, wrong_b[(a, b)] / n
            joint = both_wrong[(a, b)] / n
            if min(wa, wb) > 0 and joint >= threshold * min(wa, wb) and joint > 3 * wa * wb:
                g = self.group_of(a); self.groups[b] = g; self.groups[a] = g
        return dict(self.groups)


class _StatsView(dict):
    """Vue {source: [α, β]} pour la compatibilité (écriture : fixe un a priori informatif pour la source)."""

    def __init__(self, model): super().__init__(); self.m = model

    def __getitem__(self, s):
        r = self.m.record(s); return [r.alpha, r.beta]

    def __setitem__(self, s, v):
        r = self.m.record(s); r.alpha0, r.beta0 = float(v[0]), float(v[1]); r.correct = r.incorrect = 0.0
        self.m._cache.clear()

    def __contains__(self, s): return s in self.m.records

    def __iter__(self): return iter(self.m.records)


# ================================================================ apprentissage depuis des mondes étiquetés

def selection_rates(parse_records):
    """P(analysée | vraie), P(analysée | fausse) par source ; parse_records : (source, correcte, analysée)."""
    c = defaultdict(Counter)
    for s, ok, parsed in parse_records:
        c[s][("T" if ok else "F", parsed)] += 1
    out = {}
    for s, k in c.items():
        nt = k[("T", True)] + k[("T", False)]; nf = k[("F", True)] + k[("F", False)]
        if nt >= 5 and nf >= 5: out[s] = (k[("T", True)] / nt, k[("F", True)] / nf)
    return out


def mnar_audit(emissions):
    """emissions : (source, correcte|None, analysée). Table par source et biais induit sur la fiabilité mesurée."""
    t = defaultdict(Counter)
    for s, ok, parsed in emissions:
        c = t[s]; c["emitted"] += 1; c["parsed" if parsed else "unparsed"] += 1
        if ok is None: c["unknown"] += 1
        elif parsed: c["correct_parsed" if ok else "incorrect_parsed"] += 1
        c["correct_emitted" if ok else ("incorrect_emitted" if ok is False else "unknown_emitted")] += 1
    out = {}
    for s, c in t.items():
        emitted_acc = c["correct_emitted"] / max(1, c["correct_emitted"] + c["incorrect_emitted"])
        parsed_acc = c["correct_parsed"] / max(1, c["correct_parsed"] + c["incorrect_parsed"])
        out[s] = {**dict(c), "accuracy_emitted": round(emitted_acc, 4), "accuracy_parsed": round(parsed_acc, 4),
                  "selection_bias": round(parsed_acc - emitted_acc, 4)}
    return out

"""Monitoring géométrique des trajectoires du cœur neuronal — prototype mesurable.

Inspiré du transport de distributions de représentations couche à couche. Honnêteté :
- on ne calcule PAS W2 exact en grande dimension ;
- chaque couche est projetée (projection aléatoire fixe) dans un espace commun de dimension 16,
  normalisée par une échelle de référence : les distances inter-couches sont donc des PROXIES ;
- les distributions sont des nuages de passes bruitées (bruit gaussien sur les embeddings de caractères) ;
- les distances entre nuages sont des Wasserstein-2 « sliced » (projections 1D, 32 directions).
Aucune de ces quantités n'est une preuve causale de vérité ou d'erreur.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

OBSERVABLES = ("transport_length", "geodesic_defect", "dissipation", "sensitivity", "branch_disagreement", "drift")


def sliced_w2(a: torch.Tensor, b: torch.Tensor, directions: torch.Tensor) -> float:
    """Approximation sliced-Wasserstein-2 entre deux nuages de même taille."""
    pa, pb = (a @ directions.T).sort(0).values, (b @ directions.T).sort(0).values
    return float(((pa - pb) ** 2).mean().sqrt())


class TrajectoryMonitor:
    def __init__(self, samples=24, noise=0.3, dim=16, projections=32, seed=0):
        self.samples, self.noise, self.dim = samples, noise, dim
        gen = torch.Generator().manual_seed(seed)
        self.directions = torch.nn.functional.normalize(torch.randn(projections, dim, generator=gen), dim=1)
        self._proj: Dict[int, torch.Tensor] = {}
        self.scales: Optional[List[float]] = None
        self.centroids: Dict[str, torch.Tensor] = {}
        self.latent_scale = 1.0
        self.records: List[Dict[str, float]] = []
        self.seed = seed

    def _projection(self, index, in_dim):
        key = index * 1000 + in_dim
        if key not in self._proj:
            gen = torch.Generator().manual_seed(self.seed + key)
            self._proj[key] = torch.randn(in_dim, self.dim, generator=gen) / math.sqrt(in_dim)
        return self._proj[key]

    def _layers(self, core, words, noise=0.0, gen=None):
        from core.language_core import encode_words
        with torch.no_grad():
            layers = core.layers(encode_words(words), noise=noise * float(core.emb.weight.std()), gen=gen)
        layers[-1] = torch.softmax(layers[-1], 1)
        return layers

    def _project(self, layers):
        out = []
        for i, act in enumerate(layers):
            p = act @ self._projection(i, act.shape[1])
            if self.scales: p = p / self.scales[i]
            out.append(p)
        return out

    def fit_reference(self, core, words: Sequence[str], labels: Sequence[str]):
        """Échelles par couche et centroïdes de trajectoires fiables (mots que le cœur classe correctement)."""
        layers = self._layers(core, words)
        self.scales = None
        proj = self._project(layers)
        self.scales = [float(p.norm(dim=1).mean()) + 1e-6 for p in proj]
        latent = layers[3]
        pred = layers[-1].argmax(1).tolist()
        self.centroids = {}
        for c in set(labels):
            idx = [i for i, (l, p) in enumerate(zip(labels, pred)) if l == c and core.concepts[p] == c]
            if idx: self.centroids[c] = latent[idx].mean(0)
        dists = [float((latent[i] - self.centroids[labels[i]]).norm()) for i in range(len(words)) if labels[i] in self.centroids]
        self.latent_scale = (sorted(dists)[len(dists) // 2] if dists else 1.0) + 1e-6
        return self

    def observe(self, core, word) -> Dict[str, float]:
        gen = torch.Generator().manual_seed(self.seed + sum(map(ord, word)))
        clean = self._layers(core, [word])
        noisy = self._layers(core, [word] * self.samples, noise=self.noise, gen=gen)
        clouds = self._project(noisy)
        steps = [sliced_w2(clouds[i], clouds[i + 1], self.directions) for i in range(len(clouds) - 1)]
        length = sum(steps)
        direct = sliced_w2(clouds[0], clouds[-1], self.directions)
        spread = [float((c - c.mean(0)).norm(dim=1).mean()) + 1e-6 for c in clouds]
        probs_clean, probs_noisy = clean[-1][0], noisy[-1]
        latent = clean[3][0]
        drift = min((float((latent - c).norm()) for c in self.centroids.values()), default=0.0) / self.latent_scale
        z = {"transport_length": length,                                        # sliced-W2 proxy, somme inter-couches
             "geodesic_defect": length / max(direct, 1e-6),                      # >= 1 ; proxy de non-géodésicité
             "dissipation": math.log(spread[-1] / spread[0]),                    # proxy : contraction/expansion du nuage
             "sensitivity": float((probs_noisy - probs_clean).abs().sum(1).mean()),
             "branch_disagreement": float((probs_noisy.argmax(1) != probs_clean.argmax()).float().mean()),
             "drift": drift,
             "confidence": float(probs_clean.max())}                             # neuronal, pas géométrique
        self.records.append(z)
        return z


class GeometricRiskEstimator:
    """Régression logistique L2 sur les observables géométriques standardisés -> risque [0, 1].
    Ce signal ne décide jamais seul : il alimente la politique du MetaController."""

    def __init__(self, features=OBSERVABLES, low=0.33, high=0.66):
        self.features, self.low, self.high = tuple(features), low, high
        self.mean = self.std = self.weight = None
        self.bias = 0.0

    def _x(self, zs):
        return torch.tensor([[z[f] for f in self.features] for z in zs], dtype=torch.float)

    def fit(self, zs: List[Dict[str, float]], incorrect: List[bool], steps=500, l2=1e-2):
        x = self._x(zs)
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        x = (x - self.mean) / self.std
        y = torch.tensor(incorrect, dtype=torch.float)
        w = torch.zeros(x.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, y) + l2 * (w ** 2).sum()
            loss.backward(); opt.step()
        self.weight, self.bias = w.detach(), b.item()
        return self

    def score(self, z) -> float:
        x = (self._x([z]) - self.mean) / self.std
        return float(torch.sigmoid(x @ self.weight + self.bias)[0])

    def level(self, z) -> str:
        s = self.score(z)
        return "LOW" if s < self.low else "MEDIUM" if s < self.high else "HIGH"

    def coefficients(self):
        return {f: round(float(w), 3) for f, w in zip(self.features, self.weight)}


def auroc(scores: Sequence[float], positives: Sequence[bool]) -> Optional[float]:
    pos = [s for s, p in zip(scores, positives) if p]; neg = [s for s, p in zip(scores, positives) if not p]
    if not pos or not neg: return None
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return round(wins / (len(pos) * len(neg)), 3)


# ================================================================ protocole d'audit

EXTRA_NONSENSE = ["brudel", "kaflo", "mizrant", "plonque", "vardix", "sulbre", "gronfu", "tepalo", "quirse",
                  "flandor", "moxite", "zebrule", "crapsin", "dulmex", "frovi", "jastel", "noquipe", "palvor"]


def _truth(word):
    from core.language_core import TRUE_CONCEPT
    t = TRUE_CONCEPT.get(word)
    if t is None: return "OTHER"
    return t


@dataclass
class AuditCase:
    word: str
    category: str
    truth: object
    predicted: str
    correct: bool
    z: Dict[str, float] = field(default_factory=dict)


def audit_cases(lc, monitor, words_by_category: Dict[str, List[str]], core=None):
    core = core or lc.core
    cases = []
    for category, words in words_by_category.items():
        for w in words:
            truth = _truth(w)
            pred = core.propose(w, 1)[0][0]
            ok = pred in truth if isinstance(truth, set) else pred == truth
            cases.append(AuditCase(w, category, truth, pred, ok, monitor.observe(core, w)))
    return cases


def audit_split(lc, exclude=()):
    """Mots d'audit, partitionnés de façon déterministe en calibration / test (disjoints)."""
    from core.language_core import NONSENSE, TRAPS, TRUE_CONCEPT
    known = sorted(lc.lexicon.grounded)
    heldout = sorted(f for f, c in TRUE_CONCEPT.items() if isinstance(c, str) and f not in lc.lexicon.grounded and f not in exclude)
    nonsense = sorted(set(NONSENSE + EXTRA_NONSENSE + TRAPS) - set(exclude))
    ambiguous = sorted(f for f, c in TRUE_CONCEPT.items() if isinstance(c, set))
    split = lambda ws: ([w for i, w in enumerate(ws) if i % 2 == 0], [w for i, w in enumerate(ws) if i % 2 == 1])
    cal, test = {}, {}
    for name, ws in (("known", known), ("heldout_forms", heldout), ("unknown_or_trap", nonsense)):
        cal[name], test[name] = split(ws)
    test["ambiguous"] = ambiguous
    return cal, test


def calibrate(lc, exclude=(), monitor=None):
    """Monitor ajusté sur les trajectoires fiables ; estimateur calibré sur les cas de calibration uniquement."""
    from core.language_core import OTHER_WORDS
    monitor = monitor or TrajectoryMonitor()
    words = list(lc.lexicon.grounded) + OTHER_WORDS
    labels = [lc.lexicon.grounded[w].most_common(1)[0][0] for w in lc.lexicon.grounded] + ["OTHER"] * len(OTHER_WORDS)
    monitor.fit_reference(lc.core, words, labels)
    cal, _ = audit_split(lc, exclude)
    cases = audit_cases(lc, monitor, cal)
    estimator = GeometricRiskEstimator().fit([c.z for c in cases], [not c.correct for c in cases])
    return monitor, estimator


def hallucination_audit(lc, exclude=()):
    """Compare Z entre sorties correctes et incorrectes, par catégorie ; AUROC sur l'ensemble de test."""
    import copy
    from core.language_core import CorruptCore
    monitor, estimator = calibrate(lc, exclude)
    _, test = audit_split(lc, exclude)
    cases = audit_cases(lc, monitor, test)
    corrupted = copy.deepcopy(lc.core); CorruptCore(scale=1.5, seed=3).apply(corrupted)
    cases += [AuditCase(c.word, "corrupted_module", c.truth, c.predicted, c.correct, c.z)
              for c in audit_cases(lc, monitor, {"corrupted_module": test["known"] + test["heldout_forms"]}, core=corrupted)]
    # mauvais routage : on force le 2e concept ; la trajectoire d'entrée est identique par construction
    for w in test["known"]:
        second = lc.core.propose(w, 2)[1][0]
        cases.append(AuditCase(w, "wrong_routing", _truth(w), second, False, monitor.observe(lc.core, w)))
    scored = [c for c in cases if c.category not in ("wrong_routing", "ambiguous")]
    incorrect = [not c.correct for c in scored]
    report = {"n_test": len(scored), "n_incorrect": sum(incorrect), "auroc": {}, "by_category": {}, "cases": cases,
              "coefficients": estimator.coefficients()}
    for f in OBSERVABLES:
        report["auroc"][f] = auroc([c.z[f] for c in scored], incorrect)
    report["auroc"]["neural_confidence(1-p)"] = auroc([1 - c.z["confidence"] for c in scored], incorrect)
    report["auroc"]["geometric_risk"] = auroc([estimator.score(c.z) for c in scored], incorrect)
    combo = GeometricRiskEstimator(features=OBSERVABLES + ("confidence",))
    cal, _ = audit_split(lc, exclude)
    cal_cases = audit_cases(lc, monitor, cal)
    combo.fit([c.z for c in cal_cases], [not c.correct for c in cal_cases])
    report["auroc"]["confidence+geometric"] = auroc([combo.score(c.z) for c in scored], incorrect)
    natural = [c for c in scored if c.category != "corrupted_module"]
    nat_inc = [not c.correct for c in natural]
    report["auroc_natural_errors_only"] = {
        "n": len(natural), "n_incorrect": sum(nat_inc),
        "neural_confidence(1-p)": auroc([1 - c.z["confidence"] for c in natural], nat_inc),
        "drift": auroc([c.z["drift"] for c in natural], nat_inc),
        "geometric_risk": auroc([estimator.score(c.z) for c in natural], nat_inc),
        "confidence+geometric": auroc([combo.score(c.z) for c in natural], nat_inc)}
    wrong = [c for c in cases if c.category == "wrong_routing"]
    report["wrong_routing_risk_mean"] = round(sum(estimator.score(c.z) for c in wrong) / max(1, len(wrong)), 3)
    for cat in sorted({c.category for c in cases}):
        for status in (True, False):
            group = [c for c in cases if c.category == cat and c.correct == status]
            if group:
                report["by_category"][f"{cat}/{'correct' if status else 'incorrect'}"] = {
                    "n": len(group), **{f: round(sum(c.z[f] for c in group) / len(group), 3) for f in OBSERVABLES + ("confidence",)},
                    "risk": round(sum(estimator.score(c.z) for c in group) / len(group), 3)}
    return report


def monitor_factory(exclude=()):
    """Pour language_core.experiment_matrix : calibration sur des mots disjoints de l'évaluation."""
    def factory(lc):
        return calibrate(lc, exclude=exclude)
    return factory


def evaluation_words():
    from core.language_core import evaluation_set, language_stream, tokenize_fr
    texts = evaluation_set() + [e["text"] for e in language_stream()]
    return {w for t in texts for w in tokenize_fr(t)}

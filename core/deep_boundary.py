"""Frontière de classe de modèles : le système doit pouvoir conclure qu'AUCUNE structure représentable n'explique les
observations, même si une structure interne a la meilleure vraisemblance relative.

Niveaux d'incertitude explicites : ParameterUncertainty, StructureUncertainty, FamilyUncertainty,
ModelClassUncertainty. Décision à trois issues : KNOWN / UNKNOWN (information insuffisante) / NONE_OF_THE_ABOVE
(observations incompatibles avec les hypothèses disponibles).

Les signaux hors classe sont calculés et évalués SÉPARÉMENT ; leur combinaison est une régression logistique à poids
lisibles, ajustée sur la validation uniquement. Ce module n'importe aucun module de monde.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch

from core.boundary_program import predict
from core.deep_fit import DT, FitData, fit_batch
from core.deep_search import SearchOutcome
from core.deep_structure import KNOWN, NONE_OF_THE_ABOVE, UNKNOWN, WorldHypothesis, family_of

SIGNALS = ("holdout_ppc", "residual_clustering", "nonparametric_gap", "mdl_bits_per_obs", "parameter_instability", "depth_improvement", "temporal_drift",
           "precise_in", "precise_out", "repeat_noise_excess", "structural_disagreement")
DECISION_COST = {"correct": 0.0, "unknown": 1.0, "false_none": 3.0, "false_known_ooc": 5.0, "wrong_known": 2.0}      # pré-enregistré (addendum §10)


def _z(k, n, p):
    """Écart normalisé d'un nombre d'erreurs k sur n à un taux attendu p (binomial)."""
    if n <= 0: return 0.0
    p = min(0.49, max(0.005, p)); return (k - n * p) / math.sqrt(n * p * (1 - p))


def split_holdout(data: FitData, frac=0.2, seed=0):
    """Retenue interne déterministe : (ajustement, retenue)."""
    n = len(data.X); perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)); k = int(round(n * frac)); h, t = perm[:k], perm[k:]
    sub = lambda i: FitData(data.X[i], data.y[i], data.flip[i], data.weight[i])
    return sub(t), sub(h), t, h


# ================================================================ signaux (chacun renvoie un flottant ; plus grand = plus suspect)

def sig_holdout_ppc(best: WorldHypothesis, train: FitData, hold: FitData):
    """Contrôle prédictif : erreurs sur la retenue contre le taux d'erreur à l'ajustement."""
    e_in = float((predict(best.program, train.X.float()).to(DT) != train.y).double().mean()); k = float((predict(best.program, hold.X.float()).to(DT) != hold.y).double().sum())
    return _z(k, len(hold.X), e_in)


def sig_residual_clustering(best: WorldHypothesis, data: FitData):
    """Regroupement spatial du résidu : parmi les erreurs, part dont le plus proche voisin est aussi une erreur, contre
    le taux attendu si les erreurs étaient du bruit i.i.d."""
    err = predict(best.program, data.X.float()).to(DT) != data.y; n = len(err); k = int(err.sum())
    if k < 3: return 0.0
    D = torch.cdist(data.X, data.X); D.fill_diagonal_(float("inf")); nn = D.argmin(1); joint = float(err[nn][err].double().sum())
    return _z(joint, k, k / n)


def knn_loo_error(data: FitData, k=7):
    """Erreur « leave-one-out » d'un vote des k plus proches voisins : référence non paramétrique."""
    D = torch.cdist(data.X, data.X); D.fill_diagonal_(float("inf")); idx = D.topk(min(k, len(data.X) - 1), largest=False).indices
    return float(((data.y[idx].mean(1) >= 0.5).to(DT) != data.y).double().mean())


def sig_nonparametric_gap(best: WorldHypothesis, data: FitData):
    """Écart à la référence non paramétrique : si un vote de voisins prédit mieux que la meilleure structure, il existe
    une régularité que la classe de modèles n'exprime pas."""
    e = float((predict(best.program, data.X.float()).to(DT) != data.y).double().mean()); n = len(data.X); return _z(e * n, n, knn_loo_error(data))


def sig_mdl_bits(best: WorldHypothesis, data: FitData): return -best.evidence / max(1, len(data.X)) / math.log(2)


def sig_parameter_instability(best: WorldHypothesis, data: FitData, probes, n_boot=3, seed=0):
    """Instabilité des paramètres : désaccord moyen, sur des sondes, entre réajustements sur des rééchantillonnages."""
    g = torch.Generator().manual_seed(seed); sk = best.program.skeleton(); th0 = torch.tensor([best.program.param_vector()], dtype=DT); preds = []
    for _ in range(n_boot):
        i = torch.randint(0, len(data.X), (len(data.X),), generator=g); d = FitData(data.X[i], data.y[i], data.flip[i], data.weight[i])
        th, _ = fit_batch(sk, d, th0, steps=40); preds.append(predict(sk.with_params(th[0].tolist()), probes))
    base = predict(best.program, probes); return float(torch.stack([(p != base).float().mean() for p in preds]).mean())


def sig_depth_improvement(outcome: SearchOutcome, data: FitData):
    """Amélioration encore en cours à la profondeur maximale explorée : baisse du taux d'erreur entre les deux derniers
    niveaux. Positive et forte ⇒ la structure dépasse probablement le budget de composition."""
    by = {}
    for k, h in outcome.all.items():
        e = float((outcome.preds[k] != data.y).double().mean()) if k in outcome.preds and len(outcome.preds[k]) == len(data.y) else None
        if e is not None: by[h.depth] = min(by.get(h.depth, 1.0), e)
    ds = sorted(by)
    return 0.0 if len(ds) < 2 else by[ds[-2]] - by[ds[-1]]


def sig_temporal_drift(best: WorldHypothesis, X, y, t):
    """Dérive temporelle : différence de taux d'erreur entre la moitié tardive et la moitié précoce des mesures."""
    err = (predict(best.program, X.float()).to(DT) != y.to(DT)).double(); med = t.median(); late, early = err[t > med], err[t <= med]
    if len(late) < 5 or len(early) < 5: return 0.0
    p = float(err.mean()); return (float(late.mean()) - float(early.mean())) / math.sqrt(max(1e-9, p * (1 - p) * (1 / len(late) + 1 / len(early))))


def sig_precise(best: WorldHypothesis, Xv, yv, sensor_flip=0.01, slack=0.03):
    """Validation au capteur précis sur des points NEUFS : erreurs du modèle au-delà du bruit du capteur."""
    if len(Xv) == 0: return 0.0
    return _z(float((predict(best.program, Xv).to(DT) != yv.to(DT)).double().sum()), len(Xv), sensor_flip + slack)


def repeat_noise(y_first, y_repeat):
    """Bruit SANS modèle par répétition : désaccord d = 2ε(1−ε) entre deux mesures du même point."""
    d = float((y_first.to(DT) != y_repeat.to(DT)).double().mean()); return (1 - math.sqrt(max(0.0, 1 - 2 * min(d, 0.499)))) / 2


def sig_repeat_noise_excess(best: WorldHypothesis, data: FitData, eps_repeat, n_repeat):
    """Excès d'erreur du modèle par rapport au bruit mesuré par répétition (indépendant de tout modèle)."""
    e = float((predict(best.program, data.X.float()).to(DT) != data.y).double().mean()); se = math.sqrt(max(1e-6, eps_repeat * (1 - eps_repeat)) / max(1, n_repeat) + e * (1 - e) / len(data.X))
    return (e - eps_repeat) / se


# ================================================================ niveaux d'incertitude

@dataclass
class Uncertainty:
    """Quatre niveaux distincts, jamais fusionnés : paramètres, structure, famille, classe de modèles."""
    parameter: float = 0.0            # ParameterUncertainty : instabilité des paramètres de la structure retenue
    structure: float = 0.0            # StructureUncertainty : 1 − masse de la meilleure structure
    family: float = 0.0               # FamilyUncertainty : 1 − masse de la meilleure famille
    model_class: float = 0.0          # ModelClassUncertainty : P(hors classe)


@dataclass
class CompatibleFamilies:
    """Ensemble pondéré de familles ; une famille est rejetable EN BLOC quand toutes ses structures sont incompatibles."""
    mass: Dict[str, float]
    best_error: Dict[str, float]
    rejected: List[str] = field(default_factory=list)

    @staticmethod
    def build(hyps: Sequence[WorldHypothesis], preds: Dict[str, torch.Tensor], y, reject_z=4.0, noise=0.1):
        mass: Dict[str, float] = {}; err: Dict[str, float] = {}; n = len(y)
        for h in hyps:
            mass[h.family] = mass.get(h.family, 0.0) + h.mass
            if h.key in preds and len(preds[h.key]) == n: err[h.family] = min(err.get(h.family, 1.0), float((preds[h.key] != y).double().mean()))
        rej = [f for f, e in err.items() if _z(e * n, n, noise) > reject_z]
        return CompatibleFamilies(mass, err, rej)

    def compatible(self): return [f for f in self.mass if f not in self.rejected]
    def all_rejected(self): return len(self.mass) > 0 and not self.compatible()
    def best(self): return max(self.mass, key=self.mass.get) if self.mass else None


# ================================================================ combinaison transparente et décision

class BoundaryModel:
    """Régression logistique sur les signaux standardisés. Poids lisibles (un par signal). Ajustée sur validation."""

    def __init__(self, signals=SIGNALS): self.signals = tuple(signals); self.mu = torch.zeros(len(signals), dtype=DT); self.sd = torch.ones(len(signals), dtype=DT); self.w = torch.zeros(len(signals), dtype=DT); self.b = 0.0; self.fitted = False

    def _x(self, rows): return torch.tensor([[float(r.get(s, 0.0)) for s in self.signals] for r in rows], dtype=DT).clamp(-50, 50)

    def fit(self, rows: Sequence[dict], ooc: Sequence[bool], l2=1.0, steps=400):
        X = self._x(rows); y = torch.tensor([float(v) for v in ooc], dtype=DT); self.mu = X.mean(0); self.sd = X.std(0).clamp_min(1e-6); Z = (X - self.mu) / self.sd
        w = torch.zeros(Z.shape[1], dtype=DT, requires_grad=True); b = torch.zeros(1, dtype=DT, requires_grad=True); opt = torch.optim.LBFGS([w, b], max_iter=steps)
        def closure():
            opt.zero_grad(); p = torch.sigmoid(Z @ w + b).clamp(1e-9, 1 - 1e-9); loss = -(y * p.log() + (1 - y) * (1 - p).log()).mean() + l2 * (w ** 2).sum() / len(y); loss.backward(); return loss
        opt.step(closure); self.w = w.detach(); self.b = float(b.detach()); self.fitted = True; return self

    def prob(self, rows): return torch.sigmoid(((self._x(rows) - self.mu) / self.sd) @ self.w + self.b).tolist()
    def weights(self): return {s: round(float(v), 4) for s, v in zip(self.signals, self.w)}
    def to_dict(self): return {"signals": list(self.signals), "mu": self.mu.tolist(), "sd": self.sd.tolist(), "w": self.w.tolist(), "b": self.b}

    @staticmethod
    def from_dict(d):
        m = BoundaryModel(d["signals"]); m.mu, m.sd, m.w, m.b, m.fitted = torch.tensor(d["mu"], dtype=DT), torch.tensor(d["sd"], dtype=DT), torch.tensor(d["w"], dtype=DT), d["b"], True; return m


@dataclass
class Decision:
    label: str                       # KNOWN | UNKNOWN | NONE_OF_THE_ABOVE
    p_out_of_class: float
    uncertainty: Uncertainty
    reason: str = ""


def decide(p_ooc: float, class_size: int, structure_mass: float, tau_none=0.5, tau_known=0.2, min_mass=0.5, use_none=True, use_classes=True,
           families_all_rejected=False) -> str:
    """NONE_OF_THE_ABOVE : les observations contredisent la classe de modèles (P(hors classe) ≥ tau_none, ou toutes
    les familles rejetées). UNKNOWN : pas de contradiction établie mais la structure n'est pas tranchée (classe
    d'équivalence non réduite, masse insuffisante, ou P(hors classe) intermédiaire). KNOWN sinon."""
    if use_none and (p_ooc >= tau_none or families_all_rejected): return NONE_OF_THE_ABOVE
    if p_ooc > tau_known and use_none: return UNKNOWN
    if use_classes and class_size > 1: return UNKNOWN
    if structure_mass < min_mass: return UNKNOWN
    return KNOWN


def decision_cost(label: str, in_class: bool, functionally_correct: bool) -> float:
    """Coût pré-enregistré d'une décision (addendum §10)."""
    c = DECISION_COST
    if not in_class: return c["correct"] if label == NONE_OF_THE_ABOVE else c["unknown"] if label == UNKNOWN else c["false_known_ooc"]
    if label == NONE_OF_THE_ABOVE: return c["false_none"]
    if label == UNKNOWN: return c["correct"] if not functionally_correct else c["unknown"]
    return c["correct"] if functionally_correct else c["wrong_known"]


def tune_thresholds(p_ooc, class_sizes, masses, in_class, correct, grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """Seuils (tau_none, tau_known) minimisant le coût décisionnel moyen — sur VALIDATION uniquement."""
    best = None
    for tn in grid:
        for tk in grid:
            if tk > tn: continue
            c = sum(decision_cost(decide(p, cs, m, tn, tk), ic, ok) for p, cs, m, ic, ok in zip(p_ooc, class_sizes, masses, in_class, correct)) / max(1, len(p_ooc))
            if best is None or c < best[0] - 1e-12: best = (c, tn, tk)
    return {"cost": best[0], "tau_none": best[1], "tau_known": best[2]}


def auroc(scores, labels):
    """AUROC par rangs (égalités à ½). labels : True = hors classe."""
    pos = [s for s, l in zip(scores, labels) if l]; neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg: return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def auprc(scores, labels):
    order = sorted(range(len(scores)), key=lambda i: -scores[i]); tp = 0; ap = 0.0; npos = sum(bool(l) for l in labels)
    if npos == 0: return None
    for r, i in enumerate(order, 1):
        if labels[i]: tp += 1; ap += tp / r
    return ap / npos


def suggest_missing_primitive(best: WorldHypothesis, data: FitData, drift_z=0.0, precise_out_z=0.0, depth_improvement=0.0):
    """Diagnostic EXPLORATOIRE : quelle opération minimale manque-t-il ? Suggestion contrôlée parmi quatre types, à
    partir de la géométrie du résidu — jamais une génération ouverte de primitives."""
    err = predict(best.program, data.X.float()).to(DT) != data.y; E = data.X[err]
    if drift_z > 3.0: return {"missing": "observable", "detail": "le temps de mesure (le résidu dérive avec le temps)"}
    if precise_out_z > 3.0 and int(err.sum()) < 0.15 * len(err): return {"missing": "relation", "detail": "une loi différente hors de la zone observée (structure valide seulement localement)"}
    if depth_improvement > 0.02: return {"missing": "combinaison", "detail": "une composition plus profonde que le budget (l'erreur baisse encore au dernier niveau)"}
    if len(E) >= 8:
        c = E - E.mean(0); ev = torch.linalg.eigvalsh(c.T @ c / len(E)); aniso = float(ev[1] / ev[0].clamp_min(1e-9))
        if aniso > 6.0: return {"missing": "transformation", "detail": "une frontière courbe ou une dilatation anisotrope (résidu allongé le long d'une courbe)"}
        return {"missing": "transformation", "detail": "un motif périodique ou une frontière non circulaire / non rectiligne (résidu dispersé en amas)"}
    return {"missing": "indéterminé", "detail": "résidu trop faible pour suggérer une primitive"}

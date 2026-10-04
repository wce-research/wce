"""Attribution de crédit GÉNÉRALISÉE sur un système composite (neuronal + symbolique).

Le système : un petit ModelState (tâche A) ET une chaîne symbolique de réponse
  texte → correspondance lexicale → entité → capteur (étalonnage) → transformation d'unités (cm→m)
  → témoignages de sources (fiabilités estimées) → croyance a priori → fusion → planificateur (seuil d'engagement).

Une panne (cachée) touche UN composant parmi : weights, expert, routing, lexical, measurement, transformation,
source, belief, planner. Trois attributions comparées :
  - gradient seul (observationnel, ne voit que les paramètres différentiables) ;
  - aléatoire ;
  - généralisée INTERVENTIONNELLE : remplacer réellement chaque composant par une version par défaut / réestimée
    (instantané pour le neuronal, défauts ou réestimation sur un petit lot d'étalonnage pour le symbolique), réévaluer.
Corrélation ≠ causalité : seule l'intervention mesure un effet causal (dans ce banc contrôlé).
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List

import torch

from core.model_state import CreateConnection, ActivateModule, RestoreModule, RemoveConnection, RestoreSnapshot
from core.perturbation import Credit, gradient_credit, interventional_credit
from core.plasticity import build_base_model, pretrain, task_a_labels, task_split

COLORS = ["rouge", "bleu", "vert", "jaune"]
CONCEPT = {"rouge": "red", "bleu": "blue", "vert": "green", "jaune": "yellow"}
FAULTS = ("weights:enc", "expert:expert_a", "weights:head", "routing", "lexical", "measurement", "transformation",
          "source", "belief", "planner")
DEFAULTS = {"threshold": 0.7, "prior_long": 0.5, "factor": 0.01, "offset": 0.0}


@dataclass
class Pipeline:
    lexicon: Dict[str, str]
    objects: Dict[str, float]                 # couleur (concept) -> longueur vraie (cm)
    offset: float = 0.0                       # étalonnage du capteur (cm)
    factor: float = 0.01                      # transformation cm -> m
    reliab: Dict[str, float] = field(default_factory=lambda: {"s1": 0.9, "s2": 0.6, "s3": 0.85})
    prior_long: float = 0.5
    threshold: float = 0.7
    sensor_r: float = 0.85


def make_pipeline(seed):
    rng = random.Random(seed)
    objs = {CONCEPT[c]: rng.choice([rng.uniform(15, 40), rng.uniform(60, 95)]) for c in COLORS}
    return Pipeline(dict(CONCEPT), objs)


def queries(pipe: Pipeline, seed, n=40):
    """Requêtes « longueur du cube <couleur> ? » + témoignages simulés (VRAIES fiabilités : s1 .9, s2 .6, s3 .85)."""
    rng = random.Random(seed); out = []
    true_r = {"s1": 0.9, "s2": 0.6, "s3": 0.85}
    for i in range(n):
        w = rng.choice(COLORS); truth = "long" if pipe.objects[CONCEPT[w]] >= 50 else "court"
        reports = []
        for s, r in true_r.items():
            if rng.random() < 0.6: reports.append((s, truth if rng.random() < r else ("court" if truth == "long" else "long")))
        noise = rng.gauss(0, 4.0)
        out.append({"word": w, "truth": truth, "reports": reports, "noise": noise})
    return out


def answer(pipe: Pipeline, q):
    concept = pipe.lexicon.get(q["word"])
    if concept not in pipe.objects: return None
    reading_cm = pipe.objects[concept] + q["noise"] + pipe.offset          # capteur (avec erreur d'étalonnage éventuelle)
    meters = reading_cm * pipe.factor                                      # transformation d'unités
    p_long = min(max(pipe.prior_long, 1e-6), 1 - 1e-6)
    odds = p_long / (1 - p_long)
    r = pipe.sensor_r; odds *= (r / (1 - r)) if meters >= 0.5 else ((1 - r) / r)
    for s, v in q["reports"]:
        rs = min(max(pipe.reliab.get(s, 0.5), 1e-3), 1 - 1e-3)
        odds *= (rs / (1 - rs)) if v == "long" else ((1 - rs) / rs)
    p = odds / (1 + odds); best = "long" if p >= 0.5 else "court"
    return best if max(p, 1 - p) >= pipe.threshold else None


def pipeline_score(pipe, qs):
    n = len(qs); ok = sum(answer(pipe, q) == q["truth"] for q in qs)
    wrong = sum((a := answer(pipe, q)) is not None and a != q["truth"] for q in qs)
    return (ok - wrong) / n


@dataclass
class CompositeSystem:
    model: object
    pipe: Pipeline
    xv: torch.Tensor
    yv: torch.Tensor
    qs: list
    calib: list                                   # petit lot d'étalonnage étiqueté (diagnostic)


def score(sys: CompositeSystem):
    return 0.5 * sys.model.accuracy(sys.xv, sys.yv) + 0.5 * pipeline_score(sys.pipe, sys.qs)


_BASE = {}


def base_model(seed=0):
    if seed not in _BASE:
        x, y, xv, yv = task_split(task_a_labels, seed=seed)
        m = build_base_model(seed=seed); pretrain(m, x, y, steps=250, seed=seed); m.snapshot("good")
        _BASE[seed] = (m, xv, yv)
    m, xv, yv = _BASE[seed]; return copy.deepcopy(m), xv, yv


def make_system(seed):
    m, xv, yv = base_model(0)
    pipe = make_pipeline(seed)
    return CompositeSystem(m, pipe, xv, yv, queries(pipe, seed * 7 + 1), queries(pipe, seed * 7 + 2, 30))


def inject(sys: CompositeSystem, fault, rng: random.Random):
    m, p = sys.model, sys.pipe
    if fault.startswith(("weights:", "expert:")):
        mod = fault.split(":")[1]
        for k, v in m.modules[mod].slow.items():
            m.modules[mod].slow[k] = v + rng.uniform(0.8, 1.5) * v.std() * torch.randn(v.shape, generator=torch.Generator().manual_seed(rng.randrange(10 ** 6)))
    elif fault == "routing":
        m.apply(ActivateModule(module="expert_b")); m.apply(CreateConnection(source="input", target="expert_b"))
        m.apply(CreateConnection(source="expert_b", target="head", strength=rng.uniform(1.0, 2.0)))
    elif fault == "lexical":
        a, b = rng.sample(COLORS, 2); p.lexicon[a], p.lexicon[b] = p.lexicon[b], p.lexicon[a]
    elif fault == "measurement": p.offset = rng.choice([-1, 1]) * rng.uniform(35, 60)
    elif fault == "transformation": p.factor = rng.choice([0.1, 0.001])
    elif fault == "source": p.reliab["s2"] = 0.97; p.reliab["s1"] = 0.1
    elif fault == "belief": p.prior_long = rng.choice([0.999, 0.001])
    elif fault == "planner": p.threshold = rng.choice([0.5, 0.9999])


# ---- interventions SANS accès à la panne : instantané (neuronal) ou défaut / réestimation (symbolique) ----

def _reestimate_lexicon(sys):
    """Réapprend la correspondance couleur→concept à partir du lot d'étalonnage (mesures étiquetées)."""
    p = sys.pipe
    for w in COLORS:
        qs = [q for q in sys.calib if q["word"] == w]
        if not qs: continue
        best = max(p.objects, key=lambda c: sum((("long" if p.objects[c] >= 50 else "court") == q["truth"]) for q in qs)
                   - 0.01 * (c != p.lexicon.get(w)))
        p.lexicon[w] = best
    return sys


def _reestimate_sources(sys):
    p = sys.pipe; counts = {}
    for q in sys.calib:
        for s, v in q["reports"]:
            a, n = counts.get(s, (1.0, 2.0)); counts[s] = (a + (v == q["truth"]), n + 1)
    p.reliab = {s: a / n for s, (a, n) in counts.items()}
    return sys


def _set(attr, val):
    def f(sys): setattr(sys.pipe, attr, val); return sys
    return f


def _restore(mod):
    def f(sys): sys.model.apply(RestoreModule(module=mod, snapshot="good")); return sys
    return f


def _unroute(sys):
    if ("expert_b", "head") in sys.model.connectivity: sys.model.apply(RemoveConnection(source="expert_b", target="head"))
    return sys


COMPONENTS = {
    "weights:enc": ("weights", _restore("enc")), "expert:expert_a": ("expert", _restore("expert_a")),
    "weights:head": ("weights", _restore("head")), "routing": ("routing", _unroute),
    "lexical": ("lexical", _reestimate_lexicon), "measurement": ("measurement", _set("offset", 0.0)),
    "transformation": ("transformation", _set("factor", 0.01)), "source": ("source", _reestimate_sources),
    "belief": ("belief", _set("prior_long", 0.5)), "planner": ("planner", _set("threshold", 0.7)),
}


def generalized_credit(sys):
    """Crédit interventionnel sur TOUS les types de composants + crédit observationnel (gradient) pour le neuronal."""
    base, credits = interventional_credit(sys, COMPONENTS, score)
    g = gradient_credit(sys.model, sys.xv, sys.yv)
    for c in credits:
        key = f"weights:{c.component.split(':')[1]}" if ":" in c.component else None
        if c.component == "routing": c.observational = max([v for k, v in g.items() if k.startswith("routing:expert_b")] or [0.0])
        elif key: c.observational = g.get(key, 0.0)
    return base, credits


def gradient_only_cause(sys):
    """Cause selon le gradient seul : composant différentiable au plus fort crédit observationnel."""
    g = gradient_credit(sys.model, sys.xv, sys.yv)
    k = max(g, key=g.get)
    if k.startswith("routing:"): return "routing"
    mod = k.split(":")[1]
    return "expert:expert_a" if mod == "expert_a" else f"weights:{mod}"


def credit_benchmark(n_trials=40, seed=0, min_drop=0.03):
    """Seules les pannes OBSERVABLES (chute du score ≥ min_drop vs le système sain) sont des « erreurs » à expliquer ;
    les pannes silencieuses sont comptées à part (aucune méthode ne peut les attribuer par leur effet)."""
    rng = random.Random(seed); res = {"generalized": 0, "gradient_only": 0, "random": 0, "repaired": 0, "n": 0, "silent": 0,
                                       "neural": {"generalized": 0, "gradient_only": 0, "n": 0},
                                       "symbolic": {"generalized": 0, "gradient_only": 0, "n": 0}}
    for i in range(n_trials):
        fault = FAULTS[i % len(FAULTS)]
        sys = make_system(seed * 1000 + i); inject(sys, fault, rng)
        healthy = make_system(seed * 1000 + i)
        if score(healthy) - score(sys) < min_drop: res["silent"] += 1; continue
        base, credits = generalized_credit(sys)
        top = credits[0].component
        res["generalized"] += top == fault; g = gradient_only_cause(sys); res["gradient_only"] += g == fault
        res["random"] += rng.choice(FAULTS) == fault; res["n"] += 1
        fam = "neural" if fault in ("weights:enc", "expert:expert_a", "weights:head", "routing") else "symbolic"
        res[fam]["generalized"] += top == fault; res[fam]["gradient_only"] += g == fault; res[fam]["n"] += 1
        repaired = COMPONENTS[top][1](copy.deepcopy(sys))
        res["repaired"] += score(repaired) >= score(healthy) - 0.05
    n = max(1, res["n"])
    out = {k: round(res[k] / n, 3) for k in ("generalized", "gradient_only", "random", "repaired")}
    for fam in ("neural", "symbolic"):
        f = res[fam]; out[fam] = {k: round(f[k] / max(1, f["n"]), 3) for k in ("generalized", "gradient_only")}
    out["n"] = res["n"]; out["silent"] = res["silent"]
    return out

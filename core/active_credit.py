"""Crédit actif (deux phases), identifiabilité, observabilité, ensemble minimal de diagnostics, pannes silencieuses.

Phase 1 — classement BON MARCHÉ : vérifications de cohérence peu coûteuses + sensibilité (gradient) → a priori
          P(cause | signes) appris par simulation sur des graines d'ENTRAÎNEMENT (Bayes naïf lissé).
Phase 2 — INTERVENTIONS sur les candidats les plus probables, choisies par valeur d'information (single-fault :
          ordre p(c)/coût), arrêt dès qu'une intervention restaure le score.
Comparé à : toutes les interventions, top-k aléatoire, top-k par sensibilité, VOI.

Observabilité (inspirée des systèmes dynamiques) : matrice cause × mesure → distinguabilité. Une panne silencieuse
est classée : NON IDENTIFIABLE (aucune mesure disponible ne la distingue du sain), IDENTIFIABLE MAIS NON DÉTECTÉE
(une mesure existante la distingue, mais pas la mesure courante = score des requêtes), ou SANS EFFET FONCTIONNEL
(comportement strictement identique : seule une inspection structurelle pourrait la voir).
"""
from __future__ import annotations

import copy
import itertools
import random
from collections import Counter, defaultdict

import torch

from core.generalized_credit import (COMPONENTS, FAULTS, answer, gradient_only_cause, inject, make_system, pipeline_score, score)
from core.perturbation import gradient_credit

HEALTHY = "healthy"
CAUSES = (HEALTHY,) + FAULTS


# ================================================================ mesures diagnostiques sur le système composite

def _declared_topology(sys): return {("input", "enc"), ("enc", "expert_a"), ("expert_a", "head")}


def m_query_score(sys):
    s = pipeline_score(sys.pipe, sys.calib); a = sys.model.accuracy(sys.xv, sys.yv)
    return "high" if 0.5 * a + 0.5 * s >= 0.85 else "mid" if 0.5 * a + 0.5 * s >= 0.7 else "low"


def m_model_acc(sys): return "ok" if sys.model.accuracy(sys.xv, sys.yv) >= 0.97 else "degraded"
def m_routing(sys): return "ok" if set(sys.model.connectivity) == _declared_topology(sys) else "changed"


def _drift(mod):
    def f(sys):
        snap = sys.model.snapshots["good"].modules[mod].slow; cur = sys.model.modules[mod].slow
        rel = sum(float((cur[k] - snap[k]).norm()) for k in snap) / max(1e-9, sum(float(snap[k].norm()) for k in snap))
        return "none" if rel < 1e-6 else "large"
    return f


def m_unit(sys): return "ok" if abs(100 * sys.pipe.factor - 1.0) < 1e-9 else "bad"


def m_sensor_ref(sys):
    """Mesure d'un objet de référence de 50 cm par le capteur (bruit compris)."""
    reading = 50.0 + sys.pipe.offset + random.Random(1).gauss(0, 4.0)
    return "ok" if abs(reading - 50.0) < 15 else "biased"


def m_lexicon(sys):
    """Audit lexical : par mot, exactitude des réponses du lot d'étalonnage ; un mot mal mappé échoue."""
    bad = 0
    for w in {q["word"] for q in sys.calib}:
        qs = [q for q in sys.calib if q["word"] == w]
        if sum(answer(sys.pipe, q) == q["truth"] for q in qs) / len(qs) < 0.5: bad += 1
    return "ok" if bad == 0 else "word_fails"


def m_source(sys):
    counts = {}
    for q in sys.calib:
        for s, v in q["reports"]:
            a, n = counts.get(s, (1.0, 2.0)); counts[s] = (a + (v == q["truth"]), n + 1)
    dev = max(abs(sys.pipe.reliab.get(s, 0.5) - a / n) for s, (a, n) in counts.items())
    return "ok" if dev < 0.25 else "miscalibrated"


def m_prior(sys): return "ok" if 0.05 <= sys.pipe.prior_long <= 0.95 else "extreme"
def m_threshold(sys): return "ok" if 0.6 <= sys.pipe.threshold <= 0.95 else "out_of_range"


MEASUREMENTS = {  # nom : (fonction, coût)
    "query_score": (m_query_score, 0.30), "model_acc": (m_model_acc, 0.20), "routing": (m_routing, 0.02),
    "drift:enc": (_drift("enc"), 0.05), "drift:expert_a": (_drift("expert_a"), 0.05), "drift:head": (_drift("head"), 0.05),
    "unit_check": (m_unit, 0.02), "sensor_ref": (m_sensor_ref, 0.10), "lexicon_audit": (m_lexicon, 0.10),
    "source_audit": (m_source, 0.10), "prior_range": (m_prior, 0.02), "threshold_range": (m_threshold, 0.02),
}
CURRENT = ("query_score",)                       # mesure « courante » du jalon précédent (effet sur le score)


def observe_all(sys): return {m: f(sys) for m, (f, c) in MEASUREMENTS.items()}


# ================================================================ observabilité, identifiabilité, ensemble minimal

def simulate_outcomes(n_per=12, seed=0):
    """Issues de toutes les mesures, par cause (y compris sain), sur des graines d'ENTRAÎNEMENT."""
    rng = random.Random(seed); data = defaultdict(list)
    for c in CAUSES:
        for i in range(n_per):
            sys = make_system(5000 + 100 * CAUSES.index(c) + i)
            if c != HEALTHY: inject(sys, c, rng)
            data[c].append(observe_all(sys))
    return data


def observability_matrix(data):
    """D[cause][mesure] = variation totale entre la distribution des issues sous la cause et sous l'état sain."""
    out = {}
    for c in CAUSES:
        out[c] = {}
        for m in MEASUREMENTS:
            pc, ph = Counter(o[m] for o in data[c]), Counter(o[m] for o in data[HEALTHY])
            keys = set(pc) | set(ph); nc, nh = sum(pc.values()), sum(ph.values())
            out[c][m] = round(0.5 * sum(abs(pc[k] / nc - ph[k] / nh) for k in keys), 3)
    return out


def distinguishable(data, a, b, measures, thr=0.5):
    for m in measures:
        pa, pb = Counter(o[m] for o in data[a]), Counter(o[m] for o in data[b])
        keys = set(pa) | set(pb); na, nb = sum(pa.values()), sum(pb.values())
        if 0.5 * sum(abs(pa[k] / na - pb[k] / nb) for k in keys) >= thr: return True
    return False


def identifiability(data, measures):
    pairs = list(itertools.combinations(CAUSES, 2))
    ok = [(a, b) for a, b in pairs if distinguishable(data, a, b, measures)]
    return {"pairs": len(pairs), "distinguishable": len(ok), "fraction": round(len(ok) / len(pairs), 3),
            "confusable": [(a, b) for a, b in pairs if (a, b) not in ok]}


def minimal_diagnostic_set(data, measures=None, required=None):
    """Plus petit (puis moins coûteux) ensemble de mesures distinguant toutes les paires de causes distinguables
    avec l'ensemble complet. Recherche exacte par taille croissante (12 mesures : faisable)."""
    measures = list(measures or MEASUREMENTS)
    target = set(map(tuple, [p for p in itertools.combinations(CAUSES, 2) if distinguishable(data, *p, measures)]))
    for k in range(1, len(measures) + 1):
        best = None
        for sub in itertools.combinations(measures, k):
            got = {p for p in target if distinguishable(data, *p, sub)}
            if got == target:
                cost = sum(MEASUREMENTS[m][1] for m in sub)
                if best is None or cost < best[1]: best = (sub, cost)
        if best: return {"set": list(best[0]), "cost": round(best[1], 3), "full_cost": round(sum(MEASUREMENTS[m][1] for m in measures), 3),
                         "pairs_covered": len(target)}
    return None


def silent_failure_analysis(n_trials=60, seed=0, min_drop=0.03):
    """Pour chaque panne injectée SANS effet sur la mesure courante (score des requêtes) : quelles mesures la
    distinguent ? Classification : identifiable-non-détectée / non-identifiable / sans effet fonctionnel."""
    rng = random.Random(seed); out = Counter(); by_fault = defaultdict(Counter); witnesses = Counter()
    for i in range(n_trials):
        fault = FAULTS[i % len(FAULTS)]; sys = make_system(seed * 1000 + i); healthy = make_system(seed * 1000 + i)
        inject(sys, fault, rng)
        if score(healthy) - score(sys) >= min_drop: out["detected"] += 1; continue
        o_f, o_h = observe_all(sys), observe_all(healthy)
        diff = [m for m in MEASUREMENTS if o_f[m] != o_h[m]]
        same_behavior = all(answer(sys.pipe, q) == answer(healthy.pipe, q) for q in sys.qs + sys.calib) and \
            bool((sys.model.predict(sys.xv) == healthy.model.predict(sys.xv)).all())
        if diff: cls = "identifiable_not_detected"; witnesses.update(diff)
        elif same_behavior: cls = "no_functional_effect"
        else: cls = "non_identifiable"
        out[cls] += 1; by_fault[fault][cls] += 1
    return {"counts": dict(out), "by_fault": {k: dict(v) for k, v in by_fault.items()}, "witness_measurements": dict(witnesses)}


# ================================================================ crédit actif

def cheap_signs(sys):
    """Signes bon marché (sans intervention) : vérifications de cohérence + composant le plus sensible (gradient)."""
    o = {m: MEASUREMENTS[m][0](sys) for m in ("model_acc", "routing", "unit_check", "prior_range", "threshold_range")}
    g = gradient_credit(sys.model, sys.xv, sys.yv); top = max(g, key=g.get)
    o["grad_top"] = top.split(":")[1] if top.startswith("weights") else "routing"
    return o, sum(MEASUREMENTS[m][1] for m in ("model_acc", "routing", "unit_check", "prior_range", "threshold_range")) + 0.05


class CausePrior:
    """P(cause | signes) — Bayes naïf lissé, appris sur des pannes simulées (graines d'entraînement)."""

    def __init__(self, alpha=0.5): self.alpha = alpha; self.counts = defaultdict(lambda: defaultdict(Counter)); self.n = Counter()

    def fit(self, n_per=10, seed=0):
        rng = random.Random(seed)
        for f in FAULTS:
            for i in range(n_per):
                sys = make_system(7000 + 100 * FAULTS.index(f) + i); inject(sys, f, rng); o, _ = cheap_signs(sys)
                self.n[f] += 1
                for k, v in o.items(): self.counts[f][k][v] += 1
        return self

    def posterior(self, signs):
        lp = {}
        for f in FAULTS:
            s = 0.0
            for k, v in signs.items():
                c = self.counts[f][k]; s += torch.log(torch.tensor((c[v] + self.alpha) / (self.n[f] + self.alpha * (len(c) + 1)))).item()
            lp[f] = s
        m = max(lp.values()); z = sum(torch.exp(torch.tensor(v - m)).item() for v in lp.values())
        return {f: torch.exp(torch.tensor(v - m)).item() / z for f, v in lp.items()}


def _sensitivity_rank(sys):
    """Classement par SENSIBILITÉ seule : gradient pour le neuronal ; |Δscore| sous petite perturbation pour le
    symbolique (paramètre décalé de 10 %)."""
    base = score(sys); g = gradient_credit(sys.model, sys.xv, sys.yv); gmax = max(g.values()) or 1.0
    s = {"weights:enc": g.get("weights:enc", 0) / gmax, "expert:expert_a": g.get("weights:expert_a", 0) / gmax,
         "weights:head": g.get("weights:head", 0) / gmax, "routing": max([v for k, v in g.items() if k.startswith("routing:expert_b")] or [0]) / gmax}
    for name, attr, f in (("measurement", "offset", lambda v: v + 5), ("transformation", "factor", lambda v: v * 1.1), ("belief", "prior_long", lambda v: min(0.999, v + 0.05)),
                          ("planner", "threshold", lambda v: min(0.999, v + 0.05))):
        t = copy.deepcopy(sys); setattr(t.pipe, attr, f(getattr(t.pipe, attr))); s[name] = abs(score(t) - base)
    s["lexical"] = s["source"] = 0.0
    return sorted(FAULTS, key=lambda c: -s[c])


def active_diagnosis(sys, policy, prior: CausePrior = None, k=3, rng=None, restore_frac=0.6, base_ref=None):
    """Renvoie (cause choisie, nb d'interventions, coût des signes)."""
    base = score(sys); target_gain = restore_frac * max(1e-6, (base_ref if base_ref is not None else 1.0) - base)
    def intervene(c): return score(COMPONENTS[c][1](copy.deepcopy(sys))) - base
    if policy == "all":
        gains = {c: intervene(c) for c in FAULTS}; return max(gains, key=gains.get), len(FAULTS), 0.0
    if policy == "random_topk":
        cands = rng.sample(list(FAULTS), k); gains = {c: intervene(c) for c in cands}; return max(gains, key=gains.get), k, 0.0
    if policy == "sensitivity_topk":
        cands = _sensitivity_rank(sys)[:k]; gains = {c: intervene(c) for c in cands}; return max(gains, key=gains.get), k, 0.05
    if policy == "voi":
        signs, cost = cheap_signs(sys); post = prior.posterior(signs); n = 0; best, bestg = None, -1.0
        while post and n < len(FAULTS):
            c = max(post, key=post.get); g = intervene(c); n += 1
            if g > bestg: best, bestg = c, g
            if g >= target_gain: return c, n, cost                    # cause confirmée par intervention
            post.pop(c); z = sum(post.values()) or 1.0; post = {kk: v / z for kk, v in post.items()}
        return best, n, cost
    raise KeyError(policy)


def active_credit_benchmark(n_trials=60, seed=1, prior=None, min_drop=0.03, k=3):
    prior = prior or CausePrior().fit(); rng = random.Random(seed)
    res = {p: Counter() for p in ("all", "random_topk", "sensitivity_topk", "voi")}
    for i in range(n_trials):
        fault = FAULTS[i % len(FAULTS)]; sys = make_system(seed * 1000 + i); healthy = make_system(seed * 1000 + i)
        inject(sys, fault, rng); h = score(healthy)
        if h - score(sys) < min_drop: continue
        for p in res:
            c, n, cost = active_diagnosis(sys, p, prior, k, random.Random(i), base_ref=h)
            r = res[p]; r["n"] += 1; r["correct"] += c == fault; r["interventions"] += n; r["sign_cost"] += cost
    return {p: {"accuracy": round(r["correct"] / max(1, r["n"]), 3), "interventions": round(r["interventions"] / max(1, r["n"]), 2),
                "sign_cost": round(r["sign_cost"] / max(1, r["n"]), 3), "n": r["n"]} for p, r in res.items()}

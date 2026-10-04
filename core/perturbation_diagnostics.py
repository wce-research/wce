"""Diagnostic guidé par PERTURBATIONS VIRTUELLES : avant les diagnostics coûteux, de petites perturbations sont
appliquées à des COPIES du modèle (le modèle diagnostiqué n'est jamais modifié) ; leurs réponses donnent un a priori
sur les causes (table P(profil | cause) APPRISE par simulation sur des graines d'entraînement), puis le
MeasurementPlanner habituel termine. Comparé au planificateur seul (a priori uniforme)."""
from __future__ import annotations

import copy
import random
from collections import Counter

import torch

from core.measurement import CompatibleStateSet, MeasurementPlanner, new_graph
from core.model_diagnostics import CAUSES, DIAGNOSTICS, MODULES, REPAIR, TableOperator, inject, learn_tables, observe, setup
from core.perturbation import Perturbation
from core.stability import perturbed_model

PROBE_COST = 0.01                     # coût d'une perturbation virtuelle (une passe avant sur une copie)


def response(model, xv, yv, modules, mag, seeds=(0, 1)):
    with torch.no_grad():
        base = torch.nn.functional.cross_entropy(model.forward(xv), yv)
        r = [float(torch.nn.functional.cross_entropy(perturbed_model(model, modules, mag, s).forward(xv), yv) - base) for s in seeds]
    return sum(r) / len(r)


def profile(model, healthy_ref, xv, yv, mag=0.05):
    """Profil discret des réponses : pour chaque module, réponse AMPLIFIÉE / normale vs la référence saine ;
    réponse au retrait des poids rapides ; réponse à l'affaiblissement de la connexion la plus forte vers la tête."""
    sig, cost = [], 0.0
    for m in MODULES:
        r = response(model, xv, yv, (m,), mag); cost += PROBE_COST
        sig.append("amp" if r > 3 * healthy_ref[m] + 0.01 else "norm")
    with torch.no_grad():
        base = model.accuracy(xv, yv)
        m2 = copy.deepcopy(model); m2.fast_parameters = {}; cost += PROBE_COST
        sig.append("fast_helps" if m2.accuracy(xv, yv) > base + 0.02 else "fast_neutral")
        incoming = [k for k in model.connectivity if k[1] == "head"]; cost += PROBE_COST * len(incoming)
        best = None
        for k in incoming:
            acc = Perturbation("connection", k, -1.0).apply(model).accuracy(xv, yv)       # retrait virtuel (force × 0)
            if acc > base + 0.02: best = k[0]
        sig.append(f"cut_helps:{best}" if best else "cut_neutral")
    return tuple(sig), cost


def healthy_reference(model, xv, yv, mag=0.05):
    return {m: max(0.0, response(model, xv, yv, (m,), mag)) for m in MODULES}


def learn_profile_table(n_per_cause=6, seed=0):
    rng = random.Random(seed); base, xv, yv, acc = setup(seed); ref = healthy_reference(base, xv, yv)
    counts = {c: Counter() for c in CAUSES}
    for c in CAUSES:
        for _ in range(n_per_cause):
            m = copy.deepcopy(base); inject(m, c, rng); counts[c][profile(m, ref, xv, yv)[0]] += 1
    return counts


def prior_from_profile(counts, sig, alpha=0.5):
    """P(cause | profil) ∝ P(profil | cause) avec lissage (profils jamais vus : quasi uniforme)."""
    vocab = set(s for c in counts.values() for s in c) | {sig}
    lik = {c: (counts[c][sig] + alpha) / (sum(counts[c].values()) + alpha * len(vocab)) for c in CAUSES}
    z = sum(lik.values()); return {c: v / z for c, v in lik.items()}


def diagnose(model, tables, xv, yv, base_acc, prior=None, seed=0, max_steps=6):
    graph = new_graph("diagnostic-perturbation")
    S = CompatibleStateSet.product({("model", "cause"): CAUSES}, graph=graph)
    if prior is not None:
        S = S._fork(S.logw + torch.log(torch.tensor([prior[c] for c in CAUSES], dtype=torch.float64))[S.column(("model", "cause"))], "a priori perturbatif")
    planner = MeasurementPlanner(policy="max_utility", confidence=0.9, budget=1.5, seed=seed, graph=graph, value_of_information=1.0, min_ig=0.01)
    used, cost = 0, 0.0
    while used < max_steps:
        c = planner.choose(S, [TableOperator(d, tables[d], DIAGNOSTICS[d][1]) for d in DIAGNOSTICS], ("model", "cause"))
        if c is None: break
        out = observe(model, c.operator.diag, xv, yv, base_acc)
        L = c.operator.likelihood(out, S.column(("model", "cause")))
        S = S._fork(S.logw + torch.log(L.clamp_min(1e-300)), f"{c.operator.diag}={out}"); used += 1; cost += c.cost
    post = S.marginal(("model", "cause")); cause = max(post, key=post.get)
    return cause, used, cost


def guided_benchmark(n_trials=36, seed=1, tables=None, counts=None):
    tables = tables or learn_tables(seed=0); counts = counts or learn_profile_table(seed=0)
    rng = random.Random(seed); base, xv, yv, acc = setup(seed); ref = healthy_reference(base, xv, yv)
    res = {"planner_only": Counter(), "perturbation_guided": Counter()}
    for i in range(n_trials):
        cause = CAUSES[i % len(CAUSES)]; m = copy.deepcopy(base); inject(m, cause, rng)
        c0, n0, k0 = diagnose(m, tables, xv, yv, acc, None, i)
        sig, pc = profile(m, ref, xv, yv); c1, n1, k1 = diagnose(m, tables, xv, yv, acc, prior_from_profile(counts, sig), i)
        for key, (c, n, k) in (("planner_only", (c0, n0, k0)), ("perturbation_guided", (c1, n1, k1 + pc))):
            r = res[key]; r["n"] += 1; r["correct"] += c == cause; r["measurements"] += n; r["cost"] += k
            fixed = copy.deepcopy(m); fixed.apply(REPAIR[c]()); r["repaired"] += fixed.accuracy(xv, yv) >= acc - 0.02
    return {k: {"cause_accuracy": round(r["correct"] / r["n"], 3), "measurements": round(r["measurements"] / r["n"], 2),
                "cost": round(r["cost"] / r["n"], 3), "repaired": round(r["repaired"] / r["n"], 3)} for k, r in res.items()}

"""Diagnostic du ModelState par la MÊME algèbre de mesure, puis réparation minimale.

panne → hypothèses de cause (ensemble compatible) → mesure diagnostique choisie (gain d'information / coût)
→ postérieure sur les causes → TransformationSpec de réparation MINIMALE ; comparée à des réparations directes.

Observables du modèle : dérive des poids d'un module (vs instantané), routage (connectivité), capacité (unités),
magnitude des poids rapides, effet de l'ablation d'un module, exactitude globale. Les tables P(issue | cause) sont
APPRISES par simulation de pannes sur des graines d'entraînement (le test ne sert qu'à mesurer).
"""
from __future__ import annotations

import copy
import random
from collections import Counter, defaultdict

import torch

from core.measurement import (
    CompatibleStateSet, MeasurementContext, MeasurementOperator, MeasurementOutcome, MeasurementPlanner, NoiseModel,
    Observable, new_graph,
)
from core.model_state import (
    INPUT, ActivateModule, ClearFastWeights, CreateConnection, PruneUnits, RemoveConnection, RestoreModule, RestoreSnapshot,
)
from core.plasticity import build_base_model, pretrain, task_a_labels, task_split

CAUSES = ("corrupt:enc", "corrupt:expert_a", "corrupt:head", "bad_routing", "insufficient_capacity", "fast_drift")
MODULES = ("enc", "expert_a", "head")


# ================================================================ pannes (cause cachée)

def inject(model, cause, rng: random.Random):
    if cause.startswith("corrupt:"):
        m = cause.split(":")[1]
        for k, v in model.modules[m].slow.items():
            model.modules[m].slow[k] = v + rng.uniform(0.8, 1.6) * v.std() * torch.randn(v.shape, generator=torch.Generator().manual_seed(rng.randrange(10 ** 6)))
    elif cause == "bad_routing":                             # un expert dormant est branché à tort sur la tête
        model.apply(ActivateModule(module="expert_b"))
        if (INPUT, "expert_b") not in model.connectivity: model.apply(CreateConnection(source=INPUT, target="expert_b"))
        model.apply(CreateConnection(source="expert_b", target="head", strength=rng.uniform(1.0, 2.0)))
    elif cause == "insufficient_capacity":
        model.apply(PruneUnits(module="expert_a", keep=2))
    elif cause == "fast_drift":
        m = rng.choice(MODULES); d = model.create_fast_delta(m)
        for k, v in model.modules[m].slow.items(): d[k] = rng.uniform(1.0, 2.0) * v.std() * torch.randn(v.shape)


# ================================================================ observables du modèle

def observe(model, name, x, y, base_acc):
    """Issue discrète d'une mesure diagnostique (lit le MODÈLE, jamais la cause cachée)."""
    snap = model.snapshots["good"]
    if name.startswith("drift:"):
        m = name.split(":")[1]
        if m not in model.modules or m not in snap.modules: return "large"
        cur, ref = model.modules[m].slow, snap.modules[m].slow
        if any(cur[k].shape != ref[k].shape for k in ref): return "shape"
        rel = sum(float((cur[k] - ref[k]).norm()) for k in ref) / max(1e-9, sum(float(ref[k].norm()) for k in ref))
        return "none" if rel < 1e-6 else "small" if rel < 0.3 else "large"
    if name == "routing":
        return "ok" if set(model.connectivity) == set(snap.connectivity) else "changed"
    if name == "capacity":
        return "ok" if all(model.modules[m].hidden == snap.modules[m].hidden for m in snap.modules if m in model.modules) else "reduced"
    if name == "fast":
        mag = sum(float(t.norm()) for d in model.fast_parameters.values() for t in d.values())
        return "large" if mag > 1e-3 else "ok"
    if name.startswith("ablation:"):
        m = name.split(":")[1]
        override = {m: {k: v for k, v in snap.modules[m].slow.items()}} if m in snap.modules else None
        try:
            with torch.no_grad(): acc = float((model.forward(x, override).argmax(1) == y).float().mean())
        except Exception: return "error"
        return "recovers" if acc >= base_acc - 0.03 else "no"
    if name == "accuracy":
        acc = model.accuracy(x, y)
        return "ok" if acc >= base_acc - 0.03 else "degraded"
    raise KeyError(name)


DIAGNOSTICS = {  # nom : (issues possibles, coût)
    **{f"drift:{m}": (("none", "small", "large", "shape"), 0.05) for m in MODULES},
    "routing": (("ok", "changed"), 0.02), "capacity": (("ok", "reduced"), 0.02), "fast": (("ok", "large"), 0.02),
    **{f"ablation:{m}": (("recovers", "no", "error"), 0.3) for m in MODULES},
    "accuracy": (("ok", "degraded"), 0.2),
}


class TableOperator(MeasurementOperator):
    """Opérateur diagnostique : P(issue | cause) = table APPRISE (fréquences lissées)."""

    def __init__(self, name, table, cost):
        outs = DIAGNOSTICS[name][0]
        super().__init__(Observable(name, "model", "cause", CAUSES), MeasurementContext(sensor=name, cost=cost, domain="model"))
        self.diag, self.table, self.outs = name, table, outs

    def outcomes(self, grid=None): return list(self.outs)

    def likelihood(self, outcome, values):
        o = outcome.value if isinstance(outcome, MeasurementOutcome) else outcome
        probs = torch.tensor([self.table[c][o] for c in CAUSES], dtype=torch.float64)
        return probs[values]


def setup(seed=0):
    x, y, xv, yv = task_split(task_a_labels, seed=seed)
    model = build_base_model(seed=seed); pretrain(model, x, y, steps=250, seed=seed)
    model.snapshot("good")
    return model, xv, yv, model.accuracy(xv, yv)


def learn_tables(n_per_cause=6, seed=0):
    """Simulations de pannes sur des graines d'ENTRAÎNEMENT -> P(issue | cause) (lissage de Laplace)."""
    rng = random.Random(seed)
    counts = {c: {d: Counter() for d in DIAGNOSTICS} for c in CAUSES}
    base, xv, yv, acc = setup(seed)
    for c in CAUSES:
        for i in range(n_per_cause):
            m = copy.deepcopy(base); inject(m, c, rng)
            for d in DIAGNOSTICS: counts[c][d][observe(m, d, xv, yv, acc)] += 1
    tables = {}
    for d, (outs, cost) in DIAGNOSTICS.items():
        tables[d] = {c: {o: (counts[c][d][o] + 0.5) / (n_per_cause + 0.5 * len(outs)) for o in outs} for c in CAUSES}
    return tables


REPAIR = {
    "corrupt:enc": lambda: RestoreModule(module="enc", snapshot="good"),
    "corrupt:expert_a": lambda: RestoreModule(module="expert_a", snapshot="good"),
    "corrupt:head": lambda: RestoreModule(module="head", snapshot="good"),
    "bad_routing": lambda: RemoveConnection(source="expert_b", target="head"),
    "insufficient_capacity": lambda: RestoreModule(module="expert_a", snapshot="good"),
    "fast_drift": lambda: ClearFastWeights(),
}


def diagnose_and_repair(model, tables, xv, yv, base_acc, policy="max_utility", seed=0, max_steps=6):
    """Boucle : hypothèses -> diagnostic -> postérieure -> réparation minimale (TransformationSpec)."""
    graph = new_graph("diagnostic")
    S = CompatibleStateSet.product({("model", "cause"): CAUSES}, graph=graph)
    planner = MeasurementPlanner(policy=policy, confidence=0.9, budget=1.5, seed=seed, graph=graph, value_of_information=1.0, min_ig=0.01)
    used, cost, log = 0, 0.0, []
    while used < max_steps:
        ops = [TableOperator(d, tables[d], DIAGNOSTICS[d][1]) for d in DIAGNOSTICS]
        c = planner.choose(S, ops, ("model", "cause"))
        if c is None: break
        out = observe(model, c.operator.diag, xv, yv, base_acc)
        L = c.operator.likelihood(out, S.column(("model", "cause")))
        S = S._fork(S.logw + torch.log(L.clamp_min(1e-300)), f"{c.operator.diag}={out}")
        used += 1; cost += c.cost; log.append((c.operator.diag, out))
    post = S.marginal(("model", "cause"))
    cause, p = max(post.items(), key=lambda kv: kv[1])
    before = sum(t.numel() for m in model.modules.values() for t in m.slow.values())
    spec = REPAIR[cause]()
    try: model.apply(spec); applied = spec.name
    except Exception as e: applied = f"échec : {e}"
    return {"cause": cause, "p": round(p, 3), "measurements": used, "cost": round(cost, 3), "log": log, "repair": applied,
            "accuracy": model.accuracy(xv, yv), "posterior": post, "graph_nodes": len(graph.nodes)}


def direct_repair(model, policy, xv, yv):
    """Politiques DIRECTES (sans diagnostic) : restaurer tout l'instantané, ou toujours restaurer le module expert."""
    if policy == "restore_all": model.apply(RestoreSnapshot(snapshot="good"))
    elif policy == "restore_expert": model.apply(RestoreModule(module="expert_a", snapshot="good"))
    return model.accuracy(xv, yv)


def repair_benchmark(n_trials=60, seed=1, tables=None):
    tables = tables or learn_tables(seed=0)
    rng = random.Random(seed)
    base, xv, yv, acc = setup(seed)
    out = defaultdict(Counter)
    for i in range(n_trials):
        cause = CAUSES[i % len(CAUSES)]
        broken = copy.deepcopy(base); inject(broken, cause, rng)
        for pol in ("max_utility", "random", "fixed"):
            m = copy.deepcopy(broken)
            r = diagnose_and_repair(m, tables, xv, yv, acc, pol, seed=i)
            o = out[f"diagnose_{pol}"]
            o["n"] += 1; o["correct_cause"] += r["cause"] == cause; o["repaired"] += r["accuracy"] >= acc - 0.02
            o["measurements"] += r["measurements"]; o["cost"] += r["cost"]
            o["minimal"] += r["repair"] in ("restore_module", "remove_connection", "clear_fast_weights")
        for pol in ("restore_all", "restore_expert"):
            m = copy.deepcopy(broken); a = direct_repair(m, pol, xv, yv)
            o = out[pol]; o["n"] += 1; o["repaired"] += a >= acc - 0.02; o["minimal"] += pol != "restore_all"
    res = {}
    for k, c in out.items():
        n = c["n"]
        res[k] = {"repaired": round(c["repaired"] / n, 3), "minimal_repair": round(c["minimal"] / n, 3)}
        if k.startswith("diagnose"):
            res[k].update({"cause_accuracy": round(c["correct_cause"] / n, 3), "measurements": round(c["measurements"] / n, 2),
                           "cost": round(c["cost"] / n, 3)})
    res["base_accuracy"] = round(acc, 3)
    return res

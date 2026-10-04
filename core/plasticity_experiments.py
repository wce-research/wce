"""Expériences ciblées de la plasticité compositionnelle : synthèse de sondes (I), sous-espaces orthogonaux /
espace nul (J, K), régénération sans instantané (AJ), économie de transformations (AK), compression de
l'historique d'apprentissage (AL), décision de croissance et réutilisation future (N, P), macros (S, T)."""
from __future__ import annotations

import copy
import random
from collections import Counter

import torch

from core.compositional_plasticity import (PlannerConfig, PlasticityPlanner, StrategyLibrary, induce_macros, regenerate)
from core.multitask_benchmark import D_CUE, mono_acc, mono_forward, mono_init, mono_train, jac_basis, probes_for
from core.skillnet import ALL_DATA, SkillNet, skill_modules, task_stream


# ================================================================ I — synthèse de sondes vs replay complet

def probe_synthesis_experiment(seed=0, ks=(0, 4, 8, 16, 32, 64), steps=250):
    """Ancienne tâche A puis nouvelle B (réseau monolithique) ; protection de A par k sondes SYNTHÉTISÉES
    (sorties de référence, aucune donnée brute) vs k sondes ALÉATOIRES vs replay de TOUTES les données de A."""
    tasks = task_stream(seed + 50, 2); rng = random.Random(seed); A, B = tasks
    Xa, Ca, ya = A.sample(128, rng); Xb, Cb, yb = B.sample(128, rng); ta, tb = A.test_set(1), B.test_set(2)
    p = mono_train(mono_init(seed), torch.cat([Xa, Ca], 1), ya, steps, seed=seed); proto = Ca.mean(0).round()
    out = {"reference_A": round(mono_acc(p, *ta), 4)}
    for k in ks:
        row = {}
        for kind in ("synthesized", "random"):
            if k == 0: anchors = None
            else:
                if kind == "synthesized": Zp = probes_for(p, proto, k)
                else:
                    g = torch.Generator().manual_seed(seed); Zp = torch.cat([ALL_DATA[torch.randint(0, 256, (k,), generator=g)], proto.expand(k, D_CUE)], 1)
                with torch.no_grad(): anchors = (Zp, mono_forward(p, Zp))
            q = mono_train(p, torch.cat([Xb, Cb], 1), yb, steps, seed=seed, anchors=anchors)
            row[kind] = {"retention_A": round(mono_acc(q, *ta), 4), "new_B": round(mono_acc(q, *tb), 4)}
        out[f"k={k}"] = row
    q = mono_train(p, torch.cat([Xb, Cb], 1), yb, steps, seed=seed, replay=(torch.cat([Xa, Ca], 1), ya))
    out["full_replay(128)"] = {"retention_A": round(mono_acc(q, *ta), 4), "new_B": round(mono_acc(q, *tb), 4)}
    return out


# ================================================================ J/K — gradient vanilla vs projeté vs contraint vs croissance

def subspace_experiment(seed=0, steps=250, k=32):
    tasks = task_stream(seed + 60, 2); rng = random.Random(seed); A, B = tasks
    Xa, Ca, ya = A.sample(128, rng); Xb, Cb, yb = B.sample(128, rng); ta, tb = A.test_set(1), B.test_set(2)
    p = mono_train(mono_init(seed), torch.cat([Xa, Ca], 1), ya, steps, seed=seed); proto = Ca.mean(0).round()
    Zp = probes_for(p, proto, k); basis = jac_basis(p, Zp)
    with torch.no_grad(): anchors = (Zp, mono_forward(p, Zp))
    Zb = torch.cat([Xb, Cb], 1); out = {"null_space_rank": int(basis.shape[1]), "parameters": int(sum(v.numel() for v in p.values()))}
    for name, kw in (("vanilla", {}), ("projected_null_space", {"basis": basis}), ("invariant_constrained", {"anchors": anchors})):
        q = mono_train(p, Zb, yb, steps, seed=seed, **kw)
        v0 = torch.cat([p[kk].flatten() for kk in p]); v1 = torch.cat([q[kk].flatten() for kk in q])
        out[name] = {"retention_A": round(mono_acc(q, *ta), 4), "new_B": round(mono_acc(q, *tb), 4),
                     "J_old·Δθ": round(float((basis.T @ (v1 - v0)).norm() / (v1 - v0).norm().clamp_min(1e-12)), 4), "params_added": 0}
    net = SkillNet(seed=seed); certs = {}; pl = PlasticityPlanner(PlannerConfig(allow_reuse=False, allow_compose=False, allow_adapter=False), seed=seed)
    for t, (X, C, y) in ((A, (Xa, Ca, ya)), (B, (Xb, Cb, yb))): pl.learn(net, X, C, y, certs)
    out["structural_growth"] = {"retention_A": round(net.accuracy(*ta), 4), "new_B": round(net.accuracy(*tb), 4), "params_added": net.parameter_count() // 2}
    return out


# ================================================================ AJ — régénération sans instantané

def regeneration_experiment(seed=0, n_tasks=10):
    net = SkillNet(seed=seed); certs = {}; pl = PlasticityPlanner(seed=seed); rng = random.Random(seed); tasks = task_stream(seed, n_tasks)
    for t in tasks: X, C, y = t.sample(128, rng); pl.learn(net, X, C, y, certs)
    tests = [t.test_set(100 + i) for i, t in enumerate(tasks)]
    base = sum(net.accuracy(*ts) for ts in tests) / len(tests)
    res = Counter(); per = []
    for m in [m for m in net.modules if "w1" in net.modules[m]]:
        n2 = copy.deepcopy(net); g = torch.Generator().manual_seed(len(m))
        n2.modules[m] = {k: v + 1.5 * v.std() * torch.randn(v.shape, generator=g) for k, v in n2.modules[m].items()}
        broken = sum(n2.accuracy(*ts) for ts in tests) / len(tests)
        c2 = {s: c for s, c in certs.items()}
        r = regenerate(n2, c2, m, seed=seed); after = sum(n2.accuracy(*ts) for ts in tests) / len(tests)
        res["modules"] += 1; res["certs_failing"] += len(r["failing"]); res["certs_repaired"] += len(r["failing"]) - len(r["still_failing"])
        per.append({"module": m, "before": round(base, 3), "corrupted": round(broken, 3), "regenerated": round(after, 3)})
    mean = lambda k: round(sum(p[k] for p in per) / max(1, len(per)), 4)
    return {"modules": res["modules"], "certificate_repair_rate": round(res["certs_repaired"] / max(1, res["certs_failing"]), 3),
            "accuracy_before": mean("before"), "accuracy_corrupted": mean("corrupted"), "accuracy_regenerated": mean("regenerated"), "per_module": per}


# ================================================================ AK/AL/S/T — économie, compression, macros, bibliothèque

def economy_and_compression(seed=0, n_tasks=16):
    tasks = task_stream(seed, n_tasks); rng = random.Random(seed)
    net = SkillNet(seed=seed); certs = {}; lib = StrategyLibrary(); pl = PlasticityPlanner(library=lib, seed=seed)
    raw_bytes = 0
    for t in tasks:
        X, C, y = t.sample(128, rng); raw_bytes += X.numel() // 8 + C.numel() // 8 + y.numel() // 8 + 1
        pl.learn(net, X, C, y, certs)
    fams = Counter(f for p in pl.programs for f in p.signature())
    macros = induce_macros(pl.programs)
    stored = sum(c.memory_bytes() for c in certs.values()) + sum(16 * len(m.template) for m in macros)
    grown = [r for r in pl.log if r["kind"] == "grow"]
    later_reuse = sum(1 for s, prog in net.skills.items() if prog[0] in ("compose", "not", "adapter") for g in grown if f"skill_{g['context']}" in str(prog))
    return {"families_used": dict(fams), "n_families_used": len(fams), "n_families_available": 15,
            "programs": len(pl.programs), "macros": [(m.name, m.support, len(m.variables)) for m in macros],
            "history_raw_bytes": raw_bytes, "history_stored_bytes": stored, "compression_ratio": round(raw_bytes / max(1, stored), 2),
            "grown": len(grown), "grown_later_reused": later_reuse, "library_entries": lib.size()}


def library_ablation(seed=0, n_tasks=16):
    """Bibliothèque de stratégies : nombre d'évaluations de candidats et calcul, avec / sans."""
    out = {}
    for use in (False, True):
        tasks = task_stream(seed, n_tasks); rng = random.Random(seed); net = SkillNet(seed=seed); certs = {}
        pl = PlasticityPlanner(PlannerConfig(use_library=use), library=StrategyLibrary(), seed=seed)
        for t in tasks: X, C, y = t.sample(128, rng); pl.learn(net, X, C, y, certs)
        out["with_library" if use else "without_library"] = {"evaluations": pl.evaluations, "train_steps": net.counters["train_steps"],
                                                             "accuracy": round(sum(net.accuracy(*t.test_set(100 + i)) for i, t in enumerate(tasks)) / len(tasks), 4)}
    return out

"""Compositional plasticity benchmark: multi-seed (16) and sealed set (evaluated once)."""
from __future__ import annotations

import hashlib
import json

from core.multitask_benchmark import METHODS, run
from core.skillnet import task_stream

KEYS = ("average_accuracy", "new_task", "retention", "forgetting", "backward_transfer", "forward_transfer", "param_growth", "memory_bytes", "compute_steps")


def sealed_streams(): return [task_stream(990100 + i, 20) for i in range(3)]


def sealed_hash():
    return hashlib.sha256(json.dumps([[(t.name, t.kind, repr(t.spec), t.cue.tolist()) for t in s] for s in sealed_streams()], sort_keys=True).encode()).hexdigest()


SEALED_SHA256 = "a4b07759504a4857c2f643318fdeb235deab3780ecc094d88b6f141dc9410182"   # frozen BEFORE any evaluation


def multi_seed(seeds=range(16), n_tasks=14, log=print):
    from core.active_credit import CausePrior, active_credit_benchmark
    from core.grounded_benchmark import paired, summarize_values
    from core.plasticity_experiments import economy_and_compression, probe_synthesis_experiment, regeneration_experiment, subspace_experiment
    from core.symbolic_plasticity import cross_domain
    prior = CausePrior().fit(); per = []
    for s in seeds:
        tasks = task_stream(1000 + s, n_tasks)
        r = {"multitask": {m: run(m, tasks, seed=s) for m in METHODS}, "credit": active_credit_benchmark(40, seed=300 + s, prior=prior),
             "cross_domain": cross_domain(500 + s), "probes": probe_synthesis_experiment(s, ks=(4, 8, 16, 32)), "subspace": subspace_experiment(s),
             "regeneration": {k: v for k, v in regeneration_experiment(s).items() if k != "per_module"}, "economy": economy_and_compression(s)}
        per.append(r)
        log(s, {m: (r["multitask"][m]["average_accuracy"], r["multitask"][m]["forgetting"]) for m in METHODS}, r["credit"]["voi"])
    S = lambda f: summarize_values([f(r) for r in per])
    summary = {"multitask": {m: {k: S(lambda r, m=m, k=k: r["multitask"][m][k]) for k in KEYS} for m in METHODS},
               "credit": {p: {k: S(lambda r, p=p, k=k: r["credit"][p][k]) for k in ("accuracy", "interventions")} for p in per[0]["credit"]},
               "cross_domain": {d: {k: S(lambda r, d=d, k=k: r["cross_domain"][d][k]) for k in ("generalizes", "retention", "evaluations")} for d in per[0]["cross_domain"]},
               "probes": {k: {kind: S(lambda r, k=k, kind=kind: r["probes"][k][kind]["retention_A"]) for kind in ("synthesized", "random")} for k in ("k=4", "k=8", "k=16", "k=32")},
               "full_replay": S(lambda r: r["probes"]["full_replay(128)"]["retention_A"]),
               "subspace": {m: {k: S(lambda r, m=m, k=k: r["subspace"][m][k]) for k in ("retention_A", "new_B")} for m in ("vanilla", "projected_null_space", "invariant_constrained", "structural_growth")},
               "regeneration": {k: S(lambda r, k=k: r["regeneration"][k]) for k in ("certificate_repair_rate", "accuracy_before", "accuracy_corrupted", "accuracy_regenerated")},
               "economy": {k: S(lambda r, k=k: r["economy"][k]) for k in ("n_families_used", "compression_ratio", "grown", "grown_later_reused")}}
    reuse = {k: S(lambda r, k=k: r["multitask"]["compositional"].get("kinds", {}).get(k, 0)) for k in ("reactivate", "reuse", "compose", "adapter", "grow", "refine_inplace")}
    summary["compositional_kinds"] = reuse
    comps = {f"compositional_vs_{b}": {k: paired([r["multitask"]["compositional"][k] for r in per], [r["multitask"][b][k] for r in per])
                                       for k in ("average_accuracy", "new_task", "forgetting", "param_growth")} for b in METHODS if b != "compositional"}
    comps["voi_vs_all_interventions"] = paired([r["credit"]["voi"]["interventions"] for r in per], [r["credit"]["all"]["interventions"] for r in per])
    return {"summary": summary, "comparisons": comps, "per_seed": per}


def sealed_eval():
    from core.active_credit import CausePrior, active_credit_benchmark
    from core.symbolic_plasticity import cross_domain
    out = {"multitask": [], "credit": active_credit_benchmark(60, seed=990, prior=CausePrior().fit()), "cross_domain": cross_domain(990)}
    for i, tasks in enumerate(sealed_streams()):
        out["multitask"].append({m: {k: v for k, v in run(m, tasks, seed=990 + i).items() if k in KEYS or k in ("kinds", "recurrence")} for m in METHODS})
    return out

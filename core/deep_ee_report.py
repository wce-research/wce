"""Reports of the sub-milestone End-to-End Composition with Discovered Atoms (dev only; never the sealed set).

  tune      : configuration of chain C on the TUNING keys (990–994) → results/deep_configs/e2e.json
  smoke     : short local demonstration (key 999)
  aggregate : aggregates the AWS runs `deepee` → results/deep_end_to_end_summary.json
  report    : prints the tables
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

from core.deep_rp_report import _mean, paired_ci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
SUMMARY = os.path.join(OUT, "deep_end_to_end_summary.json")
CONFIG = os.path.join(OUT, "deep_configs", "e2e.json")


def load_rows():
    rows = []; bad = 0
    for f in glob.glob(os.path.join(OUT, "aws", "wce-deepee-*", "artifacts", "*", "deepee_*.json")):
        r = json.load(open(f))
        if r["status"] == "ok": rows += r["result"]
        else: bad += 1
    return rows, bad


def summarize(rows, bad=0):
    S = {"n_rows": len(rows), "units_in_error": bad}; depths = sorted({r["depth"] for r in rows if r["experiment"] == "chain"}); groups = [("all", lambda r: True), ("d3-6", lambda r: r["depth"] >= 3), ("d4-6", lambda r: r["depth"] >= 4)] + [(f"d{d}", (lambda r, d=d: r["depth"] == d)) for d in depths]
    K = ("accord", "accord_in", "accord_out", "functional", "functional_097", "exact", "class_equal", "obs_agreement", "candidate_099", "candidate_097", "candidate_best", "size", "literals", "kept", "false_kept", "false_used", "atom_recall", "atom_recall_literals", "atom_recall_kept",
         "generated_099", "generated_097", "top5_099", "top20_099", "prefit_accord", "prefit_functional", "fit_committed", "classes", "considered", "unseen_mass", "pool", "fit_equivalents", "seconds", "discover_seconds", "n_alternatives")
    for exp in ("chain", "source", "reduce", "robust"):
        x = [r for r in rows if r["experiment"] == exp]; arms = sorted({r["arm"] for r in x}); S[exp] = {f"{g}/{a}": {"n": sum(f(r) and r["arm"] == a for r in x), **{k: _mean(r.get(k) for r in x if f(r) and r["arm"] == a) for k in K}} for g, f in groups for a in arms}
        by = {a: {r["task_id"]: r for r in x if r["arm"] == a} for a in arms}; S[f"{exp}_paired"] = {}
        refs = {"chain": ("A", "B", "C"), "source": ("flat_scene_k3",), "reduce": ("greedy_floor",), "robust": ("true",)}[exp]
        for ref in refs:
            for a in arms:
                if a == ref or ref not in by: continue
                for g, f in groups[:3] if exp == "chain" else groups[:1]:
                    ts = sorted(t for t in set(by[a]) & set(by[ref]) if f(by[a][t])); S[f"{exp}_paired"][f"{g}: {a} − {ref}"] = {k: paired_ci([by[ref][t].get(k) for t in ts], [by[a][t].get(k) for t in ts]) for k in ("accord", "functional", "functional_097", "candidate_097", "accord_out")}
    ch = [r for r in rows if r["experiment"] == "chain"]; c = [r for r in ch if r["arm"] == "C"]
    S["causes"] = {g: dict(Counter(r["cause"] for r in c if f(r) and r.get("cause"))) for g, f in groups}; S["labels_in_class"] = {g: dict(Counter(r["label"] for r in c if f(r))) for g, f in groups[:1] + groups[3:]}
    S["label_vs_outcome"] = {lab: {"n": sum(r["label"] == lab for r in c), "functional": _mean(r["functional"] for r in c if r["label"] == lab), "accord": _mean(r["accord"] for r in c if r["label"] == lab), "gap": _mean(r["gap"] for r in c if r["label"] == lab)} for lab in sorted({r["label"] for r in c})}
    rb = [r for r in rows if r["experiment"] == "robust"]; S["robust_labels"] = {a: dict(Counter(r["label"] for r in rb if r["arm"] == a and r.get("applicable", True))) for a in sorted({r["arm"] for r in rb})}
    bd = [r for r in rows if r["experiment"] == "boundary"]; S["boundary"] = {a: {"n": sum(r["arm"] == a for r in bd), "labels": dict(Counter(r["label"] for r in bd if r["arm"] == a)), "gap": _mean(r["gap"] for r in bd if r["arm"] == a), "accord": _mean(r["accord"] for r in bd if r["arm"] == a)} for a in sorted({r["arm"] for r in bd})}
    S["boundary_all"] = {"n": len(bd), "labels": dict(Counter(r["label"] for r in bd))}
    return S


def aggregate():
    rows, bad = load_rows(); S = summarize(rows, bad); json.dump(S, open(SUMMARY, "w"), indent=1); print("rows", len(rows), "units in error", bad); return S


def show(S):
    ci = lambda v: {x: (y["diff"], y["ci95"]) if y else None for x, y in v.items()}; cut = lambda v, ks: {k: v.get(k) for k in ks if v.get(k) is not None}
    print("\nCHAINS"); [print(k.ljust(20), cut(v, ("n", "accord", "accord_out", "functional", "functional_097", "candidate_097", "exact", "class_equal", "literals", "kept", "false_used", "atom_recall", "atom_recall_literals", "atom_recall_kept", "generated_099", "top20_099", "prefit_accord", "fit_committed", "classes", "seconds"))) for k, v in S["chain"].items()]
    print("\npaired"); [print(k.ljust(30), ci(v)) for k, v in S["chain_paired"].items()]; print("\ncauses (chain C)", S["causes"]); print("in-class decisions", S["labels_in_class"]); print("decision → outcome", S["label_vs_outcome"])
    print("\nSOURCES"); [print(k.ljust(24), cut(v, ("n", "accord", "functional", "functional_097", "literals", "kept", "atom_recall_literals", "atom_recall_kept", "false_kept", "generated_099", "seconds"))) for k, v in S["source"].items() if k.startswith(("all/", "d4-6/"))]; [print(k.ljust(36), ci(v)) for k, v in S["source_paired"].items()]
    print("\nREDUCTION AND SCALING"); [print(k.ljust(24), cut(v, ("n", "accord", "functional", "functional_097", "kept", "atom_recall_literals", "atom_recall_kept", "classes", "considered", "seconds"))) for k, v in S["reduce"].items() if k.startswith(("all/", "d4-6/"))]; [print(k.ljust(36), ci(v)) for k, v in S["reduce_paired"].items()]
    print("\nCONTROLLED CASES"); [print(k.ljust(22), cut(v, ("n", "accord", "accord_out", "functional", "functional_097", "kept", "false_kept", "false_used"))) for k, v in S["robust"].items() if k.startswith(("all/", "d4-6/"))]; print("decisions", S["robust_labels"])
    print("\nCLASS BOUNDARY"); [print(k.ljust(20), v) for k, v in S["boundary"].items()]; print(S["boundary_all"])


def report(): show(json.load(open(SUMMARY)))


def _tasks(keys, regimes=("standard", "rich")):
    from core.deep_e2e_exp import DEPTHS, task_at
    return [(f"{k}_{d}_{r}", d, r, task_at(k, d, r)) for k in keys for d in DEPTHS for r in regimes]


def tune():
    """Configuration of chain C on the tuning keys 990–994 (60 tasks). Criterion fixed in advance: mean
    final agreement. The discovery is done once per task, then reused for all the configurations."""
    import itertools
    from core.deep_e2e import DEFAULTS, discover, run_chain
    from core.deep_e2e_exp import _acc, _regions
    T = _tasks(range(990, 995)); disc = {tid: {k: discover(t.X, t.y, k, 0) for k in (3,)} for tid, _, _, t in T}; reg = {tid: _regions(t) for tid, _, _, t in T}; res = []
    for mg, fit, sel in itertools.product((1.0, 2.0, 4.0), ("none", "validated", "forced"), ("bic", "errors")):
        cfg = {**DEFAULTS, "min_gain": mg, "fit": fit, "selector": sel}; acc = [_acc(run_chain(t.X, t.y, 0, cfg, disc[tid][3]).program, reg[tid]) for tid, _, _, t in T]
        res.append({"min_gain": mg, "fit": fit, "selector": sel, "accord": _mean(a["accord"] for a in acc), "functional_097": _mean(a["functional_097"] for a in acc), "accord_d4_6": _mean(a["accord"] for a, (_, d, _, _) in zip(acc, T) if d >= 4)}); print(res[-1], flush=True)
    best = max(res, key=lambda r: r["accord"]); cfg = {"min_gain": best["min_gain"], "fit": best["fit"], "selector": best["selector"]}
    json.dump({"config": cfg, "selected_on": f"{len(T)} tuning tasks (keys 990–994)", "criterion": "mean final agreement", "grid": res}, open(CONFIG, "w"), indent=1, ensure_ascii=False); print("SELECTED", cfg, best)


def smoke():
    from core.deep_e2e_exp import C_VARIANTS, OOC, ROBUST, SOURCES, boundary_row, chain_ab, chain_c, ooc_task, reduce_rows, robust_row, source_row
    rows = []; ds = [int(x) for x in os.environ.get("EE_DEPTHS", "2,4").split(",")]
    for tid, d, r, t in _tasks([999], ("standard",)):
        if d not in ds: continue
        tag = {"task_id": tid, "depth": d, "regime": r}; rows += [{**tag, **chain_c(t, v)} for v in C_VARIANTS] + [{**tag, **source_row(t, s)} for s in SOURCES] + [{**tag, **x} for x in reduce_rows(t)] + [{**tag, **robust_row(t, c)} for c in ROBUST]
        if os.environ.get("EE_AB"): rows += [{**tag, **chain_ab(t, a)} for a in ("A", "B")]
    rows += [{"task_id": f"999_{c}", "depth": 0, "regime": "standard", **boundary_row(ooc_task(999, c), c)} for c in OOC]; show(summarize(rows))


if __name__ == "__main__":
    {"aggregate": aggregate, "report": report, "smoke": smoke, "tune": tune}[sys.argv[1]]()

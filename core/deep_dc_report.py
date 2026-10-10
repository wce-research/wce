"""Reports of the Deep Composition & Final Reconstruction with Oracle Atoms sub-milestone (dev only).

  smoke     : short local demonstration on the TUNING key (899)
  aggregate : aggregates the AWS `deepdc` runs → results/deep_composition_oracle_summary.json
  report    : prints the tables
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict

from core.deep_rp_report import _mean, paired_ci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
SUMMARY = os.path.join(OUT, "deep_composition_oracle_summary.json")


def load_rows():
    rows = []; bad = 0
    for f in glob.glob(os.path.join(OUT, "aws", "wce-deepdc-*", "artifacts", "*", "deepdc_*.json")):
        r = json.load(open(f))
        if r["status"] == "ok": rows += r["result"]
        else: bad += 1
    return rows, bad


def _median(xs):
    xs = sorted(float(x) for x in xs if x is not None); return xs[len(xs) // 2] if xs else None


def summarize(rows, bad=0):
    from core.deep_compose import SELECTORS
    from core.deep_compose_exp import BEAMS, BUDGETS
    S = {"n_rows": len(rows), "units_in_error": bad}; co = [r for r in rows if r["experiment"] == "compose"]; rf = [r for r in rows if r["experiment"] == "ref"]; tr = [r for r in rows if r["experiment"] == "trace"]; depths = sorted({r["n_atoms"] for r in co})
    groups = [("all", lambda r: True)] + [(f"d{d}", (lambda r, d=d: r["n_atoms"] == d)) for d in depths]
    # ---- space and dynamic programming
    S["space"] = {g: {"n": sum(f(r) for r in co), **{k: _mean(r.get(k) for r in co if f(r)) for k in ("cells", "cells_extended", "dp_classes", "classes_extended", "classes_merged_by_observation", "dp_considered", "dp_seconds", "unseen_mass_full", "unseen_mass_out", "noise_rate", "truth_generated", "truth_is_majority", "classes_better_or_equal", "classes_strictly_better", "classes_quasi_equivalent", "seconds")},
                         "dp_classes_max": max((r["dp_classes"] for r in co if f(r)), default=None), "dp_considered_max": max((r["dp_considered"] for r in co if f(r)), default=None), "dp_seconds_max": max((r["dp_seconds"] for r in co if f(r)), default=None)} for g, f in groups}
    sc = defaultdict(lambda: defaultdict(list))
    for r in co:
        for l in r["scaling"]:
            for k in ("programs", "distinct_masks", "new_classes", "cumulative_classes"): sc[(r["n_atoms"], l["size"])][k].append(l[k])
    S["scaling"] = {f"n{n}/size{s}": {k: _mean(v) for k, v in d.items()} for (n, s), d in sorted(sc.items())}
    O = ("functional", "functional_097", "exact", "class_equal", "accord", "accord_in", "accord_out", "compact", "size", "errors", "obs_agreement")
    S["selectors"] = {f"{g}/{sel}": {"rank_median": _median(r.get(f"rank_{sel}") for r in co if f(r)), "rank_mean": _mean(r.get(f"rank_{sel}") for r in co if f(r)), **{f"top{k}": _mean(r.get(f"top{k}_{sel}") for r in co if f(r)) for k in (1, 5, 20)}, **{k: _mean(r.get(f"dp_{sel}_{k}") for r in co if f(r)) for k in O}} for g, f in groups for sel in SELECTORS}
    S["plus_one"] = {f"{g}/{a}": {k: _mean(r.get(f"dp_{a}_plus1_{k}") for r in co if f(r)) for k in ("functional", "exact", "class_equal", "accord", "size")} for g, f in groups for a in ("errors", "bic")}
    S["beam"] = {f"{g}/beam{b}": {"truth": _mean(r.get(f"beam{b}_truth") for r in co if f(r)), "class_equal": _mean(r.get(f"beam{b}_class_equal") for r in co if f(r)), "accord": _mean(r.get(f"beam{b}_accord") for r in co if f(r)), "considered": _mean(r.get(f"beam{b}_considered") for r in co if f(r))} for g, f in groups for b in BEAMS}
    S["budget"] = {f"{g}/budget{b}": {"truth": _mean(r.get(f"budget{b}_truth") for r in co if f(r)), "class_equal": _mean(r.get(f"budget{b}_class_equal") for r in co if f(r)), "accord": _mean(r.get(f"budget{b}_accord") for r in co if f(r))} for g, f in groups for b in BUDGETS}
    S["mitm"] = {f"{g}/half{h}": {**{k: _mean(r.get(f"mitm{h}_{k}") for r in co if f(r)) for k in ("functional", "class_equal", "exact", "accord", "matches_exhaustive", "joins", "table", "seconds")}} for g, f in groups for h in (2, 3)}
    S["fit"] = {f"{g}/{w}": {**{v: _mean(r[f"fit_{w}"][v] for r in co if f(r)) for v in ("A", "B", "C", "D")}, **{f"{v}_099": _mean(r[f"fit_{w}"][v] >= 0.99 for r in co if f(r)) for v in ("A", "B", "C", "D")}, "D_committed": _mean(r[f"fit_{w}"]["D_committed"] for r in co if f(r)),
                              "errors_A": _mean(r[f"fit_{w}"]["errors_A"] for r in co if f(r)), "errors_C": _mean(r[f"fit_{w}"]["errors_C"] for r in co if f(r))} for g, f in groups for w in ("selected", "truth")}
    S["fit_paired"] = {f"{w}: {b} − A": {"accord": paired_ci([r[f"fit_{w}"]["A"] for r in co], [r[f"fit_{w}"][b] for r in co]), "recovery_099": paired_ci([r[f"fit_{w}"]["A"] >= 0.99 for r in co], [r[f"fit_{w}"][b] >= 0.99 for r in co])} for w in ("selected", "truth") for b in ("B", "C", "D")} if co else {}
    shapes = sorted({r["truth_shape"] for r in co}); S["shape"] = {}
    for sh in shapes:
        x = [r for r in co if r["truth_shape"] == sh]; y = [r for r in rf if r["truth_shape"] == sh and r["arm"] == "ref_b150"]
        S["shape"][sh] = {"n": len(x), "dp_functional": _mean(r["dp_errors_functional"] for r in x), "dp_exact": _mean(r["dp_errors_exact"] for r in x), "dp_classes": _mean(r["dp_classes"] for r in x), "rank_errors": _mean(r["rank_errors"] for r in x), "ref_functional": _mean(r["functional"] for r in y), "ref_accord": _mean(r["accord"] for r in y), "ref_good_in_pool": _mean(r["good_in_pool"] for r in y)}
    for name, key in (("operators", "truth_operators"), ("strahler", "truth_strahler"), ("n_xor", "truth_n_xor")):
        S[f"by_{name}"] = {str(v): {"n": sum(r[key] == v for r in co), "dp_functional": _mean(r["dp_errors_functional"] for r in co if r[key] == v), "ref_functional": _mean(r["functional"] for r in rf if r[key] == v and r["arm"] == "ref_b150"), "ref_accord": _mean(r["accord"] for r in rf if r[key] == v and r["arm"] == "ref_b150")} for v in sorted({r[key] for r in co})}
    # ---- existing pipeline
    arms = sorted({r["arm"] for r in rf}); R = ("functional", "functional_097", "accord", "accord_out", "exact", "equivalent", "truth_form_in_pool", "good_in_pool", "good_found_then_lost", "candidate_best", "candidates", "size", "fit_equivalents", "seconds")
    S["reference"] = {f"{g}/{a}": {"n": sum(f(r) and r["arm"] == a for r in rf), **{k: _mean(r.get(k) for r in rf if f(r) and r["arm"] == a) for k in R}} for g, f in groups for a in arms}
    cb = {a: {r["task_id"]: r for r in rf if r["arm"] == a} for a in arms}; cd = {r["task_id"]: r for r in co}; S["paired"] = {}
    if "ref_b150" in cb:
        for g, f in groups:
            ts = sorted(t for t in set(cd) & set(cb["ref_b150"]) if f(cd[t])); S["paired"][f"{g}: dp_errors − ref_b150"] = {"functional": paired_ci([cb["ref_b150"][t]["functional"] for t in ts], [cd[t]["dp_errors_functional"] for t in ts]), "accord": paired_ci([cb["ref_b150"][t]["accord"] for t in ts], [cd[t]["dp_errors_accord"] for t in ts]), "accord_out": paired_ci([cb["ref_b150"][t]["accord_out"] for t in ts], [cd[t]["dp_errors_accord_out"] for t in ts])}
        for a in arms:
            if a != "ref_b150": ts = sorted(set(cb[a]) & set(cb["ref_b150"])); S["paired"][f"{a} − ref_b150"] = {k: paired_ci([cb["ref_b150"][t][k] for t in ts], [cb[a][t][k] for t in ts]) for k in ("functional", "accord", "good_in_pool", "candidate_best")}
    for a, b in (("errors", "bic"), ("errors", "posterior"), ("errors", "likelihood")): S["paired"][f"dp_{b} − dp_{a}"] = {"functional": paired_ci([r[f"dp_{a}_functional"] for r in co], [r[f"dp_{b}_functional"] for r in co]), "accord": paired_ci([r[f"dp_{a}_accord"] for r in co], [r[f"dp_{b}_accord"] for r in co])}
    for h in (2, 3): S["paired"][f"mitm{h} − dp_errors"] = {"functional": paired_ci([r["dp_errors_functional"] for r in co], [r[f"mitm{h}_functional"] for r in co]), "accord": paired_ci([r["dp_errors_accord"] for r in co], [r[f"mitm{h}_accord"] for r in co])}
    # ---- trace per stage of the existing pipeline
    S["trace"] = {g: {"n": sum(f(r) for r in tr), "causes": dict(Counter(r["cause"] for r in tr if f(r))), "final_accord": _mean(r["final_accord"] for r in tr if f(r)), "n_candidates": _mean(r["n_candidates"] for r in tr if f(r)), "good_before_fit": _mean(r["n_good_candidates"] > 0 for r in tr if f(r)),
                      "good_candidate_rank": _median(r.get("good_candidate_rank") for r in tr if f(r)), "final_good": _mean(r["final_accord"] >= 0.99 for r in tr if f(r)),
                      **{f"cf_{k}": _mean((r.get("counterfactual") or {}).get(k) for r in tr if f(r)) for k in ("hold_best", "no_simplification", "pre_fit_top_score", "best_available")}, "truth_fit_accord": _mean((r.get("truth_fit") or {}).get("accord") for r in tr if f(r))} for g, f in groups}
    return S


def aggregate():
    rows, bad = load_rows(); S = summarize(rows, bad); json.dump(S, open(SUMMARY, "w"), indent=1); print("rows", len(rows), "units in error", bad); return S


def show(S):
    ci = lambda v: {x: (y["diff"], y["ci95"]) if y else None for x, y in v.items()}
    print("\nCOMPOSITION SPACE"); [print(k.ljust(6), v) for k, v in S["space"].items()]; print("\nscaling law"); [print(k.ljust(12), v) for k, v in S["scaling"].items()]
    print("\nSELECTORS (exhaustive dynamic programming)"); [print(k.ljust(18), v) for k, v in S["selectors"].items()]; print("\none more leaf"); [print(k.ljust(14), v) for k, v in S["plus_one"].items()]
    print("\nBEAM OVER THE CLASSES"); [print(k.ljust(16), v) for k, v in S["beam"].items()]; print("\nBUDGET"); [print(k.ljust(20), v) for k, v in S["budget"].items()]; print("\nMEET IN THE MIDDLE"); [print(k.ljust(12), v) for k, v in S["mitm"].items()]
    print("\nFITS (A none, B atoms alone, C composite, D validated composite)"); [print(k.ljust(14), v) for k, v in S["fit"].items()]; [print(k.ljust(22), ci(v)) for k, v in S["fit_paired"].items()]
    print("\nSHAPE OF THE TREE"); [print(k.ljust(10), v) for k, v in S["shape"].items()]
    for n in ("operators", "strahler", "n_xor"): print(f"by {n}", S[f"by_{n}"])
    print("\nEXISTING PIPELINE (true atoms)"); [print(k.ljust(26), v) for k, v in S["reference"].items()]; print("\npaired"); [print(k.ljust(34), ci(v)) for k, v in S["paired"].items()]
    print("\nTRACE PER STAGE"); [print(k.ljust(6), v) for k, v in S["trace"].items()]


def report(): show(json.load(open(SUMMARY)))


def smoke():
    from core.deep_compose_exp import DEPTHS, REF_ARMS, compose_row, ref_row, task_at, trace_row
    rows = []; ds = [int(x) for x in os.environ.get("DC_DEPTHS", ",".join(map(str, DEPTHS))).split(",")]; arms = os.environ.get("DC_ARMS", "ref_b150,ref_never_b150").split(",")
    for d in ds:
        t = task_at(899, d, "standard"); tag = {"task_id": f"899_{d}", "depth": d, "regime": "standard"}; rows.append({**tag, **compose_row(t)}); rows += [{**tag, **ref_row(t, a)} for a in arms if a]
        if os.environ.get("DC_TRACE") and d >= 2: rows.append({**tag, **trace_row(t)})
    show(summarize(rows))


if __name__ == "__main__":
    {"aggregate": aggregate, "report": report, "smoke": smoke}[sys.argv[1]]()

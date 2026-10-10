"""Reports of the High-Precision Primitive Estimation sub-milestone (dev only; never the sealed set).

  aggregate : aggregates the AWS `deephp` runs → results/deep_high_precision_summary.json
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
SUMMARY = os.path.join(OUT, "deep_high_precision_summary.json")
PRIMARY_REGIMES = ("standard", "rich")
CHAIN_KEYS = ("accord", "accord_out", "functional", "functional_097", "candidate_097", "pool_recall_functional", "pool_all_functional", "pool_most_functional", "pool_rel10", "pool_rel05", "pool_all_rel10", "kept_recall_functional", "kept_all_functional", "kept_rel10", "kept_all_rel10", "pool_param_error",
              "hypotheses", "literals", "kept", "false_kept", "seconds", "hp_added", "hp_committed", "hp_births", "hp_evaluations", "rss_mb")
PAIR_KEYS = ("accord", "accord_out", "functional", "functional_097", "candidate_097", "pool_recall_functional", "pool_all_functional", "pool_rel10", "pool_rel05", "pool_all_rel10", "kept_rel10", "kept_all_functional", "pool_param_error", "seconds")
PAIRS = (("B", "C"), ("A", "C"), ("B5", "C"), ("C_prev", "C"), ("B_full", "C"), ("A_full", "C"), ("A", "B"), ("B", "B5"), ("C_pool", "C"), ("C_nocurves", "C"), ("C_norefine", "C"), ("C_nobirth", "C"), ("C_cv", "C"), ("C_k1", "C"), ("C_k3", "C"), ("C_nojoint", "C"), ("B", "C_pool"))


def load_rows():
    rows = []; bad = 0
    for f in glob.glob(os.path.join(OUT, "aws", "wce-deephp-*", "artifacts", "*", "deephp_*.json")):
        r = json.load(open(f))
        if r["status"] == "ok": rows += r["result"]
        else: bad += 1
    return rows, bad


def _paired(rows, a, b, keys):
    A = {r["task_id"]: r for r in rows if r["arm"] == a}; B = {r["task_id"]: r for r in rows if r["arm"] == b}; ts = sorted(set(A) & set(B))
    return {k: paired_ci([A[t].get(k) for t in ts], [B[t].get(k) for t in ts]) for k in keys} if len(ts) >= 3 else None


def tolerances(se):
    """Preregistered rule: largest perturbation whose final recovery ≥ 0.99 keeps 80% of the level without
    perturbation (rows whose base is functional: level without perturbation = 1)."""
    out = {}
    for kind, fam in sorted({(r["kind"], r["family"]) for r in se}):
        x = [r for r in se if r["kind"] == kind and r["family"] == fam and r["base_functional"]]; curve = {}
        for d in sorted({r["delta"] for r in x}):
            y = [r for r in x if r["delta"] == d]; curve[str(d)] = {"n": len(y), "atom_rel": _mean(r["atom_rel"] for r in y), "functional": _mean(r["functional"] for r in y), "functional_097": _mean(r["functional_097"] for r in y), "accord": _mean(r["accord"] for r in y)}
        ok = [float(d) for d, v in curve.items() if v["functional"] is not None and v["functional"] >= 0.80]; bad = [float(d) for d, v in curve.items() if v["functional"] is not None and v["functional"] < 0.80]; ok = [d for d in ok if all(d < b for b in bad)]
        out[f"{kind}/{fam}"] = {"tolerance": max(ok) if ok else None, "curve": curve}
    return out


def summarize(rows, bad=0):
    S = {"n_rows": len(rows), "units_in_error": bad}
    # ---- chains
    S["chain"] = {}; S["paired"] = {}; S["set_scaling"] = {}
    for exp, regs in (("chain", ("primary", "standard", "rich", "dense")), ("ood", ("hard",))):
        for reg in regs:
            ch = [r for r in rows if r["experiment"] == exp and (r["regime"] in PRIMARY_REGIMES if reg == "primary" else r["regime"] == reg)]
            if not ch: continue
            groups = [("all", lambda r: True), ("d4-6", lambda r: r["depth"] >= 4)]
            for g, f in groups:
                for a in sorted({r["arm"] for r in ch}):
                    x = [r for r in ch if r["arm"] == a and f(r)]; S["chain"][f"{reg}/{g}/{a}"] = {"n": len(x), **{k: _mean(r.get(k) for r in x) for k in CHAIN_KEYS}}
                for a, b in PAIRS:
                    p = _paired([r for r in ch if f(r)], a, b, PAIR_KEYS)
                    if p: S["paired"][f"{reg}/{g}: {b} − {a}"] = p
            for a in sorted({r["arm"] for r in ch if "atoms" in r}):
                for d in sorted({r["depth"] for r in ch}):
                    x = [r for r in ch if r["arm"] == a and r["depth"] == d]; at = [t for r in x for t in r.get("atoms", [])]; hp = [t for r in x for t in r.get("atoms_hp", [])]; rc = _mean(t["pool"] for t in at)
                    S["set_scaling"][f"{reg}/{a}/n{d}"] = {"n": len(x), "atom_recall": rc, "atom_rel10": _mean(t["rel_pool"] <= 0.10 for t in hp), "atom_rel05": _mean(t["rel_pool"] <= 0.05 for t in hp), "all_present": _mean(r["pool_all_functional"] for r in x), "all_rel10": _mean(r.get("pool_all_rel10") for r in x),
                                                         "independent_model": round(rc ** d, 4) if rc is not None else None, "kept_all": _mean(r["kept_all_functional"] for r in x), "accord": _mean(r["accord"] for r in x), "functional": _mean(r["functional"] for r in x)}
    # ---- by type, by visibility, parameter errors (primary regimes and dense)
    S["by_type"] = {}; S["by_visibility"] = {}; S["param_errors"] = {}
    for reg in ("primary", "dense"):
        ch = [r for r in rows if r["experiment"] == "chain" and (r["regime"] in PRIMARY_REGIMES if reg == "primary" else r["regime"] == reg)]
        for a in ("A", "B", "B5", "C", "C_pool", "C_prev"):
            at = [(t, h) for r in ch if r["arm"] == a for t, h in zip(r.get("atoms", []), r.get("atoms_hp", []))]
            if not at: continue
            S["by_type"][f"{reg}/{a}"] = {k: {"n": sum(t["type"] == k for t, _ in at), "recall": _mean(t["pool"] for t, _ in at if t["type"] == k), "rel10": _mean(h["rel_pool"] <= 0.10 for t, h in at if t["type"] == k), "rel05": _mean(h["rel_pool"] <= 0.05 for t, h in at if t["type"] == k), "kept_rel10": _mean(h["rel_kept"] <= 0.10 for t, h in at if t["type"] == k)} for k in sorted({t["type"] for t, _ in at})}
            S["by_visibility"][f"{reg}/{a}"] = {f"{lo:.1f}-{hi:.1f}": {"n": sum(lo <= t["visible"] < hi + 1e-9 for t, _ in at), "recall": _mean(t["pool"] for t, _ in at if lo <= t["visible"] < hi + 1e-9), "rel10": _mean(h["rel_pool"] <= 0.10 for t, h in at if lo <= t["visible"] < hi + 1e-9)} for lo, hi in ((0.0, 0.4), (0.4, 0.8), (0.8, 1.0))}
            pe = defaultdict(list)
            for t, h in at:
                for f, v in h["errors"].items(): pe[f"{t['type']}/{f}"].append(v)
            S["param_errors"][f"{reg}/{a}"] = {k: {"n": len(v), "median": round(sorted(v)[len(v) // 2], 4), "mean": _mean(v)} for k, v in sorted(pe.items())}
    # ---- sensitivity map, tolerances, share of atoms within tolerance
    se = [r for r in rows if r["experiment"] == "sens"]; S["sensitivity"] = tolerances(se); S["sens_base"] = {"n": len(se), "base_functional": _mean(r["base_functional"] for r in se)}; S["within_tolerance"] = {}
    tol = {k: v["tolerance"] for k, v in S["sensitivity"].items()}
    for reg in ("primary", "dense"):
        ch = [r for r in rows if r["experiment"] == "chain" and (r["regime"] in PRIMARY_REGIMES if reg == "primary" else r["regime"] == reg)]
        for a in ("A", "B", "C", "C_pool"):
            hp = [h for r in ch if r["arm"] == a for h in r.get("atoms_hp", [])]
            if not hp: continue
            ok = [bool(h["errors"]) and all(tol.get(f"{h['type']}/{f}") is not None and v <= tol[f"{h['type']}/{f}"] for f, v in h["errors"].items()) for h in hp]
            S["within_tolerance"][f"{reg}/{a}"] = {"n": len(hp), "all": _mean(ok), **{k: _mean(o for o, h in zip(ok, hp) if h["type"] == k) for k in sorted({h["type"] for h in hp})}}
    # ---- bench per primitive
    si = [r for r in rows if r["experiment"] == "single"]; S["single"] = {}; S["single_paired"] = {}; S["identifiability"] = {}
    cut = lambda r: {"n": len(r), "top1_rel10": _mean(x["rel_top1"] <= 0.10 for x in r), "top1_rel05": _mean(x["rel_top1"] <= 0.05 for x in r), "top3_rel10": _mean(x["rel_top3"] <= 0.10 for x in r), "top5_rel10": _mean(x["rel_top5"] <= 0.10 for x in r), "catastrophic": _mean(x["rel_top1"] > 0.25 for x in r), "rel_median": round(sorted(x["rel_top1"] for x in r)[len(r) // 2], 4) if r else None, "obs_error": _mean(x["obs_error"] for x in r), "seconds": _mean(x["seconds"] for x in r)}
    arms = sorted({r["arm"] for r in si})
    for a in arms:
        x = [r for r in si if r["arm"] == a]; S["single"][f"all/{a}"] = cut(x)
        for kd in sorted({r["kind"] for r in x}): S["single"][f"kind:{kd}/{a}"] = cut([r for r in x if r["kind"] == kd])
        for n in sorted({r["n"] for r in x}): S["single"][f"n:{n}/{a}"] = cut([r for r in x if r["n"] == n])
        for cond, f in (("clean", lambda r: r["noise"] == 0 and r["mask"] == 0), ("noise_only", lambda r: r["noise"] > 0 and r["mask"] == 0), ("mask_only", lambda r: r["noise"] == 0 and r["mask"] > 0), ("noise+mask", lambda r: r["noise"] > 0 and r["mask"] > 0)): S["single"][f"cond:{cond}/{a}"] = cut([r for r in x if f(r)])
        for m in sorted({r["mask"] for r in x}): S["single"][f"mask:{m}/{a}"] = cut([r for r in x if r["mask"] == m])
        for kd in sorted({r["kind"] for r in x}):
            for n in sorted({r["n"] for r in x}): S["single"][f"kind:{kd}/n:{n}/nomask/{a}"] = cut([r for r in x if r["kind"] == kd and r["n"] == n and r["mask"] == 0])
            for m in sorted({r["mask"] for r in x}): S["single"][f"kind:{kd}/mask:{m}/{a}"] = cut([r for r in x if r["kind"] == kd and r["mask"] == m])
    ind = lambda r, k: float(r[k] <= 0.10)
    for a, b, tag in (("gradient", "curves", "top1"), ("flat", "curves", "top1"), ("gradient", "flat", "top1"), ("curves", "refined", "top1"), ("refined", "consensus", "top1"), ("curves", "direct", "top1"), ("curves_independent", "curves", "top1"), ("curves", "truth_refined", "top1")):
        A = {r["task_id"]: r for r in si if r["arm"] == a}; B = {r["task_id"]: r for r in si if r["arm"] == b}; ts = sorted(set(A) & set(B))
        if len(ts) < 3: continue
        S["single_paired"][f"{b} − {a}"] = {"rel10": paired_ci([ind(A[t], "rel_top1") for t in ts], [ind(B[t], "rel_top1") for t in ts]), "catastrophic": paired_ci([float(A[t]["rel_top1"] > 0.25) for t in ts], [float(B[t]["rel_top1"] > 0.25) for t in ts]), "rel": paired_ci([min(A[t]["rel_top1"], 1.0) for t in ts], [min(B[t]["rel_top1"], 1.0) for t in ts])}
        for kd in sorted({r["kind"] for r in si}):
            tk = [t for t in ts if A[t]["kind"] == kd]
            if len(tk) >= 3: S["single_paired"][f"{b} − {a} [{kd}]"] = {"rel10": paired_ci([ind(A[t], "rel_top1") for t in tk], [ind(B[t], "rel_top1") for t in tk])}
    cu = {r["task_id"]: r for r in si if r["arm"] == "curves"}
    if cu: S["single_paired"]["curves: top5 − top1"] = {"rel10": paired_ci([ind(r, "rel_top1") for r in cu.values()], [ind(r, "rel_top5") for r in cu.values()])}
    fl = {r["task_id"]: r for r in si if r["arm"] == "flat"}
    if fl: S["single_paired"]["flat: top5 − top1"] = {"rel10": paired_ci([ind(r, "rel_top1") for r in fl.values()], [ind(r, "rel_top5") for r in fl.values()])}
    one = [r for r in si if r["arm"] == "curves"]
    for kd in sorted({r["kind"] for r in one}):
        for key, f in [(f"n:{n}", (lambda r, n=n: r["n"] == n)) for n in sorted({r["n"] for r in one})] + [(f"mask:{m}", (lambda r, m=m: r["mask"] == m)) for m in sorted({r["mask"] for r in one})]:
            x = [r for r in one if r["kind"] == kd and f(r)]; S["identifiability"][f"{kd}/{key}"] = {"n": len(x), **{k: round(v / max(1, len(x)), 3) for k, v in Counter(r["identifiability"] for r in x).items()}, "stability": _mean(r["stability"] for r in x)}
        x = [r for r in one if r["kind"] == kd]; un = defaultdict(list)
        for r in x:
            for f, v in r["uncertainty"].items(): un[f].append(v)
        S["identifiability"][f"{kd}/uncertainty"] = {f: _mean(v) for f, v in un.items()}
        for cls in sorted({r["identifiability"] for r in x}): S["identifiability"][f"{kd}/{cls}"] = {"n": sum(r["identifiability"] == cls for r in x), "curves_rel10": _mean(r["rel_top1"] <= 0.10 for r in x if r["identifiability"] == cls)}
    # ---- birth
    bi = [r for r in rows if r["experiment"] == "birth"]; S["birth"] = {}; S["birth_paired"] = {}; comp = [r for r in bi if r["arm"] == "complete"]
    S["birth"]["complete"] = {"n": len(comp), "false_birth": _mean(r["false_birth"] for r in comp), "proposed_birth": _mean(r["proposed_birth"] for r in comp)}
    BK = ("born", "committed", "found_rel10", "found_free", "found_old_measure", "same_type", "functional", "accord", "before_functional", "before_accord", "seconds", "evaluations")
    for ctx in ("exact", "noisy"):
        x = [r for r in bi if r.get("context") == ctx]
        for a in sorted({r["arm"] for r in x}):
            y = [r for r in x if r["arm"] == a]; S["birth"][f"{ctx}/{a}"] = {"n": len(y), **{k: _mean(r.get(k) for r in y) for k in BK}}
            for t in sorted({r["missing_type"] for r in y}): z = [r for r in y if r["missing_type"] == t]; S["birth"][f"{ctx}/{a}/{t}"] = {"n": len(z), **{k: _mean(r.get(k) for r in z) for k in ("found_rel10", "found_free", "found_old_measure", "found_in_alternatives", "same_type", "functional")}}
            for d in sorted({r["depth"] for r in y}): z = [r for r in y if r["depth"] == d]; S["birth"][f"{ctx}/{a}/d{d}"] = {"n": len(z), **{k: _mean(r.get(k) for r in z) for k in ("found_rel10", "found_free", "functional")}}
        for a in ("new", "new_nomorph", "new_unvalidated", "new_onebirth"):
            p = _paired(x, "old", a, ("found_free", "found_rel10", "functional", "accord"))
            if p: S["birth_paired"][f"{ctx}: {a} − old"] = p
        p = _paired(x, "new_nomorph", "new", ("found_free", "found_rel10", "functional"))
        if p: S["birth_paired"][f"{ctx}: new − new_nomorph"] = p
        nw = [r for r in x if r["arm"] == "new_unvalidated" and r.get("morphology")]; S["birth"][f"{ctx}/morphology"] = {t: dict(Counter(r["morphology"] for r in nw if r["missing_type"] == t)) for t in sorted({r["missing_type"] for r in nw})}
        pe = defaultdict(list)
        for r in x:
            if r["arm"] == "new_unvalidated" and r["found_rel10"]:
                for f, v in r["errors"].items(): pe[f"{r['missing_type']}/{f}"].append(v)
        S["birth"][f"{ctx}/param_errors"] = {k: _mean(v) for k, v in sorted(pe.items())}
    return S


def aggregate():
    rows, bad = load_rows(); S = summarize(rows, bad); json.dump(S, open(SUMMARY, "w"), indent=1); print("rows", len(rows), "units in error", bad); return S


def show(S):
    ci = lambda v: {x: (y["diff"], y["ci95"]) if y else None for x, y in v.items()}; cut = lambda v, ks: {k: v.get(k) for k in ks if v.get(k) is not None}
    print("rows", S["n_rows"], "units in error", S["units_in_error"])
    print("\nCHAINS"); [print(k.ljust(28), cut(v, ("n", "accord", "functional", "functional_097", "candidate_097", "pool_recall_functional", "pool_rel10", "pool_rel05", "pool_all_functional", "pool_all_rel10", "kept_all_functional", "pool_param_error", "hypotheses", "seconds"))) for k, v in S["chain"].items()]
    print("\nPAIRED"); [print(k.ljust(34), ci({a: b for a, b in v.items() if a in ("accord", "functional", "candidate_097", "pool_recall_functional", "pool_rel10", "pool_all_functional", "seconds")})) for k, v in S["paired"].items()]
    print("\nSCALING"); [print(k.ljust(22), v) for k, v in S["set_scaling"].items() if k.split("/")[1] in ("A", "B", "C")]
    print("\nBY TYPE"); [print(k.ljust(16), v) for k, v in S["by_type"].items()]; print("\nBY VISIBILITY"); [print(k.ljust(16), v) for k, v in S["by_visibility"].items()]; print("\nPARAMETER ERRORS"); [print(k.ljust(16), v) for k, v in S["param_errors"].items()]
    print("\nSENSITIVITY", S["sens_base"]); [print(k.ljust(24), v["tolerance"], {d: (c["atom_rel"], c["functional"], c["accord"]) for d, c in v["curve"].items()}) for k, v in S["sensitivity"].items()]; print("within tolerance", S["within_tolerance"])
    print("\nBENCH PER PRIMITIVE"); [print(k.ljust(40), v) for k, v in S["single"].items() if k.count("/") == 1]; print(); [print(k.ljust(40), ci(v)) for k, v in S["single_paired"].items()]
    print("\nby type × density (no mask)"); [print(k.ljust(44), cut(v, ("n", "top1_rel10", "top5_rel10", "catastrophic"))) for k, v in S["single"].items() if "/nomask/" in k and k.split("/")[-1] in ("gradient", "flat", "curves", "direct", "refined", "truth_refined")]
    print("\nby type × mask"); [print(k.ljust(44), cut(v, ("n", "top1_rel10", "top5_rel10", "catastrophic"))) for k, v in S["single"].items() if "/mask:" in k and k.split("/")[-1] in ("gradient", "flat", "curves", "direct", "given_center", "given_orientation", "given_size", "truth_refined")]
    print("\nIDENTIFIABILITY"); [print(k.ljust(30), v) for k, v in S["identifiability"].items()]
    print("\nBIRTH"); [print(k.ljust(34), v) for k, v in S["birth"].items()]; [print(k.ljust(34), ci(v)) for k, v in S["birth_paired"].items()]


def report(): show(json.load(open(SUMMARY)))


if __name__ == "__main__":
    {"aggregate": aggregate, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()

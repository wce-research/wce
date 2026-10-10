"""Reports of the Robust Primitive Discovery sub-milestone (dev only; never the sealed set).

  smoke     : short local demonstration on TUNING scenes (indices 5000+, never evaluated)
  aggregate : aggregates the AWS `deeprp` runs → results/deep_robust_primitive_summary.json
  report    : prints the tables
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
SUMMARY = os.path.join(OUT, "deep_robust_primitive_summary.json")


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]; return round(sum(xs) / len(xs), 4) if xs else None


def paired_ci(a, b, n_boot=4000, seed=0):
    """Mean paired difference b − a and 95% bootstrap CI (resampling of the paired units)."""
    d = [float(y) - float(x) for x, y in zip(a, b) if x is not None and y is not None]
    if len(d) < 3: return None
    rng = random.Random(seed); n = len(d); bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"n": n, "diff": round(sum(d) / n, 4), "ci95": [round(bs[int(0.025 * n_boot)], 4), round(bs[int(0.975 * n_boot) - 1], 4)]}


def load_rows():
    rows = []; bad = 0
    for f in glob.glob(os.path.join(OUT, "aws", "wce-deeprp-*", "artifacts", "*", "deeprp_*.json")):
        r = json.load(open(f))
        if r["status"] == "ok": rows += r["result"]
        else: bad += 1
    return rows, bad


ATOMIC_KEYS = ("exact", "functional", "equivalence", "component_recall", "precision", "false_positives", "retained", "classes", "joint", "compactness", "proposals", "fit_equivalents", "seconds", "equivalence_identifiable")


def summarize(rows, bad=0):
    from core.deep_robust_prim import AUDIT_CLASSES, BEST, REFERENCE
    S = {"n_rows": len(rows), "units_in_error": bad}
    # ---- atomic bench
    at = [r for r in rows if r["experiment"] == "atomic" and not r.get("unavailable")]; methods = sorted({r["method"] for r in at}); S["atomic"] = {}; S["atomic_by_category"] = {}; S["by_type"] = {}; S["paired_vs_reference"] = {}
    by = {m: {r["scene"]: r for r in at if r["method"] == m} for m in methods}
    for m in methods:
        x = list(by[m].values()); S["atomic"][m] = {"n": len(x), **{k: _mean(r.get(k) for r in x) for k in ATOMIC_KEYS}}
        for c in sorted({r["category"] for r in x}): S["atomic_by_category"][f"{c}/{m}"] = {k: _mean(r.get(k) for r in x if r["category"] == c) for k in ("exact", "equivalence", "component_recall", "precision", "false_positives")}
        ty = defaultdict(list)
        for r in x:
            for a in r["atoms"]: ty[a["type"]].append(a)
        S["by_type"][m] = {t: {"n": len(v), "exact": _mean(a["exact"] for a in v), "equivalence": _mean(a["equivalence"] for a in v), "component_recall": _mean(a["component_recall"] for a in v)} for t, v in sorted(ty.items())}
        if m != REFERENCE and REFERENCE in by:
            sc = sorted(set(by[m]) & set(by[REFERENCE])); S["paired_vs_reference"][m] = {k: paired_ci([by[REFERENCE][s].get(k) for s in sc], [by[m][s].get(k) for s in sc]) for k in ("equivalence", "component_recall", "precision", "false_positives", "exact", "retained")}
    # ---- audit
    au = [r for r in rows if r["experiment"] == "audit"]; S["audit"] = {}; S["audit_false_positives"] = {}; S["audit_reject_reasons"] = {}
    for m in sorted({r["method"] for r in au}):
        comp = [r for r in au if r["method"] == m and r["kind"] == "component"]; S["audit"][m] = {}
        for t in ["all"] + sorted({r["type"] for r in comp}):
            x = [r for r in comp if t == "all" or r["type"] == t]; c = Counter(r["class"] for r in x); S["audit"][m][t] = {"n": len(x), **{k: round(c.get(k, 0) / max(1, len(x)), 4) for k in AUDIT_CLASSES}}
        fp = [r for r in au if r["method"] == m and r["kind"] == "false_positive"]; ns = len({r["scene"] for r in au if r["method"] == m})
        S["audit_false_positives"][m] = {"per_scene": round(len(fp) / max(1, ns), 3), "near_true_share": _mean(r["near_true"] for r in fp), "by_type": dict(Counter(r["type"] for r in fp))}
        S["audit_reject_reasons"][m] = dict(Counter(r["reason"] for r in comp if r["class"] == "rejected"))
    # ---- validator
    va = [r for r in rows if r["experiment"] == "validator"]; S["validator"] = {}
    for v in sorted({r["validator"] for r in va}):
        x = [r for r in va if r["validator"] == v]; ta, fa, tr, fr = (sum(r[k] for r in x) for k in ("TA", "FA", "TR", "FR"))
        S["validator"][v] = {"TA": ta, "FA": fa, "TR": tr, "FR": fr, "true_accept_rate": round(ta / max(1, ta + fr), 4), "false_accept_rate": round(fa / max(1, fa + tr), 4), "accept_precision": round(ta / max(1, ta + fa), 4), "seconds": _mean(r["seconds"] for r in x)}
    # ---- diversity
    dv = [r for r in rows if r["experiment"] == "diversity"]; names = sorted({r["proposer"] for r in dv}); hits = {p: {(r["scene"], i) for r in dv if r["proposer"] == p for i in r["hits"]} for p in names}
    total = sum(r["n_true"] for r in dv if r["proposer"] == names[0]) if names else 0; allh = set().union(*hits.values()) if hits else set(); S["diversity"] = {"n_true_components": total, "union_recall": round(len(allh) / max(1, total), 4), "proposers": {}, "jaccard": {}}
    for p in names:
        x = [r for r in dv if r["proposer"] == p]; others = set().union(*[hits[q] for q in names if q != p]) if len(names) > 1 else set()
        S["diversity"]["proposers"][p] = {"recall": round(len(hits[p]) / max(1, total), 4), "unique": len(hits[p] - others), "union_without": round(len(others) / max(1, total), 4), "proposals": _mean(r["proposals"] for r in x), "fit_equivalents": _mean(r["fit_equivalents"] for r in x),
                                          "seconds": _mean(r["seconds"] for r in x)}
        for q in names:
            if p < q: S["diversity"]["jaccard"][f"{p}~{q}"] = round(len(hits[p] & hits[q]) / max(1, len(hits[p] | hits[q])), 3)
    # ---- active measurements
    ac = [r for r in rows if r["experiment"] == "active"]; S["active"] = {m: {"n": sum(r["method"] == m for r in ac), **{k: _mean(r[k] for r in ac if r["method"] == m) for k in ("equivalence", "component_recall", "precision", "exact")}} for m in sorted({r["method"] for r in ac})}
    # ---- downstream chain
    ch = [r for r in rows if r["experiment"] == "chain"]; S["chain"] = {}; S["chain_by_depth"] = {}; S["chain_paired"] = {}; C = ("primitive_equivalence", "primitive_exact", "component_recall", "primitive_precision", "n_literals", "candidate_functional", "candidate_best", "final_functional", "final_095", "final_full", "atoms", "fit_equivalents", "seconds")
    arms = sorted({(r["source"], r["policy"]) for r in ch}); cb = {a: {r["task_id"]: r for r in ch if (r["source"], r["policy"]) == a} for a in arms}
    for s, p in arms:
        x = list(cb[(s, p)].values()); S["chain"][f"{s}/{p}"] = {"n": len(x), **{k: _mean(r.get(k) for r in x) for k in C}}
        for d in sorted({r["depth"] for r in x}): S["chain_by_depth"][f"{s}/{p}/d{d}"] = {k: _mean(r.get(k) for r in x if r["depth"] == d) for k in ("primitive_equivalence", "candidate_functional", "final_functional", "final_full")}
        for ref in (("reference", "always"), ("gradient_disc", "always")):
            if ref in cb and (s, p) != ref:
                ts = sorted(set(cb[(s, p)]) & set(cb[ref])); S["chain_paired"][f"{s}/{p} − {ref[0]}/{ref[1]}"] = {k: paired_ci([cb[ref][t][k] for t in ts], [cb[(s, p)][t][k] for t in ts]) for k in ("primitive_equivalence", "candidate_functional", "final_functional", "final_full")}
    return S


def aggregate():
    rows, bad = load_rows(); S = summarize(rows, bad); json.dump(S, open(SUMMARY, "w"), indent=1); print("rows", len(rows), "units in error", bad); return S


def _fmt(v): return "—" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)


def show(S):
    print("\nATOMIC BENCH"); K = ("n", "exact", "equivalence", "component_recall", "precision", "false_positives", "retained", "joint", "compactness", "proposals", "fit_equivalents", "seconds")
    print("arm".ljust(20) + " ".join(k[:10].rjust(11) for k in K))
    for m, v in S["atomic"].items(): print(m.ljust(20) + " ".join(_fmt(v[k]).rjust(11) for k in K))
    print("\npaired differences against the reference (direct gradient), 95% CI")
    for m, v in S["paired_vs_reference"].items(): print(m.ljust(20), {k: (x["diff"], x["ci95"]) if x else None for k, x in v.items() if k in ("equivalence", "precision", "false_positives")})
    print("\nby atom type (exact / equivalence / components)")
    for m, v in S["by_type"].items():
        if m in ("gradient", "union_rv", "union_joint", "union_old"): print(m.ljust(14), {t: (x["n"], x["exact"], x["equivalence"], x["component_recall"]) for t, x in v.items()})
    print("\nby category (equivalence / precision)")
    for k, v in S["atomic_by_category"].items():
        if k.split("/")[1] in ("gradient", "union_joint"): print(k.ljust(26), v["equivalence"], v["precision"], v["false_positives"])
    print("\nAUDIT (share of the true boundary components)")
    for m, v in S["audit"].items():
        for t, x in v.items(): print(f"{m}/{t}".ljust(28), x)
        print("  false positives", S["audit_false_positives"][m], "| rejection reasons", S["audit_reject_reasons"][m])
    print("\nVALIDATOR (same pool)"); [print(k.ljust(14), v) for k, v in S["validator"].items()]
    print("\nDIVERSITY", {k: S["diversity"][k] for k in ("n_true_components", "union_recall")}); [print(k.ljust(14), v) for k, v in S["diversity"]["proposers"].items()]; print("Jaccard", S["diversity"]["jaccard"])
    print("\nACTIVE MEASUREMENTS"); [print(k.ljust(18), v) for k, v in S["active"].items()]
    print("\nDOWNSTREAM CHAIN"); K = ("n", "primitive_equivalence", "component_recall", "primitive_precision", "n_literals", "candidate_functional", "final_095", "final_functional", "final_full", "fit_equivalents", "seconds")
    for k, v in S["chain"].items(): print(k.ljust(30), {x: v[x] for x in K})
    for k, v in S["chain_paired"].items(): print(k.ljust(52), {x: (y["diff"], y["ci95"]) if y else None for x, y in v.items()})
    print("\nby depth"); [print(k.ljust(34), v) for k, v in S["chain_by_depth"].items()]


def report(): show(json.load(open(SUMMARY)))


def smoke():
    """Short local demonstration (tuning scenes only): checks that the chain runs end to end."""
    from core.deep_primitive_world import CATEGORIES
    from core.deep_robust_prim import BEST, REFERENCE, active, arm_row, audit, diversity, scene_at, validator_study
    n = int(os.environ.get("RP_SMOKE_SCENES", "1")); arms = os.environ.get("RP_SMOKE_ARMS", f"{REFERENCE},gradient_rv,ransac_rv,ransac_local_rv,union_rv,{BEST},union_all_joint,union_old,union_holdout,union_pairsplit").split(","); rows = []
    for cat in CATEGORIES:
        for j in range(n):
            sc = scene_at(j, cat, tune=True); tag = {"category": cat, "scene": f"{j}_{cat}"}; rows += [{"experiment": "atomic", **tag, **arm_row(sc, a, j)} for a in arms]
            rows += [{"experiment": "audit", **tag, **r} for a in (REFERENCE, BEST) for r in audit(sc, a, j)]; rows += [{"experiment": "validator", **tag, **r} for r in validator_study(sc, j)]; rows += [{"experiment": "diversity", **tag, **r} for r in diversity(sc, j)]
    rows += [{"experiment": "active", "category": "two", **active(scene_at(0, "two", tune=True), p, 0)} for p in ("none", "random", "primitive")]
    show(summarize(rows))


def diagnose():
    """Tuning (scenes 5000+): for each true boundary component, where does it get lost?"""
    import torch
    from core.deep_fit import DT, FitCounter
    from core.deep_primitive_eval import component_match, components
    from core.deep_primitive_world import CATEGORIES
    from core.deep_primitives import atom_name, close_pairs, contrast, contrast_fit
    from core.deep_primitives2 import LocalRefiner, clean_midpoints, local_refit, RobustValidator, boundary_elements, disc_atom, hough_circles, hough_lines, line_atom, split_pairs
    from core.deep_robust_prim import proposer, scene_at
    n = int(os.environ.get("RP_SMOKE_SCENES", "3")); rows = []; v = RobustValidator()
    for cat in CATEGORIES:
        for j in range(n):
            sc = scene_at(j, cat, tune=True); X, y = sc.task.X, sc.task.y; pairs = close_pairs(X, y); prop, val = split_pairs(pairs, j); M, N = boundary_elements(pairs)
            raw = [("line", a, b) for a, b, _ in hough_lines(M, N)] + [("circle", cx, cy, r) for cx, cy, r, _ in hough_circles(M, N)]
            props = {pn: [d for c in proposer(pn).propose(X, y, close_pairs(X, y), j, FitCounter()) for d in components(c.program)] for pn in ("hough", "grid", "gradient", "gradient_min", "geometric", "ransac")}
            props.update({"L_" + pn: [d for c in LocalRefiner(proposer(pn)).propose(X, y, close_pairs(X, y), j, FitCounter()) for d in components(c.program)] for pn in ("hough", "grid", "gradient_min", "geometric", "ransac")}); props["L_all"] = [d for k_, ds in props.items() if k_.startswith("L_") for d in ds]; Mc = clean_midpoints(X, y, pairs)
            for a in sc.atoms:
                for c in components(a):
                    if c[0] == "circle": prog = disc_atom(c[1], c[2], c[3])
                    elif c[0] == "line": prog = line_atom(c[1], c[2])
                    else: continue
                    e = v.evidence(prog, prop, val); th, _ = contrast_fit(prog.skeleton(), pairs, torch.tensor([prog.param_vector()], dtype=DT), steps=60); ref = components(prog.skeleton().with_params(th[0].tolist()))[0]
                    rows.append({"type": atom_name(a), "comp": c[0], "z_all": contrast(prog, pairs)["z"], "z_oos": e.z_oos, "purity": e.purity_oos, "support": e.support_oos, "segments": e.segments, "accept": v.accept(e)[0], "refit_stays": component_match(c, ref), "local_stays": component_match(c, components(local_refit(prog, Mc))[0]),
                                 "raw_hough": any(component_match(c, d) for d in raw), **{pn: any(component_match(c, d) for d in ds) for pn, ds in props.items()}})
    for t in sorted({r["type"] for r in rows}):
        x = [r for r in rows if r["type"] == t]; print(t.ljust(10), len(x), {k: _mean(r[k] for r in x) for k in ("z_all", "z_oos", "purity", "support", "segments", "accept", "refit_stays", "local_stays", "raw_hough", "hough", "grid", "gradient", "gradient_min", "geometric", "L_hough", "L_grid", "L_gradient_min", "L_geometric", "ransac", "L_ransac", "L_all")})


def tune():
    """Tuning of the validator thresholds on the TUNING scenes (5000+). Criterion fixed in advance: harmonic
    mean of the equivalence and of the precision. Writes results/deep_configs/robust_validator.json."""
    import itertools
    from core.deep_primitive_world import CATEGORIES
    from core.deep_primitives2 import CONFIG_FILE
    from core.deep_robust_prim import BEST, arm_row, scene_at
    n = int(os.environ.get("RP_SMOKE_SCENES", "3")); scenes = [(scene_at(j, cat, tune=True), j) for cat in CATEGORIES for j in range(n)]; res = []
    from core.deep_primitives2 import null_max_z
    from core.deep_robust_prim import ARMS, proposer
    nulls = {}
    for mode in ("points", "none"):
        nz = sorted(null_max_z(proposer(ARMS[BEST][0]), trials=20, split=mode)); floor = round(nz[-2], 2); nulls[mode] = {"null_max_z": nz, "z_floor": floor}; print(mode, "maximum contrast on random labels:", [round(v, 2) for v in nz], "→ floor (2nd largest out of 20)", floor, flush=True)
        for z, sup, pur, seg in itertools.product((floor, floor + 1.5), (5, 8), (0.55, 0.65), (2, 3)):
            cfg = {"split": mode, "z_oos": z, "min_support": sup, "min_purity": pur, "min_segments": seg, "prune_z": 3.0}; rows = [arm_row(sc, BEST, j, cfg) for sc, j in scenes]; e = _mean(r["equivalence"] for r in rows); p = _mean(r["precision"] if r["precision"] is not None else 0.0 for r in rows)
            res.append({"config": cfg, "equivalence": e, "precision": p, "false_positives": _mean(r["false_positives"] for r in rows), "score": round(2 * e * p / max(1e-9, e + p), 4)}); print(res[-1], flush=True)
    best = max(res, key=lambda r: r["score"]); json.dump({"config": best["config"], "null_calibration": nulls, "selected_on": f"{len(scenes)} tuning scenes (indices 5000+)", "criterion": "harmonic mean equivalence / precision", "tuning": best, "grid": res}, open(CONFIG_FILE, "w"), indent=1); print("SELECTED", best)


def chain_smoke():
    """Short local demonstration of the downstream chain on ONE tuning task (key 499, outside evaluation)."""
    from core.deep_experiments import down_task
    from core.deep_robust_prim import CHAIN_ARMS, chain
    t = down_task(499, int(os.environ.get("RP_DEPTH", "3")), "standard")
    for s_, pol in CHAIN_ARMS: r = chain(t, s_, pol, 0); print(f"{s_}/{pol}".ljust(30), {k: r[k] for k in ("primitive_equivalence", "component_recall", "n_literals", "candidate_functional", "candidate_best", "final_full", "fit_equivalents", "seconds")}, flush=True)


def inspect():
    """Tuning: retained primitives against true atoms, scene by scene."""
    from core.deep_primitive_eval import component_match, components
    from core.deep_primitives2 import discover_robust, load_config
    from core.deep_robust_prim import BEST, ARMS, proposer, scene_at, validator
    R = lambda cs: [tuple(round(v, 2) if isinstance(v, float) else v for v in c) for c in cs]
    for cat in os.environ.get("RP_CATS", "two,multi,noise").split(","):
        for j in range(int(os.environ.get("RP_SMOKE_SCENES", "2"))):
            sc = scene_at(j, cat, tune=True); tc = [c for a in sc.atoms for c in components(a)]; cf = load_config(); d = discover_robust(sc.task.X, sc.task.y, proposer(ARMS[BEST][0]), validator(ARMS[BEST][1], cf), seed=j, joint=True, prune_z=cf["prune_z"], split=cf["split"]); print(f"\n{cat} {j} TRUE", R(tc))
            for l in d.candidates:
                if l["kept"]: cs = components(l["program"]); print("   ", "OK " if all(any(component_match(c, t) for t in tc) for c in cs) else "FP ", R(cs), l["source"], "round", l["round"], "z_in %.1f z_oos %.1f pur %.2f sup %.0f seg %s" % (l["z_in"], l["z_oos"], l["purity"], l["support"], l["segments"]), l["reason"])


if __name__ == "__main__":
    {"aggregate": aggregate, "report": report, "smoke": smoke, "diagnose": diagnose, "inspect": inspect, "tune": tune, "chain-smoke": chain_smoke}[sys.argv[1]]()

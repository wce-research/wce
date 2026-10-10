"""Report of the Validated Optimizer sub-milestone (dev, true atoms provided; never the sealed set).
  aggregate : AWS runs `deepvo` (arms) and `deepopt` (optimizer) → results/deep_validated_summary.json
  report    : tables
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042, 60: 2.0, 120: 1.98}


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]; return round(sum(xs) / len(xs), 4) if xs else None


def _t(df):
    ks = sorted(T975); return next((T975[k] for k in ks if df <= k), 1.96)


def paired(diffs):
    """Paired difference: mean, 95% CI (Student), effect size (paired Cohen's d), shares improved / degraded / unchanged."""
    n = len(diffs)
    if n == 0: return {}
    m = sum(diffs) / n; sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0; h = _t(n - 1) * sd / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": round(m, 4), "ci95": [round(m - h, 4), round(m + h, 4)], "effect_size": round(m / sd, 3) if sd > 0 else None, "improved": round(sum(d > 1e-9 for d in diffs) / n, 4), "degraded": round(sum(d < -1e-9 for d in diffs) / n, 4),
            "unchanged": round(sum(abs(d) <= 1e-9 for d in diffs) / n, 4)}


def load(kind):
    rows = []; bad = 0
    for f in glob.glob(os.path.join(OUT, "aws", f"wce-{kind}-*", "artifacts", "*", f"{kind}_*.json")):
        r = json.load(open(f))
        if r["status"] == "ok": rows += r["result"]
        else: bad += 1
    return rows, bad


def arms_block(rows, other="B"):
    by = defaultdict(dict)
    for r in rows: by[r["task_id"]][r["arm"]] = r
    T = [v for v in by.values() if "A" in v and other in v]
    if not T: return {}
    out = {"n": len(T)}
    for arm in ("A", other):
        x = [v[arm] for v in T]; out[arm] = {"thresholds": {t: _mean(r["thresholds"][t] for r in x) for t in ("0.9", "0.95", "0.97", "0.99")}, "accord": _mean(r["accord"] for r in x), "in_region": _mean(r["in_region"] for r in x), "likelihood": _mean(r["likelihood"] for r in x),
                                              "errors": _mean(r["errors"] for r in x), "exact": _mean(r["exact"] for r in x), "equivalent": _mean(r["equivalent"] for r in x), "atoms": _mean(r["atoms"] for r in x), "commits": _mean(r["commits"] for r in x),
                                              "rollbacks": _mean(r["rollbacks"] for r in x), "fit_equivalents": _mean(r["fit_equivalents"] for r in x), "seconds": _mean(r["seconds"] for r in x)}
    out["diff_accord"] = paired([v[other]["accord"] - v["A"]["accord"] for v in T]); out["diff_likelihood"] = paired([v[other]["likelihood"] - v["A"]["likelihood"] for v in T])
    for t in ("0.9", "0.95", "0.97", "0.99"): out[f"diff_recovery_{t}"] = paired([float(v[other]["thresholds"][t]) - float(v["A"]["thresholds"][t]) for v in T])
    out["recovery_099_table"] = dict(Counter(("A+" if v["A"]["thresholds"]["0.99"] else "A-") + (other + "+" if v[other]["thresholds"]["0.99"] else other + "-") for v in T))
    worst = sorted(T, key=lambda v: v[other]["accord"] - v["A"]["accord"])[:3]; out["worst_degradations"] = [{"task": v["A"]["task_id"], "A": round(v["A"]["accord"], 4), other: round(v[other]["accord"], 4)} for v in worst if v[other]["accord"] < v["A"]["accord"] - 1e-9]
    return out


def opt_block(rows, start):
    S = [r["starts"][start] for r in rows]; out = {"n": len(S), "noop": {k: _mean(s["noop"][k] for s in S) for k in ("accord", "errors", "profiled", "surrogate")}, "noop_ge_099": _mean(s["noop"]["accord"] >= 0.99 for s in S), "configs": {}}
    for c in S[0]["configs"]:
        X = [s["configs"][c] for s in S]; N = [s["noop"] for s in S]
        out["configs"][c] = {"accord": _mean(x["final"]["accord"] for x in X), "ge_099": _mean(x["final"]["accord"] >= 0.99 for x in X), "errors": _mean(x["final"]["errors"] for x in X), "accord_degraded": _mean(x["accord_degraded"] for x in X),
                             "surrogate_improved": _mean(x["surrogate_improved"] for x in X), "profiled_improved": _mean(x["profiled_improved"] for x in X),
                             "surrogate_up_profiled_down": _mean(x["surrogate_improved"] and x["final"]["profiled"] < n["profiled"] - 1e-9 for x, n in zip(X, N)),
                             "surrogate_up_accord_down": _mean(x["surrogate_improved"] and x["accord_degraded"] for x in X), "max_shift": _mean(x["max_shift"] for x in X),
                             "best_profiled_accord": _mean(x["best_profiled"]["accord"] for x in X), "best_profiled_ge_099": _mean(x["best_profiled"]["accord"] >= 0.99 for x in X), "best_profiled_step": _mean(x["best_profiled"]["step"] for x in X),
                             "best_profiled_is_initial": _mean(x["best_profiled"]["step"] == 0 for x in X), "best_surrogate_accord": _mean(x["best_surrogate"]["accord"] for x in X), "best_surrogate_ge_099": _mean(x["best_surrogate"]["accord"] >= 0.99 for x in X),
                             "seconds": _mean(x["seconds"] for x in X)}
    D = [s["configs"]["default"] for s in S]
    out["default"] = {"corr_surrogate_accord": _mean(d["corr_surrogate_accord"] for d in D), "corr_loss_accord": _mean(d["corr_loss_accord"] for d in D), "corr_profiled_accord": _mean(d["corr_profiled_accord"] for d in D),
                      "corr_surrogate_accord_negative": _mean((d["corr_surrogate_accord"] or 0) < 0 for d in D), "first_drop_step": _mean(d["first_drop_step"] for d in D), "drop_within_5_steps": _mean(d["first_drop_step"] is not None and d["first_drop_step"] <= 5 for d in D),
                      "never_drops": _mean(d["first_drop_step"] is None for d in D), "early_stop_accord": _mean(d["early_stop"]["accord"] for d in D), "early_stop_ge_099": _mean(d["early_stop"]["accord"] >= 0.99 for d in D),
                      "early_stop_step": _mean(d["early_stop"]["step"] for d in D), "matches_fit_batch_max": max(d["matches_fit_batch"] for d in D)}
    steps = sorted({t["step"] for d in D for t in d["trajectory"]}); out["trajectory"] = [{"step": st, **{k: _mean(t[k] for d in D for t in d["trajectory"] if t["step"] == st) for k in ("k", "accord", "errors", "surrogate_loglik", "profiled_loglik", "loss", "grad_norm", "distance", "max_shift")}} for st in steps]
    out["geometry"] = {k: {m: _mean(s["geometry"][k][m] for s in S) for m in ("grad_norm", "grad_per_point", "condition", "min_eig", "negative_eigs", "near_zero_eigs", "cross_atom_coupling", "saturated_fraction", "n_params")} for k in ("k4", "k24")}
    out["geometry"]["condition_median_k24"] = sorted(s["geometry"]["k24"]["condition"] for s in S)[len(S) // 2]; out["geometry"]["condition_gt_1e4_k24"] = _mean(s["geometry"]["k24"]["condition"] > 1e4 for s in S)
    out["geometry"]["has_negative_curvature_k24"] = _mean(s["geometry"]["k24"]["negative_eigs"] > 0 for s in S); out["geometry"]["has_flat_direction_k24"] = _mean(s["geometry"]["k24"]["near_zero_eigs"] > 0 for s in S)
    F = [s["fitters"] for s in S]
    out["fitters"] = {"param_diff_max": max(f["param_diff"] for f in F), "param_diff_gt_1e-6": _mean(f["param_diff"] > 1e-6 for f in F), "param_diff_gt_1e-3": _mean(f["param_diff"] > 1e-3 for f in F), "accord_old": _mean(f["old"]["accord"] for f in F), "accord_new": _mean(f["new"]["accord"] for f in F),
                      "accord_diff_max": max(abs(f["old"]["accord"] - f["new"]["accord"]) for f in F), "profiled_old": _mean(f["old"]["profiled"] for f in F), "profiled_new": _mean(f["new"]["profiled"] for f in F), "surrogate_old": _mean(f["old"]["surrogate"] for f in F),
                      "surrogate_new": _mean(f["new"]["surrogate"] for f in F), "errors_old": _mean(f["old"]["errors"] for f in F), "errors_new": _mean(f["new"]["errors"] for f in F), "same_099_outcome": _mean((f["old"]["accord"] >= 0.99) == (f["new"]["accord"] >= 0.99) for f in F),
                      "seconds_old": _mean(f["seconds_old"] for f in F), "seconds_new": _mean(f["seconds_new"] for f in F), "init_rounding": max(f["init_rounding"] for f in F)}
    f32 = [s["configs"]["float32"] for s in S]; out["float32_vs_float64"] = {"accord_diff_max": max(abs(a["final"]["accord"] - d["final"]["accord"]) for a, d in zip(f32, D)), "accord_diff_gt_0.002": _mean(abs(a["final"]["accord"] - d["final"]["accord"]) > 0.002 for a, d in zip(f32, D))}
    return out


def aggregate():
    rows, bad = load("deepvo"); S = {"arms": {"n_rows": len(rows), "units_in_error": bad}}
    for ts in ("diagnostic", "new"):
        x = [r for r in rows if r["task_set"] == ts]; S["arms"][ts] = {"all": {"B": arms_block(x, "B"), "C": arms_block(x, "C")}}
        for reg in ("standard", "rich"): S["arms"][ts][reg] = {"B": arms_block([r for r in x if r["regime"] == reg], "B")}
        for d in sorted({r["depth"] for r in x}): S["arms"][ts][f"d{d}"] = {"B": arms_block([r for r in x if r["depth"] == d], "B")}
        for reg in ("standard", "rich"):
            for d in sorted({r["depth"] for r in x}): S["arms"][ts][f"{reg}/d{d}"] = {"B": arms_block([r for r in x if r["depth"] == d and r["regime"] == reg], "B")}
    nb = S["arms"].get("new", {}).get("all", {}).get("B", {})
    if nb: S["arms"]["criterion"] = {"gain_recovery_099_points": round(100 * nb["diff_recovery_0.99"]["mean"], 1), "accord_diff": nb["diff_accord"]["mean"], "met": nb["diff_recovery_0.99"]["mean"] >= 0.20 and nb["diff_accord"]["mean"] >= 0.0}
    orow, obad = load("deepopt"); S["optimizer"] = {"n_tasks": len(orow), "units_in_error": obad}
    if orow:
        for st in ("true_parameters", "top_candidate"):
            S["optimizer"][st] = opt_block(orow, st)
            for reg in ("standard", "rich"): S["optimizer"][f"{st}/{reg}"] = {k: v for k, v in opt_block([r for r in orow if r["regime"] == reg], st).items() if k in ("n", "noop", "configs", "default")}
            for d in sorted({r["depth"] for r in orow}): S["optimizer"][f"{st}/d{d}"] = {k: v for k, v in opt_block([r for r in orow if r["depth"] == d], st).items() if k in ("n", "noop", "configs", "default")}
    json.dump(S, open(os.path.join(OUT, "deep_validated_summary.json"), "w"), indent=1); print("arm rows", len(rows), "errors", bad, "| optimizer tasks", len(orow), "errors", obad); return S


def report():
    S = json.load(open(os.path.join(OUT, "deep_validated_summary.json"))); A = S["arms"]
    for ts in ("diagnostic", "new"):
        print(f"\n===== ARMS — tasks {ts}")
        for k, v in A[ts].items():
            b = v["B"]
            if not b: continue
            print(f"{k.ljust(14)} n={b['n']:3d} A {b['A']['thresholds']} acc {b['A']['accord']} | B {b['B']['thresholds']} acc {b['B']['accord']}")
            print(" " * 14, "Δ0.99", b["diff_recovery_0.99"]["mean"], b["diff_recovery_0.99"]["ci95"], "d", b["diff_recovery_0.99"]["effect_size"], "| Δagreement", b["diff_accord"]["mean"], b["diff_accord"]["ci95"], "d", b["diff_accord"]["effect_size"],
                  "| impr/degr/unch", b["diff_accord"]["improved"], b["diff_accord"]["degraded"], b["diff_accord"]["unchanged"], "| Δlik", b["diff_likelihood"]["mean"])
        b = A[ts]["all"]["B"]; c = A[ts]["all"]["C"]
        print("table 0.99 (A,B)", b["recovery_099_table"], "| worst degradations", b["worst_degradations"]); print("detail A", b["A"]); print("detail B", b["B"])
        print("thresholds Δ", {t: (b[f'diff_recovery_{t}']["mean"], b[f'diff_recovery_{t}']["ci95"]) for t in ("0.9", "0.95", "0.97", "0.99")})
        print("arm C (surrogate criterion)", c["C"]["thresholds"], "acc", c["C"]["accord"], "Δ0.99", c["diff_recovery_0.99"]["mean"], c["diff_recovery_0.99"]["ci95"], "commits/rollbacks", c["C"]["commits"], c["C"]["rollbacks"])
    print("\npredefined criterion", A.get("criterion"))
    O = S["optimizer"]
    for st in ("true_parameters", "top_candidate"):
        if st not in O: continue
        o = O[st]; print(f"\n===== OPTIMIZER — start: {st} (n={o['n']})"); print("without optimizing", o["noop"], "≥0.99", o["noop_ge_099"])
        for c, v in o["configs"].items(): print(c.ljust(22), {k: v[k] for k in ("accord", "ge_099", "errors", "accord_degraded", "surrogate_improved", "profiled_improved", "surrogate_up_profiled_down", "surrogate_up_accord_down", "max_shift")})
        print("best state encountered (default)", {k: o["configs"]["default"][k] for k in ("best_profiled_accord", "best_profiled_ge_099", "best_profiled_step", "best_profiled_is_initial", "best_surrogate_accord", "best_surrogate_ge_099")})
        print("default", o["default"]); print("trajectory", [(t["step"], t["k"], t["accord"], t["errors"], t["surrogate_loglik"], t["profiled_loglik"], t["grad_norm"], t["max_shift"]) for t in o["trajectory"]])
        print("geometry", o["geometry"]); print("fitters", o["fitters"]); print("float32", o["float32_vs_float64"])
        for sub in ("standard", "rich", "d2", "d3", "d4", "d5"):
            x = O[f"{st}/{sub}"]; print(sub.ljust(10), "noop", x["noop"]["accord"], "default", x["configs"]["default"]["accord"], x["configs"]["default"]["ge_099"], "sur↑prof↓", x["configs"]["default"]["surrogate_up_profiled_down"], "corr", x["default"]["corr_surrogate_accord"], "mono", x["configs"]["monotone_k24"]["accord"], x["configs"]["monotone_k24"]["ge_099"])


if __name__ == "__main__":
    {"aggregate": aggregate, "report": report}[sys.argv[1]]()

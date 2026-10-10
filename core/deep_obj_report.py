"""Report of the sub-milestone Objective Alignment & Conditional Optimization (dev; never the sealed set).
  tune      : tunes the conditional gate on the TUNING set → results/deep_configs/optimization_gate.json
  aggregate : → results/deep_objective_summary.json (policies evaluated on the EVALUATION set)
  report    : tables
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import sys
from collections import Counter, defaultdict

from core.deep_optgate import GATE_FILE, OBJECTIVES, need_for_optimization
from core.deep_vo_report import _mean, paired

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
GOOD = 0.99
TOL = 1e-9
POLICIES = ("always", "never", "validated", "validated_best", "gate")
GRID = {"residual_z": (0.0, 1.0, 2.0, 3.0, 4.0), "gap_z": (-1.0, 0.0, 1.0, 2.0, 99.0), "min_gain": (0.0, 1.0, 2.0, 4.0)}


def load(kind):
    rows = []; bad = 0
    for f in glob.glob(os.path.join(OUT, "aws", f"wce-{kind}-*", "artifacts", "*", f"{kind}_*.json")):
        r = json.load(open(f))
        if r["status"] == "ok": rows += r["result"]
        else: bad += 1
    return rows, bad


def outcome(row, policy, objective="surrogate", gate=None):
    """Agreement (and action) obtained by a policy on a candidate, from the recorded states. The decisions
    use only observable quantities (profiled likelihood, signals); the agreement serves to MEASURE."""
    ini = row["initial"]; o = row["objectives"][objective]
    if policy == "never": return ini["accord"], "preserve"
    if policy == "always": return o["final"]["accord"], "optimize"
    if policy == "validated": return (o["final"]["accord"], "accept") if o["final"]["profiled"] > ini["profiled"] + TOL else (ini["accord"], "rollback")
    if policy == "validated_best": return (o["best"]["accord"], "accept") if o["best"]["profiled"] > ini["profiled"] + TOL else (ini["accord"], "rollback")
    if policy == "gate":
        action = need_for_optimization(row["signals"], gate)[0]
        if action == "PRESERVE": return ini["accord"], "preserve"
        return (o["best"]["accord"], "accept") if o["best"]["profiled"] > ini["profiled"] + max(TOL, gate["min_gain"]) else (ini["accord"], "rollback")
    raise KeyError(policy)


def rates(rows, policy, objective, gate=None):
    """Four requested rates: good ones preserved, bad ones improved, good ones destroyed, bad ones not repaired."""
    res = [(r["initial"]["accord"], *outcome(r, policy, objective, gate)) for r in rows]; good = [x for x in res if x[0] >= GOOD]; bad = [x for x in res if x[0] < GOOD]
    return {"n": len(res), "accord": _mean(x[1] for x in res), "accord_initial": _mean(x[0] for x in res), "ge_099": _mean(x[1] >= GOOD for x in res), "ge_097": _mean(x[1] >= 0.97 for x in res), "n_good": len(good), "n_bad": len(bad),
            "good_preserved": _mean(x[1] >= GOOD for x in good), "good_destroyed": _mean(x[1] < GOOD for x in good), "bad_improved": _mean(x[1] > x[0] + 0.002 for x in bad), "bad_not_repaired": _mean(x[1] <= x[0] + 0.002 for x in bad),
            "bad_worsened": _mean(x[1] < x[0] - 0.002 for x in bad), "bad_repaired_to_099": _mean(x[1] >= GOOD for x in bad), "optimized": _mean(x[2] in ("optimize", "accept", "rollback") for x in res), "accepted": _mean(x[2] in ("optimize", "accept") for x in res)}


def tune(write=True):
    """Tuning of the gate on the TUNING set only (keys 300–307): objective and thresholds that maximise the mean
    agreement, the three input regimes counting in equal shares."""
    rows, _ = load("deepobj"); T = [r for r in rows if r["task_set"] == "tuning"]; best = None; regs = sorted({r["candidate_regime"] for r in T})
    for obj in OBJECTIVES:
        for rz, gz, mg in itertools.product(GRID["residual_z"], GRID["gap_z"], GRID["min_gain"]):
            g = {"objective": obj, "residual_z": rz, "gap_z": gz, "min_gain": mg}; sc = sum(_mean(outcome(r, "gate", obj, g)[0] for r in T if r["candidate_regime"] == reg) for reg in regs) / len(regs)
            if best is None or sc > best[0] + 1e-12: best = (sc, g)
    gate = {**best[1], "tuning_score": round(best[0], 5), "tuned_on": f"{len(T)} candidates of the tuning set (dev, keys 300–307)"}
    if write: os.makedirs(os.path.dirname(GATE_FILE), exist_ok=True); json.dump(gate, open(GATE_FILE, "w"), indent=1)
    print("retained gate:", gate); return gate


def aggregate():
    rows, bad = load("deepobj"); E = [r for r in rows if r["task_set"] == "evaluation"]; T = [r for r in rows if r["task_set"] == "tuning"]
    gate = json.load(open(GATE_FILE)) if os.path.exists(GATE_FILE) else tune(write=False); S = {"n_candidates": len(rows), "n_evaluation": len(E), "n_tuning": len(T), "units_in_error": bad, "gate": gate}
    groups = {"A": lambda r: r["candidate_regime"] == "A", "B_low": lambda r: r["level"] == "low", "B_medium": lambda r: r["level"] == "medium", "B_high": lambda r: r["level"] == "high", "B": lambda r: r["candidate_regime"] == "B",
              "C": lambda r: r["candidate_regime"] == "C", "all": lambda r: True}
    S["objectives"] = {}                                       # alignment of each objective, per regime
    for obj in OBJECTIVES:
        S["objectives"][obj] = {}
        for gname, sel in groups.items():
            x = [r for r in E if sel(r)]; o = [r["objectives"][obj] for r in x]
            S["objectives"][obj][gname] = {"n": len(x), "accord_initial": _mean(r["initial"]["accord"] for r in x), "accord_final": _mean(v["final"]["accord"] for v in o), "delta_accord": _mean(v["final"]["accord"] - r["initial"]["accord"] for v, r in zip(o, x)),
                                           "improved": _mean(v["final"]["accord"] > r["initial"]["accord"] + 0.002 for v, r in zip(o, x)), "degraded": _mean(v["final"]["accord"] < r["initial"]["accord"] - 0.002 for v, r in zip(o, x)),
                                           "objective_improved": _mean(v["objective_improved"] for v in o), "objective_up_accord_down": _mean(v["objective_improved"] and v["final"]["accord"] < r["initial"]["accord"] - 0.002 for v, r in zip(o, x)),
                                           "corr_objective_accord": _mean(v["corr_objective_accord"] for v in o), "errors_delta": _mean(v["errors_delta"] for v in o), "profiled_improved": _mean(v["final"]["profiled"] > r["initial"]["profiled"] + TOL for v, r in zip(o, x)),
                                           "ge_099_final": _mean(v["final"]["accord"] >= GOOD for v in o), "max_shift": _mean(v["max_shift"] for v in o), "seconds": _mean(v["seconds"] for v in o)}
    S["policies"] = {}                                         # policies, current objective and objective of the gate
    for obj in sorted({"surrogate", gate["objective"]}):
        S["policies"][obj] = {gname: {p: rates([r for r in E if sel(r)], p, obj, gate) for p in POLICIES} for gname, sel in groups.items()}
    S["policies_by_objective_all"] = {obj: {p: {k: v for k, v in rates(E, p, obj, {**gate, "objective": obj}).items() if k in ("accord", "ge_099", "good_preserved", "good_destroyed", "bad_improved", "bad_not_repaired", "bad_worsened")} for p in POLICIES} for obj in OBJECTIVES}
    S["policies_by_depth"] = {f"{reg}/d{d}": {p: {k: v for k, v in rates([r for r in E if r["candidate_regime"] == reg and r["depth"] == d], p, gate["objective"], gate).items() if k in ("accord", "ge_099", "good_destroyed", "bad_improved")} for p in POLICIES}
                              for reg in ("A", "B", "C") for d in sorted({r["depth"] for r in E})}
    S["perturbation_curve"] = {}                               # initial parameter error → expected benefit of the optimisation
    for lvl in ("true", "low", "medium", "high"):
        x = [r for r in E if r["level"] == lvl]
        S["perturbation_curve"][lvl] = {"n": len(x), "param_error": _mean(r["param_error"] for r in x), "accord_initial": _mean(r["initial"]["accord"] for r in x),
                                        "benefit": {obj: {p: _mean(outcome(r, p, obj, {**gate, "objective": obj})[0] - r["initial"]["accord"] for r in x) for p in ("always", "validated", "validated_best", "gate")} for obj in OBJECTIVES}}
    sig = defaultdict(list)                                    # do the signals distinguish need and danger?
    for r in E:
        b = r["objectives"][gate["objective"]]["best"]["accord"] - r["initial"]["accord"]; sig["helped" if b > 0.002 else "harmed" if b < -0.002 else "neutral"].append(r["signals"])
    S["signals_by_outcome"] = {k: {"n": len(v), **{s: _mean(x[s] for x in v) for s in ("error_rate", "residual_z", "gap_z", "grad_per_point", "probe_delta_errors", "profiled_per_point")}} for k, v in sig.items()}
    S["gate_decisions"] = {gname: dict(Counter(need_for_optimization(r["signals"], gate)[0] for r in E if sel(r))) for gname, sel in groups.items()}
    e2e, bad2 = load("deepe2e"); S["e2e"] = {"n_rows": len(e2e), "units_in_error": bad2}
    if e2e:
        by = defaultdict(dict)
        for r in e2e: by[(r["task_id"], r["literals"])][r["policy"]] = r
        for lit in ("discovered", "true"):
            S["e2e"][lit] = {}
            for gname, sel in {"all": lambda r: True, "standard": lambda r: r["regime"] == "standard", "rich": lambda r: r["regime"] == "rich", **{f"d{d}": (lambda r, d=d: r["depth"] == d) for d in (2, 3, 4, 5)}}.items():
                T2 = [v for (tid, l), v in by.items() if l == lit and len(v) == 4 and sel(v["always"])]
                if not T2: continue
                blk = {"n": len(T2)}
                for p in ("always", "never", "validated", "gate"):
                    blk[p] = {"thresholds": {t: _mean(v[p]["thresholds"][t] for v in T2) for t in ("0.9", "0.95", "0.97", "0.99")}, "accord": _mean(v[p]["accord"] for v in T2), "likelihood": _mean(v[p]["likelihood"] for v in T2), "atoms": _mean(v[p]["atoms"] for v in T2),
                              "exact": _mean(v[p]["exact"] for v in T2), "fit_equivalents": _mean(v[p]["fit_equivalents"] for v in T2), "seconds": _mean(v[p]["seconds"] for v in T2), "preserved": _mean(v[p]["preserved"] for v in T2), "decisions": _mean(v[p]["decisions"] for v in T2),
                              "rollbacks": _mean(v[p]["rollbacks"] + v[p]["gate_rollbacks"] for v in T2)}
                    if p != "always": blk[f"{p}_vs_always"] = {"accord": paired([v[p]["accord"] - v["always"]["accord"] for v in T2]), "recovery_099": paired([float(v[p]["thresholds"]["0.99"]) - float(v["always"]["thresholds"]["0.99"]) for v in T2]),
                                                               "recovery_097": paired([float(v[p]["thresholds"]["0.97"]) - float(v["always"]["thresholds"]["0.97"]) for v in T2])}
                blk["gate_vs_validated"] = {"accord": paired([v["gate"]["accord"] - v["validated"]["accord"] for v in T2])}; blk["validated_vs_never"] = {"accord": paired([v["validated"]["accord"] - v["never"]["accord"] for v in T2])}
                S["e2e"][lit][gname] = blk
    res, bad3 = load("deepresid"); S["residual"] = {"n": len(res), "units_in_error": bad3, "explanations": dict(Counter(r["explanation"] for r in res)),
                                                   "cases": [{k: r[k] for k in ("task_id", "explanation", "accepted", "accepted_harmful", "accepted_harmful_on_truth_form", "rollbacks", "rollback_lost_gain", "same_final_structure", "data_prefer_B", "evidence_prefer_B")} |
                                                             {"A": {k: r["A"][k] for k in ("accord", "atoms", "profiled", "evidence", "errors", "is_truth_form")}, "B": {k: r["B"][k] for k in ("accord", "atoms", "profiled", "evidence", "errors", "is_truth_form")}} for r in res]}
    json.dump(S, open(os.path.join(OUT, "deep_objective_summary.json"), "w"), indent=1); print("candidates", len(rows), "errors", bad, "| e2e", len(e2e), "| residual", len(res)); return S


def report():
    S = json.load(open(os.path.join(OUT, "deep_objective_summary.json"))); print("gate", S["gate"], "| candidates evaluation", S["n_evaluation"], "tuning", S["n_tuning"])
    print("\n=== ALIGNMENT OF THE OBJECTIVES (always optimise with this objective)")
    for g in ("A", "B_low", "B_medium", "B_high", "C"):
        print(f"\n regime {g} (initial agreement {S['objectives']['surrogate'][g]['accord_initial']}, n={S['objectives']['surrogate'][g]['n']})")
        for o, v in S["objectives"].items(): x = v[g]; print("  ", o.ljust(18), {k: x[k] for k in ("accord_final", "delta_accord", "improved", "degraded", "objective_improved", "objective_up_accord_down", "corr_objective_accord", "errors_delta", "ge_099_final")})
    print("\n=== POLICIES (per objective, all regimes)")
    for o, v in S["policies_by_objective_all"].items(): print(o.ljust(18), {p: (x["accord"], x["ge_099"]) for p, x in v.items()})
    for obj, P in S["policies"].items():
        print(f"\n=== POLICIES — objective {obj}")
        for g in ("A", "B_low", "B_medium", "B_high", "C", "all"):
            print(f" regime {g}: n={P[g]['always']['n']} good={P[g]['always']['n_good']} bad={P[g]['always']['n_bad']} initial agreement {P[g]['always']['accord_initial']}")
            for p in POLICIES: x = P[g][p]; print("   ", p.ljust(15), {k: x[k] for k in ("accord", "ge_099", "ge_097", "good_preserved", "good_destroyed", "bad_improved", "bad_not_repaired", "bad_worsened", "bad_repaired_to_099", "optimized", "accepted")})
    print("\n=== per depth (objective of the gate)"); [print(k.ljust(8), {p: (x["accord"], x["ge_099"]) for p, x in v.items()}) for k, v in S["policies_by_depth"].items()]
    print("\n=== PERTURBATION CURVE")
    for lvl, v in S["perturbation_curve"].items(): print(lvl.ljust(8), "param error", v["param_error"], "initial agreement", v["accord_initial"], {o: b for o, b in v["benefit"].items() if o in ("surrogate", S["gate"]["objective"], "discrete_error", "likelihood_prior")})
    print("\n=== signals by outcome", S["signals_by_outcome"]); print("decisions of the gate", S["gate_decisions"])
    if "discovered" in S["e2e"]:
        for lit in ("discovered", "true"):
            print(f"\n=== END TO END — literals {lit}")
            for g, b in S["e2e"][lit].items():
                print(f" {g} n={b['n']}"); [print("   ", p.ljust(10), b[p]["thresholds"], "acc", b[p]["accord"], "lik", b[p]["likelihood"], "fe", b[p]["fit_equivalents"], "s", b[p]["seconds"]) for p in ("always", "never", "validated", "gate")]
                for p in ("never", "validated", "gate"): d = b[f"{p}_vs_always"]; print("      ", p, "− always: agreement", d["accord"]["mean"], d["accord"]["ci95"], "| 0.99", d["recovery_099"]["mean"], d["recovery_099"]["ci95"], "| impr/degr", d["accord"]["improved"], d["accord"]["degraded"])
                print("       gate − validated", b["gate_vs_validated"]["accord"]["mean"], b["gate_vs_validated"]["accord"]["ci95"], "| validated − never", b["validated_vs_never"]["accord"]["mean"], b["validated_vs_never"]["accord"]["ci95"])
    print("\n=== RESIDUAL DEGRADATIONS", S["residual"]["explanations"]); [print("  ", c) for c in S["residual"]["cases"]]


if __name__ == "__main__":
    {"tune": tune, "aggregate": aggregate, "report": report}[sys.argv[1]]()

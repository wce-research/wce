"""Rapport du sous-jalon Downstream Selection Loss (dev, vrais atomes fournis ; jamais le lot scellé).
  aggregate : runs AWS `deepdown` → results/deep_downstream_summary.json
  report    : tableaux
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
STAGES = ("true_atoms", "candidate_generation", "initial_fit", "refit_polish", "simplification", "canonicalization", "scoring", "ranking", "final_selection")
SCORES = ("posterior_score", "likelihood", "robust_likelihood", "mdl", "complexity_penalty", "structural_penalty")


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]; return round(sum(xs) / len(xs), 4) if xs else None


def load_rows():
    rows = []; bad = 0
    for f in glob.glob(os.path.join(OUT, "aws", "wce-deepdown-*", "artifacts", "*", "deepdown_*.json")):
        r = json.load(open(f))
        if r["status"] == "ok": rows += r["result"]
        else: bad += 1
    return rows, bad


def calibration(conf, hit, bins=5):
    """ECE, Brier et courbe de fiabilité de la confiance interne (masse de la structure retenue)."""
    n = len(conf)
    if not n: return {}
    rel = []; ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins; idx = [i for i in range(n) if lo <= conf[i] < hi or (b == bins - 1 and conf[i] == 1.0)]
        if idx: c = sum(conf[i] for i in idx) / len(idx); a = sum(hit[i] for i in idx) / len(idx); rel.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(idx), "confidence": round(c, 3), "recovery": round(a, 3)}); ece += len(idx) / n * abs(c - a)
    return {"ece": round(ece, 4), "brier": round(sum((c - h) ** 2 for c, h in zip(conf, hit)) / n, 4), "reliability": rel, "mean_confidence": round(sum(conf) / n, 3), "recovery_rate": round(sum(hit) / n, 3)}


def aggregate():
    rows, bad = load_rows(); S = {"n_tasks": len(rows), "units_in_error": bad, "cells": {}, "overall": {}}
    def block(x):
        pol = [p for r in x for p in r["polish"]]; good = [p for p in pol if p["accord_before"] >= 0.97]; truthp = [p for p in pol if p["accord_before"] >= 0.999]
        b = {"n": len(x), "causes": dict(Counter(r["cause"] for r in x)), "thresholds": {t: _mean(r["thresholds"][t] for r in x) for t in ("0.9", "0.95", "0.97", "0.99")}, "final_exact": _mean(r["final_exact"] for r in x), "final_equivalent": _mean(r["final_equivalent"] for r in x),
             "functional_not_equivalent": _mean(r["functional_not_equivalent"] for r in x), "final_accord": _mean(r["final_accord"] for r in x), "final_in_region": _mean(r["final_in_region"] for r in x),
             "trajectory": {s: _mean(r["trajectory"][s]["accord"] for r in x) for s in STAGES}, "good_at_stage": {s: _mean(r["trajectory"][s]["accord"] >= 0.99 for r in x) for s in STAGES},
             "good_found": _mean(r["trajectory_summary"]["good_found_at"] is not None for r in x), "destroyed": _mean(r["trajectory_summary"]["destroyed"] for r in x), "good_lost_at": dict(Counter(r["trajectory_summary"]["good_lost_at"] for r in x)),
             "good_candidate_generated": _mean(r["n_good_candidates"] > 0 for r in x), "good_candidate_is_top_prefit": _mean(r["good_candidate_rank"] == 0 for r in x if r["n_good_candidates"] > 0),
             "polish": {"n_good_before": len(good), "accord_before": _mean(p["accord_before"] for p in good), "accord_after": _mean(p["accord_after"] for p in good), "degraded": _mean(p["accord_after"] < p["accord_before"] - 0.002 for p in good),
                        "degraded_below_099": _mean(p["accord_after"] < 0.99 for p in truthp), "n_perfect_before": len(truthp), "likelihood_worse": _mean(p["likelihood_after"] < p["likelihood_before"] - 1e-9 for p in good), "errors_delta": _mean(p["errors_after"] - p["errors_before"] for p in good),
                        "param_shift": _mean(p["param_shift"] for p in good), "accord_drop": _mean(p["accord_before"] - p["accord_after"] for p in good)},
             "simplification": {"changed": _mean(r["simplification"]["changed"] for r in x), "removed_needed_atom": _mean(r["simplification"]["removed_needed_atom"] for r in x),
                                "accord_delta_when_changed": _mean(r["simplification"]["accord_after"] - r["simplification"]["accord_before"] for r in x if r["simplification"]["changed"]), "n_changed": sum(r["simplification"]["changed"] for r in x)},
             "canonicalization_changed_behavior": sum(r["canonicalization_changed_behavior"] for r in x), "dedup_removed": _mean(r["dedup_removed"] for r in x),
             "oracle_ranking": {s: {"spearman": _mean(r["oracle_ranking"][s]["spearman"] for r in x), "top1_agreement": _mean(r["oracle_ranking"][s]["top1_agreement"] for r in x), "top3_recall": _mean(r["oracle_ranking"][s]["top3_recall"] for r in x),
                                    "top5_recall": _mean(r["oracle_ranking"][s]["top5_recall"] for r in x), "top1_accord": _mean(r["oracle_ranking"][s]["top1_accord"] for r in x)} for s in SCORES},
             "best_minus_selected": {k: _mean(r["best_minus_selected"][k] for r in x) for k in ("likelihood", "robust_likelihood", "complexity_penalty", "structural_penalty", "posterior_score", "errors")},
             "truth_fit": {"accord": _mean(r["truth_fit"]["accord"] for r in x), "ge_099": _mean(r["truth_fit"]["accord"] >= 0.99 for r in x), "fit_improved_likelihood": _mean(r["truth_fit"]["fit_improved_likelihood"] for r in x),
                           "errors_true_params": _mean(r["truth_fit"]["errors_true_params"] for r in x), "errors_fitted": _mean(r["truth_fit"]["errors_fitted"] for r in x)},
             "counterfactual": {k: {"accord": _mean(r["counterfactual"][k] for r in x), "ge_099": _mean(r["counterfactual"][k] >= 0.99 for r in x), "ge_097": _mean(r["counterfactual"][k] >= 0.97 for r in x)} for k in ("hold_best", "no_simplification", "no_dedup", "pre_fit_top_score", "best_available")},
             "fitter": {"max_param_error": max((r["fitter"].get("max_param_error", 0.0) for r in x), default=None), "param_error_gt_1e-3": _mean(r["fitter"].get("max_param_error", 0.0) > 1e-3 for r in x), "same_top1": _mean(r["fitter"].get("same_top1") for r in x if r["fitter"].get("n")),
                        "same_order": _mean(r["fitter"].get("same_order") for r in x if r["fitter"].get("n")), "rank_correlation": _mean(r["fitter"].get("rank_correlation") for r in x), "max_accord_diff": max((r["fitter"].get("max_accord_diff", 0.0) for r in x), default=None),
                        "max_score_diff": max((r["fitter"].get("max_score_diff", 0.0) for r in x), default=None)},
             "calibration": calibration([r["confidence"] for r in x], [float(r["final_accord"] >= 0.99) for r in x]), "calibration_097": calibration([r["confidence"] for r in x], [float(r["final_accord"] >= 0.97) for r in x]),
             "seconds": _mean(r["seconds"] for r in x), "fit_equivalents": _mean(r["fit_equivalents"] for r in x)}
        return b
    for reg in ("standard", "rich"):
        for d in sorted({r["depth"] for r in rows}):
            x = [r for r in rows if r["regime"] == reg and r["depth"] == d]
            if x: S["cells"][f"{reg}/d{d}"] = block(x)
        x = [r for r in rows if r["regime"] == reg]
        if x: S["overall"][reg] = block(x)
    if rows: S["overall"]["all"] = block(rows)
    json.dump(S, open(os.path.join(OUT, "deep_downstream_summary.json"), "w"), indent=1); print("tâches", len(rows), "unités en erreur", bad); return S


def report():
    S = json.load(open(os.path.join(OUT, "deep_downstream_summary.json"))); C = S["cells"]; keys = list(C)
    print("\ncauses par cellule"); [print(k.ljust(14), C[k]["n"], C[k]["causes"]) for k in keys]
    print("\nseuils (part des tâches dont l'accord final ≥ seuil) + exact / équivalent"); [print(k.ljust(14), C[k]["thresholds"], "exact", C[k]["final_exact"], "équiv", C[k]["final_equivalent"], "accord", C[k]["final_accord"]) for k in keys]
    print("\ntrajectoire : part des tâches avec un candidat ≥ 0,99 à chaque étape"); print("".ljust(14) + " ".join(s[:9].rjust(10) for s in STAGES))
    for k in keys: print(k.ljust(14) + " ".join(f"{C[k]['good_at_stage'][s]:.2f}".rjust(10) for s in STAGES))
    print("\ntrajectoire : meilleur accord moyen par étape")
    for k in keys: print(k.ljust(14) + " ".join(f"{C[k]['trajectory'][s]:.3f}".rjust(10) for s in STAGES))
    print("\nétape où la bonne solution est perdue"); [print(k.ljust(14), C[k]["good_lost_at"], "détruite", C[k]["destroyed"]) for k in keys]
    for name in ("polish", "simplification", "truth_fit", "fitter"):
        print(f"\n{name}"); [print(k.ljust(14), C[k][name]) for k in keys + []]; print("TOUT".ljust(14), S["overall"]["all"][name])
    print("\ncontrefactuels (part ≥ 0,99)"); [print(k.ljust(14), {c: v["ge_099"] for c, v in C[k]["counterfactual"].items()}, "| actuel", C[k]["thresholds"]["0.99"]) for k in keys]
    print("TOUT".ljust(14), {c: (v["ge_099"], v["ge_097"], v["accord"]) for c, v in S["overall"]["all"]["counterfactual"].items()}, "| actuel", S["overall"]["all"]["thresholds"])
    print("\nclassement oracle (tout)"); [print(s.ljust(22), v) for s, v in S["overall"]["all"]["oracle_ranking"].items()]
    print("\nmeilleur disponible − retenu (tout)", S["overall"]["all"]["best_minus_selected"])
    print("\ncalibration ≥0,99", S["overall"]["all"]["calibration"]); print("calibration ≥0,97", S["overall"]["all"]["calibration_097"])
    print("\ncauses (tout)", S["overall"]["all"]["causes"], "| standard", S["overall"]["standard"]["causes"], "| riche", S["overall"]["rich"]["causes"])
    print("divers (tout)", {k: S["overall"]["all"][k] for k in ("canonicalization_changed_behavior", "dedup_removed", "functional_not_equivalent", "good_candidate_generated", "good_candidate_is_top_prefit", "seconds", "fit_equivalents")})


if __name__ == "__main__":
    {"aggregate": aggregate, "report": report}[sys.argv[1]]()

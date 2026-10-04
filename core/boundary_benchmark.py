"""Banc multi-graines + lot scellé + hypothèses pré-enregistrées (docs/hypotheses_boundary_induction.md)."""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict

import torch

from core.boundary_methods import METHODS, evaluate_all
from core.boundary_world import (HELD_OUT, PARTIAL, SEALED, UNREPRESENTABLE, VALIDATION, balanced_inst, composed_instance, double_held_out,
                                 instances, make_instance, sample_task, sealed_spec)

JOBS = ("main", "family", "recall", "active", "anchor", "crystal", "residual", "shared", "budget", "overfit", "stability", "confidence", "cross_domain", "ablation", "depth")


def main_tasks(seed):
    rng = random.Random(seed); g = torch.Generator().manual_seed(seed)
    T = [(i, "validation") for i in instances(VALIDATION, seed + 2000, 1)] + [(i, "held_out") for i in instances(HELD_OUT, seed + 2001, 2)] + \
        [(i, "partial") for i in instances(PARTIAL, seed + 2002, 1)] + [(i, "unrepresentable") for i in instances(UNREPRESENTABLE, seed + 2003, 1)] + \
        [(balanced_inst(composed_instance, g, rng), "held_out_composition") for _ in range(2)] + \
        [(balanced_inst(lambda r: double_held_out(r.choice(["xor_halfspaces", "crescent"]), r), g, rng), "double_held_out") for _ in range(1)]
    return T


def run_job(args):
    seed, job = args; torch.set_num_threads(1)
    import core.boundary_experiments as E
    if job == "main":
        lib, _ = E.induce_family_library(seed); macros = [f.macro.skeleton for f in lib if f.status in ("STABLE", "SUPPORTED")]
        rows = []; g = torch.Generator().manual_seed(seed + 3)
        for inst, cat in main_tasks(seed):
            X, y = sample_task(inst, 200, g); R = evaluate_all(inst, X, y, g, macros)
            for m, r in R.items():
                if r is not None: rows.append({"family": inst.family, "category": cat, "representable": inst.representable, "method": m,
                                               **{k: v for k, v in r.items() if k != "program"}, "program": r.get("program")})
        return seed, job, {"rows": rows, "library": [(f.status, len(f.support), str(f.macro.skeleton)) for f in lib]}
    if job == "family":
        lib, n = E.induce_family_library(seed); return seed, job, {"reuse": E.family_reuse_experiment(seed, lib), "novelty": E.novelty_detection(seed, lib),
                                                                   "library": [(f.status, len(f.support)) for f in lib], "programs": n}
    if job == "recall": return seed, job, E.discovery_recall(seed)
    if job == "active": return seed, job, E.active_identification(seed)
    if job == "anchor": return seed, job, E.truth_anchor_experiment(seed)
    if job == "crystal": return seed, job, E.crystallization_experiment(seed)
    if job == "residual": return seed, job, E.residual_experiment(seed)
    if job == "shared": return seed, job, E.shared_basis_experiment(seed)
    if job == "budget": return seed, job, [E.hard_budget_stream(seed, mode="hard"), E.hard_budget_stream(seed, mode="soft")]
    if job == "overfit": return seed, job, E.symbolic_overfitting(seed)
    if job == "stability": return seed, job, E.program_stability(seed)
    if job == "confidence": return seed, job, E.confidence_calibration(seed)
    if job == "cross_domain":
        from core.program_induction_core import cross_domain_experiment
        return seed, job, cross_domain_experiment(seed)
    if job == "ablation": return seed, job, E.primitive_ablation(seed) if seed % 4 == 0 else None
    if job == "depth": return seed, job, E.search_depth_study(seed) if seed % 4 == 0 else None
    raise KeyError(job)


# ================================================================ agrégation

def _ci(xs):
    xs = [float(x) for x in xs if x is not None]; n = len(xs)
    if n == 0: return None
    m = sum(xs) / n
    if n < 2: return {"mean": round(m, 4), "ci95": [round(m, 4), round(m, 4)], "n": n}
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)); h = (2.131 if n <= 16 else 2.04 if n <= 32 else 1.96) * sd / math.sqrt(n)
    return {"mean": round(m, 4), "ci95": [round(m - h, 4), round(m + h, 4)], "n": n}


def aggregate(results):
    by = defaultdict(dict)
    for seed, job, r in results: by[job][seed] = r
    seeds = sorted(by["main"]); S = {}
    def per_seed(job, f): return [f(by[job][s]) for s in sorted(by[job]) if by[job][s] is not None]
    # --- principal : par méthode × catégorie
    cats = ("validation", "held_out", "partial", "unrepresentable", "held_out_composition", "double_held_out")
    S["main"] = {}
    for m in METHODS:
        S["main"][m] = {}
        for c in cats:
            def mean_of(key, s, m=m, c=c):
                v = [float(r[key]) for r in by["main"][s]["rows"] if r["method"] == m and r["category"] == c and r.get(key) is not None]
                return sum(v) / len(v) if v else None
            S["main"][m][c] = {k: _ci([mean_of(k, s) for s in seeds]) for k in ("in_region", "extrapolation", "functional_recovery", "structural_recovery", "adequate", "params", "evaluations", "seconds", "queries", "program_length")}
    held = lambda m, k, s: [float(r[k]) for r in by["main"][s]["rows"] if r["method"] == m and r["category"] in ("held_out", "held_out_composition") and r.get(k) is not None]
    S["held_out_families_recovered"] = Counter(r["family"] for s in seeds for r in by["main"][s]["rows"] if r["method"] == "bpi_full" and r["category"] == "held_out" and r["functional_recovery"])
    S["extrapolation_diff"] = {f"bpi_full_vs_{b}": _ci([sum(held("bpi_full", "extrapolation", s)) / len(held("bpi_full", "extrapolation", s)) - sum(held(b, "extrapolation", s)) / len(held(b, "extrapolation", s)) for s in seeds])
                               for b in ("mlp", "memorization", "knn", "search_no_au")}
    fam = per_seed("family", lambda r: r)
    S["family_reuse"] = {k: _ci([sum(x[k] for x in f["reuse"]) / len(f["reuse"]) for f in fam]) for k in ("macro_acc", "search_acc", "macro_evals", "search_evals", "macro_seconds", "search_seconds")}
    nov = [row for f in fam for row in f["novelty"]]
    S["novelty"] = {"accuracy": round(sum(r["truth"] == r["decided"] for r in nov) / len(nov), 3), "confusion": dict(Counter(f"{r['truth']}->{r['decided']}" for r in nov))}
    S["family_library"] = Counter(st for f in fam for st, _ in f["library"])
    rec = per_seed("recall", lambda r: r)
    S["discovery"] = {"recall": _ci([r["recall"] for r in rec]), "false_promotions": _ci([r["false_promotions"] for r in rec]), "promoted": _ci([r["promoted"] for r in rec]),
                      "families_recovered": Counter(f for r in rec for f in r["recovered_families"])}
    act = [row for r in per_seed("active", lambda r: r) for row in r]
    S["active"] = {}
    for pol in ("ig", "disagreement", "random"):
        q = sorted(r["queries"] for r in act if r["policy"] == pol); S["active"][pol] = {"median_queries": q[len(q) // 2], "identified": round(sum(r["identified"] for r in act if r["policy"] == pol) / len(q), 3), "n": len(q)}
    anc = [row for r in per_seed("anchor", lambda r: r) for row in r]
    S["anchor"] = {}
    for pol in ("active", "random"):
        w = [r for r in anc if r["policy"] == pol and r["wrong"]]; c = [r for r in anc if r["policy"] == pol and not r["wrong"]]
        S["anchor"][pol] = {"refutation_of_wrong": round(sum(r["refuted"] for r in w) / max(1, len(w)), 3), "false_refutation": round(sum(r["refuted"] for r in c) / max(1, len(c)), 3),
                            "wrong_self_consistent": round(sum(r["self_consistent"] for r in w) / max(1, len(w)), 3), "n_wrong": len(w), "n_correct": len(c)}
    cry = [row for r in per_seed("crystal", lambda r: r) for row in r]
    rep = [r for r in cry if r["representable"]]
    S["crystallization"] = {"replaced_fraction_representable": round(sum(r["decision"] == "replace" for r in rep) / max(1, len(rep)), 3),
                            "decisions": dict(Counter(r["decision"] for r in cry)), "decisions_unrepresentable": dict(Counter(r["decision"] for r in cry if not r["representable"])),
                            "acc_loss_replaced": _ci([r["expert_acc"] - r["acc"] for r in cry if r["decision"] == "replace"]),
                            "params_expert": _ci([r["expert_params"] for r in cry]), "params_after": _ci([r["params"] for r in cry]),
                            "infer_speedup": _ci([r["expert_infer_s"] / max(1e-9, r["program_infer_s"]) for r in cry if "program_infer_s" in r])}
    res = [row for r in per_seed("residual", lambda r: r) for row in r]
    S["residual"] = {k: {"in": _ci([r[k][0] for r in res]), "extrapolation": _ci([r[k][1] for r in res]), "params": _ci([r[k][2] for r in res])} for k in ("expert", "program", "program_residual")}
    sh = per_seed("shared", lambda r: r); S["shared_basis"] = {k: _ci([r[k] for r in sh]) for k in sh[0]}
    bud = per_seed("budget", lambda r: r)
    S["budget"] = {mode: {"violated": sum(r[i]["budget_violated"] for r in bud), "skills": _ci([r[i]["skills"] for r in bud]), "acc": _ci([r[i]["mean_acc"] for r in bud]),
                          "used": _ci([r[i]["used"] for r in bud]), "operations": dict(sum((Counter(r[i]["operations"]) for r in bud), Counter()))} for i, mode in enumerate(("hard", "soft"))}
    ovf = [row for r in per_seed("overfit", lambda r: r) for row in r]
    S["overfitting"] = {"overfit_rate": round(sum(r["overfit"] for r in ovf) / max(1, len(ovf)), 3), "detected_given_overfit": round(sum(r["detected"] for r in ovf if r["overfit"]) / max(1, sum(r["overfit"] for r in ovf)), 3),
                        "false_detection": round(sum(r["detected"] for r in ovf if not r["overfit"]) / max(1, sum(not r["overfit"] for r in ovf)), 3)}
    stb = [row for r in per_seed("stability", lambda r: r) for row in r]
    S["stability"] = {"same_structure": round(sum(r["same_structure"] for r in stb) / max(1, len(stb)), 3), "functional_agreement": _ci([r["functional_agreement"] for r in stb])}
    cf = per_seed("confidence", lambda r: r)
    S["confidence"] = {"prediction_ece": _ci([r["prediction_ece"] for r in cf]),
                       "family_conf_when_structure_correct": _ci([x["family_confidence"] for r in cf for x in r["rows"] if x["structure_correct"]]),
                       "family_conf_when_structure_wrong": _ci([x["family_confidence"] for r in cf for x in r["rows"] if not x["structure_correct"]]),
                       "param_sd": _ci([x["param_sd"] for r in cf for x in r["rows"]])}
    cd = per_seed("cross_domain", lambda r: r)
    S["cross_domain"] = {d: {k: _ci([r[d][k] for r in cd]) for k in ("solved", "generalizes", "family_reuse_success", "family_evals", "full_search_evals")} for d in cd[0]}
    ab = per_seed("ablation", lambda r: r); S["ablation"] = {k: _ci([r[k][0] for r in ab]) for k in ab[0]} if ab else {}
    dp = per_seed("depth", lambda r: r); S["depth"] = {str(d): {k: _ci([r[str(d) if str(d) in r else d][k] for r in dp]) for k in ("coverage", "evaluations", "seconds")} for d in (1, 2, 3, 4, 5)} if dp else {}
    S["hypotheses"] = hypotheses(S, by, seeds)
    return S


def hypotheses(S, by, seeds):
    H = {}; M = S["main"]
    fr = M["bpi_full"]["held_out"]["functional_recovery"]["mean"]; nfam = len([f for f, c in S["held_out_families_recovered"].items() if c > 0])
    H["H1"] = {"value": fr, "families_recovered": dict(S["held_out_families_recovered"]), "verdict": "SOUTENUE" if fr >= 0.6 and nfam >= 3 else "RÉFUTÉE" if fr < 0.3 else "NON CONCLUANTE"}
    dm, dn = S["extrapolation_diff"]["bpi_full_vs_mlp"], S["extrapolation_diff"]["bpi_full_vs_memorization"]
    H["H2"] = {"vs_mlp": dm, "vs_memorization": dn, "verdict": "SOUTENUE" if dm["mean"] > 0.05 and dn["mean"] > 0.05 and dm["ci95"][0] > 0 and dn["ci95"][0] > 0 else "RÉFUTÉE" if dm["ci95"][0] <= 0 and dm["mean"] <= 0 else "NON CONCLUANTE"}
    fr_ = S["family_reuse"]
    ok3 = fr_["macro_acc"]["mean"] >= fr_["search_acc"]["mean"] - 0.02 and fr_["macro_evals"]["mean"] <= 0.5 * fr_["search_evals"]["mean"]
    H["H3"] = {"value": fr_, "verdict": "SOUTENUE" if ok3 else "RÉFUTÉE" if fr_["macro_acc"]["mean"] < fr_["search_acc"]["mean"] - 0.05 else "NON CONCLUANTE"}
    a, r = S["active"]["ig"], S["active"]["random"]
    H["H4"] = {"ig": a, "random": r, "verdict": "SOUTENUE" if a["median_queries"] <= 0.6 * r["median_queries"] and a["identified"] >= r["identified"] else "RÉFUTÉE" if a["median_queries"] >= r["median_queries"] else "NON CONCLUANTE"}
    an = S["anchor"]["active"]
    H["H5"] = {"value": an, "verdict": "SOUTENUE" if an["refutation_of_wrong"] >= 0.7 and an["false_refutation"] <= 0.1 else "RÉFUTÉE" if an["refutation_of_wrong"] < 0.3 else "NON CONCLUANTE"}
    cr = S["crystallization"]; loss = cr["acc_loss_replaced"]["mean"] if cr["acc_loss_replaced"] else 1.0
    H["H6"] = {"fraction": cr["replaced_fraction_representable"], "loss": loss, "verdict": "SOUTENUE" if cr["replaced_fraction_representable"] >= 0.4 and loss <= 0.02 else "RÉFUTÉE" if cr["replaced_fraction_representable"] < 0.15 or loss > 0.05 else "NON CONCLUANTE"}
    rs = S["residual"]; ea, pa = rs["expert"]["in"]["mean"], rs["program_residual"]["in"]["mean"]; ep, pp = rs["expert"]["params"]["mean"], rs["program_residual"]["params"]["mean"]
    H["H7"] = {"expert": (ea, ep), "program_residual": (pa, pp), "verdict": "SOUTENUE" if pa >= ea - 0.01 and pp <= 0.5 * ep else "RÉFUTÉE" if pa < ea - 0.05 else "NON CONCLUANTE"}
    un = M["bpi_full"]["unrepresentable"]["adequate"]["mean"]; rp = M["bpi_full"]["held_out"]["adequate"]["mean"]
    abst, frej = 1 - un, 1 - rp
    H["H8"] = {"abstention": round(abst, 3), "false_reject": round(frej, 3), "verdict": "SOUTENUE" if abst >= 0.8 and frej <= 0.2 else "RÉFUTÉE" if abst < 0.5 else "NON CONCLUANTE"}
    from core.boundary_program import PRIMITIVES
    needed = [k[len("without_"):] for k, v in S.get("ablation", {}).items() if k.startswith("without_") and v and v["mean"] < S["ablation"]["all"]["mean"] - 0.02]
    covered = 4 + 3 + 3                                                          # familles représentables du banc dév. (train + val + held-out récupérables)
    H["H9"] = {"primitive_types": len(PRIMITIVES), "necessary_primitives": needed, "families_covered_estimate": covered, "coverage_all": S.get("ablation", {}).get("all"),
               "verdict": "SOUTENUE" if len(PRIMITIVES) <= 0.5 * covered or len(needed) <= 0.5 * covered else "RÉFUTÉE" if len(PRIMITIVES) >= covered else "NON CONCLUANTE"}
    return H


# ================================================================ lot scellé

def sealed_eval(seed=990777):
    spec = sealed_spec(); g = torch.Generator().manual_seed(seed); rows = []
    from core.boundary_experiments import induce_family_library
    lib, _ = induce_family_library(seed); macros = [f.macro.skeleton for f in lib if f.status in ("STABLE", "SUPPORTED")]
    for inst in instances(spec["families"], spec["seed"], spec["per_family"]):
        X, y = sample_task(inst, 200, g, label_noise=0.08)                       # étiquettes plus bruitées que le dév.
        R = evaluate_all(inst, X, y, g, macros)
        for m, r in R.items():
            if r is not None: rows.append({"family": inst.family, "representable": inst.representable, "method": m, **{k: v for k, v in r.items()}})
    from core.boundary_experiments import truth_anchor_experiment
    return {"rows": rows, "anchor": truth_anchor_experiment(seed)}

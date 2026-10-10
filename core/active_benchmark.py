"""Multi-seed runs, aggregation, PREREGISTERED hypotheses (docs/hypotheses_active_structure.md), sealed set."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import torch

import core.active_experiments as E

JOBS = ("main", "qp", "family", "anchor", "stability", "residual", "capacity", "ident", "cross", "noise")
FN = {"main": E.main_job, "qp": E.query_policy_job, "family": E.family_job, "anchor": E.anchor_job, "stability": E.stability_job,
      "residual": E.residual_job, "capacity": E.capacity_job, "ident": E.identifiability_job, "cross": E.cross_domain_job, "noise": E.noise_sweep_job}


def run_job(args):
    seed, job = args; torch.set_num_threads(1); return seed, job, FN[job](seed)


def _ci(xs):
    xs = [float(x) for x in xs if x is not None]; n = len(xs)
    if not n: return None
    m = sum(xs) / n
    if n < 2: return {"mean": round(m, 4), "ci95": [round(m, 4)] * 2, "n": n}
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)); h = (2.131 if n <= 16 else 2.04) * sd / math.sqrt(n)
    return {"mean": round(m, 4), "ci95": [round(m - h, 4), round(m + h, 4)], "n": n}


def _ece(p, y, bins=5):
    e = 0.0
    for b in range(bins):
        idx = [i for i, v in enumerate(p) if b / bins <= v < (b + 1) / bins or (b == bins - 1 and v == 1.0)]
        if idx: e += len(idx) / len(p) * abs(sum(p[i] for i in idx) / len(idx) - sum(y[i] for i in idx) / len(idx))
    return e


def aggregate(results):
    by = defaultdict(dict)
    for s, j, r in results: by[j][s] = r
    seeds = sorted(by["main"]); S = {}
    cats = ("held_out", "depth34", "hard_negative", "multi_syntax", "unsupported", "double_held_out")
    def cell(s, v, c, k):
        x = [float(r[k]) for r in by["main"][s]["rows"] if r["variant"] == v and r["category"] in c and r.get(k) is not None]; return sum(x) / len(x) if x else None
    S["main"] = {v: {c: {k: _ci([cell(s, v, (c,), k) for s in seeds]) for k in ("recovery", "extrapolation", "in_region", "evaluations", "queries", "seconds")} for c in cats} for v in E.VARIANTS}
    S["main_status_unsupported"] = {v: dict(Counter(r["status"] for s in seeds for r in by["main"][s]["rows"] if r["variant"] == v and r["category"] == "unsupported")) for v in E.VARIANTS}
    S["main_status_representable"] = {v: dict(Counter(r["status"] for s in seeds for r in by["main"][s]["rows"] if r["variant"] == v and r["category"] in ("held_out", "depth34"))) for v in E.VARIANTS}
    diff = lambda a, b, c, k: _ci([cell(s, a, c, k) - cell(s, b, c, k) for s in seeds])
    HD = ("held_out", "depth34")
    S["diffs"] = {"interleaved_vs_after_recovery": diff("full", "after_search", HD, "recovery"), "full_vs_random_queries_extrap": diff("full", "full_random_queries", HD + ("hard_negative",), "extrapolation"),
                  "full_vs_passive_extrap": diff("full", "passive", HD, "extrapolation"), "proposer_evals_ratio": _ci([cell(s, "full", ("depth34",), "evaluations") / max(1, cell(s, "interleaved_functional", ("depth34",), "evaluations")) for s in seeds]),
                  "depth_recovery_guided": _ci([cell(s, "full", ("depth34",), "recovery") for s in seeds]), "depth_recovery_unguided": _ci([cell(s, "interleaved_functional", ("depth34",), "recovery") for s in seeds])}
    # H2: reduction of the structural entropy after 8 queries
    qp = [r for s in sorted(by["qp"]) for r in by["qp"][s]]
    def red(r, k=8):
        e = r["entropy"]; i = min(k, len(e) - 1); return (e[0] - e[i]) / max(1e-9, e[0]) if e[0] > 1e-9 else 0.0
    S["query_policies"] = {p: {"entropy_reduction_at_8": _ci([red(r) for r in qp if r["policy"] == p]), "recovery": _ci([r["recovery"] for r in qp if r["policy"] == p]),
                               "extrapolation": _ci([r["extrapolation"] for r in qp if r["policy"] == p]), "version_space_end": _ci([r["version_space"][-1] for r in qp if r["policy"] == p])}
                           for p in ("structure_ig", "parameter_ig", "predictive_entropy", "disagreement", "random")}
    per_seed_red = {s: {p: sum(red(r) for r in by["qp"][s] if r["policy"] == p) / max(1, sum(1 for r in by["qp"][s] if r["policy"] == p)) for p in ("structure_ig", "random")} for s in by["qp"]}
    S["h2_seeds_better"] = sum(v["structure_ig"] > v["random"] for v in per_seed_red.values())
    S["query_efficiency_curve"] = {p: [round(sum(r["entropy"][min(i, len(r["entropy"]) - 1)] for r in qp if r["policy"] == p) / max(1, sum(1 for r in qp if r["policy"] == p)), 3) for i in range(17)] for p in ("structure_ig", "random")}
    fam = [by["family"][s] for s in sorted(by["family"])]
    S["family"] = {m: {k: _ci([f[m][k] for f in fam]) for k in ("recall", "false_promotions", "promoted", "representations_per_family", "compression")} for m in ("ast", "functional_raw", "functional_canonical")}
    S["transformed_reuse"] = {m: {k: _ci([f["transformed_reuse"][m][k] for f in fam]) for k in ("recovery", "evaluations")} for m in ("canonical", "raw")}
    an = [r for s in sorted(by["anchor"]) for r in by["anchor"][s]]; W = [r for r in an if r["wrong"]]; C = [r for r in an if not r["wrong"]]
    S["anchor"] = {"n_wrong": len(W), "n_correct": len(C), "new_false_rejection": sum(r["decision"] == "reject" for r in C) / max(1, len(C)),
                   "new_detection": sum(r["decision"] == "reject" for r in W) / max(1, len(W)), "new_undecided": sum(r["decision"] == "undecided" for r in an) / max(1, len(an)),
                   "old_false_rejection": sum(r["old_refuted"] for r in C) / max(1, len(C)), "old_detection": sum(r["old_refuted"] for r in W) / max(1, len(W)),
                   "ece": round(_ece([r["p_wrong"] for r in an], [float(r["wrong"]) for r in an]), 4),
                   "brier": round(sum((r["p_wrong"] - r["wrong"]) ** 2 for r in an) / max(1, len(an)), 4),
                   "mean_queries": round(sum(r["queries"] for r in an) / max(1, len(an)), 2), "repeat_rate": round(sum(r["repeats"] for r in an) / max(1, sum(r["queries"] for r in an)), 3),
                   "by_kind": {k: round(sum(r["decision"] == "reject" for r in an if r["kind"] == k) / max(1, sum(1 for r in an if r["kind"] == k)), 3) for k in {r["kind"] for r in an}}}
    st = [r for s in sorted(by["stability"]) for r in by["stability"][s]]
    S["stability"] = {str(nz): {"new_accept_correct": _rate([r["new_accept"] for r in st if r["noise"] == nz and r["correct"]]), "old_accept_correct": _rate([r["old_accept"] for r in st if r["noise"] == nz and r["correct"]]),
                                "new_accept_unrepresentable": _rate([r["new_accept"] for r in st if r["noise"] == nz and not r["representable"]]), "old_accept_unrepresentable": _rate([r["old_accept"] for r in st if r["noise"] == nz and not r["representable"]]),
                                "modes_correct": {k: _rate([r[k] for r in st if r["noise"] == nz and r["correct"]]) for k in ("prediction_stability", "functional_class_stability", "parameter_stability", "ast_stability")},
                                "n_correct": sum(1 for r in st if r["noise"] == nz and r["correct"])} for nz in (0.08, 0.12)}
    rs = [r for s in sorted(by["residual"]) for r in by["residual"][s]]
    S["residual"] = {k: {"in": _ci([r[k][0] for r in rs]), "out": _ci([r[k][1] for r in rs]), "params": _ci([r[k][2] for r in rs])} for k in ("program", "global", "gated")}
    S["residual"]["gate_out"] = _ci([r["gate_fraction_out"] for r in rs])
    cp = [by["capacity"][s] for s in sorted(by["capacity"])]
    S["capacity"] = {m: {"retained_ok": _ci([c[m]["retained_ok"] for c in cp]), "rejected": _ci([c[m]["rejected"] for c in cp]), "used": _ci([c[m]["used"] for c in cp]),
                         "violations": sum(c[m]["violated"] for c in cp), "mean_acc": _ci([c[m]["mean_acc"] for c in cp]),
                         "operations": dict(sum((Counter(c[m]["operations"]) for c in cp), Counter()))} for m in ("experts_only", "full")}
    idn = [r for s in sorted(by["ident"]) for r in by["ident"][s]]
    S["identifiability"] = {"non_identifiable_from_observed_region": _rate([not r["identifiable_observed_only"] for r in idn]), "identifiable_with_full_region": _rate([r["identifiable_full"] for r in idn]),
                            "minimal_set": _ci([r["minimal_set"] for r in idn]), "plausible_classes": _ci([r["plausible_classes"] for r in idn])}
    cd = [by["cross"][s] for s in sorted(by["cross"])]
    S["cross_domain"] = {d: {p: {k: _ci([c[d][p][k] for c in cd]) for k in ("queries", "identified")} for p in ("active", "random")} for d in cd[0]}
    nz = [r for s in sorted(by["noise"]) for r in by["noise"][s]]
    S["noise_sweep"] = {str(n): {"passive_recovery": _rate([r["recovery"] for r in nz if r["noise"] == n and r["variant"] == "passive"]),
                                 "full_recovery": _rate([r["recovery"] for r in nz if r["noise"] == n and r["variant"] == "full"]),
                                 "full_extrap": _rate([r["extrapolation"] for r in nz if r["noise"] == n and r["variant"] == "full"]),
                                 "passive_extrap": _rate([r["extrapolation"] for r in nz if r["noise"] == n and r["variant"] == "passive"]),
                                 "full_queries": _rate([r["queries"] for r in nz if r["noise"] == n and r["variant"] == "full"]),
                                 "anchor_false_rejection": _rate([r["decision"] == "reject" for r in nz if r["noise"] == n and r["variant"] == "anchor" and not r["wrong"]]),
                                 "anchor_detection": _rate([r["decision"] == "reject" for r in nz if r["noise"] == n and r["variant"] == "anchor" and r["wrong"]])} for n in E.NOISE_LEVELS}
    S["search_macros"] = Counter(str(m) for s in seeds for m, c in by["main"][s].get("search_macros", []) for _ in range(c)).most_common(5)
    S["hypotheses"] = hypotheses(S, seeds)
    return S


def _rate(xs):
    xs = [float(x) for x in xs]; return round(sum(xs) / len(xs), 4) if xs else None


def hypotheses(S, seeds):
    H = {}; d = S["diffs"]
    v = d["interleaved_vs_after_recovery"]; H["H1"] = {"value": v, "verdict": "SOUTENUE" if v["mean"] >= 0.10 and v["ci95"][0] > 0 else "RÉFUTÉE" if v["mean"] <= 0 else "NON CONCLUANTE"}
    a, r = S["query_policies"]["structure_ig"]["entropy_reduction_at_8"]["mean"], S["query_policies"]["random"]["entropy_reduction_at_8"]["mean"]
    H["H2"] = {"structure_ig": a, "random": r, "seeds_better": S["h2_seeds_better"], "verdict": "SOUTENUE" if a >= r + 0.20 and S["h2_seeds_better"] >= 12 else "RÉFUTÉE" if a <= r else "NON CONCLUANTE"}
    f = S["family"]; ra, rf = f["ast"]["recall"]["mean"], f["functional_canonical"]["recall"]["mean"]; fa, ff = f["ast"]["false_promotions"]["mean"], f["functional_canonical"]["false_promotions"]["mean"]
    H["H3"] = {"ast": (ra, fa), "functional": (rf, ff), "verdict": "SOUTENUE" if rf >= ra + 0.2 and ff <= fa + 0.5 else "RÉFUTÉE" if rf <= ra else "NON CONCLUANTE"}
    an = S["anchor"]; H["H4"] = {"false_rejection": an["new_false_rejection"], "detection": an["new_detection"], "old": (an["old_false_rejection"], an["old_detection"]),
                                 "verdict": "SOUTENUE" if an["new_false_rejection"] <= 0.10 and an["new_detection"] >= 0.6 else "RÉFUTÉE" if an["new_false_rejection"] > 0.25 or an["new_detection"] < 0.4 else "NON CONCLUANTE"}
    st = S["stability"]; ok = all((st[k]["new_accept_correct"] or 0) >= 0.7 for k in st) and all((st[k]["new_accept_unrepresentable"] or 0) <= 0.35 for k in st)
    H["H5"] = {"value": {k: (st[k]["new_accept_correct"], st[k]["new_accept_unrepresentable"]) for k in st}, "verdict": "SOUTENUE" if ok else "RÉFUTÉE" if any((st[k]["new_accept_correct"] or 0) < 0.4 for k in st) else "NON CONCLUANTE"}
    ratio = d["proposer_evals_ratio"]["mean"]; rg, ru = d["depth_recovery_guided"]["mean"], d["depth_recovery_unguided"]["mean"]
    H["H6"] = {"evals_ratio": ratio, "recovery_guided": rg, "recovery_unguided": ru, "verdict": "SOUTENUE" if ratio <= 0.7 and rg >= ru - 0.05 else "RÉFUTÉE" if ratio >= 1.0 else "NON CONCLUANTE"}
    tc, tr = S["transformed_reuse"]["canonical"]["recovery"]["mean"], S["transformed_reuse"]["raw"]["recovery"]["mean"]
    H["H7"] = {"canonical": tc, "raw": tr, "verdict": "SOUTENUE" if tc >= tr + 0.1 else "RÉFUTÉE" if tc <= tr else "NON CONCLUANTE"}
    v = d["full_vs_random_queries_extrap"]; H["H8"] = {"value": v, "verdict": "SOUTENUE" if v["mean"] >= 0.03 and v["ci95"][0] > 0 else "RÉFUTÉE" if v["mean"] <= 0 else "NON CONCLUANTE"}
    R = S["residual"]; dg = R["program"]["out"]["mean"] - R["gated"]["out"]["mean"]; dl = R["program"]["out"]["mean"] - R["global"]["out"]["mean"]
    H["H9"] = {"ood_degradation_gated": round(dg, 4), "ood_degradation_global": round(dl, 4), "in_gated": R["gated"]["in"]["mean"], "in_global": R["global"]["in"]["mean"],
               "verdict": "SOUTENUE" if (dg <= 0.5 * max(dl, 1e-9) and R["gated"]["in"]["mean"] >= R["global"]["in"]["mean"] - 0.01) else "RÉFUTÉE" if dg >= dl else "NON CONCLUANTE"}
    c = S["capacity"]; fe, ex = c["full"]["retained_ok"]["mean"], c["experts_only"]["retained_ok"]["mean"]
    H["H10"] = {"full": fe, "experts_only": ex, "violations": c["full"]["violations"], "verdict": "SOUTENUE" if fe >= 1.3 * ex and c["full"]["violations"] == 0 else "RÉFUTÉE" if fe <= ex else "NON CONCLUANTE"}
    return H


# ================================================================ sealed set

def sealed_job(args):
    fam_idx, noise = args; torch.set_num_threads(1)
    from core.active_world import balanced_instances, noisy_task, sealed_spec, train_tasks
    from core.active_structure import train_policy, sequential_truth_test
    from core.boundary_program import predict
    from core.boundary_world import full_region
    spec = sealed_spec(); insts = balanced_instances(spec["families"], spec["seed"], spec["per_family"]); inst = insts[fam_idx]
    policy = train_policy(train_tasks(991000, noise), eps=noise, seed=991); rows = []
    g = torch.Generator().manual_seed(991000 + fam_idx); X, y = noisy_task(inst, E.N, g, noise)
    for v in ("passive", "after_search", "full"):
        prog, info = E.run_variant(v, inst, X, y, noise, policy, 991000 + fam_idx)
        m = E.metrics(lambda Q, p=prog: predict(p, Q), inst, torch.Generator().manual_seed(4)) if prog is not None else {"recovery": False, "extrapolation": 0, "in_region": 0, "full": 0}
        rows.append({"family": inst.family, "representable": inst.representable, "noise": noise, "variant": v, **m, **{k: info[k] for k in ("queries", "evaluations", "status")}})
    for kind, rule, wrong, alts in E.candidate_rules(inst, X, y, noise, 991):
        r = sequential_truth_test(rule, alts, inst.label, full_region, torch.Generator().manual_seed(fam_idx), noise, budget=E.BUDGET)
        rows.append({"family": inst.family, "noise": noise, "variant": "anchor", "kind": kind, "wrong": wrong, "decision": r["decision"], "p_wrong": r["p_wrong"]})
    return rows

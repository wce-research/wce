"""Banc du jalon « certificats génératifs » : flux continus de 100 régimes, 9 méthodes, mise à l'échelle
(10/25/50/100), expériences ciblées, hypothèses pré-enregistrées (docs/hypotheses_generative_certificates.md),
multi-graines et lot scellé (évalué une fois)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import torch

from core.continuous_world import stream
from core.continuum import METHODS, run_stream

MAIN_KEYS = ("average_accuracy", "new_task", "retention", "forgetting", "backward_transfer", "calibration_ece", "total_memory_bytes", "reuse_ratio")


def seed_run(seed, n=100, methods=tuple(METHODS), capacity_full=1500):
    """Une graine : flux de n régimes pour toutes les méthodes + toutes les expériences ciblées."""
    from core.continuum_experiments import (adversarial_experiment, capacity_both, cross_domain_certificates, fit_growth, latent_factors, lifetime_cost,
                                            novelty_classification, regeneration_experiment, reuse_law)
    from core.discovery import discover_invariants, law_discovery_experiment, rot, transformation_discovery_experiment
    from core.generative_certificate import coverage_experiment
    from core.second_order_experiment import run as so_run
    regs = stream(2000 + seed, n); out = {"seed": seed, "methods": {}}
    for m in methods:
        o, s = run_stream(m, regs, seed, capacity=capacity_full if m == "full" else None)
        cps = {k: {kk: v[kk] for kk in MAIN_KEYS} | {"neural_params": v["memory"]["neural_params"], "compute": v["compute"]} for k, v in o["checkpoints"].items()}
        o2 = {k: o[k] for k in MAIN_KEYS} | {"memory": o["memory"], "compute": o["compute"], "kinds": o["kinds"], "counters": o["counters"], "checkpoints": cps,
                                             "wall_seconds": o["wall_seconds"]}
        if m not in ("vanilla", "raw_replay", "generative_replay", "first_order_constraints"):
            o2["lifetime"] = lifetime_cost(o["learning_curve"], o["kinds_curve"]); o2["reuse_law"] = reuse_law(o["kinds_curve"])
            ks = sorted(cps); o2["growth_fit"] = fit_growth(ks, [max(1, cps[k]["neural_params"]) + 1 for k in ks]) if len(ks) >= 2 else {"power_exponent": 0.0, "exponent_se": 0.0, "power_r2": 0.0, "linear_r2": 0.0}
            o2["refine_count"] = sum(k in ("refine", "refine_rule", "refine_adapter") for k in o["kinds_curve"])
            o2["forward"] = forward_transfer(o["kinds_curve"], o["learning_curve"])
        out["methods"][m] = o2
    rows, adapt = so_run(seed); out["second_order"] = {"rows": rows, "adapt": adapt}
    regs_cov = [r for r in stream(3000 + seed, 60) if r.kind in ("new", "new_domain", "composition")]
    out["coverage"] = coverage_experiment(regs_cov, seed)
    r = regeneration_experiment(seed); r.pop("per_module"); out["regeneration"] = r
    out["capacity"] = capacity_both(seed)
    out["law"] = law_discovery_experiment(seed, n_worlds=8)
    out["transformation_discovery"] = transformation_discovery_experiment(seed, 12)
    h = discover_invariants(lambda P, g: P @ rot(float(torch.rand(1, generator=g)) * 6).T + torch.randn(2, generator=g), seed=seed)
    out["invariant_discovery"] = {k: v.status for k, v in h.items()}
    out["novelty"] = novelty_classification(seed); out["latent"] = latent_factors(seed); out["adversarial"] = adversarial_experiment(seed)
    out["cross_domain"] = cross_domain_certificates(seed)
    return out


def forward_transfer(kinds, steps):
    """Coût d'acquisition des nouvelles compétences : moitié précoce vs tardive (pas d'entraînement, et part sans entraînement)."""
    new = [(i, s, k) for i, (s, k) in enumerate(zip(steps, kinds)) if k not in ("reactivate", "noise_ignored")]
    half = len(steps) // 2
    e = [s for i, s, k in new if i < half]; l = [s for i, s, k in new if i >= half]
    z = lambda xs: round(sum(x == 0 for x in xs) / max(1, len(xs)), 3)
    return {"early_steps": round(sum(e) / max(1, len(e)), 1), "late_steps": round(sum(l) / max(1, len(l)), 1), "early_zero_train": z(e), "late_zero_train": z(l)}


# ================================================================ lot scellé

def sealed_streams(): return [stream(990500 + i, 100, p_novel=0.1, p_drift=0.3) for i in range(2)]


def sealed_hash():
    return hashlib.sha256(json.dumps([[(r.name, r.kind, r.noise, r.context_id, repr(r.rule), [round(x, 6) for x in r.mu.tolist()], round(r.scale, 6))
                                      for r in s] for s in sealed_streams()], sort_keys=True).encode()).hexdigest()


SEALED_SHA256 = "0260f9c4802d331b92daaa393cdb2fbf5c75f5d10433230c806e44c6426ec385"   # figée AVANT toute évaluation


def sealed_eval():
    from core.continuum_experiments import capacity_both
    from core.discovery import law_discovery_experiment
    out = {"streams": []}
    for i, regs in enumerate(sealed_streams()):
        res = {}
        for m in METHODS:
            o, s = run_stream(m, regs, 990 + i, capacity=1500 if m == "full" else None)
            res[m] = {k: o[k] for k in MAIN_KEYS} | {"neural_params": o["memory"]["neural_params"], "kinds": o["kinds"]}
        out["streams"].append(res)
    out["law"] = law_discovery_experiment(990, n_worlds=12); out["capacity"] = capacity_both(990)
    return out


# ================================================================ agrégation et hypothèses

def _ci(xs):
    n = len(xs); m = sum(xs) / n
    if n < 2: return {"mean": round(m, 4), "ci95": [round(m, 4), round(m, 4)], "n": n}
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)); h = 2.131 * sd / math.sqrt(n) if n <= 16 else 1.96 * sd / math.sqrt(n)
    return {"mean": round(m, 4), "ci95": [round(m - h, 4), round(m + h, 4)], "n": n}


def aggregate(per):
    S = {}
    ms = list(per[0]["methods"])
    S["methods"] = {m: {k: _ci([p["methods"][m][k] for p in per]) for k in MAIN_KEYS} | {"neural_params": _ci([p["methods"][m]["memory"]["neural_params"] for p in per])} for m in ms}
    S["scaling"] = {m: {str(c): {"accuracy": _ci([p["methods"][m]["checkpoints"][str(c) if str(c) in p["methods"][m]["checkpoints"] else c]["average_accuracy"] for p in per]),
                                 "params": _ci([p["methods"][m]["checkpoints"][str(c) if str(c) in p["methods"][m]["checkpoints"] else c]["neural_params"] for p in per]),
                                 "forgetting": _ci([p["methods"][m]["checkpoints"][str(c) if str(c) in p["methods"][m]["checkpoints"] else c]["forgetting"] for p in per]),
                                 "memory": _ci([p["methods"][m]["checkpoints"][str(c) if str(c) in p["methods"][m]["checkpoints"] else c]["total_memory_bytes"] for p in per]),
                                 "reuse": _ci([p["methods"][m]["checkpoints"][str(c) if str(c) in p["methods"][m]["checkpoints"] else c]["reuse_ratio"] for p in per])}
                        for c in sorted(int(c) for c in per[0]["methods"][ms[0]]["checkpoints"])} for m in ms}
    sysm = [m for m in ms if "lifetime" in per[0]["methods"][m]]
    S["lifetime"] = {m: {"slope": _ci([p["methods"][m]["lifetime"]["slope"] for p in per if p["methods"][m]["lifetime"]["slope"] is not None]),
                         "growth_exponent": _ci([p["methods"][m]["growth_fit"]["power_exponent"] for p in per]),
                         "linear_r2": _ci([p["methods"][m]["growth_fit"]["linear_r2"] for p in per]), "power_r2": _ci([p["methods"][m]["growth_fit"]["power_r2"] for p in per]),
                         "refine_count": _ci([p["methods"][m]["refine_count"] for p in per]),
                         "reuse_law": [_ci([p["methods"][m]["reuse_law"][b] for p in per])["mean"] for b in range(4)],
                         "forward": {k: _ci([p["methods"][m]["forward"][k] for p in per]) for k in ("early_steps", "late_steps", "early_zero_train", "late_zero_train")}} for m in sysm}
    from core.second_order_experiment import summarize
    S["second_order"] = summarize([(p["second_order"]["rows"], p["second_order"]["adapt"]) for p in per])
    S["coverage"] = {k: {kk: _ci([p["coverage"][k][kk] for p in per]) for kk in ("boundary_shift", "rare_region", "local_bump", "global", "bytes")} for k in per[0]["coverage"]}
    S["regeneration"] = {k: _ci([p["regeneration"][k] for p in per if p["regeneration"][k] is not None]) for k in ("median_recovery", "mean_recovery", "detected", "certificates_pass", "before", "corrupted", "regenerated")}
    S["law"] = {pol: {k: _ci([p["law"][pol][k] for p in per]) for k in per[0]["law"][pol]} for pol in per[0]["law"]}
    S["transformation_discovery"] = {k: _ci([p["transformation_discovery"][k] for p in per]) for k in ("discovered_exact", "rule_error", "memo_error", "candidates")}
    S["invariant_discovery"] = {k: sum(p["invariant_discovery"][k] in ("STABLE", "SUPPORTED") for p in per) / len(per) for k in per[0]["invariant_discovery"]}
    S["novelty"] = {"accuracy": _ci([p["novelty"]["accuracy"] for p in per]), "per_class": {c: _ci([p["novelty"]["per_class_recall"].get(c, 0) for p in per]) for c in per[0]["novelty"]["per_class_recall"]}}
    S["latent"] = _ci([p["latent"]["factor_recall"] for p in per]); S["adversarial"] = {k: _ci([p["adversarial"][k] for p in per]) for k in ("central", "boundary", "adversarial")}
    S["cross_domain"] = {d: {"detects": sum(p["cross_domain"][d]["detects_alteration"] for p in per) / len(per), "intact": _ci([p["cross_domain"][d]["intact_agreement"] for p in per])} for d in per[0]["cross_domain"]}
    capn = lambda key, part: [p["capacity"][key][part] for p in per]
    S["capacity"] = {key: {"accuracy_loss": _ci([p["capacity"][key]["accuracy_loss"] for p in per]),
                           "operations": {op: sum(p["capacity"][key]["pressure"]["operations"].get(op, 0) for p in per) for op in ("prune", "prune_skill", "merge", "distill", "distill_symbolic")},
                           "distill_rejected": sum(p["capacity"][key]["pressure"]["distill_rejected"] for p in per),
                           "budget_exceeded_runs": sum(p["capacity"][key]["pressure"]["budget_exceeded_at_end"] for p in per)} for key in ("neural_only", "mixed")}
    S["hypotheses"] = hypotheses(per, S)
    return S


def hypotheses(per, S):
    """Évaluation STRICTE selon les critères pré-enregistrés (non modifiés)."""
    M = lambda m, k, c=25: [p["methods"][m]["checkpoints"][str(c) if str(c) in p["methods"][m]["checkpoints"] else c][k] for p in per]
    H = {}
    gc, rr = M("generative_certificates", "retention"), M("raw_replay", "retention")
    d = sum(gc) / len(gc) - sum(rr) / len(rr); raw = max(p["methods"]["generative_certificates"]["memory"]["raw_data_bytes"] for p in per)
    H["H1"] = {"metric": "rétention(25) gen_cert − raw_replay", "value": round(d, 4), "raw_bytes": raw,
               "verdict": "SUPPORTÉE" if d >= -0.02 and raw == 0 else "RÉFUTÉE" if d < -0.05 else "NON CONCLUANTE"}
    so = S["second_order"]; rd = so["recall"]["second_order_flags"] - so["recall"]["first_order_flags"]
    H["H2"] = {"metric": "rappel 2nd − 1er ; graines rétention 2nd > 1er", "value": round(rd, 3), "seeds": so["second_beats_first_retention_seeds"], "n": so["n_seeds"],
               "verdict": "SUPPORTÉE" if rd >= 0.2 and so["second_beats_first_retention_seeds"] >= 12 else "RÉFUTÉE" if rd < 0.05 else "NON CONCLUANTE"}
    full, rp, sg = [S["methods"][m]["average_accuracy"]["mean"] for m in ("full", "raw_replay", "structural_growth")]
    H["H3"] = {"metric": "exactitude full vs max(raw replay, croissance)", "value": round(full - max(rp, sg), 4),
               "verdict": "SUPPORTÉE" if full >= max(rp, sg) - 0.03 else "RÉFUTÉE" if full < rp - 0.05 else "NON CONCLUANTE"}
    last = str(max(int(c) for c in S["scaling"]["full"]))
    pf, ps = S["scaling"]["full"][last]["params"]["mean"], S["scaling"]["structural_growth"][last]["params"]["mean"]; ratio = pf / max(1, ps)
    H["H4"] = {"metric": "paramètres(100) full / systématique", "value": round(ratio, 3), "verdict": "SUPPORTÉE" if ratio <= 0.6 else "RÉFUTÉE" if ratio >= 0.9 else "NON CONCLUANTE"}
    sl = S["lifetime"]["full"]["slope"]
    H["H5"] = {"metric": "pente du coût d'acquisition (full)", "value": sl, "verdict": "SUPPORTÉE" if sl["ci95"][1] < 0 else "RÉFUTÉE"}
    capm = S["capacity"]["mixed"]; capn = S["capacity"]["neural_only"]
    ops_ok = all(sum(S["capacity"][k]["operations"].get(op, 0) for k in ("mixed", "neural_only")) >= 1 for op in ("prune", "merge", "distill"))
    loss = max(capm["accuracy_loss"]["mean"], capn["accuracy_loss"]["mean"])
    H["H6"] = {"metric": "opérations déclenchées (élagage, fusion, distillation) et perte", "operations": {k: S["capacity"][k]["operations"] for k in ("mixed", "neural_only")}, "loss": loss,
               "verdict": "SUPPORTÉE" if ops_ok and loss <= 0.03 else "RÉFUTÉE" if (loss > 0.10 or not any(sum(S["capacity"][k]["operations"].values()) for k in ("mixed", "neural_only"))) else "NON CONCLUANTE (critère « chaque type » non atteint)"}
    rec = S["regeneration"]["median_recovery"]["mean"]
    H["H7"] = {"metric": "récupération médiane", "value": rec, "verdict": "SUPPORTÉE" if rec >= 0.95 else "RÉFUTÉE" if rec < 0.90 else "NON CONCLUANTE"}
    ida = S["law"]["active"]["identification"]["mean"]
    wins = sum(p["law"]["active"]["experiments"] < p["law"]["random"]["experiments"] for p in per)
    H["H8"] = {"metric": "identification active ; graines exp(active) < exp(aléatoire)", "value": ida, "seeds": wins, "n": len(per),
               "verdict": "SUPPORTÉE" if ida >= 0.8 and wins >= 12 else "RÉFUTÉE" if ida < 0.5 else "NON CONCLUANTE"}
    return H

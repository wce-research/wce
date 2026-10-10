""""Robust Structural Reconstruction" experiments — one seed × one job type = one AWS compute unit."""
from __future__ import annotations

import math
import random
import time
from collections import Counter

import torch

from core.boundary_program import BoundaryProgramSearch, predict, render, structure_hash
from core.robust_reconstruction import (SENSOR_MODELS, CompatibleStructureSet, Observations, RobustMeasurementPlanner, RobustStructuralReconstruction,
                                        StructureEntry, attribute_discrepancy, beta_calibration, calibration_metrics, fit_robust, isotonic,
                                        log_evidence, mdl, obs_loglik, platt, posterior_predictive_check)
from core.robust_world import NOISE_LEVELS, SENSORS, full_region, hard_noise_pair, in_region, make_task, out_region

BUDGET = 16.0
METHODS = ("old_system", "robust_fit_only", "robust_posterior", "robust_active_structural_ig", "robust_active_efa", "full", "random_queries",
           "disagreement_queries", "passive", "oracle_policy")
COMPACT_PARAMS, COMPACT_POINTS = 64, 32


def obs_of(task):
    return Observations.build(task.X, task.y, SENSOR_MODELS["standard"], task.spec.rate)


def evaluate_prog(prog, task, g):
    if prog is None or prog == "UNKNOWN": return {"full": 0.0, "recovery": False, "extrapolation": 0.0, "in_region": 0.0, "structure": False}
    F, O, I = full_region(3000, g), out_region(1500, g), in_region(1500, g); lab = task.world.label
    a = float((predict(prog, F) == lab(F)).float().mean())
    return {"full": a, "recovery": a >= 0.97, "extrapolation": float((predict(prog, O) == lab(O)).float().mean()),
            "in_region": float((predict(prog, I) == lab(I)).float().mean()),
            "structure": task.world.truth is not None and structure_hash(prog) == structure_hash(task.world.truth)}


class _Oracle:
    """ORACLE measurement policy (upper bound): queries where the current MAP structure is wrong (truth used)."""
    def __init__(self, task): self.task = task; self.criterion = "oracle"; self.world_noise = task.spec.rate; self.sensors = ("standard",); self.lam = 0.0; self.allow_repeat = False

    def choose(self, css, pool, history, probes, budget_left):
        from core.robust_reconstruction import MeasurementChoice
        if budget_left < 1.0: return MeasurementChoice("stop", None, "", 0.0, 0.0)
        wrong = predict(css.best().program, pool) != self.task.world.label(pool); idx = wrong.nonzero().flatten()
        i = int(idx[0]) if len(idx) else 0; return MeasurementChoice("new_point", pool[i:i + 1], "standard", 1.0, 0.0)


def run_method(m, task, seed, monitor=None):
    g = torch.Generator().manual_seed(seed); probes = full_region(400, torch.Generator().manual_seed(seed + 77)); obs = obs_of(task); nz = task.spec.rate
    t0 = time.time()
    if m in ("old_system", "passive"):
        v = task.y >= 0; X, y = task.X[v], task.y[v]
        if m == "passive":
            s = BoundaryProgramSearch(max_depth=3, seed=seed).run(X[:90], y[:90], X[90:], y[90:]); return s.best, {"queries": 0, "evaluations": s.stats["evaluated"], "seconds": round(time.time() - t0, 2), "depth": s.best.size() if s.best else 0, "status": "point_estimate", "trace": []}
        from core.active_structure import ActiveStructuralDiscovery
        a = ActiveStructuralDiscovery(budget=int(BUDGET), eps=max(nz, 0.01), seed=seed); a.eps_world = nz
        r = a.run(X, y, lambda x: task.world.label(x), full_region, g)
        return r.program, {"queries": r.queries, "evaluations": r.evaluations, "seconds": r.seconds, "depth": r.program.size(), "status": r.status, "trace": []}
    cfg = {"robust_fit_only": dict(planner=None, depth_rule="threshold"), "robust_posterior": dict(planner=None),
           "robust_active_structural_ig": dict(planner=RobustMeasurementPlanner("structural_ig", world_noise=nz, allow_repeat=False, seed=seed)),
           "robust_active_efa": dict(planner=RobustMeasurementPlanner("efa", world_noise=nz, allow_repeat=False, seed=seed)),
           "full": dict(planner=RobustMeasurementPlanner("efa", sensors=tuple(SENSOR_MODELS), world_noise=nz, allow_repeat=True, seed=seed)),
           "random_queries": dict(planner=RobustMeasurementPlanner("random", world_noise=nz, allow_repeat=False, seed=seed)),
           "disagreement_queries": dict(planner=RobustMeasurementPlanner("disagreement", world_noise=nz, allow_repeat=False, seed=seed)),
           "oracle_policy": dict(planner=_Oracle(task))}[m]
    rr = RobustStructuralReconstruction(budget=BUDGET, seed=seed, **cfg)
    oracle = (lambda x, sensor, gg: task.oracle(x, sensor, gg))
    r = rr.run(obs, oracle, full_region, probes, g, monitor=monitor)
    prog = r.css.best().program
    return prog, {"queries": r.queries, "cost": r.cost, "evaluations": r.evaluations, "seconds": r.seconds, "depth": prog.size(), "status": r.status,
                  "answer_unknown": r.answer == "UNKNOWN", "discrepancy": r.discrepancy.get("label"), "trace": r.trace, "entropy": r.css.entropy()}


# ================================================================ reproducible sub-units (checkpoints)
# Each job = ORDERED list of sub-units (identifier, function → rows). A sub-unit depends only on
# (type, seed, identifier): no shared random state, so resuming after an interruption = same results.

MAIN_SPECS = [(nz, d) for nz in (0.08, 0.12) for d in (1, 2, 3, 4, 0)]          # d = 0: not representable


def main_task(seed, k):
    nz, d = MAIN_SPECS[k]
    if d: return "main", make_task(seed * 100 + d + int(nz * 1000), noise=nz, depth=d)
    return "unrepresentable", make_task(seed * 100 + 9 + int(nz * 1000), noise=nz, representable=False)


def main_tasks(seed): return [main_task(seed, k) for k in range(len(MAIN_SPECS))]


def _main_sub(seed, k, m):
    cat, task = main_task(seed, k)
    mon = (lambda css, task=task: float((predict(css.best().program, full_region(800, torch.Generator().manual_seed(3))) == task.world.label(full_region(800, torch.Generator().manual_seed(3)))).float().mean())) if m == "full" else None
    prog, info = run_method(m, task, seed * 31 + k, mon)
    ev = evaluate_prog(prog, task, torch.Generator().manual_seed(11))
    tr = info.pop("trace", []); pairs = []
    if m == "full":
        prev = None
        for t in tr:
            if "true_acc_after" in t and prev is not None and "predicted_gain" in t: pairs.append((t["predicted_gain"], t["true_acc_after"] - prev))
            if "true_acc_after" in t: prev = t["true_acc_after"]
    return [{"category": cat, "noise": task.spec.rate, "depth": task.world.depth, "representable": task.world.representable, "method": m, **ev, **info,
             "efa_pairs": pairs, "program": render(prog) if prog not in (None, "UNKNOWN") else None, "true_depth": task.world.truth.size() if task.world.truth else None}]


def main_subunits(seed): return [(f"t{k}_{m}", (lambda k=k, m=m: _main_sub(seed, k, m))) for k in range(len(MAIN_SPECS)) for m in METHODS]


NOISE_SWEEP_KINDS = ("iid", "grouped_outliers", "boundary")


def _noise_sub(seed, nz, kind, m):
    # deterministic task seed (fixed: `hash(kind)` depended on PYTHONHASHSEED, hence on the process)
    task = make_task(seed * 1000 + int(nz * 1000) + NOISE_SWEEP_KINDS.index(kind), noise=nz, kind=kind, depth=1 + (seed + int(nz * 100)) % 3)
    prog, info = run_method(m, task, seed * 7 + int(nz * 100)); info.pop("trace", None)
    return [{"noise": nz, "kind": kind, "method": m, **evaluate_prog(prog, task, torch.Generator().manual_seed(12)), **{k: info[k] for k in ("queries", "evaluations", "seconds")},
             "params": prog.n_params() if prog not in (None, "UNKNOWN") else 0, "ast": prog.size() if prog not in (None, "UNKNOWN") else 0}]


def noise_subunits(seed):
    return [(f"n{int(nz * 100)}_{kind}_{m}", (lambda nz=nz, kind=kind, m=m: _noise_sub(seed, nz, kind, m)))
            for nz in NOISE_LEVELS for kind in NOISE_SWEEP_KINDS for m in ("passive", "old_system", "robust_posterior", "full")]


def _hard_sub(seed, ci, name):
    """AA — A: simple + 8 % noise; B: complex + 0 %. Size of the retained AST (vs true) and attribution."""
    case = ("simple_noisy", "complex_clean")[ci]; task = hard_noise_pair(seed + 500)[ci]; obs = obs_of(task)
    kw = {"naive_threshold": dict(likelihood="naive", depth_rule="threshold"), "robust_bayes": dict()}[name]
    r = RobustStructuralReconstruction(seed=seed, **kw).run(obs); p = r.css.best().program
    return [{"case": case, "method": name, "ast": p.size(), "true_ast": task.world.truth.size(), "excess_depth": max(0, p.size() - task.world.truth.size()),
             "discrepancy": r.discrepancy["label"], **evaluate_prog(p, task, torch.Generator().manual_seed(13)), "mdl_naive": mdl(p, obs, "naive")["total"], "mdl_measurement": mdl(p, obs, "contamination")["total"]}]


def hard_subunits(seed): return [(f"c{ci}_{name}", (lambda ci=ci, name=name: _hard_sub(seed, ci, name))) for ci in (0, 1) for name in ("naive_threshold", "robust_bayes")]


def _repeat_sub(seed, nz, d, allow):
    task = make_task(seed * 77 + d + int(nz * 100), noise=nz, depth=d)
    pl = RobustMeasurementPlanner("efa", world_noise=nz, allow_repeat=allow, seed=seed)
    r = RobustStructuralReconstruction(planner=pl, budget=BUDGET, seed=seed).run(obs_of(task), lambda x, s, gg: task.oracle(x, s, gg), full_region, full_region(400, torch.Generator().manual_seed(seed)), torch.Generator().manual_seed(seed))
    reps = sum(1 for t in r.trace if t.get("action") == "repeat")
    return [{"noise": nz, "repeat": allow, "repeats_used": reps, **evaluate_prog(r.css.best().program, task, torch.Generator().manual_seed(14)), "queries": r.queries}]


def repeat_subunits(seed):
    return [(f"n{int(nz * 100)}_d{d}_{'rep' if allow else 'norep'}", (lambda nz=nz, d=d, allow=allow: _repeat_sub(seed, nz, d, allow))) for nz in (0.12, 0.15) for d in (1, 2) for allow in (True, False)]


# ================================================================ calibration of P(rule wrong)

def rule_cases(task_seed, noise):
    """Correct / wrong rules + raw P(rule wrong) (noise-sensitive anchor) — truth for the label."""
    from core.active_structure import sequential_truth_test
    task = make_task(task_seed, noise=noise, depth=1 + task_seed % 3); obs = obs_of(task); out = []
    r = RobustStructuralReconstruction(seed=task_seed).run(obs)
    cands = [e.program for e in sorted(r.css.entries, key=lambda e: -e.mass)[:4]]
    F = full_region(2000, torch.Generator().manual_seed(task_seed))
    for prog in cands:
        acc = float((predict(prog, F) == task.world.label(F)).float().mean())
        if 0.93 <= acc < 0.97: continue
        alts = [c for c in cands if c is not prog]
        res = sequential_truth_test(prog, alts, lambda x: task.world.label(x), full_region, torch.Generator().manual_seed(task_seed + 1), max(noise, 0.01), budget=16, noise=noise)
        out.append({"p_wrong": res["p_wrong"], "wrong": acc < 0.93})
    return out


TRAIN_CAL_SEEDS = range(700000, 700060)
CAL_NOISES = (0.05, 0.08, 0.12)


def caltrain_subunits(key):
    """Training cases for the calibrators (TRAIN + VALIDATION, seeds 700000+), computed ONCE (6 units × 10 seeds)
    instead of being recomputed identically in each of the 16 test seeds."""
    return [(f"s{s}", (lambda s=s: [{"split": "train", "task_seed": s, **c} for c in rule_cases(s, CAL_NOISES[s % 3])])) for s in TRAIN_CAL_SEEDS[key * 10:(key + 1) * 10]]


def calibration_subunits(seed):
    """TEST cases of the seed. The calibrators are fitted at aggregation time on the `caltrain` cases only."""
    return [(f"k{k}", (lambda k=k: [{"split": "test", "task_seed": 800000 + seed * 10 + k, **c} for c in rule_cases(800000 + seed * 10 + k, CAL_NOISES[k % 3])])) for k in range(6)]


def fit_calibrators(train, test):
    xs, ys = [c["p_wrong"] for c in train], [float(c["wrong"]) for c in train]; tp, ty = [c["p_wrong"] for c in test], [float(c["wrong"]) for c in test]
    out = {"n_train": len(train), "n_test": len(test), "raw": calibration_metrics(tp, ty)}
    for name, mk in (("isotonic", isotonic), ("platt", platt), ("beta", beta_calibration)):
        try: out[name] = calibration_metrics(mk(xs, ys)(tp), ty)
        except Exception as e: out[name] = {"error": str(e)}
    return out


# ================================================================ active validation of the families

def _family_sub(seed, k):
    """Q/R — candidate family = skeleton discovered on one instance; promotion by (a) CLUSTERING: canonical
    signature identical to that of a 2nd discovered program; (b) ACTIVE VALIDATION: on a NEW instance, the
    family-prior reconstruction (fitted skeleton + active verification) succeeds and costs less."""
    from core.active_structure import program_signature
    rows = []
    if True:
        base = make_task(seed * 50 + k, noise=0.05, depth=2)
        rng = random.Random(seed * 50 + k)
        same = make_task(seed * 50 + k + 7, noise=0.05, depth=2); same_truth = base.world.truth.skeleton().with_params(
            [v + rng.uniform(-0.2, 0.2) for v in base.world.truth.param_vector()])
        same.world.truth = same_truth; same.world.label = (lambda X, p=same_truth: predict(p, X))
        g = torch.Generator().manual_seed(seed); same.y = __import__("core.robust_world", fromlist=["corrupt"]).corrupt(same.world, same.X, SENSORS["standard"], same.spec, g)
        other = make_task(seed * 50 + k + 13, noise=0.05, depth=2)
        rA = RobustStructuralReconstruction(planner=RobustMeasurementPlanner("efa", world_noise=0.05), seed=seed).run(obs_of(base), lambda x, s, gg: base.oracle(x, s, gg), full_region, full_region(400, g), g)
        fam = rA.css.best().program.skeleton(); fam_init = rA.css.best().program.param_vector()   # family = skeleton + typical parameters (prior)
        for tgt, is_same in ((same, True), (other, False)):
            rB = RobustStructuralReconstruction(planner=RobustMeasurementPlanner("efa", world_noise=0.05), seed=seed).run(obs_of(tgt), lambda x, s, gg: tgt.oracle(x, s, gg), full_region, full_region(400, g), g)
            cluster = program_signature(rA.css.best().program).distance(program_signature(rB.css.best().program), canonical=True) <= 0.08
            o = obs_of(tgt); p, ll = fit_robust(fam, o, n_init=6, init=fam_init); e_fam = log_evidence(p, o, ll); e_search = rB.css.best().evidence
            Xn = full_region(60, g)                                         # same points for X and y (fixed: they were drawn separately)
            ppc = posterior_predictive_check(p, Observations.build(Xn, tgt.oracle(Xn, "standard", g), SENSOR_MODELS["standard"], 0.05), g=g)
            active = e_fam >= e_search - 2.0 and not ppc["reject"]
            rows.append({"same_structure": is_same, "promote_cluster": cluster, "promote_active": active,
                         "family_recovery": evaluate_prog(p, tgt, g)["recovery"], "search_recovery": evaluate_prog(rB.css.best().program, tgt, g)["recovery"],
                         "family_evals": 1, "search_evals": rB.evaluations})
    return rows


def family_subunits(seed): return [(f"k{k}", (lambda k=k: _family_sub(seed, k))) for k in range(3)]


# ================================================================ capacity, compression of the residual

def _capacity_sub(seed, k):
    rows = []
    if True:
        task = make_task(seed * 9 + k, noise=0.08, depth=1 + k % 4)
        prog, info = run_method("full", task, seed * 9 + k); info.pop("trace", None)
        X = task.X[task.y >= 0]; y = task.y[task.y >= 0]; err = X[predict(prog, X) != y]
        # AH — residuals: raw gated (stored points), prototypes (k-means, 4), small local model, none
        from core.active_structure import gated_residual
        gp, cost, gate = gated_residual(prog, X, y, steps=200)
        protos = _kmeans(err, 4) if len(err) >= 4 else err
        I = in_region(1500, torch.Generator().manual_seed(1)); O = out_region(1500, torch.Generator().manual_seed(1)); lab = task.world.label
        def proto_pred(Q, P=protos):
            base = predict(prog, Q)
            if len(P) == 0: return base
            m = torch.cdist(Q, P).min(1).values < 0.3; return torch.where(m, 1 - base, base)
        acc = lambda f, Q: float((f(Q) == lab(Q)).float().mean())
        rows.append({"params": prog.n_params(), "ast": prog.size(), "raw_points": len(err), "violated": prog.n_params() > COMPACT_PARAMS or len(protos) > COMPACT_POINTS,
                     "none": (acc(lambda Q: predict(prog, Q), I), acc(lambda Q: predict(prog, Q), O), 0), "raw_gated": (acc(gp, I), acc(gp, O), 2 * len(err)),
                     "prototype": (acc(proto_pred, I), acc(proto_pred, O), 2 * len(protos))})
    return rows


def capacity_subunits(seed): return [(f"k{k}", (lambda k=k: _capacity_sub(seed, k))) for k in range(4)]


def _kmeans(P, k, iters=20):
    C = P[torch.randperm(len(P), generator=torch.Generator().manual_seed(0))[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(P, C).argmin(1)
        for j in range(k):
            if int((a == j).sum()): C[j] = P[a == j].mean(0)
    return C


# ================================================================ identifiability

IDENT_SPECS = ((True, 1), (True, 3), (True, 4), (False, 0))


def _ident_sub(seed, k):
    from core.active_structure import StructuralPosterior, identifiability
    rows = []; rep, d = IDENT_SPECS[k]
    if True:
        task = make_task(seed * 11 + k, noise=0.08, depth=d or None, representable=rep)
        r = RobustStructuralReconstruction(seed=seed).run(obs_of(task)); css = r.css
        post = StructuralPosterior([e.program for e in css.entries], 0.08); post.logw = torch.tensor([e.evidence for e in css.entries], dtype=torch.float64)
        obs_only = identifiability(post, in_region(300, torch.Generator().manual_seed(2))); full = identifiability(post, full_region(300, torch.Generator().manual_seed(2)))
        if r.status == "model_mismatch": cls = "OUT_OF_HYPOTHESIS_CLASS"
        elif full["identifiable"] and obs_only["identifiable"]: cls = "IDENTIFIABLE"
        elif full["identifiable"]: cls = "WEAKLY_IDENTIFIABLE"
        else: cls = "NON_IDENTIFIABLE"
        rows.append({"representable": rep, "depth": d, "class": cls, "discrepancy": r.discrepancy["label"], **evaluate_prog(css.best().program, task, torch.Generator().manual_seed(15))})
    return rows


def identifiability_subunits(seed): return [(f"k{k}", (lambda k=k: _ident_sub(seed, k))) for k in range(len(IDENT_SPECS))]


SEALED_METHODS = ("old_system", "passive", "robust_posterior", "full", "random_queries")


def _sealed_sub(index, m):
    from core.robust_world import sealed_tasks
    task = sealed_tasks()[index]; prog, info = run_method(m, task, 992000 + index); info.pop("trace", None)
    return [{"index": index, "noise": task.spec.rate, "kind": task.spec.kind, "representable": task.world.representable, "method": m,
             **evaluate_prog(prog, task, torch.Generator().manual_seed(16)), **{k: info.get(k) for k in ("queries", "evaluations", "status", "answer_unknown")}}]


def sealed_subunits(index): return [(m, (lambda m=m: _sealed_sub(index, m))) for m in SEALED_METHODS]


def smoke_subunits(key):
    def one(i):
        import os, platform
        return [{"i": i, "python": platform.python_version(), "torch": torch.__version__, "cpu": os.cpu_count(), "check": float(torch.arange(1000.0).sum())}]
    return [(f"p{i}", (lambda i=i: one(i))) for i in range(4)]


SUBUNITS = {"main": main_subunits, "noise": noise_subunits, "hard": hard_subunits, "repeat": repeat_subunits, "calibration": calibration_subunits,
            "caltrain": caltrain_subunits, "family": family_subunits, "capacity": capacity_subunits, "identifiability": identifiability_subunits}
ALL_SUBUNITS = {**SUBUNITS, "sealed": sealed_subunits, "smoke": smoke_subunits}


def subunit_ids(kind, key): return [sid for sid, _ in ALL_SUBUNITS[kind](int(key))]


def run_subunit(kind, key, sid):
    """Runs ONE sub-unit (deterministic in (type, key, identifier)) and returns its rows."""
    return dict(ALL_SUBUNITS[kind](int(key)))[sid]()


def run_job(kind, key, done=None):
    """Full unit = ordered concatenation of the sub-units; `done` = {identifier: rows} already computed (resume)."""
    done = done or {}; rows = []
    for sid, fn in ALL_SUBUNITS[kind](int(key)): rows += done[sid] if sid in done else fn()
    return rows


JOBS = {k: (lambda seed, k=k: run_job(k, seed)) for k in SUBUNITS}


def sealed_job(index): return run_job("sealed", index)

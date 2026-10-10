"""Experiments of the milestone Deep Compositional Structure Discovery — EVALUATION side (the only module that sees both the
world and the system). The diagnostics that use the truth never return anything to the system.

AWS units = (type, key) split into deterministic sub-units (checkpoints, resume):
  deepprofile : cost and fate of the true structure per depth, all methods (micro-benchmark)
  deepdev     : exploration in dev (settings)
  deepval     : validation (selection of the primary, calibration of the boundary)
  deeptest    : confirmatory test (16 seeds)
  deepsealed  : sealed set (only once)
"""
from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch

from core.boundary_program import BNode, predict, render
from core.deep_boundary import (SIGNALS, BoundaryModel, CompatibleFamilies, Uncertainty, decide, decision_cost, repeat_noise, sig_depth_improvement,
                                sig_holdout_ppc, sig_mdl_bits, sig_nonparametric_gap, sig_parameter_instability, sig_precise, sig_repeat_noise_excess,
                                sig_residual_clustering, sig_temporal_drift, split_holdout, suggest_missing_primitive)
from core.deep_equivalence import HypothesisPool, active_identification, identifiability
from core.deep_fit import DT, FitData
from core.deep_search import STRATEGIES, DeepSearch
from core.deep_structure import KNOWN, NONE_OF_THE_ABOVE, UNKNOWN, FragmentLibrary, algebraic_form, n_atoms, skel_key, structurally_equivalent, subtrees
from core.deep_world import DEPTHS, HARD_VALID, OOC_CATEGORIES, full_region, make_task, obs_region, out_region, sealed_spec, sealed_task

QUERY_BUDGET, PRECISE_BUDGET, REPEAT_BUDGET = 16, 24, 24            # preregistered (addendum §10)


@dataclass
class DeepConfig:
    strategy: str = "bottom_up"
    budget: float = 150.0
    patience: int = 1
    beam: int = 3
    policy: str = "class_split"
    hierarchy: bool = True
    use_classes: bool = True
    use_none: bool = True
    use_families: bool = True
    library: Optional[dict] = None                # fragment library (learned in dev), serialised
    boundary: Optional[dict] = None               # serialised BoundaryModel (fitted on validation)
    tau_none: float = 0.5
    tau_known: float = 0.2
    max_depth: int = 6

    def hash(self): return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def _mem_mb(): return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


# ================================================================ metrics (truth: evaluation only)

def evaluate_program(prog: Optional[BNode], task):
    """ExactStructure / EquivalentStructure / FunctionallyEquivalent / PredictivelyAdequate, never merged."""
    g = torch.Generator().manual_seed(11); F, I, O = full_region(4000, g), obs_region(2000, g), out_region(2000, g); lab = lambda X: task.world.label(X, 1.0)
    if prog is None: return {"exact": False, "equivalent": False, "functional": False, "adequate": False, "full": 0.0, "in_region": 0.0, "extrapolation": 0.0, "atoms": 0, "params": 0}
    a = float((predict(prog, F) == lab(F)).float().mean()); ai = float((predict(prog, I) == lab(I)).float().mean()); ao = float((predict(prog, O) == lab(O)).float().mean()); tr = task.world.truth
    return {"exact": tr is not None and skel_key(prog) == skel_key(tr), "equivalent": tr is not None and structurally_equivalent(prog, tr), "functional": a >= 0.99,
            "adequate": ai >= 0.95 and ao >= 0.95, "full": a, "in_region": ai, "extrapolation": ao, "atoms": n_atoms(prog), "params": prog.n_params()}


def truth_fate(task, candidates: Dict[str, BNode], selected: Optional[BNode], class_forms=()):
    """Fate of the true structure in a search. `candidates`: {key: fitted program} of EVERYTHING that was
    fitted. Categories: never_generated / generated_bad_fit / good_fit_misranked / equivalent_class / selected."""
    tr = task.world.truth
    if tr is None: return {"fate": "out_of_class", "rank": None}
    form = algebraic_form(tr); F = full_region(3000, torch.Generator().manual_seed(12)); lab = task.world.label(F, 1.0)
    same = [p for p in candidates.values() if algebraic_form(p) == form]
    cand_forms = {algebraic_form(s) for c in candidates.values() for s in subtrees(c, min_atoms=2)}          # sub-structures present in the candidates
    sub_found = sum(1 for s in subtrees(tr, min_atoms=2) if algebraic_form(s) in cand_forms)
    out = {"generated": bool(same), "true_substructures_found": sub_found, "true_substructures": len(subtrees(tr, min_atoms=2)), "candidates": len(candidates)}
    if not same: return {**out, "fate": "never_generated"}
    acc = max(float((predict(p, F) == lab).float().mean()) for p in same); out["true_structure_best_fit"] = acc
    if selected is not None and algebraic_form(selected) == form: return {**out, "fate": "selected"}
    if acc < 0.97: return {**out, "fate": "generated_bad_fit"}
    if form in class_forms: return {**out, "fate": "equivalent_class"}
    return {**out, "fate": "good_fit_misranked"}


# ================================================================ new system

def run_new(task, cfg: DeepConfig, seed=0):
    """Full pipeline of the new system on a task. The noise of the world is NOT supplied."""
    t0 = time.time(); g = torch.Generator().manual_seed(seed); data = FitData.build(task.X, task.y, 0.10); train, hold, itr, iho = split_holdout(data, 0.2, seed)
    lib = FragmentLibrary.from_dict(cfg.library) if cfg.library else None
    out = DeepSearch(cfg.strategy, budget=cfg.budget, max_depth=cfg.max_depth, beam=cfg.beam, seed=seed, library=lib, use_families=cfg.use_families, hierarchy=cfg.hierarchy,
                     dedup=cfg.use_classes, patience=cfg.patience).run(train)
    t_search = time.time() - t0; pool_pts = full_region(300, g); oracle = lambda x, s, gg: task.oracle(x, s, gg)
    pre = HypothesisPool(list(out.all.values()), train, pool_pts); class_before = pre.best_class().size
    policy = cfg.policy if cfg.use_classes or cfg.policy != "class_split" else "structural_ig"
    act = active_identification(out, train, oracle, pool_pts, policy=policy, budget=QUERY_BUDGET, seed=seed); hp = act.pool
    best = hp.hyps[hp.best_index()]; ident = identifiability(hp, out.stats["budget_exhausted"], any(h.depth >= cfg.max_depth for h in hp.hyps[:3]))
    # boundary validation measurements (NOT used for the fit)
    Xin, Xout = obs_region(PRECISE_BUDGET // 2, g), out_region(PRECISE_BUDGET // 2, g); yin, yout = task.oracle(Xin, "precise", g), task.oracle(Xout, "precise", g)
    ri = torch.randperm(len(task.X), generator=g)[:REPEAT_BUDGET]; yrep = task.oracle(task.X[ri], "standard", g); eps_rep = repeat_noise(task.y[ri], yrep)
    tq = torch.cat([task.t[itr], torch.linspace(1.0, 1.0 + 0.02 * QUERY_BUDGET, QUERY_BUDGET)])
    sig = {"holdout_ppc": sig_holdout_ppc(best, train, hold), "residual_clustering": sig_residual_clustering(best, hp.data), "nonparametric_gap": sig_nonparametric_gap(best, hp.data),
           "mdl_bits_per_obs": sig_mdl_bits(best, hp.data), "parameter_instability": sig_parameter_instability(best, hp.data, pool_pts, seed=seed),
           "depth_improvement": sig_depth_improvement(out, train), "temporal_drift": sig_temporal_drift(best, hp.data.X, hp.data.y, tq),
           "precise_in": sig_precise(best, Xin, yin), "precise_out": sig_precise(best, Xout, yout), "repeat_noise_excess": sig_repeat_noise_excess(best, hp.data, eps_rep, REPEAT_BUDGET),
           "structural_disagreement": 1.0 - float(hp.mass.max())}
    fam = CompatibleFamilies.build(hp.hyps, {h.key: hp.P_obs[i] for i, h in enumerate(hp.hyps)}, hp.data.y, noise=max(eps_rep, 0.02))
    p_ooc = BoundaryModel.from_dict(cfg.boundary).prob([sig])[0] if cfg.boundary else None
    label = decide(p_ooc if p_ooc is not None else 0.0, ident["class_size"], float(hp.mass.max()), cfg.tau_none, cfg.tau_known, use_none=cfg.use_none and p_ooc is not None,
                   use_classes=cfg.use_classes, families_all_rejected=cfg.use_none and fam.all_rejected())
    unc = Uncertainty(sig["parameter_instability"], 1.0 - float(hp.mass.max()), 1.0 - max(fam.mass.values()) if fam.mass else 1.0, p_ooc if p_ooc is not None else float("nan"))
    ev = evaluate_program(best.program, task); cls = hp.best_class()
    fate = truth_fate(task, {k: h.program for k, h in out.all.items()}, best.program, cls.forms)
    truth_in_class = task.world.truth is not None and algebraic_form(task.world.truth) in cls.forms
    return {"method": cfg.strategy, **ev, "decision": label, "p_ooc": p_ooc, "signals": sig, "identifiability": ident["status"], "class_size": ident["class_size"], "class_size_before": class_before,
            "classes": ident["classes"], "breakable": ident["breakable"], "reasons": ident["reasons"], "truth_in_class": truth_in_class, **fate,
            "fit_equivalents": out.stats["fit_equivalents"], "generated": out.stats["generated"], "fitted": out.stats["fitted"], "pruned": out.stats["pruned"], "distinct_structures": out.stats["distinct_structures"],
            "families": out.stats["families"], "families_rejected": len(fam.rejected), "per_depth": out.stats["per_depth"], "store": out.stats["store"], "budget_exhausted": out.stats["budget_exhausted"],
            "seconds": round(time.time() - t0, 2), "search_seconds": round(t_search, 2), "fit_seconds": out.stats["fit_seconds"], "memory_mb": _mem_mb(), "queries": QUERY_BUDGET,
            "eps_repeat": eps_rep, "uncertainty": asdict(unc), "program": render(best.program), "program_dict": best.program.to_dict(),
            "suggestion": suggest_missing_primitive(best, hp.data, sig["temporal_drift"], sig["precise_out"], sig["depth_improvement"]), "trace_stages": _stage_counts(out.trace)}


def _stage_counts(trace):
    c = {}
    for t in trace: c[t["stage"]] = c.get(t["stage"], 0) + 1
    return c


# ================================================================ frozen baseline (previous milestone), instrumented without modifying it

def run_baseline(task, seed=0, internal=False):
    """System of the previous milestone ("full system" configuration), unchanged. It receives the TRUE noise of the world.
    The instrumentation is a subclass that RECORDS the fits without changing any computation."""
    from core.robust_reconstruction import SENSOR_MODELS, Observations, RobustMeasurementPlanner, RobustStructuralReconstruction
    rec = {"fits": {}, "fe": 0.0}

    class Instrumented(RobustStructuralReconstruction):
        def _fit(self, skel, obs, init=None):
            p, ll = super()._fit(skel, obs, init); rec["fe"] += (1 if init is not None else 2); k = skel_key(p)
            if k not in rec["fits"] or ll > rec["fits"][k][1]: rec["fits"][k] = (p, ll)
            return p, ll

    t0 = time.time(); nz = task.spec.rate; g = torch.Generator().manual_seed(seed); obs = Observations.build(task.X, task.y, SENSOR_MODELS["standard"], nz)
    pl = RobustMeasurementPlanner("efa", sensors=tuple(SENSOR_MODELS), world_noise=nz, allow_repeat=True, seed=seed)
    r = Instrumented(planner=pl, budget=16.0, seed=seed).run(obs, lambda x, s, gg: task.oracle(x, s, gg), full_region, full_region(400, torch.Generator().manual_seed(seed + 77)), g)
    best = r.css.best().program; ev = evaluate_program(best, task); cands = {k: p for k, (p, _) in rec["fits"].items()}
    kept = {algebraic_form(e.program) for e in r.css.entries}; fate = truth_fate(task, cands, best)
    if fate.get("fate") in ("good_fit_misranked", "generated_bad_fit") and task.world.truth is not None and algebraic_form(task.world.truth) not in kept: fate["fate_detail"] = "pruned"
    label = {"identified": KNOWN, "unknown": UNKNOWN, "model_mismatch": NONE_OF_THE_ABOVE}[r.status]
    if internal: return cands, [e.program for e in sorted(r.css.entries, key=lambda e: -e.mass)], best, fate.get("true_structure_best_fit", 0.0), ev, rec["fe"], round(time.time() - t0, 2)
    return {"method": "baseline", **ev, "decision": label, "p_ooc": None, **fate, "fit_equivalents": rec["fe"], "generated": len(cands), "fitted": r.evaluations, "distinct_structures": len(cands),
            "seconds": round(time.time() - t0, 2), "memory_mb": _mem_mb(), "queries": r.queries, "search_depth": r.depth, "program": render(best), "class_size": 1}


# ================================================================ sub-units

PROFILE_METHODS = ("baseline",) + STRATEGIES


def _row(task, res, **extra):
    w = task.world; return {"depth": w.depth, "in_class": w.in_class, "category": w.category, "unseen": w.unseen, "regime": task.regime, "noise": task.spec.rate, "kind": task.spec.kind, "n_obs": len(task.y), **extra, **res}


def _run(task, method, cfg: DeepConfig, seed):
    if method == "baseline": return run_baseline(task, seed)
    c = DeepConfig(**{**asdict(cfg), "strategy": method}); return run_new(task, c, seed)


def profile_subunits(key):
    """Micro-benchmark: depths 1–6, 2 dev tasks per depth (standard regime), all the methods at the default
    budget. Used to learn the cost per depth and the consumption of the baseline (matched budget)."""
    key = int(key); out = []
    for d in DEPTHS:
        for j in range(2):
            for m in PROFILE_METHODS:
                out.append((f"d{d}_t{j}_{m}", (lambda d=d, j=j, m=m: [_row(make_task(900 + key * 20 + d * 2 + j, d, "standard", split="dev"), _run(make_task(900 + key * 20 + d * 2 + j, d, "standard", split="dev"), m, DeepConfig(), key), task_index=j)])))
    return out


def load_config(name: str) -> DeepConfig:
    """Frozen configuration read from results/deep_configs/<name>.json (created in dev / validation, fingerprint in the manifest)."""
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "deep_configs", f"{name}.json")
    return DeepConfig(**json.load(open(p))) if os.path.exists(p) else DeepConfig()


def dev_grid():
    """Exploration grid IN DEV (settings): strategy × budget × patience. Never applied to the test nor to the sealed set."""
    return [{"strategy": s, "budget": b, "patience": p} for s in STRATEGIES for b in (150.0,) for p in (1, 2)]


def dev_tasks(key):
    """Dev tasks of the key: depths 1–5 in the standard regime + one out-of-class world + one hard-valid one."""
    key = int(key); T = [("in", make_task(key * 40 + d, d, "standard", kind=("iid", "boundary", "grouped_outliers")[(key + d) % 3], split="dev")) for d in (1, 2, 3, 4, 5)]
    T.append(("ooc", make_task(key * 40 + 10, category=OOC_CATEGORIES[key % len(OOC_CATEGORIES)], split="dev")))
    T.append(("hard", make_task(key * 40 + 11, category=HARD_VALID[key % len(HARD_VALID)], split="dev")))
    return T


def dev_subunits(key):
    out = []
    for ti in range(7):
        out.append((f"t{ti}_baseline", (lambda ti=ti: [_row(dev_tasks(key)[ti][1], run_baseline(dev_tasks(key)[ti][1], int(key)), task_index=ti, config="baseline")])))
        for gi, gcfg in enumerate(dev_grid()):
            out.append((f"t{ti}_g{gi}", (lambda ti=ti, gi=gi, gcfg=gcfg: [_row(dev_tasks(key)[ti][1], run_new(dev_tasks(key)[ti][1], DeepConfig(**gcfg), int(key)), task_index=ti, config=json.dumps(gcfg, sort_keys=True))])))
    return out


# ================================================================ sub-milestone Deep Candidate Generation (dev only)

GEN_METHODS = (("baseline", 0.0), ("flat", 150.0), ("bottom_up", 50.0), ("bottom_up", 150.0), ("bottom_up", 450.0), ("bidirectional", 150.0),
               ("factorized", 50.0), ("factorized", 150.0), ("factorized", 450.0), ("factorized_sep", 150.0))
GEN_REGIMES = ("standard", "rich")


def gen_task(key, d, regime): return make_task(7000 + int(key) * 20 + d, d, regime, kind=("iid", "boundary", "grouped_outliers")[(int(key) + d) % 3], split="dev")


def run_generation(task, method, budget, seed=0):
    """Search ALONE (no active measurements and no boundary) + generation diagnostic: recalls of the true
    structure at each stage, main cause of the failure, separate costs (candidates, fits, compositions, CPU)."""
    from core.deep_reachability import classify
    t0 = time.time(); tr = task.world.truth; F = full_region(3000, torch.Generator().manual_seed(12)); lab = task.world.label(F, 1.0)
    if method == "baseline":
        cands, ranked, best, fitq, ev, fe, secs = run_baseline(task, seed, internal=True); c = classify(tr, "baseline", cands, ranked, best, fitq >= 0.97, False)
        return {"method": method, "budget": budget, **{k: ev[k] for k in ("functional", "exact", "equivalent", "adequate", "full", "extrapolation", "atoms")}, **c, "true_structure_fit": fitq, "fit_equivalents": fe,
                "n_candidates": len(cands), "n_fitted": len(cands), "n_structures": len(cands), "compositions": 0, "seconds": secs, "fit_seconds": secs, "synthesis_seconds": 0.0, "class_size": 1, "generation": {}}
    kw = {"fact_separation": True} if method == "factorized_sep" else {}; strat = "factorized" if method.startswith("factorized") else method
    data = FitData.build(task.X, task.y, 0.10); out = DeepSearch(strat, budget=budget, seed=seed, **kw).run(data); best = out.best(); form = algebraic_form(tr)
    same = [h.program for h in out.all.values() if algebraic_form(h.program) == form]; fitq = max((float((predict(p, F) == lab).float().mean()) for p in same), default=0.0)
    hp = HypothesisPool(list(out.all.values()), data, full_region(200, torch.Generator().manual_seed(seed))); cls = hp.best_class()
    c = classify(tr, strat, {k: h.program for k, h in out.all.items()}, [h.program for h in out.hypotheses], best.program, fitq >= 0.97, out.stats["budget_exhausted"], cls.forms)
    ev = evaluate_program(best.program, task); g = out.stats.get("generation", {})
    return {"method": method, "budget": budget, **{k: ev[k] for k in ("functional", "exact", "equivalent", "adequate", "full", "extrapolation", "atoms")}, **c, "true_structure_fit": fitq,
            "fit_equivalents": out.stats["fit_equivalents"], "n_candidates": out.stats["generated"], "n_fitted": out.stats["fitted"], "n_structures": out.stats["distinct_structures"], "compositions": out.stats["compositions"],
            "seconds": round(time.time() - t0, 2), "fit_seconds": out.stats["fit_seconds"], "synthesis_seconds": out.stats["synthesis_seconds"], "class_size": cls.size,
            "generation": {k: g.get(k) for k in ("S1", "S2", "S3", "merged2", "merged3", "degenerate2", "degenerate3", "pairs_scored", "holes", "holes_pruned", "literals_dropped", "type_rejected", "t_tables", "t_synthesis")}}


def mechanics(task, seed=0):
    """MECHANICAL diagnostics of a task (truth used, diagnostic only): reachability per operator set,
    oracle path, identifiability bound, logic stage with the true atoms, atom recall per discovery
    method."""
    from core.boundary_program import atoms
    from core.deep_fit import FitCounter, approx_evidence, fit_batch, heuristic_inits
    from core.deep_generation import discover_atoms_by_separation, generate_candidates
    from core.deep_reachability import OPERATORS, basis_quality, leaves, operator_reachable, oracle_path, parent_closure, representable
    from core.deep_search import profiled_loglik
    from core.deep_world import _drop_variants
    tr = task.world.truth; data = FitData.build(task.X, task.y, 0.10); A = atoms(); P = obs_region(3000, torch.Generator().manual_seed(5)); tl = leaves(tr); form = algebraic_form(tr)
    ev = lambda p: approx_evidence(p.n_params(), n_atoms(p), profiled_loglik(predict(p, data.X.float()).to(DT), data), len(data))
    e = ev(tr); drops = _drop_variants(tr); base = predict(tr, task.X); eff = min((int((predict(v, task.X) != base).sum()) for v in drops), default=len(task.y))
    op = oracle_path(tr); out = {"representable": representable(tr)[0], "reachable": {m: operator_reachable(tr, ops)[0] for m, ops in OPERATORS.items()}, "oracle_length": op["length"], "width": op["width"],
                                  "needs_compose_pair": op["needs_compose_pair"], "alternatives_log10": op["alternatives_product_log10"], "closure": parent_closure(tr)["closed"], **basis_quality(tr),
                                  "identifiable_bound": all(e > ev(v) + math.log(3.0) for v in drops), "weakest_atom_effect_points": eff, "expected_flips": round(len(task.y) * (task.spec.rate + 0.05), 1)}
    def recall(lits, thr=0.95):
        return sum(max((float((predict(x, P) == predict(l, P)).float().mean()) for x in lits if skel_key(x) == skel_key(l)), default=0.0) >= thr for l in tl) / len(tl)
    c1 = FitCounter(); glob = []
    for a in A:
        th, _ = fit_batch(a, data, heuristic_inits(a, data, 6, seed), steps=50, counter=c1); glob += [a.with_params(r.tolist()) for r in th]
    c2 = FitCounter(); sep, st = discover_atoms_by_separation(A, task.X, task.y, seed=seed, counter=c2)
    out["atom_recall"] = {"global_gradient": recall(glob), "global_gradient_090": recall(glob, 0.90), "separation": recall(sep), "separation_090": recall(sep, 0.90)}
    out["atom_fit_equivalents"] = {"global_gradient": c1.fit_equivalents, "separation": c2.fit_equivalents}; out["boundary_pairs"] = st["pairs"]
    for name, lits in (("true_atoms", tl), ("true_atoms_plus_distractors", tl + glob[:12])):                      # LITERAL ORACLE: the logic stage alone
        t0 = time.time(); gen = generate_candidates(task.X, task.y, max_depth=n_atoms(tr), top=6, extra_literals=lits, geometry=False, max_literals=18)
        order = sorted(range(len(gen.programs)), key=lambda i: -gen.info[i]["score"]); hit = [order.index(i) for i, p in enumerate(gen.programs) if algebraic_form(p) == form]
        out[f"logic_{name}"] = {"generated": bool(hit), "rank": hit[0] if hit else None, "seconds": round(time.time() - t0, 3), "pairs_scored": gen.stats.get("pairs_scored"), "attempted": gen.stats.get("attempted"),
                                "merged": (gen.stats.get("merged2") or 0) + (gen.stats.get("merged3") or 0), "degenerate": (gen.stats.get("degenerate2") or 0) + (gen.stats.get("degenerate3") or 0)}
    return out


def gen_subunits(key):
    out = []
    for d in DEPTHS:
        for reg in GEN_REGIMES:
            out.append((f"d{d}_{reg}_mechanics", (lambda d=d, reg=reg: [_row(gen_task(key, d, reg), {"method": "mechanics", **mechanics(gen_task(key, d, reg), int(key))})])))
            for m, b in GEN_METHODS:
                out.append((f"d{d}_{reg}_{m}_{int(b)}", (lambda d=d, reg=reg, m=m, b=b: [_row(gen_task(key, d, reg), run_generation(gen_task(key, d, reg), m, b, int(key)))])))
    return out


# ---- exhaustive coverage at small depth -------------------------------------------------------------------

COVER_CHUNKS = 8
COVER_STRATEGIES = ("baseline", "flat", "bottom_up", "bidirectional", "factorized")


def cover_task(key, d, regime): return make_task(9000 + int(key) * 10 + d, d, regime, split="dev")


def _all_forms(max_depth):
    from core.deep_gen_report import enumerate_forms
    lv = enumerate_forms(max_depth); out = []
    for d in range(1, max_depth + 1): out += sorted(lv[d].items())
    return out


def cover_exhaustive(task, depth, chunk, n_init=4, seed=0):
    """EXHAUSTIVE ENUMERATION: all the forms of depth ≤ `depth` are fitted (batch of `n_init`
    initialisations). Returns, for the chunk `chunk`, the evidence of each form and the agreement of the best one."""
    from core.deep_fit import approx_evidence, fit_batch, heuristic_inits
    from core.deep_search import profiled_loglik
    forms = _all_forms(depth); mine = forms[chunk::COVER_CHUNKS]; data = FitData.build(task.X, task.y, 0.10); tf = algebraic_form(task.world.truth); F = full_region(3000, torch.Generator().manual_seed(12)); lab = task.world.label(F, 1.0); out = []
    t0 = time.time()
    for f, skel in mine:
        th, ll = fit_batch(skel, data, heuristic_inits(skel, data, n_init, seed), steps=100); i = int(ll.argmax()); p = skel.with_params(th[i].tolist()); pred = predict(p, data.X.float()).to(DT)
        ev = approx_evidence(p.n_params(), n_atoms(p), profiled_loglik(pred, data), len(data)); out.append({"is_truth": f == tf, "atoms": n_atoms(p), "evidence": ev, "full": float((predict(p, F) == lab).float().mean())})
    return {"method": "exhaustive", "cover_depth": depth, "chunk": chunk, "n_forms": len(mine), "forms_total": len(forms), "fit_equivalents": len(mine) * n_init, "seconds": round(time.time() - t0, 2), "forms": out}


def cover_strategy(task, depth, method, seed=0):
    """Coverage of a strategy at the reference budget: distinct forms generated, per depth."""
    if method == "baseline": cands = run_baseline(task, seed, internal=True)[0].values()
    else: cands = [h.program for h in DeepSearch(method, budget=150.0, seed=seed, max_depth=depth).run(FitData.build(task.X, task.y, 0.10)).all.values()]
    by = {}
    for p in cands: by.setdefault(n_atoms(p), set()).add(algebraic_form(p))
    return {"method": method, "cover_depth": depth, "forms_generated": {str(d): len(v) for d, v in sorted(by.items())}, "truth_generated": algebraic_form(task.world.truth) in set().union(*by.values()) if by else False}


def cover_subunits(key):
    out = []
    for d in (2, 3):
        for reg in GEN_REGIMES:
            for c in range(COVER_CHUNKS): out.append((f"d{d}_{reg}_exh{c}", (lambda d=d, reg=reg, c=c: [_row(cover_task(key, d, reg), cover_exhaustive(cover_task(key, d, reg), d, c, seed=int(key)))])))
            for m in COVER_STRATEGIES: out.append((f"d{d}_{reg}_{m}", (lambda d=d, reg=reg, m=m: [_row(cover_task(key, d, reg), cover_strategy(cover_task(key, d, reg), d, m, int(key)))])))
    return out


# ================================================================ sub-milestone Primitive Discovery (dev only)

PRIM_METHODS = ("gradient", "geometric", "contrast", "hybrid", "hybrid_minimal", "neural", "neural_hybrid")
CHAIN_METHODS = ("gradient", "hybrid", "hybrid_minimal", "neural_hybrid")
CHAIN_DEPTHS = (2, 3, 4, 5)


def prim_proposer(name):
    """Proposers compared: direct gradient (reference), analytic geometric, contrast, hybrid, compact neural."""
    from core.deep_neural_proposer import NeuralProposer
    from core.deep_primitives import ContrastProposer, GeometricProposer, GradientProposer, RefiningProposer
    if name == "gradient": return GradientProposer()
    if name == "geometric": return GeometricProposer()
    if name == "contrast": return ContrastProposer()
    if name == "hybrid": return RefiningProposer([GeometricProposer(), ContrastProposer(n_init=2)])
    if name == "hybrid_minimal": return RefiningProposer([GeometricProposer("minimal"), ContrastProposer("minimal", n_init=3)])
    if name == "neural": return NeuralProposer()
    if name == "neural_single": return NeuralProposer(variant="single")
    if name == "neural_hybrid": return RefiningProposer([NeuralProposer(), GeometricProposer(), ContrastProposer(n_init=2)])
    raise KeyError(name)


def _neural_missing(name):
    from core.deep_neural_proposer import load
    return name.startswith("neural") and load("single" if name == "neural_single" else "full") is None


def prim_scene(scene, method, seed=0, extra=None):
    """Primitive discovery on an atomic scene + metrics (truth: evaluation only)."""
    from core.deep_primitive_eval import primitive_metrics
    from core.deep_primitives import discover
    t = scene.task
    if _neural_missing(method): return {"method": method, "unavailable": True}
    X, y = t.X, t.y
    if extra is not None: X, y = torch.cat([X, extra[0]]), torch.cat([y, extra[1]])
    d = discover(X, y, prim_proposer(method), seed=seed); found = [m.program for c in d.primitives.classes for m in c.members]; P = obs_region(3000, torch.Generator().manual_seed(5))
    m = primitive_metrics(scene.atoms, found, X, y, P); reps = [c.representative for c in d.primitives.classes]
    return {"method": method, **{k: m[k] for k in m if k != "atoms"}, "atoms": m["atoms"], "classes": len(d.primitives), "proposals": d.proposals, "validated": d.validated, "rounds": d.rounds, "seconds": d.seconds,
            "fit_equivalents": d.fit_equivalents, "first_validated_at": d.first_validated_at, "residual_left": d.residual_left, "description_bits": sum(r.description_bits for r in reps),
            "found": [{"program": r.program.to_dict(), "z": r.z, "support": r.support, "purity": r.purity, "type": r.type} for r in reps], "types_true": [a["type"] for a in m["atoms"]]}


def prim_active(scene, policy, seed=0, budget=16):
    """Active measurements directed at the primitives (disagreement internal to the classes) versus random measurements."""
    from core.deep_primitives import discover, primitive_query_scores
    t = scene.task; g = torch.Generator().manual_seed(seed); pool = obs_region(400, g); d = discover(t.X, t.y, prim_proposer("hybrid"), seed=seed)
    if policy == "primitive": sc = primitive_query_scores(d.primitives, pool); idx = sc.topk(budget).indices if float(sc.max()) > 0 else torch.randperm(len(pool), generator=g)[:budget]
    else: idx = torch.randperm(len(pool), generator=g)[:budget]
    x = pool[idx]; y = t.oracle(x, "standard", g); r = prim_scene(scene, "hybrid", seed, extra=(x, y)); r["method"] = f"active_{policy}"; r["queries"] = budget; return r


def prim_chain(task, method, seed=0):
    """Causal chain: recall of the primitives → recall of the candidate structure → final recovery, and oracle
    decomposition matrix (true atoms / discovered atoms) × (true composition / WCE composition)."""
    from core.deep_primitive_eval import primitive_metrics
    from core.deep_primitives import discover
    from core.deep_reachability import leaves
    tr = task.world.truth; tl = leaves(tr); P = obs_region(3000, torch.Generator().manual_seed(5)); F = full_region(3000, torch.Generator().manual_seed(12)); lab = task.world.label(F, 1.0); form = algebraic_form(tr); data = FitData.build(task.X, task.y, 0.10)
    def compose(kw):
        out = DeepSearch("factorized", budget=150.0, seed=seed, max_depth=min(6, n_atoms(tr) + 1), **kw).run(data); best = out.best(); acc = [float((predict(h.program, F) == lab).float().mean()) for h in out.all.values()]
        return out, {"candidate_form": form in {algebraic_form(h.program) for h in out.all.values()}, "candidate_functional": max(acc, default=0.0) >= 0.97, "final_functional": float((predict(best.program, F) == lab).float().mean()) >= 0.99,
                     "final_full": float((predict(best.program, F) == lab).float().mean()), "final_form": algebraic_form(best.program) == form, "fit_equivalents": out.stats["fit_equivalents"], "seconds": out.stats["seconds"]}
    if method == "oracle_atoms":                                              # true atoms + WCE composition (diagnostic)
        _, c = compose({"fact_literals": tl, "fact_residual": False}); return {"method": method, "primitive_equivalence": 1.0, "primitive_exact": 1.0, **c, "discovered_true_composition": 1.0}
    if _neural_missing(method): return {"method": method, "unavailable": True}
    if method == "gradient": out, c = compose({}); found = [l for h in out.all.values() for l in leaves(h.program)]
    else:
        out, c = compose({"fact_discovery": prim_proposer(method)}); found = [m.program for cl in out.discovery.primitives.classes for m in cl.members] if out.discovery is not None else []
    m = primitive_metrics(tl, found, task.X, task.y, P)
    def sub(n):                                                               # discovered atoms + TRUE composition
        if n.kind in ("and", "or", "xor"): return BNode(n.kind, children=tuple(sub(ch) for ch in n.children))
        tn = predict(n, P); return max(found, key=lambda q: float((predict(q, P) == tn).float().mean())) if found else n
    dtc = float((predict(sub(tr), F) == lab).float().mean()) if found else 0.0
    return {"method": method, "primitive_equivalence": m["equivalence"], "primitive_exact": m["exact"], "primitive_functional": m["functional"], "component_recall": m["component_recall"], "primitive_precision": m["precision"],
            "n_identifiable": m["n_identifiable"], "equivalence_identifiable": m["equivalence_identifiable"], **c, "discovered_true_composition": dtc, "n_found": len(found)}


def prim_subunits(key):
    from core.deep_primitive_world import CATEGORIES, OOD_SHIFTS, make_scene
    k = int(key); out = []
    for cat in CATEGORIES:
        for j in range(2):
            for m in PRIM_METHODS:
                out.append((f"{cat}_{j}_{m}", (lambda cat=cat, j=j, m=m: [{"experiment": "atomic", "category": cat, "shift": None, **prim_scene(make_scene(k * 2 + j, cat), m, k)}])))
    for sh in OOD_SHIFTS:
        for m in ("hybrid", "neural", "neural_single", "neural_hybrid"):
            out.append((f"ood_{sh}_{m}", (lambda sh=sh, m=m: [{"experiment": "ood", "category": "two", "shift": sh, **prim_scene(make_scene(k, "two", sh), m, k)}])))
    for cat in ("two", "multi"):
        for pol in ("primitive", "random"):
            out.append((f"active_{cat}_{pol}", (lambda cat=cat, pol=pol: [{"experiment": "active", "category": cat, "shift": None, **prim_active(make_scene(k * 2, cat), pol, k)}])))
        out.append((f"active_{cat}_none", (lambda cat=cat: [{"experiment": "active", "category": cat, "shift": None, **{**prim_scene(make_scene(k * 2, cat), "hybrid", k), "method": "active_none"}}])))
    for d in CHAIN_DEPTHS:
        for reg in GEN_REGIMES:
            for m in CHAIN_METHODS + ("oracle_atoms",):
                out.append((f"chain_d{d}_{reg}_{m}", (lambda d=d, reg=reg, m=m: [_row(gen_task(k + 50, d, reg), {"experiment": "chain", **prim_chain(gen_task(k + 50, d, reg), m, k)})])))
    return out


def neural_train_scenes(variant, n):
    """TRAINING scenes of the proposer (dev partition, dedicated base, nominal distribution). `single`: a single atom;
    `full`: full curriculum (atomic scenes of all categories + deep dev compositions)."""
    from core.deep_primitive_eval import components
    from core.deep_primitive_world import CATEGORIES, TRAIN_BASE, make_scene
    from core.deep_reachability import leaves
    out = []
    for i in range(n):
        if variant == "single": sc = make_scene(i, "single", base=TRAIN_BASE); atoms_ = sc.atoms; t = sc.task
        elif i % 4 == 3: t = make_task(600_000 + i, 2 + i % 3, ("standard", "rich")[i % 2], split="dev"); atoms_ = leaves(t.world.truth)
        else: sc = make_scene(i, CATEGORIES[i % len(CATEGORIES)], base=TRAIN_BASE); atoms_ = sc.atoms; t = sc.task
        out.append((t.X, t.y, [c for a in atoms_ for c in components(a)]))
    return out


def neural_subunits(key):
    """Training of the compact neural proposer on AWS. Key 0: full curriculum; key 1: atoms alone."""
    variant = {0: "full", 1: "single"}[int(key)]
    def run():
        from core.deep_neural_proposer import state_to_json, train
        t0 = time.time(); scenes = neural_train_scenes(variant, 4000); tg = time.time() - t0; net, log = train(scenes, epochs=30, seed=0)
        return [{"variant": variant, "state": state_to_json(net), "data_seconds": round(tg, 1), **log}]
    return [("train", run)]


# ================================================================ sub-milestone Downstream Selection Loss (dev, true atoms supplied)

DOWN_DEPTHS = (2, 3, 4, 5)


def down_task(key, d, regime): return make_task(11000 + int(key) * 10 + d, d, regime, kind=("iid", "boundary", "grouped_outliers")[(int(key) + d) % 3], split="dev")


def down_subunits(key):
    from core.deep_downstream import trace_task
    return [(f"d{d}_{reg}", (lambda d=d, reg=reg: [_row(down_task(key, d, reg), {"experiment": "downstream", **trace_task(down_task(key, d, reg), int(key))})])) for d in DOWN_DEPTHS for reg in GEN_REGIMES]


# ================================================================ sub-milestone Validated Optimizer (dev, true atoms supplied)

VO_ARMS = ("A", "B", "C")


def vo_subunits(key):
    """Controlled experiment: keys 0–15 = the 128 tasks of the downstream diagnostic; keys 100–115 = 128 NEW dev tasks."""
    from core.deep_validated import arm_run
    return [(f"d{d}_{reg}_{arm}", (lambda d=d, reg=reg, arm=arm: [_row(down_task(key, d, reg), {"experiment": "arms", "task_set": "new" if int(key) >= 100 else "diagnostic", "task_id": f"{key}_{d}_{reg}", **arm_run(down_task(key, d, reg), arm, int(key))})]))
            for d in DOWN_DEPTHS for reg in GEN_REGIMES for arm in VO_ARMS]


def opt_subunits(key):
    from core.deep_validated import optimizer_diagnosis
    return [(f"d{d}_{reg}", (lambda d=d, reg=reg: [_row(down_task(key, d, reg), optimizer_diagnosis(down_task(key, d, reg), int(key)))])) for d in DOWN_DEPTHS for reg in GEN_REGIMES]


# ================================================================ sub-milestone Objective Alignment & Conditional Optimization (dev)

def obj_subunits(key):
    """Per-candidate study. Keys 300–307: TUNING set of the gate; keys 200–215: EVALUATION set (disjoint structures)."""
    from core.deep_objective import candidate_study
    return [(f"d{d}_{reg}", (lambda d=d, reg=reg: [_row(down_task(key, d, reg), {"task_set": "tuning" if int(key) >= 300 else "evaluation", "task_id": f"{key}_{d}_{reg}", **r}) for r in candidate_study(down_task(key, d, reg), int(key))])) for d in DOWN_DEPTHS for reg in GEN_REGIMES]


def e2e_subunits(key):
    from core.deep_objective import E2E_POLICIES, e2e_run
    return [(f"d{d}_{reg}_{lit}_{pol}", (lambda d=d, reg=reg, lit=lit, pol=pol: [_row(down_task(key, d, reg), {"task_id": f"{key}_{d}_{reg}", **e2e_run(down_task(key, d, reg), pol, lit, int(key))})]))
            for d in DOWN_DEPTHS for reg in GEN_REGIMES for lit in ("discovered", "true") for pol in E2E_POLICIES]


def resid_subunits(key):
    from core.deep_objective import RESIDUAL_TASKS, residual_case
    return [(f"t{k}_{d}_{reg}", (lambda k=k, d=d, reg=reg: [_row(down_task(k, d, reg), {"task_id": f"{k}_{d}_{reg}", **residual_case(down_task(k, d, reg), int(k))})])) for k, d, reg in RESIDUAL_TASKS]


# ================================================================ sub-milestone Robust Primitive Discovery (dev)

def rp_subunits(key):
    """Keys 0–15: atomic bench (scenes 2000+); keys 400+: downstream chain with oracle ceilings (tasks never used)."""
    from core.deep_robust_prim import subunits
    return subunits(key)


# ================================================================ sub-milestone Partial-Observation Primitive Parameter Inference (dev)

def po_subunits(key):
    """Keys 0–15: controlled occlusion and active measurements; keys 500+: downstream chain with known type (tasks never used)."""
    from core.deep_partial_exp import subunits
    return subunits(key)


# ================================================================ sub-milestone Fragment Attribution & Primitive Type Inference (dev)

def ft_subunits(key):
    """Keys 0–15: attribution, type, occlusion ownership, sectors; keys 600+: downstream chain WITHOUT type oracle."""
    from core.deep_fragments_exp import subunits
    return subunits(key)


# ================================================================ sub-milestone Boundary Ownership & Fragment Attribution (dev)

def bo_subunits(key):
    """Keys 0–15: attribution, oracle matrix, visibility, sectors; keys 700+: downstream chain and confirmation of flat_scene_k3."""
    from core.deep_ownership_exp import subunits
    return subunits(key)


# ================================================================ sub-milestone Deep Composition & Final Reconstruction with Oracle Atoms (dev)

def dc_subunits(key):
    """Keys 800–831: true atoms supplied; composition, selection, fit; existing pipeline for comparison."""
    from core.deep_compose_exp import subunits
    return subunits(key)


# ================================================================ sub-milestone End-to-End Composition with Discovered Atoms (dev)

def ee_subunits(key):
    """Keys 900–931: three chains without oracle, sources, reduction, controlled cases, class boundary."""
    from core.deep_e2e_exp import subunits
    return subunits(key)


# ================================================================ sub-milestone Latent Primitive Recovery (dev)

def lp_subunits(key):
    """Keys 1000–1031: three primitive sources in the same strong downstream, sensitivity, false positives, completeness."""
    from core.deep_lpr_exp import subunits
    return subunits(key)


# ================================================================ sub-milestone High-Precision Primitive Estimation (dev)

def hp_subunits(key):
    """Keys 1100–1131: per-primitive bench, sensitivity map, birth, chains A / B / C (standard, rich, dense), OOD."""
    from core.deep_hp_exp import subunits
    return subunits(key)


SUBUNITS = {"deepbenchhp": hp_subunits, "deephp": hp_subunits, "deeplp": lp_subunits, "deepee": ee_subunits, "deepdc": dc_subunits, "deepbo": bo_subunits, "deepft": ft_subunits, "deeppo": po_subunits, "deeprp": rp_subunits, "deepobj": obj_subunits, "deepe2e": e2e_subunits, "deepresid": resid_subunits, "deepvo": vo_subunits, "deepopt": opt_subunits, "deepdown": down_subunits, "deepprofile": profile_subunits, "deepdev": dev_subunits, "deepgen": gen_subunits, "deepcover": cover_subunits, "deepprim": prim_subunits, "deepneural": neural_subunits}


def subunit_ids(kind, key): return [sid for sid, _ in SUBUNITS[kind](key)]
def run_subunit(kind, key, sid): return dict(SUBUNITS[kind](key))[sid]()

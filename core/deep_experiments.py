"""Expériences du jalon Deep Compositional Structure Discovery — côté ÉVALUATION (seul module qui voit à la fois le
monde et le système). Les diagnostics qui utilisent la vérité ne renvoient jamais rien au système.

Unités AWS = (type, clé) découpées en sous-unités déterministes (checkpoints, reprise) :
  deepprofile : coût et devenir de la vraie structure par profondeur, toutes méthodes (micro-benchmark)
  deepdev     : exploration en dev (réglages)
  deepval     : validation (sélection de la primaire, calibration de la frontière)
  deeptest    : test confirmatoire (16 graines)
  deepsealed  : lot scellé (une seule fois)
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

QUERY_BUDGET, PRECISE_BUDGET, REPEAT_BUDGET = 16, 24, 24            # pré-enregistrés (addendum §10)


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
    library: Optional[dict] = None                # bibliothèque de fragments (apprise en dev), sérialisée
    boundary: Optional[dict] = None               # BoundaryModel sérialisé (ajusté sur validation)
    tau_none: float = 0.5
    tau_known: float = 0.2
    max_depth: int = 6

    def hash(self): return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def _mem_mb(): return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


# ================================================================ métriques (vérité : évaluation seulement)

def evaluate_program(prog: Optional[BNode], task):
    """ExactStructure / EquivalentStructure / FunctionallyEquivalent / PredictivelyAdequate, jamais fusionnées."""
    g = torch.Generator().manual_seed(11); F, I, O = full_region(4000, g), obs_region(2000, g), out_region(2000, g); lab = lambda X: task.world.label(X, 1.0)
    if prog is None: return {"exact": False, "equivalent": False, "functional": False, "adequate": False, "full": 0.0, "in_region": 0.0, "extrapolation": 0.0, "atoms": 0, "params": 0}
    a = float((predict(prog, F) == lab(F)).float().mean()); ai = float((predict(prog, I) == lab(I)).float().mean()); ao = float((predict(prog, O) == lab(O)).float().mean()); tr = task.world.truth
    return {"exact": tr is not None and skel_key(prog) == skel_key(tr), "equivalent": tr is not None and structurally_equivalent(prog, tr), "functional": a >= 0.99,
            "adequate": ai >= 0.95 and ao >= 0.95, "full": a, "in_region": ai, "extrapolation": ao, "atoms": n_atoms(prog), "params": prog.n_params()}


def truth_fate(task, candidates: Dict[str, BNode], selected: Optional[BNode], class_forms=()):
    """Devenir de la vraie structure dans une recherche. `candidates` : {clé: programme ajusté} de TOUT ce qui a été
    ajusté. Catégories : never_generated / generated_bad_fit / good_fit_misranked / equivalent_class / selected."""
    tr = task.world.truth
    if tr is None: return {"fate": "out_of_class", "rank": None}
    form = algebraic_form(tr); F = full_region(3000, torch.Generator().manual_seed(12)); lab = task.world.label(F, 1.0)
    same = [p for p in candidates.values() if algebraic_form(p) == form]
    cand_forms = {algebraic_form(s) for c in candidates.values() for s in subtrees(c, min_atoms=2)}          # sous-structures présentes dans les candidats
    sub_found = sum(1 for s in subtrees(tr, min_atoms=2) if algebraic_form(s) in cand_forms)
    out = {"generated": bool(same), "true_substructures_found": sub_found, "true_substructures": len(subtrees(tr, min_atoms=2)), "candidates": len(candidates)}
    if not same: return {**out, "fate": "never_generated"}
    acc = max(float((predict(p, F) == lab).float().mean()) for p in same); out["true_structure_best_fit"] = acc
    if selected is not None and algebraic_form(selected) == form: return {**out, "fate": "selected"}
    if acc < 0.97: return {**out, "fate": "generated_bad_fit"}
    if form in class_forms: return {**out, "fate": "equivalent_class"}
    return {**out, "fate": "good_fit_misranked"}


# ================================================================ nouveau système

def run_new(task, cfg: DeepConfig, seed=0):
    """Pipeline complet du nouveau système sur une tâche. Le bruit du monde n'est PAS fourni."""
    t0 = time.time(); g = torch.Generator().manual_seed(seed); data = FitData.build(task.X, task.y, 0.10); train, hold, itr, iho = split_holdout(data, 0.2, seed)
    lib = FragmentLibrary.from_dict(cfg.library) if cfg.library else None
    out = DeepSearch(cfg.strategy, budget=cfg.budget, max_depth=cfg.max_depth, beam=cfg.beam, seed=seed, library=lib, use_families=cfg.use_families, hierarchy=cfg.hierarchy,
                     dedup=cfg.use_classes, patience=cfg.patience).run(train)
    t_search = time.time() - t0; pool_pts = full_region(300, g); oracle = lambda x, s, gg: task.oracle(x, s, gg)
    pre = HypothesisPool(list(out.all.values()), train, pool_pts); class_before = pre.best_class().size
    policy = cfg.policy if cfg.use_classes or cfg.policy != "class_split" else "structural_ig"
    act = active_identification(out, train, oracle, pool_pts, policy=policy, budget=QUERY_BUDGET, seed=seed); hp = act.pool
    best = hp.hyps[hp.best_index()]; ident = identifiability(hp, out.stats["budget_exhausted"], any(h.depth >= cfg.max_depth for h in hp.hyps[:3]))
    # mesures de validation de frontière (ne servent PAS à l'ajustement)
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


# ================================================================ baseline gelée (jalon précédent), instrumentée sans la modifier

def run_baseline(task, seed=0):
    """Système du jalon précédent (configuration « système complet »), inchangé. Il reçoit le bruit VRAI du monde.
    L'instrumentation est une sous-classe qui ENREGISTRE les ajustements sans changer aucun calcul."""
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
    return {"method": "baseline", **ev, "decision": label, "p_ooc": None, **fate, "fit_equivalents": rec["fe"], "generated": len(cands), "fitted": r.evaluations, "distinct_structures": len(cands),
            "seconds": round(time.time() - t0, 2), "memory_mb": _mem_mb(), "queries": r.queries, "search_depth": r.depth, "program": render(best), "class_size": 1}


# ================================================================ sous-unités

PROFILE_METHODS = ("baseline",) + STRATEGIES


def _row(task, res, **extra):
    w = task.world; return {"depth": w.depth, "in_class": w.in_class, "category": w.category, "unseen": w.unseen, "regime": task.regime, "noise": task.spec.rate, "kind": task.spec.kind, "n_obs": len(task.y), **extra, **res}


def _run(task, method, cfg: DeepConfig, seed):
    if method == "baseline": return run_baseline(task, seed)
    c = DeepConfig(**{**asdict(cfg), "strategy": method}); return run_new(task, c, seed)


def profile_subunits(key):
    """Micro-benchmark : profondeurs 1–6, 2 tâches dev par profondeur (régime standard), toutes les méthodes au budget
    par défaut. Sert à connaître le coût par profondeur et la consommation de la baseline (budget apparié)."""
    key = int(key); out = []
    for d in DEPTHS:
        for j in range(2):
            for m in PROFILE_METHODS:
                out.append((f"d{d}_t{j}_{m}", (lambda d=d, j=j, m=m: [_row(make_task(900 + key * 20 + d * 2 + j, d, "standard", split="dev"), _run(make_task(900 + key * 20 + d * 2 + j, d, "standard", split="dev"), m, DeepConfig(), key), task_index=j)])))
    return out


def load_config(name: str) -> DeepConfig:
    """Configuration figée lue dans results/deep_configs/<nom>.json (créée en dev / validation, empreinte au manifeste)."""
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "deep_configs", f"{name}.json")
    return DeepConfig(**json.load(open(p))) if os.path.exists(p) else DeepConfig()


def dev_grid():
    """Grille d'exploration EN DEV (réglages) : stratégie × budget × patience. Jamais appliquée au test ni au scellé."""
    return [{"strategy": s, "budget": b, "patience": p} for s in STRATEGIES for b in (150.0,) for p in (1, 2)]


def dev_tasks(key):
    """Tâches de dev de la clé : profondeurs 1–5 en régime standard + un monde hors classe + un difficile-valide."""
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


SUBUNITS = {"deepprofile": profile_subunits, "deepdev": dev_subunits}


def subunit_ids(kind, key): return [sid for sid, _ in SUBUNITS[kind](key)]
def run_subunit(kind, key, sid): return dict(SUBUNITS[kind](key))[sid]()

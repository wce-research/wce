"""Experiments of the sub-milestone End-to-End Composition with Discovered Atoms — EVALUATION module (dev only; it
sees the truth in order to MEASURE; never the sealed set).

  chain    : the three preregistered chains (A frozen pipeline, B flat_scene_k3 + current downstream, C flat_scene_k3 +
             composition over classes + validated finalisation) and the variants of C, with no oracle at all
  source   : other primitive sources in the same engine
  reduce   : literal reduction mechanisms; wide engine and hierarchical composition
  robust   : controlled cases (true atoms, false ones added, approximate atoms, missing atoms) — oracle diagnostic
  boundary : KNOWN / UNKNOWN / NONE_OF_THE_ABOVE, in class, missing atoms, out of class

Tasks: keys 900–931 of `down_task`, depths 1 to 6 (never used; tuning: keys 990–999).
"""
from __future__ import annotations

import math
import random
import time
from typing import List, Optional, Sequence

import numpy as np
import torch

from core.boundary_program import BNode, predict
from core.deep_compose import CellSpace, bottom_up, class_scores, order_by, to_program, tree_atoms, tree_size
from core.deep_e2e import load_config, DEFAULTS, FITS, REDUCTIONS, compose, decide, discover, finalize, hierarchical, hypotheses_from, reduce, run_chain, wide_compose
from core.deep_fit import FitData
from core.deep_structure import KNOWN, NONE_OF_THE_ABOVE, UNKNOWN

DEPTHS = (1, 2, 3, 4, 5, 6)
REGIMES = ("standard", "rich")
PRIMARY = ("A", "B", "C")
C_VARIANTS = {"C": {}, "C_nofit": {"fit": "none"}, "C_forcedfit": {"fit": "forced"}, "C_errors": {"selector": "errors"}, "C_posterior": {"selector": "posterior"}, "C_k1": {"k": 1}, "C_k5": {"k": 5}}
SOURCES = ("flat_scene_k1", "flat_scene_k3", "flat_scene_k5", "fragments_k3", "robust", "frozen_atoms", "true_atoms")
ROBUST = ("true", "true_plus1", "true_plus3", "approx_010", "approx_025", "false_only", "missing1", "missing2")
OOC = ("primitive_absent", "composition_absent", "relation_absent")
GOOD = 0.99
CAUSES = ("recovered", "primitive_absent", "primitive_eliminated", "too_many_false", "class_not_generated", "class_misranked", "fit_degraded", "non_identifiable", "other")


def task_at(key, d, regime):
    from core.deep_experiments import down_task
    return down_task(key, d, regime)


def ooc_task(key, category):
    from core.deep_world import make_task
    return make_task(13000 + int(key) * 10 + OOC.index(category), None, "standard", category=category, split="dev")


def _regions(task):
    from core.deep_world import full_region, obs_region, out_region
    g = lambda s: torch.Generator().manual_seed(s); F, I, O = full_region(4000, g(11)), obs_region(2000, g(11)), out_region(2000, g(13)); return (F, task.world.label(F, 1.0)), (I, task.world.label(I, 1.0)), (O, task.world.label(O, 1.0))


def _acc(prog: Optional[BNode], regions) -> dict:
    if prog is None: return {"accord": 0.0, "accord_in": 0.0, "accord_out": 0.0, "functional": False, "functional_097": False}
    (F, lF), (I, lI), (O, lO) = regions; a = float((predict(prog, F) == lF).float().mean()); return {"accord": a, "accord_in": float((predict(prog, I) == lI).float().mean()), "accord_out": float((predict(prog, O) == lO).float().mean()), "functional": a >= GOOD, "functional_097": a >= 0.97}


def _truth_obs(task): return task.world.label(task.X, 1.0)


def _structure(prog: Optional[BNode], task) -> dict:
    """Exact tree (up to associativity / commutativity); equivalent class: same values as the truth on all
    the observations (true labels, without noise)."""
    from core.deep_structure import algebraic_form, n_atoms
    if prog is None or task.world.truth is None: return {"exact": False, "class_equal": False, "obs_agreement": 0.0, "size": 0}
    ag = float((predict(prog, task.X.float()) == _truth_obs(task)).float().mean()); return {"exact": algebraic_form(prog) == algebraic_form(task.world.truth), "class_equal": ag >= 1.0, "obs_agreement": ag, "size": n_atoms(prog)}


def atom_recall(programs: Sequence[BNode], task) -> Optional[float]:
    """Share of the true atoms that a given primitive reproduces (functional agreement ≥ 0.95 on the observed region)."""
    from core.deep_partial_exp import _probe, functional
    from core.deep_reachability import leaves
    if task.world.truth is None: return None
    P = _probe(); tl = leaves(task.world.truth); return sum(any(functional(a, p, P) >= 0.95 for p in programs) for a in tl) / len(tl)


def _false_used(prog: Optional[BNode], task) -> int:
    from core.deep_partial_exp import _probe, functional
    from core.deep_reachability import leaves
    if prog is None or task.world.truth is None: return 0
    P = _probe(); tl = leaves(task.world.truth); return sum(not any(functional(a, l, P) >= 0.95 for a in tl) for l in leaves(prog))


# ================================================================ the three chains

def chain_ab(task, arm: str, seed=0) -> dict:
    """A: frozen pipeline (no literal supplied). B: literals of flat_scene_k3, current downstream. Maximum depth 6 for
    both: no information on the true size."""
    from core.deep_fragments import flat_types, literals
    from core.deep_partial import observe
    from core.deep_reachability import leaves
    from core.deep_search import DeepSearch
    t0 = time.time(); regions = _regions(task); data = FitData.build(task.X, task.y, 0.10); kw = {}
    if arm == "B": kw = {"fact_literals": literals(flat_types(observe(task.X, task.y), 3, seed)), "fact_residual": False}
    out = DeepSearch("factorized", budget=150.0, seed=seed, max_depth=6, validated=None, **kw).run(data); best = out.best().program; pool = [h.program for h in out.all.values()]; acc = [_acc(p, regions)["accord"] for p in pool]
    lits = kw.get("fact_literals") or [l for p in pool for l in leaves(p)]
    return {"experiment": "chain", "arm": arm, **_acc(best, regions), **_structure(best, task), "candidate_099": max(acc, default=0.0) >= GOOD, "candidate_097": max(acc, default=0.0) >= 0.97, "candidate_best": max(acc, default=0.0), "pool": len(pool), "literals": len(lits),
            "atom_recall": atom_recall(lits, task), "false_used": _false_used(best, task), "fit_equivalents": out.stats["fit_equivalents"], "seconds": round(time.time() - t0, 2)}


def _attribute(task, e2e, regions, row) -> str:
    """Main cause of a failure of chain C (diagnostic: uses the truth)."""
    if row["functional"]: return "recovered"
    if row["atom_recall_literals"] is not None and row["atom_recall_literals"] < 1.0: return "primitive_absent"
    if row["atom_recall_kept"] is not None and row["atom_recall_kept"] < 1.0: return "primitive_eliminated"
    if not row["generated_099"]: return "non_identifiable" if row["unseen_mass"] > 0.05 else "class_not_generated"
    if row["prefit_functional"] and not row["functional"]: return "fit_degraded"
    if row["false_used"] > 0 and row["selected_errors"] is not None and row["best_good_errors"] is not None and row["selected_errors"] < row["best_good_errors"]: return "too_many_false"
    return "class_misranked"


def chain_c(task, variant="C", seed=0, discovered=None) -> dict:
    """Chain C and its variants: real discovery → hypotheses → reduction → composition over classes → finalisation."""
    t0 = time.time(); regions = _regions(task); cfg = {**load_config(), **C_VARIANTS.get(variant, {})}; e = run_chain(task.X, task.y, seed, cfg, discovered); row = {"experiment": "chain", "arm": variant, **_acc(e.program, regions), **_structure(e.program, task)}
    atoms = [h.program for h in e.kept]; R = e.result; units = getattr(R, "units", atoms); pre = to_program(R.selected.tree, units) if R and R.selected else None; row.update({"prefit_accord": _acc(pre, regions)["accord"], "prefit_functional": _acc(pre, regions)["functional"], "literals": e.stats["literals"], "hypotheses": e.stats["hypotheses"], "kept": e.stats["kept"],
                "classes": e.stats.get("classes"), "considered": e.stats.get("considered"), "label": e.label, "gap": e.decision["gap"], "rivals": e.decision["rivals"], "fit_committed": bool(e.finalization.committed) if e.finalization else False,
                "atom_recall_literals": atom_recall([h.program for h in e.hypotheses] + [a for h in e.hypotheses for a in h.alternatives], task), "atom_recall_kept": atom_recall(atoms, task), "false_kept": sum(1 for a in atoms if (atom_recall([a], task) or 0) == 0) if task.world.truth is not None else None,
                "false_used": _false_used(e.program, task), "n_alternatives": sum(h.n_alternatives for h in e.hypotheses), "seconds": round(time.time() - t0, 2), "discover_seconds": e.stats["discover_seconds"]})
    # candidate recall: good class generated / in the top-k / selected
    gen99 = gen97 = False; top5 = top20 = False; best_good = None; unseen = 0.0
    if atoms:
        T = R.table; sp = T.space; masks, sizes, idx = T.all(); sc = class_scores(sp, masks, sizes); o = order_by(sc, cfg["selector"]); F, lF = regions[0]; codes = sp.codes_of(F); unseen = sp.unseen_mass(F); lab = lF.numpy().astype(bool)
        take = o[:400]; accs = np.array([float((sp.value_on_codes(T.tree(int(sizes[i]), int(idx[i])), codes) == lab).mean()) for i in take]); gen99 = bool((accs >= GOOD).any()); gen97 = bool((accs >= 0.97).any()); top5 = bool((accs[:5] >= GOOD).any()); top20 = bool((accs[:20] >= GOOD).any())
        if gen99: best_good = float(sc["errors"][take[int(np.argmax(accs >= GOOD))]])
        row["oracle_top20_accord"] = float(accs[:20].max()) if len(accs) else 0.0
    row.update({"generated_099": gen99, "generated_097": gen97, "top5_099": top5, "top20_099": top20, "candidate_099": top20, "candidate_097": bool(atoms) and row.get("oracle_top20_accord", 0.0) >= 0.97, "candidate_best": row.get("oracle_top20_accord", 0.0), "unseen_mass": unseen,
                "selected_errors": R.selected.errors if R and R.selected else None, "best_good_errors": best_good})
    row["cause"] = _attribute(task, e, regions, row) if task.world.truth is not None else None; return row


# ================================================================ primitive sources

def source_literals(task, source: str, seed=0):
    from core.deep_fragments import flat_types, infer_types
    from core.deep_partial import observe
    from core.deep_reachability import leaves
    X, y = task.X, task.y
    if source.startswith("flat_scene_k"): return discover(X, y, int(source[-1]), seed)
    if source == "fragments_k3": return [(h.program, float(h.score), f"fragments:{h.type}") for h in infer_types(observe(X, y), 3, seed=seed).selected]
    if source == "robust":
        from core.deep_primitives2 import discover_robust, literals_of, load_config
        from core.deep_robust_prim import proposer, validator
        c = load_config(); d = discover_robust(X, y, proposer("union"), validator("full", c), seed=seed, joint=True, prune_z=c["prune_z"], split=c["split"]); L = literals_of(d, alternatives=2); return [(p, float(len(L) - i), "robust") for i, p in enumerate(L)]
    if source == "frozen_atoms":                                                   # atoms of the frozen pipeline: leaves of its hypotheses, weighted by the evidence of the hypothesis
        from core.deep_search import DeepSearch
        out = DeepSearch("factorized", budget=150.0, seed=seed, max_depth=6).run(FitData.build(X, y, 0.10)); hs = sorted(out.all.values(), key=lambda h: -h.evidence); seen = set(); L = []
        for r, h in enumerate(hs):
            for l in leaves(h.program):
                if repr(l) not in seen: seen.add(repr(l)); L.append((l, float(len(hs) - r), "frozen"))
        return L
    if source == "true_atoms": return [(a, 1.0, "oracle") for a in leaves(task.world.truth)]
    raise KeyError(source)


def source_row(task, source: str, seed=0) -> dict:
    t0 = time.time(); disc = source_literals(task, source, seed); r = chain_c(task, "C", seed, disc); return {**r, "experiment": "source", "arm": source, "source_seconds": round(time.time() - t0, 2)}


# ================================================================ reduction, scaling, hierarchical composition

def reduce_rows(task, seed=0) -> List[dict]:
    """Same literals (flat_scene_k3), different reduction mechanisms; then wide engine and hierarchical
    composition on more atoms. No fit: only the composition is compared."""
    regions = _regions(task); disc = discover(task.X, task.y, 3, seed); H = hypotheses_from(disc, task.X); out = []; base = {"experiment": "reduce", "literals": len(disc), "hypotheses": len(H), "atom_recall_literals": atom_recall([h.program for h in H], task)}
    for m in REDUCTIONS:
        t0 = time.time(); kept, info = reduce(H, task.y, m); atoms = [h.program for h in kept]; R = compose(atoms, task.X, task.y, "bic") if atoms else None; prog = to_program(R.selected.tree, R.units) if R else None
        out.append({**base, "arm": m, "kept": len(kept), "atom_recall_kept": atom_recall(atoms, task), **_acc(prog, regions), "classes": R.stats.get("classes") if R else 0, "seconds": round(time.time() - t0, 3)})
    order = [h.program for h in sorted(H, key=lambda h: -h.confidence)]
    for name, n, fn in (("wide_6", 6, None), ("wide_9", 9, None), ("wide_12", 12, None), ("wide_all", len(order), None), ("hier_12", 12, "h"), ("hier_all", len(order), "h")):
        atoms = order[:n]; t0 = time.time()
        if not atoms: continue
        R = hierarchical(atoms, task.X, task.y) if fn else wide_compose(atoms, task.X, task.y); prog = to_program(R.selected.tree, atoms)
        out.append({**base, "arm": name, "kept": len(atoms), "atom_recall_kept": atom_recall(atoms, task), **_acc(prog, regions), "classes": R.stats.get("classes"), "considered": R.stats.get("considered"), "cells": R.stats.get("cells"), "size": tree_size(R.selected.tree), "seconds": round(time.time() - t0, 3)})
    return out


# ================================================================ controlled cases: false atoms, approximate atoms, missing atoms

def robust_atoms(task, condition: str, seed=0):
    """Starting atoms of a controlled case (DIAGNOSTIC: starts from the true atoms)."""
    from core.deep_reachability import leaves
    from core.deep_world import ATOM_TYPES, atom
    tl = list(leaves(task.world.truth)); rng = random.Random(seed * 7919 + ROBUST.index(condition)); fake = lambda k: [atom(rng, rng.choice(ATOM_TYPES)) for _ in range(k)]
    if condition == "true": return tl
    if condition == "true_plus1": return tl + fake(1)
    if condition == "true_plus3": return tl + fake(3)
    if condition.startswith("approx_"): s = int(condition[-3:]) / 100.0; return [a.skeleton().with_params([v + rng.gauss(0.0, s) for v in a.param_vector()]) for a in tl]
    if condition == "false_only": return fake(len(tl))
    if condition == "missing1": return tl[1:] if len(tl) > 1 else []
    if condition == "missing2": return tl[2:] if len(tl) > 2 else []
    raise KeyError(condition)


def robust_row(task, condition: str, seed=0) -> dict:
    from core.deep_reachability import leaves
    atoms = robust_atoms(task, condition, seed); n_true = len(leaves(task.world.truth)); r = chain_c(task, "C_nofit", seed, [(a, 1.0, condition) for a in atoms]) if atoms else {**_acc(None, _regions(task)), "label": UNKNOWN, "kept": 0, "false_used": 0, "gap": None, "cause": None, "false_kept": 0}
    return {"experiment": "robust", "arm": condition, "given": len(atoms), "n_true": n_true, **{k: r.get(k) for k in ("accord", "accord_out", "functional", "functional_097", "label", "gap", "kept", "false_kept", "false_used", "cause", "atom_recall_kept")}, "applicable": bool(atoms) or condition.startswith("missing")}


# ================================================================ class boundary

def boundary_row(task, kind: str, seed=0) -> dict:
    """Decision of chain C on an out-of-class task (no representable structure is the truth)."""
    e = run_chain(task.X, task.y, seed); regions = _regions(task); a = _acc(e.program, regions)
    return {"experiment": "boundary", "arm": kind, "label": e.label, "gap": e.decision["gap"], "knn_error_rate": e.decision["knn_error_rate"], "error_rate": e.decision["error_rate"], "rivals": e.decision["rivals"], "accord": a["accord"], "kept": e.stats["kept"]}


# ================================================================ AWS sub-units

def subunits(key):
    from core.deep_experiments import _row
    k = int(key); out = []
    for d in DEPTHS:
        for reg in REGIMES:
            T = lambda d=d, reg=reg: task_at(key, d, reg); tag = {"task_id": f"{key}_{d}_{reg}"}
            for arm in ("A", "B"): out.append((f"chain_d{d}_{reg}_{arm}", (lambda T=T, tag=tag, arm=arm: [_row(T(), {**tag, **chain_ab(T(), arm, k)})])))
            out.append((f"chain_d{d}_{reg}_C", (lambda T=T, tag=tag: [_row(T(), {**tag, **chain_c(T(), v, k)}) for v in C_VARIANTS])))
            out.append((f"source_d{d}_{reg}", (lambda T=T, tag=tag: [_row(T(), {**tag, **source_row(T(), s, k)}) for s in SOURCES])))
            out.append((f"reduce_d{d}_{reg}", (lambda T=T, tag=tag: [_row(T(), {**tag, **r}) for r in reduce_rows(T(), k)])))
            out.append((f"robust_d{d}_{reg}", (lambda T=T, tag=tag: [_row(T(), {**tag, **robust_row(T(), c, k)}) for c in ROBUST])))
    for cat in OOC: out.append((f"boundary_{cat}", (lambda cat=cat: [{"task_id": f"{key}_{cat}", "depth": 0, "regime": "standard", **boundary_row(ooc_task(key, cat), cat, k)}])))
    return out

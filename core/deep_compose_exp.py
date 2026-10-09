"""Expériences du sous-jalon Deep Composition & Final Reconstruction with Oracle Atoms — module d'ÉVALUATION (dev
uniquement ; il voit la vérité ; jamais le lot scellé).

Les VRAIS atomes (types et paramètres) sont fournis : l'amont est retiré de l'équation. Restent la composition,
l'ajustement du composite, l'équivalence, le score, la simplification et la sélection.

  compose : espace de composition, programmation dynamique sur classes, sélecteurs, faisceaux, budgets, rencontre au
            milieu, ajustements A / B / C / D — par tâche
  ref     : pipeline existant avec les mêmes atomes (budgets, garde-fou, sans ajustement), et sa trace par étape

Tâches : clés 800–831 de `down_task`, profondeurs 1 à 6 (jamais utilisées ; mise au point : clé 899).
"""
from __future__ import annotations

import time
from typing import List

import numpy as np
import torch

from core.boundary_program import BNode, predict
from core.deep_compose import (SELECTORS, CellSpace, Table, bottom_up, class_scores, from_program, order_by, rank_of, scaling, select, to_program, top_down, tree_form, tree_shape, tree_size)
from core.deep_fit import DT, FitData, fit_batch, validated_fit

DEPTHS = (1, 2, 3, 4, 5, 6)
REGIMES = ("standard", "rich")
BEAMS = (10, 100, 1000)
BUDGETS = (100, 1000, 10000, 100000)
REF_ARMS = (("ref_b50", 50.0, None), ("ref_b150", 150.0, None), ("ref_b450", 450.0, None), ("ref_never_b150", 150.0, "never"), ("ref_validated_b150", 150.0, "profiled"))
REF_BIG = ("ref_b1500", 1500.0, None)             # 10 × la référence, sur un sous-ensemble (clés multiples de 8)
MAIN, REFERENCE = "dp_errors", "ref_b150"
GOOD = 0.99


def task_at(key, d, regime):
    from core.deep_experiments import down_task
    return down_task(key, d, regime)


def _regions(task):
    from core.deep_world import full_region, obs_region, out_region
    g = lambda s: torch.Generator().manual_seed(s); F, I, O = full_region(4000, g(11)), obs_region(2000, g(11)), out_region(2000, g(13)); return (F, task.world.label(F, 1.0)), (I, task.world.label(I, 1.0)), (O, task.world.label(O, 1.0))


def _accords(prog: BNode, regions) -> dict:
    (F, lF), (I, lI), (O, lO) = regions; return {"accord": float((predict(prog, F) == lF).float().mean()), "accord_in": float((predict(prog, I) == lI).float().mean()), "accord_out": float((predict(prog, O) == lO).float().mean())}


def _outcome(tree, sp: CellSpace, atoms, tt, tm, regions, prefix="") -> dict:
    """exact : même arbre à associativité / commutativité près ; class_equal : même fonction sur les observations ;
    functional : accord ≥ 0,99 sur toute la région ; compact : plus petit que la vérité ET fonctionnel."""
    a = _accords(to_program(tree, atoms), regions); m = sp.mask_of(tree); s = tree_size(tree)
    return {f"{prefix}{k}": v for k, v in {"exact": tree_form(tree) == tree_form(tt), "class_equal": m == tm, "functional": a["accord"] >= GOOD, "functional_097": a["accord"] >= 0.97, **a, "size": s, "errors": float(sp.errors([m])[0]),
                                           "compact": s < tree_size(tt) and a["accord"] >= GOOD, "obs_agreement": 1.0 - float(sum(sp.count[c] for c in range(sp.K) if ((m ^ tm) >> c) & 1)) / sp.N}.items()}


def _fit_variants(tree, atoms, task, regions, seed=0) -> dict:
    """A aucun ajustement (vrais paramètres) ; B chaque atome ré-ajusté SEUL (contraste local), puis recomposé ;
    C ajustement du composite entier (optimiseur actuel) ; D le même, validé (commit seulement si le critère observable
    progresse, sinon retour à l'état initial)."""
    from core.deep_primitives import close_pairs, contrast_fit
    prog = to_program(tree, atoms); data = FitData.build(task.X, task.y, 0.10); sk = prog.skeleton(); init = torch.tensor([prog.param_vector()], dtype=DT); out = {"A": _accords(prog, regions)["accord"]}
    pairs = close_pairs(task.X, task.y); refit = [a.skeleton().with_params(contrast_fit(a.skeleton(), pairs, torch.tensor([a.param_vector()], dtype=DT), steps=40)[0][0].tolist()) for a in atoms]; out["B"] = _accords(to_program(tree, refit), regions)["accord"]
    th, _ = fit_batch(sk, data, init, steps=100); out["C"] = _accords(sk.with_params(th[0].tolist()), regions)["accord"]; row, tx = validated_fit(sk, data, init[0], steps=100); out["D"] = _accords(sk.with_params(row.tolist()), regions)["accord"]; out["D_committed"] = bool(tx.accepted)
    pe = lambda p: float((predict(p, task.X.float()) != task.y).sum()); out["errors_A"] = pe(prog); out["errors_C"] = pe(sk.with_params(th[0].tolist())); return out


def compose_row(task, seed=0) -> dict:
    from core.deep_reachability import leaves
    from core.deep_world import full_region, out_region
    t0 = time.time(); tr = task.world.truth; atoms = leaves(tr); d = len(atoms); sp = CellSpace(atoms, task.X, task.y); tt = from_program(tr, atoms); tm = sp.mask_of(tt); regions = _regions(task); te = float(sp.errors([tm])[0])
    row = {"experiment": "compose", "n_atoms": d, **{f"truth_{k}": v for k, v in tree_shape(tt).items()}, "cells": sp.K, "floor_errors": sp.floor, "truth_errors": te, "noise_rate": te / sp.N, "unseen_mass_full": sp.unseen_mass(regions[0][0]), "unseen_mass_out": sp.unseen_mass(regions[2][0]),
           "truth_is_majority": tm == sp.target}
    # ---- loi d'échelle et programmation dynamique exhaustive
    row["scaling"] = scaling(sp, d); T = bottom_up(sp, d); masks, sizes, idx = T.all(); sc = class_scores(sp, masks, sizes); row.update({"dp_classes": T.classes, "dp_considered": T.stats["considered"], "dp_seconds": T.stats["seconds"], "truth_generated": T.find(tm) is not None})
    agree = 1.0 - (sp.bits(masks ^ np.uint64(tm)) @ sp.count) / sp.N; row["classes_better_or_equal"] = int((sc["errors"] <= te + 1e-9).sum()); row["classes_strictly_better"] = int((sc["errors"] < te - 1e-9).sum()); row["classes_quasi_equivalent"] = int((agree >= 0.97).sum())
    for sel in SELECTORS:
        o = order_by(sc, sel); r = rank_of(T, tm, sel); best = T.tree(int(sizes[o[0]]), int(idx[o[0]])); row[f"rank_{sel}"] = r
        for k in (1, 5, 20): row[f"top{k}_{sel}"] = r is not None and r <= k
        row.update(_outcome(best, sp, atoms, tt, tm, regions, f"dp_{sel}_"))
    main = select(T, "errors").selected.tree
    # ---- une feuille de plus (la taille vraie n'est pas donnée au système)
    if d < 6: T1 = bottom_up(sp, d + 1, budget=3_000_000); row.update(_outcome(select(T1, "errors").selected.tree, sp, atoms, tt, tm, regions, "dp_errors_plus1_")); row.update(_outcome(select(T1, "bic").selected.tree, sp, atoms, tt, tm, regions, "dp_bic_plus1_"))
    # ---- signature étendue : des classes fusionnées sur les observations divergent-elles ailleurs ?
    P = torch.cat([task.X.float(), regions[0][0], regions[2][0]]); spx = CellSpace(atoms, P, torch.zeros(len(P), dtype=torch.long)); row["cells_extended"] = spx.K
    if d <= 5: Tx = bottom_up(spx, d, budget=3_000_000); row["classes_extended"] = Tx.classes; row["classes_merged_by_observation"] = Tx.classes - T.classes
    # ---- faisceau sur les classes, budgets
    for b in BEAMS: Tb = bottom_up(sp, d, beam=b); rb = select(Tb, "errors").selected; row[f"beam{b}_truth"] = Tb.find(tm) is not None; row[f"beam{b}_accord"] = _accords(to_program(rb.tree, atoms), regions)["accord"]; row[f"beam{b}_considered"] = Tb.stats["considered"]; row[f"beam{b}_class_equal"] = rb.mask == tm
    for bud in BUDGETS: Tq = bottom_up(sp, d, budget=bud); rq = select(Tq, "errors").selected; row[f"budget{bud}_truth"] = Tq.find(tm) is not None; row[f"budget{bud}_accord"] = _accords(to_program(rq.tree, atoms), regions)["accord"]; row[f"budget{bud}_class_equal"] = rq.mask == tm
    # ---- rencontre au milieu (table de taille ≤ 2 ou ≤ 3, trous contraints)
    for h in (2, 3):
        Th = bottom_up(sp, min(h, d)); R = top_down(sp, Th, d, beam=6, budget=4000, half=min(h, d)); row.update(_outcome(R.selected.tree, sp, atoms, tt, tm, regions, f"mitm{h}_")); row[f"mitm{h}_joins"] = R.stats["joins"]; row[f"mitm{h}_table"] = R.stats["table_classes"]; row[f"mitm{h}_seconds"] = R.stats["seconds"]
        row[f"mitm{h}_matches_exhaustive"] = R.selected.errors <= float(sc["errors"].min()) + 1e-9
    # ---- ajustements, sur la structure retenue et sur la vraie structure
    row["fit_selected"] = _fit_variants(main, atoms, task, regions, seed); row["fit_truth"] = _fit_variants(tt, atoms, task, regions, seed); row["seconds"] = round(time.time() - t0, 2); return row


def ref_row(task, arm, seed=0) -> dict:
    """Pipeline existant, vrais atomes fournis comme littéraux. Seuls le budget et la politique d'ajustement changent."""
    from core.deep_reachability import leaves
    from core.deep_search import DeepSearch
    from core.deep_structure import algebraic_form, n_atoms, structurally_equivalent
    name, budget, pol = next(a for a in REF_ARMS + (REF_BIG,) if a[0] == arm); t0 = time.time(); tr = task.world.truth; atoms = leaves(tr); d = len(atoms); regions = _regions(task); data = FitData.build(task.X, task.y, 0.10); sp = CellSpace(atoms, task.X, task.y); tt = from_program(tr, atoms)
    out = DeepSearch("factorized", budget=budget, seed=seed, max_depth=min(6, d + 1), fact_literals=atoms, fact_residual=False, validated=pol).run(data); best = out.best().program; a = _accords(best, regions); allp = [h.program for h in out.all.values()]; acc = [_accords(p, regions)["accord"] for p in allp]; tf = algebraic_form(tr)
    return {"experiment": "ref", "arm": name, "n_atoms": d, **{f"truth_{k}": v for k, v in tree_shape(tt).items()}, **a, "functional": a["accord"] >= GOOD, "functional_097": a["accord"] >= 0.97, "exact": algebraic_form(best) == tf, "equivalent": structurally_equivalent(best, tr), "size": n_atoms(best),
            "candidates": len(allp), "truth_form_in_pool": any(algebraic_form(p) == tf for p in allp), "candidate_best": max(acc, default=0.0), "good_in_pool": max(acc, default=0.0) >= GOOD, "good_found_then_lost": max(acc, default=0.0) >= GOOD and a["accord"] < GOOD,
            "fit_equivalents": out.stats["fit_equivalents"], "seconds": round(time.time() - t0, 2)}


def trace_row(task, seed=0) -> dict:
    """Trace par étape du pipeline existant (instrumentation du sous-jalon Downstream Selection Loss, inchangée)."""
    from core.deep_downstream import trace_task
    from core.deep_reachability import leaves
    t = trace_task(task, seed, fitter_audit=False); keep = {k: v for k, v in t.items() if not isinstance(v, (list, dict)) or k in ("trajectory", "summary", "counterfactual", "truth_fit", "truth_fate", "ranks")}
    atoms = leaves(task.world.truth); return {"experiment": "trace", "n_atoms": len(atoms), **{f"truth_{k}": v for k, v in tree_shape(from_program(task.world.truth, atoms)).items()}, **keep}


def subunits(key):
    from core.deep_experiments import _row
    k = int(key); out = []
    for d in DEPTHS:
        for reg in REGIMES:
            tag = {"task_id": f"{key}_{d}_{reg}"}
            out.append((f"compose_d{d}_{reg}", (lambda d=d, reg=reg, tag=tag: [_row(task_at(key, d, reg), {**tag, **compose_row(task_at(key, d, reg), k)})])))
            for arm in [a[0] for a in REF_ARMS] + ([REF_BIG[0]] if k % 8 == 0 else []): out.append((f"ref_d{d}_{reg}_{arm}", (lambda d=d, reg=reg, arm=arm, tag=tag: [_row(task_at(key, d, reg), {**tag, **ref_row(task_at(key, d, reg), arm, k)})])))
            if k % 2 == 0 and d >= 2: out.append((f"trace_d{d}_{reg}", (lambda d=d, reg=reg, tag=tag: [_row(task_at(key, d, reg), {**tag, **trace_row(task_at(key, d, reg), k)})])))
    return out

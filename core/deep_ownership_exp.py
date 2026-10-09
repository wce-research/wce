"""Expériences du sous-jalon Boundary Ownership & Fragment Attribution — module d'ÉVALUATION (dev uniquement ; il voit
la vérité, rien n'en revient au système ; jamais le lot scellé).

  attr    : attribution par méthode (pureté, rappel, morcellement, appartenance top-1 / top-3, calibration)
  oracle  : matrice appartenance vraie / estimée × type vrai / inféré
  vis     : score de visibilité contrefactuelle contre contraste ; détection du masque
  sector  : les secteurs échouent-ils par attribution ou par paramétrage ?
  chain   : appartenance → primitives → structure candidate → fonction, et CONFIRMATION de `flat_scene_k3`

Scènes : banc de masquage, indices 200+ (jamais utilisés) ; mise au point : sous-plage + 350 000.
Tâches de composition : clés 700–731 de `down_task` (jamais utilisées ; mise au point : clé 799).
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence

import torch

from core.boundary_program import BNode, predict
from core.deep_fit import DT, FitData
from core.deep_fragments import flat_types, group_points, literals, type_of
from core.deep_fragments_exp import _equivalent, pair_scores, restrict, true_sources
from core.deep_ownership import (OwnershipSolution, filter_by_ownership, find_occlusions, load_config, owned, ownership_from_program, ownership_hypotheses, primitives_from, solve, visible_support, _same)
from core.deep_partial import global_score, infer, observe, rendering_errors
from core.deep_partial_exp import KINDS, MASKS, MaskedScene, _probe, functional, masked_scene, param_errors, recovered
from core.deep_primitive_eval import component_match, components, primitive_metrics
from core.deep_primitives import atom_name

SCENE_OFFSET = 200
METHODS = ("tangent", "shared", "consensus", "solver", "solver_no_chance", "solver_local", "solver_no_smooth", "solver_absent", "solver_absent_naive", "composition")
# `solver` : configuration retenue en mise au point (terme « absent » à zéro). `solver_absent` : preuve négative limitée à la
# frontière attendue visible (masques explicites) ; `solver_absent_naive` : toute frontière manquante compte.
# `solver_no_chance` : sans le coût de calibration par le hasard (meilleur en mise au point, mais attribue 96 % des milieux
# sur des étiquettes aléatoires).
SOLVER_CFG = {"solver": {}, "solver_no_chance": {"chance": 0.0}, "solver_local": {"long_range": False}, "solver_no_smooth": {"beta": 0.0}, "solver_absent": {"gamma": 1.0, "occlusion": True}, "solver_absent_naive": {"gamma": 1.0, "occlusion": False}}
MAIN = "solver"


def scene_at(index, kind, mask, noise, tune=False): return masked_scene((0 if tune else SCENE_OFFSET) + int(index), kind, mask, noise, tune=tune)


def _atoms(sc: MaskedScene): return [sc.target] + ([sc.occluder] if sc.occluder is not None else [])
def _comp(o): return ("circle", *o.params) if o.kind == "circle" else ("line", *o.params)


# ================================================================ attribution : pureté, morcellement, appartenance

def fragmentation(lab: torch.Tensor, src: torch.Tensor, M: torch.Tensor) -> dict:
    """Pureté moyenne des groupes, nombre de morceaux par vraie frontière, longueur moyenne des groupes."""
    groups = [g for g in torch.unique(lab).tolist() if g >= 0 and int((lab == g).sum()) >= 2]; pur = []; ln = []
    for g in groups:
        m = lab == g; s = src[m]; s = s[s >= 0]; pur.append(float(torch.bincount(s).max()) / len(s) if len(s) else None); ln.append(float(torch.cdist(M[m], M[m]).max()))
    pieces = [len({int(l) for l in lab[src == c].tolist() if l >= 0 and int(((lab == l) & (src == c)).sum()) >= 2}) for c in torch.unique(src[src >= 0]).tolist() if int((src == c).sum()) >= 4]
    pv = [p for p in pur if p is not None]; return {"purity": sum(pv) / len(pv) if pv else None, "pieces_per_boundary": sum(pieces) / len(pieces) if pieces else None, "mean_length": sum(ln) / len(ln) if ln else None, "groups": len(groups)}


def compose_small(prims: Sequence[BNode], X, y, top=4):
    """Petite hypothèse de composition, SANS vérité : la meilleure primitive seule, ou la meilleure combinaison ET / OU /
    OU exclusif de deux primitives, selon l'erreur du rendu sur les observations."""
    P = list(prims)[:top]; best, be = None, len(y) + 1
    for i, a in enumerate(P):
        for cand in [a] + [BNode(k, children=(a, b)) for b in P[i + 1:] for k in ("and", "or", "xor")] + [BNode("and", children=(a, BNode("not", children=(b,)))) for b in P[i + 1:]] + [BNode("and", children=(b, BNode("not", children=(a,)))) for b in P[i + 1:]]:
            e = min(rendering_errors(cand, X, y), len(y) - rendering_errors(cand, X, y))
            if e < be: best, be = cand, e
    return best, be


def ownership_accuracy(sol: OwnershipSolution, src: torch.Tensor, comps: Sequence[tuple]) -> dict:
    """Appartenance par milieu de source connue : la courbe attribuée (top-1) ou l'une des trois plus probables (top-3)
    coïncide-t-elle avec la vraie frontière du milieu ? Avec histogramme de calibration et score de Brier."""
    m = src >= 0; n = int(m.sum()); K = len(sol.owners)
    if n == 0: return {"top1": None, "top3": None, "brier": None, "cal": None}
    ok = torch.zeros(len(src), K + 1, dtype=torch.bool)
    for j, o in enumerate(sol.owners):
        for c in torch.unique(src[m]).tolist():
            if component_match(_comp(o), comps[c]): ok[src == c, j] = True
    top = sol.top_k(3); hit1 = ok.gather(1, top[:, :1])[:, 0][m]; hit3 = ok.gather(1, top).any(1)[m]; conf = sol.posterior.gather(1, top[:, :1])[:, 0][m]; cal = [[0, 0.0, 0.0] for _ in range(10)]
    for c, h in zip(conf.tolist(), hit1.tolist()): b = min(9, int(c * 10)); cal[b][0] += 1; cal[b][1] += float(h); cal[b][2] += c
    return {"top1": float(hit1.float().mean()), "top3": float(hit3.float().mean()), "brier": float(((conf - hit1.to(DT)) ** 2).mean()), "cal": cal}


def attr_rows(sc: MaskedScene, seed=0) -> List[dict]:
    ob = observe(sc.X, sc.y); atoms = _atoms(sc); comps = [c for a in atoms for c in components(a)]; src, atom = true_sources(ob.points, atoms); out = []; base = {"kind": sc.kind, "mask": sc.mask, "noise": sc.noise}; main = None
    for meth in METHODS:
        t0 = time.time(); extra = {}
        if meth in SOLVER_CFG:
            sol = solve(ob, SOLVER_CFG[meth], seed); lab = sol.labels; extra = {**ownership_accuracy(sol, src, comps), "owners": len(sol.owners), "assignments_tested": sol.stats["assignments_tested"], "energy": {k: v for k, v in sol.terms.items()}}
            if meth == MAIN:
                main = sol; H = ownership_hypotheses(sol, ob); extra["fragments"] = len(H); extra["ambiguous_fragments"] = sum(h.ambiguous for h in H) / max(1, len(H)); occ = sol.occlusions; tc = components(sc.target); oc = components(sc.occluder) if sc.occluder is not None else []
                extra["occlusion_found"] = bool(occ); extra["occlusion_correct"] = any(any(component_match(_comp(sol.owners[o.occluded]), c) for c in tc) and any(component_match(_comp(sol.owners[o.occluder]), c) for c in oc) for o in occ)
        elif meth == "composition":                                               # la composition ré-attribue : feuilles du petit composite, visibilité prise en compte
            sel, _ = primitives_from(main, ob, 1); prog, _ = compose_small([h.program for h in sel], sc.X, sc.y) if sel else (None, None)
            lab = ownership_from_program(prog, ob)[0] if prog is not None else torch.full((len(ob.points),), -1, dtype=torch.long)
        else: lab = group_points(ob, meth, seed=seed)
        out.append({"method": meth, **base, **pair_scores(lab, src), **fragmentation(lab, src, ob.points), **extra, "seconds": round(time.time() - t0, 3)})
    return out


# ================================================================ matrice oracle

def oracle_row(sc: MaskedScene, seed=0) -> dict:
    """Récupération exacte de la cible en première hypothèse (n premières pour le type inféré, n = primitives vraies).
      TT appartenance vraie + type vrai            TI appartenance vraie + type inféré
      ET appartenance estimée + type vrai          EI appartenance estimée + type inféré
      NT / NI : aucune attribution (inférence sur la scène entière), type vrai / inféré
    L'appartenance estimée sert de VALIDATION : seules les hypothèses dont toutes les frontières sont des courbes
    possédées sont gardées. `OWN` : primitives construites directement depuis les courbes possédées."""
    T = sc.target; ob = observe(sc.X, sc.y); atoms = _atoms(sc); _, atom = true_sources(ob.points, atoms); tt = type_of(T); n = len(atoms); sk = T.skeleton(); obt = restrict(ob, atom == 0); sol = solve(ob, None, seed)
    hit = lambda hs, k=1: any(type_of(h) == tt and recovered(T, h) for h in hs[:k])
    nt = [h.program for h in infer(sk, ob, 5, seed=seed, regions=False)]; ni = [h.program for h in flat_types(ob, 3, seed)]; enough = len(obt.points) >= 4
    tt_h = [h.program for h in infer(sk, obt, 3, seed=seed, regions=False)] if enough else []; ti_h = [h.program for h in flat_types(obt, 1, seed)] if enough else []; own_h = [h.program for h in primitives_from(sol, ob, 1)[0]]
    return {"kind": sc.kind, "mask": sc.mask, "noise": sc.noise, "TT": hit(tt_h), "TI": hit(ti_h), "ET": hit(filter_by_ownership(nt, sol)), "EI": hit(filter_by_ownership(ni, sol), n), "NT": hit(nt), "NI": hit(ni, n), "OWN": hit(own_h, n),
            "kept_fraction": len(filter_by_ownership(ni, sol)) / max(1, len(ni)), "target_owned": owned(T, sol)}


# ================================================================ visibilité contrefactuelle contre contraste

def vis_row(sc: MaskedScene, seed=0) -> dict:
    """Même vivier d'hypothèses (type vrai, 5 hypothèses), trois classements : contraste global ; visibilité
    contrefactuelle avec les masques trouvés par l'attribution ; visibilité contrefactuelle avec le masque VRAI."""
    T = sc.target; ob = observe(sc.X, sc.y); H = [h.program for h in infer(T.skeleton(), ob, 5, score="global", seed=seed, regions=False)]; sol = solve(ob, None, seed)
    def occluders(p):
        from core.deep_ownership import _curves
        cs = _curves(p); return [(sol.owners[o.occluder].program(), o.side) for o in sol.occlusions if any(_same(k, a, sol.owners[o.occluded]) for k, a in cs)]
    best = lambda key: (lambda L: bool(L) and recovered(T, L[0]))(sorted(H, key=key))
    return {"kind": sc.kind, "mask": sc.mask, "noise": sc.noise, "contrast": best(lambda p: -global_score(p, ob.pairs)), "visible_inferred": best(lambda p: -visible_support(p, ob, occluders(p))["score"]),
            "visible_no_occluder": best(lambda p: -visible_support(p, ob, [])["score"]), "visible_true_occluder": best(lambda p: -visible_support(p, ob, [(sc.occluder, False)] if sc.occluder is not None else [])["score"]), "pool_has_truth": any(recovered(T, p) for p in H), "n": len(H)}


# ================================================================ secteurs : attribution ou paramétrage ?

def sector_row(sc: MaskedScene, seed=0) -> dict:
    T = sc.target; ob = observe(sc.X, sc.y); atoms = _atoms(sc); comps = [c for a in atoms for c in components(a)]; src, atom = true_sources(ob.points, atoms); sol = solve(ob, None, seed); acc = ownership_accuracy(sol, torch.where(atom == 0, src, torch.full_like(src, -1)), comps)
    obt = restrict(ob, atom == 0); tt = [h.program for h in infer(T.skeleton(), obt, 3, seed=seed, regions=False)] if len(obt.points) >= 4 else []; nt = [h.program for h in infer(T.skeleton(), ob, 3, seed=seed, regions=False)]; e = param_errors(T, tt[0]) if tt else {}
    rays = [any(component_match(_comp(o), c) for o in sol.owners) for c in components(T)]; n_ray = [int((src == i).sum()) for i in range(2)]
    stage = "ok" if (tt and recovered(T, tt[0])) else "no_hypothesis" if not tt else "vertex" if e.get("center", 9) > 0.35 else "angles"
    return {"kind": sc.kind, "mask": sc.mask, "noise": sc.noise, "ray_top1": acc["top1"], "ray_top3": acc["top3"], "rays_owned": sum(rays), "ray_points_min": min(n_ray), "TT": bool(tt) and recovered(T, tt[0]), "NT": bool(nt) and recovered(T, nt[0]), "tt_stage": stage,
            "tt_center_error": e.get("center"), "tt_direction_error": e.get("direction"), "tt_width_error": e.get("half_width")}


# ================================================================ chaîne aval et confirmation de flat_scene_k3

CHAIN_ARMS = ("reference", "flat_scene_k3", "own_k3", "own_nc_k3", "flat_owned_k3", "flat_own_union_k3", "true", "true_own_type", "true_own_infer", "po_type_k3", "po_owned_k3")
PREREGISTERED = ("reference", "flat_scene_k3", "own_k3")


def chain(task, arm: str, seed=0) -> dict:
    from core.deep_partial_exp import po_literals
    from core.deep_reachability import leaves
    from core.deep_search import DeepSearch
    from core.deep_world import full_region
    t0 = time.time(); tr = task.world.truth; tl = leaves(tr); X, y = task.X, task.y; F = full_region(3000, torch.Generator().manual_seed(12)); lab = task.world.label(F, 1.0); data = FitData.build(X, y, 0.10); ob = observe(X, y); comps = [c for a in tl for c in components(a)]
    src, atom = true_sources(ob.points, tl); lits = None; extra = {}; sol = None
    if arm in ("own_k3", "own_nc_k3", "flat_owned_k3", "flat_own_union_k3", "po_owned_k3"):
        sol = solve(ob, {"chance": 0.0} if arm == "own_nc_k3" else None, seed); extra = {"own_f1": pair_scores(sol.labels, src)["f1"], **{f"own_{k}": v for k, v in ownership_accuracy(sol, src, comps).items() if k != "cal"}, "owners": len(sol.owners), "assignments_tested": sol.stats["assignments_tested"], "own_seconds": sol.stats["seconds"]}
    if arm == "true": lits = list(tl)
    elif arm == "flat_scene_k3": lits = literals(flat_types(ob, 3, seed))
    elif arm in ("own_k3", "own_nc_k3"): lits = literals(primitives_from(sol, ob, 3)[0])
    elif arm == "flat_owned_k3": fl = flat_types(ob, 3, seed); keep = [h for h in fl if owned(h.program, sol) >= 1.0]; extra["kept_fraction"] = len(keep) / max(1, len(fl)); lits = literals(keep or fl)
    elif arm == "flat_own_union_k3": lits = literals(list(flat_types(ob, 3, seed)) + list(primitives_from(sol, ob, 3)[0]))
    elif arm == "po_type_k3": lits, _ = po_literals(task, 3, None, "visible", seed)
    elif arm == "po_owned_k3": pl, _ = po_literals(task, 3, None, "visible", seed); lits = filter_by_ownership(pl, sol) or pl
    elif arm in ("true_own_type", "true_own_infer"):
        lits = []
        for i, a in enumerate(tl):
            sub = restrict(ob, atom == i)
            if len(sub.points) >= 4: lits += [h.program for h in infer(a.skeleton(), sub, 3, seed=seed, regions=False)] if arm == "true_own_type" else literals(flat_types(sub, 1, seed))
    kw = {"fact_literals": lits, "fact_residual": False} if lits is not None else {}; depth = min(6, sum(len(components(a)) for a in tl) + 1); A = lambda p: float((predict(p, F) == lab).float().mean())
    if lits is not None and not lits: acc, best_acc, fe, best = [], 0.0, 0.0, None
    else:
        out = DeepSearch("factorized", budget=150.0, seed=seed, max_depth=depth, validated=None, **kw).run(data); acc = [A(h.program) for h in out.all.values()]; best = out.best().program; best_acc = A(best); fe = out.stats["fit_equivalents"]
        if lits is None: lits = [l for h in out.all.values() for l in leaves(h.program)]
    if arm == "own_k3" and best is not None:                                      # la composition aide-t-elle à ré-attribuer ?
        cl, lv = ownership_from_program(best, ob); extra["comp_f1"] = pair_scores(cl, src)["f1"]; m = src >= 0
        ok = torch.tensor([any(component_match(c, comps[int(s)]) for c in components(lv[int(l)])) if l >= 0 else False for l, s in zip(cl[m], src[m])]); extra["comp_top1"] = float(ok.float().mean()) if len(ok) else None
    P = _probe(); m = primitive_metrics(tl, lits, X, y, P) if lits else {"equivalence": 0.0, "precision": None}
    per = [{"type": atom_name(a), "type_hit": any(atom_name(l) == atom_name(a) and functional(a, l, P) >= 0.85 for l in lits), "recovered": any(atom_name(l) == atom_name(a) and recovered(a, l) for l in lits), "equivalent": _equivalent(a, lits)} for a in tl]
    return {"experiment": "chain", "arm": arm, **extra, "type_recall": sum(x["type_hit"] for x in per) / len(per), "primitive_recovery": sum(x["recovered"] for x in per) / len(per), "primitive_equivalence": m["equivalence"], "primitive_precision": m["precision"], "n_literals": len(lits), "leaves": per,
            "candidate_functional": max(acc, default=0.0) >= 0.97, "candidate_best": max(acc, default=0.0), "final_full": best_acc, "final_099": best_acc >= 0.99, "final_095": best_acc >= 0.95, "fit_equivalents": fe, "seconds": round(time.time() - t0, 2)}


# ================================================================ sous-unités AWS

def subunits(key):
    from core.deep_experiments import DOWN_DEPTHS, GEN_REGIMES, _row, down_task
    k = int(key); out = []
    if k >= 700:
        for d in DOWN_DEPTHS:
            for reg in GEN_REGIMES:
                for arm in CHAIN_ARMS: out.append((f"chain_d{d}_{reg}_{arm}", (lambda d=d, reg=reg, a=arm: [_row(down_task(key, d, reg), {"task_id": f"{key}_{d}_{reg}", **chain(down_task(key, d, reg), a, k)})])))
        return out
    for kind in KINDS:
        for mask in MASKS:
            def run(kind=kind, mask=mask):
                rows = []
                for j in range(2):
                    for nz in (0.0, 0.08):
                        sc = scene_at(k * 2 + j, kind, mask, nz); tag = {"scene": f"{k}_{j}_{kind}"}
                        rows += [{"experiment": "attr", **tag, **r} for r in attr_rows(sc, k)] + [{"experiment": "oracle", **tag, **oracle_row(sc, k)}, {"experiment": "vis", **tag, **vis_row(sc, k)}]
                        if kind == "sector": rows.append({"experiment": "sector", **tag, **sector_row(sc, k)})
                return rows
            out.append((f"bench_{kind}_{int(mask * 100)}", run))
    return out

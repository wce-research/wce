"""Expériences du sous-jalon Partial-Observation Primitive Parameter Inference — module d'ÉVALUATION (dev uniquement ;
il voit la vérité, rien n'en revient au système ; jamais le lot scellé).

  mask    : une primitive vraie, progressivement masquée (0 / 20 / 40 / 60 / 80 % de frontière cachée), avec ou sans bruit
            → fraction visible → erreur par paramètre → récupération fonctionnelle, par méthode d'estimation
  active  : mesures dirigées vers les paramètres contre mesures aléatoires
  chain   : type connu → paramètres inférés → structure candidate → récupération finale ; matrice d'oracles partiels ;
            top-k ; échange primitive ↔ composition

Graines : sous-plage dédiée de la partition dev (base + 300 000 ; mise au point : + 350 000).
Tâches de composition : clés 500+ de `down_task` (jamais utilisées ; mise au point : clé 599).
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import torch

from core.boundary_program import BNode, _inits, predict
from core.deep_fit import DT, FitData, fit_batch, heuristic_inits
from core.deep_partial import (IDENT, Hypothesis, complete, exchange, geometry, infer, joint_select, leaf_paths, observe, occlusion_score, parameter_queries, render, rendering_errors)
from core.deep_primitive_eval import component_match, components, primitive_metrics
from core.deep_primitives import atom_name, close_pairs, contrast, contrast_fit
from core.deep_primitives2 import _wrap
from core.deep_world import OBS_XMAX, SPLIT_BASE, obs_region

PO_BASE, PO_TUNE = SPLIT_BASE["dev"] + 300_000, SPLIT_BASE["dev"] + 350_000
KINDS = ("half", "band", "disc", "antidisc", "ring", "sector")
MASKS = (0.0, 0.2, 0.4, 0.6, 0.8)
NOISES = (0.0, 0.08)
BOX = (-3.0, OBS_XMAX, -3.0, 3.0)
METHODS = ("global_contrast", "global_target", "arc_kasa", "arc_taubin", "arc_normals", "arc_geometric", "arc_joint", "arc_visible", "ctx_oracle", "ctx_joint", "given_center", "given_orientation", "given_size", "minimal_components")
MAIN, BASELINE = "arc_visible", "global_contrast"


@dataclass
class MaskedScene:
    X: torch.Tensor
    y: torch.Tensor
    target: BNode                   # primitive vraie (évaluation seulement)
    occluder: Optional[BNode]       # masque vrai (évaluation, et bras oracle `ctx_oracle`)
    program: BNode
    kind: str
    mask: float
    noise: float
    seed: int


def _occluder(T: BNode, kind: str, mask: float, rng: random.Random) -> Optional[BNode]:
    """Masque qui cache une fraction `mask` de la frontière de T (dans la zone observée) : demi-plan pour les primitives
    circulaires, disque pour les primitives rectilignes (le masque n'est jamais du même type géométrique que la cible)."""
    B = render(T, n=200, box=BOX); th = rng.uniform(-math.pi, math.pi); i0 = rng.randrange(max(1, len(B)))
    if mask <= 0 or len(B) < 20: return None
    if kind in ("disc", "antidisc", "ring"):
        n = torch.tensor([math.cos(th), math.sin(th)], dtype=DT); b = float(torch.quantile(B @ n, 1 - mask)); return BNode("pred", "dot", "le", (th, b))
    p0 = B[i0]; r = float(torch.quantile((B - p0).norm(dim=1), 1 - mask)); return BNode("translate", params=(float(p0[0]), float(p0[1])), children=(BNode("pred", "norm", "le", (max(0.3, r),)),))


def masked_scene(index: int, kind: str, mask: float, noise: float, n=300, tune=False) -> MaskedScene:
    """Scène déterministe. La primitive et la direction du masque ne dépendent que de (index, type) : les niveaux de
    masquage et de bruit sont donc APPARIÉS sur la même primitive."""
    from core.deep_primitive_world import atom
    seed = (PO_TUNE if tune else PO_BASE) + int(index) * 131 + KINDS.index(kind) * 17; X = obs_region(n, torch.Generator().manual_seed(seed + 1))
    for attempt in range(200):
        T = atom(random.Random(seed * 7 + attempt), kind); ok = len(render(T, n=200, box=BOX)) >= 40
        for m in (0.0, 0.8):
            O = _occluder(T, kind, m, random.Random(seed * 11 + attempt)); p = T if O is None else BNode("and", children=(T, O)); f = float(predict(p, X).float().mean()); ok = ok and 0.04 < f < 0.96
        if ok: break
    O = _occluder(T, kind, mask, random.Random(seed * 11 + attempt)); prog = T if O is None else BNode("and", children=(T, O)); y = predict(prog, X); flip = torch.rand(n, generator=torch.Generator().manual_seed(seed + 2)) < noise
    return MaskedScene(X, torch.where(flip, 1 - y, y), T, O, prog, kind, mask, noise, seed)


def true_visible_fraction(sc: MaskedScene) -> float:
    B = render(sc.target, n=200, box=BOX); return 1.0 if sc.occluder is None or not len(B) else float(predict(sc.occluder, B.float()).float().mean())


# ================================================================ erreurs par paramètre

def param_errors(true: BNode, hyp: BNode) -> dict:
    """Erreur absolue par paramètre géométrique nommé (indépendante de la polarité)."""
    a, b = geometry(true), geometry(hyp)
    if a["kind"] != b["kind"]: return {}
    if a["kind"] == "circle":
        out = {"center": math.hypot(a["center"][0] - b["center"][0], a["center"][1] - b["center"][1])}; ra, rb = sorted(a["radii"]), sorted(b["radii"])
        if len(ra) == 1: out["radius"] = abs(ra[0] - rb[0])
        else: out["radius_in"], out["radius_out"] = abs(ra[0] - rb[0]), abs(ra[1] - rb[1])
        return out
    if a["kind"] == "line":
        d = _wrap(a["angle"] - b["angle"]); ob = list(b["offsets"])
        if abs(d) > math.pi / 2: d = _wrap(d + math.pi); ob = [-v for v in ob]
        oa, ob = sorted(a["offsets"]), sorted(ob); out = {"orientation": abs(d), "offset": sum(abs(u - v) for u, v in zip(oa, ob)) / len(oa)}
        if len(oa) == 2: out["width"] = abs((oa[1] - oa[0]) - (ob[1] - ob[0]))
        return out
    return {"center": math.hypot(a["center"][0] - b["center"][0], a["center"][1] - b["center"][1]), "direction": abs(_wrap(a["direction"] - b["direction"])), "half_width": abs(a["half_width"] - b["half_width"])}


def recovered(true: BNode, hyp: BNode) -> bool:
    hc = components(hyp); return all(any(component_match(c, d) for d in hc) for c in components(true))


def _probe(): return obs_region(3000, torch.Generator().manual_seed(5))


def functional(true: BNode, hyp: BNode, P=None) -> float:
    P = _probe() if P is None else P; a = float((predict(true, P) == predict(hyp, P)).float().mean()); return max(a, 1 - a) if atom_name(true) == "dot.le" else a


# ================================================================ méthodes d'estimation (type connu)

def given_of(true: BNode, what: str) -> dict:
    """Oracle PARTIEL : une seule famille de paramètres vrais est fournie."""
    g = geometry(true)
    if what == "center": return {"center": g["center"]} if g["kind"] in ("circle", "sector") else {}
    if what == "orientation": return {"orientation": g["angle"]} if g["kind"] == "line" else {"orientation": g["direction"]} if g["kind"] == "sector" else {}
    if what == "size": return {"size": g["radii"]} if g["kind"] == "circle" else {"size": (abs(g["offsets"][1] - g["offsets"][0]),)} if g["kind"] == "line" and len(g["offsets"]) == 2 else {"size": (g["half_width"],)} if g["kind"] == "sector" else {}
    return {}


def estimate(sc: MaskedScene, method: str, seed=0, k=5):
    """Renvoie (hypothèses classées [BNode], hypothèses évaluées, équivalents-ajustement, objets Hypothesis ou None)."""
    X, y = sc.X, sc.y; sk = sc.target.skeleton()
    if method == "global_contrast":                                               # optimisation globale actuelle : contraste, initialisations génériques
        g = torch.Generator().manual_seed(seed + 31); pairs = close_pairs(X, y); th, obj = contrast_fit(sk, pairs, torch.stack([v.to(DT) for v in _inits(sk, X.float(), y.long(), 6, g)]), steps=60)
        return [sk.with_params(th[i].tolist()) for i in obj.argsort(descending=True).tolist()][:k], 6, 6 * 0.6, None
    if method == "global_target":                                                 # gradient direct sur la cible
        d = FitData.build(X, y, 0.10); th, ll = fit_batch(sk, d, heuristic_inits(sk, d, 6, seed), steps=60); return [sk.with_params(th[i].tolist()) for i in ll.argsort(descending=True).tolist()][:k], 6, 6 * 0.6, None
    ob = observe(X, y)
    if method.startswith("arc_"):
        e = method[4:]; h = infer(sk, ob, k, score="visible" if e == "visible" else "global", joint=e in ("joint", "visible"), estimator=e if e in ("kasa", "taubin", "normals") else "geometric", seed=seed); return [x.program for x in h], len(h), 0.0, h
    if method.startswith("given_"):
        g = given_of(sc.target, method[6:]); h = infer(sk, ob, k, given=g, seed=seed) if g else []; return [x.program for x in h], len(h), 0.0, h
    if method == "minimal_components":                                            # base minimale : les composantes seules, sans contrainte partagée
        base = BNode("pred", "dot", "le") if geometry(sc.target)["kind"] != "circle" else BNode("translate", children=(BNode("pred", "norm", "le"),)); h = infer(base, ob, k, seed=seed); return [x.program for x in h], len(h), 0.0, h
    h = infer(sk, ob, k, seed=seed); hp = [x.program for x in h]
    if not hp: return [], 0, 0.0, h
    if method == "ctx_oracle":                                                    # contexte VRAI fourni : rendu contrefactuel de AND(hypothèse, masque vrai)
        return sorted(hp, key=lambda p: rendering_errors(p if sc.occluder is None else BNode("and", children=(p, sc.occluder)), X, y)), len(hp), 0.0, h
    if method == "ctx_joint":                                                     # structure connue, masque INFÉRÉ : choix conjoint puis échange primitive ↔ composition
        if sc.occluder is None: return sorted(hp, key=lambda p: rendering_errors(p, X, y)), len(hp), 0.0, h      # structure connue : la primitive seule
        ho = [x.program for x in infer(sc.occluder.skeleton(), ob, k, seed=seed)]
        if not ho: return hp, len(hp), 0.0, h
        best, _, n = joint_select([hp, ho], BNode("and", children=(hp[0], ho[0])), X, y); ex = exchange(best, ob, {repr(sk): hp}); top = ex.program.children[0]
        return [top] + [p for p in hp if p is not top], len(hp) + len(ho) + n + ex.evaluations, 0.0, h
    raise KeyError(method)


def mask_row(sc: MaskedScene, method: str, seed=0) -> dict:
    t0 = time.time(); hyps, n_h, fe, H = estimate(sc, method, seed); T = sc.target; P = _probe(); row = {"method": method, "kind": sc.kind, "mask": sc.mask, "noise": sc.noise, "visible_true": true_visible_fraction(sc), "n_hypotheses": n_h, "fit_equivalents": fe}
    if method == "minimal_components":
        hc = [c for p in hyps for c in components(p)]; tc = components(T); hit = [any(component_match(c, d) for d in hc) for c in tc]; row.update({"recovered_5": all(hit), "component_recall": sum(hit) / len(hit), "seconds": round(time.time() - t0, 3)}); return row
    if (method.startswith("given_") and not given_of(T, method[6:])): row.update({"not_applicable": True}); return row
    row.update({f"recovered_{k}": any(recovered(T, p) for p in hyps[:k]) for k in (1, 3, 5)}); row["functional_1"] = functional(T, hyps[0], P) if hyps else 0.0; row["functional_5"] = max((functional(T, p, P) for p in hyps[:5]), default=0.0)
    row["errors"] = param_errors(T, hyps[0]) if hyps else {}; row["no_hypothesis"] = not hyps; row["seconds"] = round(time.time() - t0, 3)
    if method == MAIN and H:
        h0: Hypothesis = H[0]; ob = observe(sc.X, sc.y); lb = complete(h0.program, ob, box=BOX); TB = render(T, n=200, box=BOX); row["visible_estimated"] = h0.visible_fraction; row["observed_fraction"] = lb.observed_fraction
        row["completion_accuracy"] = float((torch.cdist(lb.completion, TB).min(1).values < 0.3).float().mean()) if len(lb.completion) and len(TB) else None
        if h0.region is not None:
            g = geometry(T); truth = {"cx": g["center"][0], "cy": g["center"][1]} if g["kind"] == "circle" else {}; r = h0.region; ids = r.identifiability(); e = row["errors"]
            if g["kind"] == "circle": gh = geometry(h0.program); rr = min(g["radii"], key=lambda v: min(abs(v - q) for q in gh["radii"])); truth["r"] = rr
            if g["kind"] == "line": ta = g["angle"]; lo, hi = r.intervals["angle"]; ta = min((ta, ta + math.pi, ta - math.pi, ta + 2 * math.pi, ta - 2 * math.pi), key=lambda v: abs(v - (lo + hi) / 2)); truth = {"angle": ta}
            row.update({"region_covers": r.contains(**truth), "identifiability": ids, "widths": r.widths(), "posterior_sd": {k_: v[1] for k_, v in r.posterior.items()}, "sensitivity": r.sensitivity, "region_size": len(r.hypotheses)})
    if method == MAIN and sc.kind == "sector":                                    # autopsie des secteurs
        pairs = close_pairs(sc.X, sc.y); cs = components(T); ob = observe(sc.X, sc.y); from core.deep_partial import line_components; ls = [("line", c.params[0], c.params[1]) for c in line_components(ob, seed=seed)]
        B = render(T, n=200, box=BOX); vis = predict(sc.occluder, B.float()).bool() if sc.occluder is not None else torch.ones(len(B), dtype=torch.bool); half = len(B) // 2
        row["sector"] = {"true_z": contrast(T, pairs, use_weight=False)["z"], "rays_found": sum(any(component_match(c, d) for d in ls) for c in cs), "ray_visible_length": [round(4.5 * float(vis[:half].float().mean()), 2), round(4.5 * float(vis[half:].float().mean()), 2)] if len(B) else [0, 0], "lines_proposed": len(ls)}
    return row


# ================================================================ mesures actives pour les paramètres

def active_row(sc: MaskedScene, policy: str, seed=0, budget=16) -> dict:
    X, y = sc.X, sc.y; g = torch.Generator().manual_seed(seed + 77); sk = sc.target.skeleton()
    if policy != "none":
        pool = obs_region(600, g)
        if policy == "random": idx = torch.randperm(len(pool), generator=g)[:budget]
        else:
            ob = observe(X, y); H = infer(sk, ob, 5, seed=seed); hp = [h.program for h in H]; ctx = None
            if policy == "disagreement_ctx" and hp and sc.occluder is not None:
                ho = [h.program for h in infer(sc.occluder.skeleton(), ob, 3, seed=seed)]
                if ho: ctx = (joint_select([hp, ho], BNode("and", children=(hp[0], ho[0])), X, y)[0], (0,))
            idx = parameter_queries(hp, pool, budget, [max(1e-6, h.score) for h in H], ctx)
            if len(idx) < budget: rest = torch.tensor([i for i in torch.randperm(len(pool), generator=g).tolist() if i not in set(idx.tolist())][: budget - len(idx)], dtype=torch.long); idx = torch.cat([idx, rest])
        x = pool[idx]; yn = predict(sc.program, x); flip = torch.rand(len(x), generator=g) < sc.noise; X, y = torch.cat([X, x]), torch.cat([y, torch.where(flip, 1 - yn, yn)])
    r = mask_row(MaskedScene(X, y, sc.target, sc.occluder, sc.program, sc.kind, sc.mask, sc.noise, sc.seed), MAIN, seed); return {"method": f"active_{policy}", "kind": sc.kind, "mask": sc.mask, "queries": 0 if policy == "none" else budget, **{k: r.get(k) for k in ("recovered_1", "recovered_5", "functional_1", "errors")}}


# ================================================================ chaîne aval : type connu → paramètres → structure → fonction

# (nom, source des littéraux, k, oracle partiel, politique d'ajustement)
CHAIN_ARMS = (("reference", "reference", 0, None, "always"), ("true", "true", 0, None, "always"), ("types_old", "true_types", 0, None, "always"), ("partial_old", "partial", 0, None, "always"), ("robust_none", "robust", 0, None, "always"),
              ("po_type_k1", "po", 1, None, "always"), ("po_type_k3", "po", 3, None, "always"), ("po_type_k5", "po", 5, None, "always"), ("po_center_k1", "po", 1, "center", "always"), ("po_orientation_k1", "po", 1, "orientation", "always"),
              ("po_size_k1", "po", 1, "size", "always"), ("po_type_k3_global", "po_global", 3, None, "always"), ("reference_validated", "reference", 0, None, "validated"), ("true_validated", "true", 0, None, "validated"), ("po_type_k3_validated", "po", 3, None, "validated"))


def po_literals(task, k, what=None, score="visible", seed=0):
    """Type de chaque feuille vraie fourni ; paramètres inférés depuis la frontière partiellement observée (top-k)."""
    from core.deep_reachability import leaves
    ob = observe(task.X, task.y); out = []; per_leaf = []
    for a in leaves(task.world.truth):
        g = given_of(a, what) if what else {}; H = infer(a.skeleton(), ob, k, score=score, given=g, seed=seed, regions=False); hp = [h.program for h in H]; out += hp
        if atom_name(a) == "dot.le": out += [BNode("pred", "dot", "le", (_wrap(p.params[0] + math.pi), -p.params[1])) for p in hp]
        per_leaf.append({"type": atom_name(a), "top1": bool(hp) and recovered(a, hp[0]), "topk": any(recovered(a, p) for p in hp), "functional_k": max((functional(a, p) for p in hp), default=0.0), "errors": param_errors(a, hp[0]) if hp else {}, "n": len(hp)})
    seen = set(); lits = [p for p in out if not (repr(p) in seen or seen.add(repr(p)))]; return lits, per_leaf


def chain(task, arm: str, seed=0) -> dict:
    from core.deep_objective import E2E_POLICIES
    from core.deep_reachability import leaves
    from core.deep_robust_prim import oracle_literals, proposer, validator
    from core.deep_primitives2 import discover_robust, literals_of, load_config
    from core.deep_search import DeepSearch
    from core.deep_structure import n_atoms
    from core.deep_world import full_region
    _, source, k, what, policy = next(a for a in CHAIN_ARMS if a[0] == arm); t0 = time.time(); tr = task.world.truth; tl = leaves(tr); X, y = task.X, task.y; F = full_region(3000, torch.Generator().manual_seed(12)); lab = task.world.label(F, 1.0)
    data = FitData.build(X, y, 0.10); lits = None; per_leaf = []; kw = {}
    if source in ("true", "true_types", "partial"): lits = oracle_literals(task, source, seed)
    elif source == "robust": c = load_config(); d = discover_robust(X, y, proposer("union"), validator("full", c), seed=seed, joint=True, prune_z=c["prune_z"], split=c["split"]); lits = literals_of(d, alternatives=2)
    elif source.startswith("po"): lits, per_leaf = po_literals(task, k, what, "global" if source == "po_global" else "visible", seed)
    if lits is not None: kw = {"fact_literals": lits, "fact_residual": False}
    depth = min(6, sum(len(components(a)) for a in tl) + 1); A = lambda p: float((predict(p, F) == lab).float().mean())
    if lits is not None and not lits: acc, best_acc, loop_acc, ex, fe = [], 0.0, 0.0, None, 0.0
    else:
        out = DeepSearch("factorized", budget=150.0, seed=seed, max_depth=depth, validated=E2E_POLICIES[policy], **kw).run(data); best = out.best().program; acc = [A(h.program) for h in out.all.values()]; best_acc = A(best); fe = out.stats["fit_equivalents"]
        pool = lits if lits is not None else [l for h in out.all.values() for l in leaves(h.program)]; alt = {}
        for p in pool: alt.setdefault(repr(p.skeleton()), []).append(p)
        ex = exchange(best, observe(X, y), {s: v[:12] for s, v in alt.items()}); loop_acc = A(ex.program)                 # échange primitive ↔ composition, borné, validé par le rendu
        if lits is None: lits = pool
    m = primitive_metrics(tl, lits, X, y, _probe()) if lits else {"equivalence": 0.0, "component_recall": 0.0, "precision": None}
    return {"experiment": "chain", "arm": arm, "primitive_equivalence": m["equivalence"], "component_recall": m["component_recall"], "primitive_precision": m["precision"], "n_literals": len(lits), "leaves": per_leaf,
            "candidate_functional": max(acc, default=0.0) >= 0.97, "candidate_best": max(acc, default=0.0), "final_full": best_acc, "final_099": best_acc >= 0.99, "final_095": best_acc >= 0.95, "loop_full": loop_acc, "loop_099": loop_acc >= 0.99, "loop_095": loop_acc >= 0.95,
            "loop_commits": sum(l["action"] == "commit" for l in ex.log) if ex else 0, "loop_evaluations": ex.evaluations if ex else 0, "loop_errors": [ex.errors_before, ex.errors_after] if ex else None, "fit_equivalents": fe, "seconds": round(time.time() - t0, 2)}


# ================================================================ sous-unités AWS

def subunits(key):
    from core.deep_experiments import DOWN_DEPTHS, GEN_REGIMES, _row, down_task
    k = int(key); out = []
    if k >= 500:
        for d in DOWN_DEPTHS:
            for reg in GEN_REGIMES:
                for arm in CHAIN_ARMS: out.append((f"chain_d{d}_{reg}_{arm[0]}", (lambda d=d, reg=reg, a=arm[0]: [_row(down_task(key, d, reg), {"task_id": f"{key}_{d}_{reg}", **chain(down_task(key, d, reg), a, k)})])))
        return out
    for kind in KINDS:
        for mask in MASKS:
            for j in range(2):
                out.append((f"mask_{kind}_{int(mask * 100)}_{j}", (lambda kind=kind, mask=mask, j=j: [{"experiment": "mask", "scene": f"{k}_{j}_{kind}", **mask_row(masked_scene(k * 2 + j, kind, mask, nz), m, k)} for nz in NOISES for m in METHODS])))
        for mask in (0.4, 0.6):
            out.append((f"active_{kind}_{int(mask * 100)}", (lambda kind=kind, mask=mask: [{"experiment": "active", "scene": f"{k}_{kind}", **active_row(masked_scene(k * 2, kind, mask, 0.05), pol, k)} for pol in ("none", "random", "disagreement", "disagreement_ctx")])))
    return out

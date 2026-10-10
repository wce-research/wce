"""Experiments of the Fragment Attribution & Primitive Type Inference sub-milestone — EVALUATION module (dev only; it
sees the truth, nothing flows back to the system; never the sealed set).

  group   : quality of the attribution of midpoints to their true source, per grouping mechanism
  type    : type inference (without parameters as the main problem), oracle matrix fragments × type
  own     : occlusion ownership (hidden / out of domain / noise / absent)
  active  : additional measurement to resolve a type ambiguity
  chain   : attribution → type → parameters → candidate structure → function, WITHOUT a type oracle

Scenes: those of the occlusion bench (dev sub-range + 300,000), indices 100+ never used; tuning: + 350,000.
Composition tasks: keys 600+ of `down_task` (never used; tuning: key 699).
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence

import torch

from core.boundary_program import BNode, predict
from core.deep_fit import DT, FitData
from core.deep_fragments import (typed_labels, GROUPINGS, OWNERSHIP, SKELETONS, TypeHypothesis, assignments, build_graph, extract_fragments, flat_types, group_points, infer_types, literals, ownership, rank_by_group, type_of)
from core.deep_partial import ObservedBoundary, infer, observe, parameter_queries, render
from core.deep_partial_exp import BOX, KINDS, MASKS, MaskedScene, _probe, functional, masked_scene, recovered
from core.deep_primitive_eval import _ray_as_line, component_match, components, primitive_metrics
from core.deep_primitives import Pairs, atom_name
from core.deep_world import obs_region

SCENE_OFFSET = 100                      # scene indices 100+: never used by the previous sub-milestone (0–31)
VARIANTS = ("hier", "hier_no_negative", "hier_naive_negative", "hier_loop", "flat_fragments", "flat_scene")
MAIN = "hier"


def scene_at(index, kind, mask, noise, tune=False): return masked_scene((0 if tune else SCENE_OFFSET) + int(index), kind, mask, noise, tune=tune)


# ================================================================ true sources and attribution quality

def _dist(c: tuple, M: torch.Tensor) -> torch.Tensor:
    if c[0] == "circle": return ((M - torch.tensor(c[1:3], dtype=DT)).norm(dim=1) - c[3]).abs()
    if c[0] == "line": return ((M @ torch.tensor([math.cos(c[1]), math.sin(c[1])], dtype=DT)) - c[2]).abs()
    v = torch.tensor(c[1:3], dtype=DT); d = torch.tensor([math.cos(c[3]), math.sin(c[3])], dtype=DT); t = ((M - v) @ d).clamp_min(0.0); return (M - v - t[:, None] * d).norm(dim=1)


def true_sources(M: torch.Tensor, atoms: Sequence[BNode], tol=0.25):
    """True source of each midpoint: (atom index, component index) of the nearest true boundary within
    `tol`; −1 otherwise (noise). Returns (component [m], atom [m])."""
    cs = [(i, c) for i, a in enumerate(atoms) for c in components(a)]
    if not len(M) or not cs: return torch.full((len(M),), -1, dtype=torch.long), torch.full((len(M),), -1, dtype=torch.long)
    D = torch.stack([_dist(c, M) for _, c in cs], 1); d, j = D.min(1); comp = torch.where(d < tol, j, torch.full_like(j, -1)); atom = torch.tensor([cs[int(x)][0] if x >= 0 else -1 for x in comp], dtype=torch.long); return comp, atom


def pair_scores(lab: torch.Tensor, src: torch.Tensor) -> dict:
    """Precision / recall over pairs of midpoints: "same group" versus "same true source" (midpoints of known source)."""
    m = src >= 0; lab, src = lab[m], src[m]
    if len(lab) < 2: return {"precision": None, "recall": None, "f1": None, "assigned": None}
    sg = (lab[:, None] == lab[None, :]) & (lab[:, None] >= 0); ss = src[:, None] == src[None, :]; off = ~torch.eye(len(lab), dtype=torch.bool); tp = float((sg & ss & off).sum()); p = tp / max(1.0, float((sg & off).sum())); r = tp / max(1.0, float((ss & off).sum()))
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(1e-9, p + r), "assigned": float((lab >= 0).float().mean())}


def group_rows(sc: MaskedScene, seed=0) -> List[dict]:
    ob = observe(sc.X, sc.y); atoms = [sc.target] + ([sc.occluder] if sc.occluder is not None else []); comp, atom = true_sources(ob.points, atoms); out = []
    for mode in GROUPINGS + ("typed",):
        t0 = time.time(); lab = typed_labels(ob, seed) if mode == "typed" else group_points(ob, mode, seed=seed); s = pair_scores(lab, comp)
        mixed = [float((atom[lab == g] == 0).float().mean()) for g in torch.unique(lab).tolist() if g >= 0 and int(((lab == g) & (atom >= 0)).sum()) >= 4]            # "target" share of each group: 0 or 1 = pure
        out.append({"grouping": mode, "kind": sc.kind, "mask": sc.mask, "noise": sc.noise, **s, "groups": int(len(torch.unique(lab[lab >= 0]))), "mixed_groups": sum(0.1 < v < 0.9 for v in mixed) / max(1, len(mixed)), "seconds": round(time.time() - t0, 4)})
    return out


# ================================================================ type inference and oracle matrix

def restrict(ob: ObservedBoundary, keep: torch.Tensor) -> ObservedBoundary:
    """Observed boundary restricted to the midpoints of one source ("true fragments" oracle)."""
    from core.deep_partial import link_fragments
    return ObservedBoundary(ob.points[keep], ob.normals[keep], link_fragments(ob.points[keep]), ob.pairs, ob.X, ob.y)


def _overlap(p: BNode, M: torch.Tensor, is_target: torch.Tensor) -> float:
    """Share of the midpoints supporting the hypothesis (within 0.25 of its boundary) that come from the target primitive."""
    B = render(p, n=160, box=(-9.0, 9.0, -9.0, 9.0), reach=6.0)
    if not len(B) or not len(M): return 0.0
    s = torch.cdist(M, B).min(1).values < 0.25; return float(is_target[s].float().mean()) if int(s.sum()) >= 3 else 0.0


def _equivalent(T: BNode, hyps: Sequence[BNode]) -> bool:
    hc = [c for h in hyps for c in components(h)]; return all(any(component_match(c, d) for d in hc) for c in components(T))


def run_variant(ob: ObservedBoundary, variant: str, k=3, seed=0):
    """Returns (ranked hypotheses, statistics). k = types kept per group."""
    if variant == "flat_scene": t0 = time.time(); H = flat_types(ob, k, seed); return H, {"types_tested": len(H), "seconds": round(time.time() - t0, 3), "fragments": 0, "groups": 0}
    inf = infer_types(ob, k, negative={"hier_no_negative": "none", "hier_naive_negative": "naive"}.get(variant, "ownership"), hierarchical=variant != "flat_fragments", iters=1 if variant == "hier_loop" else 0, seed=seed)
    return inf.selected, {**inf.stats, "loop": [l["action"] for l in inf.log[1:]]}


def type_row(sc: MaskedScene, variant: str, seed=0) -> dict:
    """Type inference bench. `n` = number of true primitives in the scene (target, plus the occluder if there is one);
    "top-1" = among the first n hypotheses, "top-3" = among the first 3n.
      exact       same type AND boundaries within tolerance
      type        same type, supported mostly by midpoints of the target (the parameters are not judged)
      equivalent  all the boundaries of the target are recovered by the retained hypotheses, whatever their type
      functional  functional agreement ≥ 0.95 with the target"""
    T = sc.target; ob = observe(sc.X, sc.y); atoms = [T] + ([sc.occluder] if sc.occluder is not None else []); comp, atom = true_sources(ob.points, atoms); is_t = atom == 0; n = len(atoms); P = _probe(); tt = type_of(T)
    H, st = run_variant(ob, variant, 3, seed); row = {"variant": variant, "kind": sc.kind, "mask": sc.mask, "noise": sc.noise, "n_hypotheses": len(H), **{k: st.get(k) for k in ("types_tested", "fragments", "groups", "seconds")}}
    for name, top in (("1", n), ("3", 3 * n)):
        L = H[:top]; row[f"exact_{name}"] = any(h.type == tt and recovered(T, h.program) for h in L); row[f"type_{name}"] = any(h.type == tt and _overlap(h.program, ob.points, is_t) > 0.5 for h in L)
        row[f"equivalent_{name}"] = _equivalent(T, [h.program for h in L]); row[f"functional_{name}"] = any(functional(T, h.program, P) >= 0.95 for h in L)
    own = [(h, _overlap(h.program, ob.points, is_t)) for h in H]; own = [x for x in own if x[1] > 0.5]                         # hypotheses attributed to the target: the best-ranked one gives the predicted type
    row["predicted"] = own[0][0].type if own else "none"; row["confidence"] = own[0][0].confidence if own else None; row["correct"] = bool(own) and own[0][0].type == tt
    if own: row["signals"] = {k: v for k, v in own[0][0].signals.items() if v is not None}
    return row


def oracle_row(sc: MaskedScene, seed=0) -> dict:
    """Oracle matrix (exact recovery of the target as first hypothesis):
      TT true fragments + true type        TI true fragments + inferred type
      ET estimated fragments + true type    EI estimated fragments + inferred type"""
    T = sc.target; ob = observe(sc.X, sc.y); atoms = [T] + ([sc.occluder] if sc.occluder is not None else []); _, atom = true_sources(ob.points, atoms); tt = type_of(T); n = len(atoms); obt = restrict(ob, atom == 0); sk = T.skeleton()
    hit = lambda hs: bool(hs) and recovered(T, hs[0])
    TT = [h.program for h in infer(sk, obt, 3, seed=seed, regions=False)] if len(obt.points) >= 4 else []; ET = [h.program for h in infer(sk, ob, 3, seed=seed, regions=False)]
    ti = infer_types(obt, 1, seed=seed).selected if len(obt.points) >= 4 else []; ei = infer_types(ob, 1, seed=seed).selected
    return {"kind": sc.kind, "mask": sc.mask, "noise": sc.noise, "TT": hit(TT), "ET": hit(ET), "TI": any(h.type == tt and recovered(T, h.program) for h in ti[:1]), "TI_type": bool(ti) and ti[0].type == tt,
            "EI": any(h.type == tt and recovered(T, h.program) for h in ei[:n]), "target_points": int((atom == 0).sum())}


# ================================================================ occlusion ownership

def own_row(sc: MaskedScene, seed=0) -> dict:
    """The TRUE primitive is given; the system must say why its missing portions are missing. The other
    hypotheses (occluder candidates) are those the system inferred itself."""
    ob = observe(sc.X, sc.y); H = infer_types(ob, 2, seed=seed).hypotheses; others = [h.program for h in H if not recovered(sc.target, h.program)][:8]; o = ownership(sc.target, ob, others); B = render(sc.target, n=96, box=(-9.0, 9.0, -9.0, 9.0), reach=6.0)
    hidden = ~predict(sc.occluder, B.float()).bool() if sc.occluder is not None else torch.zeros(len(B), dtype=torch.bool); inside = (B[:, 0] >= BOX[0]) & (B[:, 0] <= BOX[1]) & (B[:, 1] >= BOX[2]) & (B[:, 1] <= BOX[3]); lab = o.labels
    th = hidden & inside; tv = ~hidden & inside; f = lambda m, names: (sum(1 for l, x in zip(lab, m.tolist()) if x and l in names) / max(1, int(m.sum()))) if int(m.sum()) else None
    return {"kind": sc.kind, "mask": sc.mask, "noise": sc.noise, "hidden_as_occluded": f(th, ("occluded",)), "hidden_as_absent": f(th, ("absent",)), "visible_as_observed": f(tv, ("observed", "noise")), "visible_as_occluded": f(tv, ("occluded",)), "visible_as_absent": f(tv, ("absent",)),
            "owner_found": o.owner is not None, "owner_is_occluder": o.owner is not None and sc.occluder is not None and (recovered(sc.occluder, others[o.owner]) or functional(sc.occluder, others[o.owner]) >= 0.9), **{f"frac_{k}": v for k, v in o.fractions.items()}}


# ================================================================ sector branch

def sector_row(sc: MaskedScene, seed=0) -> dict:
    T = sc.target; ob = observe(sc.X, sc.y); inf = infer_types(ob, 3, seed=seed); rays = components(T); g = geometry_of(T); fr = [f for f in inf.fragments if f.straight]; lines = [("line", f.params[0], f.params[1]) for f in fr]
    found = [any(component_match(r, l) for l in lines) for r in rays]; vert = [(i, j, s) for i, j, r, s in inf.graph.edges if r == "vertex"]; sect = [h for h in inf.hypotheses if h.type == "sector"]
    verr = min((math.hypot(h.program.params[0] - g["center"][0], h.program.params[1] - g["center"][1]) for h in sect), default=None); top = inf.selected[0].type if inf.selected else "none"
    stage = "ok" if any(recovered(T, h.program) for h in inf.selected[:2]) else "rays" if sum(found) < 2 else "vertex_relation" if not vert else "vertex_position" if verr is None or verr > 0.35 else "angles_or_ranking"
    return {"kind": sc.kind, "mask": sc.mask, "noise": sc.noise, "rays_found": sum(found), "straight_fragments": len(fr), "curved_fragments": len(inf.fragments) - len(fr), "vertex_relations": len(vert), "sector_hypotheses": len(sect), "vertex_error": verr, "stage": stage, "top_type": top,
            "confused_with_disc": top in ("disc", "antidisc"), "exact_1": any(recovered(T, h.program) for h in inf.selected[:2])}


def geometry_of(p):
    from core.deep_partial import geometry
    return geometry(p)


# ================================================================ active measurement to resolve a type ambiguity

def active_row(sc: MaskedScene, policy: str, seed=0, budget=16) -> dict:
    X, y = sc.X, sc.y; g = torch.Generator().manual_seed(seed + 91)
    if policy != "none":
        pool = obs_region(600, g)
        if policy == "random": idx = torch.randperm(len(pool), generator=g)[:budget]
        else:
            H = infer_types(observe(X, y), 3, seed=seed).selected[:6]; idx = parameter_queries([h.program for h in H], pool, budget, [max(1e-6, h.score) for h in H])            # where the competing TYPES predict different labels
            if len(idx) < budget: rest = torch.tensor([i for i in torch.randperm(len(pool), generator=g).tolist() if i not in set(idx.tolist())][: budget - len(idx)], dtype=torch.long); idx = torch.cat([idx, rest])
        x = pool[idx]; yn = predict(sc.program, x); flip = torch.rand(len(x), generator=g) < sc.noise; X, y = torch.cat([X, x]), torch.cat([y, torch.where(flip, 1 - yn, yn)])
    r = type_row(MaskedScene(X, y, sc.target, sc.occluder, sc.program, sc.kind, sc.mask, sc.noise, sc.seed), MAIN, seed); return {"policy": policy, "kind": sc.kind, "mask": sc.mask, **{k: r[k] for k in ("correct", "exact_1", "type_1", "functional_1")}}


# ================================================================ downstream chain WITHOUT a type oracle

# (name, source, k, policy)
CHAIN_ARMS = (("reference", "reference", 0, "always"), ("true", "true", 0, "always"), ("po_type_k3", "po", 3, "always"), ("robust_none", "robust", 0, "always"),
              ("hier_k1", "hier", 1, "always"), ("hier_k3", "hier", 3, "always"), ("hier_k3_no_negative", "hier_no_negative", 3, "always"), ("hier_k3_loop", "hier_loop", 3, "always"), ("flat_fragments_k3", "flat_fragments", 3, "always"),
              ("flat_scene_k1", "flat_scene", 1, "always"), ("flat_scene_k3", "flat_scene", 3, "always"), ("hier_k3_resid", "hier+resid", 3, "always"), ("flat_scene_k3_resid", "flat_scene+resid", 3, "always"),
              ("true_fragments_type", "oracle_tt", 3, "always"), ("true_fragments_infer", "oracle_ti", 3, "always"), ("reference_validated", "reference", 0, "validated"), ("hier_k3_validated", "hier", 3, "validated"),
              ("flat_scene_k3_resid_validated", "flat_scene+resid", 3, "validated"))


def chain(task, arm: str, seed=0) -> dict:
    from core.deep_objective import E2E_POLICIES
    from core.deep_partial_exp import po_literals
    from core.deep_primitives2 import discover_robust, literals_of, load_config
    from core.deep_reachability import leaves
    from core.deep_robust_prim import proposer, validator
    from core.deep_search import DeepSearch
    from core.deep_world import full_region
    _, source, k, policy = next(a for a in CHAIN_ARMS if a[0] == arm); t0 = time.time(); tr = task.world.truth; tl = leaves(tr); X, y = task.X, task.y; F = full_region(3000, torch.Generator().manual_seed(12)); lab = task.world.label(F, 1.0)
    data = FitData.build(X, y, 0.10); lits = None; st = {}; resid = source.endswith("+resid"); base = source.replace("+resid", ""); ob = observe(X, y); comp, atom = true_sources(ob.points, tl); attr = None
    if base == "true": lits = list(tl)
    elif base == "robust": c = load_config(); d = discover_robust(X, y, proposer("union"), validator("full", c), seed=seed, joint=True, prune_z=c["prune_z"], split=c["split"]); lits = literals_of(d, alternatives=2)
    elif base == "po": lits, _ = po_literals(task, k, None, "visible", seed)
    elif base in ("oracle_tt", "oracle_ti"):                                       # TRUE fragments: the midpoints of each leaf are isolated
        lits = []
        for i, a in enumerate(tl):
            sub = restrict(ob, atom == i)
            if len(sub.points) < 4: continue
            lits += [h.program for h in infer(a.skeleton(), sub, k, seed=seed, regions=False)] if base == "oracle_tt" else literals(infer_types(sub, k, seed=seed).selected)
    elif base != "reference":
        H, st = run_variant(ob, base, k, seed); lits = literals(H)
        if base != "flat_scene": attr = pair_scores(typed_labels(ob, seed), comp)
    kw = {"fact_literals": lits, "fact_residual": resid} if lits is not None else {}; depth = min(6, sum(len(components(a)) for a in tl) + 1); A = lambda p: float((predict(p, F) == lab).float().mean())
    if lits is not None and not lits and not resid: acc, best_acc, fe = [], 0.0, 0.0
    else:
        out = DeepSearch("factorized", budget=150.0, seed=seed, max_depth=depth, validated=E2E_POLICIES[policy], **kw).run(data); acc = [A(h.program) for h in out.all.values()]; best_acc = A(out.best().program); fe = out.stats["fit_equivalents"]
        if lits is None: lits = [l for h in out.all.values() for l in leaves(h.program)]
    m = primitive_metrics(tl, lits, X, y, _probe()) if lits else {"equivalence": 0.0, "component_recall": 0.0, "precision": None}; P = _probe()
    per = [{"type": atom_name(a), "type_hit": any(atom_name(l) == atom_name(a) and functional(a, l, P) >= 0.85 for l in lits), "recovered": any(atom_name(l) == atom_name(a) and recovered(a, l) for l in lits), "equivalent": _equivalent(a, lits)} for a in tl]
    return {"experiment": "chain", "arm": arm, "attribution_f1": attr["f1"] if attr else None, "type_recall": sum(x["type_hit"] for x in per) / len(per), "parameter_recovery": sum(x["recovered"] for x in per) / len(per), "primitive_equivalence": m["equivalence"], "primitive_precision": m["precision"],
            "n_literals": len(lits), "leaves": per, "candidate_functional": max(acc, default=0.0) >= 0.97, "candidate_best": max(acc, default=0.0), "final_full": best_acc, "final_099": best_acc >= 0.99, "final_095": best_acc >= 0.95, "types_tested": st.get("types_tested"), "fragments": st.get("fragments"),
            "groups": st.get("groups"), "inference_seconds": st.get("seconds"), "fit_equivalents": fe, "seconds": round(time.time() - t0, 2)}


# ================================================================ AWS sub-units

def subunits(key):
    from core.deep_experiments import DOWN_DEPTHS, GEN_REGIMES, _row, down_task
    k = int(key); out = []
    if k >= 600:
        for d in DOWN_DEPTHS:
            for reg in GEN_REGIMES:
                for arm in CHAIN_ARMS: out.append((f"chain_d{d}_{reg}_{arm[0]}", (lambda d=d, reg=reg, a=arm[0]: [_row(down_task(key, d, reg), {"task_id": f"{key}_{d}_{reg}", **chain(down_task(key, d, reg), a, k)})])))
        return out
    for kind in KINDS:
        for mask in MASKS:
            def run(kind=kind, mask=mask):
                rows = []
                for j in range(2):
                    for nz in (0.0, 0.08):
                        sc = scene_at(k * 2 + j, kind, mask, nz); tag = {"scene": f"{k}_{j}_{kind}"}
                        rows += [{"experiment": "group", **tag, **r} for r in group_rows(sc, k)] + [{"experiment": "type", **tag, **type_row(sc, v, k)} for v in VARIANTS] + [{"experiment": "oracle", **tag, **oracle_row(sc, k)}, {"experiment": "own", **tag, **own_row(sc, k)}]
                        if kind == "sector": rows.append({"experiment": "sector", **tag, **sector_row(sc, k)})
                return rows
            out.append((f"bench_{kind}_{int(mask * 100)}", run))
        for mask in (0.2, 0.4):
            out.append((f"active_{kind}_{int(mask * 100)}", (lambda kind=kind, mask=mask: [{"experiment": "active", "scene": f"{k}_{kind}", **active_row(scene_at(k * 2, kind, mask, 0.05), pol, k)} for pol in ("none", "random", "type_disagreement")])))
    return out

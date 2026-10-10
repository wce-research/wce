"""Experiments of the Robust Primitive Discovery sub-milestone — EVALUATION module (dev only; it sees the truth, nothing
of it comes back to the system; never the sealed set).

  audit      : breakdown of the errors of a method, by true boundary component and by atom type
  arms       : comparison of the proposal × validation × assembly chains on the atomic bench
  validator  : true acceptances / false acceptances / true rejections / false rejections, on a same pool of candidates
  diversity  : overlap and complementarity of the proposers
  active     : measurements directed at the undecided primitives versus random measurements
  chain      : primitives → candidate structure → final recovery, with oracle ceilings

Scenes: indices ≥ 2000 of the atomic bench (never used before); indices ≥ 5000 for tuning.
Composition tasks: keys 400+ of `down_task` (never used before).
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence

import torch

from core.boundary_program import BNode, _inits, predict
from core.deep_fit import DT, FitCounter, FitData
from core.deep_primitive_eval import component_match, components, primitive_metrics
from core.deep_primitives import (CompatiblePrimitives, ContrastProposer, GeometricProposer, GradientProposer, PrimitiveValidation, atom_name, close_pairs, contrast, contrast_fit, discover,
                                  separates)
from core.deep_primitives2 import (clean_pairs, load_config, null_floor, LocalRefiner, RansacProposer, GridProposer, HoughProposer, RobustValidator, UnionProposer, borderline, boundary_query_scores, discover_robust, literals_of, split_pairs, split_points)
from core.deep_world import obs_region

EVAL_BASE, TUNE_BASE = 2000, 5000
LOOSE = dict(tol_angle=0.55, tol_offset=0.75, tol_center=0.9, tol_radius=0.75)
AUDIT_CLASSES = ("found", "unrecognized_equivalent", "dominated", "rejected", "far_parameters", "wrong_type", "never_proposed")


def _neural():
    from core.deep_neural_proposer import NeuralProposer, load
    return NeuralProposer() if load("full") is not None else None


def proposer(name):
    if name == "gradient": return GradientProposer()
    if name == "gradient_min": return GradientProposer("minimal")
    if name == "geometric": return GeometricProposer("minimal")
    if name == "contrast": return ContrastProposer("minimal")
    if name == "hough": return HoughProposer()
    if name == "grid": return GridProposer()
    if name == "ransac": return RansacProposer()
    if name == "neural": return _neural()
    if name == "ransac_local": return LocalRefiner(RansacProposer())
    if name == "union": return LocalRefiner(UnionProposer([RansacProposer(), HoughProposer()]))
    if name == "union_all": return LocalRefiner(UnionProposer([RansacProposer(), HoughProposer(), GridProposer(), GradientProposer("minimal")]))
    if name == "union_neural": n = _neural(); return None if n is None else LocalRefiner(UnionProposer([RansacProposer(), HoughProposer(), n]))
    raise KeyError(name)


def validator(name, cfg=None):
    c = cfg or load_config(); kw = dict(z_oos=c["z_oos"], min_support=c["min_support"], min_purity=c["min_purity"], min_segments=c["min_segments"])
    if name == "full": return RobustValidator(**kw)
    if name == "stab": return RobustValidator(**kw, max_instability=0.12)
    if name == "oos_only": return RobustValidator(**{**kw, "min_purity": 0.0, "min_segments": 0})
    if name == "oos_purity": return RobustValidator(**{**kw, "min_segments": 0})
    if name == "oos_segments": return RobustValidator(**{**kw, "min_purity": 0.0})
    if name == "z_only": return RobustValidator(**{**kw, "min_purity": 0.0, "min_segments": 0, "min_support": 0})
    raise KeyError(name)


# arms: (proposer, validator | "old" = validator of the previous sub-milestone, joint assembly)
ARMS = {"gradient": ("gradient", "old", False), "gradient_min": ("gradient_min", "old", False), "union_old": ("union", "old", False),
        "gradient_rv": ("gradient", "full", False), "gradmin_rv": ("gradient_min", "full", False), "hough_rv": ("hough", "full", False), "grid_rv": ("grid", "full", False),
        "ransac_rv": ("ransac", "full", False), "ransac_local_rv": ("ransac_local", "full", False), "union_all_joint": ("union_all", "full", True),
        "union_rv": ("union", "full", False), "union_joint": ("union", "full", True), "union_stab": ("union", "stab", True), "union_z_only": ("union", "z_only", True),
        "union_pairsplit": ("union", "full", True), "union_holdout": ("union", "full", True), "union_insample": ("union", "full", True), "gradient_joint": ("gradient", "full", True), "neural_rv": ("neural", "full", False), "union_neural_joint": ("union_neural", "full", True)}
REFERENCE, BEST = "gradient", "union_joint"


def scene_at(index, category, tune=False):
    from core.deep_primitive_world import make_scene
    return make_scene((TUNE_BASE if tune else EVAL_BASE) + int(index), category)


def discover_logged(X, y, prop, rounds=4, keep_per_round=3, seed=0, max_classes=12):
    """Reproduces EXACTLY `core.deep_primitives.discover` (validator of the previous sub-milestone) while logging each
    candidate: required for the audit. Returns (CompatiblePrimitives, log, proposals, fit-equivalents)."""
    v = PrimitiveValidation(); pairs = close_pairs(X, y); cp = CompatiblePrimitives(X); counter = FitCounter(); log = []; n = 0
    for r in range(rounds):
        cands = prop.propose(X, y, pairs, seed + r, counter); n += len(cands); base = v.z_min; v.z_min = max(base, math.sqrt(2 * math.log(max(2, len(cands)))) + 1.0)
        cands = [v.validate(c, pairs, X, cp, seed) for c in cands]; rows = {id(c): {"program": c.program, "source": c.source, "round": r, "z_in": c.z, "z_oos": c.z, "accepted": c.validated, "reason": c.reason, "kept": False} for c in cands}; kept = 0
        for c in sorted((c for c in cands if c.validated), key=lambda c: -c.z):
            if kept >= keep_per_round or len(cp) >= max_classes: break
            c = v.validate(c, pairs, X, cp, seed)
            if not c.validated: continue
            status, _ = cp.add(c); rows[id(c)]["kept"] = True
            if status == "new": kept += 1; s = separates(c.program, pairs); pairs.weight = torch.where(s & (pairs.opposite > 0), torch.zeros_like(pairs.weight), pairs.weight)
        v.z_min = base; log += list(rows.values())
        if kept == 0: break
    return cp, log, n, counter.fit_equivalents


def run_arm(X, y, arm, seed=0, cfg=None):
    """Returns (retained programs [all members + assembled atoms], representatives, log, cost) or None if unavailable."""
    pn, vn, joint = ARMS[arm]; p = proposer(pn); t0 = time.time()
    if p is None: return None
    if vn == "old": cp, log, n, fe = discover_logged(X, y, p, seed=seed); jt = []
    else: c = cfg or load_config(); d = discover_robust(X, y, p, validator(vn, c), seed=seed, joint=joint, prune_z=c["prune_z"], split={"union_pairsplit": "pairs", "union_holdout": "points", "union_insample": "none"}.get(arm, c["split"])); cp, log, n, fe, jt = d.primitives, d.candidates, d.proposals, d.fit_equivalents, d.joint
    found = [m.program for c in cp.classes for m in c.members] + list(jt); reps = [c.representative.program for c in cp.classes] + list(jt)
    return {"found": found, "reps": reps, "log": log, "proposals": n, "fit_equivalents": fe, "seconds": round(time.time() - t0, 3), "classes": len(cp), "joint": len(jt)}


def _probe(): return obs_region(3000, torch.Generator().manual_seed(5))


def arm_row(scene, arm, seed=0, cfg=None):
    t = scene.task; r = run_arm(t.X, t.y, arm, seed, cfg)
    if r is None: return {"method": arm, "unavailable": True}
    m = primitive_metrics(scene.atoms, r["found"], t.X, t.y, _probe()); tc = [c for a in scene.atoms for c in components(a)]
    fp = sum(not all(any(component_match(c, d) for d in tc) for c in components(p)) for p in r["reps"])
    return {"method": arm, **{k: v for k, v in m.items()}, "classes": r["classes"], "joint": r["joint"], "retained": len(r["reps"]), "false_positives": fp, "proposals": r["proposals"], "fit_equivalents": r["fit_equivalents"], "seconds": r["seconds"],
            "compactness": len(r["reps"]) / max(1, len(tc))}


# ================================================================ audit

def audit(scene, arm=REFERENCE, seed=0) -> List[dict]:
    """One row per true boundary COMPONENT (line, circle, half-line), labelled with the type of its atom, classified:
    found | unrecognized_equivalent | dominated | rejected | far_parameters | wrong_type | never_proposed.
    Plus one `false_positive` row per retained primitive with no matching true boundary."""
    t = scene.task; r = run_arm(t.X, t.y, arm, seed)
    if r is None: return []
    P = _probe(); pairs = close_pairs(t.X, t.y); log = r["log"]; lc = [components(l["program"]) for l in log]; kept_c = [c for p in r["found"] for c in components(p)]; out = []
    for a in scene.atoms:
        ta = predict(a, P); eq = any((lambda v: v >= 0.95 or v <= 0.05)(float((predict(p, P) == ta).float().mean())) for p in r["found"]); sa = separates(a, pairs) & (pairs.opposite > 0); na = max(1, int(sa.sum()))
        for c in components(a):
            hit = [i for i, cs in enumerate(lc) if any(component_match(c, d) for d in cs)]; reason = None
            if any(component_match(c, d) for d in kept_c): cls = "found"
            elif eq: cls = "unrecognized_equivalent"
            elif any(log[i]["accepted"] for i in hit): cls = "dominated"
            elif hit: cls = "rejected"; reason = max((log[i] for i in hit), key=lambda l: l["z_in"])["reason"]
            elif any(component_match(c, d, **LOOSE) for cs in lc for d in cs if d[0] == c[0] or {d[0], c[0]} == {"line", "ray"}): cls = "far_parameters"
            elif any(float((separates(l["program"], pairs) & sa).sum()) / na >= 0.6 for l in log): cls = "wrong_type"
            else: cls = "never_proposed"
            out.append({"kind": "component", "method": arm, "type": atom_name(a), "component": c[0], "class": cls, "reason": reason, "true_z": contrast(a, pairs, use_weight=False)["z"]})
    tc = [c for a in scene.atoms for c in components(a)]
    for p in r["reps"]:
        cs = components(p)
        if not all(any(component_match(c, d) for d in tc) for c in cs):
            out.append({"kind": "false_positive", "method": arm, "type": atom_name(p), "near_true": all(any(component_match(c, d, **LOOSE) for d in tc) for c in cs), "z": contrast(p, pairs, use_weight=False)["z"]})
    return out


# ================================================================ validator: TA / FA / TR / FR on a same pool

VALIDATORS = ("old", "holdout_z3", "holdout_cal", "insample_z3", "insample_cal", "full", "stab")


def validator_study(scene, seed=0) -> List[dict]:
    """Common pool: first round of the `union` proposer on one half of the POINTS. A candidate is TRUE if all its
    components match a true boundary. Variants:
      old          previous validator (all the pairs, z ≥ max(3, multiplicity))
      holdout_z3   contrast on the OTHER half of the points, nominal threshold 3 (not calibrated)
      holdout_cal  same, threshold calibrated on random labels
      insample_z3  contrast on all the cleaned pairs, nominal threshold 3
      insample_cal same, threshold calibrated on random labels
      full         retained configuration (calibrated threshold + support + purity + multi-region consistency)
      stab         full + stability under subsampling"""
    t = scene.task; X, y = t.X, t.y; allp = close_pairs(X, y); ia, ib = split_points(X, y, seed); A, B = clean_pairs(X[ia], y[ia]), clean_pairs(X[ib], y[ib]); C = clean_pairs(X, y); cands = proposer("union").propose(X[ia], y[ia], A, seed, FitCounter())
    tc = [c for a in scene.atoms for c in components(a)]; truth = [all(any(component_match(c, d) for d in tc) for c in components(p.program)) for p in cands]; out = []; cfg = load_config(); bare = dict(min_support=0, min_purity=0.0, min_segments=0)
    for vn in VALIDATORS:
        t0 = time.time()
        if vn == "old": v = PrimitiveValidation(); v.z_min = max(v.z_min, math.sqrt(2 * math.log(max(2, len(cands)))) + 1.0); acc = [v.validate(c, allp, X, None, seed).validated for c in cands]
        else:
            v, pv = {"holdout_z3": (RobustValidator(z_oos=3.0, **bare), (A, B)), "holdout_cal": (RobustValidator(z_oos=null_floor("points"), **bare), (A, B)), "insample_z3": (RobustValidator(z_oos=3.0, **bare), (C, C)),
                     "insample_cal": (RobustValidator(z_oos=null_floor("none", cfg["z_oos"]), **bare), (C, C)), "full": (validator("full", cfg), (C, C) if cfg["split"] == "none" else (A, B)), "stab": (validator("stab", cfg), (C, C) if cfg["split"] == "none" else (A, B))}[vn]
            acc = [v.accept(v.evidence(c.program, pv[0], pv[1], X, seed), 1)[0] for c in cands]
        n = lambda a, b: sum(1 for x, z in zip(acc, truth) if x == a and z == b)
        out.append({"validator": vn, "candidates": len(cands), "TA": n(True, True), "FA": n(True, False), "TR": n(False, False), "FR": n(False, True), "seconds": round(time.time() - t0, 3)})
    return out


# ================================================================ diversity of the proposers

PROPOSERS = ("gradient", "gradient_min", "geometric", "contrast", "hough", "grid", "ransac", "ransac_local", "neural")


def diversity(scene, seed=0) -> List[dict]:
    """Per proposer (first round, all the pairs): indices of the true components reached, proposals, cost."""
    t = scene.task; X, y = t.X, t.y; tc = [c for a in scene.atoms for c in components(a)]; out = []
    for pn in PROPOSERS:
        p = proposer(pn)
        if p is None: continue
        counter = FitCounter(); t0 = time.time(); cs = [d for c in p.propose(X, y, close_pairs(X, y), seed, counter) for d in components(c.program)]
        out.append({"proposer": pn, "hits": [i for i, c in enumerate(tc) if any(component_match(c, d) for d in cs)], "n_true": len(tc), "proposals": len(cs), "fit_equivalents": counter.fit_equivalents, "seconds": round(time.time() - t0, 3)})
    return out


# ================================================================ active measurements specific to primitives

def active(scene, policy, seed=0, budget=24):
    t = scene.task; g = torch.Generator().manual_seed(seed + 77); X, y = t.X, t.y
    if policy != "none":
        pool = obs_region(600, g)
        if policy == "primitive":
            c = load_config(); d = discover_robust(X, y, proposer("union"), validator("full", c), seed=seed, joint=True, prune_z=c["prune_z"], split=c["split"]); sc = boundary_query_scores(borderline(d), pool) + 1e-3 * torch.rand(len(pool), generator=g); idx = sc.topk(budget).indices
        else: idx = torch.randperm(len(pool), generator=g)[:budget]
        x = pool[idx]; X, y = torch.cat([X, x]), torch.cat([y, t.oracle(x, "standard", g)])
    r = run_arm(X, y, BEST, seed); m = primitive_metrics(scene.atoms, r["found"], X, y, _probe()); return {"method": f"active_{policy}", "queries": 0 if policy == "none" else budget, **{k: m[k] for k in ("exact", "equivalence", "component_recall", "precision")}}


# ================================================================ downstream chain and oracle ceilings

SOURCES = ("reference", "true", "true_types", "partial", "gradient_disc", "robust_min", "robust_joint", "robust_joint_resid")
CHAIN_ARMS = tuple((s, "always") for s in SOURCES) + (("reference", "validated"), ("true", "validated"), ("robust_joint", "validated"))


def _estimate(skels: Sequence[BNode], inits_of, X, y, seed):
    """Parameters ESTIMATED by contrast for given skeletons; the best initialisation (contrast) is kept."""
    pairs = close_pairs(X, y); out = []
    for i, sk in enumerate(skels):
        th, obj = contrast_fit(sk, pairs, inits_of(i, sk), steps=60); out.append(sk.with_params(th[int(obj.argmax())].tolist()))
    return out


def oracle_literals(task, source, seed=0):
    """Ceilings: `true` true atoms; `true_types` true TYPES, parameters to be estimated from generic
    initialisations; `partial` true centres and orientations, radii / offsets / widths to be estimated."""
    from core.deep_reachability import leaves
    tl = leaves(task.world.truth); X, y = task.X, task.y
    if source == "true": return tl
    g = torch.Generator().manual_seed(seed + 31)
    if source == "true_types": return _estimate([a.skeleton() for a in tl], lambda i, sk: torch.stack([v.to(DT) for v in _inits(sk, X.float(), y.long(), 6, g)]), X, y, seed)
    def generic(i, sk):
        a = tl[i]; rows = []
        for s in (0.6, 1.0, 1.5):
            v = list(a.param_vector()); l = a.children[0] if a.kind == "translate" else a; off = 2 if a.kind == "translate" else 0
            if l.feat == "norm": v[off:] = [s] if l.op != "in" else [0.6 * s, 1.4 * s]
            elif l.feat == "dot": v[off + 1:] = [0.0 + (s - 1.0)] if l.op != "in" else [-0.5 * s, 0.5 * s]
            elif l.feat == "angle": v[off + 1:] = [0.8 * s]
            rows.append(torch.tensor(v, dtype=DT))
        return torch.stack(rows)
    return _estimate([a.skeleton() for a in tl], generic, X, y, seed)


def chain(task, source, policy="always", seed=0):
    """discovered primitives → recall of the primitives → candidate structure → final functional recovery. The
    composition, the score, the simplification and the selection are those of the existing pipeline, unchanged."""
    from core.deep_objective import E2E_POLICIES
    from core.deep_reachability import leaves
    from core.deep_search import DeepSearch
    from core.deep_structure import n_atoms
    from core.deep_world import full_region
    t0 = time.time(); tr = task.world.truth; tl = leaves(tr); X, y = task.X, task.y; F = full_region(3000, torch.Generator().manual_seed(12)); lab = task.world.label(F, 1.0); data = FitData.build(X, y, 0.10); kw = {}; lits = None; cost = 0.0
    if source in ("true", "true_types", "partial"): lits = oracle_literals(task, source, seed)
    elif source == "gradient_disc": d = discover(X, y, GradientProposer(), seed=seed); lits = d.primitives.literals(alternatives=2); cost = d.fit_equivalents
    elif source.startswith("robust"): c = load_config(); d = discover_robust(X, y, proposer("union"), validator("full", c), seed=seed, joint=source != "robust_min", prune_z=c["prune_z"], split=c["split"]); lits = literals_of(d, alternatives=2, with_joint=source != "robust_min"); cost = d.fit_equivalents
    if lits is not None: kw = {"fact_literals": lits, "fact_residual": source.endswith("_resid")}
    depth = min(6, sum(len(components(a)) for a in tl) + 1)                      # same depth for all the sources: the minimal basis needs one node per boundary
    if lits is not None and not lits: acc, best_acc, n = [], 0.0, 0; fe = cost
    else:
        out = DeepSearch("factorized", budget=150.0, seed=seed, max_depth=depth, validated=E2E_POLICIES[policy], **kw).run(data); best = out.best(); acc = [float((predict(h.program, F) == lab).float().mean()) for h in out.all.values()]
        best_acc = float((predict(best.program, F) == lab).float().mean()); n = n_atoms(best.program); fe = out.stats["fit_equivalents"] + cost
        if lits is None: lits = [l for h in out.all.values() for l in leaves(h.program)]
    m = primitive_metrics(tl, lits, X, y, _probe()) if lits else {"equivalence": 0.0, "exact": 0.0, "component_recall": 0.0, "precision": None}
    return {"experiment": "chain", "source": source, "policy": policy, "primitive_equivalence": m["equivalence"], "primitive_exact": m["exact"], "component_recall": m["component_recall"], "primitive_precision": m["precision"], "n_literals": len(lits),
            "candidate_functional": max(acc, default=0.0) >= 0.97, "candidate_best": max(acc, default=0.0), "final_full": best_acc, "final_functional": best_acc >= 0.99, "final_095": best_acc >= 0.95, "atoms": n, "fit_equivalents": fe, "seconds": round(time.time() - t0, 2)}


# ================================================================ AWS sub-units

def subunits(key):
    from core.deep_experiments import DOWN_DEPTHS, GEN_REGIMES, _row, down_task
    from core.deep_primitive_world import CATEGORIES
    k = int(key); out = []
    if k >= 400:                                                                   # keys 400+: downstream chain (composition tasks never used)
        for d in DOWN_DEPTHS:
            for reg in GEN_REGIMES:
                for s, pol in CHAIN_ARMS: out.append((f"chain_d{d}_{reg}_{s}_{pol}", (lambda d=d, reg=reg, s=s, pol=pol: [_row(down_task(key, d, reg), {"task_id": f"{key}_{d}_{reg}", **chain(down_task(key, d, reg), s, pol, k)})])))
        return out
    for cat in CATEGORIES:
        for j in range(2):
            sc = lambda cat=cat, j=j: scene_at(k * 2 + j, cat); tag = {"category": cat, "scene": f"{k}_{j}_{cat}"}
            out.append((f"arms_{cat}_{j}", (lambda sc=sc, tag=tag: [{"experiment": "atomic", **tag, **arm_row(sc(), a, k)} for a in ARMS])))
            out.append((f"audit_{cat}_{j}", (lambda sc=sc, tag=tag: [{"experiment": "audit", **tag, **r} for a in (REFERENCE, BEST) for r in audit(sc(), a, k)])))
            out.append((f"validator_{cat}_{j}", (lambda sc=sc, tag=tag: [{"experiment": "validator", **tag, **r} for r in validator_study(sc(), k)])))
            out.append((f"diversity_{cat}_{j}", (lambda sc=sc, tag=tag: [{"experiment": "diversity", **tag, **r} for r in diversity(sc(), k)])))
    for cat in ("two", "multi", "partial"):
        out.append((f"active_{cat}", (lambda cat=cat: [{"experiment": "active", "category": cat, **active(scene_at(k * 2, cat), pol, k)} for pol in ("none", "random", "primitive")])))
    return out

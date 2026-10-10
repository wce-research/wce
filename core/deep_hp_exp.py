"""Expériences du sous-jalon High-Precision Primitive Estimation — module d'ÉVALUATION (dev uniquement ; il voit la vérité
pour MESURER ; jamais le lot scellé).

  single : banc par primitive ISOLÉE ou masquée — type × densité × bruit × masquage ; gradient générique, inférence à type
           connu existante, estimateurs spécialisés (courbes), recherche directe, raffinement, consensus, oracles partiels,
           identifiabilité de la vraie primitive
  sens   : carte de sensibilité — UN paramètre d'UN vrai atome perturbé → dégradation fonctionnelle → récupération finale
  birth  : un vrai atome retiré → naissance par gain de plancher (nouvelle) contre recherche sur le résidu (ancienne) ;
           fausses naissances sur ensembles complets ; diagnostic de polarité (anti-disque)
  chain  : A atomes du pipeline gelé, B flat_scene_k3, C haute précision — MÊME aval fort ; régimes standard, rich, dense
  ood    : régime `hard` (160 observations, 10 % de bruit), jamais utilisé en mise au point

Tâches : clés 1100–1131 (jamais utilisées ; mise au point : clés 1090–1099 uniquement).
"""
from __future__ import annotations

import math
import random
import resource
import time
from typing import List, Optional, Sequence

import torch

from core.boundary_program import BNode, predict
from core.deep_fit import DT
from core.deep_hp import DEFAULTS, IDENT, assemble, birth, consensus, describe, extract_circles, extract_lines, families, improve_set, joint_refine, recover, strip, type_name
from core.deep_lpr_exp import _acc, _name, _regions, _row, atom_visibility, frozen_run

DEPTHS = (1, 2, 3, 4, 5, 6)
REGIMES = ("standard", "rich", "dense")
DENSE = (2400, 0.02)                 # régime dense pré-enregistré : (observations, bruit)
OOD_REGIME = "hard"
PRIMARY = ("A", "B", "C")
PREREGISTERED_SECONDARY = ("B5",)
C_VARIANTS = {"C": {}, "C_pool": {"refine": False, "births": 0}, "C_nocurves": {"curves": False}, "C_norefine": {"refine": False}, "C_nobirth": {"births": 0}, "C_cv": {"validate": "cv"}, "C_k1": {"k": 1}, "C_k3": {"k": 3}, "C_nojoint": {"joint": False}}
HISTORICAL = ("A_full", "B_full", "C_prev")
KINDS = ("half", "band", "disc", "antidisc", "ring", "sector")
DENSITIES = (150, 300, 600, 2400)
NOISES = (0.0, 0.05)
MASKS = (0.0, 0.3, 0.6)             # visibilité forte, moyenne, faible
DELTAS = (0.02, 0.05, 0.10, 0.20)
PERTURB = ("center", "radius", "width", "orientation", "offset", "direction", "half_width")
REL_GOOD, REL_FINE = 0.10, 0.05     # erreur fonctionnelle RELATIVE (désaccord / aire de la primitive), indépendante de la polarité
GOOD = 0.99


def task_at(key, d, regime):
    """Même MONDE que `down_task(key, d, ·)` quel que soit le régime : les régimes sont appariés sur la vraie structure."""
    from core.deep_experiments import down_task
    if regime != "dense": return down_task(key, d, regime)
    from core.deep_world import NOISE_KINDS, SPLIT_BASE, DeepTask, NoiseSpec, corrupt, make_world, obs_region
    from core.robust_world import SENSORS
    s = SPLIT_BASE["dev"] + 11000 + int(key) * 10 + d; w = make_world(s, d, "in_class", False); n, rate = DENSE; g = torch.Generator().manual_seed(s + 1); X = obs_region(n, g); t = torch.rand(n, generator=g); spec = NoiseSpec(rate, NOISE_KINDS[(int(key) + d) % 3])
    return DeepTask(w, X, corrupt(w, X, t, SENSORS["standard"], spec, g), t, spec, "dense", s, "dev")


def _probe():
    from core.deep_partial_exp import _probe as p
    return p()


def rel_error(true: BNode, hyp: BNode, P: torch.Tensor) -> float:
    """Erreur fonctionnelle relative : part de désaccord rapportée à l'aire (minoritaire) de la vraie primitive, la
    polarité étant libre (la composition dispose du complément). 0 = identique ; ≥ 1 = aucune ressemblance."""
    a = predict(true, P); d = float((a != predict(hyp, P)).float().mean()); d = min(d, 1 - d); m = float(a.float().mean()); return d / max(1e-6, min(m, 1 - m))


def _same_family(a: BNode, p: BNode) -> bool: return _name(a).replace("not_", "").replace("norm.ge", "norm.le") == _name(p).replace("not_", "").replace("norm.ge", "norm.le")


def family_errors(true: BNode, hyp: BNode) -> dict:
    """Erreur absolue par famille de paramètres ; vide si les deux primitives ne sont pas de même géométrie."""
    from core.deep_partial_exp import param_errors
    if not _same_family(true, hyp): return {}
    e = param_errors(strip(true), strip(hyp)); out = {}
    for k, v in e.items():
        f = "radius" if k.startswith("radius") else k; out[f] = max(out.get(f, 0.0), float(v))
    return out


def atom_precision(programs: Sequence[BNode], task, P=None) -> List[dict]:
    """Par vrai atome : meilleure erreur relative parmi `programs`, et erreurs de paramètres du meilleur candidat de même
    géométrie."""
    from core.deep_reachability import leaves
    P = _probe() if P is None else P; progs = [strip(p) for p in programs]; out = []
    for a in leaves(task.world.truth):
        rs = [rel_error(a, p, P) for p in progs]; best = min(rs, default=9.0); same = [(r, p) for r, p in zip(rs, progs) if _same_family(a, p)]; q = min(same, key=lambda z: z[0])[1] if same else None
        out.append({"type": _name(a), "rel": best, "same_rel": min((r for r, _ in same), default=9.0), "errors": family_errors(a, q) if q is not None else {}})
    return out


def _precision_extra(pool: Sequence[BNode], kept: Sequence[BNode], task, P=None) -> dict:
    P = _probe() if P is None else P; a, b = atom_precision(pool, task, P), atom_precision(kept, task, P); n = len(a); vis = atom_visibility(task)
    return {"pool_rel10": sum(x["rel"] <= REL_GOOD for x in a) / n, "pool_rel05": sum(x["rel"] <= REL_FINE for x in a) / n, "pool_all_rel10": all(x["rel"] <= REL_GOOD for x in a), "kept_rel10": sum(x["rel"] <= REL_GOOD for x in b) / n, "kept_all_rel10": all(x["rel"] <= REL_GOOD for x in b),
            "atoms_hp": [{"type": x["type"], "visible": v, "rel_pool": x["rel"], "rel_kept": z["rel"], "errors": x["errors"]} for x, z, v in zip(a, b, vis)], "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}


# ================================================================ chaînes complètes

def _chain_row(task, name, rec, regions, t0, extra=None) -> dict:
    H = rec.hypotheses; pool = [h.program for h in H] + [a for h in H for a in h.alternatives]; kept = rec.chosen.programs if rec.chosen else []; return _row(task, name, rec, regions, t0, {**_precision_extra(pool, kept, task), **(extra or {})})


def chain_rows(task, seed=0, arms: Optional[Sequence[str]] = None, historical=True) -> List[dict]:
    """Chaînes primaires (même aval fort), bras secondaire pré-enregistré, variantes de C, chaînes historiques."""
    from core.deep_e2e import discover
    from core.deep_lpr import recover as lpr_recover, strong_downstream
    regions = _regions(task); X, y = task.X, task.y; rows = []; want = lambda a: arms is None or a in arms
    if want("A"):
        t0 = time.time(); fprog, fatoms, fpool = frozen_run(task, seed); tf = time.time() - t0; rows.append(_chain_row(task, "A", strong_downstream(fatoms, X, y), regions, t0)); (F, lF) = regions[0]; cb = max(float((predict(p, F) == lF).float().mean()) for p in fpool)
        rows.append({"experiment": "chain", "arm": "A_full", **_acc(fprog, regions), "candidate_097": cb >= 0.97, "candidate_099": cb >= GOOD, "candidate_best": cb, "seconds": round(tf, 2), "literals": len(fatoms)})
    for name, k in (("B", 3), ("B5", 5)):
        if want(name): t0 = time.time(); rows.append(_chain_row(task, name, strong_downstream(discover(X, y, k, seed), X, y), regions, t0))
    if historical and want("B_full"):
        from core.deep_fit import FitData
        from core.deep_fragments import flat_types, literals
        from core.deep_partial import observe
        from core.deep_search import DeepSearch
        t0 = time.time(); lits = literals(flat_types(observe(X, y), 3, seed)); out = DeepSearch("factorized", budget=150.0, seed=seed, max_depth=6, fact_literals=lits, fact_residual=False, validated=None).run(FitData.build(X, y, 0.10)); (F, lF) = regions[0]; cb = max(float((predict(h.program, F) == lF).float().mean()) for h in out.all.values())
        rows.append({"experiment": "chain", "arm": "B_full", **_acc(out.best().program, regions), "candidate_097": cb >= 0.97, "candidate_099": cb >= GOOD, "candidate_best": cb, "seconds": round(time.time() - t0, 2), "literals": len(lits)})
    if historical and want("C_prev"): t0 = time.time(); rows.append(_chain_row(task, "C_prev", lpr_recover(X, y, seed), regions, t0))
    for name, cfg in C_VARIANTS.items():
        if not want(name): continue
        t0 = time.time(); R = recover(X, y, seed, cfg); im = R.improvements
        rows.append(_chain_row(task, name, R.recovery, regions, t0, {"hp_pool": R.stats["pool"], "hp_added": R.stats["added"], "hp_committed": R.stats["committed_sets"], "hp_births": R.stats["births"], "hp_evaluations": R.stats["evaluations"], "hp_cv_before": im[0].cv_before if im else None, "hp_cv_after": im[0].cv_after if im else None}))
    return rows


# ================================================================ banc par primitive

def _direct(sk_type: str, X, y, seed, k=5) -> List[BNode]:
    """Recherche DIRECTE de la primitive entière dans son espace de paramètres (aucune décomposition en courbes)."""
    from core.deep_fragments import SKELETONS
    from core.deep_hp import _climb, evaluate_batch, floor_batch, sample_params
    from core.deep_partial import observe
    g = torch.Generator().manual_seed(seed); Xd = X.to(DT)[None]; n = len(y); others = torch.zeros(n, dtype=torch.long); sk = SKELETONS[sk_type]; C = sample_params(sk_type, 3000, observe(X, y).points, g); v = floor_batch(evaluate_batch(sk, C, Xd), others, y, 1); out = []
    for i in v.argsort()[:k].tolist(): th, f, _ = _climb(sk, C[i], others, Xd, y, 1, g, 6, 48, 0.12); out.append((f, sk.with_params(th.tolist())))
    return [p for _, p in sorted(out, key=lambda z: z[0])]


def _methods(sc, seed) -> dict:
    """Méthode → (hypothèses classées, équivalents d'évaluation). Type CONNU pour toutes."""
    from core.deep_hp import load_config
    from core.deep_partial import infer, observe
    from core.deep_partial_exp import estimate
    X, y = sc.X, sc.y; ob = observe(X, y); c = load_config(); sk = sc.target.skeleton(); kind = sc.kind; out = {}
    def timed(f):
        t0 = time.time()
        try: r = f()
        except Exception as e: r = []; out.setdefault("_errors", []).append(repr(e)[:80])
        return r, time.time() - t0
    out["gradient"] = timed(lambda: estimate(sc, "global_target", seed)[0]); out["contrast"] = timed(lambda: estimate(sc, "global_contrast", seed)[0]); out["flat"] = timed(lambda: [h.program for h in infer(sk, ob, 5, seed=seed, regions=False)])
    def curves(joint=True):
        L = extract_lines(ob, c["lines"], seed, c["band_factor"], c["cos_normal"]); Cc = extract_circles(ob, c["circles"], seed, c["band_factor"], c["cos_normal"]); return [p for p, _, src in sorted(assemble(ob, L, Cc, 5, joint), key=lambda z: -z[1]) if src == "curves:" + kind][:5]
    out["curves"] = timed(curves)
    if kind in ("band", "ring"): out["curves_independent"] = timed(lambda: curves(False))
    out["direct"] = timed(lambda: _direct("disc" if kind == "antidisc" else kind, X, y, seed))
    base = out["curves"][0] or out["flat"][0]
    def refined():
        R = [joint_refine([p], X, y, seed + i)[0][0] for i, p in enumerate(base[:5])]; e = [float((predict(p, X.float()) != y).float().mean()) for p in R]; return [p for _, _, p in sorted(zip([min(v, 1 - v) for v in e], range(len(R)), R), key=lambda z: z[:2])]
    out["refined"] = timed(refined)
    out["consensus"] = timed(lambda: consensus(out["refined"][0][:1], X, y, seed, validate=False)[0] if out["refined"][0] else [])
    for g in ("center", "orientation", "size"): out["given_" + g] = timed(lambda g=g: estimate(sc, "given_" + g, seed)[0])
    out["truth_refined"] = timed(lambda: joint_refine([sc.target], X, y, seed, margin=0.0)[0])
    return out


def single_rows(key, kind: str, seed=0) -> List[dict]:
    from core.deep_partial_exp import masked_scene, true_visible_fraction
    P = _probe(); rows = []
    for n in DENSITIES:
        for noise in NOISES:
            for mask in MASKS:
                sc = masked_scene(20000 + int(key), kind, mask, noise, n); M = _methods(sc, seed); errs = M.pop("_errors", None); E = describe([sc.target], sc.X, sc.y, seed)[0]; vis = true_visible_fraction(sc)
                for name, (progs, dt) in M.items():
                    rs = [rel_error(sc.target, p, P) for p in progs]; e = family_errors(sc.target, progs[0]) if progs else {}
                    rows.append({"experiment": "single", "arm": name, "kind": kind, "n": n, "noise": noise, "mask": mask, "visible": vis, "n_hyp": len(progs), "rel_top1": rs[0] if rs else 9.0, "rel_top3": min(rs[:3], default=9.0), "rel_top5": min(rs[:5], default=9.0), "errors": e,
                                 "obs_error": (lambda v: min(v, 1 - v))(float((predict(progs[0], sc.X.float()) != sc.y).float().mean())) if progs else None, "seconds": round(dt, 3), "identifiability": E.identifiability, "uncertainty": E.by_family(), "stability": E.stability, "method_errors": errs if name == "curves" else None})
    return rows


# ================================================================ carte de sensibilité

def perturb(a: BNode, family: str, delta: float, rng: random.Random) -> Optional[BNode]:
    """UN vrai atome, UNE famille de paramètres, amplitude `delta` (signe tiré). `orientation` tourne la droite autour de
    son point le plus proche du centre de la zone observée (pas autour de l'origine) ; `width` écarte les deux bords."""
    a = strip(a); fam = families(a); th = list(a.param_vector()); sgn = rng.choice((-1.0, 1.0)); idx = [i for i, f in enumerate(fam) if f == family]
    if family == "width":
        idx = [i for i, f in enumerate(fam) if f in ("offset", "radius")]
        if len(idx) != 2: return None
        th[idx[0]] -= sgn * delta / 2; th[idx[1]] += sgn * delta / 2; return a.skeleton().with_params(th)
    if not idx: return None
    if family == "center": t = rng.uniform(-math.pi, math.pi); th[idx[0]] += delta * math.cos(t); th[idx[1]] += delta * math.sin(t)
    elif family == "orientation":
        i = idx[0]; n0 = (math.cos(th[i]), math.sin(th[i])); c0 = (-0.9, 0.0); offs = [j for j, f in enumerate(fam) if f == "offset"]; mid = sum(th[j] for j in offs) / len(offs); s = n0[0] * c0[0] + n0[1] * c0[1] - mid; p0 = (c0[0] - s * n0[0], c0[1] - s * n0[1]); th[i] += sgn * delta; n1 = (math.cos(th[i]), math.sin(th[i])); shift = (n1[0] * p0[0] + n1[1] * p0[1]) - mid
        for j in offs: th[j] += shift
    else:
        for i in idx: th[i] += sgn * delta
    return a.skeleton().with_params(th)


def sens_rows(task, seed=0, max_atoms=3) -> List[dict]:
    """parameter perturbation → functional degradation → structural recovery degradation. Les AUTRES atomes restent exacts ;
    aucune finalisation (on mesure la composition, pas l'ajustement)."""
    from core.deep_lpr import strong_downstream
    from core.deep_reachability import leaves
    A = leaves(task.world.truth); regions = _regions(task); P = _probe(); rng = random.Random(seed * 7919 + len(A)); base = _acc(strong_downstream([(a, 1.0, "true") for a in A], task.X, task.y, {"fit": "none"}).program, regions); rows = []
    for i in [(seed + j) % len(A) for j in range(min(max_atoms, len(A)))]:
        for fam in PERTURB:
            for dl in DELTAS:
                q = perturb(A[i], fam, dl, rng)
                if q is None: continue
                R = strong_downstream([(q if j == i else a, 1.0, "true") for j, a in enumerate(A)], task.X, task.y, {"fit": "none"}); ac = _acc(R.program, regions)
                rows.append({"experiment": "sens", "arm": "perturb", "kind": _name(A[i]), "family": fam, "delta": dl, "atom_rel": rel_error(A[i], q, P), "base_functional": base["functional"], "base_accord": base["accord"], "functional": ac["functional"], "functional_097": ac["functional_097"], "accord": ac["accord"], "n_true": len(A)})
    return rows


# ================================================================ naissance d'une primitive manquante

def birth_rows(task, seed=0) -> List[dict]:
    from core.deep_lpr import recover as lpr_recover, strong_downstream
    from core.deep_lpr_exp import jitter
    from core.deep_partial import observe
    from core.deep_partial_exp import functional
    from core.deep_reachability import leaves
    A = leaves(task.world.truth); X, y = task.X, task.y; regions = _regions(task); P = _probe(); M = observe(X, y).points; rows = []; tag = {"experiment": "birth", "n_true": len(A)}
    t0 = time.time(); im = improve_set(A, X, y, seed, {"refine": False}, M); rows.append({**tag, "arm": "complete", "false_birth": any(p.startswith("birth") for p in im.provenance), "proposed_birth": any(b.get("accepted") for b in im.births), "committed": im.committed, "seconds": round(time.time() - t0, 2)})
    if len(A) < 2: return rows
    k = seed % len(A); miss = A[k]; rest0 = [a for j, a in enumerate(A) if j != k]; mt = _name(miss)
    for ctx in ("exact", "noisy"):
        rest = rest0 if ctx == "exact" else jitter(rest0, 0.05, random.Random(seed + 5)); before = _acc(strong_downstream([(a, 1.0, "true") for a in rest], X, y, {"fit": "none"}).program, regions); common = {**tag, "context": ctx, "missing_type": mt, "before_functional": before["functional"], "before_accord": before["accord"]}
        for arm, cfg in (("new", {}), ("new_nomorph", {"morphology": False}), ("new_unvalidated", {"validate": False}), ("new_onebirth", {"births": 1, "validate": False, "refine": False})):
            t0 = time.time(); im = improve_set(rest, X, y, seed, cfg, M); born = [p for p, s in zip(im.programs, im.provenance) if s.startswith("birth")]; rs = [rel_error(miss, p, P) for p in born]; j = min(range(len(rs)), key=lambda i: rs[i]) if rs else None
            after = _acc(strong_downstream([(a, 1.0, "set") for a in im.programs], X, y, {"fit": "none"}).program, regions) if im.programs else before
            rows.append({**common, "arm": arm, "born": len(born), "committed": im.committed, "rel": rs[j] if rs else 9.0, "found_rel10": bool(rs) and rs[j] <= REL_GOOD, "found_free": any(max(functional(miss, p, P), 1 - functional(miss, p, P)) >= 0.95 for p in born), "found_old_measure": any(functional(miss, p, P) >= 0.95 for p in born),
                         "same_type": bool(rs) and _same_family(miss, born[j]), "errors": family_errors(miss, born[j]) if rs else {}, "morphology": im.births[0]["morphology"] if im.births else None, "functional": after["functional"], "accord": after["accord"], "evaluations": im.evaluations, "seconds": round(time.time() - t0, 2)})
        t0 = time.time(); R = lpr_recover(X, y, seed, {"fit": "none", "iters": 2}, [(a, 1.0, "true") for a in rest]); ac = _acc(R.program, regions); known = {repr(strip(a)) for a in rest}; new = [strip(h.program) for h in R.hypotheses if repr(strip(h.program)) not in known]; alt = [strip(q) for h in R.hypotheses for q in h.alternatives]; rs = [rel_error(miss, p, P) for p in new]
        rows.append({**common, "arm": "old", "born": len(new), "committed": True, "rel": min(rs, default=9.0), "found_rel10": min(rs, default=9.0) <= REL_GOOD, "found_free": any(max(functional(miss, p, P), 1 - functional(miss, p, P)) >= 0.95 for p in new), "found_old_measure": any(functional(miss, p, P) >= 0.95 for p in new),
                     "found_in_alternatives": any(functional(miss, p, P) >= 0.95 for p in alt), "same_type": False, "errors": {}, "morphology": None, "functional": ac["functional"], "accord": ac["accord"], "evaluations": None, "seconds": round(time.time() - t0, 2)})
    return rows


# ================================================================ sous-unités AWS

def subunits(key):
    from core.deep_experiments import _row as wrap
    k = int(key); out = []
    for d in DEPTHS:
        for reg in REGIMES:
            T = lambda d=d, reg=reg: task_at(key, d, reg); tag = {"task_id": f"{key}_{d}_{reg}"}
            out.append((f"chain_d{d}_{reg}", (lambda T=T, tag=tag, reg=reg: [wrap(T(), {**tag, **r}) for r in chain_rows(T(), k, historical=reg != "dense")])))
            if reg != "dense": out.append((f"diag_d{d}_{reg}", (lambda T=T, tag=tag: [wrap(T(), {**tag, **r}) for r in sens_rows(T(), k) + birth_rows(T(), k)])))
        T = lambda d=d: task_at(key, d, OOD_REGIME); tag = {"task_id": f"{key}_{d}_{OOD_REGIME}"}
        out.append((f"ood_d{d}", (lambda T=T, tag=tag: [wrap(T(), {**tag, **r, "experiment": "ood"}) for r in chain_rows(T(), k, arms=("B", "C", "C_pool"), historical=False)])))
    for kind in KINDS: out.append((f"single_{kind}", (lambda kind=kind: [{"task_id": f"{key}_{kind}_{r['n']}_{r['noise']}_{r['mask']}", "depth": 1, "regime": "single", **r} for r in single_rows(key, kind, k)])))
    return out

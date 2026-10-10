"""Expériences du sous-jalon Latent Primitive Recovery — module d'ÉVALUATION (dev uniquement ; il voit la vérité pour
MESURER ; jamais le lot scellé).

  chain    : A atomes du pipeline gelé, B flat_scene_k3, C LatentPrimitiveSetRecovery — tous dans le MÊME aval fort ;
             récupération de l'ENSEMBLE des vraies primitives, précision des paramètres, aval
  sens     : erreur de paramètre → récupération de la composition, par type de primitive et par famille de paramètres
  fp       : vrais atomes + 1 / 3 / 5 / 10 faux ; défaut de l'ancienne réduction
  complete : signal de complétude ; recherche d'une primitive manquante par le résidu ; hors-classe (exploratoire)

Tâches : clés 1000–1031 de `down_task`, profondeurs 1 à 6 (jamais utilisées ; mise au point : clés 1090–1099).
"""
from __future__ import annotations

import math
import random
import time
from typing import List, Optional, Sequence

import numpy as np
import torch

from core.boundary_program import BNode, predict
from core.deep_compose import to_program
from core.deep_e2e import compose, discover, hypotheses_from, reduce
from core.deep_fit import FitData
from core.deep_lpr import completeness, load_config, pool, propose_missing, recover, select_sets, strong_downstream, utilities

DEPTHS = (1, 2, 3, 4, 5, 6)
REGIMES = ("standard", "rich")
PRIMARY = ("A", "B", "C")
PREREGISTERED_SECONDARY = ("B5",)
C_VARIANTS = {"C": {}, "C_noloop": {"iters": 0}, "C_nopairs": {"pairs": False}, "C_norefine": {"refine": False}, "C_k3": {"k": 3}, "C_flatonly": {"sources": ("flat",)}}
HISTORICAL = ("A_full", "B_full")
SIGMAS = (0.02, 0.05, 0.10, 0.20)
FAMILIES = ("center", "radius", "orientation", "offset", "width")
ATOM_KINDS = ("dot.le", "dot.in", "norm.le", "norm.ge", "norm.in", "angle.in")
FALSE_COUNTS = (0, 1, 3, 5, 10)
OOC = ("primitive_absent", "composition_absent", "relation_absent")
GOOD = 0.99


def task_at(key, d, regime):
    from core.deep_experiments import down_task
    return down_task(key, d, regime)


def ooc_task(key, category):
    from core.deep_world import make_task
    return make_task(14000 + int(key) * 10 + OOC.index(category), None, "standard", category=category, split="dev")


def _regions(task):
    from core.deep_world import full_region, obs_region, out_region
    g = lambda s: torch.Generator().manual_seed(s); F, I, O = full_region(4000, g(11)), obs_region(2000, g(11)), out_region(2000, g(13)); return (F, task.world.label(F, 1.0)), (I, task.world.label(I, 1.0)), (O, task.world.label(O, 1.0))


def _acc(prog: Optional[BNode], regions) -> dict:
    if prog is None: return {"accord": 0.0, "accord_out": 0.0, "functional": False, "functional_097": False}
    (F, lF), _, (O, lO) = regions; a = float((predict(prog, F) == lF).float().mean()); return {"accord": a, "accord_out": float((predict(prog, O) == lO).float().mean()), "functional": a >= GOOD, "functional_097": a >= 0.97}


# ================================================================ récupération de l'ENSEMBLE des vraies primitives

def atom_visibility(task) -> List[float]:
    """Part de la frontière de chaque vrai atome (dans la zone observée) qui est VISIBLE dans la vraie composition."""
    from core.deep_partial import leaf_paths, render, visible_mask
    from core.deep_partial_exp import BOX
    tr = task.world.truth; out = []
    for path, leaf in leaf_paths(tr):
        P = render(leaf, n=160, box=BOX); out.append(float(visible_mask(tr, path, P).float().mean()) if len(P) else 0.0)
    return out


def set_recovery(programs: Sequence[BNode], task) -> dict:
    """Trois niveaux d'équivalence, par vrai atome puis pour l'ENSEMBLE :
      exact       toutes les frontières de l'atome dans la tolérance de paramètres (équivalence de paramètres)
      functional  accord fonctionnel ≥ 0,95 sur la région observée, quel que soit le paramétrage
      strict      accord fonctionnel ≥ 0,99
    `all_*` : toutes les vraies primitives présentes ; `most_*` : au plus une absente."""
    from core.deep_partial_exp import _probe, functional, param_errors, recovered
    from core.deep_reachability import leaves
    tl = leaves(task.world.truth); P = _probe(); progs = [p.children[0] if p.kind == "not" else p for p in programs]; rows = []
    for a in tl:
        fs = [functional(a, p, P) for p in progs]; bf = max(fs, default=0.0); ex = any(atom_name_eq(a, p) and recovered(a, p) for p in progs); same = [(f, p) for f, p in zip(fs, progs) if atom_name_eq(a, p) and f >= 0.95]; best = max(same, key=lambda z: z[0])[1] if same else None
        e = param_errors(a, best) if best is not None else {}                     # l'erreur de paramètre n'a de sens qu'entre primitives de MÊME type
        rows.append({"type": _name(a), "exact": ex, "functional": bf >= 0.95, "strict": bf >= 0.99, "best": bf, "param_error": (sum(e.values()) / len(e)) if e else None, "errors": e})
    n = len(tl); out = {"n_true": n, "atoms": rows}
    for k in ("exact", "functional", "strict"):
        h = sum(r[k] for r in rows); out[f"recall_{k}"] = h / n; out[f"all_{k}"] = h == n; out[f"most_{k}"] = h >= n - 1
    pe = [r["param_error"] for r in rows if r["param_error"] is not None]; out["param_error"] = sum(pe) / len(pe) if pe else None; return out


def _name(a):
    from core.deep_primitives import atom_name
    return atom_name(a)


def atom_name_eq(a, p): return _name(a).replace("not_", "") == _name(p).replace("not_", "")


def _row(task, name, rec, regions, t0, extra=None) -> dict:
    """Ligne commune : aval + récupération de l'ensemble (vivier, ensemble retenu, union des trois meilleurs ensembles)."""
    H = rec.hypotheses; allp = [h.program for h in H] + [a for h in H for a in h.alternatives]; kept = rec.chosen.programs if rec.chosen else []; top3 = [h.program for h in rec.sets.union(3)] if rec.sets else []
    sp, sk, s3 = set_recovery(allp, task), set_recovery(kept, task), set_recovery(top3, task); vis = atom_visibility(task); row = {"experiment": "chain", "arm": name, **_acc(rec.program, regions)}
    for tag, s in (("pool", sp), ("kept", sk), ("top3", s3)):
        row.update({f"{tag}_{k}": s[k] for k in ("recall_exact", "recall_functional", "recall_strict", "all_exact", "all_functional", "all_strict", "most_functional", "most_strict", "param_error")})
    row["atoms"] = [{"type": a["type"], "visible": v, "pool": a["functional"], "pool_strict": a["strict"], "kept": b["functional"], "kept_strict": b["strict"], "param_error": a["param_error"], "errors": a["errors"]} for a, b, v in zip(sp["atoms"], sk["atoms"], vis)]
    R = rec.result; units = getattr(R, "units", None); cand = 0.0
    if R is not None and R.ranked and units: (F, lF) = regions[0]; cand = max(float((predict(to_program(c.tree, units), F) == lF).float().mean()) for c in R.ranked[:20])
    row.update({"candidate_097": cand >= 0.97, "candidate_099": cand >= GOOD, "candidate_best": cand, "n_true": sp["n_true"], "literals": rec.stats.get("literals"), "hypotheses": rec.stats.get("hypotheses"), "kept": rec.stats.get("kept"), "sets": rec.stats.get("sets"), "considered": rec.stats.get("considered_total", rec.stats.get("considered")),
                "false_kept": sum(1 for p in kept if not set_recovery([p], task)["recall_functional"] > 0), "state": rec.state, "excess": rec.completeness.excess if rec.completeness else None, "gap": rec.completeness.gap if rec.completeness else None,
                "fit_committed": bool(rec.finalization.committed) if rec.finalization else False, "iterations": rec.stats.get("iterations", 1), "seconds": round(time.time() - t0, 2), **(extra or {})})
    if rec.log: row["loop"] = [{k: l.get(k) for k in ("iteration", "pool", "errors", "excess", "action", "proposed", "refined")} for l in rec.log]
    row["n_pairs"] = sum(1 for st in (rec.chosen.trace if rec.chosen else []) if st.get("pair")); return row


def frozen_run(task, seed=0):
    """Pipeline gelé complet, une seule exécution : son programme final (historique) et ses atomes (source A)."""
    from core.deep_reachability import leaves
    from core.deep_search import DeepSearch
    out = DeepSearch("factorized", budget=150.0, seed=seed, max_depth=6).run(FitData.build(task.X, task.y, 0.10)); hs = sorted(out.all.values(), key=lambda h: -h.evidence); seen = set(); L = []
    for r, h in enumerate(hs):
        for l in leaves(h.program):
            if repr(l) not in seen: seen.add(repr(l)); L.append((l, float(len(hs) - r), "frozen"))
    return out.best().program, L, [h.program for h in hs]


def chain_rows(task, seed=0) -> List[dict]:
    """Les trois chaînes primaires (même aval fort), le bras secondaire pré-enregistré, les variantes de C, et les deux
    chaînes historiques complètes."""
    from core.deep_fragments import flat_types, literals
    from core.deep_partial import observe
    from core.deep_search import DeepSearch
    regions = _regions(task); X, y = task.X, task.y; rows = []
    t0 = time.time(); fprog, fatoms, fpool = frozen_run(task, seed); tf = time.time() - t0; rows.append(_row(task, "A", strong_downstream(fatoms, X, y), regions, t0))
    a = _acc(fprog, regions); (F, lF) = regions[0]; cb = max(float((predict(p, F) == lF).float().mean()) for p in fpool); rows.append({"experiment": "chain", "arm": "A_full", **a, "candidate_097": cb >= 0.97, "candidate_099": cb >= GOOD, "candidate_best": cb, "seconds": round(tf, 2), "literals": len(fatoms)})
    for name, k in (("B", 3), ("B5", 5)): t0 = time.time(); rows.append(_row(task, name, strong_downstream(discover(X, y, k, seed), X, y), regions, t0))
    t0 = time.time(); lits = literals(flat_types(observe(X, y), 3, seed)); out = DeepSearch("factorized", budget=150.0, seed=seed, max_depth=6, fact_literals=lits, fact_residual=False, validated=None).run(FitData.build(X, y, 0.10)); cb = max(float((predict(h.program, F) == lF).float().mean()) for h in out.all.values())
    rows.append({"experiment": "chain", "arm": "B_full", **_acc(out.best().program, regions), "candidate_097": cb >= 0.97, "candidate_099": cb >= GOOD, "candidate_best": cb, "seconds": round(time.time() - t0, 2), "literals": len(lits)})
    for name, cfg in C_VARIANTS.items(): t0 = time.time(); rows.append(_row(task, name, recover(X, y, seed, cfg), regions, t0))
    return rows


# ================================================================ sensibilité : erreur de paramètre → récupération

def _family_index(a: BNode) -> dict:
    """Positions des paramètres d'un atome, par famille."""
    from core.deep_structure import _leaf
    l = _leaf(a); off = 2 if a.kind == "translate" else 0; out = {"center": list(range(off))}
    if l.feat == "norm": out["radius"] = [off + i for i in range(len(l.params))]
    elif l.feat == "dot": out["orientation"] = [off]; out["offset"] = [off + i for i in range(1, len(l.params))]
    else: out["orientation"] = [off]; out["width"] = [off + 1]
    return out


def jitter(atoms: Sequence[BNode], sigma: float, rng: random.Random, kind: Optional[str] = None, family: Optional[str] = None) -> List[BNode]:
    """Bruit gaussien sur les paramètres des atomes du type `kind` (tous types si None), limité à la famille `family`."""
    out = []
    for a in atoms:
        v = list(a.param_vector()); fi = _family_index(a); idx = [i for f, ii in fi.items() for i in ii if family is None or f == family] if (kind is None or _name(a).replace("not_", "") == kind) else []
        for i in idx: v[i] += rng.gauss(0.0, sigma)
        out.append(a.skeleton().with_params(v))
    return out


def sens_rows(task, seed=0) -> List[dict]:
    """Vrais atomes, dont une partie est bruitée ; composition exhaustive sur classes, sélection par erreurs, aucun
    ajustement. Isole la sensibilité de la COMPOSITION à l'erreur de paramètre."""
    from core.deep_reachability import leaves
    tl = leaves(task.world.truth); regions = _regions(task); kinds = sorted({_name(a).replace("not_", "") for a in tl}); out = []
    def run(atoms): R = compose(atoms, task.X, task.y, "errors", complements=False); return _acc(to_program(R.selected.tree, R.units), regions)
    base = run(tl)
    for s in SIGMAS:
        for mode, key in [("all", None)] + [("kind", k) for k in kinds] + [("family", f) for f in FAMILIES]:
            rng = random.Random(seed * 1009 + int(s * 1000) + hash((mode, key)) % 997); a = jitter(tl, s, rng, key if mode == "kind" else None, key if mode == "family" else None)
            if mode == "family" and all(repr(x) == repr(z) for x, z in zip(a, tl)): continue
            r = run(a); out.append({"experiment": "sens", "sigma": s, "mode": mode, "key": key or "all", "accord": r["accord"], "functional": r["functional"], "functional_097": r["functional_097"], "base_functional": base["functional"], "n_true": len(tl),
                                   "n_affected": sum(repr(x) != repr(z) for x, z in zip(a, tl))})
    return out


# ================================================================ tolérance aux faux atomes, défaut de l'ancienne réduction

def fp_rows(task, seed=0) -> List[dict]:
    from core.deep_reachability import leaves
    from core.deep_world import ATOM_TYPES, atom
    tl = leaves(task.world.truth); regions = _regions(task); rng = random.Random(seed * 7919 + 5); fakes = [atom(rng, rng.choice(ATOM_TYPES)) for _ in range(max(FALSE_COUNTS))]; out = []
    for n in FALSE_COUNTS:
        disc = [(a, 1.0, "true") for a in tl] + [(a, 1.0, "false") for a in fakes[:n]]; random.Random(seed + n).shuffle(disc); rec = strong_downstream(disc, task.X, task.y, {"fit": "none"}); s = set_recovery(rec.chosen.programs if rec.chosen else [], task)
        out.append({"experiment": "fp", "arm": f"plus{n}", "engine": "sets", **_acc(rec.program, regions), "kept": rec.stats["kept"], "kept_recall": s["recall_strict"], "all_kept": s["all_strict"], "false_kept": rec.stats["kept"] - sum(a["strict"] for a in s["atoms"]), "n_true": len(tl)})
        if n == 0:                                                               # ancienne réduction (sous-jalon précédent) sur les mêmes vrais atomes : diagnostic du défaut
            H = hypotheses_from(disc, task.X); kept, _ = reduce(H, task.y, "greedy_floor", 6, 1.0); atoms = [h.program for h in kept]; R = compose(atoms, task.X, task.y, "bic") if atoms else None; so = set_recovery(atoms, task)
            out.append({"experiment": "fp", "arm": "plus0", "engine": "old_reduction", **_acc(to_program(R.selected.tree, R.units) if R else None, regions), "kept": len(atoms), "kept_recall": so["recall_strict"], "all_kept": so["all_strict"], "false_kept": 0, "n_true": len(tl)})
            U = utilities(H, task.y); out[-1]["conditional_only"] = sum(u.conditional_only for u in U); out[-2]["conditional_only"] = out[-1]["conditional_only"]
    return out


# ================================================================ complétude, primitive manquante, hors-classe

def complete_rows(task, seed=0) -> List[dict]:
    """Contrôlé : vrais atomes complets, puis un atome retiré. Signal de complétude ; la recherche par le résidu
    retrouve-t-elle l'atome retiré ?"""
    from core.deep_partial_exp import _probe, functional
    from core.deep_reachability import leaves
    tl = leaves(task.world.truth); regions = _regions(task); out = []; P = _probe()
    full = strong_downstream([(a, 1.0, "true") for a in tl], task.X, task.y, {"fit": "none"}); out.append({"experiment": "complete", "arm": "complete", **_acc(full.program, regions), "excess": full.completeness.excess if full.completeness else None, "gap": full.completeness.gap if full.completeness else None, "flag_incomplete": not (full.completeness and full.completeness.complete), "n_true": len(tl)})
    if len(tl) >= 2:
        missing = tl[seed % len(tl)]; rest = [(a, 1.0, "true") for a in tl if a is not missing]; before = strong_downstream(rest, task.X, task.y, {"fit": "none"}); after = recover(task.X, task.y, seed, {"fit": "none", "iters": 2}, rest)
        found = any(functional(missing, (h.program.children[0] if h.program.kind == "not" else h.program), P) >= 0.95 for h in after.hypotheses); kept = any(functional(missing, (p.children[0] if p.kind == "not" else p), P) >= 0.95 for p in (after.chosen.programs if after.chosen else []))
        out.append({"experiment": "complete", "arm": "missing1", **{f"before_{k}": v for k, v in _acc(before.program, regions).items()}, **_acc(after.program, regions), "excess": before.completeness.excess if before.completeness else None, "gap": before.completeness.gap if before.completeness else None,
                    "flag_incomplete": not (before.completeness and before.completeness.complete), "missing_type": _name(missing), "missing_found": found, "missing_kept": kept, "iterations": after.stats.get("iterations"), "state": after.state, "n_true": len(tl)})
    return out


def ooc_row(task, kind: str, seed=0) -> dict:
    r = recover(task.X, task.y, seed); regions = _regions(task); return {"experiment": "ooc", "arm": kind, "state": r.state, "excess": r.completeness.excess if r.completeness else None, "gap": r.completeness.gap if r.completeness else None, "accord": _acc(r.program, regions)["accord"], "iterations": r.stats.get("iterations")}


# ================================================================ sous-unités AWS

def subunits(key):
    from core.deep_experiments import _row as wrap
    k = int(key); out = []
    for d in DEPTHS:
        for reg in REGIMES:
            T = lambda d=d, reg=reg: task_at(key, d, reg); tag = {"task_id": f"{key}_{d}_{reg}"}
            out.append((f"chain_d{d}_{reg}", (lambda T=T, tag=tag: [wrap(T(), {**tag, **r}) for r in chain_rows(T(), k)])))
            out.append((f"diag_d{d}_{reg}", (lambda T=T, tag=tag: [wrap(T(), {**tag, **r}) for r in sens_rows(T(), k) + fp_rows(T(), k) + complete_rows(T(), k)])))
    for cat in OOC: out.append((f"ooc_{cat}", (lambda cat=cat: [{"task_id": f"{key}_{cat}", "depth": 0, "regime": "standard", **ooc_row(ooc_task(key, cat), cat, k)}])))
    return out

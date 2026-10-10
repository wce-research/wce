"""End-to-End Composition with Discovered Atoms — de primitives découvertes, imparfaites et parfois fausses, à une
composition sur classes d'équivalence.

    DiscoveredPrimitive → PrimitiveHypothesis → réduction → composition sur classes → finalisation validée → décision

Couche explicite entre découverte et composition :
  PrimitiveHypothesis   type, paramètres, confiance, incertitude, provenance, alternatives équivalentes — une
                        primitive découverte n'est jamais traitée comme une vérité
  reduce                une vingtaine de littéraux → au plus six atomes, par mécanismes compacts et comparables
  compose               moteur exact sur cellules (≤ 6 atomes) ; `wide_compose` : faisceau sur classes pour plus d'atomes ;
                        `hierarchical` : classes d'un premier groupe réutilisées comme unités d'un second niveau
  finalize              aucun ajustement, ou ajustement VALIDÉ (l'état initial est conservé ; commit seulement si le
                        critère observable progresse) ; l'ajustement forcé n'existe que comme bras de comparaison
  decide                KNOWN / UNKNOWN / NONE_OF_THE_ABOVE à partir de signaux explicites

Aucune vérité de monde ; aucun oracle de type, de paramètre, d'atome ou d'appartenance.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from core.boundary_program import BNode, predict
from core.deep_compose import OPS, CellSpace, CompositionResult, ObservationalEquivalenceClass, Tree, bottom_up, class_scores, order_by, profiled_ll, select, to_program, tree_atoms, tree_size
from core.deep_fit import DT, FitData, fit_batch, validated_fit
from core.deep_primitives import atom_name
from core.deep_structure import KNOWN, NONE_OF_THE_ABOVE, UNKNOWN

DEFAULTS = {"k": 3, "reduction": "greedy_floor", "max_atoms": 6, "min_gain": 4.0, "selector": "bic", "fit": "validated", "same": 0.97, "gap_unknown": 0.04, "gap_none": 0.10, "tie": 2.0}
REDUCTIONS = ("confidence", "dedupe", "family", "dominance", "greedy_floor")
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "deep_configs", "e2e.json")


def load_config() -> dict:
    """Configuration de la chaîne, réglée sur les tâches de MISE AU POINT uniquement ; valeurs par défaut sinon."""
    try: return {**DEFAULTS, **json.load(open(CONFIG_FILE))["config"]}
    except (OSError, ValueError, KeyError): return dict(DEFAULTS)
FITS = ("none", "validated", "forced")


# ================================================================ hypothèses de primitives

@dataclass
class PrimitiveHypothesis:
    """Une primitive DÉCOUVERTE, avec ce qu'on sait de sa fiabilité. `alternatives` : primitives observationnellement
    équivalentes (ou complémentaires) qu'elle représente — elles restent disponibles, elles ne sont pas des atomes de plus."""
    program: BNode
    type: str
    confidence: float               # score de découverte, ramené à [0, 1] dans la scène
    provenance: str
    signature: np.ndarray           # valeur sur les observations
    alternatives: List[BNode] = field(default_factory=list)
    uncertainty: Optional[float] = None       # désaccord fonctionnel moyen avec ses alternatives (None s'il n'y en a pas)

    @property
    def params(self): return self.program.param_vector()
    @property
    def n_alternatives(self): return len(self.alternatives)


def hypotheses_from(discovered: Sequence[Tuple[BNode, float, str]], X: torch.Tensor, same=0.97) -> List[PrimitiveHypothesis]:
    """(programme, score, provenance) → hypothèses. Deux primitives de même signature sur les observations, ou de
    signatures complémentaires (accord ≥ `same` ou ≤ 1 − `same`), sont UNE hypothèse : la mieux notée la représente, les
    autres deviennent ses alternatives. C'est le dédoublonnage et l'équivalence de primitives."""
    items = sorted(discovered, key=lambda d: -d[1]); top = max((s for _, s, _ in items), default=1.0) or 1.0; out: List[PrimitiveHypothesis] = []
    for p, s, src in items:
        sig = predict(p, X.float()).numpy().astype(bool); host = None
        for h in out:
            a = float((sig == h.signature).mean())
            if a >= same or a <= 1 - same: host = h; break
        if host is None: out.append(PrimitiveHypothesis(p, atom_name(p), max(0.0, s) / max(top, 1e-9), src, sig))
        else: host.alternatives.append(p); d = float((sig != host.signature).mean()); d = min(d, 1 - d); host.uncertainty = d if host.uncertainty is None else (host.uncertainty * (host.n_alternatives - 1) + d) / host.n_alternatives
    return out


# ================================================================ réduction avant composition

def floor_errors(sigs: Sequence[np.ndarray], y: np.ndarray) -> float:
    """Plus petit nombre d'erreurs atteignable par UNE fonction booléenne quelconque de ces atomes : étiquette
    majoritaire de chaque cellule. Ne dépend d'aucune structure."""
    if not sigs: return float(min(y.sum(), len(y) - y.sum()))
    code = np.zeros(len(y), dtype=np.uint64)
    for i, s in enumerate(sigs): code |= s.astype(np.uint64) << np.uint64(i)
    _, inv = np.unique(code, return_inverse=True); pos = np.bincount(inv, weights=y.astype(np.float64)); cnt = np.bincount(inv).astype(np.float64); return float(np.minimum(pos, cnt - pos).sum())


def reduce(hyps: Sequence[PrimitiveHypothesis], y: torch.Tensor, method="greedy_floor", max_atoms=6, min_gain=4.0, per_family=2, dominance=0.90) -> Tuple[List[PrimitiveHypothesis], dict]:
    """Une vingtaine d'hypothèses → au plus `max_atoms`. Mécanismes (comparables séparément) :
      confidence    les plus confiantes
      dedupe        idem — le dédoublonnage est déjà fait par `hypotheses_from` ; ce bras prend les premières SANS autre critère que l'ordre
      family        au plus `per_family` par type, puis les plus confiantes
      dominance     une hypothèse est écartée si une plus confiante lui est fonctionnellement proche (accord ≥ `dominance`)
      greedy_floor  sélection avant : à chaque pas, l'atome qui réduit le plus l'erreur plancher (étiquette majoritaire par
                    cellule) ; arrêt quand le gain passe sous `min_gain` erreurs — l'utilité pour la COMPOSITION décide"""
    H = sorted(hyps, key=lambda h: -h.confidence); yb = y.numpy().astype(bool); info = {"method": method, "input": len(H)}
    if method in ("confidence", "dedupe"): kept = H[:max_atoms]
    elif method == "family":
        cnt: Dict[str, int] = {}; kept = []
        for h in H:
            if cnt.get(h.type, 0) < per_family: kept.append(h); cnt[h.type] = cnt.get(h.type, 0) + 1
        kept = kept[:max_atoms]
    elif method == "dominance":
        kept = []
        for h in H:
            if all(max(float((h.signature == o.signature).mean()), 1 - float((h.signature == o.signature).mean())) < dominance for o in kept): kept.append(h)
            if len(kept) >= max_atoms: break
    elif method == "greedy_floor":
        kept = []; cur = floor_errors([], yb); trace = [cur]
        while len(kept) < max_atoms:
            best = None
            for h in H:
                if any(h is o for o in kept): continue
                f = floor_errors([o.signature for o in kept] + [h.signature], yb)
                if best is None or f < best[0] - 1e-9: best = (f, h)
            if best is None or cur - best[0] < min_gain: break
            kept.append(best[1]); cur = best[0]; trace.append(cur)
        info["floor_trace"] = trace
    else: raise KeyError(method)
    info["kept"] = len(kept); info["floor"] = floor_errors([o.signature for o in kept], yb); return kept, info


# ================================================================ moteur large : faisceau sur classes, plus de six atomes

class WideSpace:
    """Cellules pour un nombre quelconque d'atomes (≤ 60) : une fonction est un vecteur booléen sur les cellules
    occupées (au plus N), plus un masque de 64 bits."""

    def __init__(self, atoms: Sequence[BNode], X: torch.Tensor, y: torch.Tensor):
        if not 1 <= len(atoms) <= 60: raise ValueError("entre 1 et 60 atomes")
        self.atoms = list(atoms); self.n = len(atoms); self.N = len(X); A = np.stack([predict(a, X.float()).numpy().astype(bool) for a in atoms]); code = np.zeros(self.N, dtype=np.uint64)
        for i in range(self.n): code |= A[i].astype(np.uint64) << np.uint64(i)
        self.codes, first, inv = np.unique(code, return_index=True, return_inverse=True); self.K = len(self.codes); yb = y.numpy().astype(np.float64)
        self.pos = np.bincount(inv, weights=yb, minlength=self.K); self.count = np.bincount(inv, minlength=self.K).astype(np.float64); self.gain = 2 * self.pos - self.count; self.total_pos = float(self.pos.sum()); self.rows = A[:, first]
        self.floor = float(np.minimum(self.pos, self.count - self.pos).sum())

    def errors(self, F: np.ndarray) -> np.ndarray: return self.total_pos - F.astype(np.float64) @ self.gain
    def row_of(self, t: Tree) -> np.ndarray:
        if isinstance(t, (int, np.integer)): return self.rows[int(t)]
        a, b = self.row_of(t[1]), self.row_of(t[2]); return a & b if t[0] == "and" else a | b if t[0] == "or" else a ^ b


def _np_op(op: int, a: np.ndarray, b: np.ndarray) -> np.ndarray: return a & b if op == 0 else a | b if op == 1 else a ^ b


def wide_compose(atoms: Sequence[BNode], X: torch.Tensor, y: torch.Tensor, max_size=6, beam=300, pair=48, selector="bic", budget: Optional[int] = None, seeds: Sequence[Tuple[Tree, int]] = ()) -> CompositionResult:
    """Composition ascendante avec FAISCEAU SUR LES CLASSES, pour plus de six atomes. Niveau s : les `beam` classes les
    moins erronées de taille s. Un niveau est combiné avec les atomes (tous) et avec les `pair` meilleures classes des
    niveaux inférieurs : jamais d'énumération exhaustive. `seeds` : (arbre, taille) de classes déjà construites
    ailleurs, réutilisées comme unités (composition hiérarchique). `budget` : lignes évaluées au total."""
    t0 = time.time(); sp = WideSpace(atoms, X, y); seen = set(); levels: Dict[int, Tuple[np.ndarray, List[Tree]]] = {}; considered = 0; trunc = False

    def keep(rows: np.ndarray, trees: List[Tree], limit: int):
        if not len(rows): return rows, trees
        packed = np.packbits(rows, axis=1); _, first = np.unique(packed, axis=0, return_index=True); first = np.sort(first); fresh = [i for i in first if packed[i].tobytes() not in seen]
        if not fresh: return rows[:0], []
        fr = np.array(fresh); e = sp.errors(rows[fr]); o = np.argsort(e, kind="stable")[:limit]; sel = fr[o]
        for i in sel: seen.add(packed[i].tobytes())
        return rows[sel], [trees[i] for i in sel]

    r1, t1 = keep(sp.rows.copy(), list(range(sp.n)), 10 ** 9); levels[1] = (r1, t1)
    for tree, size in seeds:
        row = sp.row_of(tree)[None, :]; r, t = keep(row, [tree], 1)
        if len(t): pr, pt = levels.get(size, (row[:0], [])); levels[size] = (np.concatenate([pr, r]) if len(pr) else r, pt + t)
    for s in range(2, max_size + 1):
        rows = []; trees: List[Tree] = []
        for i in range(1, s // 2 + 1):
            j = s - i
            if i not in levels or j not in levels: continue
            (ra, ta), (rb, tb) = levels[i], levels[j]
            if i > 1: ra, ta = ra[:pair], ta[:pair]
            if i > 1 or len(rb) > beam: rb, tb = rb[: (pair if i > 1 else beam)], tb[: (pair if i > 1 else beam)]
            cost = 3 * len(ra) * len(rb)
            if budget is not None and considered + cost > budget: trunc = True; break
            considered += cost
            for op in range(3):
                rows.append(_np_op(op, ra[:, None, :], rb[None, :, :]).reshape(-1, sp.K)); trees += [(OPS[op], a, b) for a in ta for b in tb]
        if not rows: break
        R, T = keep(np.concatenate(rows), trees, beam); pr, pt = levels.get(s, (R[:0], []))
        if len(pt): R = np.concatenate([pr, R]) if len(R) else pr; T = pt + T; o = np.argsort(sp.errors(R), kind="stable")[:beam]; R, T = R[o], [T[i] for i in o]
        if len(T): levels[s] = (R, T)
        if trunc: break
    allr = np.concatenate([levels[s][0] for s in sorted(levels)]); allt = [t for s in sorted(levels) for t in levels[s][1]]; sizes = np.array([tree_size(t) for t in allt]); e = sp.errors(allr); n = float(sp.N); ll = profiled_ll(e, n)
    p = float(np.mean([a.n_params() for a in sp.atoms])); sc = {"errors": e, "likelihood": ll, "bic": ll - 0.5 * sizes * math.log(max(1, sp.N)), "posterior": ll - (2 * sizes - 1) * 3.0 * math.log(2) - 0.5 * p * sizes * math.log(max(1, sp.N)), "size": sizes.astype(np.float64)}; o = order_by(sc, selector)[:20]
    R = [ObservationalEquivalenceClass(-1, allt[i], int(sizes[i]), float(e[i]), float(sc[selector][i])) for i in o]
    return CompositionResult(R[0] if R else None, R, {"considered": considered, "classes": len(allt), "cells": sp.K, "atoms": sp.n, "floor_errors": sp.floor, "truncated": trunc, "seconds": round(time.time() - t0, 4), "selector": selector})


def with_complements(atoms: Sequence[BNode]) -> List[BNode]:
    """Unités de composition : chaque atome et son complément. Une primitive découverte peut avoir la mauvaise polarité,
    et la grammaire de composition (et / ou / ou exclusif) ne sait pas nier : le complément est donc fourni comme unité.
    Il ne crée aucune cellule nouvelle."""
    return list(atoms) + [BNode("not", children=(a,)) for a in atoms]


def compose(atoms: Sequence[BNode], X: torch.Tensor, y: torch.Tensor, selector="bic", top=20, max_size: Optional[int] = None, complements=True) -> CompositionResult:
    """Moteur exact sur cellules (au plus six atomes, plus leurs compléments) : toutes les classes jusqu'à `max_size`
    feuilles (défaut : le nombre d'atomes), classées par le sélecteur. Les arbres renvoyés portent sur `R.units`."""
    units = with_complements(atoms) if complements else list(atoms); sp = CellSpace(units, X, y, max_atoms=12); T = bottom_up(sp, max_size or len(atoms), budget=3_000_000); R = select(T, selector, top); R.stats["atoms"] = len(atoms); R.units = units; R.table = T; return R


def hierarchical(atoms: Sequence[BNode], X: torch.Tensor, y: torch.Tensor, group=6, keep=12, selector="bic", beam=300) -> CompositionResult:
    """primitive hypotheses → small equivalence classes → higher-level classes : les `group` premiers atomes sont
    composés exactement ; leurs `keep` meilleures classes deviennent des unités, combinées avec les atomes restants par
    le moteur large (taille ≤ 3 unités). Étend la composition au-delà de six atomes sans énumération."""
    t0 = time.time(); atoms = list(atoms)
    if len(atoms) <= group: R = compose(atoms, X, y, selector, complements=False); R.stats["levels"] = 1; return R
    first = compose(atoms[:group], X, y, selector, top=keep, complements=False); seeds = [(c.tree, c.size) for c in first.ranked]; R = wide_compose(atoms, X, y, max_size=min(8, group + 2), beam=beam, pair=24, selector=selector, seeds=seeds)
    R.stats.update({"levels": 2, "first_level_classes": first.stats.get("classes"), "seconds": round(time.time() - t0, 4)}); return R


# ================================================================ finalisation : conserver l'état initial

@dataclass
class Finalization:
    program: BNode
    fit: str
    committed: bool
    errors_before: float
    errors_after: float


def finalize(tree: Tree, atoms: Sequence[BNode], X: torch.Tensor, y: torch.Tensor, fit="validated", steps=100) -> Finalization:
    """Structure retenue → programme final. `none` : les paramètres découverts sont gardés tels quels. `validated` :
    l'optimiseur PROPOSE, la proposition n'est acceptée que si le critère observable progresse, sinon l'état initial est
    rendu intact. `forced` : l'optimiseur remplace toujours l'état (bras de comparaison uniquement)."""
    prog = to_program(tree, atoms); e0 = float((predict(prog, X.float()) != y).sum())
    if fit == "none" or prog.n_params() == 0: return Finalization(prog, fit, False, e0, e0)
    data = FitData.build(X, y, 0.10); sk = prog.skeleton(); init = torch.tensor(prog.param_vector(), dtype=DT)
    if fit == "validated": row, tx = validated_fit(sk, data, init, steps=steps); out = sk.with_params(row.tolist()); ok = bool(tx.accepted)
    elif fit == "forced": th, _ = fit_batch(sk, data, init.reshape(1, -1), steps=steps); out = sk.with_params(th[0].tolist()); ok = True
    else: raise KeyError(fit)
    return Finalization(out, fit, ok, e0, float((predict(out, X.float()) != y).sum()))


# ================================================================ KNOWN / UNKNOWN / NONE_OF_THE_ABOVE

def knn_errors(X: torch.Tensor, y: torch.Tensor, k=7) -> float:
    """Erreurs d'un classifieur des k plus proches voisins, un point laissé de côté : ce qu'une méthode SANS structure
    explique déjà (même calcul que `core.deep_boundary.knn_loo_error`, sans objet intermédiaire)."""
    D = torch.cdist(X.double(), X.double()); D.fill_diagonal_(float("inf")); nb = D.topk(min(k, len(X) - 1), largest=False).indices; return float(((y[nb].float().mean(1) > 0.5).long() != y).sum())


def decide(result: CompositionResult, final_errors: float, X: torch.Tensor, y: torch.Tensor, gap_unknown=0.04, gap_none=0.10, tie=2.0) -> dict:
    """Décision à trois issues, sur des signaux explicites :
      gap        (erreurs de la structure retenue − erreurs des plus proches voisins) / N : ce que la structure N'EXPLIQUE
                 PAS alors que les données le permettent
      rivals     classes dont le score est à moins de `tie` du premier (structures concurrentes non départagées)
    gap ≥ gap_none → NONE_OF_THE_ABOVE ; gap ≥ gap_unknown, ou aucune structure → UNKNOWN (modèle partiel) ; sinon KNOWN."""
    N = len(X); kn = knn_errors(X, y); gap = (final_errors - kn) / max(1, N); rivals = sum(1 for c in result.ranked[1:] if result.ranked and abs(c.score - result.ranked[0].score) <= tie) if result.ranked else 0
    label = UNKNOWN if result.selected is None else NONE_OF_THE_ABOVE if gap >= gap_none else UNKNOWN if gap >= gap_unknown else KNOWN
    return {"label": label, "gap": gap, "knn_error_rate": kn / max(1, N), "error_rate": final_errors / max(1, N), "rivals": rivals, "partial_model": label == UNKNOWN and result.selected is not None}


# ================================================================ chaîne complète

# `CompositionResult` reçoit deux attributs facultatifs posés par `compose` : `units` (atomes et compléments sur lesquels
# portent les arbres) et `table` (toutes les classes construites).

@dataclass
class EndToEnd:
    program: Optional[BNode]
    label: str
    hypotheses: List[PrimitiveHypothesis]
    kept: List[PrimitiveHypothesis]
    result: Optional[CompositionResult]
    finalization: Optional[Finalization]
    decision: dict
    stats: Dict[str, float]


def discover(X: torch.Tensor, y: torch.Tensor, k=3, seed=0) -> List[Tuple[BNode, float, str]]:
    """Source de primitives de la chaîne principale : `flat_scene_k` (baseline candidate) — chaque type essayé sur la
    scène entière, k hypothèses par type. Aucun oracle."""
    from core.deep_fragments import flat_types
    from core.deep_partial import observe
    return [(h.program, float(h.score), f"flat_scene:{h.type}") for h in flat_types(observe(X, y), k, seed)]


def run_chain(X: torch.Tensor, y: torch.Tensor, seed=0, cfg: Optional[dict] = None, discovered: Optional[Sequence[Tuple[BNode, float, str]]] = None) -> EndToEnd:
    """découverte → hypothèses → réduction → composition sur classes → finalisation → décision."""
    c = {**load_config(), **(cfg or {})}; t0 = time.time(); disc = list(discovered) if discovered is not None else discover(X, y, c["k"], seed); t1 = time.time(); H = hypotheses_from(disc, X, c["same"]); kept, info = reduce(H, y, c["reduction"], c["max_atoms"], c["min_gain"])
    st = {"literals": len(disc), "hypotheses": len(H), "kept": len(kept), "discover_seconds": round(t1 - t0, 3), "reduction": info}
    if not kept:
        empty = CompositionResult(None, [], {}); e = float(min(y.sum(), len(y) - y.sum())); d = decide(empty, e, X, y, c["gap_unknown"], c["gap_none"], c["tie"]); st["seconds"] = round(time.time() - t0, 3); return EndToEnd(None, d["label"], H, kept, empty, None, d, st)
    atoms = [h.program for h in kept]; R = compose(atoms, X, y, c["selector"]); F = finalize(R.selected.tree, R.units, X, y, c["fit"]); d = decide(R, F.errors_after, X, y, c["gap_unknown"], c["gap_none"], c["tie"])
    st.update({"classes": R.stats.get("classes"), "considered": R.stats.get("considered"), "compose_seconds": R.stats.get("seconds"), "seconds": round(time.time() - t0, 3)}); return EndToEnd(F.program, d["label"], H, kept, R, F, d, st)

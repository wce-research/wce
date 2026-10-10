"""Latent Primitive Recovery — reconstruire, à partir des observations seules, un ENSEMBLE latent de primitives.

Le problème n'est plus une liste de détections indépendantes : une tâche n'est exploitable que si les primitives
nécessaires sont présentes ENSEMBLE. Le système raisonne donc sur des ensembles.

    vivier (rappel d'abord) → hypothèses → utilité conditionnelle → ensembles compatibles (top-k)
        → composition sur classes par ensemble → finalisation validée
        → résidu de composition → primitive manquante ciblée → recomposition        (boucle bornée)

Objets :
  ConditionalUtility            valeur d'une primitive seule, en paire, et dans l'ensemble courant
  LatentPrimitiveSetHypothesis  un ensemble cohérent : primitives, score global, couverture, contradictions, coût
  CompatiblePrimitiveSets       les k meilleurs ensembles ; la composition les départage
  PrimitiveSetCompleteness      signal explicite : il manque probablement encore une primitive
  LatentRecovery                résultat, crédits structurels par primitive, journal de la boucle

Asymétrie assumée (pré-enregistrée) : un faux atome de plus coûte peu, un vrai atome manquant est catastrophique.
Le vivier privilégie donc le rappel ; l'élimination est laissée à la sélection d'ensembles et à la composition.
Toute transformation de paramètres est validée : l'état initial est conservé, rien n'est détruit.

Aucune vérité de monde ; aucun oracle.
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
from core.deep_compose import CompositionResult, profiled_ll, to_program, tree_atoms
from core.deep_e2e import Finalization, PrimitiveHypothesis, compose, finalize, floor_errors, hypotheses_from, knn_errors
from core.deep_primitives import atom_name

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "deep_configs", "lpr.json")
DEFAULTS = {"k": 5, "sources": ("flat", "fragments", "components"), "max_atoms": 6, "beam": 5, "pair_pool": 12, "min_gain": 1.0, "n_sets": 3, "selector": "bic", "fit": "validated", "iters": 2, "residual_k": 3,
            "incomplete": 0.10, "state_threshold": 0.10, "pairs": True, "refine": True}
# `incomplete` : excès de regroupement des erreurs au-dessus duquel la boucle cherche une primitive manquante (−1 : toujours
# essayer ; la reprise n'est de toute façon gardée que si les erreurs baissent). `state_threshold` : seuil, fixe, de l'état
# COMPLETE rapporté — il ne dépend pas du réglage de la boucle.
STATES = ("COMPLETE", "INCOMPLETE", "POSSIBLY_OUT_OF_CLASS")


def load_config() -> dict:
    """Réglages choisis sur les tâches de MISE AU POINT uniquement ; valeurs par défaut sinon."""
    try: return {**DEFAULTS, **json.load(open(CONFIG_FILE))["config"]}
    except (OSError, ValueError, KeyError): return dict(DEFAULTS)


# ================================================================ vivier : le rappel d'abord

def pool(X: torch.Tensor, y: torch.Tensor, k=5, seed=0, sources=("flat", "fragments", "components")) -> List[Tuple[BNode, float, str]]:
    """Vivier de primitives, RAPPEL D'ABORD : plusieurs sources complémentaires, plusieurs paramétrisations par type
    (`k`), aucune élimination. Les scores sont ramenés à un rang par source pour rester comparables.
      flat        chaque type essayé sur la scène entière, k paramétrisations par type
      fragments   toutes les hypothèses de type issues des fragments (pas seulement les retenues)
      components  cercles et droites de consensus, comme disques, anti-disques et demi-plans"""
    from core.deep_fragments import SKELETONS, flat_types, infer_types
    from core.deep_partial import build, circle_components, line_components, observe, orient
    ob = observe(X, y); out: List[Tuple[BNode, float, str]] = []
    def ranked(items, src):
        n = len(items); return [(p, 1.0 - i / max(1, n), f"{src}:{t}") for i, (p, t) in enumerate(items)]
    if "flat" in sources: out += ranked([(h.program, h.type) for h in flat_types(ob, k, seed)], "flat")
    if "fragments" in sources: out += ranked([(h.program, h.type) for h in infer_types(ob, 3, seed=seed).hypotheses if h.score > 0], "fragments")
    if "components" in sources:
        items = []
        for c in sorted(circle_components(ob, seed=seed) + line_components(ob, seed=seed), key=lambda c: -(len(c.inliers) - 5 * c.rms)):
            if c.kind == "circle": items += [(build(SKELETONS["disc"], c.params[:2], (c.params[2],)), "disc")]
            else: items += [(orient(build(SKELETONS["half"], None, c.params), X, y), "half")]
        out += ranked(items, "components")
    return out


# ================================================================ utilité conditionnelle

@dataclass
class ConditionalUtility:
    """Valeur d'une primitive pour la COMPOSITION, en erreurs plancher évitées (étiquette majoritaire par cellule) :
    seule, dans sa meilleure paire, et ajoutée à l'ensemble courant. Une primitive peut ne rien valoir seule et
    beaucoup en combinaison (les deux opérandes d'un ou exclusif, par exemple)."""
    index: int
    alone: float
    pair: float                     # meilleur gain d'une paire qui la contient
    partner: Optional[int]
    in_context: float               # gain quand elle est ajoutée à l'ensemble donné

    @property
    def conditional_only(self): return self.alone < 1.0 <= self.pair / 2.0


def utilities(hyps: Sequence[PrimitiveHypothesis], y: torch.Tensor, selected: Sequence[int] = (), pair_pool=12) -> List[ConditionalUtility]:
    """Utilités de chaque hypothèse. Les paires ne sont évaluées qu'entre les `pair_pool` hypothèses les plus confiantes
    (au plus 66 paires) : pas d'explosion combinatoire."""
    yb = y.numpy().astype(bool); base = floor_errors([], yb); sig = [h.signature for h in hyps]; ctx = [sig[i] for i in selected]; cur = floor_errors(ctx, yb)
    alone = [base - floor_errors([s], yb) for s in sig]; inc = [cur - floor_errors(ctx + [s], yb) if i not in selected else 0.0 for i, s in enumerate(sig)]
    top = sorted(range(len(hyps)), key=lambda i: -hyps[i].confidence)[:pair_pool]; best = {i: (0.0, None) for i in range(len(hyps))}
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            i, j = top[a], top[b]; g = base - floor_errors([sig[i], sig[j]], yb)
            if g > best[i][0]: best[i] = (g, j)
            if g > best[j][0]: best[j] = (g, i)
    return [ConditionalUtility(i, alone[i], best[i][0], best[i][1], inc[i]) for i in range(len(hyps))]


# ================================================================ ensembles de primitives

@dataclass
class LatentPrimitiveSetHypothesis:
    """Un ENSEMBLE cohérent de primitives candidates."""
    atoms: List[PrimitiveHypothesis]
    indices: Tuple[int, ...]                    # positions dans le vivier d'hypothèses
    floor: float                                # erreurs plancher : ce qu'aucune composition de ces atomes ne peut expliquer
    score: float                                # vraisemblance profilée du plancher − ½·ln N par atome
    coverage: float                             # 1 − plancher / N
    contradictions: int                         # paires d'atomes presque redondants (accord ≥ 0,9, ou ≤ 0,1)
    cost: float                                 # coût descriptif : ½·ln N par paramètre
    trace: List[dict] = field(default_factory=list)

    @property
    def programs(self) -> List[BNode]: return [h.program for h in self.atoms]
    @property
    def provenance(self) -> List[str]: return [h.provenance for h in self.atoms]
    @property
    def equivalences(self) -> int: return sum(h.n_alternatives for h in self.atoms)
    @property
    def uncertainties(self) -> List[Optional[float]]: return [h.uncertainty for h in self.atoms]


@dataclass
class CompatiblePrimitiveSets:
    """Les k meilleurs ensembles. Aucun n'est imposé : la composition les départage."""
    sets: List[LatentPrimitiveSetHypothesis]

    @property
    def best(self): return self.sets[0] if self.sets else None
    def union(self, k: Optional[int] = None) -> List[PrimitiveHypothesis]:
        seen = set(); out = []
        for s in self.sets[:k]:
            for i, h in zip(s.indices, s.atoms):
                if i not in seen: seen.add(i); out.append(h)
        return out
    def __len__(self): return len(self.sets)


def _make_set(hyps, idx: Tuple[int, ...], y: torch.Tensor, trace=None) -> LatentPrimitiveSetHypothesis:
    yb = y.numpy().astype(bool); N = len(yb); atoms = [hyps[i] for i in idx]; f = floor_errors([h.signature for h in atoms], yb); con = 0
    for a in range(len(atoms)):
        for b in range(a + 1, len(atoms)):
            ag = float((atoms[a].signature == atoms[b].signature).mean()); con += ag >= 0.9 or ag <= 0.1
    return LatentPrimitiveSetHypothesis(atoms, tuple(idx), f, float(profiled_ll(f, float(N))) - 0.5 * math.log(max(1, N)) * len(idx), 1.0 - f / max(1, N), con, 0.5 * math.log(max(1, N)) * sum(h.program.n_params() for h in atoms), trace or [])


def select_sets(hyps: Sequence[PrimitiveHypothesis], y: torch.Tensor, max_atoms=6, beam=5, pair_pool=12, min_gain=1.0, n_sets=3, pairs=True) -> CompatiblePrimitiveSets:
    """Recherche en faisceau sur les ENSEMBLES. À chaque pas, un ensemble peut recevoir une primitive, ou — sauvetage
    par paires — DEUX primitives à la fois quand leur gain commun par atome dépasse le meilleur gain d'une seule : une
    primitive qui n'agit qu'en combinaison n'est donc pas écartée parce qu'elle semble inutile seule. Un ensemble
    cesse de grandir quand aucun ajout ne gagne `min_gain` erreurs par atome. Renvoie les `n_sets` meilleurs ensembles
    distincts, tous états du faisceau confondus."""
    yb = y.numpy().astype(bool); H = list(hyps); n = len(H)
    if n == 0: return CompatiblePrimitiveSets([])
    sig = [h.signature for h in H]; cand = sorted(range(n), key=lambda i: -H[i].confidence); pp = cand[:pair_pool]; base = floor_errors([], yb); F: Dict[frozenset, float] = {frozenset(): base}
    def fl(s: frozenset) -> float:
        if s not in F: F[s] = floor_errors([sig[i] for i in sorted(s)], yb)
        return F[s]
    frontier: List[Tuple[frozenset, List[dict]]] = [(frozenset(), [])]; done: Dict[frozenset, List[dict]] = {}
    for _ in range(max_atoms):
        nxt: Dict[frozenset, Tuple[float, List[dict]]] = {}
        for s, tr in frontier:
            cur = fl(s); moves = []
            for i in cand:
                if i not in s and len(s) < max_atoms: g = cur - fl(s | {i}); moves.append((g, g, (i,)))
            if pairs and len(s) + 2 <= max_atoms:
                for a in range(len(pp)):
                    for b in range(a + 1, len(pp)):
                        i, j = pp[a], pp[b]
                        if i in s or j in s: continue
                        g = cur - fl(s | {i, j}); moves.append((g / 2.0, g, (i, j)))
            moves = [m for m in moves if m[0] >= min_gain]
            if not moves: done.setdefault(s, tr); continue
            for per, g, add in sorted(moves, key=lambda m: (-m[0], m[2]))[:beam]:
                t = s | set(add); step = {"added": list(add), "gain": g, "pair": len(add) == 2, "floor": fl(t)}
                if t not in nxt or fl(t) < nxt[t][0]: nxt[t] = (fl(t), tr + [step])
        if not nxt: break
        frontier = [(s, tr) for s, (f, tr) in sorted(nxt.items(), key=lambda kv: (kv[1][0], sorted(kv[0])))[:beam]]
        for s, tr in frontier: done.setdefault(s, tr)
    for s, tr in frontier: done.setdefault(s, tr)
    sets = [_make_set(H, tuple(sorted(s)), y, tr) for s, tr in done.items() if s]; sets.sort(key=lambda z: (-z.score, z.indices)); return CompatiblePrimitiveSets(sets[:n_sets])


# ================================================================ complétude de l'ensemble

@dataclass
class PrimitiveSetCompleteness:
    """Signal explicite : manque-t-il probablement une primitive ? Les erreurs d'un modèle complet sont du bruit, donc
    isolées ; les erreurs d'un modèle incomplet sont groupées le long de la frontière manquante."""
    error_rate: float
    clustering: float               # part des voisins d'une erreur qui sont aussi des erreurs
    excess: float                   # clustering − taux d'erreur : ce qui dépasse un bruit sans structure
    gap: float                      # (erreurs − erreurs des plus proches voisins) / N
    complete: bool


def completeness(X: torch.Tensor, y: torch.Tensor, pred: torch.Tensor, k=3, threshold=0.10) -> PrimitiveSetCompleteness:
    err = (pred != y); n = len(y); e = float(err.float().mean())
    if int(err.sum()) < 2: return PrimitiveSetCompleteness(e, 0.0, 0.0, (float(err.sum()) - knn_errors(X, y)) / max(1, n), True)
    D = torch.cdist(X.double(), X.double()); D.fill_diagonal_(float("inf")); nb = D.topk(min(k, n - 1), largest=False).indices; cl = float(err[nb][err].float().mean()); ex = cl - e
    return PrimitiveSetCompleteness(e, cl, ex, (float(err.sum()) - knn_errors(X, y)) / max(1, n), ex < threshold)


def propose_missing(X: torch.Tensor, y: torch.Tensor, pred: torch.Tensor, k=3, seed=0) -> List[Tuple[BNode, float, str]]:
    """current primitive set → residual unexplained structure → targeted primitive proposal.

    Le résidu est l'étiquette « la structure courante se trompe ici ». Si une primitive manque, les frontières du résidu
    sont les siennes : la découverte est relancée sur le résidu, et non sur les observations."""
    from core.deep_fragments import flat_types
    from core.deep_partial import observe
    r = (pred != y).long()
    if int(r.sum()) < 6 or int(r.sum()) > len(r) - 6: return []
    H = flat_types(observe(X, r), k, seed); n = len(H); return [(h.program, 1.0 - i / max(1, n), f"residual:{h.type}") for i, h in enumerate(H)]


# ================================================================ récupération complète

@dataclass
class LatentRecovery:
    program: Optional[BNode]
    sets: CompatiblePrimitiveSets
    chosen: Optional[LatentPrimitiveSetHypothesis]
    hypotheses: List[PrimitiveHypothesis]
    result: Optional[CompositionResult]
    finalization: Optional[Finalization]
    completeness: Optional[PrimitiveSetCompleteness]
    state: str
    credit: Dict[int, float]                    # crédit structurel par hypothèse du vivier
    log: List[dict]
    stats: Dict[str, float]


def compose_sets(sets: CompatiblePrimitiveSets, X, y, selector="bic", fit="validated"):
    """Chaque ensemble est composé ; le meilleur est celui dont la meilleure structure a le meilleur score : la composition
    départage les ensembles. Renvoie (ensemble, résultat, finalisation, compositions évaluées)."""
    best = None; considered = 0
    for s in sets.sets:
        R = compose(s.programs, X, y, selector); considered += int(R.stats.get("considered", 0))
        if R.selected is not None and (best is None or R.selected.score > best[1].selected.score): best = (s, R)
    if best is None: return None, None, None, considered
    return best[0], best[1], finalize(best[1].selected.tree, best[1].units, X, y, fit), considered


def strong_downstream(discovered: Sequence[Tuple[BNode, float, str]], X: torch.Tensor, y: torch.Tensor, cfg: Optional[dict] = None) -> LatentRecovery:
    """Aval FORT, commun à toutes les sources de primitives : hypothèses → ensembles (utilité conditionnelle, paires) →
    composition sur classes par ensemble → finalisation validée. Aucune boucle, aucune recherche de primitive."""
    c = {**load_config(), **(cfg or {})}; t0 = time.time(); H = hypotheses_from(list(discovered), X); sets = select_sets(H, y, c["max_atoms"], c["beam"], c["pair_pool"], c["min_gain"], c["n_sets"], c["pairs"]); s, R, F, cons = compose_sets(sets, X, y, c["selector"], c["fit"])
    prog = F.program if F else None; comp = completeness(X, y, predict(prog, X.float()), c["residual_k"], c["state_threshold"]) if prog is not None else None
    return LatentRecovery(prog, sets, s, H, R, F, comp, "COMPLETE" if comp and comp.complete else "INCOMPLETE", _credit(s, R), [], {"literals": len(discovered), "hypotheses": len(H), "sets": len(sets), "kept": len(s.atoms) if s else 0, "considered": cons, "seconds": round(time.time() - t0, 3)})


def _credit(s: Optional[LatentPrimitiveSetHypothesis], R: Optional[CompositionResult], top=10) -> Dict[int, float]:
    """Crédit structurel EXPLICITE : part des `top` meilleures structures qui utilisent chaque primitive de l'ensemble.
    Une primitive que la composition n'emploie jamais reçoit 0 ; rien d'opaque."""
    if s is None or R is None or not R.ranked: return {}
    n = len(s.atoms); use = np.zeros(n); rk = R.ranked[:top]
    for c in rk:
        for u in set(tree_atoms(c.tree)): use[u % n] += 1
    return {int(i): float(use[k] / len(rk)) for k, i in enumerate(s.indices)}


def recover(X: torch.Tensor, y: torch.Tensor, seed=0, cfg: Optional[dict] = None, discovered: Optional[Sequence[Tuple[BNode, float, str]]] = None) -> LatentRecovery:
    """Boucle bornée et instrumentée :
      1. vivier (rappel d'abord) ; 2. ensembles ; 3. composition et finalisation validée ;
      4. si l'ajustement validé a été accepté, les primitives AJUSTÉES rejoignent le vivier (les originales restent) ;
      5. complétude ; si l'ensemble paraît incomplet, primitives proposées sur le RÉSIDU, puis retour en 2.
    Une reprise n'est gardée que si le nombre d'erreurs sur les observations baisse ; sinon l'état précédent est rendu."""
    c = {**load_config(), **(cfg or {})}; t0 = time.time(); disc = list(discovered) if discovered is not None else pool(X, y, c["k"], seed, tuple(c["sources"])); log = []; best: Optional[LatentRecovery] = None; cons = 0; n_pool0 = len(disc)
    for it in range(c["iters"] + 1):
        cur = strong_downstream(disc, X, y, c); cons += cur.stats["considered"]; e = cur.finalization.errors_after if cur.finalization else float(len(y))
        ok = best is None or e < best.finalization.errors_after - 1e-9
        log.append({"iteration": it, "pool": len(disc), "hypotheses": len(cur.hypotheses), "kept": cur.stats["kept"], "errors": e, "excess": cur.completeness.excess if cur.completeness else None, "complete": bool(cur.completeness and cur.completeness.complete), "action": "commit" if ok else "rollback"})
        if ok: best = cur
        if not ok or cur.program is None or (cur.completeness and cur.completeness.excess < c["incomplete"]) or it == c["iters"]: break
        add: List[Tuple[BNode, float, str]] = []
        if c["refine"] and cur.finalization and cur.finalization.committed:
            from core.deep_partial import leaf_paths
            add += [(l.children[0] if l.kind == "not" else l, 1.5, "refined") for _, l in leaf_paths(cur.program)]
        new = propose_missing(X, y, predict(cur.program, X.float()), c["residual_k"], seed + it + 1); add += [(p, 1.2 * s, src) for p, s, src in new]; log[-1]["proposed"] = len(new); log[-1]["refined"] = len(add) - len(new)
        if not add: break
        disc = disc + add
    out = best; found = any(l.get("proposed", 0) > 0 for l in log); comp = out.completeness
    out.state = "COMPLETE" if comp and comp.complete else "POSSIBLY_OUT_OF_CLASS" if (len(log) > 1 or not found) and comp is not None and comp.gap > 0.02 else "INCOMPLETE"
    out.log = log; out.stats.update({"pool": n_pool0, "iterations": len(log), "considered_total": cons, "seconds": round(time.time() - t0, 3)}); return out

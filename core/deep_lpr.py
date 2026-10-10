"""Latent Primitive Recovery — reconstruct, from the observations alone, a latent SET of primitives.

The problem is no longer a list of independent detections: a task is usable only if the necessary primitives
are present TOGETHER. The system therefore reasons over sets.

    pool (recall first) → hypotheses → conditional utility → compatible sets (top-k)
        → composition over classes per set → validated finalization
        → composition residual → targeted missing primitive → recomposition        (bounded loop)

Objects:
  ConditionalUtility            value of a primitive alone, in a pair, and within the current set
  LatentPrimitiveSetHypothesis  a coherent set: primitives, global score, coverage, contradictions, cost
  CompatiblePrimitiveSets       the k best sets; composition decides between them
  PrimitiveSetCompleteness      explicit signal: a primitive is probably still missing
  LatentRecovery                result, structural credits per primitive, loop log

Deliberate asymmetry (preregistered): one more false atom costs little, one missing true atom is catastrophic.
The pool therefore favors recall; elimination is left to set selection and to composition.
Every parameter transformation is validated: the initial state is kept, nothing is destroyed.

No world truth; no oracle.
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
# `incomplete`: excess of error clustering above which the loop looks for a missing primitive (−1: always
# try; the retry is in any case kept only if the errors decrease). `state_threshold`: fixed threshold of the reported
# COMPLETE state — it does not depend on the loop setting.
STATES = ("COMPLETE", "INCOMPLETE", "POSSIBLY_OUT_OF_CLASS")


def load_config() -> dict:
    """Settings chosen on the TUNING tasks only; default values otherwise."""
    try: return {**DEFAULTS, **json.load(open(CONFIG_FILE))["config"]}
    except (OSError, ValueError, KeyError): return dict(DEFAULTS)


# ================================================================ pool: recall first

def pool(X: torch.Tensor, y: torch.Tensor, k=5, seed=0, sources=("flat", "fragments", "components")) -> List[Tuple[BNode, float, str]]:
    """Pool of primitives, RECALL FIRST: several complementary sources, several parameterizations per type
    (`k`), no elimination. Scores are reduced to a rank per source so that they stay comparable.
      flat        each type tried on the whole scene, k parameterizations per type
      fragments   all the type hypotheses coming from the fragments (not only the retained ones)
      components  consensus circles and lines, as discs, anti-discs and half-planes"""
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


# ================================================================ conditional utility

@dataclass
class ConditionalUtility:
    """Value of a primitive for COMPOSITION, in floor errors avoided (majority label per cell):
    alone, in its best pair, and added to the current set. A primitive may be worth nothing alone and
    a lot in combination (the two operands of an exclusive or, for example)."""
    index: int
    alone: float
    pair: float                     # best gain of a pair that contains it
    partner: Optional[int]
    in_context: float               # gain when it is added to the given set

    @property
    def conditional_only(self): return self.alone < 1.0 <= self.pair / 2.0


def utilities(hyps: Sequence[PrimitiveHypothesis], y: torch.Tensor, selected: Sequence[int] = (), pair_pool=12) -> List[ConditionalUtility]:
    """Utilities of each hypothesis. Pairs are evaluated only among the `pair_pool` most confident hypotheses
    (at most 66 pairs): no combinatorial explosion."""
    yb = y.numpy().astype(bool); base = floor_errors([], yb); sig = [h.signature for h in hyps]; ctx = [sig[i] for i in selected]; cur = floor_errors(ctx, yb)
    alone = [base - floor_errors([s], yb) for s in sig]; inc = [cur - floor_errors(ctx + [s], yb) if i not in selected else 0.0 for i, s in enumerate(sig)]
    top = sorted(range(len(hyps)), key=lambda i: -hyps[i].confidence)[:pair_pool]; best = {i: (0.0, None) for i in range(len(hyps))}
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            i, j = top[a], top[b]; g = base - floor_errors([sig[i], sig[j]], yb)
            if g > best[i][0]: best[i] = (g, j)
            if g > best[j][0]: best[j] = (g, i)
    return [ConditionalUtility(i, alone[i], best[i][0], best[i][1], inc[i]) for i in range(len(hyps))]


# ================================================================ sets of primitives

@dataclass
class LatentPrimitiveSetHypothesis:
    """A coherent SET of candidate primitives."""
    atoms: List[PrimitiveHypothesis]
    indices: Tuple[int, ...]                    # positions in the hypothesis pool
    floor: float                                # floor errors: what no composition of these atoms can explain
    score: float                                # profiled likelihood of the floor − ½·ln N per atom
    coverage: float                             # 1 − floor / N
    contradictions: int                         # pairs of nearly redundant atoms (agreement ≥ 0.9, or ≤ 0.1)
    cost: float                                 # descriptive cost: ½·ln N per parameter
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
    """The k best sets. None is imposed: composition decides between them."""
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
    """Beam search over SETS. At each step, a set may receive one primitive, or — pair
    rescue — TWO primitives at once when their joint gain per atom exceeds the best gain of a single one: a
    primitive that acts only in combination is therefore not discarded because it seems useless alone. A set
    stops growing when no addition gains `min_gain` errors per atom. Returns the `n_sets` best distinct
    sets, across all beam states."""
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


# ================================================================ completeness of the set

@dataclass
class PrimitiveSetCompleteness:
    """Explicit signal: is a primitive probably missing? The errors of a complete model are noise, hence
    isolated; the errors of an incomplete model are clustered along the missing boundary."""
    error_rate: float
    clustering: float               # share of an error's neighbors that are also errors
    excess: float                   # clustering − error rate: what exceeds structureless noise
    gap: float                      # (errors − nearest-neighbor errors) / N
    complete: bool


def completeness(X: torch.Tensor, y: torch.Tensor, pred: torch.Tensor, k=3, threshold=0.10) -> PrimitiveSetCompleteness:
    err = (pred != y); n = len(y); e = float(err.float().mean())
    if int(err.sum()) < 2: return PrimitiveSetCompleteness(e, 0.0, 0.0, (float(err.sum()) - knn_errors(X, y)) / max(1, n), True)
    D = torch.cdist(X.double(), X.double()); D.fill_diagonal_(float("inf")); nb = D.topk(min(k, n - 1), largest=False).indices; cl = float(err[nb][err].float().mean()); ex = cl - e
    return PrimitiveSetCompleteness(e, cl, ex, (float(err.sum()) - knn_errors(X, y)) / max(1, n), ex < threshold)


def propose_missing(X: torch.Tensor, y: torch.Tensor, pred: torch.Tensor, k=3, seed=0) -> List[Tuple[BNode, float, str]]:
    """current primitive set → residual unexplained structure → targeted primitive proposal.

    The residual is the label "the current structure is wrong here". If a primitive is missing, the boundaries of the
    residual are its own: discovery is relaunched on the residual, and not on the observations."""
    from core.deep_fragments import flat_types
    from core.deep_partial import observe
    r = (pred != y).long()
    if int(r.sum()) < 6 or int(r.sum()) > len(r) - 6: return []
    H = flat_types(observe(X, r), k, seed); n = len(H); return [(h.program, 1.0 - i / max(1, n), f"residual:{h.type}") for i, h in enumerate(H)]


# ================================================================ full recovery

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
    credit: Dict[int, float]                    # structural credit per hypothesis of the pool
    log: List[dict]
    stats: Dict[str, float]


def compose_sets(sets: CompatiblePrimitiveSets, X, y, selector="bic", fit="validated"):
    """Each set is composed; the best one is the set whose best structure has the best score: composition
    decides between the sets. Returns (set, result, finalization, compositions evaluated)."""
    best = None; considered = 0
    for s in sets.sets:
        R = compose(s.programs, X, y, selector); considered += int(R.stats.get("considered", 0))
        if R.selected is not None and (best is None or R.selected.score > best[1].selected.score): best = (s, R)
    if best is None: return None, None, None, considered
    return best[0], best[1], finalize(best[1].selected.tree, best[1].units, X, y, fit), considered


def strong_downstream(discovered: Sequence[Tuple[BNode, float, str]], X: torch.Tensor, y: torch.Tensor, cfg: Optional[dict] = None) -> LatentRecovery:
    """STRONG downstream, common to all primitive sources: hypotheses → sets (conditional utility, pairs) →
    composition over classes per set → validated finalization. No loop, no primitive search."""
    c = {**load_config(), **(cfg or {})}; t0 = time.time(); H = hypotheses_from(list(discovered), X); sets = select_sets(H, y, c["max_atoms"], c["beam"], c["pair_pool"], c["min_gain"], c["n_sets"], c["pairs"]); s, R, F, cons = compose_sets(sets, X, y, c["selector"], c["fit"])
    prog = F.program if F else None; comp = completeness(X, y, predict(prog, X.float()), c["residual_k"], c["state_threshold"]) if prog is not None else None
    return LatentRecovery(prog, sets, s, H, R, F, comp, "COMPLETE" if comp and comp.complete else "INCOMPLETE", _credit(s, R), [], {"literals": len(discovered), "hypotheses": len(H), "sets": len(sets), "kept": len(s.atoms) if s else 0, "considered": cons, "seconds": round(time.time() - t0, 3)})


def _credit(s: Optional[LatentPrimitiveSetHypothesis], R: Optional[CompositionResult], top=10) -> Dict[int, float]:
    """EXPLICIT structural credit: share of the `top` best structures that use each primitive of the set.
    A primitive that composition never uses receives 0; nothing opaque."""
    if s is None or R is None or not R.ranked: return {}
    n = len(s.atoms); use = np.zeros(n); rk = R.ranked[:top]
    for c in rk:
        for u in set(tree_atoms(c.tree)): use[u % n] += 1
    return {int(i): float(use[k] / len(rk)) for k, i in enumerate(s.indices)}


def recover(X: torch.Tensor, y: torch.Tensor, seed=0, cfg: Optional[dict] = None, discovered: Optional[Sequence[Tuple[BNode, float, str]]] = None) -> LatentRecovery:
    """Bounded and instrumented loop:
      1. pool (recall first); 2. sets; 3. composition and validated finalization;
      4. if the validated fit was accepted, the FITTED primitives join the pool (the originals stay);
      5. completeness; if the set looks incomplete, primitives proposed on the RESIDUAL, then back to 2.
    A retry is kept only if the number of errors on the observations decreases; otherwise the previous state is returned."""
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

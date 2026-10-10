"""End-to-End Composition with Discovered Atoms — from discovered primitives, imperfect and sometimes false, to a
composition over equivalence classes.

    DiscoveredPrimitive → PrimitiveHypothesis → reduction → composition over classes → validated finalisation → decision

Explicit layer between discovery and composition:
  PrimitiveHypothesis   type, parameters, confidence, uncertainty, provenance, equivalent alternatives — a
                        discovered primitive is never treated as a truth
  reduce                about twenty literals → at most six atoms, through compact and comparable mechanisms
  compose               exact engine over cells (≤ 6 atoms); `wide_compose`: beam over classes for more atoms;
                        `hierarchical`: classes of a first group reused as units of a second level
  finalize              no fit, or VALIDATED fit (the initial state is kept; commit only if the
                        observable criterion improves); the forced fit exists only as a comparison arm
  decide                KNOWN / UNKNOWN / NONE_OF_THE_ABOVE from explicit signals

No world truth; no oracle of type, parameter, atom or membership.
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
    """Configuration of the chain, tuned on the TUNING tasks only; default values otherwise."""
    try: return {**DEFAULTS, **json.load(open(CONFIG_FILE))["config"]}
    except (OSError, ValueError, KeyError): return dict(DEFAULTS)
FITS = ("none", "validated", "forced")


# ================================================================ primitive hypotheses

@dataclass
class PrimitiveHypothesis:
    """A DISCOVERED primitive, with what is known about its reliability. `alternatives`: observationally
    equivalent (or complementary) primitives that it represents — they remain available, they are not additional atoms."""
    program: BNode
    type: str
    confidence: float               # discovery score, rescaled to [0, 1] within the scene
    provenance: str
    signature: np.ndarray           # value on the observations
    alternatives: List[BNode] = field(default_factory=list)
    uncertainty: Optional[float] = None       # mean functional disagreement with its alternatives (None if there are none)

    @property
    def params(self): return self.program.param_vector()
    @property
    def n_alternatives(self): return len(self.alternatives)


def hypotheses_from(discovered: Sequence[Tuple[BNode, float, str]], X: torch.Tensor, same=0.97) -> List[PrimitiveHypothesis]:
    """(program, score, provenance) → hypotheses. Two primitives with the same signature on the observations, or with
    complementary signatures (agreement ≥ `same` or ≤ 1 − `same`), are ONE hypothesis: the best-scored one represents it, the
    others become its alternatives. This is the deduplication and the equivalence of primitives."""
    items = sorted(discovered, key=lambda d: -d[1]); top = max((s for _, s, _ in items), default=1.0) or 1.0; out: List[PrimitiveHypothesis] = []
    for p, s, src in items:
        sig = predict(p, X.float()).numpy().astype(bool); host = None
        for h in out:
            a = float((sig == h.signature).mean())
            if a >= same or a <= 1 - same: host = h; break
        if host is None: out.append(PrimitiveHypothesis(p, atom_name(p), max(0.0, s) / max(top, 1e-9), src, sig))
        else: host.alternatives.append(p); d = float((sig != host.signature).mean()); d = min(d, 1 - d); host.uncertainty = d if host.uncertainty is None else (host.uncertainty * (host.n_alternatives - 1) + d) / host.n_alternatives
    return out


# ================================================================ reduction before composition

def floor_errors(sigs: Sequence[np.ndarray], y: np.ndarray) -> float:
    """Smallest number of errors reachable by ANY single Boolean function of these atoms: majority
    label of each cell. Depends on no structure."""
    if not sigs: return float(min(y.sum(), len(y) - y.sum()))
    code = np.zeros(len(y), dtype=np.uint64)
    for i, s in enumerate(sigs): code |= s.astype(np.uint64) << np.uint64(i)
    _, inv = np.unique(code, return_inverse=True); pos = np.bincount(inv, weights=y.astype(np.float64)); cnt = np.bincount(inv).astype(np.float64); return float(np.minimum(pos, cnt - pos).sum())


def reduce(hyps: Sequence[PrimitiveHypothesis], y: torch.Tensor, method="greedy_floor", max_atoms=6, min_gain=4.0, per_family=2, dominance=0.90) -> Tuple[List[PrimitiveHypothesis], dict]:
    """About twenty hypotheses → at most `max_atoms`. Mechanisms (separately comparable):
      confidence    the most confident ones
      dedupe        same — the deduplication is already done by `hypotheses_from`; this arm takes the first ones with NO criterion other than the order
      family        at most `per_family` per type, then the most confident ones
      dominance     a hypothesis is discarded if a more confident one is functionally close to it (agreement ≥ `dominance`)
      greedy_floor  forward selection: at each step, the atom that most reduces the floor error (majority label per
                    cell); stop when the gain falls below `min_gain` errors — usefulness for the COMPOSITION decides"""
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


# ================================================================ wide engine: beam over classes, more than six atoms

class WideSpace:
    """Cells for any number of atoms (≤ 60): a function is a Boolean vector over the occupied
    cells (at most N), no longer a 64-bit mask."""

    def __init__(self, atoms: Sequence[BNode], X: torch.Tensor, y: torch.Tensor):
        if not 1 <= len(atoms) <= 60: raise ValueError("between 1 and 60 atoms")
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
    """Bottom-up composition with a BEAM OVER THE CLASSES, for more than six atoms. Level s: the `beam` classes with the
    fewest errors of size s. A level is combined with the atoms (all of them) and with the `pair` best classes of the
    lower levels: never an exhaustive enumeration. `seeds`: (tree, size) of classes already built
    elsewhere, reused as units (hierarchical composition). `budget`: rows evaluated in total."""
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
    """Composition units: each atom and its complement. A discovered primitive may have the wrong polarity,
    and the composition grammar (and / or / exclusive or) cannot negate: the complement is therefore supplied as a unit.
    It creates no new cell."""
    return list(atoms) + [BNode("not", children=(a,)) for a in atoms]


def compose(atoms: Sequence[BNode], X: torch.Tensor, y: torch.Tensor, selector="bic", top=20, max_size: Optional[int] = None, complements=True) -> CompositionResult:
    """Exact engine over cells (at most six atoms, plus their complements): all the classes up to `max_size`
    leaves (default: the number of atoms), ranked by the selector. The returned trees refer to `R.units`."""
    units = with_complements(atoms) if complements else list(atoms); sp = CellSpace(units, X, y, max_atoms=12); T = bottom_up(sp, max_size or len(atoms), budget=3_000_000); R = select(T, selector, top); R.stats["atoms"] = len(atoms); R.units = units; R.table = T; return R


def hierarchical(atoms: Sequence[BNode], X: torch.Tensor, y: torch.Tensor, group=6, keep=12, selector="bic", beam=300) -> CompositionResult:
    """primitive hypotheses → small equivalence classes → higher-level classes: the first `group` atoms are
    composed exactly; their `keep` best classes become units, combined with the remaining atoms by
    the wide engine (size ≤ 3 units). Extends the composition beyond six atoms without enumeration."""
    t0 = time.time(); atoms = list(atoms)
    if len(atoms) <= group: R = compose(atoms, X, y, selector, complements=False); R.stats["levels"] = 1; return R
    first = compose(atoms[:group], X, y, selector, top=keep, complements=False); seeds = [(c.tree, c.size) for c in first.ranked]; R = wide_compose(atoms, X, y, max_size=min(8, group + 2), beam=beam, pair=24, selector=selector, seeds=seeds)
    R.stats.update({"levels": 2, "first_level_classes": first.stats.get("classes"), "seconds": round(time.time() - t0, 4)}); return R


# ================================================================ finalisation: keep the initial state

@dataclass
class Finalization:
    program: BNode
    fit: str
    committed: bool
    errors_before: float
    errors_after: float


def finalize(tree: Tree, atoms: Sequence[BNode], X: torch.Tensor, y: torch.Tensor, fit="validated", steps=100) -> Finalization:
    """Selected structure → final program. `none`: the discovered parameters are kept as they are. `validated`:
    the optimiser PROPOSES, the proposal is accepted only if the observable criterion improves, otherwise the initial state is
    returned intact. `forced`: the optimiser always replaces the state (comparison arm only)."""
    prog = to_program(tree, atoms); e0 = float((predict(prog, X.float()) != y).sum())
    if fit == "none" or prog.n_params() == 0: return Finalization(prog, fit, False, e0, e0)
    data = FitData.build(X, y, 0.10); sk = prog.skeleton(); init = torch.tensor(prog.param_vector(), dtype=DT)
    if fit == "validated": row, tx = validated_fit(sk, data, init, steps=steps); out = sk.with_params(row.tolist()); ok = bool(tx.accepted)
    elif fit == "forced": th, _ = fit_batch(sk, data, init.reshape(1, -1), steps=steps); out = sk.with_params(th[0].tolist()); ok = True
    else: raise KeyError(fit)
    return Finalization(out, fit, ok, e0, float((predict(out, X.float()) != y).sum()))


# ================================================================ KNOWN / UNKNOWN / NONE_OF_THE_ABOVE

def knn_errors(X: torch.Tensor, y: torch.Tensor, k=7) -> float:
    """Errors of a k-nearest-neighbours classifier, leave-one-out: what a method WITHOUT structure
    already explains (same computation as `core.deep_boundary.knn_loo_error`, without an intermediate object)."""
    D = torch.cdist(X.double(), X.double()); D.fill_diagonal_(float("inf")); nb = D.topk(min(k, len(X) - 1), largest=False).indices; return float(((y[nb].float().mean(1) > 0.5).long() != y).sum())


def decide(result: CompositionResult, final_errors: float, X: torch.Tensor, y: torch.Tensor, gap_unknown=0.04, gap_none=0.10, tie=2.0) -> dict:
    """Three-outcome decision, on explicit signals:
      gap        (errors of the selected structure − errors of the nearest neighbours) / N: what the structure DOES NOT
                 EXPLAIN although the data allow it
      rivals     classes whose score is within `tie` of the first (competing structures not separated)
    gap ≥ gap_none → NONE_OF_THE_ABOVE; gap ≥ gap_unknown, or no structure → UNKNOWN (partial model); otherwise KNOWN."""
    N = len(X); kn = knn_errors(X, y); gap = (final_errors - kn) / max(1, N); rivals = sum(1 for c in result.ranked[1:] if result.ranked and abs(c.score - result.ranked[0].score) <= tie) if result.ranked else 0
    label = UNKNOWN if result.selected is None else NONE_OF_THE_ABOVE if gap >= gap_none else UNKNOWN if gap >= gap_unknown else KNOWN
    return {"label": label, "gap": gap, "knn_error_rate": kn / max(1, N), "error_rate": final_errors / max(1, N), "rivals": rivals, "partial_model": label == UNKNOWN and result.selected is not None}


# ================================================================ full chain

# `CompositionResult` receives two optional attributes set by `compose`: `units` (atoms and complements that the
# trees refer to) and `table` (all the classes built).

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
    """Primitive source of the main chain: `flat_scene_k` (candidate baseline) — each type tried on the
    whole scene, k hypotheses per type. No oracle."""
    from core.deep_fragments import flat_types
    from core.deep_partial import observe
    return [(h.program, float(h.score), f"flat_scene:{h.type}") for h in flat_types(observe(X, y), k, seed)]


def run_chain(X: torch.Tensor, y: torch.Tensor, seed=0, cfg: Optional[dict] = None, discovered: Optional[Sequence[Tuple[BNode, float, str]]] = None) -> EndToEnd:
    """discovery → hypotheses → reduction → composition over classes → finalisation → decision."""
    c = {**load_config(), **(cfg or {})}; t0 = time.time(); disc = list(discovered) if discovered is not None else discover(X, y, c["k"], seed); t1 = time.time(); H = hypotheses_from(disc, X, c["same"]); kept, info = reduce(H, y, c["reduction"], c["max_atoms"], c["min_gain"])
    st = {"literals": len(disc), "hypotheses": len(H), "kept": len(kept), "discover_seconds": round(t1 - t0, 3), "reduction": info}
    if not kept:
        empty = CompositionResult(None, [], {}); e = float(min(y.sum(), len(y) - y.sum())); d = decide(empty, e, X, y, c["gap_unknown"], c["gap_none"], c["tie"]); st["seconds"] = round(time.time() - t0, 3); return EndToEnd(None, d["label"], H, kept, empty, None, d, st)
    atoms = [h.program for h in kept]; R = compose(atoms, X, y, c["selector"]); F = finalize(R.selected.tree, R.units, X, y, c["fit"]); d = decide(R, F.errors_after, X, y, c["gap_unknown"], c["gap_none"], c["tie"])
    st.update({"classes": R.stats.get("classes"), "considered": R.stats.get("considered"), "compose_seconds": R.stats.get("seconds"), "seconds": round(time.time() - t0, 3)}); return EndToEnd(F.program, d["label"], H, kept, R, F, d, st)

"""Deep Composition with given atoms — deep composition when the atoms are GIVEN.

With fixed atoms, any Boolean composition is entirely described by its value on the CELLS: the classes
of observations that have the same vector of atom values. With n ≤ 6 atoms there are at most 64 occupied cells; a
composition then reduces to a 64-bit mask. This mask is the canonical functional signature of the structure
on the observations — two programs with the same mask are observationally equivalent.

Objects:
  CellSpace                      atoms → cells, label counts per cell, errors of a mask in O(1)
  ObservationalEquivalenceClass  a mask, its smallest known structure, its number of errors
  Table                          bottom-up dynamic programming: levels by size, ONE structure per class
  PartialStructure               structure with a hole "(A op ?B)"; the hole carries derived functional CONSTRAINTS
  CompositionResult              ranked classes, selected structure, log, counters

Searches (all bounded by a budget in evaluated pairs, no brute force over the programs):
  bottom_up      all the classes of size 1, then 2, …: the substructures are built once and reused;
                 with `beam`, a beam over the equivalence CLASSES (not over the programs)
  top_down       meet in the middle: the target is decomposed into "known left op hole", the constraints of the hole are
                 propagated, and the hole is solved in the table or decomposed in turn

This module sees neither truth nor world: it receives atoms, observations, and nothing else.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from core.boundary_program import BNode, predict

OPS = ("and", "or", "xor")
Tree = Union[int, tuple]                      # atom index | (op, left, right)
U = np.uint64
MAX_ATOMS = 6


# ================================================================ trees

def tree_size(t: Tree) -> int: return 1 if isinstance(t, (int, np.integer)) else tree_size(t[1]) + tree_size(t[2])
def tree_atoms(t: Tree) -> List[int]: return [int(t)] if isinstance(t, (int, np.integer)) else tree_atoms(t[1]) + tree_atoms(t[2])
def tree_depth(t: Tree) -> int: return 0 if isinstance(t, (int, np.integer)) else 1 + max(tree_depth(t[1]), tree_depth(t[2]))


def tree_eval(t: Tree, values):
    """Evaluates the tree on atom values: integers (cell masks) or Boolean / integer arrays."""
    if isinstance(t, (int, np.integer)): return values[int(t)]
    a, b = tree_eval(t[1], values), tree_eval(t[2], values); return a & b if t[0] == "and" else a | b if t[0] == "or" else a ^ b


def tree_form(t: Tree):
    """Canonical form up to associativity and commutativity (same rewrites as `algebraic_form`); hashable."""
    if isinstance(t, (int, np.integer)): return int(t)
    items = []
    def flat(z):
        if not isinstance(z, (int, np.integer)) and z[0] == t[0]: flat(z[1]); flat(z[2])
        else: items.append(tree_form(z))
    flat(t); return (t[0], tuple(sorted(items, key=repr)))


def tree_render(t: Tree) -> str: return f"a{int(t)}" if isinstance(t, (int, np.integer)) else f"({tree_render(t[1])} {t[0].upper()} {tree_render(t[2])})"
def to_program(t: Tree, atoms: Sequence[BNode]) -> BNode: return atoms[int(t)] if isinstance(t, (int, np.integer)) else BNode(t[0], children=(to_program(t[1], atoms), to_program(t[2], atoms)))


def from_program(p: BNode, atoms: Sequence[BNode]) -> Tree:
    """Program → tree over the given atoms (the leaves are recognised by equality of representation)."""
    if p.kind in OPS: return (p.kind, from_program(p.children[0], atoms), from_program(p.children[1], atoms))
    r = repr(p); return next(i for i, a in enumerate(atoms) if repr(a) == r)


def tree_shape(t: Tree) -> dict:
    """Shape of the tree: chain (each node has a leaf as a child), balanced (sizes of the two sides equal up to
    one, at each node), otherwise mixed; Strahler number; operators."""
    def chain(z): return isinstance(z, (int, np.integer)) or ((isinstance(z[1], (int, np.integer)) or isinstance(z[2], (int, np.integer))) and chain(z[1]) and chain(z[2]))
    def bal(z): return isinstance(z, (int, np.integer)) or (abs(tree_size(z[1]) - tree_size(z[2])) <= 1 and bal(z[1]) and bal(z[2]))
    def strahler(z):
        if isinstance(z, (int, np.integer)): return 1
        a, b = strahler(z[1]), strahler(z[2]); return max(a, b) if a != b else a + 1
    def ops(z): return [] if isinstance(z, (int, np.integer)) else [z[0]] + ops(z[1]) + ops(z[2])
    o = ops(t); n = tree_size(t); c, b = chain(t), bal(t)
    return {"size": n, "depth": tree_depth(t), "shape": "atom" if n == 1 else "pair" if n == 2 else "both" if c and b else "chain" if c else "balanced" if b else "mixed", "strahler": strahler(t), "operators": "none" if not o else "single" if len(set(o)) == 1 else "mixed",
            "n_xor": o.count("xor"), "n_and": o.count("and"), "n_or": o.count("or")}


# ================================================================ cells and signatures

class CellSpace:
    """Given atoms + observations → cells. A mask (64-bit integer, one bit per occupied cell) describes a
    composition; its number of errors on the observations is read directly from the counts per cell."""

    def __init__(self, atoms: Sequence[BNode], X: torch.Tensor, y: torch.Tensor, max_atoms: int = MAX_ATOMS):
        """`max_atoms` can be raised when units do not create new cells (an atom and its complement):
        the only real constraint is that the OCCUPIED cells fit in 64 bits."""
        if not 1 <= len(atoms) <= max_atoms: raise ValueError(f"between 1 and {max_atoms} atoms (the cells must fit in 64 bits)")
        self.atoms = list(atoms); self.n = len(atoms); self.N = len(X); code = self.codes_of(X); self.codes, inv = np.unique(code, return_inverse=True); self.K = len(self.codes); yb = y.numpy().astype(np.float64)
        if self.K > 64: raise ValueError(f"{self.K} occupied cells: more than 64")
        self.pos = np.bincount(inv, weights=yb, minlength=self.K); self.count = np.bincount(inv, minlength=self.K).astype(np.float64); self.neg = self.count - self.pos; self.gain = self.pos - self.neg; self.total_pos = float(self.pos.sum())
        self.atom_masks = [self._pack(((self.codes >> U(i)) & U(1)).astype(bool)) for i in range(self.n)]; self.full = (1 << self.K) - 1
        self.target = self._pack(self.pos > self.neg); self.floor = float(np.minimum(self.pos, self.neg).sum()); self._shift = np.arange(self.K, dtype=U)

    def codes_of(self, X: torch.Tensor) -> np.ndarray:
        """Cell code of each point: one bit per atom."""
        c = np.zeros(len(X), dtype=U)
        for i, a in enumerate(self.atoms): c |= predict(a, X.float()).numpy().astype(U) << U(i)
        return c

    @staticmethod
    def _pack(bits: np.ndarray) -> int: return int(sum(1 << i for i, b in enumerate(bits) if b))

    def bits(self, masks: np.ndarray) -> np.ndarray: return ((masks[:, None] >> self._shift[None, :]) & U(1)).astype(np.float64)
    def errors(self, masks) -> np.ndarray:
        """Errors on the observations of each mask: positives of the cells at 0 + negatives of the cells at 1."""
        m = np.atleast_1d(np.asarray(masks, dtype=U)); return self.total_pos - self.bits(m) @ self.gain
    def mask_of(self, t: Tree) -> int: return int(tree_eval(t, self.atom_masks)) & self.full
    def value_on_codes(self, t: Tree, codes: np.ndarray) -> np.ndarray:
        """Value of the tree on arbitrary points given by their cell code (including cells never observed)."""
        return tree_eval(t, [((codes >> U(i)) & U(1)).astype(bool) for i in range(self.n)])
    def unseen_mass(self, X: torch.Tensor) -> float:
        """Share of the points of X that fall in a cell WITHOUT any observation: there, the observations do not
        constrain the composition — intrinsic ambiguity."""
        return float((~np.isin(self.codes_of(X), self.codes)).mean()) if len(X) else 0.0


@dataclass
class ObservationalEquivalenceClass:
    """All the structures with the same mask are indistinguishable on the observations; only one is kept."""
    mask: int
    tree: Tree
    size: int
    errors: float
    score: float = 0.0


# ================================================================ bottom-up dynamic programming

@dataclass
class Level:
    size: int
    masks: np.ndarray               # [M] uint64, classes NEW at this size (absent from the smaller sizes)
    op: np.ndarray                  # [M] 0 / 1 / 2 (and / or / xor), −1 for an atom
    li: np.ndarray                  # size of the left child
    ai: np.ndarray                  # index of the left child in its level
    aj: np.ndarray                  # index of the right child in the level (size − li)
    considered: int = 0             # programs (pairs × operators) evaluated to build this level
    distinct: int = 0               # distinct masks produced at this size
    truncated: bool = False


@dataclass
class Table:
    space: CellSpace
    levels: Dict[int, Level]
    stats: Dict[str, float]

    def tree(self, size: int, idx: int) -> Tree:
        L = self.levels[size]
        if L.op[idx] < 0: return int(L.ai[idx])
        return (OPS[int(L.op[idx])], self.tree(int(L.li[idx]), int(L.ai[idx])), self.tree(size - int(L.li[idx]), int(L.aj[idx])))

    def all(self, max_size: Optional[int] = None):
        """(masks, sizes, indices in the level) of all the classes up to `max_size`."""
        ss = [s for s in sorted(self.levels) if max_size is None or s <= max_size]
        if not ss: return np.zeros(0, dtype=U), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        return np.concatenate([self.levels[s].masks for s in ss]), np.concatenate([np.full(len(self.levels[s].masks), s) for s in ss]), np.concatenate([np.arange(len(self.levels[s].masks)) for s in ss])

    def find(self, mask: int) -> Optional[Tuple[int, int]]:
        for s in sorted(self.levels):
            w = np.nonzero(self.levels[s].masks == U(mask))[0]
            if len(w): return s, int(w[0])
        return None

    @property
    def classes(self): return int(sum(len(l.masks) for l in self.levels.values()))


def _apply(op: int, a: np.ndarray, b: np.ndarray) -> np.ndarray: return a & b if op == 0 else a | b if op == 1 else a ^ b


def bottom_up(space: CellSpace, max_size: int, beam: Optional[int] = None, budget: Optional[int] = None) -> Table:
    """Factorised bottom-up composition. Level s = classes reached with s leaves and no fewer; it is obtained by
    combining the levels i and s − i already built: each substructure is computed only once.
    `beam`: maximal number of CLASSES kept per level (those with the fewest errors). `budget`: total number of pairs ×
    operators that can be evaluated; beyond it, the current level is truncated and the construction stops."""
    t0 = time.time(); n = space.n; m0 = np.array(space.atom_masks, dtype=U); u, first = np.unique(m0, return_index=True); order = np.sort(first)
    L1 = Level(1, m0[order], np.full(len(order), -1, dtype=np.int8), np.zeros(len(order), dtype=np.int8), order.astype(np.int64), np.zeros(len(order), dtype=np.int64), n, len(order)); levels = {1: L1}; seen = np.sort(L1.masks); used = 0; stop = False
    for s in range(2, max_size + 1):
        segs = []; cands = []
        for i in range(1, s // 2 + 1):
            j = s - i
            if i not in levels or j not in levels: continue
            a, b = levels[i].masks, levels[j].masks; cost = 3 * len(a) * len(b)
            if budget is not None and used + cost > budget:
                keep = max(0, (budget - used) // (3 * max(1, len(b)))); a = a[:keep]; cost = 3 * len(a) * len(b); stop = True
            if not len(a) or not len(b): continue
            for op in range(3): cands.append(_apply(op, a[:, None], b[None, :]).ravel()); segs.append((op, i, len(a), len(b)))
            used += cost
        if not cands: break
        allm = np.concatenate(cands); uniq, first = np.unique(allm, return_index=True); new = ~np.isin(uniq, seen, assume_unique=True); uniq, first = uniq[new], first[new]; distinct = len(uniq)
        if beam is not None and len(uniq) > beam: e = space.errors(uniq); k = np.argpartition(e, beam - 1)[:beam]; uniq, first = uniq[k], first[k]
        o = np.argsort(first, kind="stable"); uniq, first = uniq[o], first[o]; op = np.zeros(len(uniq), dtype=np.int8); li = np.zeros(len(uniq), dtype=np.int8); ai = np.zeros(len(uniq), dtype=np.int64); aj = np.zeros(len(uniq), dtype=np.int64); off = 0
        for sop, si, la, lb in segs:
            m = (first >= off) & (first < off + la * lb); r = first[m] - off; op[m] = sop; li[m] = si; ai[m] = r // lb; aj[m] = r % lb; off += la * lb
        levels[s] = Level(s, uniq, op, li, ai, aj, int(len(allm)), distinct, stop); seen = np.union1d(seen, uniq)
        if stop: break
    return Table(space, levels, {"considered": int(sum(l.considered for l in levels.values())), "classes": int(sum(len(l.masks) for l in levels.values())), "max_size": max(levels), "truncated": stop, "seconds": round(time.time() - t0, 4)})


# ================================================================ ranking of the classes

def profiled_ll(errors, n: float):
    """Likelihood of the hard predictions with profiled noise (same formula as `core.deep_fit.profiled_loglik`)."""
    e = np.asarray(errors, dtype=np.float64); eps = np.clip((e + 0.5) / (n + 1.0), 1e-3, 0.49); return e * np.log(eps) + (n - e) * np.log(1 - eps)


def class_scores(space: CellSpace, masks: np.ndarray, sizes: np.ndarray, params_per_atom: Optional[float] = None) -> Dict[str, np.ndarray]:
    """INTERNAL scores of a class, separated: errors, profiled likelihood, structural penalty (that of the existing
    posterior score: (2s − 1)·3·ln 2), parameter penalty (½·k·ln N), posterior score, size."""
    e = space.errors(masks); ll = profiled_ll(e, float(space.N)); p = float(np.mean([a.n_params() for a in space.atoms])) if params_per_atom is None else params_per_atom
    pen_s = (2 * sizes - 1) * 3.0 * math.log(2); pen_k = 0.5 * p * sizes * math.log(max(1, space.N))
    return {"errors": e, "likelihood": ll, "structural_penalty": pen_s, "parameter_penalty": pen_k, "posterior": ll - pen_s - pen_k, "bic": ll - 0.5 * sizes * math.log(max(1, space.N)), "size": sizes.astype(np.float64)}


SELECTORS = ("errors", "likelihood", "bic", "posterior", "size")


def order_by(scores: Dict[str, np.ndarray], selector: str) -> np.ndarray:
    """Indices of the classes, from the best to the worst. `errors`: fewer errors, then smaller; `posterior`:
    likelihood minus penalties of the existing score; `bic`: likelihood minus ½·ln N per leaf (the atoms are given:
    only the choice of the leaf is paid for); `likelihood`: likelihood alone; `size`: smaller, then fewer errors."""
    if selector == "errors": return np.lexsort((scores["size"], scores["errors"]))
    if selector == "likelihood": return np.lexsort((scores["size"], -scores["likelihood"]))
    if selector in ("posterior", "bic"): return np.lexsort((scores["size"], -scores[selector]))
    if selector == "size": return np.lexsort((scores["errors"], scores["size"]))
    raise KeyError(selector)


@dataclass
class CompositionResult:
    selected: Optional[ObservationalEquivalenceClass]
    ranked: List[ObservationalEquivalenceClass]         # the `top` best classes according to the selector
    stats: Dict[str, float]
    log: List[dict] = field(default_factory=list)


def select(table: Table, selector="posterior", top=20, max_size: Optional[int] = None) -> CompositionResult:
    sp = table.space; masks, sizes, idx = table.all(max_size)
    if not len(masks): return CompositionResult(None, [], dict(table.stats))
    sc = class_scores(sp, masks, sizes); o = order_by(sc, selector)[:top]
    R = [ObservationalEquivalenceClass(int(masks[i]), table.tree(int(sizes[i]), int(idx[i])), int(sizes[i]), float(sc["errors"][i]), float(sc[selector][i])) for i in o]
    return CompositionResult(R[0], R, {**table.stats, "selector": selector, "floor_errors": sp.floor, "cells": sp.K})


def rank_of(table: Table, mask: int, selector="posterior", max_size: Optional[int] = None) -> Optional[int]:
    """Rank (1 = first) of the class with the given mask among all the classes built; None if it is absent."""
    masks, sizes, _ = table.all(max_size); w = np.nonzero(masks == U(mask))[0]
    if not len(w): return None
    o = order_by(class_scores(table.space, masks, sizes), selector); return int(np.nonzero(o == w[0])[0][0]) + 1


# ================================================================ structures with a hole and constraint propagation

@dataclass
class PartialStructure:
    """"(left op ?hole)": the hole carries the TARGET it must realise and the cells where its value matters."""
    op: str
    left: Tree
    target: int                     # required value of the hole, cell by cell
    care: int                       # cells where the value of the hole has an effect on the result
    violation: float                # weight of the cells already lost whatever the hole
    size_left: int

    def render(self) -> str: return f"({tree_render(self.left)} {self.op.upper()} ?)"


def propagate(op: str, left: int, target: int, care: int, full: int) -> Tuple[int, int, int]:
    """Constraints of the right subprogram for (left op right) to equal `target` on the cells `care`.
      and : where left = 1, right must equal the target; where left = 0 the result is 0 (lost if the target is 1)
      or  : where left = 0, right must equal the target; where left = 1 the result is 1 (lost if the target is 0)
      xor : right = left xor target, everywhere
    Returns (target of the right, cells that matter for the right, lost cells)."""
    if op == "and": return target & left & full, care & left, care & ~left & target & full
    if op == "or": return target & ~left & full, care & ~left & full, care & left & ~target & full
    if op == "xor": return (left ^ target) & full, care, 0
    raise KeyError(op)


def _weights(space: CellSpace) -> np.ndarray: return np.abs(space.gain)


def _wsum(space: CellSpace, mask: int, w: np.ndarray) -> float: return float(sum(w[c] for c in range(space.K) if (mask >> c) & 1))


def top_down(space: CellSpace, table: Table, size: int, beam=6, budget=4000, half: Optional[int] = None) -> CompositionResult:
    """Meet in the middle, guided by constraints. The target (majority label of each cell) is decomposed into
    "left op ?hole" with a left taken from the table (size ≤ `half`); the constraints of the hole are propagated;
    the hole is solved by the class of the table that violates its constraints the least, or decomposed in turn if the
    remaining size exceeds `half`. `beam` lefts kept per hole; `budget` = total number of table joins."""
    t0 = time.time(); w = _weights(space); half = half or max(l for l in table.levels); masks, sizes, idx = table.all(half); B = space.bits(masks); joins = 0; log: List[dict] = []
    if not len(masks): return CompositionResult(None, [], {"joins": 0, "seconds": 0.0})

    def best_in_table(target: int, care: int, max_size: int):
        """Class of size ≤ max_size that violates the target the least on the cells that matter (weighted cost)."""
        nonlocal joins; joins += 1; t = np.array([(target >> c) & 1 for c in range(space.K)], dtype=np.float64); cr = np.array([(care >> c) & 1 for c in range(space.K)], dtype=np.float64) * w
        cost = B @ (cr * (1 - t)) + (1 - B) @ (cr * t); cost = np.where(sizes <= max_size, cost, np.inf); i = int(np.lexsort((sizes, cost))[0]); return i, float(cost[i])

    def solve(target: int, care: int, max_size: int, depth: int):
        """Returns (tree, weighted cost on `care`)."""
        i, c = best_in_table(target, care, min(max_size, half)); best = (table.tree(int(sizes[i]), int(idx[i])), c)
        if max_size <= half or c <= 1e-9 or joins >= budget or depth >= 4: return best
        t = np.array([(target >> k) & 1 for k in range(space.K)], dtype=np.float64); cr = np.array([(care >> k) & 1 for k in range(space.K)], dtype=np.float64) * w; cands = []
        for op in OPS:                                                             # bound per left: what is lost outright + the best right FROM THE TABLE
            if op == "and": viol = (1 - B) @ (cr * t); A = B * (cr * (1 - t))[None, :]; C = B * (cr * t)[None, :]
            elif op == "or": viol = B @ (cr * (1 - t)); A = (1 - B) * (cr * (1 - t))[None, :]; C = (1 - B) * (cr * t)[None, :]
            else: viol = np.zeros(len(B)); tl = np.abs(B - t[None, :]); A = (1 - tl) * cr[None, :]; C = tl * cr[None, :]
            joined = (A @ B.T + C.sum(1, keepdims=True) - C @ B.T).min(1); lb = viol + joined; nonlocal_join()
            for j in np.argsort(lb, kind="stable")[: beam]:
                if sizes[j] < max_size: cands.append((float(lb[j]), op, int(j), float(viol[j])))
        for lb, op, j, viol in sorted(cands):                                        # `beam` lefts PER operator: an operator without immediate loss (xor) must not crowd out the others
            if joins >= budget: break
            tr, cr2, _ = propagate(op, int(masks[j]), target, care, space.full); left = table.tree(int(sizes[j]), int(idx[j])); ps = PartialStructure(op, left, tr, cr2, viol, int(sizes[j]))
            rt, rc = solve(tr, cr2, max_size - int(sizes[j]), depth + 1); tot = viol + rc; log.append({"partial": ps.render(), "depth": depth, "violation": round(viol, 3), "hole_cost": round(rc, 3), "bound": round(lb, 3)})
            if tot < best[1] - 1e-9: best = ((op, left, rt), tot)
        return best

    def nonlocal_join():
        nonlocal joins; joins += 1

    tree, cost = solve(space.target, space.full, size, 0); m = space.mask_of(tree); e = float(space.errors(np.array([m], dtype=U))[0]); cls = ObservationalEquivalenceClass(m, tree, tree_size(tree), e, -cost)
    return CompositionResult(cls, [cls], {"joins": joins, "budget": budget, "table_classes": int(len(masks)), "half": half, "seconds": round(time.time() - t0, 4), "exhausted": joins >= budget, "floor_errors": space.floor, "cells": space.K}, log)


# ================================================================ scaling law of the composition space

def scaling(space: CellSpace, max_size: int, budget: Optional[int] = None) -> List[dict]:
    """size → evaluated programs → distinct masks → NEW classes → cumulative classes → quasi-optimal classes."""
    T = bottom_up(space, max_size, None, budget); out = []; cum = 0; best = None
    for s in sorted(T.levels):
        L = T.levels[s]; cum += len(L.masks); e = space.errors(L.masks) if len(L.masks) else np.zeros(0); best = float(e.min()) if best is None and len(e) else min(best, float(e.min())) if len(e) else best
        out.append({"size": s, "programs": L.considered, "distinct_masks": L.distinct, "new_classes": int(len(L.masks)), "cumulative_classes": cum, "min_errors": float(e.min()) if len(e) else None, "best_so_far": best, "truncated": L.truncated})
    return out

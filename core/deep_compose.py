"""Deep Composition with given atoms — composition profonde quand les atomes sont DONNÉS.

Avec des atomes fixés, toute composition booléenne est entièrement décrite par sa valeur sur les CELLULES : les classes
d'observations qui ont le même vecteur de valeurs d'atomes. Avec n ≤ 6 atomes il y a au plus 64 cellules occupées ; une
composition se résume alors à un masque de 64 bits. Ce masque est la signature fonctionnelle canonique de la structure
sur les observations — deux programmes de même masque sont observationnellement équivalents.

Objets :
  CellSpace                      atomes → cellules, comptes d'étiquettes par cellule, erreurs d'un masque en O(1)
  ObservationalEquivalenceClass  un masque, sa plus petite structure connue, son nombre d'erreurs
  Table                          programmation dynamique ascendante : niveaux par taille, UNE structure par classe
  PartialStructure               structure à trou « (A op ?B) » ; le trou porte des CONTRAINTES fonctionnelles dérivées
  CompositionResult              classes classées, structure retenue, journal, compteurs

Recherches (toutes bornées par un budget en paires évaluées, aucune force brute sur les programmes) :
  bottom_up      toutes les classes de taille 1, puis 2, … : les sous-structures sont construites une fois et réutilisées ;
                 avec `beam`, un faisceau sur les CLASSES d'équivalence (pas sur les programmes)
  top_down       rencontre au milieu : la cible est décomposée en « gauche connue op trou », les contraintes du trou sont
                 propagées, et le trou est résolu dans la table ou décomposé à son tour

Ce module ne voit ni vérité ni monde : il reçoit des atomes, des observations, et rien d'autre.
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
Tree = Union[int, tuple]                      # indice d'atome | (op, gauche, droite)
U = np.uint64
MAX_ATOMS = 6


# ================================================================ arbres

def tree_size(t: Tree) -> int: return 1 if isinstance(t, (int, np.integer)) else tree_size(t[1]) + tree_size(t[2])
def tree_atoms(t: Tree) -> List[int]: return [int(t)] if isinstance(t, (int, np.integer)) else tree_atoms(t[1]) + tree_atoms(t[2])
def tree_depth(t: Tree) -> int: return 0 if isinstance(t, (int, np.integer)) else 1 + max(tree_depth(t[1]), tree_depth(t[2]))


def tree_eval(t: Tree, values):
    """Évalue l'arbre sur des valeurs d'atomes : entiers (masques de cellules) ou tableaux booléens / entiers."""
    if isinstance(t, (int, np.integer)): return values[int(t)]
    a, b = tree_eval(t[1], values), tree_eval(t[2], values); return a & b if t[0] == "and" else a | b if t[0] == "or" else a ^ b


def tree_form(t: Tree):
    """Forme canonique à associativité et commutativité près (mêmes réécritures que `algebraic_form`) ; hachable."""
    if isinstance(t, (int, np.integer)): return int(t)
    items = []
    def flat(z):
        if not isinstance(z, (int, np.integer)) and z[0] == t[0]: flat(z[1]); flat(z[2])
        else: items.append(tree_form(z))
    flat(t); return (t[0], tuple(sorted(items, key=repr)))


def tree_render(t: Tree) -> str: return f"a{int(t)}" if isinstance(t, (int, np.integer)) else f"({tree_render(t[1])} {t[0].upper()} {tree_render(t[2])})"
def to_program(t: Tree, atoms: Sequence[BNode]) -> BNode: return atoms[int(t)] if isinstance(t, (int, np.integer)) else BNode(t[0], children=(to_program(t[1], atoms), to_program(t[2], atoms)))


def from_program(p: BNode, atoms: Sequence[BNode]) -> Tree:
    """Programme → arbre sur les atomes donnés (les feuilles sont reconnues par égalité de représentation)."""
    if p.kind in OPS: return (p.kind, from_program(p.children[0], atoms), from_program(p.children[1], atoms))
    r = repr(p); return next(i for i, a in enumerate(atoms) if repr(a) == r)


def tree_shape(t: Tree) -> dict:
    """Forme de l'arbre : chaîne (chaque nœud a une feuille pour enfant), équilibré (tailles des deux côtés égales à un
    près, à chaque nœud), sinon mixte ; nombre de Strahler ; opérateurs."""
    def chain(z): return isinstance(z, (int, np.integer)) or ((isinstance(z[1], (int, np.integer)) or isinstance(z[2], (int, np.integer))) and chain(z[1]) and chain(z[2]))
    def bal(z): return isinstance(z, (int, np.integer)) or (abs(tree_size(z[1]) - tree_size(z[2])) <= 1 and bal(z[1]) and bal(z[2]))
    def strahler(z):
        if isinstance(z, (int, np.integer)): return 1
        a, b = strahler(z[1]), strahler(z[2]); return max(a, b) if a != b else a + 1
    def ops(z): return [] if isinstance(z, (int, np.integer)) else [z[0]] + ops(z[1]) + ops(z[2])
    o = ops(t); n = tree_size(t); c, b = chain(t), bal(t)
    return {"size": n, "depth": tree_depth(t), "shape": "atom" if n == 1 else "pair" if n == 2 else "both" if c and b else "chain" if c else "balanced" if b else "mixed", "strahler": strahler(t), "operators": "none" if not o else "single" if len(set(o)) == 1 else "mixed",
            "n_xor": o.count("xor"), "n_and": o.count("and"), "n_or": o.count("or")}


# ================================================================ cellules et signatures

class CellSpace:
    """Atomes donnés + observations → cellules. Un masque (entier 64 bits, un bit par cellule occupée) décrit une
    composition ; son nombre d'erreurs sur les observations se lit directement sur les comptes par cellule."""

    def __init__(self, atoms: Sequence[BNode], X: torch.Tensor, y: torch.Tensor, max_atoms: int = MAX_ATOMS):
        """`max_atoms` peut être relevé quand des unités ne créent pas de cellules nouvelles (un atome et son complément) :
        la seule contrainte réelle est que les cellules OCCUPÉES tiennent sur 64 bits."""
        if not 1 <= len(atoms) <= max_atoms: raise ValueError(f"entre 1 et {max_atoms} atomes (les cellules doivent tenir sur 64 bits)")
        self.atoms = list(atoms); self.n = len(atoms); self.N = len(X); code = self.codes_of(X); self.codes, inv = np.unique(code, return_inverse=True); self.K = len(self.codes); yb = y.numpy().astype(np.float64)
        if self.K > 64: raise ValueError(f"{self.K} cellules occupées : plus de 64")
        self.pos = np.bincount(inv, weights=yb, minlength=self.K); self.count = np.bincount(inv, minlength=self.K).astype(np.float64); self.neg = self.count - self.pos; self.gain = self.pos - self.neg; self.total_pos = float(self.pos.sum())
        self.atom_masks = [self._pack(((self.codes >> U(i)) & U(1)).astype(bool)) for i in range(self.n)]; self.full = (1 << self.K) - 1
        self.target = self._pack(self.pos > self.neg); self.floor = float(np.minimum(self.pos, self.neg).sum()); self._shift = np.arange(self.K, dtype=U)

    def codes_of(self, X: torch.Tensor) -> np.ndarray:
        """Code de cellule de chaque point : un bit par atome."""
        c = np.zeros(len(X), dtype=U)
        for i, a in enumerate(self.atoms): c |= predict(a, X.float()).numpy().astype(U) << U(i)
        return c

    @staticmethod
    def _pack(bits: np.ndarray) -> int: return int(sum(1 << i for i, b in enumerate(bits) if b))

    def bits(self, masks: np.ndarray) -> np.ndarray: return ((masks[:, None] >> self._shift[None, :]) & U(1)).astype(np.float64)
    def errors(self, masks) -> np.ndarray:
        """Erreurs sur les observations de chaque masque : positives des cellules à 0 + négatives des cellules à 1."""
        m = np.atleast_1d(np.asarray(masks, dtype=U)); return self.total_pos - self.bits(m) @ self.gain
    def mask_of(self, t: Tree) -> int: return int(tree_eval(t, self.atom_masks)) & self.full
    def value_on_codes(self, t: Tree, codes: np.ndarray) -> np.ndarray:
        """Valeur de l'arbre sur des points quelconques donnés par leur code de cellule (y compris des cellules jamais observées)."""
        return tree_eval(t, [((codes >> U(i)) & U(1)).astype(bool) for i in range(self.n)])
    def unseen_mass(self, X: torch.Tensor) -> float:
        """Part des points de X qui tombent dans une cellule SANS aucune observation : là, les observations ne
        contraignent pas la composition — ambiguïté intrinsèque."""
        return float((~np.isin(self.codes_of(X), self.codes)).mean()) if len(X) else 0.0


@dataclass
class ObservationalEquivalenceClass:
    """Toutes les structures de même masque sont indiscernables sur les observations ; une seule est gardée."""
    mask: int
    tree: Tree
    size: int
    errors: float
    score: float = 0.0


# ================================================================ programmation dynamique ascendante

@dataclass
class Level:
    size: int
    masks: np.ndarray               # [M] uint64, classes NOUVELLES à cette taille (absentes des tailles inférieures)
    op: np.ndarray                  # [M] 0 / 1 / 2 (and / or / xor), −1 pour un atome
    li: np.ndarray                  # taille de l'enfant gauche
    ai: np.ndarray                  # indice de l'enfant gauche dans son niveau
    aj: np.ndarray                  # indice de l'enfant droit dans le niveau (size − li)
    considered: int = 0             # programmes (paires × opérateurs) évalués pour construire ce niveau
    distinct: int = 0               # masques distincts produits à cette taille
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
        """(masques, tailles, indices dans le niveau) de toutes les classes jusqu'à `max_size`."""
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
    """Composition ascendante factorisée. Niveau s = classes atteintes avec s feuilles et pas moins ; il est obtenu en
    combinant les niveaux i et s − i déjà construits : chaque sous-structure n'est calculée qu'une fois.
    `beam` : nombre maximal de CLASSES gardées par niveau (les moins erronées). `budget` : nombre total de paires ×
    opérateurs évaluables ; au-delà, le niveau en cours est tronqué et la construction s'arrête."""
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


# ================================================================ classement des classes

def profiled_ll(errors, n: float):
    """Vraisemblance des prédictions dures à bruit profilé (même formule que `core.deep_fit.profiled_loglik`)."""
    e = np.asarray(errors, dtype=np.float64); eps = np.clip((e + 0.5) / (n + 1.0), 1e-3, 0.49); return e * np.log(eps) + (n - e) * np.log(1 - eps)


def class_scores(space: CellSpace, masks: np.ndarray, sizes: np.ndarray, params_per_atom: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Scores INTERNES d'une classe, séparés : erreurs, vraisemblance profilée, pénalité de structure (celle du score a
    posteriori existant : (2s − 1)·3·ln 2), pénalité de paramètres (½·k·ln N), score a posteriori, taille."""
    e = space.errors(masks); ll = profiled_ll(e, float(space.N)); p = float(np.mean([a.n_params() for a in space.atoms])) if params_per_atom is None else params_per_atom
    pen_s = (2 * sizes - 1) * 3.0 * math.log(2); pen_k = 0.5 * p * sizes * math.log(max(1, space.N))
    return {"errors": e, "likelihood": ll, "structural_penalty": pen_s, "parameter_penalty": pen_k, "posterior": ll - pen_s - pen_k, "bic": ll - 0.5 * sizes * math.log(max(1, space.N)), "size": sizes.astype(np.float64)}


SELECTORS = ("errors", "likelihood", "bic", "posterior", "size")


def order_by(scores: Dict[str, np.ndarray], selector: str) -> np.ndarray:
    """Indices des classes, de la meilleure à la moins bonne. `errors` : moins d'erreurs, puis plus petite ; `posterior` :
    vraisemblance moins pénalités du score existant ; `bic` : vraisemblance moins ½·ln N par feuille (les atomes sont donnés :
    seul le choix de la feuille est payé) ; `likelihood` : vraisemblance seule ; `size` : plus petite, puis moins d'erreurs."""
    if selector == "errors": return np.lexsort((scores["size"], scores["errors"]))
    if selector == "likelihood": return np.lexsort((scores["size"], -scores["likelihood"]))
    if selector in ("posterior", "bic"): return np.lexsort((scores["size"], -scores[selector]))
    if selector == "size": return np.lexsort((scores["errors"], scores["size"]))
    raise KeyError(selector)


@dataclass
class CompositionResult:
    selected: Optional[ObservationalEquivalenceClass]
    ranked: List[ObservationalEquivalenceClass]         # les `top` meilleures classes selon le sélecteur
    stats: Dict[str, float]
    log: List[dict] = field(default_factory=list)


def select(table: Table, selector="posterior", top=20, max_size: Optional[int] = None) -> CompositionResult:
    sp = table.space; masks, sizes, idx = table.all(max_size)
    if not len(masks): return CompositionResult(None, [], dict(table.stats))
    sc = class_scores(sp, masks, sizes); o = order_by(sc, selector)[:top]
    R = [ObservationalEquivalenceClass(int(masks[i]), table.tree(int(sizes[i]), int(idx[i])), int(sizes[i]), float(sc["errors"][i]), float(sc[selector][i])) for i in o]
    return CompositionResult(R[0], R, {**table.stats, "selector": selector, "floor_errors": sp.floor, "cells": sp.K})


def rank_of(table: Table, mask: int, selector="posterior", max_size: Optional[int] = None) -> Optional[int]:
    """Rang (1 = premier) de la classe de masque donné parmi toutes les classes construites ; None si elle est absente."""
    masks, sizes, _ = table.all(max_size); w = np.nonzero(masks == U(mask))[0]
    if not len(w): return None
    o = order_by(class_scores(table.space, masks, sizes), selector); return int(np.nonzero(o == w[0])[0][0]) + 1


# ================================================================ structures à trou et propagation de contraintes

@dataclass
class PartialStructure:
    """« (gauche op ?trou) » : le trou porte la CIBLE qu'il doit réaliser et les cellules où sa valeur compte."""
    op: str
    left: Tree
    target: int                     # valeur requise du trou, cellule par cellule
    care: int                       # cellules où la valeur du trou a un effet sur le résultat
    violation: float                # poids des cellules déjà perdues quel que soit le trou
    size_left: int

    def render(self) -> str: return f"({tree_render(self.left)} {self.op.upper()} ?)"


def propagate(op: str, left: int, target: int, care: int, full: int) -> Tuple[int, int, int]:
    """Contraintes du sous-programme droit pour que (gauche op droit) vaille `target` sur les cellules `care`.
      and : là où gauche = 1, droit doit valoir la cible ; là où gauche = 0 le résultat est 0 (perdu si la cible vaut 1)
      or  : là où gauche = 0, droit doit valoir la cible ; là où gauche = 1 le résultat est 1 (perdu si la cible vaut 0)
      xor : droit = gauche xor cible, partout
    Renvoie (cible du droit, cellules qui comptent pour le droit, cellules perdues)."""
    if op == "and": return target & left & full, care & left, care & ~left & target & full
    if op == "or": return target & ~left & full, care & ~left & full, care & left & ~target & full
    if op == "xor": return (left ^ target) & full, care, 0
    raise KeyError(op)


def _weights(space: CellSpace) -> np.ndarray: return np.abs(space.gain)


def _wsum(space: CellSpace, mask: int, w: np.ndarray) -> float: return float(sum(w[c] for c in range(space.K) if (mask >> c) & 1))


def top_down(space: CellSpace, table: Table, size: int, beam=6, budget=4000, half: Optional[int] = None) -> CompositionResult:
    """Rencontre au milieu, guidée par contraintes. La cible (étiquette majoritaire de chaque cellule) est décomposée en
    « gauche op ?trou » avec une gauche prise dans la table (taille ≤ `half`) ; les contraintes du trou sont propagées ;
    le trou est résolu par la classe de la table qui viole le moins ses contraintes, ou décomposé à son tour si la
    taille restante dépasse `half`. `beam` gauches gardées par trou ; `budget` = nombre total de jointures de table."""
    t0 = time.time(); w = _weights(space); half = half or max(l for l in table.levels); masks, sizes, idx = table.all(half); B = space.bits(masks); joins = 0; log: List[dict] = []
    if not len(masks): return CompositionResult(None, [], {"joins": 0, "seconds": 0.0})

    def best_in_table(target: int, care: int, max_size: int):
        """Classe de taille ≤ max_size qui viole le moins la cible sur les cellules qui comptent (coût pondéré)."""
        nonlocal joins; joins += 1; t = np.array([(target >> c) & 1 for c in range(space.K)], dtype=np.float64); cr = np.array([(care >> c) & 1 for c in range(space.K)], dtype=np.float64) * w
        cost = B @ (cr * (1 - t)) + (1 - B) @ (cr * t); cost = np.where(sizes <= max_size, cost, np.inf); i = int(np.lexsort((sizes, cost))[0]); return i, float(cost[i])

    def solve(target: int, care: int, max_size: int, depth: int):
        """Renvoie (arbre, coût pondéré sur `care`)."""
        i, c = best_in_table(target, care, min(max_size, half)); best = (table.tree(int(sizes[i]), int(idx[i])), c)
        if max_size <= half or c <= 1e-9 or joins >= budget or depth >= 4: return best
        t = np.array([(target >> k) & 1 for k in range(space.K)], dtype=np.float64); cr = np.array([(care >> k) & 1 for k in range(space.K)], dtype=np.float64) * w; cands = []
        for op in OPS:                                                             # borne par gauche : ce qui est perdu d'office + le meilleur droit DE LA TABLE
            if op == "and": viol = (1 - B) @ (cr * t); A = B * (cr * (1 - t))[None, :]; C = B * (cr * t)[None, :]
            elif op == "or": viol = B @ (cr * (1 - t)); A = (1 - B) * (cr * (1 - t))[None, :]; C = (1 - B) * (cr * t)[None, :]
            else: viol = np.zeros(len(B)); tl = np.abs(B - t[None, :]); A = (1 - tl) * cr[None, :]; C = tl * cr[None, :]
            joined = (A @ B.T + C.sum(1, keepdims=True) - C @ B.T).min(1); lb = viol + joined; nonlocal_join()
            for j in np.argsort(lb, kind="stable")[: beam]:
                if sizes[j] < max_size: cands.append((float(lb[j]), op, int(j), float(viol[j])))
        for lb, op, j, viol in sorted(cands):                                        # `beam` gauches PAR opérateur : un opérateur sans perte immédiate (xor) ne doit pas évincer les autres
            if joins >= budget: break
            tr, cr2, _ = propagate(op, int(masks[j]), target, care, space.full); left = table.tree(int(sizes[j]), int(idx[j])); ps = PartialStructure(op, left, tr, cr2, viol, int(sizes[j]))
            rt, rc = solve(tr, cr2, max_size - int(sizes[j]), depth + 1); tot = viol + rc; log.append({"partial": ps.render(), "depth": depth, "violation": round(viol, 3), "hole_cost": round(rc, 3), "bound": round(lb, 3)})
            if tot < best[1] - 1e-9: best = ((op, left, rt), tot)
        return best

    def nonlocal_join():
        nonlocal joins; joins += 1

    tree, cost = solve(space.target, space.full, size, 0); m = space.mask_of(tree); e = float(space.errors(np.array([m], dtype=U))[0]); cls = ObservationalEquivalenceClass(m, tree, tree_size(tree), e, -cost)
    return CompositionResult(cls, [cls], {"joins": joins, "budget": budget, "table_classes": int(len(masks)), "half": half, "seconds": round(time.time() - t0, 4), "exhausted": joins >= budget, "floor_errors": space.floor, "cells": space.K}, log)


# ================================================================ loi d'échelle de l'espace de composition

def scaling(space: CellSpace, max_size: int, budget: Optional[int] = None) -> List[dict]:
    """taille → programmes évalués → masques distincts → classes NOUVELLES → classes cumulées → classes quasi optimales."""
    T = bottom_up(space, max_size, None, budget); out = []; cum = 0; best = None
    for s in sorted(T.levels):
        L = T.levels[s]; cum += len(L.masks); e = space.errors(L.masks) if len(L.masks) else np.zeros(0); best = float(e.min()) if best is None and len(e) else min(best, float(e.min())) if len(e) else best
        out.append({"size": s, "programs": L.considered, "distinct_masks": L.distinct, "new_classes": int(len(L.masks)), "cumulative_classes": cum, "min_errors": float(e.min()) if len(e) else None, "best_so_far": best, "truncated": L.truncated})
    return out

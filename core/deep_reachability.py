"""REACHABILITY diagnostic (Deep Candidate Generation sub-milestone) — EVALUATION module.

It receives the true structure: it serves ONLY to answer "is this structure representable, reachable by
our operators, and where does the actual search depart from it?". Nothing in this module is called by the inference
system nor by an evaluated method (test: no import from core/deep_search, deep_generation, deep_structure…).

Notions:
  Reachability            REPRESENTABLE_AND_REACHABLE / REPRESENTABLE_BUT_NOT_REACHED /
                          UNREACHABLE_WITH_CURRENT_OPERATORS / NOT_REPRESENTABLE
  generation oracle       minimal sequence of compositions that builds the true structure
  parent closure          do all the necessary intermediates exist as search objects?
  error taxonomy          one main cause per failure
"""
from __future__ import annotations

import json
import math
from typing import Dict, Iterable, List, Optional, Sequence, Set

from core.boundary_program import BNode, atoms
from core.deep_structure import COMBINATORS, COMPACT_PARAMS, algebraic_form, is_composite, n_atoms, skel_key, subtrees


class Reachability:
    REACHED = "REPRESENTABLE_AND_REACHABLE"
    NOT_REACHED = "REPRESENTABLE_BUT_NOT_REACHED"
    UNREACHABLE = "UNREACHABLE_WITH_CURRENT_OPERATORS"
    NOT_REPRESENTABLE = "NOT_REPRESENTABLE"
    ALL = (REACHED, NOT_REACHED, UNREACHABLE, NOT_REPRESENTABLE)


ERROR_TAXONOMY = ("GRAMMAR_UNREPRESENTABLE", "OPERATORS_INSUFFICIENT", "TRUE_PARENT_NOT_GENERATED", "PRUNED_TOO_EARLY", "BUDGET_EXHAUSTED", "CANONICALIZATION_COLLAPSE",
                  "FRAGMENT_MISSING", "TYPE_MISMATCH", "FIT_FAILURE", "RANKING_FAILURE", "OBSERVATIONAL_NON_IDENTIFIABILITY")
RECOVERED = "RECOVERED"

# Generation operators of each strategy (read from the code, verified by the coverage tests).
#   extend_atom    : parent ∘ atom            (chains: every internal node has at least one atom child)
#   anchor_atom    : atom ∘ (rest)            (descent: same class of trees as extend_atom)
#   extend_fragment: parent ∘ library fragment
#   compose_pair   : composition of two substructures already built (balanced trees)
#   synthesize     : boolean synthesis over signatures (any tree shape over the candidate atoms)
OPERATORS = {"baseline": ("extend_atom",), "flat": ("extend_atom",), "bottom_up": ("extend_atom", "compose_pair"), "top_down": ("anchor_atom",),
             "bidirectional": ("extend_atom", "anchor_atom", "meet_existing"), "best_first": ("extend_atom", "compose_pair"), "factorized": ("synthesize",)}


def grammar_atom_keys(allowed=None): return {skel_key(a) for a in (atoms(allowed) if allowed is not None else atoms())}


def leaves(n: BNode):
    if is_composite(n): return [l for c in n.children for l in leaves(c)]
    return [n]


def representable(truth: Optional[BNode], max_depth=6, allowed=None):
    """Is the structure expressible in the grammar: known atoms, known combinators, depth and compact
    budget respected? Returns (bool, reason)."""
    if truth is None: return False, "aucun programme de la grammaire (monde hors classe)"
    G = grammar_atom_keys(allowed); miss = [skel_key(l) for l in leaves(truth) if skel_key(l) not in G]
    if miss: return False, f"atome absent de la grammaire : {miss[0][:60]}"
    if n_atoms(truth) > max_depth: return False, f"profondeur {n_atoms(truth)} > limite {max_depth}"
    if truth.n_params() > COMPACT_PARAMS: return False, "budget compact dépassé"
    return True, ""


def flatten(n: BNode):
    """N-ary tree up to associativity: ('and'|'or'|'xor', [children]) | ('atom', key)."""
    if not is_composite(n): return ("atom", skel_key(n))
    items = []
    def flat(z):
        if z.kind == n.kind:
            for c in z.children: flat(c)
        else: items.append(flatten(z))
    flat(n); return (n.kind, items)


def chain_reachable(truth: BNode):
    """Reachable by operators that add only ONE ATOM at a time: there must exist a re-association into a
    chain, that is, each (flattened) node has at most one composite child, recursively."""
    def ok(t):
        if t[0] == "atom": return True
        comp = [c for c in t[1] if c[0] != "atom"]; return len(comp) <= 1 and all(ok(c) for c in comp)
    return ok(flatten(truth))


def operator_reachable(truth: BNode, operators: Sequence[str], library_forms: Iterable[str] = ()):
    """Does AT LEAST ONE valid sequence of the given operators exist that produces the structure (without budget)?
    Returns (bool, missing operator | '')."""
    ops = set(operators); lib = set(library_forms)
    if "synthesize" in ops or "compose_pair" in ops: return True, ""
    if chain_reachable(truth): return True, ""
    if "extend_fragment" in ops or "meet_existing" in ops:
        def ok(t):                                             # one additional composite child is allowed if it is supplied (library / meeting)
            if not is_composite(t): return True
            kids = _flat_children(t); comp = [c for c in kids if is_composite(c)]; free = [c for c in comp if algebraic_form(c) not in lib]
            return (len(free) <= 1 or "meet_existing" in ops) and all(ok(c) for c in comp)
        if ok(truth): return True, ""
    return False, "compose_pair"


def _flat_children(n: BNode):
    out = []
    def flat(z):
        if z.kind == n.kind and is_composite(z):
            for c in z.children: flat(c)
        else: out.append(z)
    for c in n.children: flat(c)
    return out


def strahler(n: BNode):
    """Required width: number of partial results to hold simultaneously to build the tree (Strahler
    number). Chain → 2; balanced tree with 4 leaves → 3."""
    if not is_composite(n): return 1
    a, b = strahler(n.children[0]), strahler(n.children[1]); return max(a, b) if a != b else a + 1


def oracle_path(truth: BNode, n_atom_types=8, pool=8):
    """GENERATION ORACLE (diagnostic only): minimal sequence of operations that builds the true structure,
    in post-order. Each step gives the form produced and the number of ALTERNATIVE CHOICES that an unguided generator
    would have at that point."""
    steps = []
    def build(n):
        if not is_composite(n): steps.append({"op": "fit_atom", "form": algebraic_form(n), "size": 1, "alternatives": n_atom_types}); return
        a, b = n.children; build(a); build(b)
        if not is_composite(b) or not is_composite(a): steps.append({"op": "extend_atom", "form": algebraic_form(n), "size": n_atoms(n), "alternatives": len(COMBINATORS) * n_atom_types})
        else: steps.append({"op": "compose_pair", "form": algebraic_form(n), "size": n_atoms(n), "alternatives": len(COMBINATORS) * pool * (pool - 1) // 2})
    build(truth); comp = [s for s in steps if s["op"] != "fit_atom"]
    return {"steps": steps, "length": len(steps), "compositions": len(comp), "intermediates": [s["form"] for s in comp[:-1]], "width": strahler(truth),
            "alternatives_product_log10": sum(math.log10(s["alternatives"]) for s in steps), "needs_compose_pair": any(s["op"] == "compose_pair" for s in steps), "depth": n_atoms(truth)}


def parent_closure(truth: BNode, max_depth=6):
    """Are all the necessary intermediates (prefixes, suffixes, subtrees, partial compositions)
    valid search objects: representable, and identifiable by their canonical key without collision?"""
    subs = subtrees(truth, min_atoms=1); out = {"subtrees": len(subs), "invalid": [], "collisions": []}; seen: Dict[str, str] = {}
    for s in subs:
        ok, why = representable(s, max_depth)
        if not ok: out["invalid"].append(why)
        k = skel_key(s); f = algebraic_form(s)
        if k in seen and seen[k] != f: out["collisions"].append(k[:80])
        seen[k] = f
    flat = flatten(truth); partial = []
    def walk(t):                                                  # partial compositions of an n-ary node: any subset of children is a valid object
        if t[0] != "atom":
            partial.append(len(t[1]))
            for c in t[1]: walk(c)
    walk(flat); out["nary_nodes"] = len(partial); out["max_arity"] = max(partial, default=0); out["closed"] = not out["invalid"] and not out["collisions"]
    return out


def space_size(depth: int, n_atom_types=8):
    """Number of distinct algebraic forms at `depth` atoms (binary trees up to commutativity; upper bound because
    associativity merges further forms). T(1) = types; T(d) = 3 · Σ_{i ≤ d−i} T(i)·T(d−i) (÷2 if i = d−i)."""
    T = {1: float(n_atom_types)}
    for d in range(2, depth + 1):
        s = 0.0
        for i in range(1, d // 2 + 1):
            j = d - i; s += T[i] * T[j] if i != j else T[i] * (T[i] + 1) / 2
        T[d] = len(COMBINATORS) * s
    return T[depth]


# ================================================================ analysis of an actual search

def forms_of(programs: Iterable[BNode]) -> Set[str]: return {algebraic_form(p) for p in programs}


def parents_used(programs: Iterable[BNode]) -> Set[str]:
    """Forms that served as a PARENT: direct child of the root of at least one generated candidate."""
    out = set()
    for p in programs:
        if is_composite(p):
            for c in p.children: out.add(algebraic_form(c))
    return out


def first_divergence(truth: BNode, generated: Set[str], used_as_parent: Set[str]):
    """First point where the actual search leaves the oracle path. Walks through the true intermediates by increasing
    size. Returns {'size', 'kind'} with kind ∈ {not_generated, not_extended, none}."""
    inter = sorted({algebraic_form(s): n_atoms(s) for s in subtrees(truth, min_atoms=1)}.items(), key=lambda kv: kv[1]); full = algebraic_form(truth)
    for form, size in inter:
        if form == full: continue
        if size >= 2 and form not in generated: return {"size": size, "kind": "not_generated"}
    for form, size in inter:
        if form == full or size < 2: continue
        if form not in used_as_parent: return {"size": size, "kind": "not_extended"}
    return {"size": n_atoms(truth), "kind": "none" if full in generated else "not_generated"}


def classify(truth: Optional[BNode], strategy: str, generated_programs: Dict[str, BNode], ranked_programs: Sequence[BNode], selected: Optional[BNode], well_fitted: bool,
             budget_exhausted: bool, class_forms: Iterable[str] = (), max_depth=6, library_forms: Iterable[str] = (), top_k=5, type_rejected: Iterable[str] = ()):
    """Reachability, recalls of the true structure at each stage, and MAIN CAUSE of a failure (taxonomy)."""
    ok, why = representable(truth, max_depth)
    if not ok: return {"reachability": Reachability.NOT_REPRESENTABLE, "cause": "GRAMMAR_UNREPRESENTABLE", "detail": why, **{k: False for k in RECALLS}}
    form = algebraic_form(truth); gen = forms_of(generated_programs.values()); used = parents_used(generated_programs.values()); ranked = [algebraic_form(p) for p in ranked_programs]
    reach_op, missing = operator_reachable(truth, OPERATORS.get(strategy, ("extend_atom",)), library_forms)
    rec = {"recall_generated": form in gen, "recall_pre_pruning": form in gen, "recall_post_pruning": form in ranked, "recall_fitted": form in gen and well_fitted,
           "recall_top_k": form in ranked[:top_k], "recall_selected": selected is not None and algebraic_form(selected) == form}
    div = first_divergence(truth, gen, used); base = {**rec, "divergence_size": div["size"], "divergence_kind": div["kind"], "oracle_length": oracle_path(truth)["length"], "width": strahler(truth)}
    if rec["recall_selected"]: return {"reachability": Reachability.REACHED, "cause": RECOVERED, **base}
    if not reach_op:
        cause = "FRAGMENT_MISSING" if "extend_fragment" in OPERATORS.get(strategy, ()) else "OPERATORS_INSUFFICIENT"
        return {"reachability": Reachability.UNREACHABLE, "cause": cause, "missing_operator": missing, **base}
    if form in set(type_rejected): return {"reachability": Reachability.NOT_REACHED, "cause": "TYPE_MISMATCH", **base}
    if not rec["recall_generated"]:
        if div["kind"] == "not_extended": cause = "BUDGET_EXHAUSTED" if budget_exhausted else "PRUNED_TOO_EARLY"
        elif div["size"] < n_atoms(truth): cause = "TRUE_PARENT_NOT_GENERATED"
        else: cause = "BUDGET_EXHAUSTED" if budget_exhausted else "PRUNED_TOO_EARLY"
        return {"reachability": Reachability.NOT_REACHED, "cause": cause, **base}
    if not well_fitted: return {"reachability": Reachability.REACHED, "cause": "FIT_FAILURE", **base}
    if not rec["recall_post_pruning"]: return {"reachability": Reachability.REACHED, "cause": "CANONICALIZATION_COLLAPSE", **base}
    if form in set(class_forms): return {"reachability": Reachability.REACHED, "cause": "OBSERVATIONAL_NON_IDENTIFIABILITY", **base}
    return {"reachability": Reachability.REACHED, "cause": "RANKING_FAILURE", **base}


RECALLS = ("recall_generated", "recall_pre_pruning", "recall_post_pruning", "recall_fitted", "recall_top_k", "recall_selected")


def basis_quality(truth: BNode):
    """Basis quality: does the structure contain REDUNDANT atoms (expressible by composition of simpler
    atoms)? band = half ∧ half (parallel), ring = disc ∧ antidisc (concentric), sector = half ∧ half
    (through the apex). Depth in the minimal basis {half-plane, disc} and number of tied parameters."""
    red = {("dot", "in"): 2, ("norm", "in"): 2, ("angle", "in"): 2}; minimal = 0; redundant = 0
    for l in leaves(truth):
        x = l
        while x.kind in ("translate", "not"): x = x.children[0]
        k = red.get((x.feat, x.op), 1); minimal += k; redundant += k > 1
    return {"atoms": n_atoms(truth), "atoms_in_minimal_basis": minimal, "redundant_atoms": redundant, "params": truth.n_params(), "description_inflation": minimal / max(1, n_atoms(truth))}

"""HIERARCHICAL compositional representation (Deep Compositional Structure Discovery milestone).

    Primitive → Substructure → CompositeStructure → StructureFamily → WorldHypothesis

A program is no longer a flat candidate: it is a node of a graph of SHARED substructures (interned by
canonical skeleton). `A∘B∘C` references the already discovered `A∘B` node, with its cached fits.

Each structure exposes: children, invariants, parameters, depth, description cost, observational support,
equivalent alternatives, dependencies.

This module contains NO world truth and imports no world module.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from core.boundary_program import BNode, atoms, canonical, predict

COMBINATORS = ("and", "or", "xor")
COMPACT_PARAMS = 64                       # compact budget (parameters) — unchanged since the previous milestone
MAX_FRAGMENTS = 32                        # budget of the fragment library

# Invariants of an atom: transformations of the space that leave its function unchanged (declarative, tested).
_ATOM_INVARIANTS = {("dot", "le"): ("translation_along_boundary",), ("dot", "ge"): ("translation_along_boundary",), ("dot", "in"): ("translation_along_boundary", "reflection_across_axis"),
                    ("norm", "le"): ("rotation_about_center",), ("norm", "ge"): ("rotation_about_center",), ("norm", "in"): ("rotation_about_center",),
                    ("angle", "in"): ("scaling_about_center", "reflection_across_bisector")}


def _leaf(n: BNode):
    while n.kind in ("translate", "not", "rotate", "scale", "reflect"): n = n.children[0]
    return n


def is_composite(n: BNode): return n.kind in COMBINATORS
def n_atoms(n: BNode): return sum(n_atoms(c) for c in n.children) if is_composite(n) else 1
def skel_key(n: BNode): return json.dumps(canonical(n.skeleton()).to_dict(), sort_keys=True)


# ================================================================ algebraic equivalence (structure)

def algebraic_form(n: BNode):
    """Normal form up to rewritings: associativity and commutativity of and / or / xor (flattening + sorting).
    De Morgan's laws and absorption are NOT applied (acknowledged limitation)."""
    n = canonical(n.skeleton())
    def norm(x):
        if is_composite(x):
            items = []
            def flat(z):
                if z.kind == x.kind:
                    for c in z.children: flat(c)
                else: items.append(norm(z))
            flat(x); return [x.kind, sorted(items, key=json.dumps)]
        return ["atom", json.dumps(x.to_dict(), sort_keys=True)]
    return json.dumps(norm(n))


def structurally_equivalent(a: BNode, b: BNode): return algebraic_form(a) == algebraic_form(b)


# ================================================================ substructure graph

@dataclass
class Substructure:
    """Node of the hierarchical graph. `key` = canonical skeleton (structural identity, without parameters)."""
    key: str
    skeleton: BNode
    children: Tuple[str, ...] = ()
    combinator: str = ""                                   # "" for a primitive
    alternatives: set = field(default_factory=set)         # keys of observationally equivalent structures
    fits: Dict[str, tuple] = field(default_factory=dict)   # cache: dataset → (parameters, logL)
    uses: int = 0                                          # number of compositions that reuse it

    @property
    def depth(self): return n_atoms(self.skeleton)                       # compositional depth (number of atoms)
    @property
    def n_params(self): return self.skeleton.n_params()
    @property
    def is_primitive(self): return not self.combinator

    def description_bits(self, n_atom_types=8, library: Optional["FragmentLibrary"] = None):
        """Description cost (bits): choice of the combinator / of the atom type at each node; a fragment of the
        library is referenced by its index instead of being spelled out."""
        return description_bits(self.skeleton, n_atom_types, library)

    def invariants(self):
        """Invariants guaranteed by construction: those of a primitive; for a composite, those common to children
        with the same center are not decidable without parameters → only the logical invariants are declared."""
        if self.is_primitive: l = _leaf(self.skeleton); return set(_ATOM_INVARIANTS.get((l.feat, l.op), ()))
        return {f"commutative_{self.combinator}", f"associative_{self.combinator}"} | ({"self_inverse_xor"} if self.combinator == "xor" else set())


class Primitive(Substructure):
    pass


class CompositeStructure(Substructure):
    pass


def description_bits(skel: BNode, n_atom_types=8, library: Optional["FragmentLibrary"] = None):
    if library is not None and is_composite(skel):
        i = library.index_of(skel)
        if i is not None: return 1.0 + math.log2(max(2, len(library)))            # 1 "fragment" bit + index
    if is_composite(skel):
        return 1.0 + math.log2(len(COMBINATORS)) + sum(description_bits(c, n_atom_types, library) for c in skel.children)
    return 1.0 + math.log2(n_atom_types)


class StructureStore:
    """Interning: a substructure (canonical skeleton) exists only ONCE; every composition references its
    children. This is what makes `A∘B∘C` depend on the already discovered `A∘B`."""

    def __init__(self, allowed=None):
        self.nodes: Dict[str, Substructure] = {}; self.primitives: List[str] = []
        for a in (atoms(allowed) if allowed is not None else atoms()): self.primitives.append(self.intern(a).key)

    def intern(self, skel: BNode) -> Substructure:
        skel = canonical(skel.skeleton()); k = skel_key(skel)
        if k in self.nodes: return self.nodes[k]
        if is_composite(skel):
            ch = tuple(self.intern(c).key for c in skel.children)
            for c in ch: self.nodes[c].uses += 1
            node = CompositeStructure(k, skel, ch, skel.kind)
        else: node = Primitive(k, skel)
        self.nodes[k] = node; return node

    def compose(self, comb: str, a: Substructure, b: Substructure) -> Substructure:
        return self.intern(BNode(comb, children=(a.skeleton, b.skeleton)))

    def dependencies(self, key: str):
        """All the substructures `key` depends on (transitive closure)."""
        out = []; stack = list(self.nodes[key].children)
        while stack:
            k = stack.pop()
            if k not in out: out.append(k); stack += list(self.nodes[k].children)
        return out

    def dependents(self, key: str): return [k for k, n in self.nodes.items() if key in n.children]
    def __len__(self): return len(self.nodes)
    def composites(self): return [n for n in self.nodes.values() if not n.is_primitive]

    def stats(self):
        c = self.composites(); return {"nodes": len(self.nodes), "primitives": len(self.primitives), "composites": len(c),
                                       "reused": sum(1 for n in self.nodes.values() if n.uses > 1), "max_depth": max((n.depth for n in self.nodes.values()), default=0)}


def subtrees(skel: BNode, min_atoms=1):
    """All the subtrees (skeletons) of a structure, composites and atoms."""
    out = []
    def walk(n):
        if n_atoms(n) >= min_atoms: out.append(n)
        if is_composite(n):
            for c in n.children: walk(c)
    walk(canonical(skel.skeleton())); return out


def normalize(prog: BNode) -> BNode:
    """Orders the children of commutative combinators by skeleton key (parameters follow their node), without
    any parameter-dependent rewriting: normalize(p).skeleton() is the interned canonical skeleton."""
    if not prog.children: return prog
    ch = tuple(normalize(c) for c in prog.children)
    if is_composite(prog): ch = tuple(sorted(ch, key=skel_key))
    return BNode(prog.kind, prog.feat, prog.op, prog.params, ch)


def _eval_override(prog: BNode, X, path, value):
    """Hard predictions of the program when the substructure at path `path` is replaced by the constant `value`."""
    if path == (): return torch.full((len(X),), int(value), dtype=torch.long)
    if not is_composite(prog): return predict(prog, X)
    a = _eval_override(prog.children[0], X, path[1:] if path and path[0] == 0 else None, value)
    b = _eval_override(prog.children[1], X, path[1:] if path and path[0] == 1 else None, value)
    return a & b if prog.kind == "and" else a | b if prog.kind == "or" else a ^ b


def observational_support(prog: BNode, sub_path: Tuple[int, ...], X):
    """Observational support of a substructure INSIDE a fitted program: points where the prediction changes when the
    substructure is replaced by the constant that replaces it BEST. Empty support ⇒ the substructure is not
    constrained by these observations (it could be removed without changing anything)."""
    base = predict(prog, X); c0 = _eval_override(prog, X, tuple(sub_path), 0) != base; c1 = _eval_override(prog, X, tuple(sub_path), 1) != base
    return c0 if int(c0.sum()) <= int(c1.sum()) else c1


def composite_paths(prog: BNode):
    """Paths (child indices) to each combinator operand."""
    out = []
    def walk(n, path):
        if is_composite(n):
            for i, c in enumerate(n.children): out.append(path + (i,)); walk(c, path + (i,))
    walk(prog, ()); return out


# ================================================================ families: actual anti-unification

@dataclass(frozen=True)
class Var:
    """Structure variable (hole) of a generalized schema."""
    name: str


def _term(n: BNode):
    """Term: ('and'|'or'|'xor', t1, t2) | ('atom', skeleton key)."""
    if is_composite(n): return (n.kind, _term(n.children[0]), _term(n.children[1]))
    return ("atom", skel_key(n))


def _size(t): return 1 if isinstance(t, Var) or t[0] == "atom" else 1 + _size(t[1]) + _size(t[2])
def _n_vars(t): return 1 if isinstance(t, Var) else 0 if t[0] == "atom" else _n_vars(t[1]) + _n_vars(t[2])
def _var_names(t): return {t.name} if isinstance(t, Var) else set() if t[0] == "atom" else _var_names(t[1]) | _var_names(t[2])


def _lgg(a, b, table):
    """Least general generalization of two terms. A given pair of distinct subterms receives the SAME variable
    (this is what distinguishes anti-unification from mere masking by position). Commutative combinators
    are aligned in the order that yields the most specific schema."""
    if a == b: return a
    if not isinstance(a, Var) and not isinstance(b, Var) and a[0] == b[0] and a[0] != "atom":
        cands = []
        for (b1, b2) in ((b[1], b[2]), (b[2], b[1])):
            t = dict(table); r = (a[0], _lgg(a[1], b1, t), _lgg(a[2], b2, t)); cands.append((_n_vars(r), len(t), r, t))
        cands.sort(key=lambda c: (c[0], c[1])); _, _, r, t = cands[0]; table.clear(); table.update(t); return r
    k = (json.dumps(a, default=lambda v: v.name), json.dumps(b, default=lambda v: v.name))
    if k not in table: table[k] = Var(f"?{len(table)}")
    return table[k]


@dataclass
class StructureFamily:
    """Generalized schema `generalized_structure(parameters)`: a term with structure variables. The numerical
    parameters stay free (fitted per instance): the family is never a memorized instance."""
    pattern: object
    support: int = 0
    provenance: List[str] = field(default_factory=list)

    @property
    def variables(self): return sorted(_var_names(self.pattern))
    @property
    def size(self): return _size(self.pattern)
    @property
    def is_trivial(self): return isinstance(self.pattern, Var)

    def matches(self, skel: BNode):
        """Bindings {variable: term} if the skeleton is an instance of the schema (up to commutativity), otherwise None."""
        return _match(self.pattern, _term(canonical(skel.skeleton())), {})

    def instantiate(self, bindings: Dict[str, BNode]) -> BNode:
        def build(t):
            if isinstance(t, Var): return bindings[t.name].skeleton()
            if t[0] == "atom": return BNode.from_dict(json.loads(t[1]))
            return BNode(t[0], children=(build(t[1]), build(t[2])))
        return canonical(build(self.pattern))

    def render(self):
        def r(t):
            if isinstance(t, Var): return t.name
            if t[0] == "atom": return _atom_name(t[1])
            return f"{t[0].upper()}({r(t[1])}, {r(t[2])})"
        return r(self.pattern)


def _atom_name(key):
    l = _leaf(BNode.from_dict(json.loads(key))); return f"{l.feat}.{l.op}"


def _match(p, t, b):
    if isinstance(p, Var):
        if p.name in b: return b if b[p.name] == t else None
        nb = dict(b); nb[p.name] = t; return nb
    if p[0] != t[0]: return None
    if p[0] == "atom": return b if p[1] == t[1] else None
    for (t1, t2) in ((t[1], t[2]), (t[2], t[1])):
        r = _match(p[1], t1, b)
        if r is not None:
            r = _match(p[2], t2, r)
            if r is not None: return r
    return None


def anti_unify(skeletons: Sequence[BNode]) -> StructureFamily:
    """Anti-unification of several structures → most specific common schema (shared variables)."""
    terms = [_term(canonical(s.skeleton())) for s in skeletons]; pat = terms[0]
    for t in terms[1:]: pat = _lgg(pat, t, {})
    return StructureFamily(_rename(pat), len(skeletons))


def _rename(t):
    names = {}
    def r(x):
        if isinstance(x, Var): return Var(names.setdefault(x.name, f"?{len(names)}"))
        return x if x[0] == "atom" else (x[0], r(x[1]), r(x[2]))
    return r(t)


def generalization_quality(family: StructureFamily, positives: Sequence[BNode], negatives: Sequence[BNode]):
    """Correct generalization: the schema covers the positives; overgeneralization: it covers negatives.
    Compression: bits saved by describing the positives by (schema + bindings) rather than one by one."""
    cov = sum(family.matches(p) is not None for p in positives) / max(1, len(positives))
    over = sum(family.matches(n) is not None for n in negatives) / max(1, len(negatives))
    raw = sum(description_bits(p.skeleton()) for p in positives)
    bind = 0.0
    for p in positives:
        m = family.matches(p)
        bind += sum(description_bits(_term_to_skel(v)) for v in m.values()) if m is not None else description_bits(p.skeleton())
    schema = description_bits_term(family.pattern)
    return {"coverage": cov, "overgeneralization": over, "trivial": family.is_trivial, "compression_bits": raw - (schema + bind), "variables": len(family.variables)}


def _term_to_skel(t): return BNode.from_dict(json.loads(t[1])) if t[0] == "atom" else BNode(t[0], children=(_term_to_skel(t[1]), _term_to_skel(t[2])))


def description_bits_term(t, n_atom_types=8):
    if isinstance(t, Var): return 2.0
    if t[0] == "atom": return 1.0 + math.log2(n_atom_types)
    return 1.0 + math.log2(len(COMBINATORS)) + description_bits_term(t[1]) + description_bits_term(t[2])


# ================================================================ fragment library

@dataclass
class Fragment:
    """Reusable GENERIC composition: a skeleton (free parameters), never an instance of the world."""
    skeleton: BNode
    key: str
    frequency: int = 0                                     # number of DISTINCT solutions that contain it
    parameterizations: int = 0                             # number of distinct parameter sets observed
    sources: List[str] = field(default_factory=list)

    @property
    def n_atoms(self): return n_atoms(self.skeleton)


class FragmentLibrary:
    """Fragments promoted if they appear in several compatible solutions AND under several parameterizations
    (otherwise it would be instance memorization). Bounded size (MAX_FRAGMENTS)."""

    def __init__(self, min_support=2, min_parameterizations=2, max_fragments=MAX_FRAGMENTS):
        self.min_support, self.min_param, self.max_fragments = min_support, min_parameterizations, max_fragments; self.family_support, self.max_families = 3, 8
        self._cand: Dict[str, Fragment] = {}; self._params: Dict[str, set] = defaultdict(set); self.fragments: List[Fragment] = []
        self.families: List[StructureFamily] = []

    def observe(self, program: BNode, source: str):
        """Records the composite substructures of a solution (FITTED program) of a task `source`."""
        seen = set()
        def walk(n):
            if is_composite(n):
                k = skel_key(n)
                if k not in seen:
                    seen.add(k); f = self._cand.setdefault(k, Fragment(canonical(n.skeleton()), k))
                    if source not in f.sources: f.sources.append(source); f.frequency += 1
                    self._params[k].add(tuple(round(v, 1) for v in n.param_vector()))
                for c in n.children: walk(c)
        walk(normalize(program))                                       # no parameter-dependent rewriting

    def promote(self):
        """Promotion: support ≥ min_support, ≥ min_parameterizations parameterizations; the most frequent then the most
        compressive first; bounded. The generalized schemas (anti-unification) of fragments of the same size are
        added as families."""
        ok = []
        for k, f in self._cand.items():
            f.parameterizations = len(self._params[k])
            if f.frequency >= self.min_support and f.parameterizations >= self.min_param: ok.append(f)
        ok.sort(key=lambda f: (-f.frequency * (f.n_atoms - 1), f.key)); self.fragments = ok[: self.max_fragments]
        # Families: PAIRWISE anti-unification of fragments of the same size (anti-unifying a whole heterogeneous group
        # yields only the trivial schema). We keep the schemas with ONE variable supported by at least `family_support`
        # fragments, the most supported first.
        pats: Dict[str, StructureFamily] = {}
        for i, f in enumerate(self.fragments):
            for g2 in self.fragments[i + 1:]:
                if f.n_atoms != g2.n_atoms: continue
                fam = anti_unify([f.skeleton, g2.skeleton])
                if fam.is_trivial or len(fam.variables) != 1: continue
                pats.setdefault(fam.render(), fam)
        for fam in pats.values(): fam.support = sum(fam.matches(f.skeleton) is not None for f in self.fragments)
        self.families = sorted((f for f in pats.values() if f.support >= self.family_support), key=lambda f: (-f.support, f.render()))[: self.max_families]
        return self.fragments

    def __len__(self): return len(self.fragments)
    def index_of(self, skel: BNode):
        k = skel_key(skel)
        for i, f in enumerate(self.fragments):
            if f.key == k: return i
        return None

    def contains(self, skel: BNode): return self.index_of(skel) is not None

    def proposals(self, max_atoms=None):
        """Skeletons to try as building blocks (decreasing frequency)."""
        return [f.skeleton for f in self.fragments if max_atoms is None or f.n_atoms <= max_atoms]

    def compression_gain(self, skeletons: Sequence[BNode]):
        """Bits saved on a set of structures by referencing the fragments, with the library cost deducted."""
        raw = sum(description_bits(s) for s in skeletons); withlib = sum(description_bits(s, library=self) for s in skeletons)
        cost = sum(description_bits(f.skeleton) for f in self.fragments)
        return {"raw_bits": raw, "with_library_bits": withlib, "library_bits": cost, "net_gain_bits": raw - withlib - cost}

    def to_dict(self): return {"fragments": [{"skeleton": f.skeleton.to_dict(), "frequency": f.frequency, "parameterizations": f.parameterizations} for f in self.fragments],
                               "families": [fam.render() for fam in self.families]}

    @staticmethod
    def from_dict(d):
        lib = FragmentLibrary()
        for x in d["fragments"]:
            s = BNode.from_dict(x["skeleton"]); lib.fragments.append(Fragment(canonical(s), skel_key(s), x["frequency"], x["parameterizations"]))
        return lib


# ================================================================ world hypotheses

NONE_OF_THE_ABOVE = "NONE_OF_THE_ABOVE"       # observations INCOMPATIBLE with the available hypotheses
UNKNOWN = "UNKNOWN"                           # INSUFFICIENT information to decide
KNOWN = "KNOWN"


@dataclass
class WorldHypothesis:
    """Hypothesis about the world: a fitted structure (or the special hypothesis "none of the known structures")."""
    key: str                                   # key of the substructure, or NONE_OF_THE_ABOVE
    program: Optional[BNode] = None
    loglik: float = float("-inf")
    evidence: float = float("-inf")
    mass: float = 0.0
    family: str = ""                           # family (algebraic form) it belongs to
    origin: str = ""                           # search step that produced it (trace)
    fit_cost: int = 0

    @property
    def is_none(self): return self.key == NONE_OF_THE_ABOVE
    @property
    def depth(self): return 0 if self.program is None else n_atoms(self.program)
    @property
    def n_params(self): return 0 if self.program is None else self.program.n_params()

    def within_compact_budget(self): return self.n_params <= COMPACT_PARAMS


def family_of(prog: BNode):
    """Family of a structure = multiset of atom types + number of combinators (coarse granularity,
    rejectable as a whole)."""
    c = Counter(_atom_name(skel_key(s)) for s in subtrees(prog) if not is_composite(s))
    return json.dumps(sorted(c.items()))

"""First-class invariants, signatures, stabilizers, orbits, equivalence classes, canonical forms.

**Galois-inspired** transformation ↔ invariant duality (Galois-inspired transformation/invariant duality):
Stabilizer(I) = set of the KNOWN transformations that preserve the invariant I, established EMPIRICALLY on
trial objects (and not by a Galois correspondence in the strict sense: no group, no field extension,
the transformations do not in general even form a group — some are not invertible).

Domains: algebraic equations, sentences, sequences, 2D geometry, mini code domain (minimal AST).

NB: `InvariantSet` is here a set of CHECKABLE invariants (signature functions); it is distinct from
`core.invariants.InvariantSet` (set of labels declared by a TransformationSpec).
"""
from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import torch

from core.algebra import Add, Divide, Equation, Multiply, Number, Power, Subtract, Symbol, simplify
from core.transformations import TransformationDomain, TransformationSpec


# ================================================================ checkable invariants

@dataclass(frozen=True)
class CheckedInvariant:
    """First-class invariant: name, domain, signature function (hashable value), optional tolerance."""
    name: str
    domain: str
    signature_fn: Callable[[Any], Any] = field(compare=False, hash=False)
    description: str = ""

    def signature(self, x):
        try: return self.signature_fn(x)
        except Exception as e: return ("INVALID", type(e).__name__)

    def holds(self, before, after): return self.signature(before) == self.signature(after)


class InvariantSet:
    """Set of checkable invariants (output equivalence, task accuracy, logical consistency, solution set,
    topology, semantic meaning, old-skill performance, unit consistency, probability normalization, …)."""

    def __init__(self, invariants: Iterable[CheckedInvariant] = ()):
        self.items: Dict[str, CheckedInvariant] = {i.name: i for i in invariants}

    def __iter__(self): return iter(self.items.values())
    def __len__(self): return len(self.items)
    def __contains__(self, name): return name in self.items
    def __getitem__(self, name): return self.items[name]
    def add(self, inv): self.items[inv.name] = inv; return self
    def subset(self, names): return InvariantSet(self.items[n] for n in names)

    def signature(self, x, names=None) -> "InvariantSignature":
        names = sorted(names or self.items)
        return InvariantSignature(tuple((n, _freeze(self.items[n].signature(x))) for n in names))

    def check(self, before, after) -> Dict[str, bool]:
        return {n: i.holds(before, after) for n, i in self.items.items()}


def _freeze(v):
    if isinstance(v, (list, tuple)): return tuple(_freeze(e) for e in v)
    if isinstance(v, set): return frozenset(_freeze(e) for e in v)
    if isinstance(v, dict): return tuple(sorted((k, _freeze(e)) for k, e in v.items()))
    if isinstance(v, float): return round(v, 6)
    if isinstance(v, torch.Tensor): return tuple(round(float(e), 6) for e in v.flatten())
    return v


@dataclass(frozen=True)
class InvariantSignature:
    """Fingerprint of a state relative to a set of invariants: two different states may share it."""
    items: Tuple[Tuple[str, Any], ...]

    def restrict(self, names): return InvariantSignature(tuple(kv for kv in self.items if kv[0] in names))
    def as_dict(self): return dict(self.items)
    def matches(self, other, names=None):
        if names is None: return self == other
        return self.restrict(set(names)) == other.restrict(set(names))


# ================================================================ domain transformations (TransformationSpec)

class DomainTransformation(TransformationSpec):
    """Named transformation of a domain (pure function x -> x'), traceable as a TransformationSpec."""

    def __init__(self, name, fn, domain="generic", invertible=False, params=None):
        object.__setattr__(self, "name", name); object.__setattr__(self, "fn", fn); object.__setattr__(self, "dname", domain)
        object.__setattr__(self, "domain", TransformationDomain.GENERIC); object.__setattr__(self, "parameters", dict(params or {}))
        object.__setattr__(self, "invertible", invertible); object.__setattr__(self, "cost", 1.0)
        from core.invariants import InvariantSet as LabelSet
        object.__setattr__(self, "preserves", LabelSet(set())); object.__setattr__(self, "changes", set())
        object.__setattr__(self, "preconditions", ()); object.__setattr__(self, "postconditions", ())

    def apply(self, x): return self.fn(x)
    def __call__(self, x): return self.fn(x)
    def __hash__(self): return hash(self.name)
    def __eq__(self, other): return isinstance(other, DomainTransformation) and other.name == self.name
    def __repr__(self): return f"T({self.name})"


# ================================================================ duality (bidirectional index), stabilizer

class DualityIndex:
    """Bidirectional transformation ↔ invariant index, established EMPIRICALLY on trial objects.
    preserves(T): invariants kept by T on ALL trials; stabilizer(I): transformations keeping I.
    `support` keeps the number of trials; `counterexamples` keeps one counterexample (checkability)."""

    def __init__(self, invariants: InvariantSet, transformations: Sequence[DomainTransformation], samples: Sequence[Any]):
        self.invariants, self.transformations = invariants, list(transformations)
        self.table: Dict[Tuple[str, str], bool] = {}; self.counterexamples: Dict[Tuple[str, str], Any] = {}
        self.support = 0
        for x in samples:
            sig0 = {i.name: i.signature(x) for i in invariants}
            for T in self.transformations:
                try: y = T(x)
                except Exception: y = None
                for i in invariants:
                    ok = y is not None and i.signature(y) == sig0[i.name]
                    key = (T.name, i.name)
                    if not ok and key not in self.counterexamples: self.counterexamples[key] = x
                    self.table[key] = self.table.get(key, True) and ok
            self.support += 1

    def preserves(self, T) -> FrozenSet[str]:
        name = T if isinstance(T, str) else T.name
        return frozenset(i for (t, i), ok in self.table.items() if t == name and ok)

    def stabilizer(self, invariant) -> "Stabilizer":
        name = invariant if isinstance(invariant, str) else invariant.name
        return Stabilizer(name, frozenset(t for (t, i), ok in self.table.items() if i == name and ok))

    def chain_preserves(self, chain: Sequence[str]) -> FrozenSet[str]:
        """Compositional prediction: a chain preserves I if every link is in Stabilizer(I)."""
        out = frozenset(self.invariants.items)
        for t in chain: out &= self.preserves(t)
        return out

    def matrix(self):
        return {T.name: sorted(self.preserves(T)) for T in self.transformations}


@dataclass(frozen=True)
class Stabilizer:
    """Stabilizer(I) — (Galois-inspired) analogue of the stabilizer: known transformations preserving I."""
    invariant: str
    transformations: FrozenSet[str]

    def __contains__(self, t): return (t if isinstance(t, str) else t.name) in self.transformations
    def __len__(self): return len(self.transformations)
    def admissible(self, chain): return all(t in self for t in chain)


# ================================================================ orbits, equivalence classes, canonical forms

class TransformationOrbit:
    """Orbit(x, G): states reachable from x under the transformations G (bounded breadth-first traversal)."""

    def __init__(self, x, G: Sequence[DomainTransformation], key=None, max_states=500, max_depth=4):
        self.key = key or _default_key
        self.root = x; self.states: Dict[Any, Any] = {self.key(x): x}; self.paths: Dict[Any, Tuple[str, ...]] = {self.key(x): ()}
        frontier = deque([(x, 0)]); self.truncated = False
        while frontier:
            y, d = frontier.popleft()
            if d >= max_depth: continue
            for T in G:
                try: z = T(y)
                except Exception: continue
                k = self.key(z)
                if k in self.states: continue
                if len(self.states) >= max_states: self.truncated = True; break
                self.states[k] = z; self.paths[k] = self.paths[self.key(y)] + (T.name,); frontier.append((z, d + 1))

    def __len__(self): return len(self.states)
    def __contains__(self, x): return self.key(x) in self.states
    def path_to(self, x): return self.paths.get(self.key(x))


def _default_key(x):
    if isinstance(x, torch.Tensor): return _freeze(x)
    if isinstance(x, (list, dict)): return json.dumps(x, sort_keys=True, default=str)
    return str(x) if not isinstance(x, (str, int, tuple)) else x


class EquivalenceClass:
    """Equivalence class relative to a subset of invariants: same restricted signature."""

    def __init__(self, invariants: InvariantSet, names=None):
        self.invariants, self.names = invariants, sorted(names or invariants.items)
        self.classes: Dict[InvariantSignature, List[Any]] = defaultdict(list)

    def signature(self, x): return self.invariants.signature(x, self.names)
    def add(self, x): sig = self.signature(x); self.classes[sig].append(x); return sig
    def equivalent(self, a, b): return self.signature(a) == self.signature(b)
    def __len__(self): return len(self.classes)

    def canonical(self, x, cost=None):
        """Canonical representative: member of the class of x minimising `cost` (then lexical order)."""
        cost = cost or (lambda y: (len(str(y)), str(y)))
        members = self.classes.get(self.signature(x)) or [x]
        return min(members, key=cost)


def canonical_form(x, G, key=None, cost=None, max_states=500, max_depth=4):
    """Canonical form = minimal-cost element of the orbit (deterministic)."""
    orb = TransformationOrbit(x, G, key, max_states, max_depth)
    cost = cost or (lambda y: (len(str(y)), str(y)))
    return min(orb.states.values(), key=cost), orb


def compression(items, canon, encode=None):
    """Memory: storing all examples vs (unique canonical forms + transformation paths).
    Returns raw bytes, compressed bytes, ratio."""
    encode = encode or (lambda y: len(str(y).encode()))
    raw = sum(encode(x) for x in items)
    classes, paths = {}, 0
    for x in items:
        c, path = canon(x)
        classes.setdefault(str(c), c); paths += 2 * len(path) + 1          # ~2 bytes / transformation identifier
    stored = sum(encode(c) for c in classes.values()) + paths
    return {"items": len(items), "classes": len(classes), "raw_bytes": raw, "compressed_bytes": stored,
            "ratio": round(raw / max(1, stored), 3)}


# ================================================================ domain: algebra (linear equations in x)

X = Symbol("x")
GRID = [Fraction(k, 2) for k in range(-24, 25)]


def eval_expr(e, x):
    if isinstance(e, Number): return Fraction(e.value) if not isinstance(e.value, float) else Fraction(e.value).limit_denominator(10 ** 6)
    if isinstance(e, Symbol): return x
    l, r = eval_expr(e.left, x), eval_expr(e.right, x)
    if isinstance(e, Add): return l + r
    if isinstance(e, Subtract): return l - r
    if isinstance(e, Multiply): return l * r
    if isinstance(e, Divide):
        if r == 0: raise ZeroDivisionError
        return l / r
    if isinstance(e, Power): return l ** int(r)
    raise TypeError(type(e))


def solution_set(eq: Equation):
    """Solution set over a grid of rationals (0 solutions, a few, or "all")."""
    sols = []
    for g in GRID:
        try:
            if eval_expr(eq.left, g) == eval_expr(eq.right, g): sols.append(g)
        except ZeroDivisionError: continue
    return "ALL" if len(sols) == len(GRID) else tuple(sols)


def canonical_equation(eq: Equation):
    """Canonical form of a LINEAR equation for the solution_set invariant: "x = s" (s exact rational);
    a non-linear or degenerate equation is returned as is."""
    try:
        f0 = eval_expr(eq.left, Fraction(0)) - eval_expr(eq.right, Fraction(0))
        f1 = eval_expr(eq.left, Fraction(1)) - eval_expr(eq.right, Fraction(1))
        f2 = eval_expr(eq.left, Fraction(2)) - eval_expr(eq.right, Fraction(2))
    except ZeroDivisionError: return eq
    if f1 == f0 or (f2 - f1) != (f1 - f0): return eq
    sol = -f0 / (f1 - f0)
    return Equation(X, Number(sol.numerator) if sol.denominator == 1 else Divide(Number(sol.numerator), Number(sol.denominator)))


def _count_nodes(e): return 1 + (_count_nodes(e.left) + _count_nodes(e.right) if hasattr(e, "left") else 0)


ALG_INVARIANTS = InvariantSet([
    CheckedInvariant("solution_set", "algebra", solution_set, "solution set"),
    CheckedInvariant("equality_form", "algebra", lambda e: isinstance(e, Equation), "remains an equation"),
    CheckedInvariant("surface_form", "algebra", str, "exact written form"),
    CheckedInvariant("size", "algebra", lambda e: _count_nodes(e), "size of the tree"),
])


def _both(op, v, name, inv=True):
    return DomainTransformation(name, lambda e: Equation(op(e.left, v), op(e.right, v)), "algebra", inv)


def algebra_transformations(values=(2, 3, -1)):
    G = [DomainTransformation("Simplify", simplify, "algebra"),
         DomainTransformation("SwapSides", lambda e: Equation(e.right, e.left), "algebra", True)]
    for v in values:
        G += [_both(Add, Number(v), f"AddBothSides({v})"), _both(Subtract, Number(v), f"SubtractBothSides({v})"),
              _both(Multiply, Number(v), f"MultiplyBothSides({v})"), _both(Divide, Number(v), f"DivideBothSides({v})")]
    # transformations OUTSIDE the stabilizer of the solution set (to be detected, not forbidden a priori)
    G += [_both(Multiply, Number(0), "MultiplyBothSides(0)", False), _both(Multiply, X, "MultiplyBothSides(x)", False),
          _both(Power, Number(2), "SquareBothSides", False),
          DomainTransformation("DropConstantLeft", lambda e: Equation(e.left.left if isinstance(e.left, (Add, Subtract)) else e.left, e.right), "algebra"),
          DomainTransformation("AddLeftOnly(1)", lambda e: Equation(Add(e.left, Number(1)), e.right), "algebra")]
    return G


def linear_equation(a, b, c):
    """a·x + b = c."""
    return Equation(Add(Multiply(Number(a), X), Number(b)), Number(c))


def algebra_samples():
    return [linear_equation(a, b, c) for a, b, c in ((2, 3, 7), (3, -1, 5), (1, 4, 0), (-2, 1, 9), (4, 0, 2), (5, 2, -3))]


def algebra_key(e): return str(e)


# ================================================================ domain: language (controlled sentences)

LEX_ENTITY = {"la balle": "ball", "le ballon": "ball", "le bloc": "block", "le cube": "cube", "la boule": "ball"}
LEX_ACTION = {"tombe": "FALL", "chute": "FALL", "dégringole": "FALL", "roule": "ROLL", "glisse": "SLIDE"}
SYN_ENTITY = {"la balle": "la boule", "la boule": "la balle"}
SYN_ACTION = {"tombe": "chute", "chute": "dégringole", "dégringole": "tombe"}


def sentence_parts(s):
    for np in sorted(LEX_ENTITY, key=len, reverse=True):
        if s.startswith(np): return np, s[len(np):].strip()
    raise ValueError(s)


LANG_INVARIANTS = InvariantSet([
    CheckedInvariant("semantic_action", "language", lambda s: LEX_ACTION[sentence_parts(s)[1]]),
    CheckedInvariant("entity_identity", "language", lambda s: LEX_ENTITY[sentence_parts(s)[0]]),
    CheckedInvariant("surface_form", "language", lambda s: s),
    CheckedInvariant("semantic_meaning", "language", lambda s: (LEX_ENTITY[sentence_parts(s)[0]], LEX_ACTION[sentence_parts(s)[1]])),
])


def _verb_syn(s):
    np, v = sentence_parts(s); return f"{np} {SYN_ACTION.get(v, v)}"


def _np_syn(s):
    np, v = sentence_parts(s); return f"{SYN_ENTITY.get(np, np)} {v}"


def _swap_entity(s):
    np, v = sentence_parts(s); return f"{'le bloc' if np != 'le bloc' else 'la balle'} {v}"


def _swap_action(s):
    np, v = sentence_parts(s); return f"{np} {'roule' if LEX_ACTION[v] != 'ROLL' else 'tombe'}"


def language_transformations():
    return [DomainTransformation("VerbSynonym", _verb_syn, "language"), DomainTransformation("NounSynonym", _np_syn, "language"),
            DomainTransformation("ReplaceEntity", _swap_entity, "language"), DomainTransformation("ReplaceAction", _swap_action, "language"),
            DomainTransformation("Identity", lambda s: s, "language", True)]


LANG_SAMPLES = ["la balle tombe", "la balle chute", "le bloc tombe", "le cube roule", "la boule glisse", "le ballon tombe"]


def canonical_sentence(s):
    """Canonical linguistic form: (entity, action) -> reference sentence."""
    e = LEX_ENTITY[sentence_parts(s)[0]]; a = LEX_ACTION[sentence_parts(s)[1]]
    np = {"ball": "la balle", "block": "le bloc", "cube": "le cube"}[e]; v = {"FALL": "tombe", "ROLL": "roule", "SLIDE": "glisse"}[a]
    return f"{np} {v}"


# ================================================================ domain: sequences

SEQ_INVARIANTS = InvariantSet([
    CheckedInvariant("multiset", "sequence", lambda s: tuple(sorted(s))),
    CheckedInvariant("length", "sequence", len),
    CheckedInvariant("cyclic_order", "sequence", lambda s: min(tuple(s[i:] + s[:i]) for i in range(len(s))) if s else ()),
    CheckedInvariant("differences", "sequence", lambda s: tuple(b - a for a, b in zip(s, s[1:]))),
    CheckedInvariant("surface_form", "sequence", tuple),
])


def sequence_transformations():
    return [DomainTransformation("Rotate", lambda s: list(s[1:]) + list(s[:1]), "sequence", True),
            DomainTransformation("Reverse", lambda s: list(reversed(s)), "sequence", True),
            DomainTransformation("Shift(+1)", lambda s: [v + 1 for v in s], "sequence", True),
            DomainTransformation("Sort", lambda s: sorted(s), "sequence"),
            DomainTransformation("DropLast", lambda s: list(s[:-1]), "sequence")]


SEQ_SAMPLES = [[3, 1, 4, 1, 5], [2, 7, 1, 8], [9, 2, 6, 5, 3, 5], [1, 2, 3, 4]]


# ================================================================ domain: 2D geometry (point clouds)

def _pairwise(P): return torch.cdist(P, P)


def _angles(P):
    v1 = P[1:] - P[0]; return tuple(round(float(torch.atan2(a[0] * b[1] - a[1] * b[0], (a * b).sum())), 6) for a, b in zip(v1, v1[1:]))


GEO_INVARIANTS = InvariantSet([
    CheckedInvariant("distances", "geometry", lambda P: tuple(round(float(v), 4) for v in _pairwise(P).flatten())),
    CheckedInvariant("angles", "geometry", _angles),
    CheckedInvariant("orientation", "geometry", lambda P: tuple(math.copysign(1, a) if abs(a) > 1e-9 else 0 for a in _angles(P))),
    CheckedInvariant("distance_ratios", "geometry", lambda P: tuple(round(float(v), 5) for v in (_pairwise(P) / _pairwise(P).max()).flatten())),
    CheckedInvariant("coordinates", "geometry", lambda P: tuple(round(float(v), 5) for v in P.flatten())),
])


def _rot(theta):
    c, s = math.cos(theta), math.sin(theta); return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)


def geometry_transformations():
    return [DomainTransformation("Rotate(30°)", lambda P: P @ _rot(math.pi / 6).T, "geometry", True),
            DomainTransformation("Translate(1,2)", lambda P: P + torch.tensor([1.0, 2.0], dtype=torch.float64), "geometry", True),
            DomainTransformation("Reflect(x)", lambda P: P * torch.tensor([1.0, -1.0], dtype=torch.float64), "geometry", True),
            DomainTransformation("Scale(2)", lambda P: 2 * P, "geometry", True),
            DomainTransformation("Shear(0.5)", lambda P: P @ torch.tensor([[1.0, 0.5], [0.0, 1.0]], dtype=torch.float64).T, "geometry", True)]


def geometry_samples(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(4, 2, generator=g, dtype=torch.float64) for _ in range(n)]


def canonical_geometry(P):
    """Normalised configuration: centred, first point on the positive x axis, orientation fixed (if possible)."""
    Q = P - P.mean(0)
    a = math.atan2(float(Q[0, 1]), float(Q[0, 0])); Q = Q @ _rot(-a).T
    if len(Q) > 2 and float(Q[1, 1]) < 0: Q = Q * torch.tensor([1.0, -1.0], dtype=torch.float64)
    return Q


# ================================================================ mini-domain: code (minimal AST)
# program = list of statements (target, expr); expr = int | name | (op, expr, expr); last var = output.

def _vars(e):
    if isinstance(e, str): return {e}
    if isinstance(e, tuple): return _vars(e[1]) | _vars(e[2])
    return set()


def run_program(prog, env=None):
    env = dict(env or {})
    def ev(e):
        if isinstance(e, bool): raise TypeError("bool")
        if isinstance(e, int): return e
        if isinstance(e, str): return env[e]
        op, a, b = e; a, b = ev(a), ev(b)
        return {"+": a + b, "-": a - b, "*": a * b}[op]
    for tgt, e in prog: env[tgt] = ev(e)
    return env[prog[-1][0]] if prog else None


def type_valid(prog):
    defined = set()
    for tgt, e in prog:
        if not _vars(e) <= defined: return False
        defined.add(tgt)
    return True


def data_dependencies(prog):
    """Def-use graph by positions (independent of names): statement i depends on the last definition of each var read."""
    last, deps = {}, []
    for i, (tgt, e) in enumerate(prog):
        deps.append(tuple(sorted(last[v] for v in _vars(e) if v in last))); last[tgt] = i
    return _dep_shape(prog, deps)


def _dep_shape(prog, deps):
    # canonical form of the DAG: multiset of (depth, number of inputs) — insensitive to the order of independent statements
    depth = []
    for i, d in enumerate(deps): depth.append(1 + max((depth[j] for j in d), default=0))
    return tuple(sorted((depth[i], len(d), _op_sig(prog[i][1])) for i, d in enumerate(deps)))


def _op_sig(e):
    if isinstance(e, tuple): return e[0]
    return "var" if isinstance(e, str) else "const"


CODE_INVARIANTS = InvariantSet([
    CheckedInvariant("program_output", "code", run_program),
    CheckedInvariant("type_validity", "code", type_valid),
    CheckedInvariant("data_dependencies", "code", data_dependencies),
    CheckedInvariant("surface_form", "code", lambda p: json.dumps(p)),
])


def _rename(prog, old, new):
    def r(e):
        if isinstance(e, str): return new if e == old else e
        if isinstance(e, tuple): return (e[0], r(e[1]), r(e[2]))
        return e
    return [(new if t == old else t, r(e)) for t, e in prog]


def rename_variable(prog):
    names = [t for t, _ in prog]
    if not names: return prog
    old = names[0]; new = old + "_r"
    return _rename(prog, old, new) if new not in names else prog


def rename_capture(prog):
    """FAULTY renaming: renames the 1st variable with the name of another one (capture)."""
    names = [t for t, _ in prog]
    return _rename(prog, names[0], names[1]) if len(set(names)) > 1 else prog


def _fold(e):
    if isinstance(e, tuple):
        op, a, b = e[0], _fold(e[1]), _fold(e[2])
        if isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool): return {"+": a + b, "-": a - b, "*": a * b}[op]
        return (op, a, b)
    return e


def constant_fold(prog): return [(t, _fold(e)) for t, e in prog]


def reorder_independent(prog):
    """Swaps the 1st pair of independent adjacent statements (no cross read nor cross write)."""
    p = list(prog)
    for i in range(len(p) - 2):          # the last statement (output) stays in place
        (t1, e1), (t2, e2) = p[i], p[i + 1]
        if t1 not in _vars(e2) and t2 not in _vars(e1) and t1 != t2:
            p[i], p[i + 1] = p[i + 1], p[i]; return p
    return p


def reorder_any(prog):
    """FAULTY reordering: swaps the first two statements without checking the dependencies."""
    p = list(prog)
    if len(p) > 2: p[0], p[1] = p[1], p[0]
    return p


def extract_expression(prog):
    """Extracts the 1st nested binary sub-expression into a temporary variable."""
    p = list(prog)
    for i, (t, e) in enumerate(p):
        if isinstance(e, tuple):
            for side in (1, 2):
                if isinstance(e[side], tuple):
                    tmp = f"tmp{i}"; new = list(e); new[side] = tmp
                    return p[:i] + [(tmp, e[side]), (t, tuple(new))] + p[i + 1:]
    return p


def code_transformations():
    return [DomainTransformation("RenameVariable", rename_variable, "code", True),
            DomainTransformation("ConstantFold", constant_fold, "code"),
            DomainTransformation("ReorderIndependentStatements", reorder_independent, "code", True),
            DomainTransformation("ExtractExpression", extract_expression, "code"),
            DomainTransformation("RenameCapture", rename_capture, "code"),
            DomainTransformation("ReorderAny", reorder_any, "code")]


CODE_SAMPLES = [
    [("a", 2), ("b", 3), ("c", ("+", "a", ("*", "b", 4)))],
    [("x", ("+", 1, 2)), ("y", 5), ("z", ("-", ("*", "x", "y"), 1))],
    [("p", 7), ("q", ("*", "p", 2)), ("r", ("+", "q", ("-", 10, 3)))],
    [("u", 1), ("v", 2), ("w", ("+", ("+", "u", "v"), ("*", 3, 3)))],
]


def canonical_program(prog):
    """Normalised program: constants folded, variables renamed v0, v1, … in definition order."""
    p = constant_fold(prog); mapping = {}
    for t, _ in p: mapping.setdefault(t, f"v{len(mapping)}")
    def r(e):
        if isinstance(e, str): return mapping.get(e, e)
        if isinstance(e, tuple): return (e[0], r(e[1]), r(e[2]))
        return e
    return [(mapping[t], r(e)) for t, e in p]


# ================================================================ domain registry

DOMAINS = {
    "algebra": (ALG_INVARIANTS, algebra_transformations, algebra_samples),
    "language": (LANG_INVARIANTS, language_transformations, lambda: list(LANG_SAMPLES)),
    "sequence": (SEQ_INVARIANTS, sequence_transformations, lambda: [list(s) for s in SEQ_SAMPLES]),
    "geometry": (GEO_INVARIANTS, geometry_transformations, geometry_samples),
    "code": (CODE_INVARIANTS, code_transformations, lambda: [list(p) for p in CODE_SAMPLES]),
}


def build_index(domain) -> DualityIndex:
    inv, tf, samples = DOMAINS[domain]
    return DualityIndex(inv, tf(), samples())

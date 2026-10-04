"""CŒUR d'induction de programmes INDÉPENDANT du domaine : rechercher → composer → ajuster → tester →
anti-unifier → promouvoir → certifier. Seules les primitives changent selon le domaine (algèbre, langage,
mini-code, séquences) ; la mécanique (énumération par complexité, objectif type MDL, anti-unification d'arbres,
cycle de vie des familles, certificat) est commune.

Un programme = arbre (op, paramètre, enfants). Les primitives d'un domaine déclarent : nom, valeurs de paramètres
candidates (l'ajustement est une énumération finie), arité, et une sémantique `apply(param, *args, x)`.
"""
from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Prim:
    name: str
    params: Tuple[Any, ...]                    # valeurs candidates (ajustement par énumération)
    arity: int                                 # 0 : feuille appliquée à x ; 1 : transforme le résultat d'un enfant ; 2 : combine
    fn: Callable = field(compare=False, hash=False, repr=False)


@dataclass(frozen=True)
class Prog:
    op: str
    param: Any = None
    children: Tuple["Prog", ...] = ()

    def size(self): return 1 + sum(c.size() for c in self.children)
    def depth(self): return 1 + max((c.depth() for c in self.children), default=0)
    def skeleton(self): return Prog(self.op, None, tuple(c.skeleton() for c in self.children))
    def __str__(self):
        p = "" if self.param is None else f"({self.param})"
        return f"{self.op}{p}" + (("[" + ", ".join(str(c) for c in self.children) + "]") if self.children else "")


def run(prog: Prog, prims: Dict[str, Prim], x):
    pr = prims[prog.op]; args = [run(c, prims, x) for c in prog.children]
    return pr.fn(prog.param, *args, x)


def complexity(p: Prog): return p.size() + 0.5 * p.depth() + 0.5 * sum(1 for _ in _walk(p) if _.param is not None)


def _walk(p):
    yield p
    for c in p.children: yield from _walk(c)


def enumerate_programs(prims: Dict[str, Prim], max_depth=3, max_size=6, commutative=()):
    """Énumération par profondeur croissante, PARAMÈTRES INCLUS (ajustement par énumération des valeurs)."""
    level = {1: [Prog(p.name, v) for p in prims.values() if p.arity == 0 for v in (p.params or (None,))]}
    seen = set(level[1])
    for d in range(2, max_depth + 1):
        new = []
        lower = [q for k in range(1, d) for q in level[k]]
        for p in prims.values():
            if p.arity == 1:
                for c in level[d - 1]:
                    for v in (p.params or (None,)):
                        q = Prog(p.name, v, (c,))
                        if q.size() <= max_size and q not in seen: seen.add(q); new.append(q)
            elif p.arity == 2:
                for a, b in itertools.product(lower, repeat=2):
                    if a.depth() != d - 1 and b.depth() != d - 1: continue
                    if p.name in commutative and str(a) > str(b): continue
                    for v in (p.params or (None,)):
                        q = Prog(p.name, v, (a, b))
                        if q.size() <= max_size and q not in seen: seen.add(q); new.append(q)
        level[d] = new
    return [q for d in sorted(level) for q in level[d]]


@dataclass
class InductionResult:
    best: Optional[Prog]
    accuracy: float
    evaluations: int
    candidates: List[Tuple[Prog, float]]
    adequate: bool


def induce(prims, examples, max_depth=3, max_size=6, lam=0.01, adequacy=1.0, commutative=(), equal=lambda a, b: a == b, macros=None):
    """Recherche ordonnée par complexité ; score = erreur + λ·L ; arrêt au premier programme cohérent (adequacy)
    si on parcourt par complexité croissante. macros : squelettes de familles induites, essayés d'abord."""
    evals = 0; scored = []
    def score(q):
        nonlocal evals
        evals += 1; ok = 0
        for x, y in examples:
            try: ok += equal(run(q, prims, x), y)
            except Exception: pass
        return ok / len(examples)
    cands = []
    if macros:
        for m in macros: cands += instantiate(m, prims)
    cands += sorted(enumerate_programs(prims, max_depth, max_size, commutative), key=complexity)
    for q in cands:
        acc = score(q); scored.append((q, acc, (1 - acc) + lam * complexity(q)))
        if acc >= adequacy: break
    scored.sort(key=lambda r: r[2]); best = scored[0] if scored else (None, 0.0, 0)
    return InductionResult(best[0], best[1], evals, [(q, a) for q, a, _ in scored[:8]], best[1] >= adequacy)


def anti_unify(progs: Sequence[Prog]):
    """Plus petit généralisant d'arbres : mêmes opérateurs → conservés ; paramètres différents → variable '?' ;
    opérateurs différents → trou '□'."""
    def lgg(a, b):
        if a.op != b.op or len(a.children) != len(b.children): return Prog("□")
        return Prog(a.op, a.param if a.param == b.param else "?", tuple(lgg(x, y) for x, y in zip(a.children, b.children)))
    t = progs[0]
    for p in progs[1:]: t = lgg(t, p)
    return t


def instantiate(template: Prog, prims):
    """Toutes les instances d'un gabarit (variables '?' = valeurs candidates de la primitive)."""
    if template.op == "□": return []
    kids = [instantiate(c, prims) for c in template.children]
    if any(len(k) == 0 for k in kids) and template.children: return []
    vals = prims[template.op].params if template.param == "?" else (template.param,)
    return [Prog(template.op, v, tuple(ch)) for v in (vals or (None,)) for ch in itertools.product(*kids)] if template.children else [Prog(template.op, v) for v in (vals or (None,))]


@dataclass
class FamilyHypothesis:
    template: Prog
    status: str = "HYPOTHESIS"
    support: int = 0
    against: int = 0

    def update(self, ok):
        self.support += ok; self.against += (not ok)
        self.status = "REFUTED" if self.against >= max(2, self.support) else "STABLE" if self.support >= 4 and self.against == 0 else "SUPPORTED" if self.support >= 2 else "HYPOTHESIS"


# ================================================================ domaines

def domain_algebra():
    """Transformations d'équations a·x + b = c représentées par (a, b, c) ; découvrir la composition qui mène
    d'une forme à une autre."""
    P = {
        "sub_both": Prim("sub_both", (1, 2, 3), 1, lambda v, e, x: (e[0], e[1] - v, e[2] - v)),
        "add_both": Prim("add_both", (1, 2, 3), 1, lambda v, e, x: (e[0], e[1] + v, e[2] + v)),
        "mul_both": Prim("mul_both", (2, 3, -1), 1, lambda v, e, x: (e[0] * v, e[1] * v, e[2] * v)),
        "id": Prim("id", (), 0, lambda v, x: x),
    }
    return P, (), lambda a, b: a == b


def domain_language():
    """Règle d'acceptation de phrases (jetons) : prédicats atomiques + ET / OU / NON."""
    ENT, VERB = ("balle", "bloc", "cube"), ("tombe", "roule", "glisse")
    P = {
        "has": Prim("has", ENT + VERB + ("ne",), 0, lambda v, x: v in x.split()),
        "and": Prim("and", (), 2, lambda v, a, b, x: a and b),
        "or": Prim("or", (), 2, lambda v, a, b, x: a or b),
        "not": Prim("not", (), 1, lambda v, a, x: not a),
    }
    return P, ("and", "or"), lambda a, b: a == b


def domain_code():
    """Condition logique d'un mini-programme (variables entières) : comparaisons + ET / OU."""
    P = {
        "gt": Prim("gt", tuple((v, t) for v in ("a", "b") for t in (0, 2, 4)), 0, lambda p, x: x[p[0]] > p[1]),
        "even": Prim("even", ("a", "b"), 0, lambda v, x: x[v] % 2 == 0),
        "and": Prim("and", (), 2, lambda v, a, b, x: a and b),
        "or": Prim("or", (), 2, lambda v, a, b, x: a or b),
        "not": Prim("not", (), 1, lambda v, a, x: not a),
    }
    return P, ("and", "or"), lambda a, b: a == b


def domain_sequences():
    P = {
        "id": Prim("id", (), 0, lambda v, x: tuple(x)),
        "rev": Prim("rev", (), 1, lambda v, s, x: tuple(reversed(s))),
        "shift": Prim("shift", (1, 2, -1), 1, lambda v, s, x: tuple(e + v for e in s)),
        "rot": Prim("rot", (1, 2), 1, lambda v, s, x: tuple(s[v:] + s[:v])),
        "sort": Prim("sort", (), 1, lambda v, s, x: tuple(sorted(s))),
    }
    return P, (), lambda a, b: a == b


DOMAINS = {"algebra": domain_algebra, "language": domain_language, "code": domain_code, "sequences": domain_sequences}


def domain_inputs(name, rng: random.Random, n):
    if name == "algebra": return [(rng.choice([1, 2, 3]), rng.randint(-5, 5), rng.randint(-5, 5)) for _ in range(n)]
    if name == "language":
        return [" ".join([rng.choice(["la", "le"]), rng.choice(["balle", "bloc", "cube"])] + (["ne"] if rng.random() < 0.3 else []) + [rng.choice(["tombe", "roule", "glisse"])]) for _ in range(n)]
    if name == "code": return [{"a": rng.randint(-3, 7), "b": rng.randint(-3, 7)} for _ in range(n)]
    return [tuple(rng.randint(0, 9) for _ in range(rng.randint(3, 6))) for _ in range(n)]


def random_target(name, rng: random.Random, depth=2):
    prims, comm, eq = DOMAINS[name]()
    progs = [p for p in enumerate_programs(prims, depth, 5, comm) if p.depth() == depth]
    return rng.choice(progs)


def cross_domain_experiment(seed=0, tasks_per_domain=6, n_examples=16):
    """Pour chaque domaine : cibles cachées de profondeur 2 ; induction depuis exemples ; test sur entrées
    nouvelles ; anti-unification des solutions d'une même structure -> famille ; famille testée sur nouvelles cibles."""
    rng = random.Random(seed); out = {}
    for name, mk in DOMAINS.items():
        prims, comm, eq = mk(); solved = gen = evals = 0; sols = []
        for _ in range(tasks_per_domain):
            tgt = random_target(name, rng)
            xs = domain_inputs(name, rng, n_examples); ex = [(x, run(tgt, prims, x)) for x in xs]
            r = induce(prims, ex, 3, 6, commutative=comm, equal=eq); evals += r.evaluations; solved += r.adequate
            test = domain_inputs(name, rng, 30)
            if r.best is not None:
                ok = sum(eq(run(r.best, prims, x), run(tgt, prims, x)) for x in test) / 30; gen += ok >= 0.97; sols.append(r.best)
        groups = defaultdict(list)
        for s in sols: groups[str(s.skeleton())].append(s)
        fams = [anti_unify(g) for g in groups.values() if len(g) >= 2]
        fam_hits = 0; fam_evals = 0; full_evals = 0
        for f in fams:
            for _ in range(2):
                tgt = rng.choice(instantiate(f, prims)); xs = domain_inputs(name, rng, n_examples); ex = [(x, run(tgt, prims, x)) for x in xs]
                rm = induce(prims, ex, 3, 6, commutative=comm, equal=eq, macros=[f]); rf = induce(prims, ex, 3, 6, commutative=comm, equal=eq)
                fam_hits += rm.adequate; fam_evals += rm.evaluations; full_evals += rf.evaluations
        out[name] = {"solved": round(solved / tasks_per_domain, 3), "generalizes": round(gen / tasks_per_domain, 3), "mean_evaluations": round(evals / tasks_per_domain, 1),
                     "families": len(fams), "family_reuse_success": round(fam_hits / max(1, 2 * len(fams)), 3),
                     "family_evals": fam_evals, "full_search_evals": full_evals, "primitives": len(prims)}
    return out

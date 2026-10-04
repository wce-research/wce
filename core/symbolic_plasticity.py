"""Même mécanique de plasticité compositionnelle dans des domaines SYMBOLIQUES (langage, algèbre, mini-code) et
un domaine NOUVEAU (chaînes de caractères) jamais vu pendant le développement du planificateur.

Une compétence = programme de primitives du domaine (DomainTransformation). Une tâche = exemples entrée→sortie
(sans identifiant). Ordre commun : réutilisation > composition de deux compétences > adaptation locale (changer
UN paramètre d'une primitive d'une compétence existante) > croissance (recherche bornée d'un nouveau programme
de primitives). Certificats : sondes canoniques + sorties attendues ; vérification avant consolidation.
"""
from __future__ import annotations

import itertools
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from core.invariant_algebra import (
    DomainTransformation, Equation, Number, X, Add, Multiply, Subtract, Divide, linear_equation, simplify,
    CODE_SAMPLES, constant_fold, rename_variable, reorder_independent, extract_expression, run_program,
)


# ================================================================ primitives paramétrées par domaine

@dataclass(frozen=True)
class Prim:
    family: str
    param: object = None

    def __str__(self): return self.family if self.param is None else f"{self.family}({self.param})"


@dataclass
class Domain:
    name: str
    families: Dict[str, Callable]             # famille -> (param) -> fonction
    params: Dict[str, Sequence]               # valeurs de paramètres possibles (None si sans paramètre)
    inputs: Callable[[random.Random, int], list]
    key: Callable = str

    def prims(self): return [Prim(f, p) for f in self.families for p in (self.params.get(f) or [None])]
    def run(self, prog, x):
        for pr in prog: x = self.families[pr.family](pr.param)(x)
        return x


def _lang():
    SYN_V = {"tombe": "chute", "chute": "tombe", "roule": "glisse", "glisse": "roule"}
    SYN_N = {"la balle": "la boule", "la boule": "la balle", "le bloc": "le cube", "le cube": "le bloc"}
    def split(s):
        for n in ("la balle", "la boule", "le bloc", "le cube"):
            if s.startswith(n): return n, s[len(n) + 1:]
        return s, ""
    fam = {"VerbSynonym": lambda p: (lambda s: (lambda n, v: f"{n} {SYN_V.get(v, v)}")(*split(s))),
           "NounSynonym": lambda p: (lambda s: (lambda n, v: f"{SYN_N.get(n, n)} {v}")(*split(s))),
           "Negate": lambda p: (lambda s: (lambda n, v: f"{n} ne {v} pas" if " pas" not in s else s)(*split(s))),
           "Prefix": lambda p: (lambda s: f"{p} {s}"),
           "Upper": lambda p: (lambda s: s.upper())}
    inputs = lambda rng, n: [f"{rng.choice(['la balle', 'la boule', 'le bloc', 'le cube'])} {rng.choice(['tombe', 'chute', 'roule', 'glisse'])}" for _ in range(n)]
    return Domain("language", fam, {"Prefix": ["hier", "demain", "souvent"]}, inputs)


def _alg():
    fam = {"AddBothSides": lambda v: (lambda e: Equation(Add(e.left, Number(v)), Add(e.right, Number(v)))),
           "SubtractBothSides": lambda v: (lambda e: Equation(Subtract(e.left, Number(v)), Subtract(e.right, Number(v)))),
           "MultiplyBothSides": lambda v: (lambda e: Equation(Multiply(e.left, Number(v)), Multiply(e.right, Number(v)))),
           "Simplify": lambda p: simplify, "SwapSides": lambda p: (lambda e: Equation(e.right, e.left))}
    inputs = lambda rng, n: [linear_equation(rng.choice([1, 2, 3, -2]), rng.randint(-5, 5), rng.randint(-5, 5)) for _ in range(n)]
    return Domain("algebra", fam, {"AddBothSides": [1, 2, 3, 5], "SubtractBothSides": [1, 2, 3, 5], "MultiplyBothSides": [2, 3, -1]}, inputs)


def _code():
    from core.perturbation_benchmark import random_program
    fam = {"ConstantFold": lambda p: constant_fold, "RenameVariable": lambda p: rename_variable,
           "ReorderIndependent": lambda p: reorder_independent, "ExtractExpression": lambda p: extract_expression,
           "AppendPrint": lambda p: (lambda prog: list(prog) + [(f"o{p}", (("+", prog[-1][0], p)))])}
    return Domain("code", fam, {"AppendPrint": [1, 2, 3]}, lambda rng, n: [random_program(rng) for _ in range(n)])


def _strings():
    """Domaine NOUVEAU (jamais vu pendant le développement du planificateur)."""
    fam = {"Reverse": lambda p: (lambda s: s[::-1]), "Upper": lambda p: (lambda s: s.upper()),
           "RotateLeft": lambda p: (lambda s: s[p:] + s[:p]), "Append": lambda p: (lambda s: s + p),
           "DropFirst": lambda p: (lambda s: s[1:]), "Double": lambda p: (lambda s: s + s)}
    inputs = lambda rng, n: ["".join(rng.choice("abcdefg") for _ in range(rng.randint(3, 6))) for _ in range(n)]
    return Domain("strings", fam, {"RotateLeft": [1, 2], "Append": ["x", "yz", "!"]}, inputs)


DOMAINS = {"language": _lang, "algebra": _alg, "code": _code, "strings": _strings}


# ================================================================ planificateur symbolique (même mécanique)

@dataclass
class SymCertificate:
    skill: str
    probes: list
    outputs: list
    def check(self, lib, dom): return all(dom.key(dom.run(lib.skills[self.skill], x)) == dom.key(o) for x, o in zip(self.probes, self.outputs))


class SymbolicSkillLibrary:
    def __init__(self, dom: Domain): self.dom = dom; self.skills: Dict[str, Tuple[Prim, ...]] = {}; self.certs: Dict[str, SymCertificate] = {}; self.log = []

    def fits(self, prog, xs, ys):
        try: return all(self.dom.key(self.dom.run(prog, x)) == self.dom.key(y) for x, y in zip(xs, ys))
        except Exception: return False

    def learn(self, xs, ys, probe_inputs, max_len=3):
        evals = 0; kind = None; prog = None
        for name, p in self.skills.items():                                         # 1 réutilisation
            evals += 1
            if self.fits(p, xs, ys): kind, prog = "reuse", p; break
        if prog is None:
            for (a, pa), (b, pb) in itertools.permutations(self.skills.items(), 2):  # 2 composition
                evals += 1
                if self.fits(pa + pb, xs, ys): kind, prog = "compose", pa + pb; break
        if prog is None:
            for name, p in self.skills.items():                                     # 3 adaptation locale (un paramètre)
                for i, pr in enumerate(p):
                    for v in self.dom.params.get(pr.family) or []:
                        if v == pr.param: continue
                        q = p[:i] + (Prim(pr.family, v),) + p[i + 1:]; evals += 1
                        if self.fits(q, xs, ys): kind, prog = "adapt", q; break
                    if prog: break
                if prog: break
        if prog is None:                                                            # 4 croissance : recherche bornée
            for L in range(1, max_len + 1):
                for q in itertools.product(self.dom.prims(), repeat=L):
                    evals += 1
                    if self.fits(q, xs, ys): kind, prog = "grow", tuple(q); break
                if prog: break
        if prog is None: self.log.append({"kind": "fail", "evals": evals}); return None
        if kind == "reuse":
            name = next(n for n, p in self.skills.items() if p == prog)
            self.log.append({"kind": kind, "evals": evals, "program": tuple(str(p) for p in prog), "retained": True}); return name
        name = f"s{len(self.skills)}"
        before = {s: c.check(self, self.dom) for s, c in self.certs.items()}
        self.skills[name] = prog
        self.certs[name] = SymCertificate(name, probe_inputs, [self.dom.run(prog, x) for x in probe_inputs])
        after = {s: c.check(self, self.dom) for s, c in self.certs.items() if s in before}
        self.log.append({"kind": kind, "evals": evals, "program": tuple(str(p) for p in prog), "retained": all(after.values())})
        return name


def symbolic_stream(dom: Domain, seed=0, n_tasks=10, max_len=2):
    rng = random.Random(seed); progs = []
    while len(progs) < n_tasks:
        u = rng.random()
        if progs and u < 0.2: progs.append(rng.choice(progs))                       # récurrence
        elif len(progs) >= 2 and u < 0.45: a, b = rng.sample(progs, 2); progs.append(a + b)     # composition
        elif progs and u < 0.6:                                                     # variante paramétrique
            p = list(rng.choice(progs)); i = rng.randrange(len(p)); vs = dom.params.get(p[i].family)
            if vs: p[i] = Prim(p[i].family, rng.choice(vs))
            progs.append(tuple(p))
        else: progs.append(tuple(rng.choice(dom.prims()) for _ in range(rng.randint(1, max_len))))
    return progs


def run_domain(name, seed=0, n_tasks=10, n_examples=4):
    dom = DOMAINS[name](); rng = random.Random(seed + 1); lib = SymbolicSkillLibrary(dom)
    probes = dom.inputs(random.Random(99), 6); solved = 0
    for prog in symbolic_stream(dom, seed, n_tasks):
        xs = dom.inputs(rng, n_examples)
        try: ys = [dom.run(prog, x) for x in xs]
        except Exception: continue
        s = lib.learn(xs, ys, probes)
        if s is not None:
            test = dom.inputs(rng, 8)
            try: solved += all(dom.key(dom.run(lib.skills[s], x)) == dom.key(dom.run(prog, x)) for x in test)
            except Exception: pass
    kinds = Counter(r["kind"] for r in lib.log)
    return {"domain": name, "tasks": len(lib.log), "generalizes": round(solved / max(1, len(lib.log)), 3), "kinds": dict(kinds),
            "retention": round(sum(r.get("retained", True) for r in lib.log) / max(1, len(lib.log)), 3),
            "evaluations": sum(r["evals"] for r in lib.log), "primitive_families": len(dom.families),
            "skills": len(lib.skills)}


def cross_domain(seed=0):
    return {d: run_domain(d, seed) for d in DOMAINS}

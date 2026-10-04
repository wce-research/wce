"""Représentation compositionnelle HIÉRARCHIQUE (jalon Deep Compositional Structure Discovery).

    Primitive → Substructure → CompositeStructure → StructureFamily → WorldHypothesis

Un programme n'est plus un candidat plat : c'est un nœud d'un graphe de sous-structures PARTAGÉES (internées par
squelette canonique). `A∘B∘C` référence le nœud `A∘B` déjà découvert, avec ses ajustements en cache.

Chaque structure expose : enfants, invariants, paramètres, profondeur, coût descriptif, support observationnel,
alternatives équivalentes, dépendances.

Ce module ne contient AUCUNE vérité de monde et n'importe aucun module de monde.
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
COMPACT_PARAMS = 64                       # budget compact (paramètres) — inchangé depuis le jalon précédent
MAX_FRAGMENTS = 32                        # budget de la bibliothèque de fragments

# Invariants d'un atome : transformations de l'espace qui laissent sa fonction inchangée (déclaratifs, testés).
_ATOM_INVARIANTS = {("dot", "le"): ("translation_along_boundary",), ("dot", "ge"): ("translation_along_boundary",), ("dot", "in"): ("translation_along_boundary", "reflection_across_axis"),
                    ("norm", "le"): ("rotation_about_center",), ("norm", "ge"): ("rotation_about_center",), ("norm", "in"): ("rotation_about_center",),
                    ("angle", "in"): ("scaling_about_center", "reflection_across_bisector")}


def _leaf(n: BNode):
    while n.kind in ("translate", "not", "rotate", "scale", "reflect"): n = n.children[0]
    return n


def is_composite(n: BNode): return n.kind in COMBINATORS
def n_atoms(n: BNode): return sum(n_atoms(c) for c in n.children) if is_composite(n) else 1
def skel_key(n: BNode): return json.dumps(canonical(n.skeleton()).to_dict(), sort_keys=True)


# ================================================================ équivalence algébrique (structure)

def algebraic_form(n: BNode):
    """Forme normale à réécritures près : associativité et commutativité de and / or / xor (aplatissement + tri).
    Les lois de De Morgan et l'absorption ne sont PAS appliquées (limite assumée)."""
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


# ================================================================ graphe de sous-structures

@dataclass
class Substructure:
    """Nœud du graphe hiérarchique. `key` = squelette canonique (identité structurelle, sans paramètres)."""
    key: str
    skeleton: BNode
    children: Tuple[str, ...] = ()
    combinator: str = ""                                   # "" pour une primitive
    alternatives: set = field(default_factory=set)         # clés de structures observationnellement équivalentes
    fits: Dict[str, tuple] = field(default_factory=dict)   # cache : jeu de données → (paramètres, logL)
    uses: int = 0                                          # nombre de compositions qui la réutilisent

    @property
    def depth(self): return n_atoms(self.skeleton)                       # profondeur compositionnelle (nb d'atomes)
    @property
    def n_params(self): return self.skeleton.n_params()
    @property
    def is_primitive(self): return not self.combinator

    def description_bits(self, n_atom_types=8, library: Optional["FragmentLibrary"] = None):
        """Coût descriptif (bits) : choix du combinateur / du type d'atome à chaque nœud ; un fragment de la
        bibliothèque se référence par son indice au lieu d'être épelé."""
        return description_bits(self.skeleton, n_atom_types, library)

    def invariants(self):
        """Invariants garantis par construction : ceux d'une primitive ; pour un composite, ceux communs aux enfants
        de même centre ne sont pas décidables sans paramètres → seuls les invariants logiques sont déclarés."""
        if self.is_primitive: l = _leaf(self.skeleton); return set(_ATOM_INVARIANTS.get((l.feat, l.op), ()))
        return {f"commutative_{self.combinator}", f"associative_{self.combinator}"} | ({"self_inverse_xor"} if self.combinator == "xor" else set())


class Primitive(Substructure):
    pass


class CompositeStructure(Substructure):
    pass


def description_bits(skel: BNode, n_atom_types=8, library: Optional["FragmentLibrary"] = None):
    if library is not None and is_composite(skel):
        i = library.index_of(skel)
        if i is not None: return 1.0 + math.log2(max(2, len(library)))            # 1 bit « fragment » + indice
    if is_composite(skel):
        return 1.0 + math.log2(len(COMBINATORS)) + sum(description_bits(c, n_atom_types, library) for c in skel.children)
    return 1.0 + math.log2(n_atom_types)


class StructureStore:
    """Internement : une sous-structure (squelette canonique) n'existe qu'UNE fois ; toute composition référence ses
    enfants. C'est ce qui rend `A∘B∘C` dépendant de `A∘B` déjà découvert."""

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
        """Toutes les sous-structures dont dépend `key` (fermeture transitive)."""
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
    """Tous les sous-arbres (squelettes) d'une structure, composites et atomes."""
    out = []
    def walk(n):
        if n_atoms(n) >= min_atoms: out.append(n)
        if is_composite(n):
            for c in n.children: walk(c)
    walk(canonical(skel.skeleton())); return out


def normalize(prog: BNode) -> BNode:
    """Ordonne les enfants des combinateurs commutatifs par clé de squelette (les paramètres suivent leur nœud), sans
    aucune réécriture dépendant des paramètres : normalize(p).skeleton() est le squelette canonique interné."""
    if not prog.children: return prog
    ch = tuple(normalize(c) for c in prog.children)
    if is_composite(prog): ch = tuple(sorted(ch, key=skel_key))
    return BNode(prog.kind, prog.feat, prog.op, prog.params, ch)


def _eval_override(prog: BNode, X, path, value):
    """Prédictions dures du programme quand la sous-structure au chemin `path` est remplacée par la constante `value`."""
    if path == (): return torch.full((len(X),), int(value), dtype=torch.long)
    if not is_composite(prog): return predict(prog, X)
    a = _eval_override(prog.children[0], X, path[1:] if path and path[0] == 0 else None, value)
    b = _eval_override(prog.children[1], X, path[1:] if path and path[0] == 1 else None, value)
    return a & b if prog.kind == "and" else a | b if prog.kind == "or" else a ^ b


def observational_support(prog: BNode, sub_path: Tuple[int, ...], X):
    """Support observationnel d'une sous-structure DANS un programme ajusté : points où la prédiction change quand la
    sous-structure est remplacée par la constante qui la remplace LE MIEUX. Support vide ⇒ la sous-structure n'est pas
    contrainte par ces observations (elle pourrait être retirée sans rien changer)."""
    base = predict(prog, X); c0 = _eval_override(prog, X, tuple(sub_path), 0) != base; c1 = _eval_override(prog, X, tuple(sub_path), 1) != base
    return c0 if int(c0.sum()) <= int(c1.sum()) else c1


def composite_paths(prog: BNode):
    """Chemins (indices d'enfants) vers chaque opérande de combinateur."""
    out = []
    def walk(n, path):
        if is_composite(n):
            for i, c in enumerate(n.children): out.append(path + (i,)); walk(c, path + (i,))
    walk(prog, ()); return out


# ================================================================ familles : anti-unification réelle

@dataclass(frozen=True)
class Var:
    """Variable de structure (trou) d'un schéma généralisé."""
    name: str


def _term(n: BNode):
    """Terme : ('and'|'or'|'xor', t1, t2) | ('atom', clé de squelette)."""
    if is_composite(n): return (n.kind, _term(n.children[0]), _term(n.children[1]))
    return ("atom", skel_key(n))


def _size(t): return 1 if isinstance(t, Var) or t[0] == "atom" else 1 + _size(t[1]) + _size(t[2])
def _n_vars(t): return 1 if isinstance(t, Var) else 0 if t[0] == "atom" else _n_vars(t[1]) + _n_vars(t[2])
def _var_names(t): return {t.name} if isinstance(t, Var) else set() if t[0] == "atom" else _var_names(t[1]) | _var_names(t[2])


def _lgg(a, b, table):
    """Plus petit généralisé de deux termes. Une même paire de sous-termes distincts reçoit la MÊME variable
    (c'est ce qui distingue l'anti-unification d'un simple masquage par position). Les combinateurs commutatifs
    sont alignés dans l'ordre qui donne le schéma le plus spécifique."""
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
    """Schéma généralisé `generalized_structure(parameters)` : un terme avec variables de structure. Les paramètres
    numériques restent libres (ajustés par instance) : la famille n'est jamais une instance mémorisée."""
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
        """Liaisons {variable: terme} si le squelette est une instance du schéma (à commutativité près), sinon None."""
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
    """Anti-unification de plusieurs structures → schéma commun le plus spécifique (variables partagées)."""
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
    """Généralisation correcte : le schéma couvre les positifs ; surgénéralisation : il couvre des négatifs.
    Compression : bits économisés en décrivant les positifs par (schéma + liaisons) plutôt qu'un par un."""
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


# ================================================================ bibliothèque de fragments

@dataclass
class Fragment:
    """Composition GÉNÉRIQUE réutilisable : un squelette (paramètres libres), jamais une instance du monde."""
    skeleton: BNode
    key: str
    frequency: int = 0                                     # nombre de solutions DISTINCTES qui le contiennent
    parameterizations: int = 0                             # nombre de jeux de paramètres distincts observés
    sources: List[str] = field(default_factory=list)

    @property
    def n_atoms(self): return n_atoms(self.skeleton)


class FragmentLibrary:
    """Fragments promus s'ils apparaissent dans plusieurs solutions compatibles ET sous plusieurs paramétrages
    (sinon ce serait de la mémorisation d'instance). Taille bornée (MAX_FRAGMENTS)."""

    def __init__(self, min_support=2, min_parameterizations=2, max_fragments=MAX_FRAGMENTS):
        self.min_support, self.min_param, self.max_fragments = min_support, min_parameterizations, max_fragments
        self._cand: Dict[str, Fragment] = {}; self._params: Dict[str, set] = defaultdict(set); self.fragments: List[Fragment] = []
        self.families: List[StructureFamily] = []

    def observe(self, program: BNode, source: str):
        """Enregistre les sous-structures composites d'une solution (programme AJUSTÉ) d'une tâche `source`."""
        seen = set()
        def walk(n):
            if is_composite(n):
                k = skel_key(n)
                if k not in seen:
                    seen.add(k); f = self._cand.setdefault(k, Fragment(canonical(n.skeleton()), k))
                    if source not in f.sources: f.sources.append(source); f.frequency += 1
                    self._params[k].add(tuple(round(v, 1) for v in n.param_vector()))
                for c in n.children: walk(c)
        walk(normalize(program))                                       # pas de réécriture dépendant des paramètres

    def promote(self):
        """Promotion : support ≥ min_support, ≥ min_parameterizations paramétrages ; les plus fréquents puis les plus
        compressifs d'abord ; bornée. Les schémas généralisés (anti-unification) des fragments de même taille sont
        ajoutés comme familles."""
        ok = []
        for k, f in self._cand.items():
            f.parameterizations = len(self._params[k])
            if f.frequency >= self.min_support and f.parameterizations >= self.min_param: ok.append(f)
        ok.sort(key=lambda f: (-f.frequency * (f.n_atoms - 1), f.key)); self.fragments = ok[: self.max_fragments]
        by_size = defaultdict(list)
        for f in self.fragments: by_size[f.n_atoms].append(f.skeleton)
        self.families = [fam for fam in (anti_unify(v) for v in by_size.values() if len(v) >= 2) if not fam.is_trivial]
        return self.fragments

    def __len__(self): return len(self.fragments)
    def index_of(self, skel: BNode):
        k = skel_key(skel)
        for i, f in enumerate(self.fragments):
            if f.key == k: return i
        return None

    def contains(self, skel: BNode): return self.index_of(skel) is not None

    def proposals(self, max_atoms=None):
        """Squelettes à essayer comme blocs de construction (fréquence décroissante)."""
        return [f.skeleton for f in self.fragments if max_atoms is None or f.n_atoms <= max_atoms]

    def compression_gain(self, skeletons: Sequence[BNode]):
        """Bits économisés sur un ensemble de structures en référençant les fragments, coût de la bibliothèque déduit."""
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


# ================================================================ hypothèses de monde

NONE_OF_THE_ABOVE = "NONE_OF_THE_ABOVE"       # observations INCOMPATIBLES avec les hypothèses disponibles
UNKNOWN = "UNKNOWN"                           # information INSUFFISANTE pour trancher
KNOWN = "KNOWN"


@dataclass
class WorldHypothesis:
    """Hypothèse sur le monde : une structure ajustée (ou l'hypothèse spéciale « aucune des structures connues »)."""
    key: str                                   # clé de la sous-structure, ou NONE_OF_THE_ABOVE
    program: Optional[BNode] = None
    loglik: float = float("-inf")
    evidence: float = float("-inf")
    mass: float = 0.0
    family: str = ""                           # famille (forme algébrique) à laquelle elle appartient
    origin: str = ""                           # étape de la recherche qui l'a produite (trace)
    fit_cost: int = 0

    @property
    def is_none(self): return self.key == NONE_OF_THE_ABOVE
    @property
    def depth(self): return 0 if self.program is None else n_atoms(self.program)
    @property
    def n_params(self): return 0 if self.program is None else self.program.n_params()

    def within_compact_budget(self): return self.n_params <= COMPACT_PARAMS


def family_of(prog: BNode):
    """Famille d'une structure = multiensemble des types d'atomes + nombre de combinateurs (granularité grossière,
    rejetable en bloc)."""
    c = Counter(_atom_name(skel_key(s)) for s in subtrees(prog) if not is_composite(s))
    return json.dumps(sorted(c.items()))

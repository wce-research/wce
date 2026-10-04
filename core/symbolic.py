"""Séquences symboliques : alphabet commun, primitives WCE, familles, experts neuronaux, induction.

Toutes les familles partagent le MÊME alphabet et la MÊME distribution d'entrée :
la tâche ne peut pas être lue dans x. Elle ne se révèle que par le comportement (x -> y),
par un Goal / une commande, ou par l'historique récent.
"""
from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from core.invariants import Invariant, InvariantSet
from core.model_state import (
    INPUT, AdaptWeights, AddModule, BindRoute, ClearFastWeights, CreateConnection, LoadFastDelta, ModelState,
)
from core.transformations import CompositeTransformation, TransformationDomain, TransformationSpec

# Réseaux minuscules : un seul thread CPU est ~3x plus rapide que le parallélisme intra-op.
torch.set_num_threads(1)

ALPHABET = "abcdefgh"
V = len(ALPHABET)
L = 5
SEQ_INPUT = L * V


# ================================================================ représentation

def encode(text: str) -> torch.Tensor:
    return torch.tensor([ALPHABET.index(c) for c in text], dtype=torch.long)


def decode(seq) -> str:
    return "".join(ALPHABET[int(i)] for i in seq)


def one_hot(x: torch.Tensor) -> torch.Tensor:
    """(n, L) entiers -> (n, L*V) : l'entrée est la séquence elle-même, rien d'autre."""
    return torch.nn.functional.one_hot(x, V).float().reshape(len(x), -1)


def random_sequences(n, gen) -> torch.Tensor:
    return torch.randint(0, V, (n, L), generator=gen)


# ================================================================ primitives WCE

PERMUTATION = InvariantSet({Invariant.SYMBOL_MULTISET, Invariant.SEQUENCE_LENGTH})
LENGTH_ONLY = InvariantSet({Invariant.SEQUENCE_LENGTH})


@dataclass(frozen=True)
class SequenceTransformation(TransformationSpec):
    """Primitive symbolique exacte, opérant sur un lot (n, L)."""
    name: str = "sequence"
    domain: TransformationDomain = TransformationDomain.SEQUENCE
    kind: str = "identity"
    k: int = 0
    invertible: bool = True
    cost: float = 1.0

    def apply(self, x, *args, **kwargs):
        if self.kind == "identity": return x.clone()
        if self.kind == "reverse": return x.flip(1)
        if self.kind == "rotl": return torch.roll(x, -self.k, dims=1)
        if self.kind == "rotr": return torch.roll(x, self.k, dims=1)
        if self.kind == "sub": return (x + self.k) % V
        if self.kind == "swap": return x[:, [1, 0, 3, 2, 4]]
        if self.kind == "sort": return x.sort(dim=1).values
        raise ValueError(self.kind)

    def get_inverse(self):
        inverse = {"identity": ("identity", 0), "reverse": ("reverse", 0), "rotl": ("rotr", self.k),
                   "rotr": ("rotl", self.k), "sub": ("sub", (-self.k) % V), "swap": ("swap", 0)}
        if self.kind not in inverse: return None
        kind, k = inverse[self.kind]
        return primitive(kind, k)


def primitive(kind, k=0):
    name = kind if kind in ("identity", "reverse", "swap", "sort") else f"{kind}{k}"
    return SequenceTransformation(name=name, kind=kind, k=k,
                                  preserves=LENGTH_ONLY if kind == "sub" else PERMUTATION,
                                  invertible=kind != "sort")


PRIMITIVES = {p.name: p for p in (
    primitive("identity"), primitive("reverse"), primitive("rotl", 1), primitive("rotl", 2),
    primitive("rotr", 1), primitive("rotr", 2), primitive("sub", 1), primitive("sub", 2),
    primitive("sub", 3), primitive("sub", 7), primitive("swap"), primitive("sort"))}


def compose(*specs) -> TransformationSpec:
    """[T1, T2] : T1 d'abord. Le nom suit l'ordre d'application : 'T1>T2'."""
    if len(specs) == 1: return specs[0]
    return CompositeTransformation(name=">".join(s.name for s in specs), transformations=tuple(specs),
                                   domain=TransformationDomain.SEQUENCE)


def components(spec) -> Tuple[TransformationSpec, ...]:
    return tuple(spec.transformations) if isinstance(spec, CompositeTransformation) else (spec,)


# ================================================================ experts neuronaux comme TransformationSpec

@dataclass(frozen=True)
class NeuralSequenceTransformation(TransformationSpec):
    """Un expert neuronal exposé à WCE comme une transformation composable."""
    name: str = "neural"
    domain: TransformationDomain = TransformationDomain.SEQUENCE
    module: Any = field(default=None, compare=False, repr=False)
    fast: Any = field(default=None, compare=False, repr=False)
    state: Any = field(default=None, compare=False, repr=False)     # résolution par nom : suit les adoptions
    module_name: str = ""
    invertible: bool = False
    cost: float = 2.0

    @property
    def resolved(self):
        return self.state.modules[self.module_name] if self.state is not None else self.module

    def apply(self, x, *args, **kwargs):
        with torch.no_grad():
            logits = self.resolved.forward(one_hot(x), self.fast)
        return logits.reshape(len(x), L, V).argmax(-1)


# ================================================================ familles

@dataclass
class Family:
    name: str
    kind: str                     # primitive | composition | neural | hybrid | variant
    fn: Callable[[torch.Tensor], torch.Tensor] = field(repr=False)
    parts: Tuple[str, ...] = ()
    token: Optional[str] = None   # commande symbolique éventuelle (REV, ROT...)

    def apply(self, x): return self.fn(x)


def _perm(order):
    idx = list(order)
    return lambda x: x[:, idx]


def _symmap(table):
    t = torch.tensor(table)
    return lambda x: t[x]


def _chain(*fns):
    def f(x):
        for g in fns: x = g(x)
        return x
    return f


PERM_A = (2, 0, 4, 1, 3)
PERM_B = (3, 4, 0, 2, 1)
MAP_A = (3, 6, 0, 7, 1, 4, 2, 5)
MAP_B = (5, 2, 7, 4, 0, 1, 6, 3)
MAP_A2 = (6, 3, 0, 7, 1, 4, 2, 5)      # MAP_A avec deux images échangées


def build_families() -> Dict[str, Family]:
    P = PRIMITIVES
    fam = {}
    for name, token in (("identity", "COPY"), ("reverse", "REV"), ("rotl1", "ROT"), ("rotr1", "ROTR"),
                        ("rotl2", "ROT2"), ("sub1", "SUB"), ("sub3", "SUB3"), ("swap", "SWAP"), ("sort", "SORT")):
        fam[name] = Family(name, "primitive", P[name].apply, (name,), token)
    for a, b in (("reverse", "rotl1"), ("sub1", "reverse"), ("swap", "rotl1"), ("rotl2", "sub2")):
        fam[f"{a}>{b}"] = Family(f"{a}>{b}", "composition", _chain(P[a].apply, P[b].apply), (a, b))
    fam["perm_A"] = Family("perm_A", "neural", _perm(PERM_A), ("perm_A",))
    fam["perm_B"] = Family("perm_B", "neural", _perm(PERM_B), ("perm_B",))
    fam["map_A"] = Family("map_A", "neural", _symmap(MAP_A), ("map_A",))
    fam["map_B"] = Family("map_B", "neural", _symmap(MAP_B), ("map_B",))
    fam["perm_A>reverse"] = Family("perm_A>reverse", "hybrid", _chain(_perm(PERM_A), P["reverse"].apply), ("perm_A", "reverse"))
    fam["map_A>rotl1"] = Family("map_A>rotl1", "hybrid", _chain(_symmap(MAP_A), P["rotl1"].apply), ("map_A", "rotl1"))
    fam["sub1>perm_B"] = Family("sub1>perm_B", "hybrid", _chain(P["sub1"].apply, _perm(PERM_B)), ("sub1", "perm_B"))
    fam["map_A2"] = Family("map_A2", "variant", _symmap(MAP_A2), ("map_A2",))
    return fam


FAMILIES = build_families()


def sample_family(family: Family, n, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = random_sequences(n, gen)
    return x, family.apply(x)


# ================================================================ induction structurelle WCE

@dataclass
class ProgramResult:
    program: Optional[TransformationSpec]
    explored: int
    consistent: List[str] = field(default_factory=list)

    @property
    def found(self): return self.program is not None


def _preserves_multiset(spec):
    return all(Invariant.SYMBOL_MULTISET in s.preserves.invariants for s in components(spec))


def search_programs(xs, ys, library: Sequence[TransformationSpec], max_depth=2, first=True, min_fraction=1.0):
    """BFS sur les compositions de la bibliothèque (primitives + experts + skills).

    Les invariants WCE élaguent : si le multiset de symboles change, un programme composé
    uniquement de transformations préservant ce multiset est rejeté sans exécution.
    """
    multiset_changed = bool((xs.sort(1).values != ys.sort(1).values).any())
    explored, consistent, best = 0, [], None
    for depth in range(1, max_depth + 1):
        for combo in itertools.product(library, repeat=depth):
            if depth > 1 and any(c.name == "identity" for c in combo): continue
            if multiset_changed and all(Invariant.SYMBOL_MULTISET in c.preserves.invariants for c in combo): continue
            explored += 1
            out = xs
            for c in combo: out = c.apply(out)
            fraction = float((out == ys).all(1).float().mean())
            if fraction >= min_fraction:
                program = compose(*combo)
                consistent.append(program.name)
                if best is None: best = program
                if first: return ProgramResult(best, explored, consistent)
    return ProgramResult(best, explored, consistent)


def ambiguity_rate(families: Dict[str, Family], n=512, seed=0):
    """Part des paires (x, y) d'une famille que produit aussi au moins une autre famille."""
    gen = torch.Generator().manual_seed(seed)
    x = random_sequences(n, gen)
    outs = {name: f.apply(x) for name, f in families.items()}
    ambiguous, total = 0, 0
    for name, y in outs.items():
        others = [(outs[o] == y).all(1) for o in outs if o != name]
        ambiguous += int(torch.stack(others).any(0).sum()); total += n
    same_input = 1.0  # à x seul, toutes les familles sont possibles
    return {"single_pair": ambiguous / total, "input_only": same_input}


# ================================================================ modèle de séquences dynamique

def sequence_accuracy(pred, y): return float((pred == y).all(1).float().mean())


def train_expert(state: ModelState, name, x, y, *, hidden=64, steps=300, lr=0.01, seed=0, config=None):
    """Nouvel expert neuronal, créé et entraîné uniquement via des specs tracées."""
    specs = [AddModule(module=name, input_dim=SEQ_INPUT, output_dim=SEQ_INPUT, hidden=hidden, activation=False,
                       specialization="sequence", seed=seed),
             CreateConnection(source=INPUT, target=name),
             AdaptWeights(modules=(name,), fast=False, output=name, positions=L, inputs=one_hot(x), targets=y,
                          steps=steps, lr=lr, seed=seed, local=False)]
    if config: specs.append(BindRoute(configuration=config, module=name))
    for spec in specs: state.apply(spec)
    return NeuralSequenceTransformation(name=name, state=state, module_name=name)


def adapt_expert_fast(state: ModelState, module, x, y, *, steps=200, lr=0.03, seed=0, keys=("w2", "b2"), base_fast=None):
    """Adaptation rapide locale d'un expert existant ; renvoie le delta (le module lent est intact)."""
    state.apply(LoadFastDelta(module=module, delta=dict(base_fast or {})))
    state.apply(AdaptWeights(name="few_shot_fast", modules=(module,), fast=True, keys=keys, output=module,
                             positions=L, inputs=one_hot(x), targets=y, steps=steps, lr=lr, seed=seed, local=False))
    delta = {k: v.clone() for k, v in state.fast_parameters[module].items()}
    state.apply(ClearFastWeights(modules=(module,)))
    return delta


# ================================================================ raffinement structurel few-shot

@dataclass(frozen=True)
class SymbolTableRefinement(TransformationSpec):
    """Skill existante suivie d'une table de correction de symboles induite de quelques exemples."""
    name: str = "refinement"
    domain: TransformationDomain = TransformationDomain.SEQUENCE
    base: Any = field(default=None, compare=False, repr=False)
    table: Tuple[int, ...] = tuple(range(V))
    invertible: bool = False
    cost: float = 1.5

    def apply(self, x, *args, **kwargs):
        return torch.tensor(self.table)[self.base.apply(x)]


def induce_refinement(xs, ys, base, min_agreement=0.6, min_pairs=4, min_consistency=0.9):
    """Raffinement d'une Skill proche (pas d'apprentissage ex nihilo) :
    la base doit déjà être correcte sur >= min_agreement des positions ; la table de correction
    est un vote majoritaire, acceptée si elle explique >= min_consistency des positions
    (tolère une base neuronale imparfaite) et reste une bijection."""
    if len(xs) < min_pairs: return None
    guess = base.apply(xs)
    if float((guess == ys).float().mean()) < min_agreement: return None
    votes = {}
    for g, y in zip(guess.reshape(-1).tolist(), ys.reshape(-1).tolist()):
        votes.setdefault(g, Counter())[y] += 1
    table = [None] * V
    for g, counter in votes.items(): table[g] = counter.most_common(1)[0][0]
    # symboles non observés : identité si libre, sinon cibles restantes (la table reste une bijection)
    free = [t for t in range(V) if t not in table]
    for g in range(V):
        if table[g] is None and g in free: table[g] = g; free.remove(g)
    for g in range(V):
        if table[g] is None: table[g] = free.pop(0)
    if table == list(range(V)) or len(set(table)) != V: return None
    if float((torch.tensor(table)[guess] == ys).float().mean()) < min_consistency: return None
    return SymbolTableRefinement(name=f"{base.name}+table", base=base, table=tuple(table))

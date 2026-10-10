"""Compact WCE Language Model — first integrated prototype.

TEXT -> characters -> small neural core (concept candidates)
      -> structured parsing (clauses, entities, relations, 'puis')
      -> Goals (WCE planner) / programs (CompositeTransformation)
      -> WCE structural verification -> risk (neural, structural, geometric, external)
      -> policy ACCEPT / VERIFY / REROUTE / ABSTAIN / REJECT / ASK -> execution / answer.

The corpus is artificial and controlled: it does not claim to cover real French.
"""
from __future__ import annotations

import copy
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from core.invariants import Invariant, InvariantSet
from core.plasticity import ConfigurationAmplitude
from core.reasoning import Equals, GreaterThanOrEqual, Goal, PropertyRef, TruthValue
from core.symbolic import ALPHABET as SEQ_ALPHABET, PRIMITIVES, compose, decode as seq_decode, encode as seq_encode
from core.transform_graph import TransformGraph, TransformNode
from core.transformation import Change, RelationChange, Transformation
from core.transformations import DeclarativeTransformationSpec, TransformationDomain, TransformationSpec
from core.world import Entity, World

torch.set_num_threads(1)

# ================================================================ A. characters

CHARSET = " 'abcdefghijklmnopqrstuvwxyzàâäçéèêëîïôöûùüÿœ-.,?!0123456789"
PAD, UNK = 0, 1
MAX_WORD = 16


def char_ids(word: str) -> torch.Tensor:
    ids = [CHARSET.index(c) + 2 if c in CHARSET else UNK for c in word[:MAX_WORD]]
    return torch.tensor(ids + [PAD] * (MAX_WORD - len(ids)), dtype=torch.long)


def encode_words(words: Sequence[str]) -> torch.Tensor:
    return torch.stack([char_ids(w) for w in words])


def tokenize_fr(text: str) -> List[str]:
    t = text.lower().replace("’", "'").strip()
    t = re.sub(r"([.,!?])(\s|$)", r" \1 ", t)
    out = []
    for w in t.split():
        if w in (".", ","): continue
        m = re.match(r"^(l|d|j|qu)'(.+)$", w)
        if m: out += [m.group(1) + "'", m.group(2)]
        else: out.append(w)
    return out


# ================================================================ controlled corpus

CONCEPTS = ["FALL", "RISE", "PLACE", "REVERSE", "ROTL", "ROTR", "SHIFT", "SORT", "SWAP", "OTHER"]
PHYSICAL = {"FALL", "RISE", "PLACE"}
SEQUENCE_CONCEPTS = {"REVERSE": "reverse", "ROTL": "rotl1", "ROTR": "rotr1", "SHIFT": "sub1", "SORT": "sort", "SWAP": "swap"}
PRIMITIVE_TO_CONCEPT = {v: k for k, v in SEQUENCE_CONCEPTS.items()}

# (stem, class); classes: 'er' (tombe, tomber, tombait, tombé, tombent, tombera), 're' (descend, descendre...)
LEMMAS = {
    "FALL": [("tomb", "er"), ("chut", "er"), ("descend", "re"), ("dégringol", "er")],
    "RISE": [("mont", "er"), ("grimp", "er"), ("s'élev", "elev")],
    "PLACE": [("plac", "er"), ("pos", "er"), ("dépos", "er"), ("install", "er")],
    "REVERSE": [("retourn", "er"), ("invers", "er"), ("renvers", "er")],
    "ROTL": [("tourn", "er")],
    "ROTR": [("pivot", "er")],
    "SHIFT": [("décal", "er"), ("avanc", "er"), ("incrément", "er")],
    "SORT": [("tri", "er"), ("ordonn", "er")],
    "SWAP": [("échang", "er"), ("permut", "er")],
}


def forms_of(stem, cls):
    if cls == "er": return [stem + "e", stem + "er", stem + "ait", stem + "é", stem + "ent", stem + "era"]
    if cls == "re": return [stem, stem + "re", stem + "ait", stem + "u", stem + "ent", stem + "ra"]
    if cls == "elev": return ["s'élève", "s'élever", "s'élevait", "s'élevé", "s'élèvent", "s'élèvera"]
    raise ValueError(cls)


TRUE_CONCEPT: Dict[str, Any] = {}
for _concept, _lemmas in LEMMAS.items():
    for _stem, _cls in _lemmas:
        for _f in forms_of(_stem, _cls): TRUE_CONCEPT[_f] = _concept
TRUE_CONCEPT["tourne"] = {"ROTL", "ROTR"}       # real lexical ambiguity in the corpus
TRUE_CONCEPT["range"] = {"SORT", "PLACE"}       # ambiguity resolved by the domain of the arguments

ENTITIES = {"balle": ("sol", {"couleur": "rouge"}), "bloc": ("table", {"couleur": "bleu"}),
            "cube": ("étagère", {"couleur": "vert"}), "boîte": ("sol", {"couleur": "jaune"})}
SUPPORTS = {"sol": 0.0, "table": 1.0, "étagère": 2.0, "support": 1.5}
DETERMINERS = {"la", "le", "les", "l'", "un", "une", "des"}
PREPOSITIONS = {"sur", "du", "de", "d'", "depuis", "à", "au", "vers", "en"}
UNITS = {"mètres", "mètre", "m"}
FUNCTION = {"puis", "et", "est", "où", "quelle", "quel", "hauteur", "?", "!", "-"}
NOUNS = set(ENTITIES) | set(SUPPORTS) | {"girafe", "lampe"}
OTHER_WORDS = sorted(NOUNS | DETERMINERS | PREPOSITIONS | UNITS | {"puis", "et", "est", "où", "quelle", "hauteur",
                                                                    "rouge", "bleu", "bois", "verre"})
NONSENSE = ["zorphe", "blimpe", "trakule", "vornit", "glissouille", "frumax"]
TRAPS = ["montagne", "tombola", "triangle", "tournesol", "décalcomanie", "posture"]   # misleading prefixes


def is_number(tok):
    return re.fullmatch(r"-?\d+(\.\d+)?", tok) is not None


def is_sequence(tok):
    return len(tok) == 5 and tok.isalpha() and tok.isascii()


# ================================================================ A. character neural core

SIZES = {"tiny": (8, 16), "small": (16, 32), "medium": (32, 96)}


class CharCore(torch.nn.Module):
    """Character embeddings -> 2 convolutions -> max-pool -> head -> concepts.
    The fast weights (fast) act on the output layer; every training run is traced."""

    def __init__(self, size="small", seed=0, concepts=CONCEPTS):
        super().__init__()
        torch.manual_seed(seed)
        e, h = SIZES[size]
        self.size, self.concepts = size, list(concepts)
        self.emb = torch.nn.Embedding(len(CHARSET) + 2, e, padding_idx=PAD)
        self.conv1 = torch.nn.Conv1d(e, h, 3, padding=1)
        self.conv2 = torch.nn.Conv1d(h, h, 3, padding=1)
        self.head = torch.nn.Linear(h, h)
        self.out = torch.nn.Linear(h, len(self.concepts))
        self.fast: Dict[str, torch.Tensor] = {}
        self.graph = TransformGraph(TransformNode("L0"))
        self._snapshots: Dict[str, Dict[str, torch.Tensor]] = {}

    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())

    def layers(self, ids, noise=0.0, gen=None):
        """Per-layer activations (for the TrajectoryMonitor): emb, conv1, conv2, head, logits."""
        e = self.emb(ids)
        if noise: e = e + noise * torch.randn(e.shape, generator=gen)
        mask = (ids != PAD).float().unsqueeze(-1)
        e_pool = (e * mask).sum(1) / mask.sum(1).clamp_min(1)
        neg = (1 - mask.transpose(1, 2)) * 1e4
        h1 = torch.relu(self.conv1(e.transpose(1, 2)))
        h2 = torch.relu(self.conv2(h1))
        p1, p2 = (h1 - neg).max(2).values, (h2 - neg).max(2).values
        z = torch.relu(self.head(p2))
        w = self.out.weight + self.fast.get("weight", 0)
        b = self.out.bias + self.fast.get("bias", 0)
        return [e_pool, p1, p2, z, z @ w.T + b]

    def forward(self, ids): return self.layers(ids)[-1]

    def probs(self, words):
        with torch.no_grad(): return torch.softmax(self(encode_words(words)), 1)

    def propose(self, word, k=3):
        p = self.probs([word])[0]
        top = torch.topk(p, k)
        return [(self.concepts[i], float(v)) for v, i in zip(top.values.tolist(), top.indices.tolist())]

    # ---- trace / snapshots -----------------------------------------------
    def record(self, spec, effects):
        node = self.graph.current.add_child(spec, effects, name=f"L{len(self.graph.nodes)}")
        self.graph.nodes[node.name] = node; self.graph.current = node

    def snapshot(self, name="good"):
        self._snapshots[name] = {k: v.detach().clone() for k, v in self.state_dict().items()}

    def rollback(self, name="good"):
        spec = RollbackCore(snapshot=name); spec.apply(self); return name in self._snapshots


@dataclass(frozen=True)
class TrainCore(TransformationSpec):
    """(Slow) training of the neural core on word -> concept pairs verified by WCE (distillation)."""
    name: str = "train_char_core"
    domain: TransformationDomain = TransformationDomain.LANGUAGE
    words: Tuple[str, ...] = ()
    labels: Tuple[str, ...] = ()
    steps: int = 300
    lr: float = 0.02
    invertible: bool = False

    def apply(self, core, *args, **kwargs):
        ids = encode_words(self.words)
        y = torch.tensor([core.concepts.index(c) for c in self.labels])
        opt = torch.optim.Adam(core.parameters(), lr=self.lr)
        for _ in range(self.steps):
            opt.zero_grad(); loss = torch.nn.functional.cross_entropy(core(ids), y); loss.backward(); opt.step()
        core.record(self, {"loss": loss.item(), "examples": len(self.words)})
        return core


@dataclass(frozen=True)
class FastAdaptCore(TransformationSpec):
    """Fast plasticity: only a delta on the output layer learns (slow weights intact)."""
    name: str = "fast_adapt_core"
    domain: TransformationDomain = TransformationDomain.LANGUAGE
    words: Tuple[str, ...] = ()
    labels: Tuple[str, ...] = ()
    steps: int = 100
    lr: float = 0.05
    invertible: bool = False

    def apply(self, core, *args, **kwargs):
        ids = encode_words(self.words)
        y = torch.tensor([core.concepts.index(c) for c in self.labels])
        dw = (core.fast.get("weight", torch.zeros_like(core.out.weight))).clone().requires_grad_(True)
        db = (core.fast.get("bias", torch.zeros_like(core.out.bias))).clone().requires_grad_(True)
        opt = torch.optim.Adam([dw, db], lr=self.lr)
        with torch.no_grad(): z = core.layers(ids)[3]
        for _ in range(self.steps):
            opt.zero_grad()
            logits = z @ (core.out.weight.detach() + dw).T + core.out.bias.detach() + db
            loss = torch.nn.functional.cross_entropy(logits, y); loss.backward(); opt.step()
        core.fast = {"weight": dw.detach(), "bias": db.detach()}
        core.record(self, {"loss": loss.item(), "examples": len(self.words)})
        return core


@dataclass(frozen=True)
class RollbackCore(TransformationSpec):
    name: str = "rollback_core"
    domain: TransformationDomain = TransformationDomain.LANGUAGE
    snapshot: str = "good"
    invertible: bool = False

    def apply(self, core, *args, **kwargs):
        if self.snapshot in core._snapshots:
            core.load_state_dict(core._snapshots[self.snapshot]); core.fast = {}
            core.record(self, {"restored": self.snapshot})
        return core


@dataclass(frozen=True)
class CorruptCore(TransformationSpec):
    """Controlled damage (audit protocol)."""
    name: str = "corrupt_core"
    domain: TransformationDomain = TransformationDomain.LANGUAGE
    scale: float = 1.0
    seed: int = 0
    invertible: bool = False

    def apply(self, core, *args, **kwargs):
        gen = torch.Generator().manual_seed(self.seed)
        with torch.no_grad():
            for p in (core.conv2.weight, core.head.weight):
                p.add_(self.scale * float(p.std()) * torch.randn(p.shape, generator=gen))
        core.record(self, {"scale": self.scale})
        return core


class WordCore(torch.nn.Module):
    """Word-tokenizer baseline: one embedding per seen word; unknown word -> OOV."""

    def __init__(self, vocab, size="small", seed=0):
        super().__init__()
        torch.manual_seed(seed)
        e, h = SIZES[size]
        self.vocab = {w: i + 1 for i, w in enumerate(sorted(set(vocab)))}
        self.emb = torch.nn.Embedding(len(self.vocab) + 1, e)
        self.head = torch.nn.Sequential(torch.nn.Linear(e, h), torch.nn.ReLU(), torch.nn.Linear(h, len(CONCEPTS)))

    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())

    def fit(self, words, labels, steps=300):
        ids = torch.tensor([self.vocab.get(w, 0) for w in words]); y = torch.tensor([CONCEPTS.index(c) for c in labels])
        opt = torch.optim.Adam(self.parameters(), lr=0.02)
        for _ in range(steps):
            opt.zero_grad(); torch.nn.functional.cross_entropy(self.head(self.emb(ids)), y).backward(); opt.step()

    def predict(self, word):
        if word not in self.vocab: return "OTHER"
        with torch.no_grad(): return CONCEPTS[int(self.head(self.emb(torch.tensor([self.vocab[word]]))).argmax())]


# ================================================================ B. learned morphology

@dataclass(frozen=True)
class SuffixRule(TransformationSpec):
    """Learned morphological transformation: form -> reference form (suffix strip/add)."""
    name: str = "suffix_rule"
    domain: TransformationDomain = TransformationDomain.LANGUAGE
    strip: str = ""
    add: str = ""
    invertible: bool = True

    def applies(self, form): return form.endswith(self.strip) and len(form) - len(self.strip) >= 3
    def apply(self, form, *args, **kwargs): return form[:len(form) - len(self.strip)] + self.add
    def get_inverse(self): return SuffixRule(name=f"inv({self.name})", strip=self.add, add=self.strip)


def _common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]: n += 1
    return n


class MorphologyLearner:
    """Suffix rules induced from pairs (form, reference form) with the same concept and the same stem.
    A rule is 'confident' only after observation on >= 2 distinct stems."""

    def __init__(self, min_stem=3, confident_support=2):
        self.rules: Dict[Tuple[str, str], set] = {}
        self.min_stem, self.confident_support = min_stem, confident_support

    def learn_pair(self, form, reference):
        n = _common_prefix(form, reference)
        if n < self.min_stem or form == reference: return None
        key = (form[n:], reference[n:])
        self.rules.setdefault(key, set()).add(form[:n])
        return key

    def rule_list(self):
        return [SuffixRule(name=f"-{s}→-{a}", strip=s, add=a) for (s, a) in self.rules]

    def support(self, rule): return len(self.rules.get((rule.strip, rule.add), ()))

    def candidates(self, form, known: Dict[str, str]):
        out = []
        for rule in self.rule_list():
            if rule.strip and rule.applies(form):
                ref = rule.apply(form)
                if ref in known and ref != form: out.append((ref, known[ref], rule, self.support(rule)))
        return sorted(out, key=lambda t: -t[3])


# ================================================================ C. lexical memory

@dataclass
class LexicalEvidence:
    form: str
    concepts: Counter
    source: str                 # grounded | morphology | morphology_weak | unknown
    support: int = 0
    via: Optional[str] = None   # reference form / rule

    @property
    def known(self): return self.source in ("grounded", "morphology")

    def supports(self, concept):
        if not self.concepts: return TruthValue.UNKNOWN
        if concept in self.concepts:
            return TruthValue.TRUE if self.source in ("grounded", "morphology") else TruthValue.UNKNOWN
        return TruthValue.FALSE if self.source == "grounded" else TruthValue.UNKNOWN


class LexicalConceptMemory:
    """Surface forms -> concepts, only from experiences explained by WCE.
    A word never grounded nor linked by a morphological rule remains UNKNOWN."""

    def __init__(self):
        self.grounded: Dict[str, Counter] = {}
        self.reference: Dict[str, str] = {}
        self.episodes: List[Dict[str, Any]] = []
        self.morphology = MorphologyLearner()

    def ground(self, form, concept, utterance="", evidence=""):
        self.grounded.setdefault(form, Counter())[concept] += 1
        self.episodes.append({"utterance": utterance, "form": form, "concept": concept, "evidence": evidence})
        for other, counts in self.grounded.items():
            if other != form and concept in counts and _common_prefix(other, form) >= 3:
                ref = self.reference.get(other, other)
                if ref != form:
                    self.reference.setdefault(form, ref)
                    self.morphology.learn_pair(form, ref)
                break
        self.reference.setdefault(form, form)

    def known_concepts(self):
        return {f: c.most_common(1)[0][0] for f, c in self.grounded.items()}

    def lookup(self, form) -> LexicalEvidence:
        if form in self.grounded: return LexicalEvidence(form, Counter(self.grounded[form]), "grounded", sum(self.grounded[form].values()))
        cands = self.morphology.candidates(form, self.known_concepts())
        if cands:
            ref, concept, rule, support = cands[0]
            source = "morphology" if support >= self.morphology.confident_support else "morphology_weak"
            return LexicalEvidence(form, Counter({concept: 1}), source, support, f"{rule.name} → {ref}")
        return LexicalEvidence(form, Counter(), "unknown")

    def retrieve(self, form):
        return [e for e in self.episodes if e["form"] == form]


# ================================================================ physical scene and WCE specs

def make_scene() -> World:
    w = World()
    for name, top in SUPPORTS.items():
        w.add_entity(Entity(name, "support", {"y": 0.0, "top": top, "support": True, "fixe": True}))
    for name, (support, props) in ENTITIES.items():
        w.add_entity(Entity(name, "objet", {"y": SUPPORTS[support], "masse": 1.0, **props}))
        w.add_relation(name, "supported_by", support)
    return w


def support_of(world, entity):
    return next((r.object for r in world.relations if r.subject == entity and r.predicate == "supported_by"), None)


def place_spec(entity, support):
    def compiler(world):
        before = world.entities[entity].properties["y"]
        top = world.entities[support].properties["top"]
        rel = [RelationChange("remove", entity, "supported_by", r.object) for r in world.relations
               if r.subject == entity and r.predicate == "supported_by"]
        return Transformation(f"place({entity},{support})", [Change(entity, "y", before, top)],
                              rel + [RelationChange("add", entity, "supported_by", support)])
    return DeclarativeTransformationSpec(name=f"place({entity},{support})", domain=TransformationDomain.PHYSICS,
                                         parameters={"entity": entity, "support": support},
                                         preserves=InvariantSet({Invariant.ENTITY_IDENTITY, Invariant.MASS}),
                                         changes={"y", "supported_by"}, compiler=compiler)


def lift_spec(entity, height):
    from core.action import LiftAction
    return DeclarativeTransformationSpec(name=f"lift({entity},{height:g})", domain=TransformationDomain.PHYSICS,
                                         parameters={"entity": entity, "height": height},
                                         preserves=InvariantSet({Invariant.ENTITY_IDENTITY, Invariant.MASS}),
                                         changes={"y"}, compiler=lambda world: LiftAction(entity, height)._compile_executable(world))


def physical_candidate_generator(state, goal):
    """Proposals for the WCE planner: place on each support, lift to the target height."""
    target = goal.target if isinstance(goal.target, tuple) else (goal.target,)
    entity = target[0]
    supports = [n for n, e in state.entities.items() if e.properties.get("support") and n != entity]
    cands = [place_spec(entity, s) for s in supports]
    if goal.predicate == "height" and goal.condition is not None:
        h = float(goal.condition.right)
        if h >= 0: cands.append(lift_spec(entity, h))
    return cands


def apply_spec(world, spec):
    from core.transformation import apply
    apply(world, spec.compile(world))
    return world


def describe(world, entity):
    if entity not in world.entities: return f"{entity}: ?"
    return f"{entity}: y={world.entities[entity].properties['y']:g}, sur={support_of(world, entity) or '-'}"


# ================================================================ D. sentence parsing

@dataclass
class Clause:
    verb: Optional[str] = None
    entity: Optional[str] = None
    support: Optional[str] = None
    source: Optional[str] = None
    height: Optional[float] = None
    unknown_nouns: List[str] = field(default_factory=list)


@dataclass
class Parsed:
    text: str
    kind: str                            # command | question | claim
    clauses: List[Clause] = field(default_factory=list)
    sequence: Optional[str] = None
    invalid_sequence: Optional[str] = None
    question: Optional[Tuple[str, str]] = None
    claim: Optional[Tuple[str, str, str]] = None      # (subject, 'sur'|'attr', value)


def parse(text) -> Parsed:
    toks = tokenize_fr(text)
    seq = next((t for t in toks if is_sequence(t) and set(t) <= set(SEQ_ALPHABET)), None)
    bad = next((t for t in toks if is_sequence(t) and not set(t) <= set(SEQ_ALPHABET)
                and t not in NOUNS and t not in NONSENSE and t not in TRAPS and t not in TRUE_CONCEPT), None)
    if "?" in toks or (toks and toks[0] in ("où", "quelle", "quel")):
        entity = next((t for t in toks if t in NOUNS), None)
        qtype = "where" if "où" in toks else "height"
        return Parsed(text, "question", question=(qtype, entity))
    if "est" in toks:
        i = toks.index("est")
        subject = next((t for t in toks[:i] if t in NOUNS), None)
        rest = [t for t in toks[i + 1:] if t not in DETERMINERS]
        if rest and rest[0] == "sur":
            return Parsed(text, "claim", claim=(subject, "sur", next((t for t in rest[1:] if t in NOUNS), None)))
        value = next((t for t in rest if t not in PREPOSITIONS), None)
        return Parsed(text, "claim", claim=(subject, "attr", value))
    clauses, current = [], []
    for t in toks:
        if t == "puis": clauses.append(current); current = []
        else: current.append(t)
    clauses.append(current)
    out = []
    for ctoks in clauses:
        c = Clause()
        i = 0
        while i < len(ctoks):
            t = ctoks[i]
            nxt = [x for x in ctoks[i + 1:] if x not in DETERMINERS]
            if t == "sur" and nxt and nxt[0] in NOUNS: c.support = nxt[0]; i = ctoks.index(nxt[0], i + 1) + 1; continue
            if t in ("du", "de", "d'", "depuis") and nxt and nxt[0] in NOUNS:
                c.source = nxt[0]; i = ctoks.index(nxt[0], i + 1) + 1; continue
            if t in ("à", "au") and nxt and is_number(nxt[0]): c.height = float(nxt[0]); i = ctoks.index(nxt[0], i + 1) + 1; continue
            if t in NOUNS:
                if c.entity is None: c.entity = t
            elif t in DETERMINERS or t in PREPOSITIONS or t in UNITS or t in FUNCTION or is_number(t) or t == seq or t == bad:
                pass
            elif c.verb is None:
                c.verb = t
            i += 1
        out.append(c)
    return Parsed(text, "command", out, seq, bad)


# ================================================================ E/F. interpretations -> Goals / programs

@dataclass
class Check:
    name: str
    value: TruthValue
    detail: str = ""
    contradiction: bool = False


@dataclass
class Interpretation:
    parsed: Parsed
    concepts: Tuple[Optional[str], ...]
    neural_confidence: float = 0.0
    prior: float = 0.0
    sources: Tuple[str, ...] = ()
    domain: str = "invalid"
    goals: List[Goal] = field(default_factory=list)
    program: Optional[TransformationSpec] = None
    checks: List[Check] = field(default_factory=list)
    answer: Optional[str] = None
    plan_specs: List[str] = field(default_factory=list)
    search_nodes: int = 0

    @property
    def label(self): return "+".join(c or "?" for c in self.concepts)
    @property
    def contradictions(self): return [c for c in self.checks if c.value is TruthValue.FALSE and c.contradiction]
    @property
    def failures(self): return [c for c in self.checks if c.value is TruthValue.FALSE]
    @property
    def structural_support(self):
        return sum(c.value is TruthValue.TRUE for c in self.checks) / len(self.checks) if self.checks else 0.0


def goal_for(concept, clause, world):
    """Language produces a Goal (not an action); the WCE planner finds the transformations."""
    e = clause.entity
    if concept == "FALL":
        if clause.support: return Goal("supported_by", (e, clause.support))
        return Goal("height", (e,), Equals(PropertyRef(e, "y"), 0.0))
    if concept == "RISE":
        if clause.height is not None: return Goal("height", (e,), Equals(PropertyRef(e, "y"), clause.height))
        if clause.support: return Goal("supported_by", (e, clause.support))
        y = world.entities[e].properties["y"] if e in world.entities else 0.0
        return Goal("height", (e,), GreaterThanOrEqual(PropertyRef(e, "y"), y + 1.0))
    if concept == "PLACE":
        if clause.support: return Goal("supported_by", (e, clause.support))
        if clause.height is not None: return Goal("height", (e,), Equals(PropertyRef(e, "y"), clause.height))
        return None
    return None


def build_interpretation(parsed, concepts, world):
    interp = Interpretation(parsed, tuple(concepts))
    domains = {("sequence" if c in SEQUENCE_CONCEPTS else "physics" if c in PHYSICAL else "unknown") for c in concepts}
    if len(domains) == 1: interp.domain = domains.pop()
    if interp.domain == "sequence":
        interp.program = compose(*[PRIMITIVES[SEQUENCE_CONCEPTS[c]] for c in concepts])
    elif interp.domain == "physics":
        interp.goals = [goal_for(c, cl, world) for c, cl in zip(concepts, parsed.clauses)]
    return interp


# ================================================================ W. external verification

class ExternalVerifier(ABC):
    @abstractmethod
    def verify(self, subject, value) -> str: ...


class FakeExternalVerifier(ExternalVerifier):
    """Simulated factual source: SUPPORTED / REFUTED / UNKNOWN (no network access)."""

    def __init__(self, facts=None):
        self.facts = facts or {("table", "bois"): True, ("balle", "verre"): False, ("bloc", "bois"): True,
                               ("cube", "bois"): False}
        self.calls = 0

    def verify(self, subject, value):
        self.calls += 1
        if (subject, value) not in self.facts: return "UNKNOWN"
        return "SUPPORTED" if self.facts[(subject, value)] else "REFUTED"


# ================================================================ H. WCE structural verification

class StructuralVerifier:
    def __init__(self, lexicon: LexicalConceptMemory):
        self.lexicon = lexicon
        self.checks_run = 0
        self.search_nodes = 0

    def _add(self, interp, name, value, detail="", contradiction=False):
        interp.checks.append(Check(name, value, detail, contradiction)); self.checks_run += 1

    def verify(self, interp: Interpretation, world: World):
        p = interp.parsed
        interp.checks = []
        for concept, clause in zip(interp.concepts, p.clauses):
            ev = self.lexicon.lookup(clause.verb) if clause.verb else LexicalEvidence("", Counter(), "unknown")
            value = ev.supports(concept) if concept else TruthValue.UNKNOWN
            self._add(interp, f"concept({clause.verb}={concept})", value,
                      f"{ev.source}{' via ' + ev.via if ev.via else ''}", contradiction=value is TruthValue.FALSE)
        if interp.domain == "invalid":
            self._add(interp, "domain", TruthValue.FALSE, "concepts from incompatible domains", True); return interp
        if interp.domain == "unknown":
            self._add(interp, "domain", TruthValue.UNKNOWN, "unknown concept"); return interp
        if interp.domain == "sequence":
            self._add(interp, "context_match(sequence)", TruthValue.TRUE if p.sequence else TruthValue.FALSE,
                      p.sequence or "no sequence", contradiction=p.invalid_sequence is not None or not p.sequence)
            if p.invalid_sequence:
                self._add(interp, "program_executable", TruthValue.FALSE, f"symbols outside the alphabet: {p.invalid_sequence}", True)
            elif p.sequence:
                out = interp.program.apply(seq_encode(p.sequence).unsqueeze(0))[0]
                interp.answer = seq_decode(out)
                self._add(interp, "program_executable", TruthValue.TRUE, interp.program.name)
                changed = sorted(out.tolist()) != sorted(seq_encode(p.sequence).tolist())
                expected = not all(Invariant.SYMBOL_MULTISET in s.preserves.invariants
                                   for s in getattr(interp.program, "transformations", (interp.program,)))
                self._add(interp, "invariant(multiset)", TruthValue.TRUE if changed == expected or not changed else TruthValue.FALSE,
                          "multiset consistent with the declared invariants")
            return interp
        # physics: entities, relations, coherence, preconditions, plan, invariants
        branch = copy.deepcopy(world)
        from core.planner import TransformationPlanner
        planner = TransformationPlanner([physical_candidate_generator], max_depth=2, max_nodes=60)
        for concept, clause, goal in zip(interp.concepts, p.clauses, interp.goals):
            ent = clause.entity
            self._add(interp, f"entity({ent})", TruthValue.TRUE if ent in branch.entities else TruthValue.FALSE,
                      "entity absent from the world" if ent not in branch.entities else "", contradiction=ent is not None and ent not in branch.entities)
            if ent not in branch.entities: continue
            if goal is None:
                self._add(interp, "goal_complete", TruthValue.FALSE, f"{concept} requires a complement (sur X / à N)", True); continue
            if clause.support:
                ok = clause.support in branch.entities and branch.entities[clause.support].properties.get("support") and clause.support != ent
                self._add(interp, f"relation_plausible(sur {clause.support})", TruthValue.TRUE if ok else TruthValue.FALSE,
                          "" if ok else "impossible support", contradiction=not ok)
                if not ok: continue
            if clause.height is not None:
                ok = clause.height >= 0
                self._add(interp, "goal_coherent(hauteur >= 0)", TruthValue.TRUE if ok else TruthValue.FALSE,
                          f"height {clause.height:g}", contradiction=not ok)
                if not ok: continue
            if clause.source:
                ok = support_of(branch, ent) == clause.source
                self._add(interp, f"precondition(sur {clause.source})", TruthValue.TRUE if ok else TruthValue.FALSE,
                          f"{ent} is on {support_of(branch, ent)}", contradiction=not ok)
                if not ok: continue
            y = branch.entities[ent].properties["y"]
            if concept == "RISE" and clause.support and SUPPORTS.get(clause.support, 0) <= y:
                self._add(interp, "goal_coherent(monter)", TruthValue.FALSE, f"{clause.support} is not higher", True); continue
            if concept == "FALL" and clause.support and SUPPORTS.get(clause.support, 0) >= y:
                self._add(interp, "goal_coherent(tomber)", TruthValue.FALSE, f"{clause.support} is not lower", True); continue
            result = planner.search(branch, goal)
            self.search_nodes += result.explored_nodes; interp.search_nodes += result.explored_nodes
            self._add(interp, f"plan({goal.predicate})", TruthValue.TRUE if result.success else TruthValue.FALSE,
                      " -> ".join(s.name for s in result.transformations) or result.reason_if_failed or "")
            if not result.success: continue
            interp.plan_specs += [s.name for s in result.transformations]
            branch = result.final_state
        ok = all(e.properties.get("y", 0) >= 0 for e in branch.entities.values()) and \
             all(sum(1 for r in branch.relations if r.subject == n and r.predicate == "supported_by") <= 1 for n in branch.entities)
        self._add(interp, "invariants(y>=0, un seul support)", TruthValue.TRUE if ok else TruthValue.FALSE, "", not ok)
        interp.branch_world = branch
        last = next((c.entity for c in reversed(p.clauses) if c.entity), None)
        if not interp.failures: interp.answer = describe(branch, last)
        return interp


# ================================================================ I/M. risk and policy

SUPPORTED, UNCERTAIN, CONTRADICTED, UNKNOWN_STATUS, NEEDS_VERIFICATION = \
    "SUPPORTED", "UNCERTAIN", "CONTRADICTED", "UNKNOWN", "NEEDS_VERIFICATION"


@dataclass
class HallucinationRiskState:
    neural_confidence: float = 0.0
    structural_support: float = 0.0
    contradiction_count: int = 0
    unknown_concepts: List[str] = field(default_factory=list)
    branch_disagreement: float = 0.0
    external_support_state: str = "NOT_APPLICABLE"
    geometric_risk: Optional[float] = None
    geometric_level: Optional[str] = None
    neural_only_concepts: List[str] = field(default_factory=list)
    lexical_conflict_only: bool = False

    @property
    def overall_status(self):
        if self.unknown_concepts: return UNKNOWN_STATUS
        if self.contradiction_count or self.external_support_state == "REFUTED": return CONTRADICTED
        if self.neural_only_concepts: return NEEDS_VERIFICATION
        if self.branch_disagreement >= 0.5: return UNCERTAIN
        if self.structural_support >= 1.0 and self.geometric_level != "HIGH": return SUPPORTED
        return UNCERTAIN


ACCEPT, VERIFY, REROUTE, ABSTAIN, REJECT, ASK, ACCEPT_TENTATIVE = \
    "ACCEPT", "VERIFY", "REROUTE", "ABSTAIN", "REJECT", "ASK", "ACCEPT_TENTATIVE"


class AntiHallucinationPolicy:
    """Explicit multi-signal policy. The neural score alone never authorises a physical action."""

    def __init__(self, use_geometric=True, tentative_confidence=0.9):
        self.use_geometric, self.tentative_confidence = use_geometric, tentative_confidence

    def decide(self, risk: HallucinationRiskState, domain="physics"):
        geo = risk.geometric_level if self.use_geometric else None
        # an unknown concept comes before everything: the contradictions of a guessed concept are meaningless
        if risk.unknown_concepts: return ABSTAIN, f"unknown concept: {', '.join(risk.unknown_concepts)}"
        if risk.contradiction_count and risk.lexical_conflict_only and geo == "HIGH":
            return REROUTE, "neural proposal contradicted by the lexical memory + high geometric risk"
        if risk.contradiction_count: return REJECT, "structural contradiction: branch rejected"
        if risk.external_support_state == "REFUTED": return REJECT, "refuted by the external verification"
        if geo == "HIGH" and risk.branch_disagreement >= 0.3: return REROUTE, "high geometric risk + branch disagreement"
        if risk.branch_disagreement >= 0.5: return ASK, "valid but divergent interpretations"
        if risk.neural_only_concepts:
            if domain == "sequence" and geo == "LOW" and risk.neural_confidence >= self.tentative_confidence:
                return ACCEPT_TENTATIVE, "confident neural proposal, low geometric risk, reversible domain"
            return VERIFY, f"neural support only for {', '.join(risk.neural_only_concepts)}"
        if risk.external_support_state == "UNKNOWN": return VERIFY, "internally coherent, not externally verifiable"
        if risk.structural_support >= 1.0: return ACCEPT, "complete structural support"
        return VERIFY, "partial structural support"


# ================================================================ pipeline

@dataclass
class Response:
    text: str
    decision: str
    status: str
    answer: Optional[str]
    reason: str
    interpretation: Optional[Interpretation] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    risk: Optional[HallucinationRiskState] = None
    trace: List[str] = field(default_factory=list)
    memory: set = field(default_factory=set)
    corrected: bool = False
    checks: int = 0
    candidates_evaluated: int = 0
    search_nodes: int = 0

    @property
    def output(self):
        """Form comparable with the oracle."""
        if self.decision in (ACCEPT, ACCEPT_TENTATIVE): return self.answer
        if self.decision == REJECT: return "REJECT"
        return "ABSTAIN"


SYSTEMS = {
    "1_neural_only": dict(neural=True, wce=False),
    "2_neural_plasticity": dict(neural=True, wce=False, plasticity=True),
    "3_wce_only": dict(neural=False, wce=True),
    "4_neural_wce": dict(neural=True, wce=True, geometric=False, correction=False),
    "5_neural_wce_geometric": dict(neural=True, wce=True, geometric=True, correction=False),
    "6_full_self_correction": dict(neural=True, wce=True, geometric=True, correction=True),
}


class LanguageCore:
    def __init__(self, *, neural=True, wce=True, geometric=True, correction=True, plasticity=False,
                 size="small", seed=0, external: ExternalVerifier = None):
        self.use_neural, self.use_wce, self.use_geometric = neural, wce, geometric
        self.correction, self.plasticity = correction, plasticity
        self.lexicon = LexicalConceptMemory()
        self.verifier = StructuralVerifier(self.lexicon)
        self.policy = AntiHallucinationPolicy(use_geometric=geometric)
        self.external = external or FakeExternalVerifier()
        self.core = CharCore(size, seed) if neural else None
        self.monitor, self.risk_estimator = None, None
        self.log: List[Response] = []
        self.pending: Dict[str, set] = {}
        self.seed = seed

    # ---- learning from experience (WCE explains, the neural part distils) ----
    def learn_physics(self, text, before: World, after: World, label=None):
        """Explains the observation: which concepts, via the WCE planner, reproduce the observed state?
        Cross-situational learning: intersection of the explanations; grounding when only one remains."""
        parsed = parse(text); clause = parsed.clauses[0]
        if not clause.verb or clause.entity not in before.entities: return None
        if not self.use_wce:
            if label: self._ground(clause.verb, label, text, "physics(label)")
            return label
        explaining = set()
        for concept in sorted(PHYSICAL):
            interp = build_interpretation(parsed, [concept], before)
            if not interp.goals or interp.goals[0] is None: continue
            StructuralVerifier(self.lexicon).verify(interp, before)
            world = getattr(interp, "branch_world", None)
            hard = [c for c in interp.checks if c.value is TruthValue.FALSE and not c.name.startswith("concept")]
            if world is not None and not hard and describe(world, clause.entity) == describe(after, clause.entity):
                explaining.add(concept)
        known = self.pending.get(clause.verb)
        candidates = explaining if known is None else known & explaining
        self.pending[clause.verb] = candidates
        if len(candidates) == 1:
            concept = next(iter(candidates))
            self._ground(clause.verb, concept, text, "physics")
            return concept
        return None

    def learn_sequence(self, text, output, label=None):
        from core.symbolic import search_programs
        parsed = parse(text)
        if not parsed.sequence or not parsed.clauses[0].verb: return None
        concept = None
        if self.use_wce:
            r = search_programs(seq_encode(parsed.sequence).unsqueeze(0), seq_encode(output).unsqueeze(0),
                                [PRIMITIVES[p] for p in SEQUENCE_CONCEPTS.values()], max_depth=1, first=False)
            if len(r.consistent) == 1: concept = PRIMITIVE_TO_CONCEPT[r.consistent[0]]
        else:
            concept = label
        if concept: self._ground(parsed.clauses[0].verb, concept, text, "sequence")
        return concept

    def _ground(self, form, concept, text, evidence):
        self.lexicon.ground(form, concept, text, evidence)
        if self.core is not None and self.plasticity:
            words, labels = [form] * 4, [concept] * 4
            for f, counts in self.lexicon.grounded.items():      # replay: avoids interference from a single word
                words.append(f); labels.append(counts.most_common(1)[0][0])
            FastAdaptCore(words=tuple(words), labels=tuple(labels)).apply(self.core)

    def fit_core(self, steps=300):
        """Distillation: the neural core learns the verified bindings (and the non-verb words)."""
        if self.core is None: return
        words, labels = [], []
        for form, counts in self.lexicon.grounded.items():
            for concept, n in counts.items(): words += [form] * n; labels += [concept] * n
        words += OTHER_WORDS; labels += ["OTHER"] * len(OTHER_WORDS)
        TrainCore(words=tuple(words), labels=tuple(labels), steps=steps).apply(self.core)
        self.core.fast = {}
        self.core.snapshot("good")

    def attach_monitor(self, monitor, estimator):
        self.monitor, self.risk_estimator = monitor, estimator

    # ---- candidates -------------------------------------------------------
    def concept_candidates(self, verb, k=3):
        """Ordered list: neural proposals first (fast), then lexical memory."""
        cands: Dict[str, Dict[str, Any]] = {}
        if self.use_neural and verb:
            for concept, p in self.core.propose(verb, k):
                if concept != "OTHER": cands[concept] = {"neural": p, "lexical": 0.0}
        if self.use_wce and verb:
            ev = self.lexicon.lookup(verb)
            total = sum(ev.concepts.values()) or 1
            for concept, n in ev.concepts.items():
                w = n / total * (1.0 if ev.source in ("grounded", "morphology") else 0.4)
                cands.setdefault(concept, {"neural": 0.0, "lexical": 0.0})["lexical"] = w
        return [(c, v["neural"], v["lexical"]) for c, v in cands.items()]

    def interpretations(self, parsed, world, k=3):
        per_clause = []
        for clause in parsed.clauses:
            cands = self.concept_candidates(clause.verb, k)
            per_clause.append(cands or [(None, 0.0, 0.0)])
        import itertools
        out = []
        for combo in itertools.product(*per_clause):
            concepts = [c for c, _, _ in combo]
            interp = build_interpretation(parsed, concepts, world)
            interp.neural_confidence = min(n for _, n, _ in combo)
            lexical = min(l for _, _, l in combo)
            interp.prior = (interp.neural_confidence if self.use_neural else 0) + (lexical if self.use_wce else 0)
            out.append(interp)
        if self.use_neural:   # the neural part proposes: its order prevails, the memory breaks ties
            out.sort(key=lambda i: (-i.neural_confidence, -i.prior))
        else:
            out.sort(key=lambda i: -i.prior)
        return out[:9]

    # ---- risk ----------------------------------------------------------
    def risk_for(self, interp, others):
        risk = HallucinationRiskState(neural_confidence=interp.neural_confidence,
                                      structural_support=interp.structural_support,
                                      contradiction_count=len(interp.contradictions),
                                      lexical_conflict_only=any(c.name.startswith("concept") for c in interp.contradictions))
        for concept, clause in zip(interp.concepts, interp.parsed.clauses):
            ev = self.lexicon.lookup(clause.verb) if clause.verb else None
            if ev is not None and ev.known: continue
            if ev is not None and ev.source == "morphology_weak":
                risk.neural_only_concepts.append(clause.verb); continue
            # unknown to WCE: a neural proposal is retained (to be verified) only if it is confident and sound
            confident = self.use_neural and concept is not None and \
                dict(self.core.propose(clause.verb, 3)).get(concept, 0.0) >= 0.9
            healthy = not self.use_geometric or self.monitor is None or self._geometric(clause.verb)[1] == "LOW"
            if confident and healthy: risk.neural_only_concepts.append(clause.verb)
            else: risk.unknown_concepts.append(clause.verb or "?")
        valid = [o for o in others if not o.failures and o.answer is not None]
        if len(valid) >= 2 and len({o.answer for o in valid}) > 1:
            w = sorted([max(o.prior, 1e-6) for o in valid], reverse=True)
            risk.branch_disagreement = round(w[1] / w[0], 3)
        if self.monitor is not None and self.use_geometric:
            scores = [self._geometric(c.verb) for c in interp.parsed.clauses if c.verb]
            if scores:
                risk.geometric_risk = max(s for s, _ in scores)
                risk.geometric_level = max((l for _, l in scores), key=["LOW", "MEDIUM", "HIGH"].index)
        return risk

    def _geometric(self, word):
        if self.monitor is None or self.risk_estimator is None or not word: return 0.0, "LOW"
        z = self.monitor.observe(self.core, word)
        return self.risk_estimator.score(z), self.risk_estimator.level(z)

    # ---- processing --------------------------------------------------------
    def process(self, text, world: World, *, execute=True) -> Response:
        parsed = parse(text)
        if parsed.kind == "question": return self._question(parsed, world)
        if parsed.kind == "claim": return self._claim(parsed, world)
        if not self.use_wce: return self._neural_only(parsed, world, execute)

        interps = self.interpretations(parsed, world)
        budget = 1 if self.use_neural else len(interps)       # neural+WCE: only the 1st candidate is verified
        checks_before, nodes_before = self.verifier.checks_run, self.verifier.search_nodes
        for interp in interps[:budget]: self.verifier.verify(interp, world)
        if any(len(self.lexicon.lookup(c.verb).concepts) > 1 for c in parsed.clauses if c.verb):
            for interp in interps[budget:]: self.verifier.verify(interp, world)
            budget = len(interps)
        amplitudes = [ConfigurationAmplitude(i.label, i, weight=max(i.prior, 1e-6)) for i in interps]
        chosen = self._select(interps[:budget])
        risk = self.risk_for(chosen, interps[:budget]) if chosen else HallucinationRiskState(unknown_concepts=[c.verb or "?" for c in parsed.clauses])
        decision, reason = self.policy.decide(risk, chosen.domain if chosen else "physics")
        trace = [f"candidates: {[i.label for i in interps]}", f"verified: {chosen.label if chosen else None} -> {decision}"]
        if decision == REROUTE and self.use_geometric and len(interps) > budget:
            # the geometric risk triggers the verification of the alternative branches
            for interp in interps[budget:]: self.verifier.verify(interp, world)
            budget = len(interps)
            alt = self._select([i for i in interps if not i.failures] or interps)
            if alt is not None:
                chosen = alt; risk = self.risk_for(chosen, interps); decision, reason = self.policy.decide(risk, chosen.domain)
                trace.append(f"reroute -> {chosen.label}: {decision}")
        corrected = False
        if self.correction and decision in (REJECT, VERIFY, REROUTE, ABSTAIN):
            chosen, risk, decision, reason, steps, budget = self._self_correct(parsed, world, interps, chosen, risk, decision, reason, budget)
            trace += steps; corrected = any(s.startswith("✓") for s in steps)
        if decision == VERIFY and chosen is not None and not risk.neural_only_concepts and risk.external_support_state == "NOT_APPLICABLE":
            decision, reason = (ACCEPT, "complete structural verification") if chosen.structural_support >= 1.0 else (ABSTAIN, reason)
        if decision == VERIFY: decision, reason = ABSTAIN, f"NEEDS_VERIFICATION : {reason}"
        if decision == ASK: reason = f"ASK : {reason} ({', '.join(sorted({i.answer for i in interps if i.answer and not i.failures}))})"
        if decision in (ACCEPT, ACCEPT_TENTATIVE) and execute and chosen.domain == "physics":
            self._execute(chosen, world)
        response = Response(text, decision, risk.overall_status, chosen.answer if chosen else None, reason, chosen,
                            [{"label": a.label, "weight": round(a.weight, 3), "support": round(a.state.structural_support, 2),
                              "failures": [f.name for f in a.state.failures]} for a in amplitudes],
                            risk, trace, self._memory(chosen, risk), corrected,
                            self.verifier.checks_run - checks_before, budget,
                            self.verifier.search_nodes - nodes_before)
        if decision == ABSTAIN and risk.unknown_concepts:
            response.reason = f"UNKNOWN CONCEPT: {', '.join(risk.unknown_concepts)}"
        self.log.append(response)
        return response

    def _select(self, interps):
        if not interps: return None
        scored = sorted(interps, key=lambda i: (-(len(i.failures) == 0), -i.structural_support, -i.prior))
        return scored[0]

    def _self_correct(self, parsed, world, interps, chosen, risk, decision, reason, budget):
        """1 rollback of the core, 2 reroute, 3 structural search, 4 alternative interpretation,
        5 episodic memory, 6 re-execution. Every step is traced."""
        steps = [f"self-correction: start {chosen.label if chosen else None} ({decision})"]
        if risk.unknown_concepts and not any(self.lexicon.retrieve(w) for w in risk.unknown_concepts):
            steps.append("memory: no experience for " + ", ".join(risk.unknown_concepts) + " -> abstention maintained")
            return chosen, risk, decision, reason, steps, budget
        if risk.geometric_level == "HIGH" and self.core is not None and self.core._snapshots:
            self.core.rollback("good"); steps.append("1. rollback of the neural core to the snapshot 'good'")
            interps = self.interpretations(parsed, world)
        for interp in interps:
            if not interp.checks: self.verifier.verify(interp, world)
        budget = len(interps)
        steps.append(f"2-4. reroute + structural search over {len(interps)} interpretation(s)")
        valid = [i for i in interps if not i.failures]
        if not valid:
            saved = self.use_neural; self.use_neural = False
            alt = self.interpretations(parsed, world); self.use_neural = saved
            for interp in alt: self.verifier.verify(interp, world)
            valid = [i for i in alt if not i.failures]; interps = interps + alt
            steps.append(f"3. structural search without neural proposal: {len(valid)} valid")
        for clause in parsed.clauses:
            if clause.verb and self.lexicon.retrieve(clause.verb):
                steps.append(f"5. episodic memory: '{clause.verb}' seen in « {self.lexicon.retrieve(clause.verb)[-1]['utterance']} »")
        if not valid:
            steps.append("6. no valid branch -> decision unchanged")
            return chosen, risk, decision, reason, steps, budget
        new = self._select(valid)
        new_risk = self.risk_for(new, valid)
        new_decision, new_reason = self.policy.decide(new_risk, new.domain)
        mark = "✓" if new_decision in (ACCEPT, ACCEPT_TENTATIVE) and (chosen is None or new.label != chosen.label or decision != new_decision) else "·"
        steps.append(f"{mark} 6. re-execution: {new.label} -> {new_decision}")
        return new, new_risk, new_decision, new_reason, steps, budget

    def _execute(self, interp, world):
        for spec_name in []: pass
        final = getattr(interp, "branch_world", None)
        if final is None: return
        world.entities.clear(); world.entities.update(copy.deepcopy(final.entities))
        world.relations[:] = copy.deepcopy(final.relations)

    def _neural_only(self, parsed, world, execute):
        concepts = []
        for clause in parsed.clauses:
            cands = [(c, p) for c, p in self.core.propose(clause.verb or "?", 3) if c != "OTHER"]
            concepts.append(cands[0][0] if cands else None)
        interp = build_interpretation(parsed, concepts, world)
        interp.neural_confidence = min((self.core.propose(c.verb or "?", 1)[0][1] for c in parsed.clauses), default=0.0)
        answer = None
        try:
            if interp.domain == "sequence" and parsed.sequence:
                answer = seq_decode(interp.program.apply(seq_encode(parsed.sequence).unsqueeze(0))[0])
            elif interp.domain == "physics":
                target = world if execute else copy.deepcopy(world)
                for concept, clause in zip(concepts, parsed.clauses):
                    ent = clause.entity
                    if concept == "FALL": apply_spec(target, place_spec(ent, clause.support or "sol"))
                    elif concept == "RISE":
                        if clause.height is not None: apply_spec(target, lift_spec(ent, clause.height))
                        elif clause.support: apply_spec(target, place_spec(ent, clause.support))
                        else: apply_spec(target, lift_spec(ent, target.entities[ent].properties["y"] + 1.0))
                    elif concept == "PLACE": apply_spec(target, place_spec(ent, clause.support))
                answer = describe(target, next((c.entity for c in reversed(parsed.clauses) if c.entity), None))
        except Exception:
            answer = "ERREUR"
        response = Response(parsed.text, ACCEPT, UNCERTAIN, answer, "neural only, without verification", interp,
                            memory={"slow_parametric"} | ({"fast_parametric"} if self.core.fast else set()))
        self.log.append(response)
        return response

    def _question(self, parsed, world):
        qtype, entity = parsed.question
        if entity not in world.entities:
            return self._finish(Response(parsed.text, ABSTAIN, UNKNOWN_STATUS, None, f"UNKNOWN ENTITY: {entity}", memory={"current_world"}))
        if qtype == "where":
            answer = f"{entity} est sur {support_of(world, entity) or 'rien'}"
        else:
            answer = f"{entity}: y={world.entities[entity].properties['y']:g}"
        return self._finish(Response(parsed.text, ACCEPT, SUPPORTED, answer, "read from the world state", memory={"current_world", "structural"}, checks=1))

    def _claim(self, parsed, world):
        subject, kind, value = parsed.claim
        risk = HallucinationRiskState(structural_support=1.0)
        if subject not in world.entities:
            return self._finish(Response(parsed.text, ABSTAIN, UNKNOWN_STATUS, None, f"UNKNOWN ENTITY: {subject}", risk=risk))
        if kind == "sur":
            truth = support_of(world, subject) == value
            return self._finish(Response(parsed.text, ACCEPT, SUPPORTED if truth else CONTRADICTED, "VRAI" if truth else "FAUX",
                                         "internal coherence (world state)", risk=risk, memory={"current_world"}, checks=1))
        props = world.entities[subject].properties
        if value in props.values():
            return self._finish(Response(parsed.text, ACCEPT, SUPPORTED, "VRAI", "internal coherence (world property)",
                                         risk=risk, memory={"current_world"}, checks=1))
        state = self.external.verify(subject, value) if self.external else "UNKNOWN"
        risk.external_support_state = state
        if state == "SUPPORTED": decision, answer = ACCEPT, "VRAI"
        elif state == "REFUTED": decision, answer = ACCEPT, "FAUX"
        else: decision, answer = ABSTAIN, None
        return self._finish(Response(parsed.text, decision, SUPPORTED if state == "SUPPORTED" else CONTRADICTED if state == "REFUTED" else NEEDS_VERIFICATION,
                                     answer, f"external verification: {state} (the internal world says nothing)", risk=risk,
                                     memory={"external_verification", "current_world"}, checks=1))

    def _finish(self, response):
        self.log.append(response); return response

    def _memory(self, interp, risk):
        used = set()
        if interp is None: return used
        if interp.domain == "physics": used.add("current_world")
        if self.use_neural and interp.neural_confidence > 0: used.add("slow_parametric")
        if self.core is not None and self.core.fast: used.add("fast_parametric")
        for clause in interp.parsed.clauses:
            ev = self.lexicon.lookup(clause.verb) if clause.verb else None
            if ev is not None and ev.source == "grounded": used |= {"episodic", "structural"}
            elif ev is not None and ev.source.startswith("morphology"): used.add("structural")
        if interp.goals or interp.program is not None: used.add("structural")
        if risk.external_support_state not in ("NOT_APPLICABLE",): used.add("external_verification")
        return used


# ================================================================ Y. bootstrapping, stream, evaluation

def bootstrap_experiences():
    """Initial experiences (sentence + observation). The labels are used only by the systems without WCE."""
    phys = [("le cube tombe", "cube", "FALL"), ("le cube tomber", "cube", "FALL"), ("le cube tombait", "cube", "FALL"),
            ("le cube tombé", "cube", "FALL"), ("la balle monte", "balle", "RISE"),
            ("place la balle sur la table", "balle", "PLACE"), ("place le cube sur la table", "cube", "PLACE"),
            ("pose la balle sur l'étagère", "balle", "PLACE"), ("pose le cube sur le support", "cube", "PLACE"),
            ("range la boîte sur la table", "boîte", "PLACE"), ("range le cube sur la table", "cube", "PLACE")]
    seq = [("retourne abcde", "REVERSE"), ("retourner bcdea", "REVERSE"), ("retournait cabed", "REVERSE"),
           ("retourné bdaec", "REVERSE"), ("tourne abcde", "ROTL"), ("décale abcde", "SHIFT"), ("trie dcbea", "SORT"),
           ("échange abcde", "SWAP"), ("range dcbea", "SORT")]
    return phys, seq


def observe_physics_event(text, concept):
    """Simulates the observation: the world changes according to the TRUE semantics of the concept (data generator,
    never transmitted as an answer to the system)."""
    world = make_scene()
    parsed = parse(text)
    lex = LexicalConceptMemory(); lex.ground(parsed.clauses[0].verb, concept)
    interp = build_interpretation(parsed, [concept], world)
    StructuralVerifier(lex).verify(interp, world)
    return world, getattr(interp, "branch_world", world)


def observe_sequence_event(text, concept):
    parsed = parse(text)
    return seq_decode(PRIMITIVES[SEQUENCE_CONCEPTS[concept]].apply(seq_encode(parsed.sequence).unsqueeze(0))[0])


def build_core(size="small", seed=0, bootstrap=True, fit=True, **flags):
    lc = LanguageCore(size=size, seed=seed, **flags)
    if bootstrap:
        phys, seq = bootstrap_experiences()
        for text, _, concept in phys: lc.learn_physics(text, *observe_physics_event(text, concept), label=concept)
        for text, concept in seq: lc.learn_sequence(text, observe_sequence_event(text, concept), label=concept)
    if fit: lc.fit_core()
    return lc


def language_stream():
    """Continuous stream without task label: learning steps (with observation) and queries mixed."""
    L = []
    def learn(text, concept, domain): L.append({"type": "learn", "text": text, "concept": concept, "domain": domain})
    def ask(text, category): L.append({"type": "query", "text": text, "category": category})
    ask("la balle tombe", "known"); ask("retourne abcde", "known")
    learn("le cube chute", "FALL", "physics"); ask("le bloc chute", "synonym_transfer")
    learn("la balle grimpe", "RISE", "physics"); ask("le cube grimpe", "synonym_transfer")
    learn("inverse abcde", "REVERSE", "sequence"); ask("inverse hgfab", "synonym_transfer")
    ask("retourne puis décale abcde", "zero_shot_composition"); ask("décale puis trie hgfab", "zero_shot_composition")
    ask("le bloc zorphe", "unknown"); ask("le cube montagne", "unknown_trap")
    learn("pivote abcde", "ROTR", "sequence"); learn("tourne hgfab", "ROTR", "sequence")
    ask("tourne abcde", "ambiguous")
    ask("le bloc montait", "morphology"); ask("la balle chutait", "morphology"); ask("le bloc descendait", "morphology_unseen")
    ask("le bloc tombe du support", "contradiction"); ask("monte la balle à -2 mètres", "contradiction")
    ask("range dcbea", "domain_disambiguation"); ask("range le cube sur la table", "domain_disambiguation")
    learn("la balle s'élève", "RISE", "physics"); ask("le bloc s'élève", "synonym_transfer")
    ask("la balle tombe", "return_known"); ask("retourne abcde", "return_known")
    return L


def evaluation_set():
    """Independent test queries (world reset every time)."""
    cases = []
    for e in ENTITIES:
        cases += [f"le {e} tombe" if e != "balle" and e != "boîte" else f"la {e} tombe", ]
    cases = ["la balle tombe", "le bloc tombe", "le cube tombe", "la boîte monte", "le bloc monte",
             "place la balle sur la table", "pose le bloc sur l'étagère", "monte la balle à 2 mètres",
             "retourne abcde", "décale hgfab", "trie dcbea", "échange abcde",
             "le bloc chute", "le cube grimpe", "inverse hgfab", "le bloc s'élève",
             "retourne puis décale abcde", "décale puis trie hgfab", "échange puis retourne abcgh",
             "le bloc montait", "la boîte monter", "le cube monté", "le bloc chutait", "la boîte grimpait",
             "décalait abcde", "triait hgfab", "échangé abcde", "le bloc descendait", "renverse abcde",
             "permute abcde", "ordonne hgfab", "pivoter abcde",
             "le bloc zorphe", "la balle blimpe", "trakule abcde", "le cube montagne", "triangle abcde",
             "tournesol hgfab", "la girafe tombe",
             "tourne abcde", "range dcbea", "range le cube sur la table",
             "le bloc tombe du support", "monte la balle à -2 mètres", "retourne abcdz", "place le bloc",
             "où est la balle ?", "quelle est la hauteur du cube ?",
             "la balle est sur le sol", "la balle est sur la table", "la table est en bois",
             "la balle est en verre", "le cube est magique", "la balle est rouge"]
    return cases


def oracle(text):
    """Expected answer according to the TRUE semantics of the corpus (independent of the memories of the system)."""
    world = make_scene()
    parsed = parse(text)
    if parsed.kind == "question":
        qtype, e = parsed.question
        return f"{e} est sur {support_of(world, e)}" if qtype == "where" else f"{e}: y={world.entities[e].properties['y']:g}"
    if parsed.kind == "claim":
        s, kind, v = parsed.claim
        if kind == "sur": return "VRAI" if support_of(world, s) == v else "FAUX"
        if v in world.entities[s].properties.values(): return "VRAI"
        verdict = FakeExternalVerifier().verify(s, v)
        return {"SUPPORTED": "VRAI", "REFUTED": "FAUX"}.get(verdict, "ABSTAIN")
    concepts = []
    for clause in parsed.clauses:
        truth = TRUE_CONCEPT.get(clause.verb)
        if truth is None: return "ABSTAIN"
        if isinstance(truth, set):
            if parsed.sequence and len(truth & set(SEQUENCE_CONCEPTS)) > 1: return "ABSTAIN"
            truth = next(iter(truth & (set(SEQUENCE_CONCEPTS) if parsed.sequence else PHYSICAL)))
        concepts.append(truth)
    interp = build_interpretation(parsed, concepts, world)
    lex = LexicalConceptMemory()
    for clause, c in zip(parsed.clauses, concepts): lex.ground(clause.verb, c)
    StructuralVerifier(lex).verify(interp, world)
    if interp.contradictions or interp.failures: return "REJECT"
    return interp.answer


def evaluate(system: LanguageCore, cases=None, corrupt=False):
    cases = cases or evaluation_set()
    if corrupt and system.core is not None:
        system = copy.deepcopy(system); CorruptCore(scale=1.5, seed=3).apply(system.core)
    rows = []
    for text in cases:
        gold = oracle(text)
        r = system.process(text, make_scene())
        out = r.output
        rows.append({"text": text, "gold": gold, "output": out, "decision": r.decision, "correct": out == gold,
                     "abstained": out == "ABSTAIN", "answered": out not in ("ABSTAIN",), "checks": r.checks,
                     "candidates": r.candidates_evaluated, "search_nodes": r.search_nodes, "corrected": r.corrected,
                     "memory": sorted(r.memory)})
    n = len(rows)
    answered_wrong = [r for r in rows if r["answered"] and not r["correct"] and r["output"] != "REJECT"]
    return {
        "task_success": round(sum(r["correct"] for r in rows) / n, 3),
        "false_confident_rate": round(len(answered_wrong) / n, 3),
        "wrong_rejections": sum(r["output"] == "REJECT" and r["gold"] != "REJECT" for r in rows),
        "abstention_rate": round(sum(r["abstained"] for r in rows) / n, 3),
        "correct_abstentions": sum(r["abstained"] and r["gold"] == "ABSTAIN" for r in rows),
        "unnecessary_abstentions": sum(r["abstained"] and r["gold"] != "ABSTAIN" for r in rows),
        "correction_rate": round(sum(r["corrected"] for r in rows) / n, 3),
        "verification_cost": sum(r["checks"] for r in rows),
        "routing_cost": sum(r["candidates"] for r in rows),
        "search_nodes": sum(r["search_nodes"] for r in rows),
        "parameters_total": system.core.parameter_count if system.core else 0,
        "parameters_active": system.core.parameter_count if system.core else 0,
        "rows": rows,
    }


def run_stream(system: LanguageCore):
    """Processes the stream; returns the results per category (the concept labels are
    transmitted only to the systems without WCE, which cannot explain an observation)."""
    results = []
    for event in language_stream():
        if event["type"] == "learn":
            if event["domain"] == "physics":
                learned = system.learn_physics(event["text"], *observe_physics_event(event["text"], event["concept"]), label=event["concept"])
            else:
                learned = system.learn_sequence(event["text"], observe_sequence_event(event["text"], event["concept"]), label=event["concept"])
            results.append({"type": "learn", "text": event["text"], "learned": learned})
            if system.core is not None and not system.plasticity and system.use_wce: system.fit_core(steps=150)
        else:
            gold = oracle(event["text"])
            r = system.process(event["text"], make_scene())
            results.append({"type": "query", "text": event["text"], "category": event["category"], "gold": gold,
                            "output": r.output, "decision": r.decision, "correct": r.output == gold, "reason": r.reason,
                            "memory": sorted(r.memory)})
    return results


def experiment_matrix(size="small", seed=0, monitor_factory=None):
    """The 6 systems, same learning stream, same evaluation set."""
    out = {}
    for name, flags in SYSTEMS.items():
        system = build_core(size=size, seed=seed, **flags)
        run_stream(system)
        if flags.get("geometric") and monitor_factory: system.attach_monitor(*monitor_factory(system))
        res = evaluate(system); res.pop("rows")
        stress = evaluate(system, corrupt=True); stress.pop("rows")
        out[name] = res
        out[name]["corrupted_core"] = {k: stress[k] for k in ("task_success", "false_confident_rate", "wrong_rejections", "abstention_rate",
                                                               "correction_rate", "verification_cost")}
    return out


def char_vs_word(seed=0):
    """Character vs word-tokenizer reference on forms never seen in training."""
    lc = build_core(seed=seed)
    words = list(lc.lexicon.grounded) + OTHER_WORDS
    labels = [lc.lexicon.grounded[w].most_common(1)[0][0] for w in lc.lexicon.grounded] + ["OTHER"] * len(OTHER_WORDS)
    held = [f for f, c in TRUE_CONCEPT.items() if isinstance(c, str) and f not in lc.lexicon.grounded]
    wc = WordCore(words); wc.fit(words, labels)
    char_acc = sum(lc.core.propose(f, 1)[0][0] == TRUE_CONCEPT[f] for f in held) / len(held)
    word_acc = sum(wc.predict(f) == TRUE_CONCEPT[f] for f in held) / len(held)
    lengths = [len(w) for w in words]
    return {"char": {"parameters": lc.core.parameter_count, "heldout_accuracy": round(char_acc, 3)},
            "word": {"parameters": wc.parameter_count, "heldout_accuracy": round(word_acc, 3), "vocabulary": len(wc.vocab)},
            "heldout_forms": len(held), "mean_word_length_chars": round(sum(lengths) / len(lengths), 2),
            "charset": len(CHARSET)}


def morphology_report(system: LanguageCore):
    """Cautious morphological generalisation: correct / over-generalisations / abstentions."""
    rows = []
    for form, truth in TRUE_CONCEPT.items():
        if form in system.lexicon.grounded or isinstance(truth, set): continue
        ev = system.lexicon.lookup(form)
        predicted = ev.concepts.most_common(1)[0][0] if ev.concepts and ev.source == "morphology" else None
        rows.append({"form": form, "truth": truth, "predicted": predicted, "source": ev.source, "via": ev.via})
    for form in TRAPS + NONSENSE:
        ev = system.lexicon.lookup(form)
        rows.append({"form": form, "truth": None, "predicted": ev.concepts.most_common(1)[0][0] if ev.source == "morphology" else None,
                     "source": ev.source, "via": ev.via})
    correct = sum(r["predicted"] is not None and r["predicted"] == r["truth"] for r in rows)
    over = sum(r["predicted"] is not None and r["predicted"] != r["truth"] for r in rows)
    abstain = sum(r["predicted"] is None for r in rows)
    return {"rules": {f"-{s}→-{a}": sorted(v) for (s, a), v in system.lexicon.morphology.rules.items()},
            "correct_generalizations": correct, "over_generalizations": over, "abstentions": abstain, "rows": rows}


def parameter_scaling(sizes=("tiny", "small", "medium"), seed=0, monitor_factory=None):
    """Size of the neural core vs neural accuracy vs dependence on WCE."""
    out = {}
    held = [f for f, c in TRUE_CONCEPT.items() if isinstance(c, str)]
    for size in sizes:
        full = build_core(size=size, seed=seed, **SYSTEMS["6_full_self_correction"]); run_stream(full)
        if monitor_factory: full.attach_monitor(*monitor_factory(full))
        neural = build_core(size=size, seed=seed, **SYSTEMS["1_neural_only"]); run_stream(neural)
        unseen = [f for f in held if f not in full.lexicon.grounded]
        acc_unseen = sum(full.core.propose(f, 1)[0][0] == TRUE_CONCEPT[f] for f in unseen) / len(unseen)
        e_full, e_neural = evaluate(full), evaluate(neural)
        overrides = sum(1 for r in e_full["rows"] if r["decision"] not in (ACCEPT, ACCEPT_TENTATIVE) or r["corrected"])
        out[size] = {"parameters": full.core.parameter_count, "neural_accuracy_unseen_forms": round(acc_unseen, 3),
                     "neural_only_success": e_neural["task_success"], "neural_wce_success": e_full["task_success"],
                     "neural_only_false_confident": e_neural["false_confident_rate"],
                     "wce_gain": round(e_full["task_success"] - e_neural["task_success"], 3),
                     "wce_interventions": overrides, "verification_cost": e_full["verification_cost"]}
    return out

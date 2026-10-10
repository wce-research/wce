"""Discourse and identity memory: mentions -> discourse referents -> world entities.

TEXT -> Mentions -> DiscourseState (referents, focus, history) -> cross-sentence constraints
-> joint resolution (constraint satisfaction, vectorized) -> EntityIdentityGraph -> revisions.

Identity ≠ state: a referent keeps its identity when its mutable properties (position) change.
No identity is forced: below 0.9 probability, the hypotheses stay open (weighted branches).
Declared innate: determiners (definite / indefinite / demonstrative), « autre », pronouns, numbers « deux/trois »,
two discourse state verbs (« tombe », « reste »). The rest (nouns, attributes, relations) comes from the learned lexicon.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from core.independent_agent import token_class, tokenize
from core.plasticity import ConfigurationAmplitude

DEF, INDEF, DEM = {"le", "la", "l'", "les"}, {"un", "une"}, {"ce", "cet", "cette"}
PRONOUNS = {"il": "m", "elle": "f", "-le": "m", "-la": "f", "celui-ci": "m", "celle-ci": "f"}
NUMBERS = {"deux": 2, "trois": 3}
EVENTS = {"tombe": "fall", "reste": "stay"}
PLACE_TYPES = {"table", "shelf", "floor", "list"}
COMMIT = 0.9
AMBIGUOUS, NONEXISTENT, HYPOTHETICAL = "AMBIGUOUS", "NONEXISTENT", "HYPOTHETICAL"


# ================================================================ representations

@dataclass
class Mention:
    id: str
    surface: str
    head: str
    kind: str                         # np | pron | other | adj | plural | place
    definiteness: Optional[str] = None
    type: Optional[str] = None
    attrs: Dict[str, str] = field(default_factory=dict)
    relation: Optional[Tuple[str, str, str]] = None     # (rel, place, tense)
    role: str = "object"
    sentence: int = 0
    position: int = 0
    gender: Optional[str] = None
    number: int = 1
    referent: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)   # competing referents (ambiguous pronoun)
    candidates: Dict[str, float] = field(default_factory=dict)


@dataclass
class Referent:
    id: str
    type: Optional[str]
    gender: Optional[str]
    introduced: str
    attrs: Dict[str, str] = field(default_factory=dict)          # asserted properties (stable, identity-bearing)
    states: List[Tuple[int, str, Tuple[str, str]]] = field(default_factory=list)  # (sentence, tense, (rel, place)): mutable state
    events: List[Tuple[int, str]] = field(default_factory=list)
    different_from: Set[str] = field(default_factory=set)
    mentions: List[str] = field(default_factory=list)
    group: Optional[str] = None
    hypothetical: bool = False
    last_sentence: int = 0
    frozen: Optional[str] = None      # consolidated identity (memory): excluded from the joint search, reversible


@dataclass
class Revision:
    referent: str
    before: Any
    after: Any
    confidence_before: float
    confidence_after: float
    sentence: int
    reason: str


@dataclass
class HypotheticalEntity:
    referent: str
    description: Dict[str, Any]
    support: int = 1


@dataclass
class DiscourseState:
    entities_mentioned: Dict[str, Referent] = field(default_factory=dict)
    mentions: Dict[str, Mention] = field(default_factory=dict)
    history: List[List[str]] = field(default_factory=list)
    focus: Optional[str] = None
    recent: List[str] = field(default_factory=list)
    unresolved: Set[str] = field(default_factory=set)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)
    groups: Dict[str, List[str]] = field(default_factory=dict)
    time: int = 0
    confidence: Dict[str, float] = field(default_factory=dict)


class EntityIdentityGraph:
    """Nodes: mentions, referents, world entities. Edges: SAME_AS, POSSIBLY_SAME_AS, DIFFERENT_FROM, REFERS_TO, CONTRADICTS."""

    def __init__(self):
        self.edges: List[Tuple[str, str, str, float, int]] = []   # (type, a, b, confidence, sentence)

    def add(self, kind, a, b, conf=1.0, sentence=0): self.edges.append((kind, a, b, conf, sentence))

    def reset_world_edges(self):
        self.edges = [e for e in self.edges if e[0] == "REFERS_TO" and e[1].startswith("m")]

    def of(self, kind): return [e for e in self.edges if e[0] == kind]

    def same_as_classes(self):
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for _, a, b, _, _ in self.of("SAME_AS"): parent[find(a)] = find(b)
        classes = defaultdict(set)
        for x in list(parent): classes[find(x)].add(x)
        return [c for c in classes.values() if len(c) > 1]

    def invariant_violations(self, state: DiscourseState, resolution):
        """SAME_AS transitivity, SAME_AS/DIFFERENT_FROM incompatibility, uniqueness at the same instant, exclusive attributes."""
        v = []
        classes = self.same_as_classes()
        cls_of = {x: i for i, c in enumerate(classes) for x in c}
        for _, a, b, _, _ in self.of("DIFFERENT_FROM"):
            if a in cls_of and cls_of.get(a) == cls_of.get(b): v.append(("same_and_different", a, b))
            ea, eb = resolution.get(a), resolution.get(b)
            if ea and ea == eb and not ea.startswith(("AMB", "HYP", "NON")): v.append(("different_referents_same_entity", a, b))
        for c in classes:
            attrs = defaultdict(set)
            for r in c:
                ref = state.entities_mentioned.get(r)
                if ref:
                    for k, val in ref.attrs.items(): attrs[k].add(val)
            if any(len(s) > 1 for s in attrs.values()): v.append(("exclusive_attributes", tuple(sorted(c))))
        return v


# ================================================================ resolver

class DiscourseResolver:
    def __init__(self, lexicon_agent, given: List[Tuple[Tuple, Any]] = ()):
        self.lx = lexicon_agent
        self.state = DiscourseState()
        self.graph = EntityIdentityGraph()
        self.world: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.revisions: List[Revision] = []
        self.hypotheticals: Dict[str, HypotheticalEntity] = {}
        self.episodic: List[str] = []
        self.resolution: Dict[str, str] = {}
        self.dists: Dict[str, Dict[str, float]] = {}
        self.checks = 0
        self.contradictions = 0
        self.clarifications = 0
        self._n_mentions = 0
        self._n_refs = 0
        self.predicates: List[Tuple] = []     # (sentence, referent, kind, value, tense): asserted content, not descriptive
        self.predicate_enabled = True
        self.predicate_attrs_identity = True  # False (unified state): ASSERTED attribute / state = weighted evidence, not a hard constraint
        for key, value in given: self.add_fact(key, value)

    # ---- world knowledge (data) ------------------------------------------
    def add_fact(self, key, value):
        if key[0] == "type": self.world[key[1]]["type"] = value
        elif key[0] == "attr": self.world[key[1]]["attr"] = value
        elif key[0] == "loc": self.world[key[1]]["loc0" if key[2] == 0 else "loc"] = value

    def entities_of(self, type_):
        return [e for e, f in self.world.items() if f.get("type") not in PLACE_TYPES and (type_ is None or f.get("type") == type_)]

    def place_id(self, type_, attr=None):
        cands = [e for e, f in self.world.items() if f.get("type") == type_ and (attr is None or f.get("attr") in (None, attr))]
        return cands[0] if len(cands) == 1 else (type_ if type_ in PLACE_TYPES else None)

    # ---- lexicon -------------------------------------------------------------------
    def meaning(self, tok):
        m = self.lx.word_meaning(tok)[0]
        if m is None and tok.endswith("s") and len(tok) > 3:
            base = self.lx.word_meaning(tok[:-1])[0]
            if base and base.startswith("type:"): return base
        if m is None and tok in ("fils",): return self.lx.word_meaning("fils")[0]
        return m

    def gender_of(self, head):
        g = self.lx.grammar.gender_of(head)
        return g

    # ---- mention extraction ---------------------------------------------------------
    def extract(self, tokens, sentence):
        meanings = [self.meaning(t) for t in tokens]
        cls = [token_class(m) for m in meanings]
        mentions, i = [], 0
        def new(**kw):
            self._n_mentions += 1
            mm = Mention(f"m{self._n_mentions}", sentence=sentence, position=kw.pop("position"), **kw)
            mentions.append(mm); return mm
        while i < len(tokens):
            t = tokens[i]
            if t in PRONOUNS:
                new(surface=t, head=t, kind="pron", gender=PRONOUNS[t], position=i); i += 1; continue
            if t in NUMBERS and i + 1 < len(tokens) and cls[i + 1] == "N":
                new(surface=f"{t} {tokens[i + 1]}", head=tokens[i + 1], kind="plural", type=meanings[i + 1].split(":")[1],
                    number=NUMBERS[t], gender=self.gender_of(tokens[i + 1].rstrip("s")), position=i); i += 2; continue
            if t in DEF | INDEF | DEM and i + 1 < len(tokens):
                d = "def" if t in DEF else "indef" if t in INDEF else "dem"
                j = i + 1
                if tokens[j] == "autre":
                    new(surface=f"{t} autre", head="autre", kind="other", definiteness=d, position=j); i = j + 1; continue
                if cls[j] == "N":
                    typ = meanings[j].split(":")[1]
                    k = j + 1; attrs = {}
                    while k < len(tokens) and cls[k] == "C": attrs["attr"] = meanings[k].split(":")[1]; k += 1
                    kind = "place" if typ in PLACE_TYPES else "np"
                    mm = new(surface=" ".join(tokens[i:k]), head=tokens[j], kind=kind, definiteness=d, type=typ, attrs=attrs,
                             gender=self.gender_of(tokens[j]), position=j)
                    # relational modifier: « le cube sur la table », « le cube qui était sur la table »
                    tense = "now"
                    if k < len(tokens) and tokens[k] == "qui" and k + 1 < len(tokens) and tokens[k + 1] in ("était", "est"):
                        tense = "past" if tokens[k + 1] == "était" else "now"; k += 2
                    if kind == "np" and k < len(tokens) and cls[k] == "R":
                        place, k2 = self._place_after(tokens, meanings, cls, k + 1)
                        if place: mm.relation = (meanings[k].split(":")[1], place, tense); k = k2
                    i = k; continue
                if cls[j] == "C":
                    new(surface=f"{t} {tokens[j]}", head=tokens[j], kind="adj", definiteness=d, attrs={"attr": meanings[j].split(":")[1]},
                        gender="f" if t == "la" else "m", position=j); i = j + 1; continue
            i += 1
        return mentions, meanings, cls

    def _place_after(self, tokens, meanings, cls, k):
        while k < len(tokens) and tokens[k] in DEF | INDEF: k += 1
        if k < len(tokens) and cls[k] == "N":
            typ = meanings[k].split(":")[1]; attr = None; k2 = k + 1
            while k2 < len(tokens) and cls[k2] == "C": attr = meanings[k2].split(":")[1]; k2 += 1
            if typ in PLACE_TYPES: return self.place_id(typ, attr), k2
            return ("entity", typ, attr), k2
        return None, k

    # ---- discourse referents ------------------------------------------------------------
    def _new_referent(self, mention, introduced):
        self._n_refs += 1
        r = Referent(f"d{self._n_refs}", mention.type, mention.gender, introduced, dict(mention.attrs), last_sentence=mention.sentence)
        self.state.entities_mentioned[r.id] = r
        return r

    def _salience(self, r, sentence):
        age = sentence - r.last_sentence
        return 1.0 / (1 + age) + (0.6 if r.id == self.state.focus else 0.0)

    def assign(self, mention):
        refs = self.state.entities_mentioned
        s = mention.sentence
        if mention.kind == "pron":
            cands = [r for r in refs.values() if (r.gender in (mention.gender, None)) and not r.hypothetical or r.gender == mention.gender]
            cands = [r for r in cands if r.gender in (mention.gender, None)]
            if not cands: return None
            ranked = sorted(cands, key=lambda r: -self._salience(r, s))
            if len(ranked) > 1 and self._salience(ranked[0], s) < 1.5 * self._salience(ranked[1], s):
                mention.alternatives = [ranked[0].id, ranked[1].id]
            return ranked[0]
        if mention.kind == "other":
            focus = refs.get(self.state.focus)
            if focus is None: return None
            mention.type = focus.type
            group = self.state.groups.get(focus.group) if focus.group else None
            # « l'autre » = a referent KNOWN to be distinct from the focus (not merely another referent: it could corefer)
            known_distinct = lambda r: r.id in focus.different_from or (
                self.resolution.get(r.id, AMBIGUOUS) not in (AMBIGUOUS, NONEXISTENT, HYPOTHETICAL)
                and self.resolution.get(focus.id, AMBIGUOUS) not in (AMBIGUOUS, NONEXISTENT, HYPOTHETICAL)
                and self.resolution[r.id] != self.resolution[focus.id])
            pool = [refs[x] for x in group if x != focus.id] if group else \
                   [r for r in refs.values() if r.type == focus.type and r.id != focus.id and known_distinct(r)]
            if len(pool) == 1: return pool[0]
            r = self._new_referent(mention, "other"); r.type = focus.type; r.gender = focus.gender
            r.different_from.add(focus.id); focus.different_from.add(r.id)
            return r
        if mention.kind == "adj":
            typed = sorted([r for r in refs.values() if r.type], key=lambda r: -r.last_sentence)
            if not typed:                                    # ellipsis without antecedent: « le rouge » = the red object of the world
                exact = [r for r in refs.values() if r.type is None and r.attrs.get("attr") == mention.attrs.get("attr")
                         and r.gender in (None, mention.gender)]
                return exact[0] if exact else self._new_referent(mention, "def")
            mention.type = typed[0].type
            if typed[0].group:
                members = [refs[x] for x in self.state.groups[typed[0].group]]
                same = [r for r in members if r.attrs.get("attr") == mention.attrs["attr"]]
                if same: return same[0]
                free = [r for r in members if "attr" not in r.attrs]
                if free: free[0].attrs.update(mention.attrs); return free[0]
        if mention.kind in ("np", "adj"):
            if mention.definiteness == "indef":
                return self._new_referent(mention, "indef")
            same_type = [r for r in refs.values() if r.type == mention.type]
            if mention.attrs:
                exact = [r for r in same_type if r.attrs.get("attr") == mention.attrs.get("attr")]
                if len(exact) == 1: return exact[0]
                return self._new_referent(mention, "def")
            if mention.relation is None and len(same_type) == 1: return same_type[0]
            return self._new_referent(mention, "def")
        return None

    # ---- speaking turn ---------------------------------------------------------------------
    def process(self, text):
        s = self.state.time = self.state.time + 1
        tokens = tokenize(text)
        self.episodic.append(text)
        mentions, meanings, cls = self.extract(tokens, s)
        tracked = []
        for mm in mentions:
            self.state.mentions[mm.id] = mm
            if mm.kind == "place": continue
            if mm.kind == "plural":
                gid = f"g{mm.id}"; members = []
                for _ in range(mm.number):
                    r = self._new_referent(Mention(mm.id, mm.surface, mm.head, "np", type=mm.type, gender=mm.gender, sentence=s), "plural")
                    r.group = gid; members.append(r.id)
                for a in members:
                    for b in members:
                        if a != b: self.state.entities_mentioned[a].different_from.add(b)
                self.state.groups[gid] = members
                self.state.focus = members[0]
                continue
            r = self.assign(mm)
            if r is None: self.state.unresolved.add(mm.id); tracked.append(mm); continue
            mm.referent = r.id; r.mentions.append(mm.id); r.last_sentence = s
            if mm.attrs and "attr" not in r.attrs: r.attrs.update(mm.attrs)
            if mm.relation: r.states.append((s, mm.relation[2], mm.relation[:2]))
            self.graph.add("REFERS_TO", mm.id, r.id, 1.0 if not mm.alternatives else 0.5, s)
            tracked.append(mm)
        subject = next((mm for mm in tracked if mm.referent), None)
        if subject is not None:
            r = self.state.entities_mentioned[subject.referent]
            if self.predicate_enabled: self._predicate(tokens, meanings, cls, subject, r, s)
            self.state.focus = r.id
            self.state.recent = ([r.id] + [x for x in self.state.recent if x != r.id])[:5]
        self.state.history.append([mm.id for mm in tracked])
        self.solve(s, reason=f"phrase {s}")
        return tracked

    def _predicate(self, tokens, meanings, cls, subject, r, s):
        start = subject.position + 1
        for i in range(start, len(tokens)):
            t = tokens[i]
            if t in EVENTS:
                r.events.append((s, EVENTS[t]))
                self.predicates.append((s, r.id, "event", EVENTS[t], "now"))
                if EVENTS[t] == "fall" and self.predicate_attrs_identity: r.states.append((s, "after", ("on", "floor")))
                if EVENTS[t] == "stay" and i + 1 < len(tokens) and cls[i + 1] == "R":
                    place, _ = self._place_after(tokens, meanings, cls, i + 2)
                    if isinstance(place, str):
                        if self.predicate_attrs_identity: r.states.append((s, "now", (meanings[i + 1].split(":")[1], place)))
                        self.predicates.append((s, r.id, "loc", (meanings[i + 1].split(":")[1], place), "now"))
                return
            if t in ("est", "était"):
                if i + 1 < len(tokens) and cls[i + 1] == "C":
                    if self.predicate_attrs_identity: r.attrs.setdefault("attr", meanings[i + 1].split(":")[1])
                    self.predicates.append((s, r.id, "attr", meanings[i + 1].split(":")[1], "now")); return
                if i + 1 < len(tokens) and cls[i + 1] == "R":
                    place, _ = self._place_after(tokens, meanings, cls, i + 2)
                    if isinstance(place, str):
                        tense = "past" if t == "était" else "now"
                        if self.predicate_attrs_identity: r.states.append((s, tense, (meanings[i + 1].split(":")[1], place)))
                        self.predicates.append((s, r.id, "loc", (meanings[i + 1].split(":")[1], place), tense))
                return

    # ---- joint resolution (vectorized constraint satisfaction) --------------------------
    def _compatible(self, r: Referent, e: str):
        f = self.world[e]
        if r.type and f.get("type") != r.type: return False
        if "attr" in r.attrs and f.get("attr") is not None and f["attr"] != r.attrs["attr"]: return False
        fell_before = lambda s: any(es < s and kind == "fall" for es, kind in r.events)
        for s, tense, (rel, place) in r.states:
            if isinstance(place, tuple):                       # relation to another described entity (e.g. box)
                _, ptype, pattr = place
                targets = [x for x, fx in self.world.items() if fx.get("type") == ptype and (pattr is None or fx.get("attr") in (None, pattr))]
                known = f.get("loc")
                if known and (known[0] != rel or known[1] not in targets): return False
                continue
            if tense == "past":
                if f.get("loc0") and f["loc0"] != (rel, place): return False
            elif tense == "now" and not fell_before(s):
                if f.get("loc") and f["loc"] != (rel, place) and f.get("loc_time", 0) <= s: return False   # more recent state: may have moved
        return True

    def solve(self, sentence, reason=""):
        refs = [r for r in self.state.entities_mentioned.values()]
        # asserted exclusive properties -> distinct referents
        for a, b in itertools.combinations(refs, 2):
            if a.type == b.type and "attr" in a.attrs and "attr" in b.attrs and a.attrs["attr"] != b.attrs["attr"]:
                a.different_from.add(b.id); b.different_from.add(a.id)
        cands = {}
        frozen = {r.id: r.frozen for r in refs if r.frozen}
        for r in refs:
            if r.frozen: cands[r.id] = [r.frozen]; continue
            excluded = {frozen[o] for o in r.different_from if o in frozen}
            c = [e for e in self.entities_of(r.type) if e not in excluded and self._compatible(r, e)]
            self.checks += len(self.entities_of(r.type))
            cands[r.id] = c
        new_dists = {rid: {e: 1.0} for rid, e in frozen.items()}
        free = [r for r in refs if len(cands[r.id]) >= 1 and not r.frozen]
        idx = {r.id: i for i, r in enumerate(free)}
        if free:
            lists = [torch.tensor([hash(e) for e in cands[r.id]]) for r in free]
            names = [cands[r.id] for r in free]
            sizes = [len(n) for n in names]
            total = math.prod(sizes)
            if total <= 200000:
                grids = torch.cartesian_prod(*[torch.arange(n) for n in sizes]) if len(sizes) > 1 else torch.arange(sizes[0]).view(-1, 1)
                grids = grids.view(-1, len(sizes))
                # chosen entities (global indices) to test DIFFERENT_FROM
                universe = sorted({e for n in names for e in n}); uid = {e: i for i, e in enumerate(universe)}
                maps = [torch.tensor([uid[e] for e in n]) for n in names]
                chosen = torch.stack([maps[j][grids[:, j]] for j in range(len(names))], 1)
                valid = torch.ones(len(grids), dtype=torch.bool)
                for r in free:
                    for other in r.different_from:
                        if other in idx and idx[other] > idx[r.id]:
                            valid &= chosen[:, idx[r.id]] != chosen[:, idx[other]]
                self.checks += int(len(grids))
                n_valid = int(valid.sum())
                if n_valid == 0:
                    self.contradictions += 1
                    for r in free: self.graph.add("CONTRADICTS", r.id, "world", 1.0, sentence)
                    valid = torch.ones(len(grids), dtype=torch.bool); n_valid = len(grids)
                for r in free:
                    j = idx[r.id]
                    counts = torch.bincount(grids[valid, j], minlength=len(names[j])).float() / n_valid
                    new_dists[r.id] = {e: float(p) for e, p in zip(names[j], counts)}
            else:
                for r in free: new_dists[r.id] = {e: 1 / len(cands[r.id]) for e in cands[r.id]}
        for r in refs:
            if not cands[r.id]:
                if r.introduced in ("indef", "plural"):
                    r.hypothetical = True
                    self.hypotheticals.setdefault(r.id, HypotheticalEntity(r.id, {"type": r.type, **r.attrs}))
                    new_dists[r.id] = {HYPOTHETICAL: 1.0}
                else:
                    new_dists[r.id] = {NONEXISTENT: 1.0}
        # revisions (never silent)
        for rid, dist in new_dists.items():
            before = self.resolution.get(rid)
            after, conf = self._decide(dist)
            prev_conf = self.dists.get(rid, {}).get(before, 0.0) if before else 0.0
            if before is not None and before != after:
                self.revisions.append(Revision(rid, before, after, prev_conf, conf, sentence, reason))
            self.resolution[rid] = after
        self.dists = new_dists
        self._rebuild_edges(sentence)

    @staticmethod
    def _decide(dist):
        e, p = max(dist.items(), key=lambda t: t[1])
        return (e if p >= COMMIT else AMBIGUOUS), p

    def _rebuild_edges(self, sentence):
        self.graph.reset_world_edges()
        refs = self.state.entities_mentioned
        for r in refs.values():
            for o in r.different_from:
                if r.id < o: self.graph.add("DIFFERENT_FROM", r.id, o, 1.0, sentence)
            for e, p in self.dists.get(r.id, {}).items():
                self.graph.add("REFERS_TO" if p >= COMMIT else "POSSIBLY_SAME_AS", r.id, e, p, sentence)
        by_entity = defaultdict(list)
        for rid, e in self.resolution.items():
            if e not in (AMBIGUOUS, NONEXISTENT, HYPOTHETICAL): by_entity[e].append(rid)
        for e, rs in by_entity.items():
            for a, b in zip(rs, rs[1:]): self.graph.add("SAME_AS", a, b, 1.0, sentence)

    # ---- querying ---------------------------------------------------------------------------
    def mention_resolution(self, mention_id):
        mm = self.state.mentions[mention_id]
        if mm.referent is None: return AMBIGUOUS if mm.kind == "pron" else NONEXISTENT
        refs = [mm.referent] + [a for a in mm.alternatives if a != mm.referent]
        mix = defaultdict(float)
        for rid in refs:
            for e, p in self.dists.get(rid, {}).items(): mix[e] += p / len(refs)
        if not mix: return AMBIGUOUS
        e, p = max(mix.items(), key=lambda t: t[1])
        mm.candidates = dict(mix)
        return e if p >= COMMIT or e in (NONEXISTENT, HYPOTHETICAL) else AMBIGUOUS

    def branches(self, referent_id):
        """Maintained identity hypotheses (weighted classical representation, without arbitrary choice)."""
        return [ConfigurationAmplitude(e, None, p) for e, p in sorted(self.dists.get(referent_id, {}).items(), key=lambda t: -t[1])]

    def presupposition(self, mention_id):
        """Distinguishes existence / property / relation for a presupposition question."""
        mm = self.state.mentions[mention_id]
        r = self.state.entities_mentioned.get(mm.referent) if mm.referent else None
        exists = bool(r) and self.resolution.get(r.id) not in (NONEXISTENT,)
        return {"entity_exists": exists, "property_supported": exists and bool(mm.attrs), "relation_supported": exists and mm.relation is not None}

    # ---- clarification and active identity learning ---------------------------------------
    def clarification_question(self, mention_id):
        mm = self.state.mentions[mention_id]
        rid = mm.referent
        options = [e for e, p in self.dists.get(rid, {}).items() if p > 0]
        descr = [self.world[e].get("attr") for e in options]
        if len(options) < 2 or any(d is None for d in descr): return None
        self.clarifications += 1
        gender = mm.gender or "m"
        words = {}
        for w, sym in self.lx.lex.lexicon().items():
            if token_class(sym) == "C":
                v = sym.split(":")[1]
                g = self.gender_of(w)
                if v not in words or (w.endswith("e") == (gender == "f")): words[v] = w
        det_ = "la" if gender == "f" else "le"
        return f"Quel{'le' if gender == 'f' else ''} {mm.head} ? " + " ou ".join(f"{det_} {words.get(d, d)}" for d in descr) + " ?"

    def apply_clarification(self, mention_id, answer_text):
        mm = self.state.mentions[mention_id]
        r = self.state.entities_mentioned[mm.referent]
        for t in tokenize(answer_text):
            m = self.meaning(t)
            if m and token_class(m) == "C": r.attrs["attr"] = m.split(":")[1]
        self.solve(self.state.time, reason="clarification")

    def identity_entropy(self, rid, dists=None):
        dist = (dists or self.dists).get(rid, {})
        return -sum(p * math.log(p) for p in dist.values() if p > 0)

    def best_information_request(self, rid, attr_domain, places=None):
        """Information gain: which revelation (attribute OR position of a candidate entity) reduces the
        identity entropy the most? IG = H(before) − E[H(after)], equiprobable outcomes."""
        r = self.state.entities_mentioned[rid]
        before = self.identity_entropy(rid)
        places = places or [e for e, f in self.world.items() if f.get("type") in PLACE_TYPES]
        best, best_gain = None, 0.0
        for e in self.dists.get(rid, {}):
            if e in (AMBIGUOUS, NONEXISTENT, HYPOTHETICAL): continue
            options = []
            if self.world[e].get("attr") is None: options.append(("attr", list(attr_domain)))
            if self.world[e].get("loc") is None and places: options.append(("loc", [("on" if self.world[p].get("type") != "list" else "in", p) for p in places]))
            for slot, outcomes in options:
                expected = 0.0
                for v in outcomes:
                    saved = self.world[e].get(slot); self.world[e][slot] = v
                    c = [x for x in self.entities_of(r.type) if self._compatible(r, x)]
                    self.world[e][slot] = saved
                    expected += (math.log(len(c)) if c else 0.0) / len(outcomes)
                gain = before - expected
                if gain > best_gain + 1e-9: best, best_gain = ((slot, e) if slot == "attr" else ("loc", e, "now")), gain
        return best, round(best_gain, 3)

    # ---- consolidated memory ---------------------------------------------------------------------------
    def consolidated(self):
        """Compressed discourse: entities, facts, identities, unresolved references — without the raw text."""
        records = defaultdict(lambda: {"mentions": 0, "attrs": {}, "states": [], "referents": []})
        unresolved = []
        for rid, r in self.state.entities_mentioned.items():
            e = self.resolution.get(rid, AMBIGUOUS)
            if e in (AMBIGUOUS, NONEXISTENT, HYPOTHETICAL): unresolved.append((rid, e, {k: round(v, 2) for k, v in self.dists.get(rid, {}).items()})); continue
            rec = records[e]; rec["mentions"] += len(r.mentions); rec["attrs"].update(r.attrs); rec["states"] += r.states[-2:]; rec["referents"].append(rid)
        return {"entities": dict(records), "unresolved": unresolved, "revisions": len(self.revisions)}

    def memory_size(self):
        c = self.consolidated()
        compressed = sum(2 + len(v["attrs"]) + len(v["states"]) for v in c["entities"].values()) + 3 * len(c["unresolved"])
        raw = sum(len(tokenize(t)) for t in self.episodic)
        return {"raw_tokens": raw, "consolidated_fields": compressed, "ratio": round(compressed / max(1, raw), 3)}

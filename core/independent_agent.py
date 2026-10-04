"""Agent WCE dans l'environnement indépendant.

L'agent ne voit QUE : des légendes (phrase + structure perçue) pour apprendre, des observations
de sources (structurées ou textuelles), et des requêtes. Il n'importe PAS le générateur de langue.

Inné (déclaré) : opérateurs interrogatifs, pronoms, « si », « qui », « et », verbes d'attention
(« regarde »), nom générique « objet », connaissance structurelle des affordances (seule une boîte
contient). Appris : lexique, rôles, prépositions, connecteurs d'ordre, unités, genres, morphologie.
"""
from __future__ import annotations

import copy
import itertools
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from core.language_core import CharCore, MorphologyLearner, TrainCore, encode_words
from core.planner import TransformationPlanner
from core.reasoning import Goal
from core.transformation import Change, RelationChange, Transformation
from core.transformations import DeclarativeTransformationSpec, TransformationDomain
from core.world import Entity, World

INNATE_QUESTION = {"où", "se", "trouve", "quelle", "quel", "couleur", "pourquoi", "?", "-elle", "-il", "est-ce"}
INNATE_PRONOUN = {"-la": "f", "-le": "m", "elle": "f", "il": "m", "celle-ci": "f", "celui-ci": "m"}
INNATE_OTHER = {"si", "qui", "et", "regarde", "voici", ".", ",", "objet", "l'objet"}
SUPPORTS = {"table": 1.0, "shelf": 2.0, "floor": 0.0}
KIND_DOMAIN = {"color": 4, "loc": 12, "height": 6, "owner": 3, "event": 12}
ABSTAIN = {"UNKNOWN", "UNKNOWN_WORD", "ASK"}
CONFLICTED = "CONFLICTED"          # deux valeurs soutenues par des sources fiables, aucune discriminante


# ================================================================ tokenisation propre à l'agent

def tokenize(text):
    t = text.lower().replace("’", "'")
    t = re.sub(r"(\d),(\d)", r"\1#\2", t)
    t = re.sub(r"([?.,;:!])", r" \1 ", t)
    t = t.replace("#", ",")
    out = []
    for w in t.split():
        m = re.match(r"^(\w+)-(la|le|elle|il)$", w)
        if m: out += [m.group(1), "-" + m.group(2)]; continue
        m = re.match(r"^(l|d|qu|c)'(.+)$", w)
        if m: out += [m.group(1) + "'", m.group(2)]; continue
        out.append(w)
    return out


def is_number(tok): return re.fullmatch(r"\d+(,\d+)?", tok) is not None
def to_float(tok): return float(tok.replace(",", "."))


# ================================================================ induction lexicale (trans-situationnelle)

def structure_symbols(s):
    out = set()
    def desc(d):
        if not d: return
        for k in ("type", "color", "person"):
            if k in d: out.add(f"{k}:{d[k]}")
    kind = s["kind"]
    out.add(f"kind:{kind}")
    if kind == "sequence":
        for e in s["events"]: out |= structure_symbols(e)
        return out
    for slot in ("subject", "agent", "patient", "target"): desc(s.get(slot))
    if "rel" in s: out.add(f"rel:{s['rel']}")
    if "action" in s: out.add(f"act:{s['action']}")
    if s.get("time") == "past": out.add("time:past")
    if kind == "color": out.add(f"color:{s['value']}")
    if kind == "owner": out.add(f"person:{s['value']}")
    if kind == "count": out.add(f"num:{s['value']}")
    return out


class LexiconInducer:
    def __init__(self, min_count=3, min_precision=0.97):
        self.tok = Counter(); self.sym = Counter(); self.co = defaultdict(Counter); self.n = 0
        self.units = defaultdict(Counter)
        self.min_count, self.min_precision = min_count, min_precision
        self._cache = {}

    def observe(self, tokens, structure):
        self.n += 1; self._cache = {}
        symbols = structure_symbols(structure)
        for t in set(tokens):
            self.tok[t] += 1
            for s in symbols: self.co[t][s] += 1
        for s in symbols: self.sym[s] += 1
        if structure["kind"] == "height" and structure["value"] > 0:
            for i, t in enumerate(tokens[:-1]):
                if is_number(t) and to_float(t) > 0:
                    self.units[tokens[i + 1]][round(structure["value"] / to_float(t), 4)] += 1

    def meaning(self, token):
        if token in self._cache: return self._cache[token]
        result = None
        # a priori déclaré : les jetons très courts ou élidés sont grammaticaux (jamais porteurs de sens)
        if self.tok[token] >= self.min_count and len(token) > 2 and not token.endswith("'"):
            scored = [(c / self.tok[token], c / self.sym[s], s) for s, c in self.co[token].items()]
            if scored:
                p, rev, s = max(scored)
                if p >= self.min_precision: result = s
        self._cache[token] = result
        return result

    def seen(self, token): return self.tok[token] > 0
    def unit_scale(self, token):
        return self.units[token].most_common(1)[0][0] if self.units.get(token) else None

    def lexicon(self):
        return {t: self.meaning(t) for t in self.tok if self.meaning(t)}


def token_class(meaning):
    if meaning is None: return "F"
    prefix = meaning.split(":")[0]
    return {"type": "N", "color": "C", "person": "P", "rel": "R", "act": "A", "kind": "K", "time": "T", "num": "Q"}.get(prefix, "F")


# ================================================================ syntagmes et rôles appris

@dataclass
class NP:
    start: int
    end: int
    type: Optional[str] = None
    color: Optional[str] = None
    person: Optional[str] = None
    pronoun: Optional[str] = None
    generic: bool = False
    relative: Optional[Tuple[str, str]] = None
    head: Optional[str] = None

    @property
    def desc(self): return {k: v for k, v in (("type", self.type), ("color", self.color), ("person", self.person)) if v}


ROLES = {"loc": ("subject", "target"), "event": ("agent", "patient", "target"), "color": ("subject",), "height": ("subject",),
         "owner": ("subject", "value"), "count": ("target",), "command": ("patient", "target")}


class GrammarInducer:
    """Rôles induits : P(rôle | mot précédent), P(rôle | classe précédente), P(rôle | rang), par type de phrase.
    Décodage conjoint (rôles distincts). Aucune règle de grammaire écrite à la main."""

    def __init__(self, lexicon: LexiconInducer):
        self.lex = lexicon
        self.counts = defaultdict(Counter)
        self.determiners = Counter(); self.before_np = Counter()
        self.connectors = defaultdict(Counter)
        self.gender = defaultdict(Counter)
        self.class_after = defaultdict(Counter)
        self.examples = 0

    def meanings(self, tokens): return [self.lex.meaning(t) for t in tokens]

    def chunks(self, tokens, meanings=None):
        meanings = meanings or self.meanings(tokens)
        nps, i = [], 0
        while i < len(tokens):
            m = meanings[i]
            if tokens[i] in INNATE_PRONOUN:
                nps.append(NP(i, i + 1, pronoun=INNATE_PRONOUN[tokens[i]])); i += 1; continue
            if tokens[i] in ("objet", "l'objet"):
                nps.append(NP(i, i + 1, generic=True)); i += 1; continue
            if token_class(m) in ("N", "P"):
                np = NP(i, i + 1, head=tokens[i])
                if token_class(m) == "N": np.type = m.split(":")[1]
                else: np.person = m.split(":")[1]
                j = i + 1
                while j < len(tokens) and token_class(meanings[j]) == "C" and np.type:
                    np.color = meanings[j].split(":")[1]; j += 1
                np.end = j; nps.append(np); i = j; continue
            i += 1
        return nps

    def _prev(self, tokens, np):
        j = np.start - 1
        while j >= 0 and self._is_determiner(tokens[j]): j -= 1
        return tokens[j] if j >= 0 else "<s>"

    def _is_determiner(self, tok):
        return self.before_np[tok] >= 3 and self.before_np[tok] / max(1, self.lex.tok[tok]) >= 0.5 and self.lex.meaning(tok) is None

    def features(self, tokens, np, rank, kind):
        prev = self._prev(tokens, np)
        return [("prev", kind, prev), ("pclass", kind, token_class(self.lex.meaning(prev))), ("rank", kind, rank)]

    def learn(self, tokens, structure):
        kind = structure["kind"]
        for a, b in zip(["<s>"] + tokens, tokens):
            self.class_after[a][token_class(self.lex.meaning(b))] += 1
        if kind == "sequence": return self._learn_sequence(tokens, structure)
        nps = self.chunks(tokens)
        for np in nps:
            if np.start > 0: self.before_np[tokens[np.start - 1]] += 1
            if np.start > 0 and np.head: self.gender[np.head][tokens[np.start - 1]] += 1
        slots = {r: structure.get(r) for r in ROLES.get(kind, ()) if isinstance(structure.get(r), dict)}
        if kind == "owner": slots["value"] = {"person": structure["value"]}
        gold = {}
        for i, np in enumerate(nps):
            matches = [r for r, d in slots.items() if d and all(d.get(k) == v for k, v in np.desc.items()) and np.desc]
            if len(matches) == 1: gold[i] = matches[0]
        if not gold: return
        self.examples += 1
        for i, role in gold.items():
            for f in self.features(tokens, nps[i], i, kind): self.counts[f][role] += 1

    def _learn_sequence(self, tokens, structure):
        for i, t in enumerate(tokens):
            if self.lex.meaning(t) == "kind:sequence" or t in ("puis", "après"):
                left = self.chunks(tokens[:i]); right = self.chunks(tokens[i + 1:])
                e1, e2 = structure["events"]
                first = e1["patient"]
                l_match = any(np.desc == {k: v for k, v in first.items() if k in np.desc} and np.type == first.get("type") for np in left)
                r_match = any(np.type == first.get("type") and np.color == first.get("color") for np in right)
                if l_match != r_match: self.connectors[t]["keep" if l_match else "reverse"] += 1
                break

    def role_scores(self, tokens, np, rank, kind):
        roles = ROLES[kind]
        scores = {}
        for role in roles:
            s = 0.0
            for f in self.features(tokens, np, rank, kind if kind != "command" else "event"):
                c = self.counts.get(f)
                total = sum(c.values()) if c else 0
                s += math.log(((c[role] if c else 0) + 0.5) / (total + 0.5 * len(roles)))
            scores[role] = s
        return scores

    def assign(self, tokens, nps, kind):
        roles = ROLES[kind]
        scored = [self.role_scores(tokens, np, i, kind) for i, np in enumerate(nps)]
        best, best_s = None, -1e9
        for perm in itertools.permutations(roles, min(len(roles), len(nps))):
            s = sum(scored[i][r] for i, r in enumerate(perm))
            if s > best_s: best, best_s = perm, s
        return {i: r for i, r in enumerate(best or ())}

    def class_plausible(self, prev, cls, min_p=0.1):
        c = self.class_after.get(prev)
        if not c or sum(c.values()) < 3: return True
        return c[cls] / sum(c.values()) >= min_p

    def connector(self, tok):
        c = self.connectors.get(tok)
        return None if not c else c.most_common(1)[0][0]

    def gender_of(self, head):
        """Genre d'un nom appris des déterminants observés devant lui (par mot, pas par type)."""
        c = self.gender.get(head)
        if not c: return None
        fem = sum(n for t, n in c.items() if t in ("la", "une")); masc = sum(n for t, n in c.items() if t in ("le", "un", "du"))
        return "f" if fem > masc else "m" if masc > fem else None


# ================================================================ lexique neuronal (élève distillé)

class NeuralLexicon:
    """Cœur caractères (réutilisé de language_core) distillé depuis le lexique induit par WCE."""

    def __init__(self, lexicon: LexiconInducer, size="small", seed=0, steps=300):
        lex = lexicon.lexicon()
        self.symbols = sorted(set(lex.values())) + ["FUNCTION"]
        self.core = CharCore(size, seed, concepts=self.symbols)
        words = list(lex) + [t for t in lexicon.tok if t not in lex and lexicon.tok[t] >= 2]
        labels = [lex.get(w, "FUNCTION") for w in words]
        TrainCore(words=tuple(words), labels=tuple(labels), steps=steps).apply(self.core)
        self.core.snapshot("good")
        self.train_words = set(words)

    def propose(self, word, k=3): return self.core.propose(word, k)


# ================================================================ preuves, sources, croyances

FACT, BELIEF, HYPOTHESIS, UNKNOWN, REFUTED = "FACT", "BELIEF", "HYPOTHESIS", "UNKNOWN", "REFUTED"


@dataclass
class Evidence:
    source: str
    value: Any
    time: int
    via: str
    text: Optional[str] = None
    weight: float = 1.0          # confiance de l'identité dont dépend la preuve (1 = directe)


@dataclass
class BeliefEntry:
    key: Tuple
    value: Any
    posterior: float
    status: str
    supporting: List[str] = field(default_factory=list)
    refuting: List[str] = field(default_factory=list)


class SourceReliability:
    """Beta(a, b) par source ; mis à jour par accord avec le consensus des autres sources (leave-one-out)."""

    def __init__(self, prior=(3.0, 1.5)):
        self.stats = defaultdict(lambda: list(prior))

    def estimate(self, source):
        a, b = self.stats[source]; return a / (a + b)

    def update(self, source, agree):
        self.stats[source][0 if agree else 1] += 1

    def table(self): return {s: round(self.estimate(s), 3) for s in sorted(self.stats)}

    def fit_em(self, vote_sets, iterations=15, prior=(2.0, 1.0)):
        """Dawid-Skene simplifié : alterne postérieures des propositions et fiabilité de chaque source.
        vote_sets : liste de (taille du domaine, {source: valeur})."""
        rel = defaultdict(lambda: 0.7)
        for _ in range(iterations):
            correct, total = Counter(), Counter()
            for k, votes in vote_sets:
                if len(votes) < 2: continue
                scores = Counter()
                for src, v in votes.items():
                    r = min(max(rel[src], 0.05), 0.99)
                    scores[v] += math.log(r * (k - 1) / (1 - r))
                z = sum(math.exp(x) for x in scores.values()) + (k - len(scores))
                post = {v: math.exp(x) / z for v, x in scores.items()}
                for src, v in votes.items(): correct[src] += post[v]; total[src] += 1
            for src in total: rel[src] = (correct[src] + prior[0]) / (total[src] + prior[0] + prior[1])
        for src, r in rel.items():
            n = total[src] + prior[0] + prior[1]
            self.stats[src] = [r * n, (1 - r) * n]
        return self.table()


class EvidenceGraph:
    def __init__(self, reliability: SourceReliability, naive=False):
        self.props: Dict[Tuple, List[Evidence]] = defaultdict(list)
        self.reliability, self.naive = reliability, naive

    def add(self, key, evidence: Evidence): self.props[key].append(evidence)

    def votes(self, key):
        latest = {}
        for ev in self.props.get(key, ()): latest[ev.source] = ev          # un vote par source : la répétition ne prouve rien
        return latest

    def belief(self, key, exclude=None, via=None) -> BeliefEntry:
        votes = {s: e for s, e in self.votes(key).items() if s != exclude and (via is None or e.via == via)}
        if not votes: return BeliefEntry(key, None, 0.0, UNKNOWN)
        if self.naive:
            ev = self.props[key][-1]
            return BeliefEntry(key, ev.value, 1.0, FACT, [ev.source])
        k = KIND_DOMAIN.get(key[0], 6)
        rel_of = getattr(self.reliability, "estimate_for", None)
        est = (lambda s: rel_of(s, key[0])) if rel_of else self.reliability.estimate
        group_of = getattr(self.reliability, "group_of", None)
        if group_of is not None:                            # sources dépendantes : un seul vote par groupe (le plus fiable)
            by_group = {}
            for s, e in votes.items():
                g = group_of(s)
                if g not in by_group or est(s) > est(by_group[g][0]): by_group[g] = (s, e)
            votes = dict(by_group.values())
        scores = Counter()
        for s, e in votes.items():
            r = min(est(s), 0.99)
            r = e.weight * r + (1 - e.weight) / k           # identité incertaine : mélange avec un vote aléatoire
            scores[e.value] += math.log(r * (k - 1) / max(1 - r, 1e-6))
        z = sum(math.exp(v) for v in scores.values()) + (k - len(scores))
        value, sc = max(scores.items(), key=lambda t: t[1])
        post = math.exp(sc) / z
        supporting = [s for s, e in votes.items() if e.value == value]
        refuting = [s for s, e in votes.items() if e.value != value]
        strong = len(supporting) >= 2 or any(est(s) * votes[s].weight >= 0.9 for s in supporting)
        status = FACT if post >= 0.95 and strong else BELIEF if post >= 0.9 else HYPOTHESIS if post >= 0.5 else UNKNOWN
        if getattr(self.reliability, "conflict_detection", False) and status in (HYPOTHESIS, UNKNOWN) and len(scores) > 1:
            ranked = sorted((math.exp(v) / z for v in scores.values()), reverse=True)
            # crédibilité pour un CONFLIT (moyenne a posteriori, seuil propre) ≠ pondération prudente (borne basse)
            cred = getattr(self.reliability, "conflict_credibility", None) or est
            thr = getattr(self.reliability, "conflict_threshold", 0.7)
            strong_ref = [s for s in refuting if cred(s) >= thr]
            if ranked[1] >= 0.2 and strong_ref and any(cred(s) >= thr for s in supporting): status = CONFLICTED
        return BeliefEntry(key, value, post, status, supporting, refuting)

    def update_reliability(self):
        for key in list(self.props):
            for s, e in self.votes(key).items():
                others = self.belief(key, exclude=s)
                if others.status in (FACT, BELIEF): self.reliability.update(s, e.value == others.value)


class BeliefState:
    """Ce que l'agent croit — distinct de WorldTruth. Inventaire + graphe de preuves."""

    def __init__(self, reliability, naive=False):
        self.inventory: Dict[str, str] = {}
        self.graph = EvidenceGraph(reliability, naive)
        self.unresolved = 0

    def known(self, entry): return entry.status in (FACT, BELIEF)

    def color(self, o): return self.graph.belief(("color", o))

    def loc(self, o, time="now"):
        if time == "now":
            ev = self.graph.belief(("event", o))
            if self.known(ev): return BeliefEntry(("loc", o, "now"), ev.value, ev.posterior, ev.status, ev.supporting, ev.refuting)
        return self.graph.belief(("loc", o, time))

    def height(self, o):
        loc = self.loc(o)
        if self.known(loc) and (loc.value[1] in SUPPORTS or loc.value[0] == "under"):
            h = 0.0 if loc.value[0] == "under" else SUPPORTS[loc.value[1]]
            return BeliefEntry(("height", o), h, loc.posterior, loc.status, loc.supporting, loc.refuting)
        return self.graph.belief(("height", o, "now"))

    def owner(self, o): return self.graph.belief(("owner", o))

    def objects_of(self, type_): return [o for o, k in self.inventory.items() if k == type_]


# ================================================================ spécifications WCE pour les commandes

def put_spec(obj, rel, target):
    def compiler(world):
        rels = [RelationChange("remove", obj, r.predicate, r.object) for r in world.relations if r.subject == obj]
        return Transformation(f"put({obj},{rel},{target})", [], rels + [RelationChange("add", obj, rel, target)])
    return DeclarativeTransformationSpec(name=f"put({obj},{rel},{target})", domain=TransformationDomain.PHYSICS,
                                         parameters={"object": obj, "rel": rel, "target": target}, changes={"relations"},
                                         compiler=compiler)


def affordance(kinds, patient, rel, target):
    """Connaissance structurelle innée : seule une boîte contient ; on pose sur un support ou une boîte."""
    if patient == target or patient in SUPPORTS: return False
    if rel == "in": return kinds.get(target) == "box" and kinds.get(patient) != "box"
    if rel == "on": return target in SUPPORTS or kinds.get(target) == "box"
    if rel == "under": return target in ("table", "shelf")
    return rel == "next_to" and target not in SUPPORTS


# ================================================================ réponse et explication

@dataclass
class Answer:
    text: str
    output: str
    decision: str
    intent: str
    claim: str = ""
    evidence: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    neural_confidence: Optional[float] = None
    geometric_risk: Optional[float] = None
    external: str = "NOT_USED"
    checks: int = 0
    candidate_output: Optional[str] = None
    corrected: bool = False
    initial_output: Optional[str] = None
    flags: List[str] = field(default_factory=list)
    entries: List[Any] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    word_meanings: Dict[str, str] = field(default_factory=dict)
    neural_words: List[str] = field(default_factory=list)
    memory: set = field(default_factory=set)

    def explain(self):
        geo = f"{self.geometric_risk:.2f}" if self.geometric_risk is not None else "n/a"
        conf = f"{self.neural_confidence:.2f}" if self.neural_confidence is not None else "n/a"
        return "\n".join([f"Claim: {self.claim or self.text}", f"Evidence: {'; '.join(self.evidence) or 'aucune'}",
                          f"Sources: {', '.join(sorted(set(self.sources))) or 'aucune'}",
                          f"Structural reasoning: {' -> '.join(self.reasoning) or '-'}", f"Neural confidence: {conf}",
                          f"Geometric risk: {geo}", f"External verifier: {self.external}", f"Decision: {self.decision} -> {self.output}"])


# ================================================================ agent

@dataclass
class AgentConfig:
    neural: bool = True
    wce: bool = True
    evidence_graph: bool = True
    geometry: bool = True
    correction: bool = True
    plasticity: bool = False
    verification: str = "geometry"      # exhaustive | selective | geometry (sélectif, exhaustif si risque élevé)
    size: str = "small"
    tentative_confidence: float = 0.9


class IndependentAgent:
    def __init__(self, config: AgentConfig = None, seed=0):
        self.cfg = config or AgentConfig()
        self.lex = LexiconInducer()
        self.grammar = GrammarInducer(self.lex)
        self.morph = MorphologyLearner()
        self.reliability = SourceReliability()
        self.neural: Optional[NeuralLexicon] = None
        self.monitor = self.estimator = None
        self.belief: Optional[BeliefState] = None
        self.seed = seed
        self.stats = Counter()

    # ---- apprentissage ------------------------------------------------------
    def learn_captions(self, caps):
        for c in caps:
            toks = tokenize(c["text"])
            self.lex.observe(toks, c["structure"])
        for c in caps: self.grammar.learn(tokenize(c["text"]), c["structure"])
        by_meaning = defaultdict(list)
        for t, m in self.lex.lexicon().items(): by_meaning[m].append(t)
        for m, forms in by_meaning.items():
            for a in forms:
                for b in forms:
                    if a != b and len(a) > len(b): self.morph.learn_pair(a, b)
        if self.cfg.neural:
            self.neural = NeuralLexicon(self.lex, self.cfg.size, self.seed, steps=300 if not self.cfg.plasticity else 500)
        return self

    def attach_geometry(self, monitor, estimator): self.monitor, self.estimator = monitor, estimator

    def warm_up_sources(self, train_worlds):
        """Estime la fiabilité des sources sur la partition d'entraînement (EM ; aucune donnée de test)."""
        vote_sets = []
        saved = self.cfg.evidence_graph
        for x in train_worlds:
            self.new_world(); self.ingest(x["observations"], update=False)
            for key in self.belief.graph.props:
                votes = {src: ev.value for src, ev in self.belief.graph.votes(key).items()}
                vote_sets.append((KIND_DOMAIN.get(key[0], 6), votes))
        return self.reliability.fit_em(vote_sets)

    # ---- lexique : mémoire structurelle -> morphologie -> proposition neuronale
    def word_meaning(self, tok):
        """(sens, source, confiance). Jamais d'invention silencieuse : un mot sans évidence reste inconnu."""
        if tok in INNATE_QUESTION or tok in INNATE_PRONOUN or tok in INNATE_OTHER or is_number(tok): return None, "innate", 1.0
        if self.lex.seen(tok):
            m = self.lex.meaning(tok)
            return m, "grounded" if m else "function", 1.0
        if self.cfg.wce:
            known = self.lex.lexicon()
            cands = self.morph.candidates(tok, known)
            if cands:
                ref, meaning, rule, support = cands[0]
                if support >= 2: return meaning, "morphology", 0.95
        if self.neural is not None:
            (sym, p), = self.neural.propose(tok, 1)
            return (None if sym == "FUNCTION" else sym), "neural", p
        return None, "unknown", 0.0

    # ---- ingestion des observations ------------------------------------------
    def new_world(self):
        self.belief = BeliefState(self.reliability, naive=not self.cfg.evidence_graph)
        return self.belief

    def resolve(self, np: NP, belief=None, discourse=None, strict=True):
        """Référents candidats d'un syntagme selon les croyances (jamais selon la vérité).
        strict (requêtes) : une couleur décrite doit être vérifiée, même pour un objet unique.
        tolérant (ingestion) : un objet unique de ce type suffit."""
        belief = belief or self.belief
        if np.pronoun:
            cands = [o for o, g in (discourse or []) if g == np.pronoun or g is None]
            return list(dict.fromkeys(cands)), []
        if np.type in SUPPORTS: return [np.type], []
        if not np.type: return [], []
        objs = belief.objects_of(np.type)
        if not np.color or (len(objs) <= 1 and (not strict or not self.cfg.wce)): return objs, []
        match, unknown = [], []
        for o in objs:
            c = belief.color(o)
            if belief.known(c):
                if c.value == np.color: match.append(o)
            else: unknown.append(o)
        return match, unknown

    def _gender(self, o):
        t = self.belief.inventory.get(o)
        return self.grammar.gender_of(t) if t else None

    def ingest(self, observations, update=True):
        belief = self.belief or self.new_world()
        texts = []
        for ob in observations:
            key = ob.key
            if key[0] == "type": belief.inventory[key[1]] = ob.value; continue
            if ob.text is None:
                if ob.implied: continue
                if key[0] == "event": key, value = ("event", ob.value.patient), (ob.value.rel, ob.value.target)
                else: value = ob.value
                if key[0] == "loc": key = ("loc", key[1], "now" if key[2] != 0 else 0)
                belief.graph.add(key, Evidence(ob.source, value, ob.time, "structured"))
            else:
                texts.append(ob)
        for ob in texts:
            prop = self.parse_statement(tokenize(ob.text))
            if prop is None: belief.unresolved += 1; continue
            for key, value in prop:
                belief.graph.add(key, Evidence(ob.source, value, ob.time, "text", ob.text))
        if self.cfg.evidence_graph and update: belief.graph.update_reliability()
        return belief

    def parse_statement(self, tokens):
        """Phrase d'observation -> liste (clé, valeur) sur les objets de l'inventaire, ou None."""
        meanings = [self.word_meaning(t)[0] for t in tokens]
        nps = self.grammar.chunks(tokens, meanings)
        syms = set(m for m in meanings if m)
        kind = self._kind(syms, tokens)
        if kind is None or not nps: return None
        roles = self.grammar.assign(tokens, nps, kind)
        by_role = {r: nps[i] for i, r in roles.items()}
        out = []
        for np in nps:
            if np.color and np.type:
                objs, _ = self.resolve(NP(0, 0, type=np.type), strict=False)
                if len(objs) == 1: out.append((("color", objs[0]), np.color))
        subj = by_role.get("subject") or by_role.get("patient")
        if subj is None: return out or None
        objs, unknown = self.resolve(subj, strict=False)
        if len(objs) != 1 or unknown: return out or None
        o = objs[0]
        if kind == "color":
            color = next((m.split(":")[1] for i, m in enumerate(meanings) if m and m.startswith("color:") and not any(n.start <= i < n.end for n in nps)), None)
            if color: out.append((("color", o), color))
        elif kind in ("loc", "event"):
            tgt = by_role.get("target")
            rel = next((m.split(":")[1] for m in meanings if m and m.startswith("rel:")), None)
            if kind == "event" and "act:fall" in syms: rel, tgt_id = "on", "floor"
            else:
                tobjs, tunk = self.resolve(tgt, strict=False) if tgt else ([], [])
                if len(tobjs) != 1 or tunk or not rel: return out or None
                tgt_id = tobjs[0]
            if kind == "event": out.append((("event", o), (rel, tgt_id)))
            else: out.append((("loc", o, 0 if "time:past" in syms else "now"), (rel, tgt_id)))
        elif kind == "height":
            value = self._number(tokens)
            if value is not None: out.append((("height", o, "now"), value))
        elif kind == "owner":
            person = next((m.split(":")[1] for m in meanings if m and m.startswith("person:")), None)
            if person: out.append((("owner", o), person))
        return out or None

    def _kind(self, syms, tokens):
        for k in ("height", "owner", "count"):
            if f"kind:{k}" in syms: return k
        if any(s.startswith("act:") for s in syms): return "event"
        if any(s.startswith("rel:") for s in syms): return "loc"
        if any(is_number(t) for t in tokens): return "height"
        if any(s.startswith("person:") for s in syms): return "owner"
        if any(s.startswith("color:") for s in syms): return "color"
        return None

    def _number(self, tokens):
        for i, t in enumerate(tokens):
            if is_number(t):
                unit = tokens[i + 1] if i + 1 < len(tokens) else None
                scale = self.lex.unit_scale(unit) if unit else None
                return round(to_float(t) * (scale if scale is not None else 1.0), 4)
        return None

    # ---- requêtes -----------------------------------------------------------------
    def answer(self, text, mode=None) -> Answer:
        mode = mode or self.cfg.verification
        sentences = [s for s in re.split(r"(?<=[.])\s+", text.strip()) if s]
        discourse = []
        for s in sentences[:-1]:
            toks = tokenize(s)
            for np in self.grammar.chunks(toks, [self.word_meaning(t)[0] for t in toks]):
                objs, unk = self.resolve(np)
                gender = self.grammar.gender_of(np.head) if np.head else None
                if len(objs) == 1 and not unk: discourse.append((objs[0], gender))
                else: discourse.append(("AMBIGUOUS", gender))
        tokens = tokenize(sentences[-1])
        ans = Answer(text, "UNKNOWN", "ABSTAIN", "?")
        meanings, sources, confs, neural_words = [], [], [], []
        for t in tokens:
            m, src, conf = self.word_meaning(t)
            meanings.append(m); sources.append(src)
            if src == "neural": neural_words.append(t); confs.append(conf)
        ans.neural_words = neural_words
        ans.neural_confidence = min(confs) if confs else 1.0
        ans.features = {"length": len(tokens), "novelty": len(neural_words) / max(1, len(tokens)),
                        "neural_confidence": ans.neural_confidence, "support": 0.0, "source_reliability": 0.0, "ambiguity": 0.0}
        if self.monitor is not None and neural_words:
            zs = [self.monitor.observe(self.neural.core, w) for w in neural_words]
            ans.geometric_risk = max(self.estimator.score(z) for z in zs)
            for k in zs[0]: ans.features[f"geo_{k}"] = max(z[k] for z in zs)
        else:
            ans.geometric_risk = 0.0 if self.monitor is not None else None
        ans.features["geometric_risk"] = ans.geometric_risk or 0.0
        if self.cfg.wce:
            for i, src in enumerate(sources):
                if src in ("neural", "unknown"): meanings[i] = self._resolve_oov(tokens, meanings, i, ans, mode)
        unknown = [t for i, t in enumerate(tokens) if sources[i] in ("neural", "unknown") and meanings[i] is None and self.cfg.wce]
        for t, m, src in zip(tokens, meanings, sources):
            if src not in ("innate", "function"): ans.word_meanings[t] = m or "?"
        ans.memory.add("current_state")
        if any(s == "grounded" for s in sources): ans.memory |= {"structural", "episodic"}
        if any(s == "morphology" for s in sources): ans.memory.add("structural")
        if neural_words: ans.memory.add("slow_parametric")
        nps = self.grammar.chunks(tokens, meanings)
        if nps and nps[0].type and nps[0].type not in SUPPORTS:
            ans.features["ambiguity"] = max(0, len(self.belief.objects_of(nps[0].type)) - 1)
        if unknown:
            ans.output, ans.decision = "UNKNOWN_WORD", "ABSTAIN"
            ans.reasoning.append(f"mot(s) sans évidence suffisante : {', '.join(unknown)}")
            if self.neural is not None:                      # candidat qu'aurait donné la proposition neuronale brute
                raw = [(self.neural.propose(t, 1)[0][0] if src in ("neural", "unknown") else m) for t, m, src in zip(tokens, meanings, sources)]
                raw = [None if r == "FUNCTION" else r for r in raw]
                ans.candidate_output = self._candidate(tokens, raw, discourse, Answer(text, "", "", "?"))
            return self._finalize(ans, text)
        out = self._candidate(tokens, meanings, discourse, ans)
        ans.candidate_output = out
        ans.output = out
        ans.decision = "ABSTAIN" if out in ABSTAIN else ("REJECT" if out in ("REJECT", "INVALID_PREMISE") else "ACCEPT")
        return self._finalize(ans, text)

    def _resolve_oov(self, tokens, meanings, i, ans, mode):
        """Sens d'un mot hors lexique : proposition neuronale vérifiée (sélectif) ou inférence
        structurelle exhaustive (tous les sens compatibles avec les croyances ; un seul -> retenu)."""
        word = tokens[i]
        geo_high = False
        if self.neural is not None and mode in ("selective", "geometry"):
            (sym, p), = self.neural.propose(word, 1)
            if self.monitor is not None and self.cfg.geometry:
                geo_high = self.estimator.level(self.monitor.observe(self.neural.core, word)) == "HIGH"
            if sym != "FUNCTION" and p >= self.cfg.tentative_confidence and not geo_high:
                ans.checks += 1
                if self._consistent(tokens, meanings, i, sym):
                    ans.reasoning.append(f"« {word} » ≈ {sym} (neuronal {p:.2f}, vérifié dans le contexte)")
                    return sym
            if mode == "selective" or (mode == "geometry" and not geo_high): return None
        candidates = sorted({m for m in self.lex.lexicon().values() if token_class(m) in ("N", "C")})
        ok = []
        for sym in candidates:
            ans.checks += 1
            if self._consistent(tokens, meanings, i, sym): ok.append(sym)
        if len(ok) == 1:
            ans.reasoning.append(f"« {word} » = {ok[0]} (seul sens compatible avec les croyances)")
            ans.memory.add("structural")
            return ok[0]
        return None

    def _consistent(self, tokens, meanings, i, sym):
        """Le sens proposé désigne-t-il un référent existant (et unique au besoin) dans l'état cru ?"""
        prev = tokens[i - 1] if i > 0 else "<s>"
        if not self.grammar.class_plausible(prev, token_class(sym)): return False
        trial = list(meanings); trial[i] = sym
        nps = self.grammar.chunks(tokens, trial)
        np = next((n for n in nps if n.start <= i < n.end), None)
        if np is None: return token_class(sym) not in ("N",)
        objs, unknown = self.resolve(np)
        return len(objs) >= 1

    def _affordance(self, kinds, patient, rel, target):
        return affordance(kinds, patient, rel, target)

    def _neural_ok(self, word):
        """Une proposition neuronale sur un mot inconnu n'est utilisable que si elle est sûre ET géométriquement saine."""
        if not self.cfg.wce: return True
        (sym, p), = self.neural.propose(word, 1)
        if p < self.cfg.tentative_confidence: return False
        if self.cfg.geometry and self.monitor is not None:
            return self.estimator.level(self.monitor.observe(self.neural.core, word)) == "LOW"
        return True

    def _finalize(self, ans, text=None):
        ans.initial_output = ans.output
        if self.cfg.correction and self.cfg.wce and text is not None: self._self_correct(ans, text)
        self.stats[ans.decision] += 1
        return ans

    # ---- interprétation structurée ------------------------------------------------
    def _candidate(self, tokens, meanings, discourse, ans, force=False):
        syms = set(m for m in meanings if m)
        nps = self.grammar.chunks(tokens, meanings)
        ans.checks += 1
        if tokens and tokens[0] == "si" and "," in tokens:
            i = tokens.index(",")
            cond = self._claim(tokens[1:i], meanings[1:i], ans)
            ans.reasoning.append(f"condition -> {cond}")
            if cond == "SUPPORTED": return self._command(tokens[i + 1:], meanings[i + 1:], discourse, ans)
            if cond == "REFUTED": return "NONE"
            if cond == "UNKNOWN": ans.flags.append("unsupported_inference")
            return "UNKNOWN" if cond == "UNKNOWN" else cond
        if "pourquoi" in tokens: ans.intent = "why"; return self._why(tokens, meanings, nps, ans)
        if "où" in tokens: ans.intent = "where"; return self._where(tokens, meanings, nps, ans)
        if "couleur" in tokens: ans.intent = "color"; return self._color(nps, ans)
        if "kind:height" in syms and "?" in tokens: ans.intent = "height"; return self._height(nps, ans)
        first = meanings[0] if meanings else None
        if first and first.startswith("act:") or (tokens and tokens[0] in ("mets",)):
            ans.intent = "command"; return self._command(tokens, meanings, discourse, ans)
        ans.intent = "claim"
        return self._claim(tokens, meanings, ans)

    def _single(self, np, ans, discourse=None):
        objs, unknown = self.resolve(np, discourse=discourse)
        ans.checks += 1
        if np.pronoun:
            if "AMBIGUOUS" in objs: return "ASK"
            return objs[0] if len(objs) == 1 else ("ASK" if len(objs) > 1 else "UNKNOWN")
        if len(objs) == 1 and not unknown: return objs[0]
        if len(objs) > 1 or (not np.color and len(self.belief.objects_of(np.type)) > 1): return "ASK"
        if not objs and not unknown: return "INVALID_PREMISE"
        ans.flags.append("unknown_entity")
        return "UNKNOWN"

    def _entry_text(self, entry, ans, label):
        ans.entries.append(entry)
        ans.evidence.append(f"{label}: {entry.value} ({entry.status}, p={entry.posterior:.2f}, pour={entry.supporting}, contre={entry.refuting})")
        ans.sources += entry.supporting
        ans.features["support"] = entry.posterior
        ans.features["source_reliability"] = (sum(self.reliability.estimate(s) for s in entry.supporting) / len(entry.supporting)) if entry.supporting else 0.0
        if entry.supporting: ans.memory.add("episodic")

    def _where(self, tokens, meanings, nps, ans):
        if not nps: return "UNKNOWN"
        o = self._single(nps[0], ans)
        if o in ("ASK", "UNKNOWN", "INVALID_PREMISE"): return o
        time = 0 if "time:past" in meanings or "début" in tokens else "now"
        entry = self.belief.loc(o, time)
        self._entry_text(entry, ans, f"loc({o},{time})")
        ans.reasoning.append(f"référent {o} ; croyance {entry.status}")
        return f"{entry.value[0]}:{entry.value[1]}" if self.belief.known(entry) else "UNKNOWN"

    def _color(self, nps, ans):
        if not nps: return "UNKNOWN"
        o = self._single(nps[0], ans)
        if o in ("ASK", "UNKNOWN", "INVALID_PREMISE"): return o
        entry = self.belief.color(o); self._entry_text(entry, ans, f"color({o})")
        return entry.value if self.belief.known(entry) else "UNKNOWN"

    def _height(self, nps, ans):
        if not nps: return "UNKNOWN"
        o = self._single(nps[0], ans)
        if o in ("ASK", "UNKNOWN", "INVALID_PREMISE"): return o
        entry = self.belief.height(o); self._entry_text(entry, ans, f"height({o})")
        return f"{entry.value:g}" if self.belief.known(entry) else "UNKNOWN"

    def _why(self, tokens, meanings, nps, ans):
        if not nps: return "UNKNOWN"
        o = self._single(nps[0], ans)
        if o in ("ASK", "UNKNOWN", "INVALID_PREMISE"):
            ans.reasoning.append("présupposition d'existence non satisfaite" if o == "INVALID_PREMISE" else "référent incertain")
            return o
        rel = next((m.split(":")[1] for m in meanings if m and m.startswith("rel:")), None)
        target = nps[1].type if len(nps) > 1 else None
        entry = self.belief.loc(o); self._entry_text(entry, ans, f"loc({o})")
        if not self.belief.known(entry): return "UNKNOWN"
        if entry.value == (rel, target):
            ans.reasoning.append("présupposition vraie ; aucune cause connue"); return "UNKNOWN_CAUSE"
        ans.reasoning.append(f"présupposition fausse : {o} est {entry.value}"); return "INVALID_PREMISE"

    def _claim(self, tokens, meanings, ans):
        syms = set(m for m in meanings if m)
        nps = self.grammar.chunks(tokens, meanings)
        kind = self._kind(syms, tokens)
        if not nps or kind is None: return "UNKNOWN"
        roles = self.grammar.assign(tokens, nps, kind)
        by_role = {r: nps[i] for i, r in roles.items()}
        subj = by_role.get("subject") or nps[0]
        ans.claim = f"{kind}({subj.desc})"
        if kind == "loc":
            tgt = by_role.get("target")
            rel = next((m.split(":")[1] for m in meanings if m and m.startswith("rel:")), None)
            if self.cfg.wce and subj.type in SUPPORTS and tgt is not None:
                kinds = {**self.belief.inventory}
                t_objs, _ = self.resolve(tgt)
                if not self._affordance(kinds, subj.type, rel, t_objs[0] if t_objs else "?"):
                    ans.reasoning.append("relation impossible (affordance structurelle)"); ans.checks += 1; return "REJECT"
        if kind == "color":
            colors = [m.split(":")[1] for i, m in enumerate(meanings) if m and m.startswith("color:") and not any(n.start <= i < n.end for n in nps)]
            if self.cfg.wce and subj.color and colors and subj.color != colors[0]:
                ans.reasoning.append("contradiction interne : la description et l'affirmation divergent"); ans.checks += 1; return "REJECT"
        o = self._single(subj, ans)
        if o in ("ASK", "UNKNOWN"): return o
        if o == "INVALID_PREMISE": return "UNKNOWN"
        if kind == "color":
            colors = [m.split(":")[1] for i, m in enumerate(meanings) if m and m.startswith("color:") and not any(n.start <= i < n.end for n in nps)]
            if not colors: return "UNKNOWN"
            entry, claimed = self.belief.color(o), colors[0]
        elif kind == "loc":
            tgt = by_role.get("target")
            rel = next((m.split(":")[1] for m in meanings if m and m.startswith("rel:")), None)
            if tgt is None or rel is None: return "UNKNOWN"
            t = self._single(tgt, ans)
            if t in ("ASK", "UNKNOWN", "INVALID_PREMISE"): return "UNKNOWN"
            entry, claimed = self.belief.loc(o, 0 if "time:past" in syms else "now"), (rel, t)
        elif kind == "height":
            claimed = self._number(tokens)
            if claimed is None: return "UNKNOWN"
            entry = self.belief.height(o)
        elif kind == "owner":
            claimed = next((m.split(":")[1] for m in meanings if m and m.startswith("person:")), None)
            entry = self.belief.owner(o)
        else:
            return "UNKNOWN"
        self._entry_text(entry, ans, f"{kind}({o})")
        if not self.belief.known(entry): return "UNKNOWN"
        same = abs(entry.value - claimed) < 1e-6 if kind == "height" else entry.value == claimed
        ans.reasoning.append(f"croyance {entry.value} vs affirmation {claimed}")
        return "SUPPORTED" if same else "REFUTED"

    def _command(self, tokens, meanings, discourse, ans):
        """Commandes compositionnelles -> Goals -> planner WCE sur le monde CRU (pas la vérité)."""
        clauses, current = [], []
        for t, m in zip(tokens, meanings):
            if t in ("puis",) or (m == "kind:sequence"): clauses.append(current); current = []
            else: current.append((t, m))
        clauses.append(current)
        order = self.grammar.connector("puis")
        if order == "reverse": clauses = clauses[::-1]
        effects = []
        world = self._belief_world()
        for clause in clauses:
            ctoks = [t for t, _ in clause]; cms = [m for _, m in clause]
            parsed = self._command_clause(ctoks, cms, discourse, ans)
            if isinstance(parsed, str): return parsed
            patient, rel, target = parsed
            if self.cfg.wce and not self._affordance(self.belief.inventory, patient, rel, target):
                ans.reasoning.append(f"affordance violée : {patient} {rel} {target}"); return "REJECT"
            goal = Goal(rel, (patient, target))
            planner = TransformationPlanner([lambda state, g: [put_spec(g.target[0], g.predicate, g.target[1])]], max_depth=1, max_nodes=10)
            result = planner.search(world, goal)
            ans.checks += 1
            ans.reasoning.append(f"Goal {rel}({patient},{target}) -> {'plan ' + result.transformations[0].name if result.success and result.transformations else 'déjà satisfait' if result.success else 'échec'}")
            if not result.success: ans.flags.append("planner_failure"); return "REJECT"
            world = result.final_state
            effects = [e for e in effects if e[0] != patient] + [(patient, rel, target)]
            discourse = discourse + [(patient, None)]
        ans.claim = "commande"
        return ";".join(sorted(f"{p}:{r}:{t}" for p, r, t in effects)) or "NONE"

    def _command_clause(self, tokens, meanings, discourse, ans):
        nps = self.grammar.chunks(tokens, meanings)
        rels = [(i, m.split(":")[1]) for i, m in enumerate(meanings) if m and m.startswith("rel:")]
        if "qui" in tokens:                                                   # « pose dans X l'objet qui est R Y »
            q = tokens.index("qui")
            head = [np for np in nps if np.end <= q]
            tail = [np for np in nps if np.start > q]
            target_np = next((np for np in head if not np.generic), None)
            rel_t = next((r for i, r in rels if i < q), None)
            rel_c = next((r for i, r in rels if i > q), None)
            if not target_np or not tail or not rel_t or not rel_c: return "UNKNOWN"
            ref_target = self._single(tail[0], ans)
            if ref_target in ("ASK", "UNKNOWN", "INVALID_PREMISE"): return "UNKNOWN"
            matches, unknown = [], []
            for o in self.belief.inventory:
                entry = self.belief.loc(o)
                if not self.belief.known(entry): unknown.append(o)
                elif entry.value == (rel_c, ref_target): matches.append(o)
            ans.checks += len(self.belief.inventory)
            if len(matches) > 1: return "ASK"
            if len(matches) != 1 or unknown: return "UNKNOWN"
            t = self._single(target_np, ans)
            if t in ("ASK", "UNKNOWN", "INVALID_PREMISE"): return t if t != "INVALID_PREMISE" else "UNKNOWN"
            return (matches[0], rel_t, t)
        if len(rels) >= 2 and len(nps) >= 3:                                  # attachement ambigu : branches
            (i1, r1), (i2, r2) = rels[0], rels[1]
            patient = self._single(nps[0], ans)
            box = self._single(nps[1], ans)
            if patient in ("ASK", "UNKNOWN", "INVALID_PREMISE") or box in ("ASK", "UNKNOWN", "INVALID_PREMISE"): return "UNKNOWN"
            last = nps[2].type
            readings = []
            p_loc, b_loc = self.belief.loc(patient), self.belief.loc(box)
            ans.checks += 2
            if self.belief.known(p_loc) and p_loc.value == (r1, box): readings.append((patient, r2, last))
            if self.belief.known(b_loc) and b_loc.value == (r2, last): readings.append((patient, r1, box))
            ans.reasoning.append(f"attachements valides : {readings}")
            if len(readings) == 2: return "ASK"
            if not readings: return "UNKNOWN"
            return readings[0]
        if not nps: return "UNKNOWN"
        roles = self.grammar.assign(tokens, nps, "command")
        by_role = {r: nps[i] for i, r in roles.items()}
        patient_np, target_np = by_role.get("patient"), by_role.get("target")
        if patient_np is None or target_np is None or not rels: return "UNKNOWN"
        patient = self._single(patient_np, ans, discourse)
        target = self._single(target_np, ans)
        for x in (patient, target):
            if x in ("ASK", "UNKNOWN"): return x
            if x == "INVALID_PREMISE": return "UNKNOWN"
        return (patient, rels[0][1], target)

    def _belief_world(self):
        w = World()
        for s in SUPPORTS: w.add_entity(Entity(s, "support", {"y": SUPPORTS[s]}))
        for o, k in self.belief.inventory.items():
            w.add_entity(Entity(o, k, {"y": 0.0}))
            entry = self.belief.loc(o)
            if self.belief.known(entry): w.add_relation(o, entry.value[0], entry.value[1])
        return w

    # ---- auto-correction --------------------------------------------------------
    def _self_correct(self, ans, text):
        """Si un mot inconnu a bloqué la réponse, relance une inférence structurelle exhaustive.
        Ne transforme jamais une abstention en réponse devinée : seule une réponse soutenue est acceptée."""
        if ans.output != "UNKNOWN_WORD": return
        if self.neural is not None and self.monitor is not None:
            for w in ans.neural_words:
                if self.estimator.level(self.monitor.observe(self.neural.core, w)) == "HIGH" and self.neural.core._snapshots:
                    self.neural.core.rollback("good"); ans.reasoning.append(f"rollback du cœur neuronal (risque élevé sur « {w} »)")
        retry = self.answer_once(text, "exhaustive") if not getattr(self, "_retrying", False) else None
        if retry is None: return
        if retry.output not in ABSTAIN:
            ans.output, ans.decision, ans.corrected = retry.output, retry.decision, True
            ans.reasoning += ["auto-correction : inférence structurelle exhaustive"] + retry.reasoning
            ans.checks += retry.checks; ans.evidence += retry.evidence; ans.sources += retry.sources

    def answer_once(self, text, mode):
        self._retrying = True
        try: return IndependentAgent.answer(self, text, mode)
        finally: self._retrying = False


# ================================================================ baseline neuronale seule

class NeuralOnlyAgent(IndependentAgent):
    """Sans vérification ni abstention : lexique neuronal pour tous les mots inconnus, dernière observation gagne."""

    def __init__(self, seed=0, plasticity=False, size="small"):
        super().__init__(AgentConfig(neural=True, wce=False, evidence_graph=False, geometry=False, correction=False,
                                     plasticity=plasticity, size=size), seed)

    def word_meaning(self, tok):
        if tok in INNATE_QUESTION or tok in INNATE_PRONOUN or tok in INNATE_OTHER or is_number(tok): return None, "innate", 1.0
        (sym, p), = self.neural.propose(tok, 1)
        return (None if sym == "FUNCTION" else sym), ("neural" if not self.lex.seen(tok) else "neural_known"), p

    def answer(self, text):
        ans = super().answer(text)
        if ans.output in ("UNKNOWN", "UNKNOWN_WORD", "ASK"):
            ans.output = self._guess(ans)                      # un modèle sans WCE répond toujours
            ans.decision = "ACCEPT"
        return ans

    def _guess(self, ans):
        if ans.intent == "claim": return "SUPPORTED"
        if ans.intent == "color": return "red"
        if ans.intent == "where": return "on:table"
        if ans.intent == "height": return "1"
        if ans.intent == "why": return "UNKNOWN_CAUSE"
        return "NONE"

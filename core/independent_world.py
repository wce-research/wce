"""Linguistic environment INDEPENDENT of WCE.

This module imports NO WCE component (neither planner, nor parser, nor agent lexicon).
It produces:
- WorldTruth: the hidden truth (objects, relations, heights, owners, events);
- partial and noisy observations, emitted by sources of differing reliability;
- French sentences via its OWN templates (paraphrases, synonyms, variations, orders);
- captions (sentence, perceived structure) for the agent's learning;
- a benchmark of queries with reference answers computed here, without WCE;
- IndependentWorldVerifier: SUPPORTED / REFUTED / UNKNOWN oracle on WorldTruth.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

KINDS = ("ball", "cube", "block", "cone")
SUPPORT_HEIGHT = {"table": 1.0, "shelf": 2.0, "floor": 0.0}
COLORS = ("red", "blue", "green", "yellow")
RELATIONS = ("on", "under", "in", "next_to")
PERSONS = ("paul", "marie")
NUMBERS = {1: "un", 2: "deux", 3: "trois", 4: "quatre", 5: "cinq", 0: "zéro"}


# ================================================================ hidden truth

@dataclass
class Obj:
    id: str
    kind: str
    color: Optional[str] = None
    owner: Optional[str] = None


@dataclass(frozen=True)
class Event:
    time: int
    action: str                  # place | fall | push
    agent: Optional[str]         # person or object
    patient: str
    rel: str
    target: str


@dataclass
class WorldTruth:
    seed: int
    objects: Dict[str, Obj]
    loc0: Dict[str, Tuple[str, str]]
    events: List[Event] = field(default_factory=list)

    @property
    def movables(self): return [o for o in self.objects if o not in SUPPORT_HEIGHT]

    def loc(self, oid, time="now"):
        loc = dict(self.loc0)
        if time == "now":
            for e in self.events: loc[e.patient] = (e.rel, e.target)
        return loc[oid]

    def height(self, oid, time="now", depth=0):
        rel, target = self.loc(oid, time)
        if rel == "under": return 0.0
        base = SUPPORT_HEIGHT[target] if target in SUPPORT_HEIGHT else (self.height(target, time, depth + 1) if depth < 5 else 0.0)
        if rel == "on" and target not in SUPPORT_HEIGHT: return base + 0.5
        return base

    def facts(self):
        out = {}
        for o in self.movables:
            obj = self.objects[o]
            out[("type", o)] = obj.kind
            out[("color", o)] = obj.color
            out[("owner", o)] = obj.owner
            out[("loc", o, 0)] = self.loc(o, 0)
            out[("loc", o, "now")] = self.loc(o, "now")
            out[("height", o, "now")] = self.height(o, "now")
        for i, e in enumerate(self.events): out[("event", i)] = e
        return out


def generate_world(seed) -> WorldTruth:
    rng = random.Random(seed)
    objects = {s: Obj(s, "support") for s in SUPPORT_HEIGHT}
    n = rng.randint(3, 5)
    has_box = rng.random() < 0.6
    kinds = [rng.choice(KINDS) for _ in range(n)] + (["box"] if has_box else [])
    for i, k in enumerate(kinds):
        objects[f"o{i + 1}"] = Obj(f"o{i + 1}", k, rng.choice(COLORS), rng.choice(PERSONS) if rng.random() < 0.35 else None)
    box = next((o for o, v in objects.items() if v.kind == "box"), None)
    loc0, placed = {}, []
    order = ([box] if box else []) + [o for o in objects if o not in SUPPORT_HEIGHT and o != box]
    for o in order:
        options = [("on", s) for s in SUPPORT_HEIGHT] * 3 + [("under", "table"), ("under", "shelf")]
        if box and o != box: options += [("in", box)] * 3
        options += [("next_to", p) for p in placed if p != box or True]
        loc0[o] = rng.choice(options)
        placed.append(o)
    world = WorldTruth(seed, objects, loc0)
    for t in range(1, rng.choice([0, 1, 1, 2]) + 1):
        patient = rng.choice(world.movables)
        action = rng.choice(["place", "place", "fall", "push"])
        if action == "place":
            dests = [("on", s) for s in SUPPORT_HEIGHT] + ([("in", box)] if box and patient != box else [])
            rel, target = rng.choice(dests)
            world.events.append(Event(t, "place", rng.choice(PERSONS), patient, rel, target))
        elif action == "fall":
            world.events.append(Event(t, "fall", None, patient, "on", "floor"))
        else:
            others = [o for o in world.movables if o != patient]
            if not others: continue
            agent = rng.choice(others)
            world.events.append(Event(t, "push", agent, patient, "next_to", agent))
    return world


# ================================================================ sources and observations

@dataclass(frozen=True)
class Source:
    name: str
    reliability: float
    coverage: float
    modality: str                # structured | text


SOURCES = (Source("sensor_A", 0.95, 0.55, "structured"), Source("sensor_B", 0.85, 0.45, "structured"),
           Source("document_A", 0.85, 0.35, "text"), Source("speaker_A", 0.7, 0.35, "text"),
           Source("speaker_B", 0.55, 0.45, "text"))
SOURCE_BY_NAME = {s.name: s for s in SOURCES}


@dataclass
class Observation:
    source: str
    key: Tuple
    value: Any
    time: int
    text: Optional[str] = None
    implied: bool = False        # colour implied by a description ("la balle rouge")


def domain_size(kind):
    return {"color": 4, "loc": 12, "height": 6, "owner": 3, "event": 12}[kind]


def _corrupt(world, key, value, rng):
    kind = key[0]
    if kind == "color": return rng.choice([c for c in COLORS if c != value])
    if kind == "owner": return rng.choice([p for p in PERSONS if p != value]) if value else rng.choice(PERSONS)
    if kind == "height": return rng.choice([h for h in (0.0, 1.0, 1.5, 2.0, 2.5) if h != value])
    if kind == "loc":
        o = key[1]
        options = [("on", s) for s in SUPPORT_HEIGHT] + [("under", "table"), ("under", "shelf")] + \
                  [("next_to", m) for m in world.movables if m != o]
        return rng.choice([x for x in options if x != value])
    return value


def emit_observations(world: WorldTruth, seed=0, sources=SOURCES) -> List[Observation]:
    rng = random.Random(seed * 7919 + world.seed)
    gen = LanguageGenerator(seed=seed * 31 + world.seed)
    obs = [Observation("sensor_A", ("type", o), world.objects[o].kind, 0) for o in world.movables]   # full inventory
    facts = world.facts()
    for key, value in facts.items():
        kind = key[0]
        if kind == "type": continue
        if kind == "owner" and value is None: continue
        for src in sources:
            coverage = src.coverage * (0.3 if kind == "owner" else 1.0)
            if kind == "event" and src.modality == "structured": coverage *= 0.8
            if rng.random() >= coverage: continue
            v = value if (kind == "event" or rng.random() < src.reliability) else _corrupt(world, key, value, rng)
            time = key[2] if kind in ("loc", "height") else 0
            text = gen.describe(world, key, v) if src.modality == "text" else None
            obs.append(Observation(src.name, key, v, time if time != "now" else 1, text))
            if text is not None:
                # a colour is implied by a description only if the reference is unambiguous (unique type)
                for o in gen.last_mentioned:
                    if sum(1 for m in world.movables if world.objects[m].kind == world.objects[o].kind) == 1:
                        obs.append(Observation(src.name, ("color", o), world.objects[o].color, 0, None, implied=True))
    return obs


# ================================================================ ideal knowability (reference)

def ideal_value(observations, key, threshold=0.9):
    """Ideal agent knowing the TRUE reliabilities: value if the posterior >= threshold, otherwise None.
    A source votes only once per proposition (repetition does not create truth)."""
    votes = {}
    for ob in observations:
        if ob.key == key: votes[ob.source] = ob.value
    if not votes: return None
    k = domain_size(key[0])
    scores = Counter()
    for src, value in votes.items():
        r = SOURCE_BY_NAME[src].reliability if src in SOURCE_BY_NAME else 0.99
        scores[value] += math.log(r * (k - 1) / max(1 - r, 1e-6))
    z = sum(math.exp(s) for s in scores.values()) + (k - len(scores))
    value, s = max(scores.items(), key=lambda t: t[1])
    return value if math.exp(s) / z >= threshold else None


def knowable(world, observations, key):
    """Value knowable AND true; None if unknowable or if the evidence is misleading."""
    truth = world.facts().get(key)
    v = ideal_value(observations, key)
    if key[0] == "loc" and key[2] == "now":
        last = [i for i, e in enumerate(world.events) if e.patient == key[1]]
        if last:
            ev = ideal_value(observations, ("event", last[-1]))
            if ev is not None: v = (ev.rel, ev.target)
    if key[0] == "height":
        loc = knowable(world, observations, ("loc", key[1], "now"))
        if loc is not None and (loc[1] in SUPPORT_HEIGHT or loc[0] == "under"): v = world.height(key[1])
    return v if v == truth else None


# ================================================================ independent oracle

class IndependentWorldVerifier:
    """Knows WorldTruth. Answers on structured propositions, without planner."""

    def __init__(self, world: WorldTruth):
        self.world = world
        self.calls = 0

    def verify(self, proposition):
        self.calls += 1
        kind = proposition.get("kind")
        facts = self.world.facts()
        if kind in ("color", "owner"):
            key = (kind, proposition["subject"])
            if key not in facts: return "UNKNOWN"
            truth = facts[key]
            if truth is None: return "UNKNOWN"
            return "SUPPORTED" if truth == proposition["value"] else "REFUTED"
        if kind == "loc":
            truth = facts.get(("loc", proposition["subject"], proposition.get("time", "now")))
            if truth is None: return "UNKNOWN"
            return "SUPPORTED" if truth == (proposition["rel"], proposition["target"]) else "REFUTED"
        if kind == "height":
            truth = facts.get(("height", proposition["subject"], "now"))
            return "UNKNOWN" if truth is None else ("SUPPORTED" if abs(truth - proposition["value"]) < 1e-6 else "REFUTED")
        return "UNKNOWN"


# ================================================================ language generator (own templates)

LEX = {"ball": [("balle", "f"), ("boule", "f"), ("ballon", "m")], "cube": [("cube", "m")],
       "block": [("bloc", "m"), ("brique", "f")], "cone": [("cône", "m")], "box": [("boîte", "f"), ("caisse", "f")],
       "table": [("table", "f")], "shelf": [("étagère", "f"), ("rayon", "m")], "floor": [("sol", "m"), ("plancher", "m")]}
COLOR_FORMS = {"red": ("rouge", "rouge"), "blue": ("bleu", "bleue"), "green": ("vert", "verte"), "yellow": ("jaune", "jaune")}
REL_FORMS = {"on": ["sur"], "under": ["sous", "en dessous de"], "in": ["dans", "à l'intérieur de"], "next_to": ["à côté de", "près de"]}
VERB_FORMS = {"place": {"imp": ["pose", "place", "mets"], "pc": ["a posé", "a placé", "a mis"], "pp": ["posé", "placé", "mis"]},
              "fall": {"pc": ["est tombé", "a chuté"]}, "push": {"pc": ["a poussé"], "pp": ["poussé"]}}
HELD_OUT_FORMS = {"boule", "brique", "caisse", "rayon", "plancher", "mets"}     # never in the training captions
NONSENSE = ["flurpe", "zorvax", "blimpeux", "trakul", "vornite", "glissonne"]


def _elide(det, word):
    return "l'" + word if det in ("le", "la") and word[0] in "aeiouyéèêàâîôûh" else f"{det} {word}"


class LanguageGenerator:
    def __init__(self, seed=0, allow_held_out=False):
        self.rng = random.Random(seed)
        self.allow_held_out = allow_held_out
        self.last_mentioned: List[str] = []
        self.meanings: Dict[str, str] = {}          # for evaluation only

    def _word(self, kind):
        forms = [f for f in LEX[kind] if self.allow_held_out or f[0] not in HELD_OUT_FORMS]
        w, g = self.rng.choice(forms)
        self.meanings[w] = f"type:{kind}"
        return w, g

    def np(self, world, oid, definite=True, with_color=True, color=None):
        if oid in SUPPORT_HEIGHT:
            w, g = self._word(oid)
            return _elide("la" if g == "f" else "le", w) if definite else f"{'une' if g == 'f' else 'un'} {w}", g
        obj = world.objects[oid]
        w, g = self._word(obj.kind)
        det = ("la" if g == "f" else "le") if definite else ("une" if g == "f" else "un")
        text = _elide(det, w) if definite else f"{det} {w}"
        c = color or (obj.color if with_color else None)
        if c:
            text += " " + COLOR_FORMS[c][1 if g == "f" else 0]; self.meanings[COLOR_FORMS[c][1 if g == "f" else 0]] = f"color:{c}"
            if color is None: self.last_mentioned.append(oid)
        return text, g

    def rel(self, rel):
        form = self.rng.choice(REL_FORMS[rel])
        self.meanings[(form.split()[-2] if " " in form else form).replace("l'", "")] = f"rel:{rel}"
        return form

    def _join(self, prep, np):
        if prep == "de" or prep.endswith(" de"):
            base = prep[:-2]
            if np.startswith("le "): return f"{base}du {np[3:]}"
            if np.startswith("les "): return f"{base}des {np[4:]}"
        return f"{prep} {np}"

    def describe(self, world, key, value):
        """Sentence describing a proposition (possibly false)."""
        self.last_mentioned = []
        kind = key[0]
        o = key[1] if len(key) > 1 else None
        if kind == "color":
            subj, g = self.np(world, o, with_color=False)
            adj = COLOR_FORMS[value][1 if g == "f" else 0]
            return self.rng.choice([f"{subj} est {adj}", f"c'est {'une' if g == 'f' else 'un'} {subj.split()[-1] if ' ' in subj else subj[2:]} {adj}"])
        if kind == "loc":
            rel, target = value
            subj, _ = self.np(world, o)
            tgt, _ = self.np(world, target)
            prep = self.rel(rel)
            past = key[2] == 0
            verb = "était" if past else "est"
            templates = [f"{subj} {verb} {self._join(prep, tgt)}", f"{self._join(prep, tgt)}, il y {'avait' if past else 'a'} {self.np(world, o, definite=False)[0]}"]
            s = self.rng.choice(templates)
            return ("au début, " + s) if past and self.rng.random() < 0.5 else s
        if kind == "height":
            subj, _ = self.np(world, o)
            return self.rng.choice([f"{subj} est à {self.number(value)}", f"la hauteur {self._join('de', subj)} est de {self.number(value)}"])
        if kind == "owner":
            subj, _ = self.np(world, o)
            return self.rng.choice([f"{subj} appartient à {value.capitalize()}", f"{subj} est à {value.capitalize()}"])
        if kind == "event":
            return self.event(world, value)
        return ""

    def number(self, meters):
        style = self.rng.choice(["m", "cm", "mètres"])
        if style == "cm": return f"{int(round(meters * 100))} cm"
        text = f"{meters:g}".replace(".", ",")
        return f"{text} {'m' if style == 'm' else 'mètres'}"

    def event(self, world, e: Event):
        patient, g = self.np(world, e.patient)
        if e.action == "fall":
            verb = self.rng.choice(VERB_FORMS["fall"]["pc"])
            if g == "f" and verb.endswith("tombé"): verb += "e"
            return f"{patient} {verb}"
        target, _ = self.np(world, e.target)
        prep = self.rel(e.rel)
        if e.action == "push":
            agent, _ = self.np(world, e.agent)
            return self.rng.choice([f"{agent} a poussé {patient}", f"{patient} a été poussé{'e' if g == 'f' else ''} par {agent}"])
        agent = e.agent.capitalize()
        pc = self.rng.choice(VERB_FORMS["place"]["pc"])
        pp = pc.split()[1]
        return self.rng.choice([f"{agent} {pc} {patient} {self._join(prep, target)}",
                                f"{patient} a été {pp}{'e' if g == 'f' and not pp.endswith('s') else ''} {self._join(prep, target)} par {agent}"])


# ================================================================ learning captions (sentence + perceived structure)

def _desc(world, oid, with_color=True):
    if oid in SUPPORT_HEIGHT: return {"type": oid}
    obj = world.objects[oid]
    d = {"type": obj.kind}
    if with_color: d["color"] = obj.color
    return d


def captions(n, seed=0, allow_held_out=False):
    """(sentence, perceived structure) pairs from training worlds (true, not noisy).
    The structure is what a sensor sees (types, colours, relations, actions) — never words."""
    out = []
    rng = random.Random(seed)
    i = 0
    while len(out) < n:
        world = generate_world(100000 + seed * 1000 + i); i += 1
        gen = LanguageGenerator(seed=seed * 7 + i, allow_held_out=allow_held_out)
        o = rng.choice(world.movables)
        choice = rng.choice(["color", "loc", "loc", "loc0", "height", "owner", "event", "sequence", "count", "command"])
        if choice == "color":
            text = gen.describe(world, ("color", o), world.objects[o].color)
            s = {"kind": "color", "subject": _desc(world, o, False), "value": world.objects[o].color}
        elif choice in ("loc", "loc0"):
            t = 0 if choice == "loc0" else "now"
            rel, tgt = world.loc(o, t)
            text = gen.describe(world, ("loc", o, t), (rel, tgt))
            s = {"kind": "loc", "subject": _desc(world, o), "rel": rel, "target": _desc(world, tgt), "time": "past" if t == 0 else "now"}
        elif choice == "height":
            h = world.height(o)
            text = gen.describe(world, ("height", o, "now"), h)
            s = {"kind": "height", "subject": _desc(world, o), "value": h}
        elif choice == "owner":
            if not world.objects[o].owner: continue
            text = gen.describe(world, ("owner", o), world.objects[o].owner)
            s = {"kind": "owner", "subject": _desc(world, o), "value": world.objects[o].owner}
        elif choice == "event":
            if not world.events: continue
            e = world.events[0]
            text = gen.event(world, e)
            s = _event_struct(world, e)
        elif choice == "sequence":
            if len(world.events) < 2: continue
            e1, e2 = world.events[0], world.events[1]
            s1, s2 = gen.event(world, e1), gen.event(world, e2)
            conn = rng.choice(["puis", "après"])
            text = f"{s1} puis {s2}" if conn == "puis" else f"{s2} après que {s1}"
            s = {"kind": "sequence", "events": [_event_struct(world, e1), _event_struct(world, e2)]}
        elif choice == "command":
            # demonstration: imperative instruction + observed effect ("mets" stays out of learning)
            rel, tgt = rng.choice([("on", "table"), ("on", "floor"), ("on", "shelf")] + [("in", b) for b in world.movables if world.objects[b].kind == "box" and b != o])
            verb = rng.choice([v for v in VERB_FORMS["place"]["imp"] if allow_held_out or v not in HELD_OUT_FORMS])
            gen.meanings[verb] = "act:place"
            text = f"{verb} {gen.np(world, o)[0]} {gen._join(gen.rel(rel), gen.np(world, tgt)[0])}"
            s = {"kind": "event", "action": "place", "agent": None, "patient": _desc(world, o), "rel": rel, "target": _desc(world, tgt)}
        else:
            rel, tgt = rng.choice([("on", "table"), ("on", "floor"), ("on", "shelf")])
            count = sum(1 for m in world.movables if world.loc(m) == (rel, tgt))
            gen.last_mentioned = []
            tgt_np, _ = gen.np(world, tgt)
            text = f"il y a {NUMBERS[count]} objet{'s' if count > 1 else ''} {gen._join(gen.rel(rel), tgt_np)}"
            s = {"kind": "count", "rel": rel, "target": {"type": tgt}, "value": count}
        out.append({"text": text, "structure": s, "meanings": dict(gen.meanings)})
    return out


def _event_struct(world, e):
    agent = {"person": e.agent} if e.agent in PERSONS else (_desc(world, e.agent) if e.agent else None)
    return {"kind": "event", "action": e.action, "agent": agent, "patient": _desc(world, e.patient), "rel": e.rel,
            "target": _desc(world, e.target)}


# ================================================================ benchmark (references computed here)

ABSTAIN_ANSWERS = {"UNKNOWN", "UNKNOWN_WORD", "ASK"}


@dataclass
class Item:
    world_seed: int
    text: str
    gold: str
    intent: str
    category: str
    meta: Dict[str, Any] = field(default_factory=dict)


def _unique_kind(world, kind):
    return sum(1 for m in world.movables if world.objects[m].kind == kind) == 1


def _answer_loc(v): return f"{v[0]}:{v[1]}"


def ref_status(world, observations, oid, with_color=True):
    """Is a description (type [+ colour]) resolvable by an ideal agent? ok / ASK / UNKNOWN."""
    if oid in SUPPORT_HEIGHT: return "ok"
    kind = world.objects[oid].kind
    same = [m for m in world.movables if world.objects[m].kind == kind]
    color_known = knowable(world, observations, ("color", oid)) is not None
    if len(same) == 1: return "ok" if (not with_color or color_known) else "UNKNOWN"
    if not with_color: return "ASK"
    color = world.objects[oid].color
    if sum(1 for m in same if world.objects[m].color == color) > 1: return "ASK"
    others_known = all(knowable(world, observations, ("color", m)) is not None for m in same if m != oid)
    return "ok" if others_known else "UNKNOWN"


def benchmark_items(world: WorldTruth, observations, seed=0) -> List[Item]:
    rng = random.Random(seed * 131 + world.seed)
    gen = LanguageGenerator(seed=seed * 17 + world.seed, allow_held_out=True)
    items = []
    facts = world.facts()
    conflicting = lambda key: len({ob.value for ob in observations if ob.key == key}) > 1
    observed = lambda key: any(ob.key == key for ob in observations)

    def cat(key, gold):
        if gold in ("UNKNOWN",): return "partial" if any(ob.key[1:2] == key[1:2] for ob in observations if ob.key[0] != "type") else "unknown"
        return "noisy" if conflicting(key) else "correct"

    def add(text, gold, intent, category, **meta):
        items.append(Item(world.seed, text, gold, intent, category, {**meta, "meanings": dict(gen.meanings)}))

    def gated(gold, o, with_color=True):
        status = ref_status(world, observations, o, with_color)
        return gold if status == "ok" else status

    for o in world.movables:
        kind = world.objects[o].kind
        unique = _unique_kind(world, kind)
        # where / when
        for t in ("now", 0):
            key = ("loc", o, t)
            v = knowable(world, observations, key)
            gold = _answer_loc(v) if v else "UNKNOWN"
            subj, _ = gen.np(world, o)
            text = rng.choice([f"où est {subj} ?", f"où se trouve {subj} ?"]) if t == "now" else \
                   rng.choice([f"où était {subj} au début ?", f"au début, où était {subj} ?"])
            add(text, gated(gold, o), "where", "temporal" if t == 0 else cat(key, gold), subject=o, time=t, key=key)
        # colour (reference by type alone: ambiguous if several objects of the same type)
        subj, _ = gen.np(world, o, with_color=False)
        key = ("color", o)
        v = knowable(world, observations, key)
        gold = ("ASK" if not unique else (v if v else "UNKNOWN"))
        add(f"de quelle couleur est {subj} ?", gold, "color", "ambiguous" if not unique else cat(key, gold), subject=o, key=key)
        # height
        key = ("height", o, "now")
        v = knowable(world, observations, key)
        subj, _ = gen.np(world, o)
        add(rng.choice([f"à quelle hauteur est {subj} ?", f"quelle est la hauteur {gen._join('de', subj)} ?"]),
            gated(f"{v:g}" if v is not None else "UNKNOWN", o), "height", cat(key, "UNKNOWN" if v is None else "x"), subject=o, key=key)
        # true / false colour claims (type alone if unique)
        if unique:
            vc = knowable(world, observations, ("color", o))
            claimed = rng.choice(COLORS)
            subj, g = gen.np(world, o, with_color=False)
            gold = "UNKNOWN" if vc is None else ("SUPPORTED" if claimed == vc else "REFUTED")
            add(f"{subj} est {COLOR_FORMS[claimed][1 if g == 'f' else 0]}", gold, "claim",
                "false" if gold == "REFUTED" else cat(("color", o), gold), subject=o, key=("color", o), claimed=claimed)
        # position claim (true or false), sometimes with units
        rel, tgt = world.loc(o)
        if rng.random() < 0.5:
            rel, tgt = rng.choice([("on", "table"), ("on", "shelf"), ("on", "floor"), ("under", "table")])
        vl = knowable(world, observations, ("loc", o, "now"))
        subj, _ = gen.np(world, o); tnp, _ = gen.np(world, tgt)
        gold = "UNKNOWN" if vl is None else ("SUPPORTED" if vl == (rel, tgt) else "REFUTED")
        add(f"{subj} est {gen._join(gen.rel(rel), tnp)}", gated(gold, o), "claim", "false" if gold == "REFUTED" else cat(("loc", o, "now"), gold),
            subject=o, key=("loc", o, "now"), claimed=(rel, tgt))
        vh = knowable(world, observations, ("height", o, "now"))
        if vh is not None or rng.random() < 0.3:
            h = vh if (vh is not None and rng.random() < 0.5) else rng.choice([0.0, 1.0, 2.0])
            gold = "UNKNOWN" if vh is None else ("SUPPORTED" if abs(h - vh) < 1e-6 else "REFUTED")
            add(f"{gen.np(world, o)[0]} est à {gen.number(h)}", gated(gold, o), "claim", "false" if gold == "REFUTED" else "correct" if gold == "SUPPORTED" else "unknown",
                subject=o, key=("height", o, "now"), claimed=h)
        # unsupported ownership
        person = rng.choice(PERSONS)
        vo = knowable(world, observations, ("owner", o))
        gold = "UNKNOWN" if vo is None else ("SUPPORTED" if vo == person else "REFUTED")
        add(f"{gen.np(world, o)[0]} appartient à {person.capitalize()}", gated(gold, o), "claim", "unknown" if gold == "UNKNOWN" else "correct",
            subject=o, key=("owner", o), claimed=person)
    # ---- adversarial
    present = {(world.objects[m].kind, world.objects[m].color) for m in world.movables}
    missing = [(k, c) for k in KINDS for c in COLORS if (k, c) not in present]
    if missing:
        k, c = rng.choice(missing)
        same_kind = [m for m in world.movables if world.objects[m].kind == k]
        known_absent = all(knowable(world, observations, ("color", m)) is not None for m in same_kind)
        w, g = gen._word(k); adj = COLOR_FORMS[c][1 if g == "f" else 0]; gen.meanings[adj] = f"color:{c}"
        np_ = f"{_elide('la' if g == 'f' else 'le', w)} {adj}"
        gold = "INVALID_PREMISE" if known_absent else "UNKNOWN"
        add(f"pourquoi {np_} est-{'elle' if g == 'f' else 'il'} sur la table ?", gold, "why", "adversarial", kind="nonexistent_entity",
            absent=(k, c))
        add(f"où est {np_} ?", gold, "where", "adversarial", kind="nonexistent_entity", absent=(k, c))
    o = rng.choice(world.movables)
    vl = knowable(world, observations, ("loc", o, "now"))
    subj, g = gen.np(world, o)
    presup = ("on", "table")
    gold = gated("UNKNOWN" if vl is None else ("UNKNOWN_CAUSE" if vl == presup else "INVALID_PREMISE"), o)
    add(f"pourquoi {subj} est-{'elle' if g == 'f' else 'il'} sur la table ?", gold, "why", "adversarial", kind="false_presupposition",
        subject=o, key=("loc", o, "now"))
    subj, g = gen.np(world, o, with_color=False)
    add(f"{subj} est {rng.choice(NONSENSE)}", "UNKNOWN_WORD", "claim", "adversarial", kind="invented_word", subject=o)
    add(f"la table est dans {gen.np(world, o)[0]}", "REJECT", "claim", "adversarial", kind="impossible_relation", subject=o)
    obj = world.objects[o]
    other = rng.choice([c for c in COLORS if c != obj.color])
    subj, g = gen.np(world, o)
    add(f"{subj} est {COLOR_FORMS[other][1 if g == 'f' else 0]}", "REJECT", "claim", "adversarial", kind="internal_contradiction", subject=o)
    # ---- compositional commands
    items += command_items(world, observations, gen, rng)
    return items


def _valid_placement(world, patient, rel, target):
    if patient == target: return False
    if rel == "in": return target in world.objects and world.objects[target].kind == "box" and world.objects[patient].kind != "box"
    if rel == "on": return target in SUPPORT_HEIGHT or world.objects[target].kind == "box"
    if rel == "under": return target in ("table", "shelf")
    return rel == "next_to" and target not in SUPPORT_HEIGHT


def _effect(pairs): return ";".join(sorted(f"{p}:{r}:{t}" for p, r, t in pairs)) or "NONE"


def command_items(world, observations, gen, rng):
    items = []
    movables = world.movables
    box = next((m for m in movables if world.objects[m].kind == "box"), None)

    def add(text, gold, category, refs=(), **meta):
        ungated = gold
        for o in refs:
            status = ref_status(world, observations, o)
            if status != "ok" and gold not in ("ASK",): gold = status; break
        items.append(Item(world.seed, text, gold, "command", category,
                          {**meta, "refs": tuple(refs), "ungated": ungated, "meanings": dict(gen.meanings)}))

    a = rng.choice(movables)
    dests = [("on", "table"), ("on", "shelf"), ("on", "floor")] + ([("in", box)] if box else [])
    rel, tgt = rng.choice(dests)
    na, _ = gen.np(world, a); nt, _ = gen.np(world, tgt)
    verb = rng.choice(VERB_FORMS["place"]["imp"] + (["mets"] if gen.allow_held_out else []))
    ok = _valid_placement(world, a, rel, tgt)
    add(f"{verb} {na} {gen._join(gen.rel(rel), nt)}", _effect([(a, rel, tgt)]) if ok else "REJECT", "command", refs=(a, tgt))
    # sequence
    b = rng.choice([m for m in movables if m != a] or [a])
    rel2, tgt2 = rng.choice(dests)
    nb, _ = gen.np(world, b); nt2, _ = gen.np(world, tgt2)
    ok2 = ok and _valid_placement(world, b, rel2, tgt2)
    text = f"pose {na} {gen._join(gen.rel(rel), nt)} puis pose {nb} {gen._join(gen.rel(rel2), nt2)}"
    eff = {(a, rel, tgt)} if a != b else set()
    eff = [(a, rel, tgt)] + [(b, rel2, tgt2)] if a != b else [(a, rel2, tgt2)]
    add(text, _effect(eff) if ok2 else "REJECT", "command", refs=(a, tgt, b, tgt2), kind="sequence")
    # discourse reference (pronoun)
    subj, g = gen.np(world, a)
    pron = "-la" if g == "f" else "-le"
    add(f"regarde {subj}. pose{pron} {gen._join(gen.rel(rel), nt)}", _effect([(a, rel, tgt)]) if ok else "REJECT", "command",
        refs=(a, tgt), kind="pronoun")
    for c in [m for m in movables if m != a]:
        s2, g2 = gen.np(world, c)
        if g2 == g:
            add(f"regarde {subj} et {s2}. pose{pron} sur la table", "ASK", "ambiguous", kind="ambiguous_pronoun")
            break
    # condition
    cond_obj = rng.choice(movables)
    crel, ctgt = rng.choice([("on", "table"), ("on", "floor"), ("on", "shelf")])
    known = knowable(world, observations, ("loc", cond_obj, "now"))
    cond_meta = {"cond": (cond_obj, (crel, ctgt)), "if_true": _effect([(a, rel, tgt)]) if ok else "REJECT"}
    if known is None: gold = "UNKNOWN"
    elif known == (crel, ctgt): gold = _effect([(a, rel, tgt)]) if ok else "REJECT"
    else: gold = "NONE"
    add(f"si {gen.np(world, cond_obj)[0]} est {gen._join(gen.rel(crel), gen.np(world, ctgt)[0])}, pose {na} {gen._join(gen.rel(rel), nt)}",
        gold, "command", refs=(cond_obj, a, tgt), kind="condition", **cond_meta)
    # relative clause: "l'objet qui est sous l'étagère"
    rrel, rtgt = rng.choice([("under", "shelf"), ("under", "table"), ("on", "shelf"), ("on", "table")])
    matches = [m for m in movables if knowable(world, observations, ("loc", m, "now")) == (rrel, rtgt)]
    unknown = [m for m in movables if knowable(world, observations, ("loc", m, "now")) is None]
    if len(matches) == 1 and not unknown: gold_rel = matches[0]
    elif len(matches) > 1: gold_rel = "ASK"
    else: gold_rel = "UNKNOWN"
    target = box or "table"; trel = "in" if box else "on"
    gold = gold_rel if gold_rel in ("ASK", "UNKNOWN") else (_effect([(gold_rel, trel, target)]) if _valid_placement(world, gold_rel, trel, target) else "REJECT")
    add(f"pose {gen._join(gen.rel(trel), gen.np(world, target)[0])} l'objet qui est {gen._join(gen.rel(rrel), gen.np(world, rtgt)[0])}",
        gold, "command", kind="relative", relative=((rrel, rtgt), trel, target))
    # ambiguous attachment: "pose la balle dans la boîte sur la table"
    if box:
        p = rng.choice([m for m in movables if m != box])
        np_p, _ = gen.np(world, p); np_box, _ = gen.np(world, box)
        in_box = knowable(world, observations, ("loc", p, "now"))
        box_on = knowable(world, observations, ("loc", box, "now"))
        r1 = in_box == ("in", box)                      # "la balle (qui est) dans la boîte" -> on the table
        r2 = box_on == ("on", "table")                  # "dans la boîte (qui est) sur la table"
        if r1 and r2: gold = "ASK"
        elif r1: gold = _effect([(p, "on", "table")])
        elif r2: gold = _effect([(p, "in", box)])
        else: gold = "UNKNOWN"
        add(f"pose {np_p} dans {np_box} sur la table", gold, "ambiguous" if gold == "ASK" else "command", refs=(p, box), kind="attachment",
            attachment=(p, box))
    return items


def build_dataset(n_worlds=25, seed=0):
    """Worlds -> observations -> items. No dependency on WCE."""
    data = []
    for i in range(n_worlds):
        world = generate_world(seed * 100000 + i)
        obs = emit_observations(world, seed)
        data.append({"world": world, "observations": obs, "items": benchmark_items(world, obs, seed)})
    return data


def split_of(index, n):
    """Fixed split per world: 40 % train, 15 % calibration, 15 % validation, 30 % held-out test."""
    f = index / n
    return "train" if f < 0.4 else "calibration" if f < 0.55 else "validation" if f < 0.7 else "test"

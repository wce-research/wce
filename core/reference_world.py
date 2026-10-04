"""Épisodes de discours pour la référence (générateur INDÉPENDANT : aucun import WCE).

Chaque épisode : des entités cachées, des faits donnés à l'agent (partiels), des tours de parole.
La référence de chaque mention est fixée PAR CONSTRUCTION du gabarit (entité, AMBIGUOUS,
NONEXISTENT, HYPOTHETICAL). Un simulateur répond aux demandes de clarification.
Domaines : physique, inventaire, famille fictive.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.independent_world import COLOR_FORMS

COLORS4 = ("red", "blue", "green", "yellow")
DOMAINS = {
    "physical": {"kinds": {"cube": ("cube", "m"), "ball": ("balle", "f"), "block": ("bloc", "m"), "cone": ("cône", "m")},
                 "attr": "color", "places": {"table": ("sur", "la table"), "shelf": ("sur", "l'étagère"), "floor": ("sur", "le sol")}},
    "inventory": {"kinds": {"pen": ("stylo", "m"), "book": ("livre", "m"), "key": ("clé", "f"), "card": ("carte", "f")},
                  "attr": "color", "places": {"l_courses": ("dans", "la liste courses"), "l_voyage": ("dans", "la liste voyage")}},
    "family": {"kinds": {"son": ("fils", "m"), "daughter": ("fille", "f")}, "attr": "rank", "places": {}},
}
RANK_FORMS = {"oldest": ("aîné", "aînée"), "youngest": ("cadet", "cadette"), "middle": ("second", "seconde")}
ATTR_VALUES = {"color": COLORS4, "rank": ("oldest", "youngest", "middle")}
CATEGORIES = ("trivial", "same_type_attr", "by_position", "relational", "pronoun", "other", "past_reference",
              "real_ambiguity", "false_presupposition", "introduced_later", "cross_sentence", "clarification",
              "dialogue", "hypothetical")


@dataclass
class REntity:
    id: str
    kind: str
    attr: Optional[str]
    loc: Optional[Tuple[str, str]] = None        # état au début du discours
    loc0: Optional[Tuple[str, str]] = None       # état passé (avant le discours)


@dataclass
class Turn:
    text: str
    mentions: List[Dict[str, Any]]               # [{"head": mot, "gold": entité | AMBIGUOUS | ...}] dans l'ordre du texte
    kind: str = "statement"                      # statement | question | command
    clarify_target: Optional[str] = None         # entité visée (pour le simulateur)


@dataclass
class Episode:
    domain: str
    category: str
    entities: Dict[str, REntity]
    given: List[Tuple[Tuple, Any]]
    turns: List[Turn]
    meta: Dict[str, Any] = field(default_factory=dict)


# ================================================================ surface

def word(domain, kind): return DOMAINS[domain]["kinds"][kind]


def attr_form(domain, value, gender):
    if DOMAINS[domain]["attr"] == "color": return COLOR_FORMS[value][1 if gender == "f" else 0]
    return RANK_FORMS[value][1 if gender == "f" else 0]


def det(gender, definite=True, vowel=False):
    if definite: return "l'" if vowel else ("la" if gender == "f" else "le")
    return "une" if gender == "f" else "un"


def np(domain, e: REntity, definite=True, with_attr=True):
    w, g = word(domain, e.kind)
    d = det(g, definite, w[0] in "aeiouéèêô")
    text = f"{d}{w}" if d.endswith("'") else f"{d} {w}"
    if with_attr and e.attr: text += " " + attr_form(domain, e.attr, g)
    return text


def place_phrase(domain, place): rel, p = DOMAINS[domain]["places"][place]; return f"{rel} {p}"


def pron(gender, clitic=False): return ("-la" if gender == "f" else "-le") if clitic else ("elle" if gender == "f" else "il")


def cap(s): return s[0].upper() + s[1:]


def m(head, gold, **kw): return {"head": head, "gold": gold, **kw}


# ================================================================ scènes

def scene(rng, domain, n_same=2, extra=1, distinct_attr=True):
    kinds = list(DOMAINS[domain]["kinds"])
    main = rng.choice(kinds)
    values = list(ATTR_VALUES[DOMAINS[domain]["attr"]]); rng.shuffle(values)
    places = list(DOMAINS[domain]["places"])
    ents = {}
    for i in range(n_same):
        ents[f"e{i + 1}"] = REntity(f"e{i + 1}", main, values[i] if distinct_attr else values[0],
                                    (("on" if domain == "physical" else "in"), rng.choice(places)) if places else None)
    others = [k for k in kinds if k != main]
    for j in range(extra):
        k = rng.choice(others)
        ents[f"e{n_same + j + 1}"] = REntity(f"e{n_same + j + 1}", k, values[(n_same + j) % len(values)],
                                              (("on" if domain == "physical" else "in"), rng.choice(places)) if places else None)
    return main, ents


PLACE_FACTS = {"physical": [(("type", "table"), "table"), (("type", "shelf"), "shelf"), (("type", "floor"), "floor")],
               "inventory": [(("type", "l_courses"), "list"), (("attr", "l_courses"), "courses"),
                             (("type", "l_voyage"), "list"), (("attr", "l_voyage"), "voyage")], "family": []}


def given_types(ents):
    domain = next((d for d, spec in DOMAINS.items() if any(e.kind in spec["kinds"] for e in ents.values())), "physical")
    return PLACE_FACTS[domain] + [(("type", e.id), e.kind) for e in ents.values()]
def given_attrs(ents, ids=None): return [(("attr", e.id), e.attr) for e in ents.values() if ids is None or e.id in ids]
def given_locs(ents, ids=None): return [(("loc", e.id, "now"), e.loc) for e in ents.values() if e.loc and (ids is None or e.id in ids)]


# ================================================================ gabarits (référence fixée par construction)

def build(category, domain, rng) -> Optional[Episode]:
    places = list(DOMAINS[domain]["places"])
    main, ents = scene(rng, domain)
    e1, e2 = ents["e1"], ents["e2"]
    w, g = word(domain, main)
    ep = lambda turns, given, **meta: Episode(domain, category, ents, given_types(ents) + given, turns, meta)

    if category == "trivial":
        solo = ents["e3"]; sw, sg = word(domain, solo.kind)
        pred = f" est {place_phrase(domain, solo.loc[1])}" if solo.loc else f" est {attr_form(domain, solo.attr, sg)}"
        return ep([Turn(cap(np(domain, solo, with_attr=False)) + pred + ".", [m(sw, "e3")])], given_attrs(ents, ["e3"]))
    if category == "same_type_attr":
        return ep([Turn(cap(np(domain, e1)) + (" tombe." if domain == "physical" else " est là."), [m(w, "e1")])], given_attrs(ents))
    if category == "by_position" and places:
        e2.loc = (e2.loc[0], next(p for p in places if p != e1.loc[1]))
        return ep([Turn(f"{cap(np(domain, e1, with_attr=False))} {place_phrase(domain, e1.loc[1])} est {attr_form(domain, e1.attr, g)}.",
                        [m(w, "e1")])], given_locs(ents, ["e1", "e2"]))
    if category == "relational" and domain == "physical":
        box = REntity("b1", "box", rng.choice(COLORS4), ("on", rng.choice(places)))
        ents["b1"] = box
        e1.loc, e2.loc = ("in", "b1"), ("on", rng.choice(places))
        return Episode(domain, category, ents, given_types(ents) + given_locs(ents, ["e1", "e2"]),
                       [Turn(f"{cap(np(domain, e1, with_attr=False))} dans la boîte est {attr_form(domain, e1.attr, g)}.", [m(w, "e1")])])
    if category == "pronoun":
        fem = next((e for e in ents.values() if word(domain, e.kind)[1] != g), None)
        if fem is None: return None
        fw, fg = word(domain, fem.kind)
        return ep([Turn(cap(np(domain, e1)) + " est là.", [m(w, "e1")]),
                   Turn(cap(np(domain, fem)) + " est là aussi.", [m(fw, fem.id)]),
                   Turn(cap(pron(g)) + (" tombe." if domain == "physical" else " est important."), [m(pron(g), "e1")])], given_attrs(ents))
    if category == "other":
        second = " reste " + place_phrase(domain, places[0]) + "." if places else " est là."
        return ep([Turn(cap(np(domain, e1)) + (" tombe." if domain == "physical" else " est là."), [m(w, "e1")]),
                   Turn("L'autre" + second, [m("autre", "e2")])], given_attrs(ents))
    if category == "past_reference" and places:
        p1 = places[0]; p2 = places[1] if len(places) > 1 else places[0]
        e1.loc0, e2.loc0 = (e1.loc[0], p1), (e2.loc[0], p2)
        if p1 == p2: return None
        return Episode(domain, category, ents, given_types(ents) + [(("loc", "e1", 0), e1.loc0), (("loc", "e2", 0), e2.loc0)],
                       [Turn(f"{cap(np(domain, e1, with_attr=False))} qui était {place_phrase(domain, p1)} est {attr_form(domain, e1.attr, g)}.",
                             [m(w, "e1")])])
    if category == "real_ambiguity":
        return ep([Turn(cap(np(domain, e1, with_attr=False)) + (" tombe." if domain == "physical" else " est là."), [m(w, "AMBIGUOUS", intended="e1")])], [])
    if category == "false_presupposition":
        used = {e.attr for e in ents.values() if e.kind == main}
        other = next((v for v in ATTR_VALUES[DOMAINS[domain]["attr"]] if v not in used), None)
        if other is None: return None
        fake = REntity("x", main, other)
        text = f"Pourquoi {np(domain, fake)} est-{'elle' if g == 'f' else 'il'} là ?"
        return ep([Turn(text, [m(w, "NONEXISTENT")], kind="question")], given_attrs(ents))
    if category == "introduced_later":
        return ep([Turn(cap(np(domain, e1, definite=False, with_attr=False)) + " est là.", [m(w, "e1", final=True)]),
                   Turn(f"{cap(pron(g))} est {attr_form(domain, e1.attr, g)}.", [m(pron(g), "e1")])], given_attrs(ents))
    if category == "cross_sentence":
        fall = " tombe." if domain == "physical" else " est là."
        stay = (" reste " + place_phrase(domain, e2.loc[1]) + ".") if places else " reste là."   # position réelle de e2
        if places: e1.loc = (e1.loc[0], places[0])        # la phrase décrit la position RÉELLE de l'entité visée
        return ep([Turn(cap(np(domain, e1, definite=False, with_attr=False)) + (" " + place_phrase(domain, places[0]) if places else "") + " est là.",
                        [m(w, "AMBIGUOUS", intended="e1")]),
                   Turn(cap(np(domain, e1)) + fall, [m(w, "e1")]),
                   Turn("L'autre" + stay, [m("autre", "e2")]),
                   Turn(f"Prends{pron(word(domain, e2.kind)[1], True)}.", [m(pron(word(domain, e2.kind)[1], True), "e2")], kind="command")],
                  given_attrs(ents))
    if category == "clarification":
        target = rng.choice([e1, e2])
        return ep([Turn(f"Prends {np(domain, e1, with_attr=False)}.", [m(w, "AMBIGUOUS")], kind="command", clarify_target=target.id)],
                  given_attrs(ents), resolved_after_clarification=target.id)
    if category == "dialogue":
        plural = {"cube": "cubes", "ball": "balles", "block": "blocs", "cone": "cônes", "pen": "stylos", "book": "livres", "key": "clés",
                  "card": "cartes", "son": "fils", "daughter": "filles"}[main]
        adj = attr_form(domain, e1.attr, g)
        return ep([Turn(f"Il y a deux {plural}.", []),
                   Turn(f"{cap(det(g))} {adj} est là.", [m(adj, "e1")]),
                   Turn(f"Prends{pron(g, True)}.", [m(pron(g, True), "e1")], kind="command"),
                   Turn(f"Regarde{pron(g, True)} encore.", [m(pron(g, True), "e1")], kind="command")], given_attrs(ents))
    if category == "hypothetical":
        known = {e.attr for e in ents.values()}
        other = next((v for v in ATTR_VALUES[DOMAINS[domain]["attr"]] if v not in known), None)
        if other is None: return None
        fake = REntity("x", main, other)
        return ep([Turn(cap(np(domain, fake, definite=False)) + " est là.", [m(w, "HYPOTHETICAL")])], given_attrs(ents))
    return None


def long_episode(n_sentences, domain="physical", seed=0):
    """Discours de N phrases : 2 entités de même type + 1 autre ; la référence de chaque mention est suivie par construction."""
    rng = random.Random(seed)
    main, ents = scene(rng, domain, n_same=2, extra=1)
    ids = list(ents)
    turns, last_subject = [], None
    for s in range(n_sentences):
        target = ents[rng.choice(ids)]
        tw, tg = word(domain, target.kind)
        others_same_gender = [e for e in ents.values() if e.id != target.id and word(domain, e.kind)[1] == tg]
        options = ["attr"]
        if last_subject == target.id: options.append("pronoun")
        if last_subject and last_subject != target.id and ents[last_subject].kind == target.kind: options.append("other")
        if sum(1 for e in ents.values() if e.kind == target.kind) == 1: options.append("plain")
        mode = rng.choice(options)
        if mode == "pronoun" and others_same_gender and turns and len(turns[-1].mentions) > 1: mode = "attr"
        pred = rng.choice([" est là.", " reste là."] + ([" tombe."] if domain == "physical" else []))
        if mode == "attr": text, head = cap(np(domain, target)) + pred, tw
        elif mode == "plain": text, head = cap(np(domain, target, with_attr=False)) + pred, tw
        elif mode == "pronoun": text, head = cap(pron(tg)) + pred, pron(tg)
        else: text, head = "L'autre" + pred, "autre"
        turns.append(Turn(text, [m(head, target.id, mode=mode)]))
        last_subject = target.id
    return Episode(domain, "long", ents, given_types(ents) + given_attrs(ents), turns, {"length": n_sentences})


def reference_benchmark(n_per_category=12, seed=0, domains=("physical", "inventory", "family"), noise=0.0):
    """noise : probabilité qu'un attribut DONNÉ à l'agent soit faux (test de robustesse ; le texte reste vrai)."""
    rng = random.Random(seed)
    episodes = []
    for domain in domains:
        for cat in CATEGORIES:
            made = tries = 0
            while made < n_per_category and tries < 10 * n_per_category:
                tries += 1
                e = build(cat, domain, rng)
                if e is None: continue
                if noise:
                    values = ATTR_VALUES[DOMAINS[domain]["attr"]]
                    noisy = []
                    for k, val in e.given:
                        if k[0] == "attr" and k[1].startswith("e") and rng.random() < noise:
                            val = rng.choice([v for v in values if v != val])
                        noisy.append((k, val))
                    e.given = noisy
                episodes.append(e); made += 1
    return episodes


class ClarificationSimulator:
    """Utilisateur simulé : répond à « Quel X ? » en donnant l'attribut de l'entité visée."""

    def __init__(self, episode: Episode): self.ep = episode; self.questions = 0

    def answer(self, question, target):
        self.questions += 1
        e = self.ep.entities[target]; w, g = word(self.ep.domain, e.kind)
        return f"{det(g)} {attr_form(self.ep.domain, e.attr, g)}"

    def reveal(self, key):
        """Réponse à une demande d'information (apprentissage actif d'identité)."""
        self.questions += 1
        e = self.ep.entities.get(key[1])
        if e is None: return None
        return e.attr if key[0] == "attr" else e.loc


def family_captions(n=120, seed=0):
    """Légendes (phrase, structure perçue) pour le domaine famille — mêmes conventions que les autres légendes."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        kind = rng.choice(["son", "daughter"]); rank = rng.choice(list(RANK_FORMS))
        w, g = word("family", kind); a = RANK_FORMS[rank][1 if g == "f" else 0]
        text = rng.choice([f"{det(g)} {w} est {a}", f"c'est {det(g, False)} {w} {a}", f"{det(g)} {w} {a} est là"])
        out.append({"text": text, "structure": {"kind": "color", "subject": {"type": kind}, "value": rank}})
    return out

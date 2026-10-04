"""Pont vers la langue : micro-corpus français -> mêmes primitives WCE, même mécanisme.

Le mécanisme est celui des commandes symboliques (REV, ROT...) :
une expérience (phrase, avant, après) est EXPLIQUÉE par une recherche sur les primitives
connues ; le mot est alors lié à la primitive qui explique l'observation (lexique = routage).
Un mot jamais observé reste inconnu : le système ne devine pas.
"""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from core.invariants import Invariant, InvariantSet
from core.planner import StrategyLibrary
from core.reasoning import Goal
from core.symbolic import PRIMITIVES, decode, encode, search_programs
from core.transformations import DeclarativeTransformationSpec, TransformationDomain
from core.world import Entity, World
from language.lexer import tokenize

DETERMINERS = {"la", "le", "l", "les", "un", "une"}
KNOWN_NOUNS = {"balle", "bloc", "table", "sol"}


# ================================================================ primitives physiques (specs WCE)

def fall_spec(entity):
    from core.action import LiftAction
    return DeclarativeTransformationSpec(
        name="FALL", domain=TransformationDomain.PHYSICS, parameters={"entity": entity, "source_name": "language"},
        preserves=InvariantSet({Invariant.ENTITY_IDENTITY, Invariant.MASS}), changes={"y"},
        compiler=lambda world: LiftAction(entity, 0.0)._compile_executable(world))


def rise_spec(entity):
    from core.action import LiftAction
    return DeclarativeTransformationSpec(
        name="RISE", domain=TransformationDomain.PHYSICS, parameters={"entity": entity, "source_name": "language"},
        preserves=InvariantSet({Invariant.ENTITY_IDENTITY, Invariant.MASS}), changes={"y"},
        compiler=lambda world: LiftAction(entity, world.entities[entity].properties["y"] + 1.0)._compile_executable(world))


PHYSICS_PRIMITIVES = {"FALL": fall_spec, "RISE": rise_spec}


def make_world(heights: Dict[str, float]) -> World:
    world = World()
    for name, y in heights.items(): world.add_entity(Entity(name, "objet", {"y": float(y), "masse": 1.0}))
    return world


def apply_spec(world, spec):
    from core.transformation import apply
    apply(world, spec.compile(world))
    return world


# ================================================================ analyse

@dataclass
class Parsed:
    verb: str
    entity: Optional[str] = None
    sequence: Optional[str] = None


def parse(sentence, entities=()):
    tokens = [t for t in tokenize(sentence) if t not in DETERMINERS]
    nouns = set(entities) | KNOWN_NOUNS
    entity = next((t for t in tokens if t in nouns), None)
    sequence = next((t for t in tokens if t and set(t) <= set("abcdefgh") and len(t) == 5), None)
    rest = [t for t in tokens if t != entity and t != sequence]
    return Parsed(verb=rest[0] if rest else "", entity=entity, sequence=sequence)


# ================================================================ pont

@dataclass
class Interpretation:
    sentence: str
    verb: str
    skill: Optional[str]
    known: bool
    goal: Optional[Goal] = None
    reason: str = ""


class LanguageBridge:
    def __init__(self):
        self.lexicon: Dict[str, Counter] = {}
        self.library = StrategyLibrary()
        self.log: List[str] = []

    # ---- apprentissage par l'expérience ---------------------------------
    def observe_physics(self, sentence, before: World, after: World):
        p = parse(sentence, before.entities)
        if p.entity is None: return None
        target_y = after.entities[p.entity].properties["y"]
        explaining = []
        for name, make in PHYSICS_PRIMITIVES.items():
            trial = copy.deepcopy(before)
            try: apply_spec(trial, make(p.entity))
            except Exception: continue
            if trial.entities[p.entity].properties["y"] == target_y and before.entities[p.entity].properties["y"] != target_y:
                explaining.append(name)
        return self._bind(p.verb, explaining, sentence)

    def observe_sequence(self, sentence, output: str):
        p = parse(sentence)
        if p.sequence is None: return None
        result = search_programs(encode(p.sequence).unsqueeze(0), encode(output).unsqueeze(0),
                                 list(PRIMITIVES.values()), max_depth=1, first=False)
        return self._bind(p.verb, result.consistent, sentence)

    def _bind(self, verb, explaining, sentence):
        if len(explaining) != 1:
            self.log.append(f"'{sentence}': {len(explaining)} explication(s) -> pas de liaison")
            return None
        skill = explaining[0]
        self.lexicon.setdefault(verb, Counter())[skill] += 1
        self.library.store(Goal("verb", verb), skill)
        self.log.append(f"'{sentence}': '{verb}' -> {skill}")
        return skill

    # ---- interprétation / exécution -------------------------------------
    def interpret(self, sentence, entities=()):
        p = parse(sentence, entities)
        if p.verb in self.lexicon:
            skill = self.lexicon[p.verb].most_common(1)[0][0]
            return Interpretation(sentence, p.verb, skill, True, Goal(skill, p.entity or p.sequence),
                                  f"'{p.verb}' appris sur {sum(self.lexicon[p.verb].values())} expérience(s)")
        return Interpretation(sentence, p.verb, None, False, None, f"verbe inconnu '{p.verb}' : aucune expérience")

    def execute(self, sentence, world: World = None):
        interp = self.interpret(sentence, world.entities if world is not None else ())
        if not interp.known: return interp, None
        p = parse(sentence, world.entities if world is not None else ())
        if interp.skill in PHYSICS_PRIMITIVES:
            apply_spec(world, PHYSICS_PRIMITIVES[interp.skill](p.entity))
            return interp, world.entities[p.entity].properties["y"]
        return interp, decode(PRIMITIVES[interp.skill].apply(encode(p.sequence).unsqueeze(0))[0])


def default_corpus():
    """Expériences d'apprentissage (phrase, monde avant, monde après)."""
    return [("la balle tombe", {"balle": 2.0, "bloc": 1.0}, "balle", "FALL"),
            ("la balle monte", {"balle": 0.0, "bloc": 1.0}, "balle", "RISE"),
            ("le bloc tombe", {"balle": 0.0, "bloc": 3.0}, "bloc", "FALL"),
            ("le bloc monte", {"balle": 0.0, "bloc": 0.0}, "bloc", "RISE")]


def observed_transition(heights, entity, primitive):
    before = make_world(heights)
    after = apply_spec(copy.deepcopy(before), PHYSICS_PRIMITIVES[primitive](entity))
    return before, after

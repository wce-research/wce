"""Extensions of the independent environment (still WITHOUT any WCE import).

- third domain: inventory (objects stored in named lists);
- real polysemy: "range" = place (physical) / assign to a list (inventory);
- verbal construction "contient" (inverted roles);
- sentences with 2-3 clauses (conjunctions, chained commands);
- prefix "re-" (repose, replace) for lexical composition;
- held-out synonyms PRESENT in the textual observations (never in the captions),
  hence learnable by grounding — and only by grounding;
- separate seeds: world, language, splits.
"""
from __future__ import annotations

import random
from typing import Dict, List

from core.independent_world import (
    COLOR_FORMS, COLORS, HELD_OUT_FORMS, LEX, SOURCE_BY_NAME, SOURCES, SUPPORT_HEIGHT, Item, LanguageGenerator,
    Obj, Observation, Source, WorldTruth, _corrupt, _desc, _effect, _elide, benchmark_items, captions,
    generate_world, knowable, ref_status,
)

ITEM_KINDS = ("pen", "book", "key", "card")
LIST_NAMES = ("courses", "voyage", "travail")
EXT_LEX = {**LEX, "pen": [("stylo", "m")], "book": [("livre", "m"), ("bouquin", "m")], "key": [("clé", "f")],
           "card": [("carte", "f")], "list": [("liste", "f")]}
EXT_HELD_OUT = set(HELD_OUT_FORMS) | {"bouquin"}
INVENTORY_DB = Source("inventory_db", 0.97, 0.6, "structured")
SOURCE_BY_NAME.setdefault("inventory_db", INVENTORY_DB)
TEXT_SOURCES = tuple(s for s in SOURCES if s.modality == "text")
GOLD_MEANING = {w: f"type:{k}" for k, forms in EXT_LEX.items() for w, _ in forms}     # evaluation only


class ExtWorld(WorldTruth):
    """Physical world + inventory. Inventory objects are not physical."""

    @property
    def movables(self): return [o for o, v in self.objects.items() if o not in SUPPORT_HEIGHT and v.kind not in ITEM_KINDS + ("list",)]
    @property
    def items(self): return [o for o, v in self.objects.items() if v.kind in ITEM_KINDS]
    @property
    def lists(self): return [o for o, v in self.objects.items() if v.kind == "list"]

    def height(self, oid, time="now", depth=0):
        if oid in self.items: return 0.0
        return super().height(oid, time, depth)

    def facts(self):
        out = super().facts()
        for o in self.items:
            obj = self.objects[o]
            out[("type", o)] = obj.kind; out[("color", o)] = obj.color; out[("owner", o)] = None
            out[("loc", o, 0)] = self.loc(o, 0); out[("loc", o, "now")] = self.loc(o, "now")
        for l in self.lists: out[("type", l)] = "list"; out[("color", l)] = self.objects[l].color
        return out


def generate_world_ext(seed, inventory=True) -> ExtWorld:
    base = generate_world(seed)
    world = ExtWorld(base.seed, dict(base.objects), dict(base.loc0), list(base.events))
    if not inventory: return world
    rng = random.Random(seed * 13 + 777)
    names = rng.sample(LIST_NAMES, rng.choice([2, 3]))
    for i, name in enumerate(names): world.objects[f"l{i + 1}"] = Obj(f"l{i + 1}", "list", name)
    for i in range(rng.randint(2, 3)):
        oid = f"i{i + 1}"
        world.objects[oid] = Obj(oid, rng.choice(ITEM_KINDS), rng.choice(COLORS))
        world.loc0[oid] = ("in", rng.choice(world.lists))
    return world


class ExtLanguageGenerator(LanguageGenerator):
    def _word(self, kind):
        forms = [f for f in EXT_LEX[kind] if self.allow_held_out or f[0] not in EXT_HELD_OUT]
        w, g = self.rng.choice(forms)
        self.meanings[w] = f"type:{kind}"
        return w, g

    def np(self, world, oid, definite=True, with_color=True, color=None):
        obj = world.objects.get(oid)
        if obj is not None and obj.kind == "list":
            self.meanings["liste"] = "type:list"; self.meanings[obj.color] = f"color:{obj.color}"
            return (f"la liste {obj.color}" if definite else f"une liste {obj.color}"), "f"
        return super().np(world, oid, definite, with_color, color)

    def describe(self, world, key, value):
        if key[0] == "loc" and value[0] == "in" and self.rng.random() < 0.35:
            self.last_mentioned = []
            subj, _ = self.np(world, key[1]); tgt, _ = self.np(world, value[1])
            self.meanings["contient"] = "rel:in"
            verb = "contenait" if key[2] == 0 else "contient"
            return f"{tgt} {verb} {subj}"                               # inverted roles: target first
        return super().describe(world, key, value)


def emit_observations_ext(world: ExtWorld, seed=0, lang_seed=None) -> List[Observation]:
    """Physical observations + inventory; the texts may contain never-captioned synonyms."""
    lang_seed = seed if lang_seed is None else lang_seed
    rng = random.Random(seed * 7919 + world.seed)
    gen = ExtLanguageGenerator(seed=lang_seed * 31 + world.seed, allow_held_out=True)
    obs = [Observation("sensor_A", ("type", o), world.objects[o].kind, 0) for o in world.movables]
    obs += [Observation("inventory_db", ("type", o), world.objects[o].kind, 0) for o in world.items + world.lists]
    obs += [Observation("inventory_db", ("color", l), world.objects[l].color, 0) for l in world.lists]
    inventory = set(world.items)
    for key, value in world.facts().items():
        kind = key[0]
        if kind == "type" or (kind == "owner" and value is None) or (kind == "color" and key[1] in world.lists): continue
        sources = ((INVENTORY_DB,) + TEXT_SOURCES) if key[1] in inventory else SOURCES
        for src in sources:
            coverage = src.coverage * (0.3 if kind == "owner" else 1.0)
            if kind == "event" and src.modality == "structured": coverage *= 0.8
            if rng.random() >= coverage: continue
            if kind == "event" or rng.random() < src.reliability: v = value
            elif key[1] in inventory and kind == "loc": v = ("in", rng.choice([l for l in world.lists if ("in", l) != value] or world.lists))
            else: v = _corrupt(world, key, value, rng)
            time = key[2] if kind in ("loc", "height") else 0
            text = gen.describe(world, key, v) if src.modality == "text" else None
            obs.append(Observation(src.name, key, v, time if time != "now" else 1, text))
            if text is not None:
                for o in gen.last_mentioned:
                    pool = world.items if o in inventory else world.movables
                    if sum(1 for m in pool if world.objects[m].kind == world.objects[o].kind) == 1:
                        obs.append(Observation(src.name, ("color", o), world.objects[o].color, 0, None, implied=True))
    return obs


# ================================================================ extended captions

def captions_ext(n, seed=0, inventory=True, polysemy=True):
    """Physical captions + inventory + "range" in its two senses + "contient"."""
    if not inventory and not polysemy: return captions(n, seed=seed)       # nothing extended to produce
    base = captions(int(n * 0.6), seed=seed)
    rng = random.Random(seed + 4242)
    out = list(base)
    i = 0
    while len(out) < n:
        world = generate_world_ext(200000 + seed * 1000 + i, inventory=inventory); i += 1
        gen = ExtLanguageGenerator(seed=seed * 11 + i)
        choice = rng.choice(["item_loc", "item_loc", "item_cmd", "range_physical", "contains"]) if inventory else "range_physical"
        if choice in ("item_loc", "contains") and world.items:
            o = rng.choice(world.items); rel, tgt = world.loc(o)
            text = gen.describe(world, ("loc", o, "now"), (rel, tgt)) if choice == "item_loc" else \
                   f"{gen.np(world, tgt)[0]} contient {gen.np(world, o)[0]}"
            s = {"kind": "loc", "subject": _desc(world, o), "rel": rel, "target": {"type": "list", "color": world.objects[tgt].color}, "time": "now"}
        elif choice == "item_cmd" and world.items and polysemy:
            o = rng.choice(world.items); l = rng.choice(world.lists)
            text = f"range {gen.np(world, o)[0]} dans {gen.np(world, l)[0]}"
            s = {"kind": "event", "action": "assign", "agent": None, "patient": _desc(world, o), "rel": "in",
                 "target": {"type": "list", "color": world.objects[l].color}}
        elif choice == "range_physical" and polysemy:
            o = rng.choice(world.movables)
            box = [b for b in world.movables if world.objects[b].kind == "box" and b != o]
            rel, tgt = rng.choice([("on", "table"), ("on", "shelf"), ("on", "floor")] + [("in", b) for b in box])
            text = f"range {gen.np(world, o)[0]} {gen._join(gen.rel(rel), gen.np(world, tgt)[0])}"
            s = {"kind": "event", "action": "place", "agent": None, "patient": _desc(world, o), "rel": rel, "target": _desc(world, tgt)}
        else:
            continue
        out.append({"text": text, "structure": s, "meanings": dict(gen.meanings)})
    return out


# ================================================================ extended benchmark

def _gold_claim(world, obs, key, claimed):
    v = knowable(world, obs, key)
    return "UNKNOWN" if v is None else ("SUPPORTED" if v == claimed else "REFUTED")


def _combine(golds):
    if any(g == "REFUTED" for g in golds): return "REFUTED"
    if all(g == "SUPPORTED" for g in golds): return "SUPPORTED"
    return "UNKNOWN"


def extended_items(world: ExtWorld, observations, seed=0) -> List[Item]:
    rng = random.Random(seed * 991 + world.seed)
    gen = ExtLanguageGenerator(seed=seed * 57 + world.seed, allow_held_out=True)
    items = []

    def add(text, gold, intent, category, **meta):
        items.append(Item(world.seed, text, gold, intent, category, {**meta, "meanings": dict(gen.meanings)}))

    def ref_ok(o):
        if o in world.items:
            same = [m for m in world.items if world.objects[m].kind == world.objects[o].kind]
            if len(same) == 1: return "ok" if knowable(world, observations, ("color", o)) is not None else "UNKNOWN"
            if sum(1 for m in same if world.objects[m].color == world.objects[o].color) > 1: return "ASK"
            return "ok" if all(knowable(world, observations, ("color", m)) is not None for m in same) else "UNKNOWN"
        return ref_status(world, observations, o)

    def gate(gold, *objs):
        for o in objs:
            s = ref_ok(o)
            if s != "ok": return s
        return gold

    def gm(gold, *objs):                  # metadata for relabelling (expected answer before gating + references)
        return {"ungated": gold, "refs": tuple(objs)}

    # ---- inventory (third domain) and transfer of "X est dans Y"
    for o in world.items:
        v = knowable(world, observations, ("loc", o, "now"))
        subj, _ = gen.np(world, o)
        add(f"où est {subj} ?", gate(f"{v[0]}:{v[1]}" if v else "UNKNOWN", o), "where", "inventory", domain="inventory", clauses=1,
            subject=o, key=("loc", o, "now"), item_ref=True)
        l = rng.choice(world.lists)
        add(f"{subj} est dans {gen.np(world, l)[0]}", gate(_gold_claim(world, observations, ("loc", o, "now"), ("in", l)), o),
            "claim", "inventory", domain="inventory", clauses=1, subject=o, key=("loc", o, "now"), claimed=("in", l), item_ref=True)
        target = rng.choice(world.lists)
        add(f"range {subj} dans {gen.np(world, target)[0]}", gate(_effect([(o, "in", target)]), o), "command", "polysemy",
            domain="inventory", sense="act:assign", clauses=1, item_ref=True, **gm(_effect([(o, "in", target)]), o))
    # ---- polysemy on the physical side
    box = next((b for b in world.movables if world.objects[b].kind == "box"), None)
    phys = [m for m in world.movables if m != box]
    if phys:
        p = rng.choice(phys)
        rel, tgt = ("in", box) if box else ("on", "shelf")
        add(f"range {gen.np(world, p)[0]} {gen._join(gen.rel(rel), gen.np(world, tgt)[0])}", gate(_effect([(p, rel, tgt)]), p, tgt),
            "command", "polysemy", domain="physical", sense="act:place", clauses=1, **gm(_effect([(p, rel, tgt)]), p, tgt))
        # lexical composition: re- prefix
        verb = rng.choice(["repose", "replace"])
        gen.meanings[verb] = "act:place"
        add(f"{verb} {gen.np(world, p)[0]} sur la table", gate(_effect([(p, "on", "table")]), p), "command", "composition",
            domain="physical", clauses=1, **gm(_effect([(p, "on", "table")]), p))
    # ---- long sentences: 2 and 3 clauses
    objs = world.movables
    if len(objs) >= 3:
        parts, golds, refs = [], [], []
        for o in rng.sample(objs, 3):
            rel, tgt = rng.choice([("on", "table"), ("on", "floor"), ("on", "shelf"), ("under", "table")])
            parts.append(f"{gen.np(world, o)[0]} est {gen._join(gen.rel(rel), gen.np(world, tgt)[0])}")
            golds.append(gate(_gold_claim(world, observations, ("loc", o, "now"), (rel, tgt)), o))
            refs.append((o, (rel, tgt)))
        for k in (2, 3):
            g = golds[:k]
            gold = "ASK" if "ASK" in g else ("UNKNOWN" if any(x == "UNKNOWN" for x in g) and "REFUTED" not in g else _combine(g))
            text = " et ".join(parts[:k]) if k == 2 else f"{parts[0]}, {parts[1]} et {parts[2]}"
            add(text, gold, "claim", "long", domain="physical", clauses=k, parts=tuple(refs[:k]))
        dests = [("on", "table"), ("on", "shelf"), ("on", "floor")]
        chosen = rng.sample(objs, 3)
        steps = [(o, *rng.choice(dests)) for o in chosen]
        text = " puis ".join(f"pose {gen.np(world, o)[0]} {gen._join(gen.rel(r), gen.np(world, t)[0])}" for o, r, t in steps)
        add(text, gate(_effect(steps), *chosen), "command", "long", domain="physical", clauses=3, **gm(_effect(steps), *chosen))
    return items


def build_dataset_ext(n_worlds=20, world_seed=0, lang_seed=None, obs_seed=None):
    """Separate seeds: worlds (world_seed), language of the observations (lang_seed), noise (obs_seed)."""
    lang_seed = world_seed + 100 if lang_seed is None else lang_seed
    obs_seed = world_seed if obs_seed is None else obs_seed
    data = []
    for i in range(n_worlds):
        world = generate_world_ext(world_seed * 100000 + 50000 + i)
        obs = emit_observations_ext(world, obs_seed, lang_seed)
        items = benchmark_items(world, obs, lang_seed) + extended_items(world, obs, lang_seed)
        for it in items:
            it.meta.setdefault("domain", "physical"); it.meta.setdefault("clauses", 1)
        data.append({"world": world, "observations": obs, "items": items})
    return data


def split_worlds(data, seed=0, fractions=(0.4, 0.15, 0.15)):
    """Random split (seed) of the worlds: train / calibration / validation / test."""
    idx = list(range(len(data))); random.Random(seed).shuffle(idx)
    n = len(idx); a = int(fractions[0] * n); b = a + int(fractions[1] * n); c = b + int(fractions[2] * n)
    parts = {"train": idx[:a], "calibration": idx[a:b], "validation": idx[b:c], "test": idx[c:]}
    return {k: [data[i] for i in v] for k, v in parts.items()}

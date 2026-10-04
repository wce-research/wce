"""Frontière épistémique : VRAI ≠ OBSERVÉ ≠ INFÉRABLE ≠ AFFIRMABLE.

IdealAgent : référence épistémique INDÉPENDANTE du système testé (n'importe ni analyseur, ni résolveur, ni grammaire).
Il reçoit la vue PUBLIQUE des observations :
  - observations structurées (capteurs : ils voient l'objet, l'identifiant est public) ;
  - phrases réduites à leur CONTENU : descripteur du sujet (type, couleur dite) + prédicat, SANS identifiant d'objet
    (qui est une information cachée du générateur) ; les couleurs « impliquées » par le générateur ne sont pas fournies.
Il applique des règles publiques : vote par source pondéré par la fiabilité, résolution d'un descripteur par type unique
(niveau 0) ou par élimination explicite (niveau 1), le descripteur étant une affirmation de la même source.

Réétiquetage : chaque item est rejugé avec la même logique que le générateur, mais en remplaçant l'oracle « knowable »
(qui lisait l'identité cachée des phrases et n'acceptait les couleurs décrites que pour les types uniques) par l'IdealAgent.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.independent_world import COLORS, SUPPORT_HEIGHT, domain_size

KNOWABLE, UNKNOWN, CONFLICTED = "IDEALLY_KNOWABLE", "IDEALLY_UNKNOWN", "IDEALLY_CONFLICTED"
L0, L1, L2, L3 = 0, 1, 2, 3
LEVEL_NAMES = {L0: "direct observation", L1: "deterministic inference", L2: "probabilistic inference", L3: "speculative"}


# ================================================================ vue publique

@dataclass(frozen=True)
class Desc:
    type: str
    color: Optional[str] = None
    place: Optional[str] = None            # support (identifiant public : table, étagère, sol)


@dataclass
class PublicText:
    source: str
    time: int
    kind: str                               # color | loc | height | owner | event
    subject: Desc
    value: Any                              # valeur ; pour loc/event : (rel, Desc cible)
    loc_time: Any = None                    # 0 | "now" pour loc
    event_index: Optional[int] = None


@dataclass
class PublicStructured:
    source: str
    key: Tuple
    value: Any
    time: int


def _desc(world, oid, with_color=True):
    if oid in SUPPORT_HEIGHT: return Desc(oid, None, oid)
    o = world.objects[oid]
    return Desc(o.kind, o.color if (with_color or o.kind == "list") else None)


def public_view(world, observations, hidden=False):
    """Réécrit les observations sous forme publique (évaluation : seul ce module lit le monde pour RENDRE le contenu
    des phrases, exactement ce que le texte dit ; aucun identifiant d'objet n'est transmis pour le texte)."""
    structured, texts, inventory = [], [], {}
    for ob in observations:
        if hidden and (ob.implied or ob.text is not None):        # contrôle : l'ancien oracle lisait l'identité cachée
            if ob.key[0] != "type": structured.append(PublicStructured(ob.source, ob.key, ob.value, ob.time))
            continue
        if ob.implied: continue
        if ob.key[0] == "type": inventory[ob.key[1]] = ob.value; continue
        if ob.text is None:
            structured.append(PublicStructured(ob.source, ob.key, ob.value, ob.time)); continue
        k = ob.key[0]
        if k == "color": texts.append(PublicText(ob.source, ob.time, "color", _desc(world, ob.key[1], False), ob.value))
        elif k == "loc":
            rel, tgt = ob.value
            texts.append(PublicText(ob.source, ob.time, "loc", _desc(world, ob.key[1]), (rel, _desc(world, tgt)), ob.key[2]))
        elif k in ("height", "owner"): texts.append(PublicText(ob.source, ob.time, k, _desc(world, ob.key[1]), ob.value))
        elif k == "event":
            e = ob.value
            texts.append(PublicText(ob.source, ob.time, "event", _desc(world, e.patient), (e.rel, _desc(world, e.target)),
                                    event_index=ob.key[1]))
    return inventory, structured, texts


# ================================================================ règle explicite d'élimination

@dataclass
class EliminateCandidates:
    """Conclusion par élimination, auditée comme telle."""
    descriptor: Tuple
    candidates: List[str]
    excluded: Dict[str, Any]               # candidat -> couleur établie (prémisse)
    remaining: Optional[str]
    premises: List[Tuple] = field(default_factory=list)   # (clé, valeur, postérieure)
    provenance: str = ""
    confidence: float = 0.0
    level: int = L1

    @property
    def conclusive(self): return self.remaining is not None


def eliminate(descriptor, candidates, known_color, premise_posterior, provenance=""):
    """candidats d'un type ; les objets dont la couleur ÉTABLIE diffère sont exclus ; conclusion si un seul reste."""
    typ, color = descriptor
    excluded = {o: known_color[o] for o in candidates if o in known_color and known_color[o] != color}
    matching = [o for o in candidates if known_color.get(o) == color]
    remaining = [o for o in candidates if o not in excluded]
    premises = [(("color", o), v, premise_posterior.get(o, 1.0)) for o, v in excluded.items()]
    conf = math.prod(p for _, _, p in premises) if premises else 1.0
    if len(remaining) == 1:
        level = L0 if matching == remaining and not excluded else L1
        return EliminateCandidates(descriptor, list(candidates), excluded, remaining[0], premises, provenance, conf, level)
    return EliminateCandidates(descriptor, list(candidates), excluded, None, premises, provenance, 0.0, L3)


# ================================================================ agent idéal

class IdealAgent:
    def __init__(self, reliability: Dict[str, float], threshold=0.9, iterations=3):
        self.rel = dict(reliability); self.threshold = threshold; self.iterations = iterations
        self.inventory: Dict[str, str] = {}
        self.votes: Dict[Tuple, Dict[str, Tuple[Any, int, str]]] = defaultdict(dict)   # clé -> source -> (valeur, niveau, provenance)
        self.rules: List[EliminateCandidates] = []

    def r(self, s): return min(max(self.rel.get(s, 0.7), 0.01), 0.99)

    # ---- postérieure publique (mêmes formules que les sources, sans identité cachée) ---------------
    def posterior(self, key):
        votes = self.votes.get(key, {})
        if not votes: return None, 0.0, {}
        k = domain_size(key[0]) if key[0] in ("color", "loc", "height", "owner", "event") else 6
        scores = Counter()
        for s, (v, lvl, _) in votes.items(): scores[v] += math.log(self.r(s) * (k - 1) / (1 - self.r(s)))
        z = sum(math.exp(x) for x in scores.values()) + (k - len(scores))
        v, sc = max(scores.items(), key=lambda t: t[1])
        return v, math.exp(sc) / z, {val: math.exp(x) / z for val, x in scores.items()}

    def level(self, key):
        votes = self.votes.get(key, {})
        return min((lvl for _, lvl, _ in votes.values()), default=L3)

    def known(self, key):
        v, p, _ = self.posterior(key)
        return v if v is not None and p >= self.threshold else None

    def status(self, key):
        v, p, dist = self.posterior(key)
        if v is None: return UNKNOWN
        if p >= self.threshold: return KNOWABLE
        second = sorted(dist.values(), reverse=True)[1] if len(dist) > 1 else 0.0
        return CONFLICTED if second >= 0.2 else UNKNOWN

    # ---- résolution publique d'un descripteur ---------------------------------------------------
    def objects_of(self, typ): return [o for o, k in self.inventory.items() if k == typ]

    def resolve(self, d: Desc, provenance=""):
        if d.place is not None: return d.place, L0, None
        cands = self.objects_of(d.type)
        if len(cands) == 1: return cands[0], L0, None
        if len(cands) == 0 or d.color is None: return None, L3, None
        known = {o: self.known(("color", o)) for o in cands}
        known = {o: v for o, v in known.items() if v is not None}
        post = {o: self.posterior(("color", o))[1] for o in known}
        rule = eliminate((d.type, d.color), cands, known, post, provenance)
        if rule.conclusive: return rule.remaining, rule.level, rule
        return None, L3, rule

    # ---- apprentissage sur un monde -------------------------------------------------------------
    def observe(self, inventory, structured, texts):
        self.inventory = dict(inventory)
        base = defaultdict(dict)
        for ob in structured:
            key = ob.key
            if key[0] == "event": key, value = ("event", ob.value.patient), (ob.value.rel, ob.value.target)
            else: value = ob.value
            if key[0] == "loc": key = ("loc", key[1], "now" if key[2] != 0 else 0)
            base[key][ob.source] = (value, L0, "capteur")
        for key in list(base): self.votes[key] = dict(base[key])
        self.events = {}
        for _ in range(self.iterations):                           # identités -> votes -> croyances (point fixe)
            votes = defaultdict(dict, {k: dict(v) for k, v in base.items()})
            self.rules = []
            events = {}
            for t in texts:
                o, lvl, rule = self.resolve(t.subject, f"{t.source}:{t.kind}")
                if rule is not None: self.rules.append(rule)
                if o is None: continue
                if t.subject.color is not None and t.subject.place is None:     # le descripteur est une affirmation de la source
                    votes[("color", o)].setdefault(t.source, (t.subject.color, lvl, "descripteur"))
                if t.kind in ("loc", "event"):
                    rel, tgt = t.value
                    to, tl, trule = self.resolve(tgt, f"{t.source}:cible")
                    if trule is not None: self.rules.append(trule)
                    if to is None: continue
                    lvl = max(lvl, tl)
                    if t.kind == "loc": votes[("loc", o, "now" if t.loc_time != 0 else 0)][t.source] = ((rel, to), lvl, "texte")
                    else: events.setdefault(o, {})[t.event_index] = ((rel, to), t.source, lvl)
                elif t.kind == "color": votes[("color", o)][t.source] = (t.value, lvl, "texte")
                else: votes[(t.kind, o, "now") if t.kind == "height" else (t.kind, o)][t.source] = (t.value, lvl, "texte")
            for key, v in list(votes.items()):
                if key[0] == "event": events.setdefault(key[1], {})
            self.votes = votes; self.events = events
        return self

    # ---- valeurs publiques ---------------------------------------------------------------------
    def value(self, key):
        """Valeur idéalement connaissable (ou None), niveau et statut."""
        if key[0] == "loc" and key[2] == "now":
            ev = self.known(("event", key[1]))
            if ev is not None: return ev, self.level(("event", key[1])), KNOWABLE
            last = self.events.get(key[1], {})
            if last:                                               # événement textuel (jamais corrompu : même règle de vote)
                i = max(last); (val, src, lvl) = last[i]
                self.votes[("event_text", key[1])] = {src: (val, lvl, "texte")}
                p = self.posterior(("event_text", key[1]))[1]
                if p >= self.threshold: return val, lvl, KNOWABLE
        if key[0] == "height":
            loc, lvl, st = self.value(("loc", key[1], "now"))
            if loc is not None and (loc[1] in SUPPORT_HEIGHT or loc[0] == "under"):
                return (0.0 if loc[0] == "under" else SUPPORT_HEIGHT[loc[1]]), max(lvl, L1), KNOWABLE
        v = self.known(key)
        return v, self.level(key), self.status(key)

    def ref_status(self, oid, with_color=True, pool=None):
        """Une description (type [+ couleur]) est-elle résoluble d'après les observations publiques ? ok / ASK / UNKNOWN."""
        if oid in SUPPORT_HEIGHT: return "ok"
        kind = self.inventory.get(oid)
        same = [m for m in (pool or self.inventory) if self.inventory.get(m) == kind]
        color_known = self.value(("color", oid))[0] is not None
        if len(same) == 1: return "ok" if (not with_color or color_known) else "UNKNOWN"
        if not with_color: return "ASK"
        return "ok_if_distinct", same


# ================================================================ frontière de connaissance

@dataclass
class KnowledgeBoundary:
    key: Tuple
    truth: Any
    true_in_world: bool
    directly_observed: bool
    inferable_from_observations: bool
    inferable_only_with_hidden_information: bool
    contradicted: bool
    underdetermined: bool
    safe_to_assert: bool
    level: int
    ideal_value: Any = None
    ideal_status: str = UNKNOWN
    misleading: bool = False


def boundary(key, truth, ideal: IdealAgent, oracle_value):
    v, lvl, st = ideal.value(key)
    votes = ideal.votes.get(key, {})
    direct = any(l == L0 for _, l, _ in votes.values())
    contradicted = any(val != truth for val, _, _ in votes.values())
    return KnowledgeBoundary(key, truth, truth is not None, direct, v is not None and v == truth,
                             oracle_value is not None and v is None, contradicted, v is None, v is not None,
                             lvl if v is not None else L3, v, st, v is not None and v != truth)


# ================================================================ réétiquetage des items avec l'agent idéal

def _fmt_loc(v): return f"{v[0]}:{v[1]}"


class Relabeler:
    """Même logique d'attendus que le générateur ; l'oracle de connaissabilité est l'IdealAgent (observations publiques)."""

    def __init__(self, world, observations, reliability, hidden=False):
        inv, st, tx = public_view(world, observations, hidden)
        self.world, self.obs = world, observations
        self.ideal = IdealAgent(reliability).observe(inv, st, tx)
        self.facts = world.facts()

    def knowable(self, key):
        """Valeur connaissable ET vraie (même convention que le générateur : une preuve trompeuse ne rend rien connaissable)."""
        if key[0] == "loc" and key[2] == "now":
            truth = self.world.loc(key[1], "now")
        elif key[0] == "height": truth = self.world.height(key[1], "now")
        else: truth = self.facts.get(key)
        v = self.ideal.value(key)[0]
        return v if v == truth else None

    def ref_status(self, o, with_color=True, pool=None):
        r = self.ideal.ref_status(o, with_color, pool)
        if isinstance(r, str): return r
        _, same = r
        color = self.world.objects[o].color
        if sum(1 for m in same if self.world.objects[m].color == color) > 1: return "ASK"
        others = same if pool is not None else [m for m in same if m != o]       # inventaire : toutes les couleurs (générateur)
        return "ok" if all(self.knowable(("color", m)) is not None for m in others) else "UNKNOWN"

    def gated(self, gold, *objs, pool=None):
        for o in objs:
            s = self.ref_status(o, pool=pool)
            if s != "ok": return s
        return gold

    def claim(self, key, claimed):
        v = self.knowable(key)
        if v is None: return "UNKNOWN"
        same = abs(v - claimed) < 1e-6 if key[0] == "height" else v == claimed
        return "SUPPORTED" if same else "REFUTED"

    def relabel(self, it):
        """(attendu idéal, réétiqueté ?)."""
        m, g = it.meta, it.gold
        if g in ("UNKNOWN_WORD", "REJECT") and "refs" not in m: return g, False
        pool = self.world.items if m.get("item_ref") else None
        try:
            if m.get("kind") == "nonexistent_entity":
                k, c = m["absent"]
                same = [o for o, kk in self.ideal.inventory.items() if kk == k]
                return ("INVALID_PREMISE" if all(self.knowable(("color", o)) is not None for o in same) else "UNKNOWN"), True
            if m.get("kind") == "false_presupposition":
                o = m["subject"]; vl = self.knowable(("loc", o, "now"))
                return self.gated("UNKNOWN" if vl is None else ("UNKNOWN_CAUSE" if vl == ("on", "table") else "INVALID_PREMISE"), o), True
            if "parts" in m:
                golds = [self.gated(self.claim(("loc", o, "now"), c), o) for o, c in m["parts"]]
                if "ASK" in golds: return "ASK", True
                if any(x == "UNKNOWN" for x in golds) and "REFUTED" not in golds: return "UNKNOWN", True
                if any(x == "REFUTED" for x in golds): return "REFUTED", True
                return ("SUPPORTED" if all(x == "SUPPORTED" for x in golds) else "UNKNOWN"), True
            if it.intent == "where" and "key" in m:
                v = self.knowable(m["key"]); gold = _fmt_loc(v) if v else "UNKNOWN"
                return self.gated(gold, m["subject"], pool=pool), True
            if it.intent == "color" and "key" in m:
                o = m["subject"]; kind = self.world.objects[o].kind
                unique = sum(1 for x in self.world.movables if self.world.objects[x].kind == kind) == 1
                v = self.knowable(m["key"])
                return ("ASK" if not unique else (v if v else "UNKNOWN")), True
            if it.intent == "height" and "key" in m:
                v = self.knowable(m["key"])
                return self.gated(f"{v:g}" if v is not None else "UNKNOWN", m["subject"]), True
            if it.intent == "claim" and "claimed" in m:
                gold = self.claim(m["key"], m["claimed"])
                if m["key"][0] == "color": return gold, True               # affirmations de couleur : type unique, non filtrées
                return self.gated(gold, m["subject"], pool=pool), True
            if it.intent == "command" and "refs" in m:
                gold = m["ungated"]
                if m.get("kind") == "condition":
                    o, loc = m["cond"]; known = self.knowable(("loc", o, "now"))
                    gold = "UNKNOWN" if known is None else (m["if_true"] if known == loc else "NONE")
                elif m.get("kind") == "relative":
                    (rrel, rtgt), trel, target = m["relative"]
                    locs = {x: self.knowable(("loc", x, "now")) for x in self.world.movables}
                    matches = [x for x, v in locs.items() if v == (rrel, rtgt)]; unknown = [x for x, v in locs.items() if v is None]
                    if len(matches) == 1 and not unknown:
                        from core.independent_world import _effect, _valid_placement
                        gold = _effect([(matches[0], trel, target)]) if _valid_placement(self.world, matches[0], trel, target) else "REJECT"
                    else: gold = "ASK" if len(matches) > 1 else "UNKNOWN"
                    return gold, True
                elif m.get("kind") == "attachment":
                    from core.independent_world import _effect
                    p, box = m["attachment"]
                    r1 = self.knowable(("loc", p, "now")) == ("in", box); r2 = self.knowable(("loc", box, "now")) == ("on", "table")
                    gold = "ASK" if r1 and r2 else _effect([(p, "on", "table")]) if r1 else _effect([(p, "in", box)]) if r2 else "UNKNOWN"
                if it.category == "ambiguous" and g == "ASK": return "ASK", True
                for o in m["refs"]:
                    s = self.ref_status(o, pool=self.world.items if o in getattr(self.world, "items", ()) else None)
                    if s != "ok" and gold not in ("ASK",): return s, True
                return gold, True
        except (KeyError, AttributeError, TypeError):
            return g, False
        return g, False


def relabel_dataset(data, reliability, hidden=False):
    """Attendus idéaux pour tous les mondes : {id(item): (attendu idéal, réétiqueté)}."""
    out = {}
    for x in data:
        rl = Relabeler(x["world"], x["observations"], reliability, hidden)
        for it in x["items"]: out[id(it)] = rl.relabel(it)
        x["ideal"] = rl
    return out


def true_reliabilities():
    """Fiabilités publiées des sources (connaissance publique, comme le générateur les déclare)."""
    from core.independent_world import SOURCE_BY_NAME
    return {s: v.reliability for s, v in SOURCE_BY_NAME.items()}

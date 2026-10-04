"""Langage conscient de la mesure (langage contrôlé) : la phrase modifie le CONTEXTE DE MESURE.

« le capteur voit X qui tombe vers Y »        -> événement observé par un capteur (fiabilité du capteur)
« selon Alice, X est sur Y » / « Alice dit que »-> observation par l'observateur Alice (fiabilité d'Alice)
« la mesure indique environ 1.2 m pour X »     -> mesure continue, bruit σ (« environ »)
« on ne sait qu'à 20 cm près »                 -> RÉSOLUTION de lecture 0,2 m (≠ incertitude)
« le cube sur la table », « la balle qui était dans la boîte » -> descriptions relationnelles (présent / passé)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

TYPE_WORDS = {"cube": "cube", "balle": "ball", "bloc": "block", "objet": None}
COLOR_WORDS = {"rouge": "red", "bleu": "blue", "bleue": "blue", "vert": "green", "verte": "green", "jaune": "yellow"}
PLACE_PATTERNS = [("sous l'étagère", "under_shelf"), ("sur la table", "table"), ("sur l'étagère", "shelf"), ("sur le sol", "floor"),
                  ("dans la boîte", "box"), ("vers la table", "table"), ("vers l'étagère", "shelf"), ("vers le sol", "floor"),
                  ("vers la boîte", "box"), ("vers sous l'étagère", "under_shelf"), ("la table", "table"), ("l'étagère", "shelf"),
                  ("le sol", "floor"), ("la boîte", "box")]
ENVIRON_SIGMA = 0.15


@dataclass
class Description:
    type: Optional[str] = None
    color: Optional[str] = None
    support: Optional[str] = None             # relation spatiale décrite
    past: bool = False                         # « qui était … »


@dataclass
class ParsedMeasurement:
    kind: str                                  # observation | event | continuous | question
    observer: str
    description: Description
    attribute: str = "loc"
    value: Any = None
    resolution: float = 0.0
    sigma: float = 0.0
    sensor: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


def parse_place(text):
    for pat, sup in PLACE_PATTERNS:
        if pat in text: return sup
    return None


def parse_description(text) -> Description:
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", text.lower())
    d = Description()
    for i, w in enumerate(words):
        if w in TYPE_WORDS and d.type is None and (TYPE_WORDS[w] or w == "objet"):
            d.type = TYPE_WORDS[w]
            if i + 1 < len(words) and words[i + 1] in COLOR_WORDS: d.color = COLOR_WORDS[words[i + 1]]
    m = re.search(r"qui était (sur la table|sur l'étagère|sur le sol|dans la boîte|sous l'étagère)", text)
    if m: d.support, d.past = parse_place(m.group(1)), True
    else:
        m = re.search(r"(?:cube|balle|bloc|objet)(?: \w+)? (sur la table|sur l'étagère|sur le sol|dans la boîte|sous l'étagère)", text)
        if m: d.support = parse_place(m.group(1))
    return d


def parse(text: str, default_observer="inconnu") -> Optional[ParsedMeasurement]:
    t = text.lower().strip()
    m = re.match(r"le capteur voit (.+?) qui (tombe|est déplacée?) vers (.+)$", t)
    if m:
        return ParsedMeasurement("event", "camera_event", parse_description(m.group(1)), "loc", parse_place("vers " + m.group(3)),
                                 sensor=True, extra={"verb": m.group(2)})
    m = re.match(r"la mesure indique environ ([0-9.]+) m pour (.+?)(?:, on ne sait qu'à ([0-9]+) cm près)?$", t)
    if m:
        res = int(m.group(3)) / 100 if m.group(3) else 0.0
        return ParsedMeasurement("continuous", "doc", parse_description(m.group(2)), "height", float(m.group(1)),
                                 resolution=res, sigma=ENVIRON_SIGMA)
    m = re.match(r"selon (\w+), (.+?) est (.+)$", t) or re.match(r"(\w+) dit que (.+?) est (.+)$", t)
    if m:
        return ParsedMeasurement("observation", m.group(1), parse_description(m.group(2)), "loc", parse_place(m.group(3)))
    return None


def parse_question(text):
    t = text.lower()
    if t.startswith("où est"): return {"intent": "where_now", "description": parse_description(t[6:])}
    m = re.match(r"où était (.+?) à t=(\d+)", t)
    if m: return {"intent": "where_at", "description": parse_description(m.group(1)), "t": int(m.group(2))}
    if t.startswith("quelle est la hauteur"): return {"intent": "height_now", "description": parse_description(t)}
    if t.startswith("de quelle couleur est"): return {"intent": "color_of_relational", "description": parse_description(t[22:])}
    m = re.match(r"quel objet était (sur la table|sur l'étagère|sur le sol|dans la boîte|sous l'étagère) avant la chute", t)
    if m: return {"intent": "before_fall", "support": parse_place(m.group(1))}
    return None

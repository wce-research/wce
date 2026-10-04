"""Monde TEMPOREL de mesure (bibliothèque standard seulement ; aucun composant de WCE).

T pas (20–50) ; objets typés (deux du même type au moins : ambiguïté d'identité), couleurs, positions sur des supports
(table, étagère, sol, boîte, sous l'étagère) avec un rang gauche→droite, hauteur (support + empilement), propriétaire rare.
Événements : déplacements, chutes, mises en boîte, empilements, transferts de propriétaire — partiellement observés.
Capteurs à fréquences différentes, mesures manquantes et bruitées, témoignages en langage (descriptions
relationnelles, « selon Alice… », « le capteur voit… », « la mesure indique environ… », « à 10 cm près »).
Mesures ACTIVES disponibles à la question (avec coût, bruit, perturbation) : simulées sur l'état latent final.
Questions datées : où est X maintenant / à t, couleur de « le cube sur la table », quel objet était sur la table
avant la chute, hauteur. Attendu : la VÉRITÉ (une abstention n'est ni juste ni fausse confiante).
"""
from __future__ import annotations

import hashlib
import json
import random

SUPPORTS = ["table", "shelf", "floor", "box", "under_shelf"]
SUPPORT_WORD = {"table": "la table", "shelf": "l'étagère", "floor": "le sol", "box": "la boîte", "under_shelf": "sous l'étagère"}
REL_WORD = {"table": "sur", "shelf": "sur", "floor": "sur", "box": "dans", "under_shelf": ""}
HEIGHT = {"table": 1.0, "shelf": 2.0, "floor": 0.0, "box": 0.5, "under_shelf": 0.0}
TYPES = {"cube": ("le", "cube", False), "ball": ("la", "balle", True), "block": ("le", "bloc", False)}
COLORS = {"red": ("rouge", "rouge"), "blue": ("bleu", "bleue"), "green": ("vert", "verte"), "yellow": ("jaune", "jaune")}
OWNERS = ["marie", "paul", "lea"]
SPEAKERS = {"alice": 0.85, "bob": 0.55, "doc": 0.8}


def np_text(typ, color=None, definite=True):
    det, noun, fem = TYPES[typ]
    if not definite: det = "une" if fem else "un"
    s = f"{det} {noun}"
    if color: s += " " + COLORS[color][1 if fem else 0]
    return s


def of_np(np):
    return ("du " + np[3:]) if np.startswith("le ") else ("de " + np)


def place_text(sup):
    return SUPPORT_WORD[sup] if sup == "under_shelf" else f"{REL_WORD[sup]} {SUPPORT_WORD[sup]}"


def episode(seed, T=30, n_objects=4):
    rng = random.Random(seed)
    types = ["cube", "cube"] + [rng.choice(list(TYPES)) for _ in range(n_objects - 2)]
    objs = {}
    for i, typ in enumerate(types):
        same = [o for o in objs.values() if o["type"] == typ]
        color = rng.choice([c for c in COLORS if all(o["color"] != c for o in same)] or list(COLORS))
        objs[f"o{i}"] = {"type": typ, "color": color, "owner": rng.choice(OWNERS)}
    traj = {o: {"loc": [], "rank": [], "height": [], "owner": []} for o in objs}
    loc = {o: rng.choice(["table", "shelf", "floor", "box"]) for o in objs}
    stack = {o: 0.0 for o in objs}; owner = {o: objs[o]["owner"] for o in objs}
    events = []
    for t in range(T + 1):
        if t > 0:
            for o in objs:
                u = rng.random()
                if u < 0.04 and loc[o] in ("table", "shelf"):
                    events.append({"t": t, "o": o, "kind": "fall", "to": "floor"}); loc[o] = "floor"; stack[o] = 0.0
                elif u < 0.09:
                    to = rng.choice([s for s in SUPPORTS if s != loc[o]])
                    events.append({"t": t, "o": o, "kind": "move", "to": to}); loc[o] = to; stack[o] = 0.0
                elif u < 0.11:
                    stack[o] = 0.5 - stack[o]; events.append({"t": t, "o": o, "kind": "stack", "to": loc[o]})
                if rng.random() < 0.01:
                    owner[o] = rng.choice([p for p in OWNERS if p != owner[o]])
                    events.append({"t": t, "o": o, "kind": "transfer", "to": owner[o]})
        order = sorted(objs, key=lambda o: (loc[o], rng.random()))
        for o in objs:
            traj[o]["loc"].append(loc[o]); traj[o]["height"].append(HEIGHT[loc[o]] + stack[o]); traj[o]["owner"].append(owner[o])
            traj[o]["rank"].append([p for p in order if loc[p] == loc[o]].index(o))
    obs = []
    for o, d in objs.items():                                                           # capteur de couleur, à t = 0, parfois absent
        if rng.random() < 0.6:
            v = d["color"] if rng.random() < 0.95 else rng.choice([c for c in COLORS if c != d["color"]])
            obs.append({"t": 0, "kind": "sensor", "source": "color_cam", "entity": o, "attribute": "color", "value": v})
    for t in range(T + 1):
        for o, d in objs.items():
            if t % 3 == 0 and rng.random() < 0.8:                                       # caméra : positions, toutes les 3 étapes
                v = traj[o]["loc"][t] if rng.random() < 0.9 else rng.choice([s for s in SUPPORTS if s != traj[o]["loc"][t]])
                obs.append({"t": t, "kind": "sensor", "source": "camera", "entity": o, "attribute": "loc", "value": v})
            if t % 5 == 0 and rng.random() < 0.7:                                       # balance : hauteur continue, résolution 0,1
                h = traj[o]["height"][t] + rng.gauss(0, 0.08)
                obs.append({"t": t, "kind": "sensor", "source": "scale", "entity": o, "attribute": "height", "value": round(h, 1),
                            "resolution": 0.1})
            if rng.random() < 0.08:                                                     # témoignage en langage
                sp = rng.choice(list(SPEAKERS)); r = SPEAKERS[sp]
                v = traj[o]["loc"][t] if rng.random() < r else rng.choice([s for s in SUPPORTS if s != traj[o]["loc"][t]])
                same = [p for p in objs if objs[p]["type"] == d["type"]]
                desc = np_text(d["type"], d["color"] if len(same) > 1 or rng.random() < 0.5 else None)
                text = rng.choice([f"selon {sp}, {desc} est {place_text(v)}", f"{sp} dit que {desc} est {place_text(v)}"])
                obs.append({"t": t, "kind": "text", "source": sp, "text": text, "entity_truth": o})
            if rng.random() < 0.015:
                h = traj[o]["height"][t] + rng.gauss(0, 0.15)
                obs.append({"t": t, "kind": "text", "source": "doc",
                            "text": f"la mesure indique environ {h:.1f} m pour {np_text(d['type'], d['color'])}, on ne sait qu'à 20 cm près",
                            "entity_truth": o})
        for e in events:
            if e["t"] == t and e["kind"] in ("fall", "move") and rng.random() < 0.5:   # événements partiellement observés
                d = objs[e["o"]]
                verb = "tombe" if e["kind"] == "fall" else "est déplacé"
                obs.append({"t": t, "kind": "text", "source": "camera_event",
                            "text": f"le capteur voit {np_text(d['type'], d['color'])} qui {verb} vers {SUPPORT_WORD[e['to']]}",
                            "entity_truth": e["o"], "event_to": e["to"]})
    questions = []
    for o, d in objs.items():
        same = [p for p in objs if objs[p]["type"] == d["type"]]
        desc = np_text(d["type"], d["color"] if len(same) > 1 else None)
        questions.append({"type": "where_now", "text": f"où est {desc} maintenant ?", "entity": o, "t": T, "truth": traj[o]["loc"][T]})
        tp = rng.randint(1, T - 1)
        questions.append({"type": "where_at", "text": f"où était {desc} à t={tp} ?", "entity": o, "t": tp, "truth": traj[o]["loc"][tp]})
        questions.append({"type": "height_now", "text": f"quelle est la hauteur {of_np(desc)} ?", "entity": o, "t": T,
                          "truth": traj[o]["height"][T]})
    for typ in sorted({d["type"] for d in objs.values()}):        # ordre déterministe (indépendant de PYTHONHASHSEED)
        same = [p for p in objs if objs[p]["type"] == typ]
        if len(same) < 2: continue
        for sup in ("table", "shelf", "floor", "box"):
            on = [p for p in same if traj[p]["loc"][T] == sup]
            if len(on) == 1:
                questions.append({"type": "color_of_relational", "text": f"de quelle couleur est {np_text(typ)} {place_text(sup)} ?",
                                  "description": {"type": typ, "support": sup}, "t": T, "truth": objs[on[0]]["color"],
                                  "entity": on[0]})
    falls = [e for e in events if e["kind"] == "fall"]
    if falls:
        e = falls[-1]
        questions.append({"type": "before_fall", "text": f"quel objet était sur {SUPPORT_WORD[traj[e['o']]['loc'][e['t'] - 1]]} avant la chute ?",
                          "t": e["t"] - 1, "support": traj[e["o"]]["loc"][e["t"] - 1], "entity": e["o"],
                          "truth": f"{objs[e['o']]['type']}:{objs[e['o']]['color']}"})
    return {"seed": seed, "T": T, "objects": {o: {"type": d["type"]} for o, d in objs.items()}, "observations": obs,
            "questions": questions, "_truth": {"objects": objs, "trajectories": traj, "events": events}}


ACTIVE_MENU = {
    # mesure : (attribut, coût, fiabilité / bruit, perturbation)
    "color": ("color", 0.15, 0.97, "PASSIVE"),
    "loc_precise": ("loc", 0.5, 0.97, "PASSIVE"),
    "loc_coarse": ("loc_region", 0.1, 0.95, "PASSIVE"),
    "owner": ("owner", 0.3, 0.9, "PASSIVE"),
    "weigh": ("height", 0.2, 0.9, "DESTRUCTIVE"),        # pesée : déplace l'objet au sol (perturbe la position)
}
REGION = {"table": "high", "shelf": "high", "floor": "low", "box": "low", "under_shelf": "low"}


def train_episodes(n=30): return [episode(300000 + i) for i in range(n)]


def eval_episodes(seed=0, n=20): return [episode(400000 + 1000 * seed + i) for i in range(n)]


def sealed_episodes(): return [episode(990000 + i) for i in range(30)]


def sealed_hash():
    return hashlib.sha256(json.dumps(sealed_episodes(), sort_keys=True, default=str).encode()).hexdigest()


SEALED_SHA256 = "9d372b0762feedf18b3e10c4e877cebb215f0d9149538842d87eed536d85137c"   # empreinte du lot scellé (figée avant toute évaluation)

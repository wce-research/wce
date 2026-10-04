"""Banc ADVERSARIAL des sources (bibliothèque standard seulement ; aucun composant de WCE).

Chaque épisode : des faits (clé -> valeur), des sources au comportement caché, une phase d'HISTORIQUE étiquetée
(la vérité est révélée : c'est le passé des sources) puis une phase de TEST non étiquetée, et des questions.
Comportements : fiable qui ment parfois, mauvaise parfois juste, copieuse (corrélée), dérive, rare, conflit entre deux
sources fiables, bonne sur les couleurs / mauvaise sur les positions, capteur d'ancrage.
Référence (attendu) : postérieure IDÉALE avec les vraies fiabilités, la copieuse comptée une seule fois avec sa source ;
valeur si ≥ 0,9 ET vraie, « conflit » si deux valeurs gardent chacune ≥ 0,2, sinon « ? ».
"""
from __future__ import annotations

import hashlib
import json
import math
import random

COLORS = ["red", "blue", "green", "yellow"]
PLACES = [f"p{i}" for i in range(12)]
DOMAIN = {"color": COLORS, "loc": PLACES}

BEHAVIORS = {
    "anchor_sensor": {"type": "sensor", "r": 0.95, "rate": 0.5},
    "reliable_liar": {"type": "document", "r": 0.9, "rate": 0.6},
    "bad_sometimes_right": {"type": "speaker", "r": 0.35, "rate": 0.6},
    "original": {"type": "speaker", "r": 0.75, "rate": 0.5},
    "copier": {"type": "document", "copies": "original", "p_copy": 0.9, "r": 0.75, "rate": 0.5},
    "drifting": {"type": "speaker", "r": 0.9, "r_after": 0.35, "rate": 0.5},
    "rare": {"type": "speaker", "r": 0.8, "rate": 0.04},
    "domain_split": {"type": "speaker", "r_by_key": {"color": 0.92, "loc": 0.4}, "rate": 0.5},
}


def _corrupt(rng, kind, value):
    return rng.choice([v for v in DOMAIN[kind] if v != value])


def episode(seed, n_history=300, n_test=150):
    rng = random.Random(seed)
    def fact():
        kind = rng.choice(["color", "loc"]); return kind, rng.choice(DOMAIN[kind])
    claims, history = [], []
    def emit(phase, t, key, kind, value, out):
        orig_claim = None
        for name, b in BEHAVIORS.items():
            if "copies" in b: continue
            if rng.random() >= b["rate"]: continue
            if "r_by_key" in b: r = b["r_by_key"][kind]
            elif "r_after" in b and (phase == "test" or t > 2 * n_history // 3): r = b["r_after"]     # dérive fin d'historique
            else: r = b["r"]
            v = value if rng.random() < r else _corrupt(rng, kind, value)
            out.append({"source": name, "key": key, "kind": kind, "value": v, "t": t, "phase": phase})
            if name == "original": orig_claim = v
        b = BEHAVIORS["copier"]
        if rng.random() < b["rate"]:
            if orig_claim is not None and rng.random() < b["p_copy"]: v = orig_claim
            else: v = value if rng.random() < b["r"] else _corrupt(rng, kind, value)
            out.append({"source": "copier", "key": key, "kind": kind, "value": v, "t": t, "phase": phase})
    truth = {}
    for i in range(n_history):
        kind, value = fact(); key = f"h{i}"; truth[key] = value
        emit("history", i, key, kind, value, history)
    for i in range(n_test):
        kind, value = fact(); key = f"k{i}"; truth[key] = value
        emit("test", i, key, kind, value, claims)
    # conflit contrôlé : deux sources fiables disent deux valeurs différentes, rien d'autre
    for j in range(10):
        kind = "color"; value = rng.choice(COLORS); key = f"c{j}"; truth[key] = value
        other = _corrupt(rng, kind, value)
        claims.append({"source": "reliable_liar", "key": key, "kind": kind, "value": value, "t": n_test + j, "phase": "test"})
        claims.append({"source": "anchor_sensor", "key": key, "kind": kind, "value": other, "t": n_test + j, "phase": "test"})
    return {"seed": seed, "history": [dict(c, truth=truth[c["key"]]) for c in history], "claims": claims,
            "questions": _questions(claims, truth), "behaviors": BEHAVIORS}


def _true_r(source, kind, t, n_test):
    b = BEHAVIORS[source]
    if "r_by_key" in b: return b["r_by_key"][kind]
    if "r_after" in b: return b["r_after"]                    # en test, la source a dérivé
    if "copies" in b: return BEHAVIORS[b["copies"]]["r"]
    return b["r"]


def _questions(claims, truth, n_test=150):
    by_key = {}
    for c in claims: by_key.setdefault(c["key"], []).append(c)
    out = []
    for key, cs in by_key.items():
        kind = cs[0]["kind"]; k = len(DOMAIN[kind])
        votes = {}
        for c in cs:
            src = "original" if c["source"] == "copier" and any(x["source"] == "original" for x in cs) else c["source"]
            if src in votes and c["source"] == "copier": continue                # la copie ne compte pas en plus
            votes[src] = c
        scores = {}
        for s, c in votes.items():
            r = _true_r(c["source"], kind, c["t"], n_test); r = min(max(r, 0.01), 0.99)
            scores[c["value"]] = scores.get(c["value"], 0.0) + math.log(r * (k - 1) / (1 - r))
        z = sum(math.exp(v) for v in scores.values()) + (k - len(scores))
        post = {v: math.exp(x) / z for v, x in scores.items()}
        best = max(post, key=post.get)
        ranked = sorted(post.values(), reverse=True)
        if post[best] >= 0.9: expected = truth[key] if best == truth[key] else "?"
        elif len(ranked) > 1 and ranked[1] >= 0.2: expected = "conflit"
        else: expected = "?"
        out.append({"key": key, "kind": kind, "expected": expected, "truth": truth[key]})
    return out


def dev_episodes(n=6): return [episode(1000 + i) for i in range(n)]


def sealed_episodes(): return [episode(90000 + i) for i in range(4)]


def sealed_hash():
    return hashlib.sha256(json.dumps(sealed_episodes(), sort_keys=True).encode()).hexdigest()


SEALED_SHA256 = "88f9fe41043db602ab2ed1f6dfe6fbc8e31318eabc7cc3f07d222aee8f2362d4"   # empreinte du lot scellé (figée à sa création)

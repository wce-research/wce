"""Modèle temporel branché dans l'environnement indépendant : position au début (t = 0) et maintenant (t = 1),
événements entre les deux (observés ou NON observés), taux de changement appris sur les mondes d'entraînement.

La croyance BeliefState.loc(o, temps) devient : filtre de Markov caché (présent) / lissage (passé) sur les preuves
déjà résolues du graphe de preuves ; une position au début ne répond plus automatiquement au présent.
"""
from __future__ import annotations

import math
from collections import defaultdict

from core.independent_agent import BELIEF, CONFLICTED, FACT, HYPOTHESIS, UNKNOWN, BeliefEntry
from core.temporal_belief import KeyDynamics, TemporalBeliefState

LOC_DOMAIN = 12


def learn_change_rate(train_worlds):
    """P(position au début ≠ position maintenant) sur les mondes d'ENTRAÎNEMENT (vérité), lissée (Beta(1, 1))."""
    ch = n = 0
    for x in train_worlds:
        w = x["world"]
        for o in w.movables:
            ch += w.loc(o, 0) != w.loc(o, "now"); n += 1
    return (ch + 1) / (n + 2)


def temporal_loc(graph, rel, o, p_change, time="now"):
    """BeliefEntry temporelle pour la position de o (présent : filtré ; début : lissé)."""
    votes0 = graph.votes(("loc", o, 0)); votes1 = graph.votes(("loc", o, "now")); evs = graph.votes(("event", o))
    values = list(dict.fromkeys([e.value for e in votes0.values()] + [e.value for e in votes1.values()] + [e.value for e in evs.values()]))
    if not values: return BeliefEntry(("loc", o, time), None, 0.0, UNKNOWN)
    domain = values + [f"__autre{i}" for i in range(max(0, LOC_DOMAIN - len(values)))]
    lam = -math.log(max(1 - p_change, 1e-6))
    est = getattr(rel, "estimate_for", None)
    r_of = (lambda s, k: est(s, k)) if est else (lambda s, k: rel.estimate(s))
    st = TemporalBeliefState(o, "loc", domain, KeyDynamics("loc", lam))
    for s, e in votes0.items(): st.observe(0, e.value, s, min(r_of(s, "loc"), 0.99) * e.weight + (1 - e.weight) / LOC_DOMAIN)
    for s, e in votes1.items(): st.observe(1, e.value, s, min(r_of(s, "loc"), 0.99) * e.weight + (1 - e.weight) / LOC_DOMAIN)
    for s, e in evs.items(): st.event(1, e.value, s, min(r_of(s, "event"), 0.99) * e.weight)
    d = st.current(1) if time == "now" else st.historical(0, 1)
    value, post = max(d.items(), key=lambda kv: kv[1])
    if str(value).startswith("__autre"): return BeliefEntry(("loc", o, time), None, 0.0, UNKNOWN)
    t = 1 if time == "now" else 0
    pool = {**(votes1 if t == 1 else votes0)}
    supporting = [s for s, e in pool.items() if e.value == value] + ([s for s, e in evs.items() if e.value == value] if t == 1 else [])
    refuting = [s for s, e in pool.items() if e.value != value]
    strong = len(set(supporting)) >= 2 or any(r_of(s, "loc") >= 0.9 for s in supporting)
    status = FACT if post >= 0.95 and strong else BELIEF if post >= 0.9 else HYPOTHESIS if post >= 0.5 else UNKNOWN
    if status in (HYPOTHESIS, UNKNOWN) and st.conflict_at(t) and sorted(d.values(), reverse=True)[1] >= 0.2: status = CONFLICTED
    return BeliefEntry(("loc", o, time), value, post, status, supporting, refuting)


def install_temporal(agent, p_change):
    """Remplace BeliefState.loc de chaque nouveau monde par la croyance temporelle."""
    original_new_world = agent.new_world

    def new_world():
        b = original_new_world()
        b.loc = lambda o, time="now", _b=b: temporal_loc(_b.graph, agent.reliability, o, p_change, time)
        return b
    agent.new_world = new_world
    agent.temporal_p_change = p_change
    return agent


def loc_belief_points(agent, worlds):
    """(postérieure, exacte) pour la position présente et passée (une ingestion par monde)."""
    pts = defaultdict(list)
    for x in worlds:
        w = x["world"]
        agent.new_world(); agent.ingest(x["observations"])
        for o in w.movables:
            for time, kt in (("now", "loc_present"), (0, "loc_past")):
                b = agent.belief.loc(o, time)
                if b.value is None: continue
                pts[kt].append((b.posterior, b.value == w.loc(o, time)))
            h = agent.belief.height(o)
            if h.value is not None: pts["height"].append((h.posterior, abs(h.value - w.height(o, "now")) < 1e-6))
    return pts

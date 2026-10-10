"""Independent TEMPORAL bench (standard library only; no WCE component).

Episodes: entities with position (fast), height (tied to the supports, can "rise" without changing nominal
relation), owner (slow), colour (static). True trajectories over T steps: observed events, UNOBSERVED
events, false reported events, DELAYED reports (event at time t, reported later), old
testimonies, contradictory sources at the same instant, several events.
Questions: present, past, "before falling", "still on the table?".
Reference: IDEAL filter / smoother coded here in pure Python, with the TRUE rates and the TRUE reliabilities;
expected = true value if P ≥ 0.9; "?" otherwise; "conflit" if two credible sources contradict each other at the same instant.
"""
from __future__ import annotations

import hashlib
import json
import math
import random

PLACES = ["table", "shelf", "floor", "box"]
HEIGHTS = {"table": 1.0, "shelf": 2.0, "floor": 0.0, "box": 0.5}
HEIGHT_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
OWNERS = ["marie", "paul", "lea"]
COLORS = ["red", "blue", "green", "yellow"]
DOMAINS = {"loc": PLACES, "height": HEIGHT_VALUES, "owner": OWNERS, "color": COLORS}
TRUE_HAZARD = {"loc": 0.08, "height": 0.1, "owner": 0.01, "color": 0.0}
SOURCES = {"sensor": 0.95, "camera": 0.9, "doc": 0.8, "witness": 0.7, "gossip": 0.5}
EVENT_SOURCES = {"sensor": 0.95, "doc": 0.85, "witness": 0.7, "gossip": 0.45}


def _trajectory(rng, kind, T):
    dom = DOMAINS[kind]; v = rng.choice(dom); out = [v]; events = []
    for t in range(1, T + 1):
        if rng.random() < 1 - math.exp(-TRUE_HAZARD[kind]):
            v = rng.choice([x for x in dom if x != v]); events.append((t, v))
        out.append(v)
    return out, events


def episode(seed, T=30, n_entities=3):
    rng = random.Random(seed)
    ents, reports, questions = {}, [], []
    for i in range(n_entities):
        e = f"e{i}"
        traj = {}
        for kind in ("loc", "owner", "color"):
            traj[kind], ev = _trajectory(rng, kind, T)
            traj[kind + "_events"] = ev
        # height: determined by the support, but can "rise" (+0.5) without changing nominal relation
        h, hev = [], []
        bump = 0.0
        for t in range(T + 1):
            if t and rng.random() < 0.03: bump = 0.5 if bump == 0 else 0.0; hev.append((t, None))
            h.append(min(HEIGHTS[traj["loc"][t]] + bump, 2.5))
        traj["height"] = h
        ents[e] = traj
        for kind in ("loc", "owner", "color", "height"):
            for t in range(T + 1):
                for s, r in SOURCES.items():
                    rate = {"loc": 0.06, "owner": 0.03, "color": 0.03, "height": 0.04}[kind] * (2.0 if s == "sensor" else 1.0)
                    if rng.random() >= rate: continue
                    truth = traj[kind][t]
                    v = truth if rng.random() < r else rng.choice([x for x in DOMAINS[kind] if x != truth])
                    delay = rng.choice([0, 0, 0, 2, 5]) if s in ("doc", "witness") else 0       # delayed report
                    reports.append({"entity": e, "key": kind, "t": t, "value": v, "source": s, "kind": "observation",
                                    "ingested_at": min(T, t + delay)})
        for kind in ("loc", "owner"):
            for t, v in traj[kind + "_events"]:
                for s, r in EVENT_SOURCES.items():
                    if rng.random() >= 0.35: continue                                           # many UNOBSERVED events
                    tgt = v if rng.random() < r else rng.choice([x for x in DOMAINS[kind] if x != v])
                    reports.append({"entity": e, "key": kind, "t": t, "value": tgt, "source": s, "kind": "event",
                                    "ingested_at": min(T, t + rng.choice([0, 0, 3]))})
            for _ in range(rng.randint(0, 2)):                                                 # false events (weak sources)
                t = rng.randint(1, T)
                reports.append({"entity": e, "key": kind, "t": t, "value": rng.choice(DOMAINS[kind]), "source": "gossip",
                                "kind": "event", "ingested_at": t})
        # controlled conflict at the same instant between two credible sources
        tc = rng.randint(5, T - 5); truth = traj["loc"][tc]; other = rng.choice([x for x in PLACES if x != truth])
        right, wrong = rng.sample(["camera", "doc"], 2)                                         # the faulty source is drawn at random
        reports.append({"entity": e, "key": "loc", "t": tc, "value": truth, "source": right, "kind": "observation", "ingested_at": tc})
        reports.append({"entity": e, "key": "loc", "t": tc, "value": other, "source": wrong, "kind": "observation", "ingested_at": tc})
        ents[e]["conflict_t"] = tc
    now = T
    for e, traj in ents.items():
        for kind in ("loc", "height", "owner", "color"):
            questions.append({"entity": e, "key": kind, "type": "present", "t": now, "truth": traj[kind][now]})
            tp = rng.randint(0, T - 1)
            questions.append({"entity": e, "key": kind, "type": "past", "t": tp, "truth": traj[kind][tp]})
        falls = [t for t, v in traj["loc_events"] if v == "floor"]
        if falls:
            questions.append({"entity": e, "key": "loc", "type": "before_fall", "t": falls[-1] - 1, "truth": traj["loc"][falls[-1] - 1]})
        questions.append({"entity": e, "key": "loc", "type": "still_on_table", "t": now, "truth": traj["loc"][now] == "table"})
        questions.append({"entity": e, "key": "loc", "type": "past", "t": traj["conflict_t"], "truth": traj["loc"][traj["conflict_t"]]})
    ep = {"seed": seed, "T": T, "reports": reports, "questions": questions,
          "train_trajectories": [(k, ents[e][k]) for e in ents for k in ("loc", "height", "owner", "color")],
          "truth": {e: {k: ents[e][k] for k in ("loc", "height", "owner", "color")} for e in ents}}     # training / scoring only
    for q in questions: q["expected"] = _ideal_expected(ep, q)
    return ep


# ================================================================ ideal reference (pure Python, true parameters)

def _ideal_post(ep, entity, key, T_upto, smooth_at=None):
    dom = DOMAINS[key]; K = len(dom); lam = TRUE_HAZARD[key] if key != "height" else 0.1
    p_stay = math.exp(-lam)
    reps = [r for r in ep["reports"] if r["entity"] == entity and r["key"] == key and r["t"] <= T_upto]
    obs = {}; evs = {}
    for r in reps: (obs if r["kind"] == "observation" else evs).setdefault(r["t"], []).append(r)
    def lik(t):
        L = [1.0] * K
        for r in obs.get(t, []):
            rr = SOURCES[r["source"]]
            for i, v in enumerate(dom): L[i] *= rr if v == r["value"] else (1 - rr) / (K - 1)
        return L
    def trans(t):
        M = [[p_stay if i == j else (1 - p_stay) / (K - 1) for j in range(K)] for i in range(K)]
        for r in evs.get(t, []):
            re = EVENT_SOURCES[r["source"]]; j = dom.index(r["value"])
            M = [[re * (1.0 if jj == j else 0.0) + (1 - re) * M[i][jj] for jj in range(K)] for i in range(K)]
        return M
    f = []; a = [x / K * y for x, y in zip([1.0] * K, lik(0))]; z = sum(a); f.append([x / z for x in a])
    for t in range(1, T_upto + 1):
        M = trans(t); prev = f[-1]
        a = [sum(prev[i] * M[i][j] for i in range(K)) * lik(t)[j] for j in range(K)]; z = sum(a) or 1e-30
        f.append([x / z for x in a])
    if smooth_at is None: return dict(zip(dom, f[T_upto]))
    b = [1.0] * K
    for t in range(T_upto - 1, smooth_at - 1, -1):
        M = trans(t + 1); L = lik(t + 1)
        b = [sum(M[i][j] * L[j] * b[j] for j in range(K)) for i in range(K)]; z = sum(b) or 1e-30; b = [x / z for x in b]
    x = [f[smooth_at][i] * b[i] for i in range(K)]; z = sum(x) or 1e-30
    return dict(zip(dom, [v / z for v in x]))


BUMP_TOGGLE = 0.03


def _ideal_height(ep, entity, T_upto, smooth_at=None):
    """IDEAL height: exact filter on the joint state (position × elevation); the height is a function of this
    state (public rule of the world), observed through the height reports; the position through its own reports."""
    states = [(l, b) for l in PLACES for b in (0, 1)]; S = len(states)
    hval = [min(HEIGHTS[l] + 0.5 * b, 2.5) for l, b in states]
    p_stay = math.exp(-TRUE_HAZARD["loc"]); K = len(PLACES)
    reps = [r for r in ep["reports"] if r["entity"] == entity and r["key"] in ("loc", "height") and r["t"] <= T_upto]
    def lik(t):
        L = [1.0] * S
        for r in reps:
            if r["t"] != t or r["kind"] != "observation": continue
            rr = SOURCES[r["source"]]
            for i, (l, b) in enumerate(states):
                if r["key"] == "loc": L[i] *= rr if l == r["value"] else (1 - rr) / (K - 1)
                else: L[i] *= rr if hval[i] == r["value"] else (1 - rr) / (len(HEIGHT_VALUES) - 1)
        return L
    def trans(t):
        evs = [r for r in reps if r["t"] == t and r["kind"] == "event" and r["key"] == "loc"]
        M = [[0.0] * S for _ in range(S)]
        for i, (l, b) in enumerate(states):
            for j, (l2, b2) in enumerate(states):
                pl = p_stay if l == l2 else (1 - p_stay) / (K - 1)
                for r in evs:
                    re = EVENT_SOURCES[r["source"]]; pl = re * (1.0 if l2 == r["value"] else 0.0) + (1 - re) * pl
                pb = (1 - BUMP_TOGGLE) if b == b2 else BUMP_TOGGLE
                M[i][j] = pl * pb
        return M
    prior0 = [(1.0 / K if b == 0 else 0.0) for l, b in states]                  # true process: without elevation at t = 0
    f = []; a = [p0 * x for p0, x in zip(prior0, lik(0))]; z = sum(a) or 1e-30; f.append([x / z for x in a])
    for t in range(1, T_upto + 1):
        M = trans(t); prev = f[-1]; L = lik(t)
        a = [sum(prev[i] * M[i][j] for i in range(S)) * L[j] for j in range(S)]; z = sum(a) or 1e-30; f.append([x / z for x in a])
    post = f[T_upto]
    if smooth_at is not None:
        bvec = [1.0] * S
        for t in range(T_upto - 1, smooth_at - 1, -1):
            M = trans(t + 1); L = lik(t + 1)
            bvec = [sum(M[i][j] * L[j] * bvec[j] for j in range(S)) for i in range(S)]; z = sum(bvec) or 1e-30; bvec = [x / z for x in bvec]
        x = [f[smooth_at][i] * bvec[i] for i in range(S)]; z = sum(x) or 1e-30; post = [v / z for v in x]
    out = {h: 0.0 for h in HEIGHT_VALUES}
    for i, p in enumerate(post): out[hval[i]] += p
    return out


def _ideal_expected(ep, q):
    T = ep["T"]
    same = [r for r in ep["reports"] if r["entity"] == q["entity"] and r["key"] == q["key"] and r["t"] == q["t"]
            and r["kind"] == "observation" and SOURCES[r["source"]] >= 0.7]
    if q["key"] == "height":
        d = _ideal_height(ep, q["entity"], T, None if q["type"] in ("present", "still_on_table") else q["t"])
    elif q["type"] in ("present", "still_on_table"): d = _ideal_post(ep, q["entity"], q["key"], T)
    else: d = _ideal_post(ep, q["entity"], q["key"], T, smooth_at=q["t"])
    if len({r["value"] for r in same}) >= 2:
        top = sorted(d.values(), reverse=True)
        if top[1] >= 0.2: return "conflit"
    if q["type"] == "still_on_table":
        p = d["table"]
        return ("oui" if q["truth"] else "?") if p >= 0.9 else (("non" if not q["truth"] else "?") if p <= 0.1 else "?")
    v, p = max(d.items(), key=lambda kv: kv[1])
    return v if p >= 0.9 and v == q["truth"] else "?"


def train_episodes(n=20): return [episode(5000 + i) for i in range(n)]


def dev_episodes(n=10): return [episode(6000 + i) for i in range(n)]


def sealed_episodes(): return [episode(77000 + i) for i in range(12)]


def sealed_hash():
    return hashlib.sha256(json.dumps(sealed_episodes(), sort_keys=True, default=str).encode()).hexdigest()


SEALED_SHA256 = "b5fb2ed3c671167d99a016d7efdeee184ef64d7725dd1b43fa53746ad571d62b"   # fingerprint of the sealed set (frozen at its creation, never evaluated before)

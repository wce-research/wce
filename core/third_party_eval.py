"""Evaluation of the unified state on the third-party bench (adapter: the only place that knows both conventions).

The bench (core.third_party_discourse) does not know the agent; the agent never sees the truth of the scenarios:
it receives the sensor facts, the speakers' sentences and the declared record of the speakers.
"""
from __future__ import annotations

import time as _time
from collections import Counter, defaultdict

import torch

from core.discourse import AMBIGUOUS, HYPOTHETICAL, NONEXISTENT
from core.independent_agent import BELIEF, FACT
from core.third_party_discourse import NON_ENGAGEMENT, banc_tiers
from core.unified import (
    ABSTAIN, ANSWER, ASK, CONFLICTED, LIKELY, REFUTED, SUPPORTED, IsotonicCalibrator, RiskPolicy, UnifiedCognitiveState,
    calibrate_threshold, calibration_report,
)

SORTE = {"cube": "cube", "balle": "ball", "livre": "book", "clé": "key", "boîte": "box"}
TEINTE = {"rouge": "red", "bleu": "blue", "vert": "green", "jaune": "yellow"}
LIEU = {"table": ("on", "table"), "étagère": ("on", "shelf"), "sol": ("on", "floor")}
INV_TEINTE = {v: k for k, v in TEINTE.items()}
INV_LIEU = {f"{v[0]}:{v[1]}": k for k, v in LIEU.items()}
SYSTEMS = {
    "unified_calibrated": dict(source_calibrated=True, policy=True, calibrated=True),
    "unified_uncalibrated": dict(source_calibrated=True, policy=True, calibrated=False),
    "unified_no_source": dict(source_calibrated=False, policy=True, calibrated=True),
    "fixed_threshold_0.9": dict(source_calibrated=True, policy=False, calibrated=False),
}


def _fact(obj, champ, valeur):
    if champ == "sorte": return ("type", obj), SORTE[valeur]
    if champ == "teinte": return ("color", obj), TEINTE[valeur]
    if champ == "lieu": return ("loc", obj, "now"), LIEU[valeur]
    raise KeyError(champ)


def new_state(lexicon, scenario, source_calibrated=True, policy=True, calibrator=None, thresholds=None, answer_calibrator=None):
    pol = RiskPolicy(thresholds=thresholds) if policy else RiskPolicy(thresholds={"observation": 0.9, "query": 0.9, "action": 0.9})
    u = UnifiedCognitiveState(lexicon, policy=pol, calibrator=calibrator, source_calibrated=source_calibrated,
                              answer_calibrator=answer_calibrator)
    u.reliability.stats["capteur"] = [60.0, 1.0]
    for src, (ok, bad) in scenario["historique"].items(): u.reliability.stats[src] = [3.0 + ok, 1.5 + bad]
    for obj, champs in scenario["objets"].items():               # types first (inventory), then declared sensors
        if ("sorte" in [c for o, c in scenario["capteurs"] if o == obj]):
            k, v = _fact(obj, "sorte", champs["sorte"]); u.observe("capteur", k, v)
    for obj, champ in scenario["capteurs"]:
        if champ != "sorte":
            k, v = _fact(obj, champ, scenario["objets"][obj][champ]); u.observe("capteur", k, v)
    return u


def to_third_party(ans, intent):
    """Agent answer -> bench conventions."""
    out = ans.output
    if ans.action == ANSWER:
        if intent == "command": return f"agir:{ans.value}"
        if out in ("SUPPORTED",): return "oui"
        if out in ("REFUTED",): return "non"
        if out in INV_LIEU: return INV_LIEU[out]
        if out in INV_TEINTE: return INV_TEINTE[out]
        return str(out)
    if ans.reason == "INVALID_PREMISE": return "présupposé"
    if ans.reason == CONFLICTED: return "conflit"
    return "?"


def _intent(text):
    t = text.lower()
    return "command" if t.split()[0] in ("prends", "pose", "donne") else "where" if t.startswith("où") else \
        "color" if "couleur" in t else "check"


def run_scenario(lexicon, scenario, **kw):
    u = new_state(lexicon, scenario, **kw)
    rows, t0 = [], _time.perf_counter()
    for tour in scenario["tours"]:
        if tour[0] == "dit": u.say(tour[1], tour[2])
        elif tour[0] == "capteur":
            k, v = _fact(tour[1], tour[2], tour[3]); u.contradict(k, v)
        else:
            text, expected = tour[1], tour[2]
            intent = _intent(text)
            a = u.query(text)
            got = to_third_party(a, intent)
            committed = got not in NON_ENGAGEMENT
            rows.append({"scenario": scenario["id"], "category": scenario["catégorie"], "origin": scenario["origine"],
                         "text": text, "expected": expected, "got": got, "status": a.status, "action": a.action,
                         "committed": committed, "correct": got == expected or (expected in NON_ENGAGEMENT and not committed),
                         "exact": got == expected, "false_confident": committed and got != expected,
                         "confidence": a.confidence, "identity_confidence": a.identity_confidence, "entity": a.entity,
                         "identity_correct": _identity_correct(scenario, text, a.entity),
                         "asked": a.action == ASK, "likely": a.status == LIKELY, "answer": a})
    ident = []
    for c in u.claims.values():
        if c.entity is None: continue
        ref = u.state.entities_mentioned.get(c.referent)
        desc = ref.attrs.get("attr") if ref else None
        if desc is None: continue
        truth = TEINTE.get(scenario["objets"][c.entity]["teinte"])
        ident.append((c.identity_confidence, truth == desc))
    beliefs = []
    for key in list(u.belief.graph.props):
        b = u.belief.graph.belief(key)
        if b.status not in (FACT, BELIEF) or key[1] not in scenario["objets"]: continue
        o = scenario["objets"][key[1]]
        truth = TEINTE[o["teinte"]] if key[0] == "color" else LIEU.get(o["lieu"]) if key[0] == "loc" and key[2] == "now" else None
        if truth is not None: beliefs.append((b.posterior, b.value == truth))
    return rows, {"identity": ident, "belief": beliefs, "state": u, "seconds": _time.perf_counter() - t0}


def _identity_correct(scenario, text, entity):
    """Truth (evaluation): scenario objects compatible with the words of the question."""
    if entity in (None, AMBIGUOUS, NONEXISTENT, HYPOTHETICAL): return None
    words = text.lower().replace("'", " ").replace("?", " ").replace(".", " ").split()
    match = []
    for oid, o in scenario["objets"].items():
        if o["sorte"] not in words: continue
        teintes = [w for w in words if w.rstrip("e") in TEINTE or w in TEINTE]
        if teintes and not any(w.startswith(o["teinte"][:4]) for w in teintes): continue
        match.append(oid)
    if len(match) != 1: return None
    return entity == match[0]


def metrics(rows):
    n = max(1, len(rows))
    exp_commit = [r for r in rows if r["expected"] not in NON_ENGAGEMENT]
    return {"n": len(rows), "success": round(sum(r["correct"] for r in rows) / n, 4),
            "false_confident": round(sum(r["false_confident"] for r in rows) / n, 4),
            "abstention": round(sum(not r["committed"] for r in rows) / n, 4),
            "clarification": round(sum(r["asked"] for r in rows) / n, 4),
            "likely": round(sum(r["likely"] for r in rows) / n, 4),
            "exact_class": round(sum(r["exact"] for r in rows) / n, 4),
            "coverage": round(sum(r["committed"] and r["correct"] for r in exp_commit) / max(1, len(exp_commit)), 4),
            "identity_accuracy": _ratio([r["identity_correct"] for r in rows if r["identity_correct"] is not None])}


def _ratio(xs): return round(sum(xs) / len(xs), 4) if xs else None


def run(lexicon, bench=None, system="unified_calibrated", calibration=None):
    bench = bench or banc_tiers()
    cfg = dict(SYSTEMS[system])
    calibration = calibration if cfg.get("calibrated", True) else None
    cal = calibration or {}
    rows, ident, beliefs, secs = [], [], [], 0.0
    for scn in bench:
        r, info = run_scenario(lexicon, scn, source_calibrated=cfg["source_calibrated"], policy=cfg["policy"],
                               calibrator=cal.get("identity"), thresholds=cal.get("thresholds"),
                               answer_calibrator=cal.get("answer"))
        rows += r; ident += info["identity"]; beliefs += info["belief"]; secs += info["seconds"]
    m = metrics(rows)
    m["identity_calibration"] = calibration_report([c for c, _ in ident], [y for _, y in ident])
    m["belief_calibration"] = calibration_report([c for c, _ in beliefs], [y for _, y in beliefs])
    m["belief_accuracy"] = _ratio([y for _, y in beliefs])
    m["commitment_identity_accuracy"] = _ratio([y for c, y in ident if c >= 0.65])
    m["seconds"] = round(secs, 3)
    return m, rows


def by(rows, key):
    g = defaultdict(list)
    for r in rows: g[r[key]].append(r)
    return {k: metrics(v) for k, v in sorted(g.items())}


def answer_calibration(rows):
    """Final answer: stated confidence vs accuracy (committed AND "likely" answers)."""
    pts = [(r["confidence"], r["got"] == r["expected"] or (r["likely"] and _likely_ok(r))) for r in rows if r["committed"] or r["likely"]]
    return calibration_report([c for c, _ in pts], [y for _, y in pts])


def _likely_ok(r):
    a = r["answer"]; v = a.value
    if v in INV_LIEU: v = INV_LIEU[v]
    elif v in INV_TEINTE: v = INV_TEINTE[v]
    elif v == "SUPPORTED": v = "oui"
    elif v == "REFUTED": v = "non"
    return v == r["expected"]


# ================================================================ active observation, self-correction, distillation

def active_commands(lexicon, bench=None, max_requests=3):
    """Commands with uncertain identity: passive (ASK / abstention) vs active (observations requested from the sensor)."""
    from core.third_party_discourse import scenarios_commandes
    bench = bench or scenarios_commandes()
    out = Counter()
    for scn in bench:
        for mode in ("passive", "active"):
            u = new_state(lexicon, scn)
            text, expected = scn["tours"][0][1], scn["tours"][0][2]
            a = u.query(text)
            if mode == "active" and a.action != ANSWER:
                rid = next((m.referent for m in reversed(list(u.state.mentions.values())) if m.referent and m.kind != "place"), None)

                def oracle(field_, e, scn=scn):
                    o = scn["objets"].get(e)
                    return None if o is None else TEINTE[o["teinte"]] if field_ == "attr" else LIEU.get(o["lieu"])
                out["requests"] += u.request_observation(rid, oracle, max_requests)
                a = u.query(text)
            got = to_third_party(a, "command")
            out[f"{mode}_executed"] += got.startswith("agir:")
            out[f"{mode}_correct"] += got == expected
            out[f"{mode}_wrong"] += got.startswith("agir:") and got != expected
            out[f"{mode}_asked"] += a.action == ASK
    n = len(bench)
    return {"n": n, **{k: v for k, v in out.items()}, "requests_per_command": round(out["requests"] / max(1, n), 3),
            "passive_success": round(out["passive_correct"] / n, 3), "active_success": round(out["active_correct"] / n, 3)}


def active_observation(lexicon, bench=None):
    """Questions not resolved by identity: request an observation (colour / position) from the sensor simulator."""
    bench = bench or banc_tiers()
    out = Counter()
    for scn in bench:
        u = new_state(lexicon, scn)
        for tour in scn["tours"]:
            if tour[0] == "dit": u.say(tour[1], tour[2]); continue
            if tour[0] == "capteur":
                k, v = _fact(tour[1], tour[2], tour[3]); u.contradict(k, v); continue
            a = u.query(tour[1])
            if a.action != ASK: continue
            out["ask_questions"] += 1
            rid = next((m.referent for m in reversed(list(u.state.mentions.values())) if m.referent and m.kind != "place"), None)
            if rid is None: continue

            def oracle(field_, e, scn=scn):
                o = scn["objets"].get(e)
                if o is None: return None
                return TEINTE[o["teinte"]] if field_ == "attr" else LIEU.get(o["lieu"])
            used = u.request_observation(rid, oracle)
            out["requests"] += used
            _, ev = u.identity(rid)
            if ev and ev.confidence >= u.policy.threshold("query"):
                out["resolved"] += 1
                truth = _identity_correct(scn, tour[1], ev.entity)
                out["resolved_correct"] += bool(truth)
                out["resolved_wrong"] += truth is False
    out["requests_per_resolved"] = round(out["requests"] / max(1, out["resolved"]), 3)
    return dict(out)


def self_correction(lexicon, bench=None):
    """By default: third-party bench + "misleading" scenarios (genuinely wrong answers to correct)."""
    from core.third_party_discourse import scenarios_trompeurs
    return _self_correction(lexicon, bench or (banc_tiers() + scenarios_trompeurs()))


def _self_correction(lexicon, bench):
    """Committed or likely answer, wrong -> a reliable observation contradicts it -> dependencies -> recomputation.
    Harm control: a reliable confirmation on a right answer must not degrade it."""
    bench = bench or banc_tiers()
    c = Counter()
    for scn in bench:
        u = new_state(lexicon, scn)
        for tour in scn["tours"]:
            if tour[0] == "dit": u.say(tour[1], tour[2]); continue
            if tour[0] == "capteur":
                k, v = _fact(tour[1], tour[2], tour[3]); u.contradict(k, v); continue
            text, expected = tour[1], tour[2]
            intent = _intent(text)
            if intent not in ("where", "color"): continue
            a = u.query(text)
            if not (a.action == ANSWER or a.status == LIKELY) or a.entity not in scn["objets"]: continue
            o = scn["objets"][a.entity]
            champ = "lieu" if intent == "where" else "teinte"
            truth_word = o[champ]
            got = to_third_party(a, intent) if a.action == ANSWER else _word(a.value)
            k, v = _fact(a.entity, champ, truth_word)
            if got != truth_word:
                c["wrong_before"] += 1
                u.contradict(k, v)
                new, trace = u.self_correct(a, text)
                c["corrections_attempted"] += 1
                after = to_third_party(new, intent)
                c["fixed"] += after == truth_word
                c["now_abstains"] += after in NON_ENGAGEMENT
                c["still_wrong"] += after not in NON_ENGAGEMENT and after != truth_word
            else:
                c["right_before"] += 1
                u.contradict(k, v)
                new, _ = u.self_correct(a, text)
                after = to_third_party(new, intent)
                c["harmful"] += after not in NON_ENGAGEMENT and after != truth_word
    att = max(1, c["corrections_attempted"])
    return {**dict(c), "correction_precision": round(c["fixed"] / max(1, c["fixed"] + c["still_wrong"]), 3),
            "correction_recall": round(c["fixed"] / max(1, c["wrong_before"]), 3),
            "harmful_rate": round(c["harmful"] / max(1, c["right_before"]), 3), "attempted": att}


def _word(v):
    if v in INV_LIEU: return INV_LIEU[v]
    if v in INV_TEINTE: return INV_TEINTE[v]
    return v


def calibration_bench(n=120, seed=11):
    """CALIBRATION partition: same second generator, other seed; never the test scenarios."""
    from core.third_party_discourse import scenarios_generes
    return scenarios_generes(n, seed)


def calibrate_on(lexicon, calib_bench=None):
    """Isotonic calibrators (identity, answer) + question threshold by empirical risk, on the calibration."""
    calib_bench = calib_bench or calibration_bench()
    rows, ident = [], []
    for scn in calib_bench:
        r, info = run_scenario(lexicon, scn); rows += r; ident += info["identity"]
    pts = [(r["confidence"], _likely_ok(r) if r["likely"] else r["got"] == r["expected"]) for r in rows if r["committed"] or r["likely"]]
    answer = IsotonicCalibrator().fit([c for c, _ in pts], [y for _, y in pts])
    identity = IsotonicCalibrator().fit([c for c, _ in ident], [y for _, y in ident])
    cal_pts = [(float(answer(c)), y) for c, y in pts]
    q = calibrate_threshold([c for c, _ in cal_pts], [y for _, y in cal_pts], cost_wrong=20.0)
    return {"answer": answer, "identity": identity if identity.x is not None else None,
            "thresholds": {"query": max(q, 0.9)}, "n_answer_points": len(pts), "n_identity_points": len(ident),
            "raw_answer_calibration": calibration_report([c for c, _ in pts], [y for _, y in pts])}


def memory_scaling(lexicon, lengths=(50, 100, 500), seed=0):
    """Long discourses (reference_world generator, independent of the third-party bench) through the unified state."""
    from core.reference_world import long_episode
    out = {}
    for n in lengths:
        ep = long_episode(n, "physical", seed)
        u = UnifiedCognitiveState(lexicon)
        u.reliability.stats["capteur"] = [60.0, 1.0]; u.reliability.stats["dora"] = [30.0, 1.0]
        for (key, value) in ep.given:
            if key[0] == "type": u.observe("capteur", key, value)
        for (key, value) in ep.given:
            if key[0] == "attr": u.observe("capteur", ("color", key[1]), value)
        t0 = _time.perf_counter()
        made = []
        for turn in ep.turns:
            _, claims = u.say("dora", turn.text)
            gold = turn.mentions[0]["gold"] if turn.mentions else None
            made += [(c, gold) for c in claims]
        dt = _time.perf_counter() - t0
        judged = [(c.entity == g) for c, g in made if c.entity is not None and g in ep.entities]
        size = u.memory_size()
        out[n] = {"seconds_per_sentence": round(dt / n, 4), **size, "revisions": len(u.revisions),
                  "identity_commitments": len(judged), "identity_accuracy": _ratio(judged),
                  "frozen_claims": sum(c.frozen for c in u.claims.values()),
                  "tms": u.tms.counts()}
    return out


def minimal_self_correction(lexicon, bench=None):
    """Challenge without a new observation ("non, c'est faux") on WRONG answers: the minimal revision
    must find the faulty dependency (misleading source / identity) and revise only that one."""
    from core.third_party_discourse import scenarios_trompeurs
    bench = bench or scenarios_trompeurs()
    c = Counter(); sizes = []; causes = Counter()
    for scn in bench:
        u = new_state(lexicon, scn)
        for tour in scn["tours"]:
            if tour[0] == "dit": u.say(tour[1], tour[2]); continue
            if tour[0] != "demande": continue
            text = tour[1]; intent = _intent(text)
            a = u.query(text)
            if not (a.action == ANSWER or a.status == LIKELY) or a.entity not in scn["objets"]: continue
            truth = scn["objets"][a.entity]["lieu" if intent == "where" else "teinte"]
            got = to_third_party(a, intent) if a.action == ANSWER else _word(a.value)
            if got == truth: c["right_not_contested"] += 1; continue
            c["contested_wrong"] += 1
            new, info = u.minimal_revision(a, text)
            after = to_third_party(new, intent)
            c["now_right"] += after == truth; c["now_abstains"] += after in NON_ENGAGEMENT
            c["still_wrong"] += after not in NON_ENGAGEMENT and after != truth
            sizes.append((len(info["revised"]), info["support_size"])); causes[info["cause"]] += 1
    n = max(1, c["contested_wrong"])
    return {**dict(c), "no_longer_wrong_rate": round((c["now_right"] + c["now_abstains"]) / n, 3),
            "mean_revised_nodes": round(sum(a for a, _ in sizes) / max(1, len(sizes)), 3),
            "mean_support_nodes": round(sum(b for _, b in sizes) / max(1, len(sizes)), 3), "causes": dict(causes)}

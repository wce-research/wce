"""Calibration of sources in the independent environment: labelled history (training seeds / worlds),
hierarchical priors (calibration seeds), MNAR audit, calibration measurements (sources, beliefs per key)."""
from __future__ import annotations

from collections import Counter, defaultdict

from core.independent_agent import BELIEF, FACT, HYPOTHESIS, KIND_DOMAIN, tokenize
from core.source_reliability import SourceReliabilityModel, mnar_audit, selection_rates, source_type
from core.unified import calibration_report

KEY_TYPES = ("color", "loc", "loc0", "height", "owner", "event")


def key_type(key):
    if key[0] == "loc": return "loc0" if key[2] == 0 else "loc"
    return key[0]


def _truth(world, key, value):
    if key[0] == "event": return value                                       # an emitted event is always true here
    return world.facts().get(key)


def emissions(agent, worlds):
    """(source, key type, correct|None, parsed) for each emitted observation (evaluation / training)."""
    out = []
    for x in worlds:
        w = x["world"]
        agent.new_world(); n0 = len(getattr(agent, "parse_log", [])); agent.ingest(x["observations"], update=False)
        parsed_texts = {(p["source"], p["text"]) for p in agent.parse_log[n0:] if p["parsed"]} if hasattr(agent, "parse_log") else set()
        for ob in x["observations"]:
            if ob.key[0] == "type" or ob.implied: continue
            key = ob.key if ob.key[0] != "loc" else ("loc", ob.key[1], 0 if ob.key[2] == 0 else "now")
            t = _truth(w, key, ob.value)
            ok = None if t is None else (ob.value == t)
            parsed = True if ob.text is None else (ob.source, ob.text) in parsed_texts
            out.append((ob.source, key_type(key), ok, parsed, ob.text is not None))
    return out


def fit_supervised(emits, mode="lower", type_priors=None, selection=True, discount=1.0, level=0.8):
    """Beta-binomial on the labelled EMITTED history; selection correction towards the parsed channel (text)."""
    m = SourceReliabilityModel(type_priors=type_priors, mode=mode, discount=discount, level=level)
    for s, kt, ok, parsed, is_text in emits: m.observe(s, ok, kt)
    if selection:
        m.selection = selection_rates([(s, ok, parsed) for s, kt, ok, parsed, is_text in emits if is_text and ok is not None])
    return m


def fit_parsed_only(emits, mode="lower", type_priors=None, level=0.8):
    """Variant: history restricted to the PARSED claims (what the agent actually ingested), without correction."""
    m = SourceReliabilityModel(type_priors=type_priors, mode=mode, level=level)
    for s, kt, ok, parsed, is_text in emits:
        if parsed: m.observe(s, ok, kt)
    return m


def hierarchical_priors(emits_by_seed):
    """Type priors by empirical Bayes on the CALIBRATION seeds (never the test): each (source, seed)
    is a realisation of the type."""
    m = SourceReliabilityModel(mode="mean")
    for seed, emits in emits_by_seed.items():
        for s, kt, ok, parsed, is_text in emits: m.observe(f"{s}@{seed}", ok, kt)
    return m.fit_empirical_bayes()


def install(agent, model):
    agent.reliability = model
    return agent


# ================================================================ measurements

def reliability_points(model, emits, parsed_only=True):
    """(announced effective reliability, correct claim) on the parsed channel of the TEST worlds."""
    pts = defaultdict(list)
    for s, kt, ok, parsed, is_text in emits:
        if ok is None or (parsed_only and not parsed): continue
        p = model.estimate_for(s, kt) if hasattr(model, "estimate_for") else model.estimate(s)
        pts[source_type(s)].append((p, ok)); pts["all"].append((p, ok))
    return {t: calibration_report([c for c, _ in v], [y for _, y in v]) for t, v in pts.items()}, pts


def source_table(model, emits):
    """Per source: posterior mean, actual accuracy (parsed / emitted), error, interval, coverage."""
    acc = defaultdict(Counter)
    for s, kt, ok, parsed, is_text in emits:
        if ok is None: continue
        acc[s]["n_emit"] += 1; acc[s]["ok_emit"] += ok
        if parsed: acc[s]["n_parsed"] += 1; acc[s]["ok_parsed"] += ok
    out = {}
    for s, c in acc.items():
        rec = model.record(s) if hasattr(model, "record") else None
        true_parsed = c["ok_parsed"] / max(1, c["n_parsed"])
        eff = model.estimate(s)
        lo, hi = rec.interval(0.9) if rec else (None, None)
        out[s] = {"type": source_type(s), "effective": round(eff, 4), "posterior_mean": round(rec.mean, 4) if rec else None,
                  "interval90": (round(lo, 3), round(hi, 3)) if rec else None, "true_parsed": round(true_parsed, 4),
                  "true_emitted": round(c["ok_emit"] / max(1, c["n_emit"]), 4), "error": round(eff - true_parsed, 4),
                  "covered": (lo <= c["ok_emit"] / max(1, c["n_emit"]) <= hi) if rec else None}
    return out


def belief_points(agent, worlds, pts=None, conflicted=None, ingest=True):
    """Beliefs (value, posterior) vs truth, per key type; CONFLICTED status counted separately.
    ingest=False: the current state of the agent is read (ONE single ingestion per world, otherwise the lexical memory inflates)."""
    pts = pts if pts is not None else defaultdict(list); conflicted = conflicted if conflicted is not None else Counter()
    for x in worlds:
        w = x["world"]; facts = w.facts()
        if ingest: agent.new_world(); agent.ingest(x["observations"])
        for key in list(agent.belief.graph.props):
            b = agent.belief.graph.belief(key)
            if key[0] == "event":
                last = [e for e in w.events if e.patient == key[1]]
                truth = (last[-1].rel, last[-1].target) if last else None
            else:
                truth = facts.get(key)
            if truth is None or b.value is None: continue
            if b.status == "CONFLICTED": conflicted[key_type(key)] += 1
            pts[key_type(key)].append((b.posterior, b.value == truth)); pts["all"].append((b.posterior, b.value == truth))
    return {t: calibration_report([c for c, _ in v], [y for _, y in v]) for t, v in pts.items()}, dict(conflicted)


def evaluate_answers(agent, worlds):
    """A single ingestion per world: answers + beliefs read on the same state."""
    from core.independent_agent import ABSTAIN
    rows, pts, conf = [], defaultdict(list), Counter()
    for x in worlds:
        agent.new_world(); agent.ingest(x["observations"])
        belief_points(agent, [x], pts, conf, ingest=False)
        for it in x["items"]:
            a = agent.answer(it.text)
            rows.append({"out": a.output, "orig": it.gold, "answer": a, "item": it})
    return rows, {t: calibration_report([c for c, _ in v], [y for _, y in v]) for t, v in pts.items()}, dict(conf), pts


def predictive_coverage(model, emits, level=0.9, parsed_only=True):
    """For each source: is the proportion of its correct claims on the TEST within the interval?
    Beta credible (parameter uncertainty alone) vs beta-binomial PREDICTIVE (parameter + noise of the n future ones)."""
    from core.temporal_belief import beta_binomial_interval
    per = defaultdict(lambda: [0, 0])
    for s, kt, ok, parsed, is_text in emits:
        if ok is None or (parsed_only and not parsed): continue
        per[s][0] += ok; per[s][1] += 1
    out = {}
    for s, (k, n) in per.items():
        if n < 3: continue
        rec = model.record(s); frac = k / n
        lo, hi = rec.interval(level); plo, phi = beta_binomial_interval(rec.alpha, rec.beta, n, level)
        out[s] = {"n": n, "frac": round(frac, 3), "credible": lo <= frac <= hi, "predictive": plo <= frac <= phi,
                  "w_credible": hi - lo, "w_predictive": phi - plo}
    return out

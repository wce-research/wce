"""Evaluation of the temporal dynamics on the temporal bench (adapter; the bench is unaware of the system)."""
from __future__ import annotations

import time
from collections import Counter, defaultdict

from core.source_reliability import SourceReliabilityModel
from core.temporal_bench import DOMAINS, HEIGHTS, dev_episodes, episode, sealed_episodes, train_episodes
from core.temporal_belief import (
    ASSERTED, CONFLICTED, KNOWN, LIKELY, STALE, HazardModel, JointDerivedState, KeyDynamics, TemporalMemory,
    beta_binomial_interval,
)
from core.unified import calibration_report

SYSTEMS_T = ("last_observation", "static_votes", "no_hazard", "temporal")


def fit_training(episodes, mode="mean"):
    """Rate of change per key type + reliabilities of the sources (observations and events) + probability of
    elevation (height ≠ height of the support), on the TRAINING."""
    hz = HazardModel().fit([tr for ep in episodes for tr in ep["train_trajectories"]])
    bump = n = 0
    for ep in episodes:
        for e, tr in ep["truth"].items():
            for l, h in zip(tr["loc"], tr["height"]): bump += h != HEIGHTS[l]; n += 1
    hz.bump = (bump + 1) / (n + 2)
    toggles = exp = 0
    for ep in episodes:
        for e, tr in ep["truth"].items():
            off = [h != HEIGHTS[l] for l, h in zip(tr["loc"], tr["height"])]
            toggles += sum(a != b for a, b in zip(off, off[1:])); exp += len(off) - 1
    hz.toggle = (toggles + 1) / (exp + 50)                           # toggle rate of the elevation (learned)
    starts = sum(tr["height"][0] != HEIGHTS[tr["loc"][0]] for ep in episodes for tr in ep["truth"].values())
    n0 = sum(len(ep["truth"]) for ep in episodes)
    hz.mod_prior = [1 - (starts + 1) / (n0 + 2), (starts + 1) / (n0 + 2)]       # initial elevation (learned)
    rel = SourceReliabilityModel(mode=mode, global_prior=(2.0, 1.0))
    for ep in episodes:
        for r in ep["reports"]:
            truth = ep["truth"][r["entity"]][r["key"]][r["t"]]
            name = r["source"] + ("|event" if r["kind"] == "event" else "")
            rel.observe(name, r["value"] == truth, r["key"])
    return hz, rel


def build_memory(ep, hz, rel, system):
    if system == "no_hazard":
        hz = HazardModel({k: (1e-6, "STATIC_LIKE") for k in DOMAINS})
    mem = TemporalMemory(hz, DOMAINS)
    for r in ep["reports"]:
        st = mem.state(r["entity"], r["key"])
        rr = rel.estimate_for(r["source"] + ("|event" if r["kind"] == "event" else ""), r["key"])
        if r["kind"] == "event": st.event(r["t"], r["value"], r["source"], rr, r["ingested_at"])
        else: st.observe(r["t"], r["value"], r["source"], rr, r["ingested_at"])
    return mem


def _baseline_answer(ep, q, rel, system):
    """Naive references: last observation, or votes without time (one source = one vote)."""
    reps = [r for r in ep["reports"] if r["entity"] == q["entity"] and r["key"] == q["key"] and r["kind"] == "observation"
            and (r["t"] <= q["t"] if q["type"] != "present" else True)]
    if q["type"] in ("present", "still_on_table"): reps = [r for r in reps]
    if not reps: return {}, None
    if system == "last_observation":
        r = max(reps, key=lambda r: (r["t"], rel.estimate_for(r["source"], r["key"])))
        p = rel.estimate_for(r["source"], r["key"]); K = len(DOMAINS[q["key"]])
        return {v: (p if v == r["value"] else (1 - p) / (K - 1)) for v in DOMAINS[q["key"]]}, r["t"]
    import math
    latest = {}
    for r in reps: latest[r["source"]] = r
    K = len(DOMAINS[q["key"]]); sc = defaultdict(float)
    for s, r in latest.items():
        p = min(rel.estimate_for(s, r["key"]), 0.99); sc[r["value"]] += math.log(p * (K - 1) / (1 - p))
    z = sum(math.exp(x) for x in sc.values()) + (K - len(sc))
    return {v: (math.exp(sc[v]) if v in sc else 1.0) / z for v in DOMAINS[q["key"]]}, None


def derived_height(d_height, d_loc, bump):
    """Product of experts: height deduced from the support (with possible elevation) × own model of the height."""
    derived = {h: 0.0 for h in d_height}
    for l, pl in d_loc.items():
        base = HEIGHTS[l]; up = min(base + 0.5, 2.5)
        derived[base] = derived.get(base, 0.0) + pl * (1 - bump)
        derived[up] = derived.get(up, 0.0) + pl * bump
    prod = {h: d_height[h] * max(derived.get(h, 0.0), 1e-6) for h in d_height}
    z = sum(prod.values()) or 1.0
    return {h: v / z for h, v in prod.items()}


def answer(ep, q, mem, rel, system):
    if system in ("last_observation", "static_votes"):
        d, _ = _baseline_answer(ep, q, rel, system); conflict = False
    else:
        st = mem.state(q["entity"], q["key"])
        if q["type"] in ("present", "still_on_table"):
            d = st.current(ep["T"])
            conflict = st.conflict_at(ep["T"]) and sorted(d.values(), reverse=True)[1] >= 0.2     # conflict AT THE PRESENT only
        else:
            d = st.historical(q["t"], ep["T"]); conflict = st.conflict_at(q["t"]) and sorted(d.values(), reverse=True)[1] >= 0.2
        if q["key"] == "height" and system == "temporal":                       # height = f(support, elevation): joint state
            js = JointDerivedState(mem.state(q["entity"], "loc"), (0, 1), lambda l, b: min(HEIGHTS[l] + 0.5 * b, 2.5),
                                   getattr(mem.hz, "toggle", 0.03), DOMAINS["height"], getattr(mem.hz, "mod_prior", None))
            for r in mem.state(q["entity"], "height").reports:
                if r.kind == "observation": js.observe(r.t, r.value, r.source, r.reliability)
            d = js.current(ep["T"]) if q["type"] in ("present", "still_on_table") else js.historical(q["t"], ep["T"])
    if not d: return "?", 0.0, None, d
    if conflict: return "conflit", max(d.values()), None, d
    if q["type"] == "still_on_table":
        p = d.get("table", 0.0)
        return ("oui" if p >= 0.9 else "non" if p <= 0.1 else "?"), max(p, 1 - p), ("table" if p >= 0.5 else "not"), d
    v, p = max(d.items(), key=lambda kv: kv[1])
    return (v if p >= 0.9 else "?"), p, v, d


def evaluate(system, episodes, hz, rel):
    rows = []
    t0 = time.perf_counter()
    for ep in episodes:
        mem = build_memory(ep, hz, rel, system) if system not in ("last_observation", "static_votes") else None
        for q in ep["questions"]:
            got, p, best, d = answer(ep, q, mem, rel, system)
            truth = q["truth"]
            committed = got not in ("?", "conflit")
            correct_value = (best == truth) if q["type"] != "still_on_table" else ((best == "table") == truth)
            pset = []
            if d and q["type"] != "still_on_table":
                acc = 0.0
                for v, pv in sorted(d.items(), key=lambda kv: -kv[1]):
                    pset.append(v); acc += pv
                    if acc >= 0.9: break
            rows.append({"type": q["type"], "key": q["key"], "expected": q["expected"], "got": got, "p": p,
                         "committed": committed, "value_ok": correct_value,
                         "correct": got == q["expected"] or (not committed and q["expected"] in ("?", "conflit")),
                         "false_confident": committed and got != q["expected"],
                         "covered": (truth in pset) if pset else None, "width": len(pset)})
    secs = time.perf_counter() - t0
    return rows, secs


def metrics(rows):
    n = max(1, len(rows))
    out = {"n": len(rows), "success": round(sum(r["correct"] for r in rows) / n, 4),
           "false_confident": round(sum(r["false_confident"] for r in rows) / n, 4),
           "abstention": round(sum(not r["committed"] for r in rows) / n, 4)}
    tp = sum(r["got"] == "conflit" and r["expected"] == "conflit" for r in rows)
    fp = sum(r["got"] == "conflit" and r["expected"] != "conflit" for r in rows)
    fn = sum(r["got"] != "conflit" and r["expected"] == "conflit" for r in rows)
    out["conflict"] = {"precision": round(tp / max(1, tp + fp), 3), "recall": round(tp / max(1, tp + fn), 3), "tp": tp, "fp": fp, "fn": fn}
    ece = {}
    for key in ("loc", "height", "owner", "color"):
        for typ in ("present", "past"):
            sub = [r for r in rows if r["key"] == key and r["type"] == typ and r["p"] > 0]
            if sub: ece[f"{key}_{typ}"] = calibration_report([r["p"] for r in sub], [r["value_ok"] for r in sub])["ece"]
    out["ece"] = ece
    cov = {}
    for key in ("loc", "height", "owner", "color"):
        sub = [r for r in rows if r["key"] == key and r["covered"] is not None]
        if sub: cov[key] = {"coverage90": round(sum(r["covered"] for r in sub) / len(sub), 3), "width": round(sum(r["width"] for r in sub) / len(sub), 2)}
    out["prediction_sets"] = cov
    by = defaultdict(list)
    for r in rows: by[r["type"]].append(r)
    out["by_type"] = {k: {"success": round(sum(r["correct"] for r in v) / len(v), 3), "fc": round(sum(r["false_confident"] for r in v) / len(v), 3)} for k, v in by.items()}
    return out


def compare(episodes=None, train=None):
    hz, rel = fit_training(train or train_episodes(60))
    res = {}
    for s in SYSTEMS_T:
        rows, secs = evaluate(s, episodes or dev_episodes(), hz, rel)
        res[s] = metrics(rows) | {"seconds": round(secs, 2)}
    res["hazards"] = {k: (round(d.hazard, 4), d.klass) for k, d in hz.dyn.items()}
    return res


def source_interval_coverage(episodes=None, train=None, level=0.9):
    """Beta CREDIBLE interval vs beta-binomial PREDICTIVE interval: does it contain the proportion of correct reports
    of a source in a test episode?"""
    _, rel = fit_training(train or train_episodes(60))
    cov = Counter(); widths = defaultdict(list)
    for ep in episodes or dev_episodes():
        per = defaultdict(lambda: [0, 0])
        for r in ep["reports"]:
            name = r["source"] + ("|event" if r["kind"] == "event" else "")
            per[name][0] += r["value"] == ep["truth"][r["entity"]][r["key"]][r["t"]]; per[name][1] += 1
        for s, (ok, n) in per.items():
            if n < 3: continue
            rec = rel.record(s); frac = ok / n
            lo, hi = rec.interval(level); cov["credible"] += lo <= frac <= hi; widths["credible"].append(hi - lo)
            plo, phi = beta_binomial_interval(rec.alpha, rec.beta, n, level); cov["predictive"] += plo <= frac <= phi
            widths["predictive"].append(phi - plo); cov["n"] += 1
    return {k: round(cov[k] / max(1, cov["n"]), 3) for k in ("credible", "predictive")} | \
        {f"width_{k}": round(sum(v) / len(v), 3) for k, v in widths.items()} | {"n": cov["n"]}


def long_stream(T, seed=1):
    """Long stream: time, memory (reports vs segments), accuracy and calibration of the historical questions."""
    ep = episode(seed, T=T)
    hz, rel = fit_training(train_episodes(8))
    t0 = time.perf_counter(); mem = build_memory(ep, hz, rel, "temporal")
    rows, _ = evaluate("temporal", [ep], hz, rel)
    secs = time.perf_counter() - t0
    m = metrics(rows); size = mem.size(ep["T"])
    hist = [r for r in rows if r["type"] == "past"]
    return {"T": T, "seconds": round(secs, 2), **size, "historical_value_accuracy": round(sum(r["value_ok"] for r in hist) / max(1, len(hist)), 3),
            "success": m["success"], "false_confident": m["false_confident"],
            "ece_past": calibration_report([r["p"] for r in hist], [r["value_ok"] for r in hist])["ece"]}


def multi_bench(n_seeds=8, n_train=60, n_eval=30, log=print):
    """BENCH seeds: each seed has its own training episodes (rates, reliabilities) and evaluation episodes."""
    from core.grounded_benchmark import paired, summarize_values
    per = []
    for k in range(n_seeds):
        train = [episode(100000 + 1000 * k + i) for i in range(n_train)]
        evals = [episode(200000 + 1000 * k + i) for i in range(n_eval)]
        hz, rel = fit_training(train)
        out = {}
        for s in SYSTEMS_T:
            rows, secs = evaluate(s, evals, hz, rel); out[s] = metrics(rows)
        per.append(out)
        log(k, {s: (out[s]["success"], out[s]["false_confident"]) for s in SYSTEMS_T})
    keys = ("success", "false_confident", "abstention")
    summary = {s: {k: summarize_values([o[s][k] for o in per]) for k in keys} for s in SYSTEMS_T}
    ece = {s: {k: summarize_values([o[s]["ece"].get(k) for o in per]) for k in per[0]["temporal"]["ece"]} for s in SYSTEMS_T}
    cov = {s: {k: summarize_values([o[s]["prediction_sets"].get(k, {}).get("coverage90") for o in per]) for k in ("loc", "height", "owner", "color")}
           for s in SYSTEMS_T}
    conf = {s: {m: sum(o[s]["conflict"][m] for o in per) for m in ("tp", "fp", "fn")} for s in SYSTEMS_T}
    for s, c in conf.items():
        p_, r_ = c["tp"] / max(1, c["tp"] + c["fp"]), c["tp"] / max(1, c["tp"] + c["fn"])
        c.update({"precision": round(p_, 3), "recall": round(r_, 3), "f1": round(2 * p_ * r_ / max(1e-9, p_ + r_), 3)})
    comps = {f"temporal_vs_{b}": {k: paired([o["temporal"][k] for o in per], [o[b][k] for o in per]) for k in keys}
             for b in ("no_hazard", "last_observation", "static_votes")}
    return {"summary": summary, "ece": ece, "coverage": cov, "conflicts": conf, "comparisons": comps, "per_seed": per}

"""Multi-seed experiment: source reliability model -> evidence -> beliefs -> entitlement to assert -> L1 identity.

Split: TRAINING seeds = train partition of each seed (labelled history of the sources);
CALIBRATION seeds 200–205 (hierarchical priors, lexicon: 100–111); no measurement on the test tunes anything.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict

from core.entitlement import ASSERTED
from core.epistemic_benchmark import (
    COMMITTED, error_attribution, evaluate, knowability_alignment, pareto, prepare, score, train_lexical_posterior,
)
from core.grounded_benchmark import SYSTEMS, paired, seed_data, summarize_values
from core.knowledge_boundary import relabel_dataset, true_reliabilities
from core.source_calibration import (
    belief_points, emissions, fit_supervised, hierarchical_priors, install, mnar_audit, reliability_points, source_table,
)
from core.unified import calibration_report

SOURCE_CAL_SEEDS = tuple(range(200, 206))
PRIORS_PATH = os.path.join("results", "source_type_priors.json")
SYSTEMS_S = ("em", "em_entitled", "beta_mean", "beta_lower", "l2_identity", "l1_identity")


def type_priors(cache=True):
    if cache and os.path.exists(PRIORS_PATH): return {k: tuple(v) for k, v in json.load(open(PRIORS_PATH)).items()}
    cal = {}
    for s in SOURCE_CAL_SEEDS:
        d, sp, caps = seed_data(s); a = SYSTEMS["no_elimination"](); a.learn_captions(caps); a.warm_up_sources(sp["train"])
        cal[s] = emissions(a, sp["train"])
    pri = hierarchical_priors(cal)
    if cache:
        os.makedirs("results", exist_ok=True); json.dump(pri, open(PRIORS_PATH, "w"))
    return pri


def build(name, caps, splits, posterior, priors, train_emits):
    from core.temporal_independent import install_temporal, learn_change_rate
    from core.unified_agent import UnifiedAgent, UnifiedConfig
    if name in ("l2_identity", "l1_identity", "l1_mean", "l1_temporal"):
        agent = UnifiedAgent(UnifiedConfig(identity_mode="l2" if name == "l2_identity" else "l1"))
    else:
        agent = SYSTEMS["no_elimination"]()
    agent.epistemic = {"calibrated_lexicon": True, "entitled": name != "em"}
    cal = prepare(agent, caps, splits, posterior)
    model = None
    if name not in ("em", "em_entitled"):
        mode = "mean" if name in ("beta_mean", "l1_mean", "temporal_mean", "l1_temporal") else "lower"
        model = fit_supervised(train_emits, mode, priors, selection=False)
        install(agent, model)
    if name in ("temporal_mean", "l1_temporal"):
        install_temporal(agent, learn_change_rate(splits["train"]))       # temporal spatial beliefs
    return agent, cal, model


def run_seed(seed, systems=SYSTEMS_S, posterior=None, priors=None, n_worlds=20):
    posterior = posterior or train_lexical_posterior(); priors = priors or type_priors()
    data, splits, caps = seed_data(seed, n_worlds)
    probe = SYSTEMS["no_elimination"](); probe.learn_captions(caps); probe.warm_up_sources(splits["train"])
    train_emits = emissions(probe, splits["train"]); test_emits = emissions(probe, splits["test"])
    em_rel = {s: probe.reliability.estimate(s) for s in true_reliabilities()}
    beta = fit_supervised(train_emits, "mean", priors, selection=False)
    beta_rel = {s: beta.estimate(s) for s in true_reliabilities()}
    labels = {"true": relabel_dataset(splits["test"], true_reliabilities()),
              "em": relabel_dataset(splits["test"], em_rel), "calibrated": relabel_dataset(splits["test"], beta_rel)}
    out, rows_by = {}, {}
    for name in systems:
        t0 = time.perf_counter()
        agent, cal, model = build(name, caps, splits, posterior, priors, train_emits)
        rows = evaluate(agent, splits["test"], labels["true"], cal)
        m = {f"{k}_orig": v for k, v in score(rows, "out", "orig").items()}
        m.update({f"{k}_ideal": v for k, v in score(rows, "out", "ideal").items()})
        ref_lab = labels["em"] if name.startswith("em") else labels["calibrated"]
        rows_cal = [dict(r, ideal=ref_lab[k][0]) for r, k in zip(rows, [id(it) for x in splits["test"] for it in x["items"]])]
        m["alignment_true"] = knowability_alignment(rows)["f1"]
        m["alignment_same_model"] = knowability_alignment(rows_cal)["f1"]
        m["reference_errors"] = error_attribution(rows, "ideal")[0]["REFERENCE"]
        rel = model or agent.reliability
        m["reliability"] = {t: v["ece"] for t, v in reliability_points(rel, test_emits)[0].items()}
        m["reliability_brier"] = {t: v["brier"] for t, v in reliability_points(rel, test_emits)[0].items()}
        m["seconds"] = time.perf_counter() - t0
        out[name] = m; rows_by[name] = rows
    # beliefs: a single ingestion per world (fresh agents, read-only access to the state)
    bel = {}
    for name in ("em", "beta_mean", "beta_lower", "temporal_mean"):
        if name not in systems: continue
        agent, cal, model = build(name, caps, splits, posterior, priors, train_emits)
        rep, conf = belief_points(agent, splits["test"])
        bel[name] = {"ece": {t: v["ece"] for t, v in rep.items()}, "brier": {t: v["brier"] for t, v in rep.items()}, "conflicted": conf}
    table = source_table(beta, test_emits)
    mnar = mnar_audit([(s, ok, p) for s, kt, ok, p, t in train_emits if t])
    return out, rows_by, {"beliefs": bel, "source_table": table, "mnar": mnar, "em_rel": em_rel, "beta_rel": beta_rel}


def multi_seed(seeds=range(16), systems=SYSTEMS_S, log=print):
    posterior = train_lexical_posterior(); priors = type_priors()
    per, extra, all_rows = [], [], defaultdict(list)
    for s in seeds:
        out, rows, ex = run_seed(s, systems, posterior, priors)
        per.append(out); extra.append(ex)
        for n in systems: all_rows[n] += rows[n]
        log(s, {n: (round(out[n]["success_orig"], 4), round(out[n]["false_confident_orig"], 4)) for n in systems})
    keys = ["success_orig", "false_confident_orig", "abstention_orig", "coverage_orig", "success_ideal", "false_confident_ideal",
            "alignment_true", "alignment_same_model", "reference_errors"]
    summary = {n: {k: summarize_values([o[n][k] for o in per]) for k in keys} for n in systems}
    comps = {}
    for a, b in (("l1_mean", "beta_mean"), ("temporal_mean", "beta_mean"), ("l1_temporal", "beta_mean"), ("l1_temporal", "l1_mean"),
                 ("beta_lower", "em_entitled"), ("beta_mean", "em_entitled"), ("beta_lower", "beta_mean"), ("em_entitled", "em"),
                 ("l1_identity", "beta_lower"), ("l2_identity", "beta_lower"), ("l1_identity", "l2_identity"), ("beta_lower", "em")):
        if a in systems and b in systems:
            comps[f"{a}_vs_{b}"] = {k: paired([o[a][k] for o in per], [o[b][k] for o in per]) for k in keys}
    rel = {n: {t: summarize_values([o[n]["reliability"].get(t) for o in per]) for t in ("all", "sensor", "document", "speaker", "database")}
           for n in systems}
    bel = {n: {t: summarize_values([e["beliefs"][n]["ece"].get(t) for e in extra if n in e["beliefs"]])
               for t in ("all", "color", "loc", "loc0", "height", "owner", "event")} for n in ("em", "beta_mean", "beta_lower", "temporal_mean") if n in systems}
    srcs = defaultdict(lambda: defaultdict(list))
    for e in extra:
        for s, v in e["source_table"].items():
            for k in ("effective", "posterior_mean", "true_parsed", "true_emitted", "error"): srcs[s][k].append(v[k])
            srcs[s]["covered"].append(float(bool(v["covered"])))
        for s, v in e["em_rel"].items(): srcs[s]["em"].append(v)
    srcs = {s: {k: round(sum(v) / len(v), 4) for k, v in d.items()} for s, d in srcs.items()}
    mnar = defaultdict(lambda: defaultdict(list))
    for e in extra:
        for s, v in e["mnar"].items():
            for k in ("accuracy_emitted", "accuracy_parsed", "selection_bias"): mnar[s][k].append(v[k])
    mnar = {s: {k: round(sum(v) / len(v), 4) for k, v in d.items()} for s, d in mnar.items()}
    conv = {}
    base_l1, base_ref = ("l1_temporal", "beta_mean") if "l1_temporal" in systems else ("l1_identity", "beta_lower")
    if base_l1 in systems and base_ref in systems:
        c = Counter()
        for a, b in zip(all_rows[base_l1], all_rows[base_ref]):
            if COMMITTED(a["raw"]) and not COMMITTED(b["raw"]):
                ok = a["raw"] == a["ideal"]
                c[("correct_" if ok else "wrong_") + ("asserted" if a["decision"] == ASSERTED else "qualified_or_withheld")] += 1
        conv = dict(c)
    return {"seeds": list(seeds), "summary": summary, "comparisons": comps, "reliability_ece": rel, "belief_ece": bel,
            "sources": srcs, "mnar": mnar, "l1_conversion": conv,
            "pareto": {n: pareto(all_rows[n])[1] for n in ("em", "beta_mean", "beta_lower", "l1_identity", "temporal_mean", "l1_temporal") if n in systems},
            "per_seed": per}


def reliability_only(seeds=range(32), priors=None):
    """RELIABILITY metrics alone (cheap) over many seeds: ECE / Brier per type, error per source,
    interval coverage; EM vs hierarchical Beta (mean)."""
    priors = priors or type_priors()
    per = {"em": defaultdict(list), "beta_mean": defaultdict(list)}
    err = defaultdict(lambda: defaultdict(list))
    for s in seeds:
        data, splits, caps = seed_data(s)
        probe = SYSTEMS["no_elimination"](); probe.learn_captions(caps); probe.warm_up_sources(splits["train"])
        tr = emissions(probe, splits["train"]); te = emissions(probe, splits["test"])
        beta = fit_supervised(tr, "mean", priors, selection=False)
        for name, m in (("em", probe.reliability), ("beta_mean", beta)):
            rep, _ = reliability_points(m, te)
            for t, v in rep.items(): per[name][t].append(v["ece"])
        for src, v in source_table(beta, te).items():
            err[src]["beta"].append(v["error"]); err[src]["covered"].append(float(bool(v["covered"])))
            err[src]["em"].append(probe.reliability.estimate(src) - v["true_parsed"])
    return {"ece": {n: {t: summarize_values(v) for t, v in d.items()} for n, d in per.items()},
            "source_error": {s: {k: summarize_values(v) for k, v in d.items()} for s, d in err.items()}, "seeds": len(list(seeds))}

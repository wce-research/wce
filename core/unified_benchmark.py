"""Expériences multi-graines de l'état unifié (environnement indépendant) : 5 systèmes, conversion du succès,
budget d'erreurs, courbes de calibration (identité, lexique, croyance, réponse finale), taille du cœur.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict

from core.grounded_benchmark import (
    COMPARISONS, SYSTEMS, _mark_resolvable, bootstrap_item_difference, metrics as g_metrics, paired, seed_data, summarize_values,
)
from core.independent_agent import ABSTAIN
from core.independent_world_ext import GOLD_MEANING, captions_ext
from core.unified import calibration_report
from core.unified_agent import belief_accuracy, identity_accuracy

FIVE = ["anchored_old", "no_elimination", "discourse_identity", "unified_belief", "unified_calibrated"]
BUDGET = ("lexical", "grammar", "reference", "source_reliability", "belief_propagation", "planner", "temporal",
          "unsupported_inference")


def error_class(row):
    """Classe d'erreur (faux confiant ou abstention inutile), à partir de la trace de la réponse."""
    a = row["answer"]
    reason = getattr(a, "abstain_reason", None)
    if row["category"] == "temporal": return "temporal"
    if reason == "lexical_unknown" or a.output == "UNKNOWN_WORD": return "lexical"
    if reason == "grammar_ambiguity" and a.output != "ASK": return "grammar"
    if reason in ("unknown_entity",) or a.output == "ASK" or "identity_p" in a.features: return "reference"
    if reason in ("low_source_reliability", "conflicting_evidence"): return "source_reliability"
    if reason == "planner_failure" or (a.intent == "command" and row["false_confident"]): return "planner"
    if reason == "unsupported_inference" or "unsupported_inference" in a.flags: return "unsupported_inference"
    if row["false_confident"] and a.entries and any(e.refuting for e in a.entries): return "source_reliability"
    return "belief_propagation"


def evaluate(agent, worlds):
    rows = []
    start = len(getattr(agent, "parse_log", []))
    for x in worlds:
        agent.new_world(); agent.ingest(x["observations"])
        if hasattr(agent, "parse_log"): _mark_resolvable(agent, x["world"], len(agent.parse_log) - sum(1 for o in x["observations"] if o.text))
        for it in x["items"]:
            a = agent.answer(it.text)
            asserted = a.output not in ABSTAIN
            conf = max((e.posterior for e in a.entries), default=None)
            rows.append({"category": it.category, "gold": it.gold, "output": a.output, "correct": a.output == it.gold,
                         "false_confident": asserted and a.output != it.gold, "abstained": not asserted,
                         "unnecessary": (not asserted) and it.gold not in ABSTAIN, "asked": a.output == "ASK",
                         "checks": a.checks, "corrected": a.corrected, "initial": getattr(a, "initial_output", a.output),
                         "confidence": conf, "answer": a, "sense_ok": None, "routing_ok": True, "reason": getattr(a, "abstain_reason", None)})
    return rows, getattr(agent, "parse_log", [])[start:]


def lexical_points(agent):
    pts = []
    for w, h in getattr(agent, "lexmem", None).hyps.items() if hasattr(agent, "lexmem") else []:
        if h.best is None or w not in GOLD_MEANING: continue
        pts.append((h.confidence, h.best == GOLD_MEANING[w]))
    return pts


def run_seed(seed, systems=FIVE, n_worlds=20):
    data, splits, caps = seed_data(seed, n_worlds)
    held = captions_ext(300, seed=seed + 900)
    out, rows_by = {}, {}
    for name in systems:
        t0 = time.perf_counter()
        agent = SYSTEMS[name](); agent.seed = seed
        agent.learn_captions(caps); agent.warm_up_sources(splits["train"])
        if hasattr(agent, "calibrate"): agent.calibrate(splits["calibration"], splits["validation"])
        rows, parse = evaluate(agent, splits["test"])
        m = g_metrics(rows, parse, agent, held)
        m["clarification"] = sum(r["asked"] for r in rows) / len(rows)
        ident = identity_accuracy(agent, splits["test"]) if hasattr(agent, "identity_points") else []
        beliefs = belief_accuracy(agent, splits["test"])
        m["identity_accuracy"] = sum(y for _, y in ident) / len(ident) if ident else None
        m["identity_commitments"] = len(ident)
        m["belief_accuracy"] = sum(y for _, y in beliefs) / len(beliefs) if beliefs else None
        m["seconds"] = time.perf_counter() - t0
        m["neural_parameters"] = m.get("parameters", 0)
        errs = Counter(error_class(r) for r in rows if r["false_confident"] or r["unnecessary"])
        m["error_budget"] = {k: errs.get(k, 0) / len(rows) for k in BUDGET}
        m["calibration_points"] = {
            "identity": ident, "belief": beliefs, "lexical": lexical_points(agent),
            "answer": [(r["confidence"], r["correct"]) for r in rows if r["confidence"] is not None and not r["abstained"]]}
        if hasattr(agent, "calibration"):
            m["calibration_choice"] = {k: v for k, v in agent.calibration.items() if k in ("observation_threshold", "query_threshold")}
            m["rollbacks"] = agent.rollbacks
        out[name] = m
        rows_by[name] = [{k: v for k, v in r.items() if k != "answer"} for r in rows]
    return out, rows_by, splits, (data, caps)


def success_conversion(rows_new, rows_base, parse_new=None, parse_base=None):
    """Références supplémentaires résolues -> combien de bonnes réponses en plus, combien de faux confiants en plus."""
    gained = sum(1 for a, b in zip(rows_new, rows_base) if a["correct"] and not b["correct"])
    lost = sum(1 for a, b in zip(rows_new, rows_base) if b["correct"] and not a["correct"])
    new_fc = sum(1 for a, b in zip(rows_new, rows_base) if a["false_confident"] and not b["false_confident"])
    fixed_fc = sum(1 for a, b in zip(rows_new, rows_base) if b["false_confident"] and not a["false_confident"])
    return {"new_correct": gained, "lost_correct": lost, "new_false_confident": new_fc, "removed_false_confident": fixed_fc,
            "net_correct": gained - lost, "net_false_confident": new_fc - fixed_fc}


def multi_seed(seeds=range(8), systems=FIVE, n_worlds=20, log=print):
    per_seed, conv, curves = [], defaultdict(list), defaultdict(lambda: defaultdict(list))
    boot = {}
    for s in seeds:
        out, rows, _, _ = run_seed(s, systems, n_worlds)
        per_seed.append(out)
        for name in systems:
            for kind, pts in out[name].pop("calibration_points").items(): curves[name][kind] += pts
            if name != "no_elimination" and "no_elimination" in rows:
                c = success_conversion(rows[name], rows["no_elimination"])
                c["extra_parse_coverage"] = out[name]["parse_coverage"] - out["no_elimination"]["parse_coverage"]
                conv[name].append(c)
        if "unified_calibrated" in rows and "no_elimination" in rows:
            boot[s] = bootstrap_item_difference(rows["unified_calibrated"], rows["no_elimination"], "correct")
        log(s, {n: {k: round(v, 4) for k, v in out[n].items() if k in ("success", "false_confident", "parse_coverage", "abstention")} for n in systems})
    keys = ("success", "false_confident", "abstention", "unnecessary_abstention", "clarification", "parse_coverage",
            "parse_coverage_resolvable", "lexical_coverage", "identity_accuracy", "belief_accuracy", "verification_cost",
            "correction_rate", "seconds")
    summary = {n: {k: summarize_values([o[n].get(k) for o in per_seed]) for k in keys} for n in systems}
    budget = {n: {k: summarize_values([o[n]["error_budget"][k] for o in per_seed]) for k in BUDGET} for n in systems}
    comps = {}
    for label, (a, b) in (("unified_vs_no_elimination", ("unified_belief", "no_elimination")),
                          ("calibrated_vs_no_elimination", ("unified_calibrated", "no_elimination")),
                          ("calibrated_vs_discourse", ("unified_calibrated", "discourse_identity")),
                          ("calibrated_vs_unified", ("unified_calibrated", "unified_belief")),
                          ("calibrated_vs_anchored_old", ("unified_calibrated", "anchored_old"))):
        if a in systems and b in systems:
            comps[label] = {k: paired([o[a].get(k) for o in per_seed], [o[b].get(k) for o in per_seed])
                            for k in ("success", "false_confident", "parse_coverage", "abstention")}
    conversion = {n: {k: sum(c[k] for c in v) for k in v[0]} for n, v in conv.items() if v}
    calib = {n: {kind: calibration_report([c for c, _ in pts], [y for _, y in pts]) for kind, pts in d.items()} for n, d in curves.items()}
    choices = [o["unified_calibrated"].get("calibration_choice") for o in per_seed if "unified_calibrated" in o]
    return {"seeds": list(seeds), "summary": summary, "comparisons": comps, "error_budget": budget,
            "success_conversion": conversion, "calibration": calib, "threshold_choices": choices, "bootstrap": boot,
            "per_seed": per_seed}

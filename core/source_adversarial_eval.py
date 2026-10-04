"""Évaluation des modèles de fiabilité sur le banc adversarial des sources (adaptateur : connaît les deux côtés)."""
from __future__ import annotations

from collections import Counter, defaultdict

from core.independent_agent import BELIEF, CONFLICTED, FACT, Evidence, EvidenceGraph, SourceReliability
from core.source_adversarial import BEHAVIORS, COLORS, PLACES, dev_episodes, sealed_episodes
from core.source_reliability import SourceReliabilityModel, source_type
from core.unified import calibration_report

MODELS = ("em", "beta_mean", "beta_lower", "beta_full")


def _type(name): return BEHAVIORS[name]["type"]


def fit(model_name, ep):
    if model_name == "em":
        rel = SourceReliability()
        groups = defaultdict(dict)
        for c in ep["history"] + ep["claims"]: groups[c["key"]][c["source"]] = c["value"]
        rel.fit_em([(len(COLORS) if k.startswith(("h", "k", "c")) else 4, v) for k, v in groups.items()])
        rel.conflict_detection = True
        return rel
    m = SourceReliabilityModel(mode="mean" if model_name == "beta_mean" else "lower", level=0.8,
                               discount=0.99 if model_name == "beta_full" else 1.0, domain=model_name == "beta_full")
    for c in ep["history"]:
        r = m.record(c["source"]); r.source_type = _type(c["source"])
        m.observe(c["source"], c["value"] == c["truth"], c["kind"], t=c["t"])
    m.fit_empirical_bayes()
    if model_name == "beta_full":
        by_key = defaultdict(dict); truth = {}
        for c in ep["history"]: by_key[c["key"]][c["source"]] = c["value"]; truth[c["key"]] = c["truth"]
        m.detect_dependencies([(v, truth[k]) for k, v in by_key.items()])
        for s in list(m.records): m.drift(s)
    return m


def answer(model, ep):
    by_key = defaultdict(list)
    for c in ep["claims"]: by_key[c["key"]].append(c)
    kinds = {q["key"]: q["kind"] for q in ep["questions"]}
    out = {}
    for key, cs in by_key.items():
        g = EvidenceGraph(model)
        kind = kinds[key]
        for c in cs: g.add((kind, key), Evidence(c["source"], c["value"], c["t"], "text"))
        b = g.belief((kind, key))
        out[key] = (b.value if b.status in (FACT, BELIEF) else "conflit" if b.status == CONFLICTED else "?", b.posterior, b.value)
    return out


def evaluate(model_name, episodes):
    rows, rel_pts, src = [], [], defaultdict(lambda: defaultdict(list))
    drift_flags, dep = Counter(), Counter()
    for ep in episodes:
        m = fit(model_name, ep)
        ans = answer(m, ep)
        for q in ep["questions"]:
            got, p, v = ans[q["key"]]
            committed = got not in ("?", "conflit")
            rows.append({"expected": q["expected"], "got": got, "committed": committed, "p": p, "value_ok": v == q["truth"],
                         "correct": got == q["expected"] or (not committed and q["expected"] in ("?", "conflit")),
                         "exact": got == q["expected"], "false_confident": committed and got != q["expected"]})
        truth = {q["key"]: q["truth"] for q in ep["questions"]}
        for c in ep["claims"]:
            if hasattr(m, "posterior"):                               # calibration : moyenne a posteriori (la borne basse pondère)
                a_, b_ = m.posterior(c["source"], c["kind"]); est = a_ / (a_ + b_)
            else: est = m.estimate(c["source"])
            rel_pts.append((est, c["value"] == truth[c["key"]]))
        emp = defaultdict(Counter)
        for c in ep["claims"]: emp[c["source"]]["n"] += 1; emp[c["source"]]["ok"] += c["value"] == truth[c["key"]]
        for s, e in emp.items():
            eff = m.estimate(s)
            src[s]["effective"].append(eff); src[s]["true_test"].append(e["ok"] / e["n"])
            if hasattr(m, "record"):
                lo, hi = m.record(s).interval(0.9); src[s]["covered"].append(float(lo <= e["ok"] / e["n"] <= hi))
                src[s]["mean"].append(m.record(s).mean)
        if hasattr(m, "records"):
            for s, r in m.records.items(): drift_flags[(s, r.calibration_state == "drifting")] += 1
            for s in m.records:
                if m.group_of(s) != s or any(m.group_of(o) == s and o != s for o in m.records): dep[s] += 1
            if model_name == "beta_full":
                src["domain_split"]["color_r"].append(m.estimate_for("domain_split", "color"))
                src["domain_split"]["loc_r"].append(m.estimate_for("domain_split", "loc"))
    n = len(rows)
    met = {"n": n, "success": round(sum(r["correct"] for r in rows) / n, 4),
           "false_confident": round(sum(r["false_confident"] for r in rows) / n, 4),
           "abstention": round(sum(not r["committed"] for r in rows) / n, 4),
           "exact_class": round(sum(r["exact"] for r in rows) / n, 4),
           "conflict_recall": round(sum(r["got"] == "conflit" for r in rows if r["expected"] == "conflit") /
                                    max(1, sum(r["expected"] == "conflit" for r in rows)), 4)}
    tp = sum(r["got"] == "conflit" and r["expected"] == "conflit" for r in rows)
    fp = sum(r["got"] == "conflit" and r["expected"] != "conflit" for r in rows)
    fn = sum(r["got"] != "conflit" and r["expected"] == "conflit" for r in rows)
    p_, r_ = tp / max(1, tp + fp), tp / max(1, tp + fn)
    met["conflict"] = {"precision": round(p_, 3), "recall": round(r_, 3), "f1": round(2 * p_ * r_ / max(1e-9, p_ + r_), 3), "tp": tp, "fp": fp, "fn": fn}
    committed = [r for r in rows if r["committed"]]
    met["belief_calibration"] = calibration_report([r["p"] for r in committed], [r["value_ok"] for r in committed])
    met["reliability_calibration"] = calibration_report([p for p, _ in rel_pts], [y for _, y in rel_pts])
    met["sources"] = {s: {k: round(sum(v) / len(v), 3) for k, v in d.items()} for s, d in src.items()}
    met["drift_flagged"] = {s: flagged for (s, flagged), c in drift_flags.items() if flagged}
    met["dependency_grouped"] = dict(dep)
    return met


def heldout_episodes():
    """Épisodes NEUFS (graines jamais vues) pour mesurer les conflits après correction (le lot scellé a déjà servi)."""
    from core.source_adversarial import episode
    return [episode(95000 + i) for i in range(6)]


def compare(episodes=None):
    episodes = episodes or dev_episodes()
    return {m: evaluate(m, episodes) for m in MODELS}

"""Banc épistémique : attendus d'origine ET attendus de l'IdealAgent, droit d'affirmer, lexique calibré, audit,
attribution des erreurs, conversion du succès, alignement de la connaissabilité, front de Pareto coût / sûreté.
"""
from __future__ import annotations

import math
import os
import statistics
import time
from collections import Counter, defaultdict

import torch

from core.entitlement import ASKED, ASSERTED, QUALIFIED, WITHHELD, ClaimEntitlement, EntitlementPolicy, provenance
from core.grounded_benchmark import SYSTEMS, T975, paired, seed_data, summarize_values
from core.independent_agent import ABSTAIN
from core.independent_world_ext import GOLD_MEANING
from core.knowledge_boundary import L1, L2, relabel_dataset, true_reliabilities
from core.lexical_calibration import (
    CalibratedLexicon, LexicalPosterior, attach, calibration_points, examples_from_agent, precision_at,
)
from core.unified import calibration_report

CAL_SEEDS = tuple(range(100, 112))          # graines de CALIBRATION du lexique (jamais les graines de test)
POSTERIOR_PATH = os.path.join("results", "lexical_posterior.pt")
COMMITTED = lambda o: o not in ABSTAIN


# ================================================================ posterior lexical (ajusté hors test)

def lexical_examples(seed, system="no_elimination", snapshots=False):
    """Hypothèses lexicales avec leur sens correct. snapshots=True : un instantané après CHAQUE monde ingéré
    (couvre le régime « peu de preuves », sinon absent de la calibration)."""
    data, splits, caps = seed_data(seed)
    a = SYSTEMS[system](); a.learn_captions(caps); a.warm_up_sources(splits["train"])
    ex = []
    for x in splits["test"]:
        a.new_world(); a.ingest(x["observations"])
        if snapshots: ex += examples_from_agent(a, GOLD_MEANING)
    if not snapshots: ex = examples_from_agent(a, GOLD_MEANING)
    return ex, {w: (h.confidence, h.best) for w, h in a.lexmem.hyps.items()}


def train_lexical_posterior(seeds=CAL_SEEDS, cache=True):
    if cache and os.path.exists(POSTERIOR_PATH):
        m = LexicalPosterior(); m.load_state_dict(torch.load(POSTERIOR_PATH)); m.fitted = True; return m
    ex = [e for s in seeds for e in lexical_examples(s, snapshots=True)[0] if e.gold is not None]
    m = LexicalPosterior().fit(ex)
    if cache:
        os.makedirs("results", exist_ok=True); torch.save(m.state_dict(), POSTERIOR_PATH)
    return m


# ================================================================ systèmes

def make_system(name, posterior=None):
    from core.unified_agent import UnifiedAgent, UnifiedConfig
    base = "unified" if name.startswith("unified") else "no_elimination"
    agent = UnifiedAgent(UnifiedConfig()) if base == "unified" else SYSTEMS["no_elimination"]()
    agent.epistemic = {"calibrated_lexicon": "callex" in name or "entitled" in name, "entitled": "entitled" in name}
    return agent


SYSTEMS_E = ("no_elimination", "callex", "entitled", "unified", "unified_entitled")


def prepare(agent, caps, splits, posterior):
    agent.learn_captions(caps); agent.warm_up_sources(splits["train"])
    cal = None
    if agent.epistemic["calibrated_lexicon"]:
        cal = CalibratedLexicon(posterior); attach(agent, cal)
    return cal


def lexical_p(agent, cal):
    if cal is None: return None
    return lambda w, m: cal.meaning(agent, w)[1]


# ================================================================ évaluation

def evaluate(agent, worlds, labels, cal=None, policy=None):
    ce = ClaimEntitlement(policy)
    rows = []
    lp = lexical_p(agent, cal)
    for x in worlds:
        agent.new_world(); agent.ingest(x["observations"])
        for it in x["items"]:
            a = agent.answer(it.text)
            prov = provenance(agent, a, lp)
            d = ce.decide(a, prov)
            final = ce.output(a, d) if agent.epistemic["entitled"] else a.output
            subj = it.meta.get("subject")
            ident = None
            if subj is not None and a.entries:
                ident = any(len(e.key) > 1 and e.key[1] == subj for e in a.entries)
            rows.append({"category": it.category, "intent": it.intent, "orig": it.gold, "ideal": labels[id(it)][0],
                         "relabeled": labels[id(it)][1], "raw": a.output, "out": final, "decision": d,
                         "level": prov.level, "q": ce.claim_probability(prov), "p_belief": prov.belief_probability,
                         "lin_unc": prov.linguistic_uncertainty, "epi_unc": prov.epistemic_uncertainty,
                         "id_unc": prov.identity_uncertainty, "diversity": prov.source_diversity,
                         "reason": getattr(a, "abstain_reason", None), "flags": list(a.flags),
                         "identity_ok": ident, "lexical": dict(prov.lexical_assumptions), "rules": list(prov.inference_rules)})
    return rows


def score(rows, out="out", gold="orig"):
    n = max(1, len(rows))
    return {"success": sum(r[out] == r[gold] for r in rows) / n,
            "false_confident": sum(COMMITTED(r[out]) and r[out] != r[gold] for r in rows) / n,
            "abstention": sum(not COMMITTED(r[out]) for r in rows) / n,
            "coverage": sum(COMMITTED(r[out]) and r[out] == r[gold] for r in rows) / max(1, sum(COMMITTED(r[gold]) for r in rows))}


def knowability_alignment(rows):
    """L'agent dit « j'ai le droit d'affirmer » vs l'agent idéal dit « connaissable »."""
    tp = sum(r["decision"] == ASSERTED and COMMITTED(r["ideal"]) for r in rows)
    fp = sum(r["decision"] == ASSERTED and not COMMITTED(r["ideal"]) for r in rows)
    fn = sum(r["decision"] != ASSERTED and COMMITTED(r["ideal"]) for r in rows)
    p = tp / max(1, tp + fp); rcl = tp / max(1, tp + fn)
    cond = {}
    for d in (ASSERTED, QUALIFIED, ASKED, WITHHELD):
        sub = [r for r in rows if r["decision"] == d]
        cond[d] = {"n": len(sub), "raw_correct_vs_ideal": round(sum(r["raw"] == r["ideal"] for r in sub) / max(1, len(sub)), 4)}
    return {"precision": p, "recall": rcl, "f1": 2 * p * rcl / max(1e-9, p + rcl), "by_entitlement": cond}


# ================================================================ audit de connaissabilité

AUDIT = ("oracle_too_strict", "logical_elimination", "hidden_information", "real_ambiguity", "bad_propagation",
         "bad_lexical_hypothesis", "temporal_error")


def audit(rows):
    """Réponses engagées avec identité correcte mais attendu d'origine non engagé (ou faux) : pourquoi ?"""
    c = Counter()
    for r in rows:
        if not COMMITTED(r["raw"]) or r["raw"] == r["orig"]: continue
        if r["category"] == "temporal" and r["raw"] != r["ideal"]: c["temporal_error"] += 1; continue
        if r["orig"] == "ASK" or r["ideal"] == "ASK": c["real_ambiguity"] += 1; continue
        if r["raw"] == r["ideal"]:
            c["logical_elimination" if any("Eliminate" in x for x in r["rules"]) else "oracle_too_strict"] += 1; continue
        if r["lexical"] and min(r["lexical"].values()) < 0.95: c["bad_lexical_hypothesis"] += 1; continue
        if r["identity_ok"] and not COMMITTED(r["ideal"]):
            c["hidden_information" if r["level"] >= L2 else "bad_propagation"] += 1; continue
        c["bad_propagation"] += 1
    return {k: c.get(k, 0) for k in AUDIT}


# ================================================================ attribution des erreurs

CAUSES = ("LEXICAL", "REFERENCE", "EVIDENCE", "SOURCE", "TEMPORAL", "GRAMMAR", "PLANNER", "KNOWABILITY_REFERENCE",
          "UNSUPPORTED_INFERENCE")


def attribute(r, gold="ideal"):
    """(cause principale, causes secondaires) d'une erreur finale (faux confiant ou abstention inutile)."""
    causes = []
    if r["out"] == r["ideal"] and r["out"] != r["orig"] and gold == "orig": causes.append("KNOWABILITY_REFERENCE")
    if r["category"] == "temporal": causes.append("TEMPORAL")
    if r["raw"] == "UNKNOWN_WORD" or r["reason"] == "lexical_unknown" or (r["lexical"] and min(r["lexical"].values()) < 0.95):
        causes.append("LEXICAL")
    if "grammar_ambiguity" in r["flags"]: causes.append("GRAMMAR")
    if r["raw"] == "ASK" or r["reason"] == "unknown_entity" or r["id_unc"] > 0.05 or r["level"] >= L2: causes.append("REFERENCE")
    if r["reason"] in ("low_source_reliability", "conflicting_evidence"): causes.append("SOURCE")
    if r["reason"] == "planner_failure" or "planner_failure" in r["flags"]: causes.append("PLANNER")
    if "unsupported_inference" in r["flags"]: causes.append("UNSUPPORTED_INFERENCE")
    if not causes: causes.append("EVIDENCE")
    return causes[0], causes[1:]


def error_attribution(rows, gold="ideal"):
    prim, sec = Counter(), Counter()
    for r in rows:
        wrong = (COMMITTED(r["out"]) and r["out"] != r[gold]) or (not COMMITTED(r["out"]) and COMMITTED(r[gold]))
        if not wrong: continue
        p, s = attribute(r, gold); prim[p] += 1
        for x in s: sec[x] += 1
    n = max(1, len(rows))
    return {k: round(prim.get(k, 0) / n, 4) for k in CAUSES}, {k: round(sec.get(k, 0) / n, 4) for k in CAUSES}


# ================================================================ conversion du succès (revisitée)

def conversion(rows_new, rows_base):
    """Références supplémentaires résolues (réponse brute engagée ici, pas dans la base) -> devenir."""
    c = Counter()
    for a, b in zip(rows_new, rows_base):
        if not (COMMITTED(a["raw"]) and not COMMITTED(b["raw"])): continue
        ok = a["raw"] == a["ideal"]
        if a["decision"] == ASSERTED: c["correct_asserted" if ok else "wrong_asserted"] += 1
        elif a["decision"] == QUALIFIED: c["correct_qualified" if ok else "wrong_qualified"] += 1
        else: c["correctly_withheld" if not ok else "withheld_but_correct"] += 1
    return dict(c)


# ================================================================ front de Pareto coût / sûreté

def pareto(rows, gold="orig", wrong=(2, 5, 10, 20, 50, 100), abstain=(0.5, 1, 2), allow_l2=(False, True)):
    """Rejoue la décision d'affirmer pour chaque matrice de coût (vectorisé) : succès / faux confiants / abstention."""
    q = torch.tensor([r["q"] for r in rows]); lvl = torch.tensor([r["level"] for r in rows])
    raw_commit = torch.tensor([COMMITTED(r["raw"]) for r in rows])
    raw_ok = torch.tensor([r["raw"] == r[gold] for r in rows])
    gold_unknown = torch.tensor([r[gold] == "UNKNOWN" for r in rows])           # une retenue (UNKNOWN) est juste ici
    pts = []
    for W in wrong:
        for A in abstain:
            for l2 in allow_l2:
                t = 1 - A / W
                assert_ = raw_commit & (q >= t) & ((lvl <= L1) | torch.tensor(l2))
                succ = torch.where(assert_, raw_ok, torch.where(raw_commit, gold_unknown, raw_ok))
                fc = assert_ & ~raw_ok
                pts.append({"wrong": W, "abstain": A, "allow_l2": l2, "threshold": round(t, 3),
                            "success": round(float(succ.float().mean()), 4), "false_confident": round(float(fc.float().mean()), 4),
                            "abstention": round(float((~assert_).float().mean()), 4)})
    front = [p for p in pts if not any(o["success"] >= p["success"] and o["false_confident"] <= p["false_confident"]
                                       and (o["success"] > p["success"] or o["false_confident"] < p["false_confident"]) for o in pts)]
    return pts, sorted(front, key=lambda p: p["false_confident"])


def lexical_coverage(agent, cal=None):
    """Formes jamais légendées apprises avec le BON sens (lexique calibré si attaché, sinon statut natif)."""
    from core.grounded_benchmark import LEARNABLE
    ok = wrong = 0
    for w in LEARNABLE:
        m = cal.meaning(agent, w)[0] if cal is not None else agent.lexmem.meaning(w)
        if m is None: continue
        if m == GOLD_MEANING.get(w): ok += 1
        else: wrong += 1
    return ok / len(LEARNABLE), wrong


# ================================================================ multi-graines

def run_seed(seed, systems=SYSTEMS_E, posterior=None, n_worlds=20):
    posterior = posterior or train_lexical_posterior()
    data, splits, caps = seed_data(seed, n_worlds)
    labels = relabel_dataset(splits["test"], true_reliabilities())
    out, rows_by = {}, {}
    for name in systems:
        t0 = time.perf_counter()
        agent = make_system(name); agent.seed = seed
        cal = prepare(agent, caps, splits, posterior)
        rows = evaluate(agent, splits["test"], labels, cal)
        m = {f"{k}_orig": v for k, v in score(rows, "out", "orig").items()}
        m.update({f"{k}_ideal": v for k, v in score(rows, "out", "ideal").items()})
        m["alignment"] = knowability_alignment(rows)
        m["audit"] = audit(rows)
        m["attribution_ideal"], m["attribution_secondary"] = error_attribution(rows, "ideal")
        m["attribution_orig"], _ = error_attribution(rows, "orig")
        lex_ex = examples_from_agent(agent, GOLD_MEANING)
        m["lexical_points"] = calibration_points(lex_ex, posterior) if cal else calibration_points(lex_ex, None, {w: (h.confidence, h.best) for w, h in agent.lexmem.hyps.items()})
        m["lexical_coverage"], m["lexical_wrong"] = lexical_coverage(agent, cal)
        ids = [r["identity_ok"] for r in rows if r["identity_ok"] is not None and COMMITTED(r["raw"])]
        m["object_identity_accuracy"] = sum(ids) / len(ids) if ids else None
        m["seconds"] = time.perf_counter() - t0
        out[name] = m; rows_by[name] = rows
    for name in systems:
        if name != "no_elimination" and "no_elimination" in rows_by: out[name]["conversion"] = conversion(rows_by[name], rows_by["no_elimination"])
    return out, rows_by, labels


def multi_seed(seeds=range(8), systems=SYSTEMS_E, log=print):
    posterior = train_lexical_posterior()
    per, all_rows = [], defaultdict(list)
    for s in seeds:
        out, rows, _ = run_seed(s, systems, posterior)
        per.append(out)
        for n in systems: all_rows[n] += rows[n]
        log(s, {n: (round(out[n]["success_orig"], 4), round(out[n]["false_confident_orig"], 4),
                    round(out[n]["success_ideal"], 4), round(out[n]["false_confident_ideal"], 4)) for n in systems})
    keys = [f"{k}_{g}" for g in ("orig", "ideal") for k in ("success", "false_confident", "abstention", "coverage")] + \
           ["lexical_coverage", "lexical_wrong", "object_identity_accuracy"]
    summary = {n: {k: summarize_values([o[n][k] for o in per]) for k in keys} for n in systems}
    comps = {}
    for a, b in (("entitled", "no_elimination"), ("callex", "no_elimination"), ("unified_entitled", "no_elimination"),
                 ("unified_entitled", "entitled"), ("unified", "no_elimination")):
        if a in systems and b in systems:
            comps[f"{a}_vs_{b}"] = {k: paired([o[a][k] for o in per], [o[b][k] for o in per]) for k in keys[:8]}
    agg = lambda n, f: {k: sum(o[n][f].get(k, 0) for o in per) for k in per[0][n][f]}
    return {"seeds": list(seeds), "summary": summary, "comparisons": comps,
            "alignment": {n: {k: summarize_values([o[n]["alignment"][k] for o in per]) for k in ("precision", "recall", "f1")} for n in systems},
            "alignment_by_entitlement": {n: per[0][n]["alignment"]["by_entitlement"] for n in systems},
            "audit": {n: agg(n, "audit") for n in systems},
            "attribution_ideal": {n: {k: round(sum(o[n]["attribution_ideal"][k] for o in per) / len(per), 4) for k in per[0][n]["attribution_ideal"]} for n in systems},
            "attribution_orig": {n: {k: round(sum(o[n]["attribution_orig"][k] for o in per) / len(per), 4) for k in per[0][n]["attribution_orig"]} for n in systems},
            "conversion": {n: {k: sum(o[n].get("conversion", {}).get(k, 0) for o in per) for k in
                               ("correct_asserted", "correct_qualified", "correctly_withheld", "withheld_but_correct", "wrong_asserted", "wrong_qualified")}
                           for n in systems if n != "no_elimination"},
            "lexical_calibration": {n: calibration_report([c for o in per for c, _ in o[n]["lexical_points"]],
                                                          [y for o in per for _, y in o[n]["lexical_points"]]) for n in systems},
            "pareto": {n: pareto(all_rows[n])[1] for n in ("no_elimination", "callex", "unified") if n in systems},
            "pareto_points": {n: pareto(all_rows[n])[0] for n in ("no_elimination", "callex", "unified") if n in systems},
            "curves": curves(all_rows), "per_seed": [{n: {k: v for k, v in o[n].items() if k != "lexical_points"} for n in systems} for o in per]}


def curves(all_rows):
    """Courbes confiance -> exactitude : croyance, droit d'affirmer, réponse finale (par système)."""
    out = {}
    for n, rows in all_rows.items():
        # réponses À VALEUR fondées sur une croyance (les rejets structurels et les commandes n'ont pas de postérieure)
        committed = [r for r in rows if COMMITTED(r["raw"]) and r["intent"] != "command" and r["p_belief"] > 0
                     and r["raw"] not in ("REJECT", "INVALID_PREMISE", "UNKNOWN_CAUSE", "NONE")]
        out[n] = {"belief": calibration_report([r["p_belief"] for r in committed], [r["raw"] == r["ideal"] for r in committed]),
                  "entitlement": calibration_report([r["q"] for r in committed], [r["raw"] == r["ideal"] for r in committed]),
                  "final": calibration_report([r["q"] for r in committed if COMMITTED(r["out"])], [r["out"] == r["ideal"] for r in committed if COMMITTED(r["out"])])}
    return out

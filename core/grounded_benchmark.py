"""Benchmark multi-graines de la compréhension ancrée : moyennes, écarts-types, IC95, comparaisons appariées.

Chaque graine change : les mondes, la langue des observations, le bruit, les légendes, la partition,
et l'initialisation neuronale. Le test d'une graine n'est jamais utilisé pour régler quoi que ce soit.
"""
from __future__ import annotations

import math
import random
import statistics
import time
from collections import Counter, defaultdict
from typing import Dict, List

from core.grounded import (
    LEX_BELIEF, LEX_FACT, GroundedAgent, GroundedConfig, LexicalMemory,
)
from core.independent_agent import ABSTAIN, AgentConfig, IndependentAgent, tokenize
from core.independent_benchmark import grammar_accuracy
from core.independent_world_ext import (
    EXT_HELD_OUT, GOLD_MEANING, build_dataset_ext, captions_ext, split_worlds,
)

SYSTEMS = {
    "previous": lambda: IndependentAgent(AgentConfig(neural=False, geometry=False, correction=True, evidence_graph=True,
                                                     verification="exhaustive")),
    "grounded_full": lambda: GroundedAgent(GroundedConfig()),
    "no_evidence_graph": lambda: GroundedAgent(GroundedConfig(evidence_graph=False)),
    "local_grammar": lambda: GroundedAgent(GroundedConfig(joint_grammar=False)),
    "static_lexicon": lambda: GroundedAgent(GroundedConfig(grounded_lexicon=False)),
    "distilled": lambda: GroundedAgent(GroundedConfig(distilled_roles=True)),
    "no_correction": lambda: GroundedAgent(GroundedConfig(correction=False)),
    "no_elimination": lambda: GroundedAgent(GroundedConfig(elimination=False)),
    "anchored_old": lambda: GroundedAgent(GroundedConfig(elimination=True)),
    "static_lexicon_anchored": lambda: GroundedAgent(GroundedConfig(grounded_lexicon=False, elimination=True)),
    "discourse_identity": lambda: _discourse_agent(),
    "unified_belief": lambda: _unified_agent(calibrate=False),
    "unified_calibrated": lambda: _unified_agent(calibrate=True),
}


def _unified_agent(**kw):
    from core.unified_agent import UnifiedAgent, UnifiedConfig
    return UnifiedAgent(UnifiedConfig(**kw))


def _discourse_agent():
    from core.discourse_agent import DiscourseAgent
    return DiscourseAgent()
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
LEARNABLE = sorted(EXT_HELD_OUT - {"mets"})


# ================================================================ une graine

def seed_data(seed, n_worlds=20, n_captions=500):
    data = build_dataset_ext(n_worlds, world_seed=seed, lang_seed=seed + 100, obs_seed=seed)
    return data, split_worlds(data, seed), captions_ext(n_captions, seed=seed + 1)


def evaluate_agent(agent, test_worlds):
    rows = []
    start = len(getattr(agent, "parse_log", []))
    for x in test_worlds:
        agent.new_world(); agent.ingest(x["observations"])
        if hasattr(agent, "parse_log"): _mark_resolvable(agent, x["world"], len(agent.parse_log) - sum(1 for o in x["observations"] if o.text))
        for it in x["items"]:
            a = agent.answer(it.text)
            asserted = a.output not in ABSTAIN
            sense_ok = None
            if it.meta.get("sense"):
                chosen = a.word_meanings.get("range")
                sense_ok = chosen == it.meta["sense"]
            rows.append({"category": it.category, "domain": it.meta.get("domain"), "clauses": it.meta.get("clauses", 1),
                         "gold": it.gold, "output": a.output, "initial": getattr(a, "initial_output", a.output),
                         "correct": a.output == it.gold, "false_confident": asserted and a.output != it.gold,
                         "abstained": not asserted, "unnecessary": (not asserted) and it.gold not in ABSTAIN,
                         "reason": getattr(a, "abstain_reason", None), "corrected": a.corrected, "checks": a.checks,
                         "routing_ok": a.intent == it.intent or (it.intent == "command" and a.intent == "claim" and False),
                         "sense_ok": sense_ok})
    parse = getattr(agent, "parse_log", [])[start:]
    return rows, parse


def _mark_resolvable(agent, world, start):
    """Évaluation seulement : la référence principale était-elle résoluble avec les croyances de l'agent ?
    (type unique, ou couleurs des autres objets du même type établies et différentes)."""
    for p in agent.parse_log[start:]:
        oid = p["key"][1] if p["key"][0] != "event" else p["value"].patient
        kind = agent.belief.inventory.get(oid)
        same = [o for o, k in agent.belief.inventory.items() if k == kind]
        ok = len(same) <= 1
        if not ok:
            mine = world.objects[oid].color
            ok = all(agent.belief.known(agent.belief.color(o)) and agent.belief.color(o).value != mine for o in same if o != oid)
        p["resolvable"] = ok


def parse_coverage_resolvable(parse_log):
    sub = [p for p in parse_log if p.get("resolvable")]
    return parse_coverage(sub) if sub else 0.0, len(sub) / max(1, len(parse_log))


def parse_coverage(parse_log):
    """Proposition principale correctement extraite (évaluation : comparée à ce que la source affirmait)."""
    if not parse_log: return 0.0
    good = 0
    for p in parse_log:
        k = p["key"] if p["key"][0] != "loc" else ("loc", p["key"][1], 0 if p["key"][2] == 0 else "now")
        v = p["value"]
        if k[0] == "event": k, v = ("event", v.patient), (v.rel, v.target)
        good += bool(p["parsed"] and p["props"] and any(kk == k and vv == v for kk, vv in p["props"]))
    return good / len(parse_log)


def lexical_coverage(agent):
    """Formes jamais légendées, apprises depuis les observations (BELIEF ou mieux) avec le BON sens."""
    if not hasattr(agent, "lexmem"): return 0.0, 0
    correct = wrong = 0
    for w in LEARNABLE:
        m = agent.lexmem.meaning(w)
        if m is None: continue
        if m == GOLD_MEANING.get(w): correct += 1
        else: wrong += 1
    return correct / len(LEARNABLE), wrong


def metrics(rows, parse, agent, held):
    n = len(rows)
    poly = [r["sense_ok"] for r in rows if r["sense_ok"] is not None]
    lex_cov, lex_wrong = lexical_coverage(agent)
    corr = [r for r in rows if r["corrected"]]
    return {"success": sum(r["correct"] for r in rows) / n,
            "false_confident": sum(r["false_confident"] for r in rows) / n,
            "abstention": sum(r["abstained"] for r in rows) / n,
            "unnecessary_abstention": sum(r["unnecessary"] for r in rows) / n,
            "correct_abstention": sum(r["abstained"] and not r["unnecessary"] for r in rows) / n,
            "parse_coverage": parse_coverage(parse),
            "parse_coverage_resolvable": parse_coverage_resolvable(parse)[0],
            "resolvable_fraction": parse_coverage_resolvable(parse)[1],
            "role_accuracy": grammar_accuracy(agent, held)["roles"],
            "lexical_coverage": lex_cov, "lexical_wrong": lex_wrong,
            "verification_cost": sum(r["checks"] for r in rows) / n,
            "routing_accuracy": sum(r["routing_ok"] for r in rows) / n,
            "correction_rate": len(corr) / n,
            "correction_to_gold": sum(r["correct"] for r in corr) / max(1, len(corr)) if corr else None,
            "harmful_corrections": sum(1 for r in corr if r["initial"] == r["gold"] and not r["correct"]),
            "polysemy_accuracy": sum(poly) / len(poly) if poly else None,
            "parameters": getattr(getattr(agent.grammar, "student", None), "parameter_count", 0) if hasattr(agent, "grammar") else 0,
            "n_items": n}


def run_seed(seed, systems=None, n_worlds=20, n_captions=500):
    data, splits, caps = seed_data(seed, n_worlds, n_captions)
    held = captions_ext(300, seed=seed + 900)
    out, raw = {}, {}
    for name in systems or SYSTEMS:
        agent = SYSTEMS[name]()
        agent.seed = seed
        agent.learn_captions(caps); agent.warm_up_sources(splits["train"])
        if hasattr(agent, "calibrate"): agent.calibrate(splits["calibration"], splits["validation"])      # jamais le test
        rows, parse = evaluate_agent(agent, splits["test"])
        out[name] = metrics(rows, parse, agent, held)
        raw[name] = {"rows": rows, "parse": parse, "agent": agent}
    return out, raw


# ================================================================ statistiques

def summarize_values(values):
    values = [v for v in values if v is not None]
    if not values: return None
    n = len(values); mean = sum(values) / n
    sd = statistics.stdev(values) if n > 1 else 0.0
    half = T975.get(n - 1, 1.96) * sd / math.sqrt(n) if n > 1 else 0.0
    return {"mean": round(mean, 4), "sd": round(sd, 4), "ci95": (round(mean - half, 4), round(mean + half, 4)), "n": n}


def paired(values_a, values_b):
    """Différence appariée par graine (a - b) : moyenne, IC95 (t), nombre de graines où a > b."""
    diffs = [a - b for a, b in zip(values_a, values_b) if a is not None and b is not None]
    s = summarize_values(diffs)
    if s: s["seeds_positive"] = sum(d > 0 for d in diffs); s["seeds_negative"] = sum(d < 0 for d in diffs)
    return s


def bootstrap_item_difference(rows_a, rows_b, field, iterations=1000, seed=0):
    """Différence de proportions au niveau des items (mêmes items, systèmes différents), IC95 bootstrap."""
    rng = random.Random(seed)
    pairs = [(float(a[field]), float(b[field])) for a, b in zip(rows_a, rows_b)]
    n = len(pairs)
    diffs = []
    for _ in range(iterations):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append(sum(a - b for a, b in sample) / n)
    diffs.sort()
    return {"diff": round(sum(a - b for a, b in pairs) / n, 4), "ci95": (round(diffs[int(0.025 * iterations)], 4), round(diffs[int(0.975 * iterations) - 1], 4)), "n_items": n}


COMPARISONS = {
    "evidence_graph_vs_none": ("grounded_full", "no_evidence_graph"),
    "joint_vs_local_grammar": ("grounded_full", "local_grammar"),
    "grounded_vs_static_lexicon": ("grounded_full", "static_lexicon"),
    "grounded_full_vs_previous": ("grounded_full", "previous"),
    "distilled_vs_teacher": ("distilled", "grounded_full"),
    "correction_vs_none": ("grounded_full", "no_correction"),
    "elimination_vs_none": ("anchored_old", "no_elimination"),
    "grounded_vs_static_lexicon_anchored": ("anchored_old", "static_lexicon_anchored"),
    "discourse_vs_no_elimination": ("discourse_identity", "no_elimination"),
    "discourse_vs_anchored_old": ("discourse_identity", "anchored_old"),
}
KEY_METRICS = ("success", "false_confident", "abstention", "unnecessary_abstention", "parse_coverage",
               "parse_coverage_resolvable", "resolvable_fraction", "role_accuracy",
               "lexical_coverage", "verification_cost", "routing_accuracy", "correction_rate", "polysemy_accuracy")


def multi_seed(seeds=(0, 1, 2, 3, 4, 5, 6, 7), systems=None, n_worlds=20):
    per_seed, raws = {}, {}
    t0 = time.time()
    for s in seeds: per_seed[s], raws[s] = run_seed(s, systems, n_worlds)
    names = list(next(iter(per_seed.values())))
    summary = {name: {m: summarize_values([per_seed[s][name][m] for s in seeds]) for m in KEY_METRICS} for name in names}
    comparisons = {}
    for label, (a, b) in COMPARISONS.items():
        if a not in names or b not in names: continue
        comparisons[label] = {m: paired([per_seed[s][a][m] for s in seeds], [per_seed[s][b][m] for s in seeds])
                              for m in ("success", "false_confident", "unnecessary_abstention", "parse_coverage", "role_accuracy",
                                        "lexical_coverage", "verification_cost")}
        rows_a = [r for s in seeds for r in raws[s][a]["rows"]]; rows_b = [r for s in seeds for r in raws[s][b]["rows"]]
        comparisons[label]["item_bootstrap_success"] = bootstrap_item_difference(rows_a, rows_b, "correct")
        comparisons[label]["item_bootstrap_false_confident"] = bootstrap_item_difference(rows_a, rows_b, "false_confident")
    return {"seeds": list(seeds), "summary": summary, "comparisons": comparisons, "per_seed": per_seed, "raws": raws,
            "seconds": round(time.time() - t0, 1)}


# ================================================================ analyses sur les sorties multi-graines

def abstention_breakdown(raws, system="grounded_full"):
    table = defaultdict(Counter)
    for s, r in raws.items():
        for row in r[system]["rows"]:
            if row["abstained"]: table[row["reason"] or "unclassified"]["unnecessary" if row["unnecessary"] else "correct"] += 1
    return {k: dict(v) for k, v in sorted(table.items(), key=lambda kv: -sum(kv[1].values()))}


def by_group(raws, system, key):
    groups = defaultdict(list)
    for r in raws.values():
        for row in r[system]["rows"]: groups[row[key]].append(row)
    return {g: {"n": len(v), "success": round(sum(x["correct"] for x in v) / len(v), 3),
                "false_confident": round(sum(x["false_confident"] for x in v) / len(v), 3)} for g, v in sorted(groups.items(), key=lambda t: str(t[0]))}


def correction_report(raws, system="grounded_full"):
    stats = Counter()
    for r in raws.values():
        for row in r[system]["rows"]:
            if not row["corrected"]: continue
            stats["attempted"] += 1
            if row["correct"]: stats["now_correct"] += 1
            if row["initial"] == row["gold"] and not row["correct"]: stats["harmful"] += 1
            if row["initial"] != row["gold"] and row["initial"] not in ABSTAIN and row["abstained"]: stats["wrong_to_abstain"] += 1
            if row["initial"] in ABSTAIN and not row["abstained"] and not row["correct"]: stats["unknown_to_wrong"] += 1
    needing = sum(1 for r in raws.values() for row in r[system]["rows"] if row["initial"] != row["gold"])
    stats["needing"] = needing
    stats["precision"] = round(stats["now_correct"] / stats["attempted"], 3) if stats["attempted"] else None
    stats["recall"] = round(stats["now_correct"] / needing, 3) if needing else None
    stats["harmful_rate"] = round(stats["harmful"] / stats["attempted"], 3) if stats["attempted"] else 0.0
    return dict(stats)


def learning_needs(raws, system="grounded_full", top=8):
    needs = Counter()
    for r in raws.values():
        for n in r[system]["agent"].learning_needs: needs[(n.kind, n.target if n.kind == "lexical" else n.reason)] += 1
    return needs.most_common(top)


def lexical_lifecycle(agent, words=None):
    out = {}
    for w in words or LEARNABLE + ["contenait"]:
        h = agent.lexmem.get(w)
        if h: out[w] = {"status": h.status, "best": h.best, "confidence": round(h.confidence, 3), "history": h.history,
                        "sources": {c: sorted(s) for c, s in h.sources.items() if c == h.best},
                        "experiences": len(h.experiences.get(h.best, ())) if h.best else 0,
                        "gold": GOLD_MEANING.get(w)}
    return out


# ================================================================ courbes

def coverage_growth(seed=0, sizes=(50, 100, 200, 400, 800), n_worlds=20):
    data, splits, _ = seed_data(seed, n_worlds)
    held = captions_ext(300, seed=seed + 900)
    out = {}
    for n in sizes:
        agent = GroundedAgent(GroundedConfig()); agent.learn_captions(captions_ext(n, seed=seed + 1))
        agent.warm_up_sources(splits["train"])
        rows, parse = evaluate_agent(agent, splits["test"])
        m = metrics(rows, parse, agent, held)
        out[n] = {k: round(m[k], 3) for k in ("parse_coverage", "lexical_coverage", "abstention", "unnecessary_abstention",
                                               "success", "false_confident", "role_accuracy")}
    return out


def budget_curve(seed=0, budgets=(1, 2, 20, 99), n_worlds=20):
    """Budget de vérification (hypothèses par requête) vs succès vs faux confiants vs coût."""
    data, splits, caps = seed_data(seed, n_worlds)
    held = captions_ext(200, seed=seed + 900)
    out = {}
    for b in budgets:
        for distilled in (False, True):
            agent = GroundedAgent(GroundedConfig(budget=b, distilled_roles=distilled))
            agent.learn_captions(caps); agent.warm_up_sources(splits["train"])
            rows, parse = evaluate_agent(agent, splits["test"])
            m = metrics(rows, parse, agent, held)
            out[f"budget={b}{' +distilled' if distilled else ''}"] = {k: round(m[k], 3) for k in ("success", "false_confident", "verification_cost", "abstention")}
    return out


def structural_distillation(seed=0, n_worlds=20):
    """Hypothèses de rôles explorées : perceptron exhaustif (enseignant) vs élève distillé + vérification WCE."""
    data, splits, caps = seed_data(seed, n_worlds)
    held = captions_ext(300, seed=seed + 900)
    out = {}
    for label, distilled in (("teacher_exhaustive", False), ("student_then_verify", True)):
        agent = GroundedAgent(GroundedConfig(distilled_roles=distilled)); agent.learn_captions(caps)
        agent.grammar.explored = 0
        acc = grammar_accuracy(agent, held)
        explored = agent.grammar.explored
        nps = sum(1 for c in held if c["structure"]["kind"] != "sequence")
        out[label] = {"role_accuracy": acc["roles"], "hypotheses_explored_per_sentence": round(explored / max(1, nps), 2),
                      "student_parameters": agent.grammar.student.parameter_count if agent.grammar.student else 0}
    # désambiguïsation des sens (élève = table de contexte apprise ; enseignant = choix structurel)
    return out


def long_sentence_report(raws, system="grounded_full"): return by_group(raws, system, "clauses")


def domain_report(raws, system="grounded_full"): return by_group(raws, system, "domain")


def domain_transfer(seed=0):
    """La même construction (« X est dans Y », « range X dans Y ») transfère-t-elle à l'inventaire ?
    Rôles sur des légendes d'inventaire, grammaire apprise SANS aucune légende d'inventaire vs avec."""
    held = [c for c in captions_ext(400, seed=seed + 950) if c["structure"].get("target", {}).get("type") == "list"]
    from core.independent_world import captions as physical_captions
    physical_only = GroundedAgent(GroundedConfig()).learn_captions(physical_captions(500, seed=seed + 1))
    mixed = GroundedAgent(GroundedConfig()).learn_captions(captions_ext(500, seed=seed + 1))
    def role_acc_on(agent):
        # les mots d'inventaire sont inconnus du premier agent : on ne mesure que la structure (positions des rôles)
        ok = n = 0
        for c in held:
            toks = tokenize(c["text"]); s = c["structure"]
            nps = agent.grammar.chunks(toks)
            if len(nps) < 2:
                # sans lexique d'inventaire, on reconstruit les syntagmes par position (déterminant + nom)
                continue
            roles = agent.grammar.assign(toks, nps, s["kind"] if s["kind"] != "event" else "command")
            want = ["target", "subject"] if " contient " in f" {c['text']} " else ["subject", "target"]
            if s["kind"] == "event": want = ["patient", "target"]
            n += 1; ok += [roles.get(i) for i in range(2)] == want
        return round(ok / max(1, n), 3), n
    physical_only.grammar.lex = mixed.lex       # lexique ancré fourni ; les POIDS de rôles restent appris sur le physique
    return {"inventory_captions": len(held), "grammar_trained_on_physical_only": role_acc_on(physical_only),
            "grammar_trained_on_mixed": role_acc_on(mixed),
            "note": "grammaire physique + lexique d'inventaire ancré : mesure du transfert des poids de rôles seuls"}


# ================================================================ échelle lexicale

def lexical_scale(sizes=(100, 250, 500, 1000, 2000), exposures=(2, 5, 10, 20), seed=0):
    """Acquisition trans-situationnelle de synonymes artificiels : chaque exposition présente le vrai sens
    parmi des distracteurs ; source de fiabilité variable. Taux d'acquisition (BELIEF+) et erreurs."""
    from core.independent_benchmark import lexicon_growth
    rng = random.Random(seed)
    out = {}
    for n in sizes:
        meanings = [f"m{i}" for i in range(max(10, n // 5))]
        row = {}
        for k in exposures:
            mem = LexicalMemory()
            words = [f"w{i}" for i in range(n)]
            truth = {w: rng.choice(meanings) for w in words}
            for w in words:
                for e in range(k):
                    distract = rng.sample(meanings, 2)
                    cands = set(distract) | {truth[w]}
                    weight = rng.choice([0.9, 0.7, 0.1])       # capteur fiable / document / locuteur peu fiable
                    results = {c: "support" for c in cands}
                    mem.observe(w, results, f"src{rng.randint(0, 3)}", weight, f"e{e}")
            learned = {w: mem.meaning(w) for w in words}
            acq = sum(1 for w in words if learned[w] == truth[w]) / n
            wrong = sum(1 for w in words if learned[w] not in (None, truth[w])) / n
            row[k] = {"acquired": round(acq, 3), "wrong": round(wrong, 3)}
        out[n] = {"by_exposures": row, "structural_memory_entries": n}
    growth = lexicon_growth(sizes=tuple(s for s in sizes if s <= 2000), seed=seed)
    return {"cross_situational": out, "neural_vs_structural": growth}


# ================================================================ géométrie tenue à l'écart

def geometry_holdout(seeds=(0, 1, 2, 3, 4, 5, 6, 7), n_worlds=20):
    """Observables enregistrés à chaque graine ; analyse sur l'échantillon poolé. N'influence aucune décision."""
    from core.independent_benchmark import Setup, calibrate_geometry, oov_words, word_cases
    from core.independent_agent import IndependentAgent, AgentConfig
    from core.trajectory import auroc
    rows = []
    for s in seeds:
        data, splits, caps = seed_data(s, n_worlds)
        setup = Setup(data, caps, splits)
        agent = IndependentAgent(AgentConfig(geometry=True, correction=False, verification="selective"), s)
        agent.learn_captions(caps); agent.warm_up_sources(splits["train"])
        monitor, est = calibrate_geometry(agent, setup)
        words = oov_words([it for x in splits["test"] for it in x["items"]], agent)
        for r in word_cases(agent, monitor, words):
            r["risk"] = est.score(r["z"]); r["seed"] = s; rows.append(r)
    lab = [not r["correct"] for r in rows]
    conf = [1 - r["z"]["confidence"] for r in rows]; geo = [r["risk"] for r in rows]
    rng = random.Random(0); n = len(rows); deltas = []
    for _ in range(500):
        idx = [rng.randrange(n) for _ in range(n)]
        a = auroc([geo[i] for i in idx], [lab[i] for i in idx]); b = auroc([conf[i] for i in idx], [lab[i] for i in idx])
        if a is not None and b is not None: deltas.append(a - b)
    deltas.sort()
    return {"n_words": n, "n_errors": sum(lab), "auroc_confidence": auroc(conf, lab), "auroc_geometry": auroc(geo, lab),
            "delta_geometry_minus_confidence": {"mean": round(sum(deltas) / len(deltas), 3) if deltas else None,
                                                 "ci95": (round(deltas[12], 3), round(deltas[-13], 3)) if len(deltas) > 25 else None},
            "used_in_decisions": False}

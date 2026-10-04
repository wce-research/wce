"""Benchmark de référence : résolution d'entités à travers le discours, métriques, apprentissage actif, distillation."""
from __future__ import annotations

import math
import random
import time
from collections import Counter, defaultdict
from typing import Dict, List

import torch

from core.discourse import AMBIGUOUS, HYPOTHETICAL, NONEXISTENT, DiscourseResolver
from core.grounded import GroundedAgent, GroundedConfig, LexicalMemory
from core.independent_agent import tokenize
from core.independent_world_ext import captions_ext
from core.reference_world import (
    ATTR_VALUES, CATEGORIES, ClarificationSimulator, DOMAINS, family_captions, long_episode, reference_benchmark,
)

SPECIAL = {AMBIGUOUS, NONEXISTENT, HYPOTHETICAL}
_LEXICON_CACHE = {}


def lexicon_agent(seed=0):
    """Lexique appris (légendes physiques, inventaire, famille) — aucune connaissance des épisodes."""
    if seed not in _LEXICON_CACHE:
        a = GroundedAgent(GroundedConfig()); a.learn_captions(captions_ext(500, seed=seed + 1) + family_captions(150, seed=seed))
        _LEXICON_CACHE[seed] = a
    return _LEXICON_CACHE[seed]


def _align(turn_mentions, gold):
    """Aligne les mentions de l'agent sur les mentions de référence (ordre + mot tête)."""
    out, used = [], set()
    for g in gold:
        m = next((x for x in turn_mentions if x.id not in used and (x.head == g["head"] or x.head.rstrip("s") == g["head"])), None)
        if m: used.add(m.id)
        out.append(m)
    return out


def run_episode(episode, lex, clarify=True, active_entity=False, student=None):
    res = DiscourseResolver(lex, episode.given)
    sim = ClarificationSimulator(episode)
    rows = []
    for ti, turn in enumerate(episode.turns):
        mentions = res.process(turn.text)
        aligned = _align(mentions, turn.mentions)
        for g, mm in zip(turn.mentions, aligned):
            at_time = res.mention_resolution(mm.id) if mm else AMBIGUOUS
            rows.append({"gold": g["gold"], "intended": g.get("intended"), "mention": mm.id if mm else None, "at_time": at_time, "turn": ti,
                         "clarified": False, "candidates": len(mm.candidates) if mm else 0})
            if mm and at_time == AMBIGUOUS and turn.kind == "command" and turn.clarify_target and clarify:
                q = res.clarification_question(mm.id)
                if q:
                    res.apply_clarification(mm.id, sim.answer(q, turn.clarify_target))
                    rows[-1]["clarified"] = True; rows[-1]["gold"] = turn.clarify_target
            if mm and active_entity and res.mention_resolution(mm.id) == AMBIGUOUS and mm.referent:
                for _ in range(3):
                    req, gain = res.best_information_request(mm.referent, ATTR_VALUES[DOMAINS[episode.domain]["attr"]])
                    if req is None: break
                    value = sim.reveal(req)
                    if value is None: break
                    res.add_fact(req, value); res.solve(res.state.time, reason=f"information {req} (IG {gain})")
                    if res.mention_resolution(mm.id) != AMBIGUOUS: break
    for r in rows:
        r["final"] = res.mention_resolution(r["mention"]) if r["mention"] else AMBIGUOUS
        r["category"], r["domain"] = episode.category, episode.domain
        if active_entity and r["gold"] == AMBIGUOUS and r["intended"] and r["final"] not in SPECIAL:
            r["gold"] = r["intended"]           # l'information obtenue a rendu l'entité visée identifiable : on juge contre elle
    return rows, res, sim


def reference_metrics(rows):
    n = len(rows)
    resolvable = [r for r in rows if r["gold"] not in SPECIAL]
    committed = [r for r in rows if r["final"] not in SPECIAL]
    correct = [r for r in committed if r["final"] == r["gold"]]
    false_commit = [r for r in committed if r["final"] != r["gold"]]
    amb = [r for r in rows if r["gold"] == AMBIGUOUS]
    special_ok = [r for r in rows if r["gold"] in (NONEXISTENT, HYPOTHETICAL) and r["final"] == r["gold"]]
    recovered = [r for r in rows if r["at_time"] in SPECIAL and r["gold"] not in SPECIAL and r["final"] == r["gold"]]
    return {"mentions": n,
            "accuracy": round(sum(r["final"] == r["gold"] for r in rows) / n, 3),
            "precision": round(len(correct) / max(1, len(committed)), 3),
            "recall": round(len(correct) / max(1, len(resolvable)), 3),
            "coverage": round(len(committed) / n, 3),
            "false_identity_commitment": round(len(false_commit) / n, 3),
            "ambiguity_preserved": round(sum(r["final"] == AMBIGUOUS for r in amb) / max(1, len(amb)), 3),
            "nonexistent_hypothetical_correct": len(special_ok),
            "clarification_rate": round(sum(r["clarified"] for r in rows) / n, 3),
            "avg_candidates": round(sum(r["candidates"] for r in rows) / n, 2),
            "cross_sentence_recovery": len(recovered)}


def run_benchmark(n_per_category=14, seed=0, clarify=True, active_entity=False, noise=0.0):
    lex = lexicon_agent(seed)
    episodes = reference_benchmark(n_per_category, seed, noise=noise)
    rows, revisions, checks, questions = [], 0, 0, 0
    violations = 0
    for ep in episodes:
        r, res, sim = run_episode(ep, lex, clarify, active_entity)
        rows += r; revisions += len(res.revisions); checks += res.checks; questions += sim.questions
        violations += len(res.graph.invariant_violations(res.state, res.resolution))
    m = reference_metrics(rows)
    m.update({"episodes": len(episodes), "belief_revisions": revisions, "verification_checks": checks,
              "questions_asked": questions, "invariant_violations": violations})
    return m, rows


def by_key(rows, key):
    groups = defaultdict(list)
    for r in rows: groups[r[key]].append(r)
    return {k: reference_metrics(v) for k, v in sorted(groups.items())}


def long_discourse(lengths=(5, 10, 20, 50), n=12, seed=0):
    lex = lexicon_agent(seed)
    out = {}
    for L in lengths:
        rows, mem, t0, revs = [], [], time.time(), 0
        for i in range(n):
            ep = long_episode(L, "physical", seed=seed * 1000 + i)
            r, res, _ = run_episode(ep, lex)
            rows += r; mem.append(res.memory_size()); revs += len(res.revisions)
        m = reference_metrics(rows)
        out[L] = {"accuracy": m["accuracy"], "false_identity_commitment": m["false_identity_commitment"], "coverage": m["coverage"],
                  "raw_tokens": round(sum(x["raw_tokens"] for x in mem) / n, 1),
                  "consolidated_fields": round(sum(x["consolidated_fields"] for x in mem) / n, 1),
                  "seconds_per_sentence": round((time.time() - t0) / (n * L), 4), "revisions_per_episode": round(revs / n, 2)}
    return out


# ================================================================ distillation du résolveur

FEATURES = ("type_match", "attr_match", "attr_unknown", "relation_ok", "recency", "focus", "gender_match", "definite", "indefinite", "other")


def _features(res, mm, e):
    f = res.world[e]
    r = res.state.entities_mentioned.get(mm.referent) if mm.referent else None
    attr = (r.attrs.get("attr") if r else None) or mm.attrs.get("attr")
    return [float(r is not None and r.type == f.get("type")), float(attr is not None and f.get("attr") == attr),
            float(f.get("attr") is None), float(r is not None and res._compatible(r, e)), 1.0 / (1 + res.state.time - mm.sentence),
            float(r is not None and r.id == res.state.focus), float(mm.gender is not None), float(mm.definiteness == "def"),
            float(mm.definiteness == "indef"), float(mm.kind == "other")]


def distill_resolver(seed=0, n_per_category=10):
    """Enseignant : résolveur WCE (contraintes jointes). Élève : petit classement de candidats.
    L'élève propose le meilleur candidat ; WCE vérifie ses contraintes (1 contrôle) ; en cas d'échec, l'enseignant reprend."""
    lex = lexicon_agent(seed)
    train_eps = reference_benchmark(n_per_category, seed + 50)
    test_eps = reference_benchmark(n_per_category, seed + 60)
    X, Y = [], []
    for ep in train_eps:
        _, res, _ = run_episode(ep, lex)
        for mm in res.state.mentions.values():
            if mm.kind == "place" or not mm.referent: continue
            teacher = res.resolution.get(mm.referent)
            if teacher in SPECIAL or teacher is None: continue
            for e in res.entities_of(None):
                X.append(_features(res, mm, e)); Y.append(float(e == teacher))
    X = torch.tensor(X); Y = torch.tensor(Y)
    net = torch.nn.Sequential(torch.nn.Linear(len(FEATURES), 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.02)
    for _ in range(300):
        opt.zero_grad(); torch.nn.functional.binary_cross_entropy_with_logits(net(X).squeeze(1), Y).backward(); opt.step()
    student_ok = teacher_checks = student_checks = total = fallback = 0
    for ep in test_eps:
        _, res, _ = run_episode(ep, lex)
        for mm in res.state.mentions.values():
            if mm.kind == "place" or not mm.referent: continue
            teacher = res.resolution.get(mm.referent)
            if teacher in SPECIAL or teacher is None: continue
            ents = res.entities_of(None)
            with torch.no_grad(): scores = net(torch.tensor([_features(res, mm, e) for e in ents])).squeeze(1)
            proposal = ents[int(scores.argmax())]
            total += 1
            r = res.state.entities_mentioned[mm.referent]
            teacher_checks += len(ents) * max(1, len(res.dists))
            student_checks += 1
            if not res._compatible(r, proposal): fallback += 1; student_checks += len(ents); proposal = teacher
            student_ok += proposal == teacher
    return {"decisions": total, "student_agreement_with_teacher": round(student_ok / max(1, total), 3),
            "fallback_to_teacher": fallback, "checks_teacher": teacher_checks, "checks_student_then_verify": student_checks,
            "student_parameters": sum(p.numel() for p in net.parameters())}


# ================================================================ apprentissage actif lexical

def _verdicts(agent, toks, ob, word):
    """Verdicts de l'ancrage (sens -> soutenu/réfuté/neutre) pour une observation, sans mise à jour de la mémoire."""
    i = toks.index(word)
    from core.independent_agent import token_class as tc
    cands = sorted({m for m in agent.lex.lexicon().values() if tc(m) in ("N", "C", "R")})
    out = {}
    for sym in cands:
        agent._forced[word] = sym
        try: out[sym] = agent._check_against_evidence(toks, i, sym, ob.source)
        finally: agent._forced.pop(word, None)
    return out, max(0.0, 2 * agent.reliability.estimate(ob.source) - 1)


def observation_pool(seed=0, n_worlds=30, words=("brique", "caisse", "boule", "rayon", "plancher", "bouquin")):
    """Observations textuelles contenant les mots cibles, avec leurs verdicts pré-calculés (monde ingéré sans ces textes)."""
    from core.grounded_benchmark import seed_data
    data, splits, caps = seed_data(seed, n_worlds)
    agent = GroundedAgent(GroundedConfig()); agent.learn_captions(caps); agent.warm_up_sources(splits["train"][:4])
    pool = []
    for wi, x in enumerate(data):
        agent.new_world()
        agent.ingest([o for o in x["observations"] if o.text is None])
        for o in x["observations"]:
            if not o.text: continue
            toks = tokenize(o.text)
            for w in words:
                if w in toks:
                    v, weight = _verdicts(agent, toks, o, w)
                    pool.append({"word": w, "world": wi, "source": o.source, "verdicts": v, "weight": weight, "text": o.text,
                                 "types": {f"type:{k}" for k in agent.belief.inventory.values()}})      # public avant observation
    return pool, agent


def _entropy_after(h_scores, verdicts, weight):
    scores = dict(h_scores)
    supports = [c for c, v in verdicts.items() if v == "support"]
    for c in supports: scores[c] = scores.get(c, 0.0) + weight / len(supports)
    for c, v in verdicts.items():
        if v == "refute": scores[c] = scores.get(c, 0.0) - weight
    return _entropy(scores)


def _entropy(scores):
    if not scores: return math.log(20)
    vals = torch.tensor(list(scores.values())) * 2.0
    p = torch.softmax(vals, 0)
    return float(-(p * p.clamp_min(1e-12).log()).sum())


def active_vs_passive(seed=0, budget=12, pool=None):
    """Expositions nécessaires pour ancrer chaque mot : flux passif (ordre d'arrivée) vs sélection active (gain d'information)."""
    from core.independent_world_ext import GOLD_MEANING
    pool, agent = pool or observation_pool(seed)
    out = {}
    for w in sorted({p["word"] for p in pool}):
        items = [p for p in pool if p["word"] == w]
        res = {}
        for mode in ("passive", "active"):
            mem = LexicalMemory(); remaining = list(items); used = 0; learned_at = None
            while remaining and used < budget:
                h = mem.get(w)
                scores = {c: h.support[c] - h.refute[c] for c in (set(h.support) | set(h.refute))} if h else {}
                if mode == "passive": item = remaining.pop(0)
                else:
                    before = _entropy(scores)
                    gains = [before - _entropy_after(scores, it["verdicts"], it["weight"]) for it in remaining]
                    item = remaining.pop(int(torch.tensor(gains).argmax()))
                mem.observe(w, item["verdicts"], item["source"], item["weight"], f"w{item['world']}:{item['text']}")
                used += 1
                if learned_at is None and mem.meaning(w) == GOLD_MEANING.get(w): learned_at = used
            res[mode] = {"exposures_to_learn": learned_at, "learned": mem.meaning(w) == GOLD_MEANING.get(w),
                         "wrong": mem.meaning(w) not in (None, GOLD_MEANING.get(w)), "pool": len(items)}
        out[w] = res
    summary = {}
    for mode in ("passive", "active"):
        learned = [v[mode] for v in out.values()]
        summary[mode] = {"words_learned": sum(x["learned"] for x in learned), "wrong": sum(x["wrong"] for x in learned),
                         "mean_exposures_when_learned": round(sum(x["exposures_to_learn"] for x in learned if x["exposures_to_learn"]) /
                                                              max(1, sum(1 for x in learned if x["exposures_to_learn"])), 2)}
    return {"per_word": out, "summary": summary, "budget": budget}


def active_entity_learning(seed=0, n_per_category=14):
    """Désambiguïsation par demandes d'information choisies au gain d'information vs aucune demande."""
    base, _ = run_benchmark(n_per_category, seed, clarify=False, active_entity=False)
    active, _ = run_benchmark(n_per_category, seed, clarify=False, active_entity=True)
    return {"passive": {k: base[k] for k in ("accuracy", "coverage", "false_identity_commitment", "questions_asked")},
            "active": {k: active[k] for k in ("accuracy", "coverage", "false_identity_commitment", "questions_asked")}}

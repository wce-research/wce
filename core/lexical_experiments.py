"""Expériences lexicales : polysémie calibrée, désambiguïsation active, politique des mots inconnus (provenance),
incertitude morphologique, similarité par comportement, classes émergentes, composition."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import torch

from core.grounded import LexicalMemory
from core.independent_agent import tokenize
from core.independent_world_ext import GOLD_MEANING
from core.lexical_calibration import (
    LEX_BELIEF, LEX_HYPOTHESIS, LEX_STABLE, NULL, CalibratedLexicon, LexicalLifecycle, PolysemyPosterior,
    SimilarityByBehavior, class_purity, compose, entropy, examples_from_agent, morphology_probability,
)
from core.unified import calibration_report


# ================================================================ polysémie

def polysemy_eval(seeds=range(8)):
    """« range » : P(sens | type du patient, type de la cible, relation, domaine) sur les items de polysémie."""
    from core.grounded_benchmark import SYSTEMS, seed_data
    pts, base_ok, n = [], 0, 0
    for seed in seeds:
        data, splits, caps = seed_data(seed)
        pp = PolysemyPosterior().fit(caps)
        agent = SYSTEMS["no_elimination"](); agent.learn_captions(caps)
        for x in splits["test"]:
            w = x["world"]
            for it in x["items"]:
                sense = it.meta.get("sense")
                if not sense: continue
                toks = tokenize(it.text)
                nps = agent.grammar.chunks(toks, [agent.word_meaning(t)[0] for t in toks])
                typ = lambda np: (np.type if np else None)
                rel = next((m.split(":")[1] for m in (agent.word_meaning(t)[0] for t in toks) if m and m.startswith("rel:")), None)
                ctx = {"patient": typ(nps[0]) if nps else None, "target": typ(nps[1]) if len(nps) > 1 else None, "rel": rel,
                       "domain": "inventory" if (len(nps) > 1 and nps[1].type == "list") else "physical"}
                post = pp.posterior(ctx)
                best = max(post, key=post.get)
                pts.append((post[best], best == sense))
                agent._ctx = toks
                senses = agent.senses("range")
                base = agent._choose_sense("range", senses) if len(senses) >= 2 else (senses[0] if senses else None)
                base_ok += base == sense; n += 1
    return {"n": n, "posterior_accuracy": round(sum(y for _, y in pts) / max(1, n), 4),
            "baseline_accuracy": round(base_ok / max(1, n), 4),
            "calibration": calibration_report([c for c, _ in pts], [y for _, y in pts])}


# ================================================================ désambiguïsation lexicale active (posterior calibré)

def active_lexical(posterior, seed=0, budget=12, pool=None):
    """Exposition choisie par gain d'information attendu sur le posterior calibré ; promotion = statut BELIEF calibré."""
    from core.discourse_benchmark import observation_pool
    pool, agent = pool or observation_pool(seed)
    cal = CalibratedLexicon(posterior)
    out = {}
    saved = agent.lexmem
    try:
        for w in sorted({p["word"] for p in pool}):
            items = [p for p in pool if p["word"] == w]
            res = {}
            for mode in ("passive", "active"):
                mem = LexicalMemory(); agent.lexmem = mem
                remaining, used, learned_at, ent = list(items), 0, None, []
                while remaining and used < budget:
                    if mode == "passive": item = remaining.pop(0)
                    else:
                        K = len({m for m in agent.lex.lexicon().values() if m})
                        d0 = cal.dist(agent, w)
                        before = entropy(d0, K)
                        gains = []
                        for it in remaining:
                            # issue PRÉDITE sans regarder l'observation : seuls les types présents dans le monde visé
                            # (inventaire public) restent compatibles ; les autres hypothèses de type seraient réfutées
                            pred = {c: ("neutral" if c in it["types"] else "refute") for c in d0 if c != NULL and c.startswith("type:")}
                            m2 = LexicalMemory(); m2.hyps = {k: _copy(h) for k, h in mem.hyps.items()}
                            m2.observe(w, pred, it["source"], it["weight"], "sim")
                            agent.lexmem = m2; gains.append(before - entropy(cal.dist(agent, w), K)); agent.lexmem = mem
                        item = remaining.pop(int(torch.tensor(gains).argmax()))
                    mem.observe(w, item["verdicts"], item["source"], item["weight"], f"w{item['world']}:{item['text']}")
                    used += 1
                    d = cal.dist(agent, w); ent.append(round(entropy(d, len({m for m in agent.lex.lexicon().values() if m})), 3))
                    st, c, p = cal.status(agent, w)
                    if learned_at is None and st in (LEX_BELIEF, LEX_STABLE) and c == GOLD_MEANING.get(w): learned_at = used
                st, c, p = cal.status(agent, w)
                res[mode] = {"exposures_to_belief": learned_at, "final_status": st, "final_meaning": c, "p": round(p, 3),
                             "wrong": st in (LEX_BELIEF, LEX_STABLE) and c != GOLD_MEANING.get(w), "entropy": ent, "pool": len(items)}
            out[w] = res
    finally:
        agent.lexmem = saved
    summary = {mode: {"learned": sum(1 for r in out.values() if r[mode]["exposures_to_belief"]),
                      "mean_exposures": _mean([r[mode]["exposures_to_belief"] for r in out.values() if r[mode]["exposures_to_belief"]]),
                      "wrong": sum(r[mode]["wrong"] for r in out.values())} for mode in ("passive", "active")}
    return {"words": out, "summary": summary}


def _copy(h):
    import copy
    return copy.deepcopy(h)


def _mean(xs): return round(sum(xs) / len(xs), 2) if xs else None


# ================================================================ politique des mots inconnus (provenance)

STRUCTURAL, MORPHOLOGICAL, ANALOGY, UNKNOWN = "structural_inference", "morphological", "analogy", "unknown"


def unknown_word_candidates(agent, word, calibrated=None, similarity=None, contexts=()):
    """Quatre voies, chacune avec sa provenance ; l'analogie n'est jamais promue en fait (au plus HYPOTHESIS)."""
    out = []
    if calibrated is not None and agent.lexmem.get(word) is not None:
        st, c, p = calibrated.status(agent, word)
        if c: out.append({"concept": c, "provenance": STRUCTURAL, "p": p, "status": st})
    for ref, meaning, rule, support in agent.morph.candidates(word, agent.lex.lexicon()):
        p = morphology_probability(agent, word, meaning)
        out.append({"concept": meaning, "provenance": MORPHOLOGICAL, "p": p,
                    "status": LEX_BELIEF if p >= 0.95 else LEX_HYPOTHESIS, "via": f"{rule.name} ({ref})"})
        break
    if similarity is not None and contexts:
        c, sim = similarity.analogy(contexts, {w: m for w, m in agent.lex.lexicon().items() if m})
        if c: out.append({"concept": c, "provenance": ANALOGY, "p": round(sim, 3), "status": LEX_HYPOTHESIS})
    if not out: out.append({"concept": None, "provenance": UNKNOWN, "p": 0.0, "status": "UNKNOWN"})
    return sorted(out, key=lambda d: (-{LEX_STABLE: 3, LEX_BELIEF: 2, LEX_HYPOTHESIS: 1}.get(d["status"], 0), -d["p"]))


def unknown_word_policy_eval(posterior, seeds=range(4)):
    from core.grounded_benchmark import LEARNABLE, SYSTEMS, seed_data
    stats = defaultdict(Counter)
    promoted_analogy = 0
    for seed in seeds:
        data, splits, caps = seed_data(seed)
        a = SYSTEMS["no_elimination"](); a.learn_captions(caps); a.warm_up_sources(splits["train"])
        for x in splits["test"]: a.new_world(); a.ingest(x["observations"])
        cal = CalibratedLexicon(posterior); sim = SimilarityByBehavior(caps)
        for w in LEARNABLE:
            ctx = []
            for x in data:
                for o in x["observations"]:
                    if o.text and w in tokenize(o.text):
                        t = tokenize(o.text); ctx.append((t, t.index(w)))
            for c in unknown_word_candidates(a, w, cal, sim, ctx[:20]):
                st = stats[c["provenance"]]; st["n"] += 1
                if c["concept"] is not None: st["correct"] += c["concept"] == GOLD_MEANING.get(w)
                if c["provenance"] == ANALOGY and c["status"] in (LEX_BELIEF, LEX_STABLE): promoted_analogy += 1
    return {k: {"n": v["n"], "precision": round(v["correct"] / max(1, v["n"]), 3)} for k, v in stats.items()} | \
        {"analogy_promoted_to_fact": promoted_analogy}


# ================================================================ incertitude morphologique

def morphology_eval(seeds=range(4)):
    """Règles de suffixe : probabilité (soutien vs contre-exemples) sur des formes réelles et des PIÈGES."""
    from core.grounded_benchmark import SYSTEMS, seed_data
    traps = ["cuber", "boulet", "tablette", "rougeur", "bleuet", "caissier"]         # ressemblent à des formes connues
    out = Counter(); probs = []
    for seed in seeds:
        data, splits, caps = seed_data(seed)
        a = SYSTEMS["no_elimination"](); a.learn_captions(caps)
        known = a.lex.lexicon()
        for form, meaning in known.items():                       # formes réelles : fléchies connues (auto-cohérence)
            for ref, m, rule, sup in a.morph.candidates(form, known):
                if ref == form: continue
                p = morphology_probability(a, form, m); probs.append((p, m == meaning)); break
        for t in traps:
            c = a.morph.candidates(t, known)
            if c:
                p = morphology_probability(a, t, c[0][1]); out["trap_candidates"] += 1; out["trap_promoted"] += p >= 0.95
            else: out["trap_unknown"] += 1
    return {**dict(out), "calibration": calibration_report([p for p, _ in probs], [y for _, y in probs])}


# ================================================================ similarité par comportement / classes / composition

def semantic_classes(seed=0, threshold=0.9):
    from core.grounded_benchmark import seed_data
    _, _, caps = seed_data(seed)
    sim = SimilarityByBehavior(caps)
    classes = sim.classes(threshold)
    meaning = {}
    for c in caps:
        for w, m in c.get("meanings", {}).items(): meaning[w] = m
    return {"n_words": len(sim.words), "n_classes": len(classes), "purity": round(class_purity(classes, meaning), 3),
            "examples": [g[:6] for g in sorted(classes, key=len, reverse=True)[:6]]}


def composition_eval(seed=0):
    from core.grounded_benchmark import SYSTEMS, seed_data
    _, _, caps = seed_data(seed)
    a = SYSTEMS["no_elimination"](); a.learn_captions(caps)
    wm = lambda t: a.word_meaning(t)[0] or (a.senses(t) or [None])[0]
    cases = {"remet": "act:place", "repose": "act:place", "fait tomber": "cause(act:fall)", "met dedans": ("act:place", "rel:in"),
             "repousse": "act:push", "pose dessus": ("act:place", "rel:on")}
    out = {}
    for e, gold in cases.items():
        m, trace = compose(e, wm, a.lex.lexicon())
        out[e] = {"meaning": m, "correct": m == gold, "trace": trace, "surface_in_lexicon": a.lex.seen(e.split()[0])}
    return out

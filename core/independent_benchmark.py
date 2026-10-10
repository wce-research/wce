"""Independent benchmark: ablations, taxonomy of hallucinations, conditional geometry,
verification costs, grammar learning, lexical growth, representations.

Rule: all tuning is done on train / calibration / validation; the test set is read only at evaluation.
"""
from __future__ import annotations

import copy
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from core.independent_agent import (
    ABSTAIN, AgentConfig, IndependentAgent, NeuralOnlyAgent, TrainCore, token_class, tokenize,
)
from core.independent_world import (
    NONSENSE, SOURCE_BY_NAME, build_dataset, captions, split_of,
)
from core.language_core import CharCore, CorruptCore, encode_words
from core.trajectory import OBSERVABLES, GeometricRiskEstimator, TrajectoryMonitor, auroc

ASSERTIVE_SPECIAL = {"REJECT", "INVALID_PREMISE", "SUPPORTED", "REFUTED", "NONE", "UNKNOWN_CAUSE"}


# ================================================================ preparation

@dataclass
class Setup:
    data: list
    captions: list
    splits: Dict[str, list]


def prepare(n_worlds=25, n_captions=400, seed=0):
    data = build_dataset(n_worlds, seed)
    splits = defaultdict(list)
    for i, x in enumerate(data): splits[split_of(i, len(data))].append(x)
    return Setup(data, captions(n_captions, seed=seed + 1), dict(splits))


VARIANTS = {
    "A_neural_only": None,
    "B_neural_plasticity": None,
    "C_wce_only": AgentConfig(neural=False, evidence_graph=False, geometry=False, correction=False, verification="exhaustive"),
    "D_neural_wce": AgentConfig(evidence_graph=False, geometry=False, correction=False, verification="selective"),
    "E_neural_wce_evidence": AgentConfig(evidence_graph=True, geometry=False, correction=False, verification="selective"),
    "F_neural_wce_geometry": AgentConfig(evidence_graph=False, geometry=True, correction=False, verification="geometry"),
    "G_full": AgentConfig(evidence_graph=True, geometry=True, correction=True, verification="geometry"),
}


def make_agent(name, setup: Setup, size="small", seed=0):
    if name == "A_neural_only": agent = NeuralOnlyAgent(seed=seed, size=size)
    elif name == "B_neural_plasticity": agent = NeuralOnlyAgent(seed=seed, size=size, plasticity=True)
    else:
        cfg = copy.deepcopy(VARIANTS[name]); cfg.size = size
        agent = IndependentAgent(cfg, seed)
    agent.learn_captions(setup.captions)
    if name == "B_neural_plasticity":       # plasticity: additional experience (new captions), without verification
        extra = captions(200, seed=seed + 50)
        agent.learn_captions(setup.captions + extra)
    agent.warm_up_sources(setup.splits["train"])
    if agent.cfg.geometry and agent.neural is not None:
        agent.attach_geometry(*calibrate_geometry(agent, setup))
    return agent


# ================================================================ geometry: word-level calibration

def oov_words(items, agent):
    """Out-of-lexicon words in queries, with their reference meaning (generator; evaluation only)."""
    out = {}
    for it in items:
        for tok in tokenize(it.text):
            if agent.lex.seen(tok) or not tok.isalpha(): continue
            gold = it.meta["meanings"].get(tok)
            if gold is None and tok in NONSENSE: gold = "NONSENSE"
            if gold is not None: out[tok] = gold
    return out


def word_cases(agent, monitor, words: Dict[str, str], core=None):
    core = core or agent.neural.core
    rows = []
    for w, gold in sorted(words.items()):
        (sym, p), = core.propose(w, 1)
        rows.append({"word": w, "gold": gold, "predicted": sym, "correct": sym == gold, "z": monitor.observe(core, w)})
    return rows


def calibrate_geometry(agent, setup: Setup):
    monitor = TrajectoryMonitor()
    lex = agent.lex.lexicon()
    monitor.fit_reference(agent.neural.core, list(lex), list(lex.values()))
    cal_items = [it for x in setup.splits["calibration"] for it in x["items"]]
    words = oov_words(cal_items, agent)
    rows = word_cases(agent, monitor, words)
    rows += [{**r, "correct": True} for r in word_cases(agent, monitor, {w: lex[w] for w in list(lex)[::2]})]
    estimator = GeometricRiskEstimator().fit([r["z"] for r in rows], [not r["correct"] for r in rows])
    return monitor, estimator


# ================================================================ evaluation

def taxonomy(item, answer):
    """Category of a wrong and asserted answer (evaluation only)."""
    out, gold = answer.output, item.gold
    kind = item.meta.get("kind")
    for tok, meaning in answer.word_meanings.items():
        ref = item.meta["meanings"].get(tok)
        if ref and meaning not in ("?", None) and meaning != ref: return "wrong_lexical_interpretation"
    if gold == "UNKNOWN_WORD": return "overconfident_unknown"
    if answer.intent not in ("?", item.intent) and not (item.intent == "command" and answer.intent == "claim"): return "wrong_routing"
    if gold == "INVALID_PREMISE": return "invented_entity" if kind == "nonexistent_entity" else "invented_relation"
    if gold in ("UNKNOWN", "ASK"):
        if item.intent == "color" or (item.intent == "claim" and "color" in str(item.meta.get("key"))): return "invented_property"
        if item.intent in ("where", "why") or "loc" in str(item.meta.get("key")): return "invented_relation"
        return "unsupported_inference"
    if item.meta.get("time") == 0 or item.category == "temporal": return "wrong_temporal_state"
    if item.intent == "height" or "height" in str(item.meta.get("key")): return "wrong_numerical_value"
    return "contradiction"


def run_split(agent, worlds, record=True):
    rows = []
    for x in worlds:
        agent.new_world(); agent.ingest(x["observations"])
        for it in x["items"]:
            a = agent.answer(it.text)
            asserted = a.output not in ABSTAIN
            rows.append({"item": it, "answer": a, "correct": a.output == it.gold, "asserted": asserted,
                         "false_confident": asserted and a.output != it.gold,
                         "abstained": not asserted, "gold_abstain": it.gold in ABSTAIN,
                         "taxonomy": taxonomy(it, a) if asserted and a.output != it.gold else None})
    return rows


def summarize(rows, agent=None):
    n = len(rows)
    by_cat = defaultdict(lambda: [0, 0])
    for r in rows: by_cat[r["item"].category][0] += r["correct"]; by_cat[r["item"].category][1] += 1
    out = {"n": n, "success": round(sum(r["correct"] for r in rows) / n, 3),
           "false_confident_rate": round(sum(r["false_confident"] for r in rows) / n, 3),
           "abstention_rate": round(sum(r["abstained"] for r in rows) / n, 3),
           "correct_abstentions": sum(r["abstained"] and r["gold_abstain"] for r in rows),
           "unnecessary_abstentions": sum(r["abstained"] and not r["gold_abstain"] for r in rows),
           "verification_cost": round(sum(r["answer"].checks for r in rows) / n, 2),
           "corrections": sum(r["answer"].corrected for r in rows),
           "by_category": {c: round(a / b, 3) for c, (a, b) in sorted(by_cat.items())},
           "taxonomy": dict(Counter(r["taxonomy"] for r in rows if r["taxonomy"]))}
    if agent is not None:
        out["parameters"] = agent.neural.core.parameter_count if agent.neural else 0
        out["structural_memory"] = len(agent.lex.lexicon()) + sum(len(v) for v in agent.grammar.counts.values())
    return out


def ablations(setup: Setup, size="small", seed=0):
    results = {}
    for name in VARIANTS:
        agent = make_agent(name, setup, size, seed)
        rows = run_split(agent, setup.splits["test"])
        results[name] = summarize(rows, agent)
    return results


# ================================================================ self-correction

def correction_benchmark(setup: Setup, seed=0):
    """Initial candidate -> corrected / still wrong / abstention; harmful corrections."""
    agent = make_agent("G_full", setup, seed=seed)
    rows = run_split(agent, setup.splits["test"])
    stats = Counter()
    for r in rows:
        a, gold = r["answer"], r["item"].gold
        initial, final = a.initial_output, a.output
        init_ok = initial == gold
        if not a.corrected:
            if not init_ok and initial not in ABSTAIN: stats["initial_wrong_uncorrected"] += 1
            continue
        stats["attempted"] += 1
        if final == gold: stats["corrected_to_gold"] += 1
        elif final in ABSTAIN: stats["abstained"] += 1
        else: stats["still_wrong"] += 1
        if (init_ok or initial in ABSTAIN) and final != gold and final not in ABSTAIN: stats["harmful"] += 1
    needing = sum(1 for r in rows if r["answer"].initial_output != r["item"].gold)
    precision = stats["corrected_to_gold"] / stats["attempted"] if stats["attempted"] else None
    recall = stats["corrected_to_gold"] / needing if needing else None
    return {**stats, "needing_correction": needing, "correction_precision": precision, "correction_recall": round(recall, 3) if recall is not None else None,
            "harmful_correction_rate": round(stats["harmful"] / max(1, stats["attempted"]), 3) if stats["attempted"] else 0.0}


# ================================================================ verification costs

def verification_costs(setup: Setup, seed=0):
    out = {}
    agent = make_agent("A_neural_only", setup, seed=seed)
    s = summarize(run_split(agent, setup.splits["test"])); s["verification_cost"] = 0.0
    out["neural_only"] = s
    for label, mode, geometry in (("exhaustive_wce", "exhaustive", False), ("distilled_selective", "selective", False),
                                  ("geometry_triggered", "geometry", True)):
        cfg = AgentConfig(evidence_graph=True, geometry=geometry, correction=False, verification=mode)
        agent = IndependentAgent(cfg, seed).learn_captions(setup.captions); agent.warm_up_sources(setup.splits["train"])
        if geometry: agent.attach_geometry(*calibrate_geometry(agent, setup))
        out[label] = summarize(run_split(agent, setup.splits["test"]), agent)
    return {k: {m: v[m] for m in ("success", "false_confident_rate", "abstention_rate", "verification_cost")} for k, v in out.items()}


# ================================================================ geometry: real test and conditional value

def auprc(scores, positives):
    pairs = sorted(zip(scores, positives), key=lambda t: -t[0])
    tp = fp = 0; total = sum(positives); prec_sum = 0.0
    if not total: return None
    for s, p in pairs:
        if p: tp += 1; prec_sum += tp / (tp + fp)
        else: fp += 1
    return round(prec_sum / total, 3)


def ece(probs, labels, bins=10):
    if not probs: return None
    total, err = len(probs), 0.0
    for b in range(bins):
        idx = [i for i, p in enumerate(probs) if b / bins <= p < (b + 1) / bins or (b == bins - 1 and p == 1.0)]
        if idx: err += abs(sum(probs[i] for i in idx) / len(idx) - sum(labels[i] for i in idx) / len(idx)) * len(idx) / total
    return round(err, 3)


def geometry_word_test(setup: Setup, seed=0, size="small"):
    """Neural lexical errors on out-of-lexicon words of the TEST set (calibration done on other worlds)."""
    agent = make_agent("G_full", setup, size, seed)
    monitor, estimator = agent.monitor, agent.estimator
    test_items = [it for x in setup.splits["test"] for it in x["items"]]
    rows = word_cases(agent, monitor, oov_words(test_items, agent))
    for r in rows: r["category"] = "nonsense" if r["gold"] == "NONSENSE" else "heldout_synonym_or_form"
    lex = agent.lex.lexicon()
    known = word_cases(agent, monitor, {w: lex[w] for w in list(lex)[1::2]})
    for r in known: r["category"] = "known"
    corrupted = copy.deepcopy(agent.neural.core); CorruptCore(scale=1.5, seed=3).apply(corrupted)
    bad = word_cases(agent, monitor, {w: lex[w] for w in list(lex)[1::2]}, core=corrupted)
    for r in bad: r["category"] = "corrupted_model"
    rows += known + bad
    labels = [not r["correct"] for r in rows]
    geo = [estimator.score(r["z"]) for r in rows]
    conf = [1 - r["z"]["confidence"] for r in rows]
    natural = [i for i, r in enumerate(rows) if r["category"] != "corrupted_model"]
    by_cat = defaultdict(list)
    for r, g in zip(rows, geo): by_cat[(r["category"], "correct" if r["correct"] else "incorrect")].append((g, r["z"]))
    return {"n": len(rows), "n_incorrect": sum(labels),
            "auroc": {"confidence": auroc(conf, labels), "geometry": auroc(geo, labels)},
            "auprc": {"confidence": auprc(conf, labels), "geometry": auprc(geo, labels)},
            "natural_only": {"n": len(natural), "n_incorrect": sum(labels[i] for i in natural),
                             "auroc_confidence": auroc([conf[i] for i in natural], [labels[i] for i in natural]),
                             "auroc_geometry": auroc([geo[i] for i in natural], [labels[i] for i in natural])},
            "calibration_ece_geometry": ece(geo, [float(l) for l in labels]),
            "by_category": {f"{c}/{s}": {"n": len(v), "risk": round(sum(g for g, _ in v) / len(v), 3),
                                         "drift": round(sum(z["drift"] for _, z in v) / len(v), 3),
                                         "confidence": round(sum(z["confidence"] for _, z in v) / len(v), 3)}
                            for (c, s), v in sorted(by_cat.items())},
            "coefficients": estimator.coefficients()}


FEATURE_SETS = {
    "confidence_only": ["neural_confidence"],
    "geometry_only": ["geometric_risk"],
    "structure_only": ["support", "source_reliability"],
    "confidence+geometry": ["neural_confidence", "geometric_risk"],
    "confidence+structure": ["neural_confidence", "support", "source_reliability"],
    "confidence+structure+geometry": ["neural_confidence", "support", "source_reliability", "geometric_risk"],
    "controls": ["neural_confidence", "support", "source_reliability", "length", "novelty", "ambiguity"],
    "controls+geometry": ["neural_confidence", "support", "source_reliability", "length", "novelty", "ambiguity", "geometric_risk"],
}


def _fit_logistic(x, y, steps=600, l2=1e-2):
    mean, std = x.mean(0), x.std(0) + 1e-6
    xs = (x - mean) / std
    w = torch.zeros(x.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xs @ w + b, y) + l2 * (w ** 2).sum()
        loss.backward(); opt.step()
    return mean, std, w.detach(), b.detach()


def _predict(model, x):
    mean, std, w, b = model
    return torch.sigmoid(((x - mean) / std) @ w + b).tolist()


def query_level_features(agent, worlds):
    """For each query: candidate BEFORE policy, error label, and explanatory variables."""
    rows = run_split(agent, worlds)
    out = []
    for r in rows:
        a = r["answer"]
        candidate = a.candidate_output if a.candidate_output is not None else a.output
        f = dict(a.features); f.setdefault("geometric_risk", 0.0)
        out.append({"features": f, "error": float(candidate != r["item"].gold), "category": r["item"].category,
                    "has_neural": bool(a.neural_words)})
    return out


def conditional_geometry(setup: Setup, seed=0, bootstrap=300):
    """Does geometry improve error prediction AFTER controlling for confidence, support,
    length, lexical novelty, ambiguity and source reliability?"""
    agent = make_agent("G_full", setup, seed=seed)
    agent.cfg.correction = False
    fit_rows = query_level_features(agent, setup.splits["calibration"] + setup.splits["validation"])
    test_rows = query_level_features(agent, setup.splits["test"])
    results = {}
    y_fit = torch.tensor([r["error"] for r in fit_rows]); y_test = [r["error"] for r in test_rows]
    models = {}
    for name, feats in FEATURE_SETS.items():
        x_fit = torch.tensor([[r["features"].get(f, 0.0) for f in feats] for r in fit_rows], dtype=torch.float)
        x_test = torch.tensor([[r["features"].get(f, 0.0) for f in feats] for r in test_rows], dtype=torch.float)
        model = _fit_logistic(x_fit, y_fit)
        p = _predict(model, x_test)
        models[name] = (model, feats, p)
        results[name] = {"auroc": auroc(p, [bool(v) for v in y_test]), "auprc": auprc(p, [bool(v) for v in y_test]),
                         "ece": ece(p, y_test)}
    # effect of geometry: Δ AUROC (controls+geometry - controls) with bootstrap CI on the test set
    rng = random.Random(seed)
    p0, p1 = models["controls"][2], models["controls+geometry"][2]
    deltas = []
    n = len(test_rows)
    for _ in range(bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        lab = [bool(y_test[i]) for i in idx]
        a0, a1 = auroc([p0[i] for i in idx], lab), auroc([p1[i] for i in idx], lab)
        if a0 is not None and a1 is not None: deltas.append(a1 - a0)
    deltas.sort()
    mean, std, w, _ = models["controls+geometry"][0]
    coef = {f: round(float(c), 3) for f, c in zip(FEATURE_SETS["controls+geometry"], w)}
    neural_subset = [i for i, r in enumerate(test_rows) if r["has_neural"]]
    sub = {}
    if neural_subset:
        lab = [bool(y_test[i]) for i in neural_subset]
        for name in ("confidence_only", "geometry_only", "controls", "controls+geometry"):
            sub[name] = auroc([models[name][2][i] for i in neural_subset], lab)
    return {"n_fit": len(fit_rows), "n_test": len(test_rows), "test_error_rate": round(sum(y_test) / len(y_test), 3),
            "feature_sets": results,
            "delta_auroc_geometry_after_controls": {"mean": round(sum(deltas) / len(deltas), 4) if deltas else None,
                                                    "ci95": (round(deltas[int(0.025 * len(deltas))], 4), round(deltas[int(0.975 * len(deltas)) - 1], 4)) if deltas else None},
            "standardized_coefficients": coef,
            "subset_with_neural_words": {"n": len(neural_subset), "auroc": sub}}


# ================================================================ grammar: learning curve

def grammar_accuracy(agent, held):
    role_ok = role_n = prep_ok = prep_n = conn_ok = conn_n = 0
    for c in held:
        toks = tokenize(c["text"]); s = c["structure"]
        if s["kind"] == "sequence":
            conn = next((t for t in toks if t in ("puis", "après")), None)
            if conn:
                conn_n += 1
                conn_ok += agent.grammar.connector(conn) == ("keep" if conn == "puis" else "reverse")
            continue
        nps = agent.grammar.chunks(toks)
        from core.independent_agent import ROLES
        kind = s["kind"]
        if kind not in ROLES or not nps: continue
        roles = agent.grammar.assign(toks, nps, kind)
        slots = {r: s.get(r) for r in ROLES[kind] if isinstance(s.get(r), dict)}
        for i, np in enumerate(nps):
            matches = [r for r, d in slots.items() if d and np.desc and all(d.get(k) == v for k, v in np.desc.items())]
            if len(matches) == 1:
                role_n += 1; role_ok += roles.get(i) == matches[0]
        if "rel" in s:
            rels = [agent.lex.meaning(t) for t in toks if token_class(agent.lex.meaning(t)) == "R"]
            prep_n += 1; prep_ok += bool(rels) and rels[0] == f"rel:{s['rel']}"
    return {"roles": round(role_ok / max(1, role_n), 3), "prepositions": round(prep_ok / max(1, prep_n), 3),
            "clause_order": round(conn_ok / max(1, conn_n), 3) if conn_n else None}


def grammar_learning_curve(sizes=(10, 20, 40, 80, 160, 320), seed=0):
    held = captions(300, seed=seed + 9)
    out = {}
    for n in sizes:
        agent = IndependentAgent(AgentConfig(neural=False, geometry=False), seed).learn_captions(captions(n, seed=seed + 1))
        out[n] = grammar_accuracy(agent, held)
    return out


def preposition_table(agent):
    return {t: m for t, m in agent.lex.lexicon().items() if m.startswith("rel:")}


# ================================================================ induced lexical classes

def induced_classes(agent, k=6, iterations=20, seed=0):
    """Functional clusters from behaviour (co-occurring symbols, position) — without class names."""
    toks = [t for t, c in agent.lex.tok.items() if c >= 3]
    prefixes = ["type", "color", "rel", "act", "kind", "time", "person", "num"]
    vecs = []
    for t in toks:
        co = agent.lex.co[t]; total = agent.lex.tok[t]
        best = Counter()
        for sym, c in co.items(): best[sym.split(":")[0]] = max(best[sym.split(":")[0]], c / total)
        after = agent.grammar.class_after.get(t, Counter()); s = sum(after.values()) or 1
        vecs.append([best[p] for p in prefixes] + [after[c] / s for c in ("N", "C", "R", "A", "F")])
    x = torch.tensor(vecs)
    rng = random.Random(seed)
    centers = x[rng.sample(range(len(toks)), k)].clone()
    for _ in range(iterations):
        assign = torch.cdist(x, centers).argmin(1)
        for j in range(k):
            if (assign == j).any(): centers[j] = x[assign == j].mean(0)
    clusters = defaultdict(list)
    for t, a in zip(toks, assign.tolist()): clusters[a].append(t)
    gold = lambda t: (agent.lex.meaning(t) or "function:").split(":")[0]
    purity = sum(Counter(gold(t) for t in ts).most_common(1)[0][1] for ts in clusters.values()) / len(toks)
    return {"clusters": {c: sorted(ts)[:12] for c, ts in clusters.items()}, "purity_vs_meaning_class": round(purity, 3), "tokens": len(toks)}


# ================================================================ lexical growth and representations

def synthetic_lexicon(n_forms, seed=0):
    """Artificial forms (stem + inflectional suffix) for meanings; variants held out."""
    rng = random.Random(seed)
    syll = ["ka", "lo", "mi", "ru", "ten", "vo", "shi", "pa", "dor", "lin", "fu", "ze", "bra", "qui", "nel"]
    meanings = [f"m{i}" for i in range(max(4, n_forms // 5))]
    suffixes = ["", "e", "s", "es", "ait"]
    train, held = {}, {}
    for m in meanings:
        stem = "".join(rng.choice(syll) for _ in range(rng.randint(2, 3)))
        forms = [stem + s for s in suffixes]
        for f in forms[:4]: train[f] = m
        held[forms[4]] = m
    items = list(train.items())[:n_forms]
    return dict(items), {f: m for f, m in held.items() if m in {v for _, v in items}}


class BPE:
    """Minimal subwords (frequent merges of character pairs), without external library."""

    def __init__(self, words, merges=60):
        self.merges = []
        corpus = [list(w) for w in words]
        for _ in range(merges):
            pairs = Counter((a, b) for w in corpus for a, b in zip(w, w[1:]))
            if not pairs: break
            (a, b), _ = pairs.most_common(1)[0]
            self.merges.append((a, b))
            corpus = [self._merge(w, a, b) for w in corpus]
        self.vocab = {p: i + 1 for i, p in enumerate(sorted({p for w in corpus for p in w} | set("".join(words))))}

    @staticmethod
    def _merge(w, a, b):
        out, i = [], 0
        while i < len(w):
            if i + 1 < len(w) and w[i] == a and w[i + 1] == b: out.append(a + b); i += 2
            else: out.append(w[i]); i += 1
        return out

    def encode(self, word):
        w = list(word)
        for a, b in self.merges: w = self._merge(w, a, b)
        return [self.vocab.get(p, 0) for p in w]


class BagModel(torch.nn.Module):
    def __init__(self, vocab, classes, dim=16, hidden=32):
        super().__init__()
        self.emb = torch.nn.EmbeddingBag(vocab + 1, dim, mode="mean")
        self.head = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, classes))

    def forward(self, ids, offsets): return self.head(self.emb(ids, offsets))


def _bag(encoded):
    ids, offsets = [], []
    for e in encoded: offsets.append(len(ids)); ids += e or [0]
    return torch.tensor(ids), torch.tensor(offsets)


def representation_comparison(n_forms=250, seed=0):
    """Characters vs subwords (minimal BPE) vs words, on held-out inflected forms."""
    train, held = synthetic_lexicon(n_forms, seed)
    labels = sorted(set(train.values()))
    y = torch.tensor([labels.index(m) for m in train.values()])
    out = {}
    t0 = time.time()
    core = CharCore("small", seed, concepts=labels)
    TrainCore(words=tuple(train), labels=tuple(train.values()), steps=300).apply(core)
    out["characters"] = {"parameters": core.parameter_count, "oov_rate": 0.0,
                         "heldout_accuracy": round(sum(core.propose(w, 1)[0][0] == m for w, m in held.items()) / len(held), 3),
                         "train_seconds": round(time.time() - t0, 2)}
    for name, encoder in (("subword_bpe", BPE(list(train))), ("word", None)):
        t0 = time.time()
        if encoder is None:
            vocab = {w: i + 1 for i, w in enumerate(train)}
            enc = lambda w: [vocab.get(w, 0)]
            size = len(vocab)
        else:
            enc, size = encoder.encode, len(encoder.vocab)
        model = BagModel(size, len(labels))
        ids, offs = _bag([enc(w) for w in train])
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        for _ in range(300):
            opt.zero_grad(); torch.nn.functional.cross_entropy(model(ids, offs), y).backward(); opt.step()
        hid, hoffs = _bag([enc(w) for w in held])
        with torch.no_grad(): pred = model(hid, hoffs).argmax(1).tolist()
        oov = sum(all(i == 0 for i in enc(w)) for w in held) / len(held)
        out[name] = {"parameters": sum(p.numel() for p in model.parameters()), "oov_rate": round(oov, 3),
                     "heldout_accuracy": round(sum(labels[p] == m for p, m in zip(pred, held.values())) / len(held), 3),
                     "train_seconds": round(time.time() - t0, 2), "vocabulary": size}
    out["n_train_forms"], out["n_heldout_forms"] = len(train), len(held)
    return out


def lexicon_growth(sizes=(50, 100, 250, 500, 1000), seed=0):
    """Lexical size vs neural parameters vs structural memory vs accuracy."""
    out = {}
    for n in sizes:
        train, held = synthetic_lexicon(n, seed)
        labels = sorted(set(train.values()))
        core = CharCore("small", seed, concepts=labels)
        t0 = time.time()
        TrainCore(words=tuple(train), labels=tuple(train.values()), steps=300).apply(core)
        train_acc = sum(core.propose(w, 1)[0][0] == m for w, m in train.items()) / len(train)
        held_acc = sum(core.propose(w, 1)[0][0] == m for w, m in held.items()) / len(held)
        out[n] = {"lexicon_forms": len(train), "meanings": len(labels), "neural_parameters": core.parameter_count,
                  "structural_memory_entries": len(train), "neural_train_accuracy": round(train_acc, 3),
                  "neural_heldout_form_accuracy": round(held_acc, 3), "structural_exact_recall": 1.0,
                  "train_seconds": round(time.time() - t0, 2)}
    return out


# ================================================================ distillation and scale

def distillation_report(setup: Setup, seed=0):
    """WCE (teacher) -> neural student: distilled lexicon; structural verifications before/after."""
    exhaustive = IndependentAgent(AgentConfig(neural=False, geometry=False, correction=False, verification="exhaustive"), seed)
    exhaustive.learn_captions(setup.captions); exhaustive.warm_up_sources(setup.splits["train"])
    distilled = IndependentAgent(AgentConfig(geometry=False, correction=False, verification="selective"), seed)
    distilled.learn_captions(setup.captions); distilled.warm_up_sources(setup.splits["train"])
    a, b = summarize(run_split(exhaustive, setup.splits["test"])), summarize(run_split(distilled, setup.splits["test"]))
    lex = distilled.lex.lexicon()
    student_fidelity = sum(distilled.neural.propose(w, 1)[0][0] == m for w, m in lex.items()) / len(lex)
    return {"teacher_lexicon": len(lex), "student_fidelity_on_lexicon": round(student_fidelity, 3),
            "student_parameters": distilled.neural.core.parameter_count,
            "checks_before (WCE exhaustive)": a["verification_cost"], "checks_after (distilled + selective)": b["verification_cost"],
            "success_before": a["success"], "success_after": b["success"],
            "false_confident_before": a["false_confident_rate"], "false_confident_after": b["false_confident_rate"],
            "distilled": ["lexical concepts", "routing of OOV words (selective verification)"],
            "not_distilled": ["grammar roles (still WCE counts)", "ambiguity detection (still WCE)"]}


def scaling(setup: Setup, sizes=("tiny", "small", "medium"), seed=0):
    out = {}
    for size in sizes:
        agent = make_agent("G_full", setup, size, seed)
        s = summarize(run_split(agent, setup.splits["test"]), agent)
        test_items = [it for x in setup.splits["test"] for it in x["items"]]
        words = oov_words(test_items, agent)
        rows = word_cases(agent, agent.monitor, words)
        lex_acc = sum(r["correct"] for r in rows) / max(1, len(rows))
        geo = auroc([agent.estimator.score(r["z"]) for r in rows], [not r["correct"] for r in rows])
        out[size] = {"parameters": s["parameters"], "success": s["success"], "false_confident_rate": s["false_confident_rate"],
                     "abstention_rate": s["abstention_rate"], "verification_cost": s["verification_cost"],
                     "grammar": grammar_accuracy(agent, captions(150, seed=seed + 9)), "neural_oov_lexical_accuracy": round(lex_acc, 3),
                     "geometry_auroc_oov": geo}
    return out


def source_report(agent):
    return {s: {"estimated": agent.reliability.table().get(s), "true": SOURCE_BY_NAME[s].reliability} for s in SOURCE_BY_NAME}

"""Calibrated lexical posterior, entropy, lifecycle, polysemy, similarity by behaviour, classes, composition.

P(c | w, evidence) = softmax_c( θ · φ(w, c) ) over the candidate concepts + a "none of these" option (NULL).
φ : net support, number of compatible situations, diversity of sources, contradictions, margin over the competitor,
    morphological agreement (probability of the rule), grammatical plausibility of the context, co-occurrence with
    observed transformations. θ is fitted on CALIBRATION seeds (never the test seeds).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from core.independent_agent import token_class, tokenize
from core.unified import calibration_report

LEX_UNKNOWN, LEX_HYPOTHESIS, LEX_BELIEF, LEX_STABLE = "UNKNOWN", "HYPOTHESIS", "BELIEF", "STABLE_MAPPING"
FEATURES = ("net_support", "log_situations", "sources", "refute", "margin", "morphology", "grammar", "transformation", "bias")
NULL = "NULL"


# ================================================================ morphological uncertainty

def morphology_probability(agent, word, concept):
    """Probability of a suffix rule: Beta(stems that support it, counter-examples), + grounding.
    Counter-example: a form that applies the rule but whose grounded meaning differs from that of the reference."""
    known = agent.lex.lexicon()
    best = 0.0
    for ref, meaning, rule, support in agent.morph.candidates(word, known):
        if meaning != concept: continue
        counter = 0
        for form, m in known.items():
            if rule.applies(form):
                r2 = rule.apply(form)
                if r2 in known and known[r2] != m: counter += 1
        p = (support + 1) / (support + counter + 2)
        best = max(best, p)
    return best


# ================================================================ features and posterior

def _contexts(h):
    out = []
    for exps in h.experiences.values():
        for e in exps:
            text = e.split(":", 1)[1] if ":" in e else e
            toks = tokenize(text)
            if h.surface in toks: out.append(toks)
    return out


def features(agent, h, concept):
    s, r = h.support[concept], h.refute[concept]
    second = max((h.support[c] for c in h.support if c != concept), default=0.0)
    ctx = _contexts(h)
    gram = 0.0
    if ctx:
        ok = 0
        for toks in ctx:
            i = toks.index(h.surface); prev = toks[i - 1] if i > 0 else "<s>"
            ok += agent.grammar.class_plausible(prev, token_class(concept))
        gram = ok / len(ctx)
    trans = 0.0
    if ctx and concept.startswith("act:"):
        trans = sum(any((agent.lex.meaning(t) or "").startswith("rel:") for t in toks) for toks in ctx) / len(ctx)
    return [s - r, math.log1p(len(h.experiences[concept])), len(h.sources[concept]), r, s - second,
            morphology_probability(agent, h.surface, concept), gram, trans, 1.0]


@dataclass
class LexicalExample:
    word: str
    concepts: List[str]
    feats: torch.Tensor          # [C, F]
    gold: Optional[str]          # correct concept (evaluation) or None


class LexicalPosterior(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(len(FEATURES)))
        self.null_bias = torch.nn.Parameter(torch.zeros(1))
        self.fitted = False

    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())

    def forward(self, feats, mask):                 # [B, C, F], [B, C] -> P [B, C + 1] (last = NULL)
        s = (feats * self.theta).sum(-1).masked_fill(~mask, -1e9)
        s = torch.cat([s, self.null_bias.expand(s.shape[0], 1)], 1)
        return torch.softmax(s, 1)

    def fit(self, examples: List[LexicalExample], epochs=400, lr=0.05, l2=1e-3):
        feats, mask, y = _pad(examples)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            p = self(feats, mask)
            loss = -(p.gather(1, y.view(-1, 1)).clamp_min(1e-9).log()).mean() + l2 * (self.theta ** 2).sum()
            loss.backward(); opt.step()
        self.fitted = True
        return self

    def distribution(self, ex: LexicalExample) -> Dict[str, float]:
        if not ex.concepts: return {NULL: 1.0}
        with torch.no_grad():
            p = self(ex.feats.unsqueeze(0), torch.ones(1, len(ex.concepts), dtype=torch.bool))[0]
        return {**{c: float(p[i]) for i, c in enumerate(ex.concepts)}, NULL: float(p[-1])}


def _pad(examples):
    C = max(1, max(len(e.concepts) for e in examples))
    F = len(FEATURES)
    feats = torch.zeros(len(examples), C, F); mask = torch.zeros(len(examples), C, dtype=torch.bool)
    y = torch.full((len(examples),), C, dtype=torch.long)                      # C = NULL
    for i, e in enumerate(examples):
        n = len(e.concepts)
        if n: feats[i, :n] = e.feats; mask[i, :n] = True
        if e.gold in e.concepts: y[i] = e.concepts.index(e.gold)
    return feats, mask, y


def examples_from_agent(agent, gold: Dict[str, str] = None, words=None):
    out = []
    for w, h in agent.lexmem.hyps.items():
        if words is not None and w not in words: continue
        concepts = sorted(c for c in set(h.support) | set(h.refute) if h.support[c] > 0)
        if not concepts: continue
        f = torch.tensor([features(agent, h, c) for c in concepts], dtype=torch.float)
        out.append(LexicalExample(w, concepts, f, (gold or {}).get(w)))
    return out


# ================================================================ entropy and calibrated lifecycle

def entropy(dist, n_concepts=None):
    """Entropy H(C | w). With n_concepts: "open" entropy — the NULL mass is spread over the concepts not
    yet considered (no candidate = maximal ignorance log K, and not certainty)."""
    h = -sum(p * math.log(p) for c, p in dist.items() if p > 0 and c != NULL)
    pn = dist.get(NULL, 0.0)
    if pn > 0:
        rest = max(1, (n_concepts or 1) - sum(1 for c in dist if c != NULL))
        h += -pn * math.log(pn / rest)
    return h


@dataclass
class LexicalLifecycle:
    """Transitions driven by the calibrated posterior (thresholds = expected risk: cost of an error 20, abstention 1)."""
    hypothesis: float = 0.5
    belief: float = 0.95
    stable: float = 0.99
    max_entropy: float = 0.5
    min_situations: int = 2          # an isolated observation cannot be told apart from a coincidence

    def status(self, dist, n_sources=0, n_situations=0):
        best = max((c for c in dist if c != NULL), key=lambda c: dist[c], default=None)
        if best is None: return LEX_UNKNOWN, None
        p = dist[best]
        if p >= self.stable and n_sources >= 2 and n_situations >= 3 and entropy(dist) <= self.max_entropy: return LEX_STABLE, best
        if p >= self.belief and n_situations >= self.min_situations: return LEX_BELIEF, best
        if p >= self.hypothesis: return LEX_HYPOTHESIS, best
        return LEX_UNKNOWN, best


class CalibratedLexicon:
    """Attaches to an agent: meaning of a word learned from the observations according to the calibrated posterior."""

    def __init__(self, posterior: LexicalPosterior, lifecycle: LexicalLifecycle = None, min_status=LEX_BELIEF):
        self.post, self.life, self.min_status = posterior, lifecycle or LexicalLifecycle(), min_status
        self.history = defaultdict(list)

    def dist(self, agent, word):
        h = agent.lexmem.get(word)
        if h is None: return {NULL: 1.0}
        ex = examples_from_agent(agent, words={word})
        return self.post.distribution(ex[0]) if ex else {NULL: 1.0}

    def status(self, agent, word):
        d = self.dist(agent, word)
        h = agent.lexmem.get(word)
        best = max((c for c in d if c != NULL), key=lambda c: d[c], default=None)
        st, c = self.life.status(d, len(h.sources[best]) if h and best else 0, len(h.experiences[best]) if h and best else 0)
        hist = self.history[word]
        if not hist or hist[-1] != st: hist.append(st)            # regression visible if a contradiction lowers P
        return st, c, d.get(c, 0.0) if c else 0.0

    def meaning(self, agent, word):
        order = [LEX_UNKNOWN, LEX_HYPOTHESIS, LEX_BELIEF, LEX_STABLE]
        st, c, p = self.status(agent, word)
        return (c, p) if order.index(st) >= order.index(self.min_status) else (None, p)


def attach(agent, calibrated: CalibratedLexicon):
    """Replaces the lexical decision (fixed thresholds of LexicalMemory) with the calibrated posterior."""
    original = agent.word_meaning

    def word_meaning(tok):
        m, src, conf = original(tok)
        if src in ("grounded_obs",) or (src == "unknown" and agent.lexmem.get(tok) is not None):
            c, p = calibrated.meaning(agent, tok)
            if c is not None: return c, "grounded_obs", p
            if src == "grounded_obs": return None, "unknown", 0.0
        return m, src, conf
    agent.word_meaning = word_meaning
    agent.calibrated_lexicon = calibrated
    return agent


# ================================================================ polysemy: P(sense | word, syntax, domain, entity type)

class PolysemyPosterior:
    """Smoothed naive Bayes learned from the captions: sense | (patient type, target type, preposition, domain)."""

    def __init__(self, alpha=1.0): self.alpha = alpha; self.counts = defaultdict(Counter); self.prior = Counter()

    @staticmethod
    def context(structure):
        pt = (structure.get("patient") or {}).get("type"); tt = (structure.get("target") or {}).get("type")
        return {"patient": pt, "target": tt, "rel": structure.get("rel"),
                "domain": "inventory" if tt == "list" else "physical"}

    def fit(self, captions, word="range"):
        for c in captions:
            toks = tokenize(c["text"]); s = c["structure"]
            if word not in toks or s.get("kind") != "event": continue
            sense = f"act:{s['action']}"; self.prior[sense] += 1
            for k, v in self.context(s).items(): self.counts[(sense, k)][v] += 1
        return self

    def posterior(self, ctx):
        senses = list(self.prior); total = sum(self.prior.values())
        if not senses: return {}
        logp = {}
        for s in senses:
            lp = math.log((self.prior[s] + self.alpha) / (total + self.alpha * len(senses)))
            for k, v in ctx.items():
                cnt = self.counts[(s, k)]
                lp += math.log((cnt[v] + self.alpha) / (sum(cnt.values()) + self.alpha * (len(cnt) + 1)))
            logp[s] = lp
        m = max(logp.values()); z = sum(math.exp(v - m) for v in logp.values())
        return {s: math.exp(v - m) / z for s, v in logp.items()}


# ================================================================ similarity by behaviour, classes, analogy

def behavior_matrix(captions):
    """Word -> behaviour profile (structure type, action, relation, role), without external embeddings."""
    rows = defaultdict(Counter)
    for c in captions:
        s = c["structure"]; toks = tokenize(c["text"])
        feats = [f"kind={s.get('kind')}", f"action={s.get('action')}", f"rel={s.get('rel')}"]
        for i, t in enumerate(toks):
            nxt = toks[i + 1] if i + 1 < len(toks) else "</s>"; prv = toks[i - 1] if i else "<s>"
            for f in feats: rows[t][f] += 1
            rows[t][f"prev_class={_cls(prv)}"] += 1; rows[t][f"next_class={_cls(nxt)}"] += 1
    words = sorted(w for w in rows if sum(rows[w].values()) >= 3)
    vocab = sorted({f for w in words for f in rows[w]})
    idx = {f: i for i, f in enumerate(vocab)}
    M = torch.zeros(len(words), len(vocab))
    for i, w in enumerate(words):
        for f, v in rows[w].items(): M[i, idx[f]] = v
    M = M / M.sum(1, keepdim=True).clamp_min(1)
    return words, vocab, M


def _cls(tok):
    if tok in ("<s>", "</s>"): return tok
    if tok in ("le", "la", "l'", "les", "un", "une", "du", "des", "de"): return "DET"
    if tok in ("est", "était", "a", "sont"): return "AUX"
    return "W"


class SimilarityByBehavior:
    def __init__(self, captions):
        self.words, self.vocab, M = behavior_matrix(captions)
        self.M = M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
        self.index = {w: i for i, w in enumerate(self.words)}

    def similarity(self): return self.M @ self.M.T                       # vectorised cosine

    def classes(self, threshold=0.9):
        """Emergent functional classes: connected components of the similarity graph (no hard-coded names)."""
        S = self.similarity() >= threshold
        n = len(self.words); parent = list(range(n))
        def find(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        ii, jj = torch.nonzero(torch.triu(S, 1), as_tuple=True)
        for i, j in zip(ii.tolist(), jj.tolist()): parent[find(i)] = find(j)
        groups = defaultdict(list)
        for i in range(n): groups[find(i)].append(self.words[i])
        return [sorted(g) for g in groups.values() if len(g) > 1]

    def analogy(self, contexts: List[List[str]], known: Dict[str, str], k=1):
        """Unknown word: frame profile (neighbouring classes) -> closest known word -> CANDIDATE by analogy
        (provenance 'analogy': never promoted to a fact)."""
        v = torch.zeros(len(self.vocab)); vidx = {f: i for i, f in enumerate(self.vocab)}
        for toks, i in contexts:
            for f in (f"prev_class={_cls(toks[i - 1] if i else '<s>')}", f"next_class={_cls(toks[i + 1] if i + 1 < len(toks) else '</s>')}"):
                if f in vidx: v[vidx[f]] += 1
        if v.sum() == 0: return None, 0.0
        v = v / v.norm()
        mask = torch.tensor([w in known for w in self.words])
        sims = (self.M @ v).masked_fill(~mask, -1)
        j = int(sims.argmax())
        return (known[self.words[j]], float(sims[j])) if float(sims[j]) > 0 else (None, 0.0)


def class_purity(classes, meaning_of):
    """Evaluation only: fraction of the words whose meaning shares the majority family of their class."""
    tot = ok = 0
    for g in classes:
        fams = Counter((meaning_of.get(w) or "none").split(":")[0] for w in g)
        tot += len(g); ok += fams.most_common(1)[0][1]
    return ok / tot if tot else 0.0


# ================================================================ lexical composition

def stem_lookup(word, lexicon, prefix=4):
    """Candidate by shared stem (≥ 4 letters) with a known action form; unique, otherwise None (uncertain)."""
    if len(word) < prefix: return None
    hits = {m for f, m in lexicon.items() if m and m.startswith("act:") and f[:prefix] == word[:prefix]}
    return hits.pop() if len(hits) == 1 else None


def compose(expr, word_meaning, lexicon=None):
    """"remet / repose" (re- + verb), "fait tomber" (causative), "met dedans" (verb + place particle).
    Returns (meaning, trace) or (None, trace): no independent entry per surface form."""
    toks = expr.split()
    trace = []
    if lexicon is not None:
        base_wm = word_meaning
        word_meaning = lambda t: base_wm(t) or stem_lookup(t, lexicon)
    if len(toks) == 1 and toks[0].startswith("re") and len(toks[0]) > 4:
        inner = word_meaning(toks[0][2:]) or word_meaning(toks[0][2:] + "s") or word_meaning(toks[0][2:-1] + "s")
        trace.append(f"re- + {toks[0][2:]} -> {inner}")
        return (inner, trace) if inner and inner.startswith("act:") else (None, trace)
    if len(toks) == 2 and toks[0] in ("fait", "faire", "fais"):
        inner = word_meaning(toks[1]); trace.append(f"causatif(faire) + {toks[1]} -> {inner}")
        return (f"cause({inner})", trace) if inner and inner.startswith("act:") else (None, trace)
    if len(toks) == 2 and toks[1] in ("dedans", "dessus", "dessous"):
        verb = word_meaning(toks[0]) or word_meaning(toks[0] + "s")
        rel = {"dedans": "rel:in", "dessus": "rel:on", "dessous": "rel:under"}[toks[1]]
        trace.append(f"{toks[0]} -> {verb} ; {toks[1]} -> {rel}")
        return ((verb, rel), trace) if verb and verb.startswith("act:") else (None, trace)
    return None, trace


# ================================================================ lexical calibration measures

def calibration_points(examples, model: Optional[LexicalPosterior] = None, agent_conf=None):
    """(confidence, exact): calibrated posterior if a model is given, otherwise native confidence (LexicalHypothesis.confidence)."""
    pts = []
    for ex in examples:
        if ex.gold is None: continue
        if model is not None:
            d = model.distribution(ex); best = max(d, key=d.get)
            pts.append((d[best], best == ex.gold) if best != NULL else (d[NULL], ex.gold not in ex.concepts))
        else:
            c, best = agent_conf[ex.word]
            pts.append((c, best == ex.gold))
    return pts


def precision_at(pts, thresholds=(0.5, 0.7, 0.9, 0.95)):
    out = {}
    for t in thresholds:
        sel = [y for c, y in pts if c >= t]
        out[t] = (round(sum(sel) / len(sel), 3) if sel else None, len(sel))
    return out

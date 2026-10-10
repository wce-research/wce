"""Grounded understanding: joint grammar, evidence-grounded lexicon, polysemy, explained abstentions.

The agent stays isolated from the truth: it does not import the generator; it learns only from captions,
from observations (sensors, texts) and from consistency with its own evidence graph.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from core.independent_agent import (
    ABSTAIN, BELIEF, FACT, INNATE_OTHER, INNATE_PRONOUN, INNATE_QUESTION, NP, ROLES, SUPPORTS, AgentConfig,
    Evidence, GrammarInducer, IndependentAgent, LexiconInducer, affordance, is_number, token_class, tokenize,
)


class GroundedLexicon(LexiconInducer):
    """Cross-situational lexicon + EM alignment (IBM Model 1): each perceived symbol must be "produced"
    by a word of the sentence; the words compete for the symbols. A word that is always next to another
    ("contient" / "liste") therefore does not steal its meaning."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._pairs = []
        self.t = defaultdict(dict)

    def observe(self, tokens, structure):
        super().observe(tokens, structure)
        from core.independent_agent import structure_symbols
        self._pairs.append((list(set(tokens)), list(structure_symbols(structure))))

    def align(self, iterations=12):
        """Vectorised IBM Model 1: flat arrays (sentence, symbol, word) + index_add, no loop over the pairs."""
        words = sorted({w for toks, _ in self._pairs for w in toks}); syms = sorted({x for _, ss in self._pairs for x in ss})
        wi = {w: i for i, w in enumerate(words)}; si = {x: i for i, x in enumerate(syms)}
        cap, sid, wid = [], [], []
        for c, (toks, ss) in enumerate(self._pairs):
            for x in ss:
                for w in toks: cap.append(c * len(syms) + si[x]); sid.append(si[x]); wid.append(wi[w])
        cap, sid, wid = torch.tensor(cap), torch.tensor(sid), torch.tensor(wid)
        S, W = len(syms), len(words)
        t = torch.full((S, W), 1.0 / max(1, S), dtype=torch.float64)
        for _ in range(iterations):
            vals = t[sid, wid]
            z = torch.zeros(len(self._pairs) * S, dtype=torch.float64).index_add_(0, cap, vals)
            post = vals / z[cap].clamp_min(1e-12)
            counts = torch.zeros(S * W, dtype=torch.float64).index_add_(0, sid * W + wid, post).view(S, W)
            t = counts / counts.sum(0, keepdim=True).clamp_min(1e-12)
        self.t = defaultdict(dict)
        nz = (t > 1e-6).nonzero().tolist()
        for a, b in nz: self.t[words[b]][syms[a]] = float(t[a, b])
        self._cache = {}

    def meaning(self, token):
        if token in self._cache: return self._cache[token]
        result = None
        if self.t and self.tok[token] >= self.min_count and len(token) > 2 and not token.endswith("'"):
            probs = self.t.get(token, {})
            cands = [(v, s) for s, v in probs.items() if self.co[token][s] / self.tok[token] >= 0.9]
            content = [(v, s) for v, s in cands if not s.startswith("kind:") and v >= 0.15]
            best = max(content) if content else (max(cands) if cands else None)
            if best and best[0] >= 0.15: result = best[1]
        elif not self.t:
            result = super().meaning(token)
        self._cache[token] = result
        return result


LEX_UNKNOWN, LEX_HYPOTHESIS, LEX_BELIEF, LEX_FACT = "UNKNOWN", "HYPOTHESIS", "BELIEF", "FACT_LIKE"
REQUIRED = {"loc": ("subject", "target"), "event": ("patient",), "color": ("subject",), "height": ("subject",),
            "owner": ("subject",), "count": ("target",)}


# ================================================================ joint grammar (structured perceptron)

@dataclass
class ParseHypothesis:
    tokens: List[str]
    kind: str
    roles: Dict[int, str]
    relations: List[str]
    clauses: int
    references: Dict[int, Any]
    score: float
    structural_support: float
    propositions: List[Tuple] = field(default_factory=list)


class JointGrammarInducer(GrammarInducer):
    """Roles decided JOINTLY for all the phrases of a sentence.

    The constraints (distinct roles, possible null role, required roles, relative order, markers such as "par",
    next word, phrase type) are primitives; their WEIGHTS are learned by averaged perceptron
    on the captions (aligned observations). No French grammar rule is written by hand."""

    def __init__(self, lex, joint=True):
        super().__init__(lex)
        self.joint = joint
        self._data = []
        self.w = defaultdict(float); self._u = defaultdict(float); self._c = 1
        self.rank = 0
        self.last_margin = math.inf
        self.explored = 0
        self.student = None

    def learn(self, tokens, structure):
        super().learn(tokens, structure)
        if structure["kind"] != "sequence": self._data.append((tokens, structure))

    @staticmethod
    def np_kind(np):
        return "pron" if np.pronoun else "generic" if np.generic else "person" if np.person else "type"

    def gold(self, tokens, nps, structure):
        kind = structure["kind"]
        slots = {r: structure.get(r) for r in ROLES.get(kind, ()) if isinstance(structure.get(r), dict)}
        if kind == "owner": slots["value"] = {"person": structure["value"]}
        out = []
        for np in nps:
            m = [r for r, d in slots.items() if d and np.desc and all(d.get(k) == v for k, v in np.desc.items())]
            if len(m) > 1: return None
            out.append(m[0] if m else "none")
        return tuple(out)

    def feats(self, tokens, nps, kind, assign):
        f = []
        for i, (np, role) in enumerate(zip(nps, assign)):
            prev = tokens[np.start - 1] if np.start > 0 else "<s>"
            prev2 = tokens[np.start - 2] if np.start > 1 else "<s>"
            nxt = tokens[np.end] if np.end < len(tokens) else "</s>"
            f += [("p", kind, prev, role), ("p2", kind, prev2, role), ("n", kind, nxt, role),
                  ("pc", kind, token_class(self.lex.meaning(prev)), role), ("r", kind, min(i, 3), role),
                  ("nk", kind, self.np_kind(np), role), ("b", kind, role)]
        used = [r for r in assign if r != "none"]
        for i in range(len(assign)):
            for j in range(i + 1, len(assign)):
                if assign[i] != "none" and assign[j] != "none": f.append(("o", kind, assign[i], assign[j]))
        for r in ROLES.get(kind, ()):
            if r not in used: f.append(("miss", kind, r))
        return f

    def _score(self, feats): return sum(self.w[x] for x in feats)

    @staticmethod
    def assignments(n, kind):
        roles = ROLES.get(kind, ()) + ("none",)
        for combo in itertools.product(roles, repeat=n):
            used = [r for r in combo if r != "none"]
            if len(used) == len(set(used)): yield combo

    _assign_cache: Dict[Tuple[int, str], Any] = {}

    def _assignment_table(self, n, kind):
        key = (n, kind)
        if key not in JointGrammarInducer._assign_cache:
            roles = ROLES.get(kind, ()) + ("none",)
            combos = list(self.assignments(n, kind))
            idx = torch.tensor([[roles.index(r) for r in c] for c in combos], dtype=torch.long).view(len(combos), n)
            req = torch.tensor([all(r in c for r in REQUIRED.get(kind, ())) for c in combos])
            JointGrammarInducer._assign_cache[key] = (combos, idx, req, roles)
        return JointGrammarInducer._assign_cache[key]

    def decode(self, tokens, nps, kind, k=1, hard=True):
        """Vectorised constraint satisfaction: score(a) = Σ unary[i, a_i] + Σ_{i<j} pair[a_i, a_j] + Σ missing.
        Parses violating a hard constraint (required role absent) only come last."""
        combos, idx, req, roles = self._assignment_table(len(nps), kind)
        R = len(roles); none = R - 1
        U = torch.zeros(len(nps), R)
        for i, np in enumerate(nps):
            for j, r in enumerate(roles): U[i, j] = self._score(self.feats_unary(tokens, nps, kind, i, np, r))
        P = torch.zeros(R, R); M = torch.zeros(R)
        for a, ra in enumerate(roles[:-1]):
            M[a] = self.w[("miss", kind, ra)]
            for b, rb in enumerate(roles[:-1]): P[a, b] = self.w[("o", kind, ra, rb)]
        n = len(nps)
        score = U[torch.arange(n), idx].sum(1) if n else torch.zeros(len(combos))
        for i in range(n):
            for j in range(i + 1, n): score = score + P[idx[:, i], idx[:, j]]
        for a in range(R - 1): score = score + M[a] * (~(idx == a).any(1)).float()
        self.explored += len(combos)
        order = torch.argsort(score + (1e6 * req.float() if hard else 0), descending=True)
        top = order if not k else order[:k]
        return [(float(score[i]), combos[i]) for i in top.tolist()]

    def feats_unary(self, tokens, nps, kind, i, np, role):
        prev = tokens[np.start - 1] if np.start > 0 else "<s>"
        prev2 = tokens[np.start - 2] if np.start > 1 else "<s>"
        nxt = tokens[np.end] if np.end < len(tokens) else "</s>"
        return [("p", kind, prev, role), ("p2", kind, prev2, role), ("n", kind, nxt, role),
                ("pc", kind, token_class(self.lex.meaning(prev)), role), ("r", kind, min(i, 3), role),
                ("nk", kind, self.np_kind(np), role), ("b", kind, role)]

    def train_joint(self, epochs=8):
        data = []
        for tokens, s in self._data:
            nps = self.chunks(tokens)
            if not nps or len(nps) > 6 or s["kind"] not in ROLES: continue
            g = self.gold(tokens, nps, s)
            if g is not None: data.append((tokens, nps, s["kind"], g))
        for _ in range(epochs):
            for tokens, nps, kind, g in data:
                pred = self.decode(tokens, nps, kind, 1, hard=False)[0][1]
                if pred != g:
                    for x in self.feats(tokens, nps, kind, g): self.w[x] += 1; self._u[x] += self._c
                    for x in self.feats(tokens, nps, kind, pred): self.w[x] -= 1; self._u[x] -= self._c
                self._c += 1
        for x in list(self.w): self.w[x] -= self._u[x] / self._c          # perceptron averaging
        self.explored = 0
        return len(data)

    def satisfies(self, kind, assign):
        """Hard constraints verified by WCE (required roles present, distinct roles)."""
        used = [r for r in assign if r != "none"]
        return len(used) == len(set(used)) and all(r in used for r in REQUIRED.get(kind, ()))

    def assign(self, tokens, nps, kind):
        if not self.joint or not nps or len(nps) > 6: return super().assign(tokens, nps, kind)
        k = "event" if kind == "command" else kind
        if self.student is not None and self.rank == 0:
            a = self.student.predict(tokens, nps, k)
            self.explored += 1
            if a is not None and self.satisfies(k, a) and all(r in ROLES.get(k, ()) + ("none",) for r in a):
                self.last_margin = min(self.last_margin, self.student.last_margin)
                return {i: r for i, r in enumerate(a) if r != "none" and r in ROLES[kind]}
        ranked = self.decode(tokens, nps, k, 0)
        if len(ranked) > 1: self.last_margin = min(self.last_margin, ranked[0][0] - ranked[1][0])
        best = ranked[min(self.rank, len(ranked) - 1)][1]
        return {i: r for i, r in enumerate(best) if r != "none" and r in ROLES[kind]}


class RoleStudent(torch.nn.Module):
    """Compact student distilled from the joint perceptron: role per phrase (previous word, next word, rank, type)."""

    def __init__(self, vocab, roles, dim=16, hidden=32, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.vocab = {w: i + 1 for i, w in enumerate(sorted(vocab))}
        self.roles = list(roles)
        self.emb = torch.nn.Embedding(len(self.vocab) + 1, dim)
        self.kind_emb = torch.nn.Embedding(8, 4)
        self.net = torch.nn.Sequential(torch.nn.Linear(2 * dim + 4 + 4 + 4, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, len(self.roles)))
        self.kinds = {k: i for i, k in enumerate(sorted(ROLES))}
        self.last_margin = 0.0

    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())

    def _x(self, tokens, nps, kind):
        rows = []
        for i, np in enumerate(nps):
            prev = tokens[np.start - 1] if np.start > 0 else "<s>"
            nxt = tokens[np.end] if np.end < len(tokens) else "</s>"
            rank = torch.zeros(4); rank[min(i, 3)] = 1
            nk = torch.zeros(4); nk[["pron", "generic", "person", "type"].index(JointGrammarInducer.np_kind(np))] = 1
            rows.append((self.vocab.get(prev, 0), self.vocab.get(nxt, 0), self.kinds.get(kind, 0), rank, nk))
        return rows

    def logits(self, tokens, nps, kind):
        rows = self._x(tokens, nps, kind)
        p = torch.tensor([r[0] for r in rows]); n = torch.tensor([r[1] for r in rows]); k = torch.tensor([r[2] for r in rows])
        x = torch.cat([self.emb(p), self.emb(n), self.kind_emb(k), torch.stack([r[3] for r in rows]), torch.stack([r[4] for r in rows])], 1)
        return self.net(x)

    def fit(self, examples, epochs=150):
        """A single batch per epoch (all sentences concatenated)."""
        rows = [r for t, nps, k, a in examples for r in self._x(t, nps, k)]
        targets = torch.tensor([self.roles.index(r) for t, nps, k, a in examples for r in a])
        p = torch.tensor([r[0] for r in rows]); n = torch.tensor([r[1] for r in rows]); kk = torch.tensor([r[2] for r in rows])
        rank = torch.stack([r[3] for r in rows]); nk = torch.stack([r[4] for r in rows])
        opt = torch.optim.Adam(self.parameters(), lr=0.02)
        for _ in range(epochs):
            opt.zero_grad()
            x = torch.cat([self.emb(p), self.emb(n), self.kind_emb(kk), rank, nk], 1)
            torch.nn.functional.cross_entropy(self.net(x), targets).backward(); opt.step()
        return self

    def predict(self, tokens, nps, kind):
        """Greedy assignment (distinct roles); None if the role of a phrase is outside the sentence type."""
        with torch.no_grad(): probs = torch.softmax(self.logits(tokens, nps, kind), 1)
        allowed = set(ROLES.get(kind, ())) | {"none"}
        order = sorted(range(len(nps)), key=lambda i: -float(probs[i].max()))
        out, used = [None] * len(nps), set()
        margins = []
        for i in order:
            ranked = sorted(((float(p), self.roles[j]) for j, p in enumerate(probs[i])), reverse=True)
            ranked = [(p, r) for p, r in ranked if r in allowed and (r == "none" or r not in used)]
            if not ranked: return None
            out[i] = ranked[0][1]; used.add(out[i])
            margins.append(ranked[0][0] - (ranked[1][0] if len(ranked) > 1 else 0))
        self.last_margin = min(margins) * 10 if margins else 0.0
        return tuple(out)


# ================================================================ evidence-grounded lexicon

@dataclass
class LexicalHypothesis:
    surface: str
    support: Counter = field(default_factory=Counter)
    refute: Counter = field(default_factory=Counter)
    experiences: Dict[str, set] = field(default_factory=lambda: defaultdict(set))
    sources: Dict[str, set] = field(default_factory=lambda: defaultdict(set))
    history: List[str] = field(default_factory=list)

    def ranked(self):
        return sorted(((self.support[c] - self.refute[c], c) for c in set(self.support) | set(self.refute)), reverse=True)

    @property
    def best(self):
        r = self.ranked()
        return r[0][1] if r and self.support[r[0][1]] > 0 else None

    @property
    def candidate_concepts(self): return {c: round(self.support[c] - self.refute[c], 2) for c in set(self.support) | set(self.refute)}

    @property
    def confidence(self):
        b = self.best
        if b is None: return 0.0
        s = self.support[b]; rest = sum(v for c, v in self.support.items() if c != b)
        return s / (s + rest + self.refute[b] + 1.0)

    @property
    def status(self):
        """UNKNOWN -> HYPOTHESIS -> BELIEF -> FACT_LIKE according to quantity, margin and diversity; regression possible."""
        b = self.best
        if b is None: return LEX_UNKNOWN
        s, r = self.support[b], self.refute[b]
        second = max((self.support[c] for c in self.support if c != b), default=0.0)
        n_exp, n_src = len(self.experiences[b]), len(self.sources[b])
        if s >= 4.0 and n_src >= 2 and n_exp >= 3 and r <= 0.1 * s and s - second >= 2.0: return LEX_FACT
        if s >= 2.0 and n_exp >= 2 and r <= 0.25 * s and s - second >= 1.0: return LEX_BELIEF
        return LEX_HYPOTHESIS


class LexicalMemory:
    def __init__(self): self.hyps: Dict[str, LexicalHypothesis] = {}

    def get(self, w): return self.hyps.get(w)

    def observe(self, word, results: Dict[str, str], source, weight, experience):
        h = self.hyps.setdefault(word, LexicalHypothesis(word))
        supports = [c for c, v in results.items() if v == "support"]
        for c in supports:
            h.support[c] += weight / len(supports); h.experiences[c].add(experience); h.sources[c].add(source)
        for c, v in results.items():
            if v == "refute": h.refute[c] += weight
        before = h.history[-1] if h.history else LEX_UNKNOWN
        if h.status != before: h.history.append(h.status)
        return h

    def meaning(self, word, min_status=LEX_BELIEF):
        h = self.hyps.get(word)
        order = [LEX_UNKNOWN, LEX_HYPOTHESIS, LEX_BELIEF, LEX_FACT]
        return h.best if h and order.index(h.status) >= order.index(min_status) else None


@dataclass
class LearningNeed:
    kind: str                    # lexical | evidence | reference | grammar
    target: str
    reason: str
    suggestion: str
    hypothesis: Dict[str, float] = field(default_factory=dict)


# ================================================================ grounded agent

@dataclass
class GroundedConfig(AgentConfig):
    joint_grammar: bool = True
    grounded_lexicon: bool = True
    polysemy: bool = True
    distilled_roles: bool = False
    budget: int = 99                   # hypotheses (lexical / parse) verifiable per query
    grammar_margin: float = 1.0        # below this margin, an alternative parse is tested
    ground_on_sensors: bool = False    # check candidate meanings on sensor evidence only (anti-circularity)
    elimination: bool = False          # disabled by default (8 seeds: slightly harmful); reference by elimination at ingestion (separate evidence channel)
    neural: bool = False
    geometry: bool = False
    verification: str = "selective"


class GroundedAgent(IndependentAgent):
    def __init__(self, config: GroundedConfig = None, seed=0):
        cfg = config or GroundedConfig()
        super().__init__(cfg, seed)
        self.lex = GroundedLexicon()
        self.grammar = JointGrammarInducer(self.lex, joint=cfg.joint_grammar)
        self.lexmem = LexicalMemory()
        self.sense_ctx = defaultdict(Counter)
        self.contains = defaultdict(Counter)
        self.learning_needs: List[LearningNeed] = []
        self.parse_log: List[Dict[str, Any]] = []
        self._ctx: List[str] = []
        self._world_id = 0
        self._forced: Dict[str, str] = {}
        self._eliminated = False

    # ---- learning ------------------------------------------------------------
    def learn_captions(self, caps):
        for c in caps: self.lex.observe(tokenize(c["text"]), c["structure"])
        self.lex.align()
        self.lex.tok.clear(); self.lex.sym.clear(); self.lex.co.clear(); self.lex.units.clear()
        super().learn_captions(caps)
        self.lex.align()
        for c in caps:
            s = c["structure"]; toks = tokenize(c["text"])
            if s["kind"] == "event" and s.get("patient"):
                for t in toks:
                    if f"act:{s['action']}" in self.lex.co[t]: self.sense_ctx[(t, f"act:{s['action']}")][s["patient"].get("type")] += 1
            if s.get("rel") == "in":
                sub = s.get("subject") or s.get("patient"); tgt = s.get("target")
                if sub and tgt: self.contains[tgt.get("type")][sub.get("type")] += 1
        if self.cfg.joint_grammar: self.grammar.train_joint()
        if self.cfg.distilled_roles: self.distill_roles(caps)
        return self

    def distill_roles(self, caps):
        """Compact student trained on the decisions of the joint perceptron (WCE teacher)."""
        examples, vocab = [], set()
        for c in caps:
            toks = tokenize(c["text"]); s = c["structure"]
            if s["kind"] not in ROLES or s["kind"] == "sequence": continue
            nps = self.grammar.chunks(toks)
            if not nps or len(nps) > 6: continue
            a = self.grammar.decode(toks, nps, s["kind"], 1)[0][1]
            examples.append((toks, nps, s["kind"], a)); vocab |= set(toks)
        roles = sorted({r for k in ROLES.values() for r in k} | {"none"})
        self.grammar.student = RoleStudent(vocab, roles, seed=self.seed).fit(examples)
        self.grammar.explored = 0
        return len(examples)

    # ---- word meanings: lexicon -> senses -> evidence -> composition -> morphology ----
    def senses(self, tok):
        if self.lex.tok[tok] < 3: return []
        out = [(c / self.lex.tok[tok], s) for s, c in self.lex.co[tok].items()
               if s.startswith("act:") and c / self.lex.tok[tok] >= 0.3 and c >= 3]
        return [s for _, s in sorted(out, reverse=True)]

    def _choose_sense(self, tok, senses):
        """Polysemy: the sense is chosen by the type of the patient (learned from the captions), not by the word alone."""
        if tok in self._ctx:
            i = self._ctx.index(tok)
            for t in self._ctx[i + 1:]:
                m = self.lex.meaning(t) or self.lexmem.meaning(t)
                if m and m.startswith("type:"):
                    ptype = m.split(":")[1]
                    scored = [(self.sense_ctx[(tok, s)][ptype], s) for s in senses]
                    best = max(scored)
                    if best[0] > 0: return best[1]
                    # fallback: sense whose observed patients are the closest (same container possible)
                    break
        return senses[0]

    def word_meaning(self, tok):
        if tok in self._forced: return self._forced[tok], "forced", 1.0
        if tok in INNATE_QUESTION or tok in INNATE_PRONOUN or tok in INNATE_OTHER or is_number(tok): return None, "innate", 1.0
        if self.lex.seen(tok):
            m = self.lex.meaning(tok)
            if self.cfg.polysemy:
                senses = self.senses(tok)
                if len(senses) >= 2: return self._choose_sense(tok, senses), "polysemy", 1.0
            return m, "grounded" if m else "function", 1.0
        if self.cfg.grounded_lexicon:
            m = self.lexmem.meaning(tok)
            if m: return m, "grounded_obs", self.lexmem.get(tok).confidence
        if tok.startswith("re") and len(tok) > 4:                       # composition: re- + known verb
            inner = self.lex.meaning(tok[2:]) or (self.senses(tok[2:]) or [None])[0]
            if inner and inner.startswith("act:"): return inner, "composition", 0.9
        cands = self.morph.candidates(tok, self.lex.lexicon())
        if cands:
            ref, meaning, rule, support = cands[0]
            h = self.lexmem.get(tok)
            refuted = h is not None and h.refute[meaning] > h.support[meaning]
            if support >= 2 and not refuted: return meaning, "morphology", 0.95
        return None, "unknown", 0.0

    def resolve(self, np, belief=None, discourse=None, strict=True):
        """Elimination (ingestion only): if no known object matches and a single candidate has an
        unknown colour, the others being established and different, the asserted description designates it.
        Refused for queries: a question may carry a false presupposition."""
        objs, unknown = super().resolve(np, belief, discourse, strict)
        if self.cfg.elimination and not strict and np.color and not objs and len(unknown) == 1 and not np.pronoun:
            self._eliminated = True
            return unknown, []
        return objs, unknown

    def _is_unknown(self, tok):
        return self.word_meaning(tok)[1] == "unknown" and tok.isalpha()

    def _affordance(self, kinds, patient, rel, target):
        """Affordances of "dans" learned from the captions: a non-physical (learned) container contains
        only the types observed in it, and these types go nowhere else."""
        pk, tk = kinds.get(patient), kinds.get(target, target)
        learned = {t for t in self.contains if t not in SUPPORTS and t != "box"}
        inventory_types = {p for t in learned for p in self.contains[t]}
        if tk in learned: return rel == "in" and pk in inventory_types
        if pk in inventory_types or pk in learned: return False
        return affordance(kinds, patient, rel, target)

    def _resolve_oov(self, tokens, meanings, i, ans, mode):
        if self.cfg.budget < 20: return None                  # exhaustive contextual inference = costly
        return super()._resolve_oov(tokens, meanings, i, ans, mode)

    # ---- multi-pass ingestion + lexical grounding ------------------------------
    def ingest(self, observations, update=True):
        belief = self.belief or self.new_world()
        self._world_id += 1
        texts = []
        for ob in observations:
            key = ob.key
            if key[0] == "type": belief.inventory[key[1]] = ob.value; continue
            if ob.text is None:
                if ob.implied: continue
                if key[0] == "event": key, value = ("event", ob.value.patient), (ob.value.rel, ob.value.target)
                else: value = ob.value
                if key[0] == "loc": key = ("loc", key[1], "now" if key[2] != 0 else 0)
                belief.graph.add(key, Evidence(ob.source, value, ob.time, "structured"))
            else:
                texts.append(ob)
        done = set()
        for _ in range(2):                                  # passes: new beliefs unlock references
            for j, ob in enumerate(texts):
                if j in done: continue
                toks = tokenize(ob.text)
                if self.cfg.grounded_lexicon and any(self._is_unknown(t) for t in toks): continue
                self._eliminated = False
                self._cur_source = ob.source
                props = self.parse_statement(toks)
                if props:
                    done.add(j)
                    src = ob.source + ("|elim" if self._eliminated else "")
                    for key, value in props: belief.graph.add(key, Evidence(src, value, ob.time, "text", ob.text))
        if self.cfg.grounded_lexicon:
            for j, ob in enumerate(texts):
                if j in done: continue
                toks = tokenize(ob.text)
                if any(self._is_unknown(t) for t in toks):
                    self._ground(toks, ob)
                    if not any(self._is_unknown(t) for t in toks):
                        props = self.parse_statement(toks)
                        if props:
                            done.add(j)
                            for key, value in props: belief.graph.add(key, Evidence(ob.source, value, ob.time, "text", ob.text))
            self._morphology_from_grounding()
        for j, ob in enumerate(texts):
            self.parse_log.append({"text": ob.text, "key": ob.key, "value": ob.value, "source": ob.source,
                                   "parsed": j in done, "props": self.parse_statement(tokenize(ob.text)) if j in done else None})
        if self.cfg.evidence_graph and update: belief.graph.update_reliability()
        return belief

    def _ground(self, toks, ob):
        """Cross-situational grounding: which meanings make the sentence consistent with the evidence from the OTHER sources?"""
        unknown = [i for i, t in enumerate(toks) if self._is_unknown(t)]
        if len(unknown) != 1: return None
        i = unknown[0]; word = toks[i]
        weight = max(0.0, 2 * self.reliability.estimate(ob.source) - 1)
        candidates = sorted({m for m in self.lex.lexicon().values() if token_class(m) in ("N", "C", "R")})
        results = {}
        for sym in candidates:
            self._forced[word] = sym
            try: results[sym] = self._check_against_evidence(toks, i, sym, ob.source)
            finally: self._forced.pop(word, None)
        return self.lexmem.observe(word, results, ob.source, weight, f"w{self._world_id}:{ob.text}")

    def _check_against_evidence(self, toks, i, sym, source):
        if token_class(sym) == "N":
            nps = self.grammar.chunks(toks, [self.word_meaning(t)[0] for t in toks])
            np = next((n for n in nps if n.start <= i < n.end), None)
            if np is not None and np.type not in SUPPORTS and not self.belief.objects_of(np.type): return "refute"
        props = self.parse_statement(toks)
        if not props: return "neutral"
        verdicts = []
        for key, value in props:
            if key[0] == "color" and len(props) > 1: continue
            b = self.belief.graph.belief(key, exclude=source, via="structured" if self.cfg.ground_on_sensors else None)
            if b.status in (FACT, BELIEF): verdicts.append("support" if b.value == value else "refute")
        if "refute" in verdicts: return "refute"
        return "support" if verdicts else "neutral"

    def _morphology_from_grounding(self):
        """Morphology + grounding: a family is induced only if the forms are grounded to the same concept."""
        for w, h in self.lexmem.hyps.items():
            if h.status in (LEX_BELIEF, LEX_FACT):
                for form, m in self.lex.lexicon().items():
                    if m == h.best and form != w:
                        if len(w) > len(form): self.morph.learn_pair(w, form)

    # ---- counterfactual parsing of observations ------------------------------------------
    def parse_statement(self, tokens):
        props = self._parse_statement(tokens)
        meanings = [self.word_meaning(t)[0] for t in tokens]
        syms = {m for m in meanings if m}
        nps = self.grammar.chunks(tokens, meanings)
        if self._kind(syms, tokens) == "color" and len(nps) == 1 and nps[0].color and nps[0].type:
            standalone = [m for i, m in enumerate(meanings) if m and m.startswith("color:") and not (nps[0].start <= i < nps[0].end)]
            if not standalone:                                  # "c'est une clé bleue": the attribute is the asserted value
                objs, unknown = self.resolve(NP(0, 0, type=nps[0].type), strict=False)
                if len(objs) == 1 and not unknown: return [(("color", objs[0]), nps[0].color)]
                return None
        return props

    def _parse_statement(self, tokens):
        if not self.cfg.joint_grammar: return super().parse_statement(tokens)
        hyps = []
        for rank in range(3):
            self.grammar.rank = rank
            self.grammar.last_margin = math.inf
            try: props = super().parse_statement(tokens)
            finally: self.grammar.rank = 0
            if props: hyps.append((rank, props))
            if rank == 0 and self.grammar.last_margin >= self.cfg.grammar_margin: break
        if not hyps: return None
        distinct = {tuple(sorted(map(str, p))) for _, p in hyps}
        if len(distinct) == 1: return hyps[0][1]
        # diverging and close parses: consistency with the evidence breaks the tie, otherwise nothing is ingested
        scored = []
        for rank, props in hyps:
            s = 0
            for key, value in props:
                b = self.belief.graph.belief(key)
                if b.status in (FACT, BELIEF): s += 1 if b.value == value else -1
            scored.append((s, -rank, props))
        scored.sort(reverse=True)
        return scored[0][2]              # on an evidence tie: the grammatical preference (rank 0) wins

    # ---- answers ------------------------------------------------------------------------
    def answer(self, text, mode=None):
        import re as _re
        last = [s for s in _re.split(r"(?<=[.])\s+", text.strip()) if s][-1]
        self._ctx = tokenize(last)
        if " et " in f" {last} " and "?" not in last and not self._ctx[0] in ("regarde", "si"):
            return self._conjunction(text, last)
        self.grammar.last_margin = math.inf
        self.grammar.explored = 0
        ans = IndependentAgent.answer(self, text, mode)
        ans.checks += self.grammar.explored
        return self._post(ans, text, mode)

    def _conjunction(self, text, last):
        """"A et B" / "A, B et C": each clause is verified; the conjunction combines the verdicts."""
        parts = [p.strip() for p in _split_clauses(last)]
        results = [self.answer(p) for p in parts]
        outs = [r.output for r in results]
        if any(o == "REFUTED" for o in outs): out = "REFUTED"
        elif all(o == "SUPPORTED" for o in outs): out = "SUPPORTED"
        elif "ASK" in outs: out = "ASK"
        else: out = "UNKNOWN"
        ans = results[0]
        ans.text, ans.output, ans.intent = text, out, "claim"
        ans.decision = "ABSTAIN" if out in ABSTAIN else "ACCEPT"
        ans.checks = sum(r.checks for r in results)
        ans.reasoning = [f"clause {i + 1}: {o}" for i, o in enumerate(outs)]
        ans.features["clauses"] = len(parts)
        return ans

    def _post(self, ans, text, mode):
        ans.initial_output = ans.output
        # counter-parse: if the best grammatical parse is not clearly better, the next one is tested
        if self.cfg.joint_grammar and ans.output not in ABSTAIN and self.grammar.last_margin < self.cfg.grammar_margin \
                and self.cfg.budget >= 2:
            self.grammar.rank = 1
            try: alt = IndependentAgent.answer(self, text, mode)
            finally: self.grammar.rank = 0
            ans.checks += 1
            if alt.output != ans.output and alt.output not in ABSTAIN:
                ans.output, ans.decision = "UNKNOWN", "ABSTAIN"
                ans.flags.append("grammar_ambiguity"); ans.corrected = True
                ans.reasoning.append(f"analyses concurrentes : {ans.initial_output} / {alt.output} -> abstention")
        if self.cfg.correction and ans.output == "UNKNOWN_WORD": self._evidence_correction(ans, text, mode)
        ans.abstain_reason = self.abstention_reason(ans) if ans.output in ABSTAIN else None
        if ans.abstain_reason: self._learning_need(ans)
        return ans

    def _evidence_correction(self, ans, text, mode):
        """Self-correction through the evidence graph: an unknown word whose lexical hypothesis is already supported
        (without refutation) is tried; the answer is kept only if it rests on an established belief."""
        words = [t for t in self._ctx if self._is_unknown(t)]
        if len(words) != 1: return
        h = self.lexmem.get(words[0])
        if not h or h.best is None or h.refute[h.best] > 0: return
        second = max((h.support[c] for c in h.support if c != h.best), default=0.0)
        # strict requirement: at least two independent experiences, clear support, no refutation
        if h.support[h.best] < 1.5 or len(h.experiences[h.best]) < 2 or h.support[h.best] - second < 1.0: return
        self._forced[words[0]] = h.best
        try: retry = IndependentAgent.answer(self, text, mode)
        finally: self._forced.pop(words[0], None)
        grounded = any(e.status in (FACT, BELIEF) for e in retry.entries) or retry.intent == "command"
        ans.checks += retry.checks + 1
        if retry.output not in ABSTAIN and grounded:
            ans.output, ans.decision, ans.corrected = retry.output, retry.decision, True
            ans.reasoning += [f"auto-correction : « {words[0]} » ≈ {h.best} (hypothèse ancrée, support {h.support[h.best]:.2f})"] + retry.reasoning
            ans.evidence += retry.evidence; ans.sources += retry.sources

    # ---- explained abstentions and learning needs ----------------------------------
    def abstention_reason(self, ans):
        if ans.output == "UNKNOWN_WORD": return "lexical_unknown"
        if ans.output == "ASK" or "grammar_ambiguity" in ans.flags: return "grammar_ambiguity"
        if "unknown_entity" in ans.flags: return "unknown_entity"
        if "planner_failure" in ans.flags: return "planner_failure"
        if "unsupported_inference" in ans.flags: return "unsupported_inference"
        if ans.entries:
            e = ans.entries[-1]
            if e.refuting: return "conflicting_evidence"
            if e.supporting and all(self.reliability.estimate(s) < 0.75 for s in e.supporting): return "low_source_reliability"
        return "insufficient_evidence"

    def _learning_need(self, ans):
        reason = ans.abstain_reason
        if reason == "lexical_unknown":
            for w in [t for t in self._ctx if self._is_unknown(t)]:
                h = self.lexmem.get(w)
                self.learning_needs.append(LearningNeed("lexical", w, reason,
                    f"observer « {w} » dans des phrases alignées sur des faits capteurs (source fiable, plusieurs situations)",
                    h.candidate_concepts if h else {}))
        elif reason in ("insufficient_evidence", "low_source_reliability", "conflicting_evidence"):
            key = ans.entries[-1].key if ans.entries else None
            self.learning_needs.append(LearningNeed("evidence", str(key), reason, "observation par une source fiable (capteur)"))
        elif reason in ("grammar_ambiguity", "unknown_entity"):
            self.learning_needs.append(LearningNeed("reference" if reason == "unknown_entity" else "grammar", ans.text, reason,
                                                    "désambiguïsation (couleur, relation ou exemple aligné)"))


def _split_clauses(text):
    parts = []
    for chunk in text.split(" et "):
        parts += [p for p in chunk.split(", ") if p]
    return parts

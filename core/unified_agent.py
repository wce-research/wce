"""Unified agent for the independent environment: a single identity function (ingestion AND questions),
calibrated by source reliability, evidence weighted by identity, reversible propagation.

    P(o | description d, context) = (1 − ε) · [ g · p_joint(o) + (1 − g) / n ]
p_joint: marginal of the injective matchings descriptions -> objects (vectorised, DiscourseAgent);
g: probability that the description is genuine — at ingestion, reliability of the speaking source;
    at question time, 1 − Π(1 − r_s) over the sources that used this description (none -> no identity);
n: number of objects of the type; ε: prior of false presupposition.
Thresholds: observation (0.67 by default) < question (0.95); adjustable on the CALIBRATION split.

Propagation: a description « le cube rouge » attached to o becomes evidence ("color", o) = rouge from the same
source, with weight P (the evidence depends on the identity). If the identity changes after recomputation, the evidence is retracted.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from core.discourse_agent import DiscourseAgent, DiscourseConfig
from core.grounded import GroundedAgent
from core.independent_agent import Evidence, tokenize
from core.unified import IsotonicCalibrator, calibrate_threshold, calibration_report


@dataclass
class UnifiedConfig(DiscourseConfig):
    observation_threshold: float = 0.67
    query_threshold: float = 0.95
    presupposition_prior: float = 0.02
    propagate_descriptions: bool = True
    rollback: bool = True
    source_calibrated: bool = True
    calibrate: bool = False               # thresholds + calibrator fitted on the calibration split
    query_identity: bool = True           # coreference at question time (same function, strict threshold)
    identity_mode: str = "l2"             # l2 (probabilistic joint, old) | l1 (explicit elimination only) | none
    l1_min_premise: float = 0.95          # elimination premises: colours established as FACT (calibrated posterior)
    l1_min_source: float = 0.8            # minimum effective reliability of the describing source (ingestion)
    l1_min_attestation: float = 0.9       # question: presupposed colour attested (combination of the sources) ≥ 0.9


class UnifiedAgent(DiscourseAgent):
    duplicate_prior = 0.25            # an undescribed object may share a described colour (frequent duplicates)

    def __init__(self, config: UnifiedConfig = None, seed=0):
        super().__init__(config or UnifiedConfig(), seed)
        self._desc_sources = defaultdict(lambda: defaultdict(set))
        self._cur_weight = 1.0
        self._cur_identity = None
        self._cur_rule = None
        self.dependencies = []           # (key, evidence, (type, colour, object)): evidence that depends on an identity
        self.identity_points = []        # (P, context, type, colour, object) for calibration / evaluation
        self.rollbacks = 0
        self.calibrator = None
        self.calibration = {}
        self.rule_of = {}                # id(evidence) -> EliminateCandidates (explicit rule, audited)
        self.rules = []

    def warm_up_sources(self, train_worlds):
        """Reliabilities estimated on DIRECT evidence only: no evidence depending on an inferred identity."""
        saved = self.calibration
        self.calibration = {**saved, "observation_threshold": 1.01, "query_threshold": 1.01}
        try: return super().warm_up_sources(train_worlds)
        finally: self.calibration = saved; self.identity_log.clear(); self.identity_points.clear()

    # ---- single identity ----------------------------------------------------------------
    def identity_prob(self, typ, color, context, source=None):
        if self.cfg.identity_mode == "none": return None, 0.0
        if self.cfg.identity_mode == "l1": return self._l1_identity(typ, color, context, source)
        hit = self._mapping.get((typ, color))
        if hit is None: return None, 0.0
        o, p = hit
        n = max(1, len(self.belief.objects_of(typ)))
        if not self.cfg.source_calibrated: g = 1.0
        elif context == "observation" and source is not None: g = min(self.reliability.estimate(source.split("|")[0]), 0.99)
        else:
            srcs = self._desc_sources.get(typ, {}).get(color, set())
            if not srcs: return o, 0.0
            miss = 1.0
            for s in srcs: miss *= 1 - min(self.reliability.estimate(s), 0.99)
            g = 1 - miss
        P = (1 - self.cfg.presupposition_prior) * (g * p + (1 - g) / n)
        if self.calibrator is not None: P = float(self.calibrator(P))
        return o, P

    def _l1_identity(self, typ, color, context, source=None):
        """Identity ONLY by explicit elimination on calibrated premises (level ≤ L1); no probabilistic
        matching (L2). At ingestion, the describing source must be sufficiently reliable."""
        rule = self.explicit_elimination(typ, color, min_posterior=self.cfg.l1_min_premise)
        if not rule.conclusive or rule.level > 1: return None, 0.0
        g = 1.0
        if context == "observation":
            if source is None: return None, 0.0
            g = self.reliability.estimate(source.split("|")[0])
            if g < self.cfg.l1_min_source: return None, 0.0
        else:
            # question: the presupposition (colour of the remaining candidate) must be ATTESTED by the observations;
            # otherwise a question with a false presupposition (a non-existent « le ballon vert ») would be "resolved"
            from core.independent_agent import BELIEF, FACT
            b = self.belief.color(rule.remaining)
            if not (b.status in (FACT, BELIEF) and b.value == color):
                miss = 1.0
                for s_ in self._desc_sources.get(typ, {}).get(color, set()): miss *= 1 - self.reliability.estimate(s_)
                g = 1 - miss
                if g < self.cfg.l1_min_attestation: return None, 0.0
        return rule.remaining, (1 - self.cfg.presupposition_prior) * rule.confidence * g

    def threshold(self, context):
        key = "observation_threshold" if context == "observation" else "query_threshold"
        return self.calibration.get(key, getattr(self.cfg, key))

    # ---- ingestion ------------------------------------------------------------------------
    def ingest(self, observations, update=True):
        self._desc_sources = defaultdict(lambda: defaultdict(set))
        self.dependencies = []; self.rules = []; self.rule_of = {}
        for ob in observations:
            if not ob.text: continue
            toks = tokenize(ob.text)
            for np in self.grammar.chunks(toks, [self.word_meaning(t)[0] for t in toks]):
                if np.type and np.color: self._desc_sources[np.type][np.color].add(ob.source)
        belief = self.belief or self.new_world()
        g = belief.graph
        agent = self

        def add(key, ev, _orig=type(g).add):
            if agent._eliminated and agent._cur_identity is not None:
                ev.weight = agent._cur_weight
                agent.dependencies.append((key, ev, agent._cur_identity))
                rule = agent._cur_rule
                if rule is not None and rule.conclusive: agent.rule_of[id(ev)] = rule
            _orig(g, key, ev)
        g.add = add
        try: out = super().ingest(observations, update)
        finally: del g.add
        if self.cfg.joint_reference: self.solve_identity()
        if self.cfg.propagate_descriptions: self._propagate()
        if self.cfg.rollback: self._revise()
        return out

    def _propagate(self):
        """Description -> probable entity -> attached property (same source, weight = P(identity))."""
        g = self.belief.graph
        for typ, colors in self._desc_sources.items():
            for color, srcs in colors.items():
                for s in srcs:
                    o, P = self.identity_prob(typ, color, "observation", s)
                    if o is None or P < self.threshold("observation"): continue
                    self.identity_points.append((P, "observation", typ, color, o))
                    if s in g.votes(("color", o)): continue
                    ev = Evidence(s, color, 0, "description", None, P)
                    g.add(("color", o), ev)
                    rule = self.explicit_elimination(typ, color)
                    if rule.conclusive and rule.remaining == o: self.rule_of[id(ev)] = rule
                    self.dependencies.append((("color", o), ev, (typ, color, o)))

    def _revise(self):
        """Recomputation with all the evidence: evidence whose identity no longer holds is retracted (backtracking)."""
        g = self.belief.graph
        for _ in range(2):
            self.solve_identity()
            removed = 0
            for key, ev, (typ, color, o) in list(self.dependencies):
                new_o, P = self.identity_prob(typ, color, "observation", ev.source)
                if new_o != o or P < self.threshold("observation"):
                    if ev in g.props.get(key, []): g.props[key].remove(ev); removed += 1
                    self.dependencies.remove((key, ev, (typ, color, o)))
            self.rollbacks += removed
            if not removed: break

    def explicit_elimination(self, typ, color, min_posterior=0.0):
        """Explicit elimination: candidates of the type, excluded = different ESTABLISHED colour; conclusion if only one remains.
        Premises: FACT/BELIEF beliefs (≥ min_posterior) that do NOT themselves depend on an inferred identity."""
        from core.independent_agent import BELIEF, FACT
        from core.knowledge_boundary import eliminate
        cands = self.belief.objects_of(typ)
        known, post = {}, {}
        for o in cands:
            b = self.belief.color(o)
            votes = self.belief.graph.votes(("color", o)).values()
            derived = any(ev.via == "description" or "|elim" in ev.source or getattr(ev, "weight", 1.0) < 1.0 for ev in votes if ev.value == b.value)
            if b.status in (FACT, BELIEF) and b.posterior >= min_posterior and not derived:
                known[o] = b.value; post[o] = b.posterior
        rule = eliminate((typ, color), cands, known, post, "agent")
        self.rules.append(rule)
        return rule

    # ---- resolution: same function, threshold by context ----------------------------------
    def resolve(self, np, belief=None, discourse=None, strict=True):
        objs, unknown = GroundedAgent.resolve(self, np, belief, discourse, strict)
        self._cur_identity = None
        if not self.cfg.joint_reference or not np.color or not np.type or np.pronoun: return objs, unknown
        if len(objs) == 1 and not unknown: return objs, unknown
        if getattr(self, "_pending_solve", False) and self.belief is not None:
            self.solve_identity(); self._pending_solve = False
        context = "query" if strict else "observation"
        if strict and not self.cfg.query_identity: return objs, unknown
        source = getattr(self, "_cur_source", None)
        o, P = self.identity_prob(np.type, np.color, context, source)
        if o is None or P < self.threshold(context): return objs, unknown
        if strict and (objs and o not in objs): return objs, unknown          # never against an established belief
        self.identity_points.append((P, context, np.type, np.color, o))
        self.identity_log.append((context, np.type, np.color, o, round(P, 3)))
        if not strict:
            self._eliminated = True; self._cur_weight = P; self._cur_identity = (np.type, np.color, o)
            rule = self.explicit_elimination(np.type, np.color)
            self._cur_rule = rule if rule.conclusive and rule.remaining == o else None
        else:
            self._last_query_identity = P
            rule = self.explicit_elimination(np.type, np.color)
            self._last_query_rule = rule if rule.conclusive and rule.remaining == o else None
        return [o], []

    def answer(self, text, mode=None):
        self._last_query_identity = None; self._last_query_rule = None
        ans = super().answer(text, mode)
        if self._last_query_identity is not None:
            ans.features["identity_p"] = self._last_query_identity
            if self._last_query_rule is not None: ans.features["identity_rule"] = self._last_query_rule
        return ans

    # ---- calibration on the CALIBRATION + VALIDATION splits (reference answers; never the test) ----
    GRID = {"observation_threshold": (0.8, 0.9, 0.97, 1.01), "query_threshold": (0.97, 1.01)}
    COSTS = {"wrong": 20.0, "abstain": 1.0}

    def calibrate(self, worlds, extra_worlds=()):
        """Thresholds = argmin of the expected risk (confident wrong answer 20, abstention 1) on non-test worlds;
        + reliability curve of the identity commitments (truth available on these worlds)."""
        if not self.cfg.calibrate: return {}
        worlds = list(worlds) + list(extra_worlds)
        import copy
        saved = copy.deepcopy({k: v for k, v in self.__dict__.items() if k not in ("calibration", "calibrator")})
        try: return self._calibrate(worlds)
        finally:                                          # calibration sets ONLY thresholds: no learning is kept
            keep = (self.calibration, self.calibrator)
            self.__dict__.update(saved); self.calibration, self.calibrator = keep

    def _calibrate(self, worlds):
        from core.independent_agent import ABSTAIN
        best, table = None, []
        # from the most cautious to the least cautious: a risky option wins only if its risk is STRICTLY lower
        for ot in sorted(self.GRID["observation_threshold"], reverse=True):
            for qt in sorted(self.GRID["query_threshold"], reverse=True):
                self.calibration = {"observation_threshold": ot, "query_threshold": qt}
                cost = n = wrong = 0
                for x in worlds:
                    self.new_world(); self.ingest(x["observations"])
                    for it in x["items"]:
                        out = self.answer(it.text).output
                        if out in ABSTAIN: cost += self.COSTS["abstain"] * (it.gold not in ABSTAIN)
                        elif out != it.gold: cost += self.COSTS["wrong"]; wrong += 1
                        n += 1
                table.append((round(cost / max(1, n), 4), ot, qt, round(wrong / max(1, n), 4)))
                if best is None or cost < best[0]: best = (cost, ot, qt)
        self.calibration = {"observation_threshold": best[1], "query_threshold": best[2], "risk_table": table}
        pts = identity_accuracy(self, worlds)
        self.calibration["report"] = calibration_report([c for c, _ in pts], [y for _, y in pts])
        self.calibration["n"] = len(pts)
        return self.calibration


def identity_accuracy(agent, worlds):
    """Evaluation: accuracy of the identity commitments (true colour of the chosen object = described colour)."""
    pts = []
    for x in worlds:
        agent.new_world(); n0 = len(agent.identity_points); agent.ingest(x["observations"])
        for P, ctx, typ, color, o in agent.identity_points[n0:]:
            pts.append((P, x["world"].objects[o].color == color))
    return pts


def belief_accuracy(agent, worlds):
    """Evaluation: FACT/BELIEF beliefs compared with the truth of the world."""
    from core.independent_agent import BELIEF, FACT
    pts = []
    for x in worlds:
        agent.new_world(); agent.ingest(x["observations"])
        facts = x["world"].facts()
        for key in list(agent.belief.graph.props):
            b = agent.belief.graph.belief(key)
            if b.status not in (FACT, BELIEF): continue
            truth = facts.get(key)
            if key[0] == "event":
                last = [e for e in x["world"].events if e.patient == key[1]]
                truth = (last[-1].rel, last[-1].target) if last else None
            if truth is None: continue
            pts.append((b.posterior, b.value == truth))
    return pts

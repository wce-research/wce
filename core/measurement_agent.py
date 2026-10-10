"""Measurement agent in the temporal world: temporal beliefs + compatible set for identity +
active measurement planner (information gain / cost / disturbance / stopping). Six systems compared.

The agent NEVER accesses the truth: active measurements go through LatentWorldState (simulator), which counts
every direct read by the agent (NO_HIDDEN_TRUTH_ACCESS).
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

import torch

from core.measurement import (
    DESTRUCTIVE, PASSIVE, CompatibleStateSet, LatentWorldState, MeasurementContext, MeasurementOperator,
    MeasurementOutcome, MeasurementPlanner, MeasurementRecord, NoiseModel, Observable, information_gain, new_graph, trace,
    MeasurementOp,
)
from core.measurement_language import parse, parse_question
from core.measurement_world import ACTIVE_MENU, COLORS, HEIGHT, REGION, SUPPORTS
from core.source_reliability import SourceReliabilityModel
from core.temporal_belief import HazardModel, JointDerivedState, KeyDynamics, TemporalBeliefState

SYSTEMS_M = ("last_observation", "static", "temporal", "temporal_passive", "temporal_active", "full_algebra")
COLOR_LIST = list(COLORS)
HEIGHT_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
REGIONS = ["high", "low"]
COMMIT = 0.9                 # ANSWER threshold (on the calibrated confidence)
STOP_CONFIDENCE = 0.99       # STOPPING rule of the measurements (keep checking as long as the utility remains positive)
WRONG_COST, ABSTAIN_COST = 20.0, 8.0     # abstention cost chosen on VALIDATION (max. success under 3 % false confident)
NOMINAL = {"camera": 0.9, "color_cam": 0.95, "scale": 0.9, "camera_event": 0.85, "alice": 0.7, "bob": 0.7, "doc": 0.7}


# ================================================================ training (TRAINING episodes only)

def fit_training(episodes):
    trajs = []
    for ep in episodes:
        for o, tr in ep["_truth"]["trajectories"].items():
            trajs.append(("loc", tr["loc"])); trajs.append(("owner", tr["owner"]))
    hz = HazardModel().fit(trajs)
    tog = exp = st0 = n0 = 0
    for ep in episodes:
        for o, tr in ep["_truth"]["trajectories"].items():
            off = [h != HEIGHT[l] for l, h in zip(tr["loc"], tr["height"])]
            tog += sum(a != b for a, b in zip(off, off[1:])); exp += len(off) - 1; st0 += off[0]; n0 += 1
    hz.toggle = (tog + 1) / (exp + 50); hz.mod_prior = [1 - (st0 + 1) / (n0 + 2), (st0 + 1) / (n0 + 2)]
    rel = SourceReliabilityModel(mode="mean", global_prior=(2.0, 1.0))
    for ep in episodes:
        tr = ep["_truth"]
        for ob in ep["observations"]:
            if ob["kind"] == "sensor":
                truth = tr["objects"][ob["entity"]]["color"] if ob["attribute"] == "color" else tr["trajectories"][ob["entity"]][ob["attribute"]][ob["t"]]
                ok = abs(ob["value"] - truth) <= 0.25 if ob["attribute"] == "height" else ob["value"] == truth
                rel.observe(ob["source"], ok, ob["attribute"])
            else:
                p = parse(ob["text"])
                if p is None or p.kind == "continuous": continue
                truth = ob.get("event_to") or tr["trajectories"][ob["entity_truth"]]["loc"][ob["t"]]
                rel.observe(p.observer, p.value == truth, "loc")
    return hz, rel


# ================================================================ coarse operator (resolution: region)

class RegionOperator(MeasurementOperator):
    """COARSE position: the outcome is a region (high / low) — resolution ≠ uncertainty."""

    def outcomes(self, grid=None): return list(REGIONS)

    def likelihood(self, outcome, values):
        r = self.context.noise.reliability
        o = outcome.value if isinstance(outcome, MeasurementOutcome) else outcome
        reg = torch.tensor([REGIONS.index(REGION[SUPPORTS[int(v)]]) for v in values.tolist()])
        return torch.where(reg == REGIONS.index(o), torch.tensor(r, dtype=torch.float64), torch.tensor(1 - r, dtype=torch.float64))

    def sample(self, true_value, rng):
        reg = REGION[true_value]
        return MeasurementOutcome(reg if rng.random() < self.context.noise.reliability else [x for x in REGIONS if x != reg][0])


class WeighOperator(MeasurementOperator):
    """Weighing: informs about the height via the position, but DESTRUCTIVE (the object is put on the floor)."""

    def outcomes(self, grid=None): return list(HEIGHT_GRID)

    def likelihood(self, outcome, values):
        o = outcome.value if isinstance(outcome, MeasurementOutcome) else outcome
        r = self.context.noise.reliability
        h = torch.tensor([HEIGHT[SUPPORTS[int(v)]] for v in values.tolist()], dtype=torch.float64)
        return torch.where((h - o).abs() < 0.26, torch.tensor(r, dtype=torch.float64), torch.tensor((1 - r) / 5, dtype=torch.float64))

    def sample(self, true_value, rng):
        h = HEIGHT[true_value]
        return MeasurementOutcome(h if rng.random() < self.context.noise.reliability else rng.choice([x for x in HEIGHT_GRID if x != h]))


def _nearest_height_reliability(sigma, resolution):
    """P(the reading rounded to the 0.5 grid denotes the true height): noise AND resolution (distinct)."""
    s = math.sqrt(sigma ** 2 + resolution ** 2 / 12.0)
    if s <= 0: return 0.99
    return min(0.99, math.erf(0.25 / (s * math.sqrt(2))))


# ================================================================ agent

class MeasurementAgent:
    def __init__(self, system, hz, rel, policy="max_utility", seed=0, graph=None, calibrators=None):
        self.system, self.hz, self.policy = system, hz, policy
        self.learned = system == "full_algebra"
        self.calibrators = calibrators if system == "full_algebra" else None
        self.min_attribution = 0.9 if system == "full_algebra" else 0.0      # cautious attribution of testimonies
        self.rel = rel
        self.rng = random.Random(seed)
        self.graph = graph if graph is not None else new_graph(f"agent-{system}")
        self.stats = Counter()
        self.value_model = None          # MeasurementValueLearner (learned / hybrid policies)
        self.collector = None            # list: collection (TRAINING only) of the realised values of the measurements
        self._label = None               # truth of the answer variable (TRAINING only, for labelling)

    def r(self, source, key="loc"):
        """Reliability used: learned and CAUTIOUS (credible lower bound) for the full algebra; nominal otherwise."""
        if self.learned: return min(self.rel.effective(source, key, mode="lower") if hasattr(self.rel, "effective") else self.rel.estimate_for(source, key), 0.99)
        return NOMINAL.get(source, 0.7)

    # ---- ingestion ----------------------------------------------------------------------
    def ingest(self, ep):
        self.ep, self.T = ep, ep["T"]
        self.types = {o: d["type"] for o, d in ep["objects"].items()}
        self.color = {o: torch.full((len(COLOR_LIST),), 1.0 / len(COLOR_LIST), dtype=torch.float64) for o in self.types}
        lam = self.hz.get("loc") if self.system not in ("static", "last_observation") else KeyDynamics("loc", 1e-6)
        self.loc = {o: TemporalBeliefState(o, "loc", SUPPORTS, lam) for o in self.types}
        self.height = {o: JointDerivedState(self.loc[o], (0.0, 0.5), lambda l, s: HEIGHT[l] + s, getattr(self.hz, "toggle", 0.02),
                                            HEIGHT_GRID, getattr(self.hz, "mod_prior", None)) for o in self.types}
        self.last = {}; self.votes = defaultdict(Counter); self.events = []
        for ob in sorted(ep["observations"], key=lambda o: o["t"]):
            if ob["kind"] == "sensor":
                o, a, v, t = ob["entity"], ob["attribute"], ob["value"], ob["t"]
                if a == "color":
                    r = self.r("color_cam", "color")
                    L = torch.full((len(COLOR_LIST),), (1 - r) / 3, dtype=torch.float64); L[COLOR_LIST.index(v)] = r
                    self.color[o] = self.color[o] * L; self.color[o] /= self.color[o].sum()
                elif a == "loc":
                    self.loc[o].observe(t, v, "camera", self.r("camera")); self._baseline(o, "loc", t, v, self.r("camera"))
                elif a == "height":
                    rr = _nearest_height_reliability(0.08 if self.system == "full_algebra" else 0.0, ob.get("resolution", 0.1)) \
                        if self.system == "full_algebra" else 0.9
                    hv = min(HEIGHT_GRID, key=lambda h: abs(h - v))
                    self.height[o].observe(t, hv, "scale", rr); self._baseline(o, "height", t, hv, rr)
            else:
                p = parse(ob["text"])
                if p is None: continue
                w = self.identity_weights(p.description, ob["t"])
                if self.min_attribution and (not w or max(w.values()) < self.min_attribution):
                    self.stats["unattributed"] += 1; continue                 # uncertain identity: no apportioning
                if p.kind == "event":
                    for o, wo in w.items():
                        if wo > 0.01: self.loc[o].event(ob["t"], p.value, "camera_event", wo * self.r("camera_event"))
                    self.events.append((ob["t"], p, w))
                elif p.kind == "observation" and p.value is not None:
                    r = self.r(p.observer)
                    for o, wo in w.items():
                        if wo > 0.01:
                            self.loc[o].observe(ob["t"], p.value, p.observer, wo * r + (1 - wo) / len(SUPPORTS))
                            if wo > 0.5: self._baseline(o, "loc", ob["t"], p.value, r)
                elif p.kind == "continuous":
                    hv = min(HEIGHT_GRID, key=lambda h: abs(h - p.value))
                    rr = _nearest_height_reliability(p.sigma, p.resolution) if self.system == "full_algebra" else 0.9
                    for o, wo in w.items():
                        if wo > 0.01: self.height[o].observe(ob["t"], hv, "doc", wo * rr + (1 - wo) / len(HEIGHT_GRID))

    def _baseline(self, o, a, t, v, r):
        if (o, a) not in self.last or self.last[(o, a)][0] <= t: self.last[(o, a)] = (t, v, r)
        self.votes[(o, a)][v] += math.log(max(r, 0.01) * 4 / max(1 - r, 0.01))

    # ---- identity ------------------------------------------------------------------------------
    def candidates(self, desc):
        return [o for o, t in self.types.items() if desc.type is None or t == desc.type]

    def identity_weights(self, desc, t):
        cands = self.candidates(desc)
        if not cands: return {}
        w = torch.ones(len(cands), dtype=torch.float64)
        if desc.color is not None:
            w = w * torch.stack([self.color[o][COLOR_LIST.index(desc.color)] for o in cands])
        if desc.support is not None:
            w = w * torch.tensor([self.loc[o].historical(t, max(t, 1))[desc.support] if not desc.past else
                                  max(self.loc[o].historical(tt, t)[desc.support] for tt in range(0, t + 1)) for o in cands], dtype=torch.float64)
        if float(w.sum()) <= 0: return {}
        w = w / w.sum()
        return {o: float(x) for o, x in zip(cands, w)}

    # ---- distributions per system ----------------------------------------------------------------
    def loc_dist(self, o, t, now):
        if self.system == "last_observation":
            past = [(tt, v, r) for (oo, a), (tt, v, r) in self.last.items() if oo == o and a == "loc" and tt <= t]
            if not past: return {s: 1 / len(SUPPORTS) for s in SUPPORTS}
            tt, v, r = max(past)
            return {s: (r if s == v else (1 - r) / 4) for s in SUPPORTS}
        if self.system == "static":
            sc = self.votes[(o, "loc")]; z = sum(math.exp(x) for x in sc.values()) + (len(SUPPORTS) - len(sc))
            return {s: (math.exp(sc[s]) if s in sc else 1.0) / z for s in SUPPORTS}
        return self.loc[o].current(t) if t == now else self.loc[o].historical(t, now)

    # ---- compatible set for a question ----------------------------------------------------
    def ensemble(self, intent, desc, cands, t):
        dom = {}
        for o in cands:
            dom[(o, "color")] = COLOR_LIST; dom[(o, "loc")] = SUPPORTS
        S = CompatibleStateSet.product(dom, graph=self.graph)
        lw = torch.zeros(len(S), dtype=torch.float64)
        for o in cands:
            lw += torch.log(self.color[o].clamp_min(1e-12))[S.column((o, "color"))]
            ld = self.loc_dist(o, t, self.T)
            lw += torch.log(torch.tensor([ld[s] for s in SUPPORTS], dtype=torch.float64).clamp_min(1e-12))[S.column((o, "loc"))]
        # referent (presupposition: exactly one candidate satisfies the description)
        if intent == "color_of_relational":
            match = torch.stack([S.column((o, "loc")) == SUPPORTS.index(desc.support) for o in cands], 1)
        elif desc.color is not None:
            match = torch.stack([S.column((o, "color")) == COLOR_LIST.index(desc.color) for o in cands], 1)
        else:
            match = torch.ones(len(S), len(cands), dtype=torch.bool) if len(cands) == 1 else None
        if match is None:                                                     # type alone, several candidates: undetermined referent
            ref = torch.full((len(S),), -1)
            ok = torch.zeros(len(S), dtype=torch.bool)
        else:
            ok = match.sum(1) == 1; ref = match.float().argmax(1)
        lw = torch.where(ok, lw, torch.full_like(lw, -float("inf")))
        S.logw = lw
        if intent == "color_of_relational":
            ans = torch.stack([S.column((o, "color")) for o in cands], 1).gather(1, ref.clamp_min(0).view(-1, 1)).view(-1)
            adom = COLOR_LIST
        else:
            ans = torch.stack([S.column((o, "loc")) for o in cands], 1).gather(1, ref.clamp_min(0).view(-1, 1)).view(-1)
            adom = SUPPORTS
        S.states = torch.cat([S.states, ans.view(-1, 1), ref.clamp_min(0).view(-1, 1)], 1)
        S.vars = S.vars + [("ans", "value"), ("ans", "ref")]; S.index = {v: i for i, v in enumerate(S.vars)}
        S.domains[("ans", "value")] = adom; S.domains[("ans", "ref")] = cands
        return S

    def operators(self, cands, intent=None):
        """Admissible measurements: a PRESENT measurement does not observe the past — for a dated question, only the
        IDENTITY measurements (static properties: colour) are admissible."""
        ops = []
        for o in cands:
            name, (attr, cost, r, dist) = "color", ACTIVE_MENU["color"]
            ops.append(MeasurementOperator(Observable("color", o, "color", tuple(COLOR_LIST)),
                                           MeasurementContext(sensor="color_probe", noise=NoiseModel("confusion", r), cost=cost)))
            if intent == "where_at": continue
            attr, cost, r, dist = ACTIVE_MENU["loc_precise"]
            ops.append(MeasurementOperator(Observable("loc", o, "loc", tuple(SUPPORTS)),
                                           MeasurementContext(sensor="loc_precise", noise=NoiseModel("confusion", r), cost=cost)))
            attr, cost, r, dist = ACTIVE_MENU["loc_coarse"]
            ops.append(RegionOperator(Observable("loc", o, "loc", tuple(SUPPORTS)),
                                      MeasurementContext(sensor="loc_coarse", noise=NoiseModel("confusion", r), cost=cost)))
            attr, cost, r, dist = ACTIVE_MENU["weigh"]
            ops.append(WeighOperator(Observable("loc", o, "loc", tuple(SUPPORTS)),
                                     MeasurementContext(sensor="weigh", noise=NoiseModel("confusion", r), cost=cost,
                                                        disturbance=DESTRUCTIVE, domain="disturbs:value")))
        return ops

    def latent(self):
        tr = self.ep["_truth"]
        vals = {}
        for o in self.types:
            vals[(o, "color")] = tr["objects"][o]["color"]; vals[(o, "loc")] = tr["trajectories"][o]["loc"][self.T]
        return LatentWorldState(vals, seed=self.rng.randrange(10 ** 6))

    def calibrated(self, intent, p):
        cal = (self.calibrators or {}).get(intent)
        return float(cal(p)) if cal is not None else p

    def measure_loop(self, S, cands, max_steps=4, intent=None):
        """Scientific loop: hypotheses -> choose a measurement -> observe -> condition -> repeat / stop."""
        if self.system in ("temporal", "static", "last_observation"): return S, 0, 0.0, [], 0
        world = self.latent()
        policy = "fixed" if self.system == "temporal_passive" else self.policy
        planner = MeasurementPlanner(policy=policy, confidence=(1.01 if self.calibrators is not None else COMMIT), budget=2.0, seed=self.rng.randrange(10 ** 6),
                                     graph=self.graph, value_of_information=2.0)
        if self.system == "full_algebra":                                    # utility = decision value − cost
            planner.decision_costs = (WRONG_COST, ABSTAIN_COST)
            planner.calibrate = (lambda p, _i=intent: self.calibrated(_i, p))
        planner.value_model = self.value_model; planner.context = {"intent": intent}
        n, cost, steps, destructive = 0, 0.0, [], 0
        target = ("ans", "value")
        while n < max_steps:
            ops = self.operators(cands, intent)
            if self.calibrators is not None and max(S.marginal(target).values()) >= STOP_CONFIDENCE: break     # stopping ≠ answer threshold
            if self.system == "temporal_passive":
                if n >= 2: break
                ops = [op for op in ops if op.context.disturbance == PASSIVE]
                c = planner.choose(S, ops[n % len(ops):] + ops[:n % len(ops)], target)
            else:
                c = planner.choose(S, ops, target)
            if self.collector is not None and self._label is not None: self._collect(S, ops, target, intent, world)
            if c is None: break
            h0 = S.entropy(target)
            rec = world.measure(c.operator, c.operator.context.sensor, self.graph)
            S = self.update(S, c.operator, rec)
            if c.operator.context.disturbance == DESTRUCTIVE: destructive += 1
            steps.append((c.information_gain, h0 - S.entropy(target)))
            n += 1; cost += c.cost
        self.stats["agent_peeks"] += world.agent_peeks
        return S, n, cost, steps, destructive

    def update(self, S, op, rec):
        if isinstance(op, (RegionOperator, WeighOperator)):
            col = S.column((op.observable.entity, "loc")); L = op.likelihood(rec.outcome, col)
            return S._fork(S.logw + torch.log(L.clamp_min(1e-300)), f"mesure {op.name}")
        return S.condition(rec)

    def realized_loss(self, S, target, intent):
        """REALISED decision loss (truth known: TRAINING only)."""
        m = S.marginal(target); v, raw = max(m.items(), key=lambda kv: kv[1]); p = self.calibrated(intent, raw)
        if p < COMMIT: return ABSTAIN_COST
        return 0.0 if v == self._label else WRONG_COST

    def _collect(self, S, ops, target, intent, world):
        """For EACH candidate measurement: simulate the measurement on a copy of the latent world (training) and
        record (features, realised value = loss before − loss after)."""
        from core.measurement import decision_value
        before = self.realized_loss(S, target, intent)
        for op in ops:
            ig = information_gain(S, op, target)
            dv = decision_value(S, op, target, WRONG_COST, ABSTAIN_COST, calibrate=lambda p, _i=intent: self.calibrated(_i, p))
            vals = []
            for k in range(2):
                w = LatentWorldState(dict(world._values), seed=self.rng.randrange(10 ** 6))
                vals.append(before - self.realized_loss(self.update(S, op, w.measure(op, "simulation")), target, intent))
            self.collector.append({"features": self.value_model_features(S, op, target, ig, dv, intent), "realized": sum(vals) / len(vals),
                                   "analytic": dv, "ig": ig, "cost": op.context.cost})

    def value_model_features(self, S, op, target, ig, dv, intent):
        from core.measurement_value import features
        return features(S, op, target, ig, dv, {"intent": intent})

    # ---- answers -----------------------------------------------------------------------------------
    def answer(self, q):
        pq = parse_question(q["text"])
        if pq is None: return {"answer": None, "p": 0.0, "n": 0, "cost": 0.0, "steps": [], "destructive": 0}
        intent = pq["intent"]
        if intent == "before_fall": return self._before_fall(pq)
        desc = pq.get("description")
        cands = self.candidates(desc)
        t = pq.get("t", self.T)
        if not cands: return {"answer": None, "p": 0.0, "n": 0, "cost": 0.0, "steps": [], "destructive": 0}
        S = self.ensemble(intent, desc, cands, t)
        n, cost, steps, destr = 0, 0.0, [], 0
        if self.collector is not None and "truth" in q:          # label (TRAINING only)
            tr = self.ep["_truth"]
            self._label = tr["trajectories"][q["entity"]]["loc"][self.T] if intent == "height_now" and "entity" in q else q["truth"]
        if intent in ("where_now", "height_now", "color_of_relational") or (intent == "where_at" and desc.color is not None):
            S, n, cost, steps, destr = self.measure_loop(S, cands, intent=intent)
        if not torch.isfinite(S.logw).any():
            return {"answer": None, "p": 0.0, "n": n, "cost": cost, "steps": steps, "destructive": destr}
        m = S.marginal(("ans", "value"))
        if intent == "height_now":
            refm = S.marginal(("ans", "ref"))
            hd = defaultdict(float)
            for o, pr in refm.items():
                if pr < 1e-6: continue
                jd = self.height[o].current(self.T) if self.system not in ("static", "last_observation") else None
                if jd is None:
                    last = self.last.get((o, "height"))
                    jd = {h: (last[2] if last and h == last[1] else ((1 - last[2]) / 5 if last else 1 / 6)) for h in HEIGHT_GRID}
                # fusion: height deduced from the measured position × own model of the height (product of experts)
                stack_p = 1 - (getattr(self.hz, "mod_prior", [0.95])[0])
                for loc, pl in m.items():
                    for sv, ps in ((0.0, 1 - stack_p), (0.5, stack_p)):
                        hd[HEIGHT[loc] + sv] += pr * pl * ps * jd.get(HEIGHT[loc] + sv, 1e-6)
            z = sum(hd.values()) or 1.0
            m = {h: v / z for h, v in hd.items()}
        v, raw = max(m.items(), key=lambda kv: kv[1])
        p = self.calibrated(intent, raw)
        return {"answer": v if p >= COMMIT else None, "p": p, "raw_p": raw, "best": v, "n": n, "cost": cost, "steps": steps,
                "destructive": destr}

    def _before_fall(self, pq):
        falls = [(t, p, w) for t, p, w in self.events if p.extra.get("verb") == "tombe"]
        if not falls: return {"answer": None, "p": 0.0, "n": 0, "cost": 0.0, "steps": [], "destructive": 0}
        t, p, w = falls[-1]
        scores = {}
        for o, wo in w.items():
            on = self.loc_dist(o, t - 1, self.T).get(pq["support"], 0.0)
            col = COLOR_LIST[int(self.color[o].argmax())]; pc = float(self.color[o].max())
            scores[f"{self.types[o]}:{col}"] = scores.get(f"{self.types[o]}:{col}", 0.0) + wo * on * pc
        z = sum(scores.values())
        if z <= 0: return {"answer": None, "p": 0.0, "n": 0, "cost": 0.0, "steps": [], "destructive": 0}
        v, s = max(scores.items(), key=lambda kv: kv[1])
        raw = s / max(z, 1e-12) * min(1.0, z); p = self.calibrated("before_fall", raw)
        return {"answer": v if p >= COMMIT else None, "p": p, "raw_p": raw, "best": v, "n": 0, "cost": 0.0, "steps": [], "destructive": 0}


# ================================================================ evaluation

def fit_calibrators(train, hz, rel, passes=2):
    """Isotonic calibrators of the confidence of the answers, PER TYPE, on the TRAINING episodes (truth known).
    Fitted WITH the final policy (fixed point in `passes` passes): otherwise the calibration reflects another policy."""
    from core.unified import IsotonicCalibrator
    cals = {}
    for _ in range(passes):
        pts = defaultdict(list)
        for ep in train:
            ag = MeasurementAgent("full_algebra", hz, rel, calibrators=cals); ag.calibrators = cals; ag.ingest(ep)
            for q in ep["questions"]:
                a = ag.answer(q)
                truth = q["truth"]; b = a.get("best")
                ok = b is not None and (abs(b - truth) < 1e-6 if isinstance(truth, float) else b == truth)
                if a.get("raw_p", a["p"]) > 0: pts[q["type"]].append((a.get("raw_p", a["p"]), ok))
        cals = {k: IsotonicCalibrator(prior_strength=5.0, bins=40).fit([c for c, _ in v], [y for _, y in v]) for k, v in pts.items()}
    return cals


def evaluate(system, episodes, hz, rel, policy="max_utility", seed=0, calibrators=None):
    rows = []
    for ep in episodes:
        ag = MeasurementAgent(system, hz, rel, policy, seed, calibrators=calibrators)
        ag.ingest(ep)
        for q in ep["questions"]:
            a = ag.answer(q)
            truth = q["truth"]
            same = (lambda x: x is not None and (abs(x - truth) < 1e-6 if isinstance(truth, float) else x == truth))
            rows.append({"type": q["type"], "answer": a["answer"], "p": a["p"], "committed": a["answer"] is not None,
                         "correct": same(a["answer"]), "best_ok": same(a.get("best")),
                         "false_confident": a["answer"] is not None and not same(a["answer"]),
                         "n": a["n"], "cost": a["cost"], "steps": a["steps"], "destructive": a["destructive"],
                         "peeks": ag.stats["agent_peeks"]})
    return rows


def metrics(rows):
    from core.unified import calibration_report
    n = max(1, len(rows))
    steps = [s for r in rows for s in r["steps"]]
    out = {"n": len(rows), "success": round(sum(r["correct"] for r in rows) / n, 4),
           "false_confident": round(sum(r["false_confident"] for r in rows) / n, 4),
           "abstention": round(sum(not r["committed"] for r in rows) / n, 4),
           "measurements": round(sum(r["n"] for r in rows) / n, 3), "cost": round(sum(r["cost"] for r in rows) / n, 3),
           "destructive": sum(r["destructive"] for r in rows), "agent_peeks": sum(r["peeks"] for r in rows),
           "calibration_ece": calibration_report([r["p"] for r in rows if r["p"] > 0], [r["best_ok"] for r in rows if r["p"] > 0])["ece"]}
    if steps:
        p = torch.tensor(steps, dtype=torch.float64)
        out["ig_predicted"] = round(float(p[:, 0].mean()), 4); out["ig_realized"] = round(float(p[:, 1].mean()), 4)
        out["ig_corr"] = round(float(torch.corrcoef(p.T)[0, 1]), 3) if len(steps) > 2 and float(p[:, 0].std()) > 0 and float(p[:, 1].std()) > 0 else None
    by = defaultdict(list)
    for r in rows: by[r["type"]].append(r)
    out["by_type"] = {k: {"success": round(sum(r["correct"] for r in v) / len(v), 3), "fc": round(sum(r["false_confident"] for r in v) / len(v), 3),
                          "n": len(v)} for k, v in by.items()}
    return out


def multi_seed(seeds=range(16), n_eval=15, policies=("random", "fixed", "most_reliable", "max_ig", "max_utility"),
               ablation_seeds=8, log=print):
    """Training ONCE (rates, reliabilities, calibrators: training episodes); evaluation on independent episode
    seeds; ablation of the measurement policies on the full algebra."""
    from core.grounded_benchmark import paired, summarize_values
    from core.measurement_world import train_episodes, eval_episodes
    train = train_episodes(30)
    hz, rel = fit_training(train); cal = fit_calibrators(train, hz, rel)
    per, abl = [], []
    keys = ("success", "false_confident", "abstention", "measurements", "cost", "calibration_ece")
    for s in seeds:
        eps = eval_episodes(s, n_eval)
        out = {name: metrics(evaluate(name, eps, hz, rel, calibrators=cal, seed=s)) for name in SYSTEMS_M}
        per.append(out)
        if s < ablation_seeds:
            abl.append({p: metrics(evaluate("full_algebra", eps, hz, rel, policy=p, calibrators=cal, seed=s)) for p in policies})
        log(s, {n: (out[n]["success"], out[n]["false_confident"], out[n]["measurements"]) for n in SYSTEMS_M})
    summary = {n: {k: summarize_values([o[n][k] for o in per]) for k in keys} for n in SYSTEMS_M}
    comps = {f"{a}_vs_{b}": {k: paired([o[a][k] for o in per], [o[b][k] for o in per]) for k in keys}
             for a, b in (("temporal_active", "temporal_passive"), ("full_algebra", "temporal_passive"),
                          ("full_algebra", "temporal_active"), ("temporal", "static"), ("temporal_passive", "temporal"),
                          ("full_algebra", "temporal"))}
    ablation = {p: {k: summarize_values([a[p][k] for a in abl]) for k in keys + ("ig_corr",)} for p in policies}
    igc = summarize_values([o["full_algebra"].get("ig_corr") for o in per])
    return {"summary": summary, "comparisons": comps, "ablation": ablation, "ig_corr_full": igc,
            "per_seed": per, "by_type": {n: per[0][n]["by_type"] for n in SYSTEMS_M}}

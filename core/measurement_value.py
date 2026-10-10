"""Learning of the REALISED VALUE of measurements (MeasurementValueLearner) and meta-planner.

Data: TRAINING episodes. At each measurement decision, each candidate measurement is SIMULATED on a copy
of the latent world; label = realised decision loss before − after (truth known in training only).
Features: analytic IG, cost, disturbance, goal (question type), current uncertainty (entropy,
max confidence), sensor type. Model: ridge regression (closed form, small) + fast online correction
(linear "fast weights", decay) — no large network.

Policies compared: max_ig, max_utility (analytic decision value, manual costs), learned
(predicted value alone), hybrid (analytic value + learned residual). Every decision remains verifiable: the
prediction, its features and the analytic value are traced (SelectMeasurement).

Counterfactual measurement: `counterfactual_measurement` answers "if I measured X, what reduction of uncertainty
would I expect?" WITHOUT measuring: neither the world nor the belief changes, and the result is marked HYPOTHETICAL.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from core.measurement import DESTRUCTIVE, PASSIVE, WEAKLY_DISTURBING, expected_posterior_entropy, information_gain

INTENTS = ("where_now", "where_at", "height_now", "color_of_relational")
SENSORS = ("color_probe", "loc_precise", "loc_coarse", "weigh")


def features(S, op, target, ig, dv, context):
    m = S.marginal(target); pmax = max(m.values()); H = S.entropy(target)
    intent = context.get("intent"); sensor = op.context.sensor
    dist = {PASSIVE: 0.0, WEAKLY_DISTURBING: 0.5, DESTRUCTIVE: 1.0}.get(op.context.disturbance, 0.0)
    f = [1.0, ig, ig * ig, H, pmax, pmax * ig, op.context.cost, dist, op.context.noise.reliability, ig * (1 - pmax)]
    f += [1.0 if intent == i else 0.0 for i in INTENTS]
    f += [1.0 if sensor == s else 0.0 for s in SENSORS]
    f += [ig * (1.0 if intent == i else 0.0) for i in INTENTS]
    return f


ANALYTIC_INDEX = None      # the analytic value is NOT a feature of the "learned" model


class MeasurementValueLearner:
    """RealizedValue ≈ f(IG, cost, disturbance, goal, uncertainty); hybrid mode: analytic + learned residual."""

    def __init__(self, ridge=1.0, fast_lr=0.05, fast_decay=0.9):
        self.ridge, self.fast_lr, self.fast_decay = ridge, fast_lr, fast_decay
        self.w = None; self.w_res = None
        self.fast = None; self.fast_res = None
        self.stats = {}

    features = staticmethod(features)

    def _fit(self, X, y):
        A = X.T @ X + self.ridge * torch.eye(X.shape[1], dtype=X.dtype); A[0, 0] -= self.ridge     # bias not regularised
        return torch.linalg.solve(A, X.T @ y)

    def fit(self, rows: List[Dict]):
        X = torch.tensor([r["features"] for r in rows], dtype=torch.float64)
        y = torch.tensor([r["realized"] for r in rows], dtype=torch.float64)
        a = torch.tensor([r["analytic"] for r in rows], dtype=torch.float64)
        self.w = self._fit(X, y); self.w_res = self._fit(X, y - a)
        self.fast = torch.zeros_like(self.w); self.fast_res = torch.zeros_like(self.w_res)
        pred, pred_h = X @ self.w, a + X @ self.w_res
        self.stats = {"n": len(rows), "mse_learned": float(((pred - y) ** 2).mean()), "mse_hybrid": float(((pred_h - y) ** 2).mean()),
                      "mse_analytic": float(((a - y) ** 2).mean()), "mse_constant": float(((y - y.mean()) ** 2).mean()),
                      "corr_ig_realized": _corr([r["ig"] for r in rows], y.tolist()),
                      "corr_analytic_realized": _corr(a.tolist(), y.tolist()), "corr_learned_realized": _corr(pred.tolist(), y.tolist())}
        return self

    def predict(self, f, hybrid=False, analytic=0.0):
        x = torch.tensor(f, dtype=torch.float64)
        if hybrid: return float(analytic + x @ (self.w_res + self.fast_res))
        return float(x @ (self.w + self.fast))

    def adapt(self, f, observed, hybrid=False, analytic=0.0):
        """LOCAL online adaptation (linear fast weights, decay towards 0) from an observed value."""
        x = torch.tensor(f, dtype=torch.float64); err = observed - self.predict(f, hybrid, analytic)
        step = self.fast_lr * err * x / (1.0 + float(x @ x))
        if hybrid: self.fast_res = self.fast_decay * self.fast_res + step
        else: self.fast = self.fast_decay * self.fast + step
        return err

    def reset_fast(self):
        self.fast = torch.zeros_like(self.w); self.fast_res = torch.zeros_like(self.w_res)


def _corr(a, b):
    t = torch.tensor([a, b], dtype=torch.float64)
    if float(t[0].std()) == 0 or float(t[1].std()) == 0: return None
    return round(float(torch.corrcoef(t)[0, 1]), 3)


# ================================================================ collection (TRAINING) and evaluation

def collect(train, hz, rel, cal, seed=0):
    from core.measurement_agent import MeasurementAgent
    rows = []
    for ep in train:
        ag = MeasurementAgent("full_algebra", hz, rel, seed=seed, calibrators=cal); ag.collector = rows; ag.ingest(ep)
        for q in ep["questions"]:
            ag._label = None; ag.answer(q)
    return rows


def train_value_learner(n_train=30, seed=0):
    from core.measurement_agent import fit_calibrators, fit_training
    from core.measurement_world import train_episodes
    train = train_episodes(n_train); hz, rel = fit_training(train); cal = fit_calibrators(train, hz, rel)
    rows = collect(train, hz, rel, cal, seed)
    return MeasurementValueLearner().fit(rows), (hz, rel, cal), rows


def evaluate_policies(episodes, hz, rel, cal, learner, policies=("max_ig", "max_utility", "learned", "hybrid"), seed=0):
    from core.measurement_agent import MeasurementAgent, metrics
    out = {}
    for pol in policies:
        rows = []
        for ep in episodes:
            ag = MeasurementAgent("full_algebra", hz, rel, pol, seed, calibrators=cal); ag.value_model = learner; ag.ingest(ep)
            for q in ep["questions"]:
                a = ag.answer(q); truth = q["truth"]
                same = (lambda x: x is not None and (abs(x - truth) < 1e-6 if isinstance(truth, float) else x == truth))
                rows.append({"type": q["type"], "answer": a["answer"], "p": a["p"], "committed": a["answer"] is not None,
                             "correct": same(a["answer"]), "best_ok": same(a.get("best")),
                             "false_confident": a["answer"] is not None and not same(a["answer"]), "n": a["n"], "cost": a["cost"],
                             "steps": a["steps"], "destructive": a["destructive"], "peeks": ag.stats["agent_peeks"]})
        out[pol] = metrics(rows)
    return out


# ================================================================ counterfactual measurement

@dataclass(frozen=True)
class CounterfactualMeasurement:
    """HYPOTHETICAL result: nothing was measured; no record is produced."""
    operator: str
    expected_entropy_after: float
    expected_reduction: float
    outcome_probabilities: Dict
    status: str = "HYPOTHETICAL"


def counterfactual_measurement(S, op, target=None):
    """"If I measured X, what reduction of uncertainty would I expect?" — without observing (neither world nor belief)."""
    h_after, po = expected_posterior_entropy(S, op, target)
    h0 = S.entropy(target)
    outs = op.outcomes()
    return CounterfactualMeasurement(op.name, float(h_after), float(h0 - h_after),
                                     {o: round(float(p), 6) for o, p in zip(outs, po.tolist())})


# ================================================================ online adaptation (drift of the real value)

def online_adaptation_experiment(rows, seed=0, drift_sensor="loc_precise", drift=-3.0):
    """The model is fitted on the 1st half; in the 2nd half the realised value of a sensor DRIFTS
    (e.g. degraded sensor). Squared error: frozen model vs model + fast weights adapted online
    (predict, then observe the realised value, then adapt)."""
    rng = random.Random(seed); rows = list(rows); rng.shuffle(rows)
    half = len(rows) // 2; L = MeasurementValueLearner(fast_lr=0.3, fast_decay=0.995).fit(rows[:half])
    frozen = MeasurementValueLearner().fit(rows[:half])
    idx = 10 + len(INTENTS) + SENSORS.index(drift_sensor)
    static, online = [], []
    for r in rows[half:]:
        y = r["realized"] + (drift if r["features"][idx] == 1.0 else 0.0)
        static.append((frozen.predict(r["features"]) - y) ** 2)
        online.append(L.adapt(r["features"], y) ** 2)          # error BEFORE adaptation on this example
    return {"mse_static": round(sum(static) / len(static), 4), "mse_online": round(sum(online) / len(online), 4), "n": len(static)}


def multi_seed_policies(seeds=range(16), n_eval=10, log=print):
    from core.grounded_benchmark import paired, summarize_values
    from core.measurement_world import eval_episodes
    L, (hz, rel, cal), rows = train_value_learner(30)
    per = []
    for s in seeds:
        per.append(evaluate_policies(eval_episodes(s, n_eval), hz, rel, cal, L, seed=s))
        log(s, {p: (per[-1][p]["success"], per[-1][p]["false_confident"], per[-1][p]["cost"]) for p in per[-1]})
    keys = ("success", "false_confident", "abstention", "measurements", "cost", "calibration_ece")
    pols = list(per[0])
    summary = {p: {k: summarize_values([o[p][k] for o in per]) for k in keys} for p in pols}
    comps = {f"{a}_vs_{b}": {k: paired([o[a][k] for o in per], [o[b][k] for o in per]) for k in ("success", "false_confident", "cost")}
             for a, b in (("learned", "max_utility"), ("hybrid", "max_utility"), ("learned", "max_ig"), ("hybrid", "max_ig"), ("hybrid", "learned"))}
    return {"learner": L.stats, "online": online_adaptation_experiment(rows), "summary": summary, "comparisons": comps, "per_seed": per}

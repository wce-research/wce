import inspect
import json
import math
import os
import random
import re
import unittest

import torch

import core.robust_experiments as RE
import core.robust_reconstruction as RR
import core.robust_world as RW
import wce_aws.orchestrate as ORC
import wce_aws.sm_entry as ENT
from core.boundary_program import BNode, predict, structure_hash
from core.robust_aggregate import aggregate, ci, spearman
from core.robust_reconstruction import (ACTIONS, DISCREPANCY, LIKELIHOODS, SENSOR_MODELS, CompatibleStructureSet, MeasurementModel, MeasurementOutcome,
                                        Observations, ParameterPosterior, RobustMeasurementPlanner, RobustStructuralReconstruction, StructureEntry,
                                        aic, attribute_discrepancy, beta_calibration, bic, calibration_metrics, correlated_weights, disagreement,
                                        estimate_noise, expected_decision_loss_utility, expected_functional_accuracy_gain, fit_robust, fourier_reconstruction,
                                        frame_transform, functional_ig, isotonic, log_evidence, mdl, obs_loglik, outlier_posterior, platt,
                                        posterior_predictive_check, simplify_robust, structural_ig, structure_bits, temporal_weights, uncertainty)
from core.robust_world import (NOISE_KINDS, NOISE_LEVELS, SEALED_SHA256, SENSORS, NoiseSpec, corrupt, full_region, hard_noise_pair, in_region, make_task,
                               make_world, out_region, random_program, sealed_hash, sealed_spec, sealed_tasks)

G = lambda s=0: torch.Generator().manual_seed(s)
P = lambda *a, **k: BNode(*a, **k)
T = lambda c, p: P("translate", params=p, children=(c,))
DISC = T(P("pred", "norm", "le", (1.0,)), (0.2, 0.0))
HALF = P("pred", "dot", "le", (0.0, 0.0))
HALF2 = P("pred", "dot", "le", (0.0, 0.8))
STD = SENSOR_MODELS["standard"]


def X0(n=200, s=0): return torch.rand(n, 2, generator=G(s)) * 4 - 2


def noisy(prog, n=200, rate=0.08, s=0):
    X = X0(n, s); y = predict(prog, X); flip = torch.rand(n, generator=G(s + 1)) < rate; return X, torch.where(flip, 1 - y, y)


def obs_for(prog, n=200, rate=0.08, s=0):
    X, y = noisy(prog, n, rate, s); return Observations.build(X, y, STD, max(0.0, rate - 0.05))


def css_of(progs, obs):
    c = CompatibleStructureSet()
    for p in progs: ll = float(obs_loglik(p, obs)); c.add(StructureEntry(p, ll, log_evidence(p, obs, ll)))
    return c


# ================================================================ A — measurement model

class MeasurementModelTests(unittest.TestCase):
    def test_fields(self):
        for f in ("observable", "source", "frame", "observation_time", "ingestion_time", "resolution", "flip", "contamination", "missingness", "cost", "repeatable"): self.assertTrue(hasattr(STD, f))
    def test_sensor_tradeoff(self): self.assertLess(SENSOR_MODELS["precise"].flip, SENSOR_MODELS["cheap"].flip); self.assertGreater(SENSOR_MODELS["precise"].cost, SENSOR_MODELS["cheap"].cost)
    def test_build(self): o = Observations.build(X0(10), torch.zeros(10, dtype=torch.long), STD, 0.03); self.assertAlmostEqual(float(o.flip[0]), 0.08)
    def test_flip_capped(self): self.assertLessEqual(float(Observations.build(X0(3), torch.zeros(3, dtype=torch.long), STD, 0.9).flip[0]), 0.45)
    def test_add(self): o = Observations.build(X0(10), torch.zeros(10, dtype=torch.long), STD).add(X0(2, 1), torch.ones(2, dtype=torch.long), SENSOR_MODELS["precise"]); self.assertEqual(len(o.X), 12); self.assertEqual(o.source[-1], "precise")
    def test_missing_excluded(self):
        y = torch.tensor([1, -1, 0]); o = Observations.build(X0(3), y, STD); self.assertEqual(len(o), 2)
    def test_outcome_not_state(self): m = MeasurementOutcome(1, STD); self.assertFalse(hasattr(m, "world_state"))
    def test_temporal_weights_decay(self): w = temporal_weights(torch.tensor([0.0, 9.0]), 10.0, half_life=2.0); self.assertLess(float(w[0]), float(w[1]))
    def test_temporal_weights_none(self): self.assertTrue(bool((temporal_weights(torch.tensor([0.0, 5.0]), 10.0) == 1).all()))
    def test_old_observation_not_present(self): self.assertLess(float(temporal_weights(torch.tensor([0.0]), 100.0, 1.0)[0]), 1e-6)
    def test_correlated_weights(self): w = correlated_weights(["a", "a", "a", None]); self.assertLess(float(w[0]), 1.0); self.assertEqual(float(w[3]), 1.0)
    def test_correlated_total_less_than_count(self): self.assertLess(float(correlated_weights(["a"] * 10).sum()), 10.0)
    def test_frame_transform_roundtrip(self):
        X = X0(20); Y = frame_transform(X, 0.7, (1.0, -2.0)); back = frame_transform(Y - torch.tensor([1.0, -2.0]), -0.7); self.assertTrue(torch.allclose(back, X, atol=1e-5))
    def test_frame_distance_invariant(self):
        X = X0(20); Y = frame_transform(X, 1.1, (0.3, 0.4)); self.assertTrue(torch.allclose(torch.cdist(X, X), torch.cdist(Y, Y), atol=1e-5))
    def test_frame_relation_invariant(self):
        X = X0(100); Y = frame_transform(X, 0.5, (1.0, 0.0)); a = predict(DISC, X)
        moved = T(P("pred", "norm", "le", (1.0,)), tuple(frame_transform(torch.tensor([[0.2, 0.0]]), 0.5, (1.0, 0.0))[0].tolist())); self.assertTrue(torch.equal(a, predict(moved, Y)))


# ================================================================ B/C — likelihoods

class LikelihoodTests(unittest.TestCase):
    def setUp(self): self.obs = obs_for(DISC)
    def test_kinds(self): self.assertEqual(set(LIKELIHOODS), {"naive", "flip", "contamination", "soft_margin"})
    def test_true_beats_wrong(self):
        for k in LIKELIHOODS: self.assertGreater(float(obs_loglik(DISC, self.obs, k)), float(obs_loglik(HALF, self.obs, k)), k)
    def test_naive_punishes_noise_more(self):
        self.assertLess(float(obs_loglik(DISC, self.obs, "naive")), float(obs_loglik(DISC, self.obs, "contamination")))
    def test_per_point(self): self.assertEqual(obs_loglik(DISC, self.obs, per_point=True).shape[0], 200)
    def test_loglik_negative(self): self.assertLess(float(obs_loglik(DISC, self.obs)), 0)
    def test_missing_ignored(self):
        o = Observations.build(self.obs.X, torch.full_like(self.obs.y, -1), STD); self.assertEqual(float(obs_loglik(DISC, o)), 0.0)
    def test_weights_scale(self):
        o = Observations(self.obs.X, self.obs.y, self.obs.flip, self.obs.resolution, self.obs.weight * 0.5, self.obs.source)
        self.assertAlmostEqual(float(obs_loglik(DISC, o)), 0.5 * float(obs_loglik(DISC, self.obs)), places=6)
    def test_contamination_bounded(self):
        X = torch.tensor([[0.2, 0.0]]); o = Observations.build(X, torch.tensor([0]), STD); self.assertGreater(float(obs_loglik(DISC, o, "contamination", gamma=0.1)), float(obs_loglik(DISC, o, "naive")))
    def test_outlier_posterior_range(self): p = outlier_posterior(DISC, self.obs); self.assertTrue(bool(((p >= 0) & (p <= 1)).all()))
    def test_outlier_posterior_high_for_contradiction(self):
        X = torch.tensor([[0.2, 0.0], [0.2, 0.1]]); o = Observations.build(X, torch.tensor([0, 1]), SENSOR_MODELS["precise"]); p = outlier_posterior(DISC, o, 0.05); self.assertGreater(float(p[0]), float(p[1]))
    def test_soft_margin_tolerant_at_boundary(self):
        X = torch.tensor([[1.21, 0.0]]); o = Observations.build(X, torch.tensor([1]), STD)
        self.assertGreater(float(obs_loglik(DISC, o, "soft_margin")), float(obs_loglik(DISC, o, "flip")))
    def test_repeats_accumulate(self):
        X = torch.tensor([[0.2, 0.0]]); o1 = Observations.build(X, torch.tensor([1]), STD); o2 = o1.add(X, torch.tensor([1]), STD)
        self.assertAlmostEqual(float(obs_loglik(DISC, o2)), 2 * float(obs_loglik(DISC, o1)), places=6)
    def test_fit_robust_recovers(self):
        p, ll = fit_robust(DISC.skeleton(), self.obs); self.assertGreater(float((predict(p, X0(500, 9)) == predict(DISC, X0(500, 9))).float().mean()), 0.93)
    def test_fit_init(self):
        p, ll = fit_robust(DISC.skeleton(), self.obs, init=DISC.param_vector(), steps=20); self.assertEqual(p.skeleton(), DISC.skeleton())


def _make_fit_noise_test(rate):
    def t(self):
        o = obs_for(DISC, 240, rate, 3); p, ll = fit_robust(DISC.skeleton(), o); Q = X0(600, 5)
        self.assertGreater(float((predict(p, Q) == predict(DISC, Q)).float().mean()), 0.88 if rate >= 0.12 else 0.92)
    return t


for _r in NOISE_LEVELS:
    setattr(LikelihoodTests, f"test_fit_under_noise_{int(_r * 100)}", _make_fit_noise_test(_r))


def _make_lik_order_test(kind, s):
    def t(self):
        o = obs_for(HALF2, 150, 0.08, s); self.assertGreater(float(obs_loglik(HALF2, o, kind)), float(obs_loglik(DISC, o, kind)))
    return t


for _k in LIKELIHOODS:
    for _s in range(5):
        setattr(LikelihoodTests, f"test_order_{_k}_{_s}", _make_lik_order_test(_k, _s))


# ================================================================ E/F — evidence, MDL

class EvidenceTests(unittest.TestCase):
    def setUp(self): self.obs = obs_for(DISC)
    def test_evidence_penalizes_params(self):
        ll = -50.0; self.assertGreater(log_evidence(HALF, self.obs, ll), log_evidence(P("and", children=(DISC, HALF)), self.obs, ll))
    def test_structure_bits_monotone(self): self.assertLess(structure_bits(HALF), structure_bits(P("and", children=(DISC, HALF))))
    def test_mdl_fields(self): self.assertEqual(set(mdl(DISC, self.obs)), {"structure_bits", "parameter_bits", "residual_bits", "measurement_error_bits", "total"})
    def test_mdl_total_sum(self): m = mdl(DISC, self.obs); self.assertAlmostEqual(m["total"], m["structure_bits"] + m["parameter_bits"] + m["residual_bits"] + m["measurement_error_bits"], places=6)
    def test_mdl_naive_exceptions(self): m = mdl(DISC, self.obs, "naive"); self.assertGreater(m["residual_bits"], 0); self.assertEqual(m["measurement_error_bits"], 0.0)
    def test_mdl_true_shorter(self): self.assertLess(mdl(DISC, self.obs)["total"], mdl(HALF, self.obs)["total"])
    def test_simple_noise_beats_complex(self):
        cx = P("or", children=(DISC, T(P("pred", "norm", "le", (0.05,)), (1.9, 1.9)))); self.assertLess(mdl(DISC, self.obs)["total"], mdl(cx, self.obs)["total"])
    def test_bic_aic(self): ll = float(obs_loglik(DISC, self.obs)); self.assertGreater(bic(DISC, self.obs, ll), aic(DISC, self.obs, ll) - 1e-9)
    def test_not_called_bayesian_truth(self): self.assertIn("APPROXIMATE", inspect.getsource(log_evidence))


class CompatibleSetTests(unittest.TestCase):
    def setUp(self): self.obs = obs_for(DISC); self.css = css_of([DISC, HALF, HALF2], self.obs)
    def test_mass_normalized(self): self.assertAlmostEqual(float(self.css.masses().sum()), 1.0, places=9)
    def test_best_is_truth(self): self.assertEqual(structure_hash(self.css.best().program), structure_hash(DISC))
    def test_answer_structure(self): self.assertNotEqual(self.css.answer(), "UNKNOWN")
    def test_unknown_when_ambiguous(self):
        o = Observations.build(X0(2), torch.tensor([1, 0]), STD); c = css_of([HALF, HALF2, DISC], o); c.unknown_threshold = 0.9; self.assertEqual(c.answer(), "UNKNOWN")
    def test_unknown_empty(self): self.assertEqual(CompatibleStructureSet().answer(), "UNKNOWN")
    def test_dedup_structure(self): c = css_of([DISC, DISC], self.obs); self.assertEqual(len(c.entries), 1)
    def test_entropy_nonneg(self): self.assertGreaterEqual(self.css.entropy(), 0)
    def test_entropy_decreases_with_data(self):
        small = Observations.build(self.obs.X[:6], self.obs.y[:6], STD); self.assertGreater(css_of([DISC, HALF, HALF2], small).entropy(), self.css.entropy())
    def test_predictive_range(self): p = self.css.predictive(X0(50)); self.assertTrue(bool(((p >= 0) & (p <= 1)).all()))
    def test_expected_accuracy(self): self.assertTrue(0.5 <= self.css.expected_accuracy(X0(100)) <= 1.0)
    def test_functional_entropy(self): self.assertGreaterEqual(self.css.functional_entropy(X0(100)), 0)
    def test_rescore(self): self.css.rescore(self.obs); self.assertAlmostEqual(float(self.css.masses().sum()), 1.0, places=9)
    def test_prune(self): self.css.prune(max_entries=2); self.assertLessEqual(len(self.css.entries), 2)
    def test_parameter_samples(self):
        self.css.add_parameter_samples(self.obs, "contamination", 0.02); self.assertTrue(any(e.variant > 0 for e in self.css.entries)); self.assertEqual(self.css.best().variant, 0)
    def test_structure_masses(self): self.css.add_parameter_samples(self.obs, "contamination", 0.02); self.assertAlmostEqual(sum(self.css.structure_masses().values()), 1.0, places=6)
    def test_entry_fields(self):
        e = self.css.best()
        for f in ("mass", "program", "evidence", "complexity", "stability"): self.assertTrue(hasattr(e, f))
    def test_parameter_posterior(self): pp = ParameterPosterior(DISC, self.obs); self.assertEqual(len(pp.sd), 3); self.assertEqual(len(pp.sample(2, G())), 2)
    def test_parameter_posterior_sd_shrinks(self):
        a = ParameterPosterior(DISC, obs_for(DISC, 60)).sd.mean(); b = ParameterPosterior(DISC, obs_for(DISC, 400)).sd.mean(); self.assertLess(float(b), float(a))
    def test_estimate_noise(self):
        X, y = noisy(DISC, 400, 0.17, 3); o = Observations.build(X, y, STD, 0.0); self.assertGreaterEqual(estimate_noise(DISC, o), 0.05)
    def test_estimate_noise_clean(self):
        X = X0(300); o = Observations.build(X, predict(DISC, X), SENSOR_MODELS["precise"]); self.assertLessEqual(estimate_noise(DISC, o), 0.02)


# ================================================================ L — attribution

class AttributionTests(unittest.TestCase):
    def test_labels(self): self.assertEqual(len(DISCREPANCY), 5)
    def test_noise(self): self.assertEqual(attribute_discrepancy(DISC, obs_for(DISC))["label"], "MEASUREMENT_NOISE")
    def test_mismatch(self):
        X = X0(300); o = Observations.build(X, predict(HALF, X), SENSOR_MODELS["precise"]); self.assertIn(attribute_discrepancy(DISC, o)["label"], ("MODEL_MISMATCH", "PARAMETER_UNCERTAINTY"))
    def test_outlier(self):
        X = X0(200); y = predict(DISC, X); i = int((predict(DISC, X) == 1).nonzero()[0]); y[i] = 0
        o = Observations.build(X, y, SENSOR_MODELS["precise"]); self.assertIn(attribute_discrepancy(DISC, o, gamma=0.05)["label"], ("OUTLIER", "MEASUREMENT_NOISE"))
    def test_parameter_uncertainty(self):
        X = X0(400); o = Observations.build(X, predict(T(P("pred", "norm", "le", (1.25,)), (0.2, 0.0)), X), SENSOR_MODELS["precise"]); self.assertEqual(attribute_discrepancy(DISC, o)["label"], "PARAMETER_UNCERTAINTY")
    def test_zero_errors(self): X = X0(50); o = Observations.build(X, predict(DISC, X), STD); self.assertEqual(attribute_discrepancy(DISC, o)["errors"], 0)
    def test_single_bad_obs_no_deepening(self):
        X = X0(150); y = predict(HALF, X); y[0] = 1 - y[0]; o = Observations.build(X, y, STD); r = RobustStructuralReconstruction().run(o); self.assertEqual(r.css.best().program.size(), 1)


# ================================================================ H/I/J/K — expected gains, planner

class GainTests(unittest.TestCase):
    def setUp(self):
        self.obs = Observations.build(X0(12, 2), predict(DISC, X0(12, 2)), STD); self.css = css_of([DISC, HALF, HALF2], self.obs); self.pool, self.probes = X0(80, 4), X0(120, 5)
    def test_efa_nonneg(self): self.assertTrue(bool((expected_functional_accuracy_gain(self.css, self.pool, self.probes) >= -1e-9).all()))
    def test_efa_zero_single(self): self.assertLess(float(expected_functional_accuracy_gain(css_of([DISC], self.obs), self.pool, self.probes).abs().max()), 1e-9)
    def test_sig_nonneg(self): self.assertTrue(bool((structural_ig(self.css, self.pool) >= -1e-9).all()))
    def test_sig_bounded(self): self.assertLessEqual(float(structural_ig(self.css, self.pool).max()), self.css.entropy() + 1e-9)
    def test_fig(self): self.assertTrue(bool((functional_ig(self.css, self.pool, self.probes) >= -1e-6).all()))
    def test_disagreement_zero_where_agree(self):
        x = torch.tensor([[-1.9, 0.0]]); c = css_of([HALF, HALF2], self.obs); self.assertAlmostEqual(float(disagreement(c, x)[0]), 0.0, places=6)
    def test_uncertainty(self): self.assertTrue(bool((uncertainty(self.css, self.pool) >= 0).all()))
    def test_noisier_sensor_less_gain(self):
        c = css_of([DISC, HALF], Observations.build(X0(2), torch.tensor([1, 0]), STD))   # ambiguous posterior (2 distinct structures)
        a = structural_ig(c, self.pool, 0.01).max(); b = structural_ig(c, self.pool, 0.3).max(); self.assertGreater(float(a), float(b))
        a = expected_functional_accuracy_gain(c, self.pool, self.probes, 0.01).max(); b = expected_functional_accuracy_gain(c, self.pool, self.probes, 0.3).max(); self.assertGreaterEqual(float(a), float(b) - 1e-9)
    def test_decision_utility(self): self.assertLess(expected_decision_loss_utility(0.1, 3.0), expected_decision_loss_utility(0.1, 1.0))
    def test_actions(self): self.assertIn("repeat", ACTIONS); self.assertIn("deepen", ACTIONS); self.assertIn("stop", ACTIONS)
    def test_planner_choice(self):
        ch = RobustMeasurementPlanner("efa").choose(self.css, self.pool, None, self.probes, 16.0); self.assertIn(ch.action, ("new_point", "stop"))
    def test_planner_budget(self): ch = RobustMeasurementPlanner("efa", sensors=("precise",)).choose(self.css, self.pool, None, self.probes, 1.0); self.assertEqual(ch.action, "stop")
    def test_planner_stop_when_certain(self): self.assertEqual(RobustMeasurementPlanner("efa").choose(css_of([DISC], self.obs), self.pool, None, self.probes, 16.0).action, "stop")
    def test_planner_repeat_candidate(self):
        ch = RobustMeasurementPlanner("efa", allow_repeat=True).choose(self.css, self.pool[:1], self.pool[1:], self.probes, 16.0); self.assertIn(ch.action, ("new_point", "repeat", "stop"))
    def test_planner_no_repeat(self):
        ch = RobustMeasurementPlanner("disagreement", allow_repeat=False).choose(self.css, self.pool, self.pool, self.probes, 16.0); self.assertEqual(ch.action, "new_point")
    def test_planner_sensor_choice(self):
        ch = RobustMeasurementPlanner("efa", sensors=tuple(SENSOR_MODELS)).choose(self.css, self.pool, None, self.probes, 16.0); self.assertIn(ch.sensor, list(SENSOR_MODELS) + [""])
    def test_planner_does_not_see_answers(self): self.assertNotIn("oracle", inspect.signature(RobustMeasurementPlanner.choose).parameters)
    def test_repeat_informative_under_noise(self):
        c = css_of([DISC, HALF], Observations.build(X0(2), torch.tensor([1, 0]), STD)); x = torch.tensor([[0.4, 0.0]])   # DISC and HALF oppose each other there
        self.assertGreater(float(structural_ig(c, x, 0.2)[0]), 0.0)


def _make_criterion_test(c):
    def t(self):
        obs = Observations.build(X0(12, 2), predict(DISC, X0(12, 2)), STD); css = css_of([DISC, HALF, HALF2], obs)
        sc = RobustMeasurementPlanner(c).scores(css, X0(40, 4), X0(60, 5), 0.05); self.assertEqual(len(sc), 40)
    return t


for _c in ("efa", "structural_ig", "functional_ig", "disagreement", "uncertainty", "random"):
    setattr(GainTests, f"test_criterion_{_c}", _make_criterion_test(_c))


# ================================================================ M/N — reconstruction

class ReconstructionTests(unittest.TestCase):
    def test_recovers_simple(self):
        r = RobustStructuralReconstruction().run(obs_for(HALF2, 150)); Q = X0(500, 7); self.assertGreater(float((predict(r.css.best().program, Q) == predict(HALF2, Q)).float().mean()), 0.93)
    def test_no_overdeepening_under_noise(self): self.assertEqual(RobustStructuralReconstruction().run(obs_for(HALF2, 150, 0.12)).depth, 1)
    def test_status(self): self.assertIn(RobustStructuralReconstruction().run(obs_for(HALF2, 100)).status, ("identified", "unknown", "model_mismatch"))
    def test_trace_and_fields(self):
        r = RobustStructuralReconstruction().run(obs_for(HALF2, 100))
        for f in ("answer", "css", "queries", "cost", "evaluations", "seconds", "depth", "discrepancy"): self.assertTrue(hasattr(r, f))
    def test_passive_zero_queries(self): self.assertEqual(RobustStructuralReconstruction().run(obs_for(HALF2, 100)).queries, 0)
    def test_simplify(self):
        o = obs_for(DISC); cx = P("or", children=(DISC, T(P("pred", "norm", "le", (0.02,)), (1.9, 1.9)))); self.assertEqual(structure_hash(simplify_robust(cx, o)), structure_hash(DISC))
    def test_simplify_keeps_needed(self):
        tr = P("and", children=(DISC, HALF)); X = X0(400); o = Observations.build(X, predict(tr, X), SENSOR_MODELS["precise"]); self.assertEqual(simplify_robust(tr, o).size(), tr.size())
    def test_depth_rules(self): self.assertIn("bayes_factor", inspect.getsource(RobustStructuralReconstruction)); self.assertIn("threshold", inspect.getsource(RobustStructuralReconstruction._should_deepen))
    def test_ppc_accepts_true(self):
        X = X0(120, 8); o = obs_for(DISC, 120, 0.08, 8); self.assertFalse(posterior_predictive_check(DISC, o, 200, G())["reject"])
    def test_ppc_rejects_wrong(self): self.assertTrue(posterior_predictive_check(HALF, obs_for(DISC, 160, 0.05, 8), 200, G())["reject"])
    def test_ppc_fields(self): self.assertEqual(set(posterior_predictive_check(DISC, obs_for(DISC, 60), 50, G())), {"errors", "pairs", "p_errors", "p_pairs", "reject"})


# ================================================================ O — calibration

class CalibrationTests(unittest.TestCase):
    def data(self, n=300):
        rng = random.Random(0); s, y = [], []
        for _ in range(n):
            v = rng.random(); s.append(v); y.append(float(rng.random() < v ** 3))
        return s, y
    def test_metrics_fields(self): s, y = self.data(); self.assertEqual(set(calibration_metrics(s, y)), {"ece", "brier", "reliability", "at_0.5", "at_0.9"})
    def test_perfect(self): self.assertLess(calibration_metrics([0.0, 1.0, 1.0, 0.0], [0, 1, 1, 0])["ece"], 1e-9)
    def test_isotonic_reduces_ece(self): s, y = self.data(); self.assertLess(calibration_metrics(isotonic(s, y)(s), y)["ece"], calibration_metrics(s, y)["ece"])
    def test_platt_runs(self): s, y = self.data(); p = platt(s, y)(s); self.assertTrue(all(0 <= v <= 1 for v in p))
    def test_beta_reduces_ece(self): s, y = self.data(); self.assertLess(calibration_metrics(beta_calibration(s, y)(s), y)["ece"], calibration_metrics(s, y)["ece"])
    def test_precision_recall(self): m = calibration_metrics([0.9, 0.1], [1, 0]); self.assertEqual(m["at_0.5"]["precision"], 1.0)
    def test_calibration_seeds_disjoint_from_test(self): self.assertTrue(all(s < 800000 for s in RE.TRAIN_CAL_SEEDS))
    def test_sealed_not_in_calibration(self): 
        for fn in (RE.calibration_subunits, RE.caltrain_subunits, RE.rule_cases, RE.fit_calibrators): self.assertNotIn("sealed", inspect.getsource(fn)); self.assertNotIn("99200", inspect.getsource(fn))
    def test_calibrators_fit_on_train_only(self):
        src = inspect.getsource(RE.fit_calibrators); self.assertIn("mk(xs, ys)(tp)", src); self.assertTrue(all(r["split"] == "train" for sid, fn in RE.caltrain_subunits(0)[:0] for r in fn()))
        self.assertEqual([sid for sid, _ in RE.caltrain_subunits(5)][-1], "s700059")


# ================================================================ T/U — Fourier, micro-states

class RepresentationTests(unittest.TestCase):
    def sig(self):
        t = torch.arange(128, dtype=torch.float64); return torch.sin(2 * math.pi * 5 * t / 128) + 0.5 * torch.sin(2 * math.pi * 17 * t / 128)
    def test_fourier_sparse(self): r = fourier_reconstruction(self.sig()); self.assertLessEqual(r["k"], 6); self.assertLess(r["rel_error"], 0.15)
    def test_fourier_inverse(self): r = fourier_reconstruction(self.sig(), k=128); self.assertLess(r["rel_error"], 1e-6)
    def test_world_unchanged(self): s = self.sig(); s0 = s.clone(); fourier_reconstruction(s); self.assertTrue(torch.equal(s, s0))
    def test_direct_space_not_sparse(self): self.assertGreater(int((self.sig().abs() > 0.1).sum()), 60)
    def test_microstate_entropy(self): o = obs_for(DISC, 8); self.assertGreater(RR.microstate_entropy(css_of([DISC, HALF, HALF2], o)), 0)
    def test_no_quantum_claim(self):
        src = inspect.getsource(RR); self.assertIn("no quantum mechanics", src); self.assertNotIn("ħ", src)


# ================================================================ Z — world

class WorldTests(unittest.TestCase):
    def test_noise_levels(self): self.assertEqual(NOISE_LEVELS, (0.0, 0.02, 0.05, 0.08, 0.12, 0.15))
    def test_sensors(self): self.assertEqual(set(SENSORS), {"precise", "standard", "cheap"})
    def test_opaque_uid(self):
        w = make_world(3)
        for n in ("disc", "ring", "sector", "wedge", "halfspace", "triangle"): self.assertNotIn(n, w.uid)
    def test_task_shapes(self): t = make_task(1); self.assertEqual(t.X.shape, (120, 2)); self.assertEqual(t.y.shape, (120,))
    def test_missing_present(self): self.assertTrue(any(int((make_task(s, noise=0.05).y == -1).sum()) > 0 for s in range(5)))
    def test_oracle_noisy_not_truth(self):
        t = make_task(2, noise=0.15); X = full_region(2000, G()); self.assertLess(float((t.oracle(X, "cheap", G(1)) == t.world.label(X)).float().mean()), 0.9)
    def test_precise_sensor_better(self):
        t = make_task(2, noise=0.0); X = full_region(3000, G()); a = (t.oracle(X, "precise", G(1)) == t.world.label(X)).float().mean(); b = (t.oracle(X, "cheap", G(1)) == t.world.label(X)).float().mean(); self.assertGreater(float(a), float(b))
    def test_hard_pair(self): a, b = hard_noise_pair(1); self.assertEqual((a.world.depth, b.world.depth), (1, 3)); self.assertEqual((a.spec.rate, b.spec.rate), (0.08, 0.0))
    def test_unrepresentable(self): w = make_world(4, representable=False); self.assertIsNone(w.truth)
    def test_regions(self): self.assertTrue(bool((in_region(100, G())[:, 0] <= 0.5).all())); self.assertTrue(bool((out_region(100, G())[:, 0] >= 0.5).all()))
    def test_sealed_hash(self): self.assertEqual(sealed_hash(), SEALED_SHA256)
    def test_sealed_size(self): self.assertEqual(len(sealed_tasks()), 24)
    def test_sealed_noise_levels(self): self.assertEqual(sealed_spec()["noise"], [0.08, 0.12, 0.15])
    def test_sealed_seeds_disjoint(self): self.assertTrue(all(s >= 992000 for s in sealed_spec()["seeds"]))


def _make_depth_test(d):
    def t(self):
        p = random_program(random.Random(d), d); self.assertEqual(sum(1 for _ in __import__("core.boundary_program", fromlist=["_walk"])._walk(p) if _.kind in ("and", "or", "xor")), d - 1)
    return t


def _make_kind_test(kind):
    def t(self):
        w = make_world(5, depth=1); X = in_region(1500, G()); y = corrupt(w, X, SENSORS["standard"], NoiseSpec(0.1, kind, 0.0), G(1)); self.assertGreater(float((y != w.label(X)).float().mean()), 0.03)
    return t


for _d in (1, 2, 3, 4): setattr(WorldTests, f"test_depth_{_d}", _make_depth_test(_d))
for _k in NOISE_KINDS: setattr(WorldTests, f"test_noise_kind_{_k}", _make_kind_test(_k))


# ================================================================ AX — leaks

class LeakageTests(unittest.TestCase):
    def test_inference_does_not_import_world(self): self.assertNotIn("robust_world", inspect.getsource(RR)); self.assertNotIn("boundary_world", inspect.getsource(RR))
    def test_no_truth_attribute_access(self): self.assertNotIn(".truth", inspect.getsource(RR)); self.assertNotIn("world.label", inspect.getsource(RR))
    def test_no_family_conditionals(self): self.assertIsNone(re.search(r"family\s*==", inspect.getsource(RR)))
    def test_run_signature_no_labels(self): self.assertNotIn("truth", inspect.signature(RobustStructuralReconstruction.run).parameters)
    def test_monitor_not_used_for_decisions(self):
        src = inspect.getsource(RobustStructuralReconstruction.run); self.assertNotIn("if monitor(css)", src); self.assertIn("never used to decide", src)
    def test_oracle_policy_marked(self): self.assertIn("upper bound", inspect.getsource(RE._Oracle))
    def test_task_oracle_goes_through_noise(self): self.assertIn("corrupt", inspect.getsource(RW.Task.oracle))
    def test_prereg_exists(self):
        t = open("docs/hypotheses_robust_reconstruction.md").read()
        for i in range(1, 11): self.assertIn(f"H{i} ", t)
        self.assertIn("16", t)
    def test_sealed_only_in_sealed_job(self):
        for name, fn in RE.SUBUNITS.items(): self.assertNotIn("sealed", inspect.getsource(fn), name)
        self.assertNotIn("sealed", RE.SUBUNITS); self.assertEqual(inspect.getsource(RE).count("sealed_tasks"), 2)
    def test_no_cache_of_truth(self): self.assertNotIn("lru_cache", inspect.getsource(RE))
    def test_metadata_no_family_name(self): t = make_task(1); self.assertFalse(hasattr(t.world, "family"))


# ================================================================ AWS — pure orchestration (no network call)

class OrchestrationTests(unittest.TestCase):
    def test_region(self): self.assertEqual(ORC.REGION, "eu-north-1")
    def test_bucket(self): self.assertEqual(ORC.BUCKET, "wce-research")
    def test_tags(self):
        d = {t["Key"]: t["Value"] for t in ORC.TAGS}; self.assertEqual(d["Project"], "WCE"); self.assertEqual(d["Purpose"], "Research"); self.assertEqual(d["ManagedBy"], "WCE"); self.assertEqual(d["Component"], "RobustReconstruction")
    def test_identity_ok(self): self.assertTrue(ORC.check_identity("arn:aws:sts::1:assumed-role/AWSReservedSSO_WCE-Research-Runner_x/wce-runner"))
    def test_identity_admin_rejected(self):
        with self.assertRaises(SystemExit): ORC.check_identity("arn:aws:sts::1:assumed-role/AWSReservedSSO_AdministratorAccess_x/me")
    def test_identity_other_rejected(self):
        with self.assertRaises(SystemExit): ORC.check_identity("arn:aws:iam::1:user/bob")
    def test_parse_keys(self): self.assertEqual(ORC.parse_keys("0-3,7"), ["0", "1", "2", "3", "7"])
    def test_processing_spec(self):
        s = ORC.job_spec("n", "arn:role", [{"kind": "smoke", "key": "0"}], "s3://b/c/", "s3://b/o/", "ml.m5.large", 600, "r")
        self.assertEqual(s["ProcessingResources"]["ClusterConfig"]["InstanceType"], "ml.m5.large"); self.assertEqual(s["Tags"], ORC.TAGS); self.assertIn("wce_aws.sm_entry", s["AppSpecification"]["ContainerEntrypoint"][2])
    def test_processing_spec_serializable(self): json.dumps(ORC.job_spec("n", "r", [], "s3://a/", "s3://b/", "ml.m5.large", 60, "r"))
    def test_training_spec(self): s = ORC.training_spec("n", "r", [], "s3://a/", "s3://b/", "ml.m5.large", 60, "r"); self.assertEqual(s["ResourceConfig"]["InstanceCount"], 1); self.assertEqual(s["Tags"], ORC.TAGS)
    def test_ec2_script_auto_terminates(self): sc = ORC.ec2_job_script([], "s3://a/", "s3://b/", "c5.xlarge", "r"); self.assertIn("trap 'shutdown -h now' EXIT", sc)
    def test_ec2_script_syncs_to_bucket(self): self.assertTrue(any("aws s3 sync" in c for c in ORC.ec2_job_script([], "s3://a/", "s3://wce-research/x/", "c5.xlarge", "r")))
    def test_ec2_user_data_deadman(self): self.assertIn("shutdown -h +", ORC.ec2_user_data(3600))
    def test_ec2_profile(self): self.assertEqual(ORC.EC2_PROFILE, "WCE-EC2-ExecutionRole")
    def test_ec2_tags(self): self.assertIn({"Key": "Project", "Value": "WCE"}, ORC.EC2_TAGS)
    def test_ec2_no_ssh(self):
        src = inspect.getsource(ORC); self.assertNotIn("--key-name", src); self.assertNotIn("ssh ", src)
    def test_ec2_terminate_behavior(self): self.assertIn("--instance-initiated-shutdown-behavior", inspect.getsource(ORC.ec2_launch)); self.assertIn('"terminate"', inspect.getsource(ORC.ec2_launch))
    def test_backend_explicit(self): self.assertIn('os.environ.get("WCE_BACKEND", "processing")', inspect.getsource(ORC))
    def test_no_iam_mutation(self):
        src = inspect.getsource(ORC)
        for bad in ("create-role", "put-role-policy", "attach-role-policy", "create-user", "create-policy", "organizations", "identitystore"): self.assertNotIn(bad, src)
    def test_only_iam_call_is_read(self):
        self.assertEqual(re.findall(r'aws\("iam", "([a-z-]+)"', inspect.getsource(ORC)), ["list-roles"]); self.assertTrue(all(p[0].startswith("list-") for p in ORC.GUARDRAIL_IAM_PROBES))
    def test_guardrails_fail_if_iam_allowed(self):
        old = ORC.aws; ORC.aws = lambda *a, **k: {"Arn": "arn:aws:sts::1:assumed-role/AWSReservedSSO_WCE-Research-Runner_x/u", "Account": "1"}
        try:
            with self.assertRaises(SystemExit) as c: ORC.check_guardrails()
        finally: ORC.aws = old
        self.assertEqual(c.exception.code, 4)
    def test_guardrails_ok_when_iam_denied(self):
        def fake(*a, **k):
            if a[0] == "iam": raise ORC.AwsDenied("denied")
            return {"Arn": "arn:aws:sts::1:assumed-role/AWSReservedSSO_WCE-Research-Runner_x/u", "Account": "1"}
        old = ORC.aws; ORC.aws = fake
        try: r = ORC.check_guardrails()
        finally: ORC.aws = old
        self.assertEqual(r["guardrails"], "OK")
    def test_guardrails_fail_if_admin(self):
        def fake(*a, **k):
            if a[0] == "iam": raise ORC.AwsDenied("denied")
            return {"Arn": "arn:aws:sts::1:assumed-role/AWSReservedSSO_AdministratorAccess_x/u", "Account": "1"}
        old = ORC.aws; ORC.aws = fake
        try:
            with self.assertRaises(SystemExit): ORC.check_guardrails()
        finally: ORC.aws = old
    def test_whoami_rejects_admin(self):
        old = ORC.aws; ORC.aws = lambda *a, **k: {"Arn": "arn:aws:iam::1:user/admin", "Account": "1"}
        try:
            with self.assertRaises(SystemExit): ORC.whoami()
        finally: ORC.aws = old
    def test_login_device_code(self):
        mk = open("Makefile").read(); self.assertIn("aws sso login --profile $(AWS_PROFILE) --use-device-code", mk); self.assertIn("aws-whoami:", mk); self.assertIn("aws-check-guardrails:", mk)
    def test_clean_safe_checks_tags(self): self.assertIn('"Project"', inspect.getsource(ORC.clean_safe)); self.assertIn("smoke/", inspect.getsource(ORC.clean_safe))
    def test_tarball(self): data, h = ORC.build_tarball(); self.assertGreater(len(data), 10000); self.assertEqual(len(h), 16)
    def test_tarball_deterministic_hash(self): self.assertEqual(ORC.build_tarball()[1], ORC.build_tarball()[1])
    def test_tarball_has_no_results(self):
        import io, tarfile
        names = tarfile.open(fileobj=io.BytesIO(ORC.build_tarball()[0])).getnames(); self.assertTrue(all(not n.startswith("results/aws") for n in names)); self.assertIn("core/robust_reconstruction.py", names)
    def test_denied_class(self): self.assertTrue(issubclass(ORC.AwsDenied, RuntimeError))
    def test_makefile_targets(self):
        mk = open("Makefile").read()
        for t in ("aws-smoke:", "aws-package:", "aws-status:", "aws-logs:", "aws-fetch-results:", "aws-clean-safe:", "aws-noise-sweep:", "aws-sealed-eval:", "aws-login:", "robust-aggregate:"): self.assertIn(t, mk)
    def test_makefile_profile(self): self.assertIn("AWS_PROFILE ?= wce-runner", open("Makefile").read())
    def test_entry_serialization(self): self.assertEqual(json.loads(ENT.serialize_units([{"kind": "main", "key": "3"}])), [{"kind": "main", "key": "3"}])
    def test_entry_smoke_unit(self):
        import tempfile
        d = tempfile.mkdtemp(); old = ENT.OUT; ENT.OUT = d
        try: r = ENT.run_unit({"kind": "smoke", "key": "0"}); rec = json.load(open(os.path.join(d, "smoke_0.json")))
        finally: ENT.OUT = old
        self.assertEqual(r["status"], "ok"); self.assertIn("torch", rec["result"][0]); self.assertEqual(len(rec["result"]), 4)
    def test_entry_error_captured(self):
        import tempfile
        d = tempfile.mkdtemp(); old = ENT.OUT; ENT.OUT = d
        try: r = ENT.run_unit({"kind": "does_not_exist", "key": "0"})
        finally: ENT.OUT = old
        self.assertEqual(r["status"], "error")


class AggregateTests(unittest.TestCase):
    def test_ci(self): c = ci([1, 2, 3]); self.assertEqual(c["mean"], 2.0)
    def test_ci_empty(self): self.assertIsNone(ci([]))
    def test_spearman(self): self.assertAlmostEqual(spearman([1, 2, 3, 4], [2, 4, 6, 9]), 1.0)
    def test_spearman_neg(self): self.assertAlmostEqual(spearman([1, 2, 3, 4], [9, 6, 4, 2]), -1.0)
    def test_aggregate_empty(self): self.assertEqual(aggregate({}, [])["hypotheses"], {})
    def test_aggregate_capacity(self):
        row = {"params": 3, "ast": 2, "raw_points": 5, "violated": False, "none": (0.9, 0.9, 0), "raw_gated": (0.9, 0.9, 10), "prototype": (0.9, 0.9, 8)}
        S = aggregate({"capacity": {"0": {"result": [row]}}}, []); self.assertEqual(S["hypotheses"]["H9"]["verdict"], "SOUTENUE")
    def test_aggregate_world_depth_from_order(self):
        rows = [{"method": m, "representable": d > 0, "depth": 9, "recovery": float(d == 2), "structure": 0, "extrapolation": 0.5, "in_region": 0.5, "queries": 0, "evaluations": 1, "seconds": 1.0} for nz, d in RE.MAIN_SPECS for m in RE.METHODS]
        S = aggregate({"main": {"0": {"result": rows}}}, []); self.assertEqual(S["main_by_depth"]["full"]["2"]["mean"], 1.0); self.assertEqual(S["main_by_depth"]["full"]["3"]["mean"], 0.0)
    def test_methods(self): self.assertEqual(len(RE.METHODS), 10)
    def test_jobs(self): self.assertEqual(set(RE.JOBS), {"main", "noise", "hard", "repeat", "calibration", "caltrain", "family", "capacity", "identifiability"})
    def test_budget_preregistered(self): self.assertEqual(RE.BUDGET, 16.0); self.assertEqual((RE.COMPACT_PARAMS, RE.COMPACT_POINTS), (64, 32))


# ================================================================ observability, checkpoints, resume

class _OutDir:
    def __enter__(self):
        import tempfile
        self.old = ENT.OUT; ENT.OUT = tempfile.mkdtemp(); return ENT.OUT
    def __exit__(self, *a): ENT.OUT = self.old


class CheckpointTests(unittest.TestCase):
    U = {"kind": "smoke", "key": "0"}
    def test_subunit_counts(self): self.assertEqual({k: len(RE.subunit_ids(k, 0)) for k in ("main", "noise", "hard", "repeat", "calibration", "caltrain", "family", "capacity", "identifiability", "sealed")},
                                                    {"main": 100, "noise": 72, "hard": 4, "repeat": 8, "calibration": 6, "caltrain": 10, "family": 3, "capacity": 4, "identifiability": 4, "sealed": 5})
    def test_subunit_ids_unique(self):
        for k in RE.ALL_SUBUNITS: ids = RE.subunit_ids(k, 3); self.assertEqual(len(ids), len(set(ids)), k)
    def test_subunit_deterministic(self): self.assertEqual(RE.run_subunit("smoke", 0, "p2"), RE.run_subunit("smoke", 0, "p2"))
    def test_run_job_uses_done(self): self.assertEqual(RE.run_job("smoke", 0, {"p0": ["x"], "p1": ["y"], "p2": [], "p3": ["z"]}), ["x", "y", "z"])
    def test_main_task_matches_list(self):
        for k in (0, 4, 7): self.assertEqual(RE.main_task(2, k)[1].world.uid, RE.main_tasks(2)[k][1].world.uid)
    def test_main_specs(self): self.assertEqual(len(RE.MAIN_SPECS), 10); self.assertEqual(sum(1 for nz, d in RE.MAIN_SPECS if d == 0), 2)
    def test_noise_seed_not_process_dependent(self): self.assertNotIn("hash(kind)", inspect.getsource(RE._noise_sub).split("#")[0]); self.assertIn("NOISE_SWEEP_KINDS.index(kind)", inspect.getsource(RE._noise_sub))
    def test_checkpoint_written(self):
        with _OutDir() as d:
            ENT.run_unit(self.U); lines = open(os.path.join(d, "ckpt_smoke_0.jsonl")).read().splitlines(); self.assertEqual(len(lines), 4); self.assertEqual(json.loads(lines[0])["sub"], "p0")
    def test_resume_skips_done(self):
        with _OutDir() as d:
            ENT.run_unit(self.U); lines = open(os.path.join(d, "ckpt_smoke_0.jsonl")).read().splitlines()
            open(os.path.join(d, "ckpt_smoke_0.jsonl"), "w").write("\n".join(lines[:2]) + "\n"); sh = ENT.run_units([self.U], pool=False)
            self.assertEqual((sh["subunits_run"], sh["subunits_resumed"]), (2, 2)); self.assertEqual(sh["units"][0]["status"], "ok")
    def test_resume_same_result(self):
        with _OutDir() as d:
            ENT.run_unit(self.U); full = json.load(open(os.path.join(d, "smoke_0.json")))["result"]; lines = open(os.path.join(d, "ckpt_smoke_0.jsonl")).read().splitlines()
            open(os.path.join(d, "ckpt_smoke_0.jsonl"), "w").write(lines[3] + "\n" + lines[1] + "\n"); os.remove(os.path.join(d, "smoke_0.json")); ENT.run_unit(self.U)
            self.assertEqual(json.load(open(os.path.join(d, "smoke_0.json")))["result"], full)
    def test_truncated_checkpoint_line_ignored(self):
        with _OutDir() as d:
            ENT.run_unit(self.U); f = os.path.join(d, "ckpt_smoke_0.jsonl"); lines = open(f).read().splitlines(); open(f, "w").write(lines[0] + "\n" + lines[1][:17])
            self.assertEqual(list(ENT.load_ckpt("smoke", "0")), ["p0"]); self.assertEqual(ENT.run_units([self.U], pool=False)["subunits_run"], 3)
    def test_full_resume_runs_nothing(self):
        with _OutDir():
            ENT.run_unit(self.U); sh = ENT.run_units([self.U], pool=False); self.assertEqual(sh["subunits_run"], 0); self.assertEqual(sh["units"][0]["status"], "ok")
    def test_progress_fields(self):
        with _OutDir() as d:
            ENT.run_unit(self.U); g = json.load(open(os.path.join(d, "_progress.json")))
        for f in ("state", "step", "items_done", "items_total", "elapsed_s", "eta_s", "cpu_avg_pct", "cpu_peak_pct", "mem_available_min_mb", "last_checkpoint", "in_flight", "load_avg", "processes", "vcpus"): self.assertIn(f, g)
        self.assertEqual((g["state"], g["items_done"], g["items_total"]), ("done", 4, 4)); self.assertEqual(g["last_checkpoint"]["sub_id"], "smoke_0:p3")
    def test_progress_multi_unit(self):
        with _OutDir() as d:
            ENT.run_units([self.U, {"kind": "smoke", "key": "1"}], pool=False); g = json.load(open(os.path.join(d, "_progress.json")))
        self.assertEqual(g["units"], {"smoke_0": {"done": 4, "total": 4}, "smoke_1": {"done": 4, "total": 4}})
    def test_error_not_checkpointed(self):
        with _OutDir() as d:
            sh = ENT.run_units([{"kind": "does_not_exist", "key": "0"}], pool=False); self.assertEqual(sh["units"][0]["status"], "error"); self.assertFalse(os.path.exists(os.path.join(d, "ckpt_does_not_exist_0.jsonl")))
    def test_subunit_error_reported(self):
        old = RE.ALL_SUBUNITS["smoke"]; RE.ALL_SUBUNITS["smoke"] = lambda key: [("a", lambda: [1]), ("b", lambda: 1 / 0)]
        try:
            with _OutDir() as d:
                sh = ENT.run_units([self.U], pool=False); rec = json.load(open(os.path.join(d, "smoke_0.json"))); g = json.load(open(os.path.join(d, "_progress.json")))
        finally: RE.ALL_SUBUNITS["smoke"] = old
        self.assertEqual(rec["status"], "error"); self.assertEqual(rec["result"]["missing"], ["b"]); self.assertIn("ZeroDivisionError", rec["result"]["error"]); self.assertEqual((g["state"], g["items_failed"]), ("failed", 1))
    def test_unit_seconds_recorded(self):
        with _OutDir() as d:
            ENT.run_unit(self.U); rec = json.load(open(os.path.join(d, "smoke_0.json")))
        self.assertEqual(set(rec["subunits"]), {"p0", "p1", "p2", "p3"}); self.assertIn("cpu_s", rec["subunits"]["p0"])
    def test_shard_summary_fields(self):
        with _OutDir():
            sh = ENT.run_units([self.U], pool=False)
        for f in ("wall_s", "procs", "work_s", "cpu_s", "parallel_efficiency", "peak_rss_mb", "cpu_avg_pct", "subunits_run", "subunits_resumed"): self.assertIn(f, sh)
    def test_atomic_write(self):
        with _OutDir() as d:
            ENT._write(os.path.join(d, "x.json"), {"a": 1}); self.assertEqual(os.listdir(d), ["x.json"])
    def test_stack_reports_project_frames(self):
        import threading
        st = ENT._stack(threading.main_thread().ident, depth=50); self.assertTrue(all(":" in x for x in st))
    def test_cpu_sampler(self):
        s = ENT.CpuSampler(0.05); s.start(); sum(i * i for i in range(300000)); import time as _t; _t.sleep(0.2); s.stop(); m = s.summary(); self.assertIn("cpu_avg_pct", m)
        if m["samples"]: self.assertTrue(0 <= m["cpu_avg_pct"] <= 100.0001)
    def test_one_torch_thread_per_process(self): self.assertIn('os.environ.get("WCE_TORCH_THREADS", "1")', inspect.getsource(ENT.run_sub))
    def test_pool_is_spawn_over_subunits(self): src = inspect.getsource(ENT.run_units); self.assertIn('get_context("spawn")', src); self.assertIn("imap_unordered(run_sub, todo", src)
    def test_ec2_script_resumes_and_syncs(self):
        sc = ORC.ec2_job_script([], "s3://a/", "s3://b/o/", "c6i.large", "r"); self.assertTrue(any("--include 'ckpt_*.jsonl'" in c and c.startswith("aws s3 sync s3://b/o/ /opt/wce/out") for c in sc))
        self.assertTrue(any("while true" in c and "aws s3 sync /opt/wce/out s3://b/o/" in c for c in sc)); self.assertEqual(sc[-1], "exit $rc")
    def test_ec2_script_valid_bash(self):
        import subprocess, tempfile
        f = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False); f.write("\n".join(ORC.ec2_job_script([{"kind": "main", "key": "0"}], "s3://a/", "s3://b/", "c6i.large", "r"))); f.close()
        self.assertEqual(subprocess.run(["bash", "-n", f.name]).returncode, 0)
    def test_processing_continuous_upload(self): self.assertEqual(ORC.job_spec("n", "r", [], "s3://a/", "s3://b/", "ml.m5.large", 60, "r")["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3UploadMode"], "Continuous")
    def test_stop_checks_tags(self): src = inspect.getsource(ORC.stop_run); self.assertIn('tags.get("Project") == "WCE"', src); self.assertIn("cancel-command", src)
    def test_resume_same_prefix(self): src = inspect.getsource(ORC.resume); self.assertIn('j["output"]', src); self.assertIn('j.get("status") == "Completed"', src)
    def test_manifest_saved_per_launch(self): self.assertIn("save(); continue", inspect.getsource(ORC.submit))
    def test_prices_only_allowed_types(self): self.assertEqual(set(ORC.PRICE_USD_H), {"t3.medium", "t3.large", "c6i.large", "c6i.xlarge", "c7i.xlarge", "c7i-flex.xlarge", "c8i.xlarge"})
    def test_make_targets_observability(self):
        mk = open("Makefile").read()
        for t in ("aws-progress:", "aws-resume:", "aws-stop:", "aws-inspect:", "aws-profile:", "aws-bench:", "aws-timing:", "aws-run-caltrain:", "aws-ec2-diagnose:"): self.assertIn(t, mk)


class CacheTests(unittest.TestCase):
    """Caching n_params / own_count changes no value and does not change node identity."""
    NODES = [DISC, HALF, HALF2, P("and", children=(DISC, HALF)), P("not", children=(HALF,)), P("pred", "dot", "in", (0.1, -0.2, 0.3)), P("pred", "norm", "in", (0.3, 0.9)),
             P("pred", "angle", "in", (0.2, 0.7)), P("rotate", params=(0.3,), children=(HALF,)), P("scale", params=(1.5,), children=(DISC,)), P("xor", children=(P("or", children=(DISC, HALF)), HALF2))]
    def ref(self, n):
        own = {("pred", "dot", "le"): 2, ("pred", "dot", "ge"): 2, ("pred", "dot", "in"): 3, ("pred", "norm", "le"): 1, ("pred", "norm", "ge"): 1, ("pred", "norm", "in"): 2, ("pred", "angle", "in"): 2}.get((n.kind, n.feat, n.op), 0)
        return own + {"translate": 2, "rotate": 1, "scale": 1}.get(n.kind, 0)
    def test_own_count_matches_reference(self):
        for n in self.NODES: self.assertEqual(n.own_count(), self.ref(n)); self.assertEqual(n.own_count(), n.own_count())
    def test_n_params_matches_param_vector(self):
        for n in self.NODES: self.assertEqual(n.n_params(), len(n.param_vector())); self.assertEqual(n.skeleton().n_params(), n.n_params())
    def test_cache_not_in_equality_or_hash(self):
        a = P("pred", "norm", "le", (1.0,)); b = P("pred", "norm", "le", (1.0,)); a.n_params(); self.assertEqual(a, b); self.assertEqual(hash(a), hash(b)); self.assertEqual(structure_hash(a), structure_hash(b))
    def test_cache_survives_pickle(self):
        import pickle
        n = P("and", children=(DISC, HALF)); n.n_params(); m = pickle.loads(pickle.dumps(n)); self.assertEqual(m, n); self.assertEqual(m.n_params(), 5)
    def test_with_params_roundtrip(self):
        for n in self.NODES: self.assertEqual(n.skeleton().with_params(n.param_vector()), n)
    def test_evaluate_unchanged_by_cache(self):
        X = X0(64); n = P("xor", children=(DISC, HALF2)); a = predict(n, X); n.n_params(); n.own_count(); self.assertTrue(torch.equal(a, predict(n, X)))
    def test_aggregate_ignores_test_keys(self):
        from core.robust_aggregate import CAMPAIGN_KEYS
        self.assertNotIn(900, CAMPAIGN_KEYS["main"]); self.assertEqual(len(CAMPAIGN_KEYS["sealed"]), 24); self.assertNotIn("profile", CAMPAIGN_KEYS); self.assertNotIn("bench", CAMPAIGN_KEYS)
    def test_hard_stop_option(self): self.assertIn("hard", inspect.signature(ORC.stop_run).parameters); self.assertIn("$(if $(HARD),--hard,)", open("Makefile").read())


class BenchTests(unittest.TestCase):
    def setUp(self):
        from wce_aws import bench
        self.B = bench
    def test_categories(self):
        c = self.B.category; self.assertEqual(c("~", "<built-in method torch.sigmoid>"), "torch_noyaux_natifs"); self.assertEqual(c("/x/core/robust_reconstruction.py", "fit_robust"), "python_projet")
        self.assertEqual(c("/usr/lib/python3/site-packages/torch/optim/adam.py", "step"), "torch_couche_python"); self.assertEqual(c("~", "<built-in method posix.fsync>"), "es_serialisation")
        self.assertEqual(c("/usr/lib/python3.11/random.py", "uniform"), "python_stdlib_autre")
    def test_summarize(self):
        import cProfile
        pr = cProfile.Profile(); pr.enable(); float(predict(DISC, X0(50)).sum()); pr.disable(); r = self.B.summarize(pr)
        self.assertAlmostEqual(sum(r["self_time_by_category_pct"].values()), 100.0, delta=1.0); self.assertTrue(r["top_self"]); self.assertTrue(any("predict" in x["func"] for x in r["project_phases_by_cumulative"]))
    def test_profile_subunits(self): self.assertEqual([s for s, _ in self.B.profile_subunits("1")], list(RE.METHODS))
    def test_bench_workload_is_small(self): self.assertLessEqual(len(self.B.BENCH_WORKLOAD), 3); self.assertTrue(all(w[2] in RE.subunit_ids(w[0], w[1]) for w in self.B.BENCH_WORKLOAD))
    def test_bench_phases(self): src = inspect.getsource(self.B.bench_sub); self.assertIn("A_sequentiel_1_thread", src); self.assertIn("C_processus_paralleles", src)
    def test_special_ids(self): self.assertEqual(ENT.sub_ids("bench", "0"), ["all"]); self.assertEqual(len(ENT.sub_ids("profile", "1")), 10)
    def test_fit_calibrators(self):
        tr = [{"p_wrong": i / 20, "wrong": i > 12} for i in range(20)]; r = RE.fit_calibrators(tr, tr); self.assertEqual((r["n_train"], r["n_test"]), (20, 20)); self.assertIn("ece", r["isotonic"])


if __name__ == "__main__":
    unittest.main()

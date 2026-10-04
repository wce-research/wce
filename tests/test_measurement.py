import ast
import copy
import math
import os
import random
import unittest

import torch

from core.measurement import (
    CONFLICTED, DESTRUCTIVE, PASSIVE, WEAKLY_DISTURBING, WORLD_FRAME, BasisTransform, CompatibleStateSet, FrameError,
    FramedValue, LatentWorldState, MeasurementContext, MeasurementDistribution, MeasurementInvariant, MeasurementOp,
    MeasurementOperator, MeasurementOutcome, MeasurementPlanner, MeasurementRecord, MeasurementResolution,
    MeasurementTradeoff, NoiseModel, Observable, ReferenceFrame, RepresentationBasis, UnitError, change_frame,
    check_invariants, decision_value, distance, expected_posterior_entropy, fuse, information_gain, new_graph, trace,
)
from core.measurement_experiments import (
    CANDIDATE_FREQS, fourier_experiment, frame_experiment, macro_left_count, macrostate_experiment, macrostate_set,
    make_signal, raw_reasoning, spectral_reasoning, tradeoff_experiment,
)
from core.transformations import TransformationDomain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOURIER = FRAMES = MACRO = TRADE = None


def setUpModule():
    global FOURIER, FRAMES, MACRO, TRADE
    FOURIER = fourier_experiment(80); FRAMES = frame_experiment(80); MACRO = macrostate_experiment(10); TRADE = tradeoff_experiment(200)


def two_cubes(graph=None):
    dom = {(e, a): d for e in ("e1", "e2") for a, d in (("loc", ["table", "shelf"]), ("color", ["red", "blue"]))}
    return CompatibleStateSet.product(dom, graph=graph)


def op(entity="e1", attr="color", dom=("red", "blue"), r=0.9, cost=0.1, **kw):
    return MeasurementOperator(Observable(attr, entity, attr, tuple(dom)), MeasurementContext(sensor="s", noise=NoiseModel("confusion", r), cost=cost, **kw))


def rec(entity="e1", attr="color", dom=("red", "blue"), value="red", r=0.9, prov="s"):
    o = op(entity, attr, dom, r)
    return MeasurementRecord(o.observable, o.context, MeasurementOutcome(value), prov)


# ================================================================ séparation état / observable / opérateur / issue

class SeparationTests(unittest.TestCase):
    def test_observable_has_no_value(self): self.assertFalse(hasattr(Observable("loc", "b", "loc", ("t",)), "value"))
    def test_observable_str(self): self.assertEqual(str(Observable("position", "ball", "loc")), "loc(ball)")
    def test_observable_discrete(self): self.assertTrue(Observable("c", "e", "color", ("r",)).discrete)
    def test_observable_continuous(self): self.assertFalse(Observable("h", "e", "height").discrete)
    def test_context_fields(self):
        c = MeasurementContext()
        for f in ("observer", "sensor", "frame", "time", "resolution", "noise", "cost", "disturbance", "domain", "latency"): self.assertTrue(hasattr(c, f))
    def test_same_observable_different_context(self):
        a, b = op(r=0.99), op(r=0.6)
        self.assertNotEqual(float(a.distribution("red").probs[0]), float(b.distribution("red").probs[0]))
    def test_operator_returns_distribution(self): self.assertIsInstance(op().distribution("red"), MeasurementDistribution)
    def test_distribution_normalized(self): self.assertTrue(op().distribution("red").normalized())
    def test_distribution_mass_on_truth(self): d = op(r=0.9).distribution("red"); self.assertAlmostEqual(float(d.probs[0]), 0.9)
    def test_credible_region(self): self.assertEqual(op(r=0.95).distribution("red").credible_region(0.9), ["red"])
    def test_credible_region_wide(self): self.assertEqual(len(op(r=0.6).distribution("red").credible_region(0.9)), 2)
    def test_likelihood_vectorized(self):
        L = op(r=0.9).likelihood("red", torch.tensor([0, 1, 0])); self.assertEqual(L.shape, (3,))
    def test_record_fields(self):
        r = rec(); self.assertEqual((r.outcome.value, r.provenance), ("red", "s"))
    def test_latent_distinct_from_records(self):
        w = LatentWorldState({("e1", "color"): "red"}); r = w.measure(op(r=1.0)); self.assertEqual(r.outcome.value, "red")
        self.assertIsNot(r, w._values)
    def test_measurement_noise_can_mislead(self):
        w = LatentWorldState({("e1", "color"): "red"}, seed=1)
        outs = {w.measure(op(r=0.5)).outcome.value for _ in range(40)}; self.assertEqual(outs, {"red", "blue"})
    def test_measurement_domain_enum(self): self.assertEqual(TransformationDomain.MEASUREMENT.value, "measurement")


class LatentAccessTests(unittest.TestCase):
    def test_no_agent_peek(self):
        w = LatentWorldState({("e1", "color"): "red"}); w.measure(op()); self.assertEqual(w.agent_peeks, 0)
    def test_evaluation_peek_counted(self):
        w = LatentWorldState({("e1", "color"): "red"}); w.peek(("e1", "color")); self.assertEqual(w.evaluation_peeks, 1)
    def test_agent_peek_violates(self):
        w = LatentWorldState({("e1", "color"): "red"}); w.peek(("e1", "color"), by="agent")
        self.assertFalse(check_invariants(latent=w)[MeasurementInvariant.NO_HIDDEN_TRUTH_ACCESS])
    def test_log(self): w = LatentWorldState({("e1", "color"): "red"}); w.measure(op()); self.assertEqual(len(w.log), 1)
    def test_destructive_changes_world(self):
        def effect(world): world.set(("e1", "loc"), "floor")
        w = LatentWorldState({("e1", "color"): "red", ("e1", "loc"): "table"})
        o = op(); o.context = MeasurementContext(sensor="x", noise=NoiseModel("confusion", 1.0), disturbance=DESTRUCTIVE, disturbance_effect=effect)
        w.measure(o); self.assertEqual(w.peek(("e1", "loc")), "floor")
    def test_passive_does_not_change_world(self):
        w = LatentWorldState({("e1", "color"): "red", ("e1", "loc"): "table"}); w.measure(op()); self.assertEqual(w.peek(("e1", "loc")), "table")


# ================================================================ résolution vs incertitude

class ResolutionTests(unittest.TestCase):
    def test_bin(self): self.assertAlmostEqual(MeasurementResolution(0.1).bin(2.07), 2.0)
    def test_interval(self): self.assertEqual(MeasurementResolution(0.5).interval(1.0), (1.0, 1.5))
    def test_zero_resolution(self): self.assertEqual(MeasurementResolution(0).interval(1.3), (1.3, 1.3))
    def test_continuous_sample_has_interval(self):
        o = MeasurementOperator(Observable("h", "e", "height"), MeasurementContext(noise=NoiseModel("gaussian", sigma=0.01), resolution=0.1))
        out = o.sample(2.07, random.Random(0)); self.assertAlmostEqual(out.interval[1] - out.interval[0], 0.1, places=6)
    def test_resolution_distinct_from_noise(self):
        precise_coarse = MeasurementContext(noise=NoiseModel("gaussian", sigma=0.001), resolution=1.0)
        noisy_fine = MeasurementContext(noise=NoiseModel("gaussian", sigma=0.5), resolution=0.01)
        self.assertNotEqual((precise_coarse.resolution, precise_coarse.noise.sigma), (noisy_fine.resolution, noisy_fine.noise.sigma))
    def test_coarse_reading_likelihood_flat_within_bin(self):
        o = MeasurementOperator(Observable("h", "e", "height"), MeasurementContext(noise=NoiseModel("gaussian", sigma=0.001), resolution=1.0))
        L = o.likelihood(MeasurementOutcome(1.0), torch.tensor([1.2, 1.8])); self.assertAlmostEqual(float(L[0]), float(L[1]), places=4)
    def test_fine_reading_discriminates(self):
        o = MeasurementOperator(Observable("h", "e", "height"), MeasurementContext(noise=NoiseModel("gaussian", sigma=0.05), resolution=0.1))
        L = o.likelihood(MeasurementOutcome(1.2), torch.tensor([1.25, 1.8])); self.assertGreater(float(L[0]), float(L[1]) * 100)
    def test_gaussian_density(self):
        o = MeasurementOperator(Observable("h", "e", "height"), MeasurementContext(noise=NoiseModel("gaussian", sigma=0.1)))
        L = o.likelihood(MeasurementOutcome(2.0), torch.tensor([2.0, 2.3])); self.assertGreater(float(L[0]), float(L[1]))


# ================================================================ référentiels

class FrameTests(unittest.TestCase):
    def frames(self): return {"A": ReferenceFrame("A", (2.0, 1.0), math.pi / 3), "B": ReferenceFrame("B", (-1.0, 4.0), -math.pi / 4)}
    def test_roundtrip(self):
        f = self.frames(); p = FramedValue(torch.tensor([1.5, -0.5], dtype=torch.float64), "A")
        back = change_frame(change_frame(p, f, "world"), f, "A"); self.assertTrue(torch.allclose(back.value, p.value))
    def test_distance_invariant(self):
        f = self.frames(); P = FramedValue(torch.tensor([1.0, 2.0], dtype=torch.float64), "world"); Q = FramedValue(torch.tensor([3.0, -1.0], dtype=torch.float64), "world")
        self.assertAlmostEqual(distance(f["A"].from_world(P), f["A"].from_world(Q)), distance(P, Q))
    def test_coordinates_change(self):
        f = self.frames(); P = FramedValue(torch.tensor([1.0, 2.0], dtype=torch.float64), "world")
        self.assertFalse(torch.allclose(f["A"].from_world(P).value, P.value))
    def test_invalid_comparison_raises(self):
        with self.assertRaises(FrameError): distance(FramedValue(torch.zeros(2), "A"), FramedValue(torch.zeros(2), "B"))
    def test_unit_mismatch_raises(self):
        with self.assertRaises(UnitError): FramedValue(torch.zeros(2), "A", unit="m").same_frame(FramedValue(torch.zeros(2), "A", unit="cm"))
    def test_vector_ignores_translation(self):
        f = ReferenceFrame("T", (10.0, 10.0), 0.0); v = FramedValue(torch.tensor([1.0, 0.0], dtype=torch.float64), "world", "vector")
        self.assertTrue(torch.allclose(f.from_world(v).value, v.value))
    def test_point_uses_translation(self):
        f = ReferenceFrame("T", (10.0, 10.0), 0.0); p = FramedValue(torch.tensor([1.0, 0.0], dtype=torch.float64), "world")
        self.assertTrue(torch.allclose(f.from_world(p).value, torch.tensor([-9.0, -10.0], dtype=torch.float64)))
    def test_velocity_galilean(self):
        f = ReferenceFrame("M", velocity=(1.0, 0.0)); v = FramedValue(torch.tensor([3.0, 0.0], dtype=torch.float64), "world", "velocity")
        self.assertTrue(torch.allclose(f.from_world(v).value, torch.tensor([2.0, 0.0], dtype=torch.float64)))
    def test_orientation(self):
        f = ReferenceFrame("R", angle=0.5); o = FramedValue(torch.tensor(1.0, dtype=torch.float64), "world", "orientation")
        self.assertAlmostEqual(float(f.from_world(o).value), 0.5)
    def test_wrong_frame_to_world(self):
        with self.assertRaises(FrameError): ReferenceFrame("A").to_world(FramedValue(torch.zeros(2), "B"))
    def test_world_frame_name(self): self.assertEqual(WORLD_FRAME.name, "world")
    def test_traced(self):
        g = new_graph(); change_frame(FramedValue(torch.zeros(2, dtype=torch.float64), "A"), self.frames(), "world", g)
        self.assertEqual(g.current.get_path_names()[-1], "ChangeFrame")
    def test_frame_invariant_check(self):
        r = check_invariants(frames=self.frames(), values=[FramedValue(torch.zeros(2), "A"), FramedValue(torch.zeros(2), "Z")])
        self.assertFalse(r[MeasurementInvariant.REFERENCE_FRAME_VALID])
    def test_unit_invariant_check(self):
        r = check_invariants(values=[FramedValue(torch.zeros(2), "A", unit="m"), FramedValue(torch.zeros(2), "A", unit="cm")])
        self.assertFalse(r[MeasurementInvariant.UNIT_CONSISTENT])


class FrameExperimentTests(unittest.TestCase):
    def test_aware_perfect(self): self.assertEqual(FRAMES["aware_correct"], 1.0)
    def test_naive_worse(self): self.assertLess(FRAMES["naive_correct"], 0.9)
    def test_invalid_detected(self): self.assertEqual(FRAMES["invalid_detected"], 1.0)
    def test_distance_invariant(self): self.assertEqual(FRAMES["distance_invariant"], 1.0)
    def test_angle_invariant(self): self.assertEqual(FRAMES["angle_invariant"], 1.0)
    def test_coordinates_change(self): self.assertEqual(FRAMES["coordinates_change"], 1.0)


# ================================================================ bases de représentation

class BasisTests(unittest.TestCase):
    def setUp(self): self.bt = BasisTransform(64); self.x = torch.randn(64, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    def test_inverse(self): self.assertLess(float((self.bt.inverse(self.bt.forward(self.x)) - self.x).abs().max()), 1e-9)
    def test_parseval(self): self.assertAlmostEqual(self.bt.energy(self.x), self.bt.energy(self.bt.forward(self.x)), places=6)
    def test_dimension(self): self.assertEqual(self.bt.forward(self.x).shape[-1], 64)
    def test_invariants_dict(self): self.assertTrue(all(self.bt.invariants(self.x).values()))
    def test_basis_names(self): self.assertEqual((self.bt.time.name, self.bt.frequency.name), ("time", "frequency"))
    def test_sinusoid_peak(self):
        t = torch.arange(64, dtype=torch.float64); X = self.bt.forward(torch.sin(2 * math.pi * 7 * t / 64))
        self.assertEqual(int(X.abs()[:32].argmax()), 7)
    def test_world_unchanged(self):
        x0 = self.x.clone(); self.bt.forward(self.x); self.assertTrue(torch.equal(self.x, x0))
    def test_traced(self):
        g = new_graph(); BasisTransform(16, g).forward(torch.zeros(16)); self.assertEqual(g.current.get_path_names()[-1], "ChangeBasis")
    def test_representation_basis(self): self.assertEqual(RepresentationBasis("time", 8, "s").dimension, 8)
    def test_linearity(self):
        y = torch.randn(64, dtype=torch.float64)
        self.assertTrue(torch.allclose(self.bt.forward(self.x + y), self.bt.forward(self.x) + self.bt.forward(y)))


class FourierExperimentTests(unittest.TestCase):
    def test_spectral_beats_raw_dominant(self): self.assertGreater(FOURIER["spectral"]["dominant_accuracy"], FOURIER["raw"]["dominant_accuracy"] + 0.3)
    def test_spectral_set(self): self.assertGreater(FOURIER["spectral"]["exact_set"], 0.8)
    def test_raw_set_poor(self): self.assertLess(FOURIER["raw"]["exact_set"], 0.5)
    def test_invariants(self): self.assertEqual(FOURIER["invariants"], {"invertible": 1.0, "parseval": 1.0})
    def test_make_signal(self):
        x, truth, dom = make_signal(random.Random(0)); self.assertEqual(x.shape[0], 128); self.assertIn(dom, truth)
    def test_raw_returns_candidate(self):
        x, _, _ = make_signal(random.Random(1)); s, d = raw_reasoning(x); self.assertIn(d, CANDIDATE_FREQS)
    def test_spectral_returns_candidates(self):
        x, _, _ = make_signal(random.Random(2)); s, d = spectral_reasoning(x, BasisTransform(128)); self.assertTrue(s <= set(CANDIDATE_FREQS))


# ================================================================ ensemble compatible

class EnsembleTests(unittest.TestCase):
    def test_product_size(self): self.assertEqual(len(two_cubes()), 16)
    def test_weights_normalized(self): self.assertAlmostEqual(float(two_cubes().weights.sum()), 1.0)
    def test_entropy_uniform(self): self.assertAlmostEqual(two_cubes().entropy(), math.log(16))
    def test_constraint(self):
        dom = {("a", "x"): [0, 1], ("b", "x"): [0, 1]}
        S = CompatibleStateSet.product(dom, lambda s: s.column(("a", "x")) != s.column(("b", "x"))); self.assertEqual(len(S), 2)
    def test_marginal(self): self.assertAlmostEqual(two_cubes().marginal(("e1", "loc"))["table"], 0.5)
    def test_condition_shifts(self):
        S = two_cubes().condition(rec(r=0.9)); self.assertAlmostEqual(S.marginal(("e1", "color"))["red"], 0.9)
    def test_condition_shares_states(self):
        S = two_cubes(); T = S.condition(rec()); self.assertIs(S.states, T.states)
    def test_condition_does_not_mutate_prior(self):
        S = two_cubes(); S.condition(rec()); self.assertAlmostEqual(S.marginal(("e1", "color"))["red"], 0.5)
    def test_condition_history(self): self.assertTrue(two_cubes().condition(rec()).history)
    def test_restrict(self):
        S = two_cubes(); T = S.restrict(S.column(("e1", "loc")) == 0); self.assertAlmostEqual(T.marginal(("e1", "loc"))["table"], 1.0)
    def test_support_size(self):
        S = two_cubes(); self.assertEqual(S.restrict(S.column(("e1", "loc")) == 0).support_size(), 8)
    def test_entropy_decreases_after_restrict(self):
        S = two_cubes(); self.assertLess(S.restrict(S.column(("e1", "loc")) == 0).entropy(), S.entropy())
    def test_marginal_entropy(self): self.assertAlmostEqual(two_cubes().entropy(("e1", "loc")), math.log(2))
    def test_map_state(self):
        st, p = two_cubes().condition(rec(r=0.99)).map_state(); self.assertEqual(st[("e1", "color")], "red")
    def test_multiplicity(self):
        S = two_cubes(); m = S.multiplicity(lambda s: (s[:, [0, 2]] == 0).sum(1)); self.assertEqual(sum(m.values()), 16)
    def test_traced_condition(self):
        g = new_graph(); two_cubes(g).condition(rec()); self.assertIn("Condition", g.current.get_path_names())
    def test_traced_marginalize(self):
        g = new_graph(); two_cubes(g).marginal(("e1", "loc")); self.assertEqual(g.current.get_path_names()[-1], "Marginalize")
    def test_continuous_condition(self):
        dom = {("e", "h"): [0.0, 1.0, 2.0]}; S = CompatibleStateSet.product(dom)
        o = MeasurementOperator(Observable("h", "e", "h"), MeasurementContext(noise=NoiseModel("gaussian", sigma=0.1)))
        T = S.condition(MeasurementRecord(o.observable, o.context, MeasurementOutcome(1.05), "scale"))
        self.assertGreater(T.marginal(("e", "h"))[1.0], 0.99)


class FusionConflictTests(unittest.TestCase):
    def test_fusion_agreement(self):
        S = fuse(two_cubes(), [rec(r=0.8, prov="a"), rec(r=0.8, prov="b")]); self.assertGreater(S.marginal(("e1", "color"))["red"], 0.9)
    def test_fusion_uses_reliability(self):
        S1 = fuse(two_cubes(), [rec(value="red", prov="good"), rec(value="blue", prov="bad")], lambda p: 0.95 if p == "good" else 0.55)
        self.assertGreater(S1.marginal(("e1", "color"))["red"], 0.8)
    def test_fusion_traced(self):
        g = new_graph(); fuse(two_cubes(g), [rec()]); self.assertIn("Fuse", g.current.get_path_names())
    def test_contradiction_multimodal(self):
        S = fuse(two_cubes(), [rec(value="red", r=0.9), rec(value="blue", r=0.9)])
        m = S.marginal(("e1", "color")); self.assertAlmostEqual(m["red"], m["blue"], places=6)
    def test_conflicted_declared(self):
        rs = [rec(value="red", r=0.9), rec(value="blue", r=0.9)]
        self.assertTrue(fuse(two_cubes(), rs).conflict(("e1", "color"), rs))
    def test_no_conflict_weak_source(self):
        rs = [rec(value="red", r=0.95), rec(value="blue", r=0.5)]
        self.assertFalse(fuse(two_cubes(), rs).conflict(("e1", "color"), rs))
    def test_conflicted_constant(self): self.assertEqual(CONFLICTED, "CONFLICTED")


# ================================================================ gain d'information, valeur décisionnelle

class InformationGainTests(unittest.TestCase):
    def test_ig_positive(self): self.assertGreater(information_gain(two_cubes(), op()), 0)
    def test_ig_perfect_sensor(self): self.assertAlmostEqual(information_gain(two_cubes(), op(r=1.0)), math.log(2), places=6)
    def test_ig_useless_sensor(self): self.assertAlmostEqual(information_gain(two_cubes(), op(r=0.5)), 0.0, places=6)
    def test_ig_target_irrelevant(self): self.assertAlmostEqual(information_gain(two_cubes(), op(), ("e1", "loc")), 0.0, places=6)
    def test_ig_target_relevant(self): self.assertGreater(information_gain(two_cubes(), op(), ("e1", "color")), 0.0)
    def test_ig_monotone_in_reliability(self):
        igs = [information_gain(two_cubes(), op(r=r)) for r in (0.6, 0.8, 0.95)]; self.assertEqual(igs, sorted(igs))
    def test_expected_entropy_probs(self):
        _, po = expected_posterior_entropy(two_cubes(), op()); self.assertAlmostEqual(float(po.sum()), 1.0)
    def test_ig_predicts_realized(self):
        S = two_cubes(); ig = information_gain(S, op(r=1.0)); T = S.condition(rec(r=1.0))
        self.assertAlmostEqual(ig, S.entropy() - T.entropy(), places=6)
    def test_ig_zero_when_known(self):
        S = two_cubes().condition(rec(r=1.0)); self.assertAlmostEqual(information_gain(S, op(r=0.9), ("e1", "color")), 0.0, places=6)
    def test_decision_value_nonnegative(self): self.assertGreaterEqual(decision_value(two_cubes(), op(), ("e1", "color")), 0.0)
    def test_decision_value_irrelevant_zero(self): self.assertAlmostEqual(decision_value(two_cubes(), op(), ("e1", "loc")), 0.0)
    def test_decision_value_relevant(self): self.assertGreater(decision_value(two_cubes(), op(r=0.99), ("e1", "color"), 20.0, 8.0), 0.0)
    def test_decision_value_bounded(self): self.assertLessEqual(decision_value(two_cubes(), op(r=0.99), ("e1", "color"), 20.0, 8.0), 8.0)


class PlannerTests(unittest.TestCase):
    def ops(self):
        return [op("e1", "color", r=0.9, cost=0.1), op("e1", "loc", ("table", "shelf"), r=0.99, cost=0.4),
                op("e2", "color", r=0.6, cost=0.05)]
    def test_max_ig(self):
        c = MeasurementPlanner("max_ig").choose(two_cubes(), self.ops(), ("e1", "loc")); self.assertEqual(c.operator.observable.attribute, "loc")
    def test_max_utility_considers_cost(self):
        c = MeasurementPlanner("max_utility", value_of_information=1.0).choose(two_cubes(), self.ops(), ("e1", "color"))
        self.assertEqual(c.operator.observable.entity, "e1")
    def test_most_reliable(self):
        c = MeasurementPlanner("most_reliable").choose(two_cubes(), self.ops(), None); self.assertAlmostEqual(c.operator.context.noise.reliability, 0.99)
    def test_random_returns(self): self.assertIsNotNone(MeasurementPlanner("random", seed=1).choose(two_cubes(), self.ops(), None))
    def test_stop_confident(self):
        S = two_cubes().condition(rec(r=0.99)); self.assertIsNone(MeasurementPlanner(confidence=0.95).choose(S, self.ops(), ("e1", "color")))
    def test_stop_budget(self):
        p = MeasurementPlanner(budget=0.0); self.assertIsNone(p.choose(two_cubes(), self.ops(), ("e1", "color")))
    def test_stop_no_candidates(self): self.assertIsNone(MeasurementPlanner().choose(two_cubes(), [], ("e1", "color")))
    def test_stop_negative_utility(self):
        self.assertIsNone(MeasurementPlanner(value_of_information=0.01).choose(two_cubes(), [op(cost=5.0)], ("e1", "color")))
    def test_spent_accumulates(self):
        p = MeasurementPlanner(); p.choose(two_cubes(), self.ops(), ("e1", "color")); self.assertGreater(p.spent, 0)
    def test_disturbance_penalized(self):
        a = op(cost=0.1); b = op(cost=0.1); b.context = MeasurementContext(sensor="d", noise=NoiseModel("confusion", 0.9), cost=0.1, disturbance=DESTRUCTIVE)
        c = MeasurementPlanner().choose(two_cubes(), [b, a], ("e1", "color")); self.assertEqual(c.operator.context.disturbance, PASSIVE)
    def test_weak_disturbance_cheaper(self):
        p = MeasurementPlanner(); w = op(); w.context = MeasurementContext(disturbance=WEAKLY_DISTURBING); d = op(); d.context = MeasurementContext(disturbance=DESTRUCTIVE)
        self.assertLess(p.disturbance_cost(w, None), p.disturbance_cost(d, None))
    def test_decision_costs_mode(self):
        p = MeasurementPlanner(); p.decision_costs = (20.0, 8.0); c = p.choose(two_cubes(), self.ops(), ("e1", "color"))
        self.assertIsNotNone(c)
    def test_select_traced(self):
        g = new_graph(); MeasurementPlanner(graph=g).choose(two_cubes(), self.ops(), ("e1", "color")); self.assertIn("SelectMeasurement", g.current.get_path_names())
    def test_policies(self): self.assertEqual(set(MeasurementPlanner.POLICIES), {"random", "fixed", "most_reliable", "max_ig", "max_utility"})


# ================================================================ macro / micro, compromis

class MacrostateTests(unittest.TestCase):
    def test_multiplicity_binomial(self): self.assertEqual(MACRO["multiplicity"], {0: 81, 1: 324, 2: 486, 3: 324, 4: 81})
    def test_entropy_log_multiplicity(self): self.assertAlmostEqual(MACRO["entropy_by_macro"][2], round(math.log(486), 3))
    def test_more_micro_more_entropy(self): self.assertTrue(MACRO["multiplicity_entropy_monotone"])
    def test_ambiguity_kept(self): self.assertEqual(MACRO["max_ig"]["ambiguity_kept_after_macro"], 1.0)
    def test_max_ig_beats_random(self): self.assertLess(MACRO["max_ig"]["mean_probes"], MACRO["random"]["mean_probes"])
    def test_ig_correlates(self): self.assertGreater(MACRO["ig_vs_realized_corr"], 0.5)
    def test_macro_function(self): self.assertEqual(int(macro_left_count(torch.tensor([[0, 1, 5, 4]]))[0]), 2)
    def test_set_size(self): self.assertEqual(len(macrostate_set(2, 4)), 16)


class TradeoffTests(unittest.TestCase):
    def test_sigmas_tradeoff(self):
        t = MeasurementTradeoff(); a, b = t.sigmas(0.9), t.sigmas(0.1); self.assertLess(a[0], b[0]); self.assertGreater(a[1], b[1])
    def test_best_for_position(self): self.assertGreater(MeasurementTradeoff().best_setting((1.0, 0.0)), 0.9)
    def test_best_for_velocity(self): self.assertLess(MeasurementTradeoff().best_setting((0.0, 1.0)), 0.1)
    def test_best_balanced(self): self.assertAlmostEqual(MeasurementTradeoff().best_setting((1.0, 1.0)), 0.5, places=1)
    def test_planner_beats_fixed_position(self): self.assertLess(TRADE["position"]["planner"], TRADE["position"]["fixed_0.5"])
    def test_planner_beats_fixed_velocity(self): self.assertLess(TRADE["velocity"]["planner"], TRADE["velocity"]["fixed_0.5"])
    def test_wrong_goal_bad(self): self.assertGreater(TRADE["position"]["wrong_goal"], TRADE["position"]["planner"] * 10)
    def test_not_quantum_named(self):
        src = open(os.path.join(ROOT, "core/measurement.py")).read(); self.assertIn("PAS un principe d'incertitude quantique", src)


class InvariantTests(unittest.TestCase):
    def test_normalized(self): self.assertTrue(check_invariants(op().distribution("red"))[MeasurementInvariant.NORMALIZED_DISTRIBUTION])
    def test_unnormalized(self):
        d = MeasurementDistribution(["a", "b"], torch.tensor([0.5, 0.7])); self.assertFalse(check_invariants(d)[MeasurementInvariant.NORMALIZED_DISTRIBUTION])
    def test_provenance(self): self.assertTrue(check_invariants(record=rec())[MeasurementInvariant.PROVENANCE_PRESENT])
    def test_missing_provenance(self): self.assertFalse(check_invariants(record=rec(prov=""))[MeasurementInvariant.PROVENANCE_PRESENT])
    def test_outcome_compatible(self): self.assertTrue(check_invariants(record=rec())[MeasurementInvariant.OUTCOME_COMPATIBLE])
    def test_outcome_incompatible(self):
        r = rec(); r.outcome = MeasurementOutcome("purple"); self.assertFalse(check_invariants(record=r)[MeasurementInvariant.OUTCOME_COMPATIBLE])
    def test_all_six_defined(self): self.assertEqual(len(MeasurementInvariant), 6)
    def test_op_is_transformation(self): self.assertEqual(MeasurementOp("Measure").domain, TransformationDomain.MEASUREMENT)
    def test_trace_none_graph(self): self.assertIsNotNone(trace(None, MeasurementOp("Fuse")))
    def test_change_basis_invertible_flag(self): self.assertTrue(MeasurementOp("ChangeBasis").invertible)


class NoQuantumClaimTests(unittest.TestCase):
    def test_no_hbar(self):
        src = open(os.path.join(ROOT, "core/measurement.py")).read(); self.assertNotIn("ħ/2 ≥", src); self.assertIn("analogie", src.lower())
    def test_no_complex_amplitudes(self):
        tree = ast.parse(open(os.path.join(ROOT, "core/measurement.py")).read())
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}; self.assertNotIn("complex128", names)



# ================================================================ langage conscient de la mesure

from core.measurement_language import parse, parse_description, parse_place, parse_question
from core.measurement_world import (
    ACTIVE_MENU, REGION, SEALED_SHA256, SPEAKERS, SUPPORTS, episode, eval_episodes, sealed_episodes, sealed_hash, train_episodes,
)


class LanguageTests(unittest.TestCase):
    def test_selon(self): p = parse("selon alice, le cube rouge est sur la table"); self.assertEqual((p.kind, p.observer, p.value), ("observation", "alice", "table"))
    def test_dit_que(self): p = parse("bob dit que la balle est dans la boîte"); self.assertEqual((p.observer, p.value, p.description.type), ("bob", "box", "ball"))
    def test_color_parsed(self): self.assertEqual(parse("selon doc, le cube bleu est sur le sol").description.color, "blue")
    def test_under_shelf(self): self.assertEqual(parse("selon alice, le bloc est sous l'étagère").value, "under_shelf")
    def test_sensor_event(self):
        p = parse("le capteur voit le cube rouge qui tombe vers le sol"); self.assertEqual((p.kind, p.sensor, p.value, p.extra["verb"]), ("event", True, "floor", "tombe"))
    def test_sensor_move(self): self.assertEqual(parse("le capteur voit la balle verte qui est déplacée vers la boîte").value, "box")
    def test_continuous_sigma(self): p = parse("la mesure indique environ 1.2 m pour le cube rouge"); self.assertEqual((p.kind, p.value), ("continuous", 1.2)); self.assertGreater(p.sigma, 0)
    def test_resolution_phrase(self): self.assertAlmostEqual(parse("la mesure indique environ 1.2 m pour le cube rouge, on ne sait qu'à 20 cm près").resolution, 0.2)
    def test_resolution_not_sigma(self):
        p = parse("la mesure indique environ 1.2 m pour le cube rouge, on ne sait qu'à 20 cm près"); self.assertNotEqual(p.resolution, p.sigma)
    def test_unparseable(self): self.assertIsNone(parse("bonjour tout le monde"))
    def test_relational_present(self): d = parse_description("le cube sur la table"); self.assertEqual((d.type, d.support, d.past), ("cube", "table", False))
    def test_relational_past(self): d = parse_description("la balle qui était dans la boîte"); self.assertEqual((d.support, d.past), ("box", True))
    def test_place(self): self.assertEqual(parse_place("vers l'étagère"), "shelf")
    def test_q_where_now(self): self.assertEqual(parse_question("où est le cube rouge maintenant ?")["intent"], "where_now")
    def test_q_where_at(self): q = parse_question("où était la balle à t=7 ?"); self.assertEqual((q["intent"], q["t"]), ("where_at", 7))
    def test_q_height(self): self.assertEqual(parse_question("quelle est la hauteur du cube bleu ?")["intent"], "height_now")
    def test_q_color_rel(self):
        q = parse_question("de quelle couleur est le cube sur la table ?"); self.assertEqual((q["intent"], q["description"].support), ("color_of_relational", "table"))
    def test_q_before_fall(self): q = parse_question("quel objet était sur l'étagère avant la chute ?"); self.assertEqual(q["support"], "shelf")
    def test_q_unknown(self): self.assertIsNone(parse_question("pourquoi ?"))
    def test_all_episode_texts_parse(self):
        ep = episode(123); texts = [o["text"] for o in ep["observations"] if o["kind"] == "text"]
        self.assertTrue(texts); self.assertTrue(all(parse(t) is not None for t in texts))
    def test_all_episode_questions_parse(self):
        ep = episode(321); self.assertTrue(all(parse_question(q["text"]) is not None for q in ep["questions"]))


# ================================================================ monde temporel

class WorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.ep = episode(7)
    def test_deterministic(self): self.assertEqual(episode(7)["observations"], self.ep["observations"])
    def test_T(self): self.assertEqual(len(next(iter(self.ep["_truth"]["trajectories"].values()))["loc"]), 31)
    def test_two_cubes(self): self.assertGreaterEqual([d["type"] for d in self.ep["objects"].values()].count("cube"), 2)
    def test_same_type_distinct_colors(self):
        objs = self.ep["_truth"]["objects"]; cubes = [d["color"] for d in objs.values() if d["type"] == "cube"]; self.assertEqual(len(cubes), len(set(cubes)))
    def test_supports(self):
        for tr in self.ep["_truth"]["trajectories"].values(): self.assertTrue(set(tr["loc"]) <= set(SUPPORTS))
    def test_question_types(self):
        types = {q["type"] for q in self.ep["questions"]}; self.assertTrue({"where_now", "where_at", "height_now"} <= types)
    def test_truth_hidden_from_objects(self): self.assertNotIn("color", next(iter(self.ep["objects"].values())))
    def test_scale_resolution(self): self.assertTrue(all(o["resolution"] == 0.1 for o in self.ep["observations"] if o.get("source") == "scale"))
    def test_camera_period(self): self.assertTrue(all(o["t"] % 3 == 0 for o in self.ep["observations"] if o.get("source") == "camera"))
    def test_speakers(self): self.assertEqual(set(SPEAKERS), {"alice", "bob", "doc"})
    def test_menu_disturbance(self): self.assertEqual(ACTIVE_MENU["weigh"][3], "DESTRUCTIVE")
    def test_menu_passive(self): self.assertEqual(ACTIVE_MENU["color"][3], "PASSIVE")
    def test_region(self): self.assertEqual(set(REGION), set(SUPPORTS))
    def test_splits_disjoint(self):
        tr = {e["seed"] for e in train_episodes(5)}; ev = {e["seed"] for e in eval_episodes(0, 5)}; se = {e["seed"] for e in sealed_episodes()}
        self.assertFalse(tr & ev or tr & se or ev & se)
    def test_eval_seeds_disjoint(self): self.assertFalse({e["seed"] for e in eval_episodes(0, 15)} & {e["seed"] for e in eval_episodes(1, 15)})
    def test_sealed_hash(self): self.assertEqual(sealed_hash(), SEALED_SHA256)
    def test_sealed_size(self): self.assertEqual(len(sealed_episodes()), 30)


# ================================================================ agent (petit échantillon)

from core.measurement_agent import SYSTEMS_M, MeasurementAgent, evaluate, fit_training, metrics


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hz, cls.rel = fit_training(train_episodes(6)); cls.eps = eval_episodes(7, 2)
        cls.res = {s: metrics(evaluate(s, cls.eps, cls.hz, cls.rel)) for s in ("last_observation", "temporal_passive", "full_algebra")}
    def test_systems(self): self.assertEqual(len(SYSTEMS_M), 6)
    def test_metrics_keys(self):
        for k in ("success", "false_confident", "abstention", "measurements", "cost", "calibration_ece"): self.assertIn(k, self.res["full_algebra"])
    def test_rates_bounded(self):
        for m in self.res.values(): self.assertTrue(0 <= m["success"] + m["false_confident"] + m["abstention"] <= 1.0001)
    def test_partition(self):
        m = self.res["full_algebra"]; self.assertAlmostEqual(m["success"] + m["false_confident"] + m["abstention"], 1.0, places=3)
    def test_no_peek(self):
        for m in self.res.values(): self.assertEqual(m["agent_peeks"], 0)
    def test_passive_measures(self): self.assertGreater(self.res["temporal_passive"]["measurements"], 0)
    def test_last_obs_no_measure(self): self.assertEqual(self.res["last_observation"]["measurements"], 0)
    def test_active_has_cost(self): self.assertGreaterEqual(self.res["full_algebra"]["cost"], 0)
    def test_answer_structure(self):
        ag = MeasurementAgent("full_algebra", self.hz, self.rel); ag.ingest(self.eps[0]); a = ag.answer(self.eps[0]["questions"][0])
        for k in ("answer", "p", "n", "cost", "steps"): self.assertIn(k, a)
    def test_answer_probability(self):
        ag = MeasurementAgent("temporal", self.hz, self.rel); ag.ingest(self.eps[0])
        self.assertTrue(all(0 <= ag.answer(q)["p"] <= 1 for q in self.eps[0]["questions"]))
    def test_unknown_question(self):
        ag = MeasurementAgent("full_algebra", self.hz, self.rel); ag.ingest(self.eps[0]); self.assertIsNone(ag.answer({"text": "pourquoi ?"})["answer"])
    def test_lower_bound_reliability(self):
        a = MeasurementAgent("full_algebra", self.hz, self.rel).r("bob"); b = MeasurementAgent("temporal", self.hz, self.rel).r("bob")
        self.assertLessEqual(a, b + 1e-9)


# ================================================================ diagnostic du modèle

from core.model_diagnostics import CAUSES, DIAGNOSTICS, REPAIR, TableOperator, diagnose_and_repair, direct_repair, inject, learn_tables, observe, setup


class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.xv, cls.yv, cls.acc = setup(0); cls.tables = learn_tables(n_per_cause=3, seed=0)
    def broken(self, cause, seed=5):
        m = copy.deepcopy(self.base); inject(m, cause, random.Random(seed)); return m
    def test_healthy_observations(self):
        self.assertEqual(observe(self.base, "routing", self.xv, self.yv, self.acc), "ok"); self.assertEqual(observe(self.base, "fast", self.xv, self.yv, self.acc), "ok")
    def test_drift_detected(self): self.assertEqual(observe(self.broken("corrupt:head"), "drift:head", self.xv, self.yv, self.acc), "large")
    def test_routing_detected(self): self.assertEqual(observe(self.broken("bad_routing"), "routing", self.xv, self.yv, self.acc), "changed")
    def test_capacity_detected(self): self.assertEqual(observe(self.broken("insufficient_capacity"), "capacity", self.xv, self.yv, self.acc), "reduced")
    def test_fast_detected(self): self.assertEqual(observe(self.broken("fast_drift"), "fast", self.xv, self.yv, self.acc), "large")
    def test_tables_normalized(self):
        for d, tab in self.tables.items():
            for c in CAUSES: self.assertAlmostEqual(sum(tab[c].values()), 1.0)
    def test_table_operator_likelihood(self):
        o = TableOperator("routing", self.tables["routing"], 0.02); self.assertEqual(o.likelihood("changed", torch.arange(len(CAUSES))).shape, (len(CAUSES),))
    def test_repair_map_complete(self): self.assertEqual(set(REPAIR), set(CAUSES))
    def test_diagnostic_costs(self): self.assertTrue(all(c > 0 for _, c in DIAGNOSTICS.values()))
    def test_diagnose_routing(self):
        r = diagnose_and_repair(self.broken("bad_routing"), self.tables, self.xv, self.yv, self.acc); self.assertEqual(r["cause"], "bad_routing")
    def test_diagnose_fast(self):
        r = diagnose_and_repair(self.broken("fast_drift"), self.tables, self.xv, self.yv, self.acc); self.assertEqual(r["repair"], "clear_fast_weights")
    def test_repair_restores_accuracy(self):
        r = diagnose_and_repair(self.broken("corrupt:head"), self.tables, self.xv, self.yv, self.acc); self.assertGreaterEqual(r["accuracy"], self.acc - 0.02)
    def test_diagnosis_traced(self):
        r = diagnose_and_repair(self.broken("fast_drift"), self.tables, self.xv, self.yv, self.acc); self.assertGreater(r["graph_nodes"], 1)
    def test_diagnosis_cost_bounded(self):
        r = diagnose_and_repair(self.broken("corrupt:enc"), self.tables, self.xv, self.yv, self.acc); self.assertLessEqual(r["cost"], 1.5 + 0.31)
    def test_direct_restore_all(self): self.assertGreaterEqual(direct_repair(self.broken("corrupt:enc"), "restore_all", self.xv, self.yv), self.acc - 0.02)
    def test_posterior_normalized(self):
        r = diagnose_and_repair(self.broken("bad_routing"), self.tables, self.xv, self.yv, self.acc); self.assertAlmostEqual(sum(r["posterior"].values()), 1.0)

if __name__ == "__main__":
    unittest.main()

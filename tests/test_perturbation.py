import copy
import math
import random
import unittest

import torch

from core.generalized_credit import (COMPONENTS, FAULTS, answer, base_model, credit_benchmark, generalized_credit,
                                     gradient_only_cause, inject, make_system, pipeline_score, score)
from core.invariant_algebra import (
    ALG_INVARIANTS, CODE_INVARIANTS, CODE_SAMPLES, DOMAINS, GEO_INVARIANTS, LANG_INVARIANTS, SEQ_INVARIANTS, CheckedInvariant,
    DomainTransformation, DualityIndex, EquivalenceClass, InvariantSet, InvariantSignature, Stabilizer, TransformationOrbit,
    algebra_key, algebra_samples, algebra_transformations, build_index, canonical_equation, canonical_form, canonical_geometry,
    canonical_program, canonical_sentence, code_transformations, compression, constant_fold, data_dependencies,
    extract_expression, linear_equation, rename_capture, rename_variable, reorder_any, reorder_independent, run_program,
    solution_set, type_valid,
)
from core.measurement import CompatibleStateSet, LatentWorldState, MeasurementContext, MeasurementOperator, MeasurementPlanner, NoiseModel, Observable
from core.measurement_value import (CounterfactualMeasurement, MeasurementValueLearner, counterfactual_measurement, features,
                                    online_adaptation_experiment)
from core.perturbation import (KINDS, DifferentiableSensitivity, Perturbation, PerturbationResponse, StructuralEdit,
                               StructuralSensitivity, belief_sensitivity, gradient_credit, interventional_credit, local_sensitivity,
                               magnitudes, model_response, new_graph, sensitivity_curve, traced, unit_direction)
from core.perturbation_benchmark import (FAMILIES, SEALED_SHA256, compression_experiment, evaluate_items, make_items,
                                         new_object, random_program, sealed_hash, sealed_items)
from core.perturbation_diagnostics import PROBE_COST, prior_from_profile
from core.safe_plasticity import METHODS, CODES, changed_parameters, make_invariants, old_model, run_method, safe_consolidate, task_data, train_fast
from core.stability import (InvariantPersistence, PerturbationInteractionGraph, StableRegion, StructuralResonanceCandidate,
                            TransitionBoundary, accuracy_invariant, algebra_chain_perturbation, detect_transitions,
                            fourier_sensitivity, frame_perturbation, interaction, perturbed_model, persistence,
                            routing_transition_observer, stability_map, stability_radius)
from core.transformations import TransformationSpec

M = XV = YV = None
OLD = INV = None


def setUpModule():
    global M, XV, YV, OLD, INV
    M, XV, YV = base_model(0); OLD = old_model(0); INV = make_invariants(OLD, 0)


def cubes():
    return CompatibleStateSet.product({("e1", "color"): ["red", "blue"], ("e1", "loc"): ["table", "shelf", "floor"]})


def color_op(r=0.9):
    return MeasurementOperator(Observable("color", "e1", "color", ("red", "blue")), MeasurementContext(sensor="s", noise=NoiseModel("confusion", r), cost=0.1))


# ================================================================ A/B — perturbations

class PerturbationTests(unittest.TestCase):
    def test_kinds(self): self.assertEqual(len(KINDS), 7)
    def test_is_transformation_spec(self): self.assertIsInstance(Perturbation("input", "x", 0.1), TransformationSpec)
    def test_fields(self):
        p = Perturbation("parameter", ("enc", "w1"), 0.2, norm="l2", locality="module", domain="model")
        self.assertEqual((p.magnitude, p.norm, p.locality, p.pdomain), (0.2, "l2", "module", "model"))
    def test_unknown_kind(self):
        with self.assertRaises(AssertionError): Perturbation("cosmic", "x", 1.0)
    def test_scaled(self): self.assertAlmostEqual(Perturbation("input", "x", 0.1).scaled(4).magnitude, 0.4)
    def test_magnitudes(self): self.assertEqual(magnitudes(0.01, 3), [0.01, 0.02, 0.04])
    def test_input_apply_pure(self):
        x = torch.zeros(4); y = Perturbation("input", "x", 1.0, torch.ones(4) / 2).apply(x); self.assertTrue(torch.equal(x, torch.zeros(4))); self.assertAlmostEqual(float(y.norm()), 1.0)
    def test_input_inverse(self):
        x = torch.randn(5); p = Perturbation("input", "x", 0.3, unit_direction((5,), 1)); self.assertTrue(torch.allclose(p.inverse().apply(p.apply(x)), x))
    def test_parameter_does_not_mutate(self):
        W = M.modules["enc"].slow["w1"].clone(); Perturbation("parameter", ("enc", "w1"), 0.5).apply(M); self.assertTrue(torch.equal(W, M.modules["enc"].slow["w1"]))
    def test_parameter_inverse_exact(self):
        p = Perturbation("parameter", ("expert_a", "w1"), 0.2); back = p.inverse().apply(p.apply(M))
        self.assertLess(float((back.modules["expert_a"].slow["w1"] - M.modules["expert_a"].slow["w1"]).abs().max()), 1e-5)
    def test_parameter_relative_magnitude(self):
        p = Perturbation("parameter", ("head", "w2"), 0.1); W = M.modules["head"].slow["w2"]
        self.assertAlmostEqual(float((p.apply(M).modules["head"].slow["w2"] - W).norm() / W.norm()), 0.1, places=4)
    def test_connection(self):
        k = ("enc", "expert_a"); m2 = Perturbation("connection", k, 0.5).apply(M); self.assertAlmostEqual(m2.connectivity[k].strength, 1.5 * M.connectivity[k].strength)
    def test_connection_cut(self): self.assertAlmostEqual(Perturbation("connection", ("enc", "expert_a"), -1.0).apply(M).connectivity[("enc", "expert_a")].strength, 0.0)
    def test_routing_adds_branch(self):
        m2 = Perturbation("routing", ("expert_b", "head"), 1.0).apply(M); self.assertIn("expert_b", m2.topological_order()); self.assertNotIn("expert_b", M.topological_order())
    def test_routing_zero_noop(self): self.assertEqual(Perturbation("routing", ("expert_b", "head"), 0.0).apply(M).topological_order(), M.topological_order())
    def test_belief(self):
        S = cubes(); T = Perturbation("belief", "logw", 0.5).apply(S); self.assertFalse(torch.allclose(S.weights, T.weights)); self.assertAlmostEqual(float(T.weights.sum()), 1.0)
    def test_belief_inverse(self):
        S = cubes(); p = Perturbation("belief", "logw", 0.5); self.assertTrue(torch.allclose(p.inverse().apply(p.apply(S)).weights, S.weights))
    def test_measurement(self):
        o = color_op(0.9); o2 = Perturbation("measurement", "s", 0.2, domain="measurement").apply(o)
        self.assertAlmostEqual(o2.context.noise.reliability, 0.7); self.assertAlmostEqual(o.context.noise.reliability, 0.9)
    def test_measurement_clamped(self): self.assertGreaterEqual(Perturbation("measurement", "s", 5.0).apply(color_op()).context.noise.reliability, 0.0)
    def test_transformation(self):
        ch = ["a", "b", "c"]; self.assertEqual(Perturbation("transformation", 1, 1.0, "z").apply(ch), ["a", "z", "c"]); self.assertEqual(ch, ["a", "b", "c"])
    def test_traced(self):
        g = new_graph(); traced(g, Perturbation("input", "x", 0.1)); self.assertEqual(len(g.nodes), 2)
    def test_traced_none(self): self.assertIsNone(traced(None, Perturbation("input", "x", 0.1)))
    def test_custom_fn(self): self.assertEqual(Perturbation("input", "x", 2.0, fn=lambda o, p: o * p.magnitude).apply(3.0), 6.0)
    def test_not_invertible_routing(self):
        with self.assertRaises(ValueError): Perturbation("routing", ("expert_b", "head"), 1.0).inverse()
    def test_name(self): self.assertIn("parameter", Perturbation("parameter", ("enc", "w1"), 0.1).name)
    def test_unit_direction(self): self.assertAlmostEqual(float(unit_direction((3, 4), 0).norm()), 1.0, places=5)


# ================================================================ C/D — response, local sensitivity

class ResponseSensitivityTests(unittest.TestCase):
    def test_response_fields(self):
        p = Perturbation("parameter", ("enc", "w1"), 0.1); r = model_response(M, p.apply(M), XV, YV, p)
        for f in ("output_change", "belief_change", "decision_change", "invariant_violations", "task_loss_change", "branch_change", "measurement_policy_change"): self.assertTrue(hasattr(r, f))
    def test_zero_perturbation(self):
        p = Perturbation("parameter", ("enc", "w1"), 0.0); r = model_response(M, p.apply(M), XV, YV, p); self.assertAlmostEqual(r.output_change, 0.0, places=6)
    def test_response_grows(self):
        rs = [model_response(M, Perturbation("parameter", ("enc", "w1"), e).apply(M), XV, YV).output_change for e in (0.01, 0.1)]; self.assertLess(rs[0], rs[1])
    def test_branch_change_flag(self):
        p = Perturbation("routing", ("expert_b", "head"), 1.0); self.assertTrue(model_response(M, p.apply(M), XV, YV, p).branch_change)
    def test_invariant_violation_reported(self):
        p = Perturbation("parameter", ("enc", "w1"), 5.0); r = model_response(M, p.apply(M), XV, YV, p, {"acc>=0.99": lambda m: m.accuracy(XV, YV) >= 0.99})
        self.assertEqual(r.invariant_violations, ("acc>=0.99",))
    def test_sensitivity_property(self): self.assertAlmostEqual(PerturbationResponse("p", 0.5, output_change=1.0).sensitivity, 2.0)
    def test_local_sensitivity_linear(self):
        A = torch.tensor([[2.0, 0.0], [0.0, 3.0]]); self.assertAlmostEqual(local_sensitivity(lambda v: A @ v, torch.zeros(2), torch.tensor([0.0, 1e-3])), 3.0, places=4)
    def test_local_sensitivity_constant(self): self.assertAlmostEqual(local_sensitivity(lambda v: torch.ones(2), torch.zeros(2), torch.ones(2)), 0.0)
    def test_curve_linear_slope(self): self.assertAlmostEqual(sensitivity_curve(lambda e: 3 * e, 1e-3, 5)["loglog_slope"], 1.0, places=6)
    def test_curve_quadratic_slope(self): self.assertAlmostEqual(sensitivity_curve(lambda e: e * e, 1e-3, 5)["loglog_slope"], 2.0, places=6)
    def test_curve_magnitudes(self): self.assertEqual(len(sensitivity_curve(lambda e: e, 1e-2, 4)["magnitudes"]), 4)
    def test_model_sensitivity_stable_small(self):
        d = unit_direction(M.modules["enc"].slow["w1"].shape, 3)
        s = sensitivity_curve(lambda e: model_response(M, Perturbation("parameter", ("enc", "w1"), e, d).apply(M), XV, YV).output_change, 1e-4, 4)["sensitivity"]
        self.assertLess(max(s) / min(s), 1.05)
    def test_belief_sensitivity(self): s, T = belief_sensitivity(cubes(), 0.1); self.assertGreater(s, 0)
    def test_belief_sensitivity_marginal(self): s, _ = belief_sensitivity(cubes(), 0.1, ("e1", "color")); self.assertGreaterEqual(s, 0)
    def test_belief_sensitivity_original_intact(self):
        S = cubes(); belief_sensitivity(S, 1.0); self.assertAlmostEqual(S.marginal(("e1", "color"))["red"], 0.5)


# ================================================================ E/F — autograd, local backpropagation

class DifferentiableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.ds = DifferentiableSensitivity(M); cls.cmp = cls.ds.compare(XV[:32], YV[:32], n_weights=6)
    def test_input_jacobian_shape(self): self.assertEqual(tuple(self.ds.input_jacobian(XV[0]).shape), (2, 10))
    def test_autograd_vs_fd_input(self): self.assertLess(self.cmp["input_rel_error"], 1e-6)
    def test_autograd_vs_fd_weights(self): self.assertLess(self.cmp["weight_rel_error_max"], 1e-4)
    def test_weight_gradients_modules(self): self.assertEqual(set(self.ds.weight_gradients(XV, YV)), {"enc", "expert_a", "head"})
    def test_weight_gradient_shapes(self):
        g = self.ds.weight_gradients(XV, YV); self.assertEqual(g["enc"]["w1"].shape, M.modules["enc"].slow["w1"].shape)
    def test_routing_gradients_keys(self): self.assertIn(("enc", "expert_a"), self.ds.routing_gradients(XV, YV))
    def test_routing_strengths_restored(self):
        before = {k: c.strength for k, c in M.connectivity.items()}; self.ds.routing_gradients(XV, YV)
        self.assertEqual(before, {k: c.strength for k, c in M.connectivity.items()})
    def test_routing_representation_jacobian(self): self.assertEqual(self.ds.routing_representation_jacobian(XV, "expert_a", "head").shape[0], 24)
    def test_weights_not_modified(self):
        W = M.modules["head"].slow["w2"].clone(); self.ds.weight_gradients(XV, YV); self.assertTrue(torch.equal(W, M.modules["head"].slow["w2"]))
    def test_fd_input_shape(self): self.assertEqual(tuple(self.ds.finite_difference_input(XV[0]).shape), (2, 10))
    def test_gradient_credit_keys(self):
        g = gradient_credit(M, XV, YV); self.assertTrue(any(k.startswith("weights:") for k in g)); self.assertTrue(any(k.startswith("routing:") for k in g))


# ================================================================ G — structural sensitivity

class StructuralTests(unittest.TestCase):
    def test_structural_ranks(self):
        sys_ = {"rules": [1, 2, 3], "proof": "p"}
        edits = [StructuralEdit("drop_rule", "rules", "rule", lambda s: {**s, "rules": s["rules"][:-1]}),
                 StructuralEdit("drop_proof", "proof", "proof", lambda s: {**s, "proof": None})]
        base, out = StructuralSensitivity(lambda s: sum(s["rules"]) + (10 if s["proof"] else 0)).run(sys_, edits)
        self.assertEqual(out[0]["edit"], "drop_proof"); self.assertEqual(base, 16)
    def test_structural_original_intact(self):
        sys_ = {"r": [1, 2]}; StructuralSensitivity(lambda s: len(s["r"])).run(sys_, [StructuralEdit("x", "r", "rule", lambda s: (s["r"].pop(), s)[1])])
        self.assertEqual(sys_["r"], [1, 2])
    def test_structural_traced(self):
        g = new_graph(); StructuralSensitivity(lambda s: 0).run({}, [StructuralEdit("e", "c", "rule", lambda s: s)], g); self.assertEqual(len(g.nodes), 2)
    def test_zero_sensitivity(self):
        _, out = StructuralSensitivity(lambda s: 1.0).run({}, [StructuralEdit("e", "c", "rule", lambda s: s)]); self.assertEqual(out[0]["sensitivity"], 0.0)
    def test_pipeline_lexical_sensitivity(self):
        s = make_system(4); base = pipeline_score(s.pipe, s.qs); p = copy.deepcopy(s.pipe); p.lexicon = {}
        self.assertLess(pipeline_score(p, s.qs), base)


# ================================================================ H/I — generalized credit, interventional vs observational

class CreditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.bench = credit_benchmark(30, seed=7)
    def test_faults_cover_kinds(self): self.assertEqual({COMPONENTS[f][0] for f in FAULTS} >= {"weights", "routing", "expert", "measurement", "source", "belief", "lexical", "transformation", "planner"}, True)
    def test_generalized_beats_gradient(self): self.assertGreater(self.bench["generalized"], self.bench["gradient_only"])
    def test_gradient_blind_to_symbolic(self): self.assertEqual(self.bench["symbolic"]["gradient_only"], 0.0)
    def test_generalized_symbolic(self): self.assertGreater(self.bench["symbolic"]["generalized"], 0.8)
    def test_repaired(self): self.assertGreater(self.bench["repaired"], 0.8)
    def test_silent_reported(self): self.assertIn("silent", self.bench)
    def test_interventional_credit_pure(self):
        s = make_system(2); inject(s, "transformation", random.Random(0)); f = s.pipe.factor
        interventional_credit(s, COMPONENTS, score); self.assertEqual(s.pipe.factor, f)
    def test_transformation_fault_found(self):
        s = make_system(2); inject(s, "transformation", random.Random(0)); _, c = generalized_credit(s); self.assertEqual(c[0].component, "transformation")
    def test_belief_fault_found(self):
        s = make_system(3); inject(s, "belief", random.Random(1)); _, c = generalized_credit(s)
        if score(make_system(3)) - score(s) > 0.03: self.assertEqual(c[0].component, "belief")
    def test_observational_not_causal(self):
        s = make_system(2); inject(s, "transformation", random.Random(0)); _, c = generalized_credit(s)
        top_obs = max(c, key=lambda x: x.observational); self.assertNotEqual(top_obs.component, "transformation")
    def test_gradient_only_neural(self):
        s = make_system(1); inject(s, "lexical", random.Random(0)); self.assertIn(gradient_only_cause(s), ("weights:enc", "expert:expert_a", "weights:head", "routing"))
    def test_answer_none_unknown_word(self):
        s = make_system(1); self.assertIsNone(answer(s.pipe, {"word": "violet", "noise": 0, "reports": []}))
    def test_score_bounds(self): s = make_system(1); self.assertLessEqual(score(s), 1.0)


# ================================================================ J/K — invariants and signatures

class InvariantTests(unittest.TestCase):
    def test_checked_invariant(self): self.assertTrue(CheckedInvariant("len", "seq", len).holds([1, 2], [3, 4]))
    def test_invalid_signature(self): self.assertEqual(CheckedInvariant("x", "d", lambda v: 1 / 0).signature(1)[0], "INVALID")
    def test_set_ops(self):
        I = InvariantSet([CheckedInvariant("len", "s", len)]); self.assertIn("len", I); self.assertEqual(len(I), 1)
    def test_add(self): self.assertEqual(len(InvariantSet().add(CheckedInvariant("a", "d", len))), 1)
    def test_subset(self): self.assertEqual(len(SEQ_INVARIANTS.subset(["length"])), 1)
    def test_signature_shared(self):
        self.assertEqual(SEQ_INVARIANTS.signature([1, 2, 3], ["multiset"]), SEQ_INVARIANTS.signature([3, 1, 2], ["multiset"]))
    def test_signature_differs(self): self.assertNotEqual(SEQ_INVARIANTS.signature([1, 2, 3]), SEQ_INVARIANTS.signature([3, 1, 2]))
    def test_signature_restrict(self):
        a, b = SEQ_INVARIANTS.signature([1, 2, 3]), SEQ_INVARIANTS.signature([2, 3, 1]); self.assertTrue(a.matches(b, ["cyclic_order"])); self.assertFalse(a.matches(b))
    def test_signature_dict(self): self.assertIn("length", SEQ_INVARIANTS.signature([1]).as_dict())
    def test_check(self): self.assertEqual(SEQ_INVARIANTS.check([1, 2], [2, 1])["multiset"], True)
    def test_solution_set(self): self.assertEqual(solution_set(linear_equation(2, 3, 7)), (2,))
    def test_solution_set_all(self):
        from core.invariant_algebra import X, Equation
        self.assertEqual(solution_set(Equation(X, X)), "ALL")
    def test_solution_set_none(self):
        from core.invariant_algebra import Equation, Number
        self.assertEqual(solution_set(Equation(Number(1), Number(2))), ())
    def test_probability_normalization_invariant(self): self.assertTrue(INV.check(OLD)["probability_normalization"])
    def test_old_skill_invariant(self): self.assertTrue(all(INV.check(OLD).values()))
    def test_unit_consistency_invariant(self):
        from core.measurement import FramedValue, MeasurementInvariant, check_invariants
        self.assertFalse(check_invariants(values=[FramedValue(torch.zeros(2), "A", unit="m"), FramedValue(torch.zeros(2), "A", unit="cm")])[MeasurementInvariant.UNIT_CONSISTENT])
    def test_topology_invariant(self):
        m = Perturbation("routing", ("expert_b", "head"), 1.0).apply(OLD); self.assertFalse(INV.check(m)["topology"])


# ================================================================ L/M — stabiliser, duality (bidirectional index), generated per domain

class DualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.ix = {d: build_index(d) for d in DOMAINS}
    def test_stabilizer_contains_add(self): self.assertIn("AddBothSides(2)", self.ix["algebra"].stabilizer("solution_set"))
    def test_stabilizer_excludes_mul0(self): self.assertNotIn("MultiplyBothSides(0)", self.ix["algebra"].stabilizer("solution_set"))
    def test_stabilizer_excludes_square(self): self.assertNotIn("SquareBothSides", self.ix["algebra"].stabilizer("solution_set"))
    def test_stabilizer_excludes_mulx(self): self.assertNotIn("MultiplyBothSides(x)", self.ix["algebra"].stabilizer("solution_set"))
    def test_simplify_in_stabilizer(self): self.assertIn("Simplify", self.ix["algebra"].stabilizer("solution_set"))
    def test_bidirectional_consistency(self):
        ix = self.ix["algebra"]
        for T in ix.transformations:
            for i in ix.invariants.items: self.assertEqual(i in ix.preserves(T), T.name in ix.stabilizer(i))
    def test_counterexample_recorded(self): self.assertIn(("SquareBothSides", "solution_set"), self.ix["algebra"].counterexamples)
    def test_chain_preserves(self): self.assertIn("solution_set", self.ix["algebra"].chain_preserves(["AddBothSides(2)", "Simplify"]))
    def test_chain_breaks(self): self.assertNotIn("solution_set", self.ix["algebra"].chain_preserves(["AddBothSides(2)", "SquareBothSides"]))
    def test_stabilizer_admissible(self): self.assertTrue(Stabilizer("I", frozenset({"a"})).admissible(["a", "a"]))
    def test_stabilizer_len(self): self.assertGreater(len(self.ix["algebra"].stabilizer("solution_set")), 10)
    def test_language_verb_synonym(self): self.assertIn("semantic_action", self.ix["language"].preserves("VerbSynonym"))
    def test_language_replace_entity(self):
        p = self.ix["language"].preserves("ReplaceEntity"); self.assertIn("semantic_action", p); self.assertNotIn("entity_identity", p)
    def test_language_replace_action(self):
        p = self.ix["language"].preserves("ReplaceAction"); self.assertIn("entity_identity", p); self.assertNotIn("semantic_action", p)
    def test_language_surface_only_identity(self): self.assertEqual(self.ix["language"].stabilizer("surface_form").transformations, frozenset({"Identity"}))
    def test_code_rename(self): self.assertTrue({"program_output", "data_dependencies", "type_validity"} <= self.ix["code"].preserves("RenameVariable"))
    def test_code_fold(self): self.assertIn("program_output", self.ix["code"].preserves("ConstantFold"))
    def test_code_reorder_independent(self): self.assertIn("program_output", self.ix["code"].preserves("ReorderIndependentStatements"))
    def test_code_extract(self): self.assertIn("program_output", self.ix["code"].preserves("ExtractExpression"))
    def test_code_capture_breaks(self): self.assertNotIn("program_output", self.ix["code"].preserves("RenameCapture"))
    def test_code_reorder_any_breaks(self): self.assertNotIn("type_validity", self.ix["code"].preserves("ReorderAny"))
    def test_geometry_rotation(self): self.assertTrue({"distances", "angles"} <= self.ix["geometry"].preserves("Rotate(30°)"))
    def test_geometry_scale(self):
        p = self.ix["geometry"].preserves("Scale(2)"); self.assertNotIn("distances", p); self.assertIn("angles", p)
    def test_geometry_reflect_orientation(self): self.assertNotIn("orientation", self.ix["geometry"].preserves("Reflect(x)"))
    def test_geometry_shear(self): self.assertNotIn("angles", self.ix["geometry"].preserves("Shear(0.5)"))
    def test_sequence_rotate(self): self.assertIn("cyclic_order", self.ix["sequence"].preserves("Rotate"))
    def test_sequence_shift(self):
        p = self.ix["sequence"].preserves("Shift(+1)"); self.assertIn("differences", p); self.assertNotIn("multiset", p)
    def test_matrix(self): self.assertEqual(len(self.ix["code"].matrix()), 6)
    def test_support(self): self.assertEqual(self.ix["algebra"].support, 6)
    def test_galois_inspired_named(self):
        src = open("core/invariant_algebra.py").read(); self.assertIn("Galois-inspired", src); self.assertIn("no group", src)


def _make_domain_test(d, tname):
    def t(self):
        ix = build_index(d); T = {x.name: x for x in DOMAINS[d][1]()}[tname]; rng = random.Random(sum(map(ord, tname)))
        for _ in range(3):
            x = new_object(d, rng)
            try: y = T(x)
            except Exception: continue
            for name in ix.preserves(tname):
                inv = DOMAINS[d][0][name]
                if d == "algebra" and name == "solution_set" and "Divide" in tname: continue
                self.assertEqual(inv.signature(y), inv.signature(x), f"{tname} should have preserved {name}")
    return t


for _d in DOMAINS:
    for _T in DOMAINS[_d][1]():
        setattr(DualityTests, f"test_generalizes_{_d}_{''.join(c if c.isalnum() else '_' for c in _T.name)}", _make_domain_test(_d, _T.name))


# ================================================================ N/O/P — orbits, classes, canonical forms, compression

class OrbitCanonicalTests(unittest.TestCase):
    def test_orbit_sequence_rotation(self):
        o = TransformationOrbit((1, 2, 3, 4), [DomainTransformation("R", lambda s: s[1:] + s[:1])]); self.assertEqual(len(o), 4)
    def test_orbit_contains(self):
        o = TransformationOrbit((1, 2, 3), [DomainTransformation("R", lambda s: s[1:] + s[:1])]); self.assertIn((3, 1, 2), o)
    def test_orbit_path(self):
        o = TransformationOrbit((1, 2, 3), [DomainTransformation("R", lambda s: s[1:] + s[:1])]); self.assertEqual(o.path_to((2, 3, 1)), ("R",))
    def test_orbit_truncated(self):
        o = TransformationOrbit(0, [DomainTransformation("inc", lambda v: v + 1)], max_states=5, max_depth=50); self.assertTrue(o.truncated); self.assertEqual(len(o), 5)
    def test_orbit_algebra_preserves_solution(self):
        ix = build_index("algebra"); G = [t for t in algebra_transformations() if t.name in ix.stabilizer("solution_set")][:5]
        o = TransformationOrbit(linear_equation(2, 3, 7), G, algebra_key, max_depth=2)
        self.assertTrue(all(solution_set(e) == (2,) for e in o.states.values()))
    def test_orbit_language(self):
        G = [t for t in DOMAINS["language"][1]() if t.name == "VerbSynonym"]; o = TransformationOrbit("la balle tombe", G); self.assertEqual(len(o), 3)
    def test_orbit_geometry(self):
        G = [t for t in DOMAINS["geometry"][1]() if t.name == "Rotate(30°)"]; o = TransformationOrbit(torch.eye(2, dtype=torch.float64), G, max_depth=12); self.assertGreaterEqual(len(o), 12)
    def test_equivalence_class(self):
        ec = EquivalenceClass(SEQ_INVARIANTS, ["multiset"]); ec.add([1, 2]); ec.add([2, 1]); ec.add([3]); self.assertEqual(len(ec), 2)
    def test_equivalent(self): self.assertTrue(EquivalenceClass(LANG_INVARIANTS, ["semantic_action"]).equivalent("la balle tombe", "le bloc tombe"))
    def test_not_equivalent_meaning(self): self.assertFalse(EquivalenceClass(LANG_INVARIANTS, ["semantic_meaning"]).equivalent("la balle tombe", "le bloc tombe"))
    def test_class_canonical(self):
        ec = EquivalenceClass(SEQ_INVARIANTS, ["multiset"]); ec.add([2, 1, 3]); ec.add([1, 2, 3]); self.assertEqual(ec.canonical([3, 2, 1]), [1, 2, 3])
    def test_canonical_equation(self): self.assertEqual(str(canonical_equation(linear_equation(2, 3, 7))), "x = 2")
    def test_canonical_equation_fraction(self): self.assertEqual(solution_set(canonical_equation(linear_equation(2, 0, 1))), solution_set(linear_equation(2, 0, 1)))
    def test_canonical_equation_nonlinear_unchanged(self):
        from core.invariant_algebra import Equation, Multiply, X, Number
        e = Equation(Multiply(X, X), Number(4)); self.assertEqual(canonical_equation(e), e)
    def test_canonical_sentence(self): self.assertEqual(canonical_sentence("la boule chute"), "la balle tombe")
    def test_canonical_program_output(self):
        for p in CODE_SAMPLES: self.assertEqual(run_program(canonical_program(p)), run_program(p))
    def test_canonical_program_rename_invariant(self): self.assertEqual(canonical_program(rename_variable(CODE_SAMPLES[0])), canonical_program(CODE_SAMPLES[0]))
    def test_canonical_geometry_invariant(self):
        P = torch.randn(4, 2, dtype=torch.float64); R = DOMAINS["geometry"][1]()[0](P)
        self.assertTrue(torch.allclose(canonical_geometry(P), canonical_geometry(R), atol=1e-9))
    def test_canonical_form_min(self):
        c, orb = canonical_form([3, 1, 2], [DomainTransformation("R", lambda s: s[1:] + s[:1])], cost=lambda s: tuple(s)); self.assertEqual(c, [1, 2, 3])
    def test_compression_function(self):
        r = compression(["aa", "aa", "aa"], lambda x: (x, [])); self.assertEqual(r["classes"], 1); self.assertGreater(r["ratio"], 1)
    def test_compression_experiment(self):
        r = compression_experiment(3, n=100)
        for d, v in r.items(): self.assertGreater(v["ratio"], 1.0, d); self.assertEqual(v["faithful"], 1.0, d)


# ================================================================ AJ — mini-domain code

class CodeTests(unittest.TestCase):
    def test_run(self): self.assertEqual(run_program(CODE_SAMPLES[0]), 14)
    def test_type_valid(self): self.assertTrue(type_valid(CODE_SAMPLES[1]))
    def test_type_invalid(self): self.assertFalse(type_valid([("a", "b"), ("b", 1)]))
    def test_fold(self): self.assertEqual(constant_fold([("a", ("+", 1, 2))]), [("a", 3)])
    def test_rename(self): self.assertEqual(run_program(rename_variable(CODE_SAMPLES[1])), run_program(CODE_SAMPLES[1]))
    def test_capture_changes(self): self.assertNotEqual(data_dependencies(rename_capture(CODE_SAMPLES[0])), data_dependencies(CODE_SAMPLES[0]))
    def test_reorder_independent_keeps_output(self):
        for p in CODE_SAMPLES: self.assertEqual(run_program(reorder_independent(p)), run_program(p))
    def test_reorder_independent_changes_surface(self): self.assertNotEqual(reorder_independent(CODE_SAMPLES[0]), CODE_SAMPLES[0])
    def test_reorder_any_breaks_types(self): self.assertFalse(type_valid(reorder_any(CODE_SAMPLES[2])))
    def test_extract(self):
        p = extract_expression(CODE_SAMPLES[0]); self.assertEqual(len(p), 4); self.assertEqual(run_program(p), 14)
    def test_dependencies_reorder_invariant(self): self.assertEqual(data_dependencies(reorder_independent(CODE_SAMPLES[0])), data_dependencies(CODE_SAMPLES[0]))
    def test_random_program_valid(self):
        rng = random.Random(0)
        for _ in range(20): self.assertTrue(type_valid(random_program(rng)))
    def test_code_transformations(self): self.assertEqual(len(code_transformations()), 6)


# ================================================================ Q/R/S — persistence, radius, map

class StabilityTests(unittest.TestCase):
    def test_persistence_prob(self): self.assertEqual(persistence(lambda m, s: s % 2 == 0, 0.1, 10).probability, 0.5)
    def test_wilson_interval(self):
        lo, hi = InvariantPersistence("I", 0.1, 8, 10).interval; self.assertLess(lo, 0.8); self.assertGreater(hi, 0.8)
    def test_interval_empty(self): self.assertEqual(InvariantPersistence("I", 0.1, 0, 0).interval, (0.0, 1.0))
    def test_radius_threshold_function(self):
        r = stability_radius(lambda m, s: m <= 0.3, 0.9, hi=1.0, trials=4, iters=10); self.assertAlmostEqual(r.radius, 0.3, places=2)
    def test_radius_zero(self): self.assertEqual(stability_radius(lambda m, s: False, 0.9, trials=2).radius, 0.0)
    def test_radius_full(self): self.assertEqual(stability_radius(lambda m, s: True, 0.9, hi=1.0, trials=2).radius, 1.0)
    def test_stable_region_contains(self): self.assertTrue(StableRegion("I", 0.3, 0.9).contains(0.2))
    def test_model_radius_positive(self):
        r = stability_radius(accuracy_invariant(M, XV, YV, ("head",), 0.98), 0.9, hi=1.0, trials=6, iters=2); self.assertGreater(r.radius, 0.05)
    def test_small_perturbation_survives(self): self.assertEqual(persistence(accuracy_invariant(M, XV, YV, ("enc",), 0.98), 0.01, 6).probability, 1.0)
    def test_large_perturbation_breaks(self): self.assertLess(persistence(accuracy_invariant(M, XV, YV, ("enc",), 0.98), 2.0, 6).probability, 0.5)
    def test_perturbed_model_pure(self):
        W = M.modules["enc"].slow["w1"].clone(); perturbed_model(M, ("enc",), 0.5, 0); self.assertTrue(torch.equal(W, M.modules["enc"].slow["w1"]))
    def test_perturbed_model_seeded(self):
        a, b = perturbed_model(M, ("enc",), 0.5, 3), perturbed_model(M, ("enc",), 0.5, 3); self.assertTrue(torch.equal(a.modules["enc"].slow["w1"], b.modules["enc"].slow["w1"]))
    def test_stability_map(self):
        mp = stability_map(M, XV, YV, 0.98, grid=(0.01, 2.0), trials=4); self.assertEqual(set(mp), {"enc", "expert_a", "head"}); self.assertEqual(mp["enc"][0.01], 1.0)
    def test_kam_inspired_disclaimer(self):
        src = open("core/stability.py").read(); self.assertIn("KAM-inspired", src); self.assertIn("NO Hamiltonian system", src)


# ================================================================ T/U/V — interactions, graph, transitions

class InteractionTransitionTests(unittest.TestCase):
    def test_additive_zero(self): self.assertAlmostEqual(interaction(lambda c: float(len(c)), ("a", "b")).interaction, 0.0)
    def test_nonadditive(self): self.assertAlmostEqual(interaction(lambda c: float(len(c) ** 2), ("a", "b")).interaction, 2.0)
    def test_relative(self): self.assertGreater(interaction(lambda c: float(len(c) ** 2), ("a", "b")).relative, 0.25)
    def test_graph_edges(self): G = PerturbationInteractionGraph(["a", "b", "c"], lambda c: float(len(c))); self.assertEqual(len(G.edges), 3); self.assertEqual(len(G.triples), 1)
    def test_graph_no_danger_additive(self): self.assertEqual(PerturbationInteractionGraph(["a", "b", "c"], lambda c: float(len(c))).dangerous(), [])
    def test_graph_detects_pair(self):
        G = PerturbationInteractionGraph(["a", "b", "c"], lambda c: 5.0 if set(c) >= {"a", "b"} else 0.1 * len(c))
        self.assertEqual(G.dangerous()[0].components, ("a", "b"))
    def test_graph_triple_pure(self):
        G = PerturbationInteractionGraph(["a", "b", "c"], lambda c: 10.0 if len(c) == 3 else 0.0); self.assertAlmostEqual(G.triples[("a", "b", "c")].interaction, 10.0)
    def test_graph_cache(self):
        calls = []; PerturbationInteractionGraph(["a", "b", "c"], lambda c: calls.append(c) or 0.0); self.assertEqual(len(calls), 7)
    def test_resonance_candidate_type(self): self.assertIsInstance(interaction(lambda c: 0.0, ("a",)), StructuralResonanceCandidate)
    def test_transition_step(self):
        tb = detect_transitions([0, 1, 2, 3, 4], lambda v: {"perf": 1.0 if v < 2.5 else 0.0}, refine=10)
        self.assertEqual(len(tb), 1); self.assertAlmostEqual(tb[0].threshold, 2.5, places=2)
    def test_transition_discrete(self):
        tb = detect_transitions([0, 1, 2], lambda v: {"branch": "A" if v < 1.3 else "B"}, refine=10); self.assertEqual(tb[0].kind, "switch"); self.assertAlmostEqual(tb[0].threshold, 1.3, places=2)
    def test_no_transition(self): self.assertEqual(detect_transitions([0, 1, 2], lambda v: {"perf": 1.0}), [])
    def test_transition_boundary_type(self): self.assertIsInstance(detect_transitions([0, 1], lambda v: {"q": v > 0.5})[0], TransitionBoundary)
    def test_routing_branch_transition(self):
        tb = detect_transitions([0, 1], routing_transition_observer(M, XV, YV)); self.assertTrue(any(b.quantity == "branch" for b in tb))


# ================================================================ W/X/Y — safe plasticity, fast weights, continual learning

class SafePlasticityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.res = {m: run_method(m, 0, base=OLD) for m in METHODS}
    def test_methods(self): self.assertEqual(len(METHODS), 5)
    def test_codes(self): self.assertEqual(len(CODES), 3)
    def test_task_data_dim(self): self.assertEqual(task_data("B")[0].shape[1], 12)
    def test_old_model_masters(self): self.assertGreater(min(INV.reference.values()), 0.95)
    def test_vanilla_forgets(self): self.assertGreater(self.res["vanilla"]["forgetting"], 0.2)
    def test_vanilla_learns(self): self.assertGreater(self.res["vanilla"]["new_task"], 0.9)
    def test_constrained_reduces_forgetting(self): self.assertLess(self.res["invariant_constrained"]["forgetting"], self.res["vanilla"]["forgetting"])
    def test_guided_reduces_forgetting(self): self.assertLess(self.res["perturbation_guided"]["forgetting"], self.res["vanilla"]["forgetting"])
    def test_guided_fewer_params(self): self.assertLess(self.res["perturbation_guided"]["parameters_changed"], self.res["vanilla"]["parameters_changed"])
    def test_replay_between(self): self.assertLess(self.res["replay"]["forgetting"], self.res["vanilla"]["forgetting"])
    def test_report_keys(self):
        for k in ("new_task", "retention", "parameters_changed", "rollbacks", "adaptation_cost"): self.assertIn(k, self.res["vanilla"])
    def test_fast_weights_slow_intact(self):
        m = copy.deepcopy(OLD); W = m.modules["head"].slow["w2"].clone(); xb, yb, _, _ = task_data("B", 100)
        load = __import__("core.safe_plasticity", fromlist=["load"]).load; load(m, train_fast(m, ("head",), xb, yb, 20)); self.assertTrue(torch.equal(W, m.modules["head"].slow["w2"]))
    def test_rollback_restores(self):
        m = copy.deepcopy(OLD); xb, yb, xbv, ybv = task_data("B", 100); mods = tuple(m.topological_order())
        rep = safe_consolidate(m, train_fast(m, mods, xb, yb, 300), xbv, ybv, INV, OLD.accuracy(xbv, ybv))
        self.assertEqual(rep.decision, "rollback"); self.assertEqual(m.fast_parameters, {}); self.assertTrue(all(INV.check(m).values()))
    def test_rollback_on_no_gain(self):
        m = copy.deepcopy(OLD); xb, yb, xbv, ybv = task_data("B", 100)
        zero = {mod: {k: torch.zeros_like(v) for k, v in m.modules[mod].slow.items()} for mod in m.topological_order()}
        rep = safe_consolidate(m, zero, xbv, ybv, INV, OLD.accuracy(xbv, ybv)); self.assertEqual(rep.decision, "rollback"); self.assertTrue(rep.reasons[0].startswith("gain"))
    def test_changed_parameters(self): self.assertEqual(changed_parameters({"m": {"w": torch.tensor([0.0, 1.0])}}), 1)
    def test_masked_training(self):
        m = copy.deepcopy(OLD); xb, yb, _, _ = task_data("B", 100)
        mask = {"head": {k: torch.zeros_like(v) for k, v in m.modules["head"].slow.items()}}
        self.assertEqual(changed_parameters(train_fast(m, ("head",), xb, yb, 10, mask=mask)), 0)


# ================================================================ Z–AC — learning the value of measurements, counterfactual

class MeasurementValueTests(unittest.TestCase):
    def rows(self, n=300, seed=0):
        rng = random.Random(seed); out = []
        for _ in range(n):
            f = [1.0] + [rng.random() for _ in range(9)] + [0.0] * 12; f[10 + rng.randrange(4)] = 1.0; f[14 + rng.randrange(4)] = 1.0
            y = 2.0 * f[1] - f[6] + rng.gauss(0, 0.05); out.append({"features": f, "realized": y, "analytic": f[1], "ig": f[1], "cost": f[6]})
        return out
    def test_fit_reduces_mse(self):
        L = MeasurementValueLearner().fit(self.rows()); self.assertLess(L.stats["mse_learned"], L.stats["mse_constant"])
    def test_hybrid_residual(self):
        L = MeasurementValueLearner().fit(self.rows()); self.assertLess(L.stats["mse_hybrid"], L.stats["mse_analytic"])
    def test_predict(self):
        rows = self.rows(); L = MeasurementValueLearner(ridge=1e-3).fit(rows); self.assertAlmostEqual(L.predict(rows[0]["features"]), rows[0]["realized"], delta=0.3)
    def test_adapt_reduces_error(self):
        rows = self.rows(); L = MeasurementValueLearner(fast_lr=0.5).fit(rows); f = rows[0]["features"]
        e0 = abs(L.adapt(f, 10.0)); e1 = abs(10.0 - L.predict(f)); self.assertLess(e1, e0)
    def test_reset_fast(self):
        rows = self.rows(); L = MeasurementValueLearner().fit(rows); L.adapt(rows[0]["features"], 5.0); L.reset_fast(); self.assertEqual(float(L.fast.abs().sum()), 0.0)
    def test_online_experiment(self):
        r = online_adaptation_experiment(self.rows(400), drift=-3.0); self.assertLess(r["mse_online"], r["mse_static"])
    def test_features_length(self): self.assertEqual(len(features(cubes(), color_op(), ("e1", "color"), 0.3, 0.1, {"intent": "where_now"})), 22)
    def test_planner_learned_policy(self):
        L = MeasurementValueLearner().fit(self.rows()); p = MeasurementPlanner(policy="learned"); p.value_model = L
        c = p.choose(cubes(), [color_op()], ("e1", "color")); self.assertTrue(c is None or c.operator is not None)
    def test_planner_hybrid_uses_analytic(self):
        L = MeasurementValueLearner().fit(self.rows()); p = MeasurementPlanner(policy="hybrid"); p.value_model = L
        s = p.score(cubes(), color_op(), ("e1", "color")); self.assertIsInstance(s.utility, float)
    def test_counterfactual_no_side_effect(self):
        S = cubes(); w = LatentWorldState({("e1", "color"): "red"}); cf = counterfactual_measurement(S, color_op(), ("e1", "color"))
        self.assertEqual(len(w.log), 0); self.assertAlmostEqual(S.marginal(("e1", "color"))["red"], 0.5); self.assertEqual(cf.status, "HYPOTHETICAL")
    def test_counterfactual_matches_ig(self):
        from core.measurement import information_gain
        S = cubes(); self.assertAlmostEqual(counterfactual_measurement(S, color_op(), ("e1", "color")).expected_reduction, information_gain(S, color_op(), ("e1", "color")), places=9)
    def test_counterfactual_probs(self): self.assertAlmostEqual(sum(counterfactual_measurement(cubes(), color_op()).outcome_probabilities.values()), 1.0)
    def test_counterfactual_type(self): self.assertIsInstance(counterfactual_measurement(cubes(), color_op()), CounterfactualMeasurement)


# ================================================================ AD — perturbation-guided diagnosis

class PerturbationDiagnosisTests(unittest.TestCase):
    def test_prior_normalized(self):
        from collections import Counter
        from core.model_diagnostics import CAUSES
        counts = {c: Counter({("a",): 3}) for c in CAUSES}; self.assertAlmostEqual(sum(prior_from_profile(counts, ("a",)).values()), 1.0)
    def test_prior_informative(self):
        from collections import Counter
        from core.model_diagnostics import CAUSES
        counts = {c: Counter({("x",): 5}) for c in CAUSES}; counts["fast_drift"] = Counter({("fast",): 5})
        p = prior_from_profile(counts, ("fast",)); self.assertEqual(max(p, key=p.get), "fast_drift")
    def test_probe_cost_positive(self): self.assertGreater(PROBE_COST, 0)


# ================================================================ AF/AG/AH — Fourier, reference frames, algebraic chains

class DomainSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.f = fourier_sensitivity(trials=20, eps=0.05)
    def test_spectral_invariant_stable_small_time(self): self.assertEqual(self.f["time_perturbation"]["line_set"], 1.0)
    def test_local_window_destroyed_by_frequency(self): self.assertEqual(self.f["frequency_perturbation"]["local_window"], 0.0)
    def test_local_window_mostly_kept_by_impulse(self): self.assertGreater(self.f["time_perturbation"]["local_window"], 0.7)
    def test_frame_distance(self): self.assertEqual(frame_perturbation(30)["distance"], 1.0)
    def test_frame_angle(self): self.assertEqual(frame_perturbation(30)["angle"], 1.0)
    def test_frame_coordinates_not_invariant(self): self.assertEqual(frame_perturbation(30)["coordinates"], 0.0)
    def test_algebra_chain(self):
        r = algebra_chain_perturbation(80); self.assertGreater(r["prediction_accuracy"], 0.9); self.assertGreater(r["ood_links"], 0)


# ================================================================ AL/AM — bench and sealed set

class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.r = evaluate_items(make_items(5, n_per=6))
    def test_families(self): self.assertEqual(set(self.r), set(FAMILIES))
    def test_duality_beats_always(self): self.assertGreater(self.r["composed"]["duality"], self.r["composed"]["always_preserved"])
    def test_duality_small(self): self.assertGreater(self.r["small"]["duality"], 0.8)
    def test_ood(self): self.assertGreater(self.r["ood"]["duality"], 0.8)
    def test_transition_error(self): self.assertLess(self.r["transition"]["abs_error"], 0.3)
    def test_interaction_measured(self): self.assertIn("relative_interaction", self.r["interaction"])
    def test_items_deterministic(self): self.assertEqual(len(make_items(9, 3)), len(make_items(9, 3)))
    def test_sealed_hash(self): self.assertEqual(sealed_hash(), SEALED_SHA256)
    def test_sealed_disjoint(self): self.assertNotEqual(str(make_items(1000, 16)[0]["x"]), str(sealed_items()[0]["x"]))
    def test_new_objects(self):
        rng = random.Random(0)
        for d in DOMAINS: self.assertIsNotNone(new_object(d, rng))


if __name__ == "__main__":
    unittest.main()

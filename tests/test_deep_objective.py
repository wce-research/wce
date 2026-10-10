import inspect
import json
import math
import unittest

import torch

import core.deep_experiments as DX
import core.deep_obj_report as OR
import core.deep_objective as DO
import core.deep_optgate as OG
import core.deep_world as DW
from core.boundary_program import BNode, predict
from core.deep_fit import ACCEPT_TOL, DT, FitCounter, FitData, fit_batch
from core.deep_optgate import (ACTIONS, DEFAULT_GATE, OBJECTIVES, OptimizationDecision, OptTrace, conditional_fit, hard_errors, make_objective, need_for_optimization, optimize,
                               pattern_search, profiled, signals)
from core.deep_search import DeepSearch
from core.deep_structure import normalize

G = lambda s=0: torch.Generator().manual_seed(s)
T = lambda c, ch: BNode("translate", params=c, children=(ch,))
C = lambda k, a, b: BNode(k, children=(a, b))
DISC = T((0.2, 0.0), BNode("pred", "norm", "le", (1.0,)))
HALF = BNode("pred", "dot", "le", (0.3, 0.1))
TR = normalize(C("xor", DISC, HALF))
SK = TR.skeleton()
P0 = torch.tensor(TR.param_vector(), dtype=DT)
GATE = {"objective": "surrogate", "residual_z": 2.0, "gap_z": 1.0, "min_gain": 0.0}


def X0(n=300, s=0): return torch.rand(n, 2, generator=G(s)) * 6 - 3


def data_for(prog, n=300, rate=0.05, s=0):
    X = X0(n, s); y = predict(prog, X); f = torch.rand(n, generator=G(s + 1)) < rate; return FitData.build(X, torch.where(f, 1 - y, y), 0.10)


D = data_for(TR)
BAD = P0 + torch.tensor([0.6, -0.5, 0.4, 0.5, 0.6][: len(P0)], dtype=DT)


class ObjectiveTests(unittest.TestCase):
    def test_list(self): self.assertEqual(OBJECTIVES, ("surrogate", "profiled_soft", "discrete_error", "likelihood_prior", "local_boundary", "robust"))
    def test_each_objective_finite(self):
        for o in OBJECTIVES:
            if o == "discrete_error": continue
            v = make_objective(o, SK, D, P0)(P0.reshape(1, -1), 24.0); self.assertEqual(v.shape, (1,)); self.assertTrue(math.isfinite(float(v[0])), o)
    def test_unknown(self):
        with self.assertRaises(KeyError): make_objective("x", SK, D, P0)
    def test_objectives_are_separate_not_fused(self): self.assertIn("testés SÉPARÉMENT", inspect.getsource(OG)); self.assertEqual(len({float(make_objective(o, SK, D, P0)(BAD.reshape(1, -1), 24.0)[0]) for o in OBJECTIVES if o != "discrete_error"}), 5)
    def test_surrogate_matches_fit_batch_objective(self):
        tr = optimize(SK, D, P0, "surrogate"); ref, _ = fit_batch(SK, D, P0.reshape(1, -1), steps=100); self.assertEqual(float((tr.final - ref[0]).abs().max()), 0.0)
    def test_prior_zero_at_initial(self):
        a = make_objective("likelihood_prior", SK, D, P0)(P0.reshape(1, -1), 24.0); b = make_objective("surrogate", SK, D, P0)(P0.reshape(1, -1), 24.0); self.assertAlmostEqual(float(a[0]), float(b[0]), places=9)
    def test_prior_penalizes_displacement(self):
        a = make_objective("likelihood_prior", SK, D, P0)(BAD.reshape(1, -1), 24.0); b = make_objective("surrogate", SK, D, P0)(BAD.reshape(1, -1), 24.0); self.assertGreater(float(a[0]), float(b[0]))
    def test_prior_moves_less(self): self.assertLess(float((optimize(SK, D, P0, "likelihood_prior").final - P0).abs().max()), float((optimize(SK, D, P0, "surrogate").final - P0).abs().max()) + 1e-9)
    def test_robust_differs(self): self.assertNotEqual(float(make_objective("robust", SK, D, P0)(P0.reshape(1, -1), 24.0)[0]), float(make_objective("surrogate", SK, D, P0)(P0.reshape(1, -1), 24.0)[0]))
    def test_local_boundary_downweights_far_points(self): self.assertLess(float(make_objective("local_boundary", SK, D, P0)(P0.reshape(1, -1), 24.0)[0]), float(make_objective("surrogate", SK, D, P0)(P0.reshape(1, -1), 24.0)[0]))
    def test_profiled_soft_true_beats_bad(self): f = make_objective("profiled_soft", SK, D, P0); self.assertLess(float(f(P0.reshape(1, -1), 24.0)[0]), float(f(BAD.reshape(1, -1), 24.0)[0]))
    def test_no_truth_in_objectives(self):
        src = inspect.getsource(OG); self.assertNotIn("deep_world", src); self.assertNotIn(".truth", src); self.assertNotIn("world.label", src); self.assertNotIn("sealed", src)
    def test_hard_errors(self): self.assertEqual(float(hard_errors(SK, P0.reshape(1, -1), D)[0]), float((predict(TR, D.X.float()).to(DT) != D.y).sum()))
    def test_same_mechanism_for_gradient_objectives(self): src = inspect.getsource(optimize); self.assertIn("torch.optim.Adam([th], lr=lr)", src); self.assertIn("4.0 + 20.0 * t / steps", src)


class TraceTests(unittest.TestCase):
    def test_initial_never_lost(self):
        for o in OBJECTIVES: tr = optimize(SK, D, P0, o); self.assertTrue(torch.equal(tr.initial, P0), o); self.assertIsInstance(tr, OptTrace)
    def test_input_not_mutated(self): init = P0.clone(); optimize(SK, D, init, "surrogate"); self.assertTrue(torch.equal(init, P0))
    def test_best_never_worse_than_initial_on_valid_criterion(self):
        for o in OBJECTIVES: tr = optimize(SK, D, P0, o); self.assertGreaterEqual(profiled(SK, tr.best, D), profiled(SK, tr.initial, D) - 1e-12, o)
    def test_best_never_worse_than_final(self):
        for o in ("surrogate", "robust", "local_boundary"): tr = optimize(SK, D, BAD, o); self.assertGreaterEqual(profiled(SK, tr.best, D), profiled(SK, tr.final, D) - 1e-12, o)
    def test_three_states_kept(self): tr = optimize(SK, D, BAD, "surrogate"); self.assertEqual({len(tr.initial), len(tr.best), len(tr.final)}, {len(P0)}); self.assertIn(tr.best_step, [s["step"] for s in tr.steps])
    def test_steps_logged(self): tr = optimize(SK, D, P0, "surrogate"); self.assertEqual([s["step"] for s in tr.steps], list(range(0, 101, 10))); self.assertEqual(set(tr.steps[0]), {"step", "objective", "profiled", "errors", "params"})
    def test_deterministic(self): a, b = optimize(SK, D, BAD, "robust"), optimize(SK, D, BAD, "robust"); self.assertTrue(torch.equal(a.final, b.final)); self.assertTrue(torch.equal(a.best, b.best))
    def test_pattern_search_monotone(self): tr = pattern_search(SK, D, BAD.reshape(1, -1)); e = [s["errors"] for s in tr.steps]; self.assertTrue(all(b <= a for a, b in zip(e, e[1:])))
    def test_pattern_search_never_increases_errors_from_truth(self): tr = optimize(SK, D, P0, "discrete_error"); self.assertLessEqual(float(hard_errors(SK, tr.final.reshape(1, -1), D)[0]), float(hard_errors(SK, P0.reshape(1, -1), D)[0]))
    def test_pattern_search_improves_bad_start(self): tr = optimize(SK, D, BAD, "discrete_error"); self.assertLess(float(hard_errors(SK, tr.final.reshape(1, -1), D)[0]), float(hard_errors(SK, BAD.reshape(1, -1), D)[0]))
    def test_budget_charged(self): c = FitCounter(); optimize(SK, D, P0, "surrogate", counter=c); self.assertEqual(c.fit_equivalents, 1.0)


class SignalTests(unittest.TestCase):
    def test_fields(self): self.assertEqual(set(signals(SK, D, P0)), {"error_rate", "errors", "profiled_per_point", "residual_z", "gap_z", "knn_error", "grad_per_point", "probe_delta_errors", "n", "n_params"})
    def test_no_truth_argument(self): self.assertEqual(list(inspect.signature(signals).parameters), ["skel", "data", "init", "gamma"])
    def test_good_candidate_low_error(self): self.assertLess(signals(SK, D, P0)["error_rate"], 0.12)
    def test_bad_candidate_flags_structure(self): a, b = signals(SK, D, P0), signals(SK, D, BAD); self.assertGreater(b["error_rate"], a["error_rate"]); self.assertGreater(b["residual_z"], a["residual_z"]); self.assertGreater(b["gap_z"], a["gap_z"])
    def test_deterministic(self): self.assertEqual(signals(SK, D, BAD), signals(SK, D, BAD))
    def test_computed_before_fit(self): src = inspect.getsource(conditional_fit); self.assertLess(src.index("signals("), src.index("optimize("))
    def test_json(self): json.dumps(signals(SK, D, P0))


class GateTests(unittest.TestCase):
    def sig(self, rz, gz): return {"error_rate": 0.12, "knn_error": 0.10, "n": 300, "residual_z": rz, "gap_z": gz}
    def test_actions(self): self.assertEqual(ACTIONS, ("PRESERVE", "OPTIMIZE"))
    def test_preserve_when_residual_is_noise(self): self.assertEqual(need_for_optimization(self.sig(0.0, -1.0), GATE)[0], "PRESERVE")
    def test_optimize_when_residual_structured(self): a, why, _, _ = need_for_optimization(self.sig(5.0, -1.0), GATE); self.assertEqual(a, "OPTIMIZE"); self.assertIn("structuré", why)
    def test_optimize_when_worse_than_local_reference(self): a, why, _, _ = need_for_optimization(self.sig(0.0, 3.0), GATE); self.assertEqual(a, "OPTIMIZE"); self.assertIn("référence locale", why)
    def test_threshold_boundary(self): self.assertEqual(need_for_optimization(self.sig(2.0, -9.0), GATE)[0], "OPTIMIZE"); self.assertEqual(need_for_optimization(self.sig(1.999, -9.0), GATE)[0], "PRESERVE")
    def test_gain_and_risk_estimates(self):
        _, _, g, r = need_for_optimization(self.sig(0.0, -1.0), GATE); self.assertAlmostEqual(g, 6.0); self.assertTrue(0.5 < r < 1.0); self.assertLess(need_for_optimization(self.sig(6.0, 4.0), GATE)[3], r)
    def test_gate_is_explicit_rule_not_learned(self):
        src = inspect.getsource(OG); self.assertNotIn("torch.nn", src); self.assertNotIn("nn.Module", src); self.assertIn("règle à seuils", src)
    def test_default_gate(self): self.assertEqual(set(DEFAULT_GATE), {"objective", "residual_z", "gap_z", "min_gain"})
    def test_preserve_returns_initial_untouched(self):
        d = conditional_fit(SK, D, P0, {**GATE, "residual_z": 99.0, "gap_z": 99.0}); self.assertEqual(d.action, "PRESERVE"); self.assertEqual(list(d.result), P0.tolist()); self.assertIsNone(d.proposal); self.assertFalse(d.rolled_back)
    def test_preserve_costs_nothing(self): c = FitCounter(); conditional_fit(SK, D, P0, {**GATE, "residual_z": 99.0, "gap_z": 99.0}, counter=c); self.assertEqual(c.fit_equivalents, 0.0)
    def test_optimize_then_validate(self):
        d = conditional_fit(SK, D, BAD, {**GATE, "residual_z": -99.0}); self.assertEqual(d.action, "OPTIMIZE"); self.assertIsNotNone(d.proposal); self.assertIsNotNone(d.best_so_far); self.assertEqual(d.rolled_back, not d.accepted)
        self.assertEqual(list(d.result), list(d.best_so_far) if d.accepted else BAD.tolist())
    def test_result_never_worse_on_criterion(self):
        for init in (P0, BAD):
            d = conditional_fit(SK, D, init, {**GATE, "residual_z": -99.0}); self.assertGreaterEqual(profiled(SK, torch.tensor(d.result, dtype=DT), D), profiled(SK, init, D) - 1e-12)
    def test_min_gain_forces_rollback(self):
        d = conditional_fit(SK, D, P0, {**GATE, "residual_z": -99.0, "min_gain": 1e9}); self.assertTrue(d.rolled_back); self.assertEqual(list(d.result), P0.tolist()); self.assertEqual(list(d.initial), P0.tolist())
    def test_decision_is_auditable(self):
        d = conditional_fit(SK, D, BAD, {**GATE, "residual_z": -99.0}); self.assertIsInstance(d, OptimizationDecision)
        for f in ("initial", "signals", "gain_estimate", "risk_estimate", "action", "reason", "objective", "proposal", "best_so_far", "value_before", "value_after", "accepted", "rolled_back", "result"): self.assertTrue(hasattr(d, f))
        json.dumps(d.to_dict())
    def test_decision_keeps_initial_state(self): d = conditional_fit(SK, D, BAD, {**GATE, "residual_z": -99.0}); self.assertEqual(list(d.initial), BAD.tolist())
    def test_deterministic(self): a = conditional_fit(SK, D, BAD, {**GATE, "residual_z": -99.0}); b = conditional_fit(SK, D, BAD, {**GATE, "residual_z": -99.0}); self.assertEqual(a.to_dict(), b.to_dict())
    def test_gate_objective_selectable(self): d = conditional_fit(SK, D, BAD, {**GATE, "residual_z": -99.0, "objective": "discrete_error"}); self.assertEqual(d.objective, "discrete_error")
    def test_gate_never_uses_truth(self): self.assertEqual(list(inspect.signature(conditional_fit).parameters), ["skel", "data", "init", "gate", "counter", "steps"])


class PipelinePolicyTests(unittest.TestCase):
    def kw(self, **k): return dict(budget=60, max_depth=3, seed=1, fact_literals=[DISC, HALF], fact_residual=False, **k)
    def test_policies(self): self.assertEqual(DO.E2E_POLICIES, {"always": None, "never": "never", "validated": "profiled", "gate": "gate"})
    def test_never_keeps_candidate_parameters(self):
        o = DeepSearch("factorized", **self.kw(validated="never")).run(D); P = X0(2000, 9); self.assertGreater(float((predict(o.best().program, P) == predict(TR, P)).float().mean()), 0.995)
    def test_never_spends_less(self):
        a = DeepSearch("factorized", **self.kw()).run(D); b = DeepSearch("factorized", **self.kw(validated="never")).run(D); self.assertLess(b.stats["fit_equivalents"], a.stats["fit_equivalents"])
    def test_gate_records_decisions(self):
        o = DeepSearch("factorized", **self.kw(validated="gate")).run(D); self.assertGreater(o.stats["decisions"], 0); self.assertLessEqual(o.stats["preserved"], o.stats["decisions"])
    def test_always_is_frozen_default(self): a = DeepSearch("factorized", **self.kw()).run(D); self.assertEqual((a.stats["transactions"], a.stats["decisions"]), (0, 0))
    def test_only_fit_stage_changes(self):
        import core.deep_search as DS
        for m in (DS.DeepSearch._simplify, DS.DeepSearch._ranked, DS.DeepSearch._register, DS.DeepSearch._run_factorized): self.assertNotIn("conditional_fit", inspect.getsource(m)); self.assertNotIn('"never"', inspect.getsource(m))
        self.assertIn("conditional_fit", inspect.getsource(DS.DeepSearch._fit))
    def test_gate_deterministic(self):
        a = DeepSearch("factorized", **self.kw(validated="gate")).run(D); b = DeepSearch("factorized", **self.kw(validated="gate")).run(D); self.assertEqual(a.best().program, b.best().program)


class StudyTests(unittest.TestCase):
    def test_perturbation_levels(self): self.assertEqual(DO.PERTURBATIONS, {"low": 0.03, "medium": 0.10, "high": 0.30})
    def test_three_regimes(self):
        c = DO.candidates_for(DX.down_task(300, 2, "standard"), 300); self.assertEqual([x[0] for x in c][:4], ["A", "B", "B", "B"]); self.assertTrue(all(x[0] == "C" for x in c[4:])); self.assertTrue(1 <= len(c) - 4 <= 3)
    def test_regime_a_is_truth(self): t = DX.down_task(300, 2, "standard"); c = DO.candidates_for(t, 300); self.assertEqual(c[0][2], normalize(t.world.truth)); self.assertEqual(c[0][3], 0.0)
    def test_perturbation_grows(self): c = DO.candidates_for(DX.down_task(300, 3, "standard"), 300); self.assertLess(c[1][3], c[2][3]); self.assertLess(c[2][3], c[3][3])
    def test_perturbed_keep_structure(self): c = DO.candidates_for(DX.down_task(300, 3, "standard"), 300); self.assertEqual({str(x[2].skeleton()) for x in c[:4]}.__len__(), 1)
    def test_regime_c_uses_discovered_atoms(self): src = inspect.getsource(DO.candidates_for); self.assertIn("_atom_literals", src); self.assertNotIn("leaves(tr)", src)
    def test_candidates_deterministic(self): a = DO.candidates_for(DX.down_task(300, 2, "rich"), 5); b = DO.candidates_for(DX.down_task(300, 2, "rich"), 5); self.assertEqual([x[2] for x in a], [x[2] for x in b])
    def test_task_sets_disjoint(self):
        ids = lambda ks: {DX.down_task(k, d, "standard").world.uid for k in ks for d in DX.DOWN_DEPTHS}; tune, ev, old = ids(range(300, 308)), ids(range(200, 216)), ids(list(range(16)) + list(range(100, 116)))
        self.assertEqual(len(tune & ev), 0); self.assertEqual(len((tune | ev) & old), 0)
    def test_task_set_labels(self): self.assertIn('"tuning" if int(key) >= 300 else "evaluation"', inspect.getsource(DX.obj_subunits))
    def test_dev_only(self): self.assertEqual(DX.down_task(307, 5, "rich").split, "dev"); self.assertLess(DX.down_task(307, 5, "rich").seed, DW.SPLIT_BASE["validation"]); self.assertEqual(DW.sealed_hash(), DW.SEALED_SHA256)
    def test_subunits(self): self.assertEqual(len(DX.subunit_ids("deepobj", 300)), 8); self.assertEqual(len(DX.subunit_ids("deepe2e", 200)), 8 * 2 * 4); self.assertEqual(len(DX.subunit_ids("deepresid", 0)), len(DO.RESIDUAL_TASKS))
    def test_residual_tasks_cover_degraded(self): self.assertEqual(len(DO.RESIDUAL_TASKS), 9); self.assertEqual(sum(int(k) >= 100 for k, _, _ in DO.RESIDUAL_TASKS), 6); self.assertIn(("104", 4, "standard"), DO.RESIDUAL_TASKS); self.assertIn(("9", 4, "standard"), DO.RESIDUAL_TASKS)
    def test_residual_explanations(self):
        src = inspect.getsource(DO.residual_case)
        for e in ("NOISE_EXPLOITED_ACCEPTED", "DATA_PREFER_WORSE_FUNCTION", "PATH_CHANGED_STRUCTURE", "NOT_DEGRADED"): self.assertIn(e, src)
    def test_no_sealed(self):
        for m in (DO, OR, OG): self.assertNotIn("sealed_task", inspect.getsource(m)); self.assertNotIn("sealed_spec", inspect.getsource(m))
    def test_make_targets(self):
        mk = open("Makefile").read()
        for t in ("deep-run-objectives:", "deep-run-residual:", "deep-tune-gate:", "deep-run-e2e:", "deep-obj-aggregate:", "deep-obj-report:"): self.assertIn(t, mk)


def _row(init, final, best, pi, pf, pb, rz=0.0, gz=0.0):
    return {"initial": {"accord": init, "profiled": pi}, "signals": {"error_rate": 0.1, "knn_error": 0.1, "n": 300, "residual_z": rz, "gap_z": gz}, "objectives": {"surrogate": {"final": {"accord": final, "profiled": pf}, "best": {"accord": best, "profiled": pb}}}}


class PolicyEvaluationTests(unittest.TestCase):
    def test_never(self): self.assertEqual(OR.outcome(_row(1.0, 0.9, 0.95, -10, -9, -9), "never"), (1.0, "preserve"))
    def test_always(self): self.assertEqual(OR.outcome(_row(1.0, 0.9, 0.95, -10, -9, -9), "always"), (0.9, "optimize"))
    def test_validated_accepts_on_data_not_truth(self): self.assertEqual(OR.outcome(_row(1.0, 0.9, 0.95, -10, -9, -9), "validated"), (0.9, "accept"))          # more likely on the data → accepted, even if worse in reality
    def test_validated_rollback(self): self.assertEqual(OR.outcome(_row(1.0, 0.9, 0.95, -10, -11, -10), "validated"), (1.0, "rollback"))
    def test_validated_equality_rollback(self): self.assertEqual(OR.outcome(_row(1.0, 0.9, 0.95, -10, -10, -10), "validated")[1], "rollback")
    def test_validated_best(self): self.assertEqual(OR.outcome(_row(0.8, 0.7, 0.95, -20, -25, -12), "validated_best"), (0.95, "accept"))
    def test_gate_preserve(self): self.assertEqual(OR.outcome(_row(1.0, 0.9, 0.95, -10, -9, -9), "gate", gate=GATE), (1.0, "preserve"))
    def test_gate_optimize_accept(self): self.assertEqual(OR.outcome(_row(0.8, 0.7, 0.95, -20, -25, -12, rz=5.0), "gate", gate=GATE), (0.95, "accept"))
    def test_gate_min_gain(self): self.assertEqual(OR.outcome(_row(0.8, 0.7, 0.95, -20, -25, -19, rz=5.0), "gate", gate={**GATE, "min_gain": 4.0}), (0.8, "rollback"))
    def test_rates(self):
        rows = [_row(1.0, 0.9, 1.0, -10, -9, -9), _row(1.0, 1.0, 1.0, -10, -10, -10), _row(0.8, 0.95, 0.95, -20, -12, -12), _row(0.8, 0.8, 0.8, -20, -20, -20)]; r = OR.rates(rows, "always", "surrogate")
        self.assertEqual((r["n_good"], r["n_bad"]), (2, 2)); self.assertEqual((r["good_preserved"], r["good_destroyed"], r["bad_improved"], r["bad_not_repaired"]), (0.5, 0.5, 0.5, 0.5))
    def test_rates_never(self): r = OR.rates([_row(1.0, 0.9, 1.0, -10, -9, -9), _row(0.8, 0.95, 0.95, -20, -12, -12)], "never", "surrogate"); self.assertEqual((r["good_preserved"], r["bad_improved"], r["optimized"]), (1.0, 0.0, 0.0))
    def test_rates_sum(self):
        r = OR.rates([_row(1.0, 0.9, 1.0, -10, -9, -9), _row(0.8, 0.95, 0.95, -20, -12, -12)], "validated", "surrogate"); self.assertAlmostEqual(r["good_preserved"] + r["good_destroyed"], 1.0); self.assertAlmostEqual(r["bad_improved"] + r["bad_not_repaired"], 1.0)
    def test_policies_listed(self): self.assertEqual(OR.POLICIES, ("always", "never", "validated", "validated_best", "gate"))
    def test_tuning_only_on_tuning_set(self): src = inspect.getsource(OR.tune); self.assertIn('r["task_set"] == "tuning"', src); self.assertNotIn('"evaluation"', src)
    def test_policies_evaluated_on_evaluation_set(self): self.assertIn('E = [r for r in rows if r["task_set"] == "evaluation"]', inspect.getsource(OR.aggregate))
    def test_decisions_use_observables(self):
        src = inspect.getsource(OR.outcome); self.assertIn('["profiled"]', src); i = src.index('if policy == "validated"'); self.assertNotIn('["accord"] >', src[i:]); self.assertNotIn('["accord"] <', src[i:])


if __name__ == "__main__":
    unittest.main()

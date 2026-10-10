import inspect
import json
import math
import unittest

import torch

import core.deep_experiments as DX
import core.deep_fit as DF
import core.deep_validated as DV
import core.deep_vo_report as VR
import core.deep_world as DW
from core.boundary_program import BNode, predict
from core.deep_downstream import TracedSearch
from core.deep_fit import ACCEPT_TOL, CRITERIA, DT, FitCounter, FitData, FitTransaction, decide, fit_batch, fit_criterion, profiled_loglik, validated_fit
from core.deep_search import DeepSearch
from core.deep_structure import n_atoms, normalize
from core.deep_validated import ARMS, CONFIGS, arm_run, local_geometry, optimize_traced, optimizer_diagnosis

G = lambda s=0: torch.Generator().manual_seed(s)
T = lambda c, ch: BNode("translate", params=c, children=(ch,))
C = lambda k, a, b: BNode(k, children=(a, b))
DISC = T((0.2, 0.0), BNode("pred", "norm", "le", (1.0,)))
HALF = BNode("pred", "dot", "le", (0.3, 0.1))
TR = normalize(C("xor", DISC, HALF))


def X0(n=300, s=0): return torch.rand(n, 2, generator=G(s)) * 6 - 3


def data_for(prog, n=300, rate=0.05, s=0):
    X = X0(n, s); y = predict(prog, X); f = torch.rand(n, generator=G(s + 1)) < rate; return FitData.build(X, torch.where(f, 1 - y, y), 0.10)


class DecisionRuleTests(unittest.TestCase):
    def test_accept_improvement(self): self.assertEqual(decide(-10.0, -9.0, True), (True, "improved"))
    def test_reject_degradation(self): self.assertEqual(decide(-9.0, -10.0, True), (False, "not_improved"))
    def test_equality_keeps_initial(self): self.assertEqual(decide(-9.0, -9.0, True), (False, "equal"))
    def test_within_tolerance_is_equality(self): self.assertEqual(decide(-9.0, -9.0 + 0.5 * ACCEPT_TOL, True), (False, "equal")); self.assertEqual(decide(-9.0, -9.0 - 0.5 * ACCEPT_TOL, True), (False, "equal"))
    def test_just_above_tolerance(self): self.assertEqual(decide(-9.0, -9.0 + 3 * ACCEPT_TOL, True), (True, "improved"))
    def test_nan_proposal_rolled_back(self): self.assertEqual(decide(-9.0, float("nan"), True), (False, "non_finite_proposal"))
    def test_inf_proposal_rolled_back(self): self.assertEqual(decide(-9.0, float("inf"), True), (False, "non_finite_proposal")); self.assertEqual(decide(-9.0, float("-inf"), True), (False, "non_finite_proposal"))
    def test_non_finite_parameters_rolled_back(self): self.assertEqual(decide(-9.0, -1.0, False), (False, "non_finite_proposal"))
    def test_non_finite_before_accepts_finite(self): self.assertEqual(decide(float("nan"), -5.0, True), (True, "non_finite_before"))
    def test_both_non_finite(self): self.assertEqual(decide(float("nan"), float("nan"), True), (False, "non_finite_proposal"))
    def test_deterministic(self): self.assertEqual([decide(-3.0, -2.5, True) for _ in range(3)], [(True, "improved")] * 3)
    def test_tolerance_documented(self): self.assertEqual(ACCEPT_TOL, 1e-9); self.assertIn("equality up to the tolerance", inspect.getsource(decide)); self.assertIn("rollback", inspect.getsource(decide))
    def test_criteria(self): self.assertEqual(CRITERIA, ("profiled", "surrogate"))


class TransactionTests(unittest.TestCase):
    def setUp(self): self.d = data_for(TR)
    def test_initial_state_preserved(self):
        init = torch.tensor(TR.param_vector(), dtype=DT); keep = init.clone(); validated_fit(TR.skeleton(), self.d, init, steps=30); self.assertTrue(torch.equal(init, keep))
    def test_rollback_returns_initial(self):
        init = torch.tensor(TR.param_vector(), dtype=DT); th, tx = validated_fit(TR.skeleton(), self.d, init, steps=100)
        if not tx.accepted: self.assertTrue(torch.equal(th, init)); self.assertEqual(tx.committed, tx.before)
        else: self.assertEqual(tx.committed, tx.proposal); self.assertGreater(tx.value_after, tx.value_before)
    def test_never_worse_on_criterion(self):
        for s in range(6):
            d = data_for(TR, s=s); init = torch.tensor(TR.param_vector(), dtype=DT); th, tx = validated_fit(TR.skeleton(), d, init, steps=60)
            self.assertGreaterEqual(fit_criterion(TR.skeleton(), th, d), fit_criterion(TR.skeleton(), init, d) - 1e-12)
    def test_accepts_improvement_from_bad_start(self):
        bad = torch.tensor([0.9, 0.6, 0.5, 0.3, 0.1], dtype=DT) if len(TR.param_vector()) == 5 else torch.tensor(TR.param_vector(), dtype=DT) + 0.6
        th, tx = validated_fit(TR.skeleton(), self.d, bad, steps=100); self.assertEqual(tx.accepted, tx.value_after > tx.value_before + ACCEPT_TOL)
    def test_zero_steps_is_equal_rollback(self):
        init = torch.tensor(TR.param_vector(), dtype=DT); th, tx = validated_fit(TR.skeleton(), self.d, init, steps=0); self.assertFalse(tx.accepted); self.assertEqual(tx.reason, "equal"); self.assertTrue(torch.equal(th, init))
    def test_transaction_immutable(self):
        _, tx = validated_fit(TR.skeleton(), self.d, torch.tensor(TR.param_vector(), dtype=DT), steps=10)
        with self.assertRaises(Exception): tx.accepted = True
    def test_transaction_fields(self):
        _, tx = validated_fit(TR.skeleton(), self.d, torch.tensor(TR.param_vector(), dtype=DT), steps=10); self.assertIsInstance(tx, FitTransaction)
        for f in ("before", "proposal", "criterion", "value_before", "value_after", "accepted", "reason"): self.assertTrue(hasattr(tx, f))
    def test_before_is_the_given_state(self): _, tx = validated_fit(TR.skeleton(), self.d, torch.tensor(TR.param_vector(), dtype=DT), steps=10); self.assertEqual(list(tx.before), TR.param_vector())
    def test_deterministic(self):
        a = validated_fit(TR.skeleton(), self.d, torch.tensor(TR.param_vector(), dtype=DT), steps=40); b = validated_fit(TR.skeleton(), self.d, torch.tensor(TR.param_vector(), dtype=DT), steps=40); self.assertTrue(torch.equal(a[0], b[0])); self.assertEqual(a[1], b[1])
    def test_criterion_uses_observations_only(self): self.assertEqual(list(inspect.signature(fit_criterion).parameters), ["skel", "theta_row", "data", "criterion", "gamma"]); self.assertNotIn("truth", inspect.getsource(validated_fit)); self.assertNotIn("world", inspect.getsource(validated_fit))
    def test_profiled_criterion_value(self):
        th = torch.tensor(TR.param_vector(), dtype=DT); self.assertAlmostEqual(fit_criterion(TR.skeleton(), th, self.d, "profiled"), profiled_loglik(predict(TR, self.d.X.float()).to(DT), self.d), places=9)
    def test_surrogate_criterion_differs(self): th = torch.tensor(TR.param_vector(), dtype=DT); self.assertNotEqual(fit_criterion(TR.skeleton(), th, self.d, "profiled"), fit_criterion(TR.skeleton(), th, self.d, "surrogate"))
    def test_unknown_criterion(self):
        with self.assertRaises(KeyError): fit_criterion(TR.skeleton(), torch.tensor(TR.param_vector(), dtype=DT), self.d, "x")
    def test_optimizer_unchanged(self):
        init = torch.tensor(TR.param_vector(), dtype=DT); _, tx = validated_fit(TR.skeleton(), self.d, init, steps=50); ref, _ = fit_batch(TR.skeleton(), self.d, init.reshape(1, -1), steps=50); self.assertEqual(list(tx.proposal), ref[0].tolist())
    def test_budget_charged_even_on_rollback(self): c = FitCounter(); validated_fit(TR.skeleton(), self.d, torch.tensor(TR.param_vector(), dtype=DT), steps=100, counter=c); self.assertEqual(c.fit_equivalents, 1.0)
    def test_propose_evaluate_commit_semantics(self):
        src = inspect.getsource(validated_fit); self.assertLess(src.index("fit_batch("), src.index("decide(")); self.assertIn("frozen = before.clone()", src); self.assertIn("the optimiser modified the initial state in place", src)


class PipelineIsolationTests(unittest.TestCase):
    """The guardrail touches ONLY the fitting step."""
    def setUp(self): self.d = data_for(TR)
    def kw(self, **k): return dict(budget=60, max_depth=3, seed=1, fact_literals=[DISC, HALF], fact_residual=False, **k)
    def test_default_is_frozen_pipeline(self): self.assertIsNone(DeepSearch("factorized").validated); self.assertIsNone(ARMS["A"]); self.assertEqual(ARMS["B"], "profiled")
    def test_arm_a_identical_to_previous_pipeline(self):
        a = DeepSearch("factorized", **self.kw()).run(self.d); b = DeepSearch("factorized", **self.kw(validated=None)).run(self.d); self.assertEqual([h.program for h in a.hypotheses], [h.program for h in b.hypotheses]); self.assertEqual(a.stats["transactions"], 0)
    def test_same_candidates_generated(self):
        a = TracedSearch("factorized", **self.kw()); a.run(self.d); b = TracedSearch("factorized", **self.kw(validated="profiled")); b.run(self.d)
        r0 = lambda s: [f["inits"][0] for f in s.fits if f["origin"] == "factorized_r0"]; self.assertEqual(r0(a), r0(b))                      # the round-0 generator receives and produces the same thing
    def test_same_fit_cost_per_fit(self):
        a = DeepSearch("factorized", **self.kw()).run(self.d); b = DeepSearch("factorized", **self.kw(validated="profiled")).run(self.d); self.assertGreater(b.stats["transactions"], 0); self.assertEqual(b.stats["commits"] + b.stats["rollbacks"], b.stats["transactions"])
    def test_heuristic_multi_init_fits_not_guarded(self):
        s = DeepSearch("bottom_up", budget=20, seed=0, validated="profiled"); s._setup(self.d, 0.1); s._fit_atoms(); self.assertEqual(len(s.transactions), 0)             # no initial state to protect: K random initialisations
    def test_guard_only_in_fit(self):
        import core.deep_search as DS
        src = inspect.getsource(DS); self.assertEqual(src.count("validated_fit("), 1); self.assertIn("validated_fit(", inspect.getsource(DS.DeepSearch._fit))
        for m in (DS.DeepSearch._simplify, DS.DeepSearch._ranked, DS.DeepSearch._register, DS.DeepSearch._run_factorized): self.assertNotIn("validated", inspect.getsource(m))
    def test_scoring_unchanged(self):
        import core.deep_search as DS
        self.assertIn("approx_evidence(prog.n_params(), d, ll, self.n)", inspect.getsource(DS.DeepSearch._register)); self.assertIn("math.log(3.0)", inspect.getsource(DS.DeepSearch._simplify))
    def test_final_never_less_likely_than_frozen_on_fixture(self):
        a = DeepSearch("factorized", **self.kw()).run(self.d); b = DeepSearch("factorized", **self.kw(validated="profiled")).run(self.d); self.assertGreaterEqual(b.best().evidence, a.best().evidence - 1e-9)
    def test_guarded_run_deterministic(self):
        a = DeepSearch("factorized", **self.kw(validated="profiled")).run(self.d); b = DeepSearch("factorized", **self.kw(validated="profiled")).run(self.d); self.assertEqual(a.best().program, b.best().program); self.assertEqual(a.stats["commits"], b.stats["commits"])
    def test_stats_report_transactions(self):
        st = DeepSearch("factorized", **self.kw(validated="profiled")).run(self.d).stats
        for f in ("validated", "transactions", "commits", "rollbacks"): self.assertIn(f, st)


class OptimizerDiagnosisTests(unittest.TestCase):
    def setUp(self): self.d = data_for(TR)
    def test_traced_matches_fit_batch(self):
        th, traj, _ = optimize_traced(TR.skeleton(), self.d, TR.param_vector()); ref, _ = fit_batch(TR.skeleton(), self.d, torch.tensor([TR.param_vector()], dtype=DT), steps=100); self.assertEqual(float((th - ref[0]).abs().max()), 0.0)
    def test_trajectory_fields(self):
        _, traj, _ = optimize_traced(TR.skeleton(), self.d, TR.param_vector(), steps=20, accord=lambda p: 1.0)
        for f in ("step", "k", "loss", "surrogate_loglik", "profiled_loglik", "errors", "accord", "grad_norm", "distance", "max_shift"): self.assertIn(f, traj[0])
        self.assertEqual([t["step"] for t in traj], [0, 5, 10, 15, 20]); self.assertEqual(traj[0]["distance"], 0.0)
    def test_softness_schedule(self): _, traj, _ = optimize_traced(TR.skeleton(), self.d, TR.param_vector(), steps=20); self.assertEqual(traj[0]["k"], 4.0); self.assertEqual(traj[-1]["k"], 24.0)
    def test_best_so_far_never_worse_than_initial(self):
        _, traj, best = optimize_traced(TR.skeleton(), self.d, TR.param_vector(), steps=60); self.assertGreaterEqual(best["profiled"][0], traj[0]["profiled_loglik"] - 1e-9); self.assertGreaterEqual(best["surrogate"][0], traj[0]["surrogate_loglik"] - 1e-9)
    def test_best_so_far_never_worse_than_final(self):
        _, traj, best = optimize_traced(TR.skeleton(), self.d, TR.param_vector(), steps=60); self.assertGreaterEqual(best["profiled"][0], traj[-1]["profiled_loglik"] - 1e-9)
    def test_best_so_far_records_step(self): _, traj, best = optimize_traced(TR.skeleton(), self.d, TR.param_vector(), steps=30); self.assertIn(best["profiled"][2], [t["step"] for t in traj])
    def test_monotone_variant_is_monotone_on_its_criterion(self):
        _, traj, _ = optimize_traced(TR.skeleton(), self.d, [v + 0.3 for v in TR.param_vector()], steps=40, k_start=24.0, monotone=True, every=1); s = [t["loss"] for t in traj]            # objective at fixed stiffness 24: the criterion the variant claims to optimise
        self.assertTrue(all(b <= a + 1e-9 for a, b in zip(s, s[1:])))
    def test_monotone_never_below_initial_surrogate(self):
        th, traj, _ = optimize_traced(TR.skeleton(), self.d, TR.param_vector(), steps=40, k_start=24.0, monotone=True); self.assertLessEqual(traj[-1]["loss"], traj[0]["loss"] + 1e-9)
    def test_configs(self): self.assertEqual(set(CONFIGS), {"default", "k_start_12", "k_const_24", "lr_0.02", "lr_0.005", "k_const_24_lr_0.005", "k_reversed", "monotone_k24", "float32"}); self.assertEqual(CONFIGS["default"], {})
    def test_geometry_fields(self):
        g = local_geometry(TR.skeleton(), self.d, TR.param_vector()); self.assertEqual(set(g), {"k4", "k24"})
        for f in ("grad_norm", "condition", "min_eig", "negative_eigs", "near_zero_eigs", "cross_atom_coupling", "saturated_fraction", "n_params"): self.assertIn(f, g["k24"])
        self.assertTrue(0.0 <= g["k24"]["cross_atom_coupling"] <= 1.0); self.assertGreater(g["k24"]["saturated_fraction"], g["k4"]["saturated_fraction"])
    def test_optimizer_not_modified_by_diagnosis(self): self.assertIn("The optimizer is not modified", inspect.getsource(optimizer_diagnosis)); self.assertNotIn("deep_validated", inspect.getsource(DF))
    def test_diagnosis_row(self):
        r = optimizer_diagnosis(DX.down_task(0, 2, "standard"), 0); json.dumps(r); s = r["starts"]["true_parameters"]; self.assertEqual(set(r["starts"]), {"true_parameters", "top_candidate"}); self.assertEqual(s["noop"]["accord"], 1.0)
        self.assertEqual(set(s["configs"]), set(CONFIGS)); d = s["configs"]["default"]
        for f in ("trajectory", "corr_surrogate_accord", "first_drop_step", "early_stop", "matches_fit_batch", "best_profiled", "best_surrogate", "surrogate_improved", "profiled_improved", "accord_degraded"): self.assertIn(f, d)
        self.assertEqual(d["matches_fit_batch"], 0.0); self.assertGreaterEqual(d["best_profiled"]["profiled"], s["noop"]["profiled"] - 1e-9)
        for f in ("old", "new", "param_diff", "seconds_old", "seconds_new", "init_rounding"): self.assertIn(f, s["fitters"])
    def test_noop_baseline_present(self): self.assertIn('"noop": noop', inspect.getsource(optimizer_diagnosis))


class ExperimentTests(unittest.TestCase):
    def test_arms(self): self.assertEqual(DX.VO_ARMS, ("A", "B", "C")); self.assertEqual(len(DX.subunit_ids("deepvo", 0)), 24); self.assertEqual(len(DX.subunit_ids("deepopt", 0)), 8)
    def test_new_tasks_are_independent(self):
        old = {DX.down_task(k, d, r).world.uid for k in range(16) for d in DX.DOWN_DEPTHS for r in DX.GEN_REGIMES}; new = {DX.down_task(k, d, r).world.uid for k in range(100, 116) for d in DX.DOWN_DEPTHS for r in DX.GEN_REGIMES}
        self.assertEqual((len(old), len(new)), (64, 64)); self.assertEqual(len(old & new), 0)             # 128 tasks = 64 structures × 2 regimes; no structure shared between the two sets
    def test_regimes_share_structure_not_observations(self):
        a, b = DX.down_task(3, 3, "standard"), DX.down_task(3, 3, "rich"); self.assertEqual(a.world.uid, b.world.uid); self.assertNotEqual(len(a.y), len(b.y))
    def test_new_tasks_are_dev(self): t = DX.down_task(115, 5, "rich"); self.assertEqual(t.split, "dev"); self.assertLess(t.seed, DW.SPLIT_BASE["validation"])
    def test_task_set_label(self): self.assertIn('"new" if int(key) >= 100 else "diagnostic"', inspect.getsource(DX.vo_subunits))
    def test_arm_row(self):
        r = arm_run(DX.down_task(0, 2, "standard"), "B", 0); json.dumps(r)
        for f in ("arm", "accord", "in_region", "likelihood", "errors", "thresholds", "exact", "equivalent", "commits", "rollbacks", "transactions"): self.assertIn(f, r)
        self.assertEqual(set(r["thresholds"]), {"0.9", "0.95", "0.97", "0.99"})
    def test_arm_a_has_no_transactions(self): self.assertEqual(arm_run(DX.down_task(0, 2, "standard"), "A", 0)["transactions"], 0)
    def test_arms_differ_only_by_guard(self): src = inspect.getsource(arm_run); self.assertIn("validated=ARMS[arm]", src); self.assertEqual(src.count("DeepSearch("), 1)
    def test_true_atoms_only(self): self.assertIn("fact_literals=tl", inspect.getsource(arm_run)); self.assertNotIn("fact_discovery", inspect.getsource(DV))
    def test_no_sealed(self):
        for m in (DV, VR): self.assertNotIn("sealed_task", inspect.getsource(m)); self.assertNotIn("sealed_spec", inspect.getsource(m))
        self.assertEqual(DW.sealed_hash(), DW.SEALED_SHA256)
    def test_make_targets(self):
        mk = open("Makefile").read()
        for t in ("deep-run-arms:", "deep-run-optimizer:", "deep-vo-aggregate:", "deep-vo-report:"): self.assertIn(t, mk)
        self.assertIn("--keys 0-15,100-115", mk)
    def test_paired_stats(self):
        p = VR.paired([0.1, 0.2, 0.0, -0.1, 0.3]); self.assertEqual(p["n"], 5); self.assertAlmostEqual(p["mean"], 0.1); self.assertEqual((p["improved"], p["degraded"], p["unchanged"]), (0.6, 0.2, 0.2)); self.assertLess(p["ci95"][0], 0.1); self.assertGreater(p["effect_size"], 0)
    def test_paired_ci_uses_student(self): p = VR.paired([1.0, 1.0, 1.0]); self.assertEqual(p["ci95"], [1.0, 1.0]); self.assertIsNone(p["effect_size"]); self.assertEqual(VR._t(3), 3.182)
    def test_paired_empty(self): self.assertEqual(VR.paired([]), {})
    def test_criterion_fixed_in_code(self): self.assertIn('nb["diff_recovery_0.99"]["mean"] >= 0.20 and nb["diff_accord"]["mean"] >= 0.0', inspect.getsource(VR.aggregate))


if __name__ == "__main__":
    unittest.main()

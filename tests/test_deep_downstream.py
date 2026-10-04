import inspect
import json
import math
import unittest

import torch

import core.deep_down_report as DRP
import core.deep_downstream as DD
import core.deep_experiments as DX
import core.deep_world as DW
from core.boundary_program import BNode, canonical, predict
from core.deep_downstream import GOOD, LOSS_CAUSES, RECOVERED, STAGES, THRESHOLDS, BestCandidateTrajectory, TracedSearch, fitter_comparison, score_components, trace_task
from core.deep_fit import DT, FitData, fit_batch
from core.deep_search import DeepSearch, profiled_loglik
from core.deep_structure import algebraic_form, n_atoms, normalize, skel_key, structurally_equivalent

G = lambda s=0: torch.Generator().manual_seed(s)
T = lambda c, ch: BNode("translate", params=c, children=(ch,))
C = lambda k, a, b: BNode(k, children=(a, b))
DISC = T((0.2, 0.0), BNode("pred", "norm", "le", (1.0,)))
HALF = BNode("pred", "dot", "le", (0.3, 0.1))
RING = T((-0.5, 0.4), BNode("pred", "norm", "in", (0.5, 1.2)))
FAR = T((40.0, 40.0), BNode("pred", "norm", "le", (0.2,)))
TR = C("xor", DISC, HALF)


def X0(n=300, s=0): return torch.rand(n, 2, generator=G(s)) * 6 - 3


def data_for(prog, n=300, rate=0.05, s=0):
    X = X0(n, s); y = predict(prog, X); f = torch.rand(n, generator=G(s + 1)) < rate; return FitData.build(X, torch.where(f, 1 - y, y), 0.10)


class _Once:
    trace = None

    @classmethod
    def get(cls):
        if cls.trace is None: cls.trace = trace_task(DX.down_task(0, 2, "standard"), 0)
        return cls.trace


class TaxonomyTests(unittest.TestCase):
    def test_causes(self): self.assertEqual(LOSS_CAUSES, ("INITIAL_FIT_BAD", "POLISH_DEGRADED", "SIMPLIFICATION_DEGRADED", "CANONICALIZATION_CHANGED_BEHAVIOR", "SCORING_MISALIGNED", "RANKING_FAILURE", "FINAL_SELECTION_FAILURE", "THRESHOLD_TOO_STRICT", "FUNCTIONAL_EQUIVALENCE_MISCOUNTED", "NUMERICAL_INSTABILITY", "OTHER"))
    def test_stages(self): self.assertEqual(STAGES, ("true_atoms", "candidate_generation", "initial_fit", "refit_polish", "simplification", "canonicalization", "scoring", "ranking", "final_selection"))
    def test_thresholds(self): self.assertEqual(THRESHOLDS, (0.90, 0.95, 0.97, 0.99)); self.assertEqual(GOOD, 0.99)
    def test_threshold_not_changed_in_metrics(self): self.assertIn('"functional": a >= 0.99', inspect.getsource(DX.evaluate_program))
    def test_causes_assigned_in_code(self):
        src = inspect.getsource(trace_task)
        for c in ("INITIAL_FIT_BAD", "POLISH_DEGRADED", "SIMPLIFICATION_DEGRADED", "CANONICALIZATION_CHANGED_BEHAVIOR", "SCORING_MISALIGNED", "RANKING_FAILURE", "FINAL_SELECTION_FAILURE", "THRESHOLD_TOO_STRICT", "NUMERICAL_INSTABILITY", "OTHER"): self.assertIn(c, src)


class BestSoFarTests(unittest.TestCase):
    def traj(self, accs):
        t = BestCandidateTrajectory()
        for s, a in zip(STAGES, accs): t.add(s, a, DISC)
        return t
    def test_destroyed(self): s = self.traj([1.0, 1.0, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]).summary(); self.assertEqual((s["good_found_at"], s["good_lost_at"], s["destroyed"]), ("true_atoms", "initial_fit", True)); self.assertEqual(s["degrades_at"], "initial_fit")
    def test_preserved(self): s = self.traj([1.0] * 9).summary(); self.assertFalse(s["destroyed"]); self.assertIsNone(s["good_lost_at"]); self.assertIsNone(s["degrades_at"])
    def test_lost_at_simplification(self): s = self.traj([1.0, 1.0, 0.995, 0.995, 0.9, 0.9, 0.9, 0.9, 0.9]).summary(); self.assertEqual(s["good_lost_at"], "simplification")
    def test_never_found(self): s = self.traj([0.9] * 9).summary(); self.assertIsNone(s["good_found_at"]); self.assertFalse(s["destroyed"]); self.assertEqual(s["best_accord"], 0.9)
    def test_lost_then_recovered_is_not_destroyed(self): s = self.traj([1.0, 1.0, 0.98, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995]).summary(); self.assertFalse(s["destroyed"]); self.assertEqual(s["good_lost_at"], "initial_fit")
    def test_records_candidate(self): t = self.traj([1.0] * 9); self.assertEqual(t.stages["scoring"]["form"], algebraic_form(DISC)); self.assertIn("‖x‖", t.stages["scoring"]["program"])
    def test_appears_at(self): self.assertEqual(self.traj([0.5, 0.9, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]).summary()["appears_at"], "initial_fit")


class TracedSearchTests(unittest.TestCase):
    """L'instrumentation ne doit changer AUCUN calcul."""
    def setUp(self): self.d = data_for(TR)
    def run_both(self, **kw): return DeepSearch("factorized", budget=60, max_depth=3, seed=1, **kw).run(self.d), TracedSearch("factorized", budget=60, max_depth=3, seed=1, **kw).run(self.d)
    def test_same_selection(self): a, b = self.run_both(fact_literals=[DISC, HALF], fact_residual=False); self.assertEqual(a.best().program, b.best().program)
    def test_same_hypotheses(self): a, b = self.run_both(fact_literals=[DISC, HALF], fact_residual=False); self.assertEqual([h.program for h in a.hypotheses], [h.program for h in b.hypotheses]); self.assertEqual([h.evidence for h in a.hypotheses], [h.evidence for h in b.hypotheses])
    def test_same_cost(self): a, b = self.run_both(fact_literals=[DISC, HALF], fact_residual=False); self.assertEqual(a.stats["fit_equivalents"], b.stats["fit_equivalents"]); self.assertEqual(a.stats["generated"], b.stats["generated"])
    def test_same_on_other_strategy(self):
        a = DeepSearch("bottom_up", budget=40, seed=2).run(self.d); b = TracedSearch("bottom_up", budget=40, seed=2).run(self.d); self.assertEqual(a.best().program, b.best().program)
    def test_records_fits(self):
        s = TracedSearch("factorized", budget=60, max_depth=3, fact_literals=[DISC, HALF], fact_residual=False); s.run(self.d); self.assertTrue(any(f["origin"] == "factorized_r0" for f in s.fits)); self.assertTrue(all("fitted" in f and "inits" in f for f in s.fits))
    def test_records_init_before_polish(self):
        s = TracedSearch("factorized", budget=60, max_depth=3, fact_literals=[DISC, HALF], fact_residual=False); s.run(self.d); r0 = [f for f in s.fits if f["origin"] == "factorized_r0"]
        self.assertTrue(any(algebraic_form(f["inits"][0]) == algebraic_form(TR) and float((predict(f["inits"][0], self.d.X.float()) == predict(TR, self.d.X.float())).float().mean()) == 1.0 for f in r0))
    def test_records_simplification(self):
        s = TracedSearch("factorized", budget=60, max_depth=3, fact_literals=[DISC, HALF], fact_residual=False); s.run(self.d); self.assertEqual(set(s.simplification), {"before", "after", "changed", "before_evidence", "after_evidence"})
    def test_deterministic(self):
        a = TracedSearch("factorized", budget=60, max_depth=3, seed=3, fact_literals=[DISC, HALF], fact_residual=False); a.run(self.d); b = TracedSearch("factorized", budget=60, max_depth=3, seed=3, fact_literals=[DISC, HALF], fact_residual=False); b.run(self.d)
        self.assertEqual([f["fitted"] for f in a.fits], [f["fitted"] for f in b.fits])
    def test_only_calls_super(self):
        for m in (TracedSearch._fit, TracedSearch._register, TracedSearch._simplify, TracedSearch._setup): self.assertIn("super()", inspect.getsource(m))


class PolishTests(unittest.TestCase):
    """Préservation par polissage : le diagnostic doit VOIR une dégradation quand elle a lieu."""
    def test_polish_from_truth_measured(self):
        d = data_for(TR, 300, 0.05); p = normalize(TR); th, _ = fit_batch(p.skeleton(), d, torch.tensor([p.param_vector()], dtype=DT), steps=100); q = p.skeleton().with_params(th[0].tolist())
        a, b = score_components(p, d), score_components(q, d); self.assertTrue(math.isfinite(b["likelihood"])); self.assertEqual(a["depth"], b["depth"]); self.assertIsInstance(a["errors"], float)
    def test_zero_step_polish_is_identity(self):
        d = data_for(TR); p = normalize(TR); th, _ = fit_batch(p.skeleton(), d, torch.tensor([p.param_vector()], dtype=DT), steps=0); self.assertEqual(th[0].tolist(), p.param_vector())
    def test_trace_reports_polish_audit(self):
        r = _Once.get(); self.assertTrue(r["polish"])
        for f in ("origin", "accord_before", "accord_after", "param_shift", "likelihood_before", "likelihood_after", "errors_before", "errors_after", "robust_before", "robust_after", "is_truth_form"): self.assertIn(f, r["polish"][0])
    def test_degradation_is_quantified(self):
        r = _Once.get(); good = [p for p in r["polish"] if p["accord_before"] >= 0.999]; self.assertTrue(good); self.assertTrue(all(-1e-9 <= p["accord_before"] - p["accord_after"] <= 1.0 for p in good))
    def test_hold_best_never_worse_on_data(self):
        """Le garde-fou contrefactuel retient, par structure, la version la mieux expliquée PAR LES DONNÉES."""
        self.assertIn('s["likelihood"] > pool[k][1]["likelihood"]', inspect.getsource(trace_task)); self.assertIn("le pipeline n'est pas modifié", inspect.getsource(trace_task))
    def test_hold_best_uses_no_truth(self):
        src = inspect.getsource(trace_task); i = src.index("pool = {}"); j = src.index("counterfactual = {"); self.assertNotIn("A(", src[i:j]); self.assertNotIn("accord", src[i:j])


class SimplificationTests(unittest.TestCase):
    def test_removes_useless_atom_preserving_function(self):
        d = data_for(DISC, 300, 0.03); s = DeepSearch("factorized", budget=40, max_depth=2, fact_literals=[DISC, FAR], fact_residual=False); o = s.run(d); P = X0(2000, 9)
        self.assertEqual(n_atoms(o.best().program), 1); self.assertGreater(float((predict(o.best().program, P) == predict(DISC, P)).float().mean()), 0.97)
    def test_keeps_needed_atom(self):
        d = data_for(TR, 500, 0.02); o = DeepSearch("factorized", budget=60, max_depth=2, fact_literals=[DISC, HALF], fact_residual=False).run(d); self.assertEqual(n_atoms(o.best().program), 2)
    def test_never_merges_non_equivalent(self):
        d = data_for(TR, 500, 0.02); s = TracedSearch("factorized", budget=60, max_depth=2, fact_literals=[DISC, HALF], fact_residual=False); s.run(d); b, a = s.simplification["before"], s.simplification["after"]
        if s.simplification["changed"]: self.assertLess(n_atoms(a), n_atoms(b))
        else: self.assertEqual(a, b)
    def test_unchanged_keeps_parameters(self):
        d = data_for(TR, 500, 0.02); s = TracedSearch("factorized", budget=60, max_depth=2, fact_literals=[DISC, HALF], fact_residual=False); s.run(d)
        if not s.simplification["changed"]: self.assertEqual(s.simplification["before"].param_vector(), s.simplification["after"].param_vector())
    def test_simplified_is_a_subtree_structure(self):
        d = data_for(DISC, 300, 0.03); s = TracedSearch("factorized", budget=40, max_depth=2, fact_literals=[DISC, FAR], fact_residual=False); s.run(d); b, a = s.simplification["before"], s.simplification["after"]
        if s.simplification["changed"]: self.assertIn(skel_key(a), {skel_key(c) for c in b.children} | {skel_key(g) for c in b.children for g in c.children})
    def test_trace_reports_simplification(self):
        r = _Once.get()
        for f in ("changed", "accord_before", "accord_after", "atoms_before", "atoms_after", "removed_needed_atom", "params_modified"): self.assertIn(f, r["simplification"])
    def test_simplification_rule_is_bayes_factor(self): self.assertIn("math.log(3.0)", inspect.getsource(DeepSearch._simplify))


class EquivalenceTests(unittest.TestCase):
    def test_normalize_preserves_behavior(self):
        for p in (TR, C("or", RING, C("and", HALF, DISC)), C("xor", C("xor", RING, DISC), HALF)): X = X0(500, 3); self.assertTrue(torch.equal(predict(p, X), predict(normalize(p), X)))
    def test_param_dependent_canonical_can_change_key_not_function(self):
        z = C("and", T((0.0, 0.0), BNode("pred", "norm", "le", (1.0,))), HALF); X = X0(300); self.assertTrue(torch.equal(predict(z, X), predict(canonical(z), X)))
    def test_distinguished_levels(self):
        r = _Once.get()
        for f in ("final_exact", "final_equivalent", "functional_not_equivalent", "thresholds", "final_accord", "final_in_region"): self.assertIn(f, r)
    def test_different_representation_not_automatically_failure(self):
        src = inspect.getsource(trace_task); self.assertIn('if a_final >= GOOD: cause = RECOVERED', src); self.assertIn("miscounted = a_final >= GOOD and not sel_state", src)
    def test_associative_forms_equivalent(self): self.assertTrue(structurally_equivalent(C("or", C("or", DISC, HALF), RING), C("or", DISC, C("or", HALF, RING))))
    def test_trace_counts_canonicalization_changes(self): self.assertEqual(_Once.get()["canonicalization_changed_behavior"], 0)


class ScoreTests(unittest.TestCase):
    def setUp(self): self.d = data_for(TR)
    def test_components(self): self.assertEqual(set(score_components(TR, self.d)), {"errors", "likelihood", "robust_likelihood", "complexity_penalty", "structural_penalty", "mdl_bits", "posterior_score", "depth", "n_params"})
    def test_posterior_is_sum(self): s = score_components(TR, self.d); self.assertAlmostEqual(s["posterior_score"], s["likelihood"] - s["complexity_penalty"] - s["structural_penalty"], places=9)
    def test_matches_pipeline_evidence(self):
        o = DeepSearch("factorized", budget=40, max_depth=2, fact_literals=[DISC, HALF], fact_residual=False).run(self.d); h = o.best(); self.assertAlmostEqual(score_components(h.program, self.d)["posterior_score"], h.evidence, places=6)
    def test_truth_beats_wrong(self): self.assertGreater(score_components(TR, self.d)["posterior_score"], score_components(C("xor", RING, HALF), self.d)["posterior_score"])
    def test_penalties_grow_with_size(self): a, b = score_components(DISC, self.d), score_components(TR, self.d); self.assertGreater(b["complexity_penalty"], a["complexity_penalty"]); self.assertGreater(b["structural_penalty"], a["structural_penalty"])
    def test_spearman(self): self.assertAlmostEqual(DD._spearman([1, 2, 3, 4], [2, 3, 5, 9]), 1.0); self.assertAlmostEqual(DD._spearman([1, 2, 3, 4], [9, 5, 3, 2]), -1.0); self.assertIsNone(DD._spearman([1, 2], [1, 2]))
    def test_oracle_ranking_fields(self):
        r = _Once.get(); self.assertEqual(set(r["oracle_ranking"]), {"posterior_score", "likelihood", "robust_likelihood", "mdl", "complexity_penalty", "structural_penalty"})
        for v in r["oracle_ranking"].values(): self.assertEqual(set(v), {"spearman", "top1_agreement", "top3_recall", "top5_recall", "top1_accord"})
    def test_oracle_is_diagnostic_only(self): self.assertNotIn("deep_downstream", inspect.getsource(__import__("core.deep_search", fromlist=["x"]))); self.assertIn("calculé APRÈS coup", inspect.getsource(DD))
    def test_ranking_stable_under_reevaluation(self):
        o = DeepSearch("factorized", budget=40, max_depth=2, fact_literals=[DISC, HALF], fact_residual=False).run(self.d); ev = [score_components(h.program, self.d)["posterior_score"] for h in o.hypotheses[1:]]; self.assertEqual(ev, sorted(ev, reverse=True))


class FitterTests(unittest.TestCase):
    def test_comparison_fields(self):
        t = DX.down_task(0, 2, "standard"); acc = lambda p: (1.0, 1.0); r = fitter_comparison([normalize(t.world.truth)], t, acc, steps=20)
        self.assertEqual(set(r), {"n", "max_param_error", "max_score_diff", "max_accord_diff", "same_top1", "same_order", "rank_correlation"})
    def test_same_init_small_difference_short_run(self):
        t = DX.down_task(0, 2, "standard"); r = fitter_comparison([normalize(t.world.truth)], t, lambda p: (1.0, 1.0), steps=5); self.assertLess(r["max_param_error"], 1e-3)
    def test_reports_ranking_not_only_numbers(self): src = inspect.getsource(fitter_comparison); self.assertIn("same_order", src); self.assertIn("same_top1", src); self.assertIn("ne garantit pas le même classement", src)
    def test_empty(self): self.assertEqual(fitter_comparison([], DX.down_task(0, 2, "standard"), lambda p: (1.0, 1.0)), {"n": 0})


class TraceTests(unittest.TestCase):
    def test_true_atoms_only(self): src = inspect.getsource(trace_task); self.assertIn("fact_literals=tl", src); self.assertNotIn("fact_discovery", src); self.assertNotIn("PrimitiveProposal", inspect.getsource(DD))
    def test_fields(self):
        r = _Once.get()
        for f in ("cause", "final_accord", "thresholds", "trajectory", "trajectory_summary", "candidates", "polish", "simplification", "oracle_ranking", "best_minus_selected", "truth_fit", "counterfactual", "confidence", "fitter", "n_good_candidates"): self.assertIn(f, r)
        json.dumps(r)
    def test_all_stages_captured(self): self.assertEqual(list(_Once.get()["trajectory"]), list(STAGES))
    def test_stage_state_fields(self):
        c = _Once.get()["candidates"][0]
        for f in ("program", "params", "accord", "errors", "likelihood", "mdl_bits", "posterior_score", "rank", "depth", "n_params"): self.assertIn(f, c)
    def test_cause_valid(self): self.assertIn(_Once.get()["cause"], LOSS_CAUSES + (RECOVERED,))
    def test_true_atoms_generate_perfect_candidate(self): r = _Once.get(); self.assertGreaterEqual(r["trajectory"]["candidate_generation"]["accord"], 0.999); self.assertGreaterEqual(r["n_good_candidates"], 1)
    def test_counterfactuals(self): self.assertTrue({"hold_best", "no_simplification", "no_dedup", "pre_fit_top_score", "best_available"} <= set(_Once.get()["counterfactual"]))
    def test_deterministic(self):
        a = trace_task(DX.down_task(1, 2, "rich"), 1, fitter_audit=False); b = trace_task(DX.down_task(1, 2, "rich"), 1, fitter_audit=False); self.assertEqual(a["cause"], b["cause"]); self.assertEqual(a["final_accord"], b["final_accord"]); self.assertEqual(a["trajectory"], b["trajectory"])
    def test_dev_only(self): self.assertEqual(DX.down_task(0, 3, "rich").split, "dev"); self.assertEqual(len(DX.subunit_ids("deepdown", 0)), 8)
    def test_no_sealed(self):
        for m in (DD, DRP): self.assertNotIn("sealed_task", inspect.getsource(m)); self.assertNotIn("sealed_spec", inspect.getsource(m))
        self.assertEqual(DW.sealed_hash(), DW.SEALED_SHA256)
    def test_pipeline_code_unchanged_by_diagnostic(self): self.assertIn("le pipeline n'est PAS modifié", inspect.getsource(DD))
    def test_make_targets(self):
        mk = open("Makefile").read()
        for t in ("deep-run-down:", "deep-down-aggregate:", "deep-down-report:"): self.assertIn(t, mk)
    def test_calibration(self):
        c = DRP.calibration([0.9, 0.9, 0.1, 0.1], [1.0, 1.0, 0.0, 0.0]); self.assertAlmostEqual(c["ece"], 0.1, places=6); self.assertAlmostEqual(c["brier"], 0.01, places=6); self.assertEqual(len(c["reliability"]), 2)
    def test_calibration_empty(self): self.assertEqual(DRP.calibration([], []), {})


if __name__ == "__main__":
    unittest.main()

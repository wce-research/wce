import inspect
import math
import unittest

import numpy as np
import torch

import core.deep_e2e as EE
import core.deep_e2e_exp as EX
import core.deep_ee_report as ER
import core.deep_experiments as DE
import core.deep_search as DS
from core.boundary_program import BNode, predict
from core.deep_compose import to_program, tree_size
from core.deep_e2e import (DEFAULTS, FITS, REDUCTIONS, EndToEnd, PrimitiveHypothesis, WideSpace, compose, decide, finalize, floor_errors, hierarchical, hypotheses_from, knn_errors, load_config, reduce, run_chain, wide_compose, with_complements)
from core.deep_primitives2 import disc_atom, line_atom
from core.deep_structure import KNOWN, NONE_OF_THE_ABOVE, UNKNOWN

G = lambda s=0: torch.Generator().manual_seed(s)
A0, A1, A2, A3 = disc_atom(-1.0, 0.5, 1.2), line_atom(0.3, 0.1), disc_atom(0.6, -0.8, 1.0), line_atom(2.0, 0.4)
FAKES = [disc_atom(2.2, 2.2, 0.5), line_atom(1.1, -2.0), disc_atom(-2.3, -2.2, 0.6), line_atom(-0.7, 1.9), disc_atom(1.9, -2.1, 0.5), line_atom(2.6, 2.1), disc_atom(-0.1, 2.4, 0.5)]
TRUTH = BNode("xor", children=(BNode("and", children=(A0, A1)), A2))
NOT = lambda a: BNode("not", children=(a,))


def data(prog=TRUTH, n=500, rate=0.0, s=0):
    X = torch.rand(n, 2, generator=G(s)) * 6 - 3; y = predict(prog, X); f = torch.rand(n, generator=G(s + 1)) < rate; return X, torch.where(f, 1 - y, y)


def acc(prog, ref=TRUTH): P = torch.rand(3000, 2, generator=G(9)) * 6 - 3; return float((predict(prog, P) == predict(ref, P)).float().mean())
D = lambda atoms, src="t": [(a, float(len(atoms) - i), src) for i, a in enumerate(atoms)]


class HypothesisTests(unittest.TestCase):
    def setUp(self): self.X, self.y = data()
    def test_fields(self):
        H = hypotheses_from(D([A0, A1]), self.X); self.assertEqual(len(H), 2); h = H[0]; self.assertIsInstance(h, PrimitiveHypothesis)
        for k in ("program", "type", "params", "confidence", "uncertainty", "provenance", "alternatives", "signature"): self.assertTrue(hasattr(h, k), k)
        self.assertEqual(h.type, "norm.le"); self.assertEqual(h.confidence, 1.0); self.assertEqual(h.provenance, "t"); self.assertIsNone(h.uncertainty)
    def test_equivalent_primitives_become_alternatives(self):
        near = disc_atom(-1.0, 0.5, 1.201); H = hypotheses_from(D([A0, near, A1]), self.X); self.assertEqual(len(H), 2); self.assertEqual(H[0].n_alternatives, 1); self.assertIsNotNone(H[0].uncertainty); self.assertLess(H[0].uncertainty, 0.03)
    def test_complement_is_an_alternative_not_a_new_atom(self): H = hypotheses_from(D([A0, NOT(A0)]), self.X); self.assertEqual(len(H), 1); self.assertEqual(H[0].n_alternatives, 1)
    def test_best_scored_represents(self): H = hypotheses_from([(disc_atom(-1.0, 0.5, 1.201), 1.0, "low"), (A0, 5.0, "high")], self.X); self.assertEqual(H[0].provenance, "high"); self.assertEqual(repr(H[0].program), repr(A0))
    def test_confidence_normalised(self): H = hypotheses_from([(A0, 4.0, "a"), (A1, 1.0, "b"), (A2, -3.0, "c")], self.X); self.assertEqual([h.confidence for h in H], [1.0, 0.25, 0.0]); self.assertEqual(hypotheses_from([], self.X), [])


class ReductionTests(unittest.TestCase):
    def setUp(self): self.X, self.y = data(rate=0.03); self.H = hypotheses_from(D(FAKES[:4] + [A0, A1, A2] + FAKES[4:]), self.X)
    def test_floor_errors(self):
        yb = self.y.numpy().astype(bool); sig = lambda a: predict(a, self.X).numpy().astype(bool); self.assertEqual(floor_errors([], yb), float(min(yb.sum(), len(yb) - yb.sum()))); f3 = floor_errors([sig(A0), sig(A1), sig(A2)], yb)
        self.assertLessEqual(f3, floor_errors([sig(A0), sig(A1)], yb)); self.assertLessEqual(f3, 0.06 * len(yb)); self.assertEqual(floor_errors([sig(A0)], yb), floor_errors([~sig(A0)], yb))
    def test_greedy_floor_finds_true_atoms_among_false(self):
        kept, info = reduce(self.H, self.y, "greedy_floor", min_gain=4.0); progs = {repr(h.program) for h in kept}; self.assertTrue({repr(A0), repr(A1), repr(A2)} <= progs); self.assertLessEqual(len(kept), 6); self.assertEqual(info["floor_trace"], sorted(info["floor_trace"], reverse=True))
    def test_confidence_keeps_the_false_ones_first(self): kept, _ = reduce(self.H, self.y, "confidence", max_atoms=4); self.assertEqual({repr(h.program) for h in kept}, {repr(a) for a in FAKES[:4]})
    def test_all_methods_bounded(self):
        for m in REDUCTIONS: kept, info = reduce(self.H, self.y, m); self.assertLessEqual(len(kept), 6, m); self.assertEqual(info["kept"], len(kept)); self.assertEqual(len({id(h) for h in kept}), len(kept))
        with self.assertRaises(KeyError): reduce(self.H, self.y, "nope")
    def test_family_cap(self): kept, _ = reduce(self.H, self.y, "family", per_family=1); self.assertEqual(len({h.type for h in kept}), len(kept))
    def test_dominance_drops_close_duplicates(self):
        H = hypotheses_from(D([A0, disc_atom(-1.0, 0.55, 1.25), A1]), self.X, same=0.999); self.assertEqual(len(H), 3); kept, _ = reduce(H, self.y, "dominance", dominance=0.9); self.assertEqual(len(kept), 2)
    def test_min_gain_stops_selection(self): a, _ = reduce(self.H, self.y, "greedy_floor", min_gain=1.0); b, _ = reduce(self.H, self.y, "greedy_floor", min_gain=40.0); self.assertGreaterEqual(len(a), len(b)); self.assertEqual(reduce([], self.y)[0], [])


class CompositionTests(unittest.TestCase):
    def setUp(self): self.X, self.y = data(rate=0.03)
    def test_true_atoms(self): R = compose([A0, A1, A2], self.X, self.y); self.assertGreater(acc(to_program(R.selected.tree, R.units)), 0.99); self.assertEqual(len(R.units), 6); self.assertTrue(hasattr(R, "table"))
    def test_wrong_polarity_is_recovered_through_complement(self):
        X, y = data(NOT(TRUTH)); R = compose([A0, A1, A2], X, y); self.assertGreater(acc(to_program(R.selected.tree, R.units), NOT(TRUTH)), 0.99); X2, y2 = data(NOT(A0)); R2 = compose([A0], X2, y2); self.assertEqual(float(R2.selected.errors), 0.0)
        self.assertGreater(float(compose([A0], X2, y2, complements=False).selected.errors), 0.5 * len(X2) * 0.3)
    def test_false_atoms_are_left_out(self):
        R = compose([A0, A1, A2, FAKES[0], FAKES[1]], self.X, self.y, "bic"); used = {repr(R.units[i]) for i in __import__("core.deep_compose", fromlist=["x"]).tree_atoms(R.selected.tree)}; self.assertFalse(used & {repr(FAKES[0]), repr(FAKES[1])}); self.assertGreater(acc(to_program(R.selected.tree, R.units)), 0.99)
    def test_missing_atom_gives_partial_model(self): R = compose([A0, A1], self.X, self.y); a = acc(to_program(R.selected.tree, R.units)); self.assertLess(a, 0.97); self.assertGreater(a, 0.6)
    def test_too_many_cells_rejected(self):
        many = [line_atom(0.3 * i, 0.1 * i - 0.5) for i in range(8)]
        with self.assertRaises(ValueError): compose(many, self.X, self.y)
    def test_with_complements(self): u = with_complements([A0, A1]); self.assertEqual(len(u), 4); self.assertEqual(u[2].kind, "not"); self.assertTrue(torch.equal(predict(u[2], self.X), 1 - predict(A0, self.X)))


class WideTests(unittest.TestCase):
    def setUp(self): self.X, self.y = data(rate=0.03); self.atoms = FAKES[:4] + [A0, A1, A2] + FAKES[4:]
    def test_space(self):
        sp = WideSpace(self.atoms, self.X, self.y); self.assertEqual(sp.rows.shape, (10, sp.K)); self.assertLessEqual(sp.K, len(self.X)); self.assertEqual(float(sp.count.sum()), len(self.X)); t = ("xor", ("and", 4, 5), 6); self.assertEqual(float(sp.errors(sp.row_of(t)[None])[0]), float((predict(to_program(t, self.atoms), self.X) != self.y).sum()))
        with self.assertRaises(ValueError): WideSpace([], self.X, self.y)
    def test_ten_atoms_without_enumeration(self):
        R = wide_compose(self.atoms, self.X, self.y, max_size=4, beam=200); self.assertGreater(acc(to_program(R.selected.tree, self.atoms)), 0.99); self.assertLess(R.stats["considered"], 2_000_000); self.assertEqual(R.stats["atoms"], 10); self.assertLessEqual(tree_size(R.selected.tree), 4)
    def test_beam_bounds_classes(self): a = wide_compose(self.atoms, self.X, self.y, max_size=4, beam=20); b = wide_compose(self.atoms, self.X, self.y, max_size=4, beam=200); self.assertLess(a.stats["classes"], b.stats["classes"]); self.assertLessEqual(a.stats["classes"], 10 + 3 * 20)
    def test_budget_enforced(self): R = wide_compose(self.atoms, self.X, self.y, max_size=5, beam=200, budget=500); self.assertLessEqual(R.stats["considered"], 500); self.assertTrue(R.stats["truncated"]); self.assertIsNotNone(R.selected)
    def test_deterministic(self):
        f = lambda: (wide_compose(self.atoms, self.X, self.y, max_size=4).selected.tree, wide_compose(self.atoms, self.X, self.y, max_size=4).stats["classes"]); self.assertEqual(f(), f())
    def test_hierarchical_extends_beyond_six(self):
        order = [A0, A1, FAKES[0], FAKES[1], FAKES[2], FAKES[3], A2, FAKES[4]]; R = hierarchical(order, self.X, self.y); self.assertEqual(R.stats["levels"], 2); self.assertGreater(acc(to_program(R.selected.tree, order)), 0.99)
        self.assertEqual(hierarchical([A0, A1, A2], self.X, self.y).stats["levels"], 1)
    def test_seeds_are_reused(self): R = wide_compose(self.atoms, self.X, self.y, max_size=2, beam=50, seeds=[(("xor", ("and", 4, 5), 6), 3)]); self.assertGreater(acc(to_program(R.selected.tree, self.atoms)), 0.99)


class FinalizeTests(unittest.TestCase):
    def setUp(self): self.X, self.y = data(rate=0.03); self.off = [disc_atom(-0.9, 0.6, 1.1), line_atom(0.35, 0.15), disc_atom(0.7, -0.75, 0.95)]; self.tree = ("xor", ("and", 0, 1), 2)
    def test_none_keeps_parameters(self): F = finalize(self.tree, self.off, self.X, self.y, "none"); self.assertEqual(F.program.param_vector(), to_program(self.tree, self.off).param_vector()); self.assertFalse(F.committed); self.assertEqual(F.errors_before, F.errors_after)
    def test_validated_never_worse_on_observations(self):
        for atoms in (self.off, [A0, A1, A2]):
            F = finalize(self.tree, atoms, self.X, self.y, "validated"); self.assertLessEqual(F.errors_after, F.errors_before)
            if not F.committed: self.assertEqual(F.program.param_vector(), to_program(self.tree, atoms).param_vector())
    def test_initial_state_is_conserved(self): before = [a.param_vector() for a in self.off]; finalize(self.tree, self.off, self.X, self.y, "forced"); finalize(self.tree, self.off, self.X, self.y, "validated"); self.assertEqual(before, [a.param_vector() for a in self.off])
    def test_forced_always_replaces(self): F = finalize(self.tree, self.off, self.X, self.y, "forced"); self.assertTrue(F.committed); self.assertNotEqual(F.program.param_vector(), to_program(self.tree, self.off).param_vector())
    def test_structure_unchanged(self):
        for f in FITS: self.assertEqual(repr(finalize(self.tree, self.off, self.X, self.y, f).program.skeleton()), repr(to_program(self.tree, self.off).skeleton()))
        with self.assertRaises(KeyError): finalize(self.tree, self.off, self.X, self.y, "nope")


class DecisionTests(unittest.TestCase):
    def setUp(self): self.X, self.y = data(rate=0.03)
    def test_known_with_true_atoms(self): e = run_chain(self.X, self.y, 0, {"fit": "none"}, D([A0, A1, A2])); self.assertEqual(e.label, KNOWN); self.assertLess(e.decision["gap"], 0.04); self.assertGreater(acc(e.program), 0.99)
    def test_unknown_or_none_when_atoms_missing(self): e = run_chain(self.X, self.y, 0, {"fit": "none"}, D([A1])); self.assertIn(e.label, (UNKNOWN, NONE_OF_THE_ABOVE)); self.assertGreater(e.decision["gap"], 0.04)
    def test_unknown_without_any_primitive(self): e = run_chain(self.X, self.y, 0, None, []); self.assertIsNone(e.program); self.assertEqual(e.label, UNKNOWN); self.assertEqual(e.kept, [])
    def test_thresholds_order_labels(self):
        R = compose([A1], self.X, self.y); n = len(self.X); kn = knn_errors(self.X, self.y); self.assertEqual(decide(R, kn, self.X, self.y)["label"], KNOWN); self.assertEqual(decide(R, kn + 0.06 * n, self.X, self.y)["label"], UNKNOWN); self.assertEqual(decide(R, kn + 0.2 * n, self.X, self.y)["label"], NONE_OF_THE_ABOVE)
        self.assertTrue(decide(R, kn + 0.06 * n, self.X, self.y)["partial_model"])
    def test_knn_errors_bounded(self): k = knn_errors(self.X, self.y); self.assertGreaterEqual(k, 0); self.assertLess(k, 0.2 * len(self.X))


class ChainTests(unittest.TestCase):
    def setUp(self): self.X, self.y = data(rate=0.03)
    def test_result_fields(self):
        e = run_chain(self.X, self.y, 0, None, D(FAKES[:3] + [A0, A1, A2])); self.assertIsInstance(e, EndToEnd); self.assertGreater(acc(e.program), 0.98)
        for k in ("literals", "hypotheses", "kept", "classes", "considered", "seconds", "reduction"): self.assertIn(k, e.stats)
        self.assertLessEqual(e.stats["kept"], 6)
    def test_no_oracle_in_system_module(self):
        s = inspect.getsource(EE)
        for w in ("deep_world", "truth", "leaves(", "_exp", "deep_primitive_eval"): self.assertNotIn(w, s.replace("vérité", ""))
        self.assertIn("flat_types", inspect.getsource(EE.discover))
    def test_real_discovery_runs(self): X, y = data(BNode("xor", children=(A0, A1)), n=400, rate=0.03); e = run_chain(X, y); self.assertIsNotNone(e.program); self.assertGreater(e.stats["literals"], 3); self.assertIn(e.label, (KNOWN, UNKNOWN, NONE_OF_THE_ABOVE))
    def test_deterministic(self):
        f = lambda: run_chain(self.X, self.y, 0, None, D(FAKES[:3] + [A0, A1, A2])); a, b = f(), f(); self.assertEqual(repr(a.program.skeleton()), repr(b.program.skeleton())); self.assertTrue(all(abs(u - v) < 1e-9 for u, v in zip(a.program.param_vector(), b.program.param_vector())))
    def test_config(self): c = load_config(); self.assertLessEqual(set(DEFAULTS), set(c)); self.assertIn(c["fit"], FITS); self.assertIn(c["reduction"], REDUCTIONS)


class BenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.task = EX.task_at(999, 2, "standard"); cls.row = EX.chain_c(cls.task, "C")
    def test_tasks_new_and_dev(self):
        from core.deep_experiments import down_task
        used = {down_task(k, 3, "rich").seed for r in ((0, 16), (100, 116), (200, 216), (300, 308), (400, 416), (500, 516), (600, 616), (700, 732), (800, 832)) for k in range(*r)}; new = {EX.task_at(k, d, "rich").seed for k in range(900, 932) for d in EX.DEPTHS}
        self.assertEqual(len(new), 192); self.assertFalse(new & used); self.assertEqual(self.task.split, "dev"); self.assertFalse(new & {EX.task_at(k, d, "rich").seed for k in range(990, 1000) for d in EX.DEPTHS})
    def test_primary_chains_have_no_oracle(self):
        for f in (EX.chain_ab,): s = inspect.getsource(f); self.assertNotIn("world.truth", s); self.assertIn("max_depth=6", s)
        s = inspect.getsource(EX.chain_c); self.assertIn("run_chain(task.X, task.y, seed, cfg, discovered)", s); self.assertEqual(EX.PRIMARY, ("A", "B", "C")); self.assertEqual(EX.C_VARIANTS["C"], {})
    def test_chain_c_row(self):
        for k in ("accord", "accord_out", "functional", "functional_097", "exact", "class_equal", "obs_agreement", "prefit_accord", "literals", "hypotheses", "kept", "atom_recall_literals", "atom_recall_kept", "false_kept", "false_used", "generated_099", "top5_099", "top20_099", "candidate_097", "label", "gap", "cause", "seconds"): self.assertIn(k, self.row, k)
        self.assertIn(self.row["cause"], EX.CAUSES); self.assertTrue(self.row["generated_099"] or not self.row["top20_099"])
    def test_attribution_order(self):
        base = {"functional": False, "atom_recall_literals": 1.0, "atom_recall_kept": 1.0, "generated_099": True, "prefit_functional": False, "false_used": 0, "selected_errors": 10.0, "best_good_errors": 10.0, "unseen_mass": 0.0}; A = lambda **k: EX._attribute(None, None, None, {**base, **k})
        self.assertEqual(A(functional=True), "recovered"); self.assertEqual(A(atom_recall_literals=0.5), "primitive_absent"); self.assertEqual(A(atom_recall_kept=0.5), "primitive_eliminated"); self.assertEqual(A(generated_099=False), "class_not_generated"); self.assertEqual(A(generated_099=False, unseen_mass=0.2), "non_identifiable")
        self.assertEqual(A(prefit_functional=True), "fit_degraded"); self.assertEqual(A(false_used=1, selected_errors=8.0), "too_many_false"); self.assertEqual(A(), "class_misranked")
    def test_robust_conditions(self):
        from core.deep_reachability import leaves
        t = EX.task_at(999, 3, "standard"); n = len(leaves(t.world.truth)); self.assertEqual(len(EX.robust_atoms(t, "true")), n); self.assertEqual(len(EX.robust_atoms(t, "true_plus3")), n + 3); self.assertEqual(len(EX.robust_atoms(t, "missing1")), n - 1); self.assertEqual(len(EX.robust_atoms(t, "false_only")), n)
        a = EX.robust_atoms(t, "approx_010"); self.assertEqual([repr(x.skeleton()) for x in a], [repr(x.skeleton()) for x in leaves(t.world.truth)]); self.assertNotEqual(a[0].param_vector(), leaves(t.world.truth)[0].param_vector()); self.assertEqual(EX.robust_atoms(t, "true_plus1", 3)[-1].param_vector(), EX.robust_atoms(t, "true_plus1", 3)[-1].param_vector())
        r = EX.robust_row(t, "true"); self.assertGreater(r["accord"], 0.97); self.assertEqual(r["arm"], "true")
    def test_sources_and_reduce_rows(self):
        r = EX.source_row(self.task, "true_atoms"); self.assertEqual(r["experiment"], "source"); self.assertEqual(r["atom_recall_literals"], 1.0); rows = EX.reduce_rows(self.task); self.assertTrue({"greedy_floor", "confidence", "wide_all", "hier_all"} <= {x["arm"] for x in rows})
    def test_boundary_row(self): r = EX.boundary_row(EX.ooc_task(999, "primitive_absent"), "primitive_absent"); self.assertIn(r["label"], (KNOWN, UNKNOWN, NONE_OF_THE_ABOVE)); self.assertIsNone(EX.ooc_task(999, "primitive_absent").world.truth)
    def test_subunits(self): self.assertIn("deepee", DE.SUBUNITS); a = DE.subunit_ids("deepee", "900"); self.assertEqual(len(a), 12 * 6 + 3); self.assertEqual(len(set(a)), len(a))
    def test_sealed_and_pipeline_untouched(self):
        for m in (EE, EX, ER): s = inspect.getsource(m); self.assertNotIn("sealed_task", s); self.assertNotIn('"sealed"', s); self.assertNotIn('"test"', s)
        s = inspect.getsource(DS); self.assertNotIn("deep_e2e", s); self.assertNotIn("deep_compose", s); self.assertIsNone(inspect.signature(DS.DeepSearch.__init__).parameters["validated"].default)
    def test_summarize(self):
        tag = {"task_id": "999_2", "depth": 2, "regime": "standard"}; rows = [{**tag, **self.row}, {**tag, **EX.chain_c(self.task, "C_nofit")}, {**tag, **EX.source_row(self.task, "true_atoms")}, {**tag, **EX.robust_row(self.task, "true")}] + [{**tag, **x} for x in EX.reduce_rows(self.task)]
        rows += [{**tag, "experiment": "chain", "arm": a, "accord": 0.9, "functional": False, "functional_097": False, "candidate_097": False, "accord_out": 0.9} for a in ("A", "B")] + [{"task_id": "999_x", "depth": 0, "regime": "standard", **EX.boundary_row(EX.ooc_task(999, "primitive_absent"), "primitive_absent")}]
        S = ER.summarize(rows); self.assertIn("all/C", S["chain"]); self.assertIn("all: C − A", S["chain_paired"]); self.assertIn("all: C − B", S["chain_paired"]); self.assertIn("all", S["causes"]); self.assertIn("primitive_absent", S["boundary"]); ER.show(S)


if __name__ == "__main__":
    unittest.main()

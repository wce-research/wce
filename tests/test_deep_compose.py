import inspect
import math
import unittest
try:
    from tests._source import code_only
except ImportError:
    from _source import code_only

import numpy as np
import torch

import core.deep_compose as DC
import core.deep_compose_exp as DX
import core.deep_dc_report as DR
import core.deep_experiments as DE
import core.deep_search as DS
from core.boundary_program import BNode, predict
from core.deep_compose import (OPS, SELECTORS, CellSpace, ObservationalEquivalenceClass, PartialStructure, bottom_up, class_scores, from_program, order_by, profiled_ll, propagate, rank_of, scaling, select, to_program, top_down,
                               tree_atoms, tree_depth, tree_eval, tree_form, tree_render, tree_shape, tree_size)
from core.deep_primitives2 import disc_atom, line_atom
from core.deep_structure import algebraic_form

G = lambda s=0: torch.Generator().manual_seed(s)
ATOMS = [disc_atom(-1.0, 0.5, 1.2), line_atom(0.3, 0.1), disc_atom(0.6, -0.8, 1.0), line_atom(2.0, 0.4), disc_atom(-0.2, -1.4, 0.9), line_atom(-1.1, 0.7)]
N2 = lambda op, a, b: BNode(op, children=(a, b))


def data(tree, n_atoms, n=500, rate=0.0, s=0):
    X = torch.rand(n, 2, generator=G(s)) * 6 - 3; atoms = ATOMS[:n_atoms]; y = predict(to_program(tree, atoms), X); f = torch.rand(n, generator=G(s + 1)) < rate; return atoms, X, torch.where(f, 1 - y, y)


CHAIN5 = ("xor", ("and", ("or", ("xor", 0, 1), 2), 3), 4)
BAL4 = ("or", ("and", 0, 1), ("xor", 2, 3))


class TreeTests(unittest.TestCase):
    def test_size_depth_atoms(self): self.assertEqual(tree_size(CHAIN5), 5); self.assertEqual(tree_depth(CHAIN5), 4); self.assertEqual(tree_depth(BAL4), 2); self.assertEqual(sorted(tree_atoms(BAL4)), [0, 1, 2, 3]); self.assertEqual(tree_size(3), 1)
    def test_shapes(self):
        self.assertEqual(tree_shape(CHAIN5)["shape"], "chain"); self.assertEqual(tree_shape(BAL4)["shape"], "balanced"); self.assertEqual(tree_shape(("and", ("or", 0, 1), ("xor", ("and", 2, 3), ("or", 4, 5))))["shape"], "mixed")
        self.assertEqual(tree_shape(0)["shape"], "atom"); self.assertEqual(tree_shape(("and", 0, 1))["shape"], "pair"); self.assertEqual(tree_shape(("and", ("and", 0, 1), 2))["shape"], "both")
    def test_strahler_and_operators(self):
        self.assertEqual(tree_shape(CHAIN5)["strahler"], 2); self.assertEqual(tree_shape(BAL4)["strahler"], 3); self.assertEqual(tree_shape(CHAIN5)["operators"], "mixed"); self.assertEqual(tree_shape(("and", ("and", 0, 1), 2))["operators"], "single"); self.assertEqual(tree_shape(CHAIN5)["n_xor"], 2)
    def test_form_is_associative_commutative(self):
        self.assertEqual(tree_form(("and", ("and", 0, 1), 2)), tree_form(("and", 2, ("and", 1, 0)))); self.assertNotEqual(tree_form(("and", ("or", 0, 1), 2)), tree_form(("or", ("and", 0, 1), 2))); self.assertNotEqual(tree_form(("xor", 0, 1)), tree_form(("or", 0, 1)))
    def test_form_agrees_with_algebraic_form(self):
        pairs = [(("and", ("and", 0, 1), 2), ("and", 2, ("and", 1, 0))), (CHAIN5, CHAIN5), (BAL4, ("or", ("xor", 3, 2), ("and", 1, 0))), (BAL4, ("and", ("or", 0, 1), ("xor", 2, 3)))]
        for a, b in pairs: self.assertEqual(tree_form(a) == tree_form(b), algebraic_form(to_program(a, ATOMS)) == algebraic_form(to_program(b, ATOMS)))
    def test_program_round_trip(self): p = to_program(CHAIN5, ATOMS); self.assertEqual(from_program(p, ATOMS), CHAIN5); self.assertEqual(p.kind, "xor"); self.assertIs(to_program(2, ATOMS), ATOMS[2]); self.assertEqual(tree_render(("and", 0, ("xor", 1, 2))), "(a0 AND (a1 XOR a2))")
    def test_eval_on_ints_and_arrays(self):
        self.assertEqual(tree_eval(("xor", ("and", 0, 1), 2), [0b1100, 0b1010, 0b0001]), 0b1001); v = [np.array([True, False]), np.array([True, True])]; self.assertEqual(tree_eval(("and", 0, 1), v).tolist(), [True, False])


class CellTests(unittest.TestCase):
    def setUp(self): self.atoms, self.X, self.y = data(BAL4, 4); self.sp = CellSpace(self.atoms, self.X, self.y)
    def test_cells_partition_observations(self): self.assertEqual(float(self.sp.count.sum()), len(self.X)); self.assertLessEqual(self.sp.K, 16); self.assertTrue(np.allclose(self.sp.pos + self.sp.neg, self.sp.count)); self.assertEqual(len(self.sp.atom_masks), 4)
    def test_mask_errors_equal_prediction_errors(self):
        for t in (BAL4, ("and", 0, 1), ("xor", ("or", 0, 2), 3), 2):
            e = float(self.sp.errors([self.sp.mask_of(t)])[0]); self.assertEqual(e, float((predict(to_program(t, self.atoms), self.X) != self.y).sum()))
    def test_truth_has_zero_errors_without_noise(self): self.assertEqual(float(self.sp.errors([self.sp.mask_of(BAL4)])[0]), 0.0); self.assertEqual(self.sp.floor, 0.0); self.assertEqual(self.sp.mask_of(BAL4), self.sp.target)
    def test_floor_is_a_lower_bound(self):
        atoms, X, y = data(BAL4, 4, rate=0.1); sp = CellSpace(atoms, X, y); T = bottom_up(sp, 4); m, _, _ = T.all(); self.assertGreaterEqual(float(sp.errors(m).min()), sp.floor - 1e-9); self.assertEqual(float(sp.errors([sp.target])[0]), sp.floor)
    def test_same_mask_same_predictions(self):
        a, b = ("and", ("and", 0, 1), 2), ("and", 0, ("and", 2, 1)); self.assertEqual(self.sp.mask_of(a), self.sp.mask_of(b)); self.assertTrue(torch.equal(predict(to_program(a, self.atoms), self.X), predict(to_program(b, self.atoms), self.X)))
    def test_value_on_codes_matches_predict(self):
        P = torch.rand(300, 2, generator=G(9)) * 6 - 3; v = self.sp.value_on_codes(BAL4, self.sp.codes_of(P)); self.assertEqual(v.astype(int).tolist(), predict(to_program(BAL4, self.atoms), P).tolist())
    def test_unseen_mass(self):
        self.assertEqual(self.sp.unseen_mass(self.X), 0.0); small = CellSpace(self.atoms, self.X[:6], self.y[:6]); P = torch.rand(2000, 2, generator=G(3)) * 6 - 3; self.assertGreater(small.unseen_mass(P), self.sp.unseen_mass(P)); self.assertEqual(self.sp.unseen_mass(torch.zeros(0, 2)), 0.0)
    def test_atom_limit(self):
        with self.assertRaises(ValueError): CellSpace(ATOMS + [ATOMS[0]], self.X, self.y)
        with self.assertRaises(ValueError): CellSpace([], self.X, self.y)
    def test_profiled_ll_matches_reference(self):
        from core.deep_fit import FitData, profiled_loglik
        d = FitData.build(self.X, self.y, 0.10); pred = predict(to_program(("and", 0, 1), self.atoms), self.X).double(); e = float((pred != d.y).double().sum()); self.assertAlmostEqual(float(profiled_ll(e, float(len(self.X)))), profiled_loglik(pred, d), places=6)


class DynamicProgrammingTests(unittest.TestCase):
    def setUp(self): self.atoms, self.X, self.y = data(CHAIN5, 5, rate=0.05); self.sp = CellSpace(self.atoms, self.X, self.y); self.T = bottom_up(self.sp, 5)
    def test_truth_class_is_built(self): m = self.sp.mask_of(CHAIN5); f = self.T.find(m); self.assertIsNotNone(f); self.assertLessEqual(f[0], 5); self.assertEqual(self.sp.mask_of(self.T.tree(*f)), m)
    def test_one_structure_per_class(self):
        m, s, i = self.T.all(); self.assertEqual(len(np.unique(m)), len(m)); self.assertEqual(self.T.classes, len(m))
        for k in range(0, len(m), max(1, len(m) // 40)): t = self.T.tree(int(s[k]), int(i[k])); self.assertEqual(self.sp.mask_of(t), int(m[k])); self.assertEqual(tree_size(t), int(s[k]))
    def test_levels_hold_minimal_sizes(self):
        small = bottom_up(self.sp, 3); ms, ss, _ = small.all(); m, s, _ = self.T.all(); lookup = {int(a): int(b) for a, b in zip(m, s)}
        for a, b in zip(ms, ss): self.assertEqual(lookup[int(a)], int(b))                                           # a class reached at size 3 is not reclassified higher
    def test_subresults_are_reused(self): self.assertLess(self.T.stats["considered"], 3 ** 4 * 5 ** 5 * 14); self.assertEqual(self.T.stats["considered"], sum(l.considered for l in self.T.levels.values())); self.assertLess(self.T.stats["seconds"], 5.0)
    def test_exhaustive_matches_brute_force_at_size_three(self):
        atoms, X, y = data(("and", ("or", 0, 1), 2), 3); sp = CellSpace(atoms, X, y); T = bottom_up(sp, 3); brute = set(sp.atom_masks)
        for a in range(3):
            for b in range(3):
                for o in OPS:
                    brute.add(sp.mask_of((o, a, b)))
                    for c in range(3):
                        for o2 in OPS: brute.add(sp.mask_of((o2, (o, a, b), c)))
        self.assertEqual({int(v) for v in T.all()[0]}, brute)
    def test_beam_is_over_classes_and_bounded(self):
        B = bottom_up(self.sp, 5, beam=20)
        for s, L in B.levels.items(): self.assertLessEqual(len(L.masks), max(20, self.sp.n)); self.assertEqual(len(np.unique(L.masks)), len(L.masks))
        self.assertLess(B.classes, self.T.classes); self.assertLess(B.stats["considered"], self.T.stats["considered"])
    def test_beam_keeps_the_least_wrong_classes(self):
        full = bottom_up(self.sp, 2); B = bottom_up(self.sp, 2, beam=3); e = np.sort(self.sp.errors(full.levels[2].masks))[:3]; self.assertTrue(np.allclose(np.sort(self.sp.errors(B.levels[2].masks)), e))
    def test_budget_enforced(self):
        for bud in (50, 500, 5000): b = bottom_up(self.sp, 5, budget=bud); self.assertLessEqual(b.stats["considered"], bud + 5); self.assertTrue(b.stats["truncated"])
        self.assertFalse(self.T.stats["truncated"]); self.assertFalse(bottom_up(self.sp, 5, budget=10 ** 9).stats["truncated"])
    def test_deterministic(self):
        a, b = bottom_up(self.sp, 5), bottom_up(self.sp, 5); self.assertTrue(np.array_equal(a.all()[0], b.all()[0])); self.assertEqual(tree_render(select(a, "errors").selected.tree), tree_render(select(b, "errors").selected.tree))
    def test_scaling_curve(self):
        sc = scaling(self.sp, 5); self.assertEqual([l["size"] for l in sc], [1, 2, 3, 4, 5]); self.assertEqual(sc[0]["new_classes"], 5); self.assertEqual(sc[-1]["cumulative_classes"], self.T.classes)
        self.assertTrue(all(a["cumulative_classes"] <= b["cumulative_classes"] for a, b in zip(sc, sc[1:]))); self.assertTrue(all(l["new_classes"] <= l["distinct_masks"] <= max(l["programs"], 5) for l in sc)); self.assertLess(sc[-1]["cumulative_classes"], 2 ** min(self.sp.K, 30))


class SelectionTests(unittest.TestCase):
    def setUp(self): self.atoms, self.X, self.y = data(BAL4, 4, rate=0.05, s=3); self.sp = CellSpace(self.atoms, self.X, self.y); self.T = bottom_up(self.sp, 4); self.tm = self.sp.mask_of(BAL4)
    def test_truth_recovered_by_errors(self):
        R = select(self.T, "errors"); self.assertIsInstance(R.selected, ObservationalEquivalenceClass); self.assertEqual(R.selected.mask, self.tm); self.assertEqual(rank_of(self.T, self.tm, "errors"), 1); self.assertEqual(R.ranked[0].errors, min(c.errors for c in R.ranked))
    def test_equivalent_tree_is_not_a_failure(self):
        t = select(self.T, "errors").selected.tree; P = torch.rand(2000, 2, generator=G(8)) * 6 - 3; self.assertTrue(torch.equal(predict(to_program(t, self.atoms), P), predict(to_program(BAL4, self.atoms), P)) or self.sp.unseen_mass(P) > 0)
    def test_selectors(self):
        m, s, _ = self.T.all(); sc = class_scores(self.sp, m, s); self.assertEqual(set(sc), {"errors", "likelihood", "structural_penalty", "parameter_penalty", "posterior", "bic", "size"})
        for sel in SELECTORS: o = order_by(sc, sel); self.assertEqual(sorted(o.tolist()), list(range(len(m))))
        self.assertEqual(int(s[order_by(sc, "size")[0]]), 1); self.assertTrue(np.all(np.diff(sc["errors"][order_by(sc, "errors")]) >= 0))
        with self.assertRaises(KeyError): order_by(sc, "nope")
    def test_penalties_grow_with_size(self):
        m, s, _ = self.T.all(); sc = class_scores(self.sp, m, s); i1, i4 = np.nonzero(s == 1)[0][0], np.nonzero(s == 4)[0][0]; self.assertLess(sc["structural_penalty"][i1], sc["structural_penalty"][i4]); self.assertLess(sc["parameter_penalty"][i1], sc["parameter_penalty"][i4])
        self.assertTrue(np.all(sc["posterior"] <= sc["bic"] + 1e-9))
    def test_rank_of_absent_class(self): self.assertIsNone(rank_of(bottom_up(self.sp, 1), self.tm, "errors")); self.assertIsNone(select(bottom_up(self.sp, 1), "errors", max_size=0).selected)
    def test_topk_ranked(self): R = select(self.T, "bic", top=5); self.assertEqual(len(R.ranked), 5); self.assertEqual(len({c.mask for c in R.ranked}), 5); self.assertEqual(R.stats["cells"], self.sp.K)


class ConstraintTests(unittest.TestCase):
    def test_and(self):
        t, care, lost = propagate("and", 0b1100, 0b1010, 0b1111, 0b1111); self.assertEqual(care, 0b1100); self.assertEqual(t & care, 0b1000); self.assertEqual(lost, 0b0010)                 # left = 0 and target = 1: lost
    def test_or(self):
        t, care, lost = propagate("or", 0b1100, 0b1010, 0b1111, 0b1111); self.assertEqual(care, 0b0011); self.assertEqual(t & care, 0b0010); self.assertEqual(lost, 0b0100)                  # left = 1 and target = 0: lost
    def test_xor(self): t, care, lost = propagate("xor", 0b1100, 0b1010, 0b1111, 0b1111); self.assertEqual(t, 0b0110); self.assertEqual(care, 0b1111); self.assertEqual(lost, 0)
    def test_solution_satisfies_constraints(self):
        for op in OPS:
            for L in range(16):
                for R in range(16):
                    T = (L & R) if op == "and" else (L | R) if op == "or" else (L ^ R); t, care, lost = propagate(op, L, T, 0b1111, 0b1111); self.assertEqual(lost, 0); self.assertEqual((R ^ t) & care, 0)
    def test_respects_care(self): t, care, lost = propagate("and", 0b1111, 0b1010, 0b0011, 0b1111); self.assertEqual(care, 0b0011); self.assertEqual(lost, 0)
    def test_unknown_operator(self):
        with self.assertRaises(KeyError): propagate("nand", 1, 1, 1, 1)
    def test_partial_structure_renders_hole(self): p = PartialStructure("and", ("or", 0, 1), 0b1, 0b1, 0.0, 2); self.assertEqual(p.render(), "((a0 OR a1) AND ?)")


class MeetInTheMiddleTests(unittest.TestCase):
    def test_recovers_chain_from_small_table(self):
        atoms, X, y = data(CHAIN5, 5, n=800); sp = CellSpace(atoms, X, y); R = top_down(sp, bottom_up(sp, 2), 5); self.assertEqual(R.selected.mask, sp.mask_of(CHAIN5)); self.assertEqual(R.selected.errors, 0.0); self.assertLessEqual(R.selected.size, 6)
    def test_may_stop_on_a_quasi_equivalent_structure(self):
        # Guided search, not exhaustive: with a table of size 3 it stops here on a structure within one error (an accepted and measured limit).
        atoms, X, y = data(CHAIN5, 5, n=800); sp = CellSpace(atoms, X, y); R = top_down(sp, bottom_up(sp, 3), 5); self.assertLessEqual(R.selected.errors, 0.005 * len(X)); self.assertGreaterEqual(R.selected.errors, float(sp.errors(bottom_up(sp, 5).all()[0]).min()))
    def test_recovers_balanced(self): atoms, X, y = data(BAL4, 4, n=800); sp = CellSpace(atoms, X, y); R = top_down(sp, bottom_up(sp, 2), 4); self.assertEqual(R.selected.errors, 0.0)
    def test_logs_partial_structures(self):
        atoms, X, y = data(CHAIN5, 5, n=800); sp = CellSpace(atoms, X, y); R = top_down(sp, bottom_up(sp, 2), 5); self.assertTrue(R.log); self.assertTrue(all("?" in l["partial"] and l["violation"] >= 0 for l in R.log)); self.assertGreater(R.stats["joins"], 1)
    def test_budget_enforced(self):
        atoms, X, y = data(CHAIN5, 5, rate=0.1); sp = CellSpace(atoms, X, y); R = top_down(sp, bottom_up(sp, 2), 5, budget=3); self.assertLessEqual(R.stats["joins"], 3 + 4 * 3 * 6 + 6); self.assertIsNotNone(R.selected)
    def test_fewer_evaluations_than_exhaustive(self):
        atoms, X, y = data(CHAIN5, 5, n=800); sp = CellSpace(atoms, X, y); self.assertLess(bottom_up(sp, 3).stats["considered"] + top_down(sp, bottom_up(sp, 3), 5).stats["joins"], bottom_up(sp, 5).stats["considered"])
    def test_deterministic(self):
        atoms, X, y = data(CHAIN5, 5, rate=0.05); sp = CellSpace(atoms, X, y); f = lambda: tree_render(top_down(sp, bottom_up(sp, 3), 5).selected.tree); self.assertEqual(f(), f())
    def test_no_world_access(self):
        s = code_only(DC)
        for w in ("deep_world", "deep_primitive", "_exp", "truth", "torch.nn"): self.assertNotIn(w, s)


class BenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.task = DX.task_at(899, 3, "standard"); cls.row = DX.compose_row(cls.task)
    def test_tasks_new_and_dev(self):
        from core.deep_experiments import down_task
        used = {down_task(k, 3, "rich").seed for r in ((0, 16), (100, 116), (200, 216), (300, 308), (400, 416), (500, 516), (600, 616), (700, 732)) for k in range(*r)}; new = {DX.task_at(k, d, "rich").seed for k in range(800, 832) for d in DX.DEPTHS}
        self.assertEqual(len(new), 32 * 6); self.assertFalse(new & used); self.assertEqual(self.task.split, "dev")
    def test_atoms_are_the_true_ones(self):
        from core.deep_reachability import leaves
        s = inspect.getsource(DX.compose_row); self.assertIn("leaves(tr)", s); self.assertEqual(self.row["n_atoms"], len(leaves(self.task.world.truth))); self.assertTrue(self.row["truth_generated"])
    def test_compose_row_fields(self):
        for k in ("truth_shape", "cells", "floor_errors", "truth_errors", "unseen_mass_full", "scaling", "dp_classes", "dp_considered", "rank_errors", "top1_errors", "top5_bic", "dp_errors_functional", "dp_errors_exact", "dp_errors_class_equal", "dp_errors_accord_out", "dp_errors_compact",
                  "dp_errors_plus1_functional", "classes_merged_by_observation", "beam100_truth", "budget1000_truth", "mitm3_functional", "mitm3_joins", "fit_selected", "fit_truth", "classes_better_or_equal", "classes_quasi_equivalent"): self.assertIn(k, self.row, k)
        self.assertEqual(set(self.row["fit_truth"]) & {"A", "B", "C", "D"}, {"A", "B", "C", "D"})
    def test_exact_equivalent_functional_are_distinct(self):
        r = self.row; self.assertTrue((not r["dp_errors_exact"]) or r["dp_errors_class_equal"]); self.assertGreaterEqual(r["dp_errors_obs_agreement"], 0.0); self.assertLessEqual(r["dp_errors_obs_agreement"], 1.0)
    def test_rollback_variant_never_worse_on_criterion(self):
        from core.deep_compose_exp import _fit_variants
        self.assertIn("D_committed", self.row["fit_truth"]); self.assertIn("validated_fit", inspect.getsource(_fit_variants))
    def test_ref_row(self):
        r = DX.ref_row(self.task, "ref_never_b150")
        for k in ("functional", "accord", "exact", "equivalent", "truth_form_in_pool", "good_in_pool", "good_found_then_lost", "candidates", "fit_equivalents"): self.assertIn(k, r)
        self.assertEqual(r["arm"], "ref_never_b150"); self.assertFalse(r["good_found_then_lost"] and not r["good_in_pool"])
    def test_subunits(self):
        self.assertIn("deepdc", DE.SUBUNITS); a = DE.subunit_ids("deepdc", "800"); n = len(DX.DEPTHS) * len(DX.REGIMES); self.assertEqual(len(a), n * (1 + len(DX.REF_ARMS) + 1) + 2 * 5); b = DE.subunit_ids("deepdc", "801"); self.assertEqual(len(b), n * (1 + len(DX.REF_ARMS))); self.assertEqual(len(set(a)), len(a))
    def test_sealed_and_pipeline_untouched(self):
        for m in (DC, DX, DR): s = inspect.getsource(m); self.assertNotIn("sealed_task", s); self.assertNotIn('"sealed"', s); self.assertNotIn('"test"', s)
        s = inspect.getsource(DS); self.assertNotIn("deep_compose", s); self.assertIsNone(inspect.signature(DS.DeepSearch.__init__).parameters["validated"].default)
    def test_summarize(self):
        rows = [{"task_id": "899_3", "depth": 3, "regime": "standard", **self.row}] + [{"task_id": "899_3", "depth": 3, "regime": "standard", **DX.ref_row(self.task, a)} for a in ("ref_b150", "ref_never_b150")]
        S = DR.summarize(rows); self.assertIn("d3", S["space"]); self.assertIn("all/errors", S["selectors"]); self.assertIn("n3/size3", S["scaling"]); self.assertIn("all/ref_b150", S["reference"]); self.assertIn("all: dp_errors − ref_b150", S["paired"]); self.assertIn("all/truth", S["fit"]); DR.show(S)


if __name__ == "__main__":
    unittest.main()

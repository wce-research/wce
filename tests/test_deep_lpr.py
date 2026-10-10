import inspect
import math
import random
import unittest
try:
    from tests._source import code_only
except ImportError:
    from _source import code_only

import torch

import core.deep_experiments as DE
import core.deep_lp_report as LR
import core.deep_lpr as LP
import core.deep_lpr_exp as LX
import core.deep_search as DS
from core.boundary_program import BNode, predict
from core.deep_e2e import hypotheses_from, reduce
from core.deep_lpr import (DEFAULTS, STATES, CompatiblePrimitiveSets, ConditionalUtility, LatentPrimitiveSetHypothesis, LatentRecovery, PrimitiveSetCompleteness, completeness, compose_sets, load_config, pool, propose_missing, recover, select_sets,
                           strong_downstream, utilities)
from core.deep_primitives2 import disc_atom, line_atom

G = lambda s=0: torch.Generator().manual_seed(s)
A0, A1, A2 = disc_atom(-1.0, 0.5, 1.2), line_atom(0.3, 0.1), disc_atom(0.6, -0.8, 1.0)
XA, XB = line_atom(0.0, 0.0), line_atom(math.pi / 2, 0.0)                      # x ≤ 0 and y ≤ 0: their exclusive or is a checkerboard
FAKES = [disc_atom(2.2, 2.2, 0.5), line_atom(1.1, -2.0), disc_atom(-2.3, -2.2, 0.6), line_atom(-0.7, 1.9), disc_atom(1.9, -2.1, 0.5), line_atom(2.6, 2.1), disc_atom(-0.1, 2.4, 0.5), line_atom(0.9, 2.4), disc_atom(2.4, 0.0, 0.4), line_atom(-2.2, 2.2)]
TRUTH = BNode("xor", children=(BNode("and", children=(A0, A1)), A2))
XOR = BNode("xor", children=(XA, XB))


def data(prog=TRUTH, n=500, rate=0.0, s=0):
    X = torch.rand(n, 2, generator=G(s)) * 6 - 3; y = predict(prog, X); f = torch.rand(n, generator=G(s + 1)) < rate; return X, torch.where(f, 1 - y, y)


def acc(prog, ref=TRUTH): P = torch.rand(3000, 2, generator=G(9)) * 6 - 3; return float((predict(prog, P) == predict(ref, P)).float().mean())
D = lambda atoms, src="t": [(a, float(len(atoms) - i), src) for i, a in enumerate(atoms)]


class UtilityTests(unittest.TestCase):
    def setUp(self): self.X, self.y = data(XOR, rate=0.02); self.H = hypotheses_from(D(FAKES[:3] + [XA, XB]), self.X)
    def test_conditional_only_primitives_are_visible(self):
        U = utilities(self.H, self.y); self.assertIsInstance(U[0], ConditionalUtility); xa = next(u for u, h in zip(U, self.H) if repr(h.program) == repr(XA)); xb = next(u for u, h in zip(U, self.H) if repr(h.program) == repr(XB))
        self.assertLess(xa.alone, 25.0); self.assertGreater(xa.pair, 150.0); self.assertEqual(self.H[xa.partner].program.params, XB.params); self.assertTrue(xa.conditional_only or xa.alone >= 1.0); self.assertGreater(xb.pair, 150.0)
    def test_in_context(self):
        i = next(k for k, h in enumerate(self.H) if repr(h.program) == repr(XA)); j = next(k for k, h in enumerate(self.H) if repr(h.program) == repr(XB)); U = utilities(self.H, self.y, selected=[i]); self.assertGreater(U[j].in_context, 150.0); self.assertEqual(U[i].in_context, 0.0)
    def test_pair_pool_bounds_work(self): U = utilities(self.H, self.y, pair_pool=2); self.assertEqual(sum(u.partner is not None for u in U) <= 2, True); self.assertEqual(len(U), len(self.H))


class SetTests(unittest.TestCase):
    def setUp(self): self.X, self.y = data(XOR, rate=0.02); self.H = hypotheses_from(D(FAKES[:4] + [XA, XB]), self.X)
    def _has(self, s, *atoms): return {repr(a) for a in atoms} <= {repr(p) for p in s.programs}
    def test_pair_rescue_keeps_interaction_only_atoms(self):
        S = select_sets(self.H, self.y, min_gain=4.0); self.assertIsInstance(S, CompatiblePrimitiveSets); self.assertTrue(self._has(S.best, XA, XB)); self.assertTrue(any(st["pair"] for st in S.best.trace))
    def test_without_pairs_they_are_lost(self): S = select_sets(self.H, self.y, min_gain=40.0, pairs=False); self.assertFalse(S.best is not None and self._has(S.best, XA, XB)); S2 = select_sets(self.H, self.y, min_gain=40.0, pairs=True); self.assertTrue(self._has(S2.best, XA, XB))
    def test_old_reduction_has_the_defect(self): kept, _ = reduce(self.H, self.y, "greedy_floor", min_gain=40.0); self.assertFalse({repr(XA), repr(XB)} <= {repr(h.program) for h in kept})
    def test_set_fields(self):
        s = select_sets(self.H, self.y).best; self.assertIsInstance(s, LatentPrimitiveSetHypothesis)
        for k in ("atoms", "indices", "floor", "score", "coverage", "contradictions", "cost", "trace", "programs", "provenance", "equivalences", "uncertainties"): self.assertTrue(hasattr(s, k), k)
        self.assertEqual(len(s.atoms), len(s.indices)); self.assertAlmostEqual(s.coverage, 1 - s.floor / len(self.y)); self.assertGreater(s.cost, 0); self.assertEqual(len(s.provenance), len(s.atoms))
    def test_topk_sets_distinct_and_ordered(self):
        S = select_sets(self.H, self.y, n_sets=3); self.assertLessEqual(len(S), 3); self.assertEqual(len({s.indices for s in S.sets}), len(S)); self.assertEqual([s.score for s in S.sets], sorted((s.score for s in S.sets), reverse=True)); self.assertGreaterEqual(len(S.union(3)), len(S.best.atoms)); self.assertEqual(len(S.union(1)), len(S.best.atoms))
    def test_max_atoms_and_beam(self):
        H = hypotheses_from(D(FAKES + [XA, XB]), self.X); S = select_sets(H, self.y, max_atoms=3, beam=2); self.assertTrue(all(len(s.atoms) <= 3 for s in S.sets)); self.assertEqual(len(select_sets([], self.y)), 0)
    def test_contradictions_counted(self):
        H = hypotheses_from(D([XA, line_atom(0.0, 0.25), XB]), self.X, same=0.999); s = LP._make_set(H, (0, 1, 2), self.y); self.assertGreaterEqual(s.contradictions, 1); self.assertEqual(LP._make_set(H, (0, 2), self.y).contradictions, 0)
    def test_min_gain_controls_size(self): a = select_sets(self.H, self.y, min_gain=1.0).best; b = select_sets(self.H, self.y, min_gain=60.0).best; self.assertGreaterEqual(len(a.atoms), len(b.atoms))
    def test_deterministic(self): f = lambda: [s.indices for s in select_sets(self.H, self.y).sets]; self.assertEqual(f(), f())


class CompletenessTests(unittest.TestCase):
    def setUp(self): self.X, self.y = data(rate=0.04)
    def test_complete_model_errors_are_isolated(self): c = completeness(self.X, self.y, predict(TRUTH, self.X)); self.assertIsInstance(c, PrimitiveSetCompleteness); self.assertTrue(c.complete); self.assertLess(c.excess, 0.10); self.assertAlmostEqual(c.error_rate, float((predict(TRUTH, self.X) != self.y).float().mean()))
    def test_missing_primitive_gives_structured_residual(self):
        part = BNode("and", children=(A0, A1)); c = completeness(self.X, self.y, predict(part, self.X)); self.assertFalse(c.complete); self.assertGreater(c.excess, 0.3); self.assertGreater(c.gap, completeness(self.X, self.y, predict(TRUTH, self.X)).gap)
    def test_no_errors(self): X, y = data(); c = completeness(X, y, y.clone()); self.assertTrue(c.complete); self.assertEqual(c.excess, 0.0)
    def test_threshold(self): p = predict(BNode("and", children=(A0, A1)), self.X); self.assertTrue(completeness(self.X, self.y, p, threshold=0.99).complete); self.assertFalse(completeness(self.X, self.y, p, threshold=-1.0).complete)
    def test_residual_proposal_targets_the_missing_primitive(self):
        from core.deep_partial_exp import functional
        X, y = data(n=900); part = BNode("and", children=(A0, A1)); new = propose_missing(X, y, predict(part, X), 3); self.assertTrue(new); P = torch.rand(3000, 2, generator=G(4)) * 6 - 3; self.assertGreater(max(functional(A2, p, P) for p, _, _ in new), 0.93); self.assertTrue(all(src.startswith("residual:") for _, _, src in new))
    def test_no_proposal_without_residual(self): X, y = data(); self.assertEqual(propose_missing(X, y, y.clone()), [])


class RecoveryTests(unittest.TestCase):
    def setUp(self): self.X, self.y = data(n=900, rate=0.02)
    def test_strong_downstream_with_true_atoms(self):
        r = strong_downstream(D(FAKES[:3] + [A0, A1, A2]), self.X, self.y, {"fit": "none"}); self.assertIsInstance(r, LatentRecovery); self.assertGreater(acc(r.program), 0.99); self.assertEqual(r.state, "COMPLETE"); self.assertLessEqual(r.stats["kept"], 6); self.assertGreater(r.stats["considered"], 0)
    def test_ten_false_atoms_tolerated(self): r = strong_downstream(D(FAKES + [A0, A1, A2]), self.X, self.y, {"fit": "none"}); self.assertGreater(acc(r.program), 0.99)
    def test_loop_recovers_a_missing_primitive(self):
        before = strong_downstream(D([A0, A1]), self.X, self.y, {"fit": "none"}); after = recover(self.X, self.y, 0, {"fit": "none", "iters": 2, "incomplete": 0.10}, D([A0, A1])); self.assertLess(acc(before.program), 0.95); self.assertGreater(acc(after.program), 0.97)
        self.assertGreater(after.stats["iterations"], 1); self.assertTrue(any(l.get("proposed", 0) > 0 for l in after.log)); self.assertEqual(after.log[0]["action"], "commit")
    def test_loop_is_bounded_and_monotone(self):
        r = recover(self.X, self.y, 0, {"fit": "none", "iters": 2, "incomplete": -1.0}, D([A0, A1])); self.assertLessEqual(len(r.log), 3); e = [l["errors"] for l in r.log if l["action"] == "commit"]; self.assertEqual(e, sorted(e, reverse=True)); self.assertEqual(r.finalization.errors_after, min(e))
    def test_rollback_keeps_previous_state(self):
        r = recover(self.X, self.y, 0, {"fit": "none", "iters": 2, "incomplete": -1.0}, D([A0, A1, A2])); self.assertGreater(acc(r.program), 0.99)
        if any(l["action"] == "rollback" for l in r.log): self.assertEqual(r.finalization.errors_after, min(l["errors"] for l in r.log if l["action"] == "commit"))
    def test_no_loop_when_complete(self): r = recover(self.X, self.y, 0, {"fit": "none", "incomplete": 0.10}, D([A0, A1, A2])); self.assertEqual(r.stats["iterations"], 1); self.assertEqual(r.state, "COMPLETE")
    def test_credit_is_explicit(self):
        r = strong_downstream(D([A0, A1, A2, FAKES[0]]), self.X, self.y, {"fit": "none"}); self.assertTrue(r.credit); self.assertTrue(all(0.0 <= v <= 1.0 for v in r.credit.values())); self.assertEqual(set(r.credit), set(r.chosen.indices)); self.assertGreater(max(r.credit.values()), 0.5)
    def test_compose_sets_picks_by_composition(self):
        H = hypotheses_from(D([A0, A1, A2]), self.X); S = select_sets(H, self.y); s, R, F, n = compose_sets(S, self.X, self.y, "bic", "none"); self.assertIn(s, S.sets); self.assertGreater(n, 0); self.assertEqual(compose_sets(CompatiblePrimitiveSets([]), self.X, self.y)[0], None)
    def test_validated_finalization_never_worse(self): r = strong_downstream(D([disc_atom(-0.9, 0.6, 1.1), line_atom(0.35, 0.15), disc_atom(0.7, -0.75, 0.95)]), self.X, self.y); self.assertLessEqual(r.finalization.errors_after, r.finalization.errors_before)
    def test_states_and_config(self): self.assertEqual(STATES, ("COMPLETE", "INCOMPLETE", "POSSIBLY_OUT_OF_CLASS")); c = load_config(); self.assertLessEqual(set(DEFAULTS), set(c)); self.assertLessEqual(c["max_atoms"], 6); self.assertLessEqual(c["iters"], 3)
    def test_deterministic(self):
        f = lambda: recover(self.X, self.y, 0, {"fit": "none"}, D(FAKES[:3] + [A0, A1, A2])); a, b = f(), f(); self.assertEqual(repr(a.program), repr(b.program)); self.assertEqual(a.chosen.indices, b.chosen.indices)
    def test_no_oracle(self):
        s = code_only(LP)
        for w in ("deep_world", "truth", "leaves(", "_exp", "deep_primitive_eval"): self.assertNotIn(w, s)


class PoolTests(unittest.TestCase):
    def test_recall_first_pool(self):
        X, y = data(BNode("xor", children=(A0, A1)), n=400, rate=0.03); a = pool(X, y, 5, 0, ("flat",)); b = pool(X, y, 5, 0); self.assertGreater(len(b), len(a)); self.assertTrue({"flat", "fragments", "components"} >= {s.split(":")[0] for _, _, s in b}); self.assertTrue(all(0.0 <= s <= 1.0 for _, s, _ in b))
        self.assertGreater(len(pool(X, y, 5, 0, ("flat",))), len(pool(X, y, 1, 0, ("flat",))))
    def test_real_chain_runs(self): X, y = data(BNode("xor", children=(A0, A1)), n=400, rate=0.03); r = recover(X, y); self.assertIsNotNone(r.program); self.assertIn(r.state, STATES); self.assertGreater(r.stats["pool"], 5)


class BenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.task = LX.task_at(1099, 3, "standard")
    def test_tasks_new_and_dev(self):
        from core.deep_experiments import down_task
        used = {down_task(k, 3, "rich").seed for r in ((0, 16), (100, 116), (200, 216), (300, 308), (400, 416), (500, 516), (600, 616), (700, 732), (800, 832), (900, 932), (990, 1000)) for k in range(*r)}; new = {LX.task_at(k, d, "rich").seed for k in range(1000, 1032) for d in LX.DEPTHS}
        self.assertEqual(len(new), 192); self.assertFalse(new & used); self.assertEqual(self.task.split, "dev"); self.assertFalse(new & {LX.task_at(k, d, "rich").seed for k in range(1090, 1100) for d in LX.DEPTHS})
    def test_set_recovery_levels(self):
        from core.deep_reachability import leaves
        tl = leaves(self.task.world.truth); s = LX.set_recovery(tl, self.task); self.assertTrue(s["all_exact"] and s["all_functional"] and s["all_strict"]); self.assertEqual(s["recall_strict"], 1.0); self.assertEqual(s["param_error"], 0.0)
        m = LX.set_recovery(tl[1:], self.task); self.assertFalse(m["all_functional"]); self.assertTrue(m["most_functional"]); self.assertAlmostEqual(m["recall_functional"], (len(tl) - 1) / len(tl)); e = LX.set_recovery([], self.task); self.assertEqual(e["recall_functional"], 0.0); self.assertIsNone(e["param_error"])
    def test_approximate_atom_is_functional_not_exact(self):
        from core.deep_reachability import leaves
        tl = leaves(self.task.world.truth); j = LX.jitter(tl, 0.03, random.Random(1)); s = LX.set_recovery(j, self.task); self.assertGreater(s["recall_functional"], 0.5); self.assertGreater(s["param_error"], 0.0)
    def test_complement_counts_as_the_atom(self):
        from core.deep_reachability import leaves
        tl = leaves(self.task.world.truth); s = LX.set_recovery([BNode("not", children=(a,)) for a in tl], self.task); self.assertTrue(s["all_exact"])
    def test_jitter_families(self):
        from core.deep_reachability import leaves
        tl = leaves(self.task.world.truth); self.assertEqual([a.param_vector() for a in LX.jitter(tl, 0.0, random.Random(0))], [a.param_vector() for a in tl]); k = LX._name(tl[0]).replace("not_", ""); j = LX.jitter(tl, 0.1, random.Random(0), kind=k)
        self.assertTrue(all((a.param_vector() != b.param_vector()) == (LX._name(b).replace("not_", "") == k) for a, b in zip(j, tl))); self.assertEqual([repr(a.skeleton()) for a in j], [repr(a.skeleton()) for a in tl])
        for a in tl: fi = LX._family_index(a); self.assertEqual(sorted(i for ii in fi.values() for i in ii), list(range(a.n_params())))
    def test_visibility(self): v = LX.atom_visibility(self.task); self.assertEqual(len(v), 3); self.assertTrue(all(0.0 <= x <= 1.0 for x in v))
    def test_primary_arms_fixed(self):
        self.assertEqual(LX.PRIMARY, ("A", "B", "C")); self.assertEqual(LX.PREREGISTERED_SECONDARY, ("B5",)); self.assertEqual(LX.C_VARIANTS["C"], {}); s = inspect.getsource(LX.chain_rows); self.assertNotIn("world.truth", s); self.assertIn("strong_downstream(fatoms", s); self.assertIn("recover(X, y, seed, cfg)", s)
    def test_diag_rows(self):
        s = LX.sens_rows(self.task); self.assertTrue(s); self.assertEqual({r["sigma"] for r in s}, set(LX.SIGMAS)); self.assertTrue({"all", "kind", "family"} >= {r["mode"] for r in s}); f = LX.fp_rows(self.task); self.assertEqual([r["arm"] for r in f if r["engine"] == "sets"], [f"plus{n}" for n in LX.FALSE_COUNTS])
        self.assertEqual(sum(r["engine"] == "old_reduction" for r in f), 1); c = LX.complete_rows(self.task); self.assertEqual([r["arm"] for r in c], ["complete", "missing1"]); self.assertIn("missing_found", c[1])
    def test_subunits(self): self.assertIn("deeplp", DE.SUBUNITS); a = DE.subunit_ids("deeplp", "1000"); self.assertEqual(len(a), 12 * 2 + 3); self.assertEqual(len(set(a)), len(a))
    def test_sealed_and_pipeline_untouched(self):
        for m in (LP, LX, LR): s = inspect.getsource(m); self.assertNotIn("sealed_task", s); self.assertNotIn('"sealed"', s); self.assertNotIn('"test"', s)
        s = inspect.getsource(DS); self.assertNotIn("deep_lpr", s); self.assertIsNone(inspect.signature(DS.DeepSearch.__init__).parameters["validated"].default)
    def test_summarize(self):
        tag = {"task_id": "1099_3", "depth": 3, "regime": "standard"}; reg = LX._regions(self.task); import time
        rows = [{**tag, **LX._row(self.task, n, strong_downstream(LX.discover(self.task.X, self.task.y, 3, 0), self.task.X, self.task.y), reg, time.time())} for n in ("A", "B")] + [{**tag, **LX._row(self.task, "C", recover(self.task.X, self.task.y), reg, time.time())}]
        rows += [{**tag, **r} for r in LX.sens_rows(self.task) + LX.fp_rows(self.task) + LX.complete_rows(self.task)] + [{"task_id": "x", "depth": 0, "regime": "standard", **LX.ooc_row(LX.ooc_task(1099, "primitive_absent"), "primitive_absent")}]
        S = LR.summarize(rows); self.assertIn("all/C", S["chain"]); self.assertIn("all: C − B", S["paired"]); self.assertIn("C/n3", S["set_scaling"]); self.assertIn("independent_model", S["set_scaling"]["C/n3"]); self.assertIn("C", S["miss_correlation"]); self.assertIn("missing1", S["completeness"]); self.assertIn("primitive_absent", S["ooc"]); LR.show(S)


if __name__ == "__main__":
    unittest.main()

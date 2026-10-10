import inspect
import math
import unittest

import torch

import core.deep_experiments as DX
import core.deep_fragments as FR
import core.deep_fragments_exp as FX
import core.deep_ft_report as FRP
import core.deep_search as DS
from core.boundary_program import BNode, predict
from core.deep_fit import DT
from core.deep_fragments import (ATOM_TYPE, FAMILIES, FAMILY_TYPES, GROUPINGS, OWNERSHIP, SKELETONS, BoundaryFragment, CompatibleFragmentAssignments, FragmentGraph, OcclusionOwnership, TypeHypothesis, assignments, build_graph,
                                 extract_fragments, fit_fragment, flat_types, fuse, group_points, infer_types, literals, make_fragment, ownership, rank_by_group, reassign, signals_of, type_hypotheses, type_of, typed_labels)
from core.deep_partial import observe
from core.deep_primitives import atom_name
from core.deep_primitives2 import disc_atom, line_atom

G = lambda s=0: torch.Generator().manual_seed(s)
T = lambda c, ch: BNode("translate", params=c, children=(ch,))
DISC = disc_atom(0.2, 0.0, 1.2)
ANTI = T((0.2, 0.0), BNode("pred", "norm", "ge", (1.2,)))
HALF = line_atom(0.3, 0.1)
RING = T((-0.3, 0.2), BNode("pred", "norm", "in", (0.7, 1.6)))
BAND = BNode("pred", "dot", "in", (1.0, -0.6, 0.6))
SECTOR = T((-0.5, 0.0), BNode("pred", "angle", "in", (0.4, 0.8)))
OCC = line_atom(0.0, 0.3)
AND = lambda a, b: BNode("and", children=(a, b))


def scene(prog, n=900, rate=0.0, s=0):
    X = torch.rand(n, 2, generator=G(s)) * 6 - 3; y = predict(prog, X); f = torch.rand(n, generator=G(s + 1)) < rate; return X, torch.where(f, 1 - y, y)


def arc(c=(0.2, 0.0), r=1.2, a0=0.0, a1=math.pi / 2, n=40):
    t = torch.linspace(a0, a1, n, dtype=DT); return torch.tensor(c, dtype=DT) + r * torch.stack([torch.cos(t), torch.sin(t)], 1)


def seg(p0, p1, n=30): t = torch.linspace(0, 1, n, dtype=DT)[:, None]; return torch.tensor(p0, dtype=DT) * (1 - t) + torch.tensor(p1, dtype=DT) * t


def OB(prog, **kw): X, y = scene(prog, **kw); return observe(X, y)


class FragmentTests(unittest.TestCase):
    def test_fit_prefers_simple_model(self):
        self.assertEqual(fit_fragment(seg((0, 0), (2, 1)))[0], "line"); m, p, rl, rc = fit_fragment(arc(a1=2.0)); self.assertEqual(m, "circle"); self.assertAlmostEqual(p[2], 1.2, places=3); self.assertGreater(rl, rc)
        self.assertEqual(fit_fragment(arc(r=3.4, a1=0.05, n=8))[0], "line")                                 # nearly flat arc: the line suffices
    def test_fragment_fields(self):
        ob = OB(DISC); fs = extract_fragments(ob); self.assertTrue(fs); f = fs[0]; self.assertIsInstance(f, BoundaryFragment)
        for k in ("ids", "points", "normals", "model", "params", "curvature", "length", "endpoints", "confidence", "polarity", "neighbors", "uncertainty", "rms_line", "rms_circle"): self.assertTrue(hasattr(f, k), k)
        self.assertEqual(f.endpoints.shape, (2, 2)); self.assertTrue(torch.equal(f.points, ob.points[f.ids])); self.assertTrue(0 < f.confidence <= 1); self.assertGreater(f.length, 0)
    def test_curvature(self):
        ob = OB(DISC); c = [f for f in extract_fragments(ob) if not f.straight and abs(f.params[2] - 1.2) < 0.2]; self.assertTrue(c); self.assertAlmostEqual(c[0].curvature, 1 / c[0].params[2]); l = [f for f in extract_fragments(OB(HALF)) if f.straight]; self.assertTrue(l); self.assertEqual(l[0].curvature, 0.0)
    def test_polarity_distinguishes_disc_from_antidisc(self):
        pd = [f.polarity for f in extract_fragments(OB(DISC)) if not f.straight and abs(f.params[2] - 1.2) < 0.2 and math.hypot(f.params[0] - 0.2, f.params[1]) < 0.3]; pa = [f.polarity for f in extract_fragments(OB(ANTI)) if not f.straight and abs(f.params[2] - 1.2) < 0.2 and math.hypot(f.params[0] - 0.2, f.params[1]) < 0.3]
        self.assertEqual(pd[0], -1); self.assertEqual(pa[0], 1)
    def test_consensus_fragments_may_overlap(self):
        fs = extract_fragments(OB(AND(DISC, OCC))); ids = [i for f in fs for i in f.ids.tolist()]; self.assertGreater(len(ids), len(set(ids)))                     # a middle can remain compatible with several readings
    def test_disjoint_modes_do_not_overlap(self):
        fs = extract_fragments(OB(AND(DISC, OCC)), "tangent"); ids = [i for f in fs for i in f.ids.tolist()]; self.assertEqual(len(ids), len(set(ids)))
    def test_forced_model(self):
        ob = OB(DISC); ids = torch.arange(min(12, len(ob.points))); f = make_fragment(0, ids, ob, ("circle", (0.2, 0.0, 1.2))); self.assertEqual(f.model, "circle"); self.assertEqual(f.params, (0.2, 0.0, 1.2)); self.assertTrue(math.isfinite(f.rms_circle))
    def test_no_boundary(self): X, _ = scene(DISC); ob = observe(X, torch.zeros(len(X), dtype=torch.long)); self.assertEqual(extract_fragments(ob), []); self.assertEqual(len(group_points(ob, "tangent")), 0); self.assertEqual(infer_types(ob).selected, [])


class GroupingTests(unittest.TestCase):
    def setUp(self): self.ob = OB(AND(DISC, OCC))
    def test_all_modes_return_labels(self):
        for m in GROUPINGS: lab = group_points(self.ob, m); self.assertEqual(len(lab), len(self.ob.points), m); self.assertGreaterEqual(int(lab.max()), 0, m)
    def test_tangent_splits_corner(self):
        M = torch.cat([seg((0, 0), (2, 0), 20), seg((2, 0), (2, 2), 20)]); N = torch.cat([torch.tensor([[0.0, 1.0]], dtype=DT).expand(20, 2), torch.tensor([[1.0, 0.0]], dtype=DT).expand(20, 2)]); ob = self.ob.__class__(M, N, torch.zeros(40, dtype=torch.long), self.ob.pairs, self.ob.X, self.ob.y)
        self.assertEqual(len(torch.unique(group_points(ob, "proximity"))), 1); t = group_points(ob, "tangent"); self.assertEqual(len(torch.unique(t)), 2); self.assertNotEqual(int(t[0]), int(t[-1]))
    def test_orientation_merges_collinear_pieces(self):
        M = torch.cat([seg((0, 0), (1, 0), 12), seg((2, 0), (3, 0), 12)]); N = torch.tensor([[0.0, 1.0]], dtype=DT).expand(24, 2); ob = self.ob.__class__(M, N, torch.zeros(24, dtype=torch.long), self.ob.pairs, self.ob.X, self.ob.y)
        self.assertEqual(len(torch.unique(group_points(ob, "tangent"))), 2); self.assertEqual(len(torch.unique(group_points(ob, "orientation"))), 1); self.assertEqual(len(torch.unique(group_points(ob, "center"))), 2)
    def test_center_merges_arcs_of_one_circle(self):
        M = torch.cat([arc(a0=0.0, a1=1.2, n=20), arc(a0=2.5, a1=3.7, n=20)]); c = torch.tensor([0.2, 0.0], dtype=DT); N = (M - c) / 1.2; ob = self.ob.__class__(M, N, torch.zeros(40, dtype=torch.long), self.ob.pairs, self.ob.X, self.ob.y)
        self.assertEqual(len(torch.unique(group_points(ob, "tangent"))), 2); self.assertEqual(len(torch.unique(group_points(ob, "center"))), 1); self.assertEqual(len(torch.unique(group_points(ob, "orientation"))), 2); self.assertEqual(len(torch.unique(group_points(ob, "shared"))), 1)
    def test_deterministic(self):
        for m in GROUPINGS: self.assertTrue(torch.equal(group_points(self.ob, m, seed=2), group_points(self.ob, m, seed=2)), m)
    def test_pair_scores(self):
        src = torch.tensor([0, 0, 0, 1, 1, 1]); self.assertEqual(FX.pair_scores(src.clone(), src)["f1"], 1.0); s = FX.pair_scores(torch.zeros(6, dtype=torch.long), src); self.assertEqual(s["recall"], 1.0); self.assertAlmostEqual(s["precision"], 0.4)
        s = FX.pair_scores(torch.arange(6), src); self.assertEqual(s["recall"], 0.0); self.assertIsNone(FX.pair_scores(torch.tensor([0]), torch.tensor([0]))["f1"]); self.assertEqual(FX.pair_scores(torch.tensor([0, 0, 5]), torch.tensor([1, 1, -1]))["f1"], 1.0)
    def test_true_sources(self):
        M = torch.cat([arc(n=10), seg((0.3, -2), (0.3, 2), 10), torch.tensor([[2.8, 2.8]], dtype=DT)]); comp, atom = FX.true_sources(M, [DISC, OCC]); self.assertEqual(atom[:10].tolist(), [0] * 10); self.assertEqual(int(atom[-1]), -1); self.assertTrue(bool((atom[10:20] == 1).float().mean() > 0.7))


class GraphTests(unittest.TestCase):
    def test_relations(self):
        g = infer_types(OB(RING), 1).graph; self.assertIsInstance(g, FragmentGraph); self.assertIn("concentric", {r for _, _, r, _ in g.edges}); self.assertIn("parallel", {r for _, _, r, _ in infer_types(OB(BAND), 1).graph.edges}); self.assertIn("vertex", {r for _, _, r, _ in infer_types(OB(SECTOR, n=1400), 1).graph.edges})
    def test_no_vertex_for_crossing_lines(self):
        a, b = line_atom(0.0, 0.0), line_atom(math.pi / 2, 0.0); g = infer_types(OB(BNode("xor", children=(a, b))), 1).graph; long = [f.index for f in g.fragments if f.straight and f.length > 4.0]; self.assertFalse([e for e in g.edges if e[2] == "vertex" and e[0] in long and e[1] in long])
    def test_groups_cover_all_fragments(self): g = infer_types(OB(RING), 1).graph; self.assertEqual(sorted(i for grp in g.groups() for i in grp), list(range(len(g.fragments))))
    def test_assignments_keep_alternatives(self):
        g = infer_types(OB(RING), 1).graph; A = assignments(g); self.assertEqual(len(A), len(g.fragments)); self.assertIsInstance(A[0], CompatibleFragmentAssignments); roles = {r for a in A for r, _ in a.roles}; self.assertTrue({"ring_inner", "ring_outer"} <= roles); self.assertIn("noise", roles)
        for a in A: self.assertAlmostEqual(sum(w for _, w in a.roles), 1.0, places=6); self.assertGreater(len(a.roles), 1); self.assertEqual([w for _, w in a.roles], sorted((w for _, w in a.roles), reverse=True))
    def test_assignment_roles_by_structure(self):
        self.assertIn("band_edge", {r for a in assignments(infer_types(OB(BAND), 1).graph) for r, _ in a.roles}); self.assertIn("sector_ray", {r for a in assignments(infer_types(OB(SECTOR, n=1400), 1).graph) for r, _ in a.roles})
        self.assertIn("occluder", {r for a in assignments(infer_types(OB(AND(DISC, OCC)), 1).graph) for r, _ in a.roles}); self.assertTrue(CompatibleFragmentAssignments(0, []).top() == "noise")


class TypeTests(unittest.TestCase):
    def test_grammar(self):
        self.assertEqual(len(FAMILIES), 5); self.assertEqual({t for ts in FAMILY_TYPES.values() for t in ts}, set(SKELETONS)); self.assertEqual(set(ATOM_TYPE.values()), set(SKELETONS)); self.assertEqual(type_of(RING), "ring"); self.assertEqual(type_of(T((0.0, 0.0), BNode("not", children=(BNode("pred", "angle", "in", (0.1, 0.5)),)))), "sector")
        for t, sk in SKELETONS.items(): self.assertEqual(ATOM_TYPE[atom_name(sk)], t)
    def test_each_type_inferred_unmasked(self):
        for prog, t in ((DISC, "disc"), (ANTI, "antidisc"), (HALF, "half"), (RING, "ring"), (BAND, "band")):
            inf = infer_types(OB(prog), 1); self.assertTrue(inf.selected, t); self.assertEqual(inf.selected[0].type, t); self.assertEqual({v: f for f, vs in FAMILY_TYPES.items() for v in vs}[t], inf.selected[0].family)
    def test_sector_inferred(self): inf = infer_types(OB(SECTOR, n=1400), 3); self.assertIn("sector", {h.type for h in inf.hypotheses}); self.assertIn("radial-composite", {h.family for h in inf.hypotheses})
    def test_signals_reported_separately(self):
        h = infer_types(OB(RING), 1).selected[0]; self.assertEqual(set(h.signals), {"geometric", "coherence", "support", "explained", "seen", "absent", "occluded", "complexity", "stability"}); self.assertEqual(h.signals["complexity"], 4.0); self.assertAlmostEqual(h.score, fuse(h.signals))
    def test_fuse_penalises_absence_and_complexity(self):
        s = {"support": 40.0, "absent": 0.0, "complexity": 3.0}; self.assertGreater(fuse(s), fuse({**s, "absent": 0.5})); self.assertGreater(fuse(s), fuse({**s, "complexity": 4.0})); self.assertGreater(fuse({**s, "support": 46.0, "complexity": 4.0}), fuse(s))
    def test_topk_keeps_competing_types(self):
        ob = OB(RING); k1, k3 = infer_types(ob, 1).selected, infer_types(ob, 3).selected; self.assertGreater(len(k3), len(k1)); self.assertGreater(len({h.type for h in k3}), 1); self.assertEqual(k3[0].type, k1[0].type)
        for h in k3: self.assertTrue(0.0 <= h.confidence <= 1.0)
    def test_rank_groups_independent_primitives(self):
        ob = OB(BNode("xor", children=(disc_atom(-1.6, 1.3, 0.9), line_atom(0.0, 0.8)))); sel = infer_types(ob, 1).selected; self.assertGreaterEqual(len(sel), 2); self.assertTrue({"disc", "antidisc"} & {h.type for h in sel}); self.assertIn("half", {h.type for h in sel})
    def test_rank_without_observation(self): self.assertEqual(rank_by_group([TypeHypothesis("line-like", "half", HALF, (0,), {}, 5.0)], FragmentGraph([], []), 1), [])
    def test_flat_scene_tries_every_type(self): H = flat_types(OB(RING), 1); self.assertGreaterEqual(len({h.type for h in H}), 4); self.assertEqual([h.score for h in H], sorted((h.score for h in H), reverse=True))
    def test_flat_fragments_proposes_both_models(self):
        ob = OB(DISC); g = build_graph(extract_fragments(ob)); a, b = type_hypotheses(g, ob, hierarchical=True), type_hypotheses(g, ob, hierarchical=False); self.assertGreater(len(b), len(a))
    def test_literals(self):
        L = literals([TypeHypothesis("line-like", "half", HALF, (0,)), TypeHypothesis("circle-like", "disc", DISC, (1,)), TypeHypothesis("circle-like", "disc", DISC, (1,))]); self.assertEqual(len(L), 3); P = torch.rand(200, 2, generator=G(1)) * 6 - 3; self.assertGreater(float((predict(L[0], P) != predict(L[1], P)).float().mean()), 0.97)
    def test_loop_is_bounded_and_logged(self):
        ob = OB(AND(DISC, OCC)); inf = infer_types(ob, 1, iters=1); self.assertLessEqual(len(inf.log), 2); self.assertEqual(inf.log[0]["action"], "init"); self.assertIn(inf.log[-1]["action"], ("init", "commit", "rollback")); self.assertEqual(len(infer_types(ob, 1, iters=0).log), 1)
        if len(inf.log) == 2 and inf.log[1]["action"] == "commit": self.assertGreater(inf.log[1]["score"], inf.log[0]["score"])
    def test_reassign(self):
        ob = OB(BNode("xor", children=(disc_atom(-1.6, 1.3, 0.9), line_atom(0.0, 0.8)))); inf = infer_types(ob, 1); lab = reassign(ob, inf.selected); self.assertEqual(len(lab), len(ob.points)); self.assertGreaterEqual(len(torch.unique(lab[lab >= 0])), 2); self.assertTrue(torch.equal(typed_labels(ob), lab))
        self.assertTrue(bool((reassign(ob, []) == -1).all()))
    def test_stats(self): st = infer_types(OB(RING), 1).stats; self.assertEqual(set(st), {"fragments", "groups", "edges", "types_tested", "seconds"}); self.assertLess(st["types_tested"], 80)
    def test_deterministic(self):
        ob = OB(AND(RING, OCC)); f = lambda: [(h.type, h.program.param_vector()) for h in infer_types(ob, 3, seed=1).selected]; a, b = f(), f(); self.assertEqual([t for t, _ in a], [t for t, _ in b])
        for (_, p), (_, q) in zip(a, b): self.assertTrue(all(abs(u - v) < 1e-9 for u, v in zip(p, q)))
    def test_no_learned_model_no_world(self):
        s = inspect.getsource(FR)
        for w in ("deep_world", "deep_primitive_world", "deep_primitive_eval", "deep_fragments_exp", "torch.nn", "neural"): self.assertNotIn(w, s)


class OwnershipTests(unittest.TestCase):
    def test_occluded_not_absent(self):
        ob = OB(AND(DISC, OCC)); o = ownership(DISC, ob, [OCC]); self.assertIsInstance(o, OcclusionOwnership); self.assertEqual(o.owner, 0); self.assertGreater(o.fractions["occluded"], 0.2); self.assertLess(o.absent, 0.1); self.assertAlmostEqual(sum(o.fractions.values()), 1.0, places=6); self.assertEqual(set(o.fractions), set(OWNERSHIP))
    def test_absent_without_owner(self):
        ob = OB(AND(DISC, OCC)); o = ownership(DISC, ob, []); self.assertIsNone(o.owner); self.assertGreater(o.absent, 0.2); self.assertEqual(o.fractions["occluded"], 0.0); self.assertIsNone(ownership(DISC, ob, [disc_atom(-2.5, -2.5, 0.3)]).owner)
    def test_fully_visible(self): o = ownership(DISC, OB(DISC), [OCC]); self.assertGreater(o.fractions["observed"], 0.85); self.assertEqual(o.fractions["occluded"], 0.0); self.assertLess(o.absent, 0.1)
    def test_domain_limit(self):
        p = disc_atom(2.6, 0.0, 1.2); X, y = scene(p); o = ownership(p, observe(X, y), []); self.assertGreater(o.fractions["domain"], 0.3); self.assertLess(o.absent, 0.15)
    def test_wrong_primitive_is_absent(self): o = ownership(disc_atom(-2.0, 2.0, 0.6), OB(AND(DISC, OCC)), [OCC]); self.assertEqual(o.fractions["observed"], 0.0); self.assertIsNone(o.owner)
    def test_negative_evidence_modes(self):
        ob = OB(AND(DISC, OCC)); ids = torch.arange(len(ob.points)); a = signals_of(DISC, ids, ob, others=[OCC], negative="ownership")["absent"]; b = signals_of(DISC, ids, ob, others=[OCC], negative="naive")["absent"]; c = signals_of(DISC, ids, ob, others=[OCC], negative="none")["absent"]
        self.assertEqual(c, 0.0); self.assertGreater(b, a + 0.2)                                              # counting the hidden part as absent would penalise a correct primitive
    def test_empty_render(self): self.assertEqual(ownership(disc_atom(50.0, 50.0, 0.2), OB(DISC)).labels, [])


class BenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.sc = FX.scene_at(0, "ring", 0.4, 0.0, tune=True)
    def test_scenes_unused_before(self):
        from core.deep_partial_exp import masked_scene
        self.assertEqual(FX.scene_at(3, "disc", 0.2, 0.0).seed, masked_scene(103, "disc", 0.2, 0.0).seed); used = {masked_scene(i, k, 0.0, 0.0).seed for i in range(32) for k in FX.KINDS}; self.assertFalse({FX.scene_at(i, k, 0.0, 0.0).seed for i in range(32) for k in FX.KINDS} & used)
    def test_group_rows(self):
        rows = FX.group_rows(self.sc); self.assertEqual([r["grouping"] for r in rows], list(GROUPINGS) + ["typed"])
        for r in rows: self.assertTrue(r["f1"] is None or 0 <= r["f1"] <= 1)
    def test_type_row(self):
        for v in FX.VARIANTS:
            r = FX.type_row(self.sc, v)
            for k in ("exact_1", "exact_3", "type_1", "type_3", "equivalent_1", "equivalent_3", "functional_1", "functional_3", "predicted", "correct"): self.assertIn(k, r, v)
            self.assertTrue(r["exact_3"] >= r["exact_1"]); self.assertTrue(r["type_3"] >= r["type_1"]); self.assertTrue(r["equivalent_3"] >= r["exact_3"])
    def test_equivalent_representation_counts(self):
        self.assertTrue(FX._equivalent(RING, [disc_atom(-0.3, 0.2, 0.7), disc_atom(-0.3, 0.2, 1.6)])); self.assertFalse(FX._equivalent(RING, [disc_atom(-0.3, 0.2, 1.6)])); self.assertTrue(FX._equivalent(BAND, [line_atom(1.0, -0.6), line_atom(1.0, 0.6)]))
    def test_oracle_row(self):
        r = FX.oracle_row(self.sc); self.assertEqual({"TT", "TI", "TI_type", "ET", "EI"} - set(r), set()); self.assertGreater(r["target_points"], 4)
    def test_restrict_keeps_only_target(self):
        ob = observe(self.sc.X, self.sc.y); _, atom = FX.true_sources(ob.points, [self.sc.target, self.sc.occluder]); sub = FX.restrict(ob, atom == 0); self.assertEqual(len(sub.points), int((atom == 0).sum())); self.assertTrue(bool((FX.true_sources(sub.points, [self.sc.target, self.sc.occluder])[1] == 0).all()))
    def test_own_row(self):
        r = FX.own_row(self.sc)
        for k in ("hidden_as_occluded", "hidden_as_absent", "visible_as_observed", "owner_found", "owner_is_occluder"): self.assertIn(k, r)
        self.assertIsNone(FX.own_row(FX.scene_at(0, "ring", 0.0, 0.0, tune=True))["hidden_as_occluded"])
    def test_sector_row(self):
        r = FX.sector_row(FX.scene_at(0, "sector", 0.0, 0.0, tune=True)); self.assertIn(r["stage"], ("ok", "rays", "vertex_relation", "vertex_position", "angles_or_ranking")); self.assertLessEqual(r["rays_found"], 2); self.assertIsInstance(r["confused_with_disc"], bool)
    def test_active_row(self):
        sc = FX.scene_at(0, "disc", 0.2, 0.05, tune=True)
        for p in ("none", "random", "type_disagreement"): self.assertEqual(FX.active_row(sc, p)["policy"], p)
    def test_subunits(self):
        self.assertIn("deepft", DX.SUBUNITS); a = DX.subunit_ids("deepft", "0"); self.assertEqual(len(a), 6 * 5 + 6 * 2); self.assertEqual(len(set(a)), len(a)); c = DX.subunit_ids("deepft", "600"); self.assertEqual(len(c), 4 * 2 * len(FX.CHAIN_ARMS)); self.assertEqual(len({x[0] for x in FX.CHAIN_ARMS}), len(FX.CHAIN_ARMS))
    def test_chain_tasks_new(self):
        t = DX.down_task(600, 3, "rich"); self.assertEqual(t.split, "dev"); used = {DX.down_task(k, 3, "rich").seed for k in list(range(16)) + list(range(100, 116)) + list(range(200, 216)) + list(range(300, 308)) + list(range(400, 416)) + list(range(500, 516))}; self.assertNotIn(t.seed, used)
    def test_type_free_arms_use_no_truth(self):
        s = inspect.getsource(FX.run_variant); self.assertNotIn("truth", s); self.assertNotIn("target", s); self.assertNotIn("atoms", s); c = inspect.getsource(FX.chain); self.assertIn('DeepSearch("factorized"', c)
    def test_sealed_and_pipeline_untouched(self):
        for m in (FR, FX, FRP): s = inspect.getsource(m); self.assertNotIn("sealed_task", s); self.assertNotIn('"sealed"', s); self.assertNotIn('"test"', s)
        s = inspect.getsource(DS); self.assertNotIn("deep_fragments", s); self.assertIsNone(inspect.signature(DS.DeepSearch.__init__).parameters["validated"].default)
    def test_summarize(self):
        rows = []
        for kd in ("disc", "sector"):
            for ms in (0.0, 0.4):
                sc = FX.scene_at(0, kd, ms, 0.0, tune=True); tag = {"scene": f"0_{kd}"}; rows += [{"experiment": "group", **tag, **r} for r in FX.group_rows(sc)] + [{"experiment": "type", **tag, **FX.type_row(sc, v)} for v in ("hier", "flat_scene")] + [{"experiment": "oracle", **tag, **FX.oracle_row(sc)}, {"experiment": "own", **tag, **FX.own_row(sc)}]
                if kd == "sector": rows.append({"experiment": "sector", **tag, **FX.sector_row(sc)})
        rows += [{"experiment": "active", "scene": "0_disc", **FX.active_row(FX.scene_at(0, "disc", 0.2, 0.05, tune=True), p)} for p in ("none", "random")]
        rows += [{"experiment": "chain", "arm": a, "task_id": str(i), "depth": 2, "type_recall": 0.5, "parameter_recovery": 0.5, "primitive_equivalence": 0.5, "candidate_functional": True, "final_095": True, "final_full": 0.9 + 0.01 * (a == "hier_k3"), "leaves": [{"type": "norm.le", "type_hit": True, "recovered": False, "equivalent": True}]} for a in ("reference", "hier_k3") for i in range(4)]
        S = FRP.summarize(rows); self.assertIn("typed", S["grouping"]); self.assertIn("hier", S["type"]); self.assertIn("disc", S["confusion"]); self.assertIn("all", S["oracle"]); self.assertIn("0", S["sector"]); self.assertAlmostEqual(S["chain_paired"]["hier_k3 − reference"]["final_full"]["diff"], 0.01); FRP.show(S)


if __name__ == "__main__":
    unittest.main()

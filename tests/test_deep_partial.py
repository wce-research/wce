import inspect
import math
import unittest

import torch

import core.deep_experiments as DX
import core.deep_partial as PO
import core.deep_partial_exp as PX
import core.deep_po_report as PR
import core.deep_search as DS
from core.boundary_program import BNode, predict
from core.deep_fit import DT
from core.deep_partial import (IDENT, CompositeNode, Hypothesis, LatentBoundary, ObservedBoundary, ParameterRegion, PrimitiveNode, arc_angle, build, circle_components, circle_geometric, circle_kasa, circle_normals,
                               circle_region, circle_sensitivity, circle_taubin, complete, concentric_fit, eval_forced, exchange, geometry, global_score, infer, joint_select, leaf_paths, line_components, line_region,
                               line_tls, link_fragments, modes, observe, occlusion_score, orient, parallel_fit, parameter_queries, refit_visible, render, rendering_errors, replace_at, segment_profile, visible_mask,
                               visible_score)
from core.deep_primitives2 import disc_atom, line_atom

G = lambda s=0: torch.Generator().manual_seed(s)
T = lambda c, ch: BNode("translate", params=c, children=(ch,))
DISC = disc_atom(0.2, 0.0, 1.2)
HALF = line_atom(0.3, 0.1)
RING = T((-0.3, 0.2), BNode("pred", "norm", "in", (0.7, 1.6)))
BAND = BNode("pred", "dot", "in", (1.0, -0.6, 0.6))
SECTOR = T((-0.5, 0.0), BNode("pred", "angle", "in", (0.4, 0.8)))
OCC = line_atom(0.0, 0.3)                                   # garde x ≤ 0,3 : cache la partie droite de DISC
AND = lambda a, b: BNode("and", children=(a, b))
SK = lambda p: p.skeleton()


def scene(prog, n=700, rate=0.0, s=0):
    X = torch.rand(n, 2, generator=G(s)) * 6 - 3; y = predict(prog, X); f = torch.rand(n, generator=G(s + 1)) < rate; return X, torch.where(f, 1 - y, y)


def arc(c=(0.2, 0.0), r=1.2, a0=0.0, a1=math.pi / 2, n=40, noise=0.0, s=0):
    t = torch.linspace(a0, a1, n, dtype=DT); return torch.tensor(c, dtype=DT) + r * torch.stack([torch.cos(t), torch.sin(t)], 1) + noise * torch.randn(n, 2, generator=G(s), dtype=DT)


class EstimatorTests(unittest.TestCase):
    def test_exact_on_clean_arc(self):
        S = arc(a1=math.pi / 3)
        for f in (circle_kasa, circle_taubin):
            c, r = f(S); self.assertAlmostEqual(r, 1.2, places=5); self.assertLess(float((c - torch.tensor([0.2, 0.0], dtype=DT)).norm()), 1e-5)
    def test_normals_intersection(self):
        S = arc(a1=math.pi / 3); N = (S - torch.tensor([0.2, 0.0], dtype=DT)) / 1.2; c, r = circle_normals(S, N); self.assertAlmostEqual(r, 1.2, places=5); self.assertLess(float((c - torch.tensor([0.2, 0.0], dtype=DT)).norm()), 1e-5)
    def test_normals_reject_parallel(self): S = torch.stack([torch.linspace(0, 1, 10, dtype=DT), torch.zeros(10, dtype=DT)], 1); N = torch.tensor([[0.0, 1.0]], dtype=DT).expand(10, 2); self.assertIsNone(circle_normals(S, N))
    def test_taubin_less_biased_than_kasa_on_short_noisy_arc(self):
        ek = et = 0.0
        for s in range(20): S = arc(a1=math.pi / 3, n=30, noise=0.04, s=s); ek += abs(circle_kasa(S)[1] - 1.2); et += abs(circle_taubin(S)[1] - 1.2)
        self.assertLess(et, ek)
    def test_kasa_biased_small(self): self.assertLess(sum(circle_kasa(arc(a1=math.pi / 3, n=30, noise=0.05, s=s))[1] for s in range(20)) / 20, 1.2)
    def test_degenerate_inputs(self): self.assertIsNone(circle_kasa(arc(n=2))); self.assertIsNone(circle_taubin(arc(n=2))); self.assertIsNone(line_tls(arc(n=1))); self.assertIsNone(concentric_fit(arc(n=1), arc(n=1))); self.assertIsNone(parallel_fit(arc(n=1), arc(n=5)))
    def test_geometric_from_perturbed_start(self): c, r, rms = circle_geometric(arc(a1=math.pi), (0.5, 0.3), 0.9); self.assertAlmostEqual(r, 1.2, places=3); self.assertLess(rms, 1e-3)
    def test_geometric_fixed_radius(self): c, r, _ = circle_geometric(arc(a1=math.pi), (0.4, 0.2), 1.2, fix_radius=True); self.assertEqual(r, 1.2); self.assertLess(float((c - torch.tensor([0.2, 0.0], dtype=DT)).norm()), 1e-3)
    def test_geometric_fixed_center(self): c, r, _ = circle_geometric(arc(), (0.2, 0.0), 0.5, fix_center=True); self.assertEqual(c.tolist(), [0.2, 0.0]); self.assertAlmostEqual(r, 1.2, places=4)
    def test_concentric_shared_center_helps_short_arc(self):
        big = arc(r=1.6, a1=math.pi, n=60, noise=0.03, s=1); small = arc(r=0.7, a1=0.5, n=8, noise=0.03, s=2); c, r1, r2 = concentric_fit(small, big); alone = circle_taubin(small)
        self.assertLess(abs(r1 - 0.7), 0.1); self.assertLess(abs(r2 - 1.6), 0.1); self.assertLess(float((c - torch.tensor([0.2, 0.0], dtype=DT)).norm()), float((alone[0] - torch.tensor([0.2, 0.0], dtype=DT)).norm()) + 1e-9)
    def test_line_tls(self):
        t = torch.linspace(-1, 1, 20, dtype=DT); n0 = torch.tensor([math.cos(0.3), math.sin(0.3)], dtype=DT); tg = torch.tensor([-math.sin(0.3), math.cos(0.3)], dtype=DT); n, b, rms = line_tls(0.1 * n0 + t[:, None] * tg)
        self.assertAlmostEqual(abs(float(n @ n0)), 1.0, places=6); self.assertAlmostEqual(abs(b), 0.1, places=6); self.assertLess(rms, 1e-6)
    def test_parallel_shared_direction(self):
        tg = torch.tensor([0.0, 1.0], dtype=DT); a = torch.tensor([0.5, 0.0], dtype=DT) + torch.linspace(-2, 2, 40, dtype=DT)[:, None] * tg; b = torch.tensor([-0.7, 0.0], dtype=DT) + torch.linspace(0, 0.3, 4, dtype=DT)[:, None] * tg; n, b1, b2 = parallel_fit(a, b)
        self.assertAlmostEqual(abs(float(n[0])), 1.0, places=6); self.assertAlmostEqual(abs(b1 - b2), 1.2, places=6)
    def test_modes(self): v = torch.cat([torch.full((10,), 0.7), torch.full((6,), 1.6), torch.tensor([2.9])]); m = modes(v, 0.0, 3.5, 0.15, 3); self.assertEqual(len(m), 2); self.assertAlmostEqual(m[0], 0.7, places=5); self.assertAlmostEqual(m[1], 1.6, places=5); self.assertEqual(modes(torch.zeros(0), 0, 1, 0.1), [])


class RepresentationTests(unittest.TestCase):
    def test_fragments_connected(self): M = torch.cat([arc(n=20, a1=1.0), arc(c=(-2.0, -2.0), r=0.5, n=10, a1=1.0)]); f = link_fragments(M); self.assertEqual(int(f.max()) + 1, 2); self.assertEqual(len(set(f[:20].tolist())), 1); self.assertEqual(len(link_fragments(torch.zeros(0, 2, dtype=DT))), 0)
    def test_observe(self):
        X, y = scene(DISC); ob = observe(X, y); self.assertIsInstance(ob, ObservedBoundary); self.assertEqual(len(ob.points), len(ob.normals)); self.assertEqual(len(ob.points), len(ob.fragment)); self.assertGreaterEqual(ob.n_fragments, 1); self.assertGreater(ob.extent(int(ob.fragment.mode().values)), 1.0)
        self.assertLess(float(((ob.points - torch.tensor([0.2, 0.0], dtype=DT)).norm(dim=1) - 1.2).abs().median()), 0.15)
    def test_observed_is_partial_under_mask(self): X, y = scene(DISC); Xm, ym = scene(AND(DISC, OCC)); a, b = observe(X, y), observe(Xm, ym); self.assertFalse(bool((b.points[:, 0] > 0.45).any())); self.assertTrue(bool((a.points[:, 0] > 0.8).any()))
    def test_geometry_and_build(self):
        self.assertEqual(geometry(DISC), {"kind": "circle", "center": (0.2, 0.0), "radii": (1.2,)}); self.assertEqual(geometry(RING)["radii"], (0.7, 1.6)); self.assertEqual(geometry(BAND), {"kind": "line", "angle": 1.0, "offsets": (-0.6, 0.6)}); self.assertEqual(geometry(SECTOR)["half_width"], 0.8)
        self.assertEqual(repr(build(SK(RING), (-0.3, 0.2), (0.7, 1.6))), repr(RING)); self.assertEqual(repr(build(SK(HALF), None, HALF.params)), repr(HALF))
        n = T((0.0, 0.0), BNode("not", children=(BNode("pred", "angle", "in", (0.1, 0.5)),))); self.assertEqual(geometry(n)["kind"], "sector"); self.assertEqual(repr(build(SK(n), (0.0, 0.0), (0.1, 0.5))), repr(n))
    def test_orient_picks_polarity(self): X, y = scene(HALF); flipped = line_atom(0.3 + math.pi, -0.1); self.assertGreater(float((predict(orient(flipped, X, y), X) == y).float().mean()), 0.97); self.assertIs(orient(DISC, X, y), DISC)
    def test_render_inside_box(self):
        for p, n in ((DISC, 96), (RING, 192), (HALF, 96), (BAND, 192), (SECTOR, 192)): P = render(p); self.assertLessEqual(len(P), n); self.assertGreater(len(P), 10); self.assertLessEqual(float(P.abs().max()), 3.0 + 1e-9)
        self.assertLess(float(((render(DISC) - torch.tensor([0.2, 0.0], dtype=DT)).norm(dim=1) - 1.2).abs().max()), 1e-9)
    def test_completion_marks_hidden_part(self):
        X, y = scene(AND(DISC, OCC)); lb = complete(DISC, observe(X, y)); self.assertIsInstance(lb, LatentBoundary); self.assertGreater(lb.observed_fraction, 0.3); self.assertLess(lb.observed_fraction, 0.85); self.assertGreater(float(lb.completion[:, 0].mean()), 0.3)
        X2, y2 = scene(DISC); self.assertGreater(complete(DISC, observe(X2, y2)).observed_fraction, lb.observed_fraction)


class ScoreTests(unittest.TestCase):
    def setUp(self): self.X, self.y = scene(AND(DISC, OCC), rate=0.02); self.ob = observe(self.X, self.y)
    def test_visible_score_tolerates_occlusion(self):
        X, y = scene(DISC, rate=0.02); full = observe(X, y); zf, vf = visible_score(DISC, full.pairs); zm, vm = visible_score(DISC, self.ob.pairs); self.assertGreater(vf, vm); self.assertGreater(zm, 5.0); self.assertGreater(zm / global_score(DISC, self.ob.pairs), zf / global_score(DISC, full.pairs))
    def test_visible_score_zero_without_boundary(self): self.assertEqual(visible_score(disc_atom(-2.5, -2.5, 0.3), self.ob.pairs)[0], 0.0); self.assertEqual(segment_profile(disc_atom(9.0, 9.0, 0.1), self.ob.pairs), [])
    def test_segment_profile(self): seg = segment_profile(DISC, self.ob.pairs); self.assertTrue(any(s["seen"] for s in seg)); self.assertTrue(any(not s["seen"] for s in seg)); self.assertTrue(all(s["k"] <= s["n"] for s in seg))
    def test_visible_mask_and(self):
        P = torch.tensor([[-1.0, 0.0], [1.3, 0.0]], dtype=DT); prog = AND(DISC, OCC); self.assertEqual(visible_mask(prog, (0,), P).tolist(), [True, False]); self.assertEqual(visible_mask(prog, (1,), torch.tensor([[0.3, 0.5], [0.3, 2.5]], dtype=DT)).tolist(), [True, False])
    def test_visible_mask_or_xor_not(self):
        P = torch.tensor([[-1.0, 0.0], [1.3, 0.0]], dtype=DT); self.assertEqual(visible_mask(BNode("or", children=(DISC, OCC)), (0,), P).tolist(), [False, True]); self.assertEqual(visible_mask(BNode("xor", children=(DISC, OCC)), (0,), P).tolist(), [True, True])
        self.assertEqual(visible_mask(BNode("not", children=(AND(DISC, OCC),)), (0, 0), P).tolist(), [True, False])
    def test_eval_forced_matches_predict(self):
        prog = BNode("xor", children=(AND(DISC, OCC), RING)); P = torch.rand(300, 2, generator=G(3)) * 6 - 3; t = predict(prog, P).long(); d = predict(DISC, P).long(); f1, f0 = eval_forced(prog, P, (0, 0), True), eval_forced(prog, P, (0, 0), False)
        self.assertTrue(torch.equal(torch.where(d == 1, f1, f0), t)); self.assertTrue(torch.equal(eval_forced(prog, P, (9, 9), True), t))
    def test_leaf_paths_and_replace(self):
        prog = BNode("xor", children=(AND(DISC, OCC), RING)); lp = leaf_paths(prog); self.assertEqual([p for p, _ in lp], [(0, 0), (0, 1), (1,)]); q = replace_at(prog, (0, 1), HALF); self.assertEqual(repr(dict(leaf_paths(q))[(0, 1)]), repr(HALF)); self.assertEqual(repr(dict(leaf_paths(q))[(1,)]), repr(RING))
        self.assertEqual(leaf_paths(DISC), [((), DISC)]); self.assertIs(replace_at(DISC, (), HALF), HALF)
    def test_occlusion_score_separates_hidden_from_visible(self):
        o = occlusion_score(AND(DISC, OCC), (0,), self.ob.pairs); self.assertGreater(o["z_visible"], 5.0); self.assertLess(o["z_hidden"], 1.0); self.assertGreater(o["hidden_pairs"], 0); self.assertLess(o["visible_fraction"], 1.0)
    def test_wrong_primitive_differs_from_hidden_primitive(self):
        wrong = disc_atom(-1.5, 1.5, 0.6); o = occlusion_score(AND(wrong, OCC), (0,), self.ob.pairs); self.assertGreater(o["visible_pairs"], 0); self.assertLess(o["z_visible"], 3.0)          # visible mais sans contraste : fausse, pas cachée
    def test_rendering_prefers_true_parameters(self): self.assertLess(rendering_errors(AND(DISC, OCC), self.X, self.y), rendering_errors(AND(disc_atom(0.2, 0.0, 0.8), OCC), self.X, self.y))


class RegionTests(unittest.TestCase):
    def test_region_narrow_on_full_circle(self): r = circle_region(arc(a1=2 * math.pi, n=80, noise=0.03), (0.2, 0.0), 1.2); self.assertTrue(r.contains(cx=0.2, cy=0.0, r=1.2)); self.assertEqual(set(r.identifiability().values()), {IDENT[0]}); self.assertLess(r.widths()["r"], 0.5)
    def test_region_wide_on_short_arc(self):
        S = arc(a1=0.5, n=20, noise=0.03); short = circle_region(S, (0.2, 0.0), 1.2); full = circle_region(arc(a1=2 * math.pi, n=80, noise=0.03), (0.2, 0.0), 1.2); self.assertGreater(short.widths()["r"], 2 * full.widths()["r"]); self.assertTrue(short.contains(r=1.2)); self.assertIn(short.identifiability()["r"], IDENT[1:])
        self.assertGreater(len(short.hypotheses), len(full.hypotheses)); self.assertGreater(short.posterior["r"][1], full.posterior["r"][1])
    def test_region_weights(self): r = circle_region(arc(a1=1.0, n=20, noise=0.03), (0.2, 0.0), 1.2); self.assertAlmostEqual(sum(r.weights), 1.0, places=6); self.assertEqual(len(r.weights), len(r.hypotheses))
    def test_sensitivity_flat_direction_on_short_arc(self):
        c = torch.tensor([0.2, 0.0], dtype=DT); a, b = circle_sensitivity(arc(a1=0.4), c, 1.2), circle_sensitivity(arc(a1=2 * math.pi, n=80), c, 1.2); self.assertGreater(a["condition"], 50 * b["condition"]); self.assertGreater(abs(a["corr_radius_center"]), 0.95); self.assertLess(abs(b["corr_radius_center"]), 0.3)
        self.assertAlmostEqual(a["arc_angle"], 0.4, places=5); self.assertGreater(b["arc_angle"], 6.0)
    def test_arc_angle(self): c = torch.tensor([0.2, 0.0], dtype=DT); self.assertAlmostEqual(arc_angle(arc(a0=-0.5, a1=0.5), c), 1.0, places=5); self.assertEqual(arc_angle(arc(n=1), c), 0.0)
    def test_line_region(self):
        tg = torch.tensor([0.0, 1.0], dtype=DT); n = torch.tensor([1.0, 0.0], dtype=DT); long = 0.5 * n + torch.linspace(-2, 2, 40, dtype=DT)[:, None] * tg; short = 0.5 * n + torch.linspace(0, 0.3, 6, dtype=DT)[:, None] * tg; a, b = line_region(long, n, 0.5), line_region(short, n, 0.5)
        self.assertLess(a.widths()["angle"], b.widths()["angle"]); self.assertTrue(a.contains(angle=0.0, offset=0.5)); self.assertEqual(a.identifiability()["angle"], IDENT[0]); self.assertIn(b.identifiability()["angle"], IDENT[1:])
    def test_identifiability_labels(self):
        r = ParameterRegion({"cx": (0.0, 0.5), "r": (0.5, 2.0), "angle": (0.0, 3.0)}, []); self.assertEqual(r.identifiability(), {"cx": IDENT[0], "r": IDENT[1], "angle": IDENT[2]}); self.assertTrue(r.contains(cx=0.2, unknown=9.0)); self.assertFalse(r.contains(r=2.5))


class InferenceTests(unittest.TestCase):
    def test_components(self):
        X, y = scene(BNode("xor", children=(DISC, HALF))); ob = observe(X, y); cs = circle_components(ob); ls = line_components(ob)
        self.assertTrue(any(math.hypot(c.params[0] - 0.2, c.params[1]) < 0.3 and abs(c.params[2] - 1.2) < 0.25 for c in cs)); self.assertTrue(any(abs(math.cos(c.params[0] - 0.3)) > 0.97 for c in ls)); self.assertTrue(all(len(c.inliers) >= 4 for c in cs + ls))
    def test_infer_each_type_unmasked(self):
        for prog in (DISC, HALF, RING, BAND, SECTOR):
            X, y = scene(prog, n=900); H = infer(SK(prog), observe(X, y), 5); P = torch.rand(2000, 2, generator=G(9)) * 6 - 3; t = predict(prog, P); self.assertTrue(H, prog); self.assertGreater(max(float((predict(h.program, P) == t).float().mean()) for h in H), 0.93, prog)
            self.assertTrue(all(repr(SK(h.program)) == repr(SK(prog)) for h in H)); self.assertEqual([h.score for h in H], sorted((h.score for h in H), reverse=True))
    def test_infer_masked_disc(self):
        X, y = scene(AND(DISC, OCC)); H = infer(SK(DISC), observe(X, y), 5); self.assertTrue(any(PX.recovered(DISC, h.program) for h in H)); self.assertLess(H[0].visible_fraction, 1.0); self.assertIsInstance(H[0], Hypothesis); self.assertIsNotNone(H[0].region)
    def test_topk_distinct_and_bounded(self):
        X, y = scene(AND(DISC, OCC)); ob = observe(X, y); H = infer(SK(DISC), ob, 3); self.assertLessEqual(len(H), 3); P = X.float()
        for i, a in enumerate(H):
            for b in H[i + 1:]: self.assertGreater(float((predict(a.program, P) != predict(b.program, P)).float().mean()), 0.02)
        self.assertLessEqual(len(infer(SK(DISC), ob, 1)), 1)
    def test_given_center_recovers_radius(self): X, y = scene(AND(DISC, OCC)); H = infer(SK(DISC), observe(X, y), 3, given={"center": (0.2, 0.0)}); self.assertTrue(all(geometry(h.program)["center"] == (0.2, 0.0) for h in H)); self.assertTrue(any(abs(geometry(h.program)["radii"][0] - 1.2) < 0.15 for h in H))
    def test_given_size_keeps_radius(self): X, y = scene(AND(DISC, OCC)); H = infer(SK(DISC), observe(X, y), 3, given={"size": (1.2,)}); self.assertTrue(H); self.assertTrue(all(geometry(h.program)["radii"] == (1.2,) for h in H))
    def test_given_orientation(self): X, y = scene(BAND); H = infer(SK(BAND), observe(X, y), 3, given={"orientation": 1.0}); self.assertTrue(all(geometry(h.program)["angle"] == 1.0 for h in H)); self.assertTrue(any(PX.recovered(BAND, h.program) for h in H))
    def test_joint_uses_shared_constraint(self):
        X, y = scene(RING, n=900); ob = observe(X, y); self.assertIn("concentric_joint", {h.source for h in infer(SK(RING), ob, 5, joint=True)}); self.assertNotIn("concentric_joint", {h.source for h in infer(SK(RING), ob, 5, joint=False)})
    def test_deterministic(self):
        # Même graine → mêmes hypothèses, à la précision machine près : l'algèbre linéaire multi-fil (lstsq, eigh) peut différer de 1e-16 d'un appel à l'autre.
        X, y = scene(AND(RING, OCC)); f = lambda: [h.program.param_vector() for h in infer(SK(RING), observe(X, y), 5, seed=3)]; a, b = f(), f(); self.assertEqual(len(a), len(b)); self.assertTrue(a)
        for p, q in zip(a, b): self.assertEqual(len(p), len(q)); self.assertTrue(all(abs(u - v) < 1e-9 for u, v in zip(p, q)))
    def test_no_boundary_no_hypothesis(self): X, _ = scene(DISC); self.assertEqual(infer(SK(DISC), observe(X, torch.zeros(len(X), dtype=torch.long)), 5), [])
    def test_no_world_or_eval_access(self):
        s = inspect.getsource(PO)
        for w in ("deep_world", "deep_primitive_world", "deep_primitive_eval", "deep_partial_exp"): self.assertNotIn(w, s)


class ExchangeTests(unittest.TestCase):
    def setUp(self): self.X, self.y = scene(AND(DISC, OCC), rate=0.02); self.ob = observe(self.X, self.y)
    def test_joint_select(self):
        best, e, n = joint_select([[disc_atom(0.2, 0.0, 0.8), DISC], [line_atom(0.0, -0.5), OCC]], AND(disc_atom(0.2, 0.0, 0.8), line_atom(0.0, -0.5)), self.X, self.y); self.assertEqual(repr(best), repr(AND(DISC, OCC))); self.assertEqual(e, rendering_errors(AND(DISC, OCC), self.X, self.y)); self.assertLessEqual(n, 8)
    def test_joint_select_bounded(self): _, _, n = joint_select([[DISC] * 50, [OCC] * 50], AND(DISC, OCC), self.X, self.y, max_combos=7); self.assertEqual(n, 7)
    def test_refit_visible(self):
        M = self.ob.points; r = refit_visible(disc_atom(0.3, 0.1, 1.1), M[(M - torch.tensor([0.2, 0.0], dtype=DT)).norm(dim=1) < 1.5]); self.assertIsNotNone(r); self.assertEqual(repr(SK(r)), repr(SK(DISC))); self.assertIsNone(refit_visible(DISC, M[:2])); self.assertIsNone(refit_visible(SECTOR, M))
    def test_exchange_improves_and_logs(self):
        start = AND(disc_atom(0.45, 0.2, 1.0), OCC); ex = exchange(start, self.ob, {repr(SK(DISC)): [DISC]}); self.assertLess(ex.errors_after, ex.errors_before); self.assertEqual(ex.errors_after, rendering_errors(ex.program, self.X, self.y)); self.assertTrue(any(l["action"] == "commit" for l in ex.log))
        for l in ex.log: self.assertLessEqual({"iteration", "path", "type", "proposals", "visible_fraction", "action", "errors_before", "errors_after"}, set(l)); self.assertLessEqual(l["errors_after"], l["errors_before"])
    def test_exchange_never_degrades(self):
        ex = exchange(AND(DISC, OCC), self.ob, {repr(SK(DISC)): [disc_atom(2.0, 2.0, 0.3)]}); self.assertLessEqual(ex.errors_after, ex.errors_before); self.assertFalse(any(l["kind"] == "alternative" for l in ex.log))
    def test_exchange_bounded(self): ex = exchange(AND(disc_atom(0.45, 0.2, 1.0), OCC), self.ob, iters=1); self.assertEqual(ex.iterations, 1); self.assertLessEqual(len(ex.log), 2); self.assertLessEqual(exchange(AND(DISC, OCC), self.ob, iters=5).iterations, 5)
    def test_exchange_structure_unchanged(self): ex = exchange(AND(disc_atom(0.45, 0.2, 1.0), OCC), self.ob); self.assertEqual(repr(SK(ex.program)), repr(SK(AND(DISC, OCC))))
    def test_nodes(self): c = CompositeNode(AND(DISC, OCC)); P = torch.tensor([[-1.0, 0.0], [1.3, 0.0]], dtype=DT); self.assertEqual(c.visibility((0,), P).tolist(), [True, False]); n = PrimitiveNode((0,), DISC); self.assertEqual(n.alternatives, []); self.assertIsNone(n.visible_fraction)
    def test_parameter_queries_target_disagreement(self):
        pool = torch.rand(500, 2, generator=G(4)) * 6 - 3; a, b = disc_atom(0.2, 0.0, 1.0), disc_atom(0.2, 0.0, 1.6); i = parameter_queries([a, b], pool, 10); d = (pool[i] - torch.tensor([0.2, 0.0])).norm(dim=1); self.assertEqual(len(i), 10); self.assertTrue(bool(((d > 1.0) & (d < 1.6)).all()))
        self.assertTrue(all(float((pool[i[u]] - pool[i[v]]).norm()) > 0.2 for u in range(10) for v in range(u + 1, 10)))
    def test_parameter_queries_skip_hidden(self):
        pool = torch.rand(800, 2, generator=G(4)) * 6 - 3; a, b = disc_atom(0.2, 0.0, 1.0), disc_atom(0.2, 0.0, 1.6); i = parameter_queries([a, b], pool, 10, context=(AND(a, OCC), (0,))); self.assertTrue(len(i) > 0); self.assertTrue(bool((pool[i][:, 0] <= 0.3).all()))
        self.assertEqual(len(parameter_queries([], pool, 5)), 0); self.assertEqual(len(parameter_queries([a, a], pool, 5)), 0)


class BenchTests(unittest.TestCase):
    def test_masked_scene_deterministic_and_paired(self):
        a, b = PX.masked_scene(1, "disc", 0.4, 0.0, tune=True), PX.masked_scene(1, "disc", 0.4, 0.0, tune=True); self.assertTrue(torch.equal(a.X, b.X)); self.assertTrue(torch.equal(a.y, b.y)); c = PX.masked_scene(1, "disc", 0.8, 0.08, tune=True)
        self.assertEqual(repr(a.target), repr(c.target)); self.assertTrue(torch.equal(a.X, c.X)); self.assertIsNone(PX.masked_scene(1, "disc", 0.0, 0.0, tune=True).occluder); self.assertNotEqual(a.seed, PX.masked_scene(1, "disc", 0.4, 0.0).seed)
    def test_mask_hides_requested_fraction(self):
        for kind in PX.KINDS:
            for m in (0.2, 0.6): self.assertAlmostEqual(PX.true_visible_fraction(PX.masked_scene(0, kind, m, 0.0, tune=True)), 1 - m, delta=0.12, msg=kind)
    def test_occluder_type_differs_from_target(self): self.assertEqual(geometry(PX.masked_scene(0, "ring", 0.4, 0.0, tune=True).occluder)["kind"], "line"); self.assertEqual(geometry(PX.masked_scene(0, "band", 0.4, 0.0, tune=True).occluder)["kind"], "circle")
    def test_noise_only_flips_labels(self): a, b = PX.masked_scene(2, "half", 0.4, 0.0, tune=True), PX.masked_scene(2, "half", 0.4, 0.08, tune=True); f = float((a.y != b.y).float().mean()); self.assertGreater(f, 0.02); self.assertLess(f, 0.16)
    def test_seeds_in_dev_range_and_unused(self):
        from core.deep_world import SPLIT_BASE
        s = PX.masked_scene(31, "sector", 0.8, 0.08).seed; self.assertGreater(s, SPLIT_BASE["dev"]); self.assertLess(s, SPLIT_BASE["dev"] + 350_000); self.assertGreaterEqual(PX.masked_scene(0, "half", 0.0, 0.0, tune=True).seed, PX.PO_TUNE)
    def test_param_errors(self):
        self.assertEqual(PX.param_errors(DISC, DISC), {"center": 0.0, "radius": 0.0}); e = PX.param_errors(RING, T((-0.3, 0.5), BNode("pred", "norm", "in", (0.8, 1.6)))); self.assertAlmostEqual(e["center"], 0.3); self.assertAlmostEqual(e["radius_in"], 0.1); self.assertAlmostEqual(e["radius_out"], 0.0)
        e = PX.param_errors(HALF, line_atom(0.3 + math.pi, -0.1)); self.assertAlmostEqual(e["orientation"], 0.0, places=6); self.assertAlmostEqual(e["offset"], 0.0, places=6); self.assertIn("width", PX.param_errors(BAND, BAND)); self.assertEqual(set(PX.param_errors(SECTOR, SECTOR)), {"center", "direction", "half_width"}); self.assertEqual(PX.param_errors(DISC, HALF), {})
    def test_given_of(self): self.assertEqual(PX.given_of(DISC, "center"), {"center": (0.2, 0.0)}); self.assertEqual(PX.given_of(HALF, "center"), {}); self.assertEqual(PX.given_of(HALF, "size"), {}); self.assertEqual(PX.given_of(BAND, "size"), {"size": (1.2,)}); self.assertEqual(PX.given_of(SECTOR, "orientation"), {"orientation": 0.4})
    def test_mask_row_fields(self):
        sc = PX.masked_scene(0, "disc", 0.4, 0.0, tune=True); r = PX.mask_row(sc, PX.MAIN, 0)
        for k in ("recovered_1", "recovered_3", "recovered_5", "functional_1", "errors", "visible_true", "visible_estimated", "observed_fraction", "n_hypotheses"): self.assertIn(k, r)
        self.assertTrue(r["recovered_5"] >= r["recovered_3"] >= r["recovered_1"]); self.assertTrue(PX.mask_row(PX.masked_scene(0, "half", 0.4, 0.0, tune=True), "given_center", 0)["not_applicable"])
    def test_all_methods_run(self):
        sc = PX.masked_scene(0, "ring", 0.4, 0.0, tune=True)
        for m in PX.METHODS: self.assertEqual(PX.mask_row(sc, m, 0)["method"], m)
        self.assertIn("sector", PX.mask_row(PX.masked_scene(0, "sector", 0.2, 0.0, tune=True), PX.MAIN, 0))
    def test_active_row(self):
        sc = PX.masked_scene(0, "disc", 0.4, 0.05, tune=True)
        for p, q in (("none", 0), ("random", 16), ("disagreement", 16), ("disagreement_ctx", 16)): r = PX.active_row(sc, p, 0); self.assertEqual(r["queries"], q); self.assertEqual(r["method"], f"active_{p}")
    def test_subunits(self):
        self.assertIn("deeppo", DX.SUBUNITS); a = DX.subunit_ids("deeppo", "0"); self.assertEqual(len(a), 6 * 5 * 2 + 6 * 2); self.assertEqual(len(set(a)), len(a)); c = DX.subunit_ids("deeppo", "500"); self.assertEqual(len(c), 4 * 2 * len(PX.CHAIN_ARMS)); self.assertEqual(len({x[0] for x in PX.CHAIN_ARMS}), len(PX.CHAIN_ARMS))
    def test_chain_tasks_new(self):
        t = DX.down_task(500, 3, "rich"); self.assertEqual(t.split, "dev"); used = {DX.down_task(k, 3, "rich").seed for k in list(range(16)) + list(range(100, 116)) + list(range(200, 216)) + list(range(300, 308)) + list(range(400, 416))}; self.assertNotIn(t.seed, used)
    def test_po_literals_keep_types(self):
        from core.deep_reachability import leaves
        t = DX.down_task(599, 2, "standard"); lits, per = PX.po_literals(t, 3); self.assertEqual(len(per), len(leaves(t.world.truth))); self.assertLessEqual({repr(l.skeleton()) for l in lits}, {repr(a.skeleton()) for a in leaves(t.world.truth)} | {repr(BNode("pred", "dot", "le"))}); self.assertEqual(len({repr(l) for l in lits}), len(lits))
    def test_sealed_and_pipeline_untouched(self):
        for m in (PO, PX, PR): s = inspect.getsource(m); self.assertNotIn("sealed_task", s); self.assertNotIn('"sealed"', s); self.assertNotIn('"test"', s)
        s = inspect.getsource(DS); self.assertNotIn("deep_partial", s); self.assertIsNone(inspect.signature(DS.DeepSearch.__init__).parameters["validated"].default)
    def test_summarize(self):
        rows = []
        for kd in ("disc", "sector"):
            for ms in (0.0, 0.4):
                sc = PX.masked_scene(0, kd, ms, 0.0, tune=True); rows += [{"experiment": "mask", "scene": f"0_{kd}", **PX.mask_row(sc, m, 0)} for m in (PX.MAIN, PX.BASELINE, "minimal_components", "given_center")]
        rows += [{"experiment": "active", "scene": "0_disc", **PX.active_row(PX.masked_scene(0, "disc", 0.4, 0.05, tune=True), p, 0)} for p in ("none", "random")]
        rows += [{"experiment": "chain", "arm": a, "task_id": str(i), "depth": 2, "primitive_equivalence": 0.5, "candidate_functional": True, "final_full": 0.9, "final_095": False, "loop_full": 0.95, "loop_095": True, "leaves": [{"type": "norm.le", "top1": True, "topk": True, "functional_k": 0.9}]} for a in ("reference", "po_type_k3") for i in range(4)]
        S = PR.summarize(rows); self.assertIn(PX.MAIN, S["by_method"]); self.assertIn("disc", S["visible_needed"]); self.assertIn("0", S["sector"]); self.assertIn("po_type_k3 − reference", S["chain_paired"]); self.assertEqual(S["chain_paired"]["po_type_k3 : loop − final"]["final_full"]["diff"], 0.05); PR.show(S)


if __name__ == "__main__":
    unittest.main()

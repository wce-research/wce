import inspect
import math
import unittest
try:
    from tests._source import code_only
except ImportError:
    from _source import code_only

import torch

import core.deep_experiments as DX
import core.deep_primitives2 as P2
import core.deep_robust_prim as RP
import core.deep_rp_report as RR
import core.deep_search as DS
from core.boundary_program import BNode, predict
from core.deep_fit import DT, FitCounter
from core.deep_primitive_eval import component_match, components
from core.deep_primitives import CompatiblePrimitives, GradientProposer, PrimitiveCandidate, atom_name, close_pairs, contrast, discover, separates
from core.deep_primitives2 import (CompatiblePrimitiveHypotheses, Evidence, GridProposer, HoughProposer, LocalRefiner, RansacProposer, RobustValidator, UnionProposer, _nms, borderline,
                                   boundary_elements, boundary_query_scores, boundary_segments, clean_midpoints, clean_pairs, disc_atom, discover_robust, hough_circles, hough_lines, is_kin, isolated_points,
                                   joint_candidates, joint_fit, line_atom, literals_of, load_config, local_refit, near_boundary, prune, sector_exclusive, split_pairs, stability)

G = lambda s=0: torch.Generator().manual_seed(s)
T = lambda c, ch: BNode("translate", params=c, children=(ch,))
DISC = disc_atom(0.2, 0.0, 1.2)
DISC2 = disc_atom(-1.6, 1.4, 0.9)
HALF = line_atom(0.3, 0.1)
RING = T((-0.3, 0.2), BNode("pred", "norm", "in", (0.7, 1.6)))
BAND = BNode("pred", "dot", "in", (1.0, -0.6, 0.6))
SECTOR = T((-0.5, 0.0), BNode("pred", "angle", "in", (0.4, 0.8)))
XOR = lambda a, b: BNode("xor", children=(a, b))


def scene(prog, n=600, rate=0.03, s=0):
    X = torch.rand(n, 2, generator=G(s)) * 6 - 3; y = predict(prog, X); f = torch.rand(n, generator=G(s + 1)) < rate; return X, torch.where(f, 1 - y, y)


def hits(prog_or_comps, target):
    cs = prog_or_comps if isinstance(prog_or_comps, list) else components(prog_or_comps); return any(component_match(c, d) for c in components(target) for d in cs)


class PairTests(unittest.TestCase):
    def test_split_is_a_partition(self):
        X, y = scene(DISC); p = close_pairs(X, y); a, b = split_pairs(p, 3); self.assertEqual(len(a) + len(b), len(p)); self.assertLess(abs(len(a) - len(b)), 2)
        key = lambda q: {tuple(r) for r in torch.cat([q.P, q.Q], 1).tolist()}; self.assertFalse(key(a) & key(b))
    def test_split_is_deterministic(self): X, y = scene(DISC); p = close_pairs(X, y); self.assertTrue(torch.equal(split_pairs(p, 1)[0].P, split_pairs(p, 1)[0].P)); self.assertFalse(torch.equal(split_pairs(p, 1)[0].P, split_pairs(p, 2)[0].P))
    def test_isolated_point_detected(self):
        X, y = scene(DISC, rate=0.0); i = int((X - torch.tensor([0.2, 0.0])).norm(dim=1).argmin()); y = y.clone(); y[i] = 1 - y[i]; self.assertTrue(bool(isolated_points(X, y)[i])); self.assertLess(int(isolated_points(X, y).sum()), 6)
    def test_clean_pairs_drop_noise_pairs(self):
        X, y = scene(DISC, rate=0.0); i = int((X - torch.tensor([0.2, 0.0])).norm(dim=1).argmin()); y2 = y.clone(); y2[i] = 1 - y2[i]; a, b = close_pairs(X, y2), clean_pairs(X, y2)
        self.assertGreater(float(a.opposite.sum()), float(b.opposite.sum())); self.assertFalse(bool(((b.P - X[i].to(DT)).norm(dim=1) < 1e-9).any()))
    def test_clean_pairs_keep_boundary(self): X, y = scene(DISC, rate=0.0); self.assertGreater(contrast(DISC, clean_pairs(X, y))["z"], 8.0)
    def test_clean_midpoints_near_boundary(self):
        X, y = scene(DISC, rate=0.05); M = clean_midpoints(X, y, close_pairs(X, y)); d = ((M - torch.tensor([0.2, 0.0], dtype=DT)).norm(dim=1) - 1.2).abs(); self.assertGreater(float((d < 0.3).float().mean()), 0.7)


class LocalGlobalTests(unittest.TestCase):
    def test_boundary_normals_perpendicular_to_line(self):
        X, y = scene(HALF, rate=0.0); M, N = boundary_elements(close_pairs(X, y)); n = torch.tensor([math.cos(0.3), math.sin(0.3)], dtype=DT); self.assertGreater(float((N @ n).abs().median()), 0.85); self.assertTrue(torch.allclose(N.norm(dim=1), torch.ones(len(N), dtype=DT), atol=1e-6))
    def test_boundary_elements_empty(self):
        X, y = scene(HALF, rate=0.0); p = close_pairs(X, torch.zeros_like(y)); M, N = boundary_elements(p); self.assertEqual(len(M), 0); self.assertEqual(hough_lines(M, N), []); self.assertEqual(hough_circles(M, N), [])
    def test_hough_finds_line(self): X, y = scene(HALF, rate=0.0); M, N = boundary_elements(close_pairs(X, y)); self.assertTrue(any(component_match(("line", a, b), ("line", 0.3, 0.1)) for a, b, _ in hough_lines(M, N)))
    def test_hough_finds_circle(self): X, y = scene(DISC, rate=0.0); M, N = boundary_elements(close_pairs(X, y)); self.assertTrue(any(component_match(("circle", cx, cy, r), ("circle", 0.2, 0.0, 1.2)) for cx, cy, r, _ in hough_circles(M, N)))
    def test_nms_keeps_distinct(self): self.assertEqual(_nms([1.0, 1.05, 3.0, 3.02], [5, 4, 3, 2], lambda a, b: abs(a - b) < 0.2, 5), [1.0, 3.0]); self.assertEqual(_nms([1.0, 2.0, 3.0], [1, 2, 3], lambda a, b: False, 2), [3.0, 2.0])
    def test_ransac_proposes_both_atoms(self):
        X, y = scene(XOR(DISC2, HALF)); cs = [d for c in RansacProposer().propose(X, y, close_pairs(X, y), 0) for d in components(c.program)]; self.assertTrue(hits(cs, DISC2)); self.assertTrue(hits(cs, HALF))
    def test_ransac_minimal_basis_only(self): X, y = scene(XOR(DISC2, HALF)); self.assertLessEqual({c.type for c in RansacProposer().propose(X, y, close_pairs(X, y), 0)}, {"dot.le", "norm.le"})
    def test_ransac_no_fit_cost(self): X, y = scene(DISC); c = FitCounter(); RansacProposer().propose(X, y, close_pairs(X, y), 0, c); self.assertEqual(c.fit_equivalents, 0.0)
    def test_ransac_too_few_points(self): X, y = scene(DISC, n=600); self.assertEqual(RansacProposer().propose(X, torch.zeros_like(y), close_pairs(X, torch.zeros_like(y)), 0), [])
    def test_proposers_are_deterministic(self):
        X, y = scene(DISC)
        for p in (RansacProposer(), HoughProposer(refine_steps=0), GridProposer(refine_steps=0)): self.assertEqual([c.params for c in p.propose(X, y, close_pairs(X, y), 4)], [c.params for c in p.propose(X, y, close_pairs(X, y), 4)])
    def test_grid_is_structured(self): X, y = scene(DISC); cs = GridProposer(top=5, refine_steps=0).propose(X, y, close_pairs(X, y)); self.assertEqual(len(cs), 10); self.assertTrue(any(hits(c.program, DISC) for c in cs))
    def test_union_concatenates(self): X, y = scene(DISC); p = close_pairs(X, y); a, b = RansacProposer(), GridProposer(refine_steps=0); self.assertEqual(len(UnionProposer([a, b]).propose(X, y, p)), len(a.propose(X, y, p)) + len(b.propose(X, y, p)))


class EstimationTests(unittest.TestCase):
    def test_local_refit_recovers_circle(self):
        X, y = scene(DISC, rate=0.0); M = clean_midpoints(X, y, close_pairs(X, y)); q = local_refit(disc_atom(0.4, 0.15, 1.05), M); self.assertTrue(hits(q, DISC)); self.assertLess(math.hypot(q.params[0] - 0.2, q.params[1]), 0.15)
    def test_local_refit_recovers_line(self): X, y = scene(HALF, rate=0.0); M = clean_midpoints(X, y, close_pairs(X, y)); q = local_refit(line_atom(0.45, 0.3), M); self.assertLess(abs(q.params[0] - 0.3), 0.08); self.assertLess(abs(q.params[1] - 0.1), 0.12)
    def test_local_refit_not_attracted_by_other_boundary(self):
        X, y = scene(XOR(DISC, DISC2), rate=0.0); M = clean_midpoints(X, y, close_pairs(X, y)); self.assertTrue(hits(local_refit(DISC, M), DISC)); self.assertTrue(hits(local_refit(DISC2, M), DISC2))
    def test_local_refit_unchanged_without_support(self): X, y = scene(DISC, rate=0.0); M = clean_midpoints(X, y, close_pairs(X, y)); far = disc_atom(2.6, 2.6, 0.3); self.assertIs(local_refit(far, M), far); self.assertIs(local_refit(DISC, M[:2]), DISC)
    def test_local_refit_ignores_rich_atoms(self): X, y = scene(RING, rate=0.0); M = clean_midpoints(X, y, close_pairs(X, y)); self.assertIs(local_refit(RING, M), RING)
    def test_local_refit_keeps_operator(self): X, y = scene(DISC, rate=0.0); M = clean_midpoints(X, y, close_pairs(X, y)); a = T((0.25, 0.05), BNode("pred", "norm", "ge", (1.15,))); self.assertEqual(atom_name(local_refit(a, M)), "norm.ge")
    def test_refiner_keeps_original(self):
        X, y = scene(DISC); p = close_pairs(X, y); raw = RansacProposer().propose(X, y, p, 0); out = LocalRefiner(RansacProposer()).propose(X, y, p, 0); self.assertEqual([c.params for c in out[: len(raw)]], [c.params for c in raw]); self.assertGreater(len(out), len(raw))
        self.assertTrue(all(c.source.startswith("local(") for c in out[len(raw):]))


class ValidatorTests(unittest.TestCase):
    def setUp(self): self.X, self.y = scene(DISC); self.prop, self.val = split_pairs(clean_pairs(self.X, self.y), 0)
    def test_true_primitive_accepted(self): v = RobustValidator(); e = v.evidence(DISC, self.prop, self.val); self.assertEqual(v.accept(e), (True, "retenue")); self.assertGreater(e.z_oos, 5.0); self.assertGreaterEqual(e.segments, 3)
    def test_far_primitive_rejected(self): v = RobustValidator(); self.assertFalse(v.accept(v.evidence(disc_atom(-2.0, -2.0, 0.5), self.prop, self.val))[0])
    def test_reasons(self):
        v = RobustValidator(z_oos=3.0, min_support=4, min_purity=0.55, min_segments=2); E = lambda **k: Evidence(**{**dict(z_in=9.0, z_oos=9.0, support_oos=20, purity_oos=0.9, segments=4, segments_used=5), **k})
        self.assertEqual(v.accept(E())[1], "retenue"); self.assertEqual(v.accept(E(support_oos=2))[1], "support insuffisant"); self.assertEqual(v.accept(E(z_oos=1.0))[1], "contraste insuffisant"); self.assertEqual(v.accept(E(purity_oos=0.3))[1], "pureté insuffisante")
        self.assertEqual(v.accept(E(segments=1))[1], "contraste trop local"); self.assertEqual(RobustValidator(max_instability=0.1).accept(E(instability=0.5))[1], "instable"); self.assertTrue(RobustValidator(max_instability=0.1).accept(E(instability=0.01))[0])
    def test_oos_uses_validation_half_only(self): v = RobustValidator(z_oos=3.0); self.assertFalse(v.accept(Evidence(z_in=20.0, z_oos=0.5, support_oos=20, purity_oos=0.9, segments=4, segments_used=5))[0])
    def test_in_sample_mode_applies_multiplicity(self):
        v = RobustValidator(z_oos=3.0, oos=False, min_support=0); e = Evidence(z_in=3.5, z_oos=0.0, support_oos=20, purity_oos=0.9, segments=4, segments_used=5); self.assertTrue(v.accept(e, 1)[0]); self.assertFalse(v.accept(e, 500)[0])
    def test_segments_true_vs_tangent(self):
        full, _ = boundary_segments(DISC, self.val); part, _ = boundary_segments(disc_atom(0.2, 1.5, 2.7), self.val); self.assertGreaterEqual(full, 4); self.assertLess(part, full)
    def test_segments_without_support(self): self.assertEqual(boundary_segments(disc_atom(2.9, 2.9, 0.05), self.val), (0, 0))
    def test_stability_true_is_stable(self): self.assertLess(stability(DISC, self.prop, self.X, n_boot=2), 0.1)
    def test_no_acceptance_on_random_labels(self):
        X = torch.rand(500, 2, generator=G(5)) * 6 - 3; y = (torch.rand(500, generator=G(6)) < 0.5).long(); 
        for split in ("points", "none"):
            d = discover_robust(X, y, LocalRefiner(UnionProposer([RansacProposer(), HoughProposer()])), RobustValidator(z_oos=P2.null_floor(split, 11.0)), seed=0, joint=True, split=split); self.assertEqual(len(d.primitives), 0, split); self.assertEqual(d.joint, [])
    def test_config_defaults(self): c = load_config(); self.assertEqual(set(c), {"split", "z_oos", "min_support", "min_purity", "min_segments", "prune_z"}); self.assertGreaterEqual(c["z_oos"], 3.0); self.assertIn(c["split"], ("points", "none", "pairs"))
    def test_multiplicity_applies_out_of_sample_too(self):
        v = RobustValidator(z_oos=3.0, min_support=0, min_purity=0.0, min_segments=0); e = Evidence(z_in=0.0, z_oos=3.5, support_oos=20, purity_oos=0.9, segments=4, segments_used=5); self.assertTrue(v.accept(e, 1)[0]); self.assertFalse(v.accept(e, 500)[0])
    def test_null_calibration(self):
        z = P2.null_max_z(LocalRefiner(RansacProposer()), trials=3); self.assertEqual(len(z), 3); self.assertTrue(all(v > 0 for v in z)); self.assertEqual(z, P2.null_max_z(LocalRefiner(RansacProposer()), trials=3))
        self.assertGreater(max(P2.null_max_z(LocalRefiner(RansacProposer()), trials=3, split="none")), 3.0)                      # the "z" is not a standard normal: a nominal threshold of 3 would be crossed without any boundary
    def test_split_points_disjoint(self):
        a, b = P2.split_points(self.X, self.y, 1); self.assertEqual(len(a) + len(b), len(self.X)); self.assertFalse(set(a.tolist()) & set(b.tolist()))
    def test_split_modes(self):
        for m in ("points", "pairs", "none"): d = discover_robust(self.X, self.y, LocalRefiner(RansacProposer()), RobustValidator(z_oos=P2.null_floor(m, 8.0)), seed=0, split=m); self.assertTrue(any(hits(c.representative.program, DISC) for c in d.primitives.classes), m)
    def test_evidence_needs_no_truth(self): self.assertEqual(list(inspect.signature(RobustValidator.evidence).parameters), ["self", "prog", "prop", "val", "X", "seed"])


class SetTests(unittest.TestCase):
    def test_near_boundary(self): M = torch.tensor([[1.4, 0.0], [0.2, 0.0], [1.3, 0.0]], dtype=DT); self.assertEqual(near_boundary(DISC, M, 0.2).tolist(), [True, False, True]); self.assertFalse(bool(near_boundary(DISC, M, 0.0).any()))
    def test_is_kin(self):
        self.assertTrue(is_kin(disc_atom(0, 0, 1.0), disc_atom(0.1, 0.1, 1.8))); self.assertFalse(is_kin(disc_atom(0, 0, 1.0), disc_atom(0.1, 0.1, 1.1))); self.assertFalse(is_kin(disc_atom(0, 0, 1.0), disc_atom(1.5, 0, 1.8)))
        self.assertTrue(is_kin(line_atom(0.3, 0.0), line_atom(0.35, 1.0))); self.assertTrue(is_kin(line_atom(0.3, 0.0), line_atom(0.3 + math.pi, -1.0))); self.assertFalse(is_kin(line_atom(0.3, 0.0), line_atom(1.3, 1.0))); self.assertFalse(is_kin(line_atom(0.3, 0.0), disc_atom(0, 0, 1)))
    def test_prune_removes_in_between_fit(self):
        X, y = scene(RING, rate=0.0); pairs = clean_pairs(X, y); cp = CompatiblePrimitives(X)
        for p in (disc_atom(-0.3, 0.2, 0.7), disc_atom(-0.3, 0.2, 1.6), disc_atom(-0.3, 0.2, 1.75)): cp.add(PrimitiveCandidate(p, "t", z=contrast(p, pairs)["z"]))
        self.assertEqual(len(cp), 3); gone = prune(cp, pairs, 3.0); self.assertEqual(len(cp), 2); self.assertEqual(len(gone), 1); self.assertTrue(all(hits(c.representative.program, RING) for c in cp.classes))
    def test_prune_keeps_independent(self):
        X, y = scene(XOR(DISC, DISC2), rate=0.0); pairs = clean_pairs(X, y); cp = CompatiblePrimitives(X); [cp.add(PrimitiveCandidate(p, "t", z=5.0)) for p in (DISC, DISC2)]; self.assertEqual(prune(cp, pairs, 3.0), []); self.assertEqual(len(cp), 2)
    def test_prune_never_empties(self): X, y = scene(DISC); cp = CompatiblePrimitives(X); cp.add(PrimitiveCandidate(disc_atom(2.5, 2.5, 0.3), "t")); prune(cp, clean_pairs(X, y), 99.0); self.assertEqual(len(cp), 1)
    def test_discover_two_atoms(self):
        X, y = scene(XOR(DISC2, HALF)); d = discover_robust(X, y, LocalRefiner(UnionProposer([RansacProposer(), HoughProposer()])), seed=0); reps = [c.representative.program for c in d.primitives.classes]
        self.assertTrue(any(hits(p, DISC2) for p in reps)); self.assertTrue(any(hits(p, HALF) for p in reps)); self.assertLessEqual(len(reps), 4)
    def test_discovery_log_is_complete(self):
        X, y = scene(DISC); d = discover_robust(X, y, LocalRefiner(RansacProposer()), seed=0); self.assertEqual(len([l for l in d.candidates if l["source"] != "joint"]), d.proposals)
        for l in d.candidates: self.assertLessEqual({"program", "source", "round", "z_in", "z_oos", "support", "purity", "segments", "accepted", "reason", "kept"}, set(l))
        self.assertGreaterEqual(sum(l["kept"] for l in d.candidates), len(d.primitives))
    def test_discovery_deterministic(self):
        X, y = scene(XOR(DISC2, HALF)); f = lambda: [c.representative.params for c in discover_robust(X, y, LocalRefiner(RansacProposer()), seed=2).primitives.classes]; self.assertEqual(f(), f())
    def test_single_fold_option(self): X, y = scene(DISC); d = discover_robust(X, y, LocalRefiner(RansacProposer()), seed=0, folds=1, rounds=1); self.assertEqual(d.rounds, 1)
    def test_no_world_access(self):
        src = inspect.getsource(P2); self.assertNotIn("deep_world", src); self.assertNotIn("deep_primitive_world", src); self.assertNotIn("truth", code_only(P2))
    def test_literals_closed_by_complement(self):
        X, y = scene(XOR(DISC2, HALF)); d = discover_robust(X, y, LocalRefiner(RansacProposer()), seed=0); L = literals_of(d); n = {atom_name(p) for p in L}; self.assertIn("norm.ge", n); self.assertEqual(len(L), 2 * len(d.primitives))
        P = torch.rand(200, 2, generator=G(1)) * 6 - 3; a, b = L[0], L[len(d.primitives)]; self.assertGreater(float((predict(a, P) != predict(b, P)).float().mean()), 0.97)


class JointTests(unittest.TestCase):
    def test_ring_from_concentric_discs(self): j = joint_candidates([disc_atom(0, 0, 0.7), disc_atom(0.1, 0.05, 1.5)]); self.assertEqual([atom_name(p) for p in j], ["norm.in"]); self.assertEqual(j[0].children[0].params, (0.7, 1.5))
    def test_no_ring_from_distant_discs(self): self.assertEqual(joint_candidates([disc_atom(0, 0, 0.7), disc_atom(1.5, 0, 1.5)]), [])
    def test_band_from_parallel_lines(self): j = joint_candidates([line_atom(1.0, -0.6), line_atom(1.0 + math.pi, -0.6)]); self.assertEqual([atom_name(p) for p in j], ["dot.in"]); self.assertAlmostEqual(j[0].params[1], -0.6); self.assertAlmostEqual(j[0].params[2], 0.6)
    def test_sectors_from_secant_lines(self):
        j = joint_candidates([line_atom(0.0, 0.5), line_atom(math.pi / 2, -0.25)]); self.assertTrue(j); self.assertTrue(all(atom_name(p) == "angle.in" for p in j))
        for p in j: self.assertAlmostEqual(p.params[0], 0.5, places=6); self.assertAlmostEqual(p.params[1], -0.25, places=6)
        self.assertEqual(joint_candidates([line_atom(0.0, 0.5), line_atom(math.pi / 2, -0.25)], sectors=False), [])
    def test_sector_recovered_from_its_lines(self):
        cs = components(SECTOR); lines = [line_atom(*__import__("core.deep_primitive_eval", fromlist=["x"])._ray_as_line(c)[1:]) for c in cs]; j = joint_candidates(lines); P = torch.rand(2000, 2, generator=G(2)) * 6 - 3; t = predict(SECTOR, P)
        self.assertGreater(max(float((predict(p, P) == t).float().mean()) for p in j), 0.97)
    def test_sector_not_exclusive_for_full_lines(self):
        a, b = line_atom(0.0, 0.0), line_atom(math.pi / 2, 0.0); X, y = scene(XOR(a, b), rate=0.0); pairs = clean_pairs(X, y)
        for s in joint_candidates([a, b]): self.assertFalse(sector_exclusive(s, [a, b], pairs))
    def test_sector_exclusive_for_true_sector(self):
        X, y = scene(SECTOR, n=900, rate=0.0); pairs = clean_pairs(X, y); cs = components(SECTOR); from core.deep_primitive_eval import _ray_as_line
        self.assertTrue(sector_exclusive(SECTOR, [line_atom(*_ray_as_line(c)[1:]) for c in cs], pairs))
    def test_joint_fit_keeps_type(self): X, y = scene(RING, rate=0.0); r = joint_fit(joint_candidates([disc_atom(-0.25, 0.2, 0.75), disc_atom(-0.3, 0.25, 1.5)]), clean_pairs(X, y), steps=10); self.assertEqual([atom_name(p) for p in r], ["norm.in"])
    def test_ring_discovered(self):
        X, y = scene(RING, n=900, rate=0.02); d = discover_robust(X, y, LocalRefiner(UnionProposer([RansacProposer(), HoughProposer()])), seed=0, joint=True); P = torch.rand(2000, 2, generator=G(2)) * 6 - 3; t = predict(RING, P)
        self.assertTrue(any(atom_name(p) == "norm.in" and float((predict(p, P) == t).float().mean()) > 0.93 for p in d.joint)); self.assertIn("joint", {l["source"] for l in d.candidates})
    def test_joint_off_by_default(self): X, y = scene(RING, n=900); self.assertEqual(discover_robust(X, y, LocalRefiner(RansacProposer()), seed=0).joint, [])
    def test_hypotheses_keep_alternatives(self):
        X, y = scene(RING, n=900, rate=0.02); d = discover_robust(X, y, LocalRefiner(UnionProposer([RansacProposer(), HoughProposer()])), seed=0, joint=True); H = CompatiblePrimitiveHypotheses(d, X); amb = H.ambiguous(); self.assertTrue(amb)
        for g in H.groups: self.assertAlmostEqual(sum(w for _, w in g.alternatives), 1.0, places=6)
        self.assertTrue(any(atom_name(ps[0]) == "norm.in" for ps, _ in amb[0].alternatives)); self.assertGreaterEqual(len(H.programs()), len(H))
    def test_borderline_and_query_scores(self):
        X, y = scene(XOR(DISC2, HALF)); d = discover_robust(X, y, LocalRefiner(RansacProposer()), seed=0); b = borderline(d); self.assertTrue(all(isinstance(p, BNode) for p in b)); pool = torch.rand(50, 2, generator=G(3)) * 6 - 3
        s = boundary_query_scores([DISC], pool); self.assertEqual(s.shape, (50,)); near = ((pool - torch.tensor([0.2, 0.0])).norm(dim=1) - 1.2).abs() < 0.1; self.assertTrue(bool((s[near] > 0).all())); self.assertEqual(float(boundary_query_scores([], pool).sum()), 0.0)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.sc = RP.scene_at(0, "two", tune=True)
    def test_scenes_disjoint_from_previous_bench(self):
        from core.deep_primitive_world import make_scene
        self.assertGreaterEqual(RP.EVAL_BASE, 2000); self.assertNotEqual(RP.EVAL_BASE, RP.TUNE_BASE); self.assertEqual(RP.scene_at(3, "two").task.seed, make_scene(2003, "two").task.seed); self.assertEqual(RP.scene_at(3, "two", tune=True).task.seed, make_scene(5003, "two").task.seed)
        self.assertEqual(RP.scene_at(0, "single").task.split, "dev")
    def test_logged_reference_matches_discover(self):
        X, y = self.sc.task.X, self.sc.task.y; cp, log, n, fe = RP.discover_logged(X, y, GradientProposer(), seed=1); d = discover(X, y, GradientProposer(), seed=1)
        self.assertEqual([c.representative.params for c in cp.classes], [c.representative.params for c in d.primitives.classes]); self.assertEqual(n, d.proposals); self.assertEqual(len(log), n)
    def test_arms_defined(self):
        self.assertEqual(RP.ARMS[RP.REFERENCE], ("gradient", "old", False)); self.assertIn(RP.BEST, RP.ARMS)
        for a, (p, v, j) in RP.ARMS.items(): self.assertTrue(v == "old" or RP.validator(v) is not None, a); self.assertTrue(p.startswith(("neural", "union_neural")) or RP.proposer(p) is not None, a)
    def test_arm_row(self):
        r = RP.arm_row(self.sc, RP.BEST, 0)
        for k in ("equivalence", "exact", "precision", "false_positives", "retained", "compactness", "proposals", "fit_equivalents", "atoms"): self.assertIn(k, r)
        self.assertEqual(r["method"], RP.BEST); self.assertGreaterEqual(r["retained"], r["false_positives"])
    def test_audit_classes(self):
        rows = RP.audit(self.sc, RP.BEST, 0); comp = [r for r in rows if r["kind"] == "component"]; self.assertEqual(len(comp), sum(len(components(a)) for a in self.sc.atoms))
        for r in comp: self.assertIn(r["class"], RP.AUDIT_CLASSES)
        for r in rows: self.assertIn(r["kind"], ("component", "false_positive"))
    def test_validator_study_counts(self):
        rows = RP.validator_study(self.sc, 0); self.assertEqual([r["validator"] for r in rows], list(RP.VALIDATORS))
        for r in rows: self.assertEqual(r["TA"] + r["FA"] + r["TR"] + r["FR"], r["candidates"])
        self.assertEqual(len({r["TA"] + r["FR"] for r in rows}), 1)
    def test_diversity_rows(self):
        rows = RP.diversity(self.sc, 0); n = sum(len(components(a)) for a in self.sc.atoms)
        for r in rows: self.assertEqual(r["n_true"], n); self.assertTrue(all(0 <= i < n for i in r["hits"]))
        self.assertLessEqual({"gradient", "ransac", "hough", "grid"}, {r["proposer"] for r in rows})
    def test_oracle_literals(self):
        from core.deep_reachability import leaves
        t = DX.down_task(400, 2, "standard"); tl = leaves(t.world.truth); self.assertIs(RP.oracle_literals(t, "true"), tl) if False else self.assertEqual([repr(a) for a in RP.oracle_literals(t, "true")], [repr(a) for a in tl])
        for s in ("true_types", "partial"): self.assertEqual([atom_name(a) for a in RP.oracle_literals(t, s)], [atom_name(a) for a in tl])
    def test_chain_arms(self):
        self.assertEqual(RP.SOURCES[0], "reference"); self.assertIn(("robust_joint", "validated"), RP.CHAIN_ARMS); self.assertEqual(len(set(RP.CHAIN_ARMS)), len(RP.CHAIN_ARMS))
    def test_subunits(self):
        self.assertIn("deeprp", DX.SUBUNITS); a = DX.subunit_ids("deeprp", "0"); self.assertEqual(len(a), 9 * 2 * 4 + 3); self.assertEqual(len(set(a)), len(a)); c = DX.subunit_ids("deeprp", "400"); self.assertEqual(len(c), 4 * 2 * len(RP.CHAIN_ARMS)); self.assertTrue(all(i.startswith("chain_") for i in c))
    def test_chain_tasks_are_new_dev_tasks(self):
        t = DX.down_task(400, 3, "rich"); self.assertEqual(t.split, "dev"); used = {DX.down_task(k, 3, "rich").seed for k in list(range(16)) + list(range(100, 116)) + list(range(200, 216)) + list(range(300, 308))}; self.assertNotIn(t.seed, used)
    def test_sealed_set_never_referenced(self):
        for m in (P2, RP, RR): s = inspect.getsource(m); self.assertNotIn("sealed_task", s); self.assertNotIn('"sealed"', s); self.assertNotIn('"test"', s)
    def test_downstream_pipeline_untouched(self):
        s = inspect.getsource(DS); self.assertNotIn("deep_primitives2", s); self.assertNotIn("deep_robust_prim", s); self.assertIsNone(inspect.signature(DS.DeepSearch.__init__).parameters["validated"].default)
        c = inspect.getsource(RP.chain); self.assertIn("fact_literals", c); self.assertIn('DeepSearch("factorized"', c)


class ReportTests(unittest.TestCase):
    def test_paired_ci(self):
        r = RR.paired_ci([0.0] * 20, [1.0] * 20); self.assertEqual(r["diff"], 1.0); self.assertEqual(r["ci95"], [1.0, 1.0]); self.assertIsNone(RR.paired_ci([0.0], [1.0])); r = RR.paired_ci([0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]); self.assertLessEqual(r["ci95"][0], 0.0); self.assertGreaterEqual(r["ci95"][1], 0.0)
    def test_paired_ci_ignores_missing(self): self.assertEqual(RR.paired_ci([0.0, None, 0.0, 0.0], [1.0, 1.0, 1.0, None])["n"], 2) if False else self.assertIsNone(RR.paired_ci([0.0, None, 0.0, 0.0], [1.0, 1.0, 1.0, None]))
    def test_summarize(self):
        sc = RP.scene_at(1, "single", tune=True); tag = {"category": "single", "scene": "1_single"}; rows = [{"experiment": "atomic", **tag, **RP.arm_row(sc, a, 0)} for a in (RP.REFERENCE, RP.BEST)]
        rows += [{"experiment": "audit", **tag, **r} for a in (RP.REFERENCE, RP.BEST) for r in RP.audit(sc, a, 0)] + [{"experiment": "validator", **tag, **r} for r in RP.validator_study(sc, 0)] + [{"experiment": "diversity", **tag, **r} for r in RP.diversity(sc, 0)]
        rows += [{"experiment": "chain", "source": s, "policy": "always", "task_id": str(i), "depth": 2, "primitive_equivalence": 0.5, "candidate_functional": True, "final_functional": i % 2 == 0, "final_full": 0.9, "final_095": True} for s in ("reference", "robust_joint") for i in range(4)]
        S = RR.summarize(rows); self.assertEqual(set(S["atomic"]), {RP.REFERENCE, RP.BEST}); self.assertIn(RP.BEST, S["paired_vs_reference"]); self.assertAlmostEqual(sum(S["audit"][RP.BEST]["all"][k] for k in RP.AUDIT_CLASSES), 1.0, places=2)
        self.assertIn("old", S["validator"]); self.assertIn("union_recall", S["diversity"]); self.assertEqual(S["chain"]["robust_joint/always"]["n"], 4); self.assertIn("robust_joint/always − reference/always", S["chain_paired"]); RR.show(S)


if __name__ == "__main__":
    unittest.main()

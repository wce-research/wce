import inspect
import math
import unittest

import torch

import core.deep_bo_report as BR
import core.deep_experiments as DX
import core.deep_ownership as OW
import core.deep_ownership_exp as OX
import core.deep_search as DS
from core.boundary_program import BNode, predict
from core.deep_fit import DT
from core.deep_ownership import (DEFAULTS, ROLES, BoundaryOwnershipHypothesis, Occlusion, Owner, OwnershipSolution, absent_fraction, candidates, energy, expected_visibility, filter_by_ownership, find_occlusions, load_config, owned,
                                 ownership_from_program, ownership_hypotheses, primitives_from, refit, solve, visible_support)
from core.deep_partial import observe
from core.deep_primitives2 import disc_atom, line_atom

G = lambda s=0: torch.Generator().manual_seed(s)
T = lambda c, ch: BNode("translate", params=c, children=(ch,))
DISC = disc_atom(0.2, 0.0, 1.2)
RING = T((-0.3, 0.2), BNode("pred", "norm", "in", (0.7, 1.6)))
SECTOR = T((-0.5, 0.0), BNode("pred", "angle", "in", (0.4, 0.8)))
OCC = line_atom(0.0, 0.3)
AND = lambda a, b: BNode("and", children=(a, b))
CFG = {**DEFAULTS, "gamma": 0.0, "chance": 0.0, "lam": 10.0, "temperature": 1.0}
O_DISC, O_OCC = Owner("circle", (0.2, 0.0, 1.2)), Owner("line", (0.0, 0.3))


def scene(prog, n=900, rate=0.0, s=0):
    X = torch.rand(n, 2, generator=G(s)) * 6 - 3; y = predict(prog, X); f = torch.rand(n, generator=G(s + 1)) < rate; return X, torch.where(f, 1 - y, y)


def OB(prog, **kw): X, y = scene(prog, **kw); return observe(X, y)


class OwnerTests(unittest.TestCase):
    def test_distance_and_program(self):
        M = torch.tensor([[1.4, 0.0], [0.2, 0.0], [0.3, 2.0]], dtype=DT); self.assertTrue(torch.allclose(O_DISC.distance(M), torch.tensor([0.0, 1.2, 0.8025], dtype=DT), atol=1e-3)); self.assertTrue(torch.allclose(O_OCC.distance(M), torch.tensor([1.1, 0.1, 0.0], dtype=DT), atol=1e-6))
        self.assertEqual(O_DISC.n_params, 3); self.assertEqual(O_OCC.n_params, 2); self.assertEqual(repr(O_DISC.program()), repr(DISC)); self.assertGreater(len(O_DISC.boundary()), 50)
    def test_candidates_contain_truth(self):
        C = candidates(OB(AND(DISC, OCC))); self.assertTrue(any(c.kind == "circle" and math.hypot(c.params[0] - 0.2, c.params[1]) < 0.3 and abs(c.params[2] - 1.2) < 0.25 for c in C)); self.assertTrue(any(c.kind == "line" and abs(math.cos(c.params[0])) > 0.97 and abs(abs(c.params[1]) - 0.3) < 0.25 for c in C))
    def test_refit(self):
        t = torch.linspace(0, 2, 30, dtype=DT); S = torch.tensor([0.2, 0.0], dtype=DT) + 1.2 * torch.stack([torch.cos(t), torch.sin(t)], 1); p = refit(Owner("circle", (0.4, 0.1, 1.0)), S); self.assertAlmostEqual(p[2], 1.2, places=2); self.assertIsNone(refit(O_DISC, S[:2]))
        L = torch.stack([torch.full((20,), 0.3, dtype=DT), torch.linspace(-2, 2, 20, dtype=DT)], 1); q = refit(Owner("line", (0.1, 0.2)), L); self.assertAlmostEqual(abs(math.cos(q[0])), 1.0, places=4); self.assertAlmostEqual(q[1], 0.3, places=4); self.assertIsNone(refit(Owner("line", (math.pi / 2, 0.0)), L))
    def test_refit_refuses_runaway_circle(self): L = torch.stack([torch.linspace(-2, 2, 30, dtype=DT), torch.zeros(30, dtype=DT)], 1); self.assertIsNone(refit(Owner("circle", (0.0, 1.0, 1.0)), L))


class OcclusionTests(unittest.TestCase):
    def setUp(self): self.ob = OB(AND(DISC, OCC))
    def test_occluder_found(self):
        occ = find_occlusions([O_DISC, O_OCC], self.ob); d = [o for o in occ if o.occluded == 0]; self.assertEqual(len(d), 1); self.assertIsInstance(d[0], Occlusion); self.assertEqual(d[0].occluder, 1); self.assertGreater(d[0].hidden_fraction, 0.2)
        P = torch.tensor([[1.3, 0.0]]); self.assertEqual(bool(predict(O_OCC.program(), P)[0]), d[0].side)              # the hidden side is indeed the one where x > 0.3
    def test_no_occlusion_when_fully_visible(self): self.assertEqual([o for o in find_occlusions([O_DISC, O_OCC], OB(DISC)) if o.occluded == 0], [])
    def test_unrelated_curve_is_not_an_occluder(self): self.assertEqual([o for o in find_occlusions([O_DISC, Owner("line", (0.0, 2.9))], self.ob) if o.occluded == 0], [])
    def test_expected_visibility(self):
        v = expected_visibility(DISC, self.ob, [(OCC, False)]); self.assertEqual(len(v["points"]), len(v["seen"])); self.assertGreater(float(v["hidden"].float().mean()), 0.2); self.assertFalse(bool((v["hidden"] & v["expected"]).any()))
        self.assertGreater(float(v["seen"][v["expected"]].float().mean()), 0.85); self.assertLess(float(v["seen"][v["hidden"]].float().mean()), 0.3); self.assertEqual(int(expected_visibility(DISC, self.ob)["hidden"].sum()), 0)
    def test_visible_support_ignores_hidden_part(self):
        a, b = visible_support(DISC, self.ob, [(OCC, False)]), visible_support(DISC, self.ob, []); self.assertGreater(a["coverage"], b["coverage"] + 0.15); self.assertGreater(a["hidden"], 0.2); self.assertEqual(b["hidden"], 0.0); self.assertGreater(a["score"], b["score"])
    def test_wrong_primitive_scores_low(self): self.assertLess(visible_support(disc_atom(-2.0, 2.0, 0.6), self.ob, [(OCC, False)])["score"], 0.1 * visible_support(DISC, self.ob, [(OCC, False)])["score"])
    def test_absent_fraction_uses_occlusion(self):
        own = [O_DISC, O_OCC]; occ = find_occlusions(own, self.ob); self.assertLess(absent_fraction(0, own, occ, self.ob, True), 0.15); self.assertGreater(absent_fraction(0, own, occ, self.ob, False), 0.2)
    def test_domain_is_not_absence(self):
        p = disc_atom(2.6, 0.0, 1.2); ob = OB(p); v = expected_visibility(p, ob); self.assertGreater(float((~v["domain"]).float().mean()), 0.3); self.assertGreater(float(v["seen"][v["expected"]].float().mean()), 0.8)


class EnergyTests(unittest.TestCase):
    def setUp(self): self.ob = OB(AND(DISC, OCC)); self.M = self.ob.points
    def _labels(self, owners): U = torch.stack([o.distance(self.M) for o in owners], 1); d, i = U.min(1); return torch.where(d < 0.25, i, torch.full_like(i, -1))
    def test_terms_reported_separately(self):
        e = energy(self._labels([O_DISC, O_OCC]), [O_DISC, O_OCC], self.ob, CFG); self.assertEqual(set(e), {"fit", "noise", "smooth", "complexity", "absent", "chance", "total", "owners"}); self.assertAlmostEqual(e["total"], e["fit"] + e["noise"] + e["smooth"] + e["complexity"] + e["absent"] + e["chance"], places=6)
        self.assertEqual(e["complexity"], CFG["lam"] * 5); self.assertEqual(e["owners"], 2); self.assertEqual(e["absent"], 0.0)
    def test_true_attribution_beats_alternatives(self):
        own = [O_DISC, O_OCC]; good = energy(self._labels(own), own, self.ob, CFG)["total"]; none = energy(torch.full((len(self.M),), -1, dtype=torch.long), own, self.ob, CFG)["total"]; one = energy(self._labels([O_DISC]), [O_DISC], self.ob, CFG)["total"]
        self.assertLess(good, one); self.assertLess(one, none); bad = [Owner("circle", (0.5, 0.3, 0.9)), O_OCC]; self.assertLess(good, energy(self._labels(own), bad, self.ob, CFG)["total"])
    def test_complexity_penalises_extra_owner(self):
        own = [O_DISC, O_OCC]; lab = self._labels(own); extra = own + [Owner("circle", (2.5, 2.5, 0.3))]; self.assertEqual(energy(lab, extra, self.ob, CFG)["total"], energy(lab, own, self.ob, CFG)["total"])                       # a curve without a midpoint costs nothing and is not "used"
        split = lab.clone(); idx = torch.nonzero(lab == 0)[:, 0][:6]; split[idx] = 2; self.assertGreater(energy(split, extra, self.ob, CFG)["total"], energy(lab, own, self.ob, CFG)["total"])
    def test_absent_term_and_occlusion(self):
        own = [O_DISC, O_OCC]; lab = self._labels(own); a = energy(lab, own, self.ob, {**CFG, "gamma": 1.0, "occlusion": True})["absent"]; b = energy(lab, own, self.ob, {**CFG, "gamma": 1.0, "occlusion": False})["absent"]; self.assertGreater(b, a)
    def test_smooth_term(self):
        own = [O_DISC, O_OCC]; lab = self._labels(own); m = torch.nonzero(lab == 0)[:, 0]; noisy = lab.clone(); noisy[m[::2]] = -1; self.assertGreater(energy(noisy, own, self.ob, CFG)["smooth"], energy(lab, own, self.ob, CFG)["smooth"]); self.assertEqual(energy(noisy, own, self.ob, {**CFG, "beta": 0.0})["smooth"], 0.0)


class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.ob = OB(AND(DISC, OCC)); cls.sol = solve(cls.ob, CFG)
    def test_finds_both_curves(self):
        s = self.sol; self.assertIsInstance(s, OwnershipSolution); self.assertTrue(any(o.kind == "circle" and math.hypot(o.params[0] - 0.2, o.params[1]) < 0.3 and abs(o.params[2] - 1.2) < 0.25 for o in s.owners)); self.assertTrue(any(o.kind == "line" and abs(abs(o.params[1]) - 0.3) < 0.25 for o in s.owners)); self.assertLessEqual(len(s.owners), 4)
    def test_labels_and_owner_ids_consistent(self):
        s = self.sol; self.assertEqual(len(s.labels), len(self.ob.points)); self.assertLess(int(s.labels.max()), len(s.owners))
        for i, o in enumerate(s.owners): self.assertTrue(torch.equal(o.ids, torch.nonzero(s.labels == i)[:, 0])); self.assertGreaterEqual(len(o.ids), 4)
    def test_posterior(self):
        P = self.sol.posterior; self.assertEqual(P.shape, (len(self.ob.points), len(self.sol.owners) + 1)); self.assertTrue(torch.allclose(P.sum(1), torch.ones(len(P), dtype=DT), atol=1e-6)); t = self.sol.top_k(3); self.assertEqual(t.shape[0], len(P)); self.assertLessEqual(t.shape[1], 3)
        m = self.sol.labels >= 0; self.assertGreater(float((t[m, 0] == self.sol.labels[m]).float().mean()), 0.9)
    def test_temperature_flattens_posterior(self): hot = solve(self.ob, {**CFG, "temperature": 8.0}).posterior.max(1).values.mean(); cold = solve(self.ob, {**CFG, "temperature": 0.5}).posterior.max(1).values.mean(); self.assertLess(float(hot), float(cold))
    def test_energy_log_decreases(self):
        tot = [l["total"] for l in self.sol.log if l["step"] == "add" or l.get("action") == "commit"]; self.assertTrue(tot); self.assertEqual(tot, sorted(tot, reverse=True)); self.assertLessEqual(sum(l["step"] == "refine" for l in self.sol.log), DEFAULTS["max_iters"])
        self.assertLessEqual(self.sol.terms["total"], CFG["tau"] * len(self.ob.points))
    def test_bounded_search(self): st = self.sol.stats; self.assertLessEqual(st["assignments_tested"], st["candidates"] * (len(self.sol.owners) + 1) + DEFAULTS["max_iters"]); self.assertLess(st["assignments_tested"], 200)
    def test_long_range_association(self):
        occ2 = BNode("or", children=(line_atom(0.0, -0.4), line_atom(math.pi, -0.9)))                                    # the disc is visible only for x ≤ −0.4 or x ≥ 0.9: two disjoint arcs
        ob = OB(AND(DISC, occ2)); far = solve(ob, {**CFG, "long_range": True}); loc = solve(ob, {**CFG, "long_range": False}); arc = lambda s: max((len(o.ids) for o in s.owners if o.kind == "circle" and abs(o.params[2] - 1.2) < 0.3), default=0)
        xs = lambda s: [ob.points[o.ids][:, 0] for o in s.owners if o.kind == "circle" and abs(o.params[2] - 1.2) < 0.3]; self.assertGreater(arc(far), arc(loc)); self.assertTrue(any(float(x.min()) < -0.4 and float(x.max()) > 0.9 for x in xs(far)))
    def test_occlusions_reported(self): s = solve(self.ob, {**CFG, "occlusion": True}); self.assertTrue(all(isinstance(o, Occlusion) and o.occluder != o.occluded for o in s.occlusions)); self.assertEqual(solve(self.ob, {**CFG, "occlusion": False}).occlusions, [])
    def test_no_boundary(self): X, _ = scene(DISC); s = solve(observe(X, torch.zeros(len(X), dtype=torch.long))); self.assertEqual(s.owners, []); self.assertEqual(len(s.labels), 0); self.assertEqual(s.stats["assignments_tested"], 0)
    def test_chance_cost_reduces_attribution_on_random_labels(self):
        X = torch.rand(300, 2, generator=G(5)) * 6 - 3; y = (torch.rand(300, generator=G(6)) < 0.5).long(); ob = observe(X, y); a = float((solve(ob, CFG).labels >= 0).float().mean()); b = float((solve(ob, {**CFG, "chance": 3.0}).labels >= 0).float().mean())
        self.assertGreater(a, 0.8); self.assertLess(b, a - 0.2)                                                # without calibration, almost everything is attributed even though no boundary exists
        e = energy(self.sol.labels, self.sol.owners, self.ob, {**CFG, "chance": 3.0}); self.assertGreater(e["chance"], 0.0); self.assertEqual(energy(self.sol.labels, self.sol.owners, self.ob, CFG)["chance"], 0.0)
    def test_null_assigned(self):
        from core.deep_ownership import null_assigned
        v = null_assigned({**CFG, "chance": 3.0}, trials=2); self.assertEqual(len(v), 2); self.assertTrue(all(0.0 <= x <= 1.0 for x in v)); self.assertEqual(v, null_assigned({**CFG, "chance": 3.0}, trials=2))
    def test_chance_support(self):
        from core.deep_ownership import chance_support
        a, b = chance_support(O_DISC, self.ob, CFG), chance_support(Owner("circle", (0.2, 0.0, 0.4)), self.ob, CFG); self.assertGreater(a, b); self.assertGreater(b, 0.0); self.assertLess(a, 0.5 * len(self.ob.points)); self.assertEqual(chance_support(Owner("circle", (40.0, 40.0, 0.3)), self.ob, CFG), 0.0)
        part = chance_support(O_DISC, self.ob, CFG, self.sol.owners[0].ids[:5]); self.assertLess(part, a); self.assertEqual(chance_support(O_DISC, self.ob, CFG, torch.zeros(0, dtype=torch.long)), 0.0)
    def test_deterministic(self):
        a, b = solve(self.ob, CFG, seed=2), solve(self.ob, CFG, seed=2); self.assertTrue(torch.equal(a.labels, b.labels)); self.assertEqual([o.kind for o in a.owners], [o.kind for o in b.owners])
        for o, q in zip(a.owners, b.owners): self.assertTrue(all(abs(u - v) < 1e-9 for u, v in zip(o.params, q.params)))
    def test_config(self): c = load_config(); self.assertLessEqual(set(DEFAULTS), set(c)); self.assertGreater(c["temperature"], 0); self.assertGreaterEqual(c["gamma"], 0)
    def test_no_learned_model_no_world(self):
        s = inspect.getsource(OW)
        for w in ("deep_world", "deep_primitive_world", "deep_primitive_eval", "deep_ownership_exp", "deep_partial_exp", "deep_fragments_exp", "torch.nn.Module", "neural"): self.assertNotIn(w, s)


class HypothesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.ob = OB(AND(DISC, OCC)); cls.sol = solve(cls.ob, {**CFG, "temperature": 4.0}); cls.H = ownership_hypotheses(cls.sol, cls.ob)
    def test_options_are_distributions(self):
        self.assertTrue(self.H)
        for h in self.H:
            self.assertIsInstance(h, BoundaryOwnershipHypothesis); self.assertAlmostEqual(sum(w for _, _, w in h.options), 1.0, places=6); self.assertEqual([w for _, _, w in h.options], sorted((w for _, _, w in h.options), reverse=True))
            for r, i, _ in h.options: self.assertIn(r, ROLES); self.assertTrue((i is None) == (r in ("domain", "noise", "unknown")))
    def test_no_single_owner_forced(self): self.assertTrue(any(len(h.options) > 1 for h in self.H)); self.assertTrue(all(len(h.top(3)) <= 3 for h in self.H)); self.assertTrue(all(0 < h.confidence <= 1 for h in self.H))
    def test_fragments_partition_points(self): ids = torch.cat([h.ids for h in self.H]); self.assertEqual(sorted(ids.tolist()), list(range(len(self.ob.points))))
    def test_occluder_role(self):
        s = solve(self.ob, {**CFG, "occlusion": True}); H = ownership_hypotheses(s, self.ob); roles = {r for h in H for r, _, _ in h.options}
        if s.occlusions: self.assertIn("occluder", roles)
        self.assertIn("noise", roles)
    def test_domain_role_near_box_edge(self):
        p = disc_atom(2.9, 0.0, 1.0); ob = OB(p, n=1400); s = solve(ob, CFG); roles = {r for h in ownership_hypotheses(s, ob, edge=0.5) for r, _, _ in h.options}; self.assertIn("domain", roles)
    def test_owners_listed(self): h = max(self.H, key=lambda h: len(h.ids)); self.assertTrue(h.owners(3)); self.assertTrue(all(0 <= i < len(self.sol.owners) for i in h.owners(3)))


class CompositionTests(unittest.TestCase):
    def setUp(self): self.ob = OB(AND(DISC, OCC))
    def test_hidden_leaf_cannot_own(self):
        lab, lv = ownership_from_program(AND(DISC, OCC), self.ob); self.assertEqual(len(lv), 2); M = self.ob.points; d = (M - torch.tensor([0.2, 0.0], dtype=DT)).norm(dim=1); on_line_inside = ((M[:, 0] - 0.3).abs() < 0.1) & (d < 1.0)
        self.assertGreater(float((lab[on_line_inside] == 1).float().mean()), 0.8); arc = ((d - 1.2).abs() < 0.1) & (M[:, 0] < 0.0); self.assertGreater(float((lab[arc] == 0).float().mean()), 0.8)
    def test_single_leaf(self): lab, lv = ownership_from_program(DISC, OB(DISC)); self.assertEqual(len(lv), 1); self.assertGreater(float((lab == 0).float().mean()), 0.8)
    def test_owned_and_filter(self):
        s = solve(self.ob, CFG); self.assertEqual(owned(DISC, s), 1.0); self.assertEqual(owned(disc_atom(-2.0, 2.0, 0.5), s), 0.0); self.assertEqual(owned(T((0.2, 0.0), BNode("pred", "norm", "in", (0.5, 1.2))), s), 0.5)
        keep = filter_by_ownership([DISC, disc_atom(-2.0, 2.0, 0.5), OCC], s); self.assertEqual([repr(p) for p in keep], [repr(DISC), repr(OCC)]); self.assertEqual(len(filter_by_ownership([T((0.2, 0.0), BNode("pred", "norm", "in", (0.5, 1.2)))], s, 0.5)), 1)
    def test_sector_curves_are_its_rays(self): c = OW._curves(SECTOR); self.assertEqual([k for k, _ in c], ["line", "line"]); self.assertEqual(len(OW._curves(RING)), 2)
    def test_primitives_from_owners(self):
        ob = OB(RING); sel, H = primitives_from(solve(ob, CFG), ob, 3); self.assertTrue(sel); self.assertIn("ring", {h.type for h in H}); self.assertLessEqual(len(sel), len(H))
    def test_compose_small(self):
        prog, e = OX.compose_small([DISC, OCC], self.ob.X, self.ob.y); self.assertEqual(prog.kind, "and"); self.assertEqual(e, 0); self.assertEqual(OX.compose_small([], self.ob.X, self.ob.y)[0], None)


class BenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.sc = OX.scene_at(0, "disc", 0.4, 0.0, tune=True)
    def test_scenes_unused_before(self):
        from core.deep_partial_exp import masked_scene
        self.assertEqual(OX.scene_at(3, "disc", 0.2, 0.0).seed, masked_scene(203, "disc", 0.2, 0.0).seed); used = {masked_scene(i, k, 0.0, 0.0).seed for i in list(range(32)) + list(range(100, 132)) for k in OX.KINDS}; self.assertFalse({OX.scene_at(i, k, 0.0, 0.0).seed for i in range(32) for k in OX.KINDS} & used)
    def test_attr_rows(self):
        rows = OX.attr_rows(self.sc); self.assertEqual([r["method"] for r in rows], list(OX.METHODS)); s = next(r for r in rows if r["method"] == "solver")
        for k in ("f1", "purity", "pieces_per_boundary", "mean_length", "top1", "top3", "brier", "cal", "energy", "assignments_tested", "occlusion_found", "occlusion_correct", "ambiguous_fragments"): self.assertIn(k, s)
        self.assertGreaterEqual(s["top3"], s["top1"]); self.assertEqual(len(s["cal"]), 10); self.assertGreater(sum(b[0] for b in s["cal"]), 0)
    def test_fragmentation(self):
        src = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]); M = torch.arange(16, dtype=DT).reshape(8, 2); f = OX.fragmentation(src.clone(), src, M); self.assertEqual(f["purity"], 1.0); self.assertEqual(f["pieces_per_boundary"], 1.0)
        f = OX.fragmentation(torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]), src, M); self.assertEqual(f["pieces_per_boundary"], 2.0); self.assertEqual(OX.fragmentation(torch.zeros(8, dtype=torch.long), src, M)["purity"], 0.5)
    def test_ownership_accuracy_perfect(self):
        ob = observe(self.sc.X, self.sc.y); atoms = OX._atoms(self.sc); from core.deep_primitive_eval import components
        comps = [c for a in atoms for c in components(a)]; src, _ = OX.true_sources(ob.points, atoms); from core.deep_partial import geometry
        g = geometry(self.sc.target); o = Owner("circle", (*g["center"], g["radii"][0])); P = torch.zeros(len(src), 2, dtype=DT); P[:, 0] = 0.9; P[:, 1] = 0.1; sol = OwnershipSolution([o], torch.zeros(len(src), dtype=torch.long), P, [], {}, [], {})
        a = OX.ownership_accuracy(sol, torch.where(src == 0, src, torch.full_like(src, -1)), comps); self.assertEqual(a["top1"], 1.0); self.assertAlmostEqual(a["brier"], 0.01, places=6); self.assertIsNone(OX.ownership_accuracy(sol, torch.full_like(src, -1), comps)["top1"])
    def test_oracle_vis_sector_rows(self):
        r = OX.oracle_row(self.sc); self.assertEqual({"TT", "TI", "ET", "EI", "NT", "NI", "OWN"} - set(r), set()); v = OX.vis_row(self.sc); self.assertEqual({"contrast", "visible_inferred", "visible_no_occluder", "visible_true_occluder", "pool_has_truth"} - set(v), set())
        s = OX.sector_row(OX.scene_at(0, "sector", 0.0, 0.0, tune=True)); self.assertIn(s["tt_stage"], ("ok", "no_hypothesis", "vertex", "angles")); self.assertLessEqual(s["rays_owned"], 2)
    def test_subunits_and_arms(self):
        self.assertIn("deepbo", DX.SUBUNITS); a = DX.subunit_ids("deepbo", "0"); self.assertEqual(len(a), 30); self.assertIn("solver_no_chance", OX.METHODS); c = DX.subunit_ids("deepbo", "700"); self.assertEqual(len(c), 8 * len(OX.CHAIN_ARMS)); self.assertEqual(len(set(OX.CHAIN_ARMS)), len(OX.CHAIN_ARMS)); self.assertLessEqual(set(OX.PREREGISTERED), set(OX.CHAIN_ARMS))
    def test_chain_tasks_new(self):
        used = {DX.down_task(k, 3, "rich").seed for r in ((0, 16), (100, 116), (200, 216), (300, 308), (400, 416), (500, 516), (600, 616)) for k in range(*r)}; new = {DX.down_task(k, 3, "rich").seed for k in range(700, 732)}; self.assertEqual(len(new), 32); self.assertFalse(new & used); self.assertEqual(DX.down_task(700, 2, "standard").split, "dev")
    def test_no_oracle_in_type_free_arms(self):
        s = inspect.getsource(OX.chain); body = s[s.index('elif arm == "flat_scene_k3"'): s.index('elif arm == "po_type_k3"')]; self.assertNotIn("tl", body.replace("literals", "")); self.assertNotIn("atom ==", body); self.assertIn("validated=None", s)
    def test_sealed_and_pipeline_untouched(self):
        for m in (OW, OX, BR): s = inspect.getsource(m); self.assertNotIn("sealed_task", s); self.assertNotIn('"sealed"', s); self.assertNotIn('"test"', s)
        s = inspect.getsource(DS); self.assertNotIn("deep_ownership", s); self.assertIsNone(inspect.signature(DS.DeepSearch.__init__).parameters["validated"].default)
    def test_calibration_and_summarize(self):
        c = BR.calibration([[[0, 0, 0]] * 9 + [[10, 9.0, 9.5]], None]); self.assertEqual(c["n"], 10); self.assertAlmostEqual(c["ece"], 0.05); self.assertIsNone(BR.calibration([])["ece"]); rows = []
        for kd in ("disc", "sector"):
            sc = OX.scene_at(0, kd, 0.4, 0.0, tune=True); tag = {"scene": f"0_{kd}"}; rows += [{"experiment": "attr", **tag, **r} for r in OX.attr_rows(sc)] + [{"experiment": "oracle", **tag, **OX.oracle_row(sc)}, {"experiment": "vis", **tag, **OX.vis_row(sc)}]
            if kd == "sector": rows.append({"experiment": "sector", **tag, **OX.sector_row(sc)})
        rows += [{"experiment": "chain", "arm": a, "task_id": str(i), "depth": 2, "type_recall": 0.5, "primitive_recovery": 0.5, "candidate_functional": True, "final_095": True, "final_full": 0.9 + 0.02 * (a != "reference"), "own_f1": 0.5, "comp_f1": 0.6, "own_top1": 0.4, "comp_top1": 0.5, "leaves": [{"type": "norm.le", "type_hit": True, "recovered": False}]} for a in OX.PREREGISTERED for i in range(4)]
        S = BR.summarize(rows); self.assertIn("solver", S["attribution"]); self.assertIn("solver − tangent", S["attribution_paired"]); self.assertIn("all", S["oracle"]); self.assertIn("40", S["sector"]); self.assertAlmostEqual(S["chain_paired"]["flat_scene_k3 − reference"]["final_full"]["diff"], 0.02)
        self.assertAlmostEqual(S["composition_reattribution"]["f1"]["diff"], 0.1); BR.show(S)


if __name__ == "__main__":
    unittest.main()

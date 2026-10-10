"""Sous-jalon High-Precision Primitive Estimation : plancher par cellules, raffinement conjoint, naissance, validation,
estimateurs spécialisés, incertitude, et garde-fous (rien n'est branché par défaut, lot scellé jamais touché)."""
import math
import random
import unittest

import numpy as np
import torch

from core.boundary_program import BNode, predict
from core.deep_fit import DT
from core.deep_fragments import SKELETONS
from core.deep_hp import (IDENT, MORPHOLOGIES, assemble, birth, bits_of, cell_code, consensus, cv_floor, describe, extract_circles, extract_lines, families, floor_batch, floor_prediction, hp_pool, improve_set, joint_refine, load_config, prune,
                          recover, residual_morphology, sample_params, set_floor, strip, transfer_error)
from core.deep_world import obs_region

HALF = BNode("pred", "dot", "le", (0.4, -0.3))
DISC = BNode("translate", params=(-1.0, 0.5), children=(BNode("pred", "norm", "le", (1.1,)),))
RING = BNode("translate", params=(-1.2, -0.2), children=(BNode("pred", "norm", "in", (0.7, 1.5)),))
SECT = BNode("translate", params=(-1.5, 0.0), children=(BNode("pred", "angle", "in", (0.2, 0.8)),))
XOR = BNode("xor", children=(HALF, DISC))


def scene(prog, n=600, noise=0.0, seed=3):
    g = torch.Generator().manual_seed(seed); X = obs_region(n, g); y = predict(prog, X); flip = torch.rand(n, generator=g) < noise; return X, torch.where(flip, 1 - y, y)


def rel(a, p):
    P = obs_region(3000, torch.Generator().manual_seed(5)); x = predict(a, P); d = float((x != predict(p, P)).float().mean()); d = min(d, 1 - d); m = float(x.float().mean()); return d / min(m, 1 - m)


class FloorTest(unittest.TestCase):
    def test_floor_matches_reference(self):
        from core.deep_e2e import floor_errors
        X, y = scene(XOR, 300, 0.05); Xd = X.to(DT)[None]; b = [bits_of(HALF, Xd), bits_of(DISC, Xd)]
        self.assertAlmostEqual(set_floor(b, y), float(floor_errors([x.numpy().astype(bool) for x in b], y.numpy().astype(bool))), places=9)
        self.assertAlmostEqual(set_floor([], y), float(min(y.sum(), len(y) - y.sum())), places=9)

    def test_floor_is_polarity_free(self):
        X, y = scene(XOR, 300, 0.05); Xd = X.to(DT)[None]; b = [bits_of(HALF, Xd), bits_of(DISC, Xd)]
        self.assertEqual(set_floor(b, y), set_floor([1 - b[0], b[1]], y)); self.assertTrue(torch.equal(bits_of(BNode("not", children=(DISC,)), Xd), b[1]))

    def test_exact_atoms_explain_a_clean_scene(self):
        X, y = scene(XOR); Xd = X.to(DT)[None]; b = [bits_of(HALF, Xd), bits_of(DISC, Xd)]
        self.assertEqual(set_floor(b, y), 0.0); self.assertTrue(torch.equal(floor_prediction(b, y), y)); self.assertGreater(set_floor(b[:1], y), 20)

    def test_batch_agrees_with_single(self):
        X, y = scene(XOR, 300, 0.05); Xd = X.to(DT)[None]; b = [bits_of(HALF, Xd), bits_of(DISC, Xd)]; B = torch.stack([b[1], 1 - b[1], torch.zeros_like(b[1])]).to(DT); f = floor_batch(B, cell_code(b[:1], len(y)), y, 2)
        self.assertAlmostEqual(float(f[0]), set_floor(b, y)); self.assertAlmostEqual(float(f[1]), set_floor(b, y)); self.assertAlmostEqual(float(f[2]), set_floor(b[:1], y))

    def test_cross_validated_floor(self):
        X, y = scene(XOR, 300, 0.05); Xd = X.to(DT)[None]; b = [bits_of(HALF, Xd), bits_of(DISC, Xd)]; a = cv_floor(b, y, seed=1)
        self.assertEqual(a, cv_floor(b, y, seed=1)); self.assertTrue(0.0 <= a <= 0.2); self.assertGreater(cv_floor(b[:1], y, seed=1), a)

    def test_transfer_error(self):
        X, y = scene(XOR, 600, 0.0); self.assertEqual(transfer_error([HALF, DISC], X[:400], y[:400], X[400:], y[400:]), 0.0); self.assertGreater(transfer_error([HALF], X[:400], y[:400], X[400:], y[400:]), 0.05)


class RefineTest(unittest.TestCase):
    def test_never_worse_and_initial_point_kept(self):
        X, y = scene(XOR, 300, 0.05); P, info = joint_refine([HALF, DISC], X, y, 0)
        self.assertLessEqual(info["floor_after"], info["floor_before"]); self.assertEqual(len(P), 2)
        Q, i0 = joint_refine([HALF, DISC], X, y, 0, sweeps=0); self.assertEqual([repr(q) for q in Q], [repr(HALF), repr(DISC)]); self.assertEqual(i0["moves"], 0)

    def test_brings_perturbed_atoms_back(self):
        X, y = scene(XOR, 600, 0.0); bad = [BNode("pred", "dot", "le", (0.5, -0.2)), BNode("translate", params=(-0.9, 0.42), children=(BNode("pred", "norm", "le", (1.0,)),))]; P, info = joint_refine(bad, X, y, 0, margin=0.0)
        self.assertLess(info["floor_after"], info["floor_before"]); self.assertLess(rel(HALF, P[0]) + rel(DISC, P[1]), rel(HALF, bad[0]) + rel(DISC, bad[1]))

    def test_deterministic(self):
        X, y = scene(XOR, 300, 0.05); a, _ = joint_refine([HALF, DISC], X, y, 4); b, _ = joint_refine([HALF, DISC], X, y, 4); self.assertEqual([repr(p) for p in a], [repr(p) for p in b])


class BirthTest(unittest.TestCase):
    def test_missing_atom_is_born_from_conditional_gain(self):
        X, y = scene(XOR, 600, 0.0); new, info = birth([HALF], X, y, None, 0)
        self.assertIsNotNone(new); self.assertTrue(info["accepted"]); self.assertGreater(info["gain"], 20); self.assertLess(rel(DISC, new), 0.3); self.assertIn(info["morphology"], MORPHOLOGIES)

    def test_no_birth_on_a_complete_set(self):
        X, y = scene(XOR, 600, 0.0); new, info = birth([HALF, DISC], X, y, None, 0); self.assertIsNone(new); self.assertFalse(info["accepted"])

    def test_prune_keeps_xor_operands_and_drops_useless(self):
        X, y = scene(XOR, 600, 0.0); useless = BNode("pred", "dot", "le", (2.0, 5.0)); P = prune([HALF, DISC, useless], X, y)
        self.assertEqual({repr(p) for p in P}, {repr(HALF), repr(DISC)})

    def test_samples_pass_through_boundary_points(self):
        g = torch.Generator().manual_seed(0); M = torch.tensor([[0.5, -1.0]], dtype=DT)
        for t in ("half", "band", "disc", "ring", "sector"):
            C = sample_params(t, 16, M, g); self.assertEqual(C.shape[1], len(SKELETONS[t].with_params([0.0] * SKELETONS[t].n_params()).param_vector()))
        C = sample_params("half", 16, M, g); self.assertLess(float((torch.cos(C[:, 0]) * 0.5 - torch.sin(C[:, 0]) * 1.0 - C[:, 1]).abs().max()), 1e-9)
        C = sample_params("disc", 16, M, g); self.assertLess(float(((C[:, :2] - M).norm(dim=1) - C[:, 2]).abs().max()), 1e-9)


class ImproveTest(unittest.TestCase):
    def test_rollback_returns_the_initial_set(self):
        g = torch.Generator().manual_seed(1); X = obs_region(300, g); y = (torch.rand(300, generator=g) < 0.5).long(); im = improve_set([HALF, DISC], X, y, 0)
        if not im.committed: self.assertEqual([repr(p) for p in im.programs], [repr(HALF), repr(DISC)]); self.assertEqual(im.provenance, ["initial", "initial"])
        self.assertFalse(im.committed)                                       # étiquettes aléatoires : rien à gagner hors-échantillon

    def test_incomplete_set_is_completed_and_validated(self):
        X, y = scene(XOR, 600, 0.02); im = improve_set([HALF], X, y, 0)
        self.assertTrue(im.committed); self.assertLess(im.cv_after, im.cv_before); self.assertLess(im.floor_after, im.floor_before); self.assertTrue(any(p.startswith("birth") for p in im.provenance)); self.assertLess(min(rel(DISC, p) for p in im.programs), 0.3)

    def test_validation_modes(self):
        X, y = scene(XOR, 300, 0.05)
        for mode in ("holdout", "cv", False): im = improve_set([HALF], X, y, 0, {"validate": mode}); self.assertLessEqual(len(im.programs), 6); self.assertEqual(len(im.programs), len(im.provenance))
        self.assertEqual(load_config()["validate"], "holdout")


class CurveTest(unittest.TestCase):
    def test_lines_and_circles_of_a_dense_clean_scene(self):
        from core.deep_partial import observe
        X, y = scene(XOR, 2400, 0.0); ob = observe(X, y); L = extract_lines(ob, 6, 0); C = extract_circles(ob, 6, 0)
        self.assertTrue(any(min(abs(math.remainder(c.params[0] - 0.4, 2 * math.pi)) + abs(c.params[1] + 0.3), abs(abs(math.remainder(c.params[0] - 0.4, 2 * math.pi)) - math.pi) + abs(c.params[1] - 0.3)) < 0.1 for c in L))
        self.assertTrue(any(math.hypot(c.params[0] + 1.0, c.params[1] - 0.5) < 0.1 and abs(c.params[2] - 1.1) < 0.08 for c in C))

    def test_assembly_by_family(self):
        from core.deep_partial import observe
        X, y = scene(RING, 2400, 0.0); ob = observe(X, y); A = assemble(ob, extract_lines(ob, 6, 0), extract_circles(ob, 6, 0), 5); rings = [p for p, _, s in A if s == "curves:ring"]
        self.assertTrue(rings); self.assertLess(min(rel(RING, p) for p in rings), 0.15)
        X, y = scene(SECT, 2400, 0.0); ob = observe(X, y); A = assemble(ob, extract_lines(ob, 6, 0), extract_circles(ob, 6, 0), 5); sect = [p for p, _, s in A if s == "curves:sector"]
        self.assertTrue(sect); self.assertLess(min(rel(SECT, p) for p in sect), 0.2); self.assertTrue(all(0.0 <= s <= 1.0 for _, s, _ in A))

    def test_empty_scene(self):
        from core.deep_partial import observe
        g = torch.Generator().manual_seed(0); X = obs_region(100, g); ob = observe(X, torch.zeros(100, dtype=torch.long)); self.assertEqual(extract_lines(ob), []); self.assertEqual(extract_circles(ob), []); self.assertEqual(assemble(ob, [], []), [])

    def test_residual_morphology(self):
        X, y = scene(XOR, 1200, 0.0); Xd = X.to(DT)[None]
        self.assertIn(residual_morphology(X, floor_prediction([bits_of(DISC, Xd)], y) != y).label, ("linear", "parallel", "radial")); self.assertIn(residual_morphology(X, floor_prediction([bits_of(HALF, Xd)], y) != y).label, ("arc", "concentric"))
        self.assertEqual(residual_morphology(X, torch.zeros(len(y), dtype=torch.bool)).label, "none")


class EstimateTest(unittest.TestCase):
    def test_families(self):
        self.assertEqual(families(HALF), ["orientation", "offset"]); self.assertEqual(families(DISC), ["center", "center", "radius"]); self.assertEqual(families(RING), ["center", "center", "radius", "radius"]); self.assertEqual(families(SECT), ["center", "center", "direction", "half_width"])
        self.assertEqual(families(BNode("not", children=(HALF,))), ["orientation", "offset"]); self.assertEqual(repr(strip(BNode("not", children=(HALF,)))), repr(HALF))

    def test_estimates_expose_uncertainty(self):
        X, y = scene(XOR, 600, 0.02); E = describe([HALF, DISC], X, y, 0)
        self.assertEqual(len(E), 2)
        for e in E:
            self.assertEqual(len(e.central), len(e.uncertainty)); self.assertEqual(len(e.region), len(e.central)); self.assertIn(e.identifiability, IDENT); self.assertTrue(0.0 <= e.stability <= 1.0); self.assertTrue(all(lo <= c <= hi for (lo, hi), c in zip(e.region, e.central))); self.assertIsNotNone(e.consensus)
        few = describe([HALF, DISC], X[:25], y[:25], 0); self.assertGreaterEqual(sum(max(e.uncertainty) for e in few), sum(max(e.uncertainty) for e in E))

    def test_consensus_is_validated(self):
        X, y = scene(XOR, 300, 0.05); P, info = consensus([HALF, DISC], X, y, 0)
        self.assertEqual(len(P), 2); self.assertTrue(info["committed"] or [repr(p) for p in P] == [repr(HALF), repr(DISC)])


class ChainTest(unittest.TestCase):
    def test_recover_runs_and_destroys_nothing(self):
        X, y = scene(XOR, 300, 0.03); pool = hp_pool(X, y, 0); R = recover(X, y, 0, discovered=pool)
        self.assertIsNotNone(R.program); self.assertEqual(R.stats["pool"], len(pool)); self.assertTrue(all(s.startswith("hp:") for _, _, s in R.added)); self.assertGreater(float((predict(R.program, X.float()) == y).float().mean()), 0.85)
        self.assertTrue(any(s.startswith("curves:") for _, _, s in pool)); self.assertTrue(any(not s.startswith("curves:") for _, _, s in pool))

    def test_nothing_is_plugged_by_default(self):
        import inspect
        import core.deep_e2e, core.deep_lpr, core.deep_search
        from core.deep_search import DeepSearch
        self.assertIsNone(inspect.signature(DeepSearch.__init__).parameters["validated"].default)
        for m in (core.deep_search, core.deep_e2e, core.deep_lpr): self.assertNotIn("deep_hp", inspect.getsource(m))

    def test_system_module_sees_no_truth(self):
        import inspect
        import core.deep_hp as m
        src = inspect.getsource(m); self.assertNotIn("deep_world", src); self.assertNotIn("sealed", src); self.assertNotIn("world.truth", src)


class ExperimentTest(unittest.TestCase):
    def test_relative_error_is_polarity_free(self):
        from core.deep_hp_exp import _probe, family_errors, rel_error
        P = _probe(); anti = BNode("translate", params=(-1.0, 0.5), children=(BNode("pred", "norm", "ge", (1.1,)),))
        self.assertEqual(rel_error(DISC, DISC, P), 0.0); self.assertEqual(rel_error(DISC, anti, P), 0.0); self.assertGreater(rel_error(DISC, HALF, P), 0.5)
        self.assertEqual(family_errors(DISC, anti), {"center": 0.0, "radius": 0.0}); self.assertEqual(family_errors(DISC, RING), {}); self.assertEqual(family_errors(DISC, HALF), {})

    def test_perturbation_touches_one_family(self):
        from core.deep_hp_exp import perturb
        rng = random.Random(0); q = perturb(DISC, "radius", 0.1, rng); self.assertEqual(q.param_vector()[:2], DISC.param_vector()[:2]); self.assertAlmostEqual(abs(q.param_vector()[2] - 1.1), 0.1)
        q = perturb(DISC, "center", 0.1, rng); self.assertAlmostEqual(math.hypot(q.param_vector()[0] + 1.0, q.param_vector()[1] - 0.5), 0.1); self.assertEqual(q.param_vector()[2], 1.1)
        q = perturb(RING, "width", 0.2, rng); self.assertAlmostEqual(abs((q.param_vector()[3] - q.param_vector()[2]) - 0.8), 0.2); self.assertIsNone(perturb(DISC, "width", 0.1, rng)); self.assertIsNone(perturb(HALF, "radius", 0.1, rng))
        q = perturb(HALF, "orientation", 0.1, rng); self.assertAlmostEqual(abs(q.param_vector()[0] - 0.4), 0.1)

    def test_tolerance_rule(self):
        from core.deep_hp_report import tolerances
        rows = [{"kind": "norm.le", "family": "radius", "delta": d, "base_functional": True, "atom_rel": d, "functional": f, "functional_097": True, "accord": 0.99} for d, fs in ((0.02, (1, 1, 1, 1, 1)), (0.05, (1, 1, 1, 1, 0)), (0.1, (1, 0, 0, 0, 0)), (0.2, (1, 1, 1, 1, 1))) for f in fs]
        self.assertEqual(tolerances(rows)["norm.le/radius"]["tolerance"], 0.05)
        self.assertIsNone(tolerances([{**r, "functional": 0} for r in rows])["norm.le/radius"]["tolerance"])

    def test_dense_regime_shares_the_world(self):
        from core.deep_hp_exp import DENSE, task_at
        a, b = task_at(1090, 2, "standard"), task_at(1090, 2, "dense"); self.assertEqual(repr(a.world.truth), repr(b.world.truth)); self.assertEqual(len(b.y), DENSE[0]); self.assertEqual(b.split, "dev")

    def test_rows_and_summary(self):
        from core.deep_hp_exp import birth_rows, chain_rows, sens_rows, subunits, task_at
        from core.deep_hp_report import summarize
        T = task_at(1090, 2, "standard"); rows = [{"task_id": "t", "depth": 2, "regime": "standard", **r} for r in chain_rows(T, 0, arms=("B", "C"), historical=False) + sens_rows(T, 0, max_atoms=1) + birth_rows(T, 0)]
        self.assertEqual({r["arm"] for r in rows if r["experiment"] == "chain"}, {"B", "C"}); self.assertTrue(all("pool_rel10" in r and "atoms_hp" in r for r in rows if r["experiment"] == "chain"))
        self.assertTrue({"complete", "new", "old", "new_nomorph"} <= {r["arm"] for r in rows if r["experiment"] == "birth"}); S = summarize(rows); self.assertIn("primary/all/C", S["chain"]); self.assertIn("exact/new", S["birth"]); self.assertTrue(S["sensitivity"])
        names = [n for n, _ in subunits(1100)]; self.assertEqual(len(names), len(set(names))); self.assertEqual(sum(n.startswith("chain_") for n in names), 18); self.assertEqual(sum(n.startswith("single_") for n in names), 6); self.assertEqual(sum(n.startswith("ood_") for n in names), 6)

    def test_kind_is_registered_and_sealed_set_untouched(self):
        import inspect
        import core.deep_hp_exp as m
        from core.deep_experiments import SUBUNITS
        from core.deep_world import SEALED_SHA256, sealed_hash
        self.assertIn("deephp", SUBUNITS); self.assertNotIn("sealed_task", inspect.getsource(m)); self.assertEqual(sealed_hash(), SEALED_SHA256)


if __name__ == "__main__":
    unittest.main()

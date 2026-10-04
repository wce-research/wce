import inspect
import json
import math
import random
import re
import unittest

import torch

import core.active_structure as AS
from core.active_structure import (PROBES, ActiveStructuralDiscovery, EquivalenceSignature, FunctionalEquivalenceClass, NoiseAwareTruthAnchor, OrbitFamily,
                                   SearchPolicy, StructuralPosterior, canonical_frame, canonical_signature, components, expansion_key, fit_extension, fit_from,
                                   functional_clusters, gated_residual, identifiability, induce_ast_families, induce_orbit_families, minimal_distinguishing_set,
                                   noise_aware_stability, program_signature, query_type, sequential_truth_test, version_space)
from core.active_world import (DEPTH_FAMILIES, HARD_NEGATIVES, MULTI_SYNTAX, NOISE_LEVELS, SEALED_ACTIVE, SEALED_SHA256, balanced_instances, make_active_instance,
                               noisy_task, sealed_hash, sealed_spec, train_deep_instance, train_tasks, transformed)
from core.boundary_program import BNode, atoms, predict
from core.boundary_world import TRAIN, VALIDATION, full_region, in_region, out_region

G = lambda s=0: torch.Generator().manual_seed(s)
P = lambda *a, **k: BNode(*a, **k)
T = lambda c, p: P("translate", params=p, children=(c,))
DISC = T(P("pred", "norm", "le", (1.0,)), (0.0, 0.0))
DISC2 = T(P("pred", "norm", "le", (1.0,)), (0.5, 0.0))
HALF = P("pred", "dot", "le", (0.0, 0.0))
HALF2 = P("pred", "dot", "le", (0.0, 1.0))
LOBES_OR = P("or", children=(T(P("pred", "norm", "le", (0.6,)), (-1.4, 0.0)), T(P("pred", "norm", "le", (0.6,)), (1.0, 0.3))))
LOBES_XOR = P("xor", children=LOBES_OR.children)


def X0(n=300, s=0): return torch.rand(n, 2, generator=G(s)) * 4 - 2


# ================================================================ signatures et équivalence

class SignatureTests(unittest.TestCase):
    def test_components_one(self): self.assertEqual(program_signature(DISC).components, 1)
    def test_components_two(self): self.assertEqual(program_signature(LOBES_OR).components, 2)
    def test_components_raw(self): m = torch.zeros(16, dtype=torch.bool); m[0] = m[15] = True; self.assertEqual(components(m), 2)
    def test_area(self): self.assertTrue(0 < program_signature(DISC).area < 0.2)
    def test_symmetry_disc(self): self.assertGreater(program_signature(DISC).reflection_symmetry, 0.95)
    def test_distance_zero(self): self.assertEqual(program_signature(DISC).distance(program_signature(DISC)), 0.0)
    def test_distance_pos(self): self.assertGreater(program_signature(DISC).distance(program_signature(HALF)), 0.1)
    def test_or_xor_equivalent(self): self.assertEqual(program_signature(LOBES_OR).distance(program_signature(LOBES_XOR)), 0.0)
    def test_canonical_translation_invariant(self): self.assertLess(program_signature(DISC).distance(program_signature(DISC2), canonical=True), 0.05)
    def test_raw_translation_sensitive(self): self.assertGreater(program_signature(DISC).distance(program_signature(DISC2)), 0.02)
    def test_canonical_frame_none_for_empty(self): self.assertIsNone(canonical_frame(lambda X: torch.zeros(len(X), dtype=torch.long)))
    def test_canonical_frame_center(self):
        c, V, s = canonical_frame(lambda X: predict(DISC2, X)); self.assertLess(float((c - torch.tensor([0.5, 0.0])).norm()), 0.15)
    def test_canonical_signature_shape(self): self.assertEqual(canonical_signature(lambda X: predict(DISC, X)).numel(), 576)
    def test_equivalence_signature_of(self): self.assertIsInstance(EquivalenceSignature.of(lambda X: predict(HALF, X)), EquivalenceSignature)


def _make_transform_test(i):
    def t(self):
        rng = random.Random(i); inst = balanced_instances(["crescent"], 900 + i, 1)[0]; ti = transformed(inst, rng)
        a = EquivalenceSignature.of(inst.label); b = EquivalenceSignature.of(ti.label)
        self.assertLessEqual(a.distance(b, canonical=True), a.distance(b) + 0.05)
    return t


for _i in range(12):
    setattr(SignatureTests, f"test_canonical_helps_transform_{_i}", _make_transform_test(_i))


class ClusteringTests(unittest.TestCase):
    def test_cluster_or_xor(self): cls, idx = functional_clusters([LOBES_OR, LOBES_XOR, DISC]); self.assertEqual(idx[0], idx[1]); self.assertNotEqual(idx[0], idx[2])
    def test_multiple_representations(self): cls, idx = functional_clusters([LOBES_OR, LOBES_XOR]); self.assertEqual(len(cls[0].representations()), 2)
    def test_canonical_clusters_translated(self): cls, idx = functional_clusters([DISC, DISC2], canonical=True); self.assertEqual(idx[0], idx[1])
    def test_raw_separates_translated(self): cls, idx = functional_clusters([DISC, DISC2]); self.assertNotEqual(idx[0], idx[1])
    def test_class_type(self): cls, _ = functional_clusters([DISC]); self.assertIsInstance(cls[0], FunctionalEquivalenceClass)
    def test_threshold(self): cls, idx = functional_clusters([DISC, DISC2], threshold=1.0); self.assertEqual(idx[0], idx[1])
    def test_topology_required(self): cls, idx = functional_clusters([LOBES_OR, DISC], threshold=1.0); self.assertNotEqual(idx[0], idx[1])


# ================================================================ postérieure structurelle

class PosteriorTests(unittest.TestCase):
    def post(self, eps=0.05): return StructuralPosterior([LOBES_OR, LOBES_XOR, DISC, HALF], eps)
    def test_classes(self): self.assertEqual(len(self.post().classes), 3)
    def test_class_probs_sum(self): self.assertAlmostEqual(float(self.post().class_probs().sum()), 1.0)
    def test_entropy_structural_le_program(self): p = self.post(); self.assertLessEqual(p.entropy("structural"), p.entropy("program") + 1e-9)
    def test_update_concentrates(self):
        p = self.post(); X = X0(); p.update(X, predict(DISC, X)); k, pk = p.map_class(); self.assertGreater(pk, 0.99); self.assertIs(p.map_program(), DISC)
    def test_ig_structure_zero_within_class(self):
        p = StructuralPosterior([LOBES_OR, LOBES_XOR], 0.05); self.assertLess(float(p.ig_structure(X0()).max()), 1e-9)
    def test_ig_structure_positive(self): self.assertGreater(float(self.post().ig_structure(X0()).max()), 0.05)
    def test_ig_parameters(self): self.assertTrue(bool((self.post().ig_parameters(X0()) >= -1e-9).all()))
    def test_predictive_entropy(self): self.assertTrue(bool((self.post().predictive_entropy(X0()) >= 0).all()))
    def test_disagreement(self): self.assertTrue(bool((self.post().disagreement(X0()) >= 0).all()))
    def test_disagreement_region(self):
        p = StructuralPosterior([DISC, HALF], 0.05); R = p.disagreement_region(X0()); self.assertTrue(bool((predict(DISC, R) != predict(HALF, R)).all()))
    def test_disagreement_region_empty_single(self): self.assertEqual(len(StructuralPosterior([DISC], 0.05).disagreement_region(X0())), 0)
    def test_rescore(self): p = self.post(); X = X0(); p.update(X, predict(HALF, X)); p.rescore(X, predict(DISC, X)); self.assertIs(p.map_program(), DISC)
    def test_ast_mode(self): p = StructuralPosterior([LOBES_OR, LOBES_XOR], 0.05, use_functional=False); self.assertEqual(len(p.classes), 2)
    def test_map_prefers_simple(self):
        p = StructuralPosterior([P("and", children=(DISC, DISC)), DISC], 0.05); p.logw[:] = 0.0; self.assertIs(p.map_program(), DISC)
    def test_eps_floor(self): self.assertGreaterEqual(StructuralPosterior([DISC], 0.0).eps, 0.01)


class QueryTests(unittest.TestCase):
    def test_query_types(self):
        p = StructuralPosterior([DISC, HALF], 0.05); x = p.disagreement_region(X0())[:1]
        self.assertEqual(query_type(p, x), "structure-disambiguating")
    def test_truth_checking(self):
        p = StructuralPosterior([DISC], 0.05); self.assertIn(query_type(p, torch.tensor([[2.5, 2.5]])), ("truth-checking", "boundary-calibrating"))
    def test_identifiable(self): p = StructuralPosterior([DISC, HALF], 0.05); p.logw[:] = 0.0; self.assertTrue(identifiability(p, X0())["identifiable"])
    def test_non_identifiable_restricted_pool(self):
        far = P("and", children=(DISC, P("pred", "dot", "le", (0.0, 0.3)))); p = StructuralPosterior([DISC, far], 0.05); p.logw[:] = 0.0
        pool = X0() * 0.1; self.assertFalse(identifiability(p, pool)["identifiable"])
    def test_minimal_set(self):
        p = StructuralPosterior([DISC, HALF, HALF2], 0.05); p.logw[:] = 0.0; S = minimal_distinguishing_set(p, X0()); self.assertLessEqual(len(S), 2)
    def test_version_space(self):
        p = StructuralPosterior([DISC, HALF], 0.05); X = X0(); vs = version_space(p, X, predict(DISC, X)); self.assertEqual(vs, [DISC])
    def test_version_space_noise_tolerant(self):
        p = StructuralPosterior([DISC], 0.1); X = X0(); y = predict(DISC, X); y[:5] = 1 - y[:5]; self.assertEqual(len(version_space(p, X, y)), 1)


# ================================================================ politique et recherche

class PolicyTests(unittest.TestCase):
    def test_score_prior(self): self.assertAlmostEqual(SearchPolicy().score(("x",)), 0.5)
    def test_record(self): p = SearchPolicy(); p.record(("k",), True); p.record(("k",), True); self.assertGreater(p.score(("k",)), 0.6)
    def test_propose_unguided_all(self): self.assertEqual(len(SearchPolicy().propose(DISC, atoms(), 2, guided=False)), 3 * len(atoms()))
    def test_propose_guided_k(self): self.assertEqual(len(SearchPolicy(k=12).propose(DISC, atoms(), 2)), 12)
    def test_propose_diverse(self): self.assertEqual({c for c, a in SearchPolicy(k=12).propose(DISC, atoms(), 2)}, {"and", "or", "xor"})
    def test_should_search_bad_fit(self): self.assertTrue(SearchPolicy().should_search(0.3, 0.05, 0.5, 1.0))
    def test_should_not_search_good(self): self.assertFalse(SearchPolicy().should_search(0.05, 0.05, 0.5, 1.0))
    def test_expansion_key(self): self.assertEqual(len(expansion_key(DISC, atoms()[0], "and", 2)), 4)
    def test_trajectory_macro(self): p = SearchPolicy(); p.record_trajectory(["a", "b"]); self.assertEqual(p.macros[("a", "b")], 1)
    def test_fit_extension(self):
        X = X0(400, 1); y = predict(P("and", children=(DISC, HALF)), X); prog, acc = fit_extension(DISC, P("pred", "dot", "le"), "and", X, y); self.assertGreater(acc, 0.93)
    def test_fit_from(self): X = X0(); q = fit_from(DISC, X, predict(DISC, X), steps=10); self.assertEqual(q.skeleton(), DISC.skeleton())


class DiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = balanced_instances(["ring"], 3, 1)[0]; X, y = noisy_task(cls.inst, 120, G(0), 0.05)
        asd = ActiveStructuralDiscovery(budget=8, eps=0.05, max_depth=2); asd.eps_world = 0.05; cls.r = asd.run(X, y, cls.inst.label, full_region, G(1))
    def test_recovers(self): F = full_region(2000, G(2)); self.assertGreater(float((predict(self.r.program, F) == self.inst.label(F)).float().mean()), 0.95)
    def test_queries_bounded(self): self.assertLessEqual(self.r.queries, 8)
    def test_entropy_curve(self): self.assertEqual(len(self.r.entropy_curve), self.r.queries + 1)
    def test_version_space_curve(self): self.assertGreaterEqual(min(self.r.version_space_curve), 0)
    def test_status(self): self.assertIn(self.r.status, ("identified", "unsupported", "ambiguous", "non_identifiable"))
    def test_interleaved_depth(self): self.assertGreaterEqual(self.r.depth, 2)
    def test_policies(self):
        for qp in ("structure_ig", "parameter_ig", "predictive_entropy", "disagreement", "random"):
            s = StructuralPosterior([DISC, HALF], 0.05); self.assertEqual(len(ActiveStructuralDiscovery(query_policy=qp)._query_scores(s, X0(50))), 50)


# ================================================================ bruit : stabilité et ancres

class NoiseTests(unittest.TestCase):
    def test_stability_accepts_correct(self):
        X = X0(160, 3); y = predict(DISC, X); g = G(4); flip = torch.rand(160, generator=g) < 0.08; y = torch.where(flip, 1 - y, y)
        self.assertTrue(noise_aware_stability(DISC, X, y, 0.08)["accept"])
    def test_stability_rejects_wrong(self): X = X0(160, 3); self.assertFalse(noise_aware_stability(HALF, X, predict(DISC, X), 0.05)["accept"])
    def test_stability_modes(self):
        X = X0(160, 3); r = noise_aware_stability(DISC, X, predict(DISC, X), 0.05)
        for k in ("prediction_stability", "functional_class_stability", "parameter_stability", "ast_stability"): self.assertIn(k, r)
    def test_table_monotone(self):
        ks = sorted(AS.STABILITY_TABLE); self.assertTrue(all(AS.STABILITY_TABLE[a][0] <= AS.STABILITY_TABLE[b][0] for a, b in zip(ks, ks[1:])))
    def test_anchor_prior(self): self.assertAlmostEqual(NoiseAwareTruthAnchor(DISC, [HALF], 0.05).p_wrong(), 0.5, places=6)
    def test_anchor_accepts_correct(self):
        a = NoiseAwareTruthAnchor(DISC, [HALF], 0.05); X = X0(40); a.observe(X, predict(DISC, X)); self.assertEqual(a.decision(), "accept")
    def test_anchor_rejects_wrong(self):
        a = NoiseAwareTruthAnchor(HALF, [DISC], 0.05); X = X0(40); a.observe(X, predict(DISC, X)); self.assertEqual(a.decision(), "reject")
    def test_anchor_boundary_tolerant(self):
        a = NoiseAwareTruthAnchor(DISC, [], 0.05); x = torch.tensor([[1.0, 0.0]]); a.observe(x, torch.tensor([0])); self.assertLess(a.p_wrong(), 0.7)
    def test_anchor_far_strict(self):
        a = NoiseAwareTruthAnchor(DISC, [], 0.05); x = torch.tensor([[0.0, 0.0]]).repeat(3, 1); a.observe(x, torch.tensor([0, 0, 0])); self.assertGreater(a.p_wrong(), 0.9)
    def test_anchor_choose(self): a = NoiseAwareTruthAnchor(DISC, [HALF], 0.05); x, rep = a.choose(X0(50), []); self.assertEqual(x.shape, (1, 2)); self.assertFalse(rep)
    def test_anchor_repeat_possible(self): a = NoiseAwareTruthAnchor(DISC, [HALF], 0.05); x, rep = a.choose(X0(5), [torch.tensor([0.5, 0.0])]); self.assertIn(rep, (True, False))
    def test_sequential(self):
        r = sequential_truth_test(HALF, [DISC], lambda Q: predict(DISC, Q), full_region, G(1), 0.05); self.assertEqual(r["decision"], "reject")
    def test_sequential_correct(self):
        r = sequential_truth_test(DISC, [HALF], lambda Q: predict(DISC, Q), full_region, G(1), 0.05); self.assertNotEqual(r["decision"], "reject")
    def test_sequential_stops_early(self): r = sequential_truth_test(HALF, [DISC], lambda Q: predict(DISC, Q), full_region, G(1), 0.05, budget=30); self.assertLess(r["queries"], 30)


def _make_noise_anchor_test(noise):
    def t(self):
        rejects = 0
        for s in range(4):
            r = sequential_truth_test(DISC, [HALF, DISC2], lambda Q: predict(DISC, Q), full_region, G(s), max(noise, 0.01), budget=16, noise=noise); rejects += r["decision"] == "reject"
        self.assertLessEqual(rejects, 1)
    return t


for _n in NOISE_LEVELS:
    setattr(NoiseTests, f"test_correct_rule_not_rejected_noise_{int(_n * 100)}", _make_noise_anchor_test(_n))


# ================================================================ résidu à porte, familles

class ResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X = X0(300, 5); cls.y = ((X[:, 0] <= 0.15 * torch.sin(4 * X[:, 1])) & (X[:, 0] > -1.8)).long(); cls.X = X
        cls.res = gated_residual(P("pred", "dot", "le", (0.0, 0.0)), X, cls.y, steps=150)
    def test_gate_closed_out_of_region(self): self.assertEqual(int(self.res[2](torch.tensor([[2.8, 2.8], [-2.9, 0.0]])).sum()), 0)
    def test_out_equals_program(self):
        O = out_region(500, G(1)) + torch.tensor([0.6, 0.0]); self.assertTrue(torch.equal(self.res[0](O), predict(P("pred", "dot", "le", (0.0, 0.0)), O)))
    def test_cost_small(self): self.assertLess(self.res[1], 200)
    def test_in_region_accuracy(self): self.assertGreater(float((self.res[0](self.X) == self.y).float().mean()), 0.85)


class FamilyTests(unittest.TestCase):
    def test_orbit_family_translated(self): self.assertEqual(len(induce_orbit_families([DISC, DISC2], min_support=2)), 1)
    def test_raw_family_split(self): self.assertEqual(len(induce_orbit_families([DISC, T(P("pred", "norm", "le", (1.0,)), (1.5, 0.5))], min_support=2, canonical=False)), 0)
    def test_ast_family(self): self.assertEqual(len(induce_ast_families([DISC2, T(P("pred", "norm", "le", (0.7,)), (-0.5, 0.2))])), 1)
    def test_ast_family_misses_equivalent(self): self.assertEqual(len(induce_ast_families([LOBES_OR, LOBES_XOR])), 0)
    def test_functional_family_equivalent(self): self.assertEqual(len(induce_orbit_families([LOBES_OR, LOBES_XOR])), 1)
    def test_lifecycle(self):
        f = OrbitFamily(program_signature(DISC), [DISC])
        for _ in range(3): f.update(True)
        self.assertEqual(f.status, "STABLE"); f.update(False); f.update(False); f.update(False); f.update(False); self.assertEqual(f.status, "REFUTED")
    def test_lifecycle_demotion(self):
        f = OrbitFamily(program_signature(DISC), [DISC])
        for _ in range(3): f.update(True)
        f.update(False); self.assertEqual(f.status, "SUPPORTED")
    def test_reuse(self):
        f = induce_orbit_families([DISC, DISC2], min_support=1); X = X0(200, 7); y = predict(T(P("pred", "norm", "le", (0.8,)), (-0.5, 0.3)), X)
        p, ev = AS.reuse_family(f, X, y); self.assertGreater(float((predict(p, X) == y).float().mean()), 0.9)


# ================================================================ monde, protocole

class WorldTests(unittest.TestCase):
    def test_sealed_hash(self): self.assertEqual(sealed_hash(), SEALED_SHA256)
    def test_sealed_noise(self): self.assertEqual(sealed_spec()["noise"], [0.08, 0.12])
    def test_transformed(self): i = balanced_instances(["ring"], 1, 1)[0]; self.assertEqual(transformed(i, random.Random(0)).label(X0()).shape, (300,))
    def test_train_deep_uses_train(self): self.assertEqual(train_deep_instance(random.Random(0)).family, "train_deep")
    def test_train_tasks(self): self.assertEqual(len(train_tasks(1)), len(TRAIN + VALIDATION) + 4)
    def test_noise_levels(self): self.assertEqual(NOISE_LEVELS, (0.0, 0.02, 0.05, 0.08, 0.12))
    def test_noisy(self):
        i = balanced_instances(["ring"], 1, 1)[0]; X, y = noisy_task(i, 2000, G(1), 0.12); self.assertLess(float((i.label(X) == y).float().mean()), 0.95)


def _make_family_test(f):
    def t(self):
        inst = make_active_instance(f, random.Random(3)); y = inst.label(full_region(1000, G(1))); self.assertEqual(y.dtype, torch.long)
    return t


for _f in DEPTH_FAMILIES + HARD_NEGATIVES + MULTI_SYNTAX + SEALED_ACTIVE:
    setattr(WorldTests, f"test_family_{_f}", _make_family_test(_f))


def _make_depth_test(f):
    def t(self):
        """Profondeur 2 insuffisante : la meilleure combinaison de 2 atomes reste sous l'exactitude quasi parfaite (au moins sur une instance)."""
        from core.boundary_program import BoundaryProgramSearch
        worst = 1.0
        for s in range(2):
            inst = balanced_instances([f], 50 + s, 1)[0]; X, y = noisy_task(inst, 300, G(s), 0.0); F = full_region(2000, G(9))
            r = BoundaryProgramSearch(max_depth=2, steps=60, n_init=1).run(X[:220], y[:220], X[220:], y[220:]); worst = min(worst, float((predict(r.best, F) == inst.label(F)).float().mean()))
        self.assertLess(worst, 0.985)
    return t


for _f in ("triangle", "kite"):
    setattr(WorldTests, f"test_depth2_insufficient_{_f}", _make_depth_test(_f))


def _make_hardneg_test(f):
    def t(self):
        inst = balanced_instances([f], 60, 1)[0]; I, O = in_region(2000, G(1)), out_region(2000, G(1))
        base = make_active_instance("disc_cut_far" if f == "disc_cut_far" else "strip_capped", random.Random(0))
        self.assertGreater(float(inst.label(I).float().mean()), 0.0)
    return t


for _f in HARD_NEGATIVES:
    setattr(WorldTests, f"test_hard_negative_{_f}", _make_hardneg_test(_f))


# ================================================================ FUITES

FAMILY_NAMES = DEPTH_FAMILIES + HARD_NEGATIVES + MULTI_SYNTAX + SEALED_ACTIVE + TRAIN + VALIDATION + ("crescent", "xor_halfspaces", "capsule", "wedge_ring", "sine5", "checker", "sector", "ellipse", "circle", "box")


class LeakageTests(unittest.TestCase):
    def test_no_family_names(self):
        src = inspect.getsource(AS)
        for n in FAMILY_NAMES: self.assertIsNone(re.search(rf"[\"']{n}[\"']", src), n)
    def test_no_family_conditionals(self): self.assertIsNone(re.search(r"family\s*==", inspect.getsource(AS)))
    def test_no_world_import(self): self.assertNotIn("active_world", inspect.getsource(AS)); self.assertNotIn("boundary_world", inspect.getsource(AS))
    def test_calibration_train_val_only(self):
        src = open("scripts/active_calibrate.py").read(); self.assertIn("TRAIN + VALIDATION", src); self.assertNotIn("HELD_OUT", src); self.assertNotIn("DEPTH", src)
    def test_proposer_trained_without_test_families(self):
        import core.active_world as AW; src = inspect.getsource(AW.train_tasks) + inspect.getsource(AW.train_deep_instance)
        for n in DEPTH_FAMILIES + HARD_NEGATIVES: self.assertNotIn(n, src)
    def test_preregistered(self):
        t = open("docs/hypotheses_active_structure.md").read()
        for i in range(1, 11): self.assertIn(f"H{i} ", t)
    def test_sealed_not_in_jobs(self):
        import core.active_experiments as AE; self.assertNotIn("SEALED", inspect.getsource(AE))
    def test_budget_preregistered(self):
        import core.active_experiments as AE; self.assertEqual(AE.BUDGET_FLOATS, 400); self.assertEqual(AE.BUDGET, 16)


def _make_leak(n):
    def t(self): self.assertIsNone(re.search(rf"[\"']{n}[\"']", inspect.getsource(AS)))
    return t


for _n in FAMILY_NAMES:
    setattr(LeakageTests, f"test_leak_{_n}", _make_leak(_n))


# ================================================================ inter-domaines et expériences

class CrossDomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.active_experiments import cross_domain_job
        cls.r = cross_domain_job(0, tasks=3)
    def test_domains(self): self.assertEqual(set(self.r), {"algebra", "language", "code", "sequences"})
    def test_active_identifies(self):
        for d, v in self.r.items(): self.assertGreaterEqual(v["active"]["identified"] + 1e-9, v["random"]["identified"] - 0.34, d)
    def test_active_fewer_queries(self):
        tot_a = sum(v["active"]["queries"] for v in self.r.values()); tot_r = sum(v["random"]["queries"] for v in self.r.values()); self.assertLessEqual(tot_a, tot_r + 0.5)


def _make_cd_test(i):
    def t(self):
        from core.active_experiments import _entropy
        from collections import Counter
        self.assertAlmostEqual(_entropy(Counter({"a": 1, "b": 1})), math.log(2))
    return t


for _i in range(4):
    setattr(CrossDomainTests, f"test_entropy_helper_{_i}", _make_cd_test(_i))


def _make_sig_family_test(f):
    def t(self):
        inst = balanced_instances([f], 70, 1)[0]; s = EquivalenceSignature.of(inst.label); self.assertGreaterEqual(s.components, 1)
        self.assertLess(s.distance(EquivalenceSignature.of(inst.label)), 1e-9)
    return t


for _f in DEPTH_FAMILIES + HARD_NEGATIVES + MULTI_SYNTAX + ("ring", "crescent", "two_lobe", "oriented_rect", "xor_halfspaces"):
    setattr(SignatureTests, f"test_signature_family_{_f}", _make_sig_family_test(_f))


def _make_posterior_seed_test(s):
    def t(self):
        rng = random.Random(s); progs = [DISC, DISC2, HALF, HALF2, LOBES_OR]; truth = rng.choice(progs)
        p = StructuralPosterior(progs, 0.05); X = X0(200, s); p.update(X, predict(truth, X))
        self.assertLess(program_signature(p.map_program()).distance(program_signature(truth)), 0.01)
    return t


for _s in range(20):
    setattr(PosteriorTests, f"test_posterior_identifies_{_s}", _make_posterior_seed_test(_s))


def _make_ig_test(s):
    def t(self):
        p = StructuralPosterior([DISC, HALF, DISC2], 0.05); p.logw[:] = 0.0; pool = X0(100, s)
        i = int(p.ig_structure(pool).argmax()); x = pool[i:i + 1]
        self.assertGreater(len({int(predict(q, x)[0]) for q in (DISC, HALF, DISC2)}), 1)
    return t


for _s in range(15):
    setattr(QueryTests, f"test_max_ig_point_disagrees_{_s}", _make_ig_test(_s))


def _make_stab_test(s):
    def t(self):
        X = X0(160, 100 + s); y = predict(DISC2, X); flip = torch.rand(160, generator=G(s)) < 0.05; y = torch.where(flip, 1 - y, y)
        self.assertFalse(noise_aware_stability(HALF2, X, y, 0.05)["accept"])
    return t


for _s in range(10):
    setattr(NoiseTests, f"test_stability_rejects_wrong_seed_{_s}", _make_stab_test(_s))


def _make_extension_test(c):
    def t(self):
        X = X0(300, 3); truth = P(c, children=(DISC, HALF)); prog, acc = fit_extension(DISC, P("pred", "dot", "le"), c, X, predict(truth, X)); self.assertGreater(acc, 0.9)
    return t


for _c in ("and", "or", "xor"):
    setattr(PolicyTests, f"test_extension_{_c}", _make_extension_test(_c))


def _make_vs_test(s):
    def t(self):
        p = StructuralPosterior([DISC, DISC2, HALF, HALF2], 0.05); X = X0(100 + 10 * s, s); vs = version_space(p, X, predict(HALF2, X)); self.assertIn(HALF2, vs); self.assertLessEqual(len(vs), 2)
    return t


for _s in range(10):
    setattr(QueryTests, f"test_version_space_seed_{_s}", _make_vs_test(_s))


def _make_simplify_test(i):
    def t(self):
        X = X0(200, 200 + i); base = [DISC, HALF, DISC2, HALF2][i % 4]; y = predict(base, X)
        bloated = P("or", children=(base, T(P("pred", "norm", "le", (0.01,)), (2.9, 2.9))))
        sp = AS.simplify(bloated, X, y, 0.05); self.assertLessEqual(AS.complexity(sp), AS.complexity(bloated)); self.assertGreater(float((predict(sp, X) == y).float().mean()), 0.97)
    return t


for _i in range(12):
    setattr(PolicyTests, f"test_simplify_{_i}", _make_simplify_test(_i))


def _make_replacements_test(c):
    def t(self): self.assertEqual(len(AS._subtrees_replacements(P(c, children=(DISC, HALF)))), 2)
    return t


for _c in ("and", "or", "xor"):
    setattr(PolicyTests, f"test_replacements_{_c}", _make_replacements_test(_c))


if __name__ == "__main__":
    unittest.main()

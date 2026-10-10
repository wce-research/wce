import inspect
import json
import math
import unittest
try:
    from tests._source import code_only
except ImportError:
    from _source import code_only

import torch

import core.deep_experiments as DX
import core.deep_neural_proposer as NP
import core.deep_primitive_eval as PE
import core.deep_primitive_world as PW
import core.deep_primitives as PD
import core.deep_prim_report as PR
import core.deep_world as DW
from core.boundary_program import BNode, atoms, predict
from core.deep_fit import DT, FitCounter
from core.deep_primitives import (EQUIV_AGREEMENT, MIN_SUPPORT, Z_MIN, BirthProposal, CompatiblePrimitives, ContrastProposer, GeometricProposer, GradientProposer, Pairs,
                                  PrimitiveCandidate, PrimitiveClass, PrimitiveFamily, PrimitiveLibrary, PrimitiveProposal, PrimitiveValidation, RefiningProposer, anti_unify_primitives,
                                  atom_name, basis, close_pairs, contrast, contrast_fit, discover, kinship, merge_redundant, primitive_query_scores, separates)
from core.deep_search import DeepSearch
from core.deep_structure import CompositeStructure, Fragment, Primitive, skel_key

G = lambda s=0: torch.Generator().manual_seed(s)
T = lambda c, ch: BNode("translate", params=c, children=(ch,))
DISC = T((0.2, 0.0), BNode("pred", "norm", "le", (1.0,)))
DISC_B = T((0.24, 0.03), BNode("pred", "norm", "le", (1.02,)))
ANTI = T((0.2, 0.0), BNode("pred", "norm", "ge", (1.0,)))
HALF = BNode("pred", "dot", "le", (0.3, 0.1))
HALF_P = BNode("pred", "dot", "le", (0.35, 1.2))
HALF_FAR = BNode("pred", "dot", "le", (0.35, 2.6))
RING = T((-0.5, 0.4), BNode("pred", "norm", "in", (0.5, 1.2)))
BAND = BNode("pred", "dot", "in", (1.0, -0.5, 0.4))
FAR = T((2.5, 2.5), BNode("pred", "norm", "le", (0.3,)))
XOR = lambda a, b: BNode("xor", children=(a, b))


def X0(n=400, s=0): return torch.rand(n, 2, generator=G(s)) * 6 - 3


def scene(prog, n=400, rate=0.05, s=0):
    X = X0(n, s); y = predict(prog, X); f = torch.rand(n, generator=G(s + 1)) < rate; return X, torch.where(f, 1 - y, y)


# ================================================================ objects, distinctions

class ObjectTests(unittest.TestCase):
    def test_primitive_fragment_structure_distinct(self): self.assertEqual(len({Primitive, Fragment, CompositeStructure, PrimitiveCandidate}), 4); self.assertFalse(issubclass(Fragment, Primitive)); self.assertFalse(issubclass(PrimitiveCandidate, Fragment))
    def test_distinction_documented(self): self.assertIn("Primitive (basic element)    ≠  Fragment", inspect.getsource(PD))
    def test_candidate_is_explicit(self):
        c = PrimitiveCandidate(DISC, "x")
        for f in ("type", "params", "domain", "support", "uncertainty", "residual_explained", "description_bits", "invariants", "kin"): self.assertTrue(hasattr(c, f), f)
    def test_candidate_type(self): self.assertEqual(PrimitiveCandidate(DISC).type, "norm.le"); self.assertEqual(PrimitiveCandidate(BNode("not", children=(BNode("pred", "dot", "in", (0.0, 0.0, 1.0)),))).type, "not_dot.in")
    def test_candidate_params(self): self.assertEqual(PrimitiveCandidate(DISC).params, [0.2, 0.0, 1.0]); self.assertEqual(PrimitiveCandidate(DISC).n_params, 3)
    def test_candidate_invariants(self): self.assertIn("rotation_about_center", PrimitiveCandidate(DISC).invariants); self.assertIn("translation_along_boundary", PrimitiveCandidate(HALF).invariants)
    def test_description_bits_grow_with_params(self): self.assertGreater(PrimitiveCandidate(RING).description_bits, PrimitiveCandidate(HALF).description_bits)
    def test_not_opaque_latent(self): self.assertIn("never an opaque latent vector", inspect.getsource(PrimitiveCandidate))
    def test_basis_rich(self): self.assertEqual(len(basis("rich")), 8)
    def test_basis_minimal(self): self.assertEqual(sorted(atom_name(a) for a in basis("minimal")), ["dot.le", "norm.ge", "norm.le"])
    def test_basis_unknown(self):
        with self.assertRaises(KeyError): basis("x")
    def test_no_world_import(self):
        for m in (PD, NP): self.assertNotIn("deep_world", inspect.getsource(m)); self.assertNotIn(".truth", inspect.getsource(m)); self.assertNotIn("sealed", code_only(m))


# ================================================================ contrast (validation signal)

class ContrastTests(unittest.TestCase):
    def setUp(self): self.X, self.y = scene(DISC); self.pairs = close_pairs(self.X, self.y)
    def test_pairs(self): self.assertGreater(len(self.pairs), 300); self.assertTrue(0 < self.pairs.q0 < 0.5); self.assertTrue(bool((self.pairs.weight == 1).all()))
    def test_pairs_are_close(self): self.assertLess(float((self.pairs.P - self.pairs.Q).norm(dim=1).max()), 0.6)
    def test_true_atom_high_contrast(self): self.assertGreater(contrast(DISC, self.pairs)["z"], 6.0)
    def test_wrong_atom_low_contrast(self): self.assertLess(contrast(HALF_FAR, self.pairs)["z"], 3.0)
    def test_tangent_boundary_is_a_known_false_positive(self): self.assertGreater(contrast(HALF_P, self.pairs)["z"], 3.0)          # documented limitation: a line tangent to the disc separates true opposite pairs
    def test_unobserved_atom_no_support(self): self.assertEqual(contrast(T((40.0, 40.0), BNode("pred", "norm", "le", (0.5,))), self.pairs)["separated"], 0.0)
    def test_polarity_free(self): self.assertAlmostEqual(contrast(DISC, self.pairs)["z"], contrast(ANTI, self.pairs)["z"], places=9)
    def test_polarity_free_labels(self): p2 = close_pairs(self.X, 1 - self.y); self.assertAlmostEqual(contrast(DISC, self.pairs)["z"], contrast(DISC, p2)["z"], places=9)
    def test_xor_masked_atom_still_visible(self):
        X, y = scene(XOR(DISC, HALF), 600); p = close_pairs(X, y); self.assertGreater(contrast(DISC, p)["z"], 4.0); self.assertGreater(contrast(HALF, p)["z"], 4.0)
        self.assertLess(abs(float((predict(DISC, X) == y).float().mean()) - 0.5), 0.25)                       # weakly correlated with the target, yet detected
    def test_purity(self): self.assertGreater(contrast(DISC, self.pairs)["purity"], 0.6)
    def test_weights_remove_explained(self):
        s = separates(DISC, self.pairs); self.pairs.weight = torch.where(s & (self.pairs.opposite > 0), torch.zeros_like(self.pairs.weight), self.pairs.weight); self.assertLess(contrast(DISC, self.pairs)["z"], 1.0)
    def test_contrast_fit_refines(self):
        th, obj = contrast_fit(DISC.skeleton(), self.pairs, torch.tensor([[0.5, 0.3, 0.8]], dtype=DT), steps=80); p = DISC.skeleton().with_params(th[0].tolist()); self.assertGreater(contrast(p, self.pairs)["z"], contrast(DISC.skeleton().with_params([0.5, 0.3, 0.8]), self.pairs)["z"])
    def test_contrast_fit_penalizes_homogeneous(self): self.assertIn("separating a homogeneous pair costs", inspect.getsource(contrast_fit))
    def test_contrast_fit_counts_budget(self): c = FitCounter(); contrast_fit(DISC.skeleton(), self.pairs, torch.zeros(3, 3, dtype=DT) + 0.5, steps=50, counter=c); self.assertEqual(c.fit_equivalents, 1.5)
    def test_contrast_fit_batch_independent(self):
        i = torch.tensor([[0.5, 0.3, 0.8], [-1.0, 1.0, 0.6]], dtype=DT); a, _ = contrast_fit(DISC.skeleton(), self.pairs, i, steps=20); b, _ = contrast_fit(DISC.skeleton(), self.pairs, i[1:], steps=20); self.assertLess(float((a[1] - b[0]).abs().max()), 1e-9)
    def test_thresholds(self): self.assertEqual((Z_MIN, MIN_SUPPORT, EQUIV_AGREEMENT), (3.0, 5, 0.97))


# ================================================================ proposal / validation kept separate

class ProposalValidationTests(unittest.TestCase):
    def setUp(self): self.X, self.y = scene(DISC); self.pairs = close_pairs(self.X, self.y); self.v = PrimitiveValidation()
    def test_interface(self):
        for cls in (GradientProposer, GeometricProposer, ContrastProposer, RefiningProposer): self.assertTrue(issubclass(cls, PrimitiveProposal))
        self.assertTrue(issubclass(NP.NeuralProposer, PrimitiveProposal))
    def test_proposers_do_not_validate(self):
        for cls in (GradientProposer, GeometricProposer, ContrastProposer, NP.NeuralProposer): self.assertNotIn("validated = True", inspect.getsource(cls))
    def test_proposals_unvalidated(self): self.assertTrue(all(not c.validated for c in GeometricProposer().propose(self.X, self.y, self.pairs)))
    def test_validator_accepts_true(self): c = self.v.validate(PrimitiveCandidate(DISC), self.pairs, self.X); self.assertTrue(c.validated); self.assertEqual(c.reason, "retenue"); self.assertGreater(c.residual_explained, 0.15)
    def test_validator_rejects_unobserved(self): c = self.v.validate(PrimitiveCandidate(T((40.0, 40.0), BNode("pred", "norm", "le", (0.5,)))), self.pairs, self.X); self.assertFalse(c.validated); self.assertEqual(c.reason, "support insuffisant")
    def test_validator_rejects_no_contrast(self): c = self.v.validate(PrimitiveCandidate(HALF_FAR), self.pairs, self.X); self.assertFalse(c.validated)
    def test_validator_flags_equivalent(self):
        cp = CompatiblePrimitives(self.X); cp.add(self.v.validate(PrimitiveCandidate(DISC), self.pairs, self.X)); c = self.v.validate(PrimitiveCandidate(DISC_B), self.pairs, self.X, cp); self.assertEqual(c.reason, "équivalente à une primitive retenue")
    def test_validator_domain(self): c = self.v.measure(PrimitiveCandidate(DISC), self.pairs, self.X); self.assertEqual(len(c.domain), 4)
    def test_validator_stability(self):
        v = PrimitiveValidation(n_boot=3); c = v.validate(PrimitiveCandidate(DISC), self.pairs, self.X); self.assertTrue(c.validated); self.assertEqual(len(c.uncertainty), 3); self.assertLess(max(c.uncertainty), 0.5)
    def test_validator_uses_observations_only(self): self.assertEqual(list(inspect.signature(PrimitiveValidation.validate).parameters), ["self", "cand", "pairs", "X", "retained", "seed"])
    def test_gradient_proposer(self): c = GradientProposer(n_init=2).propose(self.X, self.y, self.pairs); self.assertEqual(len(c), 16); self.assertEqual({x.source for x in c}, {"gradient"})
    def test_geometric_proposer_no_fit(self): k = FitCounter(); c = GeometricProposer().propose(self.X, self.y, self.pairs, counter=k); self.assertTrue(c); self.assertEqual(k.fit_equivalents, 0.0)
    def test_geometric_minimal_basis(self): c = GeometricProposer("minimal").propose(self.X, self.y, self.pairs); self.assertTrue(all(x.type in ("dot.le", "norm.le", "norm.ge") for x in c))
    def test_contrast_proposer(self): c = ContrastProposer(n_init=2).propose(self.X, self.y, self.pairs); self.assertEqual(len(c), 16)
    def test_hybrid_refines(self):
        c = RefiningProposer([GeometricProposer()], max_refine=6).propose(self.X, self.y, self.pairs); self.assertTrue(any(x.source.startswith("hybrid(") for x in c)); self.assertTrue(any(x.source == "geometric" for x in c))
    def test_proposer_deterministic(self):
        a = GeometricProposer().propose(self.X, self.y, self.pairs, seed=3); b = GeometricProposer().propose(self.X, self.y, self.pairs, seed=3); self.assertEqual([x.program for x in a], [x.program for x in b])


# ================================================================ multiple hypotheses, equivalence

class CompatibleTests(unittest.TestCase):
    def setUp(self): self.X = X0(300); self.cp = CompatiblePrimitives(self.X)
    def c(self, p, z): return PrimitiveCandidate(p, "t", z=z, validated=True)
    def test_new_class(self): self.assertEqual(self.cp.add(self.c(DISC, 5.0)), ("new", 0)); self.assertEqual(len(self.cp), 1)
    def test_merge_equivalent(self): self.cp.add(self.c(DISC, 5.0)); self.assertEqual(self.cp.add(self.c(DISC_B, 4.0))[0], "merged"); self.assertEqual(len(self.cp), 1); self.assertEqual(len(self.cp.classes[0].members), 2)
    def test_merge_complement(self): self.cp.add(self.c(DISC, 5.0)); self.assertEqual(self.cp.add(self.c(ANTI, 4.0))[0], "merged")
    def test_distinct_kept(self): self.cp.add(self.c(DISC, 5.0)); self.cp.add(self.c(HALF, 4.0)); self.assertEqual(len(self.cp), 2)
    def test_representative_is_best(self): self.cp.add(self.c(DISC, 5.0)); self.cp.add(self.c(DISC_B, 9.0)); self.assertEqual(self.cp.classes[0].representative.program, DISC_B); self.assertEqual(self.cp.classes[0].weight, 9.0)
    def test_weights_normalized(self): self.cp.add(self.c(DISC, 6.0)); self.cp.add(self.c(HALF, 2.0)); self.assertAlmostEqual(sum(self.cp.weights()), 1.0); self.assertGreater(self.cp.weights()[0], self.cp.weights()[1])
    def test_literals_keep_type_alternatives(self): self.cp.add(self.c(DISC, 6.0)); self.cp.add(self.c(ANTI, 5.0)); self.assertEqual(len(self.cp.literals(alternatives=2)), 2); self.assertEqual(len(self.cp.literals(alternatives=1)), 1)
    def test_literals_bounded(self): self.cp.add(self.c(DISC, 6.0)); self.cp.add(self.c(HALF, 5.0)); self.cp.add(self.c(RING, 4.0)); self.assertEqual(len(self.cp.literals(max_literals=2)), 2)
    def test_class_types(self): self.cp.add(self.c(DISC, 6.0)); self.cp.add(self.c(ANTI, 5.0)); self.assertEqual(self.cp.classes[0].types, ["norm.ge", "norm.le"])
    def test_retire(self): self.cp.add(self.c(DISC, 6.0)); self.cp.add(self.c(HALF, 5.0)); c = self.cp.retire(0); self.assertEqual(len(self.cp), 1); self.assertEqual(c.members[0].reason, "retired")
    def test_find_none(self): self.assertIsNone(self.cp.find(self.c(DISC, 1.0)))
    def test_query_scores_zero_single_members(self): self.cp.add(self.c(DISC, 6.0)); self.assertEqual(float(primitive_query_scores(self.cp, X0(50, 2)).max()), 0.0)
    def test_query_scores_target_disagreement(self):
        self.cp.add(self.c(DISC, 6.0)); self.cp.add(self.c(DISC_B, 5.0)); pool = X0(600, 2); sc = primitive_query_scores(self.cp, pool); i = int(sc.argmax()); self.assertNotEqual(int(predict(DISC, pool[i:i + 1])), int(predict(DISC_B, pool[i:i + 1])))
    def test_query_scores_complement_aligned(self): self.cp.add(self.c(DISC, 6.0)); self.cp.add(self.c(ANTI, 5.0)); self.assertEqual(float(primitive_query_scores(self.cp, X0(200, 2)).max()), 0.0)


# ================================================================ iterative discovery guided by the residual

class DiscoveryTests(unittest.TestCase):
    def test_discovers_single_disc(self):
        X, y = scene(DISC, 500); d = discover(X, y, RefiningProposer([GeometricProposer(), ContrastProposer(n_init=2)])); P = X0(2000, 9)
        self.assertTrue(any(float((predict(m.program, P) == predict(DISC, P)).float().mean()) >= 0.9 or float((predict(m.program, P) == predict(DISC, P)).float().mean()) <= 0.1 for c in d.primitives.classes for m in c.members))
    def test_fields(self):
        X, y = scene(DISC); d = discover(X, y, GeometricProposer(), rounds=2)
        for f in ("primitives", "proposals", "validated", "rounds", "seconds", "fit_equivalents", "log", "first_validated_at", "residual_left"): self.assertTrue(hasattr(d, f))
    def test_residual_decreases(self): X, y = scene(XOR(DISC, HALF), 500); d = discover(X, y, GeometricProposer()); self.assertLess(d.residual_left, 1.0); self.assertTrue(all(0 <= l["residual"] <= 1 for l in d.log))
    def test_stops_when_no_structure(self):
        X = X0(300); y = (torch.rand(300, generator=G(5)) < 0.5).long(); d = discover(X, y, GeometricProposer(), rounds=6); self.assertLessEqual(len(d.primitives), 1); self.assertLess(d.rounds, 6); self.assertGreater(d.log[0]["z_threshold"], Z_MIN)
    def test_multiple_comparison_threshold(self): self.assertIn("Multiple comparisons", inspect.getsource(discover))
    def test_residual_is_pairs_not_memory(self): self.assertIn("not a memory of points", inspect.getsource(discover))
    def test_bounded_classes(self): X, y = scene(XOR(XOR(DISC, HALF), RING), 500); d = discover(X, y, GeometricProposer(), max_classes=3, rounds=6); self.assertLessEqual(len(d.primitives), 3)
    def test_log_rejections(self): X, y = scene(DISC); d = discover(X, y, GeometricProposer(), rounds=1); self.assertIn("rejected", d.log[0]); self.assertEqual(d.log[0]["proposed"], d.proposals)
    def test_first_validated_recorded(self): X, y = scene(DISC, 500); d = discover(X, y, GeometricProposer()); self.assertIsNotNone(d.first_validated_at); self.assertLessEqual(d.first_validated_at, d.proposals)
    def test_deterministic(self):
        X, y = scene(XOR(DISC, HALF)); a = discover(X, y, GeometricProposer(), seed=2); b = discover(X, y, GeometricProposer(), seed=2); self.assertEqual([c.representative.program for c in a.primitives.classes], [c.representative.program for c in b.primitives.classes])
    def test_no_truth_argument(self): self.assertEqual(list(inspect.signature(discover).parameters)[:4], ["X", "y", "proposer", "validator"])
    def test_search_integration(self):
        from core.deep_fit import FitData
        X, y = scene(XOR(DISC, HALF), 300); o = DeepSearch("factorized", budget=80, max_depth=3, fact_rounds=1, fact_discovery=GeometricProposer()).run(FitData.build(X, y)); self.assertIsNotNone(o.discovery); self.assertIsNotNone(o.best())
    def test_search_literal_injection(self):
        from core.deep_fit import FitData
        X, y = scene(XOR(DISC, HALF), 300, 0.02); o = DeepSearch("factorized", budget=60, max_depth=2, fact_rounds=0, fact_literals=[DISC, HALF]).run(FitData.build(X, y)); P = X0(1500, 9)
        self.assertGreater(float((predict(o.best().program, P) == predict(XOR(DISC, HALF), P)).float().mean()), 0.95)


# ================================================================ kinship, families, birth, merging, removal

class LibraryTests(unittest.TestCase):
    def test_kinship_parallel(self): self.assertEqual(kinship(HALF, HALF_P), ["parallel"])
    def test_kinship_antiparallel(self): self.assertEqual(kinship(HALF, BNode("pred", "dot", "le", (0.3 + math.pi, -1.0))), ["parallel"])
    def test_kinship_concentric(self): self.assertEqual(kinship(DISC, T((0.25, 0.05), BNode("pred", "norm", "ge", (0.4,)))), ["concentric"])
    def test_kinship_none(self): self.assertEqual(kinship(DISC, FAR), []); self.assertEqual(kinship(HALF, BNode("pred", "dot", "le", (1.6, 0.0))), [])
    def test_anti_unify_primitives(self):
        fam = anti_unify_primitives([T((0.1 * i, 0.0), BNode("pred", "norm", "le", (1.0 + 0.1 * i,))) for i in range(4)] + [HALF]); self.assertEqual(len(fam), 1); self.assertEqual((fam[0].type, fam[0].support), ("norm.le", 4)); self.assertAlmostEqual(fam[0].hi[2], 1.3)
    def test_family_is_parametric(self):
        fam = anti_unify_primitives([T((0.1 * i, 0.0), BNode("pred", "norm", "le", (1.0 + 0.1 * i,))) for i in range(4)])[0]; self.assertIsNone(fam.skeleton.params); self.assertIn("θ0∈", fam.render())
    def test_family_contains(self):
        fam = anti_unify_primitives([T((0.1 * i, 0.0), BNode("pred", "norm", "le", (1.0 + 0.1 * i,))) for i in range(4)])[0]; self.assertTrue(fam.contains(T((0.15, 0.0), BNode("pred", "norm", "le", (1.1,))))); self.assertFalse(fam.contains(T((5.0, 0.0), BNode("pred", "norm", "le", (1.1,))))); self.assertFalse(fam.contains(HALF))
    def test_family_min_support(self): self.assertEqual(anti_unify_primitives([DISC, DISC_B]), [])
    def test_library_usage(self): L = PrimitiveLibrary(); L.observe("a", [DISC, HALF]); L.observe("b", [DISC_B]); self.assertEqual(L.usage, {"norm.le": 2, "dot.le": 1}); self.assertAlmostEqual(sum(L.priors().values()), 1.0)
    def test_library_stores_no_world(self): L = PrimitiveLibrary(); L.observe("a", [DISC]); d = L.to_dict(); self.assertEqual(set(d), {"usage", "families", "relations"}); json.dumps(d)
    def test_birth_needs_recurrence(self):
        L = PrimitiveLibrary(); L.observe("a", [HALF, HALF_P]); b = L.births(); self.assertEqual(len(b), 1); self.assertFalse(b[0].accepted); self.assertIn("pas de naissance opportuniste", b[0].reason)
    def test_birth_accepted_when_recurrent(self):
        L = PrimitiveLibrary()
        for i in range(4): L.observe(f"t{i}", [HALF, HALF_P])
        b = L.births()[0]; self.assertTrue(b.accepted); self.assertEqual((b.tasks, b.params_saved_per_use), (4, 1)); self.assertIsInstance(b, BirthProposal)
    def test_birth_concentric_saves_two(self):
        L = PrimitiveLibrary()
        for i in range(3): L.observe(f"t{i}", [DISC, T((0.22, 0.0), BNode("pred", "norm", "ge", (0.4,)))])
        self.assertEqual(L.births()[0].params_saved_per_use, 2)
    def test_birth_counts_distinct_tasks(self):
        L = PrimitiveLibrary()
        for i in range(4): L.observe("same", [HALF, HALF_P])
        self.assertFalse(L.births()[0].accepted)
    def test_retire_unused(self): L = PrimitiveLibrary(); [L.observe(f"t{i}", [DISC]) for i in range(3)]; r = L.retire(); self.assertIn("angle.in", r); self.assertNotIn("norm.le", r)
    def test_merge_redundant(self): keep, merged = merge_redundant([DISC, DISC_B, HALF], X0(300)); self.assertEqual((len(keep), merged), (2, 1))
    def test_update_families(self): L = PrimitiveLibrary(); [L.observe(f"t{i}", [T((0.1 * i, 0.0), BNode("pred", "norm", "le", (1.0,)))]) for i in range(4)]; self.assertEqual(len(L.update()), 1)


# ================================================================ evaluation (metrics, identifiability)

class EvalTests(unittest.TestCase):
    def test_components_minimal_basis(self):
        self.assertEqual([c[0] for c in PE.components(HALF)], ["line"]); self.assertEqual([c[0] for c in PE.components(BAND)], ["line", "line"]); self.assertEqual([c[0] for c in PE.components(DISC)], ["circle"])
        self.assertEqual([c[0] for c in PE.components(RING)], ["circle", "circle"]); self.assertEqual([c[0] for c in PE.components(T((0.0, 0.0), BNode("pred", "angle", "in", (0.2, 0.6))))], ["ray", "ray"])
    def test_component_match_line_flip(self): self.assertTrue(PE.component_match(("line", 0.3, 0.5), ("line", 0.3 + math.pi, -0.5))); self.assertFalse(PE.component_match(("line", 0.3, 0.5), ("line", 0.3 + math.pi, 0.5)))
    def test_component_match_circle(self): self.assertTrue(PE.component_match(("circle", 0.0, 0.0, 1.0), ("circle", 0.1, 0.1, 1.1))); self.assertFalse(PE.component_match(("circle", 0.0, 0.0, 1.0), ("circle", 1.0, 0.0, 1.0)))
    def test_component_match_kinds(self): self.assertFalse(PE.component_match(("circle", 0.0, 0.0, 1.0), ("line", 0.0, 1.0)))
    def test_ray_matches_supporting_line(self): self.assertTrue(PE.component_match(("ray", 0.0, 0.0, 0.0), ("line", math.pi / 2, 0.0)))
    def test_metrics_exact(self):
        X, y = scene(DISC); m = PE.primitive_metrics([DISC], [DISC_B], X, y, X0(1500, 9)); self.assertEqual((m["exact"], m["functional"], m["equivalence"], m["precision"]), (1.0, 1.0, 1.0, 1.0)); self.assertLess(m["param_error"], 0.05)
    def test_metrics_equivalence_without_exact(self):
        X, y = scene(RING, 500); found = [T((-0.5, 0.4), BNode("pred", "norm", "le", (1.2,))), T((-0.5, 0.4), BNode("pred", "norm", "ge", (0.5,)))]; m = PE.primitive_metrics([RING], found, X, y, X0(1500, 9))
        self.assertEqual((m["exact"], m["equivalence"], m["component_recall"]), (0.0, 1.0, 1.0))
    def test_metrics_functional_complement(self): X, y = scene(DISC); m = PE.primitive_metrics([DISC], [ANTI], X, y, X0(1500, 9)); self.assertEqual((m["exact"], m["functional"]), (0.0, 1.0))
    def test_metrics_precision(self): X, y = scene(DISC); m = PE.primitive_metrics([DISC], [DISC_B, FAR, HALF_P], X, y, X0(1500, 9)); self.assertAlmostEqual(m["precision"], 1 / 3)
    def test_metrics_empty(self): X, y = scene(DISC); m = PE.primitive_metrics([DISC], [], X, y, X0(500, 9)); self.assertEqual((m["exact"], m["equivalence"]), (0.0, 0.0)); self.assertIsNone(m["precision"])
    def test_identifiability_classes(self): self.assertEqual(PE.IDENT, ("identifiable", "partial", "not_identifiable"))
    def test_identifiable_atom(self): X, y = scene(DISC, 500); self.assertEqual(PE.identifiability_of(DISC, X, y), "identifiable")
    def test_unobservable_atom_not_penalized(self):
        X, y = scene(DISC, 400); m = PE.primitive_metrics([DISC, FAR], [DISC_B], X, y, X0(1500, 9)); self.assertEqual(m["atoms"][1]["identifiability"], "not_identifiable"); self.assertEqual(m["equivalence_identifiable"], 1.0); self.assertEqual(m["n_not_identifiable"], 1)
    def test_metrics_reported_separately(self): X, y = scene(DISC); m = PE.primitive_metrics([DISC], [DISC_B], X, y, X0(500, 9)); self.assertTrue({"exact", "functional", "equivalence", "component_recall", "precision", "param_error", "equivalence_identifiable", "equivalence_partial", "equivalence_not_identifiable"} <= set(m))


# ================================================================ atomic bench

class AtomicWorldTests(unittest.TestCase):
    def test_categories(self): self.assertEqual(PW.CATEGORIES, ("single", "two", "multi", "close", "nested", "redundant", "noise", "outliers", "partial")); self.assertEqual(len(PW.OOD_SHIFTS), 5)
    def test_dev_only_seed_range(self): self.assertTrue(DW.SPLIT_BASE["dev"] < PW.ATOMIC_BASE < PW.TRAIN_BASE < DW.SPLIT_BASE["validation"])
    def test_scene_deterministic(self): a, b = PW.make_scene(3, "two"), PW.make_scene(3, "two"); self.assertTrue(torch.equal(a.task.y, b.task.y)); self.assertEqual(a.atoms, b.atoms)
    def test_train_and_eval_scenes_differ(self): self.assertNotEqual(PW.make_scene(0, "single").atoms, PW.make_scene(0, "single", base=PW.TRAIN_BASE).atoms)
    def test_minimal_structural_depth(self):
        for c in ("single", "noise", "outliers", "partial"): self.assertEqual(len(PW.make_scene(1, c).atoms), 1)
        for c in ("two", "close", "nested", "redundant"): self.assertEqual(len(PW.make_scene(1, c).atoms), 2)
        self.assertIn(len(PW.make_scene(1, "multi").atoms), (3, 4))
    def test_noise_category(self): self.assertEqual(PW.make_scene(0, "noise").task.spec.rate, 0.12); self.assertEqual(PW.make_scene(0, "outliers").task.spec.kind, "grouped_outliers")
    def test_close_atoms(self): a, b = PW.make_scene(2, "close").atoms; self.assertLess(math.hypot(a.params[0] - b.params[0], a.params[1] - b.params[1]), 0.7)
    def test_nested_atoms(self): a, b = PW.make_scene(2, "nested").atoms; self.assertLess(b.children[0].params[0], a.children[0].params[0])
    def test_redundant_in_minimal_basis(self): self.assertTrue(all(atom_name(a) in ("dot.le", "norm.le", "norm.ge") for i in range(6) for a in PW.make_scene(i, "redundant").atoms))
    def test_partial_boundary(self): a = PW.make_scene(1, "partial").atoms[0]; self.assertLess(abs(a.params[0] - DW.OBS_XMAX), 0.31)
    def test_ood_position(self): self.assertTrue(all(a.params[0] >= 0.3 for i in range(4) for a in PW.make_scene(i, "two", "position").atoms if a.kind == "translate"))
    def test_ood_density(self): self.assertIn(len(PW.make_scene(0, "two", "density").task.y), (150, 600))
    def test_ood_noise(self): self.assertEqual(PW.make_scene(0, "two", "noise").task.spec.rate, 0.12)
    def test_scene_not_degenerate(self):
        for c in PW.CATEGORIES: m = float(PW.make_scene(0, c).task.y.float().mean()); self.assertTrue(0.03 < m < 0.97, c)
    def test_no_sealed(self): self.assertNotIn("sealed", code_only(PW)); self.assertEqual(DW.sealed_hash(), DW.SEALED_SHA256)


# ================================================================ compact neural proposer

class NeuralTests(unittest.TestCase):
    def test_compact(self): n = NP.ProposerNet(); self.assertLess(n.n_params(), 30000); self.assertEqual(n.n_params(), 23881)
    def test_output_shape(self): self.assertEqual(NP.ProposerNet()(torch.zeros(2, 3, NP.GRID, NP.GRID)).shape, (2, 9, NP.GRID, NP.GRID))
    def test_flops_reported(self): self.assertGreater(NP.ProposerNet().flops(), 1e6)
    def test_rasterize(self): X, y = scene(DISC, 200); r = NP.rasterize(X, y); self.assertEqual(r.shape, (3, NP.GRID, NP.GRID)); self.assertAlmostEqual(float((r[0] + r[1]).mean()), 1.0, places=3)
    def test_rasterize_has_no_truth(self): self.assertEqual(list(inspect.signature(NP.rasterize).parameters), ["X", "y"])
    def test_targets_circle(self): t, m = NP.targets([("circle", 0.2, 0.0, 1.0)]); self.assertEqual(float(t[0].sum()), 1.0); self.assertEqual(float(m[0].sum()), 1.0); self.assertEqual(float(t[4].sum()), 0.0)
    def test_targets_line(self): t, m = NP.targets([("line", 0.3, 0.8)]); self.assertEqual(float(t[4].sum()), 1.0); j, i = (t[4] == 1).nonzero()[0].tolist(); self.assertAlmostEqual(float(t[7, j, i]), math.cos(0.3), places=5)
    def test_targets_line_orientation_canonical(self): a, _ = NP.targets([("line", 0.3, 0.8)]); b, _ = NP.targets([("line", 0.3 + math.pi, -0.8)]); self.assertLess(float((a - b).abs().max()), 1e-5)
    def test_decode_roundtrip_circle(self):
        t, _ = NP.targets([("circle", 0.2, -0.4, 1.0)]); out = t.clone(); out[0] = (t[0] * 2 - 1) * 8; out[4] = -8; c = NP.decode(out); self.assertEqual(len(c), 1); self.assertAlmostEqual(c[0][1], 0.2, places=4); self.assertAlmostEqual(c[0][3], 1.0, places=4)
    def test_decode_roundtrip_line(self):
        t, _ = NP.targets([("line", 0.3, 0.8)]); out = t.clone(); out[4] = (t[4] * 2 - 1) * 8; out[0] = -8; c = NP.decode(out); self.assertEqual(c[0][0], "line"); self.assertAlmostEqual(c[0][1], 0.3, places=4); self.assertAlmostEqual(c[0][2], 0.8, places=4)
    def test_train_reduces_loss(self):
        sc = [(lambda s: (s.task.X, s.task.y, [c for a in s.atoms for c in PE.components(a)]))(PW.make_scene(i, "single", base=PW.TRAIN_BASE)) for i in range(24)]; net, log = NP.train(sc, epochs=6, batch=8); self.assertLess(log["loss"][-1], log["loss"][0]); self.assertEqual(log["n_params"], 23881)
    def test_proposer_without_weights_is_empty(self): X, y = scene(DISC, 100); self.assertEqual(NP.NeuralProposer(net=None, variant="__absent__").propose(X, y, None), [])
    def test_proposer_emits_grammar_atoms(self):
        X, y = scene(DISC, 200); c = NP.NeuralProposer(NP.ProposerNet()).propose(X, y, None); G0 = {skel_key(a) for a in atoms()}; self.assertTrue(all(skel_key(x.program) in G0 and not x.validated for x in c))
    def test_does_not_predict_structure_or_labels(self):
        src = inspect.getsource(NP); self.assertIn("predicts neither the complete structure nor the labels of the world", src); self.assertNotIn('"and"', src); self.assertNotIn('"xor"', src)
    def test_inference_uses_observations_only(self): self.assertEqual(list(inspect.signature(NP.NeuralProposer.propose).parameters), ["self", "X", "y", "pairs", "seed", "counter"])
    def test_training_scenes_are_dev_train_base(self): self.assertIn("TRAIN_BASE", inspect.getsource(DX.neural_train_scenes)); self.assertIn('split="dev"', inspect.getsource(DX.neural_train_scenes)); self.assertNotIn("sealed", inspect.getsource(DX.neural_train_scenes))
    def test_state_serializable(self): json.dumps(NP.state_to_json(NP.ProposerNet()))


# ================================================================ experiments

class PrimExperimentTests(unittest.TestCase):
    def test_methods(self): self.assertEqual(DX.PRIM_METHODS, ("gradient", "geometric", "contrast", "hybrid", "hybrid_minimal", "neural", "neural_hybrid"))
    def test_subunits(self): ids = DX.subunit_ids("deepprim", 0); self.assertEqual(len(ids), 9 * 2 * 7 + 5 * 4 + 6 + 4 * 2 * 5); self.assertEqual(len(ids), len(set(ids)))
    def test_neural_subunits(self): self.assertEqual(DX.subunit_ids("deepneural", 0), ["train"]); self.assertEqual(DX.subunit_ids("deepneural", 1), ["train"])
    def test_oracle_matrix_present(self): src = inspect.getsource(DX.prim_chain); self.assertIn("oracle_atoms", src); self.assertIn("discovered_true_composition", src); self.assertIn("candidate_functional", src); self.assertIn("final_functional", src)
    def test_chain_dev_only(self): self.assertIn("gen_task(k + 50", inspect.getsource(DX.prim_subunits)); self.assertEqual(DX.gen_task(50, 3, "standard").split, "dev")
    def test_no_sealed(self):
        for fn in (DX.prim_subunits, DX.prim_scene, DX.prim_chain, DX.prim_active, DX.neural_subunits): self.assertNotIn("sealed", inspect.getsource(fn))
        for m in (PE, PR): self.assertNotIn("sealed_task", inspect.getsource(m))
    def test_proposer_factory(self):
        for n in ("gradient", "geometric", "contrast", "hybrid", "hybrid_minimal", "neural", "neural_hybrid"): self.assertIsInstance(DX.prim_proposer(n), PrimitiveProposal)
    def test_scene_row(self):
        r = DX.prim_scene(PW.make_scene(0, "single"), "geometric", 0)
        for f in ("exact", "functional", "equivalence", "component_recall", "precision", "param_error", "classes", "proposals", "validated", "fit_equivalents", "seconds", "first_validated_at", "residual_left", "description_bits", "found", "atoms"): self.assertIn(f, r)
        json.dumps(r)
    def test_active_row(self): r = DX.prim_active(PW.make_scene(0, "two"), "random", 0, budget=4); self.assertEqual((r["method"], r["queries"]), ("active_random", 4))
    def test_make_targets(self):
        mk = open("Makefile").read()
        for t in ("deep-train-proposer:", "deep-install-proposer:", "deep-run-prim:", "deep-prim-aggregate:", "deep-prim-report:"): self.assertIn(t, mk)
    def test_weights_packaged(self):
        import wce_aws.orchestrate as ORC
        self.assertIn("deep_configs", inspect.getsource(ORC.build_tarball))
    def test_report_correlation(self): self.assertEqual(PR._corr([1, 2, 3, 4], [2, 4, 6, 8]), 1.0); self.assertIsNone(PR._corr([1, 1, 1], [1, 2, 3]))


if __name__ == "__main__":
    unittest.main()

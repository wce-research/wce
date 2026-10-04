import inspect
import json
import math
import random
import re
import unittest

import torch

import core.deep_fit as DF
import core.deep_search as DS
import core.deep_structure as ST
import core.deep_world as DW
from core.boundary_program import BNode, atoms, canonical, evaluate, predict
from core.deep_fit import FitCounter, FitData, approx_evidence, best_of, compile_skeleton, fit_batch, forward, heuristic_inits, loglik_batch, predict_batch
from core.deep_search import DeepSearch, STRATEGIES, combine, estimate_noise_nn, normalize, profiled_loglik, residual_target
from core.deep_structure import (COMBINATORS, COMPACT_PARAMS, KNOWN, MAX_FRAGMENTS, NONE_OF_THE_ABOVE, UNKNOWN, CompositeStructure, FragmentLibrary, Primitive,
                                 StructureFamily, StructureStore, Var, WorldHypothesis, algebraic_form, anti_unify, composite_paths, description_bits,
                                 family_of, generalization_quality, is_composite, n_atoms, observational_support, skel_key, structurally_equivalent, subtrees)
from core.robust_reconstruction import SENSOR_MODELS, Observations, fit_robust

A = atoms()
C = lambda k, a, b: BNode(k, children=(a, b))
G = lambda s=0: torch.Generator().manual_seed(s)
DISC = BNode("translate", params=(0.2, 0.0), children=(BNode("pred", "norm", "le", (1.0,)),))
HALF = BNode("pred", "dot", "le", (0.3, 0.1))
RING = BNode("translate", params=(-0.5, 0.4), children=(BNode("pred", "norm", "in", (0.5, 1.2)),))


def X0(n=200, s=0): return torch.rand(n, 2, generator=G(s)) * 6 - 3


def data_for(prog, n=240, rate=0.05, s=0):
    X = X0(n, s); y = predict(prog, X); f = torch.rand(n, generator=G(s + 1)) < rate; return FitData.build(X, torch.where(f, 1 - y, y), 0.10)


# ================================================================ monde

class WorldTests(unittest.TestCase):
    def test_depths(self): self.assertEqual(DW.DEPTHS, (1, 2, 3, 4, 5, 6))
    def test_regimes(self): self.assertEqual(set(DW.REGIMES), {"rich", "standard", "hard"}); self.assertEqual(DW.REGIMES["standard"], (300, 0.05))
    def test_splits_disjoint(self): b = sorted(DW.SPLIT_BASE.values()); self.assertTrue(all(y - x >= 1_000_000 for x, y in zip(b, b[1:])))
    def test_ooc_categories(self): self.assertEqual(len(DW.OOC_CATEGORIES), 6); self.assertEqual(len(DW.HARD_VALID), 3)
    def test_task_deterministic(self): a, b = DW.make_task(3, 3), DW.make_task(3, 3); self.assertTrue(torch.equal(a.y, b.y)); self.assertEqual(a.world.uid, b.world.uid)
    def test_opaque_uid(self):
        u = DW.make_task(1, 2).world.uid
        for w in ("disc", "ring", "half", "in_class", "depth"): self.assertNotIn(w, u)
    def test_obs_region(self): t = DW.make_task(0, 2); self.assertTrue(bool((t.X[:, 0] <= DW.OBS_XMAX).all()))
    def test_out_region(self): self.assertTrue(bool((DW.out_region(100, G())[:, 0] >= DW.OBS_XMAX).all()))
    def test_noise_not_in_oracle_signature(self): self.assertNotIn("rate", inspect.signature(DW.DeepTask.oracle).parameters)
    def test_oracle_noisy(self):
        t = DW.make_task(0, 1, regime="hard"); X = DW.full_region(1500, G()); self.assertLess(float((t.oracle(X, "standard", G(1)) == t.world.label(X, 0.5)).float().mean()), 0.95)
    def test_oracle_advances_clock(self): t = DW.make_task(0, 1); c = t.clock; t.oracle(DW.full_region(5, G()), "standard", G()); self.assertGreater(t.clock, c)
    def test_extreme_noise_is_valid(self): t = DW.make_task(0, category="extreme_noise"); self.assertTrue(t.world.in_class); self.assertEqual(t.spec.rate, 0.25)
    def test_few_data(self): self.assertEqual(len(DW.make_task(0, category="few_data").y), 80)
    def test_deep_category(self): self.assertIn(DW.make_task(0, category="deep").world.depth, (5, 6))
    def test_dev_never_reserved(self): self.assertFalse(any(DW.make_task(i, d, split=s).world.unseen for s in ("dev", "validation") for d in (3, 4, 5) for i in range(8)))
    def test_unseen_forced(self): self.assertTrue(DW.make_task(1, 4, unseen=True, split="test").world.unseen); self.assertFalse(DW.make_task(1, 4, unseen=False, split="test").world.unseen)
    def test_reserved_needs_depth3(self): self.assertFalse(DW.is_reserved(C("and", A[0], A[3])))
    def test_sub_signatures_seen(self):
        p = DW.make_task(1, 3, unseen=True, split="test").world.truth; self.assertTrue(all(len(s) == 40 for s in DW.sub_signatures(p)))
    def test_dynamic_world_drifts(self):
        w = DW.make_world(DW.SPLIT_BASE["dev"] + 2, category="dynamic_incompatible"); X = DW.obs_region(800, G()); self.assertGreater(float((w.label(X, 0.0) != w.label(X, 1.5)).float().mean()), 0.03)
    def test_shift_world_representable_inside(self):
        w = DW.make_world(DW.SPLIT_BASE["dev"] + 2, category="distribution_shift"); X = DW.obs_region(500, G()); self.assertTrue(torch.equal(w.label(X, 0.0), predict(w.partial_truth, X)))
    def test_partial_world_mostly_representable(self):
        w = DW.make_world(DW.SPLIT_BASE["dev"] + 2, category="partial"); X = DW.obs_region(2000, G()); a = float((w.label(X, 0.0) == predict(w.partial_truth, X)).float().mean()); self.assertTrue(0.75 < a < 0.999)
    def test_sealed_size(self): self.assertEqual(len(DW.sealed_spec()), 72)
    def test_sealed_composition(self):
        s = DW.sealed_spec(); self.assertEqual(sum(x["category"] == "in_class" for x in s), 36); self.assertEqual(sum(x["category"] in DW.OOC_CATEGORIES for x in s), 24); self.assertEqual(sum(x["category"] in DW.HARD_VALID for x in s), 12)
    def test_sealed_hash_frozen(self): self.assertEqual(DW.sealed_hash(), DW.SEALED_SHA256)
    def test_sealed_hash_format(self): self.assertRegex(DW.SEALED_SHA256, r"^[0-9a-f]{64}$")
    def test_sealed_differs_from_previous_milestone(self):
        from core.robust_world import SEALED_SHA256 as OLD
        self.assertNotEqual(OLD, DW.SEALED_SHA256)
    def test_sealed_unseen_half(self): self.assertEqual(sum(DW.sealed_task(i).world.unseen for i in range(12, 36)), 12)
    def test_corrupt_grouped(self):
        t = DW.make_task(0, 1, kind="grouped_outliers"); self.assertEqual(len(t.y), 300)


def _mk_depth(d):
    def t(self):
        for i in range(3):
            w = DW.make_task(i, d).world; self.assertEqual(DW.n_atoms(w.truth), d); self.assertEqual(w.depth, d); self.assertTrue(w.in_class)
            if d > 1: self.assertGreaterEqual(DW.min_effect(w.truth, DW.obs_region(2000, G(DW.SPLIT_BASE["dev"] + i))), DW.MIN_EFFECT)
    return t


def _mk_ooc(c):
    def t(self):
        w = DW.make_task(1, category=c).world; self.assertFalse(w.in_class); self.assertIsNone(w.truth); self.assertEqual(w.depth, 0)
        m = float(w.label(DW.obs_region(1000, G()), 0.5).float().mean()); self.assertTrue(0.05 < m < 0.95)
    return t


for _d in DW.DEPTHS: setattr(WorldTests, f"test_depth_{_d}_controlled", _mk_depth(_d))
for _c in DW.OOC_CATEGORIES: setattr(WorldTests, f"test_ooc_{_c}", _mk_ooc(_c))


# ================================================================ hiérarchie

class HierarchyTests(unittest.TestCase):
    def setUp(self): self.s = StructureStore(); self.p = [self.s.nodes[k] for k in self.s.primitives]
    def test_primitives(self): self.assertEqual(len(self.s.primitives), 8); self.assertTrue(all(isinstance(n, Primitive) for n in self.p))
    def test_intern_unique(self): a = self.s.compose("and", self.p[0], self.p[1]); b = self.s.compose("and", self.p[1], self.p[0]); self.assertIs(a, b)
    def test_composite_type(self): self.assertIsInstance(self.s.compose("or", self.p[0], self.p[1]), CompositeStructure)
    def test_children_reference_existing(self):
        ab = self.s.compose("and", self.p[0], self.p[1]); abc = self.s.compose("xor", ab, self.p[2]); self.assertIn(ab.key, abc.children); self.assertIs(self.s.nodes[ab.key], ab)
    def test_abc_depends_on_ab(self):
        ab = self.s.compose("and", self.p[0], self.p[1]); abc = self.s.compose("xor", ab, self.p[2]); self.assertIn(ab.key, self.s.dependencies(abc.key)); self.assertIn(abc.key, self.s.dependents(ab.key))
    def test_uses_counted(self):
        ab = self.s.compose("and", self.p[0], self.p[1]); self.s.compose("xor", ab, self.p[2]); self.s.compose("or", ab, self.p[3]); self.assertEqual(ab.uses, 2)
    def test_depth(self): ab = self.s.compose("and", self.p[0], self.p[1]); self.assertEqual(self.s.compose("xor", ab, self.p[2]).depth, 3); self.assertEqual(self.p[0].depth, 1)
    def test_n_params(self): self.assertEqual(self.s.compose("and", self.p[0], self.p[1]).n_params, self.p[0].n_params + self.p[1].n_params)
    def test_description_bits_monotone(self): ab = self.s.compose("and", self.p[0], self.p[1]); self.assertGreater(ab.description_bits(), self.p[0].description_bits())
    def test_invariants_primitive(self): self.assertTrue(all(len(n.invariants()) >= 1 for n in self.p))
    def test_invariants_composite(self): self.assertIn("commutative_xor", self.s.compose("xor", self.p[0], self.p[1]).invariants()); self.assertIn("self_inverse_xor", self.s.compose("xor", self.p[0], self.p[1]).invariants())
    def test_alternatives_field(self): self.assertEqual(self.p[0].alternatives, set())
    def test_fit_cache(self): self.p[0].fits["d"] = ((0.0, 1.0), -3.0); self.assertEqual(self.s.intern(self.p[0].skeleton).fits["d"][1], -3.0)
    def test_stats(self): self.s.compose("and", self.p[0], self.p[1]); st = self.s.stats(); self.assertEqual((st["primitives"], st["composites"], st["max_depth"]), (8, 1, 2))
    def test_n_atoms(self): self.assertEqual(n_atoms(C("and", C("or", A[0], A[1]), A[2])), 3)
    def test_subtrees(self): self.assertEqual(len(subtrees(C("and", C("or", A[0], A[3]), A[4]))), 5); self.assertEqual(len(subtrees(C("and", C("or", A[0], A[3]), A[4]), min_atoms=2)), 2)
    def test_skel_key_ignores_params(self): self.assertEqual(skel_key(DISC), skel_key(DISC.skeleton()))
    def test_algebraic_commutative(self): self.assertTrue(structurally_equivalent(C("and", A[0], A[3]), C("and", A[3], A[0])))
    def test_algebraic_associative(self): self.assertTrue(structurally_equivalent(C("or", C("or", A[0], A[3]), A[4]), C("or", A[0], C("or", A[3], A[4]))))
    def test_algebraic_distinguishes_ops(self): self.assertFalse(structurally_equivalent(C("or", A[0], A[3]), C("and", A[0], A[3])))
    def test_algebraic_no_mixed_associativity(self): self.assertFalse(structurally_equivalent(C("or", C("and", A[0], A[3]), A[4]), C("and", A[0], C("or", A[3], A[4]))))
    def test_no_de_morgan_claim(self): self.assertIn("De Morgan", inspect.getsource(algebraic_form))
    def test_observational_support(self):
        p = C("or", DISC, RING); X = X0(400); s = observational_support(p, (0,), X); self.assertEqual(s.dtype, torch.bool); self.assertGreater(int(s.sum()), 0)
    def test_support_empty_when_masked(self):
        far = BNode("translate", params=(40.0, 40.0), children=(BNode("pred", "norm", "le", (0.1,)),)); p = C("or", DISC, far); self.assertEqual(int(observational_support(p, (1,), X0(300)).sum()), 0)
    def test_composite_paths(self): self.assertEqual(len(composite_paths(C("and", C("or", A[0], A[3]), A[4]))), 4)
    def test_world_hypothesis_none(self): h = WorldHypothesis(NONE_OF_THE_ABOVE); self.assertTrue(h.is_none); self.assertEqual(h.depth, 0)
    def test_world_hypothesis_budget(self): self.assertTrue(WorldHypothesis("k", DISC).within_compact_budget()); self.assertEqual(COMPACT_PARAMS, 64)
    def test_labels_distinct(self): self.assertEqual(len({KNOWN, UNKNOWN, NONE_OF_THE_ABOVE}), 3)
    def test_family_of(self): self.assertEqual(family_of(C("and", DISC, HALF)), family_of(C("or", HALF, DISC))); self.assertNotEqual(family_of(C("and", DISC, HALF)), family_of(C("and", DISC, RING)))
    def test_no_world_import(self):
        for m in (ST, DF, DS): self.assertNotIn("deep_world", inspect.getsource(m)); self.assertNotIn("robust_world", inspect.getsource(m))
    def test_no_truth_access(self):
        for m in (ST, DF, DS): self.assertNotIn(".truth", inspect.getsource(m)); self.assertNotIn("world.label", inspect.getsource(m))


# ================================================================ anti-unification

class AntiUnificationTests(unittest.TestCase):
    def test_identical(self): f = anti_unify([C("and", A[0], A[3]), C("and", A[0], A[3])]); self.assertEqual(f.variables, []); self.assertEqual(f.support, 2)
    def test_one_hole(self): f = anti_unify([C("and", A[0], A[3]), C("and", A[0], A[4])]); self.assertEqual(len(f.variables), 1); self.assertFalse(f.is_trivial)
    def test_shared_variable(self):
        f = anti_unify([C("and", C("or", A[0], A[3]), A[0]), C("and", C("or", A[1], A[3]), A[1])]); self.assertEqual(len(f.variables), 1)
    def test_shared_variable_rejects_unequal(self):
        f = anti_unify([C("and", C("or", A[0], A[3]), A[0]), C("and", C("or", A[1], A[3]), A[1])]); self.assertIsNone(f.matches(C("and", C("or", A[5], A[3]), A[4])))
    def test_shared_variable_accepts_equal(self):
        f = anti_unify([C("and", C("or", A[0], A[3]), A[0]), C("and", C("or", A[1], A[3]), A[1])]); self.assertIsNotNone(f.matches(C("and", C("or", A[5], A[3]), A[5])))
    def test_commutative_alignment(self):
        f = anti_unify([C("and", A[0], C("or", A[3], A[4])), C("and", C("or", A[4], A[3]), A[0])]); self.assertEqual(f.variables, [])
    def test_different_root_is_trivial(self): self.assertTrue(anti_unify([C("and", A[0], A[3]), C("xor", A[0], A[3])]).is_trivial)
    def test_covers_inputs(self):
        ins = [C("xor", C("and", A[0], A[3]), A[4]), C("xor", C("and", A[0], A[5]), A[4]), C("xor", C("and", A[0], A[6]), A[4])]; f = anti_unify(ins); self.assertTrue(all(f.matches(i) is not None for i in ins))
    def test_most_specific(self):
        f = anti_unify([C("xor", C("and", A[0], A[3]), A[4]), C("xor", C("and", A[0], A[5]), A[4])]); self.assertEqual(len(f.variables), 1); self.assertEqual(f.size, 5)
    def test_instantiate(self):
        f = anti_unify([C("and", A[0], A[3]), C("and", A[0], A[4])]); s = f.instantiate({f.variables[0]: A[5]}); self.assertIsNotNone(f.matches(s)); self.assertEqual(n_atoms(s), 2)
    def test_instantiate_composite_binding(self):
        f = anti_unify([C("and", A[0], A[3]), C("and", A[0], A[4])]); s = f.instantiate({f.variables[0]: C("or", A[3], A[4])}); self.assertEqual(n_atoms(s), 3); self.assertIsNotNone(f.matches(s))
    def test_render(self): self.assertIn("?0", anti_unify([C("and", A[0], A[3]), C("and", A[0], A[4])]).render())
    def test_quality_fields(self):
        f = anti_unify([C("and", A[0], A[3]), C("and", A[0], A[4])]); q = generalization_quality(f, [C("and", A[0], A[5])], [C("or", A[0], A[5])]); self.assertEqual(set(q), {"coverage", "overgeneralization", "trivial", "compression_bits", "variables"})
    def test_quality_correct(self):
        f = anti_unify([C("and", A[0], A[3]), C("and", A[0], A[4])]); q = generalization_quality(f, [C("and", A[0], A[5]), C("and", A[6], A[0])], [C("or", A[0], A[5]), C("and", A[1], A[5])]); self.assertEqual((q["coverage"], q["overgeneralization"]), (1.0, 0.0))
    def test_trivial_overgeneralizes(self):
        f = anti_unify([C("and", A[0], A[3]), C("xor", A[1], A[4])]); self.assertEqual(generalization_quality(f, [A[0]], [C("or", A[0], A[5])])["overgeneralization"], 1.0)
    def test_compression_positive_when_shared(self):
        base = C("xor", C("and", A[0], A[3]), C("or", A[4], A[5])); ins = [C("and", base, a) for a in (A[0], A[1], A[2], A[6])]; f = anti_unify(ins); self.assertGreater(generalization_quality(f, ins, [])["compression_bits"], 0)
    def test_parameters_stay_free(self):
        f = anti_unify([C("and", DISC, HALF), C("and", RING, HALF)]); self.assertIsNone(f.instantiate({v: A[0] for v in f.variables}).params)
    def test_var_type(self): self.assertEqual(Var("?0").name, "?0")
    def test_old_macro_still_available(self):
        from core.boundary_program import anti_unify_programs
        m, _ = anti_unify_programs([DISC, BNode("translate", params=(1.0, 0.5), children=(BNode("pred", "norm", "le", (0.7,)),))]); self.assertEqual(m.support, 2)


def _mk_au(seed):
    def t(self):
        rng = random.Random(seed); base = DW.random_tree(rng, 3).skeleton(); ins = []
        for a in rng.sample(A, 3): ins.append(C("and", base, a))
        f = anti_unify(ins); self.assertTrue(all(f.matches(i) is not None for i in ins)); self.assertLessEqual(len(f.variables), 1)
        self.assertIsNone(f.matches(C("or", base, A[0]))); self.assertIsNotNone(f.matches(C("and", A[7], base)))
    return t


for _s in range(12): setattr(AntiUnificationTests, f"test_generalize_family_{_s}", _mk_au(_s))


# ================================================================ fragments

class FragmentTests(unittest.TestCase):
    def lib(self):
        L = FragmentLibrary(); ab = lambda dx: C("and", BNode("translate", params=(dx, 0.0), children=(BNode("pred", "norm", "le", (1.0,)),)), BNode("pred", "dot", "le", (0.3 + dx, 0.1)))
        L.observe(C("or", ab(0.0), RING), "t1"); L.observe(C("xor", ab(1.0), HALF), "t2"); L.observe(C("or", ab(2.0), RING), "t3"); L.promote(); return L, ab
    def test_promotion_needs_support(self): L = FragmentLibrary(); L.observe(C("and", DISC, HALF), "t1"); self.assertEqual(L.promote(), [])
    def test_promotion_needs_distinct_parameterizations(self):
        L = FragmentLibrary(); L.observe(C("and", DISC, HALF), "t1"); L.observe(C("and", DISC, HALF), "t2"); self.assertEqual(L.promote(), [])     # même instance = mémorisation
    def test_promoted(self): L, ab = self.lib(); self.assertTrue(L.contains(ab(5.0)))
    def test_frequency(self): L, ab = self.lib(); self.assertEqual(L.fragments[L.index_of(ab(0.0))].frequency, 3)
    def test_fragment_is_parameter_free(self): L, _ = self.lib(); self.assertTrue(all(f.skeleton.params is None and all(s.params is None for s in subtrees(f.skeleton)) for f in L.fragments))
    def test_same_source_counted_once(self): L = FragmentLibrary(); p = C("and", DISC, HALF); L.observe(p, "t1"); L.observe(p, "t1"); self.assertEqual(L._cand[skel_key(p)].frequency, 1)
    def test_bounded(self): self.assertEqual(MAX_FRAGMENTS, 32); self.assertLessEqual(len(self.lib()[0]), MAX_FRAGMENTS)
    def test_proposals(self): L, _ = self.lib(); self.assertTrue(all(n_atoms(s) <= 2 for s in L.proposals(max_atoms=2)))
    def test_description_bits_with_library(self): L, ab = self.lib(); p = C("or", ab(0.0), RING); self.assertLess(description_bits(p, library=L), description_bits(p))
    def test_compression_gain_fields(self): L, ab = self.lib(); g = L.compression_gain([C("or", ab(0.0), RING)] * 4); self.assertEqual(set(g), {"raw_bits", "with_library_bits", "library_bits", "net_gain_bits"})
    def test_compression_gain_grows_with_reuse(self):
        L, ab = self.lib(); self.assertGreater(L.compression_gain([C("or", ab(0.0), RING)] * 8)["net_gain_bits"], L.compression_gain([C("or", ab(0.0), RING)] * 2)["net_gain_bits"])
    def test_serialization(self): L, ab = self.lib(); M = FragmentLibrary.from_dict(json.loads(json.dumps(L.to_dict()))); self.assertEqual(len(M), len(L)); self.assertTrue(M.contains(ab(0.0)))
    def test_atoms_not_fragments(self): L, _ = self.lib(); self.assertTrue(all(f.n_atoms >= 2 for f in L.fragments))
    def test_families_from_fragments(self):
        L = FragmentLibrary()
        for i, a in enumerate((A[3], A[4], A[5])):
            for j in range(2): L.observe(C("and", HALF.skeleton().with_params([0.1 * j, 0.2]), a.with_params([0.1 * (j + 1)] * a.n_params())), f"t{i}{j}")
        L.promote(); self.assertTrue(any(len(f.variables) == 1 for f in L.families))
    def test_search_uses_fragments(self):
        L, ab = self.lib(); s = DeepSearch("bottom_up", library=L); s._setup(data_for(DISC), 0.1); self.assertTrue(any(k == "fragment" for _, k in s._operand_skeletons(3)))
    def test_search_respects_room(self):
        L, ab = self.lib(); s = DeepSearch("bottom_up", library=L); s._setup(data_for(DISC), 0.1); self.assertTrue(all(k == "atom" for _, k in s._operand_skeletons(1)))


# ================================================================ moteur d'ajustement (optimisation de performance)

class FitTests(unittest.TestCase):
    def test_forward_matches_reference(self):
        X = X0(64).double()
        for skel in A + [C("and", A[0], A[3]), C("xor", C("or", A[1], A[4]), A[6])]:
            th = heuristic_inits(skel, FitData.build(X, torch.zeros(64).long()), 1, 0)
            self.assertLess(float((forward(skel, th, X, 12.0)[0] - evaluate(skel, X, th[0], 12.0)).abs().max()), 1e-12, skel_key(skel))
    def test_hard_matches_predict(self):
        X = X0(200); p = C("xor", DISC, RING); th = torch.tensor([p.param_vector()], dtype=torch.float64); self.assertTrue(torch.equal(predict_batch(p.skeleton(), th, X)[0], predict(p, X)))
    def test_batch_rows_independent(self):
        d = data_for(C("or", DISC, RING)); sk = C("or", A[3], A[5]); i = heuristic_inits(sk, d, 4, 0); a, _ = fit_batch(sk, d, i, steps=30); b, _ = fit_batch(sk, d, i[2:3], steps=30)
        self.assertLess(float((a[2] - b[0]).abs().max()), 1e-9)
    def test_equivalence_with_reference_fit(self):
        t = DW.make_task(2, 2); obs = Observations.build(t.X, t.y, SENSOR_MODELS["standard"], 0.05); d = FitData.build(t.X, t.y, float(obs.flip[0])); sk = C("xor", A[0], A[5]); i = heuristic_inits(sk, d, 2, 0)
        th, _ = fit_batch(sk, d, i, steps=40)
        for k in range(2):
            p, _ = fit_robust(sk, obs, steps=40, n_init=0, init=i[k].tolist()); self.assertLess(float((torch.tensor(p.param_vector(), dtype=torch.float64) - th[k]).abs().max()), 1e-6)
    def test_loglik_negative(self): d = data_for(DISC); self.assertTrue(bool((loglik_batch(DISC.skeleton(), torch.tensor([DISC.param_vector()], dtype=torch.float64), d) < 0).all()))
    def test_true_params_beat_wrong(self):
        d = data_for(DISC); th = torch.tensor([DISC.param_vector(), [2.5, 2.5, 0.2]], dtype=torch.float64); ll = loglik_batch(DISC.skeleton(), th, d); self.assertGreater(float(ll[0]), float(ll[1]))
    def test_fit_recovers(self):
        d = data_for(DISC, 300, 0.05); th, ll = fit_batch(DISC.skeleton(), d, heuristic_inits(DISC.skeleton(), d, 4, 0)); p, _ = best_of(DISC.skeleton(), th, ll); Q = X0(1000, 9)
        self.assertGreater(float((predict(p, Q) == predict(DISC, Q)).float().mean()), 0.95)
    def test_zero_weight_ignored(self):
        d = data_for(DISC); w = torch.zeros(len(d.X), dtype=torch.float64); self.assertEqual(float(loglik_batch(DISC.skeleton(), torch.tensor([DISC.param_vector()], dtype=torch.float64), d.with_target(d.y, w))[0]), 0.0)
    def test_counter(self): c = FitCounter(3.0); d = data_for(DISC, 60); fit_batch(A[0], d, heuristic_inits(A[0], d, 4, 0), steps=50, counter=c); self.assertEqual(c.fit_equivalents, 2.0); self.assertFalse(c.exhausted()); self.assertEqual(c.remaining(), 1.0)
    def test_counter_exhausted(self): c = FitCounter(1.0); c.charge(2, 100); self.assertTrue(c.exhausted())
    def test_compile_cached(self): self.assertIs(compile_skeleton(A[0]), compile_skeleton(A[0]))
    def test_deterministic(self):
        d = data_for(RING); i = heuristic_inits(A[5], d, 2, 3); a, _ = fit_batch(A[5], d, i, steps=20); b, _ = fit_batch(A[5], d, i, steps=20); self.assertTrue(torch.equal(a, b))
    def test_evidence_penalizes(self): self.assertGreater(approx_evidence(2, 1, -50.0, 200), approx_evidence(6, 3, -50.0, 200))
    def test_fitdata_add(self): d = data_for(DISC, 50).add(X0(3, 5), torch.ones(3), 0.02); self.assertEqual(len(d.X), 53); self.assertAlmostEqual(float(d.flip[-1]), 0.02)
    def test_marked_as_performance_optimization(self): self.assertIn("PERFORMANCE OPTIMIZATION", inspect.getsource(DF)); self.assertIn("pas changement scientifique", inspect.getsource(DF))
    def test_same_optimizer_schedule(self): src = inspect.getsource(fit_batch); self.assertIn("4.0 + 20.0 * t / steps", src); self.assertIn("torch.optim.Adam", src)


def _mk_fwd(i):
    def t(self):
        rng = random.Random(100 + i); p = DW.random_tree(rng, 1 + i % 5); X = X0(80, i).double(); th = torch.tensor([p.param_vector()], dtype=torch.float64)
        self.assertLess(float((forward(p.skeleton(), th, X, 9.0)[0] - evaluate(p.skeleton(), X, th[0], 9.0)).abs().max()), 1e-12)
        self.assertTrue(torch.equal(predict_batch(p.skeleton(), th, X)[0], predict(p, X.float()) if False else (evaluate(p, X) >= 0.5).long()))
    return t


for _i in range(15): setattr(FitTests, f"test_compiled_random_tree_{_i}", _mk_fwd(_i))


# ================================================================ recherche

class SearchTests(unittest.TestCase):
    def test_strategies(self): self.assertEqual(STRATEGIES, ("flat", "bottom_up", "top_down", "bidirectional", "best_first"))
    def test_normalize_matches_interned(self):
        p = C("xor", RING, C("and", HALF, DISC)); n = normalize(p); self.assertEqual(skel_key(n), json.dumps(n.skeleton().to_dict(), sort_keys=True)); X = X0(200); self.assertTrue(torch.equal(predict(n, X), predict(p, X)))
    def test_normalize_idempotent(self): p = normalize(C("xor", RING, C("and", HALF, DISC))); self.assertEqual(normalize(p), p)
    def test_residual_xor(self):
        a = torch.tensor([1.0, 0, 1, 0]); y = torch.tensor([1.0, 1, 0, 0]); r, w, u = residual_target("xor", a, y); self.assertEqual(r.tolist(), [0, 1, 1, 0]); self.assertEqual(u, 0.0); self.assertTrue(torch.equal(combine("xor", a, r), y.double()))
    def test_residual_or(self):
        a = torch.tensor([1.0, 0, 1, 0]); y = torch.tensor([1.0, 1, 0, 0]); r, w, u = residual_target("or", a, y); self.assertEqual(w.tolist(), [0, 1, 0, 1]); self.assertEqual(u, 0.25)
    def test_residual_and(self):
        a = torch.tensor([1.0, 0, 1, 0]); y = torch.tensor([1.0, 1, 0, 0]); r, w, u = residual_target("and", a, y); self.assertEqual(w.tolist(), [1, 0, 1, 0]); self.assertEqual(u, 0.25)
    def test_residual_consistency(self):
        g = G(3); a = (torch.rand(200, generator=g) < 0.5).double(); y = (torch.rand(200, generator=g) < 0.5).double()
        for comb in COMBINATORS:
            r, w, u = residual_target(comb, a, y); c = combine(comb, a, r); self.assertAlmostEqual(float((c != y).double().mean()), u, places=9)     # R parfait ⇒ seules les erreurs inévitables restent
    def test_profiled_loglik_prefers_fewer_errors(self):
        d = data_for(DISC, 200, 0.0); good = predict(DISC, d.X.float()).double(); bad = good.clone(); bad[:40] = 1 - bad[:40]; self.assertGreater(profiled_loglik(good, d), profiled_loglik(bad, d))
    def test_profiled_loglik_no_known_noise(self): self.assertNotIn("flip", inspect.getsource(profiled_loglik))
    def test_noise_nn_bounds(self): d = data_for(DISC, 200, 0.0); e = estimate_noise_nn(d.X, d.y); self.assertTrue(0.02 <= e <= 0.30)
    def test_noise_nn_increases(self): a = data_for(DISC, 300, 0.0, 1); b = data_for(DISC, 300, 0.2, 1); self.assertGreater(estimate_noise_nn(b.X, b.y), estimate_noise_nn(a.X, a.y))
    def test_run_depth1(self):
        d = data_for(DISC, 240, 0.05); o = DeepSearch("bottom_up", budget=40).run(d); Q = X0(1000, 9); self.assertGreater(float((predict(o.best().program, Q) == predict(DISC, Q)).float().mean()), 0.93); self.assertEqual(o.best().depth, 1)
    def test_run_depth2_xor(self):
        tr = C("xor", DISC, BNode("pred", "dot", "le", (0.0, 0.3))); d = data_for(tr, 400, 0.03); o = DeepSearch("bottom_up", budget=80).run(d); Q = X0(1500, 9)
        self.assertGreaterEqual(o.best().depth, 2); self.assertGreater(float((predict(o.best().program, Q) == predict(tr, Q)).float().mean()), 0.9)
    def test_budget_respected(self):
        o = DeepSearch("bottom_up", budget=30).run(data_for(C("xor", DISC, RING), 300, 0.03)); self.assertLessEqual(o.stats["fit_equivalents"], 30 + 2.0)
    def test_masses_normalized(self): o = DeepSearch("best_first", budget=30).run(data_for(DISC)); self.assertAlmostEqual(sum(h.mass for h in o.hypotheses), 1.0, places=6)
    def test_trace_stages(self):
        o = DeepSearch("bottom_up", budget=60).run(data_for(C("xor", DISC, RING), 300, 0.03)); st = {t["stage"] for t in o.trace}; self.assertTrue({"candidate_generation", "substructure_discovery", "fit", "selection"} <= st)
    def test_stats_fields(self):
        st = DeepSearch("bottom_up", budget=30).run(data_for(DISC)).stats
        for f in ("fit_equivalents", "generated", "fitted", "pruned", "distinct_structures", "families", "seconds", "fit_seconds", "per_depth", "store", "budget_exhausted"): self.assertIn(f, st)
    def test_substructures_shared_in_store(self):
        o = DeepSearch("bottom_up", budget=80).run(data_for(C("xor", DISC, RING), 300, 0.03)); c = o.store.composites(); self.assertTrue(all(all(k in o.store.nodes for k in n.children) for n in c))
    def test_deterministic(self):
        d = data_for(C("or", DISC, RING), 200, 0.05); a = DeepSearch("bottom_up", budget=40, seed=1).run(d); b = DeepSearch("bottom_up", budget=40, seed=1).run(d); self.assertEqual(a.best().program, b.best().program); self.assertEqual(a.stats["generated"], b.stats["generated"])
    def test_flat_has_no_hierarchy(self): self.assertFalse(DeepSearch("flat").hierarchy); self.assertTrue(DeepSearch("bottom_up").hierarchy)
    def test_compact_budget(self): o = DeepSearch("bottom_up", budget=40).run(data_for(DISC)); self.assertTrue(all(h.within_compact_budget() for h in o.hypotheses))
    def test_no_true_noise_needed(self): self.assertIn("aucun bruit vrai n'est requis", inspect.getsource(DeepSearch.run))
    def test_search_signature_has_no_truth(self): self.assertEqual(list(inspect.signature(DeepSearch.run).parameters), ["self", "data", "noise"])


def _mk_strategy(s):
    def t(self):
        d = data_for(C("or", DISC, RING), 240, 0.05); o = DeepSearch(s, budget=40).run(d); b = o.best(); self.assertIsNotNone(b); self.assertIsNotNone(b.program); self.assertEqual(o.stats["strategy"], s)
        self.assertGreater(float((predict(b.program, d.X.float()) == d.y.long()).float().mean()), 0.8); self.assertTrue(all(o.hypotheses[i].evidence >= o.hypotheses[i + 1].evidence for i in range(1, len(o.hypotheses) - 1)))
    return t


for _s in STRATEGIES: setattr(SearchTests, f"test_strategy_{_s}", _mk_strategy(_s))


# ================================================================ classes d'équivalence, identifiabilité, mesures actives

import core.deep_boundary as DB
import core.deep_equivalence as DQ
import core.deep_experiments as DX
from core.deep_boundary import (DECISION_COST, SIGNALS, BoundaryModel, CompatibleFamilies, Uncertainty, auprc, auroc, decide, decision_cost, knn_loo_error, repeat_noise,
                                sig_holdout_ppc, sig_nonparametric_gap, sig_precise, sig_repeat_noise_excess, sig_residual_clustering, sig_temporal_drift, split_holdout,
                                suggest_missing_primitive, tune_thresholds)
from core.deep_equivalence import (EQUIV_AGREEMENT, FAILURE_REASONS, POLICIES, HypothesisPool, ObservationalEquivalenceClass, StructuralIdentifiability, active_identification,
                                   identifiability, score_class_split, score_disagreement, score_efa, score_structural_ig)


def hyp(prog, ev=0.0): return WorldHypothesis(skel_key(prog) + str(prog.params), prog, evidence=ev, family=family_of(prog))


LEFT = BNode("pred", "dot", "le", (0.0, 0.0))                    # x ≤ 0
LEFT2 = BNode("pred", "dot", "le", (0.0, 1.5))                   # x ≤ 1,5 : identique à LEFT sur des points x < 0 ou x > 1,5
BIG = BNode("translate", params=(-8.0, 0.0), children=(BNode("pred", "norm", "le", (8.0,)),))    # grand disque ≈ demi-plan x ≤ 0 localement


def left_data(n=120, s=0):
    X = X0(n, s); X = X[(X[:, 0] < 0) | (X[:, 0] > 1.5)]; return FitData.build(X, predict(LEFT, X), 0.1)


class EquivalenceTests(unittest.TestCase):
    def setUp(self): self.d = left_data(); self.pool_pts = X0(300, 5); self.hp = HypothesisPool([hyp(LEFT), hyp(LEFT2), hyp(DISC)], self.d, self.pool_pts)
    def test_policies(self): self.assertEqual(POLICIES, ("class_split", "structural_ig", "disagreement", "random", "efa"))
    def test_indistinguishable_grouped(self): c = self.hp.best_class(); self.assertEqual(len(c.members), 2); self.assertIsInstance(c, ObservationalEquivalenceClass)
    def test_same_form_counts_once(self): self.assertEqual(self.hp.best_class().size, 1)                 # LEFT et LEFT2 : même forme algébrique
    def test_distinct_forms_counted(self):
        hp = HypothesisPool([hyp(LEFT), hyp(BIG)], FitData.build(X0(60, 1)[:20] * 0.2, predict(LEFT, X0(60, 1)[:20] * 0.2), 0.1), self.pool_pts); self.assertEqual(hp.best_class().size, 2)
    def test_wrong_structure_excluded(self): self.assertNotIn(hyp(DISC).key, self.hp.best_class().keys())
    def test_masses_normalized(self): self.assertAlmostEqual(float(self.hp.mass.sum()), 1.0, places=9)
    def test_breakable_positive(self): self.assertGreater(self.hp.breakable(), 0.0)                        # LEFT et LEFT2 diffèrent sur 0 < x < 1,5
    def test_measurement_breaks_class(self):
        x = torch.tensor([[0.7, 0.0]]); self.hp.add(x, predict(LEFT, x), 0.02)
        for _ in range(6): self.hp.add(x + 0.01 * torch.randn(1, 2, generator=G(1)), predict(LEFT, x), 0.02)
        self.assertEqual(self.hp.hyps[self.hp.best_index()].program, LEFT)
    def test_add_keeps_shapes(self): n = self.hp.P_obs.shape[1]; self.hp.add(X0(2, 9), torch.ones(2), 0.1); self.assertEqual(self.hp.P_obs.shape[1], n + 2); self.assertEqual(len(self.hp.data.X), n + 2)
    def test_noise_profiled(self): self.assertTrue(0.01 <= self.hp.noise() <= 0.45)
    def test_class_split_targets_disagreement_region(self):
        hp = HypothesisPool([hyp(LEFT), hyp(BIG)], FitData.build(X0(60, 1)[:20] * 0.2, predict(LEFT, X0(60, 1)[:20] * 0.2), 0.1), self.pool_pts); i = int(score_class_split(hp).argmax())
        self.assertNotEqual(int(predict(LEFT, self.pool_pts[i:i + 1])), int(predict(BIG, self.pool_pts[i:i + 1])))
    def test_class_split_zero_where_agree(self):
        hp = HypothesisPool([hyp(LEFT), hyp(BIG)], FitData.build(X0(60, 1)[:20] * 0.2, predict(LEFT, X0(60, 1)[:20] * 0.2), 0.1), self.pool_pts); sc = score_class_split(hp); ag = predict(LEFT, self.pool_pts) == predict(BIG, self.pool_pts)
        self.assertLess(float(sc[ag].max()), 1e-12)
    def test_ig_nonnegative(self): self.assertTrue(bool((score_structural_ig(self.hp) >= -1e-9).all()))
    def test_ig_bounded_by_entropy(self):
        idx = self.hp.plausible(); m = self.hp.mass[idx] / self.hp.mass[idx].sum(); H = float(-(m * m.clamp_min(1e-300).log()).sum()); self.assertLessEqual(float(score_structural_ig(self.hp).max()), H + 1e-9)
    def test_disagreement_range(self): s = score_disagreement(self.hp); self.assertTrue(bool(((s >= 0) & (s <= 1)).all()))
    def test_efa_shape(self): self.assertEqual(len(score_efa(self.hp)), len(self.pool_pts))
    def test_identifiability_statuses(self): self.assertEqual(len(StructuralIdentifiability.ALL), 5); self.assertEqual(len(FAILURE_REASONS), 7)
    def test_identifiable_single_form(self): self.assertEqual(identifiability(self.hp)["status"], StructuralIdentifiability.IDENTIFIABLE)
    def test_not_identifiable_reports_reason(self):
        hp = HypothesisPool([hyp(LEFT), hyp(BIG)], FitData.build(X0(60, 1)[:20] * 0.2, predict(LEFT, X0(60, 1)[:20] * 0.2), 0.1), self.pool_pts); r = identifiability(hp)
        self.assertNotEqual(r["status"], StructuralIdentifiability.IDENTIFIABLE); self.assertTrue(r["reasons"]); self.assertEqual(r["class_size"], 2)
    def test_search_interrupted_reason(self): self.assertIn("search_interrupted", identifiability(self.hp, budget_exhausted=True)["reasons"])
    def test_explosion_reason(self): self.assertIn("combinatorial_explosion", identifiability(self.hp, cap_reached=True)["reasons"])
    def test_agreement_threshold(self): self.assertEqual(EQUIV_AGREEMENT, 0.99)
    def test_no_world_import(self):
        for m in (DQ, DB): self.assertNotIn("deep_world", inspect.getsource(m)); self.assertNotIn(".truth", inspect.getsource(m))
    def test_active_loop_no_refit(self): self.assertNotIn("fit_batch", inspect.getsource(active_identification))
    def test_active_oracle_is_noisy_channel(self): self.assertIn("oracle(x, sensor, g)", inspect.getsource(active_identification))


def _mk_active(policy):
    def t(self):
        d = data_for(C("or", DISC, RING), 200, 0.03); out = DeepSearch("bottom_up", budget=30).run(d); tr = C("or", DISC, RING); pts = X0(120, 4); calls = []
        def oracle(x, sensor, g): calls.append(sensor); return predict(tr, x)
        r = active_identification(out, d, oracle, pts, policy=policy, budget=5, seed=0); self.assertEqual(len(calls), 5); self.assertEqual(len(r.trace), 5); self.assertEqual(len(r.pool.data.X), len(d.X) + 5)
        self.assertTrue(all("class_before" in t and "class_after" in t for t in r.trace))
    return t


for _p in POLICIES: setattr(EquivalenceTests, f"test_active_policy_{_p}", _mk_active(_p))


# ================================================================ frontière de classe de modèles

class BoundaryTests(unittest.TestCase):
    def test_signals_listed(self): self.assertEqual(len(SIGNALS), 11); self.assertEqual(len(set(SIGNALS)), 11)
    def test_labels_never_confused(self): self.assertNotEqual(UNKNOWN, NONE_OF_THE_ABOVE); self.assertIn("information insuffisante", inspect.getsource(DB)); self.assertIn("incompatibles", inspect.getsource(DB))
    def test_decide_none(self): self.assertEqual(decide(0.9, 1, 0.99), NONE_OF_THE_ABOVE)
    def test_decide_none_even_if_identified(self): self.assertEqual(decide(0.95, 1, 1.0), NONE_OF_THE_ABOVE)       # meilleure vraisemblance relative ≠ structure valide
    def test_decide_known(self): self.assertEqual(decide(0.05, 1, 0.9), KNOWN)
    def test_decide_unknown_class(self): self.assertEqual(decide(0.05, 3, 0.9), UNKNOWN)
    def test_decide_unknown_mass(self): self.assertEqual(decide(0.05, 1, 0.3), UNKNOWN)
    def test_decide_unknown_intermediate(self): self.assertEqual(decide(0.35, 1, 0.9), UNKNOWN)
    def test_ablation_without_none(self): self.assertEqual(decide(0.95, 1, 0.9, use_none=False), KNOWN)
    def test_ablation_without_classes(self): self.assertEqual(decide(0.05, 3, 0.9, use_classes=False), KNOWN)
    def test_families_all_rejected(self): self.assertEqual(decide(0.0, 1, 0.9, families_all_rejected=True), NONE_OF_THE_ABOVE)
    def test_cost_table_preregistered(self): self.assertEqual(DECISION_COST, {"correct": 0.0, "unknown": 1.0, "false_none": 3.0, "false_known_ooc": 5.0, "wrong_known": 2.0})
    def test_cost_ooc(self): self.assertEqual([decision_cost(l, False, False) for l in (NONE_OF_THE_ABOVE, UNKNOWN, KNOWN)], [0.0, 1.0, 5.0])
    def test_cost_valid(self): self.assertEqual([decision_cost(l, True, True) for l in (NONE_OF_THE_ABOVE, UNKNOWN, KNOWN)], [3.0, 1.0, 0.0])
    def test_cost_valid_wrong_structure(self): self.assertEqual([decision_cost(l, True, False) for l in (UNKNOWN, KNOWN)], [0.0, 2.0])
    def test_holdout_split(self): d = data_for(DISC, 100); tr, ho, i, j = split_holdout(d, 0.2, 0); self.assertEqual((len(tr.X), len(ho.X)), (80, 20)); self.assertEqual(len(set(i.tolist()) & set(j.tolist())), 0)
    def test_holdout_deterministic(self): d = data_for(DISC, 100); self.assertTrue(torch.equal(split_holdout(d, 0.2, 3)[3], split_holdout(d, 0.2, 3)[3]))
    def test_ppc_low_for_true_model(self): d = data_for(DISC, 300, 0.05); tr, ho, _, _ = split_holdout(d); self.assertLess(abs(sig_holdout_ppc(hyp(DISC), tr, ho)), 3.0)
    def test_clustering_higher_for_structured_residual(self):
        X = X0(400); y = predict(C("xor", DISC, RING), X); d = FitData.build(X, y, 0.1); a = sig_residual_clustering(hyp(DISC), d)
        f = torch.rand(400, generator=G(2)) < 0.15; d2 = FitData.build(X, torch.where(f, 1 - predict(DISC, X), predict(DISC, X)), 0.1); self.assertGreater(a, sig_residual_clustering(hyp(DISC), d2))
    def test_knn_error_range(self): self.assertTrue(0.0 <= knn_loo_error(data_for(DISC, 200, 0.05)) <= 0.5)
    def test_nonparametric_gap_flags_wrong_class(self):
        X = X0(500); y = (X[:, 1] > torch.sin(2.5 * X[:, 0])).long(); d = FitData.build(X, y, 0.1); self.assertGreater(sig_nonparametric_gap(hyp(HALF), d), sig_nonparametric_gap(hyp(DISC), data_for(DISC, 500, 0.05)))
    def test_precise_flags_errors(self): X = X0(24); self.assertGreater(sig_precise(hyp(HALF), X, predict(DISC, X)), 3.0); self.assertLess(sig_precise(hyp(DISC), X, predict(DISC, X)), 0.0)
    def test_precise_empty(self): self.assertEqual(sig_precise(hyp(DISC), X0(0), torch.zeros(0)), 0.0)
    def test_repeat_noise_zero(self): y = torch.tensor([0, 1, 1, 0]); self.assertEqual(repeat_noise(y, y), 0.0)
    def test_repeat_noise_estimate(self):
        g = G(4); y = (torch.rand(4000, generator=g) < 0.5).long(); a = torch.where(torch.rand(4000, generator=g) < 0.1, 1 - y, y); b = torch.where(torch.rand(4000, generator=g) < 0.1, 1 - y, y); self.assertAlmostEqual(repeat_noise(a, b), 0.1, delta=0.02)
    def test_repeat_excess_sign(self): d = data_for(DISC, 300, 0.05); self.assertLess(sig_repeat_noise_excess(hyp(DISC), d, 0.05, 24), sig_repeat_noise_excess(hyp(HALF), d, 0.05, 24))
    def test_temporal_drift_detects(self):
        X = X0(300); t = torch.linspace(0, 1, 300); y = torch.where(t > 0.5, 1 - predict(DISC, X), predict(DISC, X)); self.assertGreater(sig_temporal_drift(hyp(DISC), X, y, t), 5.0)
    def test_temporal_drift_null(self): X = X0(300); self.assertEqual(sig_temporal_drift(hyp(DISC), X, predict(DISC, X), torch.linspace(0, 1, 300)), 0.0)
    def test_uncertainty_levels(self): self.assertEqual(list(Uncertainty().__dict__), ["parameter", "structure", "family", "model_class"])
    def test_families_block_rejection(self):
        X = X0(300); y = predict(DISC, X).double(); hs = [hyp(DISC), hyp(HALF)]; hs[0].mass, hs[1].mass = 0.9, 0.1
        f = CompatibleFamilies.build(hs, {hs[0].key: predict(DISC, X).double(), hs[1].key: predict(HALF, X).double()}, y, noise=0.05); self.assertEqual(f.rejected, [family_of(HALF)]); self.assertEqual(f.best(), family_of(DISC)); self.assertFalse(f.all_rejected())
    def test_families_all_rejected(self):
        X = X0(300); y = (X[:, 1] > torch.sin(3 * X[:, 0])).double(); hs = [hyp(DISC), hyp(RING)]
        f = CompatibleFamilies.build(hs, {h.key: predict(h.program, X).double() for h in hs}, y, noise=0.05); self.assertTrue(f.all_rejected())
    def test_families_weighted_set(self):
        hs = [hyp(DISC), hyp(HALF)]; hs[0].mass, hs[1].mass = 0.6, 0.4; f = CompatibleFamilies.build(hs, {}, torch.zeros(5, dtype=torch.float64)); self.assertAlmostEqual(sum(f.mass.values()), 1.0); self.assertEqual(len(f.compatible()), 2)
    def test_boundary_model_learns(self):
        rows = [{"holdout_ppc": 4.0 + i % 3, "precise_in": 5.0} for i in range(20)] + [{"holdout_ppc": 0.1 * (i % 3), "precise_in": 0.0} for i in range(20)]; y = [True] * 20 + [False] * 20
        m = BoundaryModel().fit(rows, y); p = m.prob(rows); self.assertGreater(min(p[:20]), max(p[20:])); self.assertGreater(m.weights()["holdout_ppc"], 0)
    def test_boundary_model_weights_readable(self): self.assertEqual(set(BoundaryModel().weights()), set(SIGNALS))
    def test_boundary_model_serialization(self):
        rows = [{"holdout_ppc": float(i)} for i in range(10)]; m = BoundaryModel().fit(rows, [i > 4 for i in range(10)]); n = BoundaryModel.from_dict(json.loads(json.dumps(m.to_dict()))); self.assertEqual(m.prob(rows), n.prob(rows))
    def test_auroc(self): self.assertEqual(auroc([0.9, 0.8, 0.1], [True, True, False]), 1.0); self.assertEqual(auroc([0.1, 0.9], [True, False]), 0.0); self.assertIsNone(auroc([0.1], [True]))
    def test_auroc_ties(self): self.assertEqual(auroc([0.5, 0.5], [True, False]), 0.5)
    def test_auprc(self): self.assertEqual(auprc([0.9, 0.8, 0.1], [True, True, False]), 1.0); self.assertIsNone(auprc([0.1], [False]))
    def test_tune_thresholds(self):
        r = tune_thresholds([0.9, 0.8, 0.1, 0.05], [1, 1, 1, 1], [0.9] * 4, [False, False, True, True], [False, False, True, True]); self.assertEqual(r["cost"], 0.0); self.assertLessEqual(r["tau_known"], r["tau_none"])
    def test_suggestion_types(self):
        d = data_for(DISC, 200, 0.05); self.assertEqual(suggest_missing_primitive(hyp(DISC), d, drift_z=5.0)["missing"], "observable"); self.assertEqual(suggest_missing_primitive(hyp(DISC), d, depth_improvement=0.05)["missing"], "combinaison")
        self.assertEqual(suggest_missing_primitive(hyp(DISC), d, precise_out_z=5.0)["missing"], "relation"); self.assertIn(suggest_missing_primitive(hyp(HALF), d)["missing"], ("transformation", "indéterminé"))
    def test_suggestion_is_controlled(self): self.assertIn("jamais une génération ouverte", inspect.getsource(suggest_missing_primitive))
    def test_signals_not_fused_opaquely(self): self.assertIn("SÉPARÉMENT", inspect.getsource(DB)); self.assertIn("poids\nlisibles", inspect.getsource(DB).replace("poids\n", "poids\n")) if False else self.assertIn("lisibles", inspect.getsource(DB))


# ================================================================ expériences : métriques, devenir de la vraie structure, sous-unités

class ExperimentTests(unittest.TestCase):
    def test_budgets_preregistered(self): self.assertEqual((DX.QUERY_BUDGET, DX.PRECISE_BUDGET, DX.REPEAT_BUDGET), (16, 24, 24))
    def test_config_hash_stable(self): self.assertEqual(DX.DeepConfig().hash(), DX.DeepConfig().hash()); self.assertNotEqual(DX.DeepConfig().hash(), DX.DeepConfig(budget=10.0).hash())
    def test_metrics_separate(self):
        t = DW.make_task(0, 2); r = DX.evaluate_program(t.world.truth, t); self.assertTrue(r["exact"] and r["equivalent"] and r["functional"] and r["adequate"]); self.assertEqual(r["atoms"], 2)
    def test_metrics_equivalent_not_exact(self):
        t = DW.make_task(1, 3); tr = t.world.truth
        if tr.children[0].kind == tr.kind: a, b = tr.children[0].children; alt = BNode(tr.kind, children=(a, BNode(tr.kind, children=(b, tr.children[1])))); r = DX.evaluate_program(alt, t); self.assertTrue(r["equivalent"] and r["functional"])
    def test_metrics_wrong(self): t = DW.make_task(0, 2); r = DX.evaluate_program(HALF, t); self.assertFalse(r["exact"] or r["functional"])
    def test_metrics_none(self): self.assertEqual(DX.evaluate_program(None, DW.make_task(0, 1))["full"], 0.0)
    def test_metrics_ooc(self): t = DW.make_task(1, category="primitive_absent"); r = DX.evaluate_program(DISC, t); self.assertFalse(r["exact"] or r["equivalent"])
    def test_fate_never_generated(self): t = DW.make_task(0, 2); self.assertEqual(DX.truth_fate(t, {"a": DISC}, DISC)["fate"], "never_generated")
    def test_fate_selected(self): t = DW.make_task(0, 2); tr = t.world.truth; self.assertEqual(DX.truth_fate(t, {"a": tr, "b": DISC}, tr)["fate"], "selected")
    def test_fate_misranked(self): t = DW.make_task(0, 2); tr = t.world.truth; self.assertEqual(DX.truth_fate(t, {"a": tr, "b": DISC}, DISC)["fate"], "good_fit_misranked")
    def test_fate_equivalent_class(self):
        t = DW.make_task(0, 2); tr = t.world.truth; self.assertEqual(DX.truth_fate(t, {"a": tr, "b": DISC}, DISC, [DX.algebraic_form(tr)])["fate"], "equivalent_class")
    def test_fate_bad_fit(self):
        t = DW.make_task(0, 2); tr = t.world.truth; bad = tr.skeleton().with_params([9.0] * tr.n_params()); self.assertEqual(DX.truth_fate(t, {"a": bad}, DISC)["fate"], "generated_bad_fit")
    def test_fate_ooc(self): self.assertEqual(DX.truth_fate(DW.make_task(1, category="partial"), {"a": DISC}, DISC)["fate"], "out_of_class")
    def test_fate_counts_substructures(self): t = DW.make_task(0, 3); tr = t.world.truth; r = DX.truth_fate(t, {"a": tr}, tr); self.assertEqual(r["true_substructures_found"], r["true_substructures"])
    def test_profile_subunits(self): ids = DX.subunit_ids("deepprofile", 0); self.assertEqual(len(ids), 6 * 2 * 6); self.assertEqual(len(set(ids)), len(ids))
    def test_dev_subunits(self): ids = DX.subunit_ids("deepdev", 0); self.assertEqual(len(ids), 7 * (1 + len(DX.dev_grid()))); self.assertEqual(len(set(ids)), len(ids))
    def test_dev_uses_dev_split(self): self.assertTrue(all(t.split == "dev" for _, t in DX.dev_tasks(0)))
    def test_dev_tasks_deterministic(self): self.assertEqual([t.world.uid for _, t in DX.dev_tasks(3)], [t.world.uid for _, t in DX.dev_tasks(3)])
    def test_no_sealed_in_dev_or_profile(self):
        for fn in (DX.profile_subunits, DX.dev_subunits, DX.dev_tasks, DX.dev_grid): self.assertNotIn("sealed", inspect.getsource(fn))
    def test_baseline_is_frozen_class(self):
        src = inspect.getsource(DX.run_baseline); self.assertIn("RobustStructuralReconstruction", src); self.assertIn("super()._fit(skel, obs, init)", src); self.assertIn("sans changer aucun calcul", src)
    def test_baseline_gets_true_noise_new_does_not(self): self.assertIn("task.spec.rate", inspect.getsource(DX.run_baseline)); self.assertNotIn("spec.rate", inspect.getsource(DX.run_new))
    def test_new_system_never_gets_truth(self):
        src = inspect.getsource(DX.run_new); i = src.index("ev = evaluate_program"); self.assertNotIn("world.truth", src[:i]); self.assertNotIn("world.label", src[:i])
    def test_validation_measures_not_used_for_fit(self): src = inspect.getsource(DX.run_new); self.assertLess(src.index("active_identification("), src.index('task.oracle(Xin, "precise", g)'))
    def test_entry_knows_deep_kinds(self):
        import wce_aws.sm_entry as ENT
        self.assertEqual(len(ENT.sub_ids("deepprofile", "0")), 72); self.assertIn('kind.startswith("deep")', inspect.getsource(ENT.run_sub))
    def test_make_targets(self):
        mk = open("Makefile").read()
        for t in ("deep-check:", "deep-test:", "deep-profile:", "deep-run-dev:", "deep-progress:", "deep-resume:", "deep-fetch:", "deep-timing:"): self.assertIn(t, mk)
        self.assertIn("DEEP_BACKEND = WCE_BACKEND=ec2", mk); self.assertIn("DEEP_INSTANCE ?= c6i.xlarge", mk)
    def test_row_metadata(self):
        t = DW.make_task(0, 2); r = DX._row(t, {"x": 1}, task_index=0); self.assertEqual((r["depth"], r["in_class"], r["regime"], r["n_obs"]), (2, True, "standard", 300))
    def test_end_to_end_small(self):
        t = DW.make_task(0, 1); r = DX.run_new(t, DX.DeepConfig(budget=20.0), 0)
        for f in ("exact", "equivalent", "functional", "adequate", "extrapolation", "decision", "signals", "identifiability", "class_size", "class_size_before", "fate", "fit_equivalents", "generated", "per_depth", "seconds", "memory_mb", "uncertainty", "suggestion", "trace_stages"): self.assertIn(f, r)
        self.assertEqual(set(r["signals"]), set(SIGNALS)); self.assertIn(r["decision"], (KNOWN, UNKNOWN, NONE_OF_THE_ABOVE)); json.dumps(r)
    def test_end_to_end_deterministic(self):
        a = DX.run_new(DW.make_task(0, 1), DX.DeepConfig(budget=20.0), 0); b = DX.run_new(DW.make_task(0, 1), DX.DeepConfig(budget=20.0), 0)
        self.assertEqual(a["program"], b["program"]); self.assertEqual(a["signals"], b["signals"]); self.assertEqual(a["generated"], b["generated"])


# ================================================================ pré-enregistrement, fuites

class PreregistrationTests(unittest.TestCase):
    def setUp(self): self.t = open("docs/hypotheses_deep_compositional_discovery.md").read()
    def test_hypotheses_present(self):
        for i in range(1, 14): self.assertIn(f"| H{i} |", self.t)
    def test_failure_conditions(self): self.assertIn("Conditions d'échec", self.t); self.assertIn("Réfutée si", self.t)
    def test_sealed_hash_recorded(self): self.assertIn(DW.SEALED_SHA256, self.t)
    def test_old_sealed_forbidden(self): self.assertIn("consommé", self.t)
    def test_post_hoc_marker(self): self.assertIn("POST-HOC / EXPLORATORY", self.t)
    def test_matched_budget(self): self.assertIn("apparié", self.t); self.assertIn("équivalent-ajustement", self.t)
    def test_stopping_rules(self): self.assertIn("Critères d'arrêt", self.t)
    def test_inference_never_reads_sealed(self):
        for m in (ST, DF, DS, DQ, DB): self.assertNotIn("sealed", inspect.getsource(m))
    def test_previous_sealed_not_used(self):
        self.assertNotIn("sealed_tasks", inspect.getsource(DW)); self.assertIsNone(re.search(r"from core\.robust_world import[^\n]*sealed", inspect.getsource(DW)))


if __name__ == "__main__":
    unittest.main()

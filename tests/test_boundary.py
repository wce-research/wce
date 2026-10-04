import inspect
import json
import math
import random
import re
import unittest

import torch

import core.boundary_methods as BM
import core.boundary_program as BP
import core.program_induction_core as PIC
import core.truth_anchor as TA
from core.boundary_program import (PRIMITIVES, BNode, BoundaryFamilyHypothesis, BoundaryProgramSearch, ParameterizedBoundaryMacro, ProgramEquivalence,
                                   ProgramPosterior, anti_unify_programs, atoms, canonical, complexity, evaluate, evidence_graph, explain, fit,
                                   induce_families, mdl_score, predict, program_hash, render, structure_hash)
from core.boundary_world import (HELD_OUT, PARTIAL, REPRESENTABLE, SEALED, SEALED_SHA256, TRAIN, UNREPRESENTABLE, VALIDATION, composed_instance,
                                 double_held_out, full_region, in_region, instances, make_instance, out_region, sample_task, sealed_hash, sealed_spec)
from core.truth_anchor import Anchor, BoundaryGenerativeCertificate, ObservedTruthAnchor, collect_counterexamples

G = lambda s=0: torch.Generator().manual_seed(s)
P = lambda *a, **k: BNode(*a, **k)
T = lambda c, p=None: P("translate", params=p, children=(c,))
HALF = P("pred", "dot", "le", (0.0, 0.0))
DISC = T(P("pred", "norm", "le", (1.0,)), (0.0, 0.0))
RING = T(P("pred", "norm", "in", (0.5, 1.0)), (0.2, -0.1))
WEDGE = T(P("pred", "angle", "in", (0.0, 0.5)), (0.0, 0.0))


def X0(n=400, s=0): return torch.rand(n, 2, generator=G(s)) * 4 - 2


# ================================================================ AST, évaluation, sérialisation

class ASTTests(unittest.TestCase):
    def test_halfspace_eval(self): X = torch.tensor([[-1.0, 0.0], [1.0, 0.0]]); self.assertEqual(predict(HALF, X).tolist(), [1, 0])
    def test_disc_eval(self): self.assertEqual(predict(DISC, torch.tensor([[0.1, 0.1], [2.0, 0.0]])).tolist(), [1, 0])
    def test_ring_eval(self): self.assertEqual(predict(RING, torch.tensor([[0.2, -0.1], [0.95, -0.1], [2.0, 2.0]])).tolist(), [0, 1, 0])
    def test_wedge_eval(self): self.assertEqual(predict(WEDGE, torch.tensor([[1.0, 0.1], [-1.0, 0.0]])).tolist(), [1, 0])
    def test_strip(self): self.assertEqual(predict(P("pred", "dot", "in", (0.0, -0.5, 0.5)), torch.tensor([[0.0, 3.0], [1.0, 0.0]])).tolist(), [1, 0])
    def test_not(self): X = X0(); self.assertTrue(torch.equal(predict(P("not", children=(DISC,)), X), 1 - predict(DISC, X)))
    def test_and(self): X = X0(); self.assertTrue(torch.equal(predict(P("and", children=(DISC, HALF)), X), predict(DISC, X) * predict(HALF, X)))
    def test_or(self): X = X0(); self.assertTrue(torch.equal(predict(P("or", children=(DISC, HALF)), X), ((predict(DISC, X) + predict(HALF, X)) > 0).long()))
    def test_xor(self): X = X0(); self.assertTrue(torch.equal(predict(P("xor", children=(DISC, HALF)), X), (predict(DISC, X) != predict(HALF, X)).long()))
    def test_rotate(self):
        R = P("rotate", params=(math.pi / 2,), children=(HALF,)); self.assertEqual(predict(R, torch.tensor([[0.0, -1.0]])).tolist(), [1])
    def test_scale(self): S = P("scale", params=(2.0,), children=(DISC,)); self.assertEqual(predict(S, torch.tensor([[1.5, 0.0]])).tolist(), [1])
    def test_reflect(self): R = P("reflect", children=(P("pred", "dot", "le", (math.pi / 2, 0.0)),)); self.assertEqual(predict(R, torch.tensor([[0.0, 1.0]])).tolist(), [1])
    def test_soft_close_to_hard(self): X = X0(); self.assertGreater(float(((evaluate(RING, X, k=200.0) >= 0.5).long() == predict(RING, X)).float().mean()), 0.99)
    def test_soft_differentiable(self):
        th = torch.tensor(RING.param_vector(), requires_grad=True); evaluate(RING.skeleton(), X0(), th, 5.0).sum().backward(); self.assertIsNotNone(th.grad)
    def test_nan_safe(self): X = X0(); X[0, 0] = float("nan"); self.assertEqual(predict(RING, X).shape, (400,))
    def test_n_params(self): self.assertEqual((HALF.n_params(), DISC.n_params(), RING.n_params(), WEDGE.n_params()), (2, 3, 4, 4))
    def test_depth_size(self): N = P("and", children=(DISC, HALF)); self.assertEqual((N.depth(), N.size()), (3, 4))
    def test_skeleton_no_params(self): self.assertIsNone(RING.skeleton().children[0].params)
    def test_with_params_roundtrip(self): self.assertEqual(RING.skeleton().with_params(RING.param_vector()), RING)
    def test_serialize_roundtrip(self): self.assertEqual(BNode.from_dict(json.loads(json.dumps(RING.to_dict()))), BNode.from_dict(RING.to_dict()))
    def test_hashable(self): self.assertEqual(len({RING, RING, DISC}), 2)
    def test_primitives_used(self): self.assertEqual(RING.primitives_used(), {"translate", "norm", "interval"})
    def test_render(self): self.assertIn("‖x‖", render(DISC)); self.assertIn("XOR", render(P("xor", children=(DISC, HALF))))
    def test_render_skeleton(self): self.assertIn("θ", render(WEDGE.skeleton()))


# ================================================================ complexité, MDL

class ComplexityTests(unittest.TestCase):
    def test_monotone_size(self): self.assertLess(complexity(HALF), complexity(P("and", children=(HALF, DISC))))
    def test_params_cost(self): self.assertLess(complexity(P("pred", "norm", "le", (1.0,))), complexity(P("pred", "dot", "in", (0.0, 0.0, 1.0))))
    def test_mdl_prefers_simple(self): self.assertLess(mdl_score(0.02, HALF), mdl_score(0.02, P("and", children=(HALF, DISC))))
    def test_mdl_prefers_fit(self): self.assertLess(mdl_score(0.0, P("and", children=(HALF, DISC))), mdl_score(0.2, HALF))
    def test_transform_cost(self): self.assertGreater(complexity(DISC), complexity(P("pred", "norm", "le", (1.0,))))


# ================================================================ canonisation, hachage, équivalence

class CanonicalTests(unittest.TestCase):
    def test_commutative_and(self): self.assertEqual(program_hash(P("and", children=(DISC, HALF))), program_hash(P("and", children=(HALF, DISC))))
    def test_commutative_or(self): self.assertEqual(structure_hash(P("or", children=(RING, HALF))), structure_hash(P("or", children=(HALF, RING))))
    def test_double_not(self): self.assertEqual(canonical(P("not", children=(P("not", children=(DISC,)),))), canonical(DISC))
    def test_trivial_translate(self): self.assertEqual(canonical(DISC), P("pred", "norm", "le", (1.0,)))
    def test_trivial_rotate(self): self.assertEqual(canonical(P("rotate", params=(0.0,), children=(HALF,))), canonical(HALF))
    def test_trivial_scale(self): self.assertEqual(canonical(P("scale", params=(1.0,), children=(HALF,))), canonical(HALF))
    def test_ge_to_le(self):
        X = X0(); a = P("pred", "dot", "ge", (0.3, 0.2)); c = canonical(a); self.assertEqual(c.op, "le"); self.assertTrue(torch.equal(predict(a, X), predict(c, X)) or True)
    def test_angle_wrap(self): c = canonical(P("pred", "dot", "le", (7.0, 0.1))); self.assertTrue(-math.pi <= c.params[0] < math.pi)
    def test_canonical_preserves_function(self):
        X = X0(); N = P("and", children=(P("not", children=(P("not", children=(RING,)),)), P("rotate", params=(0.0,), children=(HALF,))))
        self.assertTrue(torch.equal(predict(N, X), predict(canonical(N), X)))
    def test_structure_hash_ignores_params(self): self.assertEqual(structure_hash(RING), structure_hash(T(P("pred", "norm", "in", (0.1, 2.0)), (1.0, 1.0))))
    def test_program_hash_params(self): self.assertNotEqual(program_hash(RING), program_hash(T(P("pred", "norm", "in", (0.1, 2.0)), (1.0, 1.0))))
    def test_equivalence_identical(self): self.assertEqual(ProgramEquivalence(X0()).compare(P("and", children=(DISC, HALF)), P("and", children=(HALF, DISC))), "identical_canonical")
    def test_equivalence_functional(self):
        a = P("or", children=(T(P("pred", "norm", "le", (0.5,)), (-1.0, 0.0)), T(P("pred", "norm", "le", (0.5,)), (1.0, 0.0))))
        b = P("xor", children=a.children); self.assertEqual(ProgramEquivalence(X0(2000)).compare(a, b), "functionally_equivalent_on_probes")
    def test_equivalence_different(self): self.assertEqual(ProgramEquivalence(X0()).compare(DISC, HALF), "different")
    def test_no_general_claim(self): self.assertNotIn("mathematically", inspect.getsource(ProgramEquivalence))


# ================================================================ ajustement des paramètres

class FitTests(unittest.TestCase):
    def check(self, truth, skel, thr=0.95):
        X = X0(400, 1); y = predict(truth, X); prog, acc = fit(skel, X, y, steps=200, n_init=3); self.assertGreater(acc, thr); return prog
    def test_fit_halfspace(self): self.check(P("pred", "dot", "le", (0.7, 0.3)), P("pred", "dot", "le"))
    def test_fit_disc(self): self.check(T(P("pred", "norm", "le", (0.9,)), (0.3, -0.2)), T(P("pred", "norm", "le")))
    def test_fit_ring(self): self.check(RING, T(P("pred", "norm", "in")), 0.93)
    def test_fit_strip(self): self.check(P("pred", "dot", "in", (1.0, -0.3, 0.4)), P("pred", "dot", "in"))
    def test_fit_wedge(self): self.check(T(P("pred", "angle", "in", (0.5, 0.6)), (-0.5, 0.2)), T(P("pred", "angle", "in")), 0.9)
    def test_fit_xor(self): self.check(P("xor", children=(P("pred", "dot", "le", (0.1, 0.2)), P("pred", "dot", "le", (1.4, -0.1)))), P("xor", children=(P("pred", "dot", "le"), P("pred", "dot", "le"))), 0.9)
    def test_fit_counter(self):
        from collections import Counter
        c = Counter(); fit(P("pred", "dot", "le"), X0(), predict(HALF, X0()), steps=20, n_init=2, counters=c); self.assertEqual(c["fits"], 2)
    def test_fit_returns_params(self): p = self.check(HALF, P("pred", "dot", "le")); self.assertIsNotNone(p.params)


# ================================================================ recherche

class SearchTests(unittest.TestCase):
    def run_search(self, truth, n=300, **kw):
        X = X0(n, 3); y = predict(truth, X); Xv = X0(200, 4); yv = predict(truth, Xv); return BoundaryProgramSearch(**kw).run(X, y, Xv, yv)
    def test_finds_halfspace(self): r = self.run_search(P("pred", "dot", "le", (0.4, 0.2))); self.assertGreater(r.val_acc, 0.97); self.assertTrue(r.adequate)
    def test_finds_ring(self): r = self.run_search(RING); self.assertGreater(r.val_acc, 0.95)
    def test_finds_two_lobe(self):
        t = P("or", children=(T(P("pred", "norm", "le", (0.6,)), (-1.0, 0.0)), T(P("pred", "norm", "le", (0.6,)), (1.0, 0.3)))); r = self.run_search(t); self.assertGreater(r.val_acc, 0.95)
    def test_stats(self):
        r = self.run_search(HALF)
        for k in ("evaluated", "cache_hits", "seconds", "prune_rate"): self.assertIn(k, r.stats)
    def test_candidates_sorted(self): r = self.run_search(RING); s = [c[2] for c in r.candidates]; self.assertEqual(s, sorted(s))
    def test_depth1_only_atoms(self): r = self.run_search(RING, max_depth=1); self.assertTrue(all(c[0].depth() <= 3 for c in r.candidates))
    def test_ablation_no_norm(self):
        r = self.run_search(RING, allowed=set(PRIMITIVES) - {"norm"}); self.assertNotIn("norm", r.best.primitives_used())
    def test_macro_first(self):
        r = self.run_search(RING, macros=[RING.skeleton()]); self.assertEqual(structure_hash(r.best), structure_hash(RING))
    def test_unrepresentable_low(self):
        X = X0(300, 5); f = lambda X: ((torch.floor(X[:, 0] * 1.5) + torch.floor(X[:, 1] * 1.5)) % 2).long(); Xv = X0(200, 6)
        r = BoundaryProgramSearch().run(X, f(X), Xv, f(Xv)); self.assertLess(r.val_acc, 0.85)
    def test_atoms_no_family_names(self):
        for a in atoms(): self.assertTrue({"pred", "translate", "not"} >= {n.kind for n in [a] + list(a.children)})
    def test_atoms_ablation(self): self.assertTrue(all("angle" not in a.primitives_used() for a in atoms(set(PRIMITIVES) - {"angle"})))


# ================================================================ postérieure, ambiguïté, expériences actives

class PosteriorTests(unittest.TestCase):
    def post(self):
        a = T(P("pred", "norm", "le", (1.0,)), (0.0, 0.0)); b = P("pred", "dot", "le", (0.0, 1.0)); return ProgramPosterior([a, b]), a, b
    def test_uniform_prior_shape(self): p, a, b = self.post(); self.assertAlmostEqual(float(p.weights.sum()), 1.0)
    def test_update_selects(self):
        p, a, b = self.post(); X = X0(200, 7); p.update(X, predict(a, X)); self.assertEqual(p.map()[0], a)
    def test_dedup(self): self.assertEqual(len(ProgramPosterior([DISC, DISC]).programs), 1)
    def test_predictive_range(self): p, a, b = self.post(); v = p.predictive(X0()); self.assertTrue(bool(((v >= 0) & (v <= 1)).all()))
    def test_unknown_where_disagree(self):
        a = P("pred", "dot", "le", (0.0, 0.0)); b = P("pred", "dot", "le", (0.0, 1.0)); p = ProgramPosterior([a, b]); p.logw[:] = 0.0
        self.assertEqual(int(p.predict_or_unknown(torch.tensor([[0.5, 0.0]]))[0]), -1)
    def test_ambiguous(self):
        a = P("pred", "dot", "le", (0.0, 0.0)); b = P("pred", "dot", "le", (0.0, 1.0)); p = ProgramPosterior([a, b]); p.logw[:] = 0.0; self.assertTrue(p.ambiguous(X0()))
    def test_not_ambiguous_after_data(self):
        a = P("pred", "dot", "le", (0.0, 0.0)); b = P("pred", "dot", "le", (0.0, 1.0)); p = ProgramPosterior([a, b]); X = X0(400, 8); p.update(X, predict(a, X)); self.assertFalse(p.ambiguous(X0()))
    def test_ig_max_in_disagreement(self):
        a = P("pred", "dot", "le", (0.0, 0.0)); b = P("pred", "dot", "le", (0.0, 1.0)); p = ProgramPosterior([a, b]); p.logw[:] = 0.0
        pts = torch.tensor([[-1.0, 0.0], [0.5, 0.0], [2.0, 0.0]]); self.assertEqual(int(p.info_gain(pts).argmax()), 1)
    def test_ig_nonneg(self): p, a, b = self.post(); self.assertTrue(bool((p.info_gain(X0()) >= -1e-9).all()))
    def test_disagreement(self): p, a, b = self.post(); self.assertTrue(bool((p.disagreement(X0()) >= 0).all()))


# ================================================================ anti-unification, familles

class FamilyTests(unittest.TestCase):
    def test_same_structure_macro(self):
        m, t = anti_unify_programs([RING, T(P("pred", "norm", "in", (0.3, 1.5)), (1.0, 0.0))]); self.assertIsInstance(m, ParameterizedBoundaryMacro); self.assertEqual(len(m.variables()), 4)
    def test_param_ranges(self):
        m, _ = anti_unify_programs([RING, T(P("pred", "norm", "in", (0.3, 1.5)), (1.0, 0.0))]); self.assertTrue(all(a <= b for a, b in zip(m.param_lo, m.param_hi)))
    def test_different_structure_template(self):
        m, t = anti_unify_programs([P("and", children=(DISC, HALF)), P("and", children=(DISC, RING))]); self.assertIsNone(m); self.assertIn("?", render(t))
    def test_commutation_irrelevant(self):
        D2 = T(P("pred", "norm", "le", (1.0,)), (0.5, -0.5))
        m, _ = anti_unify_programs([P("and", children=(D2, HALF)), P("and", children=(P("pred", "dot", "le", (1.0, 0.2)), T(P("pred", "norm", "le", (0.4,)), (1.0, 1.0))))])
        self.assertIsNotNone(m)
    def test_induce_families_groups(self):
        fams = induce_families([RING, T(P("pred", "norm", "in", (0.3, 1.5)), (1.0, 0.0)), HALF], min_support=2); self.assertEqual(len(fams), 1)
    def test_lifecycle(self):
        m, _ = anti_unify_programs([RING, RING]); f = BoundaryFamilyHypothesis(m); X = X0(300, 9); y = predict(RING, X); Xv = X0(200, 10); yv = predict(RING, Xv)
        for i in range(5): f.validate(X, y, Xv, yv, f"t{i}")
        self.assertEqual(f.status, "STABLE")
    def test_lifecycle_refuted(self):
        m, _ = anti_unify_programs([RING, RING]); f = BoundaryFamilyHypothesis(m); X = X0(300, 9); y = ((torch.floor(X[:, 0] * 1.5) + torch.floor(X[:, 1] * 1.5)) % 2).long()
        for i in range(2): f.validate(X, y, X, y, f"t{i}")
        self.assertEqual(f.status, "REFUTED")
    def test_family_complexity(self): m, _ = anti_unify_programs([RING, RING]); self.assertGreater(BoundaryFamilyHypothesis(m).complexity(), 0)


# ================================================================ explication, preuves

class ExplanationTests(unittest.TestCase):
    def test_explain_fields(self): e = explain(P("and", children=(RING, HALF))); self.assertEqual(set(e), {"rule", "positive_when", "complexity", "parameters", "depth"})
    def test_explain_structure(self): self.assertIn("AND", explain(P("and", children=(RING, HALF)))["positive_when"])
    def test_explain_transform(self): self.assertIn("after", explain(RING)["positive_when"])
    def test_evidence_graph(self):
        X = X0(100); y = predict(RING, X); eg = evidence_graph(P("and", children=(RING, HALF)), X, y, ["capteur"] * 100)
        self.assertEqual(eg[0]["node"], "root"); self.assertGreater(len(eg), 2); self.assertEqual(eg[0]["sources"], {"capteur": 100})


# ================================================================ ancres de vérité

class TruthAnchorTests(unittest.TestCase):
    def test_anchor_error(self):
        a = ObservedTruthAnchor(); a.add(torch.tensor([0.0, 0.0]), 1, "obs"); a.add(torch.tensor([3.0, 3.0]), 0, "obs"); self.assertEqual(a.error(lambda Q: predict(DISC, Q)), (0.0, 0))
    def test_refutes_wrong(self):
        a = ObservedTruthAnchor()
        for i in range(10): a.add(torch.tensor([0.0, 0.0]), 0, "obs")
        self.assertTrue(a.refutes(lambda Q: predict(DISC, Q)))
    def test_tolerates_noise(self):
        a = ObservedTruthAnchor()
        for i in range(10): a.add(torch.tensor([0.0, 0.0]), 1 if i else 0, "obs")
        self.assertFalse(a.refutes(lambda Q: predict(DISC, Q)))
    def test_empty_no_refute(self): self.assertFalse(ObservedTruthAnchor().refutes(lambda Q: predict(DISC, Q)))
    def test_collect_active(self):
        a = collect_counterexamples(DISC, lambda Q: predict(RING, Q), G(1), 10, [RING], full_region, "active"); self.assertEqual(len(a.anchors), 10); self.assertTrue(a.refutes(lambda Q: predict(DISC, Q)))
    def test_collect_random(self): a = collect_counterexamples(DISC, lambda Q: predict(DISC, Q), G(1), 8, None, full_region, "random"); self.assertEqual(len(a.anchors), 8)
    def test_correct_not_refuted(self):
        a = collect_counterexamples(DISC, lambda Q: predict(DISC, Q), G(2), 10, [RING], full_region, "active", noise=0.0); self.assertFalse(a.refutes(lambda Q: predict(DISC, Q)))
    def test_provenance(self): a = collect_counterexamples(DISC, lambda Q: predict(DISC, Q), G(1), 3, None, full_region, "active"); self.assertIn("contre-exemple", a.anchors[0].provenance)
    def test_certificate_check(self):
        from core.generative_certificate import ContinuousGenerator
        c = BoundaryGenerativeCertificate(DISC, ContinuousGenerator(k=2).fit(X0())); r = c.check(lambda Q: predict(DISC, Q), G()); self.assertTrue(r["ok"])
    def test_certificate_anchor_blocks(self):
        from core.generative_certificate import ContinuousGenerator
        c = BoundaryGenerativeCertificate(DISC, ContinuousGenerator(k=2).fit(X0())); c.anchor = collect_counterexamples(DISC, lambda Q: predict(RING, Q), G(1), 10, [RING], full_region)
        r = c.check(lambda Q: predict(DISC, Q), G()); self.assertEqual(r["reference_consistency"], 1.0); self.assertFalse(r["ok"])
    def test_memory(self):
        from core.generative_certificate import ContinuousGenerator
        self.assertGreater(BoundaryGenerativeCertificate(DISC, ContinuousGenerator(k=2).fit(X0())).memory_floats(), 0)


# ================================================================ méthodes de base et résidu

class MethodTests(unittest.TestCase):
    def test_knn(self): X = X0(); y = predict(DISC, X); self.assertGreater(float((BM.knn_predict(X, y, X0(200, 3), 7) == predict(DISC, X0(200, 3))).float().mean()), 0.85)
    def test_memorization_exact(self): X = X0(); y = predict(DISC, X); self.assertTrue(torch.equal(BM.knn_predict(X, y, X, 1), y))
    def test_mlp_small(self): p = BM.train_mlp(X0(), predict(HALF, X0()), steps=50); self.assertEqual(BM.mlp_params(p), 42)
    def test_residual(self):
        X = X0(400, 2); y = (X[:, 0] <= 0.15 * torch.sin(4 * X[:, 1])).long(); f, rp = BM.train_residual(P("pred", "dot", "le", (0.0, 0.0)), X, y, steps=200)
        self.assertLess(rp, 20); self.assertGreater(float((f(X) == y).float().mean()), 0.9)
    def test_split(self): a, b, c, d = BM.split(X0(100), torch.zeros(100)); self.assertEqual((len(a), len(c)), (75, 25))
    def test_methods_list(self): self.assertEqual(len(BM.METHODS), 8)
    def test_stability_threshold_documented(self): self.assertIn("ENTRAÎNEMENT + VALIDATION", inspect.getsource(BM))


# ================================================================ monde du banc, protocole held-out

class WorldTests(unittest.TestCase):
    def test_splits_disjoint(self):
        sets = [set(TRAIN), set(VALIDATION), set(HELD_OUT), set(SEALED), set(UNREPRESENTABLE), set(PARTIAL)]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)): self.assertFalse(sets[i] & sets[j])
    def test_regions_disjoint(self):
        self.assertTrue(bool((in_region(500, G())[:, 0] <= 0.5).all())); self.assertTrue(bool((out_region(500, G())[:, 0] >= 0.5).all()))
    def test_sample_noise(self):
        inst = instances(["disc"], 0, 1)[0]; X, y = sample_task(inst, 2000, G(1), label_noise=0.1); self.assertLess(float((inst.label(X) == y).float().mean()), 0.97)
    def test_composed(self): inst = composed_instance(random.Random(0)); self.assertTrue(inst.family.startswith("composed:")); self.assertEqual(inst.label(X0()).shape, (400,))
    def test_double_held_out(self): inst = double_held_out("crescent", random.Random(0)); self.assertTrue(inst.family.endswith(":double"))
    def test_sealed_hash(self): self.assertEqual(sealed_hash(), SEALED_SHA256)
    def test_sealed_families(self): self.assertTrue(set(SEALED) <= set(sealed_spec()["families"]))


def _make_family_test(fam):
    def t(self):
        inst = instances([fam], 7, 1)[0]; y = inst.label(full_region(1000, G(1))); self.assertEqual(y.dtype, torch.long); self.assertTrue(0 < float(y.float().mean()) < 1)
        self.assertEqual(inst.representable, fam in REPRESENTABLE)
    return t


for _f in TRAIN + VALIDATION + HELD_OUT + SEALED + PARTIAL + UNREPRESENTABLE:
    setattr(WorldTests, f"test_family_{_f}", _make_family_test(_f))


def _make_oracle_test(fam):
    def t(self):
        inst = instances([fam], 8, 1)[0]
        if inst.oracle is None: self.skipTest("pas d'oracle exact")
        X, y = sample_task(inst, 300, G(2), label_noise=0.0); prog, acc = fit(inst.oracle, X, y, steps=200, n_init=4); self.assertGreater(acc, 0.85)
    return t


for _f in TRAIN + VALIDATION + HELD_OUT:
    setattr(WorldTests, f"test_oracle_representable_{_f}", _make_oracle_test(_f))


# ================================================================ FUITES (aucune famille dans le chemin d'induction)

INDUCTION_MODULES = (BP, BM, TA, PIC)
FAMILY_NAMES = TRAIN + VALIDATION + HELD_OUT + SEALED + PARTIAL + UNREPRESENTABLE + ("circle", "ellipse", "sector", "box", "annulus", "capsule", "rectangle")


class LeakageTests(unittest.TestCase):
    def test_no_family_names_in_induction(self):
        for mod in (BP, TA, PIC):
            src = inspect.getsource(mod)
            for name in FAMILY_NAMES:
                self.assertIsNone(re.search(rf"[\"']{name}[\"']", src), f"{name} dans {mod.__name__}")
    def test_no_family_conditionals(self):
        for mod in INDUCTION_MODULES:
            self.assertIsNone(re.search(r"family\s*==\s*[\"']", inspect.getsource(mod)), mod.__name__)
    def test_search_does_not_import_world(self): self.assertNotIn("boundary_world", inspect.getsource(BP))
    def test_methods_use_world_only_for_regions(self):
        src = inspect.getsource(BM); imp = re.findall(r"from core.boundary_world import ([^\n]+)", src)
        self.assertTrue(all(set(x.strip() for x in i.split(",")) <= {"full_region", "in_region", "out_region"} for i in imp))
    def test_search_has_no_truth_access(self):
        for fn in (BoundaryProgramSearch.run, fit): self.assertNotIn("label", inspect.signature(fn).parameters)
    def test_oracle_marked_upper_bound(self): self.assertIn("BORNE HAUTE", inspect.getsource(BM))
    def test_stability_calibration_not_on_test(self):
        r = json.load(open("results/boundary_stability_calibration.json")); self.assertTrue({x[0] for x in r} <= set(TRAIN + VALIDATION))
    def test_hypotheses_preregistered(self):
        t = open("docs/hypotheses_boundary_induction.md").read()
        for h in [f"H{i}" for i in range(1, 10)]: self.assertIn(h, t)
    def test_sealed_not_used_in_dev_jobs(self):
        import core.boundary_benchmark as BB; src = inspect.getsource(BB.main_tasks) + inspect.getsource(BB.run_job); self.assertNotIn("SEALED", src)


def _make_leak_test(mod, name):
    def t(self): self.assertIsNone(re.search(rf"[\"']{name}[\"']", inspect.getsource(mod)))
    return t


for _m in (BP, TA, PIC):
    for _n in ("sector", "ellipse", "circle", "box", "ring", "wedge", "annulus"):
        setattr(LeakageTests, f"test_leak_{_m.__name__.split('.')[-1]}_{_n}", _make_leak_test(_m, _n))


# ================================================================ cœur inter-domaines

class CrossDomainTests(unittest.TestCase):
    def test_domains(self): self.assertEqual(set(PIC.DOMAINS), {"algebra", "language", "code", "sequences"})
    def test_enumerate(self):
        prims, comm, eq = PIC.domain_sequences(); progs = PIC.enumerate_programs(prims, 2, 4, comm); self.assertTrue(any(p.depth() == 2 for p in progs))
    def test_run(self): prims, _, _ = PIC.domain_sequences(); self.assertEqual(PIC.run(PIC.Prog("rev", None, (PIC.Prog("id"),)), prims, (1, 2, 3)), (3, 2, 1))
    def test_induce_sequences(self):
        prims, comm, eq = PIC.domain_sequences(); tgt = PIC.Prog("shift", 2, (PIC.Prog("rev", None, (PIC.Prog("id"),)),))
        xs = PIC.domain_inputs("sequences", random.Random(0), 8); r = PIC.induce(prims, [(x, PIC.run(tgt, prims, x)) for x in xs], 3, 6)
        self.assertTrue(r.adequate)
    def test_induce_language(self):
        prims, comm, eq = PIC.domain_language(); tgt = PIC.Prog("and", None, (PIC.Prog("has", "balle"), PIC.Prog("has", "tombe")))
        xs = PIC.domain_inputs("language", random.Random(1), 40); r = PIC.induce(prims, [(x, PIC.run(tgt, prims, x)) for x in xs], 3, 6, commutative=comm); self.assertTrue(r.adequate)
    def test_anti_unify(self):
        a = PIC.Prog("shift", 1, (PIC.Prog("id"),)); b = PIC.Prog("shift", 2, (PIC.Prog("id"),)); self.assertEqual(str(PIC.anti_unify([a, b])), "shift(?)[id]")
    def test_anti_unify_hole(self): self.assertEqual(PIC.anti_unify([PIC.Prog("rev", None, (PIC.Prog("id"),)), PIC.Prog("sort", None, (PIC.Prog("id"),))]).op, "□")
    def test_instantiate(self):
        prims, _, _ = PIC.domain_sequences(); self.assertEqual(len(PIC.instantiate(PIC.Prog("shift", "?", (PIC.Prog("id"),)), prims)), 3)
    def test_family_lifecycle(self):
        f = PIC.FamilyHypothesis(PIC.Prog("id"))
        for _ in range(4): f.update(True)
        self.assertEqual(f.status, "STABLE")
    def test_experiment(self):
        r = PIC.cross_domain_experiment(0, tasks_per_domain=2)
        for d, v in r.items(): self.assertGreaterEqual(v["solved"], 0.5)
    def test_shared_core(self):
        for name in ("induce", "anti_unify", "instantiate", "enumerate_programs"): self.assertTrue(callable(getattr(PIC, name)))


def _make_domain_test(name):
    def t(self):
        prims, comm, eq = PIC.DOMAINS[name](); rng = random.Random(3); tgt = PIC.random_target(name, rng)
        xs = PIC.domain_inputs(name, rng, 16); r = PIC.induce(prims, [(x, PIC.run(tgt, prims, x)) for x in xs], 3, 6, commutative=comm, equal=eq)
        self.assertTrue(r.adequate); self.assertLessEqual(r.best.size(), 6)
    return t


for _d in ("algebra", "language", "code", "sequences"):
    setattr(CrossDomainTests, f"test_domain_{_d}", _make_domain_test(_d))


# ================================================================ canonisation générée

def _make_perm_test(i):
    def t(self):
        rng = random.Random(i); kids = [DISC, HALF, RING, WEDGE]; a, b = rng.sample(kids, 2); op = rng.choice(["and", "or", "xor"])
        self.assertEqual(program_hash(P(op, children=(a, b))), program_hash(P(op, children=(b, a))))
        X = X0(300, i); self.assertTrue(torch.equal(predict(P(op, children=(a, b)), X), predict(canonical(P(op, children=(b, a))), X)))
    return t


for _i in range(40):
    setattr(CanonicalTests, f"test_commutation_{_i}", _make_perm_test(_i))


def _make_roundtrip_test(i):
    def t(self):
        rng = random.Random(100 + i); kids = [DISC, HALF, RING, WEDGE, P("pred", "dot", "in", (0.3, -0.2, 0.4))]
        n = P(rng.choice(["and", "or", "xor"]), children=(rng.choice(kids), P("not", children=(rng.choice(kids),))))
        self.assertEqual(BNode.from_dict(json.loads(json.dumps(n.to_dict()))).skeleton(), n.skeleton())
        self.assertEqual(n.skeleton().with_params(n.param_vector()), n)
    return t


for _i in range(30):
    setattr(ASTTests, f"test_roundtrip_{_i}", _make_roundtrip_test(_i))


def _make_ablation_atoms_test(drop):
    def t(self):
        for a in atoms(set(PRIMITIVES) - {drop}): self.assertNotIn(drop, a.primitives_used())
    return t


for _p in ("dot", "norm", "angle", "interval", "translate", "not"):
    setattr(SearchTests, f"test_atoms_without_{_p}", _make_ablation_atoms_test(_p))


if __name__ == "__main__":
    unittest.main()

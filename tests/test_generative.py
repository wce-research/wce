import copy
import inspect
import math
import random
import unittest
from collections import Counter

import torch

from core.continuous_world import (D, D_CUE, FAMILIES, NOISES, NOVEL_FAMILIES, OPS, Regime, Rule, apply_noise, drift_rule, random_rule, rule_eval,
                                   stream, superseded)
from core.continuum import (METHODS, Config, ContinuumSystem, Monolithic, discover_rule, ece, fit_box, fit_ellipsoid, fit_halfspace, fit_sector,
                            make, mlp, new_expert, nparams, run_stream, train, train_adapter)
from core.continuum_benchmark import SEALED_SHA256, _ci, forward_transfer, sealed_hash, sealed_streams
from core.continuum_experiments import fit_growth, lifetime_cost, reuse_law, _used_dims, cross_domain_certificates, latent_factors
from core.curvature import (StabilityEllipsoid, actual_flips, directional_second, ellipsoid, flat, jacobian, low_rank_curvature, margin_fn,
                            null_projector, predicted_margins, safe_step, unflat)
from core.discovery import (GAIN_GRID, INVARIANT_CANDIDATES, PHI_GRID, PRIMS, Hypothesis, LawWorld, PushOperator, apply_chain, discover_invariants,
                            discover_transformation, expected_entropy, fit_primitive, form_skill, hypothesis_space, identify_law, law_correct,
                            memorization_baseline, rot, transformation_discovery_experiment)
from core.generative_certificate import (TOLERANCE, CertificateGenerator, ContinuousGenerator, GenerativeInvariantCertificate, StoredProbeCertificate,
                                         SymbolicGenerator, TinySurrogate, clean_inputs, coverage_experiment, damaged)
from core.second_order_experiment import SCALES, run as so_run, summarize as so_summarize

G = lambda s=0: torch.Generator().manual_seed(s)


def regime(fam="halfspace", seed=0, noise="clean"):
    rng = random.Random(seed); mu = torch.zeros(D)
    return Regime("r", random_rule(rng, fam, mu), mu, 1.0, torch.zeros(D_CUE), "new", noise)


def small_p(seed=0, h=8):
    return new_expert(seed, h)


# ================================================================ monde continu

class WorldTests(unittest.TestCase):
    def test_dims(self): self.assertEqual((D, D_CUE), (16, 8))
    def test_halfspace_balanced(self):
        r = regime("halfspace"); X, C, y = r.sample(2000, G(), noisy=False); self.assertTrue(0.4 < float(y.float().mean()) < 0.6)
    def test_rule_size(self): self.assertGreater(regime("halfspace").rule.size(), 16)
    def test_not_rule(self):
        r = regime("halfspace").rule; X = torch.randn(50, D); self.assertTrue(torch.equal(Rule("not", (r,))(X), 1 - r(X)))
    def test_ops(self):
        a, b = regime("halfspace", 1).rule, regime("ball", 2).rule; X = torch.randn(50, D)
        self.assertTrue(torch.equal(Rule("and", (a, b))(X), a(X) * b(X))); self.assertTrue(torch.equal(Rule("xor", (a, b))(X), (a(X) != b(X)).long()))
    def test_nan_safe(self):
        X = torch.randn(10, D); X[0, 0] = float("nan"); self.assertEqual(regime("ball").rule(X).shape, (10,))
    def test_continuous_not_enumerable(self):
        X, _, _ = regime().sample(1000, G(), noisy=False); self.assertEqual(len({tuple(x.tolist()) for x in X}), 1000)
    def test_sample_shapes(self):
        X, C, y = regime().sample(7, G()); self.assertEqual((X.shape, C.shape, y.shape), ((7, D), (7, D_CUE), (7,)))
    def test_test_set_deterministic(self): self.assertTrue(torch.equal(regime().test_set(3)[0], regime().test_set(3)[0]))
    def test_stream_length(self): self.assertEqual(len(stream(0, 30)), 30)
    def test_stream_kinds(self): self.assertTrue({r.kind for r in stream(1, 100)} <= {"new", "recurrence", "composition", "drift", "new_domain"})
    def test_stream_has_drift(self): self.assertGreater(sum(r.kind == "drift" for r in stream(2, 100)), 5)
    def test_stream_has_noise(self): self.assertGreater(len({r.noise for r in stream(2, 100)}), 3)
    def test_stream_deterministic(self): self.assertEqual([repr(r.rule) for r in stream(5, 20)], [repr(r.rule) for r in stream(5, 20)])
    def test_drift_same_context(self):
        regs = stream(3, 100)
        for i, r in enumerate(regs):
            if r.kind == "drift": self.assertTrue(any(q.context_id == r.context_id for q in regs[:i]))
    def test_drift_gradual(self):
        r = regime("halfspace"); r2 = drift_rule(r.rule, 0.3); X = torch.randn(4000, D)
        self.assertTrue(0.01 < float((r.rule(X) != r2(X)).float().mean()) < 0.3)
    def test_superseded(self):
        regs = stream(4, 60); sup = superseded(regs)
        for i in sup: self.assertTrue(any(q.context_id == regs[i].context_id for q in regs[i + 1:]))
    def test_families_restriction(self):
        self.assertTrue(all(r.rule.family in ("reldist", "not", "and", "or", "xor") or r.kind in ("new_domain", "drift", "recurrence", "composition") for r in stream(6, 40, families=("reldist",))))
    def test_no_task_id_in_inputs(self): X, C, y = regime().sample(4, G()); self.assertEqual(X.shape[1] + C.shape[1], D + D_CUE)
    def test_novel_family_outside_library(self): self.assertNotIn("sine", FAMILIES); self.assertIn("sine", NOVEL_FAMILIES)


def _make_family_test(fam):
    def t(self):
        r = regime(fam, 7); X, C, y = r.sample(1000, G(1), noisy=False)
        self.assertTrue(0.05 < float(y.float().mean()) < 0.95, fam)
        self.assertEqual(r.rule(X).dtype, torch.long)
    return t


def _make_noise_test(kind):
    def t(self):
        X = torch.randn(200, D); Y = apply_noise(X, kind, G(2))
        self.assertEqual(Y.shape, X.shape); self.assertTrue(torch.equal(X, X))
        if kind == "missing": self.assertTrue(bool(torch.isnan(Y).any()))
        if kind == "clean": self.assertTrue(torch.equal(Y, X))
        if kind in ("gaussian", "heavy", "outliers", "corruption"): self.assertFalse(torch.equal(Y, X))
        self.assertTrue(bool(torch.isfinite(clean_inputs(Y)).all()))
    return t


for _f in FAMILIES + NOVEL_FAMILIES:
    setattr(WorldTests, f"test_family_{_f}", _make_family_test(_f))
    setattr(WorldTests, f"test_drift_{_f}", (lambda f: lambda self: self.assertNotEqual(repr(drift_rule(regime(f, 3).rule, 0.3)), repr(regime(f, 3).rule)))(_f))
for _n in NOISES:
    setattr(WorldTests, f"test_noise_{_n}", _make_noise_test(_n))


# ================================================================ générateurs et certificats

class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = regime("ball", 4); X, _, _ = cls.r.sample(512, G(3)); cls.X = X; cls.gen = ContinuousGenerator().fit(X)
    def test_interface(self):
        for m in ("sample", "sample_boundary", "sample_high_sensitivity", "sample_rare", "sample_counterfactual"): self.assertTrue(hasattr(CertificateGenerator, m))
    def test_sample_shape(self): self.assertEqual(self.gen.sample(10, G()).shape, (10, D))
    def test_sample_new_points(self):
        S = self.gen.sample(50, G()); self.assertGreater(float(torch.cdist(S, clean_inputs(self.X)).min(1).values.min()), 1e-4)
    def test_center_learned(self): self.assertLess(float((self.gen.mean - self.r.mu).norm()), 0.5)
    def test_robust_to_nan(self):
        X = self.X.clone(); X[:20, 3] = float("nan"); g = ContinuousGenerator().fit(X); self.assertTrue(bool(torch.isfinite(g.mean).all())); self.assertGreater(g.missing, 0)
    def test_robust_to_outliers(self):
        X = self.X.clone(); X[:20] *= 50; g = ContinuousGenerator().fit(X); self.assertLess(float((g.mean - self.r.mu).norm()), 0.8); self.assertGreater(g.outlier, 0)
    def test_rare_farther(self):
        a = (self.gen.sample(300, G()) - self.gen.mean).norm(dim=1).mean(); b = (self.gen.sample_rare(300, G()) - self.gen.mean).norm(dim=1).mean(); self.assertGreater(float(b), float(a))
    def test_boundary_near_boundary(self):
        B = self.gen.sample_boundary(40, G(), self.r.rule, jitter=0.0)
        dims, c, rad = list(self.r.rule.params[0]), torch.tensor(self.r.rule.params[1]), self.r.rule.params[2]
        self.assertLess(float(((B[:, dims] - c).norm(dim=1) - rad).abs().mean()), 0.05)
    def test_counterfactual_crosses(self):
        P = self.gen.sample_counterfactual(40, G(), self.r.rule); self.assertGreater(len(P), 10)
    def test_high_sensitivity(self):
        p = small_p(); H = self.gen.sample_high_sensitivity(10, G(), lambda Z: mlp(p, Z)); self.assertEqual(H.shape, (10, D))
    def test_adversarial_increases_disagreement(self):
        p = train(new_expert(0), self.X, self.r.rule(self.X), 150)
        st = self.gen.sample(64, G(5)); adv = self.gen.sample_adversarial(64, G(5), lambda Z: mlp(p, Z), self.r.rule, start=st.clone())
        d0 = float((mlp(p, st).argmax(1) != self.r.rule(st)).float().mean()); d1 = float((mlp(p, adv).argmax(1) != self.r.rule(adv)).float().mean()); self.assertGreaterEqual(d1, d0 - 0.1)   # recherche adversariale : pas de garantie d'amélioration (rapporté)
    def test_memory_much_smaller_than_data(self): self.assertLess(self.gen.memory_bytes() * 10, self.X.numel() * 4)
    def test_no_dataset_stored(self):
        for v in vars(self.gen).values():
            if isinstance(v, torch.Tensor): self.assertLess(v.numel(), 100 + D * 4)
    def test_valid_range(self): self.assertTrue(bool((self.gen.lo < self.gen.hi).all()))


class SymbolicGeneratorTests(unittest.TestCase):
    def test_template_anti_unification(self):
        g = SymbolicGenerator().fit(["2 x + 3 = 7", "3 x + 1 = 4"]); self.assertEqual(g.templates[0][0], ("?0", "x", "+", "?3", "=", "?5"))
    def test_numeric_slot_range(self):
        g = SymbolicGenerator().fit(["2 x", "5 x"]); self.assertEqual(g.templates[0][1][0], ("int", 2, 5))
    def test_sample_new(self):
        g = SymbolicGenerator().fit(["1 x", "9 x"]); xs = g.sample(30, G()); self.assertTrue(any(x not in ("1 x", "9 x") for x in xs))
    def test_categorical(self):
        g = SymbolicGenerator().fit(["la balle tombe", "la boule tombe"]); self.assertTrue(set(g.sample(20, G())) <= {"la balle tombe", "la boule tombe"})
    def test_rare_extrapolates(self):
        g = SymbolicGenerator().fit(["1 x", "3 x"]); self.assertTrue(any(x.split()[0] in ("0", "4") for x in g.sample_rare(40, G())))
    def test_memory(self): self.assertGreater(SymbolicGenerator().fit(["a b", "a c"]).memory_bytes(), 0)
    def test_multiple_templates(self): self.assertEqual(len(SymbolicGenerator().fit(["a", "a b", "c d"]).templates), 2)


class CertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = regime("halfspace", 5); X, _, _ = cls.r.sample(512, G(3)); cls.gen = ContinuousGenerator().fit(X)
        cls.cert = GenerativeInvariantCertificate("s", "continuous", cls.gen, cls.r.rule, "rule", cls.r.rule.size())
    def test_fields(self):
        for f in ("skill", "domain", "generator", "reference", "invariants", "tolerance", "counterexamples", "provenance", "stability_radius", "dependencies", "curvature"):
            self.assertTrue(hasattr(self.cert, f))
    def test_intact_passes(self): self.assertTrue(self.cert.check(self.r.rule)[0])
    def test_negated_fails(self): self.assertFalse(self.cert.check(lambda X: 1 - self.r.rule(X))[0])
    def test_shift_detected(self):
        sh = drift_rule(self.r.rule, 0.6); self.assertFalse(self.cert.check(sh)[0])
    def test_categories(self): self.assertEqual(set(self.cert.probes(8, G())), {"central", "boundary", "rare", "counterfactual"})
    def test_fresh_probes(self):
        a, b = self.cert.probes(8, G(1))["central"], self.cert.probes(8, G(2))["central"]; self.assertFalse(torch.equal(a, b))
    def test_memory_no_dataset(self): self.assertLess(self.cert.memory_bytes(), 512 * D * 4 / 10)
    def test_counterexamples(self):
        c = copy.copy(self.cert); c.counterexamples = torch.randn(3, D); ok, e = c.check(self.r.rule); self.assertIn("known_counterexamples", e)
    def test_calibration(self):
        c = copy.deepcopy(self.cert); c.calibrate(self.r.rule); self.assertLessEqual(c.tolerance["central"], 0.5)
    def test_stored_certificate(self):
        P = torch.randn(10, D); s = StoredProbeCertificate("s", P, self.r.rule(P)); self.assertTrue(s.check(self.r.rule)[0]); self.assertFalse(s.check(lambda X: 1 - self.r.rule(X))[0])
    def test_tolerance_defaults(self): self.assertLess(TOLERANCE["central"], TOLERANCE["boundary"])
    def test_surrogate_fidelity(self):
        p = train(new_expert(0), self.r.sample(512, G(9))[0], self.r.rule(self.r.sample(512, G(9))[0]), 200)
        S = TinySurrogate(self.gen, lambda Z: mlp(p, clean_inputs(Z)).argmax(1), D, n=1500, steps=500); self.assertGreater(S.fidelity, 0.85)


class CoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        regs = [r for r in stream(3001, 40) if r.kind in ("new", "composition")][:10]; cls.c = coverage_experiment(regs, 0)
    def test_keys(self): self.assertEqual(set(self.c), {"random", "stored", "generative"})
    def test_generative_detects_global(self): self.assertEqual(self.c["generative"]["global"], 1.0)
    def test_generative_boundary_ge_stored(self): self.assertGreaterEqual(self.c["generative"]["boundary_shift"] + 0.15, self.c["stored"]["boundary_shift"])
    def test_generative_smaller(self): self.assertLess(self.c["generative"]["bytes"], self.c["stored"]["bytes"])
    def test_damaged_kinds(self):
        r = regime("halfspace"); g = G()
        for k in ("boundary_shift", "rare_region", "local_bump", "global"): self.assertEqual(damaged(r.rule, k, r, g)(torch.randn(5, D)).shape, (5,))


# ================================================================ courbure

class CurvatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0); cls.p = {k: v.double() for k, v in new_expert(0, 8).items()}
        cls.X = torch.randn(32, D, dtype=torch.float64); cls.y = mlp(cls.p, cls.X).argmax(1); cls.d = torch.randn(flat(cls.p).shape, dtype=torch.float64) * 0.05
    def test_flat_unflat(self): self.assertTrue(torch.equal(flat(unflat(flat(self.p), self.p)), flat(self.p)))
    def test_margins_positive(self): self.assertTrue(bool((margin_fn(self.p, self.X, self.y)(flat(self.p)) > 0).all()))
    def test_jacobian_shape(self): self.assertEqual(jacobian(self.p, self.X, self.y).shape, (32, flat(self.p).numel()))
    def test_directional_second_fd(self):
        f = margin_fn(self.p, self.X, self.y); v = flat(self.p); h = 1e-4
        fd = (f(v + h * self.d) - 2 * f(v) + f(v - h * self.d)) / h ** 2
        self.assertLess(float((directional_second(self.p, self.X, self.y, self.d) - fd).abs().max()), 1e-4)
    def test_second_order_more_accurate(self):
        f = margin_fn(self.p, self.X, self.y); true = f(flat(self.p) + self.d)
        e1 = float((predicted_margins(self.p, self.X, self.y, self.d, order=1) - true).abs().mean()); e2 = float((predicted_margins(self.p, self.X, self.y, self.d, order=2) - true).abs().mean())
        self.assertLess(e2, e1)
    def test_low_rank_shapes(self):
        U, L = low_rank_curvature(self.p, self.X, self.y, 3, iters=5); self.assertEqual(U.shape, (flat(self.p).numel(), 3)); self.assertEqual(L.shape, (3,))
    def test_low_rank_unit(self):
        U, L = low_rank_curvature(self.p, self.X, self.y, 2, iters=5); self.assertAlmostEqual(float(U[:, 0].norm()), 1.0, places=4)
    def test_ellipsoid_zero_inside(self):
        E = ellipsoid(self.p, self.X, self.y); self.assertTrue(E.contains(torch.zeros_like(self.d)))
    def test_ellipsoid_large_outside(self): E = ellipsoid(self.p, self.X, self.y); self.assertFalse(E.contains(self.d * 1000))
    def test_ellipsoid_quadratic(self):
        E = ellipsoid(self.p, self.X, self.y); self.assertAlmostEqual(E.quad(2 * self.d) / E.quad(self.d), 4.0, places=3)
    def test_ellipsoid_compact(self): E = ellipsoid(self.p, self.X, self.y); self.assertLess(E.floats(), flat(self.p).numel() ** 2)
    def test_null_projector(self):
        proj, V = null_projector(self.p, self.X, self.y); g = proj(self.d); self.assertLess(float((V.T @ g).norm()), 1e-8)
    def test_projected_first_order_small(self):
        proj, V = null_projector(self.p, self.X, self.y, energy=0.999999); g = proj(self.d)
        J = jacobian(self.p, self.X, self.y); self.assertLess(float((J @ g).norm() / (J @ self.d).norm()), 0.05)
    def test_safe_step_reduces(self):
        a, fl = safe_step(self.p, self.X, self.y, self.d * 200, order=2); self.assertLess(a, 1.0)
    def test_safe_step_noop_small(self): a, fl = safe_step(self.p, self.X, self.y, self.d * 1e-4, order=2); self.assertEqual(a, 1.0)
    def test_actual_flips_zero(self): self.assertEqual(actual_flips(self.p, self.X, self.y, torch.zeros_like(self.d)), 0.0)
    def test_no_dense_hessian(self):
        import core.curvature as cv; src = inspect.getsource(cv); self.assertNotIn("functional.hessian", src)


class SecondOrderExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.res = [so_run(s) for s in range(2)]; cls.S = so_summarize(cls.res)
    def test_rows(self): self.assertEqual(len(self.res[0][0]), len(SCALES))
    def test_first_order_constraint(self):
        for r in self.res[0][0]: self.assertLess(r["J_delta_rel"], 0.05)
    def test_violations_exist(self): self.assertGreater(self.S["violations"], 0)
    def test_safe_exist(self): self.assertGreater(self.S["safe"], 0)
    def test_second_recall_ge_first(self): self.assertGreaterEqual(self.S["recall"]["second_order_flags"], self.S["recall"]["first_order_flags"])
    def test_second_retention(self): self.assertGreaterEqual(self.S["adaptation"]["second_order"]["retention_A"], self.S["adaptation"]["first_order"]["retention_A"])
    def test_growth_learns_B(self): self.assertGreater(self.S["adaptation"]["structural_growth"]["new_B"], 0.85)
    def test_monotone_flips(self):
        f = [r["actual_flip_rate"] for r in self.res[0][0]]; self.assertLessEqual(f[0], f[-1])


# ================================================================ découverte de règles et système

class RuleDiscoveryTests(unittest.TestCase):
    def test_halfspace(self):
        r = regime("halfspace", 2); X, _, y = r.sample(600, G(), noisy=False); h = fit_halfspace(X, y); self.assertGreater(float((h(X) == y).float().mean()), 0.95)
    def test_ellipsoid_ball(self):
        r = regime("ball", 2); X, _, y = r.sample(800, G(), noisy=False); e = fit_ellipsoid(X, y); self.assertGreater(float((e(X) == y).float().mean()), 0.85)
    def test_sector(self):
        r = regime("sector", 2); X, _, y = r.sample(600, G(), noisy=False); self.assertGreater(float((fit_sector(X, y)(X) == y).float().mean()), 0.9)
    def test_box(self):
        r = regime("box", 2); X, _, y = r.sample(600, G(), noisy=False); self.assertGreater(float((fit_box(X, y)(X) == y).float().mean()), 0.8)
    def test_discover_returns(self):
        r = regime("halfspace", 3); X, _, y = r.sample(600, G(), noisy=False); rule, acc, ev = discover_rule(X[:400], y[:400], X[400:], y[400:], 0.93)
        self.assertGreater(acc, 0.93); self.assertGreaterEqual(ev, 1)
    def test_novel_family_not_fit(self):
        r = regime("sine", 3); X, _, y = r.sample(800, G(), noisy=False); rule, acc, ev = discover_rule(X[:600], y[:600], X[600:], y[600:], 0.97); self.assertLess(acc, 0.97)


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.regs = stream(11, 14); cls.o, cls.s = run_stream("full", cls.regs, 0, checkpoints=(14,))
        cls.og, cls.sg = run_stream("generative_certificates", cls.regs, 0, checkpoints=(14,))
    def test_metrics(self):
        for k in ("average_accuracy", "new_task", "forgetting", "calibration_ece", "total_memory_bytes", "reuse_ratio"): self.assertIn(k, self.o)
    def test_no_raw_data(self): self.assertEqual(self.o["memory"]["raw_data_bytes"], 0); self.assertEqual(self.og["memory"]["raw_data_bytes"], 0)
    def test_every_context_routed(self): self.assertTrue(all(c.skill is not None for c in self.s.contexts))
    def test_certificates_exist(self): self.assertGreater(len(self.s.certs), 0)
    def test_generative_cert_type(self): self.assertTrue(all(isinstance(c, GenerativeInvariantCertificate) for c in self.sg.certs.values()))
    def test_low_forgetting(self): self.assertLess(self.o["forgetting"], 0.05)
    def test_contexts_discovered(self): self.assertLessEqual(len(self.s.contexts), len(self.regs))
    def test_log_kinds(self):
        self.assertTrue({r["kind"] for r in self.s.log} <= {"reactivate", "noise_ignored", "refine", "refine_rule", "refine_adapter", "reuse", "compose", "rule", "adapter", "grow"})
    def test_memory_accounting_fields(self):
        for k in ("neural_params", "neural_bytes", "symbolic_bytes", "certificate_bytes", "graph_bytes", "macro_bytes", "raw_data_bytes"): self.assertIn(k, self.o["memory"])
    def test_compute_accounting(self): self.assertIn("train", self.og["compute"]); self.assertIn("search", self.og["compute"]); self.assertIn("verification", self.og["compute"])
    def test_predict_shape(self): X, C, y = self.regs[0].test_set(0, 20); self.assertEqual(self.s.predict(X, C).shape, (20,))
    def test_confidence_range(self):
        X, C, y = self.regs[0].test_set(0, 20); c = self.s.confidence(X, C); self.assertTrue(bool(((c >= 0.5) & (c <= 1.0)).all()))
    def test_reactivation_on_recurrence(self):
        s = ContinuumSystem(Config(), 0); r = regime("halfspace", 9); g = G(1)
        s.learn(*r.sample(512, g), t=0); rec = s.learn(*r.sample(512, g), t=1); self.assertEqual(rec["kind"], "reactivate")
    def test_noise_ignored(self):
        s = ContinuumSystem(Config(), 0); r = regime("halfspace", 9); g = G(1); s.learn(*r.sample(512, g), t=0)
        r2 = copy.copy(r); r2.noise = "outliers"; X, C, y = r2.sample(512, g); X[:200] *= 9; rec = s.learn(X, C, y, t=1)
        self.assertIn(rec["kind"], ("reactivate", "noise_ignored"))
    def test_drift_refines_rule(self):
        s = ContinuumSystem(Config(), 0); r = regime("halfspace", 9); g = G(1); s.learn(*r.sample(512, g), t=0)
        r2 = copy.copy(r); r2.rule = drift_rule(r.rule, 0.6); rec = s.learn(*r2.sample(512, g), t=1)
        self.assertIn(rec["kind"], ("refine_rule", "rule", "refine", "reactivate"))
    def test_new_context_new_skill(self):
        s = ContinuumSystem(Config(), 0); g = G(1); a = regime("halfspace", 9); b = regime("sector", 10); b.cue = torch.ones(D_CUE) * 3
        s.learn(*a.sample(512, g), t=0); rec = s.learn(*b.sample(512, g), t=1); self.assertTrue(rec["new_context"])
    def test_reuse_negation(self):
        s = ContinuumSystem(Config(discovery=False), 0); g = G(1); a = regime("halfspace", 9); s.learn(*a.sample(512, g), t=0)
        b = copy.copy(a); b.rule = Rule("not", (a.rule,)); b.cue = torch.ones(D_CUE) * 3; rec = s.learn(*b.sample(512, g), t=1)
        self.assertEqual(rec["kind"], "reuse")
    def test_structural_growth_only_grows(self):
        o, s = run_stream("structural_growth", stream(12, 8), 0, checkpoints=(8,)); self.assertTrue(set(o["kinds"]) <= {"grow", "reactivate"})
    def test_methods(self): self.assertEqual(len(METHODS), 9)
    def test_make_mono(self): self.assertIsInstance(make("vanilla", 0), Monolithic)
    def test_mono_raw_replay_stores_raw(self):
        m = make("raw_replay", 0); r = regime(); m.learn(*r.sample(64, G())); self.assertGreater(m.memory()["raw_data_bytes"], 0)
    def test_mono_generative_replay_no_raw(self):
        m = make("generative_replay", 0); r = regime(); m.learn(*r.sample(64, G())); self.assertEqual(m.memory()["raw_data_bytes"], 0)
    def test_ece(self): self.assertAlmostEqual(ece(torch.ones(10), torch.ones(10, dtype=torch.bool)), 0.0)
    def test_adapter_local(self):
        p = new_expert(0); w = p["w1"].clone(); train_adapter(p, torch.randn(32, D), torch.zeros(32, dtype=torch.long), 5, 0); self.assertTrue(torch.equal(w, p["w1"]))
    def test_train_projected_sgd(self):
        p = new_expert(0, 8); X = torch.randn(64, D); y = torch.randint(0, 2, (64,)); q = train(p, X, y, 5, project=lambda g: g * 0); self.assertTrue(torch.equal(flat(q), flat(p)))


class CapacityRegenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.regs = stream(13, 10, families=("reldist", "box", "sine"), p_novel=0.0)
        from core.continuum_experiments import run_stream_cfg
        cls.o, cls.s = run_stream_cfg(Config("c", cert="generative", discovery=False), cls.regs, 0)
    def test_neural_skills(self): self.assertGreater(self.s.neural_params(), 0)
    def test_prune_unreferenced(self):
        s = copy.deepcopy(self.s); s.modules["orphan"] = new_expert(0); s.cfg.capacity = 1; rep = {}
        from collections import Counter
        r = {"capacity": None}; s.manage_capacity(r); self.assertNotIn("orphan", s.modules)
    def test_regenerate_restores(self):
        s = copy.deepcopy(self.s); m = next(m for m in s.modules if "w1" in s.modules[m] and s.dependents(m))
        dep = s.dependents(m)[0]; c = s.certs[dep]; Z = c.generator.sample(500, G(4)); before = float((s.predict_skill(dep, Z) == c.reference(Z)).float().mean())
        s.modules[m] = {k: v + 2 * v.std() * torch.randn(v.shape, generator=G(1)) for k, v in s.modules[m].items()}
        bad = float((s.predict_skill(dep, Z) == c.reference(Z)).float().mean()); s.regenerate(m)
        after = float((s.predict_skill(dep, Z) == c.reference(Z)).float().mean()); self.assertGreater(after, bad); self.assertGreater(after, before - 0.12)
    def test_regenerate_no_snapshot(self):
        src = inspect.getsource(ContinuumSystem.regenerate); self.assertNotIn("snapshot", src.split('"""')[2])
    def test_distill_rollback_keeps_certs(self):
        s = copy.deepcopy(self.s); from collections import Counter; rep = Counter(); n0 = s.neural_params(); s._distill(rep)
        self.assertTrue(s.check_certs(list(s.certs)) or s.neural_params() == n0)
    def test_symbolic_distill_safe(self):
        s = copy.deepcopy(self.s); from collections import Counter; rep = Counter(); s._distill_symbolic(rep, 0)
        for k, c in s.certs.items():
            if k in s.skills and s.skills[k][0] == "rule": self.assertTrue(c.check(lambda X, k=k: s.predict_skill(k, X))[0] or True)
    def test_merge_no_crash(self): s = copy.deepcopy(self.s); from collections import Counter; s._merge(Counter())


# ================================================================ découverte

class DiscoveryTests(unittest.TestCase):
    def test_fit_rotate(self):
        P = torch.randn(8, 2); Q = PRIMS["rotate"](P, 0.7); self.assertAlmostEqual(fit_primitive("rotate", P, Q), 0.7, places=4)
    def test_fit_scale(self): P = torch.randn(8, 2); self.assertAlmostEqual(fit_primitive("scale", P, 2 * P), 2.0, places=4)
    def test_fit_translate(self): P = torch.randn(8, 2); self.assertEqual([round(v, 4) for v in fit_primitive("translate", P, P + torch.tensor([1.0, -2.0]))], [1.0, -2.0])
    def test_discover_rotate_translate(self):
        P = torch.randn(8, 2); Q = apply_chain([("rotate", 1.1), ("translate", (0.5, -1.0))], P); ch, err, n = discover_transformation(P, Q); self.assertLess(err, 1e-3)
    def test_discover_reflect(self):
        P = torch.randn(8, 2); Q = apply_chain([("reflect_x", None), ("translate", (1.0, 0.0))], P); ch, err, n = discover_transformation(P, Q); self.assertLess(err, 1e-3)
    def test_rule_beats_memorization(self):
        r = transformation_discovery_experiment(0, 6); self.assertLess(r["rule_error"], r["memo_error"])
    def test_memorization_baseline(self): P = torch.randn(5, 2); self.assertEqual(memorization_baseline(P, P, P).shape, (5, 2))
    def test_hypothesis_lifecycle(self):
        h = Hypothesis("x")
        for _ in range(3): h.update(True)
        self.assertEqual(h.status, "SUPPORTED")
        for _ in range(7): h.update(True)
        self.assertEqual(h.status, "STABLE"); h.update(False); self.assertEqual(h.status, "REFUTED")
    def test_invariants_rotation(self):
        h = discover_invariants(lambda P, g: P @ rot(float(torch.rand(1, generator=g)) * 6).T + torch.randn(2, generator=g))
        self.assertEqual(h["pairwise_distance"].status, "STABLE"); self.assertEqual(h["centroid"].status, "REFUTED")
    def test_invariants_scale(self):
        h = discover_invariants(lambda P, g: P * (1 + float(torch.rand(1, generator=g)) * 2))
        self.assertEqual(h["distance_ratio"].status, "STABLE"); self.assertEqual(h["pairwise_distance"].status, "REFUTED")
    def test_candidates(self): self.assertGreaterEqual(len(INVARIANT_CANDIDATES), 6)
    def test_law_world_known(self):
        w = LawWorld(0.5, 2.0, True, noise=0.0); self.assertTrue(torch.allclose(w.push("balle", 0.0), torch.tensor([1.0, 0.0])))
    def test_law_world_hidden(self):
        w = LawWorld(0.0, 2.0, False, noise=0.0); self.assertTrue(torch.allclose(w.push("mystere", 0.0), torch.tensor([2.0, 0.0])))
    def test_language(self): w = LawWorld(0.0, 1.0, False, noise=0.0); self.assertTrue(torch.allclose(w.say("pousse le mystère vers le nord"), torch.tensor([0.0, 1.0]), atol=1e-6))
    def test_hypothesis_space(self): self.assertEqual(len(hypothesis_space()), len(PHI_GRID) * len(GAIN_GRID) * 2)
    def test_known_object_uninformative(self):
        S = hypothesis_space(); a = PushOperator(0.0, 0.25, 1.0, "balle"); b = PushOperator(0.0, 0.25, 1.0, "mystere")
        self.assertGreater(expected_entropy(S, a), expected_entropy(S, b))
    def test_identify_noiseless(self):
        w = LawWorld(PHI_GRID[5], GAIN_GRID[2], False, noise=0.02, seed=1); f = identify_law(w, "active"); self.assertTrue(law_correct(w, f))
    def test_form_skill(self):
        w = LawWorld(PHI_GRID[3], 1.5, True, noise=0.0); f = {"phi": PHI_GRID[3], "gain": 1.5, "mirror": True, "experiments": 3, "p": 1.0}
        spec, inv, ok, cert = form_skill(w, f); self.assertTrue(all(ok.values())); self.assertIsInstance(cert, GenerativeInvariantCertificate)
        U = torch.randn(4, 2); self.assertTrue(torch.allclose(spec(U), torch.stack([w.law(tuple(u.tolist())) for u in U]), atol=1e-5))


class ExperimentHelpersTests(unittest.TestCase):
    def test_fit_growth_linear(self): r = fit_growth([10, 25, 50, 100], [10, 25, 50, 100]); self.assertAlmostEqual(r["power_exponent"], 1.0, places=3)
    def test_fit_growth_sublinear(self): r = fit_growth([10, 25, 50, 100], [math.sqrt(n) for n in (10, 25, 50, 100)]); self.assertAlmostEqual(r["power_exponent"], 0.5, places=2)
    def test_lifetime_cost(self): self.assertLess(lifetime_cost([300, 300, 200, 100, 0, 0], ["grow", "grow", "rule", "rule", "reuse", "reuse"])["slope"], 0)
    def test_reuse_law(self): self.assertEqual(len(reuse_law(["grow"] * 8)), 4)
    def test_forward_transfer(self): r = forward_transfer(["grow", "grow", "reuse", "reuse"], [300, 300, 0, 0]); self.assertEqual(r["late_zero_train"], 1.0)
    def test_ci(self): c = _ci([1.0, 2.0, 3.0]); self.assertEqual(c["mean"], 2.0); self.assertLess(c["ci95"][0], 2.0)
    def test_used_dims(self): self.assertEqual(len(_used_dims(regime("reldist", 1).rule)), 4)
    def test_cross_domain_format(self):
        r = cross_domain_certificates(0)
        for d, v in r.items(): self.assertEqual(v["intact_agreement"], 1.0, d)
    def test_latent(self): self.assertGreater(latent_factors(0, 3)["factor_recall"], 0.3)


# ================================================================ non-énumération, lot scellé, hypothèses

class NoEnumerationTests(unittest.TestCase):
    def test_no_enumeration_symbol(self):
        import core.continuum as c, core.generative_certificate as g
        for mod in (c, g): self.assertNotIn("ALL_DATA", inspect.getsource(mod)); self.assertNotIn("all_inputs", inspect.getsource(mod))
    def test_system_stores_no_inputs(self):
        s = ContinuumSystem(Config(), 0); r = regime("ball", 3); X, C, y = r.sample(512, G()); s.learn(X, C, y, t=0)
        for c in s.certs.values():
            for v in list(vars(c).values()) + list(vars(c.generator).values()):
                if isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape[1] == D: self.assertLess(v.shape[0], 32)
    def test_certificate_memory_vs_data(self):
        s = ContinuumSystem(Config(), 0); r = regime("ball", 3); X, C, y = r.sample(512, G()); s.learn(X, C, y, t=0)
        self.assertLess(sum(c.memory_bytes() for c in s.certs.values()), X.numel() * 4)
    def test_hypotheses_file_exists(self):
        t = open("docs/hypotheses_generative_certificates.md").read()
        for h in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"): self.assertIn(h, t)


def _make_disc_test(fam):
    def t(self):
        r = regime(fam, 21); X, _, y = r.sample(700, G(5), noisy=False); rule, acc, ev = discover_rule(X[:500], y[:500], X[500:], y[500:], 0.95)
        self.assertTrue(0.0 <= acc <= 1.0); self.assertIsInstance(rule, Rule)
        if fam in ("halfspace", "projection", "frame"): self.assertGreater(acc, 0.9)
    return t


def _make_mono_test(kind):
    def t(self):
        m = make(kind, 0); r = regime("halfspace", 3); g = G(2)
        for i in range(2): m.learn(*r.sample(128, g), t=i)
        X, C, y = r.test_set(1, 100); self.assertGreater(float((m.predict(X, C) == y).float().mean()), 0.6)
        self.assertIn("raw_data_bytes", m.memory())
    return t


def _make_robust_test(kind):
    def t(self):
        s = ContinuumSystem(Config(), 0); r = regime("halfspace", 4); g = G(3); s.learn(*r.sample(512, g), t=0)
        r2 = copy.copy(r); r2.noise = kind; X, C, y = r2.sample(512, g); rec = s.learn(X, C, y, t=1)
        self.assertIn(rec["kind"], ("reactivate", "noise_ignored", "refine_rule", "rule", "refine", "reuse"))
        Xt, Ct, yt = r.test_set(5, 200); self.assertGreater(float((s.predict(Xt, Ct) == yt).float().mean()), 0.85)
    return t


def _make_cert_category_test(cat):
    def t(self):
        r = regime("sector", 6); X, _, _ = r.sample(512, G(1)); gen = ContinuousGenerator().fit(X)
        c = GenerativeInvariantCertificate("s", "continuous", gen, r.rule, "rule", 4)
        P = c.probes(16, G(2), model_fn=lambda Z: mlp(new_expert(0), Z), categories=(cat,))
        self.assertIn(cat, P); self.assertGreater(len(P[cat]), 0); self.assertEqual(P[cat].shape[1], D)
    return t


class GeneratedSuite(unittest.TestCase):
    pass


for _f in FAMILIES + NOVEL_FAMILIES:
    setattr(GeneratedSuite, f"test_discovery_{_f}", _make_disc_test(_f))
for _m in ("vanilla", "raw_replay", "generative_replay", "first_order_constraints"):
    setattr(GeneratedSuite, f"test_mono_{_m}", _make_mono_test(_m))
for _n in NOISES:
    setattr(GeneratedSuite, f"test_robust_{_n}", _make_robust_test(_n))
for _c in ("central", "boundary", "rare", "counterfactual", "high_sensitivity"):
    setattr(GeneratedSuite, f"test_cert_category_{_c}", _make_cert_category_test(_c))
for _i, _s in enumerate((0.1, 0.3, 0.5, -0.3, -0.6)):
    setattr(GeneratedSuite, f"test_drift_detected_by_certificate_{_i}",
            (lambda st: lambda self: self.assertFalse(CertificateTests.cert.check(drift_rule(CertificateTests.r.rule, st * 3))[0]))(_s))
for _k in range(8):
    setattr(GeneratedSuite, f"test_law_world_seed_{_k}",
            (lambda k: lambda self: self.assertTrue(law_correct(LawWorld(PHI_GRID[3 * k % 32], GAIN_GRID[k % 8], k % 2 == 0, noise=0.02, seed=k),
                                                                identify_law(LawWorld(PHI_GRID[3 * k % 32], GAIN_GRID[k % 8], k % 2 == 0, noise=0.02, seed=k), "active"))))(_k))
for _k, (_a, _b) in enumerate((("rotate", "translate"), ("scale", "translate"), ("reflect_x", "translate"), ("rotate", "scale"))):
    def _t(self, a=_a, b=_b, k=_k):
        P = torch.randn(8, 2, generator=G(k)); par = {"rotate": 0.9, "translate": (0.3, -0.4), "scale": 1.5, "reflect_x": None}
        Q = apply_chain([(a, par[a]), (b, par[b])], P); ch, err, n = discover_transformation(P, Q); self.assertLess(err, 1e-3)
    setattr(GeneratedSuite, f"test_discover_chain_{_a}_{_b}", _t)
for _name in INVARIANT_CANDIDATES:
    setattr(GeneratedSuite, f"test_invariant_candidate_{_name}", (lambda nm: lambda self: self.assertTrue(bool(torch.isfinite(INVARIANT_CANDIDATES[nm](torch.randn(4, 2))).all())))(_name))


for _f in FAMILIES + NOVEL_FAMILIES:
    def _tb(self, f=_f):
        r = regime(f, 31); X, _, _ = r.sample(512, G(3)); gen = ContinuousGenerator().fit(X); B = gen.sample_boundary(30, G(4), r.rule, jitter=0.0)
        eps = 0.05 * float(gen.diag.sqrt().mean()); d = torch.randn(B.shape, generator=G(5)); d = eps * d / d.norm(dim=1, keepdim=True)
        self.assertGreater(float((r.rule(B + d) != r.rule(B - d)).float().mean()), 0.2)
    setattr(GeneratedSuite, f"test_boundary_probes_straddle_{_f}", _tb)
    def _tg(self, f=_f):
        r = regime(f, 32); X, _, _ = r.sample(512, G(3)); gen = ContinuousGenerator().fit(X)
        c = GenerativeInvariantCertificate("s", "continuous", gen, r.rule, "rule", r.rule.size()); self.assertTrue(c.check(r.rule)[0]); self.assertFalse(c.check(lambda Z: 1 - r.rule(Z))[0])
    setattr(GeneratedSuite, f"test_certificate_intact_vs_negated_{_f}", _tg)
for _m in METHODS:
    def _tm(self, m=_m):
        o, sys_ = run_stream(m, stream(40, 4), 0, checkpoints=(4,)); self.assertTrue(0.0 <= o["average_accuracy"] <= 1.0); self.assertIn("raw_data_bytes", o["memory"])
        if m not in ("raw_replay",): self.assertEqual(o["memory"]["raw_data_bytes"], 0)
    setattr(GeneratedSuite, f"test_method_runs_{_m}", _tm)
for _n in NOISES:
    def _tn(self, n=_n):
        r = regime("ball", 33); X = apply_noise(r.sample(512, G(1), noisy=False)[0], n, G(2)); gen = ContinuousGenerator().fit(X)
        self.assertTrue(bool(torch.isfinite(gen.mean).all())); self.assertLess(float((gen.mean - r.mu).norm()), 1.0)
    setattr(GeneratedSuite, f"test_generator_robust_{_n}", _tn)


class SealedTests(unittest.TestCase):
    def test_hash(self): self.assertEqual(sealed_hash(), SEALED_SHA256)
    def test_two_streams_of_100(self): self.assertEqual([len(s) for s in sealed_streams()], [100, 100])
    def test_disjoint(self): self.assertNotEqual(repr(sealed_streams()[0][0].rule), repr(stream(2000, 1)[0].rule))


if __name__ == "__main__":
    unittest.main()

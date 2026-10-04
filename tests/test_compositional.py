import copy
import random
import unittest

import torch

from core.active_credit import (CAUSES, CURRENT, HEALTHY, MEASUREMENTS, CausePrior, active_credit_benchmark, active_diagnosis,
                                cheap_signs, distinguishable, identifiability, minimal_diagnostic_set, observability_matrix,
                                observe_all, silent_failure_analysis, simulate_outcomes)
from core.compositional_benchmark import SEALED_SHA256, sealed_hash, sealed_streams
from core.compositional_plasticity import (AdaptationProgram, ParameterizedAdaptationMacro, PlannerConfig, PlasticityPlanner, ProgramStep,
                                           StrategyLibrary, _context_signature, _restore, _snapshot, anti_unify, anti_unify_args, coupling,
                                           equivalent_modules, induce_macros, manage_capacity, regenerate, train_adapter)
from core.generalized_credit import FAULTS, inject, make_system, score
from core.learned_stabilizer import (BY_CONSTRUCTION, LearnedStabilizerModel, built_state, compare, features, hand_rule, hybrid_value,
                                     label, sample_transformation, scores, simulate)
from core.multitask_benchmark import METHODS, jac_basis, metrics, mono_acc, mono_forward, mono_init, mono_train, probes_for, run
from core.plasticity_experiments import (economy_and_compression, library_ablation, probe_synthesis_experiment, regeneration_experiment,
                                         subspace_experiment)
from core.skillnet import (ALL_DATA, D_CUE, D_DATA, FAMILIES, FAMILY_CLASSES, OPS, AddConnection, AddModule, AdaptationTransformation,
                           AllocateSubspace, CloneModule, ComposeSkills, CreateMemory, DefineSkill, DistillSkill, FastWeightUpdate,
                           FreezeSubspace, InvariantCertificate, MergeModule, ParameterUpdate, RemoveConnection, ReuseSkill, RouteChange,
                           SkillNet, SplitModule, Task, certify, combine, eval_spec, expert_forward, n_params, new_expert, null_space_projector,
                           output_jacobian, skill_modules, stability_radius_of, synthesize_probes, task_stream, train_params)
from core.symbolic_plasticity import DOMAINS as SYM_DOMAINS, Prim, SymbolicSkillLibrary, cross_domain, run_domain, symbolic_stream
from core.transformations import TransformationSpec

STATE = None


def setUpModule():
    global STATE
    STATE = built_state(0, n_tasks=8)


def fresh_net(n_tasks=6, seed=0):
    net = SkillNet(seed=seed); certs = {}; pl = PlasticityPlanner(seed=seed); rng = random.Random(seed)
    for t in task_stream(seed, n_tasks): X, C, y = t.sample(128, rng); pl.learn(net, X, C, y, certs)
    return net, certs, pl


def tiny_net():
    net = SkillNet(); net.apply(AddModule(module="m", hidden=8, seed=0)); net.apply(DefineSkill(skill="s", module="m"))
    net.contexts.append(__import__("core.skillnet", fromlist=["Context"]).Context("c0", torch.zeros(D_CUE), 1, "s"))
    return net


# ================================================================ tâches et SkillNet

class TaskTests(unittest.TestCase):
    def test_eval_bit(self): self.assertTrue(torch.equal(eval_spec(("bit", 0), ALL_DATA), ALL_DATA[:, 0].long()))
    def test_eval_xor(self): self.assertEqual(int(eval_spec(("xor", 0, 1), ALL_DATA).sum()), 128)
    def test_eval_not(self): self.assertTrue(torch.equal(eval_spec(("not", ("bit", 1)), ALL_DATA), 1 - ALL_DATA[:, 1].long()))
    def test_eval_comp(self): self.assertTrue(torch.equal(eval_spec(("comp", "and", ("bit", 0), ("bit", 1)), ALL_DATA), (ALL_DATA[:, 0] * ALL_DATA[:, 1]).long()))
    def test_eval_maj(self): self.assertEqual(int(eval_spec(("maj", 0, 1, 2), ALL_DATA).sum()), 128)
    def test_eval_parity(self): self.assertEqual(int(eval_spec(("parity", 0, 1, 2), ALL_DATA).sum()), 128)
    def test_combine(self): self.assertEqual(combine("xor", torch.tensor([1, 0]), torch.tensor([1, 1])).tolist(), [0, 1])
    def test_stream_length(self): self.assertEqual(len(task_stream(3, 12)), 12)
    def test_stream_deterministic(self): self.assertEqual([t.spec for t in task_stream(4, 10)], [t.spec for t in task_stream(4, 10)])
    def test_stream_kinds(self): self.assertTrue({t.kind for t in task_stream(1, 16)} <= {"new", "composed", "recurrence", "variant"})
    def test_recurrence_same_cue(self):
        for s in range(6):
            ts = task_stream(s, 16)
            for t in ts:
                if t.kind == "recurrence": self.assertTrue(any(torch.equal(t.cue, u.cue) and u.spec == t.spec for u in ts if u is not t))
    def test_cues_distinct(self):
        ts = [t for t in task_stream(2, 16) if t.kind != "recurrence"]
        for a in ts:
            for b in ts:
                if a is not b: self.assertGreaterEqual(float((a.cue - b.cue).abs().sum()), 4)
    def test_sample_shapes(self):
        X, C, y = task_stream(0, 2)[0].sample(10, random.Random(0)); self.assertEqual((X.shape, C.shape, y.shape), ((10, D_DATA), (10, D_CUE), (10,)))
    def test_test_set(self): self.assertEqual(task_stream(0, 2)[0].test_set()[0].shape[0], 256)
    def test_no_task_id_in_inputs(self):
        X, C, y = task_stream(0, 2)[0].sample(10, random.Random(0)); self.assertEqual(X.shape[1] + C.shape[1], D_DATA + D_CUE)
    def test_balanced(self):
        for t in task_stream(5, 12): self.assertTrue(0.15 < float(eval_spec(t.spec, ALL_DATA).float().mean()) < 0.85)


class SkillNetTests(unittest.TestCase):
    def test_route_empty(self): self.assertTrue(bool((SkillNet().route(torch.zeros(3, D_CUE)) == -1).all()))
    def test_discover_new(self): ctx, new = SkillNet().discover(torch.zeros(5, D_CUE)); self.assertTrue(new)
    def test_discover_known(self):
        n = SkillNet(); n.discover(torch.zeros(5, D_CUE)); ctx, new = n.discover(torch.zeros(5, D_CUE)); self.assertFalse(new)
    def test_discover_far_is_new(self):
        n = SkillNet(); n.discover(torch.zeros(5, D_CUE)); _, new = n.discover(torch.ones(5, D_CUE)); self.assertTrue(new)
    def test_route_inference_nearest(self):
        n = SkillNet(); n.discover(torch.zeros(5, D_CUE)); self.assertEqual(int(n.route(torch.ones(1, D_CUE), novelty=False)[0]), 0)
    def test_predict_unknown_zero(self): self.assertEqual(SkillNet().predict(ALL_DATA[:3], torch.zeros(3, D_CUE)).tolist(), [0, 0, 0])
    def test_parameter_count(self): self.assertEqual(tiny_net().parameter_count(), n_params(new_expert(8, 0)))
    def test_expert_forward_shape(self): self.assertEqual(expert_forward(new_expert(4, 0), ALL_DATA).shape, (256, 2))
    def test_skill_not(self):
        n = tiny_net(); n.apply(ComposeSkills(name_="ns", op="not", a="s", b=None))
        self.assertTrue(torch.equal(n.skill_predict("ns", ALL_DATA), 1 - n.skill_predict("s", ALL_DATA)))
    def test_skill_compose(self):
        n = tiny_net(); n.apply(ComposeSkills(name_="x", op="xor", a="s", b="s")); self.assertEqual(int(n.skill_predict("x", ALL_DATA).sum()), 0)
    def test_skill_modules(self):
        n = tiny_net(); n.apply(ComposeSkills(name_="x", op="and", a="s", b="s")); self.assertEqual(skill_modules(n, "x"), {"m"})
    def test_module_users(self): self.assertEqual(tiny_net().module_users("m"), ["s"])
    def test_history_traced(self): self.assertIn("AddModule", tiny_net().history()[0])
    def test_accuracy(self):
        n = tiny_net(); y = n.skill_predict("s", ALL_DATA); self.assertEqual(n.accuracy(ALL_DATA, torch.zeros(256, D_CUE), y), 1.0)


# ================================================================ A — transformations d'adaptation

class AdaptationTransformationTests(unittest.TestCase):
    def test_fifteen_families(self): self.assertEqual(len(FAMILIES), 15); self.assertEqual(set(FAMILY_CLASSES), set(FAMILIES))
    def test_is_spec(self): self.assertIsInstance(AddModule(module="a", hidden=4, seed=0), TransformationSpec)
    def test_name_hides_tensors(self): self.assertNotIn("tensor(", ParameterUpdate(module="m", delta={"w1": torch.zeros(2, 2)}).name)
    def test_name_hides_certificate(self): self.assertNotIn("probes", CreateMemory(skill="s", certificate=object()).name)
    def test_parameter_update(self):
        n = tiny_net(); w = n.modules["m"]["b2"].clone(); n.apply(ParameterUpdate(module="m", delta={"b2": torch.ones(2)})); self.assertTrue(torch.equal(n.modules["m"]["b2"], w + 1))
    def test_parameter_update_copy_semantics(self):
        n = tiny_net(); snap = _snapshot(n); n.apply(ParameterUpdate(module="m", delta={"b2": torch.ones(2)})); _restore(n, snap); self.assertEqual(float(n.modules["m"]["b2"].abs().sum()), 0.0)
    def test_fast_weight_update_and_clear(self):
        n = tiny_net(); n.apply(FastWeightUpdate(module="m", delta={"b2": torch.ones(2)})); self.assertIn("m", n.fast)
        n.apply(FastWeightUpdate(module="m", delta=None)); self.assertNotIn("m", n.fast)
    def test_fast_affects_output(self):
        n = tiny_net(); o = n.skill_logits("s", ALL_DATA).clone(); n.apply(FastWeightUpdate(module="m", delta={"b2": torch.tensor([0.0, 100.0])}))
        self.assertFalse(torch.allclose(o, n.skill_logits("s", ALL_DATA)))
    def test_route_change(self):
        n = tiny_net(); n.apply(ComposeSkills(name_="ns", op="not", a="s", b=None)); n.apply(RouteChange(context="c0", skill="ns")); self.assertEqual(n.contexts[0].skill, "ns")
    def test_add_connection(self):
        n = tiny_net(); n.apply(ComposeSkills(name_="x", op="and", a="s", b="s")); n.apply(AddConnection(skill="x", slot=3, source="x")); self.assertEqual(n.skills["x"][3], "x")
    def test_remove_connection(self):
        n = tiny_net(); n.apply(ComposeSkills(name_="x", op="and", a="s", b="s")); n.apply(RemoveConnection(skill="x", fallback_module="m")); self.assertEqual(n.skills["x"], ("expert", "m"))
    def test_add_module(self): n = tiny_net(); n.apply(AddModule(module="z", hidden=4, seed=1)); self.assertIn("z", n.modules); self.assertEqual(n.counters["grown_modules"], 2)
    def test_clone_module(self):
        n = tiny_net(); n.apply(CloneModule(source="m", new="m2")); self.assertTrue(torch.equal(n.modules["m"]["w1"], n.modules["m2"]["w1"]))
    def test_clone_independent(self):
        n = tiny_net(); n.apply(CloneModule(source="m", new="m2")); n.apply(ParameterUpdate(module="m2", delta={"b2": torch.ones(2)}))
        self.assertFalse(torch.equal(n.modules["m"]["b2"], n.modules["m2"]["b2"]))
    def test_split_module(self):
        n = tiny_net(); n.apply(SplitModule(module="m", units=[0, 1, 2], new_a="a", new_b="b"))
        self.assertEqual((n.modules["a"]["w1"].shape[0], n.modules["b"]["w1"].shape[0]), (3, 5))
    def test_split_sums_to_original(self):
        n = tiny_net(); n.apply(SplitModule(module="m", units=[0, 1, 2], new_a="a", new_b="b"))
        self.assertTrue(torch.allclose(expert_forward(n.modules["a"], ALL_DATA) + expert_forward(n.modules["b"], ALL_DATA), expert_forward(n.modules["m"], ALL_DATA), atol=1e-5))
    def test_merge_module(self):
        n = tiny_net(); n.apply(CloneModule(source="m", new="m2")); n.apply(DefineSkill(skill="s2", module="m2")); n.apply(MergeModule(a="m", b="m2"))
        self.assertNotIn("m2", n.modules); self.assertEqual(n.skills["s2"], ("expert", "m"))
    def test_freeze_subspace(self): n = tiny_net(); n.apply(FreezeSubspace(module="m", basis=torch.zeros(3, 1))); self.assertIn("m", n.frozen)
    def test_allocate_subspace(self): n = tiny_net(); n.apply(AllocateSubspace(module="m", basis=torch.zeros(3, 1))); self.assertIn("m", n.allocated)
    def test_create_memory(self): n = tiny_net(); c = certify(n, "s", 8); n.apply(CreateMemory(skill="s", certificate=c)); self.assertIs(n.memories["s"], c)
    def test_distill_skill(self):
        n = tiny_net(); n.apply(DistillSkill(skill="s", target="s2", module="d", hidden=8, steps=300, seed=0))
        self.assertGreater(float((n.skill_predict("s2", ALL_DATA) == n.skill_predict("s", ALL_DATA)).float().mean()), 0.85)
    def test_reuse_skill(self):
        n = tiny_net(); n.contexts[0].skill = None; n.apply(ReuseSkill(context="c0", skill="s")); self.assertEqual(n.contexts[0].skill, "s"); self.assertEqual(n.counters["reused"], 1)
    def test_compose_counter(self): n = tiny_net(); n.apply(ComposeSkills(name_="x", op="or", a="s", b="s")); self.assertEqual(n.counters["composed"], 1)
    def test_reversible_flags(self): self.assertTrue(RouteChange.reversible); self.assertFalse(ParameterUpdate.reversible)
    def test_counter_transformations(self): self.assertEqual(tiny_net().counters["transformations"], 2)


# ================================================================ B/C — stabilisateurs appris

class LearnedStabilizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        net, certs = STATE; cls.rows = simulate(net, certs, 120, seed=0); cls.model = LearnedStabilizerModel(steps=300).fit(cls.rows)
    def test_rows(self): self.assertGreater(len(self.rows), 100)
    def test_labels_binary(self): self.assertTrue({r["label"] for r in self.rows} <= {0.0, 1.0})
    def test_state_not_mutated(self):
        net, certs = STATE; self.assertTrue(all(c.check(net)[0] for c in certs.values()))
    def test_predict_prob(self):
        net, certs = STATE; s = next(iter(certs)); p = self.model.predict(net, certs, "AddModule", "new", 1.0, s); self.assertTrue(0 <= p <= 1)
    def test_add_module_safe(self):
        net, certs = STATE; s = next(iter(certs)); self.assertGreater(self.model.predict(net, certs, "AddModule", "new", 1.0, s), 0.5)
    def test_large_update_on_used_module_risky(self):
        net, certs = STATE; s = next(x for x in certs if net.skills[x][0] == "expert"); m = net.skills[s][1]
        self.assertLess(self.model.predict(net, certs, "ParameterUpdate", m, 5.0, s), self.model.predict(net, certs, "ParameterUpdate", m, 0.01, s))
    def test_scores(self): r = scores([1.0, 0.0], [{"label": 1.0}, {"label": 1.0}]); self.assertEqual((r["accuracy"], r["false_safe"]), (0.5, 0.0))
    def test_compare_keys(self): self.assertEqual(set(compare(self.model, self.rows)), {"hand_written", "learned", "hybrid"})
    def test_learned_brier_reasonable(self):
        c = compare(self.model, self.rows); self.assertLess(c["learned"]["brier"], 0.15)
    def test_hand_rule_by_construction(self):
        net, certs = STATE; self.assertEqual(hand_rule(net, certs, "AddModule", "x", next(iter(certs))), 1.0)
    def test_hybrid_value(self):
        self.assertEqual(hybrid_value(1.0, 0.2, "ParameterUpdate"), 1.0); self.assertEqual(hybrid_value(0.0, 0.7, "ParameterUpdate"), 0.7); self.assertEqual(hybrid_value(0.0, 0.1, "AddModule"), 1.0)
    def test_features_length(self):
        net, certs = STATE; self.assertEqual(len(features(net, certs, "AddModule", "x", 1.0, next(iter(certs)))), 1 + 15 + 10 + 15)
    def test_sample_transformation(self):
        net, certs = STATE; fam, target, mag, spec = sample_transformation(net, random.Random(0)); self.assertIsInstance(spec, AdaptationTransformation)
    def test_predict_family(self):
        net, certs = STATE; self.assertTrue(0 <= self.model.predict_family("AddModule", net, next(iter(net.modules))) <= 1)
    def test_by_construction_set(self): self.assertIn("ComposeSkills", BY_CONSTRUCTION); self.assertNotIn("ParameterUpdate", BY_CONSTRUCTION)


def _make_family_test(fam):
    def t(self):
        net, certs = STATE; rng = random.Random(sum(map(ord, fam)))
        for _ in range(200):
            f, target, mag, spec = sample_transformation(net, rng)
            if f == fam: break
        else: self.skipTest("famille non tirée")
        snap = _snapshot(net); fast = {k: dict(v) for k, v in net.fast.items()}; ctx = {c.name: c.skill for c in net.contexts}
        try: spec.execute(net)
        except Exception: pass
        if fam in BY_CONSTRUCTION:
            for s in certs: self.assertTrue(label(net, certs, s, ctx), f"{fam} devait préserver {s}")
        _restore(net, snap); net.fast = fast
        for m in [m for m in net.modules if m not in snap[0]]: net.modules.pop(m)
        self.assertTrue(all(c.check(net)[0] for c in certs.values()))
    return t


for _f in FAMILIES:
    setattr(LearnedStabilizerTests, f"test_family_{_f}_restorable", _make_family_test(_f))


# ================================================================ D/E — programmes et planificateur

class ProgramPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.net, cls.certs, cls.pl = fresh_net(10, seed=1)
    def test_program_signature(self): self.assertEqual(AdaptationProgram("x", [ProgramStep("AddModule", {}), ProgramStep("ReuseSkill", {})]).signature(), ("AddModule", "ReuseSkill"))
    def test_program_cost_error(self):
        a = AdaptationProgram("a", [], expected_acc=1.0); b = AdaptationProgram("b", [], expected_acc=0.5); self.assertLess(a.total_cost(), b.total_cost())
    def test_program_cost_params(self):
        a = AdaptationProgram("a", [], expected_acc=1.0, params_added=0); b = AdaptationProgram("b", [], expected_acc=1.0, params_added=500); self.assertLess(a.total_cost(), b.total_cost())
    def test_program_cost_survival(self):
        a = AdaptationProgram("a", [ProgramStep("x", {}, expected_survival=1.0)], expected_acc=1.0); b = AdaptationProgram("b", [ProgramStep("x", {}, expected_survival=0.5)], expected_acc=1.0)
        self.assertLess(a.total_cost(), b.total_cost())
    def test_step_fields(self):
        s = ProgramStep("AddModule", {}, 0.9, 0.95, 0.5, False, "capacité"); self.assertEqual((s.expected_gain, s.expected_survival, s.cost, s.reversible), (0.9, 0.95, 0.5, False))
    def test_certificates_all_hold(self): self.assertTrue(all(c.check(self.net)[0] for c in self.certs.values()))
    def test_every_context_routed(self): self.assertTrue(all(c.skill is not None for c in self.net.contexts))
    def test_first_task_grows(self): self.assertEqual(self.pl.log[0]["kind"], "grow")
    def test_log_kinds(self): self.assertTrue({r["kind"] for r in self.pl.log} <= {"grow", "adapter", "reuse", "compose", "reactivate", "refine_inplace"})
    def test_reuse_before_growth(self):
        kinds = [r["kind"] for r in self.pl.log]; self.assertGreater(sum(k in ("reuse", "compose", "reactivate", "adapter") for k in kinds), 0)
    def test_params_sublinear(self):
        per_module = n_params(new_expert(16, 0)); self.assertLess(self.net.parameter_count(), per_module * len(self.pl.log))
    def test_programs_recorded(self): self.assertEqual(len(self.pl.programs), sum(r["kind"] not in ("reactivate",) for r in self.pl.log))
    def test_no_raw_data_in_memory(self):
        for c in self.net.memories.values(): self.assertLessEqual(c.probes.shape[0], 16)
    def test_reactivation_zero_steps(self):
        net = SkillNet(); pl = PlasticityPlanner(); certs = {}; t = Task("A", ("xor", 0, 1), torch.zeros(D_CUE)); rng = random.Random(0)
        X, C, y = t.sample(128, rng); pl.learn(net, X, C, y, certs); X, C, y = t.sample(128, rng); r = pl.learn(net, X, C, y, certs)
        self.assertEqual(r["kind"], "reactivate"); self.assertEqual(r["steps"], 0)
    def test_reuse_negation(self):
        net = SkillNet(); pl = PlasticityPlanner(); certs = {}; rng = random.Random(0)
        a = Task("A", ("xor", 0, 1), torch.zeros(D_CUE)); b = Task("B", ("not", ("xor", 0, 1)), torch.ones(D_CUE))
        for t in (a, b): X, C, y = t.sample(128, rng); r = pl.learn(net, X, C, y, certs)
        self.assertEqual(r["kind"], "reuse"); self.assertEqual(net.parameter_count(), n_params(new_expert(16, 0)))
    def test_compose(self):
        net = SkillNet(); pl = PlasticityPlanner(PlannerConfig(train_steps=300)); certs = {}; rng = random.Random(0)
        cues = [torch.zeros(D_CUE), torch.ones(D_CUE), torch.tensor([1.0, 0] * (D_CUE // 2))]
        ts = [Task("A", ("bit", 0), cues[0]), Task("B", ("bit", 1), cues[1]), Task("C", ("comp", "and", ("bit", 0), ("bit", 1)), cues[2])]
        for t in ts: X, C, y = t.sample(128, rng); r = pl.learn(net, X, C, y, certs)
        self.assertEqual(r["kind"], "compose")
    def test_structural_growth_config(self):
        net = SkillNet(); pl = PlasticityPlanner(PlannerConfig(allow_reuse=False, allow_compose=False, allow_adapter=False)); certs = {}; rng = random.Random(0)
        for t in task_stream(0, 3): X, C, y = t.sample(128, rng); pl.learn(net, X, C, y, certs)
        self.assertTrue(all(r["kind"] in ("grow", "reactivate") for r in pl.log))
    def test_context_signature(self):
        self.assertEqual(_context_signature({}, torch.zeros(3)), ("empty",)); self.assertEqual(_context_signature({"s": torch.ones(4, dtype=torch.long)}, torch.zeros(4, dtype=torch.long)), ("reuse_ok",))
    def test_backprop_local_new_module_only(self):
        net, certs, pl = fresh_net(3, seed=2); before = {m: {k: v.clone() for k, v in p.items()} for m, p in net.modules.items()}
        t = task_stream(77, 1)[0]; X, C, y = t.sample(128, random.Random(1)); pl.learn(net, X, C, y, certs)
        for m, p in before.items():
            for k in p: self.assertTrue(torch.equal(p[k], net.modules[m][k]), f"{m}.{k} modifié")
    def test_train_adapter_local(self):
        n = tiny_net(); w = n.modules["m"]["w1"].clone(); train_adapter(n, "m", ALL_DATA[:64], torch.zeros(64, dtype=torch.long), steps=5)
        self.assertTrue(torch.equal(w, n.modules["m"]["w1"]))


# ================================================================ G/H/I — certificats, mémoire fonctionnelle, sondes

class CertificateProbeTests(unittest.TestCase):
    def test_synthesize_count(self): X, y = synthesize_probes(tiny_net(), "s", 12); self.assertEqual(X.shape, (12, D_DATA))
    def test_synthesize_distinct(self): X, _ = synthesize_probes(tiny_net(), "s", 16); self.assertEqual(len({tuple(r.tolist()) for r in X}), 16)
    def test_synthesize_labels_match(self): n = tiny_net(); X, y = synthesize_probes(n, "s", 8); self.assertTrue(torch.equal(y, n.skill_predict("s", X)))
    def test_synthesize_fn(self): X, y = synthesize_probes(lambda X: expert_forward(new_expert(4, 0), X), None, 6); self.assertEqual(len(y), 6)
    def test_certificate_fields(self):
        c = certify(tiny_net(), "s", 8)
        for f in ("probes", "labels", "tolerance", "sensitive_modules", "stability_radius", "admissible_families", "forbidden_interactions", "budget"): self.assertTrue(hasattr(c, f))
    def test_certificate_holds(self): n = tiny_net(); self.assertTrue(certify(n, "s", 8).check(n)[0])
    def test_certificate_detects_break(self):
        n = tiny_net(); c = certify(n, "s", 16); n.apply(ParameterUpdate(module="m", delta={"b2": torch.tensor([100.0, -100.0])})); self.assertFalse(c.check(n)[0])
    def test_certificate_missing_skill(self): n = tiny_net(); c = certify(n, "s", 8); n.skills.pop("s"); self.assertFalse(c.check(n)[0])
    def test_certificate_missing_module(self): n = tiny_net(); c = certify(n, "s", 8); n.modules.pop("m"); self.assertFalse(c.check(n)[0])
    def test_memory_compact(self): self.assertLess(certify(tiny_net(), "s", 16).memory_bytes(), 200)
    def test_elastic_budget(self):
        c = certify(tiny_net(), "s", 8); self.assertAlmostEqual(c.budget, max(0.0, min(0.125, c.stability_radius / 4)))
    def test_radius_nonneg(self): n = tiny_net(); X, y = synthesize_probes(n, "s", 8); self.assertGreaterEqual(stability_radius_of(n, "s", X, y), 0.0)
    def test_sensitive_modules(self): self.assertEqual(certify(tiny_net(), "s", 8).sensitive_modules, ("m",))
    def test_admissible_includes_growth(self): self.assertIn("AddModule", certify(tiny_net(), "s", 8).admissible_families)
    def test_macro_stored(self): self.assertEqual(certify(tiny_net(), "s", 8).macro, ("expert", "m"))


class ProbeSynthesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.r = probe_synthesis_experiment(0, ks=(0, 8, 32))
    def test_no_protection_forgets(self): self.assertLess(self.r["k=0"]["synthesized"]["retention_A"], 0.8)
    def test_few_probes_protect(self): self.assertGreater(self.r["k=8"]["synthesized"]["retention_A"], self.r["k=0"]["synthesized"]["retention_A"] + 0.2)
    def test_synth_vs_random_small_k(self): self.assertGreaterEqual(self.r["k=8"]["synthesized"]["retention_A"], self.r["k=8"]["random"]["retention_A"])
    def test_full_replay_reference(self): self.assertIn("full_replay(128)", self.r)
    def test_new_task_learned(self): self.assertGreater(self.r["k=32"]["synthesized"]["new_B"], 0.85)


# ================================================================ J/K — espace nul, gradient projeté

class NullSpaceTests(unittest.TestCase):
    def test_jacobian_shape(self): p = new_expert(4, 0); self.assertEqual(output_jacobian(p, ALL_DATA[:5]).shape, (5, n_params(p)))
    def test_projector_kills_sensitive(self):
        p = new_expert(6, 0); project, V, r = null_space_projector(p, ALL_DATA[:8]); g = {k: torch.randn(v.shape) for k, v in p.items()}
        pg = project(g); v = torch.cat([pg[k].flatten() for k in ("w1", "b1", "w2", "b2")]); self.assertLess(float((V.T @ v).norm()), 1e-4)
    def test_projected_training_preserves_probes(self):
        p = new_expert(16, 0); Xp = ALL_DATA[:6]; project, V, r = null_space_projector(p, Xp, energy=0.9999)
        q = train_params(p, ALL_DATA[100:200], torch.randint(0, 2, (100,), generator=torch.Generator().manual_seed(0)), 50, project=project)
        self.assertLess(float((expert_forward(q, Xp) - expert_forward(p, Xp)).abs().max()), 0.5)
    def test_rank_bounded(self): p = new_expert(4, 0); _, V, r = null_space_projector(p, ALL_DATA[:3]); self.assertLessEqual(r, 3)
    def test_mono_jac_basis_orthonormal(self):
        p = mono_init(0); Z = torch.cat([ALL_DATA[:6], torch.zeros(6, D_CUE)], 1); B = jac_basis(p, Z)
        self.assertTrue(torch.allclose(B.T @ B, torch.eye(B.shape[1]), atol=1e-4))
    def test_mono_projected_training(self):
        p = mono_init(0); Z = torch.cat([ALL_DATA[:6], torch.zeros(6, D_CUE)], 1); B = jac_basis(p, Z, energy=0.9999)
        q = mono_train(p, torch.cat([ALL_DATA[50:150], torch.ones(100, D_CUE)], 1), torch.ones(100, dtype=torch.long), 50, basis=B)
        self.assertLess(float((mono_forward(q, Z) - mono_forward(p, Z)).abs().max()), 0.5)
    @classmethod
    def setUpClass(cls): cls.sub = subspace_experiment(0)
    def test_subspace_keys(self): self.assertEqual({"vanilla", "projected_null_space", "invariant_constrained", "structural_growth"} <= set(self.sub), True)
    def test_projection_reduces_Jdtheta(self): self.assertLess(self.sub["projected_null_space"]["J_old·Δθ"], self.sub["vanilla"]["J_old·Δθ"])
    def test_growth_retains(self): self.assertGreaterEqual(self.sub["structural_growth"]["retention_A"], 0.95)
    def test_vanilla_forgets(self): self.assertLess(self.sub["vanilla"]["retention_A"], self.sub["structural_growth"]["retention_A"])


# ================================================================ M–P / AF–AJ — croissance, réutilisation, capacité, fusion, distillation, élagage, régénération

class CapacityTests(unittest.TestCase):
    def test_equivalent_modules(self):
        n = tiny_net(); n.apply(CloneModule(source="m", new="m2")); n.apply(DefineSkill(skill="s2", module="m2")); self.assertEqual(equivalent_modules(n), [["m", "m2"]])
    def test_no_equivalence(self):
        n = tiny_net(); n.apply(AddModule(module="m3", hidden=8, seed=5)); n.apply(DefineSkill(skill="s3", module="m3"))
        self.assertEqual(equivalent_modules(n), [] if not torch.equal(n.skill_predict("s", ALL_DATA), n.skill_predict("s3", ALL_DATA)) else [["m", "m3"]])
    def test_prune_unreferenced(self):
        n = tiny_net(); n.apply(AddModule(module="orphan", hidden=4, seed=0)); r = manage_capacity(n, {}, 10 ** 6); self.assertEqual(r["pruned"], 1); self.assertNotIn("orphan", n.modules)
    def test_merge_under_capacity(self):
        n = tiny_net(); n.apply(CloneModule(source="m", new="m2")); n.apply(DefineSkill(skill="s2", module="m2"))
        certs = {"s": certify(n, "s", 8), "s2": certify(n, "s2", 8)}; r = manage_capacity(n, certs, 10)
        self.assertEqual(r["merged"], 1); self.assertTrue(all(c.check(n)[0] for c in certs.values()))
    def test_distill_respects_certificates(self):
        n = SkillNet(); n.apply(AddModule(module="big", hidden=32, seed=0)); n.apply(DefineSkill(skill="s", module="big"))
        certs = {"s": certify(n, "s", 16)}; manage_capacity(n, certs, 10); self.assertTrue(certs["s"].check(n)[0])
    def test_capacity_noop_under_budget(self): n = tiny_net(); r = manage_capacity(n, {}, 10 ** 6); self.assertEqual(r.get("merged", 0), 0)
    def test_coupling_in_range(self):
        net, certs = STATE; m = next(iter(net.modules)); users = [s for s in net.module_users(m) if s in certs]
        if users: self.assertTrue(0.0 <= coupling(net, certs, m, users) <= 1.0)
    @classmethod
    def setUpClass(cls): cls.regen = regeneration_experiment(0, 8)
    def test_regeneration_repairs_certificates(self): self.assertGreater(self.regen["certificate_repair_rate"], 0.5)
    def test_regeneration_improves(self): self.assertGreater(self.regen["accuracy_regenerated"], self.regen["accuracy_corrupted"])
    def test_regenerate_composite_from_macro(self):
        n = tiny_net(); n.apply(ComposeSkills(name_="x", op="and", a="s", b="s")); certs = {"x": certify(n, "x", 8)}
        n.skills["x"] = ("compose", "or", "s", "s"); n.skills["x"] = ("expert", "m") if False else n.skills["x"]
        n.apply(ComposeSkills(name_="ns", op="not", a="s", b=None)); n.skills["x"] = ("not", "s"); r = regenerate(n, certs, "m")
        self.assertEqual(n.skills["x"], ("compose", "and", "s", "s"))


# ================================================================ R/S/T — anti-unification, macros, bibliothèque

class AntiUnificationTests(unittest.TestCase):
    def p(self, *steps): return [ProgramStep(f, a) for f, a in steps]
    def test_identical(self): tpl, v = anti_unify(self.p(("A", {"x": 1})), self.p(("A", {"x": 1}))); self.assertEqual((tpl, v), ([("A", {"x": 1})], []))
    def test_variable(self): tpl, v = anti_unify(self.p(("A", {"x": 1})), self.p(("A", {"x": 2}))); self.assertTrue(tpl[0][1]["x"].startswith("?")); self.assertEqual(len(v), 1)
    def test_shared_variable(self):
        tpl, v = anti_unify(self.p(("A", {"x": 1}), ("B", {"y": 1})), self.p(("A", {"x": 2}), ("B", {"y": 2}))); self.assertEqual(tpl[0][1]["x"], tpl[1][1]["y"]); self.assertEqual(len(v), 1)
    def test_alignment_skips(self):
        tpl, v = anti_unify(self.p(("A", {}), ("X", {}), ("B", {})), self.p(("A", {}), ("B", {}))); self.assertEqual([f for f, _ in tpl], ["A", "B"])
    def test_disjoint(self): self.assertIsNone(anti_unify(self.p(("A", {})), self.p(("B", {}))))
    def test_args(self): env = {}; self.assertEqual(anti_unify_args(3, 3, env), 3); self.assertEqual(anti_unify_args(3, 4, env), anti_unify_args(3, 4, env))
    def test_macro_instantiate(self):
        m = ParameterizedAdaptationMacro("m", [("AddModule", {"hidden": "?v0"}), ("ReuseSkill", {"skill": "?v1"})], ["?v0", "?v1"])
        self.assertEqual(m.instantiate({"?v0": 16, "?v1": "s"}), [("AddModule", {"hidden": 16}), ("ReuseSkill", {"skill": "s"})])
    def test_macro_size(self): self.assertEqual(ParameterizedAdaptationMacro("m", [("A", {})], []).size(), 1)
    def test_induce_macros(self):
        progs = [AdaptationProgram("compose", [ProgramStep("ComposeSkills", {"op": op, "a": a}), ProgramStep("ReuseSkill", {"skill": a})]) for op, a in (("and", "s1"), ("or", "s2"))]
        ms = induce_macros(progs); self.assertEqual(len(ms), 1); self.assertEqual(ms[0].support, 2); self.assertEqual(len(ms[0].variables), 2)
    def test_induce_groups(self):
        progs = [AdaptationProgram("a", [ProgramStep("A", {})]), AdaptationProgram("b", [ProgramStep("B", {})])]; self.assertEqual(len(induce_macros(progs)), 2)
    def test_library_order(self):
        lib = StrategyLibrary(); lib.record(("far",), AdaptationProgram("grow", [])); self.assertEqual(lib.order(("far",), ["reuse", "grow"])[0], "grow")
    def test_library_default(self): self.assertEqual(StrategyLibrary().order(("x",), ["reuse", "grow"]), ["reuse", "grow"])
    def test_library_size(self): lib = StrategyLibrary(); lib.record(("a",), AdaptationProgram("grow", [])); self.assertEqual(lib.size(), 1)
    @classmethod
    def setUpClass(cls): cls.econ = economy_and_compression(0, 10); cls.lib = library_ablation(0, 10)
    def test_economy_few_families(self): self.assertLessEqual(self.econ["n_families_used"], 15)
    def test_history_compressed(self): self.assertGreater(self.econ["compression_ratio"], 1.0)
    def test_macros_found(self): self.assertGreater(len(self.econ["macros"]), 0)
    def test_library_reduces_evaluations(self): self.assertLess(self.lib["with_library"]["evaluations"], self.lib["without_library"]["evaluations"])


# ================================================================ U/V/W/X/Y — crédit actif, identifiabilité, observabilité

class ActiveCreditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior = CausePrior().fit(n_per=6); cls.bench = active_credit_benchmark(30, seed=3, prior=cls.prior)
        cls.data = simulate_outcomes(6); cls.silent = silent_failure_analysis(30, 0)
    def test_policies(self): self.assertEqual(set(self.bench), {"all", "random_topk", "sensitivity_topk", "voi"})
    def test_voi_fewer_interventions(self): self.assertLess(self.bench["voi"]["interventions"], self.bench["all"]["interventions"])
    def test_voi_accuracy(self): self.assertGreaterEqual(self.bench["voi"]["accuracy"], self.bench["random_topk"]["accuracy"])
    def test_all_ten(self): self.assertEqual(self.bench["all"]["interventions"], 10.0)
    def test_prior_normalized(self):
        s = make_system(3); signs, _ = cheap_signs(s); self.assertAlmostEqual(sum(self.prior.posterior(signs).values()), 1.0, places=6)
    def test_cheap_signs_keys(self): signs, cost = cheap_signs(make_system(3)); self.assertIn("grad_top", signs); self.assertGreater(cost, 0)
    def test_planner_fault_prior(self):
        s = make_system(4); inject(s, "planner", random.Random(0)); p = self.prior.posterior(cheap_signs(s)[0])
        if s.pipe.threshold < 0.6 or s.pipe.threshold > 0.95: self.assertEqual(max(p, key=p.get), "planner")
    def test_active_diagnosis_voi(self):
        s = make_system(5); h = score(s); inject(s, "transformation", random.Random(0)); c, n, _ = active_diagnosis(s, "voi", self.prior, base_ref=h)
        self.assertEqual(c, "transformation"); self.assertLessEqual(n, 3)
    def test_measurement_set(self): self.assertEqual(len(MEASUREMENTS), 12)
    def test_observe_all(self): self.assertEqual(set(observe_all(make_system(1))), set(MEASUREMENTS))
    def test_healthy_observations(self): o = observe_all(make_system(1)); self.assertEqual(o["unit_check"], "ok"); self.assertEqual(o["routing"], "ok")
    def test_observability_healthy_zero(self):
        om = observability_matrix(self.data); self.assertTrue(all(v == 0 for v in om[HEALTHY].values()))
    def test_observability_unit(self): self.assertEqual(observability_matrix(self.data)["transformation"]["unit_check"], 1.0)
    def test_current_measure_limited(self): self.assertLess(identifiability(self.data, CURRENT)["fraction"], 1.0)
    def test_full_set_identifies(self): self.assertEqual(identifiability(self.data, list(MEASUREMENTS))["fraction"], 1.0)
    def test_distinguishable_self(self): self.assertFalse(distinguishable(self.data, "routing", "routing", list(MEASUREMENTS)))
    def test_minimal_set_cheaper(self):
        m = minimal_diagnostic_set(self.data); self.assertLess(m["cost"], m["full_cost"]); self.assertLess(len(m["set"]), len(MEASUREMENTS))
    def test_minimal_set_complete(self):
        m = minimal_diagnostic_set(self.data); self.assertEqual(identifiability(self.data, m["set"])["fraction"], identifiability(self.data, list(MEASUREMENTS))["fraction"])
    def test_silent_classes(self): self.assertTrue(set(self.silent["counts"]) <= {"detected", "identifiable_not_detected", "non_identifiable", "no_functional_effect"})
    def test_silent_mostly_identifiable(self):
        c = self.silent["counts"]; sil = sum(v for k, v in c.items() if k != "detected")
        if sil: self.assertGreaterEqual(c.get("identifiable_not_detected", 0) / sil, 0.5)
    def test_causes(self): self.assertEqual(len(CAUSES), 11)


def _make_cause_test(f):
    def t(self):
        data = ActiveCreditTests.data; om = observability_matrix(data)
        if f != HEALTHY: self.assertTrue(any(v >= 0.5 for v in om[f].values()), f"{f} inobservable")
    return t


for _c in FAULTS:
    setattr(ActiveCreditTests, f"test_observable_{''.join(ch if ch.isalnum() else '_' for ch in _c)}", _make_cause_test(_c))


# ================================================================ AC/AD/AE — banc multi-tâches

class MultiTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = task_stream(1, 10); cls.res = {m: run(m, cls.tasks, seed=1) for m in METHODS}
    def test_methods(self): self.assertEqual(len(METHODS), 7)
    def test_metric_keys(self):
        for k in ("average_accuracy", "backward_transfer", "forward_transfer", "forgetting", "param_growth", "memory_bytes", "compute_steps"): self.assertIn(k, self.res["vanilla"])
    def test_vanilla_forgets(self): self.assertGreater(self.res["vanilla"]["forgetting"], 0.1)
    def test_compositional_low_forgetting(self): self.assertLess(self.res["compositional"]["forgetting"], 0.05)
    def test_compositional_beats_replay(self): self.assertGreater(self.res["compositional"]["average_accuracy"], self.res["replay"]["average_accuracy"])
    def test_compositional_no_raw_memory_smaller(self): self.assertLess(self.res["compositional"]["memory_bytes"], self.res["replay"]["memory_bytes"])
    def test_growth_sublinear(self): self.assertLess(self.res["compositional"]["param_growth"], self.res["structural_growth"]["param_growth"])
    def test_monolithic_no_growth(self): self.assertEqual(self.res["ewc"]["param_growth"], 1.0)
    def test_recurrence_latency(self):
        for r in self.res["compositional"]["recurrence"]: self.assertEqual(r["latency_steps"], 0)
    def test_metrics_function(self):
        m = metrics([[1.0, 0.0], [0.5, 1.0]], [None, None], [0.5, 0.5]); self.assertAlmostEqual(m["forgetting"], 0.5); self.assertAlmostEqual(m["backward_transfer"], -0.5)
    def test_probes_for_shape(self): self.assertEqual(probes_for(mono_init(0), torch.zeros(D_CUE), 8).shape, (8, D_DATA + D_CUE))
    def test_mono_acc_bounds(self): t = self.tasks[0]; self.assertTrue(0 <= mono_acc(mono_init(0), *t.test_set()) <= 1)


# ================================================================ AM/AN — transfert inter-domaines, domaine nouveau

class CrossDomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.r = cross_domain(0)
    def test_domains(self): self.assertEqual(set(self.r), {"language", "algebra", "code", "strings"})
    def test_retention(self):
        for d, v in self.r.items(): self.assertEqual(v["retention"], 1.0, d)
    def test_generalizes(self):
        for d, v in self.r.items(): self.assertGreaterEqual(v["generalizes"], 0.8, d)
    def test_novel_domain_reuses(self): self.assertGreater(sum(k in ("reuse", "compose", "adapt") for k in self.r["strings"]["kinds"]), 0)
    def test_prim(self): self.assertEqual(str(Prim("Append", "x")), "Append(x)")
    def test_library_reuse(self):
        dom = SYM_DOMAINS["strings"](); lib = SymbolicSkillLibrary(dom); xs = ["abc", "de"]; ys = [x[::-1] for x in xs]
        a = lib.learn(xs, ys, ["xyz"]); b = lib.learn(xs, ys, ["xyz"]); self.assertEqual(a, b); self.assertEqual(lib.log[-1]["kind"], "reuse")
    def test_library_compose(self):
        dom = SYM_DOMAINS["strings"](); lib = SymbolicSkillLibrary(dom); xs = ["abc", "de"]
        lib.learn(xs, [x[::-1] for x in xs], ["q"]); lib.learn(xs, [x.upper() for x in xs], ["q"]); lib.learn(xs, [x[::-1].upper() for x in xs], ["q"])
        self.assertEqual(lib.log[-1]["kind"], "compose")
    def test_library_adapt(self):
        dom = SYM_DOMAINS["strings"](); lib = SymbolicSkillLibrary(dom); xs = ["abc", "de"]
        lib.learn(xs, [x + "x" for x in xs], ["q"]); lib.learn(xs, [x + "!" for x in xs], ["q"]); self.assertEqual(lib.log[-1]["kind"], "adapt")
    def test_symbolic_stream(self): self.assertEqual(len(symbolic_stream(SYM_DOMAINS["algebra"](), 0, 6)), 6)
    def test_run_domain(self): self.assertEqual(run_domain("code", 1)["domain"], "code")


# ================================================================ AO — lot scellé

class SealedTests(unittest.TestCase):
    def test_hash(self): self.assertEqual(sealed_hash(), SEALED_SHA256)
    def test_streams(self): self.assertEqual([len(s) for s in sealed_streams()], [20, 20, 20])
    def test_disjoint_from_dev(self): self.assertNotEqual([t.spec for t in sealed_streams()[0]], [t.spec for t in task_stream(1000, 20)])


if __name__ == "__main__":
    unittest.main()

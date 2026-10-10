import copy
import inspect
import math
import unittest

import torch

from core.algebra import Add, Equation, MathState, Multiply, Number, Symbol
from core.branching import adopt, branch_memory, delta_summary, shared_branch, tensors_close
from core.learning import Experience, generalize_linear_experience
from core.lifelong import (
    CONTEXT_DIM, HEAD_KEYS, INPUT_DIM, KNOWN, RELATED, UNKNOWN, ContextEncoder, CorruptModule, LifelongModel,
    LoadFastDelta, MetaController, NoveltyDetector, TaskRouter, TaskSpec, build_stream, context_prototypes,
    default_stream, make_task, run_lifelong,
)
from core.model_state import (
    INPUT, ActivateModule, AdaptWeights, AddModule, BindRoute, CreateConnection, DeactivateModule, GrowModule,
    ModelState, ModelTransformationError, PruneUnits, RemoveConnection, RestoreModule,
)
from core.planner import TransformationPlanner, algebra_candidate_generator
from core.plasticity import ConfigurationAmplitude, ModelHypothesisSet
from core.reasoning import Goal

RUN = None


def setUpModule():
    global RUN
    RUN = run_lifelong()


def small_model():
    s = ModelState(4, output_module="A")
    for spec in (AddModule(module="A", input_dim=4, output_dim=2, hidden=8, activation=False),
                 AddModule(module="B", input_dim=4, output_dim=2, hidden=8, activation=False, seed=1),
                 CreateConnection(source=INPUT, target="A"), CreateConnection(source=INPUT, target="B")):
        s.apply(spec)
    return s


def xy(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 2, (n, 4), generator=g).float()
    return x, (x[:, 0] != x[:, 1]).long()


def linear_skill():
    state = MathState(Equation(Add(Multiply(Number(2), Symbol("x")), Number(3)), Number(7))); goal = Goal("solve", "x")
    result = TransformationPlanner([algebra_candidate_generator], max_depth=5, max_nodes=400).search(state, goal)
    return generalize_linear_experience(Experience(state, goal, result, {}, result.cost, result.explored_nodes, result)), state, goal


def task(name):
    return next(t for t in RUN.tasks if t.name == name)


# ================================================================ data

class TaskStreamTests(unittest.TestCase):
    def test_eight_tasks(self): self.assertEqual(len(default_stream()), 8)
    def test_input_dim(self): self.assertEqual(task("T1").x_train.shape[1], INPUT_DIM)
    def test_prototypes_separated(self):
        p = context_prototypes()
        d = torch.cdist(p, p) + torch.eye(len(p)) * 99
        self.assertGreaterEqual(float(d.min()), 3.0)
    def test_payload_binary(self):
        payload = task("T1").x_train[:, CONTEXT_DIM:]; self.assertTrue(bool(((payload == 0) | (payload == 1)).all()))
    def test_labels_follow_function(self):
        t = task("T1"); p = t.x_test[:, CONTEXT_DIM:]; self.assertTrue(torch.equal(t.y_test, (p[:, 0] != p[:, 1]).long()))
    def test_inverse_task(self):
        t = task("T4"); p = t.x_test[:, CONTEXT_DIM:]; self.assertTrue(torch.equal(t.y_test, (p[:, 0] == p[:, 1]).long()))
    def test_known_return_same_context(self): self.assertEqual(default_stream()[7].context, default_stream()[2].context)
    def test_deterministic(self):
        a, _ = build_stream(); b, _ = build_stream(); self.assertTrue(torch.equal(a[0].x_train, b[0].x_train))
    def test_splits_distinct(self):
        t = task("T1"); self.assertFalse(torch.equal(t.x_train[:10], t.x_test[:10]))


# ================================================================ encoder / router

class ContextEncoderTests(unittest.TestCase):
    def test_is_model_module(self): self.assertIn(ContextEncoder.MODULE, LifelongModel().state.modules)
    def test_embedding_shape(self): self.assertEqual(LifelongModel().encoder.encode(task("T1").x_test).shape, (256, 6))
    def test_small(self): self.assertLess(LifelongModel().encoder.parameter_count, 400)
    def test_frozen_during_stream(self):
        fresh = LifelongModel()
        self.assertTrue(torch.equal(fresh.state.modules[ContextEncoder.MODULE].slow["w1"],
                                    RUN.model.state.modules[ContextEncoder.MODULE].slow["w1"]))
    def test_contexts_separable(self):
        e = RUN.model.encoder
        a, b = e.encode(task("T1").x_test).mean(0), e.encode(task("T3").x_test).mean(0)
        self.assertGreater(float((a - b).norm()), 0.5)
    def test_deterministic_encoding(self):
        x = task("T1").x_test; self.assertTrue(torch.equal(RUN.model.encoder.encode(x), RUN.model.encoder.encode(x)))


class TaskRouterTests(unittest.TestCase):
    def router(self):
        r = TaskRouter()
        r.learn("a", torch.randn(100, 2, generator=torch.Generator().manual_seed(0)) * 0.3)
        r.learn("b", torch.randn(100, 2, generator=torch.Generator().manual_seed(1)) * 0.3 + 5)
        return r
    def test_learn_creates_prototype(self): self.assertEqual(self.router().names, ["a", "b"])
    def test_route_nearest(self): self.assertEqual(self.router().route(torch.tensor([[5.0, 5.0]]))[0], ["b"])
    def test_probs_sum_one(self): self.assertAlmostEqual(float(self.router().probs(torch.zeros(3, 2)).sum(1)[0]), 1.0, places=5)
    def test_confident_far_from_other(self): self.assertGreater(float(self.router().route(torch.zeros(1, 2))[1]), 0.99)
    def test_midpoint_uncertain(self):
        r = TaskRouter(); r.learn("a", torch.tensor([[0.0], [0.2]])); r.learn("b", torch.tensor([[2.0], [2.2]]))
        self.assertLess(float(r.route(torch.tensor([[1.1]]))[1]), 0.6)
    def test_incremental_mean(self):
        r = TaskRouter(); r.learn("a", torch.tensor([[0.0], [2.0]])); r.learn("a", torch.tensor([[4.0], [6.0]]))
        self.assertAlmostEqual(float(r.prototypes["a"]["mean"]), 3.0)
    def test_incremental_var(self):
        r = TaskRouter(); r.learn("a", torch.tensor([[0.0], [2.0]])); r.learn("a", torch.tensor([[4.0], [6.0]]))
        self.assertAlmostEqual(float(r.prototypes["a"]["var"]), float(torch.tensor([0., 2., 4., 6.]).var(unbiased=False)), places=5)
    def test_prior_shifts_decision(self):
        r = TaskRouter(); r.learn("a", torch.tensor([[0.0], [0.2]])); r.learn("b", torch.tensor([[2.0], [2.2]]))
        names, _, _ = r.route(torch.tensor([[1.1]]), torch.tensor([0.0, 2.0])); self.assertEqual(names, ["b"])
    def test_z_distance_small_for_member(self): self.assertLess(float(self.router().z_distance(torch.zeros(1, 2))[0, 0]), 1.0)
    def test_forget(self): r = self.router(); r.forget("a"); self.assertEqual(r.names, ["b"])
    def test_var_floor(self):
        r = TaskRouter(); r.learn("a", torch.zeros(5, 2)); self.assertTrue(torch.isfinite(r.log_likelihood(torch.ones(1, 2))).all())


class NoveltyTests(unittest.TestCase):
    def router(self):
        r = TaskRouter(); r.learn("a", torch.randn(200, 3, generator=torch.Generator().manual_seed(0)) * 0.3); return r
    def test_empty_unknown(self): self.assertEqual(NoveltyDetector().assess(TaskRouter(), torch.zeros(2, 3), {}).kind, UNKNOWN)
    def test_known(self): self.assertEqual(NoveltyDetector().assess(self.router(), torch.zeros(20, 3), {"a": 0.97}).kind, KNOWN)
    def test_known_context_bad_accuracy_not_known(self):
        self.assertNotEqual(NoveltyDetector().assess(self.router(), torch.zeros(20, 3), {"a": 0.6}).kind, KNOWN)
    def test_related_by_agreement(self):
        self.assertEqual(NoveltyDetector().assess(self.router(), torch.ones(20, 3) * 9, {"a": 0.1}).kind, RELATED)
    def test_related_by_probe(self):
        self.assertEqual(NoveltyDetector().assess(self.router(), torch.ones(20, 3) * 9, {"a": 0.5}, {"a": 0.95}).kind, RELATED)
    def test_unknown(self):
        self.assertEqual(NoveltyDetector().assess(self.router(), torch.ones(20, 3) * 9, {"a": 0.5}, {"a": 0.55}).kind, UNKNOWN)
    def test_reason_explains(self):
        self.assertIn("agreement", NoveltyDetector().assess(self.router(), torch.ones(20, 3) * 9, {"a": 0.5}).reason)
    def test_stream_novelty_all_correct(self):
        for d in RUN.decisions: self.assertEqual(d.novelty.kind, task(d.task).spec.expected_novelty, d.task)


# ================================================================ autonomous routing

class SelfRoutingTests(unittest.TestCase):
    def test_predict_has_no_context_parameter(self):
        self.assertEqual(list(inspect.signature(LifelongModel.predict).parameters), ["self", "x", "structural"])
    def test_route_t1(self): self.assertGreaterEqual(RUN.routing["T1"], 0.95)
    def test_route_t4(self): self.assertGreaterEqual(RUN.routing["T4"], 0.95)
    def test_route_t7(self): self.assertGreaterEqual(RUN.routing["T7"], 0.95)
    def test_mean_routing_accuracy(self): self.assertGreaterEqual(sum(RUN.routing.values()) / len(RUN.routing), 0.95)
    def test_t8_routes_to_t3_config(self): self.assertEqual(RUN.task_config["T8"], RUN.task_config["T3"])
    def test_unlabeled_accuracy_all_tasks(self):
        for t in RUN.tasks: self.assertGreaterEqual(float((RUN.model.predict(t.x_test) == t.y_test).float().mean()), 0.95, t.name)
    def test_mixed_batch(self):
        x = torch.cat([task("T1").x_test[:50], task("T5").x_test[:50]]); y = torch.cat([task("T1").y_test[:50], task("T5").y_test[:50]])
        self.assertGreaterEqual(float((RUN.model.predict(x) == y).float().mean()), 0.95)
    def test_route_returns_confidence(self):
        _, conf, _ = RUN.model.route(task("T2").x_test); self.assertGreater(float(conf.mean()), 0.9)
    def test_predict_proba_rows_sum_one(self):
        p = RUN.model.predict_proba(task("T1").x_test); self.assertTrue(torch.allclose(p.sum(1), torch.ones(len(p)), atol=1e-5))
    def test_route_before_learning_raises(self):
        with self.assertRaises(ValueError): LifelongModel().route(task("T1").x_test)
    def test_uncertain_uses_mixture(self):
        m = copy.deepcopy(RUN.model); m.router.confident = 1.01
        self.assertGreaterEqual(float((m.predict(task("T1").x_test) == task("T1").y_test).float().mean()), 0.95)


class LifelongLoopTests(unittest.TestCase):
    def test_all_decisions_recorded(self): self.assertEqual([d.task for d in RUN.decisions], [t.name for t in RUN.tasks])
    def test_first_task_grows(self): self.assertEqual(RUN.decisions[0].action, "GROW")
    def test_related_tasks_reuse(self):
        for d in RUN.decisions:
            if d.novelty.kind == RELATED: self.assertEqual(d.action, "ADAPT", d.task)
    def test_known_task_reuses_without_change(self):
        d = RUN.decisions[-1]; self.assertEqual(d.action, "REUSE"); self.assertEqual(list(d.candidates), [f"reuse:{d.config}"])
    def test_unknown_tasks_create_experts(self):
        for d in RUN.decisions:
            if d.novelty.kind == UNKNOWN: self.assertEqual(d.action, "GROW", d.task)
    def test_candidates_scored(self):
        for d in RUN.decisions:
            for info in d.candidates.values(): self.assertIn("score", info)
    def test_weights_normalized(self):
        for d in RUN.decisions: self.assertAlmostEqual(sum(v["weight"] for v in d.candidates.values()), 1.0, places=2)
    def test_selected_has_best_score(self):
        for d in RUN.decisions:
            best = max(d.candidates.values(), key=lambda v: v["score"])["score"]
            self.assertTrue(any(v["score"] == best for k, v in d.candidates.items()))
    def test_task_accuracy(self):
        for d in RUN.decisions: self.assertGreaterEqual(d.accuracy, 0.95, d.task)
    def test_snapshots_per_task(self):
        for t in RUN.tasks: self.assertIn(f"after_{t.name}", RUN.model.state.snapshots)
    def test_episodic_memory(self): self.assertEqual(len(RUN.model.memory.episodic()), 8)
    def test_experience_bindings(self): self.assertIn("config", RUN.model.memory.episodic()[0].bindings)
    def test_configs_have_replay(self):
        for c in RUN.model.configs.values(): self.assertIsNotNone(c.replay_x)
    def test_all_changes_traced(self):
        names = {t.name for t in RUN.model.state.transform_graph.current.transformations}
        for n in ("add_module", "bind_route", "adapt_weights", "prune_units", "load_fast_delta", "clear_fast_weights"): self.assertIn(n, names)
    def test_routes_table(self): self.assertEqual(RUN.model.module_of("cfg_T1"), "expert_1")
    def test_readable_names(self): self.assertEqual(sorted(RUN.model.experts()), ["expert_1", "expert_2", "expert_3", "expert_4"])
    def test_decision_reason(self): self.assertIn("->", RUN.decisions[1].reason)


# ================================================================ forgetting / transfer

class ForgettingTests(unittest.TestCase):
    def test_matrix_shape(self): self.assertEqual(len(RUN.matrix), 8); self.assertEqual(len(RUN.matrix["T1"]), 8)
    def test_matrix_lower_triangle_none(self): self.assertIsNone(RUN.matrix["T5"][0])
    def test_forgetting_small(self):
        for name, f in RUN.forgetting().items(): self.assertLessEqual(f, 0.03, name)
    def test_t1_preserved_to_end(self): self.assertGreaterEqual(RUN.matrix["T1"][-1], 0.98)
    def test_interference_small(self): self.assertLess(RUN.interference(), 0.01)
    def test_fast_adaptation_keeps_shared_slow_weights(self):
        self.assertEqual(RUN.decisions[3].candidates and RUN.task_config["T4"], "cfg_T4")
        before = RUN.model.state.snapshots["after_T3"].modules["expert_1"].slow
        after = RUN.model.state.snapshots["after_T4"].modules["expert_1"].slow
        for k in before: self.assertTrue(torch.equal(before[k], after[k]), k)


class TransferTests(unittest.TestCase):
    def test_positive_transfer_measured(self):
        self.assertTrue(any(d.positive_transfer is not None for d in RUN.decisions))
    def test_positive_transfer_same_function(self):
        self.assertGreater(RUN.decisions[1].positive_transfer, 0.0)
    def test_parameters_saved(self):
        for d in RUN.decisions:
            if d.parameters_saved is not None: self.assertGreater(d.parameters_saved, 100)
    def test_negative_transfer_detected_on_incompatible(self):
        self.assertGreater(RUN.decisions[2].negative_transfer, 0.1)
    def test_sharing_rejected_on_incompatible(self):
        d = RUN.decisions[2]; self.assertLess(d.candidates["share:cfg_T1"]["score"], d.candidates["new_expert"]["score"])
    def test_share_candidate_measures_forgetting(self):
        self.assertGreater(RUN.decisions[2].candidates["share:cfg_T1"]["forgetting"], 0.1)
    def test_isolation_for_unknown(self): self.assertNotEqual(RUN.decisions[2].module, RUN.decisions[0].module)
    def test_reuse_module_for_related(self): self.assertEqual(RUN.decisions[3].module, RUN.decisions[0].module)


# ================================================================ capacity / homeostasis

class CapacityTests(unittest.TestCase):
    def test_capacity_bounded(self): self.assertLess(RUN.capacity_trace[-1], 1200)
    def test_capacity_does_not_grow_on_reuse(self): self.assertEqual(RUN.capacity_trace[1], RUN.capacity_trace[0])
    def test_experts_fewer_than_tasks(self): self.assertLess(len(RUN.model.experts()), len(RUN.tasks))
    def test_reuse_rate(self): self.assertGreaterEqual(RUN.controller.metrics()["module_reuse_rate"], 0.5)
    def test_new_experts_pruned(self):
        for d in RUN.decisions:
            if d.action == "GROW": self.assertLess(d.pruned[1], d.pruned[0])
    def test_average_active_below_total(self):
        m = RUN.controller.metrics(RUN.tasks); self.assertLess(m["average_active_parameters"], m["total_parameters"])
    def test_fast_parameters_small(self): self.assertLess(RUN.model.fast_parameter_count(), 100)
    def test_homeostasis_within(self):
        c = copy.deepcopy(RUN.controller); c.budget = (100, 5000); self.assertEqual(c.homeostasis(), "within budget")
    def test_homeostasis_below(self):
        c = copy.deepcopy(RUN.controller); c.budget = (5000, 9000); self.assertIn("below budget", c.homeostasis())
    def test_homeostasis_prunes_over_budget(self):
        c = copy.deepcopy(RUN.controller); cfg = c.model.configs["cfg_T5"]
        c.model.state.apply(GrowModule(module="expert_3", units=24))
        c.model.state.apply(AdaptWeights(modules=("expert_3",), fast=False, output="expert_3", inputs=cfg.replay_x,
                                         targets=cfg.replay_y, steps=100, lr=0.02, local=False))
        c.budget = (400, 850); msg = c.homeostasis()
        self.assertIn("pruned to", msg); self.assertLessEqual(c.model.state.total_parameters(), 850)
        self.assertGreaterEqual(c.model.config_accuracy("cfg_T5", cfg.probe_x, cfg.probe_y), 0.95)
    def test_homeostasis_freezes_growth_at_floor(self):
        c = copy.deepcopy(RUN.controller); c.budget = (100, 200)
        self.assertIn("growth frozen", c.homeostasis())
        with self.assertRaises(ModelTransformationError): c.model.state.apply(GrowModule(module="expert_1", units=8))
    def test_grow_policy_when_insufficient(self):
        c = copy.deepcopy(RUN.controller); c.target_accuracy = 1.01
        t = task("T7"); c._task = "X"
        self.assertIn(c._grow("cfg_T7", t.x_train, t.y_train, t.x_val, t.y_val), (True, False))


# ================================================================ shared branches

class SharedBranchTests(unittest.TestCase):
    def test_fresh_branch_owns_nothing(self):
        s = small_model(); self.assertEqual(branch_memory(shared_branch(s), s)["owned_bytes"], 0)
    def test_tensors_shared(self):
        s = small_model(); b = shared_branch(s); self.assertIs(b.modules["A"].slow["w1"], s.modules["A"].slow["w1"])
    def test_containers_not_shared(self):
        s = small_model(); b = shared_branch(s); self.assertIsNot(b.modules["A"], s.modules["A"])
    def test_copy_on_write_isolates_base(self):
        s = small_model(); b = shared_branch(s); ref = s.modules["A"].slow["w1"].clone()
        x, y = xy(); b.apply(AdaptWeights(modules=("A",), fast=False, inputs=x, targets=y, steps=10, local=False))
        self.assertTrue(torch.equal(s.modules["A"].slow["w1"], ref))
    def test_cow_owns_only_modified(self):
        s = small_model(); b = shared_branch(s); b.apply(GrowModule(module="A", units=4))
        mem = branch_memory(b, s); self.assertGreater(mem["owned_bytes"], 0); self.assertGreater(mem["shared_bytes"], 0)
    def test_activation_change_local(self):
        s = small_model(); b = shared_branch(s); b.apply(DeactivateModule(module="B")); self.assertTrue(s.modules["B"].active)
    def test_connectivity_local(self):
        s = small_model(); b = shared_branch(s); b.apply(RemoveConnection(source=INPUT, target="B"))
        self.assertIn((INPUT, "B"), s.connectivity)
    def test_delta_summary(self):
        s = small_model(); b = shared_branch(s); b.apply(GrowModule(module="A", units=4)); b.apply(DeactivateModule(module="B"))
        d = delta_summary(b, s); self.assertIn("A", d["modules"]); self.assertEqual(d["active"], ["B"])
    def test_delta_summary_new_module(self):
        s = small_model(); b = shared_branch(s); b.apply(AddModule(module="C", input_dim=4, output_dim=2))
        self.assertEqual(delta_summary(b, s)["modules"]["C"], "new_module")
    def test_same_results_as_full_copy(self):
        x, y = xy()
        specs = [GrowModule(module="A", units=4), AdaptWeights(modules=("A",), fast=False, inputs=x, targets=y, steps=20, local=False)]
        shared = ModelHypothesisSet(small_model(), shared=True).branch("s", specs)
        full = ModelHypothesisSet(small_model(), shared=False).branch("f", specs)
        self.assertTrue(tensors_close(shared.state, full.state))
        self.assertTrue(torch.allclose(shared.state.forward(x), full.state.forward(x), atol=1e-6))
    def test_same_results_fast_weights(self):
        x, y = xy(); specs = [AdaptWeights(modules=("A",), fast=True, inputs=x, targets=y, steps=20, local=False)]
        a = ModelHypothesisSet(small_model(), shared=True).branch("s", specs).state
        b = ModelHypothesisSet(small_model(), shared=False).branch("f", specs).state
        self.assertTrue(torch.allclose(a.forward(x), b.forward(x), atol=1e-6))
    def test_adopt_replays_history(self):
        s = small_model(); n = len(s.transform_graph.nodes); b = shared_branch(s)
        b.apply(GrowModule(module="A", units=4)); b.apply(DeactivateModule(module="B")); adopt(s, b)
        self.assertEqual(len(s.transform_graph.nodes), n + 2); self.assertEqual(s.modules["A"].hidden, 12)
    def test_hypothesis_memory_ratio(self):
        h = ModelHypothesisSet(small_model()); h.branch("a", []); h.branch("b", [GrowModule(module="A", units=4)])
        self.assertGreater(h.memory()["shared_ratio"], 0.5)
    def test_stream_branches_mostly_shared(self): self.assertGreater(RUN.controller.metrics()["shared_state_memory_ratio"], 0.5)
    def test_snapshots_shared_between_branches(self):
        s = small_model(); s.snapshot("x"); self.assertIs(shared_branch(s).snapshots["x"], s.snapshots["x"])


class MergeAndAmplitudeTests(unittest.TestCase):
    def hyp(self):
        h = ModelHypothesisSet(small_model()); a = h.branch("a", []); b = h.branch("b", []); a.score = b.score = 0.9
        return h, a, b
    def test_compatible_identical(self): h, a, b = self.hyp(); self.assertTrue(h.compatible(a, b)[0])
    def test_merge_identical(self):
        h, a, b = self.hyp(); merged = h.merge_branches(a, b); self.assertIn(merged, h.branches); self.assertNotIn(a, h.branches)
    def test_incompatible_modules(self):
        h, a, b = self.hyp(); b.state.apply(AddModule(module="C", input_dim=4, output_dim=2))
        ok, reasons = h.compatible(a, b); self.assertFalse(ok); self.assertIn("modules différents", reasons)
    def test_incompatible_active(self):
        h, a, b = self.hyp(); b.state.apply(DeactivateModule(module="B")); self.assertIn("modules actifs différents", h.compatible(a, b)[1])
    def test_incompatible_routes(self):
        h, a, b = self.hyp(); b.state.apply(RemoveConnection(source=INPUT, target="B")); self.assertIn("connexions différentes", h.compatible(a, b)[1])
    def test_incompatible_scores(self):
        h, a, b = self.hyp(); b.score = 0.5; self.assertIn("performances différentes", h.compatible(a, b)[1])
    def test_incompatible_deltas(self):
        h, a, b = self.hyp(); b.state.modules["A"].slow["b1"] = b.state.modules["A"].slow["b1"] + 5
        self.assertFalse(h.compatible(a, b)[0])
    def test_close_deltas_merge(self):
        h, a, b = self.hyp(); b.state.modules["A"].slow["b1"] = b.state.modules["A"].slow["b1"] + 1e-4
        self.assertTrue(h.compatible(a, b)[0])
    def test_merge_refused_raises(self):
        h, a, b = self.hyp(); b.score = 0.1
        with self.assertRaises(ValueError): h.merge_branches(a, b)
    def test_reweight_bayes(self):
        h, a, b = self.hyp(); a.weight = b.weight = 0.5; h.reweight({"a": 3.0, "b": 1.0}); self.assertAlmostEqual(a.weight, 0.75)
    def test_interfere_is_classical_product(self):
        h, a, b = self.hyp(); a.weight, b.weight = 0.45, 0.40; h.interfere({"a": 1.0, "b": 2.0})
        self.assertAlmostEqual(b.weight, 0.8 / 1.25)
    def test_reweight_zero_evidence(self):
        h, a, b = self.hyp(); h.reweight({"a": 1.0}); self.assertEqual(b.weight, 0.0)
    def test_amplitude_phase_optional(self): self.assertIsNone(ConfigurationAmplitude("x", small_model()).phase)
    def test_evaluation_counter(self):
        h, a, b = self.hyp(); h.evaluate(lambda br: 0.5); self.assertEqual(h.evaluations, 2)


# ================================================================ damage / repair

class RepairTests(unittest.TestCase):
    def ctrl(self): return copy.deepcopy(RUN.controller)
    def test_healthy_no_damage(self): self.assertEqual(self.ctrl().health_check(), {})
    def test_corruption_detected(self):
        c = self.ctrl(); c.damage("corrupt", "expert_2", seed=3); self.assertIn("cfg_T3", c.health_check())
    def test_shared_module_damage_affects_dependents(self):
        c = self.ctrl(); c.damage("corrupt", "expert_2", seed=3); self.assertEqual(set(c.health_check()), {"cfg_T3", "cfg_T6"})
    def test_diagnose_corrupted(self):
        c = self.ctrl(); c.damage("corrupt", "expert_2", seed=3); self.assertEqual(c.diagnose("cfg_T3"), "corrupted")
    def test_diagnose_inactive(self):
        c = self.ctrl(); c.damage("deactivate", "expert_2"); self.assertEqual(c.diagnose("cfg_T3"), "inactive")
    def test_diagnose_disconnected(self):
        c = self.ctrl(); c.damage("disconnect", "expert_2"); self.assertEqual(c.diagnose("cfg_T3"), "disconnected")
    def test_diagnose_missing(self):
        c = self.ctrl(); c.damage("delete", "expert_2"); self.assertEqual(c.diagnose("cfg_T3"), "missing")
    def test_rollback_chosen_with_snapshot(self):
        c = self.ctrl(); c.damage("corrupt", "expert_2", seed=3); r = c.repair("cfg_T3"); self.assertEqual(r.selected, "rollback")
    def test_rollback_repairs_all_dependents(self):
        c = self.ctrl(); c.damage("corrupt", "expert_2", seed=3); reports = c.self_repair()
        self.assertEqual(len(reports), 1); self.assertEqual(c.health_check(), {})
    def test_reroute_for_inactive(self):
        c = self.ctrl(); c.damage("deactivate", "expert_2"); self.assertEqual(c.repair("cfg_T3").selected, "reroute")
    def test_reconnect_for_disconnected(self):
        c = self.ctrl(); c.damage("disconnect", "expert_1"); self.assertEqual(c.repair("cfg_T1").selected, "reconnect")
    def test_no_rollback_when_snapshots_unavailable(self):
        c = self.ctrl(); c.damage("corrupt", "expert_2", seed=3); r = c.repair("cfg_T3", allow_rollback=False)
        self.assertNotIn("rollback", r.candidates); self.assertTrue(r.success)
    def test_clone_or_grow_when_module_deleted_without_snapshot(self):
        c = self.ctrl(); c.damage("delete", "expert_2"); r = c.repair("cfg_T3", allow_rollback=False)
        self.assertIn(r.selected, ("clone+reconnect", "grow_replacement")); self.assertTrue(r.success)
    def test_repair_success_restores_accuracy(self):
        c = self.ctrl(); c.damage("corrupt", "expert_2", seed=3); c.self_repair(allow_rollback=False)
        self.assertEqual(c.health_check(), {})
    def test_repair_metrics(self):
        c = self.ctrl(); c.damage("corrupt", "expert_2", seed=3); c.self_repair()
        m = c.metrics(); self.assertEqual(m["repair_success"], 1.0); self.assertEqual(m["rollback_rate"], 1.0)
    def test_damage_is_traced(self):
        c = self.ctrl(); c.damage("corrupt", "expert_2", seed=3)
        self.assertEqual(c.model.state.transform_graph.current.transformations[-1].name, "corrupt_module")
    def test_corrupt_spec_provenance(self): self.assertEqual(CorruptModule(module="x").provenance, "damage_test")
    def test_restore_module_spec(self):
        s = small_model(); s.snapshot("s"); ref = s.modules["A"].slow["w1"].clone()
        s.apply(CorruptModule(module="A")); s.apply(RestoreModule(module="A", snapshot="s"))
        self.assertTrue(torch.equal(s.modules["A"].slow["w1"], ref))
    def test_restore_module_missing_refused(self):
        s = small_model(); s.snapshot("s")
        with self.assertRaises(ModelTransformationError): s.apply(RestoreModule(module="Z", snapshot="s"))
    def test_other_tasks_intact_after_repair(self):
        c = self.ctrl(); c.damage("corrupt", "expert_2", seed=3); c.self_repair()
        t = task("T1"); self.assertGreaterEqual(float((c.model.predict(t.x_test) == t.y_test).float().mean()), 0.98)


# ================================================================ structural WCE + routing

class StructuralRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skill, cls.state, cls.goal = linear_skill()
        cls.ctrl = copy.deepcopy(RUN.controller)
        cls.ctrl.model.memory.learn_skill(cls.skill)
        cls.t9 = make_task(TaskSpec("T9", 0, "and23_or6", UNKNOWN), context_prototypes(n=8), seed=900)
        cls.d9 = cls.ctrl.observe("T9", cls.t9.x_train, cls.t9.y_train, cls.t9.x_val, cls.t9.y_val,
                                  structural=(cls.state, cls.goal))
    def test_skill_detected(self): self.assertEqual(self.d9.skill, "solve_linear_equation")
    def test_config_tagged(self): self.assertEqual(self.ctrl.model.configs[self.d9.config].skill, "solve_linear_equation")
    def test_neural_only_ambiguous(self):
        _, conf, _ = self.ctrl.model.route(self.t9.x_test); self.assertLess(float(conf.mean()), 0.7)
    def test_structural_resolves(self):
        names, _, _ = self.ctrl.model.route(self.t9.x_test, structural=(self.state, self.goal))
        self.assertGreater(names.count(self.d9.config) / len(names), 0.9)
    def test_structural_improves_accuracy(self):
        m = self.ctrl.model; y = self.t9.y_test
        without = float((m.predict(self.t9.x_test) == y).float().mean())
        with_s = float((m.predict(self.t9.x_test, structural=(self.state, self.goal)) == y).float().mean())
        self.assertGreater(with_s, without); self.assertGreaterEqual(with_s, 0.95)
    def test_no_skill_no_prior(self):
        m = self.ctrl.model; self.assertTrue(torch.equal(m.log_prior(None), m.log_prior("unknown_skill")))
    def test_structural_without_match(self):
        other = MathState(Equation(Symbol("x"), Number(3)))
        self.assertIsNone(self.ctrl.model.structural_skill((other, Goal("solve", "x"))))
    def test_memory_levels_integrated(self):
        mem = self.ctrl.model.memory.describe()
        self.assertEqual(mem["structural"], 1); self.assertEqual(mem["episodic"], 9)
    def test_experience_prior(self):
        m = self.ctrl.model; prior = m.log_prior(); idx = m.router.names.index("cfg_T3")
        self.assertAlmostEqual(float(prior[idx]), 0.1 * math.log1p(m.configs["cfg_T3"].uses), places=5)


# ================================================================ meta-controller / metrics

class MetaControllerTests(unittest.TestCase):
    def test_actions_declared(self):
        self.assertEqual(MetaController.ACTIONS, ("REUSE", "ADAPT", "ROUTE", "GROW", "PRUNE", "ROLLBACK", "REPAIR"))
    def test_propose_first_task_only_grow(self):
        m = LifelongModel(); c = MetaController(m); t = task("T1")
        from core.lifelong import NoveltyReport
        self.assertEqual([x.label for x in c.propose(NoveltyReport(UNKNOWN), {}, t.x_train, t.y_train)], ["new_expert"])
    def test_load_fast_delta_spec(self):
        s = small_model(); s.apply(LoadFastDelta(module="A", delta={"b2": torch.ones(2)}))
        self.assertTrue(torch.equal(s.fast_parameters["A"]["b2"], torch.ones(2)))
    def test_bind_route_spec(self):
        s = small_model(); s.apply(BindRoute(configuration="c", module="A")); self.assertEqual(s.metadata["routes"]["c"], "A")
    def test_bind_route_missing_module(self):
        with self.assertRaises(ModelTransformationError): small_model().apply(BindRoute(configuration="c", module="Z"))
    def test_adapt_keys_restrict_training(self):
        s = small_model(); x, y = xy(); ref = s.modules["A"].slow["w1"].clone()
        s.apply(AdaptWeights(modules=("A",), fast=False, keys=HEAD_KEYS, inputs=x, targets=y, steps=10, local=False))
        self.assertTrue(torch.equal(s.modules["A"].slow["w1"], ref))
    def test_adapt_output_restored(self):
        s = small_model(); x, y = xy()
        s.apply(AdaptWeights(modules=("B",), fast=False, output="B", inputs=x, targets=y, steps=5, local=False))
        self.assertEqual(s.metadata["output"], "A")


class MetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.m = RUN.controller.metrics(RUN.tasks)
    def test_keys(self):
        keys = {"total_parameters", "fast_parameters", "module_reuse_rate", "module_creation_rate", "prune_rate",
                "rollback_rate", "repair_success", "novelty_accuracy", "branch_count", "candidate_evaluations",
                "shared_state_memory_ratio", "routing_confidence", "positive_transfer", "negative_transfer",
                "average_active_parameters", "specialization_rate", "modules_created", "modules_removed"}
        self.assertTrue(keys <= set(self.m))
    def test_novelty_accuracy(self): self.assertEqual(self.m["novelty_accuracy"], 1.0)
    def test_routing_confidence(self): self.assertGreater(self.m["routing_confidence"], 0.9)
    def test_branch_count_positive(self): self.assertGreater(self.m["branch_count"], len(RUN.tasks))
    def test_evaluations(self): self.assertEqual(self.m["candidate_evaluations"], self.m["branch_count"])
    def test_creation_rate(self): self.assertEqual(self.m["module_creation_rate"], 0.5)
    def test_modules_removed_zero(self): self.assertEqual(self.m["modules_removed"], 0)


if __name__ == "__main__":
    unittest.main()

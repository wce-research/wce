import copy
import unittest

import torch

from core.algebra import Add, Equation, MathState, Multiply, Number, Symbol
from core.invariants import Invariant
from core.learning import Experience, generalize_linear_experience
from core.model_state import (
    INPUT, ActivateModule, AdaptWeights, AddModule, ClearFastWeights, CloneModule, CommitFastWeights,
    CreateConnection, DeactivateModule, GrowModule, MergeModule, ModelState, ModelTransformationError,
    NeuralModule, PlasticityRegion, PruneModule, PruneUnits, RemoveConnection, RerouteConnection,
    RestoreSnapshot, StrengthenConnection, WeakenConnection,
)
from core.planner import TransformationPlanner, algebra_candidate_generator
from core.plasticity import (
    ConfigurationAmplitude, LoadContext, MemorySystem, ModelHypothesisSet, PlasticityController,
    all_inputs, build_base_model, pretrain, run_learning_loop, save_context, task_a_labels,
    task_b_labels, task_split,
)
from core.reasoning import Goal, LessThan, RuleEngine, TruthValue


def chain(dim=4):
    net = ModelState(dim, output_module="C")
    for spec in (AddModule(module="A", input_dim=dim, output_dim=4, hidden=8),
                 AddModule(module="B", input_dim=4, output_dim=4, hidden=8, seed=1),
                 AddModule(module="C", input_dim=4, output_dim=2, hidden=8, activation=False, seed=2),
                 CreateConnection(source=INPUT, target="A"), CreateConnection(source="A", target="B"),
                 CreateConnection(source="B", target="C")):
        net.apply(spec)
    return net


def data(dim=4, n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 2, (n, dim), generator=g).float()
    return x, (x[:, 0] != x[:, 1]).long()


def weights_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


class ModelStateTests(unittest.TestCase):
    def test_modules_registered(self): self.assertEqual(set(chain().modules), {"A", "B", "C"})
    def test_parameters_view(self): self.assertIn("w1", chain().parameters["A"])
    def test_active_modules(self): self.assertEqual(chain().active_modules, {"A", "B", "C"})
    def test_capacity(self): self.assertEqual(chain().capacity, 24)
    def test_connectivity(self): self.assertEqual(len(chain().connectivity), 3)
    def test_forward_shape(self): self.assertEqual(chain().forward(data()[0]).shape, (32, 2))
    def test_output_unreachable_raises(self):
        net = chain(); net.apply(RemoveConnection(source="B", target="C"))
        with self.assertRaises(ValueError): net.forward(data()[0])
    def test_relations_expose_graph_to_reasoning(self):
        facts = RuleEngine(chain()).facts
        self.assertTrue(facts.query("connects", "A", "B"))
        self.assertTrue(facts.query("is", "A", "active"))
    def test_metadata_and_snapshots_exist(self):
        net = chain(); self.assertEqual(net.metadata["output"], "C"); self.assertEqual(net.snapshots, {})
    def test_module_counts_parameters(self):
        m = NeuralModule("m", 3, 2, hidden=4); self.assertEqual(m.parameter_count, 4 * 3 + 4 + 2 * 4 + 2)
    def test_usage_count_increments(self):
        net = chain(); net.forward(data()[0]); self.assertEqual(net.modules["A"].usage_count, 1)


class FastSlowWeightTests(unittest.TestCase):
    def test_create_fast_delta_zero(self):
        net = chain(); d = net.create_fast_delta("A"); self.assertTrue(all(float(v.abs().sum()) == 0 for v in d.values()))
    def test_effective_weight_is_sum(self):
        net = chain(); net.apply_fast_delta("A", {"b1": torch.ones(8)})
        self.assertTrue(torch.equal(net.effective_weight("A", "b1"), net.modules["A"].slow["b1"] + 1))
    def test_fast_delta_changes_output(self):
        net = chain(); x = data()[0]; before = net.forward(x)
        net.apply_fast_delta("C", {"b2": torch.ones(2)}); self.assertFalse(torch.equal(before, net.forward(x)))
    def test_clear_fast_delta_restores_output(self):
        net = chain(); x = data()[0]; before = net.forward(x)
        net.apply_fast_delta("C", {"b2": torch.ones(2)}); net.clear_fast_delta("C")
        self.assertTrue(torch.equal(before, net.forward(x)))
    def test_commit_moves_delta_into_slow(self):
        net = chain(); old = net.modules["C"].slow["b2"].clone()
        net.apply_fast_delta("C", {"b2": torch.ones(2)}); net.commit_fast_delta("C")
        self.assertTrue(torch.equal(net.modules["C"].slow["b2"], old + 1)); self.assertNotIn("C", net.fast_parameters)
    def test_fast_adaptation_keeps_slow_weights(self):
        net = chain(); x, y = data(); slow = copy.deepcopy(net.modules["C"].slow)
        net.apply(AdaptWeights(modules=("C",), inputs=x, targets=y, steps=20))
        self.assertTrue(weights_equal(slow, net.modules["C"].slow)); self.assertGreater(net.fast_parameter_count(), 0)
    def test_slow_adaptation_changes_slow_weights(self):
        net = chain(); x, y = data(); slow = copy.deepcopy(net.modules["C"].slow)
        net.apply(AdaptWeights(modules=("C",), inputs=x, targets=y, steps=20, fast=False))
        self.assertFalse(weights_equal(slow, net.modules["C"].slow))
    def test_commit_spec(self):
        net = chain(); x, y = data(); net.apply(AdaptWeights(modules=("C",), inputs=x, targets=y, steps=10))
        effects = net.apply(CommitFastWeights(modules=("C",))); self.assertGreater(effects["committed"], 0)
    def test_clear_spec(self):
        net = chain(); x, y = data(); net.apply(AdaptWeights(modules=("C",), inputs=x, targets=y, steps=10))
        net.apply(ClearFastWeights()); self.assertEqual(net.fast_parameter_count(), 0)


class LocalityTests(unittest.TestCase):
    def test_only_region_modified(self):
        net = chain(); x, y = data(); before = {m: copy.deepcopy(net.modules[m].slow) for m in "AB"}
        net.apply(AdaptWeights(modules=("C",), inputs=x, targets=y, steps=20, fast=False))
        for m in "AB": self.assertTrue(weights_equal(before[m], net.modules[m].slow))
    def test_modified_count_bounded_by_region(self):
        net = chain(); x, y = data()
        effects = net.apply(AdaptWeights(modules=("C",), inputs=x, targets=y, steps=20))
        self.assertLessEqual(effects["modified_parameters"], net.modules["C"].parameter_count)
        self.assertGreater(effects["modified_parameters"], 0)
    def test_non_local_region_refused(self):
        net = chain(); x, y = data()
        with self.assertRaises(ModelTransformationError):
            net.apply(AdaptWeights(modules=("A", "B", "C"), inputs=x, targets=y, steps=5))
    def test_region_parameter_count(self):
        net = chain(); self.assertEqual(PlasticityRegion("r", ("A",)).parameter_count(net), net.modules["A"].parameter_count)
    def test_region_for_domain(self):
        model = build_base_model(); self.assertEqual(PlasticityController().region_for(model, "task_b"), ("expert_b",))
    def test_metrics_track_modified(self):
        net = chain(); x, y = data(); net.apply(AdaptWeights(modules=("C",), inputs=x, targets=y, steps=5))
        self.assertGreater(net.metrics()["modified_parameters"], 0)


class ConnectivityTests(unittest.TestCase):
    def test_create_connection(self):
        net = chain(); net.apply(CreateConnection(source="A", target="C", strength=0.5))
        self.assertEqual(net.connectivity[("A", "C")].strength, 0.5)
    def test_remove_connection(self):
        net = chain(); net.apply(RemoveConnection(source="A", target="B")); self.assertNotIn(("A", "B"), net.connectivity)
    def test_duplicate_connection_refused(self):
        with self.assertRaises(ModelTransformationError): chain().apply(CreateConnection(source="A", target="B"))
    def test_cycle_refused(self):
        net = chain(); net.apply(AddModule(module="D", input_dim=4, output_dim=4))
        net.apply(CreateConnection(source="B", target="D"))
        with self.assertRaises(ModelTransformationError): net.apply(CreateConnection(source="D", target="A"))
    def test_dimension_mismatch_refused(self):
        with self.assertRaises(ModelTransformationError): chain().apply(CreateConnection(source="C", target="B"))
    def test_budget_limit(self):
        net = chain(); net.metadata["max_connections"] = 3
        with self.assertRaises(ModelTransformationError): net.apply(CreateConnection(source="A", target="C"))
    def test_strengthen_bounded(self):
        net = chain()
        for _ in range(10): net.apply(StrengthenConnection(source="A", target="B", delta=0.5))
        self.assertEqual(net.connectivity[("A", "B")].strength, 2.0)
    def test_weaken_floor(self):
        net = chain()
        for _ in range(10): net.apply(WeakenConnection(source="A", target="B", delta=0.5))
        self.assertEqual(net.connectivity[("A", "B")].strength, 0.0)
    def test_reroute(self):
        net = chain(); net.apply(RerouteConnection(source="A", old_target="B", new_target="C"))
        self.assertIn(("A", "C"), net.connectivity); self.assertNotIn(("A", "B"), net.connectivity)
    def test_create_inverse_is_remove(self):
        net = chain(); spec = CreateConnection(source="A", target="C"); net.apply(spec); net.apply(spec.get_inverse())
        self.assertNotIn(("A", "C"), net.connectivity)
    def test_strengthen_inverse_restores(self):
        net = chain(); spec = WeakenConnection(source="A", target="B", delta=0.25)
        net.apply(spec); net.apply(spec.get_inverse()); self.assertEqual(net.connectivity[("A", "B")].strength, 1.0)
    def test_reroute_inverse(self):
        net = chain(); spec = RerouteConnection(source="A", old_target="B", new_target="C")
        net.apply(spec); net.apply(spec.get_inverse()); self.assertIn(("A", "B"), net.connectivity)
    def test_same_modules_two_routings(self):
        net = chain(); save_context(net, "t1"); x = data()[0]; out1 = net.forward(x)
        net.apply(RerouteConnection(source="A", old_target="B", new_target="C")); save_context(net, "t2")
        out2 = net.forward(x); self.assertFalse(torch.equal(out1, out2))
        net.apply(LoadContext(context="t1")); self.assertTrue(torch.equal(out1, net.forward(x)))
        self.assertEqual(set(net.modules), {"A", "B", "C"})


class ModuleTests(unittest.TestCase):
    def test_deactivate(self):
        net = chain(); net.apply(DeactivateModule(module="B")); self.assertNotIn("B", net.active_modules)
    def test_activate(self):
        net = chain(); net.apply(DeactivateModule(module="B")); net.apply(ActivateModule(module="B"))
        self.assertIn("B", net.active_modules)
    def test_activate_active_refused(self):
        with self.assertRaises(ModelTransformationError): chain().apply(ActivateModule(module="A"))
    def test_deactivated_module_skipped_in_active_params(self):
        net = chain(); net.apply(CreateConnection(source="A", target="C")); total = net.active_parameters()
        net.apply(DeactivateModule(module="B")); self.assertLess(net.active_parameters(), total)
    def test_clone(self):
        net = chain(); net.apply(CloneModule(module="B", new_name="B2"))
        self.assertTrue(weights_equal(net.modules["B"].slow, net.modules["B2"].slow)); self.assertFalse(net.modules["B2"].active)
    def test_clone_existing_name_refused(self):
        with self.assertRaises(ModelTransformationError): chain().apply(CloneModule(module="A", new_name="B"))
    def test_merge(self):
        net = chain(); net.apply(CloneModule(module="B", new_name="B2")); net.apply(MergeModule(module="B", other="B2"))
        self.assertNotIn("B2", net.modules)
    def test_merge_redirects_connections(self):
        net = chain(); net.apply(CloneModule(module="B", new_name="B2")); net.apply(CreateConnection(source="B2", target="C"))
        net.apply(MergeModule(module="B", other="B2")); self.assertIn(("B", "C"), net.connectivity)
    def test_prune_module(self):
        net = chain(); net.apply(PruneModule(module="B"))
        self.assertNotIn("B", net.modules); self.assertFalse(any("B" in k for k in net.connectivity))
    def test_add_module_inverse(self):
        net = chain(); spec = AddModule(module="D", input_dim=4, output_dim=4)
        net.apply(spec); net.apply(spec.get_inverse()); self.assertNotIn("D", net.modules)


class CapacityTests(unittest.TestCase):
    def test_grow_increases_capacity(self):
        net = chain(); net.apply(GrowModule(module="B", units=8)); self.assertEqual(net.capacity, 32)
    def test_grow_preserves_function(self):
        net = chain(); x = data()[0]; before = net.forward(x)
        net.apply(GrowModule(module="B", units=8)); self.assertTrue(torch.allclose(before, net.forward(x)))
    def test_grow_declares_function_invariant(self):
        self.assertIn(Invariant.MODEL_FUNCTION, GrowModule(module="B").preserves.invariants)
    def test_grow_over_budget_refused(self):
        net = chain(); net.metadata["max_capacity"] = 30
        with self.assertRaises(ModelTransformationError): net.apply(GrowModule(module="B", units=8))
    def test_prune_units_reduces_capacity(self):
        net = chain(); net.apply(PruneUnits(module="B", keep=4)); self.assertEqual(net.modules["B"].hidden, 4)
    def test_prune_keeps_strongest_units(self):
        net = chain(); m = net.modules["B"]; m.slow["w2"][:, 3] *= 100
        net.apply(PruneUnits(module="B", keep=1)); self.assertTrue(torch.equal(m.slow["w2"][:, 0], net.modules["B"].slow["w2"][:, 0]))
    def test_prune_invalid_keep_refused(self):
        with self.assertRaises(ModelTransformationError): chain().apply(PruneUnits(module="B", keep=99))
    def test_grow_then_prune_without_rebuild(self):
        net = chain(); x = data()[0]; ident = id(net.modules["A"])
        net.apply(GrowModule(module="B", units=8)); net.apply(PruneUnits(module="B", keep=8))
        self.assertEqual(net.capacity, 24); self.assertEqual(id(net.modules["A"]), ident)
        self.assertEqual(net.forward(x).shape, (32, 2))
    def test_growth_and_prune_counters(self):
        net = chain(); net.apply(GrowModule(module="B")); net.apply(PruneUnits(module="B", keep=4))
        self.assertEqual((net.counters["growths"] >= 1, net.counters["prunes"]), (True, 1))


class SnapshotTests(unittest.TestCase):
    def test_restore_base(self):
        net = chain(); x = data()[0]; ref = net.forward(x); net.snapshot("N0")
        net.apply(GrowModule(module="B")); net.apply(RemoveConnection(source="A", target="B"))
        net.apply(CreateConnection(source="A", target="C")); net.restore("N0")
        self.assertTrue(torch.equal(ref, net.forward(x))); self.assertEqual(net.capacity, 24)
    def test_rollback_b_returns_to_a(self):
        net = chain(); x, y = data(); net.apply(AdaptWeights(modules=("C",), inputs=x, targets=y, steps=10, fast=False))
        net.snapshot("A"); ref = net.forward(x)
        net.apply(AdaptWeights(modules=("B",), inputs=x, targets=1 - y, steps=10, fast=False))
        self.assertFalse(torch.equal(ref, net.forward(x)))
        net.restore("A"); self.assertTrue(torch.equal(ref, net.forward(x)))
    def test_restore_unknown_refused(self):
        with self.assertRaises(ModelTransformationError): chain().apply(RestoreSnapshot(snapshot="nope"))
    def test_history_kept_after_restore(self):
        net = chain(); net.snapshot("s"); n = len(net.transform_graph.nodes)
        net.apply(GrowModule(module="B")); net.restore("s")
        self.assertEqual(len(net.transform_graph.nodes), n + 2)
        self.assertEqual(net.transform_graph.current.transformations[-1].name, "restore_snapshot")
    def test_rollback_counter(self):
        net = chain(); net.snapshot("s"); net.restore("s"); self.assertEqual(net.metrics()["rollback_count"], 1)
    def test_snapshot_isolated_from_later_changes(self):
        net = chain(); snap = net.snapshot("s"); net.modules["A"].slow["b1"] += 1
        self.assertEqual(float(snap.modules["A"].slow["b1"].sum()), 0.0)


class SpecTests(unittest.TestCase):
    def test_every_change_recorded(self):
        net = chain(); self.assertEqual(net.history(), [f"N{i}" for i in range(7)])
    def test_record_has_before_after(self):
        net = chain(); net.apply(GrowModule(module="B"))
        record = net.transform_graph.current.record; self.assertEqual(record.before["capacity"] + 8, record.after["capacity"])
    def test_spec_domain_model(self):
        from core.transformations import TransformationDomain
        self.assertIs(GrowModule(module="B").domain, TransformationDomain.MODEL)
    def test_spec_has_cost_and_provenance(self):
        spec = GrowModule(module="B", units=16); self.assertEqual(spec.cost, 2.0); self.assertEqual(spec.provenance, "plasticity_controller")
    def test_refused_spec_not_recorded(self):
        net = chain(); n = len(net.transform_graph.nodes)
        with self.assertRaises(ModelTransformationError): net.apply(PruneModule(module="Z"))
        self.assertEqual(len(net.transform_graph.nodes), n)
    def test_rule_engine_can_apply_model_spec(self):
        net = chain(); RuleEngine(net).apply_transformation(DeactivateModule(module="B"))
        self.assertNotIn("B", net.active_modules)
    def test_invariants_intersected_along_path(self):
        net = chain(); net.apply(DeactivateModule(module="B"))
        self.assertNotIn(Invariant.MODEL_FUNCTION, net.transform_graph.current.invariants.invariants)


class HypothesisTests(unittest.TestCase):
    def hyp(self):
        h = ModelHypothesisSet(chain())
        h.branch("keep", []); h.branch("off", [DeactivateModule(module="B")]); h.branch("bad", [PruneModule(module="Z")])
        return h
    def test_branches_are_independent(self):
        h = self.hyp(); self.assertIn("B", h.branches[0].state.active_modules); self.assertNotIn("B", h.branches[1].state.active_modules)
    def test_base_untouched(self): self.assertIn("B", self.hyp().base.active_modules)
    def test_invalid_branch_marked(self): self.assertEqual(self.hyp().branches[2].status, "invalid")
    def test_normalize_sums_to_one(self):
        h = self.hyp(); h.evaluate(lambda b: 0.5 if b.label == "keep" else 0.3)
        self.assertAlmostEqual(sum(b.weight for b in h.valid()), 1.0)
    def test_select_best(self):
        h = self.hyp(); h.evaluate(lambda b: 0.5 if b.label == "keep" else 0.3); self.assertEqual(h.select().label, "keep")
    def test_select_archives_others(self):
        h = self.hyp(); h.evaluate(lambda b: 0.5 if b.label == "keep" else 0.3); h.select()
        self.assertEqual([b.label for b in h.archive], ["off", "bad"])
    def test_select_reject_mode_drops(self):
        h = self.hyp(); h.evaluate(lambda b: 1.0); h.select(keep="reject"); self.assertEqual(len(h.branches), 1)
    def test_tie_prefers_fewer_active_params(self):
        h = self.hyp(); h.evaluate(lambda b: 0.5); self.assertEqual(h.select().label, "off")
    def test_merge_averages_weights(self):
        h = ModelHypothesisSet(chain()); a = h.branch("a", []); b = h.branch("b", [])
        b.state.modules["A"].slow["b1"] = b.state.modules["A"].slow["b1"] + 2; a.weight = b.weight = 0.5
        merged = h.merge(a, b); self.assertTrue(torch.allclose(merged.state.modules["A"].slow["b1"], torch.ones(8)))
    def test_merge_incompatible_refused(self):
        h = ModelHypothesisSet(chain()); a = h.branch("a", []); b = h.branch("b", [PruneModule(module="B")])
        with self.assertRaises(ValueError): h.merge(a, b)
    def test_reject_threshold(self):
        h = self.hyp(); h.evaluate(lambda b: 0.9 if b.label == "keep" else 0.1); h.reject(0.2)
        self.assertEqual([b.label for b in h.branches], ["keep"])
    def test_amplitude_is_classical_probability(self):
        amp = ConfigurationAmplitude("a", chain(), 0.4); self.assertEqual(amp.probability, 0.4); self.assertIsNone(amp.phase)


class ControllerTests(unittest.TestCase):
    def test_weakness_detected(self): self.assertTrue(PlasticityController().detect_weakness(None, Goal("accuracy"), 0.5))
    def test_no_weakness(self): self.assertFalse(PlasticityController().detect_weakness(None, Goal("accuracy"), 0.95))
    def test_goal_threshold_used(self):
        goal = Goal("accuracy", "t", LessThan(0, 0.4))
        self.assertFalse(PlasticityController().detect_weakness(None, goal, 0.5))
    def test_propose_three_candidates(self):
        x, y = data(10)
        plans = PlasticityController().propose(build_base_model(), "task_b", x, y)
        self.assertEqual([p.label for p in plans], ["fast_weights", "activate_module", "grow_module"])
    def test_no_candidates_without_weakness(self):
        x, y = data(10)
        self.assertEqual(PlasticityController().propose(build_base_model(), "task_b", x, y, Goal("accuracy"), 0.99), [])
    def test_candidates_are_specs(self):
        x, y = data(10)
        for plan in PlasticityController().propose(build_base_model(), "task_b", x, y):
            self.assertTrue(all(hasattr(s, "preconditions") for s in plan.specs))


class SkillInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        state = MathState(Equation(Add(Multiply(Number(2), Symbol("x")), Number(3)), Number(7))); goal = Goal("solve", "x")
        result = TransformationPlanner([algebra_candidate_generator], max_depth=5, max_nodes=400).search(state, goal)
        cls.experience = Experience(state, goal, result, {}, result.cost, result.explored_nodes, result)
        cls.skill = generalize_linear_experience(cls.experience)
    def test_reliable_skill_no_neural_change(self):
        skill = copy.copy(self.skill); object.__setattr__(skill, "usage_count", 3); object.__setattr__(skill, "success_count", 3)
        self.assertEqual(PlasticityController().decide_for_skill(skill)[0], "no_neural_change")
    def test_unreliable_skill_fast_weights(self):
        self.assertEqual(PlasticityController().decide_for_skill(self.skill)[0], "fast_weights")
    def test_weak_detector_fast_weights(self):
        self.assertEqual(PlasticityController().decide_for_skill(self.skill, 0.7)[0], "fast_weights")
    def test_missing_detector_adds_module(self):
        self.assertEqual(PlasticityController().decide_for_skill(self.skill, 0.3)[0], "add_module")
    def test_memory_levels(self):
        memory = MemorySystem(chain(), experiences=[self.experience]); memory.learn_skill(self.skill)
        d = memory.describe()
        self.assertEqual((d["structural"], d["episodic"], d["fast_parametric"]), (1, 1, 0))
        self.assertIn("A", memory.parametric()); self.assertIs(memory.fast_parametric(), memory.model.fast_parameters)


class ForgettingExperimentTests(unittest.TestCase):
    """Key experiment Task A -> Task B -> back to A (real training, computed once)."""
    @classmethod
    def setUpClass(cls):
        cls.xa, cls.ya, cls.va, cls.vya = task_split(task_a_labels)
        cls.xb, cls.yb, cls.vb, cls.vyb = task_split(task_b_labels)
        cls.base = build_base_model(); pretrain(cls.base, cls.xa, cls.ya)
        save_context(cls.base, "task_a"); cls.base.snapshot("task_a")
        cls.acc_a = cls.base.accuracy(cls.va, cls.vya); cls.acc_b_before = cls.base.accuracy(cls.vb, cls.vyb)
        cls.report = run_learning_loop(cls.base, "task_b", cls.xb, cls.yb, cls.vb, cls.vyb)
        cls.model = cls.report.model
    def test_task_a_learned(self): self.assertGreaterEqual(self.acc_a, 0.95)
    def test_task_b_initially_weak(self): self.assertLess(self.acc_b_before, 0.7)
    def test_weakness_detected(self): self.assertTrue(self.report.weakness)
    def test_loop_stages(self):
        self.assertEqual(self.report.stages, ["inference", "evaluate", "detect_weakness", "propose", "branch", "test",
                                              "select", "consolidate", "prune", "snapshot"])
    def test_three_branches_scored(self): self.assertEqual(len(self.report.branches), 3)
    def test_selected_local_module(self): self.assertEqual(self.report.selected, "activate_module")
    def test_task_b_restored(self): self.assertGreaterEqual(self.report.after_accuracy, 0.9)
    def test_consolidated_only_exclusive_module(self): self.assertEqual(self.report.consolidated, ("expert_b",))
    def test_capacity_grew_then_shrank(self):
        c = self.report.capacity; self.assertGreater(c["max"], c["before"]); self.assertLess(c["final"], c["max"])
    def test_shared_slow_weights_untouched(self):
        for m in ("enc", "expert_a", "head"):
            self.assertTrue(weights_equal(self.base.modules[m].slow, self.model.modules[m].slow))
    def test_return_to_task_a_preserved(self):
        model = copy.deepcopy(self.model); model.apply(LoadContext(context="task_a"))
        self.assertGreaterEqual(model.accuracy(self.va, self.vya), self.acc_a - 0.01)
    def test_switch_back_to_b(self):
        model = copy.deepcopy(self.model); model.apply(LoadContext(context="task_a")); model.apply(LoadContext(context="task_b"))
        self.assertGreaterEqual(model.accuracy(self.vb, self.vyb), 0.9)
    def test_rollback_to_task_a_snapshot(self):
        model = copy.deepcopy(self.model); model.restore("task_a")
        self.assertEqual(model.capacity, self.base.capacity); self.assertEqual(model.accuracy(self.va, self.vya), self.acc_a)
    def test_snapshot_after_loop(self): self.assertIn("after_task_b", self.model.snapshots)
    def test_all_changes_traced(self):
        names = [t.name for t in self.model.transform_graph.current.transformations]
        for expected in ("activate_module", "grow_module", "adapt_weights", "commit_fast_weights", "prune_units"):
            self.assertIn(expected, names)
    def test_metrics_complete(self):
        keys = {"total_parameters", "active_parameters", "modified_parameters", "fast_parameters", "modules_active",
                "modules_inactive", "connections", "inference_cost", "adaptation_cost", "rollback_count",
                "growth_count", "prune_count"}
        self.assertTrue(keys <= set(self.model.metrics()))
    def test_task_b_adaptation_is_local(self):
        record = next(n.record for n in self.model.transform_graph.nodes.values()
                      if n.record and n.transformations[-1].name == "adapt_weights")
        self.assertLess(record.executable["modified_parameters"], 0.6 * self.base.total_parameters())
    def test_all_inputs_enumerated(self): self.assertEqual(all_inputs(10).shape, (1024, 10))


if __name__ == "__main__":
    unittest.main()

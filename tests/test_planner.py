import unittest

from core.action import LiftAction
from core.algebra import (
    Add, AssumeSolution, DivideBothSides, Equation, MathState, Multiply, Number,
    Simplify, Symbol,
)
from core.planner import (
    SearchNode, SearchResult, SearchStrategy, StateSignature, StrategyLibrary,
    TransformationPlanner, UnknownPolicy, algebra_candidate_generator,
    lift_candidate_generator,
)
from core.reasoning import (
    AssumptionSet, Equals, Fact, FactBase, FactPattern, Goal,
    GreaterThanOrEqual, IsNonZero, PropertyRef, Rule, RuleEngine,
)
from core.transformations import MacroTransformation, TransformationSpec
from core.world import Entity, World


def equation_state():
    x = Symbol("x")
    return MathState(Equation(Add(Multiply(Number(2), x), Number(3)), Number(7)))


def solve_goal(): return Goal("solve", "x")


def result_spec(equation, name="result", cost=1):
    spec = AssumeSolution(equation, name)
    object.__setattr__(spec, "cost", cost)
    return spec


class SearchStructureTests(unittest.TestCase):
    def test_search_node_defaults(self):
        node = SearchNode(MathState(Number(0))); self.assertEqual(node.depth, 0); self.assertEqual(node.cost, 0)
    def test_search_node_path(self):
        root = SearchNode(MathState(Number(0))); child = SearchNode(root.state, root, Simplify(), depth=1)
        self.assertEqual(child.path_nodes(), [child])
    def test_search_result_failure(self): self.assertFalse(SearchResult(False, reason_if_failed="no path").success)
    def test_search_result_fields(self):
        result = SearchResult(True, cost=2, explored_nodes=3); self.assertEqual((result.cost, result.explored_nodes), (2, 3))


class StrategyTests(unittest.TestCase):
    def test_bfs_finds_direct_goal(self):
        target = Equation(Symbol("x"), Number(2)); planner = TransformationPlanner([result_spec(target)])
        self.assertTrue(planner.search(equation_state(), solve_goal()).success)
    def test_bfs_prefers_first_shallow_goal(self):
        first = result_spec(Equation(Symbol("x"), Number(2)), "first")
        second = result_spec(Equation(Symbol("x"), Number(3)), "second")
        result = TransformationPlanner([first, second]).search(equation_state(), solve_goal())
        self.assertEqual(result.transformations[0].name, "first")
    def test_uniform_cost_prefers_cheaper(self):
        expensive = result_spec(Equation(Symbol("x"), Number(9)), "expensive", 8)
        cheap = result_spec(Equation(Symbol("x"), Number(2)), "cheap", 2)
        result = TransformationPlanner([expensive, cheap], strategy=SearchStrategy.UNIFORM_COST).search(equation_state(), solve_goal())
        self.assertEqual(result.transformations[0].name, "cheap"); self.assertEqual(result.cost, 2)
    def test_dfs_supported(self):
        spec = result_spec(Equation(Symbol("x"), Number(2)))
        self.assertTrue(TransformationPlanner([spec], strategy=SearchStrategy.DFS).search(equation_state(), solve_goal()).success)
    def test_depth_limit(self):
        result = TransformationPlanner([algebra_candidate_generator], max_depth=1).search(equation_state(), solve_goal())
        self.assertFalse(result.success)
    def test_node_limit(self):
        result = TransformationPlanner([algebra_candidate_generator], max_nodes=2).search(equation_state(), solve_goal())
        self.assertFalse(result.success); self.assertIn("node limit", result.reason_if_failed)
    def test_cost_limit(self):
        result = TransformationPlanner([result_spec(Equation(Symbol("x"), Number(2)), cost=4)], max_cost=3).search(equation_state(), solve_goal())
        self.assertFalse(result.success); self.assertGreater(result.metrics.pruned_nodes, 0)
    def test_failure_without_candidates(self):
        result = TransformationPlanner([]).search(equation_state(), solve_goal())
        self.assertFalse(result.success); self.assertIn("not reachable", result.reason_if_failed)
    def test_initial_goal_satisfied(self):
        result = TransformationPlanner([]).search(MathState(Equation(Symbol("x"), Number(2))), solve_goal())
        self.assertTrue(result.success); self.assertEqual(result.transformations, [])


class UnificationTests(unittest.TestCase):
    def test_variable(self):
        self.assertEqual(FactPattern("p", "?x").match(Fact("p", (2,))), {"?x": 2})
    def test_constant(self): self.assertEqual(FactPattern("p", 2).match(Fact("p", (2,))), {})
    def test_multiple_variables(self):
        self.assertEqual(FactPattern("p", "?x", "?y").match(Fact("p", (1, 2))), {"?x": 1, "?y": 2})
    def test_repeated_variable(self): self.assertIsNone(FactPattern("p", "?x", "?x").match(Fact("p", (1, 2))))
    def test_predicate_mismatch(self): self.assertIsNone(FactPattern("p", "?x").match(Fact("q", (1,))))
    def test_constant_mismatch(self): self.assertIsNone(FactPattern("p", 2).match(Fact("p", (3,))))
    def test_substitution(self):
        fact = FactPattern("contact", "?x", "?s").instantiate({"?x": "ball", "?s": "ground"})
        self.assertEqual(fact.args, ("ball", "ground"))
    def test_rule_multiple_premises(self):
        facts = FactBase([Fact("p", ("a",)), Fact("q", ("a",))]); engine = RuleEngine(MathState(Number(0)), facts=facts)
        rule = Rule("join", (FactPattern("p", "?x"), FactPattern("q", "?x")), FactPattern("r", "?x"))
        self.assertEqual(engine.match_rule(rule), [{"?x": "a"}])


class GenericRuleTests(unittest.TestCase):
    def rule(self): return Rule("contact", (FactPattern("supported_by", "?entity", "?support"),), FactPattern("vertical_contact", "?entity", "?support"))
    def test_generic_rule_any_entity(self):
        engine = RuleEngine(World(), facts=FactBase([Fact("supported_by", ("cube", "table"))]))
        self.assertEqual(engine.fire_generic(self.rule())[0].args, ("cube", "table"))
    def test_generic_rule_multiple_matches(self):
        facts = FactBase([Fact("supported_by", ("a", "floor")), Fact("supported_by", ("b", "floor"))]); engine = RuleEngine(World(), facts=facts)
        self.assertEqual(len(engine.fire_generic(self.rule())), 2)
    def test_generic_rule_no_match(self): self.assertEqual(RuleEngine(World()).fire_generic(self.rule()), [])
    def test_planner_reaches_relational_goal(self):
        world = World(); world.add_relation("balle", "supported_by", "sol")
        result = TransformationPlanner(rules=[self.rule()]).search(world, Goal("vertical_contact", ("balle", "sol")))
        self.assertTrue(result.success); self.assertEqual(result.rules[0].name, "contact")


class SignatureTests(unittest.TestCase):
    def test_math_signature_stable(self): self.assertEqual(StateSignature.from_state(equation_state()), StateSignature.from_state(equation_state()))
    def test_math_expression_changes_signature(self):
        self.assertNotEqual(StateSignature.from_state(equation_state()), StateSignature.from_state(MathState(Number(0))))
    def test_world_signature_stable(self):
        a = World(); b = World(); a.add_entity(Entity("x", "o", {"v": 1})); b.add_entity(Entity("x", "o", {"v": 1}))
        self.assertEqual(StateSignature.from_state(a), StateSignature.from_state(b))
    def test_time_excluded(self):
        a = World(); b = World(); a.time = 1; b.time = 9
        self.assertEqual(StateSignature.from_state(a), StateSignature.from_state(b))
    def test_relations_included(self):
        a = World(); b = World(); a.add_relation("a", "on", "b")
        self.assertNotEqual(StateSignature.from_state(a), StateSignature.from_state(b))
    def test_assumptions_included(self):
        state = equation_state(); a = AssumptionSet([IsNonZero(Symbol("a"))]); b = AssumptionSet()
        self.assertNotEqual(StateSignature.from_state(state, a), StateSignature.from_state(state, b))
    def test_facts_included(self):
        state = equation_state(); self.assertNotEqual(StateSignature.from_state(state, facts=FactBase([Fact("p", ())])), StateSignature.from_state(state))
    def test_repeated_state_pruned(self):
        planner = TransformationPlanner([Simplify()], max_depth=4); result = planner.search(equation_state(), solve_goal())
        self.assertGreater(result.metrics.pruned_nodes, 0); self.assertFalse(result.success)


class DomainPlanningTests(unittest.TestCase):
    def test_algebra_planner_discovers_solution(self):
        result = TransformationPlanner([algebra_candidate_generator], max_depth=5, max_nodes=400).search(equation_state(), solve_goal())
        self.assertTrue(result.success); self.assertEqual(result.final_state.expression, Equation(Symbol("x"), Number(2)))
    def test_algebra_does_not_use_macro(self):
        result = TransformationPlanner([algebra_candidate_generator], max_depth=5, max_nodes=400).search(equation_state(), solve_goal())
        self.assertFalse(any(isinstance(spec, MacroTransformation) for spec in result.transformations))
    def test_algebra_path_contains_subtract(self):
        result = TransformationPlanner([algebra_candidate_generator], max_depth=5, max_nodes=400).search(equation_state(), solve_goal())
        self.assertTrue(any(spec.name.startswith("SubtractBothSides") for spec in result.transformations))
    def test_algebra_path_contains_divide(self):
        result = TransformationPlanner([algebra_candidate_generator], max_depth=5, max_nodes=400).search(equation_state(), solve_goal())
        self.assertTrue(any(spec.name.startswith("DivideBothSides") for spec in result.transformations))
    def test_physics_reaches_height(self):
        world = World(); world.add_entity(Entity("balle", "object", {"x": 0.0, "y": 0.0, "mass": 1.0}))
        goal = Goal("height", ("balle",), GreaterThanOrEqual(PropertyRef("balle", "y"), 2))
        result = TransformationPlanner([lift_candidate_generator(2)], max_depth=2).search(world, goal)
        self.assertTrue(result.success); self.assertEqual(result.final_state.entities["balle"].properties["y"], 2)
    def test_physics_uses_action_spec(self):
        world = World(); world.add_entity(Entity("balle", "object", {"x": 0.0, "y": 0.0, "mass": 1.0}))
        goal = Goal("height", ("balle",), GreaterThanOrEqual(PropertyRef("balle", "y"), 2))
        result = TransformationPlanner([lift_candidate_generator(2)]).search(world, goal)
        self.assertIsInstance(result.transformations[0], TransformationSpec); self.assertEqual(result.transformations[0].parameters["source_name"], "LiftAction")


class AssumptionAndPruningTests(unittest.TestCase):
    def state(self): return MathState(Equation(Multiply(Symbol("a"), Symbol("x")), Symbol("b")))
    def test_unknown_skipped(self):
        result = TransformationPlanner([DivideBothSides(Symbol("a"))], unknown_policy=UnknownPolicy.SKIP).search(self.state(), solve_goal())
        self.assertFalse(result.success)
    def test_unknown_creates_assumption(self):
        result = TransformationPlanner([DivideBothSides(Symbol("a")), Simplify()], unknown_policy=UnknownPolicy.ASSUME, max_depth=2).search(self.state(), solve_goal())
        self.assertTrue(result.success); self.assertTrue(result.assumptions.contains(IsNonZero(Symbol("a"))))
    def test_contradictory_branch_pruned(self):
        facts = FactBase([Fact("equals", (Symbol("a"), Number(0)))])
        result = TransformationPlanner([DivideBothSides(Symbol("a"))], unknown_policy=UnknownPolicy.ASSUME).search(self.state(), solve_goal(), facts=facts)
        self.assertFalse(result.success); self.assertGreater(result.metrics.pruned_nodes, 0)
    def test_known_nonzero_no_new_assumption(self):
        assumptions = AssumptionSet([IsNonZero(Symbol("a"))])
        result = TransformationPlanner([DivideBothSides(Symbol("a")), Simplify()], max_depth=2).search(self.state(), solve_goal(), assumptions=assumptions)
        self.assertTrue(result.success); self.assertEqual(len(result.assumptions), 1)


class MacroLearningTests(unittest.TestCase):
    def solve(self, library=None):
        planner = TransformationPlanner([algebra_candidate_generator], max_depth=5, max_nodes=400, strategy_library=library)
        return planner, planner.search(equation_state(), solve_goal())
    def test_promote_successful_path(self):
        planner, result = self.solve(); macro = planner.promote_to_macro(result, "linear_v1")
        self.assertIsInstance(macro, MacroTransformation); self.assertEqual(len(macro.steps), len(result.transformations))
    def test_promote_failure_refused(self):
        with self.assertRaises(ValueError): TransformationPlanner().promote_to_macro(SearchResult(False), "bad")
    def test_strategy_store_retrieve(self):
        library = StrategyLibrary(); macro = MacroTransformation(name="m", steps=(Simplify(),)); library.store(solve_goal(), macro)
        self.assertIs(library.retrieve(solve_goal()), macro)
    def test_strategy_inspectable(self):
        library = StrategyLibrary(); library.store(solve_goal(), MacroTransformation(name="m", steps=(Simplify(),)))
        self.assertEqual(len(library.inspect()), 1)
    def test_reuse_macro(self):
        library = StrategyLibrary(); planner, first = self.solve(library); macro = planner.promote_to_macro(first, "linear_v1"); planner.remember(solve_goal(), macro)
        second = planner.search(equation_state(), solve_goal())
        self.assertTrue(second.success); self.assertTrue(second.used_strategy); self.assertIsInstance(second.transformations[0], MacroTransformation)
    def test_second_solve_explores_less(self):
        library = StrategyLibrary(); planner, first = self.solve(library); planner.remember(solve_goal(), planner.promote_to_macro(first, "linear_v1"))
        second = planner.search(equation_state(), solve_goal())
        self.assertGreater(first.explored_nodes, second.explored_nodes); self.assertEqual(second.explored_nodes, 1)
    def test_macro_cost_is_path_cost(self):
        planner, result = self.solve(); macro = planner.promote_to_macro(result, "linear_v1")
        self.assertEqual(macro.cost, result.cost)


class MetricsAndExplanationTests(unittest.TestCase):
    def result(self): return TransformationPlanner([algebra_candidate_generator], max_depth=5, max_nodes=400).search(equation_state(), solve_goal())
    def test_explored_metric(self): self.assertGreater(self.result().metrics.explored_nodes, 0)
    def test_generated_metric(self): self.assertGreater(self.result().metrics.generated_nodes, 1)
    def test_depth_metric(self): self.assertGreater(self.result().metrics.max_depth_reached, 0)
    def test_branching_factor(self): self.assertGreater(self.result().metrics.branching_factor, 0)
    def test_result_cost(self): self.assertEqual(self.result().cost, sum(spec.cost for spec in self.result().transformations))
    def test_complete_explanation(self):
        planner = TransformationPlanner([algebra_candidate_generator], max_depth=5, max_nodes=400); result = planner.search(equation_state(), solve_goal()); text = planner.explain(result, solve_goal())
        self.assertIn("Goal: solve", text); self.assertIn("Chosen path", text); self.assertIn("Final: x = 2", text); self.assertIn("Nodes explored", text); self.assertIn("Cost", text)
    def test_failure_explanation(self):
        planner = TransformationPlanner([]); result = planner.search(equation_state(), solve_goal())
        self.assertIn("Goal failed", planner.explain(result, solve_goal()))


if __name__ == "__main__": unittest.main()

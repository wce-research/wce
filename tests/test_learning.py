import unittest

from core.algebra import Add, DivideBothSides, Equation, MathState, Multiply, Number, Power, Simplify, Subtract, Symbol
from core.learning import (
    Experience, GoalPattern, ParameterizedMacroTransformation, PatternVariable,
    ProblemPattern, StepTemplate, consolidate, generalize_linear_experience,
    instantiate_structure, lift_step, match_structure, pattern_specificity,
)
from core.planner import StrategyLibrary, TransformationPlanner, algebra_candidate_generator
from core.reasoning import (
    Equals, FactPattern, Goal, GreaterThanOrEqual, IsNonZero, PropertyRef, TruthValue,
)
from core.transformations import MacroTransformation, TransformationDomain
from core.world import Entity, World


def linear(a=2, b=3, c=7, variable="x"):
    return MathState(Equation(Add(Multiply(Number(a), Symbol(variable)), Number(b)), Number(c)))


def train():
    state = linear(); goal = Goal("solve", "x")
    planner = TransformationPlanner([algebra_candidate_generator], max_depth=5, max_nodes=400)
    result = planner.search(state, goal)
    experience = Experience(state, goal, result, {}, result.cost, result.explored_nodes, result)
    return planner, result, experience, generalize_linear_experience(experience)


class PatternTests(unittest.TestCase):
    def test_variable_binds(self): self.assertEqual(match_structure(PatternVariable("a"), Number(2)), {"?a": Number(2)})
    def test_variable_reuses_binding(self): self.assertEqual(match_structure((PatternVariable("a"), PatternVariable("a")), (2, 2)), {"?a": 2})
    def test_variable_conflict(self): self.assertIsNone(match_structure((PatternVariable("a"), PatternVariable("a")), (2, 3)))
    def test_constant_match(self): self.assertEqual(match_structure(Number(2), Number(2)), {})
    def test_constant_mismatch(self): self.assertIsNone(match_structure(Number(2), Number(3)))
    def test_nested_ast(self):
        p = Multiply(PatternVariable("a"), PatternVariable("x")); value = Multiply(Number(2), Symbol("x"))
        self.assertEqual(match_structure(p, value)["?a"], Number(2))
    def test_nested_type_mismatch(self): self.assertIsNone(match_structure(Multiply(PatternVariable("a"), PatternVariable("x")), Power(Symbol("x"), Number(2))))
    def test_tuple_pattern(self): self.assertEqual(match_structure((PatternVariable("x"), 2), (1, 2))["?x"], 1)
    def test_instantiate_ast(self):
        template = Multiply(PatternVariable("a"), Symbol("x")); self.assertEqual(instantiate_structure(template, {"?a": Number(5)}), Multiply(Number(5), Symbol("x")))
    def test_specificity_variable_zero(self): self.assertEqual(pattern_specificity(PatternVariable("x")), 0)
    def test_specificity_constant_positive(self): self.assertGreater(pattern_specificity(Number(2)), 0)
    def test_problem_pattern_bindings(self):
        a, x, b, c = map(PatternVariable, ("a", "x", "b", "c")); pattern = ProblemPattern(Equation(Add(Multiply(a, x), b), c))
        self.assertEqual(pattern.match(linear(5, 10, 35))["?b"], Number(10))
    def test_goal_pattern(self): self.assertIsNotNone(GoalPattern("solve", "x").match(Goal("solve", "x")))
    def test_goal_pattern_mismatch(self): self.assertIsNone(GoalPattern("solve", "x").match(Goal("solve", "y")))


class GeneralizationTests(unittest.TestCase):
    def test_experience_created(self): self.assertTrue(train()[2].successful_plan.success)
    def test_experience_timestamp(self): self.assertGreater(train()[2].timestamp, 0)
    def test_generalize_returns_parameterized_macro(self): self.assertIsInstance(train()[3], ParameterizedMacroTransformation)
    def test_generalized_name(self): self.assertEqual(train()[3].name, "solve_linear_equation")
    def test_generalized_parameters(self): self.assertEqual(len(train()[3].parameters["pattern_parameters"]), 3)
    def test_generalized_precondition(self): self.assertIsInstance(train()[3].preconditions[0], IsNonZero)
    def test_generalized_steps(self): self.assertEqual(len(train()[3].step_templates), 4)
    def test_experience_receives_bindings(self): self.assertEqual(train()[2].bindings["?a"], Number(2))
    def test_subtract_form_normalized(self):
        _, _, experience, strategy = train(); state = MathState(Equation(Subtract(Multiply(Number(3), Symbol("x")), Number(6)), Number(9)))
        self.assertEqual(strategy.match(state, Goal("solve", "x")).bindings["?b"], Number(-6))


class ParameterizedMacroTests(unittest.TestCase):
    def test_match_returns_bindings(self): self.assertEqual(train()[3].match(linear(5, 10, 35), Goal("solve", "x")).bindings["?a"], Number(5))
    def test_precondition_true(self): self.assertIs(train()[3].match(linear(5, 10, 35), Goal("solve", "x")).precondition_status, TruthValue.TRUE)
    def test_precondition_false_zero(self): self.assertIs(train()[3].match(linear(0, 3, 7), Goal("solve", "x")).precondition_status, TruthValue.FALSE)
    def test_instantiation_returns_macro(self):
        strategy = train()[3]; self.assertIsInstance(strategy.instantiate(strategy.match(linear(5, 10, 35), Goal("solve", "x"))), MacroTransformation)
    def test_instantiated_values(self):
        strategy = train()[3]; macro = strategy.instantiate(strategy.match(linear(5, 10, 35), Goal("solve", "x")))
        self.assertEqual(macro.steps[0].parameters["value"], Number(10)); self.assertEqual(macro.steps[2].parameters["value"], Number(5))
    def test_direct_compile_refused(self):
        with self.assertRaises(TypeError): train()[3].compile(linear())
    def test_explain_match(self):
        text = train()[3].explain_match(linear(5, 10, 35), Goal("solve", "x")); self.assertIn("a=5", text); self.assertIn("Preconditions: TRUE", text)
    def test_explain_mismatch(self): self.assertIn("pattern mismatch", train()[3].explain_match(MathState(Equation(Power(Symbol("x"), Number(2)), Number(4))), Goal("solve", "x")))
    def test_invariants_aggregated(self): self.assertTrue(train()[3].preserves.preserves(__import__('core.invariants', fromlist=['Invariant']).Invariant.SOLUTION_SET))
    def test_hierarchical_macro(self):
        child = train()[3]; wrapper = ParameterizedMacroTransformation("wrapper", child.input_pattern, child.goal_pattern, (), child.preconditions, (StepTemplate(child),), TransformationDomain.ALGEBRA)
        instantiated = wrapper.instantiate(wrapper.match(linear(), Goal("solve", "x"))); self.assertIsInstance(instantiated.steps[0], MacroTransformation)


class NegativeAndRefinementTests(unittest.TestCase):
    def test_power_rejected(self): self.assertIsNone(train()[3].match(MathState(Equation(Add(Power(Symbol("x"), Number(2)), Number(3)), Number(7))), Goal("solve", "x")))
    def test_two_symbols_rejected(self): self.assertIsNone(train()[3].match(MathState(Equation(Add(Symbol("x"), Symbol("y")), Number(5))), Goal("solve", "x")))
    def test_zero_rejected_by_precondition(self): self.assertIs(train()[3].match(linear(0, 3, 7), Goal("solve", "x")).precondition_status, TruthValue.FALSE)
    def test_sine_shape_rejected(self): self.assertIsNone(train()[3].match(MathState(Equation(Symbol("sin(x)"), Number(2))), Goal("solve", "x")))
    def test_record_failure(self):
        strategy = train()[3]; strategy.record_failure(); self.assertEqual((strategy.usage_count, strategy.failure_count), (1, 1))
    def test_record_success(self):
        strategy = train()[3]; strategy.record_success(); self.assertEqual((strategy.usage_count, strategy.success_count), (1, 1))
    def test_reliability(self):
        strategy = train()[3]; strategy.record_success(); strategy.record_failure(); self.assertEqual(strategy.reliability, .5)
    def test_refine_adds_precondition(self):
        strategy = train()[3]; before = len(strategy.preconditions); strategy.refine(Equals(1, 1)); self.assertEqual(len(strategy.preconditions), before + 1)
    def test_exclude_failed_instance(self):
        strategy = train()[3]; state = linear(5, 10, 35); strategy.refine(exclude_state=state)
        self.assertIs(strategy.match(state, Goal("solve", "x")).precondition_status, TruthValue.FALSE)


class LibraryAndConsolidationTests(unittest.TestCase):
    def test_store_parameterized(self):
        library = StrategyLibrary(); library.store_parameterized(train()[3]); self.assertEqual(len(library), 1)
    def test_library_match(self):
        library = StrategyLibrary(); library.store_parameterized(train()[3]); self.assertEqual(len(library.match(linear(5, 10, 35), Goal("solve", "x"))), 1)
    def test_library_no_match(self):
        library = StrategyLibrary(); library.store_parameterized(train()[3]); self.assertEqual(library.match(MathState(Number(0)), Goal("solve", "x")), [])
    def test_specific_strategy_wins(self):
        general = train()[3]; a, x, b, c = map(PatternVariable, ("a", "x", "b", "c"))
        specialized = ParameterizedMacroTransformation("special2", ProblemPattern(Equation(Add(Multiply(Number(2), x), b), c)), GoalPattern("solve", "x"), (b,c), (), general.step_templates, TransformationDomain.ALGEBRA)
        library = StrategyLibrary(); library.store_parameterized(general); library.store_parameterized(specialized)
        self.assertEqual(library.best_match(linear(), Goal("solve", "x")).strategy.name, "special2")
    def test_invalid_specific_strategy_skipped(self):
        general = train()[3]; specialized = ParameterizedMacroTransformation("invalid", general.input_pattern, general.goal_pattern, (), (Equals(1, 2),), general.step_templates, TransformationDomain.ALGEBRA)
        library = StrategyLibrary(); library.store_parameterized(specialized); library.store_parameterized(general)
        self.assertEqual(library.best_match(linear(), Goal("solve", "x")).strategy.name, "solve_linear_equation")
    def test_consolidation_successes(self):
        strategy = train()[3]; experiences = []
        for values in ((2,3,7),(5,10,35),(3,6,12)):
            state=linear(*values); experiences.append(Experience(state, Goal("solve","x"), None, {}, 0, 0, None))
        consolidate(strategy, experiences); self.assertEqual(strategy.success_count, 3)
    def test_consolidation_failure(self):
        strategy=train()[3]; bad=Experience(MathState(Number(0)), Goal("solve","x"), None, {}, 0, 0, None)
        consolidate(strategy,[bad]); self.assertEqual(strategy.failure_count,1)


class MetricsTests(unittest.TestCase):
    def test_empty_library_is_shared_with_planner(self):
        library = StrategyLibrary()
        planner = TransformationPlanner([algebra_candidate_generator], strategy_library=library)
        self.assertIs(planner.strategy_library, library)
    def test_learning_pipeline_metrics(self):
        library = StrategyLibrary()
        planner = TransformationPlanner([algebra_candidate_generator], max_depth=5, max_nodes=400, strategy_library=library)
        state, goal = linear(), Goal("solve", "x")
        first = planner.search(state, goal); self.assertGreater(first.explored_nodes, 1)
        strategy = generalize_linear_experience(Experience(state, goal, first, {}, first.cost, first.explored_nodes, first))
        library.store_parameterized(strategy)
        self.assertEqual(planner.search(linear(), goal).explored_nodes, 1)
        self.assertEqual(planner.search(linear(5, 10, 35), goal).explored_nodes, 1)
        self.assertEqual(strategy.reliability, 1.0)
        self.assertIsNone(library.best_match(MathState(Equation(Add(Power(Symbol("x"), Number(2)), Number(3)), Number(7))), goal))


class TransferTests(unittest.TestCase):
    def test_algebra_transfer_four_examples(self):
        library=StrategyLibrary(); library.store_parameterized(train()[3]); planner=TransformationPlanner([algebra_candidate_generator],strategy_library=library)
        for values, answer in (((2,3,7),2),((5,10,35),5),((3,-6,9),5),((7,1,15),2)):
            result=planner.search(linear(*values),Goal("solve","x")); self.assertTrue(result.success); self.assertEqual(result.final_state.expression,Equation(Symbol("x"),Number(answer)))
    def test_new_family_instance_one_node(self):
        library=StrategyLibrary(); library.store_parameterized(train()[3]); result=TransformationPlanner(strategy_library=library).search(linear(5,10,35),Goal("solve","x"))
        self.assertEqual(result.explored_nodes,1)
    def physical_strategy(self):
        entity, support, height = PatternVariable("entity"), PatternVariable("support"), PatternVariable("height")
        return ParameterizedMacroTransformation("lift_supported", ProblemPattern(facts=(FactPattern("supported_by", entity, support),)),
            GoalPattern("height", (entity,), GreaterThanOrEqual(PropertyRef(entity,"y"), height)), (entity,support,height), (),
            (StepTemplate(lift_step,(entity,height)),), TransformationDomain.PHYSICS)
    def world(self,name,support):
        world=World(); world.add_entity(Entity(name,"object",{"x":0.0,"y":0.0,"mass":1.0})); world.add_relation(name,"supported_by",support); return world
    def test_physics_pattern_bindings(self):
        match=self.physical_strategy().match(self.world("bloc","table"),Goal("height",("bloc",),GreaterThanOrEqual(PropertyRef("bloc","y"),3)))
        self.assertEqual(match.bindings["?support"],"table"); self.assertEqual(match.bindings["?height"],3)
    def test_physics_transfer_execution(self):
        library=StrategyLibrary(); library.store_parameterized(self.physical_strategy()); planner=TransformationPlanner(strategy_library=library)
        result=planner.search(self.world("bloc","table"),Goal("height",("bloc",),GreaterThanOrEqual(PropertyRef("bloc","y"),3)))
        self.assertTrue(result.success); self.assertEqual(result.final_state.entities["bloc"].properties["y"],3); self.assertEqual(result.explored_nodes,1)
    def test_unrelated_physics_rejected(self):
        strategy=self.physical_strategy(); world=World(); world.add_entity(Entity("bloc","object",{"y":0.0}))
        self.assertIsNone(strategy.match(world,Goal("height",("bloc",),GreaterThanOrEqual(PropertyRef("bloc","y"),3))))


if __name__ == "__main__": unittest.main()

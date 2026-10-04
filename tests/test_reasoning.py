import unittest

from core.algebra import Divide, DivideBothSides, Equation, MathState, Multiply, Number, Power, Simplify, Symbol
from core.reasoning import (
    AssumptionSet, Contradiction, Equals, Fact, FactBase, Goal, GreaterThan,
    HasProperty, HasRelation, IsNegative, IsNonZero, IsPositive, LessThan,
    NotEquals, PreconditionError, ReasoningContext, Rule, RuleEngine, TruthValue,
    UnknownPrecondition,
)
from core.world import Entity, World


def context(assumptions=(), facts=()):
    return ReasoningContext(MathState(Number(0)), FactBase(facts), AssumptionSet(assumptions))


class TruthValueTests(unittest.TestCase):
    def test_true(self): self.assertEqual(TruthValue.TRUE.value, "true")
    def test_false(self): self.assertEqual(TruthValue.FALSE.value, "false")
    def test_unknown(self): self.assertEqual(TruthValue.UNKNOWN.value, "unknown")
    def test_not_boolean(self):
        with self.assertRaises(TypeError): bool(TruthValue.TRUE)


class ConditionTests(unittest.TestCase):
    def test_concrete_equality_true(self): self.assertIs(Equals(2, 2).evaluate(context()), TruthValue.TRUE)
    def test_concrete_equality_false(self): self.assertIs(Equals(2, 3).evaluate(context()), TruthValue.FALSE)
    def test_symbolic_equality_unknown(self): self.assertIs(Equals(Symbol("a"), 2).evaluate(context()), TruthValue.UNKNOWN)
    def test_not_equals(self): self.assertIs(NotEquals(2, 3).evaluate(context()), TruthValue.TRUE)
    def test_greater_than(self): self.assertIs(GreaterThan(3, 2).evaluate(context()), TruthValue.TRUE)
    def test_less_than(self): self.assertIs(LessThan(3, 2).evaluate(context()), TruthValue.FALSE)
    def test_nonzero_true(self): self.assertIs(IsNonZero(2).evaluate(context()), TruthValue.TRUE)
    def test_nonzero_unknown(self): self.assertIs(IsNonZero(Symbol("a")).evaluate(context()), TruthValue.UNKNOWN)
    def test_positive(self): self.assertIs(IsPositive(2).evaluate(context()), TruthValue.TRUE)
    def test_negative(self): self.assertIs(IsNegative(-2).evaluate(context()), TruthValue.TRUE)
    def test_assumed_nonzero(self): self.assertIs(IsNonZero(Symbol("a")).evaluate(context([IsNonZero(Symbol("a"))])), TruthValue.TRUE)
    def test_assumed_zero_refutes_nonzero(self): self.assertIs(IsNonZero(Symbol("a")).evaluate(context([Equals(Symbol("a"), 0)])), TruthValue.FALSE)
    def test_has_property_true(self):
        world = World(); world.add_entity(Entity("ball", "object", {"mass": 1}))
        self.assertIs(HasProperty("ball", "mass").evaluate(ReasoningContext(world, FactBase(), AssumptionSet())), TruthValue.TRUE)
    def test_has_property_value_false(self):
        world = World(); world.add_entity(Entity("ball", "object", {"mass": 1}))
        self.assertIs(HasProperty("ball", "mass", 2).evaluate(ReasoningContext(world, FactBase(), AssumptionSet())), TruthValue.FALSE)
    def test_has_relation_true(self):
        world = World(); world.add_relation("ball", "supported_by", "ground"); engine = RuleEngine(world)
        self.assertIs(HasRelation("ball", "supported_by", "ground").evaluate(engine.context), TruthValue.TRUE)
    def test_has_relation_false(self):
        self.assertIs(HasRelation("ball", "supported_by", "ground").evaluate(ReasoningContext(World(), FactBase(), AssumptionSet())), TruthValue.FALSE)


class FactTests(unittest.TestCase):
    def test_add_and_query(self):
        base = FactBase(); fact = Fact("equals", ("x", 2)); base.add(fact)
        self.assertEqual(base.query("equals"), [fact])
    def test_duplicate_is_idempotent(self):
        base = FactBase(); fact = Fact("p", (1,)); base.add(fact); base.add(fact)
        self.assertEqual(len(base), 1)
    def test_opposite_polarity_contradiction(self):
        base = FactBase([Fact("supported_by", ("ball", "ground"))]); base.add(Fact("supported_by", ("ball", "ground"), False))
        self.assertEqual(len(base.contradictions), 1)
    def test_opposite_sources_still_contradict(self):
        base = FactBase([Fact("supported_by", ("ball", "ground"), source="sensor")])
        base.add(Fact("supported_by", ("ball", "ground"), False, source="rule"))
        self.assertEqual(len(base.contradictions), 1)
    def test_conflicting_equalities(self):
        base = FactBase([Fact("equals", ("x", 2))]); base.add(Fact("equals", ("x", 3), source="rule"), branch="C2")
        self.assertIsInstance(base.contradictions[0], Contradiction); self.assertEqual(base.contradictions[0].branch, "C2")
    def test_sign_contradiction(self):
        base = FactBase([Fact("positive", ("x",))]); base.add(Fact("negative", ("x",)))
        self.assertEqual(len(base.contradictions), 1)


class AssumptionTests(unittest.TestCase):
    def test_add_query(self):
        assumptions = AssumptionSet(); condition = IsNonZero(Symbol("a")); self.assertTrue(assumptions.add(condition)); self.assertTrue(assumptions.contains(condition))
    def test_inheritance(self):
        parent = AssumptionSet([IsPositive(Symbol("x"))]); child = AssumptionSet(parent=parent)
        self.assertTrue(child.contains(IsPositive(Symbol("x"))))
    def test_copy_is_independent(self):
        original = AssumptionSet([IsPositive(Symbol("x"))]); copied = original.copy(); copied.add(IsNonZero(Symbol("a")))
        self.assertFalse(original.contains(IsNonZero(Symbol("a"))))
    def test_conflict(self):
        assumptions = AssumptionSet([IsNonZero(Symbol("a"))])
        self.assertFalse(assumptions.add(Equals(Symbol("a"), 0))); self.assertEqual(len(assumptions.conflicts), 1)


class RuleTests(unittest.TestCase):
    def test_premise_satisfied(self):
        rule = Rule("r", (Equals(2, 2),), conclusion=Fact("ok", ()))
        self.assertIs(rule.status(context()), TruthValue.TRUE)
    def test_premise_false(self):
        self.assertIs(Rule("r", (Equals(2, 3),)).status(context()), TruthValue.FALSE)
    def test_premise_unknown(self):
        self.assertIs(Rule("r", (IsNonZero(Symbol("a")),)).status(context()), TruthValue.UNKNOWN)
    def test_engine_fires_conclusion(self):
        engine = RuleEngine(MathState(Number(0))); rule = Rule("r", (Equals(1, 1),), Fact("ready", ()))
        step = engine.fire(rule); self.assertTrue(engine.facts.contains(Fact("ready", ()))); self.assertEqual(step.rule.name, "r")
    def test_false_rule_not_fired(self):
        engine = RuleEngine(MathState(Number(0))); self.assertIs(engine.fire(Rule("r", (Equals(1, 2),))), TruthValue.FALSE)
    def test_applicable_rules(self):
        engine = RuleEngine(MathState(Number(0))); engine.add_rule(Rule("yes", (Equals(1, 1),))); engine.add_rule(Rule("no", (Equals(1, 2),)))
        self.assertEqual([r.name for r in engine.applicable_rules()], ["yes"])
    def test_physics_support_rule(self):
        world = World(); world.add_relation("balle", "supported_by", "sol"); engine = RuleEngine(world)
        rule = Rule("support_implies_contact", (HasRelation("balle", "supported_by", "sol"),), Fact("vertical_contact", ("balle", "sol"), source="rule"))
        engine.fire(rule); self.assertTrue(engine.facts.contains(Fact("vertical_contact", ("balle", "sol"), source="rule")))
    def test_force_property_condition(self):
        world = World(); world.add_entity(Entity("balle", "object", {"force_up": 12}))
        rule = Rule("strong_force", (HasProperty("balle", "force_up", 12),), Fact("can_leave_support", ("balle",)))
        engine = RuleEngine(world); engine.fire(rule); self.assertTrue(engine.facts.query("can_leave_support"))


class AlgebraConditionalTests(unittest.TestCase):
    def setUp(self):
        self.a, self.x, self.b = Symbol("a"), Symbol("x"), Symbol("b")
        self.equation = Equation(Multiply(self.a, self.x), self.b)
    def test_divide_precondition_declared(self): self.assertEqual(DivideBothSides(self.a).preconditions, (IsNonZero(self.a),))
    def test_a_nonzero_applies(self):
        state = MathState(self.equation); engine = RuleEngine(state, assumptions=AssumptionSet([IsNonZero(self.a)]))
        engine.apply_transformation(DivideBothSides(self.a)); engine.apply_transformation(Simplify())
        self.assertEqual(state.expression, Equation(self.x, Divide(self.b, self.a)))
    def test_a_zero_refused(self):
        engine = RuleEngine(MathState(self.equation), assumptions=AssumptionSet([Equals(self.a, 0)]))
        with self.assertRaises(PreconditionError): engine.apply_transformation(DivideBothSides(self.a))
    def test_a_unknown_requires_assumption(self):
        engine = RuleEngine(MathState(self.equation))
        with self.assertRaises(UnknownPrecondition): engine.apply_transformation(DivideBothSides(self.a))
    def test_precondition_explanation(self):
        engine = RuleEngine(MathState(self.equation)); text = engine.explain_preconditions(DivideBothSides(self.a))
        self.assertIn("IsNonZero", text); self.assertIn("UNKNOWN", text)
    def test_division_rule_and_explanation(self):
        state = MathState(self.equation); engine = RuleEngine(state, assumptions=AssumptionSet([IsNonZero(self.a)]), goal=Goal("solve", "x"))
        rule = Rule("division_nonzero", (IsNonZero(self.a),), transformation=DivideBothSides(self.a), goal_predicate="solve")
        step = engine.fire(rule); text = engine.explain(step)
        self.assertIn("CAUSE: rule division_nonzero", text); self.assertIn("CONDITION", text); self.assertIn("TRANSFORMATION", text); self.assertIn("RESULT", text)
    def test_post_fact_extracted_from_solved_equation(self):
        state = MathState(Equation(self.x, Number(2))); engine = RuleEngine(state)
        self.assertTrue(engine.facts.query("equals", self.x, Number(2)))


class BranchAndGoalTests(unittest.TestCase):
    def test_branch_assumptions_are_distinct(self):
        state = MathState(Equation(Multiply(Symbol("a"), Symbol("x")), Symbol("b"))); engine = RuleEngine(state); root = state.transform_graph.root
        nonzero = engine.branch(root, IsNonZero(Symbol("a"))); zero = engine.branch(root, Equals(Symbol("a"), 0))
        self.assertTrue(nonzero.assumptions.contains(IsNonZero(Symbol("a")))); self.assertFalse(zero.assumptions.contains(IsNonZero(Symbol("a"))))
    def test_branch_inherits_assumption(self):
        state = MathState(Number(0)); engine = RuleEngine(state); root = state.transform_graph.root
        first = engine.branch(root, IsPositive(Symbol("x"))); second = engine.branch(first, IsNonZero(Symbol("a")))
        self.assertTrue(second.assumptions.contains(IsPositive(Symbol("x"))))
    def test_x_squared_branches_carry_facts(self):
        x = Symbol("x"); state = MathState(Equation(Power(x, Number(2)), Number(4))); engine = RuleEngine(state); root = state.transform_graph.root
        plus = engine.branch(root, Equals(x, 2)); minus = engine.branch(root, Equals(x, -2))
        self.assertEqual(len(root.children), 2); self.assertTrue(plus.assumptions.contains(Equals(x, 2))); self.assertTrue(minus.assumptions.contains(Equals(x, -2)))
    def test_goal_satisfied(self):
        engine = RuleEngine(MathState(Equation(Symbol("x"), Number(2)))); self.assertTrue(Goal("solve", "x").is_satisfied(engine.context))
    def test_goal_unsatisfied(self):
        engine = RuleEngine(MathState(Equation(Multiply(Number(2), Symbol("x")), Number(4)))); self.assertFalse(Goal("solve", "x").is_satisfied(engine.context))
    def test_condition_goal(self): self.assertTrue(Goal("reach", condition=Equals(2, 2)).is_satisfied(context()))


if __name__ == "__main__": unittest.main()

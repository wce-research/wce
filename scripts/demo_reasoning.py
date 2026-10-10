"""Demonstration of the cross-cutting conditional core."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.algebra import DivideBothSides, Equation, MathState, Multiply, Number, Simplify, Symbol
from core.reasoning import (
    AssumptionSet, Equals, Fact, Goal, HasRelation, IsNonZero, Rule, RuleEngine,
)
from core.world import World


def main():
    a, x, b = Symbol("a"), Symbol("x"), Symbol("b")
    problem = Equation(Multiply(a, x), b)
    state = MathState(problem)
    engine = RuleEngine(state, goal=Goal("solve", "x"))
    division = DivideBothSides(a)

    print("=== WCE CONDITIONAL REASONING ===")
    print(f"\nProblem:\n{problem}")
    print("\nKnown facts:\n(none about a)")
    print("\nGoal:\nsolve for x")
    print("\nCondition:\na != 0")
    print(f"\nStatus:\n{division.applicability(engine.context).name}")

    root = state.transform_graph.root
    nonzero_branch = engine.branch(root, IsNonZero(a))
    rule = Rule("division_nonzero", (IsNonZero(a),), transformation=division, goal_predicate="solve")
    step = engine.fire(rule)
    engine.apply_transformation(Simplify())
    print("\nBranch 1:\nAssume a != 0")
    print("\nRule:\ndivision_nonzero")
    print(f"\nTransformation:\n{division.name}")
    print(f"\nResult:\n{state.expression}")
    print("\nReasoning explanation:\n" + engine.explain(step))

    zero_branch = engine.branch(root, Equals(a, 0))
    reduced = Equation(Number(0), b)
    print("\nBranch 2:\nAssume a = 0")
    print(f"\nAnalysis:\nequation reduces to {reduced}")
    print(f"Divide status: {division.applicability(engine.context).name}")
    print(f"Graph branches: {root.name} -> {nonzero_branch.name}, {zero_branch.name}")

    world = World()
    world.add_relation("balle", "supported_by", "sol")
    physics = RuleEngine(world)
    contact_rule = Rule(
        "support_implies_vertical_contact",
        (HasRelation("balle", "supported_by", "sol"),),
        Fact("vertical_contact", ("balle", "sol"), source="rule"),
    )
    physics_step = physics.fire(contact_rule)
    print("\nPhysics facts/rules:")
    print("Fact: supported_by(balle, sol)")
    print("Rule: support_implies_vertical_contact")
    print("Conclusion: vertical_contact(balle, sol)")
    print("Triggered: " + str(physics_step.rule.name == contact_rule.name))


if __name__ == "__main__": main()

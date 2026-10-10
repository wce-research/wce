"""Experience -> generalization -> parameterized strategy -> transfer."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.algebra import Add, Equation, MathState, Multiply, Number, Power, Symbol
from core.learning import (
    Experience, GoalPattern, ParameterizedMacroTransformation, PatternVariable, ProblemPattern,
    StepTemplate, generalize_linear_experience, lift_step,
)
from core.planner import StrategyLibrary, TransformationPlanner, algebra_candidate_generator
from core.reasoning import FactPattern, Goal, GreaterThanOrEqual, PropertyRef
from core.transformations import TransformationDomain
from core.world import Entity, World


def linear(a, b, c):
    return MathState(Equation(Add(Multiply(Number(a), Symbol("x")), Number(b)), Number(c)))


def bindings_text(bindings):
    return " ".join(f"{k.lstrip('?')}={getattr(v, 'value', v)}" for k, v in sorted(bindings.items()))


def lift_strategy():
    entity, support, height = PatternVariable("entity"), PatternVariable("support"), PatternVariable("height")
    return ParameterizedMacroTransformation(
        "lift_supported_entity", ProblemPattern(facts=(FactPattern("supported_by", entity, support),)),
        GoalPattern("height", (entity,), GreaterThanOrEqual(PropertyRef(entity, "y"), height)),
        (entity, support, height), (), (StepTemplate(lift_step, (entity, height)),), TransformationDomain.PHYSICS)


def main():
    goal = Goal("solve", "x")
    library = StrategyLibrary()
    planner = TransformationPlanner([algebra_candidate_generator], max_depth=5, max_nodes=400, strategy_library=library)
    print("=== WCE STRUCTURAL LEARNING ===")
    print("\nExperience 1:\n2x + 3 = 7\n\nNo strategy found.")
    state = linear(2, 3, 7)
    first = planner.search(state, goal)
    print(f"\nSearch:\n{first.explored_nodes} nodes\n\nPlan discovered.\n\nGeneralizing...")
    experience = Experience(state, goal, first, {}, first.cost, first.explored_nodes, first)
    strategy = generalize_linear_experience(experience)
    library.store_parameterized(strategy)
    print(f"\nLearned strategy:\n{strategy.name}\n\nPattern:\na*x + b = c\n\nPrecondition:\na != 0")

    print("\n---\n\nNew problem:\n5x + 10 = 35")
    problem = linear(5, 10, 35)
    print("\n" + strategy.explain_match(problem, goal))
    second = planner.search(problem, goal)
    print(f"\nNodes explored:\n{second.explored_nodes}\n\nResult:\n{second.final_state.expression}")
    print(f"\nReliability: {strategy.reliability:.2f} ({strategy.success_count}/{strategy.usage_count})")

    print("\n---\n\nNegative problems:")
    negatives = {
        "x² + 3 = 7": MathState(Equation(Add(Power(Symbol("x"), Number(2)), Number(3)), Number(7))),
        "0*x + 3 = 7": linear(0, 3, 7),
    }
    for label, bad in negatives.items():
        match = strategy.match(bad, goal)
        reason = "pattern mismatch" if match is None else f"{match.reason} (a != 0 -> {match.precondition_status.name})"
        print(f"\n{label}\nStrategy rejected:\n{reason}")

    print("\n---\n\nTransfer domain:")
    lift = lift_strategy()
    physics = StrategyLibrary(); physics.store_parameterized(lift)
    for name, support, height in (("balle", "sol", 2), ("bloc", "table", 3)):
        world = World(); world.add_entity(Entity(name, "object", {"x": 0.0, "y": 0.0, "mass": 1.0}))
        world.add_relation(name, "supported_by", support)
        target = Goal("height", (name,), GreaterThanOrEqual(PropertyRef(name, "y"), height))
        match = lift.match(world, target)
        print(f"\nGoal:\nmove {name} to height {height}\n\nMatched structural strategy:\nlift supported entity\n\nBindings:\n{bindings_text(match.bindings)}")
        result = TransformationPlanner(strategy_library=physics).search(world, target)
        print(f"\nNodes explored: {result.explored_nodes}\nResult:\n{'goal reached' if result.success else 'failed'}")


if __name__ == "__main__":
    main()

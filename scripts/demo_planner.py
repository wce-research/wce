"""Search -> macro -> reuse experiment."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.algebra import Add, Equation, MathState, Multiply, Number, Symbol
from core.planner import StrategyLibrary, TransformationPlanner, algebra_candidate_generator, lift_candidate_generator
from core.reasoning import Goal, GreaterThanOrEqual, PropertyRef
from core.world import Entity, World


def problem():
    x = Symbol("x")
    return MathState(Equation(Add(Multiply(Number(2), x), Number(3)), Number(7)))


def main():
    goal = Goal("solve", "x")
    library = StrategyLibrary()
    planner = TransformationPlanner([algebra_candidate_generator], max_depth=5, max_nodes=400,
                                    strategy_library=library)
    print("=== WCE TRANSFORMATION PLANNER ===")
    print("\nProblem:\n2x + 3 = 7")
    print("\nGoal:\nsolve x")
    print("\nNo strategy found.\n\nSearching...")
    first = planner.search(problem(), goal)
    print(f"\nExplored nodes: {first.explored_nodes}")
    print("\nPlan found:")
    for index, spec in enumerate(first.transformations, 1): print(f"{index}. {spec.name}")
    print(f"\nResult:\n{first.final_state.expression}")
    print("\nPlan explanation:\n" + planner.explain(first, goal))

    macro = planner.promote_to_macro(first, "solve_linear_equation_v1")
    planner.remember(goal, macro)
    print("\nPromoting plan:\n\nMacroTransformation:\nsolve_linear_equation_v1")
    print("\nRunning same problem again...\n\nStrategy found.")
    second = planner.search(problem(), goal)
    print(f"\nExplored nodes: {second.explored_nodes}")
    print(f"\nResult:\n{second.final_state.expression}")
    ratio = first.explored_nodes / max(second.explored_nodes, 1)
    print(f"\nCompression ratio:\n{ratio:.1f}x fewer explored-node units")

    world = World()
    world.add_entity(Entity("balle", "object", {"x": 0.0, "y": 0.0, "mass": 1.0}))
    height_goal = Goal("height", ("balle",), GreaterThanOrEqual(PropertyRef("balle", "y"), 2))
    physics = TransformationPlanner([lift_candidate_generator(2)], max_depth=2).search(world, height_goal)
    print("\nPhysics planning:")
    print("Initial: balle.y = 0")
    print("Goal: balle.y >= 2")
    print("Plan: " + " -> ".join(spec.name for spec in physics.transformations))
    print(f"Result: balle.y = {physics.final_state.entities['balle'].properties['y']}")


if __name__ == "__main__": main()

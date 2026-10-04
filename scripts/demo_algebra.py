"""Démonstration transversale de Transformation Algebra sur l'algèbre."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.algebra import (
    Add, AssumeSolution, DivideBothSides, Equation, MathState, Multiply, Number,
    Power, Simplify, SubtractBothSides, Symbol, solve_linear_equation_macro,
)


def names(spec): return ", ".join(sorted(i.name for i in spec.preserves.invariants))


def main():
    x = Symbol("x")
    initial = Equation(Add(Multiply(Number(2), x), Number(3)), Number(7))
    state = MathState(initial)
    print("=== WCE ALGEBRA DEMO ===")
    print(f"\nInitial:\n{initial}")

    subtract = SubtractBothSides(3)
    state.apply(subtract); state.apply(Simplify())
    print(f"\nT1:\n{subtract.name}\n\nResult:\n{state.expression}")
    print("\nPreserves:\n" + "\n".join(sorted(i.name for i in subtract.preserves.invariants)))

    divide = DivideBothSides(2)
    state.apply(divide); state.apply(Simplify())
    print(f"\nT2:\n{divide.name}\n\nResult:\n{state.expression}")
    print("\nTransformGraph:\nS0 --subtract3--> S1 --simplify--> S2 --divide2--> S3 --simplify--> S4")
    print("\nExplanation:\n" + state.explain_chain())

    macro_state = MathState(initial)
    macro = solve_linear_equation_macro(2, 3)
    macro_state.apply(macro)
    print(f"\nMacro:\n{macro.name}")
    print("Steps: " + " -> ".join(step.name for step in macro.steps))
    print(f"Final:\n{macro_state.expression}")
    print(f"Provenance: {macro_state.transform_graph.current.record.source_type}")
    print(f"Common invariants: {names(macro)}")

    roots = MathState(Equation(Power(x, Number(2)), Number(4)))
    root = roots.transform_graph.root
    positive = roots.branch(root, AssumeSolution(Equation(x, Number(2)), "positive root"))
    negative = roots.branch(root, AssumeSolution(Equation(x, Number(-2)), "negative root"))
    print("\nBranching:")
    print("x^2 = 4")
    print(f"├── {positive.configuration['__math__']['expression']}")
    print(f"└── {negative.configuration['__math__']['expression']}")


if __name__ == "__main__": main()

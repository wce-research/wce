import unittest

from core.algebra import (
    Add, AddBothSides, AssumeSolution, Divide, DivideBothSides, Equation,
    MathState, Multiply, MultiplyBothSides, Number, Power, Simplify, Subtract,
    SubtractBothSides, Substitute, Symbol, simplify, solve_linear_equation_macro,
)
from core.invariants import Invariant
from core.transformation import Transformation
from core.transformations import CompositeTransformation, MacroTransformation, TransformationDomain, TransformationSpec


class AstTests(unittest.TestCase):
    def test_number(self): self.assertEqual(Number(2).value, 2)
    def test_number_normalizes_integer_float(self): self.assertEqual(str(Number(2.0)), "2")
    def test_symbol(self): self.assertEqual(Symbol("x").name, "x")
    def test_add(self): self.assertEqual(Add(Symbol("x"), Number(3)).right, Number(3))
    def test_subtract(self): self.assertEqual(str(Subtract(Symbol("x"), Number(2))), "x - 2")
    def test_multiply(self): self.assertEqual(str(Multiply(Number(2), Symbol("x"))), "2x")
    def test_divide(self): self.assertEqual(str(Divide(Symbol("x"), Number(2))), "x / 2")
    def test_power(self): self.assertEqual(str(Power(Symbol("x"), Number(2))), "x^2")
    def test_equation(self): self.assertEqual(str(Equation(Symbol("x"), Number(2))), "x = 2")
    def test_structural_equality(self): self.assertEqual(Add(Symbol("x"), Number(1)), Add(Symbol("x"), Number(1)))
    def test_structural_inequality(self): self.assertNotEqual(Add(Symbol("x"), Number(1)), Add(Number(1), Symbol("x")))
    def test_parentheses_display(self): self.assertEqual(str(Multiply(Add(Symbol("x"), Number(1)), Number(2))), "(x + 1) * 2")


class SimplificationTests(unittest.TestCase):
    def test_add_zero_right(self): self.assertEqual(simplify(Add(Symbol("x"), Number(0))), Symbol("x"))
    def test_add_zero_left(self): self.assertEqual(simplify(Add(Number(0), Symbol("x"))), Symbol("x"))
    def test_multiply_one(self): self.assertEqual(simplify(Multiply(Number(1), Symbol("x"))), Symbol("x"))
    def test_multiply_zero(self): self.assertEqual(simplify(Multiply(Number(0), Symbol("x"))), Number(0))
    def test_add_constants(self): self.assertEqual(simplify(Add(Number(2), Number(3))), Number(5))
    def test_subtract_constants(self): self.assertEqual(simplify(Subtract(Number(7), Number(3))), Number(4))
    def test_divide_constants(self): self.assertEqual(simplify(Divide(Number(4), Number(2))), Number(2))
    def test_power_constants(self): self.assertEqual(simplify(Power(Number(2), Number(3))), Number(8))
    def test_cancel_added_constant(self): self.assertEqual(simplify(Subtract(Add(Symbol("x"), Number(3)), Number(3))), Symbol("x"))
    def test_cancel_coefficient(self): self.assertEqual(simplify(Divide(Multiply(Number(2), Symbol("x")), Number(2))), Symbol("x"))


class AlgebraTransformationTests(unittest.TestCase):
    def setUp(self): self.equation = Equation(Symbol("x"), Number(2))
    def test_add_both_sides(self): self.assertEqual(AddBothSides(3).apply(self.equation), Equation(Add(Symbol("x"), Number(3)), Add(Number(2), Number(3))))
    def test_subtract_both_sides(self): self.assertEqual(SubtractBothSides(3).apply(self.equation).left, Subtract(Symbol("x"), Number(3)))
    def test_multiply_both_sides(self): self.assertEqual(MultiplyBothSides(2).apply(self.equation).right, Multiply(Number(2), Number(2)))
    def test_divide_both_sides(self): self.assertEqual(DivideBothSides(2).apply(self.equation).left, Divide(Symbol("x"), Number(2)))
    def test_division_by_zero_refused(self):
        with self.assertRaises(ZeroDivisionError): DivideBothSides(0)
    def test_multiply_zero_not_invertible(self): self.assertFalse(MultiplyBothSides(0).invertible)
    def test_multiply_zero_not_solution_preserving(self): self.assertFalse(MultiplyBothSides(0).preserves.preserves(Invariant.SOLUTION_SET))
    def test_substitute(self): self.assertEqual(Substitute("x", 4).apply(Add(Symbol("x"), Number(1))), Add(Number(4), Number(1)))
    def test_simplify_spec(self): self.assertEqual(Simplify().apply(Add(Number(2), Number(3))), Number(5))
    def test_domain_algebra(self): self.assertEqual(AddBothSides(1).domain, TransformationDomain.ALGEBRA)
    def test_source_type_rule(self): self.assertEqual(AddBothSides(1).source_type, "rule")
    def test_equality_invariant(self): self.assertTrue(SubtractBothSides(3).preserves.preserves(Invariant.EQUALITY))
    def test_solution_set_invariant(self): self.assertTrue(DivideBothSides(2).preserves.preserves(Invariant.SOLUTION_SET))
    def test_inverse_pair(self): self.assertIsInstance(AddBothSides(2).get_inverse(), SubtractBothSides)
    def test_requires_equation(self):
        with self.assertRaises(TypeError): AddBothSides(1).apply(Symbol("x"))


class AlgebraIntegrationTests(unittest.TestCase):
    def equation(self): return Equation(Add(Multiply(Number(2), Symbol("x")), Number(3)), Number(7))
    def test_common_transformation_spec(self): self.assertIsInstance(SubtractBothSides(3), TransformationSpec)
    def test_compile_common_executable(self): self.assertIsInstance(SubtractBothSides(3).compile(MathState(self.equation())), Transformation)
    def test_math_state_is_world(self):
        from core.world import World
        self.assertIsInstance(MathState(self.equation()), World)
    def test_apply_updates_expression(self):
        state = MathState(self.equation()); state.apply(SubtractBothSides(3))
        self.assertIsInstance(state.expression.left, Subtract)
    def test_history_linear(self):
        state = MathState(self.equation()); state.apply(SubtractBothSides(3)); state.apply(Simplify())
        self.assertEqual(state.transform_graph.current.depth, 2)
        self.assertEqual(len(state.history), 2)
    def test_provenance_rule(self):
        state = MathState(self.equation()); state.apply(SubtractBothSides(3))
        self.assertEqual(state.transform_graph.current.record.source_type, "rule")
    def test_record_domain(self):
        state = MathState(self.equation()); state.apply(Simplify())
        self.assertEqual(state.transform_graph.current.record.domain, "algebra")
    def test_composite_solves_linear_equation(self):
        state = MathState(self.equation())
        composite = CompositeTransformation(name="solve", transformations=(SubtractBothSides(3), Simplify(), DivideBothSides(2), Simplify()), domain=TransformationDomain.ALGEBRA)
        state.apply(composite)
        self.assertEqual(state.expression, Equation(Symbol("x"), Number(2)))
    def test_macro_solves_linear_equation(self):
        state = MathState(self.equation()); state.apply(solve_linear_equation_macro(2, 3))
        self.assertEqual(state.expression, Equation(Symbol("x"), Number(2)))
    def test_macro_is_shared_macro(self): self.assertIsInstance(solve_linear_equation_macro(2, 3), MacroTransformation)
    def test_macro_exposes_steps(self): self.assertEqual(len(solve_linear_equation_macro(2, 3).steps), 4)
    def test_macro_in_graph(self):
        state = MathState(self.equation()); macro = solve_linear_equation_macro(2, 3); state.apply(macro)
        self.assertIs(state.transform_graph.current.record.spec, macro)
        self.assertEqual(state.transform_graph.current.record.source_type, "macro")
    def test_macro_common_invariants(self): self.assertTrue(solve_linear_equation_macro(2, 3).preserves.preserves(Invariant.SOLUTION_SET))
    def test_last_explanation(self):
        state = MathState(self.equation()); state.apply(SubtractBothSides(3))
        text = state.explain_last_change()
        self.assertIn("SubtractBothSides(3)", text); self.assertIn("SOLUTION_SET", text)
    def test_chain_explanation(self):
        state = MathState(self.equation()); state.apply(SubtractBothSides(3)); state.apply(Simplify())
        self.assertIn("Step 2: Simplify", state.explain_chain())
    def test_branch_x_squared(self):
        x = Symbol("x"); state = MathState(Equation(Power(x, Number(2)), Number(4))); root = state.transform_graph.root
        positive = state.branch(root, AssumeSolution(Equation(x, Number(2)), "positive root"))
        negative = state.branch(root, AssumeSolution(Equation(x, Number(-2)), "negative root"))
        self.assertEqual(len(root.children), 2)
        self.assertEqual(positive.configuration["__math__"]["expression"], Equation(x, Number(2)))
        self.assertEqual(negative.configuration["__math__"]["expression"], Equation(x, Number(-2)))
    def test_branch_parent_navigation(self):
        state = MathState(Equation(Power(Symbol("x"), Number(2)), Number(4))); root = state.transform_graph.root
        child = state.branch(root, AssumeSolution(Equation(Symbol("x"), Number(2))))
        self.assertIs(child.parent, root); self.assertEqual(child.get_path_names(), ["hypothesis"])
    def test_zero_coefficient_macro_refused(self):
        with self.assertRaises(ValueError): solve_linear_equation_macro(0, 3)


if __name__ == "__main__": unittest.main()

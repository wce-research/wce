"""Small symbolic core using the common Transformation Algebra of WCE."""
from dataclasses import dataclass
from typing import Any, Mapping

from core.invariants import EQUATION_EQUIVALENCE_INVARIANTS, Invariant, InvariantSet
from core.transformation import Change, Transformation, TransformationSource, apply as apply_world
from core.transformations import (
    CompositeTransformation, MacroTransformation, TransformationDomain, TransformationSpec,
)
from core.world import Entity, World


class Expression:
    def simplify(self): return simplify(self)
    def canonical(self): return simplify(self)


@dataclass(frozen=True)
class Number(Expression):
    value: int | float
    def __post_init__(self):
        if isinstance(self.value, float) and self.value.is_integer(): object.__setattr__(self, "value", int(self.value))
    def __str__(self): return str(self.value)


@dataclass(frozen=True)
class Symbol(Expression):
    name: str
    def __str__(self): return self.name


@dataclass(frozen=True)
class Binary(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True)
class Add(Binary):
    def __str__(self): return f"{_fmt(self.left, 1)} + {_fmt(self.right, 1)}"


@dataclass(frozen=True)
class Subtract(Binary):
    def __str__(self): return f"{_fmt(self.left, 1)} - {_fmt(self.right, 2)}"


@dataclass(frozen=True)
class Multiply(Binary):
    def __str__(self):
        if isinstance(self.left, Number) and isinstance(self.right, Symbol):
            return f"{self.left}{self.right}"
        return f"{_fmt(self.left, 2)} * {_fmt(self.right, 2)}"


@dataclass(frozen=True)
class Divide(Binary):
    def __str__(self): return f"{_fmt(self.left, 2)} / {_fmt(self.right, 3)}"


@dataclass(frozen=True)
class Power(Binary):
    def __str__(self): return f"{_fmt(self.left, 3)}^{_fmt(self.right, 3)}"


@dataclass(frozen=True)
class Equation(Expression):
    left: Expression
    right: Expression
    def __str__(self): return f"{self.left} = {self.right}"


def as_expression(value):
    return value if isinstance(value, Expression) else Number(value)


def _precedence(expr):
    if isinstance(expr, (Add, Subtract)): return 1
    if isinstance(expr, (Multiply, Divide)): return 2
    if isinstance(expr, Power): return 3
    return 4


def _fmt(expr, parent_precedence):
    text = str(expr)
    return f"({text})" if _precedence(expr) < parent_precedence else text


def simplify(expr):
    if isinstance(expr, Equation): return Equation(simplify(expr.left), simplify(expr.right))
    if not isinstance(expr, Binary): return expr
    left, right = simplify(expr.left), simplify(expr.right)
    if isinstance(expr, Add):
        if right == Number(0): return left
        if left == Number(0): return right
        if isinstance(left, Number) and isinstance(right, Number): return Number(left.value + right.value)
        return Add(left, right)
    if isinstance(expr, Subtract):
        if right == Number(0): return left
        if left == right: return Number(0)
        if isinstance(left, Number) and isinstance(right, Number): return Number(left.value - right.value)
        if isinstance(left, Add) and left.right == right: return left.left
        return Subtract(left, right)
    if isinstance(expr, Multiply):
        if left == Number(0) or right == Number(0): return Number(0)
        if left == Number(1): return right
        if right == Number(1): return left
        if isinstance(left, Number) and isinstance(right, Number): return Number(left.value * right.value)
        return Multiply(left, right)
    if isinstance(expr, Divide):
        if right == Number(0): raise ZeroDivisionError("symbolic division by zero")
        if left == Number(0): return Number(0)
        if right == Number(1): return left
        if isinstance(left, Number) and isinstance(right, Number): return Number(left.value / right.value)
        if isinstance(left, Multiply) and left.left == right: return left.right
        if isinstance(left, Multiply) and left.right == right: return left.left
        return Divide(left, right)
    if isinstance(expr, Power):
        if right == Number(0): return Number(1)
        if right == Number(1): return left
        if isinstance(left, Number) and isinstance(right, Number): return Number(left.value ** right.value)
        return Power(left, right)
    return expr


def substitute(expr, symbol: Symbol | str, replacement):
    symbol = Symbol(symbol) if isinstance(symbol, str) else symbol
    replacement = as_expression(replacement)
    if expr == symbol: return replacement
    if isinstance(expr, Equation): return Equation(substitute(expr.left, symbol, replacement), substitute(expr.right, symbol, replacement))
    if isinstance(expr, Binary): return type(expr)(substitute(expr.left, symbol, replacement), substitute(expr.right, symbol, replacement))
    return expr


class AlgebraTransformation(TransformationSpec):
    source_type = "rule"

    def _init(self, name, parameters, *, invertible, preserves=EQUATION_EQUIVALENCE_INVARIANTS):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "domain", TransformationDomain.ALGEBRA)
        object.__setattr__(self, "parameters", {**parameters, "source_name": type(self).__name__})
        object.__setattr__(self, "invertible", invertible)
        object.__setattr__(self, "preserves", preserves)
        object.__setattr__(self, "changes", {"expression"})
        object.__setattr__(self, "cost", 1.0)
        object.__setattr__(self, "preconditions", ())
        object.__setattr__(self, "postconditions", ())

    def apply(self, expression, *args, **kwargs): return self.transform(expression)

    def compile(self, math_state):
        before = math_state.expression
        after = self.transform(before)
        return Transformation(self.name, [Change("__math__", "expression", before, after)],
                              spec=self, source=TransformationSource.RULE,
                              domain="algebra", invariants=self.preserves)


class _BothSides(AlgebraTransformation):
    operation = None
    verb = "operate"
    inverse_type = None

    def __init__(self, value):
        value = as_expression(value)
        self._init(f"{type(self).__name__}({value})", {"value": value}, invertible=True)

    @property
    def value(self): return self.parameters["value"]

    def transform(self, expression):
        if not isinstance(expression, Equation): raise TypeError("a transformation of both sides requires an Equation")
        return Equation(self.operation(expression.left, self.value), self.operation(expression.right, self.value))

    def get_inverse(self): return self.inverse_type(self.value)


class AddBothSides(_BothSides): operation = Add
class SubtractBothSides(_BothSides): operation = Subtract
class MultiplyBothSides(_BothSides):
    operation = Multiply
    def __init__(self, value):
        value = as_expression(value)
        if value == Number(0):
            self._init(f"MultiplyBothSides({value})", {"value": value}, invertible=False,
                       preserves=InvariantSet({Invariant.EQUALITY}))
        else: super().__init__(value)


class DivideBothSides(_BothSides):
    operation = Divide
    def __init__(self, value):
        value = as_expression(value)
        if value == Number(0): raise ZeroDivisionError("DivideBothSides(0) is forbidden")
        super().__init__(value)
        from core.reasoning import IsNonZero
        object.__setattr__(self, "preconditions", (IsNonZero(value),))


AddBothSides.inverse_type = SubtractBothSides
SubtractBothSides.inverse_type = AddBothSides
MultiplyBothSides.inverse_type = DivideBothSides
DivideBothSides.inverse_type = MultiplyBothSides


class Simplify(AlgebraTransformation):
    def __init__(self): self._init("Simplify", {}, invertible=False)
    def transform(self, expression): return simplify(expression)


class Substitute(AlgebraTransformation):
    def __init__(self, symbol, expression):
        symbol = Symbol(symbol) if isinstance(symbol, str) else symbol
        expression = as_expression(expression)
        self._init(f"Substitute({symbol}, {expression})", {"symbol": symbol, "expression": expression},
                   invertible=False, preserves=InvariantSet({Invariant.EQUALITY, Invariant.DOMAIN}))
    def transform(self, expression): return substitute(expression, self.parameters["symbol"], self.parameters["expression"])


class AssumeSolution(AlgebraTransformation):
    """Explicit rule used to represent a branch/hypothesis."""
    def __init__(self, equation, label="hypothesis"):
        self._init(label, {"equation": equation}, invertible=False,
                   preserves=InvariantSet({Invariant.EQUALITY, Invariant.DOMAIN}))
    def transform(self, expression): return self.parameters["equation"]


class MathState(World):
    """Minimal specialised World: the expression is a normal entity property."""
    def __init__(self, expression, assumptions=None):
        super().__init__()
        self.assumptions = dict(assumptions or {})
        self.add_entity(Entity("__math__", "mathematical_state", {"expression": expression}))
        self.transform_graph.root.configuration = self.get_state_snapshot()

    @property
    def expression(self): return self.entities["__math__"].properties["expression"]

    def apply(self, spec): return apply_world(self, spec.compile(self))

    def branch(self, parent, spec):
        parent = self.transform_graph.get_node_by_name(parent) if isinstance(parent, str) else parent
        snapshot = parent.configuration
        self.entities["__math__"].properties["expression"] = snapshot["__math__"]["expression"]
        self.transform_graph.branch_from(parent)
        self.apply(spec)
        return self.transform_graph.current

    def explain_last_change(self):
        record = self.transform_graph.current.record
        if record is None: return "Aucune transformation algébrique."
        before = record.before["__math__"]["expression"]
        after = record.after["__math__"]["expression"]
        invariants = ", ".join(sorted(i.name for i in record.spec.preserves.invariants)) or "aucun garanti"
        reason = f"preserves {invariants}"
        if isinstance(record.spec, DivideBothSides): reason += f"; {record.spec.value} != 0"
        return f"Initial: {before}\nTransformation: {record.spec.name}\nReason: {reason}\nResult: {after}"

    def explain_chain(self):
        records = []
        node = self.transform_graph.current
        while node.parent is not None:
            if node.record: records.append(node.record)
            node = node.parent
        lines = []
        for index, record in enumerate(reversed(records), 1):
            before = record.before["__math__"]["expression"]
            after = record.after["__math__"]["expression"]
            lines.append(f"Step {index}: {record.spec.name}\n  {before} -> {after}")
        return "\n".join(lines)


def solve_linear_equation_macro(coefficient, constant):
    """Reusable macro for ax+b=c, with a non-zero."""
    if coefficient == 0: raise ValueError("the linear coefficient must be non-zero")
    steps = (SubtractBothSides(constant), Simplify(), DivideBothSides(coefficient), Simplify())
    return MacroTransformation(name="solve_linear_equation", steps=steps,
                               domain=TransformationDomain.ALGEBRA,
                               description="isoler l'inconnue dans ax+b=c")

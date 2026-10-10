"""Explicit structural learning: experience -> parameterised strategy."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from time import time
from typing import Any

from core.invariants import InvariantSet
from core.reasoning import (
    AssumptionSet, FactPattern, Goal, IsNonZero, ReasoningContext, RuleEngine, TruthValue,
)
from core.transformations import MacroTransformation, TransformationDomain, TransformationSpec


@dataclass(frozen=True)
class PatternVariable:
    name: str
    def __post_init__(self):
        if not self.name.startswith("?"): object.__setattr__(self, "name", "?" + self.name)
    def __str__(self): return self.name


def match_structure(pattern, value, bindings=None):
    """Recursive structural unification, without quantifiers or logical terms."""
    bindings = dict(bindings or {})
    if isinstance(pattern, PatternVariable):
        if pattern.name in bindings and bindings[pattern.name] != value: return None
        bindings[pattern.name] = value; return bindings
    if type(pattern) is not type(value): return None
    if is_dataclass(pattern):
        for item in fields(pattern):
            bindings = match_structure(getattr(pattern, item.name), getattr(value, item.name), bindings)
            if bindings is None: return None
        return bindings
    if isinstance(pattern, tuple):
        if len(pattern) != len(value): return None
        for left, right in zip(pattern, value):
            bindings = match_structure(left, right, bindings)
            if bindings is None: return None
        return bindings
    return bindings if pattern == value else None


def instantiate_structure(template, bindings):
    if isinstance(template, PatternVariable): return bindings[template.name]
    if is_dataclass(template):
        return type(template)(**{item.name: instantiate_structure(getattr(template, item.name), bindings) for item in fields(template)})
    if isinstance(template, tuple): return tuple(instantiate_structure(item, bindings) for item in template)
    if isinstance(template, list): return [instantiate_structure(item, bindings) for item in template]
    return template


def pattern_specificity(pattern):
    if isinstance(pattern, PatternVariable): return 0
    if is_dataclass(pattern): return 1 + sum(pattern_specificity(getattr(pattern, item.name)) for item in fields(pattern))
    if isinstance(pattern, (tuple, list)): return sum(pattern_specificity(item) for item in pattern)
    return 1


@dataclass(frozen=True)
class GoalPattern:
    predicate: Any
    target: Any = None
    condition: Any = None

    def match(self, goal, bindings=None):
        bindings = match_structure(self.predicate, goal.predicate, bindings)
        if bindings is None: return None
        bindings = match_structure(self.target, goal.target, bindings)
        if bindings is None: return None
        return match_structure(self.condition, goal.condition, bindings)


@dataclass(frozen=True)
class ProblemPattern:
    structure: Any = None
    facts: tuple[FactPattern, ...] = ()

    def match(self, state, bindings=None):
        bindings = dict(bindings or {})
        if self.structure is not None:
            expression = getattr(state, "expression", None)
            expression = normalize_linear_expression(expression)
            bindings = match_structure(self.structure, expression, bindings)
            if bindings is None: return None
        if self.facts:
            engine = RuleEngine(state)
            candidates = [bindings]
            for pattern in self.facts:
                next_candidates = []
                for current in candidates:
                    for fact in engine.facts.query(pattern.predicate):
                        matched = None
                        if pattern.predicate == fact.predicate and pattern.polarity == fact.polarity:
                            matched = match_structure(tuple(pattern.args), tuple(fact.args), current)
                        if matched is not None: next_candidates.append(matched)
                candidates = next_candidates
            return candidates[0] if candidates else None
        return bindings

    @property
    def specificity(self): return pattern_specificity(self.structure) + sum(pattern_specificity(p) for p in self.facts)


@dataclass(frozen=True)
class StepTemplate:
    factory: Any
    args: tuple[Any, ...] = ()
    kwargs: tuple[tuple[str, Any], ...] = ()
    def instantiate(self, bindings):
        args = tuple(instantiate_structure(arg, bindings) for arg in self.args)
        kwargs = {key: instantiate_structure(value, bindings) for key, value in self.kwargs}
        if isinstance(self.factory, ParameterizedMacroTransformation):
            return self.factory.instantiate_from_bindings(bindings)
        return self.factory(*args, **kwargs)


@dataclass
class StrategyMatch:
    strategy: 'ParameterizedMacroTransformation'
    bindings: dict
    precondition_status: TruthValue
    specificity: int
    reason: str = ""


class ParameterizedMacroTransformation(TransformationSpec):
    def __init__(self, name, input_pattern, goal_pattern, parameters, preconditions,
                 step_templates, domain=TransformationDomain.GENERIC, provenance="learned"):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "parameters", {"pattern_parameters": tuple(parameters), "source_name": provenance})
        object.__setattr__(self, "invertible", False)
        object.__setattr__(self, "changes", {"parameterized_state"})
        object.__setattr__(self, "cost", float(len(step_templates)))
        object.__setattr__(self, "preconditions", tuple(preconditions))
        object.__setattr__(self, "postconditions", ())
        object.__setattr__(self, "input_pattern", input_pattern)
        object.__setattr__(self, "goal_pattern", goal_pattern)
        object.__setattr__(self, "step_templates", tuple(step_templates))
        object.__setattr__(self, "provenance", provenance)
        common = None
        for template in step_templates:
            try:
                spec = template.instantiate({p.name: 1 for p in parameters})
                common = spec.preserves if common is None else common.intersection(spec.preserves)
            except Exception: pass
        object.__setattr__(self, "preserves", common or InvariantSet(set()))
        object.__setattr__(self, "usage_count", 0)
        object.__setattr__(self, "success_count", 0)
        object.__setattr__(self, "failure_count", 0)
        object.__setattr__(self, "excluded_signatures", set())

    @property
    def reliability(self): return self.success_count / self.usage_count if self.usage_count else 0.0
    @property
    def specificity(self): return self.input_pattern.specificity + pattern_specificity(self.goal_pattern)

    def match(self, state, goal, assumptions=None):
        from core.planner import StateSignature
        if StateSignature.from_state(state).data in self.excluded_signatures:
            return StrategyMatch(self, {}, TruthValue.FALSE, self.specificity, "case explicitly excluded")
        bindings = self.input_pattern.match(state)
        if bindings is None: return None
        bindings = self.goal_pattern.match(goal, bindings)
        if bindings is None: return None
        engine = RuleEngine(state, assumptions=assumptions or AssumptionSet())
        instantiated = tuple(instantiate_structure(condition, bindings) for condition in self.preconditions)
        statuses = [condition.evaluate(engine.context) for condition in instantiated]
        status = TruthValue.FALSE if TruthValue.FALSE in statuses else (TruthValue.UNKNOWN if TruthValue.UNKNOWN in statuses else TruthValue.TRUE)
        return StrategyMatch(self, bindings, status, self.specificity,
                             "precondition failed" if status is TruthValue.FALSE else "matched")

    def instantiate(self, match): return self.instantiate_from_bindings(match.bindings)
    def instantiate_from_bindings(self, bindings):
        steps = tuple(template.instantiate(bindings) for template in self.step_templates)
        if len(steps) == 1 and self.domain is TransformationDomain.PHYSICS:
            return MacroTransformation(name=self.name, base_transformation=steps[0], domain=self.domain,
                                       description=f"instantiated {self.name} with {bindings}")
        return MacroTransformation(name=self.name, steps=steps, domain=self.domain,
                                   description=f"instantiated {self.name} with {bindings}")
    def apply(self, obj, *args, **kwargs): raise TypeError("a parameterised macro must first be instantiated")
    def compile(self, state): raise TypeError("call match() then instantiate()")
    def record_success(self):
        object.__setattr__(self, "usage_count", self.usage_count + 1); object.__setattr__(self, "success_count", self.success_count + 1)
    def record_failure(self, state=None):
        object.__setattr__(self, "usage_count", self.usage_count + 1); object.__setattr__(self, "failure_count", self.failure_count + 1)
        if state is not None:
            from core.planner import StateSignature
            self.excluded_signatures.add(StateSignature.from_state(state).data)
    def refine(self, precondition=None, exclude_state=None):
        if precondition is not None: object.__setattr__(self, "preconditions", self.preconditions + (precondition,))
        if exclude_state is not None: self.record_failure(exclude_state)
        return self
    def explain_match(self, state, goal, assumptions=None):
        match = self.match(state, goal, assumptions)
        if match is None: return f"Strategy: {self.name}\nRejected: pattern mismatch"
        bindings = " ".join(f"{key.lstrip('?')}={value}" for key, value in sorted(match.bindings.items()))
        return (f"Matched strategy: {self.name}\nPattern: {self.input_pattern.structure}\n"
                f"Bindings: {bindings}\nPreconditions: {match.precondition_status.name}\n"
                f"Selected because: specificity={match.specificity}")


@dataclass
class Experience:
    initial_state: Any
    goal: Goal
    successful_plan: Any
    bindings: dict
    cost: float
    explored_nodes: int
    result: Any
    strategy_used: Any = None
    timestamp: float = 0.0
    def __post_init__(self):
        if not self.timestamp: self.timestamp = time()


def normalize_linear_expression(expression):
    from core.algebra import Add, Equation, Number, Subtract
    if isinstance(expression, Equation) and isinstance(expression.left, Subtract) and isinstance(expression.left.right, Number):
        return Equation(Add(expression.left.left, Number(-expression.left.right.value)), expression.right)
    return expression


def generalize_linear_experience(experience):
    """Controlled generalisation of ax+b=c, without program induction."""
    from core.algebra import Add, DivideBothSides, Equation, Multiply, Simplify, SubtractBothSides
    a, x, b, c = (PatternVariable(name) for name in ("a", "x", "b", "c"))
    pattern = ProblemPattern(Equation(Add(Multiply(a, x), b), c))
    target = experience.goal.target
    goal_pattern = GoalPattern("solve", target, None)
    strategy = ParameterizedMacroTransformation(
        "solve_linear_equation", pattern, goal_pattern, (a, b, c), (IsNonZero(a),),
        (StepTemplate(SubtractBothSides, (b,)), StepTemplate(Simplify),
         StepTemplate(DivideBothSides, (a,)), StepTemplate(Simplify)),
        domain=TransformationDomain.ALGEBRA,
    )
    matched = strategy.match(experience.initial_state, experience.goal)
    if matched is None: raise ValueError("the experience is not a linear equation ax+b=c")
    experience.bindings.update(matched.bindings)
    return strategy


def consolidate(strategy, experiences):
    for experience in experiences:
        match = strategy.match(experience.initial_state, experience.goal)
        if match and match.precondition_status is TruthValue.TRUE: strategy.record_success()
        else: strategy.record_failure(experience.initial_state)
    return strategy


def lift_step(entity, height):
    """Physical template: produces an Action spec, not a parallel engine."""
    from core.action import LiftAction
    from core.transformations import DeclarativeTransformationSpec
    from core.invariants import Invariant, InvariantSet
    return DeclarativeTransformationSpec(
        name="lift", domain=TransformationDomain.PHYSICS,
        parameters={"entity": entity, "height": height, "source_name": "LiftAction"},
        preserves=InvariantSet({Invariant.ENTITY_IDENTITY, Invariant.MASS}),
        changes={"y"}, compiler=lambda world: LiftAction(entity, height)._compile_executable(world))

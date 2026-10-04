"""Noyau transversal de raisonnement conditionnel pour WCE."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable


class TruthValue(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    def __bool__(self):
        raise TypeError("TruthValue est ternaire; comparez explicitement TRUE/FALSE/UNKNOWN")


def _value(value, context):
    from core.algebra import Number, Symbol
    if isinstance(value, Number): return value.value
    if isinstance(value, Symbol):
        found = context.lookup(value)
        return found if found is not None else value
    if isinstance(value, PropertyRef):
        entity = getattr(context.state, "entities", {}).get(value.entity)
        return entity.properties.get(value.property) if entity else value
    return value


class Condition:
    def evaluate(self, context) -> TruthValue: raise NotImplementedError
    def negate(self): raise NotImplementedError


@dataclass(frozen=True)
class PropertyRef:
    entity: str
    property: str


@dataclass(frozen=True)
class Equals(Condition):
    left: Any
    right: Any
    def evaluate(self, context): return context.evaluate_comparison(self, lambda a, b: a == b)
    def negate(self): return NotEquals(self.left, self.right)


@dataclass(frozen=True)
class NotEquals(Condition):
    left: Any
    right: Any
    def evaluate(self, context): return context.evaluate_comparison(self, lambda a, b: a != b)
    def negate(self): return Equals(self.left, self.right)


@dataclass(frozen=True)
class GreaterThan(Condition):
    left: Any
    right: Any
    def evaluate(self, context): return context.evaluate_comparison(self, lambda a, b: a > b)
    def negate(self): return LessThanOrEqual(self.left, self.right)


@dataclass(frozen=True)
class LessThan(Condition):
    left: Any
    right: Any
    def evaluate(self, context): return context.evaluate_comparison(self, lambda a, b: a < b)
    def negate(self): return GreaterThanOrEqual(self.left, self.right)


@dataclass(frozen=True)
class GreaterThanOrEqual(Condition):
    left: Any
    right: Any
    def evaluate(self, context): return context.evaluate_comparison(self, lambda a, b: a >= b)
    def negate(self): return LessThan(self.left, self.right)


@dataclass(frozen=True)
class LessThanOrEqual(Condition):
    left: Any
    right: Any
    def evaluate(self, context): return context.evaluate_comparison(self, lambda a, b: a <= b)
    def negate(self): return GreaterThan(self.left, self.right)


@dataclass(frozen=True)
class IsNonZero(Condition):
    value: Any
    def evaluate(self, context):
        if context.assumptions.contains(self): return TruthValue.TRUE
        if context.assumptions.contains(self.negate()): return TruthValue.FALSE
        return NotEquals(self.value, 0).evaluate(context)
    def negate(self): return Equals(self.value, 0)


@dataclass(frozen=True)
class IsPositive(Condition):
    value: Any
    def evaluate(self, context):
        if context.assumptions.contains(self): return TruthValue.TRUE
        if context.assumptions.contains(self.negate()): return TruthValue.FALSE
        return GreaterThan(self.value, 0).evaluate(context)
    def negate(self): return LessThanOrEqual(self.value, 0)


@dataclass(frozen=True)
class IsNegative(Condition):
    value: Any
    def evaluate(self, context):
        if context.assumptions.contains(self): return TruthValue.TRUE
        if context.assumptions.contains(self.negate()): return TruthValue.FALSE
        return LessThan(self.value, 0).evaluate(context)
    def negate(self): return GreaterThanOrEqual(self.value, 0)


@dataclass(frozen=True)
class HasProperty(Condition):
    entity: str
    property: str
    value: Any = None
    def evaluate(self, context):
        entity = getattr(context.state, "entities", {}).get(self.entity)
        if entity is None or self.property not in entity.properties: return TruthValue.FALSE
        return TruthValue.TRUE if self.value is None or entity.properties[self.property] == self.value else TruthValue.FALSE
    def negate(self): return NotCondition(self)


@dataclass(frozen=True)
class HasRelation(Condition):
    subject: str
    predicate: str
    object: str
    def evaluate(self, context):
        fact = Fact(self.predicate, (self.subject, self.object))
        if context.facts.contains(fact): return TruthValue.TRUE
        if context.facts.contains(fact.negate()): return TruthValue.FALSE
        return TruthValue.FALSE if hasattr(context.state, "relations") else TruthValue.UNKNOWN
    def negate(self): return NotCondition(self)


@dataclass(frozen=True)
class NotCondition(Condition):
    condition: Condition
    def evaluate(self, context):
        value = self.condition.evaluate(context)
        return {TruthValue.TRUE: TruthValue.FALSE, TruthValue.FALSE: TruthValue.TRUE}.get(value, TruthValue.UNKNOWN)
    def negate(self): return self.condition


@dataclass(frozen=True)
class Fact:
    predicate: str
    args: tuple[Any, ...]
    polarity: bool = True
    source: str = "state"
    def negate(self): return Fact(self.predicate, self.args, not self.polarity, self.source)


@dataclass(frozen=True)
class FactPattern:
    predicate: str
    args: tuple[Any, ...]
    polarity: bool = True

    def __init__(self, predicate, *args, polarity=True):
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(self, "polarity", polarity)

    def match(self, fact, bindings=None):
        bindings = dict(bindings or {})
        if self.predicate != fact.predicate or self.polarity != fact.polarity or len(self.args) != len(fact.args): return None
        for pattern, value in zip(self.args, fact.args):
            if isinstance(pattern, str) and pattern.startswith("?"):
                if pattern in bindings and bindings[pattern] != value: return None
                bindings[pattern] = value
            elif pattern != value: return None
        return bindings

    def instantiate(self, bindings, source="rule"):
        return Fact(self.predicate, tuple(bindings.get(arg, arg) for arg in self.args), self.polarity, source)


@dataclass(frozen=True)
class Contradiction:
    facts: tuple[Any, Any]
    source: str
    branch: str
    message: str


class FactBase:
    def __init__(self, facts: Iterable[Fact] = ()):
        self._facts = set()
        self.contradictions = []
        for fact in facts: self.add(fact)
    def copy(self):
        copied = FactBase(self._facts); copied.contradictions = list(self.contradictions); return copied
    def __iter__(self): return iter(self._facts)
    def __len__(self): return len(self._facts)
    def contains(self, fact): return fact in self._facts
    def query(self, predicate, *args): return [f for f in self._facts if f.predicate == predicate and (not args or f.args == args)]
    def add(self, fact, branch="current"):
        conflict = self._find_conflict(fact)
        if conflict:
            contradiction = Contradiction((conflict, fact), fact.source, branch,
                                          f"faits incompatibles: {conflict} / {fact}")
            self.contradictions.append(contradiction)
        self._facts.add(fact)
        return conflict is None
    def _find_conflict(self, fact):
        opposite = next((existing for existing in self._facts
                         if existing.predicate == fact.predicate and existing.args == fact.args
                         and existing.polarity != fact.polarity), None)
        if opposite is not None: return opposite
        if fact.predicate == "equals" and fact.polarity and len(fact.args) == 2:
            for existing in self.query("equals"):
                if existing.polarity and existing.args[0] == fact.args[0] and existing.args[1] != fact.args[1]: return existing
        if fact.predicate == "positive" and self.contains(Fact("negative", fact.args)): return Fact("negative", fact.args)
        if fact.predicate == "negative" and self.contains(Fact("positive", fact.args)): return Fact("positive", fact.args)
        return None


class AssumptionSet:
    def __init__(self, conditions: Iterable[Condition] = (), parent=None):
        self.parent = parent
        self._conditions = set(conditions)
        self.conflicts = []
    def copy(self): return AssumptionSet(self._conditions, self.parent)
    def __iter__(self):
        inherited = list(self.parent) if self.parent else []
        return iter(inherited + list(self._conditions))
    def __len__(self): return len(set(iter(self)))
    def contains(self, condition): return condition in set(iter(self))
    def add(self, condition):
        conflicting = next((existing for existing in self if existing.negate() == condition or condition.negate() == existing), None)
        if conflicting is not None:
            contradiction = Contradiction((conflicting, condition), "assumption", "current", "hypothèses incompatibles")
            self.conflicts.append(contradiction); return False
        self._conditions.add(condition); return True


@dataclass
class ReasoningContext:
    state: Any
    facts: FactBase
    assumptions: AssumptionSet

    def lookup(self, symbol):
        name = getattr(symbol, "name", symbol)
        for condition in self.assumptions:
            if isinstance(condition, Equals) and getattr(condition.left, "name", condition.left) == name:
                return _value(condition.right, self)
        matches = self.facts.query("equals")
        for fact in matches:
            if getattr(fact.args[0], "name", fact.args[0]) == name: return _value(fact.args[1], self)
        return None

    def evaluate_comparison(self, condition, operation):
        if self.assumptions.contains(condition): return TruthValue.TRUE
        if self.assumptions.contains(condition.negate()): return TruthValue.FALSE
        left, right = _value(condition.left, self), _value(condition.right, self)
        from core.algebra import Expression
        if isinstance(left, Expression) or isinstance(right, Expression): return TruthValue.UNKNOWN
        try: return TruthValue.TRUE if operation(left, right) else TruthValue.FALSE
        except (TypeError, ValueError): return TruthValue.UNKNOWN


@dataclass(frozen=True)
class Goal:
    predicate: str
    target: Any = None
    condition: Condition = None
    def is_satisfied(self, context):
        if self.condition is not None: return self.condition.evaluate(context) is TruthValue.TRUE
        if self.predicate == "solve":
            from core.algebra import Equation, Symbol
            expression = getattr(context.state, "expression", None)
            return isinstance(expression, Equation) and expression.left == Symbol(str(self.target))
        args = self.target if isinstance(self.target, tuple) else ((self.target,) if self.target is not None else ())
        return bool(context.facts.query(self.predicate, *args))


@dataclass(frozen=True)
class Rule:
    name: str
    premises: tuple[Condition, ...]
    conclusion: Fact | Callable | None = None
    transformation: Any = None
    goal_predicate: str = None

    def status(self, context):
        values = [premise.evaluate(context) for premise in self.premises]
        if any(value is TruthValue.FALSE for value in values): return TruthValue.FALSE
        if any(value is TruthValue.UNKNOWN for value in values): return TruthValue.UNKNOWN
        return TruthValue.TRUE


@dataclass
class ReasoningStep:
    facts_before: tuple[Fact, ...]
    rule: Rule
    premises: tuple[Condition, ...]
    assumptions: tuple[Condition, ...]
    transformation: Any = None
    result: Any = None
    facts_after: tuple[Fact, ...] = ()


class PreconditionError(ValueError): pass
class UnknownPrecondition(PreconditionError): pass


class RuleEngine:
    def __init__(self, state, facts=None, assumptions=None, goal=None):
        self.state = state
        self.facts = facts or FactBase()
        self.assumptions = assumptions or AssumptionSet()
        self.goal = goal
        self.rules = []
        self.trace = []
        self.refresh_facts()
        graph = getattr(state, "transform_graph", None)
        if graph:
            graph.root.facts = self.facts.copy(); graph.root.assumptions = self.assumptions.copy()

    @property
    def context(self): return ReasoningContext(self.state, self.facts, self.assumptions)

    def refresh_facts(self):
        for relation in getattr(self.state, "relations", ()):
            self.facts.add(Fact(relation.predicate, (relation.subject, relation.object)))
        expression = getattr(self.state, "expression", None)
        from core.algebra import Equation, Symbol
        if isinstance(expression, Equation) and isinstance(expression.left, Symbol):
            self.facts.add(Fact("equals", (expression.left, expression.right)))

    def add_rule(self, rule): self.rules.append(rule)
    def applicable_rules(self): return [rule for rule in self.rules if rule.status(self.context) is TruthValue.TRUE]

    def match_rule(self, rule):
        """Unification plate des FactPattern d'une règle, sans termes récursifs."""
        bindings_list = [dict()]
        for premise in rule.premises:
            if not isinstance(premise, FactPattern): continue
            next_bindings = []
            for bindings in bindings_list:
                for fact in self.facts.query(premise.predicate):
                    matched = premise.match(fact, bindings)
                    if matched is not None: next_bindings.append(matched)
            bindings_list = next_bindings
        return bindings_list

    def fire_generic(self, rule):
        produced = []
        for bindings in self.match_rule(rule):
            conclusion = rule.conclusion.instantiate(bindings) if isinstance(rule.conclusion, FactPattern) else rule.conclusion
            if conclusion: self.facts.add(conclusion); produced.append(conclusion)
        return produced

    def apply_transformation(self, spec):
        status = spec.applicability(self.context)
        if status is TruthValue.FALSE: raise PreconditionError(self.explain_preconditions(spec, status))
        if status is TruthValue.UNKNOWN: raise UnknownPrecondition(self.explain_preconditions(spec, status))
        result = self.state.apply(spec) if hasattr(self.state, "apply") else apply_world(self.state, spec.compile(self.state))
        self.refresh_facts()
        node = self.state.transform_graph.current
        node.facts = self.facts.copy(); node.assumptions = self.assumptions.copy()
        return result

    def fire(self, rule):
        status = rule.status(self.context)
        if status is not TruthValue.TRUE: return status
        before = tuple(self.facts)
        transformation = rule.transformation(self.context) if callable(rule.transformation) else rule.transformation
        result = self.apply_transformation(transformation) if transformation else None
        conclusion = rule.conclusion(self.context) if callable(rule.conclusion) else rule.conclusion
        if conclusion: self.facts.add(conclusion, self.state.transform_graph.current.name)
        step = ReasoningStep(before, rule, rule.premises, tuple(self.assumptions), transformation,
                             getattr(self.state, "expression", result), tuple(self.facts))
        self.trace.append(step)
        return step

    def run(self):
        return [self.fire(rule) for rule in self.rules
                if (not self.goal or not rule.goal_predicate or rule.goal_predicate == self.goal.predicate)
                and rule.status(self.context) is TruthValue.TRUE]

    def branch(self, parent, condition):
        graph = self.state.transform_graph
        if isinstance(parent, str): parent = graph.get_node_by_name(parent)
        if isinstance(parent.configuration, dict) and hasattr(self.state, "entities"):
            for entity_name, properties in parent.configuration.items():
                if entity_name in self.state.entities:
                    self.state.entities[entity_name].properties.update(properties)
        graph.branch_from(parent)
        inherited = parent.assumptions.copy() if parent.assumptions else AssumptionSet()
        inherited.add(condition)
        from core.transformations import DeclarativeTransformationSpec, TransformationDomain
        from core.invariants import InvariantSet
        spec = DeclarativeTransformationSpec(name=f"assume({condition})", domain=TransformationDomain.GENERIC,
                                             parameters={"source_name": "assumption"}, preserves=InvariantSet(set()))
        node = graph.apply_transformation(spec, config=parent.configuration)
        node.assumptions = inherited; node.facts = parent.facts.copy() if parent.facts else self.facts.copy()
        self.assumptions = inherited; self.facts = node.facts.copy()
        return node

    def explain_preconditions(self, spec, status=None):
        required = ", ".join(str(c) for c in spec.preconditions) or "aucune"
        resolved_status = spec.applicability(self.context) if status is None else status
        return f"Transformation: {spec.name}\nCondition: {required}\nStatus: {resolved_status.name}"

    def explain(self, step=None):
        step = step or (self.trace[-1] if self.trace else None)
        if step is None: return "Aucun raisonnement enregistré."
        premises = ", ".join(str(p) for p in step.premises)
        transformation = step.transformation.name if step.transformation else "none"
        return (f"CAUSE: rule {step.rule.name}\nCONDITION: {premises}\n"
                f"TRANSFORMATION: {transformation}\nRESULT: {step.result}")


def apply_world(state, transformation):
    from core.transformation import apply
    return apply(state, transformation)

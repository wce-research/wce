"""Bounded search over compositions of TransformationSpec, directed by a Goal."""
from __future__ import annotations

import copy
import heapq
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from typing import Any, Callable, Iterable

from core.reasoning import (
    AssumptionSet, FactBase, FactPattern, Goal, ReasoningContext, Rule, RuleEngine,
    TruthValue, UnknownPrecondition, PreconditionError,
)
from core.transformations import MacroTransformation, TransformationDomain, TransformationSpec
from core.invariants import InvariantSet


class SearchStrategy(Enum):
    BFS = "bfs"
    UNIFORM_COST = "uniform_cost"
    DFS = "dfs"


class UnknownPolicy(Enum):
    SKIP = "skip"
    ASSUME = "assume"


class FactConclusionSpec(TransformationSpec):
    def __init__(self, fact, rule):
        object.__setattr__(self, "name", f"infer({fact.predicate})")
        object.__setattr__(self, "domain", TransformationDomain.GENERIC)
        object.__setattr__(self, "parameters", {"fact": fact, "source_name": rule.name})
        object.__setattr__(self, "invertible", False)
        object.__setattr__(self, "preserves", InvariantSet(set()))
        object.__setattr__(self, "changes", {"facts"})
        object.__setattr__(self, "cost", 1.0)
        object.__setattr__(self, "preconditions", ())
        object.__setattr__(self, "postconditions", (fact,))
        object.__setattr__(self, "fact", fact)
    def apply(self, obj, *args, **kwargs): return obj
    def compile(self, state): raise TypeError("une conclusion factuelle modifie FactBase, pas le World")


@dataclass(frozen=True)
class StateSignature:
    data: tuple

    @classmethod
    def from_state(cls, state, assumptions=(), facts=()):
        if hasattr(state, "expression"):
            state_data = ("math", state.expression)
        else:
            entities = tuple(sorted((name, tuple(sorted((key, _freeze(value)) for key, value in entity.properties.items())))
                                    for name, entity in getattr(state, "entities", {}).items()))
            relations = tuple(sorted((r.subject, r.predicate, r.object) for r in getattr(state, "relations", ())))
            state_data = ("world", entities, relations)
        assumption_data = tuple(sorted(repr(item) for item in assumptions))
        fact_data = tuple(sorted((f.predicate, tuple(map(repr, f.args)), f.polarity) for f in facts))
        return cls((state_data, assumption_data, fact_data))


def _freeze(value):
    if isinstance(value, dict): return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple, set)): return tuple(_freeze(item) for item in value)
    try: hash(value); return value
    except TypeError: return repr(value)


@dataclass
class SearchNode:
    state: Any
    parent: SearchNode | None = None
    transformation: TransformationSpec | None = None
    rule: Rule | None = None
    assumptions: AssumptionSet = field(default_factory=AssumptionSet)
    depth: int = 0
    cost: float = 0.0
    facts: FactBase = field(default_factory=FactBase)
    goal_status: bool = False
    transform_node: Any = None

    def path_nodes(self):
        nodes, current = [], self
        while current.parent is not None:
            nodes.append(current); current = current.parent
        return list(reversed(nodes))


@dataclass
class SearchMetrics:
    explored_nodes: int = 0
    generated_nodes: int = 0
    pruned_nodes: int = 0
    max_depth_reached: int = 0
    total_branching: int = 0
    expanded_nodes: int = 0

    @property
    def branching_factor(self):
        return self.total_branching / self.expanded_nodes if self.expanded_nodes else 0.0


@dataclass
class SearchResult:
    success: bool
    final_state: Any = None
    path: list[SearchNode] = field(default_factory=list)
    transformations: list[TransformationSpec] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    assumptions: AssumptionSet = field(default_factory=AssumptionSet)
    cost: float = 0.0
    explored_nodes: int = 0
    reason_if_failed: str = ""
    metrics: SearchMetrics = field(default_factory=SearchMetrics)
    used_strategy: bool = False


class StrategyLibrary:
    def __init__(self): self._strategies = {}; self._parameterized = []
    def _key(self, goal): return (goal.predicate, str(goal.target), repr(goal.condition))
    def store(self, goal, macro): self._strategies[self._key(goal)] = macro
    def retrieve(self, goal): return self._strategies.get(self._key(goal))
    def store_parameterized(self, strategy):
        if strategy not in self._parameterized: self._parameterized.append(strategy)
    def match(self, state, goal, assumptions=None):
        matches = [match for strategy in self._parameterized
                   if (match := strategy.match(state, goal, assumptions)) is not None]
        matches.sort(key=lambda item: (item.precondition_status is TruthValue.TRUE, item.specificity), reverse=True)
        return matches
    def best_match(self, state, goal, assumptions=None):
        return next((match for match in self.match(state, goal, assumptions)
                     if match.precondition_status is TruthValue.TRUE), None)
    def __len__(self): return len(self._strategies) + len(self._parameterized)
    def inspect(self):
        result = dict(self._strategies)
        result.update({("parameterized", strategy.name): strategy for strategy in self._parameterized})
        return result


class TransformationPlanner:
    def __init__(self, transformations=(), rules=(), *, strategy=SearchStrategy.BFS,
                 unknown_policy=UnknownPolicy.SKIP, max_depth=8, max_nodes=500,
                 max_cost=float("inf"), strategy_library=None):
        self.transformations = list(transformations)
        self.rules = list(rules)
        self.strategy = SearchStrategy(strategy) if isinstance(strategy, str) else strategy
        self.unknown_policy = UnknownPolicy(unknown_policy) if isinstance(unknown_policy, str) else unknown_policy
        self.max_depth, self.max_nodes, self.max_cost = max_depth, max_nodes, max_cost
        self.strategy_library = strategy_library if strategy_library is not None else StrategyLibrary()

    def _candidates(self, node, goal):
        candidates = []
        for item in self.transformations:
            produced = item(node.state, goal) if callable(item) and not isinstance(item, TransformationSpec) else item
            candidates.extend(produced if isinstance(produced, (list, tuple, set)) else [produced])
        for rule in self.rules:
            if rule.goal_predicate and rule.goal_predicate != goal.predicate: continue
            if any(isinstance(p, FactPattern) for p in rule.premises):
                engine = RuleEngine(node.state, facts=node.facts.copy(), assumptions=node.assumptions.copy())
                if isinstance(rule.conclusion, FactPattern):
                    for bindings in engine.match_rule(rule):
                        candidates.append((FactConclusionSpec(rule.conclusion.instantiate(bindings), rule), rule))
            elif all(hasattr(p, "evaluate") for p in rule.premises):
                context = ReasoningContext(node.state, node.facts, node.assumptions)
                if rule.status(context) is TruthValue.TRUE:
                    transformation = rule.transformation(context) if callable(rule.transformation) else rule.transformation
                    if transformation: candidates.append((transformation, rule))
        return candidates

    def search(self, initial_state, goal, *, assumptions=None, facts=None):
        parameterized_match = self.strategy_library.best_match(initial_state, goal, assumptions)
        if parameterized_match is not None:
            learned = parameterized_match.strategy
            macro = learned.instantiate(parameterized_match)
            state = copy.deepcopy(initial_state)
            engine = RuleEngine(state, facts=(facts or FactBase()).copy(), assumptions=(assumptions or AssumptionSet()).copy(), goal=goal)
            try:
                engine.apply_transformation(macro)
                if goal.is_satisfied(engine.context):
                    learned.record_success()
                    node = SearchNode(state, transformation=macro, assumptions=engine.assumptions,
                                      depth=1, cost=macro.cost, facts=engine.facts, goal_status=True,
                                      transform_node=state.transform_graph.current)
                    metrics = SearchMetrics(explored_nodes=1, generated_nodes=1, max_depth_reached=1)
                    return SearchResult(True, state, [node], [macro], [], engine.assumptions,
                                        macro.cost, 1, metrics=metrics, used_strategy=True)
                learned.record_failure(initial_state)
            except (ValueError, TypeError): learned.record_failure(initial_state)
        macro = self.strategy_library.retrieve(goal)
        if macro is not None:
            state = copy.deepcopy(initial_state)
            engine = RuleEngine(state, facts=(facts or FactBase()).copy(), assumptions=(assumptions or AssumptionSet()).copy(), goal=goal)
            try:
                engine.apply_transformation(macro)
                if goal.is_satisfied(engine.context):
                    node = SearchNode(state, transformation=macro, assumptions=engine.assumptions,
                                      depth=1, cost=macro.cost, facts=engine.facts, goal_status=True,
                                      transform_node=state.transform_graph.current)
                    metrics = SearchMetrics(explored_nodes=1, generated_nodes=1, max_depth_reached=1)
                    return SearchResult(True, state, [node], [macro], [], engine.assumptions,
                                        macro.cost, 1, metrics=metrics, used_strategy=True)
            except (ValueError, TypeError):
                pass

        root_state = copy.deepcopy(initial_state)
        root_engine = RuleEngine(root_state, facts=(facts or FactBase()).copy(),
                                 assumptions=(assumptions or AssumptionSet()).copy(), goal=goal)
        root = SearchNode(root_state, assumptions=root_engine.assumptions, facts=root_engine.facts,
                          goal_status=goal.is_satisfied(root_engine.context),
                          transform_node=root_state.transform_graph.current)
        metrics = SearchMetrics(generated_nodes=1)
        if root.goal_status: return self._result(root, metrics)
        frontier, serial = self._frontier(root), count()
        if self.strategy is SearchStrategy.UNIFORM_COST:
            frontier = [(0.0, next(serial), root)]
        seen = {StateSignature.from_state(root.state, root.assumptions, root.facts)}

        while frontier and metrics.explored_nodes < self.max_nodes and metrics.generated_nodes < self.max_nodes:
            node = self._pop(frontier)
            metrics.explored_nodes += 1
            if node.goal_status: return self._result(node, metrics)
            metrics.max_depth_reached = max(metrics.max_depth_reached, node.depth)
            if node.depth >= self.max_depth: metrics.pruned_nodes += 1; continue
            candidates = self._candidates(node, goal)
            metrics.expanded_nodes += 1; metrics.total_branching += len(candidates)
            for candidate in candidates:
                spec, rule = candidate if isinstance(candidate, tuple) else (candidate, None)
                child = self._successor(node, spec, rule, goal, metrics)
                if child is None: continue
                signature = StateSignature.from_state(child.state, child.assumptions, child.facts)
                if signature in seen: metrics.pruned_nodes += 1; continue
                seen.add(signature); metrics.generated_nodes += 1
                if child.goal_status and self.strategy is not SearchStrategy.UNIFORM_COST: return self._result(child, metrics)
                if self.strategy is SearchStrategy.UNIFORM_COST:
                    heapq.heappush(frontier, (child.cost, next(serial), child))
                elif self.strategy is SearchStrategy.DFS: frontier.append(child)
                else: frontier.append(child)
        reason = ("node limit reached"
                  if metrics.explored_nodes >= self.max_nodes or metrics.generated_nodes >= self.max_nodes
                  else "goal not reachable within limits")
        return SearchResult(False, explored_nodes=metrics.explored_nodes, reason_if_failed=reason, metrics=metrics)

    def _successor(self, node, spec, rule, goal, metrics):
        if node.cost + spec.cost > self.max_cost: metrics.pruned_nodes += 1; return None
        state = copy.deepcopy(node.state); assumptions = node.assumptions.copy(); facts = node.facts.copy()
        if isinstance(spec, FactConclusionSpec):
            facts.add(spec.fact)
            context = ReasoningContext(state, facts, assumptions)
            return SearchNode(state, node, spec, rule, assumptions, node.depth + 1,
                              node.cost + spec.cost, facts, goal.is_satisfied(context),
                              state.transform_graph.current)
        engine = RuleEngine(state, facts=facts, assumptions=assumptions, goal=goal)
        status = spec.applicability(engine.context)
        if status is TruthValue.FALSE: metrics.pruned_nodes += 1; return None
        if status is TruthValue.UNKNOWN:
            if self.unknown_policy is UnknownPolicy.SKIP: metrics.pruned_nodes += 1; return None
            for condition in spec.preconditions: assumptions.add(condition)
            engine.assumptions = assumptions
        if any(condition.evaluate(engine.context) is TruthValue.FALSE for condition in assumptions):
            metrics.pruned_nodes += 1; return None
        try: engine.apply_transformation(spec)
        except (ValueError, TypeError, PreconditionError, UnknownPrecondition):
            metrics.pruned_nodes += 1; return None
        return SearchNode(state, node, spec, rule, engine.assumptions, node.depth + 1,
                          node.cost + spec.cost, engine.facts,
                          goal.is_satisfied(engine.context), state.transform_graph.current)

    def _frontier(self, root): return [root] if self.strategy is SearchStrategy.DFS else deque([root])
    def _pop(self, frontier):
        if self.strategy is SearchStrategy.UNIFORM_COST: return heapq.heappop(frontier)[2]
        return frontier.pop() if self.strategy is SearchStrategy.DFS else frontier.popleft()
    def _result(self, node, metrics):
        path = node.path_nodes() if node.parent else ([node] if node.transformation else [])
        return SearchResult(True, node.state, path, [n.transformation for n in path if n.transformation],
                            [n.rule for n in path if n.rule], node.assumptions, node.cost,
                            metrics.explored_nodes, metrics=metrics)

    def promote_to_macro(self, result, name):
        if not result.success or not result.transformations: raise ValueError("a successful non-empty plan is required")
        domain = result.transformations[0].domain
        return MacroTransformation(name=name, steps=tuple(result.transformations), domain=domain,
                                   description="stratégie promue depuis un plan")

    def remember(self, goal, macro): self.strategy_library.store(goal, macro)

    def explain(self, result, goal=None):
        if not result.success: return f"Goal failed: {result.reason_if_failed}\nNodes explored: {result.explored_nodes}"
        lines = [f"Goal: {goal.predicate if goal else 'satisfied'}", "Chosen path:"]
        for index, spec in enumerate(result.transformations, 1):
            invariants = ", ".join(sorted(i.name for i in spec.preserves.invariants))
            preconditions = ", ".join(map(str, spec.preconditions)) or "none"
            lines.append(f"{index}. {spec.name}\n   preconditions: {preconditions}\n   preserves: {invariants}")
        lines += [f"Final: {getattr(result.final_state, 'expression', result.final_state)}",
                  f"Nodes explored: {result.explored_nodes}", f"Cost: {result.cost}"]
        return "\n".join(lines)


def algebra_candidate_generator(state, goal):
    """Generates the primitive operations suggested by the constants of the AST."""
    from core.algebra import (Add, AddBothSides, DivideBothSides, Multiply, MultiplyBothSides,
                              Number, Simplify, SubtractBothSides)
    expression = state.expression
    numbers = []
    def collect(expr):
        if isinstance(expr, Number): numbers.append(expr.value)
        for attr in ("left", "right"):
            child = getattr(expr, attr, None)
            if child is not None: collect(child)
    collect(expression)
    preferred = []
    left = getattr(expression, "left", None)
    if isinstance(left, Add) and isinstance(left.right, Number): preferred.append(SubtractBothSides(left.right.value))
    preferred.append(Simplify())
    if isinstance(left, Multiply) and isinstance(left.left, Number) and left.left.value != 0: preferred.append(DivideBothSides(left.left.value))
    unique = []
    for spec in preferred + [SubtractBothSides(n) for n in numbers] + [AddBothSides(n) for n in numbers] + \
            [DivideBothSides(n) for n in numbers if n != 0] + [MultiplyBothSides(n) for n in numbers if n not in (0, 1)]:
        key = (type(spec), repr(spec.parameters))
        if key not in {(type(item), repr(item.parameters)) for item in unique}: unique.append(spec)
    return unique


def lift_candidate_generator(target_height):
    def candidates(state, goal):
        from core.action import LiftAction, ImpulseAction
        entity = goal.target[0] if isinstance(goal.target, tuple) else str(goal.target)
        return [LiftAction(entity, target_height).to_spec(state), ImpulseAction(entity, target_height).to_spec(state)]
    return candidates

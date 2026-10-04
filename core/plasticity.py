"""Plasticité gouvernée : contrôleur explicite, configurations candidates, boucle d'apprentissage.

Rien ici n'est quantique : ConfigurationAmplitude est une représentation classique
de plusieurs configurations candidates pondérées, avant sélection ("collapse").
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from core.invariants import Invariant, InvariantSet
from core.model_state import (
    INPUT, PARAM_NAMES, ActivateModule, AdaptWeights, AddModule, ClearFastWeights, CommitFastWeights,
    CreateConnection, GrowModule, ModelState, ModelTransformationSpec, ModuleExists, PruneUnits,
)
from core.reasoning import Condition, Goal, TruthValue


# ---------------------------------------------------------------- tâches

def all_inputs(dim=10):
    idx = torch.arange(2 ** dim)
    return ((idx.unsqueeze(1) >> torch.arange(dim)) & 1).float()


def task_a_labels(x):
    """A : (b0 xor b1) or b2."""
    return (((x[:, 0] != x[:, 1]) | (x[:, 2] == 1))).long()


def task_b_labels(x):
    """B : majorité(b5..b9) xor b3 — dépend d'autres bits, en conflit avec A."""
    return ((x[:, 5:10].sum(dim=1) >= 3).long() ^ x[:, 3].long())


def task_split(labeler, dim=10, seed=0, train_fraction=0.7):
    x = all_inputs(dim); y = labeler(x)
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    cut = int(train_fraction * len(x))
    return x[perm[:cut]], y[perm[:cut]], x[perm[cut:]], y[perm[cut:]]


# ---------------------------------------------------------------- contexte

@dataclass(frozen=True)
class ContextExists(Condition):
    context: str
    def evaluate(self, context):
        return TruthValue.TRUE if self.context in context.state.metadata.get("contexts", {}) else TruthValue.FALSE
    def __str__(self): return f"context({self.context})"


@dataclass(frozen=True)
class LoadContext(ModelTransformationSpec):
    """Routage contextuel : active set + connectivité + fast weights ; slow weights intacts."""
    name: str = "load_context"
    context: str = ""
    invertible: bool = False
    changes: frozenset = frozenset({"active_modules", "connectivity", "fast_weights"})
    preserves: InvariantSet = field(default_factory=lambda: InvariantSet({Invariant.SLOW_WEIGHTS, Invariant.MODULE_SET}))
    cost: float = 0.1
    def __post_init__(self): object.__setattr__(self, "preconditions", (ContextExists(self.context),))
    def execute(self, model):
        ctx = model.metadata["contexts"][self.context]
        for name, module in model.modules.items(): module.active = name in ctx["active"]
        model.connectivity = {k: copy.deepcopy(c) for k, c in ctx["connectivity"].items()
                              if all(n == INPUT or n in model.modules for n in k)}
        model.fast_parameters = {m: {k: v.clone() for k, v in d.items()}
                                 for m, d in ctx["fast"].items() if m in model.modules}
        model.metadata["current_context"] = self.context
        return {"context": self.context}


def save_context(model, name):
    """Mémorise une configuration de routage (pas une modification du modèle)."""
    model.metadata.setdefault("contexts", {})[name] = {
        "active": set(model.active_modules), "connectivity": copy.deepcopy(model.connectivity),
        "fast": {m: {k: v.clone() for k, v in d.items()} for m, d in model.fast_parameters.items()}}
    model.metadata["current_context"] = name


# ---------------------------------------------------------------- construction

def build_base_model(input_dim=10, width=24, seed=0):
    """enc -> expert_a -> head actifs ; expert_b dormant (non connecté)."""
    model = ModelState(input_dim, max_capacity=256)
    for spec in (
        AddModule(module="enc", input_dim=input_dim, output_dim=width, hidden=48, specialization="shared", seed=seed),
        AddModule(module="expert_a", input_dim=width, output_dim=width, hidden=24, specialization="task_a", seed=seed + 1),
        AddModule(module="head", input_dim=width, output_dim=2, hidden=16, activation=False, specialization="shared", seed=seed + 2),
        AddModule(module="expert_b", input_dim=input_dim, output_dim=width, hidden=24, specialization="task_b",
                  active=False, seed=seed + 3),
        CreateConnection(source=INPUT, target="enc"),
        CreateConnection(source="enc", target="expert_a"),
        CreateConnection(source="expert_a", target="head"),
    ):
        spec.apply(model)
    model.counters["growths"] = 0
    return model


def pretrain(model, x, y, steps=300, lr=0.02, seed=0):
    """Apprentissage initial global (le seul non local, explicitement déclaré)."""
    return model.apply(AdaptWeights(name="pretrain", modules=tuple(sorted(model.active_modules)),
                                    inputs=x, targets=y, steps=steps, lr=lr, fast=False, local=False,
                                    seed=seed, provenance="initial_training"))


# ---------------------------------------------------------------- hypothèses

@dataclass
class ConfigurationAmplitude:
    """Configuration candidate pondérée (représentation classique, 'quantum-inspired')."""
    label: str
    state: ModelState
    weight: float = 1.0
    phase: Optional[float] = None
    score: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    status: str = "candidate"

    @property
    def probability(self): return self.weight


class ModelHypothesisSet:
    """Ensemble pondéré de configurations candidates (calcul classique, pas quantique).

    shared=True : chaque branche = base partagée + delta copy-on-write (core.branching).
    shared=False : copie complète (mode historique, conservé pour vérification).
    """

    def __init__(self, base: ModelState, shared=True):
        self.base = base
        self.shared = shared
        self.branches: List[ConfigurationAmplitude] = []
        self.archive: List[ConfigurationAmplitude] = []
        self.evaluations = 0

    def branch(self, label, specs: List[ModelTransformationSpec], weight=1.0):
        from core.branching import shared_branch
        state = shared_branch(self.base) if self.shared else copy.deepcopy(self.base)
        branch = ConfigurationAmplitude(label, state, weight)
        try:
            for spec in specs: branch.details.setdefault("effects", []).append(state.apply(spec))
            branch.details["specs"] = [s.name for s in specs]
        except ValueError as error:
            branch.status, branch.weight, branch.details["error"] = "invalid", 0.0, str(error)
        self.branches.append(branch)
        return branch

    def valid(self): return [b for b in self.branches if b.status != "invalid"]

    def evaluate(self, scorer: Callable[[ConfigurationAmplitude], float]):
        for b in self.valid(): b.score = scorer(b); self.evaluations += 1
        return self.normalize()

    def reweight(self, evidence: Dict[str, float]):
        """Multiplie chaque poids par une évidence positive puis renormalise (règle de Bayes)."""
        for b in self.valid(): b.weight *= max(evidence.get(b.label, 0.0), 0.0)
        return self._renormalize()

    def interfere(self, other: Dict[str, float]):
        """Combinaison abstraite de deux distributions (produit normalisé).
        Nom inspiré du quantique ; c'est une simple opération classique sur des poids positifs."""
        return self.reweight(other)

    def _renormalize(self):
        total = sum(b.weight for b in self.valid())
        for b in self.valid(): b.weight = b.weight / total if total else 1.0 / len(self.valid())
        return self

    def memory(self):
        from core.branching import branch_memory
        reports = [branch_memory(b.state, self.base) for b in self.valid()]
        total = sum(r["total_bytes"] for r in reports); owned = sum(r["owned_bytes"] for r in reports)
        return {"branches": len(reports), "full_copy_bytes": total, "owned_bytes": owned,
                "shared_ratio": 1 - owned / total if total else 0.0}

    @staticmethod
    def compatible(a: ConfigurationAmplitude, b: ConfigurationAmplitude, weight_tolerance=0.05, score_tolerance=0.02):
        """Fusion permise seulement si structure, routes, activations, deltas et scores concordent."""
        reasons = []
        sa, sb = a.state, b.state
        if sorted(sa.modules) != sorted(sb.modules): reasons.append("modules différents")
        elif any(sa.modules[n].slow[k].shape != sb.modules[n].slow[k].shape for n in sa.modules for k in PARAM_NAMES):
            reasons.append("formes différentes")
        else:
            for n in sa.modules:
                for k in PARAM_NAMES:
                    x, y = sa.effective_weight(n, k), sb.effective_weight(n, k)
                    if float((x - y).norm()) > weight_tolerance * max(1.0, float(x.norm())):
                        reasons.append(f"deltas éloignés ({n}.{k})"); break
        if sa.active_modules != sb.active_modules: reasons.append("modules actifs différents")
        if set(sa.connectivity) != set(sb.connectivity): reasons.append("connexions différentes")
        if a.score is not None and b.score is not None and abs(a.score - b.score) > score_tolerance:
            reasons.append("performances différentes")
        return not reasons, reasons

    def merge_branches(self, a, b, label="merged", **tolerances):
        ok, reasons = self.compatible(a, b, **tolerances)
        if not ok: raise ValueError("fusion refusée : " + ", ".join(reasons))
        merged = self.merge(a, b, label)
        merged.score = max(a.score or 0.0, b.score or 0.0)
        for x in (a, b): x.status = "merged"
        self.branches = [br for br in self.branches if br.status != "merged"]
        return merged

    def normalize(self):
        positive = [max(b.score or 0.0, 0.0) for b in self.valid()]
        total = sum(positive)
        for b, p in zip(self.valid(), positive):
            b.weight = p / total if total else 1.0 / len(self.valid())
        return self

    def distribution(self): return {b.label: round(b.weight, 3) for b in self.branches}

    def select(self, keep="archive"):
        """'Collapse' déterministe : la branche de plus haut score devient active."""
        candidates = self.valid()
        if not candidates: raise ValueError("aucune configuration valide")
        best = max(candidates, key=lambda b: (b.score, -b.state.active_parameters()))
        best.status = "selected"
        for b in self.branches:
            if b is best: continue
            b.status = "archived" if keep == "archive" else "rejected"
            if keep == "archive": self.archive.append(b)
        if keep != "archive": self.branches = [best]
        return best

    def merge(self, a: ConfigurationAmplitude, b: ConfigurationAmplitude, label="merged"):
        """Moyenne pondérée des slow weights de deux branches structurellement identiques."""
        if sorted(a.state.modules) != sorted(b.state.modules): raise ValueError("structures incompatibles")
        from core.branching import shared_branch
        merged = shared_branch(a.state) if self.shared else copy.deepcopy(a.state)
        total = (a.weight + b.weight) or 1.0
        for name, module in merged.modules.items():
            other = b.state.modules[name]
            for k in PARAM_NAMES:
                if module.slow[k].shape != other.slow[k].shape: raise ValueError(f"formes incompatibles: {name}.{k}")
                module.slow[k] = (a.weight * module.slow[k] + b.weight * other.slow[k]) / total
        amp = ConfigurationAmplitude(label, merged, a.weight + b.weight)
        self.branches.append(amp)
        return amp

    def reject(self, threshold):
        for b in self.valid():
            if b.weight < threshold: b.status = "rejected"
        self.branches = [b for b in self.branches if b.status not in ("rejected", "invalid")]
        return self


# ---------------------------------------------------------------- contrôleur

@dataclass
class CandidatePlan:
    label: str
    description: str
    specs: List[ModelTransformationSpec]
    adapted_modules: Tuple[str, ...]


class PlasticityController:
    """Contrôleur explicite (non neuronal) : faiblesse -> transformations candidates."""

    def __init__(self, threshold=0.85, interference_penalty=0.5, complexity_penalty=0.1,
                 prune_tolerance=0.02):
        self.threshold = threshold
        self.interference_penalty = interference_penalty
        self.complexity_penalty = complexity_penalty
        self.prune_tolerance = prune_tolerance

    # entrées : représentation, état, contexte WCE, but, signal de performance
    def detect_weakness(self, model, goal: Goal, performance: float):
        threshold = getattr(goal.condition, "right", self.threshold) if goal and goal.condition else self.threshold
        return performance < threshold

    def region_for(self, model, domain):
        """Région de plasticité : modules spécialisés pour le domaine de l'entrée."""
        return tuple(sorted(n for n, m in model.modules.items() if m.specialization == domain))

    def protected_modules(self, model, domain):
        return {n for n, m in model.modules.items() if m.specialization not in (domain,)}

    def propose(self, model, domain, x, y, goal=None, performance=0.0, seed=0) -> List[CandidatePlan]:
        if goal is not None and not self.detect_weakness(model, goal, performance): return []
        plans = []
        shared_path = tuple(n for n in ("expert_a", "head") if n in model.modules)
        plans.append(CandidatePlan("fast_weights", "fast-weight adaptation of existing path",
                                   [AdaptWeights(modules=shared_path, inputs=x, targets=y, seed=seed)], shared_path))
        region = self.region_for(model, domain)
        if region:
            module = region[0]
            specs = []
            if not model.modules[module].active: specs.append(ActivateModule(module=module))
            specs += [CreateConnection(source=INPUT, target=module),
                      CreateConnection(source=module, target=model.metadata["output"]),
                      GrowModule(module=module, units=16, seed=seed),
                      AdaptWeights(modules=(module,), inputs=x, targets=y, seed=seed)]
            plans.append(CandidatePlan("activate_module", f"activate module {module} + grow + local fast weights",
                                       specs, (module,)))
        grow_target = "expert_a"
        plans.append(CandidatePlan("grow_module", f"grow {grow_target} + fast weights",
                                   [GrowModule(module=grow_target, units=16, seed=seed),
                                    AdaptWeights(modules=(grow_target,), inputs=x, targets=y, seed=seed)],
                                   (grow_target,)))
        return plans

    def score(self, branch, plan, domain, x_val, y_val):
        state = branch.state
        accuracy = state.accuracy(x_val, y_val)
        protected = self.protected_modules(state, domain)
        touched = sum(state.modules[m].parameter_count for m in plan.adapted_modules if m in protected)
        interference = touched / max(1, sum(state.modules[m].parameter_count for m in protected if m in state.modules))
        complexity = state.active_parameters() / max(1, state.total_parameters())
        score = accuracy - self.interference_penalty * interference - self.complexity_penalty * complexity
        branch.details.update(accuracy=accuracy, interference=round(interference, 3),
                              complexity=round(complexity, 3), active_parameters=state.active_parameters())
        return score

    def consolidate(self, model, domain):
        """Seules les régions propres au domaine sont figées en slow weights."""
        exclusive = tuple(m for m in model.fast_parameters if m not in self.protected_modules(model, domain))
        if exclusive: model.apply(CommitFastWeights(modules=exclusive))
        return exclusive

    def prune(self, model, domain, x_val, y_val, x_train=None, y_train=None, step=8, recovery_steps=100):
        """Réduit les modules du domaine tant que la précision reste dans la tolérance.

        Chaque essai = PruneUnits + court réajustement local (slow weights du seul module élagué).
        """
        pruned = {}
        for module in self.region_for(model, domain):
            if module not in model.modules or module in model.fast_parameters: continue
            reference = model.accuracy(x_val, y_val)
            best, before = None, model.modules[module].hidden
            keep = before - step
            while keep >= step:
                trial = copy.deepcopy(model)
                trial.apply(PruneUnits(module=module, keep=keep))
                if x_train is not None and recovery_steps:
                    trial.apply(AdaptWeights(name="recover_after_prune", modules=(module,), inputs=x_train,
                                             targets=y_train, steps=recovery_steps, fast=False))
                if trial.accuracy(x_val, y_val) < reference - self.prune_tolerance: break
                best, keep = trial, keep - step
            if best is not None:
                model.__dict__.update(best.__dict__)
                pruned[module] = (before, model.modules[module].hidden)
        return pruned

    # ---- coopération avec l'apprentissage structurel ------------------
    def decide_for_skill(self, strategy, detector_accuracy=None, reliability_threshold=0.8):
        """Une Skill WCE apprise -> faut-il un changement neuronal ?"""
        reliable = strategy.reliability >= reliability_threshold
        if detector_accuracy is None or detector_accuracy >= 0.9:
            return ("no_neural_change" if reliable else "fast_weights",
                    "la compétence structurelle suffit" if reliable else "stratégie encore peu fiable : adaptation rapide")
        if detector_accuracy >= 0.6:
            return "fast_weights", "le motif est mal détecté : deltas rapides sur le détecteur"
        return "add_module", "motif non détecté : module spécialisé nécessaire"


# ---------------------------------------------------------------- boucle

@dataclass
class LoopReport:
    stages: List[str] = field(default_factory=list)
    before_accuracy: float = 0.0
    after_accuracy: float = 0.0
    weakness: bool = False
    candidates: List[str] = field(default_factory=list)
    branches: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    distribution: Dict[str, float] = field(default_factory=dict)
    selected: Optional[str] = None
    selected_description: str = ""
    consolidated: Tuple[str, ...] = ()
    pruned: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    capacity: Dict[str, int] = field(default_factory=dict)
    snapshot: Optional[str] = None
    model: Optional[ModelState] = None


def run_learning_loop(model, domain, x_train, y_train, x_val, y_val, controller=None, *, goal=None, seed=0):
    """Input -> inference -> evaluate -> weakness -> propose -> branch -> test -> select
    -> consolidate -> prune -> snapshot."""
    controller = controller or PlasticityController()
    report = LoopReport(model=model)
    report.stages.append("inference")
    report.before_accuracy = model.accuracy(x_val, y_val)
    report.stages.append("evaluate")
    goal = goal or Goal("accuracy", domain)
    report.weakness = controller.detect_weakness(model, goal, report.before_accuracy)
    report.stages.append("detect_weakness")
    if not report.weakness:
        report.after_accuracy = report.before_accuracy
        return report
    plans = controller.propose(model, domain, x_train, y_train, seed=seed)
    report.candidates = [p.description for p in plans]
    report.stages.append("propose")
    hypotheses = ModelHypothesisSet(model)
    by_label = {p.label: p for p in plans}
    for plan in plans: hypotheses.branch(plan.label, plan.specs)
    report.stages.append("branch")
    hypotheses.evaluate(lambda b: controller.score(b, by_label[b.label], domain, x_val, y_val))
    report.stages.append("test")
    report.branches = {b.label: {"score": round(b.score, 3) if b.score is not None else None, "status": b.status,
                                 **{k: v for k, v in b.details.items() if k not in ("effects", "specs")}}
                       for b in hypotheses.branches}
    report.distribution = hypotheses.distribution()
    best = hypotheses.select()
    report.selected, report.selected_description = best.label, by_label[best.label].description
    report.stages.append("select")
    chosen = best.state
    report.capacity["before"] = model.capacity
    report.capacity["max"] = max(b.state.capacity for b in hypotheses.valid())
    report.consolidated = controller.consolidate(chosen, domain)
    report.stages.append("consolidate")
    report.pruned = controller.prune(chosen, domain, x_val, y_val, x_train, y_train)
    report.stages.append("prune")
    report.capacity["final"] = chosen.capacity
    report.after_accuracy = chosen.accuracy(x_val, y_val)
    report.snapshot = f"after_{domain}"
    chosen.snapshot(report.snapshot)
    save_context(chosen, domain)
    report.stages.append("snapshot")
    report.model = chosen
    return report


# ---------------------------------------------------------------- mémoire

class MemorySystem:
    """Cinq niveaux de mémoire, chacun avec sa propre API."""

    def __init__(self, model, strategy_library=None, experiences=None, world=None):
        from core.planner import StrategyLibrary
        self.model = model
        self.strategy_library = strategy_library if strategy_library is not None else StrategyLibrary()
        self.experiences = list(experiences or [])
        self.world = world

    def parametric(self): return self.model.parameters                       # slow weights
    def fast_parametric(self): return self.model.fast_parameters             # fast weights
    def structural(self): return self.strategy_library                       # StrategyLibrary / skills
    def episodic(self): return self.experiences                              # Experience
    def current(self): return self.world                                     # World / Facts / Assumptions

    def remember_experience(self, experience): self.experiences.append(experience); return experience
    def learn_skill(self, strategy): self.strategy_library.store_parameterized(strategy); return strategy

    def describe(self):
        return {"parametric": self.model.total_parameters(),
                "fast_parametric": self.model.fast_parameter_count(),
                "structural": len(self.strategy_library),
                "episodic": len(self.experiences),
                "current": type(self.world).__name__ if self.world is not None else None}

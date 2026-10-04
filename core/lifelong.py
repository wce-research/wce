"""Routage autonome + plasticité continue (lifelong).

API publique finale : LifelongModel.predict(x) — aucune étiquette de tâche.

ContextEncoder (petit MLP réel) -> TaskRouter (prototypes appris) -> configuration
(module + fast weights propres au contexte) -> inférence.
MetaController orchestre : REUSE / ADAPT / ROUTE / GROW / PRUNE / ROLLBACK / REPAIR.
Toutes les modifications du modèle passent par des ModelTransformationSpec.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from core.branching import adopt, shared_branch
from core.invariants import Invariant, InvariantSet
from core.learning import Experience
from core.model_state import (
    INPUT, PARAM_NAMES, ActivateModule, AdaptWeights, AddModule, BindRoute, ClearFastWeights, CloneModule,
    CreateConnection, DeactivateModule, GrowModule, ModelState, ModelTransformationSpec, ModuleExists,
    PruneModule, PruneUnits, RemoveConnection, RestoreModule, _spec, LoadFastDelta,
)
from core.plasticity import MemorySystem, ModelHypothesisSet
from core.reasoning import Goal

CONTEXT_DIM, PAYLOAD_DIM = 4, 8
INPUT_DIM = CONTEXT_DIM + PAYLOAD_DIM
HEAD_KEYS = ("w2", "b2")
KNOWN, RELATED, UNKNOWN = "known", "related", "unknown"


# ================================================================ tâches

FUNCTIONS = {
    "xor01": lambda p: (p[:, 0] != p[:, 1]).long(),
    "xnor01": lambda p: (p[:, 0] == p[:, 1]).long(),
    "maj37": lambda p: (p[:, 3:8].sum(dim=1) >= 3).long(),
    "and23_or6": lambda p: ((p[:, 2] * p[:, 3] + p[:, 6]) > 0).long(),
    "xor57": lambda p: (p[:, 5] != p[:, 7]).long(),
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    context: int
    function: str
    expected_novelty: str


@dataclass
class TaskData:
    spec: TaskSpec
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor

    @property
    def name(self): return self.spec.name


def context_prototypes(n=8, dim=CONTEXT_DIM, seed=7, spread=2.5, min_distance=3.0):
    gen = torch.Generator().manual_seed(seed)
    protos = []
    while len(protos) < n:
        candidate = torch.randn(dim, generator=gen) * spread
        if all(float((candidate - p).norm()) >= min_distance for p in protos): protos.append(candidate)
    return torch.stack(protos)


def sample_inputs(proto, n, gen, noise=0.35):
    context = proto + noise * torch.randn(n, len(proto), generator=gen)
    payload = torch.randint(0, 2, (n, PAYLOAD_DIM), generator=gen).float()
    return torch.cat([context, payload], dim=1)


def make_task(spec, prototypes, seed=0, n_train=384, n_val=128, n_test=256):
    gen = torch.Generator().manual_seed(1000 + seed)
    f = FUNCTIONS[spec.function]
    parts = []
    for n in (n_train, n_val, n_test):
        x = sample_inputs(prototypes[spec.context], n, gen)
        parts += [x, f(x[:, CONTEXT_DIM:])]
    return TaskData(spec, *parts)


def default_stream():
    """8 tâches : liens positifs (même fonction), conflits, et un retour de contexte connu."""
    return [
        TaskSpec("T1", 0, "xor01", UNKNOWN),
        TaskSpec("T2", 1, "xor01", RELATED),      # même fonction, autre contexte : transfert positif
        TaskSpec("T3", 2, "maj37", UNKNOWN),      # incompatible avec T1 : transfert négatif
        TaskSpec("T4", 3, "xnor01", RELATED),     # négation de T1 : réutilisation + fast weights
        TaskSpec("T5", 4, "and23_or6", UNKNOWN),
        TaskSpec("T6", 5, "maj37", RELATED),      # même fonction que T3
        TaskSpec("T7", 6, "xor57", UNKNOWN),
        TaskSpec("T8", 2, "maj37", KNOWN),        # retour exact du contexte de T3
    ]


def build_stream(specs=None, seed=0):
    specs = specs or default_stream()
    protos = context_prototypes(n=max(s.context for s in specs) + 1)
    return [make_task(s, protos, seed=seed * 100 + i) for i, s in enumerate(specs)], protos


# ================================================================ specs spécifiques

@_spec
class CorruptModule(ModelTransformationSpec):
    """Dommage artificiel contrôlé (test de régénération)."""
    name: str = "corrupt_module"
    module: str = ""
    scale: float = 1.5
    seed: int = 0
    invertible: bool = False
    provenance: str = "damage_test"
    changes: frozenset = frozenset({"weights"})
    def __post_init__(self): object.__setattr__(self, "preconditions", (ModuleExists(self.module),))
    def execute(self, model):
        gen = torch.Generator().manual_seed(self.seed)
        m = model.modules[self.module]
        m.slow = {k: t + self.scale * (float(t.std()) + 1e-3) * torch.randn(t.shape, generator=gen) for k, t in m.slow.items()}
        return {"corrupted": self.module}


# ================================================================ encodeur / routeur / nouveauté

class ContextEncoder:
    """Petit MLP réel (12 -> 16 -> 6), gelé : projection compacte du contexte."""
    MODULE = "context_encoder"

    def __init__(self, state, embed_dim=6, hidden=16, seed=11):
        self.state, self.embed_dim = state, embed_dim
        state.apply(AddModule(module=self.MODULE, input_dim=state.input_dim, output_dim=embed_dim, hidden=hidden,
                              activation=False, specialization="context", seed=seed, provenance="context_encoder"))
        state.apply(CreateConnection(source=INPUT, target=self.MODULE))

    @property
    def parameter_count(self): return self.state.modules[self.MODULE].parameter_count

    def encode(self, x):
        with torch.no_grad(): return self.state.modules[self.MODULE].forward(x)


class TaskRouter:
    """Prototypes gaussiens diagonaux par configuration, appris localement depuis l'expérience."""

    def __init__(self, var_floor=0.05, confident=0.6):
        self.prototypes: Dict[str, Dict[str, torch.Tensor]] = {}
        self.var_floor, self.confident = var_floor, confident

    @property
    def names(self): return list(self.prototypes)

    def learn(self, name, emb):
        n, mean, var = len(emb), emb.mean(0), emb.var(0, unbiased=False)
        if name not in self.prototypes:
            self.prototypes[name] = {"mean": mean, "var": var, "count": n}
            return
        p = self.prototypes[name]; total = p["count"] + n
        delta = mean - p["mean"]
        new_mean = p["mean"] + delta * n / total
        new_var = (p["var"] * p["count"] + var * n + delta ** 2 * p["count"] * n / total) / total
        self.prototypes[name] = {"mean": new_mean, "var": new_var, "count": total}

    def forget(self, name): self.prototypes.pop(name, None)

    def _stack(self):
        means = torch.stack([p["mean"] for p in self.prototypes.values()])
        var = torch.stack([p["var"] for p in self.prototypes.values()]).clamp_min(self.var_floor)
        return means, var

    def log_likelihood(self, emb):
        means, var = self._stack()
        diff = emb.unsqueeze(1) - means.unsqueeze(0)
        return -0.5 * ((diff ** 2 / var).sum(-1) + var.log().sum(-1))

    def z_distance(self, emb):
        means, var = self._stack()
        diff = emb.unsqueeze(1) - means.unsqueeze(0)
        return (diff ** 2 / var).mean(-1).sqrt()

    def probs(self, emb, log_prior=None):
        ll = self.log_likelihood(emb)
        if log_prior is not None: ll = ll + log_prior
        return torch.softmax(ll, dim=1)

    def route(self, emb, log_prior=None):
        p = self.probs(emb, log_prior)
        conf, idx = p.max(dim=1)
        return [self.names[i] for i in idx.tolist()], conf, p


@dataclass
class NoveltyReport:
    kind: str
    nearest: Optional[str] = None
    coverage: float = 0.0
    best_config: Optional[str] = None
    best_agreement: float = 0.0
    accuracies: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


class NoveltyDetector:
    """Trois métriques explicables : couverture z du prototype le plus proche,
    précision de la configuration la plus proche, accord fonctionnel max(acc, 1-acc)."""

    def __init__(self, z_known=3.0, coverage=0.8, known_accuracy=0.9, related_agreement=0.75):
        self.z_known, self.coverage, self.known_accuracy, self.related_agreement = z_known, coverage, known_accuracy, related_agreement

    def assess(self, router, emb, accuracies, probes=None):
        if not router.prototypes: return NoveltyReport(UNKNOWN, reason="aucun contexte connu")
        z = router.z_distance(emb)
        nearest_idx = int(torch.mode(z.argmin(dim=1)).values)
        nearest = router.names[nearest_idx]
        coverage = float((z[:, nearest_idx] < self.z_known).float().mean())
        agreements = {c: max(a, 1 - a, (probes or {}).get(c, 0.0)) for c, a in accuracies.items()}
        best = max(agreements, key=agreements.get) if agreements else None
        report = NoveltyReport(UNKNOWN, nearest, coverage, best, agreements.get(best, 0.0), dict(accuracies))
        if coverage >= self.coverage and accuracies.get(nearest, 0.0) >= self.known_accuracy:
            report.kind, report.reason = KNOWN, f"contexte couvert ({coverage:.2f}) et {nearest} précis ({accuracies[nearest]:.2f})"
        elif report.best_agreement >= self.related_agreement:
            report.kind, report.reason = RELATED, f"{best} accord fonctionnel {report.best_agreement:.2f}"
        else:
            report.reason = f"couverture {coverage:.2f}, meilleur accord {report.best_agreement:.2f}"
        return report


# ================================================================ modèle

@dataclass
class Configuration:
    name: str
    fast: Dict[str, torch.Tensor] = field(default_factory=dict)
    skill: Optional[str] = None
    tasks: List[str] = field(default_factory=list)
    replay_x: Any = None
    replay_y: Any = None
    probe_x: Any = None
    probe_y: Any = None
    baseline: float = 0.0
    uses: int = 0
    origin: str = ""

    @property
    def fast_parameter_count(self): return sum(t.numel() for t in self.fast.values())


class LifelongModel:
    def __init__(self, input_dim=INPUT_DIM, seed=0):
        self.state = ModelState(input_dim, output_module=None, max_connections=64,
                                max_capacity=1024, max_region_fraction=0.75)
        self.state.metadata["routes"] = {}
        self.encoder = ContextEncoder(self.state, seed=11 + seed)
        self.router = TaskRouter()
        self.configs: Dict[str, Configuration] = {}
        self.memory = MemorySystem(self.state)
        self.seed = seed

    # ---- routage ------------------------------------------------------
    def module_of(self, config, state=None):
        return (state or self.state).metadata["routes"].get(config)

    def usable(self, config, state=None):
        state = state or self.state
        module = self.module_of(config, state)
        return (module in state.modules and state.modules[module].active
                and (INPUT, module) in state.connectivity)

    def config_logits(self, config, x, state=None, fast=None, module=None):
        state = state or self.state
        module = module or self.module_of(config, state)
        if module is None or not (module in state.modules and state.modules[module].active
                                  and (INPUT, module) in state.connectivity):
            return torch.zeros(len(x), 2)
        delta = fast if fast is not None else (self.configs[config].fast if config in self.configs else {})
        with torch.no_grad(): return state.modules[module].forward(x, delta)

    def config_accuracy(self, config, x, y, **kwargs):
        return float((self.config_logits(config, x, **kwargs).argmax(1) == y).float().mean())

    def log_prior(self, structural_skill=None):
        """Mémoire épisodique (usage) + évidence structurelle WCE (skill)."""
        prior = torch.tensor([0.1 * math.log1p(self.configs[n].uses) for n in self.router.names])
        if structural_skill:
            prior += torch.tensor([math.log(4.0) if self.configs[n].skill == structural_skill else 0.0
                                   for n in self.router.names])
        return prior

    def structural_skill(self, structural):
        if structural is None: return None
        state, goal = structural
        match = self.memory.structural().best_match(state, goal)
        return match.strategy.name if match is not None else None

    def route(self, x, structural=None):
        if not self.router.prototypes: raise ValueError("aucune configuration apprise")
        return self.router.route(self.encoder.encode(x), self.log_prior(self.structural_skill(structural)))

    def predict_proba(self, x, structural=None):
        names, conf, probs = self.route(x, structural)
        outputs = torch.stack([torch.softmax(self.config_logits(n, x), dim=1) for n in self.router.names], dim=1)
        mixture = (probs.unsqueeze(2) * outputs).sum(1)             # configurations superposées
        hard = outputs[torch.arange(len(x)), probs.argmax(1)]        # configuration dominante
        confident = (conf >= self.router.confident).unsqueeze(1)
        return torch.where(confident, hard, mixture)

    def predict(self, x, structural=None):
        """API publique : aucune étiquette de tâche."""
        return self.predict_proba(x, structural).argmax(1)

    def active_parameters(self, x):
        names, conf, _ = self.route(x)
        per = [self.encoder.parameter_count + self.state.modules[self.module_of(n)].parameter_count
               + self.configs[n].fast_parameter_count if self.usable(n) else self.encoder.parameter_count
               for n in names]
        return sum(per) / len(per)

    def fast_parameter_count(self): return sum(c.fast_parameter_count for c in self.configs.values())
    def experts(self): return sorted(n for n in self.state.modules if n != ContextEncoder.MODULE)


# ================================================================ méta-contrôleur

@dataclass
class Candidate:
    label: str
    kind: str
    specs: List[ModelTransformationSpec]
    module: str
    config: Optional[str] = None
    source: Optional[str] = None
    prior: float = 1.0
    added_parameters: int = 0


@dataclass
class TaskDecision:
    task: str
    novelty: NoveltyReport
    action: str = ""
    kind: str = ""
    config: str = ""
    module: str = ""
    candidates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    accuracy: float = 0.0
    routing_confidence: float = 0.0
    uncertain: bool = False
    positive_transfer: Optional[float] = None
    parameters_saved: Optional[int] = None
    negative_transfer: Optional[float] = None
    grew: bool = False
    pruned: Optional[Tuple[int, int]] = None
    homeostasis: str = ""
    shared_ratio: float = 0.0
    skill: Optional[str] = None
    reason: str = ""


@dataclass
class RepairReport:
    config: str
    diagnosis: str
    before: float
    baseline: float
    candidates: Dict[str, float] = field(default_factory=dict)
    selected: str = ""
    after: float = 0.0
    success: bool = False


class MetaController:
    """Petit orchestrateur inspectable : chaque décision garde ses scores et sa raison."""

    ACTIONS = ("REUSE", "ADAPT", "ROUTE", "GROW", "PRUNE", "ROLLBACK", "REPAIR")

    def __init__(self, model: LifelongModel, *, forgetting_penalty=1.0, param_penalty=2e-4, target_accuracy=0.9,
                 prune_tolerance=0.02, budget=(400, 900), expert_hidden=32, seed=0):
        self.model = model
        self.novelty = NoveltyDetector()
        self.forgetting_penalty, self.param_penalty = forgetting_penalty, param_penalty
        self.target_accuracy, self.prune_tolerance = target_accuracy, prune_tolerance
        self.budget, self.expert_hidden, self.seed = budget, expert_hidden, seed
        self.decisions: List[TaskDecision] = []
        self.repairs: List[RepairReport] = []
        self.stats = {"branches": 0, "evaluations": 0, "modules_created": 0, "modules_removed": 0,
                      "prunes": 0, "grows": 0, "shared_ratios": []}
        self._next_expert = 1
        self._task = "init"

    # ---- utilitaires ----------------------------------------------------
    def _new_expert_name(self):
        return f"expert_{self._next_expert}"      # incrémenté seulement si l'expert est adopté

    def _new_config_name(self):
        return f"cfg_{self._task}"

    def _dependents(self, module, state=None):
        routes = (state or self.model.state).metadata["routes"]
        return [c for c, m in routes.items() if m == module and c in self.model.configs]

    def _probe_accuracy(self, config, state):
        c = self.model.configs[config]
        return self.model.config_accuracy(config, c.probe_x, c.probe_y, state=state)

    # ---- candidats ------------------------------------------------------
    def propose(self, novelty, router_probs, x, y, skill=None):
        m = self.model
        cands: List[Candidate] = []
        steps = dict(inputs=x, targets=y, seed=self.seed)
        if not m.configs or novelty.kind == UNKNOWN or novelty.kind == RELATED:
            name, cfg = self._new_expert_name(), self._new_config_name()
            spec_add = AddModule(module=name, input_dim=m.state.input_dim, output_dim=2, hidden=self.expert_hidden,
                                 activation=False, specialization="expert", seed=self.seed + self._next_expert)
            cands.append(Candidate("new_expert", "GROW", [
                spec_add, CreateConnection(source=INPUT, target=name),
                AdaptWeights(modules=(name,), fast=False, output=name, steps=250, lr=0.02, **steps),
                BindRoute(configuration=cfg, module=name)], name, cfg,
                added_parameters=NeuralModuleCount.count(m.state.input_dim, self.expert_hidden, 2)))
        if not m.configs: return cands
        ranked = sorted(router_probs, key=router_probs.get, reverse=True)
        reuse = [ranked[0]] if novelty.kind == KNOWN else list(dict.fromkeys(ranked[:2] + [novelty.best_config]))
        for c in reuse:
            if c is None or not m.usable(c): continue
            cands.append(Candidate(f"reuse:{c}", "REUSE", [], m.module_of(c), c, c, prior=router_probs.get(c, 0.0)))
        if novelty.kind == KNOWN: return cands
        src = novelty.best_config
        if src is not None and m.usable(src):
            module = m.module_of(src)
            cfg = self._new_config_name()
            cands.append(Candidate(f"adapt:{src}", "ADAPT", [
                LoadFastDelta(module=module, delta=dict(m.configs[src].fast)),
                AdaptWeights(modules=(module,), fast=True, keys=HEAD_KEYS, output=module, steps=150, lr=0.05, **steps),
                BindRoute(configuration=cfg, module=module)], module, cfg, src,
                added_parameters=sum(m.state.modules[module].slow[k].numel() for k in HEAD_KEYS)))
            cfg2 = self._new_config_name()
            cands.append(Candidate(f"share:{src}", "SHARE", [
                AdaptWeights(modules=(module,), fast=False, output=module, steps=150, lr=0.02, **steps),
                BindRoute(configuration=cfg2, module=module)], module, cfg2, src))
        return cands

    def _evaluate(self, cand, branch, x_val, y_val):
        m = self.model
        details = {}
        fast = {}
        if cand.kind == "ADAPT":
            fast = {k: branch.fast_parameters[cand.module][k].clone() for k in HEAD_KEYS}
            branch.apply(ClearFastWeights(modules=(cand.module,)))
        elif cand.kind == "REUSE":
            fast = m.configs[cand.config].fast
        accuracy = m.config_accuracy(cand.config, x_val, y_val, state=branch, fast=fast, module=cand.module)
        forgetting = 0.0
        if cand.kind == "SHARE":
            drops = [max(0.0, self._probe_accuracy(c, m.state) - self._probe_accuracy(c, branch))
                     for c in self._dependents(cand.module)]
            forgetting = sum(drops) / len(drops) if drops else 0.0
        score = accuracy - self.forgetting_penalty * forgetting - self.param_penalty * cand.added_parameters
        details.update(accuracy=round(accuracy, 3), forgetting=round(forgetting, 3),
                       added_parameters=cand.added_parameters, score=round(score, 3))
        return score, details, fast

    # ---- boucle lifelong -------------------------------------------------
    def observe(self, task, x_train, y_train, x_val, y_val, structural=None):
        """input batch -> encoding -> routing -> inference -> evaluation -> novelty -> candidates
        -> branching -> selection -> adaptation -> consolidation -> pruning -> memory update."""
        m = self.model
        self._task = task
        emb_train, emb_val = m.encoder.encode(x_train), m.encoder.encode(x_val)
        skill = m.structural_skill(structural)
        router_probs, confidence = {}, 0.0
        if m.router.prototypes:
            _, conf, probs = m.router.route(emb_val, m.log_prior(skill))
            router_probs = dict(zip(m.router.names, probs.mean(0).tolist()))
            confidence = float(conf.mean())
        accuracies = {c: m.config_accuracy(c, x_val, y_val) for c in m.configs if m.usable(c)}
        probes = self.head_probes(accuracies, x_train, y_train, x_val, y_val)
        novelty = self.novelty.assess(m.router, emb_val, accuracies, probes)
        decision = TaskDecision(task, novelty, routing_confidence=confidence, skill=skill,
                                uncertain=bool(router_probs) and max(router_probs.values()) < m.router.confident)

        cands = self.propose(novelty, router_probs, x_train, y_train, skill)
        hyp = ModelHypothesisSet(m.state, shared=True)
        results = {}
        for cand in cands:
            branch = hyp.branch(cand.label, cand.specs, weight=cand.prior)
            if branch.status == "invalid": continue
            score, details, fast = self._evaluate(cand, branch.state, x_val, y_val)
            branch.score = score; hyp.evaluations += 1
            results[cand.label] = (cand, branch, fast)
            decision.candidates[cand.label] = details
        # superposition classique : a priori du routeur x évidence mesurée, puis sélection
        hyp._renormalize()
        hyp.reweight({label: math.exp(10 * details["score"]) for label, details in decision.candidates.items()})
        for label, (_, branch, _) in results.items(): decision.candidates[label]["weight"] = round(branch.weight, 3)
        memory = hyp.memory()
        decision.shared_ratio = memory["shared_ratio"]
        self.stats["shared_ratios"].append(memory["shared_ratio"])
        self.stats["branches"] += len(hyp.branches); self.stats["evaluations"] += hyp.evaluations
        best = hyp.select()
        cand, branch, fast = results[best.label]
        adopt(m.state, branch.state)

        decision.action = {"GROW": "GROW", "REUSE": "REUSE", "ADAPT": "ADAPT", "SHARE": "ADAPT"}[cand.kind]
        if decision.uncertain and cand.kind == "REUSE": decision.action = "ROUTE"
        decision.module, decision.kind = cand.module, cand.kind
        if cand.kind == "REUSE":
            config = m.configs[cand.config]
        else:
            config = Configuration(cand.config, fast=fast, origin=cand.kind.lower())
            m.configs[cand.config] = config
        if cand.kind == "GROW": self.stats["modules_created"] += 1; self._next_expert += 1
        decision.config = config.name
        config.tasks.append(task); config.uses += 1
        if skill: config.skill = skill

        # capacité insuffisante -> croissance
        if cand.kind == "GROW":
            acc = m.config_accuracy(config.name, x_val, y_val)
            if acc < self.target_accuracy: decision.grew = self._grow(config.name, x_train, y_train, x_val, y_val)
            decision.pruned = self._prune(cand.module, x_train, y_train, x_val, y_val)

        decision.accuracy = m.config_accuracy(config.name, x_val, y_val)
        self._transfer(decision, results, x_train, y_train, x_val, y_val)
        m.router.learn(config.name, emb_train)
        config.replay_x, config.replay_y = x_train[:192], y_train[:192]
        config.probe_x, config.probe_y = x_val, y_val
        config.baseline = decision.accuracy
        decision.homeostasis = self.homeostasis()
        m.state.snapshot(f"after_{task}")
        m.memory.remember_experience(Experience(
            initial_state=task, goal=Goal("classify", task), successful_plan=[s.name for s in cand.specs],
            bindings={"config": config.name, "module": cand.module}, cost=len(cand.specs),
            explored_nodes=len(results), result=decision.accuracy, strategy_used=decision.action))
        decision.reason = f"{novelty.kind}: {novelty.reason} -> {best.label}"
        self.decisions.append(decision)
        return decision

    def head_probes(self, accuracies, x_train, y_train, x_val, y_val, top=2, steps=40):
        """Accord fonctionnel : précision après 40 pas d'adaptation de la seule tête (w2, b2)."""
        m = self.model; out = {}
        for c in sorted(accuracies, key=lambda c: -max(accuracies[c], 1 - accuracies[c]))[:top]:
            module = m.module_of(c)
            probe = shared_branch(m.state)
            probe.apply(LoadFastDelta(module=module, delta=dict(m.configs[c].fast)))
            probe.apply(AdaptWeights(name="novelty_probe", modules=(module,), fast=True, keys=HEAD_KEYS, output=module,
                                     steps=steps, lr=0.05, inputs=x_train, targets=y_train, local=False))
            with torch.no_grad():
                pred = probe.modules[module].forward(x_val, probe.fast_parameters[module]).argmax(1)
            out[c] = float((pred == y_val).float().mean())
        return out

    def _transfer(self, decision, results, x_train, y_train, x_val, y_val):
        """Transfert positif : gain d'une réutilisation vs expert neuf à petit budget égal.
        Transfert négatif : oubli mesuré si le module était partagé, ou perte vs expert dédié."""
        share = next((d for label, d in decision.candidates.items() if label.startswith("share:")), None)
        new = decision.candidates.get("new_expert")
        adapt = next((d for label, d in decision.candidates.items() if label.startswith("adapt:")), None)
        if decision.action == "ADAPT" and adapt is not None:
            probe = self._budget_probe(decision.module, decision.config, x_train, y_train, x_val, y_val)
            decision.positive_transfer = probe
            if new is not None: decision.parameters_saved = new["added_parameters"] - adapt["added_parameters"]
        negatives = []
        if share is not None: negatives.append(share["forgetting"])
        if new is not None and adapt is not None and decision.action == "GROW":
            negatives.append(max(0.0, new["accuracy"] - adapt["accuracy"]))
        if negatives: decision.negative_transfer = max(negatives)

    def _budget_probe(self, module, config, x_train, y_train, x_val, y_val, steps=10):
        m = self.model
        src_fast = m.configs[config].fast
        reuse_branch = shared_branch(m.state)
        reuse_branch.apply(LoadFastDelta(module=module, delta={k: torch.zeros_like(v) for k, v in src_fast.items()}))
        reuse_branch.apply(AdaptWeights(modules=(module,), fast=True, keys=HEAD_KEYS, output=module, steps=steps,
                                        lr=0.05, inputs=x_train, targets=y_train, local=False))
        reuse_acc = float((reuse_branch.modules[module].forward(x_val, reuse_branch.fast_parameters[module]).argmax(1) == y_val).float().mean())
        scratch = shared_branch(m.state)
        scratch.apply(AddModule(module="probe_scratch", input_dim=m.state.input_dim, output_dim=2, hidden=self.expert_hidden,
                                activation=False, seed=self.seed + 99))
        scratch.apply(CreateConnection(source=INPUT, target="probe_scratch"))
        scratch.apply(AdaptWeights(modules=("probe_scratch",), fast=False, output="probe_scratch", steps=steps, lr=0.02,
                                   inputs=x_train, targets=y_train, local=False))
        with torch.no_grad():
            scratch_acc = float((scratch.modules["probe_scratch"].forward(x_val).argmax(1) == y_val).float().mean())
        return round(reuse_acc - scratch_acc, 3)

    def _grow(self, config, x_train, y_train, x_val, y_val, units=16):
        m = self.model; module = m.module_of(config)
        before = m.config_accuracy(config, x_val, y_val)
        branch = shared_branch(m.state)
        try:
            branch.apply(GrowModule(module=module, units=units, seed=self.seed))
            branch.apply(AdaptWeights(modules=(module,), fast=False, output=module, steps=150, lr=0.02,
                                      inputs=x_train, targets=y_train))
        except ValueError:
            return False
        if m.config_accuracy(config, x_val, y_val, state=branch) > before:
            adopt(m.state, branch); self.stats["grows"] += 1
            return True
        return False

    def _prune(self, module, x_train, y_train, x_val, y_val, step=8, recovery=100, min_keep=8):
        """Prune after consolidate : seulement si aucune configuration ne dépend de deltas rapides sur ce module."""
        m = self.model
        deps = self._dependents(module)
        if any(m.configs[c].fast for c in deps): return None
        reference = {c: m.config_accuracy(c, x_val, y_val) for c in deps}
        before = m.state.modules[module].hidden
        best, keep = None, before - step
        while keep >= min_keep:
            trial = shared_branch(best or m.state)
            trial.apply(PruneUnits(module=module, keep=keep))
            trial.apply(AdaptWeights(name="recover_after_prune", modules=(module,), fast=False, output=module,
                                     steps=recovery, lr=0.02, inputs=x_train, targets=y_train, local=False))
            if any(m.config_accuracy(c, x_val, y_val, state=trial) < reference[c] - self.prune_tolerance for c in deps): break
            best, keep = trial, keep - step
        if best is None: return None
        adopt(m.state, best); self.stats["prunes"] += 1
        self.stats["units_pruned"] = self.stats.get("units_pruned", 0) + before - m.state.modules[module].hidden
        return (before, m.state.modules[module].hidden)

    # ---- homéostasie ---------------------------------------------------
    def homeostasis(self):
        """Tend vers une plage cible de paramètres totaux : prune au-dessus, autorise la croissance en dessous."""
        low, high = self.budget
        m = self.model
        total = m.state.total_parameters()
        if total > high:
            for module in sorted(m.experts(), key=lambda n: -m.state.modules[n].hidden):
                deps = self._dependents(module)
                if not deps or any(m.configs[c].fast for c in deps): continue
                cfg = m.configs[deps[0]]
                self._prune(module, cfg.replay_x, cfg.replay_y, cfg.probe_x, cfg.probe_y)
                if m.state.total_parameters() <= high: break
            after = m.state.total_parameters()
            if after <= high:
                m.state.metadata["max_capacity"] = 1024
                return f"over budget ({total} > {high}) -> pruned to {after}"
            m.state.metadata["max_capacity"] = m.state.capacity  # croissance gelée tant que > budget
            return f"over budget ({total} > {high}) -> pruning floor reached at {after}, growth frozen"
        m.state.metadata["max_capacity"] = 1024
        return "within budget" if total >= low else f"below budget ({total} < {low}) -> growth allowed"

    # ---- dommages et réparation ------------------------------------------
    def damage(self, kind, module, seed=0):
        specs = {"corrupt": CorruptModule(module=module, seed=seed), "deactivate": DeactivateModule(module=module),
                 "disconnect": RemoveConnection(source=INPUT, target=module), "delete": PruneModule(module=module)}
        return self.model.state.apply(specs[kind])

    def health_check(self, tolerance=0.1):
        m = self.model
        return {c: (cfg.baseline, self._probe_accuracy(c, m.state)) for c, cfg in m.configs.items()
                if cfg.probe_x is not None and self._probe_accuracy(c, m.state) < cfg.baseline - tolerance}

    def diagnose(self, config):
        m = self.model; module = m.module_of(config); state = m.state
        if module not in state.modules: return "missing"
        if not state.modules[module].active: return "inactive"
        if (INPUT, module) not in state.connectivity: return "disconnected"
        drop = self.model.configs[config].baseline - self._probe_accuracy(config, state)
        return "corrupted" if drop > 0.15 else "drift"

    def _latest_snapshot_with(self, module):
        for name in reversed(list(self.model.state.snapshots)):
            if module in self.model.state.snapshots[name].modules: return name
        return None

    def repair_candidates(self, config, diagnosis, allow_rollback=True):
        m = self.model; module = m.module_of(config); cfg = m.configs[config]
        train = dict(inputs=cfg.replay_x, targets=cfg.replay_y, seed=self.seed)
        cands = []
        snapshot = self._latest_snapshot_with(module) if allow_rollback else None
        if diagnosis == "inactive":
            cands.append(("reroute", [ActivateModule(module=module)], 0, False))
        if diagnosis == "disconnected":
            cands.append(("reconnect", [CreateConnection(source=INPUT, target=module)], 0, False))
        if diagnosis in ("corrupted", "missing") and snapshot:
            specs = [RestoreModule(module=module, snapshot=snapshot)]
            if (INPUT, module) not in m.state.connectivity: specs.append(CreateConnection(source=INPUT, target=module))
            cands.append(("rollback", specs, 0, False))
        if diagnosis in ("corrupted", "missing"):
            healthy = [e for e in m.experts() if e != module and m.state.modules[e].active
                       and (INPUT, e) in m.state.connectivity]
            if healthy:
                src = max(healthy, key=lambda e: max(a := m.config_accuracy(config, cfg.probe_x, cfg.probe_y, module=e, fast={}), 1 - a))
                new = f"{module}_clone"
                cands.append(("clone+reconnect", [CloneModule(module=src, new_name=new), ActivateModule(module=new),
                                                  CreateConnection(source=INPUT, target=new),
                                                  AdaptWeights(modules=(new,), fast=False, output=new, steps=200, lr=0.02, local=False, **train),
                                                  BindRoute(configuration=config, module=new)],
                              m.state.modules[src].parameter_count, True))
            new = f"{module}_regrown"
            cands.append(("grow_replacement", [AddModule(module=new, input_dim=m.state.input_dim, output_dim=2,
                                                         hidden=self.expert_hidden, activation=False, seed=self.seed + 7),
                                               CreateConnection(source=INPUT, target=new),
                                               AdaptWeights(modules=(new,), fast=False, output=new, steps=250, lr=0.02, local=False, **train),
                                               BindRoute(configuration=config, module=new)],
                          NeuralModuleCount.count(m.state.input_dim, self.expert_hidden, 2), True))
        if diagnosis in ("drift", "corrupted"):
            cands.append(("fast_adaptation", [LoadFastDelta(module=module, delta=dict(cfg.fast)),
                                              AdaptWeights(modules=(module,), fast=True, output=module, steps=200, lr=0.03, local=False, **train)],
                          m.state.modules[module].parameter_count if module in m.state.modules else 0, "fast"))
        return cands

    def repair(self, config, allow_rollback=True):
        m = self.model; cfg = m.configs[config]
        diagnosis = self.diagnose(config)
        report = RepairReport(config, diagnosis, self._probe_accuracy(config, m.state), cfg.baseline)
        hyp = ModelHypothesisSet(m.state, shared=True)
        payload = {}
        for label, specs, added, resets in self.repair_candidates(config, diagnosis, allow_rollback):
            branch = hyp.branch(label, specs)
            if branch.status == "invalid": continue
            module = branch.state.metadata["routes"][config]
            fast = {}
            if resets == "fast":
                fast = {k: v.clone() for k, v in branch.state.fast_parameters[module].items()}
                branch.state.apply(ClearFastWeights(modules=(module,)))
            elif not resets:
                fast = cfg.fast
            acc = m.config_accuracy(config, cfg.probe_x, cfg.probe_y, state=branch.state, fast=fast, module=module)
            branch.score = acc - self.param_penalty * added; hyp.evaluations += 1
            report.candidates[label] = round(branch.score, 3)
            payload[label] = fast
        self.stats["branches"] += len(hyp.branches); self.stats["evaluations"] += hyp.evaluations
        best = hyp.select()
        adopt(m.state, best.state)
        cfg.fast = payload[best.label]
        report.selected = best.label
        report.after = self._probe_accuracy(config, m.state)
        report.success = report.after >= cfg.baseline - 0.05
        if best.label in ("clone+reconnect", "grow_replacement"): self.stats["modules_created"] += 1
        self.repairs.append(report)
        return report

    def self_repair(self, allow_rollback=True):
        """Répare les configurations dégradées, une à une, en revérifiant après chaque réparation
        (une réparation de module partagé peut suffire pour plusieurs configurations)."""
        reports = []
        while True:
            damaged = [c for c in self.health_check() if c not in {r.config for r in reports}]
            if not damaged: return reports
            reports.append(self.repair(damaged[0], allow_rollback))

    # ---- métriques -----------------------------------------------------
    def metrics(self, tasks=None, matrix=None):
        m = self.model
        n = len(self.decisions) or 1
        reused = sum(d.action in ("REUSE", "ADAPT", "ROUTE") for d in self.decisions)
        out = {
            "total_parameters": m.state.total_parameters(),
            "fast_parameters": m.fast_parameter_count(),
            "experts": len(m.experts()),
            "configurations": len(m.configs),
            "module_reuse_rate": round(reused / max(1, n - 1), 3),
            "module_creation_rate": round(self.stats["modules_created"] / n, 3),
            "prune_rate": round(self.stats["prunes"] / n, 3),
            "grow_count": self.stats["grows"],
            "rollback_rate": round(sum(r.selected == "rollback" for r in self.repairs) / max(1, len(self.repairs)), 3),
            "repair_success": round(sum(r.success for r in self.repairs) / max(1, len(self.repairs)), 3),
            "novelty_accuracy": None,
            "branch_count": self.stats["branches"],
            "candidate_evaluations": self.stats["evaluations"],
            "shared_state_memory_ratio": round(sum(self.stats["shared_ratios"]) / max(1, len(self.stats["shared_ratios"])), 3),
            "routing_confidence": round(sum(d.routing_confidence for d in self.decisions[1:]) / max(1, n - 1), 3),
            "positive_transfer": [d.positive_transfer for d in self.decisions if d.positive_transfer is not None],
            "negative_transfer": [d.negative_transfer for d in self.decisions if d.negative_transfer is not None],
            "capacity": m.state.capacity,
            "modules_created": self.stats["modules_created"],
            "modules_removed": self.stats["modules_created"] - len(m.experts()),
            "units_pruned": self.stats.get("units_pruned", 0),
            "specialization_rate": round(sum(d.action == "GROW" for d in self.decisions) / n, 3),
        }
        if tasks is not None:
            expected = {t.name: t.spec.expected_novelty for t in tasks}
            judged = [d for d in self.decisions if d.task in expected]
            out["novelty_accuracy"] = round(sum(d.novelty.kind == expected[d.task] for d in judged) / max(1, len(judged)), 3)
            test_x = torch.cat([t.x_test for t in tasks])
            out["average_active_parameters"] = round(m.active_parameters(test_x), 1)
        return out


class NeuralModuleCount:
    @staticmethod
    def count(input_dim, hidden, output_dim): return hidden * input_dim + hidden + output_dim * hidden + output_dim


# ================================================================ expérience complète

@dataclass
class LifelongRun:
    model: LifelongModel
    controller: MetaController
    tasks: List[TaskData]
    decisions: List[TaskDecision]
    matrix: Dict[str, List[Optional[float]]]
    routing: Dict[str, float]
    task_config: Dict[str, str]
    capacity_trace: List[int]
    active_trace: List[float]

    def forgetting(self):
        out = {}
        for name, row in self.matrix.items():
            values = [v for v in row if v is not None]
            out[name] = round(max(values) - values[-1], 3) if values else 0.0
        return out

    def interference(self):
        """Baisse moyenne des tâches anciennes à chaque nouvelle tâche."""
        drops = []
        names = list(self.matrix)
        for t in range(1, len(names)):
            for i in range(t):
                a, b = self.matrix[names[i]][t - 1], self.matrix[names[i]][t]
                if a is not None and b is not None: drops.append(max(0.0, a - b))
        return round(sum(drops) / len(drops), 4) if drops else 0.0


def routing_accuracy(model, task, config):
    names, _, _ = model.route(task.x_test)
    return sum(n == config for n in names) / len(names)


def run_lifelong(specs=None, seed=0, controller_kwargs=None):
    torch.manual_seed(seed)
    tasks, _ = build_stream(specs, seed)
    model = LifelongModel(seed=seed)
    controller = MetaController(model, seed=seed, **(controller_kwargs or {}))
    names = [t.name for t in tasks]
    matrix = {n: [None] * len(tasks) for n in names}
    task_config, capacity, active = {}, [], []
    for step, task in enumerate(tasks):
        decision = controller.observe(task.name, task.x_train, task.y_train, task.x_val, task.y_val)
        task_config[task.name] = decision.config
        for seen in tasks[:step + 1]:
            matrix[seen.name][step] = round(float((model.predict(seen.x_test) == seen.y_test).float().mean()), 3)
        capacity.append(model.state.total_parameters())
        active.append(model.active_parameters(torch.cat([t.x_test for t in tasks[:step + 1]])))
    routing = {t.name: round(routing_accuracy(model, t, task_config[t.name]), 3) for t in tasks}
    return LifelongRun(model, controller, tasks, controller.decisions, matrix, routing, task_config, capacity, active)

"""Contexte émergent + séquences symboliques : flux continu non étiqueté.

Une expérience = (x, y) sur l'alphabet commun. Le système ne reçoit AUCUNE étiquette (mode "none") :
il détecte les changements de régime, reconnaît un contexte connu, ou acquiert une nouvelle Skill
par (1) composition WCE de Skills connues, (2) raffinement structurel, (3) fast weights, (4) nouvel expert.
Les contextes internes (ctx_1, ctx_2...) émergent du comportement, pas d'un task_id.
Les familles réelles ne servent qu'à l'évaluation.
"""
from __future__ import annotations

import copy
import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from core.branching import IncrementalSnapshot, adopt, shared_branch
from core.learning import Experience
from core.model_state import (
    INPUT, AdaptWeights, AddModule, CreateConnection, ModelState, PruneUnits,
)
from core.planner import StrategyLibrary
from core.reasoning import Goal
from core.symbolic import (
    FAMILIES, L, PRIMITIVES, SEQ_INPUT, V, Family, NeuralSequenceTransformation, adapt_expert_fast, components,
    induce_refinement, one_hot, random_sequences, search_programs, sequence_accuracy, train_expert,
)
from core.transformations import MacroTransformation

FEATURES = 2 * L * V + L * L + V
K_MAX = 32


# ================================================================ encodeur de contexte appris

def pair_features(x, y):
    """Représentation relationnelle d'une expérience : entrée, résultat, alignement des positions,
    histogramme des décalages de symboles. Aucune dimension de tâche."""
    align = (y.unsqueeze(2) == x.unsqueeze(1)).float().reshape(len(x), -1)
    shift = torch.nn.functional.one_hot((y - x) % V, V).float().mean(1)
    return torch.cat([one_hot(x), one_hot(y), align, shift], dim=1)


class LearnedContextEncoder:
    """MLP 113 -> 48 -> 8 (+ tête 8 -> 16 -> 32), entraîné sur des pseudo-labels comportementaux."""

    def __init__(self, embed=8, hidden=48, seed=0):
        self.state = ModelState(FEATURES, output_module="ctx_head", max_region_fraction=1.0, max_capacity=4096)
        for spec in (AddModule(module="ctx_enc", input_dim=FEATURES, output_dim=embed, hidden=hidden,
                               activation=False, specialization="context", seed=seed),
                     AddModule(module="ctx_head", input_dim=embed, output_dim=K_MAX, hidden=16,
                               activation=False, specialization="context", seed=seed + 1),
                     CreateConnection(source=INPUT, target="ctx_enc"), CreateConnection(source="ctx_enc", target="ctx_head")):
            self.state.apply(spec)
        self.trainings = 0

    @property
    def parameter_count(self): return self.state.total_parameters()

    def embed(self, x, y):
        with torch.no_grad(): return self.state.modules["ctx_enc"].forward(pair_features(x, y))

    def train(self, feats, labels, steps=300, seed=0):
        self.state.apply(AdaptWeights(name="context_learning", modules=("ctx_enc", "ctx_head"), fast=False,
                                      inputs=feats, targets=labels, steps=steps, lr=0.01, seed=seed, local=False))
        self.trainings += 1


# ================================================================ skills et contextes

@dataclass
class Skill:
    name: str
    spec: Any
    kind: str                       # primitive | composition | neural | hybrid | refinement | fast
    parameters: int = 0
    examples: int = 0               # expériences consommées pour l'acquérir
    trained: bool = False           # gradient utilisé ?
    uses: int = 0

    @property
    def memory(self):
        m = {"episodic", "current_state"}
        if self.kind in ("primitive", "composition", "hybrid", "refinement"): m.add("structural")
        if self.kind in ("neural", "hybrid", "fast") or (self.kind == "refinement" and self.parameters): m.add("slow_parametric")
        if self.kind == "fast": m.add("fast_parametric")
        return m


@dataclass
class Context:
    cid: str
    skill: Skill
    created_step: int
    feats: List[torch.Tensor] = field(default_factory=list)
    prototype: Optional[torch.Tensor] = None
    families: Counter = field(default_factory=Counter)      # évaluation uniquement
    uses: int = 0


@dataclass
class Thresholds:
    change_errors: int = 2          # erreurs parmi la fenêtre -> changement de régime
    change_window: int = 3
    accept: float = 0.9             # précision minimale pour accepter une Skill acquise
    grow_penalty: float = 0.02      # par millier de paramètres ajoutés
    routing_confidence: float = 1.01   # au-delà, le routeur neuronal est cru sans vérification (1.01 = toujours vérifier)
    prune_tolerance: float = 0.02
    new_expert_examples: int = 128


ACQUIRE_AT = (3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160)


class SymbolicLifelongSystem:
    """mode : 'labels' (A), 'boundaries' (B), 'none' (C, continu, sans aucune étiquette)."""

    def __init__(self, mode="none", *, structural=True, neural=True, encoder=True, thresholds=None, seed=0):
        self.mode, self.use_structural, self.use_neural, self.use_encoder = mode, structural, neural, encoder
        self.th = thresholds or Thresholds()
        self.seed = seed
        self.state = ModelState(SEQ_INPUT, max_region_fraction=1.0, max_capacity=100000, max_connections=256)
        self.state.metadata["routes"] = {}
        self.encoder = LearnedContextEncoder(seed=seed) if encoder else None
        self.skills: Dict[str, Skill] = {}
        self.contexts: Dict[str, Context] = {}
        self.library = StrategyLibrary()
        self.episodic: List[Experience] = []
        self.lexicon: Dict[str, Counter] = {}
        self.current: Optional[str] = None
        self.recent = deque(maxlen=self.th.change_window)
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.acquiring = True
        self.step = 0
        self.label_contexts: Dict[str, str] = {}
        self.label_current = None
        self.snapshots: List[IncrementalSnapshot] = []
        self.events: List[Dict[str, Any]] = []
        self.changes: List[int] = []
        self.online: List[bool] = []
        self.capacity_curve: List[Dict[str, Any]] = []
        self.evaluations = {"identify": 0, "search_nodes": 0}
        self.stability: List[float] = []
        self._next_ctx = 1
        self._next_expert = 1

    # ---- bibliothèque WCE --------------------------------------------------
    def atoms(self):
        atoms = list(PRIMITIVES.values()) if self.use_structural else []
        atoms += [s.spec for s in self.skills.values() if s.kind == "neural"]
        return atoms

    def experts(self): return [n for n in self.state.modules]

    def total_parameters(self):
        return self.state.total_parameters() + (self.encoder.parameter_count if self.encoder else 0)

    def active_parameters(self, cid):
        skill = self.contexts[cid].skill
        return (self.encoder.parameter_count if self.encoder else 0) + skill.parameters

    # ---- identification ------------------------------------------------------
    def _consistent(self, cid, xs, ys):
        self.evaluations["identify"] += 1
        return float((self.contexts[cid].skill.spec.apply(xs) == ys).all(1).float().mean())

    def neural_ranking(self, xs, ys):
        """Classement des contextes par la tête apprise de l'encodeur (moyenne des log-probas sur les paires)."""
        if not self.encoder or self.encoder.trainings == 0: return [], 0.0
        ids = list(self.contexts)[:K_MAX]
        with torch.no_grad():
            logits = self.encoder.state.forward(pair_features(xs, ys))[:, :len(ids)]
            score = torch.log_softmax(logits, dim=1).mean(0)
        probs = torch.softmax(score, 0)
        order = torch.argsort(score, descending=True).tolist()
        return [ids[i] for i in order], float(probs[order[0]])

    def identify(self, xs, ys, strategy="hybrid", tolerant=False):
        """neural : prototype le plus proche ; structural : vérification exhaustive ;
        hybrid : ordre neuronal + vérification structurelle (arrêt au premier cohérent)."""
        if not self.contexts: return None
        if strategy == "neural" or (strategy == "hybrid" and not self.use_structural and self.encoder):
            ranking, _ = self.neural_ranking(xs, ys)
            return ranking[0] if ranking else None
        if strategy == "structural" or not self.encoder:
            order = list(self.contexts)
        else:
            ranking, confidence = self.neural_ranking(xs, ys)
            if ranking and confidence >= self.th.routing_confidence: return ranking[0]
            order = ranking + [c for c in self.contexts if c not in ranking]
        scores = {}
        for cid in order:
            scores[cid] = self._consistent(cid, xs, ys)
            if scores[cid] >= 1.0: return cid
        # second passage tolérant (experts neuronaux imparfaits) : jamais sur une seule paire
        if tolerant and len(xs) >= 3 and scores:
            best = max(scores, key=scores.get)
            if scores[best] >= 2 / 3: return best
        return None

    # ---- API publique --------------------------------------------------------
    def predict(self, x, support=None, token=None):
        """Aucune étiquette de tâche : commande (lexique appris), exemples de support, ou contexte courant."""
        cid = None
        if token is not None and token in self.lexicon: cid = self.lexicon[token].most_common(1)[0][0]
        elif support is not None:
            xs, ys = support
            cid = self.identify(xs, ys, tolerant=True)
        else:
            cid = self.current
        if cid is None: return x.clone()
        return self.contexts[cid].skill.spec.apply(x)

    # ---- flux ------------------------------------------------------------------
    def observe(self, x, y, family=None, boundary=False, token=None):
        """Une expérience. family n'est utilisé qu'en mode 'labels' (et pour l'évaluation)."""
        self.step += 1
        x, y = x.reshape(1, L), y.reshape(1, L)
        correct = False
        if self.current is not None and not self.acquiring:
            correct = bool((self.contexts[self.current].skill.spec.apply(x) == y).all())
        self.online.append(correct)

        if self.mode == "labels":
            if family != self.label_current:
                self.label_current = family
                if family in self.label_contexts:
                    self.current, self.acquiring = self.label_contexts[family], False
                else:
                    self.current, self.acquiring, self.buffer = None, True, []
        elif self.mode == "boundaries" and boundary:
            self.current, self.acquiring, self.buffer = None, True, []
        elif self.mode == "none" and not self.acquiring:
            self.recent.append((x, y, correct))
            if sum(not c for _, _, c in self.recent) >= self.th.change_errors:
                self.changes.append(self.step)
                self.buffer = [(a, b) for a, b, c in self.recent if not c]
                self.current, self.acquiring = None, True
                self.recent.clear()
                self.events.append({"step": self.step, "event": "change_point"})

        if self.acquiring:
            if not self.buffer or not any(torch.equal(x, a) and torch.equal(y, b) for a, b in self.buffer[-1:]):
                self.buffer.append((x, y))
            self._resolve(family, token)
        else:
            ctx = self.contexts[self.current]
            if correct:
                ctx.uses += 1; ctx.skill.uses += 1
                if len(ctx.feats) < 64: ctx.feats.append(pair_features(x, y))
                if token: self.lexicon.setdefault(token, Counter())[self.current] += 1
            if family: ctx.families[family] += 1
        return correct

    def _buffer_tensors(self):
        return torch.cat([a for a, _ in self.buffer]), torch.cat([b for _, b in self.buffer])

    def _resolve(self, family, token):
        xs, ys = self._buffer_tensors()
        n = len(xs)
        if self.mode != "labels" and n >= 2 and self.contexts:
            tail = slice(max(0, n - 3), n)
            cid = self.identify(xs[tail], ys[tail])
            if cid is not None:
                self.buffer = self.buffer[tail]
                self._activate(cid, family, token, "known_context")
                return
        if n in ACQUIRE_AT or (n > ACQUIRE_AT[-1] and n % 40 == 0):
            skill = self._acquire(xs, ys)
            if skill is None and n > 4:
                # le tampon peut mélanger deux régimes : on tente la fin du tampon seule
                skill = self._acquire(xs[-4:], ys[-4:], structural_only=True)
                if skill is not None: self.buffer = self.buffer[-4:]
            if skill is not None:
                xs, ys = self._buffer_tensors()
                cid = self._create_context(skill, xs, ys, family)
                self._activate(cid, family, token, "new_context")
            elif n >= ACQUIRE_AT[-1] + 40:
                self.buffer = self.buffer[-ACQUIRE_AT[-1]:]      # fenêtre glissante

    def _activate(self, cid, family, token, event):
        self.current, self.acquiring = cid, False
        ctx = self.contexts[cid]
        xs, ys = self._buffer_tensors()
        for i in range(len(xs)):
            if len(ctx.feats) < 64: ctx.feats.append(pair_features(xs[i:i + 1], ys[i:i + 1]))
        if family:
            ctx.families[family] += len(xs)
            self.label_contexts.setdefault(family, cid)
        if token: self.lexicon.setdefault(token, Counter())[cid] += 1
        self.events.append({"step": self.step, "event": event, "context": cid, "skill": ctx.skill.name,
                            "kind": ctx.skill.kind, "examples": len(xs)})
        self.buffer = []

    # ---- acquisition -----------------------------------------------------------
    def _acquire(self, xs, ys, structural_only=False):
        n = len(xs)
        candidates = []
        if n >= 3 and self.use_structural:
            result = search_programs(xs, ys, self.atoms(), max_depth=2, min_fraction=1.0 if n <= 4 else 0.85)
            self.evaluations["search_nodes"] += result.explored
            if result.found:
                parts = components(result.program)
                neural_parts = [p for p in parts if isinstance(p, NeuralSequenceTransformation)]
                kind = "hybrid" if neural_parts else ("composition" if len(parts) > 1 else "primitive")
                params = sum(p.resolved.parameter_count for p in neural_parts)
                acc = sequence_accuracy(result.program.apply(xs), ys)
                candidates.append((acc, Skill(result.program.name, result.program, kind, params, n, False), 0))
            for skill in list(self.skills.values()):
                refined = induce_refinement(xs, ys, skill.spec)
                if refined is not None:
                    acc = sequence_accuracy(refined.apply(xs), ys)
                    candidates.append((acc, Skill(refined.name, refined, "refinement", skill.parameters, n, False), 0))
        if self.use_neural and n >= 16 and not structural_only:
            cut = max(8, int(0.75 * n))
            neural = [s for s in self.skills.values() if s.kind == "neural"]
            if neural:
                base = max(neural, key=lambda s: float((s.spec.apply(xs) == ys).float().mean()))
                delta = adapt_expert_fast(self.state, base.spec.module_name, xs[:cut], ys[:cut], seed=self.seed)
                spec = NeuralSequenceTransformation(name=f"{base.name}+fast", state=self.state,
                                                    module_name=base.spec.module_name, fast=delta)
                acc = sequence_accuracy(spec.apply(xs[cut:]), ys[cut:])
                added = sum(t.numel() for t in delta.values())
                candidates.append((acc, Skill(spec.name, spec, "fast", base.parameters + added, n, True), added))
            if n >= self.th.new_expert_examples:
                name = f"expert_{self._next_expert}"
                trial = shared_branch(self.state)
                spec = train_expert(trial, name, xs[:cut], ys[:cut], seed=self.seed + self._next_expert)
                acc = sequence_accuracy(spec.apply(xs[cut:]), ys[cut:])
                candidates.append((acc, Skill(name, spec, "neural", spec.resolved.parameter_count, n, True),
                                   spec.resolved.parameter_count, trial))
        best, best_score = None, -1.0
        for cand in candidates:
            acc, skill, added = cand[0], cand[1], cand[2]
            score = acc - self.th.grow_penalty * added / 1000
            if acc >= self.th.accept and score > best_score: best, best_score = cand, score
        if best is None: return None
        skill = best[1]
        if skill.kind == "neural":
            adopt(self.state, best[3]); self._next_expert += 1
            skill.spec = NeuralSequenceTransformation(name=skill.name, state=self.state, module_name=skill.name)
            self._prune_expert(skill, xs, ys)
            # consolidation : la validation a réservé 25 % du tampon ; on les rend à l'expert accepté
            self.state.apply(AdaptWeights(name="consolidate_expert", modules=(skill.name,), fast=False, output=skill.name,
                                          positions=L, inputs=one_hot(xs), targets=ys, steps=200, lr=0.01, local=False))
            skill.parameters = self.state.modules[skill.name].parameter_count
        return skill

    def _prune_expert(self, skill, xs, ys, keeps=(48, 32, 16)):
        cut = max(8, int(0.75 * len(xs)))
        module = skill.name
        ref = sequence_accuracy(skill.spec.apply(xs[cut:]), ys[cut:])
        for keep in keeps:
            trial = shared_branch(self.state)
            trial.apply(PruneUnits(module=module, keep=keep))
            trial.apply(AdaptWeights(name="recover_after_prune", modules=(module,), fast=False, output=module, positions=L,
                                     inputs=one_hot(xs[:cut]), targets=ys[:cut], steps=150, lr=0.01, local=False))
            acc = sequence_accuracy(NeuralSequenceTransformation(module=trial.modules[module]).apply(xs[cut:]), ys[cut:])
            if acc < ref - self.th.prune_tolerance: break
            adopt(self.state, trial)

    def _create_context(self, skill, xs, ys, family):
        existing = next((s for s in self.skills.values() if s.name == skill.name), None)
        if existing is None:
            self.skills[skill.name] = skill
            self.library.store(Goal("transform", skill.name), MacroTransformation(name=skill.name, base_transformation=skill.spec,
                                                                                    domain=skill.spec.domain))
        cid = f"ctx_{self._next_ctx}"; self._next_ctx += 1
        self.contexts[cid] = Context(cid, existing or skill, self.step)
        self.episodic.append(Experience(initial_state=[tuple(x.tolist()) for x in xs[:3]], goal=Goal("transform", cid),
                                        successful_plan=skill.name, bindings={"context": cid}, cost=skill.parameters,
                                        explored_nodes=len(xs), result="acquired", strategy_used=skill.kind))
        self.snapshots.append(IncrementalSnapshot(f"after_{cid}", self.state, self.snapshots[-1] if self.snapshots else None))
        for i in range(len(xs)):
            if len(self.contexts[cid].feats) < 64: self.contexts[cid].feats.append(pair_features(xs[i:i + 1], ys[i:i + 1]))
        self._train_encoder()
        families = {f for c in self.contexts.values() for f in c.families} | ({family} if family else set())
        self.capacity_curve.append({"contexts": len(self.contexts), "skills": len(self.skills),
                                    "experts": len(self.state.modules), "total_parameters": self.total_parameters(),
                                    "active_parameters": self.active_parameters(cid), "families": len(families)})
        return cid

    def _train_encoder(self):
        if not self.encoder or len(self.contexts) < 2: return
        ids = {cid: i for i, cid in enumerate(self.contexts)}
        feats, labels = [], []
        for cid, ctx in self.contexts.items():
            if ctx.feats:
                feats.append(torch.cat(ctx.feats)); labels.append(torch.full((len(ctx.feats),), ids[cid] % K_MAX))
        if not feats: return
        feats, labels = torch.cat(feats), torch.cat(labels)
        before = self._assignments(feats)
        self.encoder.train(feats, labels, seed=self.seed)
        self._update_prototypes()
        after = self._assignments(feats)
        if before is not None:
            old = [i for i, a in enumerate(before) if a is not None]
            if old: self.stability.append(sum(before[i] == after[i] for i in old) / len(old))

    def _update_prototypes(self):
        for ctx in self.contexts.values():
            if ctx.feats:
                with torch.no_grad(): ctx.prototype = self.encoder.state.modules["ctx_enc"].forward(torch.cat(ctx.feats)).mean(0)

    def _assignments(self, feats):
        protos = [(cid, c.prototype) for cid, c in self.contexts.items() if c.prototype is not None]
        if not protos: return None
        with torch.no_grad(): emb = self.encoder.state.modules["ctx_enc"].forward(feats)
        dist = torch.cdist(emb, torch.stack([p for _, p in protos]))
        return [protos[i][0] for i in dist.argmin(1).tolist()]


# ================================================================ flux

@dataclass
class Segment:
    family: str
    length: int
    token_steps: int = 0            # nombre d'expériences initiales accompagnées d'une commande


def default_segments():
    S = []
    for name in ("identity", "reverse", "rotl1", "rotr1", "rotl2", "sub1", "sub3", "swap", "sort"):
        S.append(Segment(name, 24, token_steps=12))
    for name in ("reverse>rotl1", "sub1>reverse", "swap>rotl1", "rotl2>sub2"): S.append(Segment(name, 24))
    S += [Segment("perm_A", 180), Segment("map_A", 180), Segment("perm_A>reverse", 24), Segment("map_A>rotl1", 24),
          Segment("perm_B", 180), Segment("map_B", 180), Segment("sub1>perm_B", 24), Segment("map_A2", 40)]
    S += [Segment("reverse", 20), Segment("perm_A", 20), Segment("swap>rotl1", 20), Segment("map_B", 20)]
    return S


def small_segments():
    return [Segment("reverse", 24, 12), Segment("rotl1", 24, 12), Segment("sub1", 24), Segment("reverse>rotl1", 24),
            Segment("perm_A", 180), Segment("perm_A>reverse", 24), Segment("map_A", 180), Segment("map_A2", 40),
            Segment("reverse", 20), Segment("perm_A", 20)]


def stream(segments, seed=0):
    gen = torch.Generator().manual_seed(seed)
    for index, seg in enumerate(segments):
        fam = FAMILIES[seg.family]
        x = random_sequences(seg.length, gen)
        y = fam.apply(x)
        for i in range(seg.length):
            yield {"x": x[i], "y": y[i], "family": seg.family, "segment": index, "boundary": i == 0,
                   "token": fam.token if i < seg.token_steps else None}


@dataclass
class StreamRun:
    system: Any
    segments: List[Segment]
    acquired_accuracy: Dict[str, float]
    final_accuracy: Dict[str, float]
    true_changes: List[int]
    seconds: float

    def forgetting(self):
        return {f: round(max(0.0, self.acquired_accuracy[f] - self.final_accuracy[f]), 3) for f in self.final_accuracy}


def evaluate_family(system, family, n_support=3, n_query=64, seed=123):
    gen = torch.Generator().manual_seed(seed)
    fam = FAMILIES[family]
    xs = random_sequences(n_support, gen); xq = random_sequences(n_query, gen)
    return sequence_accuracy(system.predict(xq, support=(xs, fam.apply(xs))), fam.apply(xq))


def run_stream(segments=None, mode="none", seed=0, **kwargs):
    segments = segments or default_segments()
    system = SymbolicLifelongSystem(mode=mode, seed=seed, **kwargs)
    t = time.time()
    acquired, true_changes, step = {}, [], 0
    last_segment = None
    for exp in stream(segments, seed):
        step += 1
        if exp["segment"] != last_segment:
            if last_segment is not None:
                prev = segments[last_segment].family
                acquired.setdefault(prev, evaluate_family(system, prev))
                true_changes.append(step)
            last_segment = exp["segment"]
        system.observe(exp["x"], exp["y"], family=exp["family"] if mode == "labels" else None,
                       boundary=exp["boundary"] and mode == "boundaries", token=exp["token"])
        _track_family(system, exp["family"])
    acquired.setdefault(segments[-1].family, evaluate_family(system, segments[-1].family))
    final = {s.family: evaluate_family(system, s.family) for s in segments}
    return StreamRun(system, segments, acquired, final, true_changes, time.time() - t)


def _track_family(system, family):
    """Évaluation uniquement : note quelle famille a alimenté quel contexte (pureté)."""
    if system.current is not None and not system.acquiring and system.mode != "labels":
        system.contexts[system.current].families[family] += 1


# ================================================================ métriques

def change_point_scores(run, tolerance=6):
    detected, truth = run.system.changes, run.true_changes
    hits = [t for t in truth if any(t <= d <= t + tolerance for d in detected)]
    matched = [d for d in detected if any(t <= d <= t + tolerance for t in truth)]
    precision = len(matched) / len(detected) if detected else 0.0
    recall = len(hits) / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    delays = [min(d - t for d in detected if t <= d <= t + tolerance) for t in hits]
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "mean_delay": round(sum(delays) / len(delays), 2) if delays else None}


def cluster_purity(system):
    total = sum(sum(c.families.values()) for c in system.contexts.values())
    if not total: return 0.0
    return round(sum(max(c.families.values()) for c in system.contexts.values() if c.families) / total, 3)


def routing_scores(system, families, n=64, seed=321, pairs=1):
    """Précision d'identification sans étiquette (la bonne Skill ≡ même comportement que la famille)."""
    out = {}
    gen = torch.Generator().manual_seed(seed)
    probe = random_sequences(n, gen)
    for strategy in ("neural", "structural", "hybrid"):
        correct = total = 0
        evaluations_before = system.evaluations["identify"]
        for fam_name in families:
            fam = FAMILIES[fam_name]
            target = fam.apply(probe)
            for i in range(0, n - pairs + 1, pairs):
                xs = probe[i:i + pairs]; ys = fam.apply(xs)
                if strategy == "neural" and not system.encoder: break
                cid = system.identify(xs, ys, strategy)
                total += 1
                if cid is not None and sequence_accuracy(system.contexts[cid].skill.spec.apply(probe), target) >= 0.9: correct += 1
        out[strategy] = round(correct / total, 3) if total else None
        out[f"{strategy}_evaluations"] = system.evaluations["identify"] - evaluations_before
    return out


def novelty_scores(run):
    """Pour chaque début de segment : nouveau vs connu, comparé à la vérité (famille déjà vue ?)."""
    seen, correct, total = set(), 0, 0
    events = [e for e in run.system.events if e["event"] in ("new_context", "known_context")]
    steps = [sum(s.length for s in run.segments[:i]) + 1 for i in range(len(run.segments))]
    for i, seg in enumerate(run.segments):
        start, end = steps[i], steps[i] + seg.length - 1
        first = next((e for e in events if start <= e["step"] <= end), None)
        if first is not None:
            truth = "known_context" if seg.family in seen else "new_context"
            total += 1; correct += first["event"] == truth
        seen.add(seg.family)
    return round(correct / total, 3) if total else 0.0


def summarize(run):
    s = run.system
    online = s.online
    kinds = Counter(c.skill.kind for c in s.contexts.values())
    comp = [c for c in s.contexts.values() if c.skill.kind in ("composition", "hybrid")]
    families = sorted({seg.family for seg in run.segments})
    memory = Counter(m for c in s.contexts.values() for m in c.skill.memory)
    return {
        "families": len(families), "contexts": len(s.contexts), "skills": len(s.skills),
        "experts": len(s.state.modules), "kinds": dict(kinds),
        "final_accuracy": round(sum(run.final_accuracy.values()) / len(run.final_accuracy), 3),
        "online_accuracy": round(sum(online) / len(online), 3),
        "max_forgetting": max(run.forgetting().values()) if run.forgetting() else 0.0,
        "total_parameters": s.total_parameters(),
        "average_active_parameters": round(sum(s.active_parameters(c) for c in s.contexts) / max(1, len(s.contexts)), 1),
        "tasks_per_expert": round(len(families) / max(1, len(s.state.modules)), 2),
        "skills_per_primitive": round(len(s.skills) / len(PRIMITIVES), 2),
        "zero_shot_compositions": sum(1 for c in comp if not c.skill.trained and c.skill.examples <= 4),
        "composition_reuse": sum(1 for c in comp),
        "cluster_purity": cluster_purity(s),
        "change_points": change_point_scores(run) if s.mode == "none" else None,
        "novelty_accuracy": novelty_scores(run),
        "snapshot_bytes_incremental": sum(sn.owned_bytes() for sn in s.snapshots),
        "snapshot_bytes_full": sum(sn.full_bytes() for sn in s.snapshots),
        "embedding_stability": round(sum(s.stability) / len(s.stability), 3) if s.stability else None,
        "memory_usage": dict(memory),
        "search_nodes": s.evaluations["search_nodes"],
        "seconds": round(run.seconds, 1),
    }


# ================================================================ baseline statique

class StaticBaseline:
    """Un seul réseau entraîné en continu, sans routage ni contexte : le 'neural static baseline'."""

    def __init__(self, hidden=128, seed=0, every=16, steps=30):
        self.state = ModelState(SEQ_INPUT, output_module="static", max_region_fraction=1.0)
        self.state.apply(AddModule(module="static", input_dim=SEQ_INPUT, output_dim=SEQ_INPUT, hidden=hidden,
                                   activation=False, seed=seed))
        self.state.apply(CreateConnection(source=INPUT, target="static"))
        self.every, self.steps, self.seed = every, steps, seed
        self.memory = deque(maxlen=64)
        self.step = 0
        self.online = []

    def predict(self, x, support=None, token=None):
        return NeuralSequenceTransformation(module=self.state.modules["static"]).apply(x)

    def observe(self, x, y, **_):
        self.step += 1
        x, y = x.reshape(1, L), y.reshape(1, L)
        self.online.append(bool((self.predict(x) == y).all()))
        self.memory.append((x, y))
        if self.step % self.every == 0:
            xs = torch.cat([a for a, _ in self.memory]); ys = torch.cat([b for _, b in self.memory])
            self.state.apply(AdaptWeights(modules=("static",), fast=False, output="static", positions=L, inputs=one_hot(xs),
                                          targets=ys, steps=self.steps, lr=0.01, seed=self.seed, local=False))

    def total_parameters(self): return self.state.total_parameters()


def run_static(segments=None, seed=0):
    segments = segments or default_segments()
    model = StaticBaseline(seed=seed)
    t = time.time()
    for exp in stream(segments, seed): model.observe(exp["x"], exp["y"])
    final = {s.family: evaluate_family(model, s.family) for s in segments}
    return {"final_accuracy": round(sum(final.values()) / len(final), 3),
            "online_accuracy": round(sum(model.online) / len(model.online), 3),
            "total_parameters": model.total_parameters(), "experts": 1, "per_family": final,
            "seconds": round(time.time() - t, 1)}


def run_ablations(segments=None, seed=0):
    segments = segments or default_segments()
    results = {"1_neural_static": run_static(segments, seed)}
    configs = {
        "2_neural_plasticity_only": dict(mode="none", structural=False),
        "3_wce_structural_only": dict(mode="none", neural=False),
        "4_wce_plasticity_labels": dict(mode="labels"),
        "5_wce_plasticity_autonomous": dict(mode="none"),
    }
    for name, cfg in configs.items():
        summary = summarize(run_stream(segments, seed=seed, **cfg))
        results[name] = {k: summary[k] for k in ("final_accuracy", "online_accuracy", "total_parameters", "experts",
                                                   "skills", "contexts", "seconds")}
    return results


def label_ablation(segments=None, seed=0):
    segments = segments or default_segments()
    out = {}
    for mode in ("labels", "boundaries", "none"):
        s = summarize(run_stream(segments, mode=mode, seed=seed))
        out[mode] = {k: s[k] for k in ("final_accuracy", "online_accuracy", "contexts", "cluster_purity",
                                       "novelty_accuracy", "change_points", "experts")}
    return out


SENSITIVITY_GRID = {
    "change_errors": (1, 2, 3),
    "accept": (0.8, 0.9, 0.97),
    "grow_penalty": (0.0, 0.02, 0.5),
    "routing_confidence": (0.5, 0.95, 1.01),
    "prune_tolerance": (0.0, 0.02, 0.1),
}


def sensitivity(segments=None, seed=0, grid=None):
    """Une variable à la fois autour des valeurs par défaut ; 'stable' = à 2 points du meilleur."""
    segments = segments or small_segments()
    grid = grid or SENSITIVITY_GRID
    table = {}
    for param, values in grid.items():
        rows = []
        for value in values:
            th = Thresholds(**{param: value})
            s = summarize(run_stream(segments, seed=seed, thresholds=th))
            rows.append({"value": value, "final_accuracy": s["final_accuracy"], "online_accuracy": s["online_accuracy"],
                         "contexts": s["contexts"], "experts": s["experts"], "total_parameters": s["total_parameters"],
                         "f1": s["change_points"]["f1"]})
        best = max(r["final_accuracy"] for r in rows)
        for r in rows: r["stable"] = r["final_accuracy"] >= best - 0.02
        table[param] = rows
    return table


def few_shot_curve(family="map_A2", base="map_A", shots=(2, 4, 8, 16, 32, 64, 128), seed=0):
    """Précision selon le nombre d'exemples : raffinement WCE vs fast weights vs expert neuf."""
    from core.symbolic import sample_family
    state = ModelState(SEQ_INPUT, max_region_fraction=1.0, max_capacity=100000)
    xb, yb = sample_family(FAMILIES[base], 512, seed=seed)
    base_spec = train_expert(state, "base", xb, yb, seed=seed)
    xt, yt = sample_family(FAMILIES[family], 256, seed=seed + 7)
    rows = []
    for k in shots:
        xs, ys = sample_family(FAMILIES[family], k, seed=seed + 11)
        refined = induce_refinement(xs, ys, base_spec)
        ref_acc = sequence_accuracy(refined.apply(xt), yt) if refined else 0.0
        delta = adapt_expert_fast(state, "base", xs, ys, seed=seed)
        fast_acc = sequence_accuracy(NeuralSequenceTransformation(module=state.modules["base"], fast=delta).apply(xt), yt)
        scratch = ModelState(SEQ_INPUT, max_region_fraction=1.0, max_capacity=100000)
        new_acc = sequence_accuracy(train_expert(scratch, "new", xs, ys, seed=seed).apply(xt), yt)
        rows.append({"shots": k, "wce_refinement": round(ref_acc, 3), "fast_weights": round(fast_acc, 3),
                     "new_expert": round(new_acc, 3)})
    return rows


def zero_shot_composition(seed=0):
    """Reverse et RotateLeft appris séparément ; Reverse>RotateLeft demandé sans entraînement spécifique."""
    from core.symbolic import sample_family
    system = SymbolicLifelongSystem(seed=seed)
    results = {}
    for name in ("reverse", "rotl1"):
        x, y = sample_family(FAMILIES[name], 3, seed=seed)
        for i in range(3): system.observe(x[i], y[i])
    x, y = sample_family(FAMILIES["reverse>rotl1"], 3, seed=seed + 1)
    for i in range(3): system.observe(x[i], y[i])
    skill = system.contexts[system.current].skill if system.current else None
    results["wce_composition"] = {"skill": skill.name if skill else None, "trained": skill.trained if skill else None,
                                  "examples": skill.examples if skill else None,
                                  "accuracy": evaluate_family(system, "reverse>rotl1")}
    xs, ys = sample_family(FAMILIES["reverse>rotl1"], 3, seed=seed + 1)
    state = ModelState(SEQ_INPUT, max_region_fraction=1.0, max_capacity=100000)
    xt, yt = sample_family(FAMILIES["reverse>rotl1"], 256, seed=seed + 9)
    results["network_alone_3_examples"] = sequence_accuracy(train_expert(state, "n", xs, ys, seed=seed).apply(xt), yt)
    xs, ys = sample_family(FAMILIES["reverse>rotl1"], 128, seed=seed + 2)
    state = ModelState(SEQ_INPUT, max_region_fraction=1.0, max_capacity=100000)
    results["network_alone_128_examples"] = sequence_accuracy(train_expert(state, "n", xs, ys, seed=seed).apply(xt), yt)
    return results


# ================================================================ commandes / ambiguïté

def token_scores(system, n=64, seed=77):
    """Commande seule (sans support) vs rien : dépendance au signal structurel explicite."""
    gen = torch.Generator().manual_seed(seed)
    x = random_sequences(n, gen)
    out = {}
    for fam in FAMILIES.values():
        if fam.token and fam.token in system.lexicon:
            out[fam.token] = {"family": fam.name,
                              "token_only": sequence_accuracy(system.predict(x, token=fam.token), fam.apply(x)),
                              "no_token_no_support": sequence_accuracy(system.predict(x), fam.apply(x))}
    return out


def pairs_to_disambiguate(families=None, trials=200, seed=5, max_pairs=6):
    """Nombre moyen de paires (x, y) nécessaires pour qu'une seule famille reste cohérente."""
    families = families or FAMILIES
    gen = torch.Generator().manual_seed(seed)
    names = list(families)
    counts = []
    for t in range(trials):
        true = names[t % len(names)]
        x = random_sequences(max_pairs, gen)
        y = families[true].apply(x)
        alive = set(names)
        for k in range(1, max_pairs + 1):
            alive = {f for f in alive if torch.equal(families[f].apply(x[k - 1:k]), y[k - 1:k])}
            if len(alive) == 1: counts.append(k); break
        else:
            counts.append(max_pairs)
    return round(sum(counts) / len(counts), 2)

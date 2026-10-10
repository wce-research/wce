"""SkillNet: MODULAR network of skills as a WCE state, for compositional plasticity.

- Input = data bits (8) + context cue (12 noisy bits). The task number is NEVER given:
  the router DISCOVERS the contexts (cue prototypes; novelty if too far from all prototypes).
- Modules = small experts (MLP 8→h→2, slow weights + fast weights). Skills = PROGRAMS over the modules:
  ("expert", m) | ("not", s) | ("compose", op, s1, s2) — symbolic composition, parameter-free.
- Every modification goes through an AdaptationTransformation (TransformationSpec) applied by `SkillNet.apply`,
  traced in the TransformGraph (15 families, part A).
- FUNCTIONAL memory: invariant certificates + synthesised canonical probes (no raw data).
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from core.transform_graph import TransformGraph, TransformNode
from core.transformations import TransformationDomain, TransformationSpec

D_DATA, D_CUE = 8, 12
ALL_DATA = ((torch.arange(2 ** D_DATA).unsqueeze(1) >> torch.arange(D_DATA)) & 1).float()


# ================================================================ tasks (structured boolean functions)

def eval_spec(spec, X):
    """Labels (long) of a task specification on data bits X [N, 8]."""
    k = spec[0]
    if k == "bit": return X[:, spec[1]].long()
    if k == "xor": return (X[:, spec[1]] != X[:, spec[2]]).long()
    if k == "and": return (X[:, spec[1]] * X[:, spec[2]]).long()
    if k == "or": return ((X[:, spec[1]] + X[:, spec[2]]) > 0).long()
    if k == "maj": return (X[:, list(spec[1:])].sum(1) >= 2).long()
    if k == "parity": return (X[:, list(spec[1:])].sum(1) % 2).long()
    if k == "not": return 1 - eval_spec(spec[1], X)
    if k == "comp": return combine(spec[1], eval_spec(spec[2], X), eval_spec(spec[3], X))
    raise KeyError(k)


def combine(op, a, b):
    if op == "and": return a * b
    if op == "or": return ((a + b) > 0).long()
    if op == "xor": return (a != b).long()
    raise KeyError(op)


OPS = ("and", "or", "xor")


@dataclass
class Task:
    name: str
    spec: tuple
    cue: torch.Tensor                     # cue prototype (hidden from the system: only noisy samples are seen)
    kind: str = "new"                     # new | composed | recurrence | variant

    def sample(self, n, rng: random.Random, noise=0.05):
        g = torch.Generator().manual_seed(rng.randrange(10 ** 9))
        idx = torch.randint(0, len(ALL_DATA), (n,), generator=g); X = ALL_DATA[idx]
        flips = (torch.rand(n, D_CUE, generator=g) < noise).float()
        C = (self.cue.expand(n, D_CUE) + flips) % 2
        return X, C, eval_spec(self.spec, X)

    def test_set(self, seed=0, noise=0.05):
        g = torch.Generator().manual_seed(seed)
        X = ALL_DATA; flips = (torch.rand(len(X), D_CUE, generator=g) < noise).float()
        return X, (self.cue.expand(len(X), D_CUE) + flips) % 2, eval_spec(self.spec, X)


def _new_cue(rng, used, min_dist=4):
    for _ in range(1000):
        c = torch.tensor([float(rng.random() < 0.5) for _ in range(D_CUE)])
        if all(float((c - u).abs().sum()) >= min_dist for u in used): return c
    raise RuntimeError("no free cue")


def _primitive(rng):
    i, j, k = rng.sample(range(D_DATA), 3)
    return rng.choice([("xor", i, j), ("and", i, j), ("or", i, j), ("maj", i, j, k), ("parity", i, j, k), ("bit", i)])


def task_stream(seed=0, n_tasks=14, p_recur=0.15, p_comp=0.3, p_variant=0.1):
    """Sequence of tasks: new primitives, compositions of past tasks, negations / variants,
    RECURRENCES (same cue, same function). The identifiers are never provided to the system."""
    rng = random.Random(seed); tasks, cues = [], []
    while len(tasks) < n_tasks:
        u = rng.random(); prev = [t for t in tasks if t.kind != "recurrence"]
        if len(prev) >= 3 and u < p_recur:
            t = rng.choice(prev); tasks.append(Task(t.name + "*", t.spec, t.cue, "recurrence")); continue
        if len(prev) >= 2 and u < p_recur + p_comp:
            a, b = rng.sample(prev, 2); spec = ("comp", rng.choice(OPS), a.spec, b.spec); kind = "composed"
        elif len(prev) >= 1 and u < p_recur + p_comp + p_variant:
            spec = ("not", rng.choice(prev).spec); kind = "variant"
        else: spec, kind = _primitive(rng), "new"
        y = eval_spec(spec, ALL_DATA).float().mean()
        if not 0.15 < float(y) < 0.85: continue                      # tasks that are too unbalanced: rejected
        c = _new_cue(rng, cues); cues.append(c)
        tasks.append(Task(f"T{len(tasks)}", spec, c, kind))
    return tasks


# ================================================================ modules and skills

PARAMS = ("w1", "b1", "w2", "b2")


def new_expert(hidden, seed):
    g = torch.Generator().manual_seed(seed)
    return {"w1": torch.randn(hidden, D_DATA, generator=g) * (2.0 / D_DATA) ** 0.5, "b1": torch.zeros(hidden),
            "w2": torch.randn(2, hidden, generator=g) * (1.0 / hidden) ** 0.5, "b2": torch.zeros(2)}


def expert_forward(p, X):
    return torch.relu(X @ p["w1"].T + p["b1"]) @ p["w2"].T + p["b2"]


def n_params(p): return sum(v.numel() for v in p.values())


@dataclass
class Context:
    name: str
    proto: torch.Tensor                       # mean of the observed cues
    count: int
    skill: Optional[str] = None


class SkillNet:
    def __init__(self, capacity=4000, novelty=2.5, seed=0):
        self.modules: Dict[str, Dict[str, torch.Tensor]] = {}
        self.fast: Dict[str, Dict[str, torch.Tensor]] = {}
        self.frozen: Dict[str, Dict[str, torch.Tensor]] = {}           # bases of frozen subspaces (per module)
        self.allocated: Dict[str, Dict[str, torch.Tensor]] = {}        # subspaces allocated to adaptation
        self.skills: Dict[str, tuple] = {}
        self.contexts: List[Context] = []
        self.memories: Dict[str, Any] = {}                             # functional memory (certificates, probes)
        self.capacity, self.novelty, self.seed = capacity, novelty, seed
        self.counters = {"transformations": 0, "grown_modules": 0, "reused": 0, "composed": 0, "rollbacks": 0, "train_steps": 0}
        self.graph = TransformGraph(TransformNode("S0", configuration={}))

    # ---- views -------------------------------------------------------------------------------
    def parameter_count(self): return sum(n_params(p) for p in self.modules.values())
    def module_users(self, m): return [s for s, prog in self.skills.items() if m in skill_modules(self, s)]

    def skill_logits(self, skill, X):
        prog = self.skills[skill]; k = prog[0]
        if k == "expert":
            p = {n: self.modules[prog[1]][n] + self.fast.get(prog[1], {}).get(n, 0) for n in PARAMS}
            return expert_forward(p, X)
        if k == "not":
            y = 1 - self.skill_logits(prog[1], X).argmax(1); return torch.stack([(1 - y).float(), y.float()], 1) * 10.0
        if k == "adapter":                                       # SHARED base module (intact) + its own low-rank adapter
            p, a = self.modules[prog[1]], self.modules[prog[2]]
            return expert_forward({"w1": p["w1"] + a["u"] @ a["v"], "b1": p["b1"] + a["db1"], "w2": p["w2"] + a["dw2"], "b2": p["b2"] + a["db2"]}, X)
        if k == "compose":
            a = self.skill_logits(prog[2], X).argmax(1); b = self.skill_logits(prog[3], X).argmax(1)
            y = combine(prog[1], a, b); return torch.stack([(1 - y).float(), y.float()], 1) * 10.0
        raise KeyError(k)

    def skill_predict(self, skill, X):
        with torch.no_grad(): return self.skill_logits(skill, X).argmax(1)

    # ---- router (context discovery) -------------------------------------------------------
    def route(self, C, novelty=True):
        """Nearest context (L1 distance to the prototype); -1 if new (only if novelty=True:
        novelty is decided on a BATCH at learning time; at inference, nearest prototype)."""
        if not self.contexts: return torch.full((len(C),), -1, dtype=torch.long)
        P = torch.stack([c.proto for c in self.contexts]); d = torch.cdist(C, P, p=1)
        best = d.argmin(1)
        return torch.where(d.min(1).values <= self.novelty, best, torch.full_like(best, -1)) if novelty else best

    def discover(self, C):
        """Context of the batch (majority of the routing); creates a new context if the batch is new."""
        r = self.route(C); vals, counts = r.unique(return_counts=True); top = int(vals[counts.argmax()])
        if top >= 0:
            ctx = self.contexts[top]; ctx.proto = (ctx.proto * ctx.count + C.sum(0)) / (ctx.count + len(C)); ctx.count += len(C)
            return ctx, False
        ctx = Context(f"ctx{len(self.contexts)}", C.mean(0), len(C)); self.contexts.append(ctx); return ctx, True

    def predict(self, X, C):
        r = self.route(C, novelty=False); out = torch.zeros(len(X), dtype=torch.long)
        for ci in r.unique().tolist():
            mask = r == ci
            if ci < 0 or self.contexts[ci].skill is None: continue
            out[mask] = self.skill_predict(self.contexts[ci].skill, X[mask])
        return out

    def accuracy(self, X, C, y): return float((self.predict(X, C) == y).float().mean())

    # ---- governed application ------------------------------------------------------------------
    def apply(self, spec: "AdaptationTransformation"):
        effects = spec.execute(self) or {}
        self.counters["transformations"] += 1
        node = self.graph.current.add_child(spec, {"skills": sorted(self.skills), "modules": sorted(self.modules)},
                                            name=f"S{len(self.graph.nodes)}")
        self.graph.nodes[node.name] = node; self.graph.current = node
        return effects

    def history(self):
        return list(self.graph.current.get_path_names())


def skill_modules(net, skill, seen=None):
    seen = seen or set()
    if skill in seen or skill not in net.skills: return set()
    seen.add(skill); prog = net.skills[skill]
    if prog[0] == "expert": return {prog[1]}
    if prog[0] == "not": return skill_modules(net, prog[1], seen)
    if prog[0] == "adapter": return {prog[1], prog[2]}
    return skill_modules(net, prog[2], seen) | skill_modules(net, prog[3], seen)


# ================================================================ LOCAL training (backprop allowed by the planner)

def train_params(p, X, y, steps=200, lr=0.03, seed=0, project=None, anchors=None, lam=1.0, ewc=None):
    """LOCAL gradient descent on a module (dict of tensors). project(grads) -> projected grads (allowed
    subspace); anchors=(X_probe, reference outputs): output-equivalence constraint; ewc=(F, θ*, λ)."""
    torch.manual_seed(seed)
    q = {k: v.clone().requires_grad_(True) for k, v in p.items()}
    # NB: Adam normalises each coordinate and LEAVES the projected subspace; with projection: momentum SGD (stays in the subspace)
    opt = torch.optim.Adam(q.values(), lr=lr) if project is None else torch.optim.SGD(q.values(), lr=0.1, momentum=0.9)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(expert_forward(q, X), y)
        if anchors is not None: loss = loss + lam * ((expert_forward(q, anchors[0]) - anchors[1]) ** 2).mean()
        if ewc is not None:
            F, th, lam_e = ewc; loss = loss + lam_e * sum((F[k] * (q[k] - th[k]) ** 2).sum() for k in q)
        loss.backward()
        if project is not None:
            g = project({k: q[k].grad for k in q})
            for k in q: q[k].grad = g[k]
        opt.step()
    return {k: v.detach() for k, v in q.items()}


def flat(p): return torch.cat([p[k].flatten() for k in PARAMS])


def unflat(v, like):
    out, i = {}, 0
    for k in PARAMS: n = like[k].numel(); out[k] = v[i:i + n].reshape(like[k].shape); i += n
    return out


def output_jacobian(p, X):
    """J [N, P]: ∂(logit1 − logit0)/∂θ on the probes X (output invariant to preserve)."""
    v = flat(p).clone().requires_grad_(True)
    def f(vec):
        q = unflat(vec, p); o = expert_forward(q, X); return o[:, 1] - o[:, 0]
    return torch.autograd.functional.jacobian(f, v)


def null_space_projector(p, X, energy=0.99, max_rank=None):
    """Projector onto the subspace (approx.) orthogonal to the sensitive directions: J_old Δθ ≈ 0 (SVD of J)."""
    J = output_jacobian(p, X); U, S, Vh = torch.linalg.svd(J, full_matrices=False)
    cum = torch.cumsum(S ** 2, 0) / (S ** 2).sum().clamp_min(1e-12)
    r = int((cum < energy).sum()) + 1
    if max_rank: r = min(r, max_rank)
    V = Vh[:r].T                                                             # [P, r] sensitive directions
    def project(grads):
        g = torch.cat([grads[k].flatten() for k in PARAMS]); g = g - V @ (V.T @ g); return unflat(g, p)
    return project, V, r


# ================================================================ canonical probes, certificates, functional memory

def synthesize_probes(net_or_fn, skill, k=16, seed=0):
    """SYNTHESISED probes (the input space is enumerable; no old data is stored):
    low-margin points (decision boundary) + distant points (coverage), k in total."""
    logits_fn = (lambda X: net_or_fn.skill_logits(skill, X)) if isinstance(net_or_fn, SkillNet) else net_or_fn
    with torch.no_grad(): o = logits_fn(ALL_DATA); margin = (o[:, 1] - o[:, 0]).abs()
    chosen = list(margin.argsort()[: k // 2].tolist())
    while len(chosen) < k:                                                   # point farthest (Hamming) from the chosen ones
        d = torch.cdist(ALL_DATA, ALL_DATA[chosen], p=1).min(1).values; d[chosen] = -1
        chosen.append(int(d.argmax()))
    X = ALL_DATA[chosen]
    with torch.no_grad(): y = logits_fn(X).argmax(1)
    return X, y


@dataclass
class InvariantCertificate:
    """What a consolidated skill must preserve (compact FUNCTIONAL memory)."""
    skill: str
    probes: torch.Tensor                     # synthesised canonical probes
    labels: torch.Tensor                     # expected outputs (behavioural signature)
    tolerance: float                         # accepted degradation (fraction of probes)
    sensitive_modules: Tuple[str, ...]
    stability_radius: float
    admissible_families: Tuple[str, ...]
    forbidden_interactions: Tuple[str, ...] = ()
    budget: float = 0.0                      # remaining stability budget (degradation still tolerated)
    macro: Optional[tuple] = None            # program of the skill (for regeneration)

    def check(self, net: SkillNet):
        if self.skill not in net.skills: return False, 1.0
        try: err = float((net.skill_predict(self.skill, self.probes) != self.labels).float().mean())
        except (KeyError, RuntimeError): return False, 1.0                  # broken program (missing module, shapes)
        return err <= self.tolerance, err

    def memory_bytes(self): return self.probes.numel() // 8 + self.labels.numel() // 8 + 64       # bits + metadata


def stability_radius_of(net, skill, probes, labels, trials=8, grid=(0.05, 0.1, 0.2, 0.4, 0.8)):
    """Empirical radius (KAM-inspired): max relative amplitude where ≥ 90% of the perturbations keep the probes."""
    mods = sorted(skill_modules(net, skill)); best = 0.0
    for mag in grid:
        ok = 0
        for t in range(trials):
            n2 = copy.copy(net); n2.modules = dict(net.modules)
            for j, m in enumerate(mods):
                g = torch.Generator().manual_seed(1000 * t + j)
                n2.modules[m] = {k: v + mag * float(v.norm().clamp_min(1e-6)) * (lambda d: d / d.norm().clamp_min(1e-12))(torch.randn(v.shape, generator=g)) for k, v in net.modules[m].items()}
            ok += float((n2.skill_predict(skill, probes) != labels).float().mean()) == 0.0
        if ok / trials >= 0.9: best = mag
        else: break
    return best


def certify(net, skill, k=16, tolerance=0.0, seed=0, macro=None):
    X, y = synthesize_probes(net, skill, k, seed)
    r = stability_radius_of(net, skill, X, y)
    fams = ("AddModule", "CloneModule", "FastWeightUpdate", "ReuseSkill", "ComposeSkills", "CreateMemory", "FreezeSubspace",
            "AllocateSubspace") + (("ParameterUpdate",) if r >= 0.2 else ())               # elastic: robust skill
    budget = max(0.0, min(0.125, r / 4))                     # ELASTIC consolidation: robust skill => larger budget
    return InvariantCertificate(skill, X, y, max(tolerance, budget), tuple(sorted(skill_modules(net, skill))), r, fams,
                                budget=budget, macro=macro or net.skills.get(skill))


# ================================================================ adaptation transformations (part A)

FAMILIES = ("ParameterUpdate", "FastWeightUpdate", "RouteChange", "AddConnection", "RemoveConnection", "AddModule",
            "CloneModule", "SplitModule", "MergeModule", "FreezeSubspace", "AllocateSubspace", "CreateMemory",
            "DistillSkill", "ReuseSkill", "ComposeSkills")


class AdaptationTransformation(TransformationSpec):
    """Traceable adaptation transformation; `family` ∈ FAMILIES; `execute(net)` acts on a SkillNet."""
    family = "?"
    reversible = False

    def __init__(self, **kw):
        object.__setattr__(self, "kw", kw); object.__setattr__(self, "name", f"{self.family}({', '.join(f'{k}={_short(v)}' for k, v in kw.items() if k not in _HIDDEN)})")
        object.__setattr__(self, "domain", TransformationDomain.MODEL); object.__setattr__(self, "parameters", {k: _short(v) for k, v in kw.items() if k not in _HIDDEN})
        object.__setattr__(self, "invertible", self.reversible); object.__setattr__(self, "cost", 1.0)
        from core.invariants import InvariantSet as LabelSet
        object.__setattr__(self, "preserves", LabelSet(set())); object.__setattr__(self, "changes", {self.family})
        object.__setattr__(self, "preconditions", ()); object.__setattr__(self, "postconditions", ())

    def __getattr__(self, k):
        kw = object.__getattribute__(self, "kw")
        if k in kw: return kw[k]
        raise AttributeError(k)

    def apply(self, net): return net.apply(self)
    def precondition(self, net) -> bool: return True
    def execute(self, net): raise NotImplementedError


_HIDDEN = {"delta", "X", "y", "probes", "labels", "basis", "params", "project", "anchors", "certificate"}


def _short(v):
    if isinstance(v, torch.Tensor): return f"tensor{tuple(v.shape)}"
    if isinstance(v, dict): return "{…}"
    return v


class ParameterUpdate(AdaptationTransformation):
    family = "ParameterUpdate"
    def precondition(self, net): return self.module in net.modules
    def execute(self, net):
        new = dict(net.modules[self.module])                    # copy: the snapshots (shallow copies) remain valid
        for k, d in self.delta.items(): new[k] = new[k] + d
        net.modules[self.module] = new
        return {"changed": sum(int((d != 0).sum()) for d in self.delta.values())}


class FastWeightUpdate(AdaptationTransformation):
    family = "FastWeightUpdate"; reversible = True
    def execute(self, net):
        if self.delta is None: net.fast.pop(self.module, None); return {"cleared": self.module}
        cur = net.fast.setdefault(self.module, {k: torch.zeros_like(v) for k, v in net.modules[self.module].items()})
        for k, d in self.delta.items(): cur[k] = cur[k] + d
        return {}


class RouteChange(AdaptationTransformation):
    family = "RouteChange"; reversible = True
    def execute(self, net):
        ctx = next(c for c in net.contexts if c.name == self.context); old = ctx.skill; ctx.skill = self.skill; return {"previous": old}


class AddConnection(AdaptationTransformation):
    """Plugs a skill in as an INPUT of a composed skill (edge of the program)."""
    family = "AddConnection"
    def execute(self, net):
        prog = list(net.skills[self.skill]); prog[self.slot] = self.source; net.skills[self.skill] = tuple(prog); return {}


class RemoveConnection(AdaptationTransformation):
    family = "RemoveConnection"
    def execute(self, net):
        net.skills[self.skill] = ("expert", self.fallback_module); return {}


class AddModule(AdaptationTransformation):
    family = "AddModule"; reversible = True
    def execute(self, net):
        net.modules[self.module] = new_expert(self.hidden, self.seed); net.counters["grown_modules"] += 1; return {"params": n_params(net.modules[self.module])}


class CloneModule(AdaptationTransformation):
    family = "CloneModule"; reversible = True
    def execute(self, net): net.modules[self.new] = {k: v.clone() for k, v in net.modules[self.source].items()}; return {}


class SplitModule(AdaptationTransformation):
    """Splits the hidden units of a module into two modules (given partition)."""
    family = "SplitModule"
    def execute(self, net):
        p = net.modules[self.module]; idx = torch.tensor(self.units)
        rest = torch.tensor([i for i in range(p["w1"].shape[0]) if i not in set(self.units)])
        for name, ids in ((self.new_a, idx), (self.new_b, rest)):
            net.modules[name] = {"w1": p["w1"][ids].clone(), "b1": p["b1"][ids].clone(), "w2": p["w2"][:, ids].clone(), "b2": p["b2"].clone() / 2}
        return {}


class MergeModule(AdaptationTransformation):
    """Merges b into a: all the skills using b use a; b is removed."""
    family = "MergeModule"
    def execute(self, net):
        for s, prog in list(net.skills.items()):
            if prog == ("expert", self.b): net.skills[s] = ("expert", self.a)
        net.modules.pop(self.b, None); net.fast.pop(self.b, None); return {"removed": self.b}


class FreezeSubspace(AdaptationTransformation):
    family = "FreezeSubspace"
    def execute(self, net): net.frozen[self.module] = self.basis; return {"rank": int(self.basis.shape[1])}


class AllocateSubspace(AdaptationTransformation):
    family = "AllocateSubspace"
    def execute(self, net): net.allocated[self.module] = self.basis; return {}


class CreateMemory(AdaptationTransformation):
    family = "CreateMemory"
    def execute(self, net): net.memories[self.skill] = self.certificate; return {"bytes": self.certificate.memory_bytes()}


class DistillSkill(AdaptationTransformation):
    """Distils a skill (composed or large) into a compact module, trained on synthesised probes."""
    family = "DistillSkill"
    def execute(self, net):
        X = ALL_DATA
        with torch.no_grad(): y = net.skill_predict(self.skill, X)
        net.modules[self.module] = train_params(new_expert(self.hidden, self.seed), X, y, self.steps, seed=self.seed)
        net.counters["train_steps"] += self.steps
        net.skills[self.target] = ("expert", self.module); return {}


class ReuseSkill(AdaptationTransformation):
    family = "ReuseSkill"; reversible = True
    def execute(self, net):
        ctx = next(c for c in net.contexts if c.name == self.context); ctx.skill = self.skill; net.counters["reused"] += 1; return {}


class ComposeSkills(AdaptationTransformation):
    family = "ComposeSkills"
    def execute(self, net):
        if self.op == "not": net.skills[self.name_] = ("not", self.a)
        else: net.skills[self.name_] = ("compose", self.op, self.a, self.b)
        net.counters["composed"] += 1; return {}


class DefineSkill(AdaptationTransformation):
    """Declares an atomic skill served by a module (support for the families above)."""
    family = "AddConnection"
    def execute(self, net): net.skills[self.skill] = ("expert", self.module); return {}


FAMILY_CLASSES = {c.family: c for c in (ParameterUpdate, FastWeightUpdate, RouteChange, AddConnection, RemoveConnection, AddModule,
                                        CloneModule, SplitModule, MergeModule, FreezeSubspace, AllocateSubspace, CreateMemory,
                                        DistillSkill, ReuseSkill, ComposeSkills)}

"""COMPOSITIONAL PLASTICITY: goal + model + invariants to preserve → space of admissible transformations →
composition (AdaptationProgram) → growth if necessary → verification → consolidation.

Order of preference (part Q): reuse > composition > small local adaptation (low-rank adapter) >
new module; never a dense adaptation of a shared module. Backpropagation is allowed only on the
parameters chosen by the planner (adapter or new module). The protection of the old skills
relies on CERTIFICATES (synthesised probes + signatures), without any old raw data.
"""
from __future__ import annotations

import copy
import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from core.skillnet import (
    ALL_DATA, OPS, PARAMS, AddModule, AdaptationTransformation, ComposeSkills, CreateMemory, DefineSkill, DistillSkill,
    FastWeightUpdate, InvariantCertificate, MergeModule, ReuseSkill, SkillNet, certify, combine, expert_forward, n_params,
    new_expert, skill_modules, train_params,
)


# ================================================================ low-rank adapter (new bank of fast weights)

class AddAdapter(AdaptationTransformation):
    """Low-rank adapter specific to a new skill: w1' = w1 + u vᵀ, w2' = w2 + du (new parameters;
    the base module, shared, remains INTACT)."""
    family = "AllocateSubspace"
    def execute(self, net):
        net.modules[self.adapter] = self.params; net.skills[self.skill] = ("adapter", self.base, self.adapter)
        net.counters["grown_modules"] += 0; return {"params": n_params(self.params)}


def train_adapter(net, base, X, y, rank=1, steps=200, lr=0.05, seed=0):
    g = torch.Generator().manual_seed(seed); p = net.modules[base]; h = p["w1"].shape[0]
    a = {"u": (torch.randn(h, rank, generator=g) * 0.1), "v": torch.randn(rank, X.shape[1], generator=g) * 0.1,
         "db1": torch.zeros(h), "dw2": torch.zeros_like(p["w2"]), "db2": torch.zeros(2)}
    q = {k: v.clone().requires_grad_(True) for k, v in a.items()}; opt = torch.optim.Adam(q.values(), lr=lr)
    for _ in range(steps):                                     # LOCAL backprop: only the parameters of the adapter
        opt.zero_grad()
        w = {"w1": p["w1"] + q["u"] @ q["v"], "b1": p["b1"] + q["db1"], "w2": p["w2"] + q["dw2"], "b2": p["b2"] + q["db2"]}
        torch.nn.functional.cross_entropy(expert_forward(w, X), y).backward(); opt.step()
    net.counters["train_steps"] += steps
    return {k: v.detach() for k, v in q.items()}


# ================================================================ adaptation programs

@dataclass
class ProgramStep:
    family: str
    args: Dict[str, Any]
    expected_gain: float = 0.0
    expected_survival: float = 1.0
    cost: float = 0.0
    reversible: bool = True
    precondition: str = ""


@dataclass
class AdaptationProgram:
    """T_n ∘ … ∘ T_1: steps with preconditions, expected gain, expected survival of the invariants, cost, reversibility."""
    kind: str
    steps: List[ProgramStep]
    build: Callable[[SkillNet], None] = field(repr=False, default=None)
    expected_acc: float = 0.0
    params_added: int = 0
    compute: int = 0

    def signature(self): return tuple(s.family for s in self.steps)

    def total_cost(self, w_err=10.0, w_param=0.002, w_compute=0.0005, w_violation=50.0):
        surv = 1.0
        for s in self.steps: surv *= s.expected_survival
        return w_err * (1 - self.expected_acc) + w_violation * (1 - surv) + w_param * self.params_added + w_compute * self.compute + sum(s.cost for s in self.steps)


# ================================================================ plasticity planner

@dataclass
class PlannerConfig:
    target: float = 0.95
    expert_hidden: int = 16
    train_steps: int = 250
    adapter_steps: int = 200
    allow_reuse: bool = True
    allow_compose: bool = True
    allow_adapter: bool = True
    probe_k: int = 16
    capacity: int = 3000
    use_library: bool = True


class PlasticityPlanner:
    """BOUNDED best-first search over candidate programs (cost = new-task error + invariant
    violation + structural cost + compute). Reuse and composition are evaluated exactly (without
    training); adapters and growth are trained locally then VERIFIED against all the certificates."""

    def __init__(self, config: PlannerConfig = None, stabilizer=None, library=None, seed=0):
        self.cfg = config or PlannerConfig(); self.stabilizer = stabilizer; self.library = library; self.seed = seed
        self.log: List[dict] = []; self.programs: List[AdaptationProgram] = []
        self.evaluations = 0

    # ---- candidates ----------------------------------------------------------------------------
    def _skill_preds(self, net, X):
        return {s: net.skill_predict(s, X) for s in net.skills}

    def reuse_candidates(self, net, ctx, X, y, preds):
        out = []
        for s, p in preds.items():
            acc = float((p == y).float().mean()); self.evaluations += 1
            out.append(AdaptationProgram("reuse", [ProgramStep("ReuseSkill", {"skill": s}, acc, 1.0, 0.0)],
                                         lambda n, s=s: n.apply(ReuseSkill(context=ctx.name, skill=s)), acc))
            nacc = 1 - acc
            if nacc > acc:
                name = f"not_{s}"
                out.append(AdaptationProgram("reuse", [ProgramStep("ComposeSkills", {"op": "not", "a": s}, nacc, 1.0, 0.1),
                                                       ProgramStep("ReuseSkill", {"skill": name}, nacc, 1.0, 0.0)],
                                             lambda n, s=s, name=name: (n.apply(ComposeSkills(name_=name, op="not", a=s, b=None)),
                                                                        n.apply(ReuseSkill(context=ctx.name, skill=name))), nacc))
        return out

    def compose_candidates(self, net, ctx, X, y, preds, max_skills=24):
        out = []; names = list(preds)[:max_skills]
        for a, b in itertools.combinations(names, 2):
            for op in OPS:
                acc = float((combine(op, preds[a], preds[b]) == y).float().mean()); self.evaluations += 1
                if acc < 0.8: continue
                name = f"{op}({a},{b})"
                out.append(AdaptationProgram("compose", [ProgramStep("ComposeSkills", {"op": op, "a": a, "b": b}, acc, 1.0, 0.2),
                                                         ProgramStep("ReuseSkill", {"skill": name}, acc, 1.0, 0.0)],
                                             lambda n, a=a, b=b, op=op, name=name: (n.apply(ComposeSkills(name_=name, op=op, a=a, b=b)),
                                                                                    n.apply(ReuseSkill(context=ctx.name, skill=name))), acc))
        return out

    def adapter_candidate(self, net, ctx, X, y, preds, base_skill):
        base = net.skills[base_skill][1]; name = f"skill_{ctx.name}"; aname = f"adapter_{ctx.name}"
        (Xt, yt), (Xv, yv) = _split(X, y)
        params = train_adapter(net, base, Xt, yt, steps=self.cfg.adapter_steps, seed=self.seed)
        n2 = copy.copy(net); n2.modules = dict(net.modules); n2.modules[aname] = params; n2.skills = dict(net.skills)
        n2.skills[name] = ("adapter", base, aname); acc = float((n2.skill_predict(name, Xv) == yv).float().mean())   # VALIDATION accuracy
        surv = self._predicted_survival("AllocateSubspace", net, base)
        return AdaptationProgram("adapter", [ProgramStep("AllocateSubspace", {"base": base}, acc, surv, 0.3),
                                             ProgramStep("FastWeightUpdate", {"module": aname, "rank": 1}, acc, 1.0, 0.0),
                                             ProgramStep("ReuseSkill", {"skill": name}, acc, 1.0, 0.0)],
                                 lambda n: (n.apply(AddAdapter(skill=name, base=base, adapter=aname, params=params)),
                                            n.apply(ReuseSkill(context=ctx.name, skill=name))),
                                 acc, n_params(params), self.cfg.adapter_steps)

    def grow_candidate(self, net, ctx, X, y):
        m = f"expert_{ctx.name}"; name = f"skill_{ctx.name}"
        (Xt, yt), (Xv, yv) = _split(X, y)
        p = train_params(new_expert(self.cfg.expert_hidden, self.seed + len(net.modules)), Xt, yt, self.cfg.train_steps, seed=self.seed)
        acc = float((expert_forward(p, Xv).argmax(1) == yv).float().mean())                                   # validation
        p = train_params(p, X, y, self.cfg.train_steps // 5, seed=self.seed)                                  # then the whole support
        net.counters["train_steps"] += self.cfg.train_steps + self.cfg.train_steps // 5
        def build(n):
            n.apply(AddModule(module=m, hidden=self.cfg.expert_hidden, seed=0)); n.modules[m] = p
            n.apply(DefineSkill(skill=name, module=m)); n.apply(ReuseSkill(context=ctx.name, skill=name))
        return AdaptationProgram("grow", [ProgramStep("AddModule", {"hidden": self.cfg.expert_hidden}, acc, 1.0, 0.5),
                                          ProgramStep("ParameterUpdate", {"module": m, "local": True}, acc, 1.0, 0.0),
                                          ProgramStep("ReuseSkill", {"skill": name}, acc, 1.0, 0.0)], build, acc, n_params(p), self.cfg.train_steps)

    def refine_inplace(self, net, ctx, X, y, certificates, coupling_max=0.5):
        """Refinement of an existing skill in ITS OWN module, by gradient projected into the null space of the
        probes of the OTHER skills that share the module (J_old Δθ ≈ 0); avoided if the module is strongly coupled
        (interaction graph) to fragile skills. Stability budget allotted per certificate."""
        from core.skillnet import null_space_projector, ParameterUpdate
        m = net.skills[ctx.skill][1]
        others = [s for s in net.module_users(m) if s != ctx.skill and s in certificates]
        if others and coupling(net, certificates, m, others) > coupling_max: return None
        project = None
        if others:
            Xp = torch.cat([certificates[s].probes for s in others]); project, V, r = null_space_projector(net.modules[m], Xp)
        new = train_params(net.modules[m], X, y, self.cfg.train_steps, seed=self.seed, project=project)
        net.counters["train_steps"] += self.cfg.train_steps
        snap = _snapshot(net)
        net.apply(ParameterUpdate(module=m, delta={k: new[k] - net.modules[m][k] for k in new}))
        acc = float((net.skill_predict(ctx.skill, X) == y).float().mean())
        ok = acc >= self.cfg.target and all(certificates[s].check(net)[0] for s in certificates if s != ctx.skill)
        if not ok: _restore(net, snap); net.counters["rollbacks"] += 1; return None
        for s in others: certificates[s].budget -= certificates[s].check(net)[1]          # consumption of the budget
        certificates[ctx.skill] = certify(net, ctx.skill, self.cfg.probe_k)
        return {"kind": "refine_inplace", "acc": acc, "program": ("AllocateSubspace", "ParameterUpdate"), "params_added": 0}

    def _predicted_survival(self, family, net, module):
        if self.stabilizer is None: return 1.0
        return self.stabilizer.predict_family(family, net, module)

    # ---- learning one task -------------------------------------------------------------------------
    def learn(self, net: SkillNet, X, C, y, certificates: Dict[str, InvariantCertificate]):
        ctx, is_new = net.discover(C)
        rec = {"context": ctx.name, "new_context": is_new, "kind": None, "steps": 0, "rollback": 0}
        if not is_new and ctx.skill is not None:
            acc = float((net.skill_predict(ctx.skill, X) == y).float().mean())
            if acc >= self.cfg.target - 0.05:                             # recurrence: immediate reactivation (sample tolerance)
                rec.update(kind="reactivate", acc=acc); self.log.append(rec); return rec
            if net.skills[ctx.skill][0] == "expert":                       # known context but drifted task: IN-PLACE refinement
                r = self.refine_inplace(net, ctx, X, y, certificates)
                if r is not None: rec.update(r); self.log.append(rec); return rec
        preds = self._skill_preds(net, X) if net.skills else {}
        order = ["reuse", "compose", "adapter", "grow"]
        if self.library is not None and self.cfg.use_library: order = self.library.order(_context_signature(preds, y), order)
        cands: List[AdaptationProgram] = []
        for kind in order:
            before = net.counters["train_steps"]
            if kind == "reuse" and self.cfg.allow_reuse and preds: cands += self.reuse_candidates(net, ctx, X, y, preds)
            elif kind == "compose" and self.cfg.allow_compose and len(preds) >= 2: cands += self.compose_candidates(net, ctx, X, y, preds)
            elif kind == "adapter" and self.cfg.allow_adapter and preds:
                atomic = [s for s in preds if net.skills[s][0] == "expert"]
                if atomic:
                    best = max(atomic, key=lambda s: max(float((preds[s] == y).float().mean()), 0.0))
                    cands.append(self.adapter_candidate(net, ctx, X, y, preds, best))
            elif kind == "grow": cands.append(self.grow_candidate(net, ctx, X, y))
            rec["steps"] += net.counters["train_steps"] - before
            good = [c for c in cands if c.expected_acc >= self.cfg.target]
            if good: break                                                # bounded search: we stop at the first sufficient level
        cands.sort(key=lambda c: c.total_cost())
        for prog in cands:                                                # verify; rollback if a certificate breaks
            snapshot = _snapshot(net)
            prog.build(net)
            ok = all(cert.check(net)[0] for cert in certificates.values())
            if ok:
                name = ctx.skill; certificates[name] = certify(net, name, self.cfg.probe_k, macro=net.skills[name])
                net.apply(CreateMemory(skill=name, certificate=certificates[name]))
                rec.update(kind=prog.kind, acc=prog.expected_acc, program=prog.signature(), params_added=prog.params_added)
                self.programs.append(prog)
                if self.library is not None: self.library.record(_context_signature(preds, y), prog)
                break
            _restore(net, snapshot); net.counters["rollbacks"] += 1; rec["rollback"] += 1
        manage_capacity(net, certificates, self.cfg.capacity)
        self.log.append(rec); return rec


def _split(X, y, frac=0.75):
    n = int(frac * len(X)); return (X[:n], y[:n]), (X[n:], y[n:])


def coupling(net, certificates, module, skills, mag=0.1, trials=4):
    """Module ↔ skills coupling (reduced PerturbationInteractionGraph): fraction of probes changed under a
    small relative perturbation of the module."""
    out = 0.0
    for t in range(trials):
        g = torch.Generator().manual_seed(t); n2 = copy.copy(net); n2.modules = dict(net.modules)
        n2.modules[module] = {k: v + mag * float(v.norm()) * (lambda d: d / d.norm().clamp_min(1e-12))(torch.randn(v.shape, generator=g)) for k, v in net.modules[module].items()}
        out += sum(certificates[s].check(n2)[1] for s in skills) / len(skills)
    return out / trials


def _snapshot(net):
    return (dict(net.modules), dict(net.skills), [(c.name, c.skill) for c in net.contexts], dict(net.memories))


def _restore(net, snap):
    net.modules, net.skills = dict(snap[0]), dict(snap[1]); m = dict(snap[2])
    for c in net.contexts: c.skill = m.get(c.name)
    net.memories = dict(snap[3])


def _context_signature(preds, y):
    """Signature of the learning context (without task identifier): best reuse / negation."""
    if not preds: return ("empty",)
    best = max(max(float((p == y).float().mean()), 1 - float((p == y).float().mean())) for p in preds.values())
    return ("reuse_ok",) if best >= 0.95 else ("near",) if best >= 0.8 else ("far",)


# ================================================================ capacity management: merging, distillation, pruning

def equivalent_modules(net):
    """Equivalence classes of atomic skills by output signature over the WHOLE input space
    (Invariant Algebra: same invariant signature ⇒ merging admissible)."""
    from core.invariant_algebra import CheckedInvariant, EquivalenceClass, InvariantSet
    atomic = {s: p[1] for s, p in net.skills.items() if p[0] == "expert"}
    inv = InvariantSet([CheckedInvariant("output", "skill", lambda s: tuple(net.skill_predict(s, ALL_DATA).tolist()))])
    ec = EquivalenceClass(inv)
    for s in atomic: ec.add(s)
    return [[atomic[s] for s in members] for members in ec.classes.values() if len(members) > 1]


def manage_capacity(net, certificates, capacity):
    """When the budget is exceeded: merge (invariant equivalence), distil the deep compositions,
    prune the unreferenced modules. Every operation is verified by the certificates (rollback otherwise)."""
    report = Counter()
    used = set().union(*[skill_modules(net, s) for s in net.skills]) if net.skills else set()
    for m in [m for m in net.modules if m not in used and not m.startswith("adapter_")]:          # pruning
        net.modules.pop(m); report["pruned"] += 1
    if net.parameter_count() <= capacity: return report
    for group in equivalent_modules(net):                                                      # merging
        keep = group[0]
        for b in group[1:]:
            snap = _snapshot(net); net.apply(MergeModule(a=keep, b=b))
            if all(c.check(net)[0] for c in certificates.values()): report["merged"] += 1
            else: _restore(net, snap)
    if net.parameter_count() > capacity:                                                       # compact distillation
        for s, prog in list(net.skills.items()):
            if prog[0] != "expert" or net.modules[prog[1]]["w1"].shape[0] <= 8: continue
            snap = _snapshot(net); m = prog[1] + "_d"
            net.apply(DistillSkill(skill=s, target=s, module=m, hidden=8, steps=300, seed=0))
            if all(c.check(net)[0] for c in certificates.values()):
                net.modules.pop(prog[1], None); report["distilled"] += 1
            else: _restore(net, snap)
            if net.parameter_count() <= capacity: break
    return report


# ================================================================ anti-unification and macros

VAR = "?"


def anti_unify_args(a, b, env):
    """Least common generalisation of two values: identical → value; otherwise variable (consistent via env)."""
    if a == b: return a
    key = (repr(a), repr(b))
    if key not in env: env[key] = f"{VAR}v{len(env)}"
    return env[key]


def anti_unify(p1: List[ProgramStep], p2: List[ProgramStep]):
    """STRUCTURAL anti-unification of two programs: alignment of the families by longest common
    subsequence; differing arguments → shared variables. Returns (template, variables) or None."""
    f1, f2 = [s.family for s in p1], [s.family for s in p2]
    L = [[0] * (len(f2) + 1) for _ in range(len(f1) + 1)]
    for i in range(len(f1) - 1, -1, -1):
        for j in range(len(f2) - 1, -1, -1):
            L[i][j] = L[i + 1][j + 1] + 1 if f1[i] == f2[j] else max(L[i + 1][j], L[i][j + 1])
    i = j = 0; env = {}; tpl = []
    while i < len(f1) and j < len(f2):
        if f1[i] == f2[j]:
            keys = set(p1[i].args) | set(p2[j].args)
            tpl.append((f1[i], {k: anti_unify_args(p1[i].args.get(k), p2[j].args.get(k), env) for k in sorted(keys)})); i += 1; j += 1
        elif L[i + 1][j] >= L[i][j + 1]: i += 1
        else: j += 1
    if not tpl: return None
    return tpl, sorted(set(env.values()))


@dataclass
class ParameterizedAdaptationMacro:
    name: str
    template: List[Tuple[str, Dict[str, Any]]]
    variables: List[str]
    support: int = 1

    def instantiate(self, bindings: Dict[str, Any]):
        return [(f, {k: bindings.get(v, v) if isinstance(v, str) and v.startswith(VAR) else v for k, v in args.items()})
                for f, args in self.template]

    def size(self): return len(self.template)


def induce_macros(programs: List[AdaptationProgram]):
    """Groups the successful programs by family signature and anti-unifies each group into a macro."""
    groups = defaultdict(list)
    for p in programs: groups[p.signature()].append(p)
    macros = []
    for sig, ps in groups.items():
        tpl = [(s.family, dict(s.args)) for s in ps[0].steps]; variables = []
        for p in ps[1:]:
            au = anti_unify([ProgramStep(f, a) for f, a in tpl], p.steps)
            if au: tpl, variables = au
        macros.append(ParameterizedAdaptationMacro("+".join(sig), tpl, variables, len(ps)))
    return macros


class StrategyLibrary:
    """Meta-learning without a large network: context signature → statistics of the successful macros → order
    in which the levels are tried (reuse / compose / adapter / grow)."""

    def __init__(self): self.stats = defaultdict(Counter)

    def record(self, sig, prog): self.stats[sig][prog.kind] += 1

    def order(self, sig, default):
        c = self.stats.get(sig)
        if not c: return default
        best = c.most_common(1)[0][0]
        return [best] + [k for k in default if k != best]

    def size(self): return sum(len(c) for c in self.stats.values())


# ================================================================ regeneration without a full snapshot

def regenerate(net, certificates, module, seed=0, steps=600, lr=0.02):
    """Corrupted module → failing certificates → reconstruction from the FUNCTIONAL sources (without snapshot):
    (1) composed skills: program (macro) restored; (2) atomic module: refitted JOINTLY on the
    canonical probes of ALL the skills that use it (direct or via adapter, adapters fixed),
    starting from the corrupted weights; final verification by the certificates."""
    failing = [s for s, c in certificates.items() if not c.check(net)[0]]
    for s in failing:
        cert = certificates[s]
        if cert.macro is not None and cert.macro[0] in ("compose", "not") and net.skills.get(s) != cert.macro: net.skills[s] = cert.macro
    users = [s for s in certificates if module in skill_modules(net, s) and net.skills[s][0] in ("expert", "adapter")]
    if users and "w1" in net.modules.get(module, {}):
        q = {k: v.clone().requires_grad_(True) for k, v in net.modules[module].items()}; opt = torch.optim.Adam(q.values(), lr=lr)
        torch.manual_seed(seed)
        for _ in range(steps):
            opt.zero_grad(); loss = 0.0
            for s in users:
                cert = certificates[s]; prog = net.skills[s]
                if prog[0] == "expert": w = q
                else:
                    a = net.modules[prog[2]]; w = {"w1": q["w1"] + a["u"] @ a["v"], "b1": q["b1"] + a["db1"], "w2": q["w2"] + a["dw2"], "b2": q["b2"] + a["db2"]}
                loss = loss + torch.nn.functional.cross_entropy(expert_forward(w, cert.probes), cert.labels)
            loss.backward(); opt.step()
        net.modules[module] = {k: v.detach() for k, v in q.items()}
    still = [s for s, c in certificates.items() if not c.check(net)[0]]
    return {"failing": failing, "repaired": [s for s in failing if s not in still], "still_failing": still, "users": users}

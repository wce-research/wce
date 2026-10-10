"""Continual-learning benchmark WITHOUT task identifier (10 to 20 successive tasks): 7 methods.

1 vanilla (monolithic network, input = data + cue)            2 replay (buffer of RAW data)
3 EWC-like (diagonal Fisher)                                   4 projected gradient (null space of the Jacobian on synthesised probes)
5 invariant constraints (anchoring of the outputs on probes) 6 structural growth alone (one expert per context)
7 full compositional plasticity (reuse > composition > adapter > growth, certificates, library)

Metrics: final mean accuracy, backward transfer (BWT), forward transfer (FWT = accuracy at learning time −
fresh network trained alone), forgetting, parameter growth, stored memory (bytes), compute (training steps),
recurrences (reactivation latency, reuse).
"""
from __future__ import annotations

import copy
import random
from collections import Counter

import torch

from core.compositional_plasticity import PlannerConfig, PlasticityPlanner, StrategyLibrary, induce_macros
from core.skillnet import ALL_DATA, D_CUE, D_DATA, SkillNet, synthesize_probes, task_stream

METHODS = ("vanilla", "replay", "ewc", "projected", "invariant_constraints", "structural_growth", "compositional")
H, STEPS, N_SUPPORT = 64, 250, 128


# ================================================================ monolithic network (baselines)

def mono_init(seed):
    g = torch.Generator().manual_seed(seed); d = D_DATA + D_CUE
    return {"w1": torch.randn(H, d, generator=g) * (2.0 / d) ** 0.5, "b1": torch.zeros(H),
            "w2": torch.randn(2, H, generator=g) * (1.0 / H) ** 0.5, "b2": torch.zeros(2)}


def mono_forward(p, Z): return torch.relu(Z @ p["w1"].T + p["b1"]) @ p["w2"].T + p["b2"]


def mono_flat(p): return torch.cat([p[k].flatten() for k in ("w1", "b1", "w2", "b2")])


def mono_unflat(v, like):
    out, i = {}, 0
    for k in ("w1", "b1", "w2", "b2"): n = like[k].numel(); out[k] = v[i:i + n].reshape(like[k].shape); i += n
    return out


def mono_train(p, Z, y, steps=STEPS, lr=0.02, seed=0, replay=None, ewc=None, anchors=None, basis=None, lam=1.0):
    torch.manual_seed(seed); q = {k: v.clone().requires_grad_(True) for k, v in p.items()}
    # Adam would leave the projected subspace (per-coordinate normalisation): SGD with momentum when a basis is imposed
    opt = (torch.optim.Adam(q.values(), lr=lr) if basis is None else torch.optim.SGD(q.values(), lr=0.1, momentum=0.9)); f = torch.nn.functional
    for _ in range(steps):
        opt.zero_grad(); loss = f.cross_entropy(mono_forward(q, Z), y)
        if replay is not None and len(replay[0]): loss = loss + f.cross_entropy(mono_forward(q, replay[0]), replay[1])
        if ewc is not None: loss = loss + sum((F[k] * (q[k] - th[k]) ** 2).sum() for F, th in ewc for k in q) * 50.0
        if anchors is not None and len(anchors[0]): loss = loss + lam * ((mono_forward(q, anchors[0]) - anchors[1]) ** 2).mean()
        loss.backward()
        if basis is not None:                                 # projection: Δθ ⟂ sensitive directions of the old tasks
            g = torch.cat([q[k].grad.flatten() for k in q]); g = g - basis @ (basis.T @ g)
            for k, v in mono_unflat(g, q).items(): q[k].grad = v
        opt.step()
    return {k: v.detach() for k, v in q.items()}


def mono_acc(p, X, C, y):
    with torch.no_grad(): return float((mono_forward(p, torch.cat([X, C], 1)).argmax(1) == y).float().mean())


def probes_for(p, cue_proto, k=16):
    """Probes synthesised for a past task: chosen data (low margin + coverage) × cue prototype
    (canonical task summary) — no old input stored."""
    fn = lambda X: mono_forward(p, torch.cat([X, cue_proto.expand(len(X), D_CUE)], 1))
    X, _ = synthesize_probes(fn, None, k)
    return torch.cat([X, cue_proto.expand(len(X), D_CUE)], 1)


def jac_basis(p, Z, old_basis=None, energy=0.99):
    v = mono_flat(p).clone().requires_grad_(True)
    def f(vec): o = mono_forward(mono_unflat(vec, p), Z); return o[:, 1] - o[:, 0]
    J = torch.autograd.functional.jacobian(f, v)
    if old_basis is not None: J = J - (J @ old_basis) @ old_basis.T
    U, S, Vh = torch.linalg.svd(J, full_matrices=False)
    if float(S.max()) < 1e-8: return old_basis
    cum = torch.cumsum(S ** 2, 0) / (S ** 2).sum(); r = int((cum < energy).sum()) + 1
    B = Vh[:r].T
    return B if old_basis is None else torch.linalg.qr(torch.cat([old_basis, B], 1))[0]


# ================================================================ execution of a method

def run(method, tasks, seed=0, steps=STEPS, n_support=N_SUPPORT, capacity=3000):
    rng = random.Random(seed); T = len(tasks); R = [[0.0] * T for _ in range(T)]
    tests = [t.test_set(seed=100 + i) for i, t in enumerate(tasks)]
    mem_bytes, compute, recur = 0, 0, []
    if method in ("structural_growth", "compositional"):
        net = SkillNet(seed=seed); certs = {}
        cfg = PlannerConfig(train_steps=steps, capacity=capacity)
        if method == "structural_growth": cfg.allow_reuse = cfg.allow_compose = cfg.allow_adapter = False
        planner = PlasticityPlanner(cfg, library=StrategyLibrary() if method == "compositional" else None, seed=seed)
        p0 = None
        for i, t in enumerate(tasks):
            X, C, y = t.sample(n_support, rng); before = net.counters["train_steps"]
            rec = planner.learn(net, X, C, y, certs)
            if t.kind == "recurrence": recur.append({"latency_steps": net.counters["train_steps"] - before, "reused": rec["kind"] in ("reactivate", "reuse", "compose")})
            for j in range(i + 1): R[i][j] = net.accuracy(*tests[j])
            if i == 0: p0 = net.parameter_count()
        params_final, compute = net.parameter_count(), net.counters["train_steps"]
        mem_bytes = sum(c.memory_bytes() for c in certs.values()) + 16 * planner.library.size() if planner.library else sum(c.memory_bytes() for c in certs.values())
        extra = {"kinds": dict(Counter(r["kind"] for r in planner.log)), "rollbacks": sum(r["rollback"] for r in planner.log),
                 "reused_modules": net.counters["reused"], "composed": net.counters["composed"], "grown_modules": net.counters["grown_modules"],
                 "macros": len(induce_macros(planner.programs)), "programs": len(planner.programs), "evaluations": planner.evaluations}
        growth = params_final / max(1, p0)
    else:
        p = mono_init(seed); buf = ([], []); ewc = []; basis = None; anchors = []; cues = []; extra = {}
        n0 = sum(v.numel() for v in p.values())
        for i, t in enumerate(tasks):
            X, C, y = t.sample(n_support, rng); Z = torch.cat([X, C], 1)
            kw = {}
            if method == "replay" and buf[0]: kw["replay"] = (torch.cat(buf[0]), torch.cat(buf[1]))
            if method == "ewc" and ewc: kw["ewc"] = ewc
            if method == "projected": kw["basis"] = basis
            if method == "invariant_constraints" and anchors: kw["anchors"] = (torch.cat([a for a, _ in anchors]), torch.cat([o for _, o in anchors]))
            p = mono_train(p, Z, y, steps, seed=seed + i, **kw); compute += steps
            if method == "replay":
                buf[0].append(Z[:32]); buf[1].append(y[:32]); mem_bytes += 32 * (Z.shape[1] + 1)                     # RAW data
            if method == "ewc":
                q = {k: v.clone().requires_grad_(True) for k, v in p.items()}
                g = torch.autograd.grad(torch.nn.functional.cross_entropy(mono_forward(q, Z), y), list(q.values()))
                ewc.append(({k: gi ** 2 for k, gi in zip(q, g)}, {k: v.clone() for k, v in p.items()})); mem_bytes += 2 * 4 * n0
            if method in ("projected", "invariant_constraints"):
                proto = C.mean(0).round(); cues.append(proto)
                Zp = probes_for(p, proto)
                if method == "projected": basis = jac_basis(p, Zp, basis); mem_bytes = 4 * basis.numel() + len(cues) * D_CUE
                else:
                    with torch.no_grad(): anchors.append((Zp, mono_forward(p, Zp)))
                    mem_bytes += Zp.shape[0] * (Zp.shape[1] + 2 * 4)
            if t.kind == "recurrence": recur.append({"latency_steps": steps, "reused": False})
            for j in range(i + 1): R[i][j] = mono_acc(p, *tests[j])
        growth = 1.0; extra.update({"rollbacks": 0, "reused_modules": 0, "grown_modules": 0})
    return metrics(R, tasks, refs(tasks, seed, steps, n_support)) | {"param_growth": round(growth, 3), "memory_bytes": int(mem_bytes),
                                                                     "compute_steps": int(compute), "recurrence": recur, **extra}


_REFS = {}


def refs(tasks, seed, steps, n_support):
    """Reference for the forward transfer: FRESH monolithic network trained alone on the task (same data)."""
    key = (tuple(t.name for t in tasks), seed)
    if key not in _REFS:
        rng = random.Random(seed); out = []
        for i, t in enumerate(tasks):
            X, C, y = t.sample(n_support, rng)
            out.append(mono_acc(mono_train(mono_init(seed + 999), torch.cat([X, C], 1), y, steps, seed=seed), *t.test_set(seed=100 + i)))
        _REFS[key] = out
    return _REFS[key]


def metrics(R, tasks, ref):
    T = len(tasks); final = R[T - 1]
    acc = sum(final) / T
    bwt = sum(final[j] - R[j][j] for j in range(T - 1)) / max(1, T - 1)
    fgt = sum(max(R[i][j] for i in range(j, T - 1)) - final[j] for j in range(T - 1)) / max(1, T - 1)
    fwt = sum(R[j][j] - ref[j] for j in range(T)) / T
    learn = sum(R[j][j] for j in range(T)) / T
    return {"average_accuracy": round(acc, 4), "new_task": round(learn, 4), "backward_transfer": round(bwt, 4), "forward_transfer": round(fwt, 4),
            "forgetting": round(fgt, 4), "retention": round(sum(final[:-1]) / max(1, T - 1), 4)}


def benchmark(seed=0, n_tasks=14, methods=METHODS):
    tasks = task_stream(seed, n_tasks)
    return {m: run(m, tasks, seed) for m in methods}

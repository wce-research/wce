"""LEARNED stabilizers: P(invariant preserved | transformation family, context, magnitude).

Data: real SkillNet states (after a sequence of tasks) × adaptation transformations drawn at random
(15 families) × certified skills; label = does the certificate of the skill (and the behaviour of its
context) hold after the transformation? Model: logistic regression (small, interpretable).
Compared with a HAND-WRITTEN stabilizer (per-family rules) and with a HYBRID (rule when it is certain
by construction, learned model otherwise). Generalisation: unseen magnitudes, modules, parameters and compositions.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, List

import torch

from core.compositional_plasticity import PlannerConfig, PlasticityPlanner, _restore, _snapshot
from core.skillnet import (
    ALL_DATA, FAMILIES, AddConnection, AddModule, AllocateSubspace, CloneModule, ComposeSkills, CreateMemory, DistillSkill,
    FastWeightUpdate, FreezeSubspace, MergeModule, ParameterUpdate, RemoveConnection, ReuseSkill, RouteChange, SkillNet,
    SplitModule, skill_modules, task_stream,
)

BY_CONSTRUCTION = {"AddModule", "CloneModule", "SplitModule", "FreezeSubspace", "AllocateSubspace", "CreateMemory", "ComposeSkills", "ReuseSkill"}


def built_state(seed, n_tasks=10):
    net = SkillNet(seed=seed); certs = {}; pl = PlasticityPlanner(PlannerConfig(train_steps=200), seed=seed); rng = random.Random(seed)
    for t in task_stream(seed, n_tasks):
        X, C, y = t.sample(128, rng); pl.learn(net, X, C, y, certs)
    return net, certs


def _rel_delta(p, mag, g):
    out = {}
    for k, v in p.items():
        d = torch.randn(v.shape, generator=g); out[k] = mag * float(v.norm().clamp_min(1e-6)) * d / d.norm().clamp_min(1e-12)
    return out


def sample_transformation(net, rng: random.Random, mag_range=(0.01, 0.2), modules=None):
    """Draws an applicable transformation; returns (family, target, magnitude, spec)."""
    fam = rng.choice(FAMILIES); g = torch.Generator().manual_seed(rng.randrange(10 ** 9))
    mods = [m for m in (modules or list(net.modules)) if "w1" in net.modules[m]] or [m for m in net.modules if "w1" in net.modules[m]]
    m = rng.choice(mods); mag = math.exp(rng.uniform(math.log(mag_range[0]), math.log(mag_range[1])))
    skills = list(net.skills); comps = [s for s, p in net.skills.items() if p[0] == "compose"]
    if fam == "ParameterUpdate": return fam, m, mag, ParameterUpdate(module=m, delta=_rel_delta(net.modules[m], mag, g))
    if fam == "FastWeightUpdate": return fam, m, mag, FastWeightUpdate(module=m, delta=_rel_delta(net.modules[m], mag, g))
    if fam == "RouteChange":
        c = rng.choice(net.contexts); return fam, c.name, 1.0, RouteChange(context=c.name, skill=rng.choice(skills))
    if fam == "AddConnection" and comps:
        s = rng.choice(comps); return fam, s, 1.0, AddConnection(skill=s, slot=rng.choice([2, 3]), source=rng.choice(skills))
    if fam == "RemoveConnection" and comps:
        s = rng.choice(comps); return fam, s, 1.0, RemoveConnection(skill=s, fallback_module=m)
    if fam == "MergeModule":
        a, b = rng.sample(mods, 2) if len(mods) > 1 else (m, m)
        return fam, b, 1.0, MergeModule(a=a, b=b)
    if fam == "DistillSkill":
        s = rng.choice(skills); return fam, s, 1.0, DistillSkill(skill=s, target=s, module=f"d_{s}", hidden=rng.choice([2, 4, 8]), steps=rng.choice([20, 80, 300]), seed=0)
    if fam == "AddModule": return fam, "new", 1.0, AddModule(module="new_m", hidden=8, seed=0)
    if fam == "CloneModule": return fam, m, 1.0, CloneModule(source=m, new=m + "_c")
    if fam == "SplitModule":
        h = net.modules[m]["w1"].shape[0]; return fam, m, 1.0, SplitModule(module=m, units=list(range(h // 2)), new_a=m + "_a", new_b=m + "_b")
    if fam == "FreezeSubspace": return fam, m, 1.0, FreezeSubspace(module=m, basis=torch.zeros(1, 1))
    if fam == "AllocateSubspace": return fam, m, 1.0, AllocateSubspace(module=m, basis=torch.zeros(1, 1))
    if fam == "CreateMemory": return fam, "mem", 1.0, CreateMemory(skill="x", certificate=next(iter(net.memories.values())) if net.memories else None)
    if fam == "ComposeSkills": return fam, "new", 1.0, ComposeSkills(name_="new_comp", op="and", a=skills[0], b=skills[-1])
    if fam == "ReuseSkill":
        c = rng.choice(net.contexts); return "RouteChange", c.name, 1.0, RouteChange(context=c.name, skill=rng.choice(skills))
    return "ParameterUpdate", m, mag, ParameterUpdate(module=m, delta=_rel_delta(net.modules[m], mag, g))


def _context_of(net, skill):
    return [c.name for c in net.contexts if c.skill == skill]


def features(net, certs, fam, target, mag, skill):
    cert = certs[skill]; mods = skill_modules(net, skill)
    involved = float(target in mods or target == skill or target in _context_of(net, skill) or
                     (net.skills.get(target, ("",))[0] == "compose" and skill in str(net.skills.get(target))))
    depends = float(any(target == s or target in str(net.skills.get(skill)) for s in [target]))
    shared = len(net.module_users(target)) if target in net.modules else 0
    try:
        with torch.no_grad(): o = net.skill_logits(skill, cert.probes); margin = float((o[:, 1] - o[:, 0]).abs().mean())
    except (KeyError, RuntimeError): margin = 0.0
    r = cert.stability_radius
    f = [1.0] + [float(fam == x) for x in FAMILIES]
    f += [involved, depends, math.log(mag + 1e-4), involved * math.log(mag + 1e-4), involved * mag / max(r, 0.025),
          r, min(margin, 20.0) / 20.0, involved * min(margin, 20.0) / 20.0, float(shared), float(net.skills[skill][0] != "expert")]
    f += [involved * float(fam == x) for x in FAMILIES]
    return f


def label(net, certs, skill, ctx_before):
    ok, _ = certs[skill].check(net)
    for c in net.contexts:
        if c.name in ctx_before and ctx_before[c.name] == skill and c.skill != skill:
            try:
                if c.skill is None or (net.skill_predict(c.skill, certs[skill].probes) != certs[skill].labels).any(): ok = False
            except (KeyError, RuntimeError): ok = False
    return ok


def hand_rule(net, certs, fam, target, skill):
    """Hand-written stabilizer: preservation by construction, otherwise "touches the skill ⇒ broken"."""
    if fam in BY_CONSTRUCTION: return 1.0
    mods = skill_modules(net, skill)
    if fam in ("ParameterUpdate", "FastWeightUpdate"): return 0.0 if target in mods else 1.0
    if fam == "RouteChange": return 0.0 if target in _context_of(net, skill) else 1.0
    if fam in ("AddConnection", "RemoveConnection"): return 0.0 if (target == skill or target in str(net.skills.get(skill))) else 1.0
    if fam == "MergeModule": return 0.0 if target in mods else 1.0
    if fam == "DistillSkill": return 0.0 if target == skill else 1.0
    return 0.5


def simulate(net, certs, n, seed=0, mag_range=(0.01, 0.2), modules=None):
    rng = random.Random(seed); rows = []
    for _ in range(n):
        fam, target, mag, spec = sample_transformation(net, rng, mag_range, modules)
        ctx_before = {c.name: c.skill for c in net.contexts}
        pre = {s: (features(net, certs, fam, target, mag, s), hand_rule(net, certs, fam, target, s)) for s in certs if s in net.skills}
        snap = _snapshot(net); fast = {k: dict(v) for k, v in net.fast.items()}
        try: spec.execute(net)
        except Exception: _restore(net, snap); net.fast = fast; continue
        for s, (f, h) in pre.items():
            rows.append({"family": fam, "target": target, "mag": mag, "skill": s, "features": f, "hand": h, "label": float(label(net, certs, s, ctx_before))})
        _restore(net, snap); net.fast = fast
        for m in [m for m in net.modules if m not in snap[0]]: net.modules.pop(m)
    return rows


class LearnedStabilizerModel:
    def __init__(self, l2=1e-2, steps=600, lr=0.05): self.l2, self.steps, self.lr = l2, steps, lr; self.w = None

    def fit(self, rows):
        X = torch.tensor([r["features"] for r in rows]); y = torch.tensor([r["label"] for r in rows])
        w = torch.zeros(X.shape[1], requires_grad=True); opt = torch.optim.Adam([w], lr=self.lr)
        for _ in range(self.steps):
            opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w, y) + self.l2 * (w[1:] ** 2).sum()
            loss.backward(); opt.step()
        self.w = w.detach(); return self

    def predict_rows(self, rows):
        X = torch.tensor([r["features"] for r in rows]); return torch.sigmoid(X @ self.w).tolist()

    def predict(self, net, certs, fam, target, mag, skill):
        return float(torch.sigmoid(torch.tensor(features(net, certs, fam, target, mag, skill)) @ self.w))

    def predict_family(self, fam, net, module, mag=0.1):
        certs = {s: c for s, c in net.memories.items() if hasattr(c, "probes") and s in net.skills}
        if not certs: return 1.0
        p = 1.0
        for s in certs: p = min(p, self.predict(net, certs, fam, module, mag, s))
        return p


def scores(pred, rows):
    y = [r["label"] for r in rows]; n = max(1, len(y))
    acc = sum((p >= 0.5) == (t >= 0.5) for p, t in zip(pred, y)) / n
    brier = sum((p - t) ** 2 for p, t in zip(pred, y)) / n
    fs = sum(1 for p, t in zip(pred, y) if p >= 0.5 and t < 0.5) / n                    # "preserved" wrongly predicted (dangerous)
    return {"accuracy": round(acc, 4), "brier": round(brier, 4), "false_safe": round(fs, 4), "n": len(y)}


def compare(model, rows):
    learned = model.predict_rows(rows); hand = [r["hand"] for r in rows]
    hybrid = [hybrid_value(h, l, r["family"]) for h, l, r in zip(hand, learned, rows)]
    return {"hand_written": scores(hand, rows), "learned": scores(learned, rows), "hybrid": scores(hybrid, rows)}


def hybrid_value(hand, learned, family):
    """Hybrid: the rule when it guarantees preservation (by construction / component not touched);
    the learned model when the rule concludes "touched" (it estimates whether the magnitude stays in the stable zone)."""
    if family in BY_CONSTRUCTION or hand >= 1.0: return min(hand, 1.0) if family not in BY_CONSTRUCTION else 1.0
    return learned


def composition_test(model, net, certs, n=150, seed=7):
    """Pairs of transformations: predicted survival = product of the survivals (independence) vs actual survival."""
    rng = random.Random(seed); rows = []
    for _ in range(n):
        a = sample_transformation(net, rng); b = sample_transformation(net, rng)
        ctx_before = {c.name: c.skill for c in net.contexts}; ss = [s for s in certs if s in net.skills]
        pa = {s: model.predict(net, certs, a[0], a[1], a[2], s) for s in ss}; ha = {s: hand_rule(net, certs, a[0], a[1], s) for s in ss}
        snap = _snapshot(net); fast = {k: dict(v) for k, v in net.fast.items()}
        try:
            a[3].execute(net)
            pb = {s: model.predict(net, certs, b[0], b[1], b[2], s) if s in net.skills else 0.0 for s in ss}
            hb = {s: hand_rule(net, certs, b[0], b[1], s) if s in net.skills else 0.0 for s in ss}
            b[3].execute(net)
            for s in ss: rows.append({"family": "pair", "pred": pa[s] * pb[s], "hand": ha[s] * hb[s],
                                      "hyb": hybrid_value(ha[s], pa[s], a[0]) * hybrid_value(hb[s], pb[s], b[0]),
                                      "label": float(s in net.skills and label(net, certs, s, ctx_before))})
        except Exception: pass
        _restore(net, snap); net.fast = fast
        for m in [m for m in net.modules if m not in snap[0]]: net.modules.pop(m)
    return {"learned": scores([r["pred"] for r in rows], rows), "hand_written": scores([r["hand"] for r in rows], rows),
            "hybrid": scores([r["hyb"] for r in rows], rows)}


def generalization_experiment(train_seeds=(0, 1, 2), test_seeds=(3, 4), n=250):
    """Training: magnitudes 0.01–0.2 on the states of the training seeds. Tests: (a) unseen magnitudes
    0.3–1.5; (b) unseen modules / parameters (states of seeds never seen); (c) unseen compositions."""
    train, states = [], {}
    for s in train_seeds:
        net, certs = built_state(s); states[s] = (net, certs); train += simulate(net, certs, n, seed=s)
    model = LearnedStabilizerModel().fit(train)
    out = {"train": compare(model, train), "n_train": len(train)}
    big = []
    for s in train_seeds: big += simulate(*states[s], n // 2, seed=100 + s, mag_range=(0.3, 1.5))
    out["unseen_magnitudes"] = compare(model, big)
    unseen = []
    for s in test_seeds:
        net, certs = built_state(s); states[s] = (net, certs); unseen += simulate(net, certs, n, seed=200 + s)
    out["unseen_modules_params"] = compare(model, unseen)
    out["unseen_compositions"] = composition_test(model, *states[test_seeds[0]])
    return model, out

"""Targeted experiments of the milestone: capacity pressure (H6), regeneration 2.0 (H7), adversarial probes,
open-world novelty (AB), latent structure discovery (AC), skill induction (AD), common cross-domain
certificate format (AI), growth and reuse laws (AM, AN), forward transfer (AO–AQ)."""
from __future__ import annotations

import copy
import math
import random
from collections import Counter

import torch

from core.continuous_world import stream
from core.continuum import Config, ContinuumSystem, mlp, run_stream
from core.generative_certificate import (ContinuousGenerator, GenerativeInvariantCertificate, SymbolicGenerator, clean_inputs)


# ================================================================ H6 — capacity pressure

def capacity_both(seed=0):
    """Two streams under pressure: "neural" (families outside the library, budget 1300) and "mixed" (standard stream,
    discovery at acquisition disabled to force experts, budget 400)."""
    a = capacity_experiment(seed)
    regs = stream(seed + 700, 40); out = {}
    for name, cap in (("unlimited", None), ("pressure", 400)):
        cfg = Config("cap", cert="generative", discovery=False, drift="auto", capacity=cap, capacity_mgmt=cap is not None)
        o, s = run_stream_cfg(cfg, regs, seed); caps = Counter()
        for r in s.log:
            for k, v in r.get("capacity", {}).items(): caps[k] += v
        out[name] = {"accuracy": o["average_accuracy"], "params": s.neural_params(), "operations": dict(caps), "distill_rejected": s.counters.get("distill_rejected", 0),
                     "budget_exceeded_at_end": cap is not None and s.neural_params() > cap}
    out["accuracy_loss"] = round(out["unlimited"]["accuracy"] - out["pressure"]["accuracy"], 4)
    return {"neural_only": a, "mixed": out}


def capacity_experiment(seed=0, n=40, budget=1300, drift="auto"):
    """Stream of families NOT covered by the rule library (relative distance, sine, box): the neural
    experts are necessary. Same configuration with tight vs unlimited budget. drift="regrow": the old
    drifted versions become obsolete (unreferenced) — a situation where pruning has material to work on."""
    regs = stream(seed + 300, n, families=("reldist", "box", "sine"), p_novel=0.0)
    out = {}
    for name, cap in (("unlimited", None), ("pressure", budget)):
        cfg = Config("cap", cert="generative", discovery=False, drift=drift, capacity=cap, capacity_mgmt=cap is not None)
        o, s = run_stream_cfg(cfg, regs, seed)
        caps = Counter()
        for r in s.log:
            for k, v in r.get("capacity", {}).items(): caps[k] += v
        out[name] = {"accuracy": o["average_accuracy"], "new_task": o["new_task"], "params": s.neural_params(), "peak_params": max(r.get("params_after", 0) for r in s.log),
                     "operations": dict(caps), "distill_rejected": s.counters.get("distill_rejected", 0), "distill_params_saved": s.counters.get("distill_params_saved", 0),
                     "budget_exceeded_at_end": cap is not None and s.neural_params() > cap}
    out["accuracy_loss"] = round(out["unlimited"]["accuracy"] - out["pressure"]["accuracy"], 4)
    return out


def run_stream_cfg(cfg, regs, seed, n_support=512):
    from core.continuum import checkpoint_metrics
    s = ContinuumSystem(cfg, seed); g = torch.Generator().manual_seed(seed + 11); tests = [r.test_set(seed * 1000 + i, 384) for i, r in enumerate(regs)]; la = []; recs = []
    for i, r in enumerate(regs):
        X, C, y = r.sample(n_support, g); recs.append(s.learn(X, C, y, t=i))
        la.append(float((s.predict(*tests[i][:2]) == tests[i][2]).float().mean()))
    return checkpoint_metrics(s, regs, tests, la, recs), s


# ================================================================ H7 — regeneration 2.0

def regeneration_experiment(seed=0, n=30):
    """System (neural experts necessary); each module corrupted in turn → failing certificates →
    regeneration from the generative certificates (without snapshot) → recovery on the protected skills."""
    regs = stream(seed + 400, n, families=("reldist", "box", "sine", "ball"), p_novel=0.0)
    cfg = Config("regen", cert="generative", discovery=False, drift="auto")
    o, s = run_stream_cfg(cfg, regs, seed)
    tests = {}
    for i, r in enumerate(regs): tests.setdefault(r.context_id, r.test_set(seed * 77 + i, 384))
    res = []
    for m in [m for m in s.modules if "w1" in s.modules[m]]:
        deps = s.dependents(m); ctxs = [c for c in s.contexts if c.skill in deps]
        if not ctxs: continue
        def acc(sys_):
            vals = []
            for c in ctxs:
                cid = int(c.name[3:]); X, C, y = tests[cid] if cid in tests else (None, None, None)
                if X is None: continue
                vals.append(float((sys_.predict_skill(c.skill, X) == y).float().mean()))
            return sum(vals) / max(1, len(vals))
        before = acc(s); s2 = copy.deepcopy(s); g = torch.Generator().manual_seed(len(m))
        s2.modules[m] = {k: v + 1.5 * v.std() * torch.randn(v.shape, generator=g) for k, v in s2.modules[m].items()}
        corrupted = acc(s2); failing = not s2.check_certs(deps); ok = s2.regenerate(m); after = acc(s2)
        res.append({"module": m, "before": before, "corrupted": corrupted, "regenerated": after, "detected": failing, "certificates_pass": ok,
                    "recovery": after / max(1e-9, before)})
    rec = sorted(r["recovery"] for r in res)
    return {"modules": len(res), "median_recovery": rec[len(rec) // 2] if rec else None, "mean_recovery": round(sum(rec) / max(1, len(rec)), 4),
            "detected": round(sum(r["detected"] for r in res) / max(1, len(res)), 3), "certificates_pass": round(sum(r["certificates_pass"] for r in res) / max(1, len(res)), 3),
            "before": round(sum(r["before"] for r in res) / max(1, len(res)), 4), "corrupted": round(sum(r["corrupted"] for r in res) / max(1, len(res)), 4),
            "regenerated": round(sum(r["regenerated"] for r in res) / max(1, len(res)), 4), "per_module": res}


# ================================================================ adversarial probes and violation rate

def adversarial_experiment(seed=0, n=30):
    """For each certified skill (neural expert): violation rate found by (a) central probes,
    (b) boundary probes, (c) ADVERSARIAL search (maximal disagreement with the reference)."""
    regs = stream(seed + 500, n, families=("reldist", "box", "sine", "ball"), p_novel=0.0)
    o, s = run_stream_cfg(Config("adv", cert="generative", discovery=False), regs, seed)
    g = torch.Generator().manual_seed(seed); out = Counter()
    for name, c in s.certs.items():
        if not isinstance(c, GenerativeInvariantCertificate) or s.skills.get(name, ("",))[0] != "expert": continue
        p = s.modules[s.skills[name][1]]; pred = lambda X: s.predict_skill(name, X)
        cen = c.generator.sample(64, g); bnd = c.generator.sample_boundary(64, g, c.reference)
        adv = c.generator.sample_adversarial(64, g, lambda X: mlp(p, X), c.reference, start=bnd.clone())
        for k, X in (("central", cen), ("boundary", bnd), ("adversarial", adv)): out[k] += float((pred(X) != c.reference(X)).float().mean())
        out["n"] += 1
    return {k: round(v / max(1, out["n"]), 4) for k, v in out.items() if k != "n"} | {"skills": out["n"]}


# ================================================================ AB — open-world novelty

def novelty_classification(seed=0, n=60):
    """Decision of the system vs truth: known recurrence / composition / small drift / new task / new
    domain (family outside the library)."""
    regs = stream(seed + 600, n, p_novel=0.15)
    o, s = run_stream_cfg(Config("open", cert="generative", discovery=True, drift="auto"), regs, seed)
    def sysclass(r):
        k = r["kind"]
        if k in ("reactivate", "noise_ignored"): return "recurrence"
        if k in ("refine", "refine_rule", "refine_adapter") or (r.get("change") == "drift"): return "drift"
        if k == "compose" or (k == "reuse" and r.get("new_context")): return "composition"
        if k in ("rule",): return "new_task"
        if k in ("grow", "adapter"): return "new_domain"
        return "new_task"
    truth_map = {"recurrence": "recurrence", "drift": "drift", "composition": "composition", "new": "new_task", "new_domain": "new_domain"}
    conf = Counter(); ok = 0
    for r, rec in zip(regs, s.log):
        t = truth_map[r.kind]; p = sysclass(rec); conf[(t, p)] += 1; ok += t == p
    per = {}
    for t in set(truth_map.values()):
        tot = sum(v for (a, b), v in conf.items() if a == t); per[t] = round(conf[(t, t)] / max(1, tot), 3)
    return {"accuracy": round(ok / len(regs), 3), "per_class_recall": per, "confusion": {f"{a}->{b}": v for (a, b), v in conf.items()}}


# ================================================================ AC — latent structure: relevant factors

def latent_factors(seed=0, n=12):
    """For new families: discovered relevant factors = dimensions with high sensitivity (mean input
    gradient of the expert) compared with the dimensions actually used by the hidden rule."""
    from core.continuous_world import Regime, random_rule, D
    from core.continuum import new_expert, train
    rng = random.Random(seed); g = torch.Generator().manual_seed(seed); hits = []
    for _ in range(n):
        fam = rng.choice(["reldist", "box", "sine", "sector", "frame"]); mu = torch.zeros(D)
        r = Regime("x", random_rule(rng, fam, mu), mu, 1.0, torch.zeros(8), "new"); X, C, y = r.sample(768, g, noisy=False)
        p = train(new_expert(0), X, y, 400); Xr = X.clone().requires_grad_(True); o = mlp(p, Xr); (o[:, 1] - o[:, 0]).abs().sum().backward()
        sal = Xr.grad.abs().mean(0); true = set(_used_dims(r.rule)); k = len(true)
        found = set(sal.argsort(descending=True)[:k].tolist()); hits.append(len(found & true) / k)
    return {"factor_recall": round(sum(hits) / len(hits), 3), "n": n}


def _used_dims(rule):
    f, p = rule.family, rule.params
    if f in ("sector", "frame", "box", "sine"): return [p[0], p[1]]
    if f == "reldist": return list(p[:4])
    if f == "ball": return list(p[0])
    return list(range(16))


# ================================================================ AI — common cross-domain certificate format

def cross_domain_certificates(seed=0):
    """Same GenerativeInvariantCertificate for language, algebra, mini-code, sequences: SYMBOLIC generator learned
    by anti-unification of the seen examples, reference = canonical program of the skill; we check (a) that an
    intact skill passes, (b) that an altered skill (primitive replaced) fails, (c) the compression."""
    from core.symbolic_plasticity import DOMAINS, Prim
    rng = random.Random(seed); g = torch.Generator().manual_seed(seed); out = {}
    tok = {"language": (str.split, " ".join, lambda s: s), "algebra": None, "code": None, "strings": (list, "".join, lambda s: s)}
    for name, mk in DOMAINS.items():
        dom = mk(); xs = dom.inputs(rng, 30); prims = dom.prims(); prog = (rng.choice(prims),)
        if name in ("algebra", "code"):
            gen = _StructGenerator(dom, xs)
        else:
            t, d, rb = tok[name]; gen = SymbolicGenerator(t, d, rb).fit(xs)
        ref = lambda X, prog=prog, dom=dom: [str(dom.run(prog, x)) for x in X]
        bad_prog = (next(p for p in prims if p != prog[0]),)
        def check(fn):
            X = gen.sample(40, g); return sum(a == b for a, b in zip(fn(X), ref(X))) / len(X)
        intact = check(lambda X: [str(dom.run(prog, x)) for x in X]); broken = check(lambda X: [str(dom.run(bad_prog, x)) for x in X])
        raw = sum(len(str(x)) for x in xs)
        out[name] = {"intact_agreement": round(intact, 3), "altered_agreement": round(broken, 3), "detects_alteration": broken < 0.97,
                     "raw_bytes": raw, "generator_bytes": gen.memory_bytes()}
    return out


class _StructGenerator(SymbolicGenerator):
    """Structural generator for non-textual objects (equations, programs): we resample through the
    input generator of the domain with learned parameters (bounds) — here, the domain sampler, bounded."""
    def __init__(self, dom, xs): super().__init__(); self.dom = dom; self.n_seen = len(xs)
    def sample(self, n, g, rare=False): return self.dom.inputs(random.Random(int(torch.randint(0, 10 ** 9, (1,), generator=g))), n)
    def memory_bytes(self): return 64


# ================================================================ AM / AN / AO — empirical laws

def fit_growth(ns, params):
    """Fits params(N) = a·N^b (log-log) and linear; returns b, R² of both, and the rough CI of b."""
    xs = [math.log(n) for n in ns]; ys = [math.log(max(1, p)) for p in params]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys); sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx; a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(xs, ys)]; ss = sum(r * r for r in res); st = sum((y - my) ** 2 for y in ys) or 1e-9
    se = math.sqrt(ss / max(1, len(xs) - 2) / sxx) if len(xs) > 2 else float("nan")
    lm = sum(ns) / len(ns); lp = sum(params) / len(params); slope = sum((n - lm) * (p - lp) for n, p in zip(ns, params)) / max(1e-9, sum((n - lm) ** 2 for n in ns))
    lres = [p - (lp + slope * (n - lm)) for n, p in zip(ns, params)]; lst = sum((p - lp) ** 2 for p in params) or 1e-9
    return {"power_exponent": round(b, 3), "exponent_se": round(se, 3), "power_r2": round(1 - ss / st, 3), "linear_r2": round(1 - sum(r * r for r in lres) / lst, 3)}


def lifetime_cost(learning_curve, kinds, window=10):
    """Acquisition cost (training steps) per NEW skill along the lifetime + slope."""
    pts = [(i, c) for i, (c, k) in enumerate(zip(learning_curve, kinds)) if k not in ("reactivate", "noise_ignored")]
    if len(pts) < 3: return {"slope": None}
    xs, ys = zip(*pts); mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(1e-9, sum((x - mx) ** 2 for x in xs))
    early = [c for i, c in pts if i < len(learning_curve) // 4]; late = [c for i, c in pts if i >= 3 * len(learning_curve) // 4]
    return {"slope": round(slope, 4), "early_mean": round(sum(early) / max(1, len(early)), 1), "late_mean": round(sum(late) / max(1, len(late)), 1)}


def reuse_law(kinds, bins=4):
    n = len(kinds); out = []
    for b in range(bins):
        seg = [k for k in kinds[b * n // bins:(b + 1) * n // bins] if k not in ("reactivate", "noise_ignored")]
        out.append(round(sum(k in ("reuse", "compose", "refine", "refine_rule", "refine_adapter", "adapter") for k in seg) / max(1, len(seg)), 3))
    return out

"""Compared methods and evaluation of a boundary task.

1 memorization (1-NN)  2 kNN (k=7)  3 small MLP (comparable capacity)  4 "known family" oracle (true skeleton
provided: UPPER BOUND, is not an induction method)  5 primitive search WITHOUT anti-unification
6 full induction (induced family macros first)  7 induction + active experiments  8 program + residual.
"""
from __future__ import annotations

import math
import time
from collections import Counter

import torch

from core.boundary_program import (BoundaryProgramSearch, ProgramPosterior, canonical, complexity, fit, predict, program_hash, render,
                                   structure_hash)
from core.boundary_world import full_region, in_region, out_region

METHODS = ("memorization", "knn", "mlp", "oracle", "search_no_au", "bpi_full", "bpi_active", "program_residual")


def knn_predict(X, y, Q, k=1):
    d = torch.cdist(torch.nan_to_num(Q), torch.nan_to_num(X)); idx = d.topk(k, largest=False).indices
    return (y[idx].float().mean(1) >= 0.5).long()


def train_mlp(X, y, h=8, steps=600, seed=0, lr=0.03):
    g = torch.Generator().manual_seed(seed)
    p = {"w1": torch.randn(h, 2, generator=g) * 0.7, "b1": torch.zeros(h), "w2": torch.randn(2, h, generator=g) * 0.4, "b2": torch.zeros(2)}
    q = {k: v.clone().requires_grad_(True) for k, v in p.items()}; opt = torch.optim.Adam(q.values(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(); torch.nn.functional.cross_entropy(mlp_logits(q, X), y).backward(); opt.step()
    return {k: v.detach() for k, v in q.items()}


def mlp_logits(p, X): return torch.tanh(torch.nan_to_num(X) @ p["w1"].T + p["b1"]) @ p["w2"].T + p["b2"]
def mlp_params(p): return sum(v.numel() for v in p.values())


def train_residual(prog, X, y, h=3, steps=500, seed=0, scale=4.0):
    """Neural residual: logit = scale·(2·program − 1) + r(x); only r is trained (program frozen)."""
    g = torch.Generator().manual_seed(seed); base = (2 * predict(prog, X).float() - 1) * scale
    p = {"w1": torch.randn(h, 2, generator=g) * 0.7, "b1": torch.zeros(h), "w2": torch.randn(1, h, generator=g) * 0.1, "b2": torch.zeros(1)}
    q = {k: v.clone().requires_grad_(True) for k, v in p.items()}; opt = torch.optim.Adam(q.values(), lr=0.03)
    for _ in range(steps):
        opt.zero_grad(); r = (torch.tanh(X @ q["w1"].T + q["b1"]) @ q["w2"].T + q["b2"])[:, 0]
        torch.nn.functional.binary_cross_entropy_with_logits(base + r, y.float()).backward(); opt.step()
    r = {k: v.detach() for k, v in q.items()}
    def f(Q):
        Q = torch.nan_to_num(Q); b = (2 * predict(prog, Q).float() - 1) * scale
        return ((b + (torch.tanh(Q @ r["w1"].T + r["b1"]) @ r["w2"].T + r["b2"])[:, 0]) >= 0).long()
    return f, sum(v.numel() for v in r.values())


def split(X, y, frac=0.75):
    n = int(frac * len(X)); return X[:n], y[:n], X[n:], y[n:]


STABILITY_MIN = 0.86        # 5th percentile of the stability on TRAINING + VALIDATION families (results/boundary_stability_calibration.json)
VAL_MIN = 0.90


def stable_search(X, y, macros=None, seed=0, **kw):
    """Search on the whole set, plus a STABILITY test: two searches on disjoint halves must
    produce programs that functionally agree on the observed region (≥ 0.9); otherwise "no adequate
    program" (no stable structure explaining the data)."""
    Xt, yt, Xv, yv = split(X, y); s = BoundaryProgramSearch(macros=macros, seed=seed, **kw).run(Xt, yt, Xv, yv)
    n = len(X) // 2
    a = BoundaryProgramSearch(macros=macros, seed=seed + 1, **kw).run(*split(X[:n], y[:n]))
    b = BoundaryProgramSearch(macros=macros, seed=seed + 2, **kw).run(*split(X[n:], y[n:]))
    stab = 0.0
    if a.best is not None and b.best is not None:
        R = X; stab = float((predict(a.best, R) == predict(b.best, R)).float().mean())
    s.stats["stability"] = round(stab, 3); s.stats["evaluated"] = s.stats.get("evaluated", 0) + a.stats.get("evaluated", 0) + b.stats.get("evaluated", 0)
    s.adequate = bool(s.best is not None and s.val_acc >= VAL_MIN and stab >= STABILITY_MIN)
    return s


def active_refine(search, inst, X, y, g, budget=12, policy="ig", noise=0.03, refit_every=4):
    """Active experiments on the posterior over programs: point of max information gain (or max
    disagreement, or random) in the FULL region; observed label (noisy); every `refit_every`, the
    candidate skeletons are refitted on data + queries. Returns (MAP program, posterior, number of queries)."""
    cands = [c[0] for c in search.candidates][:8]; Xa, ya = X.clone(), y.clone()
    post = ProgramPosterior(cands); post.update(Xa, ya); pool = full_region(600, g); n = 0
    for q in range(budget):
        if policy == "ig": sc = post.info_gain(pool)
        elif policy == "disagreement": sc = post.disagreement(pool)
        else: sc = torch.rand(len(pool), generator=g).double()
        i = int(sc.argmax()); xq = pool[i:i + 1]; yq = inst.label(xq)
        if float(torch.rand(1, generator=g)) < noise: yq = 1 - yq
        Xa, ya = torch.cat([Xa, xq]), torch.cat([ya, yq]); n += 1
        if (q + 1) % refit_every == 0:
            cands = [fit(c.skeleton(), Xa, ya, steps=100, n_init=1, seed=q)[0] for c in post.programs]
            post = ProgramPosterior(cands + post.programs); post.update(Xa, ya)
        else: post.update(xq, yq)
    return post.map()[0], post, n


def run_shared(inst, X, y, g, macros=None, active_budget=12, search_kw=None):
    """A single search per variant (without / with macros) shared by the induction methods."""
    kw = dict(search_kw or {}); out = {}
    t0 = time.time(); s0 = stable_search(X, y, None, **kw); t_no = time.time() - t0
    t0 = time.time(); s1 = stable_search(X, y, macros, **kw) if macros else s0; t_full = (time.time() - t0) if macros else t_no
    out["search_no_au"] = (s0, t_no); out["bpi_full"] = (s1, t_full)
    t0 = time.time(); prog, post, nq = active_refine(s1, inst, X, y, g, active_budget) if s1.candidates else (s1.best, None, 0)
    out["bpi_active"] = (s1, t_full + time.time() - t0, prog, post, nq)
    return out


def run_method(method, inst, X, y, g, macros=None, active_budget=12, query_noise=0.03, search_kw=None, shared=None):
    """Returns a dict: predictor, program if any, parameters, costs, adequacy."""
    t0 = time.time(); out = {"program": None, "adequate": True, "evaluations": 0, "queries": 0}
    if method == "memorization": out.update(pred=lambda Q: knn_predict(X, y, Q, 1), params=X.numel() + y.numel())
    elif method == "knn": out.update(pred=lambda Q: knn_predict(X, y, Q, 7), params=X.numel() + y.numel())
    elif method == "mlp":
        p = train_mlp(X, y); out.update(pred=lambda Q: mlp_logits(p, Q).argmax(1), params=mlp_params(p))
    elif method == "oracle":
        if inst.oracle is None: return None
        prog, _ = fit(inst.oracle, X, y, steps=200, n_init=4); out.update(pred=lambda Q: predict(prog, Q), params=prog.n_params(), program=prog)
    else:
        shared = shared or run_shared(inst, X, y, g, macros, active_budget, search_kw)
        key = "bpi_full" if method == "program_residual" else method
        s, secs = shared[key][0], shared[key][1]; prog = s.best
        out["evaluations"] = s.stats.get("evaluated", 0); out["adequate"] = s.adequate; out["stability"] = s.stats.get("stability")
        if method == "bpi_active":
            prog, post, out["queries"] = shared[key][2], shared[key][3], shared[key][4]
            if post is not None: out["posterior_top"] = post.map()[1]; out["ambiguous"] = post.ambiguous(full_region(400, g))
        if method == "program_residual" and prog is not None:
            f, rp = train_residual(prog, X, y); out.update(pred=f, params=prog.n_params() + rp, program=prog)
        else:
            out.update(pred=(lambda Q, prog=prog: predict(prog, Q)) if prog is not None else (lambda Q: torch.zeros(len(Q), dtype=torch.long)),
                       params=prog.n_params() if prog is not None else 0, program=prog)
        out["seconds"] = round(secs, 3); return out
    out["seconds"] = round(time.time() - t0, 3)
    return out


def evaluate_task(method, inst, X, y, g, macros=None, shared=None, **kw):
    r = run_method(method, inst, X, y, g, macros, shared=shared, **kw)
    if r is None: return None
    Xi, Xo, Xf = in_region(1500, g), out_region(1500, g), full_region(3000, g)
    acc = lambda Q: float((r["pred"](Q) == inst.label(Q)).float().mean())
    m = {"in_region": acc(Xi), "extrapolation": acc(Xo), "full": acc(Xf), "params": r["params"], "adequate": r["adequate"],
         "evaluations": r["evaluations"], "seconds": r["seconds"], "queries": r["queries"], "stability": r.get("stability")}
    prog = r.get("program")
    m["functional_recovery"] = m["full"] >= 0.97
    if prog is not None:
        m["program"] = render(prog); m["program_length"] = prog.size(); m["complexity"] = round(complexity(prog), 2)
        m["structural_recovery"] = inst.oracle is not None and structure_hash(prog) == structure_hash(inst.oracle)
    for k in ("posterior_top", "ambiguous"):
        if k in r: m[k] = r[k]
    return m


def evaluate_all(inst, X, y, g, macros=None, methods=METHODS, **kw):
    shared = run_shared(inst, X, y, g, macros, kw.get("active_budget", 12), kw.get("search_kw"))
    return {m: evaluate_task(m, inst, X, y, g, macros, shared=shared) for m in methods}

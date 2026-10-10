"""Experiments of the "Active Structural Discovery" milestone (one seed = one call; parallelised per seed)."""
from __future__ import annotations

import copy
import math
import random
import time
from collections import Counter, defaultdict

import torch

from core.active_structure import (version_space, ActiveStructuralDiscovery, EquivalenceSignature, NoiseAwareTruthAnchor, StructuralPosterior, gated_residual,
                                   identifiability, induce_ast_families, induce_orbit_families, minimal_distinguishing_set, noise_aware_stability,
                                   program_signature, reuse_family, sequential_truth_test, train_policy)
from core.active_world import (DEPTH_FAMILIES, HARD_NEGATIVES, MULTI_SYNTAX, NOISE_LEVELS, balanced_instances, noisy_task, train_tasks, transformed)
from core.boundary_methods import active_refine, stable_search, train_mlp, mlp_logits, mlp_params, train_residual, split
from core.boundary_program import BoundaryProgramSearch, fit, predict, render
from core.boundary_world import HELD_OUT, PARTIAL, TRAIN, UNREPRESENTABLE, VALIDATION, double_held_out, full_region, in_region, out_region
from core.truth_anchor import collect_counterexamples

N = 120
BUDGET = 16
VARIANTS = ("passive", "after_search", "interleaved", "interleaved_proposer", "interleaved_functional", "full", "full_random_queries")


def acc(pred, inst, F): return float((pred(F) == inst.label(F)).float().mean())


def metrics(prog_pred, inst, g):
    F, I, O = full_region(3000, g), in_region(1500, g), out_region(1500, g)
    a = acc(prog_pred, inst, F); return {"full": a, "recovery": a >= 0.97, "in_region": acc(prog_pred, inst, I), "extrapolation": acc(prog_pred, inst, O)}


def run_variant(v, inst, X, y, noise, policy, seed):
    g = torch.Generator().manual_seed(seed + 17); t0 = time.time()
    if v in ("passive", "after_search"):
        Xt, yt, Xv, yv = split(X, y); s = BoundaryProgramSearch(max_depth=3, seed=seed).run(Xt, yt, Xv, yv); prog = s.best; q = 0
        ev = s.stats.get("evaluated", 0)
        if v == "after_search" and s.candidates:
            prog, post, q = active_refine(s, inst, X, y, g, budget=BUDGET, noise=noise)
        stat = noise_aware_stability(prog, X, y, noise)["accept"] if prog is not None else False
        return prog, {"queries": q, "evaluations": ev, "seconds": round(time.time() - t0, 2), "status": "identified" if stat else "unsupported", "depth": prog.depth() if prog else 0}
    cfg = {"interleaved": (False, False, "structure_ig"), "interleaved_proposer": (True, False, "structure_ig"), "interleaved_functional": (False, True, "structure_ig"),
           "full": (True, True, "structure_ig"), "full_random_queries": (True, True, "random")}[v]
    asd = ActiveStructuralDiscovery(policy, guided=cfg[0], functional=cfg[1], query_policy=cfg[2], budget=BUDGET, eps=max(noise, 0.01), seed=seed); asd.eps_world = noise
    r = asd.run(X, y, inst.label, full_region, g)
    return r.program, {"queries": r.queries, "evaluations": r.evaluations, "seconds": r.seconds, "status": r.status, "depth": r.depth,
                       "entropy_curve": r.entropy_curve, "vs_curve": r.version_space_curve}


def main_tasks(seed):
    rng = random.Random(seed); T = []
    T += [(i, "held_out") for i in balanced_instances(HELD_OUT, seed + 3000, 1)]
    T += [(i, "depth34") for i in balanced_instances(DEPTH_FAMILIES, seed + 3001, 1)]
    T += [(i, "hard_negative") for i in balanced_instances(HARD_NEGATIVES, seed + 3002, 1)]
    T += [(i, "multi_syntax") for i in balanced_instances(MULTI_SYNTAX, seed + 3003, 1)]
    T += [(i, "unsupported") for i in balanced_instances(("sine5", "checker"), seed + 3004, 1)]
    T += [(double_held_out(rng.choice(["xor_halfspaces", "crescent"]), rng), "double_held_out")]
    return T


def main_job(seed, noise=0.05):
    policy = train_policy(train_tasks(seed + 500, noise), eps=noise, seed=seed); rows = []
    for k, (inst, cat) in enumerate(main_tasks(seed)):
        g = torch.Generator().manual_seed(seed * 1000 + k); X, y = noisy_task(inst, N, g, noise)
        for v in VARIANTS:
            prog, info = run_variant(v, inst, X, y, noise, policy, seed * 1000 + k)
            m = metrics(lambda Q, p=prog: predict(p, Q), inst, torch.Generator().manual_seed(seed + 7)) if prog is not None else {"full": 0, "recovery": False, "in_region": 0, "extrapolation": 0}
            rows.append({"family": inst.family, "category": cat, "variant": v, **m, **info, "program": render(prog) if prog is not None else None})
    return {"rows": rows, "policy_keys": len(policy.tot), "search_macros": policy.macros.most_common(5)}


# ================================================================ H2: query policies and version space

def query_policy_job(seed, noise=0.05, n_q=BUDGET):
    """FROZEN candidate set (passive search at depth 2, 12 best) → queries according to each policy
    → structural entropy and size of the version space after each query (no refitting)."""
    rows = []
    for k, inst in enumerate(balanced_instances(HELD_OUT + DEPTH_FAMILIES, seed + 3100, 1)):
        g0 = torch.Generator().manual_seed(seed * 77 + k); X, y = noisy_task(inst, N, g0, noise); Xt, yt, Xv, yv = split(X, y)
        s = BoundaryProgramSearch(max_depth=2, seed=seed).run(Xt, yt, Xv, yv); cands = [c[0] for c in s.candidates[:12]]
        for qp in ("structure_ig", "parameter_ig", "predictive_entropy", "disagreement", "random"):
            post = StructuralPosterior(cands, max(noise, 0.01)); post.update(X, y); g = torch.Generator().manual_seed(seed * 77 + k + 1)
            pool = full_region(500, g); Xa, ya = X.clone(), y.clone(); ent, vs = [post.entropy()], [len(version_space(post, Xa, ya))]
            for q in range(n_q):
                sc = {"structure_ig": post.ig_structure, "parameter_ig": post.ig_parameters, "predictive_entropy": post.predictive_entropy,
                      "disagreement": post.disagreement}.get(qp)
                i = int(sc(pool).argmax()) if sc else int(torch.randint(0, len(pool), (1,), generator=g))
                xq = pool[i:i + 1]; yq = inst.label(xq)
                if float(torch.rand(1, generator=g)) < noise: yq = 1 - yq
                post.update(xq, yq); Xa, ya = torch.cat([Xa, xq]), torch.cat([ya, yq]); ent.append(post.entropy()); vs.append(len(version_space(post, Xa, ya)))
            m = metrics(lambda Q, p=post.map_program(): predict(p, Q), inst, torch.Generator().manual_seed(3))
            rows.append({"family": inst.family, "policy": qp, "entropy": ent, "version_space": vs, **m, "candidates": len(cands), "classes": len(post.classes)})
    return rows


# ================================================================ H3 / H7: functional families, orbits, reuse

def discovered_programs(families, seed, per, policy, noise=0.05):
    out = []
    for k, inst in enumerate(balanced_instances(families, seed, per)):
        g = torch.Generator().manual_seed(seed * 31 + k); X, y = noisy_task(inst, N, g, noise)
        asd = ActiveStructuralDiscovery(policy, budget=BUDGET, eps=noise, seed=seed); asd.eps_world = noise
        r = asd.run(X, y, inst.label, full_region, torch.Generator().manual_seed(seed * 31 + k + 1))
        ok = acc(lambda Q: predict(r.program, Q), inst, full_region(2000, g)) >= 0.97
        if ok and r.adequate: out.append((inst.family, r.program))
    return out


def family_job(seed, noise=0.05):
    policy = train_policy(train_tasks(seed + 700, noise), eps=noise, seed=seed)
    progs = discovered_programs(HELD_OUT + UNREPRESENTABLE[:2], seed + 3200, 3, policy, noise)
    res = {}
    for mode in ("ast", "functional_raw", "functional_canonical"):
        P = [p for _, p in progs]
        fams = induce_ast_families(P) if mode == "ast" else induce_orbit_families(P, canonical=(mode == "functional_canonical"))
        hits, false = set(), 0
        for f in fams:
            srcs = Counter(fam for fam, p in progs if any(p is r for r in f.representations)); src = srcs.most_common(1)[0][0]
            ok = 0
            for j, inst in enumerate(balanced_instances([src], seed + 3300 + len(src), 2)):
                g = torch.Generator().manual_seed(seed + j); X, y = noisy_task(inst, N, g, noise)
                best = max((fit(r.skeleton(), X, y, steps=100, n_init=2)[0] for r in f.representations[:3]), key=lambda p: float((predict(p, X) == y).float().mean()))
                ok += acc(lambda Q: predict(best, Q), inst, full_region(2000, g)) >= 0.95
            okf = ok >= 1 and len(srcs) == 1
            f.update(okf, "validation")
            if okf: hits.add(src)
            else: false += 1
        res[mode] = {"promoted": len(fams), "recovered": sorted(hits), "recall": len(hits & set(HELD_OUT)) / len(HELD_OUT), "false_promotions": false,
                     "representations_per_family": round(sum(len({str(r.skeleton()) for r in f.representations}) for f in fams) / max(1, len(fams)), 2),
                     "compression": round(len(P) / max(1, len(fams)), 2)}
    # H7: reuse on TRANSFORMED representations
    lib_progs = discovered_programs(TRAIN + VALIDATION, seed + 3400, 2, policy, noise)
    lib = {"canonical": induce_orbit_families([p for _, p in lib_progs], min_support=1, canonical=True),
           "raw": induce_orbit_families([p for _, p in lib_progs], min_support=1, canonical=False)}
    rng = random.Random(seed); reuse = defaultdict(list)
    for k, inst in enumerate(balanced_instances(TRAIN + VALIDATION, seed + 3500, 1)):
        ti = transformed(inst, rng); g = torch.Generator().manual_seed(seed + 90 + k); X, y = noisy_task(ti, N, g, noise)
        for mode, fams in lib.items():
            p, ev = reuse_family(fams, X, y, canonical=(mode == "canonical"))
            reuse[mode].append({"recovery": p is not None and acc(lambda Q: predict(p, Q), ti, full_region(2000, g)) >= 0.97, "evaluations": ev})
    res["transformed_reuse"] = {m: {"recovery": sum(r["recovery"] for r in v) / len(v), "evaluations": sum(r["evaluations"] for r in v) / len(v)} for m, v in reuse.items()}
    res["n_programs"] = len(progs)
    return res


# ================================================================ H4 / Y: noise-aware truth anchors

def candidate_rules(inst, X, y, noise, seed):
    """CORRECT rules (full accuracy ≥ 0.97) and WRONG but self-consistent rules (< 0.93): sub-region,
    small sample, biased labels."""
    F = full_region(3000, torch.Generator().manual_seed(seed)); out = []
    def search(Xs, ys):
        Xt, yt, Xv, yv = split(Xs, ys); return BoundaryProgramSearch(max_depth=2, seed=seed).run(Xt, yt, Xv, yv)
    s = search(X, y)
    for kind, r in (("full_data", s), ("restricted_region", search(X[X[:, 0] < -1.2], y[X[:, 0] < -1.2]) if int((X[:, 0] < -1.2).sum()) > 30 else None),
                    ("tiny_sample", search(X[:14], y[:14])), ("biased_labels", search(X, torch.where(X[:, 1] > 0.8, 1 - y, y)))):
        if r is None or r.best is None: continue
        a = acc(lambda Q: predict(r.best, Q), inst, F)
        if a >= 0.97 or a < 0.93: out.append((kind, r.best, a < 0.93, [c[0] for c in r.candidates[1:6]]))
    return out


def anchor_job(seed, noise=0.08):
    rows = []
    for k, inst in enumerate(balanced_instances(TRAIN + VALIDATION + HELD_OUT + HARD_NEGATIVES, seed + 3600, 1)):
        g = torch.Generator().manual_seed(seed * 13 + k); X, y = noisy_task(inst, N, g, noise)
        for kind, rule, wrong, alts in candidate_rules(inst, X, y, noise, seed):
            r = sequential_truth_test(rule, alts, inst.label, full_region, torch.Generator().manual_seed(seed + k), noise, budget=BUDGET)
            old = collect_counterexamples(rule, inst.label, torch.Generator().manual_seed(seed + k), BUDGET, alts, full_region, "active", support=X, noise=noise)
            rows.append({"family": inst.family, "kind": kind, "wrong": wrong, "p_wrong": r["p_wrong"], "decision": r["decision"], "queries": r["queries"],
                         "repeats": r["repeats"], "old_refuted": old.refutes(lambda Q: predict(rule, Q))})
    return rows


# ================================================================ H5: stability under noise

def stability_job(seed):
    rows = []
    for noise in (0.08, 0.12):
        for k, inst in enumerate(balanced_instances(HELD_OUT + VALIDATION + UNREPRESENTABLE[:2], seed + 3700 + int(noise * 100), 1)):
            g = torch.Generator().manual_seed(seed * 17 + k); X, y = noisy_task(inst, N, g, noise)
            Xt, yt, Xv, yv = split(X, y); s = BoundaryProgramSearch(max_depth=2, seed=seed).run(Xt, yt, Xv, yv)
            if s.best is None: continue
            correct = acc(lambda Q: predict(s.best, Q), inst, full_region(3000, g)) >= 0.97
            new = noise_aware_stability(s.best, X, y, noise)
            old = stable_search(X, y, seed=seed).adequate
            rows.append({"noise": noise, "family": inst.family, "representable": inst.representable, "correct": correct, "new_accept": new["accept"], "old_accept": old,
                         **{k2: new[k2] for k2 in ("prediction_stability", "functional_class_stability", "parameter_stability", "ast_stability")}})
    return rows


# ================================================================ H9: gated residual

def residual_job(seed, noise=0.05):
    rows = []
    for k, inst in enumerate(balanced_instances(PARTIAL, seed + 3800, 3)):
        g = torch.Generator().manual_seed(seed * 19 + k); X, y = noisy_task(inst, 200, g, noise)
        Xt, yt, Xv, yv = split(X, y); s = BoundaryProgramSearch(max_depth=2, seed=seed).run(Xt, yt, Xv, yv); p = s.best
        f, rp = train_residual(p, X, y); gp, grp, gate = gated_residual(p, X, y)
        I, O = in_region(2000, g), out_region(2000, g)
        A = lambda fn, Q: float((fn(Q) == inst.label(Q)).float().mean())
        rows.append({"program": (A(lambda Q: predict(p, Q), I), A(lambda Q: predict(p, Q), O), p.n_params()), "global": (A(f, I), A(f, O), p.n_params() + rp),
                     "gated": (A(gp, I), A(gp, O), p.n_params() + grp), "gate_fraction_in": float(gate(I).float().mean()), "gate_fraction_out": float(gate(O).float().mean())})
    return rows


# ================================================================ H10: strict capacity budget

BUDGET_FLOATS = 400


def capacity_job(seed, noise=0.05):
    rng = random.Random(seed); fams = list(HELD_OUT + DEPTH_FAMILIES[:2] + PARTIAL + UNREPRESENTABLE[:3] + VALIDATION)
    insts = balanced_instances(fams, seed + 3900, 1); rng.shuffle(insts); insts = insts[:12]; data = []
    for k, inst in enumerate(insts):
        g = torch.Generator().manual_seed(seed * 23 + k); X, y = noisy_task(inst, 200, g, noise); data.append((inst, X, y, g))
    out = {}
    # (a) experts only
    used = 0; kept = []; rejected = 0
    for inst, X, y, g in data:
        if used + 82 > BUDGET_FLOATS: rejected += 1; continue
        e = train_mlp(X[:150], y[:150], h=16, steps=600, seed=seed); used += 82; kept.append((inst, lambda Q, e=e: mlp_logits(e, Q).argmax(1), 82))
    out["experts_only"] = _cap_summary(kept, rejected, used, {})
    # (b) full system: accepted program (stability + OOD verified by anchors) > gated residual > head on a shared base; pruning by value
    used = 0; kept = []; rejected = 0; ops = Counter(); base_paid = False; H = 16; W = None
    for t, (inst, X, y, g) in enumerate(data):
        Xh, yh = X[150:], y[150:]; X, y = X[:150], y[:150]                          # validation specific to the skill (value)
        Xt, yt, Xv, yv = split(X, y); s = BoundaryProgramSearch(max_depth=2, seed=seed).run(Xt, yt, Xv, yv); p = s.best
        cand = None
        if p is not None:
            st = noise_aware_stability(p, X, y, noise)
            truth = sequential_truth_test(p, [c[0] for c in s.candidates[1:6]], inst.label, full_region, g, noise, budget=8)
            if st["accept"] and truth["decision"] != "reject": cand = ("program", p.n_params(), lambda Q, p=p: predict(p, Q)); ops["crystallized_program"] += 1
            elif s.val_acc >= 0.8:
                gp, grp, _ = gated_residual(p, X, y); cand = ("gated_residual", p.n_params() + grp, gp); ops["gated_residual"] += 1
        if cand is None:
            if W is None:
                gg = torch.Generator().manual_seed(seed); W = {"w1": torch.randn(H, 2, generator=gg) * 1.2, "b1": torch.randn(H, generator=gg) * 0.8}
            feat = lambda Q: torch.tanh(torch.nan_to_num(Q) @ W["w1"].T + W["b1"])
            hd = {"w": torch.zeros(2, H, requires_grad=True), "b": torch.zeros(2, requires_grad=True)}; o = torch.optim.Adam(hd.values(), lr=0.05)
            for _ in range(500): o.zero_grad(); torch.nn.functional.cross_entropy(feat(X) @ hd["w"].T + hd["b"], y).backward(); o.step()
            hd = {k2: v.detach() for k2, v in hd.items()}; cost = 2 * H + 2 + (0 if base_paid else 3 * H); base_paid = True
            cand = ("shared_head", cost, lambda Q, hd=hd: (feat(Q) @ hd["w"].T + hd["b"]).argmax(1)); ops["shared_head"] += 1
        val = float((cand[2](Xh) == yh).float().mean())
        if val < 0.85: rejected += 1; ops["rejected_low_value"] += 1; continue                  # no acquisition of a useless skill
        while used + cand[1] > BUDGET_FLOATS and kept:
            low = min(range(len(kept)), key=lambda i: kept[i][3])
            if kept[low][3] < val - 0.05: used -= kept[low][2]; kept.pop(low); ops["pruned"] += 1     # replace a capacity of LOWER value
            else: break
        if used + cand[1] > BUDGET_FLOATS: rejected += 1; ops["rejected_budget"] += 1; continue
        kept.append((inst, cand[2], cand[1], val)); used += cand[1]
    out["full"] = _cap_summary([(i, f, c) for i, f, c, *_ in kept], rejected, used, ops)
    out["budget"] = BUDGET_FLOATS
    return out


def _cap_summary(kept, rejected, used, ops):
    g = torch.Generator().manual_seed(5); F = in_region(1500, g); accs = [acc(f, i, F) for i, f, c in kept]
    return {"retained_ok": sum(a >= 0.85 for a in accs), "retained": len(kept), "rejected": rejected, "used": used, "violated": used > BUDGET_FLOATS,
            "mean_acc": round(sum(accs) / max(1, len(accs)), 4), "operations": dict(ops)}


# ================================================================ identifiability, adversarial confusion

def identifiability_job(seed, noise=0.05):
    rows = []
    for k, inst in enumerate(balanced_instances(HARD_NEGATIVES, seed + 4000, 2)):
        g = torch.Generator().manual_seed(seed * 29 + k); X, y = noisy_task(inst, N, g, noise)
        Xt, yt, Xv, yv = split(X, y); s = BoundaryProgramSearch(max_depth=2, seed=seed).run(Xt, yt, Xv, yv)
        post = StructuralPosterior([c[0] for c in s.candidates[:8]], noise); post.update(X, y)
        obs_pool, full_pool = in_region(400, g), full_region(400, g)
        a, b = identifiability(post, obs_pool), identifiability(post, full_pool)
        mds = minimal_distinguishing_set(post, full_pool)
        rows.append({"family": inst.family, "identifiable_observed_only": a["identifiable"], "identifiable_full": b["identifiable"],
                     "plausible_classes": b["plausible_classes"], "minimal_set": len(mds)})
    return rows


# ================================================================ cross-domain: active experiments

def cross_domain_job(seed, tasks=6):
    from core.program_induction_core import DOMAINS, domain_inputs, enumerate_programs, random_target, run
    rng = random.Random(seed); out = {}
    for name, mk in DOMAINS.items():
        prims, comm, eq = mk(); progs = enumerate_programs(prims, 3, 6, comm); res = {"active": [], "random": []}
        for _ in range(tasks):
            tgt = random_target(name, rng); test = domain_inputs(name, rng, 40); key = lambda p: tuple(str(_safe(run, p, prims, x)) for x in test)
            tk = key(tgt); seed_x = domain_inputs(name, rng, 2)
            for pol in ("active", "random"):
                V = [p for p in progs if all(_safe(run, p, prims, x) == _safe(run, tgt, prims, x) for x in seed_x)]; q = 0; pool = domain_inputs(name, random.Random(seed + q), 60)
                while len({key(p) for p in V}) > 1 and q < 12:
                    if pol == "active":
                        x = max(pool, key=lambda x: _entropy(Counter(str(_safe(run, p, prims, x)) for p in V)))
                    else: x = rng.choice(pool)
                    yv = _safe(run, tgt, prims, x); V = [p for p in V if _safe(run, p, prims, x) == yv]; q += 1
                res[pol].append({"queries": q, "identified": len({key(p) for p in V}) == 1 and key(V[0]) == tk})
        out[name] = {pol: {"queries": sum(r["queries"] for r in v) / len(v), "identified": sum(r["identified"] for r in v) / len(v)} for pol, v in res.items()}
    return out


def _safe(run, p, prims, x):
    try: return run(p, prims, x)
    except Exception: return "ERR"


def _entropy(c):
    n = sum(c.values()); return -sum(v / n * math.log(v / n) for v in c.values())


# ================================================================ noise sweep

def noise_sweep_job(seed):
    rows = []; policy = train_policy(train_tasks(seed + 800, 0.05), eps=0.05, seed=seed)
    for noise in NOISE_LEVELS:
        for k, inst in enumerate(balanced_instances(HELD_OUT[:2] + DEPTH_FAMILIES[:2], seed + 4100, 1)):
            g = torch.Generator().manual_seed(seed * 37 + k); X, y = noisy_task(inst, N, g, noise)
            for v in ("passive", "full"):
                prog, info = run_variant(v, inst, X, y, noise, policy, seed * 37 + k)
                m = metrics(lambda Q, p=prog: predict(p, Q), inst, g) if prog is not None else {"recovery": False, "extrapolation": 0}
                rows.append({"noise": noise, "family": inst.family, "variant": v, "recovery": m["recovery"], "extrapolation": m["extrapolation"], "queries": info["queries"]})
            if prog is not None:
                cands = candidate_rules(inst, X, y, noise, seed)
                for kind, rule, wrong, alts in cands[:2]:
                    r = sequential_truth_test(rule, alts, inst.label, full_region, torch.Generator().manual_seed(seed + k), max(noise, 0.01), budget=BUDGET, noise=noise)
                    rows.append({"noise": noise, "family": inst.family, "variant": "anchor", "wrong": wrong, "p_wrong": r["p_wrong"], "decision": r["decision"]})
    return rows

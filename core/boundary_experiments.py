"""Experiments of the "Boundary Program Induction" milestone (one seed = one call; parallelised per seed)."""
from __future__ import annotations

import copy
import math
import random
import time
from collections import Counter, defaultdict

import torch

from core.boundary_methods import (METHODS, STABILITY_MIN, active_refine, evaluate_all, knn_predict, mlp_logits, mlp_params, split,
                                   stable_search, train_mlp, train_residual)
from core.boundary_program import (PRIMITIVES, BoundaryFamilyHypothesis, BoundaryProgramSearch, ProgramPosterior, anti_unify_programs, canonical,
                                   complexity, explain, fit, induce_families, predict, program_hash, render, structure_hash)
from core.boundary_world import (HELD_OUT, PARTIAL, REPRESENTABLE, SEALED, TRAIN, UNREPRESENTABLE, VALIDATION, full_region, in_region,
                                 instances, make_instance, out_region, sample_task)
from core.generative_certificate import ContinuousGenerator
from core.truth_anchor import BoundaryGenerativeCertificate, collect_counterexamples


def task(inst, g, n=200): return sample_task(inst, n, g)


# ================================================================ families: anti-unification and life cycle (H3, AT, AU)

def induce_family_library(seed, per_family=3):
    """Successful programs (functional recovery) on TRAINING + VALIDATION families → anti-unification
    by structure → families (macros) → validation on NEW instances → life cycle."""
    g = torch.Generator().manual_seed(seed); progs = []
    for inst in instances(TRAIN + VALIDATION, seed, per_family):
        X, y = task(inst, g); s = stable_search(X, y, seed=seed)
        F = full_region(2000, g)                                          # SAME points for prediction and truth
        if s.best is not None and s.adequate and float((predict(s.best, F) == inst.label(F)).float().mean()) >= 0.95:
            progs.append(s.best)
    fams = induce_families(progs, min_support=2)
    for f in fams:                                                     # validation on new instances (train/val families)
        for inst in instances(TRAIN + VALIDATION, seed + 77, 1):
            X, y = task(inst, g); Xt, yt, Xv, yv = split(X, y)
            prog, _ = fit(f.macro.skeleton, Xt, yt); acc = float((predict(prog, Xv) == yv).float().mean())
            if acc >= 0.93: f.support.append(inst.family)
        f.status = "STABLE" if len(f.support) >= 2 else "SUPPORTED" if f.support else "REFUTED"
    return fams, len(progs)


def family_reuse_experiment(seed, fams):
    """H3: new instances → (a) family macros first, (b) full search. Accuracy and cost."""
    g = torch.Generator().manual_seed(seed + 5); macros = [f.macro.skeleton for f in fams if f.status in ("STABLE", "SUPPORTED")]
    rows = []
    for inst in instances(TRAIN + VALIDATION, seed + 900, 1):
        X, y = task(inst, g); Xt, yt, Xv, yv = split(X, y); F = full_region(2000, g)
        a = BoundaryProgramSearch(macros=macros, seed=seed).run(Xt, yt, Xv, yv); b = BoundaryProgramSearch(seed=seed).run(Xt, yt, Xv, yv)
        acc = lambda p: float((predict(p, F) == inst.label(F)).float().mean()) if p is not None else 0.0
        rows.append({"family": inst.family, "macro_acc": acc(a.best), "search_acc": acc(b.best), "macro_evals": a.stats["evaluated"], "search_evals": b.stats["evaluated"],
                     "macro_seconds": a.stats["seconds"], "search_seconds": b.stats["seconds"]})
    return rows


def discovery_recall(seed, per_family=3):
    """AT/AU: on HELD-OUT families (never seen), successful programs → promoted families (anti-unification);
    recall = held-out families for which a promoted family recovers a NEW instance; false promotions =
    promoted families that fail on new instances of the original family."""
    g = torch.Generator().manual_seed(seed + 11); progs_by_family = defaultdict(list); all_progs = []
    for inst in instances(HELD_OUT + UNREPRESENTABLE, seed + 300, per_family):
        X, y = task(inst, g); s = stable_search(X, y, seed=seed)
        if s.best is not None and s.adequate: progs_by_family[inst.family].append(s.best); all_progs.append((inst.family, s.best))
    promoted = induce_families([p for _, p in all_progs], min_support=2)
    hits, false = set(), 0
    for f in promoted:
        srcs = Counter(fam for fam, p in all_progs if structure_hash(p) == structure_hash(f.macro.skeleton.with_params([0.0] * f.macro.skeleton.n_params())))
        src = srcs.most_common(1)[0][0] if srcs else None
        ok = 0
        for inst in instances([src], seed + 555, 2) if src else []:
            X, y = task(inst, g); prog, _ = fit(f.macro.skeleton, X, y); F = full_region(2000, g)
            ok += float((predict(prog, F) == inst.label(F)).float().mean()) >= 0.93
        if ok >= 1: hits.add(src)
        else: false += 1
    representable_held = [f for f in HELD_OUT if f != "capsule"]
    return {"promoted": len(promoted), "recovered_families": sorted(hits), "recall": round(len(hits & set(representable_held)) / len(representable_held), 3),
            "false_promotions": false, "from_unrepresentable": sum(1 for fam, _ in all_progs if fam in UNREPRESENTABLE)}


# ================================================================ H4: active experiments

def active_identification(seed, budget=30):
    """Number of queries before identification (MAP functionally correct AND P ≥ 0.95); policies IG / disagreement / random."""
    g0 = torch.Generator().manual_seed(seed + 21); rows = []
    for inst in instances(VALIDATION + HELD_OUT, seed + 400, 1):
        X, y = task(inst, g0, n=80); s = stable_search(X, y, seed=seed)
        if not s.candidates: continue
        F = full_region(2000, g0); truth = inst.label(F)
        for pol in ("ig", "disagreement", "random"):
            g = torch.Generator().manual_seed(seed + 99); found = None
            cands = [c[0] for c in s.candidates][:8]; Xa, ya = X.clone(), y.clone(); post = ProgramPosterior(cands); post.update(Xa, ya)
            pool = full_region(600, g)
            for q in range(1, budget + 1):
                sc = post.info_gain(pool) if pol == "ig" else post.disagreement(pool) if pol == "disagreement" else torch.rand(len(pool), generator=g).double()
                i = int(sc.argmax()); xq = pool[i:i + 1]; yq = inst.label(xq)
                if float(torch.rand(1, generator=g)) < 0.03: yq = 1 - yq
                Xa, ya = torch.cat([Xa, xq]), torch.cat([ya, yq])
                if q % 4 == 0:
                    post = ProgramPosterior([fit(c.skeleton(), Xa, ya, steps=100, n_init=1, seed=q)[0] for c in post.programs] + post.programs); post.update(Xa, ya)
                else: post.update(xq, yq)
                m, p = post.map()
                if p >= 0.95 and float((predict(m, F) == truth).float().mean()) >= 0.97: found = q; break
            rows.append({"family": inst.family, "policy": pol, "queries": found if found is not None else budget + 1, "identified": found is not None})
    return rows


# ================================================================ H5: truth anchors

def truth_anchor_experiment(seed, budget=10):
    """WRONG but self-consistent references (fitted on a sub-region, or on a small sample, or on
    biased labels) vs CORRECT references; active vs random anchors (same budget)."""
    g = torch.Generator().manual_seed(seed + 31); rows = []
    for inst in instances(VALIDATION + HELD_OUT[:2], seed + 600, 1):
        X, y = task(inst, g, n=240); F = full_region(3000, g)
        good = stable_search(X, y, seed=seed); refs = []
        if good.best is not None and float((predict(good.best, F) == inst.label(F)).float().mean()) >= 0.95: refs.append(("correct", good.best, good, X))
        sub = X[:, 0] < -1.2                                                  # sub-region: reference wrong elsewhere
        if int(sub.sum()) > 30:
            s2 = stable_search(X[sub], y[sub], seed=seed); refs.append(("restricted_region", s2.best, s2, X[sub]))
        s3 = stable_search(X[:16], y[:16], seed=seed); refs.append(("tiny_sample", s3.best, s3, X[:16]))
        yb = torch.where(X[:, 1] > 0.8, 1 - y, y); s4 = stable_search(X, yb, seed=seed); refs.append(("biased_labels", s4.best, s4, X))
        for kind, prog, s, supp in refs:
            if prog is None: continue
            truly_wrong = float((predict(prog, F) == inst.label(F)).float().mean()) < 0.93
            gen = ContinuousGenerator(k=2).fit(supp); cert = BoundaryGenerativeCertificate(prog, gen)
            self_ok = cert.check(lambda Q: predict(prog, Q), g)["reference_consistency"] >= 0.95
            alts = [c[0] for c in s.candidates[1:6]]
            for pol in ("active", "random"):
                cert.anchor = collect_counterexamples(prog, inst.label, g, budget, alts, full_region, pol, support=supp)
                rows.append({"kind": kind, "wrong": truly_wrong, "self_consistent": self_ok, "policy": pol,
                             "refuted": cert.anchor.refutes(lambda Q: predict(prog, Q))})
    return rows


# ================================================================ H6 / H7: crystallisation, residual; shared base; strict budget

def train_expert(X, y, h=16, steps=800, seed=0): return train_mlp(X, y, h=h, steps=steps, seed=seed)


def crystallization_experiment(seed):
    """Neural expert → behaviour on GENERATED samples → induction → verification (agreement with the expert AND
    anchors = observations) → replacement if safe; otherwise program + residual if partial; otherwise expert kept."""
    g = torch.Generator().manual_seed(seed + 41); rows = []
    for inst in instances(VALIDATION + HELD_OUT + PARTIAL + UNREPRESENTABLE[:2], seed + 700, 1):
        X, y = task(inst, g); e = train_expert(X, y, seed=seed); ep = lambda Q: mlp_logits(e, Q).argmax(1)
        gen = ContinuousGenerator(k=2).fit(X); Z = torch.cat([gen.sample(600, g), X]); yz = ep(Z)
        t0 = time.time(); s = stable_search(Z, yz, seed=seed); tsearch = time.time() - t0
        Fi = in_region(2000, g); truth = inst.label(Fi); acc_e = float((ep(Fi) == truth).float().mean())
        row = {"family": inst.family, "representable": inst.representable, "expert_acc": acc_e, "expert_params": mlp_params(e), "search_seconds": round(tsearch, 2)}
        if s.best is not None:
            Zf = gen.sample(1000, g); agree = float((predict(s.best, Zf) == ep(Zf)).float().mean()); obs_ok = float((predict(s.best, X) == y).float().mean())
            row.update(agree=agree, program=render(s.best))
            if s.adequate and agree >= 0.97 and obs_ok >= float((ep(X) == y).float().mean()) - 0.02:
                row.update(decision="replace", acc=float((predict(s.best, Fi) == truth).float().mean()), params=s.best.n_params())
            elif agree >= 0.8:
                f, rp = train_residual(s.best, X, y); row.update(decision="program+residual", acc=float((f(Fi) == truth).float().mean()), params=s.best.n_params() + rp)
            else: row.update(decision="keep_expert", acc=acc_e, params=mlp_params(e))
        else: row.update(decision="keep_expert", acc=acc_e, params=mlp_params(e))
        t0 = time.time(); ep(Fi); row["expert_infer_s"] = time.time() - t0
        if row["decision"] != "keep_expert" and s.best is not None:
            t0 = time.time(); predict(s.best, Fi); row["program_infer_s"] = time.time() - t0
        rows.append(row)
    return rows


def residual_experiment(seed, n=3):
    """H7: PARTIALLY symbolisable functions: expert alone vs program alone vs program + residual."""
    g = torch.Generator().manual_seed(seed + 51); rows = []
    for inst in instances(PARTIAL, seed + 800, n):
        X, y = task(inst, g); Fi, Fo = in_region(2000, g), out_region(2000, g)
        e = train_expert(X, y, seed=seed); s = stable_search(X, y, seed=seed); f, rp = train_residual(s.best, X, y)
        A = lambda fn, F: float((fn(F) == inst.label(F)).float().mean())
        rows.append({"expert": (A(lambda Q: mlp_logits(e, Q).argmax(1), Fi), A(lambda Q: mlp_logits(e, Q).argmax(1), Fo), mlp_params(e)),
                     "program": (A(lambda Q: predict(s.best, Q), Fi), A(lambda Q: predict(s.best, Q), Fo), s.best.n_params()),
                     "program_residual": (A(f, Fi), A(f, Fo), s.best.n_params() + rp)})
    return rows


def shared_basis_experiment(seed, n_tasks=6, h=24):
    """Remaining neural skills (not representable): separate experts (h=16) vs SHARED base (h=24, learned
    on the first tasks then frozen) + linear heads; compression, interference (base refitted), regeneration
    of a head from its generator + reference labels."""
    g = torch.Generator().manual_seed(seed + 61); insts = instances(UNREPRESENTABLE + PARTIAL, seed + 850, 2)[:n_tasks]
    data = [task(i, g) for i in insts]; Fi = in_region(2000, g)
    sep = [train_expert(X, y, seed=seed + j) for j, (X, y) in enumerate(data)]
    sep_acc = [float((mlp_logits(p, Fi).argmax(1) == i.label(Fi)).float().mean()) for p, i in zip(sep, insts)]
    k0 = n_tasks // 2; gg = torch.Generator().manual_seed(seed)
    W = {"w1": (torch.randn(h, 2, generator=gg) * 0.8).requires_grad_(True), "b1": torch.zeros(h, requires_grad=True)}
    heads = [{"w": torch.zeros(2, h, requires_grad=True), "b": torch.zeros(2, requires_grad=True)} for _ in range(n_tasks)]
    feat = lambda W, X: torch.tanh(X @ W["w1"].T + W["b1"])
    opt = torch.optim.Adam(list(W.values()) + [t for hd in heads[:k0] for t in hd.values()], lr=0.03)
    for _ in range(800):
        opt.zero_grad(); loss = sum(torch.nn.functional.cross_entropy(feat(W, X) @ hd["w"].T + hd["b"], y) for (X, y), hd in zip(data[:k0], heads[:k0])); loss.backward(); opt.step()
    Wf = {k: v.detach() for k, v in W.items()}
    for j in range(k0, n_tasks):                                        # new tasks: head only (frozen base)
        X, y = data[j]; hd = heads[j]; o = torch.optim.Adam(hd.values(), lr=0.05)
        for _ in range(600): o.zero_grad(); torch.nn.functional.cross_entropy(feat(Wf, X) @ hd["w"].T + hd["b"], y).backward(); o.step()
    H = [{k: v.detach() for k, v in hd.items()} for hd in heads]
    sh_acc = [float(((feat(Wf, Fi) @ hd["w"].T + hd["b"]).argmax(1) == i.label(Fi)).float().mean()) for hd, i in zip(H, insts)]
    # interference: does refitting the base for the last task degrade the old ones?
    W2 = {k: v.clone().requires_grad_(True) for k, v in Wf.items()}; o = torch.optim.Adam(W2.values(), lr=0.03); X, y = data[-1]
    for _ in range(300): o.zero_grad(); torch.nn.functional.cross_entropy(feat(W2, X) @ H[-1]["w"].T + H[-1]["b"], y).backward(); o.step()
    W2 = {k: v.detach() for k, v in W2.items()}
    inter = [float(((feat(W2, Fi) @ hd["w"].T + hd["b"]).argmax(1) == i.label(Fi)).float().mean()) for hd, i in zip(H[:k0], insts[:k0])]
    # regeneration of a head from generator + reference (surrogate = certified output of the head)
    j = 0; X, y = data[j]; gen = ContinuousGenerator(k=2).fit(X); Z = gen.sample(1500, g); ref = (feat(Wf, Z) @ H[j]["w"].T + H[j]["b"]).argmax(1)
    hd = {"w": torch.zeros(2, h, requires_grad=True), "b": torch.zeros(2, requires_grad=True)}; o = torch.optim.Adam(hd.values(), lr=0.05)
    for _ in range(600): o.zero_grad(); torch.nn.functional.cross_entropy(feat(Wf, Z) @ hd["w"].T + hd["b"], ref).backward(); o.step()
    regen = float(((feat(Wf, Fi) @ hd["w"].detach().T + hd["b"].detach()).argmax(1) == insts[j].label(Fi)).float().mean())
    return {"separate_params": sum(mlp_params(p) for p in sep), "shared_params": h * 3 + n_tasks * (2 * h + 2),
            "separate_acc": round(sum(sep_acc) / n_tasks, 4), "shared_acc": round(sum(sh_acc) / n_tasks, 4),
            "old_acc_frozen": round(sum(sh_acc[:k0]) / k0, 4), "old_acc_after_basis_finetune": round(sum(inter) / k0, 4),
            "head_regenerated_acc": round(regen, 4), "head_original_acc": round(sh_acc[0], 4)}


def hard_budget_stream(seed, budget=260, mode="hard", n=18):
    """Mixed stream under budget (parameters + symbolic floats). Policy: program > program+residual >
    head on a shared base > expert; if exceeded: crystallise, then prune the skill of LOWEST
    VALUE (usage, dependencies, recency, regeneration cost) only if its value < that of the new one;
    otherwise ABSTAIN (hard mode). Never a silent violation: each kept skill must pass its
    anchored certificate."""
    rng = random.Random(seed); g = torch.Generator().manual_seed(seed + 71)
    fams = list(VALIDATION + HELD_OUT + PARTIAL + UNREPRESENTABLE); insts = instances(fams, seed + 950, 1); rng.shuffle(insts); insts = insts[:n]
    skills = []; log = Counter(); used = 0; violations = 0; t = 0
    for inst in insts:
        t += 1; X, y = task(inst, g); s = stable_search(X, y, seed=seed)
        if s.best is not None and s.adequate: cand = ("program", s.best.n_params(), lambda Q, p=s.best: predict(p, Q))
        elif s.best is not None and s.val_acc >= 0.8:
            f, rp = train_residual(s.best, X, y); cand = ("program+residual", s.best.n_params() + rp, f)
        else:
            e = train_expert(X, y, h=8, seed=seed); cand = ("expert", mlp_params(e), lambda Q, e=e: mlp_logits(e, Q).argmax(1))
        value_new = 1.0
        while used + cand[1] > budget and mode == "hard":
            if not skills: break
            for sk in skills: sk["value"] = sk["usage"] / (1 + t - sk["t"]) + 0.2 * sk["regen_cost"]
            low = min(skills, key=lambda sk: sk["value"])
            if low["value"] < value_new: skills.remove(low); used -= low["params"]; log["pruned"] += 1
            else: break
        if mode == "hard" and used + cand[1] > budget: log["abstained"] += 1; continue
        anchor_ok = float((cand[2](X) == y).float().mean()) >= 0.85
        if not anchor_ok: violations += 0; log["rejected_by_anchor"] += 1; continue
        skills.append({"inst": inst, "kind": cand[0], "params": cand[1], "pred": cand[2], "t": t, "usage": 1 + rng.random(), "regen_cost": cand[1] / 100.0, "value": 0})
        used += cand[1]; log[cand[0]] += 1
        for sk in skills: sk["usage"] += rng.random() * 0.2
    Fi = in_region(1500, g)
    acc = [float((sk["pred"](Fi) == sk["inst"].label(Fi)).float().mean()) for sk in skills]
    return {"mode": mode, "budget": budget, "used": used, "budget_violated": used > budget, "skills": len(skills), "mean_acc": round(sum(acc) / max(1, len(acc)), 4),
            "operations": dict(log), "invariant_violations": violations}


# ================================================================ novelty, overfitting, stability, confidence

def novelty_detection(seed, fams_lib):
    """Known (macro within range) / new parameterisation (macro out of range) / new composition / unsupported."""
    g = torch.Generator().manual_seed(seed + 81); macros = [f.macro for f in fams_lib if f.status in ("STABLE", "SUPPORTED")]; rows = []
    cases = [(i, "known") for i in instances(TRAIN + VALIDATION, seed + 1000, 1)] + [(i, "new_composition") for i in instances(HELD_OUT, seed + 1001, 1)] + \
            [(i, "unsupported") for i in instances(UNREPRESENTABLE, seed + 1002, 1)]
    for inst, truth in cases:
        X, y = task(inst, g); Xt, yt, Xv, yv = split(X, y); decided = None
        for m in macros:
            prog, _ = fit(m.skeleton, Xt, yt); acc = float((predict(prog, Xv) == yv).float().mean())
            if acc >= 0.93:
                v = torch.tensor(canonical(prog).param_vector()); lo, hi = torch.tensor(m.param_lo), torch.tensor(m.param_hi); span = (hi - lo).abs() + 0.5
                decided = "known" if bool(((v >= lo - span) & (v <= hi + span)).all()) else "new_parameterization"; break
        if decided is None:
            s = stable_search(X, y, seed=seed); decided = "new_composition" if s.adequate else "unsupported"
        rows.append({"truth": truth, "decided": decided})
    return rows


def symbolic_overfitting(seed, n_small=12):
    """Small sample → simple program that fits but extrapolates wrongly; detected by active anchors?"""
    g = torch.Generator().manual_seed(seed + 91); rows = []
    for inst in instances(VALIDATION + HELD_OUT, seed + 1100, 1):
        X, y = task(inst, g, n=n_small); s = BoundaryProgramSearch(seed=seed).run(X, y, X, y)
        if s.best is None: continue
        F = full_region(3000, g); wrong = float((predict(s.best, F) == inst.label(F)).float().mean()) < 0.93
        anc = collect_counterexamples(s.best, inst.label, g, 10, [c[0] for c in s.candidates[1:6]], full_region, "active", support=X)
        rows.append({"family": inst.family, "overfit": wrong, "fit_acc": s.val_acc, "detected": anc.refutes(lambda Q: predict(s.best, Q))})
    return rows


def program_stability(seed):
    """Slightly perturbed observations (position noise + 5 % of the labels): same structure? same function?"""
    g = torch.Generator().manual_seed(seed + 101); rows = []
    for inst in instances(VALIDATION + HELD_OUT, seed + 1200, 1):
        X, y = task(inst, g); a = stable_search(X, y, seed=seed)
        Xp = X + 0.05 * torch.randn(X.shape, generator=g); yp = torch.where(torch.rand(len(y), generator=g) < 0.05, 1 - y, y)
        b = stable_search(Xp, yp, seed=seed + 3)
        if a.best is None or b.best is None: continue
        F = full_region(2000, g)
        rows.append({"family": inst.family, "same_structure": structure_hash(a.best) == structure_hash(b.best),
                     "functional_agreement": float((predict(a.best, F) == predict(b.best, F)).float().mean())})
    return rows


def confidence_calibration(seed):
    """SEPARATE confidence: family (posterior mass of the MAP structure), parameters (bootstrap standard deviation),
    predictions (predictive probability) → ECE of the predictions and correlation family-confidence / structure-correctness."""
    g = torch.Generator().manual_seed(seed + 111); rows = []; conf_all, ok_all = [], []
    for inst in instances(VALIDATION + HELD_OUT, seed + 1300, 1):
        X, y = task(inst, g); s = stable_search(X, y, seed=seed)
        if not s.candidates: continue
        post = ProgramPosterior([c[0] for c in s.candidates]); post.update(X, y)
        mp, pw = post.map(); same = sum(float(w) for p, w in zip(post.programs, post.weights) if structure_hash(p) == structure_hash(mp))
        boots = []
        for b in range(3):
            idx = torch.randint(0, len(X), (len(X),), generator=g); boots.append(torch.tensor(fit(mp.skeleton(), X[idx], y[idx], steps=100, n_init=1, seed=b)[0].param_vector()))
        param_sd = float(torch.stack(boots).std(0).mean())
        F = full_region(1500, g); p1 = post.predictive(F); conf = torch.maximum(p1, 1 - p1); ok = ((p1 >= 0.5).long() == inst.label(F))
        conf_all.append(conf); ok_all.append(ok)
        rows.append({"family": inst.family, "family_confidence": round(same, 3), "param_sd": round(param_sd, 3),
                     "structure_correct": inst.oracle is not None and structure_hash(mp) == structure_hash(inst.oracle), "explanation": explain(mp)["rule"]})
    c = torch.cat(conf_all); o = torch.cat(ok_all).double(); e = 0.0
    for b in range(5):
        m = (c >= 0.5 + 0.1 * b) & (c < 0.6 + 0.1 * b + (1e-9 if b == 4 else 0))
        if int(m.sum()): e += float(m.double().mean()) * abs(float(c[m].mean()) - float(o[m].mean()))
    return {"rows": rows, "prediction_ece": round(e, 4)}


# ================================================================ primitives: ablation, economy, depth

def primitive_ablation(seed, per_family=1):
    g = torch.Generator().manual_seed(seed + 121); insts = instances(TRAIN + VALIDATION + HELD_OUT, seed + 1400, per_family); data = [(i, task(i, g)) for i in insts]
    def coverage(allowed, depth=3):
        hit = 0; ev = 0
        for inst, (X, y) in data:
            Xt, yt, Xv, yv = split(X, y); s = BoundaryProgramSearch(allowed=allowed, max_depth=depth, seed=seed).run(Xt, yt, Xv, yv); ev += s.stats["evaluated"]
            F = full_region(1500, g); hit += s.best is not None and float((predict(s.best, F) == inst.label(F)).float().mean()) >= 0.95
        return round(hit / len(data), 3), ev
    full = set(PRIMITIVES); res = {"all": coverage(full)}
    for drop in ("rotate", "angle", "norm", "and", "or", "translate", "interval", "xor"):
        res[f"without_{drop}"] = coverage(full - {drop})
    return res


def search_depth_study(seed):
    g = torch.Generator().manual_seed(seed + 131); insts = instances(VALIDATION + HELD_OUT, seed + 1500, 1); out = {}
    for d in (1, 2, 3, 4, 5):
        hit = ev = 0; t0 = time.time()
        for inst in insts:
            X, y = task(inst, g); Xt, yt, Xv, yv = split(X, y)
            s = BoundaryProgramSearch(max_depth=d, seed=seed).run(Xt, yt, Xv, yv, time_budget=60); ev += s.stats["evaluated"]
            F = full_region(1500, g); hit += s.best is not None and float((predict(s.best, F) == inst.label(F)).float().mean()) >= 0.95
        out[d] = {"coverage": round(hit / len(insts), 3), "evaluations": ev, "seconds": round(time.time() - t0, 1)}
    return out

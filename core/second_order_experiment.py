"""H2 — does the second order catch dangerous updates that the first order lets through?

A shared module serves skill A (protected by its generative certificate); it is adapted to B IN PLACE.
Candidate updates: steps for B projected out of the sensitive directions of A (J_A·Δθ ≈ 0, first order), at
several amplitudes. For each one: REAL violation (margins of A flipped on fresh probes > 2 %) vs
predictions (1st order; 2nd order by directional second derivative; ellipsoid δᵀQδ ≤ 1).
Adaptation compared: 1st-order projection (full step) / amplitude control at 2nd order / growth (new module).
"""
from __future__ import annotations

import random
import time

import torch

from core.continuous_world import D, Regime, random_rule
from core.continuum import new_expert, nparams, train
from core.curvature import actual_flips, ellipsoid, flat, mlp, null_projector, predicted_margins, safe_step, unflat
from core.generative_certificate import ContinuousGenerator, GenerativeInvariantCertificate, clean_inputs

SCALES = (0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)


def setup(seed):
    rng = random.Random(seed); g = torch.Generator().manual_seed(seed)
    muA = torch.tensor([rng.gauss(0, 0.5) for _ in range(D)]); muB = muA + torch.tensor([rng.gauss(0, 1.0) for _ in range(D)])
    A = Regime("A", random_rule(rng, "halfspace", muA), muA, 1.0, torch.zeros(8), "new")
    B = Regime("B", random_rule(rng, rng.choice(["frame", "sector"]), muB), muB, 1.0, torch.ones(8), "new")
    XA, _, yA = A.sample(512, g, noisy=False); XB, _, yB = B.sample(512, g, noisy=False)
    p = train(new_expert(seed), XA, yA, 400, seed=seed)
    gen = ContinuousGenerator().fit(XA)
    with torch.no_grad(): pred = lambda X: mlp(p, clean_inputs(X)).argmax(1)
    return A, B, XA, yA, XB, yB, p, gen, g


def run(seed=0, n_probe=96):
    A, B, XA, yA, XB, yB, p, gen, g = setup(seed)
    P = torch.cat([gen.sample(n_probe, g), gen.sample_boundary(n_probe // 2, g, lambda X: mlp(p, clean_inputs(X)).argmax(1))])
    with torch.no_grad(): yP = mlp(p, P).argmax(1)
    Pfresh = gen.sample(400, g)
    with torch.no_grad(): yF = mlp(p, Pfresh).argmax(1)
    project, V = null_projector(p, P, yP, energy=0.999)
    t0 = time.time(); q = train(p, XB, yB, 300, seed=seed, project=project); t_first = time.time() - t0
    base = flat(q) - flat(p)
    t0 = time.time(); E = ellipsoid(p, P, yP, tau=1.0, rank=16, k=4); t_ell = time.time() - t0
    rows = []
    for a in SCALES:
        d = a * base
        true_v = actual_flips(p, Pfresh, yF, d) > 0.02
        with torch.no_grad(): m0 = (2 * yP - 1) * (mlp(p, P)[:, 1] - mlp(p, P)[:, 0])
        p1 = float(((predicted_margins(p, P, yP, d, order=1) <= 0) & (m0 > 0)).float().mean()) > 0.02
        p2 = float(((predicted_margins(p, P, yP, d, order=2) <= 0) & (m0 > 0)).float().mean()) > 0.02
        rows.append({"scale": a, "true_violation": true_v, "first_order_flags": p1, "second_order_flags": p2, "ellipsoid_flags": not E.contains(d),
                     "J_delta_rel": float((V.T @ d).norm() / d.norm().clamp_min(1e-12)), "actual_flip_rate": round(actual_flips(p, Pfresh, yF, d), 4)})
    # adaptation: full first order / controlled second order / growth
    accA = lambda w: float((mlp(w, clean_inputs(A.test_set(seed + 5)[0])).argmax(1) == A.test_set(seed + 5)[2]).float().mean())
    accB = lambda w: float((mlp(w, clean_inputs(B.test_set(seed + 6)[0])).argmax(1) == B.test_set(seed + 6)[2]).float().mean())
    t0 = time.time(); alpha, _ = safe_step(p, P, yP, base, max_flip=0.0, order=2); t_second = time.time() - t0
    q2 = unflat(flat(p) + alpha * base, p)
    t0 = time.time(); grown = train(new_expert(seed + 1), XB, yB, 300, seed=seed); t_grow = time.time() - t0
    adapt = {"first_order": {"retention_A": accA(q), "new_B": accB(q), "params_added": 0, "seconds": round(t_first, 3)},
             "second_order": {"retention_A": accA(q2), "new_B": accB(q2), "params_added": 0, "alpha": alpha, "seconds": round(t_first + t_second, 3)},
             "structural_growth": {"retention_A": accA(p), "new_B": accB(grown), "params_added": nparams(grown), "seconds": round(t_grow, 3)},
             "reference_A": accA(p), "ellipsoid_seconds": round(t_ell, 3), "ellipsoid_floats": E.floats()}
    return rows, adapt


def summarize(results):
    tp = {k: 0 for k in ("first_order_flags", "second_order_flags", "ellipsoid_flags")}; fp = dict(tp); pos = neg = 0
    for rows, _ in results:
        for r in rows:
            pos += r["true_violation"]; neg += not r["true_violation"]
            for k in tp:
                tp[k] += r[k] and r["true_violation"]; fp[k] += r[k] and not r["true_violation"]
    rec = {k: round(tp[k] / max(1, pos), 3) for k in tp}; far = {k: round(fp[k] / max(1, neg), 3) for k in fp}
    ad = {m: {k: round(sum(a[m][k] for _, a in results) / len(results), 4) for k in ("retention_A", "new_B", "params_added", "seconds")}
          for m in ("first_order", "second_order", "structural_growth")}
    wins = sum(a["second_order"]["retention_A"] > a["first_order"]["retention_A"] for _, a in results)
    return {"violations": pos, "safe": neg, "recall": rec, "false_alarm": far, "adaptation": ad, "second_beats_first_retention_seeds": wins,
            "n_seeds": len(results), "reference_A": round(sum(a["reference_A"] for _, a in results) / len(results), 4),
            "ellipsoid_seconds": round(sum(a["ellipsoid_seconds"] for _, a in results) / len(results), 3)}

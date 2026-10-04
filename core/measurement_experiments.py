"""Expériences contrôlées de l'algèbre de mesure : Fourier, référentiels, macro/micro-états, compromis de capteur."""
from __future__ import annotations

import math
import random
from collections import Counter

import torch

from core.measurement import (
    WORLD_FRAME, BasisTransform, CompatibleStateSet, FrameError, FramedValue, LatentWorldState, MeasurementContext,
    MeasurementOperator, MeasurementPlanner, MeasurementTradeoff, NoiseModel, Observable, ReferenceFrame, change_frame,
    distance, information_gain, new_graph,
)

# ================================================================ Fourier : changement de base utile

CANDIDATE_FREQS = (3, 5, 8, 11, 14, 19, 23, 29)


def make_signal(rng, n=128, noise=0.6):
    k = rng.randint(1, 3); freqs = sorted(rng.sample(CANDIDATE_FREQS, k))
    amps = [rng.uniform(0.6, 1.5) for _ in freqs]
    t = torch.arange(n, dtype=torch.float64)
    x = sum(a * torch.sin(2 * math.pi * f * t / n + rng.uniform(0, 2 * math.pi)) for f, a in zip(freqs, amps))
    x = x + noise * torch.tensor([rng.gauss(0, 1) for _ in range(n)], dtype=torch.float64)
    dominant = freqs[max(range(len(freqs)), key=lambda i: amps[i])]
    return x, set(freqs), dominant


def raw_reasoning(x):
    """Raisonnement dans la base TEMPORELLE : fréquence estimée par passages à zéro (une seule fréquence)."""
    zc = int(((x[:-1] * x[1:]) < 0).sum())
    f = zc / 2.0
    best = min(CANDIDATE_FREQS, key=lambda c: abs(c - f))
    return {best}, best


def spectral_reasoning(x, bt: BasisTransform, k_noise=4.0):
    """Raisonnement dans la base FRÉQUENTIELLE : pics du spectre au-dessus du bruit (seuil robuste par médiane)."""
    X = bt.forward(x); mag = X.abs()[: len(x) // 2]
    thr = k_noise * float(mag.median())
    present = {f for f in CANDIDATE_FREQS if float(mag[f]) > thr}
    dominant = max(CANDIDATE_FREQS, key=lambda f: float(mag[f]))
    return present or {dominant}, dominant


def fourier_experiment(n_trials=200, seed=0, noise=0.6):
    rng = random.Random(seed); bt = BasisTransform(128)
    res = {m: Counter() for m in ("raw", "spectral")}
    inv = Counter()
    for _ in range(n_trials):
        x, truth, dom = make_signal(rng, noise=noise)
        ii = bt.invariants(x); inv["invertible"] += ii["invertible"]; inv["energy"] += ii["energy_preserved"]
        for name, (pred, pdom) in (("raw", raw_reasoning(x)), ("spectral", spectral_reasoning(x, bt))):
            c = res[name]; c["dominant"] += pdom == dom; c["exact_set"] += pred == truth
            c["tp"] += len(pred & truth); c["fp"] += len(pred - truth); c["fn"] += len(truth - pred)
    out = {}
    for name, c in res.items():
        p = c["tp"] / max(1, c["tp"] + c["fp"]); r = c["tp"] / max(1, c["tp"] + c["fn"])
        out[name] = {"dominant_accuracy": round(c["dominant"] / n_trials, 3), "exact_set": round(c["exact_set"] / n_trials, 3),
                     "f1": round(2 * p * r / max(1e-9, p + r), 3)}
    out["invariants"] = {"invertible": inv["invertible"] / n_trials, "parseval": inv["energy"] / n_trials}
    return out


# ================================================================ référentiels : transformer, invariants, comparaisons invalides

def frame_experiment(n_trials=200, seed=0):
    rng = random.Random(seed)
    c = Counter()
    for _ in range(n_trials):
        frames = {"A": ReferenceFrame("A", (rng.uniform(-5, 5), rng.uniform(-5, 5)), rng.uniform(-math.pi, math.pi)),
                  "B": ReferenceFrame("B", (rng.uniform(-5, 5), rng.uniform(-5, 5)), rng.uniform(-math.pi, math.pi))}
        P = torch.tensor([rng.uniform(-3, 3), rng.uniform(-3, 3)], dtype=torch.float64)
        Q = P + torch.tensor([rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)], dtype=torch.float64)
        truth_close = float(torch.linalg.norm(P - Q)) < 1.0
        pa = frames["A"].from_world(FramedValue(P, "world")); qb = frames["B"].from_world(FramedValue(Q, "world"))
        # système conscient des référentiels : transforme avant de comparer
        d_aware = distance(change_frame(pa, frames, "world"), change_frame(qb, frames, "world"))
        c["aware_correct"] += (d_aware < 1.0) == truth_close
        # système naïf : compare des coordonnées brutes de référentiels différents
        d_naive = float(torch.linalg.norm(torch.as_tensor(pa.value) - torch.as_tensor(qb.value)))
        c["naive_correct"] += (d_naive < 1.0) == truth_close
        try: distance(pa, qb)
        except FrameError: c["invalid_detected"] += 1
        # invariant : distance entre deux objets vue par UN même observateur = distance dans le monde
        qa = frames["A"].from_world(FramedValue(Q, "world"))
        c["distance_invariant"] += abs(distance(pa, qa) - float(torch.linalg.norm(P - Q))) < 1e-9
        c["coordinates_change"] += float(torch.linalg.norm(torch.as_tensor(pa.value) - P)) > 1e-6
        # orientation : un angle relatif entre deux directions est invariant, une direction absolue ne l'est pas
        u, v = torch.tensor([1.0, 0.0], dtype=torch.float64), torch.tensor([0.0, 1.0], dtype=torch.float64)
        ua = frames["A"].from_world(FramedValue(u, "world", "vector")).value; va = frames["A"].from_world(FramedValue(v, "world", "vector")).value
        ang = lambda a, b: math.atan2(float(a[0] * b[1] - a[1] * b[0]), float(a @ b))
        c["angle_invariant"] += abs(ang(ua, va) - ang(u, v)) < 1e-9
    return {k: round(v / n_trials, 3) for k, v in c.items()}


# ================================================================ macro / micro-états

def macrostate_set(n_particles=4, n_cells=6, graph=None):
    dom = {(f"p{i}", "cell"): list(range(n_cells)) for i in range(n_particles)}
    return CompatibleStateSet.product(dom, graph=graph)


def macro_left_count(states, n_cells=6):
    return (states < n_cells // 2).sum(1)


def macrostate_experiment(n_trials=40, seed=0, policies=("random", "fixed", "max_ig"), n_particles=4, n_cells=6):
    """Macro-observation (combien de particules à gauche) -> ambiguïté conservée ; sondes de cellule (occupée ?)
    choisies par politique ; mesure : sondes nécessaires pour isoler le micro-état, corrélation IG prédit / réalisé."""
    rng = random.Random(seed)
    base = macrostate_set(n_particles, n_cells)
    mult = base.multiplicity(lambda s: macro_left_count(s, n_cells))
    ent = {m: math.log(c) for m, c in mult.items()}
    out = {"multiplicity": mult, "entropy_by_macro": {m: round(e, 3) for m, e in ent.items()}}
    pairs = []
    for policy in policies:
        steps, kept_ambiguous = [], 0
        for trial in range(n_trials):
            truth = [rng.randrange(n_cells) for _ in range(n_particles)]
            left = sum(c < n_cells // 2 for c in truth)
            S = base.restrict(macro_left_count(base.states, n_cells) == left)
            if S.support_size() > 1: kept_ambiguous += 1
            planner = MeasurementPlanner(policy=policy, budget=1e9, confidence=1.01, min_ig=1e-6, seed=trial)
            n = 0
            while S.support_size() > 1 and n < 40:
                ops = [MeasurementOperator(Observable("cell", f"p{i}", "cell", tuple(range(n_cells))),
                                           MeasurementContext(sensor=f"probe{i}", noise=NoiseModel("confusion", 1.0), cost=0.0))
                       for i in range(n_particles)]
                # sonde « la particule i est-elle à gauche ? » (mesure grossière) ou cellule exacte (fine)
                choice = planner.choose(S, ops, None) if policy != "fixed" else planner.choose(S, ops[n % n_particles:] + ops[:n % n_particles], None)
                if choice is None: break
                h0 = S.entropy()
                pred = choice.information_gain
                var = (choice.operator.observable.entity, "cell")
                i = int(choice.operator.observable.entity[1:])
                S = S.restrict(S.column(var) == truth[i])
                pairs.append((pred, h0 - S.entropy())) if policy == "max_ig" else None
                n += 1
            steps.append(n)
        out[policy] = {"mean_probes": round(sum(steps) / len(steps), 3), "ambiguity_kept_after_macro": kept_ambiguous / n_trials}
    if pairs:
        p = torch.tensor(pairs, dtype=torch.float64)
        out["ig_vs_realized_corr"] = round(float(torch.corrcoef(p.T)[0, 1]), 3) if p[:, 0].std() > 0 else None
        out["ig_vs_realized_mean"] = (round(float(p[:, 0].mean()), 3), round(float(p[:, 1].mean()), 3))
    ms = sorted(mult.items())
    out["multiplicity_entropy_monotone"] = all(math.log(a[1]) <= math.log(b[1]) + 1e-12 for a, b in zip(sorted(ms, key=lambda t: t[1]), sorted(ms, key=lambda t: t[1])[1:]))
    return out


# ================================================================ compromis de capteur

def tradeoff_experiment(n_trials=500, seed=0):
    rng = random.Random(seed); sensor = MeasurementTradeoff(0.1, 0.1)
    goals = {"position": (1.0, 0.0), "velocity": (0.0, 1.0), "both": (1.0, 1.0)}
    out = {}
    for gname, w in goals.items():
        a_star = sensor.best_setting(w)
        errs = {"planner": 0.0, "fixed_0.5": 0.0, "wrong_goal": 0.0}
        settings = {"planner": a_star, "fixed_0.5": 0.5, "wrong_goal": sensor.best_setting(w[::-1]) if w[0] != w[1] else 0.5}
        for _ in range(n_trials):
            for k, a in settings.items():
                sp, sv = sensor.sigmas(a)
                ep, ev = rng.gauss(0, sp), rng.gauss(0, sv)
                errs[k] += w[0] * ep ** 2 + w[1] * ev ** 2
        out[gname] = {"setting": round(a_star, 3), **{k: round(v / n_trials, 5) for k, v in errs.items()}}
    return out

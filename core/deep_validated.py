"""Sous-jalon Validated Optimizer — module d'ÉVALUATION, dev uniquement, vrais atomes fournis.

1. Expérience contrôlée à deux bras : pipeline gelé (A) contre pipeline + garde-fou d'ajustement seul (B, critère
   pré-défini : vraisemblance profilée). Un bras C exploratoire utilise le critère de substitution de l'optimiseur.
2. Diagnostic de l'OPTIMISEUR lui-même (non modifié) : trajectoires, désaccord entre objectif continu et accord discret,
   expérience aux vrais paramètres, baseline sans optimisation, arrêt précoce, meilleur état rencontré, descente
   monotone, sensibilité numérique, comparaison systématique des deux ajusteurs.

La vérité n'est utilisée que pour MESURER (accord fonctionnel) ; aucune décision du système n'en dépend.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import torch

from core.boundary_program import BNode, predict
from core.deep_fit import DT, FitData, compile_skeleton, fit_batch, loglik_batch, predict_batch, profiled_loglik
from core.deep_generation import generate_candidates
from core.deep_search import DeepSearch
from core.deep_structure import algebraic_form, n_atoms, normalize, skel_key, structurally_equivalent

ARMS = {"A": None, "B": "profiled", "C": "surrogate"}            # A : pipeline gelé ; B : garde-fou pré-défini ; C : exploratoire
THRESHOLDS = (0.90, 0.95, 0.97, 0.99)


def _eval(task):
    from core.deep_world import full_region, obs_region
    F = full_region(4000, torch.Generator().manual_seed(11)); I = obs_region(2000, torch.Generator().manual_seed(11)); lF, lI = task.world.label(F, 1.0), task.world.label(I, 1.0)
    return (lambda p: float((predict(p, F) == lF).float().mean())), (lambda p: float((predict(p, I) == lI).float().mean()))


def arm_run(task, arm: str, seed=0) -> dict:
    """Un bras de l'expérience contrôlée : même pipeline, mêmes littéraux (vrais atomes), seul `validated` change."""
    from core.deep_reachability import leaves
    t0 = time.time(); tr = task.world.truth; tl = leaves(tr); data = FitData.build(task.X, task.y, 0.10); A, AI = _eval(task)
    out = DeepSearch("factorized", budget=150.0, seed=seed, max_depth=min(6, n_atoms(tr) + 1), fact_literals=tl, fact_residual=False, validated=ARMS[arm]).run(data); best = out.best().program; a = A(best)
    pred = predict_batch(best.skeleton(), torch.tensor([best.param_vector()], dtype=DT), data.X)[0].to(DT)
    return {"arm": arm, "accord": a, "in_region": AI(best), "likelihood": profiled_loglik(pred, data), "errors": float((pred != data.y).double().sum()), "thresholds": {str(t): a >= t for t in THRESHOLDS},
            "exact": skel_key(best) == skel_key(tr), "equivalent": structurally_equivalent(best, tr), "atoms": n_atoms(best), "evidence": out.best().evidence, "transactions": out.stats["transactions"], "commits": out.stats["commits"],
            "rollbacks": out.stats["rollbacks"], "fit_equivalents": out.stats["fit_equivalents"], "seconds": round(time.time() - t0, 2)}


# ================================================================ diagnostic de l'optimiseur (inchangé : on l'observe)

def _objective(fn, th, X, y, f, w, gamma, k):
    p = fn(th, X, lambda z: torch.sigmoid(k * z)).clamp(1e-6, 1 - 1e-6); p1 = (1 - gamma) * (f + (1 - 2 * f) * p) + gamma * 0.5
    return -(w * (y * torch.log(p1) + (1 - y) * torch.log(1 - p1))).sum()


def optimize_traced(skel: BNode, data: FitData, init, steps=100, lr=0.05, k_start=4.0, k_end=24.0, gamma=0.02, every=5, accord=None, monotone=False, dtype=DT):
    """Reproduit l'optimiseur de `fit_batch` (Adam, raideur k = k_start + (k_end − k_start)·t/steps) en ENREGISTRANT la
    trajectoire. Avec les valeurs par défaut, la trajectoire est celle de `fit_batch` (testé).
    `monotone=True` : variante de mesure où un pas n'est gardé que s'il améliore l'objectif À RAIDEUR FIXE k_end."""
    th = torch.as_tensor(init, dtype=dtype).reshape(1, -1).clone().requires_grad_(True); th0 = th.detach().clone().to(DT); fn = compile_skeleton(skel); X = data.X[None].to(dtype); y, f, w = data.y.to(dtype), data.flip.to(dtype), data.weight.to(dtype)
    opt = torch.optim.Adam([th], lr=lr); traj = []; best = {"profiled": (-math.inf, None, 0), "surrogate": (-math.inf, None, 0)}
    def snap(t, k, loss, gnorm):
        cur = th.detach().to(DT); pred = predict_batch(skel, cur, data.X)[0].to(DT); pl = profiled_loglik(pred, data); sur = float(loglik_batch(skel, cur, data, gamma)[0])
        if pl > best["profiled"][0] + 1e-9: best["profiled"] = (pl, cur[0].clone(), t)
        if sur > best["surrogate"][0] + 1e-9: best["surrogate"] = (sur, cur[0].clone(), t)
        prog = skel.skeleton().with_params(cur[0].tolist())
        traj.append({"step": t, "k": k, "loss": loss, "surrogate_loglik": sur, "profiled_loglik": pl, "errors": float((pred != data.y).double().sum()), "accord": accord(prog) if accord else None,
                     "grad_norm": gnorm, "distance": float((cur - th0).norm()), "max_shift": float((cur - th0).abs().max())})
    for t in range(steps):
        k = k_start + (k_end - k_start) * t / steps; opt.zero_grad(); loss = _objective(fn, th, X, y, f, w, gamma, k); loss.backward(); g = float(th.grad.norm())
        if t % every == 0: snap(t, k, float(loss.detach()), g)
        if monotone:
            prev = th.detach().clone(); ref = float(_objective(fn, prev, X, y, f, w, gamma, k_end)); opt.step()
            with torch.no_grad():
                if not float(_objective(fn, th, X, y, f, w, gamma, k_end)) < ref - 1e-12: th.copy_(prev)
        else: opt.step()
    snap(steps, k_end, float(_objective(fn, th.detach(), X, y, f, w, gamma, k_end)), 0.0)
    return th.detach().to(DT)[0], traj, best


def _spearman(a, b):
    n = len(a)
    if n < 4: return None
    def rk(v):
        o = sorted(range(n), key=lambda i: v[i]); r = [0.0] * n
        for p, i in enumerate(o): r[i] = float(p)
        return r
    ra, rb = rk(a), rk(b); ma, mb = sum(ra) / n, sum(rb) / n; sa = math.sqrt(sum((x - ma) ** 2 for x in ra)); sb = math.sqrt(sum((x - mb) ** 2 for x in rb))
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (sa * sb) if sa > 0 and sb > 0 else None


def local_geometry(skel: BNode, data: FitData, theta, gamma=0.02):
    """Géométrie locale de l'objectif continu AU POINT DE DÉPART : gradient (le point est-il stationnaire ?), hessienne
    (conditionnement, directions non identifiables, courbure négative), couplage entre atomes, saturation."""
    fn = compile_skeleton(skel); X = data.X[None]; out = {}; th0 = torch.as_tensor(theta, dtype=DT).reshape(-1)
    for k in (4.0, 24.0):
        f = lambda v: _objective(fn, v.reshape(1, -1), X, data.y, data.flip, data.weight, gamma, k); v = th0.clone().requires_grad_(True); g = torch.autograd.grad(f(v), v)[0]
        H = torch.autograd.functional.hessian(f, th0); H = (H + H.T) / 2; ev = torch.linalg.eigvalsh(H); big = float(ev.abs().max())
        sizes = [n.own_count() for n in _walk_leaves(skel)]; blocks = []; i = 0
        for sz in sizes: blocks.append((i, i + sz)); i += sz
        diag = sum(float((H[a:b, a:b] ** 2).sum()) for a, b in blocks); tot = float((H ** 2).sum())
        p = fn(th0.reshape(1, -1), X, lambda z: torch.sigmoid(k * z)); sat = float(((p < 0.02) | (p > 0.98)).double().mean())
        out[f"k{int(k)}"] = {"grad_norm": float(g.norm()), "grad_per_point": float(g.norm()) / max(1, len(data)), "condition": big / max(1e-12, float(ev.abs().min())), "min_eig": float(ev.min()), "max_eig": float(ev.max()),
                             "negative_eigs": int((ev < -1e-6 * big).sum()), "near_zero_eigs": int((ev.abs() < 1e-4 * big).sum()), "n_params": len(th0), "cross_atom_coupling": 1 - diag / tot if tot > 0 else 0.0, "saturated_fraction": sat}
    return out


def _walk_leaves(skel: BNode):
    """Atomes (sous-arbres non composites) dans l'ordre du vecteur de paramètres."""
    if skel.kind in ("and", "or", "xor"): return [l for c in skel.children for l in _walk_leaves(c)]
    return [_Whole(skel)]


class _Whole:
    def __init__(self, n): self.n = n
    def own_count(self): return self.n.n_params()


CONFIGS = {"default": {}, "k_start_12": {"k_start": 12.0}, "k_const_24": {"k_start": 24.0}, "lr_0.02": {"lr": 0.02}, "lr_0.005": {"lr": 0.005}, "k_const_24_lr_0.005": {"k_start": 24.0, "lr": 0.005},
           "k_reversed": {"k_start": 24.0, "k_end": 4.0}, "monotone_k24": {"k_start": 24.0, "monotone": True}, "float32": {"dtype": torch.float32}}


def optimizer_diagnosis(task, seed=0) -> dict:
    """Diagnostic de l'optimiseur sur deux points de départ quasi parfaits : la VRAIE structure aux VRAIS paramètres,
    et le candidat classé premier avant ajustement. L'optimiseur n'est pas modifié ; les variantes sont des MESURES."""
    from core.deep_reachability import leaves
    from core.robust_reconstruction import SENSOR_MODELS, Observations, fit_robust
    tr = normalize(task.world.truth); tl = leaves(tr); data = FitData.build(task.X, task.y, 0.10); A, _ = _eval(task); rows = {}
    gen = generate_candidates(task.X.float(), data.y, max_depth=min(6, n_atoms(tr) + 1), top=4, seed=seed, extra_literals=tl, geometry=False)
    top = normalize(gen.programs[max(range(len(gen.programs)), key=lambda i: gen.info[i]["score"])]) if gen.programs else tr
    for name, start in (("true_parameters", tr), ("top_candidate", top)):
        sk = start.skeleton(); init = start.param_vector(); st = lambda th: sk.with_params(th.tolist())
        def meas(th):
            pred = predict_batch(sk, th.reshape(1, -1), data.X)[0].to(DT); return {"accord": A(st(th)), "errors": float((pred != data.y).double().sum()), "profiled": profiled_loglik(pred, data), "surrogate": float(loglik_batch(sk, th.reshape(1, -1), data, 0.02)[0])}
        th0 = torch.tensor(init, dtype=DT); noop = meas(th0); r = {"noop": noop, "n_params": len(init), "atoms": n_atoms(start), "configs": {}, "geometry": local_geometry(sk, data, init)}
        for cname, kw in CONFIGS.items():
            t0 = time.time(); th, traj, best = optimize_traced(sk, data, init, accord=(lambda p: A(p)) if cname == "default" else None, **kw); fin = meas(th)
            c = {"final": fin, "seconds": round(time.time() - t0, 3), "max_shift": float((th - th0).abs().max()), "surrogate_improved": fin["surrogate"] > noop["surrogate"] + 1e-9, "profiled_improved": fin["profiled"] > noop["profiled"] + 1e-9,
                 "accord_degraded": fin["accord"] < noop["accord"] - 0.002, "best_profiled": {**meas(best["profiled"][1]), "step": best["profiled"][2]}, "best_surrogate": {**meas(best["surrogate"][1]), "step": best["surrogate"][2]}}
            if cname == "default":
                c["trajectory"] = traj; acc = [t["accord"] for t in traj]; c["corr_surrogate_accord"] = _spearman([t["surrogate_loglik"] for t in traj], acc); c["corr_loss_accord"] = _spearman([-t["loss"] for t in traj], acc)
                c["corr_profiled_accord"] = _spearman([t["profiled_loglik"] for t in traj], acc); c["first_drop_step"] = next((t["step"] for t in traj if t["accord"] < noop["accord"] - 0.002), None)
                c["first_drop_below_099_step"] = next((t["step"] for t in traj if t["accord"] < 0.99), None) if noop["accord"] >= 0.99 else None
                # arrêt précoce sur données accessibles : s'arrêter au premier contrôle où la vraisemblance profilée passe sous sa valeur initiale
                stop = next((i for i, t in enumerate(traj) if t["profiled_loglik"] < noop["profiled"] - 1e-9), None); c["early_stop"] = {"step": traj[stop - 1]["step"] if stop else traj[-1]["step"], "accord": traj[stop - 1]["accord"] if stop else traj[-1]["accord"], "stopped": stop is not None} if stop != 0 else {"step": 0, "accord": noop["accord"], "stopped": True}
                same, _ = fit_batch(sk, data, th0.reshape(1, -1), steps=100); c["matches_fit_batch"] = float((same[0] - th).abs().max())
            r["configs"][cname] = c
        # ancien ajusteur contre ajusteur par lots, même point de départ
        obs = Observations.build(task.X, task.y, SENSOR_MODELS["standard"], 0.05); t0 = time.time(); po, _ = fit_robust(sk, obs, steps=100, n_init=0, init=init); to = time.time() - t0; t0 = time.time(); tn, _ = fit_batch(sk, data, th0.reshape(1, -1), steps=100); tb = time.time() - t0
        tho = torch.tensor(po.param_vector(), dtype=DT); r["fitters"] = {"old": meas(tho), "new": meas(tn[0]), "param_diff": float((tho - tn[0]).abs().max()), "seconds_old": round(to, 4), "seconds_new": round(tb, 4),
                                                                      "init_rounding": float((torch.tensor(init, dtype=torch.float32).double() - th0).abs().max())}
        rows[name] = r
    return {"experiment": "optimizer", "starts": rows, "top_is_truth_form": algebraic_form(top) == algebraic_form(tr)}

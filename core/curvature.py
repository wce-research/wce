"""Stabilité du SECOND ORDRE pour les sous-systèmes différentiables (sans Hessienne dense).

Pour une compétence protégée de sorties f_i(θ) (marge signée m_i = y_i·f_i) sur des sondes générées :
  m_i(θ+δ) ≈ m_i + y_i·(J δ)_i + ½·y_i·δᵀH_iδ
- dérivée seconde DIRECTIONNELLE exacte δᵀH_iδ par double produit jvp (torch.func) — coût ≈ 2 passes ;
- courbure faible rang : directions propres dominantes de A = −Σ_i y_i H_i / max(m_i, ε) (les directions où les
  marges se referment au second ordre), par itération de puissance sur des produits Hessienne-vecteur ;
- ellipsoïde de stabilité anisotrope : δᵀQδ ≤ 1 avec Q = JᵀJ/τ² (premier ordre, faible rang) + U diag(λ/τ) Uᵀ
  (courbure, faible rang) + ε·I.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict

import torch
from torch.func import functional_call, jvp, vmap

KEYS = ("w1", "b1", "w2", "b2")


def mlp(p, X, act=torch.tanh):
    return act(X @ p["w1"].T + p["b1"]) @ p["w2"].T + p["b2"]


def flat(p): return torch.cat([p[k].flatten() for k in KEYS])


def unflat(v, like):
    out, i = {}, 0
    for k in KEYS: n = like[k].numel(); out[k] = v[i:i + n].reshape(like[k].shape); i += n
    return out


def margin_fn(p, X, y, forward=mlp):
    """Fonction vec(θ) -> marges signées (y∈{0,1} -> ±1) des sondes."""
    s = (2 * y - 1).to(torch.float32)
    def f(v):
        o = forward(unflat(v, p), X); return s * (o[:, 1] - o[:, 0])
    return f


def jacobian(p, X, y, forward=mlp):
    v = flat(p).detach()
    return torch.autograd.functional.jacobian(margin_fn(p, X, y, forward), v)


def directional_second(p, X, y, delta, forward=mlp):
    """δᵀH_iδ pour chaque sonde i (dérivée seconde exacte le long de δ), sans Hessienne."""
    f = margin_fn(p, X, y, forward); v = flat(p).detach()
    def first(t): return jvp(f, (t,), (delta,))[1]
    return jvp(first, (v,), (delta,))[1]


def predicted_margins(p, X, y, delta, forward=mlp, order=2):
    f = margin_fn(p, X, y, forward); v = flat(p).detach()
    m, d1 = jvp(f, (v,), (delta,))
    if order == 1: return m + d1
    return m + d1 + 0.5 * directional_second(p, X, y, delta, forward)


def low_rank_curvature(p, X, y, k=4, iters=20, eps=0.1, forward=mlp, seed=0):
    """k directions de courbure dominante de A(θ) = −Σ_i y-signé H_i / max(m_i, ε) (itération de puissance + déflation)."""
    f = margin_fn(p, X, y, forward); v0 = flat(p).detach()
    with torch.no_grad(): m = f(v0)
    w = -1.0 / m.clamp_min(eps)
    def scalar(v): return (w * f(v)).sum()
    def hvp(u): return torch.autograd.functional.hvp(scalar, v0, u)[1]
    g = torch.Generator().manual_seed(seed); U, L = [], []
    for _ in range(k):
        u = torch.randn(v0.shape, generator=g).to(v0.dtype); u = u / u.norm()
        lam = 0.0
        for _ in range(iters):
            z = hvp(u)
            for uj, lj in zip(U, L): z = z - lj * (uj @ u) * uj
            lam = float(u @ z); n = z.norm()
            if float(n) < 1e-12: break
            u = z / n
        U.append(u); L.append(lam)
    return torch.stack(U, 1), torch.tensor(L, dtype=v0.dtype)


@dataclass
class StabilityEllipsoid:
    """δᵀQδ ≤ 1 : région locale de stabilité anisotrope, Q = Aᵀ diag(a) A (faible rang) + ε·I (stockage compact)."""
    J_basis: torch.Tensor          # [P, r] directions sensibles premier ordre
    J_scale: torch.Tensor          # [r] valeurs singulières² / τ²
    C_basis: torch.Tensor          # [P, k] directions de courbure
    C_scale: torch.Tensor          # [k] |λ| / τ
    eps: float = 1e-4

    def quad(self, delta):
        a = (self.J_basis.T @ delta) ** 2 @ self.J_scale
        b = (self.C_basis.T @ delta).abs() ** 2 @ self.C_scale
        return float(a + b + self.eps * (delta @ delta))

    def contains(self, delta): return self.quad(delta) <= 1.0

    def floats(self): return self.J_basis.numel() + self.J_scale.numel() + self.C_basis.numel() + self.C_scale.numel() + 1


def ellipsoid(p, X, y, tau=0.5, rank=8, k=4, forward=mlp):
    J = jacobian(p, X, y, forward); U, S, Vh = torch.linalg.svd(J, full_matrices=False)
    r = min(rank, len(S)); Ub, Lc = low_rank_curvature(p, X, y, k, forward=forward)
    return StabilityEllipsoid(Vh[:r].T, (S[:r] ** 2) / tau ** 2, Ub, Lc.clamp_min(0) / tau)


def null_projector(p, X, y, energy=0.999, forward=mlp):
    J = jacobian(p, X, y, forward); U, S, Vh = torch.linalg.svd(J, full_matrices=False)
    cum = torch.cumsum(S ** 2, 0) / (S ** 2).sum().clamp_min(1e-12); r = int((cum < energy).sum()) + 1
    V = Vh[:r].T
    return (lambda g: g - V @ (V.T @ g)), V


def safe_step(p, X, y, delta, forward=mlp, max_flip=0.0, order=2, shrink=0.5, tries=12):
    """Réduit l'amplitude α de δ jusqu'à ce que le nombre prédit de marges retournées (ordre 1 ou 2) ≤ max_flip.
    Renvoie (α, flips prédits)."""
    with torch.no_grad(): m0 = margin_fn(p, X, y, forward)(flat(p))
    ok0 = m0 > 0; a = 1.0
    for _ in range(tries):
        pm = predicted_margins(p, X, y, a * delta, forward, order)
        flips = float(((pm <= 0) & ok0).float().mean())
        if flips <= max_flip: return a, flips
        a *= shrink
    return 0.0, flips


def actual_flips(p, X, y, delta, forward=mlp):
    with torch.no_grad():
        f = margin_fn(p, X, y, forward); m0 = f(flat(p)); m1 = f(flat(p) + delta)
    return float(((m1 <= 0) & (m0 > 0)).float().mean())

"""SECOND-ORDER stability for the differentiable subsystems (without a dense Hessian).

For a protected skill with outputs f_i(θ) (signed margin m_i = y_i·f_i) on generated probes:
  m_i(θ+δ) ≈ m_i + y_i·(J δ)_i + ½·y_i·δᵀH_iδ
- exact DIRECTIONAL second derivative δᵀH_iδ by a double jvp product (torch.func) — cost ≈ 2 passes;
- low-rank curvature: dominant eigendirections of A = −Σ_i y_i H_i / max(m_i, ε) (the directions along which the
  margins close at second order), by power iteration on Hessian-vector products;
- anisotropic stability ellipsoid: δᵀQδ ≤ 1 with Q = JᵀJ/τ² (first order, low rank) + U diag(λ/τ) Uᵀ
  (curvature, low rank) + ε·I.
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
    """Function vec(θ) -> signed margins (y∈{0,1} -> ±1) of the probes."""
    s = (2 * y - 1).to(torch.float32)
    def f(v):
        o = forward(unflat(v, p), X); return s * (o[:, 1] - o[:, 0])
    return f


def jacobian(p, X, y, forward=mlp):
    v = flat(p).detach()
    return torch.autograd.functional.jacobian(margin_fn(p, X, y, forward), v)


def directional_second(p, X, y, delta, forward=mlp):
    """δᵀH_iδ for each probe i (exact second derivative along δ), without a Hessian."""
    f = margin_fn(p, X, y, forward); v = flat(p).detach()
    def first(t): return jvp(f, (t,), (delta,))[1]
    return jvp(first, (v,), (delta,))[1]


def predicted_margins(p, X, y, delta, forward=mlp, order=2):
    f = margin_fn(p, X, y, forward); v = flat(p).detach()
    m, d1 = jvp(f, (v,), (delta,))
    if order == 1: return m + d1
    return m + d1 + 0.5 * directional_second(p, X, y, delta, forward)


def low_rank_curvature(p, X, y, k=4, iters=20, eps=0.1, forward=mlp, seed=0):
    """k directions of dominant curvature of A(θ) = −Σ_i signed-y H_i / max(m_i, ε) (power iteration + deflation)."""
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
    """δᵀQδ ≤ 1: local region of anisotropic stability, Q = Aᵀ diag(a) A (low rank) + ε·I (compact storage)."""
    J_basis: torch.Tensor          # [P, r] first-order sensitive directions
    J_scale: torch.Tensor          # [r] singular values² / τ²
    C_basis: torch.Tensor          # [P, k] curvature directions
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
    """Shrinks the amplitude α of δ until the predicted number of flipped margins (order 1 or 2) ≤ max_flip.
    Returns (α, predicted flips)."""
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

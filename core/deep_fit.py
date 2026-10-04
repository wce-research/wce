"""Moteur d'ajustement PAR LOTS (optimisation de performance du point chaud `fit_robust`).

PERFORMANCE OPTIMIZATION, pas changement scientifique : même vraisemblance (contamination), même schéma de raideur,
même optimiseur (Adam, mêmes hyperparamètres), mêmes initialisations possibles. Ce qui change :
  * le parcours d'arbre est COMPILÉ une fois par squelette (décalages de paramètres pré-calculés, plus de récursion
    Python ni de `n_params()` à chaque évaluation) ;
  * K initialisations (ou K candidats de même squelette) sont ajustées EN UN SEUL passage, tenseur [K, P] : le coût
    Python / autograd par pas est payé une fois au lieu de K.
Chaque ligne du lot est indépendante (la perte est une somme sur les lignes, Adam est élément par élément) : la
trajectoire d'une ligne est celle d'un ajustement isolé. `tests/test_deep.py` mesure l'écart numérique avec
`fit_robust` (référence gelée).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from core.boundary_program import BNode, _inits

DT = torch.float64


@dataclass
class FitData:
    """Observations pour l'ajustement : points, étiquettes, bruit par observation, poids (0 = point indifférent)."""
    X: torch.Tensor
    y: torch.Tensor
    flip: torch.Tensor
    weight: torch.Tensor

    @staticmethod
    def build(X, y, flip=0.05, weight=None):
        X = torch.nan_to_num(X).to(DT); n = len(X); f = flip if torch.is_tensor(flip) else torch.full((n,), float(flip))
        return FitData(X, y.to(DT), f.to(DT), (torch.ones(n) if weight is None else weight).to(DT))

    def with_target(self, y, weight=None): return FitData(self.X, y.to(DT), self.flip, self.weight if weight is None else weight.to(DT))
    def with_flip(self, flip): return FitData(self.X, self.y, torch.full((len(self.X),), float(flip), dtype=DT), self.weight)
    def add(self, x, y, flip): return FitData(torch.cat([self.X, x.to(DT)]), torch.cat([self.y, y.to(DT)]), torch.cat([self.flip, torch.full((len(x),), float(flip), dtype=DT)]), torch.cat([self.weight, torch.ones(len(x), dtype=DT)]))
    def __len__(self): return int((self.weight > 0).sum())


def _wrap(a): return torch.atan2(torch.sin(a), torch.cos(a))


_COMPILED: Dict[str, Callable] = {}


def compile_skeleton(skel: BNode) -> Callable:
    """Compile un squelette en fonction f(theta [K, P], X [1 ou K, N, 2], sig) → [K, N]. Mêmes formules que
    `core.boundary_program.evaluate`."""
    key = repr(skel.skeleton())
    if key in _COMPILED: return _COMPILED[key]
    off = [0]

    def build(n: BNode):
        m = n.own_count(); i = off[0]; off[0] += m; ch = [build(c) for c in n.children]
        if n.kind == "pred":
            if n.feat == "dot":
                def f0(th, X): return torch.cos(th[:, i:i + 1]) * X[..., 0] + torch.sin(th[:, i:i + 1]) * X[..., 1]
                if n.op == "le": return lambda th, X, sig: sig(th[:, i + 1:i + 2] - f0(th, X))
                if n.op == "ge": return lambda th, X, sig: sig(f0(th, X) - th[:, i + 1:i + 2])
                def dot_in(th, X, sig): f = f0(th, X); return sig(f - th[:, i + 1:i + 2]) * sig(th[:, i + 2:i + 3] - f)
                return dot_in
            if n.feat == "norm":
                if n.op == "le": return lambda th, X, sig: sig(th[:, i:i + 1] - X.norm(dim=-1))
                if n.op == "ge": return lambda th, X, sig: sig(X.norm(dim=-1) - th[:, i:i + 1])
                def norm_in(th, X, sig): f = X.norm(dim=-1); return sig(f - th[:, i:i + 1]) * sig(th[:, i + 1:i + 2] - f)
                return norm_in
            if n.feat == "angle":
                return lambda th, X, sig: sig(th[:, i + 1:i + 2].abs() - _wrap(torch.atan2(X[..., 1], X[..., 0]) - th[:, i:i + 1]).abs())
            raise KeyError(n.feat)
        c0 = ch[0]
        if n.kind == "translate": return lambda th, X, sig: c0(th, X - th[:, None, i:i + 2], sig)
        if n.kind == "rotate":
            def rot(th, X, sig):
                c, s = torch.cos(th[:, i:i + 1]), torch.sin(th[:, i:i + 1]); return c0(th, torch.stack([c * X[..., 0] + s * X[..., 1], -s * X[..., 0] + c * X[..., 1]], -1), sig)
            return rot
        if n.kind == "scale": return lambda th, X, sig: c0(th, X / th[:, None, i:i + 1].abs().clamp_min(1e-3), sig)
        if n.kind == "reflect":
            flipv = torch.tensor([1.0, -1.0], dtype=DT); return lambda th, X, sig: c0(th, X * flipv, sig)
        if n.kind == "not": return lambda th, X, sig: 1 - c0(th, X, sig)
        c1 = ch[1]
        if n.kind == "and": return lambda th, X, sig: c0(th, X, sig) * c1(th, X, sig)
        if n.kind == "or":
            def f_or(th, X, sig): a, b = c0(th, X, sig), c1(th, X, sig); return a + b - a * b
            return f_or
        if n.kind == "xor":
            def f_xor(th, X, sig): a, b = c0(th, X, sig), c1(th, X, sig); return a + b - 2 * a * b
            return f_xor
        raise KeyError(n.kind)

    fn = build(skel); _COMPILED[key] = fn; return fn


def forward(skel: BNode, theta, X, k: Optional[float]):
    """Probabilités adoucies [K, N] (k fini) ou valeurs dures (k None)."""
    sig = (lambda z: torch.sigmoid(k * z)) if k is not None else (lambda z: (z >= 0).to(DT))
    return compile_skeleton(skel)(theta, X[None] if X.dim() == 2 else X, sig)


def predict_batch(skel: BNode, theta, X): return (forward(skel, theta, torch.nan_to_num(X).to(DT), None) >= 0.5).long()


def loglik_batch(skel: BNode, theta, data: FitData, gamma=0.02, k=30.0):
    """log P(D | structure, θ) par ligne du lot (mélange de contamination, comme `obs_loglik`)."""
    p = forward(skel, theta, data.X, k).clamp(1e-6, 1 - 1e-6); p1 = (1 - gamma) * (data.flip + (1 - 2 * data.flip) * p) + gamma * 0.5
    return (data.weight * (data.y * torch.log(p1) + (1 - data.y) * torch.log(1 - p1))).sum(1)


class FitCounter:
    """Compte le calcul en « équivalents-ajustement » (une initialisation × 100 pas) et en candidats."""
    def __init__(self, budget: Optional[float] = None): self.fit_equivalents = 0.0; self.candidates = 0; self.calls = 0; self.budget = budget
    def charge(self, k, steps): self.fit_equivalents += k * steps / 100.0; self.calls += 1
    def exhausted(self): return self.budget is not None and self.fit_equivalents >= self.budget
    def remaining(self): return math.inf if self.budget is None else max(0.0, self.budget - self.fit_equivalents)


def fit_batch(skel: BNode, data: FitData, inits, steps=100, gamma=0.02, lr=0.05, counter: Optional[FitCounter] = None):
    """Ajuste K jeux de paramètres d'un même squelette en un passage. Renvoie (theta [K, P], logL [K]).
    Schéma identique à `fit_robust` : Adam(lr), raideur k = 4 + 20·t/steps, vraisemblance à contamination."""
    th0 = inits if torch.is_tensor(inits) else torch.stack([torch.as_tensor(i, dtype=DT) for i in inits])
    th = th0.clone().to(DT).requires_grad_(True)
    if th.shape[1] == 0: return th.detach(), loglik_batch(skel, th.detach(), data, gamma)
    opt = torch.optim.Adam([th], lr=lr); fn = compile_skeleton(skel); X = data.X[None]; f, w, y = data.flip, data.weight, data.y
    for t in range(steps):
        opt.zero_grad(); kk = 4.0 + 20.0 * t / steps
        p = fn(th, X, lambda z: torch.sigmoid(kk * z)).clamp(1e-6, 1 - 1e-6); p1 = (1 - gamma) * (f + (1 - 2 * f) * p) + gamma * 0.5
        (-(w * (y * torch.log(p1) + (1 - y) * torch.log(1 - p1))).sum()).backward(); opt.step()
    th = th.detach()
    if counter is not None: counter.charge(len(th), steps)
    return th, loglik_batch(skel, th, data, gamma)


def heuristic_inits(skel: BNode, data: FitData, n_init: int, seed: int):
    """Initialisations heuristiques du jalon précédent (`_inits`), sur les points de poids non nul."""
    m = data.weight > 0; g = torch.Generator().manual_seed(seed)
    return torch.stack([v.to(DT) for v in _inits(skel, data.X[m].float(), data.y[m].long(), n_init, g)])


def best_of(skel: BNode, theta, ll):
    """Meilleure ligne du lot → (programme ajusté, logL)."""
    i = int(ll.argmax()); return skel.skeleton().with_params(theta[i].tolist()), float(ll[i])


def approx_evidence(n_params: int, n_atoms: int, ll: float, n_obs: int, bits_per_node=3.0):
    """Évidence APPROCHÉE (type BIC + a priori structurel), même forme que le jalon précédent : ce n'est pas une
    marginale exacte."""
    return ll - 0.5 * n_params * math.log(max(1, n_obs)) - (2 * n_atoms - 1) * bits_per_node * math.log(2)


# ================================================================ ajustement par SÉPARATION (changement scientifique, pas une optimisation)

def separation_fit(skel: BNode, P, Q, pair_weight, inits, steps=50, lr=0.05, counter: Optional[FitCounter] = None):
    """SCIENTIFIC CHANGE (génération de candidats) — objectif indépendant de la polarité : ajuster un atome pour que sa
    frontière passe ENTRE les deux points de chaque paire de frontière (points voisins d'étiquettes opposées).
    Dans une composition and / or / xor, un atome masqué par un XOR n'est pas corrélé à la cible, mais sa frontière reste
    visible localement : cet objectif la trouve là où l'ajustement sur la cible échoue.
    Probabilité de séparation d'une paire : m(P)(1 − m(Q)) + m(Q)(1 − m(P)), m = appartenance adoucie.
    Renvoie (theta [K, np], séparation pondérée dure [K])."""
    th = (inits if torch.is_tensor(inits) else torch.stack([torch.as_tensor(i, dtype=DT) for i in inits])).clone().to(DT).requires_grad_(True)
    fn = compile_skeleton(skel); P, Q, w = P.to(DT)[None], Q.to(DT)[None], pair_weight.to(DT); opt = torch.optim.Adam([th], lr=lr)
    for t in range(steps):
        opt.zero_grad(); kk = 3.0 + 12.0 * t / steps; sig = lambda z: torch.sigmoid(kk * z)
        mp, mq = fn(th, P, sig), fn(th, Q, sig); (-(w * (mp * (1 - mq) + mq * (1 - mp))).sum()).backward(); opt.step()
    th = th.detach()
    if counter is not None: counter.charge(len(th), steps)
    hard = lambda z: (z >= 0).to(DT); return th, (w * (fn(th, P, hard) != fn(th, Q, hard)).to(DT)).sum(1)

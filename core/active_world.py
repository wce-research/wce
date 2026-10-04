"""Générateur du banc « Active Structural Discovery » (côté ÉVALUATION : les noms de familles n'existent QU'ICI
et dans core/boundary_world.py). Le code de découverte (core/active_structure.py) ne voit que des observations.

Nouvelles familles :
  profondeur 3/4 (non représentables à profondeur 2) : triangle (3 demi-espaces), kite (4 demi-espaces),
      notched_disc (disque − disque, coupé par un demi-espace), lobes_cut ((disque ∪ disque) ∩ demi-espace)
  négatifs difficiles (identiques à une famille connue dans la région observée, divergents hors région) :
      disc_cut_far (disque coupé seulement hors région), strip_capped (bande fermée hors région)
  syntaxe multiple (même fonction, AST différents) : two_lobe_xor (deux disques disjoints ⇒ OR ≡ XOR)
  scellées : pentagon_cut, ring_cut, triangle_minus_disc, kite_rotated
"""
from __future__ import annotations

import hashlib
import json
import math
import random

import torch

from core.boundary_world import (FamilyInstance, HELD_OUT, TRAIN, UNREPRESENTABLE, VALIDATION, full_region, in_region, make_instance,
                                 sample_task)

DEPTH_FAMILIES = ("triangle", "kite", "notched_disc", "lobes_cut")
HARD_NEGATIVES = ("disc_cut_far", "strip_capped")
MULTI_SYNTAX = ("two_lobe_xor",)
SEALED_ACTIVE = ("pentagon_cut", "ring_cut", "triangle_minus_disc", "kite_rotated")
NOISE_LEVELS = (0.0, 0.02, 0.05, 0.08, 0.12)


def _dot(X, a): return math.cos(a) * X[:, 0] + math.sin(a) * X[:, 1]
def _disc(X, c, r): return (X - torch.tensor(c)).norm(dim=1) < r


def _polygon(rng, k, center, rad, rot=None):
    rot = rng.uniform(-math.pi, math.pi) if rot is None else rot
    angs = [rot + 2 * math.pi * i / k + rng.uniform(-0.25, 0.25) for i in range(k)]
    hs = [(a, rad * rng.uniform(0.8, 1.2) + _dot(torch.tensor([center]), a).item()) for a in angs]
    return lambda X: torch.stack([_dot(X, a) <= b for a, b in hs]).all(0)


def make_active_instance(family, rng: random.Random) -> FamilyInstance:
    if family in ("triangle", "kite", "pentagon_cut", "kite_rotated"):
        k = {"triangle": 3, "kite": 4, "pentagon_cut": 5, "kite_rotated": 4}[family]
        c = (rng.uniform(-0.9, 0.2), rng.uniform(-0.6, 0.6)); poly = _polygon(rng, k, c, rng.uniform(0.9, 1.4))
        return FamilyInstance(family, {}, lambda X: poly(X).long(), True, None)
    if family == "notched_disc":
        c = (rng.uniform(-0.8, 0.0), rng.uniform(-0.5, 0.5)); r = rng.uniform(1.0, 1.4); d = (c[0] + rng.uniform(-0.6, 0.6), c[1] + r * 0.8); a, b = rng.uniform(-3, 3), rng.uniform(-0.2, 0.4)
        return FamilyInstance(family, {}, lambda X: (_disc(X, c, r) & ~_disc(X, d, 0.55 * r) & (_dot(X, a) <= b + _dot(torch.tensor([c]), a).item())).long(), True, None)
    if family == "lobes_cut":
        c1, c2 = (rng.uniform(-1.6, -0.8), rng.uniform(-1, 1)), (rng.uniform(0.0, 1.2), rng.uniform(-1, 1)); a = rng.uniform(-3, 3)
        return FamilyInstance(family, {}, lambda X: ((_disc(X, c1, 0.8) | _disc(X, c2, 0.8)) & (_dot(X, a) <= 0.3)).long(), True, None)
    if family == "disc_cut_far":
        c = (rng.uniform(-0.6, 0.2), rng.uniform(-0.5, 0.5)); r = rng.uniform(1.2, 1.6); cut = rng.uniform(0.7, 1.0)
        return FamilyInstance(family, {}, lambda X: (_disc(X, c, r) & (X[:, 0] <= cut)).long(), True, None)
    if family == "strip_capped":
        a = rng.uniform(-0.4, 0.4); t = rng.uniform(0.4, 0.7); cap = rng.uniform(0.8, 1.3)
        return FamilyInstance(family, {}, lambda X: ((_dot(X, a + math.pi / 2).abs() <= t) & (X[:, 0] <= cap)).long(), True, None)
    if family == "two_lobe_xor":
        c1, c2 = (rng.uniform(-1.7, -1.0), rng.uniform(-1, 1)), (rng.uniform(0.3, 1.2), rng.uniform(-1, 1))
        return FamilyInstance(family, {}, lambda X: (_disc(X, c1, 0.7) ^ _disc(X, c2, 0.7)).long(), True, None)
    if family == "ring_cut":
        c = (rng.uniform(-0.5, 0.3), rng.uniform(-0.4, 0.4)); r1 = rng.uniform(0.5, 0.8); r2 = r1 + 0.7; a = rng.uniform(-3, 3)
        return FamilyInstance(family, {}, lambda X: (_disc(X, c, r2) & ~_disc(X, c, r1) & (_dot(X, a) <= 0.2)).long(), True, None)
    if family == "triangle_minus_disc":
        c = (rng.uniform(-0.8, 0.0), rng.uniform(-0.4, 0.4)); poly = _polygon(rng, 3, c, 1.4); d = (c[0] + 0.3, c[1])
        return FamilyInstance(family, {}, lambda X: (poly(X) & ~_disc(X, d, 0.5)).long(), True, None)
    return make_instance(family, rng)


def balanced_instances(families, seed, per_family):
    rng = random.Random(seed); g = torch.Generator().manual_seed(seed); out = []
    for f in families:
        k = 0; tries = 0
        while k < per_family and tries < 60:
            tries += 1; inst = make_active_instance(f, rng)
            if 0.08 < float(inst.label(in_region(800, g)).float().mean()) < 0.92: out.append(inst); k += 1
    return out


def transformed(inst: FamilyInstance, rng: random.Random):
    """Même fonction après rotation / translation / échelle / réflexion (représentation transformée)."""
    phi = rng.uniform(-math.pi, math.pi); s = rng.uniform(0.7, 1.3); t = torch.tensor([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4)]); refl = rng.random() < 0.5
    c, sn = math.cos(phi), math.sin(phi); R = torch.tensor([[c, -sn], [sn, c]])
    def f(X):
        Z = ((X - t) / s) @ R
        if refl: Z = Z * torch.tensor([1.0, -1.0])
        return inst.label(Z)
    return FamilyInstance(inst.family + ":T", {}, f, inst.representable, None)


def noisy_task(inst, n, g, noise): return sample_task(inst, n, g, label_noise=noise, x_noise=0.02)


def sealed_spec(): return {"families": list(SEALED_ACTIVE) + ["disc_cut_far", "two_lobe_xor", "spiral", "radial_wave"], "seed": 991333, "per_family": 2, "noise": [0.08, 0.12]}


def sealed_hash():
    s = sealed_spec(); probe = full_region(64, torch.Generator().manual_seed(1))
    sig = [[i.family, "".join(str(int(v)) for v in i.label(probe).tolist())] for i in balanced_instances(s["families"], s["seed"], s["per_family"])]
    return hashlib.sha256(json.dumps([sig, s["noise"]]).encode()).hexdigest()


SEALED_SHA256 = "264e73ca91d45f3bfae18081b3b23d4f31fcf68e622ee00f0db7193f5560e733"   # figée AVANT toute évaluation


def train_deep_instance(rng: random.Random) -> FamilyInstance:
    """Compositions PROFONDES (3 atomes) de familles d'ENTRAÎNEMENT seulement — pour apprendre le proposeur
    sans jamais voir les familles de profondeur 3/4 du test."""
    parts = [make_instance(rng.choice(TRAIN), rng) for _ in range(3)]; ops = [rng.choice(["and", "or"]) for _ in range(2)]
    def f(X):
        a = parts[0].label(X).bool()
        for op, p in zip(ops, parts[1:]): b = p.label(X).bool(); a = (a & b) if op == "and" else (a | b)
        return a.long()
    return FamilyInstance("train_deep", {}, f, True, None)


def train_tasks(seed, noise=0.05, n=120):
    """Tâches d'apprentissage du proposeur : familles TRAIN + VALIDATION + compositions profondes de TRAIN."""
    rng = random.Random(seed); g = torch.Generator().manual_seed(seed); insts = balanced_instances(TRAIN + VALIDATION, seed, 1)
    k = 0
    while k < 4:
        i = train_deep_instance(rng)
        if 0.1 < float(i.label(in_region(800, g)).float().mean()) < 0.9: insts.append(i); k += 1
    out = []
    for j, inst in enumerate(insts):
        gg = torch.Generator().manual_seed(seed * 100 + j); X, y = noisy_task(inst, n, gg, noise); out.append((X, y, inst.label, full_region, torch.Generator().manual_seed(seed * 100 + j + 1)))
    return out

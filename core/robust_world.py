"""Monde synthétique ROBUSTE (côté évaluation ; le code d'inférence n'importe jamais ce module).

Structures VRAIES = programmes aléatoires de profondeur 1–4 composés des primitives existantes (AND / OR / XOR /
NOT, translations, rotations implicites via dot(α)), plus des fonctions NON représentables. Aucune famille n'est
nommée dans les métadonnées vues par le système (seule la vérité d'évaluation porte un identifiant opaque).

Processus de mesure : capteurs hétérogènes (précis / standard / bon marché / flou près des frontières / biaisé),
bruit i.i.d., bruit concentré près de la frontière, bruit asymétrique, source systématiquement biaisée, outliers
GROUPÉS, observations manquantes, mesures répétables, régions non observées, horodatage (observation / ingestion).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from core.boundary_program import BNode, predict

NOISE_LEVELS = (0.0, 0.02, 0.05, 0.08, 0.12, 0.15)
NOISE_KINDS = ("iid", "boundary", "asymmetric", "grouped_outliers", "biased_source")


# ================================================================ capteurs (compromis artificiels de mesure)

@dataclass(frozen=True)
class Sensor:
    name: str
    flip: float                 # taux de retournement de base
    cost: float
    blur: float = 0.0           # flou près de la frontière (résolution)
    latency: float = 0.0


SENSORS = {"precise": Sensor("precise", 0.01, 3.0, 0.02, 2.0), "standard": Sensor("standard", 0.05, 1.0, 0.05, 1.0),
           "cheap": Sensor("cheap", 0.15, 0.3, 0.15, 0.2)}


# ================================================================ structures vraies

def _atom(rng: random.Random, center):
    c = (center[0] + rng.uniform(-0.6, 0.6), center[1] + rng.uniform(-0.6, 0.6)); k = rng.random()
    T = lambda child: BNode("translate", params=c, children=(child,))
    if k < 0.35: a = rng.uniform(-math.pi, math.pi); return BNode("pred", "dot", "le", (a, math.cos(a) * c[0] + math.sin(a) * c[1]))
    if k < 0.55: r = rng.uniform(0.7, 1.4); return T(BNode("pred", "norm", "le", (r,)))
    if k < 0.65: r = rng.uniform(0.5, 0.9); return T(BNode("pred", "norm", "ge", (r,)))
    if k < 0.8:
        r1 = rng.uniform(0.4, 0.8); return T(BNode("pred", "norm", "in", (r1, r1 + rng.uniform(0.5, 0.9))))
    if k < 0.9:
        a = rng.uniform(-math.pi, math.pi); b = math.cos(a) * c[0] + math.sin(a) * c[1]; t = rng.uniform(0.3, 0.7)
        return BNode("pred", "dot", "in", (a, b - t, b + t))
    return T(BNode("pred", "angle", "in", (rng.uniform(-math.pi, math.pi), rng.uniform(0.5, 1.2))))


def random_program(rng: random.Random, depth: int):
    """Programme vrai : chaîne gauche de `depth` atomes combinés (profondeur logique = nb d'atomes)."""
    center = (rng.uniform(-1.0, 0.2), rng.uniform(-0.6, 0.6)); p = _atom(rng, center)
    for _ in range(depth - 1): p = BNode(rng.choice(("and", "or", "xor")), children=(p, _atom(rng, center)))
    return p


@dataclass
class WorldInstance:
    uid: str                                   # identifiant OPAQUE (aucun nom de famille)
    label: Callable
    truth: Optional[BNode]                     # programme vrai (évaluation seulement) ; None si non représentable
    depth: int
    representable: bool


def unrepresentable(rng: random.Random):
    k = rng.random(); ph = rng.uniform(0, 6.28)
    if k < 0.5: return lambda X: (X[:, 1] > 0.8 * torch.sin(5 * X[:, 0] + ph)).long()
    s = rng.uniform(1.3, 1.8); return lambda X: ((torch.floor(X[:, 0] * s) + torch.floor(X[:, 1] * s)) % 2).long()


def full_region(n, g): return torch.rand(n, 2, generator=g) * 6 - 3
def in_region(n, g): return torch.stack([torch.rand(n, generator=g) * 2.5 - 2.0, torch.rand(n, generator=g) * 4 - 2], 1)
def out_region(n, g): return torch.stack([torch.rand(n, generator=g) * 2.5 + 0.5, torch.rand(n, generator=g) * 6 - 3], 1)


def make_world(seed, depth=None, representable=True):
    rng = random.Random(seed); g = torch.Generator().manual_seed(seed)
    for _ in range(80):
        d = depth or rng.choice((1, 2, 3, 4))
        if representable:
            p = random_program(rng, d); lab = (lambda X, p=p: predict(p, X))
        else: p = None; lab = unrepresentable(rng)
        m = float(lab(in_region(1500, g)).float().mean()); mf = float(lab(full_region(1500, g)).float().mean())
        if 0.12 < m < 0.88 and 0.08 < mf < 0.92:
            uid = hashlib.sha1(f"{seed}-{d}-{representable}".encode()).hexdigest()[:10]
            return WorldInstance(uid, lab, p, d if representable else 0, representable)
    return WorldInstance(f"w{seed}", lab, p, d if representable else 0, representable)


# ================================================================ processus d'observation

@dataclass
class NoiseSpec:
    rate: float = 0.05
    kind: str = "iid"
    missing: float = 0.03


def corrupt(world: WorldInstance, X, sensor: Sensor, spec: NoiseSpec, g):
    """Observation bruitée selon capteur + bruit du monde. Renvoie y (−1 = manquant)."""
    y = world.label(X).clone(); n = len(X)
    base = torch.full((n,), sensor.flip + spec.rate)
    if sensor.blur > 0 or spec.kind == "boundary":                        # bruit concentré près de la frontière (proxy : désaccord sous petite perturbation)
        d = torch.randn(X.shape, generator=g) * (sensor.blur + (0.15 if spec.kind == "boundary" else 0.0))
        near = (world.label(X + d) != y).float()
        base = base * (0.5 if spec.kind == "boundary" else 1.0) + near * (0.35 if spec.kind == "boundary" else 0.25)
    if spec.kind == "asymmetric": base = torch.where(y == 1, base * 1.8, base * 0.2)
    flip = torch.rand(n, generator=g) < base.clamp(0, 0.49)
    if spec.kind == "grouped_outliers" and n >= 10:
        c = X[int(torch.randint(0, n, (1,), generator=g))]; grp = (X - c).norm(dim=1) < 0.35
        flip = flip | (grp & (torch.rand(n, generator=g) < 0.85))
    if spec.kind == "biased_source": flip = flip | ((y == 0) & (torch.rand(n, generator=g) < spec.rate * 1.5))
    y = torch.where(flip, 1 - y, y)
    miss = torch.rand(n, generator=g) < spec.missing
    return torch.where(miss, torch.full_like(y, -1), y)


@dataclass
class Task:
    world: WorldInstance
    X: torch.Tensor
    y: torch.Tensor
    sensor_ids: List[str]
    obs_time: torch.Tensor
    spec: NoiseSpec
    seed: int

    def oracle(self, x, sensor="standard", g=None):
        """Mesure ACTIVE : passe par le même processus de bruit (jamais la vérité)."""
        return corrupt(self.world, x, SENSORS[sensor], self.spec, g)


def make_task(seed, noise=0.08, kind="iid", n=120, depth=None, representable=True):
    w = make_world(seed, depth, representable); g = torch.Generator().manual_seed(seed + 1)
    X = in_region(n, g); y = corrupt(w, X, SENSORS["standard"], NoiseSpec(noise, kind), g)
    return Task(w, X, y, ["standard"] * n, torch.zeros(n), NoiseSpec(noise, kind), seed)


def hard_noise_pair(seed):
    """Cas A : structure SIMPLE + 8 % de bruit ; cas B : structure COMPLEXE + 0 % (mêmes atomes de départ)."""
    a = make_task(seed, noise=0.08, depth=1); b = make_task(seed, noise=0.0, depth=3); return a, b


def sealed_spec():
    return {"seeds": list(range(992000, 992024)), "noise": [0.08, 0.12, 0.15], "kinds": ["iid", "grouped_outliers", "boundary"], "unrepresentable_every": 6}


def sealed_tasks():
    s = sealed_spec(); out = []
    for i, sd in enumerate(s["seeds"]):
        out.append(make_task(sd, s["noise"][i % 3], s["kinds"][(i // 3) % 3], representable=(i % s["unrepresentable_every"] != 5)))
    return out


def sealed_hash():
    probe = full_region(64, torch.Generator().manual_seed(1))
    sig = [[t.world.uid, "".join(str(int(v)) for v in t.world.label(probe).tolist()), t.spec.rate, t.spec.kind, int(t.y.sum())] for t in sealed_tasks()]
    return hashlib.sha256(json.dumps(sig).encode()).hexdigest()


SEALED_SHA256 = "5cdbffc6ad8b21eda43f56a7c08219cd2434b789a51d3c2f658bf70bb8517b69"   # figée AVANT toute évaluation

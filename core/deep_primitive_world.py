"""ATOMIC bench for primitive discovery — EVALUATION module (dev only, own seeds).

Minimal structural depth (flat composition) to isolate atomic discovery from composition:
  single     a single atom
  two        two atoms (XOR: all the boundaries remain visible)
  multi      three or four atoms
  close      two close atoms (neighbouring centres)
  nested     nested atoms (a disc inside a disc)
  redundant  redundant atoms of the grammar, written in the minimal basis (band = two half-planes, ring = two discs)
  noise      high noise
  outliers   grouped outliers
  partial    partially observed boundaries (atoms straddling the limit of the observed zone)

Out of distribution (for a learned proposer): new positions, sizes, orientations, densities, noise.
The sealed set is never touched: seeds of the dev partition, dedicated base.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional

import torch

from core.boundary_program import BNode, predict
from core.deep_world import OBS_XMAX, SPLIT_BASE, DeepTask, DeepWorld, NoiseSpec, corrupt, obs_region
from core.robust_world import SENSORS

CATEGORIES = ("single", "two", "multi", "close", "nested", "redundant", "noise", "outliers", "partial")
OOD_SHIFTS = ("position", "size", "orientation", "density", "noise")
ATOMIC_BASE = SPLIT_BASE["dev"] + 500_000                 # sub-range of the dev partition reserved for the atomic bench
TRAIN_BASE = SPLIT_BASE["dev"] + 700_000                  # training scenes of the neural proposer (dev, disjoint)


def _T(c, ch): return BNode("translate", params=(float(c[0]), float(c[1])), children=(ch,))


def atom(rng: random.Random, kind: str, shift: Optional[str] = None):
    """Random atom; `shift` moves ONE parameter range out of the nominal distribution."""
    cx = rng.uniform(0.3, 1.0) if shift == "position" else rng.uniform(-2.0, 0.2); cy = rng.uniform(1.6, 2.6) * rng.choice((-1, 1)) if shift == "position" else rng.uniform(-1.6, 1.6)
    s = rng.choice((0.55, 1.6)) if shift == "size" else 1.0; a = rng.uniform(-0.4, 0.4) if shift == "orientation" else rng.uniform(-math.pi, math.pi)
    if kind == "half": return BNode("pred", "dot", "le", (a, math.cos(a) * cx + math.sin(a) * cy))
    if kind == "band": b = math.cos(a) * cx + math.sin(a) * cy; t = rng.uniform(0.4, 0.8) * s; return BNode("pred", "dot", "in", (a, b - t, b + t))
    if kind == "disc": return _T((cx, cy), BNode("pred", "norm", "le", (rng.uniform(0.8, 1.4) * s,)))
    if kind == "antidisc": return _T((cx, cy), BNode("pred", "norm", "ge", (rng.uniform(0.7, 1.2) * s,)))
    if kind == "ring": r = rng.uniform(0.5, 0.9) * s; return _T((cx, cy), BNode("pred", "norm", "in", (r, r + rng.uniform(0.6, 1.0) * s)))
    if kind == "sector": return _T((cx, cy), BNode("pred", "angle", "in", (a, rng.uniform(0.5, 1.1))))
    raise KeyError(kind)


TYPES = ("half", "band", "disc", "antidisc", "ring", "sector")


def _xor(atoms):
    p = atoms[0]
    for a in atoms[1:]: p = BNode("xor", children=(p, a))
    return p


@dataclass
class Scene:
    task: DeepTask
    atoms: List[BNode]              # true atoms (evaluation only)
    category: str
    shift: Optional[str]


def make_scene(index: int, category: str, shift: Optional[str] = None, n=300, base=ATOMIC_BASE) -> Scene:
    """Deterministic scene in (index, category, shift)."""
    seed = base + index * 97 + CATEGORIES.index(category) * 7 + (0 if shift is None else 1000 * (1 + OOD_SHIFTS.index(shift))); probe = obs_region(1500, torch.Generator().manual_seed(seed))
    rate, kind = 0.05, "iid"
    if category == "noise" or shift == "noise": rate = 0.12
    if category == "outliers": kind = "grouped_outliers"
    if shift == "density": n = 150 if index % 2 else 600
    for attempt in range(200):
        rng = random.Random(seed * 131 + attempt); sh = shift if shift in ("position", "size", "orientation") else None
        if category in ("single", "noise", "outliers"): atoms = [atom(rng, rng.choice(TYPES), sh)]
        elif category == "two": atoms = [atom(rng, rng.choice(TYPES), sh) for _ in range(2)]
        elif category == "multi": atoms = [atom(rng, rng.choice(TYPES), sh) for _ in range(rng.choice((3, 4)))]
        elif category == "close":
            a = atom(rng, "disc", sh); c = a.params; atoms = [a, _T((c[0] + rng.uniform(0.3, 0.6), c[1] + rng.uniform(-0.3, 0.3)), BNode("pred", "norm", "le", (rng.uniform(0.8, 1.3),)))]
        elif category == "nested":
            a = atom(rng, "disc", sh); c = a.params; r = a.children[0].params[0]; atoms = [a, _T((c[0] + rng.uniform(-0.15, 0.15), c[1] + rng.uniform(-0.15, 0.15)), BNode("pred", "norm", "le", (r * rng.uniform(0.4, 0.6),)))]
        elif category == "redundant":
            if rng.random() < 0.5:
                al = rng.uniform(-math.pi, math.pi); b = rng.uniform(-1.0, 0.5); t = rng.uniform(0.5, 0.9); atoms = [BNode("pred", "dot", "le", (al, b + t)), BNode("pred", "dot", "le", (math.atan2(math.sin(al + math.pi), math.cos(al + math.pi)), -(b - t)))]
            else:
                c = (rng.uniform(-1.6, 0.0), rng.uniform(-1.2, 1.2)); r = rng.uniform(0.5, 0.8); atoms = [_T(c, BNode("pred", "norm", "le", (r + rng.uniform(0.6, 1.0),))), _T(c, BNode("pred", "norm", "ge", (r,)))]
        elif category == "partial":
            a = atom(rng, rng.choice(("disc", "ring")), sh); atoms = [BNode("translate", params=(OBS_XMAX + rng.uniform(-0.3, 0.3), a.params[1]), children=a.children)]
        else: raise KeyError(category)
        prog = BNode("and", children=tuple(atoms)) if category == "redundant" else _xor(atoms); m = float(predict(prog, probe).float().mean())
        if 0.08 < m < 0.92: break
    w = DeepWorld(f"atomic-{seed}", (lambda X, t=0.0, p=prog: predict(p, X)), prog, len(atoms), True, f"atomic_{category}", False)
    g = torch.Generator().manual_seed(seed + 1); X = obs_region(n, g); t = torch.rand(n, generator=g); spec = NoiseSpec(rate, kind); y = corrupt(w, X, t, SENSORS["standard"], spec, g)
    return Scene(DeepTask(w, X, y, t, spec, "atomic", seed, "dev"), atoms, category, shift)

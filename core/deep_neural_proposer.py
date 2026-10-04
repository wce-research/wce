"""Proposeur NEURONAL COMPACT de primitives (≈ 24 000 paramètres).

Discipline :
  * il ne prédit ni la structure complète ni les étiquettes du monde ; il propose seulement des COMPOSANTES DE FRONTIÈRE
    en base minimale (cercles : centre + rayon ; droites : pied de la perpendiculaire + direction) ;
  * ses sorties sont des paramètres d'atomes explicites, converties en primitives candidates ;
  * la validation reste structurelle, côté WCE (core.deep_primitives.PrimitiveValidation) ;
  * pas de gros réseau : 4 convolutions 3×3 sur une grille 24×24, têtes 1×1.

Entrée : observations rastérisées (points positifs, points négatifs, zone observée). Sortie par cellule : présence d'un
centre de cercle (+ décalage, rayon) et d'une ancre de droite (+ décalage, cos, sin de la normale).

L'inférence (`NeuralProposer.propose`) n'utilise que les observations. L'entraînement (`train`) utilise la vérité de
scènes d'ENTRAÎNEMENT de la partition dev (jamais validation, test ni lot scellé).
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from core.deep_fit import FitCounter
from core.deep_primitives import Pairs, PrimitiveCandidate, PrimitiveProposal, basis
from core.deep_structure import skel_key

GRID = 24
LO, HI = -3.0, 3.0
CELL = (HI - LO) / GRID
OBS_LIMIT = 1.2                      # limite de la zone observée (information d'acquisition, pas de vérité)
R_SCALE = 2.0
WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "deep_configs")


class ProposerNet(nn.Module):
    """Réseau convolutif compact. Sortie [B, 9, G, G] : cercle (présence, dx, dy, rayon) + droite (présence, dx, dy, cos, sin)."""

    def __init__(self, width=32):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, width, 3, padding=2, dilation=2), nn.ReLU(),
                                  nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.ReLU(), nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
        self.head = nn.Conv2d(width, 9, 1)

    def forward(self, x): return self.head(self.body(x))
    def n_params(self): return sum(p.numel() for p in self.parameters())

    def flops(self):
        """Proxy : multiplications-additions d'un passage (convolutions seulement)."""
        f = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d): f += m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] * GRID * GRID
        return f


def rasterize(X, y):
    """Observations → tenseur [3, G, G] : densité de positifs, densité de négatifs, masque de zone observée."""
    ij = ((X.float() - LO) / CELL).long().clamp(0, GRID - 1); idx = ij[:, 1] * GRID + ij[:, 0]; pos = torch.zeros(GRID * GRID).index_add_(0, idx, (y == 1).float()); neg = torch.zeros(GRID * GRID).index_add_(0, idx, (y == 0).float())
    s = (pos.sum() + neg.sum()) / (GRID * GRID) + 1e-6; xs = LO + (torch.arange(GRID) + 0.5) * CELL; mask = (xs[None, :] <= OBS_LIMIT).float().expand(GRID, GRID)
    return torch.stack([(pos / s).view(GRID, GRID), (neg / s).view(GRID, GRID), mask])


def _cell(px, py):
    i = min(GRID - 1, max(0, int((px - LO) / CELL))); j = min(GRID - 1, max(0, int((py - LO) / CELL))); return i, j, (px - (LO + (i + 0.5) * CELL)) / CELL, (py - (LO + (j + 0.5) * CELL)) / CELL


def targets(components: Sequence[tuple]):
    """Composantes vraies d'une scène d'ENTRAÎNEMENT → cibles [9, G, G] + masque de régression [2, G, G]."""
    T = torch.zeros(9, GRID, GRID); M = torch.zeros(2, GRID, GRID)
    for c in components:
        if c[0] == "circle":
            if not (LO <= c[1] <= HI and LO <= c[2] <= HI): continue
            i, j, dx, dy = _cell(c[1], c[2]); T[0, j, i] = 1; T[1, j, i], T[2, j, i], T[3, j, i] = dx, dy, c[3] / R_SCALE; M[0, j, i] = 1
        else:
            if c[0] == "ray": a = c[3] + math.pi / 2; b = math.cos(a) * c[1] + math.sin(a) * c[2]
            else: a, b = c[1], c[2]
            if b < 0: a, b = a + math.pi, -b                                          # normale orientée vers l'extérieur de l'origine : ancre = b·n
            px, py = b * math.cos(a), b * math.sin(a)
            if not (LO <= px <= HI and LO <= py <= HI): continue
            i, j, dx, dy = _cell(px, py); T[4, j, i] = 1; T[5, j, i], T[6, j, i], T[7, j, i], T[8, j, i] = dx, dy, math.cos(a), math.sin(a); M[1, j, i] = 1
    return T, M


def loss_fn(out, T, M, pos_weight=40.0):
    pw = torch.tensor(pos_weight); obj = Fn.binary_cross_entropy_with_logits(out[:, 0], T[:, 0], pos_weight=pw) + Fn.binary_cross_entropy_with_logits(out[:, 4], T[:, 4], pos_weight=pw)
    rc = ((out[:, 1:4] - T[:, 1:4]).abs() * M[:, 0:1]).sum() / M[:, 0].sum().clamp_min(1); rl = ((out[:, 5:9] - T[:, 5:9]).abs() * M[:, 1:2]).sum() / M[:, 1].sum().clamp_min(1)
    return obj + rc + rl


def train(scenes: Sequence[tuple], epochs=30, batch=64, lr=2e-3, seed=0, width=32):
    """Entraîne le proposeur sur des scènes (X, y, composantes vraies). Renvoie (réseau, journal)."""
    torch.manual_seed(seed); net = ProposerNet(width); X = torch.stack([rasterize(x, y) for x, y, _ in scenes]); TM = [targets(c) for _, _, c in scenes]; T = torch.stack([t for t, _ in TM]); M = torch.stack([m for _, m in TM])
    opt = torch.optim.Adam(net.parameters(), lr=lr); g = torch.Generator().manual_seed(seed); t0 = time.time(); log = []
    for ep in range(epochs):
        perm = torch.randperm(len(X), generator=g); tot = 0.0
        for s in range(0, len(X), batch):
            i = perm[s:s + batch]; opt.zero_grad(); l = loss_fn(net(X[i]), T[i], M[i]); l.backward(); opt.step(); tot += float(l.detach()) * len(i)
        log.append(round(tot / len(X), 4))
    return net, {"epochs": epochs, "scenes": len(scenes), "loss": log, "seconds": round(time.time() - t0, 1), "n_params": net.n_params(), "flops_per_forward": net.flops(), "memory_bytes": net.n_params() * 4}


def decode(out, top=6, min_prob=0.15):
    """Sortie du réseau [9, G, G] → composantes proposées [('circle', cx, cy, r, p) | ('line', angle, offset, p)], avec
    suppression des voisins immédiats (un maximum local par composante)."""
    comps = []
    for ch, kind in ((0, "circle"), (4, "line")):
        p = torch.sigmoid(out[ch]); loc = (p == Fn.max_pool2d(p[None, None], 3, 1, 1)[0, 0]) & (p >= min_prob); idx = loc.nonzero(); sc = p[loc]
        for k in sc.argsort(descending=True)[:top].tolist():
            j, i = int(idx[k, 0]), int(idx[k, 1]); cx = LO + (i + 0.5 + float(out[ch + 1, j, i])) * CELL; cy = LO + (j + 0.5 + float(out[ch + 2, j, i])) * CELL
            if kind == "circle": comps.append(("circle", cx, cy, max(0.2, float(out[3, j, i]) * R_SCALE), float(sc[k])))
            else:
                a = math.atan2(float(out[8, j, i]), float(out[7, j, i])); comps.append(("line", a, math.cos(a) * cx + math.sin(a) * cy, float(sc[k])))
    return comps


def save(net: ProposerNet, name: str, info: Optional[dict] = None):
    os.makedirs(WEIGHTS, exist_ok=True); json.dump({"state": {k: v.tolist() for k, v in net.state_dict().items()}, "info": info or {}}, open(os.path.join(WEIGHTS, f"neural_proposer_{name}.json"), "w"))


def state_to_json(net: ProposerNet): return {k: v.tolist() for k, v in net.state_dict().items()}


def load(name: str = "full") -> Optional[ProposerNet]:
    p = os.path.join(WEIGHTS, f"neural_proposer_{name}.json")
    if not os.path.exists(p): return None
    d = json.load(open(p)); net = ProposerNet(); net.load_state_dict({k: torch.tensor(v) for k, v in d["state"].items()}); net.eval(); return net


class NeuralProposer(PrimitiveProposal):
    """Proposeur neuronal : observations → composantes de frontière → atomes candidats (base minimale + macros
    reconnues par contraintes). Il n'émet que des PROPOSITIONS ; rien n'est retenu sans validation WCE."""
    name = "neural"

    def __init__(self, net: Optional[ProposerNet] = None, basis_name="rich", top=6, variant="full"):
        self.net = net if net is not None else load(variant); self.basis_name, self.top = basis_name, top

    def components(self, X, y):
        if self.net is None: return []
        with torch.no_grad(): return decode(self.net(rasterize(X, y)[None])[0], self.top)

    def propose(self, X, y, pairs: Pairs, seed=0, counter: Optional[FitCounter] = None) -> List[PrimitiveCandidate]:
        from core.deep_generation import GeoPrimitive, literals_from
        prims = [GeoPrimitive("circle", (c[1], c[2], c[3]), int(100 * c[4]), torch.zeros(0, 2)) if c[0] == "circle" else GeoPrimitive("line", (c[1], c[2]), int(100 * c[3]), torch.zeros(0, 2)) for c in self.components(X, y)]
        keep = {skel_key(a) for a in basis(self.basis_name)}; return [PrimitiveCandidate(l.program, self.name) for l in literals_from(prims, macros=self.basis_name == "rich", max_sectors=0) if skel_key(l.program) in keep]

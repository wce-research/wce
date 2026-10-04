"""Certificats ANCRÉS dans la vérité observée (ObservedTruthAnchor) + collecte ACTIVE de contre-exemples.

Limite corrigée : un certificat qui ne teste que l'accord avec sa propre référence peut être auto-cohérent et faux.
Le BoundaryGenerativeCertificate vérifie (1) la cohérence avec la référence (programme) sur des sondes générées,
ET (2) l'accord avec des observations INDÉPENDANTES et fiables (ancres) : observations directes, contre-exemples
actifs (là où la règle pourrait échouer : désaccord avec des programmes alternatifs, frontière, hors du support),
expériences confirmatoires — chacune avec sa provenance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch

from core.boundary_program import BNode, ProgramPosterior, predict, render


@dataclass
class Anchor:
    x: torch.Tensor
    y: int
    provenance: str
    reliability: float = 0.97


@dataclass
class ObservedTruthAnchor:
    anchors: List[Anchor] = field(default_factory=list)

    def add(self, x, y, provenance, reliability=0.97): self.anchors.append(Anchor(x, int(y), provenance, reliability))

    def error(self, predict_fn):
        if not self.anchors: return 0.0, 0
        X = torch.stack([a.x for a in self.anchors]); y = torch.tensor([a.y for a in self.anchors])
        wrong = (predict_fn(X) != y).float(); return float(wrong.mean()), int(wrong.sum())

    def refutes(self, predict_fn, max_errors=None):
        """Réfutation si le nombre d'erreurs dépasse ce qu'explique le bruit d'étiquette (binomiale, p ≤ 0,01)."""
        n = len(self.anchors)
        if n == 0: return False
        e, k = self.error(predict_fn); eps = 1 - sum(a.reliability for a in self.anchors) / n
        if max_errors is None:
            max_errors = 0; cum = 0.0
            for j in range(n + 1):
                cum += math.comb(n, j) * eps ** j * (1 - eps) ** (n - j)
                if cum >= 0.99: max_errors = j; break
        return k > max_errors


@dataclass
class BoundaryGenerativeCertificate:
    """Certificat génératif de frontière : programme appris, postérieure, ancres de vérité, régions de désaccord,
    sondes de frontière, contre-exemples."""
    program: BNode
    generator: object
    posterior: Optional[ProgramPosterior] = None
    anchor: ObservedTruthAnchor = field(default_factory=ObservedTruthAnchor)
    counterexamples: List[Tuple[torch.Tensor, int]] = field(default_factory=list)
    disagreement_regions: Optional[torch.Tensor] = None

    def check(self, predict_fn, g, n=64):
        Xc = self.generator.sample(n, g); ref_ok = float((predict_fn(Xc) == predict(self.program, Xc)).float().mean())
        anchored = not self.anchor.refutes(predict_fn)
        return {"reference_consistency": ref_ok, "anchor_error": self.anchor.error(predict_fn)[0], "anchored_ok": anchored,
                "ok": ref_ok >= 0.95 and anchored}

    def memory_floats(self):
        return self.program.n_params() + 3 * len(self.anchor.anchors) + (self.generator.floats() if hasattr(self.generator, "floats") else 0)


def collect_counterexamples(program: BNode, oracle: Callable, g, budget=10, alternatives: Optional[List[BNode]] = None,
                            region_sampler=None, policy="active", support=None, noise=0.03):
    """« Où cette règle pourrait-elle échouer ? » — points où la règle DIVERGE de programmes alternatifs
    compatibles avec les données, puis points proches de sa frontière HORS du support observé ; on observe la
    vérité (bruitée) et on l'ajoute comme ancre. policy="random" : points uniformes (contrôle)."""
    pool = region_sampler(1500, g); anchor = ObservedTruthAnchor()
    if policy == "random":
        idx = torch.randperm(len(pool), generator=g)[:budget]
    else:
        score = torch.zeros(len(pool), dtype=torch.float64)
        if alternatives:
            P = predict(program, pool)
            for a in alternatives: score += (predict(a, pool) != P).double()
        from core.boundary_program import evaluate
        margin = (evaluate(program, pool, k=4.0) - 0.5).abs().double()              # proximité de la frontière
        far = torch.zeros(len(pool), dtype=torch.float64)
        if support is not None: far = torch.cdist(pool, support).min(1).values.double()
        score = score + 0.5 * (1 - 2 * margin) + 0.3 * far / (far.max() + 1e-9)
        idx = []; chosen = torch.zeros(len(pool), dtype=torch.bool)
        for _ in range(budget):                                                     # diversité : pas deux points trop proches
            s = score.clone(); s[chosen] = -1e9
            if idx: s -= 2.0 * (torch.cdist(pool, pool[idx]).min(1).values < 0.3).double()
            i = int(s.argmax()); idx.append(i); chosen[i] = True
        idx = torch.tensor(idx)
    for i in idx.tolist():
        y = oracle(pool[i:i + 1])
        if float(torch.rand(1, generator=g)) < noise: y = 1 - y
        anchor.add(pool[i], int(y), "contre-exemple actif" if policy != "random" else "observation aléatoire", 1 - noise)
    return anchor

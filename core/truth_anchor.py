"""Certificates ANCHORED in the observed truth (ObservedTruthAnchor) + ACTIVE collection of counterexamples.

Limitation corrected: a certificate that only tests agreement with its own reference can be self-consistent and wrong.
The BoundaryGenerativeCertificate checks (1) consistency with the reference (program) on generated probes,
AND (2) agreement with INDEPENDENT and reliable observations (anchors): direct observations, active
counterexamples (where the rule could fail: disagreement with alternative programs, boundary, outside the support),
confirmatory experiments — each with its provenance.
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
        """Refutation if the number of errors exceeds what label noise explains (binomial, p ≤ 0.01)."""
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
    """Generative boundary certificate: learned program, posterior, truth anchors, disagreement regions,
    boundary probes, counterexamples."""
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
    """"Where could this rule fail?" — points where the rule DIVERGES from alternative programs
    compatible with the data, then points close to its boundary OUTSIDE the observed support; the (noisy)
    truth is observed and added as an anchor. policy="random": uniform points (control)."""
    pool = region_sampler(1500, g); anchor = ObservedTruthAnchor()
    if policy == "random":
        idx = torch.randperm(len(pool), generator=g)[:budget]
    else:
        score = torch.zeros(len(pool), dtype=torch.float64)
        if alternatives:
            P = predict(program, pool)
            for a in alternatives: score += (predict(a, pool) != P).double()
        from core.boundary_program import evaluate
        margin = (evaluate(program, pool, k=4.0) - 0.5).abs().double()              # proximity to the boundary
        far = torch.zeros(len(pool), dtype=torch.float64)
        if support is not None: far = torch.cdist(pool, support).min(1).values.double()
        score = score + 0.5 * (1 - 2 * margin) + 0.3 * far / (far.max() + 1e-9)
        idx = []; chosen = torch.zeros(len(pool), dtype=torch.bool)
        for _ in range(budget):                                                     # diversity: no two points too close
            s = score.clone(); s[chosen] = -1e9
            if idx: s -= 2.0 * (torch.cdist(pool, pool[idx]).min(1).values < 0.3).double()
            i = int(s.argmax()); idx.append(i); chosen[i] = True
        idx = torch.tensor(idx)
    for i in idx.tolist():
        y = oracle(pool[i:i + 1])
        if float(torch.rand(1, generator=g)) < noise: y = 1 - y
        anchor.add(pool[i], int(y), "contre-exemple actif" if policy != "random" else "observation aléatoire", 1 - noise)
    return anchor

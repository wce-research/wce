"""Robust Primitive Discovery — upstream only:

    observations → primitive proposal → primitive validation → primitive parameter estimation → primitive set

Nothing here touches composition, the final score, the gate, simplification or selection.

Ideas tested (each measurable separately):
  * MINIMAL BASIS: propose only lines and circles; the rich atoms (band, ring, sector) are
    then rebuilt by composition (downstream) or by joint assembly (here, optional).
  * BOUNDARY FIRST, LOCAL → GLOBAL: local boundary elements (point + locally estimated normal), global
    merging by voting (Hough transform over lines and circles), global validation.
  * STRUCTURED MULTI-START: a small parameter grid, scored by an observable score, then refinement of the
    few best — not a massive uniform draw.
  * VALIDATION: three split modes, comparable. `points`: the POINTS are split in two, we propose and estimate
    on one half, we validate on the other (the only truly out-of-sample split). `pairs`: split of the pairs (the
    noisy points stay in both halves: not independent). `none`: all the data, threshold calibrated on a
    NULL distribution (random labels). The contrast "z" is over-dispersed (pairs sharing points):
    a nominal threshold of 3 is crossed without any boundary; the threshold therefore comes from `null_max_z`, never from a table.
  * LOCAL ESTIMATION of the parameters (`local_refit`): least squares on the boundary midpoints of a band around
    the hypothesis, instead of a global contrast fit that lets itself be attracted by the other boundaries.
  * BACKWARD ELIMINATION (`prune`): a primitive with no contribution of its own is removed.
  * MULTI-REGION CONSISTENCY and PURITY: the contrast must be present on several segments of the boundary.
  * STABILITY: the primitive must survive subsampling.
  * JOINT FIT: two concentric circles → a ring, two parallel lines → a band, fitted together.

No world truth; no import of any world module.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from core.boundary_program import BNode, predict
from core.deep_fit import DT, FitCounter
from core.deep_primitives import (EQUIV_AGREEMENT, CompatiblePrimitives, Pairs, PrimitiveCandidate, PrimitiveProposal, atom_name, close_pairs, contrast, contrast_fit,
                                  separates)

Z_OOS = 6.0                       # OUT-OF-SAMPLE contrast threshold. The pairs share points (up to k per point): the "z" is over-dispersed
                                  # and is NOT a standard normal variable; the threshold is calibrated on a null distribution (random labels).
MIN_PURITY = 0.55
MIN_SEGMENTS = 2
MIN_SUPPORT_OOS = 4
MAX_INSTABILITY = 0.12


CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "deep_configs", "robust_validator.json")
DEFAULT_CONFIG = {"split": "points", "z_oos": Z_OOS, "min_support": MIN_SUPPORT_OOS, "min_purity": MIN_PURITY, "min_segments": MIN_SEGMENTS, "prune_z": 2.0}


def load_config() -> dict:
    """Validator thresholds tuned on the TUNING scenes (never on the evaluation); default values otherwise."""
    try: return {**DEFAULT_CONFIG, **json.load(open(CONFIG_FILE))["config"]}
    except (OSError, ValueError, KeyError): return dict(DEFAULT_CONFIG)


def null_floor(split: str, default=Z_OOS) -> float:
    """Contrast floor calibrated on random labels for a split mode (tuning file)."""
    try: return float(json.load(open(CONFIG_FILE))["null_calibration"][split]["z_floor"])
    except (OSError, ValueError, KeyError): return default


def _T(c, ch): return BNode("translate", params=(float(c[0]), float(c[1])), children=(ch,))
def _wrap(a): return math.atan2(math.sin(a), math.cos(a))
def line_atom(a, b): return BNode("pred", "dot", "le", (float(_wrap(a)), float(b)))
def disc_atom(cx, cy, r): return _T((cx, cy), BNode("pred", "norm", "le", (float(abs(r)),)))


def split_pairs(pairs: Pairs, seed=0, frac=0.5) -> Tuple[Pairs, Pairs]:
    """Splits the neighbouring pairs into two disjoint sets: proposal / validation. (The two halves share
    points: the independence is that of the PAIRS, not of the points — an accepted limitation.)"""
    n = len(pairs); perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)); k = int(n * frac); a, b = perm[:k], perm[k:]
    sub = lambda i: Pairs(pairs.P[i], pairs.Q[i], pairs.opposite[i], pairs.weight[i].clone(), float(pairs.opposite[i].mean()) if len(i) else 0.0)
    return sub(a), sub(b)


def isolated_points(X, y, k=6, iso=0.8) -> torch.Tensor:
    """Points for which at least `iso` of the k nearest neighbours carry the opposite label: very probably label
    errors. Such a point generates on its own up to k opposite pairs that lie on no boundary."""
    Xd = X.to(DT); D = torch.cdist(Xd, Xd); D.fill_diagonal_(float("inf")); nb = D.topk(min(k, len(Xd) - 1), largest=False).indices; return (y[nb] != y[:, None]).to(DT).mean(1) >= iso


def clean_pairs(X, y, k=6, radius=0.6) -> Pairs:
    """Neighbouring pairs WITHOUT those that touch an isolated point. Underlying reason: the pairs of a single noisy point are
    spread over both halves (proposal / validation) and create the SAME false contrast there — out-of-sample
    validation by pairs does not protect against it."""
    X = X.to(DT); n = len(X); D = torch.cdist(X, X); D.fill_diagonal_(float("inf")); d, nb = D.topk(min(k, n - 1), largest=False); bad = isolated_points(X, y, k)
    i = torch.arange(n)[:, None].expand_as(nb); m = (d < radius) & (i < nb) & ~bad[i] & ~bad[nb]; a, b = i[m], nb[m]
    opp = (y[a] != y[b]).to(DT); return Pairs(X[a], X[b], opp, torch.ones(len(a), dtype=DT), float(opp.mean()) if len(a) else 0.0)


def prune(cp: CompatiblePrimitives, pairs: Pairs, min_z=2.0, tol=0.2) -> List[PrimitiveCandidate]:
    """BACKWARD ELIMINATION of the redundant primitives: OWN contribution of a primitive = contrast on only the
    opposite pairs that no OTHER retained primitive explains (separation or boundary within `tol`). The
    weakest one is removed as long as its own contribution is below `min_z`. Targets the "between two boundaries" fits
    and the fragments retained before the true primitive was found. Only removes."""
    removed = []; M = (pairs.P + pairs.Q) / 2
    while len(cp) > 1:
        cov = [separates(c.representative.program, pairs) | near_boundary(c.representative.program, M, tol) for c in cp.classes]; zs = []
        for i, c in enumerate(cp.classes):
            other = torch.zeros(len(pairs), dtype=torch.bool)
            for j, cv in enumerate(cov):
                if j != i: other |= cv
            w = Pairs(pairs.P, pairs.Q, pairs.opposite, torch.where(other & (pairs.opposite > 0), torch.zeros_like(pairs.weight), torch.ones_like(pairs.weight)), pairs.q0); zs.append(contrast(c.representative.program, w)["z"])
        i = min(range(len(zs)), key=lambda i: zs[i])
        if zs[i] >= min_z: break
        removed += cp.retire(i, "redondante (contribution propre insuffisante)").members
    return removed


def split_points(X, y, seed=0, frac=0.5):
    """Splits the POINTS into two disjoint halves (indices). The neighbouring pairs are then built WITHIN each
    half: no observation is shared between proposal and validation. This is the only truly
    out-of-sample split; the split by pairs (`split_pairs`) leaves each noisy point in both halves."""
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed)); k = int(len(X) * frac); return perm[:k], perm[k:]


# ================================================================ local boundary elements (local → global)

def boundary_elements(pairs: Pairs, radius=0.6, min_neighbors=4):
    """LOCAL boundary elements: for each pair of opposite labels, its midpoint and a normal estimated
    locally (principal direction of the neighbouring midpoints → tangent; normal = perpendicular). With too few
    neighbours, the normal is the direction of the pair. Returns (midpoints [M, 2], unit normals [M, 2])."""
    m = pairs.opposite * pairs.weight > 0; P, Q = pairs.P[m], pairs.Q[m]; M = (P + Q) / 2
    if len(M) == 0: return M, M
    d = Q - P; d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-9); D = torch.cdist(M, M); W = (D < radius).to(DT); cnt = W.sum(1)
    mu = (W @ M) / cnt[:, None]; X = M[None, :, :] - mu[:, None, :]; C = torch.einsum("ij,ijk,ijl->ikl", W, X, X) / cnt[:, None, None]
    ev, evec = torch.linalg.eigh(C); normal = evec[:, :, 0]                                      # smallest eigenvalue: direction normal to the trail of midpoints
    ok = (cnt >= min_neighbors) & (ev[:, 1] > 4 * ev[:, 0].clamp_min(1e-9))                        # clearly elongated trail
    return M, torch.where(ok[:, None], normal, d)


def hough_lines(M, N, top=8, da=math.pi / 24, db=0.2):
    """Global vote for lines: each element (midpoint, normal) votes for (angle, offset)."""
    if len(M) == 0: return []
    a = torch.atan2(N[:, 1], N[:, 0]); b = (N * M).sum(1); flip = a < 0; a = torch.where(flip, a + math.pi, a); b = torch.where(flip, -b, b)        # angle ∈ [0, π)
    na, nb = int(round(math.pi / da)), int(round(8.0 / db)); ia = (a / da).long().clamp(0, na - 1); ib = ((b + 4.0) / db).long().clamp(0, nb - 1)
    acc = torch.zeros(na, nb, dtype=DT).index_put_((ia, ib), torch.ones(len(M), dtype=DT), accumulate=True)
    sm = acc + 0.5 * (torch.roll(acc, 1, 1) + torch.roll(acc, -1, 1)) + 0.5 * (torch.roll(acc, 1, 0) + torch.roll(acc, -1, 0)); out = []
    for f in sm.flatten().topk(min(top * 3, sm.numel())).indices.tolist():
        i, j = divmod(f, nb)
        if float(sm[i, j]) < 3.0: break
        sel = (ia == i) & ((ib - j).abs() <= 1)
        if int(sel.sum()) < 2: continue
        aa = float(a[sel].mean()); bb = float(b[sel].mean())
        if all(abs(_wrap(aa - x)) > 0.15 or abs(bb - y) > 0.25 for x, y, _ in out): out.append((aa, bb, float(sm[i, j])))
        if len(out) >= top: break
    return out


def hough_circles(M, N, top=8, cell=0.25, radii=None):
    """Global vote for circles: the normal of an element passes through the centre; each element votes for the
    centres located at distance r along its normal, on both sides."""
    if len(M) == 0: return []
    radii = radii if radii is not None else torch.arange(0.4, 2.3, 0.15, dtype=DT); R = len(radii); nc = int(round(9.0 / cell))
    C = torch.cat([M[:, None, :] + radii[None, :, None] * N[:, None, :], M[:, None, :] - radii[None, :, None] * N[:, None, :]], 0)                  # [2M, R, 2]
    ix = ((C[..., 0] + 4.5) / cell).long(); iy = ((C[..., 1] + 4.5) / cell).long(); ok = (ix >= 0) & (ix < nc) & (iy >= 0) & (iy < nc); ir = torch.arange(R)[None, :].expand(C.shape[0], R)
    acc = torch.zeros(R, nc, nc, dtype=DT).index_put_((ir[ok], iy[ok], ix[ok]), torch.ones(int(ok.sum()), dtype=DT), accumulate=True)
    sm = acc + 0.5 * (torch.roll(acc, 1, 1) + torch.roll(acc, -1, 1) + torch.roll(acc, 1, 2) + torch.roll(acc, -1, 2)) + 0.5 * (torch.roll(acc, 1, 0) + torch.roll(acc, -1, 0)); out = []
    for f in sm.flatten().topk(min(top * 4, sm.numel())).indices.tolist():
        k, rest = divmod(f, nc * nc); j, i = divmod(rest, nc)
        if float(sm[k, j, i]) < 4.0: break
        cx, cy, r = -4.5 + (i + 0.5) * cell, -4.5 + (j + 0.5) * cell, float(radii[k])
        if all(math.hypot(cx - x, cy - y) > 0.4 or abs(r - z) > 0.3 for x, y, z, _ in out): out.append((cx, cy, r, float(sm[k, j, i])))
        if len(out) >= top: break
    return out


# ================================================================ LOCAL estimation of the parameters

def clean_midpoints(X, y, pairs: Pairs, k=6, iso=0.8):
    """Midpoints of the opposite pairs still unexplained, WITHOUT the pairs due to an isolated point (a point for which at least
    `iso` of the k nearest neighbours carry the opposite label is probably a label error: it generates
    on its own up to k opposite pairs that lie on no boundary)."""
    m = pairs.opposite * pairs.weight > 0; P, Q = pairs.P[m], pairs.Q[m]
    if len(P) == 0: return (P + Q) / 2
    Xd = X.to(DT); D = torch.cdist(Xd, Xd); D.fill_diagonal_(float("inf")); nb = D.topk(min(k, len(Xd) - 1), largest=False).indices; bad = Xd[(y[nb] != y[:, None]).to(DT).mean(1) >= iso]
    if len(bad): keep = (torch.cdist(P, bad).min(1).values > 1e-9) & (torch.cdist(Q, bad).min(1).values > 1e-9); P, Q = P[keep], Q[keep]
    return (P + Q) / 2


def local_refit(prog: BNode, M: torch.Tensor, bands=(0.4, 0.3, 0.22), min_points=5) -> BNode:
    """LOCAL and robust estimation of the parameters of a line or of a circle: least squares on only the boundary
    midpoints located in a band around the hypothesis, the band being tightened at each pass. Unlike
    the global contrast fit, the hypothesis cannot be attracted by ANOTHER boundary of the scene. If the
    band is too sparsely populated or if the result drifts away (centre > 0.8; radius outside [0.2; 3]), the hypothesis is returned
    unchanged."""
    nm = atom_name(prog)
    if len(M) < min_points or nm not in ("dot.le", "norm.le", "norm.ge"): return prog
    if nm == "dot.le":
        a, b = prog.params; n = torch.tensor([math.cos(a), math.sin(a)], dtype=DT)
        for band in bands:
            S = M[((M @ n) - b).abs() < band]
            if len(S) < min_points: return prog
            mu = S.mean(0); n2 = torch.linalg.eigh((S - mu).T @ (S - mu))[1][:, 0]; n2 = n2 if float(n2 @ n) >= 0 else -n2
            if float(n2 @ n) < 0.8: return prog
            n, b = n2, float(n2 @ mu)
        return line_atom(math.atan2(float(n[1]), float(n[0])), b)
    c0 = torch.tensor(prog.params, dtype=DT); c, r = c0.clone(), float(prog.children[0].params[0])
    for band in bands:
        S = M[((M - c).norm(dim=1) - r).abs() < band]
        if len(S) < min_points: return prog
        A = torch.cat([S, torch.ones(len(S), 1, dtype=DT)], 1); sol = torch.linalg.lstsq(A, -(S ** 2).sum(1, keepdim=True)).solution[:, 0]; c2 = -sol[:2] / 2; r2 = float((c2 @ c2 - sol[2]).clamp_min(1e-9).sqrt())
        if not (0.2 <= r2 <= 3.0) or float((c2 - c0).norm()) > 0.8 or not math.isfinite(r2): return prog
        c, r = c2, r2
    return _T((float(c[0]), float(c[1])), BNode("pred", "norm", prog.children[0].op, (r,)))


class LocalRefiner(PrimitiveProposal):
    """Wraps a proposer: each minimal-basis candidate additionally receives its LOCALLY re-estimated version. The
    original candidate is kept (the re-estimation is a proposal, never a replacement)."""
    name = "local"
    def __init__(self, inner: PrimitiveProposal): self.inner = inner; self.name = f"local({inner.name})"

    def propose(self, X, y, pairs, seed=0, counter=None):
        raw = self.inner.propose(X, y, pairs, seed, counter); M = clean_midpoints(X, y, pairs); out = list(raw)
        for c in raw:
            q = local_refit(c.program, M)
            if q is not c.program: out.append(PrimitiveCandidate(q, f"local({c.source})"))
        return out


def _nms(items, score, same, top):
    """Non-maximum suppression: best-scored distinct hypotheses (otherwise the `top` best all describe the
    most marked boundary and the other boundaries are never proposed)."""
    out = []
    for i in sorted(range(len(items)), key=lambda i: -score[i]):
        if all(not same(items[i], o) for o in out): out.append(items[i])
        if len(out) >= top: break
    return out


class RansacProposer(PrimitiveProposal):
    """LOCAL → GLOBAL by consensus, minimal basis: hypotheses of lines (2 midpoints) and of circles (3 midpoints) drawn
    half within a NEIGHBOURHOOD (the three points of a same arc) and half globally, scored by contrast,
    deduplicated (non-maxima), then re-estimated locally."""
    name = "ransac"
    def __init__(self, n_hyp=800, top=10, radius=1.3): self.n_hyp, self.top, self.radius = n_hyp, top, radius

    def propose(self, X, y, pairs, seed=0, counter=None):
        g = torch.Generator().manual_seed(seed); M = clean_midpoints(X, y, pairs); m = len(M); out = []
        if m < 6: return out
        K = self.n_hyp; D = torch.cdist(M, M); W = ((D < self.radius) & (D > 0.25)).to(DT) + 1e-6; i = torch.randint(0, m, (K,), generator=g)
        loc = torch.rand(K, generator=g) < 0.5; j = torch.where(loc, torch.multinomial(W[i], 1, generator=g)[:, 0], torch.randint(0, m, (K,), generator=g)); k = torch.where(loc, torch.multinomial(W[i], 1, generator=g)[:, 0], torch.randint(0, m, (K,), generator=g))
        P, Q = pairs.P, pairs.Q; gw = pairs.weight * pairs.opposite - pairs.q0 * torch.where(pairs.opposite > 0, pairs.weight, torch.ones_like(pairs.weight)); A, B, C = M[i], M[j], M[k]
        # lines through 2 points
        d = B - A; ok = d.norm(dim=1) > 0.3; n = torch.stack([-d[:, 1], d[:, 0]], 1)[ok]; n = n / n.norm(dim=1, keepdim=True); b = (n * A[ok]).sum(1)
        if len(n):
            sc = ((((P @ n.T - b) * (Q @ n.T - b)) < 0).to(DT) * gw[:, None]).sum(0); L = [(math.atan2(float(n[t, 1]), float(n[t, 0])), float(b[t])) for t in range(len(n))]
            same = lambda u, v: (abs(_wrap(u[0] - v[0])) < 0.2 and abs(u[1] - v[1]) < 0.3) or (abs(abs(_wrap(u[0] - v[0])) - math.pi) < 0.2 and abs(u[1] + v[1]) < 0.3)
            out += [PrimitiveCandidate(line_atom(a_, b_), self.name) for a_, b_ in _nms(L, sc.tolist(), same, self.top)]
        # circles through 3 points
        a2, b2, c2 = (A ** 2).sum(1), (B ** 2).sum(1), (C ** 2).sum(1); det = 2 * (A[:, 0] * (B[:, 1] - C[:, 1]) + B[:, 0] * (C[:, 1] - A[:, 1]) + C[:, 0] * (A[:, 1] - B[:, 1])); ok = det.abs() > 0.05; det = torch.where(ok, det, torch.ones_like(det))
        cx = (a2 * (B[:, 1] - C[:, 1]) + b2 * (C[:, 1] - A[:, 1]) + c2 * (A[:, 1] - B[:, 1])) / det; cy = (a2 * (C[:, 0] - B[:, 0]) + b2 * (A[:, 0] - C[:, 0]) + c2 * (B[:, 0] - A[:, 0])) / det; ctr = torch.stack([cx, cy], 1); r = (ctr - A).norm(dim=1)
        ok = ok & (r > 0.3) & (r < 2.6) & (ctr.abs().max(1).values < 5.0); ctr, r = ctr[ok], r[ok]
        if len(r):
            sc = ((((torch.cdist(P, ctr) - r) * (torch.cdist(Q, ctr) - r)) < 0).to(DT) * gw[:, None]).sum(0); Cs = [(float(ctr[t, 0]), float(ctr[t, 1]), float(r[t])) for t in range(len(r))]
            out += [PrimitiveCandidate(disc_atom(*c_), self.name) for c_ in _nms(Cs, sc.tolist(), lambda u, v: math.hypot(u[0] - v[0], u[1] - v[1]) < 0.3 and abs(u[2] - v[2]) < 0.25, self.top)]
        return out


class HoughProposer(PrimitiveProposal):
    """BOUNDARY FIRST, LOCAL → GLOBAL, minimal basis: local elements → votes → lines and circles → refinement by
    contrast (batched). Proposes only half-planes and discs (and their complements, by construction)."""
    name = "hough"
    def __init__(self, top=8, refine_steps=40): self.top, self.refine_steps = top, refine_steps

    def propose(self, X, y, pairs, seed=0, counter=None):
        M, N = boundary_elements(pairs); out = []
        ls = [line_atom(a, b) for a, b, _ in hough_lines(M, N, self.top)]; cs = [disc_atom(cx, cy, r) for cx, cy, r, _ in hough_circles(M, N, self.top)]
        for group in (ls, cs):
            if not group: continue
            sk = group[0].skeleton(); init = torch.tensor([p.param_vector() for p in group], dtype=DT)
            if self.refine_steps: th, _ = contrast_fit(sk, pairs, init, steps=self.refine_steps, counter=counter)
            else: th = init
            out += [PrimitiveCandidate(sk.with_params(r.tolist()), self.name) for r in th] + [PrimitiveCandidate(p, self.name + "_raw") for p in group]
        return out


class GridProposer(PrimitiveProposal):
    """STRUCTURED MULTI-START, minimal basis: a small grid (156 lines, 486 circles) SCORED by the contrast without
    any fitting, then refinement of the `top` best distinct hypotheses."""
    name = "grid"
    def __init__(self, top=8, refine_steps=40): self.top, self.refine_steps = top, refine_steps

    def propose(self, X, y, pairs, seed=0, counter=None):
        P, Q = pairs.P, pairs.Q; g = pairs.weight * pairs.opposite - pairs.q0 * torch.where(pairs.opposite > 0, pairs.weight, torch.ones_like(pairs.weight)); out = []
        ang = torch.arange(12, dtype=DT) * math.pi / 12; off = torch.arange(-3.0, 3.01, 0.5, dtype=DT); A, B = torch.meshgrid(ang, off, indexing="ij"); A, B = A.flatten(), B.flatten(); nrm = torch.stack([torch.cos(A), torch.sin(A)], 1)
        sl = ((((P @ nrm.T - B) * (Q @ nrm.T - B)) < 0).to(DT) * g[:, None]).sum(0)
        cx = torch.arange(-3.0, 1.51, 0.5, dtype=DT); cy = torch.arange(-2.5, 2.51, 0.5, dtype=DT); rr = torch.arange(0.5, 2.01, 0.3, dtype=DT); CX, CY, RR = torch.meshgrid(cx, cy, rr, indexing="ij"); C = torch.stack([CX.flatten(), CY.flatten()], 1); RR = RR.flatten()
        sc = ((((torch.cdist(P, C) - RR) * (torch.cdist(Q, C) - RR)) < 0).to(DT) * g[:, None]).sum(0)
        ls = [line_atom(float(A[i]), float(B[i])) for i in sl.topk(min(self.top, len(sl))).indices.tolist()]; cs = [disc_atom(float(C[i, 0]), float(C[i, 1]), float(RR[i])) for i in sc.topk(min(self.top, len(sc))).indices.tolist()]
        for group in (ls, cs):
            sk = group[0].skeleton(); init = torch.tensor([p.param_vector() for p in group], dtype=DT); th, _ = contrast_fit(sk, pairs, init, steps=self.refine_steps, counter=counter) if self.refine_steps else (init, None)
            out += [PrimitiveCandidate(sk.with_params(r.tolist()), self.name) for r in th]
        return out


# ================================================================ redesigned validation

def boundary_segments(prog: BNode, pairs: Pairs, n_seg=6):
    """MULTI-REGION consistency: the pairs separated by the primitive are distributed into segments along its boundary
    (by angle around the barycentre of the separated midpoints). Returns the number of segments where the contrast is positive."""
    s = separates(prog, pairs)
    if int(s.sum()) < 3: return 0, 0
    M = ((pairs.P + pairs.Q) / 2)[s]; opp = pairs.opposite[s]; c = M.mean(0); ang = torch.atan2(M[:, 1] - c[1], M[:, 0] - c[0])
    proj = ang if float((M - c).norm(dim=1).std()) < 0.6 * float((M - c).norm(dim=1).mean()) else (M - c) @ torch.linalg.eigh(((M - c).T @ (M - c)))[1][:, 1]      # ring of points: angle; trail: abscissa
    lo, hi = float(proj.min()), float(proj.max()); b = ((proj - lo) / max(1e-9, hi - lo) * n_seg).long().clamp(0, n_seg - 1); pos = used = 0
    for k in range(n_seg):
        m = b == k
        if int(m.sum()) >= 2: used += 1; pos += float(opp[m].mean()) > pairs.q0 + 0.1
    return pos, used


def stability(prog: BNode, pairs: Pairs, X, n_boot=3, seed=0, steps=25):
    """STABILITY under subsampling: contrast refit on 70% of the pairs; mean functional disagreement."""
    g = torch.Generator().manual_seed(seed); sk = prog.skeleton(); th0 = torch.tensor([prog.param_vector()], dtype=DT); base = predict(prog, X.float()); dis = []
    for _ in range(n_boot):
        i = torch.randperm(len(pairs), generator=g)[: int(0.7 * len(pairs))]; sub = Pairs(pairs.P[i], pairs.Q[i], pairs.opposite[i], pairs.weight[i], pairs.q0); th, _ = contrast_fit(sk, sub, th0, steps=steps)
        dis.append(float((predict(sk.with_params(th[0].tolist()), X.float()) != base).float().mean()))
    return sum(dis) / len(dis)


@dataclass
class Evidence:
    """Observable evidence for a candidate primitive, all of it computable without truth."""
    z_in: float                    # contrast on the PROPOSAL pairs (biased by the selection)
    z_oos: float                   # contrast on the VALIDATION pairs (unbiased)
    support_oos: float
    purity_oos: float
    segments: int
    segments_used: int
    instability: Optional[float] = None


class RobustValidator:
    """Redesigned validator. Criteria, all observable and separately switchable: out-of-sample contrast, support,
    purity, multi-region consistency, stability. Out-of-sample validation removes the FITTING bias, not the
    MULTIPLICITY bias (the best of N candidates is retained on the same validation half): the multiplicity penalty
    therefore remains applied in both modes."""

    def __init__(self, z_oos=Z_OOS, min_purity=MIN_PURITY, min_segments=MIN_SEGMENTS, min_support=MIN_SUPPORT_OOS, max_instability=None, oos=True):
        self.z_oos, self.min_purity, self.min_segments, self.min_support, self.max_instability, self.oos = z_oos, min_purity, min_segments, min_support, max_instability, oos

    def evidence(self, prog: BNode, prop: Pairs, val: Pairs, X=None, seed=0) -> Evidence:
        ci = contrast(prog, prop); co = contrast(prog, val); seg, used = boundary_segments(prog, val)
        return Evidence(ci["z"], co["z"], co["opposite_separated"], co["purity"], seg, used, stability(prog, prop, X, seed=seed) if (self.max_instability is not None and X is not None) else None)

    def accept(self, e: Evidence, n_candidates=1):
        """Returns (accepted, reason)."""
        z = e.z_oos if self.oos else e.z_in; thr = max(self.z_oos, math.sqrt(2 * math.log(max(2, n_candidates))) + 1.0)
        if e.support_oos < self.min_support: return False, "support insuffisant"
        if z < thr: return False, "contraste insuffisant"
        if e.purity_oos < self.min_purity: return False, "pureté insuffisante"
        if e.segments < self.min_segments: return False, "contraste trop local"
        if self.max_instability is not None and e.instability is not None and e.instability > self.max_instability: return False, "instable"
        return True, "retenue"


def null_max_z(proposer: PrimitiveProposal, n=300, trials=12, seed=0, box=((-3.0, 1.5), (-2.5, 2.5)), rates=(0.5, 0.2), split="points") -> List[float]:
    """TRUTH-FREE CALIBRATION: largest out-of-sample contrast obtained by the proposer on RANDOM
    labels (no boundary exists). A validation threshold must exceed these values."""
    out = []; v = RobustValidator(z_oos=0.0, min_purity=0.0, min_segments=0, min_support=0)
    for t in range(trials):
        g = torch.Generator().manual_seed(seed * 1000 + t); X = torch.rand(n, 2, generator=g); X = torch.stack([box[0][0] + X[:, 0] * (box[0][1] - box[0][0]), box[1][0] + X[:, 1] * (box[1][1] - box[1][0])], 1); y = (torch.rand(n, generator=g) < rates[t % len(rates)]).long()
        if split == "none": A = B = clean_pairs(X, y); XA, yA = X, y
        else: ia, ib = split_points(X, y, t); A, B = clean_pairs(X[ia], y[ia]), clean_pairs(X[ib], y[ib]); XA, yA = X[ia], y[ia]
        zs = [v.evidence(c.program, A, B).z_oos for c in proposer.propose(XA, yA, A, t)]; out.append(max(zs, default=0.0))
    return out


# ================================================================ joint fit (rich atoms from the minimal basis)

def joint_candidates(prims: Sequence[BNode], tol_center=0.45, tol_angle=0.25, sectors=True) -> List[BNode]:
    """Candidate rich atoms by ASSEMBLY of minimal primitives: two nearly concentric discs → ring;
    two nearly parallel half-planes → band; two intersecting half-planes → sectors at the intersection point."""
    out = []; discs = [p for p in prims if atom_name(p) in ("norm.le", "norm.ge")]; lines = [p for p in prims if atom_name(p) == "dot.le"]
    for i, a in enumerate(discs):
        for b in discs[i + 1:]:
            ra, rb = a.children[0].params[0], b.children[0].params[0]
            if math.hypot(a.params[0] - b.params[0], a.params[1] - b.params[1]) < tol_center and abs(ra - rb) > 0.3:
                c = ((a.params[0] + b.params[0]) / 2, (a.params[1] + b.params[1]) / 2); out.append(_T(c, BNode("pred", "norm", "in", (min(ra, rb), max(ra, rb)))))
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            a1, b1 = a.params; a2, b2 = b.params; d = _wrap(a1 - a2)
            if abs(d) > math.pi / 2: a2, b2 = _wrap(a2 + math.pi), -b2; d = _wrap(a1 - a2)
            if abs(d) < tol_angle and 0.3 < abs(b1 - b2) < 2.6: out.append(BNode("pred", "dot", "in", (a1, min(b1, b2), max(b1, b2))))
            elif sectors and abs(math.sin(a1 - a2)) > 0.35:                                     # intersecting lines → 4 possible sectors at the apex
                det = math.sin(a2 - a1); cx = (b1 * math.sin(a2) - b2 * math.sin(a1)) / det; cy = (b2 * math.cos(a1) - b1 * math.cos(a2)) / det
                if abs(cx) > 4.0 or abs(cy) > 4.0: continue
                for s1 in (0.0, math.pi):
                    for s2 in (0.0, math.pi):
                        t1, t2 = a1 + math.pi / 2 + s1, a2 + math.pi / 2 + s2; w = _wrap(t2 - t1) / 2
                        if 0.2 < abs(w) < 1.4: out.append(_T((cx, cy), BNode("pred", "angle", "in", (_wrap(t1 + w), abs(w)))))
    return out


def sector_exclusive(sector: BNode, parts: Sequence[BNode], pairs: Pairs, margin=0.15) -> bool:
    """A sector is preferred to its two lines only if the EXTENSION of the lines beyond the apex carries no
    boundary: the purity of the pairs separated by the lines but not by the sector must stay close to the base rate."""
    s = separates(sector, pairs); o = torch.zeros_like(s)
    for p in parts: o |= separates(p, pairs)
    m = o & ~s
    return int(m.sum()) >= 4 and float(pairs.opposite[m].mean()) < pairs.q0 + margin


def joint_fit(rich: Sequence[BNode], pairs: Pairs, steps=40, counter: Optional[FitCounter] = None) -> List[BNode]:
    """JOINT fit of the tied parameters (common centre of the ring, common direction of the band) by contrast."""
    out = []; by: Dict[str, List[BNode]] = {}
    for p in rich: by.setdefault(repr(p.skeleton()), []).append(p)
    for ps in by.values():
        sk = ps[0].skeleton(); th, _ = contrast_fit(sk, pairs, torch.tensor([p.param_vector() for p in ps], dtype=DT), steps=steps, counter=counter); out += [sk.with_params(r.tolist()) for r in th]
    return out


def near_boundary(prog: BNode, M: torch.Tensor, delta=0.2) -> torch.Tensor:
    """Points within (approximately) `delta` of the boundary of the primitive."""
    P = M.float(); near = torch.zeros(len(M), dtype=torch.bool)
    if delta <= 0: return near
    for dx, dy in ((delta, 0.0), (0.0, delta), (delta * 0.7, delta * 0.7), (delta * 0.7, -delta * 0.7)):
        o = torch.tensor([dx, dy]); near |= predict(prog, P + o) != predict(prog, P - o)
    return near


def is_kin(a: BNode, b: BNode, tol_center=0.4, tol_angle=0.2) -> bool:
    """Geometric kinship: concentric circles of distinct radii, or distinct parallel lines."""
    na, nb = atom_name(a), atom_name(b)
    if na in ("norm.le", "norm.ge") and nb in ("norm.le", "norm.ge"): return math.hypot(a.params[0] - b.params[0], a.params[1] - b.params[1]) < tol_center and abs(a.children[0].params[0] - b.children[0].params[0]) > 0.3
    if na == nb == "dot.le":
        d = _wrap(a.params[0] - b.params[0]); b2 = b.params[1] if abs(d) < math.pi / 2 else -b.params[1]; d = d if abs(d) < math.pi / 2 else _wrap(d + math.pi); return abs(d) < tol_angle and 0.3 < abs(a.params[1] - b2) < 2.6
    return False


# ================================================================ robust discovery

@dataclass
class RobustDiscovery:
    primitives: CompatiblePrimitives
    candidates: List[dict]                       # ALL the candidates examined: program, evidence, decision, reason (audit)
    proposals: int
    seconds: float
    fit_equivalents: float
    rounds: int
    joint: List[BNode] = field(default_factory=list)


def discover_robust(X, y, proposer: PrimitiveProposal, validator: Optional[RobustValidator] = None, rounds=3, keep_per_round=4, seed=0, max_classes=12, joint=False, folds=2, explain_tol=0.2, kin_z=2.5, clean=True, prune_z=2.0, split="points") -> RobustDiscovery:
    """Iterative discovery. The POINTS are split into two halves; the neighbouring pairs are built within each one.
    Propose (and estimate) on the "proposal" half, validate on the "validation" half, retain,
    remove from BOTH halves the explained pairs, start again. With `folds=2`, the roles of the two halves are then
    swapped (no pair is used both to propose and to validate the SAME candidate). With `joint=True`, the retained
    minimal primitives are additionally assembled into rich atoms (ring, band, sector), fitted jointly and
    validated like the others."""
    t0 = time.time(); validator = validator or RobustValidator(); mk = clean_pairs if clean else close_pairs; allp = mk(X, y); cp = CompatiblePrimitives(X); counter = FitCounter(); log = []; n = 0; r = 0; total = 0
    if split == "points": ia, ib = split_points(X, y, seed); XA, yA, XB, yB = X[ia], y[ia], X[ib], y[ib]; A, B = mk(XA, yA), mk(XB, yB)
    elif split == "none": A = B = allp; XA = XB = X; yA = yB = y; folds = 1                          # all the data, IN-sample validation; only a threshold calibrated on a null distribution protects
    else: A, B = split_pairs(allp, seed); XA = XB = X; yA = yB = y                                  # ablation: split by pairs (shared points)
    obs = {id(A): (XA, yA), id(B): (XB, yB)}
    def explain(prog):
        for pr in (A, B): s = separates(prog, pr) | near_boundary(prog, (pr.P + pr.Q) / 2, explain_tol); pr.weight = torch.where(s & (pr.opposite > 0), torch.zeros_like(pr.weight), pr.weight)
    for fold, (prop, val) in enumerate(((A, B), (B, A))[:folds]):
        for r in range(rounds):
            total += 1; cands = proposer.propose(*obs[id(prop)], prop, seed + r + 10 * fold, counter); n += len(cands); scored = []
            for c in cands:
                e = validator.evidence(c.program, prop, val, X, seed); ok, why = validator.accept(e, len(cands)); c.z, c.support, c.purity, c.validated, c.reason = e.z_oos, e.support_oos, e.purity_oos, ok, why
                scored.append((c, e)); log.append({"program": c.program, "source": c.source, "round": total - 1, "z_in": e.z_in, "z_oos": e.z_oos, "support": e.support_oos, "purity": e.purity_oos, "segments": e.segments, "instability": e.instability, "accepted": ok, "reason": why, "kept": False})
            kept = 0
            for c, e in sorted((s for s in scored if s[0].validated), key=lambda s: -s[1].z_oos):
                if kept >= keep_per_round or len(cp) >= max_classes: break
                e2 = validator.evidence(c.program, prop, val, None, seed); ok, _ = validator.accept(e2, len(cands))                 # re-measurement: the pairs already explained in this round no longer count
                if not ok: continue
                c.z = e2.z_oos; status, _ = cp.add(c)
                for l in log:
                    if l["program"] is c.program: l["kept"] = True
                if status == "new": kept += 1; explain(c.program)
            if kept == 0: break
    r = total - 1
    if kin_z is not None and len(cp):
        # GEOMETRIC CONSISTENCY: a circle concentric with a retained primitive, or a parallel line, is a hypothesis
        # far less free than an arbitrary candidate (a single free parameter: radius or offset). A weaker
        # out-of-sample contrast (kin_z) is then enough — it is the inner edge of a ring, the second edge of a band.
        for l in sorted((l for l in log if not l["accepted"] and l["z_oos"] >= kin_z and l["support"] >= 6 and l["purity"] >= 0.6 and l["reason"] in ("contraste insuffisant", "support insuffisant")), key=lambda l: -l["z_oos"]):
            if len(cp) >= max_classes: break
            if not any(is_kin(l["program"], c.representative.program) for c in cp.classes): continue
            c = PrimitiveCandidate(l["program"], l["source"] + "+kin", z=l["z_oos"], support=l["support"], purity=l["purity"], validated=True, reason="retenue (parenté géométrique)")
            status, _ = cp.add(c); l["accepted"] = l["kept"] = True; l["reason"] = c.reason
            if status == "new": explain(c.program)
    if prune_z is not None:
        gone = {id(m.program) for m in prune(cp, allp, prune_z, explain_tol)}
        for l in log:
            if id(l["program"]) in gone: l["kept"] = False; l["reason"] = "redondante (contribution propre insuffisante)"
    jt = []
    if joint and len(cp):
        if split == "none": prop2 = val2 = mk(X, y)
        elif split == "points": prop2, val2 = mk(XA, yA), mk(XB, yB)          # the rich atom is judged on ALL the validation pairs, not on the residual
        else: prop2, val2 = split_pairs(allp, seed)
        mins = [c.representative.program for c in cp.classes]; lines = [p for p in mins if atom_name(p) == "dot.le"]
        for p in joint_fit(joint_candidates(mins), prop2, counter=counter):
            e = validator.evidence(p, prop2, val2, None, seed); ok, why = validator.accept(e, 1)
            if ok and atom_name(p) == "angle.in" and not sector_exclusive(p, lines, val2): ok, why = False, "secteur non exclusif (les droites se prolongent)"
            log.append({"program": p, "source": "joint", "round": r + 1, "z_in": e.z_in, "z_oos": e.z_oos, "support": e.support_oos, "purity": e.purity_oos, "segments": e.segments, "instability": None, "accepted": ok, "reason": why, "kept": ok})
            if ok: jt.append(p)
    return RobustDiscovery(cp, log, n, round(time.time() - t0, 3), counter.fit_equivalents, r + 1, jt)


def literals_of(d: RobustDiscovery, alternatives=1, with_joint=True, max_literals=None) -> List[BNode]:
    """Literals for the composition: class representatives (and, if requested, type alternatives) + assembled rich
    atoms. The complements (anti-disc, opposite half-plane) are added: the minimal basis is closed under complement."""
    out = list(d.primitives.literals(max_literals, alternatives)) + (list(d.joint) if with_joint else []); extra = []
    for p in out:
        nm = atom_name(p)
        if nm == "norm.le": extra.append(_T(p.params, BNode("pred", "norm", "ge", p.children[0].params)))
        elif nm == "norm.ge": extra.append(_T(p.params, BNode("pred", "norm", "le", p.children[0].params)))
        elif nm == "dot.le": extra.append(line_atom(p.params[0] + math.pi, -p.params[1]))
    return out + extra


class UnionProposer(PrimitiveProposal):
    """Union of COMPLEMENTARY proposers (no additional refinement)."""
    name = "union"
    def __init__(self, proposers: Sequence[PrimitiveProposal]): self.proposers = list(proposers)
    def propose(self, X, y, pairs, seed=0, counter=None): return [c for p in self.proposers for c in p.propose(X, y, pairs, seed, counter)]


# ================================================================ compatible hypotheses (do not choose too early)

@dataclass
class PrimitiveHypothesisGroup:
    """COMPETING explanations of a same portion of boundary: for example {ring} versus {disc, anti-disc}."""
    alternatives: List[Tuple[Tuple[BNode, ...], float]]            # (programs of the explanation, normalised weight)

    @property
    def ambiguous(self): return len(self.alternatives) > 1


class CompatiblePrimitiveHypotheses:
    """Weighted set of primitive hypotheses. Each assembled rich atom stays in competition with the minimal
    primitives it comes from; each class keeps its type alternatives. Nothing is decided here: the composition
    (downstream) or new measurements discriminate."""

    def __init__(self, d: RobustDiscovery, X):
        Xf = X.float(); reps = [c.representative for c in d.primitives.classes]; used = set(); self.groups: List[PrimitiveHypothesisGroup] = []
        zj = {id(l["program"]): l["z_oos"] for l in d.candidates if l["source"] == "joint"}
        for j in d.joint:
            parts = []
            for i, r in enumerate(reps):
                if _is_part(r.program, j): parts.append(i)
            if not parts: self.groups.append(PrimitiveHypothesisGroup([((j,), 1.0)])); continue
            used.update(parts); wj = max(1e-9, zj.get(id(j), 0.0)); wp = max(1e-9, sum(reps[i].z for i in parts) / len(parts)); t = wj + wp
            self.groups.append(PrimitiveHypothesisGroup([((j,), wj / t), (tuple(reps[i].program for i in parts), wp / t)]))
        for i, c in enumerate(d.primitives.classes):
            if i in used: continue
            alts = {}
            for m in sorted(c.members, key=lambda m: -m.z): alts.setdefault(m.type, m)
            t = sum(max(1e-9, m.z) for m in alts.values()); self.groups.append(PrimitiveHypothesisGroup([((m.program,), max(1e-9, m.z) / t) for m in alts.values()]))

    def __len__(self): return len(self.groups)
    def ambiguous(self): return [g for g in self.groups if g.ambiguous]
    def programs(self) -> List[BNode]: return [p for g in self.groups for ps, _ in g.alternatives for p in ps]


def _is_part(minimal: BNode, rich: BNode, tol=0.35) -> bool:
    """Is the minimal primitive one of the boundaries of the rich atom?"""
    n, r = atom_name(minimal), atom_name(rich)
    if r == "norm.in" and n in ("norm.le", "norm.ge"): return math.hypot(minimal.params[0] - rich.params[0], minimal.params[1] - rich.params[1]) < tol + 0.2 and min(abs(minimal.children[0].params[0] - q) for q in rich.children[0].params) < tol
    if n != "dot.le": return False
    a, b = minimal.params
    if r == "dot.in":
        d = _wrap(a - rich.params[0]); return (abs(d) < 0.3 and min(abs(b - q) for q in rich.params[1:]) < tol) or (abs(abs(d) - math.pi) < 0.3 and min(abs(-b - q) for q in rich.params[1:]) < tol)
    if r == "angle.in": return abs(math.cos(a) * rich.params[0] + math.sin(a) * rich.params[1] - b) < tol
    return False


# ================================================================ active measurements specific to primitives

def borderline(d: RobustDiscovery, lo=1.5) -> List[BNode]:
    """UNDECIDED candidates: rejected for insufficient contrast or support but not null (out-of-sample z ≥ lo, or high
    proposal contrast not confirmed). They are the ones that an additional measurement can settle."""
    return [l["program"] for l in d.candidates if not l["accepted"] and l["source"] != "joint" and (lo <= l["z_oos"] or l["z_in"] >= 4.0) and l["reason"] in ("contraste insuffisant", "support insuffisant", "contraste trop local")]


def boundary_query_scores(programs: Sequence[BNode], pool: torch.Tensor, delta=0.15, max_programs=24) -> torch.Tensor:
    """Score of a measurement point: number of undecided candidates whose boundary passes within `delta`."""
    sc = torch.zeros(len(pool)); P = pool.float()
    for p in list(programs)[:max_programs]:
        near = torch.zeros(len(pool), dtype=torch.bool)
        for dx, dy in ((delta, 0.0), (0.0, delta)):
            o = torch.tensor([dx, dy]); near |= predict(p, P + o) != predict(p, P - o)
        sc += near.float()
    return sc

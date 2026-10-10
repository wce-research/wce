"""High-Precision Primitive Estimation — making each primitive precise enough for composition, without an oracle.

Tuning finding (keys 1090–1099, see the report): the LOCAL single-primitive objectives (contrast of
neighboring pairs, floor on a spatial grid) are not aligned with the function — starting from the true
primitive, they move away from it. The aligned objective is JOINT and structure-free: the per-cell floor,

    floor(set) = Σ_cells min(positives, negatives),   cell = vector of the values of the atoms

that is, the smallest number of errors attainable by ANY Boolean function of these atoms. This is
exactly what composition over equivalence classes can exploit; it is independent of polarity.

    pool (recall first) → compatible sets
        → joint refinement of the parameters on the floor          (local, bounded, initial point kept)
        → birth: missing primitive searched for in parameter space by floor gain,
                 guided by the morphology of the residual
        → conditional pruning → validation by repeated subsampling → commit / rollback
        → same strong downstream as the other chains (sets → composition over classes → validated finalization)

Objects:
  PrimitiveEstimate        central estimate, uncertainty, compatible region, stability, provenance, identifiability
  ResidualMorphology       shape of the residual: linear, arc, parallel pair, concentric pair, radial
  SetImprovement           log of a set: floor and validated error before / after, births, decision
  HPRecovery               result of the complete chain

Every transformation follows the Validated Transformation Principle: propose → evaluate on a valid criterion (out-of-sample
error of the floor, repeated subsampling) → commit or rollback. No world truth; no oracle.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from core.boundary_program import BNode
from core.deep_fit import DT, compile_skeleton

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "deep_configs", "hp.json")
DEFAULTS = {"k": 5, "sources": ("flat", "components"), "curves": True, "lines": 12, "circles": 10, "band_factor": 0.8, "cos_normal": 0.5, "joint": True, "n_refine_sets": 3, "sweeps": 2, "samples": 48, "scale": 0.08, "margin": 1.5, "births": 2, "birth_samples": 1500, "birth_top": 4, "birth_gain": 3.0,
            "birth_per_param": 1.4, "prune_gain": 1.5, "cv_repeats": 24, "cv_fraction": 0.2, "validate": "holdout", "holdout_repeats": 3, "holdout_fraction": 0.3, "holdout_z": 1.64, "refine": True, "morphology": True, "describe": False}
IDENT = ("IDENTIFIABLE", "PARTIALLY_IDENTIFIABLE", "UNIDENTIFIABLE")
IDENT_TOL = 0.10                 # half-width of the compatible region below which a parameter is said to be identified (fixed a priori)
MORPHOLOGIES = ("linear", "arc", "parallel", "concentric", "radial", "none")
BIRTH_TYPES = ("half", "band", "disc", "ring", "sector")          # the anti-disc is the complement of the disc: same cell, same floor
MORPH_TYPES = {"linear": ("half", "sector", "band"), "arc": ("disc", "ring"), "parallel": ("band", "half"), "concentric": ("ring", "disc"), "radial": ("sector", "half"), "none": BIRTH_TYPES}
PI = math.pi
_hard = lambda z: (z >= 0).to(DT)


def load_config() -> dict:
    """Settings chosen on the TUNING tasks only; default values otherwise."""
    try:
        with open(CONFIG_FILE) as fh: return {**DEFAULTS, **json.load(fh)["config"]}
    except (OSError, ValueError, KeyError): return dict(DEFAULTS)


# ================================================================ parameter families

def strip(p: BNode) -> BNode: return p.children[0] if p.kind == "not" else p


def families(p: BNode) -> List[str]:
    """Geometric family of each parameter of the vector of `p`: center, radius, orientation, offset, direction,
    half-aperture. Tolerances, uncertainties and the sensitivity map are reported per family."""
    p = strip(p); l = p.children[0] if p.kind == "translate" else p; pre = ["center", "center"] if p.kind == "translate" else []
    if l.feat == "dot": return pre + ["orientation"] + ["offset"] * (len(l.params) - 1)
    if l.feat == "norm": return pre + ["radius"] * len(l.params)
    return pre + ["direction", "half_width"]


def type_name(p: BNode) -> str:
    from core.deep_fragments import type_of
    return type_of(strip(p))


# ================================================================ specialized estimators: boundary curves

def spacing(X: torch.Tensor) -> float:
    """Median spacing of the observations: sets the localization scale of a boundary (no scene constant)."""
    D = torch.cdist(X.to(DT), X.to(DT)); D.fill_diagonal_(float("inf")); return float(D.min(1).values.median())


def _sequential(inl: torch.Tensor, top: int, min_inliers: int, refit) -> list:
    """SEQUENTIAL consensus ("who explains what"): the best hypothesis is re-estimated on its elements, kept,
    then its elements are removed from the count of the following ones. A short curve is therefore not drowned under the
    variants of a long curve."""
    alive = torch.ones(inl.shape[1], dtype=torch.bool); out = []
    for _ in range(top):
        sc = (inl & alive[None]).sum(1); k = int(sc.argmax())
        if int(sc[k]) < min_inliers: break
        r = refit(k)
        if r is None: inl[k] = False; continue
        comp, idx = r; out.append(comp); alive[idx] = False; alive[inl[k]] = False
    return out


def extract_lines(ob, top=12, seed=0, band_factor=0.6, cos_normal=0.8, min_inliers=4, n_pairs=1500) -> list:
    """Half-plane — orientation then offset. Votes: each boundary element proposes ITS line (local normal),
    plus lines through two neighboring elements. An element supports a line only if it is close to it AND if its
    local normal agrees (this discards crossings). Re-estimation by total least squares, sequential
    consensus. The band follows the spacing of the observations."""
    from core.deep_partial import Component, line_tls
    M, N = ob.points, ob.normals; m = len(M)
    if m < min_inliers: return []
    g = torch.Generator().manual_seed(seed); band = max(0.03, band_factor * spacing(ob.X)); i = torch.randint(0, m, (n_pairs,), generator=g); j = torch.randint(0, m, (n_pairs,), generator=g); d = M[j] - M[i]; ok = (d.norm(dim=1) > 0.3) & (d.norm(dim=1) < 2.0)
    n2 = torch.stack([-d[:, 1], d[:, 0]], 1)[ok]; n2 = n2 / n2.norm(dim=1, keepdim=True).clamp_min(1e-9); Cn = torch.cat([N, n2]); Cb = torch.cat([(N * M).sum(1), (n2 * M[i][ok]).sum(1)])
    support = lambda n, b: (((M @ n) - b).abs() < band) & ((N @ n).abs() > cos_normal); inl = ((Cn @ M.T - Cb[:, None]).abs() < band) & ((Cn @ N.T).abs() > cos_normal)
    def refit(k):
        n, b = Cn[k], float(Cb[k]); idx = torch.nonzero(inl[k])[:, 0]; rms = float("inf")
        for _ in range(2):
            if len(idx) < min_inliers: return None
            n1, b1, rms = line_tls(M[idx])
            if float(n1 @ n) < 0: n1, b1 = -n1, -b1
            n, b = n1, float(b1); idx = torch.nonzero(support(n, b))[:, 0]
        return (Component("line", (math.atan2(float(n[1]), float(n[0])), b), idx, rms), idx) if len(idx) >= min_inliers else None
    return _sequential(inl, top, min_inliers, refit)


def extract_circles(ob, top=10, seed=0, band_factor=0.6, cos_normal=0.8, min_inliers=5, n_triplets=3000, reach=1.6, r_range=(0.2, 3.5)) -> list:
    """Disc / anti-disc — center then radius. Circles through three elements of the same neighborhood (an arc); an element
    supports a circle only if it is on the band AND if its local normal is RADIAL. Taubin then geometric
    re-estimation, sequential consensus."""
    from core.deep_partial import Component, circle_geometric, circle_taubin
    M, N = ob.points, ob.normals; m = len(M)
    if m < min_inliers: return []
    g = torch.Generator().manual_seed(seed + 1); band = max(0.03, band_factor * spacing(ob.X)); D = torch.cdist(M, M); W = ((D < reach) & (D > 0.2)).to(DT) + 1e-9; i = torch.randint(0, m, (n_triplets,), generator=g); j = torch.multinomial(W[i], 1, generator=g)[:, 0]; k = torch.multinomial(W[i], 1, generator=g)[:, 0]
    A, B, C = M[i], M[j], M[k]; a2, b2, c2 = (A ** 2).sum(1), (B ** 2).sum(1), (C ** 2).sum(1); det = 2 * (A[:, 0] * (B[:, 1] - C[:, 1]) + B[:, 0] * (C[:, 1] - A[:, 1]) + C[:, 0] * (A[:, 1] - B[:, 1])); ok = det.abs() > 0.02; det = torch.where(ok, det, torch.ones_like(det))
    ctr = torch.stack([(a2 * (B[:, 1] - C[:, 1]) + b2 * (C[:, 1] - A[:, 1]) + c2 * (A[:, 1] - B[:, 1])) / det, (a2 * (C[:, 0] - B[:, 0]) + b2 * (A[:, 0] - C[:, 0]) + c2 * (B[:, 0] - A[:, 0])) / det], 1); r = (ctr - A).norm(dim=1); ok = ok & (r > r_range[0]) & (r < r_range[1]); ctr, r = ctr[ok], r[ok]
    if not len(r): return []
    def support(c, rr):
        V = M[None] - c[:, None]; dist = V.norm(dim=2); return ((dist - rr[:, None]).abs() < band) & (((V * N[None]).sum(2).abs() / dist.clamp_min(1e-9)) > cos_normal)
    inl = support(ctr, r)
    def refit(q):
        c, rr = ctr[q], float(r[q]); idx = torch.nonzero(inl[q])[:, 0]; rms = float("inf")
        for _ in range(2):
            if len(idx) < min_inliers: return None
            fit = circle_taubin(M[idx])
            if fit is None: return None
            c, rr, rms = circle_geometric(M[idx], fit[0], fit[1])
            if not (r_range[0] <= float(rr) <= r_range[1]): return None
            idx = torch.nonzero(support(torch.as_tensor(c, dtype=DT)[None], torch.tensor([float(rr)], dtype=DT))[0])[:, 0]
        return (Component("circle", (float(c[0]), float(c[1]), float(rr)), idx, float(rms)), idx) if len(idx) >= min_inliers else None
    return _sequential(inl, top, min_inliers, refit)


def assemble(ob, lines: list, circles: list, k=5, joint=True) -> List[Tuple[BNode, float, str]]:
    """Curves → primitives, by FAMILY, the rich primitives being constrained compositions of simple curves:
      half-plane  one line                                disc / anti-disc       one circle (both polarities)
      band        two parallel lines, shared orientation (`joint`) or independent lines
      ring        two concentric circles, shared center (`joint`) or independent circles
      sector      two rays with the same vertex: the vertex is the intersection, and each line must carry
                  elements on the side of ITS ray (this is what distinguishes a sector from two intersecting half-planes)
    At most `k` hypotheses per compound type, ranked by occlusion-tolerant score."""
    from core.deep_fragments import SKELETONS
    from core.deep_partial import _wrap, build, concentric_fit, orient, parallel_fit, visible_score
    M = ob.points; out: Dict[str, list] = {t: [] for t in SKELETONS}
    for c in lines: out["half"].append(orient(build(SKELETONS["half"], None, c.params), ob.X, ob.y))
    for c in circles: out["disc"].append(build(SKELETONS["disc"], c.params[:2], (c.params[2],))); out["antidisc"].append(build(SKELETONS["antidisc"], c.params[:2], (c.params[2],)))
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            d = _wrap(a.params[0] - b.params[0]); a2, b2 = (b.params[0], b.params[1]) if abs(d) < PI / 2 else (_wrap(b.params[0] + PI), -b.params[1])
            if abs(_wrap(a.params[0] - a2)) < 0.3 and 0.3 < abs(a.params[1] - b2) < 2.8:
                if joint: n, b1, bb = parallel_fit(M[a.inliers], M[b.inliers]); out["band"].append(build(SKELETONS["band"], None, (math.atan2(float(n[1]), float(n[0])), min(b1, bb), max(b1, bb))))
                else: out["band"].append(build(SKELETONS["band"], None, (a.params[0], min(a.params[1], b2), max(a.params[1], b2))))
            det = math.sin(b.params[0] - a.params[0])
            if abs(det) < 0.3: continue
            a1, o1, a3, o3 = a.params[0], a.params[1], b.params[0], b.params[1]; v = torch.tensor([(o1 * math.sin(a3) - o3 * math.sin(a1)) / det, (o3 * math.cos(a1) - o1 * math.cos(a3)) / det], dtype=DT)
            if float(v.abs().max()) > 4.5: continue
            rays = []
            for c in (a, b):
                tg = torch.tensor([-math.sin(c.params[0]), math.cos(c.params[0])], dtype=DT); pr = (M[c.inliers] - v) @ tg
                for sgn in (1.0, -1.0):
                    if int((sgn * pr > 0.15).sum()) >= 3: rays.append(math.atan2(sgn * float(tg[1]), sgn * float(tg[0])))
                rays.append(None)
            ra = [r for r in rays[:rays.index(None)]]; rb = [r for r in rays[rays.index(None) + 1:] if r is not None]
            for t1 in ra:
                for t2 in rb:
                    w = _wrap(t2 - t1) / 2
                    if 0.2 < abs(w) < 1.45: out["sector"].append(build(SKELETONS["sector"], v.tolist(), (_wrap(t1 + w), abs(w))))
    for i, a in enumerate(circles):
        for b in circles[i + 1:]:
            if math.hypot(a.params[0] - b.params[0], a.params[1] - b.params[1]) < 0.6 and abs(a.params[2] - b.params[2]) > 0.3:
                fit = concentric_fit(M[a.inliers], M[b.inliers]) if joint else None
                if fit is not None: out["ring"].append(build(SKELETONS["ring"], fit[0].tolist(), sorted(fit[1:])))
                else: out["ring"].append(build(SKELETONS["ring"], ((a.params[0] + b.params[0]) / 2, (a.params[1] + b.params[1]) / 2), sorted((a.params[2], b.params[2]))))
    res = []
    for t, progs in out.items():
        sc = sorted(((visible_score(p, ob.pairs)[0], j, p) for j, p in enumerate(progs)), key=lambda z: -z[0])
        if t in ("band", "ring", "sector"): sc = sc[:k]
        res += [(p, z, "curves:" + t) for z, _, p in sc]
    n = len(res); order = sorted(range(n), key=lambda i: -res[i][1]); rank = {i: 1.0 - r / max(1, n) for r, i in enumerate(order)}; return [(res[i][0], rank[i], res[i][2]) for i in range(n)]


def curve_pool(X: torch.Tensor, y: torch.Tensor, k=5, seed=0, cfg: Optional[dict] = None) -> List[Tuple[BNode, float, str]]:
    """Pool of the specialized estimators: lines and circles by votes with normals and sequential consensus, then
    assembly by family. Compact budget: at most `lines` lines and `circles` circles."""
    from core.deep_partial import observe
    c = {**load_config(), **(cfg or {})}; ob = observe(X, y); return assemble(ob, extract_lines(ob, c["lines"], seed, c["band_factor"], c["cos_normal"]), extract_circles(ob, c["circles"], seed, c["band_factor"], c["cos_normal"]), k, c["joint"])


# ================================================================ per-cell floor (vectorized)

def evaluate_batch(skel: BNode, TH: torch.Tensor, Xd: torch.Tensor) -> torch.Tensor:
    """[K, n] 0/1 values of K parameterizations of the same skeleton."""
    return compile_skeleton(skel)(TH, Xd, _hard)


def bits_of(p: BNode, Xd: torch.Tensor) -> torch.Tensor:
    p = strip(p); return evaluate_batch(p.skeleton(), torch.tensor([p.param_vector()], dtype=DT), Xd)[0].long()


def cell_code(bits: Sequence[torch.Tensor], n: int, skip: Optional[int] = None) -> torch.Tensor:
    o = torch.zeros(n, dtype=torch.long)
    for j, b in enumerate(bits):
        if j != skip: o = o * 2 + b
    return o


def floor_batch(B: torch.Tensor, others: torch.Tensor, y: torch.Tensor, ncell: int) -> torch.Tensor:
    """Floor of K candidates [K, n] added to a context of cells `others` [n]: Σ min(positives, negatives)."""
    K, n = B.shape; cell = others[None] * 2 + B.long(); yy = y[None].to(DT).expand(K, n)
    pos = torch.zeros(K, ncell * 2, dtype=DT).scatter_add_(1, cell, yy); tot = torch.zeros(K, ncell * 2, dtype=DT).scatter_add_(1, cell, torch.ones(K, n, dtype=DT)); return torch.minimum(pos, tot - pos).sum(1)


def set_floor(bits: Sequence[torch.Tensor], y: torch.Tensor) -> float:
    n = len(y); return float(floor_batch(torch.zeros(1, n, dtype=DT), cell_code(bits, n), y, 2 ** len(bits))[0])


def floor_prediction(bits: Sequence[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    """Majority label of the cell of each observation: the best Boolean function of these atoms."""
    n = len(y); c = cell_code(bits, n); nc = 2 ** len(bits); pos = torch.zeros(nc, dtype=DT).scatter_add_(0, c, y.to(DT)); tot = torch.zeros(nc, dtype=DT).scatter_add_(0, c, torch.ones(n, dtype=DT)); return (2 * pos > tot)[c].long()


def cv_floor(bits: Sequence[torch.Tensor], y: torch.Tensor, repeats=24, fraction=0.2, seed=0) -> float:
    """VALID criterion (self-consistency): OUT-OF-SAMPLE error rate of the floor, by repeated subsampling —
    the majority label of each cell is learned on 1 − `fraction` of the observations and judged on the rest.
    More powerful than a single split into two halves; the same draws are used before and after a transformation."""
    n = len(y); g = torch.Generator().manual_seed(seed); c = cell_code(bits, n); nc = 2 ** len(bits); test = torch.rand(repeats, n, generator=g) < fraction; tr = (~test).to(DT); yy = y.to(DT)[None].expand(repeats, n); cc = c[None].expand(repeats, n)
    pos = torch.zeros(repeats, nc, dtype=DT).scatter_add_(1, cc, yy * tr); tot = torch.zeros(repeats, nc, dtype=DT).scatter_add_(1, cc, tr); gmaj = ((yy * tr).sum(1, keepdim=True) * 2 > tr.sum(1, keepdim=True)).to(DT)
    lab = torch.where(2 * pos == tot, gmaj.expand(repeats, nc), (2 * pos > tot).to(DT)); pred = lab.gather(1, cc); return float(((pred != yy) & test).sum() / test.sum().clamp_min(1))


# ================================================================ joint refinement, local and bounded

def _climb(skel: BNode, th: torch.Tensor, others: torch.Tensor, Xd, y, ncell: int, g, iters: int, K: int, scale: float, margin=0.0) -> Tuple[torch.Tensor, float, int]:
    """Gradient-free local climb: K Gaussian perturbations per step, decreasing scale; the current point remains
    a candidate (never any degradation). A perturbation is taken only if it gains more than `margin` errors."""
    f = float(floor_batch(evaluate_batch(skel, th[None], Xd), others, y, ncell)[0]); ev = 1
    for it in range(iters):
        C = th[None] + torch.randn(K, len(th), generator=g, dtype=DT) * scale * (0.7 ** it); v = floor_batch(evaluate_batch(skel, C, Xd), others, y, ncell); i = int(v.argmin()); ev += K
        if float(v[i]) < f - margin: f = float(v[i]); th = C[i]
    return th, f, ev


def joint_refine(programs: Sequence[BNode], X: torch.Tensor, y: torch.Tensor, seed=0, sweeps=2, samples=48, scale=0.08, margin=1.5) -> Tuple[List[BNode], dict]:
    """JOINT refinement of a set: each primitive is adjusted in turn, the others fixed, on the
    per-cell floor. Local (scale `scale`), bounded (`sweeps` passes), initial point kept."""
    P = [strip(p) for p in programs]; n = len(y); Xd = X.to(DT)[None]; g = torch.Generator().manual_seed(seed); bits = [bits_of(p, Xd) for p in P]; f0 = set_floor(bits, y); ev = 0; moved = 0
    for sw in range(sweeps):
        for i, p in enumerate(P):
            sk = p.skeleton(); th0 = torch.tensor(p.param_vector(), dtype=DT); th, f, e = _climb(sk, th0, cell_code(bits, n, i), Xd, y, 2 ** (len(P) - 1), g, 4, samples, scale * (0.6 ** sw), margin); ev += e
            if not torch.equal(th, th0): P[i] = sk.with_params(th.tolist()); bits[i] = bits_of(P[i], Xd); moved += 1
    return P, {"floor_before": f0, "floor_after": set_floor(bits, y), "evaluations": ev, "moves": moved}


# ================================================================ morphology of the residual

@dataclass
class ResidualMorphology:
    """Shape of what the current set does not explain. Guides the FAMILY of primitive to propose."""
    label: str                                  # linear | arc | parallel | concentric | radial | none
    n_errors: int
    points: torch.Tensor                        # midpoints of the boundaries of the residual
    support: Dict[str, int] = field(default_factory=dict)

    @property
    def types(self) -> Tuple[str, ...]: return MORPH_TYPES[self.label]


def residual_morphology(X: torch.Tensor, residual: torch.Tensor, seed=0) -> ResidualMorphology:
    """structured residual → shape. The residual is the label "the set is wrong here"; its boundaries are those
    of the missing primitive. Consensus lines and circles are searched for in it; two parallel lines → parallel
    pair, two circles with the same center → concentric pair, two lines intersecting within the domain → radial."""
    from core.deep_partial import _wrap, circle_components, line_components, observe
    ne = int(residual.sum())
    if ne < 6 or ne > len(residual) - 6: return ResidualMorphology("none", ne, torch.zeros(0, 2, dtype=DT))
    ob = observe(X, residual.long())
    if len(ob.points) < 4: return ResidualMorphology("none", ne, ob.points)
    L = line_components(ob, seed=seed)[:3]; C = circle_components(ob, seed=seed)[:3]; nl = len(L[0].inliers) if L else 0; nc = len(C[0].inliers) if C else 0; sup = {"line": nl, "circle": nc}; lab = "linear" if nl >= nc else "arc"
    if max(nl, nc) < 4: return ResidualMorphology("none", ne, ob.points, sup)
    if lab == "linear" and len(L) > 1 and len(L[1].inliers) >= 4:
        d = abs(_wrap(L[0].params[0] - L[1].params[0])); d = min(d, PI - d); lab = "parallel" if d < 0.3 else "radial" if d > 0.5 else lab
    if lab == "arc" and len(C) > 1 and len(C[1].inliers) >= 4 and math.hypot(C[0].params[0] - C[1].params[0], C[0].params[1] - C[1].params[1]) < 0.6 and abs(C[0].params[2] - C[1].params[2]) > 0.3: lab = "concentric"
    return ResidualMorphology(lab, ne, ob.points, sup)


# ================================================================ birth of a primitive by floor gain

def sample_params(t: str, K: int, M: torch.Tensor, g, box=(-3.0, 3.0, -3.0, 3.0)) -> torch.Tensor:
    """K parameterizations of type `t` whose boundary PASSES through a point of `M` (boundary midpoints). Generic
    ranges, tied to the size of the domain only."""
    U = lambda lo, hi: torch.rand(K, generator=g, dtype=DT) * (hi - lo) + lo; coin = lambda: (torch.rand(K, generator=g) < 0.5).to(DT)
    c = M[torch.randint(0, len(M), (K,), generator=g)] if len(M) else torch.stack([U(box[0], box[1]), U(box[2], box[3])], 1); a = U(-PI, PI)
    if t == "half": return torch.stack([a, torch.cos(a) * c[:, 0] + torch.sin(a) * c[:, 1]], 1)
    if t == "band": b = torch.cos(a) * c[:, 0] + torch.sin(a) * c[:, 1]; w = U(0.3, 2.4); s = coin(); return torch.stack([a, b - s * w, b + (1 - s) * w], 1)
    if t in ("disc", "antidisc"): r = U(0.3, 2.5); return torch.stack([c[:, 0] + r * torch.cos(a), c[:, 1] + r * torch.sin(a), r], 1)
    if t == "ring": r = U(0.3, 2.5); w = U(0.3, 1.5); s = coin(); return torch.stack([c[:, 0] + r * torch.cos(a), c[:, 1] + r * torch.sin(a), (r - s * w).clamp_min(0.15), r + (1 - s) * w], 1)
    l = U(0.0, 2.5); w = U(0.2, 1.5); s = 2 * coin() - 1; return torch.stack([c[:, 0] - l * torch.cos(a), c[:, 1] - l * torch.sin(a), a + s * w, w], 1)        # sector: one ray from the vertex passes through the point


def birth(programs: Sequence[BNode], X: torch.Tensor, y: torch.Tensor, scene_points: Optional[torch.Tensor] = None, seed=0, samples=1500, top=4, gain=3.0, per_param=1.4, morphology=True) -> Tuple[Optional[BNode], dict]:
    """structured residual → missing primitive hypothesis.

    The missing primitive is searched for IN PARAMETER SPACE, by the floor gain it brings TO
    THE CURRENT SET (conditional utility), and not by its isolated score. The candidates pass through the boundaries
    of the residual (and, for half of them, through those of the scene); the morphology of the residual splits the budget between families.
    Accepted only if the gain exceeds `gain` + `per_param` errors per parameter."""
    from core.deep_fragments import SKELETONS
    P = [strip(p) for p in programs]; n = len(y); Xd = X.to(DT)[None]; g = torch.Generator().manual_seed(seed); bits = [bits_of(p, Xd) for p in P]; others = cell_code(bits, n); nc = 2 ** len(bits); f0 = set_floor(bits, y)
    mo = residual_morphology(X, floor_prediction(bits, y) != y, seed); M = mo.points if scene_points is None or len(scene_points) == 0 else torch.cat([mo.points, scene_points]) if len(mo.points) else scene_points
    info = {"morphology": mo.label, "residual": mo.n_errors, "floor_before": f0, "evaluations": 0, "accepted": False}; best = None
    for t in BIRTH_TYPES:
        K = samples * (2 if morphology and t in mo.types[:2] else 1) // (1 if morphology else 1); sk = SKELETONS[t]; C = sample_params(t, K, M, g); v = floor_batch(evaluate_batch(sk, C, Xd), others, y, nc); info["evaluations"] += K
        for i in v.argsort()[:top].tolist():
            th, f, e = _climb(sk, C[i], others, Xd, y, nc, g, 6, 48, 0.12); info["evaluations"] += e; need = gain + per_param * len(th)
            if best is None or (f0 - f) - need > best[0]: best = ((f0 - f) - need, f, t, th)
    if best is None or best[0] < 0: return None, info
    info.update({"accepted": True, "type": best[2], "gain": f0 - best[1], "floor_after": best[1]}); return SKELETONS[best[2]].with_params(best[3].tolist()), info


def prune(programs: Sequence[BNode], X: torch.Tensor, y: torch.Tensor, gain=1.5) -> List[BNode]:
    """CONDITIONAL pruning: a primitive is removed only if the set WITHOUT it loses almost nothing. A primitive
    weak in isolation but useful in interaction (operand of an exclusive or) is therefore kept."""
    P = list(programs); Xd = X.to(DT)[None]; bits = [bits_of(p, Xd) for p in P]
    while len(P) > 1:
        f = set_floor(bits, y); loss = [set_floor(bits[:i] + bits[i + 1:], y) - f for i in range(len(P))]; i = min(range(len(P)), key=lambda j: loss[j])
        if loss[i] >= gain: break
        P.pop(i); bits.pop(i)
    return P


# ================================================================ uncertainty, stability, identifiability

@dataclass
class PrimitiveEstimate:
    """A candidate primitive is never a bare point: composition can know that a parameter is uncertain."""
    program: BNode
    type: str
    families: List[str]
    central: List[float]
    uncertainty: List[float]                    # standard deviation of the compatible parameterizations, per parameter
    region: List[Tuple[float, float]]           # compatible interval, per parameter
    stability: float                            # mean functional agreement of the re-estimations on resamplings
    provenance: str
    identifiability: str
    consensus: Optional[BNode] = None           # median of the re-estimations (bootstrap)

    def by_family(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for f, u in zip(self.families, self.uncertainty): out[f] = max(out.get(f, 0.0), u)
        return out


def describe(programs: Sequence[BNode], X: torch.Tensor, y: torch.Tensor, seed=0, slack=2.0, samples=96, boots=6, provenance: Optional[Sequence[str]] = None, tol=IDENT_TOL) -> List[PrimitiveEstimate]:
    """For each primitive of a set, the rest fixed:
      compatible region   parameterizations whose floor does not exceed the best by more than `slack` errors
      uncertainty         standard deviation of these parameterizations, per parameter
      stability           re-estimation on `boots` resamplings of the observations; mean functional agreement
      consensus           median of the re-estimations
      identifiability     all parameters identified (half-width ≤ tol) / some / none."""
    P = [strip(p) for p in programs]; n = len(y); Xd = X.to(DT)[None]; g = torch.Generator().manual_seed(seed); bits = [bits_of(p, Xd) for p in P]; out = []
    for i, p in enumerate(P):
        sk = p.skeleton(); th = torch.tensor(p.param_vector(), dtype=DT); others = cell_code(bits, n, i); nc = 2 ** (len(P) - 1); f0 = float(floor_batch(bits[i][None].to(DT), others, y, nc)[0])
        C = torch.cat([th[None]] + [th[None] + torch.randn(samples, len(th), generator=g, dtype=DT) * s for s in (0.03, 0.1, 0.3)]); ok = floor_batch(evaluate_batch(sk, C, Xd), others, y, nc) <= f0 + slack; R = C[ok]
        lo, hi = R.min(0).values, R.max(0).values; sd = R.std(0) if len(R) > 1 else torch.zeros(len(th), dtype=DT); half = (hi - lo) / 2; idn = int((half <= tol).sum()); ident = IDENT[0] if idn == len(th) else IDENT[1] if idn > 0 else IDENT[2]
        est = []
        for b in range(boots):
            idx = torch.randint(0, n, (n,), generator=g); t2, _, _ = _climb(sk, th, others[idx], Xd[:, idx], y[idx], nc, g, 4, 32, 0.08); est.append(t2)
        E = torch.stack(est) if est else th[None]; med = E.median(0).values; stab = float((evaluate_batch(sk, E, Xd) == bits[i][None]).to(DT).mean()) if est else 1.0
        out.append(PrimitiveEstimate(p, type_name(p), families(p), th.tolist(), sd.tolist(), list(zip(lo.tolist(), hi.tolist())), stab, provenance[i] if provenance else "set", ident, sk.with_params(med.tolist())))
    return out


def consensus(programs: Sequence[BNode], X: torch.Tensor, y: torch.Tensor, seed=0, boots=8, validate=True) -> Tuple[List[BNode], dict]:
    """Consensus instead of a single fit: each primitive is replaced by the MEDIAN of its re-estimations on
    resamplings. Validated: the median is kept only if the out-of-sample error of the floor does not rise."""
    E = describe(programs, X, y, seed, boots=boots); P = [strip(p) for p in programs]; Q = [e.consensus for e in E]; Xd = X.to(DT)[None]; a = cv_floor([bits_of(p, Xd) for p in P], y, seed=seed); b = cv_floor([bits_of(p, Xd) for p in Q], y, seed=seed); ok = (b <= a) or not validate
    return (Q if ok else P), {"cv_before": a, "cv_after": b, "committed": bool(ok)}


# ================================================================ validated improvement of a set

@dataclass
class SetImprovement:
    programs: List[BNode]
    provenance: List[str]
    floor_before: float
    floor_after: float
    cv_before: float
    cv_after: float
    committed: bool
    births: List[dict]
    evaluations: int


def transfer_errors(programs: Sequence[BNode], Xa, ya, Xb, yb) -> torch.Tensor:
    """Errors, point by point on (Xb, yb), of the best Boolean function of the atoms learned on (Xa, ya):
    majority label of each cell, learned on one side, judged on the other (unknown cell → global majority)."""
    if not len(yb): return torch.zeros(0, dtype=torch.bool)
    A = [bits_of(p, Xa.to(DT)[None]) for p in programs]; B = [bits_of(p, Xb.to(DT)[None]) for p in programs]; nc = 2 ** len(programs); ca = cell_code(A, len(ya)); cb = cell_code(B, len(yb))
    pos = torch.zeros(nc, dtype=DT).scatter_add_(0, ca, ya.to(DT)); tot = torch.zeros(nc, dtype=DT).scatter_add_(0, ca, torch.ones(len(ya), dtype=DT)); lab = torch.where(2 * pos == tot, torch.full((nc,), float(2 * ya.sum() > len(ya)), dtype=DT), (2 * pos > tot).to(DT))
    return lab[cb] != yb.to(DT)


def transfer_error(programs: Sequence[BNode], Xa, ya, Xb, yb) -> float:
    e = transfer_errors(programs, Xa, ya, Xb, yb); return float(e.to(DT).mean()) if len(e) else 0.0


def _improve(P0: Sequence[BNode], X, y, seed, c, scene_points, max_atoms):
    P = list(P0); prov = ["refined"] * len(P); ev = 0; births = []
    if c["refine"] and P: P, i = joint_refine(P, X, y, seed, c["sweeps"], c["samples"], c["scale"], c["margin"]); ev += i["evaluations"]
    for b in range(c["births"]):
        if len(P) >= max_atoms: break
        new, info = birth(P, X, y, scene_points, seed + 17 * (b + 1), c["birth_samples"], c["birth_top"], c["birth_gain"], c["birth_per_param"], c["morphology"]); ev += info["evaluations"]; births.append(info)
        if new is None: break
        P.append(new); prov.append("birth:" + info["type"])
        if c["refine"]: P, i = joint_refine(P, X, y, seed + b + 1, 1, c["samples"], c["scale"] / 2, c["margin"]); ev += i["evaluations"]
    Q = prune(P, X, y, c["prune_gain"]) if P else P; ids = [id(p) for p in P]; return Q, [prov[ids.index(id(q))] for q in Q], births, ev


def improve_set(programs: Sequence[BNode], X: torch.Tensor, y: torch.Tensor, seed=0, cfg: Optional[dict] = None, scene_points: Optional[torch.Tensor] = None, max_atoms=6) -> SetImprovement:
    """primitive set → joint refinement → [structured residual → birth → refinement]* → pruning → validation.

    Bounded loop (at most `births` births). Validation (Validated Transformation Principle), one of:
      "holdout"  the entire PROCEDURE is replayed on training subsamples (`holdout_repeats` splits) and
                 judged on the observations it has not seen; it is applied to all the observations only if
                 it corrects significantly more errors there than it creates (paired test, z ≥ `holdout_z`);
      "cv"       out-of-sample error of the floor by repeated subsampling, same draws before and after
                 (optimistic: the parameters have seen all the observations);
      False      none.
    Without a validated gain, the INITIAL set is returned as is (rollback)."""
    c = {**load_config(), **(cfg or {})}; P0 = [strip(p) for p in programs]; Xd = X.to(DT)[None]; b0 = [bits_of(p, Xd) for p in P0]; f0 = set_floor(b0, y); n = len(y); ev = 0; mode = c["validate"]
    if mode == "holdout":
        g = torch.Generator().manual_seed(seed + 991); before, after = [], []; fixed = 0; broken = 0
        for r in range(c["holdout_repeats"]):
            te = torch.rand(n, generator=g) < c["holdout_fraction"]; tr = ~te; Q, _, _, e = _improve(P0, X[tr], y[tr], seed + r, c, scene_points, max_atoms); ev += e
            e0 = transfer_errors(P0, X[tr], y[tr], X[te], y[te]); e1 = transfer_errors(Q, X[tr], y[tr], X[te], y[te]); before.append(float(e0.to(DT).mean())); after.append(float(e1.to(DT).mean())); fixed += int((e0 & ~e1).sum()); broken += int((~e0 & e1).sum())
        cv0, cv1 = sum(before) / len(before), sum(after) / len(after)
        if not (fixed - broken) >= c["holdout_z"] * math.sqrt(max(1, fixed + broken)): return SetImprovement(P0, ["initial"] * len(P0), f0, f0, cv0, cv1, False, [], ev)
    P, prov, births, e = _improve(P0, X, y, seed, c, scene_points, max_atoms); ev += e; b1 = [bits_of(p, Xd) for p in P]; f1 = set_floor(b1, y) if P else f0
    if mode != "holdout": cv0 = cv_floor(b0, y, c["cv_repeats"], c["cv_fraction"], seed); cv1 = cv_floor(b1, y, c["cv_repeats"], c["cv_fraction"], seed) if P else cv0
    ok = mode == "holdout" or not mode or cv1 < cv0
    return SetImprovement(P if ok else P0, prov if ok else ["initial"] * len(P0), f0, f1, cv0, cv1, bool(ok), births, ev)


# ================================================================ complete chain

def hp_pool(X: torch.Tensor, y: torch.Tensor, seed=0, cfg: Optional[dict] = None) -> List[Tuple[BNode, float, str]]:
    """Recall-first pool: existing estimators (several parameterizations per type) AND specialized estimators.
    No elimination; the downstream decides."""
    from core.deep_lpr import pool
    c = {**load_config(), **(cfg or {})}; return (pool(X, y, c["k"], seed, tuple(c["sources"])) if c["sources"] else []) + (curve_pool(X, y, c["k"], seed, c) if c["curves"] else [])


@dataclass
class HPRecovery:
    recovery: object                            # LatentRecovery of the strong downstream
    improvements: List[SetImprovement]
    estimates: List[PrimitiveEstimate]
    added: List[Tuple[BNode, float, str]]
    stats: Dict[str, float]

    @property
    def program(self): return self.recovery.program


def recover(X: torch.Tensor, y: torch.Tensor, seed=0, cfg: Optional[dict] = None, discovered: Optional[Sequence[Tuple[BNode, float, str]]] = None) -> HPRecovery:
    """observations → high-precision primitives → set → composition over classes → validated finalization.

    The refined and born primitives JOIN the pool (the originals remain); the strong downstream — the same as for
    the other chains — chooses. Nothing is destroyed."""
    from core.deep_e2e import hypotheses_from
    from core.deep_lpr import load_config as lpr_config, pool, select_sets, strong_downstream
    from core.deep_partial import observe
    c = {**load_config(), **(cfg or {})}; d = lpr_config(); t0 = time.time(); disc = list(discovered) if discovered is not None else hp_pool(X, y, seed, c); H = hypotheses_from(disc, X)
    sets = select_sets(H, y, d["max_atoms"], d["beam"], d["pair_pool"], d["min_gain"], d["n_sets"], d["pairs"]); M = observe(X, y).points; imps = []; added: List[Tuple[BNode, float, str]] = []
    for j, s in enumerate(sets.sets[:c["n_refine_sets"]]):
        im = improve_set(s.programs, X, y, seed + j, c, M, d["max_atoms"]); imps.append(im)
        if im.committed: added += [(p, 2.0 - 0.1 * j, "hp:" + src) for p, src in zip(im.programs, im.provenance)]
    if not sets.sets and c["births"] > 0:                                          # empty pool: the birth starts from nothing
        im = improve_set([], X, y, seed, c, M, d["max_atoms"]); imps.append(im); added += [(p, 2.0, "hp:" + src) for p, src in zip(im.programs, im.provenance)]
    R = strong_downstream(disc + added, X, y, {"fit": d["fit"]}); est = describe(R.chosen.programs, X, y, seed, provenance=R.chosen.provenance) if c["describe"] and R.chosen else []
    return HPRecovery(R, imps, est, added, {"pool": len(disc), "added": len(added), "committed_sets": sum(i.committed for i in imps), "births": sum(b.get("accepted", False) for i in imps for b in i.births), "evaluations": sum(i.evaluations for i in imps), "seconds": round(time.time() - t0, 3)})

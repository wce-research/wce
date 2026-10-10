"""Partial-Observation Primitive Parameter Inference — infer the latent parameters of a primitive from a
PARTIALLY observable boundary.

    observed boundary (fragments) → complete latent primitive → set of compatible parameters → ranked hypotheses

Objects:
  ObservedBoundary   what is seen: midpoints of opposite pairs, local normals, connected fragments
  LatentBoundary     a COMPLETE primitive hypothesis: its rendered boundary, the observed part, the completed part
  ParameterRegion    parameters compatible with the fragment (intervals, weighted hypotheses, identifiability)
  Hypothesis         ranked candidate primitive (never a single imposed point estimate)
  PrimitiveNode / CompositeNode   bounded, instrumented exchange between primitives and composition

Principles:
  * the absence of a boundary is not a proof of the absence of a primitive: the "visible" score only counts the
    segments where a contrast is expected AND observed; silent segments do not penalise;
  * under a composition context, the part of a primitive expected to be visible is computed (`visible_mask`) and the
    fitting criterion is the counterfactual rendering of the whole composite on the observations;
  * the shared constraints (common centre, parallelism, common vertex) are used explicitly.

No world truth; no import of any world or evaluation module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from core.boundary_program import BNode, predict
from core.deep_fit import DT
from core.deep_primitives import Pairs, atom_name, contrast, separates
from core.deep_primitives2 import RansacProposer, _nms, _wrap, boundary_elements, clean_pairs, near_boundary
from core.deep_structure import _leaf

IDENT = ("IDENTIFIABLE", "PARTIALLY_IDENTIFIABLE", "UNIDENTIFIABLE")
SIGMA = 0.10                      # localisation standard deviation of a pair midpoint (observation spacing ≈ 0.3)
TOL = {"center": 0.35, "radius": 0.30, "angle": 0.22, "offset": 0.30}        # identifiability tolerances (those of the equivalence metric)
PAD = {"cx": SIGMA, "cy": SIGMA, "r": 0.05, "angle": 0.025, "offset": SIGMA}       # half grid step of the profile, or localisation noise: an interval is never a point
COMBINATORS = ("and", "or", "xor")


# ================================================================ observed boundary

@dataclass
class ObservedBoundary:
    """What is OBSERVED of one or several boundaries: nothing in it is completed."""
    points: torch.Tensor            # [m, 2] midpoints of opposite pairs (pairs due to an isolated point removed)
    normals: torch.Tensor           # [m, 2] local unit normals
    fragment: torch.Tensor          # [m] index of the connected fragment
    pairs: Pairs
    X: torch.Tensor
    y: torch.Tensor

    @property
    def n_fragments(self): return int(self.fragment.max()) + 1 if len(self.fragment) else 0
    def of(self, i): return self.points[self.fragment == i]
    def extent(self, i): S = self.of(i); return float(torch.cdist(S, S).max()) if len(S) > 1 else 0.0


def link_fragments(M: torch.Tensor, link=0.45) -> torch.Tensor:
    """Connected components (single linkage at distance < link) by label propagation."""
    m = len(M)
    if m == 0: return torch.zeros(0, dtype=torch.long)
    A = torch.cdist(M, M) < link; lab = torch.arange(m)
    for _ in range(m):
        new = torch.where(A, lab[None, :].expand(m, m), torch.full((m, m), m)).min(1).values
        if torch.equal(new, lab): break
        lab = new
    return torch.unique(lab, return_inverse=True)[1]


def observe(X, y, link=0.45) -> ObservedBoundary:
    pairs = clean_pairs(X, y); M, N = boundary_elements(pairs); return ObservedBoundary(M, N, link_fragments(M, link), pairs, X, y)


# ================================================================ estimators for partial fragments

def circle_kasa(S):
    """Algebraic least squares (Kåsa). Biased towards small radii on a short arc."""
    if len(S) < 3: return None
    A = torch.cat([S, torch.ones(len(S), 1, dtype=DT)], 1); sol = torch.linalg.lstsq(A, -(S ** 2).sum(1, keepdim=True)).solution[:, 0]; c = -sol[:2] / 2; r2 = float(c @ c - sol[2])
    return (c, math.sqrt(r2)) if r2 > 1e-9 and math.isfinite(r2) else None


def circle_taubin(S):
    """Taubin fit (normalisation by the gradient): markedly less biased than Kåsa on partial arcs."""
    if len(S) < 3: return None
    mu = S.mean(0); x, y = (S - mu).unbind(1); z = x * x + y * y; Mxx, Myy, Mxy, Mxz, Myz, Mzz = (float(v.mean()) for v in (x * x, y * y, x * y, x * z, y * z, z * z)); Mz = Mxx + Myy; Cov = Mxx * Myy - Mxy * Mxy; Var = Mzz - Mz * Mz
    A3, A2, A1, A0 = 4 * Mz, -3 * Mz * Mz - Mzz, Var * Mz + 4 * Cov * Mz - Mxz * Mxz - Myz * Myz, Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - Var * Cov; t, f = 0.0, A0
    for _ in range(30):
        d = A1 + t * (2 * A2 + 3 * A3 * t)
        if abs(d) < 1e-18: break
        tn = t - f / d; fn = A0 + tn * (A1 + tn * (A2 + tn * A3))
        if not math.isfinite(tn) or abs(fn) >= abs(f): break
        t, f = tn, fn
    det = t * t - t * Mz + Cov
    if abs(det) < 1e-14: return None
    cx, cy = (Mxz * (Myy - t) - Myz * Mxy) / det / 2, (Myz * (Mxx - t) - Mxz * Mxy) / det / 2; r2 = cx * cx + cy * cy + Mz
    return (mu + torch.tensor([cx, cy], dtype=DT), math.sqrt(r2)) if r2 > 1e-9 and math.isfinite(r2) else None


def circle_normals(S, N):
    """Intersection of the normals: the centre is the point closest (least squares) to all the lines
    normal to the boundary; the radius is the median distance."""
    if len(S) < 3: return None
    P = torch.eye(2, dtype=DT)[None] - N[:, :, None] * N[:, None, :]; A = P.sum(0)
    if float(torch.linalg.eigvalsh(A)[0]) < 1e-6 * len(S): return None                     # all normals parallel: not an arc
    c = torch.linalg.solve(A, (P @ S[:, :, None]).sum(0)[:, 0]); return c, float((S - c).norm(dim=1).median())


def circle_geometric(S, c, r, iters=15, fix_center=False, fix_radius=False, damping=1e-3):
    """GEOMETRIC least squares (damped Gauss-Newton on the distance to the circle). Centre or radius can be
    imposed (partial oracle, likelihood profile). Returns (centre, radius, root-mean-square deviation)."""
    c = torch.as_tensor(c, dtype=DT).clone(); r = float(r)
    for _ in range(iters):
        d = (S - c).norm(dim=1).clamp_min(1e-9); res = d - r; u = (S - c) / d[:, None]; cols = ([] if fix_center else [-u[:, 0], -u[:, 1]]) + ([] if fix_radius else [-torch.ones(len(S), dtype=DT)])
        if not cols: break
        J = torch.stack(cols, 1); step = torch.linalg.solve(J.T @ J + damping * torch.eye(J.shape[1], dtype=DT), -(J.T @ res))
        if not fix_center: c = c + step[:2].clamp(-0.5, 0.5)
        if not fix_radius: r = max(0.05, r + float(step[-1].clamp(-0.5, 0.5)))
        if float(step.abs().max()) < 1e-7: break
    return c, r, float(((S - c).norm(dim=1) - r).pow(2).mean().sqrt())


def concentric_fit(S1, S2):
    """SHARED CONSTRAINT — two arcs with the same centre: x² + y² + A·x + B·y + C_k = 0, linear in (A, B, C_1, C_2). A
    short arc borrows the centre that the other arc determines. Returns (centre, r1, r2) or None."""
    if len(S1) < 2 or len(S2) < 2 or len(S1) + len(S2) < 5: return None
    S = torch.cat([S1, S2]); g = torch.cat([torch.zeros(len(S1)), torch.ones(len(S2))]).to(DT); A = torch.stack([S[:, 0], S[:, 1], 1 - g, g], 1); sol = torch.linalg.lstsq(A, -(S ** 2).sum(1, keepdim=True)).solution[:, 0]
    c = -sol[:2] / 2; a, b = float(c @ c - sol[2]), float(c @ c - sol[3])
    return (c, math.sqrt(a), math.sqrt(b)) if a > 1e-9 and b > 1e-9 else None


def line_tls(S):
    """Total least squares. Returns (unit normal, offset, root-mean-square deviation)."""
    if len(S) < 2: return None
    mu = S.mean(0); n = torch.linalg.eigh((S - mu).T @ (S - mu))[1][:, 0]; b = float(n @ mu); return n, b, float(((S @ n) - b).pow(2).mean().sqrt())


def parallel_fit(S1, S2):
    """SHARED CONSTRAINT — two segments with the same direction: common normal (within-group scatter), two offsets."""
    if len(S1) < 2 or len(S2) < 2: return None
    Z = torch.cat([S1 - S1.mean(0), S2 - S2.mean(0)]); n = torch.linalg.eigh(Z.T @ Z)[1][:, 0]; return n, float(n @ S1.mean(0)), float(n @ S2.mean(0))


def modes(v: torch.Tensor, lo, hi, width, k=3, min_count=3) -> List[float]:
    """Modes of a smoothed histogram (local maxima), refined by the mean of the nearby values."""
    if len(v) == 0: return []
    nb = max(1, int(round((hi - lo) / width))); h = torch.histc(v.float(), nb, lo, hi); s = h + 0.5 * (torch.roll(h, 1) + torch.roll(h, -1)); out = []
    for i in s.argsort(descending=True).tolist():
        if float(h[i]) + float(h[max(0, i - 1)]) + float(h[min(nb - 1, i + 1)]) < min_count: continue
        x = lo + (i + 0.5) * width; near = v[(v - x).abs() < width]; x = float(near.mean()) if len(near) else x
        if all(abs(x - o) > 1.5 * width for o in out): out.append(x)
        if len(out) >= k: break
    return out


# ================================================================ geometry of an atom

def geometry(p: BNode) -> dict:
    """Named geometric parameters of an atom (independent of polarity)."""
    l = _leaf(p); c = tuple(p.params) if p.kind == "translate" else (0.0, 0.0); q = l.params
    if l.feat == "dot": return {"kind": "line", "angle": q[0], "offsets": tuple(q[1:])}
    if l.feat == "norm": return {"kind": "circle", "center": c, "radii": tuple(abs(v) for v in q)}
    return {"kind": "sector", "center": c, "direction": q[0], "half_width": abs(q[1])}


def build(skel: BNode, center=None, values=()) -> BNode:
    """Instantiates an atom skeleton: (centre if translated) + parameters of the predicate."""
    return skel.with_params((list(center) if skel.kind == "translate" else []) + [float(v) for v in values])


def orient(p: BNode, X, y) -> BNode:
    """Polarity of a half-plane: the one that agrees best with the labels near its boundary."""
    if atom_name(p) != "dot.le": return p
    q = build(p.skeleton(), None, (_wrap(p.params[0] + math.pi), -p.params[1])); m = near_boundary(p, X, 0.5)
    if int(m.sum()) < 4: return p
    return p if float((predict(p, X[m].float()) == y[m]).float().mean()) >= 0.5 else q


# ================================================================ scores: global, and occlusion-tolerant

def _coordinate(p: BNode, M: torch.Tensor):
    """Abscissa of a midpoint along the boundary of the primitive, and width of a segment."""
    g = geometry(p)
    if g["kind"] == "circle":
        c = torch.tensor(g["center"], dtype=DT); d = (M - c).norm(dim=1); which = torch.stack([(d - r).abs() for r in g["radii"]], 1).argmin(1); return torch.atan2(M[:, 1] - c[1], M[:, 0] - c[0]) + 10.0 * which, math.pi / 4
    if g["kind"] == "line":
        a = g["angle"]; t = torch.tensor([-math.sin(a), math.cos(a)], dtype=DT); n = torch.tensor([math.cos(a), math.sin(a)], dtype=DT); which = torch.stack([((M @ n) - b).abs() for b in g["offsets"]], 1).argmin(1); return (M @ t) + 40.0 * which, 0.75
    c = torch.tensor(g["center"], dtype=DT); ang = torch.atan2(M[:, 1] - c[1], M[:, 0] - c[0]); side = (torch.atan2(torch.sin(ang - g["direction"]), torch.cos(ang - g["direction"])) > 0).to(DT); return (M - c).norm(dim=1) + 40.0 * side, 0.75


def segment_profile(p: BNode, pairs: Pairs, min_pairs=2, margin=0.15):
    """Per boundary segment: separated pairs and opposite pairs. A segment is SEEN if it carries a contrast."""
    s = separates(p, pairs)
    if int(s.sum()) == 0: return []
    M = ((pairs.P + pairs.Q) / 2)[s]; opp = pairs.opposite[s]; co, w = _coordinate(p, M); key = torch.floor(co / w).long(); out = []
    for kx in torch.unique(key).tolist():
        m = key == kx; n = int(m.sum()); kk = float(opp[m].sum())
        if n >= min_pairs: out.append({"n": n, "k": kk, "seen": kk / n > pairs.q0 + margin})
    return out


def visible_score(p: BNode, pairs: Pairs) -> Tuple[float, float]:
    """OCCLUSION-TOLERANT score: contrast z computed on the seen segments only. A silent segment (boundary expected
    but without contrast) is NOT counted against the primitive — it may be occluded. Returns (z_visible, seen share)."""
    seg = segment_profile(p, pairs)
    if not seg: return 0.0, 0.0
    n = sum(s["n"] for s in seg if s["seen"]); k = sum(s["k"] for s in seg if s["seen"]); q0 = min(0.9, max(0.02, pairs.q0))
    return ((k - q0 * n) / math.sqrt(n * q0 * (1 - q0)) if n else 0.0), sum(s["seen"] for s in seg) / len(seg)


def global_score(p: BNode, pairs: Pairs) -> float: return contrast(p, pairs)["z"]


# ================================================================ region of compatible parameters

@dataclass
class ParameterRegion:
    """Set of parameters COMPATIBLE with the observed fragment — not a point estimate."""
    intervals: Dict[str, Tuple[float, float]]
    hypotheses: List[Tuple[Tuple[float, ...], float]]                 # (parameters, root-mean-square deviation from the fragment)
    weights: List[float] = field(default_factory=list)                # lightweight posterior: ∝ exp(−n·rms² / 2σ²)
    posterior: Dict[str, Tuple[float, float]] = field(default_factory=dict)        # mean, standard deviation per parameter
    sensitivity: Dict[str, float] = field(default_factory=dict)       # local diagnostic (approximate Fisher information)

    def identifiability(self) -> Dict[str, str]:
        out = {}
        for k, (lo, hi) in self.intervals.items():
            t = TOL["center" if k in ("cx", "cy") else "radius" if k == "r" else "angle" if k == "angle" else "offset"]; w = hi - lo; out[k] = IDENT[0] if w <= 2 * t else IDENT[1] if w <= 6 * t else IDENT[2]
        return out

    def contains(self, **truth) -> bool: return all(self.intervals[k][0] - 1e-9 <= v <= self.intervals[k][1] + 1e-9 for k, v in truth.items() if k in self.intervals)
    def widths(self): return {k: hi - lo for k, (lo, hi) in self.intervals.items()}


def _posterior(rows, n, names):
    w = torch.tensor([math.exp(-n * (rms ** 2) / (2 * SIGMA ** 2)) for _, rms in rows], dtype=DT); w = w / w.sum().clamp_min(1e-300); P = torch.tensor([p for p, _ in rows], dtype=DT); mu = (w[:, None] * P).sum(0); sd = (w[:, None] * (P - mu) ** 2).sum(0).sqrt()
    return w.tolist(), {k: (float(mu[i]), float(sd[i])) for i, k in enumerate(names)}


def circle_sensitivity(S, c, r) -> dict:
    """Local diagnostic: J = ∂(distance to the circle)/∂(cx, cy, r). Eigenvalues of JᵀJ (flat directions),
    conditioning, and correlation between the radius and the position of the centre along the axis of the arc."""
    d = (S - c).norm(dim=1).clamp_min(1e-9); u = (S - c) / d[:, None]; J = torch.cat([-u, -torch.ones(len(S), 1, dtype=DT)], 1); H = J.T @ J; ev, evec = torch.linalg.eigh(H); C = torch.linalg.pinv(H); sd = C.diagonal().clamp_min(1e-18).sqrt()
    axis = u.mean(0); axis = axis / axis.norm().clamp_min(1e-9); rho = float((torch.cat([axis, torch.zeros(1, dtype=DT)]) @ C @ torch.tensor([0, 0, 1.0], dtype=DT)) / (math.sqrt(max(1e-18, float(torch.cat([axis, torch.zeros(1, dtype=DT)]) @ C @ torch.cat([axis, torch.zeros(1, dtype=DT)])))) * float(sd[2])))
    return {"condition": float(ev[-1] / ev[0].clamp_min(1e-12)), "smallest_eigenvalue": float(ev[0]), "flat_direction_radius_share": float(evec[2, 0] ** 2), "corr_radius_center": rho, "arc_angle": arc_angle(S, c)}


def arc_angle(S, c) -> float:
    """Angular opening of the observed arc (2π minus the largest angular gap)."""
    a = torch.atan2(S[:, 1] - c[1], S[:, 0] - c[0]).sort().values
    if len(a) < 2: return 0.0
    gaps = torch.cat([a[1:] - a[:-1], (a[0] + 2 * math.pi - a[-1])[None]]); return float(2 * math.pi - gaps.max())


def circle_region(S, c, r, slack=1.5) -> ParameterRegion:
    """Profile over the radius: for each radius, the best centre; the compatible radii are those whose deviation from the
    fragment stays below max(slack × best deviation, σ). On a short arc, the interval is wide: the radius is not determined."""
    c = torch.as_tensor(c, dtype=DT); mu = S.mean(0); u = c - mu; u = u / u.norm().clamp_min(1e-9); rows = []
    for rr in torch.arange(0.3, 3.01, 0.1).tolist():
        cc, _, rms = circle_geometric(S, c + (rr - r) * u, rr, iters=8, fix_radius=True); rows.append(((float(cc[0]), float(cc[1]), rr), rms))
    best = min(rms for _, rms in rows); thr = max(slack * best, SIGMA); ok = [x for x in rows if x[1] <= thr] or [min(rows, key=lambda x: x[1])]; names = ("cx", "cy", "r"); w, post = _posterior(ok, len(S), names)
    return ParameterRegion({k: (min(p[i] for p, _ in ok) - PAD[k], max(p[i] for p, _ in ok) + PAD[k]) for i, k in enumerate(names)}, ok, w, post, circle_sensitivity(S, c, r))


def line_region(S, n, b, slack=1.5) -> ParameterRegion:
    """Profile over the orientation: for each angle, the best offset."""
    a0 = math.atan2(float(n[1]), float(n[0])); rows = []; t = torch.tensor([-math.sin(a0), math.cos(a0)], dtype=DT)
    for da in torch.arange(-0.6, 0.601, 0.05).tolist():
        nn = torch.tensor([math.cos(a0 + da), math.sin(a0 + da)], dtype=DT); pr = S @ nn; bb = float(pr.mean()); rows.append(((a0 + da, bb), float((pr - bb).pow(2).mean().sqrt())))
    best = min(rms for _, rms in rows); thr = max(slack * best, SIGMA); ok = [x for x in rows if x[1] <= thr] or [min(rows, key=lambda x: x[1])]; names = ("angle", "offset"); w, post = _posterior(ok, len(S), names); ext = float((S @ t).max() - (S @ t).min()) if len(S) > 1 else 0.0
    return ParameterRegion({k: (min(p[i] for p, _ in ok) - PAD[k], max(p[i] for p, _ in ok) + PAD[k]) for i, k in enumerate(names)}, ok, w, post, {"segment_length": ext, "angle_sd": SIGMA * math.sqrt(12.0 / max(1, len(S))) / max(ext, 1e-6)})


# ================================================================ component hypotheses (circles, lines)

@dataclass
class Component:
    kind: str                       # "circle" | "line"
    params: tuple                   # (cx, cy, r) | (angle, offset)
    inliers: torch.Tensor           # indices into ObservedBoundary.points
    rms: float


def _circle_inliers(M, c, r, band): return torch.nonzero(((M - torch.as_tensor(c, dtype=DT)).norm(dim=1) - r).abs() < band)[:, 0]


def circle_components(ob: ObservedBoundary, estimator="geometric", top=10, seed=0, band=0.3) -> List[Component]:
    """Circle hypotheses from PARTIAL fragments: local consensus (3 points) → points of the band →
    arc estimator → new band → estimator. `estimator`: kasa | taubin | normals | geometric (Taubin then
    geometric least squares)."""
    M, N = ob.points, ob.normals; out = []
    for cand in RansacProposer(top=top).propose(ob.X, ob.y, ob.pairs, seed):
        if atom_name(cand.program) != "norm.le": continue
        c = torch.tensor(cand.program.params, dtype=DT); r = cand.program.children[0].params[0]; idx = _circle_inliers(M, c, r, band); rms = float("inf")
        for _ in range(2):
            if len(idx) < 4: break
            S = M[idx]; fit = circle_kasa(S) if estimator == "kasa" else circle_normals(S, N[idx]) if estimator == "normals" else circle_taubin(S)
            if fit is None or not (0.15 <= fit[1] <= 4.0): break
            c, r = fit
            if estimator == "geometric": c, r, _ = circle_geometric(S, c, r)
            if not (0.15 <= r <= 4.0): break
            rms = float(((S - c).norm(dim=1) - r).pow(2).mean().sqrt()); idx = _circle_inliers(M, c, r, band)
        if len(idx) >= 4 and math.isfinite(rms): out.append(Component("circle", (float(c[0]), float(c[1]), float(r)), idx, rms))
    return _nms(out, [len(o.inliers) - 5 * o.rms for o in out], lambda a, b: math.hypot(a.params[0] - b.params[0], a.params[1] - b.params[1]) < 0.3 and abs(a.params[2] - b.params[2]) < 0.25, top)


def line_components(ob: ObservedBoundary, top=10, seed=0, band=0.25) -> List[Component]:
    M = ob.points; out = []
    for cand in RansacProposer(top=top).propose(ob.X, ob.y, ob.pairs, seed):
        if atom_name(cand.program) != "dot.le": continue
        a, b = cand.program.params; n = torch.tensor([math.cos(a), math.sin(a)], dtype=DT); rms = float("inf"); idx = torch.nonzero(((M @ n) - b).abs() < band)[:, 0]
        for _ in range(2):
            if len(idx) < 4: break
            n2, b2, rms = line_tls(M[idx])
            if float(n2 @ n) < 0: n2, b2 = -n2, -b2
            n, b = n2, b2; idx = torch.nonzero(((M @ n) - b).abs() < band)[:, 0]
        if len(idx) >= 4 and math.isfinite(rms): out.append(Component("line", (math.atan2(float(n[1]), float(n[0])), float(b)), idx, rms))
    same = lambda u, v: (abs(_wrap(u.params[0] - v.params[0])) < 0.2 and abs(u.params[1] - v.params[1]) < 0.3) or (abs(abs(_wrap(u.params[0] - v.params[0])) - math.pi) < 0.2 and abs(u.params[1] + v.params[1]) < 0.3)
    return _nms(out, [len(o.inliers) - 5 * o.rms for o in out], same, top)


# ================================================================ complete latent primitive and ranked hypotheses

@dataclass
class Hypothesis:
    program: BNode
    score: float = 0.0
    visible_fraction: float = 0.0
    source: str = ""
    region: Optional[ParameterRegion] = None


@dataclass
class LatentBoundary:
    """COMPLETE primitive hypothesis. `points`: rendered boundary; `observed`: part supported by an observed midpoint;
    the rest is a COMPLETION — a controlled and revisable hypothesis, not an observation."""
    program: BNode
    points: torch.Tensor
    observed: torch.Tensor

    @property
    def observed_fraction(self): return float(self.observed.float().mean()) if len(self.observed) else 0.0
    @property
    def completion(self): return self.points[~self.observed]


def render(p: BNode, n=96, box=(-3.0, 3.0, -3.0, 3.0), reach=4.5) -> torch.Tensor:
    """Points of the complete boundary of the primitive, restricted to the box."""
    g = geometry(p); t = torch.linspace(0, 1, n, dtype=DT)
    if g["kind"] == "circle":
        c = torch.tensor(g["center"], dtype=DT); a = 2 * math.pi * t; P = torch.cat([c + r * torch.stack([torch.cos(a), torch.sin(a)], 1) for r in g["radii"]])
    elif g["kind"] == "line":
        a = g["angle"]; nrm = torch.tensor([math.cos(a), math.sin(a)], dtype=DT); tg = torch.tensor([-math.sin(a), math.cos(a)], dtype=DT); s = (2 * t - 1) * reach; P = torch.cat([b * nrm + s[:, None] * tg for b in g["offsets"]])
    else:
        c = torch.tensor(g["center"], dtype=DT); s = t * reach; P = torch.cat([c + s[:, None] * torch.tensor([math.cos(d), math.sin(d)], dtype=DT) for d in (g["direction"] + g["half_width"], g["direction"] - g["half_width"])])
    return P[(P[:, 0] >= box[0]) & (P[:, 0] <= box[1]) & (P[:, 1] >= box[2]) & (P[:, 1] <= box[3])]


def complete(p: BNode, ob: ObservedBoundary, tol=0.3, box=None) -> LatentBoundary:
    """observed boundary → latent boundary completion: the whole boundary of the hypothesis, labelled observed / completed."""
    X = ob.X; box = box or (float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 1].min()), float(X[:, 1].max())); P = render(p, box=box)
    seen = (torch.cdist(P, ob.points).min(1).values < tol) if len(P) and len(ob.points) else torch.zeros(len(P), dtype=torch.bool); return LatentBoundary(p, P, seen)


def _circle_region(ob, comp): return circle_region(ob.points[comp.inliers], comp.params[:2], comp.params[2])
def _line_region(ob, comp): return line_region(ob.points[comp.inliers], torch.tensor([math.cos(comp.params[0]), math.sin(comp.params[0])], dtype=DT), comp.params[1])


def infer(skel: BNode, ob: ObservedBoundary, k=5, score="visible", joint=True, estimator="geometric", given: Optional[dict] = None, seed=0, regions=True) -> List[Hypothesis]:
    """KNOWN TYPE → compatible complete primitives, RANKED (top-k), each with its parameter region.

    partial boundary → primitive family → compatible parameter set

    `given` (partial oracles): {"center": (cx, cy)} | {"orientation": angle} | {"size": radii | width}.
    `joint`: shared constraints (common centre, parallelism); otherwise components fitted independently.
    `score`: "visible" (occlusion-tolerant) | "global" (contrast over the whole boundary)."""
    given = given or {}; l = _leaf(skel); M = ob.points; cands: List[Tuple[BNode, str, Optional[ParameterRegion]]] = []; ctr = torch.tensor(given["center"], dtype=DT) if "center" in given else None
    if l.feat == "norm":
        cs = circle_components(ob, estimator, seed=seed) if ctr is None else []
        if "size" in given and ctr is None:                                          # known radii: only the centre is estimated, for each possible radius
            cs2 = []
            for c in cs:
                r0 = min(given["size"], key=lambda r: abs(r - c.params[2])); cc, _, rms = circle_geometric(M[c.inliers], c.params[:2], r0, fix_radius=True); cs2.append(Component("circle", (float(cc[0]), float(cc[1]), r0), c.inliers, rms))
            cs = cs2
        if l.op != "in":
            if ctr is not None: cands += [(build(skel, ctr.tolist(), (r,)), "center+radial_mode", None) for r in ([given["size"][0]] if "size" in given else modes((M - ctr).norm(dim=1), 0.15, 3.5, 0.15, k))]
            else: cands += [(build(skel, c.params[:2], (c.params[2],)), estimator, _circle_region(ob, c) if regions else None) for c in cs]
        else:
            if ctr is not None:
                rs = sorted(given["size"]) if "size" in given else modes((M - ctr).norm(dim=1), 0.15, 3.5, 0.15, 4); cands += [(build(skel, ctr.tolist(), (rs[i], rs[j])), "center+radial_modes", None) for i in range(len(rs)) for j in range(i + 1, len(rs)) if rs[j] - rs[i] > 0.3]
            elif "size" in given: cands += [(build(skel, c.params[:2], sorted(given["size"])), "radii_given", None) for c in cs]
            else:
                for i, a in enumerate(cs):
                    for b in cs[i + 1:]:
                        if math.hypot(a.params[0] - b.params[0], a.params[1] - b.params[1]) < 0.6 and abs(a.params[2] - b.params[2]) > 0.3:
                            if joint:
                                f = concentric_fit(M[a.inliers], M[b.inliers])
                                if f is not None: cands.append((build(skel, f[0].tolist(), sorted(f[1:])), "concentric_joint", None)); continue
                            cands.append((build(skel, ((a.params[0] + b.params[0]) / 2, (a.params[1] + b.params[1]) / 2), sorted((a.params[2], b.params[2]))), "concentric_independent", None))
                    if joint:                                                    # a single arc found: the second radius is searched among the radial modes around ITS centre
                        c = torch.tensor(a.params[:2], dtype=DT)
                        for r2 in modes((M - c).norm(dim=1), 0.15, 3.5, 0.15, 3):
                            if abs(r2 - a.params[2]) > 0.3: cands.append((build(skel, a.params[:2], sorted((a.params[2], r2))), "arc+radial_mode", _circle_region(ob, a) if regions else None))
    elif l.feat == "dot":
        if "orientation" in given:
            a0 = given["orientation"]; n = torch.tensor([math.cos(a0), math.sin(a0)], dtype=DT); bs = modes(M @ n, -4.5, 4.5, 0.15, 5)
            if l.op != "in": cands += [(build(skel, None, (a0, b)), "orientation+offset_mode", None) for b in bs]
            else: cands += [(build(skel, None, (a0, bs[i], bs[j]) if bs[i] < bs[j] else (a0, bs[j], bs[i])), "orientation+offset_modes", None) for i in range(len(bs)) for j in range(i + 1, len(bs)) if 0.3 < abs(bs[i] - bs[j]) < 2.8]
        else:
            ls = line_components(ob, seed=seed)
            if l.op != "in": cands += [(build(skel, None, c.params), "tls", _line_region(ob, c) if regions else None) for c in ls]
            else:
                for i, a in enumerate(ls):
                    if "size" in given:                                           # known width
                        w = given["size"][0]; cands += [(build(skel, None, (a.params[0], min(a.params[1], a.params[1] + s * w), max(a.params[1], a.params[1] + s * w))), "width_given", None) for s in (-1.0, 1.0)]; continue
                    for b in ls[i + 1:]:
                        d = _wrap(a.params[0] - b.params[0]); a2, b2 = (b.params[0], b.params[1]) if abs(d) < math.pi / 2 else (_wrap(b.params[0] + math.pi), -b.params[1])
                        if abs(_wrap(a.params[0] - a2)) < 0.3 and 0.3 < abs(a.params[1] - b2) < 2.8:
                            if joint:
                                n, b1, bb = parallel_fit(M[a.inliers], M[b.inliers]); cands.append((build(skel, None, (math.atan2(float(n[1]), float(n[0])), min(b1, bb), max(b1, bb))), "parallel_joint", None))
                            else: cands.append((build(skel, None, (a.params[0], min(a.params[1], b2), max(a.params[1], b2))), "parallel_independent", None))
                    if joint:                                                    # a single line found: the second edge among the offset modes along ITS normal
                        n = torch.tensor([math.cos(a.params[0]), math.sin(a.params[0])], dtype=DT)
                        for b2 in modes(M @ n, -4.5, 4.5, 0.15, 4):
                            if 0.3 < abs(b2 - a.params[1]) < 2.8: cands.append((build(skel, None, (a.params[0], min(a.params[1], b2), max(a.params[1], b2))), "line+offset_mode", _line_region(ob, a) if regions else None))
    else:                                                                        # sector: two rays with the same vertex
        def wedges(c, dirs):
            out = []
            for i, t1 in enumerate(dirs):
                for t2 in dirs[i + 1:]:
                    w = _wrap(t2 - t1) / 2
                    if 0.2 < abs(w) < 1.45: out.append((build(skel, c, (_wrap(t1 + w), abs(w))), None))
            return out
        if ctr is not None:
            near = (M - ctr).norm(dim=1) < 2.5; ang = torch.atan2(M[near, 1] - ctr[1], M[near, 0] - ctr[0]); cands += [(p, "center+angular_modes", r) for p, r in wedges(ctr.tolist(), modes(ang, -math.pi, math.pi, 0.2, 4))]
        else:
            ls = line_components(ob, seed=seed)
            for i, a in enumerate(ls):
                a1, b1 = a.params
                for b in ls[i + 1:]:
                    a2, b2 = b.params; det = math.sin(a2 - a1)
                    if abs(det) < 0.3: continue
                    c = ((b1 * math.sin(a2) - b2 * math.sin(a1)) / det, (b2 * math.cos(a1) - b1 * math.cos(a2)) / det)
                    if max(abs(c[0]), abs(c[1])) > 4.5: continue
                    cands += [(p, "secant_lines", r) for p, r in wedges(c, [a1 + math.pi / 2, a1 - math.pi / 2, a2 + math.pi / 2, a2 - math.pi / 2])]
                if joint:                                                        # a single ray found: vertex at one end of the fragment, other side from angular modes
                    S = M[a.inliers]; tg = torch.tensor([-math.sin(a1), math.cos(a1)], dtype=DT); pr = S @ tg
                    for end, other in ((S[pr.argmin()], S[pr.argmax()]), (S[pr.argmax()], S[pr.argmin()])):
                        t1 = math.atan2(float(other[1] - end[1]), float(other[0] - end[0])); near = ((M - end).norm(dim=1) < 2.5) & ((M - end).norm(dim=1) > 0.4); ang = torch.atan2(M[near, 1] - end[1], M[near, 0] - end[0])
                        for t2 in modes(ang, -math.pi, math.pi, 0.2, 3):
                            w = _wrap(t2 - t1) / 2
                            if 0.2 < abs(w) < 1.45: cands.append((build(skel, end.tolist(), (_wrap(t1 + w), abs(w))), "ray+angular_mode", None))
        if "size" in given: cands = [(build(skel, geometry(p)["center"], (geometry(p)["direction"], given["size"][0])), s + "+width", r) for p, s, r in cands]
        if "orientation" in given: cands = [(build(skel, geometry(p)["center"], (given["orientation"], geometry(p)["half_width"])), s + "+direction", r) for p, s, r in cands]
    out = []
    for p, src, reg in cands:
        p = orient(p, ob.X, ob.y); zv, vf = visible_score(p, ob.pairs); out.append(Hypothesis(p, zv if score == "visible" else global_score(p, ob.pairs), vf, src, reg))
    out.sort(key=lambda h: -h.score); keep = []; P = ob.X.float()
    for h in out:                                                                # DISTINCT hypotheses: no merging, only functional near-duplicates are discarded
        s = predict(h.program, P)
        if all(float((s != predict(o.program, P)).float().mean()) > 0.02 for o in keep): keep.append(h)
        if len(keep) >= k: break
    return keep


# ================================================================ explicit occlusion model and counterfactual rendering

def _is_composite(n: BNode): return n.kind in COMBINATORS or (n.kind == "not" and bool(n.children) and _is_composite(n.children[0]))


def leaf_paths(prog: BNode, path=()) -> List[Tuple[tuple, BNode]]:
    if not _is_composite(prog): return [(path, prog)]
    return [x for i, c in enumerate(prog.children) for x in leaf_paths(c, path + (i,))]


def replace_at(prog: BNode, path: tuple, new: BNode) -> BNode:
    if not path: return new
    return BNode(prog.kind, prog.feat, prog.op, prog.params, tuple(replace_at(c, path[1:], new) if i == path[0] else c for i, c in enumerate(prog.children)))


def eval_forced(prog: BNode, X, path: tuple, value: bool, _at=()) -> torch.Tensor:
    """Value of the composite when the leaf `path` is forced to true or to false everywhere."""
    if _at == path: return torch.full((len(X),), int(value), dtype=torch.long)
    if not _is_composite(prog): return predict(prog, X.float()).long()
    v = [eval_forced(c, X, path, value, _at + (i,)) for i, c in enumerate(prog.children)]
    if prog.kind == "not": return 1 - v[0]
    out = v[0]
    for w in v[1:]: out = out & w if prog.kind == "and" else out | w if prog.kind == "or" else out ^ w
    return out


def visible_mask(prog: BNode, path: tuple, P: torch.Tensor) -> torch.Tensor:
    """Primitive + CompositeContext → VisibleRegion: the primitive is observable only at the points where forcing it to true or
    to false changes the output of the composite. Elsewhere, its boundary is OCCLUDED — absent because hidden, not because
    the primitive is false."""
    return eval_forced(prog, P, path, True) != eval_forced(prog, P, path, False)


def occlusion_score(prog: BNode, path: tuple, pairs: Pairs) -> dict:
    """P(observation | primitive, occlusion), pair version: the contrast of the leaf is measured only on the pairs
    whose midpoint is in its visible region. Also returns the contrast on the occluded part (expected: zero)."""
    leaf = dict(leaf_paths(prog))[path]; s = separates(leaf, pairs); vis = visible_mask(prog, path, (pairs.P + pairs.Q) / 2) | visible_mask(prog, path, pairs.P) | visible_mask(prog, path, pairs.Q); q0 = min(0.9, max(0.02, pairs.q0))
    def z(m):
        n = float(m.sum()); k = float(pairs.opposite[m].sum()); return (k - q0 * n) / math.sqrt(n * q0 * (1 - q0)) if n else 0.0
    return {"z_visible": z(s & vis), "z_hidden": z(s & ~vis), "visible_pairs": int((s & vis).sum()), "hidden_pairs": int((s & ~vis).sum()), "visible_fraction": float((s & vis).sum()) / max(1, int(s.sum()))}


def rendering_errors(prog: BNode, X, y) -> int:
    """COUNTERFACTUAL RENDERING: the whole composite is rendered on the observations; the occlusion induced by the composition
    is therefore applied de facto, and only the really visible part of each primitive weighs in the comparison."""
    return int((predict(prog, X.float()) != y).sum())


def joint_select(options: Sequence[Sequence[BNode]], template: BNode, X, y, max_combos=400) -> Tuple[BNode, int, int]:
    """JOINT choice of the parameters of several leaves under a given structure: the combination of hypotheses that
    minimises the rendering error. Returns (program, errors, combinations evaluated)."""
    paths = [p for p, _ in leaf_paths(template)]; best, be, n = template, rendering_errors(template, X, y), 0; idx = [0] * len(paths)
    for _ in range(2):                                                            # coordinate descent: no exhaustive enumeration
        for i, pth in enumerate(paths):
            for j, alt in enumerate(options[i]):
                if n >= max_combos: return best, be, n
                cand = replace_at(best, pth, alt); e = rendering_errors(cand, X, y); n += 1
                if e < be: best, be, idx[i] = cand, e, j
    return best, be, n


def refit_visible(leaf: BNode, S: torch.Tensor) -> Optional[BNode]:
    """Re-estimates a leaf on the midpoints of ITS visible part only (shared constraints included)."""
    g = geometry(leaf); sk = leaf.skeleton()
    if len(S) < 4: return None
    if g["kind"] == "circle":
        c = torch.tensor(g["center"], dtype=DT)
        if len(g["radii"]) == 1: c2, r2, _ = circle_geometric(S, c, g["radii"][0]); return build(sk, c2.tolist(), (r2,)) if 0.15 <= r2 <= 4.0 else None
        d = (S - c).norm(dim=1); inner = (d - g["radii"][0]).abs() < (d - g["radii"][1]).abs(); f = concentric_fit(S[inner], S[~inner]); return build(sk, f[0].tolist(), sorted(f[1:])) if f is not None else None
    if g["kind"] == "line":
        n0 = torch.tensor([math.cos(g["angle"]), math.sin(g["angle"])], dtype=DT)
        if len(g["offsets"]) == 1:
            n, b, _ = line_tls(S); n, b = (n, b) if float(n @ n0) >= 0 else (-n, -b); return build(sk, None, (math.atan2(float(n[1]), float(n[0])), b)) if float(n @ n0) > 0.8 else None
        pr = S @ n0; first = (pr - g["offsets"][0]).abs() < (pr - g["offsets"][1]).abs(); f = parallel_fit(S[first], S[~first])
        if f is None: return None
        n, b1, b2 = f; n, b1, b2 = (n, b1, b2) if float(n @ n0) >= 0 else (-n, -b1, -b2); return build(sk, None, (math.atan2(float(n[1]), float(n[0])), min(b1, b2), max(b1, b2)))
    return None


# ================================================================ primitive ↔ composition exchange

@dataclass
class PrimitiveNode:
    """Primitive node: open parametric hypotheses (upward message) and current value."""
    path: tuple
    current: BNode
    alternatives: List[BNode] = field(default_factory=list)
    visible_fraction: Optional[float] = None


@dataclass
class CompositeNode:
    """Composite node: the candidate structure. Downward message: the region where each primitive is visible."""
    program: BNode

    def visibility(self, path, P): return visible_mask(self.program, path, P)


@dataclass
class Exchange:
    program: BNode
    errors_before: int
    errors_after: int
    log: List[dict]
    iterations: int
    evaluations: int


def exchange(prog: BNode, ob: ObservedBoundary, alternatives: Optional[Dict[str, List[BNode]]] = None, iters=2, band=0.35) -> Exchange:
    """primitive hypothesis → composition hypothesis → constraints back to primitive parameters → refined hypothesis.

    At each (bounded) iteration, for each leaf: (1) the composite sends the visible region of the leaf;
    (2) the leaf proposes parameters re-estimated on its visible midpoints only, and its alternative hypotheses of
    the same type; (3) a proposal is validated only if the error of the RENDERING of the composite decreases (commit), otherwise
    the previous state is kept (rollback). Every decision is logged."""
    X, y, M = ob.X, ob.y, ob.points; alternatives = alternatives or {}; comp = CompositeNode(prog); e0 = e = rendering_errors(prog, X, y); log = []; n_eval = 0; it = 0
    for it in range(1, iters + 1):
        changed = False
        for path, leaf in leaf_paths(comp.program):
            vis = comp.visibility(path, M) if len(M) else torch.zeros(0, dtype=torch.bool); node = PrimitiveNode(path, leaf, list(alternatives.get(repr(leaf.skeleton()), [])), float(vis.float().mean()) if len(vis) else None)
            props = [("alternative", a) for a in node.alternatives]
            if len(M): r = refit_visible(leaf, M[vis & near_boundary(leaf, M, band)]); props += [("visible_refit", r)] if r is not None else []
            best = None
            for kind, p in props:
                e2 = rendering_errors(replace_at(comp.program, path, p), X, y); n_eval += 1
                if e2 < e and (best is None or e2 < best[2]): best = (kind, p, e2)
            log.append({"iteration": it, "path": path, "type": atom_name(leaf), "proposals": len(props), "visible_fraction": node.visible_fraction, "action": "commit" if best else "rollback", "kind": best[0] if best else None, "errors_before": e, "errors_after": best[2] if best else e})
            if best: comp = CompositeNode(replace_at(comp.program, path, best[1])); e = best[2]; changed = True
        if not changed: break
    return Exchange(comp.program, e0, e, log, it, n_eval)


# ================================================================ PARAMETER-SPECIFIC active measurements

def parameter_queries(hyps: Sequence[BNode], pool: torch.Tensor, budget: int, weights: Optional[Sequence[float]] = None, context: Optional[Tuple[BNode, tuple]] = None, spacing=0.2) -> torch.Tensor:
    """Where to measure to decide between PARAMETRIC hypotheses of the same primitive: where they predict different
    labels (weighted variance of the predictions). With a `context` (composite, path), the points where the primitive
    is occluded are excluded: measuring there teaches nothing about its parameters. Returns the chosen indices."""
    if not hyps: return torch.zeros(0, dtype=torch.long)
    w = torch.tensor(list(weights) if weights is not None else [1.0] * len(hyps)); w = w / w.sum(); V = torch.stack([predict(h, pool.float()).float() for h in hyps]); mu = (w[:, None] * V).sum(0); sc = mu * (1 - mu)
    if context is not None: sc = sc * visible_mask(context[0], context[1], pool).float()
    chosen: List[int] = []
    for i in sc.argsort(descending=True).tolist():
        if float(sc[i]) <= 0 or len(chosen) >= budget: break
        if all(float((pool[i] - pool[j]).norm()) > spacing for j in chosen): chosen.append(i)
    return torch.tensor(chosen, dtype=torch.long)

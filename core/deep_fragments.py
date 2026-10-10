"""Fragment Attribution & Primitive Type Inference — which observed portions belong to which primitive, and which
primitive type is compatible with each group of fragments?

    observed midpoints → fragments → compatibility graph → groups → families → types → parameters

Objects:
  BoundaryFragment               an OBSERVED boundary portion (never assigned by default to a single primitive)
  CompatibleFragmentAssignments  compatible roles of a fragment (line, arc, ring edge, sector ray, noise…)
  FragmentGraph                  nodes = fragments; typed edges = geometric compatibilities
  TypeHypothesis                 (family, type, program) + SEPARATE signals + fused score
  OcclusionOwnership             why an expected boundary portion is missing: hidden, out of domain, noise, absent

Two errors are separated: the ATTRIBUTION (grouping of the midpoints by source) and the TYPE (family then type).
Small grammar of families: line-like, circle-like, paired-lines, paired-circles, radial-composite — then projected
onto half-plane, disc / anti-disc, band, ring, sector.

Geometric rules and explicit scores only: no learned classifier. No world truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from core.boundary_program import BNode, predict
from core.deep_fit import DT
from core.deep_partial import (SIGMA, ObservedBoundary, build, circle_components, circle_geometric, circle_taubin, concentric_fit, geometry, infer, line_components, line_tls, link_fragments, orient, parallel_fit, render,
                               visible_score)
from core.deep_primitives import atom_name
from core.deep_primitives2 import _wrap, near_boundary

GROUPINGS = ("proximity", "tangent", "curvature", "center", "orientation", "shared", "consensus")
FAMILIES = ("line-like", "circle-like", "paired-lines", "paired-circles", "radial-composite")
FAMILY_TYPES = {"line-like": ("half",), "circle-like": ("disc", "antidisc"), "paired-lines": ("band",), "paired-circles": ("ring",), "radial-composite": ("sector",)}
SKELETONS = {"half": BNode("pred", "dot", "le"), "band": BNode("pred", "dot", "in"), "disc": BNode("translate", children=(BNode("pred", "norm", "le"),)), "antidisc": BNode("translate", children=(BNode("pred", "norm", "ge"),)),
             "ring": BNode("translate", children=(BNode("pred", "norm", "in"),)), "sector": BNode("translate", children=(BNode("pred", "angle", "in"),))}
ATOM_TYPE = {"dot.le": "half", "dot.in": "band", "norm.le": "disc", "norm.ge": "antidisc", "norm.in": "ring", "angle.in": "sector"}
OWNERSHIP = ("observed", "occluded", "domain", "noise", "absent")
MIN_POINTS = 4


def type_of(p: BNode) -> str: return ATOM_TYPE.get(atom_name(p).replace("not_", ""), "other")


# ================================================================ fragments

@dataclass
class BoundaryFragment:
    """Observed boundary portion. `model` is the preferred local fit, not an attribution to a primitive."""
    index: int
    ids: torch.Tensor               # provenance: indices of the midpoints in ObservedBoundary.points
    points: torch.Tensor
    normals: torch.Tensor           # local orientation
    model: str                      # "line" | "circle"
    params: tuple                   # (angle, offset) | (cx, cy, r)
    curvature: float                # 1 / radius, 0 for a line
    length: float
    endpoints: torch.Tensor         # [2, 2]
    rms_line: float
    rms_circle: float
    confidence: float               # n / (n + 4), weighted by the quality of the fit
    polarity: Optional[int] = None  # side of the model normal where the label is 1 (+1 / −1), if readable
    neighbors: List[int] = field(default_factory=list)

    @property
    def uncertainty(self): return (self.rms_line if self.model == "line" else self.rms_circle) / math.sqrt(max(1, len(self.ids)))
    @property
    def straight(self): return self.model == "line"


def fit_fragment(S: torch.Tensor):
    """Line or arc? The arc is preferred only if it clearly reduces the deviation (a line is an arc of infinite radius:
    at equal fit, the simplest model wins). Returns (model, parameters, line rms, circle rms)."""
    n, b, rl = line_tls(S); rc, circ = float("inf"), None; f = circle_taubin(S) if len(S) >= 5 else None
    if f is not None and 0.2 <= f[1] <= 3.5:
        c, r, rc = circle_geometric(S, f[0], f[1], iters=8)
        if 0.2 <= r <= 3.5: circ = (float(c[0]), float(c[1]), float(r))
        else: rc = float("inf")
    if circ is not None and rl > max(1.4 * rc, 0.6 * SIGMA): return "circle", circ, rl, rc
    return "line", (math.atan2(float(n[1]), float(n[0])), float(b)), rl, rc


def _extent(S: torch.Tensor):
    mu = S.mean(0); t = torch.linalg.eigh((S - mu).T @ (S - mu))[1][:, 1]; pr = (S - mu) @ t; return S[pr.argmin()], S[pr.argmax()]


def make_fragment(i: int, ids: torch.Tensor, ob: ObservedBoundary, forced=None) -> BoundaryFragment:
    """`forced` = (model, parameters) when the fragment comes from a consensus component: the model is then that
    of the component, and the two deviations (line, circle) are still reported."""
    S, N = ob.points[ids], ob.normals[ids]; model, params, rl, rc = fit_fragment(S); a, b = _extent(S)
    if forced is not None:
        model, params = forced
        if model == "circle": rc = float(((S - torch.tensor(params[:2], dtype=DT)).norm(dim=1) - params[2]).pow(2).mean().sqrt())
    rms = rl if model == "line" else rc
    length = float((a - b).norm()) if model == "line" else min(2 * math.pi, 2 * math.asin(min(1.0, float((a - b).norm()) / (2 * params[2])))) * params[2] if float((a - b).norm()) < 2 * params[2] else math.pi * params[2]
    f = BoundaryFragment(i, ids, S, N, model, params, 0.0 if model == "line" else 1.0 / params[2], length, torch.stack([a, b]), rl, rc, len(ids) / (len(ids) + 4.0) * math.exp(-(rms / (2 * SIGMA)) ** 2)); f.polarity = _polarity(f, ob); return f


def _signed(model, params, X):
    if model == "line": return X.to(DT) @ torch.tensor([math.cos(params[0]), math.sin(params[0])], dtype=DT) - params[1]
    return (X.to(DT) - torch.tensor(params[:2], dtype=DT)).norm(dim=1) - params[2]


def _polarity(f: BoundaryFragment, ob: ObservedBoundary, band=0.45):
    """Side where the label is 1, read from the observations close to the fragment (−1: inside of the circle / negative side)."""
    near = torch.cdist(ob.X.to(DT), f.points).min(1).values < band
    if int(near.sum()) < 6: return None
    s = _signed(f.model, f.params, ob.X[near]); y = ob.y[near].float(); pos, neg = y[s > 0], y[s <= 0]
    if len(pos) < 2 or len(neg) < 2 or abs(float(pos.mean() - neg.mean())) < 0.2: return None
    return 1 if float(pos.mean()) > float(neg.mean()) else -1


# ================================================================ groupings (separately comparable)

def _components_of(A: torch.Tensor) -> torch.Tensor:
    m = len(A); lab = torch.arange(m)
    for _ in range(m):
        new = torch.where(A, lab[None, :].expand(m, m), torch.full((m, m), m)).min(1).values
        if torch.equal(new, lab): break
        lab = new
    return torch.unique(lab, return_inverse=True)[1] if m else lab


def _same_curve(a, b, mode):
    """Can two fragments be the SAME curve? (compatible centre, compatible orientation, or compatible curvature)."""
    if mode in ("center", "shared") and a[0] == b[0] == "circle" and math.hypot(a[1][0] - b[1][0], a[1][1] - b[1][1]) < 0.45 and abs(a[1][2] - b[1][2]) < 0.3: return True
    if mode in ("orientation", "shared") and a[0] == b[0] == "line":
        d = _wrap(a[1][0] - b[1][0]); return (abs(d) < 0.2 and abs(a[1][1] - b[1][1]) < 0.25) or (abs(abs(d) - math.pi) < 0.2 and abs(a[1][1] + b[1][1]) < 0.25)
    return False


def group_points(ob: ObservedBoundary, mode="shared", link=0.45, turn=0.6, seed=0) -> torch.Tensor:
    """Group label per midpoint (−1: unassigned). Mechanisms:
      proximity    single linkage
      tangent      + tangential continuity (neighbouring normals aligned): cuts at the corners
      curvature    tangent fragments merged if a single model (line or circle) fits them together
      center       curved fragments merged if their circles coincide
      orientation  straight fragments merged if they are collinear
      shared       center + orientation + curvature
      consensus    consensus components (circles and lines), first component that contains the point"""
    M, N = ob.points, ob.normals; m = len(M)
    if m == 0: return torch.zeros(0, dtype=torch.long)
    if mode == "proximity": return link_fragments(M, link)
    if mode == "consensus":
        lab = torch.full((m,), -1, dtype=torch.long); comps = sorted(circle_components(ob, seed=seed) + line_components(ob, seed=seed), key=lambda c: -(len(c.inliers) - 5 * c.rms))
        for i, c in enumerate(comps): free = c.inliers[lab[c.inliers] < 0]; lab[free] = i
        return lab
    lab = _components_of((torch.cdist(M, M) < link) & ((N @ N.T).abs() > math.cos(turn)))
    if mode == "tangent": return lab
    groups = [torch.nonzero(lab == g)[:, 0] for g in range(int(lab.max()) + 1)]; fits = [fit_fragment(M[g])[:2] if len(g) >= MIN_POINTS else None for g in groups]; parent = list(range(len(groups)))
    def find(i):
        while parent[i] != i: i = parent[i]
        return i
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if fits[i] is None or fits[j] is None or find(i) == find(j): continue
            ok = _same_curve(fits[i], fits[j], mode)
            if not ok and mode in ("curvature", "shared"):
                S = M[torch.cat([groups[i], groups[j]])]; md, _, rl, rc = fit_fragment(S); ok = (rl if md == "line" else rc) < 1.3 * SIGMA and md == fits[i][0] == fits[j][0]
            if ok: parent[find(j)] = find(i)
    return torch.unique(torch.tensor([find(int(l)) for l in lab]), return_inverse=True)[1]


def extract_fragments(ob: ObservedBoundary, mode="consensus", seed=0) -> List[BoundaryFragment]:
    """Fragments. In `consensus` mode (default), one fragment per consensus component (circle or line): the fragments
    may OVERLAP — a given midpoint remains compatible with a line and with an arc as long as nothing settles it.
    In the other modes, the fragments are the disjoint groups of `group_points`."""
    out = []
    if mode == "consensus":
        for c in sorted(circle_components(ob, seed=seed) + line_components(ob, seed=seed), key=lambda c: -(len(c.inliers) - 5 * c.rms)):
            if len(c.inliers) >= MIN_POINTS: out.append(make_fragment(len(out), c.inliers, ob, ("circle", c.params) if c.kind == "circle" else ("line", c.params)))
    else:
        lab = group_points(ob, mode, seed=seed)
        for g in torch.unique(lab).tolist():
            ids = torch.nonzero(lab == g)[:, 0]
            if g >= 0 and len(ids) >= MIN_POINTS: out.append(make_fragment(len(out), ids, ob))
    for a in out: a.neighbors = [b.index for b in out if b is not a and float(torch.cdist(a.points, b.points).min()) < 0.6]
    return out


# ================================================================ compatibility graph and roles

@dataclass
class FragmentGraph:
    """Nodes = fragments; edges = (i, j, relation, residual). The grouping hypotheses are subgraphs."""
    fragments: List[BoundaryFragment]
    edges: List[Tuple[int, int, str, float]]

    def relations(self, i): return [(j if a == i else a, r, s) for a, j, r, s in self.edges if i in (a, j)]
    def groups(self) -> List[List[int]]:
        """Connected components through strong relations (concentric, parallel, common vertex) or through overlap."""
        parent = list(range(len(self.fragments)))
        def find(i):
            while parent[i] != i: i = parent[i]
            return i
        for a, b, r, _ in self.edges:
            if r != "adjacent": parent[find(b)] = find(a)
        out: Dict[int, List[int]] = {}
        for i in range(len(self.fragments)): out.setdefault(find(i), []).append(i)
        return list(out.values())


def _vertex(a: BoundaryFragment, b: BoundaryFragment):
    """Common vertex of two straight fragments: intersection of the lines, and for each one (is it a RAY starting from the
    vertex?, direction from the vertex to the fragment, gap between the vertex and the nearest endpoint)."""
    a1, b1 = a.params; a2, b2 = b.params; det = math.sin(a2 - a1)
    if abs(det) < 0.3: return None
    V = torch.tensor([(b1 * math.sin(a2) - b2 * math.sin(a1)) / det, (b2 * math.cos(a1) - b1 * math.cos(a2)) / det], dtype=DT); out = []
    for f in (a, b):
        tg = torch.tensor([-math.sin(f.params[0]), math.cos(f.params[0])], dtype=DT); t = (f.points - V) @ tg; side = float((t > 0).float().mean()); one = max(side, 1 - side) >= 0.9; c = f.points.mean(0) - V
        out.append((one, math.atan2(float(c[1]), float(c[0])), float(t.abs().min())))
    return V, out


def build_graph(frags: Sequence[BoundaryFragment]) -> FragmentGraph:
    E = []
    for i, a in enumerate(frags):
        for b in frags[i + 1:]:
            j = b.index
            if b.index in a.neighbors: E.append((i, j, "adjacent", 0.0))
            sa, sb = set(a.ids.tolist()), set(b.ids.tolist()); ov = len(sa & sb) / max(1, min(len(sa), len(sb)))
            if ov >= 0.5: E.append((i, j, "overlap", 1.0 - ov))                                  # same midpoints, two competing readings: a single group
            if not a.straight and not b.straight:
                d = math.hypot(a.params[0] - b.params[0], a.params[1] - b.params[1])
                if d < 0.6 and abs(a.params[2] - b.params[2]) > 0.3: E.append((i, j, "concentric", d))
            elif a.straight and b.straight:
                dd = _wrap(a.params[0] - b.params[0]); b2 = b.params[1] if abs(dd) < math.pi / 2 else -b.params[1]; dd = dd if abs(dd) < math.pi / 2 else _wrap(dd + math.pi)
                if abs(dd) < 0.3 and 0.3 < abs(a.params[1] - b2) < 2.8: E.append((i, j, "parallel", abs(dd)))
                v = _vertex(a, b)
                if v is not None and all(o[0] for o in v[1]) and max(o[2] for o in v[1]) < 0.9 and float(v[0].abs().max()) < 4.5: E.append((i, j, "vertex", max(o[2] for o in v[1])))
    return FragmentGraph(list(frags), E)


@dataclass
class CompatibleFragmentAssignments:
    """Compatible roles of a fragment, weighted. Nothing is settled here."""
    fragment: int
    roles: List[Tuple[str, float]]

    def top(self): return self.roles[0][0] if self.roles else "noise"
    def has(self, role): return any(r == role for r, _ in self.roles)


def assignments(g: FragmentGraph) -> List[CompatibleFragmentAssignments]:
    """Per fragment: line or arc according to the fit (both remain open if the difference is small), inner /
    outer ring edge if it has a concentric partner, band edge if it has a parallel one, sector ray if it
    shares a vertex, possible occluder if it cuts another fragment at its endpoint, noise if it is short or poorly fitted."""
    out = []
    for f in g.fragments:
        w = {}; rl, rc = f.rms_line, f.rms_circle; el, ec = math.exp(-(rl / (2 * SIGMA)) ** 2), (math.exp(-(rc / (2 * SIGMA)) ** 2) * 0.8 if math.isfinite(rc) else 0.0)
        w["line"] = el; w["circle_arc"] = ec if not f.straight or ec > 0.5 * el else 0.0
        for j, r, s in g.relations(f.index):
            o = g.fragments[j]
            if r == "concentric": w["ring_inner" if f.params[2] < o.params[2] else "ring_outer"] = max(w.get("ring_inner" if f.params[2] < o.params[2] else "ring_outer", 0.0), ec * math.exp(-s))
            if r == "parallel": w["band_edge"] = max(w.get("band_edge", 0.0), el * math.exp(-s))
            if r == "vertex": w["sector_ray"] = max(w.get("sector_ray", 0.0), el * math.exp(-s))
            if r == "adjacent" and o.model != f.model: w["occluder"] = max(w.get("occluder", 0.0), 0.5 * f.confidence)       # a corner between two curves of different nature: one may occlude the other
        w["noise"] = 1.0 - f.confidence; tot = sum(w.values()) or 1.0
        out.append(CompatibleFragmentAssignments(f.index, sorted(((k, v / tot) for k, v in w.items() if v > 0), key=lambda x: -x[1])))
    return out


# ================================================================ occlusion ownership

@dataclass
class OcclusionOwnership:
    """Why each point of the EXPECTED boundary of a hypothesis is, or is not, observed."""
    labels: List[str]               # per rendered point: observed | occluded | domain | noise | absent
    owner: Optional[int]            # index of the occluding hypothesis, if a single one explains the disappearance
    fractions: Dict[str, float]

    @property
    def absent(self): return self.fractions["absent"]


def ownership(p: BNode, ob: ObservedBoundary, others: Sequence[BNode] = (), tol=0.3, gap=0.5, n=96) -> OcclusionOwnership:
    """Expected boundary of `p`, point by point:
      observed  an observed midpoint within `tol`
      domain    outside the box of the observations: unobservable
      occluded  ANOTHER hypothesis separates the missing portion from the observed portion (it takes a constant value on the
                missing portion and the other value on the observed portion): the boundary is hidden, not absent
      noise     short hole (less than `gap`) between two observed portions
      absent    missing in an observable zone without explanation — the only legitimate NEGATIVE evidence."""
    X = ob.X; box = (float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 1].min()), float(X[:, 1].max())); P = render(p, n=n, box=(-9.0, 9.0, -9.0, 9.0), reach=6.0)
    if len(P) == 0: return OcclusionOwnership([], None, {k: 0.0 for k in OWNERSHIP})
    inside = (P[:, 0] >= box[0]) & (P[:, 0] <= box[1]) & (P[:, 1] >= box[2]) & (P[:, 1] <= box[3]); seen = (torch.cdist(P, ob.points).min(1).values < tol) if len(ob.points) else torch.zeros(len(P), dtype=torch.bool)
    lab = ["observed" if s else "domain" if not i else "absent" for s, i in zip(seen.tolist(), inside.tolist())]; missing = inside & ~seen; owner = None
    if int(missing.sum()) and int(seen.sum()):
        best = 0.0
        for k, q in enumerate(others):
            vq = predict(q, P.float()).bool();
            for val in (True, False):
                hid = (vq[missing] == val).float().mean(); vis = (vq[seen] != val).float().mean(); s = float(torch.minimum(hid, vis))
                if s > max(best, 0.8): best, owner, own_mask = s, k, (vq == val)
        if owner is not None: lab = ["occluded" if (m and o) else l for l, m, o in zip(lab, missing.tolist(), own_mask.tolist())]
    if int(seen.sum()):                                                           # short holes: sampling noise, not an absence
        d = torch.cdist(P, P[seen]).min(1).values; lab = ["noise" if (l == "absent" and float(x) < gap) else l for l, x in zip(lab, d)]
    return OcclusionOwnership(lab, owner, {k: lab.count(k) / len(lab) for k in OWNERSHIP})


# ================================================================ type hypotheses

@dataclass
class TypeHypothesis:
    family: str
    type: str
    program: BNode
    fragments: Tuple[int, ...]
    signals: Dict[str, float] = field(default_factory=dict)        # SEPARATE signals (see `signals_of`)
    score: float = 0.0
    confidence: float = 0.0                                          # share of the score within its group (softmax)


def _circle_of(S, f: BoundaryFragment):
    c, r, rms = circle_geometric(S, f.params[:2], f.params[2]) if not f.straight else (None, None, None)
    return c, r, rms


def signals_of(p: BNode, ids: torch.Tensor, ob: ObservedBoundary, coherence=0.0, others: Sequence[BNode] = (), negative="ownership") -> Dict[str, float]:
    """Explicit signals of a candidate type, reported separately:
      geometric   root-mean-square deviation of the midpoints of the group from the boundary of the program
      coherence   residual of the shared relation (distance between the centres, angle gap, gap at the vertex)
      support     number of observed midpoints within 0.15 of the boundary of the program
      explained   z contrast on the seen stretches of the boundary
      seen        share of the expected boundary that is observed
      absent      share of the expected boundary that is missing WITHOUT explanation (negative evidence) — according to `negative`:
                  "ownership" (hidden zones, out of domain and short holes excluded), "naive" (everything that is missing), "none"
      complexity  number of parameters
      stability   functional disagreement between the fits on two halves of the midpoints (None if not computable)"""
    S = ob.points[ids]; B = render(p, n=160, box=(-9.0, 9.0, -9.0, 9.0), reach=6.0); geo = float(torch.cdist(S, B).min(1).values.pow(2).mean().sqrt()) if len(B) and len(S) else float("inf"); z, vf = visible_score(p, ob.pairs)
    own = ownership(p, ob, others); absent = 0.0 if negative == "none" else own.absent if negative == "ownership" else 1.0 - own.fractions["observed"] - own.fractions["domain"]
    from core.deep_partial import refit_visible
    a, b = (refit_visible(p, S[0::2]), refit_visible(p, S[1::2])) if len(S) >= 10 else (None, None); stab = float((predict(a, ob.X.float()) != predict(b, ob.X.float())).float().mean()) if a is not None and b is not None else None
    sup = float((torch.cdist(ob.points, B).min(1).values < 0.15).sum()) if len(B) and len(ob.points) else 0.0
    return {"geometric": geo, "coherence": coherence, "support": sup, "explained": z, "seen": own.fractions["observed"], "absent": absent, "occluded": own.fractions["occluded"], "complexity": float(p.n_params()), "stability": stab}


def fuse(s: Dict[str, float], cost=2.0) -> float:
    """Explicit fusion, of description-length type: boundary midpoints closely explained (within 0.15),
    reduced by the share of expected boundary that is missing without explanation, minus a cost per parameter. A ring
    (4 parameters) wins over a disc (3) as soon as it explains a few more midpoints; two discs cost 6."""
    return s["support"] * (1.0 - s["absent"]) - cost * s["complexity"]


def _disc_types(f: BoundaryFragment, ob):
    """Disc or anti-disc: the polarity read from the observations decides the order; both remain proposed."""
    return ("disc", "antidisc") if f.polarity in (-1, None) else ("antidisc", "disc")


def type_hypotheses(g: FragmentGraph, ob: ObservedBoundary, negative="ownership", hierarchical=True) -> List[TypeHypothesis]:
    """fragment geometry → primitive family → specific primitive type.

    Level 1 (geometry): straight / curved. Level 2 (family): alone, or linked by a relation (parallel, concentric,
    common vertex). Level 3 (type): polarity for disc / anti-disc. The parameters are fitted on the midpoints of the
    group, with the shared constraint of the family. With `hierarchical=False`, the family is not restricted by
    the geometry: each fragment is proposed both as a line and as a circle (flat classification by score)."""
    M = ob.points; raw: List[Tuple[str, str, BNode, Tuple[int, ...], float]] = []
    for f in g.fragments:
        S = f.points
        if f.straight or not hierarchical:
            n, b, _ = line_tls(S); raw.append(("line-like", "half", orient(build(SKELETONS["half"], None, (math.atan2(float(n[1]), float(n[0])), b)), ob.X, ob.y), (f.index,), 0.0))
        if not f.straight or not hierarchical:
            fit = (torch.tensor(f.params[:2], dtype=DT), f.params[2]) if not f.straight else circle_taubin(S) if len(S) >= 5 else None
            if fit is not None and 0.2 <= fit[1] <= 3.5:
                c, r, _ = circle_geometric(S, fit[0], fit[1])
                if 0.2 <= r <= 3.5: raw += [("circle-like", t, build(SKELETONS[t], c.tolist(), (r,)), (f.index,), 0.0) for t in _disc_types(f, ob)]
    for i, j, rel, res in g.edges:
        a, b = g.fragments[i], g.fragments[j]
        if rel == "concentric":
            fit = concentric_fit(a.points, b.points)
            if fit is not None: raw.append(("paired-circles", "ring", build(SKELETONS["ring"], fit[0].tolist(), sorted(fit[1:])), (i, j), res))
        elif rel == "parallel":
            fit = parallel_fit(a.points, b.points)
            if fit is not None: n, b1, b2 = fit; raw.append(("paired-lines", "band", build(SKELETONS["band"], None, (math.atan2(float(n[1]), float(n[0])), min(b1, b2), max(b1, b2))), (i, j), res))
        elif rel == "vertex":
            V, o = _vertex(a, b); w = _wrap(o[1][1] - o[0][1]) / 2
            if 0.15 < abs(w) < 1.5: raw.append(("radial-composite", "sector", build(SKELETONS["sector"], V.tolist(), (_wrap(o[0][1] + w), abs(w))), (i, j), res))
    progs = [r[2] for r in raw]; out = []
    for k, (fam, t, p, fr, res) in enumerate(raw):
        ids = torch.cat([g.fragments[i].ids for i in fr]); s = signals_of(p, ids, ob, res, [q for u, q in enumerate(progs) if u != k and not set(raw[u][3]) & set(fr)], negative); out.append(TypeHypothesis(fam, t, p, fr, s, fuse(s)))
    return sorted(out, key=lambda h: -h.score)


def _support(h: TypeHypothesis, M: torch.Tensor, tol=0.25) -> torch.Tensor:
    B = render(h.program, n=160, box=(-9.0, 9.0, -9.0, 9.0), reach=6.0); return (torch.cdist(M, B).min(1).values < tol) if len(B) and len(M) else torch.zeros(len(M), dtype=torch.bool)


def rank_by_group(hyps: Sequence[TypeHypothesis], g: FragmentGraph, k=1, ob: Optional[ObservedBoundary] = None, min_new=4) -> List[TypeHypothesis]:
    """Top-k types PER GROUP of midpoints. The groups are not fixed in advance: the best hypothesis takes the
    midpoints it explains; a subsequent hypothesis opens a NEW group only if at least half of its
    midpoints (and at least `min_new`) remain unexplained; otherwise it is a COMPETING reading of the group it
    overlaps most, kept if the group has fewer than k types. The confidence is the softmax share of the score within the group.
    No top-1 is forced: with k > 1, several compatible types remain open for the composition."""
    M = ob.points if ob is not None else torch.zeros(0, 2, dtype=DT); hs = sorted((h for h in hyps if h.score > 0), key=lambda h: -h.score); explained = torch.zeros(len(M), dtype=torch.bool); groups: List[List[TypeHypothesis]] = []; sups: List[torch.Tensor] = []
    for h in hs:
        s = _support(h, M); n = int(s.sum()); new = int((s & ~explained).sum())
        if n >= min_new and new >= max(min_new, 0.5 * n): groups.append([h]); sups.append(s); explained |= s; continue
        if not groups or n == 0: continue
        j = max(range(len(groups)), key=lambda j: int((s & sups[j]).sum()))
        if len(groups[j]) < k and int((s & sups[j]).sum()) >= 0.5 * n and all(repr(h.program) != repr(o.program) for o in groups[j]): groups[j].append(h)
    out = []
    for grp in groups:
        z = torch.tensor([h.score for h in grp], dtype=DT); w = torch.softmax((z - z.max()) / 3.0, 0).tolist()
        for h, c in zip(grp, w): h.confidence = c
        out += grp
    return sorted(out, key=lambda h: -h.score)


# ================================================================ attribution ↔ type loop (bounded, instrumented)

@dataclass
class Inference:
    fragments: List[BoundaryFragment]
    graph: FragmentGraph
    hypotheses: List[TypeHypothesis]        # all of them, ranked
    selected: List[TypeHypothesis]          # top-k per group
    log: List[dict]
    stats: Dict[str, float]


def reassign(ob: ObservedBoundary, hyps: Sequence[TypeHypothesis], tol=0.25) -> torch.Tensor:
    """Re-attribution of the midpoints in the light of the types: each midpoint goes to the hypothesis (positive score) whose
    fitted boundary is the closest, if it is within `tol`. The type corrects the attribution."""
    M = ob.points; lab = torch.full((len(M),), -1, dtype=torch.long); keep = [h for h in hyps if h.score > 0]
    if not keep or not len(M): return lab
    D = torch.stack([torch.cdist(M, render(h.program, n=160, box=(-9.0, 9.0, -9.0, 9.0), reach=6.0)).min(1).values for h in keep], 1); d, i = D.min(1); lab[d < tol] = i[d < tol]; return lab


def infer_types(ob: ObservedBoundary, k=1, mode="consensus", negative="ownership", hierarchical=True, iters=1, seed=0) -> Inference:
    """fragment assignment → type hypothesis → parameter fit → re-score fragment assignment (at most `iters` passes).

    At each pass, the midpoints are re-attributed to the best hypotheses per group, the fragments are rebuilt
    on this attribution and the types re-inferred; the pass is kept only if the total score of the selected
    hypotheses increases (otherwise the previous state is kept)."""
    import time
    t0 = time.time(); frags = extract_fragments(ob, mode, seed); g = build_graph(frags); H = type_hypotheses(g, ob, negative, hierarchical); sel = rank_by_group(H, g, k, ob); total = sum(h.score for h in rank_by_group(H, g, 1, ob)); log = [{"iteration": 0, "fragments": len(frags), "hypotheses": len(H), "score": total, "action": "init"}]
    for it in range(1, iters + 1):
        lab = reassign(ob, rank_by_group(H, g, 1, ob)); f2 = [make_fragment(n, torch.nonzero(lab == v)[:, 0], ob) for n, v in enumerate(v for v in torch.unique(lab).tolist() if v >= 0 and int((lab == v).sum()) >= MIN_POINTS)]
        for n, f in enumerate(f2): f.index = n
        for a in f2: a.neighbors = [b.index for b in f2 if b is not a and float(torch.cdist(a.points, b.points).min()) < 0.6]
        g2 = build_graph(f2); H2 = type_hypotheses(g2, ob, negative, hierarchical); t2 = sum(h.score for h in rank_by_group(H2, g2, 1, ob)); ok = t2 > total + 1e-9
        log.append({"iteration": it, "fragments": len(f2), "hypotheses": len(H2), "score": t2, "action": "commit" if ok else "rollback"})
        if not ok: break
        frags, g, H, total = f2, g2, H2, t2; sel = rank_by_group(H, g, k, ob)
    return Inference(frags, g, H, sel, log, {"fragments": len(frags), "groups": len(g.groups()), "edges": len(g.edges), "types_tested": len(H), "seconds": round(time.time() - t0, 3)})


def typed_labels(ob: ObservedBoundary, seed=0) -> torch.Tensor:
    """Attribution AFTER type inference: each midpoint goes to the best hypothesis per group that explains it."""
    inf = infer_types(ob, 1, seed=seed); return reassign(ob, inf.selected)


def flat_types(ob: ObservedBoundary, k=1, seed=0) -> List[TypeHypothesis]:
    """FLAT classification, without fragments: each type is tried on the whole scene (known-type inference repeated
    for the six types), top-k per type. Serves as a comparison for the hierarchy."""
    out = []
    for t, sk in SKELETONS.items():
        for h in infer(sk, ob, k, seed=seed, regions=False): out.append(TypeHypothesis({v: f for f, vs in FAMILY_TYPES.items() for v in vs}[t], t, h.program, (), {"explained": h.score, "complexity": float(h.program.n_params())}, h.score - 0.5 * h.program.n_params()))
    return sorted(out, key=lambda h: -h.score)


def literals(hyps: Sequence[TypeHypothesis]) -> List[BNode]:
    """Programs to give to the composition (deduplicated); a half-plane is given with its complement."""
    out = []
    for h in hyps:
        out.append(h.program)
        if atom_name(h.program) == "dot.le": out.append(BNode("pred", "dot", "le", (_wrap(h.program.params[0] + math.pi), -h.program.params[1])))
    seen = set(); return [p for p in out if not (repr(p) in seen or seen.add(repr(p)))]

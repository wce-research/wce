"""FACTORIZED compositional generation of deep candidates (Deep Candidate Generation sub-milestone).

Finding of the diagnostic: the space of structures grows as 8·108·2,600·80,000·2.8 M·10⁸ (depths 1 to 6) whereas
fit-based search generates ≈ 150 of them; and the "one atom at a time" operators only build chains.

Idea: separate GEOMETRY from LOGIC.
  1. Geometry — in an and / or / xor composition, every piece of boundary belongs to the boundary of ONE atom.
     The atomic boundaries (lines, circles) can therefore be read locally, without knowing the composition: boundary
     points → robust consensus (RANSAC-like) → geometric primitives → literals (typed regions).
  2. Logic — once the literals are fixed, a structure is a Boolean formula over their SIGNATURES (values on
     the observed points). Compositions are built bottom-up through signature tables (deduplicated),
     the requirements descend as typed HOLES (target + mask of the points that matter), and the meeting takes
     place on the common signature, through a bilinear product that evaluates all pairs at once. No gradient
     fit is needed before having chosen the few candidates to polish.

This module imports no world module and never receives the truth.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from core.boundary_program import BNode, predict
from core.deep_structure import COMBINATORS, algebraic_form, n_atoms, normalize

F32 = torch.float32
OPS = COMBINATORS                                   # ("and", "or", "xor")

# ---- lightweight structural types ------------------------------------------------------------------------------------
# Field   : scalar function of space (dot, norm, angle)
# Region  : predicate on space (a thresholded Field, possibly transported) — this is the type of the literals
# Logic   : Region × Region → Region (and, or, xor)
# Transform : Region → Region (translate); applies only to a centered Field (norm, angle), never to dot
TYPES = ("Field", "Region", "Logic", "Transform")
TYPE_RULES = {"pred": ("Field", "Region"), "translate": ("Region", "Region"), "not": ("Region", "Region"), "and": ("Region×Region", "Region"), "or": ("Region×Region", "Region"), "xor": ("Region×Region", "Region")}


def well_typed(n: BNode) -> bool:
    """Typing rules: a combinator takes two Regions; translate only transports a centered field (norm, angle)
    or its negation; not applies to an atomic Region; no translate∘translate nor not∘not."""
    if n.kind == "pred": return n.feat in ("dot", "norm", "angle") and n.op in ("le", "ge", "in") and not n.children
    if n.kind in OPS: return len(n.children) == 2 and all(well_typed(c) for c in n.children)
    if n.kind == "translate":
        c = n.children[0]; c = c.children[0] if c.kind == "not" else c; return c.kind == "pred" and c.feat in ("norm", "angle")
    if n.kind == "not": c = n.children[0]; return c.kind == "pred"
    return False


# ================================================================ 1. geometry: atomic boundaries

def boundary_points(X, y, k=6):
    """Boundary points: midpoints of the pairs of neighbors with opposite labels (denoised by local vote)."""
    X = X.to(F32); n = len(X); D = torch.cdist(X, X); nb = D.topk(min(k + 1, n), largest=False).indices                    # [n, k+1] (self included)
    yh = (y.to(F32)[nb].mean(1) >= 0.5)                                                                                    # local vote
    opp = yh[nb] != yh[:, None]; d = torch.where(opp, D.gather(1, nb), torch.full_like(D.gather(1, nb), float("inf")))
    j = d.argmin(1); has = torch.isfinite(d.min(1).values); i = torch.arange(n)[has]; jj = nb[i, j[has]]
    pairs = torch.stack([torch.minimum(i, jj), torch.maximum(i, jj)], 1).unique(dim=0)
    return (X[pairs[:, 0]] + X[pairs[:, 1]]) / 2, X[pairs[:, 0]], X[pairs[:, 1]]


def _line_hyps(B, n_hyp, g):
    i = torch.randint(0, len(B), (n_hyp, 2), generator=g); p, q = B[i[:, 0]], B[i[:, 1]]; t = q - p; ok = t.norm(dim=1) > 0.3
    nrm = torch.stack([-t[:, 1], t[:, 0]], 1); nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return nrm[ok], (nrm * p).sum(1)[ok]


def _circle_hyps(B, n_hyp, g, rmin=0.35, rmax=2.2):
    i = torch.randint(0, len(B), (n_hyp, 3), generator=g); a, b, c = B[i[:, 0]], B[i[:, 1]], B[i[:, 2]]
    d = 2 * (a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1]) + c[:, 0] * (a[:, 1] - b[:, 1]))
    sa, sb, sc = (a ** 2).sum(1), (b ** 2).sum(1), (c ** 2).sum(1)
    ux = (sa * (b[:, 1] - c[:, 1]) + sb * (c[:, 1] - a[:, 1]) + sc * (a[:, 1] - b[:, 1])) / d
    uy = (sa * (c[:, 0] - b[:, 0]) + sb * (a[:, 0] - c[:, 0]) + sc * (b[:, 0] - a[:, 0])) / d
    ctr = torch.stack([ux, uy], 1); r = (a - ctr).norm(dim=1); ok = (d.abs() > 1e-3) & (r > rmin) & (r < rmax) & torch.isfinite(r)
    return ctr[ok], r[ok]


def _refine_line(P):
    m = P.mean(0); c = P - m; _, _, V = torch.linalg.svd(c, full_matrices=False); nrm = V[-1]; return nrm, float(nrm @ m)


def _refine_circle(P):
    """Algebraic fit (Kåsa): x² + y² = 2ax + 2by + c."""
    A = torch.cat([2 * P, torch.ones(len(P), 1)], 1); sol = torch.linalg.lstsq(A, (P ** 2).sum(1, keepdim=True)).solution.flatten()
    ctr = sol[:2]; r = math.sqrt(max(1e-6, float(sol[2] + ctr @ ctr))); return ctr, r


@dataclass
class GeoPrimitive:
    kind: str                       # "line" | "circle"
    params: tuple                   # line: (angle, offset); circle: (cx, cy, r)
    inliers: int
    support: torch.Tensor           # boundary points explained


def _sep_line(P, Q, M, nrm, b, delta):
    """A boundary pair supports a line if its two points lie on either side of it AND its midpoint is close to it."""
    return (((P @ nrm.T - b) * (Q @ nrm.T - b)) < 0) & ((M @ nrm.T - b).abs() < delta)


def _sep_circle(P, Q, M, ctr, r, delta):
    return (((torch.cdist(P, ctr) - r) * (torch.cdist(Q, ctr) - r)) < 0) & ((torch.cdist(M, ctr) - r).abs() < delta)


def detect_primitives(boundary, max_prims=10, min_inliers=6, delta=0.22, n_hyp=1500, seed=0) -> List[GeoPrimitive]:
    """Sequential robust consensus: the best line or the best circle in terms of the number of boundary
    PAIRS it SEPARATES (explicit constraint: an atomic boundary passes between two points of opposite
    labels), refined by least squares, removed, and so on."""
    B, P, Q = (t.to(F32) for t in boundary); g = torch.Generator().manual_seed(seed); out: List[GeoPrimitive] = []; alive = torch.ones(len(B), dtype=torch.bool)
    while len(out) < max_prims and int(alive.sum()) >= min_inliers:
        R, Pa, Qa = B[alive], P[alive], Q[alive]
        if len(R) < 3: break
        ln, lb = _line_hyps(R, n_hyp, g); cc, cr = _circle_hyps(R, n_hyp, g); best = None
        if len(ln):
            cnt = _sep_line(Pa, Qa, R, ln, lb[None], delta).sum(0); i = int(cnt.argmax()); best = ("line", i, int(cnt[i]))
        if len(cc):
            cnt = _sep_circle(Pa, Qa, R, cc, cr[None], delta).sum(0); i = int(cnt.argmax())
            if best is None or int(cnt[i]) > best[2] + 1: best = ("circle", i, int(cnt[i]))                              # on a tie, the line (simpler)
        if best is None or best[2] < min_inliers: break
        if best[0] == "line":
            nrm, b = ln[best[1]][None], lb[best[1]].reshape(1, 1); inl = _sep_line(Pa, Qa, R, nrm, b, delta)[:, 0]
            if int(inl.sum()) >= 3:
                n2, b2 = _refine_line(R[inl]); i2 = _sep_line(Pa, Qa, R, n2[None], torch.tensor([[b2]]), delta)[:, 0]
                if int(i2.sum()) >= int(inl.sum()): nrm, b, inl = n2[None], torch.tensor([[b2]]), i2
            prim = GeoPrimitive("line", (math.atan2(float(nrm[0, 1]), float(nrm[0, 0])), float(b)), int(inl.sum()), R[inl])
        else:
            c, r = cc[best[1]][None], cr[best[1]].reshape(1, 1); inl = _sep_circle(Pa, Qa, R, c, r, delta)[:, 0]
            if int(inl.sum()) >= 4:
                c2, r2 = _refine_circle(R[inl]); i2 = _sep_circle(Pa, Qa, R, c2[None], torch.tensor([[r2]]), delta)[:, 0]
                if 0.3 < r2 < 2.6 and int(i2.sum()) >= int(inl.sum()): c, r, inl = c2[None], torch.tensor([[r2]]), i2
            prim = GeoPrimitive("circle", (float(c[0, 0]), float(c[0, 1]), float(r)), int(inl.sum()), R[inl])
        if prim.inliers < min_inliers: break
        out.append(prim); idx = alive.nonzero().flatten(); alive[idx[inl]] = False
    return out


def discover_atoms_by_separation(atom_skeletons: Sequence[BNode], X, y, rounds=3, n_init=4, steps=50, keep_per_round=4, seed=0, counter=None, min_pairs=4.0):
    """Atom discovery by SEQUENTIAL COVERING of the boundary pairs: at each round, all atom types
    are fitted (in batch) to separate the pairs still unexplained; the best distinct atoms are kept and
    the pairs they separate are down-weighted. Returns (fitted atoms, statistics)."""
    from core.boundary_program import _inits
    from core.deep_fit import DT, separation_fit
    M, P, Q = boundary_points(X, y); stats = {"pairs": len(M), "rounds": 0, "atoms": 0, "covered": 0.0}
    if len(M) < 6: return [], stats
    w = torch.ones(len(M), dtype=DT); out: List[BNode] = []; g = torch.Generator().manual_seed(seed); Xf, yl = X.float(), y.long()
    for rnd in range(rounds):
        cands = []
        for a in atom_skeletons:
            inits = torch.stack([v.to(DT) for v in _inits(a, Xf, yl, n_init, g)])
            c = M[torch.multinomial(w.float().clamp_min(1e-6), n_init, replacement=True, generator=g)]                    # centers seeded on boundary points still unexplained
            if a.kind == "translate": inits[:, :2] = c.to(DT) + 0.6 * torch.randn(n_init, 2, generator=g, dtype=DT)
            th, sep = separation_fit(a, P, Q, w, inits, steps=steps, counter=counter)
            cands += [(float(sep[k]), a.with_params(th[k].tolist())) for k in range(n_init)]
        cands.sort(key=lambda c: -c[0]); kept = 0
        for score, prog in cands:
            if kept >= keep_per_round or score < min_pairs: break
            s = (predict(prog, P) != predict(prog, Q))
            if float((w * s.to(DT)).sum()) < min_pairs: continue                                                          # already covered by an atom kept in this round
            out.append(prog); w = torch.where(s, w * 0.2, w); kept += 1
        stats["rounds"] = rnd + 1
        if kept == 0: break
    stats["atoms"] = len(out); stats["covered"] = float(1 - w.mean()) / 0.8
    return out, stats


# ================================================================ literals (typed regions) derived from the primitives

@dataclass
class Literal:
    program: BNode                  # grammar atom, geometric parameters
    source: str                     # "half" | "disc" | "antidisc" | "band" | "notband" | "ring" | "sector" | "notsector"
    support: int


def _wrap(a): return math.atan2(math.sin(a), math.cos(a))
def _T(c, ch): return BNode("translate", params=(float(c[0]), float(c[1])), children=(ch,))


def literals_from(prims: Sequence[GeoPrimitive], macros=True, max_sectors=4) -> List[Literal]:
    """Literals = atoms of the grammar. Minimal basis: half-planes (two orientations) and discs (+ complement).
    Macros (redundant atoms of the grammar, recognized by explicit CONSTRAINTS): band = two parallel lines;
    ring = two concentric circles; sector = two rays issuing from the same point."""
    L: List[Literal] = []; lines = [p for p in prims if p.kind == "line"]; circles = [p for p in prims if p.kind == "circle"]
    for p in lines:
        a, b = p.params; L.append(Literal(BNode("pred", "dot", "le", (a, b)), "half", p.inliers)); L.append(Literal(BNode("pred", "dot", "le", (_wrap(a + math.pi), -b)), "half", p.inliers))
    for p in circles:
        cx, cy, r = p.params; L.append(Literal(_T((cx, cy), BNode("pred", "norm", "le", (r,))), "disc", p.inliers)); L.append(Literal(_T((cx, cy), BNode("pred", "norm", "ge", (r,))), "antidisc", p.inliers))
    if not macros: return L
    for i, p in enumerate(lines):                                       # band: parallelism constraint
        for q in lines[i + 1:]:
            a1, b1 = p.params; a2, b2 = q.params; da = _wrap(a1 - a2)
            if abs(da) > math.pi / 2: a2, b2 = _wrap(a2 + math.pi), -b2; da = _wrap(a1 - a2)
            if abs(da) < 0.2 and 0.3 < abs(b1 - b2) < 2.4:
                lo, hi = sorted((b1, b2)); band = BNode("pred", "dot", "in", (a1, lo, hi)); s = min(p.inliers, q.inliers)
                L.append(Literal(band, "band", s)); L.append(Literal(BNode("not", children=(band,)), "notband", s))
    for i, p in enumerate(circles):                                     # ring: concentricity constraint
        for q in circles[i + 1:]:
            c1, c2 = torch.tensor(p.params[:2]), torch.tensor(q.params[:2]); r1, r2 = sorted((p.params[2], q.params[2]))
            if float((c1 - c2).norm()) < 0.3 and r2 - r1 > 0.3: L.append(Literal(_T(((c1 + c2) / 2).tolist(), BNode("pred", "norm", "in", (r1, r2))), "ring", min(p.inliers, q.inliers)))
    sect = []
    for i, p in enumerate(lines):                                       # sector: two rays with the same origin, each supported on one side only
        for q in lines[i + 1:]:
            a1, b1 = p.params; a2, b2 = q.params; det = math.cos(a1) * math.sin(a2) - math.sin(a1) * math.cos(a2)
            if abs(det) < 0.25: continue
            ap = torch.tensor([(b1 * math.sin(a2) - b2 * math.sin(a1)) / det, (b2 * math.cos(a1) - b1 * math.cos(a2)) / det])
            if float(ap.abs().max()) > 3.2: continue
            rays = []
            for prim, a in ((p, a1), (q, a2)):
                t = torch.tensor([-math.sin(a), math.cos(a)]); s = (prim.support - ap) @ t; pos, neg = int((s > 0.15).sum()), int((s < -0.15).sum())
                if max(pos, neg) >= 4 and min(pos, neg) <= 0.25 * max(pos, neg): rays.append((math.atan2(float(t[1]), float(t[0])) if pos >= neg else math.atan2(-float(t[1]), -float(t[0])), max(pos, neg)))
            if len(rays) == 2:
                th1, th2 = rays[0][0], rays[1][0]; half = abs(_wrap(th1 - th2)) / 2; ctr = _wrap(th2 + _wrap(th1 - th2) / 2)
                if 0.2 < half < 1.45: sect.append((min(rays[0][1], rays[1][1]), ap.tolist(), ctr, half))
    for s, ap, ctr, half in sorted(sect, key=lambda z: -z[0])[:max_sectors]:
        sec = BNode("pred", "angle", "in", (ctr, half)); L.append(Literal(_T(ap, sec), "sector", s)); L.append(Literal(_T(ap, BNode("not", children=(sec,))), "notsector", s))
    return L


# ================================================================ 2. logic: signature tables, typed holes, meeting

@dataclass
class Table:
    """Level j: structures with j literals, deduplicated by SIGNATURE (values on the observed points).
    `alts` keeps alternative forms with the same signature (do not merge different structures too early)."""
    sig: torch.Tensor                       # [N, n] float32 0/1
    params: torch.Tensor                    # [N] number of parameters
    trees: List[tuple]                      # tree of each entry: ("lit", i) | (op, left, right)
    alts: Dict[int, List[tuple]] = field(default_factory=dict)
    merged: int = 0                         # structures merged by identical signature
    degenerate: int = 0                     # degenerate compositions discarded (incompatible interface)

    def __len__(self): return len(self.trees)


def _combine(op, a, b): return a * b if op == "and" else a + b - a * b if op == "or" else a + b - 2 * a * b


def _dedup(sig, params, trees, known: Sequence[torch.Tensor] = (), keep_alts=2):
    """Deduplication by signature (and elimination of the constants and of the signatures already known at a lower
    level: a composition that changes nothing is not a new structure)."""
    n = sig.shape[1]; w = torch.arange(1, n + 1, dtype=torch.float64); r = torch.Generator().manual_seed(12345); w2 = torch.rand(n, generator=r, dtype=torch.float64)
    h = (sig.double() @ w) * 1e3 + (sig.double() @ w2); s = sig.sum(1); const = (s == 0) | (s == n)
    kh = torch.cat([(k.double() @ w) * 1e3 + (k.double() @ w2) for k in known]) if len(known) else torch.zeros(0, dtype=torch.float64)
    seen: Dict[float, int] = {float(v): -1 for v in kh.tolist()}; keep = []; alts: Dict[int, List[tuple]] = {}; merged = 0; deg = int(const.sum())
    order = torch.argsort(params, stable=True).tolist()                                # the shortest description represents the signature
    for i in order:
        if bool(const[i]): continue
        v = float(h[i]); j = seen.get(v)
        if j is None: seen[v] = len(keep); keep.append(i)
        elif j == -1: deg += 1
        else:
            merged += 1
            if len(alts.setdefault(j, [])) < keep_alts: alts[j].append(trees[i])
    idx = torch.tensor(keep, dtype=torch.long)
    return Table(sig[idx], params[idx], [trees[i] for i in keep], alts, merged, deg)


def build_tables(lit_sig, lit_params, max_level3=40000) -> Tuple[List[Table], dict]:
    """Bottom-up tables S1, S2, S3 (all compositions, deduplicated by signature)."""
    L = len(lit_sig); S1 = _dedup(lit_sig, lit_params, [("lit", i) for i in range(L)]); stats = {"literals": L, "S1": len(S1), "attempted": 0}
    tabs = [S1]
    for lvl in (2, 3):
        A = tabs[-1]; sig, par, trees = [], [], []
        for op in OPS:
            s = _combine(op, A.sig[:, None, :], S1.sig[None, :, :]).reshape(-1, A.sig.shape[1]); sig.append(s)
            par.append((A.params[:, None] + S1.params[None, :]).reshape(-1)); trees += [(op, a, b) for a in A.trees for b in S1.trees]
        stats["attempted"] += len(trees); T = _dedup(torch.cat(sig), torch.cat(par), trees, [t.sig for t in tabs])
        if len(T) > max_level3: idx = torch.arange(max_level3); T = Table(T.sig[idx], T.params[idx], T.trees[:max_level3], {k: v for k, v in T.alts.items() if k < max_level3}, T.merged, T.degenerate); stats["capped"] = True
        tabs.append(T); stats[f"S{lvl}"] = len(T); stats[f"merged{lvl}"] = T.merged; stats[f"degenerate{lvl}"] = T.degenerate
    return tabs, stats


def pair_errors(A_sig, B_sig, y, w):
    """Meeting on the common signature: weighted error of op(a, b) against the target, for ALL pairs (a, b)
    and the three combinators, through a single matrix product. Returns E [3, |A|, |B|] in the order (and, or, xor)."""
    v = w * (1 - 2 * y); c0 = float((w * y).sum()); M = (A_sig * v) @ B_sig.T; a = (A_sig @ v)[:, None]; b = (B_sig @ v)[None, :]
    return torch.stack([c0 + M, c0 + a + b - M, c0 + a + b - 2 * M])


@dataclass
class Hole:
    """Typed hole of a partial program: what the missing substructure MUST do."""
    type: str                       # always "Region" in this grammar
    target: torch.Tensor            # required behavior on the observed points
    care: torch.Tensor              # weight: 0 = point irrelevant for this hole
    depth_left: int
    unavoidable: float              # errors that no filling can correct (lower bound)


@dataclass
class PartialStructure:
    """Partial program `op(known, ?X)`: a known part and a typed hole."""
    op: str
    known: tuple
    hole: Hole

    def lower_bound(self): return self.hole.unavoidable


def make_hole(op: str, known_sig, y, w, depth_left) -> Hole:
    """Top-down requirement: for op(known, ?X) = y, what must ?X be and where does it matter?"""
    if op == "xor": return Hole("Region", (y - known_sig).abs(), w, depth_left, 0.0)
    if op == "or": return Hole("Region", y, w * (1 - known_sig), depth_left, float((w * known_sig * (1 - y)).sum()))
    return Hole("Region", y, w * known_sig, depth_left, float((w * (1 - known_sig) * y).sum()))


def _evidence(err, n, params, atoms):
    """Same form as the existing approximate evidence (profiled noise, parameter and size penalty) — used
    here as a sieve BEFORE fitting; the final ranking remains that of the system."""
    eps = ((err + 0.5) / (n + 1.0)).clamp(1e-3, 0.49); ll = err * eps.log() + (n - err) * (1 - eps).log()
    return ll - 0.5 * params * math.log(max(1.0, n)) - (2 * atoms - 1) * 3.0 * math.log(2)


def _top_pairs(E, pa, pb, n, atoms, top, chunk_rows=None):
    """Best pairs (evidence score) of an error tensor [3, A, B] → [(score, error, op, i, j)]."""
    ev = _evidence(E, n, (pa[:, None] + pb[None, :])[None], atoms); k = min(top, ev.numel()); v, idx = ev.flatten().topk(k)
    A, B = E.shape[1], E.shape[2]; out = []
    for s, f in zip(v.tolist(), idx.tolist()):
        o, r = divmod(f, A * B); i, j = divmod(r, B); out.append((s, float(E[o, i, j]), OPS[o], i, j))
    return out


class SignatureSynthesizer:
    """Synthesis of structures on signatures: bottom-up (tables), top-down (holes), meeting (bilinear product)."""

    def __init__(self, lit_sig, lit_params, y, max_depth=6, top=6, hole_beam=8, time_budget=None):
        self.y = y.to(F32); self.n = float(len(y)); self.w = torch.ones(len(y), dtype=F32); self.max_depth, self.top, self.hole_beam = max_depth, top, hole_beam
        self.t0 = time.time(); self.time_budget = time_budget; self.tabs, self.stats = build_tables(lit_sig.to(F32), lit_params.to(F32)); self.stats["pairs_scored"] = 0; self.stats["holes"] = 0; self.stats["holes_pruned"] = 0
        self.S1, self.S2, self.S3 = self.tabs

    def _late(self): return self.time_budget is not None and time.time() - self.t0 > self.time_budget

    def _meet(self, A: Table, B: Table, y, w, atoms, top, chunk=4000):
        """All the compositions op(a ∈ A, b ∈ B) against (target, mask): the `top` best."""
        best = []
        for s in range(0, len(A), chunk):
            E = pair_errors(A.sig[s:s + chunk], B.sig, y, w); self.stats["pairs_scored"] += E.numel()
            for sc, e, op, i, j in _top_pairs(E, A.params[s:s + chunk], B.params, float(w.sum()), atoms, top): best.append((sc, e, (op, A.trees[s + i], B.trees[j]), float(A.params[s + i] + B.params[j])))
        best.sort(key=lambda z: -z[0]); return best[:top]

    def solve(self, y, w, depth, top):
        """Best structures with exactly `depth` literals for (target, mask). Depth ≤ 4: exact meeting
        of tables. Depth 5–6: meeting of tables + partial programs with a hole, pruned by lower bound."""
        n = float(w.sum())
        if depth == 1:
            e = (w * (self.S1.sig - y).abs()).sum(1); ev = _evidence(e, n, self.S1.params, 1); k = min(top, len(e)); v, i = ev.topk(k)
            return [(float(s), float(e[j]), self.S1.trees[j], float(self.S1.params[j])) for s, j in zip(v.tolist(), i.tolist())]
        if depth == 2: return self._meet(self.S1, self.S1, y, w, 2, top)
        if depth == 3: return self._meet(self.S2, self.S1, y, w, 3, top)
        if depth == 4: return sorted(self._meet(self.S3, self.S1, y, w, 4, top) + self._meet(self.S2, self.S2, y, w, 4, top), key=lambda z: -z[0])[:top]
        out = []
        if depth == 5: out += self._meet(self.S3, self.S2, y, w, 5, top)
        if depth == 6 and not self._late():
            e = (w * (self.S3.sig - y).abs()).sum(1); idx = e.topk(min(3000, len(e)), largest=False).indices                    # sieve: observational compatibility
            sub = Table(self.S3.sig[idx], self.S3.params[idx], [self.S3.trees[i] for i in idx.tolist()]); self.stats["cap_S3xS3"] = len(self.S3) - len(sub); out += self._meet(sub, sub, y, w, 6, top)
        # partial programs op(known, ?X): known ∈ S1 (and S2 at depth 6), hole of remaining depth
        for K, kd in ((self.S1, 1), (self.S2, 2)):
            if depth - kd < 4 or (kd == 2 and depth < 6): continue
            partials = []
            for oi, op in enumerate(OPS):
                for i in range(len(K)):
                    h = make_hole(op, K.sig[i], y, w, depth - kd); partials.append(PartialStructure(op, K.trees[i], h))
            partials.sort(key=lambda p: p.lower_bound()); self.stats["holes_pruned"] += max(0, len(partials) - self.hole_beam)
            for p in partials[: self.hole_beam]:
                if self._late(): break
                self.stats["holes"] += 1; kp = float(K.params[K.trees.index(p.known)])
                for sc, e, tree, par in self.solve(p.hole.target, p.hole.care, p.hole.depth_left, 2):
                    tot = e + p.hole.unavoidable; out.append((float(_evidence(torch.tensor(tot), n, par + kp, depth)), tot, (p.op, p.known, tree), par + kp))
        out.sort(key=lambda z: -z[0]); return out[:top]

    def candidates(self):
        """Candidates by depth (the `top` best of each depth) + alternative forms with the same signature."""
        out = []
        for d in range(1, self.max_depth + 1):
            if self._late(): self.stats["time_stopped_at_depth"] = d; break
            for sc, e, tree, par in self.solve(self.y, self.w, d, self.top): out.append({"depth": d, "score": sc, "errors": e, "tree": tree})
        return out


def select_literals(sig, max_literals=14, near=0.03):
    """Bounds the number of literals BEFORE any composition: near-duplicates discarded (disagreement < `near`), then
    greedy selection of the farthest (signature diversity). Never looks at the target: an atom masked by a
    XOR is not correlated with the target and would be eliminated by an agreement criterion."""
    n = len(sig); D = torch.cdist(sig, sig, p=1) / sig.shape[1]; keep = [0]; mind = D[0].clone()
    while len(keep) < min(max_literals, n):
        i = int(mind.argmax())
        if float(mind[i]) < near: break
        keep.append(i); mind = torch.minimum(mind, D[i])
    return keep


def tree_to_program(tree, literals: Sequence[Literal]) -> BNode:
    if tree[0] == "lit": return literals[tree[1]].program
    return BNode(tree[0], children=(tree_to_program(tree[1], literals), tree_to_program(tree[2], literals)))


# ================================================================ complete generator

@dataclass
class Generation:
    programs: List[BNode]                    # candidates (structure + geometric parameters), ready to be polished
    info: List[dict]
    stats: dict
    literals: List[Literal]


def generate_candidates(X, y, max_depth=6, top=6, seed=0, macros=True, typed=True, time_budget=40.0, max_prims=10, extra_literals: Sequence[BNode] = (), geometry=True, max_literals=14) -> Generation:
    """Factorized generation: literals (boundary geometry and / or supplied atoms, for example fitted by
    gradient) → synthesis on signatures → candidates."""
    t0 = time.time(); B = boundary_points(X, y) if geometry else (torch.zeros(0, 2),) * 3; t1 = time.time()
    prims = detect_primitives(B, max_prims=max_prims, seed=seed) if len(B[0]) >= 6 else []; t2 = time.time()
    lits = literals_from(prims, macros=macros) + [Literal(p, "fitted", 0) for p in extra_literals]; raw = len(lits)
    if typed: lits = [l for l in lits if well_typed(l.program)]
    stats = {"boundary_points": len(B[0]), "primitives": len(prims), "lines": sum(p.kind == "line" for p in prims), "circles": sum(p.kind == "circle" for p in prims), "literals_raw": raw,
             "type_rejected": raw - len(lits), "t_boundary": round(t1 - t0, 3), "t_primitives": round(t2 - t1, 3)}
    if not lits: return Generation([], [], {**stats, "t_tables": 0.0, "t_synthesis": 0.0}, [])
    Xf = X.float(); sig = torch.stack([predict(l.program, Xf).to(F32) for l in lits]); nonconst = [i for i in range(len(lits)) if 0 < float(sig[i].sum()) < sig.shape[1]]
    if not nonconst: return Generation([], [], {**stats, "t_tables": 0.0, "t_synthesis": 0.0}, [])
    lits = [lits[i] for i in nonconst]; sig = sig[nonconst]; keep = select_literals(sig, max_literals); stats["literals_dropped"] = len(lits) - len(keep)
    lits = [lits[i] for i in keep]; sig = sig[keep]; par = torch.tensor([float(l.program.n_params()) for l in lits])
    syn = SignatureSynthesizer(sig, par, y, max_depth=max_depth, top=top, time_budget=time_budget); t3 = time.time(); cands = syn.candidates(); t4 = time.time()
    progs, info, seen = [], [], set()
    for c in cands:
        p = normalize(tree_to_program(c["tree"], lits)); f = algebraic_form(p)
        if f in seen: continue
        seen.add(f); progs.append(p); info.append({"depth": c["depth"], "score": c["score"], "errors": c["errors"], "sources": sorted({lits[i].source for i in _lits_of(c["tree"])})})
    stats.update(syn.stats); stats.update({"candidates": len(progs), "t_tables": round(t3 - t2, 3), "t_synthesis": round(t4 - t3, 3), "seconds": round(time.time() - t0, 3)})
    return Generation(progs, info, stats, lits)


def _lits_of(tree): return [tree[1]] if tree[0] == "lit" else _lits_of(tree[1]) + _lits_of(tree[2])

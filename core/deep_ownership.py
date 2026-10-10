"""Boundary Ownership & Fragment Attribution — for each piece of observed boundary, which latent primitive does it
belong to?

Attribution is a GLOBAL problem: all observed midpoints are attributed together to a small number of latent
curves (circles, lines), by minimising an explicit energy whose every term is reported separately:

    fit          deviation of the midpoints from the curve that owns them (truncated)
    noise        midpoints left without an owner
    smooth       neighbours with the same tangent attributed to different owners
    complexity   cost per parameter of each curve used
    absent       EXPECTED and VISIBLE boundary of a curve, with no observed midpoint (legitimate negative evidence)
    chance       midpoints a curve would capture by chance: a curve is credited only with the excess (null calibration)

The occluder is an explicit object (`Occlusion`): a curve whose boundary disappears on only one side of another
curve is declared occluded by it on that side, and that portion is not counted as absent.

Objects:
  Owner                        candidate or retained latent curve, with the midpoints it owns
  Occlusion                    (occluder, occluded, side, hidden fraction)
  BoundaryOwnershipHypothesis  for a fragment: weighted possible owners (curve A, curve B, occluder, domain
                               limit, noise, unknown) — no single owner is forced
  OwnershipSolution            attribution, posterior per midpoint, occluders, energy, log

Compact solver: greedy addition of curves (best first), then at most three passes of
"attribution → re-estimation of the parameters → smoothed re-attribution", each kept only if the energy decreases.
No learned model. No world truth.
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
from core.deep_fit import DT
from core.deep_partial import SIGMA, ObservedBoundary, circle_components, circle_geometric, geometry, line_components, line_tls, link_fragments, render, visible_mask
from core.deep_primitives2 import _wrap, disc_atom, line_atom

DEFAULTS = {"tau": 6.25, "beta": 1.0, "lam": 10.0, "gamma": 1.0, "chance": 1.5, "temperature": 1.0, "long_range": True, "occlusion": True, "max_iters": 3, "min_points": 4}
ROLES = ("owner", "occluder", "domain", "noise", "unknown")
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "deep_configs", "ownership.json")


def load_config() -> dict:
    """Settings (energy costs, posterior temperature) chosen on the TUNING scenes; defaults otherwise."""
    try: return {**DEFAULTS, **json.load(open(CONFIG_FILE))["config"]}
    except (OSError, ValueError, KeyError): return dict(DEFAULTS)


@dataclass
class Owner:
    kind: str                       # "circle" | "line"
    params: tuple                   # (cx, cy, r) | (angle, offset)
    ids: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))

    @property
    def n_params(self): return 3 if self.kind == "circle" else 2
    def program(self) -> BNode: return disc_atom(*self.params) if self.kind == "circle" else line_atom(*self.params)
    def distance(self, M: torch.Tensor) -> torch.Tensor:
        if self.kind == "circle": return ((M - torch.tensor(self.params[:2], dtype=DT)).norm(dim=1) - self.params[2]).abs()
        return ((M @ torch.tensor([math.cos(self.params[0]), math.sin(self.params[0])], dtype=DT)) - self.params[1]).abs()
    def boundary(self, n=96) -> torch.Tensor: return render(self.program(), n=n, box=(-9.0, 9.0, -9.0, 9.0), reach=6.0)


@dataclass
class Occlusion:
    """The occluder `occluder` hides the curve `occluded` where the occluder's predicate equals `side`."""
    occluder: int
    occluded: int
    side: bool
    hidden_fraction: float


@dataclass
class BoundaryOwnershipHypothesis:
    """Possible owners of an observed fragment. `options`: (role, curve index or None, probability), sorted."""
    fragment: int
    ids: torch.Tensor
    options: List[Tuple[str, Optional[int], float]]

    def top(self, k=1): return self.options[:k]
    def owners(self, k=3) -> List[int]: return [i for r, i, _ in self.options if i is not None][:k]
    @property
    def confidence(self): return self.options[0][2] if self.options else 0.0
    @property
    def ambiguous(self): return len(self.options) > 1 and self.options[1][2] > 0.25


@dataclass
class OwnershipSolution:
    owners: List[Owner]
    labels: torch.Tensor            # [m] curve index, −1 = no owner
    posterior: torch.Tensor         # [m, K + 1]; last column = noise
    occlusions: List[Occlusion]
    terms: Dict[str, float]         # separate energy terms + total
    log: List[dict]
    stats: Dict[str, float]

    def top_k(self, k=3) -> torch.Tensor:
        """[m, k] curve indices by decreasing probability (K = noise)."""
        return self.posterior.argsort(dim=1, descending=True)[:, : min(k, self.posterior.shape[1])]
    def is_occluder(self, i): return any(o.occluder == i for o in self.occlusions)


# ================================================================ candidates

def candidates(ob: ObservedBoundary, seed=0) -> List[Owner]:
    """Candidate latent curves: consensus components (circles and lines), re-estimated on their midpoints."""
    return [Owner(c.kind, c.params) for c in circle_components(ob, seed=seed) + line_components(ob, seed=seed)]


def refit(o: Owner, S: torch.Tensor) -> Optional[tuple]:
    """Parameters re-estimated on the owned midpoints; None if the estimate leaves the reasonable domain."""
    if len(S) < 4: return None
    if o.kind == "line":
        n, b, _ = line_tls(S); n0 = torch.tensor([math.cos(o.params[0]), math.sin(o.params[0])], dtype=DT); n, b = (n, b) if float(n @ n0) >= 0 else (-n, -b)
        return (math.atan2(float(n[1]), float(n[0])), b) if float(n @ n0) > 0.8 else None
    c, r, _ = circle_geometric(S, o.params[:2], o.params[2], iters=8); return (float(c[0]), float(c[1]), r) if 0.15 <= r <= 4.0 and float((c - torch.tensor(o.params[:2], dtype=DT)).norm()) < 1.0 else None


# ================================================================ explicit occluder and counterfactual visibility

def _domain(ob: ObservedBoundary): X = ob.X; return float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 1].min()), float(X[:, 1].max())


def _inside(P, box): return (P[:, 0] >= box[0]) & (P[:, 0] <= box[1]) & (P[:, 1] >= box[2]) & (P[:, 1] <= box[3])


def find_occlusions(owners: Sequence[Owner], ob: ObservedBoundary, tol=0.3, lo=0.2, hi=0.6, share=0.1) -> List[Occlusion]:
    """For each pair (curve o, curve q): the boundary of o, inside the domain, is cut in two by the predicate
    of q (only the midpoints closer to o than to q count as support for o). If one side is supported by midpoints (fraction ≥ hi) and the other almost not (fraction ≤ lo), q OCCLUDES o on that
    side. Each side must carry at least `share` of the boundary: otherwise nothing can be concluded."""
    box = _domain(ob); out = []; M = ob.points
    for i, o in enumerate(owners):
        P = o.boundary(); P = P[_inside(P, box)]
        if len(P) < 10 or not len(M): continue
        best = None; do = o.distance(M)
        for j, q in enumerate(owners):
            if j == i: continue
            Mo = M[do <= q.distance(M)]                                               # the midpoints better explained by q are its own: they do not support o
            if not len(Mo): continue
            seen = torch.cdist(P, Mo).min(1).values < tol; side = predict(q.program(), P.float()).bool()
            for val in (True, False):
                hid = side == val
                if float(hid.float().mean()) < share or float((~hid).float().mean()) < share: continue
                sh, sv = float(seen[hid].float().mean()), float(seen[~hid].float().mean())
                if sh <= lo and sv >= hi and (best is None or sv - sh > best[0]): best = (sv - sh, Occlusion(j, i, val, float(hid.float().mean())))
        if best is not None: out.append(best[1])
    return out


def expected_visibility(p: BNode, ob: ObservedBoundary, occluders: Sequence[Tuple[BNode, bool]] = (), tol=0.3, n=96) -> Dict[str, torch.Tensor]:
    """primitive → render expected boundary → apply hypothesized occlusion → compare only visible expected portion.

    Returns masks over the rendered points: `domain` (inside the box of the observations), `hidden` (hidden by a
    hypothesised occluder), `expected` (inside the domain and not hidden), `seen` (an observed midpoint within `tol`)."""
    P = render(p, n=n, box=(-9.0, 9.0, -9.0, 9.0), reach=6.0); dom = _inside(P, _domain(ob)); hidden = torch.zeros(len(P), dtype=torch.bool)
    for q, val in occluders: hidden |= predict(q, P.float()).bool() == val
    seen = (torch.cdist(P, ob.points).min(1).values < tol) if len(P) and len(ob.points) else torch.zeros(len(P), dtype=torch.bool)
    return {"points": P, "domain": dom, "hidden": hidden & dom, "expected": dom & ~hidden, "seen": seen}


def visible_support(p: BNode, ob: ObservedBoundary, occluders: Sequence[Tuple[BNode, bool]] = (), tol=0.15) -> Dict[str, float]:
    """COUNTERFACTUAL VISIBILITY score of a primitive hypothesis: closely explained midpoints, multiplied by the fraction
    of the visible expected boundary that is actually observed. The portion hidden by a hypothesised occluder enters
    neither the numerator nor the denominator."""
    v = expected_visibility(p, ob, occluders); P = v["points"]; n_exp = int(v["expected"].sum()); cov = float(v["seen"][v["expected"]].float().mean()) if n_exp else 0.0
    sup = float((torch.cdist(ob.points, P[v["expected"]]).min(1).values < tol).sum()) if n_exp and len(ob.points) else 0.0
    return {"support": sup, "coverage": cov, "expected": n_exp / max(1, len(P)), "hidden": float(v["hidden"].float().mean()) if len(P) else 0.0, "score": sup * cov}


# ================================================================ global energy

def _neighbors(ob: ObservedBoundary, link=0.35, turn=0.6) -> torch.Tensor:
    M, N = ob.points, ob.normals; A = (torch.cdist(M, M) < link) & ((N @ N.T).abs() > math.cos(turn)); A.fill_diagonal_(False); return A


def absent_fraction(i: int, owners: Sequence[Owner], occ: Sequence[Occlusion], ob: ObservedBoundary, use_occlusion=True) -> float:
    """Fraction of the visible expected boundary of curve i that has no observed midpoint."""
    oc = [(owners[o.occluder].program(), o.side) for o in occ if o.occluded == i] if use_occlusion else []; v = expected_visibility(owners[i].program(), ob, oc); e = v["expected"]
    return float((~v["seen"][e]).float().mean()) if int(e.sum()) else 1.0


def chance_support(o: Owner, ob: ObservedBoundary, cfg: dict, ids: Optional[torch.Tensor] = None, tol=0.3) -> float:
    """CALIBRATION BY CHANCE: number of midpoints a curve would capture if the midpoints were spread uniformly
    over the domain: density × area of the capture band, along the portion of the curve where it has midpoints
    (`ids`), or along its whole portion inside the domain if `ids` is omitted. A curve is credited only with what it
    explains BEYOND that number: on random labels, a curve does not pay for itself."""
    box = _domain(ob); P = o.boundary(192); inside = _inside(P, box); area = max(1e-9, (box[1] - box[0]) * (box[3] - box[2]))
    if len(P) < 2 or not int(inside.sum()): return 0.0
    if ids is not None: inside = inside & (torch.cdist(P, ob.points[ids]).min(1).values < tol) if len(ids) else torch.zeros_like(inside)
    step = float((P[1:] - P[:-1]).norm(dim=1).median()); band = 2 * SIGMA * math.sqrt(cfg["tau"]); return len(ob.points) * float(inside.sum()) * step * band / area


def energy(labels: torch.Tensor, owners: Sequence[Owner], ob: ObservedBoundary, cfg: dict, A: Optional[torch.Tensor] = None, occ: Optional[Sequence[Occlusion]] = None) -> Dict[str, float]:
    """Global coherence score of an attribution, SEPARATE terms (lower = better)."""
    M = ob.points; tau = cfg["tau"]; fit = 0.0; ab = 0.0; ch = 0.0; used = [i for i in range(len(owners)) if int((labels == i).sum())]; occ = find_occlusions(owners, ob) if occ is None and cfg["occlusion"] else (occ or [])
    for i in used:
        m = labels == i; fit += float(((owners[i].distance(M[m]) / SIGMA) ** 2).clamp(max=tau).sum()); ab += cfg["gamma"] * tau * absent_fraction(i, owners, occ, ob, cfg["occlusion"]) * int(m.sum()) if cfg["gamma"] > 0 else 0.0
        ch += cfg["chance"] * tau * min(float(m.sum()), chance_support(owners[i], ob, cfg, torch.nonzero(m)[:, 0])) if cfg["chance"] > 0 else 0.0
    noise = tau * int((labels < 0).sum()); A = _neighbors(ob) if A is None else A; smooth = cfg["beta"] * float((A & (labels[:, None] != labels[None, :])).sum()) / 2; cx = cfg["lam"] * sum(owners[i].n_params for i in used)
    return {"fit": fit, "noise": noise, "smooth": smooth, "complexity": cx, "absent": ab, "chance": ch, "total": fit + noise + smooth + cx + ab + ch, "owners": len(used)}


def _unary(owners: Sequence[Owner], M: torch.Tensor, tau: float) -> torch.Tensor:
    """[m, K + 1] unary costs; last column = noise."""
    cols = [((o.distance(M) / SIGMA) ** 2).clamp(max=2 * tau) for o in owners] + [torch.full((len(M),), tau, dtype=DT)]; return torch.stack(cols, 1)


def _assign(owners, ob, cfg, A, sweeps=2) -> torch.Tensor:
    """Re-attribution: best owner per midpoint, then smoothing (neighbours with the same tangent reinforce each other)."""
    U = _unary(owners, ob.points, cfg["tau"]); K = len(owners); lab = U.argmin(1)
    for _ in range(sweeps if cfg["beta"] > 0 else 0):
        onehot = torch.nn.functional.one_hot(lab, K + 1).to(DT); agree = A.to(DT) @ onehot; lab = (U + cfg["beta"] * (A.sum(1, keepdim=True).to(DT) - agree)).argmin(1)
    return torch.where(lab == K, torch.full_like(lab, -1), lab)


def _local_only(labels: torch.Tensor, ob: ObservedBoundary, link=0.6) -> torch.Tensor:
    """Without LONG-RANGE association: each curve keeps only its largest spatially connected cluster."""
    out = labels.clone()
    for i in torch.unique(labels[labels >= 0]).tolist():
        ids = torch.nonzero(labels == i)[:, 0]; f = link_fragments(ob.points[ids], link); keep = f == torch.bincount(f).argmax(); out[ids[~keep]] = -1
    return out


def solve(ob: ObservedBoundary, cfg: Optional[dict] = None, seed=0) -> OwnershipSolution:
    """Global attribution. 1) candidates; 2) greedy addition, best first, of the curve that lowers the energy
    the most; 3) at most `max_iters` passes: re-estimation of the parameters on the owned midpoints →
    smoothed re-attribution → removal of the curves that have become useless; a pass is kept only if the energy decreases."""
    t0 = time.time(); cfg = {**load_config(), **(cfg or {})}; M = ob.points; m = len(M); cands = candidates(ob, seed); A = _neighbors(ob) if m else torch.zeros(0, 0, dtype=torch.bool); tested = 0; log = []
    if m == 0 or not cands: return OwnershipSolution([], torch.full((m,), -1, dtype=torch.long), torch.ones(m, 1, dtype=DT), [], {"fit": 0.0, "noise": cfg["tau"] * m, "smooth": 0.0, "complexity": 0.0, "absent": 0.0, "chance": 0.0, "total": cfg["tau"] * m, "owners": 0}, log, {"assignments_tested": 0, "candidates": len(cands), "seconds": 0.0})
    post = lambda lab: _local_only(lab, ob) if not cfg["long_range"] else lab
    chosen: List[int] = []; labels = torch.full((m,), -1, dtype=torch.long); E = energy(labels, [], ob, cfg, A)
    while True:                                                                    # greedy addition
        best = None
        for c in range(len(cands)):
            if c in chosen: continue
            own = [cands[i] for i in chosen + [c]]; lab = post(_assign(own, ob, cfg, A)); e = energy(lab, own, ob, cfg, A); tested += 1
            if e["total"] < E["total"] - 1e-9 and (best is None or e["total"] < best[1]["total"]): best = (c, e, lab)
        if best is None: break
        chosen.append(best[0]); E, labels = best[1], best[2]; log.append({"step": "add", "owner": cands[best[0]].kind, **{k: round(v, 2) for k, v in E.items()}})
    owners = [Owner(cands[i].kind, cands[i].params) for i in chosen]
    for it in range(cfg["max_iters"]):                                             # attribution ↔ parameters
        new = []
        for i, o in enumerate(owners):
            ids = torch.nonzero(labels == i)[:, 0]
            if len(ids) < cfg["min_points"]: continue
            p = refit(o, M[ids]); new.append(Owner(o.kind, p if p is not None else o.params))
        if not new: break
        lab = post(_assign(new, ob, cfg, A)); keep = [i for i in range(len(new)) if int((lab == i).sum()) >= cfg["min_points"]]; new = [new[i] for i in keep]; lab = post(_assign(new, ob, cfg, A)) if new else torch.full((m,), -1, dtype=torch.long); e = energy(lab, new, ob, cfg, A); tested += 1
        ok = e["total"] < E["total"] - 1e-9; log.append({"step": "refine", "iteration": it + 1, "action": "commit" if ok else "rollback", **{k: round(v, 2) for k, v in e.items()}})
        if not ok: break
        owners, labels, E = new, lab, e
    used = [i for i in range(len(owners)) if int((labels == i).sum()) >= cfg["min_points"]]; remap = {old: new for new, old in enumerate(used)}; owners = [owners[i] for i in used]; labels = torch.tensor([remap.get(int(l), -1) for l in labels], dtype=torch.long)
    for i, o in enumerate(owners): o.ids = torch.nonzero(labels == i)[:, 0]
    occ = find_occlusions(owners, ob) if cfg["occlusion"] else []; E = energy(labels, owners, ob, cfg, A, occ); U = _unary(owners, M, cfg["tau"]); P = torch.softmax(-U / (2 * cfg["temperature"]), 1)
    return OwnershipSolution(owners, labels, P, occ, E, log, {"assignments_tested": tested, "candidates": len(cands), "owners": len(owners), "seconds": round(time.time() - t0, 3)})


def null_assigned(cfg: Optional[dict] = None, n=300, trials=12, seed=0, box=((-3.0, 1.2), (-3.0, 3.0)), rates=(0.5, 0.2)) -> List[float]:
    """CALIBRATION WITHOUT TRUTH: fraction of the midpoints that the solver attributes to a curve on RANDOM labels
    (no boundary exists). The `chance` cost is set so that this fraction stays low."""
    from core.deep_partial import observe
    out = []
    for t in range(trials):
        g = torch.Generator().manual_seed(seed * 1000 + t); X = torch.rand(n, 2, generator=g); X = torch.stack([box[0][0] + X[:, 0] * (box[0][1] - box[0][0]), box[1][0] + X[:, 1] * (box[1][1] - box[1][0])], 1); y = (torch.rand(n, generator=g) < rates[t % len(rates)]).long()
        s = solve(observe(X, y), cfg, t); out.append(float((s.labels >= 0).float().mean()) if len(s.labels) else 0.0)
    return out


# ================================================================ ownership hypotheses per fragment

def ownership_hypotheses(sol: OwnershipSolution, ob: ObservedBoundary, link=0.45, turn=0.6, edge=0.2) -> List[BoundaryOwnershipHypothesis]:
    """Micro-fragments (proximity + tangential continuity) and, for each, the distribution of its owners:
    mean of the posterior of its midpoints. A curve that occludes another appears with the role `occluder`;
    a fragment stuck to the edge of the box of the observations receives a `domain` share; the mass missing from a
    clear-cut decision (maximum probability below 0.5) is reported as `unknown`."""
    from core.deep_fragments import _components_of
    M, N = ob.points, ob.normals; out = []
    if not len(M): return out
    lab = _components_of((torch.cdist(M, M) < link) & ((N @ N.T).abs() > math.cos(turn))); box = _domain(ob); K = len(sol.owners)
    for g in torch.unique(lab).tolist():
        ids = torch.nonzero(lab == g)[:, 0]; p = sol.posterior[ids].mean(0); S = M[ids]; near = float(((S[:, 0] - box[0]).abs().clamp(max=9).minimum((S[:, 0] - box[1]).abs()).minimum((S[:, 1] - box[2]).abs()).minimum((S[:, 1] - box[3]).abs()) < edge).float().mean())
        opts = [("occluder" if sol.is_occluder(i) else "owner", i, float(p[i])) for i in range(K) if float(p[i]) > 0.02] + [("noise", None, float(p[K]))]
        if near > 0.5: opts = [(r, i, w * (1 - 0.5 * near)) for r, i, w in opts] + [("domain", None, 0.5 * near)]
        top = max((w for _, _, w in opts), default=0.0)
        if top < 0.5: opts.append(("unknown", None, 0.5 - top))
        z = sum(w for _, _, w in opts) or 1.0; out.append(BoundaryOwnershipHypothesis(len(out), ids, sorted(((r, i, w / z) for r, i, w in opts), key=lambda x: -x[2])))
    return out


# ================================================================ support from the composition

def ownership_from_program(prog: BNode, ob: ObservedBoundary, tol=0.25) -> Tuple[torch.Tensor, List[BNode]]:
    """primitive hypotheses → composition hypothesis → constraints → ownership refinement: each midpoint goes to the LEAF
    of the composite whose boundary is the closest AMONG those that are visible at that point (a leaf occluded
    at that place cannot own the midpoint). Returns (leaf index per midpoint, leaves)."""
    from core.deep_partial import leaf_paths
    M = ob.points; lp = leaf_paths(prog); lab = torch.full((len(M),), -1, dtype=torch.long)
    if not len(M) or not lp: return lab, [l for _, l in lp]
    D = []
    for path, leaf in lp:
        B = render(leaf, n=160, box=(-9.0, 9.0, -9.0, 9.0), reach=6.0); d = torch.cdist(M, B).min(1).values if len(B) else torch.full((len(M),), 9.0, dtype=DT); vis = visible_mask(prog, path, M) if len(lp) > 1 else torch.ones(len(M), dtype=torch.bool); D.append(torch.where(vis, d, d + 9.0))
    D = torch.stack(D, 1); d, i = D.min(1); lab[d < tol] = i[d < tol]; return lab, [l for _, l in lp]


# ================================================================ from owned curves to primitives

def _curves(p: BNode) -> List[Tuple[str, tuple]]:
    g = geometry(p)
    if g["kind"] == "circle": return [("circle", (g["center"][0], g["center"][1], r)) for r in g["radii"]]
    if g["kind"] == "line": return [("line", (g["angle"], b)) for b in g["offsets"]]
    out = []
    for d in (g["direction"] + g["half_width"], g["direction"] - g["half_width"]): a = d + math.pi / 2; out.append(("line", (a, math.cos(a) * g["center"][0] + math.sin(a) * g["center"][1])))
    return out


def _same(kind, a, o: Owner, tol_c=0.35, tol_r=0.3, tol_a=0.22, tol_b=0.3) -> bool:
    if kind != o.kind: return False
    if kind == "circle": return math.hypot(a[0] - o.params[0], a[1] - o.params[1]) < tol_c and abs(a[2] - o.params[2]) < tol_r
    d = _wrap(a[0] - o.params[0]); return (abs(d) < tol_a and abs(a[1] - o.params[1]) < tol_b) or (abs(abs(d) - math.pi) < tol_a and abs(a[1] + o.params[1]) < tol_b)


def owned(p: BNode, sol: OwnershipSolution) -> float:
    """Fraction of the boundary curves of a primitive hypothesis that coincide with a curve retained by the attribution."""
    cs = _curves(p); return sum(any(_same(k, a, o) for o in sol.owners) for k, a in cs) / max(1, len(cs))


def filter_by_ownership(programs: Sequence[BNode], sol: OwnershipSolution, min_owned=1.0) -> List[BNode]:
    """Keeps the primitive hypotheses whose boundaries are ALL (by default) owned curves: the global
    attribution serves as validation for the hypotheses produced without attribution."""
    return [p for p in programs if owned(p, sol) >= min_owned]


def primitives_from(sol: OwnershipSolution, ob: ObservedBoundary, k=3):
    """Owned curves → fragments (one per curve, imposed model) → relations → type hypotheses, top-k per group."""
    from core.deep_fragments import build_graph, make_fragment, rank_by_group, type_hypotheses
    fr = [make_fragment(i, o.ids, ob, (o.kind, o.params)) for i, o in enumerate(sol.owners) if len(o.ids) >= 4]
    for i, f in enumerate(fr): f.index = i
    for a in fr: a.neighbors = [b.index for b in fr if b is not a and float(torch.cdist(a.points, b.points).min()) < 0.6]
    g = build_graph(fr); H = type_hypotheses(g, ob); return rank_by_group(H, g, k, ob), H

"""PrimitiveDiscovery — discover a small basis of explanatory primitives from the observations.

Distinctions maintained in the code:
    Primitive (basic element)    ≠  Fragment (reusable composition, core.deep_structure.Fragment)
                                 ≠  CompositeStructure (assembly, core.deep_structure.CompositeStructure)

Architecture: PROPOSAL and VALIDATION are separated.
    PrimitiveProposal   : a proposer (gradient, geometric, compact neural, hybrid) emits candidates; it is
                          allowed to be imperfect.
    PrimitiveValidation : the WCE validator decides, on the observations alone, whether a candidate really explains
                          a boundary (local contrast independent of polarity), whether it is stable, whether it is
                          redundant with or equivalent to an already retained primitive, and whether it deserves to be kept.
    CompatiblePrimitives: several primitive hypotheses are maintained, weighted, by equivalence classes.

Validation signal (explicit, not a learned score): among the pairs of NEIGHBOURING points that the boundary of the candidate
separates, the share of pairs with opposite labels must exceed the base rate. A true atomic boundary separates
points with opposite labels whatever the composition (a XOR inverts the polarity, not the contrast); an
imaginary boundary crosses homogeneous zones.

This module imports no world module and never receives the truth.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from core.boundary_program import BNode, _inits, atoms, predict
from core.deep_fit import DT, FitCounter, FitData, compile_skeleton, fit_batch, heuristic_inits
from core.deep_structure import _ATOM_INVARIANTS, _leaf, description_bits, skel_key

Z_MIN = 3.0                      # validation threshold: deviation of the contrast from the base rate (in standard deviations)
MIN_SUPPORT = 5                  # minimal number of separated opposite pairs
EQUIV_AGREEMENT = 0.97           # functional (or complement) agreement beyond which two primitives are equivalent


def atom_name(p: BNode):
    l = _leaf(p); neg = any(n.kind == "not" for n in _chain(p)); return ("not_" if neg else "") + f"{l.feat}.{l.op}"


def _chain(p: BNode):
    out = [p]
    while p.children and p.kind in ("translate", "not"): p = p.children[0]; out.append(p)
    return out


# ---- primitive bases -------------------------------------------------------------------------------------------------
def basis(name="rich") -> List[BNode]:
    """rich: the 8 atoms of the grammar. minimal: half-plane, disc, anti-disc (the other atoms are derived from them
    by composition: band = two parallel half-planes, ring = disc ∧ anti-disc, sector = two half-planes)."""
    A = atoms()
    if name == "rich": return A
    if name == "minimal": return [a for a in A if atom_name(a) in ("dot.le", "norm.le", "norm.ge")]
    raise KeyError(name)


# ================================================================ neighbouring pairs (support of the contrast)

@dataclass
class Pairs:
    P: torch.Tensor                # [m, 2]
    Q: torch.Tensor                # [m, 2]
    opposite: torch.Tensor         # [m] 1 if opposite labels
    weight: torch.Tensor           # [m] 1; 0 when an opposite pair is already explained by a retained primitive
    q0: float                      # base rate of opposite pairs

    def __len__(self): return len(self.P)


def close_pairs(X, y, k=6, radius=0.6) -> Pairs:
    """Pairs of neighbouring points (k nearest neighbours, at distance < radius), RAW labels."""
    X = X.to(DT); n = len(X); D = torch.cdist(X, X); D.fill_diagonal_(float("inf")); d, nb = D.topk(min(k, n - 1), largest=False)
    i = torch.arange(n)[:, None].expand_as(nb); m = (d < radius) & (i < nb); a, b = i[m], nb[m]
    opp = (y[a] != y[b]).to(DT); return Pairs(X[a], X[b], opp, torch.ones(len(a), dtype=DT), float(opp.mean()) if len(a) else 0.0)


def separates(prog: BNode, pairs: Pairs): return predict(prog, pairs.P.float()) != predict(prog, pairs.Q.float())


def contrast(prog: BNode, pairs: Pairs, use_weight=True) -> dict:
    """Polarity-independent contrast: separated pairs, of which opposite, and z deviation from the base rate q0."""
    s = separates(prog, pairs).to(DT); w = pairs.weight if use_weight else torch.ones_like(pairs.weight); n = float(s.sum()); k = float((s * pairs.opposite * w).sum())
    nw = float((s * torch.where(pairs.opposite > 0, w, torch.ones_like(w))).sum()); q0 = min(0.9, max(0.02, pairs.q0))
    z = (k - q0 * nw) / math.sqrt(max(1e-9, nw * q0 * (1 - q0))) if nw > 0 else 0.0
    return {"separated": n, "opposite_separated": k, "z": z, "purity": k / nw if nw > 0 else 0.0}


def contrast_fit(skel: BNode, pairs: Pairs, inits, steps=60, lr=0.05, counter: Optional[FitCounter] = None):
    """Fit of an atom by CONTRAST: maximise Σ_pairs separation(θ) · (opposite − q0). Separating an opposite
    pair pays, separating a homogeneous pair costs: a boundary that sweeps homogeneous zones is penalised
    (which the simple counting of separations did not do). Returns (theta [K, np], hard objective [K])."""
    th = (inits if torch.is_tensor(inits) else torch.stack([torch.as_tensor(i, dtype=DT) for i in inits])).clone().to(DT).requires_grad_(True)
    fn = compile_skeleton(skel); P, Q = pairs.P[None], pairs.Q[None]; g = pairs.weight * pairs.opposite - pairs.q0 * torch.where(pairs.opposite > 0, pairs.weight, torch.ones_like(pairs.weight)); opt = torch.optim.Adam([th], lr=lr)
    for t in range(steps):
        opt.zero_grad(); kk = 3.0 + 12.0 * t / steps; sig = lambda z: torch.sigmoid(kk * z); mp, mq = fn(th, P, sig), fn(th, Q, sig)
        (-(g * (mp * (1 - mq) + mq * (1 - mp))).sum()).backward(); opt.step()
    th = th.detach(); hard = lambda z: (z >= 0).to(DT)
    if counter is not None: counter.charge(len(th), steps)
    return th, (g * (fn(th, P, hard) != fn(th, Q, hard)).to(DT)).sum(1)


# ================================================================ explicit objects

@dataclass
class PrimitiveCandidate:
    """EXPLICIT candidate primitive (never an opaque latent vector)."""
    program: BNode                              # atom of the grammar + parameters
    source: str = ""                            # originating proposer
    z: float = 0.0                              # contrast (validation)
    support: float = 0.0                        # observational support: separated opposite pairs
    purity: float = 0.0
    uncertainty: Optional[List[float]] = None   # standard deviation of the parameters (resampling)
    residual_explained: float = 0.0             # share of the still unexplained opposite pairs that it explains
    domain: Optional[Tuple[float, float, float, float]] = None       # bounding box of the observations where it is defined
    kin: Dict[str, List[int]] = field(default_factory=dict)          # kinship: complement, concentric, parallel, equivalent
    validated: bool = False
    reason: str = ""

    @property
    def type(self): return atom_name(self.program)
    @property
    def params(self): return self.program.param_vector()
    @property
    def n_params(self): return self.program.n_params()
    @property
    def description_bits(self): return description_bits(self.program.skeleton()) + 8.0 * self.n_params        # 8 bits per parameter (resolution ≈ 1/256 of the domain)
    @property
    def invariants(self): l = _leaf(self.program); return set(_ATOM_INVARIANTS.get((l.feat, l.op), ()))

    def signature(self, X): return predict(self.program, X.float())


@dataclass
class PrimitiveClass:
    """Equivalence class of primitives: indistinguishable (or complementary) on the observed region."""
    members: List[PrimitiveCandidate]

    @property
    def representative(self): return max(self.members, key=lambda c: c.z)
    @property
    def weight(self): return self.representative.z
    @property
    def types(self): return sorted({m.type for m in self.members})


class CompatiblePrimitives:
    """Several primitive hypotheses, weighted, grouped by equivalence classes. A primitive remains
    uncertain as long as the composition or new measurements have not discriminated it."""

    def __init__(self, X): self.X = X.float(); self.classes: List[PrimitiveClass] = []; self._sig: List[torch.Tensor] = []

    def find(self, cand: PrimitiveCandidate):
        s = cand.signature(self.X).float()
        for i, t in enumerate(self._sig):
            a = float((s == t).float().mean())
            if a >= EQUIV_AGREEMENT or a <= 1 - EQUIV_AGREEMENT: return i
        return None

    def add(self, cand: PrimitiveCandidate):
        """Adds a validated primitive; returns ('new' | 'merged', class index)."""
        i = self.find(cand)
        if i is None: self.classes.append(PrimitiveClass([cand])); self._sig.append(cand.signature(self.X).float()); return "new", len(self.classes) - 1
        self.classes[i].members.append(cand); cand.kin.setdefault("equivalent", []).append(i)
        if cand.z >= self.classes[i].representative.z: self._sig[i] = cand.signature(self.X).float()
        return "merged", i

    def __len__(self): return len(self.classes)
    def weights(self):
        z = torch.tensor([c.weight for c in self.classes], dtype=DT); return (z / z.sum()).tolist() if len(z) and float(z.sum()) > 0 else []

    def literals(self, max_literals=None, alternatives=1) -> List[BNode]:
        """Programs to give to the composition: representative of each class (and up to `alternatives` members of
        different types: the type hypothesis stays open), most supported classes first."""
        out = []
        for c in sorted(self.classes, key=lambda c: -c.weight)[: max_literals]:
            seen = set()
            for m in sorted(c.members, key=lambda m: -m.z):
                if m.type not in seen and len(seen) < alternatives: seen.add(m.type); out.append(m.program)
        return out

    def retire(self, index: int, reason="retired"): c = self.classes.pop(index); self._sig.pop(index); [setattr(m, "reason", reason) for m in c.members]; return c


# ================================================================ PROPOSAL

class PrimitiveProposal:
    """Interface of a proposer: emits candidates from the observations and the still unexplained pairs."""
    name = "abstract"
    def propose(self, X, y, pairs: Pairs, seed=0, counter: Optional[FitCounter] = None) -> List[PrimitiveCandidate]: raise NotImplementedError


class GradientProposer(PrimitiveProposal):
    """Reference method (previous milestone): fit of each atom on the TARGET, several initialisations."""
    name = "gradient"
    def __init__(self, basis_name="rich", n_init=4, steps=50): self.atoms = basis(basis_name); self.n_init, self.steps = n_init, steps

    def propose(self, X, y, pairs, seed=0, counter=None):
        d = FitData.build(X, y, 0.10); out = []
        for a in self.atoms:
            th, _ = fit_batch(a, d, heuristic_inits(a, d, self.n_init, seed), steps=self.steps, counter=counter); out += [PrimitiveCandidate(a.with_params(r.tolist()), self.name) for r in th]
        return out


class GeometricProposer(PrimitiveProposal):
    """Analytic geometric method: hypotheses of lines (2 points) and of circles (3 points) drawn from the midpoints
    of opposite pairs, ranked by contrast; no gradient fitting."""
    name = "geometric"
    def __init__(self, basis_name="rich", n_hyp=600, top=10): self.basis_name, self.n_hyp, self.top = basis_name, n_hyp, top

    def propose(self, X, y, pairs, seed=0, counter=None):
        from core.deep_generation import GeoPrimitive, _circle_hyps, _line_hyps, literals_from
        g = torch.Generator().manual_seed(seed); m = (pairs.opposite * pairs.weight) > 0; M = ((pairs.P + pairs.Q) / 2)[m].float()
        if len(M) < 4: return []
        ln, lb = _line_hyps(M, self.n_hyp, g); cc, cr = _circle_hyps(M, self.n_hyp, g); P, Q = pairs.P.float(), pairs.Q.float(); gw = (pairs.weight * pairs.opposite - pairs.q0).float(); prims = []
        if len(ln):
            sc = ((((P @ ln.T - lb) * (Q @ ln.T - lb)) < 0).float() * gw[:, None]).sum(0)
            for i in sc.topk(min(self.top, len(sc))).indices.tolist(): prims.append(GeoPrimitive("line", (math.atan2(float(ln[i, 1]), float(ln[i, 0])), float(lb[i])), int(sc[i]), M))
        if len(cc):
            sc = ((((torch.cdist(P, cc) - cr) * (torch.cdist(Q, cc) - cr)) < 0).float() * gw[:, None]).sum(0)
            for i in sc.topk(min(self.top, len(sc))).indices.tolist(): prims.append(GeoPrimitive("circle", (float(cc[i, 0]), float(cc[i, 1]), float(cr[i])), int(sc[i]), M))
        lits = literals_from(prims, macros=self.basis_name == "rich", max_sectors=2); keep = {skel_key(a) for a in basis(self.basis_name)}
        return [PrimitiveCandidate(l.program, self.name) for l in lits if skel_key(l.program) in keep]


class ContrastProposer(PrimitiveProposal):
    """Gradient fitting of the CONTRAST objective (independent of polarity), seeded on boundary
    points still unexplained."""
    name = "contrast"
    def __init__(self, basis_name="rich", n_init=4, steps=60): self.atoms = basis(basis_name); self.n_init, self.steps = n_init, steps

    def propose(self, X, y, pairs, seed=0, counter=None):
        g = torch.Generator().manual_seed(seed); out = []; Xf, yl = X.float(), y.long(); m = (pairs.opposite * pairs.weight) > 0; M = ((pairs.P + pairs.Q) / 2)[m]
        if len(M) < 3: return out
        for a in self.atoms:
            inits = torch.stack([v.to(DT) for v in _inits(a, Xf, yl, self.n_init, g)])
            if a.kind == "translate": inits[:, :2] = M[torch.randint(0, len(M), (self.n_init,), generator=g)] + 0.7 * torch.randn(self.n_init, 2, generator=g, dtype=DT)
            th, _ = contrast_fit(a, pairs, inits, steps=self.steps, counter=counter); out += [PrimitiveCandidate(a.with_params(r.tolist()), self.name) for r in th]
        return out


class RefiningProposer(PrimitiveProposal):
    """HYBRID: the candidates of other proposers (geometric, neural, gradient) serve as INITIALISATION for
    the contrast fit."""
    name = "hybrid"
    def __init__(self, proposers: Sequence[PrimitiveProposal], steps=40, max_refine=24): self.proposers, self.steps, self.max_refine = list(proposers), steps, max_refine

    def propose(self, X, y, pairs, seed=0, counter=None):
        raw = [c for p in self.proposers for c in p.propose(X, y, pairs, seed, counter)]; raw.sort(key=lambda c: -contrast(c.program, pairs)["z"]); out = list(raw); by: Dict[str, List[PrimitiveCandidate]] = {}
        for c in raw[: self.max_refine]: by.setdefault(skel_key(c.program), []).append(c)
        for k, cs in by.items():
            sk = cs[0].program.skeleton(); th, _ = contrast_fit(sk, pairs, torch.tensor([c.params for c in cs], dtype=DT), steps=self.steps, counter=counter)
            out += [PrimitiveCandidate(sk.with_params(r.tolist()), f"{self.name}({c.source})") for r, c in zip(th, cs)]
        return out


# ================================================================ VALIDATION

class PrimitiveValidation:
    """The WCE validator. Explicit criteria: (1) contrast z ≥ z_min and sufficient support — the primitive explains a
    true boundary; (2) it reduces the residual (opposite pairs still unexplained); (3) stability under
    resampling; (4) redundancy / equivalence with a retained primitive."""

    def __init__(self, z_min=Z_MIN, min_support=MIN_SUPPORT, n_boot=0, max_instability=0.15): self.z_min, self.min_support, self.n_boot, self.max_instability = z_min, min_support, n_boot, max_instability

    def measure(self, cand: PrimitiveCandidate, pairs: Pairs, X) -> PrimitiveCandidate:
        c = contrast(cand.program, pairs); cand.z, cand.support, cand.purity = c["z"], c["opposite_separated"], c["purity"]
        rem = float((pairs.opposite * pairs.weight).sum()); cand.residual_explained = c["opposite_separated"] / rem if rem > 0 else 0.0
        cand.domain = (float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 1].min()), float(X[:, 1].max())); return cand

    def stability(self, cand: PrimitiveCandidate, pairs: Pairs, X, seed=0):
        """Resampling of the pairs + contrast refit: standard deviation of the parameters and functional disagreement."""
        g = torch.Generator().manual_seed(seed); sk = cand.program.skeleton(); th0 = torch.tensor([cand.params], dtype=DT); ths, dis = [], []
        for _ in range(self.n_boot):
            i = torch.randint(0, len(pairs), (len(pairs),), generator=g); pb = Pairs(pairs.P[i], pairs.Q[i], pairs.opposite[i], pairs.weight[i], pairs.q0); th, _ = contrast_fit(sk, pb, th0, steps=25)
            ths.append(th[0]); dis.append(float((predict(sk.with_params(th[0].tolist()), X.float()) != cand.signature(X)).float().mean()))
        cand.uncertainty = torch.stack(ths).std(0).tolist() if len(ths) > 1 else None; return sum(dis) / len(dis) if dis else 0.0

    def validate(self, cand: PrimitiveCandidate, pairs: Pairs, X, retained: Optional[CompatiblePrimitives] = None, seed=0) -> PrimitiveCandidate:
        self.measure(cand, pairs, X)
        if cand.support < self.min_support: cand.validated, cand.reason = False, "support insuffisant"
        elif cand.z < self.z_min: cand.validated, cand.reason = False, "contraste insuffisant (frontière non observée)"
        elif self.n_boot and self.stability(cand, pairs, X, seed) > self.max_instability: cand.validated, cand.reason = False, "instable"
        else:
            cand.validated = True; cand.reason = "retenue"
            if retained is not None and retained.find(cand) is not None: cand.reason = "équivalente à une primitive retenue"
        return cand


# ================================================================ iterative residual-guided discovery

@dataclass
class Discovery:
    primitives: CompatiblePrimitives
    proposals: int
    validated: int
    rounds: int
    seconds: float
    fit_equivalents: float
    log: List[dict]
    first_validated_at: Optional[int] = None      # number of proposals examined before the first retained primitive
    residual_left: float = 1.0                    # share of the opposite pairs left unexplained


def discover(X, y, proposer: PrimitiveProposal, validator: Optional[PrimitiveValidation] = None, rounds=4, keep_per_round=3, seed=0, max_classes=12) -> Discovery:
    """1. propose; 2. measure what each candidate explains; 3. retain the best VALIDATED ones; 4. remove from the
    residual the opposite pairs they separate; 5. propose again on the residual; stop when no candidate passes
    validation any more (the residual no longer has any detectable boundary structure). The residual is a set of PAIRS
    still unexplained, not a memory of points."""
    t0 = time.time(); validator = validator or PrimitiveValidation(); pairs = close_pairs(X, y); cp = CompatiblePrimitives(X); counter = FitCounter(); log = []; n_prop = n_val = 0; first = None
    total_opp = float(pairs.opposite.sum()); r = 0
    for r in range(rounds):
        cands = proposer.propose(X, y, pairs, seed + r, counter); n_prop += len(cands)
        # Multiple comparisons: the more candidates are examined, the more a high contrast can appear by chance. The
        # effective threshold grows like the expected maximum of N normal deviates (√(2·ln N)) + a margin of one standard deviation.
        z_eff = max(validator.z_min, math.sqrt(2 * math.log(max(2, len(cands)))) + 1.0); base = validator.z_min; validator.z_min = z_eff
        cands = [validator.validate(c, pairs, X, cp, seed) for c in cands]; ok = sorted((c for c in cands if c.validated), key=lambda c: -c.z); kept = 0
        for c in ok:
            if kept >= keep_per_round or len(cp) >= max_classes: break
            c = validator.validate(c, pairs, X, cp, seed)                                      # re-measurement: the pairs already explained in this round no longer count
            if not c.validated: continue
            status, idx = cp.add(c); n_val += 1
            if first is None: first = n_prop
            if status == "new": kept += 1; s = separates(c.program, pairs); pairs.weight = torch.where(s & (pairs.opposite > 0), torch.zeros_like(pairs.weight), pairs.weight)
        validator.z_min = base
        log.append({"round": r, "z_threshold": round(z_eff, 2), "proposed": len(cands), "validated": len(ok), "kept_new": kept, "classes": len(cp), "residual": float((pairs.opposite * pairs.weight).sum()) / max(1.0, total_opp),
                    "rejected": _count([c.reason for c in cands if not c.validated])})
        if kept == 0: break
    return Discovery(cp, n_prop, n_val, r + 1, round(time.time() - t0, 3), counter.fit_equivalents, log, first, float((pairs.opposite * pairs.weight).sum()) / max(1.0, total_opp))


def _count(xs):
    c: Dict[str, int] = {}
    for x in xs: c[x] = c.get(x, 0) + 1
    return c


# ================================================================ kinship, families, birth, merging, retirement

def kinship(a: BNode, b: BNode, tol_angle=0.2, tol_center=0.3):
    """Geometric relations between two primitives: parallel (half-planes), concentric (centred)."""
    la, lb = _leaf(a), _leaf(b); out = []
    if la.feat == "dot" and lb.feat == "dot":
        d = abs(math.atan2(math.sin(la.params[0] - lb.params[0]), math.cos(la.params[0] - lb.params[0])))
        if min(d, math.pi - d) < tol_angle: out.append("parallel")
    if a.kind == "translate" and b.kind == "translate" and math.hypot(a.params[0] - b.params[0], a.params[1] - b.params[1]) < tol_center: out.append("concentric")
    return out


@dataclass
class PrimitiveFamily:
    """Parametric family arising from several instances: type + observed parameter ranges (anti-unification of
    primitives: instances → family → compact parametrisation)."""
    type: str
    skeleton: BNode
    lo: List[float]
    hi: List[float]
    support: int

    def contains(self, p: BNode, slack=0.25):
        return skel_key(p) == skel_key(self.skeleton) and all(l - slack * (h - l + 1e-6) <= v <= h + slack * (h - l + 1e-6) for v, l, h in zip(p.param_vector(), self.lo, self.hi))
    def render(self): return f"{self.type}(" + ", ".join(f"θ{i}∈[{l:.2f},{h:.2f}]" for i, (l, h) in enumerate(zip(self.lo, self.hi))) + ")"


def anti_unify_primitives(instances: Sequence[BNode], min_support=3) -> List[PrimitiveFamily]:
    """Groups primitives by skeleton and returns one parametrised family per sufficiently supported group."""
    by: Dict[str, List[BNode]] = {}
    for p in instances: by.setdefault(skel_key(p), []).append(p)
    out = []
    for k, ps in by.items():
        if len(ps) < min_support: continue
        V = torch.tensor([p.param_vector() for p in ps]); out.append(PrimitiveFamily(atom_name(ps[0]), ps[0].skeleton(), V.min(0).values.tolist(), V.max(0).values.tolist(), len(ps)))
    return sorted(out, key=lambda f: -f.support)


@dataclass
class BirthProposal:
    """BIRTH proposal for a family of primitives: a recurring relation between co-present primitives
    (parallel, concentric) suggests an atom with tied parameters. Accepted only if it recurs over several
    tasks AND saves parameters (compression); never on a single example."""
    relation: str
    occurrences: int
    tasks: int
    params_saved_per_use: int
    accepted: bool
    reason: str


class PrimitiveLibrary:
    """Persistent library of validated primitives (instances per task): parametric families, birth
    proposals, merging and retirement. It stores no world: only types, ranges and counters."""

    def __init__(self): self.instances: List[Tuple[str, BNode]] = []; self.relations: Dict[str, List[str]] = {}; self.families: List[PrimitiveFamily] = []; self.usage: Dict[str, int] = {}

    def observe(self, task_id: str, prims: Sequence[BNode]):
        for p in prims: self.instances.append((task_id, p)); self.usage[atom_name(p)] = self.usage.get(atom_name(p), 0) + 1
        for i, a in enumerate(prims):
            for b in prims[i + 1:]:
                for r in kinship(a, b): self.relations.setdefault(f"{r}:{atom_name(a)}+{atom_name(b)}", []).append(task_id)

    def update(self, min_support=3): self.families = anti_unify_primitives([p for _, p in self.instances], min_support); return self.families

    def births(self, min_tasks=3) -> List[BirthProposal]:
        out = []
        for rel, tasks in sorted(self.relations.items()):
            saved = 1 if rel.startswith("parallel") else 2; n = len(set(tasks)); ok = n >= min_tasks
            out.append(BirthProposal(rel, len(tasks), n, saved, ok, "récurrente et compressive" if ok else "pas assez de tâches distinctes (pas de naissance opportuniste)"))
        return out

    def retire(self, min_usage=2):
        """Types never (or almost never) validated: candidates for retirement from the vocabulary."""
        known = {atom_name(a) for a in atoms()}; return sorted(t for t in known if self.usage.get(t, 0) < min_usage)

    def priors(self):
        """Frequencies of the validated types: proposal order for the following worlds."""
        tot = max(1, sum(self.usage.values())); return {t: n / tot for t, n in sorted(self.usage.items(), key=lambda kv: -kv[1])}

    def to_dict(self): return {"usage": self.usage, "families": [{"type": f.type, "skeleton": f.skeleton.to_dict(), "lo": f.lo, "hi": f.hi, "support": f.support} for f in self.families], "relations": {k: len(set(v)) for k, v in self.relations.items()}}


def merge_redundant(literals: Sequence[BNode], X) -> Tuple[List[BNode], int]:
    """Merging: functionally equivalent (or complementary) literals on the observed region → a single one."""
    keep: List[BNode] = []; sigs: List[torch.Tensor] = []; merged = 0
    for p in literals:
        s = predict(p, X.float()).float(); dup = any(float((s == t).float().mean()) >= EQUIV_AGREEMENT for t in sigs)
        if dup: merged += 1
        else: keep.append(p); sigs.append(s)
    return keep, merged


# ================================================================ active measurement directed at the primitives

def primitive_query_scores(cp: CompatiblePrimitives, pool: torch.Tensor):
    """Score of a candidate point = disagreement between primitive hypotheses INSIDE the classes (parameter and
    type uncertainty): measuring there discriminates the competing primitives of the same boundary."""
    sc = torch.zeros(len(pool))
    for c in cp.classes:
        if len(c.members) < 2: continue
        P = torch.stack([predict(m.program, pool).float() for m in c.members]); ref = P[0:1]; P = torch.where((P == ref).float().mean(1, keepdim=True) < 0.5, 1 - P, P); p = P.mean(0); sc += 2 * p * (1 - p)
    return sc

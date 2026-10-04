"""Découverte : transformations (composition de primitives), invariants (ce qui reste stable), cycle de vie des
hypothèses, et LOI CACHÉE dans un monde ouvert précurseur choisie par EXPÉRIMENTATION ACTIVE (MeasurementPlanner).

Boucle fermée : NOUVEAUTÉ → HYPOTHÈSES → MESURER / AGIR → DÉCOUVRIR LA TRANSFORMATION → TESTER LES INVARIANTS
→ FORMER LA COMPÉTENCE → CONSTRUIRE LE CERTIFICAT GÉNÉRATIF → CONSOLIDER → RÉUTILISER.
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from core.measurement import CompatibleStateSet, MeasurementContext, MeasurementOperator, MeasurementOutcome, NoiseModel, Observable, information_gain


# ================================================================ primitives géométriques 2D

def rot(th): c, s = math.cos(th), math.sin(th); return torch.tensor([[c, -s], [s, c]])


PRIMS = {
    "rotate": lambda P, a: P @ rot(a).T,
    "translate": lambda P, a: P + torch.tensor(a),
    "scale": lambda P, a: P * a,
    "reflect_x": lambda P, a: P * torch.tensor([1.0, -1.0]),
}


def fit_primitive(name, P, Q):
    """Paramètre optimal (moindres carrés / Procrustes) d'une primitive pour P -> Q."""
    if name == "rotate":
        Pc, Qc = P - P.mean(0), Q - Q.mean(0); M = Pc.T @ Qc
        return math.atan2(float(M[0, 1] - M[1, 0]), float(M[0, 0] + M[1, 1]))
    if name == "translate": return tuple((Q - P).mean(0).tolist())
    if name == "scale": return float((P * Q).sum() / (P * P).sum().clamp_min(1e-9))
    return None


def apply_chain(chain, P):
    for n, a in chain: P = PRIMS[n](P, a)
    return P


def discover_transformation(P, Q, max_len=2, tol=1e-3):
    """Recherche bornée sur les compositions de primitives (longueur ≤ 2) avec ajustement des paramètres étape par
    étape (le dernier maillon est ajusté sur le résidu). Renvoie (chaîne, erreur, nb de candidats)."""
    best, n = None, 0
    names = list(PRIMS)
    for L in range(1, max_len + 1):
        for combo in itertools.product(names, repeat=L):
            n += 1; chain = []; cur = P
            for i, name in enumerate(combo):
                if i < L - 1:                                     # paramètres intermédiaires : ajustés conjointement par grille
                    pass
            # ajustement : premier maillon par grille / Procrustes, second par moindres carrés sur le résidu
            if L == 1:
                a = fit_primitive(combo[0], P, Q); chain = [(combo[0], a)]
            else:
                cands = []
                for a0 in _grid(combo[0], P, Q):
                    mid = PRIMS[combo[0]](P, a0); a1 = fit_primitive(combo[1], mid, Q); cands.append([(combo[0], a0), (combo[1], a1)])
                chain = min(cands, key=lambda c: float((apply_chain(c, P) - Q).norm()))
            err = float((apply_chain(chain, P) - Q).norm() / max(1e-9, float(Q.norm())))
            if best is None or err < best[1] - 1e-9: best = (chain, err)
        if best[1] < tol: break
    return best[0], best[1], n


def _grid(name, P, Q):
    if name == "rotate": return [k * math.pi / 12 for k in range(-12, 12)] + [fit_primitive("rotate", P, Q)]
    if name == "scale": return [0.5, 0.75, 1.0, 1.5, 2.0, fit_primitive("scale", P, Q)]
    if name == "translate": return [fit_primitive("translate", P, Q), (0.0, 0.0)]
    return [None]


def memorization_baseline(P, Q, Pnew):
    """Mémorisation : plus proche voisin dans la table (P -> Q) + décalage local."""
    d = torch.cdist(Pnew, P); i = d.argmin(1); return Q[i] + (Pnew - P[i])


def transformation_discovery_experiment(seed=0, n_tasks=20, n_points=6):
    rng = random.Random(seed); out = {"discovered_exact": 0, "rule_error": 0.0, "memo_error": 0.0, "candidates": 0, "n": n_tasks}
    for _ in range(n_tasks):
        k = rng.choice([("rotate", "translate"), ("scale", "translate"), ("reflect_x", "translate"), ("rotate", "scale"), ("rotate",)])
        chain = []
        for name in k:
            a = {"rotate": rng.uniform(-3, 3), "translate": (rng.uniform(-2, 2), rng.uniform(-2, 2)), "scale": rng.choice([0.5, 1.5, 2.0]), "reflect_x": None}[name]
            chain.append((name, a))
        g = torch.Generator().manual_seed(rng.randrange(10 ** 6)); P = torch.randn(n_points, 2, generator=g); Q = apply_chain(chain, P) + 0.01 * torch.randn(n_points, 2, generator=g)
        found, err, n = discover_transformation(P, Q); out["candidates"] += n
        Pnew = torch.randn(50, 2, generator=g) * 3                                          # hors de la région vue
        truth = apply_chain(chain, Pnew)
        e_rule = float((apply_chain(found, Pnew) - truth).norm(dim=1).mean()); e_mem = float((memorization_baseline(P, Q, Pnew) - truth).norm(dim=1).mean())
        out["discovered_exact"] += e_rule < 0.1; out["rule_error"] += e_rule; out["memo_error"] += e_mem
    for k in ("rule_error", "memo_error", "candidates"): out[k] = round(out[k] / n_tasks, 4)
    out["discovered_exact"] = round(out["discovered_exact"] / n_tasks, 3); return out


# ================================================================ invariants et cycle de vie

INVARIANT_CANDIDATES = {
    "pairwise_distance": lambda P: torch.pdist(P),
    "angle": lambda P: torch.atan2(*(lambda u, v: (u[0] * v[1] - u[1] * v[0], (u * v).sum()))(P[1] - P[0], P[2] - P[0])).reshape(1),
    "norm_from_origin": lambda P: P.norm(dim=1),
    "x_order": lambda P: P[:, 0].argsort().float(),
    "area": lambda P: (0.5 * ((P[1] - P[0])[0] * (P[2] - P[0])[1] - (P[1] - P[0])[1] * (P[2] - P[0])[0])).reshape(1),
    "distance_ratio": lambda P: torch.pdist(P) / torch.pdist(P).max(),
    "centroid": lambda P: P.mean(0),
}


@dataclass
class Hypothesis:
    """HYPOTHESIS → SUPPORTED → STABLE | REFUTED, avec preuves pour / contre."""
    name: str
    status: str = "HYPOTHESIS"
    support: int = 0
    against: int = 0
    evidence: List[str] = field(default_factory=list)

    def update(self, ok, note=""):
        if ok: self.support += 1
        else: self.against += 1
        self.evidence.append(("+" if ok else "−") + note)
        if self.against > 0: self.status = "REFUTED"
        elif self.support >= 10: self.status = "STABLE"
        elif self.support >= 3: self.status = "SUPPORTED"
        return self.status


def discover_invariants(transform_sampler: Callable, n_traj=8, n_perturb=12, seed=0, tol=1e-4):
    """Trajectoires sous une famille de transformations inconnue ; candidats invariants testés : constance le long
    des trajectoires, puis PERTURBATION (nouvelles instances de la famille) ; sinon réfutés."""
    g = torch.Generator().manual_seed(seed); hyps = {k: Hypothesis(k) for k in INVARIANT_CANDIDATES}
    for t in range(n_traj + n_perturb):
        P = torch.randn(4, 2, generator=g); Q = transform_sampler(P, g)
        for k, f in INVARIANT_CANDIDATES.items():
            if hyps[k].status == "REFUTED": continue
            a, b = f(P), f(Q); ok = bool(torch.allclose(a, b, atol=tol * 100, rtol=1e-3))
            hyps[k].update(ok, "trajectoire" if t < n_traj else "perturbation")
    return hyps


# ================================================================ monde ouvert précurseur avec LOI CACHÉE

PHI_GRID = [k * math.pi / 16 for k in range(-16, 16)]        # 32 rotations
GAIN_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
MIRROR = [False, True]


@dataclass
class LawWorld:
    """Monde continu 2D : objets typés (balle, bloc, « mystère »), positions, frottement ; action pousser(objet,
    direction, intensité) ; capteur de position bruité ; langage limité (« pousse le mystère vers l'est »).
    LOI CACHÉE : le type « mystère » répond à une poussée par d = gain · R(φ) · M · u (M miroir éventuel), au lieu de d = u."""
    phi: float
    gain: float
    mirror: bool
    noise: float = 0.05
    seed: int = 0

    def __post_init__(self): self.g = torch.Generator().manual_seed(self.seed); self.experiments = 0

    def law(self, u):
        v = torch.tensor([u[0], -u[1]]) if self.mirror else torch.tensor(u)
        return self.gain * (rot(self.phi) @ v)

    def push(self, obj_type, direction, magnitude=1.0):
        self.experiments += 1
        u = (magnitude * math.cos(direction), magnitude * math.sin(direction))
        d = self.law(u) if obj_type == "mystere" else torch.tensor(u)
        return d + self.noise * torch.randn(2, generator=self.g)

    def say(self, text):
        """Langage limité : « pousse le mystère vers le nord|sud|est|ouest »."""
        dirs = {"est": 0.0, "nord": math.pi / 2, "ouest": math.pi, "sud": -math.pi / 2}
        obj = "mystere" if "mystère" in text else "balle"; d = next(v for k, v in dirs.items() if k in text)
        return self.push(obj, d)


def hypothesis_space():
    """Lois candidates = compositions de primitives (miroir ∘ rotation ∘ gain) sur une grille — 512 hypothèses."""
    return CompatibleStateSet.product({("law", "phi"): list(range(len(PHI_GRID))), ("law", "gain"): list(range(len(GAIN_GRID))),
                                       ("law", "mirror"): [0, 1]})


class PushOperator(MeasurementOperator):
    """Opérateur de mesure d'une expérience « pousser dans la direction θ » : issue = déplacement discrétisé (grille)."""

    def __init__(self, theta, noise, magnitude=1.0, obj="mystere"):
        super().__init__(Observable("push", "law", "phi", tuple(range(len(PHI_GRID)))), MeasurementContext(sensor=f"push@{theta:.2f}", noise=NoiseModel("gaussian", sigma=noise), cost=0.1))
        self.theta, self.sigma, self.magnitude, self.obj = theta, noise, magnitude, obj
        self.bins = [(x, y) for x in torch.linspace(-2.5, 2.5, 21).tolist() for y in torch.linspace(-2.5, 2.5, 21).tolist()]

    def predicted(self, S):
        th = self.theta; u = self.magnitude * torch.tensor([math.cos(th), math.sin(th)])
        if self.obj != "mystere": return u.expand(len(S), 2).clone()           # loi connue : aucune information
        phi = torch.tensor(PHI_GRID)[S.column(("law", "phi"))]; gain = torch.tensor(GAIN_GRID)[S.column(("law", "gain"))]
        mir = S.column(("law", "mirror")).float(); v = torch.stack([u[0].expand_as(mir), u[1] * (1 - 2 * mir)], 1)
        c, s = torch.cos(phi), torch.sin(phi)
        return gain[:, None] * torch.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], 1)

    def outcomes(self, grid=None): return list(range(len(self.bins)))

    def lik_all(self, S):
        pred = self.predicted(S); B = torch.tensor(self.bins)
        d2 = ((pred[None, :, :] - B[:, None, :]) ** 2).sum(-1); L = torch.exp(-d2 / (2 * max(self.sigma, 0.12) ** 2))
        return (L / L.sum(0, keepdim=True).clamp_min(1e-300)).double()

    def likelihood_obs(self, S, obs):
        pred = self.predicted(S); d2 = ((pred - obs) ** 2).sum(1); return torch.exp(-d2 / (2 * self.sigma ** 2)).double()


def expected_entropy(S, op):
    L = op.lik_all(S); w = S.weights; joint = L * w; po = joint.sum(1); post = joint / po[:, None].clamp_min(1e-300)
    H = -(post * torch.log(post.clamp_min(1e-300))).sum(1); return float((po * H).sum())


def identify_law(world: LawWorld, policy="active", max_exp=30, confidence=0.99, seed=0, directions=8):
    """Actions possibles : pousser (balle | bloc | mystère) dans 8 directions avec 3 intensités. Choix par gain
    d'information attendu (MeasurementPlanner, active) ou au hasard ; croyance mise à jour ; arrêt à P(MAP) ≥ confiance."""
    rng = random.Random(seed); S = hypothesis_space(); thetas = [k * 2 * math.pi / directions for k in range(directions)]
    actions = [(o, th, m) for o in ("balle", "bloc", "mystere") for th in thetas for m in (0.2, 0.5, 1.0)]
    for n in range(1, max_exp + 1):
        if policy == "active":
            ops = [PushOperator(th, max(world.noise, 0.05), m, o) for o, th, m in actions if o == "mystere"]
            op = min(ops, key=lambda o: expected_entropy(S, o))
        else:
            o, th, m = rng.choice(actions); op = PushOperator(th, max(world.noise, 0.05), m, o)
        obs = world.push(op.obj, op.theta, op.magnitude)
        S = S._fork(S.logw + torch.log(op.likelihood_obs(S, obs).clamp_min(1e-300)), f"pousser@{op.theta:.2f}")
        st, p = S.map_state()
        if p >= confidence: break
    st, p = S.map_state()
    return {"phi": PHI_GRID[st[("law", "phi")] if isinstance(st[("law", "phi")], int) else 0], "gain": GAIN_GRID[st[("law", "gain")]],
            "mirror": bool(st[("law", "mirror")]), "p": p, "experiments": n}


def law_correct(world, found):
    return abs(math.remainder(found["phi"] - world.phi, 2 * math.pi)) < 1e-6 and found["gain"] == world.gain and found["mirror"] == world.mirror


def form_skill(world, found):
    """Loi soutenue → TransformationSpec (DomainTransformation), invariants testés, compétence prédictive et
    CERTIFICAT GÉNÉRATIF (générateur de poussées + référence = la loi découverte)."""
    from core.invariant_algebra import DomainTransformation
    from core.generative_certificate import ContinuousGenerator, GenerativeInvariantCertificate
    phi, gain, mirror = found["phi"], found["gain"], found["mirror"]
    def T(U):
        V = U * torch.tensor([1.0, -1.0]) if mirror else U
        return gain * V @ rot(phi).T
    spec = DomainTransformation(f"PushResponse[mystere](φ={phi:.3f}, gain={gain}, miroir={mirror})", T, "physics")
    inv = {"norm_ratio": gain, "angle_offset": phi, "orientation_preserved": not mirror}
    g = torch.Generator().manual_seed(7); U = torch.randn(200, 2, generator=g)
    inv_ok = {"norm_ratio": bool(torch.allclose(T(U).norm(dim=1) / U.norm(dim=1), torch.full((200,), gain), atol=1e-4)),
              "orientation_preserved": (float(torch.det(T(torch.eye(2)))) > 0) == (not mirror)}
    gen = ContinuousGenerator(k=2).fit(U)
    cert = GenerativeInvariantCertificate("predict_push_mystere", "physics", gen, lambda X: (T(X)[:, 0] > 0).long(), "rule", 3,
                                          invariants=tuple(inv), provenance=f"{found['experiments']} expériences actives")
    return spec, inv, inv_ok, cert


def law_discovery_experiment(seed=0, n_worlds=12, noise=0.25):
    rng = random.Random(seed); out = {"active": [], "random": []}
    for i in range(n_worlds):
        w = dict(phi=rng.choice(PHI_GRID), gain=rng.choice(GAIN_GRID), mirror=rng.random() < 0.5)
        for pol in ("active", "random"):
            world = LawWorld(**w, noise=noise, seed=seed * 100 + i); f = identify_law(world, pol, seed=seed * 100 + i)
            spec, inv, inv_ok, cert = form_skill(world, f)
            g = torch.Generator().manual_seed(i); U = torch.randn(64, 2, generator=g)
            pred_err = float((spec(U) - torch.stack([world.law(tuple(u.tolist())) for u in U])).norm(dim=1).mean())
            out[pol].append({"correct": law_correct(world, f), "experiments": f["experiments"], "p": f["p"], "prediction_error": pred_err,
                             "invariants_ok": all(inv_ok.values())})
    s = lambda pol, k: round(sum(float(r[k]) for r in out[pol]) / len(out[pol]), 4)
    return {pol: {"identification": s(pol, "correct"), "experiments": s(pol, "experiments"), "prediction_error": s(pol, "prediction_error"),
                  "invariants_ok": s(pol, "invariants_ok")} for pol in out}

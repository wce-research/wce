"""Boundary Ownership & Fragment Attribution — pour chaque morceau de frontière observée, à quelle primitive latente
appartient-il ?

L'attribution est un problème GLOBAL : tous les milieux observés sont attribués ensemble à un petit nombre de courbes
latentes (cercles, droites), en minimisant une énergie explicite dont chaque terme est rapporté séparément :

    fit          écart des milieux à la courbe qui les possède (tronqué)
    noise        milieux laissés sans propriétaire
    smooth       voisins de même tangente attribués à des propriétaires différents
    complexity   coût par paramètre de chaque courbe utilisée
    absent       frontière ATTENDUE et VISIBLE d'une courbe, sans milieu observé (preuve négative légitime)
    chance       milieux qu'une courbe capterait par hasard : une courbe n'est créditée que de l'excédent (calibration nulle)

Le masque est un objet explicite (`Occlusion`) : une courbe dont la frontière disparaît d'un seul côté d'une autre
courbe est déclarée masquée par elle de ce côté, et cette portion n'est pas comptée comme absente.

Objets :
  Owner                        courbe latente candidate ou retenue, avec les milieux qu'elle possède
  Occlusion                    (masque, masquée, côté, part cachée)
  BoundaryOwnershipHypothesis  pour un fragment : propriétaires possibles pondérés (courbe A, courbe B, masque, limite
                               du domaine, bruit, inconnu) — aucun propriétaire unique n'est forcé
  OwnershipSolution            attribution, postérieure par milieu, masques, énergie, journal

Solveur compact : ajout glouton de courbes (meilleur d'abord), puis au plus trois reprises
« attribution → ré-estimation des paramètres → ré-attribution lissée », chacune gardée seulement si l'énergie baisse.
Aucun modèle appris. Aucune vérité de monde.
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
    """Réglages (coûts de l'énergie, température de la postérieure) choisis sur les scènes de MISE AU POINT ; défauts sinon."""
    try: return {**DEFAULTS, **json.load(open(CONFIG_FILE))["config"]}
    except (OSError, ValueError, KeyError): return dict(DEFAULTS)


@dataclass
class Owner:
    kind: str                       # "circle" | "line"
    params: tuple                   # (cx, cy, r) | (angle, décalage)
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
    """Le masque `occluder` cache la courbe `occluded` là où le prédicat du masque vaut `side`."""
    occluder: int
    occluded: int
    side: bool
    hidden_fraction: float


@dataclass
class BoundaryOwnershipHypothesis:
    """Propriétaires possibles d'un fragment observé. `options` : (rôle, indice de courbe ou None, probabilité), triées."""
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
    labels: torch.Tensor            # [m] indice de courbe, −1 = sans propriétaire
    posterior: torch.Tensor         # [m, K + 1] ; dernière colonne = bruit
    occlusions: List[Occlusion]
    terms: Dict[str, float]         # termes d'énergie séparés + total
    log: List[dict]
    stats: Dict[str, float]

    def top_k(self, k=3) -> torch.Tensor:
        """[m, k] indices de courbe par probabilité décroissante (K = bruit)."""
        return self.posterior.argsort(dim=1, descending=True)[:, : min(k, self.posterior.shape[1])]
    def is_occluder(self, i): return any(o.occluder == i for o in self.occlusions)


# ================================================================ candidats

def candidates(ob: ObservedBoundary, seed=0) -> List[Owner]:
    """Courbes latentes candidates : composantes par consensus (cercles et droites), ré-estimées sur leurs milieux."""
    return [Owner(c.kind, c.params) for c in circle_components(ob, seed=seed) + line_components(ob, seed=seed)]


def refit(o: Owner, S: torch.Tensor) -> Optional[tuple]:
    """Paramètres ré-estimés sur les milieux possédés ; None si l'estimation sort du domaine raisonnable."""
    if len(S) < 4: return None
    if o.kind == "line":
        n, b, _ = line_tls(S); n0 = torch.tensor([math.cos(o.params[0]), math.sin(o.params[0])], dtype=DT); n, b = (n, b) if float(n @ n0) >= 0 else (-n, -b)
        return (math.atan2(float(n[1]), float(n[0])), b) if float(n @ n0) > 0.8 else None
    c, r, _ = circle_geometric(S, o.params[:2], o.params[2], iters=8); return (float(c[0]), float(c[1]), r) if 0.15 <= r <= 4.0 and float((c - torch.tensor(o.params[:2], dtype=DT)).norm()) < 1.0 else None


# ================================================================ masque explicite et visibilité contrefactuelle

def _domain(ob: ObservedBoundary): X = ob.X; return float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 1].min()), float(X[:, 1].max())


def _inside(P, box): return (P[:, 0] >= box[0]) & (P[:, 0] <= box[1]) & (P[:, 1] >= box[2]) & (P[:, 1] <= box[3])


def find_occlusions(owners: Sequence[Owner], ob: ObservedBoundary, tol=0.3, lo=0.2, hi=0.6, share=0.1) -> List[Occlusion]:
    """Pour chaque couple (courbe o, courbe q) : la frontière de o, dans le domaine, est coupée en deux par le prédicat
    de q (seuls comptent comme soutien de o les milieux plus proches de o que de q). Si un côté est soutenu par des milieux (part ≥ hi) et l'autre presque pas (part ≤ lo), q MASQUE o de ce
    côté. Chaque côté doit porter au moins `share` de la frontière : sinon rien ne peut être conclu."""
    box = _domain(ob); out = []; M = ob.points
    for i, o in enumerate(owners):
        P = o.boundary(); P = P[_inside(P, box)]
        if len(P) < 10 or not len(M): continue
        best = None; do = o.distance(M)
        for j, q in enumerate(owners):
            if j == i: continue
            Mo = M[do <= q.distance(M)]                                               # les milieux mieux expliqués par q sont les siens : ils ne soutiennent pas o
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

    Renvoie des masques sur les points rendus : `domain` (dans la boîte des observations), `hidden` (cachée par un
    masque supposé), `expected` (dans le domaine et non cachée), `seen` (un milieu observé à moins de `tol`)."""
    P = render(p, n=n, box=(-9.0, 9.0, -9.0, 9.0), reach=6.0); dom = _inside(P, _domain(ob)); hidden = torch.zeros(len(P), dtype=torch.bool)
    for q, val in occluders: hidden |= predict(q, P.float()).bool() == val
    seen = (torch.cdist(P, ob.points).min(1).values < tol) if len(P) and len(ob.points) else torch.zeros(len(P), dtype=torch.bool)
    return {"points": P, "domain": dom, "hidden": hidden & dom, "expected": dom & ~hidden, "seen": seen}


def visible_support(p: BNode, ob: ObservedBoundary, occluders: Sequence[Tuple[BNode, bool]] = (), tol=0.15) -> Dict[str, float]:
    """Score de VISIBILITÉ CONTREFACTUELLE d'une hypothèse de primitive : milieux expliqués de près, multipliés par la part
    de la frontière attendue visible qui est effectivement observée. La portion cachée par un masque supposé n'entre
    ni au numérateur ni au dénominateur."""
    v = expected_visibility(p, ob, occluders); P = v["points"]; n_exp = int(v["expected"].sum()); cov = float(v["seen"][v["expected"]].float().mean()) if n_exp else 0.0
    sup = float((torch.cdist(ob.points, P[v["expected"]]).min(1).values < tol).sum()) if n_exp and len(ob.points) else 0.0
    return {"support": sup, "coverage": cov, "expected": n_exp / max(1, len(P)), "hidden": float(v["hidden"].float().mean()) if len(P) else 0.0, "score": sup * cov}


# ================================================================ énergie globale

def _neighbors(ob: ObservedBoundary, link=0.35, turn=0.6) -> torch.Tensor:
    M, N = ob.points, ob.normals; A = (torch.cdist(M, M) < link) & ((N @ N.T).abs() > math.cos(turn)); A.fill_diagonal_(False); return A


def absent_fraction(i: int, owners: Sequence[Owner], occ: Sequence[Occlusion], ob: ObservedBoundary, use_occlusion=True) -> float:
    """Part de la frontière attendue visible de la courbe i qui n'a aucun milieu observé."""
    oc = [(owners[o.occluder].program(), o.side) for o in occ if o.occluded == i] if use_occlusion else []; v = expected_visibility(owners[i].program(), ob, oc); e = v["expected"]
    return float((~v["seen"][e]).float().mean()) if int(e.sum()) else 1.0


def chance_support(o: Owner, ob: ObservedBoundary, cfg: dict, ids: Optional[torch.Tensor] = None, tol=0.3) -> float:
    """CALIBRATION PAR LE HASARD : nombre de milieux qu'une courbe capterait si les milieux étaient répandus uniformément
    dans le domaine : densité × aire de la bande de capture, le long de la portion de courbe où elle a des milieux
    (`ids`), ou de toute sa portion dans le domaine si `ids` est omis. Une courbe n'est créditée que de ce qu'elle
    explique AU-DELÀ de ce nombre : sur des étiquettes aléatoires, une courbe ne se paie pas."""
    box = _domain(ob); P = o.boundary(192); inside = _inside(P, box); area = max(1e-9, (box[1] - box[0]) * (box[3] - box[2]))
    if len(P) < 2 or not int(inside.sum()): return 0.0
    if ids is not None: inside = inside & (torch.cdist(P, ob.points[ids]).min(1).values < tol) if len(ids) else torch.zeros_like(inside)
    step = float((P[1:] - P[:-1]).norm(dim=1).median()); band = 2 * SIGMA * math.sqrt(cfg["tau"]); return len(ob.points) * float(inside.sum()) * step * band / area


def energy(labels: torch.Tensor, owners: Sequence[Owner], ob: ObservedBoundary, cfg: dict, A: Optional[torch.Tensor] = None, occ: Optional[Sequence[Occlusion]] = None) -> Dict[str, float]:
    """Score de cohérence globale d'une attribution, termes SÉPARÉS (plus bas = meilleur)."""
    M = ob.points; tau = cfg["tau"]; fit = 0.0; ab = 0.0; ch = 0.0; used = [i for i in range(len(owners)) if int((labels == i).sum())]; occ = find_occlusions(owners, ob) if occ is None and cfg["occlusion"] else (occ or [])
    for i in used:
        m = labels == i; fit += float(((owners[i].distance(M[m]) / SIGMA) ** 2).clamp(max=tau).sum()); ab += cfg["gamma"] * tau * absent_fraction(i, owners, occ, ob, cfg["occlusion"]) * int(m.sum()) if cfg["gamma"] > 0 else 0.0
        ch += cfg["chance"] * tau * min(float(m.sum()), chance_support(owners[i], ob, cfg, torch.nonzero(m)[:, 0])) if cfg["chance"] > 0 else 0.0
    noise = tau * int((labels < 0).sum()); A = _neighbors(ob) if A is None else A; smooth = cfg["beta"] * float((A & (labels[:, None] != labels[None, :])).sum()) / 2; cx = cfg["lam"] * sum(owners[i].n_params for i in used)
    return {"fit": fit, "noise": noise, "smooth": smooth, "complexity": cx, "absent": ab, "chance": ch, "total": fit + noise + smooth + cx + ab + ch, "owners": len(used)}


def _unary(owners: Sequence[Owner], M: torch.Tensor, tau: float) -> torch.Tensor:
    """[m, K + 1] coûts unaires ; dernière colonne = bruit."""
    cols = [((o.distance(M) / SIGMA) ** 2).clamp(max=2 * tau) for o in owners] + [torch.full((len(M),), tau, dtype=DT)]; return torch.stack(cols, 1)


def _assign(owners, ob, cfg, A, sweeps=2) -> torch.Tensor:
    """Ré-attribution : meilleur propriétaire par milieu, puis lissage (les voisins de même tangente se renforcent)."""
    U = _unary(owners, ob.points, cfg["tau"]); K = len(owners); lab = U.argmin(1)
    for _ in range(sweeps if cfg["beta"] > 0 else 0):
        onehot = torch.nn.functional.one_hot(lab, K + 1).to(DT); agree = A.to(DT) @ onehot; lab = (U + cfg["beta"] * (A.sum(1, keepdim=True).to(DT) - agree)).argmin(1)
    return torch.where(lab == K, torch.full_like(lab, -1), lab)


def _local_only(labels: torch.Tensor, ob: ObservedBoundary, link=0.6) -> torch.Tensor:
    """Sans association LONGUE DISTANCE : chaque courbe ne garde que son plus grand amas spatialement connexe."""
    out = labels.clone()
    for i in torch.unique(labels[labels >= 0]).tolist():
        ids = torch.nonzero(labels == i)[:, 0]; f = link_fragments(ob.points[ids], link); keep = f == torch.bincount(f).argmax(); out[ids[~keep]] = -1
    return out


def solve(ob: ObservedBoundary, cfg: Optional[dict] = None, seed=0) -> OwnershipSolution:
    """Attribution globale. 1) candidats ; 2) ajout glouton, meilleur d'abord, de la courbe qui fait le plus baisser
    l'énergie ; 3) au plus `max_iters` reprises : ré-estimation des paramètres sur les milieux possédés →
    ré-attribution lissée → retrait des courbes devenues inutiles ; une reprise n'est gardée que si l'énergie baisse."""
    t0 = time.time(); cfg = {**load_config(), **(cfg or {})}; M = ob.points; m = len(M); cands = candidates(ob, seed); A = _neighbors(ob) if m else torch.zeros(0, 0, dtype=torch.bool); tested = 0; log = []
    if m == 0 or not cands: return OwnershipSolution([], torch.full((m,), -1, dtype=torch.long), torch.ones(m, 1, dtype=DT), [], {"fit": 0.0, "noise": cfg["tau"] * m, "smooth": 0.0, "complexity": 0.0, "absent": 0.0, "chance": 0.0, "total": cfg["tau"] * m, "owners": 0}, log, {"assignments_tested": 0, "candidates": len(cands), "seconds": 0.0})
    post = lambda lab: _local_only(lab, ob) if not cfg["long_range"] else lab
    chosen: List[int] = []; labels = torch.full((m,), -1, dtype=torch.long); E = energy(labels, [], ob, cfg, A)
    while True:                                                                    # ajout glouton
        best = None
        for c in range(len(cands)):
            if c in chosen: continue
            own = [cands[i] for i in chosen + [c]]; lab = post(_assign(own, ob, cfg, A)); e = energy(lab, own, ob, cfg, A); tested += 1
            if e["total"] < E["total"] - 1e-9 and (best is None or e["total"] < best[1]["total"]): best = (c, e, lab)
        if best is None: break
        chosen.append(best[0]); E, labels = best[1], best[2]; log.append({"step": "add", "owner": cands[best[0]].kind, **{k: round(v, 2) for k, v in E.items()}})
    owners = [Owner(cands[i].kind, cands[i].params) for i in chosen]
    for it in range(cfg["max_iters"]):                                             # attribution ↔ paramètres
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
    """CALIBRATION SANS VÉRITÉ : part des milieux que le solveur attribue à une courbe sur des étiquettes ALÉATOIRES
    (aucune frontière n'existe). Le coût `chance` est réglé pour que cette part reste faible."""
    from core.deep_partial import observe
    out = []
    for t in range(trials):
        g = torch.Generator().manual_seed(seed * 1000 + t); X = torch.rand(n, 2, generator=g); X = torch.stack([box[0][0] + X[:, 0] * (box[0][1] - box[0][0]), box[1][0] + X[:, 1] * (box[1][1] - box[1][0])], 1); y = (torch.rand(n, generator=g) < rates[t % len(rates)]).long()
        s = solve(observe(X, y), cfg, t); out.append(float((s.labels >= 0).float().mean()) if len(s.labels) else 0.0)
    return out


# ================================================================ hypothèses d'appartenance par fragment

def ownership_hypotheses(sol: OwnershipSolution, ob: ObservedBoundary, link=0.45, turn=0.6, edge=0.2) -> List[BoundaryOwnershipHypothesis]:
    """Micro-fragments (proximité + continuité tangentielle) et, pour chacun, la distribution de ses propriétaires :
    moyenne de la postérieure de ses milieux. Une courbe qui en masque une autre apparaît avec le rôle `occluder` ;
    un fragment collé au bord de la boîte des observations reçoit une part `domain` ; la masse qui manque à une
    décision nette (probabilité maximale sous 0,5) est rapportée comme `unknown`."""
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


# ================================================================ appui de la composition

def ownership_from_program(prog: BNode, ob: ObservedBoundary, tol=0.25) -> Tuple[torch.Tensor, List[BNode]]:
    """primitive hypotheses → composition hypothesis → constraints → ownership refinement : chaque milieu va à la FEUILLE
    du composite dont la frontière est la plus proche PARMI celles qui sont visibles en ce point (une feuille masquée
    à cet endroit ne peut pas posséder le milieu). Renvoie (indice de feuille par milieu, feuilles)."""
    from core.deep_partial import leaf_paths
    M = ob.points; lp = leaf_paths(prog); lab = torch.full((len(M),), -1, dtype=torch.long)
    if not len(M) or not lp: return lab, [l for _, l in lp]
    D = []
    for path, leaf in lp:
        B = render(leaf, n=160, box=(-9.0, 9.0, -9.0, 9.0), reach=6.0); d = torch.cdist(M, B).min(1).values if len(B) else torch.full((len(M),), 9.0, dtype=DT); vis = visible_mask(prog, path, M) if len(lp) > 1 else torch.ones(len(M), dtype=torch.bool); D.append(torch.where(vis, d, d + 9.0))
    D = torch.stack(D, 1); d, i = D.min(1); lab[d < tol] = i[d < tol]; return lab, [l for _, l in lp]


# ================================================================ des courbes possédées aux primitives

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
    """Part des courbes de frontière d'une hypothèse de primitive qui coïncident avec une courbe retenue par l'attribution."""
    cs = _curves(p); return sum(any(_same(k, a, o) for o in sol.owners) for k, a in cs) / max(1, len(cs))


def filter_by_ownership(programs: Sequence[BNode], sol: OwnershipSolution, min_owned=1.0) -> List[BNode]:
    """Garde les hypothèses de primitives dont TOUTES les frontières (par défaut) sont des courbes possédées : l'attribution
    globale sert de validation aux hypothèses produites sans attribution."""
    return [p for p in programs if owned(p, sol) >= min_owned]


def primitives_from(sol: OwnershipSolution, ob: ObservedBoundary, k=3):
    """Courbes possédées → fragments (un par courbe, modèle imposé) → relations → hypothèses de type, top-k par groupe."""
    from core.deep_fragments import build_graph, make_fragment, rank_by_group, type_hypotheses
    fr = [make_fragment(i, o.ids, ob, (o.kind, o.params)) for i, o in enumerate(sol.owners) if len(o.ids) >= 4]
    for i, f in enumerate(fr): f.index = i
    for a in fr: a.neighbors = [b.index for b in fr if b is not a and float(torch.cdist(a.points, b.points).min()) < 0.6]
    g = build_graph(fr); H = type_hypotheses(g, ob); return rank_by_group(H, g, k, ob), H

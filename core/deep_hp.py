"""High-Precision Primitive Estimation — rendre chaque primitive assez précise pour la composition, sans oracle.

Constat de mise au point (clés 1090–1099, voir le rapport) : les objectifs LOCAUX à une seule primitive (contraste de
paires voisines, plancher sur une grille spatiale) ne sont pas alignés avec la fonction — en partant de la vraie
primitive, ils s'en éloignent. L'objectif aligné est CONJOINT et sans structure : le plancher par cellules,

    plancher(ensemble) = Σ_cellules min(positifs, négatifs),   cellule = vecteur des valeurs des atomes

c'est-à-dire le plus petit nombre d'erreurs atteignable par une fonction booléenne QUELCONQUE de ces atomes. C'est
exactement ce que la composition sur classes d'équivalence peut exploiter ; il est indépendant de la polarité.

    vivier (rappel d'abord) → ensembles compatibles
        → raffinement conjoint des paramètres sur le plancher      (local, borné, point initial conservé)
        → naissance : primitive manquante cherchée dans l'espace des paramètres par gain de plancher,
                      orientée par la morphologie du résidu
        → élagage conditionnel → validation par sous-échantillonnage répété → commit / rollback
        → même aval fort que les autres chaînes (ensembles → composition sur classes → finalisation validée)

Objets :
  PrimitiveEstimate        estimation centrale, incertitude, région compatible, stabilité, provenance, identifiabilité
  ResidualMorphology       forme du résidu : linéaire, arc, paire parallèle, paire concentrique, radiale
  SetImprovement           journal d'un ensemble : plancher et erreur validée avant / après, naissances, décision
  HPRecovery               résultat de la chaîne complète

Toute transformation suit le Validated Transformation Principle : proposer → évaluer sur un critère valide (erreur
hors-échantillon du plancher, sous-échantillonnage répété) → commit ou rollback. Aucune vérité de monde ; aucun oracle.
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
IDENT_TOL = 0.10                 # demi-largeur de région compatible en dessous de laquelle un paramètre est dit identifié (fixée a priori)
MORPHOLOGIES = ("linear", "arc", "parallel", "concentric", "radial", "none")
BIRTH_TYPES = ("half", "band", "disc", "ring", "sector")          # l'anti-disque est le complément du disque : même cellule, même plancher
MORPH_TYPES = {"linear": ("half", "sector", "band"), "arc": ("disc", "ring"), "parallel": ("band", "half"), "concentric": ("ring", "disc"), "radial": ("sector", "half"), "none": BIRTH_TYPES}
PI = math.pi
_hard = lambda z: (z >= 0).to(DT)


def load_config() -> dict:
    """Réglages choisis sur les tâches de MISE AU POINT uniquement ; valeurs par défaut sinon."""
    try:
        with open(CONFIG_FILE) as fh: return {**DEFAULTS, **json.load(fh)["config"]}
    except (OSError, ValueError, KeyError): return dict(DEFAULTS)


# ================================================================ familles de paramètres

def strip(p: BNode) -> BNode: return p.children[0] if p.kind == "not" else p


def families(p: BNode) -> List[str]:
    """Famille géométrique de chaque paramètre du vecteur de `p` : centre, rayon, orientation, décalage, direction,
    demi-ouverture. Les tolérances, les incertitudes et la carte de sensibilité sont rapportées par famille."""
    p = strip(p); l = p.children[0] if p.kind == "translate" else p; pre = ["center", "center"] if p.kind == "translate" else []
    if l.feat == "dot": return pre + ["orientation"] + ["offset"] * (len(l.params) - 1)
    if l.feat == "norm": return pre + ["radius"] * len(l.params)
    return pre + ["direction", "half_width"]


def type_name(p: BNode) -> str:
    from core.deep_fragments import type_of
    return type_of(strip(p))


# ================================================================ estimateurs spécialisés : courbes de frontière

def spacing(X: torch.Tensor) -> float:
    """Espacement médian des observations : fixe l'échelle de localisation d'une frontière (aucune constante de scène)."""
    D = torch.cdist(X.to(DT), X.to(DT)); D.fill_diagonal_(float("inf")); return float(D.min(1).values.median())


def _sequential(inl: torch.Tensor, top: int, min_inliers: int, refit) -> list:
    """Consensus SÉQUENTIEL (« qui explique quoi ») : la meilleure hypothèse est ré-estimée sur ses éléments, gardée,
    puis ses éléments sont retirés du décompte des suivantes. Une courbe courte n'est donc pas noyée sous les
    variantes d'une courbe longue."""
    alive = torch.ones(inl.shape[1], dtype=torch.bool); out = []
    for _ in range(top):
        sc = (inl & alive[None]).sum(1); k = int(sc.argmax())
        if int(sc[k]) < min_inliers: break
        r = refit(k)
        if r is None: inl[k] = False; continue
        comp, idx = r; out.append(comp); alive[idx] = False; alive[inl[k]] = False
    return out


def extract_lines(ob, top=12, seed=0, band_factor=0.6, cos_normal=0.8, min_inliers=4, n_pairs=1500) -> list:
    """Demi-plan — orientation puis décalage. Votes : chaque élément de frontière propose SA droite (normale locale),
    plus des droites par deux éléments voisins. Un élément ne soutient une droite que s'il en est proche ET si sa
    normale locale s'accorde (cela écarte les croisements). Ré-estimation par moindres carrés totaux, consensus
    séquentiel. La bande suit l'espacement des observations."""
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
    """Disque / anti-disque — centre puis rayon. Cercles par trois éléments d'un même voisinage (un arc) ; un élément ne
    soutient un cercle que s'il est sur la bande ET si sa normale locale est RADIALE. Ré-estimation Taubin puis
    géométrique, consensus séquentiel."""
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
    """Courbes → primitives, par FAMILLE, les primitives riches étant des compositions contraintes de courbes simples :
      demi-plan   une droite                              disque / anti-disque   un cercle (les deux polarités)
      bande       deux droites parallèles, orientation partagée (`joint`) ou droites indépendantes
      anneau      deux cercles concentriques, centre partagé (`joint`) ou cercles indépendants
      secteur     deux demi-droites de même sommet : le sommet est l'intersection, et chaque droite doit porter des
                  éléments du côté de SA demi-droite (c'est ce qui distingue un secteur de deux demi-plans sécants)
    Au plus `k` hypothèses par type composé, classées par score tolérant au masquage."""
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
    """Vivier des estimateurs spécialisés : droites et cercles par votes avec normales et consensus séquentiel, puis
    assemblage par famille. Budget compact : au plus `lines` droites et `circles` cercles."""
    from core.deep_partial import observe
    c = {**load_config(), **(cfg or {})}; ob = observe(X, y); return assemble(ob, extract_lines(ob, c["lines"], seed, c["band_factor"], c["cos_normal"]), extract_circles(ob, c["circles"], seed, c["band_factor"], c["cos_normal"]), k, c["joint"])


# ================================================================ plancher par cellules (vectorisé)

def evaluate_batch(skel: BNode, TH: torch.Tensor, Xd: torch.Tensor) -> torch.Tensor:
    """[K, n] valeurs 0/1 de K paramétrisations d'un même squelette."""
    return compile_skeleton(skel)(TH, Xd, _hard)


def bits_of(p: BNode, Xd: torch.Tensor) -> torch.Tensor:
    p = strip(p); return evaluate_batch(p.skeleton(), torch.tensor([p.param_vector()], dtype=DT), Xd)[0].long()


def cell_code(bits: Sequence[torch.Tensor], n: int, skip: Optional[int] = None) -> torch.Tensor:
    o = torch.zeros(n, dtype=torch.long)
    for j, b in enumerate(bits):
        if j != skip: o = o * 2 + b
    return o


def floor_batch(B: torch.Tensor, others: torch.Tensor, y: torch.Tensor, ncell: int) -> torch.Tensor:
    """Plancher de K candidats [K, n] ajoutés à un contexte de cellules `others` [n] : Σ min(positifs, négatifs)."""
    K, n = B.shape; cell = others[None] * 2 + B.long(); yy = y[None].to(DT).expand(K, n)
    pos = torch.zeros(K, ncell * 2, dtype=DT).scatter_add_(1, cell, yy); tot = torch.zeros(K, ncell * 2, dtype=DT).scatter_add_(1, cell, torch.ones(K, n, dtype=DT)); return torch.minimum(pos, tot - pos).sum(1)


def set_floor(bits: Sequence[torch.Tensor], y: torch.Tensor) -> float:
    n = len(y); return float(floor_batch(torch.zeros(1, n, dtype=DT), cell_code(bits, n), y, 2 ** len(bits))[0])


def floor_prediction(bits: Sequence[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    """Étiquette majoritaire de la cellule de chaque observation : la meilleure fonction booléenne de ces atomes."""
    n = len(y); c = cell_code(bits, n); nc = 2 ** len(bits); pos = torch.zeros(nc, dtype=DT).scatter_add_(0, c, y.to(DT)); tot = torch.zeros(nc, dtype=DT).scatter_add_(0, c, torch.ones(n, dtype=DT)); return (2 * pos > tot)[c].long()


def cv_floor(bits: Sequence[torch.Tensor], y: torch.Tensor, repeats=24, fraction=0.2, seed=0) -> float:
    """Critère VALIDE (auto-cohérence) : taux d'erreur HORS-ÉCHANTILLON du plancher, par sous-échantillonnage répété —
    l'étiquette majoritaire de chaque cellule est apprise sur 1 − `fraction` des observations et jugée sur le reste.
    Plus puissant qu'une coupe unique en deux moitiés ; les mêmes tirages servent avant et après une transformation."""
    n = len(y); g = torch.Generator().manual_seed(seed); c = cell_code(bits, n); nc = 2 ** len(bits); test = torch.rand(repeats, n, generator=g) < fraction; tr = (~test).to(DT); yy = y.to(DT)[None].expand(repeats, n); cc = c[None].expand(repeats, n)
    pos = torch.zeros(repeats, nc, dtype=DT).scatter_add_(1, cc, yy * tr); tot = torch.zeros(repeats, nc, dtype=DT).scatter_add_(1, cc, tr); gmaj = ((yy * tr).sum(1, keepdim=True) * 2 > tr.sum(1, keepdim=True)).to(DT)
    lab = torch.where(2 * pos == tot, gmaj.expand(repeats, nc), (2 * pos > tot).to(DT)); pred = lab.gather(1, cc); return float(((pred != yy) & test).sum() / test.sum().clamp_min(1))


# ================================================================ raffinement conjoint, local et borné

def _climb(skel: BNode, th: torch.Tensor, others: torch.Tensor, Xd, y, ncell: int, g, iters: int, K: int, scale: float, margin=0.0) -> Tuple[torch.Tensor, float, int]:
    """Montée locale sans gradient : K perturbations gaussiennes par pas, échelle décroissante ; le point courant reste
    candidat (jamais de dégradation). Une perturbation n'est prise que si elle gagne plus de `margin` erreurs."""
    f = float(floor_batch(evaluate_batch(skel, th[None], Xd), others, y, ncell)[0]); ev = 1
    for it in range(iters):
        C = th[None] + torch.randn(K, len(th), generator=g, dtype=DT) * scale * (0.7 ** it); v = floor_batch(evaluate_batch(skel, C, Xd), others, y, ncell); i = int(v.argmin()); ev += K
        if float(v[i]) < f - margin: f = float(v[i]); th = C[i]
    return th, f, ev


def joint_refine(programs: Sequence[BNode], X: torch.Tensor, y: torch.Tensor, seed=0, sweeps=2, samples=48, scale=0.08, margin=1.5) -> Tuple[List[BNode], dict]:
    """Raffinement CONJOINT d'un ensemble : chaque primitive est ajustée à tour de rôle, les autres fixées, sur le
    plancher par cellules. Local (échelle `scale`), borné (`sweeps` passages), point initial conservé."""
    P = [strip(p) for p in programs]; n = len(y); Xd = X.to(DT)[None]; g = torch.Generator().manual_seed(seed); bits = [bits_of(p, Xd) for p in P]; f0 = set_floor(bits, y); ev = 0; moved = 0
    for sw in range(sweeps):
        for i, p in enumerate(P):
            sk = p.skeleton(); th0 = torch.tensor(p.param_vector(), dtype=DT); th, f, e = _climb(sk, th0, cell_code(bits, n, i), Xd, y, 2 ** (len(P) - 1), g, 4, samples, scale * (0.6 ** sw), margin); ev += e
            if not torch.equal(th, th0): P[i] = sk.with_params(th.tolist()); bits[i] = bits_of(P[i], Xd); moved += 1
    return P, {"floor_before": f0, "floor_after": set_floor(bits, y), "evaluations": ev, "moves": moved}


# ================================================================ morphologie du résidu

@dataclass
class ResidualMorphology:
    """Forme de ce que l'ensemble courant n'explique pas. Oriente la FAMILLE de primitive à proposer."""
    label: str                                  # linear | arc | parallel | concentric | radial | none
    n_errors: int
    points: torch.Tensor                        # milieux des frontières du résidu
    support: Dict[str, int] = field(default_factory=dict)

    @property
    def types(self) -> Tuple[str, ...]: return MORPH_TYPES[self.label]


def residual_morphology(X: torch.Tensor, residual: torch.Tensor, seed=0) -> ResidualMorphology:
    """structured residual → forme. Le résidu est l'étiquette « l'ensemble se trompe ici » ; ses frontières sont celles
    de la primitive manquante. Droites et cercles de consensus y sont cherchés ; deux droites parallèles → paire
    parallèle, deux cercles de même centre → paire concentrique, deux droites sécantes dans le domaine → radiale."""
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


# ================================================================ naissance d'une primitive par gain de plancher

def sample_params(t: str, K: int, M: torch.Tensor, g, box=(-3.0, 3.0, -3.0, 3.0)) -> torch.Tensor:
    """K paramétrisations du type `t` dont la frontière PASSE par un point de `M` (milieux de frontière). Plages
    génériques, liées à la taille du domaine seulement."""
    U = lambda lo, hi: torch.rand(K, generator=g, dtype=DT) * (hi - lo) + lo; coin = lambda: (torch.rand(K, generator=g) < 0.5).to(DT)
    c = M[torch.randint(0, len(M), (K,), generator=g)] if len(M) else torch.stack([U(box[0], box[1]), U(box[2], box[3])], 1); a = U(-PI, PI)
    if t == "half": return torch.stack([a, torch.cos(a) * c[:, 0] + torch.sin(a) * c[:, 1]], 1)
    if t == "band": b = torch.cos(a) * c[:, 0] + torch.sin(a) * c[:, 1]; w = U(0.3, 2.4); s = coin(); return torch.stack([a, b - s * w, b + (1 - s) * w], 1)
    if t in ("disc", "antidisc"): r = U(0.3, 2.5); return torch.stack([c[:, 0] + r * torch.cos(a), c[:, 1] + r * torch.sin(a), r], 1)
    if t == "ring": r = U(0.3, 2.5); w = U(0.3, 1.5); s = coin(); return torch.stack([c[:, 0] + r * torch.cos(a), c[:, 1] + r * torch.sin(a), (r - s * w).clamp_min(0.15), r + (1 - s) * w], 1)
    l = U(0.0, 2.5); w = U(0.2, 1.5); s = 2 * coin() - 1; return torch.stack([c[:, 0] - l * torch.cos(a), c[:, 1] - l * torch.sin(a), a + s * w, w], 1)        # secteur : une demi-droite du sommet passe par le point


def birth(programs: Sequence[BNode], X: torch.Tensor, y: torch.Tensor, scene_points: Optional[torch.Tensor] = None, seed=0, samples=1500, top=4, gain=3.0, per_param=1.4, morphology=True) -> Tuple[Optional[BNode], dict]:
    """structured residual → missing primitive hypothesis.

    La primitive manquante est cherchée DANS L'ESPACE DES PARAMÈTRES, par le gain de plancher qu'elle apporte À
    L'ENSEMBLE COURANT (utilité conditionnelle), et non par son score isolé. Les candidats passent par les frontières
    du résidu (et, pour moitié, par celles de la scène) ; la morphologie du résidu répartit le budget entre familles.
    Acceptée seulement si le gain dépasse `gain` + `per_param` erreurs par paramètre."""
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
    """Élagage CONDITIONNEL : une primitive n'est retirée que si l'ensemble SANS elle ne perd presque rien. Une primitive
    faible isolément mais utile en interaction (opérande d'un ou exclusif) est donc conservée."""
    P = list(programs); Xd = X.to(DT)[None]; bits = [bits_of(p, Xd) for p in P]
    while len(P) > 1:
        f = set_floor(bits, y); loss = [set_floor(bits[:i] + bits[i + 1:], y) - f for i in range(len(P))]; i = min(range(len(P)), key=lambda j: loss[j])
        if loss[i] >= gain: break
        P.pop(i); bits.pop(i)
    return P


# ================================================================ incertitude, stabilité, identifiabilité

@dataclass
class PrimitiveEstimate:
    """Une primitive candidate n'est jamais un point nu : la composition peut savoir qu'un paramètre est incertain."""
    program: BNode
    type: str
    families: List[str]
    central: List[float]
    uncertainty: List[float]                    # écart-type des paramétrisations compatibles, par paramètre
    region: List[Tuple[float, float]]           # intervalle compatible, par paramètre
    stability: float                            # accord fonctionnel moyen des réestimations sur rééchantillonnages
    provenance: str
    identifiability: str
    consensus: Optional[BNode] = None           # médiane des réestimations (bootstrap)

    def by_family(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for f, u in zip(self.families, self.uncertainty): out[f] = max(out.get(f, 0.0), u)
        return out


def describe(programs: Sequence[BNode], X: torch.Tensor, y: torch.Tensor, seed=0, slack=2.0, samples=96, boots=6, provenance: Optional[Sequence[str]] = None, tol=IDENT_TOL) -> List[PrimitiveEstimate]:
    """Pour chaque primitive d'un ensemble, le reste fixé :
      région compatible   paramétrisations dont le plancher ne dépasse pas le meilleur de plus de `slack` erreurs
      incertitude         écart-type de ces paramétrisations, par paramètre
      stabilité           réestimation sur `boots` rééchantillonnages des observations ; accord fonctionnel moyen
      consensus           médiane des réestimations
      identifiabilité     tous les paramètres identifiés (demi-largeur ≤ tol) / certains / aucun."""
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
    """Consensus au lieu d'un ajustement unique : chaque primitive est remplacée par la MÉDIANE de ses réestimations sur
    rééchantillonnages. Validé : la médiane n'est gardée que si l'erreur hors-échantillon du plancher ne monte pas."""
    E = describe(programs, X, y, seed, boots=boots); P = [strip(p) for p in programs]; Q = [e.consensus for e in E]; Xd = X.to(DT)[None]; a = cv_floor([bits_of(p, Xd) for p in P], y, seed=seed); b = cv_floor([bits_of(p, Xd) for p in Q], y, seed=seed); ok = (b <= a) or not validate
    return (Q if ok else P), {"cv_before": a, "cv_after": b, "committed": bool(ok)}


# ================================================================ amélioration validée d'un ensemble

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
    """Erreurs, point par point sur (Xb, yb), de la meilleure fonction booléenne des atomes apprise sur (Xa, ya) :
    étiquette majoritaire de chaque cellule, apprise d'un côté, jugée de l'autre (cellule inconnue → majorité globale)."""
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
    """primitive set → raffinement conjoint → [résidu structuré → naissance → raffinement]* → élagage → validation.

    Boucle bornée (`births` naissances au plus). Validation (Validated Transformation Principle), au choix :
      "holdout"  la PROCÉDURE entière est rejouée sur des sous-échantillons d'apprentissage (`holdout_repeats` coupes) et
                 jugée sur les observations qu'elle n'a pas vues ; elle n'est appliquée à toutes les observations que si
                 elle y corrige significativement plus d'erreurs qu'elle n'en crée (test apparié, z ≥ `holdout_z`) ;
      "cv"       erreur hors-échantillon du plancher par sous-échantillonnage répété, mêmes tirages avant et après
                 (optimiste : les paramètres ont vu toutes les observations) ;
      False      aucune.
    Sans gain validé, l'ensemble INITIAL est rendu tel quel (rollback)."""
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


# ================================================================ chaîne complète

def hp_pool(X: torch.Tensor, y: torch.Tensor, seed=0, cfg: Optional[dict] = None) -> List[Tuple[BNode, float, str]]:
    """Vivier « rappel d'abord » : estimateurs existants (plusieurs paramétrisations par type) ET estimateurs spécialisés.
    Aucune élimination ; l'aval départage."""
    from core.deep_lpr import pool
    c = {**load_config(), **(cfg or {})}; return (pool(X, y, c["k"], seed, tuple(c["sources"])) if c["sources"] else []) + (curve_pool(X, y, c["k"], seed, c) if c["curves"] else [])


@dataclass
class HPRecovery:
    recovery: object                            # LatentRecovery de l'aval fort
    improvements: List[SetImprovement]
    estimates: List[PrimitiveEstimate]
    added: List[Tuple[BNode, float, str]]
    stats: Dict[str, float]

    @property
    def program(self): return self.recovery.program


def recover(X: torch.Tensor, y: torch.Tensor, seed=0, cfg: Optional[dict] = None, discovered: Optional[Sequence[Tuple[BNode, float, str]]] = None) -> HPRecovery:
    """observations → primitives de haute précision → ensemble → composition sur classes → finalisation validée.

    Les primitives raffinées et nées REJOIGNENT le vivier (les originales restent) ; l'aval fort — le même que pour
    les autres chaînes — choisit. Rien n'est détruit."""
    from core.deep_e2e import hypotheses_from
    from core.deep_lpr import load_config as lpr_config, pool, select_sets, strong_downstream
    from core.deep_partial import observe
    c = {**load_config(), **(cfg or {})}; d = lpr_config(); t0 = time.time(); disc = list(discovered) if discovered is not None else hp_pool(X, y, seed, c); H = hypotheses_from(disc, X)
    sets = select_sets(H, y, d["max_atoms"], d["beam"], d["pair_pool"], d["min_gain"], d["n_sets"], d["pairs"]); M = observe(X, y).points; imps = []; added: List[Tuple[BNode, float, str]] = []
    for j, s in enumerate(sets.sets[:c["n_refine_sets"]]):
        im = improve_set(s.programs, X, y, seed + j, c, M, d["max_atoms"]); imps.append(im)
        if im.committed: added += [(p, 2.0 - 0.1 * j, "hp:" + src) for p, src in zip(im.programs, im.provenance)]
    if not sets.sets and c["births"] > 0:                                          # vivier vide : la naissance part de rien
        im = improve_set([], X, y, seed, c, M, d["max_atoms"]); imps.append(im); added += [(p, 2.0, "hp:" + src) for p, src in zip(im.programs, im.provenance)]
    R = strong_downstream(disc + added, X, y, {"fit": d["fit"]}); est = describe(R.chosen.programs, X, y, seed, provenance=R.chosen.provenance) if c["describe"] and R.chosen else []
    return HPRecovery(R, imps, est, added, {"pool": len(disc), "added": len(added), "committed_sets": sum(i.committed for i in imps), "births": sum(b.get("accepted", False) for i in imps for b in i.births), "evaluations": sum(i.evaluations for i in imps), "seconds": round(time.time() - t0, 3)})

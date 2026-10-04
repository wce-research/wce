"""Génération compositionnelle FACTORISÉE de candidats profonds (sous-jalon Deep Candidate Generation).

Constat du diagnostic : l'espace des structures croît comme 8·108·2 600·80 000·2,8 M·10⁸ (profondeurs 1 à 6) alors que
la recherche par ajustements en génère ≈ 150 ; et les opérateurs « un atome à la fois » ne construisent que des chaînes.

Idée : séparer la GÉOMÉTRIE de la LOGIQUE.
  1. Géométrie — dans une composition and / or / xor, tout morceau de frontière appartient à la frontière d'UN atome.
     Les frontières atomiques (droites, cercles) se lisent donc localement, sans connaître la composition : points de
     frontière → consensus robuste (type RANSAC) → primitives géométriques → littéraux (régions typées).
  2. Logique — une fois les littéraux fixés, une structure est une formule booléenne sur leurs SIGNATURES (valeurs sur
     les points observés). Les compositions se construisent de bas en haut par tables de signatures (dédoublonnées),
     les exigences descendent sous forme de TROUS typés (cible + masque des points qui comptent), et la rencontre se
     fait sur la signature commune, par un produit bilinéaire qui évalue toutes les paires d'un coup. Aucun ajustement
     par gradient n'est nécessaire avant d'avoir choisi les quelques candidats à polir.

Ce module n'importe aucun module de monde et ne reçoit jamais la vérité.
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

# ---- types structurels légers ----------------------------------------------------------------------------------------
# Field   : fonction scalaire de l'espace (dot, norm, angle)
# Region  : prédicat sur l'espace (un Field seuillé, éventuellement transporté) — c'est le type des littéraux
# Logic   : Region × Region → Region (and, or, xor)
# Transform : Region → Region (translate) ; ne s'applique qu'à un Field centré (norm, angle), jamais à dot
TYPES = ("Field", "Region", "Logic", "Transform")
TYPE_RULES = {"pred": ("Field", "Region"), "translate": ("Region", "Region"), "not": ("Region", "Region"), "and": ("Region×Region", "Region"), "or": ("Region×Region", "Region"), "xor": ("Region×Region", "Region")}


def well_typed(n: BNode) -> bool:
    """Règles de typage : un combinateur prend deux Régions ; translate ne transporte qu'un champ centré (norm, angle)
    ou sa négation ; not porte sur une Région atomique ; pas de translate∘translate ni de not∘not."""
    if n.kind == "pred": return n.feat in ("dot", "norm", "angle") and n.op in ("le", "ge", "in") and not n.children
    if n.kind in OPS: return len(n.children) == 2 and all(well_typed(c) for c in n.children)
    if n.kind == "translate":
        c = n.children[0]; c = c.children[0] if c.kind == "not" else c; return c.kind == "pred" and c.feat in ("norm", "angle")
    if n.kind == "not": c = n.children[0]; return c.kind == "pred"
    return False


# ================================================================ 1. géométrie : frontières atomiques

def boundary_points(X, y, k=6):
    """Points de frontière : milieux des paires de voisins d'étiquettes (débruitées par vote local) opposées."""
    X = X.to(F32); n = len(X); D = torch.cdist(X, X); nb = D.topk(min(k + 1, n), largest=False).indices                    # [n, k+1] (soi compris)
    yh = (y.to(F32)[nb].mean(1) >= 0.5)                                                                                    # vote local
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
    """Ajustement algébrique (Kåsa) : x² + y² = 2ax + 2by + c."""
    A = torch.cat([2 * P, torch.ones(len(P), 1)], 1); sol = torch.linalg.lstsq(A, (P ** 2).sum(1, keepdim=True)).solution.flatten()
    ctr = sol[:2]; r = math.sqrt(max(1e-6, float(sol[2] + ctr @ ctr))); return ctr, r


@dataclass
class GeoPrimitive:
    kind: str                       # "line" | "circle"
    params: tuple                   # line : (angle, offset) ; circle : (cx, cy, r)
    inliers: int
    support: torch.Tensor           # points de frontière expliqués


def _sep_line(P, Q, M, nrm, b, delta):
    """Une paire de frontière soutient une droite si ses deux points sont de part et d'autre ET son milieu en est proche."""
    return (((P @ nrm.T - b) * (Q @ nrm.T - b)) < 0) & ((M @ nrm.T - b).abs() < delta)


def _sep_circle(P, Q, M, ctr, r, delta):
    return (((torch.cdist(P, ctr) - r) * (torch.cdist(Q, ctr) - r)) < 0) & ((torch.cdist(M, ctr) - r).abs() < delta)


def detect_primitives(boundary, max_prims=10, min_inliers=6, delta=0.22, n_hyp=1500, seed=0) -> List[GeoPrimitive]:
    """Consensus robuste séquentiel : la meilleure droite ou le meilleur cercle au sens du nombre de PAIRES de
    frontière qu'il SÉPARE (contrainte explicite : une frontière atomique passe entre deux points d'étiquettes
    opposées), raffiné par moindres carrés, retiré, et ainsi de suite."""
    B, P, Q = (t.to(F32) for t in boundary); g = torch.Generator().manual_seed(seed); out: List[GeoPrimitive] = []; alive = torch.ones(len(B), dtype=torch.bool)
    while len(out) < max_prims and int(alive.sum()) >= min_inliers:
        R, Pa, Qa = B[alive], P[alive], Q[alive]
        if len(R) < 3: break
        ln, lb = _line_hyps(R, n_hyp, g); cc, cr = _circle_hyps(R, n_hyp, g); best = None
        if len(ln):
            cnt = _sep_line(Pa, Qa, R, ln, lb[None], delta).sum(0); i = int(cnt.argmax()); best = ("line", i, int(cnt[i]))
        if len(cc):
            cnt = _sep_circle(Pa, Qa, R, cc, cr[None], delta).sum(0); i = int(cnt.argmax())
            if best is None or int(cnt[i]) > best[2] + 1: best = ("circle", i, int(cnt[i]))                              # à égalité, la droite (plus simple)
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
    """Découverte d'atomes par COUVERTURE SÉQUENTIELLE des paires de frontière : à chaque tour, tous les types d'atomes
    sont ajustés (en lot) pour séparer les paires encore inexpliquées ; les meilleurs atomes distincts sont gardés et
    les paires qu'ils séparent sont dépondérées. Renvoie (atomes ajustés, statistiques)."""
    from core.boundary_program import _inits
    from core.deep_fit import DT, separation_fit
    M, P, Q = boundary_points(X, y); stats = {"pairs": len(M), "rounds": 0, "atoms": 0, "covered": 0.0}
    if len(M) < 6: return [], stats
    w = torch.ones(len(M), dtype=DT); out: List[BNode] = []; g = torch.Generator().manual_seed(seed); Xf, yl = X.float(), y.long()
    for rnd in range(rounds):
        cands = []
        for a in atom_skeletons:
            inits = torch.stack([v.to(DT) for v in _inits(a, Xf, yl, n_init, g)])
            c = M[torch.multinomial(w.float().clamp_min(1e-6), n_init, replacement=True, generator=g)]                    # centres amorcés sur des points de frontière encore inexpliqués
            if a.kind == "translate": inits[:, :2] = c.to(DT) + 0.6 * torch.randn(n_init, 2, generator=g, dtype=DT)
            th, sep = separation_fit(a, P, Q, w, inits, steps=steps, counter=counter)
            cands += [(float(sep[k]), a.with_params(th[k].tolist())) for k in range(n_init)]
        cands.sort(key=lambda c: -c[0]); kept = 0
        for score, prog in cands:
            if kept >= keep_per_round or score < min_pairs: break
            s = (predict(prog, P) != predict(prog, Q))
            if float((w * s.to(DT)).sum()) < min_pairs: continue                                                          # déjà couvert par un atome gardé à ce tour
            out.append(prog); w = torch.where(s, w * 0.2, w); kept += 1
        stats["rounds"] = rnd + 1
        if kept == 0: break
    stats["atoms"] = len(out); stats["covered"] = float(1 - w.mean()) / 0.8
    return out, stats


# ================================================================ littéraux (régions typées) issus des primitives

@dataclass
class Literal:
    program: BNode                  # atome de la grammaire, paramètres géométriques
    source: str                     # "half" | "disc" | "antidisc" | "band" | "notband" | "ring" | "sector" | "notsector"
    support: int


def _wrap(a): return math.atan2(math.sin(a), math.cos(a))
def _T(c, ch): return BNode("translate", params=(float(c[0]), float(c[1])), children=(ch,))


def literals_from(prims: Sequence[GeoPrimitive], macros=True, max_sectors=4) -> List[Literal]:
    """Littéraux = atomes de la grammaire. Base minimale : demi-plans (deux orientations) et disques (+ complément).
    Macros (atomes redondants de la grammaire, reconnus par CONTRAINTES explicites) : bande = deux droites parallèles ;
    anneau = deux cercles concentriques ; secteur = deux demi-droites issues d'un même point."""
    L: List[Literal] = []; lines = [p for p in prims if p.kind == "line"]; circles = [p for p in prims if p.kind == "circle"]
    for p in lines:
        a, b = p.params; L.append(Literal(BNode("pred", "dot", "le", (a, b)), "half", p.inliers)); L.append(Literal(BNode("pred", "dot", "le", (_wrap(a + math.pi), -b)), "half", p.inliers))
    for p in circles:
        cx, cy, r = p.params; L.append(Literal(_T((cx, cy), BNode("pred", "norm", "le", (r,))), "disc", p.inliers)); L.append(Literal(_T((cx, cy), BNode("pred", "norm", "ge", (r,))), "antidisc", p.inliers))
    if not macros: return L
    for i, p in enumerate(lines):                                       # bande : contrainte de parallélisme
        for q in lines[i + 1:]:
            a1, b1 = p.params; a2, b2 = q.params; da = _wrap(a1 - a2)
            if abs(da) > math.pi / 2: a2, b2 = _wrap(a2 + math.pi), -b2; da = _wrap(a1 - a2)
            if abs(da) < 0.2 and 0.3 < abs(b1 - b2) < 2.4:
                lo, hi = sorted((b1, b2)); band = BNode("pred", "dot", "in", (a1, lo, hi)); s = min(p.inliers, q.inliers)
                L.append(Literal(band, "band", s)); L.append(Literal(BNode("not", children=(band,)), "notband", s))
    for i, p in enumerate(circles):                                     # anneau : contrainte de concentricité
        for q in circles[i + 1:]:
            c1, c2 = torch.tensor(p.params[:2]), torch.tensor(q.params[:2]); r1, r2 = sorted((p.params[2], q.params[2]))
            if float((c1 - c2).norm()) < 0.3 and r2 - r1 > 0.3: L.append(Literal(_T(((c1 + c2) / 2).tolist(), BNode("pred", "norm", "in", (r1, r2))), "ring", min(p.inliers, q.inliers)))
    sect = []
    for i, p in enumerate(lines):                                       # secteur : deux demi-droites de même origine, chacune soutenue d'un seul côté
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


# ================================================================ 2. logique : tables de signatures, trous typés, rencontre

@dataclass
class Table:
    """Niveau j : structures à j littéraux, dédoublonnées par SIGNATURE (valeurs sur les points observés).
    `alts` conserve des formes alternatives de même signature (ne pas fusionner trop tôt des structures différentes)."""
    sig: torch.Tensor                       # [N, n] float32 0/1
    params: torch.Tensor                    # [N] nombre de paramètres
    trees: List[tuple]                      # arbre de chaque entrée : ("lit", i) | (op, gauche, droite)
    alts: Dict[int, List[tuple]] = field(default_factory=dict)
    merged: int = 0                         # structures fusionnées par signature identique
    degenerate: int = 0                     # compositions dégénérées écartées (interface incompatible)

    def __len__(self): return len(self.trees)


def _combine(op, a, b): return a * b if op == "and" else a + b - a * b if op == "or" else a + b - 2 * a * b


def _dedup(sig, params, trees, known: Sequence[torch.Tensor] = (), keep_alts=2):
    """Dédoublonnage par signature (et élimination des constantes et des signatures déjà connues à un niveau
    inférieur : une composition qui ne change rien n'est pas une nouvelle structure)."""
    n = sig.shape[1]; w = torch.arange(1, n + 1, dtype=torch.float64); r = torch.Generator().manual_seed(12345); w2 = torch.rand(n, generator=r, dtype=torch.float64)
    h = (sig.double() @ w) * 1e3 + (sig.double() @ w2); s = sig.sum(1); const = (s == 0) | (s == n)
    kh = torch.cat([(k.double() @ w) * 1e3 + (k.double() @ w2) for k in known]) if len(known) else torch.zeros(0, dtype=torch.float64)
    seen: Dict[float, int] = {float(v): -1 for v in kh.tolist()}; keep = []; alts: Dict[int, List[tuple]] = {}; merged = 0; deg = int(const.sum())
    order = torch.argsort(params, stable=True).tolist()                                # la description la plus courte représente la signature
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
    """Tables montantes S1, S2, S3 (toutes les compositions, dédoublonnées par signature)."""
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
    """Rencontre sur la signature commune : erreur pondérée de op(a, b) contre la cible, pour TOUTES les paires (a, b)
    et les trois combinateurs, par un seul produit matriciel. Renvoie E [3, |A|, |B|] dans l'ordre (and, or, xor)."""
    v = w * (1 - 2 * y); c0 = float((w * y).sum()); M = (A_sig * v) @ B_sig.T; a = (A_sig @ v)[:, None]; b = (B_sig @ v)[None, :]
    return torch.stack([c0 + M, c0 + a + b - M, c0 + a + b - 2 * M])


@dataclass
class Hole:
    """Trou typé d'un programme partiel : ce que la sous-structure manquante DOIT faire."""
    type: str                       # toujours "Region" dans cette grammaire
    target: torch.Tensor            # comportement requis sur les points observés
    care: torch.Tensor              # poids : 0 = point indifférent pour ce trou
    depth_left: int
    unavoidable: float              # erreurs qu'aucun remplissage ne peut corriger (borne inférieure)


@dataclass
class PartialStructure:
    """Programme partiel `op(known, ?X)` : une partie connue et un trou typé."""
    op: str
    known: tuple
    hole: Hole

    def lower_bound(self): return self.hole.unavoidable


def make_hole(op: str, known_sig, y, w, depth_left) -> Hole:
    """Exigence descendante : pour que op(known, ?X) = y, que doit valoir ?X et où cela compte-t-il ?"""
    if op == "xor": return Hole("Region", (y - known_sig).abs(), w, depth_left, 0.0)
    if op == "or": return Hole("Region", y, w * (1 - known_sig), depth_left, float((w * known_sig * (1 - y)).sum()))
    return Hole("Region", y, w * known_sig, depth_left, float((w * (1 - known_sig) * y).sum()))


def _evidence(err, n, params, atoms):
    """Même forme que l'évidence approchée existante (bruit profilé, pénalité de paramètres et de taille) — utilisée
    ici comme crible AVANT ajustement ; le classement final reste celui du système."""
    eps = ((err + 0.5) / (n + 1.0)).clamp(1e-3, 0.49); ll = err * eps.log() + (n - err) * (1 - eps).log()
    return ll - 0.5 * params * math.log(max(1.0, n)) - (2 * atoms - 1) * 3.0 * math.log(2)


def _top_pairs(E, pa, pb, n, atoms, top, chunk_rows=None):
    """Meilleures paires (score d'évidence) d'un tenseur d'erreurs [3, A, B] → [(score, erreur, op, i, j)]."""
    ev = _evidence(E, n, (pa[:, None] + pb[None, :])[None], atoms); k = min(top, ev.numel()); v, idx = ev.flatten().topk(k)
    A, B = E.shape[1], E.shape[2]; out = []
    for s, f in zip(v.tolist(), idx.tolist()):
        o, r = divmod(f, A * B); i, j = divmod(r, B); out.append((s, float(E[o, i, j]), OPS[o], i, j))
    return out


class SignatureSynthesizer:
    """Synthèse de structures sur signatures : bas-en-haut (tables), haut-en-bas (trous), rencontre (produit bilinéaire)."""

    def __init__(self, lit_sig, lit_params, y, max_depth=6, top=6, hole_beam=8, time_budget=None):
        self.y = y.to(F32); self.n = float(len(y)); self.w = torch.ones(len(y), dtype=F32); self.max_depth, self.top, self.hole_beam = max_depth, top, hole_beam
        self.t0 = time.time(); self.time_budget = time_budget; self.tabs, self.stats = build_tables(lit_sig.to(F32), lit_params.to(F32)); self.stats["pairs_scored"] = 0; self.stats["holes"] = 0; self.stats["holes_pruned"] = 0
        self.S1, self.S2, self.S3 = self.tabs

    def _late(self): return self.time_budget is not None and time.time() - self.t0 > self.time_budget

    def _meet(self, A: Table, B: Table, y, w, atoms, top, chunk=4000):
        """Toutes les compositions op(a ∈ A, b ∈ B) contre (cible, masque) : les `top` meilleures."""
        best = []
        for s in range(0, len(A), chunk):
            E = pair_errors(A.sig[s:s + chunk], B.sig, y, w); self.stats["pairs_scored"] += E.numel()
            for sc, e, op, i, j in _top_pairs(E, A.params[s:s + chunk], B.params, float(w.sum()), atoms, top): best.append((sc, e, (op, A.trees[s + i], B.trees[j]), float(A.params[s + i] + B.params[j])))
        best.sort(key=lambda z: -z[0]); return best[:top]

    def solve(self, y, w, depth, top):
        """Meilleures structures à exactement `depth` littéraux pour (cible, masque). Profondeur ≤ 4 : rencontre exacte
        de tables. Profondeur 5–6 : rencontre de tables + programmes partiels à trou, élagués par borne inférieure."""
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
            e = (w * (self.S3.sig - y).abs()).sum(1); idx = e.topk(min(3000, len(e)), largest=False).indices                    # crible : compatibilité observationnelle
            sub = Table(self.S3.sig[idx], self.S3.params[idx], [self.S3.trees[i] for i in idx.tolist()]); self.stats["cap_S3xS3"] = len(self.S3) - len(sub); out += self._meet(sub, sub, y, w, 6, top)
        # programmes partiels op(known, ?X) : known ∈ S1 (et S2 à la profondeur 6), trou de profondeur restante
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
        """Candidats par profondeur (les `top` meilleurs de chaque profondeur) + formes alternatives de même signature."""
        out = []
        for d in range(1, self.max_depth + 1):
            if self._late(): self.stats["time_stopped_at_depth"] = d; break
            for sc, e, tree, par in self.solve(self.y, self.w, d, self.top): out.append({"depth": d, "score": sc, "errors": e, "tree": tree})
        return out


def select_literals(sig, max_literals=14, near=0.03):
    """Borne le nombre de littéraux AVANT toute composition : quasi-doublons écartés (désaccord < `near`), puis
    sélection gloutonne du plus éloigné (diversité de signature). Ne regarde jamais la cible : un atome masqué par un
    XOR n'est pas corrélé à la cible et serait éliminé par un critère d'accord."""
    n = len(sig); D = torch.cdist(sig, sig, p=1) / sig.shape[1]; keep = [0]; mind = D[0].clone()
    while len(keep) < min(max_literals, n):
        i = int(mind.argmax())
        if float(mind[i]) < near: break
        keep.append(i); mind = torch.minimum(mind, D[i])
    return keep


def tree_to_program(tree, literals: Sequence[Literal]) -> BNode:
    if tree[0] == "lit": return literals[tree[1]].program
    return BNode(tree[0], children=(tree_to_program(tree[1], literals), tree_to_program(tree[2], literals)))


# ================================================================ générateur complet

@dataclass
class Generation:
    programs: List[BNode]                    # candidats (structure + paramètres géométriques), prêts à être polis
    info: List[dict]
    stats: dict
    literals: List[Literal]


def generate_candidates(X, y, max_depth=6, top=6, seed=0, macros=True, typed=True, time_budget=40.0, max_prims=10, extra_literals: Sequence[BNode] = (), geometry=True, max_literals=14) -> Generation:
    """Génération factorisée : littéraux (géométrie des frontières et / ou atomes fournis, par exemple ajustés par
    gradient) → synthèse sur signatures → candidats."""
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

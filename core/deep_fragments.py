"""Fragment Attribution & Primitive Type Inference — quelles portions observées appartiennent à quelle primitive, et quel
type de primitive est compatible avec chaque groupe de fragments ?

    milieux observés → fragments → graphe de compatibilités → groupes → familles → types → paramètres

Objets :
  BoundaryFragment               une portion de frontière OBSERVÉE (jamais assignée d'office à une primitive unique)
  CompatibleFragmentAssignments  rôles compatibles d'un fragment (droite, arc, bord d'anneau, rayon de secteur, bruit…)
  FragmentGraph                  nœuds = fragments ; arêtes typées = compatibilités géométriques
  TypeHypothesis                 (famille, type, programme) + signaux SÉPARÉS + score fusionné
  OcclusionOwnership             pourquoi une portion attendue de frontière manque : cachée, hors domaine, bruit, absente

Deux erreurs sont séparées : l'ATTRIBUTION (regroupement des milieux par source) et le TYPE (famille puis type).
Petite grammaire de familles : line-like, circle-like, paired-lines, paired-circles, radial-composite — projetée ensuite
sur demi-plan, disque / anti-disque, bande, anneau, secteur.

Règles géométriques et scores explicites uniquement : aucun classifieur appris. Aucune vérité de monde.
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
    """Portion de frontière observée. `model` est l'ajustement local préféré, pas une attribution à une primitive."""
    index: int
    ids: torch.Tensor               # provenance : indices des milieux dans ObservedBoundary.points
    points: torch.Tensor
    normals: torch.Tensor           # orientation locale
    model: str                      # "line" | "circle"
    params: tuple                   # (angle, décalage) | (cx, cy, r)
    curvature: float                # 1 / rayon, 0 pour une droite
    length: float
    endpoints: torch.Tensor         # [2, 2]
    rms_line: float
    rms_circle: float
    confidence: float               # n / (n + 4), pondéré par la qualité de l'ajustement
    polarity: Optional[int] = None  # côté de la normale du modèle où l'étiquette vaut 1 (+1 / −1), si lisible
    neighbors: List[int] = field(default_factory=list)

    @property
    def uncertainty(self): return (self.rms_line if self.model == "line" else self.rms_circle) / math.sqrt(max(1, len(self.ids)))
    @property
    def straight(self): return self.model == "line"


def fit_fragment(S: torch.Tensor):
    """Droite ou arc ? L'arc n'est préféré que s'il réduit nettement l'écart (une droite est un arc de rayon infini :
    à ajustement égal, le modèle le plus simple l'emporte). Renvoie (modèle, paramètres, rms droite, rms cercle)."""
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
    """`forced` = (modèle, paramètres) quand le fragment vient d'une composante par consensus : le modèle est alors celui
    de la composante, et les deux écarts (droite, cercle) restent rapportés."""
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
    """Côté où l'étiquette vaut 1, lu sur les observations proches du fragment (−1 : intérieur du cercle / côté négatif)."""
    near = torch.cdist(ob.X.to(DT), f.points).min(1).values < band
    if int(near.sum()) < 6: return None
    s = _signed(f.model, f.params, ob.X[near]); y = ob.y[near].float(); pos, neg = y[s > 0], y[s <= 0]
    if len(pos) < 2 or len(neg) < 2 or abs(float(pos.mean() - neg.mean())) < 0.2: return None
    return 1 if float(pos.mean()) > float(neg.mean()) else -1


# ================================================================ regroupements (comparables séparément)

def _components_of(A: torch.Tensor) -> torch.Tensor:
    m = len(A); lab = torch.arange(m)
    for _ in range(m):
        new = torch.where(A, lab[None, :].expand(m, m), torch.full((m, m), m)).min(1).values
        if torch.equal(new, lab): break
        lab = new
    return torch.unique(lab, return_inverse=True)[1] if m else lab


def _same_curve(a, b, mode):
    """Deux fragments peuvent-ils être la MÊME courbe ? (centre compatible, orientation compatible, ou courbure compatible)."""
    if mode in ("center", "shared") and a[0] == b[0] == "circle" and math.hypot(a[1][0] - b[1][0], a[1][1] - b[1][1]) < 0.45 and abs(a[1][2] - b[1][2]) < 0.3: return True
    if mode in ("orientation", "shared") and a[0] == b[0] == "line":
        d = _wrap(a[1][0] - b[1][0]); return (abs(d) < 0.2 and abs(a[1][1] - b[1][1]) < 0.25) or (abs(abs(d) - math.pi) < 0.2 and abs(a[1][1] + b[1][1]) < 0.25)
    return False


def group_points(ob: ObservedBoundary, mode="shared", link=0.45, turn=0.6, seed=0) -> torch.Tensor:
    """Étiquette de groupe par milieu (−1 : non attribué). Mécanismes :
      proximity    liaison simple
      tangent      + continuité tangentielle (normales voisines alignées) : coupe aux coins
      curvature    fragments tangents fusionnés si un même modèle (droite ou cercle) les ajuste ensemble
      center       fragments courbes fusionnés si leurs cercles coïncident
      orientation  fragments droits fusionnés s'ils sont colinéaires
      shared       center + orientation + curvature
      consensus    composantes par consensus (cercles et droites), première composante qui contient le point"""
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
    """Fragments. En mode `consensus` (défaut), un fragment par composante de consensus (cercle ou droite) : les fragments
    peuvent SE RECOUVRIR — un même milieu reste compatible avec une droite et avec un arc tant que rien ne tranche.
    Dans les autres modes, les fragments sont les groupes disjoints de `group_points`."""
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


# ================================================================ graphe de compatibilités et rôles

@dataclass
class FragmentGraph:
    """Nœuds = fragments ; arêtes = (i, j, relation, résidu). Les hypothèses de groupement sont des sous-graphes."""
    fragments: List[BoundaryFragment]
    edges: List[Tuple[int, int, str, float]]

    def relations(self, i): return [(j if a == i else a, r, s) for a, j, r, s in self.edges if i in (a, j)]
    def groups(self) -> List[List[int]]:
        """Composantes connexes par relations fortes (concentrique, parallèle, sommet commun) ou par recouvrement."""
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
    """Sommet commun de deux fragments droits : intersection des droites, et pour chacun (est une DEMI-droite issue du
    sommet ?, direction du sommet vers le fragment, écart entre le sommet et l'extrémité la plus proche)."""
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
            if ov >= 0.5: E.append((i, j, "overlap", 1.0 - ov))                                  # mêmes milieux, deux lectures concurrentes : un seul groupe
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
    """Rôles compatibles d'un fragment, pondérés. Rien n'est tranché ici."""
    fragment: int
    roles: List[Tuple[str, float]]

    def top(self): return self.roles[0][0] if self.roles else "noise"
    def has(self, role): return any(r == role for r, _ in self.roles)


def assignments(g: FragmentGraph) -> List[CompatibleFragmentAssignments]:
    """Par fragment : droite ou arc selon l'ajustement (les deux restent ouverts si l'écart est faible), bord intérieur /
    extérieur d'anneau s'il a un partenaire concentrique, bord de bande s'il a un parallèle, rayon de secteur s'il
    partage un sommet, masque possible s'il coupe un autre fragment à son extrémité, bruit s'il est court ou mal ajusté."""
    out = []
    for f in g.fragments:
        w = {}; rl, rc = f.rms_line, f.rms_circle; el, ec = math.exp(-(rl / (2 * SIGMA)) ** 2), (math.exp(-(rc / (2 * SIGMA)) ** 2) * 0.8 if math.isfinite(rc) else 0.0)
        w["line"] = el; w["circle_arc"] = ec if not f.straight or ec > 0.5 * el else 0.0
        for j, r, s in g.relations(f.index):
            o = g.fragments[j]
            if r == "concentric": w["ring_inner" if f.params[2] < o.params[2] else "ring_outer"] = max(w.get("ring_inner" if f.params[2] < o.params[2] else "ring_outer", 0.0), ec * math.exp(-s))
            if r == "parallel": w["band_edge"] = max(w.get("band_edge", 0.0), el * math.exp(-s))
            if r == "vertex": w["sector_ray"] = max(w.get("sector_ray", 0.0), el * math.exp(-s))
            if r == "adjacent" and o.model != f.model: w["occluder"] = max(w.get("occluder", 0.0), 0.5 * f.confidence)       # un coin entre deux courbes de nature différente : l'une peut masquer l'autre
        w["noise"] = 1.0 - f.confidence; tot = sum(w.values()) or 1.0
        out.append(CompatibleFragmentAssignments(f.index, sorted(((k, v / tot) for k, v in w.items() if v > 0), key=lambda x: -x[1])))
    return out


# ================================================================ propriété du masquage

@dataclass
class OcclusionOwnership:
    """Pourquoi chaque point de la frontière ATTENDUE d'une hypothèse est, ou n'est pas, observé."""
    labels: List[str]               # par point rendu : observed | occluded | domain | noise | absent
    owner: Optional[int]            # indice de l'hypothèse qui masque, si une seule explique la disparition
    fractions: Dict[str, float]

    @property
    def absent(self): return self.fractions["absent"]


def ownership(p: BNode, ob: ObservedBoundary, others: Sequence[BNode] = (), tol=0.3, gap=0.5, n=96) -> OcclusionOwnership:
    """Frontière attendue de `p`, point par point :
      observed  un milieu observé à moins de `tol`
      domain    hors de la boîte des observations : inobservable
      occluded  une AUTRE hypothèse sépare la portion manquante de la portion observée (elle vaut une constante sur la
                portion manquante et l'autre valeur sur la portion observée) : la frontière est cachée, pas absente
      noise     trou court (moins de `gap`) entre deux portions observées
      absent    manque dans une zone observable sans explication — seule preuve NÉGATIVE légitime."""
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
    if int(seen.sum()):                                                           # trous courts : bruit d'échantillonnage, pas une absence
        d = torch.cdist(P, P[seen]).min(1).values; lab = ["noise" if (l == "absent" and float(x) < gap) else l for l, x in zip(lab, d)]
    return OcclusionOwnership(lab, owner, {k: lab.count(k) / len(lab) for k in OWNERSHIP})


# ================================================================ hypothèses de type

@dataclass
class TypeHypothesis:
    family: str
    type: str
    program: BNode
    fragments: Tuple[int, ...]
    signals: Dict[str, float] = field(default_factory=dict)        # signaux SÉPARÉS (voir `signals_of`)
    score: float = 0.0
    confidence: float = 0.0                                          # part du score dans son groupe (softmax)


def _circle_of(S, f: BoundaryFragment):
    c, r, rms = circle_geometric(S, f.params[:2], f.params[2]) if not f.straight else (None, None, None)
    return c, r, rms


def signals_of(p: BNode, ids: torch.Tensor, ob: ObservedBoundary, coherence=0.0, others: Sequence[BNode] = (), negative="ownership") -> Dict[str, float]:
    """Signaux explicites d'un type candidat, rapportés séparément :
      geometric   écart quadratique moyen des milieux du groupe à la frontière du programme
      coherence   résidu de la relation partagée (distance des centres, écart d'angle, écart au sommet)
      support     nombre de milieux observés à moins de 0,15 de la frontière du programme
      explained   contraste z sur les tronçons vus de la frontière
      seen        part de la frontière attendue qui est observée
      absent      part de la frontière attendue qui manque SANS explication (preuve négative) — selon `negative` :
                  "ownership" (zones cachées, hors domaine et trous courts exclus), "naive" (tout ce qui manque), "none"
      complexity  nombre de paramètres
      stability   désaccord fonctionnel entre les ajustements sur deux moitiés des milieux (None si non calculable)"""
    S = ob.points[ids]; B = render(p, n=160, box=(-9.0, 9.0, -9.0, 9.0), reach=6.0); geo = float(torch.cdist(S, B).min(1).values.pow(2).mean().sqrt()) if len(B) and len(S) else float("inf"); z, vf = visible_score(p, ob.pairs)
    own = ownership(p, ob, others); absent = 0.0 if negative == "none" else own.absent if negative == "ownership" else 1.0 - own.fractions["observed"] - own.fractions["domain"]
    from core.deep_partial import refit_visible
    a, b = (refit_visible(p, S[0::2]), refit_visible(p, S[1::2])) if len(S) >= 10 else (None, None); stab = float((predict(a, ob.X.float()) != predict(b, ob.X.float())).float().mean()) if a is not None and b is not None else None
    sup = float((torch.cdist(ob.points, B).min(1).values < 0.15).sum()) if len(B) and len(ob.points) else 0.0
    return {"geometric": geo, "coherence": coherence, "support": sup, "explained": z, "seen": own.fractions["observed"], "absent": absent, "occluded": own.fractions["occluded"], "complexity": float(p.n_params()), "stability": stab}


def fuse(s: Dict[str, float], cost=2.0) -> float:
    """Fusion explicite, de type longueur de description : milieux de frontière expliqués de près (à moins de 0,15),
    réduits par la part de frontière attendue qui manque sans explication, moins un coût par paramètre. Un anneau
    (4 paramètres) l'emporte sur un disque (3) dès qu'il explique quelques milieux de plus ; deux disques coûtent 6."""
    return s["support"] * (1.0 - s["absent"]) - cost * s["complexity"]


def _disc_types(f: BoundaryFragment, ob):
    """Disque ou anti-disque : la polarité lue sur les observations décide de l'ordre ; les deux restent proposés."""
    return ("disc", "antidisc") if f.polarity in (-1, None) else ("antidisc", "disc")


def type_hypotheses(g: FragmentGraph, ob: ObservedBoundary, negative="ownership", hierarchical=True) -> List[TypeHypothesis]:
    """fragment geometry → primitive family → specific primitive type.

    Niveau 1 (géométrie) : droit / courbe. Niveau 2 (famille) : seul, ou lié par une relation (parallèle, concentrique,
    sommet commun). Niveau 3 (type) : polarité pour disque / anti-disque. Les paramètres sont ajustés sur les milieux du
    groupe, avec la contrainte partagée de la famille. Avec `hierarchical=False`, la famille n'est pas restreinte par
    la géométrie : chaque fragment est proposé à la fois comme droite et comme cercle (classification plate par score)."""
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
    """Top-k types PAR GROUPE de milieux. Les groupes ne sont pas fixés d'avance : la meilleure hypothèse prend les
    milieux qu'elle explique ; une hypothèse suivante n'ouvre un NOUVEAU groupe que si au moins la moitié de ses
    milieux (et au moins `min_new`) restent inexpliqués ; sinon elle est une lecture CONCURRENTE du groupe qu'elle
    recouvre le plus, gardée si le groupe a moins de k types. La confiance est la part softmax du score dans le groupe.
    Aucun top-1 n'est forcé : avec k > 1, plusieurs types compatibles restent ouverts pour la composition."""
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


# ================================================================ boucle attribution ↔ type (bornée, instrumentée)

@dataclass
class Inference:
    fragments: List[BoundaryFragment]
    graph: FragmentGraph
    hypotheses: List[TypeHypothesis]        # toutes, classées
    selected: List[TypeHypothesis]          # top-k par groupe
    log: List[dict]
    stats: Dict[str, float]


def reassign(ob: ObservedBoundary, hyps: Sequence[TypeHypothesis], tol=0.25) -> torch.Tensor:
    """Ré-attribution des milieux à la lumière des types : chaque milieu va à l'hypothèse (score positif) dont la
    frontière ajustée est la plus proche, s'il est à moins de `tol`. Le type corrige l'attribution."""
    M = ob.points; lab = torch.full((len(M),), -1, dtype=torch.long); keep = [h for h in hyps if h.score > 0]
    if not keep or not len(M): return lab
    D = torch.stack([torch.cdist(M, render(h.program, n=160, box=(-9.0, 9.0, -9.0, 9.0), reach=6.0)).min(1).values for h in keep], 1); d, i = D.min(1); lab[d < tol] = i[d < tol]; return lab


def infer_types(ob: ObservedBoundary, k=1, mode="consensus", negative="ownership", hierarchical=True, iters=1, seed=0) -> Inference:
    """fragment assignment → type hypothesis → parameter fit → re-score fragment assignment (au plus `iters` reprises).

    À chaque reprise, les milieux sont ré-attribués aux meilleures hypothèses par groupe, les fragments sont refaits
    sur cette attribution et les types ré-inférés ; la reprise n'est gardée que si le score total des hypothèses
    retenues augmente (sinon l'état précédent est conservé)."""
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
    """Attribution APRÈS inférence de type : chaque milieu va à la meilleure hypothèse par groupe qui l'explique."""
    inf = infer_types(ob, 1, seed=seed); return reassign(ob, inf.selected)


def flat_types(ob: ObservedBoundary, k=1, seed=0) -> List[TypeHypothesis]:
    """Classification PLATE, sans fragments : chaque type est essayé sur toute la scène (inférence à type connu répétée
    pour les six types), top-k par type. Sert de comparaison à la hiérarchie."""
    out = []
    for t, sk in SKELETONS.items():
        for h in infer(sk, ob, k, seed=seed, regions=False): out.append(TypeHypothesis({v: f for f, vs in FAMILY_TYPES.items() for v in vs}[t], t, h.program, (), {"explained": h.score, "complexity": float(h.program.n_params())}, h.score - 0.5 * h.program.n_params()))
    return sorted(out, key=lambda h: -h.score)


def literals(hyps: Sequence[TypeHypothesis]) -> List[BNode]:
    """Programmes à donner à la composition (dédoublonnés) ; un demi-plan est donné avec son complément."""
    out = []
    for h in hyps:
        out.append(h.program)
        if atom_name(h.program) == "dot.le": out.append(BNode("pred", "dot", "le", (_wrap(h.program.params[0] + math.pi), -h.program.params[1])))
    seen = set(); return [p for p in out if not (repr(p) in seen or seen.add(repr(p)))]

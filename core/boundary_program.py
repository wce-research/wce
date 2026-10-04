"""INDUCTION DE PROGRAMMES DE FRONTIÈRE (Boundary Program Induction).

OBSERVATIONS → PRIMITIVES → PROGRAMMES CANDIDATS → COMPOSITION → TEST → SIMPLIFICATION → ANTI-UNIFICATION →
FAMILLE PARAMÉTRÉE → INVARIANTS → COMPÉTENCE → CERTIFICAT GÉNÉRATIF → RÉUTILISATION.

Algèbre de primitives (AUCUNE famille de haut niveau nommée : pas de cercle, ellipse, secteur, boîte…) :
  caractéristiques scalaires : dot(α) = cos α·x0 + sin α·x1, norm = ‖x‖, angle = atan2(x1, x0)
  prédicats atomiques        : f ≤ b, f ≥ b, f ∈ [lo, hi] (intervalle ; pour angle : |angle − θ0| ≤ w)
  transformations            : translate(c), rotate(φ), scale(s), reflect
  combinateurs               : AND, OR, NOT, XOR, composition par transformation de coordonnées
Structure et paramètres sont SÉPARÉS : la recherche énumère des squelettes, l'ajustement fixe les paramètres
(gradient local sur une version adoucie, avec initialisations heuristiques ; formes closes quand disponibles).
Critère pratique « type MDL » : erreur + λ·L(programme) — pas une théorie complète du MDL.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

PRIMITIVES = ("dot", "norm", "angle", "interval", "translate", "rotate", "and", "or", "not", "xor")


# ================================================================ AST

_OWN_PRED = {("pred", "dot", "le"): 2, ("pred", "dot", "ge"): 2, ("pred", "dot", "in"): 3, ("pred", "norm", "le"): 1,
             ("pred", "norm", "ge"): 1, ("pred", "norm", "in"): 2, ("pred", "angle", "in"): 2}
_OWN_KIND = {"translate": 2, "rotate": 1, "scale": 1}

@dataclass(frozen=True)
class BNode:
    """Nœud de programme. kind ∈ {pred, translate, rotate, scale, reflect, and, or, xor, not}.
    pred : feat ∈ {dot, norm, angle}, op ∈ {le, ge, in}. params : tuple de flottants (None = squelette)."""
    kind: str
    feat: str = ""
    op: str = ""
    params: Optional[Tuple[float, ...]] = None
    children: Tuple["BNode", ...] = ()

    # ---- structure ---------------------------------------------------------------------------
    # n_params / own_count : entiers ne dépendant que de champs IMMUABLES (nœud gelé) → mis en cache sur l'instance.
    # Pure mise en cache (mêmes valeurs) : le profilage AWS y attribuait ~6–9 % du temps (appels répétés à chaque
    # évaluation du programme pendant l'ajustement). Le cache n'est pas un champ : égalité et hachage inchangés.
    def own_count(self):
        v = self.__dict__.get("_own")
        if v is None:
            v = _OWN_PRED.get((self.kind, self.feat, self.op), 0) + _OWN_KIND.get(self.kind, 0); object.__setattr__(self, "_own", v)
        return v

    def n_params(self):
        v = self.__dict__.get("_np")
        if v is None:
            v = self.own_count() + sum(c.n_params() for c in self.children); object.__setattr__(self, "_np", v)
        return v

    def depth(self): return 1 + max((c.depth() for c in self.children), default=0)
    def size(self): return 1 + sum(c.size() for c in self.children)

    def skeleton(self): return BNode(self.kind, self.feat, self.op, None, tuple(c.skeleton() for c in self.children))

    def with_params(self, vec: Sequence[float]):
        it = iter(list(vec))
        def rebuild(n):
            k = n.own_count()
            ps = tuple(float(next(it)) for _ in range(k)) if k else None
            return BNode(n.kind, n.feat, n.op, ps, tuple(rebuild(c) for c in n.children))
        return rebuild(self)

    def param_vector(self):
        out = list(self.params or ())
        for c in self.children: out += c.param_vector()
        return out

    def primitives_used(self):
        s = set()
        if self.kind == "pred":
            s.add(self.feat)
            if self.op == "in": s.add("interval")
        else: s.add(self.kind)
        for c in self.children: s |= c.primitives_used()
        return s

    # ---- sérialisation / affichage -------------------------------------------------------------
    def to_dict(self):
        d = {"kind": self.kind}
        if self.feat: d["feat"] = self.feat
        if self.op: d["op"] = self.op
        if self.params is not None: d["params"] = [round(p, 4) for p in self.params]
        if self.children: d["children"] = [c.to_dict() for c in self.children]
        return d

    @staticmethod
    def from_dict(d):
        return BNode(d["kind"], d.get("feat", ""), d.get("op", ""), tuple(d["params"]) if "params" in d else None,
                     tuple(BNode.from_dict(c) for c in d.get("children", [])))

    def __str__(self): return render(self)


def render(n: BNode, digits=2):
    f = lambda v: f"{v:.{digits}f}"
    p = n.params or ()
    if n.kind == "pred":
        if n.feat == "dot":
            lab = f"dot(α={f(p[0])})" if p else "dot(α)"
            if n.op == "in": return f"{lab}∈[{f(p[1])},{f(p[2])}]" if p else f"{lab}∈[lo,hi]"
            return f"{lab}{'≤' if n.op == 'le' else '≥'}{f(p[1]) if p else 'b'}"
        if n.feat == "norm":
            if n.op == "in": return f"‖x‖∈[{f(p[0])},{f(p[1])}]" if p else "‖x‖∈[lo,hi]"
            return f"‖x‖{'≤' if n.op == 'le' else '≥'}{f(p[0]) if p else 'r'}"
        if n.feat == "angle": return f"|angle−{f(p[0])}|≤{f(p[1])}" if p else "|angle−θ|≤w"
    if n.kind == "translate": return f"translate({f(p[0])},{f(p[1])})[{render(n.children[0], digits)}]" if p else f"translate(c)[{render(n.children[0], digits)}]"
    if n.kind == "rotate": return f"rotate({f(p[0])})[{render(n.children[0], digits)}]" if p else f"rotate(φ)[{render(n.children[0], digits)}]"
    if n.kind == "scale": return f"scale({f(p[0])})[{render(n.children[0], digits)}]" if p else f"scale(s)[{render(n.children[0], digits)}]"
    if n.kind == "reflect": return f"reflect[{render(n.children[0], digits)}]"
    if n.kind == "not": return f"NOT({render(n.children[0], digits)})"
    return f"{n.kind.upper()}(" + ", ".join(render(c, digits) for c in n.children) + ")"


# ================================================================ évaluation (dure et adoucie)

def _wrap(a): return torch.atan2(torch.sin(a), torch.cos(a))


def evaluate(n: BNode, X, theta=None, k=None):
    """Probabilité (k fini : version adoucie différentiable) ou 0/1 (k None) d'appartenance.
    theta : vecteur de paramètres (tenseur) qui remplace les paramètres du nœud (ajustement)."""
    cursor = [0]
    vec = theta if theta is not None else torch.tensor(n.param_vector(), dtype=X.dtype)
    def take(m):
        v = vec[cursor[0]: cursor[0] + m]; cursor[0] += m; return v
    sig = (lambda z: torch.sigmoid(k * z)) if k is not None else (lambda z: (z >= 0).to(X.dtype))
    def ev(node, X):
        own = take(node.own_count())
        if node.kind == "pred":
            if node.feat == "dot":
                f = torch.cos(own[0]) * X[:, 0] + torch.sin(own[0]) * X[:, 1]
                if node.op == "le": return sig(own[1] - f)
                if node.op == "ge": return sig(f - own[1])
                return sig(f - own[1]) * sig(own[2] - f)
            if node.feat == "norm":
                f = X.norm(dim=1)
                if node.op == "le": return sig(own[0] - f)
                if node.op == "ge": return sig(f - own[0])
                return sig(f - own[0]) * sig(own[1] - f)
            if node.feat == "angle":
                d = _wrap(torch.atan2(X[:, 1], X[:, 0]) - own[0]); return sig(own[1].abs() - d.abs())
        if node.kind == "translate": return ev(node.children[0], X - own[:2])
        if node.kind == "rotate":
            c, s = torch.cos(own[0]), torch.sin(own[0]); return ev(node.children[0], torch.stack([c * X[:, 0] + s * X[:, 1], -s * X[:, 0] + c * X[:, 1]], 1))
        if node.kind == "scale": return ev(node.children[0], X / own[0].abs().clamp_min(1e-3))
        if node.kind == "reflect": return ev(node.children[0], X * torch.tensor([1.0, -1.0], dtype=X.dtype))
        if node.kind == "not": return 1 - ev(node.children[0], X)
        a, b = ev(node.children[0], X), ev(node.children[1], X)
        if node.kind == "and": return a * b
        if node.kind == "or": return a + b - a * b
        if node.kind == "xor": return a + b - 2 * a * b
        raise KeyError(node.kind)
    return ev(n, X)


def predict(n: BNode, X): return (evaluate(n, torch.nan_to_num(X)) >= 0.5).long()


# ================================================================ complexité (L) et objectif type MDL

def complexity(n: BNode):
    """L(programme) = nb de nœuds + 0,5·paramètres libres + 0,5·profondeur + 0,5·branches + coût des transformations."""
    branches = sum(1 for _ in _walk(n) if _.kind in ("and", "or", "xor"))
    transf = sum(1 for _ in _walk(n) if _.kind in ("translate", "rotate", "scale", "reflect"))
    return n.size() + 0.5 * n.n_params() + 0.5 * n.depth() + 0.5 * branches + 0.5 * transf


def _walk(n):
    yield n
    for c in n.children: yield from _walk(c)


def mdl_score(error, n: BNode, lam=0.004): return error + lam * complexity(n)


# ================================================================ canonisation, hachage, équivalence

def canonical(n: BNode) -> BNode:
    """Forme canonique (quand possible) : enfants des combinateurs commutatifs triés ; NOT(NOT a)=a ;
    transformations triviales supprimées ; angles ramenés dans [−π, π) ; dot ≥ b → dot(α+π) ≤ −b."""
    ch = tuple(canonical(c) for c in n.children); p = n.params
    if n.kind == "not" and ch[0].kind == "not": return ch[0].children[0]
    if n.kind in ("and", "or", "xor"): ch = tuple(sorted(ch, key=lambda c: json.dumps(c.skeleton().to_dict(), sort_keys=True) + str(c.params)))
    if p is not None:
        if n.kind == "translate" and abs(p[0]) < 1e-6 and abs(p[1]) < 1e-6: return ch[0]
        if n.kind == "rotate" and abs(_wrapf(p[0])) < 1e-6: return ch[0]
        if n.kind == "scale" and abs(abs(p[0]) - 1) < 1e-6: return ch[0]
        if n.kind == "pred" and n.feat == "dot" and n.op == "ge": return BNode("pred", "dot", "le", (_wrapf(p[0] + math.pi), -p[1]), ())
        if n.kind == "pred" and n.feat == "dot": p = (_wrapf(p[0]),) + tuple(p[1:])
        if n.kind == "pred" and n.feat == "angle": p = (_wrapf(p[0]), abs(p[1]))
        if n.kind == "rotate": p = (_wrapf(p[0]),)
    return BNode(n.kind, n.feat, n.op, p, ch)


def _wrapf(a): return math.atan2(math.sin(a), math.cos(a))


def structure_hash(n: BNode): return json.dumps(canonical(n).skeleton().to_dict(), sort_keys=True)
def program_hash(n: BNode): return json.dumps(canonical(n).to_dict(), sort_keys=True)


class ProgramEquivalence:
    """Équivalence EMPIRIQUE : structure canonique identique, OU accord fonctionnel ≥ seuil sur des sondes
    (génératives / région). N'affirme jamais une équivalence mathématique générale."""

    def __init__(self, probes, threshold=0.99): self.probes, self.threshold = probes, threshold

    def compare(self, a: BNode, b: BNode):
        if program_hash(a) == program_hash(b): return "identical_canonical"
        agree = float((predict(a, self.probes) == predict(b, self.probes)).float().mean())
        if agree >= self.threshold: return "functionally_equivalent_on_probes" if structure_hash(a) != structure_hash(b) else "same_structure_close_params"
        return "different"


# ================================================================ ajustement des paramètres (structure fixée)

def _inits(skel: BNode, X, y, n_init, g):
    """Initialisations heuristiques (centres = moyennes / points positifs, angles = régression logistique) + aléatoires."""
    pos = X[y == 1] if int(y.sum()) else X
    out = []
    for i in range(n_init):
        vec = []
        def walk(n):
            if n.kind == "translate":
                if i == 0: c = pos.mean(0)
                else: c = pos[int(torch.randint(0, len(pos), (1,), generator=g))] + 0.3 * torch.randn(2, generator=g)
                vec.extend(c.tolist())
            elif n.kind == "rotate": vec.append(float(torch.rand(1, generator=g)) * 6.283 - 3.1416)
            elif n.kind == "scale": vec.append(1.0)
            elif n.kind == "pred":
                if n.feat == "dot":
                    a = float(torch.rand(1, generator=g)) * 6.283 - 3.1416
                    f = torch.cos(torch.tensor(a)) * X[:, 0] + torch.sin(torch.tensor(a)) * X[:, 1]
                    if n.op in ("le", "ge"): vec.extend([a, float(f.median())])
                    else: q = torch.quantile(f, torch.tensor([0.3, 0.7])); vec.extend([a, float(q[0]), float(q[1])])
                elif n.feat == "norm":
                    r = float(X.norm(dim=1).median()) * (0.5 + float(torch.rand(1, generator=g)))
                    vec.extend([r] if n.op in ("le", "ge") else [0.5 * r, 1.2 * r])
                else: vec.extend([float(torch.rand(1, generator=g)) * 6.283 - 3.1416, 0.8])
            for c in n.children: walk(c)
        walk(skel); out.append(torch.tensor(vec))
    return out


def fit(skel: BNode, X, y, steps=160, n_init=3, lr=0.06, seed=0, counters=None):
    """Ajuste les paramètres d'un squelette : gradient sur la version adoucie (raideur croissante), plusieurs
    initialisations ; renvoie le programme ajusté et son exactitude d'ajustement."""
    g = torch.Generator().manual_seed(seed); X = torch.nan_to_num(X); yf = y.float(); best = None
    if skel.n_params() == 0:
        p = skel.with_params([]); return p, float((predict(p, X) == y).float().mean())
    for th0 in _inits(skel, X, y, n_init, g):
        th = th0.clone().requires_grad_(True); opt = torch.optim.Adam([th], lr=lr)
        for t in range(steps):
            k = 2.0 + 18.0 * t / steps
            opt.zero_grad(); p = evaluate(skel, X, th, k).clamp(1e-5, 1 - 1e-5)
            torch.nn.functional.binary_cross_entropy(p, yf).backward(); opt.step()
        prog = skel.with_params(th.detach().tolist()); acc = float((predict(prog, X) == y).float().mean())
        if counters is not None: counters["fits"] += 1
        if best is None or acc > best[1]: best = (prog, acc)
    return best


# ================================================================ recherche de structure

def atoms(allowed=PRIMITIVES):
    """Prédicats atomiques (niveau 1). Les caractéristiques norm / angle sont relatives à un centre appris
    (translate) ; NOT est exprimé par l'opérateur complémentaire quand il existe."""
    A = []
    if "dot" in allowed:
        A.append(BNode("pred", "dot", "le"))
        if "interval" in allowed: A += [BNode("pred", "dot", "in"), BNode("not", children=(BNode("pred", "dot", "in"),))] if "not" in allowed else [BNode("pred", "dot", "in")]
    if "translate" in allowed:
        if "norm" in allowed:
            A += [BNode("translate", children=(BNode("pred", "norm", "le"),)), BNode("translate", children=(BNode("pred", "norm", "ge"),))]
            if "interval" in allowed: A.append(BNode("translate", children=(BNode("pred", "norm", "in"),)))
        if "angle" in allowed and "interval" in allowed:                  # l'angle s'utilise comme intervalle centré
            A.append(BNode("translate", children=(BNode("pred", "angle", "in"),)))
            if "not" in allowed: A.append(BNode("translate", children=(BNode("not", children=(BNode("pred", "angle", "in"),)),)))
    return A


def _origin_atoms(allowed):
    O = []
    if "norm" in allowed: O += [BNode("pred", "norm", "le"), BNode("pred", "norm", "ge")]
    if "angle" in allowed and "interval" in allowed: O.append(BNode("pred", "angle", "in"))
    return O


@dataclass
class SearchResult:
    best: Optional[BNode]
    val_acc: float
    candidates: List[Tuple[BNode, float, float]]          # (programme, exactitude validation, score MDL)
    stats: Dict[str, float]
    adequate: bool


class BoundaryProgramSearch:
    """Recherche best-first / faisceau bornée, par niveaux : (1) primitives seules, (2) transformations de
    coordonnées partagées, (3) conjonctions / disjonctions / XOR de deux atomes, (4) compositions plus profondes
    (faisceau). Cache par structure canonique ; élagage par borne de complexité."""

    def __init__(self, allowed=PRIMITIVES, max_depth=3, beam=4, lam=0.004, adequacy=0.93, steps=120, n_init=2, seed=0, macros=None):
        self.allowed, self.max_depth, self.beam, self.lam, self.adequacy = set(allowed), max_depth, beam, lam, adequacy
        self.steps, self.n_init, self.seed, self.macros = steps, n_init, seed, macros or []

    def combinators(self): return [c for c in ("and", "or", "xor") if c in self.allowed]

    def run(self, X, y, Xv, yv, time_budget=None):
        t0 = time.time(); stats = Counter(); cache = {}; scored = []
        def consider(skel, level):
            h = structure_hash(skel)
            if h in cache: stats["cache_hits"] += 1; return cache[h]
            if complexity(skel) > 40: stats["pruned"] += 1; return None
            prog, acc_t = fit(skel, X, y, self.steps, self.n_init, seed=self.seed + len(cache), counters=stats)
            acc_v = float((predict(prog, Xv) == yv).float().mean()); stats["evaluated"] += 1
            r = (prog, acc_v, mdl_score(1 - acc_v, prog, self.lam), level); cache[h] = r; scored.append(r); return r
        # 0. macros (familles induites) d'abord : si une suffit, arrêt précoce
        for m in self.macros:
            r = consider(m, 0)
            if r and r[1] >= self.adequacy + 0.02: return self._result(scored, stats, t0)
        # 1. atomes
        A = atoms(self.allowed)
        for a in A: consider(a, 1)
        if self._good_enough(scored): return self._result(scored, stats, t0)
        if self.max_depth >= 2:
            # 2. transformation de coordonnées PARTAGÉE (centre commun) sur combinaisons d'atomes à l'origine
            if "translate" in self.allowed:
                O = _origin_atoms(self.allowed)
                for c in self.combinators():
                    for a, b in itertools.combinations_with_replacement(O, 2):
                        if a == b and a.op != "in" and c != "xor": continue
                        consider(BNode("translate", children=(BNode(c, children=(a, b)),)), 2)
                        if self._excellent(scored): return self._result(scored, stats, t0)
            # 3. combinaisons de deux atomes (faisceau : atomes les mieux classés + demi-espace toujours présent)
            top = [r[0].skeleton() for r in sorted([r for r in scored if r[3] == 1], key=lambda r: r[2])[: self.beam]]
            pool = list(dict.fromkeys(top + [a for a in A if a.kind == "pred" and a.op == "le"]))
            pairs = sorted(((c, a, b) for c in self.combinators() for a, b in itertools.combinations_with_replacement(pool, 2)),
                           key=lambda t: complexity(BNode(t[0], children=(t[1], t[2])).with_params([0.0] * BNode(t[0], children=(t[1], t[2])).n_params())))
            for c, a, b in pairs:
                consider(BNode(c, children=(a, b)), 2)
                if self._excellent(scored) or (time_budget and time.time() - t0 > time_budget): break
        if self.max_depth >= 3 and not self._good_enough(scored):
            # 4. profondeur supérieure : faisceau des meilleurs programmes × atomes
            for d in range(3, self.max_depth + 1):
                frontier = [r[0].skeleton() for r in sorted([r for r in scored if r[3] == d - 1], key=lambda r: r[2])[: self.beam]]
                for f in frontier:
                    for a in A:
                        for c in self.combinators(): consider(BNode(c, children=(f, a)), d)
                if self._good_enough(scored) or (time_budget and time.time() - t0 > time_budget): break
        return self._result(scored, stats, t0)

    def _good_enough(self, scored): return any(r[1] >= self.adequacy + 0.03 for r in scored)
    def _excellent(self, scored): return any(r[1] >= 0.985 for r in scored)

    def _result(self, scored, stats, t0):
        scored.sort(key=lambda r: r[2]); stats["cache_hits"] += 0; stats["pruned"] += 0
        best = scored[0] if scored else None
        stats["seconds"] = round(time.time() - t0, 3); stats["prune_rate"] = round(stats["pruned"] / max(1, stats["pruned"] + stats["evaluated"]), 3)
        adequate = best is not None and best[1] >= self.adequacy
        return SearchResult(best[0] if best else None, best[1] if best else 0.0, [(r[0], r[1], r[2]) for r in scored[:12]], dict(stats), adequate)


# ================================================================ postérieure, ambiguïté, expériences actives

class ProgramPosterior:
    """P(programme | évidence) sur un ensemble FINI de candidats ajustés : log-vraisemblance avec bruit d'étiquette
    ε + a priori exp(−L). Conserve plusieurs hypothèses ; UNKNOWN là où elles divergent."""

    def __init__(self, programs: Sequence[BNode], eps=0.05):
        uniq = {}
        for p in programs: uniq.setdefault(program_hash(p), p)
        self.programs = list(uniq.values()); self.eps = eps
        self.logw = torch.tensor([-complexity(p) * 0.5 for p in self.programs], dtype=torch.float64)

    def update(self, X, y):
        P = torch.stack([predict(p, X) for p in self.programs])                 # [K, N]
        ok = (P == y).double(); self.logw = self.logw + (ok * math.log(1 - self.eps) + (1 - ok) * math.log(self.eps)).sum(1); return self

    @property
    def weights(self): return torch.softmax(self.logw, 0)

    def map(self): i = int(self.weights.argmax()); return self.programs[i], float(self.weights[i])

    def predictive(self, X):
        P = torch.stack([predict(p, X) for p in self.programs]).double(); return (self.weights[:, None] * P).sum(0)

    def predict_or_unknown(self, X, margin=0.9):
        p1 = self.predictive(X); out = torch.full((len(X),), -1, dtype=torch.long)
        out[p1 >= margin] = 1; out[p1 <= 1 - margin] = 0; return out                 # −1 = UNKNOWN (plusieurs structures compatibles)

    def ambiguous(self, region, threshold=0.05):
        """Ambiguïté : la masse postérieure se répartit sur des programmes qui DIVERGENT sur la région."""
        p1 = self.predictive(region); return float(((p1 > 0.1) & (p1 < 0.9)).double().mean()) > threshold

    def info_gain(self, Xc):
        """IG(expérience ; identité du programme) pour chaque point candidat (étiquette bruitée ε)."""
        p1 = self.predictive(Xc).clamp(1e-9, 1 - 1e-9)
        py1 = p1 * (1 - self.eps) + (1 - p1) * self.eps
        H = lambda q: -(q * torch.log(q) + (1 - q) * torch.log(1 - q))
        return H(py1) - H(torch.tensor(self.eps, dtype=torch.float64))

    def disagreement(self, Xc):
        p1 = self.predictive(Xc); return 1 - (2 * p1 - 1).abs()


# ================================================================ anti-unification et familles

@dataclass
class ParameterizedBoundaryMacro:
    """Famille paramétrée : squelette canonique commun + plages des paramètres observées (variables)."""
    skeleton: BNode
    param_lo: Tuple[float, ...]
    param_hi: Tuple[float, ...]
    support: int

    def variables(self): return [f"θ{i}" for i in range(len(self.param_lo))]


def anti_unify_programs(programs: Sequence[BNode]):
    """Anti-unification d'AST : si toutes les formes canoniques partagent la même STRUCTURE, les paramètres
    différents deviennent des variables (avec plages) ; sinon, plus petit généralisant structurel par position
    (nœuds différents → variable de structure « ? »). Renvoie (macro | None, gabarit)."""
    can = [canonical(p) for p in programs]; skels = [c.skeleton() for c in can]
    if all(s == skels[0] for s in skels):
        vecs = torch.tensor([c.param_vector() for c in can])
        return ParameterizedBoundaryMacro(skels[0], tuple(vecs.min(0).values.tolist()), tuple(vecs.max(0).values.tolist()), len(programs)), skels[0]
    def lgg(a, b):
        if a.kind != b.kind or a.feat != b.feat or a.op != b.op or len(a.children) != len(b.children): return BNode("?")
        return BNode(a.kind, a.feat, a.op, None, tuple(lgg(x, z) for x, z in zip(a.children, b.children)))
    t = skels[0]
    for s in skels[1:]: t = lgg(t, s)
    return None, t


@dataclass
class BoundaryFamilyHypothesis:
    """HYPOTHESIS → SUPPORTED → STABLE | REFUTED ; garde supports, contre-exemples, score, complexité, provenance."""
    macro: ParameterizedBoundaryMacro
    status: str = "HYPOTHESIS"
    support: List[str] = field(default_factory=list)
    counterexamples: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    provenance: str = ""

    def complexity(self): return complexity(self.macro.skeleton.with_params([0.0] * self.macro.skeleton.n_params()))

    def validate(self, X, y, Xv, yv, name="", adequacy=0.93):
        prog, _ = fit(self.macro.skeleton, X, y, steps=160, n_init=3)
        acc = float((predict(prog, Xv) == yv).float().mean()); self.scores.append(acc)
        (self.support if acc >= adequacy else self.counterexamples).append(name)
        if len(self.counterexamples) >= max(2, len(self.support)): self.status = "REFUTED"
        elif len(self.support) >= 5 and len(self.counterexamples) == 0: self.status = "STABLE"
        elif len(self.support) >= 2: self.status = "SUPPORTED"
        return acc


def induce_families(programs_by_task: Sequence[BNode], min_support=2):
    """Regroupe des programmes réussis par structure canonique et anti-unifie chaque groupe en macro."""
    groups = defaultdict(list)
    for p in programs_by_task: groups[structure_hash(p)].append(p)
    out = []
    for h, ps in groups.items():
        if len(ps) >= min_support:
            m, _ = anti_unify_programs(ps); out.append(BoundaryFamilyHypothesis(m, provenance=f"{len(ps)} programmes réussis"))
    return out


# ================================================================ explication et graphe de preuves

def explain(n: BNode):
    """Trace structurée (sans prose décorative)."""
    return {"rule": render(n), "positive_when": _conds(n), "complexity": round(complexity(n), 2), "parameters": n.n_params(), "depth": n.depth()}


def _conds(n):
    if n.kind in ("and", "or", "xor"): return {n.kind.upper(): [_conds(c) for c in n.children]}
    if n.kind == "not": return {"NOT": _conds(n.children[0])}
    if n.kind in ("translate", "rotate", "scale", "reflect"): return {"after": render(BNode(n.kind, params=n.params, children=(BNode("pred", "norm", "le", (0.0,)),))).split("[")[0], "then": _conds(n.children[0])}
    return render(n)


def evidence_graph(n: BNode, X, y, sources=None):
    """Relie chaque sous-programme aux observations qui le SOUTIENNENT (même décision locale) et aux contre-exemples."""
    out = []
    def walk(node, path):
        P = predict(node, X); out.append({"node": path, "rule": render(node), "support_pos": int(((P == 1) & (y == 1)).sum()),
                                          "support_neg": int(((P == 0) & (y == 0)).sum()), "counterexamples": int((P != y).sum()),
                                          "sources": dict(Counter(sources)) if sources is not None else {}})
        for i, c in enumerate(node.children): walk(c, f"{path}.{i}")
    walk(n, "root"); return out

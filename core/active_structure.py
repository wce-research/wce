"""DÉCOUVERTE STRUCTURELLE ACTIVE : la recherche et l'expérimentation sont ENTRELACÉES.

PREUVES → POSTÉRIEURE SUR PROGRAMMES → CLASSES D'ÉQUIVALENCE FONCTIONNELLE → EXPÉRIENCE QUI DÉPARTAGE LES
STRUCTURES → OBSERVATION → MISE À JOUR TENANT COMPTE DU BRUIT → SCISSION / FUSION → APPROFONDIR SEULEMENT SI
NÉCESSAIRE → PROMOTION → COMPÉTENCE → CERTIFICAT.

Aucune famille n'est nommée ici (le banc nommé vit dans les générateurs d'évaluation, jamais importés ici).
Le niveau de bruit d'étiquette (fiabilité de la source) est une entrée CONNUE du modèle de bruit.
"""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from core.boundary_program import BNode, _inits, atoms, canonical, complexity, evaluate, fit, predict, render, structure_hash

COMBINATORS = ("and", "or", "xor")


def grid(n=30, lo=-3.0, hi=3.0):
    t = torch.linspace(lo, hi, n); return torch.stack(torch.meshgrid(t, t, indexing="ij"), -1).reshape(-1, 2)


PROBES = grid(30)                         # sondes canoniques (région complète)
CANON = grid(24, -2.5, 2.5)               # repère canonique


# ================================================================ signatures, équivalence fonctionnelle, canonisation

def components(mask_grid: torch.Tensor):
    """Nombre de composantes connexes (4-voisinage) d'un masque de grille carré."""
    n = int(math.isqrt(mask_grid.numel())); M = mask_grid.reshape(n, n).tolist(); seen = set(); comps = 0
    for i in range(n):
        for j in range(n):
            if M[i][j] and (i, j) not in seen:
                comps += 1; stack = [(i, j)]; seen.add((i, j))
                while stack:
                    a, b = stack.pop()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        u, v = a + da, b + db
                        if 0 <= u < n and 0 <= v < n and M[u][v] and (u, v) not in seen: seen.add((u, v)); stack.append((u, v))
    return comps


def canonical_frame(pred_fn: Callable, probes=PROBES):
    """Repère canonique de la région positive : centre, axes principaux, échelle, orientation par moments d'ordre 3
    (invariance aux translations, rotations, échelles, réflexions — empirique, pas une preuve)."""
    P = probes[pred_fn(probes).bool()]
    if len(P) < 4 or len(P) > 0.97 * len(probes): return None
    c = P.mean(0); C = torch.cov((P - c).T) + 1e-6 * torch.eye(2); ev, V = torch.linalg.eigh(C)
    V = V[:, [1, 0]]; s = float(ev.max().sqrt()) + 1e-6; Z = (P - c) @ V
    sign = torch.sign((Z ** 3).mean(0)); sign[sign == 0] = 1.0
    return c, V * sign, s


def canonical_signature(pred_fn: Callable):
    fr = canonical_frame(pred_fn)
    if fr is None: return pred_fn(CANON).bool()
    c, V, s = fr; X = c + (CANON * s) @ V.T
    return pred_fn(X).bool()


@dataclass
class EquivalenceSignature:
    probes: torch.Tensor                  # prédictions sur sondes canoniques
    canonical: torch.Tensor               # prédictions dans le repère canonique (invariant aux transformations)
    components: int                       # topologie de la région positive
    area: float
    reflection_symmetry: float            # accord sous réflexion (axe principal) dans le repère canonique

    @staticmethod
    def of(pred_fn: Callable):
        p = pred_fn(PROBES).bool(); cs = canonical_signature(pred_fn); n = int(math.isqrt(cs.numel()))
        refl = float((cs.reshape(n, n) == cs.reshape(n, n).flip(1)).float().mean())
        return EquivalenceSignature(p, cs, components(p), float(p.float().mean()), refl)

    def distance(self, other, canonical=False):
        a, b = (self.canonical, other.canonical) if canonical else (self.probes, other.probes)
        return float((a != b).float().mean())


def program_signature(prog: BNode): return EquivalenceSignature.of(lambda X: predict(prog, X))


@dataclass
class FunctionalEquivalenceClass:
    """Programmes d'AST différents regroupés par comportement (équivalence EMPIRIQUE sur sondes, jamais une preuve).
    Plusieurs représentations sont conservées (pas de syntaxe canonique imposée)."""
    members: List[BNode]
    signature: EquivalenceSignature

    def representations(self): return sorted({structure_hash(m) for m in self.members})


def functional_clusters(programs: Sequence[BNode], threshold=0.03, canonical=False, signatures=None):
    """Regroupement glouton par distance de signature (et même topologie)."""
    sigs = signatures or [program_signature(p) for p in programs]; out: List[FunctionalEquivalenceClass] = []; idx = []
    for p, s in zip(programs, sigs):
        for k, cl in enumerate(out):
            if cl.signature.components == s.components and cl.signature.distance(s, canonical) <= threshold:
                cl.members.append(p); idx.append(k); break
        else:
            out.append(FunctionalEquivalenceClass([p], s)); idx.append(len(out) - 1)
    return out, idx


# ================================================================ postérieure structurelle

class StructuralPosterior:
    """P(programme), P(classe fonctionnelle), entropies paramétrique / structurelle / fonctionnelle.
    Vraisemblance avec bruit d'étiquette ε CONNU (fiabilité de la source)."""

    def __init__(self, programs: Sequence[BNode], eps=0.03, threshold=0.03, use_functional=True):
        self.programs: List[BNode] = []; self.logw = torch.zeros(0, dtype=torch.float64); self.eps = max(eps, 0.01)
        self.threshold, self.use_functional = threshold, use_functional; self.sigs = []
        self._pred_cache = {}
        self.add(programs)

    def add(self, programs, logprior=None):
        for p in programs:
            self.programs.append(p); self.sigs.append(program_signature(p))
            lp = -0.3 * complexity(p) if logprior is None else logprior
            self.logw = torch.cat([self.logw, torch.tensor([lp], dtype=torch.float64)])
        self._recluster()

    def _recluster(self):
        if self.use_functional: self.classes, self.cls = functional_clusters(self.programs, self.threshold, signatures=self.sigs)
        else:                                                          # classes = structure AST (contrôle)
            keys = {}; self.cls = [keys.setdefault(structure_hash(p), len(keys)) for p in self.programs]
            self.classes = [FunctionalEquivalenceClass([p for p, c in zip(self.programs, self.cls) if c == k], None) for k in range(len(keys))]

    def preds(self, X): return torch.stack([predict(p, X) for p in self.programs]).double() if self.programs else torch.zeros(0, len(X))

    def update(self, X, y, weight=1.0):
        P = self.preds(X); ok = (P == y.double()).double()
        self.logw = self.logw + weight * (ok * math.log(1 - self.eps) + (1 - ok) * math.log(self.eps)).sum(1); return self

    def rescore(self, X, y):
        base = torch.tensor([-0.3 * complexity(p) for p in self.programs], dtype=torch.float64); self.logw = base; return self.update(X, y)

    @property
    def weights(self): return torch.softmax(self.logw, 0)

    def class_probs(self):
        w = self.weights; out = torch.zeros(len(self.classes), dtype=torch.float64)
        return out.index_add_(0, torch.tensor(self.cls), w)

    def entropy(self, which="structural"):
        p = self.class_probs() if which == "structural" else self.weights
        p = p[p > 1e-12]; return float(-(p * p.log()).sum())

    def map_class(self): cp = self.class_probs(); k = int(cp.argmax()); return k, float(cp[k])

    def map_program(self, slack=2.0):
        """Meilleur programme de la classe MAP ; parmi ceux à ≤ slack de log-vraisemblance, le plus SIMPLE."""
        k, _ = self.map_class(); idx = [i for i, c in enumerate(self.cls) if c == k]; top = max(float(self.logw[i]) for i in idx)
        near = [i for i in idx if float(self.logw[i]) >= top - slack]; return self.programs[min(near, key=lambda i: complexity(self.programs[i]))]

    def predictive(self, X): return (self.weights[:, None] * self.preds(X)).sum(0)

    def _ig(self, X, groups, gw):
        """IG(point ; variable de groupe) avec bruit ε : groups[i] = groupe du programme i."""
        P = self.preds(X); G = int(max(groups)) + 1; w = self.weights
        pg = torch.zeros(G, dtype=torch.float64).index_add_(0, torch.tensor(groups), w).clamp_min(1e-12)
        p1g = torch.zeros(G, len(X), dtype=torch.float64).index_add_(0, torch.tensor(groups), w[:, None] * P) / pg[:, None]
        py1g = self.eps + (1 - 2 * self.eps) * p1g; py1 = (pg[:, None] * py1g).sum(0)
        H = lambda q: -(q.clamp(1e-12, 1) * q.clamp(1e-12, 1).log() + (1 - q).clamp(1e-12, 1) * (1 - q).clamp(1e-12, 1).log())
        return H(py1) - (pg[:, None] * H(py1g)).sum(0)

    def ig_structure(self, X): return self._ig(X, self.cls, None)
    def ig_parameters(self, X):
        k, _ = self.map_class(); groups = [i if c == k else len(self.programs) for i, c in enumerate(self.cls)]
        return self._ig(X, groups, None)
    def predictive_entropy(self, X):
        p = (self.eps + (1 - 2 * self.eps) * self.predictive(X)).clamp(1e-12, 1 - 1e-12); return -(p * p.log() + (1 - p) * (1 - p).log())
    def disagreement(self, X): p = self.predictive(X); return 1 - (2 * p - 1).abs()

    def disagreement_region(self, X, min_mass=0.05):
        """DisagreementRegion : points où les CLASSES plausibles (masse ≥ min_mass) prédisent différemment."""
        cp = self.class_probs(); keep = [k for k in range(len(self.classes)) if float(cp[k]) >= min_mass]
        if len(keep) < 2: return X[:0]
        P = self.preds(X); w = self.weights; votes = []
        for k in keep:
            idx = torch.tensor([i for i, c in enumerate(self.cls) if c == k]); votes.append(((w[idx, None] * P[idx]).sum(0) / w[idx].sum()) >= 0.5)
        V = torch.stack(votes); return X[(V != V[0]).any(0)]


# ================================================================ classification des requêtes, identifiabilité, espace des versions

def query_type(post: StructuralPosterior, x):
    s, p = float(post.ig_structure(x)[0]), float(post.ig_parameters(x)[0])
    d = float(post.disagreement(x)[0])
    if s >= max(p, 0.02): return "structure-disambiguating"
    if p >= 0.02: return "parameter-disambiguating"
    if d > 0.6: return "boundary-calibrating"
    return "truth-checking"


def identifiability(post: StructuralPosterior, pool, min_mass=0.05):
    """Les classes plausibles sont-elles distinguables par une expérience du pool ? Sinon NON IDENTIFIABLE."""
    cp = post.class_probs(); keep = [k for k in range(len(post.classes)) if float(cp[k]) >= min_mass]
    if len(keep) < 2: return {"identifiable": True, "plausible_classes": len(keep), "pairs": 0, "distinguishable_pairs": 0}
    P = post.preds(pool); reps = [P[[i for i, c in enumerate(post.cls) if c == k][0]] for k in keep]; pairs = dist = 0
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)): pairs += 1; dist += bool((reps[i] != reps[j]).any())
    return {"identifiable": dist == pairs, "plausible_classes": len(keep), "pairs": pairs, "distinguishable_pairs": dist}


def minimal_distinguishing_set(post: StructuralPosterior, pool, min_mass=0.05):
    """Couverture d'ensemble GLOUTONNE : plus petit ensemble de points séparant toutes les paires de classes plausibles."""
    cp = post.class_probs(); keep = [k for k in range(len(post.classes)) if float(cp[k]) >= min_mass]
    P = post.preds(pool); reps = {k: P[[i for i, c in enumerate(post.cls) if c == k][0]] for k in keep}
    pairs = {(a, b) for a in keep for b in keep if a < b and bool((reps[a] != reps[b]).any())}; chosen = []
    while pairs:
        best = max(range(len(pool)), key=lambda t: sum(reps[a][t] != reps[b][t] for a, b in pairs))
        chosen.append(best); pairs = {(a, b) for a, b in pairs if reps[a][best] == reps[b][best]}
    return pool[chosen]


def version_space(post: StructuralPosterior, X, y, slack=None):
    """Programmes encore COMPATIBLES : erreurs ≤ attendu sous bruit (ε·n + 2,5·√(ε(1−ε)n) + 1)."""
    n = len(y); slack = slack if slack is not None else post.eps * n + 2.5 * math.sqrt(post.eps * (1 - post.eps) * n) + 1
    err = (post.preds(X) != y.double()).sum(1); keep = (err <= slack)
    return [p for p, k in zip(post.programs, keep.tolist()) if k]


# ================================================================ politique de recherche (proposeur appris)

def expansion_key(parent: BNode, atom: BNode, comb: str, depth: int):
    pk = parent.kind if parent.kind != "translate" else "T:" + parent.children[0].feat
    ak = atom.kind if atom.kind != "translate" else "T:" + (atom.children[0].feat if atom.children[0].kind == "pred" else "not")
    if atom.kind == "pred": ak = f"{atom.feat}:{atom.op}"
    if atom.kind == "not": ak = "not:" + atom.children[0].feat
    return (min(depth, 3), pk, comb, ak)


class SearchPolicy:
    """Petit proposeur (comptages lissés ≈ modèle linéaire sur indicatrices) : contexte d'expansion → utilité.
    Il PROPOSE seulement (top-k) ; la vérité vient des données et des expériences. Il décide aussi « chercher
    vs demander » (règle explicite) et le filtrage contextuel des primitives (primitive gating)."""

    def __init__(self, k=12, alpha=1.0): self.k, self.alpha = k, alpha; self.pos = Counter(); self.tot = Counter(); self.macros = Counter()

    def record(self, key, useful): self.tot[key] += 1; self.pos[key] += bool(useful)

    def score(self, key): return (self.pos[key] + self.alpha) / (self.tot[key] + 2 * self.alpha)

    def propose(self, parent, A, depth, guided=True):
        cands = [(c, a) for c in COMBINATORS for a in A]
        if not guided: return cands
        per = max(1, self.k // len(COMBINATORS)); out = []                # diversité : top-k PAR combinateur
        for c in COMBINATORS:
            out += sorted([t for t in cands if t[0] == c], key=lambda t: -self.score(expansion_key(parent, t[1], t[0], depth)))[:per]
        return out

    def should_search(self, fit_err, eps, max_ig, structural_entropy):
        """Chercher plus profond si rien n'explique les données, ou si les requêtes ne départagent plus (IG≈0) alors
        que l'ajustement reste insuffisant."""
        return fit_err > eps + 0.03 or (max_ig < 0.005 and fit_err > eps + 0.015)

    def record_trajectory(self, traj): self.macros[tuple(traj[:6])] += 1


# ================================================================ ajustement à chaud (extension de programme)

def fit_extension(parent: BNode, atom: BNode, comb: str, X, y, steps=90, n_init=2, seed=0):
    """Ajuste combinateur(parent, atome) en partant des paramètres DÉJÀ ajustés du parent (recherche à chaud)."""
    skel = BNode(comb, children=(parent.skeleton(), atom)); base = parent.param_vector()
    g = torch.Generator().manual_seed(seed); best = None; yf = y.float()
    for th_a in _inits(atom, X, y, n_init, g):
        th = torch.tensor(base + th_a.tolist()).requires_grad_(True); opt = torch.optim.Adam([th], lr=0.05)
        for t in range(steps):
            k = 4.0 + 16.0 * t / steps; opt.zero_grad()
            torch.nn.functional.binary_cross_entropy(evaluate(skel, X, th, k).clamp(1e-5, 1 - 1e-5), yf).backward(); opt.step()
        prog = skel.with_params(th.detach().tolist()); acc = float((predict(prog, X) == y).float().mean())
        if best is None or acc > best[1]: best = (prog, acc)
    return best


# ================================================================ boucle entrelacée

@dataclass
class DiscoveryResult:
    program: BNode
    posterior: StructuralPosterior
    queries: int
    evaluations: int
    depth: int
    seconds: float
    adequate: bool
    trace: List[dict] = field(default_factory=list)
    entropy_curve: List[float] = field(default_factory=list)
    version_space_curve: List[int] = field(default_factory=list)
    status: str = ""


class ActiveStructuralDiscovery:
    """query_policy ∈ {structure_ig, parameter_ig, predictive_entropy, disagreement, random} ;
    guided : proposeur ; functional : classes fonctionnelles (sinon classes AST) ; noise_aware : stabilité
    fonctionnelle avec tolérances dépendant du bruit."""

    def __init__(self, policy: Optional[SearchPolicy] = None, guided=True, functional=True, query_policy="structure_ig", budget=16,
                 max_depth=4, beam=3, eps=0.03, seed=0, allow_repeat=True, record=False, min_queries=8):
        self.policy = policy or SearchPolicy(); self.guided, self.functional, self.qp = guided, functional, query_policy
        self.min_queries = min_queries
        self.budget, self.max_depth, self.beam, self.eps, self.seed, self.allow_repeat, self.record = budget, max_depth, beam, max(eps, 0.01), seed, allow_repeat, record

    def _err(self, p, X, y): return float((predict(p, X) != y).float().mean())

    def run(self, X, y, oracle: Callable, pool_sampler: Callable, g):
        t0 = time.time(); A = atoms(); evals = 0; trace = []; traj = []
        Xa, ya = X.clone(), y.clone()
        progs = []
        for a in A:
            p, _ = fit(a, Xa, ya, steps=100, n_init=2, seed=self.seed); progs.append(p); evals += 1
        post = StructuralPosterior(progs, self.eps, use_functional=self.functional); post.update(Xa, ya)
        depth = 1; q = 0; ent, vs = [post.entropy()], [len(version_space(post, Xa, ya))]
        pool = pool_sampler(500, g); repeats = []; stalls = 0; self._force = False
        while True:
            best = post.map_program(); err = self._err(best, Xa, ya)
            ig = self._query_scores(post, pool); max_ig = float(ig.max()) if len(ig) else 0.0
            if depth < self.max_depth and (self._force or depth == 1 or self.policy.should_search(err, self.eps, max_ig, post.entropy())):
                self._force = False
                depth += 1; new = []
                parents = sorted(range(len(post.programs)), key=lambda i: -float(post.logw[i]))[: self.beam]
                for i in parents:
                    par = post.programs[i]
                    for comb, a in self.policy.propose(par, A, depth, self.guided):
                        p, acc = fit_extension(par, a, comb, Xa, ya, seed=self.seed + evals); evals += 1; new.append(p)
                        if self.record: trace.append({"key": expansion_key(par, a, comb, depth), "gain": (1 - err) - acc})
                post.add(new); post.rescore(Xa, ya); traj.append(f"deepen{depth}")
                continue
            if q >= self.budget: break
            k, pk = post.map_class()
            if q >= self.min_queries and pk >= 0.97 and max_ig < 0.01 and err <= self.eps + 0.03:
                if noise_aware_stability(best, X, y, self.eps)["accept"] or depth >= self.max_depth: break      # plus rien à départager
                stalls += 1
                if stalls <= 2: self._force = True; continue
                break
            # expérience : nouveau point ou répétition d'un point informatif bruité
            i = int(ig.argmax()); xq = pool[i:i + 1]
            if self.allow_repeat and repeats:
                R = torch.stack(repeats); igr = self._query_scores(post, R)
                if float(igr.max()) > 0.8 * float(ig[i]) and float(post.disagreement(R[int(igr.argmax()):int(igr.argmax()) + 1])[0]) > 0.5:
                    xq = R[int(igr.argmax()):int(igr.argmax()) + 1]; traj.append("repeat")
            yq = oracle(xq)
            if float(torch.rand(1, generator=g)) < self.eps_world: yq = 1 - yq
            Xa, ya = torch.cat([Xa, xq]), torch.cat([ya, yq]); q += 1; repeats.append(xq[0]); traj.append(query_type(post, xq))
            post.update(xq, yq)
            if q % 4 == 0:                                              # réajustement des survivants (paramètres)
                top = sorted(range(len(post.programs)), key=lambda i: -float(post.logw[i]))[:4]
                post.add([fit(post.programs[i].skeleton(), Xa, ya, steps=80, n_init=1, seed=q)[0] for i in top]); evals += len(top); post.rescore(Xa, ya)
            ent.append(post.entropy()); vs.append(len(version_space(post, Xa, ya)))
        best = simplify(post.map_program(), Xa, ya, self.eps); evals += 1
        st = noise_aware_stability(best, X, y, self.eps)          # sur l'échantillon passif représentatif (comme la calibration)
        if self.record: self.policy.record_trajectory(traj)
        status = "identified" if st["accept"] and post.map_class()[1] >= 0.9 else ("unsupported" if not st["accept"] else "ambiguous")
        idf = identifiability(post, pool)
        if status == "ambiguous" and not idf["identifiable"]: status = "non_identifiable"
        return DiscoveryResult(best, post, q, evals, depth, round(time.time() - t0, 2), st["accept"], trace, ent, vs, status)

    eps_world = 0.0                                                      # bruit RÉEL des requêtes (fixé par l'appelant)

    def _query_scores(self, post, pool):
        if self.qp == "structure_ig": return post.ig_structure(pool)
        if self.qp == "parameter_ig": return post.ig_parameters(pool)
        if self.qp == "predictive_entropy": return post.predictive_entropy(pool)
        if self.qp == "disagreement": return post.disagreement(pool)
        return torch.rand(len(pool), generator=torch.Generator().manual_seed(len(post.programs) + int(post.logw.sum().item()) % 1000)).double()


def train_policy(tasks, eps=0.03, seed=0):
    """Apprend le proposeur sur des tâches d'ENTRAÎNEMENT : recherche non guidée enregistrée ; une expansion est
    « utile » si elle réduit l'erreur d'ajustement de ≥ 0,02 par rapport au parent."""
    pol = SearchPolicy(); rec = ActiveStructuralDiscovery(pol, guided=False, budget=4, eps=eps, seed=seed, record=True)
    for X, y, oracle, sampler, g in tasks:
        r = rec.run(X, y, oracle, sampler, g)
        for t in r.trace: pol.record(t["key"], t["gain"] < -0.02)
    return pol


def _subtrees_replacements(n: BNode):
    """Variantes obtenues en remplaçant UN combinateur par l'un de ses enfants (élagage d'une branche)."""
    out = []
    if n.kind in COMBINATORS: out += list(n.children)
    for i, c in enumerate(n.children):
        for r in _subtrees_replacements(c):
            ch = list(n.children); ch[i] = r; out.append(BNode(n.kind, n.feat, n.op, n.params, tuple(ch)))
    return out


def simplify(prog: BNode, X, y, eps):
    """SIMPLIFIER : élagage glouton des branches tant que l'erreur n'augmente pas au-delà du bruit d'échantillonnage
    (1 erreur + 0,5·√(ε n))."""
    tol = (1 + 0.5 * math.sqrt(max(eps, 0.01) * len(X))) / len(X); cur = prog; e0 = float((predict(cur, X) != y).float().mean())
    while True:
        best = None
        for r in _subtrees_replacements(cur):
            e = float((predict(r, X) != y).float().mean())
            if e <= e0 + tol and (best is None or complexity(r) < complexity(best[0])): best = (r, e)
        if best is None or complexity(best[0]) >= complexity(cur): return cur
        cur = fit_from(best[0], X, y, steps=60) if best[0].n_params() else best[0]; e0 = float((predict(cur, X) != y).float().mean())


def fit_from(prog: BNode, X, y, steps=80, lr=0.03):
    """Réajustement À CHAUD depuis les paramètres actuels (structure fixée)."""
    skel = prog.skeleton(); th = torch.tensor(prog.param_vector()).requires_grad_(True); opt = torch.optim.Adam([th], lr=lr); yf = y.float()
    for t in range(steps):
        opt.zero_grad(); torch.nn.functional.binary_cross_entropy(evaluate(skel, X, th, 12.0).clamp(1e-5, 1 - 1e-5), yf).backward(); opt.step()
    return skel.with_params(th.detach().tolist())


# ================================================================ stabilité et ancres tenant compte du bruit

STABILITY_TABLE = {0.0: (0.04, 0.05), 0.02: (0.05, 0.08), 0.05: (0.07, 0.11), 0.08: (0.09, 0.15), 0.12: (0.11, 0.2)}   # (désaccord max, erreur en excès max) — recalibré par scripts/active_calibrate.py


def _load_calibration():
    """Table calibrée (TRAIN + VALIDATION) ; enveloppe monotone en ε (plus de bruit ⇒ tolérance ≥)."""
    import json, os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "active_stability_calibration.json")
    if not os.path.exists(path): return
    t = {float(k): tuple(v) for k, v in json.load(open(path))["table"].items()}; m = (0.0, 0.0)
    for k in sorted(t): m = (max(m[0], t[k][0]), max(m[1], t[k][1])); STABILITY_TABLE[k] = m


_load_calibration()


def _interp(eps):
    ks = sorted(STABILITY_TABLE)
    if eps <= ks[0]: return STABILITY_TABLE[ks[0]]
    for a, b in zip(ks, ks[1:]):
        if a <= eps <= b:
            t = (eps - a) / (b - a); return tuple((1 - t) * u + t * v for u, v in zip(STABILITY_TABLE[a], STABILITY_TABLE[b]))
    return STABILITY_TABLE[ks[-1]]


def noise_aware_stability(prog: BNode, X, y, eps):
    """Stabilité FONCTIONNELLE sous bruit : (1) erreur d'ajustement ≤ ε + tolérance(ε) ; (2) la même structure
    réajustée sur deux moitiés disjointes donne des fonctions dont le désaccord sur la région observée reste dans
    la variance attendue sous ce bruit. Modes séparés : prédiction / classe fonctionnelle / paramètres / AST."""
    n = len(X) // 2; dmax, emax = _interp(eps)
    a, b = fit_from(prog, X[:n], y[:n]), fit_from(prog, X[n:], y[n:])
    pred_dis = float((predict(a, X) != predict(b, X)).float().mean()); err = float((predict(prog, X) != y).float().mean())
    pa, pb = torch.tensor(a.param_vector()), torch.tensor(b.param_vector())
    sa, sb = program_signature(a), program_signature(b)
    modes = {"prediction_stability": 1 - pred_dis, "functional_class_stability": float(sa.components == sb.components and sa.distance(sb) <= 0.06),
             "parameter_stability": float(1 / (1 + (pa - pb).norm())), "ast_stability": float(structure_hash(a) == structure_hash(b))}
    emax = max(emax, 1.5 * math.sqrt(max(eps, 0.01) * (1 - eps) / len(X)))            # variance binomiale de l'erreur sous bruit
    return {"accept": err <= eps + emax and pred_dis <= dmax, "error": err, "disagreement": pred_dis, **modes}


@dataclass
class NoiseAwareTruthAnchor:
    """P(règle fausse | observations) avec P(y | règle, x, bruit) = ε + (1−2ε)·σ(k·marge) : près de la frontière la
    règle prédit ≈ 0,5 (tolérance large) ; loin, elle est stricte. Hypothèses concurrentes : programmes alternatifs
    + modèle nul (0,5). Test SÉQUENTIEL avec répétitions possibles ; seuils de décision fixés a priori (0,9 / 0,1)."""
    rule: BNode
    alternatives: List[BNode]
    eps: float
    k: float = 6.0
    prior_rule: float = 0.5

    def __post_init__(self):
        m = len(self.alternatives); self.logp = torch.tensor([math.log(self.prior_rule)] + [math.log((1 - self.prior_rule) * 0.7 / max(1, m))] * m + [math.log((1 - self.prior_rule) * 0.3)], dtype=torch.float64)
        self.n = 0

    def _lik(self, prog, X, y):
        p = (self.eps + (1 - 2 * self.eps) * evaluate(prog, X, k=self.k)).double().clamp(1e-6, 1 - 1e-6); return torch.where(y.bool(), p, 1 - p)

    def observe(self, X, y):
        L = [self._lik(self.rule, X, y)] + [self._lik(a, X, y) for a in self.alternatives] + [torch.full((len(X),), 0.5, dtype=torch.float64)]
        self.logp = self.logp + torch.stack(L).log().sum(1); self.n += len(X); return self

    def p_wrong(self): return float(1 - torch.softmax(self.logp, 0)[0])

    def decision(self, hi=0.9, lo=0.1):
        pw = self.p_wrong(); return "reject" if pw >= hi else "accept" if pw <= lo else "undecided"

    def choose(self, pool, history, cost_new=1.0, cost_repeat=0.6):
        """Utilité = discrimination attendue (IG sur {règle, alternatives, nul}) − risque de bruit − coût."""
        cands = torch.cat([pool, torch.stack(history)]) if history else pool
        w = torch.softmax(self.logp, 0)
        P = torch.stack([(self.eps + (1 - 2 * self.eps) * evaluate(h, cands, k=self.k)).double() for h in [self.rule] + self.alternatives] + [torch.full((len(cands),), 0.5, dtype=torch.float64)])
        py = (w[:, None] * P).sum(0); H = lambda q: -(q * q.clamp_min(1e-12).log() + (1 - q) * (1 - q).clamp_min(1e-12).log())
        ig = H(py) - (w[:, None] * H(P)).sum(0)
        cost = torch.full((len(cands),), cost_new, dtype=torch.float64)
        if history: cost[len(pool):] = cost_repeat
        u = ig - 0.02 * cost; i = int(u.argmax()); return cands[i:i + 1], i >= len(pool)


def sequential_truth_test(rule, alternatives, oracle, pool_sampler, g, eps, budget=16, noise=None):
    anc = NoiseAwareTruthAnchor(rule, alternatives, eps); hist = []; repeats = 0
    pool = pool_sampler(400, g)
    for t in range(budget):
        x, rep = anc.choose(pool, hist); y = oracle(x)
        if float(torch.rand(1, generator=g)) < (eps if noise is None else noise): y = 1 - y
        anc.observe(x, y); hist.append(x[0]); repeats += rep
        if anc.decision() != "undecided": break
    return {"p_wrong": anc.p_wrong(), "decision": anc.decision(), "queries": anc.n, "repeats": repeats}


# ================================================================ résidu à porte (contrôle OOD)

def gated_residual(prog: BNode, X, y, h=3, steps=400, radius=0.35, seed=0):
    """program(x) + gate(x)·résidu(x) : la porte n'est ouverte QUE près des erreurs d'entraînement du programme
    ET dans l'enveloppe des données observées (boîte englobante) ; hors région, la sortie = programme (OOD préservé)."""
    from core.boundary_methods import train_residual
    errs = X[predict(prog, X) != y]; lo, hi = X.min(0).values, X.max(0).values
    f, rp = train_residual(prog, X, y, h=h, steps=steps, seed=seed)
    def gate(Q):
        inside = ((Q >= lo) & (Q <= hi)).all(1)
        if len(errs) == 0: return torch.zeros(len(Q), dtype=torch.bool)
        return inside & (torch.cdist(Q, errs).min(1).values <= radius)
    def g_pred(Q):
        Q = torch.nan_to_num(Q); base = predict(prog, Q); m = gate(Q); out = base.clone()
        if int(m.sum()): out[m] = f(Q[m])
        return out
    return g_pred, rp + 2 * len(errs), gate


# ================================================================ familles fonctionnelles (orbites)

@dataclass
class OrbitFamily:
    """Famille = fonction canonique (signature dans le repère canonique) + orbite sous les transformations
    admissibles (translation, rotation, échelle, réflexion) + représentations (AST) connues."""
    canonical: EquivalenceSignature
    representations: List[BNode]
    support: int = 1
    status: str = "HYPOTHESIS"
    history: List[str] = field(default_factory=list)

    def matches(self, sig: EquivalenceSignature, threshold=0.08): return self.canonical.components == sig.components and self.canonical.distance(sig, canonical=True) <= threshold

    def update(self, ok, note=""):
        self.history.append(("+" if ok else "−") + note)
        pos = sum(h.startswith("+") for h in self.history); neg = len(self.history) - pos
        self.status = "REFUTED" if neg >= max(2, pos) else "STABLE" if pos >= 3 and neg == 0 else "SUPPORTED" if pos >= 2 else "HYPOTHESIS"
        if self.status == "STABLE" and neg > 0: self.status = "SUPPORTED"
        return self.status


def induce_orbit_families(programs: Sequence[BNode], min_support=2, threshold=0.08, canonical=True):
    """Regroupement fonctionnel (canonique ou brut) → familles ; plusieurs AST par famille autorisés."""
    fams: List[OrbitFamily] = []
    for p in programs:
        s = program_signature(p)
        for f in fams:
            if (f.matches(s, threshold) if canonical else (f.canonical.components == s.components and f.canonical.distance(s) <= threshold)):
                f.representations.append(p); f.support += 1; break
        else: fams.append(OrbitFamily(s, [p]))
    return [f for f in fams if f.support >= min_support]


def induce_ast_families(programs: Sequence[BNode], min_support=2):
    groups = defaultdict(list)
    for p in programs: groups[structure_hash(p)].append(p)
    return [OrbitFamily(program_signature(ps[0]), ps, len(ps)) for ps in groups.values() if len(ps) >= min_support]


def reuse_family(fams: Sequence[OrbitFamily], X, y, canonical=True, eps=0.03):
    """Nouvelle instance : choisir la famille par signature (canonique ou brute) d'un ajustement rapide, puis
    ajuster les paramètres de ses représentations (a priori de famille)."""
    probe, _ = fit(atoms()[0], X, y, steps=60, n_init=1)
    knn = lambda Q: _knn(X, y, Q); s = EquivalenceSignature.of(knn); evals = 1
    order = sorted(fams, key=lambda f: f.canonical.distance(s, canonical=canonical))
    best = None
    for f in order[:2]:
        for rep in f.representations[:3]:
            p, acc = fit(rep.skeleton(), X, y, steps=100, n_init=2); evals += 1
            if best is None or acc > best[1]: best = (p, acc)
    return best[0] if best else None, evals


def _knn(X, y, Q, k=5):
    d = torch.cdist(Q, X); idx = d.topk(k, largest=False).indices; return (y[idx].float().mean(1) >= 0.5).long()

"""Certificats d'invariants GÉNÉRATIFS : une compétence est protégée par un MODÈLE COMPACT de sa région de validité
(générateur appris), ses invariants (programme de référence : règle découverte ou petit substitut), sa géométrie
locale de sensibilité et ses dépendances — JAMAIS par son jeu de données d'origine.

Interface commune `CertificateGenerator` : sample, sample_boundary, sample_high_sensitivity, sample_rare,
sample_counterfactual (+ sample_adversarial). Générateurs : continu (gaussien robuste faible rang appris) et
symbolique (gabarits obtenus par ANTI-UNIFICATION des exemples observés + distributions de remplissage apprises).
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

FLOAT_BYTES = 4


# ================================================================ interface

class CertificateGenerator:
    domain = "?"
    def sample(self, n, g): raise NotImplementedError
    def sample_boundary(self, n, g, label_fn): raise NotImplementedError
    def sample_high_sensitivity(self, n, g, model_fn): raise NotImplementedError
    def sample_rare(self, n, g): raise NotImplementedError
    def sample_counterfactual(self, n, g, label_fn): raise NotImplementedError
    def memory_bytes(self): raise NotImplementedError


def clean_inputs(X, clip=6.0):
    return torch.nan_to_num(X, nan=0.0).clamp(-clip, clip)


# ================================================================ générateur continu appris

class ContinuousGenerator(CertificateGenerator):
    """Appris depuis l'expérience : centre et échelle ROBUSTES (médiane / MAD, NaN ignorés, outliers écrêtés),
    covariance faible rang (ACP, k composantes) + diagonale, plages de validité (quantiles 1 %–99 %)."""
    domain = "continuous"

    def __init__(self, k=3):
        self.k = k; self.mean = self.diag = self.U = self.s = self.lo = self.hi = None; self.missing = 0.0; self.outlier = 0.0

    def fit(self, X):
        Xn = X.clone(); nan = torch.isnan(Xn); self.missing = float(nan.float().mean())
        med = torch.nanmedian(Xn, 0).values; Xn[nan] = med.expand_as(Xn)[nan]
        mad = (Xn - med).abs().median(0).values * 1.4826 + 1e-3
        z = (Xn - med) / mad; out = (z.abs() > 4).any(1); self.outlier = float(out.float().mean())
        Xc = Xn[~out] if int((~out).sum()) > 20 else Xn
        self.mean = Xc.mean(0); R = Xc - self.mean
        U, S, Vh = torch.linalg.svd(R / math.sqrt(max(1, len(R) - 1)), full_matrices=False)
        self.U = Vh[: self.k].T; self.s = S[: self.k]
        resid = R - (R @ self.U) @ self.U.T; self.diag = resid.var(0).clamp_min(1e-4)
        self.lo, self.hi = torch.quantile(Xc, 0.01, 0), torch.quantile(Xc, 0.99, 0)
        return self

    def _z(self, n, g, radius=None):
        z = torch.randn(n, self.k, generator=g); e = torch.randn(n, len(self.mean), generator=g)
        if radius is not None:                                        # coquille de Mahalanobis (cas rares)
            full = torch.cat([z, e], 1); full = full / full.norm(dim=1, keepdim=True) * radius[:, None]
            z, e = full[:, : self.k], full[:, self.k:] / math.sqrt(max(1.0, (len(self.mean)) / self.k))
        return z, e

    def sample(self, n, g, radius=None):
        z, e = self._z(n, g, radius)
        return self.mean + (z * self.s) @ self.U.T + e * self.diag.sqrt()

    def sample_rare(self, n, g):
        r = 2.5 + torch.rand(n, generator=g) * 1.0
        z = torch.randn(n, self.k, generator=g); z = z / z.norm(dim=1, keepdim=True) * r[:, None]
        e = torch.randn(n, len(self.mean), generator=g) * 1.2
        return self.mean + (z * self.s) @ self.U.T + e * self.diag.sqrt()

    def sample_boundary(self, n, g, label_fn, steps=14, jitter=0.03):
        """Recherche STRUCTURELLE : bissection le long de segments entre points d'étiquettes opposées."""
        out = []
        for _ in range(8):
            A, B = self.sample(4 * n, g), self.sample(4 * n, g); la, lb = label_fn(A), label_fn(B)
            m = la != lb; A, B, la = A[m], B[m], la[m]
            for _ in range(steps):
                M = (A + B) / 2; lm = label_fn(M); same = lm == la
                A = torch.where(same[:, None], M, A); B = torch.where(same[:, None], B, M)
            out.append((A + B) / 2)
            if sum(len(o) for o in out) >= n: break
        P = torch.cat(out)[:n] if out else self.sample(n, g)
        return P + jitter * torch.randn(P.shape, generator=g) * self.diag.sqrt()

    def sample_counterfactual(self, n, g, label_fn):
        """Paires contrefactuelles : x et son image de l'autre côté de la frontière la plus proche (même direction)."""
        X = self.sample(n, g); B = self.sample_boundary(n, g, label_fn, jitter=0.0)
        k = min(len(X), len(B)); return torch.cat([X[:k], B[:k] + 0.15 * (B[:k] - X[:k]) / (B[:k] - X[:k]).norm(dim=1, keepdim=True).clamp_min(1e-6)])

    def sample_high_sensitivity(self, n, g, model_fn, pool=8):
        """Points de plus forte sensibilité d'entrée (norme du gradient de la marge, si différentiable)."""
        X = self.sample(pool * n, g).requires_grad_(True)
        o = model_fn(X); m = (o[:, 1] - o[:, 0]).sum()
        gr = torch.autograd.grad(m, X)[0].norm(dim=1)
        return X.detach()[gr.argsort(descending=True)[:n]]

    def sample_adversarial(self, n, g, model_fn, ref_fn, steps=25, lr=0.1, start=None):
        """Recherche de désaccord maximal entre le modèle et la référence (invariant), à l'intérieur de la région
        de validité (Mahalanobis ≤ 3,5 dans la base apprise)."""
        X = (start if start is not None else self.sample(n, g)).clone().requires_grad_(True)
        y = ref_fn(X.detach()); s = (2 * y - 1).float()
        opt = torch.optim.Adam([X], lr=lr)
        for _ in range(steps):
            opt.zero_grad(); o = model_fn(X); (s * (o[:, 1] - o[:, 0])).sum().backward(); opt.step()
            with torch.no_grad(): X.data = torch.maximum(torch.minimum(X.data, self.hi + 1.0), self.lo - 1.0)
        return X.detach()

    def floats(self): return sum(t.numel() for t in (self.mean, self.diag, self.U, self.s, self.lo, self.hi)) + 2
    def memory_bytes(self): return FLOAT_BYTES * self.floats()


# ================================================================ générateur symbolique appris (anti-unification)

class SymbolicGenerator(CertificateGenerator):
    """Gabarits par ANTI-UNIFICATION positionnelle des exemples (jetons) de même forme ; les positions variables
    deviennent des emplacements à distribution apprise (catégorielle, ou entière bornée si numérique)."""
    domain = "symbolic"

    def __init__(self, tokenize=lambda s: str(s).split(), detok=lambda t: " ".join(t), rebuild=None):
        self.tokenize, self.detok, self.rebuild = tokenize, detok, rebuild or (lambda s: s)
        self.templates: List[Tuple[Tuple, Dict[int, Any], int]] = []

    def fit(self, examples):
        groups = defaultdict(list)
        for x in examples: t = self.tokenize(x); groups[len(t)].append(t)
        for L, ts in groups.items():
            tpl, slots = [], {}
            for i in range(L):
                col = [t[i] for t in ts]
                if len(set(col)) == 1: tpl.append(col[0])
                else:
                    tpl.append(f"?{i}")
                    nums = [c for c in col if _isnum(c)]
                    slots[i] = ("int", min(int(float(c)) for c in nums), max(int(float(c)) for c in nums)) if len(nums) == len(col) else ("cat", Counter(col))
            self.templates.append((tuple(tpl), slots, len(ts)))
        return self

    def sample(self, n, g, rare=False):
        rng = random.Random(int(torch.randint(0, 10 ** 9, (1,), generator=g))); out = []
        w = [c for _, _, c in self.templates]
        for _ in range(n):
            tpl, slots, _ = rng.choices(self.templates, weights=w)[0]; toks = list(tpl)
            for i, sl in slots.items():
                if sl[0] == "int":
                    lo, hi = sl[1], sl[2]
                    if rare: toks[i] = str(rng.choice([lo, hi, hi + 1, lo - 1]))
                    else: toks[i] = str(rng.randint(lo, hi))
                else:
                    vals, cnt = zip(*sl[1].items())
                    toks[i] = (min(sl[1], key=sl[1].get) if rare else rng.choices(vals, weights=cnt)[0])
            out.append(self.rebuild(self.detok(toks)))
        return out

    def sample_rare(self, n, g): return self.sample(n, g, rare=True)
    def sample_boundary(self, n, g, label_fn): return self.sample(n, g, rare=True)
    def sample_high_sensitivity(self, n, g, model_fn): return self.sample(n, g)
    def sample_counterfactual(self, n, g, label_fn): return self.sample(n, g, rare=True)

    def floats(self):
        return sum(len(t) + sum(3 if s[0] == "int" else 2 * len(s[1]) for s in sl.values()) for t, sl, _ in self.templates)
    def memory_bytes(self): return 8 * self.floats()


def _isnum(c):
    try: float(c); return True
    except ValueError: return False


# ================================================================ substitut de référence (invariant de sortie)

class TinySurrogate:
    """Substitut neuronal COMPACT (16→6→2) distillé sur des échantillons GÉNÉRÉS étiquetés par la compétence au
    moment de la certification — utilisé quand aucune règle symbolique fidèle n'a été découverte."""
    def __init__(self, gen, label_fn, d, h=24, n=4000, steps=1200, seed=0):
        g = torch.Generator().manual_seed(seed)
        X = torch.cat([gen.sample(n, g), gen.sample_boundary(n // 3, g, label_fn), gen.sample_rare(n // 5, g)]); y = label_fn(X)
        self.p = {"w1": torch.randn(h, d, generator=g) * 0.5, "b1": torch.zeros(h), "w2": torch.randn(2, h, generator=g) * 0.5, "b2": torch.zeros(2)}
        q = {k: v.clone().requires_grad_(True) for k, v in self.p.items()}; opt = torch.optim.Adam(q.values(), lr=0.03)
        for _ in range(steps):
            opt.zero_grad(); torch.nn.functional.cross_entropy(torch.tanh(X @ q["w1"].T + q["b1"]) @ q["w2"].T + q["b2"], y).backward(); opt.step()
        self.p = {k: v.detach() for k, v in q.items()}
        with torch.no_grad(): self.fidelity = float((self(X) == y).float().mean())

    def __call__(self, X):
        with torch.no_grad(): X = clean_inputs(X); return (torch.tanh(X @ self.p["w1"].T + self.p["b1"]) @ self.p["w2"].T + self.p["b2"]).argmax(1)

    def floats(self): return sum(v.numel() for v in self.p.values())


# ================================================================ certificat génératif

TOLERANCE = {"central": 0.04, "boundary": 0.30, "rare": 0.15, "counterfactual": 0.25, "high_sensitivity": 0.2}


@dataclass
class GenerativeInvariantCertificate:
    skill: str
    domain: str
    generator: CertificateGenerator
    reference: Callable                       # programme de référence (règle découverte ou substitut compact)
    reference_kind: str                       # "rule" | "surrogate" | "program"
    reference_floats: int
    invariants: Tuple[str, ...] = ("output_equivalence",)
    tolerance: Dict[str, float] = field(default_factory=lambda: dict(TOLERANCE))
    counterexamples: Optional[torch.Tensor] = None     # quelques contre-exemples connus (≤ 4)
    provenance: str = ""
    stability_radius: float = 0.0
    dependencies: Tuple[str, ...] = ()
    curvature: Any = None                     # StabilityEllipsoid éventuel (module partagé)
    created_at: int = 0

    def probes(self, n, g, model_fn=None, categories=("central", "boundary", "rare", "counterfactual")):
        gen = self.generator; out = {}
        for c in categories:
            if c == "central": out[c] = gen.sample(n, g)
            elif c == "boundary": out[c] = gen.sample_boundary(n, g, self.reference)
            elif c == "rare": out[c] = gen.sample_rare(n, g)
            elif c == "counterfactual": out[c] = gen.sample_counterfactual(n // 2, g, self.reference)
            elif c == "high_sensitivity" and model_fn is not None: out[c] = gen.sample_high_sensitivity(n, g, model_fn)
        return out

    def check(self, predict_fn, n=48, seed=0, model_fn=None, categories=("central", "boundary", "rare", "counterfactual")):
        """predict_fn(X) -> étiquettes de la compétence ; erreur par catégorie vs référence ; OK si ≤ tolérance."""
        g = torch.Generator().manual_seed(seed); errs = {}
        for c, X in self.probes(n, g, model_fn, categories).items():
            if len(X) == 0: continue
            errs[c] = float((predict_fn(X) != self.reference(X)).float().mean())
        if self.counterexamples is not None and len(self.counterexamples):
            errs["known_counterexamples"] = float((predict_fn(self.counterexamples) != self.reference(self.counterexamples)).float().mean())
        ok = all(e <= self.tolerance.get(c, 0.3) for c, e in errs.items())
        return ok, errs

    def calibrate(self, predict_fn, n=96, repeats=3, slack=0.05):
        """Tolérance par catégorie = désaccord MESURÉ à la certification (moyenne sur plusieurs tirages) + marge
        d'échantillonnage + slack : le certificat détecte une DÉGRADATION par rapport au niveau certifié."""
        acc = defaultdict(list)
        for r in range(repeats):
            _, errs = self.check(predict_fn, n=n, seed=10_000 + r)
            for c, e in errs.items(): acc[c].append(e)
        for c, es in acc.items():
            e = sum(es) / len(es); self.tolerance[c] = min(0.5, e + 2 * math.sqrt(max(e * (1 - e), 0.01) / 32) + slack)
        return self

    def memory_bytes(self):
        ce = 0 if self.counterexamples is None else self.counterexamples.numel()
        curv = self.curvature.floats() if self.curvature is not None else 0
        return self.generator.memory_bytes() + FLOAT_BYTES * (self.reference_floats + ce + curv + 8) + 32


# ================================================================ certificat à sondes STOCKÉES (base de comparaison)

@dataclass
class StoredProbeCertificate:
    skill: str
    probes: torch.Tensor
    labels: torch.Tensor
    tolerance: float = 0.04
    dependencies: Tuple[str, ...] = ()
    reference_kind: str = "stored"

    def check(self, predict_fn, **kw):
        err = float((predict_fn(self.probes) != self.labels).float().mean()); return err <= self.tolerance, {"stored": err}

    def memory_bytes(self): return FLOAT_BYTES * self.probes.numel() + self.labels.numel() // 8 + 16


# ================================================================ couverture : capacité de DÉTECTION de dommages

def damaged(predict_fn, kind, regime, g):
    """Dommages définis sur la VRAIE distribution du régime (et non sur le générateur, pour ne pas le favoriser) :
    boundary_shift (frontière décalée de 0,15), rare_region (erreurs au-delà du quantile 0,98 de la vraie
    distribution), local_bump (boule couvrant ≈ 3 % de la masse autour d'un point typique), global (5 % d'erreurs)."""
    from core.continuous_world import drift_rule
    mu, sc = regime.mu, regime.scale
    T = mu + sc * torch.randn(4000, len(mu), generator=g)
    if kind == "boundary_shift":
        shifted = drift_rule(regime.rule, 0.15)
        return lambda X: shifted(X)
    if kind == "rare_region":
        q = float(((T - mu) / sc).norm(dim=1).quantile(0.98))
        return lambda X: torch.where(((clean_inputs(X) - mu) / sc).norm(dim=1) > q, 1 - predict_fn(X), predict_fn(X))
    if kind == "local_bump":
        c = T[0]; rad = float((T - c).norm(dim=1).quantile(0.03))
        return lambda X: torch.where((clean_inputs(X) - c).norm(dim=1) < rad, 1 - predict_fn(X), predict_fn(X))
    if kind == "global":
        def f(X):
            y = predict_fn(X); gg = torch.Generator().manual_seed(len(X)); return torch.where(torch.rand(len(X), generator=gg) < 0.05, 1 - y, y)
        return f
    raise KeyError(kind)


def coverage_experiment(regimes, seed=0, n_support=256, n_probe=48):
    """Pour chaque régime : la compétence = la VRAIE règle (référence parfaite), on l'endommage localement et on
    mesure si un jeu de sondes le DÉTECTE. Jeux comparés (même budget de sondes) : aléatoire global N(0, I·σ),
    sondes stockées (tirées du support), sondes du certificat génératif (central + frontière + rares +
    contrefactuelles). Plus : taux de fausses alarmes sur la compétence intacte."""
    from core.continuous_world import D
    g = torch.Generator().manual_seed(seed); res = defaultdict(Counter)
    for r in regimes:
        X, C, y = r.sample(n_support, g); gen = ContinuousGenerator().fit(X); ref = r.rule
        n_each = n_probe
        sets = {"random": torch.randn(n_each * 4, D, generator=g) * 2.0,
                "stored": clean_inputs(X[torch.randperm(len(X), generator=g)[: n_each * 4]])}
        cert = GenerativeInvariantCertificate("s", "continuous", gen, ref, "rule", r.rule.size())
        P = cert.probes(n_each, g); sets["generative"] = torch.cat(list(P.values()))
        for kind in ("boundary_shift", "rare_region", "local_bump", "global"):
            f = damaged(ref, kind, r, g)
            for name, S in sets.items():
                res[name][kind] += float((f(S) != ref(S)).float().mean()) > 0.02
        for name, S in sets.items(): res[name]["false_alarm"] += float((ref(S) != ref(S)).float().mean()) > 0.02
        for name, S in sets.items():
            res[name]["n"] += 1
            res[name]["bytes"] += (0 if name == "generative" else (S.numel() * FLOAT_BYTES if name == "stored" else 8)) if name != "generative" else gen.memory_bytes()
    return {k: {kk: (round(v / c["n"], 3) if kk != "n" else v) for kk, v in c.items()} for k, c in res.items()}

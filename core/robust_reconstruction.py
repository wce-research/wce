"""RECONSTRUCTION STRUCTURELLE ROBUSTE & INFÉRENCE CONSCIENTE DE LA MESURE.

ÉTAT DU MONDE → OPÉRATEUR DE MESURE → OBSERVATION BRUITÉE → VRAISEMBLANCE STRUCTURELLE → STRUCTURES COMPATIBLES →
POSTÉRIEURE → MESURE ACTIVE → RECONSTRUCTION → VALIDATION → PROMOTION.

Séparation explicite : structure (squelette) | paramètres θ | processus de mesure (capteur, résolution, temps) |
bruit (retournements) | observations aberrantes (contamination) | incertitude résiduelle | erreur réelle de modèle.
Une mauvaise observation ne complexifie PAS automatiquement la structure : l'approfondissement exige un facteur
de Bayes (évidence approchée) en faveur du modèle plus riche APRÈS prise en compte du bruit.

Les quantités « évidence » sont des APPROXIMATIONS (BIC / Laplace grossière + a priori structurel documenté) ;
elles ne sont pas présentées comme une vérité bayésienne exacte. Aucune famille n'est nommée ici.
Inspirations (Dirac : état ≠ observable ≠ opérateur ≠ issue ; Shannon : IG ; Boltzmann : ensemble compatible ;
Einstein : repères ; Fourier : bases ; Heisenberg : compromis de capteurs GÉNÉRIQUES) — inspirations seulement,
aucune mécanique quantique.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from core.boundary_program import BNode, _inits, atoms, complexity, evaluate, predict, render, structure_hash

COMBINATORS = ("and", "or", "xor")
N_PRIMITIVE_TYPES = 10


# ================================================================ A — modèle de mesure

@dataclass(frozen=True)
class MeasurementModel:
    """Décrit COMMENT une observation est produite (jamais l'état latent lui-même)."""
    observable: str = "membership"
    source: str = "standard"
    frame: str = "world"
    observation_time: float = 0.0
    ingestion_time: float = 0.0
    resolution: float = 0.05             # flou de frontière (échelle du lissage de la vraisemblance)
    flip: float = 0.05                   # bruit connu ou estimé (retournement)
    contamination: float = 0.02          # probabilité d'observation aberrante (y ~ Bernoulli(½))
    missingness: float = 0.0
    cost: float = 1.0
    repeatable: bool = True
    correlation_group: Optional[str] = None


SENSOR_MODELS = {"precise": MeasurementModel(source="precise", flip=0.01, resolution=0.02, cost=3.0),
                 "standard": MeasurementModel(source="standard", flip=0.05, resolution=0.05, cost=1.0),
                 "cheap": MeasurementModel(source="cheap", flip=0.15, resolution=0.15, cost=0.3)}


@dataclass
class Observations:
    X: torch.Tensor
    y: torch.Tensor                       # −1 = manquant
    flip: torch.Tensor                    # bruit par observation (capteur ⊕ monde)
    resolution: torch.Tensor
    weight: torch.Tensor                  # 1 ; réduit pour sources corrélées / observations anciennes
    source: List[str] = field(default_factory=list)

    @staticmethod
    def build(X, y, model: MeasurementModel, world_noise=0.0, weight=None):
        n = len(X); f = min(0.45, model.flip + world_noise)
        return Observations(X, y, torch.full((n,), f), torch.full((n,), model.resolution), torch.ones(n) if weight is None else weight, [model.source] * n)

    def add(self, x, y, model: MeasurementModel, world_noise=0.0):
        f = min(0.45, model.flip + world_noise)
        return Observations(torch.cat([self.X, x]), torch.cat([self.y, y]), torch.cat([self.flip, torch.full((len(x),), f)]),
                            torch.cat([self.resolution, torch.full((len(x),), model.resolution)]), torch.cat([self.weight, torch.ones(len(x))]), self.source + [model.source] * len(x))

    def valid(self): return self.y >= 0

    def __len__(self): return int(self.valid().sum())


def temporal_weights(obs_time, query_time, half_life=None):
    """AD — une observation ANCIENNE n'est pas une mesure du présent : poids exp(−Δt / demi-vie) (∞ ⇒ 1)."""
    if half_life is None: return torch.ones_like(obs_time)
    return torch.exp(-(query_time - obs_time).clamp_min(0) * math.log(2) / half_life)


def correlated_weights(groups: Sequence[Optional[str]]):
    """AC — observations d'une même source CORRÉLÉE : poids total d'un groupe = 1 + log(taille) (pas de cumul naïf)."""
    c = Counter(g for g in groups if g is not None); w = []
    for g in groups: w.append(1.0 if g is None else (1 + math.log(c[g])) / c[g])
    return torch.tensor(w)


# ================================================================ B/C — vraisemblances

LIKELIHOODS = ("naive", "flip", "contamination", "soft_margin")


def membership_prob(prog: BNode, X, theta=None, k=12.0):
    return evaluate(prog if theta is None else prog.skeleton(), torch.nan_to_num(X), theta, k).clamp(1e-6, 1 - 1e-6)


def obs_loglik(prog: BNode, obs: Observations, kind="contamination", theta=None, gamma=0.02, per_point=False):
    """log P(D | structure, θ, modèle de mesure). naive : bruit quasi nul (1e-3), sortie dure ;
    flip : Bernoulli avec bruit du capteur ; contamination : mélange (1−γ)·flip + γ·½ ; soft_margin : flip + γ +
    lissage de frontière à la résolution du capteur (tolérance près de la frontière)."""
    v = obs.valid(); X, y, f, r, w = obs.X[v], obs.y[v].double(), obs.flip[v].double(), obs.resolution[v], obs.weight[v].double()
    if kind == "naive":
        p = (evaluate(prog if theta is None else prog.skeleton(), torch.nan_to_num(X), theta, None if theta is None else 30.0)).double()
        p1 = 1e-3 + (1 - 2e-3) * p
    else:
        k = 30.0 if kind != "soft_margin" else float(1.0 / max(1e-3, float(r.mean()) * 1.5))
        p = membership_prob(prog, X, theta, k).double()
        p1 = f + (1 - 2 * f) * p
        if kind in ("contamination", "soft_margin"): p1 = (1 - gamma) * p1 + gamma * 0.5
    ll = w * (y * torch.log(p1) + (1 - y) * torch.log(1 - p1))
    return ll if per_point else ll.sum()


def outlier_posterior(prog: BNode, obs: Observations, gamma=0.02):
    """L — P(observation aberrante | données, modèle) pour chaque observation (mélange de contamination)."""
    v = obs.valid(); p = membership_prob(prog, obs.X[v], k=30.0).double(); f = obs.flip[v].double(); y = obs.y[v].double()
    p1 = f + (1 - 2 * f) * p; l_in = torch.where(y.bool(), p1, 1 - p1) * (1 - gamma); l_out = torch.full_like(l_in, 0.5 * gamma)
    return l_out / (l_in + l_out)


def fit_robust(skel: BNode, obs: Observations, kind="contamination", gamma=0.02, steps=120, n_init=2, init=None, seed=0):
    """Ajustement des paramètres θ par MAXIMUM de vraisemblance robuste (structure fixée). Renvoie (programme, logL)."""
    v = obs.valid(); X, y = obs.X[v], obs.y[v]; g = torch.Generator().manual_seed(seed); best = None
    inits = [torch.tensor(init)] if init is not None else []
    if skel.n_params() == 0: p = skel.with_params([]); return p, float(obs_loglik(p, obs, kind, gamma=gamma))
    inits += list(_inits(skel, X, y, n_init, g))
    for th0 in inits:
        th = th0.clone().double().requires_grad_(True); opt = torch.optim.Adam([th], lr=0.05)
        for t in range(steps):
            opt.zero_grad(); kk = 4.0 + 20.0 * t / steps
            p = evaluate(skel, torch.nan_to_num(X).double(), th, kk).clamp(1e-6, 1 - 1e-6)
            f = obs.flip[v].double(); p1 = f + (1 - 2 * f) * p if kind != "naive" else 1e-3 + (1 - 2e-3) * p
            if kind in ("contamination", "soft_margin"): p1 = (1 - gamma) * p1 + gamma * 0.5
            ww = obs.weight[v].double(); yy = y.double()
            (-(ww * (yy * torch.log(p1) + (1 - yy) * torch.log(1 - p1))).sum()).backward(); opt.step()
        prog = skel.with_params(th.detach().float().tolist()); ll = float(obs_loglik(prog, obs, kind, gamma=gamma))
        if best is None or ll > best[1]: best = (prog, ll)
    return best


# ================================================================ E/F — évidence approchée et MDL

def log_evidence(prog: BNode, obs: Observations, ll: float):
    """Évidence APPROCHÉE (BIC/Laplace grossière + a priori structurel) :
    log p(D|S) ≈ logL(θ̂) − ½·k·log n − bits_structure·log 2. Ce n'est PAS une vraie marginale ; c'est un critère
    qui pénalise explicitement la flexibilité (paramètres) ET la taille de l'AST."""
    n = max(1, len(obs)); k = prog.n_params()
    return ll - 0.5 * k * math.log(n) - structure_bits(prog) * math.log(2)


def structure_bits(prog: BNode): return prog.size() * math.log2(N_PRIMITIVE_TYPES)


def mdl(prog: BNode, obs: Observations, kind="contamination", gamma=0.02):
    """Longueur de description (bits) = structure + paramètres + résidus/erreurs de MESURE (−log₂ P(D|θ, mesure)).
    kind="naive" : erreurs codées comme EXCEPTIONS (log₂ C(n, e)) — sans modèle de mesure explicite."""
    n = max(1, len(obs)); k = prog.n_params(); pb = 0.5 * k * math.log2(n)
    if kind == "naive":
        v = obs.valid(); e = int((predict(prog, obs.X[v]) != obs.y[v]).sum())
        rb = (math.lgamma(n + 1) - math.lgamma(e + 1) - math.lgamma(n - e + 1)) / math.log(2) + math.log2(n + 1)
        return {"structure_bits": structure_bits(prog), "parameter_bits": pb, "residual_bits": rb, "measurement_error_bits": 0.0, "total": structure_bits(prog) + pb + rb}
    mb = -float(obs_loglik(prog, obs, kind, gamma=gamma)) / math.log(2)
    return {"structure_bits": structure_bits(prog), "parameter_bits": pb, "residual_bits": 0.0, "measurement_error_bits": mb, "total": structure_bits(prog) + pb + mb}


def bic(prog, obs, ll): return -2 * ll + prog.n_params() * math.log(max(1, len(obs)))
def aic(prog, obs, ll): return -2 * ll + 2 * prog.n_params()


# ================================================================ G — ensemble de structures compatibles

@dataclass
class StructureEntry:
    program: BNode
    loglik: float
    evidence: float
    mass: float = 0.0
    stability: str = "UNTESTED"
    variant: int = 0                      # >0 : échantillon de la postérieure PARAMÉTRIQUE de la même structure

    @property
    def complexity(self): return complexity(self.program)


class CompatibleStructureSet:
    """Plusieurs structures restent compatibles ; UNKNOWN est une sortie valide."""

    def __init__(self, entries: Sequence[StructureEntry] = (), unknown_threshold=0.6):
        self.entries: List[StructureEntry] = []; self.unknown_threshold = unknown_threshold
        for e in entries: self.add(e)

    def add(self, e: StructureEntry):
        h = structure_hash(e.program) + f"#{e.variant}"
        for i, f in enumerate(self.entries):
            if structure_hash(f.program) + f"#{f.variant}" == h:
                if e.evidence > f.evidence: self.entries[i] = e
                self._norm(); return
        self.entries.append(e); self._norm()

    def _norm(self):
        if not self.entries: return
        ev = torch.tensor([e.evidence for e in self.entries], dtype=torch.float64); m = torch.softmax(ev, 0)
        for e, v in zip(self.entries, m.tolist()): e.mass = v

    def masses(self): return torch.tensor([e.mass for e in self.entries], dtype=torch.float64)

    def best(self): return max((e for e in self.entries if e.variant == 0), key=lambda e: e.mass, default=None)

    def structure_masses(self):
        out = Counter()
        for e in self.entries: out[structure_hash(e.program)] += e.mass
        return out

    def answer(self):
        b = self.best()
        if b is None: return "UNKNOWN"
        return b.program if self.structure_masses()[structure_hash(b.program)] >= self.unknown_threshold else "UNKNOWN"

    def preds(self, X): return torch.stack([predict(e.program, X) for e in self.entries]).double()

    def predictive(self, X): return (self.masses()[:, None] * self.preds(X)).sum(0)

    def entropy(self):
        m = self.masses(); m = m[m > 1e-12]; return float(-(m * m.log()).sum())

    def functional_entropy(self, probes):
        p = self.predictive(probes).clamp(1e-9, 1 - 1e-9); return float((-(p * p.log() + (1 - p) * (1 - p).log())).mean())

    def expected_accuracy(self, probes):
        """Exactitude ATTENDUE du prédicteur de Bayes sous la postérieure (sondes fonctionnelles)."""
        p = self.predictive(probes); return float(torch.maximum(p, 1 - p).mean())

    def rescore(self, obs: Observations, kind="contamination", gamma=0.02):
        for e in self.entries:
            e.loglik = float(obs_loglik(e.program, obs, kind, gamma=gamma)); e.evidence = log_evidence(e.program, obs, e.loglik)
        self._norm(); return self

    def prune(self, max_entries=16, min_mass=1e-4):
        self.entries = sorted([e for e in self.entries if e.mass >= min_mass], key=lambda e: -e.mass)[:max_entries]; self._norm()

    def add_parameter_samples(self, obs, kind, gamma, top=2, n=3, g=None):
        """D — incertitude paramétrique : échantillons de Laplace des meilleures structures (même structure, variants)."""
        self.entries = [e for e in self.entries if e.variant == 0]; g = g or torch.Generator().manual_seed(0)
        for e in sorted(self.entries, key=lambda e: -e.mass)[:top]:
            if e.program.n_params() == 0: continue
            pp = ParameterPosterior(e.program, obs, kind)
            for j, prog in enumerate(pp.sample(n, g)):
                ll = float(obs_loglik(prog, obs, kind, gamma=gamma)); self.add(StructureEntry(prog, ll, log_evidence(prog, obs, ll), variant=j + 1))
        self._norm()


# ================================================================ D — postérieure paramétrique (Laplace diagonale), postérieure de mesure

class ParameterPosterior:
    """P(θ | structure, D) ≈ N(θ̂, diag(1 / courbure)) — courbure par différences finies du log-vraisemblance."""

    def __init__(self, prog: BNode, obs: Observations, kind="contamination", h=1e-2):
        self.prog = prog; th = torch.tensor(prog.param_vector(), dtype=torch.float64); sd = []
        f = lambda t: float(obs_loglik(prog.skeleton().with_params(t.tolist()), obs, kind))
        f0 = f(th)
        for i in range(len(th)):
            e = torch.zeros_like(th); e[i] = h; c = -(f(th + e) - 2 * f0 + f(th - e)) / h ** 2; sd.append(1 / math.sqrt(max(c, 1e-3)))
        self.mean, self.sd = th, torch.tensor(sd)

    def sample(self, n, g):
        return [self.prog.skeleton().with_params((self.mean + self.sd * torch.randn(len(self.mean), generator=g, dtype=torch.float64)).tolist()) for _ in range(n)]


def estimate_noise(prog: BNode, obs: Observations, grid=(0.0, 0.02, 0.05, 0.08, 0.12, 0.15, 0.2, 0.25)):
    """MeasurementPosterior : bruit additionnel du MONDE estimé par maximum de vraisemblance sur une grille."""
    best = None
    for w in grid:
        o = Observations(obs.X, obs.y, (obs.flip + w).clamp(max=0.45), obs.resolution, obs.weight, obs.source)
        ll = float(obs_loglik(prog, o, "contamination"))
        if best is None or ll > best[1]: best = (w, ll)
    return best[0]


# ================================================================ L — attribution de l'écart

DISCREPANCY = ("MODEL_MISMATCH", "PARAMETER_UNCERTAINTY", "MEASUREMENT_NOISE", "OUTLIER", "NON_IDENTIFIABLE")


def attribute_discrepancy(prog: BNode, obs: Observations, gamma=0.02, cluster_radius=0.35):
    """Classe les erreurs d'ajustement : isolées et P(outlier) élevé → OUTLIER ; taux compatible avec le bruit connu
    (binomial) → MEASUREMENT_NOISE ; concentrées près de la frontière → PARAMETER_UNCERTAINTY ; groupées loin de la
    frontière en excès du bruit → MODEL_MISMATCH."""
    v = obs.valid(); X, y = obs.X[v], obs.y[v]; err = predict(prog, X) != y; n = len(X); e = int(err.sum())
    if e == 0: return {"label": "MEASUREMENT_NOISE", "errors": 0}
    exp_rate = float(obs.flip[v].mean()) + gamma; sd = math.sqrt(exp_rate * (1 - exp_rate) * n)
    margin = (membership_prob(prog, X[err], k=4.0) - 0.5).abs(); near = float((margin < 0.3).float().mean())
    E = X[err]; D = torch.cdist(E, E) + torch.eye(len(E)) * 9; clustered = float((D.min(1).values < cluster_radius).float().mean()) if len(E) > 1 else 0.0
    po = outlier_posterior(prog, obs, gamma)[err]
    if e <= exp_rate * n + 2 * sd:
        lab = "OUTLIER" if (e <= 2 and float(po.mean()) > 0.5) else "MEASUREMENT_NOISE"
    elif near > 0.7: lab = "PARAMETER_UNCERTAINTY"
    elif clustered > 0.5: lab = "MODEL_MISMATCH"
    else: lab = "MEASUREMENT_NOISE" if e <= exp_rate * n + 3 * sd else "MODEL_MISMATCH"
    return {"label": lab, "errors": e, "expected": round(exp_rate * n, 2), "near_boundary": round(near, 3), "clustered": round(clustered, 3)}


# ================================================================ H/I — gains attendus

def _reweight(masses, Pq, flip):
    """Pour chaque point candidat q et issue y ∈ {0, 1} : masses a posteriori (vectorisé) [Q, 2, K]."""
    l1 = flip + (1 - 2 * flip) * Pq                                       # [K, Q]  P(y=1 | structure k, q)
    L = torch.stack([1 - l1, l1], 0).permute(2, 0, 1)                    # [Q, 2, K]
    joint = L * masses[None, None, :]; py = joint.sum(2)                 # [Q, 2]
    return joint / py[:, :, None].clamp_min(1e-300), py


def expected_functional_accuracy_gain(css: CompatibleStructureSet, cands, probes, flip=0.05):
    """EFA(q) = E_y[ExactitudeAttendue(postérieure | q, y)] − ExactitudeAttendue(postérieure)."""
    m = css.masses(); Pq = css.preds(cands); Pp = css.preds(probes)        # [K, Q], [K, P]
    post, py = _reweight(m, Pq, flip)
    pred = post @ Pp                                                      # [Q, 2, P]
    acc = torch.maximum(pred, 1 - pred).mean(2)                           # [Q, 2]
    return (py * acc).sum(1) - css.expected_accuracy(probes)


def structural_ig(css, cands, flip=0.05):
    m = css.masses(); post, py = _reweight(m, css.preds(cands), flip)
    H = lambda p: -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
    return H(m[None, :]) - (py * H(post)).sum(1)


def functional_ig(css, cands, probes, flip=0.05):
    m = css.masses(); post, py = _reweight(m, css.preds(cands), flip); Pp = css.preds(probes)
    pred = (post @ Pp).clamp(1e-9, 1 - 1e-9); Hf = (-(pred * pred.log() + (1 - pred) * (1 - pred).log())).mean(2)
    return css.functional_entropy(probes) - (py * Hf).sum(1)


def disagreement(css, cands): p = css.predictive(cands); return 1 - (2 * p - 1).abs()
def uncertainty(css, cands): p = css.predictive(cands).clamp(1e-9, 1 - 1e-9); return -(p * p.log() + (1 - p) * (1 - p).log())


def expected_decision_loss_utility(efa, cost, compute=0.0, noise_risk=0.0, disturbance=0.0, value=1.0, lam=0.01):
    """I — utilité = valeur·réduction d'erreur fonctionnelle − λ·(coût de mesure + calcul + risque de bruit + perturbation)."""
    return value * efa - lam * (cost + compute + noise_risk + disturbance)


# ================================================================ J/K — planificateur de mesures

ACTIONS = ("new_point", "repeat", "near_boundary", "far_from_boundary", "discriminate", "parameter", "validate", "deepen", "stop")


@dataclass
class MeasurementChoice:
    action: str
    x: Optional[torch.Tensor]
    sensor: str
    utility: float
    predicted_gain: float


class RobustMeasurementPlanner:
    """Planificateur DISTINCT du SearchPolicy : choisit (point, capteur) ou répétition, approfondissement, arrêt,
    selon la postérieure. criterion ∈ {efa, structural_ig, functional_ig, disagreement, uncertainty, random}."""

    def __init__(self, criterion="efa", sensors=("standard",), lam=0.01, allow_repeat=True, world_noise=0.0, seed=0):
        self.criterion, self.sensors, self.lam, self.allow_repeat, self.world_noise = criterion, sensors, lam, allow_repeat, world_noise
        self.g = torch.Generator().manual_seed(seed)

    def scores(self, css, cands, probes, flip):
        c = self.criterion
        if c == "efa": return expected_functional_accuracy_gain(css, cands, probes, flip)
        if c == "structural_ig": return structural_ig(css, cands, flip)
        if c == "functional_ig": return functional_ig(css, cands, probes, flip)
        if c == "disagreement": return disagreement(css, cands)
        if c == "uncertainty": return uncertainty(css, cands)
        return torch.rand(len(cands), generator=self.g, dtype=torch.float64)

    def choose(self, css, pool, history, probes, budget_left):
        best = None
        for s in self.sensors:
            sm = SENSOR_MODELS[s]
            if sm.cost > budget_left: continue
            flip = min(0.45, sm.flip + self.world_noise)
            for kind, C in (("new_point", pool), ("repeat", history)):
                if kind == "repeat" and (not self.allow_repeat or history is None or len(history) == 0): continue
                sc = self.scores(css, C, probes, flip); i = int(sc.argmax()); gain = float(sc[i])
                u = gain - self.lam * sm.cost if self.criterion in ("efa", "structural_ig", "functional_ig") else gain
                if best is None or u > best.utility: best = MeasurementChoice(kind, C[i:i + 1], s, u, gain)
        if best is None or (self.criterion == "efa" and best.predicted_gain <= 1e-5): return MeasurementChoice("stop", None, "", 0.0, 0.0)
        return best


# ================================================================ M/N — recherche structurelle robuste

def expand(parent: BNode, A, combs=COMBINATORS):
    return [(c, a) for c in combs for a in A]


@dataclass
class ReconstructionResult:
    answer: object
    css: CompatibleStructureSet
    queries: int
    cost: float
    evaluations: int
    seconds: float
    depth: int
    trace: List[dict] = field(default_factory=list)
    discrepancy: dict = field(default_factory=dict)
    status: str = ""


class RobustStructuralReconstruction:
    """Maintient plusieurs structures plausibles, ajuste θ robustement, estime le bruit, choisit mesure / répétition /
    approfondissement / arrêt, ne complexifie que si le facteur de Bayes l'exige APRÈS prise en compte du bruit,
    simplifie, et ne promeut qu'après contrôle prédictif a posteriori.
    likelihood ∈ LIKELIHOODS ; depth_rule ∈ {bayes_factor, threshold} ; planner : RobustMeasurementPlanner | None."""

    def __init__(self, likelihood="contamination", depth_rule="bayes_factor", planner: Optional[RobustMeasurementPlanner] = None,
                 budget=16.0, max_depth=4, beam=3, gamma=0.02, bf_threshold=3.0, policy=None, seed=0, estimate_world_noise=True):
        self.kind, self.depth_rule, self.planner, self.budget, self.max_depth, self.beam = likelihood, depth_rule, planner, budget, max_depth, beam
        self.gamma, self.bf, self.policy, self.seed, self.est = gamma, bf_threshold, policy, seed, estimate_world_noise

    def _fit(self, skel, obs, init=None):
        return fit_robust(skel, obs, self.kind, self.gamma, steps=100, n_init=1 if init is not None else 2, init=init, seed=self.seed)

    def _entry(self, prog, ll, obs): return StructureEntry(prog, ll, log_evidence(prog, obs, ll))

    def run(self, obs: Observations, oracle: Optional[Callable] = None, pool_sampler: Optional[Callable] = None, probes=None, g=None, monitor=None):
        """monitor(css) -> exactitude RÉELLE (évaluation seulement, jamais utilisée pour décider) : sert à mesurer le
        gain fonctionnel réalisé de chaque requête (H10)."""
        t0 = time.time(); evals = 0; trace = []; A = atoms(); css = CompatibleStructureSet()
        for a in A:
            p, ll = self._fit(a, obs); evals += 1; css.add(self._entry(p, ll, obs))
        depth = 1; queries = 0; spent = 0.0
        pool = pool_sampler(400, g) if pool_sampler else None; hist: List[torch.Tensor] = []
        forced = self.planner is not None                                          # alternatives plausibles (profondeur 2) pour départager activement
        while True:
            best = css.best()
            deepen = depth < self.max_depth and (forced or self._should_deepen(css, obs)); forced = False
            if deepen:
                depth += 1; parents = sorted(css.entries, key=lambda e: -e.mass)[: self.beam]; gained = False
                for par in parents:
                    exps = expand(par.program, A)
                    if self.policy is not None: exps = self.policy.propose(par.program, A, depth, True)
                    for comb, a in exps:
                        skel = BNode(comb, children=(par.program.skeleton(), a)); init = par.program.param_vector() + list(_inits(a, obs.X[obs.valid()], obs.y[obs.valid()], 1, torch.Generator().manual_seed(evals))[0].tolist())
                        p, ll = self._fit(skel, obs, init); evals += 1; e = self._entry(p, ll, obs)
                        gained |= e.evidence > par.evidence + math.log(self.bf); css.add(e)
                css.prune(); trace.append({"action": "deepen", "depth": depth, "gained": gained}); continue
            if self.planner is None or oracle is None or spent >= self.budget: break
            css.add_parameter_samples(obs, self.kind, self.gamma, g=torch.Generator().manual_seed(queries))
            ch = self.planner.choose(css, pool, torch.stack(hist) if hist else None, probes, self.budget - spent)
            if ch.action == "stop": trace.append({"action": "stop"}); break
            sm = SENSOR_MODELS[ch.sensor]; y = oracle(ch.x, ch.sensor, g)
            acc_before = _acc_vs(css, probes)
            if monitor is not None and not trace or (monitor is not None and "true_acc_after" not in trace[-1]): trace.append({"action": "baseline", "true_acc_after": monitor(css)})
            obs = obs.add(ch.x, y, sm, self.planner.world_noise); queries += 1; spent += sm.cost; hist.append(ch.x[0])
            if queries % 4 == 0:
                for e in sorted(css.entries, key=lambda e: -e.mass)[:4]:
                    p, ll = self._fit(e.program.skeleton(), obs, e.program.param_vector()); evals += 1; css.add(self._entry(p, ll, obs))
            css.rescore(obs, self.kind, self.gamma)
            rec = {"action": ch.action, "sensor": ch.sensor, "predicted_gain": ch.predicted_gain, "acc_before": acc_before}
            if monitor is not None: rec["true_acc_after"] = monitor(css)
            trace.append(rec)
        # simplification et noise estimation
        css.entries = [e for e in css.entries if e.variant == 0]; css._norm()
        best = css.best(); simp = simplify_robust(best.program, obs, self.kind, self.gamma); evals += 1
        if simp is not best.program:
            p, ll = self._fit(simp.skeleton(), obs, simp.param_vector()); css.add(self._entry(p, ll, obs)); css.rescore(obs, self.kind, self.gamma)
        best = css.best(); disc = attribute_discrepancy(best.program, obs, self.gamma)
        ans = css.answer(); status = "identified" if ans != "UNKNOWN" and disc["label"] != "MODEL_MISMATCH" else ("model_mismatch" if disc["label"] == "MODEL_MISMATCH" else "unknown")
        return ReconstructionResult(ans, css, queries, spent, evals, round(time.time() - t0, 2), depth, trace, disc, status)

    def _should_deepen(self, css, obs):
        """N — critère : seuil d'erreur (ancien) OU comparaison modèle simple + bruit vs complexe (facteur de Bayes)."""
        best = css.best(); v = obs.valid(); err = float((predict(best.program, obs.X[v]) != obs.y[v]).float().mean())
        noise = float(obs.flip[v].mean())
        if self.depth_rule == "threshold": return err > noise + 0.03
        # le meilleur modèle courant est-il INCOMPATIBLE avec son bruit (test binomial) ? sinon, ne pas approfondir
        n = int(v.sum()); e = err * n; exp = (noise + self.gamma) * n; sd = math.sqrt(max(exp * (1 - (noise + self.gamma)), 1e-9))
        return e > exp + 2.0 * sd


def _acc_vs(css, probes):
    return css.expected_accuracy(probes) if probes is not None else None


def simplify_robust(prog: BNode, obs: Observations, kind="contamination", gamma=0.02, bf=1.0):
    """Remplace un combinateur par l'un de ses enfants si l'évidence approchée ne baisse pas (structure simple + bruit
    plausible préférée à structure complexe + ajustement parfait)."""
    from core.active_structure import _subtrees_replacements
    cur = prog; ev = log_evidence(cur, obs, float(obs_loglik(cur, obs, kind, gamma=gamma)))
    while True:
        best = None
        for r in _subtrees_replacements(cur):
            e = log_evidence(r, obs, float(obs_loglik(r, obs, kind, gamma=gamma)))
            if e >= ev - math.log(bf) and (best is None or e > best[1]): best = (r, e)
        if best is None: return cur
        cur, ev = best


# ================================================================ O — calibration de P(règle fausse)

def platt(scores, labels, steps=300):
    a = torch.zeros(2, requires_grad=True); s = torch.tensor(scores, dtype=torch.float32); y = torch.tensor(labels, dtype=torch.float32)
    opt = torch.optim.Adam([a], lr=0.05)
    for _ in range(steps): opt.zero_grad(); torch.nn.functional.binary_cross_entropy_with_logits(a[0] * s + a[1], y).backward(); opt.step()
    w = a.detach(); return lambda x: torch.sigmoid(w[0] * torch.tensor(x, dtype=torch.float32) + w[1]).tolist()


def beta_calibration(scores, labels, steps=400):
    """Calibration bêta : logit(p) = a·ln s − b·ln(1−s) + c."""
    s = torch.tensor(scores, dtype=torch.float32).clamp(1e-4, 1 - 1e-4); y = torch.tensor(labels, dtype=torch.float32)
    w = torch.tensor([1.0, 1.0, 0.0], requires_grad=True); opt = torch.optim.Adam([w], lr=0.05)
    f = lambda s: w[0] * torch.log(s) - w[1] * torch.log(1 - s) + w[2]
    for _ in range(steps): opt.zero_grad(); torch.nn.functional.binary_cross_entropy_with_logits(f(s), y).backward(); opt.step()
    ww = w.detach()
    return lambda x: torch.sigmoid(ww[0] * torch.log(torch.tensor(x).clamp(1e-4, 1 - 1e-4)) - ww[1] * torch.log(1 - torch.tensor(x).clamp(1e-4, 1 - 1e-4)) + ww[2]).tolist()


def isotonic(scores, labels):
    from core.unified import IsotonicCalibrator
    c = IsotonicCalibrator(prior_strength=2.0, bins=20).fit(list(scores), [bool(v) for v in labels]); return lambda x: [float(c(v)) for v in x]


def calibration_metrics(p, y, bins=10):
    p = [float(v) for v in p]; y = [float(v) for v in y]; n = len(p); ece = 0.0; curve = []
    for b in range(bins):
        idx = [i for i, v in enumerate(p) if b / bins <= v < (b + 1) / bins or (b == bins - 1 and v == 1.0)]
        if idx:
            mp, my = sum(p[i] for i in idx) / len(idx), sum(y[i] for i in idx) / len(idx); ece += len(idx) / n * abs(mp - my); curve.append((round(mp, 3), round(my, 3), len(idx)))
    brier = sum((a - b) ** 2 for a, b in zip(p, y)) / max(1, n)
    def pr(t):
        pred = [v >= t for v in p]; tp = sum(1 for a, b in zip(pred, y) if a and b); fp = sum(1 for a, b in zip(pred, y) if a and not b); fn = sum(1 for a, b in zip(pred, y) if not a and b)
        return {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}
    return {"ece": round(ece, 4), "brier": round(brier, 4), "reliability": curve, "at_0.5": pr(0.5), "at_0.9": pr(0.9)}


# ================================================================ P — contrôle prédictif a posteriori

def posterior_predictive_check(prog: BNode, obs_new: Observations, n_sim=400, g=None, gamma=0.02):
    """Statistiques réelles sur de NOUVELLES observations vs distribution prédictive simulée (bruit compris) :
    nombre d'erreurs et nombre de paires d'erreurs voisines (structure spatiale). p-valeurs unilatérales."""
    g = g or torch.Generator().manual_seed(0); v = obs_new.valid(); X, y = obs_new.X[v], obs_new.y[v]
    p = membership_prob(prog, X, k=30.0).double(); f = obs_new.flip[v].double(); p1 = (1 - gamma) * (f + (1 - 2 * f) * p) + gamma * 0.5
    pred = predict(prog, X)
    def stats(yy):
        e = (yy != pred); E = X[e]
        pairs = int(((torch.cdist(E, E) < 0.35).sum() - len(E)) // 2) if len(E) > 1 else 0
        return int(e.sum()), pairs
    obs_e, obs_pairs = stats(y); sims = [stats((torch.rand(len(X), generator=g, dtype=torch.float64) < p1).long()) for _ in range(n_sim)]
    p_err = sum(s[0] >= obs_e for s in sims) / n_sim; p_pairs = sum(s[1] >= obs_pairs for s in sims) / n_sim
    return {"errors": obs_e, "pairs": obs_pairs, "p_errors": p_err, "p_pairs": p_pairs, "reject": min(p_err, p_pairs) < 0.01}


# ================================================================ T/U/V/W/X — couches de mesure inspirées

def fourier_reconstruction(signal, k=None, energy=0.99):
    """T — représentation : signal compliqué en espace direct → peu de raies en fréquence ; reconstruction inverse.
    Changement de BASE (représentation), jamais du monde."""
    from core.measurement import BasisTransform
    bt = BasisTransform(len(signal)); F = bt.forward(signal); mag = F.abs()
    order = mag.argsort(descending=True); cum = torch.cumsum(mag[order] ** 2, 0) / (mag ** 2).sum()
    k = k or int((cum < energy).sum()) + 1; keep = torch.zeros_like(F); keep[order[:k]] = F[order[:k]]
    rec = bt.inverse(keep).real if torch.is_complex(bt.inverse(keep)) else bt.inverse(keep)
    return {"k": k, "n": len(signal), "rel_error": float((rec - signal).norm() / signal.norm()), "reconstruction": rec}


def microstate_entropy(css: CompatibleStructureSet): return css.entropy()


def frame_transform(X, angle=0.0, shift=(0.0, 0.0)):
    """V — repère : une mesure spatiale porte son repère ; X_monde = R(angle)·X_repère + décalage."""
    c, s = math.cos(angle), math.sin(angle); R = torch.tensor([[c, -s], [s, c]], dtype=X.dtype)
    return X @ R.T + torch.tensor(shift, dtype=X.dtype)


def sensor_tradeoff_choice(css, pool, probes, budget_left, world_noise=0.0):
    """W — compromis GÉNÉRIQUES (résolution ↔ coût / bruit) : le planificateur choisit le capteur dont l'utilité
    (gain fonctionnel attendu − coût) est maximale. Aucune relation d'incertitude quantique."""
    return RobustMeasurementPlanner("efa", sensors=tuple(SENSOR_MODELS), world_noise=world_noise).choose(css, pool, None, probes, budget_left)


@dataclass(frozen=True)
class MeasurementOutcome:
    """X — séparation stricte : l'issue observée n'est JAMAIS l'état latent."""
    value: int
    model: MeasurementModel

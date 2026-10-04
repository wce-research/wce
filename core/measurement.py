"""WCE — ALGÈBRE DE MESURE ET D'OBSERVATION.

    ÉTAT DU MONDE (latent) → OBSERVABLE → CONTEXTE DE MESURE → OPÉRATEUR → DISTRIBUTION DES ISSUES
    → OBSERVATION (issue tirée) → ENSEMBLE D'ÉTATS COMPATIBLES → MISE À JOUR DES CROYANCES

Ce qui EXISTE (LatentWorldState) ≠ ce qui est MESURÉ (opérateur + contexte) ≠ ce qui est OBSERVÉ (issue enregistrée)
≠ ce qui est CRU (CompatibleStateSet pondéré).

Inspirations — ANALOGIES ARCHITECTURALES, aucune physique revendiquée :
  * Dirac (analogie) : séparation ÉTAT / OBSERVABLE / OPÉRATEUR / ISSUE ; aucun espace de Hilbert, aucune amplitude.
  * Fourier : changement de BASE de représentation (DFT orthonormée, réelle) ; le monde ne change pas.
  * Einstein (analogie classique) : toute position / vitesse / orientation porte son RÉFÉRENTIEL ; transformations
    rigides 2D classiques (galiléennes), pas de relativité.
  * Boltzmann (analogie) : une macro-observation correspond à plusieurs micro-configurations ; multiplicité et
    entropie de l'ensemble compatible ; aucune thermodynamique.
  * Shannon : entropie et gain d'information (mathématiques exactes, pas une analogie).
  * Heisenberg (analogie) : COMPROMIS définis explicitement par le modèle de capteur ; aucune relation ΔxΔp ≥ ħ/2.
  * Bohm (analogie) : état latent ≠ issue observée ; ce n'est PAS de la mécanique bohmienne.
Tout est vectorisé (torch) et chaque opération est tracée comme TransformationSpec du domaine MEASUREMENT.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from core.transform_graph import TransformGraph, TransformNode
from core.transformations import TransformationDomain, TransformationSpec

PASSIVE, WEAKLY_DISTURBING, DESTRUCTIVE = "PASSIVE", "WEAKLY_DISTURBING", "DESTRUCTIVE"
CONFLICTED = "CONFLICTED"


class MeasurementInvariant(Enum):
    NORMALIZED_DISTRIBUTION = "normalized_distribution"
    REFERENCE_FRAME_VALID = "reference_frame_valid"
    UNIT_CONSISTENT = "unit_consistent"
    OUTCOME_COMPATIBLE = "outcome_compatible"
    PROVENANCE_PRESENT = "provenance_present"
    NO_HIDDEN_TRUTH_ACCESS = "no_hidden_truth_access"


class FrameError(ValueError):
    """Comparaison de valeurs exprimées dans des référentiels différents sans transformation."""


class UnitError(ValueError):
    """Opération entre grandeurs de dimensions / unités incompatibles."""


# ================================================================ référentiels (analogie : Einstein, classique 2D)

@dataclass(frozen=True)
class ReferenceFrame:
    """Référentiel 2D rigide : origine (dans le monde) + angle. Classique, pas relativiste."""
    name: str
    origin: Tuple[float, float] = (0.0, 0.0)
    angle: float = 0.0
    velocity: Tuple[float, float] = (0.0, 0.0)          # vitesse du référentiel dans le monde (galiléen)

    def _R(self):
        c, s = math.cos(self.angle), math.sin(self.angle)
        return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)

    def to_world(self, v: "FramedValue") -> "FramedValue":
        if v.frame != self.name: raise FrameError(f"valeur en {v.frame}, référentiel {self.name}")
        x = torch.as_tensor(v.value, dtype=torch.float64)
        if v.kind == "point": out = x @ self._R().T + torch.tensor(self.origin, dtype=torch.float64)
        elif v.kind == "velocity": out = x @ self._R().T + torch.tensor(self.velocity, dtype=torch.float64)
        elif v.kind in ("vector", "displacement"): out = x @ self._R().T
        elif v.kind == "orientation": out = x + self.angle
        else: out = x
        return FramedValue(out, WORLD_FRAME.name, v.kind, v.unit)

    def from_world(self, v: "FramedValue") -> "FramedValue":
        if v.frame != WORLD_FRAME.name: raise FrameError(f"valeur en {v.frame}, attendu {WORLD_FRAME.name}")
        x = torch.as_tensor(v.value, dtype=torch.float64)
        if v.kind == "point": out = (x - torch.tensor(self.origin, dtype=torch.float64)) @ self._R()
        elif v.kind == "velocity": out = (x - torch.tensor(self.velocity, dtype=torch.float64)) @ self._R()
        elif v.kind in ("vector", "displacement"): out = x @ self._R()
        elif v.kind == "orientation": out = x - self.angle
        else: out = x
        return FramedValue(out, self.name, v.kind, v.unit)


WORLD_FRAME = ReferenceFrame("world")


@dataclass(frozen=True)
class FramedValue:
    value: Any
    frame: str
    kind: str = "point"                   # point | vector | displacement | velocity | orientation | scalar
    unit: str = "m"

    def same_frame(self, other):
        if self.frame != other.frame: raise FrameError(f"comparaison {self.frame} vs {other.frame} sans transformation")
        if self.unit != other.unit: raise UnitError(f"unités {self.unit} vs {other.unit}")


def change_frame(v: FramedValue, frames: Dict[str, ReferenceFrame], target: str, graph=None) -> FramedValue:
    """Transforme v vers le référentiel cible (via le monde) ; tracé comme ChangeFrame."""
    w = v if v.frame == WORLD_FRAME.name else frames[v.frame].to_world(v)
    out = w if target == WORLD_FRAME.name else frames[target].from_world(w)
    trace(graph, MeasurementOp("ChangeFrame", {"from": v.frame, "to": target, "kind": v.kind}))
    return out


def distance(a: FramedValue, b: FramedValue) -> float:
    a.same_frame(b)
    return float(torch.linalg.norm(torch.as_tensor(a.value, dtype=torch.float64) - torch.as_tensor(b.value, dtype=torch.float64)))


# ================================================================ bases de représentation (Fourier)

@dataclass(frozen=True)
class RepresentationBasis:
    name: str                                 # time | frequency | raw | coefficients
    dimension: int
    unit: str = ""


class BasisTransform:
    """Changement de BASE (DFT orthonormée) : la représentation change, le monde (le signal) ne change pas."""

    def __init__(self, n, graph=None):
        self.time = RepresentationBasis("time", n, "s")
        self.frequency = RepresentationBasis("frequency", n, "Hz")
        self.graph = graph

    def forward(self, x):
        trace(self.graph, MeasurementOp("ChangeBasis", {"from": "time", "to": "frequency"}))
        return torch.fft.fft(torch.as_tensor(x, dtype=torch.float64), norm="ortho")

    def inverse(self, X):
        trace(self.graph, MeasurementOp("ChangeBasis", {"from": "frequency", "to": "time"}))
        return torch.fft.ifft(X, norm="ortho").real

    @staticmethod
    def energy(x): return float((torch.as_tensor(x).abs() ** 2).sum())

    def invariants(self, x):
        X = self.forward(x); xr = self.inverse(X)
        return {"invertible": float((xr - torch.as_tensor(x, dtype=torch.float64)).abs().max()) < 1e-9,
                "energy_preserved": abs(self.energy(x) - self.energy(X)) < 1e-6 * max(1.0, self.energy(x)),   # Parseval
                "dimension_consistent": X.shape[-1] == torch.as_tensor(x).shape[-1]}


# ================================================================ mesure : observable, contexte, opérateur

@dataclass(frozen=True)
class Observable:
    """CE QUI EST INTERROGÉ (pas sa valeur) : propriété d'une entité, domaine, unité, référentiel requis."""
    name: str
    entity: str
    attribute: str
    domain: Tuple[Any, ...] = ()             # discret ; vide = continu
    unit: str = ""
    frame_required: bool = False

    @property
    def discrete(self): return bool(self.domain)

    def __str__(self): return f"{self.attribute}({self.entity})"


@dataclass(frozen=True)
class NoiseModel:
    kind: str = "confusion"                   # confusion (discret, fiabilité r) | gaussian (continu, sigma)
    reliability: float = 0.9
    sigma: float = 0.0


@dataclass(frozen=True)
class MeasurementContext:
    observer: str = "agent"
    sensor: str = "sensor"
    frame: str = "world"
    time: int = 0
    resolution: float = 0.0                   # granularité de lecture (≠ incertitude)
    noise: NoiseModel = field(default_factory=NoiseModel)
    cost: float = 1.0
    latency: float = 0.0
    disturbance: str = PASSIVE
    disturbance_effect: Optional[Callable] = field(default=None, compare=False)
    domain: str = "world"
    reliability_interval: Tuple[float, float] = (0.0, 1.0)


@dataclass
class MeasurementDistribution:
    support: List[Any]
    probs: torch.Tensor

    def normalized(self, tol=1e-6): return abs(float(self.probs.sum()) - 1.0) < tol

    def credible_region(self, level=0.9):
        order = torch.argsort(self.probs, descending=True); acc, out = 0.0, []
        for i in order.tolist():
            out.append(self.support[i]); acc += float(self.probs[i])
            if acc >= level: break
        return out


@dataclass
class MeasurementResolution:
    """La résolution est la granularité de LECTURE (ex. « à 10 cm près ») : l'issue désigne un intervalle."""
    step: float

    def bin(self, x): return math.floor(x / self.step) * self.step if self.step else x

    def interval(self, reading): return (reading, reading + self.step) if self.step else (reading, reading)


@dataclass
class MeasurementOutcome:
    value: Any
    interval: Optional[Tuple[float, float]] = None


@dataclass
class MeasurementRecord:
    observable: Observable
    context: MeasurementContext
    outcome: MeasurementOutcome
    provenance: str
    distribution: Optional[MeasurementDistribution] = None


class MeasurementOperator:
    """ÉTAT → DISTRIBUTION SUR LES ISSUES (pas une valeur exacte). Vectorisé sur un lot d'états candidats."""

    def __init__(self, observable: Observable, context: MeasurementContext):
        self.observable, self.context = observable, context

    @property
    def name(self): return f"{self.context.sensor}:{self.observable}"

    def outcomes(self, grid=None):
        if self.observable.discrete: return list(self.observable.domain)
        return list(grid) if grid is not None else []

    def likelihood(self, outcome, values: torch.Tensor):
        """P(issue | valeur vraie) pour un VECTEUR de valeurs candidates (indices du domaine, ou réels)."""
        n = self.context.noise
        if self.observable.discrete:
            K = len(self.observable.domain); r = n.reliability
            o = self.observable.domain.index(outcome.value if isinstance(outcome, MeasurementOutcome) else outcome)
            return torch.where(values == o, torch.tensor(r, dtype=torch.float64), torch.tensor((1 - r) / max(K - 1, 1), dtype=torch.float64))
        reading = outcome.value if isinstance(outcome, MeasurementOutcome) else outcome
        step = self.context.resolution; sig = max(n.sigma, 1e-9)
        vals = values.to(torch.float64)
        if step:                                                   # issue = intervalle de lecture [reading, reading + step)
            nd = torch.distributions.Normal(vals, sig)
            return (nd.cdf(torch.tensor(reading + step, dtype=torch.float64)) - nd.cdf(torch.tensor(reading, dtype=torch.float64))).clamp_min(1e-12)
        return torch.exp(-0.5 * ((reading - vals) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))

    def distribution(self, true_value, grid=None) -> MeasurementDistribution:
        sup = self.outcomes(grid)
        if self.observable.discrete:
            idx = torch.tensor([self.observable.domain.index(true_value)])
            p = torch.stack([self.likelihood(o, idx)[0] for o in sup])
        else:
            p = torch.stack([self.likelihood(o, torch.tensor([float(true_value)]))[0] for o in sup])
        return MeasurementDistribution(sup, p / p.sum())

    def sample(self, true_value, rng: random.Random):
        n = self.context.noise
        if self.observable.discrete:
            if rng.random() < n.reliability: return MeasurementOutcome(true_value)
            return MeasurementOutcome(rng.choice([v for v in self.observable.domain if v != true_value]))
        x = true_value + rng.gauss(0.0, n.sigma) if n.sigma else true_value
        res = MeasurementResolution(self.context.resolution)
        r = res.bin(x)
        return MeasurementOutcome(r, res.interval(r))


# ================================================================ monde latent (analogie : Bohm — latent ≠ observé)

class LatentWorldState:
    """L'état qui EXISTE. L'agent n'y accède QUE par des opérateurs de mesure ; l'évaluation peut le lire
    (peek), et chaque lecture est comptée pour vérifier NO_HIDDEN_TRUTH_ACCESS."""

    def __init__(self, values: Dict[Tuple[str, str], Any], seed=0):
        self._values = dict(values); self.rng = random.Random(seed)
        self.agent_peeks = 0; self.evaluation_peeks = 0; self.log: List[MeasurementRecord] = []

    def measure(self, op: MeasurementOperator, provenance="capteur", graph=None) -> MeasurementRecord:
        key = (op.observable.entity, op.observable.attribute)
        out = op.sample(self._values[key], self.rng)
        rec = MeasurementRecord(op.observable, op.context, out, provenance)
        if op.context.disturbance != PASSIVE and op.context.disturbance_effect is not None:
            op.context.disturbance_effect(self)                 # une mesure perturbatrice change le monde latent
        self.log.append(rec)
        trace(graph, MeasurementOp("Measure", {"operator": op.name, "disturbance": op.context.disturbance}))
        return rec

    def peek(self, key, by="evaluation"):
        if by == "agent": self.agent_peeks += 1
        else: self.evaluation_peeks += 1
        return self._values[key]

    def set(self, key, value): self._values[key] = value


# ================================================================ ensemble d'états compatibles (analogie : Boltzmann)

class CompatibleStateSet:
    """Croyance = ensemble pondéré de mondes possibles {état_i, poids_i}. Les configurations [N, V] sont PARTAGÉES
    (copy-on-write : conditionner ne copie que les poids). Variables discrètes (indices dans leur domaine)."""

    def __init__(self, variables: List[Tuple[str, str]], domains: Dict[Tuple[str, str], Sequence[Any]],
                 states: torch.Tensor, logw: Optional[torch.Tensor] = None, graph=None):
        self.vars = list(variables); self.domains = {k: list(v) for k, v in domains.items()}
        self.index = {v: i for i, v in enumerate(self.vars)}
        self.states = states                                    # partagé
        self.logw = logw if logw is not None else torch.zeros(states.shape[0], dtype=torch.float64)
        self.graph = graph; self.history: List[str] = []

    @classmethod
    def product(cls, domains: Dict[Tuple[str, str], Sequence[Any]], constraint=None, graph=None):
        vars_ = list(domains)
        grids = torch.cartesian_prod(*[torch.arange(len(domains[v])) for v in vars_]) if len(vars_) > 1 else \
            torch.arange(len(domains[vars_[0]])).view(-1, 1)
        grids = grids.view(-1, len(vars_))
        s = cls(vars_, domains, grids, graph=graph)
        if constraint is not None:
            keep = constraint(s)
            s.states = s.states[keep]; s.logw = s.logw[keep]
        return s

    def _fork(self, logw, note):
        out = CompatibleStateSet(self.vars, self.domains, self.states, logw, self.graph)   # états partagés
        out.history = self.history + [note]
        return out

    @property
    def weights(self):
        w = torch.exp(self.logw - self.logw.max()); return w / w.sum()

    def __len__(self): return int(self.states.shape[0])

    def column(self, var): return self.states[:, self.index[var]]

    def value_index(self, var, value): return self.domains[var].index(value)

    # ---- opérations de l'algèbre (tracées) -------------------------------------------------
    def condition(self, record: MeasurementRecord, var=None, reliability=None) -> "CompatibleStateSet":
        """ConditionOnMeasurement : prior compatible + mesure → postérieure compatible."""
        var = var or (record.observable.entity, record.observable.attribute)
        op = MeasurementOperator(record.observable, record.context if reliability is None else
                                 MeasurementContext(**{**record.context.__dict__, "noise": NoiseModel(record.context.noise.kind, reliability, record.context.noise.sigma)}))
        col = self.column(var)
        if record.observable.discrete:
            L = op.likelihood(record.outcome, col)
        else:
            vals = torch.tensor([float(x) for x in self.domains[var]], dtype=torch.float64)[col]
            L = op.likelihood(record.outcome, vals)
        trace(self.graph, MeasurementOp("Condition", {"observable": str(record.observable), "provenance": record.provenance}))
        return self._fork(self.logw + torch.log(L.clamp_min(1e-300)), f"condition {record.observable} = {record.outcome.value}")

    def restrict(self, mask: torch.Tensor, note="macro-observation") -> "CompatibleStateSet":
        """Observation déterministe (macro) : seuls les états compatibles gardent un poids."""
        lw = torch.where(mask, self.logw, torch.full_like(self.logw, -float("inf")))
        trace(self.graph, MeasurementOp("Condition", {"kind": "restrict"}))
        return self._fork(lw, note)

    def marginal(self, var) -> Dict[Any, float]:
        """Marginalize : P(var = v) sans choisir une configuration complète."""
        w = self.weights; col = self.column(var)
        p = torch.zeros(len(self.domains[var]), dtype=torch.float64).index_add_(0, col, w)
        trace(self.graph, MeasurementOp("Marginalize", {"variable": str(var)}))
        return {v: float(p[i]) for i, v in enumerate(self.domains[var])}

    def entropy(self, var=None) -> float:
        """H(S) en nats (ou H de la marginale de var)."""
        if var is not None:
            p = torch.tensor(list(self.marginal(var).values()), dtype=torch.float64)
        else:
            p = self.weights
        p = p[p > 0]
        return float(-(p * p.log()).sum())

    def support_size(self, eps=1e-12): return int((self.weights > eps).sum())

    def multiplicity(self, macro: Callable[[torch.Tensor], torch.Tensor]):
        """Macro-état -> nombre de micro-configurations compatibles (vectorisé)."""
        m = macro(self.states)
        vals, counts = torch.unique(m, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals.tolist(), counts.tolist())}

    def map_state(self):
        i = int(self.weights.argmax())
        return {v: self.domains[v][int(self.states[i, j])] for j, v in enumerate(self.vars)}, float(self.weights[i])

    def conflict(self, var, records: List[MeasurementRecord], credible=0.6, second=0.2):
        """CONFLICTED si des mesures crédibles sont incompatibles ET la postérieure reste multimodale (pas de moyenne)."""
        vals = {r.outcome.value for r in records if r.context.noise.reliability >= credible}
        if len(vals) < 2: return False
        m = sorted(self.marginal(var).values(), reverse=True)
        return m[1] >= second


def fuse(prior: CompatibleStateSet, records: List[MeasurementRecord], reliability_of: Callable[[str], float] = None,
         graph=None) -> CompatibleStateSet:
    """Fusion de mesures (capteurs, documents…) pondérées par la fiabilité (SourceReliabilityModel) ; reste une distribution."""
    post = prior
    for r in records:
        rel = reliability_of(r.provenance) if reliability_of else None
        post = post.condition(r, reliability=rel)
    trace(graph or prior.graph, MeasurementOp("Fuse", {"n": len(records)}))
    return post


# ================================================================ gain d'information (Shannon)

def expected_posterior_entropy(belief: CompatibleStateSet, op: MeasurementOperator, target=None, grid=None):
    """E_o[H(S | o)] (ou de la marginale cible), vectorisé sur les issues possibles."""
    var = (op.observable.entity, op.observable.attribute)
    col = belief.column(var)
    if op.observable.discrete:
        outs = op.outcomes(); L = torch.stack([op.likelihood(o, col) for o in outs])                  # [O, N]
    else:
        vals = torch.tensor([float(x) for x in belief.domains[var]], dtype=torch.float64)[col]
        outs = list(grid); L = torch.stack([op.likelihood(MeasurementOutcome(o), vals) for o in outs])
        L = L / L.sum(0, keepdim=True).clamp_min(1e-300)                                              # issues discrétisées
    w = belief.weights
    joint = L * w                                                                                     # [O, N]
    po = joint.sum(1)                                                                                 # p(o)
    post = joint / po.unsqueeze(1).clamp_min(1e-300)
    if target is None:
        H = -(post * post.clamp_min(1e-300).log()).sum(1)
    else:
        tcol = belief.column(target); K = len(belief.domains[target])
        m = torch.zeros(len(outs), K, dtype=torch.float64).index_add_(1, tcol, post)
        H = -(m * m.clamp_min(1e-300).log()).sum(1)
    return float((po * H).sum()), po


def information_gain(belief: CompatibleStateSet, op: MeasurementOperator, target=None, grid=None) -> float:
    """IG(M) = H(S avant) − E[H(S après M)] (sur la variable cible si donnée)."""
    h0 = belief.entropy(target)
    eh, _ = expected_posterior_entropy(belief, op, target, grid)
    return max(0.0, h0 - eh)


def decision_value(belief: CompatibleStateSet, op: MeasurementOperator, target, wrong_cost=20.0, abstain_cost=1.0,
                   calibrate=None, grid=None):
    """Valeur de l'information POUR LA DÉCISION : perte attendue avant − E[perte après la mesure].
    perte(b) = min((1 − p_cal(max b)) · C_erreur, C_abstention) ; p_cal = confiance calibrée si fournie."""
    cal = calibrate or (lambda p: p)
    loss = lambda p: min((1 - cal(p)) * wrong_cost, abstain_cost)
    before = loss(max(belief.marginal(target).values()))
    var = (op.observable.entity, op.observable.attribute)
    col = belief.column(var)
    if op.observable.discrete or not hasattr(op, "outcomes"):
        outs = op.outcomes(); L = torch.stack([op.likelihood(o, col) for o in outs])
    else:
        outs = op.outcomes(); L = torch.stack([op.likelihood(o, col) for o in outs])
    joint = L * belief.weights; po = joint.sum(1); post = joint / po.unsqueeze(1).clamp_min(1e-300)
    tcol = belief.column(target); K = len(belief.domains[target])
    m = torch.zeros(len(outs), K, dtype=torch.float64).index_add_(1, tcol, post)
    after = sum(float(po[i]) * loss(float(m[i].max())) for i in range(len(outs)))
    return max(0.0, before - after)


# ================================================================ planificateur de mesures

@dataclass
class MeasurementChoice:
    operator: MeasurementOperator
    information_gain: float
    utility: float
    cost: float


class MeasurementPlanner:
    """Choisit la prochaine mesure : utilité = IG − coût − coût de perturbation − coût de latence.
    Règle d'arrêt : confiance suffisante, IG résiduel faible, ou utilité négative / budget épuisé."""

    POLICIES = ("random", "fixed", "most_reliable", "max_ig", "max_utility")

    def __init__(self, policy="max_utility", value_of_information=1.0, disturbance_penalty=2.0, latency_weight=0.1,
                 confidence=0.95, min_ig=0.02, budget=10.0, seed=0, graph=None):
        self.policy, self.voi, self.dist_pen, self.lat_w = policy, value_of_information, disturbance_penalty, latency_weight
        self.confidence, self.min_ig, self.budget = confidence, min_ig, budget
        self.rng = random.Random(seed); self.graph = graph; self.spent = 0.0
        self.decision_costs = None; self.calibrate = None
        self.value_model = None; self.context = {}           # politiques "learned" / "hybrid" (MeasurementValueLearner)

    def disturbance_cost(self, op, target):
        d = op.context.disturbance
        if d == PASSIVE: return 0.0
        hits = target is not None and getattr(op.context, "domain", "") == "disturbs:" + str(target[1])
        return self.dist_pen * (2.0 if hits else 1.0) * (1.0 if d == WEAKLY_DISTURBING else 3.0)

    def score(self, belief, op, target, grid=None):
        ig = information_gain(belief, op, target, grid)
        cost = op.context.cost + self.disturbance_cost(op, target) + self.lat_w * op.context.latency
        if self.policy in ("learned", "hybrid") and self.value_model is not None and target is not None:
            dv = decision_value(belief, op, target, *(self.decision_costs or (20.0, 8.0)), calibrate=self.calibrate)
            f = self.value_model.features(belief, op, target, ig, dv, self.context)
            pred = self.value_model.predict(f, hybrid=self.policy == "hybrid", analytic=dv)
            return MeasurementChoice(op, ig, pred - cost, cost)
        if self.decision_costs is not None and target is not None:             # utilité décisionnelle (risque)
            v = decision_value(belief, op, target, *self.decision_costs, calibrate=self.calibrate)
            return MeasurementChoice(op, ig, v - cost, cost)
        return MeasurementChoice(op, ig, self.voi * ig - cost, cost)

    def should_stop(self, belief, target, candidates):
        if target is not None and max(belief.marginal(target).values()) >= self.confidence: return True, "confiance suffisante"
        if self.spent >= self.budget: return True, "budget épuisé"
        if not candidates: return True, "aucune mesure possible"
        return False, ""

    def choose(self, belief, candidates: List[MeasurementOperator], target=None, grid=None) -> Optional[MeasurementChoice]:
        stop, why = self.should_stop(belief, target, candidates)
        if stop: return None
        scored = [self.score(belief, op, target, grid) for op in candidates]
        if self.policy == "random": c = self.rng.choice(scored)
        elif self.policy == "fixed": c = scored[0]
        elif self.policy == "most_reliable": c = max(scored, key=lambda s: s.operator.context.noise.reliability)
        elif self.policy == "max_ig": c = max(scored, key=lambda s: s.information_gain)
        else:
            c = max(scored, key=lambda s: s.utility)
            if c.utility <= 0 or (self.decision_costs is None and c.information_gain < self.min_ig): return None   # arrêt
        if self.policy != "max_utility" and c.information_gain < self.min_ig * 0.01 and self.policy != "random": return None
        self.spent += c.cost
        trace(self.graph, MeasurementOp("SelectMeasurement", {"operator": c.operator.name, "ig": round(c.information_gain, 4),
                                                              "utility": round(c.utility, 4)}))
        return c


# ================================================================ compromis de capteur (analogie : Heisenberg, défini par le capteur)

@dataclass(frozen=True)
class MeasurementTradeoff:
    """Compromis EXPLICITE du modèle de capteur : un réglage a ∈ (0, 1) répartit la précision.
    σ_position = s0 / a ; σ_vitesse = v0 / (1 − a). Ce n'est PAS un principe d'incertitude quantique."""
    s0: float = 0.1
    v0: float = 0.1

    def sigmas(self, a):
        a = min(max(a, 1e-3), 1 - 1e-3); return self.s0 / a, self.v0 / (1 - a)

    def best_setting(self, goal_weights: Tuple[float, float], grid=None):
        """Réglage minimisant la variance pondérée selon le BUT (poids sur position, vitesse)."""
        grid = grid if grid is not None else torch.linspace(0.02, 0.98, 97)
        sp = self.s0 / grid; sv = self.v0 / (1 - grid)
        loss = goal_weights[0] * sp ** 2 + goal_weights[1] * sv ** 2
        return float(grid[int(loss.argmin())])


# ================================================================ invariants et trace

def check_invariants(dist: Optional[MeasurementDistribution] = None, record: Optional[MeasurementRecord] = None,
                     frames: Optional[Dict[str, ReferenceFrame]] = None, latent: Optional[LatentWorldState] = None,
                     values: Sequence[FramedValue] = ()):
    out = {}
    if dist is not None: out[MeasurementInvariant.NORMALIZED_DISTRIBUTION] = dist.normalized()
    if record is not None:
        out[MeasurementInvariant.PROVENANCE_PRESENT] = bool(record.provenance)
        if record.observable.discrete:
            out[MeasurementInvariant.OUTCOME_COMPATIBLE] = record.outcome.value in record.observable.domain
        if record.observable.frame_required:
            out[MeasurementInvariant.REFERENCE_FRAME_VALID] = record.context.frame in (frames or {}) or record.context.frame == "world"
    if values:
        units = {v.unit for v in values}; out[MeasurementInvariant.UNIT_CONSISTENT] = len(units) == 1
        if frames is not None:
            out[MeasurementInvariant.REFERENCE_FRAME_VALID] = all(v.frame in frames or v.frame == "world" for v in values)
    if latent is not None: out[MeasurementInvariant.NO_HIDDEN_TRUTH_ACCESS] = latent.agent_peeks == 0
    return out


@dataclass(frozen=True)
class MeasurementOp(TransformationSpec):
    """Opération de l'algèbre de mesure, tracée dans le TransformGraph (Measure, ChangeBasis, ChangeFrame,
    Condition, Marginalize, Fuse, SelectMeasurement). Ne modifie pas le monde (sauf mesure perturbatrice)."""
    name: str = "measure"
    params: Dict[str, Any] = field(default_factory=dict, compare=False)
    domain: TransformationDomain = TransformationDomain.MEASUREMENT
    invertible: bool = False
    cost: float = 0.0

    def __init__(self, name, params=None):
        object.__setattr__(self, "name", name); object.__setattr__(self, "params", dict(params or {}))
        object.__setattr__(self, "domain", TransformationDomain.MEASUREMENT); object.__setattr__(self, "invertible", name in ("ChangeBasis", "ChangeFrame"))
        object.__setattr__(self, "cost", 0.0); object.__setattr__(self, "parameters", dict(params or {}))
        from core.invariants import InvariantSet
        object.__setattr__(self, "preserves", InvariantSet(set())); object.__setattr__(self, "changes", set())
        object.__setattr__(self, "preconditions", ()); object.__setattr__(self, "postconditions", ())

    def apply(self, *args, **kwargs): return None


def new_graph(name="mesure"): return TransformGraph(TransformNode(name))


def trace(graph, op: MeasurementOp):
    if graph is not None: graph.apply_transformation(op)
    return op

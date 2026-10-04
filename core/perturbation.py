"""Perturbations comme TransformationSpec traçables ; réponse ; sensibilité locale ; Jacobiens (autograd, LOCAL) ;
sensibilité structurelle (discrète) ; attribution de crédit généralisée (observationnelle vs interventionnelle).

Principe : STATE → PERTURB → MEASURE RESPONSE → ESTIMATE SENSITIVITY → (invariants, stabilité : core/stability.py).

La rétropropagation n'est PAS le moteur de raisonnement : `DifferentiableSensitivity` ne fournit que des gradients /
Jacobiens LOCAUX pour les sous-systèmes PyTorch ; tout le reste passe par des perturbations discrètes
(`StructuralSensitivity`) et des INTERVENTIONS réelles (`interventional_credit`). Une sensibilité observationnelle
(gradient, corrélation) n'est jamais présentée comme une cause.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from core.model_state import INPUT, PARAM_NAMES, ModelState
from core.transform_graph import TransformGraph, TransformNode
from core.transformations import TransformationDomain, TransformationSpec

KINDS = ("input", "measurement", "belief", "parameter", "connection", "routing", "transformation")


# ================================================================ perturbation (TransformationSpec)

class Perturbation(TransformationSpec):
    """Perturbation traçable : kind, target, magnitude, norm, locality, direction, domain.
    `apply(obj)` renvoie un NOUVEL objet perturbé (l'original n'est jamais modifié) ; `inverse()` si réversible."""

    def __init__(self, kind, target, magnitude=1.0, direction=None, norm="l2", locality="local", domain="model",
                 fn: Optional[Callable] = None, inverse_fn: Optional[Callable] = None, name=None):
        assert kind in KINDS, kind
        object.__setattr__(self, "kind", kind); object.__setattr__(self, "target", target)
        object.__setattr__(self, "magnitude", float(magnitude)); object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "norm", norm); object.__setattr__(self, "locality", locality); object.__setattr__(self, "pdomain", domain)
        object.__setattr__(self, "fn", fn); object.__setattr__(self, "inverse_fn", inverse_fn)
        object.__setattr__(self, "name", name or f"Perturb[{kind}:{target}@{magnitude:g}]")
        object.__setattr__(self, "domain", TransformationDomain.MODEL if domain == "model" else TransformationDomain.GENERIC)
        object.__setattr__(self, "parameters", {"kind": kind, "target": str(target), "magnitude": float(magnitude), "norm": norm,
                                                "locality": locality, "domain": domain})
        object.__setattr__(self, "invertible", inverse_fn is not None or kind in ("input", "parameter", "connection", "belief"))
        object.__setattr__(self, "cost", 0.0)
        from core.invariants import InvariantSet as LabelSet
        object.__setattr__(self, "preserves", LabelSet(set())); object.__setattr__(self, "changes", {kind})
        object.__setattr__(self, "preconditions", ()); object.__setattr__(self, "postconditions", ())

    def scaled(self, k):
        return Perturbation(self.kind, self.target, self.magnitude * k, self.direction, self.norm, self.locality, self.pdomain,
                            self.fn, self.inverse_fn)

    def apply(self, obj):
        if self.fn is not None: return self.fn(obj, self)
        return DEFAULT_APPLY[self.kind](obj, self)

    def inverse(self):
        if not self.invertible: raise ValueError(f"{self.name} n'est pas réversible")
        if self.inverse_fn is not None:
            return Perturbation(self.kind, self.target, self.magnitude, self.direction, self.norm, self.locality, self.pdomain,
                                self.inverse_fn, self.fn, name=f"Inverse[{self.name}]")
        inv = Perturbation(self.kind, self.target, -self.magnitude, self.direction, self.norm, self.locality, self.pdomain)
        if hasattr(self, "_scale"): object.__setattr__(inv, "_scale", self._scale)
        else: object.__setattr__(inv, "_parent", self)
        return inv

    def __repr__(self): return self.name


def magnitudes(eps, n=5, factor=2.0): return [eps * factor ** k for k in range(n)]


def unit_direction(shape, seed, dtype=torch.float32):
    d = torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)
    return d / d.norm().clamp_min(1e-12)


# ---- applications par défaut (pures : copient l'objet) ----------------------------------------------

def _apply_input(x, p):
    d = p.direction if p.direction is not None else unit_direction(x.shape, 0, x.dtype)
    return x + p.magnitude * d


def _apply_parameter(model: ModelState, p):
    """target = (module, clé) ; direction = tenseur de même forme (norme 1) ; magnitude relative à ||W||."""
    m = copy.deepcopy(model); mod, key = p.target
    W = m.modules[mod].slow[key]
    d = p.direction if p.direction is not None else unit_direction(W.shape, sum(map(ord, f"{mod}.{key}")))
    scale = getattr(p, "_scale", None)
    if scale is None and hasattr(p, "_parent"): scale = getattr(p._parent, "_scale", None)   # inverse : échelle de l'original
    if scale is None: scale = float(W.norm().clamp_min(1e-6)); object.__setattr__(p, "_scale", scale)   # amplitude relative à ||W|| d'ORIGINE
    m.modules[mod].slow[key] = W + p.magnitude * scale * d
    return m


def _apply_connection(model, p):
    m = copy.deepcopy(model); c = m.connectivity[p.target]; c.strength = c.strength * (1.0 + p.magnitude); return m


def _apply_routing(model, p):
    """Routage : branche (avec force = magnitude) un module source sur une cible ; magnitude 0 = aucun changement."""
    from core.model_state import Connection
    m = copy.deepcopy(model); src, tgt = p.target
    if p.magnitude == 0: return m
    m.modules[src].active = True
    if src != INPUT and (INPUT, src) not in m.connectivity and m.modules[src].input_dim == m.input_dim:
        m.connectivity[(INPUT, src)] = Connection(INPUT, src, 1.0)
    m.connectivity[(src, tgt)] = Connection(src, tgt, p.magnitude)
    return m


def _apply_belief(S, p):
    """Bruit sur les log-poids d'un CompatibleStateSet (croyance) ; l'ensemble d'origine est intact."""
    d = p.direction if p.direction is not None else unit_direction(S.logw.shape, 0, torch.float64)
    return S._fork(S.logw + p.magnitude * d, f"perturbation de croyance ({p.magnitude:g})")


def _apply_measurement(op, p):
    """Perturbe le contexte de mesure : fiabilité r -> r − magnitude (bornée), opérateur copié."""
    from core.measurement import NoiseModel
    import dataclasses
    o = copy.copy(op); n = op.context.noise
    o.context = dataclasses.replace(op.context, noise=NoiseModel(n.kind, min(0.999, max(0.0, n.reliability - p.magnitude)), n.sigma * (1 + p.magnitude)))
    return o


def _apply_transformation(chain, p):
    """Chaîne de transformations : target = index ; direction = transformation de remplacement."""
    c = list(chain); c[p.target] = p.direction; return c


DEFAULT_APPLY = {"input": _apply_input, "parameter": _apply_parameter, "connection": _apply_connection,
                 "routing": _apply_routing, "belief": _apply_belief, "measurement": _apply_measurement,
                 "transformation": _apply_transformation}


def traced(graph: Optional[TransformGraph], p: Perturbation, config=None):
    """Ajoute la perturbation au TransformGraph (branche hypothétique)."""
    if graph is None: return None
    node = graph.current.add_child(p, config or {"perturbation": p.name}, name=f"P{len(graph.nodes)}")
    graph.nodes[node.name] = node
    return node


def new_graph(name="perturbation"): return TransformGraph(TransformNode(name, configuration={}))


# ================================================================ réponse

@dataclass
class PerturbationResponse:
    perturbation: str
    magnitude: float
    output_change: float = 0.0
    belief_change: float = 0.0
    decision_change: float = 0.0
    invariant_violations: Tuple[str, ...] = ()
    task_loss_change: float = 0.0
    branch_change: bool = False
    measurement_policy_change: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def sensitivity(self): return self.output_change / self.magnitude if self.magnitude else 0.0


def model_response(model, perturbed, x, y=None, p=None, invariants=None, x_old=None, y_old=None):
    """Réponse d'un ModelState à une perturbation (sortie, décision, perte, chemin de calcul, invariants)."""
    with torch.no_grad():
        o0, o1 = model.forward(x), perturbed.forward(x)
    r = PerturbationResponse(p.name if p else "?", p.magnitude if p else 0.0)
    r.output_change = float((o1 - o0).norm() / math.sqrt(len(x)))
    r.decision_change = float((o0.argmax(1) != o1.argmax(1)).float().mean())
    if y is not None:
        f = torch.nn.functional.cross_entropy
        r.task_loss_change = float(f(o1, y) - f(o0, y))
    r.branch_change = model.topological_order() != perturbed.topological_order()
    if invariants: r.invariant_violations = tuple(n for n, fn in invariants.items() if not fn(perturbed))
    return r


def local_sensitivity(F: Callable, x, delta) -> float:
    """S ≈ ||F(x + δ) − F(x)|| / ||δ|| (générique : tenseurs)."""
    fx, fy = F(x), F(x + delta)
    return float((fy - fx).norm() / delta.norm().clamp_min(1e-12))


def sensitivity_curve(fn_response: Callable[[float], float], eps=1e-3, n=6, factor=2.0):
    """Réponse pour ε, 2ε, 4ε… ; renvoie (magnitudes, réponses, sensibilités, exposant log-log ≈ ordre local)."""
    ms = magnitudes(eps, n, factor); rs = [fn_response(m) for m in ms]
    sens = [r / m for r, m in zip(rs, ms)]
    pts = [(math.log(m), math.log(r)) for m, r in zip(ms, rs) if r > 0]
    slope = None
    if len(pts) >= 2:
        xs, ys = zip(*pts); mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((a - mx) ** 2 for a in xs); slope = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / den if den else None
    return {"magnitudes": ms, "responses": rs, "sensitivity": sens, "loglog_slope": slope}


def belief_sensitivity(S, magnitude, var=None, seed=0):
    """Variation totale de la marginale (ou des poids) sous une perturbation des log-poids."""
    p = Perturbation("belief", "logw", magnitude, unit_direction(S.logw.shape, seed, torch.float64) * math.sqrt(len(S.logw)))
    T = p.apply(S)
    if var is None: change = 0.5 * float((T.weights - S.weights).abs().sum())
    else:
        a, b = S.marginal(var), T.marginal(var); change = 0.5 * sum(abs(a[k] - b.get(k, 0.0)) for k in a)
    return change / magnitude if magnitude else 0.0, T


# ================================================================ sensibilité différentiable (autograd, LOCAL)

class DifferentiableSensitivity:
    """Gradients / Jacobiens LOCAUX d'un ModelState (sous-système différentiable) pour le moteur général.
    Fournit : ∂sortie/∂entrée, ∂perte/∂poids, ∂routage/∂représentation (via les forces de connexion = portes),
    et la comparaison autograd vs différences finies."""

    def __init__(self, model: ModelState):
        self.model = model

    def _forward(self, x, overrides=None, strengths=None):
        m = self.model
        if strengths is None: return m.forward(x, overrides)
        saved = {k: c.strength for k, c in m.connectivity.items()}
        try:
            for k, s in strengths.items(): m.connectivity[k].strength = s           # tenseurs (requires_grad)
            return m.forward(x, overrides)
        finally:
            for k, s in saved.items(): m.connectivity[k].strength = s

    def input_jacobian(self, x):
        """∂sortie/∂entrée pour un exemple x (vecteur) : [out_dim, in_dim]."""
        return torch.autograd.functional.jacobian(lambda v: self._forward(v.unsqueeze(0))[0], x)

    def weight_gradients(self, x, y):
        """∂perte/∂poids lents, par module et clé."""
        params = {m: {k: v.clone().requires_grad_(True) for k, v in self.model.modules[m].slow.items()}
                  for m in self.model.topological_order()}
        over = {m: {k: d[k] + self.model.fast_parameters.get(m, {}).get(k, 0) for k in PARAM_NAMES} for m, d in params.items()}
        loss = torch.nn.functional.cross_entropy(self._forward(x, over), y)
        grads = torch.autograd.grad(loss, [t for d in params.values() for t in d.values()])
        out, i = {}, 0
        for m, d in params.items():
            out[m] = {}
            for k in d: out[m][k] = grads[i]; i += 1
        return out

    def routing_gradients(self, x, y):
        """∂perte/∂force de chaque connexion active (le routage comme porte continue)."""
        s = {k: torch.tensor(float(c.strength), requires_grad=True) for k, c in self.model.connectivity.items()}
        loss = torch.nn.functional.cross_entropy(self._forward(x, strengths=s), y)
        used = [k for k in s if s[k].requires_grad]
        g = torch.autograd.grad(loss, [s[k] for k in used], allow_unused=True)
        return {k: (0.0 if gi is None else float(gi)) for k, gi in zip(used, g)}

    def routing_representation_jacobian(self, x, source="expert_a", target="head"):
        """∂(entrée pondérée du module cible)/∂(représentation source) = force × I — exposé pour contrôle."""
        c = self.model.connectivity[(source, target)]
        dim = self.model.modules[source].output_dim
        return c.strength * torch.eye(dim)

    def finite_difference_input(self, x, h=1e-3):
        """Jacobien d'entrée par différences finies centrées (contrôle d'autograd)."""
        cols = []
        with torch.no_grad():
            for j in range(x.numel()):
                e = torch.zeros_like(x); e[j] = h
                cols.append((self._forward((x + e).unsqueeze(0))[0] - self._forward((x - e).unsqueeze(0))[0]) / (2 * h))
        return torch.stack(cols, 1)

    def finite_difference_weight(self, x, y, module, key, idx, h=1e-3):
        W = self.model.modules[module].slow[key]
        def loss_at(v):
            Wp = W.clone(); Wp.view(-1)[idx] = v
            over = {module: {k: (Wp if k == key else self.model.modules[module].slow[k]) for k in PARAM_NAMES}}
            with torch.no_grad(): return float(torch.nn.functional.cross_entropy(self.model.forward(x, over), y))
        w0 = float(W.view(-1)[idx]); return (loss_at(w0 + h) - loss_at(w0 - h)) / (2 * h)

    def compare(self, x, y, n_weights=8, seed=0, h=1e-3):
        """Écart relatif autograd vs différences finies (entrée et poids)."""
        x64 = x.double()
        model64 = copy.deepcopy(self.model)
        for mod in model64.modules.values(): mod.slow = {k: v.double() for k, v in mod.slow.items()}
        ds = DifferentiableSensitivity(model64)
        J, Jfd = ds.input_jacobian(x64[0]), ds.finite_difference_input(x64[0], h)
        rel_in = float((J - Jfd).norm() / J.norm().clamp_min(1e-12))
        g = ds.weight_gradients(x64, y); rng = random.Random(seed); errs = []
        for _ in range(n_weights):
            m = rng.choice(list(g)); k = rng.choice(list(g[m])); idx = rng.randrange(g[m][k].numel())
            a = float(g[m][k].view(-1)[idx]); f = ds.finite_difference_weight(x64, y, m, k, idx, h)
            errs.append(abs(a - f) / max(1e-6, abs(a) + abs(f)))
        return {"input_rel_error": rel_in, "weight_rel_error_max": max(errs), "weight_rel_error_mean": sum(errs) / len(errs)}


# ================================================================ sensibilité structurelle (discrète)

@dataclass
class StructuralEdit:
    """Perturbation DISCRÈTE d'un composant non différentiable (relation, Skill, règle, preuve, source, routage)."""
    name: str
    component: str
    kind: str
    apply: Callable[[Any], Any]


class StructuralSensitivity:
    """Sensibilité = |score(edit(système)) − score(système)| pour chaque édition discrète (par intervention)."""

    def __init__(self, score: Callable[[Any], float]):
        self.score = score

    def run(self, system, edits: Sequence[StructuralEdit], graph=None):
        base = self.score(system); out = []
        for e in edits:
            s = self.score(e.apply(copy.deepcopy(system)))
            if graph is not None: traced(graph, Perturbation("transformation", e.component, 1.0, name=f"Edit[{e.name}]"))
            out.append({"edit": e.name, "component": e.component, "kind": e.kind, "score": s, "delta": s - base,
                        "sensitivity": abs(s - base)})
        return base, sorted(out, key=lambda r: -r["sensitivity"])


# ================================================================ attribution de crédit généralisée

COMPONENT_KINDS = ("weights", "routing", "expert", "measurement", "source", "belief", "lexical", "transformation", "planner")


@dataclass
class Credit:
    component: str
    kind: str
    observational: float = 0.0       # sensibilité locale (gradient / corrélation) : PAS une cause
    interventional: float = 0.0      # amélioration RÉELLE du score après intervention (restauration / remplacement)


def interventional_credit(system, components: Dict[str, Tuple[str, Callable[[Any], Any]]], score: Callable[[Any], float]):
    """Pour chaque composant : intervenir réellement (le remplacer par sa version de référence / neutre), réévaluer.
    Crédit = score(intervention) − score(système). Le système d'origine n'est pas modifié."""
    base = score(system); out = []
    for name, (kind, intervene) in components.items():
        out.append(Credit(name, kind, interventional=score(intervene(copy.deepcopy(system))) - base))
    return base, sorted(out, key=lambda c: -c.interventional)


def gradient_credit(model: ModelState, x, y):
    """Crédit OBSERVATIONNEL d'un modèle : ||∂perte/∂W|| · ||W|| par module, et |∂perte/∂force| par connexion.
    Ne voit que les paramètres différentiables."""
    ds = DifferentiableSensitivity(model); g = ds.weight_gradients(x, y); out = {}
    for m, d in g.items():
        out[f"weights:{m}"] = sum(float(d[k].norm() * model.modules[m].slow[k].norm()) for k in d)
    for k, v in ds.routing_gradients(x, y).items():
        out[f"routing:{k[0]}->{k[1]}"] = abs(v) * float(model.connectivity[k].strength)
    return out

"""État cognitif unifié : OBSERVATIONS -> DISCOURS -> IDENTITÉ -> PREUVES -> CROYANCES -> QUESTIONS.

Un seul état (UnifiedCognitiveState) et un seul résolveur servent à l'ingestion, aux questions et aux commandes ;
seul le niveau de risque accepté change (observation < question < action).

Formule d'identité explicite (testable), pour une mention m émise par la source s et un candidat e :
    score(e)  = log prior(e) + log compat(e) + support(e) + log temporal(e)
    p_lang(e) = softmax(score)
    P(e | m)  = r_s · p_lang(e) + (1 − r_s) · prior(e)          (une source peu fiable peut décrire faux)
prior : uniforme sur les candidats du bon type ; compat : marginale de la satisfaction de contraintes jointe
(identité, distinction, injectivité) ; support : log-vraisemblance de l'attribut décrit selon le graphe de preuves ;
temporal : cohérence des états datés (un état mutable ancien n'exclut pas un candidat si un événement l'a changé).

Les croyances dépendent explicitement de leurs preuves, et les preuves textuelles de l'identité de leur mention.
Un mini-TMS (JUSTIFIED / DEFEATED / SUSPENDED / UNKNOWN) propage les révisions : si l'identité change,
les croyances dépendantes sont retirées puis recalculées. Tout est recalculé à partir des enregistrements
actifs (pas de copie divergente de croyance).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from core.discourse import AMBIGUOUS, HYPOTHETICAL, NONEXISTENT, PLACE_TYPES, DiscourseResolver
from core.independent_agent import BELIEF, FACT, BeliefState, Evidence, SourceReliability, token_class, tokenize

JUSTIFIED, DEFEATED, SUSPENDED, UNKNOWN_STATUS = "JUSTIFIED", "DEFEATED", "SUSPENDED", "UNKNOWN"
SUPPORTED, LIKELY, CONFLICTED, REFUTED, UNKNOWN_ANS = "SUPPORTED", "LIKELY", "CONFLICTED", "REFUTED", "UNKNOWN"
ANSWER, ASK, ABSTAIN, VERIFY = "ANSWER", "ASK", "ABSTAIN", "VERIFY"
ATTR_DOMAIN = {"color": 4, "loc": 12}
USER = "user"
NONE = "∅"                     # hypothèse « la description ne désigne aucun objet » (présupposition fausse)
PRESUPPOSITION_FAILURE = 0.02  # a priori explicite de cette hypothèse (descriptions définies seulement)


# ================================================================ identité calibrée par la source

@dataclass
class IdentityEvidence:
    mention: str
    entity: str
    source: str
    reliability: float
    supporting: List[str]
    contradicting: List[str]
    time: int
    confidence: float
    distribution: Dict[str, float] = field(default_factory=dict)


def identity_posterior(prior, compat, support, temporal, reliability):
    """Vecteurs [n candidats] -> P(e | preuves). Vectorisé ; chaque terme est séparé et testable."""
    prior = prior / prior.sum()
    score = prior.clamp_min(1e-12).log() + compat.clamp_min(1e-12).log() + support + temporal.clamp_min(1e-12).log()
    if not torch.isfinite(score).any() or float((compat * temporal).sum()) <= 0:
        return torch.zeros_like(prior)
    p_lang = torch.softmax(score, 0)
    r = float(min(max(reliability, 0.0), 1.0))
    return r * p_lang + (1 - r) * prior


# ================================================================ calibration

def calibration_report(conf, correct, bins=10):
    """Bacs de fiabilité, ECE et Brier (vectorisé)."""
    c = torch.as_tensor(conf, dtype=torch.float); y = torch.as_tensor(correct, dtype=torch.float)
    if c.numel() == 0: return {"n": 0, "ece": None, "brier": None, "bins": []}
    idx = (c * bins).long().clamp(0, bins - 1)
    count = torch.bincount(idx, minlength=bins).float()
    sconf = torch.zeros(bins).index_add_(0, idx, c); sacc = torch.zeros(bins).index_add_(0, idx, y)
    nz = count > 0
    mc = torch.where(nz, sconf / count.clamp_min(1), torch.zeros(bins)); ma = torch.where(nz, sacc / count.clamp_min(1), torch.zeros(bins))
    ece = float(((mc - ma).abs() * count).sum() / c.numel())
    table = [{"bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}", "n": int(count[i]), "confidence": round(float(mc[i]), 3),
              "accuracy": round(float(ma[i]), 3)} for i in range(bins) if nz[i]]
    return {"n": int(c.numel()), "ece": round(ece, 4), "brier": round(float(((c - y) ** 2).mean()), 4), "bins": table}


class IsotonicCalibrator:
    """Régression isotone (PAV) ajustée sur une partition de calibration ; identité si pas de données."""

    def __init__(self, prior_strength=10.0, bins=20):
        self.x = self.y = None; self.prior_strength = prior_strength; self.bins = bins

    def fit(self, conf, correct):
        """PAV, puis contraction vers l'identité : chaque palier reçoit `prior_strength` pseudo-observations dont
        l'exactitude vaut la confiance annoncée (petit échantillon -> peu de correction ; jamais 1,0 par 20 succès)."""
        pairs = sorted(zip(map(float, conf), map(float, correct)))
        if len(pairs) < 5: return self
        c = torch.tensor([x for x, _ in pairs]); y = torch.tensor([v for _, v in pairs])
        b = (c * self.bins).long().clamp(0, self.bins - 1)                # histogramme puis PAV (vectorisé)
        cnt = torch.bincount(b, minlength=self.bins).float()
        sy = torch.zeros(self.bins).index_add_(0, b, y); sc = torch.zeros(self.bins).index_add_(0, b, c)
        out = []
        for i in torch.nonzero(cnt).view(-1).tolist():
            out.append([i / self.bins, float(sy[i]), float(cnt[i]), float(sc[i])])  # [bord inférieur, Σ exact, n, Σ confiance]
            while len(out) > 1 and out[-2][1] / out[-2][2] > out[-1][1] / out[-1][2]:
                _, y2, n2, c2 = out.pop(); out[-1][1] += y2; out[-1][2] += n2; out[-1][3] += c2
        self.n = len(pairs)
        s = self.prior_strength
        self.x = torch.tensor([b[0] for b in out])
        self.y = torch.tensor([(b[1] + s * b[3] / b[2]) / (b[2] + s) for b in out])
        self.y = torch.cummax(self.y, 0).values                          # rester monotone après contraction
        return self

    def __call__(self, p):
        """Palier isotone du plus grand point de calibration ≤ p ; sous les données : jamais au-dessus de p."""
        if self.x is None: return p
        t = torch.as_tensor(p, dtype=torch.float).reshape(-1).contiguous()
        i = torch.searchsorted(self.x, t, right=True) - 1
        out = torch.where(i >= 0, self.y[i.clamp(min=0)], torch.minimum(t, self.y[0].expand_as(t)))
        return float(out[0]) if torch.as_tensor(p).dim() == 0 else out


# ================================================================ décision par risque attendu

COSTS = {"wrong": {"observation": 3.0, "query": 20.0, "action": 100.0}, "abstain": 1.0, "ask": 0.5, "verify": 0.2}


class RiskPolicy:
    """decision = argmin risque attendu. Coûts : mauvaise réponse assurée > abstention > clarification > vérification.
    Seuil implicite d'une réponse : p ≥ 1 − C_abstain / C_wrong(contexte) (0,67 / 0,95 / 0,99)."""

    def __init__(self, costs=None, thresholds=None):
        self.costs = costs or COSTS
        self.thresholds = dict(thresholds or {})

    def threshold(self, context):
        if context in self.thresholds: return self.thresholds[context]
        return 1 - self.costs["abstain"] / self.costs["wrong"][context]

    def risks(self, p, context, can_ask=False, can_verify=False, p_after_ask=1.0):
        # coût effectif d'une erreur : C_abstain / (1 − seuil) ; sans seuil calibré, c'est C_wrong(contexte)
        cw = self.costs["abstain"] / max(1 - self.threshold(context), 1e-9)
        out = {ANSWER: (1 - p) * cw, ABSTAIN: self.costs["abstain"]}
        if can_ask: out[ASK] = self.costs["ask"] + (1 - p_after_ask) * self.costs["abstain"]
        if can_verify: out[VERIFY] = self.costs["verify"] + 0.5 * min(out.values())
        return out

    def decide(self, p, context, can_ask=False, can_verify=False, p_after_ask=1.0):
        r = self.risks(p, context, can_ask, can_verify, p_after_ask)
        return min(r, key=lambda k: (r[k], [ANSWER, ASK, VERIFY, ABSTAIN].index(k)))


def calibrate_threshold(conf, correct, cost_wrong, cost_abstain=1.0, grid=None):
    """Seuil minimisant le risque empirique sur une partition de VALIDATION (jamais le test)."""
    c = torch.as_tensor(conf, dtype=torch.float); y = torch.as_tensor(correct, dtype=torch.float)
    grid = torch.as_tensor(grid if grid is not None else [i / 100 for i in range(50, 100)] + [0.995, 1.0])
    if c.numel() == 0: return float(1 - cost_abstain / cost_wrong)
    commit = c.view(1, -1) >= grid.view(-1, 1)                                   # [seuils, items]
    risk = (commit * (1 - y) * cost_wrong + (~commit) * cost_abstain).float().mean(1)
    return float(grid[int(risk.argmin())])


# ================================================================ maintenance de vérité

@dataclass
class Node:
    id: str
    kind: str                      # premise | identity | evidence | derived
    status: str = JUSTIFIED
    deps: List[str] = field(default_factory=list)
    rule: str = ""
    info: Dict[str, Any] = field(default_factory=dict)
    justifications: List[List[str]] = field(default_factory=list)   # OU de ET : (A ∧ C) ∨ (D) ∨ (E ∧ F)


class TruthMaintenance:
    """Justifications multiples (OU de ET) : un nœud est JUSTIFIED si AU MOINS UNE de ses justifications a toutes ses
    dépendances JUSTIFIED ; sinon SUSPENDED si une dépendance est défaite / suspendue, UNKNOWN sinon."""

    def __init__(self): self.nodes: Dict[str, Node] = {}; self.log: List[Tuple[str, str, str, str]] = []

    def add(self, nid, kind, deps=(), rule="", status=JUSTIFIED, justifications=None, **info):
        """deps : une justification ET ; justifications : plusieurs justifications INDÉPENDANTES (OU de ET)."""
        just = [list(j) for j in justifications] if justifications else ([list(deps)] if deps else [])
        flat = list(dict.fromkeys(d for j in just for d in j))
        self.nodes[nid] = Node(nid, kind, status, flat, rule, info, just)
        self.propagate()
        return self.nodes[nid]

    def add_justification(self, nid, deps):
        """Ajoute une justification indépendante à un nœud existant (ex. une deuxième source)."""
        n = self.nodes[nid]
        n.justifications.append(list(deps))
        n.deps = list(dict.fromkeys(n.deps + list(deps)))
        self.propagate()

    def set(self, nid, status, reason=""):
        n = self.nodes.get(nid)
        if n is None or n.status == status: return
        self.log.append((nid, n.status, status, reason)); n.status = status
        self.propagate()

    def own(self, nid):
        return self.nodes[nid].info.get("own", self.nodes[nid].status)

    def propagate(self):
        for _ in range(len(self.nodes) + 1):
            changed = False
            for n in self.nodes.values():
                if not n.deps: continue
                n.info.setdefault("own", n.status if n.status != SUSPENDED else JUSTIFIED)
                st = lambda d: self.nodes[d].status if d in self.nodes else UNKNOWN_STATUS
                just = n.justifications or [n.deps]
                if n.info["own"] in (DEFEATED, SUSPENDED): new = n.info["own"]     # décision explicite : persistante
                elif any(all(st(d) == JUSTIFIED for d in j) for j in just): new = JUSTIFIED   # UNE justification suffit
                elif any(st(d) in (DEFEATED, SUSPENDED) for j in just for d in j): new = SUSPENDED
                else: new = UNKNOWN_STATUS
                if new != n.status: self.log.append((n.id, n.status, new, "propagation")); n.status = new; changed = True
            if not changed: break

    def defeat(self, nid, reason=""):
        n = self.nodes.get(nid)
        if n is None: return
        n.info["own"] = DEFEATED
        self.set(nid, DEFEATED, reason)

    def dependents(self, nid):
        out, frontier = set(), {nid}
        while frontier:
            nxt = {n.id for n in self.nodes.values() if set(n.deps) & frontier} - out
            out |= nxt; frontier = nxt
        return out

    def support_of(self, nid):
        """Remonte les dépendances (observations, identités, sources, règles)."""
        seen, stack = [], [nid]
        while stack:
            x = stack.pop()
            if x in seen or x not in self.nodes: continue
            seen.append(x); stack += self.nodes[x].deps
        return [self.nodes[x] for x in seen]

    def counts(self): return dict(Counter(n.status for n in self.nodes.values()))


# ================================================================ affirmations candidates et réponses

@dataclass
class CandidateClaim:
    id: str
    source: str
    text: str
    time: int
    referent: str
    kind: str                     # attr | loc | event
    value: Any
    tense: str = "now"
    entity: Optional[str] = None
    identity_confidence: float = 0.0
    status: str = SUSPENDED
    evidence_node: Optional[str] = None
    origin: str = "prédicat"                  # prédicat | description (présupposition)
    frozen: bool = False                     # consolidée : plus réévaluée (dégelée par une contradiction)
    best_candidate: Optional[str] = None     # meilleur candidat quand l'identité n'est pas engagée
    contested: bool = False                  # contestée (révision minimale) : exclue des preuves du référent


@dataclass
class EpistemicAnswer:
    status: str                   # SUPPORTED | LIKELY | UNKNOWN | CONFLICTED | REFUTED
    action: str                   # ANSWER | ASK | ABSTAIN
    value: Any = None
    confidence: float = 0.0
    entity: Optional[str] = None
    identity_confidence: float = 0.0
    phrase: str = ""
    question: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    reason: str = ""
    risks: Dict[str, float] = field(default_factory=dict)
    linguistic_uncertainty: float = 0.0     # la phrase est-elle comprise ? (sens des mots)
    epistemic_uncertainty: float = 0.0      # le monde est-il établi ? (croyance × identité)

    @property
    def output(self):
        """Forme courte évaluable : valeur engagée, ou ASK / UNKNOWN / CONFLICTED / INVALID_PREMISE."""
        if self.action == ANSWER: return self.value if self.status in (SUPPORTED, REFUTED) else self.status
        if self.action == ASK: return "ASK"
        return self.reason if self.reason in ("INVALID_PREMISE", CONFLICTED) else "UNKNOWN"


PHRASES = {SUPPORTED: "Je sais que {v}.", LIKELY: "Les observations suggèrent que {v}.",
           CONFLICTED: "Les sources se contredisent sur {v}.", REFUTED: "Non : {v}.",
           "NO_ID": "Je ne peux pas identifier l'objet.", "INVALID_PREMISE": "Aucun objet ne correspond à cette description.",
           UNKNOWN_ANS: "Je ne sais pas."}
COLOR_WORD = {"red": "rouge", "blue": "bleu", "green": "vert", "yellow": "jaune"}
QUESTION_WORDS = {"où", "quelle", "couleur", "?"}
PLACE_WORD = {"on": "sur", "in": "dans", "under": "sous", "next_to": "à côté de", "table": "la table", "shelf": "l'étagère",
              "floor": "le sol"}
COMMANDS = {"prends", "donne", "pose", "mets", "range", "apporte"}


# ================================================================ état cognitif unifié

class UnifiedCognitiveState:
    def __init__(self, lexicon_agent, reliability: SourceReliability = None, policy: RiskPolicy = None,
                 calibrator=None, source_calibrated=True, answer_calibrator=None, consolidate_every=10, consolidate_age=8):
        self.lx = lexicon_agent
        self.reliability = reliability or SourceReliability()
        self.reliability.stats[USER] = [99.0, 1.0]
        self.belief = BeliefState(self.reliability)                    # EvidenceGraph + inventaire
        self.discourse = DiscourseResolver(lexicon_agent, [])          # DiscourseState + EntityIdentityGraph
        self.discourse.predicate_attrs_identity = False
        self.ref_graph = BeliefState(self.reliability).graph            # preuves au niveau du référent de discours
        self.tms = TruthMaintenance()
        self.policy = policy or RiskPolicy()
        self.calibrator = calibrator                                   # isotone (partition de calibration) ou None
        self.answer_calibrator = answer_calibrator                     # confiance de réponse -> exactitude (calibration)
        self.source_calibrated = source_calibrated
        self.lexical = getattr(lexicon_agent, "lexmem", None)
        self.records: Dict[str, Tuple] = {}                             # id -> (source, key, value, time, via, text)
        self.claims: Dict[str, CandidateClaim] = {}
        self.identity_log: List[IdentityEvidence] = []
        self.revisions: List[Tuple] = []
        self.transitions: List[Tuple] = []                              # (événement, entité, avant, après, temps)
        self.observation_requests = 0
        self.source_revisions = []
        self.consolidate_every, self.consolidate_age = consolidate_every, consolidate_age
        self.consolidations = 0
        self.checks = 0
        self.time = 0
        self._n = 0

    # ---- vues -------------------------------------------------------------------------
    @property
    def graph(self): return self.discourse.graph

    @property
    def state(self): return self.discourse.state

    def place_domain(self):
        """Nombre de lieux possibles dans CE monde (supports connus + contenants), pas une constante globale."""
        return 3 + sum(1 for k in self.belief.inventory.values() if k == "box")

    def entities(self, type_=None):
        return [e for e, k in self.belief.inventory.items() if k not in PLACE_TYPES and (type_ is None or k == type_)]

    def _id(self, prefix):
        self._n += 1; return f"{prefix}{self._n}"

    # ---- observations structurées (capteurs) ---------------------------------------------
    def observe(self, source, key, value, time=None):
        self.time = time if time is not None else self.time + 1
        if key[0] == "type":
            self.belief.inventory[key[1]] = value
            self.discourse.world[key[1]]["type"] = value
            return None
        rid = self._id("ev")
        self.records[rid] = (source, key, value, time if time is not None else self.time, "structured", None)
        self.tms.add(rid, "premise", rule=f"observation {source}", source=source, key=key, value=value)
        self.refresh()
        return rid

    # ---- langue -> affirmations candidates --------------------------------------------------
    def say(self, source, text, time=None):
        """Phrase déclarative -> CandidateClaim -> identité (même résolveur) -> preuves -> croyances."""
        self.time = time if time is not None else self.time + 1
        self.state.time = self.time - 1                              # une seule horloge : phrases et observations
        n0 = len(self.discourse.predicates)
        self.discourse.predicate_enabled = True
        mentions = self.discourse.process(text)
        out = []
        items = [(rid, kind, value, tense, "prédicat") for _, rid, kind, value, tense in self.discourse.predicates[n0:]]
        # présupposition descriptive : « le cube rouge » affirme implicitement « rouge » (preuve dépendante de l'identité)
        items += [(m.referent, "attr", m.attrs["attr"], "now", "description") for m in mentions
                  if m.referent and m.attrs.get("attr") and m.kind in ("np", "adj")
                  and not any(r == m.referent and k == "attr" for r, k, *_ in items)]
        described = {m.referent: m.attrs["attr"] for m in mentions if m.referent and m.attrs.get("attr")}
        for rid, kind, value, tense, origin in items:
            cid = self._id("c")
            c = CandidateClaim(cid, source, text, self.time, rid, kind, value, tense)
            c.origin = origin
            if kind == "attr" and origin == "prédicat" and described.get(rid) not in (None, value):
                c.status = DEFEATED                        # « le cube rouge est bleu » : affirmation incohérente, rejetée
                self.tms.add(self._id("x"), "claim", rule="contradiction interne", status=DEFEATED, claim=cid)
            self.claims[cid] = c; out.append(c)
        self.refresh()
        if self.consolidate_every and len(self.discourse.episodic) % self.consolidate_every == 0: self.consolidate_stable()
        return mentions, out

    def consolidate_stable(self):
        """Identités stabilisées (engagées au seuil des questions, assez anciennes) -> figées ; provenance conservée."""
        n = 0
        for c in self.claims.values():
            if c.frozen or c.entity is None or c.identity_confidence < self.policy.threshold("query"): continue
            if self.time - c.time < self.consolidate_age: continue
            c.frozen = True; n += 1
            ref = self.state.entities_mentioned.get(c.referent)
            if ref is not None and ref.frozen is None: ref.frozen = c.entity
        self.consolidations += n
        return n

    def thaw(self, entity):
        """Une contradiction sur une entité dégèle les identités consolidées qui la concernent."""
        for c in self.claims.values():
            if c.frozen and c.entity == entity: c.frozen = False
        for r in self.state.entities_mentioned.values():
            if r.frozen == entity: r.frozen = None

    # ---- identité : UNE fonction pour ingestion, questions et commandes --------------------
    def identity(self, rid, source=USER, context="query", mention=None):
        """Distribution calibrée P(entité | preuves) pour un référent de discours."""
        dist = self.discourse.dists.get(rid, {})
        ref = self.state.entities_mentioned.get(rid)
        if not dist or ref is None: return {}, None
        special = [e for e in dist if e in (NONEXISTENT, HYPOTHETICAL)]
        if special:
            return {special[0]: 1.0}, IdentityEvidence(mention or rid, special[0], source, 1.0, [], [], self.time, 1.0, {special[0]: 1.0})
        typed = self.entities(ref.type) or list(dist)
        cands = sorted(set(typed) | set(dist))
        presupposes = ref.introduced in ("def", "other") and ref.type is not None
        n = len(cands)
        prior = torch.ones(n)
        compat = torch.tensor([dist.get(e, 0.0) for e in cands])
        temporal = torch.tensor([1.0 if self.discourse._compatible(ref, e) else 0.0 for e in cands])
        if presupposes:                                   # ∅ : a priori explicite, vraisemblance neutre
            cands = cands + [NONE]; n += 1
            prior = torch.cat([torch.full((n - 1,), (1 - PRESUPPOSITION_FAILURE) / max(1, n - 1)), torch.tensor([PRESUPPOSITION_FAILURE])])
            compat = torch.cat([compat / compat.sum().clamp_min(1e-12), torch.tensor([1.0])])
            temporal = torch.cat([temporal, torch.tensor([1.0])])
        support = torch.zeros(n)
        k = ATTR_DOMAIN["color"]
        described = ref.attrs.get("attr")
        supporting, contradicting = [], []
        if described:
            for i, e in enumerate(cands):
                if e == NONE: continue
                b = self.belief.graph.belief(("color", e))
                if b.value is None: continue
                q = b.posterior if b.value == described else (1 - b.posterior) / (k - 1)
                support[i] = math.log(max(q, 1e-6) * k)                   # rapport à l'a priori uniforme
                if presupposes and b.value == described:                  # sous ∅, cette croyance devrait être fausse
                    support[-1] += math.log(max(1 - b.posterior, 1e-6) * k / (k - 1))
                (supporting if b.value == described else contradicting).append(e)
        # relations DESCRIPTIVES (« le cube sur la table ») : même vraisemblance que l'attribut décrit
        for st, tense, (rel, place) in ref.states:
            if not isinstance(place, str) or tense not in ("now", "past"): continue
            kl = self.place_domain()
            for i, e in enumerate(cands):
                if e == NONE: continue
                b = self.belief.graph.belief(("loc", e, 0 if tense == "past" else "now"))
                if b.status not in (FACT, BELIEF): continue
                match = b.value == (rel, place)
                support[i] += math.log(max(b.posterior if match else (1 - b.posterior) / (kl - 1), 1e-6) * kl)
                if presupposes and match: support[-1] += math.log(max(1 - b.posterior, 1e-6) * kl / (kl - 1))
        # attributs AFFIRMÉS sur ce référent (preuve douce, pondérée par la fiabilité de chaque locuteur)
        for c in self.claims.values():
            if c.referent != rid or c.kind not in ("attr", "loc") or c.origin != "prédicat" or c.status == DEFEATED: continue
            rc = min(self.reliability.estimate(c.source) if self.source_calibrated else 0.99, 0.99)
            kk = k if c.kind == "attr" else self.place_domain()
            for i, e in enumerate(cands):
                if e == NONE: continue
                key = ("color", e) if c.kind == "attr" else ("loc", e, 0 if c.tense == "past" else "now")
                b = self.belief.graph.belief(key)
                if b.status not in (FACT, BELIEF) or b.supporting == [c.source]: continue
                if c.kind == "loc" and c.tense != "past" and max((ev.time for ev in self.belief.graph.props.get(key, ())), default=0) > c.time:
                    continue                                              # état observé PLUS TARD : l'objet a pu bouger
                lik = rc if b.value == c.value else (1 - rc) / (kk - 1)
                support[i] += math.log(max(lik, 1e-6) * kk)
        self.checks += n
        r = self.reliability.estimate(source) if self.source_calibrated else 1.0
        p = identity_posterior(prior, compat, support, temporal, r)
        if float(p.sum()) <= 0: return {}, None
        p = p / p.sum()
        best = int(p.argmax())
        conf = float(p[best]) if self.calibrator is None else float(self.calibrator(float(p[best])))
        out = {e: float(x) for e, x in zip(cands, p)}
        ev = IdentityEvidence(mention or rid, cands[best], source, round(r, 3), supporting, contradicting, self.time,
                              round(conf, 4), out)
        return out, ev

    # ---- recalcul : identités -> preuves -> croyances -> monde du résolveur (point fixe) -------
    def refresh(self, max_iter=4):
        for _ in range(max_iter):
            self._rebuild_graph()
            self._sync_world()
            self.discourse.solve(self.state.time, reason="croyances mises à jour")
            if not self._attach_claims(): break
        self._rebuild_graph(); self._sync_world(); self._rebuild_ref_graph()

    def _rebuild_ref_graph(self):
        """Preuves rattachées au RÉFÉRENT DE DISCOURS (indépendantes de l'identité monde) : « le cube jaune tombe »
        permet de répondre où il est même si l'on ne sait pas lequel des cubes du monde il est."""
        g = self.ref_graph; g.props.clear()
        latest = {}
        for c in self.claims.values():
            if (c.status == DEFEATED and c.evidence_node is None) or c.contested: continue
            if c.evidence_node is not None and self.tms.nodes[c.evidence_node].status in (DEFEATED, SUSPENDED): continue
            if c.kind == "attr": items = [(("color", c.referent), c.value, c.time)]
            elif c.kind == "loc" and c.tense == "now": items = [(("loc", c.referent, "now"), c.value, c.time)]
            elif c.kind == "event" and c.value == "fall": items = [(("loc", c.referent, "now"), ("on", "floor"), c.time + 0.5)]
            else: items = []
            for key, value, t in items:
                via = "event" if c.kind == "event" else "text"
                if key[0] == "loc":
                    if latest.get((key, c.source), (-1,))[0] > t: continue
                    latest[(key, c.source)] = (t, value)
                    g.props[key] = [e for e in g.props.get(key, []) if e.source != c.source]
                g.add(key, Evidence(c.source, value, t, via))
        for key in list(g.props):                                         # états dépassés par un événement ultérieur
            if key[0] == "loc" and len(g.props[key]) > 1:
                events = [e.time for e in g.props[key] if e.via == "event"]
                if events: g.props[key] = [e for e in g.props[key] if e.time >= max(events)]

    def _attach_claims(self):
        changed = False
        thr = self.policy.threshold("observation")
        for c in self.claims.values():
            if c.frozen or (c.status == DEFEATED and c.evidence_node is None): continue
            dist, ev = self.identity(c.referent, c.source, "observation")
            if ev is None:
                new_e, conf = None, 0.0
            else:
                new_e, conf = ev.entity, ev.confidence
                if new_e in (NONEXISTENT, HYPOTHETICAL, NONE): new_e = None
            commit = new_e is not None and self.policy.decide(conf, "observation") == ANSWER and conf >= thr
            target = new_e if commit else None
            if target is None and c.entity is None: c.identity_confidence = conf; c.best_candidate = new_e; continue
            if target == c.entity and (c.evidence_node is None or abs(conf - c.identity_confidence) < 1e-6): continue
            if target == c.entity and c.evidence_node is not None:
                c.identity_confidence = conf; self.tms.nodes[c.evidence_node].info["weight"] = conf; changed = True; continue
            # révision : l'ancienne identité est défaite, les croyances dépendantes sont suspendues puis recalculées
            if c.evidence_node is not None:
                idn = self.tms.nodes[c.evidence_node].deps[0]
                self.tms.defeat(idn, f"identité révisée : {c.entity} -> {target}")
                self.revisions.append((c.id, c.entity, target, round(c.identity_confidence, 3), round(conf, 3), self.time))
            c.entity, c.identity_confidence, c.evidence_node = target, conf, None
            c.status = SUSPENDED
            if target is not None:
                self.identity_log.append(ev)
                idn = self.tms.add(self._id("id"), "identity", rule=f"{c.referent}->{target}",
                                   entity=target, referent=c.referent, confidence=conf, source=c.source).id
                key = self._key(c.kind, target, c.tense)
                if key is not None:
                    enode = self.tms.add(self._id("ev"), "evidence", deps=[idn], rule="affirmation textuelle",
                                         source=c.source, key=key, value=c.value, weight=conf, claim=c.id, time=c.time)
                    c.evidence_node = enode.id; c.status = JUSTIFIED
                    if c.kind == "event": self._transition(c, target, enode.id)
            changed = True
        return changed

    @staticmethod
    def _key(kind, e, tense):
        if kind == "attr": return ("color", e)
        if kind == "loc": return ("loc", e, 0 if tense == "past" else ("now", "t"))
        if kind == "event": return ("event", e)
        return None

    def _transition(self, c, e, enode):
        """Provenance de transformation : événement -> nouvel état (pas une causalité générale)."""
        if c.value != "fall": return
        before = self.current_loc(e)
        dn = self.tms.add(self._id("dv"), "derived", deps=[enode], rule="tomber -> sur le sol",
                          source=c.source, key=("loc", e, ("now", "t")), value=("on", "floor"), weight=c.identity_confidence,
                          time=c.time + 0.5)
        self.transitions.append((f"fall@{c.time}", e, before, ("on", "floor"), c.time, dn.id))

    def _rebuild_graph(self):
        """Le graphe de preuves est reconstruit depuis les nœuds JUSTIFIED : aucune croyance orpheline."""
        g = self.belief.graph
        g.props.clear()
        latest = {}
        # un état « maintenant » antérieur à une observation fiable plus récente est un état PASSÉ (trajectoire)
        fresh = defaultdict(float)
        for nid, n in self.tms.nodes.items():
            if n.status != JUSTIFIED or n.info.get("key") is None: continue
            k = self._now_key(n)
            # seuls un capteur fiable ou un événement (transformation) remplacent un état ; deux témoignages = conflit
            if k and (n.kind == "derived" or (n.kind == "premise" and self.reliability.estimate(n.info["source"]) >= 0.9)):
                fresh[k] = max(fresh[k], self._time_of(nid, n))
        for nid, n in self.tms.nodes.items():
            if n.kind not in ("premise", "evidence", "derived") or n.status != JUSTIFIED: continue
            key, value, src = n.info["key"], n.info["value"], n.info["source"]
            t = n.info.get("time", 0)
            if n.kind == "premise": key, t = self._premise_key(key), self.records[nid][3]
            if self._now_key(n) and t < fresh.get(self._now_key(n), -1): continue
            if len(key) == 3 and key[2] == ("now", "t"):                  # état courant = dernier état daté
                key = ("loc", key[1], "now")
                if (key, src) in latest and latest[(key, src)][0] > t: continue
                latest[(key, src)] = (t, nid)
            g.add(key, Evidence(src, value, t, "structured" if n.kind == "premise" else "text", None, n.info.get("weight", 1.0)))
            g.props[key][-1].node = nid

    def _time_of(self, nid, n):
        return self.records[nid][3] if n.kind == "premise" else n.info.get("time", 0)

    def _now_key(self, n):
        k = self._premise_key(n.info["key"]) if n.kind == "premise" else n.info["key"]
        if k[0] == "loc" and k[2] in ("now", ("now", "t")): return ("loc", k[1], "now")
        return None

    @staticmethod
    def _premise_key(key):
        if key[0] == "loc": return ("loc", key[1], 0 if key[2] == 0 else "now")
        return key

    def _sync_world(self):
        """Le résolveur lit les CROYANCES (FACT/BELIEF), jamais la vérité ni les observations brutes."""
        for e in self.entities():
            f = self.discourse.world[e]
            for field_, key in (("attr", ("color", e)), ("loc", ("loc", e, "now")), ("loc0", ("loc", e, 0))):
                b = self.belief.graph.belief(key)
                if b.status in (FACT, BELIEF):
                    f[field_] = b.value
                    if field_ == "loc":                                       # date du dernier état connu
                        f["loc_time"] = max((ev.time for ev in self.belief.graph.props.get(key, ())), default=0)
                else: f.pop(field_, None)

    # ---- trajectoires ----------------------------------------------------------------
    def current_loc(self, e):
        b = self.belief.graph.belief(("loc", e, "now"))
        return b.value if b.status in (FACT, BELIEF) else None

    def trajectory(self, e):
        """États datés d'une même entité (l'identité reste stable ; les états changent)."""
        out = []
        for nid, n in self.tms.nodes.items():
            if n.info.get("key") is None or n.status != JUSTIFIED: continue
            key = self._premise_key(n.info["key"]) if n.kind == "premise" else n.info["key"]
            if key[0] == "loc" and key[1] == e:
                t = self.records[nid][3] if n.kind == "premise" else n.info.get("time", 0)
                out.append((t, n.info["value"], n.rule))
        return sorted(out, key=lambda x: x[0])

    # ---- questions ----------------------------------------------------------------------
    def query(self, text, context=None):
        """Question / commande interprétée par le MÊME résolveur ; réponse fondée sur le BeliefState seul."""
        tokens = tokenize(text)
        intent = "command" if tokens and tokens[0] in COMMANDS else "where" if "où" in tokens else \
            "color" if "couleur" in tokens else "check"
        context = context or ("action" if intent == "command" else "query")
        self.time += 1; self.state.time = self.time - 1
        self.discourse.predicate_enabled = False
        try: mentions = self.discourse.process(text)
        finally: self.discourse.predicate_enabled = True
        self.refresh()
        target = next((m for m in mentions if m.kind != "place" and m.referent), None)
        if target is None:
            return EpistemicAnswer(UNKNOWN_ANS, ABSTAIN, phrase=PHRASES["NO_ID"], reason="no_referent")
        dist, ev = self.identity(target.referent, USER, context, target.id)
        if ev is None:
            return EpistemicAnswer(UNKNOWN_ANS, ABSTAIN, phrase=PHRASES["NO_ID"], reason="no_candidate")
        if ev.entity in (NONEXISTENT, NONE) and ev.confidence >= self.policy.threshold("observation"):
            return EpistemicAnswer(UNKNOWN_ANS, ABSTAIN, phrase=PHRASES["INVALID_PREMISE"], reason="INVALID_PREMISE")
        if ev.entity == HYPOTHETICAL:
            return EpistemicAnswer(UNKNOWN_ANS, ABSTAIN, phrase=PHRASES["NO_ID"], reason="hypothetical")
        lin = self.linguistic_uncertainty(tokens, intent)
        if intent == "command" and lin >= 0.5:
            return EpistemicAnswer(UNKNOWN_ANS, ASK, phrase=f"Que veut dire « {tokens[0]} » ici ?", question=f"Que veut dire « {tokens[0]} » ici ?",
                                   reason="linguistic", linguistic_uncertainty=lin, epistemic_uncertainty=1 - ev.confidence)
        a = self._command(target, dist, ev, context) if intent == "command" else self._answer_about(intent, tokens, target, dist, ev, context)
        a.linguistic_uncertainty = lin
        a.epistemic_uncertainty = round(1 - a.confidence * (a.identity_confidence or 1.0), 4) if intent != "command" else round(1 - ev.confidence, 4)
        return a

    def linguistic_uncertainty(self, tokens, intent):
        """Incertitude LINGUISTIQUE (séparée de l'épistémique) : verbe polysémique sans contexte, mots sans sens."""
        unc = 0.0
        if intent == "command":
            verb = tokens[0]
            senses = self.lx.senses(verb) if hasattr(self.lx, "senses") else []
            has_target = any((self.discourse.meaning(t) or "").startswith("rel:") for t in tokens[1:])
            if len(senses) >= 2 and not has_target: unc = max(unc, 1 - 1 / len(senses))     # sens non départagés
        unknown = [t for t in tokens if t.isalpha() and len(t) > 3 and self.discourse.meaning(t) is None
                   and t not in QUESTION_WORDS | COMMANDS | {"est", "était", "quelle", "pourquoi", "sont", "reste", "tombe"}]
        if unknown: unc = max(unc, 0.5)
        return round(unc, 4)

    def _value_distribution(self, key_of, dist):
        """P(valeur) = Σ_e P(e) · P(valeur | croyances sur e) — identité ET croyance, sans choix prématuré."""
        mix, deps, conflict = defaultdict(float), [], 0.0
        for e, pe in dist.items():
            if pe < 1e-6: continue
            if e == NONE: mix[None] += pe; continue
            b = self.belief.graph.belief(key_of(e))
            if b.value is None: mix[None] += pe; continue
            mix[b.value] += pe * b.posterior; mix[None] += pe * (1 - b.posterior)
            if b.refuting and b.supporting:
                conflict += pe * (1 - b.posterior) if b.posterior < 0.9 else 0.0
            deps += [getattr(ev, "node", None) for ev in self.belief.graph.props.get(key_of(e), ()) if getattr(ev, "node", None)]
        return mix, deps, conflict

    def _answer_about(self, intent, tokens, target, dist, ev, context):
        rid = target.referent
        if intent == "where": key_of = lambda e: ("loc", e, "now")
        elif intent == "color": key_of = lambda e: ("color", e)
        else:
            claim = self._check_claim(tokens, target)
            if claim is None: return EpistemicAnswer(UNKNOWN_ANS, ABSTAIN, phrase=PHRASES[UNKNOWN_ANS], reason="no_claim")
            kind, value = claim
            key_of = (lambda e: ("color", e)) if kind == "attr" else (lambda e: ("loc", e, "now"))
        n_cands = sum(1 for e, p in dist.items() if p > 0.05 and e != NONE)
        id_conf = ev.confidence
        mix, deps, conflict = self._value_distribution(key_of, dist)
        known = {v: p for v, p in mix.items() if v is not None}
        ref = self.ref_graph.belief(key_of(rid))
        if ref.value is not None:
            best_w = max(known.values()) if known else 0.0
            if best_w >= 0.65 and ref.posterior >= 0.65 and max(known, key=known.get) != ref.value:
                conflict = max(conflict, 0.5)
            elif ref.posterior > best_w:                              # référent de discours mieux établi que l'entité monde
                known = {ref.value: ref.posterior}
                id_conf = 1.0                                         # la question vise le référent, déjà identifié
                if ref.refuting and ref.supporting and ref.posterior < 0.9: conflict = max(conflict, 1 - ref.posterior)
        if self.answer_calibrator is not None and known:
            known = {v: float(self.answer_calibrator(p)) for v, p in known.items()}
        if intent == "check":
            p_true = known.get(value, 0.0); p_false = sum(p for v, p in known.items() if v != value)
            v_best, q = (True, p_true) if p_true >= p_false else (False, p_false)
        else:
            v_best, q = max(known.items(), key=lambda t: t[1]) if known else (None, 0.0)
        risks = self.policy.risks(q, context, can_ask=n_cands > 1 and id_conf < self.policy.threshold(context))
        action = min(risks, key=lambda k: (risks[k], [ANSWER, ASK, VERIFY, ABSTAIN].index(k)))
        a = EpistemicAnswer(UNKNOWN_ANS, action, confidence=round(q, 4), entity=ev.entity, identity_confidence=id_conf,
                            dependencies=sorted(set(deps)), risks={k: round(v, 3) for k, v in risks.items()})
        shown = v_best if intent != "check" else ("oui" if v_best else "non")
        if action == ANSWER:
            if intent == "check":
                a.status, a.value = (SUPPORTED, "SUPPORTED") if v_best else (REFUTED, "REFUTED")
            else:
                a.status, a.value = SUPPORTED, self._fmt(v_best)
            a.phrase = PHRASES[a.status if intent != "check" or v_best else REFUTED].format(v=self._fmt(v_best) if intent != "check" else shown)
        elif action == ASK:
            a.question = self.clarification_question(dist)
            a.phrase, a.reason = a.question, "identity"
        else:
            if conflict > 0.2: a.status, a.reason, a.phrase = CONFLICTED, CONFLICTED, PHRASES[CONFLICTED].format(v=self._fmt(v_best))
            elif q >= self.policy.threshold("observation") and v_best is not None:
                a.status, a.phrase = LIKELY, PHRASES[LIKELY].format(v=self._fmt(v_best) if intent != "check" else shown)
                a.value = self._fmt(v_best) if intent != "check" else ("SUPPORTED" if v_best else "REFUTED")
            elif id_conf < self.policy.threshold(context) and n_cands > 1: a.phrase, a.reason = PHRASES["NO_ID"], "identity"
            else: a.phrase, a.reason = PHRASES[UNKNOWN_ANS], "evidence"
        return a

    @staticmethod
    def _fmt(v):
        if isinstance(v, tuple): return f"{v[0]}:{v[1]}"
        return v

    def _check_claim(self, tokens, target):
        meanings = [self.discourse.meaning(t) for t in tokens]
        start = next((i for i in range(target.position + 1, len(tokens)) if tokens[i] in ("est", "était", "-il", "-elle")), len(tokens))
        for i in range(start + 1, len(tokens)):
            m = meanings[i]
            if m and m.startswith("color:"): return "attr", m.split(":")[1]
            if m and m.startswith("rel:"):
                cls = [token_class(x) for x in meanings]
                place, _ = self.discourse._place_after(tokens, meanings, cls, i + 1)
                if isinstance(place, str): return "loc", (m.split(":")[1], place)
        return None

    def _command(self, target, dist, ev, context):
        p = ev.confidence if ev.entity != NONE else 0.0
        n_cands = sum(1 for e, x in dist.items() if x > 0.05 and e != NONE)
        risks = self.policy.risks(p, context, can_ask=n_cands > 1)
        action = min(risks, key=lambda k: (risks[k], [ANSWER, ASK, VERIFY, ABSTAIN].index(k)))
        a = EpistemicAnswer(SUPPORTED if action == ANSWER else UNKNOWN_ANS, action, value=ev.entity if action == ANSWER else None,
                            confidence=p, entity=ev.entity, identity_confidence=p, risks={k: round(v, 3) for k, v in risks.items()})
        if action == ASK: a.question = self.clarification_question(dist); a.phrase = a.question
        elif action == ANSWER: a.phrase = f"J'agis sur {ev.entity}."
        else: a.phrase, a.reason = PHRASES["NO_ID"], "identity"
        return a

    # ---- clarification et observation active ----------------------------------------------
    def clarification_question(self, dist):
        """Question discriminante construite depuis les croyances : teinte, sinon position, sinon nombre."""
        cands = [e for e, p in sorted(dist.items(), key=lambda t: -t[1]) if p > 0.05 and e != NONE]
        for key_of, word in ((lambda e: ("color", e), lambda v: COLOR_WORD.get(v, v)),
                             (lambda e: ("loc", e, "now"), lambda v: f"celui {PLACE_WORD.get(v[0], v[0])} {PLACE_WORD.get(v[1], v[1])}")):
            vals = [self.belief.graph.belief(key_of(e)) for e in cands]
            if all(b.status in (FACT, BELIEF) for b in vals) and len({str(b.value) for b in vals}) == len(vals):
                return "Lequel : " + " ou ".join(("le " + word(b.value)) if key_of(cands[0])[0] == "color" else word(b.value) for b in vals) + " ?"
        return f"Lequel des {len(cands)} ?"

    def clarify(self, mention_id, answer_text):
        self.discourse.apply_clarification(mention_id, answer_text)
        self.refresh()

    def information_request(self, rid):
        """Quelle observation (couleur / position de quel candidat) réduit le plus l'entropie d'identité ?"""
        dist, ev = self.identity(rid)
        cands = [e for e, p in dist.items() if p > 1e-6 and e != NONE]
        if len(cands) < 2: return None, 0.0
        p = torch.tensor([dist[e] for e in cands]); p = p / p.sum()
        h0 = float(-(p * p.clamp_min(1e-12).log()).sum())
        ref = self.state.entities_mentioned[rid]
        best, gain = None, 0.0
        for field_, key_of in (("attr", lambda e: ("color", e)), ("loc", lambda e: ("loc", e, "now"))):
            desc = ref.attrs.get("attr") if field_ == "attr" else next((st[2] for st in ref.states if st[1] == "now"), None)
            if desc is None: continue
            for j, c in enumerate(cands):
                if self.belief.graph.belief(key_of(c)).status in (FACT, BELIEF): continue
                # issue « c a la valeur décrite » : c seul reste ; sinon c est exclu (vectorisé sur les deux issues)
                p_yes = float(p[j])
                keep_no = p.clone(); keep_no[j] = 0
                h_no = float(-(keep_no / keep_no.sum() * (keep_no / keep_no.sum()).clamp_min(1e-12).log()).sum()) if keep_no.sum() > 0 else 0.0
                g = h0 - (1 - p_yes) * h_no
                if g > gain + 1e-9: best, gain = (field_, c), g
        return best, gain

    def request_observation(self, rid, oracle, max_requests=3):
        """Demande des observations (simulateur capteur) jusqu'à engagement ou budget épuisé."""
        used = 0
        for _ in range(max_requests):
            dist, ev = self.identity(rid)
            if ev and ev.confidence >= self.policy.threshold("query"): break
            req, g = self.information_request(rid)
            if req is None or g <= 0: break
            field_, e = req
            value = oracle(field_, e)
            if value is None: break
            key = ("color", e) if field_ == "attr" else ("loc", e, "now")
            self.observe("capteur", key, value); used += 1; self.observation_requests += 1
        return used

    # ---- contradictions et auto-correction ----------------------------------------------------
    def contradict(self, key, value, source="capteur"):
        """Nouvelle observation fiable : les preuves textuelles incompatibles sont DÉFAITES (pas seulement notées)."""
        if key[0] != "type": self.thaw(key[1])
        rid = self.observe(source, key, value)
        k = self._premise_key(key)
        defeated = []
        if self.reliability.estimate(source) >= 0.9:
            for nid, n in list(self.tms.nodes.items()):
                if n.kind in ("evidence", "derived") and n.status == JUSTIFIED:
                    nk = n.info["key"]; nk = ("loc", nk[1], "now") if len(nk) == 3 and nk[2] == ("now", "t") else nk
                    state_change = k[0] == "loc" and n.info.get("time", 0) < self.time     # changement d'état, pas contradiction
                    if nk == k and n.info["value"] != value and not state_change:
                        self.tms.defeat(nid, f"contredit par {source}"); defeated.append(nid)
                        self.source_revision(n.info.get("source"), n.info.get("key", k))
        self.refresh()
        return rid, defeated

    def source_revision(self, source, key=None):
        """Auto-correction par révision de SOURCE : une affirmation réfutée par une observation fiable est une erreur
        dans l'historique de la source ; son posterior baisse et toutes ses autres preuves sont repondérées (localement)."""
        if not source or source == USER: return
        before = self.reliability.estimate(source)
        if hasattr(self.reliability, "observe"): self.reliability.observe(source, False, key[0] if key else None)
        else: self.reliability.stats[source][1] += 1
        self.source_revisions.append((source, round(before, 3), round(self.reliability.estimate(source), 3)))

    def self_correct(self, answer: EpistemicAnswer, text):
        """Réponse contredite -> dépendances -> identités faibles / sources peu fiables suspendues -> recalcul."""
        trace = []
        for d in answer.dependencies:
            for n in self.tms.support_of(d):
                if n.kind == "identity" and n.info.get("confidence", 1) < self.policy.threshold("query") and n.status == JUSTIFIED:
                    self.tms.set(n.id, SUSPENDED, "identité faible"); n.info["own"] = SUSPENDED; trace.append(("identity", n.id))
                if n.kind in ("premise", "evidence") and self.reliability.estimate(n.info.get("source", USER)) < 0.75 and n.status == JUSTIFIED:
                    self.tms.set(n.id, SUSPENDED, "source peu fiable"); n.info["own"] = SUSPENDED; trace.append(("source", n.id))
        self.tms.propagate()
        self._rebuild_graph(); self._sync_world()
        new = self.query(text)
        return new, trace

    def minimal_revision(self, answer: EpistemicAnswer, text, max_steps=4):
        """Réponse contredite (sans nouvelle observation) : on remonte le graphe de dépendances, on classe les causes
        possibles (identité, source, règle d'inférence) par suspicion = 1 − confiance, et on ne révise QUE la plus
        suspecte ; on recalcule ; on continue seulement si la réponse n'a pas changé. Pas de retour arrière global."""
        cands = {}
        for d in answer.dependencies:
            for n in self.tms.support_of(d):
                if n.status != JUSTIFIED: continue
                if n.kind == "identity": conf, cause = n.info.get("confidence", 1.0), "identity"
                elif n.kind in ("evidence", "premise"): conf, cause = self.reliability.estimate(n.info.get("source", USER)), "source"
                elif n.kind == "derived": conf, cause = n.info.get("weight", 1.0), "inference_rule"
                else: continue
                cands[n.id] = (1 - conf, cause, n)
        ranked = sorted(cands.values(), key=lambda t: -t[0])
        total = len({x.id for d in answer.dependencies for x in self.tms.support_of(d)})
        revised, new = [], answer
        for susp, cause, n in ranked[:max_steps]:
            self.tms.set(n.id, SUSPENDED, f"contestation : cause suspectée ({cause})"); n.info["own"] = SUSPENDED
            if n.info.get("claim") in self.claims: self.claims[n.info["claim"]].contested = True
            revised.append((n.id, cause, round(susp, 3)))
            self.tms.propagate(); self._rebuild_graph(); self._sync_world(); self._rebuild_ref_graph()
            new = self.query(text)
            if (new.status, new.value) != (answer.status, answer.value): break
        return new, {"revised": revised, "support_size": total, "cause": revised[0][1] if revised else None}

    # ---- mémoire -------------------------------------------------------------------------
    def consolidate(self):
        """Croyances stabilisées (FACT) -> résumé compact ; la provenance est gardée sous forme d'identifiants."""
        summary = {}
        for key in list(self.belief.graph.props):
            b = self.belief.graph.belief(key)
            if b.status == FACT:
                nodes = [getattr(e, "node", None) for e in self.belief.graph.props[key]]
                summary[key] = {"value": b.value, "p": round(b.posterior, 3), "sources": sorted(set(b.supporting)),
                                "provenance": [x for x in nodes if x]}
        return summary

    def memory_size(self):
        return {"tms_nodes": len(self.tms.nodes), "claims": len(self.claims), "consolidated": len(self.consolidate()),
                "episodic_sentences": len(self.discourse.episodic)}

"""Entitlement to assert (ClaimEntitlement), separate from the probability of truth.

Each answer produces an epistemic provenance: claim, inference level, observations, rules, reliabilities,
identity assumptions, lexical assumptions, remaining alternatives. Two distinct uncertainties:
  - epistemic: is the world established? (belief posterior, conflicts, identity);
  - linguistic: was the sentence understood? (word meanings, grammatical margin).
Decision: ASSERTED (L0/L1, or L2 very confident and diversified) | QUALIFIED | ASKED | WITHHELD.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.independent_agent import ABSTAIN
from core.knowledge_boundary import L0, L1, L2, L3, LEVEL_NAMES

ASSERTED, QUALIFIED, ASKED, WITHHELD = "ASSERTED", "QUALIFIED", "ASKED", "WITHHELD"
SAFE_WORD_SOURCES = {"grounded", "innate", "function", "forced"}


@dataclass
class EpistemicProvenance:
    claim: str
    level: int
    supporting_observations: List[str] = field(default_factory=list)
    inference_rules: List[str] = field(default_factory=list)
    source_reliabilities: Dict[str, float] = field(default_factory=dict)
    identity_assumptions: List[str] = field(default_factory=list)
    lexical_assumptions: Dict[str, float] = field(default_factory=dict)
    remaining_alternatives: Dict[str, float] = field(default_factory=dict)
    belief_probability: float = 0.0
    identity_uncertainty: float = 0.0
    linguistic_uncertainty: float = 0.0
    epistemic_uncertainty: float = 0.0
    source_diversity: int = 0
    source_lower: float = 1.0          # credible lower bound of the most reliable source that supports the belief
    single_source: bool = False

    def summary(self):
        return {"claim": self.claim, "level": f"L{self.level} ({LEVEL_NAMES[self.level]})",
                "observations": self.supporting_observations, "rules": self.inference_rules,
                "reliabilities": {k: round(v, 3) for k, v in self.source_reliabilities.items()},
                "identity": self.identity_assumptions, "lexical": {k: round(v, 3) for k, v in self.lexical_assumptions.items()},
                "alternatives": {str(k): round(v, 3) for k, v in self.remaining_alternatives.items()},
                "epistemic_uncertainty": round(self.epistemic_uncertainty, 3),
                "linguistic_uncertainty": round(self.linguistic_uncertainty, 3)}


def evidence_level(agent, key):
    """Level of the evidence that establishes a belief: L0 direct, L1 explicit elimination, L2 probabilistic identity."""
    b = agent.belief.graph.belief(key)
    if b.value is None: return L0, "inconnue"
    lvl, why = L0, "directe"
    for s, ev in agent.belief.graph.votes(key).items():
        if ev.value != b.value: continue
        rule = getattr(agent, "rule_of", {}).get(id(ev))
        if rule is not None: 
            if rule.level > lvl: lvl, why = rule.level, "élimination explicite"
        elif "|elim" in s or ev.via == "description" or getattr(ev, "weight", 1.0) < 1.0:
            lvl, why = L2, "identité jointe probabiliste"
    return lvl, why


def provenance(agent, ans, lexical_posterior=None) -> EpistemicProvenance:
    """Provenance of an answer of the grounded / unified agent (read from its current belief state)."""
    g = agent.belief.graph
    level = L0
    obs, rules, rel, ids, alts, supporters = [], [], {}, [], {}, set()
    weights = [1.0]
    p_belief = 1.0 if ans.intent == "command" else 0.0
    for e in ans.entries:
        p_belief = max(p_belief, e.posterior) if ans.intent != "command" else min(p_belief, e.posterior)
        votes = g.votes(e.key)
        for s, ev in votes.items():
            base = s.split("|")[0]
            rel[base] = agent.reliability.estimate(base)
            obs.append(f"{s}:{ev.via}:{e.key}={ev.value}")
            if ev.value != e.value: alts[ev.value] = alts.get(ev.value, 0.0) + agent.reliability.estimate(base)
            if ev.value == e.value:
                supporters.add(base)
                weights.append(getattr(ev, "weight", 1.0))
                rule = getattr(agent, "rule_of", {}).get(id(ev))
                if rule is not None:
                    rules.append(f"EliminateCandidates({rule.descriptor}->{rule.remaining}, L{rule.level})")
                    level = max(level, rule.level)
                elif "|elim" in s or ev.via == "description" or getattr(ev, "weight", 1.0) < 1.0:
                    rules.append(f"identité jointe probabiliste ({s})"); level = max(level, L2)
                    ids.append(f"{e.key[1]} via {s} (P={getattr(ev, 'weight', 1.0):.3f})")
    if not ans.entries and ans.intent != "command": p_belief = 0.0
    # premises of the referent identification: established colors of the objects of the same type (if the question describes a color)
    if any(str(m).startswith("color:") for m in ans.word_meanings.values()):
        for e in ans.entries:
            o = e.key[1] if len(e.key) > 1 else None
            typ = agent.belief.inventory.get(o)
            for m in (agent.belief.objects_of(typ) if typ else []):
                lvl, why = evidence_level(agent, ("color", m))
                if lvl > L0:
                    level = max(level, lvl); rules.append(f"prémisse d'identification color({m}) : {why}")
    idp = ans.features.get("identity_p")
    if idp is not None:
        qrule = ans.features.get("identity_rule")
        if qrule is not None:                                    # identity of the question established by explicit elimination
            ids.append(f"référence de la question par élimination ({qrule.descriptor} -> {qrule.remaining})")
            rules.append(f"EliminateCandidates({qrule.descriptor}->{qrule.remaining}, L{qrule.level})"); level = max(level, qrule.level)
            weights.append(qrule.confidence)
        else:
            ids.append(f"référence de la question (P={idp:.3f}, jointe probabiliste)"); weights.append(idp); level = max(level, L2)
            rules.append("coréférence probabiliste à la question")
    if ans.entries and len(ans.entries) > 1 and ans.intent != "command": level = max(level, L1)   # composition of beliefs
    # lexical assumptions
    lex = {}
    for t in agent._ctx if hasattr(agent, "_ctx") else []:
        m, src, conf = agent.word_meaning(t)
        if src in SAFE_WORD_SOURCES or m is None: continue
        if src in ("neural", "analogy"): level = L3          # an analogy is never a fact
        p = conf
        if lexical_posterior is not None and src == "grounded_obs": p = lexical_posterior(t, m)
        lex[t] = p
    lin = 1.0
    for p in lex.values(): lin *= p
    lin_unc = 1.0 - lin
    if "grammar_ambiguity" in ans.flags: lin_unc = max(lin_unc, 0.5)
    id_unc = 1.0 - min(weights)
    epi_unc = 1.0 - p_belief * (1.0 - id_unc)
    lower, single = 1.0, False
    model = agent.reliability
    if hasattr(model, "record") and supporters and ans.intent != "command":
        groups = {model.group_of(s) for s in supporters}
        single = len(groups) == 1
        lows = []
        for s in supporters:
            a_, b_ = model.posterior(s)
            from core.source_reliability import beta_quantiles
            lows.append(beta_quantiles(a_, b_, (0.1,))[0])
        lower = max(lows)
    return EpistemicProvenance(ans.output, level, obs, rules, rel, ids, lex, alts, p_belief, id_unc, lin_unc, epi_unc,
                               len(supporters), lower, single)


@dataclass
class EntitlementPolicy:
    assert_threshold: float = 0.9          # L0 / L1
    l2_threshold: float = 0.97             # L2: asserted only if allow_l2 (otherwise "likely" at best)
    allow_l2: bool = False
    qualify_threshold: float = 0.67
    max_linguistic_uncertainty: float = 0.1
    min_single_source_lower: float = 0.75   # a belief that rests on ONE uncertain source is not asserted


class ClaimEntitlement:
    def __init__(self, policy: EntitlementPolicy = None): self.p = policy or EntitlementPolicy()

    def claim_probability(self, prov: EpistemicProvenance):
        return prov.belief_probability * (1 - prov.identity_uncertainty) * (1 - prov.linguistic_uncertainty)

    def decide(self, ans, prov: EpistemicProvenance):
        if ans.output == "ASK": return ASKED
        if ans.output in ABSTAIN: return WITHHELD
        q = self.claim_probability(prov)
        if prov.linguistic_uncertainty > self.p.max_linguistic_uncertainty and q < self.p.l2_threshold:
            return QUALIFIED if q >= self.p.qualify_threshold else WITHHELD
        if ans.intent == "command" or ans.output in ("REJECT", "NONE", "INVALID_PREMISE", "UNKNOWN_CAUSE"):
            if prov.level <= L1 or (self.p.allow_l2 and q >= self.p.l2_threshold): return ASSERTED
            return QUALIFIED if q >= self.p.qualify_threshold else WITHHELD
        if prov.single_source and prov.source_lower < self.p.min_single_source_lower and prov.level > L0 - 1 \
                and ans.intent != "command":
            return QUALIFIED if q >= self.p.qualify_threshold else WITHHELD          # source risk
        if prov.level <= L1 and q >= self.p.assert_threshold: return ASSERTED
        if self.p.allow_l2 and prov.level == L2 and q >= self.p.l2_threshold and prov.source_diversity >= 2: return ASSERTED
        if q >= self.p.qualify_threshold: return QUALIFIED
        return ASKED if prov.identity_uncertainty > 0.3 else WITHHELD

    @staticmethod
    def output(ans, decision):
        """Evaluable form: only an ASSERTED claim keeps the value; QUALIFIED stays uncommitted (LIKELY).
        A raw abstention is rendered as is (no gain from a mere change of label)."""
        if decision == ASSERTED or ans.output in ABSTAIN: return ans.output
        if decision == ASKED: return "ASK"
        return "UNKNOWN"

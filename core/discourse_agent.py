"""Agent ancré + identité jointe sur les observations d'un monde (coréférence des descriptions).

Chaque description (type, couleur) employée par les sources est un référent de discours présumé unique.
Les correspondances descriptions -> objets (injectives) sont énumérées et pondérées par les postérieures de
couleur du graphe de preuves ; un objet non désigné ne peut pas porter une couleur déjà décrite (unicité).
Engagement seulement au-dessus du seuil ; sinon la référence reste ouverte. Calcul vectorisé.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass

import torch

from core.grounded import GroundedAgent, GroundedConfig
from core.independent_agent import ABSTAIN, FACT, BELIEF, NP, tokenize

COLORS = ("red", "blue", "green", "yellow", "courses", "voyage", "travail")


@dataclass
class DiscourseConfig(GroundedConfig):
    joint_reference: bool = True
    reference_revision: bool = True
    ingest_threshold: float = 0.9
    query_threshold: float = 1.01          # désactivé : +FC mesuré (5 %) sur 3 graines ; les questions restent strictes
    elimination: bool = False
    separate_channel: bool = True


class DiscourseAgent(GroundedAgent):
    duplicate_prior = 0.0
    def __init__(self, config: DiscourseConfig = None, seed=0):
        super().__init__(config or DiscourseConfig(), seed)
        self._descriptors = defaultdict(set)
        self._mapping = {}
        self._allow_query_mapping = False
        self.identity_log = []

    # ---- postérieure de couleur d'un objet (graphe de preuves) ----------------------------
    def color_distribution(self, o, domain):
        votes = self.belief.graph.votes(("color", o))
        k = len(domain)
        scores = torch.zeros(k)
        for s, e in votes.items():
            if e.value in domain:
                r = min(self.reliability.estimate(s), 0.99)
                scores[domain.index(e.value)] += math.log(r * (k - 1) / (1 - r))
        return torch.softmax(scores, 0)

    def solve_identity(self):
        """Correspondances descriptions -> objets par type, vectorisées ; marginales P(description -> objet)."""
        self._mapping = {}
        for typ, colors in self._descriptors.items():
            objs = self.belief.objects_of(typ)
            if len(objs) < 2 or not colors: continue
            domain = [c for c in COLORS if c in colors or True][:7]
            dists = torch.stack([self.color_distribution(o, domain) for o in objs])      # [objets, couleurs]
            desc = sorted(colors)
            if len(desc) > len(objs): continue
            perms = torch.tensor(list(itertools.permutations(range(len(objs)), len(desc))))   # [A, D]
            ci = torch.tensor([domain.index(c) for c in desc])
            w = dists[perms, ci.expand_as(perms)].prod(1)                                     # descripteurs attribués
            assigned = torch.zeros(len(perms), len(objs), dtype=torch.bool)
            assigned.scatter_(1, perms, True)
            dup = self.duplicate_prior                                                          # a priori : deux objets de même couleur
            not_desc = (1 - dists[:, ci].sum(1)) + dup * dists[:, ci].sum(1)                   # objet non désigné : couleur ∉ descripteurs
            w = w * torch.where(assigned, torch.ones_like(assigned, dtype=torch.float), not_desc.clamp_min(0).expand(len(perms), -1)).prod(1)
            if float(w.sum()) <= 0: continue
            w = w / w.sum()
            for d, color in enumerate(desc):
                marg = torch.zeros(len(objs)).index_add_(0, perms[:, d], w)
                p, j = float(marg.max()), int(marg.argmax())
                self._mapping[(typ, color)] = (objs[j], p)

    # ---- ingestion : les descriptions de TOUTES les observations du monde forment des contraintes jointes ----
    def ingest(self, observations, update=True):
        self._descriptors = defaultdict(set); self._mapping = {}
        if self.cfg.joint_reference:
            for ob in observations:
                if not ob.text: continue
                toks = tokenize(ob.text)
                for np in self.grammar.chunks(toks, [self.word_meaning(t)[0] for t in toks]):
                    if np.type and np.color: self._descriptors[np.type].add(np.color)
            self._pending_solve = True
        return super().ingest(observations, update)

    def resolve(self, np, belief=None, discourse=None, strict=True):
        objs, unknown = super().resolve(np, belief, discourse, strict)
        if not self.cfg.joint_reference or not np.color or not np.type or np.pronoun: return objs, unknown
        if len(objs) == 1 and not unknown: return objs, unknown
        if getattr(self, "_pending_solve", False) and self.belief is not None:
            self.solve_identity(); self._pending_solve = False
        hit = self._mapping.get((np.type, np.color))
        if hit is None: return objs, unknown
        o, p = hit
        attested = np.color in self._descriptors.get(np.type, ())
        if not strict and p >= self.cfg.ingest_threshold:
            self.identity_log.append(("ingest", np.type, np.color, o, round(p, 3)))
            if self.cfg.separate_channel: self._eliminated = True        # canal de preuve distinct (fiabilité estimée à part)
            return [o], []
        if strict and attested and (self._allow_query_mapping or p >= self.cfg.query_threshold) and p >= self.cfg.query_threshold:
            self.identity_log.append(("query", np.type, np.color, o, round(p, 3)))
            return [o], []
        return objs, unknown

    # ---- auto-correction par révision de référence ----------------------------------------------
    def _post(self, ans, text, mode):
        ans = super()._post(ans, text, mode)
        if not (self.cfg.reference_revision and ans.output in ABSTAIN and "unknown_entity" in ans.flags): return ans
        if getattr(self, "_revising", False): return ans
        self._revising = True
        self.solve_identity()                                  # révision : toutes les preuves accumulées du monde
        self._allow_query_mapping = True
        try: retry = GroundedAgent.answer(self, text, mode)
        finally: self._allow_query_mapping = False; self._revising = False
        backed = any(e.status in (FACT, BELIEF) for e in retry.entries) or retry.intent == "command"
        if retry.output not in ABSTAIN and backed and "unknown_entity" not in retry.flags:
            ans.reasoning.append("révision de référence : identité jointe recalculée -> " + retry.output)
            ans.output, ans.decision, ans.corrected = retry.output, retry.decision, True
            ans.evidence += retry.evidence; ans.sources += retry.sources; ans.checks += retry.checks
        return ans

"""Distillation du classement des identités (observation ET question, sensible à la source).

L'élève (modèle linéaire, quelques paramètres) classe les candidats à partir de traits LOCAUX peu coûteux :
compatibilité locale (type, attribut, état daté), soutien des croyances, fiabilité de la source, hypothèse ∅,
1/n. Le maître est la formule d'identité complète (recherche jointe vectorisée). WCE décide toujours :
l'élève ne propose que ; sa proposition est acceptée seulement si sa marge est nette ET si la vérification
locale réussit, sinon on revient au maître. On mesure l'accord des DÉCISIONS (engager / ne pas engager, entité)
et le coût (évaluations de candidats).
"""
from __future__ import annotations

import torch

from core.unified import NONE, UnifiedCognitiveState

N_FEATS = 5


class IdentityStudent(torch.nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.lin = torch.nn.Linear(N_FEATS, 1)

    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())

    def forward(self, feats, mask):                      # [B, C, F], [B, C] -> probabilités [B, C]
        s = self.lin(feats).squeeze(-1).masked_fill(~mask, -1e9)
        return torch.softmax(s, -1)

    def fit(self, examples, epochs=300, lr=0.05):
        feats, mask, y = pad(examples)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            p = self(feats, mask)
            loss = -(p.gather(1, y.view(-1, 1)).clamp_min(1e-9).log()).mean()
            loss.backward(); opt.step()
        return self


def pad(examples):
    C = max(len(f) for f, _, _ in examples)
    feats = torch.zeros(len(examples), C, N_FEATS); mask = torch.zeros(len(examples), C, dtype=torch.bool)
    y = torch.zeros(len(examples), dtype=torch.long)
    for i, (f, target, _) in enumerate(examples):
        feats[i, :len(f)] = f; mask[i, :len(f)] = True; y[i] = target
    return feats, mask, y


def local_features(u: UnifiedCognitiveState, rid, source):
    """Traits locaux (sans recherche jointe) pour chaque candidat ; renvoie (candidats, traits, coût)."""
    ref = u.state.entities_mentioned[rid]
    cands = u.entities(ref.type) or u.entities()
    presup = ref.introduced in ("def", "other") and ref.type is not None
    r = u.reliability.estimate(source)
    rows, names = [], []
    for e in cands:
        ok = 1.0 if u.discourse._compatible(ref, e) else 0.0
        b = u.belief.graph.belief(("color", e)); desc = ref.attrs.get("attr")
        sup = 0.0 if (desc is None or b.value is None) else (b.posterior if b.value == desc else -b.posterior)
        rows.append([ok, sup, 0.0, r, 1.0 / max(1, len(cands))]); names.append(e)
    if presup: rows.append([1.0, 0.0, 1.0, r, 1.0 / max(1, len(cands))]); names.append(NONE)
    return names, torch.tensor(rows, dtype=torch.float), len(cands)


def collect(lexicon, bench, run_scenario):
    """Exemples (traits locaux, choix du maître, décision du maître) sur un banc (partition d'entraînement)."""
    examples = []
    orig = UnifiedCognitiveState.identity

    def hooked(self, rid, source="user", context="query", mention=None):
        dist, ev = orig(self, rid, source, context, mention)
        if ev is not None and rid in self.state.entities_mentioned and ev.entity in dist:
            names, f, _ = local_features(self, rid, source)
            if ev.entity in names and len(names) >= 1:
                commit = ev.confidence >= self.policy.threshold(context)
                examples.append((f, names.index(ev.entity), (ev.entity, commit, context, len(dist), self.discourse.checks)))
        return dist, ev
    UnifiedCognitiveState.identity = hooked
    try:
        for scn in bench: run_scenario(lexicon, scn)
    finally:
        UnifiedCognitiveState.identity = orig
    return examples


def evaluate(student, examples, margin=0.9):
    """Élève propose -> vérification locale -> accepté si net, sinon maître. Accord des décisions et coût."""
    feats, mask, y = pad(examples)
    with torch.no_grad(): p = student(feats, mask)
    top = p.argmax(1); conf = p.max(1).values
    ok_local = feats[torch.arange(len(examples)), top, 0] > 0.5
    accept = (conf >= margin) & ok_local
    n_cands = mask.sum(1).float()
    agree_entity = (top == y)
    # décision finale : acceptée -> entité de l'élève (engagement si le maître aurait engagé : même seuil) ; sinon maître
    final_ok = torch.where(accept, agree_entity, torch.ones_like(agree_entity))
    teacher_cost = sum(max(1, ex[2][3]) for ex in examples) + float(n_cands.sum())      # candidats + recherche jointe
    student_cost = float(n_cands.sum()) + float(accept.sum()) + sum(max(1, ex[2][3]) + 0 for ex, a in zip(examples, accept) if not a)
    return {"decisions": len(examples), "accepted_from_student": int(accept.sum()),
            "fallback_to_teacher": int((~accept).sum()), "decision_agreement": round(float(final_ok.float().mean()), 4),
            "student_top1_agreement": round(float(agree_entity.float().mean()), 4),
            "changed_decisions": int((~final_ok).sum()), "teacher_cost": round(teacher_cost, 1),
            "student_cost": round(student_cost, 1), "parameters": student.parameter_count}


def distill(lexicon, train_bench, test_bench, run_scenario, seed=0):
    train = collect(lexicon, train_bench, run_scenario)
    test = collect(lexicon, test_bench, run_scenario)
    student = IdentityStudent(seed).fit(train)
    return {"train_examples": len(train), **evaluate(student, test)}

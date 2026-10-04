"""Sous-jalon Objective Alignment & Conditional Optimization — module d'ÉVALUATION, dev uniquement.

Trois régimes d'entrée, séparés :
  A  vrais atomes + vrais paramètres            (contrôle : optimiser devrait être inutile)
  B  vrais atomes + paramètres perturbés        (faible / moyen / fort)
  C  atomes découverts + paramètres approximatifs (le cas réellement intéressant)

Pour chaque candidat : signaux internes AVANT ajustement, puis, pour chaque objectif observable, l'état initial, le
meilleur état rencontré et l'état final, avec l'accord fonctionnel réel de chacun (vérité : MESURE seulement).
Les politiques (toujours / jamais / garde-fou / porte conditionnelle) sont ensuite évaluées sur ces états enregistrés.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List

import torch

from core.boundary_program import BNode, predict
from core.deep_fit import DT, FitData
from core.deep_generation import generate_candidates
from core.deep_optgate import OBJECTIVES, hard_errors, optimize, profiled, signals
from core.deep_search import DeepSearch
from core.deep_structure import algebraic_form, n_atoms, normalize, skel_key, structurally_equivalent

PERTURBATIONS = {"low": 0.03, "medium": 0.10, "high": 0.30}
E2E_POLICIES = {"always": None, "never": "never", "validated": "profiled", "gate": "gate"}
THRESHOLDS = (0.90, 0.95, 0.97, 0.99)


def _accord(task):
    from core.deep_world import full_region
    F = full_region(4000, torch.Generator().manual_seed(11)); lF = task.world.label(F, 1.0); return lambda p: float((predict(p, F) == lF).float().mean())


def _spearman(a, b):
    n = len(a)
    if n < 4: return None
    def rk(v):
        o = sorted(range(n), key=lambda i: v[i]); r = [0.0] * n
        for p, i in enumerate(o): r[i] = float(p)
        return r
    ra, rb = rk(a), rk(b); ma, mb = sum(ra) / n, sum(rb) / n; sa = math.sqrt(sum((x - ma) ** 2 for x in ra)); sb = math.sqrt(sum((x - mb) ** 2 for x in rb))
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (sa * sb) if sa > 0 and sb > 0 else None


def candidates_for(task, seed=0):
    """Candidats des trois régimes pour une tâche : [(régime, niveau, programme, erreur de paramètres initiale)]."""
    from core.deep_reachability import leaves
    tr = normalize(task.world.truth); out = [("A", "true", tr, 0.0)]; g = torch.Generator().manual_seed(seed + 17); p0 = torch.tensor(tr.param_vector(), dtype=DT)
    for name, s in PERTURBATIONS.items():
        d = s * torch.randn(len(p0), generator=g, dtype=DT); out.append(("B", name, tr.skeleton().with_params((p0 + d).tolist()), float(d.abs().max())))
    data = FitData.build(task.X, task.y, 0.10); S = DeepSearch("factorized", budget=150.0, seed=seed, max_depth=min(6, n_atoms(tr) + 1)); S._setup(data, 0.10); lits = S._atom_literals(data, S.fact_inits, S.atom_steps, "global")
    gen = generate_candidates(task.X.float(), data.y, max_depth=min(6, n_atoms(tr) + 1), top=4, seed=seed, extra_literals=lits, geometry=False)
    for r, i in enumerate(sorted(range(len(gen.programs)), key=lambda i: -gen.info[i]["score"])[:3]): out.append(("C", f"rank{r}", normalize(gen.programs[i]), None))
    return out


def candidate_study(task, seed=0) -> List[dict]:
    """Étude par candidat : signaux, puis chaque objectif (état initial / meilleur / final) avec accord réel."""
    A = _accord(task); data = FitData.build(task.X, task.y, 0.10); rows = []; tr = task.world.truth
    for regime, level, prog, perr in candidates_for(task, seed):
        sk = prog.skeleton(); init = torch.tensor(prog.param_vector(), dtype=DT); st = lambda th: sk.with_params(th.tolist()); sig = signals(sk, data, init)
        m = lambda th: {"accord": A(st(th)), "errors": float(hard_errors(sk, th.reshape(1, -1), data)[0]), "profiled": profiled(sk, th, data)}
        row = {"experiment": "candidates", "candidate_regime": regime, "level": level, "param_error": perr, "atoms": n_atoms(prog), "is_truth_form": algebraic_form(prog) == algebraic_form(tr), "signals": sig, "initial": m(init), "objectives": {}}
        for obj in OBJECTIVES:
            t0 = time.time(); trc = optimize(sk, data, init, obj); fin = m(trc.final); best = m(trc.best); acc = [A(st(torch.tensor(s["params"], dtype=DT))) for s in trc.steps]; ov = [s["objective"] for s in trc.steps]
            row["objectives"][obj] = {"final": fin, "best": best, "best_step": trc.best_step, "objective_before": ov[0], "objective_after": ov[-1], "objective_improved": ov[-1] < ov[0] - 1e-9,
                                      "corr_objective_accord": _spearman([-v for v in ov], acc), "corr_profiled_accord": _spearman([s["profiled"] for s in trc.steps], acc), "max_shift": float((trc.final - init).abs().max()),
                                      "errors_delta": fin["errors"] - row["initial"]["errors"], "seconds": round(time.time() - t0, 3)}
        rows.append(row)
    return rows


def e2e_run(task, policy: str, literals: str, seed=0) -> dict:
    """Chaîne complète : découverte de primitives (ou vrais atomes, contrôle) → composition → politique d'optimisation
    → sélection. Seul le paramètre `validated` de l'étape d'ajustement change entre les politiques."""
    from core.deep_reachability import leaves
    t0 = time.time(); tr = task.world.truth; A = _accord(task); data = FitData.build(task.X, task.y, 0.10); kw = {"fact_literals": leaves(tr), "fact_residual": False} if literals == "true" else {}
    out = DeepSearch("factorized", budget=150.0, seed=seed, max_depth=min(6, n_atoms(tr) + 1), validated=E2E_POLICIES[policy], **kw).run(data); best = out.best().program; a = A(best)
    return {"experiment": "e2e", "policy": policy, "literals": literals, "accord": a, "thresholds": {str(t): a >= t for t in THRESHOLDS}, "likelihood": profiled(best.skeleton(), torch.tensor(best.param_vector(), dtype=DT), data), "exact": skel_key(best) == skel_key(tr),
            "equivalent": structurally_equivalent(best, tr), "atoms": n_atoms(best), "fit_equivalents": out.stats["fit_equivalents"], "transactions": out.stats["transactions"], "rollbacks": out.stats["rollbacks"], "decisions": out.stats["decisions"],
            "preserved": out.stats["preserved"], "gate_rollbacks": out.stats["gate_rollbacks"], "seconds": round(time.time() - t0, 2)}


# Les 9 tâches dégradées par le garde-fou dans l'expérience à deux bras : 6 nouvelles (4,7 %) et 3 du diagnostic (2,3 %).
RESIDUAL_TASKS = (("104", 4, "standard"), ("106", 4, "standard"), ("110", 5, "standard"), ("107", 4, "standard"), ("101", 4, "rich"), ("107", 5, "standard"), ("9", 4, "standard"), ("0", 5, "rich"), ("1", 5, "standard"))


def residual_case(task, seed=0) -> dict:
    """Analyse d'une tâche DÉGRADÉE par le garde-fou : pour chaque ajustement accepté, l'accord réel a-t-il baissé
    alors que la vraisemblance profilée montait (exploitation du bruit) ? Les deux bras finissent-ils sur la même
    structure ? Laquelle les DONNÉES préfèrent-elles ?"""
    from core.deep_downstream import TracedSearch
    from core.deep_reachability import leaves
    tr = task.world.truth; A = _accord(task); data = FitData.build(task.X, task.y, 0.10); res = {}
    for arm, v in (("A", None), ("B", "profiled")):
        S = TracedSearch("factorized", budget=150.0, seed=seed, max_depth=min(6, n_atoms(tr) + 1), fact_literals=leaves(tr), fact_residual=False, validated=v); out = S.run(data); b = out.best().program
        res[arm] = {"accord": A(b), "form": algebraic_form(b), "atoms": n_atoms(b), "is_truth_form": algebraic_form(b) == algebraic_form(tr), "profiled": profiled(b.skeleton(), torch.tensor(b.param_vector(), dtype=DT), data), "evidence": out.best().evidence,
                    "errors": float(hard_errors(b.skeleton(), torch.tensor([b.param_vector()], dtype=DT), data)[0]), "simplified": S.simplification["changed"], "n_hypotheses": len(out.all)}
        if arm == "B":
            tx = []
            for t in S.transactions:
                pass
            fits = [f for f in S.fits if f["inits"] and len(f["inits"]) == 1]; harm = []
            for f, t in zip(fits, S.transactions):
                sk = f["inits"][0].skeleton(); ab = A(sk.with_params(list(t.before))); ap = A(sk.with_params(list(t.proposal)))
                harm.append({"accepted": t.accepted, "accord_before": ab, "accord_proposal": ap, "profiled_gain": t.value_after - t.value_before, "errors_before": float(hard_errors(sk, torch.tensor([list(t.before)], dtype=DT), data)[0]),
                             "errors_proposal": float(hard_errors(sk, torch.tensor([list(t.proposal)], dtype=DT), data)[0]), "is_truth_form": algebraic_form(f["inits"][0]) == algebraic_form(tr), "origin": f["origin"]})
            res["transactions"] = harm
    acc = [h for h in res["transactions"] if h["accepted"]]; rb = [h for h in res["transactions"] if not h["accepted"]]
    res["accepted_harmful"] = sum(h["accord_proposal"] < h["accord_before"] - 0.002 for h in acc); res["accepted"] = len(acc); res["accepted_harmful_on_truth_form"] = sum(h["accord_proposal"] < h["accord_before"] - 0.002 and h["is_truth_form"] for h in acc)
    res["rollback_lost_gain"] = sum(h["accord_proposal"] > h["accord_before"] + 0.002 for h in rb); res["rollbacks"] = len(rb); res["same_final_structure"] = res["A"]["form"] == res["B"]["form"]
    res["data_prefer_B"] = res["B"]["profiled"] > res["A"]["profiled"] + 1e-9; res["evidence_prefer_B"] = res["B"]["evidence"] > res["A"]["evidence"] + 1e-9
    if res["B"]["accord"] >= res["A"]["accord"] - 1e-9: res["explanation"] = "NOT_DEGRADED"
    elif res["accepted_harmful"] > 0 and res["data_prefer_B"]: res["explanation"] = "NOISE_EXPLOITED_ACCEPTED"          # un ajustement accepté (mieux sur les données) est fonctionnellement moins bon
    elif res["data_prefer_B"] or res["evidence_prefer_B"]: res["explanation"] = "DATA_PREFER_WORSE_FUNCTION"          # la solution B explique mieux les observations et est moins bonne : critère insuffisant
    elif not res["same_final_structure"]: res["explanation"] = "PATH_CHANGED_STRUCTURE"                              # conserver d'autres états a changé la structure finalement retenue
    else: res["explanation"] = "OTHER"
    return {"experiment": "residual", **res}

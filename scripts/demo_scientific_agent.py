"""Démo : boucle scientifique — hypothèses -> choix de mesure -> observation -> conditionnement -> mesure suivante
-> conclusion ; dans le monde temporel, puis appliquée au diagnostic du modèle (même algèbre)."""
import copy
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from core.measurement import MeasurementPlanner  # noqa: E402
from core.measurement_agent import (  # noqa: E402
    COMMIT, MeasurementAgent, RegionOperator, WeighOperator, fit_calibrators, fit_training, parse_question,
)
from core.measurement_world import episode, train_episodes  # noqa: E402
from core.model_diagnostics import CAUSES, diagnose_and_repair, inject, learn_tables, setup  # noqa: E402


def main():
    train = train_episodes(30); hz, rel = fit_training(train); cal = fit_calibrators(train, hz, rel)
    ep = episode(424242)
    ag = MeasurementAgent("full_algebra", hz, rel, calibrators=cal); ag.ingest(ep)
    q = next((q for q in ep["questions"] if q["type"] == "color_of_relational"), ep["questions"][0])
    print(f"Question : « {q['text']} »")
    pq = parse_question(q["text"]); cands = ag.candidates(pq["description"])
    S = ag.ensemble(pq["intent"], pq["description"], cands, ep["T"])
    world = ag.latent(); target = ("ans", "value")
    planner = MeasurementPlanner(confidence=1.01, budget=2.0); planner.decision_costs = (20.0, 8.0)
    planner.calibrate = lambda p: ag.calibrated(pq["intent"], p)
    for cycle in range(1, 6):
        m = S.marginal(target); ref = S.marginal(("ans", "ref"))
        print(f"\nCycle {cycle} — hypothèses : référent { {k: round(v, 3) for k, v in ref.items() if v > 0.01} } ; réponse { {k: round(v, 3) for k, v in m.items() if v > 0.01} }")
        ops = ag.operators(cands, pq["intent"])
        scored = sorted((planner.score(S, op, target) for op in ops), key=lambda c: -c.utility)[:4]
        for c in scored: print(f"   candidate {c.operator.name:28s} IG {c.information_gain:.3f} utilité {c.utility:+.3f} (perturbation {c.operator.context.disturbance})")
        c = planner.choose(S, ops, target)
        if c is None: print("   -> arrêt : plus aucune mesure ne vaut son coût (ou confiance suffisante)"); break
        rec = world.measure(c.operator, c.operator.context.sensor)
        if isinstance(c.operator, (RegionOperator, WeighOperator)):
            L = c.operator.likelihood(rec.outcome, S.column((c.operator.observable.entity, "loc"))); S = S._fork(S.logw + torch.log(L.clamp_min(1e-300)), "m")
        else: S = S.condition(rec)
        print(f"   mesure : {c.operator.name} -> {rec.outcome.value}")
    v, p = max(S.marginal(target).items(), key=lambda kv: kv[1]); pc = ag.calibrated(pq["intent"], p)
    print(f"\nConclusion : {v} (brut {p:.3f}, calibré {pc:.3f}) -> {'AFFIRMÉE' if pc >= COMMIT else 'RETENUE'} ; vérité {q['truth']}")

    print("\n=== Même algèbre appliquée au MODÈLE (diagnostic -> réparation minimale) ===")
    tables = learn_tables(); base, xv, yv, acc = setup(1)
    for cause in ("corrupt:head", "fast_drift"):
        m = copy.deepcopy(base); inject(m, cause, random.Random(7))
        print(f"panne injectée (cachée) : {cause} ; exactitude {m.accuracy(xv, yv):.3f}")
        r = diagnose_and_repair(m, tables, xv, yv, acc)
        print(f"  diagnostics {r['log']} -> cause {r['cause']} (P={r['p']}) -> réparation {r['repair']} -> exactitude {r['accuracy']:.3f}")


if __name__ == "__main__":
    main()

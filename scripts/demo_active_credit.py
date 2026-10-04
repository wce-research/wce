"""Démo : panne → causes candidates (signes bon marché) → choix actif des interventions (VOI) → diagnostic ;
identifiabilité / observabilité et ensemble minimal de diagnostics."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.active_credit import (CURRENT, MEASUREMENTS, CausePrior, active_diagnosis, cheap_signs, identifiability,  # noqa: E402
                                minimal_diagnostic_set, silent_failure_analysis, simulate_outcomes)
from core.generalized_credit import inject, make_system, score  # noqa: E402

if __name__ == "__main__":
    prior = CausePrior().fit(); rng = random.Random(4)
    for fault in ("source", "weights:head", "planner"):
        sys_ = make_system(11); h = score(sys_); inject(sys_, fault, rng)
        signs, cost = cheap_signs(sys_); post = prior.posterior(signs)
        top = sorted(post.items(), key=lambda kv: -kv[1])[:3]
        c, n, _ = active_diagnosis(sys_, "voi", prior, base_ref=h)
        print(f"Panne cachée {fault:16s} score {h:.3f}->{score(sys_):.3f} | signes {signs} (coût {cost:.2f})")
        print(f"   a priori top-3 : {[(k, round(v, 2)) for k, v in top]} -> {n} intervention(s) -> cause {c}  (toutes : 10 interventions)")
    d = simulate_outcomes(8)
    print(f"\nIdentifiabilité avec la mesure courante {CURRENT} : {identifiability(d, CURRENT)['fraction']:.3f} des paires de causes ; "
          f"avec les {len(MEASUREMENTS)} mesures : {identifiability(d, list(MEASUREMENTS))['fraction']:.3f}")
    print("Ensemble minimal de diagnostics :", minimal_diagnostic_set(d))
    print("Pannes silencieuses :", silent_failure_analysis(40, 0))

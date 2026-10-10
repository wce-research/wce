"""Demo: fault → candidate causes (cheap signs) → active choice of interventions (VOI) → diagnosis;
identifiability / observability and minimal diagnostic set."""
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
        print(f"Hidden fault {fault:16s} score {h:.3f}->{score(sys_):.3f} | signs {signs} (cost {cost:.2f})")
        print(f"   prior top-3: {[(k, round(v, 2)) for k, v in top]} -> {n} intervention(s) -> cause {c}  (all: 10 interventions)")
    d = simulate_outcomes(8)
    print(f"\nIdentifiability with the current measurement {CURRENT}: {identifiability(d, CURRENT)['fraction']:.3f} of the cause pairs; "
          f"with the {len(MEASUREMENTS)} measurements: {identifiability(d, list(MEASUREMENTS))['fraction']:.3f}")
    print("Minimal diagnostic set:", minimal_diagnostic_set(d))
    print("Silent faults:", silent_failure_analysis(40, 0))

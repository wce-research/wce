"""Demo: fault → perturbation / intervention on each component → contribution scores → diagnosis → repair."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy  # noqa: E402

from core.generalized_credit import COMPONENTS, generalized_credit, gradient_only_cause, inject, make_system, score  # noqa: E402
from core.model_diagnostics import learn_tables, setup, inject as inject_model  # noqa: E402
from core.perturbation_diagnostics import diagnose, healthy_reference, learn_profile_table, prior_from_profile, profile  # noqa: E402

if __name__ == "__main__":
    rng = random.Random(3)
    for fault in ("lexical", "source", "transformation", "expert:expert_a"):
        sys_ = make_system(5); healthy = score(sys_); inject(sys_, fault, rng)
        base, credits = generalized_credit(sys_)
        print(f"\nHidden fault: {fault} ; score {healthy:.3f} -> {base:.3f}")
        if healthy - base < 0.03:
            print("   SILENT fault (no measurable effect on the score): no attribution by effect is possible"); continue
        for c in credits[:4]: print(f"   {c.component:16s} [{c.kind:14s}] interventional credit {c.interventional:+.3f} | observational (gradient) {c.observational:.3f}")
        top = credits[0]; fixed = COMPONENTS[top.component][1](copy.deepcopy(sys_))
        print(f"   diagnosis: {top.component} (gradient alone would have said: {gradient_only_cause(sys_)}) -> repair: score {score(fixed):.3f}")
    print("\nModel diagnosis guided by virtual perturbations:")
    tables, counts = learn_tables(seed=0), learn_profile_table(seed=0)
    base, xv, yv, acc = setup(1); ref = healthy_reference(base, xv, yv)
    for cause in ("insufficient_capacity", "fast_drift"):
        m = copy.deepcopy(base); inject_model(m, cause, rng)
        sig, pc = profile(m, ref, xv, yv); prior = prior_from_profile(counts, sig)
        c, n, k = diagnose(m, tables, xv, yv, acc, prior)
        print(f"   fault {cause:22s} profile {sig} -> max prior {max(prior, key=prior.get)} -> cause {c} ({n} measurements, cost {k + pc:.3f})")

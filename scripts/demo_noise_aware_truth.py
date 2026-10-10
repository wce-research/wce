"""Demo: wrong rule + noisy boundary → observations (repeated if useful) → P(wrong rule) → accept / reject."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from core.active_experiments import candidate_rules  # noqa: E402
from core.active_structure import sequential_truth_test  # noqa: E402
from core.active_world import balanced_instances, noisy_task  # noqa: E402
from core.boundary_program import render  # noqa: E402
from core.boundary_world import full_region  # noqa: E402

if __name__ == "__main__":
    inst = balanced_instances(["two_lobe"], 3, 1)[0]; g = torch.Generator().manual_seed(0); X, y = noisy_task(inst, 120, g, 0.08)
    for kind, rule, wrong, alts in candidate_rules(inst, X, y, 0.08, 0):
        r = sequential_truth_test(rule, alts, inst.label, full_region, torch.Generator().manual_seed(1), 0.08, budget=16)
        print(f"{kind:18s} {'WRONG   ' if wrong else 'correct '} {render(rule)[:60]:62s} P(wrong)={r['p_wrong']:.3f} -> {r['decision']:9s} ({r['queries']} obs., {r['repeats']} repeats)")

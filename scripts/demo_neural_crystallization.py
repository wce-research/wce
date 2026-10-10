"""Demo: neural expert → induction of a program on its behaviour → verification → replacement / residual / retention."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.boundary_experiments import crystallization_experiment  # noqa: E402

if __name__ == "__main__":
    for r in crystallization_experiment(5):
        print(f"  {r['family']:16s} expert {r['expert_acc']:.3f} ({r['expert_params']} p.) -> {r['decision']:16s} accuracy {r['acc']:.3f} ({r['params']} p.) {r.get('program', '')[:70]}")

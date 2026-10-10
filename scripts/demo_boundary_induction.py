"""Demo: samples → candidate primitives → search → discovered program → extrapolation (HELD-OUT family)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from core.boundary_methods import evaluate_all  # noqa: E402
from core.boundary_program import atoms, explain, render  # noqa: E402
from core.boundary_world import instances, sample_task  # noqa: E402

if __name__ == "__main__":
    inst = instances(["xor_halfspaces"], 3, 1)[0]; g = torch.Generator().manual_seed(0); X, y = sample_task(inst, 200, g)
    print("Family hidden from the system (never named in the induction code). 200 observations in x0 ∈ [−2; 0.5].")
    print("Candidate atoms:", [render(a) for a in atoms()])
    R = evaluate_all(inst, X, y, g, methods=("memorization", "mlp", "oracle", "bpi_full", "bpi_active"))
    for m, r in R.items():
        if r: print(f"  {m:13s} region {r['in_region']:.3f} | EXTRAPOLATION {r['extrapolation']:.3f} | parameters {r['params']:4d} | evaluations {r['evaluations']:4d} | program: {r.get('program', '—')}")

"""SINGLE evaluation of the sealed set of the perturbation bench -> results/perturbation_sealed.json (do not rerun for tuning)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.perturbation_benchmark import SEALED_SHA256, evaluate_items, sealed_hash, sealed_items  # noqa: E402

if __name__ == "__main__":
    assert sealed_hash() == SEALED_SHA256, "sealed set modified"
    path = "results/perturbation_sealed.json"
    if os.path.exists(path) and "--force" not in sys.argv:
        print(open(path).read()); sys.exit(0)
    from core.generalized_credit import credit_benchmark
    from core.safe_plasticity import continual_benchmark
    r = {"items": evaluate_items(sealed_items()), "credit": credit_benchmark(60, seed=990),
         "continual": {k: {kk: v[kk] for kk in ("new_task", "retention", "forgetting", "parameters_changed", "rollbacks")}
                       for k, v in continual_benchmark(seed=990).items()}}
    os.makedirs("results", exist_ok=True); json.dump(r, open(path, "w"), indent=1, default=str)
    print(json.dumps(r, indent=1, default=str))

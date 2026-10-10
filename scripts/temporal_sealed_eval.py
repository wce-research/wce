"""SINGLE evaluation of the sealed set of the temporal bench -> results/temporal_sealed.json (do not rerun for tuning)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.temporal_bench import SEALED_SHA256, sealed_episodes, sealed_hash, train_episodes  # noqa: E402
from core.temporal_eval import compare  # noqa: E402

if __name__ == "__main__":
    assert sealed_hash() == SEALED_SHA256, "sealed set modified"
    path = "results/temporal_sealed.json"
    if os.path.exists(path) and "--force" not in sys.argv:
        print(open(path).read()); sys.exit(0)
    r = compare(sealed_episodes(), train_episodes(60))
    os.makedirs("results", exist_ok=True); json.dump(r, open(path, "w"), indent=1, default=str)
    print(json.dumps({s: {k: v[k] for k in ("success", "false_confident", "abstention", "conflict", "ece", "prediction_sets")}
                      for s, v in r.items() if s != "hazards"}, indent=1))

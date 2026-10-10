"""SINGLE evaluation of the sealed set of the measurement world -> results/measurement_sealed.json (do not rerun for tuning)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.measurement_agent import SYSTEMS_M, evaluate, fit_calibrators, fit_training, metrics  # noqa: E402
from core.measurement_world import SEALED_SHA256, sealed_episodes, sealed_hash, train_episodes  # noqa: E402

if __name__ == "__main__":
    assert sealed_hash() == SEALED_SHA256, "sealed set modified"
    path = "results/measurement_sealed.json"
    if os.path.exists(path) and "--force" not in sys.argv:
        print(open(path).read()); sys.exit(0)
    train = train_episodes(30); hz, rel = fit_training(train); cal = fit_calibrators(train, hz, rel)
    eps = sealed_episodes()
    r = {s: metrics(evaluate(s, eps, hz, rel, calibrators=cal)) for s in SYSTEMS_M}
    os.makedirs("results", exist_ok=True); json.dump(r, open(path, "w"), indent=1, default=str)
    print(json.dumps({s: {k: v.get(k) for k in ("success", "false_confident", "abstention", "measurements", "cost", "calibration_ece", "ig_corr")}
                      for s, v in r.items()}, indent=1))

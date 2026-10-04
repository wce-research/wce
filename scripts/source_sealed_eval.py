"""Évaluation UNIQUE du lot scellé du banc adversarial des sources -> results/source_sealed.json"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.source_adversarial import SEALED_SHA256, sealed_episodes, sealed_hash  # noqa: E402
from core.source_adversarial_eval import compare  # noqa: E402

if __name__ == "__main__":
    assert sealed_hash() == SEALED_SHA256, "lot scellé modifié"
    path = "results/source_sealed.json"
    if os.path.exists(path) and "--force" not in sys.argv:
        print(open(path).read()); sys.exit(0)
    r = compare(sealed_episodes())
    out = {m: {k: v[k] for k in ("n", "success", "false_confident", "abstention", "exact_class", "conflict_recall", "drift_flagged", "dependency_grouped")}
           | {"belief_ece": v["belief_calibration"]["ece"], "reliability_ece": v["reliability_calibration"]["ece"]} for m, v in r.items()}
    os.makedirs("results", exist_ok=True); json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))

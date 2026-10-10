"""SINGLE evaluation of the continuous sealed set -> results/continuum_sealed.json (do not rerun for tuning)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.continuum_benchmark import SEALED_SHA256, sealed_eval, sealed_hash  # noqa: E402

if __name__ == "__main__":
    assert sealed_hash() == SEALED_SHA256, "sealed set modified"
    path = "results/continuum_sealed.json"
    if os.path.exists(path) and "--force" not in sys.argv:
        print(open(path).read()); sys.exit(0)
    r = sealed_eval(); json.dump(r, open(path, "w"), indent=1, default=str); print(json.dumps(r, indent=1, default=str))

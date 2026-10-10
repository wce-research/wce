"""SINGLE evaluation of the boundary induction sealed set -> results/boundary_sealed.json."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.boundary_benchmark import sealed_eval  # noqa: E402
from core.boundary_world import SEALED_SHA256, sealed_hash  # noqa: E402

if __name__ == "__main__":
    assert sealed_hash() == SEALED_SHA256, "sealed set modified"
    path = "results/boundary_sealed.json"
    if os.path.exists(path) and "--force" not in sys.argv: print(open(path).read()[:4000]); sys.exit(0)
    r = sealed_eval(); json.dump(r, open(path, "w"), indent=1, default=str); print("written", path)

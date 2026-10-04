"""Évaluation UNIQUE du lot scellé « Active Structural Discovery » -> results/active_sealed.json."""
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.active_benchmark import sealed_job  # noqa: E402
from core.active_world import SEALED_SHA256, balanced_instances, sealed_hash, sealed_spec  # noqa: E402

if __name__ == "__main__":
    assert sealed_hash() == SEALED_SHA256, "lot scellé modifié"
    path = "results/active_sealed.json"
    if os.path.exists(path) and "--force" not in sys.argv: print(open(path).read()[:3000]); sys.exit(0)
    s = sealed_spec(); n = len(balanced_instances(s["families"], s["seed"], s["per_family"]))
    jobs = [(i, nz) for i in range(n) for nz in s["noise"]]
    with Pool(int(os.environ.get("PROCS", "12"))) as p: rows = [r for rs in p.map(sealed_job, jobs) for r in rs]
    json.dump(rows, open(path, "w"), indent=1, default=str); print("écrit", path, len(rows))

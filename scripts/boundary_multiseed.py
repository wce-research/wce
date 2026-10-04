"""Multi-graines du banc d'induction de frontières, parallélisé (graine × expérience) sur tous les cœurs.
Usage : python scripts/boundary_multiseed.py [n_graines=16] [processus=12]. Reprise : un fichier par (graine, tâche)."""
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.boundary_benchmark import JOBS, aggregate, run_job  # noqa: E402

D = "results/boundary"


def work(a):
    path = f"{D}/{a[0]}_{a[1]}.json"
    if os.path.exists(path): return path
    r = run_job(a); json.dump(r, open(path, "w"), default=str); return path


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16; procs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    os.makedirs(D, exist_ok=True)
    order = {"main": 0, "ablation": 1, "depth": 1, "active": 2, "crystal": 2, "family": 3}
    jobs = sorted([(s, j) for s in range(n) for j in JOBS], key=lambda a: order.get(a[1], 9))
    with Pool(procs) as p:
        for path in p.imap_unordered(work, jobs): print("ok", path, flush=True)
    res = [tuple(json.load(open(f"{D}/{s}_{j}.json"))) for s in range(n) for j in JOBS]
    S = aggregate(res); json.dump(S, open("results/boundary_summary.json", "w"), indent=1, default=str)
    print(json.dumps(S["hypotheses"], indent=1, ensure_ascii=False, default=str))

"""Multi-seed "Active Structural Discovery", parallelised (seed × experiment), resumable per file.
Usage: python scripts/active_multiseed.py [seeds=16] [processes=12] [jobs=all, comma-separated]"""
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.active_benchmark import JOBS, aggregate, run_job  # noqa: E402

D = "results/active"


def work(a):
    path = f"{D}/{a[0]}_{a[1]}.json"
    if os.path.exists(path): return path
    r = run_job(a); json.dump(r, open(path + ".tmp", "w"), default=str); os.replace(path + ".tmp", path); return path


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16; procs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    jobs_sel = sys.argv[3].split(",") if len(sys.argv) > 3 else JOBS
    os.makedirs(D, exist_ok=True); prio = {"main": 0, "family": 1, "noise": 2, "qp": 3}
    jobs = sorted([(s, j) for s in range(n) for j in jobs_sel], key=lambda a: (prio.get(a[1], 9), a[0]))
    with Pool(procs) as p:
        for path in p.imap_unordered(work, jobs): print("ok", path, flush=True)
    if set(jobs_sel) == set(JOBS):
        res = [tuple(json.load(open(f"{D}/{s}_{j}.json"))) for s in range(n) for j in JOBS]
        S = aggregate(res); json.dump(S, open("results/active_summary.json", "w"), indent=1, default=str)
        print(json.dumps(S["hypotheses"], indent=1, ensure_ascii=False, default=str))

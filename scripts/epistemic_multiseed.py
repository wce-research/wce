"""Graines de l'expérience épistémique -> results/epistemic_multiseed[_16].json"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.epistemic_benchmark import SYSTEMS_E, multi_seed  # noqa: E402

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    systems = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else SYSTEMS_E
    t0 = time.time()
    res = multi_seed(range(n), systems, log=lambda *a: print(*a, flush=True))
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs("results", exist_ok=True)
    path = f"results/epistemic_multiseed{'' if n == 8 else '_' + str(n)}.json"
    with open(path, "w") as f: json.dump(res, f, indent=1, default=str)
    for name, s in res["summary"].items():
        print(name, {k: (v["mean"], v["ci95"]) for k, v in s.items() if v})
    print(json.dumps(res["comparisons"], indent=1, default=str)[:4000])

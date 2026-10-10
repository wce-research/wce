"""8 seeds × 5 systems (independent environment) -> results/unified_multiseed.json"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.unified_benchmark import multi_seed  # noqa: E402

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    t0 = time.time()
    res = multi_seed(range(n), log=lambda *a: print(*a, flush=True))
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs("results", exist_ok=True)
    with open("results/unified_multiseed.json", "w") as f: json.dump(res, f, indent=1, default=str)
    for name, s in res["summary"].items():
        print(name, {k: (v["mean"], v["ci95"]) for k, v in s.items() if v and k in ("success", "false_confident", "parse_coverage", "abstention")})
    print(json.dumps(res["comparisons"], indent=1, default=str))
    print("conversion", res["success_conversion"])

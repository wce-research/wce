"""16 seeds × 6 systems (source reliability, L1 identity) -> results/source_multiseed.json"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.source_benchmark import multi_seed  # noqa: E402

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    systems = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
    tag = sys.argv[3] if len(sys.argv) > 3 else ""
    t0 = time.time()
    res = multi_seed(range(n), **({"systems": systems} if systems else {}), log=lambda *a: print(*a, flush=True))
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/source_multiseed{'' if n == 16 else '_' + str(n)}{tag}.json", "w"), indent=1, default=str)
    for name, s in res["summary"].items(): print(name, {k: (v["mean"], v["ci95"]) for k, v in s.items() if v})

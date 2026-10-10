"""One seed of the continuous benchmark -> results/continuum/seed_<s>.json (resumes if already present)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from core.continuum_benchmark import seed_run  # noqa: E402

if __name__ == "__main__":
    s = int(sys.argv[1]); path = f"results/continuum/seed_{s}.json"
    if os.path.exists(path): print("already done", path); sys.exit(0)
    torch.set_num_threads(int(os.environ.get("THREADS", "2")))
    os.makedirs("results/continuum", exist_ok=True); r = seed_run(s)
    json.dump(r, open(path, "w"), default=str); print("written", path)

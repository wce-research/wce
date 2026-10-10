"""Demo: strict budget (400 floats) → crystallize / gated residual / shared head / prune → acquire or REFUSE."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.active_experiments import capacity_job  # noqa: E402

if __name__ == "__main__":
    r = capacity_job(3)
    for m in ("experts_only", "full"): print(f"{m:13s} : {r[m]}")

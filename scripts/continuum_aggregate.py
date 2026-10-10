"""Aggregates results/continuum/seed_*.json -> results/continuum_summary.json (intervals, scaling, hypotheses)."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.continuum_benchmark import aggregate  # noqa: E402

if __name__ == "__main__":
    per = [json.load(open(f)) for f in sorted(glob.glob("results/continuum/seed_*.json"))]
    S = aggregate(per); json.dump(S, open("results/continuum_summary.json", "w"), indent=1, default=str)
    print(json.dumps(S["hypotheses"], indent=1, default=str)); print(f"{len(per)} seeds")

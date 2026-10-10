"""Demo: posterior of a source, CREDIBLE interval vs PREDICTIVE interval (beta-binomial), empirical coverage."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.source_reliability import SourceReliabilityModel  # noqa: E402
from core.temporal_belief import beta_binomial_interval  # noqa: E402
from core.temporal_eval import source_interval_coverage  # noqa: E402


def main():
    m = SourceReliabilityModel(mode="mean", global_prior=(2.0, 1.0))
    for _ in range(64): m.observe("witness", True)
    for _ in range(36): m.observe("witness", False)
    r = m.record("witness")
    print(f"Source “witness”: posterior Beta({r.alpha:.0f}, {r.beta:.0f}) mean {r.mean:.3f}")
    lo, hi = r.interval(0.9)
    print(f"  90% CREDIBLE interval on r (parameter): [{lo:.3f}, {hi:.3f}]")
    for n in (5, 10, 20, 50, 200):
        plo, phi = beta_binomial_interval(r.alpha, r.beta, n, 0.9)
        print(f"  90% PREDICTIVE interval on the correct proportion among {n:3d} future assertions: [{plo:.2f}, {phi:.2f}]")
    print("\nEmpirical coverage on the temporal benchmark (development episodes):", source_interval_coverage())
    if os.path.exists("results/predictive_coverage.json"):
        print("Empirical coverage in the independent environment (16 seeds):", json.load(open("results/predictive_coverage.json")))


if __name__ == "__main__":
    main()

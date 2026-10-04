"""Démo : posterior d'une source, intervalle CRÉDIBLE vs intervalle PRÉDICTIF (bêta-binomial), couverture empirique."""
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
    print(f"Source « witness » : posterior Beta({r.alpha:.0f}, {r.beta:.0f}) moyenne {r.mean:.3f}")
    lo, hi = r.interval(0.9)
    print(f"  intervalle CRÉDIBLE 90 % sur r (paramètre) : [{lo:.3f}, {hi:.3f}]")
    for n in (5, 10, 20, 50, 200):
        plo, phi = beta_binomial_interval(r.alpha, r.beta, n, 0.9)
        print(f"  intervalle PRÉDICTIF 90 % sur la proportion juste parmi {n:3d} futures affirmations : [{plo:.2f}, {phi:.2f}]")
    print("\nCouverture empirique sur le banc temporel (épisodes de développement) :", source_interval_coverage())
    if os.path.exists("results/predictive_coverage.json"):
        print("Couverture empirique dans l'environnement indépendant (16 graines) :", json.load(open("results/predictive_coverage.json")))


if __name__ == "__main__":
    main()

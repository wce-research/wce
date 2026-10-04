"""Démo : signal temporel -> base de Fourier -> structure fréquentielle -> inverse (le monde ne change pas)."""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from core.measurement import BasisTransform  # noqa: E402
from core.measurement_experiments import fourier_experiment, make_signal, raw_reasoning, spectral_reasoning  # noqa: E402


def main():
    bt = BasisTransform(128)
    x, truth, dom = make_signal(random.Random(4))
    print(f"Signal (base TEMPORELLE, 128 échantillons) : mélange de fréquences {sorted(truth)} (dominante {dom}) + bruit")
    print("  premiers échantillons :", [round(float(v), 2) for v in x[:10]])
    X = bt.forward(x); mag = X.abs()[:64]
    top = sorted(range(64), key=lambda f: -float(mag[f]))[:5]
    print(f"Base FRÉQUENTIELLE : pics |X(f)| aux fréquences {top} (valeurs {[round(float(mag[f]), 2) for f in top]})")
    print("Raisonnement dans la base temporelle (passages à zéro) :", raw_reasoning(x))
    print("Raisonnement dans la base fréquentielle (pics au-dessus du bruit) :", spectral_reasoning(x, bt))
    xr = bt.inverse(X)
    print(f"Inverse : erreur max {float((xr - x).abs().max()):.2e} ; énergie temps {bt.energy(x):.4f} = fréquence {bt.energy(X):.4f} (Parseval)")
    print("Invariants :", bt.invariants(x))
    print("\nBanc Fourier (200 signaux) :", fourier_experiment())


if __name__ == "__main__":
    main()

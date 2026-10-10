"""Demo: time signal -> Fourier basis -> frequency structure -> inverse (the world does not change)."""
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
    print(f"Signal (TIME basis, 128 samples): mixture of frequencies {sorted(truth)} (dominant {dom}) + noise")
    print("  first samples:", [round(float(v), 2) for v in x[:10]])
    X = bt.forward(x); mag = X.abs()[:64]
    top = sorted(range(64), key=lambda f: -float(mag[f]))[:5]
    print(f"FREQUENCY basis: peaks |X(f)| at frequencies {top} (values {[round(float(mag[f]), 2) for f in top]})")
    print("Reasoning in the time basis (zero crossings):", raw_reasoning(x))
    print("Reasoning in the frequency basis (peaks above the noise):", spectral_reasoning(x, bt))
    xr = bt.inverse(X)
    print(f"Inverse: max error {float((xr - x).abs().max()):.2e}; time energy {bt.energy(x):.4f} = frequency {bt.energy(X):.4f} (Parseval)")
    print("Invariants:", bt.invariants(x))
    print("\nFourier benchmark (200 signals):", fourier_experiment())


if __name__ == "__main__":
    main()

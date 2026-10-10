"""Demo: update safe at FIRST order (J·Δθ ≈ 0) → the CURVATURE detects the risk → reduced update."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.second_order_experiment import run  # noqa: E402

if __name__ == "__main__":
    rows, adapt = run(3)
    print("Updates projected out of the sensitive directions of A (first order), increasing amplitudes:")
    print(f"{'scale':>8s} {'|JΔθ|/|Δθ|':>11s} {'actual flips':>20s} {'1st order':>10s} {'2nd order':>10s} {'ellipsoid':>11s}")
    for r in rows:
        print(f"{r['scale']:8.3f} {r['J_delta_rel']:11.4f} {r['actual_flip_rate']:20.3f} {str(r['first_order_flags']):>10s} {str(r['second_order_flags']):>10s} {str(r['ellipsoid_flags']):>11s}")
    print(f"\nAdaptation: first order (full step): retention A {adapt['first_order']['retention_A']:.3f}, B {adapt['first_order']['new_B']:.3f}")
    print(f"            second order (α={adapt['second_order']['alpha']:.4f}): retention A {adapt['second_order']['retention_A']:.3f}, B {adapt['second_order']['new_B']:.3f}  -> REDUCED update")
    print(f"            growth (new module, {adapt['structural_growth']['params_added']} parameters): A {adapt['structural_growth']['retention_A']:.3f}, B {adapt['structural_growth']['new_B']:.3f}")
    print(f"Cost: ellipsoid {adapt['ellipsoid_seconds']} s, {adapt['ellipsoid_floats']} floats")

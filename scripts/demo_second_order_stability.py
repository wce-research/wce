"""Démo : mise à jour sûre au PREMIER ordre (J·Δθ ≈ 0) → la COURBURE détecte le risque → mise à jour réduite."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.second_order_experiment import run  # noqa: E402

if __name__ == "__main__":
    rows, adapt = run(3)
    print("Mises à jour projetées hors des directions sensibles de A (premier ordre), amplitudes croissantes :")
    print(f"{'échelle':>8s} {'|JΔθ|/|Δθ|':>11s} {'retournements réels':>20s} {'1er ordre':>10s} {'2nd ordre':>10s} {'ellipsoïde':>11s}")
    for r in rows:
        print(f"{r['scale']:8.3f} {r['J_delta_rel']:11.4f} {r['actual_flip_rate']:20.3f} {str(r['first_order_flags']):>10s} {str(r['second_order_flags']):>10s} {str(r['ellipsoid_flags']):>11s}")
    print(f"\nAdaptation : premier ordre (pas complet) : rétention A {adapt['first_order']['retention_A']:.3f}, B {adapt['first_order']['new_B']:.3f}")
    print(f"             second ordre (α={adapt['second_order']['alpha']:.4f}) : rétention A {adapt['second_order']['retention_A']:.3f}, B {adapt['second_order']['new_B']:.3f}  -> mise à jour RÉDUITE")
    print(f"             croissance (nouveau module, {adapt['structural_growth']['params_added']} paramètres) : A {adapt['structural_growth']['retention_A']:.3f}, B {adapt['structural_growth']['new_B']:.3f}")
    print(f"Coût : ellipsoïde {adapt['ellipsoid_seconds']} s, {adapt['ellipsoid_floats']} flottants")

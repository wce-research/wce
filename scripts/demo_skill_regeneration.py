"""Démo : corruption d'un module → échec du certificat → régénération SANS instantané (certificats génératifs)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.continuum_experiments import regeneration_experiment  # noqa: E402

if __name__ == "__main__":
    r = regeneration_experiment(1, n=16)
    for m in r["per_module"]:
        print(f"  {m['module']:16s} avant {m['before']:.3f} | corrompu {m['corrupted']:.3f} (certificat en échec : {m['detected']}) | régénéré {m['regenerated']:.3f} | récupération {m['recovery']:.3f}")
    print(f"\nRécupération médiane {r['median_recovery']:.3f} ; moyenne {r['mean_recovery']:.3f} ; certificats repassés {r['certificates_pass']:.2f} ; aucun instantané utilisé")

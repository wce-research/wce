"""Demo: corruption of a module → certificate failure → regeneration WITHOUT snapshot (generative certificates)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.continuum_experiments import regeneration_experiment  # noqa: E402

if __name__ == "__main__":
    r = regeneration_experiment(1, n=16)
    for m in r["per_module"]:
        print(f"  {m['module']:16s} before {m['before']:.3f} | corrupted {m['corrupted']:.3f} (certificate failing: {m['detected']}) | regenerated {m['regenerated']:.3f} | recovery {m['recovery']:.3f}")
    print(f"\nMedian recovery {r['median_recovery']:.3f}; mean {r['mean_recovery']:.3f}; certificates passing again {r['certificates_pass']:.2f}; no snapshot used")

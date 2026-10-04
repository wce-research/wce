"""Démo : budget de capacité atteint → élagage / fusion / distillation (neuronale et symbolique) → nouvelle compétence."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter  # noqa: E402

from core.continuous_world import stream  # noqa: E402
from core.continuum import Config  # noqa: E402
from core.continuum_experiments import run_stream_cfg  # noqa: E402

if __name__ == "__main__":
    regs = stream(707, 30)
    for cap in (None, 400):
        cfg = Config("cap", cert="generative", discovery=False, drift="auto", capacity=cap, capacity_mgmt=cap is not None)
        o, s = run_stream_cfg(cfg, regs, 0); ops = Counter()
        print(f"\nBudget {cap or 'illimité'} :")
        for r in s.log:
            if r.get("capacity"):
                ops.update(r["capacity"]); print(f"  régime {r['t']:2d} : {r['params_before']:5d} -> {r['params_after']:5d} paramètres ; opérations {r['capacity']}")
        print(f"  exactitude {o['average_accuracy']:.3f} ; paramètres finaux {s.neural_params()} ; opérations {dict(ops)} ; distillations rejetées (certificats) {s.counters.get('distill_rejected', 0)}")
        print(f"  dernières acquisitions : {[r['kind'] for r in s.log[-5:]]}")

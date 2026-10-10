"""Demo: capacity budget reached → pruning / merging / distillation (neural and symbolic) → new skill."""
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
        print(f"\nBudget {cap or 'unlimited'}:")
        for r in s.log:
            if r.get("capacity"):
                ops.update(r["capacity"]); print(f"  regime {r['t']:2d}: {r['params_before']:5d} -> {r['params_after']:5d} parameters; operations {r['capacity']}")
        print(f"  accuracy {o['average_accuracy']:.3f}; final parameters {s.neural_params()}; operations {dict(ops)}; rejected distillations (certificates) {s.counters.get('distill_rejected', 0)}")
        print(f"  latest acquisitions: {[r['kind'] for r in s.log[-5:]]}")

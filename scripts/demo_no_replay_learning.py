"""Demo: task sequence → NO old raw data stored → retention (vs replay, which stores some)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.multitask_benchmark import METHODS, run  # noqa: E402
from core.skillnet import task_stream  # noqa: E402

if __name__ == "__main__":
    tasks = task_stream(2, 14)
    print("14 successive tasks, without task identifier:", [f"{t.name}:{t.kind}" for t in tasks])
    print(f"\n{'method':22s} {'mean acc.':>11s} {'new':>9s} {'forget.':>7s} {'BWT':>7s} {'FWT':>7s} {'growth':>8s} {'memory (B)':>12s} {'raw data':>15s}")
    for m in METHODS:
        r = run(m, tasks, seed=2)
        print(f"{m:22s} {r['average_accuracy']:11.3f} {r['new_task']:9.3f} {r['forgetting']:7.3f} {r['backward_transfer']:7.3f} {r['forward_transfer']:7.3f} {r['param_growth']:8.2f} {r['memory_bytes']:12d} {'YES' if m == 'replay' else 'no':>15s}")

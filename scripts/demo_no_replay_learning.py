"""Démo : séquence de tâches → AUCUNE donnée brute ancienne stockée → rétention (vs replay, qui en stocke)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.multitask_benchmark import METHODS, run  # noqa: E402
from core.skillnet import task_stream  # noqa: E402

if __name__ == "__main__":
    tasks = task_stream(2, 14)
    print("14 tâches successives, sans identifiant de tâche :", [f"{t.name}:{t.kind}" for t in tasks])
    print(f"\n{'méthode':22s} {'exact. moy.':>11s} {'nouvelle':>9s} {'oubli':>7s} {'BWT':>7s} {'FWT':>7s} {'croiss.':>8s} {'mémoire (o)':>12s} {'données brutes':>15s}")
    for m in METHODS:
        r = run(m, tasks, seed=2)
        print(f"{m:22s} {r['average_accuracy']:11.3f} {r['new_task']:9.3f} {r['forgetting']:7.3f} {r['backward_transfer']:7.3f} {r['forward_transfer']:7.3f} {r['param_growth']:8.2f} {r['memory_bytes']:12d} {'OUI' if m == 'replay' else 'non':>15s}")

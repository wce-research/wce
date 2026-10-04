"""Démo : échantillons → primitives candidates → recherche → programme découvert → extrapolation (famille HELD-OUT)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from core.boundary_methods import evaluate_all  # noqa: E402
from core.boundary_program import atoms, explain, render  # noqa: E402
from core.boundary_world import instances, sample_task  # noqa: E402

if __name__ == "__main__":
    inst = instances(["xor_halfspaces"], 3, 1)[0]; g = torch.Generator().manual_seed(0); X, y = sample_task(inst, 200, g)
    print("Famille cachée au système (jamais nommée dans le code d'induction). 200 observations dans x0 ∈ [−2 ; 0,5].")
    print("Atomes candidats :", [render(a) for a in atoms()])
    R = evaluate_all(inst, X, y, g, methods=("memorization", "mlp", "oracle", "bpi_full", "bpi_active"))
    for m, r in R.items():
        if r: print(f"  {m:13s} région {r['in_region']:.3f} | EXTRAPOLATION {r['extrapolation']:.3f} | paramètres {r['params']:4d} | évaluations {r['evaluations']:4d} | programme : {r.get('program', '—')}")

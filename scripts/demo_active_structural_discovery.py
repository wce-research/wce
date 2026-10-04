"""Démo : candidats → désaccord entre classes → expérience choisie → observation → postérieure → approfondir
seulement si nécessaire → fonction découverte."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from core.active_structure import ActiveStructuralDiscovery, train_policy  # noqa: E402
from core.active_world import balanced_instances, noisy_task, train_tasks  # noqa: E402
from core.boundary_program import predict, render  # noqa: E402
from core.boundary_world import full_region, out_region  # noqa: E402

if __name__ == "__main__":
    pol = train_policy(train_tasks(42, 0.05), eps=0.05)
    inst = balanced_instances(["xor_halfspaces"], 7, 1)[0]; g = torch.Generator().manual_seed(1); X, y = noisy_task(inst, 120, g, 0.05)
    asd = ActiveStructuralDiscovery(pol, budget=16, eps=0.05); asd.eps_world = 0.05
    r = asd.run(X, y, inst.label, full_region, torch.Generator().manual_seed(2))
    print("Entropie structurelle au fil des requêtes :", [round(e, 2) for e in r.entropy_curve])
    print("Taille de l'espace des versions          :", r.version_space_curve)
    print(f"Profondeur atteinte {r.depth} ; requêtes {r.queries} ; évaluations {r.evaluations} ; statut {r.status}")
    print("Programme :", render(r.program))
    F, O = full_region(3000, g), out_region(2000, g)
    print(f"Exactitude région complète {float((predict(r.program, F) == inst.label(F)).float().mean()):.3f} ; extrapolation {float((predict(r.program, O) == inst.label(O)).float().mean()):.3f}")

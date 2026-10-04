"""Démo : famille de profondeur 3/4 → proposeur → requêtes actives → élagage → résolution ; guidé vs non guidé."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from core.active_structure import ActiveStructuralDiscovery, train_policy  # noqa: E402
from core.active_world import balanced_instances, noisy_task, train_tasks  # noqa: E402
from core.boundary_program import predict, render  # noqa: E402
from core.boundary_world import full_region  # noqa: E402

if __name__ == "__main__":
    pol = train_policy(train_tasks(43, 0.05), eps=0.05); print("Proposeur appris :", len(pol.tot), "contextes d'expansion")
    inst = balanced_instances(["kite"], 4, 1)[0]; g = torch.Generator().manual_seed(1); X, y = noisy_task(inst, 120, g, 0.05); F = full_region(3000, g)
    for guided in (True, False):
        asd = ActiveStructuralDiscovery(pol, guided=guided, eps=0.05); asd.eps_world = 0.05; r = asd.run(X, y, inst.label, full_region, torch.Generator().manual_seed(2))
        print(f"{'guidé    ' if guided else 'non guidé'} : profondeur {r.depth}, évaluations {r.evaluations:3d}, requêtes {r.queries}, {r.seconds} s, exactitude {float((predict(r.program, F) == inst.label(F)).float().mean()):.3f} | {render(r.program)[:80]}")

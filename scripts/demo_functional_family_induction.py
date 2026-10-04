"""Démo : programmes d'AST différents → classe fonctionnelle (repère canonique) → famille (orbite) → réutilisation."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from core.active_experiments import discovered_programs  # noqa: E402
from core.active_structure import induce_ast_families, induce_orbit_families, reuse_family, train_policy  # noqa: E402
from core.active_world import balanced_instances, noisy_task, train_tasks, transformed  # noqa: E402
from core.boundary_program import predict, render  # noqa: E402
from core.boundary_world import full_region  # noqa: E402

if __name__ == "__main__":
    pol = train_policy(train_tasks(44, 0.05), eps=0.05); progs = discovered_programs(["crescent", "ring"], 5, 3, pol)
    for f, p in progs: print("  découvert :", render(p)[:80])
    P = [p for _, p in progs]
    print("Familles AST :", len(induce_ast_families(P)), "; familles fonctionnelles canoniques :", len(F := induce_orbit_families(P)))
    for f in F: print("  famille : support", f.support, "représentations AST", len({str(r.skeleton()) for r in f.representations}))
    inst = transformed(balanced_instances(["crescent"], 77, 1)[0], random.Random(1)); g = torch.Generator().manual_seed(2); X, y = noisy_task(inst, 120, g, 0.05)
    p, ev = reuse_family(F, X, y); R = full_region(3000, g)
    print(f"Instance transformée (rotation/échelle/réflexion) via famille : exactitude {float((predict(p, R) == inst.label(R)).float().mean()):.3f} en {ev} évaluations")

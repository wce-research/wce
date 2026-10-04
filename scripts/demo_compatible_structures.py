"""Démo légère : ensemble de structures compatibles, UNKNOWN, entropie avant / après mesure (inspiration Boltzmann)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from core.boundary_program import predict, render
from core.robust_reconstruction import *
from core.robust_world import *

if __name__ == "__main__":
    t = make_task(11, noise=0.12, depth=2, n=40); obs = Observations.build(t.X, t.y, SENSOR_MODELS["standard"], 0.12); g = torch.Generator().manual_seed(2)
    r0 = RobustStructuralReconstruction(planner=RobustMeasurementPlanner("efa", world_noise=0.12), budget=0).run(obs, t.oracle, full_region, full_region(300, g), g)
    print("AVANT mesure active : entropie", round(r0.css.entropy(), 3), "| réponse :", "UNKNOWN" if r0.answer == "UNKNOWN" else render(r0.answer)[:50])
    for e in sorted(r0.css.entries, key=lambda e: -e.mass)[:4]: print(f"   masse {e.mass:.3f} complexité {e.complexity:.1f} {render(e.program)[:60]}")
    r1 = RobustStructuralReconstruction(planner=RobustMeasurementPlanner("efa", world_noise=0.12), budget=12).run(obs, t.oracle, full_region, full_region(300, g), g)
    print("APRÈS", r1.queries, "mesures : entropie", round(r1.css.entropy(), 3), "| réponse :", "UNKNOWN" if r1.answer == "UNKNOWN" else render(r1.answer)[:50], "| statut", r1.status)

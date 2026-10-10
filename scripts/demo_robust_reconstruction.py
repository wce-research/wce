"""Light demo: noisy observations (8%) → robust likelihood → compatible structures → reconstruction."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from core.boundary_program import predict, render
from core.robust_reconstruction import *
from core.robust_world import *

if __name__ == "__main__":
    t = make_task(2, noise=0.08, depth=2); obs = Observations.build(t.X, t.y, SENSOR_MODELS["standard"], 0.08); F = full_region(3000, torch.Generator().manual_seed(1))
    print("Truth (hidden):", render(t.world.truth))
    for name, kw in (("naive fit + threshold-based deepening", dict(likelihood="naive", depth_rule="threshold")), ("robust likelihood + Bayes factor", dict())):
        r = RobustStructuralReconstruction(**kw).run(obs); p = r.css.best().program
        print(f"{name:42s} -> {render(p)[:60]:62s} AST {p.size()} | accuracy {float((predict(p, F) == t.world.label(F)).float().mean()):.3f} | discrepancy: {r.discrepancy['label']} | answer: {'UNKNOWN' if r.answer == 'UNKNOWN' else 'structure'}")

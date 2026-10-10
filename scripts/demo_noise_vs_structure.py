"""Light demo: A = simple structure + 8% noise; B = complex structure + 0% — MDL with / without measurement model."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from core.boundary_program import predict, render
from core.robust_reconstruction import *
from core.robust_world import *

if __name__ == "__main__":
    for case, t in zip(("A simple + 8% noise", "B complex + 0% noise"), hard_noise_pair(503)):
        obs = Observations.build(t.X, t.y, SENSOR_MODELS["standard"], t.spec.rate); print(f"\n{case} — truth: AST {t.world.truth.size()}")
        for name, kw in (("naive/threshold", dict(likelihood="naive", depth_rule="threshold")), ("robust/Bayes", dict())):
            r = RobustStructuralReconstruction(**kw).run(obs); p = r.css.best().program; m1, m2 = mdl(p, obs, "naive"), mdl(p, obs)
            print(f"  {name:14s} AST {p.size()} | MDL without measurement model {m1['total']:.1f} bits | with {m2['total']:.1f} bits (structure {m2['structure_bits']:.1f} + parameters {m2['parameter_bits']:.1f} + measurement {m2['measurement_error_bits']:.1f}) | {r.discrepancy['label']}")

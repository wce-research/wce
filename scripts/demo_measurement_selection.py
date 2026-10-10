"""Light demo: measurement choice — expected functional gain (EFA) vs structural IG vs disagreement; trade-off sensors."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from core.boundary_program import predict, render
from core.robust_reconstruction import *
from core.robust_world import *

if __name__ == "__main__":
    t = make_task(5, noise=0.08, depth=2); obs = Observations.build(t.X, t.y, SENSOR_MODELS["standard"], 0.08); g = torch.Generator().manual_seed(1)
    r = RobustStructuralReconstruction(planner=None).run(obs); css = r.css
    rr = RobustStructuralReconstruction(planner=RobustMeasurementPlanner("efa", world_noise=0.08), budget=0).run(obs, t.oracle, full_region, full_region(300, g), g); css = rr.css
    css.add_parameter_samples(obs, "contamination", 0.02); pool, probes = full_region(300, g), full_region(300, g)
    for name, sc in (("EFA", expected_functional_accuracy_gain(css, pool, probes, 0.13)), ("structural IG", structural_ig(css, pool, 0.13)), ("functional IG", functional_ig(css, pool, probes, 0.13)), ("disagreement", disagreement(css, pool))):
        i = int(sc.argmax()); print(f"{name:15s} best point {[round(v, 2) for v in pool[i].tolist()]} score {float(sc[i]):.4f}")
    ch = sensor_tradeoff_choice(css, pool, probes, 16.0, 0.08); print("Chosen sensor (cost / noise trade-off):", ch.sensor, "| action:", ch.action, "| utility", round(ch.utility, 4))
    print("Entropy of the compatible structures:", round(css.entropy(), 3), "| functional entropy:", round(css.functional_entropy(probes), 3))

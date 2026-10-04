"""Démo légère : A = structure simple + 8 % de bruit ; B = structure complexe + 0 % — MDL avec / sans modèle de mesure."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from core.boundary_program import predict, render
from core.robust_reconstruction import *
from core.robust_world import *

if __name__ == "__main__":
    for case, t in zip(("A simple + bruit 8 %", "B complexe + bruit 0 %"), hard_noise_pair(503)):
        obs = Observations.build(t.X, t.y, SENSOR_MODELS["standard"], t.spec.rate); print(f"\n{case} — vérité : AST {t.world.truth.size()}")
        for name, kw in (("naïf/seuil", dict(likelihood="naive", depth_rule="threshold")), ("robuste/Bayes", dict())):
            r = RobustStructuralReconstruction(**kw).run(obs); p = r.css.best().program; m1, m2 = mdl(p, obs, "naive"), mdl(p, obs)
            print(f"  {name:14s} AST {p.size()} | MDL sans modèle de mesure {m1['total']:.1f} bits | avec {m2['total']:.1f} bits (structure {m2['structure_bits']:.1f} + paramètres {m2['parameter_bits']:.1f} + mesure {m2['measurement_error_bits']:.1f}) | {r.discrepancy['label']}")

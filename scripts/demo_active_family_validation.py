"""Démo légère : famille candidate → nouvelle instance → ajustement a priori-famille + contrôle prédictif → promotion ou non."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from core.boundary_program import predict, render
from core.robust_reconstruction import *
from core.robust_world import *

if __name__ == "__main__":
    a = make_task(21, noise=0.05, depth=2); fam = RobustStructuralReconstruction().run(Observations.build(a.X, a.y, SENSOR_MODELS["standard"], 0.05)).css.best().program; fam_init = fam.param_vector(); fam = fam.skeleton()
    print("Famille candidate (squelette) :", render(fam))
    for name, t in (("autre instance quelconque", make_task(34, noise=0.05, depth=2)), ("même monde, nouvelles observations", make_task(21, noise=0.05, depth=2, n=90))):
        o = Observations.build(t.X, t.y, SENSOR_MODELS["standard"], 0.05); p, ll = fit_robust(fam, o, n_init=6, init=fam_init); g = torch.Generator().manual_seed(3)
        Xn = full_region(60, g); ppc = posterior_predictive_check(p, Observations.build(Xn, t.oracle(Xn, "standard", g), SENSOR_MODELS["standard"], 0.05), g=g)
        s = RobustStructuralReconstruction().run(o).css.best()
        print(f"  {name:36s} évidence famille {log_evidence(p, o, ll):.1f} vs recherche {s.evidence:.1f} | contrôle prédictif : erreurs {ppc['errors']} (p={ppc['p_errors']:.2f}) -> {'PROMUE' if (log_evidence(p, o, ll) >= s.evidence - 2 and not ppc['reject']) else 'non promue'}")

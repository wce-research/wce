"""Démo : ambiguïté entre programmes → choix d'expériences (gain d'information) → élimination → identification."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from core.boundary_methods import stable_search  # noqa: E402
from core.boundary_program import ProgramPosterior, fit, predict, render  # noqa: E402
from core.boundary_world import full_region, instances, sample_task  # noqa: E402

if __name__ == "__main__":
    inst = instances(["wedge_ring"], 4, 1)[0]; g = torch.Generator().manual_seed(2); X, y = sample_task(inst, 80, g)
    s = stable_search(X, y); post = ProgramPosterior([c[0] for c in s.candidates][:8]); post.update(X, y); F = full_region(2000, g)
    print("Candidats compatibles (P) :", [(render(p, 1), round(float(w), 3)) for p, w in zip(post.programs, post.weights)][:4])
    print("Ambiguïté sur la région complète :", post.ambiguous(F), "; prédictions UNKNOWN :", float((post.predict_or_unknown(F) == -1).float().mean()))
    pool = full_region(600, g); Xa, ya = X.clone(), y.clone()
    for q in range(1, 13):
        ig = post.info_gain(pool); i = int(ig.argmax()); xq = pool[i:i + 1]; yq = inst.label(xq); Xa, ya = torch.cat([Xa, xq]), torch.cat([ya, yq])
        if q % 4 == 0: post = ProgramPosterior([fit(c.skeleton(), Xa, ya, steps=100, n_init=1)[0] for c in post.programs] + post.programs); post.update(Xa, ya)
        else: post.update(xq, yq)
        m, p = post.map(); acc = float((predict(m, F) == inst.label(F)).float().mean())
        print(f"  expérience {q:2d} : x={[round(v, 2) for v in xq[0].tolist()]} IG={float(ig[i]):.3f} -> MAP P={p:.3f}, exactitude réelle {acc:.3f}")

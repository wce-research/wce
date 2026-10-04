"""Calibration (TRAIN + VALIDATION uniquement) des tolérances de la stabilité fonctionnelle sous bruit :
pour chaque niveau de bruit, programmes découverts FONCTIONNELLEMENT corrects → 90e centile du désaccord entre
moitiés et de l'erreur en excès. Écrit results/active_stability_calibration.json (table utilisée ensuite)."""
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from core.active_structure import ActiveStructuralDiscovery, SearchPolicy, fit_from, noise_aware_stability  # noqa: E402
from core.active_world import NOISE_LEVELS, balanced_instances, noisy_task  # noqa: E402
from core.boundary_program import predict  # noqa: E402
from core.boundary_world import TRAIN, VALIDATION, full_region  # noqa: E402


def job(a):
    inst_seed, noise = a; torch.set_num_threads(1); out = []
    for inst in balanced_instances(TRAIN + VALIDATION, 700 + inst_seed, 1):
        g = torch.Generator().manual_seed(inst_seed); X, y = noisy_task(inst, 120, g, noise)
        asd = ActiveStructuralDiscovery(guided=False, eps=noise, budget=16, max_depth=3); asd.eps_world = noise
        r = asd.run(X, y, inst.label, full_region, torch.Generator().manual_seed(inst_seed + 1))
        F = full_region(3000, g); ok = float((predict(r.program, F) == inst.label(F)).float().mean()) >= 0.97
        n = len(X) // 2; Xa = X; ya = y
        a_, b_ = fit_from(r.program, X[:n], y[:n]), fit_from(r.program, X[n:], y[n:])
        dis = float((predict(a_, X) != predict(b_, X)).float().mean()); err = float((predict(r.program, X) != y).float().mean())
        out.append({"noise": noise, "correct": ok, "disagreement": dis, "excess": err - noise})
    return out


if __name__ == "__main__":
    jobs = [(s, nz) for s in range(4) for nz in NOISE_LEVELS]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 12) as p: rows = [r for rs in p.map(job, jobs) for r in rs]
    table = {}
    for nz in NOISE_LEVELS:
        c = [r for r in rows if r["noise"] == nz and r["correct"]]
        q = lambda k: sorted(r[k] for r in c)[int(0.9 * (len(c) - 1))] if c else 0.1
        table[str(nz)] = (round(q("disagreement") + 0.01, 4), round(max(q("excess"), 0.0) + 0.01, 4))
    json.dump({"table": table, "rows": rows}, open("results/active_stability_calibration.json", "w"), indent=1)
    print(table, sum(r["correct"] for r in rows), "/", len(rows))

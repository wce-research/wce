"""Demo: several instances → anti-unification of the ASTs → parameterized family → new instance solved by the macro."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from core.boundary_methods import split, stable_search  # noqa: E402
from core.boundary_program import BoundaryProgramSearch, anti_unify_programs, predict, render  # noqa: E402
from core.boundary_world import full_region, instances, sample_task  # noqa: E402

if __name__ == "__main__":
    g = torch.Generator().manual_seed(1); progs = []
    for inst in instances(["crescent"], 5, 3):
        X, y = sample_task(inst, 200, g); s = stable_search(X, y); progs.append(s.best); print("  instance:", render(s.best))
    macro, tpl = anti_unify_programs(progs)
    print("Anti-unification:", render(tpl), "->", "parameterized family" if macro else "structural template (different structures)")
    if macro:
        print("  variables:", macro.variables(), "ranges:", [(round(a, 2), round(b, 2)) for a, b in zip(macro.param_lo, macro.param_hi)])
        inst = instances(["crescent"], 99, 1)[0]; X, y = sample_task(inst, 200, g); Xt, yt, Xv, yv = split(X, y)
        a = BoundaryProgramSearch(macros=[macro.skeleton]).run(Xt, yt, Xv, yv); b = BoundaryProgramSearch().run(Xt, yt, Xv, yv); F = full_region(3000, g)
        for name, r in (("macro", a), ("full search", b)):
            print(f"  new instance via {name:18s}: accuracy {float((predict(r.best, F) == inst.label(F)).float().mean()):.3f} ; evaluations {r.stats['evaluated']}")

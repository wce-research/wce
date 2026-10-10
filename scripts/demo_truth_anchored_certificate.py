"""Demo: WRONG but self-consistent rule → active counterexamples → certificate refuted (vs correct rule accepted)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from core.boundary_methods import stable_search  # noqa: E402
from core.boundary_program import predict, render  # noqa: E402
from core.boundary_world import full_region, instances, sample_task  # noqa: E402
from core.generative_certificate import ContinuousGenerator  # noqa: E402
from core.truth_anchor import BoundaryGenerativeCertificate, collect_counterexamples  # noqa: E402

if __name__ == "__main__":
    inst = instances(["two_lobe"], 2, 1)[0]; g = torch.Generator().manual_seed(3); X, y = sample_task(inst, 240, g); F = full_region(3000, g)
    for name, mask in (("rule learned on a SUB-REGION (wrong elsewhere)", X[:, 0] < -1.2), ("rule learned on the whole support", torch.ones(len(X), dtype=torch.bool))):
        s = stable_search(X[mask], y[mask]); prog = s.best; cert = BoundaryGenerativeCertificate(prog, ContinuousGenerator(k=2).fit(X[mask]))
        print(f"\n{name}: {render(prog)} ; real accuracy {float((predict(prog, F) == inst.label(F)).float().mean()):.3f}")
        print("  self-consistency:", cert.check(lambda Q: predict(prog, Q), g)["reference_consistency"])
        for pol in ("active", "random"):
            cert.anchor = collect_counterexamples(prog, inst.label, g, 10, [c[0] for c in s.candidates[1:6]], full_region, pol, support=X[mask])
            e, k = cert.anchor.error(lambda Q: predict(prog, Q))
            print(f"  anchors {pol:6s}: {k} errors / 10 -> {'REFUTED' if cert.anchor.refutes(lambda Q: predict(prog, Q)) else 'accepted'}")

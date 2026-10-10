"""Demo: new task → candidate adaptation (fast weights) → invariant checks → consolidate or rollback."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.safe_plasticity import METHODS, make_invariants, old_model, run_method, safe_consolidate, task_data, train_fast  # noqa: E402
import copy  # noqa: E402

if __name__ == "__main__":
    base = old_model(0); inv = make_invariants(base, 0)
    print("Old model: accuracies", {k: round(v, 3) for k, v in inv.reference.items()})
    xb, yb, xbv, ybv = task_data("B", 100)
    m = copy.deepcopy(base); mods = tuple(m.topological_order())
    print("\n1) Candidate adaptation WITHOUT constraint (fast weights), then check:")
    rep = safe_consolidate(m, train_fast(m, mods, xb, yb, 600), xbv, ybv, inv, base.accuracy(xbv, ybv))
    print(f"   decision: {rep.decision}; gain {rep.task_gain:+.3f}; reasons: {rep.reasons}")
    print(f"   retention after decision: {inv.retention(m)} (slow weights intact on rollback)")
    print("\n2) Candidate adaptation CONSTRAINED by the output-equivalence invariant (unlabelled anchors):")
    m = copy.deepcopy(base)
    import torch
    ox = torch.cat([task_data(t, 0)[0][:300] for t in "AC"])
    with torch.no_grad(): an = (ox, m.forward(ox))
    rep = safe_consolidate(m, train_fast(m, mods, xb, yb, 600, anchors=an, lam=0.3), xbv, ybv, inv, base.accuracy(xbv, ybv))
    print(f"   decision: {rep.decision}; gain {rep.task_gain:+.3f}; radius before/after {rep.radius_before:.3f}/{rep.radius_after:.3f}; interaction {rep.dangerous_interaction:+.3f}; {rep.reasons}")
    print("\n3) Comparison of the 5 methods (seed 0):")
    for meth in METHODS:
        r = run_method(meth, 0, base=base)
        print(f"   {meth:22s} new {r['new_task']:.3f} | retention {r['retention']:.3f} | forgetting {r['forgetting']:.3f} | params changed {r['parameters_changed']:5d} | rollbacks {r['rollbacks']} | {r['consolidation']}")

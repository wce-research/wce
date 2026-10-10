"""Demo: model → weight perturbation → sensitivity → survival of the invariants → stability radius
(EMPIRICAL “KAM-inspired” stability — no KAM torus, no Hamiltonian system is claimed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from core.generalized_credit import base_model  # noqa: E402
from core.perturbation import DifferentiableSensitivity, Perturbation, model_response, new_graph, sensitivity_curve, traced, unit_direction  # noqa: E402
from core.stability import (PerturbationInteractionGraph, accuracy_invariant, detect_transitions, perturbed_model,  # noqa: E402
                            routing_transition_observer, stability_map, stability_radius)

if __name__ == "__main__":
    m, xv, yv = base_model(0); g = new_graph()
    print(f"Model: {m.topological_order()}; old-task accuracy = {m.accuracy(xv, yv):.3f}")
    W = m.modules["expert_a"].slow["w1"]; d = unit_direction(W.shape, 7)
    p = Perturbation("parameter", ("expert_a", "w1"), 0.1, d); traced(g, p)
    r = model_response(m, p.apply(m), xv, yv, p)
    print(f"\n1) {p.name} (traceable, reversible: {p.invertible}) -> Δoutput {r.output_change:.4f}, decisions changed {r.decision_change:.3f}, Δloss {r.task_loss_change:+.4f}")
    back = p.inverse().apply(p.apply(m))
    print(f"   inverse applied: max weight deviation {float((back.modules['expert_a'].slow['w1'] - W).abs().max()):.2e}")
    curve = sensitivity_curve(lambda e: model_response(m, Perturbation('parameter', ('expert_a', 'w1'), e, d).apply(m), xv, yv).output_change, 1e-3, 6)
    print("\n2) Local sensitivity S = ||ΔF|| / ||δ|| for ε, 2ε, 4ε, …:", [round(s, 3) for s in curve["sensitivity"]], f"; log-log slope {curve['loglog_slope']:.3f} (≈1: linear regime)")
    cmp = DifferentiableSensitivity(m).compare(xv[:64], yv[:64])
    print(f"   autograd vs finite differences: input Jacobian rel. err. {cmp['input_rel_error']:.1e}; weight gradients max err. {cmp['weight_rel_error_max']:.1e}")
    print("\n3) Survival of the invariant “accuracy ≥ 0.98” (16 draws per amplitude):")
    for mod in ("enc", "expert_a", "head"):
        reg = stability_radius(accuracy_invariant(m, xv, yv, (mod,), 0.98), 0.9, hi=1.0, trials=16, iters=5, name=mod)
        print(f"   {mod:9s} stability radius r_I ≈ {reg.radius:.3f}  (curve: {[(round(c.magnitude, 3), round(c.probability, 2)) for c in reg.curve if c.magnitude >= 0.1]})")
    print("\n4) Stability map (module × amplitude -> P(survival)):")
    for mod, row in stability_map(m, xv, yv, 0.98, grid=(0.1, 0.2, 0.4, 0.8), trials=8).items(): print(f"   {mod:9s}", row)
    def resp(combo):
        with torch.no_grad():
            f = torch.nn.functional.cross_entropy
            return float(f(perturbed_model(m, combo, 0.5, 3).forward(xv), yv) - f(m.forward(xv), yv))
    G = PerturbationInteractionGraph(["enc", "expert_a", "head"], resp)
    print("\n5) Non-additive interactions (StructuralResonanceCandidate — analogy, not a physical resonance):")
    for e in G.dangerous(threshold=0.01): print(f"   {e.components} interaction {e.interaction:+.3f} (relative {e.relative:.2f})")
    tb = detect_transitions([0, 2, 4, 8, 16, 32, 64], routing_transition_observer(m, xv, yv), jump=0.05)
    print("\n6) Transition boundaries (parasitic routing expert_b -> head of increasing strength):")
    for b in tb: print(f"   {b.quantity}: threshold ≈ {b.threshold:.3f} ({b.kind}) {b.before} -> {b.after}")
    print(f"\nTrace: {len(g.nodes)} perturbation node(s) in the TransformGraph")

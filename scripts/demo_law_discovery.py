"""Demo: world with a HIDDEN LAW → hypotheses → chosen experiments (information gain) → discovered transformation
→ invariant tests → new skill + generative certificate → reuse."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from core.discovery import LawWorld, discover_invariants, discover_transformation, form_skill, identify_law, law_correct, rot  # noqa: E402

if __name__ == "__main__":
    w = LawWorld(phi=3 * math.pi / 8, gain=1.5, mirror=True, noise=0.25, seed=2)
    print("Language: « pousse le mystère vers l'est » ->", [round(v, 2) for v in w.say("pousse le mystère vers l'est").tolist()], "(the ball would go towards (1, 0))")
    for pol in ("active", "random"):
        w2 = LawWorld(phi=w.phi, gain=w.gain, mirror=w.mirror, noise=0.25, seed=2); f = identify_law(w2, pol)
        print(f"  {pol:7s} : {f['experiments']:2d} experiments -> φ={f['phi']:.3f} gain={f['gain']} mirror={f['mirror']} (P={f['p']:.3f}) ; correct: {law_correct(w2, f)}")
    w3 = LawWorld(phi=w.phi, gain=w.gain, mirror=w.mirror, noise=0.25, seed=2); f = identify_law(w3, "active")
    spec, inv, inv_ok, cert = form_skill(w3, f)
    print(f"\nTransformationSpec: {spec.name}\nInvariants: {inv} ; perturbation tests: {inv_ok}")
    print(f"Generative certificate: {cert.skill} ({cert.memory_bytes()} bytes, provenance « {cert.provenance} »)")
    U = torch.randn(5, 2); print("Reuse: predictions vs world:", [[round(a, 2) for a in r] for r in spec(U).tolist()][:2], [[round(a, 2) for a in w3.law(tuple(u.tolist())).tolist()] for u in U][:2])
    P = torch.randn(6, 2); Q = (P * torch.tensor([1.0, -1.0])) @ rot(0.7).T + torch.tensor([0.5, -1.0])
    print("Transformation discovery (P -> Q):", discover_transformation(P, Q)[:2])
    h = discover_invariants(lambda P, g: P @ rot(float(torch.rand(1, generator=g)) * 6).T + torch.randn(2, generator=g))
    print("Invariants under rotation+translation:", {k: v.status for k, v in h.items()})

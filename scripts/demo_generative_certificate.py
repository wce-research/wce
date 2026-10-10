"""Demo: old skill → raw data DISCARDED → central / boundary / rare / adversarial probes
generated → learning of a new skill → check of the old invariant."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from core.continuous_world import Regime, random_rule  # noqa: E402
from core.continuum import Config, ContinuumSystem, mlp  # noqa: E402

if __name__ == "__main__":
    rng = random.Random(3); g = torch.Generator().manual_seed(3)
    A = Regime("A", random_rule(rng, "reldist", torch.zeros(16)), torch.zeros(16), 1.0, torch.zeros(8), "new")
    B = Regime("B", random_rule(rng, "sine", torch.ones(16)), torch.ones(16), 1.0, torch.ones(8) * 2, "new")
    s = ContinuumSystem(Config("demo", cert="generative", discovery=False), seed=0)
    X, C, y = A.sample(512, g); s.learn(X, C, y, t=0); skill = s.contexts[0].skill; cert = s.certs[skill]
    raw = X.numel() * 4; del X, C, y
    print(f"Skill A learned ({s.skills[skill][0]}) ; raw data DISCARDED ({raw} bytes) ; certificate: {cert.memory_bytes()} bytes")
    print(f"  learned generator: mean/diag/PCA rank {cert.generator.k} ; missing {cert.generator.missing:.2f} ; outliers {cert.generator.outlier:.2f} ; reference = {cert.reference_kind}")
    P = cert.probes(32, g, model_fn=lambda Z: mlp(s.modules[s.skills[skill][1]], Z), categories=("central", "boundary", "rare", "counterfactual", "high_sensitivity"))
    for k, v in P.items(): print(f"  probes {k:16s} : {len(v):3d} new situations (mean norm {float(v.norm(dim=1).mean()):.2f})")
    adv = cert.generator.sample_adversarial(32, g, lambda Z: mlp(s.modules[s.skills[skill][1]], Z), cert.reference, start=P["boundary"].clone())
    print(f"  adversarial probes: disagreement with the reference {float((s.predict_skill(skill, adv) != cert.reference(adv)).float().mean()):.3f} (vs central {float((s.predict_skill(skill, P['central']) != cert.reference(P['central'])).float().mean()):.3f})")
    X, C, y = B.sample(512, g); rec = s.learn(X, C, y, t=1)
    print(f"\nNew skill B: {rec['kind']} (accuracy {rec['acc']:.3f})")
    ok, errs = cert.check(lambda Z: s.predict_skill(skill, Z), seed=99)
    print(f"Check of the invariant of A (fresh probes): {'OK' if ok else 'VIOLATED'} {({k: round(v, 3) for k, v in errs.items()})}")
    Xt, Ct, yt = A.test_set(7); print(f"Actual accuracy of A (truth): {float((s.predict(Xt, Ct) == yt).float().mean()):.3f}")

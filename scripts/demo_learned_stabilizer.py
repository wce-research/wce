"""Demo: candidate transformation → predicted survival of the invariants (hand / learned / hybrid) → actual survival."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.compositional_plasticity import _restore, _snapshot  # noqa: E402
from core.learned_stabilizer import (LearnedStabilizerModel, built_state, hand_rule, hybrid_value, label,  # noqa: E402
                                     sample_transformation, simulate)

if __name__ == "__main__":
    train = []
    for s in (0, 1): net, certs = built_state(s); train += simulate(net, certs, 200, seed=s)
    model = LearnedStabilizerModel().fit(train)
    net, certs = built_state(3); rng = random.Random(5)
    print(f"Stabilizer learned on {len(train)} observations; NEW state (seed 3) with {len(certs)} certified skills.\n")
    shown = 0
    while shown < 8:
        fam, target, mag, spec = sample_transformation(net, rng, (0.02, 0.6))
        skill = rng.choice(list(certs)); ctx = {c.name: c.skill for c in net.contexts}
        p = model.predict(net, certs, fam, target, mag, skill); h = hand_rule(net, certs, fam, target, skill)
        snap = _snapshot(net); fast = {k: dict(v) for k, v in net.fast.items()}
        try: spec.execute(net)
        except Exception: _restore(net, snap); continue
        real = label(net, certs, skill, ctx); _restore(net, snap); net.fast = fast
        for m in [m for m in net.modules if m not in snap[0]]: net.modules.pop(m)
        print(f"  {fam:17s} target {str(target):16s} magnitude {mag:.3f} | skill {skill:28s} -> P(preserved) hand {h:.2f} learned {p:.2f} hybrid {hybrid_value(h, p, fam):.2f} | actual: {'preserved' if real else 'BROKEN'}")
        shown += 1

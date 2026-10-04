"""Démo : transformation candidate → survie prédite des invariants (main / appris / hybride) → survie réelle."""
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
    print(f"Stabilisateur appris sur {len(train)} observations ; état NOUVEAU (graine 3) avec {len(certs)} compétences certifiées.\n")
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
        print(f"  {fam:17s} cible {str(target):16s} amplitude {mag:.3f} | compétence {skill:28s} -> P(préservé) main {h:.2f} appris {p:.2f} hybride {hybrid_value(h, p, fam):.2f} | réel : {'préservé' if real else 'CASSÉ'}")
        shown += 1

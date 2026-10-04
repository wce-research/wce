"""Démo : faits calibrés -> EliminateCandidates -> identité L1 -> réponse ; même cas avec une source faible -> abstention."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.grounded_benchmark import seed_data  # noqa: E402
from core.independent_agent import Evidence  # noqa: E402
from core.source_benchmark import type_priors  # noqa: E402
from core.source_calibration import emissions, fit_supervised, install  # noqa: E402
from core.unified_agent import UnifiedAgent, UnifiedConfig  # noqa: E402


def scene(agent, describer, sensor_conf=True):
    agent.new_world()
    b = agent.belief
    b.inventory.update({"o1": "cube", "o2": "cube", "o3": "cube"})
    for o, c in (("o1", "blue"), ("o2", "green")):
        for s in (("sensor_A", "sensor_B") if sensor_conf else ("speaker_B",)):
            b.graph.add(("color", o), Evidence(s, c, 0, "structured"))
    agent._desc_sources.clear()
    for d in (describer if isinstance(describer, tuple) else (describer,)): agent._desc_sources["cube"]["red"].add(d)
    agent.solve_identity()


def main():
    data, splits, caps = seed_data(0)
    agent = UnifiedAgent(UnifiedConfig(identity_mode="l1")); agent.learn_captions(caps); agent.warm_up_sources(splits["train"])
    model = fit_supervised(emissions(agent, splits["train"]), "lower", type_priors(), selection=False)
    install(agent, model)
    print("Fiabilités effectives calibrées :", {s: round(model.estimate(s), 3) for s in ("sensor_A", "sensor_B", "document_A", "speaker_B")})
    for title, describer, conf in (("Faits calibrés (capteurs) + description par document_A", "document_A", True),
                                   ("Faits calibrés + « le cube rouge » attesté par document_A, speaker_A ET speaker_B", ("document_A", "speaker_A", "speaker_B"), True),
                                   ("Même scène, description par speaker_B (faible)", "speaker_B", True),
                                   ("Prémisses établies seulement par speaker_B", "document_A", False)):
        scene(agent, describer, conf)
        print(f"\n{title}")
        for o in ("o1", "o2", "o3"):
            bb = agent.belief.color(o); print(f"  color({o}) = {bb.value} {bb.status} p={bb.posterior:.3f}")
        rule = agent.explicit_elimination("cube", "red", min_posterior=agent.cfg.l1_min_premise)
        print(f"  EliminateCandidates(cube, red) : candidats {rule.candidates} exclus {rule.excluded} -> {rule.remaining}"
              f" (niveau L{rule.level}, confiance {rule.confidence:.3f})")
        for ctx, src in (("observation", describer if isinstance(describer, str) else describer[0]), ("query", None)):
            o, p = agent.identity_prob("cube", "red", ctx, src)
            thr = agent.threshold(ctx)
            verdict = f"identité L1 -> {o} (P={p:.3f} ≥ {thr})" if o and p >= thr else f"abstention (P={p:.3f} < {thr} ou prémisses insuffisantes)"
            print(f"  {ctx:11s}: {verdict}")
    print("\nL'identité L2 (correspondance jointe probabiliste) n'est jamais utilisée en mode L1.")


if __name__ == "__main__":
    main()

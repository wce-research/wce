"""Demo: source reliability — prior, observations, posterior, interval, effective reliability; multi-source belief."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.grounded_benchmark import SYSTEMS, seed_data  # noqa: E402
from core.independent_agent import Evidence, EvidenceGraph  # noqa: E402
from core.knowledge_boundary import true_reliabilities  # noqa: E402
from core.source_benchmark import type_priors  # noqa: E402
from core.source_calibration import emissions, fit_supervised, mnar_audit, reliability_points  # noqa: E402
from core.source_reliability import SourceReliabilityModel  # noqa: E402


def main():
    priors = type_priors()
    print("Hierarchical priors by type (empirical Bayes, calibration seeds 200–205):")
    for t, (a, b) in priors.items(): print(f"  {t:9s} Beta({a:.1f}, {b:.1f}) mean {a / (a + b):.3f}")
    data, splits, caps = seed_data(0)
    probe = SYSTEMS["no_elimination"](); probe.learn_captions(caps); probe.warm_up_sources(splits["train"])
    tr = emissions(probe, splits["train"]); te = emissions(probe, splits["test"])
    print("\nMNAR audit (text channel, training worlds):")
    for s, v in mnar_audit([(s, ok, p) for s, kt, ok, p, t in tr if t]).items():
        print(f"  {s:10s} emitted {v['emitted']} parsed {v['parsed']} | accuracy emitted {v['accuracy_emitted']} parsed {v['accuracy_parsed']} bias {v['selection_bias']:+}")
    model = fit_supervised(tr, "lower", priors, selection=False)
    truth = true_reliabilities()
    print("\nSources (labelled history of the TRAINING worlds; no tuning on the test):")
    for s in ("speaker_A", "speaker_B", "sensor_A", "document_A"):
        r = model.record(s); lo, hi = r.interval(0.9)
        print(f"  {s:10s} prior Beta({r.alpha0:.1f},{r.beta0:.1f}) | correct {r.correct:.0f} incorrect {r.incorrect:.0f} unknown {r.unknown:.0f}"
              f" | posterior {r.mean:.3f} CI90 [{lo:.3f}, {hi:.3f}] | effective (lower bound) {model.effective(s):.3f}"
              f" | risk-mean {model.effective(s, mode='risk'):.3f} | EM {probe.reliability.estimate(s):.3f} | nominal {truth[s]}")
    print("  reliability by key type (shrunk toward the source):",
          {kt: round(model.estimate_for("speaker_B", kt), 3) for kt in ("color", "loc", "height", "owner")})
    for name, m in (("EM", probe.reliability), ("hierarchical Beta", fit_supervised(tr, "mean", priors, selection=False))):
        rep, _ = reliability_points(m, te)
        print(f"  reliability calibration ({name}) on the TEST: " + ", ".join(f"{t} ECE {v['ece']}" for t, v in rep.items()))

    print("\nBelief built from several sources: “color(o9)”")
    for name, rel in (("EM (old)", probe.reliability), ("cautious Beta", model)):
        g = EvidenceGraph(rel)
        g.add(("color", "o9"), Evidence("speaker_B", "red", 0, "text"))
        g.add(("color", "o9"), Evidence("speaker_A", "red", 0, "text"))
        b = g.belief(("color", "o9"))
        print(f"  {name:13s} speaker_B + speaker_A say red -> {b.value} {b.status} p={b.posterior:.3f}")
        g.add(("color", "o9"), Evidence("sensor_A", "blue", 0, "structured"))
        b = g.belief(("color", "o9"))
        print(f"  {name:13s} + sensor_A says blue -> {b.value} {b.status} p={b.posterior:.3f} (for {b.supporting}, against {b.refuting})")
    m2 = SourceReliabilityModel(mode="lower")
    for _ in range(30): m2.observe("document_B", True)
    m2.groups["document_B"] = "speaker_A"
    g = EvidenceGraph(m2)
    g.add(("color", "o9"), Evidence("speaker_A", "green", 0, "text")); g.add(("color", "o9"), Evidence("document_B", "green", 0, "text"))
    print(f"  dependent sources (document_B copies speaker_A): votes counted once -> {g.belief(('color', 'o9')).supporting}")


if __name__ == "__main__":
    main()

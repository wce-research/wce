"""Démo : fiabilité des sources — a priori, observations, posterior, intervalle, fiabilité effective ; croyance multi-sources."""
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
    print("A priori hiérarchiques par type (Bayes empirique, graines de calibration 200–205) :")
    for t, (a, b) in priors.items(): print(f"  {t:9s} Beta({a:.1f}, {b:.1f}) moyenne {a / (a + b):.3f}")
    data, splits, caps = seed_data(0)
    probe = SYSTEMS["no_elimination"](); probe.learn_captions(caps); probe.warm_up_sources(splits["train"])
    tr = emissions(probe, splits["train"]); te = emissions(probe, splits["test"])
    print("\nAudit MNAR (canal texte, mondes d'entraînement) :")
    for s, v in mnar_audit([(s, ok, p) for s, kt, ok, p, t in tr if t]).items():
        print(f"  {s:10s} émises {v['emitted']} analysées {v['parsed']} | exactitude émise {v['accuracy_emitted']} analysée {v['accuracy_parsed']} biais {v['selection_bias']:+}")
    model = fit_supervised(tr, "lower", priors, selection=False)
    truth = true_reliabilities()
    print("\nSources (historique étiqueté des mondes d'ENTRAÎNEMENT ; aucun réglage sur le test) :")
    for s in ("speaker_A", "speaker_B", "sensor_A", "document_A"):
        r = model.record(s); lo, hi = r.interval(0.9)
        print(f"  {s:10s} a priori Beta({r.alpha0:.1f},{r.beta0:.1f}) | justes {r.correct:.0f} fausses {r.incorrect:.0f} inconnues {r.unknown:.0f}"
              f" | posterior {r.mean:.3f} IC90 [{lo:.3f}, {hi:.3f}] | effective (borne basse) {model.effective(s):.3f}"
              f" | moyenne-risque {model.effective(s, mode='risk'):.3f} | EM {probe.reliability.estimate(s):.3f} | nominale {truth[s]}")
    print("  fiabilité par type de clé (rétrécie vers la source) :",
          {kt: round(model.estimate_for("speaker_B", kt), 3) for kt in ("color", "loc", "height", "owner")})
    for name, m in (("EM", probe.reliability), ("Beta hiérarchique", fit_supervised(tr, "mean", priors, selection=False))):
        rep, _ = reliability_points(m, te)
        print(f"  calibration de la fiabilité ({name}) sur le TEST : " + ", ".join(f"{t} ECE {v['ece']}" for t, v in rep.items()))

    print("\nCroyance construite depuis plusieurs sources : « couleur(o9) »")
    for name, rel in (("EM (ancien)", probe.reliability), ("Beta prudent", model)):
        g = EvidenceGraph(rel)
        g.add(("color", "o9"), Evidence("speaker_B", "red", 0, "text"))
        g.add(("color", "o9"), Evidence("speaker_A", "red", 0, "text"))
        b = g.belief(("color", "o9"))
        print(f"  {name:13s} speaker_B + speaker_A disent rouge -> {b.value} {b.status} p={b.posterior:.3f}")
        g.add(("color", "o9"), Evidence("sensor_A", "blue", 0, "structured"))
        b = g.belief(("color", "o9"))
        print(f"  {name:13s} + sensor_A dit bleu -> {b.value} {b.status} p={b.posterior:.3f} (pour {b.supporting}, contre {b.refuting})")
    m2 = SourceReliabilityModel(mode="lower")
    for _ in range(30): m2.observe("document_B", True)
    m2.groups["document_B"] = "speaker_A"
    g = EvidenceGraph(m2)
    g.add(("color", "o9"), Evidence("speaker_A", "green", 0, "text")); g.add(("color", "o9"), Evidence("document_B", "green", 0, "text"))
    print(f"  sources dépendantes (document_B recopie speaker_A) : votes comptés une fois -> {g.belief(('color', 'o9')).supporting}")


if __name__ == "__main__":
    main()

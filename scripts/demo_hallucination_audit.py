"""Audit : observables géométriques des trajectoires, sorties correctes vs incorrectes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.language_core import SYSTEMS, build_core, make_scene, run_stream
from core.trajectory import OBSERVABLES, calibrate, evaluation_words, hallucination_audit


def fmt(z): return ", ".join(f"{k}={z[k]:.2f}" for k in OBSERVABLES + ("confidence",))


def main():
    print("=== WCE HALLUCINATION AUDIT ===")
    print("Observables = PROXIES (projections aléatoires + sliced-W2 sur passes bruitées) ; pas de W2 exact.")
    lc = build_core(**SYSTEMS["6_full_self_correction"]); run_stream(lc)
    exclude = evaluation_words()
    report = hallucination_audit(lc, exclude)
    monitor, estimator = calibrate(lc, exclude)
    lc.attach_monitor(monitor, estimator)
    cases = report["cases"]
    pick = lambda cat, ok: next((c for c in cases if c.category == cat and c.correct == ok), None)
    for title, case in (("Correct output (forme connue)", pick("known", True)),
                        ("Correct output (forme inédite)", pick("heldout_forms", True)),
                        ("Incorrect output (forme inédite)", pick("heldout_forms", False)),
                        ("Incorrect output (module corrompu)", pick("corrupted_module", False)),
                        ("Mot inconnu / piège", pick("unknown_or_trap", False) or pick("unknown_or_trap", True)),
                        ("Ambiguous output", pick("ambiguous", True) or pick("ambiguous", False)),
                        ("Mauvais routage forcé", pick("wrong_routing", False))):
        if case is None: continue
        print(f"\n{title}: « {case.word} » -> {case.predicted} (vérité {case.truth})")
        print(f"  Z = {fmt(case.z)}")
        print(f"  Geometric risk: {estimator.score(case.z):.2f} ({estimator.level(case.z)})")
    print("\nPolitique complète sur des phrases :")
    for text in ("le cube tombe", "le bloc montait", "le bloc zorphe", "tourne abcde", "le bloc tombe du support",
                 "le cube montagne", "renverse abcde", "la table est en bois"):
        r = lc.process(text, make_scene())
        risk = r.risk
        geo = f"{risk.geometric_level} ({risk.geometric_risk:.2f})" if risk and risk.geometric_risk is not None else "n/a"
        support = f"{risk.structural_support:.2f}" if risk else "n/a"
        print(f"  « {text} »: structural support {support} | geometric risk {geo} | statut {r.status} -> Final policy: {r.decision}")
    print(f"\nTest set (disjoint de la calibration) : {report['n_test']} cas, {report['n_incorrect']} erreurs neuronales")
    print("AUROC pour détecter une erreur neuronale :")
    for k, v in report["auroc"].items(): print(f"  {k:<26} {v}")
    nat = report["auroc_natural_errors_only"]
    print(f"AUROC sur erreurs NATURELLES seulement (sans module corrompu) : {nat}")
    print(f"Coefficients du GeometricRiskEstimator : {report['coefficients']}")
    print(f"Mauvais routage forcé : risque moyen {report['wrong_routing_risk_mean']} (trajectoire d'entrée identique -> invisible)")
    print("\nMoyennes par catégorie :")
    for k, v in report["by_category"].items(): print(f"  {k:<30} {v}")
    print("\nConclusion prudente : corrélation mesurée sur un corpus jouet, pas une preuve causale ;"
          " les contradictions structurelles et le mauvais routage restent invisibles à la géométrie d'entrée.")


if __name__ == "__main__":
    main()

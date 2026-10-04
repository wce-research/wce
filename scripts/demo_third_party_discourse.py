"""Démo : banc de discours tiers (jamais vu par le résolveur) — scénarios écrits à la main + second générateur."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.discourse_benchmark import lexicon_agent  # noqa: E402
from core.third_party_discourse import banc_tiers, nombre_questions, scenarios_aveugles, scenarios_main  # noqa: E402
from core.third_party_eval import (  # noqa: E402
    SYSTEMS, active_commands, active_observation, answer_calibration, by, calibrate_on, memory_scaling, metrics, run, run_scenario,
    self_correction,
)
from core.unified_distill import distill  # noqa: E402
from core.third_party_eval import calibration_bench  # noqa: E402

KEYS = ("success", "false_confident", "abstention", "clarification", "likely", "exact_class", "coverage", "identity_accuracy",
        "belief_accuracy")


def main():
    L = lexicon_agent(0)
    bench = banc_tiers()
    print(f"Banc tiers : {len(bench)} scénarios, {nombre_questions(bench)} questions "
          f"({len(scenarios_main())} écrits à la main, le reste par un second générateur à conventions différentes).")
    print("Le module du banc n'importe aucun composant de WCE (test AST).\n")
    print("Exemple de scénario (conventions tierces) :")
    s = scenarios_main()[8]
    print("  objets (vérité cachée) :", s["objets"])
    for t in s["tours"]: print("  ", t)
    rows, _ = run_scenario(L, s)
    for r in rows: print(f"   ? {r['text']}  attendu={r['expected']}  obtenu={r['got']}  ({r['status']}, {r['answer'].phrase})")

    cal = calibrate_on(L)                          # partition de calibration : autre graine du second générateur
    print(f"\nCalibration (hors test) : {cal['n_answer_points']} réponses, seuil de question {cal['thresholds']}")
    print("  confiance brute vs exactitude :", cal["raw_answer_calibration"]["bins"])
    print("\nSystèmes sur le banc tiers :")
    for name in SYSTEMS:
        m, rows = run(L, bench, name, cal)
        print(f"  {name:22s}", {k: m[k] for k in KEYS})
    m, rows = run(L, bench, "unified_calibrated", cal)
    main_rows = [r for r in rows if r["origin"] == "main"]
    print("\nScénarios écrits à la main seulement :", {k: v for k, v in metrics(main_rows).items() if k in KEYS})
    print("\nPar catégorie adversariale (écrits à la main) :")
    for k, v in by(main_rows, "category").items(): print(f"  {k:20s} succès {v['success']:.2f} faux confiants {v['false_confident']:.2f} (n={v['n']})")
    print("\nPar variation (second générateur) :")
    for k, v in by([r for r in rows if r["origin"] != "main"], "category").items():
        print(f"  {k:26s} succès {v['success']:.2f} faux confiants {v['false_confident']:.2f} (n={v['n']})")
    print("\nCourbes de calibration :")
    print("  identité :", m["identity_calibration"]["bins"], "ECE", m["identity_calibration"]["ece"])
    print("  croyance :", m["belief_calibration"]["bins"], "ECE", m["belief_calibration"]["ece"])
    print("  réponse finale :", answer_calibration(rows)["bins"])
    blind = scenarios_aveugles()
    print(f"\nLot AVEUGLE écrit après la mise au point ({len(blind)} scénarios, {nombre_questions(blind)} questions, évalué sans retouche) :")
    for name in SYSTEMS:
        mb, _ = run(L, blind, name, cal)
        print(f"  {name:22s}", {k: mb[k] for k in ("success", "false_confident", "abstention", "exact_class", "coverage")})
    print("\nObservation active (questions ASK du banc) :", active_observation(L))
    print("Observation active sur des commandes (identité exigée 0,99) :", active_commands(L))
    print("Auto-correction par révision du graphe de croyances :", self_correction(L))
    print("Distillation (élève propose, WCE vérifie) :", distill(L, calibration_bench(60, 23), bench, run_scenario))
    print("Mémoire (50/100/500 phrases) :", memory_scaling(L, (50, 100, 500)))
    from core.third_party_hard import dev_generated, dev_set, n_questions as nq
    dev = dev_set() + dev_generated()
    md, _ = run(L, dev, "unified_calibrated", cal)
    print(f"\nBanc tiers DIFFICILE — développement ({nq(dev)} questions) :", {k: md[k] for k in ("success", "false_confident", "abstention")})
    import json
    sealed = "results/sealed_blind.json"
    if os.path.exists(sealed):
        r = json.load(open(sealed))
        print(f"Lot SCELLÉ ({r['questions']} questions, évalué une seule fois ; résultat enregistré, jamais retouché) :")
        for name, v in r["systems"].items():
            print(f"  {name:22s}", {k: v["metrics"][k] for k in ("success", "false_confident", "abstention")})
    print("\nTaille du cœur : état unifié = 0 paramètre neuronal (formules explicites) ; élève d'identité distillé = 6 paramètres.")


if __name__ == "__main__":
    main()

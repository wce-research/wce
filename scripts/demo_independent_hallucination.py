"""Benchmark indépendant : hallucinations par catégorie, ablations, géométrie conditionnelle, coûts."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.independent_benchmark import (
    ablations, conditional_geometry, correction_benchmark, distillation_report, geometry_word_test,
    grammar_learning_curve, induced_classes, lexicon_growth, make_agent, prepare, representation_comparison,
    scaling, verification_costs,
)


def main():
    print("=== WCE INDEPENDENT HALLUCINATION BENCHMARK ===")
    setup = prepare()
    n = {k: sum(len(x["items"]) for x in v) for k, v in setup.splits.items()}
    print(f"Items par partition (par monde, aucun réglage sur le test) : {n} ; total {sum(n.values())}")
    print("\nAblations (test tenu à l'écart) :")
    print(f"  {'système':<24} succès  faux-sûrs  abstention  vérif./req.  params")
    res = ablations(setup)
    for name, r in res.items():
        print(f"  {name:<24} {r['success']:.3f}  {r['false_confident_rate']:.3f}      {r['abstention_rate']:.3f}       "
              f"{r['verification_cost']:<11} {r['parameters']}")
    print("\nTaxonomie des réponses fausses et affirmées :")
    for name in ("A_neural_only", "C_wce_only", "E_neural_wce_evidence", "G_full"):
        print(f"  {name:<24} {res[name]['taxonomy']}")
    print(f"\nPar catégorie (G_full) : {res['G_full']['by_category']}")
    print(f"Abstentions G_full : correctes {res['G_full']['correct_abstentions']}, inutiles {res['G_full']['unnecessary_abstentions']}")
    print(f"\nAuto-correction : {correction_benchmark(setup)}")
    print("\nCoûts de vérification :")
    for k, v in verification_costs(setup).items(): print(f"  {k:<22} {v}")
    g = geometry_word_test(setup)
    print(f"\nMoniteur géométrique (erreurs lexicales neuronales, mots hors lexique du test) : n={g['n']} erreurs={g['n_incorrect']}")
    print(f"  AUROC {g['auroc']} | AUPRC {g['auprc']} | ECE géométrie {g['calibration_ece_geometry']}")
    print(f"  erreurs naturelles seulement : {g['natural_only']}")
    for k, v in g["by_category"].items(): print(f"    {k:<38} {v}")
    c = conditional_geometry(setup)
    print(f"\nValeur conditionnelle de la géométrie (erreur du candidat par requête, test n={c['n_test']}, taux d'erreur {c['test_error_rate']}) :")
    for k, v in c["feature_sets"].items(): print(f"  {k:<32} {v}")
    print(f"  Δ AUROC (contrôles + géométrie − contrôles) : {c['delta_auroc_geometry_after_controls']}")
    print(f"  coefficients standardisés : {c['standardized_coefficients']}")
    print(f"  sous-ensemble avec mots neuronaux : {c['subset_with_neural_words']}")
    print(f"\nDistillation : {distillation_report(setup)}")
    print(f"\nCourbe grammaire (exemples -> précision) : {grammar_learning_curve()}")
    print(f"\nCroissance lexicale : ")
    for k, v in lexicon_growth().items(): print(f"  {k:>5} formes : {v}")
    print(f"\nCaractères vs sous-mots vs mots : {representation_comparison()}")
    agent = make_agent("G_full", setup)
    ic = induced_classes(agent)
    print(f"\nClasses lexicales induites (pureté {ic['purity_vs_meaning_class']}) : {ic['clusters']}")
    print("\nÉchelle du cœur neuronal (benchmark indépendant) :")
    for k, v in scaling(setup).items(): print(f"  {k:<7} {v}")


if __name__ == "__main__":
    main()

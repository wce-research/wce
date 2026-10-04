"""Croissance de la couverture : exemples -> analyse -> lexique -> abstention -> succès -> faux confiants."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.grounded_benchmark import (
    abstention_breakdown, budget_curve, correction_report, coverage_growth, domain_report, domain_transfer,
    geometry_holdout, learning_needs, lexical_scale, long_sentence_report, multi_seed, structural_distillation,
)


def fmt(s): return f"{s['mean']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]" if s else "n/a"


def main(seeds=(0, 1, 2, 3, 4, 5, 6, 7)):
    print("=== WCE COVERAGE GROWTH ===")
    print("\nExemples (légendes) -> couverture (graine 0) :")
    print("  légendes  analyse  lexique  abstention  inutiles  succès  faux-sûrs  rôles")
    for n, r in coverage_growth().items():
        print(f"  {n:>8}  {r['parse_coverage']:.3f}    {r['lexical_coverage']:.3f}    {r['abstention']:.3f}       "
              f"{r['unnecessary_abstention']:.3f}     {r['success']:.3f}   {r['false_confident']:.3f}      {r['role_accuracy']:.3f}")
    res = multi_seed(seeds)
    print(f"\nMulti-graines ({len(seeds)} graines : mondes, langue, bruit, légendes, partition, init. neuronale ; {res['seconds']} s)")
    print("moyenne [IC95]")
    for name, m in res["summary"].items():
        print(f"\n  {name}")
        for k in ("success", "false_confident", "abstention", "unnecessary_abstention", "parse_coverage", "parse_coverage_resolvable",
                  "role_accuracy", "lexical_coverage", "verification_cost", "routing_accuracy", "polysemy_accuracy"):
            print(f"    {k:<28} {fmt(m.get(k))}")
    print("\nComparaisons appariées (différence par graine a−b, IC95 t ; puis bootstrap sur items) :")
    for label, c in res["comparisons"].items():
        print(f"\n  {label}")
        for k in ("success", "false_confident", "unnecessary_abstention", "parse_coverage", "role_accuracy", "lexical_coverage", "verification_cost"):
            v = c[k]
            if v: print(f"    {k:<24} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  graines +{v['seeds_positive']} / −{v['seeds_negative']}")
        print(f"    items succès {c['item_bootstrap_success']} ; items faux-sûrs {c['item_bootstrap_false_confident']}")
    raws = res["raws"]
    print(f"\nDécomposition des abstentions (grounded_full, toutes graines) : {abstention_breakdown(raws)}")
    print(f"Auto-correction : {correction_report(raws)}")
    print(f"LearningNeed les plus fréquents : {learning_needs(raws)}")
    print(f"Longueur (clauses) : {long_sentence_report(raws)}")
    print(f"Domaines : {domain_report(raws)}")
    print(f"Transfert inter-domaines : {domain_transfer()}")
    print(f"\nDistillation structurelle (rôles) : {structural_distillation()}")
    print("\nBudget de vérification (graine 0) :")
    for k, v in budget_curve().items(): print(f"  {k:<22} {v}")
    sc = lexical_scale()
    print("\nÉchelle lexicale (acquisition trans-situationnelle, 3 candidats par exposition) :")
    for n, v in sc["cross_situational"].items(): print(f"  {n:>5} formes : {v['by_exposures']}")
    for n, v in sc["neural_vs_structural"].items():
        print(f"  {n:>5} (neuronal) : paramètres {v['neural_parameters']}, formes inédites {v['neural_heldout_form_accuracy']}, mémoire structurelle {v['structural_memory_entries']}")
    print(f"\nGéométrie (mise de côté, n'influence aucune décision) : {geometry_holdout(seeds)}")


if __name__ == "__main__":
    main()

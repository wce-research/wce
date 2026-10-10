"""Independent benchmark: hallucinations per category, ablations, conditional geometry, costs."""
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
    print(f"Items per split (per world, no tuning on the test): {n} ; total {sum(n.values())}")
    print("\nAblations (held-out test):")
    print(f"  {'system':<24} success false-conf  abstention  verif./query params")
    res = ablations(setup)
    for name, r in res.items():
        print(f"  {name:<24} {r['success']:.3f}  {r['false_confident_rate']:.3f}      {r['abstention_rate']:.3f}       "
              f"{r['verification_cost']:<11} {r['parameters']}")
    print("\nTaxonomy of false and asserted answers:")
    for name in ("A_neural_only", "C_wce_only", "E_neural_wce_evidence", "G_full"):
        print(f"  {name:<24} {res[name]['taxonomy']}")
    print(f"\nPer category (G_full): {res['G_full']['by_category']}")
    print(f"G_full abstentions: correct {res['G_full']['correct_abstentions']}, unnecessary {res['G_full']['unnecessary_abstentions']}")
    print(f"\nSelf-correction: {correction_benchmark(setup)}")
    print("\nVerification costs:")
    for k, v in verification_costs(setup).items(): print(f"  {k:<22} {v}")
    g = geometry_word_test(setup)
    print(f"\nGeometric monitor (neural lexical errors, out-of-lexicon words of the test): n={g['n']} errors={g['n_incorrect']}")
    print(f"  AUROC {g['auroc']} | AUPRC {g['auprc']} | geometry ECE {g['calibration_ece_geometry']}")
    print(f"  natural errors only: {g['natural_only']}")
    for k, v in g["by_category"].items(): print(f"    {k:<38} {v}")
    c = conditional_geometry(setup)
    print(f"\nConditional value of geometry (candidate error per query, test n={c['n_test']}, error rate {c['test_error_rate']}):")
    for k, v in c["feature_sets"].items(): print(f"  {k:<32} {v}")
    print(f"  Δ AUROC (controls + geometry − controls): {c['delta_auroc_geometry_after_controls']}")
    print(f"  standardized coefficients: {c['standardized_coefficients']}")
    print(f"  subset with neural words: {c['subset_with_neural_words']}")
    print(f"\nDistillation: {distillation_report(setup)}")
    print(f"\nGrammar curve (examples -> accuracy): {grammar_learning_curve()}")
    print(f"\nLexical growth: ")
    for k, v in lexicon_growth().items(): print(f"  {k:>5} forms: {v}")
    print(f"\nCharacters vs subwords vs words: {representation_comparison()}")
    agent = make_agent("G_full", setup)
    ic = induced_classes(agent)
    print(f"\nInduced lexical classes (purity {ic['purity_vs_meaning_class']}): {ic['clusters']}")
    print("\nScale of the neural core (independent benchmark):")
    for k, v in scaling(setup).items(): print(f"  {k:<7} {v}")


if __name__ == "__main__":
    main()

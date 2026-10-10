"""Coverage growth: examples -> parsing -> lexicon -> abstention -> success -> false confident."""
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
    print("\nExamples (captions) -> coverage (seed 0):")
    print("  captions  parse    lexicon  abstention  unneeded  success false-conf roles")
    for n, r in coverage_growth().items():
        print(f"  {n:>8}  {r['parse_coverage']:.3f}    {r['lexical_coverage']:.3f}    {r['abstention']:.3f}       "
              f"{r['unnecessary_abstention']:.3f}     {r['success']:.3f}   {r['false_confident']:.3f}      {r['role_accuracy']:.3f}")
    res = multi_seed(seeds)
    print(f"\nMulti-seed ({len(seeds)} seeds: worlds, language, noise, captions, split, neural init.; {res['seconds']} s)")
    print("mean [CI95]")
    for name, m in res["summary"].items():
        print(f"\n  {name}")
        for k in ("success", "false_confident", "abstention", "unnecessary_abstention", "parse_coverage", "parse_coverage_resolvable",
                  "role_accuracy", "lexical_coverage", "verification_cost", "routing_accuracy", "polysemy_accuracy"):
            print(f"    {k:<28} {fmt(m.get(k))}")
    print("\nPaired comparisons (per-seed difference a−b, t CI95; then bootstrap over items):")
    for label, c in res["comparisons"].items():
        print(f"\n  {label}")
        for k in ("success", "false_confident", "unnecessary_abstention", "parse_coverage", "role_accuracy", "lexical_coverage", "verification_cost"):
            v = c[k]
            if v: print(f"    {k:<24} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  seeds +{v['seeds_positive']} / −{v['seeds_negative']}")
        print(f"    items success {c['item_bootstrap_success']}; items false-confident {c['item_bootstrap_false_confident']}")
    raws = res["raws"]
    print(f"\nAbstention breakdown (grounded_full, all seeds): {abstention_breakdown(raws)}")
    print(f"Self-correction: {correction_report(raws)}")
    print(f"Most frequent LearningNeed: {learning_needs(raws)}")
    print(f"Length (clauses): {long_sentence_report(raws)}")
    print(f"Domains: {domain_report(raws)}")
    print(f"Cross-domain transfer: {domain_transfer()}")
    print(f"\nStructural distillation (roles): {structural_distillation()}")
    print("\nVerification budget (seed 0):")
    for k, v in budget_curve().items(): print(f"  {k:<22} {v}")
    sc = lexical_scale()
    print("\nLexical scale (cross-situational acquisition, 3 candidates per exposure):")
    for n, v in sc["cross_situational"].items(): print(f"  {n:>5} forms: {v['by_exposures']}")
    for n, v in sc["neural_vs_structural"].items():
        print(f"  {n:>5} (neural): parameters {v['neural_parameters']}, unseen forms {v['neural_heldout_form_accuracy']}, structural memory {v['structural_memory_entries']}")
    print(f"\nGeometry (held out, influences no decision): {geometry_holdout(seeds)}")


if __name__ == "__main__":
    main()

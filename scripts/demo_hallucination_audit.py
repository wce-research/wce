"""Audit: geometric observables of the trajectories, correct vs incorrect outputs."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.language_core import SYSTEMS, build_core, make_scene, run_stream
from core.trajectory import OBSERVABLES, calibrate, evaluation_words, hallucination_audit


def fmt(z): return ", ".join(f"{k}={z[k]:.2f}" for k in OBSERVABLES + ("confidence",))


def main():
    print("=== WCE HALLUCINATION AUDIT ===")
    print("Observables = PROXIES (random projections + sliced-W2 over noisy passes); no exact W2.")
    lc = build_core(**SYSTEMS["6_full_self_correction"]); run_stream(lc)
    exclude = evaluation_words()
    report = hallucination_audit(lc, exclude)
    monitor, estimator = calibrate(lc, exclude)
    lc.attach_monitor(monitor, estimator)
    cases = report["cases"]
    pick = lambda cat, ok: next((c for c in cases if c.category == cat and c.correct == ok), None)
    for title, case in (("Correct output (known form)", pick("known", True)),
                        ("Correct output (unseen form)", pick("heldout_forms", True)),
                        ("Incorrect output (unseen form)", pick("heldout_forms", False)),
                        ("Incorrect output (corrupted module)", pick("corrupted_module", False)),
                        ("Unknown word / trap", pick("unknown_or_trap", False) or pick("unknown_or_trap", True)),
                        ("Ambiguous output", pick("ambiguous", True) or pick("ambiguous", False)),
                        ("Forced wrong routing", pick("wrong_routing", False))):
        if case is None: continue
        print(f"\n{title}: « {case.word} » -> {case.predicted} (truth {case.truth})")
        print(f"  Z = {fmt(case.z)}")
        print(f"  Geometric risk: {estimator.score(case.z):.2f} ({estimator.level(case.z)})")
    print("\nFull policy on sentences:")
    for text in ("le cube tombe", "le bloc montait", "le bloc zorphe", "tourne abcde", "le bloc tombe du support",
                 "le cube montagne", "renverse abcde", "la table est en bois"):
        r = lc.process(text, make_scene())
        risk = r.risk
        geo = f"{risk.geometric_level} ({risk.geometric_risk:.2f})" if risk and risk.geometric_risk is not None else "n/a"
        support = f"{risk.structural_support:.2f}" if risk else "n/a"
        print(f"  « {text} »: structural support {support} | geometric risk {geo} | status {r.status} -> Final policy: {r.decision}")
    print(f"\nTest set (disjoint from calibration): {report['n_test']} cases, {report['n_incorrect']} neural errors")
    print("AUROC for detecting a neural error:")
    for k, v in report["auroc"].items(): print(f"  {k:<26} {v}")
    nat = report["auroc_natural_errors_only"]
    print(f"AUROC on NATURAL errors only (without corrupted module): {nat}")
    print(f"GeometricRiskEstimator coefficients: {report['coefficients']}")
    print(f"Forced wrong routing: mean risk {report['wrong_routing_risk_mean']} (identical input trajectory -> invisible)")
    print("\nMeans per category:")
    for k, v in report["by_category"].items(): print(f"  {k:<30} {v}")
    print("\nCautious conclusion: correlation measured on a toy corpus, not a causal proof;"
          " structural contradictions and wrong routing remain invisible to the input geometry.")


if __name__ == "__main__":
    main()

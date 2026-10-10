"""Demo: third-party discourse bench (never seen by the resolver) — hand-written scenarios + second generator."""
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
    print(f"Third-party bench: {len(bench)} scenarios, {nombre_questions(bench)} questions "
          f"({len(scenarios_main())} hand-written, the rest by a second generator with different conventions).")
    print("The bench module imports no WCE component (AST test).\n")
    print("Example scenario (third-party conventions):")
    s = scenarios_main()[8]
    print("  objects (hidden truth):", s["objets"])
    for t in s["tours"]: print("  ", t)
    rows, _ = run_scenario(L, s)
    for r in rows: print(f"   ? {r['text']}  expected={r['expected']}  got={r['got']}  ({r['status']}, {r['answer'].phrase})")

    cal = calibrate_on(L)                          # calibration partition: another seed of the second generator
    print(f"\nCalibration (outside the test): {cal['n_answer_points']} answers, question threshold {cal['thresholds']}")
    print("  raw confidence vs accuracy:", cal["raw_answer_calibration"]["bins"])
    print("\nSystems on the third-party bench:")
    for name in SYSTEMS:
        m, rows = run(L, bench, name, cal)
        print(f"  {name:22s}", {k: m[k] for k in KEYS})
    m, rows = run(L, bench, "unified_calibrated", cal)
    main_rows = [r for r in rows if r["origin"] == "main"]
    print("\nHand-written scenarios only:", {k: v for k, v in metrics(main_rows).items() if k in KEYS})
    print("\nBy adversarial category (hand-written):")
    for k, v in by(main_rows, "category").items(): print(f"  {k:20s} success {v['success']:.2f} false confident {v['false_confident']:.2f} (n={v['n']})")
    print("\nBy variation (second generator):")
    for k, v in by([r for r in rows if r["origin"] != "main"], "category").items():
        print(f"  {k:26s} success {v['success']:.2f} false confident {v['false_confident']:.2f} (n={v['n']})")
    print("\nCalibration curves:")
    print("  identity:", m["identity_calibration"]["bins"], "ECE", m["identity_calibration"]["ece"])
    print("  belief:", m["belief_calibration"]["bins"], "ECE", m["belief_calibration"]["ece"])
    print("  final answer:", answer_calibration(rows)["bins"])
    blind = scenarios_aveugles()
    print(f"\nBLIND set written after tuning ({len(blind)} scenarios, {nombre_questions(blind)} questions, evaluated without any adjustment):")
    for name in SYSTEMS:
        mb, _ = run(L, blind, name, cal)
        print(f"  {name:22s}", {k: mb[k] for k in ("success", "false_confident", "abstention", "exact_class", "coverage")})
    print("\nActive observation (ASK questions of the bench):", active_observation(L))
    print("Active observation on commands (required identity 0.99):", active_commands(L))
    print("Self-correction by revision of the belief graph:", self_correction(L))
    print("Distillation (student proposes, WCE verifies):", distill(L, calibration_bench(60, 23), bench, run_scenario))
    print("Memory (50/100/500 sentences):", memory_scaling(L, (50, 100, 500)))
    from core.third_party_hard import dev_generated, dev_set, n_questions as nq
    dev = dev_set() + dev_generated()
    md, _ = run(L, dev, "unified_calibrated", cal)
    print(f"\nHARD third-party bench — development ({nq(dev)} questions):", {k: md[k] for k in ("success", "false_confident", "abstention")})
    import json
    sealed = "results/sealed_blind.json"
    if os.path.exists(sealed):
        r = json.load(open(sealed))
        print(f"SEALED set ({r['questions']} questions, evaluated only once; result recorded, never adjusted):")
        for name, v in r["systems"].items():
            print(f"  {name:22s}", {k: v["metrics"][k] for k in ("success", "false_confident", "abstention")})
    print("\nCore size: unified state = 0 neural parameters (explicit formulas); distilled identity student = 6 parameters.")


if __name__ == "__main__":
    main()

"""Demo: unknown word -> several senses -> observations -> posterior -> calibration -> promotion / abstention."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.discourse_benchmark import observation_pool  # noqa: E402
from core.epistemic_benchmark import CAL_SEEDS, lexical_examples, train_lexical_posterior  # noqa: E402
from core.grounded import LexicalMemory  # noqa: E402
from core.independent_world_ext import GOLD_MEANING  # noqa: E402
from core.lexical_calibration import (  # noqa: E402
    FEATURES, NULL, CalibratedLexicon, calibration_points, entropy, precision_at,
)
from core.lexical_experiments import (  # noqa: E402
    active_lexical, composition_eval, morphology_eval, polysemy_eval, semantic_classes, unknown_word_policy_eval,
)
from core.unified import calibration_report  # noqa: E402


def main():
    post = train_lexical_posterior()
    print(f"Lexical posterior fitted on the CALIBRATION seeds {CAL_SEEDS[0]}–{CAL_SEEDS[-1]} (never the test);"
          f" {post.parameter_count} parameters:")
    print("  ", {f: round(float(t), 2) for f, t in zip(FEATURES, post.theta.detach())}, "NULL bias", round(float(post.null_bias.detach()), 2))

    pool, agent = observation_pool(0)
    word = "rayon"
    items = [p for p in pool if p["word"] == word][:8]
    cal = CalibratedLexicon(post)
    K = len({m for m in agent.lex.lexicon().values() if m})
    saved = agent.lexmem; agent.lexmem = LexicalMemory()
    print(f"\nUnknown word « {word} » (correct sense, evaluation only: {GOLD_MEANING[word]}) — observations one by one:")
    try:
        for i, it in enumerate(items, 1):
            agent.lexmem.observe(word, it["verdicts"], it["source"], it["weight"], f"w{it['world']}:{it['text']}")
            d = cal.dist(agent, word)
            top = sorted(((c, p) for c, p in d.items()), key=lambda t: -t[1])[:3]
            st, c, p = cal.status(agent, word)
            print(f"  {i}. « {it['text']} » ({it['source']}) -> {[(c_, round(p_, 3)) for c_, p_ in top]}"
                  f" H={entropy(d, K):.2f} -> {st}{' : ' + c if c and st != 'UNKNOWN' else ''}")
        print(f"  lifecycle history: {cal.history[word]}")
    finally:
        agent.lexmem = saved

    print("\nCalibration on the TEST seeds 0–7 (native confidence vs calibrated posterior):")
    native, calib = [], []
    for s in range(8):
        ex, conf = lexical_examples(s)
        native += calibration_points(ex, None, conf); calib += calibration_points(ex, post)
    for name, pts in (("native confidence", native), ("calibrated posterior", calib)):
        r = calibration_report([c for c, _ in pts], [y for _, y in pts])
        print(f"  {name:18s} n={r['n']} ECE={r['ece']} Brier={r['brier']} precision@threshold={precision_at(pts)}")
        print(f"      diagram: {[(b['bin'], b['n'], b['accuracy']) for b in r['bins']]}")
    snap = [e for s in range(8) for e in lexical_examples(s, snapshots=True)[0] if e.gold]
    pts = calibration_points(snap, post); r = calibration_report([c for c, _ in pts], [y for _, y in pts])
    print(f"  snapshots (little evidence): n={r['n']} ECE={r['ece']} Brier={r['brier']} precision@threshold={precision_at(pts)}")

    print("\nPolysemy « range » (P(sense | syntax, domain, entity type)):", {k: v for k, v in polysemy_eval(range(4)).items() if k != "calibration"},
          "ECE", polysemy_eval(range(4))["calibration"]["ece"])
    print("Active disambiguation (outcome predicted without looking at the observation) vs passive:", active_lexical(post, budget=12)["summary"])
    print("Unknown-word policy (provenance):", unknown_word_policy_eval(post, range(2)))
    print("Morphological uncertainty (traps included):", {k: v for k, v in morphology_eval(range(2)).items() if k != "calibration"})
    print("Emergent functional classes (similarity by behaviour):", semantic_classes())
    print("Composition:", {k: (v["meaning"], v["correct"]) for k, v in composition_eval().items()})


if __name__ == "__main__":
    main()

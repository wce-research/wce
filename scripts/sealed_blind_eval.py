"""SINGLE evaluation of the sealed set of the hard third-party bench -> results/sealed_blind.json (do not rerun for tuning)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.discourse_benchmark import lexicon_agent  # noqa: E402
from core.third_party_eval import SYSTEMS, by, calibrate_on, metrics, run  # noqa: E402
from core.third_party_hard import SEALED_SHA256, n_questions, sealed_hash, sealed_set  # noqa: E402

if __name__ == "__main__":
    assert sealed_hash() == SEALED_SHA256, "sealed set modified"
    path = "results/sealed_blind.json"
    if os.path.exists(path) and "--force" not in sys.argv:
        print(open(path).read()); sys.exit(0)
    L = lexicon_agent(0); cal = calibrate_on(L)
    bench = sealed_set()
    out = {"sha256": SEALED_SHA256, "questions": n_questions(bench), "systems": {}}
    for name in SYSTEMS:
        m, rows = run(L, bench, name, cal)
        out["systems"][name] = {"metrics": metrics(rows), "by_category": {k: {"success": v["success"], "false_confident": v["false_confident"], "n": v["n"]}
                                                                            for k, v in by(rows, "category").items()},
                                "errors": [(r["scenario"], r["text"], r["expected"], r["got"]) for r in rows if not r["correct"]]}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))

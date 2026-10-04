"""Rapports du sous-jalon Primitive Discovery (dev uniquement ; jamais le lot scellé).

  install   : installe les poids du proposeur neuronal entraîné sur AWS → results/deep_configs/
  aggregate : agrège les runs AWS `deepprim` → results/deep_primitive_summary.json
  report    : imprime les tableaux
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]; return round(sum(xs) / len(xs), 4) if xs else None


def install():
    os.makedirs(os.path.join(OUT, "deep_configs"), exist_ok=True); info = {}
    for f in sorted(glob.glob(os.path.join(OUT, "aws", "wce-deepneural-*", "artifacts", "*", "deepneural_*.json"))):
        r = json.load(open(f))
        if r["status"] != "ok": continue
        row = r["result"][0]; meta = {k: v for k, v in row.items() if k != "state"}; json.dump({"state": row["state"], "info": meta}, open(os.path.join(OUT, "deep_configs", f"neural_proposer_{row['variant']}.json"), "w")); info[row["variant"]] = meta
    json.dump(info, open(os.path.join(OUT, "deep_neural_training.json"), "w"), indent=1)
    for v, m in info.items(): print(v, {k: m[k] for k in ("n_params", "flops_per_forward", "memory_bytes", "scenes", "epochs", "seconds", "data_seconds")}, "perte", m["loss"][0], "→", m["loss"][-1])
    return info


def load_rows():
    rows = []; bad = 0
    for f in glob.glob(os.path.join(OUT, "aws", "wce-deepprim-*", "artifacts", "*", "deepprim_*.json")):
        r = json.load(open(f))
        if r["status"] == "ok": rows += r["result"]
        else: bad += 1
    return rows, bad


def _corr(x, y):
    n = len(x)
    if n < 3: return None
    mx, my = sum(x) / n, sum(y) / n; sx = sum((a - mx) ** 2 for a in x) ** 0.5; sy = sum((b - my) ** 2 for b in y) ** 0.5
    return round(sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy), 3) if sx > 0 and sy > 0 else None


def aggregate():
    rows, bad = load_rows(); rows = [r for r in rows if not r.get("unavailable")]; S = {"n_rows": len(rows), "units_in_error": bad}
    M = ("exact", "functional", "equivalence", "component_recall", "precision", "param_error", "equivalence_identifiable", "equivalence_partial", "equivalence_not_identifiable", "classes", "proposals", "validated", "fit_equivalents", "seconds",
         "first_validated_at", "residual_left", "description_bits", "n_true")
    at = [r for r in rows if r["experiment"] == "atomic"]; S["atomic"] = {}; S["atomic_by_category"] = {}
    for m in sorted({r["method"] for r in at}):
        x = [r for r in at if r["method"] == m]; S["atomic"][m] = {"n": len(x), **{k: _mean(r.get(k) for r in x) for k in M}, "compactness": _mean(r["classes"] / max(1, r["n_true"]) for r in x),
                                                              "n_identifiable": sum(r.get("n_identifiable", 0) for r in x), "n_partial": sum(r.get("n_partial", 0) for r in x), "n_not_identifiable": sum(r.get("n_not_identifiable", 0) for r in x)}
        for c in sorted({r["category"] for r in x}): S["atomic_by_category"][f"{c}/{m}"] = {k: _mean(r.get(k) for r in x if r["category"] == c) for k in ("exact", "equivalence", "component_recall", "precision")}
    by_type = defaultdict(list)
    for r in at:
        if r["method"] in ("hybrid", "neural_hybrid"):
            for a in r["atoms"]: by_type[(r["method"], a["type"])].append(a)
    S["by_type"] = {f"{m}/{t}": {"n": len(v), "exact": _mean(a["exact"] for a in v), "equivalence": _mean(a["equivalence"] for a in v), "component_recall": _mean(a["component_recall"] for a in v)} for (m, t), v in sorted(by_type.items())}
    ood = [r for r in rows if r["experiment"] == "ood"]; S["ood"] = {f"{r_sh}/{m}": {k: _mean(r.get(k) for r in ood if r["shift"] == r_sh and r["method"] == m) for k in ("exact", "equivalence", "component_recall", "precision")} for r_sh in sorted({r["shift"] for r in ood}) for m in sorted({r["method"] for r in ood})}
    S["ood_reference_two"] = {m: {k: _mean(r.get(k) for r in at if r["category"] == "two" and r["method"] == m) for k in ("exact", "equivalence", "component_recall", "precision")} for m in ("hybrid", "neural", "neural_hybrid")}
    ac = [r for r in rows if r["experiment"] == "active"]; S["active"] = {f"{c}/{m}": {k: _mean(r.get(k) for r in ac if r["category"] == c and r["method"] == m) for k in ("exact", "equivalence", "component_recall", "precision", "param_error")} for c in ("two", "multi") for m in ("active_none", "active_random", "active_primitive")}
    ch = [r for r in rows if r["experiment"] == "chain"]; S["chain"] = {}; S["chain_transitions"] = {}; S["chain_correlations"] = {}
    C = ("primitive_equivalence", "primitive_exact", "component_recall", "primitive_precision", "equivalence_identifiable", "candidate_form", "candidate_functional", "final_functional", "final_form", "final_full", "discovered_true_composition", "fit_equivalents", "seconds")
    for m in sorted({r["method"] for r in ch}):
        for reg in ("standard", "rich"):
            for d in sorted({r["depth"] for r in ch}):
                x = [r for r in ch if r["method"] == m and r["regime"] == reg and r["depth"] == d]
                if x: S["chain"][f"{reg}/d{d}/{m}"] = {"n": len(x), **{k: _mean(r.get(k) for r in x) for k in C}}
        x = [r for r in ch if r["method"] == m]
        if m != "oracle_atoms" and x:
            hi = [r for r in x if (r.get("primitive_equivalence") or 0) >= 0.999]; lo = [r for r in x if (r.get("primitive_equivalence") or 0) < 0.999]
            S["chain_transitions"][m] = {"n": len(x), "all_primitives_found": len(hi), "candidate_functional_given_all": _mean(r["candidate_functional"] for r in hi), "candidate_functional_given_not_all": _mean(r["candidate_functional"] for r in lo),
                                         "final_given_candidate": _mean(r["final_functional"] for r in x if r["candidate_functional"]), "final_given_no_candidate": _mean(r["final_functional"] for r in x if not r["candidate_functional"])}
            S["chain_correlations"][m] = {"primitive_equivalence~candidate_functional": _corr([float(r.get("primitive_equivalence") or 0) for r in x], [float(r["candidate_functional"]) for r in x]),
                                          "primitive_equivalence~final_full": _corr([float(r.get("primitive_equivalence") or 0) for r in x], [float(r["final_full"]) for r in x]),
                                          "component_recall~final_full": _corr([float(r.get("component_recall") or 0) for r in x], [float(r["final_full"]) for r in x]),
                                          "candidate_functional~final_functional": _corr([float(r["candidate_functional"]) for r in x], [float(r["final_functional"]) for r in x])}
    # bibliothèque de primitives (familles, naissances, retraits) à partir des primitives VALIDÉES des scènes atomiques
    from core.boundary_program import BNode
    from core.deep_primitives import PrimitiveLibrary
    lib = PrimitiveLibrary()
    for i, r in enumerate(r for r in at if r["method"] == "hybrid"): lib.observe(f"s{i}", [BNode.from_dict(f["program"]) for f in r["found"] if f["z"] >= 5.0])
    lib.update(); S["library"] = {"usage": lib.usage, "priors": lib.priors(), "families": [f.render() + f" ×{f.support}" for f in lib.families], "births": [vars(b) for b in lib.births()], "retire_candidates": lib.retire()}
    json.dump(S, open(os.path.join(OUT, "deep_primitive_summary.json"), "w"), indent=1); print("lignes", len(rows), "unités en erreur", bad); return S


def report():
    S = json.load(open(os.path.join(OUT, "deep_primitive_summary.json")))
    print("\nBANC ATOMIQUE (toutes catégories)"); print("méthode".ljust(16) + " ".join(k[:10].rjust(11) for k in ("exact", "functional", "equivalence", "component_recall", "precision", "param_error", "classes", "proposals", "fit_equivalents", "seconds", "compactness")))
    for m, v in S["atomic"].items(): print(m.ljust(16) + " ".join((f"{v[k]:.2f}" if v[k] is not None else "—").rjust(11) for k in ("exact", "functional", "equivalence", "component_recall", "precision", "param_error", "classes", "proposals", "fit_equivalents", "seconds", "compactness")))
    print("\néquivalence par classe d'identifiabilité (identifiable / partiel / non) et effectifs")
    for m, v in S["atomic"].items(): print(m.ljust(16), v["equivalence_identifiable"], v["equivalence_partial"], v["equivalence_not_identifiable"], "|", v["n_identifiable"], v["n_partial"], v["n_not_identifiable"])
    print("\npar catégorie (équivalence / composantes / précision)")
    for k, v in S["atomic_by_category"].items():
        if k.split("/")[1] in ("gradient", "geometric", "hybrid", "neural", "neural_hybrid"): print(k.ljust(28), v["equivalence"], v["component_recall"], v["precision"])
    print("\npar type d'atome"); [print(k.ljust(28), v) for k, v in S["by_type"].items()]
    print("\nhors distribution (scènes à deux atomes)"); print("référence nominale", S["ood_reference_two"]); [print(k.ljust(28), v) for k, v in S["ood"].items()]
    print("\nmesures actives"); [print(k.ljust(28), v) for k, v in S["active"].items()]
    print("\nCHAÎNE : primitives → candidats → récupération")
    for k, v in S["chain"].items(): print(k.ljust(34), {x: v[x] for x in ("n", "primitive_equivalence", "component_recall", "candidate_form", "candidate_functional", "final_functional", "final_full", "discovered_true_composition")})
    print("\ntransitions"); [print(k.ljust(16), v) for k, v in S["chain_transitions"].items()]
    print("\ncorrélations"); [print(k.ljust(16), v) for k, v in S["chain_correlations"].items()]
    print("\nbibliothèque"); print(json.dumps(S["library"], indent=1, ensure_ascii=False)[:2500])


if __name__ == "__main__":
    {"install": install, "aggregate": aggregate, "report": report}[sys.argv[1]]()

"""Agrège les artefacts AWS (results/aws/*/artifacts) → results/robust_summary.json + verdicts H1–H10 pré-enregistrés.
Analyse locale légère uniquement (aucun calcul expérimental)."""
import glob
import json
import math
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


CAMPAIGN_KEYS = {**{k: range(16) for k in ("main", "noise", "hard", "repeat", "calibration", "family", "capacity", "identifiability")}, "caltrain": range(6), "sealed": range(24)}


def load(root=None):
    by = defaultdict(dict); runs = []
    for mp in sorted(glob.glob(os.path.join(root or os.path.join(ROOT, "results", "aws"), "*", "manifest.json"))):
        man = json.load(open(mp)); runs.append(man)
        for f in glob.glob(os.path.join(os.path.dirname(mp), "artifacts", "**", "*.json"), recursive=True):
            if os.path.basename(f).startswith("_"): continue
            r = json.load(open(f))
            if r.get("status") == "ok" and int(r["key"]) in CAMPAIGN_KEYS.get(r["kind"], ()): by[r["kind"]][str(r["key"])] = r   # clés pré-enregistrées seulement (les runs de test d'infrastructure sont ignorés)
    return by, runs


def ci(xs):
    xs = [float(x) for x in xs if x is not None]; n = len(xs)
    if not n: return None
    m = sum(xs) / n
    if n < 2: return {"mean": round(m, 4), "ci95": [round(m, 4)] * 2, "n": n}
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)); h = (2.131 if n <= 16 else 2.04) * sd / math.sqrt(n)
    return {"mean": round(m, 4), "ci95": [round(m - h, 4), round(m + h, 4)], "n": n}


def spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        for k, i in enumerate(o): r[i] = k
        return r
    if len(a) < 3: return None
    ra, rb = rank(a), rank(b); ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / den if den else None


def _per_seed(rows_by_seed, pred, key):
    out = []
    for s, rows in rows_by_seed.items():
        v = [float(r[key]) for r in rows if pred(r) and r.get(key) is not None]
        out.append(sum(v) / len(v) if v else None)
    return out


def aggregate(by, runs):
    S = {"runs": [{k: m.get(k) for k in ("run_id", "backend", "instance_type", "status", "code_hash", "config_hash", "identity", "region", "kind")} | {"runtime_s": [j.get("runtime_s") for j in m["jobs"]]} for m in runs]}
    main = {s: r["result"] for s, r in by.get("main", {}).items()}; H = {}
    # Dans les lignes du banc principal, le champ "depth" est la TAILLE DU PROGRAMME RETENU (il écrase la profondeur du
    # monde à la construction de la ligne). La profondeur du monde est rétablie depuis l'ordre fixe des sous-unités.
    from core.robust_experiments import MAIN_SPECS, METHODS
    for rows in main.values():
        if len(rows) == len(MAIN_SPECS) * len(METHODS):
            for i, r in enumerate(rows): r["world_depth"] = MAIN_SPECS[i // len(METHODS)][1]
    if main:
        methods = sorted({r["method"] for rows in main.values() for r in rows}); rep = lambda r: r["representable"]
        S["main"] = {m: {k: ci(_per_seed(main, lambda r, m=m: r["method"] == m and rep(r), k)) for k in ("recovery", "structure", "extrapolation", "in_region", "queries", "evaluations", "seconds", "depth")} for m in methods}
        S["main_by_depth"] = {m: {str(d): ci(_per_seed(main, lambda r, m=m, d=d: r["method"] == m and r.get("world_depth") == d, "recovery")) for d in (1, 2, 3, 4)} for m in methods}
        S["unrepresentable"] = {m: dict(Counter(r.get("status") for rows in main.values() for r in rows if r["method"] == m and not rep(r))) for m in methods}
        S["unknown_rate"] = {m: ci(_per_seed(main, lambda r, m=m: r["method"] == m and r.get("answer_unknown") is not None, "answer_unknown")) for m in methods}
        def diff(a, b, k, pred=rep):
            xa, xb = _per_seed(main, lambda r: r["method"] == a and pred(r), k), _per_seed(main, lambda r: r["method"] == b and pred(r), k)
            return ci([x - y for x, y in zip(xa, xb) if x is not None and y is not None])
        d1 = diff("robust_posterior", "robust_fit_only", "recovery"); S["diff_robust_vs_naive_passive"] = d1
        # H1 : vraisemblance robuste vs fit naïf (job hard + main) — voir aussi 'hard'
        d3r, d3e = diff("robust_active_efa", "robust_active_structural_ig", "recovery"), diff("robust_active_efa", "robust_active_structural_ig", "extrapolation")
        H["H3"] = {"recovery": d3r, "extrapolation": d3e, "verdict": "SOUTENUE" if ((d3r["mean"] >= 0.05 and d3r["ci95"][0] > 0 and d3e["mean"] > -0.02) or (d3e["mean"] >= 0.05 and d3e["ci95"][0] > 0 and d3r["mean"] > -0.02)) else "RÉFUTÉE" if d3r["mean"] <= 0 and d3e["mean"] <= 0 else "NON CONCLUANTE"}
        d7 = diff("full", "old_system", "recovery"); H["H7"] = {"value": d7, "verdict": "SOUTENUE" if d7["mean"] >= 0.10 and d7["ci95"][0] > 0 else "RÉFUTÉE" if d7["mean"] <= 0 else "NON CONCLUANTE"}
        d8 = diff("full", "old_system", "extrapolation"); H["H8"] = {"value": d8, "verdict": "SOUTENUE" if d8["mean"] >= -0.02 else "RÉFUTÉE" if d8["mean"] < -0.05 else "NON CONCLUANTE"}
        S["diff_efa_vs_random"] = {"recovery": diff("robust_active_efa", "random_queries", "recovery"), "extrapolation": diff("robust_active_efa", "random_queries", "extrapolation")}
        S["diff_full_vs_passive"] = diff("full", "passive", "recovery"); S["diff_oracle_vs_full"] = diff("oracle_policy", "full", "recovery")
        rhos = []
        for rows in main.values():
            pr = [p for r in rows for p in r.get("efa_pairs", [])]
            if len(pr) >= 5: rhos.append(spearman([p[0] for p in pr], [p[1] for p in pr]))
        c = ci([r for r in rhos if r is not None]); H["H10"] = {"spearman": c, "verdict": "NON CONCLUANTE" if c is None else "SOUTENUE" if c["mean"] >= 0.3 and c["ci95"][0] > 0 else "RÉFUTÉE" if c["mean"] <= 0 else "NON CONCLUANTE"}
    hard = [r for x in by.get("hard", {}).values() for r in x["result"]]
    if hard:
        S["hard"] = {f"{c}/{m}": {"excess_ast": ci([r["excess_depth"] for r in hard if r["case"] == c and r["method"] == m]), "recovery": ci([r["recovery"] for r in hard if r["case"] == c and r["method"] == m]),
                                  "ast": ci([r["ast"] for r in hard if r["case"] == c and r["method"] == m]), "mdl_naive": ci([r["mdl_naive"] for r in hard if r["case"] == c and r["method"] == m]),
                                  "mdl_measurement": ci([r["mdl_measurement"] for r in hard if r["case"] == c and r["method"] == m])}
                     for c in ("simple_noisy", "complex_clean") for m in ("naive_threshold", "robust_bayes")}
        en, er = S["hard"]["simple_noisy/naive_threshold"]["excess_ast"]["mean"], S["hard"]["simple_noisy/robust_bayes"]["excess_ast"]["mean"]
        H["H2"] = {"excess_naive": en, "excess_robust": er, "verdict": "SOUTENUE" if er <= 0.7 * en and en > 0 else "RÉFUTÉE" if er >= en else "NON CONCLUANTE"}
        seeds = sorted(by["hard"]); dn = []
        for s in seeds:
            rows = by["hard"][s]["result"]; f = lambda m: sum(r["recovery"] for r in rows if r["method"] == m and r["case"] == "simple_noisy")
            dn.append(f("robust_bayes") - f("naive_threshold"))
        c = ci(dn); H["H1"] = {"value": c, "verdict": "SOUTENUE" if c["mean"] >= 0.10 and c["ci95"][0] > 0 else "RÉFUTÉE" if c["mean"] <= 0 else "NON CONCLUANTE"}
    rp = by.get("repeat", {})
    if rp:
        dd = []
        for s, x in rp.items():
            rows = x["result"]; f = lambda a: sum(r["recovery"] for r in rows if r["repeat"] == a) / max(1, sum(1 for r in rows if r["repeat"] == a)); dd.append(f(True) - f(False))
        c = ci(dd); allr = [r for x in rp.values() for r in x["result"]]
        S["repeat"] = {"diff": c, "repeats_used": ci([r["repeats_used"] for r in allr if r["repeat"]]), "recovery_with": ci([r["recovery"] for r in allr if r["repeat"]]), "recovery_without": ci([r["recovery"] for r in allr if not r["repeat"]])}
        H["H4"] = {"value": c, "verdict": "SOUTENUE" if c["mean"] >= 0.05 and c["ci95"][0] > 0 else "RÉFUTÉE" if c["mean"] <= 0 else "NON CONCLUANTE"}
    train = [r for x in by.get("caltrain", {}).values() for r in x["result"]]       # TRAIN + VALIDATION uniquement (graines 700000+)
    cal = []
    if train and by.get("calibration"):
        from core.robust_experiments import fit_calibrators
        cal = [fit_calibrators(train, x["result"]) for x in by["calibration"].values() if x["result"]]   # calibrateurs ajustés sur train, évalués par graine de test
    if cal:
        S["calibration"] = {k: {"ece": ci([c[k]["ece"] for c in cal if "ece" in c.get(k, {})]), "brier": ci([c[k]["brier"] for c in cal if "brier" in c.get(k, {})])} for k in ("raw", "isotonic", "platt", "beta")}
        S["calibration"]["n_train"] = cal[0]["n_train"]; S["calibration"]["n_test_total"] = sum(c["n_test"] for c in cal)
        raw = S["calibration"]["raw"]["ece"]["mean"]; best = min(S["calibration"][k]["ece"]["mean"] for k in ("isotonic", "platt", "beta") if S["calibration"][k]["ece"])
        H["H5"] = {"raw_ece": raw, "best_calibrated_ece": best, "verdict": "SOUTENUE" if best <= 0.5 * raw else "RÉFUTÉE" if best >= raw else "NON CONCLUANTE"}
    fam = [r for x in by.get("family", {}).values() for r in x["result"]]
    if fam:
        def prec(k): p = [r for r in fam if r[k]]; return sum(r["same_structure"] for r in p) / max(1, len(p)), len(p)
        pa, na = prec("promote_active"); pc, nc = prec("promote_cluster")
        rec = lambda k: sum(r[k] for r in fam if r["same_structure"]) / max(1, sum(r["same_structure"] for r in fam))
        S["family"] = {"precision_active": round(pa, 3), "promoted_active": na, "precision_cluster": round(pc, 3), "promoted_cluster": nc, "recall_active": round(rec("promote_active"), 3), "recall_cluster": round(rec("promote_cluster"), 3),
                       "family_recovery": ci([r["family_recovery"] for r in fam if r["same_structure"]]), "search_recovery": ci([r["search_recovery"] for r in fam if r["same_structure"]]), "search_evals": ci([r["search_evals"] for r in fam])}
        H["H6"] = {"active": pa, "cluster": pc, "verdict": "SOUTENUE" if pa >= pc + 0.2 else "RÉFUTÉE" if pa <= pc else "NON CONCLUANTE"}
    cap = [r for x in by.get("capacity", {}).values() for r in x["result"]]
    if cap:
        S["capacity"] = {"params": ci([r["params"] for r in cap]), "ast": ci([r["ast"] for r in cap]), "raw_points": ci([r["raw_points"] for r in cap]), "violations": sum(r["violated"] for r in cap),
                         **{k: {"in": ci([r[k][0] for r in cap]), "out": ci([r[k][1] for r in cap]), "stored": ci([r[k][2] for r in cap])} for k in ("none", "raw_gated", "prototype")}}
        H["H9"] = {"violations": S["capacity"]["violations"], "verdict": "SOUTENUE" if S["capacity"]["violations"] == 0 else "RÉFUTÉE"}
    idn = [r for x in by.get("identifiability", {}).values() for r in x["result"]]
    if idn: S["identifiability"] = {"representable": dict(Counter(r["class"] for r in idn if r["representable"])), "unrepresentable": dict(Counter(r["class"] for r in idn if not r["representable"]))}
    nz = [r for x in by.get("noise", {}).values() for r in x["result"]]
    if nz:
        S["noise_sweep"] = {str(n): {m: {k: round(sum(float(r[k]) for r in nz if r["noise"] == n and r["method"] == m) / max(1, sum(1 for r in nz if r["noise"] == n and r["method"] == m)), 4)
                                         for k in ("recovery", "structure", "extrapolation", "queries", "evaluations", "seconds", "params", "ast")} for m in sorted({r["method"] for r in nz})} for n in sorted({r["noise"] for r in nz})}
        S["noise_by_kind"] = {k: {m: round(sum(r["recovery"] for r in nz if r["kind"] == k and r["method"] == m) / max(1, sum(1 for r in nz if r["kind"] == k and r["method"] == m)), 4) for m in sorted({r["method"] for r in nz})} for k in sorted({r["kind"] for r in nz})}
    sl = [r for x in by.get("sealed", {}).values() for r in x["result"]]
    if sl:
        S["sealed"] = {m: {"recovery": ci([r["recovery"] for r in sl if r["method"] == m and r["representable"]]), "structure": ci([r["structure"] for r in sl if r["method"] == m and r["representable"]]),
                           "extrapolation": ci([r["extrapolation"] for r in sl if r["method"] == m and r["representable"]]),
                           "unrepresentable_status": dict(Counter(r.get("status") for r in sl if r["method"] == m and not r["representable"]))} for m in sorted({r["method"] for r in sl})}
    S["hypotheses"] = dict(sorted(H.items(), key=lambda kv: int(kv[0][1:])))
    return S


if __name__ == "__main__":
    by, runs = load(); S = aggregate(by, runs)
    json.dump(S, open(os.path.join(ROOT, "results", "robust_summary.json"), "w"), indent=1, default=str, ensure_ascii=False)
    print(json.dumps({"kinds": {k: len(v) for k, v in by.items()}, "hypotheses": S["hypotheses"]}, indent=1, ensure_ascii=False, default=str))

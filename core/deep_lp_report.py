"""Rapports du sous-jalon Latent Primitive Recovery (dev uniquement ; jamais le lot scellé).

  tune      : seuil de complétude et gain minimal, sur les clés de MISE AU POINT (1090–1094) → results/deep_configs/lpr.json
  aggregate : agrège les runs AWS `deeplp` → results/deep_latent_primitive_summary.json
  report    : imprime les tableaux
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict

from core.deep_rp_report import _mean, paired_ci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
SUMMARY = os.path.join(OUT, "deep_latent_primitive_summary.json")


def load_rows():
    rows = []; bad = 0
    for f in glob.glob(os.path.join(OUT, "aws", "wce-deeplp-*", "artifacts", "*", "deeplp_*.json")):
        r = json.load(open(f))
        if r["status"] == "ok": rows += r["result"]
        else: bad += 1
    return rows, bad


def _phi(a, b):
    n = len(a)
    if n < 8: return None
    ma, mb = sum(a) / n, sum(b) / n; va, vb = ma * (1 - ma), mb * (1 - mb)
    return round((sum(x * y for x, y in zip(a, b)) / n - ma * mb) / (va * vb) ** 0.5, 4) if va > 0 and vb > 0 else None


def summarize(rows, bad=0):
    S = {"n_rows": len(rows), "units_in_error": bad}; ch = [r for r in rows if r["experiment"] == "chain"]; arms = sorted({r["arm"] for r in ch}); depths = sorted({r["depth"] for r in ch})
    groups = [("all", lambda r: True), ("d3-6", lambda r: r["depth"] >= 3), ("d4-6", lambda r: r["depth"] >= 4)] + [(f"d{d}", (lambda r, d=d: r["depth"] == d)) for d in depths]
    K = ("accord", "accord_out", "functional", "functional_097", "candidate_097", "candidate_099", "pool_recall_functional", "pool_recall_strict", "pool_recall_exact", "pool_all_functional", "pool_all_strict", "pool_all_exact", "pool_most_functional", "pool_most_strict", "pool_param_error",
         "kept_recall_functional", "kept_recall_strict", "kept_all_functional", "kept_all_strict", "kept_most_functional", "kept_param_error", "top3_recall_functional", "top3_all_functional", "top3_all_strict", "hypotheses", "kept", "false_kept", "sets", "considered", "iterations", "n_pairs", "fit_committed", "seconds", "literals")
    S["chain"] = {f"{g}/{a}": {"n": sum(f(r) and r["arm"] == a for r in ch), **{k: _mean(r.get(k) for r in ch if f(r) and r["arm"] == a) for k in K}} for g, f in groups for a in arms}
    by = {a: {r["task_id"]: r for r in ch if r["arm"] == a} for a in arms}; S["paired"] = {}; P = ("accord", "functional", "functional_097", "candidate_097", "pool_all_functional", "pool_all_strict", "pool_recall_functional", "kept_all_functional", "kept_all_strict", "kept_recall_functional", "pool_param_error", "seconds")
    for a, ref in [("C", "A"), ("C", "B"), ("C", "B5"), ("C", "A_full"), ("C", "B_full"), ("B", "A"), ("B5", "B"), ("B", "B_full"), ("A", "A_full")] + [(v, "C") for v in arms if v.startswith("C_")]:
        if a in by and ref in by:
            for g, f in groups[:3]:
                ts = sorted(t for t in set(by[a]) & set(by[ref]) if f(by[a][t])); S["paired"][f"{g}: {a} − {ref}"] = {k: paired_ci([by[ref][t].get(k) for t in ts], [by[a][t].get(k) for t in ts]) for k in P}
    # ---- loi d'échelle du rappel d'ensemble, modèle indépendant, corrélations d'erreurs
    S["set_scaling"] = {}; S["miss_by_type"] = {}; S["miss_by_visibility"] = {}; S["miss_correlation"] = {}; S["param_by_type"] = {}
    for a in ("A", "B", "B5", "C"):
        x = [r for r in ch if r["arm"] == a and r.get("atoms")]
        if not x: continue
        for d in depths:
            xd = [r for r in x if r["depth"] == d]; rec = _mean(at["pool"] for r in xd for at in r["atoms"]); obs = _mean(r["pool_all_functional"] for r in xd)
            S["set_scaling"][f"{a}/n{d}"] = {"n": len(xd), "atom_recall": rec, "all_present": obs, "independent_model": round(rec ** d, 4) if rec is not None else None, "at_most_one_missing": _mean(r["pool_most_functional"] for r in xd),
                                           "kept_atom_recall": _mean(at["kept"] for r in xd for at in r["atoms"]), "kept_all_present": _mean(r["kept_all_functional"] for r in xd), "mean_visibility": _mean(at["visible"] for r in xd for at in r["atoms"])}
        ty = defaultdict(list); vis = defaultdict(list); pe = defaultdict(lambda: defaultdict(list)); pa = []; pb = []; same = defaultdict(lambda: ([], []))
        for r in x:
            for at in r["atoms"]:
                ty[at["type"]].append(at["pool"]); vis[min(4, int(at["visible"] * 5))].append(at["pool"])
                for k, v in (at.get("errors") or {}).items(): pe[at["type"]][k].append(v)
            ats = r["atoms"]
            for i in range(len(ats)):
                for j in range(i + 1, len(ats)): pa.append(1 - ats[i]["pool"]); pb.append(1 - ats[j]["pool"]); kk = "same_type" if ats[i]["type"] == ats[j]["type"] else "different_type"; same[kk][0].append(1 - ats[i]["pool"]); same[kk][1].append(1 - ats[j]["pool"])
        S["miss_by_type"][a] = {t: {"n": len(v), "recall": _mean(v)} for t, v in sorted(ty.items())}; S["miss_by_visibility"][a] = {f"{b * 0.2:.1f}-{b * 0.2 + 0.2:.1f}": {"n": len(v), "recall": _mean(v)} for b, v in sorted(vis.items())}
        both = sum(u and v for u, v in zip(pa, pb)); S["miss_correlation"][a] = {"pairs": len(pa), "phi": _phi(pa, pb), "p_miss": _mean(pa + pb), "p_miss_given_other_missed": round(both / max(1, sum(pa)), 4) if pa else None, **{k: {"pairs": len(v[0]), "phi": _phi(v[0], v[1])} for k, v in same.items()}}
        S["param_by_type"][a] = {t: {k: _mean(v) for k, v in d.items()} for t, d in sorted(pe.items())}
    c = [r for r in ch if r["arm"] == "C"]; S["loop"] = {g: {"n": sum(f(r) for r in c), "iterations": _mean(r.get("iterations") for r in c if f(r)), "looped": _mean((r.get("iterations") or 1) > 1 for r in c if f(r)), "states": dict(Counter(r["state"] for r in c if f(r)))} for g, f in groups[:1] + groups[3:]}
    S["state_vs_set"] = {st: {"n": sum(r["state"] == st for r in c), "all_present": _mean(r["pool_all_functional"] for r in c if r["state"] == st), "functional": _mean(r["functional"] for r in c if r["state"] == st), "accord": _mean(r["accord"] for r in c if r["state"] == st)} for st in sorted({r["state"] for r in c})}
    S["downstream_given_set"] = {a: {str(flag): {"n": sum(bool(r.get("kept_all_strict")) == flag for r in ch if r["arm"] == a), "functional": _mean(r["functional"] for r in ch if r["arm"] == a and bool(r.get("kept_all_strict")) == flag), "accord": _mean(r["accord"] for r in ch if r["arm"] == a and bool(r.get("kept_all_strict")) == flag)} for flag in (True, False)} for a in ("A", "B", "C") if a in arms}
    # ---- sensibilité
    se = [r for r in rows if r["experiment"] == "sens"]; S["sensitivity"] = {}
    for mode, key in sorted({(r["mode"], r["key"]) for r in se}):
        for s in sorted({r["sigma"] for r in se}):
            x = [r for r in se if r["mode"] == mode and r["key"] == key and r["sigma"] == s and r["base_functional"]]; S["sensitivity"][f"{mode}:{key}/σ{s}"] = {"n": len(x), "functional": _mean(r["functional"] for r in x), "functional_097": _mean(r["functional_097"] for r in x), "accord": _mean(r["accord"] for r in x), "affected": _mean(r["n_affected"] for r in x)}
    # ---- faux positifs, défaut de l'ancienne réduction
    fp = [r for r in rows if r["experiment"] == "fp"]; S["false_positives"] = {f"{g}/{e}/{a}": {"n": sum(f(r) and r["arm"] == a and r["engine"] == e for r in fp), **{k: _mean(r.get(k) for r in fp if f(r) and r["arm"] == a and r["engine"] == e) for k in ("accord", "functional", "kept", "kept_recall", "all_kept", "false_kept", "conditional_only")}}
                                                        for g, f in groups[:1] + groups[2:3] for e in ("sets", "old_reduction") for a in sorted({r["arm"] for r in fp}, key=lambda s: int(s[4:])) if any(r["arm"] == a and r["engine"] == e for r in fp)}
    f0 = {e: {r["task_id"]: r for r in fp if r["arm"] == "plus0" and r["engine"] == e} for e in ("sets", "old_reduction")}; ts = sorted(set(f0["sets"]) & set(f0["old_reduction"])); S["reduction_fix"] = {k: paired_ci([f0["old_reduction"][t][k] for t in ts], [f0["sets"][t][k] for t in ts]) for k in ("kept_recall", "all_kept", "functional", "accord")} if ts else {}
    fb = {a: {r["task_id"]: r for r in fp if r["arm"] == a and r["engine"] == "sets"} for a in {r["arm"] for r in fp}}; S["false_positive_paired"] = {f"{a} − plus0": {k: paired_ci([fb["plus0"][t][k] for t in sorted(set(fb[a]) & set(fb["plus0"]))], [fb[a][t][k] for t in sorted(set(fb[a]) & set(fb["plus0"]))]) for k in ("functional", "accord")} for a in fb if a != "plus0" and "plus0" in fb}
    # ---- complétude, primitive manquante, hors classe
    co = [r for r in rows if r["experiment"] == "complete"]; S["completeness"] = {a: {"n": sum(r["arm"] == a for r in co), **{k: _mean(r.get(k) for r in co if r["arm"] == a) for k in ("excess", "gap", "flag_incomplete", "accord", "functional", "before_accord", "before_functional", "missing_found", "missing_kept", "iterations")}} for a in sorted({r["arm"] for r in co})}
    m1 = [r for r in co if r["arm"] == "missing1"]; S["missing_by_type"] = {t: {"n": sum(r["missing_type"] == t for r in m1), "found": _mean(r["missing_found"] for r in m1 if r["missing_type"] == t), "kept": _mean(r["missing_kept"] for r in m1 if r["missing_type"] == t), "functional": _mean(r["functional"] for r in m1 if r["missing_type"] == t)} for t in sorted({r["missing_type"] for r in m1})}
    S["missing_by_depth"] = {f"d{d}": {"n": sum(r["depth"] == d for r in m1), "found": _mean(r["missing_found"] for r in m1 if r["depth"] == d), "functional": _mean(r["functional"] for r in m1 if r["depth"] == d), "before_functional": _mean(r["before_functional"] for r in m1 if r["depth"] == d)} for d in depths if any(r["depth"] == d for r in m1)}
    oc = [r for r in rows if r["experiment"] == "ooc"]; S["ooc"] = {a: {"n": sum(r["arm"] == a for r in oc), "states": dict(Counter(r["state"] for r in oc if r["arm"] == a)), "excess": _mean(r["excess"] for r in oc if r["arm"] == a), "gap": _mean(r["gap"] for r in oc if r["arm"] == a)} for a in sorted({r["arm"] for r in oc})}
    return S


def aggregate():
    rows, bad = load_rows(); S = summarize(rows, bad); json.dump(S, open(SUMMARY, "w"), indent=1); print("lignes", len(rows), "unités en erreur", bad); return S


def show(S):
    ci = lambda v: {x: (y["diff"], y["ci95"]) if y else None for x, y in v.items()}; cut = lambda v, ks: {k: v.get(k) for k in ks if v.get(k) is not None}
    print("\nCHAÎNES"); [print(k.ljust(18), cut(v, ("n", "accord", "functional", "functional_097", "candidate_097", "pool_recall_functional", "pool_all_functional", "pool_all_strict", "pool_most_functional", "pool_param_error", "kept_recall_functional", "kept_all_functional", "kept_all_strict", "top3_all_functional", "hypotheses", "kept", "false_kept", "considered", "iterations", "seconds"))) for k, v in S["chain"].items()]
    print("\nappariées"); [print(k.ljust(26), ci({a: b for a, b in v.items() if a in ("accord", "functional", "candidate_097", "pool_all_functional", "kept_all_strict", "seconds")})) for k, v in S["paired"].items()]
    print("\nLOI D'ÉCHELLE DU RAPPEL D'ENSEMBLE"); [print(k.ljust(8), v) for k, v in S["set_scaling"].items()]; print("par type", S["miss_by_type"]); print("par visibilité", S["miss_by_visibility"]); print("corrélation des manques", S["miss_correlation"]); print("erreur de paramètre par type", S["param_by_type"])
    print("\nboucle", S["loop"]); print("état → ensemble", S["state_vs_set"]); print("aval sachant l'ensemble", S["downstream_given_set"])
    print("\nSENSIBILITÉ"); [print(k.ljust(26), v) for k, v in S["sensitivity"].items()]
    print("\nFAUX POSITIFS"); [print(k.ljust(30), v) for k, v in S["false_positives"].items()]; print("correction de la réduction", ci(S["reduction_fix"])); [print(k.ljust(18), ci(v)) for k, v in S["false_positive_paired"].items()]
    print("\nCOMPLÉTUDE"); [print(k.ljust(10), v) for k, v in S["completeness"].items()]; print(S["missing_by_type"]); print(S["missing_by_depth"]); print("hors classe", S["ooc"])


def report(): show(json.load(open(SUMMARY)))


def tune():
    """Sur les clés 1090–1094 (60 tâches) : gain minimal de la sélection d'ensembles et seuil de complétude (qui déclenche la
    recherche par le résidu). Critère fixé d'avance : accord final moyen de la chaîne C. Le vivier est calculé une fois."""
    import itertools
    from core.deep_lpr import CONFIG_FILE, DEFAULTS, pool, recover
    from core.deep_lpr_exp import DEPTHS, REGIMES, _acc, _regions, set_recovery, task_at
    T = [(task_at(k, d, r), d) for k in range(1090, 1095) for d in DEPTHS for r in REGIMES]; P = [pool(t.X, t.y, 5, 0) for t, _ in T]; reg = [_regions(t) for t, _ in T]; res = []
    for mg, thr in itertools.product((1.0, 2.0), (0.10, 0.0, -1.0)):
        cfg = {**DEFAULTS, "min_gain": mg, "incomplete": thr}; out = [recover(t.X, t.y, 0, cfg, p) for (t, _), p in zip(T, P)]; acc = [_acc(o.program, rg) for o, rg in zip(out, reg)]
        res.append({"min_gain": mg, "incomplete": thr, "accord": _mean(a["accord"] for a in acc), "functional_097": _mean(a["functional_097"] for a in acc), "accord_d4_6": _mean(a["accord"] for a, (_, d) in zip(acc, T) if d >= 4), "iterations": _mean(o.stats["iterations"] for o in out),
                    "all_present": _mean(set_recovery([h.program for h in o.hypotheses], t)["all_functional"] for o, (t, _) in zip(out, T)), "seconds": _mean(o.stats["seconds"] for o in out)}); print(res[-1], flush=True)
    best = max(res, key=lambda r: r["accord"]); cfg = {"min_gain": best["min_gain"], "incomplete": best["incomplete"]}
    json.dump({"config": cfg, "selected_on": f"{len(T)} tâches de mise au point (clés 1090–1094)", "criterion": "accord final moyen de la chaîne C", "grid": res}, open(CONFIG_FILE, "w"), indent=1, ensure_ascii=False); print("RETENU", cfg, best)


if __name__ == "__main__":
    {"aggregate": aggregate, "report": report, "tune": tune}[sys.argv[1]]()

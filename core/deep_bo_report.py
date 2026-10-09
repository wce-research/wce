"""Rapports du sous-jalon Boundary Ownership & Fragment Attribution (dev uniquement ; jamais le lot scellé).

  smoke / chain-smoke : démonstrations locales courtes (scènes et tâche de MISE AU POINT)
  tune                : coûts de l'énergie et température de la postérieure, sur scènes de mise au point
  aggregate           : agrège les runs AWS `deepbo` → results/deep_boundary_ownership_summary.json
  report              : imprime les tableaux
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

from core.deep_rp_report import _mean, paired_ci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
SUMMARY = os.path.join(OUT, "deep_boundary_ownership_summary.json")


def load_rows():
    rows = []; bad = 0
    for f in glob.glob(os.path.join(OUT, "aws", "wce-deepbo-*", "artifacts", "*", "deepbo_*.json")):
        r = json.load(open(f))
        if r["status"] == "ok": rows += r["result"]
        else: bad += 1
    return rows, bad


def calibration(cals):
    """Histogrammes [n, corrects, somme des confiances] cumulés → fiabilité par tranche et ECE."""
    tot = [[0, 0.0, 0.0] for _ in range(10)]
    for c in cals:
        if c:
            for b in range(10): tot[b] = [tot[b][0] + c[b][0], tot[b][1] + c[b][1], tot[b][2] + c[b][2]]
    n = sum(b[0] for b in tot)
    if not n: return {"n": 0, "ece": None, "bins": {}}
    return {"n": n, "ece": round(sum(abs(b[1] - b[2]) for b in tot) / n, 4), "accuracy": round(sum(b[1] for b in tot) / n, 4), "mean_confidence": round(sum(b[2] for b in tot) / n, 4),
            "bins": {f"{i / 10:.1f}": {"n": b[0], "accuracy": round(b[1] / b[0], 3), "confidence": round(b[2] / b[0], 3)} for i, b in enumerate(tot) if b[0]}}


def summarize(rows, bad=0):
    S = {"n_rows": len(rows), "units_in_error": bad}; key = lambda r: (r["scene"], r["mask"], r["noise"])
    at = [r for r in rows if r["experiment"] == "attr"]; meths = sorted({r["method"] for r in at}); masks = sorted({r["mask"] for r in at}); kinds = sorted({r["kind"] for r in at})
    K = ("precision", "recall", "f1", "purity", "pieces_per_boundary", "mean_length", "groups", "top1", "top3", "brier", "owners", "assignments_tested", "seconds")
    S["attribution"] = {m: {"n": sum(r["method"] == m for r in at), **{k: _mean(r.get(k) for r in at if r["method"] == m) for k in K}} for m in meths}
    S["attribution_by_mask"] = {f"{m}/{int(ms * 100)}": {k: _mean(r.get(k) for r in at if r["method"] == m and r["mask"] == ms) for k in ("precision", "recall", "f1", "purity", "pieces_per_boundary", "top1", "top3")} for m in meths for ms in masks}
    S["attribution_by_kind"] = {f"{m}/{kd}": {k: _mean(r.get(k) for r in at if r["method"] == m and r["kind"] == kd) for k in ("f1", "purity", "pieces_per_boundary", "top1", "top3")} for m in ("solver", "solver_no_chance", "consensus", "tangent") if m in meths for kd in kinds}
    ba = {m: {key(r): r for r in at if r["method"] == m} for m in meths}; S["attribution_paired"] = {}
    for a, b in (("tangent", "shared"), ("tangent", "solver"), ("consensus", "solver"), ("solver_no_chance", "solver"), ("tangent", "solver_no_chance"), ("solver_local", "solver"), ("solver_no_smooth", "solver"), ("solver", "solver_absent"), ("solver_absent_naive", "solver_absent"), ("solver", "composition")):
        if a in ba and b in ba:
            ks = sorted(set(ba[a]) & set(ba[b])); S["attribution_paired"][f"{b} − {a}"] = {k: paired_ci([ba[a][s].get(k) for s in ks], [ba[b][s].get(k) for s in ks]) for k in ("f1", "purity", "pieces_per_boundary", "top1", "top3")}
    sv = [r for r in at if r["method"] == "solver"]; S["calibration"] = calibration(r.get("cal") for r in sv); S["calibration_by_mask"] = {str(int(ms * 100)): {k: v for k, v in calibration(r.get("cal") for r in sv if r["mask"] == ms).items() if k != "bins"} for ms in masks}
    S["energy"] = {k: _mean(r["energy"][k] for r in sv) for k in ("fit", "noise", "smooth", "complexity", "absent", "chance", "total", "owners")} if sv else {}
    S["occlusion"] = {str(int(ms * 100)): {"n": sum(r["mask"] == ms for r in sv), "found": _mean(r["occlusion_found"] for r in sv if r["mask"] == ms), "correct": _mean(r["occlusion_correct"] for r in sv if r["mask"] == ms), "ambiguous_fragments": _mean(r["ambiguous_fragments"] for r in sv if r["mask"] == ms)} for ms in masks}
    orc = [r for r in rows if r["experiment"] == "oracle"]; O = ("TT", "TI", "ET", "EI", "NT", "NI", "OWN", "kept_fraction", "target_owned")
    S["oracle"] = {"all": {"n": len(orc), **{k: _mean(r[k] for r in orc) for k in O}}, **{f"mask{int(ms * 100)}": {k: _mean(r[k] for r in orc if r["mask"] == ms) for k in O} for ms in masks}, **{kd: {k: _mean(r[k] for r in orc if r["kind"] == kd) for k in O} for kd in kinds}}
    S["oracle_paired"] = {f"{b} − {a}": paired_ci([r[a] for r in orc], [r[b] for r in orc]) for a, b in (("NT", "ET"), ("NI", "EI"), ("ET", "TT"), ("EI", "TI"), ("TI", "TT"), ("EI", "ET"), ("NT", "TT"), ("NI", "TI"), ("NI", "OWN"))} if orc else {}
    vs = [r for r in rows if r["experiment"] == "vis"]; V = ("contrast", "visible_no_occluder", "visible_inferred", "visible_true_occluder", "pool_has_truth")
    S["visibility"] = {"all": {"n": len(vs), **{k: _mean(r[k] for r in vs) for k in V}}, **{f"mask{int(ms * 100)}": {k: _mean(r[k] for r in vs if r["mask"] == ms) for k in V} for ms in masks}}
    S["visibility_paired"] = {f"{b} − contrast": paired_ci([r["contrast"] for r in vs], [r[b] for r in vs]) for b in V[1:4]} if vs else {}
    sx = [r for r in rows if r["experiment"] == "sector"]
    S["sector"] = {str(int(ms * 100)): {"n": sum(r["mask"] == ms for r in sx), **{k: _mean(r[k] for r in sx if r["mask"] == ms) for k in ("ray_top1", "ray_top3", "rays_owned", "ray_points_min", "TT", "NT", "tt_center_error", "tt_direction_error", "tt_width_error")}, "tt_stages": dict(Counter(r["tt_stage"] for r in sx if r["mask"] == ms))} for ms in masks}
    ch = [r for r in rows if r["experiment"] == "chain"]; arms = sorted({r["arm"] for r in ch}); cb = {a: {r["task_id"]: r for r in ch if r["arm"] == a} for a in arms}; S["chain"] = {}; S["chain_paired"] = {}; S["chain_by_depth"] = {}; S["chain_leaves"] = {}
    C = ("own_f1", "own_top1", "own_top3", "own_brier", "comp_f1", "comp_top1", "kept_fraction", "owners", "assignments_tested", "own_seconds", "type_recall", "primitive_recovery", "primitive_equivalence", "primitive_precision", "n_literals", "candidate_functional", "candidate_best", "final_095", "final_099", "final_full", "fit_equivalents", "seconds")
    for a in arms:
        x = list(cb[a].values()); S["chain"][a] = {"n": len(x), **{k: _mean(r.get(k) for r in x) for k in C}}
        for d in sorted({r["depth"] for r in x}): S["chain_by_depth"][f"{a}/d{d}"] = {k: _mean(r.get(k) for r in x if r["depth"] == d) for k in ("primitive_recovery", "candidate_functional", "final_095", "final_full")}
        lv = {}
        for r in x:
            for l in r["leaves"]: lv.setdefault(l["type"], []).append(l)
        S["chain_leaves"][a] = {t: {"n": len(v), "type_hit": _mean(l["type_hit"] for l in v), "recovered": _mean(l["recovered"] for l in v)} for t, v in sorted(lv.items())}
        for ref in ("reference", "flat_scene_k3"):
            if ref in cb and a != ref: ts = sorted(set(cb[a]) & set(cb[ref])); S["chain_paired"][f"{a} − {ref}"] = {k: paired_ci([cb[ref][t][k] for t in ts], [cb[a][t][k] for t in ts]) for k in ("type_recall", "primitive_recovery", "candidate_functional", "final_095", "final_full")}
    if "own_k3" in cb: x = list(cb["own_k3"].values()); S["composition_reattribution"] = {"f1": paired_ci([r.get("own_f1") for r in x], [r.get("comp_f1") for r in x]), "top1": paired_ci([r.get("own_top1") for r in x], [r.get("comp_top1") for r in x])}
    return S


def aggregate():
    rows, bad = load_rows(); S = summarize(rows, bad); json.dump(S, open(SUMMARY, "w"), indent=1); print("lignes", len(rows), "unités en erreur", bad); return S


def show(S):
    ci = lambda v: {x: (y["diff"], y["ci95"]) if y else None for x, y in v.items()}
    print("\nATTRIBUTION"); [print(k.ljust(22), v) for k, v in S["attribution"].items()]; print(); [print(k.ljust(34), ci(v)) for k, v in S["attribution_paired"].items()]
    print("\npar part masquée"); [print(k.ljust(28), v) for k, v in S["attribution_by_mask"].items()]; print("\npar type"); [print(k.ljust(22), v) for k, v in S["attribution_by_kind"].items()]
    print("\ncalibration", S["calibration"]); print("par masque", S["calibration_by_mask"]); print("énergie", S["energy"]); print("\nMASQUE"); [print(k.ljust(6), v) for k, v in S["occlusion"].items()]
    print("\nMATRICE ORACLE"); [print(k.ljust(10), v) for k, v in S["oracle"].items()]; print(ci(S["oracle_paired"]))
    print("\nVISIBILITÉ CONTREFACTUELLE"); [print(k.ljust(10), v) for k, v in S["visibility"].items()]; print(ci(S["visibility_paired"]))
    print("\nSECTEURS"); [print(k.ljust(6), v) for k, v in S["sector"].items()]
    print("\nCHAÎNE AVAL"); K = ("n", "own_f1", "own_top1", "own_top3", "comp_f1", "comp_top1", "kept_fraction", "type_recall", "primitive_recovery", "primitive_precision", "n_literals", "candidate_functional", "final_095", "final_099", "final_full", "own_seconds", "seconds")
    [print(k.ljust(20), {x: v.get(x) for x in K if v.get(x) is not None}) for k, v in S["chain"].items()]; print("\nappariées"); [print(k.ljust(38), ci(v)) for k, v in S["chain_paired"].items()]
    print("composition → ré-attribution", ci(S.get("composition_reattribution", {}))); print("\nfeuilles"); [print(k.ljust(20), v) for k, v in S["chain_leaves"].items()]; print("\npar profondeur"); [print(k.ljust(26), v) for k, v in S["chain_by_depth"].items()]


def report(): show(json.load(open(SUMMARY)))


def _tune_scenes(n):
    from core.deep_ownership_exp import KINDS, MASKS, scene_at
    return [scene_at(j, kd, ms, nz, tune=True) for kd in KINDS for ms in MASKS for j in range(n) for nz in (0.0, 0.08)]


def smoke():
    from core.deep_ownership_exp import attr_rows, oracle_row, sector_row, vis_row
    rows = []
    for i, sc in enumerate(_tune_scenes(int(os.environ.get("BO_SCENES", "1")))):
        tag = {"scene": f"{i}"}; rows += [{"experiment": "attr", **tag, **r} for r in attr_rows(sc, i)] + [{"experiment": "oracle", **tag, **oracle_row(sc, i)}, {"experiment": "vis", **tag, **vis_row(sc, i)}]
        if sc.kind == "sector": rows.append({"experiment": "sector", **tag, **sector_row(sc, i)})
    show(summarize(rows))


def tune():
    """Sur scènes de MISE AU POINT : coûts (lam, gamma) par F1 d'attribution, puis température par erreur de calibration."""
    import itertools
    from core.deep_fragments_exp import pair_scores, true_sources
    from core.deep_ownership import CONFIG_FILE, DEFAULTS, solve
    from core.deep_ownership_exp import _atoms, ownership_accuracy
    from core.deep_partial import observe
    from core.deep_primitive_eval import components
    data = []
    for sc in _tune_scenes(int(os.environ.get("BO_SCENES", "1"))): ob = observe(sc.X, sc.y); at = _atoms(sc); data.append((ob, true_sources(ob.points, at)[0], [c for a in at for c in components(a)]))
    from core.deep_ownership import null_assigned
    nulls = []
    for ch in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0): v = null_assigned({**DEFAULTS, "gamma": 0.0, "chance": ch}); nulls.append({"chance": ch, "assigned_on_random_labels": _mean(v)}); print(nulls[-1], flush=True)
    chance = next((r["chance"] for r in nulls if r["assigned_on_random_labels"] <= 0.10), nulls[-1]["chance"]); print("coût de hasard retenu (premier sous 10 % d'attribution sur étiquettes aléatoires) :", chance, flush=True); DEFAULTS = {**DEFAULTS, "chance": chance}
    res = []
    for lam, gamma in itertools.product((6.0, 10.0, 16.0), (0.0, 0.5, 1.0)):
        sc_ = []
        for i, (ob, src, comps) in enumerate(data): sol = solve(ob, {**DEFAULTS, "lam": lam, "gamma": gamma}, i); sc_.append((pair_scores(sol.labels, src)["f1"], ownership_accuracy(sol, src, comps)["top1"]))
        res.append({"lam": lam, "gamma": gamma, "f1": _mean(a for a, _ in sc_), "top1": _mean(b for _, b in sc_)}); print(res[-1], flush=True)
    best = max(res, key=lambda r: (r["f1"] or 0) + (r["top1"] or 0)); temps = []
    for T in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        cal = [ownership_accuracy(solve(ob, {**DEFAULTS, "lam": best["lam"], "gamma": best["gamma"], "temperature": T}, i), src, comps)["cal"] for i, (ob, src, comps) in enumerate(data)]; c = calibration(cal); temps.append({"temperature": T, "ece": c["ece"], "accuracy": c.get("accuracy"), "mean_confidence": c.get("mean_confidence")}); print(temps[-1], flush=True)
    bt = min(temps, key=lambda r: r["ece"]); cfg = {"lam": best["lam"], "gamma": best["gamma"], "chance": chance, "temperature": bt["temperature"]}
    json.dump({"config": cfg, "selected_on": f"{len(data)} scènes de mise au point", "criterion": "coût de hasard par calibration nulle, puis F1 d'attribution + appartenance top-1, puis ECE", "null_calibration": nulls, "grid": res, "temperatures": temps}, open(CONFIG_FILE, "w"), indent=1); print("RETENU", cfg)


def chain_smoke():
    from core.deep_experiments import down_task
    from core.deep_ownership_exp import CHAIN_ARMS, chain
    for d in (int(x) for x in os.environ.get("BO_DEPTHS", "2,4").split(",")):
        t = down_task(799, d, "standard")
        for a in CHAIN_ARMS: r = chain(t, a, 0); print(f"d{d}", a.ljust(20), {k: (round(r[k], 3) if isinstance(r.get(k), float) else r.get(k)) for k in ("own_f1", "own_top1", "comp_f1", "type_recall", "primitive_recovery", "n_literals", "candidate_functional", "final_full", "seconds")}, flush=True)


if __name__ == "__main__":
    {"aggregate": aggregate, "report": report, "smoke": smoke, "tune": tune, "chain-smoke": chain_smoke}[sys.argv[1]]()

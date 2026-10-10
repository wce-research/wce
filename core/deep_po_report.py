"""Reports of the Partial-Observation Primitive Parameter Inference sub-milestone (dev only; never the sealed set).

  smoke       : short local demonstration on TUNING scenes
  chain-smoke : downstream chain on a tuning task (key 599)
  aggregate   : aggregates the AWS runs `deeppo` → results/deep_partial_observation_summary.json
  report      : prints the tables
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
SUMMARY = os.path.join(OUT, "deep_partial_observation_summary.json")
TOLS = {"center": 0.35, "radius": 0.30, "radius_in": 0.30, "radius_out": 0.30, "orientation": 0.22, "offset": 0.30, "width": 0.30, "direction": 0.22, "half_width": 0.22}


def load_rows():
    rows = []; bad = 0
    for f in glob.glob(os.path.join(OUT, "aws", "wce-deeppo-*", "artifacts", "*", "deeppo_*.json")):
        r = json.load(open(f))
        if r["status"] == "ok": rows += r["result"]
        else: bad += 1
    return rows, bad


def _median(xs):
    xs = sorted(float(x) for x in xs if x is not None); return round(xs[len(xs) // 2], 4) if xs else None


def summarize(rows, bad=0):
    from core.deep_partial_exp import BASELINE, MAIN
    S = {"n_rows": len(rows), "units_in_error": bad}; mk = [r for r in rows if r["experiment"] == "mask" and not r.get("not_applicable")]; methods = sorted({r["method"] for r in mk}); kinds = sorted({r["kind"] for r in mk}); masks = sorted({r["mask"] for r in mk}); noises = sorted({r["noise"] for r in mk})
    K = ("recovered_1", "recovered_3", "recovered_5", "functional_1", "functional_5", "no_hypothesis", "n_hypotheses", "fit_equivalents", "seconds")
    cell = lambda x: {"n": len(x), **{k: _mean(r.get(k) for r in x) for k in K}}
    S["by_method"] = {m: cell([r for r in mk if r["method"] == m]) for m in methods}
    S["by_method_mask"] = {f"{m}/{int(ms * 100)}": cell([r for r in mk if r["method"] == m and r["mask"] == ms]) for m in methods for ms in masks}
    S["by_kind_mask"] = {f"{m}/{kd}/{int(ms * 100)}": cell([r for r in mk if r["method"] == m and r["kind"] == kd and r["mask"] == ms]) for m in (MAIN, BASELINE, "global_target", "ctx_joint", "ctx_oracle", "arc_joint") if m in methods for kd in kinds for ms in masks}
    S["noise_vs_mask"] = {f"{m}/noise{nz}/mask{int(ms * 100)}": cell([r for r in mk if r["method"] == m and r["noise"] == nz and r["mask"] == ms]) for m in (MAIN, BASELINE) if m in methods for nz in noises for ms in masks}
    # visible-boundary threshold: largest occluded share at which recovered_1 stays ≥ 0.5 (main method, without noise)
    S["visible_needed"] = {}
    for kd in kinds:
        ok = [ms for ms in masks if (_mean(r.get("recovered_1") for r in mk if r["method"] == MAIN and r["kind"] == kd and r["mask"] == ms and r["noise"] == 0.0) or 0) >= 0.5]
        ok5 = [ms for ms in masks if (_mean(r.get("recovered_5") for r in mk if r["method"] == MAIN and r["kind"] == kd and r["mask"] == ms and r["noise"] == 0.0) or 0) >= 0.5]
        S["visible_needed"][kd] = {"top1_min_visible": round(1 - max(ok), 2) if ok else None, "top5_min_visible": round(1 - max(ok5), 2) if ok5 else None}
    # errors per parameter
    S["param_errors"] = {}; S["param_fragility"] = {}
    for m in (MAIN, BASELINE, "global_target"):
        for kd in kinds:
            for ms in masks:
                x = [r for r in mk if r["method"] == m and r["kind"] == kd and r["mask"] == ms and r.get("errors")]; names = sorted({k for r in x for k in r["errors"]})
                if x: S["param_errors"][f"{m}/{kd}/{int(ms * 100)}"] = {k: _median(r["errors"].get(k) for r in x) for k in names}
    fr = defaultdict(list)
    for r in mk:
        if r["method"] == MAIN and r.get("errors"):
            for k, v in r["errors"].items(): fr[(r["kind"], k)].append(v > TOLS[k])
    S["param_fragility"] = {f"{kd}/{k}": {"n": len(v), "out_of_tolerance": round(sum(v) / len(v), 4)} for (kd, k), v in sorted(fr.items())}
    # paired differences (same scene, same mask, same noise)
    key = lambda r: (r["scene"], r["mask"], r["noise"]); by = {m: {key(r): r for r in mk if r["method"] == m} for m in methods}; S["paired"] = {}
    for a, b in ((BASELINE, MAIN), ("global_target", MAIN), ("arc_kasa", "arc_geometric"), ("arc_taubin", "arc_geometric"), ("arc_normals", "arc_geometric"), ("arc_geometric", "arc_joint"), ("arc_joint", MAIN), (MAIN, "ctx_joint"), (MAIN, "ctx_oracle"), (MAIN, "given_center"), (MAIN, "given_orientation"), (MAIN, "given_size")):
        if a in by and b in by:
            ks = sorted(set(by[a]) & set(by[b])); S["paired"][f"{b} − {a}"] = {k: paired_ci([by[a][s].get(k) for s in ks], [by[b][s].get(k) for s in ks]) for k in ("recovered_1", "recovered_5", "functional_1")}
            for sub, f in (("masked", lambda s: s[1] >= 0.4), ("rich", lambda s: by[a][s]["kind"] in ("ring", "band", "sector"))):
                kk = [s for s in ks if f(s)]; S["paired"][f"{b} − {a} [{sub}]"] = {k: paired_ci([by[a][s].get(k) for s in kk], [by[b][s].get(k) for s in kk]) for k in ("recovered_1", "recovered_5", "functional_1")}
    # minimal versus rich under occlusion (recall of all the components among 5 hypotheses)
    S["minimal_vs_rich"] = {f"{kd}/{int(ms * 100)}": {"rich_recovered_5": _mean(r.get("recovered_5") for r in mk if r["method"] == MAIN and r["kind"] == kd and r["mask"] == ms), "minimal_recovered_5": _mean(r.get("recovered_5") for r in mk if r["method"] == "minimal_components" and r["kind"] == kd and r["mask"] == ms)} for kd in kinds for ms in masks}
    # regions, identifiability, completion
    mm = [r for r in mk if r["method"] == MAIN and "region_covers" in r]; S["regions"] = {}
    for ms in masks:
        x = [r for r in mm if r["mask"] == ms]; ids = Counter(v for r in x for v in r["identifiability"].values()); tot = max(1, sum(ids.values()))
        S["regions"][str(int(ms * 100))] = {"n": len(x), "coverage": _mean(r["region_covers"] for r in x), "top1_recovered": _mean(r["recovered_1"] for r in x), "identifiability": {k: round(v / tot, 3) for k, v in ids.items()}, "mean_width": _mean(w for r in x for w in r["widths"].values()),
                                       "arc_angle": _mean(r["sensitivity"].get("arc_angle") for r in x), "condition": _median(r["sensitivity"].get("condition") for r in x), "corr_radius_center": _mean(abs(r["sensitivity"]["corr_radius_center"]) for r in x if "corr_radius_center" in r["sensitivity"]),
                                       "posterior_sd_r": _mean(r["posterior_sd"].get("r") for r in x)}
    lab = defaultdict(list)
    for r in mm:
        for p, v in r["identifiability"].items():
            e = {"cx": "center", "cy": "center", "r": "radius", "angle": "orientation", "offset": "offset"}[p]
            if e in r["errors"]: lab[v].append(r["errors"][e] <= TOLS[e])
    S["identifiability_calibration"] = {k: {"n": len(v), "within_tolerance": round(sum(v) / len(v), 4)} for k, v in lab.items()}
    S["completion"] = {str(int(ms * 100)): {"visible_true": _mean(r["visible_true"] for r in mk if r["method"] == MAIN and r["mask"] == ms), "visible_estimated": _mean(r.get("visible_estimated") for r in mk if r["method"] == MAIN and r["mask"] == ms),
                                             "observed_fraction": _mean(r.get("observed_fraction") for r in mk if r["method"] == MAIN and r["mask"] == ms), "completion_accuracy": _mean(r.get("completion_accuracy") for r in mk if r["method"] == MAIN and r["mask"] == ms)} for ms in masks}
    # autopsy of the sectors
    sx = [r for r in mk if r["method"] == MAIN and r["kind"] == "sector"]; S["sector"] = {}
    for ms in masks:
        x = [r for r in sx if r["mask"] == ms]; fail = [r for r in x if not r["recovered_1"]]; c = Counter()
        for r in fail:
            e = r.get("errors") or {}
            c["no_hypothesis" if r["no_hypothesis"] else "center" if e.get("center", 0) > TOLS["center"] else "angles" if max(e.get("direction", 0), e.get("half_width", 0)) > TOLS["direction"] else "other"] += 1
        S["sector"][str(int(ms * 100))] = {"n": len(x), "recovered_1": _mean(r["recovered_1"] for r in x), "recovered_5": _mean(r["recovered_5"] for r in x), "failures": dict(c), "rays_found": _mean(r["sector"]["rays_found"] for r in x), "shortest_ray_visible": _mean(min(r["sector"]["ray_visible_length"]) for r in x),
                                      "true_z": _mean(r["sector"]["true_z"] for r in x), "true_passes_calibrated": _mean(r["sector"]["true_z"] >= 10.88 for r in x), "true_passes_nominal": _mean(r["sector"]["true_z"] >= 3.0 for r in x),
                                      "median_errors": {k: _median((r.get("errors") or {}).get(k) for r in x) for k in ("center", "direction", "half_width")}}
    # active measurements
    ac = [r for r in rows if r["experiment"] == "active"]; ba = {m: {(r["scene"], r["mask"]): r for r in ac if r["method"] == m} for m in sorted({r["method"] for r in ac})}
    S["active"] = {m: {"n": len(v), "recovered_1": _mean(r["recovered_1"] for r in v.values()), "recovered_5": _mean(r["recovered_5"] for r in v.values()), "functional_1": _mean(r["functional_1"] for r in v.values())} for m, v in ba.items()}; S["active_paired"] = {}
    for m in ba:
        if m != "active_random" and "active_random" in ba: ks = sorted(set(ba[m]) & set(ba["active_random"])); S["active_paired"][f"{m} − active_random"] = {k: paired_ci([ba["active_random"][s][k] for s in ks], [ba[m][s][k] for s in ks]) for k in ("recovered_1", "functional_1")}
    # downstream chain
    ch = [r for r in rows if r["experiment"] == "chain"]; arms = sorted({r["arm"] for r in ch}); cb = {a: {r["task_id"]: r for r in ch if r["arm"] == a} for a in arms}; S["chain"] = {}; S["chain_by_depth"] = {}; S["chain_paired"] = {}; S["chain_leaves"] = {}
    C = ("primitive_equivalence", "component_recall", "primitive_precision", "n_literals", "candidate_functional", "candidate_best", "final_095", "final_099", "final_full", "loop_095", "loop_099", "loop_full", "loop_commits", "loop_evaluations", "fit_equivalents", "seconds")
    for a in arms:
        x = list(cb[a].values()); S["chain"][a] = {"n": len(x), **{k: _mean(r.get(k) for r in x) for k in C}}
        for d in sorted({r["depth"] for r in x}): S["chain_by_depth"][f"{a}/d{d}"] = {k: _mean(r.get(k) for r in x if r["depth"] == d) for k in ("primitive_equivalence", "candidate_functional", "final_full", "loop_full")}
        lv = defaultdict(list)
        for r in x:
            for l in r.get("leaves") or []: lv[l["type"]].append(l); lv["all"].append(l)
        if lv: S["chain_leaves"][a] = {t: {"n": len(v), "top1": _mean(l["top1"] for l in v), "topk": _mean(l["topk"] for l in v), "functional_k": _mean(l["functional_k"] for l in v)} for t, v in sorted(lv.items())}
        for ref in ("reference", "types_old", "po_type_k1"):
            if ref in cb and a != ref:
                ts = sorted(set(cb[a]) & set(cb[ref])); S["chain_paired"][f"{a} − {ref}"] = {k: paired_ci([cb[ref][t][k] for t in ts], [cb[a][t][k] for t in ts]) for k in ("primitive_equivalence", "candidate_functional", "final_full", "loop_full")}
        S["chain_paired"][f"{a} : loop − final"] = {"final_full": paired_ci([r["final_full"] for r in x], [r["loop_full"] for r in x]), "final_095": paired_ci([r["final_095"] for r in x], [r["loop_095"] for r in x])}
    return S


def aggregate():
    rows, bad = load_rows(); S = summarize(rows, bad); json.dump(S, open(SUMMARY, "w"), indent=1); print("rows", len(rows), "units in error", bad); return S


def show(S):
    sub = lambda v, ks: {k: v.get(k) for k in ks}; R = ("n", "recovered_1", "recovered_3", "recovered_5", "functional_1", "functional_5")
    print("\nOCCLUSION — all scenes"); [print(k.ljust(22), sub(v, R + ("n_hypotheses", "seconds"))) for k, v in S["by_method"].items()]
    print("\nby method and occluded share (recovered_1 / recovered_5 / functional_1)"); [print(k.ljust(26), v["recovered_1"], v["recovered_5"], v["functional_1"]) for k, v in S["by_method_mask"].items()]
    print("\nby type and occluded share"); [print(k.ljust(34), v["n"], v["recovered_1"], v["recovered_5"], v["functional_1"]) for k, v in S["by_kind_mask"].items()]
    print("\nnoise versus occlusion"); [print(k.ljust(38), v["n"], v["recovered_1"], v["recovered_5"], v["functional_1"]) for k, v in S["noise_vs_mask"].items()]
    print("\nminimal visible boundary (recovered ≥ 0.5, without noise)", S["visible_needed"])
    print("\nmedian error per parameter"); [print(k.ljust(34), v) for k, v in S["param_errors"].items()]
    print("\nfragility of the parameters (out of tolerance, main method)"); [print(k.ljust(24), v) for k, v in S["param_fragility"].items()]
    print("\npaired differences, 95% CI"); [print(k.ljust(48), {x: (y["diff"], y["ci95"], y["n"]) if y else None for x, y in v.items()}) for k, v in S["paired"].items()]
    print("\nminimal versus rich"); [print(k.ljust(14), v) for k, v in S["minimal_vs_rich"].items()]
    print("\nparameter regions"); [print(k.ljust(6), v) for k, v in S["regions"].items()]; print("calibration of the identifiability labels", S["identifiability_calibration"])
    print("\nboundary completion"); [print(k.ljust(6), v) for k, v in S["completion"].items()]
    print("\nautopsy of the sectors"); [print(k.ljust(6), v) for k, v in S["sector"].items()]
    print("\nactive measurements"); [print(k.ljust(26), v) for k, v in S["active"].items()]; [print(k.ljust(44), {x: (y["diff"], y["ci95"]) if y else None for x, y in v.items()}) for k, v in S["active_paired"].items()]
    print("\nDOWNSTREAM CHAIN"); [print(k.ljust(24), sub(v, ("n", "primitive_equivalence", "primitive_precision", "n_literals", "candidate_functional", "final_095", "final_099", "final_full", "loop_095", "loop_099", "loop_full", "loop_commits", "seconds"))) for k, v in S["chain"].items()]
    print("\nleaves (known type)"); [print(k.ljust(24), v) for k, v in S["chain_leaves"].items()]
    print("\npaired"); [print(k.ljust(44), {x: (y["diff"], y["ci95"]) if y else None for x, y in v.items()}) for k, v in S["chain_paired"].items()]
    print("\nby depth"); [print(k.ljust(30), v) for k, v in S["chain_by_depth"].items()]


def report(): show(json.load(open(SUMMARY)))


def smoke():
    from core.deep_partial_exp import KINDS, MASKS, METHODS, NOISES, active_row, mask_row, masked_scene
    n = int(os.environ.get("PO_SCENES", "2")); kinds = os.environ.get("PO_KINDS", ",".join(KINDS)).split(","); meth = os.environ.get("PO_METHODS", ",".join(METHODS)).split(","); rows = []
    for kd in kinds:
        for ms in MASKS:
            for j in range(n): rows += [{"experiment": "mask", "scene": f"{j}_{kd}", **mask_row(masked_scene(j, kd, ms, nz, tune=True), m, j)} for nz in NOISES for m in meth]
    if os.environ.get("PO_ACTIVE"): rows += [{"experiment": "active", "scene": f"{j}_{kd}", **active_row(masked_scene(j, kd, 0.4, 0.05, tune=True), p, j)} for kd in kinds for j in range(n) for p in ("none", "random", "disagreement", "disagreement_ctx")]
    show(summarize(rows))


def chain_smoke():
    from core.deep_experiments import down_task
    from core.deep_partial_exp import CHAIN_ARMS, chain
    t = down_task(599, int(os.environ.get("PO_DEPTH", "3")), "standard")
    for a in CHAIN_ARMS: r = chain(t, a[0], 0); print(a[0].ljust(24), {k: r[k] for k in ("primitive_equivalence", "n_literals", "candidate_functional", "final_full", "loop_full", "loop_commits", "seconds")}, flush=True)


if __name__ == "__main__":
    {"aggregate": aggregate, "report": report, "smoke": smoke, "chain-smoke": chain_smoke}[sys.argv[1]]()

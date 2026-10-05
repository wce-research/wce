"""Rapports du sous-jalon Fragment Attribution & Primitive Type Inference (dev uniquement ; jamais le lot scellé).

  smoke / chain-smoke : démonstrations locales courtes sur scènes et tâche de MISE AU POINT
  aggregate           : agrège les runs AWS `deepft` → results/deep_fragment_type_summary.json
  report              : imprime les tableaux
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
SUMMARY = os.path.join(OUT, "deep_fragment_type_summary.json")


def load_rows():
    rows = []; bad = 0
    for f in glob.glob(os.path.join(OUT, "aws", "wce-deepft-*", "artifacts", "*", "deepft_*.json")):
        r = json.load(open(f))
        if r["status"] == "ok": rows += r["result"]
        else: bad += 1
    return rows, bad


def summarize(rows, bad=0):
    S = {"n_rows": len(rows), "units_in_error": bad}; key = lambda r: (r["scene"], r["mask"], r["noise"])
    # ---- attribution
    gr = [r for r in rows if r["experiment"] == "group"]; modes = sorted({r["grouping"] for r in gr}); masks = sorted({r["mask"] for r in gr}); G = ("precision", "recall", "f1", "assigned", "groups", "mixed_groups", "seconds")
    S["grouping"] = {m: {"n": sum(r["grouping"] == m for r in gr), **{k: _mean(r[k] for r in gr if r["grouping"] == m) for k in G}} for m in modes}
    S["grouping_by_mask"] = {f"{m}/{int(ms * 100)}": {k: _mean(r[k] for r in gr if r["grouping"] == m and r["mask"] == ms) for k in ("precision", "recall", "f1", "mixed_groups")} for m in modes for ms in masks}
    bg = {m: {key(r): r for r in gr if r["grouping"] == m} for m in modes}; S["grouping_paired"] = {}
    for m in modes:
        if m != "proximity" and "proximity" in bg: ks = sorted(set(bg[m]) & set(bg["proximity"])); S["grouping_paired"][f"{m} − proximity"] = {k: paired_ci([bg["proximity"][s][k] for s in ks], [bg[m][s][k] for s in ks]) for k in ("f1", "precision", "recall", "mixed_groups")}
    # ---- type
    ty = [r for r in rows if r["experiment"] == "type"]; variants = sorted({r["variant"] for r in ty}); kinds = sorted({r["kind"] for r in ty}); T = ("correct", "exact_1", "exact_3", "type_1", "type_3", "equivalent_1", "equivalent_3", "functional_1", "functional_3", "n_hypotheses", "types_tested", "fragments", "groups", "seconds")
    S["type"] = {v: {"n": sum(r["variant"] == v for r in ty), **{k: _mean(r.get(k) for r in ty if r["variant"] == v) for k in T}} for v in variants}
    S["type_by_mask"] = {f"{v}/{int(ms * 100)}": {k: _mean(r.get(k) for r in ty if r["variant"] == v and r["mask"] == ms) for k in ("correct", "exact_1", "type_1", "type_3", "equivalent_3", "functional_1")} for v in variants for ms in masks}
    S["type_by_kind"] = {f"{v}/{kd}": {k: _mean(r.get(k) for r in ty if r["variant"] == v and r["kind"] == kd) for k in ("correct", "exact_1", "type_1", "type_3", "equivalent_3", "functional_1")} for v in variants for kd in kinds}
    S["type_by_kind_mask"] = {f"{kd}/{int(ms * 100)}": {k: _mean(r.get(k) for r in ty if r["variant"] == "hier" and r["kind"] == kd and r["mask"] == ms) for k in ("correct", "exact_1", "type_1", "type_3")} for kd in kinds for ms in masks}
    bt = {v: {key(r): r for r in ty if r["variant"] == v} for v in variants}; S["type_paired"] = {}
    for a, b in (("flat_scene", "hier"), ("flat_fragments", "hier"), ("hier_no_negative", "hier"), ("hier_naive_negative", "hier"), ("hier", "hier_loop")):
        if a in bt and b in bt: ks = sorted(set(bt[a]) & set(bt[b])); S["type_paired"][f"{b} − {a}"] = {k: paired_ci([bt[a][s][k] for s in ks], [bt[b][s][k] for s in ks]) for k in ("correct", "exact_1", "type_1", "type_3", "equivalent_3", "functional_1")}
    hx = [r for r in ty if r["variant"] == "hier"]; cm = defaultdict(Counter)
    for r in hx: cm[r["kind"]][r["predicted"]] += 1
    S["confusion"] = {kd: dict(c) for kd, c in sorted(cm.items())}; bins = defaultdict(list)
    for r in hx:
        if r["confidence"] is not None: bins[min(4, int(r["confidence"] * 5))].append(r["correct"])
    S["calibration"] = {f"{b * 0.2:.1f}-{b * 0.2 + 0.2:.1f}": {"n": len(v), "accuracy": round(sum(v) / len(v), 4)} for b, v in sorted(bins.items())}; tot = sum(len(v) for v in bins.values())
    S["ece"] = round(sum(len(v) / tot * abs(sum(v) / len(v) - (b * 0.2 + 0.1)) for b, v in bins.items()), 4) if tot else None
    sg = defaultdict(lambda: defaultdict(list))
    for r in hx:
        for k, v in (r.get("signals") or {}).items(): sg["correct" if r["correct"] else "wrong"][k].append(v)
    S["signals"] = {c: {k: _mean(v) for k, v in d.items()} for c, d in sg.items()}
    # ---- matrice oracle
    orc = [r for r in rows if r["experiment"] == "oracle"]; O = ("TT", "TI", "TI_type", "ET", "EI")
    S["oracle"] = {"all": {"n": len(orc), **{k: _mean(r[k] for r in orc) for k in O}}, **{f"mask{int(ms * 100)}": {k: _mean(r[k] for r in orc if r["mask"] == ms) for k in O} for ms in masks}, **{kd: {k: _mean(r[k] for r in orc if r["kind"] == kd) for k in O} for kd in kinds}}
    if orc: S["oracle_paired"] = {f"{b} − {a}": paired_ci([r[a] for r in orc], [r[b] for r in orc]) for a, b in (("EI", "ET"), ("EI", "TI"), ("ET", "TT"), ("TI", "TT"), ("EI", "TT"))}
    # ---- propriété du masquage
    ow = [r for r in rows if r["experiment"] == "own"]; W = ("hidden_as_occluded", "hidden_as_absent", "visible_as_observed", "visible_as_occluded", "visible_as_absent", "owner_found", "owner_is_occluder")
    S["ownership"] = {str(int(ms * 100)): {"n": sum(r["mask"] == ms for r in ow), **{k: _mean(r[k] for r in ow if r["mask"] == ms) for k in W}} for ms in masks}
    # ---- secteurs
    sx = [r for r in rows if r["experiment"] == "sector"]
    S["sector"] = {str(int(ms * 100)): {"n": sum(r["mask"] == ms for r in sx), "exact_1": _mean(r["exact_1"] for r in sx if r["mask"] == ms), "rays_found": _mean(r["rays_found"] for r in sx if r["mask"] == ms), "vertex_relations": _mean(r["vertex_relations"] > 0 for r in sx if r["mask"] == ms),
                                           "confused_with_disc": _mean(r["confused_with_disc"] for r in sx if r["mask"] == ms), "stages": dict(Counter(r["stage"] for r in sx if r["mask"] == ms)), "top_types": dict(Counter(r["top_type"] for r in sx if r["mask"] == ms))} for ms in masks}
    # ---- mesure active
    ac = [r for r in rows if r["experiment"] == "active"]; ba = {p: {(r["scene"], r["mask"]): r for r in ac if r["policy"] == p} for p in sorted({r["policy"] for r in ac})}
    S["active"] = {p: {"n": len(v), **{k: _mean(r[k] for r in v.values()) for k in ("correct", "exact_1", "type_1", "functional_1")}} for p, v in ba.items()}; S["active_paired"] = {}
    for p in ba:
        if p != "random" and "random" in ba: ks = sorted(set(ba[p]) & set(ba["random"])); S["active_paired"][f"{p} − random"] = {k: paired_ci([ba["random"][s][k] for s in ks], [ba[p][s][k] for s in ks]) for k in ("correct", "exact_1")}
    # ---- chaîne aval
    ch = [r for r in rows if r["experiment"] == "chain"]; arms = sorted({r["arm"] for r in ch}); cb = {a: {r["task_id"]: r for r in ch if r["arm"] == a} for a in arms}; S["chain"] = {}; S["chain_paired"] = {}; S["chain_by_depth"] = {}; S["chain_leaves"] = {}
    C = ("attribution_f1", "type_recall", "parameter_recovery", "primitive_equivalence", "primitive_precision", "n_literals", "candidate_functional", "candidate_best", "final_095", "final_099", "final_full", "types_tested", "fragments", "groups", "inference_seconds", "fit_equivalents", "seconds")
    for a in arms:
        x = list(cb[a].values()); S["chain"][a] = {"n": len(x), **{k: _mean(r.get(k) for r in x) for k in C}}
        for d in sorted({r["depth"] for r in x}): S["chain_by_depth"][f"{a}/d{d}"] = {k: _mean(r.get(k) for r in x if r["depth"] == d) for k in ("type_recall", "parameter_recovery", "candidate_functional", "final_full")}
        lv = defaultdict(list)
        for r in x:
            for l in r["leaves"]: lv[l["type"]].append(l)
        S["chain_leaves"][a] = {t: {"n": len(v), "type_hit": _mean(l["type_hit"] for l in v), "recovered": _mean(l["recovered"] for l in v), "equivalent": _mean(l["equivalent"] for l in v)} for t, v in sorted(lv.items())}
        for ref in ("reference", "po_type_k3", "hier_k3"):
            if ref in cb and a != ref: ts = sorted(set(cb[a]) & set(cb[ref])); S["chain_paired"][f"{a} − {ref}"] = {k: paired_ci([cb[ref][t][k] for t in ts], [cb[a][t][k] for t in ts]) for k in ("type_recall", "parameter_recovery", "primitive_equivalence", "candidate_functional", "final_095", "final_full")}
    return S


def aggregate():
    rows, bad = load_rows(); S = summarize(rows, bad); json.dump(S, open(SUMMARY, "w"), indent=1); print("lignes", len(rows), "unités en erreur", bad); return S


def show(S):
    ci = lambda v: {x: (y["diff"], y["ci95"]) if y else None for x, y in v.items()}
    print("\nATTRIBUTION (paires de milieux)"); [print(k.ljust(14), v) for k, v in S["grouping"].items()]; [print(k.ljust(28), ci(v)) for k, v in S["grouping_paired"].items()]
    print("\npar part masquée"); [print(k.ljust(20), v) for k, v in S["grouping_by_mask"].items()]
    print("\nTYPE"); [print(k.ljust(22), v) for k, v in S["type"].items()]; [print(k.ljust(34), ci(v)) for k, v in S["type_paired"].items()]
    print("\npar part masquée"); [print(k.ljust(28), v) for k, v in S["type_by_mask"].items()]
    print("\npar type vrai"); [print(k.ljust(30), v) for k, v in S["type_by_kind"].items()]
    print("\nhier, par type et masque"); [print(k.ljust(16), v) for k, v in S["type_by_kind_mask"].items()]
    print("\nconfusion (type vrai → type prédit pour le groupe de la cible)"); [print(k.ljust(10), v) for k, v in S["confusion"].items()]; print("calibration", S["calibration"], "ECE", S["ece"]); print("signaux", S["signals"])
    print("\nMATRICE ORACLE (récupération exacte)"); [print(k.ljust(10), v) for k, v in S["oracle"].items()]; print(ci(S.get("oracle_paired", {})))
    print("\nPROPRIÉTÉ DU MASQUAGE"); [print(k.ljust(6), v) for k, v in S["ownership"].items()]
    print("\nSECTEURS"); [print(k.ljust(6), v) for k, v in S["sector"].items()]
    print("\nMESURE ACTIVE"); [print(k.ljust(20), v) for k, v in S["active"].items()]; [print(k.ljust(30), ci(v)) for k, v in S["active_paired"].items()]
    print("\nCHAÎNE AVAL"); K = ("n", "attribution_f1", "type_recall", "parameter_recovery", "primitive_equivalence", "primitive_precision", "n_literals", "candidate_functional", "final_095", "final_099", "final_full", "types_tested", "inference_seconds", "seconds")
    [print(k.ljust(30), {x: v.get(x) for x in K}) for k, v in S["chain"].items()]; print("\nappariées"); [print(k.ljust(46), ci(v)) for k, v in S["chain_paired"].items()]
    print("\nfeuilles"); [print(k.ljust(30), v) for k, v in S["chain_leaves"].items()]; print("\npar profondeur"); [print(k.ljust(34), v) for k, v in S["chain_by_depth"].items()]


def report(): show(json.load(open(SUMMARY)))


def smoke():
    from core.deep_fragments_exp import KINDS, MASKS, VARIANTS, active_row, group_rows, oracle_row, own_row, scene_at, sector_row, type_row
    n = int(os.environ.get("FT_SCENES", "2")); kinds = os.environ.get("FT_KINDS", ",".join(KINDS)).split(","); rows = []
    for kd in kinds:
        for ms in MASKS:
            for j in range(n):
                for nz in (0.0, 0.08):
                    sc = scene_at(j, kd, ms, nz, tune=True); tag = {"scene": f"{j}_{kd}"}; rows += [{"experiment": "group", **tag, **r} for r in group_rows(sc, j)] + [{"experiment": "type", **tag, **type_row(sc, v, j)} for v in VARIANTS] + [{"experiment": "oracle", **tag, **oracle_row(sc, j)}, {"experiment": "own", **tag, **own_row(sc, j)}]
                    if kd == "sector": rows.append({"experiment": "sector", **tag, **sector_row(sc, j)})
    if os.environ.get("FT_ACTIVE"): rows += [{"experiment": "active", "scene": f"{j}_{kd}", **active_row(scene_at(j, kd, 0.4, 0.05, tune=True), p, j)} for kd in kinds for j in range(n) for p in ("none", "random", "type_disagreement")]
    show(summarize(rows))


def chain_smoke():
    from core.deep_experiments import down_task
    from core.deep_fragments_exp import CHAIN_ARMS, chain
    for d in (int(x) for x in os.environ.get("FT_DEPTHS", "3").split(",")):
        t = down_task(699, d, "standard")
        for a in CHAIN_ARMS: r = chain(t, a[0], 0); print(f"d{d}", a[0].ljust(30), {k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in ("attribution_f1", "type_recall", "parameter_recovery", "n_literals", "candidate_functional", "final_full", "seconds")}, flush=True)


if __name__ == "__main__":
    {"aggregate": aggregate, "report": report, "smoke": smoke, "chain-smoke": chain_smoke}[sys.argv[1]]()

"""Rapports du sous-jalon Deep Candidate Generation (dev uniquement).

  reachability : analyse MÉCANIQUE (aucun ajustement) des vraies structures de dev par profondeur
  coverage     : couverture exacte de l'espace des formes à petite profondeur (énumération exhaustive)
  aggregate    : agrège les runs AWS `deepgen` → results/deep_generation_summary.json
  report       : imprime les tableaux du rapport

Module d'évaluation : il lit la vérité des tâches de DEV. Il ne touche jamais au lot scellé.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")


def reachability(n=120):
    import core.deep_reachability as R
    import core.deep_world as W
    rows = {}
    for d in W.DEPTHS:
        ts = [W.make_structure(W.SPLIT_BASE["dev"] + 5000 + i, d, False) for i in range(n)]; op = [R.oracle_path(t) for t in ts]; bq = [R.basis_quality(t) for t in ts]
        rows[d] = {"n": n, "representable": sum(R.representable(t)[0] for t in ts) / n, "space_forms": R.space_size(d), "oracle_length": sum(o["length"] for o in op) / n, "width_mean": sum(o["width"] for o in op) / n,
                   "width_max": max(o["width"] for o in op), "needs_compose_pair": sum(o["needs_compose_pair"] for o in op) / n, "alternatives_log10": sum(o["alternatives_product_log10"] for o in op) / n,
                   "parent_closure": sum(R.parent_closure(t)["closed"] for t in ts) / n, "atoms_minimal_basis": sum(b["atoms_in_minimal_basis"] for b in bq) / n, "redundant_atoms": sum(b["redundant_atoms"] for b in bq) / n,
                   "params": sum(b["params"] for b in bq) / n, "reachable": {m: sum(R.operator_reachable(t, ops)[0] for t in ts) / n for m, ops in R.OPERATORS.items()}}
    json.dump(rows, open(os.path.join(OUT, "deep_reachability.json"), "w"), indent=1)
    print("prof | représentable | formes T(d) | chemin oracle | largeur moy/max | clôture | atomes en base minimale | atteignable : chaîne (baseline, flat, top_down) / bottom_up / factorized")
    for d, r in rows.items():
        print(d, r["representable"], f"{r['space_forms']:.3g}", r["oracle_length"], f"{r['width_mean']:.2f}/{r['width_max']}", r["parent_closure"], f"{r['atoms_minimal_basis']:.2f}", f"{r['reachable']['baseline']:.2f} / {r['reachable']['bottom_up']:.2f} / {r['reachable']['factorized']:.2f}")
    return rows


def enumerate_forms(depth, atom_list=None):
    """Toutes les formes algébriques distinctes à `depth` atomes (énumération exhaustive ; faisable jusqu'à 3)."""
    from core.boundary_program import BNode, atoms
    from core.deep_structure import COMBINATORS, algebraic_form
    A = atom_list or atoms(); level = {1: {algebraic_form(a): a for a in A}}
    for d in range(2, depth + 1):
        cur = {}
        for i in range(1, d // 2 + 1):
            for fa, a in level[i].items():
                for fb, b in level[d - i].items():
                    for c in COMBINATORS:
                        p = BNode(c, children=(a, b)); cur.setdefault(algebraic_form(p), p)
        level[d] = cur
    return level


def coverage():
    """Couverture exacte : taille de l'espace énuméré (profondeur ≤ 3) et effet du typage sur les nœuds bruts."""
    from core.boundary_program import BNode, atoms
    from core.deep_generation import well_typed
    from core.deep_reachability import space_size
    lv = enumerate_forms(3); exact = {d: len(v) for d, v in lv.items()}
    feats, ops = ("dot", "norm", "angle"), ("le", "ge", "in"); preds = [BNode("pred", f, o) for f in feats for o in ops]
    raw = preds + [BNode("not", children=(p,)) for p in preds] + [BNode("translate", children=(p,)) for p in preds] + [BNode("translate", children=(BNode("not", children=(p,)),)) for p in preds] + [BNode("not", children=(BNode("translate", children=(p,)),)) for p in preds]
    typed = [n for n in raw if well_typed(n)]; gram = len(atoms())
    out = {"exact_forms": exact, "upper_bound_forms": {d: space_size(d) for d in range(1, 7)}, "raw_unary_nodes": len(raw), "well_typed": len(typed), "grammar_atoms": gram,
           "typing_reduction_atoms": 1 - len(typed) / len(raw), "typing_reduction_depth3": 1 - (len(typed) / len(raw)) ** 3}
    json.dump(out, open(os.path.join(OUT, "deep_coverage.json"), "w"), indent=1); print(json.dumps(out, indent=1)); return out


def fragments(n_curr=60, n_eval=60):
    """Bibliothèque de fragments comme aide à la reachability — BORNE HAUTE : la bibliothèque est construite à partir
    des VRAIES structures de tâches de curriculum (profondeurs 2–3, graines de dev disjointes), comme si les
    profondeurs précédentes avaient été parfaitement résolues. Mesure, sur d'autres tâches de dev (profondeurs 3–6) :
    part des sous-structures vraies à 2 atomes présentes (fragments exacts), couvertes par un schéma anti-unifié, faux
    schémas proposés, et raccourcissement du chemin oracle."""
    import core.deep_world as W
    from core.boundary_program import atoms
    from core.deep_structure import FragmentLibrary, n_atoms, skel_key, subtrees
    lib = FragmentLibrary()
    for i in range(n_curr):
        for d in (2, 3): lib.observe(W.make_structure(W.SPLIT_BASE["dev"] + 20000 + i * 7 + d, d, False), f"c{i}_{d}")
    lib.promote(); A = atoms(); fam1 = [f for f in lib.families if len(f.variables) == 1]
    proposed = {skel_key(f.instantiate({f.variables[0]: a})) for f in fam1 for a in A}; exact = {f.key for f in lib.fragments}; out = {"library_size": len(lib), "families": [f.render() for f in lib.families], "schemas_proposed_by_families": len(proposed),
        "new_schemas_from_families": len(proposed - exact), "two_atom_forms_total": 108, "by_depth": {}}
    for d in (3, 4, 5, 6):
        tot = hit = au = short = 0
        for i in range(n_eval):
            tr = W.make_structure(W.SPLIT_BASE["dev"] + 40000 + i * 11 + d, d, False); subs = [s for s in subtrees(tr, 2) if n_atoms(s) == 2]
            tot += len(subs); hit += sum(lib.contains(s) for s in subs); au += sum(skel_key(s) in proposed or lib.contains(s) for s in subs); short += sum(lib.contains(s) for s in subs) / max(1, 2 * d - 1)
        out["by_depth"][d] = {"two_atom_substructures": tot, "in_library_exact": hit / max(1, tot), "covered_with_anti_unification": au / max(1, tot), "oracle_path_shortening": short / n_eval}
    out["compression"] = lib.compression_gain([W.make_structure(W.SPLIT_BASE["dev"] + 40000 + i * 11 + 4, 4, False) for i in range(n_eval)])
    json.dump(out, open(os.path.join(OUT, "deep_fragments.json"), "w"), indent=1); print(json.dumps(out, indent=1)); return out


def cover_aggregate():
    """Couverture exhaustive (profondeurs 2 et 3) : toutes les formes ajustées. La vraie structure est-elle
    sélectionnée quand la génération est COMPLÈTE ? Et quelle part de l'espace chaque stratégie génère-t-elle ?"""
    by = defaultdict(lambda: {"forms": [], "strategies": {}, "fe": 0.0, "sec": 0.0})
    for f in glob.glob(os.path.join(OUT, "aws", "wce-deepcover-*", "artifacts", "*", "deepcover_*.json")):
        r = json.load(open(f))
        if r["status"] != "ok": continue
        for row in r["result"]:
            k = (r["key"], row["cover_depth"], row["regime"])
            if row["method"] == "exhaustive": by[k]["forms"] += row["forms"]; by[k]["fe"] += row["fit_equivalents"]; by[k]["sec"] += row["seconds"]; by[k]["total"] = row["forms_total"]
            else: by[k]["strategies"][row["method"]] = row
    exact = {1: 8, 2: 108, 3: 2088}; S = {}
    for d in (2, 3):
        for reg in ("standard", "rich"):
            T = [v for (k, dd, rr), v in by.items() if dd == d and rr == reg and len(v["forms"]) == v.get("total")]
            if not T: continue
            sel = rank = within = func = best_full = 0.0; ranks = []
            for v in T:
                fs = sorted(v["forms"], key=lambda f: -f["evidence"]); i = next((j for j, f in enumerate(fs) if f["is_truth"]), None); ranks.append(i)
                sel += i == 0; within += sum(f["evidence"] >= fs[0]["evidence"] - math.log(20.0) for f in fs); func += fs[0]["full"] >= 0.99; best_full += fs[0]["full"]
            n = len(T); strat = {}
            for m in sorted({m for v in T for m in v["strategies"]}):
                xs = [v["strategies"][m] for v in T if m in v["strategies"]]
                strat[m] = {"coverage_depth": _mean(x["forms_generated"].get(str(d), 0) / exact[d] for x in xs), "forms_at_depth": _mean(x["forms_generated"].get(str(d), 0) for x in xs), "truth_generated": _mean(x["truth_generated"] for x in xs)}
            S[f"d{d}/{reg}"] = {"tasks": n, "forms_fitted": T[0]["total"], "truth_selected": sel / n, "truth_rank_median": sorted(r for r in ranks if r is not None)[len(ranks) // 2] if any(r is not None for r in ranks) else None,
                               "truth_in_top5": sum(r is not None and r < 5 for r in ranks) / n, "forms_within_bf20": within / n, "best_is_functional": func / n, "best_full_agreement": best_full / n,
                               "fit_equivalents": _mean(v["fe"] for v in T), "cpu_seconds": _mean(v["sec"] for v in T), "strategies": strat}
    json.dump(S, open(os.path.join(OUT, "deep_cover_summary.json"), "w"), indent=1); print(json.dumps(S, indent=1)); return S


def load_rows():
    rows = []; bad = 0
    for f in glob.glob(os.path.join(OUT, "aws", "wce-deepgen-*", "artifacts", "*", "deepgen_*.json")):
        r = json.load(open(f))
        if r["status"] == "ok" and int(r["key"]) < 100: rows += r["result"]
        elif r["status"] != "ok": bad += 1
    return rows, bad


def _mean(xs): xs = [float(x) for x in xs if x is not None]; return round(sum(xs) / len(xs), 4) if xs else None


def aggregate():
    from core.deep_reachability import RECALLS
    rows, bad = load_rows(); S = {"n_rows": len(rows), "units_in_error": bad, "by": {}, "mechanics": {}, "taxonomy": {}, "costs": {}}
    mech = [r for r in rows if r["method"] == "mechanics"]; runs = [r for r in rows if r["method"] != "mechanics"]
    for reg in sorted({r["regime"] for r in rows}):
        for d in sorted({r["depth"] for r in rows}):
            m = [r for r in mech if r["regime"] == reg and r["depth"] == d]
            if m:
                S["mechanics"][f"{reg}/d{d}"] = {"n": len(m), "identifiable_bound": _mean(r["identifiable_bound"] for r in m), "weakest_atom_effect_points": _mean(r["weakest_atom_effect_points"] for r in m), "expected_flips": _mean(r["expected_flips"] for r in m),
                    "atom_recall_global": _mean(r["atom_recall"]["global_gradient"] for r in m), "atom_recall_global_090": _mean(r["atom_recall"]["global_gradient_090"] for r in m), "atom_recall_separation": _mean(r["atom_recall"]["separation"] for r in m),
                    "atom_recall_separation_090": _mean(r["atom_recall"]["separation_090"] for r in m), "logic_true_atoms": _mean(r["logic_true_atoms"]["generated"] for r in m), "logic_true_atoms_rank0": _mean(r["logic_true_atoms"]["rank"] == 0 for r in m),
                    "logic_with_distractors": _mean(r["logic_true_atoms_plus_distractors"]["generated"] for r in m), "logic_with_distractors_rank0": _mean(r["logic_true_atoms_plus_distractors"]["rank"] == 0 for r in m),
                    "logic_seconds": _mean(r["logic_true_atoms_plus_distractors"]["seconds"] for r in m), "logic_pairs_scored": _mean(r["logic_true_atoms_plus_distractors"]["pairs_scored"] for r in m),
                    "reachable_chain": _mean(r["reachable"]["baseline"] for r in m), "boundary_pairs": _mean(r["boundary_pairs"] for r in m)}
            for name in sorted({f"{r['method']}@{int(r['budget'])}" for r in runs}):
                x = [r for r in runs if r["regime"] == reg and r["depth"] == d and f"{r['method']}@{int(r['budget'])}" == name]
                if not x: continue
                k = f"{reg}/d{d}/{name}"; S["by"][k] = {"n": len(x), **{rc: _mean(r.get(rc) for r in x) for rc in RECALLS}, "functional": _mean(r["functional"] for r in x), "equivalent": _mean(r["equivalent"] for r in x), "full": _mean(r["full"] for r in x)}
                S["taxonomy"][k] = dict(Counter(r.get("cause") for r in x)); S["costs"][k] = {c: _mean(r.get(c) for r in x) for c in ("fit_equivalents", "n_candidates", "n_fitted", "n_structures", "compositions", "seconds", "fit_seconds", "synthesis_seconds")}
    json.dump(S, open(os.path.join(OUT, "deep_generation_summary.json"), "w"), indent=1); print("lignes", len(rows), "unités en erreur", bad); return S


def report():
    S = json.load(open(os.path.join(OUT, "deep_generation_summary.json"))); names = sorted({k.split("/")[2] for k in S["by"]})
    for reg in ("standard", "rich"):
        for metric in ("recall_generated", "recall_fitted", "recall_post_pruning", "recall_top_k", "recall_selected", "functional"):
            print(f"\n{reg} — {metric}"); print("méthode@budget".ljust(22) + " ".join(f"d{d}".rjust(6) for d in range(1, 7)))
            for n in names: print(n.ljust(22) + " ".join((f"{S['by'][f'{reg}/d{d}/{n}'][metric]:.2f}" if f"{reg}/d{d}/{n}" in S["by"] and S["by"][f"{reg}/d{d}/{n}"][metric] is not None else "—").rjust(6) for d in range(1, 7)))
        print(f"\n{reg} — mécanique"); keys = ("identifiable_bound", "weakest_atom_effect_points", "expected_flips", "atom_recall_global", "atom_recall_global_090", "atom_recall_separation", "atom_recall_separation_090", "logic_true_atoms", "logic_with_distractors", "logic_with_distractors_rank0", "logic_seconds", "reachable_chain")
        for k in keys: print(k.ljust(30) + " ".join((f"{S['mechanics'][f'{reg}/d{d}'][k]:.2f}" if f"{reg}/d{d}" in S["mechanics"] and S["mechanics"][f"{reg}/d{d}"][k] is not None else "—").rjust(7) for d in range(1, 7)))
        print(f"\n{reg} — cause principale (profondeurs 3–6 cumulées)")
        for n in names:
            c = Counter()
            for d in range(3, 7): c.update(S["taxonomy"].get(f"{reg}/d{d}/{n}", {}))
            print(n.ljust(22), dict(c))
        print(f"\n{reg} — coûts moyens (toutes profondeurs)")
        for n in names:
            x = [S["costs"][k] for k in S["costs"] if k.startswith(reg + "/") and k.endswith("/" + n)]
            print(n.ljust(22), {c: _mean(v[c] for v in x) for c in ("fit_equivalents", "n_candidates", "compositions", "seconds", "fit_seconds", "synthesis_seconds")})


if __name__ == "__main__":
    {"reachability": reachability, "coverage": coverage, "fragments": fragments, "cover": cover_aggregate, "aggregate": aggregate, "report": report}[sys.argv[1]]()

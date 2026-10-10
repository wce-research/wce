"""DOWNSTREAM diagnostic (Downstream Selection Loss sub-milestone) — EVALUATION module, dev only.

Question: with the TRUE atoms provided, a structure of the right form is often generated, but the final
recovery at the 0.99 threshold collapses. Where, and why, does a good candidate become a bad final solution?

Principle: the pipeline is NOT modified. `TracedSearch` is a subclass of `DeepSearch` that RECORDS each
fit (program before, program after, origin) and the simplification, without changing any computation (tested: same
output as `DeepSearch`). Everything that uses the truth (functional agreements, oracle ranking, cause of the loss) is
computed AFTER the fact, here.

Stages followed:
  true atoms → candidate generation → initial fit → refit / polish → simplification → canonicalization → scoring →
  ranking → final selection
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from core.boundary_program import BNode, predict, render
from core.deep_fit import DT, FitData, approx_evidence, fit_batch, loglik_batch, predict_batch
from core.deep_generation import generate_candidates
from core.deep_search import DeepSearch, profiled_loglik
from core.deep_structure import algebraic_form, description_bits, n_atoms, normalize, skel_key, structurally_equivalent

LOSS_CAUSES = ("INITIAL_FIT_BAD", "POLISH_DEGRADED", "SIMPLIFICATION_DEGRADED", "CANONICALIZATION_CHANGED_BEHAVIOR", "SCORING_MISALIGNED", "RANKING_FAILURE", "FINAL_SELECTION_FAILURE",
               "THRESHOLD_TOO_STRICT", "FUNCTIONAL_EQUIVALENCE_MISCOUNTED", "NUMERICAL_INSTABILITY", "OTHER")
RECOVERED = "RECOVERED"
STAGES = ("true_atoms", "candidate_generation", "initial_fit", "refit_polish", "simplification", "canonicalization", "scoring", "ranking", "final_selection")
THRESHOLDS = (0.90, 0.95, 0.97, 0.99)
GOOD = 0.99


class TracedSearch(DeepSearch):
    """DeepSearch + recording. No computation is modified: the methods call `super()` and note what
    goes in and what comes out."""

    def _setup(self, data, noise): super()._setup(data, noise); self.fits: List[dict] = []; self.simplification: dict = {}; self._pending = None

    def _fit(self, skel, inits, steps, origin, data=None):
        if data is None: self._pending = [skel.skeleton().with_params(r.tolist()) for r in (inits if torch.is_tensor(inits) else torch.stack(list(inits)))]
        return super()._fit(skel, inits, steps, origin, data)

    def _register(self, prog, ll, origin):
        old = self.hyps.get(skel_key(normalize(prog))); h = super()._register(prog, ll, origin); now = self.hyps.get(skel_key(normalize(prog)))
        self.fits.append({"origin": origin, "inits": self._pending or [], "fitted": normalize(prog), "replaced": old is not None and now is not old, "kept_old": old is not None and now is old, "rejected": h is None,
                          "previous": old.program if old is not None else None, "cost": self.counter.fit_equivalents}); self._pending = None; return h

    def _simplify(self):
        before = self._ranked()[0]; after = super()._simplify(); self.simplification = {"before": before.program, "after": after.program, "changed": after.key != before.key, "before_evidence": before.evidence, "after_evidence": after.evidence}
        return after


@dataclass
class BestCandidateTrajectory:
    """Best functional agreement encountered at each stage, corresponding candidate, stage where the global best
    appears and stage where it disappears or degrades."""
    stages: Dict[str, dict] = field(default_factory=dict)

    def add(self, stage, accord, program): self.stages[stage] = {"accord": accord, "program": render(program) if program is not None else None, "form": algebraic_form(program) if program is not None else None}

    def summary(self, good=GOOD):
        order = [s for s in STAGES if s in self.stages]; acc = [self.stages[s]["accord"] for s in order]; best = max(acc); appears = order[acc.index(best)]
        lost = next((order[i] for i in range(acc.index(best) + 1, len(order)) if acc[i] < best - 1e-9), None)
        first_good = next((s for s, a in zip(order, acc) if a >= good), None); lost_good = None
        if first_good is not None: i0 = order.index(first_good); lost_good = next((order[i] for i in range(i0 + 1, len(order)) if acc[i] < good), None)
        return {"best_accord": best, "appears_at": appears, "degrades_at": lost, "good_found_at": first_good, "good_lost_at": lost_good, "destroyed": first_good is not None and acc[-1] < good}


def _spearman(a, b):
    n = len(a)
    if n < 3: return None
    def rk(v):
        o = sorted(range(n), key=lambda i: v[i]); r = [0.0] * n
        for p, i in enumerate(o): r[i] = float(p)
        return r
    ra, rb = rk(a), rk(b); ma, mb = sum(ra) / n, sum(rb) / n; sa = math.sqrt(sum((x - ma) ** 2 for x in ra)); sb = math.sqrt(sum((x - mb) ** 2 for x in rb))
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (sa * sb) if sa > 0 and sb > 0 else None


def score_components(prog: BNode, data: FitData, gamma=0.02) -> dict:
    """Components of the score, separated: profiled likelihood (hard predictions, noise per hypothesis), softened
    robust likelihood (contamination), parameter penalty, structural penalty, MDL in bits, posterior score."""
    pred = predict_batch(prog.skeleton(), torch.tensor([prog.param_vector()], dtype=DT), data.X)[0].to(DT); n = len(data); k = prog.n_params(); d = n_atoms(prog)
    ll = profiled_loglik(pred, data); robust = float(loglik_batch(prog.skeleton(), torch.tensor([prog.param_vector()], dtype=DT), data, gamma)[0])
    pen_k = 0.5 * k * math.log(max(1, n)); pen_s = (2 * d - 1) * 3.0 * math.log(2)
    return {"errors": float((pred != data.y).double().sum()), "likelihood": ll, "robust_likelihood": robust, "complexity_penalty": pen_k, "structural_penalty": pen_s,
            "mdl_bits": description_bits(prog.skeleton()) + 0.5 * k * math.log2(max(1, n)) - ll / math.log(2), "posterior_score": ll - pen_k - pen_s, "depth": d, "n_params": k}


def _state(prog: BNode, data: FitData, accord_fn, truth: BNode, rank=None) -> dict:
    a, ai = accord_fn(prog); sc = score_components(prog, data)
    return {"program": render(prog), "form": algebraic_form(prog), "params": [round(v, 5) for v in prog.param_vector()], "accord": a, "accord_in_region": ai, **sc, "rank": rank,
            "exact": skel_key(prog) == skel_key(truth), "equivalent": structurally_equivalent(prog, truth)}


def fitter_comparison(cands: List[BNode], task, accord_fn, steps=100):
    """Old fitter (`fit_robust`, previous milestone) against new batch fitter, same candidates, same
    initialisation. A small numerical gap does not guarantee the same ranking: we compare both."""
    from core.robust_reconstruction import SENSOR_MODELS, Observations, fit_robust
    obs = Observations.build(task.X, task.y, SENSOR_MODELS["standard"], 0.05); data = FitData.build(task.X, task.y, float(obs.flip[0])); rows = []
    for p in cands:
        sk = p.skeleton(); init = p.param_vector(); po, _ = fit_robust(sk, obs, steps=steps, n_init=0, init=init); th, _ = fit_batch(sk, data, torch.tensor([init], dtype=DT), steps=steps); pn = sk.with_params(th[0].tolist())
        so, sn = score_components(po, data)["posterior_score"], score_components(pn, data)["posterior_score"]
        rows.append({"param_error": max((abs(u - v) for u, v in zip(po.param_vector(), pn.param_vector())), default=0.0), "score_old": so, "score_new": sn, "accord_old": accord_fn(po)[0], "accord_new": accord_fn(pn)[0]})
    if not rows: return {"n": 0}
    oo = sorted(range(len(rows)), key=lambda i: -rows[i]["score_old"]); on = sorted(range(len(rows)), key=lambda i: -rows[i]["score_new"])
    return {"n": len(rows), "max_param_error": max(r["param_error"] for r in rows), "max_score_diff": max(abs(r["score_old"] - r["score_new"]) for r in rows), "max_accord_diff": max(abs(r["accord_old"] - r["accord_new"]) for r in rows),
            "same_top1": oo[0] == on[0], "same_order": oo == on, "rank_correlation": _spearman([r["score_old"] for r in rows], [r["score_new"] for r in rows])}


def trace_task(task, seed=0, fitter_audit=True) -> dict:
    """Complete trace of a task with the TRUE atoms provided as literals (no primitive discovery)."""
    from core.deep_reachability import leaves
    from core.deep_world import full_region, obs_region
    t0 = time.time(); tr = task.world.truth; tl = leaves(tr); d = n_atoms(tr); data = FitData.build(task.X, task.y, 0.10)
    F = full_region(4000, torch.Generator().manual_seed(11)); I = obs_region(2000, torch.Generator().manual_seed(11)); labF, labI = task.world.label(F, 1.0), task.world.label(I, 1.0)
    accord = lambda p: (float((predict(p, F) == labF).float().mean()), float((predict(p, I) == labI).float().mean())); A = lambda p: accord(p)[0]
    traj = BestCandidateTrajectory(); traj.add("true_atoms", 1.0, tr)
    # --- candidate generation (same call as the pipeline at round 0)
    max_depth = min(6, d + 1); gen = generate_candidates(task.X.float(), data.y, max_depth=max_depth, top=4, seed=seed, extra_literals=tl, geometry=False, time_budget=30.0)
    order = sorted(range(len(gen.programs)), key=lambda i: -gen.info[i]["score"]); cands = [gen.programs[i] for i in order]; cand_states = [_state(p, data, accord, tr, rank=r) for r, p in enumerate(cands)]
    bi = max(range(len(cands)), key=lambda i: cand_states[i]["accord"]) if cands else None; traj.add("candidate_generation", cand_states[bi]["accord"] if cands else 0.0, cands[bi] if cands else None)
    # --- traced pipeline (unchanged)
    S = TracedSearch("factorized", budget=150.0, seed=seed, max_depth=max_depth, fact_literals=tl, fact_residual=False); out = S.run(data)
    r0 = [f for f in S.fits if f["origin"] == "factorized_r0"]; later = [f for f in S.fits if f["origin"].startswith("factorized_r") and f["origin"] != "factorized_r0"]
    polish = []                                                               # audit of the polish: before / after, for each fit of a candidate
    for f in r0 + later:
        if not f["inits"]: continue
        b, a = f["inits"][0], f["fitted"]; sb, sa = score_components(b, data), score_components(a, data)
        polish.append({"origin": f["origin"], "accord_before": A(b), "accord_after": A(a), "param_shift": max((abs(u - v) for u, v in zip(b.param_vector(), a.param_vector())), default=0.0),
                       "likelihood_before": sb["likelihood"], "likelihood_after": sa["likelihood"], "errors_before": sb["errors"], "errors_after": sa["errors"], "robust_before": sb["robust_likelihood"], "robust_after": sa["robust_likelihood"],
                       "is_truth_form": algebraic_form(a) == algebraic_form(tr), "replaced_registered": f["replaced"], "kept_old": f["kept_old"]})
    best_r0 = max(r0, key=lambda f: A(f["fitted"]), default=None); traj.add("initial_fit", A(best_r0["fitted"]) if best_r0 else 0.0, best_r0["fitted"] if best_r0 else None)
    allh = list(out.all.values()); pool_best = max(allh, key=lambda h: A(h.program)); traj.add("refit_polish", A(pool_best.program), pool_best.program)
    simp = S.simplification; traj.add("simplification", A(simp["after"]), simp["after"])
    canon_changed = sum(1 for f in S.fits if f["inits"] and not torch.equal(predict(f["fitted"], I), predict(normalize(f["fitted"]), I)))
    ranked = out.hypotheses; ranked_best = max(ranked, key=lambda h: A(h.program)); traj.add("canonicalization", A(ranked_best.program), ranked_best.program)
    by_ev = sorted(allh, key=lambda h: -h.evidence); traj.add("scoring", A(by_ev[0].program), by_ev[0].program)
    dedup_ranked = sorted(ranked, key=lambda h: -h.evidence); traj.add("ranking", A(dedup_ranked[0].program), dedup_ranked[0].program)
    final = out.best().program; traj.add("final_selection", A(final), final)
    # --- oracle ranking against internal scores
    st = [_state(h.program, data, accord, tr) for h in allh]; acc = [s["accord"] for s in st]; oracle = {}
    for name, key, sign in (("posterior_score", "posterior_score", 1), ("likelihood", "likelihood", 1), ("robust_likelihood", "robust_likelihood", 1), ("mdl", "mdl_bits", -1), ("complexity_penalty", "complexity_penalty", -1), ("structural_penalty", "structural_penalty", -1)):
        v = [sign * s[key] for s in st]; o = sorted(range(len(st)), key=lambda i: -v[i]); ob = max(range(len(st)), key=lambda i: acc[i])
        oracle[name] = {"spearman": _spearman(v, acc), "top1_agreement": acc[o[0]] >= acc[ob] - 0.005, "top3_recall": any(acc[i] >= acc[ob] - 0.005 for i in o[:3]), "top5_recall": any(acc[i] >= acc[ob] - 0.005 for i in o[:5]), "top1_accord": acc[o[0]]}
    sel_state = _state(final, data, accord, tr); best_state = _state(pool_best.program, data, accord, tr)
    comp = {k: best_state[k] - sel_state[k] for k in ("likelihood", "robust_likelihood", "complexity_penalty", "structural_penalty", "posterior_score", "errors")}
    # --- estimation bound: the TRUE structure, fitted from the TRUE parameters, on these data
    th, _ = fit_batch(tr.skeleton(), data, torch.tensor([normalize(tr).param_vector()], dtype=DT), steps=100); tfit = normalize(tr).skeleton().with_params(th[0].tolist()); st_true = score_components(normalize(tr), data); st_tfit = score_components(tfit, data)
    truth_fit = {"accord": A(tfit), "likelihood_true_params": st_true["likelihood"], "likelihood_fitted": st_tfit["likelihood"], "errors_true_params": st_true["errors"], "errors_fitted": st_tfit["errors"], "fit_improved_likelihood": st_tfit["likelihood"] >= st_true["likelihood"]}
    # --- exploratory safeguards (counterfactuals computed on the recorded states; the pipeline is not modified)
    pool = {}                                                                 # best version ON THE DATA of each structure, initial version included
    for f in S.fits:
        for p in f["inits"][:1] + [f["fitted"]]:
            k = skel_key(p); s = score_components(p, data)
            if k not in pool or s["likelihood"] > pool[k][1]["likelihood"]: pool[k] = (p, s)
    hb = max(pool.values(), key=lambda v: v[1]["posterior_score"])[0] if pool else final
    counterfactual = {"hold_best": A(hb), "no_simplification": A(simp["before"]), "no_dedup": A(by_ev[0].program), "pre_fit_top_score": cand_states[0]["accord"] if cands else 0.0,
                      "best_available": A(pool_best.program), "hold_best_no_simplification_note": "hold_best = argmax du score sur les meilleures versions (sur les données) de chaque structure, sans simplification"}
    # --- cause of the loss
    num_bad = any(not math.isfinite(v) for h in allh for v in h.program.param_vector()) or any(not math.isfinite(h.evidence) for h in allh)
    a_final = A(final); pre_good = [i for i, s in enumerate(cand_states) if s["accord"] >= GOOD]; polished_keys = {skel_key(f["fitted"]) for f in r0}
    good_polished = [f for f in r0 if f["inits"] and A(f["inits"][0]) >= GOOD]
    if a_final >= GOOD: cause = RECOVERED
    elif num_bad: cause = "NUMERICAL_INSTABILITY"
    elif not pre_good: cause = "OTHER"                                        # no candidate ≥ 0.99 before fit: outside the downstream scope
    elif not good_polished: cause = "SCORING_MISALIGNED"                      # the pre-fit screen did not keep the good candidate for the polish
    elif A(pool_best.program) < GOOD:                                         # no candidate ≥ 0.99 left after the fits
        if all(A(f["fitted"]) < GOOD for f in good_polished) and not any(A(f["fitted"]) >= GOOD for f in S.fits):
            improved = all(score_components(f["fitted"], data)["likelihood"] >= score_components(f["inits"][0], data)["likelihood"] - 1e-9 for f in good_polished)
            cause = "THRESHOLD_TOO_STRICT" if improved and a_final >= 0.97 else "INITIAL_FIT_BAD"
        else: cause = "POLISH_DEGRADED"
    elif A(ranked_best.program) < GOOD: cause = "CANONICALIZATION_CHANGED_BEHAVIOR"       # present among the fitted structures, absent after deduplication
    elif A(simp["before"]) >= GOOD and a_final < GOOD: cause = "SIMPLIFICATION_DEGRADED"
    elif A(dedup_ranked[0].program) < GOOD:
        gb = _state(ranked_best.program, data, accord, tr); tp = _state(dedup_ranked[0].program, data, accord, tr)
        cause = "SCORING_MISALIGNED" if gb["likelihood"] < tp["likelihood"] else "RANKING_FAILURE"      # the data prefer the bad one / the penalties make the good one lose
    else: cause = "FINAL_SELECTION_FAILURE"
    miscounted = a_final >= GOOD and not sel_state["equivalent"]
    return {"cause": cause, "final_accord": a_final, "final_in_region": accord(final)[1], "final_exact": sel_state["exact"], "final_equivalent": sel_state["equivalent"], "functional_not_equivalent": miscounted,
            "thresholds": {str(t): a_final >= t for t in THRESHOLDS}, "trajectory": traj.stages, "trajectory_summary": traj.summary(), "candidates": cand_states[:12], "n_candidates": len(cands), "n_good_candidates": len(pre_good),
            "good_candidate_rank": pre_good[0] if pre_good else None, "n_polished": len(r0), "polish": polish, "simplification": {"changed": simp["changed"], "accord_before": A(simp["before"]), "accord_after": A(simp["after"]),
            "atoms_before": n_atoms(simp["before"]), "atoms_after": n_atoms(simp["after"]), "evidence_before": simp["before_evidence"], "evidence_after": simp["after_evidence"],
            "removed_needed_atom": simp["changed"] and A(simp["after"]) < A(simp["before"]) - 0.005, "params_modified": simp["changed"]},
            "canonicalization_changed_behavior": canon_changed, "dedup_removed": len(allh) - len(ranked), "oracle_ranking": oracle, "best_minus_selected": comp, "selected": {k: sel_state[k] for k in ("program", "accord", "likelihood", "posterior_score", "depth", "n_params", "errors")},
            "best_available": {k: best_state[k] for k in ("program", "accord", "likelihood", "posterior_score", "depth", "n_params", "errors")}, "truth_fit": truth_fit, "counterfactual": counterfactual,
            "confidence": float(out.best().mass), "n_hypotheses": len(allh), "fit_equivalents": out.stats["fit_equivalents"], "fitter": fitter_comparison(cands[:8], task, accord) if fitter_audit else {}, "seconds": round(time.time() - t0, 2)}

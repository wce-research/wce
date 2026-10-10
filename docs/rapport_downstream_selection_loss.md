# Diagnostic report — WCE Downstream Selection Loss

EXPLORATORY sub-milestone of the milestone "Deep Compositional Structure Discovery & Model-Class Boundary".
Data: **dev partition only**, 128 tasks (depths 2 to 5 × standard and rich regimes × 16 tasks), **true
atoms supplied** as literals: primitive discovery is entirely neutralised. Neither the validation set, nor the
confirmatory test, nor the sealed set was touched. Everything is `POST-HOC / EXPLORATORY`.

Source: `results/deep_downstream_summary.json` (AWS run `wce-deepdown-20261004-153523`, no unit in error).
Targets: `make deep-run-down`, `make deep-down-aggregate`, `make deep-down-report`.

Method: the pipeline was NOT modified. An instrumentation subclass (`TracedSearch`) records each
fit (program before, program after) and the simplification by calling the existing code; tests verify
that it produces exactly the same selection, the same scores and the same cost. Everything that uses the truth is
computed after the fact.

## Summary

A nearly perfect solution is found in 99 % of the tasks, then **destroyed by the first gradient fit**.

| | Share of tasks with a candidate of agreement ≥ 0.99 |
|---|---|
| After generation (before any fit) | 99 % |
| After the initial fit | 13 % |
| At final selection | 9 % |

Main cause of the 116 failures: `INITIAL_FIT_BAD` in 102 cases (88 %). Simplification, ranking, score and
canonicalisation are marginal causes (5 cases between them). The problem is one of **optimisation**, not of selection
nor of metric.

## 1. Is a good solution found before the end?

Yes, almost always. Synthesis on signatures, with the true atoms, produces a candidate of agreement ≥ 0.99 in
127 tasks out of 128 (99.2 %). In 90 % of these cases, this candidate is also the first by score before fitting. The
candidate ranked first before fitting has a mean agreement of 0.9956 and reaches 0.99 in 89 % of the tasks.

The system therefore already has the right solution in hand, with the right parameters, before it starts fitting.

## 2. At which stage is it lost?

At the initial fit. Share of tasks where a candidate of agreement ≥ 0.99 exists, stage by stage:

| Cell | True atoms | Generation | Initial fit | Refit | Simplification | Canonicalisation / deduplication | Score | Ranking | Final selection |
|---|---|---|---|---|---|---|---|---|---|
| Standard d2 | 1.00 | 1.00 | 0.12 | 0.12 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 |
| Standard d3 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Standard d4 | 1.00 | 0.94 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Standard d5 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Rich d2 | 1.00 | 1.00 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| Rich d3 | 1.00 | 1.00 | 0.38 | 0.38 | 0.19 | 0.31 | 0.19 | 0.19 | 0.19 |
| Rich d4 | 1.00 | 1.00 | 0.06 | 0.06 | 0.00 | 0.06 | 0.00 | 0.00 | 0.00 |
| Rich d5 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Stage at which the good solution is lost (`BestCandidateTrajectory`), over the 128 tasks: initial fit 110,
simplification 5, generation 1, never lost 12. The columns "refit", "canonicalisation" measure the
best candidate still available; the columns "simplification", "score", "ranking" and "final selection"
measure the candidate retained at that stage.

Best mean agreement per stage: 1.000 at generation; 0.933 to 0.986 after the initial fit; 0.898 to 0.982 at
final selection.

## 3. Which stage contributes most to the degradation?

The gradient fit (polishing) of the candidates.

| Main cause | Standard | Rich | Total |
|---|---|---|---|
| `INITIAL_FIT_BAD` | 55 | 47 | 102 |
| Recovered (final agreement ≥ 0.99) | 1 | 11 | 12 |
| `THRESHOLD_TOO_STRICT` | 6 | 2 | 8 |
| `SCORING_MISALIGNED` | 0 | 2 | 2 |
| `CANONICALIZATION_CHANGED_BEHAVIOR` (deduplication) | 1 | 1 | 2 |
| `RANKING_FAILURE` | 0 | 1 | 1 |
| `OTHER` (no candidate ≥ 0.99 generated) | 1 | 0 | 1 |
| `POLISH_DEGRADED`, `SIMPLIFICATION_DEGRADED`, `FINAL_SELECTION_FAILURE`, `NUMERICAL_INSTABILITY`, `FUNCTIONAL_EQUIVALENCE_MISCOUNTED` | 0 | 0 | 0 |

By depth, `INITIAL_FIT_BAD`: standard 12, 12, 15, 16 out of 16 (d2 to d5); rich 7, 9, 15, 16 out of 16.

Definitions used: `INITIAL_FIT_BAD` = a candidate ≥ 0.99 was submitted to the fit and no candidate ≥ 0.99
exists any longer afterwards; `THRESHOLD_TOO_STRICT` = same situation, but the fit improved the likelihood on the data
and the final agreement stays ≥ 0.97 (the loss is statistical, not an optimisation defect). `SIMPLIFICATION_DEGRADED`
never appears as a main cause because, when the simplification degrades, the candidate had already been damaged
upstream.

## 4. Is the internal score misaligned with functional quality?

Moderately, and it is not the dominant cause. Oracle ranking (by real functional agreement) against internal scores,
over all the fitted structures of each task:

| Score | Rank correlation | Agreement of the first (top-1) | Best within the first 3 | … within the first 5 | Agreement of the candidate ranked first |
|---|---|---|---|---|---|
| Posterior score (current) | 0.78 | 0.64 | 0.79 | 0.85 | 0.953 |
| Profile likelihood | 0.86 | 0.51 | 0.85 | 0.95 | 0.956 |
| Softened robust likelihood | 0.85 | 0.36 | 0.78 | 0.90 | 0.955 |
| MDL | 0.77 | 0.64 | 0.79 | 0.84 | 0.953 |
| Parameter penalty alone | −0.78 | 0.01 | 0.01 | 0.02 | 0.717 |
| Structure penalty alone | −0.74 | 0.01 | 0.01 | 0.02 | 0.717 |

Reading: the likelihood is the component best correlated with real quality; the complexity penalties go
against quality (the good structures are the largest ones). The composite score chooses the best available
candidate in 64 % of the tasks. On average over the 128 tasks (zero gap when it chooses it), the best available one has a likelihood
higher by 1.7 nat but pays 7.7 nats more of parameter penalty and 3.3 more of structure penalty: it is the
complexity penalty that loses the good structures, not the likelihood.

But the stakes are low here: systematically choosing the best available candidate after fitting would give only
13 % recovery (against 9 %), because the good candidates have already been damaged.

### Calibration of the final score

The internal confidence (mass of the retained structure) is very poorly calibrated: mean confidence 0.905 for a real
recovery rate of 0.094.

| Confidence bin | Tasks | Mean confidence | Recovery ≥ 0.99 | Recovery ≥ 0.97 |
|---|---|---|---|---|
| 0.2 – 0.4 | 1 | 0.26 | 0.00 | 0.00 |
| 0.4 – 0.6 | 12 | 0.49 | 0.00 | 0.25 |
| 0.6 – 0.8 | 8 | 0.68 | 0.13 | 0.13 |
| 0.8 – 1.0 | 107 | 0.98 | 0.10 | 0.56 |

ECE 0.81 and Brier 0.76 at threshold 0.99; ECE 0.41 and Brier 0.39 at threshold 0.97.

## 5. Does the 0.99 threshold explain a large share of the problem?

A share, not the bulk of it. Share of tasks whose final agreement reaches each threshold:

| Cell | ≥ 0.90 | ≥ 0.95 | ≥ 0.97 | ≥ 0.99 | Exact structure | Equivalent structure | Mean final agreement |
|---|---|---|---|---|---|---|---|
| Standard d2 | 1.00 | 0.81 | 0.63 | 0.06 | 0.81 | 0.81 | 0.966 |
| Standard d3 | 0.88 | 0.81 | 0.50 | 0.00 | 0.25 | 0.56 | 0.960 |
| Standard d4 | 0.75 | 0.44 | 0.13 | 0.00 | 0.31 | 0.38 | 0.923 |
| Standard d5 | 0.56 | 0.00 | 0.00 | 0.00 | 0.06 | 0.13 | 0.898 |
| Rich d2 | 1.00 | 0.94 | 0.88 | 0.50 | 0.63 | 0.63 | 0.982 |
| Rich d3 | 1.00 | 1.00 | 0.81 | 0.19 | 0.56 | 0.88 | 0.982 |
| Rich d4 | 1.00 | 0.81 | 0.75 | 0.00 | 0.50 | 0.50 | 0.968 |
| Rich d5 | 0.81 | 0.69 | 0.31 | 0.00 | 0.06 | 0.19 | 0.945 |
| **All** | 0.88 | 0.69 | 0.50 | 0.09 | | | |

41 % of the tasks lie between 0.97 and 0.99: the failure at the 0.99 threshold is partly a threshold effect. But half
of the tasks remain below 0.97, and at depth 5 in the standard regime none reaches 0.95. The failure is real.

Nor is the threshold unreachable: the candidate available BEFORE fitting reaches 0.99 in 89 % of the tasks.
Only 8 tasks out of 128 fall under `THRESHOLD_TOO_STRICT`.

Equivalent representations: a retained structure that is functionally correct (≥ 0.99) but of a form not equivalent to
the true one appears in 2.3 % of the tasks; it is counted as a functional success, not as a failure. Conversely,
the exact structure is often retained with damaged parameters (81 % exact structures in standard d2 for 6 %
recovery at 0.99).

## 6. Is the simplification safe?

Almost, and it rarely acts. It modifies the retained structure in 3 tasks out of 128 (2.3 %). In 2 of these 3 cases
it removes a necessary atom (agreement −0.050 and −0.007); in the third it slightly improves the agreement
(+0.005). It never merges two structures: it replaces a structure by one of its children, refitted
(60 steps), so it modifies the parameters when it acts. The rule (Bayes factor < 3) is therefore too permissive
in about 1.6 % of the tasks. It is the main cause of no failure.

## 7. Is the polishing safe?

**No.** Over the 713 fits of candidates whose agreement before fitting was ≥ 0.97:

| | |
|---|---|
| Mean agreement before → after | 0.994 → 0.956 |
| Degraded candidates (agreement −0.002 or more) | 95 % |
| Perfect candidates (agreement ≥ 0.999) falling below 0.99 | 93 % (379 out of 408) |
| Mean drop in agreement | 0.038 |
| Mean displacement of the most modified parameter | 0.67 |
| Fits that LOWER the likelihood on the data | 55 % |
| Mean change in the number of errors on the data | +1.35 |

The polishing therefore degrades silently, and in more than half of the cases the result is objectively worse on
the available data than the starting point — with nothing opposing it, since the initial version is never
recorded.

By cell, share of candidates ≥ 0.97 made less likely: standard 49 %, 28 %, 36 %, 42 % (d2 to d5); rich
74 %, 48 %, 72 %, 85 %. Mean drop in agreement: 0.031, 0.028, 0.064, 0.066 (standard); 0.018, 0.015, 0.039, 0.038
(rich).

Estimation bound: the TRUE structure, fitted from its TRUE parameters, ends with a mean agreement of 0.774 and
goes from 39 to 118 errors on the data on average; it improves its likelihood in only 16 % of the tasks. This
is not a problem of insufficient data: the optimiser leaves an optimum for a markedly worse point.
Most probable explanation, not tested in isolation: the fit starts with a very softened boundary (stiffness 4)
and a step size of 0.05, which moves already correct parameters away before the stiffness rises again.

## 8. Does the new fitter change the ranking?

Sometimes, and the announced numerical equivalence does not hold in general. Old fitter (`fit_robust`) against new
batched fitter, same candidates, same initialisation, 100 steps:

| | |
|---|---|
| Tasks where the parameter gap exceeds 1e-3 | 28 % |
| Maximum parameter gap | 1.01 |
| Maximum score gap | 24.5 nats |
| Maximum functional agreement gap | 0.035 |
| Same candidate ranked first | 98.4 % |
| Same complete order | 86.7 % |
| Mean rank correlation | 0.985 |

The gap of 1e-7 measured at the start of the milestone came from short, well-conditioned cases. On trajectories of 100 steps
starting from composite candidates, a rounding difference in the initialisation (the old fitter goes through
single-precision floats) is amplified: the optimisation is numerically sensitive. Consequence for the preregistration:
hypothesis H10(b) (gap ≤ 1e-6) does not hold as it is formulated on these tasks. This is not the cause of the
downstream loss — the old fitter degrades just as much — but it is a symptom of the same defect: the fit is unstable.

## 9. Is the problem mainly optimisation, selection or metric?

**Optimisation.**

| Family | Tasks | Share of the 116 failures |
|---|---|---|
| Optimisation (`INITIAL_FIT_BAD`) | 102 | 88 % |
| Metric (`THRESHOLD_TOO_STRICT`) | 8 | 7 % |
| Selection (score, ranking, deduplication) | 5 | 4 % |
| Generation (`OTHER`) | 1 | 1 % |

Counterfactuals computed on the recorded states, without modifying the pipeline:

| Scenario | ≥ 0.99 | ≥ 0.97 | Mean agreement |
|---|---|---|---|
| Current pipeline | 9 % | 50 % | — |
| Without simplification | 9 % | 50 % | 0.953 |
| Without deduplication | 9 % | 50 % | 0.953 |
| Best candidate available after fitting (oracle selection) | 13 % | 60 % | 0.968 |
| "hold-best" safeguard (see section 10) | 57 % | 72 % | 0.975 |
| Candidate ranked first BEFORE fitting, without fitting | 89 % | 96 % | 0.996 |

Improving the selection cannot gain more than 4 points; not degrading at the fit gains 48 to 80.

## 10. Which minimal correction is justified by the data?

**A safeguard at the fitting stage, and nothing else**: never replace a candidate by its fitted version if
the latter is less likely on the available data.

- It touches only one stage (the recording of the result of a fit). Score, ranking, simplification,
  canonicalisation and acceptance criteria remain unchanged.
- It uses no truth: the comparison is made on the profile likelihood of the observed data.
- Counterfactual estimate (exploratory, computed on the recorded states): recovery at 0.99 from 9 % → 57 %;
  at 0.97 from 50 % → 72 %. By cell: standard 56 %, 44 %, 31 %, 13 % (d2 to d5); rich 81 %, 69 %, 88 %, 75 %.
- It is NOT integrated into the pipeline: only the counterfactual was computed.

This safeguard does not recover everything (57 % against 89 % without fitting): when the fitted version is slightly more
likely but functionally worse, it keeps it. This residue comes from overfitting to the noise, not from
the optimiser.

### Proposed controlled experiment

1. A single change: record the initial version of each candidate and replace it only if the fit
   increases the profile likelihood. No other stage modified.
2. Two arms, same tasks, same seeds: current pipeline against pipeline + safeguard.
3. Tasks: the 128 of this diagnostic (to verify that the counterfactual materialises) and 128 new dev tasks
   never seen (so as not to judge on the tasks that motivated the correction).
4. Measures: recovery at 0.90 / 0.95 / 0.97 / 0.99, mean agreement, share of fits accepted, cost.
5. Criterion announced in advance: the safeguard is retained if the recovery at 0.99 on the new tasks exceeds that of the
   current pipeline by at least 20 points, without a drop in mean agreement.

A second possible correction — changing the optimiser schedule (initial stiffness, step size) so that it no longer leaves a
good point — touches another mechanism. It must be tested only afterwards, and separately, so as not to confound the two
effects.

## Limits

- 16 tasks per cell: the per-cell proportions have a resolution of 1/16.
- The cause "initial stiffness too low" of the polishing is a plausible explanation, not a result: it has not
  been isolated experimentally.
- The "hold-best" counterfactual is computed a posteriori; its real effect in the loop (the refit rounds
  reinject the polished atoms) may differ.
- The "canonicalisation" column mixes key normalisation (which never changes the behaviour: 0 cases) and
  functional deduplication (which discards on average 3.4 structures per task and costs the good candidate in 2 tasks).
- Diagnostic conducted with the true atoms: it says nothing about the complete chain with discovered atoms.
- Nothing here is confirmatory.

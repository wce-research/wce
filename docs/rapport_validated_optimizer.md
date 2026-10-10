# Report — WCE Validated Optimizer

EXPLORATORY sub-milestone of the milestone "Deep Compositional Structure Discovery & Model-Class Boundary".
Data: **dev split only**, **true atoms provided**. Neither the validation split, nor the confirmatory test, nor the
sealed set was touched. Everything is `POST-HOC / EXPLORATORY`.

The diagnosis that motivates this sub-milestone (`docs/rapport_downstream_selection_loss.md`) was committed BEFORE the fix
(commit `3074abe`) and has not been rewritten.

Sources: `results/deep_validated_summary.json`; AWS runs `wce-deepvo-20261004-162019` (two-arm experiment, 768
sub-units) and `wce-deepopt-20261004-163055` (optimizer diagnosis, 64 tasks), no unit in error.
Targets: `make deep-run-arms`, `make deep-run-optimizer`, `make deep-vo-aggregate`, `make deep-vo-report`.

Clarification on sample sizes: each set of "128 tasks" contains 64 distinct structures, each observed in
two regimes (standard and rich). The two sets have no structure in common.

## Verdict

| Question | Answer |
|---|---|
| Bad optimizer (mechanism)? | Secondary. Adam's step size ejects a good point in 5 steps, but reducing the step size or the softness does not change the end point. |
| Bad surrogate objective? | **Yes, main cause.** The continuous objective improves in 94% of cases while the real agreement degrades in 100%. |
| Numerical instability? | Present, not causal. Both fitters and both precisions reach the same outcome. |
| Bad acceptance criterion? | **Yes, second factor.** Without a criterion: 7% recovery; surrogate criterion: 16%; profiled likelihood: 54%. |

The guardrail generalises and the criterion fixed in advance is met. It does not recover everything: not fitting at all would do
better on these tasks where the starting parameters are already correct.

## The fix (a single stage)

`core/deep_fit.py`: `validated_fit` — initial state → optimizer proposal → evaluation → commit or rollback.

- **Scope**: only the fit of an EXISTING candidate (one initialisation, which is the candidate's state). Fits
  with several random initialisations (atoms) have no initial state to protect and are not
  concerned. Generator, simplification, ranking, MDL, metric, primitive discovery and measurement policy are
  unchanged; the optimizer itself is unchanged.
- **Criterion**: profiled likelihood of the hard predictions on the observations — the one the system then uses
  to record and rank. No truth.
- **Tolerance**: 1e-9 nat.
- **Tie** (gap ≤ tolerance): rollback — without evidence of improvement, the initial state is kept.
- **NaN / inf** in the proposed parameters or in their criterion: rollback. If the criterion of the initial state is non-finite
  and the proposal is finite: commit.
- **Rollback**: the initial state is a distinct tensor, never modified in place (checked by an assertion); rolling
  back consists in returning it as is, not in rebuilding it.
- The budget is debited even in case of rollback (the computation took place).

## 1. Does the guardrail generalise to the 128 new tasks?

Yes. Criterion fixed before the result: at least +20 points of recovery at threshold 0.99, with no drop in mean agreement.

| | Arm A (frozen pipeline) | Arm B (guardrail) | Paired difference | 95% CI | Effect size |
|---|---|---|---|---|---|
| Recovery ≥ 0.99 | 7.0% | 53.9% | **+46.9 points** | [+38.2, +55.6] | 0.94 |
| Mean functional agreement | 0.958 | 0.972 | +0.013 | [+0.010, +0.017] | 0.61 |

Criterion met. On the 128 tasks of the diagnosis: 9.4% → 59.4% (+50.0 points, CI [41.3, 58.7]), very close to the
announced counterfactual (57%).

## 2. How much recovery is actually restored?

New tasks:

| Threshold | Arm A | Arm B | Difference | 95% CI |
|---|---|---|---|---|
| ≥ 0.90 | 94.5% | 94.5% | 0.0 | [−2.2, +2.2] |
| ≥ 0.95 | 79.7% | 82.0% | +2.3 | [−1.7, +6.4] |
| ≥ 0.97 | 53.9% | 69.5% | +15.6 | [+9.3, +21.9] |
| ≥ 0.99 | 7.0% | 53.9% | +46.9 | [+38.2, +55.6] |

| New tasks | Arm A | Arm B |
|---|---|---|
| Profiled likelihood of the retained solution | −131.8 | −129.0 |
| Errors on the observations | 39.6 | 38.4 |
| Agreement in the observed zone | 0.959 | 0.975 |
| Exact / equivalent structure | 39% / 51% | 45% / 57% |
| Number of atoms retained | 3.05 | 3.05 |
| Fits accepted / rolled back per task | — | 6.1 / 5.3 |
| Fit-equivalents | 28.1 | 27.2 |

The gain is concentrated at the top of the scale: the guardrail changes nothing at thresholds 0.90 and 0.95, it turns
"almost good" solutions into exact solutions. Cross table at threshold 0.99: 60 tasks recovered by B only, 9 by
both, 59 by neither, **0 by A only**. Nearly half of the fits (46%) are rolled back.

Ceiling: without any fit, the candidate ranked first before fitting reaches 0.99 in 86 to 89% of the tasks. The
guardrail therefore restores about 60% of what is restorable.

## 3. Does it degrade some tasks?

Yes, rarely. On the new tasks: agreement improved in 59.4% of the tasks, unchanged in 35.9%, **degraded in
4.7%** (6 tasks). On the tasks of the diagnosis: 61.7% / 35.9% / 2.3%.

Worst cases: 0.964 → 0.854 and 0.900 → 0.872 (new tasks, standard, depth 4); 0.951 → 0.841 (diagnosis,
standard, depth 4). None of these tasks was above 0.99 in arm A: the guardrail caused no
loss of recovery. The mechanism has not been analysed in detail; the most probable explanation is that keeping a
different candidate changes the atoms reinjected in the following rounds, and therefore the structure finally ranked first.

## 4. Is the benefit stable across depth and regime?

Across depth, yes. Across regime, no.

New tasks, recovery ≥ 0.99:

| | Arm A | Arm B | Difference | 95% CI | Δ mean agreement | Improved / degraded |
|---|---|---|---|---|---|---|
| d2 | 25.0% | 81.3% | +56.3 | [+38.4, +74.1] | +0.013 | 78% / 0% |
| d3 | 0.0% | 62.5% | +62.5 | [+45.1, +79.9] | +0.018 | 63% / 0% |
| d4 | 3.1% | 40.6% | +37.5 | [+20.1, +54.9] | +0.007 | 47% / 13% |
| d5 | 0.0% | 31.3% | +31.3 | [+14.6, +47.9] | +0.016 | 50% / 6% |
| Standard | 1.6% | 37.5% | +35.9 | [+24.0, +47.9] | +0.011 | 47% / 8% |
| Rich | 12.5% | 70.3% | +57.8 | [+45.5, +70.1] | +0.016 | 72% / 2% |

Per cell (16 tasks each):

| New tasks | d2 | d3 | d4 | d5 |
|---|---|---|---|---|
| Standard, A → B | 6% → 75% | 0% → 56% | 0% → 13% | 0% → 6% |
| Rich, A → B | 44% → 88% | 0% → 69% | 6% → 69% | 0% → 56% |

| Tasks of the diagnosis | d2 | d3 | d4 | d5 |
|---|---|---|---|---|
| Standard, A → B | 6% → 56% | 0% → 44% | 0% → 31% | 0% → 13% |
| Rich, A → B | 50% → 81% | 19% → 75% | 0% → 88% | 0% → 88% |

The guardrail also preserves deep structures in the rich regime (56 to 88% at depth 5). In the standard regime,
the benefit vanishes at depths 4 and 5 on the new tasks: +12.5 points [−5.7, +30.7] and +6.3 points
[−7.1, +19.6], intervals including zero, and a mean agreement that does not progress at depth 4 (−0.003
[−0.021, +0.016]). This is also where the degradations are found.

## 5. Why does the optimizer move solutions that are already good?

Diagnosis on 64 tasks, two near-perfect starting points: the true structure at the true parameters (agreement 1.000)
and the candidate ranked first before fitting (agreement 0.996). The optimizer was not modified; the variants
below are measurements. The trace reproduces `fit_batch` exactly (zero gap).

### Mean trajectory from the true parameters

| Step | Steepness | Real agreement | Errors on the data | Continuous objective (robust likelihood) | Profiled likelihood | Gradient norm | Max displacement |
|---|---|---|---|---|---|---|---|
| 0 | 4 | 1.000 | 37.9 | −137.0 | −128.7 | 40.1 | 0.00 |
| 5 | 5 | 0.926 | 61.3 | −165.7 | −176.8 | 25.0 | 0.22 |
| 10 | 6 | 0.937 | 54.4 | −154.4 | −163.4 | 19.8 | 0.33 |
| 20 | 8 | 0.952 | 47.8 | −142.5 | −150.3 | 17.0 | 0.39 |
| 40 | 12 | 0.960 | 43.2 | −136.5 | −140.6 | 9.9 | 0.47 |
| 60 | 16 | 0.963 | 41.4 | −134.9 | −136.7 | 6.7 | 0.52 |
| 80 | 20 | 0.963 | 40.3 | −134.0 | −134.1 | 11.5 | 0.59 |
| 100 | 24 | 0.964 | 39.8 | −133.8 | −132.8 | — | 0.61 |

The solution degrades **within the first 5 steps, in 100% of the tasks**. The true parameters are not a stationary
point of the continuous objective (gradient norm 40 at steepness 4, 80 at steepness 24); yet Adam advances by about
one learning rate per iteration whatever the size of the gradient: in 5 steps, each parameter has moved
by about 0.22. The optimizer then climbs back towards an optimum of the continuous objective that is **better than the
starting point for this objective** (−133.8 versus −137.0) but **worse for the world** (agreement 0.964) and worse on the
discrete data (39.8 errors versus 37.9).

### Hypotheses A to J

| Hypothesis | Measurement | Conclusion |
|---|---|---|
| A. Boundary initially too softened | constant steepness 24: final agreement 0.973 (versus 0.964), recovery 12.5% (versus 12.5%) | weak effect; does not explain the loss |
| B. Learning rate too high | learning rate 0.02: 0.969; learning rate 0.005: 0.974; recovery 12.5% | weak effect; reduces the displacement (0.61 → 0.16), not the degradation |
| A + B | steepness 24 and learning rate 0.005: agreement 0.975, recovery 12.5% | the two settings together restore nothing |
| C. Poor conditioning | median conditioning of the Hessian at steepness 24: 151; greater than 10⁴ in 9% of the tasks | moderate; not the general cause |
| D. Gradient aligned with an approximation | continuous objective improved in 94% of cases, agreement degraded in 100%; non-zero gradient at the true parameters | **supported** |
| E. Saturation | 18% of the points saturated at steepness 4, 75% at steepness 24 | measured; no isolated effect of its own |
| F. Non-identifiable parameters | flat direction in 8 to 13% of the tasks; negative curvature at the true parameters in 48% | contributes: the truth is not a local minimum of the objective in half of the tasks |
| G. Interaction between atoms | 1.5 to 3.4% of the mass of the Hessian lies outside the per-atom blocks | weak; rejected |
| H. Bad steepness schedule | reversed schedule: 0.936; constant ≈ increasing | the current schedule is not the problem |
| I. Continuous optimum ≠ functional optimum | best state according to the continuous objective: agreement 0.968; the objective rises and the profiled likelihood falls in 58% of cases | **supported — main cause** |
| J. Numerical instability | single versus double precision: agreement differing by more than 0.002 in 1.6% of the tasks (maximum 0.004) | present, negligible |

All the variants of the optimizer degrade the agreement in 100% of the tasks, except monotone descent (section 8).

## 6. Is the continuous objective misaligned with discrete quality?

Yes. From the true parameters, default configuration:

| | |
|---|---|
| The continuous objective improves | 94% of the tasks |
| The real agreement degrades | 100% |
| The continuous objective improves AND the agreement degrades | 94% |
| The continuous objective improves AND the profiled (discrete) likelihood degrades | 58% |
| The profiled likelihood improves | 23% |
| Rank correlation, along the trajectory, between continuous objective and agreement | 0.51 on average; negative in 16% of the tasks |
| … between profiled likelihood and agreement | 0.57 |
| … between the optimized loss (at current steepness) and agreement | 0.37 |

The mean correlation is positive because the two quantities fall together at the beginning and then rise together; what
diverges is the end point. This is a surrogate-objective problem: the optimizer does what it is
asked, and what it is asked is not functional quality.

The discrete criterion itself is not perfectly aligned: when the profiled likelihood improves (23% of cases),
the agreement drops anyway. With 300 to 600 noisy observations, a state can explain the data better than the truth
while being functionally worse. This is what limits the guardrail.

## 7. Do the old and the new fitter have scientifically different behaviours?

Numerically yes, scientifically no.

| Same starting point, 100 steps | True parameters (64 tasks) | First candidate (64 tasks) | Candidates of the downstream diagnosis (128 tasks) |
|---|---|---|---|
| Parameter gap > 1e-6 | 33% | 36% | — |
| Parameter gap > 1e-3 | 4.7% | 1.6% | 28% |
| Maximum parameter gap | 0.13 | 0.0014 | 1.01 |
| Maximum agreement gap | 0.0075 | 0.0005 | 0.035 |
| Mean agreement, old / new | 0.9640 / 0.9639 | 0.9629 / 0.9629 | — |
| Errors on the data, old / new | 39.66 / 39.75 | 40.31 / 40.31 | — |
| Same outcome at threshold 0.99 | 100% | 100% | — |
| Same first-ranked / same complete order | — | — | 98.4% / 86.7% |
| Time per isolated fit, old / new | 0.200 s / 0.181 s | 0.192 s / 0.176 s | — |

The origin of the gap is the rounding of the initialisation to single precision in the old fitter (≈ 1e-7), amplified by
100 optimization steps. The two fitters degrade as much and reach the same outcome.

**H10(b) (gap ≤ 1e-6) is refuted on the composites.** The batch fitter is no longer described as numerically
equivalent: it is another implementation, with a measured trade-off — about 7 times faster per batch of 8
initialisations, 1.1 times for an isolated fit, same first-ranked in 98% of the tasks, same complete order in
87%. The documentation of `core/deep_fit.py` has been corrected accordingly.

## 8. Is a best-so-far or an early stopping sufficient?

No, not better than not optimizing.

| Start at the true parameters | Mean agreement | Recovery ≥ 0.99 |
|---|---|---|
| Without optimizing ("no-op" baseline) | 1.000 | 100% |
| Final state of the optimizer | 0.964 | 12.5% |
| Best state encountered according to the continuous objective | 0.968 | 19% |
| Best state encountered according to the profiled likelihood | 0.988 | 70% |
| Early stopping (as soon as the profiled likelihood falls below its initial value) | 1.000 | 100% |
| Monotone descent at fixed steepness | 0.998 | 94% |

| Start at the first candidate | Mean agreement | Recovery ≥ 0.99 |
|---|---|---|
| Without optimizing | 0.996 | 86% |
| Final state | 0.963 | 11% |
| Best state according to the continuous objective | 0.967 | 17% |
| Best state according to the profiled likelihood | 0.987 | 63% |
| Early stopping | 0.996 | 86% |
| Monotone descent | 0.994 | 80% |

- **Benefit of the optimization: none. Damage: −0.036 of agreement and −75 to −88 points of recovery.** On these
  starting points, the optimization brings nothing and destroys.
- Early stopping stops at step 0 in all the tasks: it is equivalent to not optimizing.
- The best state according to the profiled likelihood is the initial state in 64 to 67% of cases; otherwise it is a state slightly
  more likely and functionally worse.
- **Monotonicity**: the optimizer can be made monotone with respect to the objective it claims to optimize (a step is
  kept only if it improves it). But it then hardly moves any more (displacement 0.009): monotone descent "succeeds"
  by being a near no-op. And if it did advance, it would go towards the optimum of the continuous objective, which is the wrong
  place. Forcing monotonicity on a misaligned criterion solves nothing.

## 9. Is the real problem the optimization mechanism or the optimized objective?

The objective.

- Mechanism: changing the learning rate, the steepness or the schedule does not change the recovery (12.5% in all the
  variants). The mechanism explains the brutality of the initial drop, not the end point.
- Objective: the optimum of the continuous objective is at an agreement of 0.964 to 0.975, whatever the way of getting there.
- Acceptance criterion, measured by arm C (same guardrail, but criterion = continuous objective):

| New tasks | Recovery ≥ 0.99 | Mean agreement | Fits accepted / rolled back |
|---|---|---|---|
| A — no criterion | 7.0% | 0.958 | — |
| C — surrogate criterion | 16.4% | 0.962 | 11.6 / 0.6 |
| B — profiled likelihood | 53.9% | 0.972 | 6.1 / 5.3 |

The surrogate criterion accepts almost everything (95% of the proposals), since that is what the optimizer improves. A
guardrail has value only if its criterion is the one that matters.

## 10. Is the Validated Transformation Principle supported experimentally here?

Yes, for parametric fitting, with one condition.

**Principle**: any internal transformation liable to modify a state (parameters, structure, belief, fragment,
primitive, strategy, weights, code) follows `State → Transformation Proposal → Evaluation → Accept / Reject → Commit /
Rollback`. A transformation cannot silently destroy a state that is already better according to the available criteria.

**What is demonstrated**: applied to a single stage, it restores +47 points of recovery on never-seen tasks
(CI [+38, +56]), with no additional computation cost, while degrading only 4.7% of the tasks and no recovery.

**The condition**: the evaluation criterion must be aligned with what matters. The same mechanism with the surrogate
criterion restores only 9 points. The principle protects against destruction according to the chosen criterion; it does not
correct a bad criterion.

**What is not demonstrated**: its value outside parametric fitting, with discovered rather than
provided atoms, or in a confirmatory test. It is implemented only here (`validated_fit`), in accordance with the instruction.

### Possible extension (proposal, not carried out)

In order of proximity to what is measured: (1) simplification, which removes a necessary atom in 1.6% of the
tasks and would replace its rule with the same transaction; (2) the rounds of reinjection of the polished atoms; (3) the
promotion of fragments and primitives into the libraries; (4) belief updates after active measurement.
Each extension would require its own observable criterion and its own two-arm experiment.

### Link with future self-evolution (architectural implication, not a demonstrated capability)

The same mechanics could serve as an invariant for more ambitious transformations: structural learning,
consolidation, birth and retirement of primitives, neural plasticity, self-programming, architectural evolution.
In each of these cases, the system would modify its own state and would need not to lose a better state. None
of this is demonstrated here: the only result is that an optimization stage that was destroying nearly nine good solutions out of ten
stops doing so when its outputs go through a validation. The transferable lesson is twofold — keep the state
before transforming it, and choose an acceptance criterion that really measures what one wants to preserve.

## What this sub-milestone does not solve

- The optimization itself remains useless on candidates that are already correct: the guardrail reaches 54 to 59%, the no-op 86 to
  89%. The next fix, to be tested ALONE, would bear on the optimized objective (or on the decision not to optimize
  a candidate coming from the synthesis), not on the mechanism.
- In the standard regime at depths 4–5, the gain is not established.
- Everything rests on provided atoms. With discovered, approximate atoms, the optimization may be necessary and
  the balance different: not measured.
- 16 tasks per cell, 64 structures per set.
- Nothing here is confirmatory. The preregistered hypothesis H10(b) is refuted on the composites; H10(a) (speed)
  is supported only for batch fits (≈ 7× per batch of 8, according to a one-off local measurement from the start of the milestone; 1.1× measured here for an isolated fit).

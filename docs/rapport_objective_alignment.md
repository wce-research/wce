# Report — WCE Objective Alignment & Conditional Optimization

EXPLORATORY sub-milestone of the milestone "Deep Compositional Structure Discovery & Model-Class Boundary".
Data: **dev partition only**. Neither the validation set, nor the confirmatory test, nor the sealed set was touched.
Everything is `POST-HOC / EXPLORATORY`.

Sources: `results/deep_objective_summary.json`; AWS runs `wce-deepobj-20261004-165859` (per-candidate study),
`wce-deepe2e-20261004-170452` (full chain), `wce-deepresid-20261004-172752` (residual degradations), no
unit in error. Targets: `make deep-run-objectives`, `deep-tune-gate`, `deep-run-e2e`, `deep-run-residual`,
`deep-obj-aggregate`, `deep-obj-report`.

Task sets, all in dev and with structures disjoint from each other and from the previous sets: gate **tuning** (64
tasks, 448 candidates); **evaluation** (128 tasks, 896 candidates). Each set of 128 tasks contains 64 structures
observed in two regimes.

## Summary

- **When to optimize**: when the candidate is approximate. The benefit is zero or negative for a parameter error
  of about 0.07, positive from about 0.17, and strong beyond that.
- **When to do nothing**: when the candidate already explains the observations as well as the noise allows.
- **None of the six objectives tested is truly aligned** with functional quality on candidates that are already good.
- **The conditional gate** is the only policy close to the best in all three input regimes at the candidate
  level. It preserves 98% of the good solutions.
- **With discovered primitives, the gain does not survive**: the safeguard and the gate do not do significantly
  better than "always optimize". The candidates there are almost all bad; there is nothing to preserve.

## The three input regimes

| Regime | Candidates (evaluation) | Mean initial agreement | Candidates already ≥ 0.99 |
|---|---|---|---|
| A — true atoms, true parameters | 128 | 1.000 | 100% |
| B — true atoms, weak perturbation (σ = 0.03) | 128 | 0.972 | 5% |
| B — medium perturbation (σ = 0.10) | 128 | 0.924 | 0% |
| B — strong perturbation (σ = 0.30) | 128 | 0.804 | 0% |
| C — discovered atoms, first 3 candidates of the synthesis | 384 | 0.819 | 0% |

## 1. When is optimization actually useful?

Curve "initial parameter error → expected benefit of optimization" (change in agreement, current objective):

| Initial error (maximum parameter deviation) | Initial agreement | Always optimize | Safeguard | Conditional gate |
|---|---|---|---|---|
| 0.00 | 1.000 | −0.033 | −0.009 | −0.001 |
| 0.07 | 0.972 | −0.005 | −0.003 | +0.001 |
| 0.17 | 0.924 | +0.043 | +0.043 | +0.039 |
| 0.50 | 0.804 | +0.156 | +0.156 | +0.156 |

The benefit changes sign between 0.07 and 0.17. With discovered atoms (regime C, initial agreement 0.819),
optimization yields +0.019 agreement per candidate: 73% of the candidates improve, 17% degrade.

In the full chain with discovered primitives, never optimizing costs 0.045 agreement (95% CI [0.031, 0.058]) and
brings recovery at 0.97 down from 16% to 3%. **Yes: with approximate parameters, optimization is useful.**

## 2. When should nothing be done?

When the candidate is already good. In regime A, optimizing destroys 90 to 93% of the good solutions depending on the
objective; doing nothing preserves them all. In the full chain with the true atoms (128 tasks):

| Policy | Recovery ≥ 0.99 | ≥ 0.97 | Mean agreement | Fit-equivalents | Time |
|---|---|---|---|---|---|
| Always optimize | 6.3% | 58.6% | 0.963 | 28.3 | 5.6 s |
| Safeguard | 57.0% | 76.6% | 0.980 | 27.4 | 4.6 s |
| Conditional gate | 72.7% | 79.7% | 0.980 | 19.8 | 2.5 s |
| Never optimize | 88.3% | 89.1% | 0.988 | 16.0 | 1.0 s |

Observable signals that distinguish the cases (computed before any fit, mean over the evaluation candidates):

| Actual outcome of the optimization | Candidates | Initial error rate | Spatial clustering of the residual (z) | Deviation from the local reference (z) | Gradient per point |
|---|---|---|---|---|---|
| Helped | 567 | 0.18 | 5.7 | +2.6 | 0.44 |
| No effect | 167 | 0.12 | 2.8 | −1.4 | 0.19 |
| Harmed | 162 | 0.14 | 3.1 | −0.8 | 0.25 |

The deviation from a local non-parametric reference (vote of 7 neighbours) is the clearest signal: positive when
optimization helps, negative when it harms or is of no use. The "disagreement between criteria" measured by a trial
step does not separate the cases.

## 3. Which objective is best aligned with actual quality?

None is on candidates that are already good. Same mechanism for all the differentiable objectives (Adam, 100 steps);
the discrete error is optimized by direct search.

**Regime A (initial agreement 1.000)** — where an aligned objective should damage nothing:

| Objective | Final agreement | Degraded candidates | The objective improves AND the agreement drops | Correlation objective ↔ agreement | ≥ 0.99 retained |
|---|---|---|---|---|---|
| Current surrogate | 0.968 | 100% | 89% | 0.36 | 7% |
| Softened profile likelihood | 0.955 | 100% | 84% | 0.49 | 9% |
| Observable discrete error | 0.993 | 75% | 75% | −0.24 | 72% |
| Likelihood + prior on the displacement | 0.981 | 99% | 98% | 0.27 | 19% |
| Local around the boundaries | 0.966 | 100% | 88% | 0.43 | 8% |
| Robust to outliers | 0.967 | 100% | 90% | 0.40 | 10% |

**Other regimes** (final agreement; share where the objective improves and the agreement drops):

| Objective | B weak (0.972) | B medium (0.924) | B strong (0.804) | C discovered (0.819) |
|---|---|---|---|---|
| Current surrogate | 0.967; 43% | 0.967; 6% | 0.960; 2% | 0.838; 15% |
| Softened profile likelihood | 0.958; 53% | 0.955; 15% | 0.947; 2% | 0.837; 17% |
| Observable discrete error | 0.977; 26% | 0.967; 2% | 0.894; 1% | 0.824; 17% |
| Likelihood + prior on the displacement | 0.981; 16% | 0.980; 0% | 0.949; 0% | 0.829; 10% |
| Local around the boundaries | 0.966; 44% | 0.965; 9% | 0.953; 2% | 0.837; 14% |
| Robust to outliers | 0.967; 43% | 0.966; 6% | 0.959; 2% | 0.839; 15% |

Reading:

- **Discrete error**: the least destructive on a candidate that is already good (72% retained), but it still degrades
  three candidates out of four — reducing the errors on noisy observations means fitting the noise — and it is the
  weakest on the strongly perturbed or discovered candidates.
- **Prior on the displacement**: the best aligned for weak and medium perturbations (the agreement never drops
  when the objective rises under medium perturbation), but worse than the current objective on the discovered
  atoms.
- **Current, local, robust objectives**: equivalent to each other to within less than 0.01; the best on the strongly
  perturbed and discovered candidates.
- **Softened profile likelihood**: never the best.

No objective dominates in all regimes. Tuning on the tuning set retains the robust objective, practically tied
with the current objective.

## 4. Does the conditional gate beat always / never optimize?

At the candidate level, yes. Gate tuned on the tuning set only, then evaluated on the evaluation set:
optimize if the residual is spatially structured (z ≥ 4) or if the candidate is not clearly better than the
local reference (z ≥ −1); otherwise preserve; a proposal is accepted only if it gains at least 2 nats of
profile likelihood; the retained state is the best state encountered.

The four requested rates (896 candidates, 134 good at the start, 762 bad; robust objective retained by the tuning —
the figures with the current objective are identical to within 0.001 on the mean agreement):

| Policy | Good solutions preserved | Good destroyed | Bad improved | Bad not repaired | Bad worsened | Mean agreement | ≥ 0.99 | Candidates optimized |
|---|---|---|---|---|---|---|---|---|
| Always optimize | 10% | 90% | 76% | 24% | 16% | 0.911 | 6.0% | 100% |
| Never optimize | 100% | 0% | 0% | 100% | 0% | 0.880 | 15.0% | 0% |
| Safeguard | 71% | 29% | 73% | 27% | 14% | 0.914 | 15.1% | 100% |
| Safeguard on the best state encountered | 66% | 34% | 74% | 26% | 15% | 0.914 | 14.3% | 100% |
| Conditional gate | **98.5%** | 1.5% | 64% | 36% | 9% | **0.916** | **18.1%** | 72% |

By regime (mean agreement):

| Policy | A | B weak | B medium | B strong | C |
|---|---|---|---|---|---|
| Always optimize | 0.967 | 0.967 | 0.966 | 0.959 | 0.839 |
| Never optimize | 1.000 | 0.972 | 0.924 | 0.804 | 0.819 |
| Safeguard | 0.990 | 0.970 | 0.966 | 0.959 | 0.839 |
| Conditional gate | 0.999 | 0.973 | 0.963 | 0.958 | 0.838 |

Gate decisions: it preserves 121 candidates out of 128 in regime A, 89 out of 128 under weak perturbation, 21 out of 128 under
medium perturbation, 0 out of 128 under strong perturbation and 17 out of 384 in regime C. It therefore does learn to distinguish
need from danger on the controlled regimes.

Its cost: it repairs fewer bad candidates (64% versus 76%), especially under medium perturbation where it lets
through 16% of candidates that would have gained from being optimized (agreement 0.963 versus 0.966).

"Keeping the best state encountered" brings nothing on its own (0.914 in both cases): the best state according to the
profile likelihood is often a state better fitted to the noise.

No learned component was introduced: the two-threshold rule is enough to separate the controlled regimes.

## 5. Are the remaining 4.7% of degradations explained?

Yes. The 9 tasks degraded by the safeguard (6 new ones, i.e. 4.7%; 3 from the diagnostic, i.e. 2.3%) were replayed
with both arms traced.

| Explanation | Tasks | What happens |
|---|---|---|
| The search path changed the structure | 5 | The safeguard's solution is LESS likely on the data and scored lower than that of the frozen pipeline, and of a different structure (often smaller: 2 to 4 atoms versus 3 to 5). By cancelling fits, the safeguard reinjects other atoms into the synthesis, which no longer produces the structure that the other arm found. The acceptance criterion is not at fault. |
| The noise is exploited by an accepted fit | 2 | An accepted fit improves the profile likelihood and degrades the actual agreement; the safeguard's final solution explains the data better (for example −115.9 versus −117.8) and is functionally worse (0.841 versus 0.951). |
| The data prefer the worse function | 2 | The safeguard's solution has a better score (complexity penalty included) or a better likelihood, and a lower agreement. The profile score is insufficient. |

Useful detail: in these 9 tasks, 42 of the 69 fits accepted by the safeguard lower the actual agreement. Accepting
a fit because it better explains noisy observations therefore remains risky; this is the reason for the minimum gain
of 2 nats retained for the gate. In 7 of the 9 tasks, neither arm retains the true form; the agreement gaps
range from 0.001 to 0.110.

## 6. Does the gain hold with discovered primitives?

**No.** Full chain `observations → primitive discovery → composition → optimisation policy →
selection`, 128 evaluation tasks, primitives discovered by the reference method (multi-initialization gradient
fit):

| Policy | ≥ 0.90 | ≥ 0.95 | ≥ 0.97 | ≥ 0.99 | Mean agreement | Agreement difference from "always" | 95% CI | Improved / degraded |
|---|---|---|---|---|---|---|---|---|
| Always optimize | 53.9% | 28.1% | 16.4% | 5.5% | 0.876 | — | — | — |
| Never optimize | 32.0% | 9.4% | 3.1% | 0.0% | 0.831 | −0.045 | [−0.058, −0.031] | 19% / 76% |
| Safeguard | 53.9% | 28.9% | 17.2% | 5.5% | 0.877 | +0.001 | [−0.002, +0.004] | 9% / 5% |
| Conditional gate | 52.3% | 22.7% | 14.1% | 4.7% | 0.881 | +0.005 | [−0.003, +0.013] | 52% / 41% |

Neither the safeguard nor the gate is distinguishable from "always optimize": the intervals include zero and the
recovery at 0.99 does not move. The gate slightly improves the mean agreement but pushes back the 0.95 and 0.97 thresholds
(−5.5 and −2.3 points, not significant).

By depth, gate versus safeguard (agreement difference): −0.006 at d2, −0.009 at d3, **+0.017 [+0.000, +0.033]** at
d4, **+0.015 [+0.001, +0.029]** at d5. A slight advantage at depths 4 and 5, at the edge of significance, on
32 tasks per depth.

By regime: standard +0.008 [−0.005, +0.021]; rich +0.002 [−0.008, +0.013] for the gate versus "always".

Why the gain does not survive: with discovered primitives, the candidates have an initial agreement of 0.82 and none
reaches 0.99. The safeguard and the gate protect the good solutions; there are none to protect. The limiting
factor is once again primitive discovery.

## 7. Does optimization become useful when the parameters are approximate?

Yes (section 1): +0.043 agreement for a medium perturbation, +0.156 for a strong perturbation, +0.019 per candidate
with discovered atoms, and +0.045 in the full chain compared with not optimizing. In these regimes,
optimization repairs 92 to 98% of the perturbed candidates and 73% of the discovered candidates.

It is nonetheless not sufficient: after optimization, only 8 to 9% of the perturbed candidates reach 0.99, and 2%
of the discovered candidates. Optimization brings candidates closer, it does not finish the job.

## 8. Which rule should become the reference WCE implementation?

**The explicit conditional gate, followed by a validation**, with the current objective unchanged:

```
candidate → observable signals → structured residual (z ≥ 4) or candidate no better than the local reference (z ≥ −1)?
   no  → PRESERVE
   yes → optimiser proposal → best state encountered → profiled-likelihood gain ≥ 2 nats?
            yes → commit        no → rollback
```

Justification from the data:

- it is the only policy close to the best in all three regimes at the candidate level (0.999 / 0.958 to 0.973 /
  0.838), whereas "always" fails in A and "never" fails in B and C;
- with the true atoms it recovers 72.7% at 0.99 versus 57.0% for the safeguard alone, for 28% fewer fit-equivalents;
- with discovered primitives it does no worse than "always optimize";
- it is interpretable: two thresholds on two signals, one acceptance margin, and every decision is traced
  (`OptimizationDecision`).

Reservations, to be settled at review:

- with the true atoms, "never optimize" remains better (88.3%). The gate is a compromise, not a per-regime
  optimum;
- its advantage in the full chain with discovered primitives is not established;
- the thresholds were tuned on 64 dev tasks; they are validated neither on the validation set nor in the confirmatory test;
- changing the objective is not justified: none is aligned, and the prior on the displacement, the most promising for
  small perturbations, does worse on the discovered atoms.

The gate is not enabled by default: `DeepSearch(validated="gate")` enables it at the fit step, and nowhere
else. The default pipeline remains the frozen pipeline.

## Validated Transformation Principle

Formalized in `docs/architecture_validated_transformation.md`, as a principle supported experimentally for
parametric fitting only. This sub-milestone adds one element to it: deciding not to transform is an action
in its own right, and it is what takes preservation from 71% to 98.5%. The link with a self-evolving WCE is
presented there as an architectural direction, not as a demonstrated capability.

## Limits

- Everything is in dev, exploratory; 64 structures per set of 128 tasks.
- The gate and the safeguard do nothing for primitive discovery, which remains the bottleneck: in the full chain,
  5% recovery at 0.99 whatever the policy.
- Regime C uses only one discovery method (multi-initialization gradient) and the first three candidates of
  the synthesis.
- The perturbations of regime B are Gaussian and independent per parameter; real discovery errors
  may be structured differently.
- The prior on the displacement (σ = 0.1) and the robust contamination (γ = 0.2) were tested at a single value.
- The "stability" signal was not computed; "confidence" is measured by the deviation from the local reference, and
  "distance between hypotheses" is not a gate signal.
- In 5 of the 9 residual degradations, the cause is a path effect in the synthesis, which neither the safeguard nor the
  gate addresses.

# WCE Boundary Ownership & Fragment Attribution — report (dev, exploratory)

Status: exploratory sub-milestone, **dev** partition only. No confirmatory test, no 16-seed campaign,
sealed set never touched. The frozen pipeline remains the reference; nothing from this sub-milestone is wired into it by default.

## 0. Preregistration (written before launching the AWS runs)

Tuning done only on 120 scenes of the dev + 350,000 sub-range and on composition task 799.
Settings frozen before launch (`results/deep_configs/ownership.json`): cost per parameter 10, "absent" term at
zero, chance cost 3, temperature 16.

Evaluation on 1,920 never-seen occlusion scenes (indices 200–231 of the dev + 300,000 sub-range) and on **256
never-used composition tasks** (keys 700–731).

### Confirmation of `flat_scene_k3` (main comparison, fixed here)

In the previous sub-milestone, `flat_scene_k3` beat the frozen pipeline, but it had been chosen among a dozen
arms. The confirmation covers **three arms only, designated in advance**: frozen pipeline, `flat_scene_k3`, and the
proposed ownership method `own_k3`. New and independent task set, twice as large.

Criterion: `flat_scene_k3` − frozen pipeline, paired difference per task, 95% CI above zero **both** on
the candidate structure and on the final agreement. If the criterion is met, `flat_scene_k3` becomes a **candidate
baseline**; otherwise, it remains an unconfirmed exploratory result. This status is independent of the verdict of the
sub-milestone.

### Hypotheses

| # | Hypothesis | Criterion |
|---|---|---|
| BO1 | global attribution beats local micro-fragments | pairwise F1, `solver` − `tangent`, 95% CI > 0 |
| BO2 | stitching reduces fragmentation | pieces per true boundary, `solver` − `tangent`, CI < 0 |
| BO3 | long-range association helps | F1, `solver` − `solver_local`, CI > 0 |
| BO4 | the occlusion model helps | top-1 ownership: `solver_absent` − `solver` > 0 and `solver_absent` − `solver_absent_naive` > 0 |
| BO5 | the ownership top-3 contains the right owner markedly more often | top-3 − top-1 ≥ 0.10 |
| BO6 | composition helps to re-attribute | chain: F1 after composition − F1 of the solver, CI > 0 |
| BO7 | the ownership score is calibrated | calibration error ≤ 0.10 on the evaluation |
| BO8 | counterfactual visibility beats contrast | top-1 recovery, inferred masks, CI > 0 |
| BO9 | with true ownership and type, an unoccluded sector is recovered | recovery ≥ 0.5; otherwise the parameterisation is intrinsically fragile |
| BO10 | `flat_scene_k3` is confirmed | criterion above |
| BO11 | the ownership method beats the frozen pipeline | `own_k3` − reference, candidate structure and final agreement, CI > 0 |
| BO12 | the ownership method beats `flat_scene_k3` | `own_k3` − `flat_scene_k3`, final agreement, CI > 0 |

Expectation declared from the tuning: BO1, BO2, BO3 likely; BO4 probably false (the "absent" term
degrades attribution in tuning, with or without an explicit mask); BO7 uncertain; BO10 likely; BO11 and BO12
unlikely.

Limit known before launch: no chance cost tested brings below 10% the share of midpoints attributed on
random labels. The strongest (3) leaves 32%; without this cost, 96%. The arm without chance cost
(`solver_no_chance`, `own_nc_k3`) is reported separately, as a comparison.

The non-designated comparison arms (`own_nc_k3`, `flat_owned_k3`, `flat_own_union_k3`, `po_owned_k3`) and the oracle
matrix are exploratory and change neither the verdict nor the status of `flat_scene_k3`.

Verdict rule fixed in advance:

- `BOUNDARY OWNERSHIP READY FOR END-TO-END VALIDATION`: BO1, BO11 and BO12 supported.
- `BOUNDARY OWNERSHIP IMPROVED BUT INSUFFICIENT`: BO1 supported, but BO11 or BO12 not supported.
- `BOUNDARY OWNERSHIP STILL BLOCKING`: BO1 not supported.

## 1. Protocol executed

- AWS run `wce-deepbo-20261005-211529` (EC2 + SSM, 2 × c6i.xlarge, 80 and 85 min, 3,296 sub-units, 0 errors,
  26,176 lines). Commands: `make deep-run-bo`, `make deep-fetch`, `make deep-bo-aggregate`, `make deep-bo-report`.
  Summary: `results/deep_boundary_ownership_summary.json`.
- Occlusion bench: 1,920 never-seen scenes. Downstream chain: 256 never-used composition tasks.
- Code: `core/deep_ownership.py` (system, without truth, without learned model), `core/deep_ownership_exp.py`
  (evaluation), `core/deep_bo_report.py`. Tests: 54 new (`make deep-bo-test`); full suite 4,894 tests OK,
  7 skipped; sealed set intact.
- Settings frozen before launch; no threshold tuned on the evaluation. A single seed per scene.
- Deviation from the protocol: none.

## 2. Confirmation of `flat_scene_k3` (256 new tasks, three designated arms)

| Arm | Primitives recovered | Literals | Candidate structure | Final ≥ 0.95 | Final agreement | Seconds |
|---|---|---|---|---|---|---|
| frozen pipeline | 0.26 | 80.3 | 0.19 | 0.21 | 0.881 | 18.2 |
| **`flat_scene_k3`** | 0.40 | 20.9 | **0.32** | **0.37** | **0.904** | 13.7 |
| `own_k3` (ownership method) | 0.04 | 2.1 | 0.13 | 0.16 | 0.620 | 4.4 |

`flat_scene_k3` − frozen pipeline, paired differences, 95% CI:

- candidate structure: **+0.137 [+0.090, +0.184]**;
- final agreement: **+0.023 [+0.015, +0.031]**;
- final ≥ 0.95: +0.160 [+0.109, +0.215]; primitives recovered: +0.148 [+0.106, +0.189].

The preregistered criterion is met on both of its parts. **`flat_scene_k3` becomes a candidate baseline.** It remains
at unknown type, without oracle, with four times fewer literals and a quarter less time. The gain in final agreement
is smaller than in the previous sub-milestone (+0.023 versus +0.032), which is expected of an arm selected and then
re-measured. Recovery at 0.99 remains at 6%.

By depth, candidate structure (`flat_scene_k3` / frozen pipeline): 0.70 / 0.48 at 2; 0.44 / 0.17 at 3;
0.13 / 0.06 at 4; 0.03 / 0.03 at 5.

## 3. Attribution (1,920 scenes)

| Method | Precision | Recall | F1 | Purity | Pieces per true boundary | Mean length | Top-1 ownership | top-3 |
|---|---|---|---|---|---|---|---|---|
| micro-fragments (`tangent`) | 0.88 | 0.46 | 0.57 | 0.94 | 2.58 | 0.63 | — | — |
| local stitching (`shared`) | 0.80 | 0.58 | 0.64 | 0.91 | 2.15 | 1.14 | — | — |
| consensus | 0.65 | 0.83 | 0.71 | 0.81 | 1.50 | 2.09 | — | — |
| **`solver`** (global, calibrated by chance) | 0.72 | 0.78 | 0.73 | 0.80 | 1.13 | 2.69 | 0.39 | 0.41 |
| `solver_no_chance` | 0.76 | 0.81 | **0.77** | 0.84 | 1.38 | 2.67 | **0.48** | 0.53 |
| `solver_local` (without long range) | 0.82 | 0.55 | 0.62 | 0.88 | 1.21 | 1.35 | 0.32 | 0.36 |
| `solver_no_smooth` | 0.70 | 0.76 | 0.71 | 0.78 | 1.09 | 2.82 | 0.37 | 0.39 |
| `solver_absent` (negative evidence, explicit masks) | 0.56 | 0.56 | 0.53 | 0.79 | 0.78 | 2.44 | 0.27 | 0.28 |
| `solver_absent_naive` | 0.55 | 0.56 | 0.53 | 0.77 | 0.71 | 2.56 | 0.26 | 0.26 |
| composition (re-attribution by a small composite) | 0.67 | 0.74 | 0.66 | 0.77 | 0.96 | 2.82 | — | — |

Paired differences, 95% CI:

| Comparison | F1 | Purity | Pieces per boundary | Top-1 ownership |
|---|---|---|---|---|
| `solver` − `tangent` | +0.154 [+0.142, +0.165] | −0.138 [−0.145, −0.130] | −1.45 [−1.50, −1.39] | — |
| `solver` − consensus | +0.021 [+0.012, +0.029] | −0.008 [−0.014, −0.001] | −0.37 [−0.40, −0.34] | — |
| `solver` − `solver_local` | +0.113 [+0.102, +0.125] | −0.079 | −0.08 | +0.070 [+0.053, +0.088] |
| `solver` − `solver_no_smooth` | +0.023 [+0.018, +0.027] | +0.022 | +0.04 | +0.020 [+0.013, +0.027] |
| `solver` − `solver_no_chance` | −0.038 [−0.045, −0.030] | −0.031 | −0.25 | −0.085 [−0.098, −0.072] |
| `solver_absent` − `solver` | −0.197 [−0.211, −0.183] | −0.022 | −0.35 | −0.127 [−0.142, −0.112] |
| `solver_absent` − `solver_absent_naive` | +0.006 [+0.003, +0.009] | +0.014 | +0.06 | +0.006 [+0.002, +0.011] |
| composition − `solver` | −0.064 [−0.071, −0.057] | −0.036 | −0.17 | — |

Top-1 ownership by occluded share: `solver` 0.69; 0.44; 0.30; 0.22; 0.31. `solver_no_chance` 0.81; 0.56;
0.41; 0.29; 0.33. By type (`solver_no_chance`): anti-disc 0.61; disc 0.54; ring 0.49; half-plane 0.48;
band 0.46; sector 0.29.

Mean energy of the retained solution, by term: fit 25.8; midpoints without owner 36.7; smoothing 0.9;
complexity 36.8; absence 0 (term disabled); chance 80.0. The chance cost is the dominant term: it decides
the number of curves kept (1.5 on average, versus 2.0 without it). 27 attributions tested per scene, 0.21 s.

Mask detection: a mask is declared in 11 to 24% of the scenes, and it is the right one in only 2 to 5%. Without
a mask, one is wrongly declared in 11% of the scenes.

Calibration of the ownership score: calibration error 0.094, Brier score 0.25. But the confidence is almost
constant (0.29 to 0.54) for an accuracy of 0.42: the score is right on average and orders nothing. By occlusion
level, the error goes from 0.08 to 0.22. As a consequence of the chosen temperature, 95% of the fragments are classified as ambiguous.

## 4. Oracle matrix

Exact recovery of the target on the occlusion bench. The estimated ownership serves as a filter: only the hypotheses
all of whose boundaries are owned curves are kept.

| Ownership | True type | Inferred type |
|---|---|---|
| true | 0.67 | 0.44 |
| none (whole scene) | 0.54 | 0.35 |
| estimated, as a filter | 0.26 | 0.22 |
| estimated, primitives built on the owned curves | — | 0.25 |

Paired differences: true ownership − none, +0.13 [+0.11, +0.15] at true type and +0.08 [+0.07, +0.10] at
inferred type; estimated ownership − none, **−0.27 [−0.29, −0.25]** at true type and −0.13 [−0.16, −0.11] at inferred type.
The filter keeps only 14% of the hypotheses, and the target has all its boundaries owned in only 34% of the scenes.

In the downstream chain (256 tasks):

| Condition | Primitives recovered | Candidate structure | Final agreement |
|---|---|---|---|
| true atoms | 1.00 | 0.65 | 0.959 |
| true ownership, true type | 0.74 | 0.52 | 0.938 |
| true ownership, inferred type | 0.65 | 0.40 | 0.919 |
| no attribution, true type (`po_type_k3`) | 0.40 | 0.39 | 0.915 |
| no attribution, inferred type (`flat_scene_k3`) | 0.40 | 0.32 | 0.904 |
| estimated ownership without chance cost (`own_nc_k3`) | 0.25 | 0.29 | 0.893 |
| estimated ownership (`own_k3`) | 0.04 | 0.13 | 0.620 |

From `flat_scene_k3` (0.32) to the true atoms (0.65), the gap of 0.33 breaks down as follows: +0.08 by giving the true
ownership, +0.12 by then giving the type, +0.13 by finally giving the parameters.

## 5. Counterfactual visibility versus contrast

Same pool of five hypotheses at true type, four rankings (recovery in first position):

| Ranking | All scenes | Mask 40% | Mask 60% |
|---|---|---|---|
| global contrast | 0.55 | 0.59 | 0.30 |
| visibility, no mask assumed | 0.51 | 0.54 | 0.23 |
| visibility, inferred masks | 0.52 | 0.55 | 0.23 |
| visibility, true mask | 0.58 | 0.64 | 0.34 |

Against contrast: inferred masks −0.031 [−0.045, −0.017]; true mask +0.032 [+0.018, +0.046]. The principle
works when the mask is known; it is unusable as long as the mask is not found. The pool contains
the truth in only 67% of the cases.

## 6. Sectors: attribution or parameterisation?

| Mask | True ownership and type | No attribution, true type | Top-1 ownership of the half-lines | Midpoints on the poorest half-line | Apex error (true ownership) |
|---|---|---|---|---|---|
| 0% | 0.70 | 0.72 | 0.45 | 13.0 | 0.37 |
| 20% | 0.55 | 0.25 | 0.31 | 8.5 | 0.73 |
| 40% | 0.27 | 0.17 | 0.28 | 4.2 | 0.80 |
| 60% | 0.02 | 0.09 | 0.32 | 1.1 | 1.54 |
| 80% | 0.00 | 0.03 | 0.05 | 0.3 | 1.83 |

With true ownership and type, the failures are the apex in the great majority of cases (15 out of 19 without mask,
27 out of 47 at 40%), then the absence of a hypothesis when a half-line has no midpoints left.

## 7. Downstream chain, exploratory arms (256 tasks)

| Arm | Status | Attribution F1 | Top-1 ownership | Literals | Candidate structure | Final agreement |
|---|---|---|---|---|---|---|
| frozen pipeline | designated | — | — | 80.3 | 0.19 | 0.881 |
| `flat_scene_k3` | designated | — | — | 20.9 | 0.32 | 0.904 |
| `own_k3` | designated | 0.30 | 0.09 | 2.1 | 0.13 | 0.620 |
| `own_nc_k3` | exploratory | 0.57 | 0.38 | 7.0 | 0.29 | 0.893 |
| `flat_owned_k3` (hypotheses filtered by ownership) | exploratory | 0.30 | 0.09 | 12.6 | 0.22 | 0.886 |
| `flat_own_union_k3` | exploratory | 0.30 | 0.09 | 22.9 | 0.31 | 0.903 |
| `po_owned_k3` (type oracle, filtered) | exploratory | 0.30 | 0.09 | 7.5 | 0.30 | 0.904 |

Against the frozen pipeline: `own_k3` −0.261 [−0.307, −0.214] of final agreement; `own_nc_k3` +0.012 [+0.004, +0.020] and
+0.098 [+0.043, +0.152] of candidate structure. Against `flat_scene_k3`: `own_k3` −0.284 [−0.331, −0.236];
`own_nc_k3` −0.011 [−0.018, −0.004]; filtering by ownership −0.018 [−0.024, −0.012]; union 0.000.

In composition, the chance-calibrated solver keeps almost no curve: 2.1 literals per task, 4% of
primitives recovered. The boundaries there are short and numerous, and none clearly exceeds what a curve
would capture by chance.

Re-attribution by composition (arm `own_k3`): after composition, the midpoints are re-attributed to the leaves of the
best composite where they are visible. F1 0.30 → 0.53, that is +0.11 [+0.08, +0.14] on the paired tasks;
ownership top-1 0.09 → 0.32, that is +0.19 [+0.15, +0.23].

## 8. Preregistered hypotheses

| # | Result |
|---|---|
| BO1 global attribution > micro-fragments | **supported**: F1 +0.154 [+0.142, +0.165] |
| BO2 stitching reduces fragmentation | **supported**: 2.58 → 1.13 pieces, −1.45 [−1.50, −1.39]; purity 0.94 → 0.80 |
| BO3 long-range association | **supported**: F1 +0.113 [+0.102, +0.125] |
| BO4 occlusion model | **not supported**: the absence term degrades ownership (−0.127 [−0.142, −0.112]); explicit masks versus naive version +0.006 |
| BO5 top-3 − top-1 ≥ 0.10 | **not supported**: 0.41 − 0.39 = 0.02 (0.05 without chance cost) |
| BO6 composition helps to re-attribute | **supported** in the chain (+0.11 [+0.08, +0.14]), starting from a weak attribution; the opposite on the occlusion bench (−0.064) |
| BO7 calibration ≤ 0.10 | **supported on the criterion** (0.094), without discriminating power |
| BO8 counterfactual visibility > contrast | **not supported**: −0.031 [−0.045, −0.017] with inferred masks |
| BO9 unoccluded sector, true ownership and type ≥ 0.5 | **supported**: 0.70 |
| BO10 confirmation of `flat_scene_k3` | **supported**: +0.137 and +0.023, intervals above zero |
| BO11 `own_k3` > frozen pipeline | **not supported**: −0.261 of final agreement |
| BO12 `own_k3` > `flat_scene_k3` | **not supported**: −0.284 |

## 9. Answers

**1. What share of the error really comes from ownership?**
Less than I had written in the previous sub-milestone. On the occlusion bench, giving the true ownership is worth +0.13
[+0.11, +0.15] of exact recovery. In the chain, of the gap of 0.33 in candidate structure between
`flat_scene_k3` and the true atoms, the true ownership gives back 0.08, the type 0.12 and the parameters 0.13: about a
quarter. The previous report attributed 0.16 to it because it measured it at true type; at inferred type, it weighs half
as much. Ownership is one lock among three of comparable size, not the dominant lock.

**2. Does top-k ownership help?**
No. The right owner is in the first three barely more often than in first place (0.41 versus 0.39; 0.53
versus 0.48 without chance cost). When the solver is wrong, the right curve is generally not among those
it has retained: keeping several owners cannot bring it back.

**3. Does stitching reduce fragmentation?**
Yes. A true boundary goes from 2.6 pieces to 1.1 and the mean length of a group from 0.6 to 2.7. The price is
purity, from 0.94 to 0.80. Local stitching alone is not enough (2.2 pieces).

**4. Do long-range associations help?**
Yes: F1 +0.11 [+0.10, +0.12], top-1 ownership +0.07 [+0.05, +0.09]. Without them, recall falls from 0.78 to 0.55.

**5. Does the occlusion model help?**
No. The mask is correctly identified in only 2 to 5% of the scenes. Penalising the expected and absent boundary
degrades attribution, with or without explicit masks. The counterfactual visibility score loses against
contrast with the inferred masks (−0.031) and wins only with the true mask (+0.032). The idea is right, its
precondition — finding the mask — is not met.

**6. Does composition help to re-attribute?**
In the chain, yes, clearly: attribution F1 +0.11 and top-1 ownership +0.19. But the starting point is
the very weak attribution of the calibrated solver (0.30), and the level reached (0.53) remains below that of the solver without
chance cost (0.57). On the occlusion bench, where the starting attribution is better, composition degrades it
(−0.064). It repairs a bad attribution, it does not improve a correct attribution.

**7. Do sectors remain problematic?**
Yes, and the cause is now separated. Without mask, true ownership and type give 0.70, as much as without
any attribution (0.72): the parameterisation is not intrinsically fragile and attribution is not the problem.
At 20% mask, the true ownership is worth a lot (0.55 versus 0.25): there, it is attribution. From 40% onwards,
even the true ownership fails (0.27, then 0.02): the poorest half-line has only four midpoints left, then
one, and the apex is no longer determined. It is a lack of data, not a defect of method.

**8. Is flat_scene_k3 confirmed on new dev?**
Yes. On 256 new tasks, with three arms designated in advance: candidate structure +0.137 [+0.090, +0.184],
final agreement +0.023 [+0.015, +0.031]. It becomes a candidate baseline. This is not a confirmatory validation:
dev only, one seed.

**9. Does the ownership gain carry through to the final structure?**
No. Attribution improves (F1 0.57 → 0.73), but the designated ownership method does much worse downstream than
the frozen pipeline (agreement 0.62 versus 0.88). Its variant without chance cost beats the frozen pipeline (+0.012) but remains
below `flat_scene_k3` (−0.011). Using the estimated ownership as a filter loses 0.27 of recovery on the
bench and 0.018 of agreement in the chain. No use of the estimated ownership does better than attributing nothing.

**10. What is the new bottleneck?**
There is no longer a single one upstream. Between `flat_scene_k3` and the true atoms, ownership, type and parameters
share the gap in similar parts, and none of the three has yielded to an explicit geometric method. The ceiling
itself is low: with the true atoms, the candidate structure is reached in only 65% of the tasks (38% at
depth 5) and recovery at 0.99 in 10%. The next bottleneck to isolate is therefore downstream of the upstream:
deep composition and the final fit, with atoms given.

## 10. Limits

- Dev only, one seed per scene and per task. The confirmation of `flat_scene_k3` is a re-measurement on new
  data, not a confirmatory test.
- The calibration by chance does not reach its target: in tuning, 32% of the midpoints remain attributed on
  random labels (96% without the cost). It was not re-measured on the evaluation.
- The designated arm `own_k3` carries this chance cost, which collapses it in composition. The arm without this cost is better
  downstream but is not calibrated: neither one nor the other is satisfactory.
- The ownership score is calibrated on average by a high temperature that flattens it; it does not separate the
  right attributions from the wrong ones.
- The occlusion bench has only one form of composition (`target ∧ mask`).
- Greedy solver with at most three restarts; neither optimal matching nor integer programming were
  tried. The negative result concerns this solver and this energy.
- The "domain limit" and "unknown" roles of the ownership hypotheses are built and tested, not
  evaluated quantitatively.

## 11. Verdict

Preregistered rule: BO1 supported; BO11 and BO12 not supported.

`BOUNDARY OWNERSHIP IMPROVED BUT INSUFFICIENT`

The measured attribution progresses (stitching, long range), but it produces no downstream gain and, used
as a filter, it does harm. The occlusion model fails for lack of finding the mask. Separate result, established:
`flat_scene_k3` is confirmed on new data and becomes a candidate baseline. The frozen pipeline remains the
official reference; nothing from this sub-milestone is wired into it.

Status recorded in `results/deep_configs/baselines.json`: `flat_scene_k3` = `candidate baseline` (gain confirmed on
a new dev set; does not replace the official confirmatory baseline). `own_k3` is not wired in.

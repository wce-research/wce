# WCE Robust Primitive Discovery — report (dev, exploratory)

Status: exploratory sub-milestone, **dev** partition only. No confirmatory test, no 16-seed campaign,
sealed set never touched. The frozen pipeline remains the official reference; the conditional gate is not active by
default. Only the upstream is modified:

```
observations → primitive proposal → primitive validation → primitive parameter estimation → primitive set
```

Composition, final score, gate, simplification and selection are those of the existing pipeline, unchanged.

## 0. Preregistration (written before launching the AWS runs)

Tuning done only on the scenes with indices 5000+ (27 to 36 scenes) and composition task 499.
Evaluation on never-seen scenes (indices 2000–2031, 9 categories × 32 = 288 scenes) and on 128 never-used
composition tasks (keys 400–415).

Main arm: `union_joint` (local consensus + Hough vote proposers, local estimation, calibrated validator,
backward elimination, joint assembly). Reference: `gradient` (direct gradient, previous validator).

| # | Hypothesis | Criterion |
|---|---|---|
| RP1 | the equivalence recall exceeds that of the reference | paired difference per scene, 95% CI above 0 |
| RP2 | false positives per scene drop by at least half | paired difference, 95% CI below 0, and ratio ≤ 0.5 |
| RP3 | rings and sectors: equivalence recall higher than the reference | per atom type |
| RP4 | local parameter estimation improves recall at equal proposer | `ransac_local_rv` − `ransac_rv` |
| RP5 | the calibrated validator has a lower false acceptance rate than the old one | study on a common pool |
| RP6 | joint assembly recovers exact rich atoms without degrading precision | `union_joint` − `union_rv` |
| RP7 | directed active measurements beat random measurements | equivalence, same budget |
| RP8 | downstream gain: candidate structure and final recovery higher than the reference | paired differences per task, 95% CI above 0 |

Expectation declared from the tuning (27 scenes): RP2 likely; RP1 uncertain (recall close to the reference);
RP3 uncertain; RP8 unknown.

Verdict rule fixed in advance:

- `PRIMITIVE DISCOVERY READY FOR END-TO-END VALIDATION`: RP1, RP2 and RP8 supported.
- `PRIMITIVE DISCOVERY IMPROVED BUT INSUFFICIENT`: RP1 or RP2 supported, but not RP8, or equivalence recall ≤ 0.56.
- `PRIMITIVE DISCOVERY STILL BLOCKING`: neither RP1 nor RP2.

## 1. Protocol executed

- AWS run `wce-deeprp-20261004-194145` (EC2 + SSM, 2 × c6i.xlarge, 49 min each, 2,608 sub-units, 0 errors,
  13,714 lines). Commands: `make deep-run-rp DEEP_RP_KEYS=0-15,400-415`, `make deep-fetch`, `make deep-rp-aggregate`,
  `make deep-rp-report`. Summary: `results/deep_robust_primitive_summary.json`.
- Atomic bench: 288 never-seen scenes (9 categories × 32), 300 observations each. Downstream chain: 128 never-used
  composition tasks (depths 2 to 5, standard and rich regimes).
- Tuning: `make deep-rp-tune` on 27 tuning scenes → `results/deep_configs/robust_validator.json`
  (all data, contrast threshold 10.88 calibrated on random labels, support ≥ 5, purity ≥ 0.55,
  2 segments, backward elimination at 3).
- Tests: 79 new (`make deep-rp-test`), full suite 4,710 tests OK, 7 skipped; sealed set intact.
- Deviation from the protocol: none. Limit: a single seed per scene, no confirmatory test.

## 2. Main results (atomic bench, 288 scenes)

| Arm | Exact | Equivalence | Precision | False positives / scene | Retained | Fit-equiv. | Seconds |
|---|---|---|---|---|---|---|---|
| `gradient` (reference) | 0.33 | **0.61** | 0.39 | 1.39 | 2.13 | 38.8 | 1.07 |
| `gradient_min` (minimal basis, old validator) | 0.33 | 0.34 | 0.52 | 0.90 | 1.62 | 13.7 | 0.33 |
| `gradient_rv` (gradient + calibrated validator) | 0.27 | 0.45 | 0.78 | 0.16 | 0.68 | 26.8 | 0.79 |
| `ransac_rv` | 0.30 | 0.53 | 0.73 | 0.30 | 1.27 | 0 | 0.14 |
| `ransac_local_rv` | 0.30 | 0.54 | 0.72 | 0.33 | 1.33 | 0 | 0.22 |
| `hough_rv` | 0.25 | 0.46 | 0.77 | 0.25 | 1.13 | 8.6 | 0.25 |
| `grid_rv` | 0.24 | 0.45 | 0.74 | 0.26 | 1.08 | 12.2 | 0.27 |
| `neural_rv` | 0.03 | 0.06 | 0.97 | 0.00 | 0.10 | 0 | 0.04 |
| `union_old` (new proposers, old validator) | 0.34 | 0.58 | 0.30 | 3.05 | 4.38 | 19.5 | 0.65 |
| `union_rv` | 0.32 | 0.57 | 0.74 | 0.33 | 1.40 | 8.8 | 0.55 |
| **`union_joint`** (main arm) | **0.44** | 0.57 | **0.74** | **0.35** | 1.74 | 9.0 | 0.57 |
| `union_all_joint` (+ grid, + minimal gradient) | 0.45 | 0.58 | 0.72 | 0.38 | 1.78 | 34.0 | 1.23 |
| `union_neural_joint` | 0.45 | 0.58 | 0.75 | 0.35 | 1.76 | 8.9 | 0.69 |
| `union_stab` (+ stability) | 0.44 | 0.57 | 0.74 | 0.35 | 1.75 | 9.0 | 10.58 |
| `union_holdout` (split by points) | 0.10 | 0.16 | 0.85 | 0.04 | 0.32 | 8.6 | 0.48 |
| `union_pairsplit` (split by pairs) | 0.22 | 0.29 | 0.84 | 0.07 | 0.69 | 8.7 | 0.53 |

Paired differences `union_joint` − `gradient`, 95% bootstrap CI over 288 scenes:

- equivalence: −0.04 [−0.09, +0.01];
- precision: +0.33 [+0.27, +0.38];
- false positives per scene: −1.03 [−1.17, −0.90] (ratio 0.25).

Equivalence recall per atom type (reference → `union_joint`): half-plane 0.76 → 0.59; band 0.52 → 0.42;
disc 0.59 → 0.66; anti-disc 0.45 → 0.47; ring 0.43 → 0.46; sector 0.12 → 0.00.

Per category (equivalence, reference → `union_joint`): single 0.97 → 0.84; two 0.33 → 0.41; multi 0.18 → 0.13;
close 0.47 → 0.63; nested 0.69 → 0.70; redundant 0.97 → 0.88; noise 0.59 → 0.22; outliers 0.72 → 0.72;
partial 0.59 → 0.63.

### Preregistered hypotheses

| # | Result |
|---|---|
| RP1 higher equivalence recall | **not supported**: −0.04 [−0.09, +0.01] |
| RP2 false positives halved | **supported**: 1.39 → 0.35, CI below 0 |
| RP3 rings and sectors | **not supported**: rings 0.43 → 0.46, sectors 0.12 → 0.00 |
| RP4 local estimation | **not supported**: 0.53 → 0.54 at equal proposer |
| RP5 false acceptances of the validator | **supported**: 16.9% → 1.5%, at the cost of true acceptances (72.6% → 31.6%) |
| RP6 joint assembly | **supported**: exact 0.32 → 0.44, precision unchanged (0.74) |
| RP7 active measurements | **not supported**: directed 0.42, random 0.38, no measurement 0.43 (48 scenes, no CI) |
| RP8 downstream gain | **not supported** (see question 9) |

## 3. Audit of the reference (646 true boundary components)

| Class | `gradient` | `union_joint` |
|---|---|---|
| found | 54.8% | 48.1% |
| equivalent not recognised by the metric | 6.0% | 2.8% |
| true one dominated by another candidate | 5.0% | 5.7% |
| true one rejected by the validator | 9.9% | **32.2%** |
| distant parameters | **18.0%** | 9.9% |
| wrong type | 0.6% | 0.9% |
| never proposed | 5.7% | 0.3% |

False positives of the reference: 1.39 per scene, of which 12% are close to a true boundary; they are mostly
sectors (184 out of 400), rings (63) and bands (63) — the rich atoms with many free parameters.
False positives of `union_joint`: 0.35 per scene, 46% close to a true boundary, almost all of them discs.

Per type, for the reference: sectors are found at 22% (27% distant parameters, 21% rejected, 17% not
recognised); anti-discs are never proposed in 18% of cases.

## 4. Validator: same pool of 21,399 candidates

| Validator | True acceptances | False acceptances | Precision of acceptances |
|---|---|---|---|
| old | 72.6% | 16.9% | 0.41 |
| all data, nominal threshold 3 | 94.2% | 58.9% | 0.20 |
| out-of-sample (points), nominal threshold 3 | 71.9% | 22.0% | 0.34 |
| out-of-sample, calibrated threshold | 41.8% | 4.7% | 0.59 |
| all data, calibrated threshold | 33.8% | 1.7% | 0.76 |
| retained configuration | 31.6% | 1.5% | 0.77 |
| + stability | 31.4% | 1.4% | 0.78 (50 times slower) |

The "z" contrast is not a standard normal: neighbouring pairs share points. On random labels,
its maximum reaches 5.6 out-of-sample and about 11 on all the data. A nominal threshold of 3 therefore accepts
primitives with no boundary at all. The split by pairs is not independent (a noisy point is in both
halves); the split by points is, but at 300 observations it no longer leaves enough power (recall 0.16).

## 5. Diversity of the proposers (first round, 646 components)

| Proposer | Recall of the proposals | Unique components | Cost (fit-equiv.) |
|---|---|---|---|
| local consensus (`ransac`) | 0.77 | 0 | 0 |
| + local estimation | 0.78 | 1 | 0 |
| direct gradient (rich) | 0.61 | 25 | 16.0 |
| Hough | 0.52 | 7 | 6.3 |
| geometric (previous) | 0.52 | 2 | 0 |
| grid | 0.48 | 5 | 6.4 |
| neural | 0.44 | 12 | 0 |
| minimal gradient | 0.33 | 0 | 6.0 |
| contrast | 0.20 | 0 | 7.2 |

Union of all: 0.90. Redundant (no unique component, removable): contrast, minimal gradient. The local
consensus is the best single proposer and costs no fit.

## 6. Downstream chain and oracle ceilings (128 composition tasks)

| Literals | Equivalence of the primitives | Precision | Candidate structure ≥ 0.97 | Final ≥ 0.95 | Final ≥ 0.99 | Mean agreement |
|---|---|---|---|---|---|---|
| true atoms | 1.00 | 1.00 | 0.66 | 0.73 | 0.05 | 0.954 |
| true atoms + safeguard | 1.00 | 1.00 | 0.91 | 0.82 | 0.57 | 0.974 |
| true centres and orientations, the rest estimated | 0.37 | 0.43 | 0.45 | 0.46 | 0.03 | 0.908 |
| true types, estimated parameters | 0.18 | 0.23 | 0.29 | 0.30 | 0.03 | 0.881 |
| frozen pipeline (reference) | 0.51 | 0.18 | 0.29 | 0.29 | 0.05 | 0.878 |
| gradient + old validator | 0.19 | 0.20 | 0.22 | 0.27 | 0.02 | 0.854 |
| robust, minimal basis | 0.23 | 0.69 | 0.27 | 0.34 | 0.03 | 0.717 |
| robust + joint assembly | 0.23 | 0.69 | 0.25 | 0.34 | 0.04 | 0.717 |
| robust + assembly + residual rounds | 0.23 | 0.69 | 0.30 | 0.37 | 0.04 | 0.723 |
| robust + assembly + safeguard | 0.23 | 0.69 | 0.20 | 0.30 | 0.05 | 0.715 |

`robust + joint assembly` − reference, paired 95% CI: equivalence of the primitives −0.28 [−0.33, −0.22]; candidate
structure −0.04 [−0.11, +0.02]; mean agreement −0.16 [−0.22, −0.11].

`POST-HOC / EXPLORATORY`: in 26 tasks out of 128 the robust discovery retains no primitive, and the agreement is
then 0 by construction. Excluding these 26 tasks, the mean agreement is 0.900 versus 0.900 for the reference on the same
tasks, and the candidate structure 0.31 versus 0.33.

## 7. Answers

**1. Why does the direct gradient remain the best baseline?**
Its recall (0.61) rests on two measured things. It is the only proposer to bring components that the others
do not reach (25 unique), because it fits the target and not the boundary. And its validator is permissive:
it accepts 73% of the true candidates, but also 17% of the false ones. The price is 1.39 false positives per scene and a precision
of 0.39. No arm exceeds its recall; `union_joint` equals it within uncertainty (−0.04 [−0.09, +0.01]) with
four times fewer false positives and four times fewer fit-equivalents (wall time halved).

**2. Why do rings and sectors fail?**
Rings: the inner circle is small, so few pairs carry it; it is proposed but rejected (22%) or estimated
too far away (18%). The recall stays at 0.46. Joint assembly raises the exact recovery of the ring from 0.00
to 0.43 when its two circles are found.
Sectors: a sector is two half-lines, that is, half the contrast of a line. With the calibrated threshold, 61% of
their components are rejected and no sector is recovered (0.00, versus 0.12 for the reference). This is a
clear regression of the new validator on this type.

**3. Is the minimal basis superior to the rich basis?**
No, not as a gradient proposer: 0.34 versus 0.61 in equivalence. Yes as a hypothesis space for the
geometric proposers: the local consensus, which proposes only lines and circles, reaches 0.77 proposal
recall versus 0.61, without fitting. Downstream, minimal basis alone and minimal basis + assembled atoms give the
same result (agreement 0.717). The result of the previous sub-milestone ("minimal ≈ rich") therefore does not reproduce for
the gradient on these 288 scenes.

**4. Can the validator strongly reduce false positives?**
Yes: false acceptances from 16.9% to 1.5% on the same pool, retained false positives from 1.39 to 0.35 per scene. But
it also rejects two thirds of the true candidates, and 32% of the true components are now lost by rejection
(versus 10%). At 300 observations, one cannot have both a calibrated validator and the recall of the reference.
Purity, multi-region coherence and stability add almost nothing to the calibrated threshold.

**5. Do hypothesis sets help?**
Not measurable as a gain. `CompatiblePrimitiveHypotheses` keeps the assembled atom in competition with its parts, and the
transmitted literals contain both; downstream, this changes nothing (0.717 in both cases). The object exists and
is tested, its usefulness is not demonstrated.

**6. Does joint fitting help?**
Partially. It recovers exact rich atoms (exact 0.32 → 0.44, bands 0.00 → 0.42, rings 0.00 → 0.43)
at no cost in precision. It changes neither the equivalence recall (0.57) nor the downstream result. The geometric
kinship rule (threshold lowered for a concentric circle or a parallel line) was not isolated by a dedicated arm.

**7. Do active measurements become useful?**
No. With 24 measurements: directed 0.42, random 0.38, none 0.43. Neither beats the absence of measurement;
48 scenes, no interval. They are not integrated.

**8. Does a neural component finally bring something?**
No. Alone, with the calibrated validator, it recovers 6% of the atoms. Added to the union, it gives +0.01 in equivalence
(0.57 → 0.58), within the noise. It brings 12 unique components at proposal, which do not survive validation.
Abandoned for this sub-milestone; no retraining was done.

**9. Does the primitive gain translate into a structural gain?**
No. The gain obtained is a gain in precision, not in recall, and it is not transmitted: candidate structure 0.25
versus 0.29, mean final agreement 0.72 versus 0.88. On the composition tasks, the robust discovery recovers
only 23% of the primitives (the boundaries there are partially occluded by the AND / OR, hence below the calibrated threshold) and
retains nothing in 20% of the tasks. The frozen pipeline, with its 84 low-precision literals, does better: composition
tolerates false literals, not missing literals.

**10. What is the new bottleneck?**
Parameter estimation under occlusion, at a fixed number of observations. The oracle ceilings show it: with the
true atoms, the candidate structure is reached in 66% of the tasks (91% with safeguard); with the true types
but parameters to estimate, it falls back to 29%, exactly the level of the reference; additionally giving the centres and
orientations only brings it back up to 45%. Knowing the type is of no use if the parameters are not found. Second
bottleneck, already known: even with the true atoms, recovery at 0.99 is only 5% without safeguard and 57% with.

## 8. Limits

- Dev only, one seed per scene, no confirmation. The thresholds were tuned on 27 scenes.
- The recall of the reference is 0.61 here versus 0.56 at the previous sub-milestone: different scenes, not an improvement.
- The equivalence metric has fixed tolerances; 3 to 6% of the components are functionally recovered without
  being recognised.
- The "true types" and "centres and orientations" ceilings estimate by contrast from generic
  initialisations; a better estimator would move these ceilings.
- Local estimation keeps a true primitive in 86 to 96% of cases in tuning, but does not improve
  end-to-end recall (RP4): the gain is absorbed by the validator's rejection.
- Active measurements and neural component: light evaluations, sufficient to justify not integrating them, not
  sufficient to conclude that they are useless in general.

## 9. Verdict

Preregistered rule: RP2 supported, RP1 and RP8 not supported.

`PRIMITIVE DISCOVERY IMPROVED BUT INSUFFICIENT`

The improvement is real but narrow: same primitives recovered, four times fewer false ones, four times fewer
fit-equivalents, exact rich atoms by assembly. It does not exceed the recall of 0.56–0.61, regresses on sectors
and noisy scenes, and degrades the downstream. The frozen pipeline remains the reference; nothing from this sub-milestone is plugged into it.

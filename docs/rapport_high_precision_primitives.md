# WCE High-Precision Primitive Estimation — report (dev, exploratory)

Status: exploratory sub-milestone, **dev** partition only. No multi-seed campaign, sealed set never
touched. The frozen pipeline remains the official reference; `flat_scene_k3` remains *confirmed dev baseline candidate*;
nothing is wired in by default.

## 0. Preregistration (written before the AWS runs were launched)

### 0.1 What tuning showed (keys 1090–1095 only, small sample sizes, `POST-HOC / EXPLORATORY`)

These observations decided the design. They concern 24 to 36 tuning tasks and are not
results; they are taken up again, at scale, by the preregistered experiments below.

1. **Local objectives on a single primitive are not aligned with the function.** Starting from the TRUE
   primitive and climbing on the contrast of neighbouring pairs, one moves away from it (median relative error 0 → 0.10 at
   high visibility, 0 → 0.37 at low visibility); the objective found exceeds that of the truth in more than 90%
   of cases. Same observation for an error floor on a spatial grid.
2. **The joint "per-cell floor" objective is aligned.** True atoms perturbed by 0.10 are brought back to
   a median relative error of about 0.05 up to four atoms. At six atoms and 300 observations, it
   overfits (64 cells): an out-of-sample validation is necessary.
3. **A direct greedy search on the floor, without a pool, fails** from two atoms onwards (the first atom chosen
   approximates the whole composition). Abandoned.
4. **The extraction of boundary curves is the weak link, and it is algorithmic.** With 3,000 observations
   without noise, the existing lines and circles recover only 35 to 50% of the true curves at six atoms. A
   voting extractor with normals and sequential consensus rises to 0.91–1.00 (lines) and 0.73–0.80 (circles) at
   two and four atoms in this condition. At the densities of the tasks (300 and 600 noisy observations), the two
   extractors are at the same level, low (0.1 to 0.5).
5. **On 24 standard tuning tasks, the new chain does no better than the same downstream without it**
   (agreement 0.87 against 0.88). On 9 dense tasks, atomic precision rises markedly (atoms at less than 0.05
   relative error: 0.33 → 0.70) without visible gain on the final agreement.

### 0.2 Primary method: chain C

`observations → pool → compatible sets → validated improvement → same strong downstream`

- **Pool, recall first**: existing pool (five parameterisations per type, consensus components) AND
  family-specialised estimators — lines and circles by votes with local normals, band adapted to
  the spacing of the observations, sequential consensus; then assembly: half-plane = one line; disc and
  anti-disc = one circle; band = two parallel lines with shared orientation; ring = two circles with shared
  centre; sector = two half-lines with the same vertex, each line having to carry elements on the side of its
  half-line. Budget: 12 lines, 10 circles, five hypotheses at most per composite type.
- **Validated improvement of the three best sets**: local joint refinement on the per-cell floor
  (initial point kept), then at most two primitive births by conditional floor gain, oriented by
  the morphology of the residual (linear, arc, parallel pair, concentric pair, radial), then conditional pruning.
  Validated Transformation Principle: the whole procedure is replayed on three subsamples and judged on the
  observations it has not seen; it is retained only if it corrects significantly more errors there
  than it creates (paired test, z ≥ 1.64); otherwise the initial set is returned as is. This threshold was added
  before the launch: without it, the procedure was accepted on random labels (unit test).
- The improved primitives **join** the pool; the strong downstream (sets → composition on equivalence
  classes → validated finalisation), identical for A, B and C, chooses.

Frozen configuration: `results/deep_configs/hp.json`. No learned component is introduced: no precise problem
was identified that the explicit estimators could not attack, and the previous neural proposer contributed
nothing out of distribution. The sector of this world is an unbounded corner (vertex, direction, half-aperture): it has neither
radius nor arc.

### 0.3 Data

Keys **1100–1131**, never used. Depths 1 to 6. Regimes `standard` (300 observations, 5%) and `rich`
(600, 2%): **primary** regimes. Regime `dense` (2,400 observations, 2%), same worlds: **secondary** regime,
to separate lack of information from algorithmic defect. Regime `hard` (160, 10%), never used in
tuning: **out-of-distribution** condition. Per-primitive bench: new occluded scenes (indices 21100–21131).

### 0.4 Experiments and arms

| Experiment | Content |
|---|---|
| `chain` | A atoms of the frozen pipeline; B `flat_scene_k3`; C — same strong downstream. Preregistered secondary: B5. Variants of C: pool only (`C_pool`), without specialised estimators, without refinement, without birth, optimistic validation (`C_cv`), `k = 1`, `k = 3`, independent curves (`C_nojoint`). Historical (standard, rich): `A_full`, `B_full`, `C_prev` (chain of the previous sub-milestone). |
| `single` | One primitive, known type: 6 types × density (150, 300, 600, 2,400) × noise (0; 5%) × occlusion (0; 0.3; 0.6). Methods: generic gradient on the target, generic gradient on the contrast, existing inference, specialised curves, independent curves (band, ring), direct search of the whole primitive, refinement, consensus (median of resamplings), partial oracles (centre, orientation, size), refinement from the truth (information floor), identifiability of the true primitive. |
| `sens` | One parameter of a true atom perturbed by 0.02 / 0.05 / 0.10 / 0.20, the other atoms exact, without finalisation: functional degradation of the atom, then final recovery. |
| `birth` | One true atom removed (exact context, and context perturbed by 0.05): new birth, without morphology, not validated, against the old search on the residual; false births on complete sets; three measures of "recovered", one of which is independent of polarity. |
| `ood` | B, C, `C_pool` at the `hard` regime. |

No other arm will be added. No `k` other than 1, 3, 5.

### 0.5 Metrics

- **Individual recall** (continuity with the previous sub-milestone): true primitive reproduced at 0.95 functional
  agreement by a hypothesis of the pool.
- **Precision**: RELATIVE functional error (disagreement relative to the area of the primitive, free polarity) ≤ 0.10
  ("precise") and ≤ 0.05 ("fine"); errors per parameter family; share of atoms within the tolerance of their
  family.
- **Set recall**: all true primitives present (0.95), by number of atoms; precise version
  (all at ≤ 0.10).
- Candidate structure ≥ 0.97; final agreement; final ≥ 0.99; agreement in extrapolation.
- Cost: time per task, floor evaluations, hypotheses, maximum resident memory, AWS cost.

**Tolerance per family** (mechanical rule, fixed here): for each type and parameter family, the tolerance is
the largest perturbation tested such that, for it and for all smaller ones, the final recovery
≥ 0.99 remains at least equal to 80% of that without perturbation (tasks whose unperturbed true atoms give a
functional structure); "< 0.02" if none. Computed on the `sens` experiment alone; no choice of method
depends on it.

Paired differences per task, 95% bootstrap CI. Primary regimes pooled unless stated otherwise.

### 0.6 Hypotheses

| # | Hypothesis | Supported if | Refuted if |
|---|---|---|---|
| HP1 | C increases individual recall | C − B and C − A, CI > 0, AND recall of C at depths 4–6 ≥ 0.70 | one CI includes 0, or recall d4–6 < 0.60 |
| HP2 | C increases precision | share of precise atoms (≤ 0.10): C − B and C − A, CI > 0 | one CI includes 0 |
| HP3 | C increases set recall | all present: C − B and C − A, CI > 0, AND ≥ 0.25 at depths 4–6 | one CI includes 0, or < 0.10 at d4–6 |
| HP4 | C improves the candidate structure | candidate ≥ 0.97: C − B and C − A, CI > 0 | one CI includes 0 |
| HP5 | the gain carries through to the final | final agreement AND final ≥ 0.99: C − B and C − A, CI > 0 | one CI includes 0 on one of the two measures |
| HP6 | the specialised estimators beat the generic gradient | `single` bench, precise share at first rank: curves − gradient, CI > 0; reported per type | CI includes 0 or < 0 |
| HP7 | several parameterisations help | bench: truth in the first 5 − at first rank > 0.05; chain: C − `C_k1` on agreement, CI > 0 | one of the two is missing |
| HP8 | consensus reduces catastrophic errors | bench: share of relative errors > 0.25, consensus − refined, CI < 0 | CI includes 0 or > 0 |
| HP9 | validated refinement improves the parameters | bench: relative error refined − curves, CI < 0; chain: C − `C_norefine` on the precise share, CI > 0 | one of the two is missing |
| HP10 | birth recovers the missing primitive | exact context, polarity-independent measure: new − old, CI > 0, AND new ≥ 0.60; false births ≤ 0.10 | one of the three is missing |
| HP11 | recall is limited by information, not by the algorithm | bench without mask, 2,400 observations: precise share of the curves ≥ 0.85 for each type; dense chain: all present ≥ 0.25 at d4–6 | one of the two is missing |
| HP12 | the cost remains compact | C ≤ 10 s per task (primary regimes) and ≤ 3 × B | otherwise |
| HP13 | C does not degrade out of distribution | `hard` regime: C − B on agreement, lower bound of the CI > −0.01 | lower bound ≤ −0.01 |

HP1 to HP5 are judged on the primary regimes. The 0.70 threshold of HP1 expresses "qualitative leap" (the current
level is about 0.37 to 0.45 at depths 4–6); 0.9 remains the order of magnitude aimed at, not a threshold.

**Declared expectations** from tuning: HP1 to HP5 probably not supported at the primary regimes; HP6
uncertain at 300 observations, probable at 2,400; HP10 probable; HP11 uncertain (probable for lines, less so
for circles and the sector); HP12 probable.

### 0.7 Compute budget and stopping criteria

A single AWS run (`make deep-run-hp`, EC2 + SSM, `eu-north-1`, 2 × `c6i.xlarge`, 10 h at most, automatic
termination). Estimate: 20 CPU-hours, i.e. about 3 h of wall-clock time and less than 2 USD. Stop: end of the run; no
relaunch with a modified configuration; if more than 5% of the sub-units fail, fix the defect and relaunch
the whole, reporting it. No iteration on keys 1100–1131.

### 0.8 Verdict rule, fixed in advance

- `READY FOR 16-SEED CONFIRMATORY CAMPAIGN`: HP1, HP2, HP3, HP4, HP5 and HP12 all supported. Only then,
  propose the campaign — without launching it.
- `HIGH-PRECISION PRIMITIVES IMPROVED BUT INSUFFICIENT`: HP1 and HP2 supported (recall AND precision progress
  markedly at the primary regimes), but HP3, HP4 or HP5 not supported.
- `HIGH-PRECISION PRIMITIVES STILL BLOCKING`: HP1 or HP2 not supported.

A gain limited to the dense regime, to the per-primitive bench or to controlled birth does not change the verdict; it is
reported as such. The sealed set is not launched in any case.

## 1. Protocol executed

AWS run `wce-deephp-20261010-092830` (EC2 + SSM, `eu-north-1`, 2 × `c6i.xlarge`): 672 sub-units per instance,
0 failure, 2 h 54 and 2 h 52, instances terminated. 74,512 lines, 0 unit in error
(`results/deep_high_precision_summary.json`). 23 CPU-hours, maximum resident memory 588 MB, estimated cost 1.05 USD
(approximate rate). No arm added, no relaunch. 384 tasks at the primary regimes, 192 at the dense regime, 192 at the
`hard` regime, 4,608 bench scenes, 9,872 perturbations, 320 atom removals. A single task seed per key:
nothing here is confirmatory.

Deviation from the preregistration, to be reported: the arm `new_onebirth` (a single birth, without refinement or
validation) was in the code before the launch but not in table 0.4; it is reported as
`POST-HOC / EXPLORATORY`.

## 2. Complete chains, same strong downstream

Primary regimes (standard and rich, 384 tasks):

| Chain | Agreement | Final ≥ 0.99 | Final ≥ 0.97 | Candidate ≥ 0.97 | Recall | Precise ≤ 0.10 | Fine ≤ 0.05 | All present | Param. error | Hypotheses | Time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A (frozen) | 0.896 | 0.099 | 0.253 | 0.300 | 0.405 | 0.334 | 0.250 | 0.234 | 0.112 | 36.1 | 24.7 s |
| B (`flat_scene_k3`) | 0.892 | 0.096 | 0.266 | 0.255 | 0.433 | 0.321 | 0.180 | 0.208 | 0.059 | 14.1 | 2.5 s |
| B5 | 0.900 | 0.089 | 0.281 | 0.276 | 0.481 | 0.342 | 0.188 | 0.242 | 0.064 | 23.0 | 2.9 s |
| **C** | 0.912 | 0.076 | 0.305 | 0.331 | 0.567 | 0.419 | 0.273 | 0.307 | 0.060 | 44.4 | 4.6 s |
| `C_pool` (pool only) | 0.911 | 0.076 | 0.305 | 0.328 | 0.567 | 0.417 | 0.271 | 0.307 | 0.060 | 44.4 | 3.6 s |
| `C_prev` (previous sub-milestone) | 0.907 | 0.076 | 0.281 | 0.339 | 0.540 | 0.405 | 0.240 | 0.287 | 0.061 | 46.6 | 8.1 s |
| `A_full` | 0.884 | 0.120 | 0.255 | 0.323 | — | — | — | — | — | — | 22.7 s |
| `B_full` | 0.903 | 0.146 | 0.346 | 0.406 | — | — | — | — | — | — | 18.0 s |

Paired differences, primary regimes:

| Difference | Agreement | Final ≥ 0.99 | Candidate ≥ 0.97 | Recall | Precise ≤ 0.10 | All present |
|---|---|---|---|---|---|---|
| C − B | +0.019 [+0.013, +0.025] | −0.021 [−0.042, −0.003] | +0.075 [+0.042, +0.107] | +0.134 [+0.114, +0.155] | +0.098 [+0.079, +0.119] | +0.099 [+0.070, +0.130] |
| C − A | +0.015 [+0.009, +0.021] | −0.023 [−0.047, +0.000] | +0.031 [−0.003, +0.065] | +0.162 [+0.132, +0.192] | +0.086 [+0.060, +0.112] | +0.073 [+0.036, +0.112] |
| C − B5 | +0.012 [+0.007, +0.017] | −0.013 [−0.034, +0.005] | +0.055 [+0.023, +0.086] | +0.086 [+0.070, +0.102] | +0.077 [+0.060, +0.096] | +0.065 [+0.042, +0.091] |
| C − `C_prev` | +0.005 [−0.000, +0.010] | 0.000 | −0.008 [−0.039, +0.023] | +0.027 [+0.012, +0.042] | +0.014 [−0.002, +0.031] | +0.021 [−0.003, +0.044] |
| C − `B_full` | +0.009 [+0.004, +0.014] | −0.070 [−0.099, −0.044] | −0.075 [−0.109, −0.042] | — | — | — |
| C − `A_full` | +0.027 [+0.021, +0.034] | −0.044 [−0.070, −0.021] | +0.008 [−0.029, +0.044] | — | — | — |

Depths 4–6 (192 tasks): recall of C 0.374 (B 0.229; A 0.160); precise 0.236 (B 0.146); all present
0.021 (B 0; A 0); agreement 0.869 (B 0.840; A 0.852), C − B +0.030 [+0.019, +0.040]; final ≥ 0.99: 0 for B and C.

Where the gain of C comes from. Entirely from the specialised estimators: C − `C_nocurves` agreement +0.009 [+0.004, +0.014],
recall +0.063, precise +0.060, all present +0.050. Validated improvement contributes nothing at the primary regimes:
`C_nobirth` is identical to C (no birth retained), C − `C_pool` agreement +0.001 [−0.001, +0.002]; an improved
set is validated in only 4.7% of cases. Optimistic validation (`C_cv`) accepts many more (2.4 sets
per task) and increases the precise share of the pool (+0.017 [+0.011, +0.025]) without effect on agreement.

Dense regime (2,400 observations, 192 tasks):

| Chain | Agreement | Final ≥ 0.99 | Final ≥ 0.97 | Candidate ≥ 0.97 | Recall | Precise ≤ 0.10 | Fine ≤ 0.05 | All present | Param. error | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.938 | 0.260 | 0.484 | 0.526 | 0.482 | 0.447 | 0.390 | 0.240 | 0.079 | 47.3 s |
| B | 0.949 | 0.292 | 0.563 | 0.443 | 0.597 | 0.516 | 0.376 | 0.292 | 0.035 | 5.1 s |
| B5 | 0.952 | 0.271 | 0.568 | 0.505 | 0.653 | 0.551 | 0.387 | 0.349 | 0.041 | 5.5 s |
| **C** | 0.963 | 0.323 | 0.708 | 0.714 | 0.792 | 0.714 | 0.630 | 0.516 | 0.028 | 11.7 s |
| `C_pool` | 0.961 | 0.297 | 0.688 | 0.662 | 0.784 | 0.700 | 0.606 | 0.500 | 0.029 | 7.5 s |

Dense, paired: C − B agreement +0.014 [+0.008, +0.020], candidate +0.271 [+0.203, +0.333], recall +0.195, precise
+0.199, all present +0.224 [+0.167, +0.286], final ≥ 0.97 +0.146 [+0.094, +0.203], final ≥ 0.99 +0.031
[−0.026, +0.088]. Depths 4–6: recall 0.659 (B 0.412), all present 0.198 (B 0), candidate 0.479 (B 0.115),
final ≥ 0.99 0.115 (B 0.031), C − B +0.083 [+0.031, +0.146]. At the dense regime validated improvement acts: C −
`C_pool` candidate +0.052 [+0.021, +0.088], final ≥ 0.99 at d4–6 +0.042 [+0.010, +0.083].

Out of distribution (`hard`, 192 tasks): C − B agreement −0.005 [−0.018, +0.008]; agreement in extrapolation −0.027
[−0.064, +0.010], and −0.060 [−0.116, −0.005] at depths 4–6; recall +0.077 [+0.051, +0.105].

## 3. Set recall by number of atoms

| Atoms | Recall B | All present B | Recall C | Precise C | All present C | `recall^n` C | All precise C | Recall C dense | All present C dense | `recall^n` dense |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.969 | 0.969 | 0.984 | 0.953 | 0.984 | 0.984 | 0.953 | 1.000 | 1.000 | 1.000 |
| 2 | 0.539 | 0.219 | 0.742 | 0.516 | 0.563 | 0.551 | 0.281 | 0.922 | 0.875 | 0.850 |
| 3 | 0.401 | 0.063 | 0.552 | 0.339 | 0.234 | 0.168 | 0.047 | 0.854 | 0.625 | 0.623 |
| 4 | 0.297 | 0.000 | 0.465 | 0.313 | 0.047 | 0.047 | 0.000 | 0.734 | 0.313 | 0.291 |
| 5 | 0.250 | 0.000 | 0.381 | 0.250 | 0.016 | 0.008 | 0.000 | 0.681 | 0.281 | 0.147 |
| 6 | 0.141 | 0.000 | 0.276 | 0.146 | 0.000 | 0.000 | 0.000 | 0.563 | 0.000 | 0.032 |

Set recall still follows `recall^n`. At six atoms, it remains zero even at the dense regime.

Per type (pool, recall / precise ≤ 0.10):

| Type | B primary | C primary | B dense | C dense |
|---|---|---|---|---|
| half-plane | 0.463 / 0.417 | 0.656 / 0.592 | 0.578 / 0.532 | 0.826 / 0.780 |
| band | 0.342 / 0.296 | 0.451 / 0.366 | 0.620 / 0.563 | 0.739 / 0.725 |
| disc | 0.326 / 0.189 | 0.524 / 0.316 | 0.481 / 0.359 | 0.802 / 0.698 |
| anti-disc | 0.281 / 0.081 | 0.475 / 0.169 | 0.400 / 0.250 | 0.750 / 0.600 |
| ring | 0.248 / 0.126 | 0.300 / 0.170 | 0.481 / 0.393 | 0.667 / 0.541 |
| sector | 0.195 / 0.115 | 0.360 / 0.175 | 0.320 / 0.260 | 0.540 / 0.410 |

By visibility (recall): primary C 0.120 / 0.316 / 0.685 (visibility < 0.4 / 0.4–0.8 / > 0.8) against B 0.058 / 0.203 /
0.487; dense C 0.248 / 0.684 / 0.927 against B 0.058 / 0.368 / 0.751. A well-visible atom is recovered 93% of the time
at the dense regime, 68% at the primary regimes.

## 4. Sensitivity map and necessary precisions

One parameter of one atom perturbed, the others exact, without finalisation. Share of tasks keeping final ≥ 0.99
(base: 91.6% of the tasks are functional without perturbation):

| Type | Family | 0.02 | 0.05 | 0.10 | 0.20 | Tolerance |
|---|---|---|---|---|---|---|
| half-plane | orientation | 0.98 | 0.38 | 0.21 | 0.12 | 0.02 |
| half-plane | offset | 0.97 | 0.80 | 0.30 | 0.14 | 0.05 |
| band | orientation | 0.66 | 0.25 | 0.11 | 0.05 | **< 0.02** |
| band | offset | 0.96 | 0.34 | 0.14 | 0.05 | 0.02 |
| band | width | 0.99 | 0.88 | 0.33 | 0.14 | 0.05 |
| disc | centre | 0.98 | 0.93 | 0.47 | 0.13 | 0.05 |
| disc | radius | 0.96 | 0.66 | 0.18 | 0.05 | 0.02 |
| anti-disc | centre | 0.99 | 0.97 | 0.57 | 0.19 | 0.05 |
| anti-disc | radius | 0.99 | 0.76 | 0.29 | 0.11 | 0.02 |
| ring | centre | 0.99 | 0.57 | 0.15 | 0.01 | 0.02 |
| ring | radius (both) | 0.94 | 0.33 | 0.04 | 0.01 | 0.02 |
| ring | width | 0.99 | 0.83 | 0.32 | 0.02 | 0.05 |
| sector | vertex | 0.99 | 0.97 | 0.60 | 0.17 | 0.05 |
| sector | direction | 0.92 | 0.40 | 0.17 | 0.03 | 0.02 |
| sector | half-aperture | 0.93 | 0.38 | 0.19 | 0.05 | 0.02 |

Final ≥ 0.97 tolerates much more: almost all families hold at 0.05, and the centres at 0.10–0.20. The
orientations and directions are the critical parameters (an angle moves the whole boundary, in proportion to
the distance); the centres and the widths are the most tolerant.

Median error of the primitives of the pool, to be compared with the tolerances:

| Family | B primary | C primary | C dense |
|---|---|---|---|
| half-plane: orientation / offset | 0.138 / 0.170 | 0.038 / 0.057 | 0.010 / 0.016 |
| band: orientation / offset / width | 0.142 / 0.321 / 0.237 | 0.040 / 0.175 / 0.207 | 0.010 / 0.020 / 0.026 |
| disc: centre / radius | 0.537 / 0.165 | 0.210 / 0.084 | 0.032 / 0.013 |
| anti-disc: centre / radius | 0.696 / 0.206 | 0.238 / 0.095 | 0.028 / 0.015 |
| ring: centre / radius | 0.661 / 0.399 | 0.471 / 0.329 | 0.039 / 0.031 |
| sector: vertex / direction / half-aperture | 1.596 / 0.708 / 0.241 | 1.049 / 0.427 / 0.193 | 0.696 / 0.211 / 0.092 |

Share of the true atoms having ALL their parameters within tolerance: primary A 0.070, B 0.035, C 0.077; dense A
0.141, B 0.124, C 0.332. Per type for C dense: half-plane 0.66; disc 0.55; anti-disc 0.51; ring 0.33;
sector 0.08; band 0.00 (its orientation tolerance is below the smallest perturbation tested).

## 5. Per-primitive bench (known type, 4,608 scenes)

Share of scenes where the first hypothesis is precise (relative error ≤ 0.10):

| Method | All | No mask | Mask 0.3 | Mask 0.6 | 150 obs. | 300 | 600 | 2,400 | Catastrophic (> 0.25) |
|---|---|---|---|---|---|---|---|---|---|
| generic gradient (target) | 0.313 | 0.82 | 0.11 | 0.00 | 0.27 | 0.31 | 0.32 | 0.35 | 0.556 |
| generic gradient (contrast) | 0.336 | 0.56 | 0.34 | 0.11 | 0.23 | 0.33 | 0.36 | 0.43 | 0.574 |
| existing inference | 0.427 | 0.77 | 0.37 | 0.15 | 0.24 | 0.41 | 0.49 | 0.58 | 0.390 |
| **specialised curves** | 0.498 | 0.77 | 0.52 | 0.21 | 0.22 | 0.39 | 0.58 | 0.81 | 0.359 |
| direct search | 0.370 | 0.91 | 0.20 | 0.00 | 0.28 | 0.37 | 0.41 | 0.43 | 0.511 |
| curves + refinement | 0.404 | 0.87 | 0.28 | 0.06 | 0.27 | 0.40 | 0.45 | 0.50 | 0.440 |
| + consensus | 0.403 | 0.89 | 0.27 | 0.06 | 0.28 | 0.40 | 0.44 | 0.49 | 0.445 |
| refinement from the truth | 0.696 | 0.99 | 0.48 | 0.62 | 0.67 | 0.70 | 0.69 | 0.72 | 0.181 |

Noise and occlusion separated (curves / gradient / direct search): neither 0.81 / 0.84 / 0.94; noise
alone 0.73 / 0.81 / 0.89; occlusion alone 0.38 / 0.06 / 0.10; both 0.34 / 0.06 / 0.09. **Occlusion, not
noise, is what breaks the estimators**; the methods that fit the whole primitive on the labels are
the best without mask and collapse from 30% occlusion onwards.

Without mask, per type and density (curves): half-plane 0.75 / 0.98 / 1.00 / 1.00; band 0.39 / 0.88 / 1.00 / 1.00;
disc 0.47 / 0.83 / 0.97 / 1.00; anti-disc 0.27 / 0.45 / 0.83 / 1.00; ring 0.08 / 0.44 / 0.84 / 1.00; sector
0.53 / 0.83 / 0.97 / 1.00 (150 / 300 / 600 / 2,400 observations). All families are perfectly resolved at
2,400 observations when they are visible; at 300, the ring and the anti-disc are so only one time in two.

Per type and occlusion (curves, mask 0 / 0.3 / 0.6): half-plane 0.93 / 0.80 / 0.50; band 0.82 / 0.66 / 0.35; disc
0.82 / 0.52 / 0.19; anti-disc 0.64 / 0.39 / 0.10; ring 0.59 / 0.38 / 0.09; sector 0.83 / 0.36 / 0.01.

Paired: curves − gradient +0.185 [+0.170, +0.200] (positive for each type, from +0.077 for the ring to +0.378
for the half-plane); curves − existing inference +0.070 [+0.058, +0.083] (disc +0.175; band −0.004, null);
refined − curves −0.094 [−0.106, −0.081] (refinement DEGRADES); consensus − refined: catastrophic +0.005
[+0.001, +0.009]; direct search − curves −0.127; shared constraints − independent curves +0.042
[+0.027, +0.057] (band +0.055, ring +0.030); truth in the first five − at first rank: curves +0.033
[+0.028, +0.038], existing inference +0.057.

Partial oracles (mask 0 / 0.3 / 0.6): disc with true centre 0.92 / 0.83 / 0.75 (without: 0.82 / 0.52 / 0.19);
anti-disc 0.84 / 0.75 / 0.66; sector with true vertex 0.82 / 0.69 / 0.07; half-plane with true orientation
1.00 / 0.96 / 0.85; band with true orientation 0.85 / 0.77 / 0.61. The limiting parameter under occlusion is the
**centre** for circular primitives, the **vertex** for the sector, the **orientation** for rectilinear
primitives.

Identifiability (diagnostic `PrimitiveIdentifiability` on the true primitive, without mask): identifiable in
0.87 (half-plane), 0.90 (band), 0.77 (disc), 0.72 (anti-disc), 0.90 (ring), 0.68 (sector) of the scenes. When
the primitive is said to be identifiable, the curves are precise in 0.98 / 0.87 / 0.92 / 0.78 / 0.65 / 0.91 of cases;
when it is said to be non-identifiable, 0.63 / 0.34 / 0.21 / 0.24 / 0.08 / 0.11. Under occlusion this diagnostic is NOT
valid: it reasons on the primitive alone, without the mask, and declares almost everything non-identifiable.

## 6. Birth of a missing primitive

One true atom removed, the others exact (320 tasks):

| Arm | Born | Recovered (free polarity) | Recovered ≤ 0.10 | Final ≥ 0.99 after | Agreement after | Time |
|---|---|---|---|---|---|---|
| before | — | — | — | 0.031 | 0.822 | — |
| old search on residual | 16.7 hypotheses | 0.459 | 0.247 | 0.178 | 0.902 | 3.4 s |
| new, validated (preregistered arm) | 0.79 | 0.484 | 0.328 | 0.172 | 0.942 | 2.5 s |
| new without morphology | 0.78 | 0.466 | 0.316 | 0.181 | 0.942 | 2.2 s |
| new not validated | 0.97 | 0.556 | 0.356 | 0.175 | 0.947 | 0.9 s |
| a single birth, neither refinement nor validation (`POST-HOC`) | 0.95 | 0.719 | 0.519 | 0.522 | 0.973 | 0.6 s |

Paired against the old one: new validated, free polarity +0.025 [−0.038, +0.084], precise +0.081
[+0.025, +0.138], agreement +0.040 [+0.018, +0.065]; not validated +0.097 [+0.034, +0.156]; a single birth
+0.259 [+0.206, +0.316], final ≥ 0.99 +0.344 [+0.288, +0.400]. Morphology: new − without morphology +0.019
[−0.009, +0.047], null; the classifier answers "arc" for almost all missing types (65 bands out of 84, 41
half-planes out of 52): it is not informative.

By depth (new validated, free polarity): 0.81 / 0.50 / 0.48 / 0.34 / 0.28 from two to six atoms. Context
perturbed by 0.05: new 0.447, old 0.394, and final ≥ 0.99 almost zero for all (0.04 at best):
birth does not compensate for imprecise neighbours. False births on 384 complete sets: **0**. Parameter
errors of the primitives born and recovered: centre 0.03–0.05, radius 0.02, orientation 0.015–0.03, offset
0.03–0.04; sector: vertex 0.23.

**Anti-disc, the `0/42` of the previous sub-milestone: it was a MEASUREMENT defect.** Here again, the old measure
(functional agreement without freedom of polarity, after removal of the negation) gives 0 out of 24 for the old method
as for the new one. With free polarity: 0.542 (old) and 0.417 (new validated), 0.542 not validated.
The disc with the same boundary is indeed found; it is merged with its complement into a single hypothesis
represented by the disc, and the old measure compared an anti-disc with a disc. None of the six causes
considered (boundary, objective, mask, initialisation, score) is at play; only the polarity of the measure
was. The final ≥ 0.99 after birth of an anti-disc (0.42–0.46) is moreover the highest of all types.

## 7. Preregistered hypotheses

| # | Result | Status |
|---|---|---|
| HP1 | C − B +0.134, C − A +0.162, CI > 0; but recall at d4–6 = 0.374 (threshold 0.70; refutation < 0.60) | **refuted** (real gain, qualitative leap absent) |
| HP2 | precise: C − B +0.098 [+0.079, +0.119]; C − A +0.086 [+0.060, +0.112] | supported |
| HP3 | all present: C − B +0.099, C − A +0.073, CI > 0; but 0.021 at d4–6 (threshold 0.25; refutation < 0.10) | **refuted** |
| HP4 | candidate: C − B +0.075, CI > 0; C − A +0.031 [−0.003, +0.065] | **not supported** |
| HP5 | agreement: CI > 0 against A and B; final ≥ 0.99: C − B −0.021 [−0.042, −0.003] | **refuted** |
| HP6 | curves − gradient +0.185 [+0.170, +0.200], positive for each type | supported |
| HP7 | bench: +0.033 (< 0.05); chain: C − `C_k1` agreement +0.003 [−0.002, +0.008] | **refuted** |
| HP8 | catastrophic, consensus − refined +0.005 [+0.001, +0.009] | **refuted** |
| HP9 | bench: refined − curves, error +0.029 [+0.019, +0.038]; chain: +0.002 [0, +0.005] | **refuted** |
| HP10 | new − old +0.025 [−0.038, +0.084]; new 0.484 (< 0.60); false births 0 | **refuted** |
| HP11 | bench without mask at 2,400 observations: 1.00 for each type; dense chain, all present at d4–6: 0.198 (< 0.25) | **not supported** (first half only) |
| HP12 | C 4.6 s (≤ 10 s); 1.9 × B (≤ 3 ×) | supported |
| HP13 | `hard`: C − B agreement −0.005, lower bound −0.018 (≤ −0.01) | **refuted** |

Declared expectations: HP1 to HP5 probably not supported — confirmed, except HP2, supported; HP6 confirmed; HP10
contradicted; HP11 by half; HP12 confirmed.

## 8. Answers to the ten questions

**1. Which type of primitive is mainly responsible for the insufficient recall?** No single one: all are below 0.70
at the primary regimes. The weakest are the ring (0.30) and the sector (0.36), then the band (0.45) and
the anti-disc (0.48); the half-plane is the best (0.66). In precision, ring, anti-disc and sector are at
0.17. The two-curve primitives (ring, band, sector) pay for the extraction twice.

**2. Which parameters are the most difficult?** To estimate: the vertex of the sector (median error 1.05, still
0.70 at the dense regime), the centre of circular primitives (0.21 to 0.47), then the radii of the ring. To tolerate:
the orientations and directions, which break the composition from 0.05 onwards. Under occlusion, the partial oracle shows that the
centre (circles), the vertex (sector) and the orientation (lines) are the limiting parameter.

**3. Which precisions are necessary per family?** To keep final ≥ 0.99: orientations, directions,
half-aperture, radii, offset of a band: 0.02; centres of disc and anti-disc, vertex of sector,
widths, offset of a half-plane: 0.05; orientation of a band: less than 0.02; centre of a ring: 0.02.
For final ≥ 0.97, about 0.05 everywhere and 0.10 to 0.20 for the centres. The median errors of C at the primary
regimes are two to twenty times above; at the dense regime they pass below the tolerance for the lines and
the circles, not for the sector.

**4. Do the specialised estimators beat the generic gradient?** Yes on average (+0.185) and for each type,
and they beat the existing inference (+0.070). But the picture reverses depending on the condition: without mask, the
gradient (0.82) and the direct search (0.91) do better than the curves (0.77); from 30% occlusion onwards they fall
to 0.11–0.20 while the curves hold at 0.52. In composition, everything is occluded: it is the curves that count.

**5. Does parametric top-k help?** Little. Bench: +0.033 of truth between the first and the fifth rank. Chain:
`k = 5` against `k = 1`, recall +0.050 and all present +0.034, but agreement +0.003 not significant.

**6. Does consensus improve robustness?** The SEQUENTIAL consensus of the extraction, yes: it is what carries
the gain of the curves. The median of resamplings, no: it adds 0.5 point of catastrophic errors.

**7. Does the search on the residual really recover the missing primitives?** Partially, and no better
than before in its preregistered form (0.48 against 0.46, not significant), although more precise (+0.081) and without
any false birth. The simplest form — one birth, without validation or refinement — recovers 0.72 and
brings final ≥ 0.99 from 0.03 to 0.52; this result is post-hoc and to be confirmed. In the real chain, no
birth is retained: the out-of-sample validation is too severe at 300–600 observations, and the context there
is imprecise.

**8. Are the sector and the anti-disc better understood?** The anti-disc, yes: its total failure was a polarity
measurement defect, and it behaves like a disc (a little less well recovered, 0.48 against 0.52). The sector,
partially: perfectly resolved when it is visible and dense (1.00 at 2,400 observations), it collapses under
occlusion (0.36 then 0.01) because the vertex is known only through the intersection of two short half-lines;
with the true vertex it rises back to 0.69 at 30% occlusion. It remains the least well recovered type at the dense regime
(0.54) and the only one whose parameters remain out of tolerance there.

**9. Does individual recall approach a level compatible with six atoms?** No at the primary regimes: 0.57 on
average, 0.37 at depths 4–6, 0.28 at six atoms. At the dense regime: 0.79 on average, 0.66 at depths 4–6,
0.56 at six atoms, and 0.93 for well-visible atoms. The order of magnitude of 0.9 is reached only for
visible atoms, with eight times more observations.

**10. Does the gain really carry through to the final composition?** Only in part. At the primary regimes,
agreement rises (+0.019 against B, +0.015 against A) and so does the candidate structure against B, but final ≥ 0.99
falls (−0.021). At the dense regime, the candidate structure ≥ 0.97 goes from 0.44 to 0.71 and final ≥ 0.97 from 0.56 to
0.71, but final ≥ 0.99 only from 0.29 to 0.32 (not significant; +0.083 at depths 4–6). The right
structure is often among the candidates without being the one that is chosen or finalised.

## 9. Limits

- Dev partition, one task seed per key; 32 tasks per depth at the dense regime.
- HP1 and HP3 are refuted on their level threshold, not on the existence of a gain: the four CIs are above
  zero. The rule written in advance is applied as is.
- The dense regime was not used for tuning the chain, but the tuning diagnostic at 3,000
  observations motivated the extractor; its results are secondary.
- Tolerances measured with ONE parameter perturbed and the other atoms exact: they bound the need from below;
  several simultaneous errors accumulate. The smallest perturbation tested is 0.02.
- "Precise ≤ 0.10" is an error relative to the area of the primitive, more severe than the 0.95 functional agreement
  of the previous sub-milestone; both are reported.
- The identifiability diagnostic is valid only without mask.
- The refinement and the consensus of the bench fit the primitive ALONE on the labels: under occlusion this criterion
  is false by construction, which explains part of their degradation; even refinement from the truth falls
  there to 0.48–0.62.
- The classifier of the morphology of the residual is not informative; the "structure of the residual" is therefore not exploited
  as planned.
- The "a single birth" arm is post-hoc.
- Approximate AWS rates.
- No learned component tested.

## 10. New bottleneck

Three findings emerge, in order of importance.

1. **Recall per atom is limited by the quantity of information per visible boundary, and by occlusion.** At
   2,400 observations and without mask, each family is resolved at 100%; at 300 observations or at 60%
   occlusion, none is. The primary regimes (300 and 600 observations for up to six atoms) fall short
   of what the geometric estimators can resolve.
2. **Even when the atoms are there, final ≥ 0.99 does not follow.** At the dense regime, half of the tasks have all
   their primitives and 71% have a candidate structure ≥ 0.97, but only 32% finish at 0.99: the
   tolerances of 0.02 on the angles and the radii are met by only a third of the atoms. The bottleneck moves
   towards the **joint final precision** and the **selection** of the right structure among the candidates.
3. **Out-of-sample validation neutralises the improvement at the primary regimes** (4.7% acceptance), whereas
   an unvalidated birth takes final ≥ 0.99 from 0.03 to 0.52 in exact context. The power / safety
   trade-off of the validation is poorly set for 300 observations.

## 11. Verdict

`HIGH-PRECISION PRIMITIVES STILL BLOCKING`

Preregistered rule: HP1 not supported (recall of 0.37 at depths 4–6, for a threshold of 0.70). The gain is
real and significant on recall, precision, set recall and agreement, against the frozen pipeline as well as
against `flat_scene_k3`, but it is not the qualitative leap aimed at, and final ≥ 0.99 retreats at the primary regimes.
No 16-seed campaign proposed; sealed set intact; nothing wired in by default; `flat_scene_k3` remains *confirmed
dev baseline candidate*.

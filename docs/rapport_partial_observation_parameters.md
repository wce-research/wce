# WCE Partial-Observation Primitive Parameter Inference — report (dev, exploratory)

Status: exploratory sub-milestone, **dev** partition only. No confirmatory test, no 16-seed campaign,
sealed set never touched. The frozen pipeline remains the reference; nothing from this sub-milestone is plugged into it by default.

## 0. Preregistration (written before launching the AWS runs)

Tuning done only on the scenes of the sub-range dev + 350,000 (18 to 36 scenes per condition) and on
composition task 599. Evaluation on never-seen scenes (sub-range dev + 300,000: 6 types × 5 levels of
occlusion × 2 levels of noise × 32 primitives = 1,920 scenes, paired by primitive) and on 128 never-used
composition tasks (keys 500–515).

Main method: `arc_visible` (local consensus, geometric arc estimator, shared constraints,
occlusion-tolerant score). Reference: `global_contrast` (global optimisation by contrast, the one of the "true
types" ceiling of the previous sub-milestone). Downstream: `po_type_k3` against `types_old` and against the frozen pipeline.

| # | Hypothesis | Criterion |
|---|---|---|
| PO1 | inference on fragments recovers more primitives than global optimisation, type known | `recovered_1`, paired difference, 95% CI > 0 |
| PO2 | the gain persists under occlusion ≥ 40% | same, occluded subset |
| PO3 | the geometric estimator beats Kåsa on partial fragments | `arc_geometric` − `arc_kasa`, CI > 0 |
| PO4 | shared constraints help rich primitives | `arc_joint` − `arc_geometric`, ring/band/sector subset, CI > 0 |
| PO5 | the occlusion-tolerant score helps | `arc_visible` − `arc_joint`, CI > 0 |
| PO6 | the composition context (inferred mask, counterfactual rendering) improves top-1 | `ctx_joint` − `arc_visible`, CI > 0 |
| PO7 | the top-5 contains the truth clearly more often than the top-1 | `recovered_5` − `recovered_1` ≥ 0.10 |
| PO8 | the parameter regions cover the truth in ≥ 80% of cases | coverage |
| PO9 | active measurements directed at the parameters beat random | paired difference, CI > 0 |
| PO10 | downstream, type known: candidate structure above the old estimation | `po_type_k3` − `types_old`, CI > 0 |
| PO11 | downstream: final agreement above the frozen pipeline | `po_type_k3` − `reference` on `final_full`, CI > 0 |
| PO12 | the primitive ↔ composition exchange improves the final agreement | `loop_full` − `final_full`, CI > 0 |

Expectation declared from the tuning: PO1, PO2, PO7 likely; PO6 likely at 40% occlusion; sectors
expected weak; PO10–PO12 unknown.

Verdict rule fixed in advance:

- `PARAMETER INFERENCE READY FOR END-TO-END VALIDATION`: PO1, PO10 and PO11 supported.
- `PARAMETER INFERENCE IMPROVED BUT INSUFFICIENT`: PO1 supported, but PO10 or PO11 not supported.
- `PARAMETER INFERENCE STILL BLOCKING`: PO1 not supported.

## 1. Protocol executed

- AWS run `wce-deeppo-20261005-005310` (EC2 + SSM, 2 × c6i.xlarge, 57 and 60 min, 3,072 sub-units, 0 errors,
  29,568 rows). Commands: `make deep-run-po`, `make deep-fetch`, `make deep-po-aggregate`, `make deep-po-report`.
  Summary: `results/deep_partial_observation_summary.json`.
- Controlled occlusion: a true primitive (`T`), hidden by a mask (`T ∧ M`) that removes 0, 20, 40, 60 or 80% of its
  boundary in the observed zone; half-plane mask for circular primitives, disc for rectilinear ones.
  Label noise 0 or 8%. 300 observations. 1,920 scenes, paired by primitive. The TYPE is provided, not the
  parameters.
- Downstream chain: 128 never-used composition tasks, type of each leaf provided.
- Code: `core/deep_partial.py` (system, without truth), `core/deep_partial_exp.py` (evaluation),
  `core/deep_po_report.py`. Tests: 73 new (`make deep-po-test`); full suite 4,783 tests, 7 skipped; sealed set
  intact.
- Vocabulary note: in the WCE grammar a sector is an unbounded angle (apex, direction, half-aperture);
  it has no radius. Its analysed parameters are therefore the apex and the two angles.
- No threshold was tuned on the evaluation. A single seed per scene; no confirmatory test.

## 2. Recovery of the primitive, type known (1,920 scenes)

`top-1`: the first hypothesis has all its boundaries within the tolerance of the equivalence metric.

| Method | top-1 | top-3 | top-5 | Functional agreement top-1 | Seconds |
|---|---|---|---|---|---|
| `global_contrast` (global optimisation, reference) | 0.31 | 0.34 | 0.34 | 0.73 | 0.11 |
| `global_target` (direct gradient on the target) | 0.35 | 0.37 | 0.37 | 0.81 | 0.07 |
| `arc_normals` (intersection of the normals) | 0.39 | 0.46 | 0.48 | 0.79 | 0.12 |
| `arc_kasa` | 0.44 | 0.52 | 0.55 | 0.81 | 0.13 |
| `arc_taubin` | 0.43 | 0.51 | 0.54 | 0.80 | 0.13 |
| `arc_geometric` | 0.43 | 0.51 | 0.54 | 0.80 | 0.14 |
| `arc_joint` (+ shared constraints) | 0.51 | 0.61 | 0.65 | 0.84 | 0.22 |
| **`arc_visible`** (+ occlusion-tolerant score) | 0.49 | 0.61 | 0.64 | 0.84 | 0.21 |
| `ctx_joint` (structure known, mask inferred) | 0.56 | 0.63 | 0.65 | 0.87 | 0.36 |
| `ctx_oracle` (true mask provided) | 0.62 | 0.64 | 0.64 | 0.89 | 0.21 |

top-1 according to the share of boundary occluded:

| Method | 0% | 20% | 40% | 60% | 80% |
|---|---|---|---|---|---|
| `global_contrast` | 0.52 | 0.46 | 0.32 | 0.19 | 0.07 |
| `global_target` | **0.92** | 0.60 | 0.23 | 0.02 | 0.01 |
| `arc_visible` | 0.90 | 0.74 | 0.46 | 0.28 | 0.08 |
| `ctx_joint` | 0.96 | 0.80 | 0.60 | 0.35 | 0.11 |
| `ctx_oracle` | 0.96 | 0.88 | 0.67 | 0.43 | 0.14 |
| `arc_visible`, top-5 | 0.97 | 0.90 | 0.70 | 0.46 | 0.18 |

top-1 of `arc_visible` by type (in parentheses: `global_target`):

| Type | 0% | 20% | 40% | 60% | 80% |
|---|---|---|---|---|---|
| half-plane | 0.98 (1.00) | 0.86 (0.80) | 0.67 (0.00) | 0.39 (0.00) | 0.11 (0.00) |
| band | 0.97 (1.00) | 0.92 (0.92) | 0.86 (0.45) | 0.67 (0.06) | 0.17 (0.03) |
| disc | 0.92 (1.00) | 0.97 (1.00) | 0.41 (0.59) | 0.14 (0.00) | 0.02 (0.00) |
| anti-disc | 0.92 (0.97) | 0.59 (0.02) | 0.22 (0.00) | 0.11 (0.00) | 0.00 (0.00) |
| ring | 0.91 (0.92) | 0.78 (0.59) | 0.58 (0.28) | 0.39 (0.03) | 0.19 (0.02) |
| sector | 0.69 (0.64) | 0.31 (0.27) | 0.05 (0.03) | 0.00 (0.00) | 0.00 (0.00) |

Noise and occlusion separated (`arc_visible`, top-1):

| | mask 0% | 40% | 80% |
|---|---|---|---|
| no noise | 0.95 | 0.50 | 0.08 |
| noise 8% | 0.84 | 0.43 | 0.08 |

For `global_contrast`: 0.67 with neither noise nor mask, 0.36 with noise alone, 0.48 with 40% of mask alone. Noise is
the first problem of global optimisation; occlusion is the first problem of inference on fragments.

Paired differences (95% bootstrap CI, top-1):

| Comparison | All scenes | Mask ≥ 40% | Rich primitives |
|---|---|---|---|
| `arc_visible` − `global_contrast` | +0.18 [+0.16, +0.20] | +0.08 [+0.06, +0.11] | +0.25 [+0.22, +0.28] |
| `arc_visible` − `global_target` | +0.14 [+0.12, +0.16] | +0.19 [+0.17, +0.22] | +0.15 [+0.12, +0.18] |
| `arc_geometric` − `arc_kasa` | −0.01 [−0.02, −0.00] | −0.01 [−0.02, +0.00] | −0.01 [−0.01, +0.00] |
| `arc_joint` − `arc_geometric` | +0.07 [+0.06, +0.09] | +0.07 [+0.06, +0.09] | +0.15 [+0.12, +0.17] |
| `arc_visible` − `arc_joint` | −0.01 [−0.03, −0.00] | 0.00 [−0.02, +0.02] | −0.03 [−0.05, −0.01] |
| `ctx_joint` − `arc_visible` | +0.07 [+0.05, +0.09] | +0.08 [+0.05, +0.10] | +0.06 [+0.04, +0.08] |
| `ctx_oracle` − `arc_visible` | +0.12 [+0.11, +0.14] | +0.14 [+0.12, +0.16] | +0.09 [+0.07, +0.11] |

## 3. Parameters

Share of top-1 estimates out of tolerance (`arc_visible`, all occlusion levels):

| Type | Parameter | Out of tolerance |
|---|---|---|
| sector | apex | 70% |
| sector | direction | 62% |
| sector | half-aperture | 54% |
| anti-disc | centre | 62% |
| anti-disc | radius | 54% |
| disc | centre | 48% |
| disc | radius | 35% |
| ring | centre | 38% |
| ring | outer radius | 30% |
| ring | inner radius | 22% |
| half-plane | orientation | 34% |
| half-plane | offset | 32% |
| band | width / offset / orientation | 20% / 19% / 18% |

Median error of the disc: centre 0.08 and radius 0.03 without mask; 0.38 and 0.18 at 40%; 1.44 and 0.63 at 80%.
At 40% of mask, the ring keeps a centre at 0.18 and the band an orientation at 0.03 rad.

Partial oracles on the occlusion bench (top-1, paired difference with `arc_visible`):

| True parameter provided | All applicable scenes | Mask ≥ 40% |
|---|---|---|
| orientation (half-plane, band, sector) | +0.20 [+0.17, +0.22] | +0.26 [+0.22, +0.30] |
| centre (disc, anti-disc, ring, sector) | +0.21 [+0.18, +0.24] | +0.33 [+0.29, +0.38] |
| size (radii, width, half-aperture) | +0.11 [+0.09, +0.13] | +0.15 [+0.12, +0.18] |

For rings and sectors, providing the centre does not help (−0.05 [−0.09, +0.00]): the search for the radii or the angles
by histogram modes around the centre is less good than the fitting of the fragments.

### Parameter regions, identifiability, local sensitivity (simple circles and lines)

| Mask | Observed arc (rad) | Median conditioning | Radius–centre correlation | Posterior standard deviation of the radius | Mean width | Coverage of the truth |
|---|---|---|---|---|---|---|
| 0% | 4.71 | 3.6 | 0.40 | 0.02 | 0.27 | 0.87 |
| 20% | 3.99 | 5.3 | 0.56 | 0.04 | 0.37 | 0.65 |
| 40% | 3.11 | 15.9 | 0.75 | 0.09 | 0.55 | 0.42 |
| 60% | 2.54 | 54.2 | 0.83 | 0.17 | 0.84 | 0.30 |
| 80% | 2.00 | 297.9 | 0.89 | 0.26 | 1.10 | 0.22 |

The local diagnostic behaves as expected: when the arc shortens, one direction becomes flat and radius and
centre become almost interchangeable. But the region covers the truth in only 49% of cases on average: it
is built around the fragment attributed to the first hypothesis, and when this fragment mixes the boundary of the
primitive and that of the mask, the region is centred in the wrong place.

Identifiability labels: a parameter labelled `IDENTIFIABLE` is within tolerance in 72% of cases,
`PARTIALLY_IDENTIFIABLE` in 21%, `UNIDENTIFIABLE` in 18%. The order is right, the calibration is not.

Boundary completion: the share of completed boundary that falls on the true boundary goes from 0.92 (without mask)
to 0.57 (40%) then 0.30 (80%). The estimated visible fraction is correct up to 40% (0.60 for 0.60), then
overestimated (0.49 for 0.20 at 80%).

## 4. Shared constraints, minimal basis

Recall of ALL the boundary components among 5 hypotheses, rich primitive with shared constraints against
independent minimal components:

| Type | 0% | 20% | 40% | 60% | 80% |
|---|---|---|---|---|---|
| ring, rich | 0.95 | 0.92 | 0.78 | 0.55 | 0.27 |
| ring, minimal | 0.50 | 0.34 | 0.14 | 0.05 | 0.00 |
| band, rich | 1.00 | 0.98 | 0.95 | 0.88 | 0.33 |
| band, minimal | 0.95 | 0.89 | 0.67 | 0.33 | 0.08 |
| sector, rich | 0.84 | 0.61 | 0.20 | 0.05 | 0.02 |
| sector, minimal | 0.70 | 0.48 | 0.16 | 0.02 | 0.00 |

For disc, anti-disc and half-plane the two bases are identical by construction.

## 5. Autopsy of the sectors (`arc_visible`, 64 scenes per level)

| Mask | top-1 | top-5 | Rays found (out of 2) | Shortest visible ray | Failure: apex | Failure: angles | The true sector would pass the calibrated threshold |
|---|---|---|---|---|---|---|---|
| 0% | 0.69 | 0.84 | 1.81 | 4.50 | 17 out of 20 | 0 | 88% |
| 20% | 0.31 | 0.61 | 1.63 | 2.78 | 38 out of 44 | 1 | 61% |
| 40% | 0.05 | 0.20 | 1.11 | 1.16 | 51 out of 61 | 4 | 53% |
| 60% | 0.00 | 0.05 | 0.80 | 0.29 | 51 out of 64 | 12 | 36% |
| 80% | 0.00 | 0.02 | 0.47 | 0.13 | 61 out of 64 | 2 | 11% |

The dominant failure is the apex: it is the intersection of two lines, and a small angle error on a
short ray moves it a lot. From 40% of mask, one of the two rays has only about 1.2 units
visible left and is found only one time in two. The overly strict validation is a secondary factor: without
mask, 12% of the true sectors would already be rejected by the calibrated threshold of the previous sub-milestone; at 60%, 64%.

## 6. Active measurements directed at the parameters (192 scenes, mask 40 or 60%, noise 5%, 16 measurements)

| Policy | top-1 | top-5 | Functional agreement |
|---|---|---|---|
| no measurement | 0.43 | 0.61 | 0.84 |
| random | 0.43 | 0.59 | 0.82 |
| disagreement between hypotheses | 0.49 | 0.66 | 0.85 |
| disagreement, occluded zones excluded | 0.47 | 0.65 | 0.85 |

Disagreement − random: top-1 +0.06 [−0.01, +0.13]; functional agreement +0.03 [+0.01, +0.05].

## 7. Downstream chain (128 composition tasks, type of each leaf provided unless stated otherwise)

| Literals | Leaves recovered (top-1 / top-k) | Equivalence | Literals | Candidate structure ≥ 0.97 | Final ≥ 0.95 | Final ≥ 0.99 | Final agreement | After exchange |
|---|---|---|---|---|---|---|---|---|
| true atoms | — | 1.00 | 3.5 | 0.67 | 0.77 | 0.11 | 0.963 | 0.970 |
| true atoms + guardrail | — | 1.00 | 3.5 | 0.94 | 0.88 | 0.65 | 0.980 | 0.980 |
| old estimation, true centres and orientations | — | 0.41 | 3.5 | 0.44 | 0.53 | 0.11 | 0.921 | 0.921 |
| old estimation, type only (`types_old`) | — | 0.20 | 3.5 | 0.21 | 0.28 | 0.06 | 0.877 | 0.877 |
| **new inference, type only, k = 1** | 0.26 / 0.26 | 0.31 | 3.3 | 0.33 | 0.39 | 0.10 | 0.899 | 0.899 |
| **new inference, type only, k = 3** | 0.26 / 0.42 | 0.49 | 9.8 | 0.41 | 0.52 | 0.10 | 0.914 | 0.914 |
| new inference, type only, k = 5 | 0.26 / 0.49 | 0.55 | 16.2 | 0.37 | 0.51 | 0.09 | 0.915 | 0.915 |
| k = 3, global score instead of visible | 0.30 / 0.44 | 0.51 | 9.8 | 0.39 | 0.49 | 0.10 | 0.915 | 0.915 |
| k = 3 + guardrail | 0.26 / 0.42 | 0.49 | 9.8 | 0.42 | 0.49 | 0.10 | 0.912 | 0.912 |
| new inference + true centre, k = 1 | 0.34 | 0.40 | 3.5 | 0.38 | 0.44 | 0.10 | 0.896 | 0.896 |
| new inference + true orientation, k = 1 | 0.39 | 0.43 | 3.9 | 0.39 | 0.48 | 0.10 | 0.913 | 0.913 |
| new inference + true size, k = 1 | 0.38 | 0.44 | 3.8 | 0.41 | 0.50 | 0.09 | 0.910 | 0.910 |
| frozen pipeline, no oracle (reference) | — | 0.46 | 77.1 | 0.21 | 0.26 | 0.03 | 0.876 | 0.876 |
| robust discovery, no oracle | — | 0.23 | 3.2 | 0.24 | 0.32 | 0.07 | 0.654 | 0.653 |

Paired differences per task, 95% CI:

| Comparison | Equivalence of the primitives | Candidate structure | Final agreement |
|---|---|---|---|
| k = 3 − `types_old` | +0.29 [+0.23, +0.34] | +0.20 [+0.13, +0.27] | +0.037 [+0.025, +0.050] |
| k = 1 − `types_old` | +0.11 [+0.05, +0.16] | +0.12 [+0.04, +0.20] | +0.022 [+0.009, +0.035] |
| k = 3 − k = 1 | +0.18 [+0.15, +0.22] | +0.08 [+0.02, +0.14] | +0.015 [+0.004, +0.026] |
| k = 3 − frozen pipeline | +0.03 [−0.03, +0.08] | +0.20 [+0.13, +0.27] | +0.038 [+0.025, +0.051] |
| true atoms − k = 1 | +0.69 [+0.65, +0.73] | +0.34 [+0.27, +0.43] | +0.063 [+0.049, +0.078] |
| exchange − without exchange, k = 3 | — | — | 0.000 [−0.000, +0.000] |
| exchange − without exchange, true atoms | — | — | +0.008 [+0.006, +0.010] |

Leaves recovered in composition (k = 3, top-1 / top-3): half-plane 0.31 / 0.56; band 0.30 / 0.50;
disc 0.23 / 0.39; anti-disc 0.27 / 0.50; ring 0.31 / 0.38; sector 0.06 / 0.07.

By depth, equivalence of the primitives (k = 3): 0.64 (depth 2), 0.56 (3), 0.45 (4), 0.31 (5).

Cost: the new inference does no gradient fitting (0.21 s per primitive, 4 hypotheses on average).
In the chain, k = 3 costs 12.8 s per task against 19.0 s for the frozen pipeline and 11.8 s for the old estimation.
AWS: 2 × c6i.xlarge for one hour for the whole campaign.

## 8. Preregistered hypotheses

| # | Result |
|---|---|
| PO1 inference on fragments > global optimisation | **supported**: +0.18 [+0.16, +0.20] |
| PO2 gain under mask ≥ 40% | **supported**: +0.08 [+0.06, +0.11] |
| PO3 geometric estimator > Kåsa | **not supported**: −0.01 [−0.02, −0.00] |
| PO4 shared constraints, rich primitives | **supported**: +0.15 [+0.12, +0.17] |
| PO5 occlusion-tolerant score | **not supported**: −0.01 [−0.03, −0.00] |
| PO6 composition context | **supported**: +0.07 [+0.05, +0.09] |
| PO7 top-5 − top-1 ≥ 0.10 | **supported**: 0.64 − 0.49 = 0.15 |
| PO8 coverage of the regions ≥ 80% | **not supported**: 0.87 without mask, 0.49 on average |
| PO9 active measurements > random | **not supported** on the criterion (top-1 +0.06 [−0.01, +0.13]); functional agreement +0.03 [+0.01, +0.05] |
| PO10 downstream: candidate structure > old estimation | **supported**: +0.20 [+0.13, +0.27] |
| PO11 downstream: final agreement > frozen pipeline | **supported**: +0.038 [+0.025, +0.051] — but the arm receives the types, the frozen pipeline does not |
| PO12 primitive ↔ composition exchange | **not supported**: 0.000; +0.008 only with the true atoms |

## 9. Answers

**1. What fraction of boundary is needed to estimate each primitive correctly?**
Without noise, for the first hypothesis to be correct at least one time in two: band and ring 40% of
visible boundary; half-plane 60%; disc and anti-disc 80%; sector 100% (and even then: 0.69). With five
hypotheses: band, ring and half-plane 40%; disc and anti-disc 60%; sector 80%. The primitives with two
linked boundaries hold up better than the primitives with a single one: the second boundary fixes the centre or the direction.

**2. Which parameters are the most fragile?**
The apex of the sector (out of tolerance in 70% of cases), then its angles; the centre of the circles (48 to 62%), always
more fragile than the radius; the orientation and the offset of an isolated half-plane (one third). The most solid are those
of the band (less than 20%). Providing the true centre or orientation is worth +0.20 of recovery, and +0.26 to +0.33
under strong mask; providing the size is worth half as much.

**3. Does explicit occlusion improve the fit?**
Yes when it goes through the context, no when it goes through the score alone. The "occlusion-tolerant" score, which
ignores the silent segments, does no better than the global contrast (−0.01). Rendering the whole composite and
comparing only what must be visible brings +0.07 with an inferred mask and +0.12 with the true mask. The gap between
the two (0.05) is what the error on the mask itself costs.

**4. Do shared constraints help?**
Yes, it is the clearest gain of the sub-milestone: +0.15 [+0.12, +0.17] on the rich primitives. At 60% of mask,
the ring with shared centre is recovered in 55% of cases against 5% for two independent circles, the band in
88% against 33%. For the sector, the gain exists but remains small (5% against 2%).

**5. Does parametric top-k improve the composition?**
Yes up to k = 3: candidate structure 0.33 → 0.41 (+0.08 [+0.02, +0.14]), final agreement +0.015. Not beyond:
k = 5 recovers more leaves (0.49 against 0.42) but falls back to 0.37 of candidate structure, with 16 literals.

**6. Can compositional reasoning refine the primitives?**
On the occlusion bench, yes: knowing the structure and jointly choosing primitive and mask gives +0.07. In
the full chain, no: the bounded exchange after composition changes nothing (0.000), except with the true atoms
(+0.008 of agreement, and the recovery at 0.99 goes from 11% to 34%). The exchange refines parameters that are already close; it
does not repair a structure built on bad parameters.

**7. Do sectors remain the main failure?**
Yes. 0.69 without mask, 0.05 at 40%, 0.00 beyond; 6% of the sectors recovered in composition. The failure is the apex
in 80 to 95% of cases, because one of the two rays becomes too short to be found.

**8. Do active measurements become useful for the parameters?**
Not in an established way. Measuring where the hypotheses diverge gives top-1 0.49 against 0.43 for random, but
the interval includes zero; only the functional agreement progresses clearly (+0.03). Excluding the occluded zones adds
nothing. They are not integrated.

**9. Does the parametric gain really improve the downstream?**
Yes, with type known, and over the whole chain: leaves recovered, equivalence of the primitives 0.20 → 0.49, candidate
structure 0.21 → 0.41, final agreement 0.877 → 0.914, all with an interval above zero. Three reservations: the
type of each leaf is provided by oracle; the recovery at 0.99 remains at 10%; the gap to the true atoms remains
large (0.41 against 0.67 of candidate structure).

**10. What is the next bottleneck?**
Two, in order. First the attribution of the fragments: under composition, a fragment often mixes the boundary of
the primitive and that of what occludes it, and all the rest (region, identifiability, completion) is then centred in the
wrong place; this is what the coverage of the regions (49%) and the gain of the true mask (+0.12) show. Then
the absence of a chain without type oracle: the robust discovery without oracle remains at 0.65 of final agreement, and nothing here
says which type to propose for which fragment.

## 10. Limits

- Dev only, one seed per scene, no confirmation.
- The occlusion bench has only one form of composition (`T ∧ M`) and a mask of a geometric type other than the target's.
- All the downstream gains assume the type of each leaf known. The comparison with the frozen pipeline (PO11) is therefore
  not on equal terms.
- The top-1 metric has fixed tolerances; the functional agreement moves less than the recovery.
- The geometric estimator, the occlusion-tolerant score and the exchange in the chain bring nothing measurable;
  they remain in the code as comparison arms.
- Parameter regions and identifiability labels are computed only for simple circles and lines.
- Determinism: at fixed seed the hypotheses are identical up to machine precision (gaps of 2e-16 from one
  call to the next, due to multi-threaded linear algebra); the determinism test therefore compares at 1e-9.
- Formalisation `PrimitiveNode` / `CompositeNode`: only visibility flows from the composite to the primitive;
  no geometric constraint is deduced from the structure.

## 11. Verdict

Preregistered rule: PO1, PO10 and PO11 are supported, which literally gives
`PARAMETER INFERENCE READY FOR END-TO-END VALIDATION`.

I do not retain this verdict, and I flag it as a **declared deviation from the rule, in the conservative direction**. The
rule that I wrote did not foresee that PO11 would compare an arm receiving the types by oracle with a pipeline that
receives none. An end-to-end validation has no type oracle, and the only chain without oracle measured here is
worse than the frozen pipeline.

`PARAMETER INFERENCE IMPROVED BUT INSUFFICIENT`

The inference of the parameters with type known is clearly better and the gain goes through the whole chain. It remains
insufficient beyond 40% of mask, on the sectors, and as long as no chain without oracle benefits from it. The
frozen pipeline remains the reference; nothing from this sub-milestone is plugged into it.

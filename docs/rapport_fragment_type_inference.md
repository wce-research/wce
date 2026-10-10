# WCE Fragment Attribution & Primitive Type Inference — report (dev, exploratory)

Status: exploratory sub-milestone, **dev** partition only. No confirmatory test, no 16-seed campaign,
sealed set never touched. The frozen pipeline remains the reference; nothing from this sub-milestone is plugged into it by default.

## 0. Preregistration (written before launching the AWS runs)

Tuning done only on the scenes of the dev sub-range + 350,000 (at most 18 scenes per condition) and on
composition task 699. Evaluation on 1,920 never-seen occlusion scenes (indices 100–131 of the dev sub-range
+ 300,000) and on 128 never-used composition tasks (keys 600–615).

Main method: `hier` (consensus fragments that may overlap, relation graph, families then types,
description-length type score with negative evidence limited to observable zones). Downstream: `hier_k3`, without
any oracle. Reference: frozen pipeline.

| # | Hypothesis | Criterion |
|---|---|---|
| FT1 | attribution after type inference beats grouping by proximity | pairwise F1, paired difference, 95% CI > 0 |
| FT2 | the geometry → family → type hierarchy beats flat classification on fragments | `hier` − `flat_fragments`, correct type, CI > 0 |
| FT3 | inference on fragments beats trying all types on the whole scene | `hier` − `flat_scene`, correct type, CI > 0 |
| FT4 | keeping three types per group finds the right type markedly more often | `type_3` − `type_1` ≥ 0.10 |
| FT5 | negative evidence limited to observable zones helps | `hier` − `hier_no_negative` and `hier` − `hier_naive_negative`, CI > 0 |
| FT6 | the attribution ↔ type loop helps | `hier_loop` − `hier`, CI > 0 |
| FT7 | occlusion ownership recognises a hidden boundary | hidden part labelled "occluded" ≥ 0.5 at 40% mask |
| FT8 | sectors become recoverable | exact recovery ≥ 0.5 without mask |
| FT9 | directed active measurement resolves type ambiguity better than random | paired difference, CI > 0 |
| FT10 | downstream without oracle: candidate structure above the frozen pipeline | `hier_k3` − `reference`, CI > 0 |
| FT11 | downstream without oracle: final agreement above the frozen pipeline | `hier_k3` − `reference` on `final_full`, CI > 0 |
| FT12 | top-3 of types helps downstream | `hier_k3` − `hier_k1` on `final_full`, CI > 0 |

Expectation declared from the tuning: FT1, FT3, FT4 likely; FT2, FT5, FT6 probably null; FT7 and FT8
probably not reached; FT10–FT12 unknown (one tuning task, inconclusive).

The oracle matrix (true or estimated fragments × true or inferred type) serves to apportion the loss; it is not a
hypothesis.

Verdict rule fixed in advance. This time, the two arms compared have **no oracle**:

- `FRAGMENT/TYPE INFERENCE READY FOR END-TO-END VALIDATION`: FT10 and FT11 supported, with an arm without oracle.
- `FRAGMENT/TYPE INFERENCE IMPROVED BUT INSUFFICIENT`: FT1 or FT3 supported, but FT10 or FT11 not supported.
- `FRAGMENT/TYPE INFERENCE STILL BLOCKING`: neither FT1 nor FT3.

If an oracle-free arm other than `hier_k3` (for example `flat_scene_k3` or a variant with residual rounds) beats the
frozen pipeline while `hier_k3` does not beat it, this will be reported as an exploratory result, without changing the verdict.

## 1. Protocol executed

- AWS run `wce-deepft-20261005-110621` (EC2 + SSM, 2 × c6i.xlarge, 81 and 83 min, 2,976 sub-units, 0 errors, 33,920 lines).
  Commands: `make deep-run-ft`, `make deep-fetch`, `make deep-ft-aggregate`, `make deep-ft-report`.
  Summary: `results/deep_fragment_type_summary.json`.
- Occlusion bench: 1,920 never-seen scenes (6 types × 5 mask levels × 2 noise levels × 32 primitives),
  one target primitive and, except at 0%, one mask. Downstream chain: 128 never-used composition tasks.
- Code: `core/deep_fragments.py` (system, without truth, without learned model), `core/deep_fragments_exp.py`
  (evaluation), `core/deep_ft_report.py`. Tests: 57 new (`make deep-ft-test`); full suite 4,840 tests OK,
  7 skipped; sealed set intact.
- No threshold tuned on the evaluation. A single seed per scene; no confirmatory test.
- Vocabulary reminder: in the WCE grammar a sector is an unbounded angle (vertex and two rays); it
  has no arc. The sector branch therefore bears on the rays and the vertex, and on the confusion with a disc.

## 2. Fragment attribution (1,920 scenes)

Pairs of observed midpoints: "same group" versus "same true boundary".

| Grouping | Precision | Recall | F1 | Groups | Groups mixing target and mask |
|---|---|---|---|---|---|
| proximity | 0.83 | 0.51 | 0.59 | 5.1 | 31% |
| + tangential continuity | 0.87 | 0.46 | 0.57 | 6.9 | 27% |
| + compatible centre | 0.87 | 0.46 | 0.57 | 6.9 | 27% |
| + compatible orientation | 0.87 | 0.48 | 0.59 | 6.8 | 27% |
| + compatible curvature | 0.78 | 0.60 | 0.64 | 5.6 | 41% |
| shared constraints (all three) | 0.78 | 0.60 | 0.64 | 5.6 | 41% |
| consensus (circles and lines) | 0.65 | 0.82 | **0.71** | 2.9 | 60% |
| after type inference | 0.62 | 0.85 | 0.70 | 1.3 | 65% |

Paired differences in F1 against proximity: consensus +0.11 [+0.10, +0.12]; after type inference +0.11
[+0.10, +0.12]; curvature +0.04 [+0.04, +0.05]; tangential continuity −0.02 [−0.03, −0.02].

Two regimes are opposed. The local groupings are pure but break each boundary into six or seven
pieces. The model-based groupings gather each boundary but mix target and mask in 60% of the groups.
No mechanism obtains both purity and recall.

## 3. Type inference (1,920 scenes)

`correct type`: the type predicted for the group of midpoints attributed to the target is the right one. The top-1 and top-3 columns
count among the first n or 3n hypotheses, n being the number of true primitives in the scene.

| Variant | Correct type | Type top-1 | Type top-3 | Exact top-1 | Equivalent top-3 | Functional top-1 | Types tested | Seconds |
|---|---|---|---|---|---|---|---|---|
| `hier` (fragments, families, types) | 0.41 | 0.44 | 0.52 | 0.37 | 0.46 | 0.34 | 20.1 | 0.38 |
| `flat_fragments` (same fragments, without hierarchy) | 0.41 | 0.43 | 0.51 | 0.37 | 0.46 | 0.34 | 31.4 | 0.62 |
| `flat_scene` (all types on the whole scene) | 0.40 | 0.42 | **0.63** | 0.33 | **0.56** | 0.30 | 16.6 | 0.51 |
| `hier` without negative evidence | 0.41 | 0.44 | 0.51 | 0.37 | 0.46 | 0.34 | 20.1 | 0.38 |
| `hier`, naive negative evidence | 0.38 | 0.42 | 0.49 | 0.36 | 0.44 | 0.33 | 20.1 | 0.38 |
| `hier` + attribution ↔ type loop | 0.41 | 0.44 | 0.51 | 0.37 | 0.45 | 0.34 | 19.0 | 0.41 |

Correct type by occluded share (`hier`): 0.82 at 0%; 0.59 at 20%; 0.41 at 40%; 0.21 at 60%; 0.04 at 80%.
For `flat_scene`: 0.69; 0.50; 0.41; 0.29; 0.13. Inference on fragments is better when the boundary is
visible, worse when it is heavily occluded.

Correct type by true type (`hier`): band 0.65; disc 0.55; ring 0.40; half-plane 0.36; anti-disc 0.34;
sector 0.16.

Main confusions (true type → predicted type, over 320 scenes per type):

| True type | Correct | Most frequent errors |
|---|---|---|
| band | 207 | no hypothesis 43, disc 26, sector 23 |
| disc | 176 | no hypothesis 72, ring 32 |
| ring | 129 | band 67, disc 66, sector 33 |
| half-plane | 116 | no hypothesis 106, sector 53 |
| anti-disc | 110 | no hypothesis 124, half-plane 29 |
| sector | 52 | no hypothesis 99, disc 74, half-plane 66 |

Calibration: the confidence exceeds 0.8 in 1,355 cases, for an accuracy of 55%; calibration error 0.34. The
score is not calibrated.

Signals, correct versus wrong hypotheses: midpoints closely explained 37.6 versus 26.6; share of boundary seen
0.53 versus 0.39; residual of the shared constraint 0.06 versus 0.14. The geometric gap (0.09 versus 0.10), the absent
share (0.04 versus 0.05) and the stability (0.02 versus 0.02) hardly separate them.

## 4. Oracle matrix: where the loss comes from

Exact recovery of the target as first hypothesis, occlusion bench:

| | True type | Inferred type |
|---|---|---|
| **True fragments** | 0.66 | 0.50 |
| **Estimated fragments** | 0.51 | 0.34 |

Paired differences, 95% CI: attribution, at true type, −0.15 [−0.17, −0.13]; type, at true fragments, −0.16
[−0.18, −0.15]; both together −0.32 [−0.34, −0.30]. The two losses are of the same size and add up.

By occluded share:

| Mask | True fragments, true type | Estimated fragments, true type | True fragments, inferred type | Estimated fragments, inferred type |
|---|---|---|---|---|
| 0% | 0.95 | 0.92 | 0.86 | 0.80 |
| 20% | 0.88 | 0.73 | 0.77 | 0.51 |
| 40% | 0.78 | 0.51 | 0.57 | 0.32 |
| 60% | 0.53 | 0.32 | 0.25 | 0.08 |
| 80% | 0.17 | 0.09 | 0.04 | 0.00 |

Even with true fragments and true type, 34% of failures remain: this is the share due to the parameters, zero without mask and
dominant at 80%.

Same matrix in the downstream chain (candidate structure reached, 128 tasks):

| Condition | Candidate structure | Final agreement |
|---|---|---|
| true atoms | 0.59 | 0.956 |
| true fragments, true type, inferred parameters | 0.51 | 0.929 |
| true fragments, inferred type | 0.44 | 0.911 |
| estimated fragments, true type (`po_type_k3`) | 0.35 | 0.901 |
| estimated fragments, inferred type (`hier_k3`) | 0.27 | 0.874 |
| frozen pipeline | 0.20 | 0.864 |

## 5. Occlusion ownership

The true primitive is given; the system must say why its missing portions are missing, with its own
hypotheses as candidate masks.

| Mask | Hidden part labelled "hidden" | Hidden part labelled "absent" | Visible part labelled "observed" | Right owner found |
|---|---|---|---|---|
| 20% | 0.02 | 0.39 | 0.96 | 1% |
| 40% | 0.11 | 0.52 | 0.96 | 8% |
| 60% | 0.20 | 0.54 | 0.95 | 15% |
| 80% | 0.19 | 0.59 | 0.91 | 12% |

The rest of the hidden part is classified as out of domain or short gap. The visible part is almost never taken
for hidden (less than 1%). But the mask is recognised as owner in only 8 to 15% of cases: the hidden
boundary remains mostly counted as absent.

## 6. Sector branch (64 scenes per level)

| Mask | Exact | Rays found (out of 2) | Taken for a disc | Most frequent failing step |
|---|---|---|---|---|
| 0% | 0.23 | 1.61 | 30% | rays 20, vertex relation 14, vertex position 13 |
| 20% | 0.05 | 1.31 | 33% | rays 38 |
| 40% | 0.05 | 1.14 | 47% | rays 44 |
| 60% | 0.00 | 0.86 | 48% | rays 62 |
| 80% | 0.00 | 0.39 | 80% | rays 63 |

Without mask, the failure is spread over three steps. From 20% mask onwards, it is almost always upstream: the two
rays are not found as straight fragments, and the midpoints of the sector are read as a circular arc.

## 7. Active measurement to resolve a type ambiguity (192 scenes, 16 measurements)

| Policy | Correct type | Exact top-1 |
|---|---|---|
| no measurement | 0.45 | 0.48 |
| random | 0.49 | 0.45 |
| where the competing types diverge | 0.44 | 0.47 |

Directed − random: correct type −0.06 [−0.13, +0.01].

## 8. Downstream chain without type oracle (128 composition tasks)

`attribution → type → parameters → candidate structure → function`

| Arm | Oracle | Attribution F1 | Type recall | Parameters recovered | Literals | Candidate structure | Final ≥ 0.95 | Final agreement | Seconds |
|---|---|---|---|---|---|---|---|---|---|
| frozen pipeline | none | — | 0.33 | 0.21 | 77.9 | 0.20 | 0.24 | 0.864 | 18.2 |
| robust discovery | none | — | 0.30 | 0.26 | 3.6 | 0.22 | 0.30 | 0.693 | 5.6 |
| `hier_k1` | none | 0.51 | 0.23 | 0.17 | 2.7 | 0.22 | 0.23 | 0.865 | 11.5 |
| **`hier_k3`** (main arm) | none | 0.51 | 0.36 | 0.26 | 7.1 | 0.27 | 0.30 | 0.874 | 12.5 |
| `hier_k3` + residual rounds | none | 0.51 | 0.36 | 0.26 | 7.1 | 0.27 | 0.34 | 0.879 | 16.2 |
| `flat_scene_k1` | none | — | 0.33 | 0.26 | 7.0 | 0.27 | 0.32 | 0.882 | 11.4 |
| `flat_scene_k3` | none | — | 0.55 | 0.40 | 21.0 | **0.33** | **0.37** | **0.895** | 13.8 |
| `flat_scene_k3` + residual rounds | none | — | 0.55 | 0.40 | 21.0 | 0.32 | 0.37 | 0.889 | 17.7 |
| `po_type_k3` | types | — | 0.55 | 0.40 | 9.3 | 0.35 | 0.44 | 0.901 | 12.0 |
| true atoms | all | — | 1.00 | 1.00 | 3.5 | 0.59 | 0.75 | 0.956 | 8.0 |

Paired differences against the frozen pipeline, 95% CI:

| Arm | Candidate structure | Final ≥ 0.95 | Final agreement |
|---|---|---|---|
| `hier_k3` | +0.06 [−0.01, +0.13] | +0.06 [−0.01, +0.13] | +0.010 [−0.004, +0.025] |
| `hier_k3` + residual rounds | +0.06 [0.00, +0.13] | +0.10 [+0.03, +0.17] | +0.016 [+0.003, +0.029] |
| `flat_scene_k1` | +0.07 [+0.02, +0.13] | +0.08 [+0.01, +0.15] | +0.019 [+0.006, +0.032] |
| `flat_scene_k3` | +0.13 [+0.07, +0.19] | +0.13 [+0.06, +0.20] | +0.032 [+0.018, +0.046] |
| `po_type_k3` (type oracle) | +0.15 [+0.08, +0.22] | +0.20 [+0.13, +0.27] | +0.037 [+0.024, +0.051] |

Other comparisons: `flat_scene_k3` − `po_type_k3`, final agreement −0.006 [−0.016, +0.005]; `flat_scene_k3` −
`hier_k3`, final agreement +0.021 [+0.010, +0.034]; `hier_k3` − `hier_k1`, final agreement +0.009 [−0.002, +0.021] and
final ≥ 0.95 +0.08 [+0.02, +0.13]. The guardrail changes nothing on these arms.

By depth, candidate structure (`flat_scene_k3` + residual / frozen pipeline): 0.88 / 0.63 at depth 2;
0.25 / 0.13 at 3; 0.13 / 0.03 at 4; 0.03 / 0.03 at 5.

Leaves whose type is recovered (`flat_scene_k3` / `hier_k3` / frozen pipeline): disc 0.61 / 0.51 / 0.36;
anti-disc 0.63 / 0.41 / 0.14; band 0.55 / 0.45 / 0.32; half-plane 0.48 / 0.20 / 0.55; ring 0.39 / 0.16 / 0.19;
sector 0.34 / 0.07 / 0.26. Sectors whose parameters are recovered: 0.04 / 0.00 / 0.08.

## 9. Preregistered hypotheses

| # | Result |
|---|---|
| FT1 attribution after type > proximity | **supported**: F1 +0.11 [+0.10, +0.12] |
| FT2 hierarchy > flat classification on fragments | **not supported**: +0.003 [−0.003, +0.008] |
| FT3 fragments > all types on the scene | **not supported**: +0.010 [−0.015, +0.035] |
| FT4 top-3 − top-1 ≥ 0.10 | **not supported** for `hier` (0.52 − 0.44 = 0.08); 0.21 for `flat_scene` |
| FT5 negative evidence limited to observable zones | **not supported**: equal to the absence of negative evidence (−0.002); better than the naive version (+0.027 [+0.015, +0.039]) |
| FT6 attribution ↔ type loop | **not supported**: 0.000 [−0.004, +0.004] |
| FT7 hidden boundary recognised (≥ 0.5 at 40%) | **not supported**: 0.11 |
| FT8 sectors recoverable (≥ 0.5 without mask) | **not supported**: 0.23 |
| FT9 active measurement > random | **not supported**: −0.06 [−0.13, +0.01] |
| FT10 downstream without oracle: candidate structure > frozen pipeline | **not supported** for `hier_k3`: +0.06 [−0.01, +0.13] |
| FT11 downstream without oracle: final agreement > frozen pipeline | **not supported** for `hier_k3`: +0.010 [−0.004, +0.025] |
| FT12 top-3 of types helps downstream | **not supported** on the criterion (+0.009 [−0.002, +0.021]); final ≥ 0.95 +0.08 [+0.02, +0.13] |

Exploratory result provided for by the preregistration clause: another oracle-free arm, `flat_scene_k3`, beats the
frozen pipeline on the candidate structure and on the final agreement, with intervals above zero. It was
not the main arm; it does not change the verdict.

## 10. Answers

**1. What share of the loss comes from fragment attribution?**
About half of what is lost before the parameters: −0.15 [−0.17, −0.13] of exact recovery on the occlusion
bench, and −0.16 of candidate structure in the chain (0.51 with true fragments versus 0.35 with estimated
fragments, true type in both cases).

**2. What share comes from the type?**
The same on the occlusion bench: −0.16 [−0.18, −0.15] at true fragments. In the chain, it is smaller:
−0.07 of candidate structure at true fragments. And it disappears if one gives up choosing: trying the six types
on the scene and keeping three hypotheses per type gives exactly the type recall of the oracle (0.55) and an equivalent
final agreement (−0.006 [−0.016, +0.005]). In the chain, the type is therefore not a bottleneck in its own right; attribution
is one.

**3. Does the top-k of types help?**
Yes, but less per group of fragments than per type on the scene. For `hier`, going from 1 to 3 types per group
gains 0.08 of type recall and +0.08 of tasks at 0.95, with no clear effect on the mean agreement. For `flat_scene`,
going from 1 to 3 hypotheses per type takes the type recall from 0.33 to 0.55 and the candidate structure from 0.27
to 0.33.

**4. Does the type hierarchy help?**
Not in accuracy: +0.003. It only reduces the computation, from 31 to 20 types tested per scene and from 0.62 to 0.38 s.
The small grammar of five families suffices to express the six types, but does not distinguish them better than a
flat ranking by score.

**5. Do sectors become recoverable?**
No. 0.23 without mask, 0.05 at 20%, 0 beyond; this is worse than at known type (0.69 without mask in the previous
sub-milestone). The rays are not isolated as straight fragments and the sector is read as a disc in 30
to 80% of cases. In composition, the sector type is proposed for 34% of the leaves by `flat_scene_k3`, but its
parameters are recovered for only 4%.

**6. Does occlusion ownership help?**
No. The mechanism is cautious (it almost never takes a visible boundary for hidden) but it identifies the
mask in only 8 to 15% of cases, and its effect on the type is zero relative to the absence of negative evidence. The only
clear result is negative: counting every missing boundary as absent degrades the type (−0.027).

**7. Does the gain carry through to the final structure?**
For the main arm, no: `hier_k3` improves attribution but remains within uncertainty on the candidate
structure (+0.06 [−0.01, +0.13]) and the final agreement (+0.010 [−0.004, +0.025]). For `flat_scene_k3`, yes, over the whole
chain: type recall +0.22, parameters +0.19, candidate structure +0.13, final agreement +0.032, all with an
interval above zero.

**8. Does the cost remain compact?**
Yes. No gradient fitting in the inference: 0.4 to 0.7 s per scene, 17 to 33 types tested, 11 fragments. In
the chain, 12.5 to 13.8 s per task versus 18.2 s for the frozen pipeline, with 7 to 21 literals instead of 78. The whole
campaign fits on two c6i.xlarge for about one hour twenty.

**9. What is the new dominant error?**
Attribution under composition. In the chain, the attribution F1 is only 0.51; with the true fragments, the
candidate structure goes from 0.35 to 0.51 at equal type. On the bench, 60% of the model-based groups mix target and
mask. Behind it: depth (candidate structure 0.13 at depth 4 for `flat_scene_k3` with residual
rounds) and
recovery at 0.99, which remains below 5% for all the oracle-free arms.

**10. Can one at last build a chain without type oracle that is superior to the frozen pipeline?**
Yes, on these 128 dev tasks, but not with the arm I had designated. `flat_scene_k3` uses no oracle and
beats the frozen pipeline: candidate structure 0.33 versus 0.20 (+0.13 [+0.07, +0.19]), final agreement 0.895 versus 0.864
(+0.032 [+0.018, +0.046]), with almost four times fewer literals and in less time. This is an exploratory result, on a comparison
arm, to be confirmed before any conclusion.

## 11. Limits

- Dev only, one seed per scene, no confirmation.
- The arm that beats the frozen pipeline was not the preregistered main arm. A dozen oracle-free arms were
  compared with the reference; the best one is exposed to a selection bias.
- `flat_scene` uses neither explicit fragments nor attribution: its success does not validate the fragment-based approach
  of this sub-milestone, it shows that the parameter inference of the previous sub-milestone does without a type oracle.
- The occlusion bench has only one form of composition (`target ∧ mask`).
- Type score not calibrated; fragment roles (`CompatibleFragmentAssignments`) built and tested but not
  evaluated quantitatively.
- The polarity of a fragment is read locally; under composition it can be wrong (anti-disc: 0.34).
- No learned model was tried: the explicit approaches plateau, but the ceiling comes from attribution,
  which a type classifier would not correct.

## 12. Verdict

Preregistered rule: FT1 supported; FT10 and FT11 not supported for the main arm.

`FRAGMENT/TYPE INFERENCE IMPROVED BUT INSUFFICIENT`

Attribution and type are now separated and measured: they weigh as much as each other on the occlusion
bench, and attribution dominates in composition. Explicit inference by fragments brings no established downstream
gain. On the other hand, trying all types without attribution gives an oracle-free chain above the frozen pipeline;
this result is exploratory. The frozen pipeline remains the reference; nothing from this sub-milestone is plugged into it.

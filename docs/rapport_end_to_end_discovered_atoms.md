# WCE End-to-End Composition with Discovered Atoms — report (dev, exploratory)

Status: exploratory sub-milestone, **dev** partition only. No 16-seed campaign, sealed set never touched.
The frozen pipeline remains the official reference; `flat_scene_k3` remains a candidate baseline; composition over
classes is not wired in.

## 0. Preregistration (written before the AWS runs were launched)

Tuning done only on composition keys 990–999 (tuning of chain C on 60 tasks, keys
990–994; trials on key 999). Evaluation on **384 never-used tasks**: keys 900–931, depths 1 to 6,
standard and rich regimes. None of these tasks was used to select `flat_scene_k3`.

### The three chains, and they alone, are primary

| Chain | Primitives | Composition | Downstream |
|---|---|---|---|
| **A** | frozen pipeline (gradient) | frozen pipeline | frozen pipeline |
| **B** | `flat_scene_k3` | current pipeline | current pipeline (non-validated fit, current score) |
| **C** | `flat_scene_k3` | composition over equivalence classes | validated finalisation |

None of the three receives a true type, a true parameter, a true atom or true membership. The maximum
depth is set to 6 for all: the size of the truth is given to no one.

Configuration of C, frozen before launch (`results/deep_configs/e2e.json`): primitives grouped into hypotheses
(a primitive and its equivalents or complements = one hypothesis), reduction to at most six atoms by forward
selection on the floor error (minimum gain 1), exact composition over cells with the complements as units,
selection by likelihood minus ½·ln N per leaf, **validated** fit (initial state kept, commit only
if the observable criterion improves).

Any other variant (`C_nofit`, `C_forcedfit`, `C_errors`, `C_posterior`, `C_k1`, `C_k5`, other sources, other
reductions, wide engine, hierarchical composition) is secondary and cannot become primary after observation.

### Hypotheses

| # | Hypothesis | Criterion |
|---|---|---|
| EE1 | C beats the frozen pipeline | C − A, paired difference per task, 95% CI > 0 on the candidate structure (≥ 0.97) AND on the final agreement |
| EE2 | C beats `flat_scene_k3` + current downstream | C − B, same two criteria |
| EE3 | the gain holds at depth | C − A and C − B on the final agreement, depths 3 to 6: CI > 0 |
| EE4 | without a compute explosion | time per task of C lower than that of A and of B |
| EE5 | the validated fit does not harm | C − `C_nofit` ≥ 0 (non-negative CI); `C_forcedfit` − C ≤ 0 |
| EE6 | the current selection score remains misaligned | C − `C_posterior` on the agreement, CI > 0 |
| EE7 | the reduction keeps the right atoms | recall of the true atoms after reduction ≥ 0.9 × recall before |
| EE8 | the engine withstands false atoms | controlled case: true + 3 false, recovery within 0.05 of the true atoms alone |
| EE9 | a missing atom is flagged | controlled case: decision other than KNOWN in ≥ 80% of the tasks |
| EE10 | out-of-class is flagged | out-of-class tasks: decision other than KNOWN in ≥ 80%; recovered in-class tasks: KNOWN in ≥ 80% |
| EE11 | `flat_scene_k3` remains the best oracle-free source | agreement in the same engine, against the other sources |

For C, "candidate structure" denotes the top 20 classes according to the selector — not the whole table, which
would be unfair to A and B, whose pool contains only about twenty candidates.

Expectation declared from the tuning: C reaches 0.905 agreement on 60 tuning tasks, of the same order as what
B obtained on other sets. EE1 uncertain, EE2 unlikely, EE4 likely. In tuning, the validated fit
helps when the atoms are discovered (+0.017), the opposite of what was measured with the true atoms.

Limits known before launch:

- the thresholds of the KNOWN / UNKNOWN / NONE_OF_THE_ABOVE decision are untuned default values; the old
  class-boundary infrastructure requires a model fitted on validation, which does not exist: it is not
  reused, only a decision with two explicit signals is evaluated;
- the controlled cases (false atoms, missing atoms) start from the true atoms: they are oracle diagnostics,
  reported separately.

### Verdict rule and confirmatory decision, fixed in advance

- `READY FOR 16-SEED CONFIRMATORY CAMPAIGN`: EE1, EE2, EE3 and EE4 all supported. Only then, propose the
  16-seed campaign — without launching it.
- `END-TO-END IMPROVED BUT INSUFFICIENT`: EE1 supported, but EE2 or EE3 not supported.
- `END-TO-END STILL BLOCKING`: EE1 not supported.

In the last two cases: identify the new bottleneck and stop. The sealed set is launched in no case.

## 1. Protocol executed

- AWS run `wce-deepee-20261009-224553` (EC2 + SSM, 2 × c6i.xlarge, 71 and 73 min, 2,400 sub-units, 0 errors,
  13,536 lines). Commands: `make deep-ee-tune` (local, tuning), `make deep-run-ee`, `make deep-fetch`,
  `make deep-ee-aggregate`, `make deep-ee-report`. Summary: `results/deep_end_to_end_summary.json`.
- 384 never-used tasks, 64 per depth from 1 to 6; 96 out-of-class tasks for the class boundary.
- Code: `core/deep_e2e.py` (system, no oracle), `core/deep_e2e_exp.py` (evaluation), `core/deep_ee_report.py`.
  Tests: 50 new (`make deep-ee-test`); full suite 5,000 tests OK, 7 skipped; sealed set intact.
- One modification to the engine of the previous sub-milestone: `CellSpace` accepts up to 12 units when the occupied
  cells fit in 64 bits (an atom and its complement do not create a cell). Default behaviour unchanged.
- No tuning on the evaluation; configuration of C frozen before launch. One seed per task.
- Deviation from the protocol: none.

## 2. Main comparison

| Chain | Final agreement | Out of region | Candidate structure ≥ 0.97 | Final ≥ 0.97 | Final ≥ 0.99 | Literals | Seconds |
|---|---|---|---|---|---|---|---|
| A — frozen pipeline | 0.880 | 0.826 | 0.31 | 0.25 | 0.13 | 83 | 23.4 |
| B — `flat_scene_k3` + current downstream | **0.899** | **0.859** | **0.39** | **0.32** | **0.15** | 21 | 18.5 |
| C — `flat_scene_k3` + classes + validated finalisation | 0.888 | 0.845 | 0.22 | 0.23 | 0.09 | 18 → 4.6 | **1.4** |

Paired differences per task, 95% CI:

| Comparison | Final agreement | Candidate structure | Final ≥ 0.99 |
|---|---|---|---|
| C − A, all | +0.008 [−0.001, +0.015] | **−0.089 [−0.122, −0.055]** | −0.037 [−0.065, −0.010] |
| C − A, depths 3–6 | +0.010 [−0.002, +0.021] | −0.055 [−0.090, −0.020] | 0.000 |
| C − B, all | **−0.012 [−0.020, −0.005]** | −0.172 [−0.208, −0.135] | −0.055 [−0.083, −0.026] |
| C − B, depths 3–6 | −0.014 [−0.024, −0.005] | −0.137 [−0.180, −0.098] | 0.000 |
| B − A, all | +0.019 [+0.013, +0.026] | +0.083 [+0.050, +0.117] | +0.018 [0.000, +0.039] |
| B − A, depths 3–6 | +0.024 [+0.015, +0.033] | +0.082 [+0.043, +0.125] | 0.000 |

C beats neither A nor B. B again beats A, on a third independent set of tasks.

## 3. By depth

| Depth | Agreement A | Agreement B | Agreement C | Candidate A | Candidate B | Candidate C | Final ≥ 0.99: A / B / C |
|---|---|---|---|---|---|---|---|
| 1 | 0.987 | 0.985 | 0.984 | 0.98 | 0.98 | 0.86 | 0.66 / 0.67 / 0.47 |
| 2 | 0.930 | 0.952 | 0.937 | 0.52 | 0.69 | 0.33 | 0.11 / 0.20 / 0.08 |
| 3 | 0.895 | 0.917 | 0.905 | 0.17 | 0.34 | 0.05 | 0 / 0 / 0 |
| 4 | 0.871 | 0.894 | 0.866 | 0.11 | 0.19 | 0.06 | 0 / 0 / 0 |
| 5 | 0.795 | 0.831 | 0.819 | 0.03 | 0.09 | 0.00 | 0 / 0 / 0 |
| 6 | 0.802 | 0.817 | 0.815 | 0.02 | 0.03 | 0.00 | 0 / 0 / 0 |

From depth 3 onwards, none of the three chains recovers a single task at 0.99.

Recall of the candidates of chain C:

| Depth | True atoms present in the primitives | Present after reduction | False atoms kept | Right class (≥ 0.99) generated | In the top 20 | Exact tree | Equivalent class |
|---|---|---|---|---|---|---|---|
| 1 | 0.98 | 0.89 | 1.2 | 0.41 | 0.39 | 0.56 | 0.14 |
| 2 | 0.64 | 0.43 | 3.3 | 0.03 | 0.03 | 0.08 | 0.03 |
| 3 | 0.37 | 0.22 | 4.2 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 0.31 | 0.17 | 4.6 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | 0.22 | 0.12 | 5.0 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 0.17 | 0.08 | 4.9 | 0.00 | 0.00 | 0.00 | 0.00 |

A true atom is "present" if a primitive reproduces it at 0.95 agreement. At depth 6, a task has on
average one true atom out of six among its primitives, and five false atoms among the six kept.

## 4. Attribution of the failures of chain C

| Main cause | All (384) | Depths 3–6 (256) | Depths 4–6 (192) |
|---|---|---|---|
| recovered | 35 | 0 | 0 |
| correct primitive absent | **292** | 254 | 192 |
| right class not generated | 33 | 0 | 0 |
| correct primitive eliminated | 18 | 2 | 0 |
| degrading fit | 3 | 0 | 0 |
| too many false atoms | 3 | 0 | 0 |
| misranked, non-identifiable, out of class | 0 | 0 | 0 |

"Right class not generated" is concentrated at depth 1 (24 cases): the atom is present at 0.95 but not
precise enough for a class to reach 0.99 (agreement 0.97 on average). This is a defect of parameter precision, not of
composition.

## 5. Variants of chain C (secondary)

| Variant | Agreement | Final ≥ 0.97 | Final ≥ 0.99 | Agreement difference with C, 95% CI |
|---|---|---|---|---|
| C (validated fit, light penalty) | 0.888 | 0.23 | 0.09 | — |
| without fit | 0.877 | 0.19 | 0.06 | −0.011 [−0.014, −0.007] |
| forced fit | 0.887 | 0.24 | 0.10 | 0.000 [−0.002, +0.001] |
| selection by errors | 0.887 | 0.22 | 0.09 | −0.001 [−0.003, +0.002] |
| selection by existing posterior score | 0.875 | 0.25 | 0.12 | −0.012 [−0.017, −0.008] |
| one hypothesis per type (`k = 1`) | 0.821 | 0.20 | 0.11 | −0.067 [−0.085, −0.050] |
| five hypotheses per type (`k = 5`) | 0.896 | 0.25 | 0.08 | +0.009 [+0.003, +0.016] |

The validated fit accepts 79% of the proposals. The existing score loses in mean agreement but gains 3 points of
recovery at 0.99, all at depth 1, where preferring the smallest structure is right.

## 6. Sources of primitives, in the same engine

| Source | Oracle | Literals | True atoms present | After reduction | Agreement | Agreement, depths 4–6 |
|---|---|---|---|---|---|---|
| true atoms | yes | 3.5 | 1.00 | 0.87 | 0.936 | 0.898 |
| `flat_scene_k5` | no | 29 | 0.50 | 0.32 | 0.896 | 0.849 |
| `flat_scene_k3` | no | 18 | 0.45 | 0.32 | 0.888 | 0.833 |
| atoms of the frozen pipeline | no | 83 | 0.38 | 0.30 | 0.888 | 0.838 |
| fragments (`hier_k3`) | no | 5.6 | 0.29 | 0.27 | 0.847 | 0.783 |
| `flat_scene_k1` | no | 6.0 | 0.33 | 0.26 | 0.821 | 0.744 |
| robust discovery | no | 3.5 | 0.32 | 0.25 | 0.633 | 0.449 |

Against `flat_scene_k3`: `k5` +0.009 [+0.003, +0.016]; atoms of the frozen pipeline 0.000 [−0.007, +0.008]; fragments
−0.041; `k1` −0.067; robust discovery −0.255; true atoms +0.049 [+0.030, +0.067].

## 7. Reduction and scaling

Same primitives (`flat_scene_k3`, 14 hypotheses after grouping), no fit:

| Mechanism | Atoms kept | True atoms kept | Agreement | Classes | Seconds |
|---|---|---|---|---|---|
| confidence (identical to deduplication alone) | 6.0 | 0.31 | 0.845 | 150,000 | 0.43 |
| two per family | 6.0 | 0.32 | 0.850 | 151,000 | 0.44 |
| dominance | 6.0 | 0.30 | 0.852 | 196,000 | 0.52 |
| **floor error (retained)** | 3.0 | 0.30 | 0.869 | 69,000 | 0.15 |
| wide engine, 6 atoms | 6.0 | 0.31 | 0.774 | 1,100 | 0.10 |
| wide engine, 12 atoms | 11.9 | 0.38 | 0.855 | 1,400 | 0.20 |
| wide engine, all (14) | 14.1 | 0.40 | 0.876 | 1,400 | 0.24 |
| hierarchical, all (14) | 14.1 | 0.40 | 0.878 | 2,000 | 0.39 |

The recall of the true atoms is 0.40 before any reduction. Against the retained reduction: wide engine on all the
hypotheses +0.007 [+0.001, +0.014]; hierarchical +0.010 [+0.004, +0.017]; confidence −0.024 [−0.031, −0.015].

Going beyond six atoms works: fourteen atoms in a quarter of a second, without enumeration. It changes
almost nothing in the result, because the right atoms are not in the primitives.

## 8. Controlled cases (oracle diagnostic: we start from the true atoms)

Chain C without fit, primitives replaced by:

| Condition | Agreement | Final ≥ 0.99 | Atoms kept | False kept |
|---|---|---|---|---|
| true atoms | 0.944 | 0.79 | 2.9 | 0.0 |
| true + 1 false | 0.946 | 0.77 | 3.0 | 0.1 |
| true + 3 false | 0.949 | 0.77 | 3.2 | 0.3 |
| noisy parameters (standard deviation 0.10) | 0.854 | **0.02** | 2.7 | 0.7 |
| noisy parameters (standard deviation 0.25) | 0.723 | 0.00 | 2.4 | 2.0 |
| false atoms only | 0.391 | 0.00 | 1.6 | 1.6 |
| one missing atom | 0.620 | 0.01 | 1.8 | 0.0 |
| two missing atoms | 0.428 | 0.00 | 1.1 | 0.0 |

Against the true atoms alone: + 1 false −0.018 [−0.034, −0.005] of recovery; + 3 false −0.018
[−0.042, +0.005]; noisy parameters at 0.10 −0.771 [−0.813, −0.729].

Separate finding: with the true atoms and them alone, this chain recovers 0.79 of the tasks, against 0.99 for the
exhaustive composition of the previous sub-milestone. The difference comes from the reduction: forward selection on the
floor error keeps only 87% of the true atoms. An atom that acts only in combination with another (exclusive or)
does not reduce the floor error on its own and is never chosen. Recovery by depth: 1.00; 0.97; 0.78;
0.70; 0.70; 0.56.

## 9. KNOWN / UNKNOWN / NONE_OF_THE_ABOVE

| Situation | n | KNOWN | UNKNOWN | NONE_OF_THE_ABOVE |
|---|---|---|---|---|
| in class, chain C, task recovered | 35 | 35 | 0 | 0 |
| in class, chain C, task NOT recovered | 349 | 337 | 9 | 3 |
| controlled: true atoms | 384 | 346 | 28 | 10 |
| controlled: one missing atom | 320 | 136 | 125 | 59 |
| controlled: false atoms only | 384 | 3 | 202 | 179 |
| out of class: primitive absent | 32 | 32 | 0 | 0 |
| out of class: composition absent | 32 | 28 | 4 | 0 |
| out of class: relation absent | 32 | 29 | 3 | 0 |

The decision answers KNOWN in 97% of the non-recovered in-class tasks and in 93% of the out-of-class tasks. Its
main signal compares the errors of the structure with those of the nearest neighbours; yet a composition of approximate
primitives explains the observations as well as the neighbours while being false (mean gap −0.036). The
signal distinguishes only the coarse cases: false atoms alone (99% non-KNOWN), one missing atom (58%).

## 10. Preregistered hypotheses

| # | Result |
|---|---|
| EE1 C beats the frozen pipeline | **not supported**: agreement +0.008 [−0.001, +0.015]; candidate structure −0.089 [−0.122, −0.055] |
| EE2 C beats `flat_scene_k3` + current downstream | **not supported**: agreement −0.012 [−0.020, −0.005] |
| EE3 gain at depth 3–6 | **not supported**: +0.010 [−0.002, +0.021] against A; −0.014 [−0.024, −0.005] against B |
| EE4 without a compute explosion | **supported**: 1.4 s per task against 23.4 and 18.5 |
| EE5 the validated fit does not harm | **supported**: +0.011 [+0.007, +0.014] against the absence of fit; the forced fit is on a par |
| EE6 the existing score remains misaligned | **supported** on the agreement (+0.012 [+0.008, +0.017]); it nevertheless recovers 3 more points at 0.99, at depth 1 |
| EE7 the reduction keeps the right atoms | **not supported**: 0.32 after for 0.45 before (71%); 87% even when all the atoms are true |
| EE8 resistance to false atoms | **supported**: −0.018 [−0.042, +0.005] with three false |
| EE9 missing atom flagged (≥ 80%) | **not supported**: 58% |
| EE10 out-of-class flagged (≥ 80%) | **not supported**: 7%; the recovered tasks are indeed KNOWN (100%), but so are almost all the others |
| EE11 `flat_scene_k3` best source | **not supported**: `k5` does slightly better (+0.009), the atoms of the frozen pipeline are on a par |

## 11. Answers

**1. Does the class-based engine survive the discovered primitives?**
No. With the discovered primitives it brings no agreement gain over the frozen pipeline and it is worse than
the current downstream on the same primitives (−0.012). It produces a class correct at 0.99 in only 7% of the tasks,
all at depths 1 and 2. Its strength came from the exactness of the atoms: a noise of standard deviation 0.10 on the
parameters of the true atoms makes the recovery fall from 0.79 to 0.02.

**2. Does flat_scene_k3 remain the best source of primitives?**
It remains among the best without being alone in the lead. Five hypotheses per type do slightly better (+0.009), the
atoms of the frozen pipeline are on a par, the three other sources are clearly worse. No oracle-free
source exceeds 0.50 of true atoms present; with the true atoms the same engine reaches 0.936.

**3. Does the reduction of the literals preserve the right atoms?**
Imperfectly. The retained reduction keeps 71% of the true atoms present, and only 87% when it is given
only true ones. It misses the atoms that act only in combination. But it is not the blocker: keeping the
fourteen hypotheses with the wide engine gains only 0.007 of agreement.

**4. Does depth 4–6 remain viable?**
No, for none of the three chains. No task of depth 3 or more is recovered at 0.99, and the candidate
structure at 0.97 is reached in only 2 to 10% of the tasks of depth 4 to 6, depending on the chain. The cause is measured: 23% of the
true atoms are present there in the primitives.

**5. Are the false atoms the new bottleneck?**
No. Adding three false atoms to the true ones costs only 2 points of recovery: the engine discards them. They are
numerous (five out of six at depth 6), but that is the symptom of the absence of the true ones, not a cause of its own.

**6. Are the missing atoms the new bottleneck?**
Yes. "Correct primitive absent" is the main cause of 292 failures out of 349, and of all the failures from
depth 4 onwards. In the controlled setting, removing a single true atom makes the recovery fall from 0.79 to 0.01. Added to this is
precision: an atom that is present but approximate is not enough for 0.99.

**7. Is the final score still misaligned?**
Little, and it is no longer what limits. Errors and light penalty are on a par; the existing score loses 0.012
of agreement. No failure is attributed to a bad ranking: when the right class exists, it is chosen.

**8. Does the fit remain destructive?**
No, with discovered atoms it helps: +0.011 agreement in the validated version, and the forced version is on a par.
Only three failures are attributed to it, at depth 1. The result of the previous sub-milestone (the fit
destroys) holds for atoms that are already exact; for approximate atoms, fitting corrects more than it damages. The
two findings are consistent: the fit moves towards the approximate and away from the exact.

**9. Does the system know how to answer UNKNOWN / NONE_OF_THE_ABOVE?**
No. It answers KNOWN in 93% of the out-of-class tasks and in 97% of the in-class tasks it did not recover.
The existing class-boundary infrastructure was not reused (it requires a model fitted on
validation, which does not exist); the two-signal decision evaluated here is not enough.

**10. Is the chain ready for a confirmatory campaign?**
No. Chain C beats neither of the two baselines. I do not propose the 16-seed campaign.

## 12. New bottleneck

The discovery of the atoms, in recall and in precision. At depth 6, one discovered primitive out of six corresponds to
a true atom; at depths 4 to 6, 23%. All the rest of the chain has been shown functional with exact atoms
(composition, 0.99) or secondary (reduction, score, fit, false atoms). The previous upstream sub-milestones
had measured this same blocker piecemeal (parameters under occlusion, attribution, type) without lifting it.

Two results remain established:

- `flat_scene_k3` + current downstream beats the frozen pipeline on a third independent set (agreement +0.019
  [+0.013, +0.026], candidate structure +0.083 [+0.050, +0.117]);
- composition is not the blocker: the class-based engine goes to fourteen atoms in a quarter of a second.

## 13. Limits

- Dev only, one seed per task.
- Maximum depth 6 for A and B, without information on the true size: their figures are not directly
  comparable to those of the previous sub-milestones, which gave this information.
- For C, the "candidate structure" is restricted to the top 20 classes; over the whole table, a class at 0.97
  exists in any case in only 22% of the tasks.
- The configuration of C was tuned on 60 tasks; other settings (more hypotheses per type, wide engine) do
  slightly better as secondary, without approaching B.
- The KNOWN / UNKNOWN / NONE_OF_THE_ABOVE decision thresholds were not tuned.
- The defect of the reduction on atoms that act only in combination was observed on the evaluation; it was
  not corrected, so as not to tune on these tasks.

## 14. Verdict

Preregistered rule: EE1 not supported.

`END-TO-END STILL BLOCKING`

The 16-seed confirmatory campaign is not proposed. The sealed set was not touched. The frozen pipeline remains the
official reference, `flat_scene_k3` remains a candidate baseline, and nothing from this sub-milestone is wired in.

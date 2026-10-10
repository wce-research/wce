# WCE Deep Composition & Final Reconstruction with Oracle Atoms — report (dev, exploratory)

Status: exploratory sub-milestone, **dev** partition only. No confirmatory test, no 16-seed campaign,
sealed set never touched. The frozen pipeline remains the official reference; `flat_scene_k3` remains a candidate
baseline, kept out of this sub-milestone; nothing is wired in by default.

Throughout this sub-milestone the **true atoms** (types and parameters) are provided. The upstream is taken out of the equation.

## 0. Preregistration (written before the AWS runs were launched)

Tuning done only on composition key 899 (six tasks, one per depth). Evaluation on
**384 never-used tasks**: keys 800–831, depths 1 to 6 (number of atoms), standard and rich regimes.

Main method: `dp_errors` — bottom-up dynamic programming over observational equivalence classes
(cell mask), exhaustive up to the size of the truth, retained class = fewest errors, then smallest;
no parameter fitting. Reference: `ref_b150` — existing pipeline, same atoms, reference budget.

| # | Hypothesis | Criterion |
|---|---|---|
| DC1 | with the true atoms, the class of the true structure is always built | share of tasks = 1.00, and fewer than 10⁶ pairs evaluated at depth 6 |
| DC2 | composition over classes recovers better than the existing pipeline | recovery ≥ 0.99, `dp_errors` − `ref_b150`, 95% CI > 0, overall and at each depth 2–6 |
| DC3 | ranking is not the blocker when ranking by errors; it is with the current penalties | true class at rank 1 by errors in ≥ 90% of tasks; `dp_posterior` − `dp_errors` on recovery, CI < 0 |
| DC4 | intrinsic ambiguity is small | mass of never-observed cells < 5% on average; when the retained class is that of the truth, agreement ≥ 0.99 in ≥ 95% of cases |
| DC5 | budget is not the problem of the existing pipeline | `ref_b450` − `ref_b150` and `ref_b1500` − `ref_b150` on agreement: CI including 0 or effect < 0.01 |
| DC6 | fitting the composite is harmful | agreement, C − A: CI < 0, on the true structure; D (validated) − A: CI including 0 or negative |
| DC7 | meet in the middle reaches the exhaustive result with far fewer evaluations | `mitm3` equals the exhaustive in ≥ 95% of tasks |
| DC8 | a beam of 100 classes is enough | true class kept in ≥ 95% of tasks |
| DC9 | the shape of the tree changes nothing for composition over classes | recovery chain versus balanced: gap < 0.05 |
| DC10 | the effective space remains compact | cumulative classes at depth 6 < 10⁵ on average; less than one second |

Expectation declared from the tuning (six tasks): all likely, except DC7 uncertain (the guided search
sometimes stops on a nearly equivalent structure) and DC8 uncertain at depth 6.

Two limits known before launch:

- `dp_errors` receives the number of true atoms as maximum size. The "one more leaf" variant
  (`dp_errors_plus1`, `dp_bic_plus1`) is reported alongside to measure what this information is worth.
- With exact atoms, "recovering" means choosing the right Boolean function of the atoms; no parameter
  is estimated. This result says nothing about imperfect atoms.

Verdict rule fixed in advance. It bears on composition with given atoms, and on nothing else:

- `DEEP COMPOSITION READY FOR END-TO-END VALIDATION`: DC2 supported AND recovery ≥ 0.99 of `dp_errors` at least
  equal to 0.80 at each depth from 2 to 6.
- `DEEP COMPOSITION IMPROVED BUT INSUFFICIENT`: DC2 supported, but the 0.80 bar missed at at least one depth.
- `DEEP COMPOSITION STILL BLOCKING`: DC2 not supported.

## 1. Protocol executed

- AWS run `wce-deepdc-20261009-212700` (EC2 + SSM, 2 × c6i.xlarge, 28 and 23 min, 2,512 sub-units, 0 errors).
  Commands: `make deep-run-dc`, `make deep-fetch`, `make deep-dc-aggregate`, `make deep-dc-report`.
  Summary: `results/deep_composition_oracle_summary.json`.
- 384 never-used tasks: 64 per depth (number of atoms) from 1 to 6, half standard (300 observations, 5%
  noise), half rich (600 observations, 2%). True atoms provided everywhere.
- Code: `core/deep_compose.py` (system: receives atoms and observations, sees neither truth nor world),
  `core/deep_compose_exp.py` (evaluation), `core/deep_dc_report.py`. Tests: 56 new (`make deep-dc-test`); full
  suite 4,950 tests OK, 7 skipped; sealed set intact.
- No tuning on the evaluation. One seed per task. No learned guide was introduced: the explicit methods
  did not plateau.
- Deviation from the protocol: none. Instrumentation defect noted after the fact: in the scaling law, the column of
  distinct masks repeats that of the new classes; it is not cited.

### Principle of the engine

With fixed atoms, a Boolean composition is entirely described by its value on the **cells**: the
groups of observations that have the same vector of atom values. Six atoms give at most 64 cells, of which 21
are occupied on average at depth 6. A composition reduces to a 64-bit mask. This mask serves at once
as functional signature, as observational equivalence class key and as a means of counting errors without
executing the program.

## 2. Main result

Functional recovery (agreement ≥ 0.99 over the whole region):

| Depth | Existing pipeline, true atoms (`ref_b150`) | The same without fitting | The same with guardrail | Composition over classes (`dp_errors`) |
|---|---|---|---|---|
| 1 | 0.67 | 0.75 | 0.73 | 1.00 |
| 2 | 0.23 | 0.97 | 0.80 | 1.00 |
| 3 | 0.06 | 0.94 | 0.58 | 1.00 |
| 4 | 0.00 | 0.84 | 0.48 | 1.00 |
| 5 | 0.00 | 0.78 | 0.47 | 0.97 |
| 6 | 0.00 | 0.72 | 0.38 | 0.95 |
| all | 0.16 | 0.83 | 0.57 | **0.99** |

`dp_errors` − `ref_b150`, paired difference per task, 95% CI: recovery +0.826 [+0.787, +0.862]; agreement
+0.049 [+0.043, +0.055]; agreement outside the observed region +0.057 [+0.046, +0.069]. The interval is above
zero at each depth, from +0.33 [+0.22, +0.45] at depth 1 to +0.95 [+0.89, +1.00] at depth 6.

Mean agreement: `dp_errors` 0.9996 (0.9994 in the observed region, 1.0000 outside the region); `ref_b150` 0.951
(0.954 / 0.943).

## 3. Decomposition by depth

Composition over classes, exhaustive up to the size of the truth:

| Depth | True class built | Mean rank by errors | Rank 1 | Exact tree | Same function on the observations | Functional | Agreement |
|---|---|---|---|---|---|---|---|
| 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.0000 |
| 2 | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 1.00 | 0.9999 |
| 3 | 1.00 | 1.02 | 0.98 | 0.73 | 0.98 | 1.00 | 0.9999 |
| 4 | 1.00 | 1.09 | 0.94 | 0.64 | 0.94 | 1.00 | 0.9996 |
| 5 | 1.00 | 1.06 | 0.94 | 0.39 | 0.94 | 0.97 | 0.9994 |
| 6 | 1.00 | 1.20 | 0.89 | 0.44 | 0.89 | 0.95 | 0.9985 |

Existing pipeline, same atoms, per-step trace (160 tasks, depths 2 to 6):

| Depth | Good candidate (≥ 0.99) before fitting | Its median rank | Agreement of the first candidate before fitting | Final agreement | Final ≥ 0.99 | Dominant cause of the loss |
|---|---|---|---|---|---|---|
| 2 | 1.00 | first | 1.000 | 0.979 | 0.31 | initial fitting (16 of 22 losses) |
| 3 | 1.00 | first | 0.999 | 0.970 | 0.03 | initial fitting (25 of 31) |
| 4 | 1.00 | first | 0.991 | 0.936 | 0.00 | initial fitting (30 of 32) |
| 5 | 1.00 | first | 0.991 | 0.932 | 0.00 | initial fitting (32 of 32) |
| 6 | 0.97 | first | 0.979 | 0.907 | 0.00 | initial fitting (31 of 32) |

Over the 160 traced tasks: initial fitting 134, recovered 11, threshold too strict 10, ranking 3, others 2.
Simplification changes nothing (identical agreement with and without).

Existing pipeline, final pool (384 tasks):

| Arm | True shape in the pool | Good candidate in the pool | Good candidate found then not retained | Exact tree retained | Recovery |
|---|---|---|---|---|---|
| `ref_b150` | 0.84 | 0.20 | 0.04 | 0.49 | 0.16 |
| without fitting | 0.83 | 0.98 | 0.15 | 0.58 | 0.83 |
| with guardrail | 0.83 | 0.84 | 0.27 | 0.54 | 0.57 |

## 4. Composition space and scaling law

| Atoms | Occupied cells | Compositions evaluated | Equivalence classes | Time (s) | Classes nearly equivalent to the truth | Classes strictly better than the truth |
|---|---|---|---|---|---|---|
| 1 | 2.0 | 1 | 1 | 0.0000 | 1.0 | 0.00 |
| 2 | 3.9 | 14 | 6 | 0.0002 | 1.1 | 0.00 |
| 3 | 6.4 | 106 | 28 | 0.0004 | 1.6 | 0.00 |
| 4 | 10.3 | 1,748 | 276 | 0.0009 | 2.6 | 0.08 |
| 5 | 15.5 | 26,173 | 4,007 | 0.005 | 4.4 | 0.03 |
| 6 | 20.7 | 671,588 | 71,257 | 0.11 | 6.5 | 0.17 |

Each additional atom multiplies the number of classes by a factor that grows: 5, 10, 14 then 18. At depth 6: 71,000 classes on average,
337,000 at most, 0.6 s at worst. "Compositions evaluated" counts the pairs of classes combined by an operator,
not raw programs: each substructure is built only once.

New classes per size, with six atoms: 6; 38; 250; 1,698; 10,510; 58,754.

Exhaustive enumeration at depths 2 and 3: 14 and 106 compositions, 6 and 28 classes; the true class is first
by errors in 100% and 98% of tasks; 1.1 and 1.6 class is within less than 3% of the truth on the observations.

Signature on the observations versus signature extended to the whole region: at depth 5, 4,007 classes versus
4,867, so 18% of the classes that are distinct elsewhere are conflated on the observations. The functional effect is
negligible: 0.2% of the points of the region fall in a never-observed cell (0.6% at depth 6).

## 5. Ranking

Same table of classes, five selectors (functional recovery):

| Selector | All | d4 | d5 | d6 | Mean rank of the true class |
|---|---|---|---|---|---|
| errors, then size | 0.987 | 1.00 | 0.97 | 0.95 | 1.06 |
| profiled likelihood | 0.987 | 1.00 | 0.97 | 0.95 | 1.06 |
| likelihood − ½·ln N per leaf | 0.984 | 0.98 | 0.97 | 0.95 | 1.06 |
| existing posterior score | 0.906 | 0.91 | 0.80 | 0.77 | 1.35 |
| smallest structure | 0.167 | 0.00 | 0.00 | 0.00 | 2,130 |

Posterior score − errors: −0.081 [−0.109, −0.055] of recovery. Its penalties (structure and parameters,
about 13 nats per atom) make it retain a structure that is too small: mean size 5.4 instead of 6 at depth 6.

Without knowing the true size (one more leaf allowed, depths 1 to 5): 0.991 by errors, 0.991 with the
light penalty. The size information does not matter.

## 6. Equivalence

| | Share of the 384 tasks |
|---|---|
| exact tree (up to associativity and commutativity) | 0.69 |
| different tree, same function on the observations | 0.27 |
| different function on the observations, agreement ≥ 0.99 | 0.03 |
| functional failure | 0.013 (5 tasks) |
| more compact than the truth and functional | 0.010 (4 tasks) |

When the retained class is that of the truth (368 tasks), agreement is always ≥ 0.9957. Among the 118 tasks where
the tree is not exact, 96% are functionally correct. The 16 tasks where the class differs are those where the
noise makes another function as good or better on the observations (11 times strictly fewer errors than
the truth); 11 remain functional.

The 5 failures: depths 5 and 6, four chains and one mixed tree, agreements from 0.965 to 0.989. In one of them, 7.6% of the
region is in never-observed cells.

## 7. Bounded searches

Beam over the classes (true class kept):

| Beam | All | d4 | d5 | d6 | Compositions evaluated at d6 |
|---|---|---|---|---|---|
| 10 classes | 0.72 | 0.70 | 0.45 | 0.17 | 2,034 |
| 100 classes | 0.95 | 1.00 | 0.88 | 0.80 | 62,289 |
| 1,000 classes | 0.98 | 1.00 | 0.97 | 0.89 | 387,053 |

Budget in compositions evaluated (true class reached):

| Budget | d5 | d6 | Mean agreement at d6 |
|---|---|---|---|
| 100 | 0.00 | 0.00 | 0.791 |
| 1,000 | 0.03 | 0.00 | 0.854 |
| 10,000 | 0.72 | 0.03 | 0.924 |
| 100,000 | 1.00 | 0.52 | 0.977 |
| exhaustive | 1.00 | 1.00 | 0.9985 |

Meet in the middle with constrained holes:

| Table | Functional, all | d4 | d5 | d6 | Equals the exhaustive | Joins at d6 |
|---|---|---|---|---|---|---|
| size ≤ 2 | 0.92 | 0.97 | 0.83 | 0.72 | 0.91 | 2,051 |
| size ≤ 3 | 0.91 | 1.00 | 0.78 | 0.66 | 0.91 | 62 |

Against the exhaustive: −0.068 [−0.094, −0.042] and −0.081 [−0.109, −0.055] of recovery. Mean agreement stays at
0.997: the guided search stops on nearly equivalent structures.

Shape of the tree (`dp_errors` / existing pipeline): balanced 1.00 / 0.00; chain 0.97 / 0.00; mixed 0.98 / 0.00.
By number of exclusive ORs: 1.00 up to 1; 0.99 at 2; 0.94 at 3. A single operator: 1.00; mixed operators: 0.98.

## 8. Fitting the composite

Start: the TRUE structure with the TRUE parameters (agreement 1.000).

| Variant | Mean agreement | Recovery ≥ 0.99 | Difference in agreement with A, 95% CI |
|---|---|---|---|
| A — no fitting | 1.000 | 1.00 | — |
| B — each atom re-fitted alone | 0.867 | 0.08 | −0.133 [−0.143, −0.123] |
| C — fitting of the composite (current) | 0.962 | 0.17 | −0.038 [−0.042, −0.035] |
| D — the same, validated | 0.988 | 0.77 | −0.012 [−0.014, −0.009] |

Same figures on the structure retained by `dp_errors`. Fitting C even increases the errors on the
observations (37.5 → 39.3 on average). The validated version accepts 25% of the proposals; these acceptances make
the real agreement drop while the observable criterion rises.

The budget of the existing pipeline is not a factor: the results at 50, 150, 450 and 1,500 are identical task by
task. It stops by itself around 28 fit-equivalents, after having polished about twenty candidates.

## 9. Preregistered hypotheses

| # | Result |
|---|---|
| DC1 true class always built, fewer than 10⁶ pairs at d6 | **supported** for the construction (1.00); the bound holds on average (0.67 million) and in 77% of the depth-6 tasks, not at worst (2.0 million) |
| DC2 composition over classes > existing pipeline | **supported**: +0.826 [+0.787, +0.862], and at each depth |
| DC3 ranking | **supported**: rank 1 by errors in 96% of tasks; existing score −0.081 [−0.109, −0.055] |
| DC4 intrinsic ambiguity small | **supported**: 0.2% of unobserved mass; agreement ≥ 0.99 in 100% of the cases where the class is the right one |
| DC5 budget is not the problem | **supported**: difference exactly zero at 3 × and at 10 × |
| DC6 fitting the composite is harmful | **supported**: C − A −0.038 [−0.042, −0.035]; D − A −0.012 [−0.014, −0.009] |
| DC7 meet in the middle = exhaustive in ≥ 95% | **not supported**: 0.91 (0.63 at depth 6) |
| DC8 beam of 100 classes ≥ 95% | **not supported**: 0.945 (0.80 at depth 6) |
| DC9 shape of the tree indifferent | **supported**: balanced 1.00, chain 0.97 |
| DC10 compact space | **supported**: 71,000 classes and 0.11 s on average at depth 6 |

## 10. Answers

**1. Why are the true atoms not enough?**
Because the pipeline modifies them. With the true atoms, a candidate at 0.99 or more exists before fitting in 99%
of tasks, and it is at the top of the ranking. The initial fitting degrades it: this is the cause of 134 losses out of 149.
Even a single atom, given with its true parameters, is returned correct only two times out of three. The rest comes from the
score, which penalises size too much. It is neither the composition nor the search space.

**2. At what depth does composition collapse?**
It does not collapse when nothing is fitted: 1.00 up to depth 4, 0.97 at 5, 0.95 at 6. It is the existing
pipeline that collapses, from depth 2 (0.23) and totally from 4 onwards (0.00).

**3. Is the true structure generated?**
Yes. The class of the true structure is built in 100% of tasks by composition over classes. In the
existing pipeline, the true shape is in the pool in 84% of tasks (47% at depth 6).

**4. Is it lost at ranking?**
Not with a ranking by errors: rank 1 in 96% of tasks, mean rank 1.06. Yes, in part, with the existing posterior
score: it loses 8 points of recovery, 18 at depth 6, by preferring structures that are too small.

**5. Does functional equivalence explain part of the failure?**
It explains the structural gaps, not failures. In 27% of tasks the retained tree differs from the truth while
computing the same function, and in 3% a slightly different function remains correct. Counting "exact tree"
would suggest 31% of failures where there are 1.3%. The truly intrinsic ambiguity (never-observed cells)
weighs 0.2% of the region.

**6. Does dynamic programming help?**
Yes, it is what makes the problem small. Building each substructure once and keeping only one structure per
class brings depth 6 down to 71,000 classes and a tenth of a second.

**7. Does meet-in-the-middle help?**
No, not here. It uses 10 to 10,000 times fewer evaluations, but it loses 7 to 8 points of recovery and equals
the exhaustive only in 63 to 67% of tasks at depth 6. Since the exhaustive already costs a tenth of a second, there is
nothing to gain up to six atoms.

**8. Is the problem mainly budget or strategy?**
Neither for the existing pipeline: ten times more budget gives exactly the same result, because it
does not consume its budget. For composition over classes, budget matters at depth 6: 100,000
compositions reach the true class only one time in two, 670,000 are needed on average. It is a cost, not an
obstacle.

**9. Does the composite fit help or harm?**
It harms, with no measured exception. Starting from the true structure and the true parameters, it makes
recovery fall from 1.00 to 0.17. Validated, it climbs back to 0.77 but stays below the absence of fitting (−0.012
[−0.014, −0.009]): the proposals it accepts improve the observable criterion and degrade the real agreement.

**10. Can deep composition be made viable without brute force?**
Yes, with given atoms and up to six atoms. The enumeration bears on the equivalence classes, not on the programs,
and remains exhaustive for a negligible cost. The limit is structural: the mask assumes at most six atoms, and the
number of classes is multiplied by a growing factor (18 from the fifth to the sixth atom). Beyond that, a beam will be needed, which already loses 20% of the
true classes at depth 6 with 100 classes kept.

**Answer to the question of the sub-milestone.** Depth is difficult neither because the space explodes, nor because
structures are intrinsically equivalent. With the true atoms, it was difficult because the fitting
destroyed the good solutions and because the score preferred structures that are too small.

### What this implies for `flat_scene_k3` (not measured)

Nothing in this sub-milestone was executed on `flat_scene_k3`. Two implications to verify, not results:

- `flat_scene_k3` goes through the same fitting and the same score as the existing pipeline. If the degradation measured
  here is found there too, part of its gap to the true atoms would come from the downstream and not from its primitives.
- The engine over classes is not applicable to it as is: it provides about twenty literals, the mask
  accepts six. And with imperfect literals, "not fitting" is no longer obviously the right choice.

## 11. Limits

- Exact atoms: no parameter is estimated. The result does not say what becomes of composition over classes with
  approximate atoms, nor with false atoms mixed with the true ones.
- At most six atoms, by construction of the mask.
- Recovery by errors assumes that all the atoms provided are useful; with superfluous atoms, minimising
  the errors would overfit. The light per-leaf penalty does as well here (0.984), but has not been put to
  the test of superfluous atoms.
- Dev only, one seed per task, no confirmatory test.
- The existing pipeline was compared only in its factorised strategy, the only one that accepts provided atoms.
- The per-step trace covers 160 of the 384 tasks (even keys, depths 2 to 6).
- Meet in the middle and beam: simple implementations; their relative failure holds for these versions.

## 12. Verdict

Preregistered rule: DC2 supported, and recovery of `dp_errors` of 1.00; 1.00; 1.00; 0.97; 0.95 at depths
2 to 6, hence above 0.80 everywhere.

`DEEP COMPOSITION READY FOR END-TO-END VALIDATION`

Exact scope of this verdict: composition, **with exact atoms provided**, is no longer a blocker and can enter
an end-to-end validation. It does not say that the full chain is ready: the upstream remains at the level of the previous
sub-milestone, and the parameter fitting, as it exists, destroys the solutions. The frozen pipeline remains the official
reference; nothing from this sub-milestone is wired into it.

Recorded scope of this result: true atoms; true parameters; up to 6 atoms; no global confirmatory
result; no sealed set. Composition over classes is not the official pipeline.

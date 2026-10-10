# WCE Latent Primitive Recovery — report (dev, exploratory)

Status: exploratory sub-milestone, **dev** partition only. No multi-seed campaign, sealed set never
touched. The frozen pipeline remains the official reference; `flat_scene_k3` is a *confirmed dev baseline candidate*
(three independent dev sets), not a confirmatory baseline; nothing is wired in by default.

## 0. Preregistration (written before the AWS runs were launched)

Tuning done only on composition keys 1090–1099 (adjustment on 60 tasks, keys 1090–1094).
Evaluation on **384 tasks never used**: keys 1000–1031, depths 1 to 6, standard and rich regimes.

### Main experiment: three sources of primitives, a single downstream

| Chain | Primitives | Downstream |
|---|---|---|
| **A** | atoms of the frozen pipeline | strong downstream |
| **B** | `flat_scene_k3` | strong downstream |
| **C** | `LatentPrimitiveSetRecovery` | strong downstream |

Strong downstream, identical for all three: primitive hypotheses → compatible sets (conditional utility,
rescue by pairs, three best sets) → composition over equivalence classes per set → validated
finalisation. No chain receives an oracle; six atoms at most per set.

C adds, upstream of and around this downstream: a "recall first" pool (five parametrisations per type, plus the
hypotheses coming from the fragments and from the consensus components, without elimination), the reinjection of the
fitted primitives, and a bounded loop (two retries at most) that proposes primitives on the RESIDUAL of the composition;
a retry is kept only if the errors on the observations decrease.

Configuration frozen before the launch (`results/deep_configs/lpr.json`): minimal gain 1; the loop is always
attempted. In tuning, the six configurations tried give an agreement between 0.899 and 0.902.

Preregistered secondary arm: **B5** (`flat_scene_k5`, same downstream). Variants of C, secondary: without loop, without
pairs, without reinjection, `k = 3`, `flat` pool alone. Historical chains reported for continuity: `A_full`
(complete frozen pipeline) and `B_full` (`flat_scene_k3` + current downstream). No other arm will be added.

### Primary metrics

All true primitives present (in the hypothesis pool, and in the retained set); conditional parameter
error; candidate structure ≥ 0.97; final ≥ 0.99; final agreement; compute.

A true primitive is "present" if a candidate primitive reproduces it at 0.95 functional agreement ("strict":
0.99; "exact": same type and parameters within tolerance). The three levels are reported separately.

### Hypotheses

| # | Hypothesis | Criterion |
|---|---|---|
| LP1 | C improves SET recall | share of tasks where all true primitives are present in the pool: C − B and C − A, 95% CI > 0 |
| LP2 | C improves the downstream | C − B and C − A on the candidate structure AND on the final agreement, CI > 0 |
| LP3 | the "recall first" regime increases individual recall | mean recall in the pool, C − B, CI > 0 |
| LP4 | false positives remain tolerable | true + 10 false: recovery within 0.05 of the true ones alone |
| LP5 | the reduction defect is corrected | true atoms alone: all kept, set selection − old reduction, CI > 0 |
| LP6 | several parametrisations per type help | B5 − B on set recall and on agreement, CI > 0 |
| LP7 | the composition residual recovers a missing primitive | controlled, one true atom removed: recovered in ≥ 50% of tasks, and recovery after > before |
| LP8 | the loop helps in real conditions | C − `C_noloop` on agreement, CI > 0 |
| LP9 | misses are not independent | positive correlation between misses of the same task; deviation from the `recall^n` model |
| LP10 | compute remains compact | C under 5 s per task, and under `A_full` and `B_full` |
| LP11 | campaign condition | C beats A, B, `A_full` and `B_full` on final agreement, CI > 0 |

The sensitivity curve (parameter error → recovery, per type and per parameter family) and the tolerance
to false atoms (+1, +3, +5, +10) are descriptive. The exploration "incomplete set or out-of-class world" is
exploratory and will not be presented as a confirmatory result.

Expectation declared from the tuning: agreement of C of 0.902 on 60 tasks, of the same order as the chain of the
previous sub-milestone; LP2 and LP11 unlikely; LP5 and LP7 likely (verified on constructed cases); LP8
probably null. Set recall could not be read in tuning (measurement defect corrected before the run).

### Verdict rule and campaign decision, fixed in advance

- `READY FOR 16-SEED CONFIRMATORY CAMPAIGN`: LP1, LP2, LP10 and LP11 all supported. Only then, propose the
  campaign — without launching it.
- `LATENT PRIMITIVE RECOVERY IMPROVED BUT INSUFFICIENT`: LP1 supported (set recall progresses), but LP2 or
  LP11 not supported. Improving individual recall alone (LP3) is not enough for this verdict.
- `LATENT PRIMITIVE RECOVERY STILL BLOCKING`: LP1 not supported.

In the last two cases: identify the new blocker and stop. The sealed set is launched in no case.

## 1. Protocol executed

AWS run `wce-deeplp-20261010-040904` (EC2 + SSM, `eu-north-1`, 2 × `c6i.xlarge`, 432 sub-units per instance, 0 failures,
instances terminated). 18,760 aggregated rows, 0 units in error (`results/deep_latent_primitive_summary.json`).
384 dev tasks never seen (keys 1000–1031, depths 1–6, 64 tasks per depth). No arm added after the
preregistration. Paired differences per task, 95% bootstrap CI. A single task seed per key: nothing
here is confirmatory.

## 2. Main comparison (same strong downstream)

| Chain | Final agreement | Agreement d4–6 | Final ≥ 0.99 | Candidate ≥ 0.97 | Pool recall | All present (pool) | At most one absent | All kept | Param. error | Hypotheses | Time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A (frozen) | 0.895 | 0.840 | 0.075 | 0.297 | 0.413 | 0.242 | 0.354 | 0.198 | 0.134 | 36.5 | 28.3 s |
| B (`flat_scene_k3`) | 0.896 | 0.844 | 0.094 | 0.242 | 0.427 | 0.224 | 0.341 | 0.164 | 0.065 | 14.2 | 2.8 s |
| B5 (`flat_scene_k5`) | 0.898 | — | 0.091 | 0.268 | 0.477 | 0.245 | — | — | — | 23.3 | 3.3 s |
| **C (LPSR)** | 0.902 | 0.858 | 0.088 | 0.315 | 0.546 | 0.271 | 0.451 | 0.195 | 0.067 | 45.2 | 9.0 s |
| `A_full` | 0.883 | — | 0.115 | 0.320 | — | — | — | — | — | — | 26.1 s |
| `B_full` | 0.902 | — | 0.151 | 0.378 | — | — | — | — | — | — | 20.6 s |

Paired differences (all tasks):

| Difference | Final agreement | Candidate ≥ 0.97 | All present (pool) | Pool recall |
|---|---|---|---|---|
| C − A | +0.008 [+0.002, +0.014] | +0.018 [−0.010, +0.050] | +0.029 [−0.003, +0.060] | +0.133 [+0.102, +0.162] |
| C − B | +0.007 [+0.001, +0.013] | +0.073 [+0.044, +0.104] | +0.047 [+0.026, +0.068] | +0.119 [+0.100, +0.139] |
| C − B5 | +0.005 [−0.001, +0.010] | — | +0.026 [+0.010, +0.042] | — |
| C − `A_full` | +0.020 [+0.013, +0.026] | — | — | — |
| C − `B_full` | +0.001 [−0.005, +0.006] | −0.063 [−0.094, −0.034] | — | — |

On "final ≥ 0.99", C beats no one: C − B −0.005; C − `A_full` −0.026; C − `B_full` −0.063 [−0.091, −0.034].
At depths 4–6, the agreement of C exceeds A (+0.018 [+0.009, +0.027]), B (+0.014 [+0.005, +0.023]) and `B_full`
(+0.010 [+0.001, +0.018]). B − A on agreement: +0.001, not significant.

Variants of C (all tasks, agreement): without loop 0.900; without pairs 0.904; without reinjection 0.900; `k = 3`
0.905; `flat` pool alone 0.903. None differs from C on agreement (CI including 0). The fragment and
component sources add individual recall (+0.024 [+0.009, +0.039]) with no effect on the set or on the downstream.

## 3. Scaling law of set recall

| Atoms | Atom recall B | All present B | `recall^n` B | Atom recall C | All present C | `recall^n` C | At most one absent C | All kept C | Mean visibility |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.984 | 0.984 | 0.984 | 0.984 | 0.984 | 0.984 | 1.000 | 0.828 | 1.00 |
| 2 | 0.531 | 0.344 | 0.282 | 0.680 | 0.484 | 0.462 | 0.875 | 0.313 | 0.85 |
| 3 | 0.365 | 0.016 | 0.049 | 0.495 | 0.125 | 0.121 | 0.484 | 0.031 | 0.74 |
| 4 | 0.281 | 0.000 | 0.006 | 0.453 | 0.031 | 0.042 | 0.297 | 0.000 | 0.73 |
| 5 | 0.209 | 0.000 | 0.000 | 0.344 | 0.000 | 0.005 | 0.047 | 0.000 | 0.67 |
| 6 | 0.190 | 0.000 | 0.000 | 0.320 | 0.000 | 0.001 | 0.000 | 0.000 | 0.68 |

Chain A: atom recall 1.000 / 0.617 / 0.349 / 0.203 / 0.184 / 0.125; all present 1.000 / 0.438 / 0.016 / 0 /
0 / 0. Set recall closely follows the independent model `recall^n` for all chains. Two effects
multiply: the per-atom recall itself drops with depth (C: 0.98 → 0.32), then it is raised to the
power n.

Correlation of misses between atoms of the same task (2,240 pairs): phi = +0.064 (A), −0.016 (B), +0.008 (B5),
+0.050 (C). P(miss | other missed) is slightly LOWER than P(miss): 0.611 versus 0.625 for C. Pairs of the same
type (370): phi +0.146 (A), −0.007 (B), +0.176 (C); different types: +0.048, −0.019, +0.026. No CI computed on
phi.

Recall per type (pool):

| Type | A | B | B5 | C |
|---|---|---|---|---|
| half-plane | 0.522 | 0.430 | 0.530 | 0.661 |
| band | 0.315 | 0.402 | 0.442 | 0.504 |
| disc | 0.188 | 0.337 | 0.396 | 0.525 |
| anti-disc | 0.151 | 0.247 | 0.307 | 0.360 |
| ring | 0.158 | 0.213 | 0.225 | 0.263 |
| sector | 0.276 | 0.181 | 0.238 | 0.319 |

Recall according to the visibility of the atom (share of its boundary visible in the scene):

| Visibility | n | B | C |
|---|---|---|---|
| 0.0–0.2 | 68 | 0.029 | 0.059 |
| 0.2–0.4 | 186 | 0.043 | 0.134 |
| 0.4–0.6 | 214 | 0.084 | 0.234 |
| 0.6–0.8 | 212 | 0.269 | 0.420 |
| 0.8–1.0 | 664 | 0.494 | 0.642 |

Even an almost entirely visible atom is recovered only 64% of the time by C. Conditional parameter error
of C, per type: sector — vertex 0.462, direction 0.181, half-aperture 0.136; disc — centre 0.098, radius 0.049;
anti-disc — centre 0.105, radius 0.053; half-plane — orientation 0.049, offset 0.073; band — 0.015 / 0.040 /
0.057; ring — centre 0.039, radii 0.046 / 0.023.

## 4. Sensitivity: parameter error → recovery

True atoms perturbed, tasks where the unperturbed true atoms give a functional structure (n = 372).

| Perturbation σ | Final ≥ 0.99 | Final ≥ 0.97 | Agreement |
|---|---|---|---|
| 0.02 | 0.307 | 0.850 | 0.982 |
| 0.05 | 0.065 | 0.347 | 0.954 |
| 0.10 | 0.016 | 0.118 | 0.917 |
| 0.20 | 0.005 | 0.030 | 0.852 |

Per family (final ≥ 0.99 at σ = 0.02 / 0.05 / 0.10): orientation 0.61 / 0.25 / 0.15; centre 0.76 / 0.36 / 0.15;
offset 0.83 / 0.44 / 0.21; radius 0.83 / 0.39 / 0.22; width 0.85 / 0.61 / 0.36. Per type, at σ = 0.05: band 0.17;
ring 0.27; sector 0.35; half-plane 0.48; disc 0.53; anti-disc 0.62. Orientation is the most fragile
family, the band the most fragile type.

The mean parameter error of the primitives recovered by C (0.067) lies beyond σ = 0.05, where the
recovery at 0.99 falls to 6%.

## 5. False positives and correction of the reduction

True atoms plus k false ones, set selection:

| False added | Agreement | Final ≥ 0.99 | All kept | False kept | Final ≥ 0.99 (d4–6) |
|---|---|---|---|---|---|
| 0 | 0.995 | 0.945 | 0.943 | 0.00 | 0.891 |
| 1 | 0.994 | 0.932 | 0.940 | 0.01 | 0.865 |
| 3 | 0.997 | 0.951 | 0.938 | 0.06 | 0.906 |
| 5 | 0.997 | 0.922 | 0.919 | 0.19 | 0.859 |
| 10 | 0.990 | 0.854 | 0.857 | 0.52 | 0.750 |

Paired against +0: +1 final −0.013 [−0.029, +0.003]; +3 +0.005 [−0.010, +0.021]; +5 −0.023 [−0.047, −0.003];
+10 −0.091 [−0.125, −0.060], agreement −0.005 [−0.011, +0.001].

Correction of the reduction (true atoms alone, set selection − old greedy reduction): all kept
+0.141 [+0.104, +0.177] (0.802 → 0.943); kept recall +0.117 [+0.089, +0.145]; final ≥ 0.99 +0.156
[+0.117, +0.195] (0.789 → 0.945); agreement +0.052 [+0.034, +0.073]. On average 1.44 true atoms per task (2.48 at
d4–6) have only conditional utility: this is what the old reduction removed.

With DISCOVERED primitives, on the other hand, the selection does not keep what is in the pool: for C, all
present in the pool 0.271 but all kept 0.195, and 4.4 of the 5.2 atoms kept are false.

## 6. Completeness, missing primitive, loop

Control (true atoms, then one true atom removed):

| Condition | n | Flagged incomplete | Agreement | Final ≥ 0.99 |
|---|---|---|---|---|
| complete set | 384 | 0.096 | 0.995 | 0.948 |
| one atom removed, before loop | 320 | 0.944 | 0.826 | 0.022 |
| one atom removed, after loop | 320 | — | 0.915 | 0.169 |

The removed atom is recovered by the residual in 45.3% of tasks and kept in 39.1% (2.7 iterations). Per
depth: 0.70 (d2), 0.56 (d3), 0.47 (d4), 0.31 (d5), 0.22 (d6). Per type: disc 0.60; band 0.58; ring 0.52;
half-plane 0.50; sector 0.40; **anti-disc 0.00** (42 tasks). In this control, the completeness signal separates well
complete (9.6% flagged) and incomplete (94.4%).

In real conditions (chain C): the loop runs in 90.6% of tasks (2.4 iterations); C − `C_noloop`: agreement
+0.002 [−0.001, +0.006], candidate ≥ 0.97 +0.039 [+0.016, +0.065], all present +0.013 [+0.003, +0.026], recall
+0.031 [+0.022, +0.041], time +4.6 s. The `COMPLETE` state is declared for 346 tasks out of 384, whereas all the
primitives are present there in only 29.8% of cases: the state is not a reliable indicator of completeness with
discovered primitives.

Does set recall pay off downstream? Yes, strongly, but it is rare. Final ≥ 0.99 when all the true
primitives are kept (strict level) versus otherwise: A 0.528 (n = 36) versus 0.029; B 0.778 (n = 18) versus 0.060;
C 0.560 (n = 25) versus 0.056. Agreement: 0.985–0.989 versus 0.885–0.897.

Out of class (exploratory, 32 tasks per case): primitive absent → 32 `COMPLETE` out of 32; relation absent → 29
`COMPLETE`, 3 `INCOMPLETE`; composition absent → 27 / 5. `POSSIBLY_OUT_OF_CLASS` is never declared. Out-of-class
detection does not work.

## 7. Preregistered hypotheses

| # | Result | Status |
|---|---|---|
| LP1 | C − B +0.047 [+0.026, +0.068]; C − A +0.029 [−0.003, +0.060] | **not supported** (both CIs were required) |
| LP2 | candidate: C − B +0.073, CI > 0; C − A +0.018 [−0.010, +0.050]; agreement: both CIs > 0 | **not supported** |
| LP3 | individual recall C − B +0.119 [+0.100, +0.139] | supported |
| LP4 | +10 false: final ≥ 0.99 −0.091 [−0.125, −0.060]; agreement −0.005 | **not supported** at +10; held up to +5 (−0.023) |
| LP5 | all kept +0.141 [+0.104, +0.177] | supported |
| LP6 | B5 − B: set +0.021 [+0.008, +0.037]; agreement +0.002, CI including 0 | **not supported** (partial) |
| LP7 | recovered 45.3% (< 50%); after > before (0.169 versus 0.022) | **not supported** (partial) |
| LP8 | C − `C_noloop` agreement +0.002 [−0.001, +0.006] | **not supported** |
| LP9 | phi between −0.02 and +0.06; set recall ≈ `recall^n` | **not supported** |
| LP10 | 9.0 s (> 5 s); lower than `A_full` 26.1 s and `B_full` 20.6 s | **not supported** |
| LP11 | beats A, B, `A_full`; C − `B_full` +0.001 [−0.005, +0.006] | **not supported** |

Declared expectations: LP2 and LP11 unlikely — confirmed; LP8 null — confirmed; LP5 likely — confirmed; LP7
likely — contradicted on the 50% threshold.

## 8. Answers to the ten questions

**1. Why does set recall collapse with depth?** Through two multiplicative effects. The per-atom
recall decreases with depth (C: 0.98 at one atom, 0.45 at four, 0.32 at six), and the set requires all the
atoms at once, which raises this recall to the power n. The independent model reproduces the observed values
(C, three atoms: 0.125 observed, 0.121 predicted). The per-atom decrease is linked to visibility (0.06 under 20% of
visible boundary, 0.64 above 80%), but visibility does not explain everything: an almost entirely
visible atom is still missed one time in three.

**2. Are the primitive errors independent?** Roughly. Phi between −0.02 and +0.06 depending on the chain;
missing one atom does not make the other more likely to be missed. Only visible dependence: the pairs of the same
type (phi +0.15 to +0.18 for A and C, null for B), without CI. There is therefore no simple common cause to correct:
the per-atom recall itself has to be raised.

**3. Does a recall-first strategy help?** On individual recall, yes (+0.119 versus B, +0.133 versus A).
On set recall, yes versus B (+0.047), not demonstrated versus A. Beyond three atoms, no: all
present 0.031 at four atoms, 0 at five and six. And the gain of the pool is lost at selection (0.271 in the pool,
0.195 kept).

**4. Do false positives remain tolerable?** With the true atoms present, yes up to +5 (final ≥ 0.99
−0.023), no at +10 (−0.091; −0.141 at d4–6), the agreement remaining at 0.990. With discovered primitives, the problem
is of another order: 4.4 of the 5.2 atoms kept are false, because the true ones are missing or imprecise.

**5. Does top-k parameters help?** Little. `k = 5` versus `k = 3`: set recall +0.021 [+0.008, +0.037], agreement
+0.002 not significant. In C, `k = 3` does as well as `k = 5`.

**6. Can the composition guide the search for missing primitives?** In control, partially: when a
single true atom is missing and the others are exact, the residual recovers it in 45% of tasks (70% at two
atoms, 22% at six) and the final ≥ 0.99 goes from 0.022 to 0.169. The anti-disc is never recovered (0 out of 42) — cause
not established.

**7. Does the iterative loop recover missing atoms?** In real conditions, marginally: recall
+0.031, all present +0.013, candidate structure +0.039, but final agreement +0.002 not significant, for a
doubled time (4.5 s → 9.0 s). The residual is built on a composition that is already wrong, with several atoms missing.

**8. Does set recall really improve the final structure?** When it is reached, yes: final ≥ 0.99 of
0.53–0.78 versus 0.03–0.06 depending on whether the set is complete at the strict level or not. But the gain in set recall
obtained here (+0.03 to +0.05, concentrated on one to three atoms) translates only into +0.007 of agreement and into no gain in
final ≥ 0.99.

**9. Does the cost remain compact?** 9.0 s per task and 45 hypotheses: above the preregistered threshold of 5 s,
but under `A_full` (26.1 s) and `B_full` (20.6 s). Half of the cost is the loop, which brings nothing measurable
on agreement.

**10. Is the system at last ready for a confirmatory campaign?** No. LP1, LP2, LP10 and LP11 are all not
supported. The 16-seed campaign is not proposed; the sealed set has not been touched.

## 9. Limits

- Dev partition, a single task seed per key, 64 tasks per depth; the per-depth CIs are wide.
- LP1 fails on a CI that grazes zero (C − A +0.029 [−0.003, +0.060]); the rule written in advance is applied
  as is, without reinterpretation.
- "Present" rests on a functional agreement at 0.95 measured on the domain; at the strict level (0.99) the shares
  are two to three times lower (C 0.091, B 0.057, A 0.120 — A ahead of C).
- Phi without confidence interval; correlation measured by pairs, no higher-order analysis.
- Sensitivity and false positives start from the TRUE atoms: they bound the downstream, they do not describe the real chain.
- Out of class: 32 tasks per case, exploratory, negative result.
- `B_full` remains equal to C on agreement and better on final ≥ 0.99 and candidate structure: the strong downstream common to
  A, B, C is not the best known downstream for `flat_scene_k3`.
- The zero of the anti-disc in the residual search is not explained; it may come from the measurement
  (complement recovered under another type) as much as from the method. `POST-HOC / EXPLORATORY`, not verified.

## 10. New blocker

The blocker is not the correlation of the errors, nor the false positives, nor the reduction (corrected), nor the composition
(0.95 of final ≥ 0.99 with exact atoms). It is the **per-atom recall and precision in a composed scene**: 0.32 to 0.45
at depths 4–6, with a parameter error (0.067) located in the zone where the sensitivity makes the
recovery at 0.99 fall under 10%. Widening the pool is not enough: a per-atom recall of about 0.9 would be needed to
obtain 50% of complete sets at six atoms. Second point: set selection loses true atoms
present in the pool when they are imprecise.

## 11. Verdict

`LATENT PRIMITIVE RECOVERY STILL BLOCKING`

Preregistered rule: LP1 not supported. Individual recall progresses clearly (LP3) and set recall
progresses against `flat_scene_k3`, but not in a demonstrated way against the frozen pipeline, and it remains zero beyond
four atoms. No 16-seed campaign proposed, sealed set intact, nothing wired in by default; `flat_scene_k3`
remains *confirmed dev baseline candidate*.

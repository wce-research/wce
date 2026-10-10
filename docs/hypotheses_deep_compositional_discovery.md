# Preregistration — WCE Deep Compositional Structure Discovery & Model-Class Boundary

Written on 2026-10-04, BEFORE any quantitative experiment of this milestone. Branch `jalon-deep-compositional-discovery`,
started from the frozen tag `jalon-robust-reconstruction`. The sealed set of the previous milestone is consumed: it is not used for
any tuning, diagnostic or hyperparameter choice.

Any analysis added after results have been observed is marked `POST-HOC / EXPLORATORY` and stays separate from the
confirmatory analyses below.

## 0. Audit of the existing code (code reading, BEFORE any experiment — to be verified by instrumentation)

Candidate causes of the collapse beyond depth 1 in the previous milestone. These are hypotheses from
code reading; section 16 of the specification (per-depth profiling) must decide between them.

1. Generation: the search is a greedy left chain (3 parents × 3 combinators × 8 atoms). Balanced
   trees are never generated; a discovered sub-structure is not reused.
2. Triggering: deepening only happens if the residual exceeds the expected noise (binomial test); with 8–12%
   of known noise, it may never trigger beyond depth 2.
3. Fitting: a new atom receives a single random initialisation; the joint gradient fit probably
   stays in a local optimum.
4. Ranking: BIC-type penalty on 120 points, which favours short structures.
5. Pruning: the set of structures is truncated after each level.
6. Identifiability: the passive observations covered only x ≤ 0.5 whereas recovery was judged on
   [−3, 3]²; 16 queries are not enough to identify a deep structure outside the observed zone.
7. The noise of the world was PROVIDED to the system (true rate); it was not estimated.

## 1. Evaluation set (`core/deep_world.py`)

- Compositional depth d = number of atoms ∈ {1, …, 6}; each atom is effective (removing it changes ≥ 4%
  of the observed region). Trees of varied shapes (chains and balanced trees), combinators and / or / xor.
- Regimes: `rich` (600 observations, noise 2%), `standard` (300, 5%), `hard` (160, 10%). Noises: i.i.d., concentrated
  at the boundaries, grouped outliers. The noise is NOT provided to the new system (it is provided to the frozen baseline, which
  gives it an advantage).
- Passive observations on x ≤ 1.2; extrapolation judged on x > 1.2; active measurements allowed everywhere.
- Partitions with disjoint seeds: dev (base 1,000,000), validation (2,000,000), sealed set (3,000,000), confirmatory
  test (4,000,000).
- Unseen compositions: one global signature in four (d ≥ 3) is held out; it never appears in dev or in
  validation. The 2-atom sub-structures remain seen.
- Out of class (6 categories): missing primitive, composition beyond budget, missing relation, incompatible dynamics,
  distribution shift, partially representable. Hard but valid (3): extreme noise (25%), deep
  (5–6), little data (80).
- Sealed set: 72 tasks (36 in class, 12 hard-valid, 24 out of class). Hash frozen before any
  tuning: `d57e9d761d165ef9d86ec7ee5c9c5c7909c4b1622425c2d2379cbdddaf1a6ca2`. Single evaluation at the end of the milestone.

## 2. Methods compared

- `baseline`: system of the previous milestone, FROZEN (`RobustStructuralReconstruction`, EFA policy… unchanged).
- `bottom_up`: simple sub-structures fitted then composed (including with one another), initialisation from the residual.
- `top_down`: recursive decomposition of the target into (atom, combinator, residual target with restricted support).
- `bidirectional`: residual targets (top-down) matched against the library of sub-structures (bottom-up).
- `best_first`: search ordered by an explicit score (evidence, MDL, functional novelty, reuse,
  search cost).
- PRIMARY method: a single strategy, chosen on VALIDATION (mean functional recovery d = 2–4, standard
  regime), frozen before the confirmatory test. If a variant turns out to be redundant or infeasible in dev, it is
  dropped and the decision documented.
- Ablations of the primary method: without fragment library, without anti-unification, without equivalence classes,
  without NONE_OF_THE_ABOVE, without hierarchy (flat candidates, no reuse), without identifiability query.
- Measurement policies: equivalence-class reduction (new), structural IG, disagreement, random, historical
  EFA, oracle.

### Compute budget (architecture versus compute)

The unit is the "fit-equivalent": one initialisation × 100 gradient steps of one candidate. Two budgets:

- `matched`: cap equal to the mean consumption of the baseline on the same tasks (measured in dev, frozen
  afterwards). This is the budget of the CONFIRMATORY comparisons.
- `extended`: 3 × the matched budget. Reported separately; a gain that appears only at the extended budget is attributed to
  compute, not to the architecture.

Wall time and memory are reported in addition, per depth.

## 3. Metrics (never merged into a single one)

- `ExactStructure`: canonical skeleton identical to the true one.
- `EquivalentStructure`: identical up to algebraic rewritings (commutativity, associativity, complement).
- `FunctionallyEquivalent`: agreement ≥ 0.99 on 4,000 probes of [−3, 3]².
- `PredictivelyAdequate`: agreement ≥ 0.95 in the observed region AND in extrapolation.
- Extrapolation: agreement on x > 1.2. Preserved invariants: share of the invariants of the true program that are recovered.
- Per depth: all metrics are reported separately for d = 1, 2, 3, 4, 5+.
- Three-outcome decision: `KNOWN`, `UNKNOWN` (insufficient information), `NONE_OF_THE_ABOVE` (observations
  incompatible with the available hypotheses).

Statistics: 16 test seeds, mean per seed, 95% CI (Student's t), paired differences per seed.

## 4. Confirmatory hypotheses

"Supported" requires the threshold AND a 95% CI excluding 0 in the right direction, unless stated otherwise. Between "supported" and
"refuted": inconclusive.

| # | Hypothesis | Measure | Comparison | Supported if | Refuted if |
|---|---|---|---|---|---|
| H1 | Deep recovery | `FunctionallyEquivalent`, mean d = 2–4, standard regime, matched budget | primary − baseline | ≥ +0.15 | ≤ +0.03 |
| H2 | Drop with depth | ratio recovery(d = 2–4) / recovery(d = 1) | primary − baseline | ≥ +0.20 | ≤ 0 |
| H3 | Search efficiency | share of d = 2–4 tasks where the true skeleton (up to equivalence) is GENERATED, matched budget | primary − baseline | ≥ +0.20 | ≤ 0 |
| H4 | False structural error | among the `ExactStructure` errors of the primary (d ≥ 2), share where the retained structure is in the same observational equivalence class as the true one | absolute threshold | ≥ 0.30 | < 0.10 |
| H5 | Identifiability query | mean final size of the equivalence class, 16 queries | class reduction versus structural IG | reduced ≥ 25% without functional loss > 0.02 | reduction ≤ 0 |
| H6 | Out-of-class detection | false acceptance (out of class declared `KNOWN`) and false rejection (valid declared `NONE_OF_THE_ABOVE`) | absolute thresholds; baseline reported | FA ≤ 0.30 AND FR ≤ 0.15 | FA > 0.50 OR FR > 0.30 |
| H7 | Abstention calibration | ECE of P(out of class) on the test, calibrated on validation; purity: share of `NONE_OF_THE_ABOVE` among the out-of-class non-`KNOWN`, share of `UNKNOWN` among the valid non-`KNOWN` | absolute thresholds | ECE ≤ 0.10 AND both purities ≥ 0.70 | ECE > 0.20 OR one purity < 0.50 |
| H8 | Complexity | mean absolute size excess (retained atoms − true ones), d = 1–4; violations of the compact budget (64 parameters) | primary versus baseline | excess ≤ baseline AND 0 violation | excess > baseline + 0.5 OR ≥ 1 violation |
| H9 | Extrapolation | agreement on x > 1.2, d = 1–4 | primary − baseline | ≥ −0.02 (non-inferiority: lower bound of the CI ≥ −0.02) | mean < −0.05 |
| H10 | Compute cost | (a) speed-up of the fit per fit-equivalent; (b) maximal numerical gap with the reference fit | new engine versus frozen `fit_robust` | (a) ≥ 3× AND (b) ≤ 1e-6 on the final parameters | (a) < 1.5× OR (b) > 1e-3 |
| H11 | Fragment reuse | fit-equivalents until the first adequate structure, "unseen compositions" bench | with library (learned in dev) versus without | reduced ≥ 20% without recovery loss > 0.02 | no reduction OR loss > 0.05 |
| H12 | Unseen compositions | `FunctionallyEquivalent`, d = 3–4, standard regime | unseen − seen (primary) | gap ≥ −0.10 (lower bound of the CI ≥ −0.10) | mean < −0.20 |
| H13 | Anti-unification | dedicated synthetic bench: correct generalisation and over-generalisation | absolute thresholds | correct ≥ 0.80 AND over-generalisation ≤ 0.10 | correct < 0.60 OR over-generalisation > 0.25 |

Compactness safeguard for H1, H3 and H11: a gain obtained with more than 2× the candidates generated by the baseline is
not counted as an architectural win (it is reported, with this mention).

## 5. Secondary confirmatory analyses (reported without threshold)

- Recovery per depth (1, 2, 3, 4, 5+) and per regime, all methods, matched and extended budgets.
- Fate of the true structure, per depth: never generated / generated but badly fitted / well fitted but badly
  ranked / equivalent to others / removed by pruning — for the baseline and the primary.
- Per depth: candidates generated, candidates fitted, families, equivalence classes, pruning rate, search
  time, fitting time, memory, rank of the true structure.
- Out-of-class signals taken SEPARATELY (AUROC, AUPRC), then transparent combination (logistic regression fitted on
  validation only).
- Per out-of-class category and per hard-valid category: KNOWN / UNKNOWN / NONE_OF_THE_ABOVE matrix.
- Ablations of the primary.
- Mean size of the equivalence classes per depth, before and after measurements.

## 6. Exploratory analyses (announced, not confirmatory)

- Missing-primitive diagnostic on the out-of-class worlds.
- Depth 6 (stress test).
- Fragment library: frequency, compression gain, search gain, transfer.
- Regimes where d = 3–5 becomes identifiable.

## 7. Tuning and leakage rules

- All tuning is done in dev; selection of the primary and calibration on validation; confirmatory test launched once
  the configuration is frozen (configuration hash recorded in the manifest).
- The sealed set is neither read, nor plotted, nor summarised before the single final evaluation.
- The inference code never imports `core/deep_world.py`; the diagnostics that use the truth are in a
  separate evaluation module and return nothing to the system.
- The fragment library is learned on dev only.

## 8. Stopping and reduction criteria (decided in advance)

- Before any 16-seed campaign: micro-benchmark of the new system, cost per depth known, candidate
  generation instrumented.
- If the estimated cost of the confirmatory campaign exceeds 10 h on 8 vCPU, reduce in this order: (1) depth 6
  limited to the standard regime; (2) ablations on 8 seeds; (3) `rich` regime on 8 seeds. Hypotheses H1–H3
  always keep 16 seeds.
- A strategy whose cost per task exceeds 5× that of the baseline in dev is dropped (decision documented).
- If the primary does not exceed the baseline in validation, the confirmatory test is STILL run and reported.

## 9. Failure conditions of the milestone (explicit)

The milestone fails on its architectural criterion if H1 and H2 are refuted, or if the gain appears only at the extended budget
or with more than 2× the candidates of the baseline. In that case the report must say where the true structure is lost
(section 5) and why.

## 10. Design addendum (2026-10-04, BEFORE any quantitative experiment)

Protocol clarifications fixed before the first run, so that they are not chosen after the fact.

- Decision cost (H7, "abstention cost"): correct decision 0; `UNKNOWN` 1; `NONE_OF_THE_ABOVE` on a valid
  world 3; `KNOWN` on an out-of-class world 5; `KNOWN` with a functionally wrong structure on a valid
  world 2. The decision thresholds are chosen on validation to minimise this cost, then frozen.
- Correct decision: out of class → `NONE_OF_THE_ABOVE`; valid → `KNOWN` with a `FunctionallyEquivalent` structure,
  or `UNKNOWN` if the retained structure is not.
- Measurement budgets of the new system, per task: 16 identification queries (standard sensor); 24 boundary
  validation measurements with the precise sensor (12 in the observed zone, 12 outside the zone); 24 repetitions of passive points
  (model-free noise estimation). The baseline keeps its budget from the previous milestone (16 units) and receives the true
  noise. The validation measurements are NOT used for fitting the structures.
- Internal holdout: 20% of the passive observations are removed before the search and are used for the predictive check.
- Observational equivalence class: fitted structures whose predictions agree on ≥ 99% of the measured
  points (single linkage), within an evidence window of log 20 around the best one.
- The noise is not provided to the new system: each hypothesis profiles its own flip rate.
- Out-of-class signals evaluated separately: predictive check on holdout, spatial clustering of the residual, gap to a
  non-parametric model (k nearest neighbours), description length per observation, parameter
  instability, improvement still in progress at the maximal depth, temporal drift of the residual, validation with the
  precise sensor in the zone and outside the zone, excess noise relative to the repetitions, persistent structural disagreement.

## 11. Freeze (2026-10-04)

This preregistration and the manifest of the sealed set (`docs/sealed_deep_manifest.json`, hash
`d57e9d761d165ef9d86ec7ee5c9c5c7909c4b1622425c2d2379cbdddaf1a6ca2`) are committed before the milestone continues. From this commit on, the sealed set is FORBIDDEN to any
adaptive inspection until the single final evaluation. At the time of the freeze, the only things that had run were: local smoke
trials on a few dev tasks and an AWS dev micro-benchmark (2 tasks per depth); none touched the
validation, the test or the sealed set.

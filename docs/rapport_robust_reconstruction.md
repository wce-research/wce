# WCE Robust Structural Reconstruction & Measurement-Aware Inference — final report (frozen milestone)

Campaigns run on AWS on 2026-10-03 and 2026-10-04, 16 seeds, sealed set evaluated only once.
Hypotheses preregistered before any run: `docs/hypotheses_robust_reconstruction.md`.
Numerical summary: `results/robust_summary.json`. Raw artefacts: `results/aws/`. Timings: `results/aws_timing.json`.

Outcome: 5 hypotheses supported, 3 refuted, 2 inconclusive.

## 1. Architecture
- Measurement model: each observation carries its sensor, its flip rate, its resolution, its temporal weight and its source.
- Inference: naive, flip, contamination and soft-margin likelihoods; APPROXIMATE evidence and description length (MDL); set of compatible structures with a possible "UNKNOWN" answer.
- Active measurement: planner separated from the search, six criteria (EFA, structural IG, functional IG, disagreement, uncertainty, random), choice of sensor, repetition option.
- Decisions: deepening by Bayes factor, attribution of discrepancies into five classes, posterior predictive check, calibration of P(rule wrong).
- Evaluation world separated from the inference code, which never imports it.
- The physical analogies (Fourier, Boltzmann, Heisenberg, etc.) are design inspirations, nothing more.

## 2. Files
- `core/robust_world.py`, `core/robust_reconstruction.py`, `core/robust_experiments.py`, `core/robust_aggregate.py`
- `wce_aws/orchestrate.py`, `wce_aws/sm_entry.py`, `wce_aws/bench.py`
- `tests/test_robust.py`, `docs/hypotheses_robust_reconstruction.md`, five demo scripts in `scripts/`

## 3. Tests added
266 tests (target: at least 200): likelihoods, posterior, MDL, expected gains, planner, calibration, leaks, AWS orchestration without network calls, checkpoints and resume, cache.

## 4. `make check`
Syntax OK.

## 5. `make test`
3,828 tests, 7 skipped, no regression.

## 6. AWS workflow
- EC2 + SSM backend, always explicit (`WCE_BACKEND=ec2`). SageMaker unusable (quotas at 0), never tested in real execution.
- No SSH and no key, instances tagged `Project=WCE`, automatic termination, shutdown safeguard.
- Units split into deterministic sub-units; one checkpoint per sub-unit, published to S3 every 45 s.
- Targets: `aws-login`, `aws-whoami`, `aws-check-guardrails`, `aws-ec2-diagnose`, `aws-progress`, `aws-resume`, `aws-stop`, `aws-inspect`, `aws-timing`, `aws-profile`, `aws-bench`, `aws-fetch-results`, `aws-multiseed`, `aws-run-rest`, `aws-sealed-eval`, `robust-aggregate`.
- Resume validated end to end (abrupt interruption, 55 checkpoints reused, 45 recomputed).
- Deviation from the Makefile rule: a single direct `aws s3 ls` call, read-only.

## 7. Identity
`WCE-Research-Runner` (profile `wce-runner`, `eu-north-1`). IAM reads denied as expected. No permission modified by the runner.

## 8. Instances
- Scientific reference: `c6i.xlarge` (4 vCPU), 2 instances in parallel (quota of 8 vCPU).
- `t3.large`: testing and debugging only (1.7 times slower, 1.5 times more expensive per workload).
- One process per vCPU, one Torch thread. Torch threads slow things down (×0.77 to ×0.92); the vCPUs are hyperthreads (59 to 61 % of the speed of a single process).

## 9. Run times

| Campaign | Sub-units | Duration per instance |
|---|---|---|
| Main bench | 1,600 | 1 h 48 and 2 h 12 |
| Noise, calibration, auxiliaries | 1,676 | 1 h 29 and 1 h 27 |
| Sealed set | 120 | 7 and 8 min |

Mean CPU 97 to 100 %, peak memory 351 MB per process. Hot spot: gradient fitting (`fit_robust`, 78 % of the time).

## 10. Cost
About $1.50 for 8.3 instance-hours over 17 jobs, infrastructure trials included. Estimate based on on-demand prices quoted from memory, to be checked against the invoice.

## 11. Multi-seed (main bench, noise 8 and 12 %, 95 % CI)

| Method | Recovery | Exact structure | Extrapolation | Queries |
|---|---|---|---|---|
| Old system | 0.117 [0.079, 0.155] | 0.141 | 0.778 | 15.6 |
| Passive | 0.047 [0.006, 0.088] | 0.055 | 0.619 | 0 |
| Robust fit alone | 0.219 [0.152, 0.285] | 0.242 | 0.806 | 0 |
| Robust posterior | 0.211 [0.143, 0.279] | 0.242 | 0.797 | 0 |
| Active, structural IG | 0.328 [0.260, 0.396] | 0.250 | 0.847 | 16.0 |
| Active, EFA | 0.266 [0.207, 0.325] | 0.266 | 0.863 | 11.4 |
| Full system | 0.234 [0.175, 0.293] | 0.273 | 0.834 | 10.7 |
| Random queries | 0.266 [0.202, 0.329] | 0.242 | 0.838 | 16.0 |
| Disagreement queries | 0.328 [0.260, 0.396] | 0.250 | 0.845 | 16.0 |
| Oracle (upper bound) | 0.352 [0.291, 0.412] | 0.266 | 0.873 | 16.0 |

Recovery by world depth:

| Method | Depth 1 | Depth 2 | Depth 3 | Depth 4 |
|---|---|---|---|---|
| Old system | 0.41 | 0.00 | 0.00 | 0.06 |
| Robust posterior | 0.56 | 0.13 | 0.03 | 0.13 |
| Active, structural IG | 0.91 | 0.13 | 0.09 | 0.19 |
| Active, EFA | 0.72 | 0.13 | 0.06 | 0.16 |
| Full system | 0.66 | 0.13 | 0.06 | 0.09 |
| Oracle | 0.91 | 0.19 | 0.09 | 0.22 |

Reproducibility: seeds 0 and 9 replayed in a separate run, identical results (apart from duration fields).

## 12. Noise sweep (recovery)

| Noise | Full system | Robust posterior | Old system | Passive |
|---|---|---|---|---|
| 0 % | 0.479 | 0.333 | 0.292 | 0.188 |
| 2 % | 0.438 | 0.333 | 0.229 | 0.062 |
| 5 % | 0.312 | 0.292 | 0.188 | 0.104 |
| 8 % | 0.250 | 0.208 | 0.208 | 0.104 |
| 12 % | 0.271 | 0.208 | 0.062 | 0.042 |
| 15 % | 0.312 | 0.167 | 0.125 | 0.062 |

By noise type, full system: 0.385 (boundary), 0.354 (i.i.d.), 0.292 (clustered outliers); old system: 0.281, 0.135, 0.135.

## 13. Sealed set (a single evaluation, 24 tasks of which 20 representable)

| Method | Recovery | Extrapolation |
|---|---|---|
| Full system | 0.20 [0.01, 0.39] | 0.819 |
| Robust posterior | 0.15 [−0.02, 0.32] | 0.743 |
| Random queries | 0.15 [−0.02, 0.32] | 0.792 |
| Old system | 0.10 [−0.04, 0.24] | 0.687 |
| Passive | 0.00 | 0.713 |

Ranking consistent with the 16 seeds, but overlapping intervals: the sealed set does not allow a conclusion on its own.

## 14. Calibration of P(rule wrong)
215 training cases (seeds 700000+), 348 test cases; calibrators fitted on the training set only.

| Calibrator | ECE | Brier |
|---|---|---|
| Raw | 0.181 | 0.148 |
| Isotonic | 0.071 | 0.080 |
| Beta | 0.075 | 0.077 |
| Platt | 0.108 | 0.081 |

## 15. Recall
- Family promotion: recall 0.688 by active validation, 0.438 by clustering.
- Recovery on a new instance: 0.208 with the family prior versus 0.313 by full search.

## 16. Extrapolation
Full system 0.834 versus 0.778 for the old system (+0.056 [0.013, 0.099]). EFA alone: 0.863, best non-oracle method.

## 17. False promotion and false rejection
- Families: 39 promoted by active validation, precision 0.846 (6 false); 23 by clustering, precision 0.913 (2 false).
- Non-representable worlds: the full system declares "identified" in 28 cases out of 32; the identifiability test classifies 13 non-representable worlds out of 16 as identifiable.
- "UNKNOWN" answer: 1.9 % (full system), 16 % (robust posterior), 21 % (oracle).

## 18. Query efficiency
- EFA: 11.4 queries on average; full system: 10.7; other active policies: 16.
- EFA versus random: recovery 0.000 [−0.054, 0.054], extrapolation +0.025 [0.003, 0.047].
- Structural IG and disagreement: best non-oracle policies (0.328).
- Oracle versus full system: +0.117 [0.072, 0.163].

## 19. Complexity
- Mean size of the retained programs: 1.6 to 2.2 (robust methods), 5.15 (old system), 4.82 (passive).
- "Simple + 8 % noise": size excess 0.94 (naive threshold) versus 0.00 (robust Bayesian rule).
- "Complex + 0 % noise": excess 0.63 versus 0.00, recovery 0.125 in both cases.

## 20. Capacity
- No violation of the compact budget (3.6 parameters and 17.5 error points on average).
- Residual compression brings no gain: 0.911 without residual, 0.909 with gated raw residual, 0.831 with prototypes.

## 21. Supported hypotheses
- H2: depth excess brought down from 0.94 to 0.00.
- H5: ECE reduced from 0.181 to 0.071.
- H7: recovery +0.117 [0.055, 0.179] relative to the old system.
- H8: extrapolation preserved, +0.056.
- H9: 0 violation of the compact budget.

## 22. Refuted hypotheses
- H4: adaptive repetitions are never chosen (0 out of 64 runs); zero difference.
- H6: active validation less precise than clustering (0.846 versus 0.913).
- H10: gain predicted by the EFA uncorrelated with the realised gain (ρ = −0.008 [−0.093, 0.078]).

## 23. Inconclusive hypotheses
- H1: robust versus naive likelihood, +0.125 [−0.205, 0.455].
- H3: inconclusive in the preregistered sense, but unfavourable: EFA significantly worse than structural IG in recovery (−0.063 [−0.105, −0.020]).

## 24. Bugs fixed
- Noise-sweep seed dependent on `hash()` of a string (not reproducible): fixed before any run.
- Predictive check of the families: points and labels drawn separately: fixed before any run.
- World depth overwritten by the program size in the rows of the main bench: restored at aggregation, without recomputation.
- EC2 script: installation failure reported as "Completed": fixed, failures are now propagated.
- First pilot without checkpoint or progress, stopped after 52 minutes with no recoverable result.
- Unresolved: code fingerprint differs between the campaign and the reproducibility run, cause not identified; identical results.

## 25. Limits
- Recovery low in absolute terms: 0.35 for the oracle, almost nothing beyond depth 1.
- The system does not recognise that a world is outside its hypothesis class.
- Repetitions and multi-sensor choice bring no gain in this world; the full system does worse than structural IG alone.
- Approximate evidence, not exact Bayesian evidence.
- Synthetic world in two dimensions.
- Sealed set too small to conclude on its own.
- SageMaker never tested in real execution.

## 26. Next bottleneck
Structure search beyond depth 1: even with an oracle measurement policy, the right structure is almost never found. The measurement policy is not the limiting factor; it is the generation and fitting of deep candidates, which is also the hot spot in compute time.

## Milestone freeze
The results of this milestone are no longer modified after the freeze commit. The sealed set (`SEALED_SHA256` in `core/robust_world.py`) has been consumed and must no longer be used for any tuning.

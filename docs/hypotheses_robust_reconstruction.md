# Preregistration — "Robust Structural Reconstruction & Measurement-Aware Inference" (frozen BEFORE the main runs)

Freeze date: 2026-10-03. NO modification after the sealed benchmark.
Compute: AWS SageMaker Processing (eu-north-1), identity WCE-Research-Runner, execution role WCE-SageMaker-ExecutionRole.
Seeds: **16** (recorded before the run; 32 only if the measured cost of the 16 stays < 2 h of ml.c5.4xlarge instance — decision recorded in the manifest).
Measurement budget per task: **16 cost units** (standard sensor = 1, precise = 3, cheap = 0.3). Passive sample: 120 observations.
Exact functional recovery = agreement ≥ 0.97 with the true function on [−3,3]². STRUCTURE recovery = canonical skeleton identical to the true one.
Compact budget (H9): ≤ 64 parameters + ≤ 32 raw points stored per skill.
"Previous milestone" reference (H7): full ActiveStructuralDiscovery system re-run on the NEW bench (same seeds).

| H | Metric | Baseline | Success | Failure |
|---|---|---|---|---|
| H1 robust likelihood | functional recovery, noise 8–15 % | naive fit (same candidates, same queries = none) | +0.10 and CI95 > 0 | diff ≤ 0 |
| H2 structure / noise separation | mean AST depth of the retained structures on noisy targets of depth 1–2; "too deep" share | naive fit + deepening by threshold | excess depth reduced ≥ 30 % | not reduced |
| H3 EFA | functional recovery (and extrapolation) at equal budget | structural IG (same inference) | +0.05 and CI95 > 0 on at least one of the two metrics without a loss > 0.02 on the other | ≤ structural IG on both |
| H4 adaptive repetitions | recovery at noise 12–15 % | fixed policy (no repetition, same budget) | +0.05 and CI95 > 0 | ≤ 0 |
| H5 calibration of P(rule wrong) | ECE (TRAIN+VAL → test) | uncalibrated | ECE reduced ≥ 50 % | ECE not reduced |
| H6 active validation of families | promotion precision (promoted families that speed up / succeed on new instances) | promotion by clustering (previous milestone) | precision ≥ baseline + 0.2 | ≤ baseline |
| H7 recovery under 8–12 % | functional recovery | old system on the same bench | +0.10 and CI95 > 0 | ≤ 0 |
| H8 extrapolation preserved | out-of-region extrapolation | old system | ≥ old − 0.02 | < old − 0.05 |
| H9 compact budget | violations | — | 0 violation | ≥ 1 |
| H10 predicted vs actual gain | Spearman correlation between predicted EFA and realised functional gain | 0 | ρ ≥ 0.3 and CI95 > 0 | ρ ≤ 0 |

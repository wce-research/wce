# Preregistration of the hypotheses — milestone "generative certificates" (frozen before any execution)

Freeze date: 2026-10-01. These criteria will NOT be modified after observation.
Statistical unit: seed (16 seeds minimum); 95% CI by bootstrap/t over the seeds; paired comparisons.

| H | Metric | Baseline | Success | Failure |
|---|---|---|---|---|
| H1 generative certificates protect without storing the data | final retention (continuous stream of 25 regimes) | raw replay | retention(gen_cert) ≥ retention(raw replay) − 0.02 AND 0 bytes of raw data | retention(gen_cert) < retention(raw replay) − 0.05 |
| H2 the second order catches dangerous updates missed by the first order | on first-order projected updates (J·Δθ≈0): correct rejection rate (true violation) at second order vs first order; retention after update | first-order projection | recall of violations (2nd order) − recall (1st order) ≥ 0.2 AND retention (2nd) > retention (1st) on ≥ 12/16 seeds | recall difference < 0.05 |
| H3 compositional plasticity stays competitive in the noisy continual setting | final mean accuracy | raw replay AND structural growth | full ≥ max(raw replay, structural growth) − 0.03 | full < raw replay − 0.05 |
| H4 parameter growth ≪ systematic growth when the structure repeats | parameter growth at 100 regimes | systematic structural growth | ratio full / systematic ≤ 0.6 | ratio ≥ 0.9 |
| H5 the acquisition cost decreases with reusable structure | slope (training steps per new skill) over the life index, 100 regimes | zero slope | slope < 0 with 95% CI entirely < 0 | 95% CI contains 0 or slope > 0 |
| H6 capacity pressure is resolved by merging/distillation/pruning without catastrophic loss | mean accuracy loss due to the capacity operations; no. of operations triggered | no operation (unlimited budget) | ≥ 1 operation of each type triggered in a real stream AND loss ≤ 0.03 | loss > 0.10 or no operation triggered |
| H7 a corrupted skill regenerates without a snapshot | recovery = regenerated accuracy / accuracy before corruption | regeneration of the previous milestone (≈ 0.89 relative recovery) | median recovery ≥ 0.95 | median recovery < 0.90 |
| H8 active discovery of transformation laws | correct identification rate of the hidden law; no. of experiments | random experiments | identification ≥ 0.8 AND experiments(active) < experiments(random) on ≥ 12/16 seeds | identification < 0.5 |

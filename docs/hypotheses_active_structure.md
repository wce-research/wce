# Preregistration — milestone "Active Structural Discovery" (frozen BEFORE any multi-seed / sealed run)

Freeze date: 2026-10-03. Criteria NOT modifiable after observation. Unit: seed (16). 95% CI by t over seeds.
Functional recovery = agreement ≥ 0.97 with the true boundary over the full region [−3,3]².
Query budget common to all active methods: 16. Capacity budget (H10): 400 floats, chosen before the run.
Targeted false-rejection threshold (H4): ≤ 0.10 (justification: 5% is unrealistic with 8–12% noise and ≤ 16 anchors;
we accept 1 good rule out of 10 wrongly rejected, against 0.51 at the previous milestone).

| H | Metric | Baseline | Min. effect | Success | Failure |
|---|---|---|---|---|---|
| H1 interleaving | functional recov. (held-out families + depth 3/4) | active-after-search, same query budget | +0.10 | diff ≥ +0.10 and 95% CI > 0 | diff ≤ 0 |
| H2 structural IG | entropy of the version space after 8 queries | random queries | −20% | reduction ≥ 20% more than random, ≥ 12/16 seeds | ≤ random |
| H3 functional clustering | family discovery recall (held-out); false promotions | AST anti-unification | +0.2 recall | recall ≥ AST + 0.2 AND false promotions ≤ AST + 0.5 / seed | recall ≤ AST |
| H4 noise-aware anchors | false rejection (correct rules); detection of false rules | binomial anchors of the previous milestone | — | false rejection ≤ 0.10 AND detection ≥ 0.6 (noise 8%) | false rejection > 0.25 or detection < 0.4 |
| H5 stability under noise | acceptance rate of functionally correct programs at 8% and 12%; acceptance of non-representable ones | stability test of the previous milestone | — | correct acceptance ≥ 0.7 at both levels AND non-repr. acceptance ≤ 0.35 | correct acceptance < 0.4 |
| H6 guided depth | candidate evaluations to solve the depth 3/4 families; solved rate | unguided depth 3/4 search | −30% cost | cost ≤ 0.7 × unguided AND solved rate ≥ unguided − 0.05 | cost ≥ unguided |
| H7 canonical classes | family reuse on transformed instances (rotation, translation, scale, reflection): recovery | non-canonicalised families | +0.1 | diff ≥ +0.1 | diff ≤ 0 |
| H8 OOD via structural experiments | extrapolation outside the region | same system, random queries | +0.03 | diff ≥ +0.03 and 95% CI > 0 | diff ≤ 0 |
| H9 gated residual | OOD degradation of the residual vs program alone; in-region accuracy | global residual (previous milestone) | — | OOD degradation(gate) ≤ 0.5 × OOD degradation(global) AND region ≥ global − 0.01 | degradation(gate) ≥ degradation(global) |
| H10 strict budget | retained skills (accuracy ≥ 0.85) under budget 400 | growth by experts alone, same budget | +30% | retained ≥ 1.3 × experts-alone AND 0 overrun | retained ≤ experts-alone |

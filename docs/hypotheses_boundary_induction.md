# Preregistration — milestone "Boundary Program Induction" (frozen before any execution)

Freeze date: 2026-10-02. Criteria NOT modifiable after observation. Unit: seed (≥ 16). 95% CI by t over seeds.
"Representable" = expressible by the primitive algebra at the maximal search depth.
Held-out families = absent from all induction code (no name, no dedicated helper); present only in the benchmark generator.

| H | Metric | Baseline | Success | Failure |
|---|---|---|---|---|
| H1 reconstruction of a representable held-out family | FUNCTIONAL recovery (agreement ≥ 0.97 with the true boundary over the full evaluation region) averaged over representable held-out families | known-family oracle fitter (upper bound) | ≥ 0.6 of the instances recovered AND ≥ 3 distinct held-out families recovered at least once | < 0.3 |
| H2 extrapolation | accuracy OUTSIDE the observed region | small MLP (comparable capacity) and memorisation (1-NN) | BPI > MLP + 0.05 AND BPI > 1-NN + 0.05, 95% CI of the paired difference > 0 | BPI ≤ MLP (CI contains or below 0) |
| H3 reusable parameterised family | on new instances of a family induced by anti-unification: accuracy via the macro vs full search; cost (evaluations) | full search without anti-unification | macro accuracy ≥ search − 0.02 AND evaluations ≤ 0.5 × search | macro accuracy < search − 0.05 |
| H4 active experiments | no. of observations to identify the program (P ≥ 0.95 and correct) | random queries | median(active) ≤ 0.6 × median(random) AND active identification rate ≥ random | median(active) ≥ median(random) |
| H5 TruthAnchors | refutation rate of FALSE self-consistent certificates; false refutations of correct certificates | certificate without anchor (0 refutation by construction) | refutation ≥ 0.7 AND false refutations ≤ 0.1 | refutation < 0.3 |
| H6 crystallisation | fraction of neural experts replaced by a program; mean accuracy loss of the replaced ones | experts kept | fraction ≥ 0.4 on representable families AND loss ≤ 0.02 | fraction < 0.15 or loss > 0.05 |
| H7 program + residual | accuracy and parameters on partially symbolisable functions | neural expert alone | accuracy(prog+residual) ≥ expert − 0.01 AND parameters ≤ 0.5 × expert | accuracy < expert − 0.05 |
| H8 non-representable recognised | abstention rate "no adequate program" on non-representable functions; false rejections on representable ones | — | abstention ≥ 0.8 AND false rejection ≤ 0.2 | abstention < 0.5 |
| H9 small basis | no. of primitives (types) needed to cover all the representable families of the benchmark (coverage ≥ 0.9) vs no. of families covered | "one rule per family" catalogue | primitives ≤ 0.5 × families covered | primitives ≥ families covered |

# Diagnostic report — WCE Deep Candidate Generation

EXPLORATORY sub-milestone of the milestone "Deep Compositional Structure Discovery & Model-Class Boundary".
Data: **dev partition only**. Neither the validation set, nor the confirmatory test, nor the sealed set has been touched.
Nothing here is confirmatory: all analyses are `POST-HOC / EXPLORATORY` in the sense of the preregistration
(`docs/hypotheses_deep_compositional_discovery.md`, frozen at tag `prereg-deep-compositional-discovery`).

Sources: `results/deep_generation_summary.json` (AWS run `wce-deepgen-20261004-134629`, 1,056 sub-units, 8 tasks per
depth and per regime), `results/deep_cover_summary.json` (run `wce-deepcover-20261004-142805`, 6 tasks per
cell), `results/deep_reachability.json`, `results/deep_coverage.json`, `results/deep_fragments.json`.
Targets: `make deep-reachability`, `deep-coverage`, `deep-fragments`, `deep-run-gen`, `deep-run-cover`,
`deep-gen-aggregate`, `deep-cover-aggregate`, `deep-gen-report`.

Sample sizes: 8 tasks per cell (depth × regime) for the recalls, 6 for the exhaustive coverage. The
proportions below therefore have a resolution of 1/8 or 1/6; they locate the problem, they do not quantify it
finely.

## Summary

The true structure does not enter the candidates beyond depth 2. The diagnosis separates five stages:

| Stage | Verdict | Measure |
|---|---|---|
| Representation | is not the problem | 100% of true structures representable, intermediates all valid |
| Composition | solved WHEN the atoms are known | with the true atoms, the true form is generated in 100% of cases up to depth 3, 75 to 88% at depths 4–5, 50% at depth 6 |
| Identifiability | sufficient in the standard and rich regimes | the true structure (true parameters) beats its variants deprived of one atom in 75 to 100% of cases |
| Ranking | is not the observed problem | no `RANKING_FAILURE` cause over 640 runs of depth 3–6 |
| Fitting / discovery of the atoms | **bottleneck** | recall of the true atoms 0.03 to 0.12 beyond depth 1; even when fitting ALL the forms, the true form is never selected at depth 3 |

The problem is therefore neither the grammar, nor the composition logic, nor the ranking: it is the localisation of the
parameters of the atoms under composition. The next sub-milestone is primitive discovery.

## 1. Why was the true structure not generated at depth 3–6?

Three mechanical causes, measured separately.

**(a) The operators.** The baseline, the flat strategy and the top-down one add only one atom at a time: they
build only chains. Share of true structures reachable by these operators (120 structures per
depth, analysis without fitting): 100% up to depth 3, 90% at 4, 77% at 5, 63% at 6. The missing
operator is the composition of two sub-structures (`compose_pair`).

**(b) The size of the space.** Number of distinct forms: 8, 108, 2,088 (exact enumeration), then ≈ 8.0·10⁴,
2.8·10⁶, 1.0·10⁸ (bound). The strategies generate 110 to 220 candidates. Exact measured coverage of the space at
depth 2: 58% (baseline, flat), 16% (bottom-up), 11% (bidirectional), 3% (factorised); at depth 3:
1.7 to 2.9% (baseline, flat), less than 1% for the others.

**(c) The atoms.** The residual-guided strategies do not search at random: they follow the atoms they
have fitted. Yet these atoms are not the true ones (section 3). Main cause recorded on the runs of depth 3–6
(32 per method and per regime):

| Method (budget 150) | `TRUE_PARENT_NOT_GENERATED` | `PRUNED_TOO_EARLY` | `BUDGET_EXHAUSTED` | `OPERATORS_INSUFFICIENT` | `FIT_FAILURE` | Recovered |
|---|---|---|---|---|---|---|
| Baseline, standard | 20 | 7 | 0 | 4 | 0 | 1 |
| Baseline, rich | 24 | 3 | 0 | 4 | 0 | 1 |
| Flat, standard | 20 | 2 | 5 | 4 | 1 | 0 |
| Flat, rich | 24 | 0 | 2 | 4 | 0 | 2 |
| Bottom-up, standard | 29 | 1 | 1 | 0 | 1 | 0 |
| Bottom-up, rich | 31 | 0 | 1 | 0 | 0 | 0 |
| Bidirectional, standard | 30 | 2 | 0 | 0 | 0 | 0 |
| Factorised, standard | 32 | 0 | 0 | 0 | 0 | 0 |
| Factorised, rich | 30 | 1 | 0 | 0 | 1 | 0 |

No occurrence of `GRAMMAR_UNREPRESENTABLE`, `CANONICALIZATION_COLLAPSE`, `TYPE_MISMATCH`, `RANKING_FAILURE` or
`OBSERVATIONAL_NON_IDENTIFIABILITY` at these depths. `FRAGMENT_MISSING` cannot appear: no strategy
used a library in this run.

## 2. Was it theoretically reachable?

Yes for the representation: 100% of true structures are expressible, and all their prefixes, suffixes, subtrees
and partial compositions are valid search objects, without key collision (closure of the parents: 100%).

Yes for the operators as soon as `compose_pair` or the synthesis over signatures is available: 100% at all
depths. No for 10 to 37% of the structures of depth 4–6 with the "one atom at a time" operators alone.

Minimal oracle path: 2d − 1 operations (1, 3, 5, 7, 9, 11). Required width (number of partial results to
hold): 2 up to depth 3, up to 3 beyond.

`Reachability` status: `REPRESENTABLE_BUT_NOT_REACHED` for almost all the failures;
`UNREACHABLE_WITH_CURRENT_OPERATORS` for 4 cases out of 32 (baseline and flat); `NOT_REPRESENTABLE` never.

## 3. Which operator or parent was missing?

The `compose_pair` operator for the balanced trees (section 1a). But the most frequent missing parent is of
size 1 or 2: the ATOMS themselves. Recall of the true atoms (same type, agreement ≥ 0.95 on the observed region) among
48 atoms fitted by gradient on the target:

| Depth | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Standard | 0.88 | 0.06 | 0.12 | 0.03 | 0.05 | 0.06 |
| Rich | 1.00 | 0.06 | 0.12 | 0.03 | 0.03 | 0.04 |
| Standard, threshold 0.90 | 1.00 | 0.38 | 0.21 | 0.12 | 0.10 | 0.10 |

The boundary-pair separation objective tried in this sub-milestone does worse (0.00 to 0.08 beyond
depth 1): negative result.

## 4. Did the pruning remove the right path?

Marginally. `PRUNED_TOO_EARLY` accounts for 0 to 7 cases out of 32 depending on the method (mostly the baseline in the
standard regime). In the great majority of cases, the right path is not pruned: it is never opened.

## 5. Was the budget sufficient?

For an unguided search, no, structurally: 150 fit-equivalents against 2,088 forms at depth 3.
But increasing the budget does not help the current strategies: at 450, the bottom-up one consumes on average 125 (standard) to
173 (rich) fit-equivalents and stops on its no-gain rule; the factorised one consumes 129 to 136 whatever
the cap. The recalls at 150 and at 450 are identical. The recovery / budget curve is flat.

The exhaustive enumeration shows that the budget is not the true limit. When fitting ALL the forms (4
initialisations each):

| | Forms fitted | Fit-equivalents | CPU | True form selected | Median rank of the true form | Agreement of the best form |
|---|---|---|---|---|---|---|
| Depth 2, standard | 116 | 464 | 18 s | 17% | 36 | 0.926 |
| Depth 2, rich | 116 | 464 | 19 s | 33% | 40 | 0.979 |
| Depth 3, standard | 2,204 | 8,816 | 420 s | 0% | 915 | 0.866 |
| Depth 3, rich | 2,204 | 8,816 | 470 s | 0% | 821 | 0.932 |

Even with 59 times the reference budget, the true form of depth 3 is never selected: it is generated
but badly fitted (its median rank is around 900 out of 2,204). The fitting of the parameters from heuristic
initialisations fails. It is the same bottleneck as section 3, seen from the other side.

## 6. Do the fragments help?

Little, and not by construction. Upper bound (library of 32 fragments built on the TRUE structures of 120
curriculum tasks of depth 2–3): 58 to 67% of the true sub-structures with 2 atoms are in the library; the
oracle path shortens only by 8 to 12%; the net compression gain is negative (−166 bits over 60 structures of
depth 4, the library costing more than it saves). The evaluation world draws its compositions at random:
there is no regularity of composition to learn. The fragments do not change the reachability either once the
synthesis over signatures is available, since it already enumerates all the pairs.

Not measured: the effect of the fragments on the residual strategies in a complete run (the variants "exact fragments /
anti-unified / transferred" were not launched, the diagnosis having shown that the bottleneck is upstream).

## 7. Does anti-unification help?

As a group compressor, no: anti-unifying all the fragments of the same size gives the trivial schema (negative
result, corrected into pairwise anti-unification). As a generator, modestly: the one-variable schemas derived from the
pairs of fragments propose 30 schemas of which 16 are absent from the library, and the coverage of the true sub-structures
goes from 58–67% to 68–79%. Cost: these schemas cover a large share of the 108 forms with 2 atoms; it is a widening
of the generation, not a targeted generalisation.

## 8. Does typing really reduce the space?

At the level of the atoms, yes: 30 well-typed unary nodes out of 45 raw ones (one third discarded; 70% at depth 3 if one
composed raw nodes). At the logical level, no: any pair of regions is composable by and / or / xor, typing
discards no composition. The useful reduction comes from the signatures: on an example of depth 3 with 15 literals,
28% of the 2,727 compositions attempted are merged (same signature) and 30% discarded as degenerate (constants, or
identical to a child). This rate has not been aggregated over the whole run.

## 9. Does the bidirectional one finally meet the right subspace?

The bidirectional one by fits, no: recall at generation 0.12 at depth 2, 0 beyond; coverage
almost zero at depth 3. The meeting over signatures (bottom-up tables + top-down typed holes + bilinear
product), yes, PROVIDED the atoms are available:

| Depth | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| True form generated, true atoms alone (standard) | 1.00 | 1.00 | 1.00 | 0.75 | 0.88 | 0.50 |
| With 12 parasite atoms (standard) | 1.00 | 1.00 | 1.00 | 0.75 | 0.62 | 0.12 |
| … and ranked first before fitting | 1.00 | 1.00 | 0.62 | 0.50 | 0.25 | 0.00 |
| Synthesis time | 0.02 s | 0.03 s | 0.05 s | 0.06 s | 0.19 s | 2.4 s |

The failures at depths 4–6 with the true atoms are not explained in detail (hypotheses: bound on the number of
literals, merging by signature of an equivalent form, beam of the holes). At depth 6 with parasites, the
synthesis is not enough.

## 10. What is the Recall@generated per depth?

| Method (budget 150) | d1 | d2 | d3 | d4 | d5 | d6 |
|---|---|---|---|---|---|---|
| Baseline, standard | 1.00 | 0.88 | 0.12 | 0.00 | 0.00 | 0.00 |
| Flat, standard | 1.00 | 0.75 | 0.12 | 0.00 | 0.00 | 0.00 |
| Bottom-up, standard | 1.00 | 0.25 | 0.12 | 0.00 | 0.00 | 0.00 |
| Bidirectional, standard | 1.00 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |
| Factorised, standard | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Baseline, rich | 1.00 | 0.88 | 0.12 | 0.00 | 0.00 | 0.00 |
| Flat, rich | 1.00 | 1.00 | 0.25 | 0.00 | 0.00 | 0.00 |
| Bottom-up, rich | 1.00 | 0.38 | 0.00 | 0.00 | 0.00 | 0.00 |
| Factorised, rich | 1.00 | 0.00 | 0.12 | 0.00 | 0.00 | 0.00 |

No improvement: the new strategies generate the true form LESS often than the baseline at depth 2.
The other recalls (standard regime, baseline / flat / bottom-up): after pruning 0.50 / 0.62 / 0.25 at depth 2;
well fitted 0.25 / 0.12 / 0.12; in the 5 best 0.50 / 0.12 / 0.12; selected 0.25 / 0.12 / 0.12. From
depth 4 onwards, all the recalls are zero for all the methods. The recall before pruning is not measured separately:
it is equal to the recall at generation in this instrumentation.

The exploratory diagnostic objective ("substantial improvement at depth 3, non-trivial fraction at
depth 4") is NOT reached by the complete search. It is reached only by the isolated logical stage (section 9).

## 11. Cost per true structure made reachable

By the complete search: undefined (no structure of depth ≥ 4 made reachable). By the logical stage with
true atoms: 0.05 to 0.2 s of CPU and 10³ to 10⁶ pairs evaluated per structure up to depth 5, without any
gradient fitting.

Separate mean costs (standard regime, all depths):

| Method | Fit-equivalents | Candidates | Compositions attempted | Total CPU | of which fitting | of which synthesis |
|---|---|---|---|---|---|---|
| Baseline | 132 | 109 | 0 | 40.0 s | 40.0 s | 0 |
| Flat @150 | 127 | 119 | 0 | 17.0 s | 16.5 s | 0 |
| Bottom-up @150 | 116 | 173 | 0 | 12.4 s | 12.1 s | 0 |
| Bidirectional @150 | 89 | 141 | 0 | 8.1 s | 7.9 s | 0 |
| Factorised @150 | 129 | 217 | 3.5·10⁸ | 20.5 s | 7.2 s | 13.2 s |

For the factorised one, the synthesis (generation) has become the dominant item, ahead of the fitting: the hot spot is no
longer `fit`. Pruning, canonicalisation, equivalence grouping and fragment search are not
timed separately (limit of the instrumentation).

## 12. Does the gain come from better composition or from more computation?

There is no recovery gain to attribute. What is established: (i) more computation does not help (budget 450 = budget
150; the exhaustive enumeration at 8,816 fit-equivalents does not select the true form); (ii) the composition
by signatures solves the logical stage in a fraction of a second when the atoms are right. The limiting factor is
neither the composition nor the computation: it is the atoms.

## Identifiability, distinguished from generation

With the true structure and its true parameters, without search: share of the tasks where it beats all its variants deprived
of one atom (Bayes factor > 3).

| Regime | d2 | d3 | d4 | d5 | d6 | Source |
|---|---|---|---|---|---|---|
| Hard (160 points, total noise 15%) | 0.70 | 0.47 | 0.15 | 0.07 | 0.03 | 40 tasks, local |
| Standard (300 points, 10%) | 1.00 | 0.93 | 0.88 | 0.70 | 0.72 | 40 tasks, local |
| Standard | 1.00 | 1.00 | 0.88 | 0.88 | 0.75 | 8 tasks, AWS |
| Rich (600 points, 7%) | 1.00 | 1.00 | 0.97 | 1.00 | 0.97 | 40 tasks, local |
| Rich | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8 tasks, AWS |

In the hard regime, depths 4–6 are not identifiable whatever the search: the weakest atom
modifies a dozen points for 24 expected flipped labels. In the standard and rich regimes, the failure is not
an identifiability problem.

## Quality of the basis

The basis is redundant: band = two parallel half-planes, ring = disc ∧ anti-disc, sector = two half-planes through
the apex. A true structure has on average 1.5 times more atoms in the minimal basis {half-plane, disc} (1.67; 3.07;
4.52; 6.09; 7.58; 9.07 for depths 1 to 6). A structure can therefore be recovered functionally under
another form, counted as a failure of form recall. No structure requires an absent primitive; none is
artificially long relative to its declared depth.

## Canonicalisation

No case of `CANONICALIZATION_COLLAPSE` measured. Two precautions verified by tests: the key of a sub-structure
applies no rewriting that depends on the parameters (otherwise a fitted `translate(0, 0)` would change key); two
associatively equivalent forms keep distinct keys during construction and are merged only at evaluation.
In the synthesis, forms with the same signature are merged but up to two alternatives are kept.

## Bugs found

- In the rows of the previous dev exploration, the boolean "true structure generated" was overwritten by the number of
  candidates generated; the `fate` field remained correct. Fixed for this sub-milestone (distinct fields).
- The fragment library canonicalised the fitted programs with a rewriting that depends on the parameters: one and the same
  fragment received two keys. Fixed (`normalize`).
- First attempt at the factorised generator without a bound on the number of literals: tables of several million rows,
  more than 10 minutes without result. Fixed by a diversity selection (14 literals at most).

## Limits

- 8 tasks per cell (6 for the coverage): low statistical resolution.
- The recall before pruning is not distinct from the recall at generation; the times of pruning, canonicalisation and
  equivalence are not separated.
- The failures of the logical stage at depths 4–6 with the true atoms are not explained in detail.
- The direct geometric detection (lines and circles by consensus) and the separation objective failed; they are
  described here only as negative results.
- The "fragments" variants of the residual strategies and the depth curriculum were not executed under
  real conditions.

## Answer to the exit question

Why the true deep structures were not generated: because the parameters of the atoms are not
found under composition (atomic recall ≈ 0.05), which leaves the guided search without purchase and makes the exhaustive
enumeration itself ineffective; secondarily because the "one atom at a time" operators exclude 10 to 37% of the
trees of depth 4–6.

Which minimal modifications make generation possible without brute force: the synthesis over signatures (bottom-up
tables, typed holes, bilinear meeting) is enough for the composition up to depth 5 — BUT only once
the atoms are available. This condition is not met today. The exit criterion of the sub-milestone is therefore
reached only by half: the cause is established experimentally, the complete remedy is not. The next bottleneck is
primitive discovery (`docs/rapport_primitive_discovery.md`).

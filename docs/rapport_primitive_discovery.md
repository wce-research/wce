# Report — WCE Primitive Discovery

EXPLORATORY sub-milestone of the milestone "Deep Compositional Structure Discovery & Model-Class Boundary".
Data: **dev only** partition (atomic benchmark and training scenes on dedicated sub-ranges of dev).
Neither the validation, nor the confirmatory test, nor the sealed set has been touched. Nothing here is confirmatory: everything is
`POST-HOC / EXPLORATORY` in the sense of the preregistration.

Sources: `results/deep_primitive_summary.json` (AWS run `wce-deepprim-20261004-150003`, 1,536 sub-units, none in
error), `results/deep_neural_training.json` (run `wce-deepneural-20261004-145510`).
Targets: `make deep-train-proposer`, `deep-install-proposer`, `deep-run-prim`, `deep-prim-aggregate`,
`deep-prim-report`.

Sample sizes: 144 atomic scenes per method (9 categories × 16), i.e. 248 true atoms; 8 scenes per out-of-distribution
shift; 8 scenes per active policy; 8 tasks per cell (depth × regime) for the downstream chain.

## Summary

The question of the sub-milestone was: can the primitives be discovered well enough to make deep structures
accessible to composition? **Measured answer: not yet.**

- None of the new proposers exceeds direct gradient fitting, which remains the best method of the atomic
  benchmark (equivalence 0.56 versus 0.46 to 0.49 for the geometric and hybrid methods).
- The compact neural proposer brings no recall gain: alone, it is the weakest (0.22); as a hybrid, it
  equals the hybrid without a network (0.49). It is, on the other hand, the most precise (0.54) and the fastest.
- The causal chain is verified: when all the primitives are recovered, a functional candidate structure
  appears in 56 to 100% of cases, versus 8 to 23% otherwise. The lack of atoms does explain the generation failure.
- A second bottleneck appears downstream: even with the TRUE atoms, final recovery at the 0.99 threshold is reached
  in only 0 to 25% of cases, whereas the mean agreement is 0.85 to 0.98.

## 1. Architecture

`core/deep_primitives.py` — distinctions maintained in the code: `Primitive` (basic element) ≠ `Fragment`
(reusable composition) ≠ `CompositeStructure` (assembly).

- `PrimitiveCandidate`: explicit object — type, parameters, domain, observational support, uncertainty, explained
  residual, descriptive cost, invariants, lineage. Never a latent vector.
- `PrimitiveProposal` (interface): `GradientProposer` (reference), `GeometricProposer` (lines and circles by
  consensus, no fitting), `ContrastProposer` (gradient on a contrast objective), `RefiningProposer`
  (hybrid: the proposals of the others serve as initialisation), `NeuralProposer` (compact network).
- `PrimitiveValidation`: the WCE validator. Explicit signal, independent of polarity: among the pairs of neighbouring
  points that the candidate's boundary separates, the share of pairs with opposite labels must exceed the base rate
  (z gap). Threshold corrected for multiple comparisons (√(2·ln N) + 1). Criteria: support, contrast, stability under
  resampling, equivalence with a retained primitive.
- `CompatiblePrimitives`: weighted multiple hypotheses, grouped by equivalence classes (agreement or complement
  ≥ 0.97 on the observed region); the type of a primitive may remain open.
- `discover`: residual-guided loop (propose → validate → retain → remove the explained pairs → propose again),
  stopping when no candidate passes validation any more. The residual is a set of unexplained pairs, not a
  memory of points.
- `PrimitiveFamily`, `PrimitiveLibrary`, `BirthProposal`, `merge_redundant`: parametric families, library,
  birth, merging, retirement.
- `core/deep_primitive_eval.py` (evaluation): metrics and identifiability classes. `core/deep_primitive_world.py`
  (evaluation): atomic benchmark.

## 2. Results of the atomic benchmark

| Method | Exact | Functional | Equivalence | Components | Precision | Parameter error | Retained classes | Proposals | Equiv.-fitting | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Direct gradient (reference) | 0.34 | 0.47 | **0.56** | 0.60 | 0.37 | 0.24 | 2.1 | 76 | 38 | 1.0 s |
| Analytic geometric | 0.29 | 0.38 | 0.46 | 0.53 | 0.25 | 0.20 | 4.3 | 159 | 0 | 0.15 s |
| Contrast | 0.05 | 0.23 | 0.34 | 0.39 | 0.12 | 0.42 | 4.4 | 105 | 63 | 2.3 s |
| Hybrid | 0.27 | 0.38 | 0.49 | 0.56 | 0.19 | 0.28 | 4.6 | 301 | 67 | 3.4 s |
| Hybrid, minimal basis | 0.28 | 0.41 | 0.47 | 0.56 | 0.27 | 0.07 | 4.6 | 245 | 50 | 1.4 s |
| Compact neural | 0.12 | 0.17 | 0.22 | 0.26 | **0.54** | 0.12 | 0.8 | 36 | 0 | 0.03 s |
| Neural + hybrid | 0.28 | 0.38 | 0.49 | 0.56 | 0.20 | 0.29 | 4.5 | 368 | 65 | 3.4 s |

Definitions: *exact* = same type and agreement ≥ 0.95; *functional* = agreement ≥ 0.95 or complement, any type;
*equivalence* = all the boundary components of the atom (lines, circles, half-lines) recovered;
*precision* = share of the retained classes all of whose components correspond to a true boundary.

Reading: fewer than six atoms out of ten are recovered even by the best method, and between half and nearly nine
tenths of the retained primitives are false. Compactness: the hybrid methods retain 3 classes per true atom
(1.4 for the gradient, 0.5 for the network).

### By atom type (hybrid)

| Atom | n | Exact | Equivalence | Components |
|---|---|---|---|---|
| Half-plane | 43 | 0.28 | 0.93 | 0.93 |
| Band | 25 | 0.32 | 0.64 | 0.80 |
| Disc | 103 | 0.42 | 0.53 | 0.53 |
| Anti-disc | 25 | 0.04 | 0.32 | 0.32 |
| Ring | 31 | 0.03 | 0.03 | 0.26 |
| Sector | 21 | 0.05 | 0.05 | 0.29 |

Lines are recovered well; circles half of the time; rings and sectors almost never. The recognition
of macros (two concentric circles → ring, two half-lines → sector) does not work.

### By category (equivalence)

| Category | Gradient | Geometric | Hybrid | Neural | Neural + hybrid |
|---|---|---|---|---|---|
| One atom | 0.63 | 0.44 | 0.50 | 0.25 | 0.44 |
| Two atoms | 0.28 | 0.44 | 0.38 | 0.16 | 0.38 |
| Several atoms | 0.17 | 0.36 | 0.34 | 0.08 | 0.44 |
| Close atoms | 0.25 | 0.56 | 0.56 | 0.25 | 0.53 |
| Nested atoms | 0.72 | 0.56 | 0.59 | 0.25 | 0.66 |
| Redundant atoms | 0.97 | 0.75 | 0.81 | 0.38 | 0.81 |
| High noise | 0.63 | 0.38 | 0.50 | 0.13 | 0.38 |
| Clustered outliers | 0.88 | 0.56 | 0.63 | 0.31 | 0.69 |
| Partial boundary | 0.50 | 0.06 | 0.06 | 0.19 | 0.06 |

The direct gradient dominates on the simple scenes (one atom, redundant, noise, outliers, partial boundary); the
geometric and hybrid methods do better as soon as there are several atoms (two, several, close). No method
is good everywhere. Partially observed boundaries defeat the geometric methods (0.06).

## 3. Identifiability

Classification of the true atoms BEFORE judging (contrast of the true atom on the observations): 244 identifiable, 4
partial, 0 non-identifiable. The atomic benchmark therefore contains almost only identifiable atoms: the recalls
above are not penalised by atoms that are impossible to see. Equivalence on the identifiable ones only: 0.56
(gradient), 0.49 (hybrid), 0.50 (neural + hybrid), 0.22 (neural). On the 4 partial ones: 0 to 0.25.

Consequence: the discovery failure is not a problem of identifiability of the atoms. The signal exists (the true
atoms have a high contrast), but neither do the proposers find it reliably, nor does the validator isolate it from the
false primitives.

## 4. Compact neural proposer

| | |
|---|---|
| Role | proposes circles (centre, radius) and lines (anchor, direction); predicts neither structure nor labels |
| Parameters | 23,881 |
| Memory | 95.5 kB |
| Cost of one pass | 13.7 million multiply-adds |
| Training | 4,000 dev scenes, 30 epochs, 158 s on `c6i.xlarge` |
| Loss | 3.28 → 1.54 (full curriculum); 3.04 → 0.99 (atoms only) |
| Real gain on primitive recall | **none**: 0.22 alone; 0.49 as a hybrid, like the hybrid without a network |

The network is precise but mute: it proposes little (36 candidates), 0.8 retained class per scene. Its final validation remains
on the WCE side.

### Out-of-distribution generalisation (two-atom scenes, equivalence)

| Shift | Hybrid (no learning) | Neural, full curriculum | Neural, atoms only | Neural + hybrid |
|---|---|---|---|---|
| Nominal reference | 0.38 | 0.16 | not measured | 0.38 |
| New positions | 0.25 | 0.00 | 0.06 | 0.25 |
| New sizes | 0.50 | 0.06 | 0.06 | 0.50 |
| New orientations | 0.69 | 0.19 | 0.31 | 0.63 |
| Different densities | 0.56 | 0.06 | 0.19 | 0.56 |
| Different noise | 0.19 | 0.06 | 0.13 | 0.19 |

The network does not generalise: its recall falls to 0.00–0.06 on new positions, sizes, densities or noises.

### Curriculum

The network trained on the full curriculum (atoms → two atoms → multiple scenes → deep compositions) does
no better, out of distribution, than the one trained on atoms only (0.00–0.19 versus 0.06–0.31). The comparison in
nominal distribution was not run. The effect of the curriculum on deep structures is therefore not established, and
the little that is measured does not support it.

## 5. Rich, minimal, learned basis

| | Rich basis (8 atoms) | Minimal basis (half-plane, disc, anti-disc) |
|---|---|---|
| Equivalence | 0.49 | 0.47 |
| Precision | 0.19 | 0.27 |
| Parameter error | 0.28 | 0.07 |
| Time | 3.4 s | 1.4 s |
| Final downstream agreement, rich d2 / d3 | 0.978 / 0.930 | 0.969 / 0.925 |
| Final downstream agreement, standard d2 / d3 | 0.929 / 0.868 | 0.949 / 0.921 |

The minimal basis is on a par in recall, better in precision and in parameter error, for 2.4 times less time.
Nothing indicates that the redundant atoms (band, ring, sector) help discovery. Length of the structures: in
the minimal basis a true structure has 1.5 times more atoms (measured in the previous sub-milestone). The "learned" basis
was not evaluated as a working basis: only birth proposals were produced (section 8).

## 6. Active measurements directed towards the primitives

16 additional measurements, then rediscovery (hybrid).

| Scenes | No measurement | 16 random measurements | 16 directed measurements |
|---|---|---|---|
| Two atoms — equivalence | 0.31 | 0.56 | 0.56 |
| Two atoms — precision | 0.12 | 0.27 | 0.27 |
| Several atoms — equivalence | 0.28 | 0.47 | 0.47 |
| Several atoms — precision | 0.31 | 0.39 | 0.44 |

Additional measurements help. The directed policy (disagreement internal to the primitive classes) does no better
than random in recall; the precision gain on the multiple scenes (0.44 versus 0.39) rests on 8 scenes.
Negative result for the active policy.

## 7. Causal chain: primitives → candidates → recovery

### Oracle decomposition matrix

| | True composition | WCE composition |
|---|---|---|
| **True atoms** | identifiable in 75 to 100% of cases (bound) | candidate of the right form: 0.75 to 1.00; functional candidate (≥ 0.97): 0.00 to 1.00; final recovery (≥ 0.99): 0.00 to 0.25 |
| **Discovered atoms** (hybrid) | agreement 0.56 to 0.72 | functional candidate: 0.00 to 0.88; final recovery: 0.00 to 0.13 |

Detail by depth, true atoms + WCE composition:

| | d2 | d3 | d4 | d5 |
|---|---|---|---|---|
| Standard — candidate of the right form | 1.00 | 0.88 | 0.88 | 0.75 |
| Standard — functional candidate | 0.88 | 0.50 | 0.38 | 0.00 |
| Standard — final recovery | 0.13 | 0.00 | 0.00 | 0.00 |
| Standard — mean final agreement | 0.977 | 0.968 | 0.923 | 0.850 |
| Rich — candidate of the right form | 1.00 | 0.88 | 1.00 | 0.88 |
| Rich — functional candidate | 0.88 | 1.00 | 0.75 | 0.75 |
| Rich — final recovery | 0.25 | 0.13 | 0.00 | 0.00 |
| Rich — mean final agreement | 0.977 | 0.984 | 0.972 | 0.968 |

Discovered atoms + WCE composition (functional candidate):

| | d2 | d3 | d4 | d5 |
|---|---|---|---|---|
| Gradient, standard / rich | 0.25 / 0.50 | 0.25 / 0.38 | 0.00 / 0.25 | 0.00 / 0.00 |
| Hybrid, standard / rich | 0.63 / 0.88 | 0.13 / 0.50 | 0.00 / 0.25 | 0.00 / 0.00 |
| Minimal hybrid, standard / rich | 0.50 / 0.75 | 0.13 / 0.38 | 0.00 / 0.25 | 0.00 / 0.00 |
| Neural + hybrid, standard / rich | 0.63 / 0.88 | 0.13 / 0.63 | 0.00 / 0.25 | 0.00 / 0.00 |

Primitive recall on these same tasks (equivalence), standard d2 → d5: gradient 0.81 → 0.25; hybrid 0.56 → 0.15.
Rich: gradient 0.88 → 0.48; hybrid 0.75 → 0.50.

### Transitions and correlations (64 tasks per method)

| Method | Tasks where all the primitives are recovered | Functional candidate if all recovered | … otherwise | Final recovery if candidate | … otherwise | Correlation recall ↔ candidate | Correlation recall ↔ final agreement |
|---|---|---|---|---|---|---|---|
| Gradient | 16 | 0.56 | 0.08 | 0.15 | 0.00 | 0.57 | 0.47 |
| Hybrid | 7 | 0.86 | 0.23 | 0.05 | 0.00 | 0.57 | 0.50 |
| Minimal hybrid | 7 | 0.86 | 0.18 | 0.25 | 0.00 | 0.54 | 0.39 |
| Neural + hybrid | 8 | 1.00 | 0.21 | 0.05 | 0.00 | 0.64 | 0.51 |

The lack of atoms does explain the generation failure: recovering all the primitives multiplies by 4 to 7 the
probability of having a functional candidate. But the next transition is poor: having a functional candidate
yields final recovery in only 5 to 25% of cases. Without a candidate, never.

At depth 2, the hybrid proposers give a functional candidate more often than the gradient (0.63–0.88 versus
0.25–0.50) whereas their primitive recall is lower: composition tolerates approximate primitives. This is
the only point where the new proposers do better downstream, and it does not translate into final recovery.

## 8. Families, library, birth, merging, retirement

From the validated primitives of the atomic benchmark (hybrid, contrast ≥ 5):

- **Parametric families**: 8 families (one per type), for example `norm.le(θ0∈[−4,07 ; 2,39], θ1∈[−3,05 ; 3,09],
  θ2∈[0,07 ; 3,52]) ×127`. The ranges cover almost the whole domain: these families bring no information
  beyond the type. The passage instances → family → compact parametrisation is mechanically carried out, not useful.
- **Proposed births**: 5 recurrent relations detected, 2 accepted — two concentric discs (3 tasks,
  2 parameters saved) and two parallel half-planes (5 tasks, 1 parameter). These are the ring and the band: the
  mechanism recovers atoms that the grammar already possesses. Three proposals seen on a single task are
  refused (no opportunistic birth). No birth was validated by a prediction or
  compression gain measured downstream.
- **Retirement**: no type is a candidate for retirement (all are validated at least twice).
- **Merging**: carried out at discovery by the equivalence classes.
- **Persistent library**: usage frequencies of the types (disc 39%, half-plane 16%, sector 13%…). Its effect on
  discovery time, recall or transfer to new worlds was **not measured**: no second
  pass using the library was run.

## 9. Errors and negative results

- The new proposers do not beat the direct gradient on the atomic benchmark.
- The neural proposer brings no recall and does not generalise out of distribution.
- The contrast objective alone is the worst method (equivalence 0.34, precision 0.12).
- The validator lets many false primitives through (precision 0.12 to 0.37 excluding the network). Mechanisms
  identified: a boundary tangent or parallel to a true boundary separates true opposite pairs; atoms
  with two boundaries (bands, rings) sweep several boundaries at once.
- First validator without multiple-comparison correction: three primitives retained on random
  labels. Corrected (threshold √(2·ln N) + 1) before the run.
- Recognition of macros (ring, sector): almost nil.
- Directed active policy: no better than random.
- Curriculum: no positive effect measured.

## 10. Limits

- 8 tasks per cell for the downstream chain and the shifts: resolution of 1/8.
- The final recovery threshold (agreement ≥ 0.99 on [−3, 3]², extrapolation zone included) is severe; the mean
  agreement is reported alongside.
- The loss between "functional candidate" and "final recovery" was not decomposed (parameter polishing,
  simplification, ranking). The preregistration forbade modifying these stages during this sub-milestone.
- A single network, a single architecture, a single training budget.
- The learned basis, the effect of the library and birth validated downstream are not measured.
- A single synthetic world in two dimensions.

## 11. Next steps

1. Decompose the downstream loss with the true atoms: candidate of the right form in 75 to 100% of cases, final
   recovery 0 to 25%. This is measurable without touching the sealed set and it is now the best-isolated bottleneck.
2. Discovery: combine the direct gradient (best on the simple scenes) and the geometric (best on the scenes
   with several atoms), and harden the validator against tangent boundaries.
3. Work in the minimal basis: same recall, better precision, 2.4 times less time.
4. Neural proposer: keep it as a high-precision filter, or abandon it; nothing justifies enlarging it.

## Readiness for WCE.01

Objective assessment, component by component.

| Component | State | Evidence |
|---|---|---|
| Primitive discovery | **missing — blocking** | best equivalence 0.56; precision 0.12 to 0.37 excluding the network; rings and sectors not recovered; recall 0.15 to 0.25 in compositions of depth 5 in the standard regime |
| Compositional reconstruction | **partial — blocking** | with the true atoms, the right form is a candidate in 75 to 100% of cases, but final recovery does not exceed 25% |
| Fragment learning | **exists, not useful** | the library covers 58 to 67% of the sub-structures but the net compression gain is negative; the world has no compositional regularity |
| Uncertainty | **partial** | equivalence classes, multiple hypotheses of primitives and of structures, four levels of uncertainty implemented; none is calibrated or evaluated in confirmatory mode |
| Model-class boundary | **implemented, not evaluated** | eleven out-of-class signals and a three-outcome decision exist; the boundary has been neither calibrated on validation nor tested |
| Compact neural proposal | **exists, no gain** | 23,881 parameters, precision 0.54, recall 0.22, no out-of-distribution generalisation |
| Lifelong library growth | **missing** | families and births proposed; no effect measured on a second pass, no growth validated |

**Ready**: the hierarchical representation, composition synthesis on signatures, the batch fitting engine,
the reachability instrumentation, the AWS infrastructure (checkpoints, resumption, progress), the discipline of
preregistration and of the sealed set.

**Missing**: a reliable primitive discovery; a precise validator; the recognition of atoms with two
boundaries; a faithful final recovery once the atoms are known; the evaluation of the class boundary; a
library whose growth improves something.

**Blocking**: (1) primitive discovery; (2) the loss between correct candidate and final recovery. As long as
these two points are not resolved, the thesis "primitives + transformations + composition + invariants → complex
structure" is demonstrated only for the isolated composition stage, with supplied atoms.

**Verdict: WCE.01 is not ready.** The sealed set remains intact and the 16-seed confirmatory campaign has not been
run; running them in this state would amount to consuming the sealed set to confirm a failure already measured in dev.

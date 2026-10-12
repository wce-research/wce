# Frontend scientific audit — Phase A

Baseline: `bcce9eb38cd43ab877e11cd9fad558953c5d5c66`, 12 October 2026. Scope: public tracked repository, local frontend; no infrastructure work. English scientific sources take precedence over translations. No sealed task was loaded, generated or evaluated.

## Initial assessment

The frontend is already a lightweight research application, not just a landing page: bilingual routes, ten mechanism dossiers, source inventory, three source-backed evidence views, a symbolic laboratory, themes, responsive navigation, canonical URLs, OpenGraph, structured metadata exist. Its central scientific weakness is recency and coverage: the active deep-discovery programme and its latest negative verdict are absent. The site explains mechanisms better than it explains how experiments changed the programme.

The three existing evidence views read JSON artifacts and expose fingerprints. This is a sound pattern to extend, rather than replacing the framework. Static source paths are not yet actionable links. Three-horizon maturity labels do not clearly distinguish a tested implementation, a blocked capability and a future integration. The initial source inventory covers modules but cannot establish scientific completeness or comprehension.

## Scientific inventory and coverage matrix

Explanation grades: good = role, scope and example; partial = mechanism explained but evidence or connection missing; absent = no public reading path. A filename in the inventory does not count as an explanation. The visual column describes the baseline; the action column is the Phase B recommendation.

| Backend/source | Important information | Frontend before | Current page | Explanation / visual | Gap / recommended action |
|---|---|---|---|---|---|
| README | Compact compositional thesis, explicit hypotheses | Partial | Home, vision | Good introduction / configuration study | Add current negative verdict and five maturity states |
| core/world.py (World, Entity, Relation); tests/test_world.py | World, entity, relations, constraints | Yes | World-state, lab | Good / state transition | Place within separate symbolic execution path |
| core/transformations.py, invariants.py, transform_graph.py | Operators, algebra, macros, preserved properties | Yes | Primitives | Good / composition | Connect primitive → composition → verification |
| core/algebra.py, reasoning.py; related tests | Symbolic algebra, conditional reasoning | Partial | Planning | Partial / plan branches | Explain implemented small-domain operations and source entry points |
| core/planner.py; tests/test_planner.py | Search from state to goal | Yes | Planning | Good / goal graph | Do not imply a general integrated planner |
| core/model_state.py, learning.py, plasticity.py, safe_plasticity.py | Structural changes, adaptive neural components | Yes | Learning | Good / adaptation | Keep experimental label and separate integration from existence |
| core/lifelong.py, symbolic.py, continuum.py | Lifelong/retention experiments | Yes | Learning, continual | Good / source-backed metric comparison | Preserve budget and distribution qualifiers |
| core/language_core.py, grounded.py, discourse.py | Language research distinct from command grammar | Yes | Language, lab | Good / text-to-structure | Explicitly separate language experiments from language-model replacement |
| core/source_reliability.py, entitlement.py, temporal_belief.py | Evidence, ignorance, temporal belief | Partial | Evidence | Partial / evidence graph | Add temporal dynamics and epistemic boundaries to hierarchy |
| core/measurement.py | World ≠ outcome ≠ belief, compatible states | Yes | Measurement | Good / short sensor diagram | Expand operator → outcome → compatible states → update |
| results/predictive_coverage.json | Coverage is not accuracy | Yes | Research/measurement | Good / source bars | Retain provenance and metric definition |
| results/continuum_summary.json | Retention, costs, calibration across methods | Yes | Research/continual | Good / intervals and tables | Keep nine distinct metrics; no global leaderboard |
| results/active_summary.json; hypotheses_active_structure.md | Supported, refuted, inconclusive hypotheses | Yes | Research/discovery | Good / filterable verdicts | Connect historical verdicts to later discovery work |
| rapport_robust_reconstruction.md; core/robust_* | Robust reconstruction and consumed historical sealed set | Absent | Source inventory only | Absent | Add milestone record; distinguish historical consumed set from current live set |
| hypotheses_deep_compositional_discovery.md; core/deep_* | Frozen baseline, deep composition programme | Absent | — | Absent | Public programme overview and report reading order |
| rapport_deep_candidate_generation.md; deep_generation.py | Generation blocked by missing atoms; grammar coverage | Absent | — | Absent | Hypothesis → diagnosis → consequence record |
| rapport_primitive_discovery.md; deep_primitives*.py | Discovery, neural proposal failures, narrow scope | Absent | — | Absent | Record failures and avoid perception claims |
| rapport_downstream_selection_loss.md; deep_downstream.py | Good candidates lost during fitting/selection | Absent | — | Absent | Explain why an available candidate need not become a result |
| architecture_validated_transformation.md; rapport_validated_optimizer.md; deep_fit.py | Propose/evaluate/commit-or-rollback; no-op can win | Absent | Branching only | Partial / snapshot | Add accepted/rejected transaction demonstration |
| rapport_objective_alignment.md; deep_objective.py | Joint objective matters; local surrogate can be destructive | Absent | — | Absent | Explain negative objective-alignment results |
| rapport_robust_primitive_discovery.md | Precision gains do not imply downstream improvement | Absent | — | Absent | Improved but insufficient record |
| rapport_partial_observation_parameters.md; deep_partial.py | Type-oracle confound, declared conservative verdict deviation | Absent | — | Absent | Explicit oracle label and deviation, not automatic readiness |
| rapport_fragment_type_inference.md; deep_fragments.py | Attribution and type inference separate | Absent | — | Absent | Explain fragments in hierarchy; exploratory arm labelled |
| rapport_boundary_ownership.md; deep_ownership.py | Ownership improves locally, filtering can harm; candidate baseline | Absent | — | Absent | Distinguish dev candidate baseline from official reference |
| rapport_deep_composition_oracle_atoms.md; deep_compose.py | Equivalence-class composition succeeds with exact atoms | Absent | — | Absent | Source-backed depth comparison, oracle scope next to every chart |
| rapport_end_to_end_discovered_atoms.md; deep_e2e.py | END-TO-END STILL BLOCKING | Absent | — | Absent | Record why downstream success does not solve upstream |
| rapport_latent_primitive_recovery.md; deep_lpr.py | Set recall collapses with depth, roughly recall^n | Absent | — | Absent | Complete-set bottleneck and measured depth curve |
| rapport_high_precision_primitives.md; deep_hp.py; deep_high_precision_summary.json | HIGH-PRECISION PRIMITIVES STILL BLOCKING; final ≥ .99 retreats | Absent | — | Absent | Primary/secondary regimes, paired intervals, thresholds, all HP verdicts |
| tests/test_*.py, README | Test scale and its meaning | Absent | — | No reliable current counter | Generate static declaration count with file fingerprints; do not call it executed/passing total |
| Makefile, CONTRIBUTING | Every user workflow through Makefile | Inconsistent | Reproduce | Bare Python commands remain | Use Makefile targets; explain dev/confirmatory/sealed policy |
| LICENSE, SECURITY, CITATION.cff | Open source, security, citation | Partial | Open | Source links incomplete | Direct canonical links and public contribution guide |
| docs/sealed_sets.md | Never inspect/tune current sealed tasks | Absent | Methodology | Absent | Policy only; never expose private salt/tasks |

## Obsolete or misleading material

- The active programme is missing: historical active-discovery charts cannot stand in for current deep discovery.
- “Three JSON artifacts” describes the old publication surface, not the available science.
- `docs/architecture_wce.md` says licence/contributions remain to be formalised, contradicting LICENSE and CONTRIBUTING; it also recommends bare Python commands against the current Makefile rule.
- An open-page card presents licence and complete-code publication as future work.
- README's “more than 5,000 tests” is not a verified current execution count. Static analysis finds 4,306 declared test methods at baseline; dynamic generation means these counts are not equivalent. Do not silently replace the scientific suite total with a static count.
- “Confirmed dev baseline candidate” does not mean confirmatory success. New components remain opt-in; no current 16-seed campaign or live sealed evaluation is established.
- There is no evidence of general vision, video, perception of real scenes, an operational autonomous scientist, AGI or superiority over LLMs. Existing cautionary wording is largely appropriate; preserve it.
- LinkedIn `WCE_LINKEDIN_URL` remains empty. A company slug cannot be inferred safely.

## Information architecture proposal

Keep the five existing top-level sections and the visual identity. Add a research programme page with a compact reading guide and expandable milestone ledger; a dedicated deep-composition evidence page; a dedicated High-Precision evidence page; a hierarchical architecture map; and a reproducibility/source checklist. Home shows a concise current-status entry, not the entire documentation. Research pages lead from hypothesis through result/verdict/consequence to report and code. Architecture separates conceptual relationships from actual integration.

Reading levels: seconds = definition/current blocker; minutes = conceptual map and reversible transformation; implemented = maturity register and code/tests; experiments = timeline and evidence pages; reproduction = canonical reports, source JSON, Makefile, contribution/security/licence/citation.

## Recommended explanatory visuals

1. Architecture pipeline plus a separate implemented symbolic path; arrows indicate conceptual flow, not a single deployed model.
2. Primitive-to-Boolean-composition diagram with native disclosure and source links.
3. World → measurement operator → outcome → compatible states → belief; world, observation and belief remain different objects.
4. User-stepped accepted/rejected proposal with state preservation; explicitly an illustration, not a research run.
5. Native expandable scientific timeline with hypothesis, experiment, result, verdict, consequence and next question; negative and inconclusive outcomes retained.
6. Source-backed oracle depth comparison and latest HP complete-set curve. No continuous animation, no heavy runtime. Reduced motion disables transition; all numbers remain readable without JavaScript.

## UX/engineering audit plan

Baseline has responsive layouts, keyboard navigation/disclosure, focus states, reduced-motion CSS, day/night themes, metadata and favicon, bounded laboratory API and safe 500 responses. These are implementation observations, not a completed browser audit. Sitemap and robots endpoints are absent at baseline; they must be generated from the route catalog. Current browser dependency is missing. Phase B must install isolated test tooling, run existing and added checks on desktop/tablet/mobile in FR/EN and both themes, inspect screenshots, check overflow/focus/contrast/reduced motion, internal anchors, source provenance, public GitHub destinations and container artifact availability. No current score or production-readiness claim is inferred from code inspection alone.

## Phase B acceptance conditions

Every displayed metric is read from an explicitly allowlisted public summary or a generated test metadata record; references and hashes stay inspectable. Public reports alone determine verdict labels. Report order is the research progression, not an invented date chronology. No backend scientific criteria, result files or sealed protocol are changed. Every route works in both languages, packaging includes only required public artifacts, and the final report distinguishes checks passed, unavailable checks and remaining production blockers.

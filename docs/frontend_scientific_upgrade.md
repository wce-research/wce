# WCE Labs frontend scientific upgrade — Phase B report

Date: 12 October 2026. Baseline: `bcce9eb38cd43ab877e11cd9fad558953c5d5c66`. Phase A: [scientific audit and 32-row coverage matrix](frontend_scientific_audit.md). This report concerns the frontend only; it is not a scientific validation of the research programme or approval to deploy.

## 1. Initial frontend

The site already offered 30 routes in two languages, ten mechanism dossiers, a static inventory of 167 source modules, three evidence views, a small symbolic laboratory, day/night themes and responsive navigation. The documentation was stronger than the research chronology. English was not the default; saved English selection was handled only at the root, and browser language preferences were not negotiated. Sitemap and robots endpoints were absent despite existing canonical and OpenGraph metadata.

## 2. Backend → frontend coverage before intervention

The Phase A matrix identifies 32 information areas. World-state, transformations, learning, language and the three historical evidence views had useful explanations. The fourteen scientific reports from robust reconstruction through High-Precision were not presented as a research progression. A module name in the source inventory was the only entry for many important mechanisms. The active deep-discovery bottleneck was not visible on the home page.

## 3. Missing scientific information discovered

The missing chain was candidate generation → primitive discovery → fitting/selection diagnosis → validated optimization → objective alignment → robust primitives → partial observation → fragments/types → ownership → oracle composition → end-to-end discovery → complete latent sets → high-precision estimates. Especially important: composition with exact supplied atoms succeeds, but discovering a complete, jointly precise set still does not. The partial-observation report explicitly applies a conservative deviation from its preregistered rule because of a type-oracle confound. Earlier consumed sealed sets, retired deep v1 and live private-salt deep v2 must be distinguished.

## 4. Obsolete information corrected

- The old “three JSON artifacts” publication description no longer defines the entire scientific surface.
- Licence, scientific source release and contribution rules are current public resources; a collaborative hub and common training interfaces remain future work.
- Reproduction instructions now use Makefile targets instead of bare Python workflows.
- Source paths are actionable canonical GitHub links, including licence, SECURITY and CITATION.
- English is the template source language and default UI language. French remains a translation; explicit URL locale wins, then saved language preference, then supported browser preference, then English.
- README's historical “more than 5,000 tests” is not copied into the frontend. A generated, fingerprinted inventory reports 4,317 static test declarations with an explicit definition. This excludes dynamically generated cases and is not an executed/passing total.

## 5. Final information architecture

The five top-level sections remain Research, Architecture, Vision, Open research and Laboratory. Four new routes, each in EN/FR:

- `/research/programme`: reading paths, five maturity states and expandable scientific progression.
- `/research/deep-composition`: source-backed oracle comparison by atom count, paired interval and end-to-end caveat.
- `/research/high-precision`: primary/secondary evidence, set recovery, HP1/HP3 levels versus preregistered thresholds, all HP1–HP13 verdicts, limitations and final verdict.
- `/architecture/research-map`: conceptual loop, distinct world/measurement/belief objects, primitive composition, transaction illustration, hierarchy of code families and bottleneck chain.

The result is 34 routes / 68 localized pages. Reading progresses from a current-status explanation on the home page to conceptual mechanisms, implemented scope, experiments and original sources. The milestone ledger uses native disclosure; order represents scientific dependencies rather than invented dates. Editorial diagnoses are labelled separately from verbatim report verdicts.

## 6. Diagrams and informative interaction

The architecture map explains Environment → Measurement → World/Belief → Transformations → Composition/Reasoning/Planning → Verification → Accept/Reject/Rollback → Learning. It explicitly states that these arrows do not prove integration into one operational model. Separate diagrams explain WorldState ≠ Observation ≠ BeliefState and measurement outcome → compatible states → belief update; primitive predicates → transformed primitives → Boolean compositions → equivalence classes; and the current discovery bottlenecks.

The transaction study lets a keyboard or pointer user accept a proposed move or reject it while preserving State A. It is labelled as an illustration, not a benchmark. Both branches remain readable without JavaScript. The movement is user-triggered; reduced-motion disables its transition. The new production script is 837 bytes and adds no library dependency. Scientific bars are supplementary to accessible numeric tables, not substitutes for exact values.

## 7. Scientific results now exposed

All fourteen public report milestones have hypothesis, experiment/source, outcome/consequence, verdict and next question. The previous coverage, continual-learning and active-discovery evidence remains available, including refuted and inconclusive hypotheses.

The composition summary reports functional recovery `0.987` for equivalence-class selection versus `0.1615` for `ref_b150` over the same oracle tasks; paired difference `+0.8255`, 95% CI `[+0.7865, +0.8620]`. Exact atoms and parameters are provided, up to six atoms. Functional recovery is not exact-tree recovery, and this comparison is not an end-to-end perception benchmark.

The High-Precision primary C arm has mean agreement `0.9116` and final ≥ 0.99 share `0.0755`, with `n=384`. Relative to B, final recovery changes by `−0.0208`, CI `[−0.0417, −0.0026]`, despite a positive mean-agreement difference. At depths 4–6, C individual recall is `0.374` versus the report's required `0.70`; all-present recovery is `0.0208` versus required `0.25`. The secondary dense regime is separately labelled. These numbers are read from the public summary; thresholds are read from report section 0.6. This report is a dated explanation of the view, not a second runtime metric store.

Final verdict: **HIGH-PRECISION PRIMITIVES STILL BLOCKING**. No confirmatory campaign proposed, no live sealed evaluation performed, nothing wired into the frozen baseline by default.

## 8. Explicit limitations

No demonstrated AGI, general LLM replacement, superiority over established architectures, robust real-scene perception or operational autonomous scientist. Synthetic geometric observations, occlusion, insufficient complete-set recall, joint parameter tolerances, candidate selection, out-of-class detection and validation power/safety remain bounded or unresolved. A better average is not a solved capability. Current implementations, experimental prototypes, blocked capabilities, future directions and long-term vision are explicitly distinguished.

## 9. Coverage after intervention

This is a qualitative coverage assessment, not a fabricated percentage of scientific completeness.

| Source area | After intervention | Explanation and remaining boundary |
|---|---|---|
| README thesis/current status | Home, programme, vision/now | Concrete definition, implemented symbolic path and current negative verdict |
| World/Entity/Relations | Existing world-state dossier + architecture map | Roles and demo scope remain explicit |
| Transformation algebra/invariants | Existing primitives dossier + composition flow | Composition is an object; diagram is conceptual, not a runtime trace |
| Symbolic algebra/conditional reasoning | Code hierarchy + source/tests | Equality-preserving operations and TRUE/FALSE/UNKNOWN explained; no general benchmark claimed |
| Planning | Existing planning dossier + hierarchy | Goal-directed state search and speculative branches, source/tests |
| Structural learning/neural plasticity | Existing learning dossier + hierarchy | Tested mechanisms distinguished from integrated-model claims |
| Lifelong/continual adaptation | Existing source-backed continual evidence + hierarchy | Multiple metrics/budgets retained; no overall leaderboard |
| Language experiments | Existing language dossier + hierarchy | Command grammar and neural research remain separate |
| Epistemic/temporal evidence | Existing evidence dossier + expanded hierarchy | Event time versus ingestion time, filtering/smoothing and evidence provenance |
| Measurement architecture | Existing measurement dossier + expanded flow | World, operator, outcome, compatible states and belief remain distinct |
| Predictive coverage | Existing evidence, canonical source links | Coverage is not accuracy; hash and definitions retained |
| Continual summary | Existing interactive evidence | Nine methods/metrics, intervals, exact tables and scope retained |
| Active-discovery hypotheses | Existing verdict-filtered evidence + programme link | Supported/refuted/inconclusive remain visible |
| Robust reconstruction | Programme milestone + report/summary | Historical consumed sealed evaluation distinguished from current live set |
| Deep programme/preregistration | Programme + source checklist | Frozen baseline and opt-in status explicit |
| Candidate generation | Programme milestone | Missing atoms versus grammar/generation diagnosis |
| Primitive discovery | Programme milestone | Negative neural-proposer evidence and narrow domains |
| Downstream selection | Programme milestone | Good candidate can be lost; diagnostics labelled exploratory |
| Validated optimizer | Programme + transaction illustration | Guardrail/no-op distinction, reject/rollback, sources |
| Objective alignment | Programme milestone | Single-primitive surrogate versus joint criterion |
| Robust primitives | Programme milestone | Improved but insufficient; downstream regressions retained |
| Partial observation | Programme milestone | Known-type oracle and conservative rule deviation disclosed |
| Fragments/type inference | Programme milestone + hierarchy | Attribution and type separated; exploratory arm labelled |
| Boundary ownership | Programme milestone + hierarchy | Local gains, filter harm, dev candidate versus official reference |
| Exact-atom composition | Dedicated evidence page | Depth bars/tables, paired CI, exact-tree distinction and oracle limits |
| End-to-end discovered atoms | Programme + current bottleneck | Negative verdict and upstream failure explained |
| Latent primitive sets | Programme + measured HP set curve | Complete-set requirement and depth collapse explained; historical report linked |
| High-Precision | Dedicated evidence page | Real primary/secondary numbers, thresholds, all thirteen verdicts, post-hoc arm |
| Test scale | Generated inventory + reproducibility | Source count/fingerprints; never presented as total passing tests |
| Makefile/workflows | Reproduction/methodology | Declared partitions, prerequisites, commands, budgets and context |
| Open source/licence/security/citation | Open pages + clickable dossier/inventory sources | Direct canonical destinations rather than filenames alone |
| Sealed policy | Methodology/reproduction | Policy and status only; no private task/salt disclosure |

The secondary research families remain scoped explanations with code/tests, not new claims of scientific confirmation. The frontend does not attempt to reproduce every report inline; researchers can descend to the canonical reports, summaries and source.

## 10. Tests and controls

Acceptance evidence is recorded below after the final runs. Earlier browser scans found a mobile scrollable-table keyboard defect; the fix adds named, focusable regions and visible focus styles across all evidence tables. A transient host network change interrupted one run; only that navigation error has a bounded retry. No failures are suppressed.

- `make check`: passed (Python compilation).
- `make frontend-test`: 198 tests passed, including HTTP/error handling, scientific source fidelity, public artifact allowlists and language precedence.
- Complete browser audit: 408 localized responsive/theme renders, 152 WCAG AA scans, zero violations. Existing regression suite: 112 renders passed.
- Final fresh-server check after editorial adjustments: 60 renders, 36 WCAG AA scans, zero violations; existing 112-render regression repeated successfully.
- Keyboard navigation, responsive tables, validation/rollback illustration, reduced motion, no-JavaScript reading and persistent/browser-detected language passed.
- Docker image built successfully; isolated read-only, network-disabled image renders scientific evidence and contains 14 milestones without importing the research runtime.
- Internal links, canonical URLs, sitemap/robots, HTML 404 and safe API 500 handling passed. Local scientific GitHub targets exist; 230 canonical GitHub source destinations were verified in the published repository tree after push.
- `git diff --check`: passed. No scientific result files, private sealed material or infrastructure were modified.

## 11. Performance and accessibility

System fonts, existing palette/branding and server rendering are retained. No heavy production dependency, continuous animation, external font or browser-side scientific runtime was introduced. JSON loading is restricted to an explicit public allowlist and immutable summaries are cached in-process. Exact numbers and source fingerprints remain inspectable without JavaScript.

Final local samples for the new scientific pages: maximum DOMContentLoaded 183.3 ms, load 183.7 ms, HTML 40,729 bytes, reported resources 63,170 bytes and 608 DOM nodes. These are local observations, not production budgets.

Automated WCAG AA scans, keyboard checks and screenshot inspection are acceptance evidence, not certification across every assistive technology. Local timing samples do not measure production Core Web Vitals, public network latency or load capacity. No deployment/performance claim is made from the localhost measurements.

## 12. Files modified

- `.dockerignore`
- `Dockerfile`
- `Makefile`
- `app/content.py`
- `app/copy.fr.json`
- `app/i18n.py`
- `app/presentation.py`
- `app/public_identity.py`
- `app/science.py`
- `app/science_catalog.py`
- `app/site.py`
- `app/static/science.js`
- `app/static/site.js`
- `app/static/styles.css`
- `app/templates/pages/architecture.html`
- `app/templates/pages/continual.html`
- `app/templates/pages/discovery.html`
- `app/templates/pages/index.html`
- `app/templates/pages/lab.html`
- `app/templates/pages/measurement.html`
- `app/templates/pages/open.html`
- `app/templates/pages/research.html`
- `app/templates/pages/vision.html`
- `app/test_inventory.json`
- `app/understanding.py`
- `app/web.py`
- `docs/architecture_wce.md`
- `docs/browser_checks.md`
- `docs/frontend_scientific_audit.md`
- `docs/frontend_scientific_upgrade.md`
- `tests/browser/science.cjs`
- `tests/test_science.py`
- `tests/test_web.py`
- `tools/build_test_inventory.py`

## 13. Commits

Phase A audit: `0f1fa8f`. Implementation: `cb28dbf`. This validation report is committed separately; its SHA is supplied in the final delivery. No history rewrite or force push.

## 14. Material deliberately not exposed

Private sealed salt, live sealed tasks/labels, credentials, operational run logs and machine-private configuration. The research runtime is not imported into the web app. No cloud runs, production service changes, AWS, Cloudflare, DNS or tunnel operations were performed. Scientific modules and result artifacts were not modified. New Docker entries only package explicit already-public reports and summaries.

## 15. Remaining points before production

- `WCE_LINKEDIN_URL` is still empty in `.env.example`, Compose's current default and public identity defaults. The definitive public page URL is required; it is not guessed. Only LinkedIn remains pending.
- This review does not approve infrastructure, transport security, operational capacity or deployment configuration. The small symbolic lab's in-memory, bounded session lifecycle remains documented.
- The scientific bottleneck and historical numerical reproducibility discrepancies remain research limitations, not hidden frontend defects. The full PyTorch research suite and experimental campaigns were not rerun during this frontend-only intervention.
- Browser artifacts remain in a temporary local audit directory; deployment checks belong to a separate explicitly authorized stage.

**FRONTEND READY FOR INFRASTRUCTURE REVIEW**

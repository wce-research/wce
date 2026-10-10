# WCE — direction of the research site

## References consulted

- https://www.anthropic.com/research: editorial organisation of the research.
- https://openai.com/research/: clear hierarchy and navigation between dossiers.
- https://amilabs.xyz/: AMI Labs site, associated with Yann LeCun; sobriety of the scientific presentation.

The WCE identity uses an ivory background, graphite and a cobalt accent. The editorial titles, thin rules, monospace annotations and trajectory diagrams compose an identity dedicated to the project. The diagram on the home page is a conceptual illustration, not a visualisation of results.

## Navigation

- `/`: short introduction and access to the dossiers.
- `/research`: index with filters by domain.
- `/research/measurement`, `/research/continual`, `/research/discovery`: dossiers and links to the artefacts.
- `/architecture`: operational path and exploratory modules.
- `/vision`: the three requested maturity levels.
- `/open`: current availability and steps towards the open hub.
- `/lab`: isolated interface of the symbolic engine.

The server serves only the explicitly authorised routes and assets. No CDN, remote font or new service is needed. The mobile menu works with the keyboard and closes with Escape. The filters expose their state through `aria-pressed`, the lab uses a results area announced to assistive technologies.

The host port remains `127.0.0.1:3018`, reserved for `wce/web` in the existing registries. The redesign introduces no additional port.

## Bilingual extension and themes — 4 October 2026

24 editorial pages are accessible under `/fr` and `/en`. The in-depth dossiers are structured in `app/content.py`; the catalogue `app/copy.en.json` translates the historical pages. The selector preserves the current dossier. The scientific sources remain unchanged.

The CSS tokens define the day and night palettes. `theme.js` applies the choice before rendering; the system provides the initial value, then the explicit choice persists. The submenus cover five sections, with hover, keyboard, Escape and mobile accordions.

The footer symbol measures 44 px on desktop; the GitHub, X, LinkedIn and Instagram marks measure 28 px. The SVG files are local and their sources/licences are documented in `app/static/social/`.

## Redesign of the presentations and of the backend map

The home page presents the thesis of small primitives and six research mechanisms. The architecture exposes the web path then ten cards: composition, world, planning, hybrid language, measurement, evidence, certificates, adaptation, discovery and branches. Dedicated cards make the subpages accessible from each of the five sections.

The cards have a sober rounded shape, a constant hierarchy (status, mechanism, description, conceptual chain, source), a visible focus state and a discreet hover disabled when reduced motion is preferred. The existing ivory/graphite/cobalt palettes are kept. The geometric compositions are decorative and hidden from screen readers.

Thirty pages are available in French and English. The bilingual catalogue describes the interfaces actually read in the code, with limits and protocol distinctions. The planner and the hybrid language kernel are explicitly described as research prototypes; the web simulator keeps its scope.

The collapsible inventory of `/open/source` is generated from the files present on the machine of the repository. It includes the public names of classes and functions, without importing the modules or publishing the environments, cloud identities or unselected raw result files. The additional symbols expand in each entry.

## Configuration studies and result registers

The illustration on the home page becomes a manipulable study: three primitives, three configurations, a constant frame and an explicit operator. It visually exposes the composition thesis while signalling its purely illustrative nature. No call to the engine or to the network is needed for these operations.

The cards now use ten specific diagrams: composition, state/action, search towards a goal, language verification, measurement, weighting of evidence, certificate probes, consolidation/rollback, exploration loop and branches. The dossiers distinguish these conceptual diagrams from execution traces and measured results.

The result views are computed at render time from the existing JSON files. The coverage is displayed per source; the continuum methods are compared on nine metrics, with intervals and tables; the ten discovery verdicts are all present and expandable. The order follows the artefacts, without an overall ranking. The cobalt/graphite colours identify the series, and the refuted verdicts use a discreet red accent. The views are adapted to both themes and to small screens.

## Comprehension path — 4 October 2026

The home page starts with a definition of WCE and a question / evidence / ambition path. The cards present understandable questions before pointing to the mechanisms and their sources. The manifesto explains what the weights represent and why the architecture matters.

Each results dossier makes explicit its question, its scope and what the experiment does not demonstrate. The architecture offers six expandable definitions, usable with the keyboard. The subpages have a breadcrumb and the vision uses French headings in the French version.

These components are rendered server-side, without an additional search dependency or an exposed hub service. The English variants are written explicitly; no scientific result is translated or recomputed.

## Dossiers: mechanisms and scope of the evidence

The ten architecture dossiers display three markers: identified code, targeted verification and current integration. The five families linked to the three public artefacts offer links to the associated experiments, with an explicit mention of their scope. The others signal the absence of a dedicated results dossier here: no badge claims that the mere presence of tests validates a general capability.

The table of contents receives touch targets of 44 px and a compact layout on mobile. The lab examples explain the lift / observe / move forward path, in French and in English, without announcing a free conversation. The tests of the web interface no longer appear among the scientific sources of the world-state dossier.

## Orientation in the atlas

Local search of the ten mechanisms by words and module path, independent of the API and of the scientific engine. The normalisation ignores case and accents; several words are combined. An accessible counter and a no-result message accompany the field. Without JavaScript, all the cards remain readable and no inactive control appears.

The links between dossiers are editorial reading relations, explicitly distinct from an integration of the components. The paths connect transformations, world, planning, branches, measurement, sources, certificates, learning and discovery, with access to the associated experiments.

## Reading the metrics and provenance

Nine continual learning indicators have bilingual explanations based on `checkpoint_metrics` and the memory counters of the backend. Retention and mean accuracy are explicitly identified as two names of the same computation in this protocol. Forgetting, backward transfer, ECE and the costs are not confused with a general measure of intelligence.

The predictive coverage and the credible coverage are explained from the interval computation in `core/source_calibration.py`. The number 96 is presented as aggregated cases of the summary, without equating it with the number of training examples. The source types have accessible FR/EN labels.

The full SHA-256 fingerprint of each artefact is expandable with the keyboard and on a touch screen; no information is available only on hover. The raw files and the scientific values remain unchanged.

## Discovery: questions and verdicts

The ten identifiers H1–H10 receive FR/EN titles and questions established from the preregistered protocol and from `core/active_benchmark.py`. The questions expose the comparison studied; they do not present a hypothesis as an acquired capability. For H10, the comparison concerns the complete system against the experts alone, without causally attributing the result to composition alone.

The verdicts remain read from `active_summary.json`, without recomputation or modification of the thresholds. The exact fields remain available in the panels. Filters display the four supported hypotheses, three refuted or three inconclusive; the overall total remains visible. Without JavaScript, the ten hypotheses remain accessible. The original protocol is published through an explicitly authorised file route, without exploring the private documents.

## Sources and contribution: accessibility and public status

The static inventory is searched by module path or symbol, with a counter and a no-result state. The matching groups and symbols are opened during the search; the previous open states are restored when it is cleared. The filter loads no file of the repository and no remote service. Without JavaScript, the native groups remain usable.

The open research section distinguishes consultable resources, publication of the code still to be finalised and collaboration ambition. Preparing an export does not mean that a public repository or a licence is available. The scientific contribution template exposes question, mechanism, context, baseline, budget, criteria, reproduction, artefacts and limits; no submission form or operational community channel is invented. The private details of the site and of the hub remain excluded.

## Lab: execution, errors and scope

Explicitly named command, character counter and accessible status. The examples and the input are blocked during execution to avoid double submissions; the input remains available after a failure. Network errors and unreadable responses have understandable FR/EN messages. The client timeout of 15 seconds stops the wait, without claiming to cancel an action already executed: the visitor is invited to check the state before sending an action again.

The log keeps at most twenty answers in the display, the most recent first; this does not reset the world. All the answers are inserted as text. No new API, no simulated scientific data and no extension of the linguistic capabilities are introduced. The engine, its validations and the simulation limits remain unchanged.

## Home page: favouring the evidence dossiers

The teaser of six mechanisms is replaced by three editorial cards: observation, learning and discovery. The counts are read from the JSON artefacts, without decorative numbers: aggregated cases, methods compared, supported/refuted/inconclusive hypotheses. The cards lead to the protocols and results; they do not announce a level of intelligence. The atlas of the ten mechanisms keeps its dedicated page with an explicit access from the home page, which reduces the repetition and the length of the initial path.

## Consistent review: orientation and accessibility

Footer organised in five sections and short paths to evidence, mechanisms, current state, reproduction and lab. The targets of the themes and languages are enlarged on mobile. The navigation distinguishes the active section from the current page; a parent section is no longer announced as the current page of a sub-dossier. The links of the research cards occupy their width on mobile.

Systematic check of the thirty pages in both languages: internal links, anchors, uniqueness of the identifiers and main title. The review of the template texts finds no French text without an English translation in the served pages. The changes to the site remain separate from the scientific sources and from the private architecture of the hub.

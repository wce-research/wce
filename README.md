# WCE — World Configuration Engine

World Configuration Engine (WCE) is an open research project exploring compact, compositional and verifiable AI
architectures based on explicit primitives, transformations, structured world models, adaptive learning and
scientific validation.

WCE is developed by **WCE Labs** — <https://wcelabs.com> — in the GitHub organization
[`wce-research`](https://github.com/wce-research). It is released under the [Apache License 2.0](LICENSE).

> **Status.** WCE is research in progress. It is not a general AI system, not a demonstrated replacement for large
> language models, and it makes no claim of superiority over existing architectures. The current scientific verdict
> of the active milestone is negative (see [Current bottlenecks](#current-bottlenecks)). This README separates what
> is implemented, what is experimental, and what is long-term vision.

## Contents

- [What is WCE?](#what-is-wce)
- [Core thesis](#core-thesis)
- [Implemented today](#implemented-today)
- [Experimental](#experimental)
- [Long-term vision](#long-term-vision)
- [Current architecture](#current-architecture)
- [Scientific methodology](#scientific-methodology)
- [Current bottlenecks](#current-bottlenecks)
- [Negative results](#negative-results)
- [Reproducibility](#reproducibility)
- [Running through the Makefile](#running-through-the-makefile)
- [Repository structure](#repository-structure)
- [Language policy](#language-policy)
- [Contributing, security, citation, license](#contributing-security-citation-license)

## What is WCE?

WCE studies whether structured behaviour can be built from a small set of explicit parts — primitives,
transformations, invariants and their compositions — that can be inspected, tested and refuted one at a time,
instead of being absorbed into a single opaque model.

The repository contains a symbolic world engine, a library of research prototypes, the experiments that tested
them, and the reports of those experiments, including the ones that failed.

## Core thesis

1. **Explicit structure.** A world is described by primitives and transformations whose composition is itself an
   object that can be searched, compared and verified.
2. **Validated transformation.** Any change to a model is proposed, evaluated on a valid criterion, then committed
   or rolled back. Nothing is destroyed on the strength of a surrogate objective.
3. **Measured boundaries.** A system should be able to say what it knows, what it does not, and what lies outside
   its model class.
4. **Science before claims.** Hypotheses are preregistered, negative results are published, and nothing becomes a
   default until a confirmatory evaluation supports it.

These are working hypotheses of the project. They are not established results.

## Implemented today

Code that exists, runs and is covered by tests:

- **Symbolic world engine** — commands, a world state, a small physics engine (gravity, support, collision with
  predefined laws), and answers: `language/`, `core/world.py`, `core/executor.py`, `physics/`, `app/session.py`,
  `app/repl.py`.
- **Boundary programs** — a small language of geometric predicates and Boolean combinators used as the model class
  for structure discovery: `core/boundary_program.py`.
- **Composition over observational equivalence classes** — a search over Boolean compositions of candidate
  primitives that works on equivalence classes of observations: `core/deep_compose.py`. With the *true* primitives
  it reconstructs compositions of up to six atoms reliably on development tasks.
- **Validated fitting** — parameter adjustment with propose / evaluate / commit-or-rollback:
  `core/deep_fit.py`, `core/deep_e2e.py`.
- **Experiment tooling** — deterministic task generators, paired comparisons with bootstrap confidence intervals,
  aggregation and report modules, and self-terminating cloud runs driven from the Makefile: `core/deep_*_exp.py`,
  `core/deep_*_report.py`, `wce_aws/`.
- **Documentation web application** — a lightweight local site that presents the research artifacts:
  `app/`. It does not load the PyTorch research modules.
- **Test suite** — more than 5,000 tests.

"Implemented" means the code runs and is tested. It does not mean the underlying scientific hypothesis is
confirmed.

## Experimental

Prototypes under evaluation. Each is tied to a protocol and a report; several have negative or inconclusive
results:

- primitive discovery from observations (`core/deep_primitives*.py`, `core/deep_partial.py`,
  `core/deep_fragments.py`, `core/deep_ownership.py`);
- recovery of the *set* of latent primitives and specialised high-precision estimators (`core/deep_lpr.py`,
  `core/deep_hp.py`);
- end-to-end composition with discovered primitives (`core/deep_e2e.py`);
- measurement-aware inference, source reliability and epistemic boundaries (`core/measurement.py`,
  `core/source_reliability.py`);
- plasticity, retention and continual learning (`core/plasticity.py`, `core/safe_plasticity.py`,
  `core/continuum.py`);
- active structural discovery and generative certificates (`core/active_structure.py`,
  `core/generative_certificate.py`);
- grounded language components (`core/language_core.py`, `core/grounded.py`).

The presence of a module proves neither its integration into a single model nor the confirmation of its
hypothesis.

## Long-term vision

Ambitions, not capabilities:

- a unified, trainable architecture that composes these mechanisms end to end;
- structured world models beyond the small synthetic worlds studied here, including perception;
- systems that propose, test and revise their own hypotheses under explicit scientific control;
- an open hub for alternative AI architectures: inspectable protocols, reproducible environments, forks and
  experimental comparisons.

None of this is demonstrated today. In particular WCE is not an AGI, not a general vision or video system, and not
an operational autonomous scientist.

**Open weights are not enough. Open the architecture.**

## Current architecture

```text
observations
   → primitive hypotheses              (discovery: the current bottleneck)
   → compatible primitive sets         (conditional utility, no destructive pruning)
   → composition on equivalence classes
   → validated finalisation            (propose → evaluate → commit / rollback)
   → structure + explicit uncertainty
```

The symbolic engine follows a separate, explicit path:

```text
language/command.py → core/executor.py → core/world.py
                                      → physics/engine.py
                                      → language/answer.py
```

A frozen reference pipeline is kept as the official baseline. New components are opt-in and are never plugged in
by default. Details: [`docs/architecture_wce.md`](docs/architecture_wce.md),
[`docs/architecture_validated_transformation.md`](docs/architecture_validated_transformation.md).

## Scientific methodology

- **Preregistration.** Hypotheses, support and refutation thresholds, primary method, secondary arms, metrics,
  compute budget, stopping criteria and the verdict rule are written in the report before any significant run.
- **Disjoint data.** Tuning keys and evaluation keys are separate and declared. Every quantitative comparison uses
  new development data.
- **Paired statistics.** Differences are paired by task and reported with 95 % bootstrap confidence intervals.
- **Post-hoc is labelled.** Anything added after results is marked `POST-HOC / EXPLORATORY`.
- **Fixed verdict vocabulary.** Each sub-milestone ends with one of a small set of verdicts decided by a rule
  written in advance.
- **Sealed sets.** Confirmatory sets are frozen before tuning, never inspected adaptively, and used once. The live
  sealed set is generated from a private salt that is not in this repository; only its fingerprints are public.
  See [`docs/sealed_sets.md`](docs/sealed_sets.md).
- **Nothing by default.** No component enters the reference pipeline before a confirmatory, multi-seed campaign.
  No such campaign has been run for the current milestone.

## Current bottlenecks

The active milestone is *Deep Compositional Structure Discovery*. Its sub-milestones narrowed the problem step by
step:

| Finding | Source |
|---|---|
| With the true atoms, composition on equivalence classes works up to six atoms. | [`rapport_deep_composition_oracle_atoms.md`](docs/rapport_deep_composition_oracle_atoms.md) |
| With discovered atoms, the end-to-end chain does not: verdict `END-TO-END STILL BLOCKING`. | [`rapport_end_to_end_discovered_atoms.md`](docs/rapport_end_to_end_discovered_atoms.md) |
| The probability of recovering *all* true primitives follows roughly `recall^n` and collapses with depth: verdict `LATENT PRIMITIVE RECOVERY STILL BLOCKING`. | [`rapport_latent_primitive_recovery.md`](docs/rapport_latent_primitive_recovery.md) |
| Specialised estimators raise recall and precision significantly, but not to the level needed; final recovery at 0.99 does not follow: verdict `HIGH-PRECISION PRIMITIVES STILL BLOCKING`. | [`rapport_high_precision_primitives.md`](docs/rapport_high_precision_primitives.md) |

In short: **individual primitive recall and parameter precision in composed, occluded scenes are not yet
sufficient for deep composition without an oracle.** At the densities of the primary regimes there is too little
information per visible boundary for the geometric estimators, and even when the atoms are recovered the required
parameter tolerances (about 0.02 on angles and radii) are rarely met jointly.

## Negative results

Negative results are part of the record. Examples from the latest reports:

- local single-primitive objectives move *away* from the true primitive; only the joint cell-floor objective is
  aligned with the function;
- refining a single primitive on labels degrades it under occlusion; median-of-resamples consensus does not help;
- a greedy direct search in parameter space fails beyond one atom;
- the out-of-class detector does not fire; the residual-morphology classifier is not informative;
- a previous neural proposer brought no gain and generalised poorly out of distribution;
- a reported total failure on one primitive type turned out to be a measurement defect, and is documented as
  such.

Each report lists its refuted hypotheses and its limits.

## Reproducibility

- Tasks are deterministic functions of their keys and seeds.
- Every report names its run identifiers; aggregated summaries and per-key result files are in `results/`.
- Tuned configurations are stored with the keys they were tuned on (`results/deep_configs/`).
- Record code version, dependency versions, platform, numerical precision, seeds, data, budget and artifact
  fingerprints. Two cross-platform numerical divergences are known in the suite and are not hidden by changing
  criteria.
- Operational logs and machine-specific files are not published; structured results are.

## Running through the Makefile

**Every user workflow — tests, experiments, cloud runs, aggregation, reports — goes through the Makefile.** Do not
run `python`, `pytest` or `aws` directly. The Makefile resolves its own location, so the repository works from any
directory it is cloned into.

```bash
make help                 # main targets
make test                 # full test suite (PyTorch required)
make check                # Python syntax check
make deep-hp-test         # tests of one sub-milestone
make deep-hp-report       # print the tables of its aggregated summary
make sealed-verify        # integrity of the sealed set (holder of the private salt only)
```

Research environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-research.txt
make test PYTHON=.venv/bin/python
```

Documentation web application (Docker Engine with the Compose plugin):

```bash
cp .env.example .env
make docker-up
make docker-ps
```

Significant experiments run on dedicated compute through the `deep-run-*` targets, with tagged, self-terminating
resources and short-lived credentials held outside the repository.

## Repository structure

| Path | Content |
|---|---|
| `core/` | world model, primitives, composition, fitting, experiments and report modules |
| `language/`, `physics/`, `logic/` | symbolic engine components |
| `app/` | session engine, REPL and documentation web application |
| `tests/` | test suite |
| `scripts/` | demonstrations and evaluation scripts, called by Makefile targets |
| `wce_aws/` | cloud execution tooling |
| `docs/` | architecture notes, preregistered hypotheses, scientific reports, sealed-set protocol |
| `results/` | structured results, aggregated summaries, tuned configurations |
| `tools/` | repository tooling |

Report file names keep their historical prefixes (`rapport_`, `hypotheses_`) so that existing references stay
valid; their content is in English.

## Language policy

English is the canonical language for WCE source code, scientific documentation, reports, protocols and public
artifacts. The WCE Labs website may provide translations into other languages, but the English repository remains
the scientific source of truth.

Earlier commits were written in French and remain in the history unchanged.

## Contributing, security, citation, license

- [CONTRIBUTING.md](CONTRIBUTING.md) — scientific rules and the Makefile rule.
- [SECURITY.md](SECURITY.md) — how to report a vulnerability.
- [CITATION.cff](CITATION.cff) — how to cite WCE.
- [LICENSE](LICENSE) — Apache License 2.0.

WCE Labs — <https://wcelabs.com> · GitHub — <https://github.com/wce-research>

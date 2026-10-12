# WCE — World Configuration Engine

WCE is an alternative experimental paradigm: primitives, transformations, invariants and compositions to explore the construction of complex structures and behaviours. This document describes the scientific mechanisms; the implementations of the site and of the future hub are not part of this publication.

## What exists now — the verifiable foundation

The symbolic engine follows an explicit path:

```text
language/command.py → core/executor.py → core/world.py
                                      → physics/engine.py
                                      → language/answer.py
```

`app/session.py` assembles a reduced world with a ball and a support. `app/repl.py` lets one manipulate this engine. The laws of gravity and collision are defined in advance: the demonstration does not discover them.

The repository also includes scientific prototypes, tests and protocols. Public artifacts describe predictive coverage, continual learning, active discovery and the subsequent deep compositional discovery programme through High-Precision Primitive Estimation. They include limits, refuted hypotheses and inconclusive results. They do not constitute a general comparison with LLMs or world models.

## What is being explored — the mechanisms

- Composition: `core/transformations.py`, `core/invariants.py`, `core/transform_graph.py`.
- World state: `core/world.py`, `core/model_state.py`.
- Reasoning and planning: `core/reasoning.py`, `core/planner.py`.
- Grounded language: `core/language_core.py`, `core/language_bridge.py`, `core/grounded.py`.
- Measurement and observation: `core/measurement.py`, `core/source_reliability.py`.
- Certificates and anchoring: `core/generative_certificate.py`, `core/truth_anchor.py`.
- Plasticity and retention: `core/plasticity.py`, `core/safe_plasticity.py`, `core/continuum.py`.
- Structural discovery: `core/active_structure.py`, `core/active_world.py`.
- Branches and snapshots: `core/branching.py`.

The presence of these modules proves neither their integration into a single model nor the confirmation of their hypotheses. Each result must be tied to its protocol and its budget.

## Reproduction

The symbolic engine is launched with `make repl` and verified with `make frontend-test`. All user workflows go through the Makefile. The neural experiments require PyTorch, their fixtures and their own protocols. A verification of the engine does not reproduce these experiments.

Record code version, dependency versions, platform, numerical precision, seeds, data, budget and artefact fingerprints. Two cross-platform numerical divergences remain known in the scientific suite; they are not hidden by changes of criteria.

## What WCE aims to become

The scientific code is already public under Apache-2.0, with CONTRIBUTING.md, SECURITY.md and CITATION.cff. An open hub for new AI architectures remains a vision: shared training interfaces, collaborative services and broader experimental comparisons still require implementation. No superiority over the dominant architectures is demonstrated here.

**Open weights are not enough. Open the architecture.**

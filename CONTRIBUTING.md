# Contributing to WCE

WCE — World Configuration Engine — is an open research project of WCE Labs. Contributions, replications and
critical reviews are welcome.

## Language policy

English is the canonical language for WCE source code, scientific documentation, reports, protocols and public
artifacts. The WCE Labs website may provide translations into other languages, but the English repository remains
the scientific source of truth.

Earlier commits were written in French and remain in the history unchanged.

## The Makefile rule

Every user-facing workflow — tests, experiments, AWS runs, aggregation, reports — goes through the `Makefile`.
Do not ask users to run `python`, `pytest`, `torchrun`, `aws` or a bare script directly. If a workflow has no
target, add one.

```
make help          # list the main targets
make test          # full test suite
```

## Scientific rules

These rules are part of the project, not style preferences.

- **Preregister before running.** Hypotheses, thresholds, primary method, secondary arms, metrics, compute budget,
  stopping criteria and the verdict rule are written in the report before any significant quantitative run.
- **Never choose thresholds after seeing results.** Analyses added afterwards are labelled
  `POST-HOC / EXPLORATORY`.
- **Report negative results.** A refuted hypothesis is reported as refuted, with its numbers.
- **Never tune on evaluation data.** Tuning keys and evaluation keys are disjoint and declared.
- **Sealed sets are never inspected adaptively.** Do not run, print, plot or summarise sealed tasks outside an
  explicitly authorised confirmatory evaluation.
- **Nothing is plugged into the frozen reference pipeline by default.** New components are opt-in until a
  confirmatory campaign supports them.
- **Paired comparisons with confidence intervals**, not single means.
- **State limits honestly.** No embellishment; scope claims to what was measured.

## Code

- Vectorised code (`torch` / `numpy`); avoid Python loops over observations.
- Match the surrounding style: naming, comment density, idiom.
- System modules must not import world-truth or evaluation modules; evaluation modules may see the truth to
  measure, never to decide.
- Add unit tests with every component; the full suite must pass with zero regression.

## Heavy computation

Local machines are for editing, quick unit tests and short demos. Significant experiments run on dedicated
compute through the Makefile targets, with tagged, self-terminating resources. Never commit credentials.

## Pull requests

1. Open an issue describing the question and, for experiments, the preregistration.
2. Keep changes focused; do not mix unrelated work.
3. Include the report section, the aggregated summary and the tests.
4. By contributing you agree that your contribution is licensed under the Apache License 2.0.

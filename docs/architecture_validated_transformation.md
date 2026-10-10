# Validated Transformation Principle

Status: architecture principle **supported experimentally for parametric fitting only**
(exploratory, dev partition). It is not applied automatically to the other WCE modules.

## Statement

Every internal transformation that may modify a state follows the sequence:

```
State → Transformation Proposal → Evaluation → Accept / Reject → Commit / Rollback
```

A transformation must not be able to silently destroy a state that is already better according to the available criteria.

Three obligations follow.

1. **Conservation.** The initial state remains available, intact, until the decision. The rollback returns that state; it
   does not rebuild it.
2. **Observable evaluation.** Acceptance rests on a criterion computable at the time of the decision, without hidden
   truth, with a deterministic rule for equality, numerical tolerance and non-finite values.
3. **Auditability.** Every decision leaves a trace: initial state, proposal, value of the criterion before and after,
   action, reason.

## What is implemented

| Element | Location | Role |
|---|---|---|
| `validated_fit`, `FitTransaction`, `decide` | `core/deep_fit.py` | transactional fitting: propose → evaluate → commit / rollback |
| `conditional_fit`, `OptimizationDecision`, `need_for_optimization` | `core/deep_optgate.py` | conditional gate: should a transformation be proposed at all? |
| `OptTrace` | `core/deep_optgate.py` | initial state, best state encountered, final state |
| `validated` parameter of `DeepSearch` | `core/deep_search.py` | enables the policy at the fitting step, and nowhere else |

Current acceptance rule: the profiled likelihood of the hard predictions on the observations must increase by
more than the tolerance (1e-9 nat, or the minimal gain of the gate). Equality: rollback. Non-finite proposal: rollback.
Non-finite initial state and finite proposal: commit.

## What the experiments support

All in dev, true atoms provided unless stated otherwise; details in `docs/rapport_validated_optimizer.md` and
`docs/rapport_objective_alignment.md`.

- Without validation, fitting destroys nearly nine good solutions out of ten.
- With validation on 128 never-seen tasks: recovery at the 0.99 threshold from 7% to 54% (+47 points, 95% CI
  [38, 56]), with no additional compute cost.
- The choice of criterion is decisive: the same mechanism, with the objective that the optimiser maximises as criterion,
  accepts 95% of the proposals and restores only 9 points.
- Deciding not to transform is an action in its own right: a gate that preserves the candidates whose residual
  looks like noise keeps 98% of the good solutions, against 71% for validation alone.

## What the principle does not guarantee

- It protects against degradation **according to the chosen criterion**. If that criterion is poorly aligned with what
  matters, a real degradation can be accepted (measured cases: a state better fitted to the noise of the observations,
  functionally worse).
- It does not make a transformation useful; it only prevents it from doing harm unseen.
- It has been evaluated neither in a confirmatory test nor on the sealed set.

## Possible extension (not carried out)

By proximity to what is measured:

1. structure simplification (it removes a necessary atom in about 1.6% of the tasks);
2. reinjection of the polished atoms into the synthesis;
3. promotion of fragments and primitives into the libraries;
4. belief updates after an active measurement.

Each extension would require its own observable criterion and its own controlled two-arm experiment. None has been
done: applying the principle everywhere before having measured it module by module would amount to assuming the result.

## Direction: self-evolving WCE

Architectural implication, **not a demonstrated capability**. A future modification by the system of its own
parameters, primitives, structures, weights, strategies or code should follow:

```
Proposal → Evaluation → Accept / Reject → Commit / Rollback
```

The reason is the same as for fitting: a system that transforms itself needs not to lose a better state.
Two lessons from this milestone would carry over: conserve the state before transforming it, and choose an acceptance
criterion that actually measures what one wants to preserve. The deeper the transformation (structure, code),
the harder the criterion is to define; that is the open problem, and nothing here solves it.

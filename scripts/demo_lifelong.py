"""Modèle lifelong : routage autonome, séquence de tâches, oubli, transfert, réparation."""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from core.algebra import Add, Equation, MathState, Multiply, Number, Symbol
from core.branching import shared_branch, branch_memory
from core.learning import Experience, generalize_linear_experience
from core.lifelong import TaskSpec, context_prototypes, make_task, run_lifelong, UNKNOWN
from core.planner import TransformationPlanner, algebra_candidate_generator
from core.reasoning import Goal


def fmt(v): return "  -  " if v is None else f"{v:.3f}"


def linear_skill():
    state = MathState(Equation(Add(Multiply(Number(2), Symbol("x")), Number(3)), Number(7)))
    goal = Goal("solve", "x")
    result = TransformationPlanner([algebra_candidate_generator], max_depth=5, max_nodes=400).search(state, goal)
    return generalize_linear_experience(Experience(state, goal, result, {}, result.cost, result.explored_nodes, result)), state, goal


def main():
    print("=== WCE LIFELONG DYNAMIC MODEL ===")
    run = run_lifelong()
    model, meta = run.model, run.controller
    print(f"\nBase:\ncontext encoder {model.encoder.parameter_count} parameters (12 -> 16 -> 6, frozen)")
    print("\nTask stream (no task label at inference):\n" + " ".join(t.name for t in run.tasks))
    for d in run.decisions:
        print(f"\n{d.task}: {d.novelty.kind} context ({d.novelty.reason})")
        verb = {"GROW": f"created {d.module}",
                "ADAPT": (f"reused {d.module}, shared slow weights (0 forgetting measured)" if d.kind == "SHARE"
                          else f"reused {d.module} + fast weights"),
                "REUSE": f"reused configuration {d.config}", "ROUTE": f"routed to {d.config}"}[d.action]
        print(f"  {d.action}: {verb} -> config {d.config}, accuracy {d.accuracy:.3f}")
        ranked = sorted(d.candidates.items(), key=lambda kv: -kv[1]["score"])
        print("  branches: " + ", ".join(f"{k}={v['score']:.3f} (p={v.get('weight', 0):.2f})" for k, v in ranked))
        if d.positive_transfer is not None:
            print(f"  positive transfer @10 steps: {d.positive_transfer:+.3f}, parameters saved vs new expert: {d.parameters_saved}")
        if d.negative_transfer:
            print(f"  negative transfer detected: {d.negative_transfer:.3f} (sharing rejected -> isolation)")
        if d.pruned: print(f"  pruned {d.module}: {d.pruned[0]} -> {d.pruned[1]} hidden units")

    metrics = meta.metrics(run.tasks)
    print(f"\nFinal capacity:\n{metrics['total_parameters']} parameters, {metrics['experts']} experts, "
          f"{metrics['configurations']} configurations")
    print(f"capacity trace: {run.capacity_trace}")
    print(f"\nAverage active params per input:\n{metrics['average_active_parameters']}")

    print("\nRouting test without labels (model.predict(x)):")
    for name in ("T1", "T4", "T7"):
        task = next(t for t in run.tasks if t.name == name)
        names, conf, _ = model.route(task.x_test[:32])
        top = max(set(names), key=names.count)
        acc = float((model.predict(task.x_test) == task.y_test).float().mean())
        print(f"\nsample {name}\nrouter -> {top} ({model.module_of(top)})\nconfidence {float(conf.mean()):.3f}"
              f"\naccuracy {acc:.3f}")

    print("\nForgetting matrix (rows: task, columns: after task k):")
    print("      " + " ".join(f"{t.name:>6}" for t in run.tasks))
    for name, row in run.matrix.items(): print(f"{name:>5} " + " ".join(f"{fmt(v):>6}" for v in row))
    print(f"\nforgetting per task: {run.forgetting()}\nmean interference: {run.interference()}")
    print(f"routing accuracy: {run.routing}")

    print("\nDamage test:")
    for allow in (True, False):
        trial = copy.deepcopy(meta)
        trial.damage("corrupt", "expert_2", seed=3)
        damaged = trial.health_check()
        print(f"\nexpert_2 corrupted (snapshots {'available' if allow else 'unavailable'})")
        print(f"degraded configurations: {', '.join(f'{c} {b:.2f}->{a:.2f}' for c, (b, a) in damaged.items())}")
        for report in trial.self_repair(allow_rollback=allow):
            print(f"MetaController: {report.config} diagnosis={report.diagnosis} candidates={report.candidates}")
            print(f"repair strategy = {report.selected}; recovered accuracy {report.before:.3f} -> {report.after:.3f}"
                  f" ({'success' if report.success else 'failed'})")
    trial = copy.deepcopy(meta); trial.damage("disconnect", "expert_1")
    reps = trial.self_repair()
    print(f"\nexpert_1 disconnected -> {[(r.config, r.selected, round(r.after, 3)) for r in reps]}")

    print("\nShared branch memory:")
    branch = shared_branch(model.state)
    before = branch_memory(branch, model.state)
    from core.model_state import GrowModule
    branch.apply(GrowModule(module="expert_1", units=8))
    after = branch_memory(branch, model.state)
    print(f"fresh branch owns {before['owned_bytes']} / {before['total_bytes']} bytes; "
          f"after GrowModule(expert_1) owns {after['owned_bytes']} bytes (copy-on-write)")
    print(f"mean shared ratio over {metrics['branch_count']} evaluated branches: {metrics['shared_state_memory_ratio']:.3f}")

    print("\nStructural evidence + neural evidence:")
    skill, state, goal = linear_skill()
    for _ in range(3): skill.record_success()
    trial = copy.deepcopy(meta)
    trial.model.memory.learn_skill(skill)
    protos = context_prototypes(n=8)
    t9 = make_task(TaskSpec("T9", 0, "and23_or6", UNKNOWN), protos, seed=900)
    d9 = trial.observe("T9", t9.x_train, t9.y_train, t9.x_val, t9.y_val, structural=(state, goal))
    print(f"T9 shares T1's input distribution; WCE skill detected: {d9.skill} -> {d9.action} {d9.config}")
    names, conf, _ = trial.model.route(t9.x_test)
    print(f"neural evidence only: confidence {float(conf.mean()):.2f} (ambiguous), "
          f"accuracy {float((trial.model.predict(t9.x_test) == t9.y_test).float().mean()):.3f}")
    names, conf, _ = trial.model.route(t9.x_test, structural=(state, goal))
    acc = float((trial.model.predict(t9.x_test, structural=(state, goal)) == t9.y_test).float().mean())
    print(f"neural + structural evidence: router -> {max(set(names), key=names.count)}, "
          f"confidence {float(conf.mean()):.2f}, accuracy {acc:.3f}")

    print("\nHomeostasis (target budget 400-850 parameters):")
    trial = copy.deepcopy(meta); trial.budget = (400, 850)
    from core.model_state import AdaptWeights
    cfg = trial.model.configs["cfg_T5"]
    trial.model.state.apply(GrowModule(module="expert_3", units=24))
    trial.model.state.apply(AdaptWeights(modules=("expert_3",), fast=False, output="expert_3", inputs=cfg.replay_x,
                                         targets=cfg.replay_y, steps=100, lr=0.02, local=False))
    print(f"expert_3 grown to {trial.model.state.modules['expert_3'].hidden} units -> total {trial.model.state.total_parameters()}")
    print(trial.homeostasis())
    print(f"T5 accuracy after homeostasis: {trial.model.config_accuracy('cfg_T5', cfg.probe_x, cfg.probe_y):.3f}")
    trial.budget = (400, 600)
    print(f"tighter budget 400-600: {trial.homeostasis()}")

    print("\nMetrics:")
    for key in ("module_reuse_rate", "module_creation_rate", "prune_rate", "novelty_accuracy", "routing_confidence",
                "branch_count", "candidate_evaluations", "fast_parameters", "modules_created", "units_pruned"):
        print(f"  {key}: {metrics[key]}")
    print(f"  mean routing accuracy: {sum(run.routing.values()) / len(run.routing):.3f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()

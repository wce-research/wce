"""Compact self-transforming neural model, governed by WCE."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from core.algebra import Add, Equation, MathState, Multiply, Number, Symbol
from core.learning import Experience, generalize_linear_experience
from core.model_state import INPUT, AddModule, CreateConnection, ModelState, RerouteConnection
from core.planner import TransformationPlanner, algebra_candidate_generator
from core.plasticity import (
    LoadContext, MemorySystem, PlasticityController, build_base_model, pretrain, run_learning_loop,
    save_context, task_a_labels, task_b_labels, task_split,
)
from core.reasoning import Goal


def fmt(n): return f"{n:,}"


def main():
    torch.manual_seed(0)
    xa, ya, va, vya = task_split(task_a_labels)
    xb, yb, vb, vyb = task_split(task_b_labels)
    model = build_base_model()
    print("=== WCE DYNAMIC MODEL ===")
    pretrain(model, xa, ya)
    save_context(model, "task_a")
    model.snapshot("N_task_a")
    m = model.metrics()
    print(f"\nBase model:\nparameters: {fmt(m['total_parameters'])}\nactive: {fmt(m['active_parameters'])}"
          f"\ncapacity: {m['capacity']} hidden units\nmodules: {sorted(model.modules)} (inactive: expert_b)")
    acc_a = model.accuracy(va, vya)
    print(f"\nTask A:\naccuracy: {acc_a:.3f}")

    print(f"\nTask B arrives.\naccuracy with current model: {model.accuracy(vb, vyb):.3f}")
    base_node = model.transform_graph.current.name
    report = run_learning_loop(model, "task_b", xb, yb, vb, vyb, PlasticityController())
    print("\nWeakness detected." if report.weakness else "\nNo weakness.")
    print("\nCandidate transformations:")
    for i, c in enumerate(report.candidates, 1): print(f"{i}. {c}")
    print("\nBranches evaluated (classical weighted configurations):")
    for label, info in report.branches.items():
        print(f"\n{label}:\n  score {info['score']}  p={report.distribution[label]}  accuracy {info['accuracy']:.3f}"
              f"  interference {info['interference']}  active {fmt(info['active_parameters'])}")
    print(f"\nSelected:\n{report.selected_description}")
    new = report.model
    print(f"\nActive parameters:\n{fmt(new.active_parameters())}")
    print(f"\nTask B accuracy:\n{report.after_accuracy:.3f}")
    print(f"\nConsolidating...\ncommitted fast weights of: {', '.join(report.consolidated)}")
    print("\nPruning unused capacity...")
    for module, (before, after) in report.pruned.items(): print(f"{module}: {before} -> {after} hidden units")
    print(f"capacity: before {report.capacity['before']} / max {report.capacity['max']} / final {report.capacity['final']}")
    m = new.metrics()
    print(f"\nFinal parameters:\n{fmt(m['total_parameters'])}\nactive:\n{fmt(m['active_parameters'])}")

    new.apply(LoadContext(context="task_a"))
    back_a = new.accuracy(va, vya)
    print(f"\nReturn Task A (routing: expert_b off):\naccuracy {back_a:.3f} -> "
          + ("accuracy preserved" if back_a >= acc_a - 0.01 else "degraded"))
    print(f"active parameters in A context: {fmt(new.active_parameters())}")
    new.apply(LoadContext(context="task_b"))
    print(f"Switch back to Task B: accuracy {new.accuracy(vb, vyb):.3f}")

    print("\nRollback available:\n" + ("YES" if new.snapshots else "NO"))
    new.restore("N_task_a")
    print(f"restore(N_task_a): modules {sorted(new.modules)}, capacity {new.capacity}, "
          f"Task A {new.accuracy(va, vya):.3f}")
    path = new.history()
    start = path.index(base_node)
    print("\nTransformGraph (self-transformations after base):")
    for name in path[start:]:
        node = new.transform_graph.nodes[name]
        print(f"  {name}: {node.transformations[-1].name if node.transformations else 'base'}")

    print("\n---\n\nReconnection (same modules, other configuration):")
    net = ModelState(4, output_module="C")
    for spec in (AddModule(module="A", input_dim=4, output_dim=4), AddModule(module="B", input_dim=4, output_dim=4),
                 AddModule(module="C", input_dim=4, output_dim=2, activation=False),
                 CreateConnection(source=INPUT, target="A"), CreateConnection(source="A", target="B"),
                 CreateConnection(source="B", target="C")):
        net.apply(spec)
    save_context(net, "task1")
    print(f"task1 routes: {sorted(net.connectivity)}")
    net.apply(RerouteConnection(source="A", old_target="B", new_target="C"))
    save_context(net, "task2")
    print(f"task2 routes: {sorted(net.connectivity)}")
    net.apply(LoadContext(context="task1"))
    print(f"back to task1: {sorted(net.connectivity)}")

    print("\n---\n\nStructural learning + neural plasticity:")
    state = MathState(Equation(Add(Multiply(Number(2), Symbol("x")), Number(3)), Number(7)))
    goal = Goal("solve", "x")
    planner = TransformationPlanner([algebra_candidate_generator], max_depth=5, max_nodes=400)
    result = planner.search(state, goal)
    experience = Experience(state, goal, result, {}, result.cost, result.explored_nodes, result)
    skill = generalize_linear_experience(experience)
    memory = MemorySystem(new, experiences=[experience])
    memory.learn_skill(skill)
    controller = PlasticityController()
    for _ in range(3): skill.record_success()  # consolidated over 3 instances (cf. demo-learning)
    for acc in (None, 0.7, 0.4):
        decision, why = controller.decide_for_skill(skill, acc)
        print(f"skill={skill.name} detector_accuracy={acc}: {decision} ({why})")
    print(f"\nMemory levels: {memory.describe()}")
    print(f"\nMetrics (after restore N_task_a): {new.metrics()}")


if __name__ == "__main__":
    main()

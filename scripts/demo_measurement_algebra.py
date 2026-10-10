"""Demo: hidden world -> hypotheses -> candidate measurements -> information gain / cost -> chosen measurement ->
posterior -> answer (right to assert). Example: two cubes compatible with “the cube on the table”."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from core.measurement import (  # noqa: E402
    CompatibleStateSet, LatentWorldState, MeasurementContext, MeasurementOperator, MeasurementPlanner, NoiseModel,
    Observable, check_invariants, decision_value, information_gain, new_graph,
)

SUP = ["table", "shelf"]; COL = ["red", "blue"]; OWN = ["marie", "paul"]


def main():
    g = new_graph("demo")
    world = LatentWorldState({("e1", "loc"): "shelf", ("e2", "loc"): "table", ("e1", "color"): "red", ("e2", "color"): "blue",
                              ("e1", "owner"): "marie", ("e2", "owner"): "marie"}, seed=3)
    print("LATENT world (hidden from the agent): two cubes e1, e2; the question is “what colour is the cube on the table?”")
    dom = {(e, a): d for e in ("e1", "e2") for a, d in (("loc", SUP), ("color", COL), ("owner", OWN))}
    S = CompatibleStateSet.product(dom, graph=g)
    one = (S.column(("e1", "loc")) == 0) ^ (S.column(("e2", "loc")) == 0)          # presupposition: exactly one cube on the table
    S = S.restrict(one, "« le cube sur la table » : exactement un")
    ref = (S.column(("e2", "loc")) == 0).long()                                      # referent: 0 -> e1, 1 -> e2
    ans = torch.where(ref == 0, S.column(("e1", "color")), S.column(("e2", "color")))
    S.states = torch.cat([S.states, ref.view(-1, 1), ans.view(-1, 1)], 1)
    S.vars += [("ref", "id"), ("ans", "color")]; S.index = {v: i for i, v in enumerate(S.vars)}
    S.domains[("ref", "id")] = ["e1", "e2"]; S.domains[("ans", "color")] = COL
    print(f"\nHypotheses: {len(S)} compatible worlds; P(referent) = {S.marginal(('ref', 'id'))}; H(S) = {S.entropy():.3f} nats")
    menu = {
        "couleur e1": MeasurementOperator(Observable("color", "e1", "color", tuple(COL)), MeasurementContext(sensor="color_cam", noise=NoiseModel("confusion", 0.97), cost=0.15)),
        "position précise e1": MeasurementOperator(Observable("loc", "e1", "loc", tuple(SUP)), MeasurementContext(sensor="lidar", noise=NoiseModel("confusion", 0.97), cost=0.5)),
        "couleur e2": MeasurementOperator(Observable("color", "e2", "color", tuple(COL)), MeasurementContext(sensor="color_cam", noise=NoiseModel("confusion", 0.97), cost=0.15)),
        "propriétaire e1": MeasurementOperator(Observable("owner", "e1", "owner", tuple(OWN)), MeasurementContext(sensor="registre", noise=NoiseModel("confusion", 0.9), cost=0.3)),
    }
    target = ("ans", "color")
    print("\nCandidate measurements (target: colour of the cube on the table):")
    for name, op in menu.items():
        ig = information_gain(S, op, target); igs = information_gain(S, op)
        vd = decision_value(S, op, target, 20.0, 8.0)
        print(f"  {name:22s} IG(target) = {ig:.3f}  IG(S) = {igs:.3f}  decision value = {vd:.3f}  cost = {op.context.cost}")
    planner = MeasurementPlanner(graph=g); planner.decision_costs = (20.0, 8.0)
    step = 0
    while True:
        c = planner.choose(S, list(menu.values()), target)
        if c is None: print("  -> stop: sufficient confidence or no worthwhile measurement left"); break
        rec = world.measure(c.operator, c.operator.context.sensor, g)
        S = S.condition(rec); step += 1
        print(f"\nCycle {step}: chosen measurement {c.operator.name} (IG {c.information_gain:.3f}, utility {c.utility:.3f}) -> outcome “{rec.outcome.value}”")
        print(f"  P(referent) = { {k: round(v, 3) for k, v in S.marginal(('ref', 'id')).items()} }; P(colour) = { {k: round(v, 3) for k, v in S.marginal(target).items()} }")
        print("  invariants :", {k.name: v for k, v in check_invariants(c.operator.distribution(rec.outcome.value), rec, latent=world).items()})
    v, p = max(S.marginal(target).items(), key=lambda kv: kv[1])
    print(f"\nAnswer: {v} (P = {p:.3f}) -> {'ASSERTED' if p >= 0.9 else 'WITHHELD'}; truth: {world.peek(('e2', 'color'))} (evaluation read)")
    print("TransformGraph trace:", [n for n in g.current.get_path_names()])


if __name__ == "__main__":
    main()

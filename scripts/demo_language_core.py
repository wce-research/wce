"""Compact WCE Language Model: integrated demonstration."""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.language_core import (
    SYSTEMS, CorruptCore, build_core, char_vs_word, experiment_matrix, make_scene, morphology_report,
    parameter_scaling, parse, run_stream,
)
from core.guided_search import distillation_experiment, guided_search_experiment
from core.trajectory import calibrate, evaluation_words, monitor_factory


def show(lc, text, world=None, title=None):
    r = lc.process(text, world or make_scene())
    if title: print(f"\n{title}")
    print(f"  « {text} »")
    interp = r.interpretation
    if interp is not None and interp.parsed.kind == "command":
        goal = "; ".join(f"{g.predicate}{g.target}" + (f" [{g.condition}]" if g.condition is not None else "") for g in interp.goals if g) \
               or (interp.program.name if interp.program is not None else "-")
        print(f"  interpretation: {interp.label} -> {goal}")
    if len(r.candidates) > 1:
        print("  candidates: " + " | ".join(f"{c['label']} w={c['weight']} support={c['support']}"
                                           + (f" ✗{c['failures']}" if c['failures'] else "") for c in r.candidates[:4]))
    if r.risk is not None:
        g = f", geometry={r.risk.geometric_level} ({r.risk.geometric_risk:.2f})" if r.risk.geometric_risk is not None else ""
        print(f"  risk: status={r.status}, neural={r.risk.neural_confidence:.2f}, structural={r.risk.structural_support:.2f}, "
              f"contradictions={r.risk.contradiction_count}, disagreement={r.risk.branch_disagreement}{g}")
    print(f"  decision: {r.decision} -> {r.answer if r.decision.startswith('ACCEPT') else r.reason}")
    for step in r.trace[1:]: print(f"    {step}")
    print(f"  memory: {sorted(r.memory)} | verifications: {r.checks}")
    return r


def main(quick=False):
    print("=== WCE COMPACT LANGUAGE CORE ===")
    lc = build_core(**SYSTEMS["6_full_self_correction"])
    stream = run_stream(lc)
    lc.attach_monitor(*calibrate(lc, evaluation_words()))
    print(f"\nCharacter neural core: {lc.core.parameter_count} parameters; grounded lexicon: {len(lc.lexicon.grounded)} forms")
    print(f"Learned morphological rules: {sorted(lc.lexicon.morphology.rules)}")
    learned = [r for r in stream if r["type"] == "learn"]
    print("Stream: " + ", ".join(f"{r['text']}→{r['learned']}" for r in learned))

    show(lc, "le cube tombe", title="1. Known sentence -> Goal -> planner")
    show(lc, "le bloc chute", title="2. Learned synonym (« le cube chute ») -> transfer to the block")
    show(lc, "retourne puis décale abcde", title="3. Never-seen composition -> zero-shot program")
    show(lc, "le bloc montait", title="3b. Morphology: -ait→-e learned on tomber/retourner")
    show(lc, "le bloc zorphe", title="4. Unknown word -> abstention")
    show(lc, "tourne abcde", title="5. Ambiguous sentence (« tourne » observed as left AND right rotation) -> branches")
    show(lc, "range dcbea", title="5b. Ambiguity resolved by the domain (range = sort / put down)")
    geo = build_core(**SYSTEMS["5_neural_wce_geometric"]); run_stream(geo)
    geo.attach_monitor(*calibrate(geo, evaluation_words()))
    plain = build_core(**SYSTEMS["4_neural_wce"]); run_stream(plain)
    corrupt = lambda sys_: (lambda c: (CorruptCore(scale=1.5, seed=3).apply(c.core), c)[1])(copy.deepcopy(sys_))
    print("\n6. Corrupted neural core (controlled damage): « le bloc monte »")
    r4 = corrupt(plain).process("le bloc monte", make_scene())
    print(f"  neural+WCE (1 verification): {r4.decision} -> {r4.output} ({r4.reason})")
    show(corrupt(geo), "le bloc monte", title="  neural+WCE+geometry: geometrically suspect branch -> REROUTE")
    show(lc, "le bloc tombe du support", title="7. WCE contradiction -> rejection")
    show(lc, "monte la balle à -2 mètres")
    show(lc, "la table est en bois", title="8. Simulated external verification")
    show(lc, "la balle est en verre"); show(lc, "le cube est magique"); show(lc, "la balle est rouge")
    print("\n9. Self-correction (same corrupted core): « retourne puis décale abcde »")
    before = corrupt(plain).process("retourne puis décale abcde", make_scene())
    after = corrupt(lc).process("retourne puis décale abcde", make_scene())
    print(f"  without self-correction: {before.decision} -> {before.output} ({before.reason})")
    print(f"  with self-correction: {after.decision} -> {after.output}")
    for step in after.trace: print(f"    {step}")

    print("\n10. Metrics")
    print("  Language stream (no task label):")
    cats = {}
    for r in stream:
        if r["type"] == "query": cats.setdefault(r["category"], []).append(r["correct"])
    for c, v in cats.items(): print(f"    {c:<24} {sum(v)}/{len(v)}")
    print(f"  Characters vs words: {char_vs_word()}")
    m = morphology_report(lc); m.pop("rows")
    print(f"  Morphology: correct {m['correct_generalizations']}, over-generalizations {m['over_generalizations']}, abstentions {m['abstentions']}")
    if quick: return
    print("\n  Experimental matrix (same stream, same evaluation; stress = corrupted core):")
    matrix = experiment_matrix(monitor_factory=monitor_factory(evaluation_words()))
    print(f"    {'system':<26} success false-conf wrong-rejects abst.  corr.  verif.  routing  nodes  params | stress: success false-conf")
    for name, r in matrix.items():
        s = r["corrupted_core"]
        print(f"    {name:<26} {r['task_success']:.3f}  {r['false_confident_rate']:.3f}      {r['wrong_rejections']:<13} {r['abstention_rate']:.3f}  "
              f"{r['correction_rate']:.3f}  {r['verification_cost']:<6}  {r['routing_cost']:<7}  {r['search_nodes']:<5}  {r['parameters_total']:<6} | "
              f"{s['task_success']:.3f} {s['false_confident_rate']:.3f}")
    print("\n  Scale of the neural core:")
    for size, r in parameter_scaling(monitor_factory=monitor_factory(evaluation_words())).items():
        print(f"    {size:<7} {r['parameters']:>6} params | neural unseen forms {r['neural_accuracy_unseen_forms']:.3f} | "
              f"neural only {r['neural_only_success']:.3f} | neural+WCE {r['neural_wce_success']:.3f} | WCE interventions {r['wce_interventions']}")
    print("\n  Guided program search (depth 3):")
    g = guided_search_experiment(tasks=30); g.pop("rows")
    print(f"    unguided {g['unguided_nodes']} nodes (success {g['unguided_success']}) | guided {g['guided_nodes']} nodes "
          f"(success {g['guided_success']}) | proposer {g['proposer_parameters']} params | depth {g['max_depth_reached_guided']}")
    print("    language « retourne puis décale »: program given by the sentence -> 1 verification")
    from core.symbolic_lifelong import run_stream as symbolic_stream, small_segments
    system = symbolic_stream(small_segments()).system
    d = distillation_experiment(system, sorted({s.family for s in small_segments()}))
    print(f"\n  Distillation WCE -> neural router ({d['examples']} examples labelled by WCE):")
    print(f"    neural accuracy {d['before']['neural']} -> {d['after']['neural']} | hybrid verifications "
          f"{d['before']['hybrid_evaluations']} -> {d['after']['hybrid_evaluations']} (hybrid accuracy {d['after']['hybrid']})")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)

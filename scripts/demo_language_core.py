"""Compact WCE Language Model : démonstration intégrée."""
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
        print(f"  interprétation: {interp.label} -> {goal}")
    if len(r.candidates) > 1:
        print("  candidats: " + " | ".join(f"{c['label']} w={c['weight']} support={c['support']}"
                                           + (f" ✗{c['failures']}" if c['failures'] else "") for c in r.candidates[:4]))
    if r.risk is not None:
        g = f", géométrie={r.risk.geometric_level} ({r.risk.geometric_risk:.2f})" if r.risk.geometric_risk is not None else ""
        print(f"  risque: statut={r.status}, neuronal={r.risk.neural_confidence:.2f}, structurel={r.risk.structural_support:.2f}, "
              f"contradictions={r.risk.contradiction_count}, désaccord={r.risk.branch_disagreement}{g}")
    print(f"  décision: {r.decision} -> {r.answer if r.decision.startswith('ACCEPT') else r.reason}")
    for step in r.trace[1:]: print(f"    {step}")
    print(f"  mémoire: {sorted(r.memory)} | vérifications: {r.checks}")
    return r


def main(quick=False):
    print("=== WCE COMPACT LANGUAGE CORE ===")
    lc = build_core(**SYSTEMS["6_full_self_correction"])
    stream = run_stream(lc)
    lc.attach_monitor(*calibrate(lc, evaluation_words()))
    print(f"\nCœur neuronal caractères : {lc.core.parameter_count} paramètres ; lexique ancré : {len(lc.lexicon.grounded)} formes")
    print(f"Règles morphologiques apprises : {sorted(lc.lexicon.morphology.rules)}")
    learned = [r for r in stream if r["type"] == "learn"]
    print("Flux : " + ", ".join(f"{r['text']}→{r['learned']}" for r in learned))

    show(lc, "le cube tombe", title="1. Phrase connue -> Goal -> planner")
    show(lc, "le bloc chute", title="2. Synonyme appris (« le cube chute ») -> transfert au bloc")
    show(lc, "retourne puis décale abcde", title="3. Composition jamais vue -> programme zéro-shot")
    show(lc, "le bloc montait", title="3b. Morphologie : -ait→-e appris sur tomber/retourner")
    show(lc, "le bloc zorphe", title="4. Mot inconnu -> abstention")
    show(lc, "tourne abcde", title="5. Phrase ambiguë (« tourne » observé en rotation gauche ET droite) -> branches")
    show(lc, "range dcbea", title="5b. Ambiguïté levée par le domaine (range = trier / poser)")
    geo = build_core(**SYSTEMS["5_neural_wce_geometric"]); run_stream(geo)
    geo.attach_monitor(*calibrate(geo, evaluation_words()))
    plain = build_core(**SYSTEMS["4_neural_wce"]); run_stream(plain)
    corrupt = lambda sys_: (lambda c: (CorruptCore(scale=1.5, seed=3).apply(c.core), c)[1])(copy.deepcopy(sys_))
    print("\n6. Cœur neuronal corrompu (dommage contrôlé) : « le bloc monte »")
    r4 = corrupt(plain).process("le bloc monte", make_scene())
    print(f"  neuronal+WCE (1 vérification) : {r4.decision} -> {r4.output} ({r4.reason})")
    show(corrupt(geo), "le bloc monte", title="  neuronal+WCE+géométrie : branche géométriquement suspecte -> REROUTE")
    show(lc, "le bloc tombe du support", title="7. Contradiction WCE -> rejet")
    show(lc, "monte la balle à -2 mètres")
    show(lc, "la table est en bois", title="8. Vérification externe simulée")
    show(lc, "la balle est en verre"); show(lc, "le cube est magique"); show(lc, "la balle est rouge")
    print("\n9. Auto-correction (même cœur corrompu) : « retourne puis décale abcde »")
    before = corrupt(plain).process("retourne puis décale abcde", make_scene())
    after = corrupt(lc).process("retourne puis décale abcde", make_scene())
    print(f"  sans auto-correction : {before.decision} -> {before.output} ({before.reason})")
    print(f"  avec auto-correction : {after.decision} -> {after.output}")
    for step in after.trace: print(f"    {step}")

    print("\n10. Métriques")
    print("  Flux de langage (sans étiquette de tâche) :")
    cats = {}
    for r in stream:
        if r["type"] == "query": cats.setdefault(r["category"], []).append(r["correct"])
    for c, v in cats.items(): print(f"    {c:<24} {sum(v)}/{len(v)}")
    print(f"  Caractères vs mots : {char_vs_word()}")
    m = morphology_report(lc); m.pop("rows")
    print(f"  Morphologie : correctes {m['correct_generalizations']}, sur-généralisations {m['over_generalizations']}, abstentions {m['abstentions']}")
    if quick: return
    print("\n  Matrice expérimentale (même flux, même évaluation ; stress = cœur corrompu) :")
    matrix = experiment_matrix(monitor_factory=monitor_factory(evaluation_words()))
    print(f"    {'système':<26} succès  faux-sûrs  rejets-à-tort abst.  corr.  vérif.  routage  nœuds  params | stress: succès faux-sûrs")
    for name, r in matrix.items():
        s = r["corrupted_core"]
        print(f"    {name:<26} {r['task_success']:.3f}  {r['false_confident_rate']:.3f}      {r['wrong_rejections']:<13} {r['abstention_rate']:.3f}  "
              f"{r['correction_rate']:.3f}  {r['verification_cost']:<6}  {r['routing_cost']:<7}  {r['search_nodes']:<5}  {r['parameters_total']:<6} | "
              f"{s['task_success']:.3f} {s['false_confident_rate']:.3f}")
    print("\n  Échelle du cœur neuronal :")
    for size, r in parameter_scaling(monitor_factory=monitor_factory(evaluation_words())).items():
        print(f"    {size:<7} {r['parameters']:>6} params | neuronal formes inédites {r['neural_accuracy_unseen_forms']:.3f} | "
              f"neuronal seul {r['neural_only_success']:.3f} | neuronal+WCE {r['neural_wce_success']:.3f} | interventions WCE {r['wce_interventions']}")
    print("\n  Recherche de programmes guidée (profondeur 3) :")
    g = guided_search_experiment(tasks=30); g.pop("rows")
    print(f"    non guidée {g['unguided_nodes']} nœuds (succès {g['unguided_success']}) | guidée {g['guided_nodes']} nœuds "
          f"(succès {g['guided_success']}) | proposeur {g['proposer_parameters']} params | profondeur {g['max_depth_reached_guided']}")
    print("    langage « retourne puis décale » : programme donné par la phrase -> 1 vérification")
    from core.symbolic_lifelong import run_stream as symbolic_stream, small_segments
    system = symbolic_stream(small_segments()).system
    d = distillation_experiment(system, sorted({s.family for s in small_segments()}))
    print(f"\n  Distillation WCE -> routeur neuronal ({d['examples']} exemples étiquetés par WCE) :")
    print(f"    précision neuronale {d['before']['neural']} -> {d['after']['neural']} | vérifications hybrides "
          f"{d['before']['hybrid_evaluations']} -> {d['after']['hybrid_evaluations']} (précision hybride {d['after']['hybrid']})")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)

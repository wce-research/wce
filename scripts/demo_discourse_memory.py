"""Mémoire de discours : identité à travers les phrases, révisions, graphe d'identité."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.discourse import DiscourseResolver
from core.discourse_benchmark import lexicon_agent, long_discourse, run_benchmark, by_key
from core.reference_world import ClarificationSimulator, Episode, REntity, Turn, given_types


def show(res, mentions):
    for mm in mentions:
        cands = {e: round(p, 2) for e, p in res.dists.get(mm.referent, {}).items()} if mm.referent else {}
        print(f"   mention {mm.id} « {mm.surface} » ({mm.kind}{', ' + mm.definiteness if mm.definiteness else ''}) -> référent {mm.referent}"
              f" ; candidats {cands} ; décision {res.mention_resolution(mm.id)}")


def main():
    print("=== WCE DISCOURSE MEMORY ===")
    lex = lexicon_agent(0)
    ents = {"e1": REntity("e1", "cube", "red", ("on", "table")), "e2": REntity("e2", "cube", "blue", ("on", "table"))}
    given = given_types(ents) + [(("attr", "e1"), "red"), (("attr", "e2"), "blue")]
    print("\nMonde connu de l'agent : deux cubes (e1 rouge, e2 bleu) ; positions non observées")
    res = DiscourseResolver(lex, given)
    for text in ("Un cube est sur la table.", "Le cube rouge tombe.", "L'autre reste sur la table.", "Prends-le."):
        print(f"\n« {text} »")
        ms = res.process(text)
        show(res, ms)
        refs = res.state.entities_mentioned
        print(f"   contraintes : " + "; ".join(f"{r.id}: type={r.type} attrs={r.attrs} états={r.states} distinct_de={sorted(r.different_from)}" for r in refs.values()))
    print(f"\nRévisions de croyance ({len(res.revisions)}) :")
    for rv in res.revisions: print(f"   {rv.referent}: {rv.before} ({rv.confidence_before:.2f}) -> {rv.after} ({rv.confidence_after:.2f}) [{rv.reason}]")
    print("\nGraphe d'identité final :")
    for kind in ("REFERS_TO", "POSSIBLY_SAME_AS", "SAME_AS", "DIFFERENT_FROM", "CONTRADICTS"):
        edges = [(a, b, round(c, 2)) for k, a, b, c, _ in res.graph.edges if k == kind]
        if edges: print(f"   {kind}: {edges}")
    print(f"   branches maintenues pour d1 : {[(a.label, round(a.weight, 2)) for a in res.branches('d1')]}")
    print(f"   invariants violés : {res.graph.invariant_violations(res.state, res.resolution)}")
    print(f"\nMémoire consolidée (sans texte brut) : {res.consolidated()}")
    print(f"Taille : {res.memory_size()}")

    print("\n--- Révision en arrière (provenance conservée) ---")
    res4 = DiscourseResolver(lex, given)
    for text in ("Un cube est sur la table.", "Il est rouge."):
        ms = res4.process(text); print(f"« {text} »"); show(res4, ms)
    for rv in res4.revisions:
        print(f"   révision {rv.referent}: {rv.before} ({rv.confidence_before:.2f}) -> {rv.after} ({rv.confidence_after:.2f}) [{rv.reason}]")

    print("\n--- Clarification ---")
    res2 = DiscourseResolver(lex, given)
    ms = res2.process("Prends le cube.")
    q = res2.clarification_question(ms[0].id)
    ep = Episode("physical", "clarification", ents, given, [])
    sim = ClarificationSimulator(ep); answer = sim.answer(q, "e2")
    print(f"« Prends le cube. » -> {res2.mention_resolution(ms[0].id)} ; question : « {q} » ; réponse simulée : « {answer} »")
    res2.apply_clarification(ms[0].id, answer)
    print(f"après clarification -> {res2.mention_resolution(ms[0].id)}")

    print("\n--- Présupposition ---")
    res3 = DiscourseResolver(lex, given)
    ms = res3.process("Pourquoi le cube vert est-il là ?")
    print(f"« Pourquoi le cube vert est-il là ? » -> {res3.mention_resolution(ms[0].id)} ; {res3.presupposition(ms[0].id)}")

    m, rows = run_benchmark()
    print(f"\nBenchmark de référence ({m['episodes']} épisodes, {m['mentions']} mentions, 3 domaines) : {m}")
    for dom, v in by_key(rows, "domain").items(): print(f"   {dom}: précision {v['precision']} couverture {v['coverage']} engagements faux {v['false_identity_commitment']}")
    for nz in (0.1, 0.3):
        mn, _ = run_benchmark(noise=nz)
        print(f"   bruit {nz:.0%} sur les faits donnés : précision {mn['precision']} couverture {mn['coverage']} engagements faux {mn['false_identity_commitment']}")
    print(f"\nDiscours longs : {long_discourse()}")


if __name__ == "__main__":
    main()

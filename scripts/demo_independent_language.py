"""Environnement indépendant : vérité cachée, preuves partielles, croyances, décisions, puis révélation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.independent_benchmark import make_agent, prepare, source_report
from core.independent_world import IndependentWorldVerifier


def main():
    print("=== WCE INDEPENDENT LANGUAGE ENVIRONMENT ===")
    setup = prepare()
    agent = make_agent("G_full", setup)
    print(f"\nAgent : lexique induit {len(agent.lex.lexicon())} formes ; prépositions apprises "
          f"{ {t: m for t, m in agent.lex.lexicon().items() if m.startswith('rel:')} }")
    print(f"Connecteurs appris : { {t: c.most_common(1)[0][0] for t, c in agent.grammar.connectors.items()} } ; "
          f"unités : { {u: agent.lex.unit_scale(u) for u in ('m', 'cm', 'mètres')} }")
    print(f"Fiabilité estimée vs vraie : {source_report(agent)}")
    x = setup.splits["test"][0]
    world, obs = x["world"], x["observations"]
    agent.new_world(); agent.ingest(obs)
    print("\nWorldTruth : [caché à l'agent]")
    print(f"\nObserved evidence (partiel, bruité) : {len(obs)} observations, dont {sum(1 for o in obs if o.text)} phrases")
    for o in [o for o in obs if o.text][:8]: print(f"  {o.source:<11} « {o.text} »")
    for o in [o for o in obs if not o.text and not o.implied and o.key[0] != 'type'][:4]: print(f"  {o.source:<11} {o.key} = {o.value}")
    wanted = ["correct", "false", "partial", "unknown", "noisy", "temporal", "ambiguous", "adversarial", "command"]
    shown = []
    for cat in wanted:
        for it in x["items"]:
            if it.category == cat and it.text not in shown:
                shown.append(it.text); a = agent.answer(it.text)
                print(f"\n--- [{cat}] « {it.text} »")
                print("  Interprétation : " + ", ".join(f"{w}→{m}" for w, m in a.word_meanings.items()) + f" | intention {a.intent}")
                print("  " + a.explain().replace("\n", "\n  "))
                print(f"  Référence (révélée après coup) : {it.gold}  -> {'CORRECT' if a.output == it.gold else 'ABSTENTION' if a.output in ('UNKNOWN', 'UNKNOWN_WORD', 'ASK') else 'ERREUR'}")
                break
    print("\nRévélation de WorldTruth :")
    for o in world.movables:
        ob = world.objects[o]
        print(f"  {o}: {ob.kind} {ob.color} ; début {world.loc(o, 0)} ; maintenant {world.loc(o)} ; propriétaire {ob.owner}")
    for e in world.events: print(f"  événement t={e.time}: {e.action} {e.patient} -> {e.rel} {e.target} (agent {e.agent})")
    oracle = IndependentWorldVerifier(world)
    o = world.movables[0]
    print(f"\nOracle indépendant : color({o})={world.objects[o].color} -> {oracle.verify({'kind': 'color', 'subject': o, 'value': world.objects[o].color})} ;"
          f" color({o})=violet -> {oracle.verify({'kind': 'color', 'subject': o, 'value': 'violet'})}")


if __name__ == "__main__":
    main()

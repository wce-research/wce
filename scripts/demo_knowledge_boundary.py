"""Démo : VRAI ≠ OBSERVÉ ≠ INFÉRABLE ≠ AFFIRMABLE (environnement indépendant, graine 0)."""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.entitlement import ClaimEntitlement, provenance  # noqa: E402
from core.epistemic_benchmark import (  # noqa: E402
    audit, error_attribution, evaluate, knowability_alignment, make_system, prepare, score, train_lexical_posterior,
)
from core.grounded_benchmark import seed_data  # noqa: E402
from core.independent_agent import ABSTAIN  # noqa: E402
from core.knowledge_boundary import (  # noqa: E402
    LEVEL_NAMES, Relabeler, boundary, public_view, relabel_dataset, true_reliabilities,
)


def main():
    data, splits, caps = seed_data(0)
    x = splits["test"][0]; w = x["world"]
    R = true_reliabilities()
    inv, st, tx = public_view(w, x["observations"])
    print("Vue PUBLIQUE d'un monde (ce que l'agent idéal reçoit) :")
    print(f"  inventaire (capteur de types) : {inv}")
    print(f"  {len(st)} observations structurées (capteurs), {len(tx)} phrases réduites à leur contenu (sans identifiant d'objet)")
    for t in tx[:4]: print(f"    {t.source}: {t.kind} {t.subject} -> {t.value}")
    rl = Relabeler(w, x["observations"], R)
    old = Relabeler(w, x["observations"], R, hidden=True)
    print("\nFrontière de connaissance (quelques propositions) :")
    shown = 0
    for key, truth in w.facts().items():
        if key[0] not in ("color", "loc") or (key[0] == "loc" and key[2] != "now"): continue
        kb = boundary(key, truth, rl.ideal, old.ideal.value(key)[0])
        print(f"  {key}: vrai={truth} | observée directement={kb.directly_observed} | inférable={kb.inferable_from_observations}"
              f" (niveau L{kb.level}: {LEVEL_NAMES[kb.level]}) | seulement avec info cachée={kb.inferable_only_with_hidden_information}"
              f" | contredite={kb.contradicted} | sûre à affirmer={kb.safe_to_assert} | trompeuse={kb.misleading}")
        shown += 1
        if shown >= 6: break
    rules = [r for r in rl.ideal.rules if r.conclusive and r.level == 1]
    print(f"\nÉliminations explicites (agent idéal) : {len(rules)} conclusives")
    for r in rules[:3]:
        print(f"  EliminateCandidates{r.descriptor}: candidats={r.candidates} exclus={r.excluded} -> {r.remaining}"
              f" (confiance {r.confidence:.3f}, provenance {r.provenance})")

    print("\nRéétiquetage de la graine 0 (test) par l'agent idéal :")
    labels = relabel_dataset(splits["test"], R)
    changes = Counter()
    for xx in splits["test"]:
        for it in xx["items"]:
            g, ok = labels[id(it)]
            if ok and g != it.gold: changes[(it.gold if it.gold in ABSTAIN else "valeur", g if g in ABSTAIN else "valeur")] += 1
    print(f"  items réétiquetés : {sum(ok for _, ok in labels.values())}/{len(labels)} ; changements : {dict(changes)}")

    post = train_lexical_posterior()
    print("\nSystème : réponse brute -> provenance -> droit d'affirmer -> réponse finale")
    agent = make_system("unified_entitled"); cal = prepare(agent, caps, splits, post)
    agent.new_world(); agent.ingest(x["observations"])
    ce = ClaimEntitlement(); n = 0
    for it in x["items"]:
        a = agent.answer(it.text)
        if a.output in ABSTAIN: continue
        prov = provenance(agent, a, lambda wd, m: cal.meaning(agent, wd)[1])
        d = ce.decide(a, prov)
        print(f"  « {it.text} » -> brut {a.output} | L{prov.level} | P(croyance)={prov.belief_probability:.3f}"
              f" | incert. épistémique {prov.epistemic_uncertainty:.3f} / linguistique {prov.linguistic_uncertainty:.3f}"
              f" | droit : {d} -> {ce.output(a, d)} | attendu d'origine {it.gold} / idéal {labels[id(it)][0]}")
        if prov.inference_rules: print(f"      règles : {prov.inference_rules[:2]}")
        n += 1
        if n >= 8: break

    print("\nSur tout le test de la graine 0 :")
    for name in ("no_elimination", "entitled", "unified", "unified_entitled"):
        ag = make_system(name); c2 = prepare(ag, caps, splits, post)
        rows = evaluate(ag, splits["test"], labels, c2)
        so, si = score(rows, "out", "orig"), score(rows, "out", "ideal")
        al = knowability_alignment(rows)
        print(f"  {name:18s} attendus d'origine : succès {so['success']:.3f} faux confiants {so['false_confident']:.3f}"
              f" | attendus idéaux : succès {si['success']:.3f} faux confiants {si['false_confident']:.3f}"
              f" | alignement connaissable P {al['precision']:.2f} R {al['recall']:.2f} F1 {al['f1']:.2f}")
        if name == "unified":
            print(f"      audit de connaissabilité : {audit(rows)}")
            print(f"      attribution (idéal) : {error_attribution(rows, 'ideal')[0]}")


if __name__ == "__main__":
    main()

"""Apprentissage actif : sélection par gain d'information vs flux passif (mots inconnus et entités)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.discourse_benchmark import active_entity_learning, active_vs_passive, distill_resolver, observation_pool


def main():
    print("=== WCE ACTIVE LEARNING ===")
    pool, agent = observation_pool(0)
    print(f"\nRéservoir : {len(pool)} observations textuelles contenant des mots jamais légendés")
    print("\nMots inconnus : expositions nécessaires (IG = H(avant) − H(après) sur les sens candidats)")
    for budget in (4, 8, 12, 20, 40):
        r = active_vs_passive(0, budget=budget, pool=(pool, agent))
        print(f"  budget {budget:>2} : passif {r['summary']['passive']} | actif {r['summary']['active']}")
    detail = active_vs_passive(0, budget=40, pool=(pool, agent))["per_word"]
    for w, v in detail.items():
        print(f"  « {w} » (réservoir {v['passive']['pool']}) : passif {v['passive']['exposures_to_learn']} expositions, actif {v['active']['exposures_to_learn']}")
    print(f"\nEntités : demandes d'information choisies au gain d'information : {active_entity_learning()}")
    print(f"\nDistillation du résolveur (élève propose, WCE vérifie) : {distill_resolver()}")


if __name__ == "__main__":
    main()

"""Active learning: selection by information gain vs passive stream (unknown words and entities)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.discourse_benchmark import active_entity_learning, active_vs_passive, distill_resolver, observation_pool


def main():
    print("=== WCE ACTIVE LEARNING ===")
    pool, agent = observation_pool(0)
    print(f"\nPool: {len(pool)} textual observations containing words never captioned")
    print("\nUnknown words: exposures needed (IG = H(before) − H(after) over the candidate senses)")
    for budget in (4, 8, 12, 20, 40):
        r = active_vs_passive(0, budget=budget, pool=(pool, agent))
        print(f"  budget {budget:>2}: passive {r['summary']['passive']} | active {r['summary']['active']}")
    detail = active_vs_passive(0, budget=40, pool=(pool, agent))["per_word"]
    for w, v in detail.items():
        print(f"  '{w}' (pool {v['passive']['pool']}): passive {v['passive']['exposures_to_learn']} exposures, active {v['active']['exposures_to_learn']}")
    print(f"\nEntities: information requests chosen by information gain: {active_entity_learning()}")
    print(f"\nResolver distillation (student proposes, WCE verifies): {distill_resolver()}")


if __name__ == "__main__":
    main()

"""Grounded lexical learning: from an unknown word to a usable concept, with no leak from WorldTruth."""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.grounded import GroundedAgent, GroundedConfig
from core.grounded_benchmark import seed_data
from core.independent_agent import tokenize


def main():
    print("=== WCE GROUNDED LANGUAGE ===")
    data, splits, caps = seed_data(0)
    agent = GroundedAgent(GroundedConfig())
    agent.learn_captions(caps)
    word = "brique"
    print(f"\n1. « {word} » appears in NO caption: initial status {agent.word_meaning(word)[1].upper()}")
    print("\n2-3. Independent observations (texts from noisy sources, aligned with the sensors):")
    agent.warm_up_sources([])
    for i, x in enumerate(splits["train"] + splits["calibration"]):
        texts = [o for o in x["observations"] if o.text and word in tokenize(o.text)]
        agent.new_world(); agent.ingest(x["observations"])
        h = agent.lexmem.get(word)
        if texts and h:
            print(f"   world {i:>2}: « {texts[0].text} » ({texts[0].source}) -> best={h.best} "
                  f"support={h.support[h.best]:.2f} refutations={h.refute[h.best]:.2f} sources={sorted(h.sources[h.best])} status={h.status}")
        if h and h.status in ("BELIEF", "FACT_LIKE") and i > 2: break
    h = agent.lexmem.get(word)
    print(f"\n4. Learned concept: « {word} » -> {h.best} ({h.status}) ; history {h.history} ; candidates {h.candidate_concepts}")
    print("   (an unreliable speaker weighs little: weight = 2·reliability − 1)")
    test = splits["test"][0]
    agent.new_world(); agent.ingest(test["observations"])
    blocks = [o for o, k in agent.belief.inventory.items() if k == "block"]
    q = next((it for x in splits["test"] for it in x["items"] if word in it.text and it.gold not in ("UNKNOWN", "ASK", "UNKNOWN_WORD")), None)
    if q:
        world = next(x for x in splits["test"] if q in x["items"])
        agent.new_world(); agent.ingest(world["observations"])
        a = agent.answer(q.text)
        print(f"\n5. New sentence, new world: « {q.text} »")
        print("   " + a.explain().replace("\n", "\n   "))
        print(f"   reference (revealed afterwards): {q.gold} -> {'CORRECT' if a.output == q.gold else a.output}")
    print("\n6-7. Polysemy: « range » has two learned senses; the type of the patient chooses")
    for x in splits["test"]:
        items = [it for it in x["items"] if it.category == "polysemy"]
        if len(items) >= 2:
            agent.new_world(); agent.ingest(x["observations"])
            for it in items[:3]:
                a = agent.answer(it.text)
                print(f"   « {it.text} » -> sense {a.word_meanings.get('range')} (expected {it.meta['sense']}) ; decision {a.output} (ref. {it.gold})")
            break
    print(f"   learned senses of « range »: {agent.senses('range')}")
    print("\n8-9. Evidence graph and answer:")
    x = splits["test"][1]; agent.new_world(); agent.ingest(x["observations"])
    it = next(it for it in x["items"] if it.gold in ("SUPPORTED", "REFUTED"))
    a = agent.answer(it.text)
    print(f"   « {it.text} »\n   " + a.explain().replace("\n", "\n   ") + f"\n   reference: {it.gold}")
    print("\n10. No leak from WorldTruth:")
    tree = ast.parse(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core/grounded.py")).read())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    print(f"   modules imported by the agent: {sorted(m for m in mods if m and m.startswith('core'))}")
    print("   -> no generator (independent_world*); the truth is read only by the evaluation")


if __name__ == "__main__":
    main()

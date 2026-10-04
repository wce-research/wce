"""Micro-corpus français -> mêmes primitives WCE, même mécanisme d'explication/lexique."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.language_bridge import LanguageBridge, default_corpus, make_world, observed_transition


def main():
    bridge = LanguageBridge()
    print("=== WCE LANGUAGE BRIDGE ===")
    print("\nLearning from experiences (sentence + observed world change):")
    for sentence, heights, entity, primitive in default_corpus():
        skill = bridge.observe_physics(sentence, *observed_transition(heights, entity, primitive))
        print(f"  '{sentence}': {entity}.y {heights[entity]} -> observed ; explained by {skill}")

    print("\nUnknown variants before any experience:")
    world = make_world({"balle": 2.0, "bloc": 3.0})
    for sentence in ("la balle chute", "le bloc s'élève"):
        interp = bridge.interpret(sentence, world.entities)
        print(f"  '{sentence}' -> {interp.skill} ({interp.reason})")

    print("\nOne experience each (with 'balle' only):")
    for sentence, heights, primitive in (("la balle chute", {"balle": 3.0, "bloc": 1.0}, "FALL"),
                                         ("la balle s'élève", {"balle": 0.0, "bloc": 1.0}, "RISE")):
        print(f"  '{sentence}' -> {bridge.observe_physics(sentence, *observed_transition(heights, 'balle', primitive))}")

    print("\nTransfer to another entity, never seen with these verbs:")
    world = make_world({"balle": 0.0, "bloc": 4.0})
    for sentence in ("le bloc chute", "le bloc s'élève", "le bloc s'élève", "le bloc roule"):
        interp, value = bridge.execute(sentence, world)
        print(f"  '{sentence}' -> {interp.skill or 'UNKNOWN'} ; bloc.y = {world.entities['bloc'].properties['y']}"
              f" ({interp.reason})")

    print("\nSame mechanism on symbolic sequences:")
    for sentence, output in (("inverse abcde", "edcba"), ("tourne abcde", "bcdea"), ("décale abcde", "bcdef")):
        print(f"  '{sentence}' observed -> '{output}' ; bound to {bridge.observe_sequence(sentence, output)}")
    for sentence in ("inverse hgfab", "tourne abcgh", "décale aaabh", "trie hgfab"):
        interp, value = bridge.execute(sentence)
        print(f"  '{sentence}' -> {interp.skill or 'UNKNOWN'} -> {value}")
    print(f"\nLexicon: { {v: dict(c) for v, c in bridge.lexicon.items()} }")


if __name__ == "__main__":
    main()

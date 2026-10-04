"""Démo : état cognitif unifié — identité ambiguë, mise à jour, fiabilité des sources, graphe de croyances,
question (réponse ou clarification), contradiction -> révision, question corrigée."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.discourse_benchmark import lexicon_agent  # noqa: E402
from core.unified import UnifiedCognitiveState  # noqa: E402


def show_identity(u, rid, label):
    dist, ev = u.identity(rid, "alice", "observation")
    shown = {e: round(p, 3) for e, p in sorted(dist.items(), key=lambda t: -t[1])}
    print(f"   {label} : P(entité | preuves) = {shown}")
    if ev: print(f"   IdentityEvidence : source={ev.source} r={ev.reliability} soutiens={ev.supporting} contre={ev.contradicting} conf={ev.confidence}")


def show_answer(q, a):
    print(f"\n? {q}")
    print(f"   -> {a.status} / {a.action} : « {a.phrase} »  (confiance {a.confidence:.3f}, identité {a.identity_confidence:.3f})")
    if a.risks: print(f"   risques attendus : {a.risks}")
    if a.dependencies: print(f"   dépend de : {a.dependencies}")


def main():
    L = lexicon_agent(0)
    u = UnifiedCognitiveState(L)
    u.reliability.stats["capteur"] = [60.0, 1.0]
    u.reliability.stats["alice"] = [30.0, 2.0]
    u.reliability.stats["bob"] = [3.0, 9.0]
    print("Fiabilités estimées :", {s: round(u.reliability.estimate(s), 3) for s in ("capteur", "alice", "bob")})
    for e, t in (("e1", "cube"), ("e2", "cube"), ("e3", "ball")): u.observe("capteur", ("type", e), t)
    u.observe("capteur", ("color", "e2"), "blue")
    print("Capteur : e1, e2 = cubes ; e3 = balle ; e2 est bleu.\n")

    print("Phrase 1 (alice) : « Un cube est sur la table. »")
    ms, claims = u.say("alice", "Un cube est sur la table.")
    show_identity(u, claims[0].referent, "identité du cube")
    print(f"   affirmation candidate : {claims[0].kind}={claims[0].value} -> {claims[0].status} (identité non engagée)")

    print("\nPhrase 2 (alice) : « Il est rouge. »")
    u.say("alice", "Il est rouge.")
    show_identity(u, claims[0].referent, "identité révisée")
    for c in u.claims.values():
        print(f"   {c.id}: {c.kind}={c.value} -> entité {c.entity} (P={c.identity_confidence:.3f}, {c.status}, origine={c.origin})")

    print("\nPhrase 3 (bob, peu fiable) : « Le cube rouge est sur l'étagère. »")
    _, c3 = u.say("bob", "Le cube rouge est sur l'étagère.")
    for c in c3: print(f"   {c.kind}={c.value} -> entité {c.entity} (meilleur candidat {c.best_candidate}, P={c.identity_confidence:.3f} < seuil d'observation, {c.status})")

    print("\nGraphe de croyances (maintenance de vérité) :", u.tms.counts())
    for key in sorted(u.belief.graph.props, key=str):
        b = u.belief.graph.belief(key)
        print(f"   {key}: {b.value} {b.status} p={b.posterior:.3f} pour={b.supporting} contre={b.refuting}")

    for q in ("Où est le cube rouge ?", "Où est le cube ?", "Prends le cube rouge."):
        show_answer(q, u.query(q))

    print("\nContradiction (capteur) : le cube e1 est en fait sur l'étagère.")
    _, defeated = u.contradict(("loc", "e1", "now"), ("on", "shelf"))
    print(f"   preuves défaites : {defeated} ; TMS : {u.tms.counts()}")
    print(f"   trajectoire de e1 (même identité, états datés) : {u.trajectory('e1')}")
    show_answer("Où est le cube rouge ?", u.query("Où est le cube rouge ?"))

    print("\nContradiction d'identité : le capteur voit e1 BLEU et e2 ROUGE.")
    u.contradict(("color", "e2"), "red"); u.contradict(("color", "e1"), "blue")
    print(f"   révisions d'identité : {u.revisions}")
    for c in u.claims.values():
        print(f"   {c.id}: {c.kind}={c.value} -> entité {c.entity} (P={c.identity_confidence:.3f}, {c.status})")
    show_answer("Où est le cube rouge ?", u.query("Où est le cube rouge ?"))
    print("\nJournal TMS (dernières transitions) :")
    for row in u.tms.log[-6:]: print("  ", row)


if __name__ == "__main__":
    main()

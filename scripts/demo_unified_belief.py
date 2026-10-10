"""Demo: unified cognitive state — ambiguous identity, update, source reliability, belief graph,
question (answer or clarification), contradiction -> revision, corrected question."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.discourse_benchmark import lexicon_agent  # noqa: E402
from core.unified import UnifiedCognitiveState  # noqa: E402


def show_identity(u, rid, label):
    dist, ev = u.identity(rid, "alice", "observation")
    shown = {e: round(p, 3) for e, p in sorted(dist.items(), key=lambda t: -t[1])}
    print(f"   {label}: P(entity | evidence) = {shown}")
    if ev: print(f"   IdentityEvidence: source={ev.source} r={ev.reliability} supporting={ev.supporting} against={ev.contradicting} conf={ev.confidence}")


def show_answer(q, a):
    print(f"\n? {q}")
    print(f"   -> {a.status} / {a.action}: « {a.phrase} »  (confidence {a.confidence:.3f}, identity {a.identity_confidence:.3f})")
    if a.risks: print(f"   expected risks: {a.risks}")
    if a.dependencies: print(f"   depends on: {a.dependencies}")


def main():
    L = lexicon_agent(0)
    u = UnifiedCognitiveState(L)
    u.reliability.stats["capteur"] = [60.0, 1.0]
    u.reliability.stats["alice"] = [30.0, 2.0]
    u.reliability.stats["bob"] = [3.0, 9.0]
    print("Estimated reliabilities:", {s: round(u.reliability.estimate(s), 3) for s in ("capteur", "alice", "bob")})
    for e, t in (("e1", "cube"), ("e2", "cube"), ("e3", "ball")): u.observe("capteur", ("type", e), t)
    u.observe("capteur", ("color", "e2"), "blue")
    print("Sensor: e1, e2 = cubes; e3 = ball; e2 is blue.\n")

    print("Sentence 1 (alice): « Un cube est sur la table. »")
    ms, claims = u.say("alice", "Un cube est sur la table.")
    show_identity(u, claims[0].referent, "identity of the cube")
    print(f"   candidate claim: {claims[0].kind}={claims[0].value} -> {claims[0].status} (identity not committed)")

    print("\nSentence 2 (alice): « Il est rouge. »")
    u.say("alice", "Il est rouge.")
    show_identity(u, claims[0].referent, "revised identity")
    for c in u.claims.values():
        print(f"   {c.id}: {c.kind}={c.value} -> entity {c.entity} (P={c.identity_confidence:.3f}, {c.status}, origin={c.origin})")

    print("\nSentence 3 (bob, unreliable): « Le cube rouge est sur l'étagère. »")
    _, c3 = u.say("bob", "Le cube rouge est sur l'étagère.")
    for c in c3: print(f"   {c.kind}={c.value} -> entity {c.entity} (best candidate {c.best_candidate}, P={c.identity_confidence:.3f} < observation threshold, {c.status})")

    print("\nBelief graph (truth maintenance):", u.tms.counts())
    for key in sorted(u.belief.graph.props, key=str):
        b = u.belief.graph.belief(key)
        print(f"   {key}: {b.value} {b.status} p={b.posterior:.3f} for={b.supporting} against={b.refuting}")

    for q in ("Où est le cube rouge ?", "Où est le cube ?", "Prends le cube rouge."):
        show_answer(q, u.query(q))

    print("\nContradiction (sensor): the cube e1 is in fact on the shelf.")
    _, defeated = u.contradict(("loc", "e1", "now"), ("on", "shelf"))
    print(f"   defeated evidence: {defeated} ; TMS: {u.tms.counts()}")
    print(f"   trajectory of e1 (same identity, dated states): {u.trajectory('e1')}")
    show_answer("Où est le cube rouge ?", u.query("Où est le cube rouge ?"))

    print("\nIdentity contradiction: the sensor sees e1 BLUE and e2 RED.")
    u.contradict(("color", "e2"), "red"); u.contradict(("color", "e1"), "blue")
    print(f"   identity revisions: {u.revisions}")
    for c in u.claims.values():
        print(f"   {c.id}: {c.kind}={c.value} -> entity {c.entity} (P={c.identity_confidence:.3f}, {c.status})")
    show_answer("Où est le cube rouge ?", u.query("Où est le cube rouge ?"))
    print("\nTMS log (latest transitions):")
    for row in u.tms.log[-6:]: print("  ", row)


if __name__ == "__main__":
    main()

"""Démo : croyance temporelle — t0 balle sur la table ; t1 rien ; t2 un locuteur faible dit qu'elle est tombée ;
t3 le capteur la voit au sol. Chronologie, taux de changement, fiabilités, postérieure de l'événement, droit d'affirmer."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.temporal_belief import TemporalBeliefState, TemporalTMS  # noqa: E402
from core.temporal_bench import DOMAINS  # noqa: E402
from core.temporal_eval import fit_training  # noqa: E402
from core.temporal_bench import train_episodes  # noqa: E402


def show(st, now, label):
    d = st.current(now); status, v, p = st.status(now); ent = st.entitlement(now)[0]
    top = sorted(d.items(), key=lambda kv: -kv[1])[:3]
    print(f"  t={now} {label:44s} présent {[(k, round(x, 3)) for k, x in top]} -> {status} / {ent}")


def main():
    hz, rel = fit_training(train_episodes(40))
    print("Taux de changement appris sur les épisodes d'ENTRAÎNEMENT (λ par pas, classe) :")
    for k, d in hz.dyn.items(): print(f"  {k:7s} λ={d.hazard:.4f} {d.klass:14s} P(inchangé après 5 pas)={d.persistence(5):.3f}")
    r = {s: rel.estimate_for(s, "loc") for s in ("sensor", "witness", "gossip")}
    re = {s: rel.estimate_for(s + "|event", "loc") for s in ("sensor", "witness", "gossip")}
    print("Fiabilités (observations) :", {k: round(v, 3) for k, v in r.items()}, "| (événements) :", {k: round(v, 3) for k, v in re.items()})

    st = TemporalBeliefState("balle", "loc", DOMAINS["loc"], hz.get("loc"))
    print("\nChronologie :")
    st.observe(0, "table", "sensor", r["sensor"]); show(st, 0, "capteur : sur la table")
    show(st, 1, "aucune observation (persistance ≠ certitude)")
    st.event(2, "floor", "gossip", re["gossip"]); show(st, 2, "locuteur FAIBLE : « elle est tombée »")
    print(f"      postérieure de l'événement (historique t2) : {st.branches(2, 2)}")
    st.observe(3, "floor", "sensor", r["sensor"]); show(st, 3, "capteur : au sol")
    print(f"      événement révisé rétroactivement (t2, sachant t3) : {st.branches(2, 3)}")
    print(f"      « où était-elle avant de tomber ? » (historique t1) : {max(st.historical(1, 3).items(), key=lambda kv: kv[1])}")
    print(f"      « est-elle encore sur la table ? » P(table)={st.current(3)['table']:.3f}")
    for later in (6, 12, 25):
        show(st, later, f"sans nouvelle observation ({later - 3} pas)")
    print("\nSegments compressés :", [(s.value, s.valid_from, s.valid_until, round(s.confidence, 2)) for s in st.segments(25)])

    print("\nPropriétaire (dynamique lente) vs position (rapide), même preuve ancienne à t0 :")
    for key, v in (("owner", "marie"), ("loc", "table"), ("color", "red")):
        s2 = TemporalBeliefState("x", key, DOMAINS[key], hz.get(key)); s2.observe(0, v, "sensor", r["sensor"])
        print(f"  {key:6s} présent à t=10 : P={s2.current(10)[v]:.3f} -> {s2.entitlement(10)[0]}")

    print("\nSource fiable + preuve ancienne  vs  source faible + preuve fraîche :")
    s3 = TemporalBeliefState("x", "loc", DOMAINS["loc"], hz.get("loc"))
    s3.observe(0, "table", "sensor", r["sensor"]); s3.observe(9, "shelf", "witness", r["witness"])
    print(f"  à t=10 : {[(k, round(x, 3)) for k, x in sorted(s3.current(10).items(), key=lambda kv: -kv[1])[:2]]}")

    print("\nConflit temporel : deux sources crédibles au MÊME instant vs changement d'état :")
    c = TemporalBeliefState("x", "loc", DOMAINS["loc"], hz.get("loc")); c.observe(5, "table", "camera", 0.9); c.observe(5, "box", "doc", 0.85)
    d = TemporalBeliefState("x", "loc", DOMAINS["loc"], hz.get("loc")); d.observe(2, "table", "camera", 0.9); d.observe(5, "box", "doc", 0.85)
    print(f"  même instant -> {c.status(5)[0]} ; instants différents -> {d.status(5)[0]} (valeur {d.status(5)[1]})")

    print("\nCorrection rétroactive (à t5, on apprend qu'elle est tombée à t3) :")
    s4 = TemporalBeliefState("x", "loc", DOMAINS["loc"], hz.get("loc")); s4.observe(0, "table", "sensor", r["sensor"]); s4.observe(1, "table", "sensor", r["sensor"])
    tms = TemporalTMS(s4); changed = tms.late_report(3, "floor", "doc", rel.estimate_for("doc|event", "loc"), "event", 5)
    print(f"  instants révisés : {changed} ; t0..t2 conservés : {[(b['t'], b['value']) for b in tms.timeline(5)]}")
    print("\nRéponse réfutée -> révision minimale :", s4.minimal_revision(5, "floor"))


if __name__ == "__main__":
    main()

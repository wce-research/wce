"""Demo: temporal belief — t0 ball on the table; t1 nothing; t2 a weak speaker says it fell;
t3 the sensor sees it on the floor. Timeline, change rates, reliabilities, posterior of the event, entitlement to assert."""
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
    print(f"  t={now} {label:44s} present {[(k, round(x, 3)) for k, x in top]} -> {status} / {ent}")


def main():
    hz, rel = fit_training(train_episodes(40))
    print("Change rates learned on the TRAINING episodes (λ per step, class):")
    for k, d in hz.dyn.items(): print(f"  {k:7s} λ={d.hazard:.4f} {d.klass:14s} P(unchanged after 5 steps)={d.persistence(5):.3f}")
    r = {s: rel.estimate_for(s, "loc") for s in ("sensor", "witness", "gossip")}
    re = {s: rel.estimate_for(s + "|event", "loc") for s in ("sensor", "witness", "gossip")}
    print("Reliabilities (observations):", {k: round(v, 3) for k, v in r.items()}, "| (events):", {k: round(v, 3) for k, v in re.items()})

    st = TemporalBeliefState("balle", "loc", DOMAINS["loc"], hz.get("loc"))
    print("\nTimeline:")
    st.observe(0, "table", "sensor", r["sensor"]); show(st, 0, "sensor: on the table")
    show(st, 1, "no observation (persistence ≠ certainty)")
    st.event(2, "floor", "gossip", re["gossip"]); show(st, 2, "WEAK speaker: “it fell”")
    print(f"      posterior of the event (historical t2): {st.branches(2, 2)}")
    st.observe(3, "floor", "sensor", r["sensor"]); show(st, 3, "sensor: on the floor")
    print(f"      event revised retroactively (t2, given t3): {st.branches(2, 3)}")
    print(f"      “where was it before falling?” (historical t1): {max(st.historical(1, 3).items(), key=lambda kv: kv[1])}")
    print(f"      “is it still on the table?” P(table)={st.current(3)['table']:.3f}")
    for later in (6, 12, 25):
        show(st, later, f"without new observation ({later - 3} steps)")
    print("\nCompressed segments:", [(s.value, s.valid_from, s.valid_until, round(s.confidence, 2)) for s in st.segments(25)])

    print("\nOwner (slow dynamics) vs position (fast), same old evidence at t0:")
    for key, v in (("owner", "marie"), ("loc", "table"), ("color", "red")):
        s2 = TemporalBeliefState("x", key, DOMAINS[key], hz.get(key)); s2.observe(0, v, "sensor", r["sensor"])
        print(f"  {key:6s} present at t=10: P={s2.current(10)[v]:.3f} -> {s2.entitlement(10)[0]}")

    print("\nReliable source + old evidence  vs  weak source + fresh evidence:")
    s3 = TemporalBeliefState("x", "loc", DOMAINS["loc"], hz.get("loc"))
    s3.observe(0, "table", "sensor", r["sensor"]); s3.observe(9, "shelf", "witness", r["witness"])
    print(f"  at t=10: {[(k, round(x, 3)) for k, x in sorted(s3.current(10).items(), key=lambda kv: -kv[1])[:2]]}")

    print("\nTemporal conflict: two credible sources at the SAME instant vs change of state:")
    c = TemporalBeliefState("x", "loc", DOMAINS["loc"], hz.get("loc")); c.observe(5, "table", "camera", 0.9); c.observe(5, "box", "doc", 0.85)
    d = TemporalBeliefState("x", "loc", DOMAINS["loc"], hz.get("loc")); d.observe(2, "table", "camera", 0.9); d.observe(5, "box", "doc", 0.85)
    print(f"  same instant -> {c.status(5)[0]} ; different instants -> {d.status(5)[0]} (value {d.status(5)[1]})")

    print("\nRetroactive correction (at t5, we learn that it fell at t3):")
    s4 = TemporalBeliefState("x", "loc", DOMAINS["loc"], hz.get("loc")); s4.observe(0, "table", "sensor", r["sensor"]); s4.observe(1, "table", "sensor", r["sensor"])
    tms = TemporalTMS(s4); changed = tms.late_report(3, "floor", "doc", rel.estimate_for("doc|event", "loc"), "event", 5)
    print(f"  revised instants: {changed} ; t0..t2 kept: {[(b['t'], b['value']) for b in tms.timeline(5)]}")
    print("\nRefuted answer -> minimal revision:", s4.minimal_revision(5, "floor"))


if __name__ == "__main__":
    main()

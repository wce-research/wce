import ast
import math
import os
import unittest

import torch

from core.temporal_belief import (
    ASSERTED, CONFLICTED, FAST, KNOWN, LIKELY, QUALIFIED, SLOW, STALE, STATIC, UNKNOWN, WITHHELD, HazardModel,
    JointDerivedState, KeyDynamics, TemporalBeliefState, TemporalMemory, TemporalTMS, beta_binomial_interval, classify,
)
from core.temporal_bench import (
    DOMAINS, HEIGHTS, SEALED_SHA256, TRUE_HAZARD, dev_episodes, episode, sealed_hash, train_episodes,
)
from core.temporal_eval import SYSTEMS_T, answer, build_memory, compare, evaluate, fit_training, metrics, source_interval_coverage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLACES = DOMAINS["loc"]
HZ = REL = CMP = None


def setUpModule():
    global HZ, REL, CMP
    HZ, REL = fit_training(train_episodes(20))
    CMP = compare(dev_episodes(6), train_episodes(20))


def st(key="loc", lam=0.1, dom=None):
    return TemporalBeliefState("ball", key, dom or PLACES, KeyDynamics(key, lam))


# ================================================================ temporal state and persistence

class TemporalStateTests(unittest.TestCase):
    def test_fields(self):
        s = st(); s.observe(3, "table", "sensor", 0.95)
        r = s.reports[0]; self.assertEqual((r.t, r.value, r.source, r.kind), (3, "table", "sensor", "observation"))
    def test_observation_vs_ingestion_time(self):
        s = st(); s.observe(3, "table", "doc", 0.8, ingested_at=7); self.assertEqual((s.reports[0].t, s.reports[0].ingested_at), (3, 7))
    def test_known_until(self): s = st(); s.observe(3, "table", "doc", 0.8, ingested_at=7); self.assertEqual(s.known_until(), 7)
    def test_posterior_normalized(self):
        s = st(); s.observe(0, "table", "sensor", 0.95); f, sm = s.posterior(5)
        self.assertTrue(torch.allclose(f.sum(1), torch.ones(6), atol=1e-5)); self.assertTrue(torch.allclose(sm.sum(1), torch.ones(6), atol=1e-5))
    def test_confidence_decays(self):
        s = st(); s.observe(0, "table", "sensor", 0.95)
        ps = [s.current(t)["table"] for t in (0, 2, 5, 10)]; self.assertEqual(ps, sorted(ps, reverse=True))
    def test_decay_toward_uniform(self):
        s = st(lam=0.5); s.observe(0, "table", "sensor", 0.95); self.assertAlmostEqual(s.current(60)["table"], 0.25, places=2)
    def test_static_no_decay(self):
        s = st("color", 1e-5, DOMAINS["color"]); s.observe(0, "red", "sensor", 0.95); self.assertGreater(s.current(50)["red"], 0.94)
    def test_no_observation_not_no_change(self):
        s = st(); s.observe(0, "table", "sensor", 0.99); self.assertLess(s.current(10)["table"], 0.9)
    def test_persistence_formula(self): self.assertAlmostEqual(KeyDynamics("loc", 0.1).persistence(3), math.exp(-0.3))
    def test_persistence_negative_dt(self): self.assertEqual(KeyDynamics("loc", 0.1).persistence(-2), 1.0)
    def test_two_observations_agree(self):
        s = st(); s.observe(5, "table", "witness", 0.7); s.observe(5, "table", "doc", 0.8); self.assertGreater(s.current(5)["table"], 0.9)
    def test_present_vs_past_distinct(self):
        s = st(); s.observe(0, "table", "sensor", 0.95); s.observe(10, "floor", "sensor", 0.95)
        self.assertEqual(max(s.current(10).items(), key=lambda kv: kv[1])[0], "floor")
        self.assertEqual(max(s.historical(0, 10).items(), key=lambda kv: kv[1])[0], "table")
    def test_old_observation_does_not_answer_present(self):
        s = st(); s.observe(0, "table", "sensor", 0.95); self.assertNotEqual(s.entitlement(15)[0], ASSERTED)
    def test_uniform_prior(self): self.assertAlmostEqual(st().current(0)["table"], 0.25, places=5)
    def test_cache_invalidated(self):
        s = st(); s.observe(0, "table", "sensor", 0.95); a = s.current(3)["table"]; s.observe(3, "floor", "sensor", 0.95)
        self.assertLess(s.current(3)["table"], a)


class HazardTests(unittest.TestCase):
    def test_learned_hazards_order(self):
        self.assertGreater(HZ.get("loc").hazard, HZ.get("owner").hazard); self.assertGreater(HZ.get("owner").hazard, HZ.get("color").hazard)
    def test_learned_close_to_truth_loc(self): self.assertAlmostEqual(HZ.get("loc").hazard, TRUE_HAZARD["loc"], delta=0.03)
    def test_learned_close_to_truth_owner(self): self.assertAlmostEqual(HZ.get("owner").hazard, TRUE_HAZARD["owner"], delta=0.008)
    def test_classes(self):
        self.assertEqual((HZ.get("color").klass, HZ.get("owner").klass, HZ.get("loc").klass), (STATIC, SLOW, FAST))
    def test_classify(self): self.assertEqual((classify(0.0001), classify(0.01), classify(0.2)), (STATIC, SLOW, FAST))
    def test_fit_counts(self):
        h = HazardModel().fit([("loc", ["a", "a", "b", "b", "c"])]); d = h.get("loc"); self.assertEqual((d.changes, d.exposure), (2, 4))
    def test_shrinkage(self):
        h = HazardModel().fit([("loc", ["a", "b"])], {"loc": FAST}); self.assertLess(h.get("loc").hazard, 1.0)
    def test_default(self): self.assertEqual(HazardModel().get("unknown_key").klass, FAST)
    def test_configurable(self):
        h = HazardModel({"owner": (0.02, SLOW)}); self.assertEqual(h.get("owner").hazard, 0.02)
    def test_bump_learned(self): self.assertTrue(0 < HZ.bump < 0.5)
    def test_toggle_learned(self): self.assertTrue(0 < HZ.toggle < 0.2)


class EventTests(unittest.TestCase):
    def test_observed_event_changes_state(self):
        s = st(); s.observe(0, "table", "sensor", 0.95); s.event(2, "floor", "sensor", 0.95)
        self.assertEqual(max(s.current(2).items(), key=lambda kv: kv[1])[0], "floor")
    def test_weak_event_partial(self):
        s = st(); s.observe(0, "table", "sensor", 0.95); s.event(2, "floor", "gossip", 0.4)
        d = s.current(2); self.assertTrue(0.2 < d["floor"] < 0.6)
    def test_event_reliability_monotone(self):
        ps = []
        for r in (0.3, 0.6, 0.9):
            s = st(); s.observe(0, "table", "sensor", 0.95); s.event(2, "floor", "x", r); ps.append(s.current(2)["floor"])
        self.assertEqual(ps, sorted(ps))
    def test_competing_events_branches(self):
        s = st(); s.observe(0, "table", "sensor", 0.95); s.event(3, "floor", "witness", 0.7); s.event(3, "shelf", "gossip", 0.45)
        self.assertGreaterEqual(len(s.branches(3, 4)), 2)
    def test_competing_events_not_last_wins(self):
        s = st(); s.observe(0, "table", "sensor", 0.95); s.event(3, "floor", "witness", 0.8); s.event(3, "shelf", "gossip", 0.4)
        d = s.current(3); self.assertGreater(d["floor"], d["shelf"])
    def test_hidden_event_not_invented(self):
        s = st(lam=0.2); s.observe(0, "table", "sensor", 0.95); d = s.current(10)
        others = [d[v] for v in PLACES if v != "table"]; self.assertAlmostEqual(max(others), min(others), places=5)
    def test_event_then_confirmation(self):
        s = st(); s.observe(0, "table", "sensor", 0.95); s.event(2, "floor", "gossip", 0.4); s.observe(3, "floor", "sensor", 0.95)
        self.assertGreater(s.historical(2, 3)["floor"], 0.8)
    def test_false_event_rejected_by_sensor(self):
        s = st(); s.observe(0, "table", "sensor", 0.95); s.event(2, "floor", "gossip", 0.4); s.observe(3, "table", "sensor", 0.95)
        self.assertGreater(s.current(3)["table"], 0.9)
    def test_before_fall(self):
        s = st(); s.observe(0, "shelf", "sensor", 0.95); s.event(4, "floor", "sensor", 0.95); s.observe(5, "floor", "sensor", 0.95)
        self.assertEqual(max(s.historical(3, 5).items(), key=lambda kv: kv[1])[0], "shelf")


class StatusTests(unittest.TestCase):
    def test_known(self): s = st(); s.observe(0, "table", "sensor", 0.97); self.assertEqual(s.status(0)[0], KNOWN)
    def test_stale(self): s = st(); s.observe(0, "table", "sensor", 0.97); self.assertEqual(s.status(2)[0], STALE)
    def test_stale_is_distinct(self): self.assertEqual(len({STALE, UNKNOWN, CONFLICTED, "REFUTED"}), 4)
    def test_stale_with_value(self): s = st(); s.observe(0, "table", "sensor", 0.97); self.assertEqual(s.status(2)[1], "table")
    def test_unknown(self): self.assertEqual(st().status(3)[0], UNKNOWN)
    def test_likely(self): s = st(); s.observe(0, "table", "witness", 0.75); self.assertIn(s.status(0)[0], (LIKELY, KNOWN))
    def test_entitlement_progression(self):
        s = st(); s.observe(0, "table", "sensor", 0.97)
        seq = [s.entitlement(t)[0] for t in (0, 2, 15)]; self.assertEqual(seq, [ASSERTED, QUALIFIED, WITHHELD])
    def test_stale_low_withheld(self): s = st(lam=0.3); s.observe(0, "table", "sensor", 0.97); self.assertEqual(s.entitlement(10)[0], WITHHELD)


class ConflictTests(unittest.TestCase):
    def test_same_time_conflict(self):
        s = st(); s.observe(5, "table", "camera", 0.9); s.observe(5, "box", "doc", 0.85); self.assertTrue(s.conflict_at(5))
    def test_different_time_not_conflict(self):
        s = st(); s.observe(2, "table", "camera", 0.9); s.observe(5, "box", "doc", 0.85); self.assertFalse(s.conflict_at(5))
    def test_weak_source_no_conflict(self):
        s = st(); s.observe(5, "table", "camera", 0.95); s.observe(5, "box", "gossip", 0.4); self.assertFalse(s.conflict_at(5))
    def test_status_conflicted(self):
        s = st(); s.observe(5, "table", "camera", 0.9); s.observe(5, "box", "doc", 0.85); self.assertEqual(s.status(5)[0], CONFLICTED)
    def test_discriminating_no_conflict(self):
        s = st(); s.observe(5, "table", "camera", 0.99); s.observe(5, "box", "doc", 0.62); self.assertFalse(s.conflict_at(5))
    def test_conflict_threshold_separate(self):
        s = st(); s.observe(5, "table", "camera", 0.65); s.observe(5, "box", "doc", 0.65)
        self.assertTrue(s.conflict_at(5, credible=0.6)); self.assertFalse(s.conflict_at(5, credible=0.7))


class RetroactiveTests(unittest.TestCase):
    def test_late_report_revises_from_point(self):
        s = st(); s.observe(0, "table", "sensor", 0.95); s.observe(1, "table", "sensor", 0.95)
        changed = TemporalTMS(s).late_report(3, "floor", "doc", 0.9, "event", 5); self.assertEqual(min(changed), 3)
    def test_history_preserved(self):
        s = st(); s.observe(0, "table", "sensor", 0.95); s.observe(1, "table", "sensor", 0.95)
        tms = TemporalTMS(s); tms.late_report(3, "floor", "doc", 0.9, "event", 5)
        self.assertEqual([b["value"] for b in tms.timeline(5)][:3], ["table"] * 3)
    def test_belief_at_justifications(self):
        s = st(); s.observe(0, "table", "sensor", 0.95); b = TemporalTMS(s).belief_at(0, 2)
        self.assertEqual(b["justifications"][0][0], "sensor")
    def test_timeline_length(self): s = st(); s.observe(0, "table", "sensor", 0.95); self.assertEqual(len(TemporalTMS(s).timeline(4)), 5)
    def test_log(self):
        s = st(); s.observe(0, "table", "sensor", 0.95); tms = TemporalTMS(s); tms.late_report(2, "floor", "doc", 0.9, "event", 4)
        self.assertEqual(tms.log[-1]["t"], 2)


class SelfCorrectionTests(unittest.TestCase):
    def test_persistence_blamed_for_old_evidence(self):
        s = st(lam=0.08); s.observe(0, "table", "sensor", 0.95); self.assertEqual(s.minimal_revision(8, "table")["cause"], "persistence")
    def test_source_blamed_for_weak_fresh(self):
        s = st(lam=0.08); s.observe(7, "table", "gossip", 0.5); self.assertEqual(s.minimal_revision(8, "table")["cause"], "source")
    def test_event_blamed(self):
        s = st(lam=0.01); s.observe(0, "shelf", "sensor", 0.99); s.event(5, "table", "gossip", 0.3)
        self.assertEqual(s.minimal_revision(6, "table")["cause"], "event")
    def test_one_revision_only(self):
        s = st(lam=0.08); s.observe(0, "table", "sensor", 0.95); s.observe(1, "table", "witness", 0.7)
        r = s.minimal_revision(8, "table"); self.assertEqual(r["revised"], 1); self.assertGreaterEqual(r["candidates"], 2)
    def test_revision_changes_answer(self):
        s = st(lam=0.02); s.observe(7, "table", "gossip", 0.5); before = s.current(8)["table"]; s.minimal_revision(8, "table")
        self.assertLess(s.current(8)["table"], before)
    def test_nothing_to_revise(self): self.assertIsNone(st().minimal_revision(3, "table"))


class JointHeightTests(unittest.TestCase):
    def joint(self):
        base = st(); return JointDerivedState(base, (0, 1), lambda l, b: min(HEIGHTS[l] + 0.5 * b, 2.5), 0.03, DOMAINS["height"], [0.98, 0.02])
    def test_height_from_loc(self):
        j = self.joint(); j.base.observe(0, "shelf", "sensor", 0.97); self.assertEqual(max(j.current(0).items(), key=lambda kv: kv[1])[0], 2.0)
    def test_height_sums(self):
        j = self.joint(); j.base.observe(0, "table", "sensor", 0.95); self.assertAlmostEqual(sum(j.current(3).values()), 1.0, places=5)
    def test_rise_without_relation_change(self):
        j = self.joint(); j.base.observe(0, "table", "sensor", 0.97); j.observe(2, 1.5, "sensor", 0.95); j.base.observe(2, "table", "sensor", 0.97)
        self.assertEqual(max(j.current(2).items(), key=lambda kv: kv[1])[0], 1.5)
    def test_fall_changes_height(self):
        j = self.joint(); j.base.observe(0, "shelf", "sensor", 0.97); j.base.event(2, "floor", "sensor", 0.95)
        self.assertEqual(max(j.current(2).items(), key=lambda kv: kv[1])[0], 0.0)
    def test_height_decays(self):
        j = self.joint(); j.base.observe(0, "table", "sensor", 0.97); self.assertGreater(j.current(0)[1.0], j.current(10)[1.0])
    def test_historical_height(self):
        j = self.joint(); j.base.observe(0, "shelf", "sensor", 0.97); j.base.observe(5, "floor", "sensor", 0.97)
        self.assertEqual(max(j.historical(0, 5).items(), key=lambda kv: kv[1])[0], 2.0)


class MemoryTests(unittest.TestCase):
    def test_segments_compress(self):
        s = st(lam=0.01); s.observe(0, "table", "sensor", 0.99)
        for t in range(1, 8): s.observe(t, "table", "sensor", 0.99)
        segs = s.segments(8); self.assertLess(len(segs), 9); self.assertEqual(segs[0].value, "table")
    def test_segment_bounds(self):
        s = st(lam=0.01)
        for t in range(6): s.observe(t, "table", "sensor", 0.99)
        self.assertEqual((s.segments(5)[0].valid_from, s.segments(5)[-1].valid_until), (0, 5))
    def test_temporal_memory(self):
        m = TemporalMemory(HZ, DOMAINS); m.state("e0", "loc").observe(0, "table", "sensor", 0.9)
        self.assertEqual(m.size(5)["properties"], 1)
    def test_memory_shared_state(self):
        m = TemporalMemory(HZ, DOMAINS); self.assertIs(m.state("e0", "loc"), m.state("e0", "loc"))
    def test_prediction_set_covers(self):
        s = st(); s.observe(0, "table", "sensor", 0.95); self.assertIn("table", s.prediction_set(0, 0.9, present=True))
    def test_prediction_set_grows(self):
        s = st(); s.observe(0, "table", "sensor", 0.95)
        self.assertGreaterEqual(len(s.prediction_set(10, 0.9, present=True)), len(s.prediction_set(0, 0.9, present=True)))


class PredictiveIntervalTests(unittest.TestCase):
    def test_bounds(self): lo, hi = beta_binomial_interval(30, 10, 20); self.assertTrue(0 <= lo <= hi <= 1)
    def test_wider_than_credible(self):
        from core.source_reliability import beta_quantiles
        lo, hi = beta_quantiles(300, 100, (0.05, 0.95)); plo, phi = beta_binomial_interval(300, 100, 20)
        self.assertGreater(phi - plo, hi - lo)
    def test_narrows_with_n(self):
        a = beta_binomial_interval(300, 100, 10); b = beta_binomial_interval(300, 100, 1000); self.assertLess(b[1] - b[0], a[1] - a[0])
    def test_contains_mean(self): lo, hi = beta_binomial_interval(75, 25, 50); self.assertTrue(lo <= 0.75 <= hi)
    def test_coverage_better(self):
        c = source_interval_coverage(dev_episodes(4), train_episodes(10)); self.assertGreater(c["predictive"], c["credible"])
    def test_predictive_coverage_near_nominal(self):
        c = source_interval_coverage(dev_episodes(6), train_episodes(20)); self.assertGreater(c["predictive"], 0.8)


# ================================================================ temporal bench

class BenchTests(unittest.TestCase):
    def test_stdlib_only(self):
        tree = ast.parse(open(os.path.join(ROOT, "core/temporal_bench.py")).read())
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertEqual(mods - {"__future__"}, {"hashlib", "json", "math", "random"})
    def test_sealed_integrity(self): self.assertEqual(sealed_hash(), SEALED_SHA256)
    def test_deterministic(self): self.assertEqual(episode(11)["questions"], episode(11)["questions"])
    def test_question_types(self):
        types = {q["type"] for ep in dev_episodes(6) for q in ep["questions"]}
        self.assertTrue({"present", "past", "still_on_table"} <= types)
    def test_delayed_reports_exist(self):
        self.assertTrue(any(r["ingested_at"] > r["t"] for ep in dev_episodes(3) for r in ep["reports"]))
    def test_false_events_exist(self):
        self.assertTrue(any(r["kind"] == "event" and r["source"] == "gossip" for ep in dev_episodes(3) for r in ep["reports"]))
    def test_expected_values(self):
        exp = {q["expected"] for ep in dev_episodes(6) for q in ep["questions"]}; self.assertIn("?", exp)
    def test_ideal_height_uses_support(self):
        ep = episode(6001)
        for q in ep["questions"]:
            if q["key"] == "height" and q["expected"] not in ("?", "conflit"): self.assertEqual(q["expected"], q["truth"])


class SystemComparisonTests(unittest.TestCase):
    def test_systems(self): self.assertEqual(set(SYSTEMS_T) | {"hazards"}, set(CMP))
    def test_temporal_beats_no_hazard(self): self.assertGreater(CMP["temporal"]["success"], CMP["no_hazard"]["success"])
    def test_temporal_safer_than_no_hazard(self): self.assertLess(CMP["temporal"]["false_confident"], CMP["no_hazard"]["false_confident"])
    def test_temporal_beats_last_observation(self): self.assertGreater(CMP["temporal"]["success"], CMP["last_observation"]["success"])
    def test_temporal_fc_bounded(self): self.assertLess(CMP["temporal"]["false_confident"], 0.08)
    def test_height_calibrated_better(self):
        self.assertLess(CMP["temporal"]["ece"]["height_present"], CMP["no_hazard"]["ece"]["height_present"])
    def test_metrics_keys(self):
        for k in ("conflict", "ece", "prediction_sets", "by_type"): self.assertIn(k, CMP["temporal"])
    def test_answer_api(self):
        ep = dev_episodes(1)[0]; mem = build_memory(ep, HZ, REL, "temporal")
        got, p, best, d = answer(ep, ep["questions"][0], mem, REL, "temporal"); self.assertIsInstance(got, str)
    def test_no_hazard_memory(self):
        ep = dev_episodes(1)[0]; mem = build_memory(ep, HZ, REL, "no_hazard"); self.assertLess(mem.hz.get("loc").hazard, 1e-5)


def _type_case(sys_name, qtype):
    def t(self):
        bt = CMP[sys_name]["by_type"].get(qtype)
        if bt is None: self.skipTest("type absent")
        self.assertTrue(0.0 <= bt["success"] <= 1.0)
        if sys_name == "temporal": self.assertLessEqual(bt["fc"], 0.15)
    return t


class PerTypeTests(unittest.TestCase):
    pass


for _s in SYSTEMS_T:
    for _q in ("present", "past", "still_on_table", "before_fall"):
        setattr(PerTypeTests, f"test_{_s}_{_q}", _type_case(_s, _q))


def _key_case(key):
    def t(self):
        e = CMP["temporal"]["ece"]
        for typ in ("present", "past"):
            k = f"{key}_{typ}"
            if k in e: self.assertLess(e[k], 0.35)
        cov = CMP["temporal"]["prediction_sets"].get(key)
        if cov: self.assertGreater(cov["coverage90"], 0.8)
    return t


class PerKeyTests(unittest.TestCase):
    pass


for _k in ("loc", "height", "owner", "color"): setattr(PerKeyTests, f"test_key_{_k}_calibration_and_coverage", _key_case(_k))


class IndependentEnvTemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.grounded_benchmark import seed_data
        cls.data, cls.splits, cls.caps = seed_data(0, n_worlds=10)
    def test_change_rate(self):
        from core.temporal_independent import learn_change_rate
        p = learn_change_rate(self.splits["train"]); self.assertTrue(0 < p < 1)
    def test_temporal_loc_entry(self):
        from core.independent_agent import Evidence, EvidenceGraph, SourceReliability
        from core.temporal_independent import temporal_loc
        g = EvidenceGraph(SourceReliability()); g.add(("loc", "o1", 0), Evidence("sensor_A", ("on", "table"), 0, "s"))
        b = temporal_loc(g, SourceReliability(), "o1", 0.2, 0); self.assertEqual(b.value, ("on", "table"))
    def test_past_evidence_weakens_for_present(self):
        from core.independent_agent import Evidence, EvidenceGraph, SourceReliability
        from core.temporal_independent import temporal_loc
        g = EvidenceGraph(SourceReliability()); g.add(("loc", "o1", 0), Evidence("sensor_A", ("on", "table"), 0, "s"))
        self.assertLess(temporal_loc(g, SourceReliability(), "o1", 0.3, "now").posterior, temporal_loc(g, SourceReliability(), "o1", 0.3, 0).posterior)
    def test_event_overrides(self):
        from core.independent_agent import Evidence, EvidenceGraph, SourceReliability
        from core.temporal_independent import temporal_loc
        rel = SourceReliability(); rel.stats["sensor_A"] = [60, 1]
        g = EvidenceGraph(rel); g.add(("loc", "o1", 0), Evidence("sensor_A", ("on", "table"), 0, "s")); g.add(("event", "o1"), Evidence("sensor_A", ("on", "floor"), 1, "s"))
        self.assertEqual(temporal_loc(g, rel, "o1", 0.2, "now").value, ("on", "floor"))
    def test_install(self):
        from core.grounded_benchmark import SYSTEMS
        from core.temporal_independent import install_temporal
        a = SYSTEMS["no_elimination"](); a.learn_captions(self.caps); install_temporal(a, 0.2); b = a.new_world()
        self.assertEqual(b.loc("zzz").status, "UNKNOWN")


class SafeSpatialTests(unittest.TestCase):
    def test_l1_temporal_system_registered(self):
        import inspect
        from core import source_benchmark
        self.assertIn("l1_temporal", inspect.getsource(source_benchmark.build))
    def test_l2_never_in_l1_mode(self):
        from core.unified_agent import UnifiedAgent, UnifiedConfig
        a = UnifiedAgent(UnifiedConfig(identity_mode="l1")); a.new_world(); a._mapping[("cube", "red")] = ("o1", 1.0)
        a.belief.inventory.update({"o1": "cube", "o2": "cube"}); self.assertEqual(a.identity_prob("cube", "red", "query"), (None, 0.0))


if __name__ == "__main__":
    unittest.main()


class ConflictThresholdSeparationTests(unittest.TestCase):
    def model(self):
        from core.source_reliability import SourceReliabilityModel
        m = SourceReliabilityModel(mode="lower", global_prior=(1, 1))
        for s, ok, bad in (("camera", 13, 7), ("doc", 13, 7)):
            for _ in range(ok): m.observe(s, True)
            for _ in range(bad): m.observe(s, False)
        return m
    def test_credibility_uses_mean(self):
        m = self.model(); self.assertGreater(m.conflict_credibility("camera"), m.estimate("camera"))
    def test_threshold_attribute(self): self.assertEqual(self.model().conflict_threshold, 0.6)
    def test_conflict_declared_with_mean(self):
        from core.independent_agent import CONFLICTED, Evidence, EvidenceGraph
        m = self.model(); g = EvidenceGraph(m)
        g.add(("color", "o"), Evidence("camera", "red", 0, "t")); g.add(("color", "o"), Evidence("doc", "blue", 0, "t"))
        self.assertEqual(g.belief(("color", "o")).status, CONFLICTED)
    def test_old_rule_would_miss(self):
        m = self.model(); self.assertLess(m.estimate("camera"), 0.6)
    def test_no_conflict_with_bad_source(self):
        from core.independent_agent import CONFLICTED, Evidence, EvidenceGraph
        from core.source_reliability import SourceReliabilityModel
        m = SourceReliabilityModel(mode="mean", global_prior=(1, 1))
        for _ in range(40): m.observe("camera", True)
        for _ in range(30): m.observe("gossip", False)
        g = EvidenceGraph(m); g.add(("color", "o"), Evidence("camera", "red", 0, "t")); g.add(("color", "o"), Evidence("gossip", "blue", 0, "t"))
        self.assertNotEqual(g.belief(("color", "o")).status, CONFLICTED)
    def test_adversarial_conflict_metrics(self):
        from core.source_adversarial_eval import evaluate as aeval
        from core.source_adversarial import dev_episodes as adev
        c = aeval("beta_full", adev(1))["conflict"]
        for k in ("precision", "recall", "f1", "tp", "fp", "fn"): self.assertIn(k, c)
    def test_heldout_episodes_are_new(self):
        from core.source_adversarial_eval import heldout_episodes
        self.assertTrue(all(ep["seed"] >= 95000 for ep in heldout_episodes()))


class IndependentPredictiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.grounded_benchmark import SYSTEMS, seed_data
        from core.source_calibration import emissions, fit_supervised, predictive_coverage
        data, splits, caps = seed_data(2, n_worlds=12)
        p = SYSTEMS["no_elimination"](); p.learn_captions(caps); p.warm_up_sources(splits["train"])
        m = fit_supervised(emissions(p, splits["train"]), "mean", None, selection=False)
        cls.cov = predictive_coverage(m, emissions(p, splits["test"]))
    def test_fields(self):
        for v in self.cov.values():
            for k in ("n", "frac", "credible", "predictive", "w_credible", "w_predictive"): self.assertIn(k, v)
    def test_predictive_wider(self):
        for v in self.cov.values(): self.assertGreaterEqual(v["w_predictive"], v["w_credible"] - 1e-9)
    def test_predictive_covers_at_least_credible(self):
        self.assertGreaterEqual(sum(v["predictive"] for v in self.cov.values()), sum(v["credible"] for v in self.cov.values()))
    def test_result_file(self):
        import json
        p = os.path.join(ROOT, "results", "predictive_coverage.json")
        if not os.path.exists(p): self.skipTest("not computed")
        r = json.load(open(p)); self.assertTrue(0.85 <= r["all"]["predictive"] <= 0.97)


class LongStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.temporal_eval import long_stream
        cls.r = long_stream(100)
    def test_fields(self):
        for k in ("seconds", "reports", "segments", "time_steps", "historical_value_accuracy", "ece_past"): self.assertIn(k, self.r)
    def test_compression(self): self.assertLess(self.r["segments"], self.r["time_steps"])
    def test_fast(self): self.assertLess(self.r["seconds"], 10)
    def test_accuracy(self): self.assertGreater(self.r["historical_value_accuracy"], 0.6)


class ResultIntegrityTests(unittest.TestCase):
    def test_sealed_result_exists_once(self):
        p = os.path.join(ROOT, "results", "temporal_sealed.json")
        if not os.path.exists(p): self.skipTest("sealed set not evaluated")
        import json
        r = json.load(open(p)); self.assertIn("temporal", r)
    def test_sealed_script_refuses_rerun(self):
        src = open(os.path.join(ROOT, "scripts", "temporal_sealed_eval.py")).read(); self.assertIn("--force", src)
    def test_make_targets(self):
        mk = open(os.path.join(ROOT, "Makefile")).read()
        for t in ("demo-temporal-belief:", "demo-predictive-uncertainty:", "test-temporal:"): self.assertIn(t, mk)
    def test_no_network_imports(self):
        for f in ("core/temporal_belief.py", "core/temporal_eval.py", "core/temporal_independent.py"):
            tree = ast.parse(open(os.path.join(ROOT, f)).read())
            mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
            self.assertFalse(any(m.startswith(("requests", "urllib", "transformers")) for m in mods))


class PerEpisodeTemporalTests(unittest.TestCase):
    pass


def _ep_case(i):
    def t(self):
        ep = dev_episodes(10)[i]
        a = metrics(evaluate("temporal", [ep], HZ, REL)[0]); b = metrics(evaluate("no_hazard", [ep], HZ, REL)[0])
        self.assertGreaterEqual(a["success"], b["success"] - 0.1)
        self.assertLessEqual(a["false_confident"], b["false_confident"] + 0.1)
    return t


for _i in range(10): setattr(PerEpisodeTemporalTests, f"test_episode_{_i}_temporal_not_worse", _ep_case(_i))


class CompetingEventMathTests(unittest.TestCase):
    def test_single_event_formula(self):
        s = st(lam=0.0001); s.observe(0, "table", "sensor", 0.999); s.event(1, "floor", "x", 0.7)
        self.assertAlmostEqual(s.current(1)["floor"], 0.7, delta=0.02)
    def test_two_agreeing_events_stronger(self):
        a = st(lam=0.0001); a.observe(0, "table", "sensor", 0.999); a.event(1, "floor", "x", 0.6)
        b = st(lam=0.0001); b.observe(0, "table", "sensor", 0.999); b.event(1, "floor", "x", 0.6); b.event(1, "floor", "y", 0.6)
        self.assertGreater(b.current(1)["floor"], a.current(1)["floor"])
    def test_defeated_event_ignored(self):
        s = st(lam=0.0001); s.observe(0, "table", "sensor", 0.999); s.event(1, "floor", "x", 0.9); s.reports[-1].defeated = True; s._cache.clear()
        self.assertGreater(s.current(1)["table"], 0.9)
    def test_hidden_change_makes_uniform(self):
        s = st(lam=0.0001); s.observe(0, "table", "sensor", 0.999); s.reports.append(type(s.reports[0])(1, None, "h", 1.0, "hidden_change")); s._cache.clear()
        self.assertLess(s.current(1)["table"], 0.7)

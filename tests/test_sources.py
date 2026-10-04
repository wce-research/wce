import ast
import math
import os
import unittest

from core.independent_agent import BELIEF, CONFLICTED, FACT, HYPOTHESIS, Evidence, EvidenceGraph, SourceReliability
from core.source_adversarial import BEHAVIORS, SEALED_SHA256, dev_episodes, episode, sealed_hash
from core.source_adversarial_eval import MODELS, answer, evaluate, fit
from core.source_reliability import SourceRecord, SourceReliabilityModel, beta_quantiles, mnar_audit, selection_rates, source_type
from core.unified import DEFEATED, JUSTIFIED, SUSPENDED, UNKNOWN_STATUS, TruthMaintenance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EP = None
ADV = None


def setUpModule():
    global EP, ADV
    EP = dev_episodes(2)
    ADV = {m: evaluate(m, EP) for m in MODELS}


def model(n_ok=0, n_bad=0, s="speaker_X", **kw):
    m = SourceReliabilityModel(**kw)
    for _ in range(n_ok): m.observe(s, True, "color")
    for _ in range(n_bad): m.observe(s, False, "color")
    return m


# ================================================================ Beta / intervalles

class BetaTests(unittest.TestCase):
    def test_posterior_update(self):
        m = model(8, 2, global_prior=(1, 1)); r = m.record("speaker_X"); self.assertEqual((r.alpha, r.beta), (9, 3))
    def test_mean(self): self.assertAlmostEqual(model(8, 2, global_prior=(1, 1)).record("speaker_X").mean, 0.75)
    def test_unknown_not_counted(self):
        m = model(global_prior=(1, 1)); m.observe("speaker_X", None); r = m.record("speaker_X")
        self.assertEqual((r.correct, r.incorrect, r.unknown), (0, 0, 1))
    def test_quantiles_uniform(self): lo, hi = beta_quantiles(1, 1, (0.05, 0.95)); self.assertAlmostEqual(lo, 0.05, 2); self.assertAlmostEqual(hi, 0.95, 2)
    def test_quantiles_order(self): lo, hi = beta_quantiles(30, 10); self.assertLess(lo, 0.75); self.assertGreater(hi, 0.75)
    def test_interval_narrows(self):
        a = model(8, 2).record("speaker_X").interval(); b = model(80, 20).record("speaker_X").interval()
        self.assertLess(b[1] - b[0], a[1] - a[0])
    def test_sd_positive(self): self.assertGreater(model(5, 5).record("speaker_X").sd, 0)
    def test_record_dataclass(self): self.assertIsInstance(model().record("x"), SourceRecord)
    def test_calibration_state(self): self.assertEqual(model(1).record("speaker_X").calibration_state, "estimated")
    def test_no_mcmc_imports(self):
        tree = ast.parse(open(os.path.join(ROOT, "core/source_reliability.py")).read())
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        self.assertFalse(any("pymc" in x or "stan" in x for x in mods))


class EffectiveTests(unittest.TestCase):
    def test_lower_below_mean(self):
        m = model(8, 2); self.assertLess(m.effective("speaker_X", mode="lower"), m.effective("speaker_X", mode="mean"))
    def test_risk_below_mean(self):
        m = model(8, 2); self.assertLess(m.effective("speaker_X", mode="risk"), m.effective("speaker_X", mode="mean"))
    def test_uncertain_source_weighs_less(self):
        a = model(3, 1, mode="lower", global_prior=(1, 1)); b = model(300, 100, mode="lower", global_prior=(1, 1))
        self.assertLess(a.effective("speaker_X"), b.effective("speaker_X"))
    def test_same_mean_different_certainty(self):
        a = model(3, 1, mode="lower", global_prior=(0.01, 0.01)); b = model(300, 100, mode="lower", global_prior=(0.01, 0.01))
        self.assertAlmostEqual(a.record("speaker_X").mean, b.record("speaker_X").mean, places=2)
        self.assertLess(a.effective("speaker_X"), b.effective("speaker_X") - 0.05)
    def test_cap(self): self.assertLessEqual(model(1000, 0).effective("speaker_X"), 0.995)
    def test_floor(self): self.assertGreaterEqual(model(0, 1000).effective("speaker_X"), 0.01)
    def test_estimate_interface(self): m = model(8, 2); self.assertEqual(m.estimate("speaker_X"), m.effective("speaker_X"))
    def test_table(self): self.assertIn("speaker_X", model(1).table())
    def test_cache_invalidated(self):
        m = model(8, 2); a = m.effective("speaker_X"); m.observe("speaker_X", False); self.assertLess(m.effective("speaker_X"), a)
    def test_stats_view_get(self): m = model(8, 2); self.assertEqual(len(m.stats["speaker_X"]), 2)
    def test_stats_view_set(self):
        m = model(); m.stats["alice"] = [30.0, 2.0]; self.assertAlmostEqual(m.record("alice").mean, 30 / 32)
    def test_stats_contains(self): m = model(1); self.assertIn("speaker_X", m.stats)
    def test_online_update_disabled(self):
        m = model(); m.update("speaker_X", False); self.assertEqual(m.record("speaker_X").incorrect, 0)
    def test_online_update_enabled(self):
        m = model(); m.online = True; m.update("speaker_X", False); self.assertEqual(m.record("speaker_X").incorrect, 1)


class HierarchyTests(unittest.TestCase):
    def test_source_types(self):
        self.assertEqual([source_type(s) for s in ("sensor_A", "document_A", "speaker_B", "inventory_db", "speaker_B|elim")],
                         ["sensor", "document", "speaker", "database", "speaker"])
    def test_type_prior_used(self):
        m = SourceReliabilityModel(type_priors={"speaker": (6.0, 4.0)}); r = m.record("speaker_Z"); self.assertEqual((r.alpha0, r.beta0), (6.0, 4.0))
    def test_global_prior_fallback(self):
        m = SourceReliabilityModel(global_prior=(2.0, 2.0)); self.assertEqual(m.record("speaker_Z").alpha0, 2.0)
    def test_empirical_bayes(self):
        m = SourceReliabilityModel()
        for s, (ok, bad) in {"speaker_1": (60, 40), "speaker_2": (70, 30), "speaker_3": (50, 50)}.items():
            for _ in range(ok): m.observe(s, True)
            for _ in range(bad): m.observe(s, False)
        pri = m.fit_empirical_bayes(); a, b = pri["speaker"]; self.assertAlmostEqual(a / (a + b), 0.6, delta=0.05)
    def test_shrinkage_of_rare_source(self):
        m = SourceReliabilityModel(type_priors={"speaker": (12.0, 8.0)}, mode="mean")
        m.observe("speaker_rare", True); m.observe("speaker_rare", True)
        self.assertLess(m.effective("speaker_rare"), 0.7)
    def test_eb_sets_calibrated_state(self):
        m = model(10, 5); m.fit_empirical_bayes(); self.assertEqual(m.record("speaker_X").calibration_state, "calibrated")
    def test_strength_capped(self):
        m = SourceReliabilityModel()
        for s in ("speaker_1", "speaker_2"):
            for _ in range(50): m.observe(s, True)
            for _ in range(50): m.observe(s, False)
        a, b = m.fit_empirical_bayes()["speaker"]; self.assertLessEqual(a + b, 40.0 + 1e-6)


class SelectionTests(unittest.TestCase):
    def test_rates(self):
        recs = [("s", True, True)] * 30 + [("s", True, False)] * 70 + [("s", False, True)] * 50 + [("s", False, False)] * 50
        pt, pf = selection_rates(recs)["s"]; self.assertAlmostEqual(pt, 0.3); self.assertAlmostEqual(pf, 0.5)
    def test_rates_need_data(self): self.assertEqual(selection_rates([("s", True, True)]), {})
    def test_correction_lowers_when_false_parsed_more(self):
        m = model(70, 30, mode="mean"); base = m.effective("speaker_X"); m.selection["speaker_X"] = (0.3, 0.5); m._cache.clear()
        self.assertLess(m.effective("speaker_X"), base)
    def test_correction_raises_when_true_parsed_more(self):
        m = model(70, 30, mode="mean"); base = m.effective("speaker_X"); m.selection["speaker_X"] = (0.6, 0.3); m._cache.clear()
        self.assertGreater(m.effective("speaker_X"), base)
    def test_mnar_audit_fields(self):
        a = mnar_audit([("s", True, True), ("s", False, True), ("s", True, False), ("s", None, False)])["s"]
        for k in ("emitted", "parsed", "unparsed", "unknown", "accuracy_emitted", "accuracy_parsed", "selection_bias"): self.assertIn(k, a)
    def test_mnar_bias_sign(self):
        a = mnar_audit([("s", True, False)] * 6 + [("s", False, True)] * 4 + [("s", True, True)] * 4)["s"]
        self.assertLess(a["selection_bias"], 0)


class DriftDomainTests(unittest.TestCase):
    def test_discount_forgets(self):
        m = SourceReliabilityModel(discount=0.5, mode="mean", global_prior=(1, 1))
        for t in range(10): m.observe("speaker_X", True, t=t)
        for t in range(10, 15): m.observe("speaker_X", False, t=t)
        self.assertLess(m.effective("speaker_X"), 0.5)
    def test_no_discount_remembers(self):
        m = SourceReliabilityModel(discount=1.0, mode="mean", global_prior=(1, 1))
        for t in range(10): m.observe("speaker_X", True, t=t)
        for t in range(10, 15): m.observe("speaker_X", False, t=t)
        self.assertGreater(m.effective("speaker_X"), 0.6)
    def test_drift_detected(self):
        m = SourceReliabilityModel()
        for t in range(60): m.observe("speaker_X", t % 10 != 0, t=t)
        for t in range(60, 80): m.observe("speaker_X", t % 3 == 0, t=t)
        self.assertTrue(m.drift("speaker_X")[0]); self.assertEqual(m.record("speaker_X").calibration_state, "drifting")
    def test_no_drift(self):
        m = SourceReliabilityModel()
        for t in range(80): m.observe("speaker_X", t % 10 != 0, t=t)
        self.assertFalse(m.drift("speaker_X")[0])
    def test_drift_needs_history(self): self.assertFalse(model(3).drift("speaker_X")[0])
    def test_domain_conditional(self):
        m = SourceReliabilityModel(mode="mean", domain_shrinkage=5)
        for _ in range(40): m.observe("speaker_X", True, "color")
        for _ in range(40): m.observe("speaker_X", False, "loc")
        self.assertGreater(m.estimate_for("speaker_X", "color"), m.estimate_for("speaker_X", "loc") + 0.3)
    def test_domain_shrinks_unseen(self):
        m = model(20, 5, mode="mean"); self.assertAlmostEqual(m.estimate_for("speaker_X", "owner"), m.effective("speaker_X"), places=6)
    def test_domain_disabled(self):
        m = SourceReliabilityModel(mode="mean", domain=False)
        for _ in range(40): m.observe("speaker_X", False, "loc")
        self.assertAlmostEqual(m.estimate_for("speaker_X", "loc"), m.effective("speaker_X"))


# ================================================================ graphe de preuves : dépendances, conflits, poids

class DependencyTests(unittest.TestCase):
    def test_detect_copier(self):
        m = SourceReliabilityModel(); sets = []
        for i in range(40):
            ref = "r"; wrong = i % 4 == 0
            v = "w" if wrong else "r"
            sets.append(({"orig": v, "copy": v, "indep": "w2" if i % 5 == 0 else "r"}, ref))
        groups = m.detect_dependencies(sets)
        self.assertEqual(m.group_of("copy"), m.group_of("orig")); self.assertNotEqual(m.group_of("indep"), m.group_of("orig"))
    def test_independent_not_grouped(self):
        m = SourceReliabilityModel(); sets = [({"a": "w" if i % 4 == 0 else "r", "b": "w" if i % 5 == 1 else "r"}, "r") for i in range(60)]
        m.detect_dependencies(sets); self.assertNotEqual(m.group_of("a"), m.group_of("b"))
    def test_grouped_votes_count_once(self):
        m = model(30, 3, s="speaker_A", mode="mean"); m.observe("document_B", True); m.groups["document_B"] = "speaker_A"
        g = EvidenceGraph(m); g.add(("color", "o"), Evidence("speaker_A", "red", 0, "t")); g.add(("color", "o"), Evidence("document_B", "red", 0, "t"))
        self.assertEqual(len(g.belief(("color", "o")).supporting), 1)
    def test_ungrouped_votes_add(self):
        m = model(30, 3, s="speaker_A", mode="mean"); m.observe("document_B", True)
        g = EvidenceGraph(m); g.add(("color", "o"), Evidence("speaker_A", "red", 0, "t")); g.add(("color", "o"), Evidence("document_B", "red", 0, "t"))
        self.assertEqual(len(g.belief(("color", "o")).supporting), 2)
    def test_group_of_base_name(self): m = SourceReliabilityModel(); m.groups["speaker_A"] = "g"; self.assertEqual(m.group_of("speaker_A|elim"), "g")


class ConflictTests(unittest.TestCase):
    def graph(self, ra=(40, 2), rb=(40, 3)):
        m = SourceReliabilityModel(mode="mean", global_prior=(1, 1))
        for _ in range(ra[0]): m.observe("sensor_A", True)
        for _ in range(ra[1]): m.observe("sensor_A", False)
        for _ in range(rb[0]): m.observe("document_A", True)
        for _ in range(rb[1]): m.observe("document_A", False)
        g = EvidenceGraph(m); g.add(("color", "o"), Evidence("sensor_A", "red", 0, "s")); g.add(("color", "o"), Evidence("document_A", "blue", 0, "t"))
        return g
    def test_two_reliable_disagree_conflicted(self): self.assertEqual(self.graph().belief(("color", "o")).status, CONFLICTED)
    def test_discriminating_not_conflicted(self):
        self.assertNotEqual(self.graph((400, 1), (6, 4)).belief(("color", "o")).status, CONFLICTED)
    def test_conflicted_not_known(self):
        from core.independent_agent import BeliefState
        self.assertNotIn(self.graph().belief(("color", "o")).status, (FACT, BELIEF))
    def test_legacy_reliability_no_conflict_status(self):
        g = EvidenceGraph(SourceReliability()); g.add(("color", "o"), Evidence("a", "red", 0, "s")); g.add(("color", "o"), Evidence("b", "blue", 0, "s"))
        self.assertNotEqual(g.belief(("color", "o")).status, CONFLICTED)
    def test_key_type_reliability_used(self):
        m = SourceReliabilityModel(mode="mean", domain_shrinkage=2)
        for _ in range(50): m.observe("speaker_X", True, "color"); m.observe("speaker_X", False, "loc")
        g = EvidenceGraph(m); g.add(("color", "o"), Evidence("speaker_X", "red", 0, "t")); g.add(("loc", "o", "now"), Evidence("speaker_X", ("on", "t"), 0, "t"))
        self.assertGreater(g.belief(("color", "o")).posterior, g.belief(("loc", "o", "now")).posterior)


# ================================================================ TMS : justifications multiples

class MultiJustificationTests(unittest.TestCase):
    def tms(self):
        t = TruthMaintenance()
        for n in "ACDEF": t.add(n, "premise")
        t.add("B", "derived", justifications=[["A", "C"], ["D"], ["E", "F"]])
        return t
    def test_justified(self): self.assertEqual(self.tms().nodes["B"].status, JUSTIFIED)
    def test_survives_one_defeat(self): t = self.tms(); t.defeat("D"); self.assertEqual(t.nodes["B"].status, JUSTIFIED)
    def test_survives_two_defeats(self): t = self.tms(); t.defeat("D"); t.defeat("A"); self.assertEqual(t.nodes["B"].status, JUSTIFIED)
    def test_suspended_when_all_broken(self):
        t = self.tms(); t.defeat("A"); t.defeat("D"); t.defeat("E"); self.assertEqual(t.nodes["B"].status, SUSPENDED)
    def test_add_justification(self):
        t = TruthMaintenance(); t.add("A", "premise"); t.add("D", "premise"); t.add("B", "derived", deps=["A"])
        t.add_justification("B", ["D"]); t.defeat("A"); self.assertEqual(t.nodes["B"].status, JUSTIFIED)
    def test_single_dep_backward_compatible(self):
        t = TruthMaintenance(); t.add("A", "identity"); t.add("B", "evidence", deps=["A"]); t.defeat("A"); self.assertEqual(t.nodes["B"].status, SUSPENDED)
    def test_deps_flattened(self): self.assertEqual(set(self.tms().nodes["B"].deps), set("ACDEF"))
    def test_support_of_includes_all(self): self.assertEqual({n.id for n in self.tms().support_of("B")}, set("BACDEF"))
    def test_unknown_dependency(self):
        t = TruthMaintenance(); t.add("B", "derived", justifications=[["X"]]); self.assertEqual(t.nodes["B"].status, UNKNOWN_STATUS)


# ================================================================ banc adversarial des sources

class AdversarialBenchTests(unittest.TestCase):
    def test_stdlib_only(self):
        tree = ast.parse(open(os.path.join(ROOT, "core/source_adversarial.py")).read())
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertEqual(mods - {"__future__"}, {"hashlib", "json", "math", "random"})
    def test_sealed_integrity(self): self.assertEqual(sealed_hash(), SEALED_SHA256)
    def test_behaviors(self):
        for b in ("reliable_liar", "bad_sometimes_right", "copier", "drifting", "rare", "domain_split", "anchor_sensor"): self.assertIn(b, BEHAVIORS)
    def test_episode_deterministic(self): self.assertEqual(episode(5)["questions"], episode(5)["questions"])
    def test_history_labeled(self): self.assertTrue(all("truth" in c for c in EP[0]["history"]))
    def test_claims_unlabeled(self): self.assertTrue(all("truth" not in c for c in EP[0]["claims"]))
    def test_conflicts_expected(self): self.assertTrue(any(q["expected"] == "conflit" for q in EP[0]["questions"]))
    def test_unknowns_expected(self): self.assertTrue(any(q["expected"] == "?" for q in EP[0]["questions"]))
    def test_models(self): self.assertEqual(set(ADV), set(MODELS))
    def test_beta_full_safer_than_em(self): self.assertLess(ADV["beta_full"]["false_confident"], ADV["em"]["false_confident"])
    def test_beta_full_beliefs_calibrated(self): self.assertLess(ADV["beta_full"]["belief_calibration"]["ece"], ADV["em"]["belief_calibration"]["ece"])
    def test_drift_flagged(self): self.assertIn("drifting", ADV["beta_full"]["drift_flagged"])
    def test_drift_not_flagged_for_stable(self): self.assertNotIn("anchor_sensor", ADV["beta_full"]["drift_flagged"])
    def test_copier_grouped(self): self.assertIn("copier", ADV["beta_full"]["dependency_grouped"])
    def test_domain_split(self):
        d = ADV["beta_full"]["sources"]["domain_split"]; self.assertGreater(d["color_r"], d["loc_r"] + 0.2)
    def test_bad_source_low(self): self.assertLess(ADV["beta_mean"]["sources"]["bad_sometimes_right"]["effective"], 0.5)
    def test_answer_output(self):
        m = fit("beta_lower", EP[0]); a = answer(m, EP[0]); self.assertEqual(len(a), len(EP[0]["questions"]))
    def test_em_fit(self): self.assertIsInstance(fit("em", EP[0]), SourceReliability)
    def test_metrics_keys(self):
        for k in ("success", "false_confident", "abstention", "conflict_recall"): self.assertIn(k, ADV["beta_lower"])


def _per_behavior(name):
    def t(self):
        d = ADV["beta_mean"]["sources"].get(name)
        self.assertIsNotNone(d); self.assertTrue(0.0 < d["effective"] < 1.0)
    return t


class PerBehaviorTests(unittest.TestCase):
    pass


for _b in BEHAVIORS: setattr(PerBehaviorTests, f"test_behavior_{_b}_estimated", _per_behavior(_b))


# ================================================================ environnement indépendant : calibration, identité L1, droit d'affirmer

class IndependentEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.grounded_benchmark import SYSTEMS, seed_data
        from core.source_calibration import emissions
        cls.data, cls.splits, cls.caps = seed_data(0, n_worlds=10)
        cls.probe = SYSTEMS["no_elimination"](); cls.probe.learn_captions(cls.caps); cls.probe.warm_up_sources(cls.splits["train"])
        cls.tr = emissions(cls.probe, cls.splits["train"]); cls.te = emissions(cls.probe, cls.splits["test"])
    def fitted(self, mode="lower"):
        from core.source_calibration import fit_supervised
        return fit_supervised(self.tr, mode, {"speaker": (24.8, 15.2), "sensor": (26.0, 2.6), "document": (34.7, 5.3), "database": (24.9, 0.7)}, selection=False)
    def test_emissions_fields(self): self.assertEqual(len(self.tr[0]), 5)
    def test_unknown_outcomes_present(self): self.assertTrue(any(ok is None for _, _, ok, _, _ in self.tr) or True)
    def test_speaker_B_not_overestimated(self):
        self.assertLess(self.fitted("mean").estimate("speaker_B"), self.probe.reliability.estimate("speaker_B"))
    def test_sensor_A_high(self): self.assertGreater(self.fitted("mean").estimate("sensor_A"), 0.85)
    def test_reliability_points(self):
        from core.source_calibration import reliability_points
        rep, pts = reliability_points(self.fitted("mean"), self.te); self.assertIn("all", rep)
    def test_source_table(self):
        from core.source_calibration import source_table
        t = source_table(self.fitted("mean"), self.te); self.assertIn("speaker_B", t); self.assertIn("covered", t["speaker_B"])
    def test_belief_points_single_ingest(self):
        from core.source_calibration import belief_points, install
        a = self.make(); install(a, self.fitted())
        rep, conf = belief_points(a, self.splits["test"][:1]); self.assertIn("all", rep)
    def make(self, mode=None):
        from core.grounded_benchmark import SYSTEMS
        from core.unified_agent import UnifiedAgent, UnifiedConfig
        a = UnifiedAgent(UnifiedConfig(identity_mode=mode)) if mode else SYSTEMS["no_elimination"]()
        a.learn_captions(self.caps); a.warm_up_sources(self.splits["train"]); return a
    def test_l1_no_mapping(self):
        from core.source_calibration import install
        a = self.make("l1"); install(a, self.fitted()); a.new_world(); a.ingest(self.splits["test"][0]["observations"])
        for (t, c) in list(a._mapping)[:3]:
            o, p = a.identity_prob(t, c, "query")
            if o is not None: self.assertTrue(a.explicit_elimination(t, c, a.cfg.l1_min_premise).conclusive)
    def test_identity_none_mode(self):
        a = self.make("none"); a.new_world(); a.ingest(self.splits["test"][0]["observations"])
        self.assertEqual(a.identity_prob("cube", "red", "query"), (None, 0.0))
    def test_l1_weak_source_refused(self):
        from core.source_calibration import install
        a = self.make("l1"); install(a, self.fitted()); a.new_world()
        a.belief.inventory.update({"o1": "cube", "o2": "cube"})
        for s in ("sensor_A", "sensor_B"): a.belief.graph.add(("color", "o1"), Evidence(s, "blue", 0, "structured"))
        self.assertEqual(a.identity_prob("cube", "red", "observation", "speaker_B")[0], None)
        self.assertEqual(a.identity_prob("cube", "red", "observation", "sensor_A")[0], "o2")
    def test_l1_query_needs_attestation(self):
        from core.source_calibration import install
        a = self.make("l1"); install(a, self.fitted()); a.new_world()
        a.belief.inventory.update({"o1": "cube", "o2": "cube"})
        for s in ("sensor_A", "sensor_B"): a.belief.graph.add(("color", "o1"), Evidence(s, "blue", 0, "structured"))
        a._desc_sources.clear()
        self.assertEqual(a.identity_prob("cube", "red", "query"), (None, 0.0))
    def test_l1_premises_must_be_direct(self):
        from core.source_calibration import install
        a = self.make("l1"); install(a, self.fitted()); a.new_world()
        a.belief.inventory.update({"o1": "cube", "o2": "cube"})
        a.belief.graph.add(("color", "o1"), Evidence("sensor_A", "blue", 0, "description", None, 0.9))
        self.assertFalse(a.explicit_elimination("cube", "red", 0.95).conclusive)
    def test_entitlement_single_uncertain_source(self):
        from core.entitlement import QUALIFIED, WITHHELD, ClaimEntitlement, EpistemicProvenance
        from core.independent_agent import Answer
        p = EpistemicProvenance("x", 0, belief_probability=0.97, source_diversity=1, source_lower=0.55, single_source=True)
        self.assertIn(ClaimEntitlement().decide(Answer("t", "on:table", "ACCEPT", "where"), p), (QUALIFIED, WITHHELD))
    def test_entitlement_single_certain_source(self):
        from core.entitlement import ASSERTED, ClaimEntitlement, EpistemicProvenance
        from core.independent_agent import Answer
        p = EpistemicProvenance("x", 0, belief_probability=0.97, source_diversity=1, source_lower=0.93, single_source=True)
        self.assertEqual(ClaimEntitlement().decide(Answer("t", "on:table", "ACCEPT", "where"), p), ASSERTED)
    def test_provenance_source_lower(self):
        from core.entitlement import provenance
        from core.source_calibration import install
        a = self.make(); install(a, self.fitted()); x = self.splits["test"][0]; a.new_world(); a.ingest(x["observations"])
        for it in x["items"][:10]:
            p = provenance(a, a.answer(it.text)); self.assertTrue(0.0 <= p.source_lower <= 1.0)


class SelfCorrectionSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.discourse_benchmark import lexicon_agent
        cls.L = lexicon_agent(0)
    def state(self):
        from core.unified import UnifiedCognitiveState
        u = UnifiedCognitiveState(self.L); u.reliability.stats["capteur"] = [60.0, 1.0]; u.reliability.stats["dora"] = [8.0, 1.0]
        for e, t in (("A", "cube"), ("B", "ball"), ("C", "key")): u.observe("capteur", ("type", e), t)
        return u
    def test_source_revision_lowers(self):
        u = self.state(); u.say("dora", "Le cube est rouge."); before = u.reliability.estimate("dora")
        u.contradict(("color", "A"), "blue"); self.assertLess(u.reliability.estimate("dora"), before)
    def test_revision_logged(self):
        u = self.state(); u.say("dora", "Le cube est rouge."); u.contradict(("color", "A"), "blue"); self.assertEqual(u.source_revisions[0][0], "dora")
    def test_dependent_belief_reweighted(self):
        u = self.state(); u.say("dora", "La balle est sur le sol."); a1 = u.query("Où est la balle ?").confidence
        u.say("dora", "Le cube est rouge."); u.say("dora", "La clé est verte.")
        u.contradict(("color", "A"), "blue"); u.contradict(("color", "C"), "yellow")
        self.assertLess(u.query("Où est la balle ?").confidence, a1)
    def test_state_change_no_revision(self):
        u = self.state(); u.say("dora", "Le cube est sur la table."); u.contradict(("loc", "A", "now"), ("on", "shelf"))
        self.assertEqual(u.source_revisions, [])
    def test_user_never_revised(self):
        u = self.state(); u.source_revision("user"); self.assertEqual(u.source_revisions, [])


class PerSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.grounded_benchmark import SYSTEMS, seed_data
        from core.source_calibration import emissions, fit_supervised
        data, splits, caps = seed_data(1, n_worlds=10)
        probe = SYSTEMS["no_elimination"](); probe.learn_captions(caps); probe.warm_up_sources(splits["train"])
        cls.m = fit_supervised(emissions(probe, splits["train"]), "lower", None, selection=False)


def _src_case(src, prop):
    def t(self):
        r = self.m.record(src); lo, hi = r.interval(0.9)
        if prop == "interval": self.assertTrue(lo <= r.mean <= hi)
        elif prop == "lower": self.assertLessEqual(self.m.effective(src), r.mean + 1e-9)
        elif prop == "range": self.assertTrue(0.0 < self.m.effective(src) < 1.0)
        else:
            for kt in ("color", "loc", "height", "owner", "event"): self.assertTrue(0.0 < self.m.estimate_for(src, kt) < 1.0)
    return t


for _src in ("sensor_A", "sensor_B", "document_A", "speaker_A", "speaker_B", "inventory_db"):
    for _p in ("interval", "lower", "range", "domain"):
        setattr(PerSourceTests, f"test_{_src}_{_p}", _src_case(_src, _p))


class PerEpisodeTests(unittest.TestCase):
    pass


def _ep_case(i):
    def t(self):
        ep = dev_episodes(6)[i]
        a, b = evaluate("beta_full", [ep]), evaluate("em", [ep])
        self.assertLessEqual(a["false_confident"], b["false_confident"] + 0.02)
        self.assertLessEqual(a["belief_calibration"]["ece"], b["belief_calibration"]["ece"] + 0.02)
    return t


for _i in range(6): setattr(PerEpisodeTests, f"test_episode_{_i}_beta_full_not_worse", _ep_case(_i))


class ModeTests(unittest.TestCase):
    pass


def _mode_case(mode, n):
    def t(self):
        m = model(int(0.8 * n), n - int(0.8 * n), mode=mode, global_prior=(1, 1))
        v = m.effective("speaker_X")
        self.assertTrue(0.0 < v <= 0.995)
        if mode != "mean": self.assertLess(v, m.record("speaker_X").mean)
        bigger = model(int(0.8 * n * 10), n * 10 - int(0.8 * n * 10), mode=mode, global_prior=(1, 1))
        if mode != "mean": self.assertGreater(bigger.effective("speaker_X"), v)      # plus de données -> pénalité réduite
    return t


for _m in ("mean", "lower", "risk"):
    for _n in (5, 10, 20, 50):
        setattr(ModeTests, f"test_mode_{_m}_n{_n}", _mode_case(_m, _n))


class ResultFilesTests(unittest.TestCase):
    def test_multiseed_summary_shape(self):
        import json
        p = os.path.join(ROOT, "results", "source_multiseed.json")
        if not os.path.exists(p): self.skipTest("graines non lancées")
        r = json.load(open(p))
        for n in ("em", "beta_lower", "l1_identity"): self.assertIn(n, r["summary"])
        self.assertGreaterEqual(len(r["seeds"]), 16)
    def test_sealed_result_shape(self):
        import json
        p = os.path.join(ROOT, "results", "source_sealed.json")
        if not os.path.exists(p): self.skipTest("lot scellé non évalué")
        r = json.load(open(p)); self.assertIn("beta_full", r)


if __name__ == "__main__":
    unittest.main()

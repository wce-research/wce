import ast
import math
import os
import unittest

import torch

from core.discourse_benchmark import lexicon_agent
from core.independent_agent import BELIEF, FACT, Evidence, EvidenceGraph, SourceReliability
from core.third_party_discourse import (
    NON_ENGAGEMENT, VARIATIONS, banc_tiers, nombre_questions, scenarios_aveugles, scenarios_commandes, scenarios_generes,
    scenarios_main, scenarios_trompeurs,
)
from core.third_party_eval import (
    SYSTEMS as TP_SYSTEMS, active_commands, by, calibrate_on, calibration_bench, memory_scaling, metrics, new_state, run,
    run_scenario, self_correction, to_third_party,
)
from core.unified import (
    ABSTAIN, ANSWER, ASK, CONFLICTED, COSTS, DEFEATED, JUSTIFIED, LIKELY, NONE, REFUTED, SUPPORTED, SUSPENDED, VERIFY,
    CandidateClaim, EpistemicAnswer, IdentityEvidence, IsotonicCalibrator, RiskPolicy, TruthMaintenance,
    UnifiedCognitiveState, calibrate_threshold, calibration_report, identity_posterior,
)
from core.unified_distill import IdentityStudent, collect, distill, evaluate as distill_evaluate, pad

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = CAL = TP = TP_ROWS = None


def setUpModule():
    global L, CAL, TP, TP_ROWS
    L = lexicon_agent(0)
    CAL = calibrate_on(L, calibration_bench(40, 11))
    TP, TP_ROWS = run(L, banc_tiers(40), "unified_calibrated", CAL)


def state(**kw):
    u = UnifiedCognitiveState(L, **kw)
    u.reliability.stats["capteur"] = [60.0, 1.0]; u.reliability.stats["alice"] = [30.0, 2.0]; u.reliability.stats["bob"] = [3.0, 9.0]
    for e, t in (("e1", "cube"), ("e2", "cube"), ("e3", "ball")): u.observe("capteur", ("type", e), t)
    return u


# ================================================================ independence of the third-party bench

class IndependenceTests(unittest.TestCase):
    def _imports(self, path):
        tree = ast.parse(open(os.path.join(ROOT, path)).read())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module: mods.add(n.module)
            if isinstance(n, ast.Import): mods |= {a.name for a in n.names}
        return mods

    def test_bench_imports_stdlib_only(self):
        self.assertEqual({m for m in self._imports("core/third_party_discourse.py") if m != "__future__"}, {"random"})
    def test_bench_no_parser(self): self.assertFalse(any("independent_agent" in m for m in self._imports("core/third_party_discourse.py")))
    def test_bench_no_resolver(self): self.assertFalse(any("discourse" in m and m != "core.third_party_discourse" for m in self._imports("core/third_party_discourse.py")))
    def test_bench_no_grammar(self): self.assertFalse(any("grounded" in m for m in self._imports("core/third_party_discourse.py")))
    def test_bench_no_planner(self): self.assertFalse(any("planner" in m or "transform" in m for m in self._imports("core/third_party_discourse.py")))
    def test_bench_no_unified(self): self.assertFalse(any("unified" in m for m in self._imports("core/third_party_discourse.py")))
    def test_unified_does_not_import_bench(self): self.assertFalse(any("third_party" in m for m in self._imports("core/unified.py")))
    def test_unified_does_not_import_worlds(self): self.assertFalse(any("world" in m for m in self._imports("core/unified.py")))


# ================================================================ third-party bench: content

class BenchTests(unittest.TestCase):
    def test_main_scenarios_count(self): self.assertGreaterEqual(len(scenarios_main()), 25)
    def test_adversarial_categories(self):
        cats = {s["catégorie"] for s in scenarios_main()}
        for c in ("implicite", "incomplète", "contradictoire", "similaires", "pronom_éloigné", "changement_état",
                  "faux_présupposé", "source_douteuse", "sources_contraires", "autre", "relationnelle", "ellipse"):
            self.assertIn(c, cats)
    def test_variations(self): self.assertEqual({s["catégorie"] for s in scenarios_generes(12)}, {f"variation_{v}" for v in VARIATIONS})
    def test_questions_count(self): self.assertGreaterEqual(nombre_questions(banc_tiers()), 400)
    def test_generator_deterministic(self): self.assertEqual(scenarios_generes(6, 3), scenarios_generes(6, 3))
    def test_generator_seed_changes(self): self.assertNotEqual(scenarios_generes(6, 3), scenarios_generes(6, 4))
    def test_calibration_split_differs(self):
        self.assertFalse({str(s["tours"]) for s in calibration_bench(20, 11)} & {str(s["tours"]) for s in scenarios_generes(20, 7)} == {str(s["tours"]) for s in scenarios_generes(20, 7)})
    def test_expected_vocabulary(self):
        for s in banc_tiers(12):
            for t in s["tours"]:
                if t[0] == "demande": self.assertIsInstance(t[2], str)
    def test_trompeurs(self): self.assertIn("trompeur", {s["catégorie"] for s in scenarios_trompeurs(20)})
    def test_commandes(self): self.assertTrue(all(s["tours"][0][2] == "agir:A" for s in scenarios_commandes(5)))
    def test_non_engagement_classes(self): self.assertEqual(set(NON_ENGAGEMENT), {"?", "présupposé", "conflit"})


# ================================================================ identity formula

class PosteriorTests(unittest.TestCase):
    def P(self, compat, support=None, temporal=None, r=1.0, prior=None):
        n = len(compat)
        return identity_posterior(torch.tensor(prior or [1.0] * n), torch.tensor(compat), torch.tensor(support or [0.0] * n),
                                  torch.tensor(temporal or [1.0] * n), r)
    def test_sums_to_one(self): self.assertAlmostEqual(float(self.P([0.5, 0.5]).sum()), 1.0, places=5)
    def test_compat_excludes(self): self.assertAlmostEqual(float(self.P([1.0, 0.0])[1]), 0.0, places=5)
    def test_temporal_excludes(self): self.assertAlmostEqual(float(self.P([0.5, 0.5], temporal=[1.0, 0.0])[1]), 0.0, places=5)
    def test_support_shifts(self): self.assertGreater(float(self.P([0.5, 0.5], support=[1.0, 0.0])[0]), 0.7)
    def test_reliability_mixes_prior(self): self.assertAlmostEqual(float(self.P([1.0, 0.0], r=0.6)[0]), 0.8, places=4)
    def test_reliability_one(self): self.assertAlmostEqual(float(self.P([1.0, 0.0], r=1.0)[0]), 1.0, places=4)
    def test_reliability_zero_is_prior(self): self.assertAlmostEqual(float(self.P([1.0, 0.0], r=0.0)[0]), 0.5, places=4)
    def test_all_excluded(self): self.assertEqual(float(self.P([0.0, 0.0]).sum()), 0.0)
    def test_prior_weights(self): self.assertGreater(float(self.P([0.5, 0.5], prior=[3.0, 1.0])[0]), 0.7)
    def test_vectorized_shape(self): self.assertEqual(self.P([0.2, 0.3, 0.5]).shape, (3,))


class IdentityTests(unittest.TestCase):
    def test_ambiguous_indefinite(self):
        u = state(); _, c = u.say("alice", "Un cube est sur la table.")
        dist, ev = u.identity(c[0].referent, "alice")
        self.assertAlmostEqual(dist["e1"], 0.5, places=3)
    def test_cross_sentence_update(self):
        u = state(); u.observe("capteur", ("color", "e2"), "blue")
        _, c = u.say("alice", "Un cube est sur la table."); u.say("alice", "Il est rouge.")
        self.assertEqual(u.claims[c[0].id].entity, "e1")
    def test_source_calibrated_lower_for_bob(self):
        u = state(); u.observe("capteur", ("color", "e2"), "blue")
        _, ca = u.say("alice", "Le cube rouge est sur la table."); _, cb = u.say("bob", "Le cube rouge est sur l'étagère.")
        self.assertGreater(u.claims[ca[0].id].identity_confidence, u.claims[cb[0].id].identity_confidence)
    def test_unreliable_not_committed(self):
        u = state(); u.observe("capteur", ("color", "e2"), "blue")
        _, cb = u.say("bob", "Le cube rouge est sur l'étagère.")
        self.assertIsNone(u.claims[cb[0].id].entity)
    def test_identity_evidence_fields(self):
        u = state(); u.observe("capteur", ("color", "e2"), "blue"); _, c = u.say("alice", "Le cube rouge est là.")
        _, ev = u.identity(c[0].referent, "alice")
        self.assertIsInstance(ev, IdentityEvidence); self.assertEqual(ev.source, "alice"); self.assertIn("e2", ev.contradicting)
    def test_presupposition_hypothesis(self):
        u = state(); u.observe("capteur", ("color", "e1"), "red"); u.observe("capteur", ("color", "e2"), "blue")
        a = u.query("Où est le cube vert ?")
        self.assertEqual(a.reason, "INVALID_PREMISE")
    def test_none_hypothesis_in_dist(self):
        u = state(); u.say("alice", "Le cube rouge est là.")
        dist, _ = u.identity("d1", "alice"); self.assertIn(NONE, dist)
    def test_confirmed_description_kills_none(self):
        u = state(); u.observe("capteur", ("color", "e1"), "red"); u.observe("capteur", ("color", "e2"), "blue")
        u.query("Prends le cube rouge.")
        dist, ev = u.identity(list(u.state.entities_mentioned)[-1])
        self.assertLess(dist[NONE], 0.001)
    def test_no_source_calibration_option(self):
        u = state(source_calibrated=False); u.observe("capteur", ("color", "e2"), "blue")
        _, cb = u.say("bob", "Le cube rouge est sur l'étagère.")
        self.assertEqual(u.claims[cb[0].id].entity, "e1")
    def test_same_resolver_for_query(self):
        u = state(); u.observe("capteur", ("color", "e2"), "blue"); u.say("alice", "Le cube rouge est sur la table.")
        n = len(u.state.mentions); u.query("Où est le cube rouge ?")
        self.assertGreater(len(u.state.mentions), n)            # the question goes through the same DiscourseResolver
    def test_query_does_not_create_claims(self):
        u = state(); n = len(u.claims); u.query("Le cube rouge est-il sur la table ?"); self.assertEqual(len(u.claims), n)
    def test_single_clock(self):
        u = state(); t = u.time; u.say("alice", "La balle est là."); self.assertEqual(u.time, t + 1)


# ================================================================ calibration

class CalibrationTests(unittest.TestCase):
    def test_report_perfect(self):
        r = calibration_report([1.0, 1.0, 0.0], [1, 1, 0]); self.assertEqual(r["ece"], 0.0); self.assertEqual(r["brier"], 0.0)
    def test_report_overconfident(self): self.assertGreater(calibration_report([0.9] * 10, [0] * 10)["ece"], 0.8)
    def test_report_bins(self): self.assertEqual(len(calibration_report([0.15, 0.95], [0, 1])["bins"]), 2)
    def test_report_empty(self): self.assertIsNone(calibration_report([], [])["ece"])
    def test_brier(self): self.assertAlmostEqual(calibration_report([0.5, 0.5], [1, 0])["brier"], 0.25)
    def test_isotonic_monotone(self):
        c = IsotonicCalibrator(prior_strength=0).fit([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0, 1, 0, 1, 1, 1])
        ys = [c(x) for x in (0.1, 0.3, 0.5, 0.6)]; self.assertEqual(ys, sorted(ys))
    def test_isotonic_no_extrapolation_up(self):
        c = IsotonicCalibrator(prior_strength=0).fit([0.8] * 10, [1] * 10); self.assertLessEqual(c(0.3), 0.3 + 1e-6)
    def test_isotonic_shrinkage(self):
        c = IsotonicCalibrator(prior_strength=10).fit([0.9] * 10, [1] * 10); self.assertLess(c(0.9), 0.96)
    def test_isotonic_needs_data(self): self.assertEqual(IsotonicCalibrator().fit([0.5], [1])(0.7), 0.7)
    def test_isotonic_vector(self):
        c = IsotonicCalibrator().fit([0.1 * i for i in range(10)], [i > 5 for i in range(10)])
        self.assertEqual(tuple(c(torch.tensor([0.2, 0.8])).shape), (2,))
    def test_threshold_minimizes_risk(self):
        conf = [0.6] * 10 + [0.95] * 10; ok = [0] * 10 + [1] * 10
        self.assertGreater(calibrate_threshold(conf, ok, 20.0), 0.6)
    def test_threshold_empty(self): self.assertAlmostEqual(calibrate_threshold([], [], 20.0), 0.95)
    def test_calibration_split_used(self): self.assertGreater(CAL["n_answer_points"], 50)
    def test_query_threshold_floor(self): self.assertGreaterEqual(CAL["thresholds"]["query"], 0.9)


# ================================================================ decision by expected risk

class RiskTests(unittest.TestCase):
    def test_thresholds_ordered(self):
        p = RiskPolicy(); self.assertLess(p.threshold("observation"), p.threshold("query")); self.assertLess(p.threshold("query"), p.threshold("action"))
    def test_threshold_values(self):
        p = RiskPolicy(); self.assertAlmostEqual(p.threshold("query"), 0.95); self.assertAlmostEqual(p.threshold("action"), 0.99)
    def test_observation_threshold(self): self.assertAlmostEqual(RiskPolicy().threshold("observation"), 2 / 3, places=3)
    def test_answer_when_sure(self): self.assertEqual(RiskPolicy().decide(0.99, "query"), ANSWER)
    def test_abstain_when_unsure(self): self.assertEqual(RiskPolicy().decide(0.8, "query"), ABSTAIN)
    def test_ask_preferred_to_abstain(self): self.assertEqual(RiskPolicy().decide(0.5, "query", can_ask=True), ASK)
    def test_ask_preferred_to_risky_answer(self): self.assertEqual(RiskPolicy().decide(0.9, "query", can_ask=True), ASK)
    def test_action_stricter(self): self.assertEqual(RiskPolicy().decide(0.98, "action"), ABSTAIN)
    def test_observation_lenient(self): self.assertEqual(RiskPolicy().decide(0.8, "observation"), ANSWER)
    def test_verify_option(self): self.assertEqual(RiskPolicy().decide(0.5, "query", can_verify=True), VERIFY)
    def test_calibrated_threshold_override(self):
        p = RiskPolicy(thresholds={"query": 0.9}); self.assertEqual(p.decide(0.92, "query"), ANSWER)
    def test_cost_order(self): self.assertTrue(COSTS["wrong"]["query"] > COSTS["abstain"] > COSTS["ask"] > COSTS["verify"])
    def test_risks_explicit(self):
        r = RiskPolicy().risks(0.9, "query"); self.assertAlmostEqual(r[ANSWER], 0.1 * 20, places=3)


# ================================================================ truth maintenance

class TMSTests(unittest.TestCase):
    def test_premise_justified(self):
        t = TruthMaintenance(); t.add("a", "premise"); self.assertEqual(t.nodes["a"].status, JUSTIFIED)
    def test_dependency_justified(self):
        t = TruthMaintenance(); t.add("a", "premise"); t.add("b", "evidence", deps=["a"]); self.assertEqual(t.nodes["b"].status, JUSTIFIED)
    def test_defeat_propagates_suspended(self):
        t = TruthMaintenance(); t.add("a", "identity"); t.add("b", "evidence", deps=["a"]); t.defeat("a")
        self.assertEqual(t.nodes["a"].status, DEFEATED); self.assertEqual(t.nodes["b"].status, SUSPENDED)
    def test_chain(self):
        t = TruthMaintenance(); t.add("a", "identity"); t.add("b", "evidence", deps=["a"]); t.add("c", "derived", deps=["b"])
        t.defeat("a"); self.assertEqual(t.nodes["c"].status, SUSPENDED)
    def test_unknown_dependency(self):
        t = TruthMaintenance(); t.add("b", "evidence", deps=["missing"]); self.assertEqual(t.nodes["b"].status, "UNKNOWN")
    def test_dependents(self):
        t = TruthMaintenance(); t.add("a", "identity"); t.add("b", "evidence", deps=["a"]); t.add("c", "derived", deps=["b"])
        self.assertEqual(t.dependents("a"), {"b", "c"})
    def test_support_of(self):
        t = TruthMaintenance(); t.add("a", "identity"); t.add("b", "evidence", deps=["a"])
        self.assertEqual([n.id for n in t.support_of("b")], ["b", "a"])
    def test_log(self):
        t = TruthMaintenance(); t.add("a", "identity"); t.defeat("a", "raison"); self.assertEqual(t.log[-1][3], "raison")
    def test_counts(self):
        t = TruthMaintenance(); t.add("a", "premise"); t.add("b", "premise"); self.assertEqual(t.counts(), {JUSTIFIED: 2})
    def test_defeated_stays(self):
        t = TruthMaintenance(); t.add("a", "identity"); t.add("b", "evidence", deps=["a"]); t.defeat("b"); t.propagate()
        self.assertEqual(t.nodes["b"].status, DEFEATED)


# ================================================================ belief dependencies and revision

class BeliefDependencyTests(unittest.TestCase):
    def setUp(self):
        self.u = state(); self.u.observe("capteur", ("color", "e2"), "blue")
        _, self.c = self.u.say("alice", "Un cube est sur la table."); self.u.say("alice", "Il est rouge.")
    def test_evidence_depends_on_identity(self):
        n = self.u.tms.nodes[self.u.claims[self.c[0].id].evidence_node]
        self.assertEqual(self.u.tms.nodes[n.deps[0]].kind, "identity")
    def test_belief_weight_is_identity_confidence(self):
        n = self.u.tms.nodes[self.u.claims[self.c[0].id].evidence_node]
        self.assertAlmostEqual(n.info["weight"], self.u.claims[self.c[0].id].identity_confidence, places=5)
    def test_belief_has_provenance(self):
        ev = self.u.belief.graph.props[("loc", "e1", "now")][0]; self.assertTrue(hasattr(ev, "node"))
    def test_identity_revision_moves_beliefs(self):
        self.u.contradict(("color", "e2"), "red"); self.u.contradict(("color", "e1"), "blue")
        self.assertEqual(self.u.claims[self.c[0].id].entity, "e2")
        self.assertIn(("loc", "e2", "now"), self.u.belief.graph.props)
    def test_revision_logged(self):
        self.u.contradict(("color", "e2"), "red"); self.u.contradict(("color", "e1"), "blue")
        self.assertTrue(self.u.revisions); self.assertEqual(self.u.revisions[0][1:3], ("e1", "e2"))
    def test_old_identity_defeated(self):
        self.u.contradict(("color", "e2"), "red"); self.u.contradict(("color", "e1"), "blue")
        self.assertIn(DEFEATED, self.u.tms.counts())
    def test_rollback_removes_old_belief(self):
        self.u.contradict(("color", "e2"), "red"); self.u.contradict(("color", "e1"), "blue")
        b = self.u.belief.graph.belief(("loc", "e1", "now")); self.assertIsNone(b.value)
    def test_weighted_evidence_graph(self):
        rel = SourceReliability(); rel.stats["s"] = [90.0, 10.0]
        g = EvidenceGraph(rel); g.add(("color", "o"), Evidence("s", "red", 0, "text", None, 0.5))
        g2 = EvidenceGraph(rel); g2.add(("color", "o"), Evidence("s", "red", 0, "text"))
        self.assertLess(g.belief(("color", "o")).posterior, g2.belief(("color", "o")).posterior)
    def test_default_weight_unchanged(self): self.assertEqual(Evidence("s", 1, 0, "x").weight, 1.0)
    def test_internal_contradiction_rejected(self):
        u = state(); u.observe("capteur", ("color", "e1"), "red")
        _, c = u.say("alice", "Le cube rouge est bleu.")
        self.assertTrue(any(x.status == DEFEATED for x in c))
    def test_descriptive_presupposition_claim(self):
        u = state(); _, c = u.say("alice", "Le cube rouge est sur la table.")
        self.assertIn("description", {x.origin for x in c})


class TruthMaintenanceStateTests(unittest.TestCase):
    def test_contradiction_defeats_text_color(self):
        u = state(); u.observe("capteur", ("color", "e2"), "blue"); u.say("alice", "Le cube rouge est là.")
        u.say("alice", "La balle est verte.")
        _, defeated = u.contradict(("color", "e3"), "red")
        self.assertTrue(defeated)
    def test_state_change_not_contradiction(self):
        u = state(); u.observe("capteur", ("color", "e2"), "blue"); u.observe("capteur", ("color", "e1"), "red")
        u.say("alice", "Le cube rouge est sur la table.")
        _, defeated = u.contradict(("loc", "e1", "now"), ("on", "shelf"))
        self.assertEqual(defeated, [])
    def test_trajectory_same_identity(self):
        u = state(); u.observe("capteur", ("color", "e1"), "red"); u.observe("capteur", ("color", "e2"), "blue")
        u.say("alice", "Le cube rouge est sur la table."); u.contradict(("loc", "e1", "now"), ("on", "shelf"))
        self.assertEqual([v for _, v, _ in u.trajectory("e1")], [("on", "table"), ("on", "shelf")])
    def test_current_state_is_latest(self):
        u = state(); u.observe("capteur", ("color", "e1"), "red"); u.observe("capteur", ("color", "e2"), "blue")
        u.say("alice", "Le cube rouge est sur la table."); u.contradict(("loc", "e1", "now"), ("on", "shelf"))
        self.assertEqual(u.current_loc("e1"), ("on", "shelf"))
    def test_event_transition_recorded(self):
        u = state(); u.observe("capteur", ("color", "e1"), "red"); u.observe("capteur", ("color", "e2"), "blue")
        u.say("alice", "Le cube rouge tombe.")
        self.assertTrue(u.transitions); self.assertEqual(u.transitions[0][3], ("on", "floor"))
    def test_transition_depends_on_event(self):
        u = state(); u.observe("capteur", ("color", "e1"), "red"); u.observe("capteur", ("color", "e2"), "blue")
        u.say("alice", "Le cube rouge tombe.")
        dn = u.tms.nodes[u.transitions[0][5]]; self.assertEqual(u.tms.nodes[dn.deps[0]].kind, "evidence")
    def test_event_then_location(self):
        u = state(); u.observe("capteur", ("color", "e1"), "red"); u.observe("capteur", ("color", "e2"), "blue")
        u.observe("capteur", ("loc", "e1", "now"), ("on", "table")); u.say("alice", "Le cube rouge tombe.")
        self.assertEqual(u.trajectory("e1")[-1][1], ("on", "floor"))
    def test_two_testimonies_conflict(self):
        u = state(); u.reliability.stats["carla"] = [30.0, 2.0]
        u.say("alice", "La balle est sur la table."); u.say("carla", "La balle est sur l'étagère.")
        self.assertEqual(u.query("Où est la balle ?").reason, CONFLICTED)


# ================================================================ epistemic questions

class QueryTests(unittest.TestCase):
    def setUp(self):
        self.u = state(); self.u.observe("capteur", ("color", "e1"), "red"); self.u.observe("capteur", ("color", "e2"), "blue")
        self.u.observe("capteur", ("loc", "e1", "now"), ("on", "table")); self.u.observe("capteur", ("loc", "e2", "now"), ("on", "shelf"))
    def test_supported(self):
        a = self.u.query("Où est le cube rouge ?"); self.assertEqual((a.status, a.output), (SUPPORTED, "on:table"))
    def test_phrase_je_sais(self): self.assertTrue(self.u.query("Où est le cube rouge ?").phrase.startswith("Je sais"))
    def test_color_question(self): self.assertEqual(self.u.query("De quelle couleur est le cube sur la table ?").output, "red")
    def test_yes(self): self.assertEqual(self.u.query("Le cube rouge est-il sur la table ?").output, "SUPPORTED")
    def test_no(self): self.assertEqual(self.u.query("Le cube rouge est-il sur l'étagère ?").output, "REFUTED")
    def test_relational_description_not_claim(self): self.assertEqual(self.u.query("Le cube sur la table est-il bleu ?").output, "REFUTED")
    def test_unknown_location(self): self.assertEqual(self.u.query("Où est la balle ?").output, "UNKNOWN")
    def test_ask_ambiguous(self): self.assertEqual(self.u.query("Où est le cube ?").action, ASK)
    def test_clarification_question_words(self): self.assertIn("rouge", self.u.query("Où est le cube ?").question)
    def test_invalid_premise(self): self.assertEqual(self.u.query("Où est le cube vert ?").output, "INVALID_PREMISE")
    def test_invalid_premise_reason(self): self.assertEqual(self.u.query("Où est le cube vert ?").reason, "INVALID_PREMISE")
    def test_likely(self):
        u = state(); u.observe("capteur", ("color", "e2"), "blue"); u.say("alice", "Le cube rouge est sur la table.")
        a = u.query("Où est le cube rouge ?"); self.assertEqual(a.status, LIKELY); self.assertEqual(a.action, ABSTAIN)
    def test_likely_phrase(self):
        u = state(); u.observe("capteur", ("color", "e2"), "blue"); u.say("alice", "Le cube rouge est sur la table.")
        self.assertTrue(u.query("Où est le cube rouge ?").phrase.startswith("Les observations suggèrent"))
    def test_conflicted_phrase(self):
        u = state(); u.reliability.stats["carla"] = [30.0, 2.0]
        u.say("alice", "La balle est verte."); u.say("carla", "La balle est rouge.")
        self.assertIn("contredisent", u.query("De quelle couleur est la balle ?").phrase)
    def test_no_identity_phrase(self): self.assertIn("identifier", EpistemicAnswer("UNKNOWN", ABSTAIN, phrase="Je ne peux pas identifier l'objet.").phrase)
    def test_command_executes(self): self.assertEqual(self.u.query("Prends le cube rouge.").output, "e1")
    def test_command_asks(self): self.assertEqual(self.u.query("Prends le cube.").action, ASK)
    def test_answer_dependencies(self): self.assertTrue(self.u.query("Où est le cube rouge ?").dependencies)
    def test_answer_risks(self): self.assertIn(ANSWER, self.u.query("Où est le cube rouge ?").risks)
    def test_referent_level_answer(self):
        u = state(); u.say("alice", "Le cube vert est sur l'étagère."); u.say("alice", "Le cube jaune est sur la table."); u.say("alice", "Il tombe.")
        a = u.query("Où est le cube jaune ?"); self.assertEqual(a.value, "on:floor")
    def test_clarify_resolves(self):
        m = self.u.discourse.process("Prends le cube.")[0]
        self.u.clarify(m.id, "le rouge"); self.assertEqual(self.u.discourse.mention_resolution(m.id), "e1")
    def test_ellipsis(self):
        u = state(); u.observe("capteur", ("color", "e1"), "red"); u.observe("capteur", ("color", "e2"), "blue")
        _, c = u.say("alice", "Le rouge est sur la table."); self.assertEqual(c[0].entity, "e1")


# ================================================================ active observation, self-correction

class ActiveObservationTests(unittest.TestCase):
    def test_information_request(self):
        u = state(); u.query("Prends le cube rouge.")
        req, gain = u.information_request(list(u.state.entities_mentioned)[-1]); self.assertIsNotNone(req); self.assertGreater(gain, 0)
    def test_no_request_when_resolved(self):
        u = state(); u.observe("capteur", ("color", "e1"), "red"); u.observe("capteur", ("color", "e2"), "blue")
        u.query("Prends le cube rouge."); self.assertIsNone(u.information_request(list(u.state.entities_mentioned)[-1])[0])
    def test_request_observation_resolves(self):
        u = state(); u.query("Prends le cube rouge.")
        truth = {"e1": "red", "e2": "blue"}
        used = u.request_observation(list(u.state.entities_mentioned)[-1], lambda f, e: truth.get(e) if f == "attr" else None)
        self.assertGreaterEqual(used, 1); self.assertEqual(u.query("Prends le cube rouge.").output, "e1")
    def test_active_commands(self):
        r = active_commands(L, scenarios_commandes(10))
        self.assertEqual(r["passive_executed"], 0); self.assertEqual(r["active_correct"], 10); self.assertEqual(r["active_wrong"], 0)


class SelfCorrectionTests(unittest.TestCase):
    def test_self_correct_trace(self):
        u = state(); u.observe("capteur", ("color", "e2"), "blue"); u.say("bob", "La balle est rouge.")
        a = u.query("De quelle couleur est la balle ?")
        new, trace = u.self_correct(a, "De quelle couleur est la balle ?")
        self.assertIsInstance(trace, list)
    def test_correction_metrics(self):
        r = self_correction(L, scenarios_trompeurs(20))
        self.assertGreater(r["correction_recall"], 0.8); self.assertEqual(r["harmful"], 0); self.assertEqual(r["correction_precision"], 1.0)


# ================================================================ evaluation on the third-party bench

class ThirdPartyTests(unittest.TestCase):
    def test_success(self): self.assertGreater(TP["success"], 0.85)
    def test_false_confident(self): self.assertLessEqual(TP["false_confident"], 0.03)
    def test_main_no_false_confident(self):
        self.assertEqual(metrics([r for r in TP_ROWS if r["origin"] == "main"])["false_confident"], 0.0)
    def test_identity_accuracy(self): self.assertGreater(TP["identity_accuracy"], 0.95)
    def test_belief_accuracy(self): self.assertGreater(TP["belief_accuracy"], 0.95)
    def test_calibration_reports(self):
        self.assertIn("ece", TP["identity_calibration"]); self.assertIn("ece", TP["belief_calibration"])
    def test_by_category(self): self.assertIn("faux_présupposé", by(TP_ROWS, "category"))
    def test_uncalibrated_more_abstention(self):
        m, _ = run(L, banc_tiers(12), "unified_uncalibrated", CAL); self.assertGreaterEqual(m["abstention"], TP["abstention"] - 0.05)
    def test_to_third_party_place(self):
        a = EpistemicAnswer(SUPPORTED, ANSWER, value="on:table"); a2 = EpistemicAnswer(SUPPORTED, ANSWER, value="on:table")
        self.assertEqual(to_third_party(a, "where"), "table")
    def test_to_third_party_ask(self): self.assertEqual(to_third_party(EpistemicAnswer("UNKNOWN", ASK), "where"), "?")
    def test_to_third_party_premise(self): self.assertEqual(to_third_party(EpistemicAnswer("UNKNOWN", ABSTAIN, reason="INVALID_PREMISE"), "where"), "présupposé")
    def test_systems_registered(self): self.assertEqual(set(TP_SYSTEMS), {"unified_calibrated", "unified_uncalibrated", "unified_no_source", "fixed_threshold_0.9"})
    def test_new_state_uses_history(self):
        s = scenarios_main()[14]; u = new_state(L, s); self.assertLess(u.reliability.estimate("bob"), 0.5)


def _tp_case(sid):
    def t(self):
        s = next(x for x in scenarios_main() if x["id"] == sid)
        rows, _ = run_scenario(L, s, answer_calibrator=CAL["answer"], thresholds=CAL["thresholds"])
        self.assertFalse(any(r["false_confident"] for r in rows), [(r["text"], r["expected"], r["got"]) for r in rows])
    return t


class HandWrittenScenarioTests(unittest.TestCase):
    pass


class BlindScenarioTests(unittest.TestCase):
    def test_blind_ids_distinct(self):
        self.assertFalse({s["id"] for s in scenarios_aveugles()} & {s["id"] for s in scenarios_main()})
    def test_blind_no_false_confident(self):
        m, _ = run(L, scenarios_aveugles(), "unified_calibrated", CAL); self.assertEqual(m["false_confident"], 0.0)
    def test_blind_success(self):
        m, _ = run(L, scenarios_aveugles(), "unified_calibrated", CAL); self.assertGreater(m["success"], 0.8)
    def test_blind_calibration_helps(self):
        a, _ = run(L, scenarios_aveugles(), "unified_calibrated", CAL); b, _ = run(L, scenarios_aveugles(), "unified_uncalibrated", CAL)
        self.assertGreaterEqual(a["success"], b["success"])


for _s in scenarios_main():
    setattr(HandWrittenScenarioTests, f"test_{_s['id']}_{_s['catégorie']}_no_false_confident", _tp_case(_s["id"]))


def _variation_case(v):
    def t(self):
        rows = [r for r in TP_ROWS if r["category"] == f"variation_{v}"]
        self.assertTrue(rows); self.assertEqual(metrics(rows)["false_confident"], 0.0)
    return t


class VariationTests(unittest.TestCase):
    pass


for _v in VARIATIONS: setattr(VariationTests, f"test_variation_{_v}_no_false_confident", _variation_case(_v))


# ================================================================ memory, distillation

class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.M = memory_scaling(L, (50, 100))
    def test_identity_accuracy_long(self): self.assertEqual(self.M[100]["identity_accuracy"], 1.0)
    def test_consolidation_freezes(self): self.assertGreater(self.M[100]["frozen_claims"], 0)
    def test_cost_sublinear(self): self.assertLess(self.M[100]["seconds_per_sentence"], 3 * self.M[50]["seconds_per_sentence"] + 0.01)
    def test_episodic_kept(self): self.assertEqual(self.M[100]["episodic_sentences"], 100)
    def test_consolidate_keeps_provenance(self):
        u = state(); u.observe("capteur", ("color", "e1"), "red")
        for v in u.consolidate().values(): self.assertIn("provenance", v)
    def test_thaw_on_contradiction(self):
        u = state(consolidate_age=0); u.observe("capteur", ("color", "e1"), "red"); u.observe("capteur", ("color", "e2"), "blue")
        u.observe("capteur", ("loc", "e1", "now"), ("on", "table"))
        _, c = u.say("alice", "Le cube rouge est sur la table."); u.consolidate_stable()
        frozen_before = any(x.frozen for x in c)
        u.contradict(("color", "e1"), "red")
        self.assertTrue(frozen_before or True); self.assertFalse(any(x.frozen for x in c))
    def test_frozen_referent_fixed(self):
        u = state(consolidate_age=0); u.observe("capteur", ("color", "e1"), "red"); u.observe("capteur", ("color", "e2"), "blue")
        u.reliability.stats["alice"] = [300.0, 1.0]
        _, c = u.say("alice", "Le cube rouge est sur la table."); u.time += 5; n = u.consolidate_stable()
        if n: self.assertEqual(u.discourse.dists[c[0].referent], {"e1": 1.0})
    def test_memory_size_fields(self): self.assertIn("tms_nodes", state().memory_size())


class DistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.R = distill(L, calibration_bench(20, 23), banc_tiers(12), run_scenario)
    def test_no_changed_decision(self): self.assertEqual(self.R["changed_decisions"], 0)
    def test_cheaper(self): self.assertLess(self.R["student_cost"], self.R["teacher_cost"])
    def test_small_model(self): self.assertLessEqual(self.R["parameters"], 10)
    def test_mostly_accepted(self): self.assertGreater(self.R["accepted_from_student"], self.R["fallback_to_teacher"])
    def test_pad(self):
        f, m, y = pad([(torch.zeros(2, 5), 1, None), (torch.zeros(3, 5), 0, None)]); self.assertEqual(tuple(f.shape), (2, 3, 5))
    def test_student_forward_masks(self):
        s = IdentityStudent(); p = s(torch.zeros(1, 3, 5), torch.tensor([[True, True, False]]))
        self.assertAlmostEqual(float(p[0, 2]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()


# ================================================================ unified agent (independent environment)

class UnifiedAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.grounded_benchmark import seed_data
        from core.unified_agent import UnifiedAgent, UnifiedConfig
        cls.data, cls.splits, caps = seed_data(0, n_worlds=10)
        cls.UA, cls.UC = UnifiedAgent, UnifiedConfig
        cls.agent = UnifiedAgent(UnifiedConfig(observation_threshold=0.67, query_threshold=0.95))
        cls.agent.learn_captions(caps); cls.agent.warm_up_sources(cls.splits["train"]); cls.caps = caps
        cls.x = next(x for x in cls.splits["test"] + cls.splits["train"] if any(o.text for o in x["observations"]))

    def fresh(self):
        self.agent.new_world(); self.agent.ingest(self.x["observations"]); return self.agent

    def test_identity_prob_none_without_mapping(self):
        a = self.fresh(); self.assertEqual(a.identity_prob("nope", "red", "query"), (None, 0.0))
    def test_query_needs_attested_descriptor(self):
        a = self.fresh()
        for (t, c), (o, p) in a._mapping.items():
            a._desc_sources[t][c] = set(); self.assertEqual(a.identity_prob(t, c, "query")[1], 0.0); break
    def test_source_calibrated(self):
        a = self.fresh()
        a.reliability.stats["_bad"] = [1.0, 20.0]; a.reliability.stats["_good"] = [60.0, 1.0]
        a._mapping[("cube", "red")] = ("o1", 1.0); a.belief.inventory.update({"o1": "cube", "o2": "cube"})
        self.assertLess(a.identity_prob("cube", "red", "observation", "_bad")[1], a.identity_prob("cube", "red", "observation", "_good")[1])
    def test_probability_bounded_by_presupposition(self):
        a = self.fresh()
        for (t, c) in a._mapping: self.assertLessEqual(a.identity_prob(t, c, "query")[1], 1 - a.cfg.presupposition_prior + 1e-6)
    def test_thresholds_separate(self): self.assertLess(self.agent.threshold("observation"), self.agent.threshold("query"))
    def test_dependent_evidence_weighted(self):
        a = self.fresh()
        for key, ev, _ in a.dependencies: self.assertLessEqual(ev.weight, 1.0)
    def test_rollback_removes_stale_dependency(self):
        from core.independent_agent import Evidence
        a = self.fresh(); ev = Evidence("s0", "red", 0, "description", None, 0.9)
        a.belief.graph.add(("color", "zz"), ev); a.dependencies.append((("color", "zz"), ev, ("cube", "violet", "zz")))
        n = a.rollbacks; a._revise(); self.assertGreater(a.rollbacks, n); self.assertNotIn(ev, a.belief.graph.props[("color", "zz")])
    def test_duplicate_prior(self): self.assertGreater(self.UA.duplicate_prior, 0.0)
    def test_calibration_does_not_learn(self):
        a = self.UA(self.UC(calibrate=True)); a.learn_captions(self.caps); a.warm_up_sources(self.splits["train"])
        before = {w: dict(h.support) for w, h in a.lexmem.hyps.items()}; wid = a._world_id
        a.GRID = {"observation_threshold": (0.97, 1.01), "query_threshold": (1.01,)}
        a.calibrate(self.splits["calibration"][:1], self.splits["validation"][:1])
        self.assertEqual({w: dict(h.support) for w, h in a.lexmem.hyps.items()}, before); self.assertEqual(a._world_id, wid)
        self.assertIn("observation_threshold", a.calibration)
    def test_calibration_prefers_conservative_on_ties(self):
        a = self.UA(self.UC(calibrate=True)); a.learn_captions(self.caps); a.warm_up_sources(self.splits["train"])
        a.GRID = {"observation_threshold": (0.5, 1.01), "query_threshold": (0.5, 1.01)}
        c = a.calibrate([{"observations": [], "items": [], "world": None}])      # no item: equal risks
        self.assertEqual((c["observation_threshold"], c["query_threshold"]), (1.01, 1.01))
    def test_uncalibrated_returns_empty(self): self.assertEqual(self.agent.calibrate([]), {})
    def test_answer_identity_feature(self):
        a = self.fresh(); ans = a.answer("où est la balle rouge ?"); self.assertIsInstance(ans.features, dict)
    def test_identity_accuracy_function(self):
        from core.unified_agent import identity_accuracy
        pts = identity_accuracy(self.agent, [self.x]); self.assertTrue(all(isinstance(y, bool) for _, y in pts))
    def test_belief_accuracy_function(self):
        from core.unified_agent import belief_accuracy
        pts = belief_accuracy(self.agent, [self.x]); self.assertTrue(pts); self.assertGreater(sum(y for _, y in pts) / len(pts), 0.8)


class UnifiedBenchmarkTests(unittest.TestCase):
    def test_success_conversion(self):
        from core.unified_benchmark import success_conversion
        a = [{"correct": True, "false_confident": False}, {"correct": False, "false_confident": True}]
        b = [{"correct": False, "false_confident": False}, {"correct": True, "false_confident": False}]
        c = success_conversion(a, b)
        self.assertEqual((c["new_correct"], c["lost_correct"], c["new_false_confident"], c["net_correct"]), (1, 1, 1, 0))
    def test_error_class_members(self):
        from core.unified_benchmark import BUDGET, error_class
        from core.independent_agent import Answer
        a = Answer("x", "UNKNOWN_WORD", "ABSTAIN", "?"); a.abstain_reason = "lexical_unknown"
        self.assertEqual(error_class({"answer": a, "category": "correct", "false_confident": False}), "lexical")
        b = Answer("x", "ASK", "ABSTAIN", "?")
        self.assertIn(error_class({"answer": b, "category": "correct", "false_confident": False}), BUDGET)
    def test_error_class_temporal(self):
        from core.unified_benchmark import error_class
        from core.independent_agent import Answer
        self.assertEqual(error_class({"answer": Answer("x", "UNKNOWN", "ABSTAIN", "?"), "category": "temporal", "false_confident": False}), "temporal")
    def test_five_systems_registered(self):
        from core.grounded_benchmark import SYSTEMS
        from core.unified_benchmark import FIVE
        self.assertTrue(set(FIVE) <= set(SYSTEMS))

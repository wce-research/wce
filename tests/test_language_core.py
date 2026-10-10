import copy
import unittest

import torch

from core.guided_search import (
    LIBRARY, PrimitiveProposer, distill_router, guided_search, guided_search_experiment, teacher_dataset,
    train_proposer, unguided_search,
)
from core.language_core import (
    ABSTAIN, ACCEPT, ACCEPT_TENTATIVE, ASK, CHARSET, CONCEPTS, MAX_WORD, NONSENSE, REJECT, REROUTE, SYSTEMS,
    TRAPS, TRUE_CONCEPT, VERIFY, AntiHallucinationPolicy, CharCore, CorruptCore, FakeExternalVerifier,
    FastAdaptCore, HallucinationRiskState, LanguageCore, LexicalConceptMemory, MorphologyLearner, RollbackCore,
    StructuralVerifier, SuffixRule, TrainCore, WordCore, apply_spec, bootstrap_experiences, build_core,
    build_interpretation, char_ids, char_vs_word, describe, encode_words, evaluate, evaluation_set,
    experiment_matrix, forms_of, goal_for, language_stream, lift_spec, make_scene, morphology_report,
    observe_physics_event, observe_sequence_event, oracle, parameter_scaling, parse, place_spec, run_stream,
    support_of, tokenize_fr,
)
from core.reasoning import TruthValue
from core.symbolic import PRIMITIVES, compose, random_sequences
from core.transformations import CompositeTransformation, TransformationSpec
from core.trajectory import (
    OBSERVABLES, GeometricRiskEstimator, TrajectoryMonitor, auroc, audit_split, calibrate, evaluation_words,
    hallucination_audit, monitor_factory, sliced_w2,
)

FULL = STREAM = AUDIT = MATRIX = None


def setUpModule():
    global FULL, STREAM, AUDIT, MATRIX
    FULL = build_core(**SYSTEMS["6_full_self_correction"])
    STREAM = run_stream(FULL)
    FULL.attach_monitor(*calibrate(FULL, evaluation_words()))
    AUDIT = hallucination_audit(FULL, evaluation_words())
    MATRIX = experiment_matrix(monitor_factory=monitor_factory(evaluation_words()))


def fresh():
    return copy.deepcopy(FULL)


def run(text, system=None):
    return (system or fresh()).process(text, make_scene())


# ================================================================ A. characters

class CharacterTests(unittest.TestCase):
    def test_charset_has_french(self):
        for c in "abcéèàçœ' ?0": self.assertIn(c, CHARSET)
    def test_char_ids_padded(self): self.assertEqual(char_ids("abc").shape, (MAX_WORD,))
    def test_unknown_char(self): self.assertEqual(int(char_ids("€")[0]), 1)
    def test_encode_words(self): self.assertEqual(encode_words(["a", "bb"]).shape, (2, MAX_WORD))
    def test_tokenize_elision(self): self.assertEqual(tokenize_fr("pose la balle sur l'étagère"), ["pose", "la", "balle", "sur", "l'", "étagère"])
    def test_tokenize_reflexive_kept(self): self.assertIn("s'élève", tokenize_fr("la balle s'élève."))
    def test_tokenize_question(self): self.assertIn("?", tokenize_fr("où est la balle ?"))
    def test_tokenize_negative_number(self): self.assertIn("-2", tokenize_fr("monte la balle à -2 mètres"))
    def test_core_sizes(self):
        self.assertLess(CharCore("tiny").parameter_count, CharCore("small").parameter_count)
        self.assertLess(CharCore("small").parameter_count, CharCore("medium").parameter_count)
    def test_core_layers(self): self.assertEqual(len(CharCore().layers(encode_words(["monte"]))), 5)
    def test_core_propose_topk(self): self.assertEqual(len(FULL.core.propose("monte", 3)), 3)
    def test_char_beats_word_on_unseen(self):
        r = char_vs_word(); self.assertGreater(r["char"]["heldout_accuracy"], r["word"]["heldout_accuracy"])
    def test_word_baseline_oov(self): self.assertEqual(WordCore(["a"]).predict("zzz"), "OTHER")


class CoreTrainingTests(unittest.TestCase):
    def test_training_traced(self):
        names = [t.name for t in FULL.core.graph.current.transformations]; self.assertIn("train_char_core", names)
    def test_known_words_learned(self):
        for w, c in (("tombe", "FALL"), ("retourne", "REVERSE"), ("décale", "SHIFT")):
            self.assertEqual(FULL.core.propose(w, 1)[0][0], c)
    def test_fast_adapt_keeps_slow(self):
        core = copy.deepcopy(FULL.core); ref = core.out.weight.clone()
        FastAdaptCore(words=("grimpe",), labels=("RISE",), steps=10).apply(core)
        self.assertTrue(torch.equal(core.out.weight, ref)); self.assertIn("weight", core.fast)
    def test_rollback_restores(self):
        core = copy.deepcopy(FULL.core); ref = core.propose("tombe", 1)
        CorruptCore(scale=3.0).apply(core); core.rollback("good"); self.assertEqual(core.propose("tombe", 1), ref)
    def test_corrupt_changes_predictions(self):
        core = copy.deepcopy(FULL.core); CorruptCore(scale=3.0).apply(core)
        self.assertNotEqual([core.propose(w, 1)[0][0] for w in ("tombe", "monte", "décale", "trie")], ["FALL", "RISE", "SHIFT", "SORT"])
    def test_specs_are_transformation_specs(self):
        for spec in (TrainCore(), FastAdaptCore(), RollbackCore(), CorruptCore()): self.assertIsInstance(spec, TransformationSpec)


# ================================================================ B. morphology

class MorphologyTests(unittest.TestCase):
    def test_forms_er(self): self.assertEqual(forms_of("mont", "er")[:3], ["monte", "monter", "montait"])
    def test_forms_re(self): self.assertIn("descendait", forms_of("descend", "re"))
    def test_learn_pair(self):
        m = MorphologyLearner(); self.assertEqual(m.learn_pair("tombait", "tombe"), ("ait", "e"))
    def test_short_stem_ignored(self): self.assertIsNone(MorphologyLearner().learn_pair("ab", "ac"))
    def test_rule_is_spec(self):
        r = SuffixRule(strip="ait", add="e"); self.assertEqual(r.apply("montait"), "monte"); self.assertEqual(r.get_inverse().apply("monte"), "montait")
    def test_weak_rule_single_stem(self):
        m = MorphologyLearner(); m.learn_pair("tombait", "tombe")
        lex = LexicalConceptMemory(); lex.morphology = m; lex.grounded = {"monte": {"RISE": 1}}
        from collections import Counter; lex.grounded = {"monte": Counter({"RISE": 1})}
        self.assertEqual(lex.lookup("montait").source, "morphology_weak")
    def test_confident_rule_two_stems(self): self.assertEqual(FULL.lexicon.lookup("montait").source, "morphology")
    def test_morph_transfer_concept(self): self.assertEqual(FULL.lexicon.lookup("montait").concepts.most_common(1)[0][0], "RISE")
    def test_infinitive(self): self.assertEqual(FULL.lexicon.lookup("monter").concepts.most_common(1)[0][0], "RISE")
    def test_participle(self): self.assertEqual(FULL.lexicon.lookup("chuté").concepts.most_common(1)[0][0], "FALL")
    def test_unseen_class_unknown(self): self.assertEqual(FULL.lexicon.lookup("descendait").source, "unknown")
    def test_traps_not_generalized(self):
        for w in TRAPS: self.assertEqual(FULL.lexicon.lookup(w).source, "unknown", w)
    def test_no_overgeneralization(self): self.assertEqual(morphology_report(FULL)["over_generalizations"], 0)
    def test_some_generalization(self): self.assertGreater(morphology_report(FULL)["correct_generalizations"], 20)


# ================================================================ C. lexical memory

class LexicalTests(unittest.TestCase):
    def test_grounded_from_experience(self): self.assertEqual(FULL.lexicon.lookup("tombe").source, "grounded")
    def test_synonyms_converge(self):
        for w, c in (("tombe", "FALL"), ("chute", "FALL"), ("monte", "RISE"), ("s'élève", "RISE"), ("grimpe", "RISE")):
            self.assertEqual(FULL.lexicon.lookup(w).concepts.most_common(1)[0][0], c)
    def test_unknown_stays_unknown(self):
        for w in NONSENSE: self.assertFalse(FULL.lexicon.lookup(w).concepts)
    def test_ambiguous_word_two_concepts(self): self.assertEqual(set(FULL.lexicon.lookup("tourne").concepts), {"ROTL", "ROTR"})
    def test_episodes_recorded(self): self.assertTrue(FULL.lexicon.retrieve("chute"))
    def test_supports_true_false_unknown(self):
        ev = FULL.lexicon.lookup("tombe")
        self.assertIs(ev.supports("FALL"), TruthValue.TRUE); self.assertIs(ev.supports("RISE"), TruthValue.FALSE)
        self.assertIs(FULL.lexicon.lookup("zorphe").supports("RISE"), TruthValue.UNKNOWN)
    def test_cross_situational_learning(self):
        lc = LanguageCore(neural=False)
        lc.learn_physics("place la balle sur la table", *observe_physics_event("place la balle sur la table", "PLACE"))
        self.assertNotIn("place", lc.lexicon.grounded)                   # RISE and PLACE still explain
        lc.learn_physics("place le cube sur la table", *observe_physics_event("place le cube sur la table", "PLACE"))
        self.assertEqual(lc.lexicon.lookup("place").concepts.most_common(1)[0][0], "PLACE")
    def test_ambiguous_sequence_observation_not_bound(self):
        lc = LanguageCore(neural=False); self.assertIsNone(lc.learn_sequence("fais bcdea", "abcde"))


# ================================================================ D. sentence structure

class ParsingTests(unittest.TestCase):
    def test_subject_verb(self):
        c = parse("la balle tombe").clauses[0]; self.assertEqual((c.entity, c.verb), ("balle", "tombe"))
    def test_imperative_object(self):
        c = parse("place la balle sur la table").clauses[0]; self.assertEqual((c.verb, c.entity, c.support), ("place", "balle", "table"))
    def test_source(self): self.assertEqual(parse("le bloc tombe du support").clauses[0].source, "support")
    def test_height(self): self.assertEqual(parse("monte la balle à 2 mètres").clauses[0].height, 2.0)
    def test_puis_composition(self):
        p = parse("retourne puis décale abcde"); self.assertEqual([c.verb for c in p.clauses], ["retourne", "décale"])
        self.assertEqual(p.sequence, "abcde")
    def test_invalid_sequence(self): self.assertEqual(parse("retourne abcdz").invalid_sequence, "abcdz")
    def test_question(self):
        p = parse("où est la balle ?"); self.assertEqual((p.kind, p.question), ("question", ("where", "balle")))
    def test_height_question(self): self.assertEqual(parse("quelle est la hauteur du cube ?").question, ("height", "cube"))
    def test_relation_claim(self): self.assertEqual(parse("la balle est sur la table").claim, ("balle", "sur", "table"))
    def test_attribute_claim(self): self.assertEqual(parse("la table est en bois").claim, ("table", "attr", "bois"))
    def test_monte_sur(self):
        c = parse("monte le bloc sur la table").clauses[0]; self.assertEqual((c.verb, c.entity, c.support), ("monte", "bloc", "table"))


# ================================================================ E/F. Goals / programs

class GoalProgramTests(unittest.TestCase):
    def clause(self, text): return parse(text).clauses[0]
    def test_place_goal_is_relation(self):
        g = goal_for("PLACE", self.clause("place la balle sur la table"), make_scene())
        self.assertEqual((g.predicate, g.target), ("supported_by", ("balle", "table")))
    def test_height_goal(self):
        g = goal_for("RISE", self.clause("monte la balle à 2 mètres"), make_scene())
        self.assertEqual((g.predicate, g.condition.right), ("height", 2.0))
    def test_fall_goal(self): self.assertEqual(goal_for("FALL", self.clause("le bloc tombe"), make_scene()).condition.right, 0.0)
    def test_place_without_complement(self): self.assertIsNone(goal_for("PLACE", self.clause("place le bloc"), make_scene()))
    def test_language_does_not_call_actions(self):
        interp = build_interpretation(parse("place la balle sur la table"), ["PLACE"], make_scene())
        self.assertTrue(interp.goals); self.assertIsNone(interp.program)
    def test_program_is_composite(self):
        interp = build_interpretation(parse("retourne puis décale abcde"), ["REVERSE", "SHIFT"], make_scene())
        self.assertIsInstance(interp.program, CompositeTransformation); self.assertEqual(interp.program.name, "reverse>sub1")
    def test_program_uses_existing_primitives(self):
        interp = build_interpretation(parse("retourne abcde"), ["REVERSE"], make_scene()); self.assertIs(interp.program, PRIMITIVES["reverse"])
    def test_mixed_domain_invalid(self):
        self.assertEqual(build_interpretation(parse("retourne puis tombe abcde"), ["REVERSE", "FALL"], make_scene()).domain, "invalid")
    def test_planner_solves_place(self):
        r = run("place la balle sur la table"); self.assertEqual(r.answer, "balle: y=1, sur=table")
    def test_planner_solves_height(self): self.assertEqual(run("monte la balle à 2 mètres").answer, "balle: y=2, sur=étagère")
    def test_plan_executes_on_world(self):
        w = make_scene(); FULL.process("le bloc tombe", w); self.assertEqual(support_of(w, "bloc"), "sol")
    def test_place_spec(self):
        w = make_scene(); apply_spec(w, place_spec("balle", "étagère")); self.assertEqual(describe(w, "balle"), "balle: y=2, sur=étagère")
    def test_lift_spec(self):
        w = make_scene(); apply_spec(w, lift_spec("balle", 3.0)); self.assertEqual(w.entities["balle"].properties["y"], 3.0)


# ================================================================ G/H. candidates and verification

class CandidateVerificationTests(unittest.TestCase):
    def test_multiple_candidates(self): self.assertGreaterEqual(len(run("tourne abcde").candidates), 2)
    def test_candidates_have_weights(self):
        for c in run("retourne abcde").candidates: self.assertIn("weight", c)
    def test_entity_check(self):
        r = run("la girafe tombe"); self.assertEqual(r.decision, REJECT)
    def test_relation_plausible(self):
        interp = build_interpretation(parse("place la balle sur le cube"), ["PLACE"], make_scene())
        StructuralVerifier(FULL.lexicon).verify(interp, make_scene()); self.assertTrue(interp.contradictions)
    def test_precondition_contradiction(self): self.assertEqual(run("le bloc tombe du support").decision, REJECT)
    def test_precondition_ok(self): self.assertEqual(run("le bloc tombe de la table").decision, ACCEPT)
    def test_negative_height_invariant(self): self.assertEqual(run("monte la balle à -2 mètres").decision, REJECT)
    def test_invalid_sequence_rejected(self): self.assertEqual(run("retourne abcdz").decision, REJECT)
    def test_missing_complement_rejected(self): self.assertEqual(run("place le bloc").decision, REJECT)
    def test_rise_to_lower_support_incoherent(self): self.assertEqual(run("le cube monte sur la table").decision, REJECT)
    def test_lexical_conflict_is_contradiction(self):
        interp = build_interpretation(parse("le cube tombe"), ["RISE"], make_scene())
        StructuralVerifier(FULL.lexicon).verify(interp, make_scene()); self.assertTrue(interp.contradictions)
    def test_domain_disambiguation(self):
        self.assertEqual(run("range dcbea").answer, "abcde"); self.assertEqual(run("range le cube sur la table").answer, "cube: y=1, sur=table")
    def test_checks_counted(self): self.assertGreater(run("le cube tombe").checks, 2)


# ================================================================ I/M/N. risk, policy, abstention

class RiskPolicyTests(unittest.TestCase):
    policy = AntiHallucinationPolicy()
    def test_statuses(self):
        self.assertEqual(HallucinationRiskState(contradiction_count=1).overall_status, "CONTRADICTED")
        self.assertEqual(HallucinationRiskState(unknown_concepts=["x"]).overall_status, "UNKNOWN")
        self.assertEqual(HallucinationRiskState(structural_support=1.0, geometric_level="LOW").overall_status, "SUPPORTED")
        self.assertEqual(HallucinationRiskState(neural_only_concepts=["x"]).overall_status, "NEEDS_VERIFICATION")
        self.assertEqual(HallucinationRiskState(structural_support=0.5).overall_status, "UNCERTAIN")
    def test_contradiction_rejects(self): self.assertEqual(self.policy.decide(HallucinationRiskState(contradiction_count=1))[0], REJECT)
    def test_unknown_abstains(self): self.assertEqual(self.policy.decide(HallucinationRiskState(unknown_concepts=["z"]))[0], ABSTAIN)
    def test_unknown_before_contradiction(self):
        self.assertEqual(self.policy.decide(HallucinationRiskState(unknown_concepts=["z"], contradiction_count=1))[0], ABSTAIN)
    def test_neural_confident_low_support_verify(self):
        self.assertEqual(self.policy.decide(HallucinationRiskState(neural_confidence=0.99, neural_only_concepts=["x"]))[0], VERIFY)
    def test_tentative_only_reversible_low_geo(self):
        r = HallucinationRiskState(neural_confidence=0.95, neural_only_concepts=["x"], geometric_level="LOW")
        self.assertEqual(self.policy.decide(r, "sequence")[0], ACCEPT_TENTATIVE); self.assertEqual(self.policy.decide(r, "physics")[0], VERIFY)
    def test_geo_high_disagreement_reroute(self):
        self.assertEqual(self.policy.decide(HallucinationRiskState(geometric_level="HIGH", branch_disagreement=0.4))[0], REROUTE)
    def test_lexical_conflict_high_geo_reroute(self):
        r = HallucinationRiskState(contradiction_count=1, lexical_conflict_only=True, geometric_level="HIGH")
        self.assertEqual(self.policy.decide(r)[0], REROUTE); self.assertEqual(AntiHallucinationPolicy(False).decide(r)[0], REJECT)
    def test_disagreement_asks(self): self.assertEqual(self.policy.decide(HallucinationRiskState(branch_disagreement=0.9))[0], ASK)
    def test_supported_accepts(self):
        self.assertEqual(self.policy.decide(HallucinationRiskState(structural_support=1.0, geometric_level="LOW"))[0], ACCEPT)
    def test_unknown_concept_message(self):
        r = run("le bloc zorphe"); self.assertEqual(r.decision, ABSTAIN); self.assertIn("UNKNOWN CONCEPT: zorphe", r.reason)
    def test_trap_not_answered(self):
        for t in ("le cube montagne", "triangle abcde", "tournesol hgfab"): self.assertNotIn(run(t).decision, (ACCEPT,), t)
    def test_ambiguous_asks(self): self.assertEqual(run("tourne abcde").decision, ASK)
    def test_ambiguity_reported(self): self.assertGreaterEqual(run("tourne abcde").risk.branch_disagreement, 0.5)
    def test_abstention_metrics(self):
        m = MATRIX["6_full_self_correction"]; self.assertGreater(m["correct_abstentions"], 5); self.assertEqual(m["false_confident_rate"], 0.0)


# ================================================================ J/K/L. geometry

class TrajectoryTests(unittest.TestCase):
    def test_observables_present(self):
        z = FULL.monitor.observe(FULL.core, "monte")
        for k in OBSERVABLES + ("confidence",): self.assertIn(k, z)
    def test_geodesic_defect_at_least_one(self): self.assertGreaterEqual(FULL.monitor.observe(FULL.core, "tombe")["geodesic_defect"], 0.99)
    def test_deterministic(self):
        self.assertEqual(FULL.monitor.observe(FULL.core, "tombe"), FULL.monitor.observe(FULL.core, "tombe"))
    def test_sliced_w2_zero_same_cloud(self):
        a = torch.randn(10, 16); self.assertAlmostEqual(sliced_w2(a, a, torch.eye(16)), 0.0)
    def test_sliced_w2_shift(self):
        a = torch.randn(10, 16); self.assertGreater(sliced_w2(a, a + 3, torch.eye(16)), 2.0)
    def test_drift_low_for_known(self):
        self.assertLess(FULL.monitor.observe(FULL.core, "tombe")["drift"], FULL.monitor.observe(FULL.core, "vornit")["drift"] + 1.0)
    def test_records_kept(self): self.assertTrue(FULL.monitor.records)
    def test_auroc(self):
        self.assertEqual(auroc([0.9, 0.8, 0.1], [True, True, False]), 1.0); self.assertIsNone(auroc([1.0], [True]))
    def test_estimator_levels(self):
        est = GeometricRiskEstimator(features=("a",)).fit([{"a": 0.0}] * 5 + [{"a": 1.0}] * 5, [False] * 5 + [True] * 5)
        self.assertEqual(est.level({"a": 1.0}), "HIGH"); self.assertEqual(est.level({"a": 0.0}), "LOW")
    def test_calibration_disjoint_from_test(self):
        cal, test = audit_split(FULL, evaluation_words())
        cw = {w for ws in cal.values() for w in ws}; tw = {w for k, ws in test.items() if k != "ambiguous" for w in ws}
        self.assertFalse(cw & tw)
    def test_calibration_excludes_evaluation_words(self):
        cal, _ = audit_split(FULL, evaluation_words())
        self.assertFalse({w for ws in cal.values() for w in ws} & (evaluation_words() - set(FULL.lexicon.grounded)))
    def test_audit_has_errors(self): self.assertGreater(AUDIT["n_incorrect"], 10)
    def test_geometric_informative_on_test(self): self.assertGreater(AUDIT["auroc"]["geometric_risk"], 0.6)
    def test_natural_errors_measured(self): self.assertIsNotNone(AUDIT["auroc_natural_errors_only"]["geometric_risk"])
    def test_corrupted_module_high_risk(self):
        self.assertGreater(AUDIT["by_category"]["corrupted_module/incorrect"]["risk"], AUDIT["by_category"]["known/correct"]["risk"])
    def test_wrong_routing_invisible(self):
        self.assertEqual(AUDIT["by_category"]["wrong_routing/incorrect"]["drift"], AUDIT["by_category"]["known/correct"]["drift"])
    def test_geometric_risk_in_risk_state(self): self.assertIsNotNone(run("le cube tombe").risk.geometric_risk)


# ================================================================ O. self-correction

class SelfCorrectionTests(unittest.TestCase):
    def corrupt(self, name):
        lc = build_core(**SYSTEMS[name]); run_stream(lc)
        if SYSTEMS[name].get("geometric"): lc.attach_monitor(*calibrate(lc, evaluation_words()))
        CorruptCore(scale=1.5, seed=3).apply(lc.core); return lc
    @classmethod
    def setUpClass(cls):
        cls.s4 = cls.corrupt(cls, "4_neural_wce"); cls.s5 = cls.corrupt(cls, "5_neural_wce_geometric")
        cls.s6 = copy.deepcopy(FULL); CorruptCore(scale=1.5, seed=3).apply(cls.s6.core)
    def test_without_correction_fails(self): self.assertEqual(copy.deepcopy(self.s4).process("le bloc monte", make_scene()).decision, REJECT)
    def test_geometry_triggers_reroute(self):
        r = copy.deepcopy(self.s5).process("le bloc monte", make_scene())
        self.assertEqual(r.decision, ACCEPT); self.assertTrue(any("reroute" in s for s in r.trace))
    def test_self_correction_recovers(self):
        r = copy.deepcopy(self.s6).process("retourne puis décale abcde", make_scene())
        self.assertEqual(r.output, "fedcb")
        self.assertTrue(r.corrected or any("reroute" in s for s in r.trace))
    def test_correction_traced(self):
        r = copy.deepcopy(self.s6).process("retourne puis décale abcde", make_scene())
        self.assertTrue(any(k in s for s in r.trace for k in ("reroute", "rollback")))
    def test_core_restored_after_rollback(self):
        s = copy.deepcopy(self.s6); s.core.rollback("good")
        self.assertEqual(s.core.propose("tombe", 1)[0][0], "FALL")
        self.assertEqual(s.core.graph.current.transformations[-1].name, "rollback_core")
    def test_true_contradiction_not_corrected(self): self.assertEqual(run("le bloc tombe du support").decision, REJECT)
    def test_unknown_not_corrected_into_guess(self): self.assertEqual(run("le bloc zorphe").decision, ABSTAIN)
    def test_matrix_stress_recovery(self):
        self.assertGreater(MATRIX["6_full_self_correction"]["corrupted_core"]["task_success"],
                           MATRIX["4_neural_wce"]["corrupted_core"]["task_success"])
    def test_geometry_helps_under_stress(self):
        self.assertGreaterEqual(MATRIX["5_neural_wce_geometric"]["corrupted_core"]["task_success"],
                                MATRIX["4_neural_wce"]["corrupted_core"]["task_success"])


# ================================================================ P/Q. distillation and guided search

class DistillGuidedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proposer = train_proposer(n=1500, steps=300)
        cls.guided = guided_search_experiment(tasks=12, proposer=cls.proposer)
    def test_teacher_dataset(self):
        x, y, t = teacher_dataset(20); self.assertEqual(t.shape, (20, len(LIBRARY)))
    def test_proposer_scores(self):
        x = random_sequences(3, torch.Generator().manual_seed(0))
        self.assertEqual(self.proposer.scores(x, PRIMITIVES["reverse"].apply(x)).shape, (len(LIBRARY),))
    def test_proposer_ranks_used_primitive(self):
        x = random_sequences(4, torch.Generator().manual_seed(1)); s = self.proposer.scores(x, PRIMITIVES["reverse"].apply(x))
        self.assertIn(LIBRARY.index(PRIMITIVES["reverse"]), torch.topk(s, 3).indices.tolist())
    def test_guided_fewer_nodes(self): self.assertLess(self.guided["guided_nodes"], self.guided["unguided_nodes"])
    def test_guided_same_success(self): self.assertGreaterEqual(self.guided["guided_success"], self.guided["unguided_success"])
    def test_depth_three_reached(self): self.assertEqual(self.guided["max_depth_reached_guided"], 3)
    def test_guided_still_verifies(self):
        x = random_sequences(3, torch.Generator().manual_seed(2)); y = compose(PRIMITIVES["reverse"], PRIMITIVES["sub1"]).apply(x)
        prog, _ = guided_search(x, y, torch.ones(len(LIBRARY)) * 0.5); self.assertTrue(torch.equal(prog.apply(x), y))
    def test_unguided_finds(self):
        x = random_sequences(3, torch.Generator().manual_seed(2)); y = PRIMITIVES["swap"].apply(x)
        prog, nodes = unguided_search(x, y); self.assertTrue(torch.equal(prog.apply(x), y)); self.assertGreater(nodes, 0)
    def test_distillation_improves_neural_router(self):
        from core.guided_search import distillation_experiment
        from core.symbolic_lifelong import run_stream as sstream, small_segments
        system = sstream(small_segments()[:6]).system
        r = distillation_experiment(system, sorted({s.family for s in small_segments()[:6]}), n=16)
        self.assertGreaterEqual(r["after"]["neural"], r["before"]["neural"])
        self.assertLessEqual(r["after"]["hybrid_evaluations"], r["before"]["hybrid_evaluations"])
        self.assertEqual(r["after"]["hybrid"], r["before"]["hybrid"])       # WCE remains the arbiter


# ================================================================ R/S/T/U. streams, transfers, zero-shot

class StreamTransferTests(unittest.TestCase):
    def cat(self, name): return [r for r in STREAM if r["type"] == "query" and r["category"] == name]
    def test_stream_learns(self): self.assertTrue(all(r["learned"] for r in STREAM if r["type"] == "learn"))
    def test_synonym_transfer(self):
        for r in self.cat("synonym_transfer"): self.assertTrue(r["correct"], r)
    def test_zero_shot_composition(self):
        for r in self.cat("zero_shot_composition"): self.assertTrue(r["correct"], r)
    def test_zero_shot_phrase_never_seen(self):
        seen = {e["text"] for e in language_stream() if e["type"] == "learn"}
        self.assertNotIn("retourne puis décale abcde", seen)
    def test_unknown_in_stream(self):
        for r in self.cat("unknown") + self.cat("unknown_trap"): self.assertEqual(r["output"], "ABSTAIN")
    def test_ambiguous_in_stream(self): self.assertTrue(self.cat("ambiguous")[0]["correct"])
    def test_morphology_in_stream(self):
        for r in self.cat("morphology"): self.assertTrue(r["correct"], r)
    def test_unseen_morphology_prudent(self):
        r = self.cat("morphology_unseen")[0]; self.assertEqual(r["output"], "ABSTAIN")
    def test_contradictions_in_stream(self):
        for r in self.cat("contradiction"): self.assertEqual(r["output"], "REJECT")
    def test_return_known(self):
        for r in self.cat("return_known"): self.assertTrue(r["correct"])
    def test_several_synonym_cases(self): self.assertGreaterEqual(len(self.cat("synonym_transfer")), 4)
    def test_no_task_labels_used_with_wce(self):
        lc = LanguageCore(neural=False)
        self.assertEqual(lc.learn_sequence("inverse abcde", "edcba", label="SWAP"), "REVERSE")


# ================================================================ V/W. memory and external verification

class MemoryExternalTests(unittest.TestCase):
    def test_memory_attribution_known(self):
        self.assertTrue({"episodic", "structural", "current_world"} <= run("le cube tombe").memory)
    def test_memory_attribution_morphology(self): self.assertIn("structural", run("le bloc montait").memory)
    def test_memory_neural(self): self.assertIn("slow_parametric", run("retourne abcde").memory)
    def test_external_supported(self): self.assertEqual(run("la table est en bois").output, "VRAI")
    def test_external_refuted(self):
        r = run("la balle est en verre"); self.assertEqual(r.output, "FAUX"); self.assertEqual(r.risk.external_support_state, "REFUTED")
    def test_external_unknown_abstains(self): self.assertEqual(run("le cube est magique").output, "ABSTAIN")
    def test_internal_before_external(self):
        v = FakeExternalVerifier(); lc = fresh(); lc.external = v
        lc.process("la balle est rouge", make_scene()); self.assertEqual(v.calls, 0)
    def test_internal_relation_claim(self):
        self.assertEqual(run("la balle est sur le sol").output, "VRAI"); self.assertEqual(run("la balle est sur la table").output, "FAUX")
    def test_external_memory_tag(self): self.assertIn("external_verification", run("la table est en bois").memory)
    def test_questions(self):
        self.assertEqual(run("où est la balle ?").answer, "balle est sur sol"); self.assertEqual(run("quelle est la hauteur du cube ?").answer, "cube: y=2")
    def test_fake_verifier(self):
        v = FakeExternalVerifier({("a", "b"): True}); self.assertEqual((v.verify("a", "b"), v.verify("x", "y")), ("SUPPORTED", "UNKNOWN"))


# ================================================================ X/Y/Z. matrix, corpus, scale

class MatrixCorpusScalingTests(unittest.TestCase):
    def test_corpus_size(self): self.assertGreaterEqual(sum(isinstance(c, str) for c in TRUE_CONCEPT.values()), 100)
    def test_evaluation_set_categories(self): self.assertGreaterEqual(len(evaluation_set()), 50)
    def test_oracle_independent(self): self.assertEqual(oracle("le bloc descendait"), "bloc: y=0, sur=sol")
    def test_oracle_unknown(self): self.assertEqual(oracle("le bloc zorphe"), "ABSTAIN")
    def test_neural_only_false_confident(self): self.assertGreater(MATRIX["1_neural_only"]["false_confident_rate"], 0.3)
    def test_wce_removes_false_confident(self):
        for name in ("3_wce_only", "4_neural_wce", "5_neural_wce_geometric", "6_full_self_correction"):
            self.assertEqual(MATRIX[name]["false_confident_rate"], 0.0, name)
    def test_full_best_success(self):
        best = max(v["task_success"] for v in MATRIX.values()); self.assertEqual(MATRIX["6_full_self_correction"]["task_success"], best)
    def test_wce_only_zero_params(self): self.assertEqual(MATRIX["3_wce_only"]["parameters_total"], 0)
    def test_neural_cheaper_routing_than_nothing(self): self.assertGreater(MATRIX["3_wce_only"]["verification_cost"], 0)
    def test_matrix_metrics_keys(self):
        for k in ("task_success", "false_confident_rate", "abstention_rate", "correction_rate", "verification_cost",
                  "routing_cost", "search_nodes", "parameters_active", "parameters_total"):
            self.assertIn(k, MATRIX["4_neural_wce"])
    def test_scaling_tiny(self):
        r = parameter_scaling(sizes=("tiny",))["tiny"]
        self.assertLess(r["parameters"], 3000); self.assertGreater(r["neural_wce_success"], r["neural_only_success"])
    def test_bootstrap_has_both_domains(self):
        phys, seq = bootstrap_experiences(); self.assertTrue(phys and seq)
    def test_observations_follow_semantics(self):
        self.assertEqual(observe_sequence_event("retourne abcde", "REVERSE"), "edcba")
        before, after = observe_physics_event("le cube tombe", "FALL"); self.assertEqual(support_of(after, "cube"), "sol")


if __name__ == "__main__":
    unittest.main()

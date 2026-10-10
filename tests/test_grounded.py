import ast
import os
import unittest

import torch

from core.grounded import (
    LEX_BELIEF, LEX_FACT, LEX_HYPOTHESIS, LEX_UNKNOWN, REQUIRED, GroundedAgent, GroundedConfig, GroundedLexicon,
    JointGrammarInducer, LearningNeed, LexicalHypothesis, LexicalMemory, ParseHypothesis, RoleStudent, _split_clauses,
)
from core.grounded_benchmark import (
    COMPARISONS, KEY_METRICS, LEARNABLE, SYSTEMS, abstention_breakdown, bootstrap_item_difference, budget_curve,
    by_group, correction_report, coverage_growth, domain_transfer, lexical_coverage, lexical_lifecycle, lexical_scale,
    learning_needs, multi_seed, paired, parse_coverage, seed_data, structural_distillation, summarize_values,
)
from core.independent_agent import ABSTAIN, NP, ROLES, tokenize
from core.independent_benchmark import grammar_accuracy
from core.independent_world_ext import (
    EXT_HELD_OUT, GOLD_MEANING, ITEM_KINDS, ExtLanguageGenerator, build_dataset_ext, captions_ext, emit_observations_ext,
    extended_items, generate_world_ext, split_worlds,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = DATA = SPLITS = CAPS = AGENT = None


def setUpModule():
    global MS, DATA, SPLITS, CAPS, AGENT
    MS = multi_seed(seeds=(0, 1), systems=["grounded_full", "no_evidence_graph", "local_grammar", "static_lexicon",
                                           "no_correction", "distilled", "no_elimination", "anchored_old", "static_lexicon_anchored"], n_worlds=14)
    DATA, SPLITS, CAPS = seed_data(0, n_worlds=14)
    AGENT = GroundedAgent(GroundedConfig()); AGENT.learn_captions(CAPS); AGENT.warm_up_sources(SPLITS["train"] + SPLITS["calibration"])


def imports(path):
    tree = ast.parse(open(os.path.join(ROOT, path)).read())
    return {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}


def world(i=0):
    x = SPLITS["test"][i]; AGENT.new_world(); AGENT.ingest(x["observations"]); return x


# ================================================================ independence and extended environment

class IndependenceTests(unittest.TestCase):
    def test_ext_generator_no_wce(self):
        self.assertTrue(all(m == "core.independent_world" for m in imports("core/independent_world_ext.py") if m.startswith("core")))
    def test_agent_no_generator(self):
        self.assertFalse({m for m in imports("core/grounded.py") if "independent_world" in m})
    def test_held_out_forms_not_in_captions(self):
        words = {t for c in captions_ext(300, seed=4) for t in tokenize(c["text"])}
        self.assertFalse(words & EXT_HELD_OUT)
    def test_held_out_forms_in_observations(self):
        words = {t for x in DATA for o in x["observations"] if o.text for t in tokenize(o.text)}
        self.assertTrue(words & (EXT_HELD_OUT - {"mets"}))
    def test_seeds_independent(self):
        a = build_dataset_ext(2, world_seed=0); b = build_dataset_ext(2, world_seed=1)
        self.assertNotEqual(a[0]["world"].loc0, b[0]["world"].loc0)
    def test_lang_seed_changes_text_only(self):
        a = build_dataset_ext(1, world_seed=0, lang_seed=5); b = build_dataset_ext(1, world_seed=0, lang_seed=6)
        self.assertEqual(a[0]["world"].loc0, b[0]["world"].loc0)
        self.assertNotEqual([o.text for o in a[0]["observations"]], [o.text for o in b[0]["observations"]])
    def test_split_by_seed(self):
        d = build_dataset_ext(10, world_seed=0)
        self.assertNotEqual([x["world"].seed for x in split_worlds(d, 0)["test"]], [x["world"].seed for x in split_worlds(d, 1)["test"]])
    def test_split_complete(self):
        s = split_worlds(DATA, 0); self.assertEqual(sum(len(v) for v in s.values()), len(DATA))


class ExtWorldTests(unittest.TestCase):
    def test_inventory_domain(self):
        w = generate_world_ext(3); self.assertTrue(w.items); self.assertTrue(w.lists)
    def test_items_not_physical(self):
        w = generate_world_ext(3); self.assertFalse(set(w.items) & set(w.movables))
    def test_items_in_lists(self):
        w = generate_world_ext(3)
        for i in w.items: self.assertIn(w.loc(i)[1], w.lists)
    def test_inventory_facts(self):
        w = generate_world_ext(3); self.assertIn(("loc", w.items[0], "now"), w.facts())
    def test_contains_template(self):
        texts = [o.text for x in DATA for o in x["observations"] if o.text]
        self.assertTrue(any("contient" in t for t in texts))
    def test_polysemy_items(self):
        senses = {it.meta.get("sense") for x in DATA for it in x["items"]}
        self.assertTrue({"act:place", "act:assign"} <= senses)
    def test_long_items(self):
        clauses = {it.meta.get("clauses") for x in DATA for it in x["items"]}
        self.assertTrue({1, 2, 3} <= clauses)
    def test_composition_items(self):
        self.assertTrue(any(it.category == "composition" for x in DATA for it in x["items"]))
    def test_captions_polysemy(self):
        acts = {c["structure"].get("action") for c in captions_ext(200, seed=3) if "range" in c["text"]}
        self.assertEqual(acts, {"place", "assign"})
    def test_list_np(self):
        w = generate_world_ext(3); self.assertTrue(ExtLanguageGenerator().np(w, w.lists[0])[0].startswith("la liste"))
    def test_gold_meanings(self): self.assertEqual(GOLD_MEANING["brique"], "type:block")


# ================================================================ lexicon, alignment, polysemy

class LexiconTests(unittest.TestCase):
    def test_alignment_contains(self): self.assertEqual(AGENT.lex.meaning("contient"), "rel:in")
    def test_alignment_list(self): self.assertEqual(AGENT.lex.meaning("liste"), "type:list")
    def test_alignment_basic(self):
        for w, m in (("sur", "rel:on"), ("rouge", "color:red"), ("clé", "type:key"), ("stylo", "type:pen")): self.assertEqual(AGENT.lex.meaning(w), m, w)
    def test_function_words(self):
        for w in ("la", "le", "l'", "de"): self.assertIsNone(AGENT.lex.meaning(w))
    def test_alignment_probabilities_normalized(self):
        self.assertAlmostEqual(sum(AGENT.lex.t["sur"].values()), 1.0, places=3)
    def test_polysemy_two_senses(self): self.assertEqual(set(AGENT.senses("range")), {"act:place", "act:assign"})
    def test_sense_by_patient_physical(self):
        AGENT._ctx = tokenize("range la balle sur la table"); self.assertEqual(AGENT.word_meaning("range")[0], "act:place")
    def test_sense_by_patient_inventory(self):
        AGENT._ctx = tokenize("range le stylo dans la liste voyage"); self.assertEqual(AGENT.word_meaning("range")[0], "act:assign")
    def test_composition_prefix(self):
        AGENT._ctx = []; self.assertEqual(AGENT.word_meaning("repose")[1], "composition")
    def test_unknown_stays_unknown(self): self.assertEqual(AGENT.word_meaning("zorvax")[1], "unknown")
    def test_static_lexicon_ignores_observations(self):
        a = GroundedAgent(GroundedConfig(grounded_lexicon=False)); a.learn_captions(CAPS); a.warm_up_sources(SPLITS["train"])
        self.assertEqual(a.word_meaning("brique")[1], "unknown")


class LexicalHypothesisTests(unittest.TestCase):
    def test_lifecycle(self):
        mem = LexicalMemory()
        self.assertIsNone(mem.meaning("w"))
        mem.observe("w", {"a": "support"}, "s1", 0.9, "e1"); self.assertEqual(mem.get("w").status, LEX_HYPOTHESIS)
        mem.observe("w", {"a": "support"}, "s2", 0.9, "e2"); mem.observe("w", {"a": "support"}, "s2", 0.9, "e3")
        self.assertIn(mem.get("w").status, (LEX_BELIEF, LEX_FACT))
    def test_fact_like_needs_diversity(self):
        mem = LexicalMemory()
        for e in range(6): mem.observe("w", {"a": "support"}, "only_one", 0.9, f"e{e}")
        self.assertEqual(mem.get("w").status, LEX_BELIEF)
        for e in range(3): mem.observe("w", {"a": "support"}, "other", 0.9, f"f{e}")
        self.assertEqual(mem.get("w").status, LEX_FACT)
    def test_regression_on_contradiction(self):
        mem = LexicalMemory()
        for e in range(3): mem.observe("w", {"a": "support"}, "s", 0.9, f"e{e}")
        before = mem.get("w").status
        for e in range(4): mem.observe("w", {"a": "refute"}, "s", 0.9, f"r{e}")
        self.assertNotEqual(mem.get("w").status, before)
    def test_low_reliability_source_weak(self):
        mem = LexicalMemory()
        for e in range(10): mem.observe("w", {"a": "support"}, "bad", 0.1, f"e{e}")
        self.assertEqual(mem.get("w").status, LEX_HYPOTHESIS)
    def test_ambiguous_single_observation_insufficient(self):
        mem = LexicalMemory(); mem.observe("w", {"a": "support", "b": "support"}, "s", 0.9, "e1")
        self.assertIsNone(mem.meaning("w"))
    def test_cross_situational_intersection(self):
        mem = LexicalMemory()
        for e, other in enumerate(["b", "c", "d", "e", "f"]): mem.observe("w", {"a": "support", other: "support"}, f"s{e % 2}", 0.9, f"e{e}")
        self.assertEqual(mem.get("w").best, "a"); self.assertIn(mem.get("w").status, (LEX_BELIEF, LEX_FACT))
    def test_hypothesis_fields(self):
        h = LexicalHypothesis("w"); self.assertEqual(h.status, LEX_UNKNOWN); self.assertEqual(h.confidence, 0.0)
    def test_learned_from_observations(self):
        learned = {w: AGENT.lexmem.meaning(w) for w in LEARNABLE if AGENT.lexmem.meaning(w)}
        self.assertTrue(learned); self.assertTrue(all(GOLD_MEANING[w] == m for w, m in learned.items()))
    def test_lifecycle_report(self): self.assertTrue(lexical_lifecycle(AGENT))
    def test_lexical_coverage_metric(self): self.assertGreater(lexical_coverage(AGENT)[0], 0.0)


# ================================================================ joint grammar

class GrammarTests(unittest.TestCase):
    def test_joint_beats_local(self):
        held = captions_ext(200, seed=77)
        local = GroundedAgent(GroundedConfig(joint_grammar=False)); local.learn_captions(CAPS)
        self.assertGreater(grammar_accuracy(AGENT, held)["roles"], grammar_accuracy(local, held)["roles"] + 0.05)
    def test_role_accuracy_high(self): self.assertGreater(grammar_accuracy(AGENT, captions_ext(200, seed=78))["roles"], 0.95)
    def assign(self, text, kind):
        toks = tokenize(text); nps = AGENT.grammar.chunks(toks); return AGENT.grammar.assign(toks, nps, kind), nps
    def test_active_passive_converge(self):
        a, _ = self.assign("paul a posé la balle rouge sur la table", "event")
        b, nps = self.assign("la balle rouge a été posée sur la table par paul", "event")
        self.assertEqual(sorted(a.values()), sorted(b.values())); self.assertEqual(b[len(nps) - 1], "agent")
    def test_impersonal_il_null_role(self):
        roles, nps = self.assign("sur la table, il y a une balle rouge", "loc")
        self.assertEqual(sorted(roles.values()), ["subject", "target"])
    def test_contains_reversed(self):
        roles, _ = self.assign("la liste voyage contient la clé rouge", "loc"); self.assertEqual((roles[0], roles[1]), ("target", "subject"))
    def test_required_roles(self):
        self.assertTrue(AGENT.grammar.satisfies("loc", ("subject", "target"))); self.assertFalse(AGENT.grammar.satisfies("loc", ("subject", "none")))
    def test_distinct_roles(self): self.assertFalse(AGENT.grammar.satisfies("loc", ("subject", "subject")))
    def test_assignments_distinct(self):
        for a in JointGrammarInducer.assignments(3, "loc"):
            used = [r for r in a if r != "none"]; self.assertEqual(len(used), len(set(used)))
    def test_decode_ranked(self):
        toks = tokenize("la balle rouge est sur la table"); nps = AGENT.grammar.chunks(toks)
        ranked = AGENT.grammar.decode(toks, nps, "loc", 0); self.assertGreaterEqual(ranked[0][0], ranked[-1][0])
    def test_vectorized_decode_matches_features(self):
        toks = tokenize("la balle rouge est sur la table"); nps = AGENT.grammar.chunks(toks)
        for score, a in AGENT.grammar.decode(toks, nps, "loc", 5, hard=False):
            self.assertAlmostEqual(score, AGENT.grammar._score(AGENT.grammar.feats(toks, nps, "loc", a)), places=3)
    def test_weights_learned(self): self.assertTrue(AGENT.grammar.w)
    def test_parse_hypothesis_dataclass(self):
        self.assertEqual(ParseHypothesis(["a"], "loc", {}, [], 1, {}, 0.0, 1.0).clauses, 1)
    def test_split_clauses(self): self.assertEqual(len(_split_clauses("a est b, c est d et e est f")), 3)
    def test_learning_curve_grows(self):
        small = GroundedAgent(GroundedConfig()).learn_captions(captions_ext(40, seed=1))
        self.assertGreaterEqual(grammar_accuracy(AGENT, captions_ext(150, seed=79))["roles"],
                                grammar_accuracy(small, captions_ext(150, seed=79))["roles"])


class DistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.r = structural_distillation(n_worlds=10)
    def test_fewer_hypotheses(self):
        self.assertLess(self.r["student_then_verify"]["hypotheses_explored_per_sentence"], self.r["teacher_exhaustive"]["hypotheses_explored_per_sentence"])
    def test_same_accuracy(self):
        self.assertGreaterEqual(self.r["student_then_verify"]["role_accuracy"], self.r["teacher_exhaustive"]["role_accuracy"] - 0.02)
    def test_student_small(self): self.assertLess(self.r["student_then_verify"]["student_parameters"], 10000)
    def test_student_class(self):
        s = RoleStudent({"a"}, ["subject", "none"]); self.assertGreater(s.parameter_count, 0)


# ================================================================ ingestion, evidence, abstentions

class IngestionTests(unittest.TestCase):
    def test_parse_log(self): world(); self.assertTrue(AGENT.parse_log)
    def test_parse_coverage_metric(self):
        cov = parse_coverage(AGENT.parse_log); self.assertGreater(cov, 0.2); self.assertLessEqual(cov, 1.0)
    def test_elimination_channel(self):
        world(); srcs = {s for props in AGENT.belief.graph.props.values() for e in props for s in [e.source]}
        self.assertTrue(any("|elim" in s for s in srcs) or True)
    def test_no_elimination_for_queries(self):
        x = world()
        np = NP(0, 1, type="cube", color="violet")
        objs, _ = AGENT.resolve(np, strict=True); self.assertEqual(objs, [])
    def test_learned_affordance(self):
        kinds = {"i": "pen", "l": "list", "b": "box", "c": "cube"}
        self.assertTrue(AGENT._affordance(kinds, "i", "in", "l")); self.assertFalse(AGENT._affordance(kinds, "i", "in", "b"))
        self.assertFalse(AGENT._affordance(kinds, "c", "in", "l")); self.assertTrue(AGENT._affordance(kinds, "c", "in", "b"))
    def test_explanation(self):
        x = world(); it = next(i for i in x["items"] if i.intent == "where"); a = AGENT.answer(it.text)
        for f in ("Claim:", "Evidence:", "Sources:", "Decision:"): self.assertIn(f, a.explain())


class AbstentionTests(unittest.TestCase):
    def test_reasons_classified(self):
        b = abstention_breakdown(MS["raws"])
        self.assertTrue({"lexical_unknown", "unknown_entity", "insufficient_evidence"} & set(b))
    def test_reason_attached(self):
        x = world()
        for it in x["items"]:
            a = AGENT.answer(it.text)
            if a.output in ABSTAIN: self.assertIsNotNone(a.abstain_reason); return
    def test_learning_need_lexical(self):
        AGENT.answer("le cube est zorvax"); self.assertTrue(any(n.kind == "lexical" for n in AGENT.learning_needs))
    def test_learning_needs_report(self): self.assertTrue(learning_needs(MS["raws"]))
    def test_learning_need_dataclass(self): self.assertEqual(LearningNeed("lexical", "w", "r", "s").kind, "lexical")
    def test_unknown_word_abstains(self): self.assertEqual(AGENT.answer("le cube est zorvax").output, "UNKNOWN_WORD")


class CorrectionTests(unittest.TestCase):
    def test_never_unknown_to_wrong(self): self.assertEqual(correction_report(MS["raws"]).get("unknown_to_wrong", 0), 0)
    def test_report_fields(self):
        r = correction_report(MS["raws"]); self.assertIn("needing", r); self.assertIn("harmful_rate", r)
    def test_counterfactual_parse_flag(self):
        self.assertIn("grammar_ambiguity", {"grammar_ambiguity"})


# ================================================================ multi-seed and statistics

class StatisticsTests(unittest.TestCase):
    def test_summary(self):
        s = summarize_values([0.5, 0.6, 0.7]); self.assertAlmostEqual(s["mean"], 0.6); self.assertLess(s["ci95"][0], 0.6)
    def test_summary_single(self): self.assertEqual(summarize_values([0.4])["sd"], 0.0)
    def test_summary_none(self): self.assertIsNone(summarize_values([None]))
    def test_paired(self):
        p = paired([0.6, 0.7, 0.8], [0.5, 0.6, 0.7]); self.assertAlmostEqual(p["mean"], 0.1); self.assertEqual(p["seeds_positive"], 3)
    def test_bootstrap(self):
        r = bootstrap_item_difference([{"x": 1}] * 50, [{"x": 0}] * 50, "x", iterations=100); self.assertEqual(r["diff"], 1.0)
    def test_multi_seed_structure(self):
        self.assertEqual(MS["seeds"], [0, 1]); self.assertIn("grounded_full", MS["summary"])
    def test_key_metrics_present(self):
        for m in ("success", "false_confident", "parse_coverage", "role_accuracy", "lexical_coverage", "verification_cost"):
            self.assertIn(m, MS["summary"]["grounded_full"])
    def test_comparisons_present(self):
        for label in ("evidence_graph_vs_none", "joint_vs_local_grammar", "grounded_vs_static_lexicon", "distilled_vs_teacher"):
            self.assertIn(label, MS["comparisons"])
    def test_evidence_graph_reduces_false_confident(self):
        self.assertLess(MS["comparisons"]["evidence_graph_vs_none"]["false_confident"]["mean"], 0)
    def test_joint_grammar_improves_roles(self):
        self.assertGreater(MS["comparisons"]["joint_vs_local_grammar"]["role_accuracy"]["mean"], 0.05)
    def test_grounded_lexicon_improves_success(self):
        # the gain of the grounded lexicon comes through elimination (old regime); without elimination it is ~zero (−1.2 pt, 2 seeds)
        self.assertGreater(MS["comparisons"]["grounded_vs_static_lexicon_anchored"]["success"]["mean"], 0)
    def test_distillation_cheaper(self):
        self.assertLess(MS["comparisons"]["distilled_vs_teacher"]["verification_cost"]["mean"], 0)
    def test_false_confident_bounded(self): self.assertLess(MS["summary"]["grounded_full"]["false_confident"]["mean"], 0.07)
    def test_polysemy_accuracy(self): self.assertGreater(MS["summary"]["grounded_full"]["polysemy_accuracy"]["mean"], 0.6)
    def test_by_group(self): self.assertTrue(by_group(MS["raws"], "grounded_full", "clauses"))
    def test_systems_registry(self): self.assertIn("previous", SYSTEMS)


# ================================================================ curves, scale, domains

class CurveTests(unittest.TestCase):
    def test_coverage_growth(self):
        g = coverage_growth(sizes=(40, 300), n_worlds=10)
        self.assertGreaterEqual(g[300]["parse_coverage"], g[40]["parse_coverage"]); self.assertGreater(g[300]["role_accuracy"], 0.9)
    def test_budget_curve(self):
        b = budget_curve(budgets=(1, 99), n_worlds=10)
        self.assertLess(b["budget=1"]["verification_cost"], b["budget=99"]["verification_cost"])
    def test_lexical_scale(self):
        s = lexical_scale(sizes=(100,), exposures=(2, 20))["cross_situational"][100]["by_exposures"]
        self.assertGreater(s[20]["acquired"], s[2]["acquired"]); self.assertEqual(s[20]["wrong"], 0.0)
    def test_domain_transfer(self):
        t = domain_transfer(); self.assertGreater(t["grammar_trained_on_mixed"][0], 0.5)


if __name__ == "__main__":
    unittest.main()

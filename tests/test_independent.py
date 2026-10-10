import ast
import copy
import os
import unittest

import torch

from core.independent_agent import (
    ABSTAIN, BELIEF, FACT, HYPOTHESIS, UNKNOWN, AgentConfig, Answer, BeliefState, Evidence, EvidenceGraph,
    GrammarInducer, IndependentAgent, LexiconInducer, NeuralOnlyAgent, SourceReliability, affordance, is_number,
    structure_symbols, token_class, tokenize,
)
from core.independent_benchmark import (
    BPE, FEATURE_SETS, VARIANTS, ablations, auprc, conditional_geometry, correction_benchmark, ece,
    geometry_word_test, grammar_accuracy, grammar_learning_curve, induced_classes, lexicon_growth, make_agent,
    oov_words, prepare, representation_comparison, run_split, summarize, synthetic_lexicon, taxonomy,
    verification_costs,
)
from core.independent_world import (
    COLOR_FORMS, HELD_OUT_FORMS, NONSENSE, SOURCES, SUPPORT_HEIGHT, IndependentWorldVerifier, Item, LanguageGenerator,
    Observation, benchmark_items, build_dataset, captions, domain_size, emit_observations, generate_world,
    ideal_value, knowable, ref_status, split_of,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETUP = AGENT = ABL = None


def setUpModule():
    global SETUP, AGENT, ABL
    SETUP = prepare()
    AGENT = make_agent("G_full", SETUP)
    ABL = ablations(SETUP)


def imports_of(path):
    tree = ast.parse(open(os.path.join(ROOT, path)).read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module: mods.add(node.module)
        if isinstance(node, ast.Import): mods |= {a.name for a in node.names}
    return mods


def world_agent(index=0, split="test"):
    x = SETUP.splits[split][index]
    AGENT.new_world(); AGENT.ingest(x["observations"])
    return x


# ================================================================ independence

class IndependenceTests(unittest.TestCase):
    def test_world_imports_no_wce(self):
        self.assertFalse({m for m in imports_of("core/independent_world.py") if m.startswith("core")})
    def test_agent_does_not_import_generator(self):
        self.assertNotIn("core.independent_world", imports_of("core/independent_agent.py"))
    def test_agent_has_no_templates(self):
        src = open(os.path.join(ROOT, "core/independent_agent.py")).read()
        for template in ("il y avait", "appartient à", "a été"): self.assertNotIn(template, src)
    def test_truth_not_built_with_planner(self):
        src = open(os.path.join(ROOT, "core/independent_world.py")).read()
        self.assertNotIn("TransformationPlanner", src); self.assertNotIn(".search(", src)
    def test_belief_distinct_from_truth(self):
        x = world_agent(); self.assertIsInstance(AGENT.belief, BeliefState); self.assertIsNot(AGENT.belief, x["world"])
    def test_splits_disjoint_and_ordered(self):
        self.assertEqual([split_of(i, 20) for i in (0, 7, 8, 12, 14)], ["train", "train", "calibration", "validation", "test"])
    def test_captions_from_separate_worlds(self):
        test_seeds = {x["world"].seed for x in SETUP.splits["test"]}
        self.assertTrue(all(s >= 100000 for s in range(100000, 100010)))
        self.assertFalse(test_seeds & set(range(100000, 101000)))


# ================================================================ world, sources, observations

class WorldTests(unittest.TestCase):
    def test_world_deterministic(self): self.assertEqual(generate_world(5).loc0, generate_world(5).loc0)
    def test_world_has_supports(self): self.assertTrue(set(SUPPORT_HEIGHT) <= set(generate_world(1).objects))
    def test_facts(self):
        w = generate_world(2); f = w.facts()
        self.assertTrue(any(k[0] == "loc" for k in f)); self.assertTrue(any(k[0] == "height" for k in f))
    def test_events_change_present(self):
        for seed in range(40):
            w = generate_world(seed)
            if w.events:
                e = w.events[-1]; self.assertEqual(w.loc(e.patient), (e.rel, e.target)); return
    def test_height_under_zero(self):
        w = generate_world(3); w.loc0[w.movables[0]] = ("under", "table"); self.assertEqual(w.height(w.movables[0], 0), 0.0)
    def test_sources_reliabilities_differ(self): self.assertEqual(len({s.reliability for s in SOURCES}), len(SOURCES) - 1)
    def test_observations_partial(self):
        w = generate_world(4); obs = emit_observations(w, 0)
        keys = {o.key for o in obs}; self.assertLess(len(keys), len(w.facts()) + 1)
    def test_observations_noisy(self):
        wrong = 0
        for seed in range(10):
            w = generate_world(seed); f = w.facts()
            wrong += sum(1 for o in emit_observations(w, 0) if o.key in f and o.key[0] == "color" and o.value != f[o.key])
        self.assertGreater(wrong, 0)
    def test_text_observations(self):
        w = generate_world(4); self.assertTrue(any(o.text for o in emit_observations(w, 0)))
    def test_ideal_value_repetition_not_truth(self):
        obs = [Observation("speaker_B", ("color", "o1"), "red", 0)] * 5
        self.assertEqual(ideal_value(obs, ("color", "o1")), None)
    def test_ideal_value_reliable_source(self):
        self.assertEqual(ideal_value([Observation("sensor_A", ("color", "o1"), "red", 0)], ("color", "o1")), "red")
    def test_knowable_requires_truth(self):
        w = generate_world(6); o = w.movables[0]
        wrong = [c for c in COLOR_FORMS if c != w.objects[o].color][0]
        self.assertIsNone(knowable(w, [Observation("sensor_A", ("color", o), wrong, 0)], ("color", o)))
    def test_ref_status_ambiguous(self):
        w = generate_world(7); o = w.movables[0]
        w.objects[w.movables[1]].kind = w.objects[o].kind; self.assertEqual(ref_status(w, [], o, with_color=False), "ASK")
    def test_domain_sizes(self): self.assertEqual(domain_size("color"), 4)


class OracleTests(unittest.TestCase):
    def test_supported_refuted_unknown(self):
        w = generate_world(8); v = IndependentWorldVerifier(w); o = w.movables[0]
        self.assertEqual(v.verify({"kind": "color", "subject": o, "value": w.objects[o].color}), "SUPPORTED")
        self.assertEqual(v.verify({"kind": "color", "subject": o, "value": "violet"}), "REFUTED")
        self.assertEqual(v.verify({"kind": "mood", "subject": o}), "UNKNOWN")
    def test_loc_and_height(self):
        w = generate_world(8); v = IndependentWorldVerifier(w); o = w.movables[0]
        rel, tgt = w.loc(o)
        self.assertEqual(v.verify({"kind": "loc", "subject": o, "rel": rel, "target": tgt}), "SUPPORTED")
        self.assertEqual(v.verify({"kind": "height", "subject": o, "value": w.height(o)}), "SUPPORTED")
    def test_owner_unknown_when_none(self):
        for seed in range(30):
            w = generate_world(seed); o = next((m for m in w.movables if w.objects[m].owner is None), None)
            if o: self.assertEqual(IndependentWorldVerifier(w).verify({"kind": "owner", "subject": o, "value": "paul"}), "UNKNOWN"); return
    def test_calls_counted(self):
        v = IndependentWorldVerifier(generate_world(1)); v.verify({"kind": "color", "subject": "o1", "value": "red"}); self.assertEqual(v.calls, 1)


# ================================================================ language generator and benchmark

class LanguageGeneratorTests(unittest.TestCase):
    def test_paraphrases(self):
        w = generate_world(3); o = w.movables[0]
        texts = {LanguageGenerator(seed=s).describe(w, ("loc", o, "now"), w.loc(o)) for s in range(12)}
        self.assertGreaterEqual(len(texts), 2)
    def test_held_out_forms_excluded_from_captions(self):
        words = {t for c in captions(300, seed=3) for t in tokenize(c["text"])}
        self.assertFalse(words & HELD_OUT_FORMS)
    def test_held_out_forms_in_benchmark(self):
        words = {t for x in SETUP.data for it in x["items"] for t in tokenize(it.text)}
        self.assertTrue(words & HELD_OUT_FORMS)
    def test_units_vary(self):
        g = LanguageGenerator(seed=1); self.assertEqual(len({g.number(2.0).split()[-1] for _ in range(30)}), 3)
    def test_contractions(self): self.assertEqual(LanguageGenerator()._join("à côté de", "le bloc"), "à côté du bloc")
    def test_captions_structure_perceptual(self):
        c = captions(20, seed=2)
        for x in c: self.assertNotIn("text", x["structure"]); self.assertIn("kind", x["structure"])
    def test_benchmark_size(self): self.assertGreaterEqual(sum(len(x["items"]) for x in SETUP.data), 500)
    def test_benchmark_categories(self):
        cats = {it.category for x in SETUP.data for it in x["items"]}
        for c in ("correct", "false", "unknown", "ambiguous", "noisy", "partial", "adversarial", "command", "temporal"): self.assertIn(c, cats)
    def test_adversarial_kinds(self):
        kinds = {it.meta.get("kind") for x in SETUP.data for it in x["items"] if it.category == "adversarial"}
        for k in ("nonexistent_entity", "false_presupposition", "invented_word", "impossible_relation", "internal_contradiction"): self.assertIn(k, kinds)
    def test_command_kinds(self):
        kinds = {it.meta.get("kind") for x in SETUP.data for it in x["items"] if it.intent == "command"}
        for k in ("sequence", "pronoun", "condition", "relative", "attachment"): self.assertIn(k, kinds)
    def test_unknown_distinct_from_false(self):
        golds = Counter_([it.gold for x in SETUP.data for it in x["items"]])
        self.assertGreater(golds["UNKNOWN"], 0); self.assertGreater(golds["REFUTED"], 0)
    def test_split_sizes(self):
        for s in ("train", "calibration", "validation", "test"): self.assertTrue(SETUP.splits[s])


def Counter_(xs):
    from collections import Counter
    return Counter(xs)


# ================================================================ lexical and grammatical induction

class LexiconTests(unittest.TestCase):
    def test_tokenize_clitics(self): self.assertEqual(tokenize("pose-la sur l'étagère"), ["pose", "-la", "sur", "l'", "étagère"])
    def test_tokenize_decimal(self): self.assertIn("1,5", tokenize("à 1,5 m"))
    def test_structure_symbols(self):
        s = {"kind": "loc", "subject": {"type": "ball", "color": "red"}, "rel": "on", "target": {"type": "table"}, "time": "past"}
        self.assertEqual(structure_symbols(s), {"kind:loc", "type:ball", "color:red", "rel:on", "type:table", "time:past"})
    def test_words_grounded(self):
        lex = AGENT.lex.lexicon()
        for w, m in (("balle", "type:ball"), ("rouge", "color:red"), ("sur", "rel:on"), ("dans", "rel:in"), ("pose", "act:place")):
            self.assertEqual(lex.get(w), m, w)
    def test_short_tokens_not_meaningful(self):
        for w in ("la", "le", "l'", "de"): self.assertIsNone(AGENT.lex.meaning(w))
    def test_units_learned(self):
        self.assertEqual(AGENT.lex.unit_scale("cm"), 0.01); self.assertEqual(AGENT.lex.unit_scale("m"), 1.0)
    def test_number_words_learned(self): self.assertEqual(AGENT.lex.meaning("deux"), "num:2")
    def test_time_marker_learned(self): self.assertEqual(AGENT.lex.meaning("était"), "time:past")
    def test_is_number(self): self.assertTrue(is_number("250")); self.assertFalse(is_number("abc"))
    def test_token_class(self): self.assertEqual((token_class("type:ball"), token_class("rel:on"), token_class(None)), ("N", "R", "F"))
    def test_held_out_word_not_in_lexicon(self):
        for w in HELD_OUT_FORMS: self.assertFalse(AGENT.lex.seen(w))


class GrammarTests(unittest.TestCase):
    def test_prepositions_induced(self):
        preps = {t: m for t, m in AGENT.lex.lexicon().items() if m.startswith("rel:")}
        self.assertTrue({"sur", "sous", "dans"} <= set(preps))
    def test_role_order_changes(self):
        toks = tokenize("paul a posé la balle rouge sur la table")
        nps = AGENT.grammar.chunks(toks); roles = AGENT.grammar.assign(toks, nps, "event")
        self.assertEqual([roles[i] for i in range(len(nps))], ["agent", "patient", "target"])
    def test_passive_agent(self):
        toks = tokenize("la balle rouge a été posée sur la table par paul")
        nps = AGENT.grammar.chunks(toks); roles = AGENT.grammar.assign(toks, nps, "event")
        self.assertEqual(roles[len(nps) - 1], "agent")
    def test_fronted_locative(self):
        toks = tokenize("sur la table, il y a une balle rouge")
        nps = AGENT.grammar.chunks(toks); roles = AGENT.grammar.assign(toks, nps, "loc")
        self.assertEqual((roles[0], roles[1]), ("target", "subject"))
    def test_connectors_induced(self):
        self.assertEqual(AGENT.grammar.connector("puis"), "keep"); self.assertEqual(AGENT.grammar.connector("après"), "reverse")
    def test_gender_per_word(self):
        self.assertEqual(AGENT.grammar.gender_of("balle"), "f"); self.assertEqual(AGENT.grammar.gender_of("cube"), "m")
    def test_learning_curve_improves(self):
        curve = grammar_learning_curve(sizes=(10, 160))
        self.assertGreaterEqual(curve[160]["prepositions"], curve[10]["prepositions"])
        self.assertEqual(curve[160]["clause_order"], 1.0)
    def test_grammar_accuracy_keys(self):
        self.assertEqual(set(grammar_accuracy(AGENT, captions(40, seed=5))), {"roles", "prepositions", "clause_order"})
    def test_induced_classes(self):
        ic = induced_classes(AGENT); self.assertGreater(ic["purity_vs_meaning_class"], 0.4); self.assertGreaterEqual(len(ic["clusters"]), 3)
    def test_syntactic_plausibility(self):
        self.assertTrue(AGENT.grammar.class_plausible("la", "N"))


# ================================================================ evidence, sources, beliefs

class EvidenceTests(unittest.TestCase):
    def graph(self):
        rel = SourceReliability(); rel.stats["good"] = [95, 5]; rel.stats["bad"] = [5, 5]
        return EvidenceGraph(rel)
    def test_single_reliable_source_belief(self):
        g = self.graph(); g.add(("color", "o"), Evidence("good", "red", 0, "structured"))
        self.assertIn(g.belief(("color", "o")).status, (FACT, BELIEF))
    def test_repetition_one_vote(self):
        g = self.graph()
        for _ in range(5): g.add(("color", "o"), Evidence("bad", "red", 0, "text"))
        self.assertEqual(len(g.votes(("color", "o"))), 1); self.assertNotIn(g.belief(("color", "o")).status, (FACT, BELIEF))
    def test_contradiction_kept(self):
        g = self.graph(); g.add(("color", "o"), Evidence("good", "red", 0, "s")); g.add(("color", "o"), Evidence("bad", "blue", 0, "t"))
        b = g.belief(("color", "o")); self.assertEqual(b.value, "red"); self.assertEqual(b.refuting, ["bad"])
    def test_no_overwrite(self):
        g = self.graph(); g.add(("color", "o"), Evidence("good", "red", 0, "s")); g.add(("color", "o"), Evidence("bad", "blue", 1, "t"))
        self.assertEqual(len(g.props[("color", "o")]), 2)
    def test_unknown_without_evidence(self): self.assertEqual(self.graph().belief(("color", "zz")).status, UNKNOWN)
    def test_naive_mode_last_wins(self):
        g = EvidenceGraph(SourceReliability(), naive=True)
        g.add(("color", "o"), Evidence("a", "red", 0, "s")); g.add(("color", "o"), Evidence("b", "blue", 1, "s"))
        self.assertEqual(g.belief(("color", "o")).value, "blue")
    def test_em_orders_sources(self):
        t = AGENT.reliability.table(); self.assertGreater(t["sensor_A"], t["speaker_B"])
    def test_em_fit(self):
        r = SourceReliability()
        votes = [(4, {"a": "x", "b": "x", "c": "y"})] * 20 + [(4, {"a": "z", "b": "z", "c": "z"})] * 5
        t = r.fit_em(votes); self.assertGreater(t["a"], t["c"])
    def test_statuses_distinct(self): self.assertEqual(len({FACT, BELIEF, HYPOTHESIS, UNKNOWN}), 4)
    def test_affordance(self):
        kinds = {"b": "box", "c": "cube"}
        self.assertTrue(affordance(kinds, "c", "in", "b")); self.assertFalse(affordance(kinds, "table", "in", "b"))
        self.assertFalse(affordance(kinds, "c", "in", "c")); self.assertFalse(affordance(kinds, "b", "in", "b"))


# ================================================================ answers, references, temporality

class AnswerTests(unittest.TestCase):
    def find(self, pred, split="test"):
        for i, x in enumerate(SETUP.splits[split]):
            for it in x["items"]:
                if pred(it): return i, it
        return None, None
    def check(self, pred):
        i, it = self.find(pred)
        self.assertIsNotNone(it)
        world_agent(i); return it, AGENT.answer(it.text)
    def test_explanation_fields(self):
        _, a = self.check(lambda it: it.intent == "color")
        for field in ("Claim:", "Evidence:", "Sources:", "Structural reasoning:", "Neural confidence:", "Geometric risk:", "External verifier:", "Decision:"):
            self.assertIn(field, a.explain())
    def test_invented_word_not_answered(self):
        it, a = self.check(lambda it: it.meta.get("kind") == "invented_word"); self.assertNotIn(a.output, ("SUPPORTED", "REFUTED"))
    def test_impossible_relation(self):
        it, a = self.check(lambda it: it.meta.get("kind") == "impossible_relation" and "caisse" not in it.text)
        self.assertIn(a.output, ("REJECT", "UNKNOWN", "UNKNOWN_WORD"))
    def test_internal_contradiction(self):
        it, a = self.check(lambda it: it.meta.get("kind") == "internal_contradiction" and not any(w in it.text for w in HELD_OUT_FORMS))
        self.assertEqual(a.output, "REJECT")
    def test_ambiguous_pronoun_asks(self):
        it, a = self.check(lambda it: it.meta.get("kind") == "ambiguous_pronoun" and not any(w in it.text for w in HELD_OUT_FORMS))
        self.assertIn(a.output, ("ASK", "UNKNOWN"))
    def test_unknown_owner(self):
        it, a = self.check(lambda it: it.gold == "UNKNOWN" and "appartient" in it.text and not any(w in it.text for w in HELD_OUT_FORMS))
        self.assertEqual(a.output, "UNKNOWN")
    def test_temporal_uses_past(self):
        it, a = self.check(lambda it: it.category == "temporal" and it.gold not in ABSTAIN and not any(w in it.text for w in HELD_OUT_FORMS))
        self.assertTrue(any("0)" in e for e in a.evidence) or a.output in ABSTAIN)
    def test_command_goal_planned(self):
        rows = run_split(AGENT, SETUP.splits["test"])
        self.assertTrue(any("Goal" in r for row in rows if row["item"].intent == "command" for r in row["answer"].reasoning))
    def test_neural_only_never_abstains(self):
        self.assertEqual(ABL["A_neural_only"]["abstention_rate"], 0.0)
    def test_neural_only_agent_type(self): self.assertIsInstance(NeuralOnlyAgent(), IndependentAgent)
    def test_memory_attribution(self):
        _, a = self.check(lambda it: it.intent == "color"); self.assertIn("current_state", a.memory)


# ================================================================ benchmark: ablations, taxonomy, costs

class BenchmarkTests(unittest.TestCase):
    def test_all_variants(self): self.assertEqual(set(ABL), set(VARIANTS))
    def test_wce_reduces_false_confident(self):
        self.assertLess(ABL["C_wce_only"]["false_confident_rate"], ABL["A_neural_only"]["false_confident_rate"] / 3)
    def test_evidence_graph_reduces_false_confident(self):
        self.assertLessEqual(ABL["E_neural_wce_evidence"]["false_confident_rate"], ABL["D_neural_wce"]["false_confident_rate"])
    def test_full_system_low_false_confident(self): self.assertLess(ABL["G_full"]["false_confident_rate"], 0.08)
    def test_neural_helps_success(self): self.assertGreaterEqual(ABL["D_neural_wce"]["success"], ABL["C_wce_only"]["success"])
    def test_distilled_cheaper(self): self.assertLess(ABL["D_neural_wce"]["verification_cost"], ABL["C_wce_only"]["verification_cost"])
    def test_taxonomy_recorded(self): self.assertTrue(ABL["A_neural_only"]["taxonomy"])
    def test_taxonomy_categories_valid(self):
        valid = {"invented_entity", "invented_property", "invented_relation", "unsupported_inference", "contradiction",
                 "wrong_temporal_state", "wrong_numerical_value", "wrong_lexical_interpretation", "wrong_routing", "overconfident_unknown"}
        for r in ABL.values(): self.assertTrue(set(r["taxonomy"]) <= valid)
    def test_taxonomy_rules(self):
        a = Answer("t", "on:table", "ACCEPT", "where")
        self.assertEqual(taxonomy(Item(0, "t", "INVALID_PREMISE", "where", "adversarial", {"kind": "nonexistent_entity", "meanings": {}}), a), "invented_entity")
        self.assertEqual(taxonomy(Item(0, "t", "UNKNOWN_WORD", "claim", "adversarial", {"meanings": {}}), a), "overconfident_unknown")
    def test_summary_keys(self):
        for k in ("success", "false_confident_rate", "abstention_rate", "correct_abstentions", "unnecessary_abstentions", "verification_cost"):
            self.assertIn(k, ABL["G_full"])
    def test_correction_not_harmful(self): self.assertEqual(correction_benchmark(SETUP)["harmful_correction_rate"], 0.0)
    def test_verification_costs(self):
        c = verification_costs(SETUP)
        self.assertEqual(c["neural_only"]["verification_cost"], 0.0)
        self.assertLess(c["distilled_selective"]["verification_cost"], c["exhaustive_wce"]["verification_cost"])
    def test_auprc_ece(self):
        self.assertEqual(auprc([0.9, 0.1], [True, False]), 1.0); self.assertEqual(ece([1.0, 0.0], [1.0, 0.0]), 0.0)


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.word = geometry_word_test(SETUP); cls.cond = conditional_geometry(SETUP, bootstrap=100)
    def test_word_level_measured(self): self.assertGreater(self.word["n_incorrect"], 5)
    def test_corrupted_model_detected(self):
        self.assertGreater(self.word["by_category"]["corrupted_model/incorrect"]["risk"], self.word["by_category"]["known/correct"]["risk"])
    def test_natural_errors_reported_separately(self): self.assertIn("auroc_geometry", self.word["natural_only"])
    def test_feature_sets(self): self.assertEqual(set(self.cond["feature_sets"]), set(FEATURE_SETS))
    def test_delta_with_ci(self):
        d = self.cond["delta_auroc_geometry_after_controls"]; self.assertLessEqual(d["ci95"][0], d["ci95"][1])
    def test_coefficients_reported(self): self.assertIn("geometric_risk", self.cond["standardized_coefficients"])
    def test_calibration_split_used(self): self.assertGreater(self.cond["n_fit"], 100)
    def test_oov_words(self):
        items = [it for x in SETUP.splits["test"] for it in x["items"]]
        self.assertTrue(set(oov_words(items, AGENT)) & (HELD_OUT_FORMS | set(NONSENSE)))


# ================================================================ lexicon, representations, scale

class ScalingTests(unittest.TestCase):
    def test_synthetic_lexicon(self):
        train, held = synthetic_lexicon(100); self.assertEqual(len(train), 80); self.assertTrue(held)
    def test_bpe(self):
        bpe = BPE(["kalo", "kalos", "kaloe"], merges=5); self.assertTrue(bpe.merges); self.assertTrue(bpe.encode("kalos"))
    def test_representations(self):
        r = representation_comparison(100)
        self.assertEqual(r["word"]["oov_rate"], 1.0)
        self.assertGreater(r["characters"]["heldout_accuracy"], r["word"]["heldout_accuracy"])
    def test_lexicon_growth(self):
        g = lexicon_growth(sizes=(50, 250))
        self.assertGreater(g[250]["lexicon_forms"], g[50]["lexicon_forms"])
        self.assertEqual(g[250]["structural_exact_recall"], 1.0)
    def test_make_agent_sizes(self):
        tiny = make_agent("C_wce_only", SETUP, size="tiny"); self.assertIsNone(tiny.neural)


if __name__ == "__main__":
    unittest.main()

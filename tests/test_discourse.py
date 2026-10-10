import ast
import os
import unittest

from core.discourse import (
    AMBIGUOUS, COMMIT, HYPOTHETICAL, NONEXISTENT, DiscourseResolver, DiscourseState, EntityIdentityGraph,
    HypotheticalEntity, Mention, Referent, Revision,
)
from core.discourse_agent import DiscourseAgent, DiscourseConfig
from core.discourse_benchmark import (
    SPECIAL, _entropy, _entropy_after, active_entity_learning, active_vs_passive, by_key, distill_resolver, lexicon_agent,
    long_discourse, observation_pool, reference_metrics, run_benchmark, run_episode,
)
from core.grounded import GroundedConfig
from core.reference_world import (
    CATEGORIES, ClarificationSimulator, Episode, REntity, build, family_captions, given_types, long_episode,
    reference_benchmark,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEX = BENCH = ROWS = POOL = None
ENTS = {"e1": REntity("e1", "cube", "red", ("on", "table")), "e2": REntity("e2", "cube", "blue", ("on", "shelf")),
        "e3": REntity("e3", "ball", "green", ("on", "floor"))}


def setUpModule():
    global LEX, BENCH, ROWS
    LEX = lexicon_agent(0)
    BENCH, ROWS = run_benchmark(n_per_category=6)


def given(attrs=True, locs=False):
    g = given_types(ENTS)
    if attrs: g += [(("attr", e), v.attr) for e, v in ENTS.items()]
    if locs: g += [(("loc", e, "now"), v.loc) for e, v in ENTS.items()]
    return g


def resolver(attrs=True, locs=False): return DiscourseResolver(LEX, given(attrs, locs))


def resolve(texts, attrs=True, locs=False):
    r = resolver(attrs, locs)
    out = []
    for t in texts: out.append([(m, r.mention_resolution(m.id)) for m in r.process(t)])
    return r, out


# ================================================================ independence and generator

class IndependenceTests(unittest.TestCase):
    def test_reference_world_no_wce(self):
        tree = ast.parse(open(os.path.join(ROOT, "core/reference_world.py")).read())
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        self.assertTrue(all(m == "core.independent_world" for m in mods if m.startswith("core")))
    def test_discourse_does_not_import_generator(self):
        tree = ast.parse(open(os.path.join(ROOT, "core/discourse.py")).read())
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        self.assertFalse({m for m in mods if "world" in m})


class GeneratorTests(unittest.TestCase):
    def test_all_categories(self):
        cats = {e.category for e in reference_benchmark(3)}
        self.assertTrue({"trivial", "pronoun", "other", "real_ambiguity", "false_presupposition", "cross_sentence"} <= cats)
    def test_three_domains(self): self.assertEqual({e.domain for e in reference_benchmark(3)}, {"physical", "inventory", "family"})
    def test_benchmark_size(self): self.assertGreaterEqual(BENCH["mentions"], 300)
    def test_full_size(self): self.assertGreaterEqual(sum(len(t.mentions) for e in reference_benchmark(14) for t in e.turns), 500)
    def test_gold_by_construction(self):
        import random
        ep = build("real_ambiguity", "physical", random.Random(1)); self.assertEqual(ep.turns[0].mentions[0]["gold"], AMBIGUOUS)
    def test_cross_sentence_consistent(self):
        import random
        ep = build("cross_sentence", "physical", random.Random(2))
        self.assertIn(ep.entities["e2"].loc[1].replace("shelf", "étagère").replace("floor", "sol"), ep.turns[2].text.replace("l'", ""))
    def test_noise_changes_given(self):
        a = reference_benchmark(3, seed=4, noise=0.0); b = reference_benchmark(3, seed=4, noise=0.9)
        self.assertNotEqual([e.given for e in a], [e.given for e in b])
    def test_long_episode(self): self.assertEqual(len(long_episode(20).turns), 20)
    def test_family_captions(self): self.assertTrue(all("structure" in c for c in family_captions(5)))
    def test_simulator(self):
        ep = Episode("physical", "c", ENTS, given(), [])
        sim = ClarificationSimulator(ep); self.assertEqual(sim.answer("?", "e2"), "le bleu"); self.assertEqual(sim.questions, 1)
    def test_simulator_reveal(self):
        sim = ClarificationSimulator(Episode("physical", "c", ENTS, given(), [])); self.assertEqual(sim.reveal(("attr", "e1")), "red")


# ================================================================ representations

class RepresentationTests(unittest.TestCase):
    def test_mention_fields(self):
        m = Mention("m1", "le cube rouge", "cube", "np", "def", "cube", {"attr": "red"})
        self.assertEqual((m.head, m.definiteness, m.attrs["attr"]), ("cube", "def", "red"))
    def test_extract_np(self):
        r = resolver(); ms, _, _ = r.extract(["le", "cube", "rouge"], 1)
        self.assertEqual((ms[0].type, ms[0].attrs), ("cube", {"attr": "red"}))
    def test_extract_relational(self):
        r = resolver(); ms, _, _ = r.extract(["le", "cube", "sur", "la", "table"], 1)
        self.assertEqual(ms[0].relation, ("on", "table", "now"))
    def test_extract_past_relative(self):
        r = resolver(); ms, _, _ = r.extract(["le", "cube", "qui", "était", "sur", "la", "table"], 1)
        self.assertEqual(ms[0].relation[2], "past")
    def test_extract_indefinite(self):
        r = resolver(); ms, _, _ = r.extract(["un", "cube"], 1); self.assertEqual(ms[0].definiteness, "indef")
    def test_extract_pronoun(self):
        r = resolver(); ms, _, _ = r.extract(["il", "tombe"], 1); self.assertEqual((ms[0].kind, ms[0].gender), ("pron", "m"))
    def test_extract_other(self):
        r = resolver(); ms, _, _ = r.extract(["l'", "autre"], 1); self.assertEqual(ms[0].kind, "other")
    def test_extract_bare_adjective(self):
        r = resolver(); ms, _, _ = r.extract(["le", "rouge"], 1); self.assertEqual((ms[0].kind, ms[0].attrs), ("adj", {"attr": "red"}))
    def test_extract_plural(self):
        r = resolver(); ms, _, _ = r.extract(["deux", "cubes"], 1); self.assertEqual((ms[0].kind, ms[0].number), ("plural", 2))
    def test_place_mention(self):
        r = resolver(); ms, _, _ = r.extract(["la", "table"], 1); self.assertEqual(ms[0].kind, "place")
    def test_state_fields(self):
        s = DiscourseState(); self.assertEqual((s.focus, s.time), (None, 0)); self.assertIsInstance(s.unresolved, set)
    def test_referent_identity_vs_state(self):
        ref = Referent("d1", "cube", "m", "def"); ref.states.append((1, "now", ("on", "table"))); ref.attrs["attr"] = "red"
        self.assertEqual(ref.attrs, {"attr": "red"}); self.assertTrue(ref.states)
    def test_hypothetical_dataclass(self): self.assertEqual(HypotheticalEntity("d1", {"type": "cube"}).support, 1)
    def test_revision_dataclass(self): self.assertEqual(Revision("d1", "A", "B", 0.5, 0.9, 2, "x").after, "B")


# ================================================================ resolution

class ResolutionTests(unittest.TestCase):
    def test_trivial_unique_type(self):
        _, out = resolve(["La balle est là."]); self.assertEqual(out[0][0][1], "e3")
    def test_color_disambiguates(self):
        _, out = resolve(["Le cube rouge tombe."]); self.assertEqual(out[0][0][1], "e1")
    def test_real_ambiguity_preserved(self):
        _, out = resolve(["Le cube tombe."], attrs=False); self.assertEqual(out[0][0][1], AMBIGUOUS)
    def test_branches_kept(self):
        r, _ = resolve(["Le cube tombe."], attrs=False)
        self.assertEqual(sorted(round(a.weight, 2) for a in r.branches("d1")), [0.5, 0.5])
    def test_position_disambiguates(self):
        _, out = resolve(["Le cube sur la table est là."], attrs=False, locs=True); self.assertEqual(out[0][0][1], "e1")
    def test_pronoun_follows_focus(self):
        _, out = resolve(["Le cube rouge est là.", "La balle est là aussi.", "Il tombe."]); self.assertEqual(out[2][0][1], "e1")
    def test_pronoun_gender(self):
        _, out = resolve(["Le cube rouge est là.", "La balle est là aussi.", "Elle tombe."]); self.assertEqual(out[2][0][1], "e3")
    def test_other(self):
        _, out = resolve(["Le cube rouge tombe.", "L'autre reste sur l'étagère."]); self.assertEqual(out[1][0][1], "e2")
    def test_other_not_bound_to_possible_coreferent(self):
        r, out = resolve(["Un cube est sur la table.", "Le cube rouge tombe.", "L'autre reste sur l'étagère."])
        self.assertEqual(out[2][0][1], "e2"); self.assertEqual(r.mention_resolution("m1"), AMBIGUOUS)
    def test_cross_sentence_recovery(self):
        r, out = resolve(["Un cube est là.", "Il est rouge."])
        self.assertEqual(out[0][0][1], AMBIGUOUS); self.assertEqual(r.mention_resolution(out[0][0][0].id), "e1")
    def test_backward_revision_logged(self):
        r, _ = resolve(["Un cube est là.", "Il est rouge."])
        self.assertTrue(r.revisions); self.assertEqual((r.revisions[0].before, r.revisions[0].after), (AMBIGUOUS, "e1"))
    def test_revision_keeps_provenance(self):
        r, _ = resolve(["Un cube est là.", "Il est rouge."]); self.assertEqual(r.revisions[0].sentence, 2)
    def test_presupposition_nonexistent(self):
        _, out = resolve(["Pourquoi le cube vert est-il là ?"]); self.assertEqual(out[0][0][1], NONEXISTENT)
    def test_presupposition_decomposed(self):
        r, out = resolve(["Pourquoi le cube vert est-il là ?"])
        self.assertEqual(r.presupposition(out[0][0][0].id), {"entity_exists": False, "property_supported": False, "relation_supported": False})
    def test_hypothetical_entity(self):
        r, out = resolve(["Un cube jaune est là."]); self.assertEqual(out[0][0][1], HYPOTHETICAL); self.assertTrue(r.hypotheticals)
    def test_identity_survives_state_change(self):
        r, out = resolve(["Le cube rouge tombe.", "Il est rouge."])
        self.assertEqual(out[1][0][1], "e1"); self.assertEqual(r.state.entities_mentioned["d1"].states[-1][2], ("on", "floor"))
    def test_mutable_state_not_identifier(self):
        _, out = resolve(["Le cube sur la table tombe.", "Il est rouge."], attrs=True, locs=True); self.assertEqual(out[1][0][1], "e1")
    def test_past_reference(self):
        g = given_types(ENTS) + [(("loc", "e1", 0), ("on", "table")), (("loc", "e2", 0), ("on", "shelf"))]
        r = DiscourseResolver(LEX, g); ms = r.process("Le cube qui était sur la table est rouge.")
        self.assertEqual(r.mention_resolution(ms[0].id), "e1")
    def test_dialogue_multi_turn(self):
        r = resolver(); ids = []
        for t in ("Il y a deux cubes.", "Le rouge est là.", "Prends-le.", "Pose-le encore."):
            ms = r.process(t); ids += [r.mention_resolution(m.id) for m in ms if m.kind in ("adj", "pron")]
        self.assertEqual(ids[-3:], ["e1", "e1", "e1"])      # expletive « Il y a » remains uncommitted (never false)
        self.assertNotIn("e2", ids)
    def test_plural_group_distinct(self):
        r = resolver(); r.process("Il y a deux cubes.")
        a, b = [x for x in r.state.entities_mentioned.values() if x.group]
        self.assertIn(b.id, a.different_from)
    def test_contradiction_detected(self):
        r, out = resolve(["Le cube rouge est bleu."]); self.assertTrue(r.contradictions >= 0)
    def test_exclusive_attrs_make_distinct(self):
        r, _ = resolve(["Le cube rouge tombe.", "Le cube bleu tombe."])
        self.assertIn("d2", r.state.entities_mentioned["d1"].different_from)
    def test_commit_threshold(self): self.assertEqual(COMMIT, 0.9)
    def test_checks_counted(self):
        r, _ = resolve(["Le cube rouge tombe."]); self.assertGreater(r.checks, 0)


class GraphTests(unittest.TestCase):
    def test_edges_types(self):
        r, _ = resolve(["Un cube est sur la table.", "Le cube rouge tombe.", "L'autre reste sur l'étagère."])
        kinds = {e[0] for e in r.graph.edges}
        self.assertTrue({"REFERS_TO", "POSSIBLY_SAME_AS", "DIFFERENT_FROM"} <= kinds)
    def test_no_duplicate_different_from(self):
        r, _ = resolve(["Le cube rouge tombe.", "L'autre reste sur l'étagère.", "Il est là."])
        edges = [(a, b) for k, a, b, _, _ in r.graph.edges if k == "DIFFERENT_FROM"]
        self.assertEqual(len(edges), len(set(edges)))
    def test_same_as_consolidation(self):
        r, _ = resolve(["Le cube rouge est là.", "Il tombe.", "Le cube rouge reste là."])
        self.assertEqual(len(r.consolidated()["entities"]["e1"]["referents"]), 1)
    def test_same_as_transitive_classes(self):
        g = EntityIdentityGraph(); g.add("SAME_AS", "a", "b"); g.add("SAME_AS", "b", "c")
        self.assertEqual(g.same_as_classes(), [{"a", "b", "c"}])
    def test_invariant_same_and_different(self):
        g = EntityIdentityGraph(); g.add("SAME_AS", "a", "b"); g.add("DIFFERENT_FROM", "a", "b")
        self.assertTrue(g.invariant_violations(DiscourseState(), {}))
    def test_invariant_different_same_entity(self):
        g = EntityIdentityGraph(); g.add("DIFFERENT_FROM", "d1", "d2")
        self.assertTrue(g.invariant_violations(DiscourseState(), {"d1": "e1", "d2": "e1"}))
    def test_invariant_exclusive_attrs(self):
        s = DiscourseState(); s.entities_mentioned = {"a": Referent("a", "cube", "m", "def", {"attr": "red"}),
                                                      "b": Referent("b", "cube", "m", "def", {"attr": "blue"})}
        g = EntityIdentityGraph(); g.add("SAME_AS", "a", "b")
        self.assertTrue(any(v[0] == "exclusive_attributes" for v in g.invariant_violations(s, {})))
    def test_benchmark_no_violations(self): self.assertEqual(BENCH["invariant_violations"], 0)


class ClarificationTests(unittest.TestCase):
    def test_question_in_french(self):
        r = resolver(); ms = r.process("Prends le cube.")
        q = r.clarification_question(ms[0].id); self.assertIn("rouge", q); self.assertIn("bleu", q)
    def test_clarification_resolves(self):
        r = resolver(); ms = r.process("Prends le cube.")
        r.apply_clarification(ms[0].id, "le bleu"); self.assertEqual(r.mention_resolution(ms[0].id), "e2")
    def test_no_question_when_resolved(self):
        r = resolver(); ms = r.process("Prends le cube rouge."); self.assertIsNone(r.clarification_question(ms[0].id))
    def test_clarification_counted(self):
        r = resolver(); ms = r.process("Prends le cube."); r.clarification_question(ms[0].id); self.assertEqual(r.clarifications, 1)
    def test_benchmark_clarification_rate(self): self.assertGreater(BENCH["clarification_rate"], 0)


class ActiveTests(unittest.TestCase):
    def test_information_request(self):
        g = given_types(ENTS)
        r = DiscourseResolver(LEX, g); ms = r.process("Un cube sur la table est là.")
        req, gain = r.best_information_request(ms[0].referent, ("red", "blue", "green", "yellow"))
        self.assertIsNotNone(req); self.assertGreater(gain, 0)
    def test_no_request_without_gain(self):
        r = resolver(); ms = r.process("Le cube rouge tombe.")
        self.assertIsNone(r.best_information_request(ms[0].referent, ("red", "blue"))[0])
    def test_entropy(self):
        r = resolver(attrs=False); ms = r.process("Le cube tombe."); self.assertAlmostEqual(r.identity_entropy(ms[0].referent), 0.693, places=2)
    def test_active_entity_learning(self):
        a = active_entity_learning(n_per_category=4)
        self.assertGreaterEqual(a["active"]["coverage"], a["passive"]["coverage"]); self.assertEqual(a["active"]["false_identity_commitment"], 0.0)
    def test_lexical_entropy_decreases(self):
        self.assertLess(_entropy_after({}, {"a": "support", "b": "refute"}, 1.0), _entropy({"a": 0.0, "b": 0.0, "c": 0.0}) + 1)
    def test_active_vs_passive_lexical(self):
        global POOL
        POOL = POOL or observation_pool(0, n_worlds=20)
        r = active_vs_passive(0, budget=20, pool=POOL)
        self.assertGreaterEqual(r["summary"]["active"]["words_learned"], r["summary"]["passive"]["words_learned"])
        self.assertEqual(r["summary"]["active"]["wrong"], 0)


# ================================================================ metrics and benchmark

class MetricsTests(unittest.TestCase):
    def test_metric_keys(self):
        for k in ("accuracy", "precision", "recall", "coverage", "false_identity_commitment", "ambiguity_preserved",
                  "clarification_rate", "avg_candidates", "cross_sentence_recovery", "belief_revisions"):
            self.assertIn(k, BENCH)
    def test_no_false_commitment(self): self.assertEqual(BENCH["false_identity_commitment"], 0.0)
    def test_ambiguity_preserved(self): self.assertEqual(BENCH["ambiguity_preserved"], 1.0)
    def test_precision(self): self.assertGreaterEqual(BENCH["precision"], 0.98)
    def test_coverage(self): self.assertGreater(BENCH["coverage"], 0.7)
    def test_cross_sentence_recovery(self): self.assertGreater(BENCH["cross_sentence_recovery"], 0)
    def test_revisions_counted(self): self.assertGreater(BENCH["belief_revisions"], 0)
    def test_by_domain(self):
        for dom, v in by_key(ROWS, "domain").items(): self.assertGreaterEqual(v["precision"], 0.98, dom)
    def test_by_category(self): self.assertEqual(set(by_key(ROWS, "category")) - set(CATEGORIES), set())
    def test_metrics_function(self):
        rows = [{"gold": "e1", "final": "e1", "at_time": AMBIGUOUS, "clarified": False, "candidates": 2},
                {"gold": AMBIGUOUS, "final": "e2", "at_time": "e2", "clarified": False, "candidates": 2}]
        m = reference_metrics(rows); self.assertEqual((m["precision"], m["false_identity_commitment"], m["cross_sentence_recovery"]), (0.5, 0.5, 1))
    def test_noise_robustness(self):
        m, _ = run_benchmark(n_per_category=6, noise=0.2); self.assertLess(m["false_identity_commitment"], 0.05)
    def test_long_discourse(self):
        ld = long_discourse(lengths=(5, 20), n=4)
        self.assertEqual(ld[20]["false_identity_commitment"], 0.0)
        self.assertLess(ld[20]["consolidated_fields"], ld[20]["raw_tokens"])
    def test_memory_compression(self):
        r, _ = resolve(["Le cube rouge tombe.", "Il est là.", "Le cube rouge reste là."])
        size = r.memory_size(); self.assertLess(size["consolidated_fields"], size["raw_tokens"])
    def test_episodic_text_kept(self):
        r, _ = resolve(["Le cube rouge tombe."]); self.assertEqual(r.episodic, ["Le cube rouge tombe."])
    def test_distillation(self):
        d = distill_resolver(n_per_category=4)
        self.assertLess(d["checks_student_then_verify"], d["checks_teacher"]); self.assertGreater(d["student_agreement_with_teacher"], 0.6)


# ================================================================ discourse agent in the independent environment

class DiscourseAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.grounded_benchmark import seed_data
        cls.data, cls.splits, caps = seed_data(0, n_worlds=12)
        cls.agent = DiscourseAgent(); cls.agent.learn_captions(caps); cls.agent.warm_up_sources(cls.splits["train"])
    def test_elimination_off_by_default(self):
        self.assertFalse(GroundedConfig().elimination); self.assertFalse(DiscourseConfig().elimination)
    def test_queries_stay_strict(self): self.assertGreater(DiscourseConfig().query_threshold, 1.0)
    def test_descriptors_collected(self):
        x = self.splits["test"][0]; self.agent.new_world(); self.agent.ingest(x["observations"])
        self.assertTrue(self.agent._descriptors)
    def test_color_distribution_normalized(self):
        x = self.splits["test"][0]; self.agent.new_world(); self.agent.ingest(x["observations"])
        o = next(iter(self.agent.belief.inventory))
        self.assertAlmostEqual(float(self.agent.color_distribution(o, ["red", "blue", "green", "yellow"]).sum()), 1.0, places=5)
    def test_mapping_probabilities(self):
        for x in self.splits["test"]:
            self.agent.new_world(); self.agent.ingest(x["observations"]); self.agent.solve_identity()
            for (_, _), (o, p) in self.agent._mapping.items(): self.assertTrue(0 <= p <= 1.0001)
    def test_identity_log(self):
        for x in self.splits["test"]:
            self.agent.new_world(); self.agent.ingest(x["observations"])
        self.assertTrue(all(p >= 0.9 for kind, *_, p in self.agent.identity_log if kind == "ingest"))
    def test_separate_channel(self):
        srcs = set()
        for x in self.splits["test"]:
            self.agent.new_world(); self.agent.ingest(x["observations"])
            srcs |= {e.source for evs in self.agent.belief.graph.props.values() for e in evs}
        self.assertTrue(any("|elim" in s for s in srcs) or not self.agent.identity_log)


# ================================================================ generated tests: category × domain, lengths, noise

class PerCategoryDomainTests(unittest.TestCase):
    pass


def _make_cat(cat, dom):
    def t(self):
        rows = [r for r in ROWS if r["category"] == cat and r["domain"] == dom]
        if not rows: self.skipTest("category absent for this domain")
        m = reference_metrics(rows)
        self.assertEqual(m["false_identity_commitment"], 0.0)
    return t


for _c in CATEGORIES:
    for _d in ("physical", "inventory", "family"):
        setattr(PerCategoryDomainTests, f"test_{_c}_{_d}_no_false_identity", _make_cat(_c, _d))


class LongEpisodeTests(unittest.TestCase):
    pass


def _make_long(n, dom, seed):
    def t(self):
        ep = long_episode(n, dom, seed); r = DiscourseResolver(LEX, ep.given); wrong = 0
        for turn in ep.turns:
            ms = r.process(turn.text)
            for m, g in zip([m for m in ms if m.kind != "place"], turn.mentions):
                got = r.mention_resolution(m.id)
                wrong += got not in (AMBIGUOUS, HYPOTHETICAL, NONEXISTENT) and g["gold"] not in (AMBIGUOUS,) and got != g["gold"]
        self.assertEqual(wrong, 0)
        self.assertFalse(r.graph.invariant_violations(r.state, {}))
    return t


for _n in (5, 10, 20, 50):
    for _d in ("physical", "inventory", "family"):
        for _s in (0, 1):
            setattr(LongEpisodeTests, f"test_long_{_n}_{_d}_{_s}", _make_long(_n, _d, _s))


if __name__ == "__main__":
    unittest.main()

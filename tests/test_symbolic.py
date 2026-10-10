import copy
import unittest

import torch

from core.branching import IncrementalSnapshot
from core.invariants import Invariant
from core.language_bridge import (
    LanguageBridge, default_corpus, make_world, observed_transition, parse,
)
from core.model_state import AdaptWeights, GrowModule, ModelState
from core.symbolic import (
    ALPHABET, FAMILIES, L, PRIMITIVES, SEQ_INPUT, V, NeuralSequenceTransformation, SymbolTableRefinement,
    adapt_expert_fast, ambiguity_rate, components, compose, decode, encode, induce_refinement, one_hot,
    primitive, random_sequences, sample_family, search_programs, sequence_accuracy, train_expert,
)
from core.symbolic_lifelong import (
    FEATURES, LearnedContextEncoder, Segment, SymbolicLifelongSystem, Thresholds, change_point_scores,
    cluster_purity, evaluate_family, few_shot_curve, novelty_scores, pair_features, pairs_to_disambiguate,
    routing_scores, run_static, run_stream, small_segments, stream, summarize, token_scores, zero_shot_composition,
)
from core.transformations import TransformationDomain

FULL = None
SMALL = None


def setUpModule():
    global FULL, SMALL
    FULL = run_stream()
    SMALL = run_stream(small_segments(), mode="none")


def seqs(*texts): return torch.stack([encode(t) for t in texts])


# ================================================================ representation

class RepresentationTests(unittest.TestCase):
    def test_alphabet(self): self.assertEqual(ALPHABET, "abcdefgh")
    def test_encode_decode(self): self.assertEqual(decode(encode("abcde")), "abcde")
    def test_one_hot_shape(self): self.assertEqual(one_hot(seqs("abcde")).shape, (1, SEQ_INPUT))
    def test_input_is_only_sequence(self): self.assertEqual(SEQ_INPUT, L * V)
    def test_same_input_distribution(self):
        a, _ = sample_family(FAMILIES["reverse"], 8, seed=3); b, _ = sample_family(FAMILIES["perm_A"], 8, seed=3)
        self.assertTrue(torch.equal(a, b))
    def test_twenty_plus_families(self): self.assertGreaterEqual(len(FAMILIES), 20)


class PrimitiveTests(unittest.TestCase):
    def ap(self, name, text): return decode(PRIMITIVES[name].apply(seqs(text))[0])
    def test_reverse(self): self.assertEqual(self.ap("reverse", "abcde"), "edcba")
    def test_rotl(self): self.assertEqual(self.ap("rotl1", "abcde"), "bcdea")
    def test_rotr(self): self.assertEqual(self.ap("rotr1", "abcde"), "eabcd")
    def test_sub(self): self.assertEqual(self.ap("sub1", "abcdh"), "bcdea")
    def test_swap(self): self.assertEqual(self.ap("swap", "abcde"), "badce")
    def test_sort(self): self.assertEqual(self.ap("sort", "dbeac"), "abcde")
    def test_identity(self): self.assertEqual(self.ap("identity", "abcde"), "abcde")
    def test_domain(self): self.assertIs(PRIMITIVES["reverse"].domain, TransformationDomain.SEQUENCE)
    def test_permutation_invariant(self):
        self.assertIn(Invariant.SYMBOL_MULTISET, PRIMITIVES["reverse"].preserves.invariants)
        self.assertNotIn(Invariant.SYMBOL_MULTISET, PRIMITIVES["sub1"].preserves.invariants)
    def test_inverse(self):
        x = seqs("abcde", "hgfea")
        for p in ("rotl1", "sub3", "reverse", "swap"):
            self.assertTrue(torch.equal(PRIMITIVES[p].get_inverse().apply(PRIMITIVES[p].apply(x)), x))
    def test_sort_not_invertible(self): self.assertIsNone(PRIMITIVES["sort"].get_inverse())
    def test_compose_order(self):
        spec = compose(PRIMITIVES["reverse"], PRIMITIVES["rotl1"])
        self.assertEqual(spec.name, "reverse>rotl1"); self.assertEqual(decode(spec.apply(seqs("abcde"))[0]), "dcbae")
    def test_components(self): self.assertEqual(len(components(compose(PRIMITIVES["reverse"], PRIMITIVES["swap"]))), 2)


class ProgramSearchTests(unittest.TestCase):
    def test_finds_primitive(self):
        x, y = sample_family(FAMILIES["rotl2"], 3); self.assertEqual(search_programs(x, y, list(PRIMITIVES.values())).program.name, "rotl2")
    def test_finds_composition(self):
        x, y = sample_family(FAMILIES["reverse>rotl1"], 3)
        r = search_programs(x, y, list(PRIMITIVES.values())); self.assertTrue(r.found)
        self.assertTrue(torch.equal(r.program.apply(x), y))
    def test_neural_family_not_expressible(self):
        x, y = sample_family(FAMILIES["perm_A"], 4); self.assertFalse(search_programs(x, y, list(PRIMITIVES.values())).found)
    def test_invariant_pruning(self):
        x, y = sample_family(FAMILIES["sub1"], 3)
        r = search_programs(x, y, list(PRIMITIVES.values()), first=False)
        self.assertLess(r.explored, len(PRIMITIVES) + len(PRIMITIVES) ** 2)
    def test_all_consistent_listed(self):
        x, y = sample_family(FAMILIES["reverse"], 3)
        self.assertIn("reverse", search_programs(x, y, list(PRIMITIVES.values()), first=False).consistent)
    def test_ambiguity(self):
        a = ambiguity_rate(FAMILIES); self.assertEqual(a["input_only"], 1.0); self.assertLess(a["single_pair"], 0.2)
    def test_pairs_to_disambiguate(self): self.assertGreaterEqual(pairs_to_disambiguate(trials=60), 1.0)


class NeuralCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = ModelState(SEQ_INPUT, max_region_fraction=1.0, max_capacity=100000)
        x, y = sample_family(FAMILIES["map_A"], 512)
        cls.spec = train_expert(cls.state, "map", x, y)
    def test_expert_learns(self):
        x, y = sample_family(FAMILIES["map_A"], 256, seed=9); self.assertGreaterEqual(sequence_accuracy(self.spec.apply(x), y), 0.95)
    def test_expert_is_transformation_spec(self): self.assertIs(self.spec.domain, TransformationDomain.SEQUENCE)
    def test_expert_traced(self):
        names = [t.name for t in self.state.transform_graph.current.transformations]
        self.assertIn("add_module", names); self.assertIn("adapt_weights", names)
    def test_expert_composable(self):
        x, y = sample_family(FAMILIES["map_A>rotl1"], 128, seed=9)
        self.assertGreaterEqual(sequence_accuracy(compose(self.spec, PRIMITIVES["rotl1"]).apply(x), y), 0.95)
    def test_resolution_by_name(self):
        state = copy.copy(self.state)
        spec = NeuralSequenceTransformation(state=self.state, module_name="map"); self.assertIs(spec.resolved, self.state.modules["map"])
    def test_fast_adaptation_keeps_slow(self):
        ref = self.state.modules["map"].slow["w2"]
        x, y = sample_family(FAMILIES["map_A2"], 16)
        adapt_expert_fast(self.state, "map", x, y, steps=20); self.assertIs(self.state.modules["map"].slow["w2"], ref)
    def test_refinement_few_shot(self):
        x, y = sample_family(FAMILIES["map_A2"], 8, seed=2); r = induce_refinement(x, y, self.spec)
        self.assertIsInstance(r, SymbolTableRefinement)
        xt, yt = sample_family(FAMILIES["map_A2"], 128, seed=9); self.assertGreaterEqual(sequence_accuracy(r.apply(xt), yt), 0.95)
    def test_refinement_needs_related_base(self):
        x, y = sample_family(FAMILIES["perm_B"], 8); self.assertIsNone(induce_refinement(x, y, self.spec))
    def test_refinement_needs_pairs(self):
        x, y = sample_family(FAMILIES["map_A2"], 2); self.assertIsNone(induce_refinement(x, y, self.spec))
    def test_sequence_loss_positions(self):
        s = ModelState(SEQ_INPUT, max_region_fraction=1.0, max_capacity=100000)
        x, y = sample_family(FAMILIES["identity"], 64)
        spec = train_expert(s, "id", x, y, steps=50); self.assertEqual(spec.apply(x).shape, (64, L))


# ================================================================ learned encoder

class ContextEncoderTests(unittest.TestCase):
    def test_features_relational(self):
        x = seqs("abcde"); f = pair_features(x, PRIMITIVES["reverse"].apply(x)); self.assertEqual(f.shape, (1, FEATURES))
    def test_features_distinguish_same_input(self):
        x = seqs("abcde")
        self.assertFalse(torch.equal(pair_features(x, PRIMITIVES["reverse"].apply(x)), pair_features(x, PRIMITIVES["rotl1"].apply(x))))
    def test_encoder_small(self): self.assertLess(LearnedContextEncoder().parameter_count, 10000)
    def test_encoder_learns_from_pseudo_labels(self):
        enc = LearnedContextEncoder(); feats, labels = [], []
        for i, f in enumerate(("reverse", "rotl1", "sub1")):
            x, y = sample_family(FAMILIES[f], 64, seed=i); feats.append(pair_features(x, y)); labels.append(torch.full((64,), i))
        enc.train(torch.cat(feats), torch.cat(labels))
        x, y = sample_family(FAMILIES["rotl1"], 32, seed=9)
        with torch.no_grad(): pred = enc.state.forward(pair_features(x, y))[:, :3].argmax(1)
        self.assertGreaterEqual(float((pred == 1).float().mean()), 0.9)
    def test_encoder_trained_during_stream(self): self.assertGreater(FULL.system.encoder.trainings, 5)
    def test_encoder_traced(self):
        names = [t.name for t in FULL.system.encoder.state.transform_graph.current.transformations]
        self.assertIn("context_learning", names)
    def test_embedding_stability(self): self.assertGreater(summarize(FULL)["embedding_stability"], 0.7)
    def test_prototypes_exist(self):
        self.assertTrue(all(c.prototype is not None for c in FULL.system.contexts.values()))


# ================================================================ continuous stream

class StreamTests(unittest.TestCase):
    def test_stream_no_labels_needed(self):
        exp = next(stream(small_segments())); self.assertIn("x", exp); self.assertIn("family", exp)
    def test_mode_none_ignores_family(self):
        s = SymbolicLifelongSystem(mode="none")
        x, y = sample_family(FAMILIES["reverse"], 4)
        for i in range(4): s.observe(x[i], y[i], family="WRONG_LABEL")
        self.assertEqual(s.contexts[s.current].skill.name, "reverse")
    def test_contexts_emerge(self): self.assertEqual(len(FULL.system.contexts), 21)
    def test_context_ids_internal(self): self.assertTrue(all(c.startswith("ctx_") for c in FULL.system.contexts))
    def test_cluster_purity(self): self.assertGreaterEqual(cluster_purity(FULL.system), 0.9)
    def test_final_accuracy(self): self.assertGreaterEqual(summarize(FULL)["final_accuracy"], 0.95)
    def test_no_forgetting(self): self.assertLessEqual(max(FULL.forgetting().values()), 0.05)
    def test_returns_to_known_contexts(self):
        self.assertGreaterEqual(sum(e["event"] == "known_context" for e in FULL.system.events), 4)
    def test_novelty_accuracy(self): self.assertGreaterEqual(novelty_scores(FULL), 0.9)
    def test_change_points(self):
        cp = change_point_scores(FULL); self.assertGreaterEqual(cp["f1"], 0.9); self.assertLessEqual(cp["mean_delay"], 3)
    def test_change_detection_disabled_with_labels(self):
        s = run_stream(small_segments()[:3], mode="labels"); self.assertEqual(s.system.changes, [])
    def test_boundaries_mode(self):
        r = run_stream(small_segments()[:4], mode="boundaries"); self.assertGreaterEqual(summarize(r)["final_accuracy"], 0.95)
    def test_primitives_three_examples(self):
        for c in FULL.system.contexts.values():
            if c.skill.kind == "primitive": self.assertLessEqual(c.skill.examples, 3)
    def test_neural_needs_more_examples(self):
        for c in FULL.system.contexts.values():
            if c.skill.kind == "neural": self.assertGreaterEqual(c.skill.examples, 128)
    def test_online_tracking(self): self.assertEqual(len(FULL.system.online), sum(s.length for s in FULL.segments))
    def test_episodic_experiences(self): self.assertEqual(len(FULL.system.episodic), len(FULL.system.contexts))
    def test_skills_in_strategy_library(self): self.assertEqual(len(FULL.system.library), len(FULL.system.skills))


class CompositionTests(unittest.TestCase):
    def test_zero_shot_compositions_in_stream(self): self.assertGreaterEqual(summarize(FULL)["zero_shot_compositions"], 5)
    def test_hybrid_uses_neural_expert(self):
        hybrids = [c.skill for c in FULL.system.contexts.values() if c.skill.kind == "hybrid"]
        self.assertTrue(hybrids); self.assertTrue(all("expert" in h.name for h in hybrids))
    def test_hybrid_not_trained(self):
        for c in FULL.system.contexts.values():
            if c.skill.kind in ("hybrid", "composition"): self.assertFalse(c.skill.trained)
    def test_zero_shot_experiment(self):
        z = zero_shot_composition()
        self.assertEqual(z["wce_composition"]["accuracy"], 1.0); self.assertFalse(z["wce_composition"]["trained"])
        self.assertLess(z["network_alone_3_examples"], 0.2)
    def test_composition_reuse(self): self.assertGreaterEqual(summarize(FULL)["composition_reuse"], 7)


class FewShotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.curve = few_shot_curve(shots=(4, 8, 128))
    def test_refinement_few_shot(self): self.assertGreaterEqual(self.curve[1]["wce_refinement"], 0.95)
    def test_new_expert_needs_many(self): self.assertLess(self.curve[0]["new_expert"], 0.2)
    def test_refinement_beats_fast_at_low_shots(self): self.assertGreater(self.curve[0]["wce_refinement"], self.curve[0]["fast_weights"])
    def test_stream_chose_refinement(self):
        kinds = {c.skill.kind for c in FULL.system.contexts.values()}; self.assertIn("refinement", kinds)


# ================================================================ hybrid routing / commands

class RoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.r = routing_scores(SMALL.system, sorted({s.family for s in SMALL.segments}), n=24)
    def test_structural_accurate(self): self.assertGreaterEqual(self.r["structural"], 0.9)
    def test_hybrid_accurate(self): self.assertGreaterEqual(self.r["hybrid"], 0.9)
    def test_hybrid_cheaper_than_structural(self): self.assertLess(self.r["hybrid_evaluations"], self.r["structural_evaluations"])
    def test_neural_measured(self): self.assertIsNotNone(self.r["neural"])
    def test_predict_with_support(self):
        x, y = sample_family(FAMILIES["reverse"], 3)
        xq, yq = sample_family(FAMILIES["reverse"], 16, seed=4)
        self.assertEqual(sequence_accuracy(SMALL.system.predict(xq, support=(x, y)), yq), 1.0)
    def test_predict_without_anything_uses_current(self):
        xq = random_sequences(4, torch.Generator().manual_seed(0))
        self.assertEqual(SMALL.system.predict(xq).shape, xq.shape)
    def test_neural_confidence_shortcut_threshold(self):
        s = copy.deepcopy(SMALL.system); s.th.routing_confidence = 0.0
        x, y = sample_family(FAMILIES["reverse"], 2); before = s.evaluations["identify"]; s.identify(x, y)
        self.assertEqual(s.evaluations["identify"], before)


class CommandTests(unittest.TestCase):
    def test_lexicon_learned(self): self.assertIn("REV", FULL.system.lexicon)
    def test_token_only_accuracy(self):
        scores = token_scores(FULL.system); self.assertTrue(scores)
        for t in scores.values(): self.assertEqual(t["token_only"], 1.0)
    def test_no_token_no_support_is_ambiguous(self):
        scores = token_scores(FULL.system)
        self.assertLess(sum(t["no_token_no_support"] for t in scores.values()) / len(scores), 0.5)
    def test_tokens_removed_later_in_segment(self):
        seg = [e for e in stream([Segment("reverse", 6, token_steps=3)])]
        self.assertEqual([e["token"] for e in seg], ["REV", "REV", "REV", None, None, None])


# ================================================================ capacity / memory / snapshots

class CapacityTests(unittest.TestCase):
    def test_experts_much_fewer_than_families(self): self.assertLessEqual(len(FULL.system.state.modules), 5)
    def test_tasks_per_expert(self): self.assertGreaterEqual(summarize(FULL)["tasks_per_expert"], 4)
    def test_sublinear_capacity(self):
        curve = FULL.system.capacity_curve
        per_family = curve[-1]["total_parameters"] / curve[-1]["contexts"]
        self.assertLess(per_family, 5224)   # less than one full expert per family
    def test_capacity_curve_monotone_contexts(self):
        c = [x["contexts"] for x in FULL.system.capacity_curve]; self.assertEqual(c, sorted(c))
    def test_structural_skills_add_no_parameters(self):
        curve = FULL.system.capacity_curve; self.assertEqual(curve[0]["total_parameters"], curve[12]["total_parameters"])
    def test_memory_usage_levels(self):
        m = summarize(FULL)["memory_usage"]
        for level in ("structural", "slow_parametric", "episodic", "current_state"): self.assertIn(level, m)
    def test_skill_memory_types(self):
        kinds = {c.skill.kind: c.skill.memory for c in FULL.system.contexts.values()}
        self.assertIn("structural", kinds["primitive"]); self.assertIn("slow_parametric", kinds["neural"])


class IncrementalSnapshotTests(unittest.TestCase):
    def state(self):
        s = ModelState(SEQ_INPUT, max_region_fraction=1.0, max_capacity=100000)
        x, y = sample_family(FAMILIES["identity"], 32); train_expert(s, "a", x, y, steps=10)
        return s, x, y
    def test_first_snapshot_owns_all(self):
        s, _, _ = self.state(); snap = IncrementalSnapshot("s0", s); self.assertEqual(snap.owned_bytes(), snap.full_bytes())
    def test_unchanged_child_owns_nothing(self):
        s, _, _ = self.state(); a = IncrementalSnapshot("a", s); b = IncrementalSnapshot("b", s, a); self.assertEqual(b.owned_bytes(), 0)
    def test_child_owns_only_modified(self):
        s, x, y = self.state(); a = IncrementalSnapshot("a", s); train_expert(s, "b", x, y, steps=5)
        b = IncrementalSnapshot("b", s, a); self.assertEqual(b.owned_bytes(), s.modules["b"].parameter_count * 4)
    def test_delta_description(self):
        s, x, y = self.state(); a = IncrementalSnapshot("a", s); train_expert(s, "b", x, y, steps=5, config="cfg")
        d = IncrementalSnapshot("b", s, a).delta(); self.assertEqual(d["added_modules"], ["b"]); self.assertEqual(d["routes"], {"cfg": "b"})
    def test_exact_restore(self):
        s, x, y = self.state(); snap = IncrementalSnapshot("a", s); ref = NeuralSequenceTransformation(state=s, module_name="a").apply(x)
        s.apply(GrowModule(module="a", units=4)); s.apply(AdaptWeights(modules=("a",), fast=False, output="a", positions=L,
                                                                       inputs=one_hot(x), targets=y, steps=20, local=False))
        snap.restore(s)
        self.assertTrue(torch.equal(NeuralSequenceTransformation(state=s, module_name="a").apply(x), ref))
        self.assertEqual(s.modules["a"].hidden, 64)
    def test_restore_traced(self):
        s, _, _ = self.state(); snap = IncrementalSnapshot("a", s); snap.restore(s)
        self.assertEqual(s.transform_graph.current.transformations[-1].name, "restore_incremental")
    def test_stream_snapshots_incremental(self):
        m = summarize(FULL); self.assertLess(m["snapshot_bytes_incremental"], m["snapshot_bytes_full"] / 3)


# ================================================================ ablations / sensitivity

class AblationTests(unittest.TestCase):
    def test_static_baseline_fails(self): self.assertLess(run_static(small_segments())["final_accuracy"], 0.3)
    def test_structural_only_misses_neural_families(self):
        r = run_stream(small_segments(), neural=False)
        self.assertLess(r.final_accuracy["perm_A"], 0.2); self.assertEqual(r.final_accuracy["reverse"], 1.0)
    def test_plasticity_only_misses_short_regimes(self):
        r = run_stream(small_segments()[:3], structural=False); self.assertLess(summarize(r)["final_accuracy"], 0.5)
    def test_autonomous_matches_labels(self):
        a = summarize(run_stream(small_segments(), mode="labels"))["final_accuracy"]
        self.assertGreaterEqual(summarize(SMALL)["final_accuracy"], a - 0.02)
    def test_thresholds_configurable(self):
        th = Thresholds(change_errors=3); self.assertEqual(SymbolicLifelongSystem(thresholds=th).th.change_errors, 3)


# ================================================================ language bridge

class LanguageBridgeTests(unittest.TestCase):
    def bridge(self):
        b = LanguageBridge()
        for sentence, heights, entity, primitive in default_corpus():
            b.observe_physics(sentence, *observed_transition(heights, entity, primitive))
        return b
    def test_parse(self):
        p = parse("la balle tombe", {"balle": None}); self.assertEqual((p.entity, p.verb), ("balle", "tombe"))
    def test_parse_sequence(self): self.assertEqual(parse("inverse abcde").sequence, "abcde")
    def test_verbs_learned_from_experience(self): self.assertEqual(self.bridge().interpret("la balle tombe").skill, "FALL")
    def test_unknown_synonym_not_guessed(self): self.assertFalse(self.bridge().interpret("la balle chute").known)
    def test_synonym_one_shot(self):
        b = self.bridge(); b.observe_physics("la balle chute", *observed_transition({"balle": 2.0}, "balle", "FALL"))
        self.assertEqual(b.interpret("la balle chute").skill, "FALL")
    def test_transfer_to_other_entity(self):
        b = self.bridge(); b.observe_physics("la balle s'élève", *observed_transition({"balle": 0.0}, "balle", "RISE"))
        w = make_world({"bloc": 2.0}); _, y = b.execute("le bloc s'élève", w); self.assertEqual(y, 3.0)
    def test_fall_executes_on_world(self):
        w = make_world({"bloc": 4.0}); _, y = self.bridge().execute("le bloc tombe", w); self.assertEqual(y, 0.0)
    def test_no_change_not_bound(self):
        b = LanguageBridge(); w = make_world({"balle": 0.0})
        self.assertIsNone(b.observe_physics("la balle dort", w, copy.deepcopy(w)))
    def test_unknown_verb_execute(self):
        interp, value = self.bridge().execute("le bloc roule", make_world({"bloc": 1.0}))
        self.assertFalse(interp.known); self.assertIsNone(value)
    def test_sequence_verbs(self):
        b = LanguageBridge(); b.observe_sequence("retourne abcde", "edcba")
        self.assertEqual(b.execute("retourne hgfab")[1], "bafgh")
    def test_ambiguous_sequence_not_bound(self):
        self.assertIsNone(LanguageBridge().observe_sequence("fais aaaaa", "aaaaa"))
    def test_lexicon_in_strategy_library(self):
        b = self.bridge(); self.assertGreaterEqual(len(b.library), 2)


if __name__ == "__main__":
    unittest.main()

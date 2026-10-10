import ast
import hashlib
import math
import os
import unittest

import torch

from core.entitlement import (
    ASKED, ASSERTED, QUALIFIED, WITHHELD, ClaimEntitlement, EntitlementPolicy, EpistemicProvenance, evidence_level, provenance,
)
from core.epistemic_benchmark import (
    AUDIT, CAUSES, attribute, audit, conversion, error_attribution, evaluate, knowability_alignment, lexical_examples,
    make_system, pareto, prepare, score,
)
from core.independent_agent import Answer, BeliefEntry
from core.independent_world import Observation
from core.knowledge_boundary import (
    CONFLICTED, KNOWABLE, L0, L1, L2, L3, UNKNOWN, Desc, EliminateCandidates, IdealAgent, KnowledgeBoundary, PublicText,
    Relabeler, boundary, eliminate, public_view, relabel_dataset, true_reliabilities,
)
from core.lexical_calibration import (
    FEATURES, LEX_BELIEF, LEX_HYPOTHESIS, LEX_STABLE, LEX_UNKNOWN, NULL, CalibratedLexicon, LexicalExample, LexicalLifecycle,
    LexicalPosterior, PolysemyPosterior, SimilarityByBehavior, attach, calibration_points, class_purity, compose, entropy,
    examples_from_agent, morphology_probability, precision_at, stem_lookup,
)
from core.lexical_experiments import (
    ANALOGY, MORPHOLOGICAL, STRUCTURAL, UNKNOWN as UW_UNKNOWN, composition_eval, semantic_classes, unknown_word_candidates,
)
from core.third_party_hard import SEALED_SHA256, dev_generated, dev_set, n_questions, sealed_hash, sealed_set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = SPLITS = CAPS = LABELS = POST = None


def setUpModule():
    global DATA, SPLITS, CAPS, LABELS, POST
    from core.grounded_benchmark import seed_data
    DATA, SPLITS, CAPS = seed_data(0, n_worlds=10)
    LABELS = relabel_dataset(SPLITS["test"], true_reliabilities())
    ex = [e for s in (100, 101) for e in lexical_examples(s, snapshots=True)[0] if e.gold is not None]
    POST = LexicalPosterior().fit(ex, epochs=200)


def imports(path):
    tree = ast.parse(open(os.path.join(ROOT, path)).read())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module: mods.add(n.module)
        if isinstance(n, ast.Import): mods |= {a.name for a in n.names}
    return mods


def ideal(votes=None, inv=None, rel=None):
    a = IdealAgent(rel or {"s1": 0.95, "s2": 0.95, "weak": 0.55})
    a.inventory = inv or {"o1": "cube", "o2": "cube", "o3": "ball"}
    for k, v in (votes or {}).items(): a.votes[k] = v
    a.events = {}
    return a


# ================================================================ independence and integrity

class IndependenceTests(unittest.TestCase):
    def test_ideal_agent_no_system_imports(self):
        mods = imports("core/knowledge_boundary.py")
        for bad in ("core.independent_agent", "core.grounded", "core.unified", "core.discourse", "core.unified_agent"):
            self.assertNotIn(bad, mods)
    def test_hard_bench_stdlib_only(self):
        self.assertEqual({m for m in imports("core/third_party_hard.py") if m != "__future__"}, {"hashlib", "json", "random"})
    def test_sealed_integrity(self): self.assertEqual(sealed_hash(), SEALED_SHA256)
    def test_sealed_size(self): self.assertGreaterEqual(n_questions(sealed_set()), 30)
    def test_dev_size(self): self.assertGreaterEqual(n_questions(dev_set() + dev_generated()), 80)
    def test_sealed_disjoint_ids(self):
        self.assertFalse({s["id"] for s in sealed_set()} & {s["id"] for s in dev_set() + dev_generated()})
    def test_hard_categories(self):
        cats = {s["catégorie"] for s in sealed_set()}
        for c in ("vocabulaire_non_vu", "expression_nouvelle", "ambiguïté_lexicale", "source_trompeuse", "élimination",
                  "connaissance_partielle", "présupposition", "temporalité", "chaîne_longue", "polysémie"):
            self.assertIn(c, cats)
    def test_generator_unchanged(self):
        from core.grounded_benchmark import seed_data
        h = hashlib.sha256()
        for s in (0, 3):
            data, _, _ = seed_data(s)
            for x in data:
                for it in x["items"]: h.update(f"{it.text}|{it.gold}|{it.intent}|{it.category}".encode())
                for o in x["observations"]: h.update(repr(o).encode())
        self.assertEqual(h.hexdigest(), "a35f58e8e2d2ebd361d5e841cbc2bb5838e855b787d17a777888fada37bc9b01")
    def test_meta_refs_added(self):
        cmds = [it for x in DATA for it in x["items"] if it.intent == "command" and it.category == "command"]
        self.assertTrue(all("refs" in it.meta and "ungated" in it.meta for it in cmds))


# ================================================================ public view and ideal agent

class PublicViewTests(unittest.TestCase):
    def setUp(self): self.x = SPLITS["test"][0]; self.inv, self.st, self.tx = public_view(self.x["world"], self.x["observations"])
    def test_inventory_from_types(self): self.assertTrue(self.inv)
    def test_texts_have_no_object_ids(self):
        for t in self.tx: self.assertNotIn(t.subject.type, self.x["world"].objects)
    def test_implied_excluded(self):
        n_implied = sum(o.implied for o in self.x["observations"])
        self.assertEqual(len(self.st) + len(self.tx) + len(self.inv) + n_implied, len(self.x["observations"]))
    def test_structured_keep_ids(self): self.assertTrue(all(len(o.key) >= 2 for o in self.st))
    def test_color_text_descriptor_without_color(self):
        for t in self.tx:
            if t.kind == "color": self.assertIsNone(t.subject.color)
    def test_hidden_mode_is_structured(self):
        _, st, tx = public_view(self.x["world"], self.x["observations"], hidden=True); self.assertEqual(tx, [])


class IdealAgentTests(unittest.TestCase):
    def test_posterior_single_strong(self):
        a = ideal({("color", "o1"): {"s1": ("red", L0, "c")}}); v, p, _ = a.posterior(("color", "o1"))
        self.assertEqual(v, "red"); self.assertGreater(p, 0.9)
    def test_posterior_weak_not_knowable(self):
        a = ideal({("color", "o1"): {"weak": ("red", L0, "c")}}); self.assertIsNone(a.known(("color", "o1")))
    def test_conflicted(self):
        a = ideal({("color", "o1"): {"s1": ("red", L0, "c"), "s2": ("blue", L0, "c")}})
        self.assertEqual(a.status(("color", "o1")), CONFLICTED)
    def test_unknown_status(self): self.assertEqual(ideal().status(("color", "o1")), UNKNOWN)
    def test_knowable_status(self):
        a = ideal({("color", "o1"): {"s1": ("red", L0, "c"), "s2": ("red", L0, "c")}}); self.assertEqual(a.status(("color", "o1")), KNOWABLE)
    def test_resolve_unique_type_L0(self): self.assertEqual(ideal().resolve(Desc("ball", "red"))[:2], ("o3", L0))
    def test_resolve_place(self): self.assertEqual(ideal().resolve(Desc("table", None, "table"))[:2], ("table", L0))
    def test_resolve_ambiguous_without_color(self): self.assertEqual(ideal().resolve(Desc("cube"))[0], None)
    def test_resolve_by_elimination_L1(self):
        a = ideal({("color", "o1"): {"s1": ("blue", L0, "c"), "s2": ("blue", L0, "c")}})
        o, lvl, rule = a.resolve(Desc("cube", "red")); self.assertEqual((o, lvl), ("o2", L1)); self.assertTrue(rule.conclusive)
    def test_resolve_match_L0(self):
        a = ideal({("color", "o1"): {"s1": ("red", L0, "c"), "s2": ("red", L0, "c")}, ("color", "o2"): {"s1": ("blue", L0, "c"), "s2": ("blue", L0, "c")}})
        self.assertEqual(a.resolve(Desc("cube", "red"))[:2], ("o1", L1))       # uniqueness established by exclusion of o2
    def test_resolve_unknown_others_fails(self): self.assertIsNone(ideal().resolve(Desc("cube", "red"))[0])
    def test_observe_text_descriptor_vote(self):
        a = IdealAgent({"s1": 0.95})
        a.observe({"o1": "ball"}, [], [PublicText("s1", 1, "loc", Desc("ball", "red"), ("on", Desc("table", None, "table")), "now")])
        self.assertIn("s1", a.votes[("color", "o1")]); self.assertEqual(a.votes[("loc", "o1", "now")]["s1"][0], ("on", "table"))
    def test_observe_unresolved_text_ignored(self):
        a = IdealAgent({"s1": 0.95})
        a.observe({"o1": "cube", "o2": "cube"}, [], [PublicText("s1", 1, "loc", Desc("cube", "red"), ("on", Desc("table", None, "table")), "now")])
        self.assertNotIn(("loc", "o1", "now"), a.votes); self.assertNotIn(("loc", "o2", "now"), a.votes)
    def test_height_derived_from_loc(self):
        a = ideal({("loc", "o3", "now"): {"s1": (("on", "table"), L0, "c"), "s2": (("on", "table"), L0, "c")}})
        v, lvl, st = a.value(("height", "o3", "now")); self.assertEqual(st, KNOWABLE); self.assertGreaterEqual(lvl, L1)
    def test_ref_status_unique(self):
        a = ideal({("color", "o3"): {"s1": ("red", L0, "c"), "s2": ("red", L0, "c")}}); self.assertEqual(a.ref_status("o3"), "ok")
    def test_ref_status_no_color(self): self.assertEqual(ideal().ref_status("o1", with_color=False), "ASK")


class EliminationTests(unittest.TestCase):
    def test_conclusive(self):
        r = eliminate(("cube", "red"), ["o1", "o2"], {"o1": "blue"}, {"o1": 0.97}); self.assertEqual(r.remaining, "o2")
    def test_level_L1(self): self.assertEqual(eliminate(("cube", "red"), ["o1", "o2"], {"o1": "blue"}, {"o1": 0.97}).level, L1)
    def test_confidence_product(self):
        r = eliminate(("cube", "red"), ["o1", "o2", "o3"], {"o1": "blue", "o2": "green"}, {"o1": 0.9, "o2": 0.8})
        self.assertAlmostEqual(r.confidence, 0.72)
    def test_premises_recorded(self):
        r = eliminate(("cube", "red"), ["o1", "o2"], {"o1": "blue"}, {"o1": 0.97}); self.assertEqual(r.premises[0][0], ("color", "o1"))
    def test_not_conclusive(self):
        r = eliminate(("cube", "red"), ["o1", "o2", "o3"], {"o1": "blue"}, {"o1": 0.97}); self.assertFalse(r.conclusive); self.assertEqual(r.level, L3)
    def test_direct_match_L0(self):
        r = eliminate(("cube", "red"), ["o1"], {"o1": "red"}, {"o1": 0.99}); self.assertEqual((r.remaining, r.level), ("o1", L0))
    def test_excluded_dict(self): self.assertEqual(eliminate(("c", "r"), ["a", "b"], {"a": "x"}, {}).excluded, {"a": "x"})
    def test_dataclass(self): self.assertIsInstance(eliminate(("c", "r"), ["a"], {}, {}), EliminateCandidates)


class RelabelTests(unittest.TestCase):
    def test_hidden_mode_reproduces_generator(self):
        lab = relabel_dataset(SPLITS["test"], true_reliabilities(), hidden=True)
        pairs = [(lab[id(it)][0], it.gold) for x in SPLITS["test"] for it in x["items"] if lab[id(it)][1]]
        self.assertGreater(sum(a == b for a, b in pairs) / len(pairs), 0.99)
    def test_coverage(self):
        n = sum(ok for _, ok in LABELS.values()); self.assertGreater(n / len(LABELS), 0.9)
    def test_public_changes_some(self):
        self.assertTrue(any(g != it.gold for x in SPLITS["test"] for it in x["items"] for g, ok in [LABELS[id(it)]] if ok))
    def test_invented_word_unchanged(self):
        for x in SPLITS["test"]:
            for it in x["items"]:
                if it.gold == "UNKNOWN_WORD": self.assertEqual(LABELS[id(it)][0], "UNKNOWN_WORD")
    def test_boundary_fields(self):
        x = SPLITS["test"][0]; rl = Relabeler(x["world"], x["observations"], true_reliabilities())
        key = next(k for k in x["world"].facts() if k[0] == "color")
        kb = boundary(key, x["world"].facts()[key], rl.ideal, None)
        self.assertIsInstance(kb, KnowledgeBoundary); self.assertEqual(kb.safe_to_assert, kb.ideal_value is not None)
    def test_knowable_requires_truth(self):
        x = SPLITS["test"][0]; rl = Relabeler(x["world"], x["observations"], true_reliabilities())
        for key in x["world"].facts():
            if key[0] == "color":
                v = rl.knowable(key); self.assertTrue(v is None or v == x["world"].facts()[key])


# ================================================================ right to assert

def prov(level=L0, p=0.99, idu=0.0, lin=0.0, div=2):
    return EpistemicProvenance("x", level, belief_probability=p, identity_uncertainty=idu, linguistic_uncertainty=lin, source_diversity=div)


def ans(out="on:table", intent="where"): return Answer("t", out, "ACCEPT", intent)


class EntitlementTests(unittest.TestCase):
    def test_ask(self): self.assertEqual(ClaimEntitlement().decide(ans("ASK"), prov()), ASKED)
    def test_abstain(self): self.assertEqual(ClaimEntitlement().decide(ans("UNKNOWN"), prov()), WITHHELD)
    def test_assert_L0(self): self.assertEqual(ClaimEntitlement().decide(ans(), prov()), ASSERTED)
    def test_assert_L1(self): self.assertEqual(ClaimEntitlement().decide(ans(), prov(L1)), ASSERTED)
    def test_L2_qualified(self): self.assertEqual(ClaimEntitlement().decide(ans(), prov(L2, 0.999)), QUALIFIED)
    def test_L2_allowed(self):
        self.assertEqual(ClaimEntitlement(EntitlementPolicy(allow_l2=True)).decide(ans(), prov(L2, 0.999)), ASSERTED)
    def test_low_probability_withheld(self): self.assertEqual(ClaimEntitlement().decide(ans(), prov(L0, 0.3)), WITHHELD)
    def test_medium_qualified(self): self.assertEqual(ClaimEntitlement().decide(ans(), prov(L0, 0.8)), QUALIFIED)
    def test_identity_uncertainty_asks(self): self.assertEqual(ClaimEntitlement().decide(ans(), prov(L0, 0.5, idu=0.5)), ASKED)
    def test_linguistic_uncertainty_qualifies(self): self.assertEqual(ClaimEntitlement().decide(ans(), prov(L0, 0.9, lin=0.2)), QUALIFIED)
    def test_linguistic_uncertainty_high_withholds(self): self.assertEqual(ClaimEntitlement().decide(ans(), prov(L0, 0.9, lin=0.5)), WITHHELD)
    def test_claim_probability(self): self.assertAlmostEqual(ClaimEntitlement().claim_probability(prov(L0, 0.9, 0.1, 0.1)), 0.9 * 0.9 * 0.9)
    def test_command_L0_asserted(self): self.assertEqual(ClaimEntitlement().decide(ans("o1:on:table", "command"), prov(L0, 1.0)), ASSERTED)
    def test_output_keeps_raw_abstention(self): self.assertEqual(ClaimEntitlement.output(ans("UNKNOWN_WORD"), WITHHELD), "UNKNOWN_WORD")
    def test_output_withheld(self): self.assertEqual(ClaimEntitlement.output(ans(), WITHHELD), "UNKNOWN")
    def test_output_qualified_not_committed(self): self.assertEqual(ClaimEntitlement.output(ans(), QUALIFIED), "UNKNOWN")
    def test_output_asked(self): self.assertEqual(ClaimEntitlement.output(ans(), ASKED), "ASK")
    def test_output_asserted(self): self.assertEqual(ClaimEntitlement.output(ans(), ASSERTED), "on:table")
    def test_summary(self): self.assertIn("level", prov().summary())
    def test_separate_uncertainties(self):
        p = prov(L0, 0.9, 0.0, 0.3); self.assertNotEqual(p.linguistic_uncertainty, p.epistemic_uncertainty)


class ProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = make_system("unified"); prepare(cls.agent, CAPS, SPLITS, POST)
        cls.x = SPLITS["test"][0]; cls.agent.new_world(); cls.agent.ingest(cls.x["observations"])
    def test_fields(self):
        a = self.agent.answer(self.x["items"][0].text); p = provenance(self.agent, a)
        for f in ("supporting_observations", "inference_rules", "source_reliabilities", "identity_assumptions", "lexical_assumptions", "remaining_alternatives"):
            self.assertTrue(hasattr(p, f))
    def test_level_range(self):
        for it in self.x["items"][:20]: self.assertIn(provenance(self.agent, self.agent.answer(it.text)).level, (L0, L1, L2, L3))
    def test_identity_query_is_L2_without_rule(self):
        a = Answer("t", "on:table", "ACCEPT", "where"); a.features["identity_p"] = 0.97
        self.assertEqual(provenance(self.agent, a).level, L2)
    def test_identity_query_with_rule_L1(self):
        a = Answer("t", "on:table", "ACCEPT", "where"); a.features["identity_p"] = 0.97
        a.features["identity_rule"] = eliminate(("cube", "red"), ["o1", "o2"], {"o1": "blue"}, {"o1": 0.99})
        self.assertEqual(provenance(self.agent, a).level, L1)
    def test_evidence_level_direct(self):
        k = next((k for k in self.agent.belief.graph.props if k[0] == "loc"), None)
        if k: self.assertIn(evidence_level(self.agent, k)[0], (L0, L1, L2))
    def test_rules_recorded(self): self.assertIsInstance(self.agent.rules, list)
    def test_explicit_elimination_method(self):
        r = self.agent.explicit_elimination("cube", "red"); self.assertIsInstance(r, EliminateCandidates)


# ================================================================ lexical calibration

class LexicalPosteriorTests(unittest.TestCase):
    def test_features_dim(self):
        agent = make_system("no_elimination"); prepare(agent, CAPS, SPLITS, POST)
        for x in SPLITS["test"][:2]: agent.new_world(); agent.ingest(x["observations"])
        for ex in examples_from_agent(agent): self.assertEqual(ex.feats.shape[1], len(FEATURES))
    def test_distribution_sums(self):
        ex = LexicalExample("w", ["type:cube", "type:ball"], torch.rand(2, len(FEATURES)), "type:cube")
        self.assertAlmostEqual(sum(POST.distribution(ex).values()), 1.0, places=5)
    def test_null_when_no_concepts(self): self.assertEqual(POST.distribution(LexicalExample("w", [], torch.zeros(0, 9), None)), {NULL: 1.0})
    def test_fit_learns(self):
        ex = [LexicalExample("w", ["a", "b"], torch.tensor([[5.0] + [0] * 8, [0.0] * 9]), "a") for _ in range(10)]
        d = LexicalPosterior().fit(ex, epochs=200).distribution(ex[0]); self.assertGreater(d["a"], d["b"])
    def test_parameter_count(self): self.assertEqual(POST.parameter_count, len(FEATURES) + 1)
    def test_calibration_points_native(self):
        ex = [LexicalExample("w", ["a"], torch.zeros(1, 9), "a")]; self.assertEqual(calibration_points(ex, None, {"w": (0.4, "a")}), [(0.4, True)])
    def test_calibration_points_model(self):
        ex = [LexicalExample("w", ["a"], torch.zeros(1, 9), "a")]; self.assertEqual(len(calibration_points(ex, POST)), 1)
    def test_precision_at(self): self.assertEqual(precision_at([(0.96, True), (0.5, False)], (0.9,))[0.9], (1.0, 1))
    def test_calibrated_better_than_native(self):
        from core.unified import calibration_report
        ex, conf = lexical_examples(0)
        a = calibration_report(*zip(*calibration_points(ex, POST)))["ece"]; b = calibration_report(*zip(*calibration_points(ex, None, conf)))["ece"]
        self.assertLess(a, b)


class EntropyLifecycleTests(unittest.TestCase):
    def test_entropy_open_no_candidates(self): self.assertAlmostEqual(entropy({NULL: 1.0}, 20), math.log(20))
    def test_entropy_certain(self): self.assertAlmostEqual(entropy({"a": 1.0}, 20), 0.0)
    def test_entropy_closed(self): self.assertAlmostEqual(entropy({"a": 0.5, "b": 0.5}), math.log(2))
    def test_unknown(self): self.assertEqual(LexicalLifecycle().status({NULL: 1.0})[0], LEX_UNKNOWN)
    def test_hypothesis(self): self.assertEqual(LexicalLifecycle().status({"a": 0.6, NULL: 0.4})[0], LEX_HYPOTHESIS)
    def test_belief_needs_two_situations(self):
        self.assertEqual(LexicalLifecycle().status({"a": 0.97, NULL: 0.03}, 1, 1)[0], LEX_HYPOTHESIS)
        self.assertEqual(LexicalLifecycle().status({"a": 0.97, NULL: 0.03}, 1, 2)[0], LEX_BELIEF)
    def test_stable(self): self.assertEqual(LexicalLifecycle().status({"a": 0.995, NULL: 0.005}, 2, 3)[0], LEX_STABLE)
    def test_low_is_unknown(self): self.assertEqual(LexicalLifecycle().status({"a": 0.3, NULL: 0.7})[0], LEX_UNKNOWN)
    def test_regression_recorded(self):
        agent = make_system("no_elimination"); prepare(agent, CAPS, SPLITS, POST)
        cal = CalibratedLexicon(POST); from core.grounded import LexicalMemory
        agent.lexmem = LexicalMemory()
        agent.lexmem.observe("zorg", {"type:cube": "support"}, "s", 1.0, "w1:le zorg est là")
        agent.lexmem.observe("zorg", {"type:cube": "support"}, "t", 1.0, "w2:le zorg est ici")
        s1 = cal.status(agent, "zorg")[0]
        for i in range(6): agent.lexmem.observe("zorg", {"type:cube": "refute"}, "u", 1.0, f"w{3 + i}:le zorg")
        s2 = cal.status(agent, "zorg")[0]
        self.assertNotEqual(s1, s2) if s1 != LEX_UNKNOWN else self.assertEqual(s2, LEX_UNKNOWN)
    def test_attach_changes_word_meaning(self):
        agent = make_system("callex"); cal = prepare(agent, CAPS, SPLITS, POST)
        self.assertIs(agent.calibrated_lexicon, cal)


class PolysemyTests(unittest.TestCase):
    def setUp(self): self.pp = PolysemyPosterior().fit(CAPS)
    def test_two_senses(self): self.assertGreaterEqual(len(self.pp.prior), 2)
    def test_posterior_sums(self):
        self.assertAlmostEqual(sum(self.pp.posterior({"patient": "cube", "target": "box", "rel": "in", "domain": "physical"}).values()), 1.0)
    def test_list_context_assign(self):
        p = self.pp.posterior({"patient": "book", "target": "list", "rel": "in", "domain": "inventory"}); self.assertEqual(max(p, key=p.get), "act:assign")
    def test_physical_context_place(self):
        p = self.pp.posterior({"patient": "cube", "target": "table", "rel": "on", "domain": "physical"}); self.assertEqual(max(p, key=p.get), "act:place")
    def test_empty(self): self.assertEqual(PolysemyPosterior().posterior({}), {})


class BehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.sim = SimilarityByBehavior(CAPS)
    def test_normalized(self): self.assertTrue(torch.allclose(self.sim.M.norm(dim=1), torch.ones(len(self.sim.words)), atol=1e-5))
    def test_similarity_symmetric(self): S = self.sim.similarity(); self.assertTrue(torch.allclose(S, S.T, atol=1e-5))
    def test_classes_emerge(self): self.assertGreater(len(self.sim.classes(0.9)), 3)
    def test_purity(self): self.assertGreater(semantic_classes()["purity"], 0.8)
    def test_class_purity_fn(self): self.assertEqual(class_purity([["a", "b"]], {"a": "type:x", "b": "type:y"}), 1.0)
    def test_analogy_candidate(self):
        c, s = self.sim.analogy([(["le", "zorg", "est", "rouge"], 1)], {w: "type:cube" for w in self.sim.words[:3]})
        self.assertTrue(c is None or isinstance(c, str))
    def test_no_embeddings_imported(self): self.assertFalse(any("gensim" in m or "transformers" in m for m in imports("core/lexical_calibration.py")))


class UnknownWordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = make_system("no_elimination"); prepare(cls.agent, CAPS, SPLITS, POST)
        for x in SPLITS["test"]: cls.agent.new_world(); cls.agent.ingest(x["observations"])
    def test_unknown_provenance(self):
        c = unknown_word_candidates(self.agent, "xyzzy")[0]; self.assertEqual(c["provenance"], UW_UNKNOWN)
    def test_analogy_never_fact(self):
        sim = SimilarityByBehavior(CAPS)
        for c in unknown_word_candidates(self.agent, "brique", CalibratedLexicon(POST), sim, [(["la", "brique", "est", "rouge"], 1)]):
            if c["provenance"] == ANALOGY: self.assertEqual(c["status"], LEX_HYPOTHESIS)
    def test_structural_provenance(self):
        cs = unknown_word_candidates(self.agent, "brique", CalibratedLexicon(POST))
        self.assertTrue(all(c["provenance"] in (STRUCTURAL, MORPHOLOGICAL, ANALOGY, UW_UNKNOWN) for c in cs))
    def test_morphology_probability_range(self):
        p = morphology_probability(self.agent, "bleue", "color:blue"); self.assertTrue(0.0 <= p <= 1.0)
    def test_trap_stays_unknown(self): self.assertEqual(self.agent.morph.candidates("tablette", self.agent.lex.lexicon()), [])


class CompositionTests(unittest.TestCase):
    def wm(self, t): return {"pose": "act:place", "tombé": "act:fall", "poussé": "act:push"}.get(t)
    def test_re_prefix(self): self.assertEqual(compose("repose", self.wm)[0], "act:place")
    def test_causative(self): self.assertEqual(compose("fait tomber", self.wm, {"tombé": "act:fall"})[0], "cause(act:fall)")
    def test_particle(self): self.assertEqual(compose("pose dedans", self.wm)[0], ("act:place", "rel:in"))
    def test_unknown_verb(self): self.assertIsNone(compose("met dedans", self.wm)[0])
    def test_stem_lookup(self): self.assertEqual(stem_lookup("pousse", {"poussé": "act:push"}), "act:push")
    def test_stem_ambiguous(self): self.assertIsNone(stem_lookup("pose", {"posé": "act:place", "posté": "act:send"}))
    def test_trace(self): self.assertTrue(compose("repose", self.wm)[1])
    def test_composition_eval(self): self.assertGreaterEqual(sum(v["correct"] for v in composition_eval().values()), 3)


# ================================================================ epistemic bench

def R(**kw):
    base = {"category": "correct", "intent": "where", "orig": "on:table", "ideal": "on:table", "relabeled": True, "raw": "on:table",
            "out": "on:table", "decision": ASSERTED, "level": L0, "q": 0.99, "p_belief": 0.99, "lin_unc": 0.0, "epi_unc": 0.01,
            "id_unc": 0.0, "diversity": 2, "reason": None, "flags": [], "identity_ok": True, "lexical": {}, "rules": []}
    base.update(kw); return base


class BenchmarkFunctionTests(unittest.TestCase):
    def test_score(self): self.assertEqual(score([R(), R(out="x")], "out", "orig")["false_confident"], 0.5)
    def test_score_abstention(self): self.assertEqual(score([R(out="UNKNOWN")])["abstention"], 1.0)
    def test_alignment_perfect(self): self.assertEqual(knowability_alignment([R()])["f1"], 1.0)
    def test_alignment_false_positive(self): self.assertEqual(knowability_alignment([R(ideal="UNKNOWN")])["precision"], 0.0)
    def test_alignment_by_entitlement(self): self.assertIn(ASSERTED, knowability_alignment([R()])["by_entitlement"])
    def test_audit_keys(self): self.assertEqual(set(audit([R()])), set(AUDIT))
    def test_audit_ambiguity(self): self.assertEqual(audit([R(orig="ASK", ideal="ASK")])["real_ambiguity"], 1)
    def test_audit_oracle_strict(self): self.assertEqual(audit([R(orig="UNKNOWN")])["oracle_too_strict"], 1)
    def test_audit_elimination(self): self.assertEqual(audit([R(orig="UNKNOWN", rules=["EliminateCandidates(x)"])])["logical_elimination"], 1)
    def test_audit_hidden(self): self.assertEqual(audit([R(orig="UNKNOWN", ideal="UNKNOWN", level=L2)])["hidden_information"], 1)
    def test_audit_temporal(self): self.assertEqual(audit([R(orig="UNKNOWN", ideal="UNKNOWN", category="temporal")])["temporal_error"], 1)
    def test_attribute_temporal(self): self.assertEqual(attribute(R(category="temporal", out="x"))[0], "TEMPORAL")
    def test_attribute_lexical(self): self.assertEqual(attribute(R(raw="UNKNOWN_WORD", out="UNKNOWN_WORD"))[0], "LEXICAL")
    def test_attribute_reference(self): self.assertEqual(attribute(R(level=L2, out="x"))[0], "REFERENCE")
    def test_attribute_knowability(self): self.assertEqual(attribute(R(orig="UNKNOWN"), "orig")[0], "KNOWABILITY_REFERENCE")
    def test_attribute_secondary(self): self.assertIn("REFERENCE", attribute(R(category="temporal", level=L2, out="x"))[1])
    def test_attribute_default(self): self.assertEqual(attribute(R(out="x"))[0], "EVIDENCE")
    def test_causes(self): self.assertEqual(set(error_attribution([R(out="x")])[0]), set(CAUSES))
    def test_conversion(self):
        c = conversion([R(), R(raw="x", decision=QUALIFIED), R(raw="y", decision=WITHHELD)], [R(raw="UNKNOWN")] * 3)
        self.assertEqual((c["correct_asserted"], c["wrong_qualified"], c["correctly_withheld"]), (1, 1, 1))
    def test_pareto_front_nondominated(self):
        rows = [R(q=0.5 + 0.05 * i, raw="on:table" if i % 3 else "x") for i in range(10)]
        pts, front = pareto(rows)
        for p in front:
            self.assertFalse(any(o["success"] > p["success"] and o["false_confident"] < p["false_confident"] for o in pts))
    def test_pareto_threshold_formula(self):
        pts, _ = pareto([R()], wrong=(20,), abstain=(1,), allow_l2=(False,)); self.assertAlmostEqual(pts[0]["threshold"], 0.95)


class SystemIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {}
        for name in ("no_elimination", "entitled", "unified_entitled"):
            a = make_system(name); cal = prepare(a, CAPS, SPLITS, POST)
            cls.rows[name] = evaluate(a, SPLITS["test"], LABELS, cal)
    def test_rows_have_both_golds(self):
        for r in self.rows["entitled"]: self.assertIn("ideal", r); self.assertIn("orig", r)
    def test_entitled_outputs_only_asserted(self):
        for r in self.rows["entitled"]:
            if r["out"] not in ("UNKNOWN", "ASK", "UNKNOWN_WORD"): self.assertEqual(r["decision"], ASSERTED)
    def test_unified_entitled_fc_bounded(self): self.assertLess(score(self.rows["unified_entitled"])["false_confident"], 0.06)
    def test_baseline_untouched_output(self):
        for r in self.rows["no_elimination"]: self.assertEqual(r["out"], r["raw"])
    def test_linguistic_epistemic_columns(self):
        for r in self.rows["entitled"]: self.assertTrue(0 <= r["lin_unc"] <= 1 and 0 <= r["epi_unc"] <= 1)
    def test_levels_present(self): self.assertTrue({r["level"] for r in self.rows["unified_entitled"]} <= {L0, L1, L2, L3})
    def test_L2_never_asserted(self):
        for r in self.rows["unified_entitled"]:
            if r["level"] == L2: self.assertNotEqual(r["decision"], ASSERTED)


# ================================================================ discourse: separate uncertainties, minimal revision

class DiscourseEpistemicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.discourse_benchmark import lexicon_agent
        cls.L = lexicon_agent(0)
    def state(self):
        from core.unified import UnifiedCognitiveState
        u = UnifiedCognitiveState(self.L); u.reliability.stats["capteur"] = [60.0, 1.0]; u.reliability.stats["alice"] = [30.0, 2.0]
        for e, t in (("e1", "cube"), ("e2", "cube"), ("e3", "key")): u.observe("capteur", ("type", e), t)
        return u
    def test_polysemous_command_asks(self):
        u = self.state(); a = u.query("Range la clé."); self.assertEqual(a.action, "ASK"); self.assertEqual(a.reason, "linguistic")
    def test_linguistic_uncertainty_field(self):
        u = self.state(); a = u.query("Range la clé."); self.assertGreaterEqual(a.linguistic_uncertainty, 0.5)
    def test_epistemic_uncertainty_field(self):
        u = self.state(); u.observe("capteur", ("color", "e3"), "red"); a = u.query("De quelle couleur est la clé ?")
        self.assertLess(a.epistemic_uncertainty, 0.1); self.assertEqual(a.linguistic_uncertainty, 0.0)
    def test_unknown_word_linguistic_not_epistemic(self):
        u = self.state(); a = u.query("Où est le gobelet ?"); self.assertNotEqual(a.action, "ANSWER")
    def test_minimal_revision_one_node(self):
        from core.third_party_eval import minimal_self_correction
        from core.third_party_discourse import scenarios_trompeurs
        r = minimal_self_correction(self.L, scenarios_trompeurs(20))
        self.assertLessEqual(r["mean_revised_nodes"], r["mean_support_nodes"]); self.assertEqual(r["still_wrong"], 0)
    def test_minimal_revision_cause(self):
        u = self.state(); u.observe("capteur", ("color", "e2"), "blue"); u.say("alice", "Le cube rouge est sur la table.")
        a = u.query("Où est le cube rouge ?"); new, info = u.minimal_revision(a, "Où est le cube rouge ?")
        self.assertIn(info["cause"], ("source", "identity", "inference_rule", None))
    def test_suspension_persistent(self):
        from core.unified import SUSPENDED, TruthMaintenance
        t = TruthMaintenance(); t.add("a", "identity"); t.add("b", "evidence", deps=["a"])
        t.set("b", SUSPENDED); t.nodes["b"].info["own"] = SUSPENDED; t.propagate(); self.assertEqual(t.nodes["b"].status, SUSPENDED)


def _dev_case(sid):
    def t(self):
        from core.discourse_benchmark import lexicon_agent
        from core.third_party_eval import run_scenario
        L = DevScenarioTests.L = getattr(DevScenarioTests, "L", None) or lexicon_agent(0)
        s = next(x for x in dev_set() + dev_generated() if x["id"] == sid)
        rows, _ = run_scenario(L, s)
        self.assertFalse(any(r["false_confident"] for r in rows), [(r["text"], r["expected"], r["got"]) for r in rows])
    return t


class DevScenarioTests(unittest.TestCase):
    pass


for _s in dev_set() + dev_generated():
    setattr(DevScenarioTests, f"test_dev_{_s['id']}_{_s['catégorie']}_no_false_confident", _dev_case(_s["id"]))


if __name__ == "__main__":
    unittest.main()

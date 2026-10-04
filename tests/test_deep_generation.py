import inspect
import json
import math
import random
import unittest

import torch

import core.deep_experiments as DX
import core.deep_gen_report as GR
import core.deep_generation as DG
import core.deep_reachability as DR
import core.deep_search as DS
import core.deep_world as DW
from core.boundary_program import BNode, atoms, canonical, predict
from core.deep_fit import FitCounter, FitData, separation_fit
from core.deep_generation import (OPS, TYPES, Hole, Literal, PartialStructure, SignatureSynthesizer, boundary_points, build_tables, detect_primitives,
                                  discover_atoms_by_separation, generate_candidates, literals_from, make_hole, pair_errors, select_literals, tree_to_program, well_typed)
from core.deep_reachability import (ERROR_TAXONOMY, OPERATORS, RECALLS, RECOVERED, Reachability, basis_quality, chain_reachable, classify, first_divergence, flatten,
                                    leaves, operator_reachable, oracle_path, parent_closure, parents_used, representable, space_size, strahler)
from core.deep_search import DeepSearch
from core.deep_structure import FragmentLibrary, algebraic_form, anti_unify, n_atoms, normalize, skel_key, structurally_equivalent, subtrees

A = atoms()
C = lambda k, a, b: BNode(k, children=(a, b))
G = lambda s=0: torch.Generator().manual_seed(s)
T = lambda c, ch: BNode("translate", params=c, children=(ch,))
DISC = T((0.2, 0.0), BNode("pred", "norm", "le", (1.0,)))
DISC2 = T((-1.2, 0.8), BNode("pred", "norm", "le", (0.9,)))
HALF = BNode("pred", "dot", "le", (0.3, 0.1))
RING = T((-0.5, 0.4), BNode("pred", "norm", "in", (0.5, 1.2)))
BAND = BNode("pred", "dot", "in", (1.0, -0.5, 0.4))
CHAIN4 = C("and", C("or", C("xor", DISC, HALF), RING), BAND)
BAL4 = C("xor", C("and", DISC, HALF), C("or", RING, BAND))


def X0(n=300, s=0): return torch.rand(n, 2, generator=G(s)) * 6 - 3


# ================================================================ reachability, oracle, parent closure

class ReachabilityTests(unittest.TestCase):
    def test_statuses(self): self.assertEqual(Reachability.ALL, ("REPRESENTABLE_AND_REACHABLE", "REPRESENTABLE_BUT_NOT_REACHED", "UNREACHABLE_WITH_CURRENT_OPERATORS", "NOT_REPRESENTABLE"))
    def test_taxonomy(self): self.assertEqual(len(ERROR_TAXONOMY), 11); self.assertIn("TRUE_PARENT_NOT_GENERATED", ERROR_TAXONOMY); self.assertIn("OBSERVATIONAL_NON_IDENTIFIABILITY", ERROR_TAXONOMY)
    def test_representable(self): self.assertTrue(representable(BAL4)[0])
    def test_not_representable_none(self): self.assertFalse(representable(None)[0])
    def test_not_representable_unknown_atom(self): ok, why = representable(C("and", DISC, BNode("pred", "norm", "le", (1.0,)))); self.assertFalse(ok); self.assertIn("atome absent", why)
    def test_not_representable_depth(self): self.assertFalse(representable(BAL4, max_depth=3)[0])
    def test_all_world_atoms_in_grammar(self):
        for k in DW.ATOM_TYPES: self.assertTrue(representable(DW.atom(random.Random(1), k))[0], k)
    def test_chain_is_chain_reachable(self): self.assertTrue(chain_reachable(CHAIN4))
    def test_balanced_mixed_ops_not_chain_reachable(self): self.assertFalse(chain_reachable(BAL4))
    def test_balanced_same_op_is_chain_reachable(self): self.assertTrue(chain_reachable(C("and", C("and", DISC, HALF), C("and", RING, BAND))))       # associativité : ré-associable en chaîne
    def test_depth3_always_chain(self):
        for s in range(20): self.assertTrue(chain_reachable(DW.random_tree(random.Random(s), 3)))
    def test_operator_sets(self): self.assertEqual(set(OPERATORS), {"baseline", "flat", "bottom_up", "top_down", "bidirectional", "best_first", "factorized"})
    def test_extend_atom_cannot_build_balanced(self): ok, miss = operator_reachable(BAL4, OPERATORS["baseline"]); self.assertFalse(ok); self.assertEqual(miss, "compose_pair")
    def test_top_down_same_class_as_chain(self): self.assertFalse(operator_reachable(BAL4, OPERATORS["top_down"])[0]); self.assertTrue(operator_reachable(CHAIN4, OPERATORS["top_down"])[0])
    def test_compose_pair_builds_balanced(self): self.assertTrue(operator_reachable(BAL4, OPERATORS["bottom_up"])[0])
    def test_synthesis_builds_any_tree(self): self.assertTrue(operator_reachable(BAL4, OPERATORS["factorized"])[0])
    def test_fragment_makes_reachable(self):
        self.assertTrue(operator_reachable(BAL4, ("extend_atom", "extend_fragment"), [algebraic_form(C("or", RING, BAND))])[0]); self.assertFalse(operator_reachable(BAL4, ("extend_atom", "extend_fragment"), [])[0])
    def test_flatten(self): f = flatten(C("and", C("and", DISC, HALF), RING)); self.assertEqual(f[0], "and"); self.assertEqual(len(f[1]), 3)
    def test_strahler(self): self.assertEqual((strahler(DISC), strahler(CHAIN4), strahler(BAL4)), (1, 2, 3))
    def test_oracle_length(self):
        for p in (DISC, C("or", DISC, HALF), CHAIN4, BAL4): self.assertEqual(oracle_path(p)["length"], 2 * n_atoms(p) - 1)
    def test_oracle_ends_at_truth(self): self.assertEqual(oracle_path(BAL4)["steps"][-1]["form"], algebraic_form(BAL4))
    def test_oracle_intermediates(self): self.assertEqual(set(oracle_path(BAL4)["intermediates"]), {algebraic_form(C("and", DISC, HALF)), algebraic_form(C("or", RING, BAND))})
    def test_oracle_compose_pair_flag(self): self.assertTrue(oracle_path(BAL4)["needs_compose_pair"]); self.assertFalse(oracle_path(CHAIN4)["needs_compose_pair"])
    def test_oracle_alternatives(self): o = oracle_path(CHAIN4); self.assertTrue(all(s["alternatives"] >= 8 for s in o["steps"])); self.assertGreater(o["alternatives_product_log10"], 4)
    def test_oracle_is_diagnostic_only(self):
        self.assertIn("diagnostic seulement", inspect.getsource(oracle_path))
        for m in (DS, DG): self.assertNotIn("deep_reachability", inspect.getsource(m)); self.assertNotIn("oracle_path", inspect.getsource(m))
    def test_inference_never_imports_world(self):
        for m in (DG,): self.assertNotIn("deep_world", inspect.getsource(m)); self.assertNotIn(".truth", inspect.getsource(m))
    def test_parent_closure_holds(self):
        for p in (CHAIN4, BAL4): c = parent_closure(p); self.assertTrue(c["closed"]); self.assertEqual(c["invalid"], []); self.assertEqual(c["collisions"], [])
    def test_parent_closure_counts(self): self.assertEqual(parent_closure(BAL4)["subtrees"], 7)
    def test_parent_closure_dev_worlds(self):
        for d in (3, 4, 5, 6): self.assertTrue(parent_closure(DW.make_task(1, d).world.truth)["closed"])
    def test_prefixes_and_suffixes_are_search_objects(self):
        from core.deep_structure import StructureStore
        st = StructureStore()
        for s in subtrees(BAL4): self.assertIn(st.intern(s).key, st.nodes)
    def test_space_size(self): self.assertEqual((space_size(1), space_size(2)), (8.0, 108.0)); self.assertGreater(space_size(6), 1e8 - 1e7)
    def test_space_exact_small(self): lv = GR.enumerate_forms(3); self.assertEqual((len(lv[1]), len(lv[2])), (8, 108)); self.assertLessEqual(len(lv[3]), space_size(3))
    def test_basis_quality(self):
        b = basis_quality(C("and", RING, BAND)); self.assertEqual((b["atoms"], b["atoms_in_minimal_basis"], b["redundant_atoms"]), (2, 4, 2)); self.assertEqual(basis_quality(C("and", DISC, HALF))["description_inflation"], 1.0)


# ================================================================ diagnostic de pruning, taxonomie

class TaxonomyTests(unittest.TestCase):
    def cl(self, truth, strategy, gen, ranked=None, sel=None, fit=True, budget=False, cls=()):
        return classify(truth, strategy, {str(i): p for i, p in enumerate(gen)}, ranked if ranked is not None else gen, sel, fit, budget, cls)
    def test_recovered(self): r = self.cl(CHAIN4, "bottom_up", [CHAIN4], sel=CHAIN4); self.assertEqual((r["cause"], r["reachability"]), (RECOVERED, Reachability.REACHED)); self.assertTrue(all(r[k] for k in RECALLS))
    def test_unrepresentable(self): r = self.cl(None, "bottom_up", [DISC], sel=DISC); self.assertEqual((r["cause"], r["reachability"]), ("GRAMMAR_UNREPRESENTABLE", Reachability.NOT_REPRESENTABLE))
    def test_operators_insufficient(self): r = self.cl(BAL4, "baseline", [DISC], sel=DISC); self.assertEqual((r["cause"], r["reachability"], r["missing_operator"]), ("OPERATORS_INSUFFICIENT", Reachability.UNREACHABLE, "compose_pair"))
    def test_parent_not_generated(self): r = self.cl(CHAIN4, "bottom_up", [DISC, HALF], sel=DISC); self.assertEqual(r["cause"], "TRUE_PARENT_NOT_GENERATED"); self.assertEqual((r["divergence_size"], r["divergence_kind"]), (2, "not_generated"))
    def test_pruned_too_early(self):
        p2, p3 = C("xor", DISC, HALF), C("or", C("xor", DISC, HALF), RING); r = self.cl(CHAIN4, "bottom_up", [p2, p3, C("and", p2, BAND)], sel=p3)
        self.assertEqual(r["cause"], "PRUNED_TOO_EARLY"); self.assertEqual((r["divergence_size"], r["divergence_kind"]), (3, "not_extended"))
    def test_budget_exhausted(self):
        p2, p3 = C("xor", DISC, HALF), C("or", C("xor", DISC, HALF), RING); r = self.cl(CHAIN4, "bottom_up", [p2, p3, C("and", p2, BAND)], sel=p3, budget=True); self.assertEqual(r["cause"], "BUDGET_EXHAUSTED")
    def test_fit_failure(self): r = self.cl(CHAIN4, "bottom_up", [CHAIN4, DISC], sel=DISC, fit=False); self.assertEqual(r["cause"], "FIT_FAILURE"); self.assertTrue(r["recall_generated"]); self.assertFalse(r["recall_fitted"])
    def test_ranking_failure(self): r = self.cl(CHAIN4, "bottom_up", [CHAIN4, DISC], sel=DISC); self.assertEqual(r["cause"], "RANKING_FAILURE"); self.assertTrue(r["recall_top_k"])
    def test_non_identifiability(self): r = self.cl(CHAIN4, "bottom_up", [CHAIN4, DISC], sel=DISC, cls=[algebraic_form(CHAIN4)]); self.assertEqual(r["cause"], "OBSERVATIONAL_NON_IDENTIFIABILITY")
    def test_canonicalization_collapse(self): r = self.cl(CHAIN4, "bottom_up", [CHAIN4, DISC], ranked=[DISC], sel=DISC); self.assertEqual(r["cause"], "CANONICALIZATION_COLLAPSE"); self.assertFalse(r["recall_post_pruning"])
    def test_type_mismatch(self):
        r = classify(CHAIN4, "factorized", {"a": DISC}, [DISC], DISC, False, False, type_rejected=[algebraic_form(CHAIN4)]); self.assertEqual(r["cause"], "TYPE_MISMATCH")
    def test_fragment_missing(self):
        OPERATORS["_frag"] = ("extend_atom", "extend_fragment")
        try: r = self.cl(BAL4, "_frag", [DISC], sel=DISC)
        finally: del OPERATORS["_frag"]
        self.assertEqual(r["cause"], "FRAGMENT_MISSING")
    def test_recall_fields(self): self.assertEqual(RECALLS, ("recall_generated", "recall_pre_pruning", "recall_post_pruning", "recall_fitted", "recall_top_k", "recall_selected"))
    def test_top_k(self): r = classify(CHAIN4, "bottom_up", {str(i): p for i, p in enumerate([DISC, HALF, RING, BAND, DISC2, C("or", DISC, HALF), CHAIN4])}, [DISC, HALF, RING, BAND, DISC2, C("or", DISC, HALF), CHAIN4], DISC, True, False); self.assertFalse(r["recall_top_k"]); self.assertTrue(r["recall_post_pruning"])
    def test_parents_used(self): self.assertEqual(parents_used([C("and", C("xor", DISC, HALF), RING)]), {algebraic_form(C("xor", DISC, HALF)), algebraic_form(RING)})
    def test_first_divergence_none(self):
        gen = {algebraic_form(s) for s in subtrees(CHAIN4)}; used = parents_used(subtrees(CHAIN4)); self.assertEqual(first_divergence(CHAIN4, gen, used)["kind"], "none")
    def test_every_cause_in_taxonomy(self):
        src = inspect.getsource(classify)
        for c in ERROR_TAXONOMY: self.assertIn(c, src, c)
    def test_equivalent_generated_counts(self): r = self.cl(C("and", C("and", DISC, HALF), RING), "bottom_up", [C("and", DISC, C("and", HALF, RING))], sel=C("and", DISC, C("and", HALF, RING))); self.assertEqual(r["cause"], RECOVERED)


# ================================================================ canonicalisation

class CanonicalizationTests(unittest.TestCase):
    def test_commutative_same_key(self): self.assertEqual(skel_key(C("and", DISC, HALF)), skel_key(C("and", HALF, DISC)))
    def test_associative_same_form_different_key(self):
        a, b = C("or", C("or", DISC, HALF), RING), C("or", DISC, C("or", HALF, RING)); self.assertEqual(algebraic_form(a), algebraic_form(b)); self.assertNotEqual(skel_key(a), skel_key(b))       # pas fusionnés pendant la construction
    def test_semantically_equal_syntactically_different(self):
        a, b = C("xor", C("xor", DISC, HALF), RING), C("xor", HALF, C("xor", RING, DISC)); X = X0(); self.assertTrue(torch.equal(predict(a, X), predict(b, X))); self.assertTrue(structurally_equivalent(a, b))
    def test_different_semantics_not_merged(self): self.assertNotEqual(algebraic_form(C("and", C("or", DISC, HALF), RING)), algebraic_form(C("or", DISC, C("and", HALF, RING))))
    def test_normalize_keeps_function(self): p = C("xor", RING, C("and", HALF, DISC)); X = X0(); self.assertTrue(torch.equal(predict(normalize(p), X), predict(p, X)))
    def test_normalize_no_param_rewrite(self):
        z = T((0.0, 0.0), BNode("pred", "norm", "le", (1.0,))); self.assertEqual(skel_key(normalize(C("and", z, HALF))), skel_key(C("and", DISC, HALF)))       # translate(0,0) n'est PAS supprimé : la clé reste celle du squelette
    def test_param_dependent_canonical_would_collapse(self):
        z = T((0.0, 0.0), BNode("pred", "norm", "le", (1.0,))); self.assertNotEqual(json.dumps(canonical(z).skeleton().to_dict(), sort_keys=True), skel_key(z))                 # d'où le choix de normalize
    def test_intermediate_not_merged_blocks_descendant(self):
        from core.deep_structure import StructureStore
        st = StructureStore(); a = st.intern(C("or", C("or", A[0], A[3]), A[4])); b = st.intern(C("or", A[0], C("or", A[3], A[4]))); self.assertIsNot(a, b); self.assertIn(st.intern(C("or", A[3], A[4])).key, b.children)
    def test_signature_dedup_keeps_alternatives(self):
        X = X0(120); lits = [DISC, HALF, RING]; sig = torch.stack([predict(l, X).float() for l in lits]); tabs, st = build_tables(sig, torch.tensor([3.0, 2.0, 4.0])); self.assertGreater(st["merged2"] + st["merged3"], 0); self.assertTrue(any(len(v) for t in tabs for v in t.alts.values()))
    def test_dedup_removes_constants_and_trivial(self):
        X = X0(120); sig = torch.stack([predict(l, X).float() for l in (DISC, HALF)]); tabs, st = build_tables(sig, torch.tensor([3.0, 2.0])); self.assertGreater(st["degenerate2"], 0)
        for t in tabs: s = t.sig.sum(1); self.assertTrue(bool(((s > 0) & (s < 120)).all()))
    def test_dedup_prefers_shorter_description(self):
        X = X0(120); sig = torch.stack([predict(DISC, X).float()] * 2); tabs, _ = build_tables(sig, torch.tensor([5.0, 3.0])); self.assertEqual(float(tabs[0].params[0]), 3.0)


# ================================================================ typage, programmes partiels, trous

class TypedPartialTests(unittest.TestCase):
    def test_types(self): self.assertEqual(TYPES, ("Field", "Region", "Logic", "Transform"))
    def test_grammar_atoms_well_typed(self): self.assertTrue(all(well_typed(a) for a in A))
    def test_translate_of_dot_rejected(self): self.assertFalse(well_typed(T((0.0, 0.0), BNode("pred", "dot", "le"))))
    def test_double_translate_rejected(self): self.assertFalse(well_typed(T((0.0, 0.0), T((0.0, 0.0), BNode("pred", "norm", "le")))))
    def test_bare_norm_is_typed(self): self.assertTrue(well_typed(BNode("pred", "norm", "le")))
    def test_logic_needs_two_regions(self): self.assertTrue(well_typed(C("and", DISC, HALF))); self.assertFalse(well_typed(BNode("and", children=(DISC,))))
    def test_not_of_composite_rejected(self): self.assertFalse(well_typed(BNode("not", children=(C("and", DISC, HALF),))))
    def test_typing_gain_measured(self): c = GR.coverage(); self.assertEqual((c["raw_unary_nodes"], c["well_typed"]), (45, 30)); self.assertAlmostEqual(c["typing_reduction_atoms"], 1 / 3)
    def test_hole_xor(self):
        k = torch.tensor([1.0, 0, 1, 0]); y = torch.tensor([1.0, 1, 0, 0]); h = make_hole("xor", k, y, torch.ones(4), 2); self.assertEqual(h.target.tolist(), [0, 1, 1, 0]); self.assertEqual((h.unavoidable, h.type, h.depth_left), (0.0, "Region", 2))
    def test_hole_and(self):
        k = torch.tensor([1.0, 0, 1, 0]); y = torch.tensor([1.0, 1, 0, 0]); h = make_hole("and", k, y, torch.ones(4), 1); self.assertEqual(h.care.tolist(), [1, 0, 1, 0]); self.assertEqual(h.unavoidable, 1.0)
    def test_hole_or(self):
        k = torch.tensor([1.0, 0, 1, 0]); y = torch.tensor([1.0, 1, 0, 0]); h = make_hole("or", k, y, torch.ones(4), 1); self.assertEqual(h.care.tolist(), [0, 1, 0, 1]); self.assertEqual(h.unavoidable, 1.0)
    def test_hole_exactness(self):
        g = G(2); k = (torch.rand(200, generator=g) < 0.5).float(); y = (torch.rand(200, generator=g) < 0.5).float()
        for op in OPS:
            h = make_hole(op, k, y, torch.ones(200), 1); fill = h.target; pred = DG._combine(op, k, fill); self.assertEqual(float((pred != y).float().sum()), h.unavoidable)       # remplissage parfait ⇒ seules les erreurs inévitables
    def test_partial_structure(self): p = PartialStructure("and", ("lit", 0), Hole("Region", torch.zeros(3), torch.ones(3), 2, 1.5)); self.assertEqual(p.lower_bound(), 1.5)
    def test_pair_errors_match_bruteforce(self):
        g = G(1); S = (torch.rand(5, 60, generator=g) < 0.5).float(); Tt = (torch.rand(4, 60, generator=g) < 0.5).float(); y = (torch.rand(60, generator=g) < 0.5).float(); w = torch.rand(60, generator=g); E = pair_errors(S, Tt, y, w)
        for o, op in enumerate(OPS):
            for i in range(5):
                for j in range(4): self.assertAlmostEqual(float(E[o, i, j]), float((w * (DG._combine(op, S[i], Tt[j]) - y).abs()).sum()), places=4)
    def test_meet_finds_exact_composition(self):
        X = X0(240); lits = [DISC, HALF, RING, BAND, DISC2]; sig = torch.stack([predict(l, X).float() for l in lits]); y = predict(BAL4, X).float()
        syn = SignatureSynthesizer(sig, torch.tensor([float(l.n_params()) for l in lits]), y, max_depth=4); best = syn.solve(y, torch.ones(240), 4, 3)[0]; self.assertEqual(best[1], 0.0)
        self.assertEqual(algebraic_form(tree_to_program(best[2], [Literal(l, "x", 0) for l in lits])), algebraic_form(BAL4))
    def test_synthesis_depth5_through_holes(self):
        X = X0(260); tr = C("or", C("and", DISC, HALF), C("xor", RING, C("and", BAND, DISC2))); lits = [DISC, HALF, RING, BAND, DISC2]; sig = torch.stack([predict(l, X).float() for l in lits]); y = predict(tr, X).float()
        syn = SignatureSynthesizer(sig, torch.tensor([float(l.n_params()) for l in lits]), y, max_depth=5); best = syn.solve(y, torch.ones(260), 5, 3)[0]; self.assertEqual(best[1], 0.0)
    def test_select_literals_bounded_and_diverse(self):
        X = X0(200); sig = torch.stack([predict(DISC, X).float()] * 6 + [predict(HALF, X).float(), predict(RING, X).float()]); k = select_literals(sig, 5); self.assertEqual(len(k), 3)
    def test_select_literals_ignores_target(self): self.assertEqual(list(inspect.signature(select_literals).parameters), ["sig", "max_literals", "near"])


def _mk_logic(seed, d):
    def t(self):
        tr = DW.make_task(seed, d).world.truth; X = X0(300, seed); lits = leaves(tr); gen = generate_candidates(X, predict(tr, X), max_depth=d, extra_literals=lits, geometry=False)
        self.assertTrue(any(algebraic_form(p) == algebraic_form(tr) or float((predict(p, X) == predict(tr, X)).float().mean()) == 1.0 for p in gen.programs))
    return t


for _s, _d in ((0, 2), (1, 2), (2, 3), (3, 3), (4, 4), (5, 4), (6, 5), (7, 5)): setattr(TypedPartialTests, f"test_logic_stage_reaches_truth_d{_d}_s{_s}", _mk_logic(_s, _d))


# ================================================================ géométrie, séparation, générateur

class GenerationTests(unittest.TestCase):
    def test_boundary_points(self):
        X = X0(400); M, P, Q = boundary_points(X, predict(DISC, X)); self.assertGreater(len(M), 10); self.assertLess(float(((M - torch.tensor([0.2, 0.0])).norm(dim=1) - 1.0).abs().median()), 0.2)
    def test_boundary_pairs_opposite_sides_clean(self): X = X0(400); M, P, Q = boundary_points(X, predict(HALF, X)); self.assertGreater(float((predict(HALF, P) != predict(HALF, Q)).float().mean()), 0.6)
    def test_detect_line(self):
        X = X0(500); pr = detect_primitives(boundary_points(X, predict(HALF, X))); self.assertTrue(pr); self.assertEqual(pr[0].kind, "line")
    def test_detect_circle(self):
        X = X0(600); pr = detect_primitives(boundary_points(X, predict(DISC, X))); self.assertTrue(any(p.kind == "circle" and abs(p.params[2] - 1.0) < 0.25 for p in pr))
    def test_literals_are_grammar_atoms(self):
        X = X0(600); L = literals_from(detect_primitives(boundary_points(X, predict(C("or", DISC, HALF), X)))); G0 = DR.grammar_atom_keys(); self.assertTrue(L); self.assertTrue(all(skel_key(l.program) in G0 for l in L))
    def test_literal_complements(self): L = literals_from([DG.GeoPrimitive("line", (0.3, 0.1), 9, torch.zeros(0, 2))]); X = X0(100); self.assertTrue(bool((predict(L[0].program, X) != predict(L[1].program, X)).all()))
    def test_band_macro_constraint(self):
        L = literals_from([DG.GeoPrimitive("line", (0.3, 0.1), 9, torch.zeros(0, 2)), DG.GeoPrimitive("line", (0.32, 1.0), 9, torch.zeros(0, 2))]); self.assertIn("band", [l.source for l in L])
    def test_no_band_when_not_parallel(self):
        L = literals_from([DG.GeoPrimitive("line", (0.3, 0.1), 9, torch.zeros(0, 2)), DG.GeoPrimitive("line", (1.5, 1.0), 9, torch.zeros(0, 2))]); self.assertNotIn("band", [l.source for l in L])
    def test_ring_macro_constraint(self):
        L = literals_from([DG.GeoPrimitive("circle", (0.0, 0.0, 0.6), 9, torch.zeros(0, 2)), DG.GeoPrimitive("circle", (0.1, 0.0, 1.4), 9, torch.zeros(0, 2))]); self.assertIn("ring", [l.source for l in L])
    def test_separation_fit_finds_boundary(self):
        X = X0(500); M, P, Q = boundary_points(X, predict(DISC, X)); sk = A[3]; th, sep = separation_fit(sk, P, Q, torch.ones(len(M)), torch.tensor([[0.0, 0.2, 0.8], [1.5, 1.5, 0.5]], dtype=torch.float64), steps=80)
        self.assertGreater(float(sep.max()), 0.6 * len(M))
    def test_separation_is_polarity_free(self):
        X = X0(500); y = predict(DISC, X); a = boundary_points(X, y); b = boundary_points(X, 1 - y); self.assertEqual(len(a[0]), len(b[0]))
    def test_separation_marked_scientific_change(self): self.assertIn("SCIENTIFIC CHANGE", inspect.getsource(separation_fit))
    def test_separation_counts_budget(self):
        X = X0(300); c = FitCounter(); discover_atoms_by_separation(A, X, predict(C("xor", DISC, HALF), X), rounds=1, n_init=2, counter=c); self.assertEqual(c.fit_equivalents, 8 * 2 * 0.5)
    def test_generate_empty_when_no_literals(self): g = generate_candidates(X0(50), torch.zeros(50), geometry=False); self.assertEqual(g.programs, [])
    def test_generate_stats(self):
        X = X0(300); g = generate_candidates(X, predict(C("xor", DISC, HALF), X), max_depth=3, extra_literals=[DISC, HALF, RING], geometry=False)
        for f in ("S1", "S2", "S3", "attempted", "pairs_scored", "merged2", "degenerate2", "candidates", "t_tables", "t_synthesis", "type_rejected", "literals_dropped"): self.assertIn(f, g.stats)
    def test_generate_deterministic(self):
        X = X0(300); y = predict(C("xor", DISC, HALF), X); a = generate_candidates(X, y, max_depth=3, seed=1); b = generate_candidates(X, y, max_depth=3, seed=1); self.assertEqual(a.programs, b.programs)
    def test_candidates_unique_forms(self):
        X = X0(300); g = generate_candidates(X, predict(BAL4, X), max_depth=4, extra_literals=[DISC, HALF, RING, BAND], geometry=False); f = [algebraic_form(p) for p in g.programs]; self.assertEqual(len(f), len(set(f)))
    def test_factorized_strategy_runs(self):
        X = X0(240); y = predict(C("xor", DISC, HALF), X); o = DeepSearch("factorized", budget=60, max_depth=3, fact_rounds=1).run(FitData.build(X, y)); self.assertIsNotNone(o.best()); self.assertIn("compositions", o.stats); self.assertGreater(o.stats["compositions"], 0)
    def test_factorized_costs_separated(self):
        X = X0(200); o = DeepSearch("factorized", budget=40, max_depth=2, fact_rounds=0).run(FitData.build(X, predict(DISC, X)))
        for f in ("fit_equivalents", "generated", "fitted", "compositions", "seconds", "fit_seconds", "synthesis_seconds"): self.assertIn(f, o.stats)
    def test_factorized_respects_budget(self):
        X = X0(200); o = DeepSearch("factorized", budget=30, max_depth=3).run(FitData.build(X, predict(C("or", DISC, HALF), X))); self.assertLessEqual(o.stats["fit_equivalents"], 30 + 4)
    def test_ranking_untouched(self): src = inspect.getsource(DeepSearch._run_factorized); self.assertIn("self._fit(", src); self.assertNotIn("approx_evidence", src); self.assertNotIn("profiled_loglik", src)


# ================================================================ fragments et anti-unification comme générateurs

class GenerativeLibraryTests(unittest.TestCase):
    def lib(self):
        L = FragmentLibrary(); rng = random.Random(0)
        for i in range(6):
            a = DW.atom(rng, "disc"); b = DW.atom(rng, ("half", "band", "ring")[i % 3]); L.observe(C("and", a, b), f"t{i}"); L.observe(C("and", DW.atom(rng, "disc"), DW.atom(rng, ("half", "band", "ring")[i % 3])), f"u{i}")
        L.promote(); return L
    def test_library_promotes_generic(self): L = self.lib(); self.assertEqual(len(L), 3)
    def test_family_generated(self): L = self.lib(); self.assertTrue(any(len(f.variables) == 1 for f in L.families))
    def test_family_reaches_unseen(self):
        L = self.lib(); fam = [f for f in L.families if len(f.variables) == 1][0]; new = C("and", DISC, T((0.0, 0.0), BNode("pred", "angle", "in", (0.1, 0.5)))); self.assertFalse(L.contains(new)); self.assertIsNotNone(fam.matches(new))
    def test_family_false_candidates_counted(self):
        L = self.lib(); fam = [f for f in L.families if len(f.variables) == 1][0]; inst = [fam.instantiate({fam.variables[0]: a}) for a in A]; self.assertEqual(len({skel_key(s) for s in inst}), 8)       # 8 schémas proposés pour 3 observés
    def test_exact_fragments_do_not_reach_unseen(self): L = self.lib(); self.assertFalse(L.contains(C("or", DISC, HALF)))
    def test_library_coverage_of_truth(self):
        L = self.lib(); tr = C("xor", C("and", DISC, HALF), RING); cov = sum(L.contains(s) for s in subtrees(tr, 2) if n_atoms(s) == 2); self.assertEqual(cov, 1)
    def test_curriculum_does_not_memorize(self): L = self.lib(); self.assertTrue(all(f.skeleton.params is None for f in L.fragments)); self.assertTrue(all(f.parameterizations >= 2 for f in L.fragments))


# ================================================================ expériences du sous-jalon, fuites

class GenExperimentTests(unittest.TestCase):
    def test_subunits(self): ids = DX.subunit_ids("deepgen", 0); self.assertEqual(len(ids), 6 * 2 * (1 + len(DX.GEN_METHODS))); self.assertEqual(len(ids), len(set(ids)))
    def test_dev_only(self): self.assertEqual(DX.gen_task(0, 3, "standard").split, "dev"); self.assertTrue(DX.gen_task(0, 3, "standard").seed < DW.SPLIT_BASE["validation"])
    def test_no_sealed(self):
        for fn in (DX.gen_subunits, DX.gen_task, DX.run_generation, DX.mechanics): self.assertNotIn("sealed", inspect.getsource(fn))
        for m in (DG, DR, GR): self.assertNotIn("sealed_task", inspect.getsource(m)); self.assertNotIn("SPLIT_BASE[\"sealed\"]", inspect.getsource(m))
    def test_budget_curve(self): self.assertEqual(sorted(b for m, b in DX.GEN_METHODS if m == "factorized"), [50.0, 150.0, 450.0]); self.assertEqual(sorted(b for m, b in DX.GEN_METHODS if m == "bottom_up"), [50.0, 150.0, 450.0])
    def test_reference_budget_kept(self): self.assertIn(("flat", 150.0), DX.GEN_METHODS); self.assertIn(("bidirectional", 150.0), DX.GEN_METHODS)
    def test_tasks_deterministic(self): self.assertEqual(DX.gen_task(2, 4, "rich").world.uid, DX.gen_task(2, 4, "rich").world.uid)
    def test_mechanics_uses_truth_only_for_diagnosis(self): self.assertIn("diagnostic seulement", inspect.getsource(DX.mechanics)); self.assertIn("ORACLE DE LITTÉRAUX", inspect.getsource(DX.mechanics))
    def test_run_generation_search_gets_no_truth(self):
        src = inspect.getsource(DX.run_generation); i = src.index("DeepSearch(strat"); j = src.index(".run(data)", i); self.assertNotIn("tr", src[i:j].replace("strat", "")); self.assertNotIn("world", src[i:j])
    def test_make_targets(self):
        mk = open("Makefile").read()
        for t in ("deep-run-gen:", "deep-reachability:", "deep-coverage:", "deep-gen-aggregate", "deep-gen-report:", "deep-profile-generation:", "deep-resume:"): self.assertIn(t, mk)
    def test_reachability_report(self):
        r = GR.reachability(n=12); self.assertEqual(r[3]["reachable"]["baseline"], 1.0); self.assertEqual(r[6]["reachable"]["factorized"], 1.0); self.assertEqual(r[1]["representable"], 1.0); self.assertLess(r[6]["reachable"]["baseline"], 1.0)
    def test_end_to_end_small(self):
        t = DX.gen_task(0, 1, "standard"); r = DX.run_generation(t, "factorized", 30.0, 0)
        for f in RECALLS + ("cause", "reachability", "fit_equivalents", "n_candidates", "compositions", "seconds", "fit_seconds", "synthesis_seconds", "functional"): self.assertIn(f, r)
        json.dumps(r)
    def test_sealed_untouched(self): self.assertEqual(DW.sealed_hash(), DW.SEALED_SHA256)
    def test_sealed_manifest(self):
        m = json.load(open("docs/sealed_deep_manifest.json")); self.assertEqual(m["sha256"], DW.SEALED_SHA256); self.assertEqual(m["n_tasks"], 72); self.assertTrue(any("inspection adaptative" in r for r in m["rules"]))


if __name__ == "__main__":
    unittest.main()

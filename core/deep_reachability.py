"""Diagnostic de REACHABILITY (sous-jalon Deep Candidate Generation) — module d'ÉVALUATION.

Il reçoit la vraie structure : il sert UNIQUEMENT à répondre « cette structure est-elle représentable, atteignable par
nos opérateurs, et où la recherche réelle s'en écarte-t-elle ? ». Rien de ce module n'est appelé par le système
d'inférence ni par une méthode évaluée (test : aucun import depuis core/deep_search, deep_generation, deep_structure…).

Notions :
  Reachability            REPRESENTABLE_AND_REACHABLE / REPRESENTABLE_BUT_NOT_REACHED /
                          UNREACHABLE_WITH_CURRENT_OPERATORS / NOT_REPRESENTABLE
  oracle de génération    séquence minimale de compositions qui construit la vraie structure
  parent closure          tous les intermédiaires nécessaires existent-ils comme objets de recherche ?
  taxonomie d'erreur      une cause principale par échec
"""
from __future__ import annotations

import json
import math
from typing import Dict, Iterable, List, Optional, Sequence, Set

from core.boundary_program import BNode, atoms
from core.deep_structure import COMBINATORS, COMPACT_PARAMS, algebraic_form, is_composite, n_atoms, skel_key, subtrees


class Reachability:
    REACHED = "REPRESENTABLE_AND_REACHABLE"
    NOT_REACHED = "REPRESENTABLE_BUT_NOT_REACHED"
    UNREACHABLE = "UNREACHABLE_WITH_CURRENT_OPERATORS"
    NOT_REPRESENTABLE = "NOT_REPRESENTABLE"
    ALL = (REACHED, NOT_REACHED, UNREACHABLE, NOT_REPRESENTABLE)


ERROR_TAXONOMY = ("GRAMMAR_UNREPRESENTABLE", "OPERATORS_INSUFFICIENT", "TRUE_PARENT_NOT_GENERATED", "PRUNED_TOO_EARLY", "BUDGET_EXHAUSTED", "CANONICALIZATION_COLLAPSE",
                  "FRAGMENT_MISSING", "TYPE_MISMATCH", "FIT_FAILURE", "RANKING_FAILURE", "OBSERVATIONAL_NON_IDENTIFIABILITY")
RECOVERED = "RECOVERED"

# Opérateurs de génération de chaque stratégie (lecture du code, vérifiée par les tests de couverture).
#   extend_atom    : parent ∘ atome           (chaînes : tout nœud interne a au moins un enfant atome)
#   anchor_atom    : atome ∘ (reste)          (descente : même classe d'arbres que extend_atom)
#   extend_fragment: parent ∘ fragment de la bibliothèque
#   compose_pair   : composition de deux sous-structures déjà construites (arbres équilibrés)
#   synthesize     : synthèse booléenne sur signatures (toute forme d'arbre sur les atomes candidats)
OPERATORS = {"baseline": ("extend_atom",), "flat": ("extend_atom",), "bottom_up": ("extend_atom", "compose_pair"), "top_down": ("anchor_atom",),
             "bidirectional": ("extend_atom", "anchor_atom", "meet_existing"), "best_first": ("extend_atom", "compose_pair"), "factorized": ("synthesize",)}


def grammar_atom_keys(allowed=None): return {skel_key(a) for a in (atoms(allowed) if allowed is not None else atoms())}


def leaves(n: BNode):
    if is_composite(n): return [l for c in n.children for l in leaves(c)]
    return [n]


def representable(truth: Optional[BNode], max_depth=6, allowed=None):
    """La structure est-elle exprimable dans la grammaire : atomes connus, combinateurs connus, profondeur et budget
    compact respectés ? Renvoie (bool, raison)."""
    if truth is None: return False, "aucun programme de la grammaire (monde hors classe)"
    G = grammar_atom_keys(allowed); miss = [skel_key(l) for l in leaves(truth) if skel_key(l) not in G]
    if miss: return False, f"atome absent de la grammaire : {miss[0][:60]}"
    if n_atoms(truth) > max_depth: return False, f"profondeur {n_atoms(truth)} > limite {max_depth}"
    if truth.n_params() > COMPACT_PARAMS: return False, "budget compact dépassé"
    return True, ""


def flatten(n: BNode):
    """Arbre n-aire à associativité près : ('and'|'or'|'xor', [enfants]) | ('atom', clé)."""
    if not is_composite(n): return ("atom", skel_key(n))
    items = []
    def flat(z):
        if z.kind == n.kind:
            for c in z.children: flat(c)
        else: items.append(flatten(z))
    flat(n); return (n.kind, items)


def chain_reachable(truth: BNode):
    """Atteignable par des opérateurs qui n'ajoutent qu'UN ATOME à la fois : il doit exister une ré-association en
    chaîne, c'est-à-dire que chaque nœud (aplati) a au plus un enfant composite, récursivement."""
    def ok(t):
        if t[0] == "atom": return True
        comp = [c for c in t[1] if c[0] != "atom"]; return len(comp) <= 1 and all(ok(c) for c in comp)
    return ok(flatten(truth))


def operator_reachable(truth: BNode, operators: Sequence[str], library_forms: Iterable[str] = ()):
    """Existe-t-il AU MOINS UNE séquence valide des opérateurs donnés qui produit la structure (sans budget) ?
    Renvoie (bool, opérateur manquant | '')."""
    ops = set(operators); lib = set(library_forms)
    if "synthesize" in ops or "compose_pair" in ops: return True, ""
    if chain_reachable(truth): return True, ""
    if "extend_fragment" in ops or "meet_existing" in ops:
        def ok(t):                                             # un enfant composite supplémentaire est permis s'il est fourni (bibliothèque / rencontre)
            if not is_composite(t): return True
            kids = _flat_children(t); comp = [c for c in kids if is_composite(c)]; free = [c for c in comp if algebraic_form(c) not in lib]
            return (len(free) <= 1 or "meet_existing" in ops) and all(ok(c) for c in comp)
        if ok(truth): return True, ""
    return False, "compose_pair"


def _flat_children(n: BNode):
    out = []
    def flat(z):
        if z.kind == n.kind and is_composite(z):
            for c in z.children: flat(c)
        else: out.append(z)
    for c in n.children: flat(c)
    return out


def strahler(n: BNode):
    """Largeur nécessaire : nombre de résultats partiels à détenir simultanément pour construire l'arbre (nombre de
    Strahler). Chaîne → 2 ; arbre équilibré à 4 feuilles → 3."""
    if not is_composite(n): return 1
    a, b = strahler(n.children[0]), strahler(n.children[1]); return max(a, b) if a != b else a + 1


def oracle_path(truth: BNode, n_atom_types=8, pool=8):
    """ORACLE DE GÉNÉRATION (diagnostic seulement) : séquence minimale d'opérations qui construit la vraie structure,
    en post-ordre. Chaque étape donne la forme produite et le nombre de CHOIX ALTERNATIFS qu'un générateur non guidé
    aurait à cet endroit."""
    steps = []
    def build(n):
        if not is_composite(n): steps.append({"op": "fit_atom", "form": algebraic_form(n), "size": 1, "alternatives": n_atom_types}); return
        a, b = n.children; build(a); build(b)
        if not is_composite(b) or not is_composite(a): steps.append({"op": "extend_atom", "form": algebraic_form(n), "size": n_atoms(n), "alternatives": len(COMBINATORS) * n_atom_types})
        else: steps.append({"op": "compose_pair", "form": algebraic_form(n), "size": n_atoms(n), "alternatives": len(COMBINATORS) * pool * (pool - 1) // 2})
    build(truth); comp = [s for s in steps if s["op"] != "fit_atom"]
    return {"steps": steps, "length": len(steps), "compositions": len(comp), "intermediates": [s["form"] for s in comp[:-1]], "width": strahler(truth),
            "alternatives_product_log10": sum(math.log10(s["alternatives"]) for s in steps), "needs_compose_pair": any(s["op"] == "compose_pair" for s in steps), "depth": n_atoms(truth)}


def parent_closure(truth: BNode, max_depth=6):
    """Tous les intermédiaires nécessaires (préfixes, suffixes, sous-arbres, compositions partielles) sont-ils des
    objets de recherche valides : représentables, et identifiables par leur clé canonique sans collision ?"""
    subs = subtrees(truth, min_atoms=1); out = {"subtrees": len(subs), "invalid": [], "collisions": []}; seen: Dict[str, str] = {}
    for s in subs:
        ok, why = representable(s, max_depth)
        if not ok: out["invalid"].append(why)
        k = skel_key(s); f = algebraic_form(s)
        if k in seen and seen[k] != f: out["collisions"].append(k[:80])
        seen[k] = f
    flat = flatten(truth); partial = []
    def walk(t):                                                  # compositions partielles d'un nœud n-aire : tout sous-ensemble d'enfants est un objet valide
        if t[0] != "atom":
            partial.append(len(t[1]))
            for c in t[1]: walk(c)
    walk(flat); out["nary_nodes"] = len(partial); out["max_arity"] = max(partial, default=0); out["closed"] = not out["invalid"] and not out["collisions"]
    return out


def space_size(depth: int, n_atom_types=8):
    """Nombre de formes algébriques distinctes à `depth` atomes (arbres binaires à commutativité près ; borne haute car
    l'associativité fusionne encore des formes). T(1) = types ; T(d) = 3 · Σ_{i ≤ d−i} T(i)·T(d−i) (÷2 si i = d−i)."""
    T = {1: float(n_atom_types)}
    for d in range(2, depth + 1):
        s = 0.0
        for i in range(1, d // 2 + 1):
            j = d - i; s += T[i] * T[j] if i != j else T[i] * (T[i] + 1) / 2
        T[d] = len(COMBINATORS) * s
    return T[depth]


# ================================================================ analyse d'une recherche réelle

def forms_of(programs: Iterable[BNode]) -> Set[str]: return {algebraic_form(p) for p in programs}


def parents_used(programs: Iterable[BNode]) -> Set[str]:
    """Formes qui ont servi de PARENT : enfant direct de la racine d'au moins un candidat généré."""
    out = set()
    for p in programs:
        if is_composite(p):
            for c in p.children: out.add(algebraic_form(c))
    return out


def first_divergence(truth: BNode, generated: Set[str], used_as_parent: Set[str]):
    """Premier point où la recherche réelle quitte le chemin oracle. Parcourt les intermédiaires vrais par taille
    croissante. Renvoie {'size', 'kind'} avec kind ∈ {not_generated, not_extended, none}."""
    inter = sorted({algebraic_form(s): n_atoms(s) for s in subtrees(truth, min_atoms=1)}.items(), key=lambda kv: kv[1]); full = algebraic_form(truth)
    for form, size in inter:
        if form == full: continue
        if size >= 2 and form not in generated: return {"size": size, "kind": "not_generated"}
    for form, size in inter:
        if form == full or size < 2: continue
        if form not in used_as_parent: return {"size": size, "kind": "not_extended"}
    return {"size": n_atoms(truth), "kind": "none" if full in generated else "not_generated"}


def classify(truth: Optional[BNode], strategy: str, generated_programs: Dict[str, BNode], ranked_programs: Sequence[BNode], selected: Optional[BNode], well_fitted: bool,
             budget_exhausted: bool, class_forms: Iterable[str] = (), max_depth=6, library_forms: Iterable[str] = (), top_k=5, type_rejected: Iterable[str] = ()):
    """Reachability, rappels de la vraie structure à chaque étape, et CAUSE PRINCIPALE d'un échec (taxonomie)."""
    ok, why = representable(truth, max_depth)
    if not ok: return {"reachability": Reachability.NOT_REPRESENTABLE, "cause": "GRAMMAR_UNREPRESENTABLE", "detail": why, **{k: False for k in RECALLS}}
    form = algebraic_form(truth); gen = forms_of(generated_programs.values()); used = parents_used(generated_programs.values()); ranked = [algebraic_form(p) for p in ranked_programs]
    reach_op, missing = operator_reachable(truth, OPERATORS.get(strategy, ("extend_atom",)), library_forms)
    rec = {"recall_generated": form in gen, "recall_pre_pruning": form in gen, "recall_post_pruning": form in ranked, "recall_fitted": form in gen and well_fitted,
           "recall_top_k": form in ranked[:top_k], "recall_selected": selected is not None and algebraic_form(selected) == form}
    div = first_divergence(truth, gen, used); base = {**rec, "divergence_size": div["size"], "divergence_kind": div["kind"], "oracle_length": oracle_path(truth)["length"], "width": strahler(truth)}
    if rec["recall_selected"]: return {"reachability": Reachability.REACHED, "cause": RECOVERED, **base}
    if not reach_op:
        cause = "FRAGMENT_MISSING" if "extend_fragment" in OPERATORS.get(strategy, ()) else "OPERATORS_INSUFFICIENT"
        return {"reachability": Reachability.UNREACHABLE, "cause": cause, "missing_operator": missing, **base}
    if form in set(type_rejected): return {"reachability": Reachability.NOT_REACHED, "cause": "TYPE_MISMATCH", **base}
    if not rec["recall_generated"]:
        if div["kind"] == "not_extended": cause = "BUDGET_EXHAUSTED" if budget_exhausted else "PRUNED_TOO_EARLY"
        elif div["size"] < n_atoms(truth): cause = "TRUE_PARENT_NOT_GENERATED"
        else: cause = "BUDGET_EXHAUSTED" if budget_exhausted else "PRUNED_TOO_EARLY"
        return {"reachability": Reachability.NOT_REACHED, "cause": cause, **base}
    if not well_fitted: return {"reachability": Reachability.REACHED, "cause": "FIT_FAILURE", **base}
    if not rec["recall_post_pruning"]: return {"reachability": Reachability.REACHED, "cause": "CANONICALIZATION_COLLAPSE", **base}
    if form in set(class_forms): return {"reachability": Reachability.REACHED, "cause": "OBSERVATIONAL_NON_IDENTIFIABILITY", **base}
    return {"reachability": Reachability.REACHED, "cause": "RANKING_FAILURE", **base}


RECALLS = ("recall_generated", "recall_pre_pruning", "recall_post_pruning", "recall_fitted", "recall_top_k", "recall_selected")


def basis_quality(truth: BNode):
    """Qualité de la base : la structure contient-elle des atomes REDONDANTS (exprimables par composition d'atomes
    plus simples) ? band = half ∧ half (parallèles), ring = disc ∧ antidisc (concentriques), sector = half ∧ half
    (par l'apex). Profondeur en base minimale {demi-plan, disque} et nombre de paramètres liés."""
    red = {("dot", "in"): 2, ("norm", "in"): 2, ("angle", "in"): 2}; minimal = 0; redundant = 0
    for l in leaves(truth):
        x = l
        while x.kind in ("translate", "not"): x = x.children[0]
        k = red.get((x.feat, x.op), 1); minimal += k; redundant += k > 1
    return {"atoms": n_atoms(truth), "atoms_in_minimal_basis": minimal, "redundant_atoms": redundant, "params": truth.n_params(), "description_inflation": minimal / max(1, n_atoms(truth))}

"""Évaluation de la découverte de primitives — module d'ÉVALUATION (il voit la vérité ; rien n'en revient au système).

Métriques (jamais fusionnées) :
  exact        : même type d'atome ET accord fonctionnel ≥ 0,95 sur la région observée
  functional   : accord ≥ 0,95 (ou complément) avec une primitive découverte, quel que soit son type
  equivalence  : toutes les COMPOSANTES de frontière de l'atome vrai (en base minimale : droites, cercles, demi-droites)
                 sont retrouvées par des composantes de primitives découvertes — un anneau retrouvé comme deux cercles
                 concentriques compte ici, pas en « exact »
  component    : part des composantes de frontière vraies retrouvées
  precision    : part des classes de primitives retenues dont toutes les composantes correspondent à une frontière vraie
  parameter    : erreur absolue moyenne des paramètres, pour les correspondances exactes

Identifiabilité d'un atome vrai (avant de juger) : contraste z du VRAI atome sur les observations —
identifiable (z ≥ 3), partiel (2 ≤ z < 3), non identifiable (z < 2).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch

from core.boundary_program import BNode, predict
from core.deep_primitives import PrimitiveCandidate, atom_name, close_pairs, contrast
from core.deep_structure import _leaf, skel_key

IDENT = ("identifiable", "partial", "not_identifiable")


def _wrap(a): return math.atan2(math.sin(a), math.cos(a))


def components(atom: BNode) -> List[tuple]:
    """Composantes de frontière en base minimale : ('line', angle, offset) | ('circle', cx, cy, r) | ('ray', px, py, direction)."""
    l = _leaf(atom); c = atom.params if atom.kind == "translate" else (0.0, 0.0); p = l.params
    if l.feat == "dot":
        if l.op in ("le", "ge"): return [("line", p[0], p[1])]
        return [("line", p[0], p[1]), ("line", p[0], p[2])]
    if l.feat == "norm":
        if l.op in ("le", "ge"): return [("circle", c[0], c[1], abs(p[0]))]
        return [("circle", c[0], c[1], abs(p[0])), ("circle", c[0], c[1], abs(p[1]))]
    if l.feat == "angle": return [("ray", c[0], c[1], _wrap(p[0] + abs(p[1]))), ("ray", c[0], c[1], _wrap(p[0] - abs(p[1])))]
    return []


def _ray_as_line(r):
    """Droite support d'une demi-droite : (angle de la normale, décalage)."""
    a = r[3] + math.pi / 2; return ("line", a, math.cos(a) * r[1] + math.sin(a) * r[2])


def component_match(a: tuple, b: tuple, tol_angle=0.22, tol_offset=0.3, tol_center=0.35, tol_radius=0.3) -> bool:
    if a[0] == "circle" or b[0] == "circle":
        return a[0] == b[0] == "circle" and math.hypot(a[1] - b[1], a[2] - b[2]) < tol_center and abs(a[3] - b[3]) < tol_radius
    if a[0] == "ray" and b[0] == "ray": return math.hypot(a[1] - b[1], a[2] - b[2]) < 0.5 and abs(_wrap(a[3] - b[3])) < tol_angle
    la = _ray_as_line(a) if a[0] == "ray" else a; lb = _ray_as_line(b) if b[0] == "ray" else b; d = _wrap(la[1] - lb[1])
    if abs(d) < tol_angle: return abs(la[2] - lb[2]) < tol_offset
    if abs(abs(d) - math.pi) < tol_angle: return abs(la[2] + lb[2]) < tol_offset
    return False


def identifiability_of(atom: BNode, X, y) -> str:
    z = contrast(atom, close_pairs(X, y), use_weight=False)["z"]; return IDENT[0] if z >= 3.0 else IDENT[1] if z >= 2.0 else IDENT[2]


def primitive_metrics(true_atoms: Sequence[BNode], found: Sequence[BNode], X, y, probe) -> dict:
    """Métriques de découverte pour une scène. `found` : programmes des primitives retenues (tous membres de classes)."""
    fc = [(p, components(p)) for p in found]; allc = [c for _, cs in fc for c in cs]; rows = []
    for a in true_atoms:
        ta = predict(a, probe); ag = [(float((predict(p, probe) == ta).float().mean()), p) for p in found]
        exact = [(v, p) for v, p in ag if v >= 0.95 and skel_key(p) == skel_key(a)]; func = any(v >= 0.95 or v <= 0.05 for v, _ in ag); tc = components(a); hit = [any(component_match(c, d) for d in allc) for c in tc]
        perr = min((sum(abs(u - w) for u, w in zip(p.param_vector(), a.param_vector())) / max(1, a.n_params()) for _, p in exact), default=None)
        rows.append({"type": atom_name(a), "identifiability": identifiability_of(a, X, y), "exact": bool(exact), "functional": func, "equivalence": all(hit), "component_recall": sum(hit) / len(hit), "param_error": perr})
    tcs = [c for a in true_atoms for c in components(a)]
    prec = [all(any(component_match(c, d) for d in tcs) for c in cs) for _, cs in fc if cs]
    def rate(key, sub=None):
        xs = [r[key] for r in rows if sub is None or r["identifiability"] == sub]; return sum(float(x) for x in xs) / len(xs) if xs else None
    out = {"n_true": len(rows), "n_found": len(found), "precision": sum(prec) / len(prec) if prec else None, "atoms": rows,
           "exact": rate("exact"), "functional": rate("functional"), "equivalence": rate("equivalence"), "component_recall": rate("component_recall"),
           "param_error": (lambda v: sum(v) / len(v) if v else None)([r["param_error"] for r in rows if r["param_error"] is not None])}
    for s in IDENT: out[f"equivalence_{s}"] = rate("equivalence", s); out[f"n_{s}"] = sum(r["identifiability"] == s for r in rows)
    return out

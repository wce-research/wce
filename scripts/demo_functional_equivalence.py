"""Démo : deux AST différents → mêmes invariants / même comportement → même classe fonctionnelle (empirique)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.active_structure import functional_clusters, program_signature  # noqa: E402
from core.boundary_program import BNode as P, render, structure_hash  # noqa: E402

if __name__ == "__main__":
    T = lambda c, p: P("translate", params=p, children=(c,))
    a = P("or", children=(T(P("pred", "norm", "le", (0.6,)), (-1.4, 0.0)), T(P("pred", "norm", "le", (0.6,)), (1.0, 0.3))))
    b = P("xor", children=a.children)
    c = P("not", children=(P("and", children=(T(P("pred", "norm", "ge", (0.6,)), (-1.4, 0.0)), T(P("pred", "norm", "ge", (0.6,)), (1.0, 0.3)))),))
    d = T(P("pred", "norm", "le", (1.0,)), (0.0, 0.0))
    for p in (a, b, c, d):
        s = program_signature(p); print(f"{render(p)[:70]:72s} composantes {s.components} aire {s.area:.3f} symétrie {s.reflection_symmetry:.2f}")
    cls, idx = functional_clusters([a, b, c, d]); print("Classes :", idx, "; AST distincts dans la classe 0 :", len(cls[0].representations()))
    print("Même structure AST a/b ?", structure_hash(a) == structure_hash(b))
